"""
03_build_crsp_sample.py
=======================
Pull CRSP daily returns and volume from WRDS, then compute market-model
cumulative abnormal returns (CARs) and abnormal trading volume for each
auditor change event.

INPUT:  Data/Processed/auditor_changes_raw.parquet   (from 01_build_edgar_event_file.py)
OUTPUT: Data/Processed/crsp_event_window.parquet
        Columns: permno, cik, gvkey, date_filed, event_date,
                 car_m1p1, abvol_m1p1, car_m2p2,
                 alpha_hat, beta_hat, r2, n_est_days,
                 ret_m1, ret_0, ret_p1, vol_m1, vol_0, vol_p1

METHODOLOGY:
    Market model estimated over trading days [-252, -46] relative to event date
    (one year of data, ending ~2 months before event to avoid contamination).
    Minimum 100 non-missing trading days required in estimation window.

    CAR(-1,+1) = Σ_{t=-1}^{+1} [ret_{i,t} - (α̂ + β̂ · mktret_t)]
    Abvol = mean over event window of [log(vol+1) - mean(log(vol+1) in estimation window)]

    Market return: CRSP value-weighted return including distributions (vwretd).
    Delisting returns included following Beaver, McNichols & Price (2007).

WRDS CONNECTION:
    Requires a WRDS account and the wrds Python library (pip install wrds).
    On first run, wrds will prompt for username/password and offer to save
    credentials to ~/.pgpass for future sessions.

LINKING CIK → PERMNO:
    CIK (EDGAR) → gvkey (Compustat comp.company) → permno (crsp.ccmxpf_lnkhist)
    Events with no valid link are logged and dropped.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
EST_WINDOW_START = -252   # trading days before event (estimation window start)
EST_WINDOW_END   = -46    # trading days before event (estimation window end)
EVENT_WINDOW     = [-1, 0, 1]   # event window in trading days
MIN_EST_DAYS     = 100    # minimum non-missing days in estimation window
WRDS_USERNAME    = "nhwang"

EVENT_FILE = Path(__file__).resolve().parent.parent / "Data/Processed/auditor_changes_raw.parquet"
OUT_FILE   = Path(__file__).resolve().parent.parent / "Data/Processed/crsp_event_window.parquet"
LOG_FILE   = Path(__file__).resolve().parent.parent / "Data/Processed/03_crsp_link_log.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Step 1: Load event file ───────────────────────────────────────────────────

def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Event file not found: {path}\n"
            "Run 01_build_edgar_event_file.py first."
        )
    events = pd.read_parquet(path)
    events["date_filed"] = pd.to_datetime(events["date_filed"])

    # Normalize CIK to 10-digit zero-padded string to match comp.company.cik format.
    # EDGAR index stores CIKs as plain integers ("919012"); Compustat stores them as
    # "0000919012". Without this, the CIK → gvkey SQL lookup returns zero rows.
    events["cik"] = events["cik"].astype(str).str.strip().str.zfill(10)

    # Drop 8-K/A amendments — keep only original filings for the main sample.
    # Amendments often refile after market has already reacted.
    n_before = len(events)
    events = events[~events.get("is_amendment", pd.Series(False, index=events.index))]
    log.info("Events loaded: %d (dropped %d amendments)", len(events), n_before - len(events))
    return events


# ── Step 2: Connect to WRDS ───────────────────────────────────────────────────

def connect_wrds():
    import wrds
    kwargs = {"wrds_username": WRDS_USERNAME} if WRDS_USERNAME else {}
    conn = wrds.Connection(**kwargs)
    log.info("Connected to WRDS.")
    return conn


# ── Step 3: Build CIK → PERMNO link ──────────────────────────────────────────

def build_cik_permno_link(conn, ciks: list[str]) -> pd.DataFrame:
    """
    CIK → gvkey via comp.company, then gvkey → permno via crsp.ccmxpf_lnkhist.
    Returns a DataFrame with columns: cik, gvkey, permno, linkdt, linkenddt.
    Uses only link types LC and LU (most reliable primary links).
    """
    cik_list = ", ".join(f"'{c}'" for c in ciks)

    # CIK → gvkey
    log.info("Fetching CIK → gvkey link from Compustat ...")
    cik_gvkey = conn.raw_sql(f"""
        SELECT cik, gvkey
        FROM comp.company
        WHERE cik IN ({cik_list})
    """)
    log.info("  CIK → gvkey matches: %d / %d CIKs", cik_gvkey["cik"].nunique(), len(set(ciks)))

    if cik_gvkey.empty:
        raise ValueError("No CIK → gvkey matches found. Check that CIKs are in Compustat.")

    gvkeys = cik_gvkey["gvkey"].unique().tolist()
    gvkey_list = ", ".join(f"'{g}'" for g in gvkeys)

    # gvkey → permno (CRSP-Compustat link, primary links only)
    log.info("Fetching gvkey → permno link from CRSP ...")
    gvkey_permno = conn.raw_sql(f"""
        SELECT gvkey, lpermno AS permno, linkdt, linkenddt
        FROM crsp.ccmxpf_lnkhist
        WHERE gvkey IN ({gvkey_list})
          AND linktype IN ('LC', 'LU')
          AND linkprim IN ('P', 'C')
    """)
    # NULL linkdt means "active from beginning of time" — fill with early sentinel.
    # NULL linkenddt means "still active" — fill with far-future sentinel.
    # Without filling linkdt, NaT comparisons return False and valid links are dropped.
    gvkey_permno["linkdt"]    = pd.to_datetime(
        gvkey_permno["linkdt"].fillna("1900-01-01")
    )
    gvkey_permno["linkenddt"] = pd.to_datetime(
        gvkey_permno["linkenddt"].fillna("2099-12-31")
    )
    log.info("  gvkey → permno matches: %d gvkeys", gvkey_permno["gvkey"].nunique())

    # Merge
    link = cik_gvkey.merge(gvkey_permno, on="gvkey", how="inner")
    log.info("Combined link table: %d rows", len(link))
    return link


def apply_link_to_events(events: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """
    For each event, find the PERMNO that was active on the filing date.
    The CRSP-Compustat link has date ranges (linkdt, linkenddt) — use only
    the link valid on the event date.
    """
    merged = events.merge(link, on="cik", how="left")

    # Keep rows where the event date falls within the link date range
    valid = (
        (merged["date_filed"] >= merged["linkdt"]) &
        (merged["date_filed"] <= merged["linkenddt"])
    )
    matched = merged[valid].copy()

    # If multiple valid links exist for an event (rare), keep the one with
    # the earliest linkdt (most established link)
    matched = (
        matched.sort_values("linkdt")
        .drop_duplicates(subset=["cik", "acc_nodash"], keep="first")
    )

    n_total   = len(events)
    n_matched = len(matched)
    n_miss    = n_total - n_matched
    log.info("Events matched to PERMNO: %d / %d (unmatched: %d)", n_matched, n_total, n_miss)

    # Log unmatched for review
    unmatched = events[~events["acc_nodash"].isin(matched["acc_nodash"])][
        ["cik", "acc_nodash", "date_filed", "company_name"]
    ].assign(reason="no_permno_link")
    unmatched.to_csv(LOG_FILE, index=False)
    log.info("Unmatched events logged to: %s", LOG_FILE)

    return matched


# ── Step 4: Pull CRSP daily data ──────────────────────────────────────────────

def pull_crsp_daily(conn, permnos: list[int],
                    start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pull daily returns and volume for all relevant PERMNOs in one query.
    Includes delisting returns to avoid survivorship bias.
    """
    permno_list = ", ".join(str(p) for p in permnos)

    log.info("Pulling CRSP daily stock file (%d PERMNOs, %s to %s)...",
             len(permnos), start_date, end_date)
    dsf = conn.raw_sql(f"""
        SELECT permno, date, ret, retx, vol, shrout, abs(prc) AS prc
        FROM crsp.dsf
        WHERE permno IN ({permno_list})
          AND date BETWEEN '{start_date}' AND '{end_date}'
    """)
    dsf["date"] = pd.to_datetime(dsf["date"])

    # Merge in delisting returns (delist return replaces last regular return)
    log.info("Pulling CRSP delisting returns ...")
    delist = conn.raw_sql(f"""
        SELECT permno, dlstdt AS date, dlret
        FROM crsp.dsedelist
        WHERE permno IN ({permno_list})
          AND dlret IS NOT NULL
    """)
    delist["date"] = pd.to_datetime(delist["date"])

    # Apply delisting return: if the last observation for a permno has a
    # delisting return, replace ret with (1 + ret) * (1 + dlret) - 1
    # (or just dlret if ret is missing on that date)
    dsf = dsf.merge(delist, on=["permno", "date"], how="left")
    has_delist = dsf["dlret"].notna()
    dsf.loc[has_delist & dsf["ret"].notna(), "ret"] = (
        (1 + dsf.loc[has_delist & dsf["ret"].notna(), "ret"]) *
        (1 + dsf.loc[has_delist & dsf["ret"].notna(), "dlret"]) - 1
    )
    dsf.loc[has_delist & dsf["ret"].isna(), "ret"] = (
        dsf.loc[has_delist & dsf["ret"].isna(), "dlret"]
    )
    dsf = dsf.drop(columns=["dlret"])

    log.info("CRSP daily rows pulled: %d", len(dsf))
    return dsf


def pull_market_returns(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """Pull CRSP value-weighted market returns (including distributions)."""
    log.info("Pulling CRSP market returns ...")
    mkt = conn.raw_sql(f"""
        SELECT date, vwretd AS mkt_ret
        FROM crsp.dsi
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
    """)
    mkt["date"] = pd.to_datetime(mkt["date"])
    return mkt


# ── Step 5: Compute CARs ──────────────────────────────────────────────────────

def get_trading_days(dsf: pd.DataFrame) -> pd.Series:
    """Return the sorted list of unique trading days in the CRSP daily file."""
    return dsf["date"].drop_duplicates().sort_values().reset_index(drop=True)


def trading_day_offset(event_date: pd.Timestamp,
                       trading_days: pd.Series,
                       offset: int) -> pd.Timestamp | None:
    """
    Return the trading day that is `offset` days from event_date.
    Positive offset = after event; negative = before event.
    Returns None if out of range.
    """
    idx_arr = trading_days.searchsorted(event_date)
    # If event_date is not itself a trading day, use the next trading day
    if idx_arr < len(trading_days) and trading_days.iloc[idx_arr] != event_date:
        idx_arr = idx_arr  # event date falls on weekend/holiday — use next day
    target_idx = idx_arr + offset
    if target_idx < 0 or target_idx >= len(trading_days):
        return None
    return trading_days.iloc[target_idx]


def compute_car_one_event(permno: int, event_date: pd.Timestamp,
                          firm_data: pd.DataFrame, mkt_data: pd.DataFrame,
                          trading_days: pd.Series) -> dict:
    """
    Compute market-model CAR and abnormal volume for one event.
    Returns a dict of results; on failure returns dict with car_m1p1=NaN.
    """
    base = {
        "permno": permno, "event_date": event_date,
        "car_m1p1": np.nan, "car_m2p2": np.nan,
        "abvol_m1p1": np.nan, "alpha_hat": np.nan,
        "beta_hat": np.nan, "r2": np.nan, "n_est_days": 0,
        "ret_m1": np.nan, "ret_0": np.nan, "ret_p1": np.nan,
        "vol_m1": np.nan, "vol_0": np.nan, "vol_p1": np.nan,
    }

    # Resolve estimation window dates
    est_start = trading_day_offset(event_date, trading_days, EST_WINDOW_START)
    est_end   = trading_day_offset(event_date, trading_days, EST_WINDOW_END)
    if est_start is None or est_end is None:
        return {**base, "status": "insufficient_history"}

    # Resolve event window dates
    ev_start = trading_day_offset(event_date, trading_days, EVENT_WINDOW[0])
    ev_end   = trading_day_offset(event_date, trading_days, EVENT_WINDOW[-1])
    if ev_start is None or ev_end is None:
        return {**base, "status": "event_window_missing"}

    # Subset to estimation window
    est_mask = (firm_data["date"] >= est_start) & (firm_data["date"] <= est_end)
    est = firm_data[est_mask].merge(
        mkt_data[["date", "mkt_ret"]], on="date", how="inner"
    ).dropna(subset=["ret", "mkt_ret"])

    if len(est) < MIN_EST_DAYS:
        return {**base, "status": f"insufficient_est_days_{len(est)}"}

    # OLS: ret = alpha + beta * mkt_ret
    X = np.column_stack([np.ones(len(est)), est["mkt_ret"].values])
    y = est["ret"].values
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return {**base, "status": "ols_failed"}

    alpha_hat, beta_hat = coef
    y_hat = X @ coef
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Estimation window log-volume mean (for abnormal volume)
    est_logvol = np.log1p(est["vol"].replace(0, np.nan)).dropna()
    mean_logvol = est_logvol.mean() if len(est_logvol) > 0 else np.nan

    # Event window returns and abnormal returns
    ev_mask = (firm_data["date"] >= ev_start) & (firm_data["date"] <= ev_end)
    ev_firm = firm_data[ev_mask].merge(
        mkt_data[["date", "mkt_ret"]], on="date", how="inner"
    )
    if len(ev_firm) == 0:
        return {**base, "status": "event_window_no_data",
                "alpha_hat": alpha_hat, "beta_hat": beta_hat,
                "r2": r2, "n_est_days": len(est)}

    ev_firm = ev_firm.copy()
    ev_firm["expected_ret"] = alpha_hat + beta_hat * ev_firm["mkt_ret"]
    ev_firm["ar"]           = ev_firm["ret"] - ev_firm["expected_ret"]
    ev_firm["logvol"]       = np.log1p(ev_firm["vol"].replace(0, np.nan))
    ev_firm["abvol"]        = ev_firm["logvol"] - mean_logvol

    car_m1p1 = ev_firm["ar"].sum()
    abvol    = ev_firm["abvol"].mean()

    # Wider event window [-2,+2] for robustness
    ev_start2 = trading_day_offset(event_date, trading_days, -2)
    ev_end2   = trading_day_offset(event_date, trading_days, +2)
    if ev_start2 and ev_end2:
        ev2_mask = (firm_data["date"] >= ev_start2) & (firm_data["date"] <= ev_end2)
        ev2 = firm_data[ev2_mask].merge(mkt_data[["date","mkt_ret"]], on="date", how="inner")
        ev2 = ev2.copy()
        ev2["ar"] = ev2["ret"] - (alpha_hat + beta_hat * ev2["mkt_ret"])
        car_m2p2 = ev2["ar"].sum()
    else:
        car_m2p2 = np.nan

    # Individual day returns for diagnostics
    def get_day_ret(offset):
        d = trading_day_offset(event_date, trading_days, offset)
        if d is None:
            return np.nan, np.nan
        row = ev_firm[ev_firm["date"] == d]
        if len(row) == 0:
            return np.nan, np.nan
        return float(row["ret"].iloc[0]), float(row["vol"].iloc[0])

    ret_m1, vol_m1 = get_day_ret(-1)
    ret_0,  vol_0  = get_day_ret(0)
    ret_p1, vol_p1 = get_day_ret(1)

    return {
        "permno":     permno,
        "event_date": event_date,
        "car_m1p1":   car_m1p1,
        "car_m2p2":   car_m2p2,
        "abvol_m1p1": abvol,
        "alpha_hat":  alpha_hat,
        "beta_hat":   beta_hat,
        "r2":         r2,
        "n_est_days": len(est),
        "ret_m1":     ret_m1,
        "ret_0":      ret_0,
        "ret_p1":     ret_p1,
        "vol_m1":     vol_m1,
        "vol_0":      vol_0,
        "vol_p1":     vol_p1,
        "status":     "ok",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== 03_build_crsp_sample.py  start ===")

    # Step 1: Load event file
    events = load_events(EVENT_FILE)

    # Step 2: Connect to WRDS
    conn = connect_wrds()

    # Step 3: Link CIKs to PERMNOs
    ciks = events["cik"].dropna().unique().tolist()
    if not ciks:
        log.error("No CIKs in event file. Exiting.")
        conn.close()
        return
    link  = build_cik_permno_link(conn, ciks)
    events_linked = apply_link_to_events(events, link)

    if events_linked.empty:
        log.error("No events matched to PERMNO. Exiting.")
        conn.close()
        return

    # Step 4: Pull CRSP daily data
    # Date range: earliest event minus estimation window, to latest event plus buffer
    # EST_WINDOW_START = -252 trading days ≈ 365 calendar days. Use 400 calendar
    # days to be safe (252 × 365/252 ≈ 365, plus ~35 days buffer).
    sample_start = (events_linked["date_filed"].min() -
                    pd.DateOffset(days=400))
    sample_end   = events_linked["date_filed"].max() + pd.DateOffset(days=10)

    permnos = events_linked["permno"].astype(int).unique().tolist()

    dsf = pull_crsp_daily(
        conn, permnos,
        sample_start.strftime("%Y-%m-%d"),
        sample_end.strftime("%Y-%m-%d"),
    )
    mkt = pull_market_returns(
        conn,
        sample_start.strftime("%Y-%m-%d"),
        sample_end.strftime("%Y-%m-%d"),
    )
    conn.close()
    log.info("WRDS connection closed.")

    # Step 5: Compute CARs
    trading_days = get_trading_days(dsf)
    log.info("Computing CARs for %d events ...", len(events_linked))

    results = []
    for _, ev_row in events_linked.iterrows():
        permno     = int(ev_row["permno"])
        event_date = ev_row["date_filed"]

        # Subset CRSP to this firm only
        firm_data = dsf[dsf["permno"] == permno].copy()

        result = compute_car_one_event(
            permno, event_date, firm_data, mkt, trading_days
        )
        # Carry through event identifiers
        result["cik"]          = ev_row["cik"]
        result["acc_nodash"]   = ev_row["acc_nodash"]
        result["gvkey"]        = ev_row.get("gvkey", np.nan)
        result["company_name"] = ev_row["company_name"]
        result["reason"]       = ev_row.get("reason", np.nan)
        result["disagreements"]= ev_row.get("disagreements", np.nan)
        result["quality_direction"] = ev_row.get("quality_direction", np.nan)
        results.append(result)

    out = pd.DataFrame(results)
    log.info("Status breakdown:\n%s", out["status"].value_counts().to_string())

    # Keep only successful events for output
    clean = out[out["status"] == "ok"].copy()
    log.info("Clean events (status=ok): %d", len(clean))

    # Sanity checks
    log.info("CAR(-1,+1) summary:\n%s", clean["car_m1p1"].describe().to_string())
    log.info("Abnormal volume summary:\n%s", clean["abvol_m1p1"].describe().to_string())

    # Write output
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(OUT_FILE, index=False)
    log.info("Output written: %s", OUT_FILE)
    log.info("=== done ===")


if __name__ == "__main__":
    main()
