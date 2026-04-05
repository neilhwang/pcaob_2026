"""
10_build_officer_change_placebo.py
==================================
Constructs an officer-change (Item 5.02) placebo test for the main
auditor-change (Item 4.01) analysis.

PURPOSE
-------
Validates that the polarization effect is specific to auditor-change
disclosures rather than to any mandatory 8-K filing requiring interpretation.
Item 5.02 officer-change filings are similarly mandatory, standardized, and
information-relevant, but they are not institution-dependent signals about
auditor credibility or regulatory oversight. A null coefficient in the
placebo regression supports the claim that polarization amplifies
disagreement specifically through the institutional-trust channel.

DESIGN
------
  Six columns in the output table (AbVol cols 1-3, |CAR| cols 4-6):
    (1)/(4) Main sample baseline -- replicates Table 2 for comparison
    (2)/(5) Officer-change placebo -- same specification, Item 5.02 events
    (3)/(6) Pooled -- stacks both samples with an Auditor Change indicator
            and a Polarization × Auditor Change interaction

INPUTS
------
  Data/Processed/analysis_sample.parquet          (from 05_merge_and_estimate.py)
  Data/Processed/placebo_events_raw.parquet       (from 09_build_placebo_event_file.py)
  Data/Processed/pol_presidential.parquet         (from 02b_build_presidential_polarization.py)
  Data/Processed/compustat_controls.parquet       (from 04_build_compustat_controls.py)
  WRDS: comp.company + comp.funda + crsp.stocknames  -- CIK->PERMNO link
  WRDS: crsp.dsf                                     -- daily stock returns + volume
  WRDS: crsp.dsi                                     -- market returns (vwretd)

OUTPUTS
-------
  Data/Processed/officer_change_placebo.parquet   -- placebo event-level dataset
  Output/Tables/tab_placebo_502.tex               -- placebo regression table

WRDS USERNAME: nhwang
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import wrds

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
PROC     = ROOT / "Data" / "Processed"
OUT_TABS = ROOT / "Output" / "Tables"
OUT_TABS.mkdir(parents=True, exist_ok=True)

ANALYSIS_SAMPLE = PROC / "analysis_sample.parquet"
PLACEBO_RAW     = PROC / "placebo_events_raw.parquet"
POL_FILE        = PROC / "pol_presidential.parquet"
COMP_FILE       = PROC / "compustat_controls.parquet"
OUT_EVENTS      = PROC / "officer_change_placebo.parquet"
OUT_TABLE       = OUT_TABS / "tab_placebo_502.tex"

WRDS_USERNAME = "nhwang"

# Market model parameters -- identical to 03_build_crsp_sample.py
EST_WINDOW_START = -252
EST_WINDOW_END   = -46
MIN_EST_DAYS     = 100

CONTROLS = ["size", "leverage", "roa", "btm", "loss", "sales_growth"]

# Exclude placebo events within this many calendar days of an auditor-change event
EXCLUSION_DAYS = 30

# Calendar-day buffer before earliest event to ensure full estimation window
CRSP_LEAD_DAYS = 380


# ── Helpers ──────────────────────────────────────────────────────────────────

def batch_list(lst, size=200):
    lst = list(lst)
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def sql_in_str(values):
    escaped = ["'" + str(v).replace("'", "''") + "'" for v in values]
    return "(" + ", ".join(escaped) + ")"


def sql_in_int(values):
    return "(" + ", ".join(str(v) for v in values) + ")"


def fmt(x, digits=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x:.{digits}f}"


def stars(pval):
    if pval < 0.01: return "***"
    if pval < 0.05: return "**"
    if pval < 0.10: return "*"
    return ""


# ── Step 1: Load analysis sample metadata ────────────────────────────────────

def load_analysis_metadata():
    cols = ["gvkey", "permno", "event_date", "state",
            "competitive_std", "absCar", "abvol", "margin",
            "year_str", "sic2_str", "gvkey_str"] + CONTROLS
    df = pd.read_parquet(ANALYSIS_SAMPLE, columns=cols)
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["permno"] = df["permno"].astype(int)
    print(f"  Analysis sample: {len(df)} events, {df['gvkey'].nunique()} firms")
    return df


# ── Step 2: Load and prepare Item 5.02 events ───────────────────────────────

def load_placebo_events():
    if not PLACEBO_RAW.exists():
        raise FileNotFoundError(
            f"Placebo event file not found: {PLACEBO_RAW}\n"
            "Run 09_build_placebo_event_file.py first."
        )
    df = pd.read_parquet(PLACEBO_RAW)
    df["date_filed"] = pd.to_datetime(df["date_filed"])

    # Drop amendments
    if "is_amendment" in df.columns:
        n0 = len(df)
        df = df[~df["is_amendment"]].copy()
        print(f"  Dropped {n0 - len(df)} amendments")

    # Normalize CIK to 10-digit zero-padded string (matches comp.company.cik)
    df["cik"] = df["cik"].astype(str).str.strip().str.zfill(10)

    print(f"  Item 5.02 events loaded: {len(df):,}")
    print(f"  Date range: {df['date_filed'].min().date()} -- {df['date_filed'].max().date()}")
    return df


# ── Step 3: CIK -> PERMNO link (same approach as 03_build_crsp_sample.py) ────

def build_cik_permno_link(db, ciks):
    """CIK -> gvkey (comp.company) -> cusip (comp.funda) -> permno (crsp.stocknames)."""
    print(f"  Linking {len(ciks):,} unique CIKs to PERMNOs...")

    # CIK -> gvkey
    records = []
    for chunk in batch_list(ciks, 500):
        sql = f"SELECT cik, gvkey FROM comp.company WHERE cik IN {sql_in_str(chunk)}"
        records.append(db.raw_sql(sql))
    cik_gvkey = pd.concat(records, ignore_index=True)
    print(f"    CIK -> gvkey: {cik_gvkey['cik'].nunique():,} / {len(ciks):,} CIKs matched")

    if cik_gvkey.empty:
        raise ValueError("No CIK -> gvkey matches found.")

    # gvkey -> most-recent CUSIP
    gvkeys = cik_gvkey["gvkey"].unique().tolist()
    records = []
    for chunk in batch_list(gvkeys, 500):
        sql = f"""
            SELECT DISTINCT ON (gvkey) gvkey, cusip
            FROM comp.funda
            WHERE gvkey IN {sql_in_str(chunk)}
              AND cusip IS NOT NULL
            ORDER BY gvkey, datadate DESC
        """
        records.append(db.raw_sql(sql))
    gvkey_cusip = pd.concat(records, ignore_index=True)
    cik_gvkey = cik_gvkey.merge(gvkey_cusip, on="gvkey", how="inner")
    print(f"    gvkey -> cusip: {gvkey_cusip['gvkey'].nunique():,} matched")

    # CUSIP -> PERMNO via crsp.stocknames
    cik_gvkey["ncusip8"] = cik_gvkey["cusip"].str[:8]
    ncusips = cik_gvkey["ncusip8"].dropna().unique().tolist()
    records = []
    for chunk in batch_list(ncusips, 500):
        sql = f"""
            SELECT permno, ncusip, namedt, nameenddt
            FROM crsp.stocknames
            WHERE ncusip IN {sql_in_str(chunk)}
        """
        records.append(db.raw_sql(sql))
    stocknames = pd.concat(records, ignore_index=True)
    stocknames["namedt"]    = pd.to_datetime(stocknames["namedt"].fillna("1900-01-01"))
    stocknames["nameenddt"] = pd.to_datetime(stocknames["nameenddt"].fillna("2099-12-31"))

    link = (
        cik_gvkey
        .merge(stocknames, left_on="ncusip8", right_on="ncusip", how="inner")
        .rename(columns={"namedt": "linkdt", "nameenddt": "linkenddt"})
        [["cik", "gvkey", "permno", "linkdt", "linkenddt"]]
    )
    print(f"    Final link table: {len(link):,} rows, "
          f"{link['cik'].nunique():,} unique CIKs with PERMNOs")
    return link


def apply_link(events, link):
    """Match each event to the PERMNO active on date_filed."""
    merged = events.merge(link, on="cik", how="left")
    valid = (merged["date_filed"] >= merged["linkdt"]) & \
            (merged["date_filed"] <= merged["linkenddt"])
    matched = (
        merged[valid].copy()
        .sort_values("linkdt")
        .drop_duplicates(subset=["cik", "acc_nodash"], keep="first")
    )
    print(f"  Events matched to PERMNO: {len(matched):,} / {len(events):,}")
    return matched


# ── Step 4: Exclude events near auditor changes ─────────────────────────────

def exclude_near_auditor_changes(placebo, analysis_df):
    """Remove Item 5.02 events within EXCLUSION_DAYS of an auditor-change event
    for the same firm (matched by gvkey)."""
    ac = analysis_df[["gvkey", "event_date"]].rename(columns={"event_date": "ac_date"})
    merged = placebo.merge(ac, on="gvkey", how="left")
    merged["gap"] = (merged["date_filed"] - merged["ac_date"]).abs().dt.days
    min_gap = (
        merged.groupby(["gvkey", "date_filed"])["gap"]
        .min().reset_index(name="min_gap")
    )
    n0 = len(placebo)
    placebo = (
        placebo.merge(min_gap, on=["gvkey", "date_filed"], how="left")
        .query("min_gap.isna() or min_gap > @EXCLUSION_DAYS")
        .drop(columns="min_gap")
        .reset_index(drop=True)
    )
    print(f"  Excluded {n0 - len(placebo):,} events within {EXCLUSION_DAYS} days "
          f"of an auditor change")
    return placebo


# ── Step 5: Pull CRSP data ───────────────────────────────────────────────────

def pull_crsp_daily(db, permnos, start_date, end_date):
    print(f"  Date range: {start_date} -> {end_date}")
    print(f"  PERMNOs: {len(permnos):,}")
    records = []
    for i, chunk in enumerate(batch_list(permnos, 200)):
        sql = f"""
            SELECT permno, date, ret, vol
            FROM crsp.dsf
            WHERE permno IN {sql_in_int(chunk)}
              AND date BETWEEN '{start_date}' AND '{end_date}'
        """
        records.append(db.raw_sql(sql))
        if (i + 1) % 5 == 0:
            print(f"    ... {i+1} batches fetched")
    daily = pd.concat(records, ignore_index=True)
    daily["date"]   = pd.to_datetime(daily["date"])
    daily["permno"] = daily["permno"].astype(int)
    print(f"  CRSP daily rows: {len(daily):,}")
    return daily


def pull_market_returns(db, start_date, end_date):
    sql = f"""
        SELECT date, vwretd AS mkt_ret
        FROM crsp.dsi
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
    """
    mkt = db.raw_sql(sql)
    mkt["date"] = pd.to_datetime(mkt["date"])
    print(f"  Market return rows: {len(mkt):,}")
    return mkt


# ── Step 6: Compute CARs and AbVol ──────────────────────────────────────────

def trading_day_offset(event_date, trading_days, offset):
    idx    = trading_days.searchsorted(event_date)
    target = idx + offset
    if target < 0 or target >= len(trading_days):
        return None
    return trading_days.iloc[target]


def compute_cars(placebo, daily, mkt):
    """Compute CAR(-1,+1) and AbVol(-1,+1) for every placebo event."""
    trading_days = (
        pd.concat([daily["date"], mkt["date"]])
        .drop_duplicates().sort_values().reset_index(drop=True)
    )
    mkt_dict = mkt.set_index("date")["mkt_ret"].to_dict()

    daily = daily.copy()
    daily["logvol"] = np.log(daily["vol"].clip(lower=0) + 1)
    daily_idx = daily.set_index("permno")

    results  = []
    n_total  = len(placebo)
    n_report = max(1, n_total // 10)

    for i, (_, row) in enumerate(placebo.iterrows()):
        permno     = row["permno"]
        event_date = row["date_filed"]

        if i % n_report == 0:
            print(f"    {i:,} / {n_total:,} ({100*i/n_total:.0f}%)")

        base = dict(
            cik=row["cik"], gvkey=row["gvkey"], permno=permno,
            event_date=event_date,
            car_m1p1=np.nan, abvol_m1p1=np.nan, n_est_days=0,
        )

        est_start = trading_day_offset(event_date, trading_days, EST_WINDOW_START)
        est_end   = trading_day_offset(event_date, trading_days, EST_WINDOW_END)
        ev_start  = trading_day_offset(event_date, trading_days, -1)
        ev_end    = trading_day_offset(event_date, trading_days, +1)

        if any(d is None for d in [est_start, est_end, ev_start, ev_end]):
            results.append(base); continue
        if permno not in daily_idx.index:
            results.append(base); continue

        firm = daily_idx.loc[[permno]]

        # Estimation window
        est = firm[(firm["date"] >= est_start) & (firm["date"] <= est_end)].copy()
        est["mkt_ret"] = est["date"].map(mkt_dict)
        est = est.dropna(subset=["ret", "mkt_ret"])

        if len(est) < MIN_EST_DAYS:
            results.append(base); continue

        # Market model OLS
        X = np.column_stack([np.ones(len(est)), est["mkt_ret"].values])
        y = est["ret"].values
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            results.append(base); continue
        alpha_hat, beta_hat = coef

        # Estimation-window mean log volume
        est_logvol = est["logvol"].dropna()
        mean_logvol = est_logvol.mean() if len(est_logvol) > 0 else np.nan

        # Event window
        ev = firm[(firm["date"] >= ev_start) & (firm["date"] <= ev_end)].copy()
        ev["mkt_ret"] = ev["date"].map(mkt_dict)
        ev_ret = ev.dropna(subset=["ret", "mkt_ret"])
        if len(ev_ret) == 0:
            results.append({**base, "n_est_days": len(est)}); continue

        car = (ev_ret["ret"] - (alpha_hat + beta_hat * ev_ret["mkt_ret"])).sum()

        ev_logvol = ev["logvol"].dropna()
        if len(ev_logvol) > 0 and not np.isnan(mean_logvol):
            abvol = (ev_logvol - mean_logvol).mean()
        else:
            abvol = np.nan

        results.append({**base, "car_m1p1": car, "abvol_m1p1": abvol,
                        "n_est_days": len(est)})

    out = pd.DataFrame(results)
    n_car = out["car_m1p1"].notna().sum()
    n_vol = out["abvol_m1p1"].notna().sum()
    print(f"\n  Valid CARs:  {n_car:,} / {n_total:,} ({n_car/n_total:.1%})")
    print(f"  Valid AbVol: {n_vol:,} / {n_total:,} ({n_vol/n_total:.1%})")
    return out


# ── Step 7: Merge polarization and controls ──────────────────────────────────

def merge_and_filter(placebo, car_df, pres_pol, comp):
    df = placebo.merge(
        car_df[["cik", "event_date", "car_m1p1", "abvol_m1p1", "n_est_days"]],
        on=["cik", "event_date"], how="inner"
    )
    df = df[df["car_m1p1"].notna()].copy()
    df["event_year"] = df["event_date"].dt.year

    # State from gvkey -> Compustat (headquarters state)
    state_col = "state_abbr" if "state_abbr" in comp.columns else "state"
    comp_state = comp[["gvkey", "fyear", state_col]].dropna(subset=[state_col])
    comp_state = comp_state.rename(columns={"fyear": "comp_year", state_col: "state"})

    # Presidential-election margin
    pres_merge = pres_pol[["year", "state_abbr", "margin"]].rename(
        columns={"year": "event_year", "state_abbr": "state"}
    )

    # Compustat controls from fiscal year prior to event year
    df["comp_year"] = df["event_year"] - 1

    # Get state from Compustat
    if "state" not in df.columns:
        state_map = (
            comp_state.sort_values("comp_year")
            .drop_duplicates(subset="gvkey", keep="last")
            [["gvkey", "state"]]
        )
        df = df.merge(state_map, on="gvkey", how="left")

    df = df.merge(pres_merge, on=["event_year", "state"], how="left")

    comp_merge = comp[["gvkey", "fyear", "sic2"] + CONTROLS].rename(
        columns={"fyear": "comp_year"}
    )
    df = df.merge(comp_merge, on=["gvkey", "comp_year"], how="left")

    # Derived columns
    df["year_str"]  = df["event_year"].astype(str)
    df["sic2_str"]  = df["sic2"].fillna(-1).astype(int).astype(str)
    df["gvkey_str"] = df["gvkey"].astype(str)
    df["absCar"]    = df["car_m1p1"].abs()
    df["abvol"]     = df["abvol_m1p1"]

    # Drop missing key variables
    n0 = len(df)
    df = df.dropna(subset=["absCar", "abvol", "margin", "sic2"] + CONTROLS)
    print(f"  After dropping missing: {len(df):,} (dropped {n0 - len(df):,})")

    # SIC exclusions -- same as main analysis
    n1 = len(df)
    df = df[~df["sic2"].isin(range(60, 70)) & ~df["sic2"].isin(range(49, 50))]
    print(f"  After SIC exclusions: {len(df):,} (dropped {n1 - len(df):,})")

    return df.reset_index(drop=True)


# ── Step 8: Regressions and table ────────────────────────────────────────────

def run_ols(formula, df, cluster_var="gvkey_str"):
    import patsy
    _, X = patsy.dmatrices(formula, data=df, return_type="dataframe",
                            NA_action="drop")
    keep = X.index
    groups = df.loc[keep, cluster_var].values
    return smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )


def write_table(models_list, out_path):
    coef_map = {
        "competitive_std":                  r"\textit{Polarization}",
        "auditor_event":                    r"Auditor Change",
        "competitive_std:auditor_event":    r"\textit{Polarization} $\times$ Auditor Change",
    }
    dep_labels = [
        r"AbVol\newline Aud. chg.",
        r"AbVol\newline Officer chg.",
        r"AbVol\newline Pooled",
        r"$|CAR|$\newline Aud. chg.",
        r"$|CAR|$\newline Officer chg.",
        r"$|CAR|$\newline Pooled",
    ]
    samples = ["Aud. chg.", "Officer chg.", "Pooled",
               "Aud. chg.", "Officer chg.", "Pooled"]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Officer-Change (Item 5.02) Placebo Test. "
        r"Columns~(1)--(3) use abnormal trading volume; columns~(4)--(6) use "
        r"$|CAR[-1,+1]|$. "
        r"Columns~(1) and~(4) replicate the baseline auditor-change specification "
        r"from Table~\ref{tab:main}. "
        r"Columns~(2) and~(5) apply the identical specification to Form~8-K "
        r"Item~5.02 officer-change filings (departures, elections, and appointments "
        r"of directors and certain officers). "
        r"Columns~(3) and~(6) pool both event types and add an \textit{Auditor Change} "
        r"indicator and its interaction with \textit{Polarization}. "
        r"Events within 30 calendar days of the same firm's auditor-change event "
        r"are excluded from the placebo sample.}"
    )
    lines.append(r"\label{tab:placebo_502}")
    ncols = len(models_list)
    lines.append(r"\begin{tabular}{l" + "c" * ncols + "}")
    lines.append(r"\toprule")
    lines.append(" & " + " & ".join(f"({i+1})" for i in range(ncols)) + r" \\")
    lines.append(" & " + " & ".join(dep_labels) + r" \\")
    lines.append(r"\midrule")

    for param, label in coef_map.items():
        coef_cells, se_cells = [], []
        for m in models_list:
            if m is not None and param in m.params:
                c = m.params[param]
                p = m.pvalues[param]
                s = m.bse[param]
                coef_cells.append(f"{fmt(c)}{stars(p)}")
                se_cells.append(f"({fmt(s)})")
            else:
                coef_cells.append("")
                se_cells.append("")
        lines.append(f"{label} & " + " & ".join(coef_cells) + r" \\")
        lines.append(" & " + " & ".join(se_cells) + r" \\")

    lines.append(r"\midrule")
    lines.append("Year + Industry FE & " + " & ".join(["Yes"] * ncols) + r" \\")
    lines.append("Controls & " + " & ".join(["Yes"] * ncols) + r" \\")
    lines.append("Cluster & " + " & ".join(["Firm"] * ncols) + r" \\")
    lines.append("Sample & " + " & ".join(samples) + r" \\")
    lines.append("N & " + " & ".join(f"{int(m.nobs):,}" for m in models_list) + r" \\")
    lines.append(r"$R^2$ & " + " & ".join(fmt(m.rsquared) for m in models_list) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{flushleft}")
    lines.append(
        r"\footnotesize Notes: Standard errors clustered at the firm level in "
        r"parentheses. $^{***}$, $^{**}$, $^{*}$ denote significance at "
        r"1\%, 5\%, 10\%. \textit{Polarization} is the standardized "
        r"headquarters-state presidential-election competitiveness index. "
        r"The market model is estimated over trading days $[-252,-46]$ relative to "
        r"each filing date using the CRSP value-weighted return."
    )
    lines.append(r"\end{flushleft}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Table written: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== 10_build_officer_change_placebo.py ===")

    # ── [1] Load data ────────────────────────────────────────────────────────
    print("\n[1] Loading analysis sample...")
    analysis = load_analysis_metadata()

    print("\n[2] Loading Item 5.02 events...")
    placebo_raw = load_placebo_events()

    # ── [2] WRDS connection ──────────────────────────────────────────────────
    print("\n[3] Connecting to WRDS...")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)

    # ── [3] CIK -> PERMNO ────────────────────────────────────────────────────
    print("\n[4] Building CIK -> PERMNO link...")
    ciks = placebo_raw["cik"].unique().tolist()
    link = build_cik_permno_link(db, ciks)

    print("\n[5] Matching events to PERMNOs...")
    placebo = apply_link(placebo_raw, link)

    # ── [4] Exclude events near auditor changes ─────────────────────────────
    print("\n[6] Excluding events near auditor changes...")
    placebo = exclude_near_auditor_changes(placebo, analysis)
    print(f"  Placebo events after exclusion: {len(placebo):,}")

    # ── [5] Pull CRSP data ───────────────────────────────────────────────────
    crsp_start = (
        placebo["date_filed"].min() - pd.Timedelta(days=CRSP_LEAD_DAYS)
    ).strftime("%Y-%m-%d")
    crsp_end = (
        placebo["date_filed"].max() + pd.Timedelta(days=5)
    ).strftime("%Y-%m-%d")
    permnos = placebo["permno"].astype(int).unique().tolist()

    print(f"\n[7] Pulling CRSP daily stock returns...")
    daily = pull_crsp_daily(db, permnos, crsp_start, crsp_end)

    print("\n[8] Pulling CRSP market returns...")
    mkt = pull_market_returns(db, crsp_start, crsp_end)

    db.close()

    # ── [6] Compute CARs and AbVol ───────────────────────────────────────────
    print("\n[9] Computing market-model CARs and AbVol...")
    car_df = compute_cars(placebo, daily, mkt)

    # ── [7] Merge and filter ─────────────────────────────────────────────────
    print("\n[10] Merging polarization and controls...")
    pres_pol = pd.read_parquet(POL_FILE)
    comp     = pd.read_parquet(COMP_FILE)

    # Script handles both state_abbr and state column names internally

    placebo_df = merge_and_filter(placebo, car_df, pres_pol, comp)
    print(f"\n  Final placebo sample: {len(placebo_df):,} events, "
          f"{placebo_df['gvkey'].nunique()} firms")

    # Standardize competitive_std within placebo sample
    mu_m  = placebo_df["margin"].mean()
    sig_m = placebo_df["margin"].std()
    placebo_df["competitive_std"] = -(placebo_df["margin"] - mu_m) / sig_m

    # Descriptives
    print(f"\n  Placebo |CAR| mean = {placebo_df['absCar'].mean():.4f}")
    print(f"  Placebo AbVol mean = {placebo_df['abvol'].mean():.4f}")

    placebo_df.to_parquet(OUT_EVENTS, index=False)
    print(f"\n  Saved: {OUT_EVENTS}")

    # ── [8] Regressions ──────────────────────────────────────────────────────
    ctrl = " + ".join(CONTROLS)
    fe   = "C(year_str) + C(sic2_str)"

    print("\n[11] Running regressions...")

    # Build pooled dataset
    keep_cols = ["gvkey", "gvkey_str", "absCar", "abvol", "margin",
                 "year_str", "sic2_str"] + CONTROLS

    main_slice = analysis[keep_cols].copy()
    main_slice["auditor_event"] = 1

    placebo_slice = placebo_df[keep_cols].copy()
    placebo_slice["auditor_event"] = 0

    pooled = pd.concat([main_slice, placebo_slice], ignore_index=True)
    pooled = pooled.dropna(subset=["margin"] + CONTROLS)
    mu_p   = pooled["margin"].mean()
    sig_p  = pooled["margin"].std()
    pooled["competitive_std"] = -(pooled["margin"] - mu_p) / sig_p

    all_models = []
    for depvar, label in [("abvol", "AbVol"), ("absCar", "|CAR|")]:
        m_main = run_ols(f"{depvar} ~ competitive_std + {ctrl} + {fe}", analysis)
        print(f"  [{label}] Main:       β={m_main.params['competitive_std']:.4f}, "
              f"p={m_main.pvalues['competitive_std']:.3f}, N={int(m_main.nobs)}")

        m_placebo = run_ols(f"{depvar} ~ competitive_std + {ctrl} + {fe}", placebo_df)
        print(f"  [{label}] Officer chg: β={m_placebo.params['competitive_std']:.4f}, "
              f"p={m_placebo.pvalues['competitive_std']:.3f}, N={int(m_placebo.nobs)}")

        m_pooled = run_ols(
            f"{depvar} ~ competitive_std + auditor_event"
            f" + competitive_std:auditor_event + {ctrl} + {fe}",
            pooled
        )
        b_int = m_pooled.params.get("competitive_std:auditor_event", np.nan)
        p_int = m_pooled.pvalues.get("competitive_std:auditor_event", np.nan)
        print(f"  [{label}] Pooled int:  β={b_int:.4f}, p={p_int:.3f}, "
              f"N={int(m_pooled.nobs)}")

        all_models.extend([m_main, m_placebo, m_pooled])

    # ── [9] Write table ──────────────────────────────────────────────────────
    print("\n[12] Writing LaTeX table...")
    write_table(all_models, OUT_TABLE)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
