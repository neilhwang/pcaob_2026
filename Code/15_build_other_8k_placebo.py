"""
15_build_other_8k_placebo.py
============================
Constructs an earnings-announcement placebo test for the same firms as the
main analysis sample.

PURPOSE
-------
Validates that the polarization effect is specific to auditor-change disclosures
rather than a general response to any corporate news in high-polarization states.
Quarterly earnings announcements (Compustat rdq) carry unambiguous quantitative
content — the reported EPS figure — leaving far less room for politically shaped
interpretation. A null coefficient in the placebo regression is consistent with
the paper's interpretation channel: politically heterogeneous investors disagree
specifically when the signal is ambiguous, not when it is numerically precise.

A significant positive interaction in the pooled column further identifies that
the polarization effect is differentially stronger for auditor-change disclosures
than for earnings announcements from the same firms.

DESIGN
------
  Three columns in the output table:
    (1) Main sample baseline — replicates Table 2 col. 2 for comparison
    (2) Earnings placebo     — same specification, earnings-announcement events
    (3) Pooled               — stacks both samples with an Auditor Change indicator
                               and a Polarization × Auditor Change interaction;
                               a positive interaction coefficient means the effect
                               is significantly stronger for auditor changes

INPUTS
------
  Data/Processed/analysis_sample.parquet          (from 05_merge_and_estimate.py)
  Data/Processed/polarization_state_year.parquet  (from 02_build_polarization.py)
  Data/Processed/pol_presidential.parquet         (from 02b_build_presidential_polarization.py)
  Data/Processed/compustat_controls.parquet       (from 04_build_compustat_controls.py)
  WRDS: comp.fundq                                -- quarterly earnings dates (rdq)
  WRDS: crsp.dsf                                  -- daily stock returns
  WRDS: crsp.dsi                                  -- market returns (vwretd)

OUTPUTS
-------
  Data/Processed/earnings_placebo.parquet    -- placebo event-level dataset
  Output/Tables/tab10_placebo_8k.tex         -- placebo regression table

MARKET MODEL
------------
  Identical to 03_build_crsp_sample.py:
    ret_{it} = alpha_i + beta_i * vwretd_t   estimated over [-252, -46]
    CAR(-1,+1) = sum of AR over event window [-1, 0, +1]
    Minimum 100 non-missing estimation-window trading days required.

SAMPLE RESTRICTIONS
-------------------
  - Restricted to gvkeys present in the main analysis sample
  - Same fiscal year range as analysis sample (event_year in [year_min, year_max])
  - Earnings events within 30 calendar days of the same firm's auditor change
    event are excluded (contamination guard)
  - Same SIC exclusions: no financials (SIC 6000-6999), no utilities (4900-4999)
  - Compustat controls from fiscal year prior to event year (fyear = event_year-1)
    must be non-missing

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
POL_FILE        = PROC / "polarization_state_year.parquet"
PRES_POL_FILE   = PROC / "pol_presidential.parquet"
COMP_FILE       = PROC / "compustat_controls.parquet"
OUT_EVENTS      = PROC / "earnings_placebo.parquet"
OUT_TABLE       = OUT_TABS / "tab10_placebo_8k.tex"

WRDS_USERNAME = "nhwang"

# Market model parameters — identical to 03_build_crsp_sample.py
EST_WINDOW_START = -252
EST_WINDOW_END   = -46
MIN_EST_DAYS     = 100

CONTROLS = ["size", "leverage", "roa", "btm", "loss", "sales_growth"]

# Exclude placebo events within this many calendar days of an auditor-change event
EXCLUSION_DAYS = 30

# Calendar-day buffer before earliest event to ensure full estimation window
CRSP_LEAD_DAYS = 380   # 252 trading days ≈ 355 calendar days; 380 is safe


# ── Helpers ──────────────────────────────────────────────────────────────────

def batch_list(lst, size=200):
    lst = list(lst)
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def sql_in_str(values):
    """SQL IN clause for character varying columns (e.g. gvkey)."""
    escaped = ["'" + str(v).replace("'", "''") + "'" for v in values]
    return "(" + ", ".join(escaped) + ")"


def sql_in_int(values):
    """SQL IN clause for integer columns (e.g. permno)."""
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


# ── Step 2: Pull earnings announcement dates (comp.fundq) ─────────────────────

def pull_earnings_dates(db, gvkeys, year_min, year_max):
    """
    Pull quarterly earnings announcement dates from comp.fundq.
    Filtered to gvkeys in analysis sample and event year range.
    One row per (gvkey, fyearq, fqtr).
    """
    print(f"  Querying comp.fundq for {len(gvkeys):,} gvkeys, "
          f"fyearq {year_min}–{year_max}...")
    records = []
    for chunk in batch_list(gvkeys, 500):
        sql = f"""
            SELECT gvkey, rdq, fyearq, fqtr
            FROM comp.fundq
            WHERE gvkey IN {sql_in_str(chunk)}
              AND rdq IS NOT NULL
              AND fyearq BETWEEN {year_min} AND {year_max}
              AND indfmt  = 'INDL'
              AND datafmt = 'STD'
              AND popsrc  = 'D'
              AND consol  = 'C'
        """
        records.append(db.raw_sql(sql))
    rdq = pd.concat(records, ignore_index=True)
    rdq["rdq"] = pd.to_datetime(rdq["rdq"])
    # One row per firm-quarter (drop any Compustat data-format duplicates)
    rdq = rdq.drop_duplicates(subset=["gvkey", "fyearq", "fqtr"])
    print(f"  Earnings dates: {len(rdq):,} events, {rdq['gvkey'].nunique()} firms")
    return rdq


# ── Step 3: Build placebo event list ─────────────────────────────────────────

def build_placebo_events(rdq, analysis_df):
    """
    Attach permno and HQ state to each earnings event, then exclude
    events within EXCLUSION_DAYS of an auditor-change event for the same firm.
    """
    # Most-recent state per gvkey (state is effectively time-invariant)
    state_map = (
        analysis_df[["gvkey", "state"]]
        .drop_duplicates(subset="gvkey", keep="last")
    )
    # Permno per gvkey
    permno_map = (
        analysis_df[["gvkey", "permno"]]
        .drop_duplicates(subset="gvkey", keep="last")
    )

    placebo = (
        rdq
        .merge(state_map,  on="gvkey", how="inner")
        .merge(permno_map, on="gvkey", how="inner")
        .rename(columns={"rdq": "event_date"})
    )
    placebo["event_year"] = placebo["event_date"].dt.year

    # Exclude earnings events close in time to an auditor-change event
    ac = analysis_df[["gvkey", "event_date"]].rename(
        columns={"event_date": "ac_date"}
    )
    merged = placebo.merge(ac, on="gvkey", how="left")
    merged["gap"] = (merged["event_date"] - merged["ac_date"]).abs().dt.days

    min_gap = (
        merged.groupby(["gvkey", "event_date"])["gap"]
        .min()
        .reset_index(name="min_gap")
    )
    n_before = len(placebo)
    placebo = (
        placebo
        .merge(min_gap, on=["gvkey", "event_date"], how="left")
        .query("min_gap.isna() or min_gap > @EXCLUSION_DAYS")
        .drop(columns="min_gap")
        .reset_index(drop=True)
    )
    print(f"  Excluded {n_before - len(placebo):,} events within "
          f"{EXCLUSION_DAYS} days of an auditor change")
    print(f"  Placebo events: {len(placebo):,}")
    return placebo


# ── Step 4: Pull CRSP data ───────────────────────────────────────────────────

def pull_crsp_daily(db, permnos, start_date, end_date):
    print(f"  Date range: {start_date} → {end_date}")
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


# ── Step 5: Compute CARs ─────────────────────────────────────────────────────

def trading_day_offset(event_date, trading_days, offset):
    """Return trading day 'offset' slots from event_date. None if out of range."""
    idx    = trading_days.searchsorted(event_date)
    target = idx + offset
    if target < 0 or target >= len(trading_days):
        return None
    return trading_days.iloc[target]


def compute_cars(placebo, daily, mkt):
    """
    Estimate market model and compute CAR(-1,+1) and AbVol(-1,+1) for every
    placebo event. Same OLS methodology as 03_build_crsp_sample.py.
    """
    # Sorted trading-day calendar from combined CRSP data
    trading_days = (
        pd.concat([daily["date"], mkt["date"]])
        .drop_duplicates().sort_values().reset_index(drop=True)
    )
    mkt_dict  = mkt.set_index("date")["mkt_ret"].to_dict()

    # Pre-compute log volume for AbVol
    daily = daily.copy()
    daily["logvol"] = np.log(daily["vol"].clip(lower=0) + 1)

    daily_idx = daily.set_index("permno")

    results  = []
    n_total  = len(placebo)
    n_report = max(1, n_total // 10)   # progress every ~10%

    for i, (_, row) in enumerate(placebo.iterrows()):
        gvkey      = row["gvkey"]
        permno     = row["permno"]
        event_date = row["event_date"]

        if i % n_report == 0:
            print(f"    {i:,} / {n_total:,} ({100*i/n_total:.0f}%)")

        base = dict(gvkey=gvkey, permno=permno, event_date=event_date,
                    car_m1p1=np.nan, abvol_m1p1=np.nan, n_est_days=0)

        # Resolve window boundaries
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

        # OLS: ret = alpha + beta * mkt_ret
        X = np.column_stack([np.ones(len(est)), est["mkt_ret"].values])
        y = est["ret"].values
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            results.append(base); continue
        alpha_hat, beta_hat = coef

        # Estimation-window mean log volume (baseline for AbVol)
        est_logvol = est["logvol"].dropna()
        mean_logvol = est_logvol.mean() if len(est_logvol) > 0 else np.nan

        # Event window AR → CAR and AbVol
        ev = firm[(firm["date"] >= ev_start) & (firm["date"] <= ev_end)].copy()
        ev["mkt_ret"] = ev["date"].map(mkt_dict)
        ev_ret = ev.dropna(subset=["ret", "mkt_ret"])
        if len(ev_ret) == 0:
            results.append({**base, "n_est_days": len(est)}); continue

        car = (ev_ret["ret"] - (alpha_hat + beta_hat * ev_ret["mkt_ret"])).sum()

        # AbVol: mean(logvol - baseline) over event window
        ev_logvol = ev["logvol"].dropna()
        if len(ev_logvol) > 0 and not np.isnan(mean_logvol):
            abvol = (ev_logvol - mean_logvol).mean()
        else:
            abvol = np.nan

        results.append({**base, "car_m1p1": car, "abvol_m1p1": abvol,
                        "n_est_days": len(est)})

    out = pd.DataFrame(results)
    n_valid_car  = out["car_m1p1"].notna().sum()
    n_valid_vol  = out["abvol_m1p1"].notna().sum()
    print(f"\n  Valid CARs:  {n_valid_car:,} / {n_total:,} ({n_valid_car/n_total:.1%})")
    print(f"  Valid AbVol: {n_valid_vol:,} / {n_total:,} ({n_valid_vol/n_total:.1%})")
    return out


# ── Step 6: Merge polarization and controls ──────────────────────────────────

def merge_and_filter(placebo, car_df, pres_pol, comp):
    """
    Attach CARs, presidential-election polarization (margin), and Compustat
    controls. Apply the same sample filters as the main analysis.
    """
    df = placebo.merge(
        car_df[["gvkey", "event_date", "car_m1p1", "abvol_m1p1", "n_est_days"]],
        on=["gvkey", "event_date"], how="inner"
    )
    df = df[df["car_m1p1"].notna()].copy()

    # Presidential-election margin (same measure as main analysis: margin = |D-R|)
    pres_merge = pres_pol[["year", "state_abbr", "margin"]].rename(
        columns={"year": "event_year", "state_abbr": "state"}
    )
    df = df.merge(pres_merge, on=["event_year", "state"], how="left")

    # Compustat controls from fiscal year prior to event year
    df["comp_year"] = df["event_year"] - 1
    comp_merge = comp[["gvkey", "fyear", "sic2"] + CONTROLS].rename(
        columns={"fyear": "comp_year"}
    )
    df = df.merge(comp_merge, on=["gvkey", "comp_year"], how="left")

    # FE string columns
    df["year_str"]  = df["event_year"].astype(str)
    df["sic2_str"]  = df["sic2"].fillna(-1).astype(int).astype(str)
    df["gvkey_str"] = df["gvkey"].astype(str)
    df["absCar"]    = df["car_m1p1"].abs()
    df["abvol"]     = df["abvol_m1p1"]

    # Drop missing key variables
    n0 = len(df)
    df = df.dropna(subset=["absCar", "margin", "sic2"] + CONTROLS)
    print(f"  After dropping missing: {len(df):,} (dropped {n0-len(df):,})")

    # SIC exclusions — same as main analysis
    n1 = len(df)
    df = df[~df["sic2"].isin(range(60, 70)) & ~df["sic2"].isin(range(49, 50))]
    print(f"  After SIC exclusions: {len(df):,} (dropped {n1-len(df):,})")

    return df.reset_index(drop=True)


# ── Step 7: Regressions and table ────────────────────────────────────────────

def run_ols(formula, df, cluster_var="gvkey_str"):
    """OLS with firm-clustered standard errors."""
    import patsy
    _, X = patsy.dmatrices(formula, data=df, return_type="dataframe",
                            NA_action="drop")
    keep = X.index
    groups = df.loc[keep, cluster_var].values
    return smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )


def write_table(models_list, out_path):
    """
    Write a 6-column LaTeX table: AbVol (cols 1-3) then |CAR| (cols 4-6),
    each with main / earnings / pooled specifications.
    """
    coef_map = {
        "competitive_std":                  r"\textit{Polarization}",
        "auditor_event":                    r"Auditor Change",
        "competitive_std:auditor_event":    r"\textit{Polarization} $\times$ Auditor Change",
    }
    dep_labels = [
        r"AbVol\newline Aud. chg.",
        r"AbVol\newline Earnings",
        r"AbVol\newline Pooled",
        r"$|CAR|$\newline Aud. chg.",
        r"$|CAR|$\newline Earnings",
        r"$|CAR|$\newline Pooled",
    ]
    samples = ["Aud. chg.", "Earnings", "Pooled",
               "Aud. chg.", "Earnings", "Pooled"]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Earnings-Announcement Placebo Test. "
        r"Columns~(1)--(3) use abnormal trading volume; columns~(4)--(6) use "
        r"$|CAR[-1,+1]|$. "
        r"Columns~(1) and~(4) replicate the baseline specification from "
        r"Table~\ref{tab:main}. "
        r"Columns~(2) and~(5) apply the identical specification to quarterly earnings "
        r"announcements (\texttt{rdq} from Compustat) for the same firms over the "
        r"same period. "
        r"Columns~(3) and~(6) pool both event types and add an \textit{Auditor Change} "
        r"indicator and its interaction with \textit{Polarization}. "
        r"Events within 30 calendar days of an auditor-change event for the same firm "
        r"are excluded from the placebo sample.}"
    )
    lines.append(r"\label{tab:placebo_8k}")
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
        r"headquarters-state presidential-election competitiveness index (mean zero, "
        r"unit standard deviation computed within each column's estimation sample). "
        r"The market model is estimated over trading days $[-252,-46]$ relative to "
        r"each announcement date using the CRSP value-weighted return."
    )
    lines.append(r"\end{flushleft}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Table written: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== 15_build_other_8k_placebo.py ===")

    # ── [1] Load analysis sample ─────────────────────────────────────────────
    print("\n[1] Loading analysis sample...")
    analysis = load_analysis_metadata()
    gvkeys   = analysis["gvkey"].unique().tolist()
    year_min = analysis["event_date"].dt.year.min()
    year_max = analysis["event_date"].dt.year.max()
    print(f"  Event year range in analysis sample: {year_min}–{year_max}")

    # ── [2] WRDS connection ──────────────────────────────────────────────────
    print("\n[2] Connecting to WRDS...")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)

    # ── [3] Pull earnings dates ──────────────────────────────────────────────
    print("\n[3] Pulling earnings announcement dates (comp.fundq)...")
    rdq = pull_earnings_dates(db, gvkeys, year_min, year_max)

    # ── [4] Build placebo event list ─────────────────────────────────────────
    print("\n[4] Building placebo event list...")
    placebo = build_placebo_events(rdq, analysis)

    # ── [5] Pull CRSP data ───────────────────────────────────────────────────
    crsp_start = (
        placebo["event_date"].min() - pd.Timedelta(days=CRSP_LEAD_DAYS)
    ).strftime("%Y-%m-%d")
    crsp_end = (
        placebo["event_date"].max() + pd.Timedelta(days=5)
    ).strftime("%Y-%m-%d")
    permnos = placebo["permno"].unique().tolist()

    print(f"\n[5] Pulling CRSP daily stock returns ({crsp_start} → {crsp_end})...")
    daily = pull_crsp_daily(db, permnos, crsp_start, crsp_end)

    print("\n[6] Pulling CRSP market returns...")
    mkt = pull_market_returns(db, crsp_start, crsp_end)

    db.close()

    # ── [6] Compute CARs ─────────────────────────────────────────────────────
    print("\n[7] Computing market-model CARs for placebo events...")
    car_df = compute_cars(placebo, daily, mkt)

    # ── [7] Merge and filter ─────────────────────────────────────────────────
    print("\n[8] Merging polarization and controls...")
    pres_pol = pd.read_parquet(PRES_POL_FILE)
    comp     = pd.read_parquet(COMP_FILE)
    placebo_df = merge_and_filter(placebo, car_df, pres_pol, comp)
    print(f"\n  Final placebo sample: {len(placebo_df):,} events, "
          f"{placebo_df['gvkey'].nunique()} firms")

    # Standardize competitive_std within placebo sample
    mu_m  = placebo_df["margin"].mean()
    sig_m = placebo_df["margin"].std()
    placebo_df["competitive_std"] = -(placebo_df["margin"] - mu_m) / sig_m

    # Describe placebo |CAR| vs main |CAR|
    print(f"\n  Placebo |CAR| mean = {placebo_df['absCar'].mean():.4f}, "
          f"median = {placebo_df['absCar'].median():.4f}")
    print(f"  Main   |CAR| mean = {analysis['absCar'].mean():.4f}, "
          f"median = {analysis['absCar'].median():.4f}")

    placebo_df.to_parquet(OUT_EVENTS, index=False)
    print(f"\n  Saved: {OUT_EVENTS}")

    # ── [8] Regressions ──────────────────────────────────────────────────────
    ctrl = " + ".join(CONTROLS)
    fe   = "C(year_str) + C(sic2_str)"

    print("\n[9] Running regressions...")

    # Build pooled dataset (shared across both outcomes)
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

    # Run for both outcomes; AbVol first (primary), then |CAR|
    all_models = []
    for depvar, label in [("abvol", "AbVol"), ("absCar", "|CAR|")]:
        # (a) Baseline: main sample
        m_main = run_ols(f"{depvar} ~ competitive_std + {ctrl} + {fe}", analysis)
        print(f"  [{label}] Main:     β={m_main.params['competitive_std']:.4f}, "
              f"p={m_main.pvalues['competitive_std']:.3f}, N={int(m_main.nobs)}")

        # (b) Placebo: earnings announcements
        m_placebo = run_ols(f"{depvar} ~ competitive_std + {ctrl} + {fe}", placebo_df)
        print(f"  [{label}] Earnings: β={m_placebo.params['competitive_std']:.4f}, "
              f"p={m_placebo.pvalues['competitive_std']:.3f}, N={int(m_placebo.nobs)}")

        # (c) Pooled: interaction
        m_pooled = run_ols(
            f"{depvar} ~ competitive_std + auditor_event"
            f" + competitive_std:auditor_event + {ctrl} + {fe}",
            pooled
        )
        b_int = m_pooled.params.get("competitive_std:auditor_event", np.nan)
        p_int = m_pooled.pvalues.get("competitive_std:auditor_event", np.nan)
        print(f"  [{label}] Pooled interaction: β={b_int:.4f}, p={p_int:.3f}, "
              f"N={int(m_pooled.nobs)}")

        all_models.extend([m_main, m_placebo, m_pooled])

    # ── [9] Write table ───────────────────────────────────────────────────────
    print("\n[10] Writing LaTeX table...")
    write_table(all_models, OUT_TABLE)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
