"""
11_build_ibes.py
================
Pulls IBES analyst coverage and forecast dispersion for each firm-year in the
analysis sample and writes the result to Data/Processed/ibes_dispersion.parquet.

INPUTS
------
  Data/Processed/analysis_sample.parquet   -- main analysis sample (has gvkey, permno,
                                              event_date, comp_year)
  WRDS: crsp.dsenames                      -- CRSP historical security names with CUSIP
  WRDS: ibes.statsumu_epsus               -- IBES unadjusted summary statistics (US firms)

OUTPUTS
-------
  Data/Processed/ibes_dispersion.parquet  -- one row per (gvkey, comp_year) with
      pre-event and post-event analyst dispersion:
      analyst_coverage       : numest as of pre-event consensus
      disp_raw_pre           : pre-event stdev (most recent statpers <= event_date)
      disp_scaled_pre        : disp_raw_pre / |meanest_pre|
      disp_raw_post          : post-event stdev (first statpers in event+[1,90])
      disp_scaled_post       : disp_raw_post / |meanest_post|
      disp_change_scaled     : disp_scaled_post - disp_scaled_pre
      (post values are NaN if no IBES observation falls within 90 days after
      the event date)

LINK STRATEGY
-------------
wrdsapps.ibcrsphist and ibes.id are not accessible under this WRDS account
(permission denied for schemas wrdsapps_link_crsp_ibes and tr_ibes).

Instead we use a pure-CUSIP bridge:
  crsp.dsenames.ncusip  (8-digit CUSIP, with date range namedt-nameendt)
  ibes.statsumu_epsus.cusip (8-digit CUSIP embedded in the summary stats file)

Both use the same 8-character CUSIP format. For each event we:
  1. Pull the ncusip(s) active for the firm's permno at event_date.
  2. Filter ibes.statsumu_epsus rows to those CUSIPs, with
     fy_end_year == comp_year and statpers <= event_date.
  3. Take the most recent statpers row and record numest, stdev, meanest.

NOTES
-----
- Uses wrds.Connection(wrds_username="nhwang") for WRDS access.
- Queries use IN-clause batching to avoid hitting WRDS row-limit issues.
- disp_scaled is NaN when stdev is missing, numest < 2, or |meanest| < 0.01.
"""

import os
import warnings
import numpy as np
import pandas as pd
import wrds

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "Data", "Processed")
ANALYSIS_SAMPLE = os.path.join(PROCESSED, "analysis_sample.parquet")
OUT_PATH = os.path.join(PROCESSED, "ibes_dispersion.parquet")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def batch_list(lst, size=500):
    """Yield successive chunks of `lst` of length `size`."""
    lst = list(lst)
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def sql_in(values):
    """Return a SQL-ready parenthesised list of quoted strings."""
    escaped = ["'" + str(v).replace("'", "''") + "'" for v in values]
    return "(" + ", ".join(escaped) + ")"


# ---------------------------------------------------------------------------
# Step 1: Load analysis sample
# ---------------------------------------------------------------------------

def load_sample():
    """Load only the columns needed for the IBES merge."""
    df = pd.read_parquet(
        ANALYSIS_SAMPLE,
        columns=["gvkey", "permno", "event_date", "comp_year"],
    )
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["permno"] = df["permno"].astype(int)
    df["comp_year"] = df["comp_year"].astype(int)
    # Keep one row per (gvkey, comp_year): earliest event_date
    df = (
        df.sort_values("event_date")
        .drop_duplicates(subset=["gvkey", "comp_year"], keep="first")
        .reset_index(drop=True)
    )
    print(f"  Analysis sample: {len(df):,} unique (gvkey, comp_year) pairs")
    return df


# ---------------------------------------------------------------------------
# Step 2: Pull CRSP historical CUSIP map
# ---------------------------------------------------------------------------

def pull_crsp_cusip_map(db, permnos):
    """
    Pull crsp.dsenames to get the historical CUSIP (ncusip) for each permno,
    with date range [namedt, nameendt].

    Returns a DataFrame with columns: permno (int), ncusip, namedt, nameendt.
    """
    records = []
    for chunk in batch_list(permnos, 500):
        sql = f"""
            SELECT permno, ncusip, namedt, nameendt
            FROM crsp.dsenames
            WHERE permno IN {sql_in(chunk)}
              AND ncusip IS NOT NULL
              AND ncusip <> ''
        """
        records.append(db.raw_sql(sql))

    crsp_names = pd.concat(records, ignore_index=True)
    crsp_names["permno"]   = crsp_names["permno"].astype(int)
    crsp_names["namedt"]   = pd.to_datetime(crsp_names["namedt"])
    crsp_names["nameendt"] = pd.to_datetime(
        crsp_names["nameendt"].fillna("9999-12-31")
    )
    print(f"  crsp.dsenames: {len(crsp_names):,} rows, "
          f"{crsp_names['ncusip'].nunique():,} unique CUSIPs")
    return crsp_names


# ---------------------------------------------------------------------------
# Step 3: Pull IBES consensus statistics by CUSIP
# ---------------------------------------------------------------------------

def pull_ibes_by_cusip(db, cusips, comp_years):
    """
    Pull annual EPS consensus from ibes.statsumu_epsus filtered by CUSIP.

    ibes.statsumu_epsus has a cusip column (same 8-digit format as
    crsp.dsenames.ncusip), so no separate link table is needed.

    Returns a DataFrame with columns:
        cusip, statpers, fpedats, fy_end_year, numest, stdev, meanest
    """
    min_year = min(comp_years)
    max_year = max(comp_years)
    start_date = f"{min_year - 1}-01-01"
    end_date   = f"{max_year + 1}-12-31"

    records = []
    for chunk in batch_list(cusips, 500):
        sql = f"""
            SELECT cusip, statpers, fpedats, fpi, fiscalp,
                   numest, stdev, meanest
            FROM ibes.statsumu_epsus
            WHERE fpi     = '1'
              AND fiscalp  = 'ANN'
              AND usfirm   = 1
              AND cusip IN {sql_in(chunk)}
              AND statpers BETWEEN '{start_date}' AND '{end_date}'
        """
        records.append(db.raw_sql(sql))

    ibes = pd.concat(records, ignore_index=True)
    ibes["statpers"]    = pd.to_datetime(ibes["statpers"])
    ibes["fpedats"]     = pd.to_datetime(ibes["fpedats"])
    ibes["fy_end_year"] = ibes["fpedats"].dt.year
    print(f"  IBES consensus: {len(ibes):,} rows for "
          f"{ibes['cusip'].nunique():,} CUSIPs")
    return ibes


# ---------------------------------------------------------------------------
# Step 4: Match each event to the right IBES observation
# ---------------------------------------------------------------------------

def _compute_disp(numest, stdev, meanest):
    """Return (disp_raw, disp_scaled) with NaN guards."""
    if (pd.notna(stdev) and pd.notna(meanest) and pd.notna(numest)
            and numest >= 2 and abs(meanest) >= 0.01):
        return float(stdev), float(stdev) / abs(float(meanest))
    return (float(stdev) if pd.notna(stdev) else np.nan, np.nan)


def build_dispersion(sample, crsp_names, ibes):
    """
    For each (gvkey, comp_year) in sample, extract PRE- and POST-event
    analyst dispersion:
      1. Find the ncusip(s) active for the firm's permno at event_date
         (from crsp_names, using the namedt-nameendt date range).
      2. Pre-event: most recent statpers <= event_date.
      3. Post-event: first statpers in (event_date, event_date + 90 days].
      4. Compute disp_scaled = stdev / |meanest| for each.

    Returns one row per (gvkey, comp_year) with pre-, post-, and change
    measures. Post values are NaN if no IBES observation falls in window.
    """
    results = []
    post_window_end = pd.Timedelta(days=90)

    for _, row in sample.iterrows():
        gvkey      = row["gvkey"]
        permno     = row["permno"]
        event_date = row["event_date"]
        comp_year  = row["comp_year"]

        active_cusips = crsp_names.loc[
            (crsp_names["permno"]   == permno)
            & (crsp_names["namedt"]   <= event_date)
            & (crsp_names["nameendt"] >= event_date),
            "ncusip",
        ].unique().tolist()

        base = dict(
            gvkey=gvkey, comp_year=comp_year,
            analyst_coverage=np.nan,
            disp_raw_pre=np.nan, disp_scaled_pre=np.nan,
            disp_raw_post=np.nan, disp_scaled_post=np.nan,
            disp_change_scaled=np.nan,
        )

        if not active_cusips:
            results.append(base)
            continue

        cusip_mask = ibes["cusip"].isin(active_cusips) & (ibes["fy_end_year"] == comp_year)

        # ── Pre-event: most recent statpers <= event_date ──
        pre = ibes[cusip_mask & (ibes["statpers"] <= event_date)]
        if not pre.empty:
            best = pre.loc[pre["statpers"].idxmax()]
            base["analyst_coverage"] = (
                float(best["numest"]) if pd.notna(best["numest"]) else np.nan
            )
            base["disp_raw_pre"], base["disp_scaled_pre"] = _compute_disp(
                best["numest"], best["stdev"], best["meanest"]
            )

        # ── Post-event: first statpers in (event_date, event_date + 90d] ──
        post = ibes[cusip_mask
                    & (ibes["statpers"] > event_date)
                    & (ibes["statpers"] <= event_date + post_window_end)]
        if not post.empty:
            best = post.loc[post["statpers"].idxmin()]
            base["disp_raw_post"], base["disp_scaled_post"] = _compute_disp(
                best["numest"], best["stdev"], best["meanest"]
            )

        # ── Change: post - pre (scaled) ──
        if pd.notna(base["disp_scaled_pre"]) and pd.notna(base["disp_scaled_post"]):
            base["disp_change_scaled"] = (
                base["disp_scaled_post"] - base["disp_scaled_pre"]
            )

        results.append(base)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== 11_build_ibes.py ===")

    # 1. Load sample
    print("\n[1] Loading analysis sample...")
    sample = load_sample()

    permnos    = sample["permno"].unique().tolist()
    comp_years = sample["comp_year"].unique().tolist()

    # 2. Connect to WRDS
    print("\n[2] Connecting to WRDS...")
    db = wrds.Connection(wrds_username="nhwang")

    # 3. Pull CRSP CUSIP map
    print("\n[3] Pulling CRSP historical CUSIP map (crsp.dsenames)...")
    crsp_names = pull_crsp_cusip_map(db, permnos)

    # 4. Pull IBES consensus by CUSIP
    cusips = crsp_names["ncusip"].unique().tolist()
    print(f"\n[4] Pulling IBES annual consensus for {len(cusips):,} CUSIPs "
          f"(ibes.statsumu_epsus)...")
    ibes = pull_ibes_by_cusip(db, cusips, comp_years)

    db.close()

    # 5. Build dispersion measures
    print("\n[5] Matching events to IBES observations...")
    disp = build_dispersion(sample, crsp_names, ibes)

    # 6. Summary
    n_total = len(disp)
    n_cov   = disp["analyst_coverage"].notna().sum()
    n_disp  = disp["disp_scaled"].notna().sum()
    print(f"\n  Rows in output          : {n_total:,}")
    print(f"  With analyst_coverage   : {n_cov:,} ({n_cov/n_total:.1%})")
    print(f"  With disp_scaled        : {n_disp:,} ({n_disp/n_total:.1%})")
    print(f"\n  disp_scaled summary:")
    print(disp["disp_scaled"].describe().to_string())
    print(f"\n  analyst_coverage summary:")
    print(disp["analyst_coverage"].describe().to_string())

    # 7. Save
    disp.to_parquet(OUT_PATH, index=False)
    print(f"\n  Saved -> {OUT_PATH}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
