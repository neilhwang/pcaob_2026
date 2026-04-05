"""
20_build_post_event_volatility.py
==================================
Compute pre- and post-event return volatility for each event in the
analysis sample. Used as a validation proxy for the filing informativeness
measure: if the filing is ambiguous, investors should remain uncertain
about firm value after the filing, generating higher return volatility
in the post-event window.

APPROACH
--------
For each event:
  1. Pull CRSP daily returns for [-20, -2] (pre-event window) and [+2, +20]
     (post-event window), plus market returns over the same period.
  2. Compute abnormal returns using alpha_hat and beta_hat already estimated
     for each event in crsp_event_window.parquet (from script 03).
  3. Compute cross-sectional SD of abnormal returns in each window.
  4. Record: pre_vol, post_vol, vol_change = post_vol - pre_vol.

INPUTS
------
  Data/Processed/analysis_sample.parquet   -- event list
  Data/Processed/crsp_event_window.parquet -- alpha_hat, beta_hat
  WRDS: crsp.dsf, crsp.dsi

OUTPUT
------
  Data/Processed/post_event_volatility.parquet
     permno, event_date, pre_vol, post_vol, vol_change,
     n_pre, n_post
"""

from pathlib import Path
import numpy as np
import pandas as pd
import wrds

ROOT     = Path(__file__).resolve().parent.parent
PROC     = ROOT / "Data" / "Processed"
SAMPLE_FILE = PROC / "analysis_sample.parquet"
CRSP_FILE   = PROC / "crsp_event_window.parquet"
OUT_PATH    = PROC / "post_event_volatility.parquet"

WRDS_USERNAME = "nhwang"

# Windows
PRE_START, PRE_END   = -20, -2   # trading days relative to event
POST_START, POST_END = 2, 20


def batch_list(lst, size=300):
    lst = list(lst)
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def main():
    print("=== 20_build_post_event_volatility.py ===")

    sample = pd.read_parquet(SAMPLE_FILE, columns=["permno", "event_date"])
    sample["event_date"] = pd.to_datetime(sample["event_date"])
    sample["permno"] = sample["permno"].astype(int)
    sample = sample.drop_duplicates(subset=["permno", "event_date"])
    print(f"  Events: {len(sample):,}")

    # Alpha_hat, beta_hat from prior market-model estimation
    crsp_params = pd.read_parquet(
        CRSP_FILE, columns=["permno", "event_date", "alpha_hat", "beta_hat"]
    )
    crsp_params["event_date"] = pd.to_datetime(crsp_params["event_date"])
    crsp_params["permno"] = crsp_params["permno"].astype(int)

    sample = sample.merge(
        crsp_params, on=["permno", "event_date"], how="left"
    )
    print(f"  With alpha/beta: {sample[['alpha_hat','beta_hat']].notna().all(axis=1).sum()}")

    # Pull CRSP daily returns for a broad window around each event
    # Use a calendar-day window wide enough to cover 20 trading days on each side
    print("  Connecting to WRDS...")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)

    permnos = sample["permno"].unique().tolist()
    date_start = (sample["event_date"].min() - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    date_end   = (sample["event_date"].max() + pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    print(f"  Date range: {date_start} to {date_end}")

    ret_records = []
    for chunk in batch_list(permnos, 300):
        permno_str = ", ".join(str(p) for p in chunk)
        sql = f"""
            SELECT permno, date, ret
            FROM crsp.dsf
            WHERE permno IN ({permno_str})
              AND date BETWEEN '{date_start}' AND '{date_end}'
              AND ret IS NOT NULL
        """
        ret_records.append(db.raw_sql(sql))
    crsp_ret = pd.concat(ret_records, ignore_index=True)
    crsp_ret["date"] = pd.to_datetime(crsp_ret["date"])
    crsp_ret["permno"] = crsp_ret["permno"].astype(int)
    print(f"  CRSP daily returns: {len(crsp_ret):,}")

    # Market returns
    mkt = db.raw_sql(f"""
        SELECT date, vwretd
        FROM crsp.dsi
        WHERE date BETWEEN '{date_start}' AND '{date_end}'
    """)
    mkt["date"] = pd.to_datetime(mkt["date"])
    print(f"  Market returns: {len(mkt):,}")

    db.close()

    crsp_ret = crsp_ret.merge(mkt, on="date", how="left")

    # Compute trading-day index relative to event date for each firm
    print("  Computing pre/post event volatility...")
    results = []
    for _, row in sample.iterrows():
        permno = row["permno"]
        event_date = row["event_date"]
        alpha = row["alpha_hat"]
        beta  = row["beta_hat"]

        base = dict(permno=permno, event_date=event_date,
                    pre_vol=np.nan, post_vol=np.nan, vol_change=np.nan,
                    n_pre=0, n_post=0)

        if pd.isna(alpha) or pd.isna(beta):
            results.append(base)
            continue

        firm_ret = crsp_ret[crsp_ret["permno"] == permno].sort_values("date").copy()
        if firm_ret.empty:
            results.append(base)
            continue

        # Find trading day index relative to event_date.
        # Event day = first trading day on or after event_date.
        firm_ret["rel_day"] = np.arange(len(firm_ret))  # will be reset below
        idx_on_or_after = firm_ret[firm_ret["date"] >= event_date].index
        if len(idx_on_or_after) == 0:
            results.append(base)
            continue
        event_idx = idx_on_or_after[0]
        # Position of event_idx in firm_ret
        pos = firm_ret.index.get_loc(event_idx)
        firm_ret["rel_day"] = np.arange(len(firm_ret)) - pos

        # Compute abnormal returns
        firm_ret["abn_ret"] = firm_ret["ret"] - (alpha + beta * firm_ret["vwretd"])

        # Pre-event window: rel_day in [-20, -2]
        pre = firm_ret[firm_ret["rel_day"].between(PRE_START, PRE_END)]
        post = firm_ret[firm_ret["rel_day"].between(POST_START, POST_END)]

        if len(pre) >= 5:
            base["pre_vol"] = pre["abn_ret"].std()
            base["n_pre"]   = len(pre)
        if len(post) >= 5:
            base["post_vol"] = post["abn_ret"].std()
            base["n_post"]   = len(post)
        if pd.notna(base["pre_vol"]) and pd.notna(base["post_vol"]):
            base["vol_change"] = base["post_vol"] - base["pre_vol"]

        results.append(base)

    out = pd.DataFrame(results)
    n_total = len(out)
    n_matched = out["vol_change"].notna().sum()
    print(f"\n  Events processed: {n_total:,}")
    print(f"  With vol_change:  {n_matched:,} ({n_matched/n_total:.1%})")
    if n_matched > 0:
        print(f"\n  pre_vol:    mean={out['pre_vol'].mean():.5f}  median={out['pre_vol'].median():.5f}")
        print(f"  post_vol:   mean={out['post_vol'].mean():.5f}  median={out['post_vol'].median():.5f}")
        print(f"  vol_change: mean={out['vol_change'].mean():.5f}  median={out['vol_change'].median():.5f}")

    out.to_parquet(OUT_PATH, index=False)
    print(f"\n  Saved: {OUT_PATH}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
