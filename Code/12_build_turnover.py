"""
12_build_turnover.py
====================
Computes pre-event share turnover for each event in the analysis sample.

Pre-event turnover is a proxy for retail / local investor presence: low-turnover
stocks are less liquid and more likely to be held by local retail investors rather
than nationally diversified institutions. The prediction from the local-bias
channel is that the polarization effect should be stronger in low-turnover firms,
where the marginal trader is more likely to share the firm's HQ-state political
environment.

INPUTS
------
  Data/Processed/analysis_sample.parquet  -- has permno, event_date
  WRDS: crsp.dsf                          -- CRSP daily stock file

OUTPUTS
-------
  Data/Processed/pre_event_turnover.parquet  -- one row per (gvkey, event_date):
      turnover_pre  : mean daily turnover in the pre-event window
                      = mean(vol / (shrout * 1000)) over calendar days
                        [event_date - 365, event_date - 22]
      turnover_days : number of trading days used in the mean (data quality check)

WINDOW CHOICE
-------------
  [-365, -22] calendar days before event_date.
  - Lower bound (-365) gives roughly one year of pre-event data.
  - Upper bound (-22) excludes the 3-week pre-event window to avoid
    contamination from anticipatory trading.
  Observations with fewer than 60 valid trading days are set to NaN.

NOTES
-----
- shrout is in thousands of shares; vol is in shares.
- Turnover = vol / (shrout * 1000), computed daily, then averaged.
- Zero-volume days are included (they represent days with no trading, which
  is informative for low-liquidity stocks). Days with shrout <= 0 or missing
  are excluded.
- WRDS username: nhwang
- Queries are batched by permno to avoid row-limit issues.
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
OUT_PATH = os.path.join(PROCESSED, "pre_event_turnover.parquet")

MIN_DAYS = 60   # minimum valid trading days required; else NaN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def batch_list(lst, size=200):
    lst = list(lst)
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def sql_in(values):
    escaped = ["'" + str(v).replace("'", "''") + "'" for v in values]
    return "(" + ", ".join(escaped) + ")"


# ---------------------------------------------------------------------------
# Step 1: Load sample
# ---------------------------------------------------------------------------

def load_sample():
    df = pd.read_parquet(
        ANALYSIS_SAMPLE,
        columns=["gvkey", "permno", "event_date"],
    )
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["permno"] = df["permno"].astype(int)
    # Keep distinct (gvkey, permno, event_date) triples
    df = df.drop_duplicates(subset=["gvkey", "permno", "event_date"]).reset_index(drop=True)
    print(f"  Events to process: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Pull CRSP daily data for pre-event windows
# ---------------------------------------------------------------------------

def pull_crsp_daily(db, sample):
    """
    For efficiency, pull all permno × date combinations in one set of queries
    spanning the full date range in the sample, then filter per-event in Python.

    Window: [earliest event_date - 365, latest event_date - 22] calendar days.
    We pull a superset and filter to the per-event window in build_turnover().
    """
    permnos = sample["permno"].unique().tolist()
    global_start = (sample["event_date"].min() - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    global_end   = (sample["event_date"].max() - pd.Timedelta(days=22)).strftime("%Y-%m-%d")

    print(f"  Date range: {global_start} to {global_end}")
    print(f"  Permnos: {len(permnos):,}")

    records = []
    for i, chunk in enumerate(batch_list(permnos, 200)):
        sql = f"""
            SELECT permno, date, vol, shrout
            FROM crsp.dsf
            WHERE permno IN {sql_in(chunk)}
              AND date BETWEEN '{global_start}' AND '{global_end}'
              AND shrout > 0
              AND shrout IS NOT NULL
        """
        chunk_df = db.raw_sql(sql)
        records.append(chunk_df)
        if (i + 1) % 5 == 0:
            print(f"    ... fetched {i+1} batches")

    daily = pd.concat(records, ignore_index=True)
    daily["date"]   = pd.to_datetime(daily["date"])
    daily["permno"] = daily["permno"].astype(int)
    # Compute daily turnover
    daily["turnover"] = daily["vol"] / (daily["shrout"] * 1000.0)
    # Cap extreme outliers at 1.0 (turnover > 100% of shares in a day is data error)
    daily["turnover"] = daily["turnover"].clip(upper=1.0)
    print(f"  CRSP daily rows pulled: {len(daily):,}")
    return daily


# ---------------------------------------------------------------------------
# Step 3: Compute per-event pre-event turnover
# ---------------------------------------------------------------------------

def build_turnover(sample, daily):
    """
    For each event, compute mean daily turnover over [event_date-365, event_date-22].
    """
    results = []

    # Index daily by permno for faster filtering
    daily_grouped = daily.set_index("permno")

    for _, row in sample.iterrows():
        gvkey      = row["gvkey"]
        permno     = row["permno"]
        event_date = row["event_date"]

        win_start = event_date - pd.Timedelta(days=365)
        win_end   = event_date - pd.Timedelta(days=22)

        if permno not in daily_grouped.index:
            results.append(dict(gvkey=gvkey, event_date=event_date,
                                turnover_pre=np.nan, turnover_days=0))
            continue

        firm_daily = daily_grouped.loc[[permno]]
        window = firm_daily[
            (firm_daily["date"] >= win_start) &
            (firm_daily["date"] <= win_end)
        ]

        n_days = len(window)
        if n_days < MIN_DAYS:
            results.append(dict(gvkey=gvkey, event_date=event_date,
                                turnover_pre=np.nan, turnover_days=n_days))
        else:
            results.append(dict(
                gvkey=gvkey,
                event_date=event_date,
                turnover_pre=window["turnover"].mean(),
                turnover_days=n_days,
            ))

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== 12_build_turnover.py ===")

    print("\n[1] Loading analysis sample...")
    sample = load_sample()

    print("\n[2] Connecting to WRDS...")
    db = wrds.Connection(wrds_username="nhwang")

    print("\n[3] Pulling CRSP daily data...")
    daily = pull_crsp_daily(db, sample)
    db.close()

    print("\n[4] Computing pre-event turnover...")
    result = build_turnover(sample, daily)

    n_total = len(result)
    n_valid = result["turnover_pre"].notna().sum()
    print(f"\n  Events processed      : {n_total:,}")
    print(f"  With valid turnover   : {n_valid:,} ({n_valid/n_total:.1%})")
    print(f"\n  turnover_pre summary:")
    print(result["turnover_pre"].describe().to_string())
    print(f"\n  turnover_days summary:")
    print(result["turnover_days"].describe().to_string())

    result.to_parquet(OUT_PATH, index=False)
    print(f"\n  Saved -> {OUT_PATH}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
