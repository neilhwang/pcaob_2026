"""
14_build_post_event_car.py
==========================
Computes post-event cumulative abnormal returns (CARs) for each event in the
analysis sample, using the market model parameters already estimated in
03_build_crsp_sample.py.

PURPOSE
-------
Tests whether the polarization effect on event-window CARs is followed by a
reversal. The prediction from a disagreement/overreaction story is that
events in high-polarization states should show a post-event reversal (negative
post-event CAR) after controlling for the event-window CAR.

Regression run in 05_merge_and_estimate.py:
    CAR[+2,+T] = β₁·competitive_std + β₂·car_m1p1 + controls + FE + ε
    Prediction: β₁ < 0 (post-event reversal in high-polarization states)

Two post-event windows are computed:
    - car_p2p20  : CAR over trading days [+2, +20]
    - car_p2p60  : CAR over trading days [+2, +60]

INPUTS
------
  Data/Processed/analysis_sample.parquet  -- has permno, event_date,
                                             alpha_hat, beta_hat, car_m1p1
  WRDS: crsp.dsf                          -- daily stock returns
  WRDS: crsp.dsi                          -- daily market returns (vwretd)

OUTPUTS
-------
  Data/Processed/post_event_car.parquet   -- one row per (permno, event_date):
      car_p2p20  : market-model CAR over trading days [+2, +20]
      car_p2p60  : market-model CAR over trading days [+2, +60]
      n_days_20  : trading days used for car_p2p20 (data quality check)
      n_days_60  : trading days used for car_p2p60

NOTES
-----
- Same market model parameters (alpha_hat, beta_hat) from 03_build_crsp_sample.py
  are reused — no re-estimation.
- Market return: CRSP value-weighted return including distributions (vwretd),
  consistent with 03_build_crsp_sample.py.
- Delisting returns are NOT merged here (post-event windows are short and
  delisting events near t=0 are already handled in script 03).
- WRDS username: nhwang
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
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "Data", "Processed")

ANALYSIS_SAMPLE = os.path.join(PROCESSED, "analysis_sample.parquet")
OUT_PATH        = os.path.join(PROCESSED, "post_event_car.parquet")

WRDS_USERNAME = "nhwang"

# Post-event windows in trading days (both starting at +2 to skip immediate
# price discovery on t=0 and t=+1)
WINDOW_SHORT_END = 20   # [+2, +20] ~ 1 month
WINDOW_LONG_END  = 60   # [+2, +60] ~ 3 months

# Calendar-day buffer beyond max(event_date) to ensure we capture enough
# trading days for the longest post-event window
CALENDAR_BUFFER_DAYS = 100


# ---------------------------------------------------------------------------
# Helper: batch SQL IN lists
# ---------------------------------------------------------------------------

def batch_list(lst, size=200):
    lst = list(lst)
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def sql_in(values):
    return "(" + ", ".join(str(v) for v in values) + ")"


# ---------------------------------------------------------------------------
# Step 1: Load analysis sample
# ---------------------------------------------------------------------------

def load_sample():
    cols = ["permno", "event_date", "alpha_hat", "beta_hat", "car_m1p1"]
    df = pd.read_parquet(ANALYSIS_SAMPLE, columns=cols)
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["permno"]     = df["permno"].astype(int)
    # Drop events with missing market model parameters (can't compute post-event CAR)
    n_before = len(df)
    df = df.dropna(subset=["alpha_hat", "beta_hat"]).reset_index(drop=True)
    print(f"  Events loaded         : {n_before:,}")
    print(f"  With valid model params: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Pull CRSP daily stock returns
# ---------------------------------------------------------------------------

def pull_crsp_daily(db, permnos, start_date, end_date):
    """
    Pull daily returns for all PERMNOs over the full post-event date range.
    We pull a superset and then filter to per-event windows in Python.
    """
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Unique PERMNOs: {len(permnos):,}")

    records = []
    for i, chunk in enumerate(batch_list(permnos, 200)):
        sql = f"""
            SELECT permno, date, ret
            FROM crsp.dsf
            WHERE permno IN {sql_in(chunk)}
              AND date BETWEEN '{start_date}' AND '{end_date}'
              AND ret IS NOT NULL
        """
        records.append(db.raw_sql(sql))
        if (i + 1) % 5 == 0:
            print(f"    ... fetched {i+1} batches")

    daily = pd.concat(records, ignore_index=True)
    daily["date"]   = pd.to_datetime(daily["date"])
    daily["permno"] = daily["permno"].astype(int)
    print(f"  CRSP daily rows pulled: {len(daily):,}")
    return daily


# ---------------------------------------------------------------------------
# Step 3: Pull CRSP market returns
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Step 4: Build trading-day calendar and offset helper
# ---------------------------------------------------------------------------

def build_trading_days(daily, mkt):
    """
    Construct a sorted series of unique trading days from the CRSP data.
    Uses the union of days in the stock file and the market index.
    """
    all_dates = pd.concat([daily["date"], mkt["date"]]).drop_duplicates().sort_values()
    return all_dates.reset_index(drop=True)


def trading_day_offset(event_date, trading_days, offset):
    """
    Return the trading day that is `offset` days away from event_date.
    Positive = after event, negative = before event.
    Returns None if out of range.
    """
    idx = trading_days.searchsorted(event_date)
    target = idx + offset
    if target < 0 or target >= len(trading_days):
        return None
    return trading_days.iloc[target]


# ---------------------------------------------------------------------------
# Step 5: Compute post-event CARs per event
# ---------------------------------------------------------------------------

def compute_post_event_cars(sample, daily, mkt):
    """
    For each event, compute market-model CAR over [+2, +20] and [+2, +60].

    AR(t) = ret(t) - [alpha_hat + beta_hat * mkt_ret(t)]
    CAR[+2,+T] = Σ_{t=+2}^{+T} AR(t)
    """
    trading_days = build_trading_days(daily, mkt)

    # Index daily by permno for fast lookups
    daily_idx = daily.set_index("permno")

    # Merge market returns into a dict keyed by date for O(1) access
    mkt_dict = mkt.set_index("date")["mkt_ret"].to_dict()

    results = []
    for _, row in sample.iterrows():
        permno     = row["permno"]
        event_date = row["event_date"]
        alpha      = row["alpha_hat"]
        beta       = row["beta_hat"]

        # Resolve trading-day boundaries
        start_20 = trading_day_offset(event_date, trading_days, +2)
        end_20   = trading_day_offset(event_date, trading_days, +WINDOW_SHORT_END)
        start_60 = start_20   # same start
        end_60   = trading_day_offset(event_date, trading_days, +WINDOW_LONG_END)

        base = dict(gvkey=gvkey, permno=permno, event_date=event_date,
                    car_p2p20=np.nan, car_p2p60=np.nan,
                    n_days_20=0, n_days_60=0)

        if start_20 is None or end_20 is None or end_60 is None:
            results.append(base)
            continue

        # Get firm's return series
        if permno not in daily_idx.index:
            results.append(base)
            continue

        firm = daily_idx.loc[[permno]].copy()

        # --- Short window [+2, +20] ---
        w20 = firm[(firm["date"] >= start_20) & (firm["date"] <= end_20)].copy()
        w20["mkt_ret"] = w20["date"].map(mkt_dict)
        w20 = w20.dropna(subset=["ret", "mkt_ret"])
        if len(w20) > 0:
            w20["ar"] = w20["ret"] - (alpha + beta * w20["mkt_ret"])
            base["car_p2p20"] = w20["ar"].sum()
            base["n_days_20"] = len(w20)

        # --- Long window [+2, +60] ---
        w60 = firm[(firm["date"] >= start_60) & (firm["date"] <= end_60)].copy()
        w60["mkt_ret"] = w60["date"].map(mkt_dict)
        w60 = w60.dropna(subset=["ret", "mkt_ret"])
        if len(w60) > 0:
            w60["ar"] = w60["ret"] - (alpha + beta * w60["mkt_ret"])
            base["car_p2p60"] = w60["ar"].sum()
            base["n_days_60"] = len(w60)

        results.append(base)

    out = pd.DataFrame(results)
    # Ensure gvkey is the first column for readability
    cols = ["gvkey", "permno", "event_date", "car_p2p20", "car_p2p60", "n_days_20", "n_days_60"]
    return out[cols]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== 14_build_post_event_car.py ===")

    print("\n[1] Loading analysis sample...")
    sample = load_sample()

    # Determine CRSP pull date range
    # Start: earliest event_date (we need returns from +2 trading days onward,
    #        so the event date itself is a safe lower bound)
    # End:   latest event_date + CALENDAR_BUFFER_DAYS to cover +60 trading days
    crsp_start = sample["event_date"].min().strftime("%Y-%m-%d")
    crsp_end   = (sample["event_date"].max() + pd.Timedelta(days=CALENDAR_BUFFER_DAYS)).strftime("%Y-%m-%d")
    print(f"\n  CRSP pull window: {crsp_start} to {crsp_end}")

    permnos = sample["permno"].unique().tolist()

    print("\n[2] Connecting to WRDS...")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)

    print("\n[3] Pulling CRSP daily stock returns...")
    daily = pull_crsp_daily(db, permnos, crsp_start, crsp_end)

    print("\n[4] Pulling CRSP market returns...")
    mkt = pull_market_returns(db, crsp_start, crsp_end)

    db.close()

    print("\n[5] Computing post-event CARs...")
    result = compute_post_event_cars(sample, daily, mkt)

    # Summary statistics
    n_total = len(result)
    n_20 = result["car_p2p20"].notna().sum()
    n_60 = result["car_p2p60"].notna().sum()
    print(f"\n  Events processed      : {n_total:,}")
    print(f"  Valid car_p2p20       : {n_20:,} ({n_20/n_total:.1%})")
    print(f"  Valid car_p2p60       : {n_60:,} ({n_60/n_total:.1%})")
    print(f"\n  car_p2p20 summary:")
    print(result["car_p2p20"].describe().to_string())
    print(f"\n  car_p2p60 summary:")
    print(result["car_p2p60"].describe().to_string())

    result.to_parquet(OUT_PATH, index=False)
    print(f"\n  Saved -> {OUT_PATH}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
