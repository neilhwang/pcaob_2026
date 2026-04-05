"""
21_build_post_event_uncertainty.py
====================================
Build a battery of post-event uncertainty proxies to validate the filing
informativeness measures and test the interpretation channel mechanism.

MEASURES
--------
For each event, compute pre- and post-event values and the change, using
pre-event window [-20, -2] and post-event window [+2, +20] trading days.

  1. vol_change          : Δ SD of abnormal daily returns
                            (already computed in script 20; re-computed here)
  2. parkinson_change    : Δ Parkinson (1980) high-low volatility estimator,
                            σ² = (1/(4 ln 2)) × mean(ln(high/low)²)
  3. amihud_change       : Δ Amihud (2002) illiquidity ratio,
                            |ret| / dollar_volume
  4. spread_change       : Δ Corwin-Schultz bid-ask spread (when bid/ask
                            are populated), otherwise NaN
  5. abvol_change        : Δ mean log(volume + 1) (abnormal volume persistence)
  6. roll_change         : Δ Roll (1984) implied spread,
                            s = 2√(-cov(Δp_t, Δp_{t-1})) when cov<0, else 0

INPUTS
------
  Data/Processed/analysis_sample.parquet     -- event list
  Data/Processed/crsp_event_window.parquet  -- alpha_hat, beta_hat
  WRDS: crsp.dsf, crsp.dsi

OUTPUT
------
  Data/Processed/post_event_uncertainty.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd
import wrds

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "Data" / "Processed"
SAMPLE_FILE = PROC / "analysis_sample.parquet"
CRSP_FILE   = PROC / "crsp_event_window.parquet"
OUT_PATH    = PROC / "post_event_uncertainty.parquet"

WRDS_USERNAME = "nhwang"

PRE_START, PRE_END   = -20, -2
POST_START, POST_END = 2, 20


def batch_list(lst, size=300):
    lst = list(lst)
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def parkinson_var(high, low):
    """Parkinson (1980) volatility estimator: (1/(4 ln 2)) × E[ln(H/L)²]."""
    ratio = high / low
    mask = (ratio > 0) & np.isfinite(ratio)
    if mask.sum() < 2:
        return np.nan
    lr = np.log(ratio[mask])
    return float(np.mean(lr ** 2) / (4.0 * np.log(2.0)))


def amihud_ratio(abs_ret, dollar_vol):
    """Amihud illiquidity: mean of |ret| / dollar_volume."""
    mask = (dollar_vol > 0) & np.isfinite(abs_ret)
    if mask.sum() < 2:
        return np.nan
    return float(np.mean(abs_ret[mask] / dollar_vol[mask]))


def bid_ask_spread(bid, ask):
    """Mean relative bid-ask spread = (ask - bid) / midpoint."""
    midpoint = (bid + ask) / 2.0
    mask = (bid > 0) & (ask > bid) & (midpoint > 0)
    if mask.sum() < 2:
        return np.nan
    return float(np.mean((ask[mask] - bid[mask]) / midpoint[mask]))


def roll_spread(returns):
    """Roll (1984) implied spread: 2·√(-cov(r_t, r_{t-1})) if cov < 0."""
    r = returns[np.isfinite(returns)]
    if len(r) < 5:
        return np.nan
    r_lag = r[:-1]
    r_cur = r[1:]
    if len(r_cur) < 2:
        return np.nan
    cov = float(np.cov(r_cur, r_lag, ddof=1)[0, 1])
    if cov >= 0:
        return 0.0
    return 2.0 * np.sqrt(-cov)


def main():
    print("=== 21_build_post_event_uncertainty.py ===")

    # Load sample
    sample = pd.read_parquet(SAMPLE_FILE, columns=["permno", "event_date"])
    sample["event_date"] = pd.to_datetime(sample["event_date"])
    sample["permno"] = sample["permno"].astype(int)
    sample = sample.drop_duplicates(subset=["permno", "event_date"]).reset_index(drop=True)
    print(f"  Events: {len(sample):,}")

    # Alpha/beta for abnormal return computation
    params = pd.read_parquet(
        CRSP_FILE, columns=["permno", "event_date", "alpha_hat", "beta_hat"]
    )
    params["event_date"] = pd.to_datetime(params["event_date"])
    params["permno"] = params["permno"].astype(int)
    sample = sample.merge(params, on=["permno", "event_date"], how="left")

    # Pull CRSP daily data (all needed fields in one query)
    print("  Connecting to WRDS...")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)

    permnos = sample["permno"].unique().tolist()
    date_start = (sample["event_date"].min() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    date_end   = (sample["event_date"].max() + pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    print(f"  Date range: {date_start} to {date_end}")

    records = []
    for chunk in batch_list(permnos, 300):
        permno_str = ", ".join(str(p) for p in chunk)
        sql = f"""
            SELECT permno, date, ret, bidlo, askhi, prc, vol, bid, ask
            FROM crsp.dsf
            WHERE permno IN ({permno_str})
              AND date BETWEEN '{date_start}' AND '{date_end}'
        """
        records.append(db.raw_sql(sql))
    dsf = pd.concat(records, ignore_index=True)
    dsf["date"] = pd.to_datetime(dsf["date"])
    dsf["permno"] = dsf["permno"].astype(int)
    print(f"  CRSP daily rows: {len(dsf):,}")

    # Market returns
    mkt = db.raw_sql(f"""
        SELECT date, vwretd FROM crsp.dsi
        WHERE date BETWEEN '{date_start}' AND '{date_end}'
    """)
    mkt["date"] = pd.to_datetime(mkt["date"])

    db.close()

    dsf = dsf.merge(mkt, on="date", how="left")

    # Pre-compute helper columns
    dsf["abs_prc"]     = dsf["prc"].abs()
    dsf["dollar_vol"]  = dsf["abs_prc"] * dsf["vol"]
    dsf["log_vol"]     = np.log1p(dsf["vol"].fillna(0))

    # Compute all measures for each event
    print("  Computing uncertainty measures...")
    results = []
    for _, row in sample.iterrows():
        permno = row["permno"]
        event_date = row["event_date"]
        alpha = row["alpha_hat"]
        beta  = row["beta_hat"]

        base = dict(
            permno=permno, event_date=event_date,
            pre_vol=np.nan, post_vol=np.nan, vol_change=np.nan,
            pre_parkinson=np.nan, post_parkinson=np.nan, parkinson_change=np.nan,
            pre_amihud=np.nan, post_amihud=np.nan, amihud_change=np.nan,
            pre_spread=np.nan, post_spread=np.nan, spread_change=np.nan,
            pre_abvol=np.nan, post_abvol=np.nan, abvol_change=np.nan,
            pre_roll=np.nan, post_roll=np.nan, roll_change=np.nan,
            n_pre=0, n_post=0,
        )

        firm = dsf[dsf["permno"] == permno].sort_values("date").reset_index(drop=True)
        if firm.empty or pd.isna(alpha) or pd.isna(beta):
            results.append(base)
            continue

        # Event day = first trading day on or after event_date
        mask_after = firm["date"] >= event_date
        if not mask_after.any():
            results.append(base)
            continue
        event_pos = mask_after.idxmax()
        firm["rel_day"] = np.arange(len(firm)) - event_pos

        # Abnormal returns
        firm["abn_ret"] = firm["ret"] - (alpha + beta * firm["vwretd"])

        pre = firm[firm["rel_day"].between(PRE_START, PRE_END)]
        post = firm[firm["rel_day"].between(POST_START, POST_END)]
        base["n_pre"] = len(pre)
        base["n_post"] = len(post)

        if len(pre) >= 5:
            # 1. Abnormal return volatility
            base["pre_vol"] = float(pre["abn_ret"].std())
            # 2. Parkinson high-low volatility
            base["pre_parkinson"] = parkinson_var(pre["askhi"].values, pre["bidlo"].values)
            # 3. Amihud illiquidity
            base["pre_amihud"] = amihud_ratio(pre["abn_ret"].abs().values, pre["dollar_vol"].values)
            # 4. Bid-ask spread
            base["pre_spread"] = bid_ask_spread(pre["bid"].values, pre["ask"].values)
            # 5. Log volume mean
            base["pre_abvol"] = float(pre["log_vol"].mean())
            # 6. Roll implied spread
            base["pre_roll"] = roll_spread(pre["ret"].values)

        if len(post) >= 5:
            base["post_vol"]       = float(post["abn_ret"].std())
            base["post_parkinson"] = parkinson_var(post["askhi"].values, post["bidlo"].values)
            base["post_amihud"]    = amihud_ratio(post["abn_ret"].abs().values, post["dollar_vol"].values)
            base["post_spread"]    = bid_ask_spread(post["bid"].values, post["ask"].values)
            base["post_abvol"]     = float(post["log_vol"].mean())
            base["post_roll"]      = roll_spread(post["ret"].values)

        # Changes
        for m in ["vol", "parkinson", "amihud", "spread", "abvol", "roll"]:
            pre_v = base[f"pre_{m}"]
            post_v = base[f"post_{m}"]
            if pd.notna(pre_v) and pd.notna(post_v):
                base[f"{m}_change"] = post_v - pre_v

        results.append(base)

    out = pd.DataFrame(results)

    print(f"\n  Events processed: {len(out):,}")
    for m in ["vol", "parkinson", "amihud", "spread", "abvol", "roll"]:
        n_valid = out[f"{m}_change"].notna().sum()
        if n_valid > 0:
            mean_val = out[f"{m}_change"].mean()
            med_val = out[f"{m}_change"].median()
            print(f"  {m:10s}: N={n_valid:4d}  mean change={mean_val:12.6f}  median change={med_val:12.6f}")
        else:
            print(f"  {m:10s}: N=0 (no valid values)")

    out.to_parquet(OUT_PATH, index=False)
    print(f"\n  Saved: {OUT_PATH}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
