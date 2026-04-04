"""
17_build_short_interest.py
==========================
Pulls short interest data from Compustat (comp.sec_shortint) and constructs
pre-event short interest levels and around-event changes for each auditor-change
event in the analysis sample.

PURPOSE
-------
Two tests of the disagreement channel:

  1. Moderator: Is the polarization effect on |CAR| stronger when pre-event
     short interest is high? High short interest signals active disagreement
     about firm value; if polarization amplifies disagreement, the interaction
     should be positive.

  2. Outcome: Does polarization predict increases in short interest around the
     auditor-change event? If polarized investors form more heterogeneous
     posteriors, the pessimistic tail should increase short positions. This test
     uses the full sample (no splitting) and directly captures disagreement
     via revealed trading behavior.

INPUTS
------
  Data/Processed/analysis_sample.parquet     -- gvkey, event_date, permno
  WRDS: comp.sec_shortint                    -- semi-monthly short interest
  WRDS: crsp.dsf                             -- shrout for normalization

OUTPUTS
-------
  Data/Processed/short_interest.parquet      -- one row per (gvkey, event_date):
      si_ratio_pre   : mean short interest ratio in [-90, -15] days before event
                       (shortintadj / (shrout * 1000))
      si_ratio_post  : short interest ratio from first report in [+15, +90] days
      si_change      : si_ratio_post - si_ratio_pre
      si_pre_raw     : raw pre-event split-adjusted short interest
      si_post_raw    : raw post-event split-adjusted short interest
      n_pre_reports  : number of SI reports in pre-event window

WINDOWS
-------
  Pre-event:  [-90, -15] calendar days before event_date
  Post-event: [+15, +90] calendar days after event_date
  The ±15 day buffer avoids contamination from the event window itself.
  Short interest is reported semi-monthly, so each window captures 3-6 reports.

NOTES
-----
- comp.sec_shortint reports short interest twice per month (mid-month and
  end-of-month settlement dates).
- shortintadj is the split-adjusted share count; we normalize by CRSP shrout
  (thousands of shares) to get the short interest ratio.
- WRDS username: nhwang
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import wrds

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
PROC     = ROOT / "Data" / "Processed"
ANALYSIS_SAMPLE = PROC / "analysis_sample.parquet"
OUT_PATH = PROC / "short_interest.parquet"

WRDS_USERNAME = "nhwang"

# Windows (calendar days relative to event_date)
PRE_START  = -90
PRE_END    = -15
POST_START = +15
POST_END   = +90

MIN_PRE_REPORTS = 2   # require at least 2 SI reports in pre-event window


# ── Helpers ──────────────────────────────────────────────────────────────────

def batch_list(lst, size=500):
    lst = list(lst)
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def sql_in_str(values):
    escaped = ["'" + str(v).replace("'", "''") + "'" for v in values]
    return "(" + ", ".join(escaped) + ")"


def sql_in_int(values):
    return "(" + ", ".join(str(v) for v in values) + ")"


# ── Step 1: Load analysis sample ────────────────────────────────────────────

def load_sample():
    cols = ["gvkey", "permno", "event_date"]
    df = pd.read_parquet(ANALYSIS_SAMPLE, columns=cols)
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["permno"] = df["permno"].astype(int)
    df = df.drop_duplicates(subset=["gvkey", "event_date"]).reset_index(drop=True)
    print(f"  Events: {len(df):,}  ({df['gvkey'].nunique()} firms)")
    return df


# ── Step 2: Pull short interest ─────────────────────────────────────────────

def pull_short_interest(db, gvkeys, date_start, date_end):
    """Pull comp.sec_shortint for sample gvkeys over the full date range."""
    print(f"  Date range: {date_start} → {date_end}")
    records = []
    for chunk in batch_list(gvkeys, 500):
        sql = f"""
            SELECT gvkey, datadate, shortintadj
            FROM comp.sec_shortint
            WHERE gvkey IN {sql_in_str(chunk)}
              AND datadate BETWEEN '{date_start}' AND '{date_end}'
              AND shortintadj IS NOT NULL
              AND shortintadj > 0
        """
        records.append(db.raw_sql(sql))
    si = pd.concat(records, ignore_index=True)
    si["datadate"] = pd.to_datetime(si["datadate"])
    # Keep one observation per gvkey × datadate (in case of multiple iid)
    si = si.groupby(["gvkey", "datadate"], as_index=False)["shortintadj"].sum()
    print(f"  Short interest rows: {len(si):,}  ({si['gvkey'].nunique()} firms)")
    return si


# ── Step 3: Pull CRSP shrout for normalization ──────────────────────────────

def pull_shrout(db, permnos, date_start, date_end):
    """
    Pull monthly shares outstanding from CRSP for normalization.
    Use end-of-month shrout from crsp.msf (monthly stock file) for efficiency.
    """
    print(f"  Pulling CRSP monthly shrout...")
    records = []
    for chunk in batch_list(permnos, 200):
        sql = f"""
            SELECT permno, date, shrout
            FROM crsp.msf
            WHERE permno IN {sql_in_int(chunk)}
              AND date BETWEEN '{date_start}' AND '{date_end}'
              AND shrout IS NOT NULL
              AND shrout > 0
        """
        records.append(db.raw_sql(sql))
    shr = pd.concat(records, ignore_index=True)
    shr["date"]   = pd.to_datetime(shr["date"])
    shr["permno"] = shr["permno"].astype(int)
    # shrout in CRSP is in thousands of shares
    print(f"  CRSP monthly rows: {len(shr):,}")
    return shr


# ── Step 4: Compute pre-event and post-event short interest ─────────────────

def build_si_measures(sample, si, shrout):
    """
    For each event, compute:
      - Pre-event mean SI ratio ([-90, -15] days)
      - Post-event SI ratio (first report in [+15, +90] days)
      - Change = post - pre
    """
    # Build a gvkey → permno map for shrout lookup
    permno_map = sample.set_index("gvkey")["permno"].to_dict()

    # Index short interest by gvkey for fast lookup
    si_by_gvkey = si.groupby("gvkey")

    # Index shrout by permno
    shrout_by_permno = shrout.groupby("permno")

    results = []
    for _, row in sample.iterrows():
        gvkey      = row["gvkey"]
        event_date = row["event_date"]
        permno     = row["permno"]

        base = dict(gvkey=gvkey, event_date=event_date,
                    si_ratio_pre=np.nan, si_ratio_post=np.nan,
                    si_change=np.nan, si_pre_raw=np.nan,
                    si_post_raw=np.nan, n_pre_reports=0)

        # Get firm's short interest series
        if gvkey not in si_by_gvkey.groups:
            results.append(base)
            continue
        firm_si = si_by_gvkey.get_group(gvkey)

        # Get firm's shrout series
        if permno not in shrout_by_permno.groups:
            results.append(base)
            continue
        firm_shr = shrout_by_permno.get_group(permno)

        # Pre-event window
        pre_start = event_date + pd.Timedelta(days=PRE_START)
        pre_end   = event_date + pd.Timedelta(days=PRE_END)
        pre_si = firm_si[
            (firm_si["datadate"] >= pre_start) &
            (firm_si["datadate"] <= pre_end)
        ]

        if len(pre_si) < MIN_PRE_REPORTS:
            results.append(base)
            continue

        # Get closest shrout to the midpoint of pre-event window
        pre_mid = event_date + pd.Timedelta(days=(PRE_START + PRE_END) // 2)
        shr_diffs = (firm_shr["date"] - pre_mid).abs()
        closest_shr = firm_shr.loc[shr_diffs.idxmin(), "shrout"]
        # shrout is in thousands; shortintadj is in shares
        shares_out = closest_shr * 1000.0

        if shares_out <= 0:
            results.append(base)
            continue

        pre_si_mean = pre_si["shortintadj"].mean()
        pre_ratio   = pre_si_mean / shares_out

        # Post-event window
        post_start = event_date + pd.Timedelta(days=POST_START)
        post_end   = event_date + pd.Timedelta(days=POST_END)
        post_si = firm_si[
            (firm_si["datadate"] >= post_start) &
            (firm_si["datadate"] <= post_end)
        ]

        if len(post_si) == 0:
            # Can still report pre-event level
            base["si_ratio_pre"]  = pre_ratio
            base["si_pre_raw"]    = pre_si_mean
            base["n_pre_reports"] = len(pre_si)
            results.append(base)
            continue

        # Use the first post-event report for the change computation
        post_first = post_si.sort_values("datadate").iloc[0]
        post_raw   = post_first["shortintadj"]

        # Get shrout closest to post-event date
        post_shr_diffs = (firm_shr["date"] - post_first["datadate"]).abs()
        post_closest   = firm_shr.loc[post_shr_diffs.idxmin(), "shrout"]
        post_shares    = post_closest * 1000.0

        if post_shares <= 0:
            post_ratio = np.nan
        else:
            post_ratio = post_raw / post_shares

        base["si_ratio_pre"]  = pre_ratio
        base["si_ratio_post"] = post_ratio
        base["si_change"]     = post_ratio - pre_ratio if not np.isnan(post_ratio) else np.nan
        base["si_pre_raw"]    = pre_si_mean
        base["si_post_raw"]   = post_raw
        base["n_pre_reports"] = len(pre_si)
        results.append(base)

    return pd.DataFrame(results)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== 17_build_short_interest.py ===")

    print("\n[1] Loading analysis sample...")
    sample = load_sample()

    gvkeys  = sample["gvkey"].unique().tolist()
    permnos = sample["permno"].unique().tolist()

    # Date range: earliest event - 90 days to latest event + 90 days
    date_start = (sample["event_date"].min() - pd.Timedelta(days=100)).strftime("%Y-%m-%d")
    date_end   = (sample["event_date"].max() + pd.Timedelta(days=100)).strftime("%Y-%m-%d")

    print("\n[2] Connecting to WRDS...")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)

    print("\n[3] Pulling short interest (comp.sec_shortint)...")
    si = pull_short_interest(db, gvkeys, date_start, date_end)

    print("\n[4] Pulling CRSP monthly shrout...")
    shrout = pull_shrout(db, permnos, date_start, date_end)

    db.close()

    print("\n[5] Computing pre/post event short interest measures...")
    result = build_si_measures(sample, si, shrout)

    # Summary
    n_total = len(result)
    n_pre   = result["si_ratio_pre"].notna().sum()
    n_post  = result["si_ratio_post"].notna().sum()
    n_chg   = result["si_change"].notna().sum()
    print(f"\n  Events processed     : {n_total:,}")
    print(f"  With pre-event SI    : {n_pre:,} ({n_pre/n_total:.1%})")
    print(f"  With post-event SI   : {n_post:,} ({n_post/n_total:.1%})")
    print(f"  With SI change       : {n_chg:,} ({n_chg/n_total:.1%})")

    if n_pre > 0:
        print(f"\n  si_ratio_pre summary:")
        print(result["si_ratio_pre"].describe().to_string())
    if n_chg > 0:
        print(f"\n  si_change summary:")
        print(result["si_change"].describe().to_string())

    result.to_parquet(OUT_PATH, index=False)
    print(f"\n  Saved: {OUT_PATH}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
