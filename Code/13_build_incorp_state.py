"""
13_build_incorp_state.py
========================
Pulls state of incorporation from Compustat (comp.company) for each gvkey
in the analysis sample.

Used in 05_merge_and_estimate.py to construct an incorporation-state
polarization placebo. The prediction: headquarters-state polarization should
predict market reactions, but incorporation-state polarization should not
(because incorporation state is chosen for legal/tax reasons, not because of
investor base considerations). If HQ-state polarization survives while
incorporation-state does not, that validates the geographic mechanism.

INPUTS
------
  Data/Processed/analysis_sample.parquet  -- has gvkey
  WRDS: comp.company                      -- gvkey -> incorp (state of incorporation)

OUTPUTS
-------
  Data/Processed/incorp_state.parquet     -- gvkey, incorp (2-letter state code)

NOTES
-----
- comp.company gives the most recent incorporation state per gvkey (time-invariant
  for nearly all firms).
- Approximately 60-70% of US public firms are incorporated in Delaware; the
  incorp_pol measure will therefore have much less cross-sectional variation than
  the HQ-state measure, making it a conservative placebo.
- WRDS username: nhwang
"""

import os
import warnings
import pandas as pd
import wrds

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "Data", "Processed")
ANALYSIS_SAMPLE = os.path.join(PROCESSED, "analysis_sample.parquet")
OUT_PATH = os.path.join(PROCESSED, "incorp_state.parquet")


def batch_list(lst, size=500):
    lst = list(lst)
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def sql_in(values):
    escaped = ["'" + str(v).replace("'", "''") + "'" for v in values]
    return "(" + ", ".join(escaped) + ")"


def main():
    print("=== 13_build_incorp_state.py ===")

    # Load gvkeys from analysis sample
    sample = pd.read_parquet(ANALYSIS_SAMPLE, columns=["gvkey"])
    gvkeys = sample["gvkey"].unique().tolist()
    print(f"  gvkeys to look up: {len(gvkeys):,}")

    print("\n  Connecting to WRDS...")
    db = wrds.Connection(wrds_username="nhwang")

    records = []
    for chunk in batch_list(gvkeys, 500):
        sql = f"""
            SELECT gvkey, incorp
            FROM comp.company
            WHERE gvkey IN {sql_in(chunk)}
              AND incorp IS NOT NULL
              AND incorp <> ''
        """
        records.append(db.raw_sql(sql))

    db.close()

    result = pd.concat(records, ignore_index=True).drop_duplicates(subset="gvkey")

    n_total = len(gvkeys)
    n_matched = result["gvkey"].nunique()
    print(f"\n  gvkeys with incorp: {n_matched:,} / {n_total:,}")
    print(f"\n  Top incorporation states:")
    print(result["incorp"].value_counts().head(10).to_string())

    result.to_parquet(OUT_PATH, index=False)
    print(f"\n  Saved -> {OUT_PATH}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
