"""
04b_build_zip_codes.py
======================
Pull headquarters zip codes for the 456 unique gvkeys in the analysis sample
from the Compustat North America `company` table on WRDS.

INPUT:  Data/Processed/analysis_sample.parquet  (gvkey list)
        WRDS connection (wrds library, interactive login)

OUTPUT: Data/Processed/compustat_zip.parquet
            gvkey  (str, zero-padded to 6 digits)
            zip    (str, raw Compustat value — may be 5- or 9-digit)
            zip5   (str, first 5 characters, standardized)

NOTES:
    - The Compustat `company` table stores the most-recent headquarters zip.
      For firms that moved, this is a small bias; it affects very few firms
      and is standard in the literature.
    - zip values are occasionally missing; affected events will be dropped
      from the county-level robustness column only.
"""

from pathlib import Path
import pandas as pd
import wrds

ROOT = Path(__file__).resolve().parent.parent

# ── Load gvkey list ────────────────────────────────────────────────────────────

sample = pd.read_parquet(ROOT / "Data/Processed/analysis_sample.parquet")
gvkeys = sorted(sample["gvkey"].dropna().unique().tolist())
print(f"{len(gvkeys)} unique gvkeys to look up")

# Format as SQL tuple
gvkey_tuple = "(" + ", ".join(f"'{g}'" for g in gvkeys) + ")"

# ── Connect to WRDS and query ──────────────────────────────────────────────────

db = wrds.Connection()

query = f"""
    SELECT gvkey, addzip, county, city, state
    FROM comp.company
    WHERE gvkey IN {gvkey_tuple}
"""

print("Querying Compustat company table...")
result = db.raw_sql(query)
db.close()

print(f"Returned {len(result)} rows")
print(result.head())

# ── Clean ──────────────────────────────────────────────────────────────────────

result["gvkey"]  = result["gvkey"].astype(str).str.zfill(6)
result["addzip"] = result["addzip"].astype(str).str.strip()
result["zip5"]   = result["addzip"].str[:5]

# Flag missing or non-numeric (some addzip values are non-US codes)
import re
valid = result["zip5"].apply(lambda z: bool(re.match(r"^\d{5}$", z)))
n_missing = (~valid).sum()
print(f"Missing or non-numeric zip5: {n_missing} / {len(result)}")
print(result[~valid][["gvkey", "addzip", "county", "state"]].head(10))

# ── Write ──────────────────────────────────────────────────────────────────────

out = ROOT / "Data/Processed/compustat_zip.parquet"
result.to_parquet(out, index=False)
print(f"Written: {out}")
