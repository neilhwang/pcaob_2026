"""
10_build_county_polarization.py
================================
Build a county-level presidential competitiveness measure and merge it to
the analysis sample via Compustat zip codes and the HUD ZIP-COUNTY crosswalk.

PURPOSE:
    Robustness column 9 in Table 5: county-level polarization check.
    Uses the same competitiveness formula as script 02b (state-level) but
    assigns each firm to its headquarters county rather than headquarters state,
    providing a finer-grained geographic alternative that is not driven by
    the state-level aggregation choice.

INPUTS:
    Data/Raw/countypres_2000-2024.tab          MIT Election Lab county returns
    Data/Raw/ZIP_COUNTY_122023.xlsx            HUD ZIP-COUNTY crosswalk (Q4 2023)
    Data/Processed/compustat_zip.parquet       gvkey -> zip5 (from script 04b)
    Data/Processed/analysis_sample.parquet     678-event sample

OUTPUT:
    Data/Processed/pol_county.parquet
        county_fips (str, 5-digit), year (int), county_comp (float),
        county_comp_std (float, standardized within analysis sample events)

    Data/Processed/analysis_sample_county.parquet
        analysis_sample with county_comp_std appended; events without a
        county match are retained with NaN (dropped only in regression).

PIPELINE POSITION:
    Runs after 04b (zip codes). Output consumed by 05_merge_and_estimate.py
    to add robustness column 9.
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ── Step 1: Parse HUD ZIP-COUNTY crosswalk ────────────────────────────────────
# The crosswalk maps zip codes to county FIPS with address-share weights.
# A zip can span multiple counties; we assign each zip to the county with
# the highest residential address ratio (RES_RATIO).

print("Step 1: Parse HUD ZIP-COUNTY crosswalk")
xwalk_raw = pd.read_excel(
    ROOT / "Data/Raw/ZIP_COUNTY_122023.xlsx",
    dtype={"ZIP": str, "COUNTY": str}
)
print(f"  Raw crosswalk rows: {len(xwalk_raw)}")
print(f"  Columns: {xwalk_raw.columns.tolist()}")

# Standardize column names to lowercase
xwalk_raw.columns = xwalk_raw.columns.str.lower()

# Confirm expected columns exist
assert "zip" in xwalk_raw.columns, f"No 'zip' column; found: {xwalk_raw.columns.tolist()}"
assert "county" in xwalk_raw.columns, f"No 'county' column"

# Identify the residential ratio column (HUD names it res_ratio or similar)
res_col = next((c for c in xwalk_raw.columns if "res" in c and "ratio" in c), None)
if res_col is None:
    # Fall back to tot_ratio
    res_col = next((c for c in xwalk_raw.columns if "tot" in c and "ratio" in c), None)
print(f"  Using address-share column: {res_col}")

# Zero-pad to standard widths
xwalk_raw["zip"]    = xwalk_raw["zip"].str.strip().str.zfill(5)
xwalk_raw["county"] = xwalk_raw["county"].str.strip().str.zfill(5)

# Keep dominant county per zip (highest residential share)
xwalk = (
    xwalk_raw
    .sort_values(res_col, ascending=False)
    .drop_duplicates(subset="zip", keep="first")
    [["zip", "county"]]
    .rename(columns={"county": "county_fips"})
    .reset_index(drop=True)
)
print(f"  Unique zips after dominant-county assignment: {len(xwalk)}")

# ── Step 2: Build county-level competitiveness from MIT Election Lab ──────────
# Same logic as script 02b but at county level.
# Forward-fill: election year e -> calendar years e+1 through e+4.

print("\nStep 2: Build county-level competitiveness")
pres_raw = pd.read_csv(
    ROOT / "Data/Raw/countypres_2000-2024.tab",
    sep="\t",
    usecols=["year", "county_fips", "party", "candidatevotes", "totalvotes", "mode"],
    dtype={"county_fips": str}
)

# Keep only TOTAL mode rows (not split by early/election day)
pres = pres_raw[pres_raw["mode"] == "TOTAL"].copy()

# Keep presidential elections only (even years divisible by 4)
pres = pres[pres["year"] % 4 == 0].copy()

# Keep elections 2000-2020 (forward-fill covers through 2024; sample ends 2023)
pres = pres[pres["year"].between(2000, 2020)].copy()

# Zero-pad county FIPS
pres["county_fips"] = pres["county_fips"].astype(str).str.strip().str.zfill(5)

# Aggregate to county-year-party
party_votes = (
    pres
    .assign(party_clean=lambda d: d["party"].str.upper().str.strip())
    .groupby(["year", "county_fips", "party_clean"])["candidatevotes"]
    .sum()
    .reset_index()
)

# Total two-party votes per county-year
dem = party_votes[party_votes["party_clean"] == "DEMOCRAT"].rename(
    columns={"candidatevotes": "dem_votes"}
)
rep = party_votes[party_votes["party_clean"] == "REPUBLICAN"].rename(
    columns={"candidatevotes": "rep_votes"}
)

county_year = dem[["year", "county_fips", "dem_votes"]].merge(
    rep[["year", "county_fips", "rep_votes"]],
    on=["year", "county_fips"],
    how="inner"
)
county_year["two_party_total"] = county_year["dem_votes"] + county_year["rep_votes"]
county_year = county_year[county_year["two_party_total"] > 0].copy()

county_year["d_share"] = county_year["dem_votes"] / county_year["two_party_total"]
county_year["county_comp"] = 1 - np.abs(2 * county_year["d_share"] - 1)

print(f"  County-year observations (election years): {len(county_year)}")
print(f"  Unique counties: {county_year['county_fips'].nunique()}")
print(f"  Years: {sorted(county_year['year'].unique())}")

# Forward-fill: election year e -> calendar years e+1 through e+4
records = []
for _, row in county_year.iterrows():
    for offset in range(1, 5):
        records.append({
            "county_fips": row["county_fips"],
            "year": int(row["year"]) + offset,
            "county_comp": row["county_comp"],
            "election_year": int(row["year"]),
        })

pol_county = pd.DataFrame(records)
pol_county = pol_county[pol_county["year"].between(2001, 2023)].copy()

print(f"  Forward-filled rows (2001-2023): {len(pol_county)}")

# Standardize within the set of county-years that appear in the analysis sample
# (standardization deferred to merge step below so it uses only matched events)

# ── Step 3: Merge to analysis sample ─────────────────────────────────────────

print("\nStep 3: Merge to analysis sample")
sample    = pd.read_parquet(ROOT / "Data/Processed/analysis_sample.parquet")
zip_df    = pd.read_parquet(ROOT / "Data/Processed/compustat_zip.parquet")

# gvkey -> zip5
sample = sample.merge(
    zip_df[["gvkey", "zip5"]],
    on="gvkey",
    how="left"
)
n_no_zip = sample["zip5"].isna().sum()
print(f"  Events without zip: {n_no_zip}")

# zip5 -> county_fips
sample = sample.merge(xwalk, left_on="zip5", right_on="zip", how="left")
n_no_county = sample["county_fips"].isna().sum()
print(f"  Events without county match: {n_no_county}")

# county_fips + event_year -> county_comp
sample = sample.merge(
    pol_county[["county_fips", "year", "county_comp"]],
    left_on=["county_fips", "event_year"],
    right_on=["county_fips", "year"],
    how="left"
)
n_no_comp = sample["county_comp"].isna().sum()
print(f"  Events without county competitiveness: {n_no_comp}")
print(f"  Events with county competitiveness: {sample['county_comp'].notna().sum()} / {len(sample)}")

# Standardize within matched events
matched = sample["county_comp"].notna()
mu  = sample.loc[matched, "county_comp"].mean()
sig = sample.loc[matched, "county_comp"].std()
sample["county_comp_std"] = (sample["county_comp"] - mu) / sig

print(f"\n  county_comp mean={mu:.4f}, sd={sig:.4f}")
print(f"  county_comp_std: mean={sample.loc[matched,'county_comp_std'].mean():.4f}, "
      f"sd={sample.loc[matched,'county_comp_std'].std():.4f}")

# ── Write outputs ─────────────────────────────────────────────────────────────

out_pol = ROOT / "Data/Processed/pol_county.parquet"
pol_county.to_parquet(out_pol, index=False)
print(f"\nWritten: {out_pol}")

out_sample = ROOT / "Data/Processed/analysis_sample_county.parquet"
sample.to_parquet(out_sample, index=False)
print(f"Written: {out_sample}")

print("\nDone. Columns added to analysis_sample_county:")
print("  zip5, county_fips, county_comp, county_comp_std")
print(f"  Usable for regression: {matched.sum()} events")
