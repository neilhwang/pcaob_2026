"""
08_build_affective_polarization.py
===================================
Build a national annual affective-polarization series from ANES feeling
thermometer data, and merge with the state partisan-exposure variable to
produce the inputs for the supplementary affective-polarization validation
test in script 05.

INPUTS:
    Data/Raw/anes_timeseries_cdf_csv_20260205.zip
        (ANES Time Series Cumulative Data File, anes.org)
    Data/Processed/state_partisan_exposure.parquet
        (from 07_build_exposure.py)

OUTPUT:
    Data/Processed/affective_polarization.parquet
        Columns: year, ap_ft          (national AP series, annual)

MEASURE — Affective Polarization (AP):
    Following Iyengar, Sood & Lelkes (2012) and the standard literature
    definition, affective polarization in year t is the mean in-party vs.
    out-party feeling thermometer differential among partisan respondents:

        AP_t = (1/2) * [mean_D(therm_D - therm_R) + mean_R(therm_R - therm_D)]

    where mean_D/mean_R average over Democrat-leaning and Republican-leaning
    respondents respectively (VCF0301 in {1,2,3} for Democrats,
    {5,6,7} for Republicans; pure independents excluded).

    Thermometer variables: VCF0218 (Democratic Party), VCF0224 (Republican Party).
    Valid range: 0-97 (98 = DK, 99 = NA; both treated as missing).

    ANES waves with thermometer data post-2000:
        2000, 2004, 2008, 2012, 2016, 2020, 2024
    (2002 wave did not include party thermometers.)

INTERPOLATION:
    Survey-year values are linearly interpolated to annual frequency and
    assigned to event years 2001-2024. The 2001 value is the linear
    interpolant between 2000 and 2004; the 2002-2003 values likewise;
    etc. Values before the first survey year and after the last are held
    constant (flat extrapolation).

IDENTIFICATION NOTE:
    Because ap_ft is national (same for all states in a year), it is
    absorbed by year fixed effects when entered alone. In script 05 it
    enters only as an interaction: ap_ft × exposure_pres (× ambiguity).
    Identification comes from the differential response of firms in high-
    vs. low-partisan-exposure states as national affective polarization
    rises over time.

REFERENCES:
    Iyengar, S., Sood, G., & Lelkes, Y. (2012). Affect, not ideology:
        A social identity perspective on polarization. Public Opinion
        Quarterly, 76(3), 405-431.
    Druckman, J. N., & Levy, J. (2021). Affective polarization in the
        American public. IPR Working Paper WP-21-27.
"""

import logging
import zipfile
import io
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
ANES_ZIP  = ROOT / "Data/Raw/anes_timeseries_cdf_csv_20260205.zip"
ANES_CSV  = "anes_timeseries_cdf_csv_20260205.csv"
OUT_FILE  = ROOT / "Data/Processed/affective_polarization.parquet"

SAMPLE_START = 2001
SAMPLE_END   = 2024

# ANES variable codes
VAR_YEAR   = "VCF0004"   # Survey year
VAR_DEM_T  = "VCF0218"   # Democratic Party feeling thermometer (0-97)
VAR_REP_T  = "VCF0224"   # Republican Party feeling thermometer (0-97)
VAR_PID    = "VCF0301"   # Party identification 7-point scale
                          # 1-3 = Dem leaning, 4 = Pure Ind, 5-7 = Rep leaning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Step 1: Load ANES ─────────────────────────────────────────────────────────

def load_anes() -> pd.DataFrame:
    log.info("Loading ANES from: %s", ANES_ZIP)
    with zipfile.ZipFile(ANES_ZIP) as z:
        with z.open(ANES_CSV) as f:
            df = pd.read_csv(
                f,
                usecols=[VAR_YEAR, VAR_DEM_T, VAR_REP_T, VAR_PID],
                low_memory=False,
            )
    log.info("ANES rows loaded: %d", len(df))
    return df


# ── Step 2: Clean thermometer variables ───────────────────────────────────────

def clean_thermometers(df: pd.DataFrame) -> pd.DataFrame:
    """Convert to numeric; treat 98 (DK) and 99 (NA) as missing."""
    for col in [VAR_DEM_T, VAR_REP_T]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] >= 98, col] = np.nan

    # Party ID: numeric, 0 = missing
    df[VAR_PID] = pd.to_numeric(df[VAR_PID], errors="coerce")
    df.loc[df[VAR_PID] == 0, VAR_PID] = np.nan

    # Year
    df[VAR_YEAR] = pd.to_numeric(df[VAR_YEAR], errors="coerce")

    # Keep only respondents with valid thermometers and party ID
    df = df.dropna(subset=[VAR_YEAR, VAR_DEM_T, VAR_REP_T, VAR_PID])
    log.info("After cleaning: %d respondents with valid data", len(df))
    return df


# ── Step 3: Compute AP by survey year ─────────────────────────────────────────

def compute_ap_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each survey year compute affective polarization as:
        AP = 0.5 * [mean_D(dem_therm - rep_therm) + mean_R(rep_therm - dem_therm)]

    Pure independents (PID == 4) are excluded; partisans are PID 1-3 (Dem)
    and PID 5-7 (Rep).
    """
    records = []
    for year, grp in df.groupby(VAR_YEAR):
        dems = grp[grp[VAR_PID].isin([1, 2, 3])]
        reps = grp[grp[VAR_PID].isin([5, 6, 7])]

        if len(dems) < 10 or len(reps) < 10:
            log.warning("Year %d: too few partisans (D=%d, R=%d), skipping",
                        year, len(dems), len(reps))
            continue

        # In-party minus out-party for each group
        ap_d = (dems[VAR_DEM_T] - dems[VAR_REP_T]).mean()
        ap_r = (reps[VAR_REP_T] - reps[VAR_DEM_T]).mean()
        ap   = (ap_d + ap_r) / 2.0

        records.append({
            "survey_year": int(year),
            "ap_ft":       ap,
            "n_dem":       len(dems),
            "n_rep":       len(reps),
        })

    ap_df = pd.DataFrame(records).sort_values("survey_year").reset_index(drop=True)
    log.info(
        "AP by survey year:\n%s",
        ap_df[["survey_year", "ap_ft", "n_dem", "n_rep"]].to_string(index=False),
    )
    return ap_df


# ── Step 4: Interpolate to annual frequency ────────────────────────────────────

def interpolate_to_annual(ap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Linearly interpolate between survey years to produce an annual series
    covering SAMPLE_START through SAMPLE_END.

    The grid starts one year before SAMPLE_START so that the nearest
    pre-sample survey observation (e.g., 2000) can serve as the left anchor
    for interpolating the early sample years (2001-2003). That anchor year
    is dropped before writing the output.
    """
    # Build a full annual index, starting one year early to capture any
    # pre-sample survey anchor (e.g., the 2000 ANES wave for 2001-2003).
    grid_start = SAMPLE_START - 1
    annual = pd.DataFrame({"year": range(grid_start, SAMPLE_END + 1)})

    # Merge survey values onto the grid
    survey = ap_df[["survey_year", "ap_ft"]].rename(columns={"survey_year": "year"})
    annual = annual.merge(survey, on="year", how="left")

    # Linear interpolation; flat extrapolation beyond the last observed year
    annual["ap_ft"] = annual["ap_ft"].interpolate(
        method="linear", limit_direction="both"
    )

    # Drop the pre-sample anchor year
    annual = annual[annual["year"] >= SAMPLE_START].reset_index(drop=True)

    log.info(
        "Annual AP series (%d-%d):\n%s",
        SAMPLE_START, SAMPLE_END,
        annual[annual["year"].isin([2001, 2004, 2008, 2012, 2016, 2020, 2024])]
        .to_string(index=False),
    )
    log.info("AP summary:\n%s", annual["ap_ft"].describe().to_string())
    return annual[["year", "ap_ft"]]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== 08_build_affective_polarization.py  start ===")

    for f in [ANES_ZIP]:
        if not f.exists():
            raise FileNotFoundError(f"Missing input: {f}")

    # Steps 1-4: build annual AP series
    raw      = load_anes()
    clean    = clean_thermometers(raw)
    ap_df    = compute_ap_by_year(clean)
    annual   = interpolate_to_annual(ap_df)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    annual.to_parquet(OUT_FILE, index=False)
    log.info("Output written: %s  (%d rows)", OUT_FILE, len(annual))
    log.info("=== done ===")


if __name__ == "__main__":
    main()
