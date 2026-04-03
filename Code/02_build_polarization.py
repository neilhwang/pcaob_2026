"""
02_build_polarization.py
========================
Build the Esteban & Ray (1994) political polarization measure from MIT Election
Lab congressional district election returns.

INPUT:  Data/Raw/mit_house_elections.tab
        (MIT Election Lab, "U.S. House 1976–2022", Harvard Dataverse DOI:10.7910/DVN/IG0UN2)
        Download instructions: see README block below.
OUTPUT: Data/Processed/polarization_state_year.parquet
        Columns: state_fips, year, pol_er_alpha1, pol_er_alpha08,
                 pol_er_alpha12, n_districts, n_contested

MEASURE:
    Esteban & Ray (1994) polarization for two-party competition:

        P^ER_{s,t} = Σ_d w_d · [s_D · s_R · (s_D^α + s_R^α)]_d

    where:
        s_D, s_R  = Democratic / Republican share of two-party vote in district d
        α         = sensitivity parameter (baseline 1.0; robustness 0.8, 1.2)
        w_d       = district weight = (total two-party votes in d) /
                                      (total two-party votes in state s)

    Aggregated to state × Congress (2-year cycle), then forward-filled to
    annual frequency so it can be merged with firm-year data.

    Theoretical justification: for α=1 and two parties, P^ER = s_D · s_R,
    maximized at 0.25 when s_D = s_R = 0.5. See Esteban & Ray (1994) Theorem 3.

HOW TO DOWNLOAD THE RAW DATA:
    1. Go to: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2
    2. Download the file "1976-2022-house.tab"
    3. Save it as: Data/Raw/mit_house_elections.tab
    The file is ~10 MB and requires a free Harvard Dataverse account.

MATCH TO FIRMS (done in merge script):
    Firms matched to state via Compustat `state` variable (HQ state).
    State FIPS codes bridged to Compustat 2-letter state abbreviations via
    standard FIPS crosswalk (built in this script).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
ALPHA_BASELINE  = 1.0
ALPHA_ROBUST    = [0.8, 1.2]   # robustness specifications
RAW_FILE   = Path(__file__).resolve().parent.parent / "Data/Raw/mit_house_elections.tab"
OUT_FILE   = Path(__file__).resolve().parent.parent / "Data/Processed/polarization_state_year.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── FIPS crosswalk (state_fips ↔ 2-letter abbreviation) ──────────────────────
# Used later when merging with Compustat state variable.
FIPS_TO_ABBR = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT",
    10: "DE", 11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL",
    18: "IN", 19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD",
    25: "MA", 26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE",
    32: "NV", 33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND",
    39: "OH", 40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD",
    47: "TN", 48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV",
    55: "WI", 56: "WY",
}


# ── Step 1: Load and validate raw data ────────────────────────────────────────

def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"\nRaw data not found: {path}\n"
            "Download '1976-2022-house.tab' from:\n"
            "  https://dataverse.harvard.edu/dataset.xhtml"
            "?persistentId=doi:10.7910/DVN/IG0UN2\n"
            "and save to Data/Raw/mit_house_elections.tab"
        )
    df = pd.read_csv(path, sep=",", low_memory=False)
    log.info("Loaded raw data: %d rows, %d cols", len(df), df.shape[1])
    log.info("Columns: %s", list(df.columns))
    return df


# ── Step 2: Clean and filter ──────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep general election, non-special, non-runoff results for D and R candidates.
    Drop uncontested races (only one major party ran) — these contribute zero
    information about polarization and would bias the measure downward.
    """
    # Standardise column names to lowercase
    df.columns = df.columns.str.lower().str.strip()

    # Keep general elections only
    df = df[df["stage"].str.upper() == "GEN"].copy()

    # Drop special elections (off-cycle, unusual circumstances)
    if "special" in df.columns:
        df = df[df["special"] == False]

    # Drop runoffs
    if "runoff" in df.columns:
        df = df[df["runoff"] == False]

    # Drop Louisiana — uses a jungle primary system where the "general"
    # election may be a runoff between two candidates of the same party,
    # making the two-party ER measure unreliable. Louisiana's single
    # at-large-equivalent district also produces extreme ER values.
    # Louisiana state_fips = 22.
    df = df[df["state_fips"] != 22]
    log.info("Dropped Louisiana (jungle primary). Remaining rows: %d", len(df))

    # Drop write-ins
    if "writein" in df.columns:
        df = df[df["writein"] == False]

    # Classify party: Democrat, Republican, Other
    df["party_clean"] = np.where(
        df["party"].str.contains("DEMOCRAT", case=False, na=False), "D",
        np.where(
            df["party"].str.contains("REPUBLICAN", case=False, na=False), "R",
            "O"
        )
    )

    # Keep only D and R rows for two-party share calculation
    df = df[df["party_clean"].isin(["D", "R"])].copy()

    # Coerce votes to numeric
    df["candidatevotes"] = pd.to_numeric(df["candidatevotes"], errors="coerce").fillna(0)
    df["totalvotes"]     = pd.to_numeric(df["totalvotes"],     errors="coerce").fillna(0)

    log.info("After cleaning: %d candidate-district-year rows", len(df))
    return df


# ── Step 3: Compute two-party vote shares by district ─────────────────────────

def compute_district_shares(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each district × year, compute Democratic and Republican shares of the
    two-party vote. Districts where either party is absent are dropped (these
    are uncontested, providing no polarization signal).
    """
    # Sum votes by district × year × party (handles fusion tickets etc.)
    agg = (
        df.groupby(["year", "state_fips", "district", "party_clean"], as_index=False)
        ["candidatevotes"].sum()
    )

    # Pivot to wide: one row per district × year
    wide = agg.pivot_table(
        index=["year", "state_fips", "district"],
        columns="party_clean",
        values="candidatevotes",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    wide.columns.name = None

    # Ensure both D and R columns exist
    for col in ["D", "R"]:
        if col not in wide.columns:
            wide[col] = 0.0

    # Two-party total
    wide["two_party_total"] = wide["D"] + wide["R"]

    # Drop uncontested districts (one party got zero votes)
    contested = wide[(wide["D"] > 0) & (wide["R"] > 0)].copy()
    n_dropped = len(wide) - len(contested)
    log.info(
        "District × year obs: %d total, %d contested, %d uncontested dropped",
        len(wide), len(contested), n_dropped,
    )

    # Compute two-party shares
    contested["s_D"] = contested["D"] / contested["two_party_total"]
    contested["s_R"] = contested["R"] / contested["two_party_total"]

    return contested


# ── Step 4: Compute ER polarization by district ───────────────────────────────

def er_measure(s_D: pd.Series, s_R: pd.Series, alpha: float) -> pd.Series:
    """
    Esteban & Ray (1994) polarization for two-party competition:
        P^ER = s_D * s_R * (s_D^alpha + s_R^alpha)

    With alpha=1: simplifies to s_D * s_R (since s_D + s_R = 1).
    Maximised at s_D = s_R = 0.5, where P^ER = 0.25.
    """
    return s_D * s_R * (s_D ** alpha + s_R ** alpha)


def add_er_columns(df: pd.DataFrame) -> pd.DataFrame:
    alphas = [ALPHA_BASELINE] + ALPHA_ROBUST
    for alpha in alphas:
        col = f"pol_er_alpha{str(alpha).replace('.', '')}"
        df[col] = er_measure(df["s_D"], df["s_R"], alpha)
    return df


# ── Step 5: Aggregate to state × election-year ────────────────────────────────

def aggregate_to_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weight each district's ER measure by its share of the state's total
    two-party votes. This gives larger (more populous) districts more weight.
    """
    # State total two-party votes (for weights)
    state_total = (
        df.groupby(["year", "state_fips"])["two_party_total"]
        .sum()
        .rename("state_two_party_total")
        .reset_index()
    )
    df = df.merge(state_total, on=["year", "state_fips"])
    df["district_weight"] = df["two_party_total"] / df["state_two_party_total"]

    # Weighted average of ER measure across districts within state
    er_cols = [c for c in df.columns if c.startswith("pol_er_alpha")]
    agg_dict = {col: lambda x, c=col: (df.loc[x.index, c] * df.loc[x.index, "district_weight"]).sum()
                for col in er_cols}
    agg_dict["n_districts"]  = ("district", "count")
    agg_dict["n_contested"]  = ("district", "count")   # all remaining are contested

    state_pol = (
        df.groupby(["year", "state_fips"])
        .apply(lambda g: pd.Series({
            **{col: (g[col] * g["district_weight"]).sum() for col in er_cols},
            "n_districts": len(g),
            "n_contested": len(g),   # by construction after filtering
        }))
        .reset_index()
    )

    state_pol["state_abbr"] = state_pol["state_fips"].map(FIPS_TO_ABBR)
    log.info("State × election-year observations: %d", len(state_pol))
    return state_pol


# ── Step 6: Forward-fill to annual frequency ──────────────────────────────────

def expand_to_annual(df: pd.DataFrame, start_year: int = 1976,
                     end_year: int = 2023) -> pd.DataFrame:
    """
    House elections occur every even year. Forward-fill each election cycle's
    polarization measure into the following odd year (e.g., 2002 measure applies
    to 2002 and 2003). This gives an annual state × year panel.

    Convention: the election-year measure is used for that year and the
    subsequent year (until the next election updates it).
    """
    all_years  = range(start_year, end_year + 1)
    states     = df["state_fips"].unique()
    er_cols    = [c for c in df.columns if c.startswith("pol_er_alpha")]

    # Build a full state × year grid
    grid = pd.MultiIndex.from_product(
        [states, list(all_years)], names=["state_fips", "year"]
    ).to_frame(index=False)

    merged = grid.merge(df, on=["state_fips", "year"], how="left")

    # Sort and forward-fill within each state
    merged = merged.sort_values(["state_fips", "year"])
    merged[er_cols + ["n_districts", "n_contested", "state_abbr"]] = (
        merged.groupby("state_fips")[er_cols + ["n_districts", "n_contested", "state_abbr"]]
        .ffill()
    )

    # Drop rows before first election observation (no valid measure yet)
    merged = merged.dropna(subset=["pol_er_alpha10"])

    log.info("Annual panel: %d state × year rows", len(merged))
    return merged


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== 02_build_polarization.py  start ===")

    # Step 1: Load
    raw = load_raw(RAW_FILE)

    # Step 2: Clean
    clean_df = clean(raw)

    # Step 3: Two-party district shares
    district_df = compute_district_shares(clean_df)

    # Step 4: ER measure by district
    district_df = add_er_columns(district_df)

    # Step 5: Aggregate to state × election-year
    state_df = aggregate_to_state(district_df)

    # Step 6: Expand to annual
    annual_df = expand_to_annual(state_df)

    # Sanity checks
    er_col = "pol_er_alpha10"
    log.info("ER measure (α=1) summary:\n%s", annual_df[er_col].describe().to_string())
    log.info("Max theoretical value = 0.25 (at s_D = s_R = 0.5)")
    log.info(
        "Most polarized state × year:\n%s",
        annual_df.nlargest(5, er_col)[
            ["year", "state_abbr", er_col, "n_districts"]
        ].to_string(index=False),
    )
    log.info(
        "Least polarized state × year:\n%s",
        annual_df.nsmallest(5, er_col)[
            ["year", "state_abbr", er_col, "n_districts"]
        ].to_string(index=False),
    )

    # Write output
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    annual_df.to_parquet(OUT_FILE, index=False)
    log.info("Polarization panel written: %s", OUT_FILE)
    log.info("=== done ===")


if __name__ == "__main__":
    main()
