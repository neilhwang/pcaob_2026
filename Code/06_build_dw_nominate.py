"""
06_build_dw_nominate.py
=======================
Build state-level and national ideological polarization measures from
Voteview DW-NOMINATE scores for the U.S. House of Representatives.

INPUT:  Voteview HSall_members.csv
        (downloaded automatically from voteview.com, or manually to Data/Raw/)
OUTPUT: Data/Processed/dw_nominate_polarization.parquet
        Columns: state_abbr, year,
                 dw_cross_party_gap, dw_delegation_spread,
                 dw_national_gap

MEASURES:
    State-level (varies by state x Congress):
        dw_cross_party_gap:  mean(R nominate_dim1) - mean(D nominate_dim1)
            among House members from the state. Captures the ideological
            distance between parties within the state's delegation.
            Only computed when a state has both D and R members.
        dw_delegation_spread:  SD of nominate_dim1 across all House members
            from the state. Captures within-delegation ideological dispersion
            regardless of party label.

    National-level (varies by Congress only — same for all states in a year):
        dw_national_gap:  mean(R nominate_dim1) - mean(D nominate_dim1)
            across ALL House members nationally. Captures elite-level partisan
            polarization as a time-series variable.

    All measures are expanded to annual frequency (each Congress covers two
    calendar years) and the output covers 1976-2024 to align with the ER
    polarization panel from script 02.

NOTE ON TIMING:
    DW-NOMINATE scores are computed over the full Congress (2 years) and are
    thus known only ex post. However, scores are highly persistent for
    returning members (Poole & Rosenthal 1997, 2007), so the score for
    Congress N closely tracks what could have been predicted from Congress N-1.
    We assign Congress N's scores to its own calendar years (standard approach
    in the literature). As robustness, the merge script can lag by one Congress.

HOW TO DOWNLOAD THE RAW DATA (optional — script downloads automatically):
    1. Go to: https://voteview.com/data
    2. Click "Members" then download the "All Congresses" CSV
    3. Save as: Data/Raw/HSall_members.csv

DEPENDENCIES:
    pip install requests pandas pyarrow
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
VOTEVIEW_URL = (
    "https://voteview.com/static/data/out/members/HSall_members.csv"
)
USER_AGENT = "Research project neil.hwang@bcc.cuny.edu"
START_CONGRESS = 95    # 1977-1978 (aligns with 1976 election data from script 02)
END_CONGRESS   = 118   # 2023-2024

ROOT     = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "Data/Raw/HSall_members.csv"
OUT_FILE = ROOT / "Data/Processed/dw_nominate_polarization.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Congress ↔ year mapping ───────────────────────────────────────────────────

def congress_start_year(congress: int) -> int:
    """First calendar year of a given Congress. Congress N covers years
    (2*N + 1787) and (2*N + 1788)."""
    return 2 * congress + 1787


# ── Step 1: Load Voteview data ───────────────────────────────────────────────

def load_voteview() -> pd.DataFrame:
    """
    Load DW-NOMINATE member data. Uses local file if available, otherwise
    downloads from Voteview.
    """
    if RAW_FILE.exists():
        log.info("Reading local Voteview file: %s", RAW_FILE)
        df = pd.read_csv(RAW_FILE)
    else:
        import requests
        log.info("Downloading Voteview data from voteview.com ...")
        resp = requests.get(
            VOTEVIEW_URL,
            headers={"User-Agent": USER_AGENT},
            timeout=60,
        )
        resp.raise_for_status()
        # Save to Data/Raw/ for future runs
        RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
        RAW_FILE.write_bytes(resp.content)
        log.info("Saved to: %s", RAW_FILE)
        import io
        df = pd.read_csv(io.StringIO(resp.text))

    log.info("Voteview data loaded: %d rows", len(df))
    return df


# ── Step 2: Filter to relevant House members ────────────────────────────────

def filter_house_members(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only House members from relevant Congresses with valid DW-NOMINATE
    scores and D/R party affiliation.
    """
    # House only
    df = df[df["chamber"] == "House"].copy()

    # Relevant Congresses
    df = df[
        (df["congress"] >= START_CONGRESS) & (df["congress"] <= END_CONGRESS)
    ]

    # Valid DW-NOMINATE first dimension score
    df = df[df["nominate_dim1"].notna()]

    # D (100) and R (200) only — third parties are excluded because the
    # cross-party gap measure requires a clear two-party split.
    df = df[df["party_code"].isin([100, 200])].copy()
    df["party"] = np.where(df["party_code"] == 100, "D", "R")

    # Exclude DC and territories (keep only 50 states)
    valid_states = {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    }
    df = df[df["state_abbrev"].isin(valid_states)]

    log.info(
        "Filtered to House D/R members: %d rows, Congresses %d-%d, %d states",
        len(df), df["congress"].min(), df["congress"].max(),
        df["state_abbrev"].nunique(),
    )
    return df


# ── Step 3: Compute state-level measures ─────────────────────────────────────

def compute_state_measures(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each state x Congress, compute:
        dw_cross_party_gap:   mean(R dim1) - mean(D dim1)
        dw_delegation_spread: SD of dim1 across all members
    """
    records = []
    for (congress, state), grp in df.groupby(["congress", "state_abbrev"]):
        d_scores = grp.loc[grp["party"] == "D", "nominate_dim1"]
        r_scores = grp.loc[grp["party"] == "R", "nominate_dim1"]

        # Cross-party gap: requires both parties present
        if len(d_scores) > 0 and len(r_scores) > 0:
            gap = r_scores.mean() - d_scores.mean()
        else:
            gap = np.nan

        # Delegation spread: SD of all members' scores
        spread = grp["nominate_dim1"].std() if len(grp) >= 2 else np.nan

        records.append({
            "congress":              congress,
            "state_abbrev":          state,
            "dw_cross_party_gap":    gap,
            "dw_delegation_spread":  spread,
            "n_house_members":       len(grp),
            "n_d":                   len(d_scores),
            "n_r":                   len(r_scores),
        })

    state_df = pd.DataFrame(records)
    log.info(
        "State x Congress measures: %d rows, %d with cross-party gap",
        len(state_df), state_df["dw_cross_party_gap"].notna().sum(),
    )
    return state_df


# ── Step 4: Compute national-level measure ───────────────────────────────────

def compute_national_measures(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each Congress, compute the national cross-party gap:
        dw_national_gap: mean(R dim1) - mean(D dim1) across ALL House members.
    This is a pure time-series variable (same for all states in a year).
    """
    records = []
    for congress, grp in df.groupby("congress"):
        d_scores = grp.loc[grp["party"] == "D", "nominate_dim1"]
        r_scores = grp.loc[grp["party"] == "R", "nominate_dim1"]

        gap = r_scores.mean() - d_scores.mean()
        records.append({
            "congress":         congress,
            "dw_national_gap":  gap,
        })

    national_df = pd.DataFrame(records)
    log.info("National measures: %d Congresses", len(national_df))
    return national_df


# ── Step 5: Expand to annual frequency ───────────────────────────────────────

def expand_to_annual(
    state_df: pd.DataFrame, national_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Each Congress covers two calendar years. Expand to annual frequency by
    assigning the same measures to both years of the Congress.

    Congress N covers years (2*N + 1787) and (2*N + 1788).
    """
    # Expand state-level
    annual_rows = []
    for _, row in state_df.iterrows():
        y1 = congress_start_year(int(row["congress"]))
        for yr in [y1, y1 + 1]:
            annual_rows.append({
                "state_abbrev":         row["state_abbrev"],
                "year":                 yr,
                "dw_cross_party_gap":   row["dw_cross_party_gap"],
                "dw_delegation_spread": row["dw_delegation_spread"],
            })
    annual_state = pd.DataFrame(annual_rows)

    # Expand national-level
    national_rows = []
    for _, row in national_df.iterrows():
        y1 = congress_start_year(int(row["congress"]))
        for yr in [y1, y1 + 1]:
            national_rows.append({
                "year":            yr,
                "dw_national_gap": row["dw_national_gap"],
            })
    annual_nat = pd.DataFrame(national_rows)

    # Merge: add national gap to each state-year row
    out = annual_state.merge(annual_nat, on="year", how="left")

    # Rename state_abbrev → state_abbr to match script 02's output convention
    out = out.rename(columns={"state_abbrev": "state_abbr"})

    # Keep only years within our sample window (align with ER panel)
    out = out[(out["year"] >= 1976) & (out["year"] <= 2024)]
    out = out.sort_values(["state_abbr", "year"]).reset_index(drop=True)

    log.info("Annual panel: %d state x year rows", len(out))
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== 06_build_dw_nominate.py  start ===")

    # Step 1: Load
    raw = load_voteview()

    # Step 2: Filter
    house = filter_house_members(raw)

    # Step 3: State-level measures
    state_df = compute_state_measures(house)

    # Step 4: National-level measures
    national_df = compute_national_measures(house)

    # Step 5: Expand to annual
    annual = expand_to_annual(state_df, national_df)

    # ── Sanity checks ────────────────────────────────────────────────────────
    log.info(
        "dw_cross_party_gap summary:\n%s",
        annual["dw_cross_party_gap"].describe().to_string(),
    )
    log.info(
        "dw_national_gap summary:\n%s",
        annual["dw_national_gap"].describe().to_string(),
    )

    # Time series: show national gap trend
    nat_trend = (
        annual[["year", "dw_national_gap"]]
        .drop_duplicates()
        .sort_values("year")
    )
    log.info(
        "National gap trend (selected years):\n%s",
        nat_trend[nat_trend["year"].isin([1980, 1990, 2000, 2010, 2020, 2024])]
        .to_string(index=False),
    )

    # Cross-section: most and least polarized states in recent Congress
    recent = annual[annual["year"] == 2022].sort_values(
        "dw_cross_party_gap", ascending=False
    )
    log.info(
        "Most polarized states (2022):\n%s",
        recent.head(5)[["state_abbr", "dw_cross_party_gap", "dw_delegation_spread"]]
        .to_string(index=False),
    )
    log.info(
        "Least polarized states (2022):\n%s",
        recent.dropna(subset=["dw_cross_party_gap"]).tail(5)[
            ["state_abbr", "dw_cross_party_gap", "dw_delegation_spread"]
        ].to_string(index=False),
    )

    # States with missing cross-party gap (single-party delegations)
    missing = annual[annual["dw_cross_party_gap"].isna()]
    if len(missing) > 0:
        log.info(
            "State x years with missing cross-party gap (single-party delegation): %d",
            len(missing),
        )
        log.info(
            "  Examples: %s",
            missing[["state_abbr", "year"]].head(10).to_string(index=False),
        )

    # ── Write output ─────────────────────────────────────────────────────────
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    annual.to_parquet(OUT_FILE, index=False)
    log.info("Output written: %s  (%d rows)", OUT_FILE, len(annual))
    log.info("=== done ===")


if __name__ == "__main__":
    main()
