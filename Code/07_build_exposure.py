"""
07_build_exposure.py
====================
Build a time-invariant state-level partisan exposure variable from MIT Election
Lab county-level presidential vote data.

INPUT:  Data/Raw/countypres_2000-2024.tab
        (MIT Election Lab county presidential returns, Harvard Dataverse
         DOI:10.7910/DVN/VOQCHQ)

OUTPUT: Data/Processed/state_partisan_exposure.parquet
        Columns: state_abbr, exposure_pres

MEASURE:
    exposure_pres = mean over all available election years of
                    |R_share_{s,t} - 0.5|

    where R_share_{s,t} = Republican two-party vote share for state s in
    presidential election year t (2000, 2004, ..., 2024).

    Interpretation: high values (e.g., Wyoming, Massachusetts) indicate states
    that consistently vote lopsidedly for one party; low values (e.g., Florida,
    Wisconsin) indicate genuine swing states.  This is a cross-sectional
    shifter used to identify the role of national affective polarization in
    script 05's supplementary validation specification.

    The measure is time-invariant (averaged over all available elections) to
    avoid reverse causality: firm-level auditor change reactions in a given
    year cannot affect long-run state partisan intensity.

NOTE ON TOTAL vs. SUB-MODE ROWS:
    The raw file contains rows for both aggregated totals (mode == 'TOTAL') and
    vote-by-mode breakdowns (EARLY VOTING, ELECTION DAY, etc.).  We use only
    mode == 'TOTAL' rows, which represent complete county totals and cover
    ~95% of county-year-party observations.  The remaining ~5% (counties
    reporting only sub-mode rows) are small counties whose exclusion does not
    materially affect state-level shares.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "Data/Raw/countypres_2000-2024.tab"
OUT_FILE = ROOT / "Data/Processed/state_partisan_exposure.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_and_filter(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    log.info("Raw rows: %d", len(df))

    # Total rows only (avoids double-counting vote-by-mode breakdowns)
    df = df[df["mode"] == "TOTAL"].copy()

    # Democrat and Republican only
    df = df[df["party"].isin(["DEMOCRAT", "REPUBLICAN"])].copy()

    log.info("After TOTAL + D/R filter: %d rows", len(df))
    return df


def compute_state_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate county rows to state × election year, compute R two-party share."""
    state_votes = (
        df.groupby(["state_po", "year", "party"])["candidatevotes"]
        .sum()
        .unstack("party")
        .rename(columns={"DEMOCRAT": "dem_votes", "REPUBLICAN": "rep_votes"})
        .reset_index()
    )
    state_votes = state_votes.dropna(subset=["dem_votes", "rep_votes"])
    state_votes["two_party_total"] = state_votes["dem_votes"] + state_votes["rep_votes"]
    state_votes["rep_share"] = state_votes["rep_votes"] / state_votes["two_party_total"]
    state_votes["abs_margin"] = (state_votes["rep_share"] - 0.5).abs()

    log.info(
        "State × election years: %d  (%d states, %d elections)",
        len(state_votes),
        state_votes["state_po"].nunique(),
        state_votes["year"].nunique(),
    )
    return state_votes


def compute_exposure(state_votes: pd.DataFrame) -> pd.DataFrame:
    """Average |R_share - 0.5| over all election years per state."""
    exposure = (
        state_votes.groupby("state_po")["abs_margin"]
        .mean()
        .reset_index()
        .rename(columns={"state_po": "state_abbr", "abs_margin": "exposure_pres"})
    )

    # Sanity checks
    log.info(
        "Exposure summary:\n%s",
        exposure["exposure_pres"].describe().to_string(),
    )
    log.info(
        "Most partisan states:\n%s",
        exposure.nlargest(5, "exposure_pres")[["state_abbr", "exposure_pres"]]
        .to_string(index=False),
    )
    log.info(
        "Least partisan (swing) states:\n%s",
        exposure.nsmallest(5, "exposure_pres")[["state_abbr", "exposure_pres"]]
        .to_string(index=False),
    )
    return exposure


def main() -> None:
    log.info("=== 07_build_exposure.py  start ===")

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {RAW_FILE}\n"
            "Download from Harvard Dataverse DOI:10.7910/DVN/VOQCHQ and save "
            "as Data/Raw/countypres_2000-2024.tab"
        )

    df       = load_and_filter(RAW_FILE)
    sv       = compute_state_shares(df)
    exposure = compute_exposure(sv)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    exposure.to_parquet(OUT_FILE, index=False)
    log.info("Output written: %s  (%d states)", OUT_FILE, len(exposure))
    log.info("=== done ===")


if __name__ == "__main__":
    main()
