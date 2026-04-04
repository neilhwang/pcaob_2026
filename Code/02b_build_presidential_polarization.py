"""
02b_build_presidential_polarization.py
=======================================
Build state-year presidential polarization measures from county-level returns.
Presidential elections show far more red/blue state variation than House data,
providing better cross-sectional identification.

Three measures produced per state per election year:
  er_pres   — Esteban-Ray index from presidential two-party shares = D*R.
               Same concept as the House ER measure; range (0, 0.25].
  margin    — Absolute two-party margin = |D_share - R_share|. Captures
               partisan homogeneity: high margin = one-party state.
               Range [0, 1]. Negatively related to within-state heterogeneity.
  county_sd — Population-weighted within-state SD of county-level D_share.
               Captures geographic partisan sorting: high county_sd means
               a state has both deep-red and deep-blue counties (genuine
               polarization), whereas low county_sd means counties are
               uniformly similar (competitive-but-moderate). Addresses the
               Fiorina (2005) critique that electoral competitiveness does
               not necessarily imply ideological polarization.

Identification note:
  Both measures vary substantially across states (Alabama 2020 ≈ 0.20 ER vs
  Nevada 2020 ≈ 0.25 ER; margin varies from ~0.01 for swing states to ~0.40
  for one-party states). This cross-sectional spread is the identification.

Forward-filling:
  Elections every 4 years. Each election-year value is forward-filled through
  the following three calendar years until the next election (e.g., 2000
  returns cover 2001–2004; 2004 returns cover 2005–2008; etc.).

INPUT:  Data/Raw/countypres_2000-2024.tab
OUTPUT: Data/Processed/pol_presidential.parquet
        Columns: state_abbr, year, dem_share, rep_share, er_pres, margin, county_sd
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent
IN_FILE = ROOT / "Data/Raw/countypres_2000-2024.tab"
OUT_FILE= ROOT / "Data/Processed/pol_presidential.parquet"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SAMPLE_START = 2001
SAMPLE_END   = 2023


def main() -> None:
    log.info("Loading county presidential data: %s", IN_FILE)
    raw = pd.read_csv(IN_FILE, sep="\t",
                      usecols=["state_po", "county_fips", "year", "party",
                               "candidatevotes", "mode"],
                      dtype={"county_fips": str})
    log.info("Raw rows: %d  |  election years: %s",
             len(raw), sorted(raw["year"].unique()))

    # Keep only TOTAL mode and Democrat/Republican votes; drop NaN votes
    dr = raw[(raw["party"].isin(["DEMOCRAT", "REPUBLICAN"]))
             & (raw["mode"] == "TOTAL")].copy()
    dr["candidatevotes"] = pd.to_numeric(dr["candidatevotes"], errors="coerce")
    dr = dr.dropna(subset=["candidatevotes"])

    # ── County-level D-share (for within-state SD) ─────────────────────────
    county_votes = (
        dr.groupby(["state_po", "county_fips", "year", "party"],
                    as_index=False)["candidatevotes"]
        .sum()
        .pivot_table(index=["state_po", "county_fips", "year"],
                     columns="party",
                     values="candidatevotes",
                     aggfunc="sum")
        .reset_index()
    )
    county_votes.columns.name = None
    county_votes = county_votes.rename(columns={
        "DEMOCRAT": "dem_votes", "REPUBLICAN": "rep_votes",
    })
    county_votes["two_party"] = county_votes["dem_votes"] + county_votes["rep_votes"]
    county_votes = county_votes[county_votes["two_party"] > 0].copy()
    county_votes["d_share"] = county_votes["dem_votes"] / county_votes["two_party"]

    # Population-weighted within-state SD of county D-share
    # Weights = county two-party votes (proxy for voting population)
    def weighted_sd(grp):
        w = grp["two_party"].values
        x = grp["d_share"].values
        w_sum = w.sum()
        if w_sum == 0 or len(x) < 2:
            return np.nan
        w_mean = np.average(x, weights=w)
        w_var = np.average((x - w_mean) ** 2, weights=w)
        # Bessel-like correction: multiply by N/(N-1) where N = effective sample size
        return np.sqrt(w_var)

    county_sd = (
        county_votes.groupby(["state_po", "year"])
        .apply(weighted_sd, include_groups=False)
        .reset_index(name="county_sd")
        .rename(columns={"state_po": "state_abbr"})
    )
    log.info("County SD: %d state-year obs, range %.4f–%.4f (mean %.4f, SD %.4f)",
             len(county_sd),
             county_sd["county_sd"].min(), county_sd["county_sd"].max(),
             county_sd["county_sd"].mean(), county_sd["county_sd"].std())

    # ── Aggregate to state-election level (sum across counties) ────────────
    state_votes = (
        dr.groupby(["state_po", "year", "party"], as_index=False)["candidatevotes"]
        .sum()
        .pivot_table(index=["state_po", "year"],
                     columns="party",
                     values="candidatevotes",
                     aggfunc="sum")
        .reset_index()
    )
    state_votes.columns.name = None
    state_votes = state_votes.rename(columns={
        "state_po": "state_abbr",
        "DEMOCRAT": "dem_votes",
        "REPUBLICAN": "rep_votes",
    })

    # Two-party shares
    two_party = state_votes["dem_votes"] + state_votes["rep_votes"]
    state_votes["dem_share"] = state_votes["dem_votes"] / two_party
    state_votes["rep_share"] = state_votes["rep_votes"] / two_party

    # ER polarization index and partisan margin
    state_votes["er_pres"] = state_votes["dem_share"] * state_votes["rep_share"]
    state_votes["margin"]  = (state_votes["dem_share"] - state_votes["rep_share"]).abs()

    # Join county SD to state-level measures
    state_votes = state_votes.merge(county_sd, on=["state_abbr", "year"], how="left")

    log.info("Election-year state observations: %d", len(state_votes))
    log.info("ER range:    %.4f – %.4f  (mean %.4f, SD %.4f)",
             state_votes["er_pres"].min(), state_votes["er_pres"].max(),
             state_votes["er_pres"].mean(), state_votes["er_pres"].std())
    log.info("Margin range: %.4f – %.4f  (mean %.4f, SD %.4f)",
             state_votes["margin"].min(), state_votes["margin"].max(),
             state_votes["margin"].mean(), state_votes["margin"].std())
    log.info("County SD range: %.4f – %.4f  (mean %.4f, SD %.4f)",
             state_votes["county_sd"].min(), state_votes["county_sd"].max(),
             state_votes["county_sd"].mean(), state_votes["county_sd"].std())

    # Forward-fill to annual: election year e → covers e+1 through e+4
    # (the 2000 election governs 2001-2004, the 2004 election 2005-2008, etc.)
    # We need coverage through SAMPLE_END.
    all_years = list(range(SAMPLE_START, SAMPLE_END + 1))
    states    = state_votes["state_abbr"].unique()

    rows = []
    for state in states:
        s = state_votes[state_votes["state_abbr"] == state].sort_values("year")
        election_years = sorted(s["year"].unique())

        for cal_year in all_years:
            # Most recent election year ≤ cal_year - 1
            # (election in November year t → data used starting calendar year t+1)
            eligible = [ey for ey in election_years if ey <= cal_year - 1]
            if not eligible:
                continue
            most_recent = max(eligible)
            row_src = s[s["year"] == most_recent].iloc[0]
            rows.append({
                "state_abbr": state,
                "year":       cal_year,
                "election_year": most_recent,
                "dem_share":  row_src["dem_share"],
                "rep_share":  row_src["rep_share"],
                "er_pres":    row_src["er_pres"],
                "margin":     row_src["margin"],
                "county_sd":  row_src["county_sd"],
            })

    annual = pd.DataFrame(rows)
    log.info("Annual panel: %d rows, %d states, years %d–%d",
             len(annual), annual["state_abbr"].nunique(),
             annual["year"].min(), annual["year"].max())

    log.info("Annual ER range:    %.4f – %.4f  (mean %.4f, SD %.4f)",
             annual["er_pres"].min(), annual["er_pres"].max(),
             annual["er_pres"].mean(), annual["er_pres"].std())
    log.info("Annual margin range: %.4f – %.4f  (mean %.4f, SD %.4f)",
             annual["margin"].min(), annual["margin"].max(),
             annual["margin"].mean(), annual["margin"].std())

    annual.to_parquet(OUT_FILE, index=False)
    log.info("Written: %s", OUT_FILE)
    log.info("=== done ===")


if __name__ == "__main__":
    main()
