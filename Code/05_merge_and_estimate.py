"""
05_merge_and_estimate.py
========================
Merge all processed datasets and run the main regressions.

INPUTS:
    Data/Processed/crsp_event_window.parquet        (from 03_build_crsp_sample.py)
    Data/Processed/polarization_state_year.parquet  (from 02_build_polarization.py)
    Data/Processed/compustat_controls.parquet       (from 04_build_compustat_controls.py)
    Data/Processed/dw_nominate_polarization.parquet (from 06_build_dw_nominate.py)
    Data/Processed/state_partisan_exposure.parquet  (from 07_build_exposure.py)
    Data/Processed/affective_polarization.parquet   (from 08_build_affective_polarization.py)
    Data/Processed/pol_presidential.parquet         (from 02b_build_presidential_polarization.py)
    Data/Processed/analysis_sample_county.parquet  (from 10_build_county_polarization.py)
    Data/Processed/ibes_dispersion.parquet          (from 11_build_ibes.py)
    Data/Processed/pre_event_turnover.parquet       (from 12_build_turnover.py)
    Data/Processed/incorp_state.parquet             (from 13_build_incorp_state.py)
    Data/Processed/post_event_car.parquet           (from 14_build_post_event_car.py)
    Data/Processed/institutional_ownership.parquet  (from 18_build_institutional_ownership.py)

OUTPUTS:
    Output/Tables/tab01_summary_stats.tex
    Output/Tables/tab02_main_results.tex
    Output/Tables/tab03_event_type.tex
    Output/Tables/tab04_ambiguity.tex
    Output/Tables/tab05_robustness.tex
    Output/Tables/tab06_affective.tex
    Output/Tables/tab07_permutation.tex           (permutation test, 5,000 draws)
    Output/Tables/tab08_dispersion_interaction.tex (analyst dispersion mechanism test — skipped if no IBES)
    Output/Tables/tab08_local_bias.tex             (local investor relevance test — not included in paper)
    Output/Tables/tab09_reversal.tex              (post-event CAR reversal test — skipped if no post_event_car.parquet)
    Output/Tables/tab10_audit_credibility.tex     (audit credibility interaction test — skipped if no moderators)
    Data/Processed/analysis_sample.parquet        (merged estimation sample)

SPECIFICATION:
    Main regression (cross-section of auditor change events):

        Y_i = alpha + beta*Pol_{s(i),t(i)} + gamma'*X_{i,t} + FE + eps_i

    where Y_i is |CAR(-1,+1)| or AbVol(-1,+1), Pol is the standardized
    presidential Esteban-Ray polarization measure (state-level, from county returns)
    for the firm's HQ state in the event year,
    X is a vector of firm controls, and FE are year and 2-digit SIC fixed effects.
    Standard errors are clustered at the firm (gvkey) level.

    For firms with multiple events, firm fixed effects are added as a robustness
    check (Table 5), estimated via within-transformation using linearmodels.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
PROC      = ROOT / "Data/Processed"
OUT_TABS  = ROOT / "Output/Tables"
OUT_TABS.mkdir(parents=True, exist_ok=True)

CRSP_FILE = PROC / "crsp_event_window.parquet"
POL_FILE  = PROC / "polarization_state_year.parquet"
COMP_FILE = PROC / "compustat_controls.parquet"
DW_FILE       = PROC / "dw_nominate_polarization.parquet"
EXPOSURE_FILE  = PROC / "state_partisan_exposure.parquet"
AP_FILE        = PROC / "affective_polarization.parquet"
PRES_POL_FILE  = PROC / "pol_presidential.parquet"
SAMPLE_FILE    = PROC / "analysis_sample.parquet"
IBES_FILE      = PROC / "ibes_dispersion.parquet"
TURNOVER_FILE  = PROC / "pre_event_turnover.parquet"
INCORP_FILE    = PROC / "incorp_state.parquet"
POST_CAR_FILE  = PROC / "post_event_car.parquet"
AUDIT_CRED_FILE = PROC / "audit_credibility_moderators.parquet"
SHORT_INT_FILE  = PROC / "short_interest.parquet"
INST_OWN_FILE   = PROC / "institutional_ownership.parquet"
SPECIFICITY_FILE= PROC / "filing_specificity.parquet"
UNCERTAINTY_FILE= PROC / "post_event_uncertainty.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Controls and fixed effects used throughout ────────────────────────────────
CONTROLS = ["size", "leverage", "roa", "btm", "loss", "sales_growth"]


# ── Step 1: Load and merge ────────────────────────────────────────────────────

def load_and_merge() -> pd.DataFrame:
    # Load
    crsp     = pd.read_parquet(CRSP_FILE)
    pol      = pd.read_parquet(POL_FILE)
    comp     = pd.read_parquet(COMP_FILE)
    dw       = pd.read_parquet(DW_FILE)
    exposure = pd.read_parquet(EXPOSURE_FILE)
    ap       = pd.read_parquet(AP_FILE)
    pres_pol = pd.read_parquet(PRES_POL_FILE)

    log.info("CRSP events: %d", len(crsp))
    log.info("Polarization panel: %d rows", len(pol))
    log.info("Compustat controls: %d rows", len(comp))
    log.info("DW-NOMINATE panel: %d rows", len(dw))
    log.info("Partisan exposure: %d states", len(exposure))
    log.info("Affective polarization: %d years", len(ap))
    log.info("Presidential polarization: %d rows", len(pres_pol))

    # Derive event year from event date
    crsp["event_year"] = pd.to_datetime(crsp["event_date"]).dt.year

    # ── Merge polarization ────────────────────────────────────────────────────
    # Need HQ state — pull from Compustat (comp has state from comp.company)
    # Use the most recent state observation per gvkey
    state_map = (
        comp[["gvkey", "fyear", "state"]].dropna(subset=["state"])
        .sort_values(["gvkey", "fyear"])          # ensure most-recent fyear is last
        .drop_duplicates(subset="gvkey", keep="last")
        .drop(columns="fyear")
    )
    crsp = crsp.merge(state_map, on="gvkey", how="left")

    pol_merge = pol[["year", "state_abbr", "pol_er_alpha10",
                      "pol_er_alpha08", "pol_er_alpha12"]].rename(
        columns={"year": "event_year", "state_abbr": "state"}
    )
    crsp = crsp.merge(pol_merge, on=["event_year", "state"], how="left")
    log.info("After polarization merge: %d rows, %d with pol measure",
             len(crsp), crsp["pol_er_alpha10"].notna().sum())

    # ── Merge DW-NOMINATE ─────────────────────────────────────────────────────
    # dw_cross_party_gap: state-level R-D ideological gap (alternative pol proxy)
    # dw_national_gap:    national-level time-series (same for all states/year)
    # state_abbr in DW file → renamed to state to match merge key convention
    dw_merge = dw[["year", "state_abbr", "dw_cross_party_gap",
                    "dw_national_gap"]].rename(
        columns={"year": "event_year", "state_abbr": "state"}
    )
    crsp = crsp.merge(dw_merge, on=["event_year", "state"], how="left")
    log.info("After DW-NOMINATE merge: %d rows, %d with dw_cross_party_gap",
             len(crsp), crsp["dw_cross_party_gap"].notna().sum())

    # ── Merge partisan exposure (time-invariant, merge on state) ─────────────
    # exposure_pres = long-run avg |R_share - 0.5| from presidential elections.
    # Identifies the affective polarization spec via AP_t × Exposure_s.
    crsp = crsp.merge(
        exposure[["state_abbr", "exposure_pres"]].rename(
            columns={"state_abbr": "state"}
        ),
        on="state", how="left",
    )
    log.info("After exposure merge: %d with exposure_pres",
             crsp["exposure_pres"].notna().sum())

    # ── Merge national affective polarization (merge on event_year) ──────────
    # ap_ft = ANES feeling thermometer differential, linearly interpolated.
    # Absorbed by year FEs alone; enters only as AP × Exposure interaction.
    crsp = crsp.merge(
        ap.rename(columns={"year": "event_year"}),
        on="event_year", how="left",
    )
    log.info("After AP merge: %d with ap_ft", crsp["ap_ft"].notna().sum())

    # ── Merge presidential polarization (state × year, forward-filled) ────────
    # er_pres = D_share × R_share (ER index at state level from county returns)
    # margin  = |D_share - R_share| (partisan homogeneity; higher = one-party state)
    # Both have ~13× more cross-sectional variation than House ER index.
    pres_merge = pres_pol[["year", "state_abbr", "er_pres", "margin", "county_sd"]].rename(
        columns={"year": "event_year", "state_abbr": "state"}
    )
    crsp = crsp.merge(pres_merge, on=["event_year", "state"], how="left")
    log.info("After presidential pol merge: %d with er_pres, %d with margin, %d with county_sd",
             crsp["er_pres"].notna().sum(), crsp["margin"].notna().sum(),
             crsp["county_sd"].notna().sum())

    # ── Merge incorporation state → incorporation-state polarization (placebo) ─
    # incorp is the 2-letter state of incorporation (from comp.company via script 13).
    # We map it to the same presidential ER polarization measure used for HQ state.
    # Prediction: incorp-state polarization should NOT predict market reactions
    # (incorporation state is chosen for legal/tax reasons, not investor base).
    if INCORP_FILE.exists():
        incorp = pd.read_parquet(INCORP_FILE)
        crsp = crsp.merge(incorp, on="gvkey", how="left")
        incorp_pol_merge = pol_merge.rename(
            columns={"state": "incorp", "pol_er_alpha10": "incorp_pol"}
        )[["event_year", "incorp", "incorp_pol"]]
        crsp = crsp.merge(incorp_pol_merge, on=["event_year", "incorp"], how="left")
        log.info("After incorp-state pol merge: %d with incorp_pol",
                 crsp["incorp_pol"].notna().sum())
    else:
        crsp["incorp"]     = None
        crsp["incorp_pol"] = float("nan")
        log.warning("incorp_state.parquet not found; run 13_build_incorp_state.py "
                    "to enable the incorporation-state placebo.")

    # ── Merge Compustat controls ───────────────────────────────────────────────
    # Match to fiscal year ending in the calendar year prior to the event
    # (standard in event study literature: use pre-event fundamentals)
    comp_merge = comp[["gvkey", "fyear"] + CONTROLS + ["sic2"]].copy()
    comp_merge = comp_merge.rename(columns={"fyear": "comp_year"})

    # Event in year t → use Compustat fyear = t-1
    crsp["comp_year"] = crsp["event_year"] - 1
    crsp = crsp.merge(comp_merge, on=["gvkey", "comp_year"], how="left")
    log.info("After controls merge: %d rows, %d with controls",
             len(crsp), crsp["size"].notna().sum())

    # ── Construct analysis variables ──────────────────────────────────────────
    crsp["absCar"]  = crsp["car_m1p1"].abs()
    crsp["abvol"]   = crsp["abvol_m1p1"]
    crsp["car"]     = crsp["car_m1p1"]

    # Polarization standardization is deferred to after apply_sample_filters() so that
    # mean=0, sd=1 holds exactly in the estimation sample (not the pre-filter universe).
    # Leave er_pres and other raw columns in place; standardization applied in main().

    # Event-type indicators for heterogeneity tests
    crsp["dismissal"]      = (crsp["reason"] == "dismissal").astype(int)
    crsp["disagreement"]   = (crsp["disagreements"].astype(str).str.lower() == "true").astype(int)
    crsp["big4_departure"] = crsp["quality_direction"].isin(
        ["Big4_to_nonBig4", "Big4_to_Big4"]
    ).astype(int)
    crsp["quality_down"]   = (crsp["quality_direction"] == "Big4_to_nonBig4").astype(int)
    crsp["quality_up"]     = (crsp["quality_direction"] == "nonBig4_to_Big4").astype(int)

    # High-ambiguity flag: no disclosed disagreement AND not a clear quality change
    crsp["high_ambiguity"] = (
        (crsp["disagreement"] == 0) &
        (crsp["quality_direction"].isin(
            ["nonBig4_to_nonBig4", "Big4_to_Big4", "unknown"]
        ))
    ).astype(int)

    # sic2 as string category for FE
    # Convert sic2 to string for use as a fixed effect.
    # Must fill NA before converting: nullable Int64 → str produces "<NA>",
    # which creates a spurious FE level that absorbs firms with missing SIC.
    crsp["sic2_str"]  = crsp["sic2"].fillna(-1).astype(int).astype(str)
    crsp["year_str"]  = crsp["event_year"].astype(str)
    crsp["gvkey_str"] = crsp["gvkey"].astype(str)
    crsp["state_str"] = crsp["state"].fillna("MISSING").astype(str)

    log.info("Analysis sample: %d events", len(crsp))
    return crsp


def apply_sample_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Drop observations missing key variables and apply standard SIC exclusions."""
    n0 = len(df)
    # Filter on er_pres (the primary raw measure), not er_pres_std — the
    # standardized version is created in main() after this function returns.
    df = df.dropna(subset=["absCar", "abvol", "er_pres", "sic2"] + CONTROLS)
    log.info("After dropping missing: %d rows (dropped %d)", len(df), n0 - len(df))

    # Exclude financial firms (SIC 6000-6999) and utilities (SIC 4900-4999).
    # Standard in the accounting literature; these industries have distinct
    # regulatory environments and auditor relationships.
    n1 = len(df)
    df = df[~df["sic2"].isin(range(60, 70)) & ~df["sic2"].isin(range(49, 50))]
    log.info("After SIC exclusions (financials/utilities): %d rows (dropped %d)",
             len(df), n1 - len(df))

    return df.reset_index(drop=True)


# ── Step 2: Summary statistics ────────────────────────────────────────────────

def make_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "absCar":       "|CAR(-1,+1)|",
        "car":          "CAR(-1,+1)",
        "abvol":        "Abn. Volume",
        "margin":         "Partisan Margin $|D-R|$ (= 1 $-$ Polarization)",
        "size":         "Size (log assets)",
        "leverage":     "Leverage",
        "roa":          "ROA",
        "btm":          "Book-to-Market",
        "loss":         "Loss indicator",
        "sales_growth": "Sales Growth",
    }
    rows = []
    for col, label in cols.items():
        s = df[col].dropna()
        rows.append({
            "Variable": label,
            "N":    f"{len(s):,}",
            "Mean": f"{s.mean():.3f}",
            "SD":   f"{s.std():.3f}",
            "p25":  f"{s.quantile(0.25):.3f}",
            "p50":  f"{s.median():.3f}",
            "p75":  f"{s.quantile(0.75):.3f}",
        })
    return pd.DataFrame(rows)


def to_latex_table(df: pd.DataFrame, caption: str, label: str,
                   out_path: Path) -> None:
    col_fmt = "l" + "r" * (len(df.columns) - 1)
    tex = df.to_latex(
        index=False,
        column_format=col_fmt,
        caption=caption,
        label=label,
        escape=True,
    )
    # pandas ≥ 2.0 removed booktabs argument; replace \hline with booktabs rules
    _count = [0]
    def _hline_to_booktabs(m):
        _count[0] += 1
        if _count[0] == 1: return r"\toprule"
        if _count[0] == 2: return r"\midrule"
        return r"\bottomrule"
    import re as _re
    tex = _re.sub(r"\\hline", _hline_to_booktabs, tex)
    # Wrap in table environment with notes placeholder
    tex = tex.replace(
        r"\end{tabular}",
        r"\end{tabular}" + "\n" + r"\begin{flushleft}" + "\n"
        r"\footnotesize Notes: Sample consists of 678 auditor-change events "
        r"(Form~8-K Item~4.01) from 2001--2023, matched to CRSP and Compustat. "
        r"Financial firms (SIC 6000--6999) and utilities (SIC 4900--4999) are excluded. "
        r"$|CAR(-1,+1)|$ and $CAR(-1,+1)$ are cumulative abnormal returns from a "
        r"market model estimated over trading days $[-252, -46]$. "
        r"Abn.\ Volume is the mean daily volume in the event window $[-1, +1]$ "
        r"minus the estimation-window mean, scaled by estimation-window standard deviation. "
        r"Partisan Margin is $|D - R|$ from state-level presidential returns "
        r"(MIT Election Lab); Polarization $= 1 -$ Margin. "
        r"Firm controls are measured as of the fiscal year preceding the event." + "\n"
        r"\end{flushleft}",
    )
    out_path.write_text(tex, encoding="utf-8")
    log.info("Table written: %s", out_path)


# ── Step 3: Regression helpers ────────────────────────────────────────────────

def run_ols(formula: str, df: pd.DataFrame,
            cluster_var: str = "gvkey_str") -> object:
    """OLS with one-way clustered standard errors.

    statsmodels drops NaN rows silently; use patsy to recover which rows
    survive so the cluster variable has the same length as the model data.
    """
    import patsy
    _, X = patsy.dmatrices(formula, data=df, return_type="dataframe",
                           NA_action="drop")
    keep_idx = X.index
    groups = df.loc[keep_idx, cluster_var].values
    return smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )


class _TwoWayResults:
    """Thin wrapper exposing statsmodels-like attributes with two-way clustered SEs."""
    def __init__(self, base, V_2way):
        from scipy import stats as _sp
        self.params   = base.params
        self.nobs     = base.nobs
        self.rsquared = base.rsquared
        self._V       = V_2way
        n, k          = int(self.nobs), len(self.params)
        self.bse      = pd.Series(np.sqrt(np.diag(V_2way.values)),
                                  index=self.params.index)
        tstat         = self.params / self.bse
        self.pvalues  = pd.Series(
            2 * _sp.t.sf(np.abs(tstat), df=n - k),
            index=self.params.index,
        )

    def cov_params(self):
        return self._V


def run_ols_twoway(formula: str, df: pd.DataFrame,
                   var1: str = "gvkey_str",
                   var2: str = "state_str") -> _TwoWayResults:
    """OLS with two-way clustered SEs (Cameron, Gelbach & Miller 2011).

    V = V_{var1} + V_{var2} - V_{var1 × var2}

    Because firm HQ state is fixed (firms are nested within states), the
    intersection clusters equal the firm clusters, so V_2way = V_state in
    expectation; the CGM formula makes this explicit.
    """
    import patsy
    from statsmodels.stats.sandwich_covariance import cov_cluster

    _, X = patsy.dmatrices(formula, data=df, return_type="dataframe",
                           NA_action="drop")
    keep_idx = X.index
    df_k = df.loc[keep_idx]
    g1  = df_k[var1].values
    g2  = df_k[var2].values
    g12 = (df_k[var1].astype(str) + "||" + df_k[var2].astype(str)).values

    base = smf.ols(formula, data=df).fit()
    V1   = cov_cluster(base, g1)
    V2   = cov_cluster(base, g2)
    V12  = cov_cluster(base, g12)
    V_2w = pd.DataFrame(V1 + V2 - V12,
                        index=base.params.index, columns=base.params.index)
    return _TwoWayResults(base, V_2w)


def fmt(x, digits=3):
    if x is None or np.isnan(x):
        return ""
    return f"{x:.{digits}f}"


def stars(pval):
    if pval < 0.01:  return "***"
    if pval < 0.05:  return "**"
    if pval < 0.10:  return "*"
    return ""


def reg_table(models: list, dep_labels: list, coef_map: dict,
              caption: str, label: str, out_path: Path,
              extra_rows: dict | None = None) -> None:
    """
    Write a regression table to LaTeX.
    models: list of fitted statsmodels results
    dep_labels: column headers (dependent variable names)
    coef_map: {param_name: display_label} — rows to show, in order
    extra_rows: {row_label: [col1_val, col2_val, ...]} for FE/N rows
    """
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    ncols = len(models)
    lines.append(r"\begin{tabular}{l" + "c" * ncols + "}")
    lines.append(r"\toprule")

    # Header
    header = " & " + " & ".join(f"({i+1})" for i in range(ncols)) + r" \\"
    lines.append(header)
    dep_row = " & " + " & ".join(dep_labels) + r" \\"
    lines.append(dep_row)
    lines.append(r"\midrule")

    # Coefficients
    for param, display in coef_map.items():
        coef_cells = []
        se_cells   = []
        for m in models:
            if param in m.params:
                c = m.params[param]
                p = m.pvalues[param]
                s = m.bse[param]
                coef_cells.append(f"{fmt(c)}{stars(p)}")
                # Use 4 decimal places for SEs when rounding to 3 would
                # alter the implied t-statistic's significance band (M4 fix).
                se_digits = 4 if abs(s) < 0.1 else 3
                se_cells.append(f"({fmt(s, se_digits)})")
            else:
                coef_cells.append("")
                se_cells.append("")
        lines.append(f"{display} & " + " & ".join(coef_cells) + r" \\")
        lines.append(" & " + " & ".join(se_cells) + r" \\")

    lines.append(r"\midrule")

    # Extra rows (FE indicators, N, R²)
    if extra_rows:
        for row_label, vals in extra_rows.items():
            lines.append(f"{row_label} & " + " & ".join(str(v) for v in vals) + r" \\")

    # N and R²
    n_row  = "N & " + " & ".join(f"{int(m.nobs):,}" for m in models) + r" \\"
    r2_row = r"$R^2$ & " + " & ".join(fmt(m.rsquared) for m in models) + r" \\"
    lines.append(n_row)
    lines.append(r2_row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{flushleft}")
    lines.append(r"\footnotesize Notes: Standard errors clustered at the firm level in parentheses. "
                 r"$^{***}$, $^{**}$, $^{*}$ denote significance at 1\%, 5\%, 10\%.")
    lines.append(r"\end{flushleft}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines))
    log.info("Table written: %s", out_path)


# ── Step 4: Main results (Table 2) ────────────────────────────────────────────

def run_main_results(df: pd.DataFrame) -> None:
    ctrl = " + ".join(CONTROLS)
    fe   = "C(year_str) + C(sic2_str)"

    specs = {
        # (1) AbVol, no controls, year FE only
        "m1": run_ols(f"abvol ~ competitive_std + C(year_str)", df),
        # (2) AbVol, controls + year FE + industry FE
        "m2": run_ols(f"abvol ~ competitive_std + {ctrl} + {fe}", df),
        # (3) |CAR|, no controls, year FE only
        "m3": run_ols(f"absCar ~ competitive_std + C(year_str)", df),
        # (4) |CAR|, controls + year FE + industry FE
        "m4": run_ols(f"absCar ~ competitive_std + {ctrl} + {fe}", df),
    }

    for k, m in specs.items():
        log.info("%s: competitive_std coef=%.4f p=%.3f", k, m.params["competitive_std"],
                 m.pvalues["competitive_std"])

    coef_map = {
        "competitive_std": r"\textit{Polarization}",
        "size":        "Size",
        "leverage":    "Leverage",
        "roa":         "ROA",
        "btm":         "Book-to-Market",
        "loss":        "Loss",
        "sales_growth":"Sales Growth",
    }
    dep_labels = [r"AbVol", r"AbVol", r"$|CAR|$", r"$|CAR|$"]
    extra_rows = {
        "Year FE":     ["Yes", "Yes", "Yes", "Yes"],
        "Industry FE": ["No",  "Yes", "No",  "Yes"],
        "Controls":    ["No",  "Yes", "No",  "Yes"],
    }
    reg_table(
        list(specs.values()), dep_labels, coef_map,
        caption="Political Polarization and Market Reactions to Auditor Changes",
        label="tab:main",
        out_path=OUT_TABS / "tab02_main_results.tex",
        extra_rows=extra_rows,
    )


# ── Step 5: Heterogeneity by event type (Table 3) ─────────────────────────────

def run_event_type(df: pd.DataFrame) -> None:
    ctrl = " + ".join(CONTROLS)
    fe   = "C(year_str) + C(sic2_str)"

    dismissals  = df[df["dismissal"] == 1]
    resignations= df[df["dismissal"] == 0]
    qual_down   = df[df["quality_down"] == 1]
    qual_up     = df[df["quality_up"] == 1]

    specs = {
        # AbVol first (primary outcome)
        "m1": run_ols(f"abvol ~ competitive_std + {ctrl} + {fe}", dismissals),
        "m2": run_ols(f"abvol ~ competitive_std + {ctrl} + {fe}", resignations),
        # |CAR| second
        "m3": run_ols(f"absCar ~ competitive_std + {ctrl} + {fe}", dismissals),
        "m4": run_ols(f"absCar ~ competitive_std + {ctrl} + {fe}", resignations),
    }

    for k, m in specs.items():
        log.info("Event-type %s: competitive_std coef=%.4f p=%.3f", k,
                 m.params["competitive_std"], m.pvalues["competitive_std"])

    coef_map = {"competitive_std": r"\textit{Polarization}"}
    dep_labels = [r"AbVol\newline Dismissals",
                  r"AbVol\newline Non-dism.",
                  r"$|CAR|$\newline Dismissals",
                  r"$|CAR|$\newline Non-dism."]
    extra_rows = {
        "Year + Industry FE": ["Yes"] * 4,
        "Controls":           ["Yes"] * 4,
    }
    reg_table(
        list(specs.values()), dep_labels, coef_map,
        caption="Heterogeneity by Auditor Change Type",
        label="tab:event_type",
        out_path=OUT_TABS / "tab03_event_type.tex",
        extra_rows=extra_rows,
    )


# ── Step 6: Filing informativeness mechanism test (Table 4) ───────────────────

def run_ambiguity(df: pd.DataFrame) -> None:
    """
    Three-step mechanism test for the interpretation channel:

    Step 1: Build clean filing informativeness measures (specificity index
        and boilerplate-adjusted nonstandard word count, residualized on
        materiality correlates). Done in scripts 19/05.

    Step 2: Validate the informativeness measure against independent
        post-event uncertainty proxies. Six measures are used (return
        volatility, Parkinson high-low volatility, Amihud illiquidity,
        bid-ask spread, abnormal volume persistence, Roll implied spread).
        Prediction: low-info filings → higher post-event uncertainty.

    Step 3: Mechanism test. Does the polarization effect on post-event
        uncertainty amplify in low-info filings?
        Prediction: positive interaction coefficient for all measures.

    Reports a sign consistency (binomial) test across the six measures
    since individual tests are underpowered at N~670.
    """
    from scipy import stats as sp_stats

    ctrl = " + ".join(CONTROLS)
    fe   = "C(year_str) + C(sic2_str)"

    if "nonstd_resid" not in df.columns or df["nonstd_resid"].notna().sum() == 0:
        log.warning("No nonstd_resid variable; skipping informativeness test.")
        return
    if "uncertainty_composite" not in df.columns or df["uncertainty_composite"].notna().sum() == 0:
        log.warning("No uncertainty measures; skipping mechanism validation.")
        return

    # Use only rows with all key variables
    need = ["nonstd_resid", "low_info", "uncertainty_composite",
            "competitive_std"] + CONTROLS
    d = df.dropna(subset=need).copy()
    log.info("Informativeness mechanism test sample: N=%d", len(d))

    # Standardize continuous informativeness measure
    d["nonstd_std"] = (d["nonstd_resid"] - d["nonstd_resid"].mean()) / d["nonstd_resid"].std()
    d["pol_x_nonstd"]  = d["competitive_std"] * d["nonstd_std"]
    d["pol_x_lowinfo"] = d["competitive_std"] * d["low_info"]

    change_measures = ["vol", "parkinson", "amihud", "spread", "abvol", "roll"]
    measure_labels = {
        "vol":       "Return Vol.",
        "parkinson": "Parkinson",
        "amihud":    "Amihud",
        "spread":    "Bid-Ask",
        "abvol":     "Volume",
        "roll":      "Roll Spread",
    }

    # ── Step 2: Validation — Does low-info predict each uncertainty change? ──
    log.info("=== Validation: low-info -> Delta uncertainty ===")
    val_signs = []
    for m in change_measures:
        y = m + "_change_std"
        pre = "pre_" + m
        if y not in d.columns or d[y].notna().sum() == 0:
            continue
        # Use only rows with this specific measure
        dm = d.dropna(subset=[y, pre])
        mod = run_ols(f"{y} ~ low_info + {pre} + {ctrl} + {fe}", dm)
        b  = mod.params.get("low_info", float("nan"))
        p  = mod.pvalues.get("low_info", float("nan"))
        log.info("  Val %-10s: low_info coef=%+.4f p=%.3f N=%d", m, b, p, int(mod.nobs))
        val_signs.append(b > 0)
    n_pos_val = sum(val_signs)
    p_val_bin = 1 - sp_stats.binom.cdf(n_pos_val - 1, len(val_signs), 0.5)
    log.info("  Validation sign consistency: %d/%d positive, binomial p=%.4f",
             n_pos_val, len(val_signs), p_val_bin)

    # ── Step 3: Mechanism — Does pol × low_info predict each uncertainty change? ──
    log.info("=== Mechanism: pol x low_info -> Delta uncertainty ===")
    mech_signs = []
    for m in change_measures:
        y = m + "_change_std"
        pre = "pre_" + m
        if y not in d.columns or d[y].notna().sum() == 0:
            continue
        dm = d.dropna(subset=[y, pre])
        mod = run_ols(
            f"{y} ~ competitive_std + low_info + pol_x_lowinfo + {pre} + {ctrl} + {fe}",
            dm)
        b  = mod.params.get("pol_x_lowinfo", float("nan"))
        p  = mod.pvalues.get("pol_x_lowinfo", float("nan"))
        log.info("  Mech %-10s: pol_x_low coef=%+.4f p=%.3f N=%d", m, b, p, int(mod.nobs))
        mech_signs.append(b > 0)
    n_pos_mech = sum(mech_signs)
    p_mech_bin = 1 - sp_stats.binom.cdf(n_pos_mech - 1, len(mech_signs), 0.5)
    log.info("  Mechanism sign consistency: %d/%d positive, binomial p=%.4f",
             n_pos_mech, len(mech_signs), p_mech_bin)

    # ── Composite mechanism test ──
    log.info("=== Composite uncertainty mechanism tests ===")
    m_comp1 = run_ols(
        f"uncertainty_composite ~ nonstd_std + {ctrl} + {fe}", d)
    log.info("  Validation (cont):  nonstd_std coef=%+.4f p=%.3f",
             m_comp1.params["nonstd_std"], m_comp1.pvalues["nonstd_std"])
    m_comp2 = run_ols(
        f"uncertainty_composite ~ low_info + {ctrl} + {fe}", d)
    log.info("  Validation (bin):   low_info coef=%+.4f p=%.3f",
             m_comp2.params["low_info"], m_comp2.pvalues["low_info"])
    m_comp3 = run_ols(
        f"uncertainty_composite ~ competitive_std + low_info + pol_x_lowinfo + {ctrl} + {fe}",
        d)
    log.info("  Mechanism (bin):    pol_x_lowinfo coef=%+.4f p=%.3f",
             m_comp3.params["pol_x_lowinfo"], m_comp3.pvalues["pol_x_lowinfo"])
    m_comp4 = run_ols(
        f"uncertainty_composite ~ competitive_std + nonstd_std + pol_x_nonstd + {ctrl} + {fe}",
        d)
    log.info("  Mechanism (cont):   pol_x_nonstd coef=%+.4f p=%.3f",
             m_comp4.params["pol_x_nonstd"], m_comp4.pvalues["pol_x_nonstd"])

    # ── Build Table 4: the six mechanism specifications side by side ──
    # Present the validation row and the mechanism row for each measure,
    # plus the composite column.
    # We'll show the binary low_info specification as the main table.
    mech_models = []
    for m in change_measures:
        y = m + "_change_std"
        pre = "pre_" + m
        dm = d.dropna(subset=[y, pre])
        mod = run_ols(
            f"{y} ~ competitive_std + low_info + pol_x_lowinfo + {pre} + {ctrl} + {fe}",
            dm)
        mech_models.append(mod)

    # Add composite
    mech_models.append(m_comp3)

    coef_map = {
        "competitive_std": r"\textit{Polarization}",
        "low_info":        r"Low Info",
        "pol_x_lowinfo":   r"Polarization $\times$ Low Info",
    }
    dep_labels = [r"Return\newline Vol.",
                  r"Parkinson\newline H-L Vol.",
                  r"Amihud\newline Illiq.",
                  r"Bid-Ask\newline Spread",
                  r"Volume\newline Persistence",
                  r"Roll\newline Spread",
                  r"Composite\newline (PC1 avg)"]
    extra_rows = {
        "Pre-event level ctrl": ["Yes"] * 6 + ["No"],
        "Year + Industry FE":   ["Yes"] * 7,
        "Controls":             ["Yes"] * 7,
    }
    reg_table(
        mech_models, dep_labels, coef_map,
        caption=(r"Filing Informativeness, Polarization, and Post-Event "
                 r"Uncertainty. Dependent variables are standardized changes "
                 r"($\Delta$) in six independent post-event uncertainty "
                 r"measures (trading days $[+2, +20]$ minus $[-20, -2]$). "
                 r"\textit{Low Info} is an indicator for filings in the "
                 r"bottom tercile of boilerplate-adjusted nonstandard word "
                 r"count, residualized on dismissal, firm size, and year. "
                 r"A sign-consistency test across the six measures yields a "
                 r"binomial $p = 0.016$ for the positive "
                 r"Polarization~$\times$~Low~Info interaction."),
        label="tab:ambiguity",
        out_path=OUT_TABS / "tab04_ambiguity.tex",
        extra_rows=extra_rows,
    )


# ── Step 7: Robustness (Table 5) ──────────────────────────────────────────────

def run_robustness(df: pd.DataFrame) -> None:
    ctrl  = " + ".join(CONTROLS)
    fe    = "C(year_str) + C(sic2_str)"
    fe_st = "C(year_str) + C(sic2_str) + C(state_str)"

    # All polarization columns are already standardized in main() before this runs.

    # Build interaction: competitiveness × within-state county SD
    # Both variables are already standardized; the interaction captures
    # whether the effect concentrates in states that are BOTH competitive
    # AND internally polarized (high geographic partisan sorting).
    if "county_sd_std" in df.columns and df["county_sd_std"].notna().sum() > 0:
        df = df.copy()
        df["comp_x_ctysd"] = df["competitive_std"] * df["county_sd_std"]
        has_county_sd = True
    else:
        has_county_sd = False

    size_median = df["size"].median()
    df_small  = df[df["size"] < size_median].copy()
    df_county = df[df["county_comp_std"].notna()].copy()
    log.info("Small-firm cutoff (median log assets): %.3f; N_small=%d",
             size_median, len(df_small))

    # Run identical battery for both outcomes; AbVol first (primary), then |CAR|.
    all_models   = []
    all_labels   = []
    all_fe       = []
    all_st       = []
    all_cl       = []
    all_ctrl     = []
    all_samp     = []

    for depvar, dep_tag in [("abvol", "AbVol"), ("absCar", r"$|CAR|$")]:
        specs = {
            # (1) Baseline
            "m1": run_ols(f"{depvar} ~ competitive_std + {ctrl} + {fe}", df),
            # (2) Pres. ER index
            "m2": run_ols(f"{depvar} ~ er_pres_std + {ctrl} + {fe}", df),
            # (3) DW-NOMINATE
            "m3": run_ols(f"{depvar} ~ dw_std + {ctrl} + {fe}", df),
            # (4) State FEs
            "m4": run_ols(f"{depvar} ~ competitive_std + {ctrl} + {fe_st}", df),
            # (5) State-level clustered SEs
            "m5": run_ols(f"{depvar} ~ competitive_std + {ctrl} + {fe}", df,
                          cluster_var="state_str"),
            # (6) Two-way clustered SEs
            "m6": run_ols_twoway(f"{depvar} ~ competitive_std + {ctrl} + {fe}", df),
            # (7) County-level competitiveness
            "m7": run_ols(f"{depvar} ~ county_comp_std + {ctrl} + {fe}", df_county),
            # (8) Small-firm subsample
            "m8": run_ols(f"{depvar} ~ competitive_std + {ctrl} + {fe}", df_small),
        }
        if has_county_sd:
            # (9) County vote-share SD
            specs["m9"] = run_ols(f"{depvar} ~ county_sd_std + {ctrl} + {fe}", df)
            # (10) Competitiveness × County SD interaction
            specs["m10"] = run_ols(
                f"{depvar} ~ competitive_std + county_sd_std + comp_x_ctysd + {ctrl} + {fe}", df)

        # Log results
        for k, m in specs.items():
            pol_param = next((p for p in ["competitive_std", "er_pres_std",
                                           "dw_std", "county_comp_std",
                                           "county_sd_std", "comp_x_ctysd"]
                              if p in m.params), None)
            if pol_param:
                log.info("Robustness %s [%s]: %s coef=%.4f p=%.3f",
                         k, dep_tag, pol_param, m.params[pol_param], m.pvalues[pol_param])

        # Collect for table
        col_tags = ["Baseline", "Pres. ER", "DW State", "+State FE",
                    "State Clust", "2-Way Clust", "County", "Small Firm"]
        fe_vals  = ["Yes"] * 8
        st_vals  = ["No", "No", "No", "Yes", "No", "No", "No", "No"]
        cl_vals  = ["Firm"] * 4 + ["State", "Firm$\\times$State", "Firm", "Firm"]
        ct_vals  = ["Yes"] * 8
        sa_vals  = ["Full"] * 7 + ["Small firms"]

        if has_county_sd:
            col_tags += ["Cty Vote SD", "Interaction"]
            fe_vals  += ["Yes", "Yes"]
            st_vals  += ["No", "No"]
            cl_vals  += ["Firm", "Firm"]
            ct_vals  += ["Yes", "Yes"]
            sa_vals  += ["Full", "Full"]

        all_models.extend(specs.values())
        all_labels.extend([rf"{dep_tag}\newline {t}" for t in col_tags])
        all_fe   += fe_vals
        all_st   += st_vals
        all_cl   += cl_vals
        all_ctrl += ct_vals
        all_samp += sa_vals

    coef_map = {
        "competitive_std":  r"Polarization (baseline)",
        "er_pres_std":      r"Pres. ER Polarization",
        "dw_std":           r"DW-NOMINATE (state gap)",
        "county_comp_std":  r"County Polarization",
        "county_sd_std":    r"County Vote SD",
        "comp_x_ctysd":    r"Polarization $\times$ County SD",
    }

    # ── Build two-panel table ──
    # Split collected lists at the midpoint (AbVol panel, then CAR panel).
    n_per_panel = len(all_models) // 2
    panels = [
        ("Panel A: Abnormal Volume (AbVol)", "AbVol",
         all_models[:n_per_panel], all_labels[:n_per_panel],
         all_fe[:n_per_panel], all_st[:n_per_panel],
         all_cl[:n_per_panel], all_ctrl[:n_per_panel],
         all_samp[:n_per_panel]),
        (r"Panel B: Absolute Abnormal Return ($|CAR|$)", r"$|CAR|$",
         all_models[n_per_panel:], all_labels[n_per_panel:],
         all_fe[n_per_panel:], all_st[n_per_panel:],
         all_cl[n_per_panel:], all_ctrl[n_per_panel:],
         all_samp[n_per_panel:]),
    ]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Robustness and Alternative Proxies}")
    lines.append(r"\label{tab:robustness}")
    lines.append(r"\small")

    for panel_title, _, models, dep_labels_p, p_fe, p_st, p_cl, p_ct, p_sa in panels:
        ncols = len(models)
        # Strip dep-var prefix from column labels (already in panel title)
        short_labels = [lbl.split(r"\newline ")[-1] if r"\newline " in lbl else lbl
                        for lbl in dep_labels_p]

        lines.append("")
        lines.append(rf"\textit{{{panel_title}}}")
        lines.append(r"\vspace{2pt}")
        lines.append(r"\begin{tabular}{l" + "c" * ncols + "}")
        lines.append(r"\toprule")
        header = " & " + " & ".join(f"({i+1})" for i in range(ncols)) + r" \\"
        lines.append(header)
        dep_row = " & " + " & ".join(short_labels) + r" \\"
        lines.append(dep_row)
        lines.append(r"\midrule")

        # Coefficients
        for param, display in coef_map.items():
            coef_cells = []
            se_cells   = []
            for m in models:
                if param in m.params:
                    c = m.params[param]
                    p = m.pvalues[param]
                    s = m.bse[param]
                    coef_cells.append(f"{fmt(c)}{stars(p)}")
                    se_cells.append(f"({fmt(s)})")
                else:
                    coef_cells.append("")
                    se_cells.append("")
            lines.append(f"{display} & " + " & ".join(coef_cells) + r" \\")
            lines.append(" & " + " & ".join(se_cells) + r" \\")

        lines.append(r"\midrule")

        # Extra rows
        extra = {"Year + Industry FE": p_fe, "State FE": p_st,
                 "Cluster": p_cl, "Controls": p_ct, "Sample": p_sa}
        for row_label, vals in extra.items():
            lines.append(f"{row_label} & " + " & ".join(str(v) for v in vals) + r" \\")

        n_row  = "N & " + " & ".join(f"{int(m.nobs):,}" for m in models) + r" \\"
        r2_row = r"$R^2$ & " + " & ".join(fmt(m.rsquared) for m in models) + r" \\"
        lines.append(n_row)
        lines.append(r2_row)
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

    lines.append(r"\begin{flushleft}")
    lines.append(r"\footnotesize Notes: Standard errors clustered at the firm level "
                 r"in parentheses. $^{***}$, $^{**}$, $^{*}$ denote significance "
                 r"at 1\%, 5\%, 10\%.")
    lines.append(r"\end{flushleft}")
    lines.append(r"\end{table}")

    out_path = OUT_TABS / "tab05_robustness.tex"
    out_path.write_text("\n".join(lines))
    log.info("Table written: %s", out_path)


# ── Step 8: Permutation test for baseline coefficient (Table 7) ───────────────

def run_permutation_test(df: pd.DataFrame, n_perm: int = 5_000,
                         seed: int = 42) -> None:
    """
    Randomization/permutation test for both baseline outcomes (AbVol and |CAR|).

    Procedure:
      1. Estimate the baseline OLS coefficient on competitive_std for each outcome.
      2. Permute the competitive_std column across observations n_perm times
         (holding all other variables fixed) and re-estimate β each time.
      3. The permutation p-value is the fraction of permuted β̂ values that
         are ≥ the actual β̂ (one-sided, since we predict β > 0).
      4. Write a LaTeX table with actual coefficients, clustered SE p-values,
         and permutation p-values for both outcomes.

    This addresses concerns about finite-sample bias in the baseline t-test:
    if the actual β is genuinely non-zero, it should fall in the extreme tail
    of the permutation null distribution.

    Note: permuted draws use OLS without clustering (for speed); clustering
    affects the SE but not the coefficient, so the permutation distribution of
    β̂ is the same under either approach.
    """
    ctrl = " + ".join(CONTROLS)
    fe   = "C(year_str) + C(sic2_str)"

    # Run for both outcomes; AbVol first (primary)
    outcomes = [("abvol", "AbVol"), ("absCar", r"$|CAR|$")]
    actual_results = {}

    for depvar, label in outcomes:
        formula = f"{depvar} ~ competitive_std + {ctrl} + {fe}"
        m_actual = run_ols(formula, df)
        beta_actual = m_actual.params["competitive_std"]
        p_actual    = m_actual.pvalues["competitive_std"]
        se_actual   = m_actual.bse["competitive_std"]
        log.info("Permutation test [%s] — actual β=%.4f SE=%.4f p(clustered)=%.3f",
                 label, beta_actual, se_actual, p_actual)

        # Permutation draws
        rng = np.random.default_rng(seed)
        perm_betas = []
        comp_vals = df["competitive_std"].values.copy()
        perm_df = df.copy()
        for _ in range(n_perm):
            perm_df["competitive_std"] = rng.permutation(comp_vals)
            m_perm = smf.ols(formula, data=perm_df).fit()
            perm_betas.append(m_perm.params["competitive_std"])

        perm_betas = np.array(perm_betas)
        perm_p = (perm_betas >= beta_actual).mean()
        log.info("Permutation p-value [%s] (one-sided): %.4f  (β_actual=%.4f at "
                 "%.1f-th percentile)", label, perm_p, beta_actual,
                 100 * (1 - perm_p))

        actual_results[depvar] = {
            "beta": beta_actual, "se": se_actual,
            "p_clust": p_actual, "p_perm": perm_p,
        }

    # Write a two-column LaTeX table (AbVol, |CAR|)
    vol = actual_results["abvol"]
    car = actual_results["absCar"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Permutation Test for Baseline Coefficients}",
        r"\label{tab:permutation}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r" & AbVol & $|CAR|$ \\",
        r"\midrule",
        rf"$\hat{{\beta}}$ (Polarization) & {fmt(vol['beta'])} & {fmt(car['beta'])} \\",
        rf"SE (firm-clustered) & ({fmt(vol['se'])}) & ({fmt(car['se'])}) \\",
        rf"$p$-value (clustered $t$-test) & {fmt(vol['p_clust'], 3)} & {fmt(car['p_clust'], 3)} \\",
        rf"$p$-value (permutation, one-sided) & {fmt(vol['p_perm'], 3)} & {fmt(car['p_perm'], 3)} \\",
        rf"Permutations & {n_perm:,} & {n_perm:,} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}",
        r"\footnotesize Notes: The permutation $p$-value is the fraction of "
        r"5{,}000 random reshufflings of the \textit{Polarization} variable "
        r"that produce a coefficient estimate $\geq$ the actual $\hat{\beta}$. "
        r"Controls and fixed effects match the baseline specification "
        r"(Table~\ref{tab:main}, columns~2 and~4). "
        r"Permuted draws use OLS without clustering; the coefficient estimate "
        r"is invariant to the choice of standard error.",
        r"\end{flushleft}",
        r"\end{table}",
    ]
    out_path = OUT_TABS / "tab07_permutation.tex"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Permutation table written: %s", out_path)


# ── Step 9: Supplementary affective-polarization test (Table 6) ───────────────

def run_affective_test(df: pd.DataFrame) -> None:
    """
    Supplementary validation test using national affective polarization (ANES
    feeling thermometer differential) interacted with state partisan exposure
    (long-run presidential vote margin).

    Because ap_ft is national (same for all states in a year), it is absorbed
    by year fixed effects when entered alone. Identification comes from the
    differential response of firms in high-partisan-exposure states as national
    affective polarization rises over time.

    Specification (memo Eq. 3):
        Y_e = α + β1·competitive_std + β2·high_ambiguity + β3·competitive_std×high_ambiguity
            + β4·(AP×Exposure) + β5·(AP×Exposure×Ambiguity)
            + ΓX + year FE + industry FE + ε

    ap_ft and exposure_pres are standardized; their product is the interaction.
    The triple interaction tests whether the affective mechanism is concentrated
    in ambiguous signals, as predicted by the paper's mechanism.

    Only events with non-missing AP and exposure data are used.
    """
    ctrl = " + ".join(CONTROLS)
    fe   = "C(year_str) + C(sic2_str)"

    # Restrict to events with affective data (should be nearly all, given
    # the 2001-2024 AP series and the exposure covering all 50 states + DC)
    ap_df = df.dropna(subset=["ap_x_exposure"]).copy()
    log.info("Affective test sample: %d events", len(ap_df))

    # Full spec formula template
    full_formula = (
        "{depvar} ~ competitive_std + high_ambiguity"
        " + competitive_std:high_ambiguity"
        " + ap_x_exposure + ap_x_exp_x_amb + {ctrl} + {fe}"
    )

    specs = {
        # AbVol columns first (primary outcome)
        # (1) AbVol: ideological pol only (baseline)
        "m1": run_ols(
            f"abvol ~ competitive_std + {ctrl} + {fe}", ap_df
        ),
        # (2) AbVol: + AP × Exposure
        "m2": run_ols(
            f"abvol ~ competitive_std + ap_x_exposure + {ctrl} + {fe}", ap_df
        ),
        # (3) AbVol: full spec
        "m3": run_ols(
            full_formula.format(depvar="abvol", ctrl=ctrl, fe=fe), ap_df
        ),
        # |CAR| columns second (secondary outcome)
        # (4) |CAR|: ideological pol only (baseline)
        "m4": run_ols(
            f"absCar ~ competitive_std + {ctrl} + {fe}", ap_df
        ),
        # (5) |CAR|: + AP × Exposure
        "m5": run_ols(
            f"absCar ~ competitive_std + ap_x_exposure + {ctrl} + {fe}", ap_df
        ),
        # (6) |CAR|: full spec
        "m6": run_ols(
            full_formula.format(depvar="absCar", ctrl=ctrl, fe=fe), ap_df
        ),
    }

    for k, m in specs.items():
        for param in ["competitive_std", "ap_x_exposure", "ap_x_exp_x_amb"]:
            if param in m.params:
                log.info("Affective %s [%s]: coef=%.4f p=%.3f",
                         k, param, m.params[param], m.pvalues[param])

    coef_map = {
        "competitive_std":                   r"\textit{Polarization}",
        "high_ambiguity":                    r"High Ambiguity",
        "competitive_std:high_ambiguity":    r"\textit{Polarization} $\times$ Ambiguity",
        "ap_x_exposure":                 r"$AP \times Exposure$",
        "ap_x_exp_x_amb":                r"$AP \times Exposure \times$ Ambiguity",
    }
    dep_labels = [
        r"AbVol\newline Ideo. only",
        r"AbVol\newline +Affective",
        r"AbVol\newline Full",
        r"$|CAR|$\newline Ideo. only",
        r"$|CAR|$\newline +Affective",
        r"$|CAR|$\newline Full",
    ]
    extra_rows = {
        "Year + Industry FE": ["Yes"] * 6,
        "Controls":           ["Yes"] * 6,
    }
    reg_table(
        list(specs.values()), dep_labels, coef_map,
        caption=(
            r"Supplementary Affective Polarization Test. "
            r"Columns~(1)--(3) use abnormal volume; columns~(4)--(6) use $|CAR|$. "
            r"$AP \times Exposure$ is the product of the standardized national "
            r"ANES feeling-thermometer differential and the standardized "
            r"state long-run presidential vote margin. "
            r"Because the national AP series is absorbed by year fixed effects, "
            r"identification comes from the differential response of firms in "
            r"high-partisan-exposure states as national affective polarization "
            r"rises over time."
        ),
        label="tab:affective",
        out_path=OUT_TABS / "tab06_affective.tex",
        extra_rows=extra_rows,
    )


# ── Step 10: Analyst dispersion mechanism test (Table 8) ─────────────────────

def run_dispersion_interaction_test(df: pd.DataFrame) -> None:
    """
    Mechanism test using analyst forecast dispersion and coverage from IBES.

    The interpretation channel predicts that polarization amplifies disagreement
    specifically when investors face more uncertainty about how to interpret the
    disclosure.  Analyst forecast dispersion (stdev / |meanest|) is a pre-event,
    firm-level proxy for uncertainty about the firm's earnings trajectory — one
    dimension of the broader informational environment investors must interpret.
    Analyst coverage (numest) proxies for information supply.

    If the polarization effect operates via disagreement in interpretation rather
    than generic salience / attention, we expect:
      - Larger polarization effects among high-dispersion firms (investors already
        disagree about fundamentals; ambiguous disclosure amplifies this further).
      - Larger polarization effects among low-coverage firms (less pre-existing
        consensus about firm quality; disclosure harder to evaluate).

    Salience / attention stories make no prediction about heterogeneity across
    dispersion or coverage groups, conditional on event controls.

    Specifications:
      (1) Baseline with IBES subsample (for comparability)
      (2) Polarization × High-Dispersion interaction
      (3) Polarization × Low-Coverage interaction
      (4) Both interactions simultaneously
      (5) AbVol outcome, both interactions
    """
    ctrl = " + ".join(CONTROLS)
    fe   = "C(year_str) + C(sic2_str)"

    ibes_df = df.dropna(subset=["high_disp", "low_coverage"]).copy()
    log.info("Dispersion test sample: %d events (%d with both IBES measures)",
             len(df), len(ibes_df))

    specs = {
        "m1": run_ols(
            f"absCar ~ competitive_std + {ctrl} + {fe}", ibes_df
        ),
        "m2": run_ols(
            f"absCar ~ competitive_std + high_disp + competitive_std:high_disp"
            f" + {ctrl} + {fe}", ibes_df
        ),
        "m3": run_ols(
            f"absCar ~ competitive_std + low_coverage + competitive_std:low_coverage"
            f" + {ctrl} + {fe}", ibes_df
        ),
        "m4": run_ols(
            f"absCar ~ competitive_std + high_disp + competitive_std:high_disp"
            f" + low_coverage + competitive_std:low_coverage + {ctrl} + {fe}", ibes_df
        ),
        "m5": run_ols(
            f"abvol ~ competitive_std + high_disp + competitive_std:high_disp"
            f" + low_coverage + competitive_std:low_coverage + {ctrl} + {fe}", ibes_df
        ),
    }

    for k, m in specs.items():
        for param in ["competitive_std",
                      "competitive_std:high_disp",
                      "competitive_std:low_coverage"]:
            if param in m.params:
                log.info("Dispersion %s [%s]: coef=%.4f p=%.3f",
                         k, param, m.params[param], m.pvalues[param])

    coef_map = {
        "competitive_std":                  r"\textit{Polarization}",
        "high_disp":                        r"High Dispersion",
        "competitive_std:high_disp":        r"\textit{Polarization} $\times$ High Disp.",
        "low_coverage":                     r"Low Coverage",
        "competitive_std:low_coverage":     r"\textit{Polarization} $\times$ Low Coverage",
    }
    dep_labels = [
        r"$|CAR|$\newline Baseline",
        r"$|CAR|$\newline $\times$ Disp.",
        r"$|CAR|$\newline $\times$ Cov.",
        r"$|CAR|$\newline Both",
        r"AbVol\newline Both",
    ]
    extra_rows = {
        "Year + Industry FE": ["Yes"] * 5,
        "Controls":           ["Yes"] * 5,
    }
    reg_table(
        list(specs.values()), dep_labels, coef_map,
        caption=(
            r"Analyst Dispersion and Coverage: Mechanism Test. "
            r"\textit{High Dispersion} equals one if the firm's pre-event analyst "
            r"forecast dispersion (standard deviation divided by the absolute value "
            r"of the mean EPS forecast) exceeds the sample median. "
            r"\textit{Low Coverage} equals one if the number of analyst forecasts "
            r"(numest) is below the sample median. "
            r"Both indicators are constructed from the IBES annual consensus "
            r"as of the most recent statpers before the event date, "
            r"for the fiscal year ending in the year prior to the event."
        ),
        label="tab:dispersion",
        out_path=OUT_TABS / "tab08_dispersion_interaction.tex",
        extra_rows=extra_rows,
    )


# ── Step 11: Local investor relevance test (Table 8) ─────────────────────────

def run_local_bias_test(df: pd.DataFrame) -> None:
    """
    Test whether the polarization effect is stronger where local / retail
    investors are more likely to be the marginal trader.

    Motivation
    ----------
    The paper uses headquarters-state polarization as a proxy for the investor
    political environment. This mapping is valid if local investors drive the
    event-window reaction, but breaks down if nationally diversified institutions
    dominate trading (institutions are not politically local). This test directly
    addresses that concern: if local investor relevance amplifies the polarization
    effect, it validates the geographic identification strategy.

    Proxies for local / retail investor dominance
    ---------------------------------------------
    small_firm    : below-median log assets. Small firms attract less institutional
                    coverage and more retail-investor participation.
    low_turnover  : below-median pre-event average daily turnover (from script 12).
                    Low turnover signals less active institutional trading; the
                    marginal trader in these stocks is more likely to be local and
                    retail-oriented (Huberman 2001, Coval & Moskowitz 1999).

    Specifications (all use baseline controls + year + industry FE)
    ---------------------------------------------------------------
    (1) Polarization × Small-firm interaction
    (2) Polarization × Low-turnover interaction  (requires turnover data)
    (3) Both interactions simultaneously         (requires turnover data)
    (4)--(5) Same as (1)--(2) but for AbVol outcome

    Prediction: the interaction coefficients should be positive (stronger
    polarization effect in small / low-turnover firms). If both proxies point in
    the same direction, the evidence for the local-bias channel is robust to the
    choice of proxy.
    """
    ctrl = " + ".join(CONTROLS)
    fe   = "C(year_str) + C(sic2_str)"

    has_turnover = df["low_turnover"].notna().any()

    # --- Interaction specifications ---
    specs = {}

    # (1) |CAR| × Small firm
    specs["m1"] = run_ols(
        f"absCar ~ competitive_std + small_firm + competitive_std:small_firm"
        f" + {ctrl} + {fe}", df
    )

    # (2) |CAR| × Low turnover
    if has_turnover:
        to_df = df.dropna(subset=["low_turnover"]).copy()
        specs["m2"] = run_ols(
            f"absCar ~ competitive_std + low_turnover + competitive_std:low_turnover"
            f" + {ctrl} + {fe}", to_df
        )
        # (3) Both interactions simultaneously
        specs["m3"] = run_ols(
            f"absCar ~ competitive_std + small_firm + competitive_std:small_firm"
            f" + low_turnover + competitive_std:low_turnover + {ctrl} + {fe}", to_df
        )

    # (4) AbVol × Small firm
    specs["m4"] = run_ols(
        f"abvol ~ competitive_std + small_firm + competitive_std:small_firm"
        f" + {ctrl} + {fe}", df
    )

    # (5) AbVol × Low turnover
    if has_turnover:
        specs["m5"] = run_ols(
            f"abvol ~ competitive_std + low_turnover + competitive_std:low_turnover"
            f" + {ctrl} + {fe}", to_df
        )

    for k, m in specs.items():
        for param in ["competitive_std",
                      "competitive_std:small_firm",
                      "competitive_std:low_turnover"]:
            if param in m.params:
                log.info("Local bias %s [%s]: coef=%.4f p=%.3f N=%d",
                         k, param, m.params[param], m.pvalues[param], int(m.nobs))

    coef_map = {
        "competitive_std":                  r"\textit{Polarization}",
        "small_firm":                       r"Small Firm",
        "competitive_std:small_firm":       r"\textit{Polarization} $\times$ Small",
        "low_turnover":                     r"Low Turnover",
        "competitive_std:low_turnover":     r"\textit{Polarization} $\times$ Low Turn.",
    }

    if has_turnover:
        dep_labels = [
            r"$|CAR|$\newline $\times$ Small",
            r"$|CAR|$\newline $\times$ Turn.",
            r"$|CAR|$\newline Both",
            r"AbVol\newline $\times$ Small",
            r"AbVol\newline $\times$ Turn.",
        ]
    else:
        dep_labels = [
            r"$|CAR|$\newline $\times$ Small",
            r"AbVol\newline $\times$ Small",
        ]

    extra_rows = {
        "Year + Industry FE": ["Yes"] * len(specs),
        "Controls":           ["Yes"] * len(specs),
    }

    caption = (
        r"Local Investor Relevance Test. "
        r"\textit{Small Firm} equals one if the firm's log total assets is below "
        r"the estimation-sample median. "
        r"\textit{Low Turnover} equals one if mean pre-event daily share turnover "
        r"(average of vol\,/\,(shrout\,$\times$\,1{,}000) over the 365 to 22 calendar "
        r"days before the event) is below the estimation-sample median; "
        r"events with fewer than 60 valid trading days are excluded from columns "
        r"using this variable. "
        r"Both proxies capture settings where local and retail investors "
        r"are more likely to be the marginal trader. "
        r"The prediction from the local-bias channel is that "
        r"\textit{Polarization} $\times$ Small and "
        r"\textit{Polarization} $\times$ Low Turnover are both positive."
    )

    reg_table(
        list(specs.values()), dep_labels, coef_map,
        caption=caption,
        label="tab:local_bias",
        out_path=OUT_TABS / "tab08_local_bias.tex",
        extra_rows=extra_rows,
    )


# ── Step 12: Audit credibility interaction test (Table 10) ────────────────────

def run_audit_credibility_test(df: pd.DataFrame) -> None:
    """
    Test whether the polarization effect is stronger when audit credibility
    is more salient, using firm-level accounting quality and distress measures.

    Prediction: the interaction coefficient should be positive for each
    moderator --- polarization amplifies investor disagreement most when the
    filing forces investors to evaluate the reliability of accounting
    oversight, not when the auditor change is a routine formality.

    Moderators (all from script 16):
        high_distress  : below-median Altman Z-score (more financial distress
                         → audit credibility more salient)
        high_daccruals : above-median |DA| from Modified Jones model (higher
                         accounting opacity → more room for politically shaped
                         inferences about audit quality)
        gc_opinion     : prior going-concern modification (direct audit
                         credibility flag; very small N, likely underpowered)

    Six columns:
        (1) |CAR| × High Distress
        (2) |CAR| × High Discretionary Accruals
        (3) |CAR| × GC Opinion
        (4) |CAR| × Distress + Accruals (both interactions simultaneously)
        (5) AbVol × Distress
        (6) AbVol × Accruals
    """
    ctrl = " + ".join(CONTROLS)
    fe   = "C(year_str) + C(sic2_str)"

    specs = {}

    # (1) |CAR| × High Distress
    z_df = df.dropna(subset=["high_distress"]).copy()
    log.info("Audit credibility sample: distress N=%d", len(z_df))
    specs["m1"] = run_ols(
        f"absCar ~ competitive_std + high_distress"
        f" + competitive_std:high_distress + {ctrl} + {fe}", z_df
    )

    # (2) |CAR| × High Discretionary Accruals
    da_df = df.dropna(subset=["high_daccruals"]).copy()
    log.info("Audit credibility sample: accruals N=%d", len(da_df))
    specs["m2"] = run_ols(
        f"absCar ~ competitive_std + high_daccruals"
        f" + competitive_std:high_daccruals + {ctrl} + {fe}", da_df
    )

    # (3) |CAR| × GC Opinion (may be very small N with gc_opinion=1)
    gc_df = df.dropna(subset=["gc_opinion"]).copy()
    n_gc1 = int(gc_df["gc_opinion"].sum())
    log.info("Audit credibility sample: GC N=%d (gc_opinion=1: %d)", len(gc_df), n_gc1)
    if n_gc1 >= 10:
        specs["m3"] = run_ols(
            f"absCar ~ competitive_std + gc_opinion"
            f" + competitive_std:gc_opinion + {ctrl} + {fe}", gc_df
        )
    else:
        log.warning("Too few GC opinions (%d); skipping column 3", n_gc1)

    # (4) |CAR| × Distress + Accruals (both simultaneously)
    both_df = df.dropna(subset=["high_distress", "high_daccruals"]).copy()
    specs["m4"] = run_ols(
        f"absCar ~ competitive_std + high_distress + competitive_std:high_distress"
        f" + high_daccruals + competitive_std:high_daccruals + {ctrl} + {fe}", both_df
    )

    # (5) AbVol × High Distress
    specs["m5"] = run_ols(
        f"abvol ~ competitive_std + high_distress"
        f" + competitive_std:high_distress + {ctrl} + {fe}", z_df
    )

    # (6) AbVol × High Accruals
    specs["m6"] = run_ols(
        f"abvol ~ competitive_std + high_daccruals"
        f" + competitive_std:high_daccruals + {ctrl} + {fe}", da_df
    )

    # Log key results
    for k, m in specs.items():
        for param in ["competitive_std",
                      "competitive_std:high_distress",
                      "competitive_std:high_daccruals",
                      "competitive_std:gc_opinion"]:
            if param in m.params:
                log.info("Audit cred %s [%s]: coef=%.4f p=%.3f N=%d",
                         k, param, m.params[param], m.pvalues[param],
                         int(m.nobs))

    coef_map = {
        "competitive_std":                   r"\textit{Polarization}",
        "high_distress":                     r"High Distress",
        "competitive_std:high_distress":     r"\textit{Polarization} $\times$ High Distress",
        "high_daccruals":                    r"High Disc.\ Accruals",
        "competitive_std:high_daccruals":    r"\textit{Polarization} $\times$ High Accruals",
        "gc_opinion":                        r"Going Concern",
        "competitive_std:gc_opinion":        r"\textit{Polarization} $\times$ Going Concern",
    }

    model_list = [specs[k] for k in ["m1", "m2"] +
                  (["m3"] if "m3" in specs else []) +
                  ["m4", "m5", "m6"]]
    n_models = len(model_list)

    if "m3" in specs:
        dep_labels = [
            r"$|CAR|$\newline $\times$ Distress",
            r"$|CAR|$\newline $\times$ Accruals",
            r"$|CAR|$\newline $\times$ GC",
            r"$|CAR|$\newline Both",
            r"AbVol\newline $\times$ Distress",
            r"AbVol\newline $\times$ Accruals",
        ]
    else:
        dep_labels = [
            r"$|CAR|$\newline $\times$ Distress",
            r"$|CAR|$\newline $\times$ Accruals",
            r"$|CAR|$\newline Both",
            r"AbVol\newline $\times$ Distress",
            r"AbVol\newline $\times$ Accruals",
        ]

    extra_rows = {
        "Year + Industry FE": ["Yes"] * n_models,
        "Controls":           ["Yes"] * n_models,
    }

    caption = (
        r"Audit Credibility Salience: Interaction Tests. "
        r"\textit{High Distress} equals one if the firm's Altman Z-score "
        r"(computed from Compustat fiscal-year data for the year prior to the event) "
        r"is below the estimation-sample median; financially distressed firms face greater "
        r"scrutiny of audit quality. "
        r"\textit{High Disc.\ Accruals} equals one if the absolute value of Modified "
        r"Jones model discretionary accruals exceeds the sample median; high accruals "
        r"signal greater accounting opacity and make audit oversight more salient. "
        r"\textit{Going Concern} equals one if the prior-year audit opinion "
        r"included a going-concern modification. "
        r"The interpretation channel predicts positive interaction coefficients: "
        r"polarization should amplify investor disagreement specifically when the "
        r"auditor-change filing forces investors to evaluate the reliability of "
        r"accounting oversight."
    )

    reg_table(
        model_list, dep_labels, coef_map,
        caption=caption,
        label="tab:audit_credibility",
        out_path=OUT_TABS / "tab10_audit_credibility.tex",
        extra_rows=extra_rows,
    )


# ── Step 13: Short interest disagreement test (Table 11) ─────────────────────

def run_short_interest_test(df: pd.DataFrame) -> None:
    """
    Test the disagreement channel using short interest data.

    Two complementary tests:

    A. Moderator: Is the polarization effect on |CAR| stronger when pre-event
       short interest is high? High SI signals active disagreement about firm
       value; the interaction should be positive if polarization amplifies
       heterogeneous interpretation.

    B. Outcome: Does polarization predict changes in the short interest ratio
       around the auditor-change event? If polarized investors form more
       dispersed posteriors, pessimistic investors should increase short
       positions. This test uses the full sample (no splitting) and captures
       disagreement directly via revealed trading behavior.

    Columns:
        (1) |CAR| × High SI          — moderator (interaction)
        (2) |CAR| × High SI + ctrls  — with controls
        (3) ΔSI ~ Pol               — outcome (SI change)
        (4) ΔSI ~ Pol + ctrls       — outcome with controls
    """
    ctrl = " + ".join(CONTROLS)
    fe   = "C(year_str) + C(sic2_str)"

    # --- Moderator tests ---
    si_df = df.dropna(subset=["high_si"]).copy()
    log.info("Short interest moderator sample: N=%d", len(si_df))

    specs = {}

    # (1) |CAR| × High SI — no controls
    specs["m1"] = run_ols(
        f"absCar ~ competitive_std + high_si"
        f" + competitive_std:high_si + C(year_str)", si_df
    )

    # (2) |CAR| × High SI — with controls + FE
    specs["m2"] = run_ols(
        f"absCar ~ competitive_std + high_si"
        f" + competitive_std:high_si + {ctrl} + {fe}", si_df
    )

    # --- Outcome tests (SI change as dependent variable) ---
    chg_df = df.dropna(subset=["si_change"]).copy()
    log.info("Short interest outcome sample: N=%d", len(chg_df))

    # (3) ΔSI ~ Pol — year FE only
    specs["m3"] = run_ols(
        f"si_change ~ competitive_std + C(year_str)", chg_df
    )

    # (4) ΔSI ~ Pol — controls + FE
    specs["m4"] = run_ols(
        f"si_change ~ competitive_std + {ctrl} + {fe}", chg_df
    )

    for k, m in specs.items():
        for param in ["competitive_std", "competitive_std:high_si"]:
            if param in m.params:
                log.info("Short interest %s [%s]: coef=%.6f p=%.3f N=%d",
                         k, param, m.params[param], m.pvalues[param],
                         int(m.nobs))

    coef_map = {
        "competitive_std":              r"\textit{Polarization}",
        "high_si":                      r"High Short Interest",
        "competitive_std:high_si":      r"\textit{Polarization} $\times$ High SI",
    }
    dep_labels = [
        r"$|CAR|$\newline No ctrls",
        r"$|CAR|$\newline Full ctrls",
        r"$\Delta SI$\newline No ctrls",
        r"$\Delta SI$\newline Full ctrls",
    ]
    extra_rows = {
        "Year FE":             ["Yes", "Yes", "Yes", "Yes"],
        "Industry FE":         ["No",  "Yes", "No",  "Yes"],
        "Controls":            ["No",  "Yes", "No",  "Yes"],
    }

    caption = (
        r"Short Interest and the Disagreement Channel. "
        r"Columns~(1)--(2) test whether the polarization effect on $|CAR(-1,+1)|$ "
        r"is stronger for firms with above-median pre-event short interest ratio, "
        r"measured over calendar days $[-90,-15]$ before the event. "
        r"\textit{High Short Interest} equals one if the pre-event short interest "
        r"ratio (split-adjusted shares short divided by shares outstanding) exceeds "
        r"the sample median. "
        r"Columns~(3)--(4) use the change in short interest ratio as the dependent "
        r"variable, computed as the difference between the first post-event report "
        r"(calendar days $[+15,+90]$) and the pre-event mean. "
        r"A positive coefficient on \textit{Polarization} in columns~(3)--(4) "
        r"indicates that auditor-change events in more polarized states trigger "
        r"larger increases in short selling, consistent with polarization amplifying "
        r"investor disagreement."
    )

    reg_table(
        list(specs.values()), dep_labels, coef_map,
        caption=caption,
        label="tab:short_interest",
        out_path=OUT_TABS / "tab11_short_interest.tex",
        extra_rows=extra_rows,
    )


# ── Step 13b: Institutional ownership / retail fraction test ─────────────────

def run_institutional_ownership_test(df: pd.DataFrame) -> None:
    """
    Test whether the polarization effect is stronger among firms with
    higher retail ownership (= lower institutional ownership).

    Retail investors are more likely to be influenced by local political
    environment. If polarization amplifies heterogeneous interpretation
    primarily through retail investors, the interaction of Pol x high_retail
    should be positive.

    Specifications:
        (1) |CAR| ~ Pol + high_retail + Pol x high_retail + year FE
        (2) |CAR| ~ Pol + high_retail + Pol x high_retail + controls + FE
        (3) AbVol ~ Pol + high_retail + Pol x high_retail + year FE
        (4) AbVol ~ Pol + high_retail + Pol x high_retail + controls + FE

    No table is generated — results are logged only for now.
    """
    ctrl = " + ".join(CONTROLS)
    fe   = "C(year_str) + C(sic2_str)"

    ret_df = df.dropna(subset=["high_retail"]).copy()
    log.info("Institutional ownership moderator sample: N=%d", len(ret_df))

    if len(ret_df) < 50:
        log.warning("Too few observations with institutional ownership data; skipping.")
        return

    # (1) |CAR| ~ Pol + high_retail + Pol x high_retail — year FE only
    m1 = run_ols(
        f"absCar ~ competitive_std + high_retail"
        f" + competitive_std:high_retail + C(year_str)", ret_df
    )

    # (2) |CAR| ~ Pol + high_retail + Pol x high_retail — full controls + FE
    m2 = run_ols(
        f"absCar ~ competitive_std + high_retail"
        f" + competitive_std:high_retail + {ctrl} + {fe}", ret_df
    )

    # (3) AbVol ~ Pol + high_retail + Pol x high_retail — year FE only
    m3 = run_ols(
        f"abvol ~ competitive_std + high_retail"
        f" + competitive_std:high_retail + C(year_str)", ret_df
    )

    # (4) AbVol ~ Pol + high_retail + Pol x high_retail — full controls + FE
    m4 = run_ols(
        f"abvol ~ competitive_std + high_retail"
        f" + competitive_std:high_retail + {ctrl} + {fe}", ret_df
    )

    # Log results (no table for now)
    specs = {"m1_absCar_noCtrl": m1, "m2_absCar_fullCtrl": m2,
             "m3_abvol_noCtrl": m3, "m4_abvol_fullCtrl": m4}
    for label, m in specs.items():
        for param in ["competitive_std", "high_retail", "competitive_std:high_retail"]:
            if param in m.params:
                log.info("InstOwn %s [%s]: coef=%.6f  se=%.6f  p=%.3f  N=%d",
                         label, param, m.params[param], m.bse[param],
                         m.pvalues[param], int(m.nobs))

    # Log median ownership stats for context
    med_retail = df.loc[df["retail_pct"].notna(), "retail_pct"].median()
    mean_inst  = df.loc[df["inst_own_pct"].notna(), "inst_own_pct"].mean()
    log.info("InstOwn: median retail_pct=%.4f  mean inst_own_pct=%.4f",
             med_retail, mean_inst)


# ── Step 14: Post-event CAR reversal test (Table 9) ──────────────────────────

def run_reversal_test(df: pd.DataFrame) -> None:
    """
    Test whether the polarization effect reflects disagreement / overreaction by
    checking for post-event price reversals.

    If headquarters-state polarization causes investors to overreact at the
    event date (inflated |CAR| due to disagreement-driven trading), prices
    should partially mean-revert in the weeks after the event. A reversal is
    consistent with the interpretation that polarized investors disagree about
    the signal's valence, causing temporary price pressure that subsequently
    unwinds.

    Specification:
        CAR[+2,+T]_i = β₁·competitive_std_i + β₂·car_m1p1_i
                     + γ'·X_i + FE + ε_i

    where CAR[+2,+T] is the signed market-model CAR for trading days [+2,+T],
    car_m1p1 is the event-window signed CAR (-1,+1), competitive_std is the
    standardized polarization proxy (one-SD units), and X is the standard
    control vector.

    Controlling for car_m1p1 partials out the tendency of large initial
    reactions to mean-revert regardless of polarization (price-pressure
    effect). β₁ < 0 is the prediction: high-polarization events show more
    post-event reversal beyond that mechanical relationship.

    Four columns:
        (1) [+2,+20] short window, controlling for signed event-window CAR
        (2) [+2,+60] long window, controlling for signed event-window CAR
        (3) [+2,+20] controlling for |CAR| magnitude instead of signed CAR
        (4) [+2,+60] controlling for |CAR| magnitude instead of signed CAR

    Columns (3)-(4) use absCar to test whether polarization predicts reversal
    beyond what the magnitude of initial price movement would imply.
    """
    ctrl = " + ".join(CONTROLS)
    fe   = "C(year_str) + C(sic2_str)"

    # Drop missing post-event CAR
    rev_df = df.dropna(subset=["car_p2p20", "car_p2p60"]).copy()
    log.info("Reversal test sample: %d events", len(rev_df))

    specs = {
        "m1": run_ols(
            f"car_p2p20 ~ competitive_std + car + {ctrl} + {fe}", rev_df
        ),
        "m2": run_ols(
            f"car_p2p60 ~ competitive_std + car + {ctrl} + {fe}", rev_df
        ),
        "m3": run_ols(
            f"car_p2p20 ~ competitive_std + absCar + {ctrl} + {fe}", rev_df
        ),
        "m4": run_ols(
            f"car_p2p60 ~ competitive_std + absCar + {ctrl} + {fe}", rev_df
        ),
    }

    for k, m in specs.items():
        for param in ["competitive_std", "car", "absCar"]:
            if param in m.params:
                log.info("Reversal %s [%s]: coef=%.4f p=%.3f N=%d",
                         k, param, m.params[param], m.pvalues[param],
                         int(m.nobs))

    coef_map = {
        "competitive_std": r"\textit{Polarization}",
        "car":             r"$CAR(-1,+1)$",
        "absCar":          r"$|CAR(-1,+1)|$",
    }
    dep_labels = [
        r"$CAR[+2,+20]$\newline Signed ctrl.",
        r"$CAR[+2,+60]$\newline Signed ctrl.",
        r"$CAR[+2,+20]$\newline $|CAR|$ ctrl.",
        r"$CAR[+2,+60]$\newline $|CAR|$ ctrl.",
    ]
    extra_rows = {
        "Window":             ["[+2,+20]", "[+2,+60]", "[+2,+20]", "[+2,+60]"],
        "Event-CAR control":  ["Signed", "Signed", "$|CAR|$", "$|CAR|$"],
        "Year + Industry FE": ["Yes"] * 4,
        "Controls":           ["Yes"] * 4,
    }

    caption = (
        r"Post-Event CAR Reversal Test. "
        r"The dependent variable is the signed market-model CAR computed over "
        r"trading days $[+2,+T]$ relative to the 8-K filing date, where $T=20$ "
        r"(approximately one month) or $T=60$ (approximately three months). "
        r"\textit{Polarization} is the standardized headquarters-state "
        r"presidential Esteban-Ray index. "
        r"Columns (1)--(2) control for the signed event-window $CAR(-1,+1)$ "
        r"to partial out mechanical mean reversion; "
        r"columns (3)--(4) control for $|CAR(-1,+1)|$. "
        r"A negative \textit{Polarization} coefficient is consistent with "
        r"short-lived disagreement-driven overreaction in high-polarization states. "
        r"Market model parameters ($\hat\alpha$, $\hat\beta$) are the same as in "
        r"the main analysis (estimation window $[-252,-46]$ trading days). "
        r"Standard errors clustered at the firm level."
    )

    reg_table(
        list(specs.values()), dep_labels, coef_map,
        caption=caption,
        label="tab:reversal",
        out_path=OUT_TABS / "tab09_reversal.tex",
        extra_rows=extra_rows,
    )


# ── Step 15: Regulatory shock tests (SOX/PCAOB) ──────────────────────────────

def run_regulatory_shock_test(df: pd.DataFrame) -> None:
    """
    Test whether the polarization effect differs before vs. after SOX/PCAOB.

    SOX was enacted July 2002; the PCAOB became operational in 2003-2004.
    We define post_sox = 1 for events in 2003+ (first full year after enactment).

    Three specifications per outcome:
      (1) Pre-SOX subsample only
      (2) Post-SOX subsample only
      (3) Full sample with Pol × PostSOX interaction

    This tests whether the institutional-trust channel (which invokes trust in
    the PCAOB) operates differently before the PCAOB existed.

    Caveat: the pre-SOX period (2001-2002) coincides with the Enron/Andersen
    crisis, which independently affected market reactions to auditor changes.
    """
    ctrl = " + ".join(CONTROLS)
    fe   = "C(year_str) + C(sic2_str)"

    df = df.copy()
    df["post_sox"] = (df["event_year"] >= 2003).astype(int)
    df["pol_x_postsox"] = df["competitive_std"] * df["post_sox"]

    pre  = df[df["post_sox"] == 0].copy()
    post = df[df["post_sox"] == 1].copy()

    log.info("SOX split: pre-SOX N=%d (2001-2002), post-SOX N=%d (2003+)",
             len(pre), len(post))

    # |CAR| specs
    m1_car = run_ols(f"absCar ~ competitive_std + {ctrl} + {fe}", pre)
    m2_car = run_ols(f"absCar ~ competitive_std + {ctrl} + {fe}", post)
    m3_car = run_ols(
        f"absCar ~ competitive_std + post_sox + pol_x_postsox + {ctrl} + {fe}",
        df)

    # AbVol specs
    m1_vol = run_ols(f"abvol ~ competitive_std + {ctrl} + {fe}", pre)
    m2_vol = run_ols(f"abvol ~ competitive_std + {ctrl} + {fe}", post)
    m3_vol = run_ols(
        f"abvol ~ competitive_std + post_sox + pol_x_postsox + {ctrl} + {fe}",
        df)

    for label, m in [("Pre-SOX |CAR|", m1_car), ("Post-SOX |CAR|", m2_car),
                     ("Full |CAR| interaction", m3_car),
                     ("Pre-SOX AbVol", m1_vol), ("Post-SOX AbVol", m2_vol),
                     ("Full AbVol interaction", m3_vol)]:
        pol_p = m.params.get("competitive_std", float("nan"))
        pol_pv = m.pvalues.get("competitive_std", float("nan"))
        log.info("SOX test %s: competitive_std coef=%.4f p=%.3f N=%d",
                 label, pol_p, pol_pv, int(m.nobs))
        if "pol_x_postsox" in m.params:
            log.info("  pol_x_postsox coef=%.4f p=%.3f",
                     m.params["pol_x_postsox"], m.pvalues["pol_x_postsox"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== 05_merge_and_estimate.py  start ===")

    # Check required inputs exist (IBES file is optional — tested separately below)
    for f in [CRSP_FILE, POL_FILE, COMP_FILE, DW_FILE, EXPOSURE_FILE, AP_FILE, PRES_POL_FILE]:
        if not f.exists():
            raise FileNotFoundError(
                f"Missing input: {f}\n"
                "Run scripts 02, 03, and 04 first."
            )

    # Merge
    df = load_and_merge()
    df = apply_sample_filters(df)

    # County-level competitiveness (robustness column 9).
    # Merge before standardization loop so county_comp_std is computed with other measures.
    county_file = ROOT / "Data/Processed/analysis_sample_county.parquet"
    if county_file.exists():
        county_df = pd.read_parquet(county_file, columns=["acc_nodash", "county_comp"])
        df = df.merge(county_df, on="acc_nodash", how="left")
        log.info("County competitiveness merged: %d / %d events matched",
                 df["county_comp"].notna().sum(), len(df))
    else:
        df["county_comp"] = np.nan
        log.warning("analysis_sample_county.parquet not found; county column will be NaN")

    # Standardize polarization measures on the final estimation sample (mean=0, sd=1).
    # Done after filtering so the coefficient is interpretable as a 1-SD effect.
    for raw_col, std_col in [
        ("pol_er_alpha10",    "pol_std"),
        ("pol_er_alpha08",    "pol_std_a08"),
        ("pol_er_alpha12",    "pol_std_a12"),
        ("dw_cross_party_gap","dw_std"),
        ("dw_national_gap",   "dw_national_std"),
        ("exposure_pres",     "exposure_std"),
        ("ap_ft",             "ap_std"),
        ("er_pres",           "er_pres_std"),
        ("margin",            "margin_std"),
        ("county_comp",       "county_comp_std"),
        ("county_sd",         "county_sd_std"),
        ("incorp_pol",        "incorp_pol_std"),
    ]:
        if raw_col in df.columns and df[raw_col].notna().sum() > 0:
            mu  = df[raw_col].mean()
            sig = df[raw_col].std()
            df[std_col] = (df[raw_col] - mu) / sig
        else:
            df[std_col] = np.nan

    # Election-year indicator: all federal election years (even calendar years).
    # Prediction: political salience peaks in election years, so the same level of
    # state polarization should generate stronger reactions when national political
    # attention is highest. Interaction competitive_std:election_year tests this.
    df["election_year"] = (df["event_year"] % 2 == 0).astype(int)
    log.info("Election-year events: %d / %d", df["election_year"].sum(), len(df))

    # Competitiveness = −margin: positive scale so higher = more competitive/polarized.
    # Algebraically identical to margin regression; coefficient sign flips to positive,
    # making the table read: "one-SD increase in competitiveness → larger |CAR|."
    df["competitive_std"] = -df["margin_std"]

    # Affective polarization interaction term: AP_t × Exposure_s.
    # This is the identifying variation — differential response of firms in
    # high-exposure states as national affective polarization rises over time.
    # ap_ft alone is absorbed by year FEs and therefore not entered separately.
    df["ap_x_exposure"] = df["ap_std"] * df["exposure_std"]
    df["ap_x_exp_x_amb"] = df["ap_x_exposure"] * df["high_ambiguity"]
    log.info("AP x Exposure non-null: %d", df["ap_x_exposure"].notna().sum())

    # IBES analyst dispersion and coverage (optional; from 11_build_ibes.py).
    # Merge on gvkey × comp_year (comp_year = event_year - 1, already in df).
    # If the file does not exist, dispersion columns are NaN and the
    # dispersion test is skipped rather than raising an error.
    if IBES_FILE.exists():
        ibes = pd.read_parquet(IBES_FILE)
        df = df.merge(ibes, on=["gvkey", "comp_year"], how="left")
        n_ibes = df["analyst_coverage"].notna().sum()
        log.info("IBES merge: %d / %d events matched (coverage non-null)",
                 n_ibes, len(df))

        # Construct above/below-median indicators on the estimation sample.
        # Use the subsample that has non-missing values for the median calculation
        # so that the 50th percentile is well-defined.
        med_disp = df["disp_scaled"].median()
        med_cov  = df["analyst_coverage"].median()
        df["high_disp"]    = (df["disp_scaled"]       > med_disp).astype(float)
        df["low_coverage"] = (df["analyst_coverage"]   < med_cov ).astype(float)
        # Keep indicators as NaN where the underlying IBES measure is missing
        df.loc[df["disp_scaled"].isna(),       "high_disp"]    = np.nan
        df.loc[df["analyst_coverage"].isna(),  "low_coverage"] = np.nan

        log.info("IBES: median disp_scaled=%.4f  median coverage=%.1f",
                 med_disp, med_cov)
        log.info("IBES: high_disp N=%d  low_coverage N=%d",
                 df["high_disp"].notna().sum(), df["low_coverage"].notna().sum())
    else:
        df["analyst_coverage"] = np.nan
        df["disp_raw"]         = np.nan
        df["disp_scaled"]      = np.nan
        df["high_disp"]        = np.nan
        df["low_coverage"]     = np.nan
        log.warning("ibes_dispersion.parquet not found; run 11_build_ibes.py "
                    "to enable the dispersion mechanism test (Table 8).")

    # Pre-event share turnover (from 12_build_turnover.py).
    # Merge on gvkey × event_date. Construct low_turnover indicator.
    if TURNOVER_FILE.exists():
        turn = pd.read_parquet(TURNOVER_FILE)
        turn["event_date"] = pd.to_datetime(turn["event_date"])
        df = df.merge(turn[["gvkey", "event_date", "turnover_pre", "turnover_days"]],
                      on=["gvkey", "event_date"], how="left")
        n_turn = df["turnover_pre"].notna().sum()
        log.info("Turnover merge: %d / %d events matched", n_turn, len(df))

        med_turn = df["turnover_pre"].median()
        df["low_turnover"] = (df["turnover_pre"] < med_turn).astype(float)
        df.loc[df["turnover_pre"].isna(), "low_turnover"] = np.nan
        log.info("Turnover: median=%.6f  low_turnover N=%d",
                 med_turn, df["low_turnover"].notna().sum())
    else:
        df["turnover_pre"] = np.nan
        df["turnover_days"] = np.nan
        df["low_turnover"] = np.nan
        log.warning("pre_event_turnover.parquet not found; run 12_build_turnover.py "
                    "to enable turnover columns in the local bias test.")

    # Small-firm indicator (below-median log assets in estimation sample)
    med_size = df["size"].median()
    df["small_firm"] = (df["size"] < med_size).astype(float)
    log.info("Small-firm indicator: median size=%.3f  N_small=%d",
             med_size, df["small_firm"].sum())

    # Post-event CARs (from 14_build_post_event_car.py).
    # Merge on permno × event_date. Columns: car_p2p20, car_p2p60, n_days_20, n_days_60.
    # If the file does not exist, reversal test is skipped at runtime.
    if POST_CAR_FILE.exists():
        post_car = pd.read_parquet(POST_CAR_FILE)
        post_car["event_date"] = pd.to_datetime(post_car["event_date"])
        post_car["permno"] = post_car["permno"].astype(int)
        # Deduplicate on (permno, event_date) before merging.
        # post_event_car.parquet may have been built from a differently-filtered
        # analysis_sample; duplicate (permno, event_date) pairs would expand N.
        # CARs are identical across rows for the same (permno, event_date),
        # so keeping first is safe.
        n_pre_dedup = len(post_car)
        post_car = post_car.drop_duplicates(subset=["permno", "event_date"], keep="first")
        if len(post_car) < n_pre_dedup:
            log.warning("post_event_car: dropped %d duplicate (permno, event_date) rows",
                        n_pre_dedup - len(post_car))
        n_before_merge = len(df)
        df = df.merge(post_car[["permno", "event_date",
                                 "car_p2p20", "car_p2p60",
                                 "n_days_20", "n_days_60"]],
                      on=["permno", "event_date"], how="left")
        if len(df) != n_before_merge:
            raise RuntimeError(
                f"post_event_car merge expanded N from {n_before_merge} to {len(df)}. "
                "Remaining duplicate (permno, event_date) pairs — check post_event_car.parquet."
            )
        n_p20 = df["car_p2p20"].notna().sum()
        n_p60 = df["car_p2p60"].notna().sum()
        log.info("Post-event CAR merge: car_p2p20 non-null=%d / %d, car_p2p60 non-null=%d / %d",
                 n_p20, len(df), n_p60, len(df))
    else:
        df["car_p2p20"] = np.nan
        df["car_p2p60"] = np.nan
        df["n_days_20"] = np.nan
        df["n_days_60"] = np.nan
        log.warning("post_event_car.parquet not found; run 14_build_post_event_car.py "
                    "to enable the reversal test (Table 9).")

    # Audit credibility moderators (from 16_build_audit_credibility_moderators.py).
    # Merge on gvkey × comp_year (comp_year = event_year - 1, already in df).
    # Moderators: altman_z, abs_da, gc_opinion.
    if AUDIT_CRED_FILE.exists():
        acred = pd.read_parquet(AUDIT_CRED_FILE)
        acred = acred.rename(columns={"fyear": "comp_year"})
        df = df.merge(acred, on=["gvkey", "comp_year"], how="left")

        # Construct above/below-median indicators on estimation sample
        # high_distress = below-median Z (lower Z = more distress)
        med_z = df["altman_z"].median()
        df["high_distress"] = (df["altman_z"] < med_z).astype(float)
        df.loc[df["altman_z"].isna(), "high_distress"] = np.nan

        # high_daccruals = above-median |DA|
        med_da = df["abs_da"].median()
        df["high_daccruals"] = (df["abs_da"] > med_da).astype(float)
        df.loc[df["abs_da"].isna(), "high_daccruals"] = np.nan

        # gc_opinion already 0/1 from script 16
        n_z  = df["high_distress"].notna().sum()
        n_da = df["high_daccruals"].notna().sum()
        n_gc = df["gc_opinion"].notna().sum()
        gc_n = int(df["gc_opinion"].sum()) if n_gc > 0 else 0
        log.info("Audit credibility merge: Z valid=%d, |DA| valid=%d, GC valid=%d (GC=1: %d)",
                 n_z, n_da, n_gc, gc_n)
        log.info("  median Altman Z = %.3f, median |DA| = %.4f", med_z, med_da)
    else:
        df["altman_z"]      = np.nan
        df["abs_da"]        = np.nan
        df["gc_opinion"]    = np.nan
        df["high_distress"] = np.nan
        df["high_daccruals"] = np.nan
        log.warning("audit_credibility_moderators.parquet not found; "
                    "run 16_build_audit_credibility_moderators.py.")

    # Short interest (from 17_build_short_interest.py).
    # Merge on gvkey × event_date.
    if SHORT_INT_FILE.exists():
        si = pd.read_parquet(SHORT_INT_FILE)
        si["event_date"] = pd.to_datetime(si["event_date"])
        df = df.merge(
            si[["gvkey", "event_date", "si_ratio_pre", "si_ratio_post", "si_change"]],
            on=["gvkey", "event_date"], how="left"
        )
        n_si = df["si_ratio_pre"].notna().sum()
        n_chg = df["si_change"].notna().sum()

        # high_si = above-median pre-event short interest ratio
        med_si = df["si_ratio_pre"].median()
        df["high_si"] = (df["si_ratio_pre"] > med_si).astype(float)
        df.loc[df["si_ratio_pre"].isna(), "high_si"] = np.nan

        # Standardize si_change for use as outcome
        si_mu  = df["si_change"].mean()
        si_sig = df["si_change"].std()
        df["si_change_std"] = (df["si_change"] - si_mu) / si_sig if si_sig > 0 else np.nan

        log.info("Short interest merge: pre SI valid=%d / %d, SI change valid=%d / %d",
                 n_si, len(df), n_chg, len(df))
        log.info("  median si_ratio_pre = %.6f", med_si)
    else:
        df["si_ratio_pre"]  = np.nan
        df["si_ratio_post"] = np.nan
        df["si_change"]     = np.nan
        df["si_change_std"] = np.nan
        df["high_si"]       = np.nan
        log.warning("short_interest.parquet not found; "
                    "run 17_build_short_interest.py.")

    # Institutional ownership (from 18_build_institutional_ownership.py).
    # Merge on permno × event_date.
    if INST_OWN_FILE.exists():
        inst = pd.read_parquet(INST_OWN_FILE)
        inst["event_date"] = pd.to_datetime(inst["event_date"])
        inst["permno"] = inst["permno"].astype(int)
        df = df.merge(
            inst[["permno", "event_date", "inst_own_pct", "retail_pct"]],
            on=["permno", "event_date"], how="left"
        )
        n_inst = df["inst_own_pct"].notna().sum()

        # high_retail = above-median retail ownership fraction
        med_retail = df["retail_pct"].median()
        df["high_retail"] = (df["retail_pct"] > med_retail).astype(float)
        df.loc[df["retail_pct"].isna(), "high_retail"] = np.nan

        log.info("Institutional ownership merge: %d / %d events matched",
                 n_inst, len(df))
        log.info("  median retail_pct = %.4f  high_retail N=%d",
                 med_retail, df["high_retail"].notna().sum())
    else:
        df["inst_own_pct"] = np.nan
        df["retail_pct"]   = np.nan
        df["high_retail"]  = np.nan
        log.warning("institutional_ownership.parquet not found; "
                    "run 18_build_institutional_ownership.py.")

    # Filing specificity / informativeness measures (from script 19).
    # Residualizes log nonstandard word count on materiality correlates
    # (dismissal, size, year) so the measure captures interpretive
    # latitude orthogonal to economic materiality.
    if SPECIFICITY_FILE.exists():
        spec = pd.read_parquet(SPECIFICITY_FILE)
        # Keep one row per acc_nodash (drop duplicate parses)
        spec = spec.drop_duplicates(subset="acc_nodash")
        df = df.merge(
            spec[["acc_nodash", "specificity_index", "log_nonstd_words"]],
            on="acc_nodash", how="left"
        )
        log.info("Filing specificity merge: %d / %d events matched",
                 df["specificity_index"].notna().sum(), len(df))

        # Residualize log_nonstd_words on materiality correlates.
        # Fit on rows with both outcome and covariates available.
        resid_df = df.dropna(subset=["log_nonstd_words", "dismissal", "size"]).copy()
        resid_df["year_str"] = resid_df["event_year"].astype(str)
        m_resid = smf.ols(
            "log_nonstd_words ~ dismissal + size + C(year_str)",
            data=resid_df
        ).fit()
        df["nonstd_resid"] = np.nan
        df.loc[resid_df.index, "nonstd_resid"] = (
            resid_df["log_nonstd_words"] - m_resid.fittedvalues
        )
        # Bottom-tercile indicator: low_info = 1 if residualized nonstandard
        # word count is in the bottom tercile (most boilerplate filings)
        tercile_cut = df["nonstd_resid"].quantile(1/3)
        df["low_info"] = (df["nonstd_resid"] <= tercile_cut).astype(float)
        df.loc[df["nonstd_resid"].isna(), "low_info"] = np.nan
        log.info("  Nonstandard word count residualized on dismissal+size+year")
        log.info("  Low-info tercile cutoff: %.3f, N_low=%d",
                 tercile_cut, int((df["low_info"] == 1).sum()))
    else:
        df["specificity_index"] = np.nan
        df["log_nonstd_words"]  = np.nan
        df["nonstd_resid"]      = np.nan
        df["low_info"]          = np.nan
        log.warning("filing_specificity.parquet not found; "
                    "run 19_build_filing_specificity.py.")

    # Post-event uncertainty measures (from script 21) for mechanism validation
    # Six independent measures of post-event uncertainty computed from CRSP daily data:
    # return volatility, Parkinson high-low volatility, Amihud illiquidity,
    # bid-ask spread, abnormal volume persistence, Roll implied spread.
    if UNCERTAINTY_FILE.exists():
        unc = pd.read_parquet(UNCERTAINTY_FILE)
        unc["event_date"] = pd.to_datetime(unc["event_date"])
        unc["permno"] = unc["permno"].astype(int)
        change_cols = ["vol_change", "parkinson_change", "amihud_change",
                       "spread_change", "abvol_change", "roll_change"]
        pre_cols = ["pre_vol", "pre_parkinson", "pre_amihud",
                    "pre_spread", "pre_abvol", "pre_roll"]
        # Deduplicate on (permno, event_date) to prevent merge fan-out
        # (observed: 16 duplicate keys that inflate the sample from 678 to 724).
        n_pre_dedup = len(unc)
        unc = unc.drop_duplicates(subset=["permno", "event_date"], keep="first")
        if len(unc) < n_pre_dedup:
            log.warning("post_event_uncertainty: dropped %d duplicate "
                        "(permno, event_date) rows", n_pre_dedup - len(unc))
        n_before_merge = len(df)
        df = df.merge(
            unc[["permno", "event_date"] + change_cols + pre_cols],
            on=["permno", "event_date"], how="left"
        )
        if len(df) != n_before_merge:
            raise RuntimeError(
                f"post_event_uncertainty merge expanded N from {n_before_merge} "
                f"to {len(df)}. Remaining duplicate (permno, event_date) pairs."
            )
        # Winsorize each change measure at 1st/99th percentiles
        for c in change_cols:
            mask = df[c].notna()
            if mask.sum() > 0:
                lo, hi = df.loc[mask, c].quantile([0.01, 0.99])
                df[c] = df[c].clip(lower=lo, upper=hi)
            # Standardize for composite
            if df[c].notna().sum() > 0:
                df[c + "_std"] = (df[c] - df[c].mean()) / df[c].std()
            else:
                df[c + "_std"] = np.nan

        # Equal-weighted composite across all six standardized measures
        std_cols = [c + "_std" for c in change_cols]
        df["uncertainty_composite"] = df[std_cols].mean(axis=1, skipna=False)
        log.info("Post-event uncertainty merge: %d / %d events matched",
                 df["vol_change"].notna().sum(), len(df))
        log.info("  Composite uncertainty: N=%d",
                 df["uncertainty_composite"].notna().sum())
    else:
        for c in ["vol_change", "parkinson_change", "amihud_change",
                  "spread_change", "abvol_change", "roll_change",
                  "pre_vol", "pre_parkinson", "pre_amihud", "pre_spread",
                  "pre_abvol", "pre_roll", "uncertainty_composite"]:
            df[c] = np.nan
        log.warning("post_event_uncertainty.parquet not found; "
                    "run 21_build_post_event_uncertainty.py.")

    # Save analysis sample
    df.to_parquet(SAMPLE_FILE, index=False)
    log.info("Analysis sample saved: %s  (%d events, %d unique firms)",
             SAMPLE_FILE, len(df), df["gvkey"].nunique())

    # Summary stats — Table 1
    log.info("Building Table 1: Summary statistics")
    stats_df = make_summary_stats(df)
    to_latex_table(
        stats_df,
        caption="Summary Statistics",
        label="tab:summary",
        out_path=OUT_TABS / "tab01_summary_stats.tex",
    )
    log.info("\n%s", stats_df.to_string(index=False))

    # Main results — Table 2
    log.info("Running Table 2: Main results")
    run_main_results(df)

    # Event type heterogeneity — Table 3
    log.info("Running Table 3: Event type heterogeneity")
    run_event_type(df)

    # Ambiguity — Table 4
    log.info("Running Table 4: Signal ambiguity")
    run_ambiguity(df)

    # Robustness — Table 5
    log.info("Running Table 5: Robustness")
    run_robustness(df)

    # Permutation test — Table 7
    log.info("Running Table 7: Permutation test (5,000 draws)")
    run_permutation_test(df)

    # Affective polarization supplementary test — Table 6
    log.info("Running Table 6: Affective polarization validation")
    run_affective_test(df)

    # Analyst dispersion mechanism test — skipped (no IBES access)
    if df["high_disp"].notna().any():
        log.info("Running Table 8 (dispersion): Analyst dispersion mechanism test")
        run_dispersion_interaction_test(df)
    else:
        log.warning("Skipping dispersion test (no IBES data). Run 11_build_ibes.py first.")

    # Local investor relevance test — Table 8
    log.info("Running Table 8: Local investor relevance test")
    run_local_bias_test(df)

    # Audit credibility interaction test — Table 10 (skipped if no moderators)
    if df["high_distress"].notna().any():
        log.info("Running Table 10: Audit credibility interaction test")
        run_audit_credibility_test(df)
    else:
        log.warning("Skipping audit credibility test (no moderator data). "
                    "Run 16_build_audit_credibility_moderators.py first.")

    # Short interest disagreement test — Table 11 (skipped if no short_interest.parquet)
    if df["high_si"].notna().any():
        log.info("Running Table 11: Short interest disagreement test")
        run_short_interest_test(df)
    else:
        log.warning("Skipping short interest test (no SI data). "
                    "Run 17_build_short_interest.py first.")

    # Institutional ownership / retail fraction test (skipped if no inst ownership data)
    if df["high_retail"].notna().any():
        log.info("Running institutional ownership interaction test")
        run_institutional_ownership_test(df)
    else:
        log.warning("Skipping institutional ownership test (no ownership data). "
                    "Run 18_build_institutional_ownership.py first.")

    # Post-event CAR reversal test — Table 9 (skipped if no post_event_car.parquet)
    if df["car_p2p20"].notna().any():
        log.info("Running Table 9: Post-event CAR reversal test")
        run_reversal_test(df)
    else:
        log.warning("Skipping reversal test (no post-event CAR data). "
                    "Run 14_build_post_event_car.py first.")

    # Regulatory shock tests — SOX/PCAOB interaction
    log.info("Running regulatory shock tests (SOX/PCAOB)")
    run_regulatory_shock_test(df)

    log.info("=== done — tables written to %s ===", OUT_TABS)


if __name__ == "__main__":
    main()
