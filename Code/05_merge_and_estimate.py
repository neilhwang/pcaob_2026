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

OUTPUTS:
    Output/Tables/tab01_summary_stats.tex
    Output/Tables/tab02_main_results.tex
    Output/Tables/tab03_event_type.tex
    Output/Tables/tab04_ambiguity.tex
    Output/Tables/tab05_robustness.tex
    Output/Tables/tab06_affective.tex
    Data/Processed/analysis_sample.parquet        (merged estimation sample)

SPECIFICATION:
    Main regression (cross-section of auditor change events):

        Y_i = alpha + beta*Pol_{s(i),t(i)} + gamma'*X_{i,t} + FE + eps_i

    where Y_i is |CAR(-1,+1)| or AbVol(-1,+1), Pol is the standardized
    Esteban-Ray polarization measure for the firm's HQ state in the event year,
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
EXPOSURE_FILE = PROC / "state_partisan_exposure.parquet"
AP_FILE       = PROC / "affective_polarization.parquet"
SAMPLE_FILE   = PROC / "analysis_sample.parquet"

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

    log.info("CRSP events: %d", len(crsp))
    log.info("Polarization panel: %d rows", len(pol))
    log.info("Compustat controls: %d rows", len(comp))
    log.info("DW-NOMINATE panel: %d rows", len(dw))
    log.info("Partisan exposure: %d states", len(exposure))
    log.info("Affective polarization: %d years", len(ap))

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

    # pol_std standardization is deferred to after apply_sample_filters() so that
    # mean=0, sd=1 holds exactly in the estimation sample (not the pre-filter universe).
    # Leave pol_er_alpha10 in place; standardization applied in main().

    # Event-type indicators for heterogeneity tests
    crsp["dismissal"]      = (crsp["reason"] == "dismissal").astype(int)
    crsp["disagreement"]   = crsp["disagreements"].fillna(False).astype(int)
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
    """Drop observations missing key variables for the main regression."""
    n0 = len(df)
    # Filter on pol_er_alpha10 (the raw measure), not pol_std — pol_std is
    # created in main() after this function returns (so it doesn't exist here yet).
    df = df.dropna(subset=["absCar", "abvol", "pol_er_alpha10", "sic2"] + CONTROLS)
    log.info("After dropping missing: %d rows (dropped %d)", len(df), n0 - len(df))
    return df.reset_index(drop=True)


# ── Step 2: Summary statistics ────────────────────────────────────────────────

def make_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "absCar":       "|CAR(-1,+1)|",
        "car":          "CAR(-1,+1)",
        "abvol":        "Abn. Volume",
        "pol_er_alpha10": "Polarization (ER, α=1)",
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
        booktabs=True,
    )
    # Wrap in table environment with notes placeholder
    tex = tex.replace(
        r"\end{tabular}",
        r"\end{tabular}" + "\n" + r"\begin{flushleft}" + "\n"
        r"\footnotesize Notes: [PLACEHOLDER]" + "\n"
        r"\end{flushleft}",
    )
    out_path.write_text(tex)
    log.info("Table written: %s", out_path)


# ── Step 3: Regression helpers ────────────────────────────────────────────────

def run_ols(formula: str, df: pd.DataFrame,
            cluster_var: str = "gvkey_str") -> object:
    """OLS with firm-clustered standard errors."""
    model  = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df[cluster_var]},
    )
    return model


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
                se_cells.append(f"({fmt(s)})")
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
        # (1) |CAR|, no controls, year FE only
        "m1": run_ols(f"absCar ~ pol_std + C(year_str)", df),
        # (2) |CAR|, controls + year FE + industry FE
        "m2": run_ols(f"absCar ~ pol_std + {ctrl} + {fe}", df),
        # (3) AbVol, no controls, year FE only
        "m3": run_ols(f"abvol ~ pol_std + C(year_str)", df),
        # (4) AbVol, controls + year FE + industry FE
        "m4": run_ols(f"abvol ~ pol_std + {ctrl} + {fe}", df),
    }

    for k, m in specs.items():
        log.info("%s: pol_std coef=%.4f p=%.3f", k, m.params["pol_std"],
                 m.pvalues["pol_std"])

    coef_map = {
        "pol_std":     r"\textit{Polarization}",
        "size":        "Size",
        "leverage":    "Leverage",
        "roa":         "ROA",
        "btm":         "Book-to-Market",
        "loss":        "Loss",
        "sales_growth":"Sales Growth",
    }
    dep_labels = [r"$|CAR|$", r"$|CAR|$", r"AbVol", r"AbVol"]
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
        "m1": run_ols(f"absCar ~ pol_std + {ctrl} + {fe}", dismissals),
        "m2": run_ols(f"absCar ~ pol_std + {ctrl} + {fe}", resignations),
        "m3": run_ols(f"absCar ~ pol_std + {ctrl} + {fe}", qual_down),
        "m4": run_ols(f"absCar ~ pol_std + {ctrl} + {fe}", qual_up),
    }

    for k, m in specs.items():
        log.info("Event-type %s: pol_std coef=%.4f p=%.3f", k,
                 m.params["pol_std"], m.pvalues["pol_std"])

    coef_map = {"pol_std": r"\textit{Polarization}"}
    # "Resignations" subsample is df[dismissal==0], which includes both
    # resignations and unclassified events; label accurately as Non-dismissals.
    dep_labels = [r"$|CAR|$\newline Dismissals",
                  r"$|CAR|$\newline Non-dismissals",
                  r"$|CAR|$\newline Big4$\to$Non",
                  r"$|CAR|$\newline Non$\to$Big4"]
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


# ── Step 6: Signal ambiguity (Table 4) ────────────────────────────────────────

def run_ambiguity(df: pd.DataFrame) -> None:
    ctrl = " + ".join(CONTROLS)
    fe   = "C(year_str) + C(sic2_str)"

    hi_amb = df[df["high_ambiguity"] == 1]
    lo_amb = df[df["high_ambiguity"] == 0]

    specs = {
        "m1": run_ols(f"absCar ~ pol_std + {ctrl} + {fe}", df),
        "m2": run_ols(f"absCar ~ pol_std + {ctrl} + {fe}", hi_amb),
        "m3": run_ols(f"absCar ~ pol_std + {ctrl} + {fe}", lo_amb),
        "m4": run_ols(f"abvol  ~ pol_std + {ctrl} + {fe}", hi_amb),
        "m5": run_ols(f"abvol  ~ pol_std + {ctrl} + {fe}", lo_amb),
    }

    coef_map = {"pol_std": r"\textit{Polarization}"}
    dep_labels = [r"$|CAR|$\newline Full",
                  r"$|CAR|$\newline High Amb.",
                  r"$|CAR|$\newline Low Amb.",
                  r"AbVol\newline High Amb.",
                  r"AbVol\newline Low Amb."]
    extra_rows = {
        "Year + Industry FE": ["Yes"] * 5,
        "Controls":           ["Yes"] * 5,
    }
    reg_table(
        list(specs.values()), dep_labels, coef_map,
        caption="Signal Ambiguity and the Effect of Polarization",
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

    specs = {
        # (1) Baseline (repeat for comparison)
        "m1": run_ols(f"absCar ~ pol_std + {ctrl} + {fe}", df),
        # (2) α = 0.8
        "m2": run_ols(f"absCar ~ pol_std_a08 + {ctrl} + {fe}", df),
        # (3) α = 1.2
        "m3": run_ols(f"absCar ~ pol_std_a12 + {ctrl} + {fe}", df),
        # (4) DW-NOMINATE: state-level cross-party gap (alternative pol proxy)
        "m4": run_ols(f"absCar ~ dw_std + {ctrl} + {fe}", df),
        # (5) DW-NOMINATE: national gap (time-series pol variation)
        "m5": run_ols(f"absCar ~ dw_national_std + {ctrl} + {fe}", df),
        # (6) Add state FEs to baseline spec (absorbs cross-state confounders)
        "m6": run_ols(f"absCar ~ pol_std + {ctrl} + {fe_st}", df),
        # (7) State-level clustered SEs (pol measure varies at state x year level)
        "m7": run_ols(f"absCar ~ pol_std + {ctrl} + {fe}", df,
                      cluster_var="state_str"),
    }

    for k, m in specs.items():
        pol_param = next((p for p in ["pol_std", "pol_std_a08", "pol_std_a12",
                                       "dw_std", "dw_national_std"]
                          if p in m.params), None)
        if pol_param:
            log.info("Robustness %s: %s coef=%.4f p=%.3f",
                     k, pol_param, m.params[pol_param], m.pvalues[pol_param])

    coef_map = {
        "pol_std":          r"ER Polarization (baseline, $\alpha=1$)",
        "pol_std_a08":      r"ER Polarization ($\alpha=0.8$)",
        "pol_std_a12":      r"ER Polarization ($\alpha=1.2$)",
        "dw_std":           r"DW-NOMINATE (state gap)",
        "dw_national_std":  r"DW-NOMINATE (national gap)",
    }
    dep_labels = [r"$|CAR|$\newline ER Base",
                  r"$|CAR|$\newline $\alpha=0.8$",
                  r"$|CAR|$\newline $\alpha=1.2$",
                  r"$|CAR|$\newline DW State",
                  r"$|CAR|$\newline DW Natl",
                  r"$|CAR|$\newline +State FE",
                  r"$|CAR|$\newline State Clust"]
    extra_rows = {
        "Year + Industry FE": ["Yes"] * 7,
        "State FE":           ["No", "No", "No", "No", "No", "Yes", "No"],
        "Cluster":            ["Firm"] * 6 + ["State"],
        "Controls":           ["Yes"] * 7,
    }
    reg_table(
        list(specs.values()), dep_labels, coef_map,
        caption="Robustness Tests",
        label="tab:robustness",
        out_path=OUT_TABS / "tab05_robustness.tex",
        extra_rows=extra_rows,
    )


# ── Step 8: Supplementary affective-polarization test (Table 6) ───────────────

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
        Y_e = α + β1·pol_std + β2·high_ambiguity + β3·pol_std×high_ambiguity
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

    specs = {
        # (1) Ideological pol only (baseline, for comparison)
        "m1": run_ols(
            f"absCar ~ pol_std + {ctrl} + {fe}", ap_df
        ),
        # (2) Add AP × Exposure (no ambiguity interaction)
        "m2": run_ols(
            f"absCar ~ pol_std + ap_x_exposure + {ctrl} + {fe}", ap_df
        ),
        # (3) Add ambiguity split for ideological pol
        "m3": run_ols(
            f"absCar ~ pol_std + high_ambiguity + pol_std:high_ambiguity"
            f" + {ctrl} + {fe}", ap_df
        ),
        # (4) Full spec: ideological + affective + ambiguity interactions
        "m4": run_ols(
            f"absCar ~ pol_std + high_ambiguity + pol_std:high_ambiguity"
            f" + ap_x_exposure + ap_x_exp_x_amb + {ctrl} + {fe}", ap_df
        ),
        # (5) AbVol outcome, full spec
        "m5": run_ols(
            f"abvol ~ pol_std + high_ambiguity + pol_std:high_ambiguity"
            f" + ap_x_exposure + ap_x_exp_x_amb + {ctrl} + {fe}", ap_df
        ),
    }

    for k, m in specs.items():
        for param in ["pol_std", "ap_x_exposure", "ap_x_exp_x_amb"]:
            if param in m.params:
                log.info("Affective %s [%s]: coef=%.4f p=%.3f",
                         k, param, m.params[param], m.pvalues[param])

    coef_map = {
        "pol_std":                   r"\textit{Ideo. Pol.} (ER)",
        "high_ambiguity":            r"High Ambiguity",
        "pol_std:high_ambiguity":    r"\textit{Ideo. Pol.} $\times$ Ambiguity",
        "ap_x_exposure":             r"$AP \times Exposure$",
        "ap_x_exp_x_amb":            r"$AP \times Exposure \times$ Ambiguity",
    }
    dep_labels = [
        r"$|CAR|$\newline Ideo. only",
        r"$|CAR|$\newline +Affective",
        r"$|CAR|$\newline +Amb. $\times$ Ideo.",
        r"$|CAR|$\newline Full",
        r"AbVol\newline Full",
    ]
    extra_rows = {
        "Year + Industry FE": ["Yes"] * 5,
        "Controls":           ["Yes"] * 5,
    }
    reg_table(
        list(specs.values()), dep_labels, coef_map,
        caption=(
            r"Supplementary Affective Polarization Test. "
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== 05_merge_and_estimate.py  start ===")

    # Check inputs exist
    for f in [CRSP_FILE, POL_FILE, COMP_FILE, DW_FILE, EXPOSURE_FILE, AP_FILE]:
        if not f.exists():
            raise FileNotFoundError(
                f"Missing input: {f}\n"
                "Run scripts 02, 03, and 04 first."
            )

    # Merge
    df = load_and_merge()
    df = apply_sample_filters(df)

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
    ]:
        if raw_col in df.columns and df[raw_col].notna().sum() > 0:
            mu  = df[raw_col].mean()
            sig = df[raw_col].std()
            df[std_col] = (df[raw_col] - mu) / sig
        else:
            df[std_col] = np.nan

    # Affective polarization interaction term: AP_t × Exposure_s.
    # This is the identifying variation — differential response of firms in
    # high-exposure states as national affective polarization rises over time.
    # ap_ft alone is absorbed by year FEs and therefore not entered separately.
    df["ap_x_exposure"] = df["ap_std"] * df["exposure_std"]
    df["ap_x_exp_x_amb"] = df["ap_x_exposure"] * df["high_ambiguity"]
    log.info("AP x Exposure non-null: %d", df["ap_x_exposure"].notna().sum())

    # Save analysis sample
    df.to_parquet(SAMPLE_FILE, index=False)
    log.info("Analysis sample saved: %s  (%d rows)", SAMPLE_FILE, len(df))

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

    # Affective polarization supplementary test — Table 6
    log.info("Running Table 6: Affective polarization validation")
    run_affective_test(df)

    log.info("=== done — tables written to %s ===", OUT_TABS)


if __name__ == "__main__":
    main()
