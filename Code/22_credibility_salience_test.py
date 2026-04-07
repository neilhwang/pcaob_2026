"""
22_credibility_salience_test.py
================================
Within-filing mechanism test: does the polarization effect depend on
whether the filing directly involves audit credibility?

DESIGN:
    If polarization amplifies disagreement specifically because investors
    differ in how they assess audit credibility, the effect should be
    stronger when the filing puts audit credibility directly at stake:
    - Disclosed disagreements between registrant and departing auditor
    - Reportable events (material weaknesses, going concerns, etc.)
    - Concrete accounting issues named in the filing

    This is a DIFFERENT prediction from the old ambiguity test (which
    conflated ambiguity with materiality). Here, credibility-salient
    filings are LESS ambiguous but MORE directly about audit credibility.

INPUTS:
    Data/Processed/analysis_sample.parquet
    Data/Processed/filing_specificity.parquet  (for component-level variables)

OUTPUTS:
    Console output with regression results
    Output/Tables/tab_credibility_salience.tex  (if results warrant inclusion)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
PROC     = ROOT / "Data/Processed"
OUT_TABS = ROOT / "Output/Tables"

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_parquet(PROC / "analysis_sample.parquet")
print(f"Analysis sample loaded: N={len(df)}")

# Merge filing specificity components (not in analysis_sample by default)
spec_file = PROC / "filing_specificity.parquet"
if spec_file.exists():
    spec = pd.read_parquet(spec_file)
    # Keep one row per filing
    spec = spec.drop_duplicates(subset="acc_nodash")
    # Columns to merge
    spec_cols = [
        "acc_nodash",
        "explicit_cause", "concrete_issue", "disagreement_domain",
        "reportable_event", "committee_process", "linked_transaction",
        "nongeneric_language"
    ]
    available = [c for c in spec_cols if c in spec.columns]
    df = df.merge(spec[available], on="acc_nodash", how="left")
    print(f"Filing specificity components merged: "
          f"{sum(c in df.columns for c in spec_cols[1:])} of 7 components")
else:
    print("WARNING: filing_specificity.parquet not found")

# ── Setup ─────────────────────────────────────────────────────────────────────
CONTROLS = ["size", "leverage", "roa", "btm", "loss", "sales_growth"]
ctrl = " + ".join(CONTROLS)
fe   = "C(year_str) + C(sic2_str)"

# Ensure string FE columns exist
for col in ["year_str", "sic2_str"]:
    if col not in df.columns:
        if col == "year_str":
            df["year_str"] = df["year"].astype(str)
        elif col == "sic2_str":
            df["sic2_str"] = df["sic2"].astype(str)


def run_ols(formula, data):
    """Run OLS with firm-level clustering."""
    mod = smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["gvkey"]}
    )
    return mod


# ── Descriptive statistics on credibility indicators ──────────────────────────
print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS: Credibility-salience indicators")
print("=" * 70)

cred_vars = ["disagreement", "reportable_event", "concrete_issue",
             "disagreement_domain", "explicit_cause"]
for v in cred_vars:
    if v in df.columns:
        n_valid = df[v].notna().sum()
        n_pos   = (df[v] == 1).sum() if n_valid > 0 else 0
        pct     = 100 * n_pos / n_valid if n_valid > 0 else 0
        print(f"  {v:25s}: N={n_valid:4d}, n=1: {n_pos:4d} ({pct:5.1f}%)")
    else:
        print(f"  {v:25s}: NOT AVAILABLE")

# ── Composite credibility-salience indicator ──────────────────────────────────
# A filing is "credibility-salient" if it discloses any of:
#   - a disagreement with the departing auditor
#   - a reportable event (material weakness, going concern, etc.)
#   - a concrete accounting issue
# These are the filings where audit credibility is directly at stake.

components = []
if "disagreement" in df.columns:
    components.append("disagreement")
if "reportable_event" in df.columns:
    components.append("reportable_event")
if "concrete_issue" in df.columns:
    components.append("concrete_issue")

if len(components) >= 1:
    df["cred_salient"] = df[components].max(axis=1)
    n_sal = (df["cred_salient"] == 1).sum()
    print(f"\nComposite credibility-salient (any of {components}):")
    print(f"  N=1: {n_sal} ({100*n_sal/len(df):.1f}%), N=0: {len(df)-n_sal}")
else:
    print("\nWARNING: Cannot construct credibility-salience indicator")


# ── Main interaction tests ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 1: Polarization x Disagreement")
print("=" * 70)

if "disagreement" in df.columns:
    df["pol_x_disagree"] = df["competitive_std"] * df["disagreement"]

    for outcome, label in [("abvol", "AbVol"), ("absCar", "|CAR|")]:
        print(f"\n--- {label} ---")

        # Baseline (for reference)
        m_base = run_ols(
            f"{outcome} ~ competitive_std + {ctrl} + {fe}", df)
        print(f"  Baseline:  beta={m_base.params['competitive_std']:+.4f}  "
              f"p={m_base.pvalues['competitive_std']:.3f}  N={int(m_base.nobs)}")

        # Interaction
        m_int = run_ols(
            f"{outcome} ~ competitive_std + disagreement + pol_x_disagree "
            f"+ {ctrl} + {fe}", df)
        print(f"  Pol:       beta={m_int.params['competitive_std']:+.4f}  "
              f"p={m_int.pvalues['competitive_std']:.3f}")
        print(f"  Disagree:  beta={m_int.params['disagreement']:+.4f}  "
              f"p={m_int.pvalues['disagreement']:.3f}")
        print(f"  Pol x Dis: beta={m_int.params['pol_x_disagree']:+.4f}  "
              f"p={m_int.pvalues['pol_x_disagree']:.3f}  N={int(m_int.nobs)}")

        # Subsample: disagreement=1 only
        d1 = df[df["disagreement"] == 1]
        if len(d1) > 30:
            m_sub1 = run_ols(f"{outcome} ~ competitive_std + {ctrl} + {fe}", d1)
            print(f"  Subsample (disagree=1): beta={m_sub1.params['competitive_std']:+.4f}  "
                  f"p={m_sub1.pvalues['competitive_std']:.3f}  N={int(m_sub1.nobs)}")

        # Subsample: disagreement=0 only
        d0 = df[df["disagreement"] == 0]
        if len(d0) > 30:
            m_sub0 = run_ols(f"{outcome} ~ competitive_std + {ctrl} + {fe}", d0)
            print(f"  Subsample (disagree=0): beta={m_sub0.params['competitive_std']:+.4f}  "
                  f"p={m_sub0.pvalues['competitive_std']:.3f}  N={int(m_sub0.nobs)}")


# ── Test 2: Reportable events ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 2: Polarization x Reportable Event")
print("=" * 70)

if "reportable_event" in df.columns and df["reportable_event"].notna().sum() > 0:
    df["pol_x_reportable"] = df["competitive_std"] * df["reportable_event"]

    for outcome, label in [("abvol", "AbVol"), ("absCar", "|CAR|")]:
        print(f"\n--- {label} ---")
        m_int = run_ols(
            f"{outcome} ~ competitive_std + reportable_event + pol_x_reportable "
            f"+ {ctrl} + {fe}", df)
        print(f"  Pol:        beta={m_int.params['competitive_std']:+.4f}  "
              f"p={m_int.pvalues['competitive_std']:.3f}")
        print(f"  Reportable: beta={m_int.params['reportable_event']:+.4f}  "
              f"p={m_int.pvalues['reportable_event']:.3f}")
        print(f"  Pol x Rep:  beta={m_int.params['pol_x_reportable']:+.4f}  "
              f"p={m_int.pvalues['pol_x_reportable']:.3f}  N={int(m_int.nobs)}")
else:
    print("  reportable_event not available; skipping")


# ── Test 3: Composite credibility-salience ────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 3: Polarization x Credibility-Salient (composite)")
print("=" * 70)

if "cred_salient" in df.columns:
    df["pol_x_cred"] = df["competitive_std"] * df["cred_salient"]

    for outcome, label in [("abvol", "AbVol"), ("absCar", "|CAR|")]:
        print(f"\n--- {label} ---")
        m_int = run_ols(
            f"{outcome} ~ competitive_std + cred_salient + pol_x_cred "
            f"+ {ctrl} + {fe}", df)
        print(f"  Pol:          beta={m_int.params['competitive_std']:+.4f}  "
              f"p={m_int.pvalues['competitive_std']:.3f}")
        print(f"  Cred Salient: beta={m_int.params['cred_salient']:+.4f}  "
              f"p={m_int.pvalues['cred_salient']:.3f}")
        print(f"  Pol x Cred:   beta={m_int.params['pol_x_cred']:+.4f}  "
              f"p={m_int.pvalues['pol_x_cred']:.3f}  N={int(m_int.nobs)}")

        # Subsample: cred_salient=1
        d1 = df[df["cred_salient"] == 1]
        if len(d1) > 30:
            m_sub1 = run_ols(f"{outcome} ~ competitive_std + {ctrl} + {fe}", d1)
            print(f"  Subsample (salient=1):  beta={m_sub1.params['competitive_std']:+.4f}  "
                  f"p={m_sub1.pvalues['competitive_std']:.3f}  N={int(m_sub1.nobs)}")

        d0 = df[df["cred_salient"] == 0]
        if len(d0) > 30:
            m_sub0 = run_ols(f"{outcome} ~ competitive_std + {ctrl} + {fe}", d0)
            print(f"  Subsample (salient=0):  beta={m_sub0.params['competitive_std']:+.4f}  "
                  f"p={m_sub0.pvalues['competitive_std']:.3f}  N={int(m_sub0.nobs)}")


# ── Test 4: Individual specificity components ─────────────────────────────────
print("\n" + "=" * 70)
print("TEST 4: Individual specificity components")
print("=" * 70)

indiv_vars = ["explicit_cause", "concrete_issue", "reportable_event",
              "disagreement_domain", "committee_process",
              "linked_transaction", "nongeneric_language"]

for v in indiv_vars:
    if v not in df.columns or df[v].notna().sum() < 50:
        continue
    df[f"pol_x_{v}"] = df["competitive_std"] * df[v]

    for outcome, label in [("abvol", "AbVol"), ("absCar", "|CAR|")]:
        m = run_ols(
            f"{outcome} ~ competitive_std + {v} + pol_x_{v} + {ctrl} + {fe}", df)
        b_int = m.params.get(f"pol_x_{v}", float("nan"))
        p_int = m.pvalues.get(f"pol_x_{v}", float("nan"))
        print(f"  {v:25s} x Pol -> {label:5s}: beta={b_int:+.4f}  p={p_int:.3f}")


# ── Test 5: Quality-tier transitions ──────────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 5: Polarization x Quality-tier transitions")
print("=" * 70)

if "quality_down" in df.columns:
    df["pol_x_qdown"] = df["competitive_std"] * df["quality_down"]
    df["pol_x_qup"]   = df["competitive_std"] * df["quality_up"]

    for outcome, label in [("abvol", "AbVol"), ("absCar", "|CAR|")]:
        print(f"\n--- {label} ---")
        # Quality downgrade interaction
        m = run_ols(
            f"{outcome} ~ competitive_std + quality_down + pol_x_qdown "
            f"+ {ctrl} + {fe}", df)
        print(f"  Pol x QualDown:  beta={m.params['pol_x_qdown']:+.4f}  "
              f"p={m.pvalues['pol_x_qdown']:.3f}  (N_down={int((df['quality_down']==1).sum())})")

        # Quality upgrade interaction
        m2 = run_ols(
            f"{outcome} ~ competitive_std + quality_up + pol_x_qup "
            f"+ {ctrl} + {fe}", df)
        print(f"  Pol x QualUp:    beta={m2.params['pol_x_qup']:+.4f}  "
              f"p={m2.pvalues['pol_x_qup']:.3f}  (N_up={int((df['quality_up']==1).sum())})")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
