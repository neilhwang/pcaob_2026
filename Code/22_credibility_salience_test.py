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


# ── Test 6: Investor-base political heterogeneity (13F + FEC) ─────────────────
print("\n" + "=" * 70)
print("TEST 6: Investor-base political heterogeneity (13F + FEC)")
print("=" * 70)
print("  Source: institutional holdings matched to FEC employer contributions")
print("  Coverage: post-2013Q2 SEC structured data; N_events ~ 65")
print("  investor_het = 1 - |Dem_share - Rep_share| (higher = more mixed base)")
print()

iph_file = PROC / "investor_political_heterogeneity.parquet"
if not iph_file.exists():
    print("  investor_political_heterogeneity.parquet not found — skipping Test 6")
else:
    iph = pd.read_parquet(iph_file)
    iph["event_date"] = pd.to_datetime(iph["event_date"])
    df["event_date"]  = pd.to_datetime(df["event_date"])

    df6 = df.merge(
        iph[["permno", "event_date", "investor_het", "dem_share",
             "rep_share", "matched_share", "n_matched_filers"]],
        on=["permno", "event_date"], how="inner",
    )
    n6 = df6["investor_het"].notna().sum()
    print(f"  Matched events: {len(df6)}  (with investor_het: {n6})")
    df6 = df6[df6["investor_het"].notna()].copy()
    print(f"  Analysis subsample N={len(df6)}")

    if len(df6) < 20:
        print("  Too few observations — skipping regressions")
    else:
        # Descriptive statistics
        print(f"\n  investor_het:   mean={df6['investor_het'].mean():.3f}  "
              f"sd={df6['investor_het'].std():.3f}  "
              f"min={df6['investor_het'].min():.3f}  "
              f"max={df6['investor_het'].max():.3f}")
        print(f"  matched_share:  mean={df6['matched_share'].mean():.3f}  "
              f"n_filers mean={df6['n_matched_filers'].mean():.1f}")

        # Correlation with state-level polarization proxy
        r_pol_het = df6["competitive_std"].corr(df6["investor_het"])
        print(f"\n  Corr(competitive_std, investor_het) = {r_pol_het:.3f}")

        # Standardize investor_het for comparability
        df6["investor_het_std"] = (
            (df6["investor_het"] - df6["investor_het"].mean())
            / df6["investor_het"].std()
        )

        # Ensure FE vars exist
        for col in ["year_str", "sic2_str"]:
            if col not in df6.columns:
                df6[col] = (df6["year"].astype(str) if col == "year_str"
                            else df6["sic2"].astype(str))

        ctrl6 = " + ".join([c for c in CONTROLS if c in df6.columns])
        fe6   = "C(year_str) + C(sic2_str)"

        print("\n--- Main: investor_het replacing competitive_std ---")
        for outcome, label in [("abvol", "AbVol"), ("absCar", "|CAR|")]:
            try:
                m = run_ols(
                    f"{outcome} ~ investor_het_std + {ctrl6} + {fe6}", df6)
                b = m.params["investor_het_std"]
                p = m.pvalues["investor_het_std"]
                print(f"  {label}: beta(investor_het_std)={b:+.4f}  p={p:.3f}  N={int(m.nobs)}")
            except Exception as e:
                print(f"  {label}: ERROR — {e}")

        print("\n--- Baseline: competitive_std in same subsample ---")
        for outcome, label in [("abvol", "AbVol"), ("absCar", "|CAR|")]:
            try:
                m = run_ols(
                    f"{outcome} ~ competitive_std + {ctrl6} + {fe6}", df6)
                b = m.params["competitive_std"]
                p = m.pvalues["competitive_std"]
                print(f"  {label}: beta(competitive_std)={b:+.4f}  p={p:.3f}  N={int(m.nobs)}")
            except Exception as e:
                print(f"  {label}: ERROR — {e}")

        print("\n--- Horse race: both measures jointly ---")
        for outcome, label in [("abvol", "AbVol"), ("absCar", "|CAR|")]:
            try:
                m = run_ols(
                    f"{outcome} ~ competitive_std + investor_het_std + {ctrl6} + {fe6}",
                    df6)
                b_pol = m.params["competitive_std"]
                p_pol = m.pvalues["competitive_std"]
                b_het = m.params["investor_het_std"]
                p_het = m.pvalues["investor_het_std"]
                print(f"  {label}: beta(competitive_std)={b_pol:+.4f} p={p_pol:.3f}  |  "
                      f"beta(investor_het_std)={b_het:+.4f} p={p_het:.3f}  N={int(m.nobs)}")
            except Exception as e:
                print(f"  {label}: ERROR — {e}")

        print()
        print("  NOTE: N=65 subsample; post-2013Q2 only; interpret with caution.")
        print("  matched_share ~0.95 indicates strong FEC coverage conditional on")
        print("  having institutional holdings data.")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
