"""
16_build_audit_credibility_moderators.py
========================================
Builds firm-year audit-credibility salience moderators from Compustat annual
fundamentals for the analysis sample.

PURPOSE
-------
Tests whether the polarization effect is tied to audit credibility specifically
rather than generic politics. Prediction: polarization × [audit credibility salient]
should be positive --- the effect should be STRONGER when the filing forces
investors to evaluate the reliability of accounting oversight.

Three moderators constructed:

  1. Financial distress (Altman Z-score)
       Z = 1.2*(WC/TA) + 1.4*(RE/TA) + 3.3*(EBIT/TA) + 0.6*(MVE/TL) + 1.0*(Sale/TA)
       high_distress = 1 if Z below sample median (more distress → audit
       credibility more salient).

  2. Discretionary accruals (Modified Jones model, Dechow et al. 1995)
       Total accruals: TA = NI - CFO
       Cross-sectional by SIC2 × year on full Compustat population:
         TA/A_{t-1} = α/A_{t-1} + β1*(ΔSale - ΔRec)/A_{t-1} + β2*(PPE/A_{t-1}) + ε
       Discretionary accruals = |residual|.
       high_daccruals = 1 if |DA| above sample median (high accounting
       opacity → audit credibility more salient).

  3. Going-concern opinion
       gc_opinion = 1 if auditor issued a going-concern modification in the
       fiscal year prior to the event (auopic ∈ {2, 3}).
       Very few firms (typically <10% of sample), so likely underpowered.

INPUTS
------
  Data/Processed/analysis_sample.parquet   -- gvkeys in the analysis sample
  WRDS: comp.funda                         -- annual fundamentals (all US firms)

OUTPUTS
-------
  Data/Processed/audit_credibility_moderators.parquet
      gvkey, fyear, altman_z, high_distress, abs_da, high_daccruals, gc_opinion

NOTES
-----
- The Modified Jones model is estimated cross-sectionally on the FULL Compustat
  universe (not just sample firms) to avoid small-sample bias in industry-year
  regressions. Minimum 10 observations per SIC2 × year cell required.
- Altman Z is computed for the full population but only sample firms are saved.
- WRDS username: nhwang
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import wrds

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
PROC     = ROOT / "Data" / "Processed"
ANALYSIS_SAMPLE = PROC / "analysis_sample.parquet"
OUT_PATH = PROC / "audit_credibility_moderators.parquet"

WRDS_USERNAME = "nhwang"

# Minimum observations per SIC2 × year for Jones model estimation
MIN_JONES_OBS = 10


# ── Step 1: Load sample gvkeys ──────────────────────────────────────────────

def load_sample_info():
    df = pd.read_parquet(ANALYSIS_SAMPLE, columns=["gvkey"])
    gvkeys = df["gvkey"].unique().tolist()
    print(f"  Sample gvkeys: {len(gvkeys)}")
    return gvkeys


# ── Step 2: Pull Compustat annual fundamentals ──────────────────────────────

def pull_compustat(db):
    """
    Pull fields needed for Altman Z, Modified Jones model, and GC opinions
    for ALL US firms (needed for cross-sectional Jones estimation).
    """
    print("  Pulling comp.funda (all US firms, 1998–2023)...")
    sql = """
        SELECT f.gvkey, f.fyear, f.sich AS sic,
               f.at, f.act, f.lct, f.re, f.oiadp, f.lt,
               f.prcc_f, f.csho, f.sale,
               f.ni, f.oancf, f.rect, f.ppegt,
               f.auopic
        FROM comp.funda f
        WHERE f.indfmt  = 'INDL'
          AND f.datafmt = 'STD'
          AND f.popsrc  = 'D'
          AND f.consol  = 'C'
          AND f.fyear BETWEEN 1998 AND 2023
          AND f.fic = 'USA'
    """
    df = db.raw_sql(sql)

    # Coerce all numeric columns to float64
    num_cols = ["at", "act", "lct", "re", "oiadp", "lt",
                "prcc_f", "csho", "sale", "ni", "oancf", "rect", "ppegt"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    # Audit opinion to nullable int
    if "auopic" in df.columns:
        df["auopic"] = pd.to_numeric(df["auopic"], errors="coerce")

    # 2-digit SIC
    df["sic"] = pd.to_numeric(df["sic"], errors="coerce")
    df["sic2"] = (df["sic"] // 100).astype("Int64")

    # Exclude financials (6000-6999) and utilities (4900-4999)
    df = df[~df["sic"].between(6000, 6999) & ~df["sic"].between(4900, 4999)]

    # Require positive total assets
    df = df[df["at"] > 0].copy()

    print(f"  Compustat rows: {len(df):,}  ({df['gvkey'].nunique():,} firms)")
    return df


# ── Step 3: Altman Z-score ──────────────────────────────────────────────────

def compute_altman_z(df):
    """
    Z = 1.2*(WC/TA) + 1.4*(RE/TA) + 3.3*(EBIT/TA) + 0.6*(MVE/TL) + 1.0*(Sale/TA)

    WC = act - lct (working capital)
    RE = re (retained earnings)
    EBIT = oiadp (operating income after depreciation)
    MVE = prcc_f * csho (market value of equity)
    TL = lt (total liabilities)
    TA = at (total assets)
    """
    z = df.copy()

    wc_ta   = (z["act"] - z["lct"]) / z["at"]
    re_ta   = z["re"] / z["at"]
    ebit_ta = z["oiadp"] / z["at"]
    mve     = z["prcc_f"] * z["csho"]
    mve_tl  = np.where(z["lt"] > 0, mve / z["lt"], np.nan)
    sale_ta = z["sale"] / z["at"]

    z["altman_z"] = (1.2 * wc_ta + 1.4 * re_ta + 3.3 * ebit_ta
                     + 0.6 * mve_tl + 1.0 * sale_ta)

    n_valid = z["altman_z"].notna().sum()
    print(f"  Altman Z computed: {n_valid:,} valid obs")
    print(f"    mean={z['altman_z'].mean():.2f}  median={z['altman_z'].median():.2f}")
    return z


# ── Step 4: Modified Jones model discretionary accruals ──────────────────────

def compute_jones_da(df):
    """
    Modified Jones model (Dechow, Sloan & Sweeney 1995).

    Total accruals (cash flow approach):
        TA_it = NI_it - CFO_it

    Cross-sectional regression by SIC2 × year:
        TA_it / A_{i,t-1} = α * (1/A_{i,t-1})
                           + β1 * (ΔSale_it - ΔRec_it) / A_{i,t-1}
                           + β2 * PPE_it / A_{i,t-1}
                           + ε_it

    DA = residual (ε_it).
    abs_da = |DA|.
    """
    j = df.sort_values(["gvkey", "fyear"]).copy()

    # Lagged total assets
    j["at_lag"] = j.groupby("gvkey")["at"].shift(1)

    # Change in sales and receivables
    j["dsale"] = j.groupby("gvkey")["sale"].diff()
    j["drect"] = j.groupby("gvkey")["rect"].diff()

    # Total accruals (cash flow approach: NI - CFO)
    j["ta"] = j["ni"] - j["oancf"]

    # Scale everything by lagged assets
    j["ta_s"]     = j["ta"] / j["at_lag"]
    j["inv_a"]    = 1.0 / j["at_lag"]
    j["dsale_drec_s"] = (j["dsale"] - j["drect"]) / j["at_lag"]
    j["ppe_s"]    = j["ppegt"] / j["at_lag"]

    # Drop rows with missing Jones inputs
    jones_cols = ["ta_s", "inv_a", "dsale_drec_s", "ppe_s", "sic2", "fyear"]
    j_clean = j.dropna(subset=jones_cols).copy()
    j_clean = j_clean[np.isfinite(j_clean["ta_s"]) &
                       np.isfinite(j_clean["inv_a"]) &
                       np.isfinite(j_clean["dsale_drec_s"]) &
                       np.isfinite(j_clean["ppe_s"])]
    print(f"  Jones model: {len(j_clean):,} obs with valid inputs")

    # Cross-sectional OLS by SIC2 × year
    da_results = []
    groups = j_clean.groupby(["sic2", "fyear"])
    n_estimated = 0
    n_skipped   = 0

    for (sic2, year), grp in groups:
        if len(grp) < MIN_JONES_OBS:
            n_skipped += 1
            continue

        y_arr = grp["ta_s"].values
        X_arr = np.column_stack([
            grp["inv_a"].values,
            grp["dsale_drec_s"].values,
            grp["ppe_s"].values,
        ])

        try:
            coef, *_ = np.linalg.lstsq(X_arr, y_arr, rcond=None)
        except np.linalg.LinAlgError:
            n_skipped += 1
            continue

        fitted = X_arr @ coef
        residuals = y_arr - fitted

        for idx, resid in zip(grp.index, residuals):
            da_results.append({
                "idx": idx,
                "disc_accruals": resid,
            })
        n_estimated += 1

    print(f"  Industry-year cells estimated: {n_estimated:,} (skipped {n_skipped:,})")

    # Map DA back to the main dataframe
    if da_results:
        da_df = pd.DataFrame(da_results).set_index("idx")
        j["disc_accruals"] = da_df["disc_accruals"]
    else:
        j["disc_accruals"] = np.nan

    j["abs_da"] = j["disc_accruals"].abs()
    n_valid = j["abs_da"].notna().sum()
    print(f"  DA computed: {n_valid:,} valid obs")
    if n_valid > 0:
        print(f"    |DA| mean={j['abs_da'].mean():.4f}  median={j['abs_da'].median():.4f}")

    return j


# ── Step 5: Going-concern opinion ───────────────────────────────────────────

def compute_gc_opinion(df):
    """
    Going-concern indicator from auditor opinion code (auopic).

    Compustat auopic coding:
        1 = Unqualified
        2 = Qualified / going concern
        3 = No opinion (disclaimer)
        4 = Adverse
        5 = Unqualified with additional language

    gc_opinion = 1 if auopic ∈ {2, 3, 4} (any non-clean opinion).
    The vast majority of GC modifications appear as auopic = 2.
    """
    if "auopic" not in df.columns or df["auopic"].isna().all():
        print("  WARNING: auopic not available; gc_opinion will be NaN")
        df["gc_opinion"] = np.nan
        return df

    # Check the distribution of auopic values
    print("  auopic distribution:")
    print(df["auopic"].value_counts().sort_index().to_string())

    # GC = any non-unqualified opinion
    df["gc_opinion"] = df["auopic"].isin([2, 3, 4]).astype(float)
    df.loc[df["auopic"].isna(), "gc_opinion"] = np.nan

    n_gc = df["gc_opinion"].sum()
    n_total = df["gc_opinion"].notna().sum()
    print(f"  Going-concern opinions: {int(n_gc):,} / {n_total:,} ({n_gc/n_total:.1%})")
    return df


# ── Step 6: Filter to sample and save ────────────────────────────────────────

def filter_and_save(df, sample_gvkeys):
    """
    Keep only rows for analysis-sample gvkeys. Select output columns.
    """
    out = df[df["gvkey"].isin(sample_gvkeys)].copy()
    out = out[["gvkey", "fyear", "altman_z", "abs_da", "gc_opinion"]].copy()

    # Winsorize Altman Z and |DA| at 1/99 to limit outlier influence
    for col in ["altman_z", "abs_da"]:
        if out[col].notna().sum() > 0:
            lo = out[col].quantile(0.01)
            hi = out[col].quantile(0.99)
            out[col] = out[col].clip(lo, hi)

    n = len(out)
    n_z  = out["altman_z"].notna().sum()
    n_da = out["abs_da"].notna().sum()
    n_gc = out["gc_opinion"].notna().sum()
    print(f"\n  Sample rows: {n:,}")
    print(f"  Altman Z valid:  {n_z:,}")
    print(f"  |DA| valid:      {n_da:,}")
    print(f"  GC opinion valid: {n_gc:,}")

    out.to_parquet(OUT_PATH, index=False)
    print(f"\n  Saved: {OUT_PATH}")
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== 16_build_audit_credibility_moderators.py ===")

    print("\n[1] Loading sample gvkeys...")
    sample_gvkeys = load_sample_info()

    print("\n[2] Connecting to WRDS...")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)

    print("\n[3] Pulling Compustat annual fundamentals...")
    comp = pull_compustat(db)
    db.close()

    print("\n[4] Computing Altman Z-score...")
    comp = compute_altman_z(comp)

    print("\n[5] Computing Modified Jones discretionary accruals...")
    comp = compute_jones_da(comp)

    print("\n[6] Extracting going-concern opinion...")
    comp = compute_gc_opinion(comp)

    print("\n[7] Filtering to sample and saving...")
    out = filter_and_save(comp, sample_gvkeys)

    # Summary for sample firms
    print("\n  === Sample-firm summary ===")
    for col in ["altman_z", "abs_da", "gc_opinion"]:
        s = out[col].dropna()
        if len(s) > 0:
            print(f"  {col}: N={len(s):,}  mean={s.mean():.4f}  "
                  f"median={s.median():.4f}  sd={s.std():.4f}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
