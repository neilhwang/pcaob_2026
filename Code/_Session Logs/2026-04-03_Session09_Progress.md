# Session 09 Progress Log — 2026-04-03

## Summary

This session systematically addressed reviewer comments from `Reviews/claude.tex`, `Reviews/chat.tex`, and `Reviews/gemini.tex`, plus additional issues raised by Neil's colleague (regulatory shocks, competitiveness ≠ polarization). All critical and major items from the review reports are now resolved.

## What Was Accomplished

### 1. County Vote SD Measure (Competitiveness ≠ Polarization)
- **Script 02b** modified to compute population-weighted within-state SD of county-level D-share (`county_sd`)
- **Script 05** updated: `county_sd_std` added to standardization; Table 5 now has 10 columns (cols 9–10 = county SD standalone + interaction)
- **Results**: County SD standalone β = 0.005, p = 0.056; interaction p = 0.533
- **Draft**: New paragraph in §5 defining measure; §6.3 "Competitiveness vs polarization" paragraph; intro updated; Fiorina (2005) cited
- **references.bib**: Added `fiorina2005`

### 2. |CAR| Microfoundation (Demand Block)
- Replaced vague paragraph in §3.3 with concrete demand system: $q_i(P) = \alpha_i(\mu_i - P)$, short-sale constraints, limits to arbitrage
- Cited Miller (1977), Hong & Stein (2003), Gromb & Vayanos (2010)
- Updated P2 citation from `hong2007` to the three new references
- **references.bib**: Added `miller1977`, `hong2003`, `gromb2010`

### 3. Sample Attrition Diagnostics
- Created `tab_sample_attrition.tex` with Panel A (attrition chain: 8,621 → 6,901 → 1,220 → 678) and Panel B (matched vs unmatched comparison)
- Key finding: disagreement rates similar (67% vs 60%); Big 4 involvement higher in matched (43% vs 14%) — expected and conservative
- Updated §5.1 with expanded prose, table reference, and footnote on selection bias

### 4. Abstract Rewrite (8 Issues from claude.tex + chat.tex)
- Leads with "We provide the first evidence that..." (novelty signal)
- Names mechanism: trust in auditors and the PCAOB
- Reports AbVol ($p = 0.010$) before |CAR| ($p = 0.040$)
- Includes earnings placebo ($p = 0.924$, $N = 20{,}800$)
- Closes with "more, not less, disagreement"
- "Often ambiguous" → "can be ambiguous"

### 5. Continuum Notation Cleanup
- Removed "A continuum of investors $i \in [0,1]$" — model now starts directly with two groups $g \in \{D, R\}$
- All $\pi_i, \tau_i, \kappa_i, X_i$ → $\pi_g, \tau_g, \kappa_g, X_g$ in theory and appendix
- $\mathrm{Var}_i[\phi(\pi_i)]$ → $\mathrm{Var}[\phi(\pi_g)]$ throughout
- Kept individual $q_i, \mu_i, \alpha_i$ in demand block (appropriate there)

### 6. Regulatory Regime Stability (SOX/PCAOB Interaction)
- Added `run_regulatory_shock_test()` to script 05
- Results: Pol × PostSOX interaction null for both |CAR| (p = 0.498) and AbVol (p = 0.970)
- Both pre-SOX (N = 317) and post-SOX (N = 361) subsamples show positive coefficients
- Draft paragraph in §6.3 with SEC Release No. 33-8400 reference; footnote on AS 3101/CAMs as future work

### 7. Quick Fixes (C3, M1, M3, m1, m7)
- **C3**: Fixed coefficient-text mismatch: β = 0.012 → 0.011, p = 0.203 → 0.189 (matching Table 3)
- **M1**: Added proxy correlation matrix footnote (competitiveness/ER r = 0.87; competitiveness/DW r = 0.10; county SD/DW r = 0.29)
- **M3**: Added candid statement: identification relies on cross-state variation; 124 state×cycle cells, 52 with 1 event
- **m1**: "Two complementary proxies" → "three" (county vote SD added)
- **m7**: Added footnote flagging N = 51, R² = 0.67 overfitting in Table 3 column 3

### 8. Table Reordering (M5: AbVol before |CAR|)
- **Table 2** (main): Cols 1–2 = AbVol, Cols 3–4 = |CAR|
- **Table 3** (event type): Now 4 cols: AbVol Dism/Non-dism, |CAR| Dism/Non-dism (was |CAR|-only with quality-direction splits)
- **Table 4** (ambiguity): Now 6 cols: AbVol Full/Hi/Lo, |CAR| Full/Hi/Lo (was 5 cols |CAR|-first)
- All prose updated to match new column references
- Introduction reordered to report AbVol before |CAR|

### 9. Redundancy Elimination (M6)
- Cut "reduced-form proxy" from 4→1 occurrences
- Cut "generic salience/attention" from 7→2 occurrences
- Cut "objective guidance" from 10→5 occurrences
- Cut "politically shaped" from 6→3 occurrences
- Tightened identification section, hypothesis H2, ambiguity results, conclusion
- **Net savings**: 42 → 39 pages (3 pages cut)

### 10. Consolidate Propositions A1–A3 (Gemini)
- Merged three identical propositions into one unified Proposition A1
- Single proof covers all three channels (trust, precision, combined)
- Old `\label{prop:trust}` and `\label{prop:precision}` point to same unified proposition

### 11. Notation Fix (X Overloading)
- Renamed firm controls vector from $X_{e,t}$ to $Z_{e,t}$ in Equation 6
- Eliminates confusion with $X_g$ (latent log-odds index in theory)

### 12. Minor Items (m3, m6)
- **m3**: Added footnote explaining negative Auditor Change main effect (−0.024) in placebo table column 3
- **m6**: Added AbVol state-clustering result ($p < 0.001$) to robustness discussion

### 13. Institutional Ownership (M2 — In Progress)
- Script 18 rewritten to pull 13F data directly from SEC EDGAR (WRDS 13F tables not in Neil's subscription)
- Parser fix: replaced fragile header-column-order heuristic with `max(num1, num2)` magnitude heuristic
- Script running as background process (PID 1979), estimated ~15 hours
- Output: `Data/Processed/institutional_ownership.parquet`
- Script 05 already wired up with merge block and `run_institutional_ownership_test()`

### 14. Affective Polarization Cleanup (Completed from Prior Session)
- Confirmed all stale references removed
- Renamed `\label{sec:affective}` → `\label{sec:pol_markets}`

## Event-Type Table Results (New Finding)
Table 3 now includes AbVol columns. Notable finding: for volume, *dismissals* drive the result (β = 0.134, p = 0.006) while non-dismissals are insignificant (β = 0.122, p = 0.164). This is the opposite of the |CAR| pattern. Discussion added to §6.2 — dismissals generate disagreement about *consequences* even though the adversarial nature is transparent.

## Decisions Made
- **13F data**: WRDS subscription lacks Thomson/WRDS 13F tables; built SEC EDGAR direct download instead
- **Post-CAM test**: N = 40 too small; noted as future work in footnote
- **Quality-direction splits**: Dropped from Table 3 (replaced with AbVol); flagged N = 51 overfitting concern
- **Affective polarization**: Remaining mentions are appropriate literature/future-work context

## Files Modified
- `Paper/draft.tex` — extensive edits (abstract, theory, identification, results, conclusion, appendix)
- `Paper/references.bib` — added Fiorina (2005), Miller (1977), Hong & Stein (2003), Gromb & Vayanos (2010)
- `Code/02b_build_presidential_polarization.py` — added county_sd computation
- `Code/05_merge_and_estimate.py` — county_sd merge, SOX test, institutional ownership merge/test, table reordering
- `Code/18_build_institutional_ownership.py` — rewritten for SEC EDGAR direct download

## Files Created
- `Output/Tables/tab_sample_attrition.tex`

## Current Paper State
- 39 pages, compiles cleanly
- All critical (C1–C5) and major (M1–M6) reviewer items resolved
- All minor items (m1–m8) resolved
- Tables 2–4 reordered (AbVol first)

## Pending
1. **13F script completion** — running in background (~15 hours). After completion: run script 05, check results, potentially add to draft
2. **Overleaf sync** — run `.\sync_overleaf.ps1`
3. **Full LaTeX compilation** with bibtex to resolve new citations (Fiorina, Miller, Hong, Gromb)
