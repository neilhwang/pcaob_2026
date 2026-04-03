# Session 06 Progress Log — 2026-04-03

## Session Overview

Implemented all changes from reviews/gemini.tex (5 iterations of critique).
All 6 items implemented; none deferred.

---

## Changes Implemented

### A — Hypothesis 1 reordered (AbVol primary)

H1 now leads with AbVol as the primary formal prediction (per Proposition 1),
with |CAR| explicitly labeled as a secondary empirical prediction. This fixes
the internal inconsistency where the theory section labeled AbVol as primary
but H1 and Table 2 presented |CAR| first.

Text changed from:
  "both the absolute cumulative abnormal return (|CAR[-1,+1]|) and abnormal
  trading volume around auditor-change events are increasing in polarization"

To:
  "abnormal trading volume (AbVol) around auditor-change events is increasing
  in political polarization (primary formal prediction, per Proposition 1),
  and absolute cumulative abnormal returns (|CAR[-1,+1]|) are increasing in
  polarization as a secondary empirical prediction."

### B — Permutation test (script 05, Table 7)

Added run_permutation_test() to Code/05_merge_and_estimate.py:
- Reshuffles competitive_std 5,000 times, re-estimates OLS β each time
- Reports: actual β, clustered SE p-value, permutation p-value (one-sided)
- Writes Output/Tables/tab07_permutation.tex
- Wired into main() after Table 5 (robustness)

Draft updated:
- Added prose in §6.5 describing the permutation result
- Added \input{../Output/Tables/tab07_permutation}
- Permutation numbers ([PERM_PCTILE] and [PERM_P]) are placeholders —
  Neil must run script 05 and update these two values in §6.5

### C — Appendix: mean-preserving spread clarification

Added "Remark (role of mean-preservation)" after the Step 2 proof. Key points:
- Mean-preservation is sufficient, not necessary, for variance to increase
- The formula Var = ω_D ω_R (φ(π_D) - φ(π_R))² increases with |π_D - π_R|
  regardless of what happens to the mean
- Mean-preservation serves to isolate dispersion from level effects, not to
  make the monotonicity result logically possible
- Asymmetric prior shifts (one group moves, other stays) still increase variance
  as long as the gap widens

### D — Appendix: τ_g, κ_g group structure clarified

Added a clarifying sentence before Propositions A1-A3:
- τ_i ≡ τ_{g(i)} and κ_i ≡ κ_{g(i)} — group-indexed, same two-point
  distribution (mass ω_D at group-D value, mass ω_R at group-R value)
- Directly answers the reviewer's question about whether τ and κ share
  the same distributional structure as π

### E — "Interpretive latitude" overuse addressed

Reduced from 5 instances to 2:
- Line ~447: "leaves the greatest interpretive latitude about the nature..."
  → "provides the least objective informational guidance about the nature..."
- Line ~485 (P3): "those where the filing leaves greater interpretive latitude"
  → "those where the filing provides less objective informational guidance"
- Line ~554 (H2): "where the filing leaves greater interpretive latitude about
  fault, severity..." → "where the filing provides limited objective guidance
  about fault, severity..."

Two retained uses: (1) the §3.4 definition sentence (canonical use), and (2)
the exploratory prediction where it echoes the §3.4 definition intentionally.

### F — §2.4 and §2.5 merged

"Political Polarization and Capital Markets" (§2.4) and "Affective Polarization,
Social Identity, and Information Processing" (§2.5) merged into a single
subsection titled "Political Polarization and Capital Markets" with label
sec:affective (preserved for internal references).

Structure of merged section:
1. Capital markets evidence (kempf2024, gerberhuber2009, coibion2020, barrioshochberg2021)
2. Explicit delineation: ideological polarization vs. affective polarization
   (iyengar2012, iyengar2019, mason2015, druckmanlevy2021)
3. Both forms relevant here + iyengar2019 nonpolitical spillover argument
4. Empirical distinction between our proxies: competitiveness = ideological
   dimension; ANES = affective animus; tyler2023 measurement caveat
5. Extension to auditing context

The internal cross-reference "reviewed in Section~\ref{sec:affective} below"
in the old §2.4 is now moot (the label sec:affective is on the merged section
itself) — this was cleaned up by merging.

---

## Action Required from Neil

1. **Run script 05**: `python Code/05_merge_and_estimate.py`
   This produces tab07_permutation.tex with the actual permutation results.

2. **Update two placeholder values in draft.tex §6.5** (Robustness section):
   - [PERM_PCTILE] → the percentile where the actual β falls (e.g., "97")
   - [PERM_P] → the permutation p-value (e.g., "0.030")
   These are logged by the script:
     "Permutation p-value (one-sided): X.XXXX  (β_actual=X.XXXX falls at XX.X-th percentile of null)"

---

## Items NOT implemented (with rationale)

- Elevating DW-NOMINATE to co-primary: DW-NOMINATE state gap is null (p=0.522);
  making it co-primary would foreground a null result and undercut the paper.
- Making size split central: Column 10 (small firms) gives p=0.175 — directionally
  consistent but not significant; keeping in robustness is more honest.
- Relaxing 100-day trading day filter: Would require re-running full pipeline
  (scripts 03–10), uncertain payoff, risk of changing point estimates.

---

## Files Modified

- Paper/draft.tex (H1, §2.4+2.5 merger, permutation prose, appendix notes,
  "interpretive latitude" variations)
- Code/05_merge_and_estimate.py (run_permutation_test() function, main() wiring,
  OUTPUTS docstring)
