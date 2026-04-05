# Session 10 Progress Log — 2026-04-04

## Summary

This session addressed (1) a broken checkpoint-resume bug in script 09, (2) the specification-hunting concern by adding AbVol robustness panels across all key tables, (3) comprehensive draft revisions to reflect the AbVol-primary framing, and (4) planning for the Item 5.02 placebo test.

## What Was Accomplished

### 1. Script 09 Checkpoint Bug Fix
- **Problem**: Script 09 (`09_build_placebo_event_file.py`) stopped at 36% (62,265/174,150) and failed to resume correctly — "Remaining to parse: 174150" despite 44,500 already parsed.
- **Root cause**: `acc_nodash` read from checkpoint CSV as `int64` (leading zeros stripped), but candidates dataframe has 18-char zero-padded strings. `.isin()` matched nothing.
- **Fix**: Zero-pad checkpoint `acc_nodash` to 18 chars (`str.zfill(18)`) and deduplicate (44k duplicates from prior broken runs).
- **Additional fix**: Added `CANDIDATES_CACHE` (parquet) so Steps 1-3 (EFTS + index download, ~55 min) are cached. Future restarts skip straight to parsing.
- **Status**: Script 09 is currently running, at ~34% of 112,398 remaining filings. ETA ~14-15 hours.

### 2. AbVol-Primary Reframing — Draft Edits
- **Introduction** (line 117): Swapped `regresses |CAR| and abnormal trading volume` → `regresses abnormal trading volume and |CAR|`
- **Dependent Variables** (§5.2): Swapped order — AbVol definition now comes first, |CAR| second; |CAR| described as "secondary dependent variable"
- **Descriptive Statistics** (§6.1): AbVol presented before |CAR|
- **Main Results** (§6.2): Economic magnitude leads with AbVol; forward reference to robustness table added
- **Introduction robustness summary** (lines 133-141): Updated to describe AbVol robustness across three proxies

### 3. AbVol Robustness Panel (Table 5)
- **Problem**: Robustness table was CAR-only (10 columns). No alternative-proxy testing for AbVol — the primary outcome.
- **Fix**: Rewrote `run_robustness()` in script 05 to run identical 10-spec battery for both AbVol and |CAR|.
- **Output**: Two-panel table — Panel A (AbVol), Panel B (|CAR|) — each with 10 columns.
- **Key findings**:
  - AbVol baseline: β = 0.111, p < 0.01 (***)
  - AbVol Pres. ER: β = 0.081, p < 0.05 (**)
  - AbVol DW-NOMINATE: β = 0.100, p < 0.10 (*)
  - AbVol small-firm: β = 0.137, p < 0.05 (**)
  - AbVol clustering: p < 0.001 (state and two-way)
  - AbVol county-level: null (β = 0.022) — addressed with horse-race test
- **Draft**: Complete rewrite of §5.3 discussing both panels systematically, with summary paragraph.

### 4. Permutation Test — Both Outcomes (Table 7)
- **Problem**: Permutation test was CAR-only.
- **Fix**: `run_permutation_test()` now runs 5,000 permutations for both AbVol and |CAR|.
- **Results**: AbVol permutation p = 0.004; |CAR| permutation p = 0.029.
- **Draft**: Updated §5.3 prose.

### 5. Earnings Placebo — Both Outcomes (Table 10)
- **Problem**: Earnings-announcement placebo was CAR-only.
- **Fix**: Script 15 now computes AbVol for all ~20,800 earnings events and produces 6-column table (AbVol cols 1-3, |CAR| cols 4-6).
- **Results**: AbVol placebo null (β = -0.010, p > 0.10); pooled interaction β = 0.141, p < 0.01. Clean null for both outcomes.
- **Draft**: Complete rewrite of §5.5.

### 6. Affective Polarization Table — Restructured (Table 6)
- **Problem**: Table had 4 CAR + 1 AbVol column; not included in draft.
- **Fix**: Restructured to 3 AbVol + 3 |CAR| columns (baseline, +affective, full spec for each).
- **Results**: AbVol ideological β = 0.111-0.130, p < 0.01 across all specs. AP × Exposure null for both outcomes.
- **Draft**: New subsection (§5.4 "Ideological versus Affective Polarization") added between permutation test and placebo test. ~10 sentences.
- **Lit review** (§2.4): Updated to reference the empirical test rather than framing it as future work.
- **Conclusion**: Updated to report the affective null as a finding.

### 7. County-Level AbVol Null — Addressed
- **Issue**: County competitiveness significant for CAR (p = 0.036) but null for AbVol (β = 0.022).
- **Analysis**: Horse-race regression (state + county simultaneously) shows state dominates for AbVol (β = 0.125, p = 0.012; county β = -0.030, p = 0.535).
- **Draft**: Robustness section already had theoretical explanation (prices = marginal local investors; volume = both sides of disagreement). Added horse-race evidence as supporting empirical result.

### 8. Script 10 (Item 5.02 Placebo Merge) — Written
- `Code/10_build_officer_change_placebo.py` written and syntax-verified.
- Takes `placebo_events_raw.parquet` from script 09, links CIK→PERMNO via WRDS, pulls CRSP daily data, computes CAR(-1,+1) and AbVol(-1,+1), merges with polarization and Compustat controls, runs same regression design as script 15 (earnings placebo).
- Output: 6-column table (AbVol cols 1-3, |CAR| cols 4-6) comparing auditor-change vs. officer-change events.
- Ready to run when script 09 completes (`placebo_events_raw.parquet` must exist first).

### 9. Draft Consistency Check
- Verified all abstract and conclusion p-values against current regression output (confirmed correct: AbVol placebo p=0.393, |CAR| placebo p=0.918).
- No hardcoded table numbers found — all use `\ref{}`.
- No coefficient/significance inconsistencies between prose and tables.

## Pending / Next Steps

1. **Script 09 still running** — ~14-15 hours remaining. When complete, write and run script 10 for the Item 5.02 placebo test.
2. **Draft consistency check** — background agent reviewing all prose against table numbers. Results pending.
3. **Sync to Overleaf** — run `.\sync_overleaf.ps1` to push all changes.

## Decisions Made

- AbVol is the primary outcome throughout; |CAR| is secondary. All tables now lead with AbVol.
- The AbVol robustness across three proxies (competitiveness, ER index, DW-NOMINATE) is the paper's main defense against specification-hunting concerns.
- The county-level AbVol null is interpreted as evidence that state-level variation captures the relevant belief-forming environment for disagreement-driven volume.
- The affective polarization null is included in the paper as a substantive finding (mechanism operates through ideological sorting, not partisan animus).
