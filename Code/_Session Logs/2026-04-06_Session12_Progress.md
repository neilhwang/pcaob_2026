# Session 12 Progress Log — 2026-04-06

## Summary

Major revision session responding to a comprehensive set of reviewer-style critiques. Restructured the lit review, compressed the theory, narrowed the contribution, downscaled mechanism claims, discovered and fixed a severe data bug in the disagreement variable, found a new significant within-filing interaction (explicit_cause), and repositioned the paper from an audit-credibility mechanism story to an interpretation-of-substantive-content story. Also scoped the feasibility of investor-level political data (13F + FEC) as a future identification upgrade.

## What Was Accomplished

### 1. Lit Review Rebuilt (§2)
- Five disconnected subsections → three conversation-organized subsections
- §2.1 "Auditor-Change Disclosures and Market Reactions": merged old §2.1+§2.2, grouped by what each paper established
- §2.2 "Investor Heterogeneity, Beliefs, and Disclosure Interpretation": merged old §2.4+§2.5, connects disagreement models to political identity literature
- §2.3 "Political Economy and Credibility of Audit Institutions": reworked old §2.3, reframed as supply-vs-demand distinction

### 2. Footnotes Audited
- Promoted to main text: selection-bias argument (matched vs unmatched), retail ownership interaction
- Compressed: ER index/presidential footnote from 25 to 10 lines
- 8 other footnotes kept as-is

### 3. Theory Section (§3) Anchored and Compressed
- Added "Mapping to Item 4.01" paragraph connecting model primitives to filing features
- Section opener reframed: "stylized model... not structurally estimated"
- §3.4 (Signal Ambiguity) cut from 54 to 20 lines — plain English only, formal apparatus in appendix
- Discussion paragraph cut from 18 to 8 lines
- Testable Predictions ambiguity paragraph cut from 10 to 5 lines
- Total theory section: 319 → ~250 lines

### 4. Contribution Narrowed to One
- Three contributions → one primary contribution to auditor-change disclosure literature
- Placebos are supporting specificity evidence, not separate contribution
- Regulation implication kept as one modest closing sentence

### 5. Conclusion Trimmed
- CAM/going-concern/internal-control speculation deleted
- ANES discussion reduced to one sentence
- Reversal framed as descriptive check, not theory test

### 6. Overclaiming Language Fixed Throughout
- 6 instances of "confirm/confirming" → "indicate/consistent with/suggests"
- "We identify this effect" → "We examine this prediction"
- "interpretation varies" → "market reactions vary"
- "politically polarized states" → "politically competitive states" (proxy framing)
- Removed two passages that converted null robustness results into theoretical wins

### 7. Four-Part Proxy-Construct Downscaling
- Intro: mechanism language changed to if-then conditional
- Contribution: "we show interpretation varies" → "we show market reactions vary"
- Identification: new paragraph explicitly separating what is observed vs. inferred vs. not identified
- Conclusion: reversal softened to "inconsistent with overreaction, consistent with though not unique to persistent heterogeneity"

### 8. Theory-Empirics Gap Fixes
- Event-type subsection compressed from ~30 to ~10 lines
- Post-event persistence reframed as descriptive check
- Channel-indistinguishability acknowledged in identification section
- "Confirm" language purged from robustness section

### 9. Disagreement Variable Bug Discovered and Fixed
- **Root cause**: Script 01 line 361 used `re.search(r"disagreement", text)` which flags TRUE whenever the word appears anywhere — including mandatory boilerplate "there were no disagreements"
- **Old rate**: 79.8% of analysis sample flagged as having disagreements
- **Fix**: New regex distinguishes affirmative mentions ("had the following disagreements") from negated boilerplate ("there were no disagreements")
- **New rate**: 4.0% (27 events in analysis sample) — consistent with literature expectations
- Script 01 updated with corrected regex
- Script 22a created to fix the raw parquet without re-downloading from EDGAR
- `crsp_event_window.parquet` updated
- Script 05 re-run — all tables regenerated
- `high_ambiguity` rate changed: 17.7% → 58.3% (because more events now correctly classified as no-disagreement)
- Draft text updated: disagreement rates (67% vs 60% → 4.6% vs 4.6%), high-ambiguity count

### 10. Within-Filing Interaction Tests (Script 22)
- **explicit_cause × Pol → AbVol: β=+0.208, p=0.013** — significant
  - Subsample: cause=1 β=0.212, p=0.022; cause=0 β=0.037, p=0.482
  - Polarization effect on volume concentrated in filings with substantive causal content
- Real disagreement × Pol: null (p=0.689 AbVol, p=0.716 |CAR|) — N=27, no power
- Reportable event × Pol: null (p=0.455 AbVol, p=0.759 |CAR|)
- Corrected composite × Pol: null (p=0.727 AbVol, p=0.839 |CAR|)
- Quality transitions: all null
- **Conclusion**: Within-filing evidence supports interpretation-of-substantive-content mechanism, not audit-credibility-specific mechanism

### 11. Paper Repositioned
- Dropped audit-credibility as headline mechanism claim
- New framing: "politically mediated interpretation of substantive auditor-change content"
- Audit credibility remains the setting's motivation and the theory's starting point, but is not claimed as the empirically proven channel
- explicit_cause result added to: abstract, intro results summary, contribution paragraph, exploratory results section, conclusion
- Credibility-specific language removed from: P1/P2 predictions, identification section, officer-change placebo, conclusion

### 12. Investor-Level Political Data Feasibility (Option 2)
- Assessed feasibility of 13F + FEC matching for investor-base political heterogeneity
- Project already has 13F data downloaded and parsed (script 18)
- Published templates: Wintoki & Xi (2019, JFQA), Mahmood (2021, SSRN)
- Pipeline: FEC individual contributions → fuzzy name match to 13F filers → aggregate to firm-event level
- Timeline: 6-8 weeks; expected 40-70% coverage of institutional shares
- **Recommended as post-first-submission project** — strongest identification upgrade available

## Decisions Made
- Paper repositioned from audit-credibility mechanism to interpretation-of-substantive-content mechanism, based on within-filing evidence
- explicit_cause interaction included as supporting evidence; disagreement interaction reported briefly as null (power limitation)
- Disagreement variable fix propagated through entire pipeline
- 13F + FEC matching deferred to post-submission

## Scripts Created/Modified
- `Code/22_credibility_salience_test.py` — within-filing interaction tests (new)
- `Code/22a_fix_disagreement_variable.py` — disagreement variable fix (new)
- `Code/01_build_edgar_event_file.py` — corrected disagreement regex (modified)
- `Data/Processed/auditor_changes_raw.parquet` — updated with corrected disagreements
- `Data/Processed/crsp_event_window.parquet` — updated with corrected disagreements
- `Data/Processed/analysis_sample.parquet` — regenerated via script 05

## Files Archived
- `Data/Processed/_Archive/auditor_changes_raw_pre_disagree_fix.parquet`

## Pending / Next Steps

### Completed since initial log
- FEC bulk data downloaded (indiv00–indiv24, including indiv20 re-download)
- Committee master files downloaded (cm00–cm24)
- rapidfuzz installed
- Attrition table updated with corrected disagreement rates (4.6% vs 4.6%)
- Packages (pyarrow, requests, wrds) installed in Python 3.13

### Still pending
1. **13F filer-level holdings — BLOCKED**
   - Thomson Reuters S34 on WRDS: permission denied (Neil's subscription doesn't include `tfn` or `tr_13f` schemas)
   - SEC structured ZIP downloads: 404 errors on all quarters — the URL pattern in script 23 is wrong
   - **Fix for next session**: either (a) try WRDS tables `wrdsapps.own_13f_holding` or `wrdssec.holding_13f` which may be accessible, or (b) port the URL-scraping logic from `get_structured_zip_urls()` in script 18 into script 23 to get correct SEC download paths
   - FEC Step 1 is done and cached (`fec_employer_lean_cache.parquet`, 1.78M employers)
2. **FEC indiv10 encoding error** — script skipped the 2009–2010 cycle due to UTF-8 decode error; fix by adding `encoding="latin-1"` to the `read_csv` call in script 23
3. **Compile PDF** via Overleaf and visual pass
4. **Commit and push** changes
5. **Run `.\sync_overleaf.ps1`**

## Scripts Created This Session
- `Code/22_credibility_salience_test.py` — within-filing interaction tests
- `Code/22a_fix_disagreement_variable.py` — disagreement variable fix
- `Code/23_build_investor_political_heterogeneity.py` — 13F + FEC matching pipeline (Step 1 working, Step 2 blocked)

## Open Questions
- Target journal: RAST remains the primary target given current evidence base. TAR/JAR/JAE would likely require the investor-level data (Option 2).
- Neil to assign script numbers for scripts 22, 22a, and 23
- Which WRDS 13F tables does Neil have access to? Run `! python -c "import wrds; db=wrds.Connection(wrds_username='nhwang'); print(db.list_tables('wrdsapps'))"` to check
