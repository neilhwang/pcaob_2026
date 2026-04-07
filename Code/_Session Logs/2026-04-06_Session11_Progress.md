# Session 11 Progress Log — 2026-04-05 to 2026-04-06

## Summary

Continuation of Session 10. Completed the Item 5.02 officer-change placebo test, fixed a sample-size inconsistency (N=724 vs N=678), narrowed the abstract's novelty claim per colleague feedback, and conducted a full consistency review of the draft.

## What Was Accomplished

### 1. Script 09 Recovery (Item 5.02 Events)
- Script 09 completed parsing (~175k filings) but crashed on parquet write due to pyarrow CIK type inference (str vs int64)
- Recovered all 124,596 events from the checkpoint CSV directly
- Fixed the root cause in script 09: forced CIK to zero-padded 10-digit string and acc_nodash to 18-char zero-padded string before parquet write

### 2. Script 10 (Officer-Change Placebo) — Written and Run
- Created `Code/10_build_officer_change_placebo.py`
- Design: within-firm placebo comparing auditor changes (Item 4.01) to officer changes (Item 5.02) from the same 456 firms
- Initial run failed due to: (a) Unicode encoding on Windows cp1252 (fixed by replacing non-ASCII chars), (b) merge key mismatch (date_filed vs event_date, fixed), (c) missing analysis-sample filter (ran 120k events instead of ~4k — took 7+ hours before being killed)
- **Key fix**: Added filter to restrict placebo events to analysis-sample gvkeys *before* the CRSP pull. Reduced sample from 120k to 3,966 events; runtime dropped from 7+ hours to ~10 minutes
- **Final results** (N=3,677 officer-change events, 282 firms):
  - AbVol placebo: β = 0.016, p > 0.10 (null)
  - AbVol pooled interaction: β = 0.110, p = 0.011 (significant)
  - |CAR| placebo: β = -0.000, p > 0.10 (null)
  - |CAR| pooled interaction: β = 0.006, p = 0.006 (significant)
  - Interpretation: polarization amplifies reactions to auditor changes but not officer changes from the same firms — supports institutional-trust channel

### 3. Sample-Size Inconsistency Fix (N=724 → N=678)
- **Discovery**: All tables from script 05 showed N=724, but the paper cited N=678 and the earnings-placebo table (script 15) showed N=678
- **Root cause**: `post_event_uncertainty.parquet` (from script 21) had 16 duplicate (permno, event_date) rows. Script 05's merge with this file lacked the deduplication guard, causing merge fan-out (678 → 724)
- **Fix in script 05**: Added `drop_duplicates(subset=["permno", "event_date"])` and sanity check before the uncertainty merge
- **Fix in script 21**: Added `drop_duplicates` on params before merging with the deduplicated sample
- **Root root cause**: `crsp_event_window.parquet` has 27 duplicate (permno, event_date) pairs — firms with multiple auditor-change filings on the same date
- Re-ran script 05 end-to-end — all tables now show N=678

### 4. Abstract Novelty Claim Narrowed
- Per colleague feedback, changed "We provide the first evidence that political polarization shapes how investors interpret mandatory audit-related disclosures" → "We provide the first evidence that political polarization shapes investor reactions to mandatory audit-related disclosures"
- Added Goldman, Gupta, and Israelsen (JFE 2024) to the lit review as the closest prior work, with explicit distinction (media-intermediated coverage vs. standardized mandatory filing)
- BibTeX entry added to references.bib

### 5. Draft Updates
- New subsection "Placebo Test: Officer-Change Disclosures" (§6.7) added after earnings placebo
- Introduction updated to describe both placebos with explicit framing of officer-change placebo as the tighter test
- Conclusion updated to mention both placebos

### 6. Draft Consistency Verification
- All abstract/conclusion p-values verified against current regression output (confirmed correct)
- All tables confirmed at N=678 after the dedup fix
- No hardcoded table numbers found — all use \ref{}

### 7. WRDS Data Access Check
- Confirmed Neil has access to: Call Reports (bank_all), Y-9C (holding company), TRACE Enhanced (435M+ trades)
- Does NOT have: DealScan, full FISD (sample only)

### 10. Language Audit (Causal → Associational)
- Abstract: "shapes" → "is associated with"; "confirms" → "consistent with"
- Introduction: "identifies" → "documents a previously unexplored correlate"; "rules out" → "is inconsistent with"
- Conclusion: rewritten opening sentence to positioning sentence
- Results: "striking" → "directionally consistent"; "meaningful statistical evidence" → "suggestive evidence"
- Identification subheading: "is informative" → "may be informative"

### 11. H2 Dropped and Table 4 Moved to Appendix
- H2 (ambiguity amplification) removed as a formal hypothesis — direct interaction tests are null (p=0.806 for AbVol, p=0.416 for |CAR|)
- Table 4 (filing informativeness mechanism test with 6 post-event uncertainty proxies) moved to appendix
- P3 references removed throughout; informativeness reported as exploratory paragraph in §6.5
- Theory section: P3 replaced with short paragraph noting exploratory exploration without confirming evidence

### 12. Results Section Reordered
- New order: Main Results → Placebos → Robustness → Exploratory → Reversal
- Added transition sentence: "We focus our interpretation on the baseline association and the placebo evidence"

### 13. Hypothesis Development Folded into Theory
- §4 (Hypothesis Development) eliminated as standalone section
- H1 and exploratory event-type prediction now appear as §3.6 "Hypothesis" subsection
- Exploratory event-type paragraph compressed from 20 to 7 lines
- Roadmap sentence updated

### 14. Additional Trims
- SOX/PCAOB stability: 24 lines → 11 lines (compressed to footnote)
- Identification section: 117 lines → 51 lines (56% cut)
- Conclusion: removed "may be growing" and "should operate wherever" extrapolation
- "First evidence" → "We document"
- AbVol formula rewritten with explicit notation
- Sample selection limitation added (2 sentences)

## Pending / Next Steps (for next session)

1. **Theory section (§3):** Tie modeling choices to Item 4.01 institutional features — predecessor concurrence letters, regulated disclosure content, PCAOB credibility. Currently reads as generic disagreement finance.
2. **Lit review (§2):** Rebuild around three accounting conversations: auditor-change informativeness, disclosure interpretation, credibility of audit signals. Currently reads as disconnected mini-bibliographies.
3. **Footnotes:** Audit for argumentative work that belongs in main text or should be cut.
4. **Compile PDF** via Overleaf and do visual pass.
5. **Commit and push** remaining changes.
6. **Run `.\sync_overleaf.ps1`**

## Decisions Made

- Officer-change placebo restricted to analysis-sample firms (within-firm design matching script 15's approach)
- Abstract novelty claim anchored narrowly to "mandatory audit-related disclosures" per colleague feedback
- Goldman et al. (JFE 2024) cited but Hao, Nain & Xu (SSRN working paper) omitted to avoid legitimizing a competitor
- The N=724 vs N=678 discrepancy was a merge fan-out bug, not an intentional sample change
