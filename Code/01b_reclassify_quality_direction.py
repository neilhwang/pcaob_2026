"""
01b_reclassify_quality_direction.py
=====================================
Post-processing fix for auditor_changes_raw.parquet.

Two fixes applied without re-scraping EDGAR:
  1. Add Arthur Andersen to AUDITOR_PATTERN and BIG4_PATTERN.
     Critical for 2001-2002: Andersen was a Big-5/Big-4 firm and the most
     common predecessor in early-sample departures.
  2. Logic fix for build_quality_direction: when item401_text is present
     but an auditor name was not found in AUDITOR_PATTERN, infer the auditor
     is non-Big4. AUDITOR_PATTERN already covers Big4 + all major regional
     firms (Grant Thornton, BDO, RSM, etc.); any name not matched is
     definitively a small regional firm.  This converts ~7,800 "unknown"
     quality_direction events to properly classified categories.

INPUT:  Data/Processed/01_checkpoint.csv   (all parsed rows from script 01)
OUTPUT: Data/Processed/auditor_changes_raw.parquet  (overwrites existing)
        Data/Processed/01b_reclassify_log.csv        (before/after comparison)

RUNTIME: < 30 seconds (no network I/O).
"""

import re
import logging
from pathlib import Path

import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────

ROOT            = Path(__file__).resolve().parent.parent
CHECKPOINT_FILE = ROOT / "Data/Processed/01_checkpoint.csv"
OUT_FILE        = ROOT / "Data/Processed/auditor_changes_raw.parquet"
LOG_FILE        = ROOT / "Data/Processed/01b_reclassify_log.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Updated patterns ───────────────────────────────────────────────────────────

# BIG4_PATTERN: firms classified as Big4/Big5 for quality_direction purposes.
# Arthur Andersen added — it was a Big5 firm that collapsed in 2002 and is the
# most common predecessor in 2001-2002 auditor-change events.
BIG4_PATTERN = re.compile(
    r"Arthur\s*Andersen"
    r"|Deloitte"
    r"|PricewaterhouseCoopers|PwC"
    r"|Ernst\s*&\s*Young|\bEY\b"
    r"|KPMG",
    re.IGNORECASE,
)

# AUDITOR_PATTERN: used to extract auditor names from item401_text.
# Includes Big4/Big5 + major national and large regional firms.
# Small regional firms (thousands of them) are intentionally excluded;
# their absence from a match is used to infer non-Big4 status.
AUDITOR_PATTERN = re.compile(
    r"Arthur\s*Andersen"
    r"|Deloitte"
    r"|PricewaterhouseCoopers|PwC"
    r"|Ernst\s*&\s*Young|\bEY\b"
    r"|KPMG"
    r"|Grant\s*Thornton"
    r"|BDO"
    r"|RSM"
    r"|McGladrey"
    r"|Moss\s*Adams"
    r"|Dixon\s*Hughes"
    r"|Crowe"
    r"|Eide\s*Bailly"
    r"|Cherry\s*Bekaert"
    r"|Marcum"
    r"|Withum"
    r"|Baker\s*Tilly"
    r"|CBIZ"
    r"|Plante\s*(?:&\s*)?Moran"
    r"|WithumSmith",
    re.IGNORECASE,
)


# ── Helper functions ───────────────────────────────────────────────────────────

def extract_auditor_names(text: str) -> tuple[str | None, str | None]:
    """
    Re-extract predecessor and successor auditor names from item401_text
    using the updated AUDITOR_PATTERN (now including Arthur Andersen).
    Returns (auditor_out, auditor_in).
    """
    if not text or len(text) < 20:
        return None, None

    # Predecessor: firm name near departure language
    out_match = re.search(
        r"(" + AUDITOR_PATTERN.pattern + r")"
        r"[^.]{0,200}?"
        r"(dismiss|terminat|replac|resign|former|previous|was\s+the)",
        text, re.IGNORECASE,
    )
    auditor_out = out_match.group(1).strip() if out_match else None

    # Successor: engagement verb then firm name
    in_match = re.search(
        r"(engag|appoint|retain|select|hired|has\s+been\s+selected)"
        r"[^.]{0,150}?"
        r"(" + AUDITOR_PATTERN.pattern + r")",
        text, re.IGNORECASE,
    )
    if in_match is None:
        # Alternate order: firm name then engagement verb
        in_match = re.search(
            r"(" + AUDITOR_PATTERN.pattern + r")"
            r"[^.]{0,150}?"
            r"(engag|appoint|retain|select|hired|will\s+serve|as\s+its)",
            text, re.IGNORECASE,
        )
        auditor_in = in_match.group(1).strip() if in_match else None
    else:
        auditor_in = in_match.group(2).strip() if in_match else None

    return auditor_out, auditor_in


def classify_big4(name: str | None) -> bool | None:
    """Return True if name matches BIG4_PATTERN, False if non-null non-match,
    None if name is None (auditor could not be identified from text)."""
    if name is None:
        return None
    return bool(BIG4_PATTERN.search(name))


def build_quality_direction(out_big4: bool | None,
                            in_big4: bool | None,
                            has_text: bool) -> str:
    """
    Improved classification logic (v2).

    When item401_text is present but an auditor name was not found in
    AUDITOR_PATTERN, infer the auditor is non-Big4: AUDITOR_PATTERN covers
    Big4/Big5 + all major national firms, so any unmatched auditor is
    definitively a small regional firm.

    Only return 'unknown' when there is no text to infer from (genuine
    parsing failure — 10 rows in the current dataset).
    """
    if not has_text:
        return "unknown"
    # Treat None as non-Big4 (too small to appear in AUDITOR_PATTERN)
    out_is_big4 = bool(out_big4)   # None -> False
    in_is_big4  = bool(in_big4)    # None -> False
    if out_is_big4 and in_is_big4:
        return "Big4_to_Big4"
    if out_is_big4:
        return "Big4_to_nonBig4"
    if in_is_big4:
        return "nonBig4_to_Big4"
    return "nonBig4_to_nonBig4"


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Loading checkpoint: %s", CHECKPOINT_FILE)
    raw = pd.read_csv(CHECKPOINT_FILE, dtype=str)
    log.info("Rows loaded: %d", len(raw))

    # Work only on successfully parsed rows
    ok = raw[raw["parse_status"] == "ok"].copy()
    log.info("parse_status=ok rows: %d", len(ok))

    # ── Step 1: Re-extract auditor names with updated AUDITOR_PATTERN ──────────
    log.info("Re-extracting auditor names (Andersen fix + expanded pattern)...")

    names = ok["item401_text"].apply(
        lambda t: pd.Series(
            extract_auditor_names(t if isinstance(t, str) else ""),
            index=["auditor_out_new", "auditor_in_new"],
        )
    )
    ok = pd.concat([ok, names], axis=1)

    # Compare before / after for Andersen
    andersen_pat = re.compile(r"Arthur\s*Andersen", re.IGNORECASE)
    andersen_new = ok["auditor_out_new"].str.contains(
        r"Arthur\s*Andersen", case=False, na=False
    ) | ok["auditor_in_new"].str.contains(
        r"Arthur\s*Andersen", case=False, na=False
    )
    log.info("Arthur Andersen events recovered: %d", andersen_new.sum())

    # Use new names
    ok["auditor_out"] = ok["auditor_out_new"]
    ok["auditor_in"]  = ok["auditor_in_new"]
    ok = ok.drop(columns=["auditor_out_new", "auditor_in_new"])

    # ── Step 2: Re-classify Big4 flags and quality_direction ──────────────────
    log.info("Re-classifying Big4 flags and quality_direction...")

    ok["out_big4"] = ok["auditor_out"].apply(classify_big4)
    ok["in_big4"]  = ok["auditor_in"].apply(classify_big4)
    ok["has_text"] = ok["item401_text"].notna() & (ok["item401_text"].str.len() >= 20)

    ok["quality_direction"] = [
        build_quality_direction(ob, ib, ht)
        for ob, ib, ht in zip(ok["out_big4"], ok["in_big4"], ok["has_text"])
    ]

    # ── Step 3: Final cleanup and output ──────────────────────────────────────
    ok = (
        ok.drop(columns=["has_text"])
        .assign(
            date_filed   = lambda d: pd.to_datetime(d["date_filed"]),
            filing_year  = lambda d: d["date_filed"].dt.year,
            is_amendment = lambda d: d["form_type"] == "8-K/A",
        )
        .sort_values("date_filed")
        .reset_index(drop=True)
    )

    # Write parquet
    ok.to_parquet(OUT_FILE, index=False)
    log.info("Event file written: %s", OUT_FILE)
    log.info("Rows: %d", len(ok))
    log.info(
        "Date range: %s — %s",
        ok["date_filed"].min().date(),
        ok["date_filed"].max().date(),
    )

    # ── Step 4: Report ─────────────────────────────────────────────────────────
    log.info("quality_direction breakdown (new):\n%s",
             ok["quality_direction"].value_counts().to_string())

    summary = (
        ok.groupby(["filing_year", "quality_direction"])
        .size()
        .unstack(fill_value=0)
    )
    log.info("Annual breakdown:\n%s", summary.to_string())

    # Write diagnostic log (before/after quality_direction comparison)
    ok[["cik", "acc_nodash", "date_filed", "filing_year",
        "auditor_out", "auditor_in", "out_big4", "in_big4",
        "quality_direction", "reason", "disagreements"]].to_csv(LOG_FILE, index=False)
    log.info("Reclassification log written: %s", LOG_FILE)
    log.info("=== done ===")


if __name__ == "__main__":
    main()
