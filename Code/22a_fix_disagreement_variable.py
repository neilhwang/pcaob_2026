"""
22a_fix_disagreement_variable.py
================================
Fix the disagreement variable in auditor_changes_raw.parquet.

The original parser (script 01) flagged disagreements=True whenever the
word "disagreement" appeared anywhere in the Item 4.01 text. This captures
boilerplate language ("there were no disagreements...") that Item 304
requires all filers to include, resulting in ~80% flagged as True.

This script re-classifies using affirmative-mention patterns that
distinguish real disagreement disclosures from negated boilerplate.

INPUTS:
    Data/Processed/auditor_changes_raw.parquet

OUTPUTS:
    Data/Processed/auditor_changes_raw.parquet  (updated in place)
    Archives original to Data/Processed/_Archive/ first
"""

import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "Data/Processed"
RAW_FILE = PROC / "auditor_changes_raw.parquet"
ARCHIVE = PROC / "_Archive"
ARCHIVE.mkdir(parents=True, exist_ok=True)

# ── Archive the original ──────────────────────────────────────────────────────
archive_path = ARCHIVE / "auditor_changes_raw_pre_disagree_fix.parquet"
if not archive_path.exists():
    shutil.copy2(RAW_FILE, archive_path)
    print(f"Archived original to {archive_path}")
else:
    print(f"Archive already exists: {archive_path}")

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_parquet(RAW_FILE)
print(f"Loaded {len(df)} rows")

old_true = (df["disagreements"].astype(str).str.lower() == "true").sum()
print(f"Old disagreements=True: {old_true} ({100*old_true/len(df):.1f}%)")

# ── Re-classify ──────────────────────────────────────────────────────────────
# Affirmative disagreement patterns — language indicating a real disagreement
# was disclosed, not just the boilerplate negation
_yes_disagree = re.compile(
    r"(?:had|have|were)\s+(?:the\s+following\s+)?disagreements?"
    r"|following\s+disagreements?"
    r"|certain\s+disagreements?"
    r"|(?:a|the)\s+disagreement\s+(?:with|between|regarding|concerning|over|about)"
    r"|disagreements?\s+(?:arose|existed|occurred|related|involved|concerned)"
    r"|disagreements?\s+(?:as|described|set\s+forth|regarding)",
    re.IGNORECASE,
)


def classify_disagreement(text):
    """Return True if filing contains affirmative disagreement disclosure."""
    if not isinstance(text, str) or len(text) < 20:
        return None
    return bool(_yes_disagree.search(text))


df["disagreements"] = df["item401_text"].apply(classify_disagreement)

new_true = df["disagreements"].sum()
new_null = df["disagreements"].isna().sum()
print(f"New disagreements=True: {int(new_true)} ({100*new_true/len(df):.1f}%)")
print(f"Null (no text): {new_null}")

# ── Also update the high_ambiguity-relevant fields ────────────────────────────
# The disagreement indicator feeds into the high_ambiguity flag in script 05.
# We don't rebuild high_ambiguity here (that's script 05's job), but we
# ensure the disagreements column is correctly typed for downstream merges.
df["disagreements"] = df["disagreements"].astype("object")  # keep None as None

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_parquet(RAW_FILE, index=False)
print(f"Saved updated {RAW_FILE}")
print("Now re-run script 05 to propagate the fix.")
