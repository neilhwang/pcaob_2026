"""
19_build_filing_specificity.py
================================
Build a Filing Specificity Index and a Boilerplate-Adjusted Length measure
for each Item 4.01 auditor-change filing in the analysis sample.

PURPOSE
-------
Replace the high_ambiguity indicator (which conflates interpretive latitude
with economic materiality) with a cleaner within-setting measure of filing
informativeness. The specificity index targets the construct the theory
needs: how much interpretive guidance the filing gives investors, measured
inside the filing text itself and orthogonal to the economic nature of the
change.

MAIN MEASURE: Filing Specificity Index (0-7 binary components, summed)
  1. explicit_cause       — filing states a specific cause/reason for change
  2. concrete_issue       — names a concrete accounting/audit topic
  3. disagreement_domain  — identifies a specific area of disagreement
  4. reportable_event     — describes a reportable event (Item 304(a)(1)(v))
  5. committee_process    — specific audit committee process detail (beyond
                             generic mention)
  6. linked_transaction   — links change to another reporting event/transaction
  7. nongeneric_language  — non-boilerplate causal language present

SECONDARY MEASURE: Boilerplate-adjusted nonstandard text length
  - Strip standard Item 4.01 compliance language from each filing
  - Compute remaining word count (log + 1)
  - Residualize on obvious materiality correlates (disagreement,
     dismissal, Big4 involvement, firm size, year)

INPUTS
------
  Data/Processed/auditor_changes_raw.parquet   — item401_text for each event
  Data/Processed/analysis_sample.parquet       — 678 final sample events

OUTPUTS
-------
  Data/Processed/filing_specificity.parquet
     acc_nodash, specificity_index, low_specificity,
     raw_words, boilerplate_words, nonstandard_words, log_nonstd_words,
     plus the 7 binary component indicators

NOTES
-----
- Text is already parsed by 01_build_edgar_event_file.py (item401_text field).
- Some filings have very short text (parser extracted only the section
  title); these score low on the index, consistent with their being
  uninformative. This is noted in the paper as a caveat.
- Keyword patterns are case-insensitive and allow for whitespace/punctuation
  variation. They are designed to be specific to the Item 4.01 domain.
"""

import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
PROC     = ROOT / "Data" / "Processed"
RAW_FILE = PROC / "auditor_changes_raw.parquet"
SAMPLE_FILE = PROC / "analysis_sample.parquet"
OUT_FILE = PROC / "filing_specificity.parquet"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── Boilerplate stripping (sentence-based, no backtracking) ────────────────
# The Reg S-K Item 304 disclaimer has a standard structure. We split the
# filing into sentences and drop sentences matching boilerplate patterns.

# Short, non-greedy keyword triggers that identify boilerplate sentences.
# A sentence is dropped if it contains ALL of the required triggers in
# any one group. Each group is a list of lowercase substrings.

BOILERPLATE_TRIGGERS = [
    # "...did not contain an adverse opinion..."
    ["adverse opinion"],
    ["disclaimer of opinion"],
    ["qualified", "uncertainty"],
    ["modified", "uncertainty"],
    # "There were no disagreements..."
    ["no disagreement"],
    ["there were no disagreement"],
    ["there have been no disagreement"],
    ["not had", "disagreement"],
    # "No reportable events"
    ["no reportable event"],
    ["there were no reportable"],
    ["no such reportable"],
    ["no reportable condition"],
    # Consultation disclaimer (Item 304(a)(2))
    ["did not consult", "application of accounting"],
    ["application of accounting principles"],
    ["type of audit opinion"],
    ["factor considered important"],
    ["accounting, auditing or financial reporting issue"],
    # Copy / concurrence letter boilerplate
    ["copy of this disclosure"],
    ["letter addressed to the commission"],
    ["letter from", "former"],
    ["exhibit 16"],
    ["concurrence", "former"],
    # "During the two most recent fiscal years" lead-in
    ["two most recent fiscal years", "no"],
    ["subsequent interim period"],
    # "...did not contain..." style (generic)
    ["did not contain an"],
    ["were not qualified"],
    ["were not modified"],
    # Reg S-K Item 304 reference
    ["item 304"],
    ["regulation s-k"],
]


def split_sentences(text):
    """Simple sentence splitter: split on periods, question marks, newlines."""
    # Normalize whitespace
    t = re.sub(r"\s+", " ", text)
    # Split on sentence boundaries
    parts = re.split(r"[.?!]\s+|\n+|;\s+", t)
    return [p.strip() for p in parts if p.strip()]


def is_boilerplate_sentence(sent):
    """Return True if the sentence matches any boilerplate trigger group."""
    s = sent.lower()
    for group in BOILERPLATE_TRIGGERS:
        if all(trigger in s for trigger in group):
            return True
    return False


def strip_boilerplate(text):
    """Drop sentences matching boilerplate patterns; return the remainder."""
    if not text:
        return ""
    sents = split_sentences(text.lower())
    kept = [s for s in sents if not is_boilerplate_sentence(s)]
    return " ".join(kept)


# ── Indicator 1: Explicit cause of the change ──────────────────────────────

CAUSE_PATTERNS = [
    r"\bdue to\b",
    r"\bbecause of\b",
    r"\bas a result of\b",
    r"\barising from\b",
    r"\bresulted from\b",
    r"\bin order to\b",
    r"\bthe reason for\b",
    r"\bthe reasons? for the (change|dismissal|resignation|engagement)\b",
    r"\bdecided to (dismiss|change|engage|retain|appoint)\b",
    r"\bdecision to (dismiss|change|engage|retain|appoint)\b",
]


def has_explicit_cause(text):
    if not text or len(text) < 50:
        return 0
    t = strip_boilerplate(text)
    return int(any(re.search(p, t) for p in CAUSE_PATTERNS))


# ── Indicator 2: Concrete accounting/audit issue named ─────────────────────

CONCRETE_ISSUES = [
    r"\brevenue recognition\b",
    r"\bmaterial weakness(es)?\b",
    r"\bsignificant deficienc(y|ies)\b",
    r"\brestatement\b",
    r"\brestate(d)?\b",
    r"\b(goodwill|asset) impairment\b",
    r"\bgoing concern\b",
    r"\bdeferred tax\b",
    r"\binventory (valuation|reserve)\b",
    r"\b(accrual|accruals) (method|accounting)\b",
    r"\blease accounting\b",
    r"\bfair value\b",
    r"\bstock[- ]based compensation\b",
    r"\brevenue (cut[- ]?off|timing)\b",
    r"\b(change in )?accounting principle(s)?\b",
    r"\bscope (limitation|of the audit)\b",
]

# Strip boilerplate negations (same as reportable_event but for concrete issues)
NEG_CONCRETE = [
    r"\bno (such )?material weakness(es)?\b",
    r"\bno going concern (opinion|modification|qualification)\b",
    r"\bno (such )?scope limitation",
    r"\bnot qualified as to.{0,30}(accounting principles|going concern|scope)",
    r"\bno (such )?restatement",
]


def has_concrete_issue(text):
    if not text or len(text) < 50:
        return 0
    t = strip_boilerplate(text)
    return int(any(re.search(p, t) for p in CONCRETE_ISSUES))


# ── Indicator 3: Disagreement domain identified ────────────────────────────

DISAGREEMENT_PATTERNS = [
    r"\bdisagreement(s)? (on|with|over|regarding|about|concerning)\b",
    r"\bdifferences? of opinion\b",
    r"\bdisagree(d|s|ment) with\b",
    r"\b(the|a) disagreement (between|with)\b",
    r"\bdiffered with (respect to|on)\b",
    # The negation of disagreement is standard boilerplate; we require
    # an affirmative statement
    r"\b(had|were|has been|have had) (a )?disagreement\b",
]

# Phrases that negate disagreement (standard boilerplate — do NOT count)
NEG_DISAGREEMENT = [
    r"\bno disagreement(s)?\b",
    r"\bthere (were|was|have been|has been) no disagreement",
    r"\bnot had (any )?disagreement",
    r"\bwithout any disagreement",
]


def has_disagreement_domain(text):
    if not text or len(text) < 50:
        return 0
    t = strip_boilerplate(text)
    # Also strip any residual negation phrases
    for neg in NEG_DISAGREEMENT:
        t = re.sub(neg, "", t)
    return int(any(re.search(p, t) for p in DISAGREEMENT_PATTERNS))


# ── Indicator 4: Reportable event described ────────────────────────────────
# NOTE: Item 304 requires every filing to state whether there were
# reportable events, so the phrase "reportable event" appears in virtually
# every filing as part of the standard negative disclaimer. We must strip
# the negation phrases before looking for affirmative statements.

REPORTABLE_PATTERNS = [
    r"\breportable event(s)?\b",
    r"\breportable condition(s)?\b",
    r"\bmaterial weakness(es)?\b",
    r"\binternal control (deficiency|weakness|failure)\b",
    r"\bgoing concern\b",
    r"\bscope limitation\b",
    r"\binability to (rely|audit)\b",
    r"\breliability of (the )?(company|registrant|firm).{0,20}(audit|financial)\b",
]

# Standard boilerplate negations that must be stripped
NEG_REPORTABLE = [
    r"\bno (such )?reportable event(s)?\b",
    r"\bthere (were|was|have been|has been) no (such )?reportable event",
    r"\bnot (had|occurred|existed) (any )?reportable event",
    r"\bwithout (any )?reportable event",
    r"\bno (such )?reportable condition(s)?\b",
    r"\bno (such )?material weakness(es)?\b",
    r"\bno going concern (opinion|modification|qualification)\b",
    r"\bno (such )?scope limitation",
    r"\b(nor|not) (a |any |were )?(there )?(were |was )?(reportable event|reportable condition|material weakness|scope limitation)",
    r"\babsence of (any )?reportable event",
    r"\babsence of (any )?material weakness",
]


def has_reportable_event(text):
    if not text or len(text) < 50:
        return 0
    t = strip_boilerplate(text)
    for neg in NEG_REPORTABLE:
        t = re.sub(neg, "", t)
    return int(any(re.search(p, t) for p in REPORTABLE_PATTERNS))


# ── Indicator 5: Specific audit committee process detail ───────────────────
# Generic mentions of "audit committee" are standard boilerplate; we look
# for process-specific language (proposals, evaluation, competitive bidding,
# vote details).

COMMITTEE_PROCESS = [
    r"\brequested proposals\b",
    r"\brequest for proposal\b",
    r"\bcompetitive (bid|process|proposal)\b",
    r"\bwritten proposals\b",
    r"\boral presentations\b",
    r"\bfirms (were |that )?(evaluated|considered|reviewed)\b",
    r"\baudit committee (reviewed|evaluated|considered) (proposals|candidates|firms)\b",
    r"\bfollowing (a|an|its) (evaluation|review|consideration) of\b",
    r"\bafter (careful|extensive|thorough) (review|consideration|evaluation|analysis)\b",
    r"\bunanimously (recommended|approved|voted)\b",
    r"\bselection process\b",
    r"\bdecision of the audit committee\b",
]


def has_committee_process(text):
    if not text or len(text) < 50:
        return 0
    t = strip_boilerplate(text)
    return int(any(re.search(p, t) for p in COMMITTEE_PROCESS))


# ── Indicator 6: Linked to another transaction/reporting event ─────────────

LINKED_TRANSACTION = [
    r"\bmerger\b",
    r"\bacquisition\b",
    r"\bbeen acquired\b",
    r"\bspin[- ]?off\b",
    r"\bspun off\b",
    r"\bgoing private\b",
    r"\btake[- ]?private\b",
    r"\binitial public offering\b",
    r"\bipo\b",
    r"\brestructur(ing|ed|e)\b",
    r"\bbankruptc(y|ies)\b",
    r"\bchapter 11\b",
    r"\breorganization\b",
    r"\bchange in control\b",
    r"\brestatement of\b",
    r"\bform 10-?k/a\b",
    r"\bform 10-?q/a\b",
    r"\bamended (annual|quarterly) report\b",
]


def has_linked_transaction(text):
    if not text or len(text) < 50:
        return 0
    t = strip_boilerplate(text)
    return int(any(re.search(p, t) for p in LINKED_TRANSACTION))


# ── Indicator 7: Non-generic causal language ───────────────────────────────
# Any of: fee/cost motivation, service quality, firm size/fit, geographic,
# complexity — specific business reasons beyond statutory phrasing.

NONGENERIC_PATTERNS = [
    r"\baudit fee(s)?\b",
    r"\bfee (considerations?|negotiation|dispute|disagreement)\b",
    r"\bhigher (audit )?fee",
    r"\blower (audit )?fee",
    r"\breduce(d)? (audit )?fee",
    r"\bincrease(d)? (in )?fee",
    r"\bcost(s)? (of|savings|effectiveness|considerations?)\b",
    r"\bservice (quality|level|concerns?|capabilities)\b",
    r"\b(firm|auditor) size\b",
    r"\bsize of (the )?(firm|auditor|company)\b",
    r"\b(local|regional|national|global) (presence|office|capabilit)",
    r"\bbetter (fit|aligned|suited|matched|able)\b",
    r"\b(increased|growing|changing|expanding) (complexity|needs|requirements|operations)\b",
    r"\binternational (presence|operations|expansion)\b",
    r"\bexpertise in\b",
    r"\bindustry (expertise|experience|knowledge|specialization)\b",
    r"\bindependence (concerns?|issues?|requirements?|considerations?)\b",
    r"\brotation (requirement|policy)\b",
    r"\bpartner rotation\b",
    r"\blead (audit )?partner\b",
    r"\bnon[- ]?audit (services|fees)\b",
    r"\bsarbanes.{0,5}oxley\b",
    r"\bconsolidation of (firms|auditors|accounting)\b",
    r"\bclient (size|fit|portfolio)\b",
    r"\bstrategic (decision|review|considerations?)\b",
    r"\bglobal (reach|capabilities|network)\b",
    r"\bconflict of interest\b",
]


def has_nongeneric_language(text):
    if not text or len(text) < 50:
        return 0
    t = strip_boilerplate(text)
    return int(any(re.search(p, t) for p in NONGENERIC_PATTERNS))


# ── Boilerplate-adjusted length ────────────────────────────────────────────

# Standard Item 4.01 compliance phrases that appear in virtually every
# filing. We strip these before counting remaining words to isolate
# nonstandard content.

BOILERPLATE_PHRASES = [
    r"changes? in registrant'?s? certifying accountant",
    r"item 4\.01",
    r"(no|not)( contain)? (an )?adverse opinion",
    r"nor (a |any )?disclaimer of opinion",
    r"not (qualified|modified) as to uncertainty",
    r"audit scope or accounting principles?",
    r"(during the )?(two |2 |most recent )?fiscal years?",
    r"subsequent interim period",
    r"preceding (the )?(date of )?(dismissal|resignation|engagement)",
    r"no disagreement(s)?( with)?",
    r"(the |no )reportable events?",
    r"(registrant|company) has (authorized|requested|provided)",
    r"(a copy of|as exhibit)",
    r"letter (addressed to the commission|from the former auditor)",
    r"(concurrence|agreement|statement) (with|of) the (former |departing )?(auditor|accountant)",
    r"(audit committee|board of directors) (of the )?(company|registrant|board)",
    r"(approved|recommended|endorsed) (the (dismissal|engagement|appointment))",
    r"independent (registered public accounting firm|auditors?|accountants?)",
    r"for the (year|fiscal year) end(ed|ing)",
    r"upon the recommendation",
]


def count_nonstandard_words(text):
    """
    Strip the standard Item 304 disclaimer and residual phrases, then count
    remaining words. Returns (raw_words, nonstandard_words, boilerplate_frac).
    """
    if not text:
        return 0, 0, 0.0
    raw = len(re.findall(r"\b\w+\b", text))
    # Strip sentence-level disclaimer via strip_boilerplate, then also strip
    # shorter residual phrases
    stripped = strip_boilerplate(text)
    for phrase in BOILERPLATE_PHRASES:
        stripped = re.sub(phrase, " ", stripped)
    nonstd = len(re.findall(r"\b\w+\b", stripped))
    boil_frac = 1.0 - (nonstd / raw) if raw > 0 else 0.0
    return raw, nonstd, boil_frac


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    log.info("=== 19_build_filing_specificity.py ===")

    # Load
    raw = pd.read_parquet(RAW_FILE)
    sample = pd.read_parquet(SAMPLE_FILE, columns=["acc_nodash"])

    # Restrict to sample events
    df = raw[raw["acc_nodash"].isin(sample["acc_nodash"])].copy()
    log.info("Sample events: %d", len(df))

    # Clean text: ensure strings, replace NaN with empty
    df["item401_text"] = df["item401_text"].fillna("").astype(str)

    # ── Compute the 7 binary indicators ────────────────────────────────────
    log.info("Computing specificity index components...")
    df["explicit_cause"]     = df["item401_text"].apply(has_explicit_cause)
    df["concrete_issue"]     = df["item401_text"].apply(has_concrete_issue)
    df["disagreement_domain"]= df["item401_text"].apply(has_disagreement_domain)
    df["reportable_event"]   = df["item401_text"].apply(has_reportable_event)
    df["committee_process"]  = df["item401_text"].apply(has_committee_process)
    df["linked_transaction"] = df["item401_text"].apply(has_linked_transaction)
    df["nongeneric_language"]= df["item401_text"].apply(has_nongeneric_language)

    components = ["explicit_cause", "concrete_issue", "disagreement_domain",
                  "reportable_event", "committee_process", "linked_transaction",
                  "nongeneric_language"]
    df["specificity_index"] = df[components].sum(axis=1)

    log.info("Specificity index distribution:")
    log.info("\n%s", df["specificity_index"].value_counts().sort_index().to_string())
    log.info("Mean specificity: %.2f  Median: %.0f",
             df["specificity_index"].mean(), df["specificity_index"].median())

    # Low specificity indicator: bottom tercile of the index
    tercile_cut = df["specificity_index"].quantile(1/3)
    df["low_specificity"] = (df["specificity_index"] <= tercile_cut).astype(int)
    log.info("Low-specificity threshold (<=): %.1f, N_low=%d",
             tercile_cut, df["low_specificity"].sum())

    # Component coverage
    log.info("Component prevalence (fraction of sample):")
    for c in components:
        log.info("  %s: %.3f", c, df[c].mean())

    # ── Boilerplate-adjusted length ────────────────────────────────────────
    log.info("Computing boilerplate-adjusted length...")
    word_counts = df["item401_text"].apply(count_nonstandard_words)
    df["raw_words"] = [w[0] for w in word_counts]
    df["nonstandard_words"] = [w[1] for w in word_counts]
    df["boilerplate_frac"] = [w[2] for w in word_counts]
    df["log_nonstd_words"] = np.log1p(df["nonstandard_words"])

    log.info("Raw words:    mean=%.0f  median=%.0f",
             df["raw_words"].mean(), df["raw_words"].median())
    log.info("Nonstd words: mean=%.0f  median=%.0f",
             df["nonstandard_words"].mean(), df["nonstandard_words"].median())
    log.info("Boilerplate fraction: mean=%.2f",
             df["boilerplate_frac"].mean())

    # ── Save ───────────────────────────────────────────────────────────────
    out_cols = ["acc_nodash"] + components + [
        "specificity_index", "low_specificity",
        "raw_words", "nonstandard_words", "boilerplate_frac", "log_nonstd_words"
    ]
    out = df[out_cols].copy()
    out.to_parquet(OUT_FILE, index=False)
    log.info("Saved: %s  (%d rows)", OUT_FILE, len(out))
    log.info("=== done ===")


if __name__ == "__main__":
    main()
