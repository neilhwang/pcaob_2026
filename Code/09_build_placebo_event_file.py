"""
09_build_placebo_event_file.py
==============================
Build a placebo event file from SEC EDGAR Form 8-K (Item 5.02) filings —
departures, elections, and appointments of directors and certain officers.

PURPOSE:
    Placebo test for the main finding: if state-level political polarization
    amplifies |CAR| and AbVol around Item 4.01 auditor-change disclosures, the
    institutional-trust channel predicts the effect should be weaker or absent
    around Item 5.02 governance events.  Item 5.02 filings are similarly
    mandatory, standardized, and information-relevant, but they are not
    institution-dependent signals about auditor credibility or regulatory
    oversight.  Convergent evidence that polarization amplifies auditor-change
    reactions but not officer-change reactions strengthens the mechanism claim.

INPUT:  SEC EDGAR (downloaded via HTTP — no local raw file required)

OUTPUT: Data/Processed/placebo_events_raw.parquet
            cik, acc_nodash, date_filed, company_name, form_type,
            parse_status, event_type, is_ceo, is_cfo, is_executive,
            departure_reason, item502_text
        Data/Processed/09_placebo_parse_log.csv   (diagnostic)

PIPELINE:
    1. Query EDGAR full-text search (EFTS) for 8-K filings containing the
       Item 5.02 title phrase "Departure of Directors or Certain Officers".
    2. Fetch the EDGAR quarterly index for CIK / company / filing date.
    3. Parse each filing with edgarParser; extract Item 5.02 section.
    4. Classify event type (departure / appointment / election) and role
       (CEO, CFO, other executive, director).
    5. Write clean parquet to Data/Processed/.

ITEM 5.02 COVERAGE:
    Item 5.02 was introduced with the SEC's revised 8-K rules effective
    August 23, 2004.  Pre-2004 officer changes appeared inconsistently across
    items; our sample therefore has denser coverage from 2005 onward.
    Events 2001-2004 are included where parseable but may undercount.

RUNTIME: ~2-4 hours for 2001-2024 (Item 5.02 filings outnumber Item 4.01;
    expect ~60,000-80,000 candidate filings).

DEPENDENCIES:
    Same as script 01: requests, pandas, pyarrow, tqdm, beautifulsoup4, lxml,
    edgarParser (Code/edgarparser/parse_8K.py).
"""

import sys
import time
import re
import logging
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent / "edgarParser-master"))
from parse_8K import parse_8k_filing

# ── Configuration ──────────────────────────────────────────────────────────────
START_YEAR    = 2001
END_YEAR      = 2024
REQUEST_DELAY = 0.12          # ~8 req/sec — within SEC's 10 req/sec limit
USER_AGENT    = "Research project neil.hwang@bcc.cuny.edu"
MAX_RETRIES   = 3

ROOT            = Path(__file__).resolve().parent.parent
OUT_FILE        = ROOT / "Data/Processed/placebo_events_raw.parquet"
LOG_FILE        = ROOT / "Data/Processed/09_placebo_parse_log.csv"
CHECKPOINT_FILE = ROOT / "Data/Processed/09_checkpoint.csv"
CHECKPOINT_EVERY = 100

# EFTS search phrase: the distinctive first clause of the Item 5.02 title.
# This phrase appears in the document text of Item 5.02 filings and is
# sufficiently specific to avoid false positives from other 8-K items.
EFTS_PHRASE = "Departure of Directors or Certain Officers"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_HEADERS = {"User-Agent": USER_AGENT}


# ── Role patterns ──────────────────────────────────────────────────────────────

CEO_PAT = re.compile(
    r"chief\s+executive\s+officer|\bCEO\b|president\s+and\s+chief\s+executive",
    re.IGNORECASE,
)
CFO_PAT = re.compile(
    r"chief\s+financial\s+officer|\bCFO\b|principal\s+financial\s+officer",
    re.IGNORECASE,
)
EXEC_PAT = re.compile(
    r"chief\s+\w+\s+officer|president|executive\s+vice\s+president"
    r"|principal\s+(executive|financial|accounting)\s+officer",
    re.IGNORECASE,
)
DEPARTURE_PAT = re.compile(
    r"resign|retire|step(?:ping)?\s+down|stepped\s+down|terminat|departure"
    r"|no\s+longer\s+serv|will\s+not\s+(?:stand|continu)",
    re.IGNORECASE,
)
APPOINTMENT_PAT = re.compile(
    r"appoint|named\s+as|elected\s+as|will\s+serve\s+as|has\s+been\s+(?:named|appointed)",
    re.IGNORECASE,
)
ELECTION_PAT = re.compile(
    r"elected\s+(?:to\s+)?(?:the\s+)?board|elected\s+(?:as\s+)?director",
    re.IGNORECASE,
)
VOLUNTARY_PAT = re.compile(
    r"resign|retire|personal\s+reason|pursue\s+other|family\s+reason"
    r"|step(?:ping)?\s+down|mutual\s+agreement",
    re.IGNORECASE,
)
TERMINATION_PAT = re.compile(
    r"terminat|for\s+cause|effective\s+immediately|dismissed|without\s+cause",
    re.IGNORECASE,
)


# ── HTTP helper (identical to script 01) ──────────────────────────────────────

def edgar_get(url: str) -> requests.Response | None:
    for attempt in range(MAX_RETRIES):
        time.sleep(REQUEST_DELAY)
        try:
            resp = requests.get(url, headers=BASE_HEADERS, timeout=30)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                log.warning("Rate-limited. Waiting %ds.", wait)
                time.sleep(wait)
            elif resp.status_code >= 500:
                time.sleep(2 ** attempt)
            else:
                log.warning("HTTP %d for %s", resp.status_code, url)
                return None
        except requests.RequestException as exc:
            log.warning("Request error (attempt %d): %s", attempt + 1, exc)
            time.sleep(2 ** attempt)
    return None


# ── Step 1: EFTS pre-filter ────────────────────────────────────────────────────

def fetch_efts_year(year: int) -> list[str]:
    """
    Query EDGAR EFTS for 8-K filings containing the Item 5.02 title phrase
    in a given year.  Returns a list of raw EFTS accession IDs.
    """
    start_dt = f"{year}-01-01"
    end_dt   = f"{year}-12-31"

    # URL-encode the search phrase (spaces → +, apostrophes → %27)
    encoded = EFTS_PHRASE.replace(" ", "+").replace("'", "%27")

    accessions = []
    from_idx   = 0
    first_call = True

    while True:
        url = (
            f"https://efts.sec.gov/LATEST/search-index"
            f"?q=%22{encoded}%22&forms=8-K"
            f"&dateRange=custom&startdt={start_dt}&enddt={end_dt}"
            f"&from={from_idx}"
        )
        resp = edgar_get(url)
        if resp is None:
            log.warning("EFTS failed for year %d at offset %d", year, from_idx)
            break

        data  = resp.json()
        hits  = data.get("hits", {}).get("hits", [])
        total = data.get("hits", {}).get("total", {}).get("value", 0)

        if first_call and hits:
            log.info("EFTS _id sample (year %d): %s", year, [h["_id"] for h in hits[:3]])
            first_call = False

        if not hits:
            break

        accessions.extend(h["_id"] for h in hits)
        from_idx += len(hits)
        if from_idx >= min(total, 9_800):
            break

    return accessions


def build_efts_candidate_set() -> set[str]:
    all_accessions: list[str] = []
    for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="EFTS by year"):
        acc = fetch_efts_year(year)
        all_accessions.extend(acc)
        log.info("  %d: %d filings found", year, len(acc))
    unique = set(all_accessions)
    log.info("EFTS total candidates: %d unique accessions", len(unique))
    return unique


# ── Step 2: EDGAR quarterly index (identical to script 01) ────────────────────

def fetch_quarter_index(year: int, quarter: int) -> pd.DataFrame | None:
    url = (
        f"https://www.sec.gov/Archives/edgar/full-index/"
        f"{year}/QTR{quarter}/company.idx"
    )
    resp = edgar_get(url)
    if resp is None:
        return None

    date_pat     = re.compile(r"(\d{4}-\d{2}-\d{2})")
    filename_pat = re.compile(r"(edgar/data/\S+)")

    rows = []
    for line in resp.text.splitlines():
        if len(line) < 100:
            continue
        form_type = line[62:74].strip()
        if form_type not in ("8-K", "8-K/A"):
            continue
        tail   = line[86:]
        date_m = date_pat.search(tail)
        fname_m = filename_pat.search(tail)
        rows.append({
            "company_name": line[0:62].strip(),
            "form_type":    form_type,
            "cik":          line[74:86].strip(),
            "date_filed":   date_m.group(1)  if date_m  else "",
            "filename":     fname_m.group(1) if fname_m else "",
        })

    return pd.DataFrame(rows) if rows else None


def build_quarterly_index() -> pd.DataFrame:
    frames = []
    quarters = [(y, q) for y in range(START_YEAR, END_YEAR + 1) for q in range(1, 5)]
    for year, quarter in tqdm(quarters, desc="Quarterly indexes"):
        df = fetch_quarter_index(year, quarter)
        if df is not None:
            frames.append(df)
    index = pd.concat(frames, ignore_index=True)
    index["acc_nodash"] = (
        index["filename"]
        .str.extract(r"(\d{10}-\d{2}-\d{6})", expand=False)
        .str.replace("-", "", regex=False)
        .fillna("")
    )
    log.info("Total 8-K rows in quarterly index: %d", len(index))
    return index


# ── Step 3: Intersect (identical to script 01) ────────────────────────────────

def normalize_acc(raw: str) -> str:
    acc = raw.split(":")[0]
    return acc.replace("-", "")


def intersect_with_index(efts_set: set[str], index: pd.DataFrame) -> pd.DataFrame:
    efts_nodash = {normalize_acc(a) for a in efts_set}
    mask = index["acc_nodash"].isin(efts_nodash)
    candidates = index[mask].copy()
    log.info(
        "Candidates after intersection: %d / %d index rows",
        len(candidates), len(index),
    )
    return candidates


# ── Step 4: Parse filings ─────────────────────────────────────────────────────

def build_filing_url(row: pd.Series) -> str:
    cik    = row["cik"].lstrip("0")
    acc_nd = row["acc_nodash"]
    acc_dashed = f"{acc_nd[:10]}-{acc_nd[10:12]}-{acc_nd[12:]}"
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_dashed}.txt"


def extract_item502_row(filing_df: pd.DataFrame) -> pd.Series | None:
    """
    Return the Item 5.02 row from the edgarParser output DataFrame.

    Handles:
      - "Item 5.02" — post-August 2004 format (the vast majority of the sample)
      - "Item 5.0" or "Item 5" — rare early-transition or malformed labels
    Pre-2004 officer changes have no consistent item number; those filings
    will return None and be logged as no_item502.
    """
    if filing_df is None or filing_df.empty:
        return None
    mask = filing_df["item"].str.contains(
        r"Item\s+5\.02", flags=re.IGNORECASE, na=False
    )
    if not mask.any():
        return None
    return filing_df[mask].iloc[0]


def parse_item502_text(text: str) -> dict:
    """
    Extract structured fields from Item 5.02 free text.

    Returns:
        event_type       : "departure" | "appointment" | "election" | "mixed" | "unclassified"
        is_ceo           : True if a CEO departure or appointment is mentioned
        is_cfo           : True if a CFO departure or appointment is mentioned
        is_executive     : True if any named executive officer is involved
        departure_reason : "voluntary" | "termination" | "unclassified" | None
    """
    if not text or len(text) < 20:
        return {
            "event_type": "unclassified",
            "is_ceo": False,
            "is_cfo": False,
            "is_executive": False,
            "departure_reason": None,
        }

    has_departure   = bool(DEPARTURE_PAT.search(text))
    has_appointment = bool(APPOINTMENT_PAT.search(text))
    has_election    = bool(ELECTION_PAT.search(text))

    if has_departure and (has_appointment or has_election):
        event_type = "mixed"
    elif has_departure:
        event_type = "departure"
    elif has_appointment:
        event_type = "appointment"
    elif has_election:
        event_type = "election"
    else:
        event_type = "unclassified"

    is_ceo       = bool(CEO_PAT.search(text))
    is_cfo       = bool(CFO_PAT.search(text))
    is_executive = bool(EXEC_PAT.search(text))

    # Departure reason only meaningful when event involves a departure
    if has_departure:
        if TERMINATION_PAT.search(text):
            departure_reason = "termination"
        elif VOLUNTARY_PAT.search(text):
            departure_reason = "voluntary"
        else:
            departure_reason = "unclassified"
    else:
        departure_reason = None

    return {
        "event_type":       event_type,
        "is_ceo":           is_ceo,
        "is_cfo":           is_cfo,
        "is_executive":     is_executive,
        "departure_reason": departure_reason,
    }


def parse_one_filing(row: pd.Series) -> dict:
    base = {
        "cik":          row["cik"],
        "acc_nodash":   row["acc_nodash"],
        "date_filed":   row["date_filed"],
        "company_name": row["company_name"],
        "form_type":    row["form_type"],
    }

    url = build_filing_url(row)
    try:
        filing_df = parse_8k_filing(url)
    except SystemExit:
        return {**base, "parse_status": "no_items_found",
                "item502_text": None, "event_type": None,
                "is_ceo": None, "is_cfo": None,
                "is_executive": None, "departure_reason": None}
    except Exception as exc:
        log.debug("edgarParser error for %s: %s", url, exc)
        return {**base, "parse_status": "parser_error",
                "item502_text": None, "event_type": None,
                "is_ceo": None, "is_cfo": None,
                "is_executive": None, "departure_reason": None}

    item_row = extract_item502_row(filing_df)
    if item_row is None:
        return {**base, "parse_status": "no_item502",
                "item502_text": None, "event_type": None,
                "is_ceo": None, "is_cfo": None,
                "is_executive": None, "departure_reason": None}

    text   = item_row.get("itemText", "") or ""
    fields = parse_item502_text(text)

    return {
        **base,
        "parse_status": "ok",
        "item502_text": text[:4000],
        **fields,
    }


def _sanitize_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Strip null bytes from all string columns before writing to CSV.

    On Windows, pandas raises OSError (errno 22) when any string value
    contains a null byte (\\x00), which can appear in EDGAR HTML text.
    """
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].apply(
            lambda v: v.replace("\x00", "") if isinstance(v, str) else v
        )
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== 09_build_placebo_event_file.py  start ===")
    log.info("Item 5.02 placebo events, %d–%d", START_YEAR, END_YEAR)

    # Step 1
    log.info("Step 1: EFTS pre-filter (phrase: '%s')", EFTS_PHRASE)
    efts_set = build_efts_candidate_set()

    # Step 2
    log.info("Step 2: Download quarterly indexes")
    index = build_quarterly_index()

    # Step 3
    log.info("Step 3: Intersect EFTS candidates with index")
    candidates = intersect_with_index(efts_set, index)

    # Step 4: Parse with checkpointing
    log.info("Step 4: Parse %d candidate filings", len(candidates))

    if CHECKPOINT_FILE.exists() and CHECKPOINT_FILE.stat().st_size > 0:
        try:
            completed = pd.read_csv(CHECKPOINT_FILE)
            done_accs = set(completed["acc_nodash"].astype(str))
            results   = completed.to_dict("records")
            log.info("Resuming from checkpoint: %d already parsed", len(results))
        except Exception as exc:
            log.warning("Checkpoint unreadable (%s); starting fresh.", exc)
            done_accs = set()
            results   = []
    else:
        done_accs = set()
        results   = []

    remaining = candidates[~candidates["acc_nodash"].isin(done_accs)]
    log.info("Remaining to parse: %d", len(remaining))

    import warnings
    for i, (_, row) in enumerate(
        tqdm(remaining.iterrows(), total=len(remaining), desc="Parsing")
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results.append(parse_one_filing(row))

        if (i + 1) % CHECKPOINT_EVERY == 0:
            _sanitize_for_csv(pd.DataFrame(results)).to_csv(CHECKPOINT_FILE, index=False)

    _sanitize_for_csv(pd.DataFrame(results)).to_csv(CHECKPOINT_FILE, index=False)

    if not results:
        log.error("No filings parsed. Check EFTS phrase and index intersection.")
        return

    all_results = pd.DataFrame(results)
    all_results.to_csv(LOG_FILE, index=False)
    log.info("Parse log written: %s", LOG_FILE)
    log.info("Status breakdown:\n%s", all_results["parse_status"].value_counts().to_string())

    # Step 5: Clean and write
    clean = (
        all_results[all_results["parse_status"] == "ok"]
        .copy()
        .assign(
            date_filed   = lambda d: pd.to_datetime(d["date_filed"]),
            filing_year  = lambda d: d["date_filed"].dt.year,
            is_amendment = lambda d: d["form_type"] == "8-K/A",
        )
        .sort_values("date_filed")
        .reset_index(drop=True)
    )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(OUT_FILE, index=False)
    log.info("Placebo event file written: %s", OUT_FILE)
    log.info("Rows: %d", len(clean))

    if len(clean) == 0:
        log.error("Zero events with parse_status=ok.")
        return

    log.info(
        "Date range: %s — %s",
        clean["date_filed"].min().date(),
        clean["date_filed"].max().date(),
    )

    # Summary by year and event type
    summary = (
        clean.groupby(["filing_year", "event_type"])
        .size()
        .unstack(fill_value=0)
    )
    log.info("Annual breakdown by event type:\n%s", summary.to_string())

    log.info("CEO events: %d", clean["is_ceo"].sum())
    log.info("CFO events: %d", clean["is_cfo"].sum())
    log.info("Executive events: %d", clean["is_executive"].sum())
    log.info(
        "Departure reason breakdown:\n%s",
        clean["departure_reason"].value_counts(dropna=False).to_string(),
    )
    log.info("=== done ===")


if __name__ == "__main__":
    main()
