"""
01_build_edgar_event_file.py
============================
Build the auditor change event file from SEC EDGAR Form 8-K (Item 4.01) filings.

INPUT:  SEC EDGAR (downloaded via HTTP — no local raw file required)
OUTPUT: Data/Processed/auditor_changes_raw.parquet
        Data/Processed/01_edgar_parse_log.csv   (diagnostic — all attempts)

PIPELINE:
    1. Query EDGAR full-text search (EFTS) API to identify 8-K filings that
       contain "Changes in Registrant's Certifying Accountant" — the exact
       Item 4.01 title (~17,600 filings across 2001–2023, per EFTS count).
    2. Fetch the EDGAR quarterly index to obtain CIK, company name, and filing
       date for each candidate accession number.
    3. Parse each filing with edgarParser (rsljr/edgarParser on GitHub) to
       extract the Item 4.01 section.
    4. Parse the extracted text for: auditor out, auditor in, reason
       (dismissal vs. resignation), and whether disagreements are disclosed.
    5. Write clean parquet to Data/Processed/.

DEPENDENCIES:
    pip install requests pandas pyarrow tqdm beautifulsoup4 lxml
    edgarParser lives locally at Code/edgarparser/parse_8K.py (rsljr/edgarParser)

RUNTIME: ~45–90 minutes for full 2000–2023 sample (rate-limited to 10 req/sec).
Run non-interactively: python 01_build_edgar_event_file.py

NOTES:
    - SEC requires a User-Agent header identifying the requester. Update
      USER_AGENT below before running.
    - EDGAR rate limit is 10 requests/second. REQUEST_DELAY = 0.12s gives
      ~8 req/sec with headroom.
    - 8-K/A amendments are included but flagged; review in script 02 whether
      to drop them.
"""

import sys
import time
import re
import logging
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm

# edgarParser lives locally at Code/edgarparser/parse_8K.py
sys.path.insert(0, str(Path(__file__).resolve().parent / "edgarparser"))
from parse_8K import parse_8k_filing

# ── Configuration ─────────────────────────────────────────────────────────────
START_YEAR    = 2001   # 2000 has 0 hits under correct EFTS query; Item 4.01 disclosures start 2001
END_YEAR      = 2024
REQUEST_DELAY = 0.12          # seconds between requests (~8 req/sec)
USER_AGENT    = "Research project neil.hwang@bcc.cuny.edu"
MAX_RETRIES   = 3

ROOT            = Path(__file__).resolve().parent.parent   # project root
OUT_FILE        = ROOT / "Data/Processed/auditor_changes_raw.parquet"
LOG_FILE        = ROOT / "Data/Processed/01_edgar_parse_log.csv"
CHECKPOINT_FILE = ROOT / "Data/Processed/01_checkpoint.csv"  # resume after interruption
CHECKPOINT_EVERY = 100   # save progress every N filings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_HEADERS = {"User-Agent": USER_AGENT}

# Known audit firm name fragments (for Big 4 / large firm classification).
# Smaller regional firms will parse as auditor_out/auditor_in = None;
# those rows are still useful for CAR/volume tests — only the name fields are missing.
BIG4_PATTERN = re.compile(
    r"Deloitte|PricewaterhouseCoopers|PwC|Ernst\s*&\s*Young|\bEY\b|KPMG",
    re.IGNORECASE,
)
LARGE_FIRM_PATTERN = re.compile(
    r"Grant\s*Thornton|BDO|RSM|McGladrey|Moss\s*Adams|Dixon\s*Hughes"
    r"|Crowe|Eide\s*Bailly|Cherry\s*Bekaert|Marcum|Withum",
    re.IGNORECASE,
)
AUDITOR_PATTERN = re.compile(
    r"Deloitte|PricewaterhouseCoopers|PwC|Ernst\s*&\s*Young|\bEY\b|KPMG"
    r"|Grant\s*Thornton|BDO|RSM|McGladrey|Moss\s*Adams|Dixon\s*Hughes"
    r"|Crowe|Eide\s*Bailly|Cherry\s*Bekaert|Marcum|Withum",
    re.IGNORECASE,
)


# ── HTTP helper ────────────────────────────────────────────────────────────────

def edgar_get(url: str, stream: bool = False) -> requests.Response | None:
    """Polite GET with exponential back-off. Returns None on persistent failure."""
    for attempt in range(MAX_RETRIES):
        time.sleep(REQUEST_DELAY)
        try:
            resp = requests.get(
                url, headers=BASE_HEADERS, timeout=30, stream=stream
            )
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
    Query EDGAR full-text search for 8-K filings containing the exact phrase
    "Changes in Registrant's Certifying Accountant" (Item 4.01 title) in a
    given year. Returns a list of accession numbers (with dashes).

    EFTS caps results at 10,000 per query. Iterating by year keeps each window
    well under that ceiling (max ~3,600 hits in 2002).
    """
    start_dt = f"{year}-01-01"
    end_dt   = f"{year}-12-31"
    accessions = []
    from_idx   = 0

    first_call = True
    while True:
        url = (
            "https://efts.sec.gov/LATEST/search-index"
            # Search for the exact Item 4.01 title text — much more precise than "4.01"
            # which matches financial figures and contract section numbers in exhibits.
            f"?q=%22Changes+in+Registrant%27s+Certifying+Accountant%22&forms=8-K"
            f"&dateRange=custom&startdt={start_dt}&enddt={end_dt}"
            f"&from={from_idx}"  # EFTS returns 100/page max; advance by len(hits) below
        )
        resp = edgar_get(url)
        if resp is None:
            log.warning("EFTS failed for year %d at offset %d", year, from_idx)
            break

        data  = resp.json()
        hits  = data.get("hits", {}).get("hits", [])
        total = data.get("hits", {}).get("total", {}).get("value", 0)

        # Diagnostic: show raw _id format on first successful call
        if first_call and hits:
            log.info("EFTS _id sample (year %d): %s", year, [h["_id"] for h in hits[:3]])
            first_call = False

        if not hits:
            break

        accessions.extend(h["_id"] for h in hits)

        # Advance by actual hits returned (EFTS caps at 100/page regardless of size=).
        from_idx += len(hits)
        if from_idx >= min(total, 9_800):  # EFTS hard cap at 10K
            break

    return accessions


def build_efts_candidate_set() -> set[str]:
    """Run EFTS query for all years and return set of raw accession IDs from EFTS."""
    all_accessions: list[str] = []
    for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="EFTS by year"):
        acc = fetch_efts_year(year)
        all_accessions.extend(acc)
        log.info("  %d: %d filings found", year, len(acc))

    unique = set(all_accessions)
    log.info("EFTS total candidates: %d unique accessions", len(unique))
    # Log a sample so we can verify the _id format
    sample = list(unique)[:5]
    log.info("EFTS _id sample: %s", sample)
    return unique


# ── Step 2: EDGAR quarterly index ─────────────────────────────────────────────

def fetch_quarter_index(year: int, quarter: int) -> pd.DataFrame | None:
    """
    Download one EDGAR quarterly index file and return 8-K rows.

    The company.idx is nominally fixed-width (62 / 12 / 12 / 12 / rest) but
    the padding varies slightly across years.  The only robust approach is:
      - Form type:   line[62:74].strip()  (stable; always 8-K or 8-K/A)
      - CIK:         line[74:86].strip()  (stable)
      - Date:        regex search for YYYY-MM-DD in line[86:]
      - Filename:    regex search for edgar/data/... in line[86:]
      - Company name: line[0:62].strip()
    This avoids the off-by-a-few-chars date truncation seen in earlier runs.
    """
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
        tail = line[86:]
        date_m  = date_pat.search(tail)
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
    """Download all quarterly indexes and return combined 8-K filing index."""
    frames = []
    quarters = [
        (y, q) for y in range(START_YEAR, END_YEAR + 1) for q in range(1, 5)
    ]
    for year, quarter in tqdm(quarters, desc="Quarterly indexes"):
        df = fetch_quarter_index(year, quarter)
        if df is not None:
            frames.append(df)

    index = pd.concat(frames, ignore_index=True)

    # Derive accession number (no dashes) from filename.
    # Filename format: edgar/data/{cik}/{acc-with-dashes}-index.htm
    # Accession pattern: XXXXXXXXXX-YY-ZZZZZZ (10-2-6 digits with dashes)
    index["acc_nodash"] = (
        index["filename"]
        .str.extract(r"(\d{10}-\d{2}-\d{6})", expand=False)
        .str.replace("-", "", regex=False)
        .fillna("")
    )
    log.info("Total 8-K rows in quarterly index: %d", len(index))
    return index


# ── Step 3: Intersect ─────────────────────────────────────────────────────────

def normalize_acc(raw: str) -> str:
    """
    Normalize an accession number to 18-digit no-dash form.
    EFTS _id format is '{accession-with-dashes}:{document-filename}',
    e.g. '0001214659-10-002151:ex99_1.htm' → '000121465910002151'
    """
    # Take only the part before the colon (strips document filename)
    acc = raw.split(":")[0]
    # Remove dashes to get 18-digit nodash form
    return acc.replace("-", "")


def intersect_with_index(
    efts_set: set[str], index: pd.DataFrame
) -> pd.DataFrame:
    """Keep only index rows whose accession number is in the EFTS candidate set."""
    # Log index sample to verify acc_nodash extraction
    log.info("Index acc_nodash sample: %s", index["acc_nodash"].dropna().head(5).tolist())

    efts_nodash = {normalize_acc(a) for a in efts_set}
    log.info("EFTS normalized sample: %s", list(efts_nodash)[:5])

    mask = index["acc_nodash"].isin(efts_nodash)
    candidates = index[mask].copy()
    log.info(
        "Candidates after intersection: %d / %d index rows",
        len(candidates), len(index),
    )
    return candidates


# ── Step 4: Parse filings with edgarParser ────────────────────────────────────

def build_filing_url(row: pd.Series) -> str:
    """
    Construct the full-submission text URL for a filing.
    edgarParser (parse_8K.py) expects the full submission .txt file.
    EDGAR stores it at: .../edgar/data/{cik}/{acc-with-dashes}.txt
    The acc_nodash (18 digits) must be converted back to dashed format
    (XXXXXXXXXX-YY-ZZZZZZ) for the URL to resolve correctly.
    """
    cik    = row["cik"].lstrip("0")
    acc_nd = row["acc_nodash"]
    # Reconstruct dashed accession: XXXXXXXXXX-YY-ZZZZZZ
    acc_dashed = f"{acc_nd[:10]}-{acc_nd[10:12]}-{acc_nd[12:]}"
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_dashed}.txt"


def extract_item401_row(filing_df: pd.DataFrame) -> pd.Series | None:
    """
    Given a DataFrame returned by parse_8k_filing(), return the Item 4.01 row,
    or None if not found.

    Matches both:
      - "Item 4.01" (post-Dec-2004 format, new Form 8-K rules)
      - "Item 4"    (pre-2004 format; old Form 8-K used Item 4 for auditor changes)
    Both patterns refer to auditor change disclosures across our 2001-2023 sample.
    """
    if filing_df is None or filing_df.empty:
        return None
    # Match "Item 4.01" (new) OR "Item 4" / "Item 4." (old pre-2004).
    # Negative lookahead (?!\.\d) prevents false matches on "Item 4.02"
    # (Departure of Directors) and other Item 4.XX sub-items: after consuming
    # "Item 4" (with empty optional .01), the lookahead fails if the next two
    # chars are "." followed by a digit — i.e., a different sub-item number.
    mask = filing_df["item"].str.contains(
        r"Item\s+4(?:\.01)?(?!\.\d)", flags=re.IGNORECASE, na=False
    )
    if not mask.any():
        return None
    return filing_df[mask].iloc[0]


def parse_item401_text(text: str) -> dict:
    """
    Extract structured fields from the Item 4.01 free text.
    Returns a dict with: auditor_out, auditor_in, reason, disagreements.
    """
    if not text or len(text) < 20:
        return {
            "auditor_out": None, "auditor_in": None,
            "reason": None, "disagreements": False,
        }

    # Classify reason: dismissal vs resignation
    dismissal_pat  = re.compile(r"dismiss|terminat|replac", re.IGNORECASE)
    resignation_pat = re.compile(r"resign|declin|withdrew", re.IGNORECASE)

    if dismissal_pat.search(text):
        reason = "dismissal"
    elif resignation_pat.search(text):
        reason = "resignation"
    else:
        reason = "unclassified"

    # Disagreements disclosed?
    # Item 304 requires firms to state whether disagreements occurred.
    # Most filings say "there were no disagreements" — boilerplate.
    # We need to distinguish negated mentions from affirmative disclosures.
    _no_disagree = re.compile(
        r"(?:were|was|have\s+been|had)\s+no\s+disagreement"
        r"|no\s+disagreements?\s+(?:with|between|on|regarding)"
        r"|(?:not|never)\s+(?:had|have|been)\s+(?:any\s+)?disagreement"
        r"|has\s+had\s+no\s+disagreement"
        r"|there\s+(?:were|have\s+been|was)\s+no\s+disagreement"
        r"|without\s+(?:any\s+)?disagreement",
        re.IGNORECASE,
    )
    _yes_disagree = re.compile(
        r"(?:had|have|were)\s+(?:the\s+following\s+)?disagreements?"
        r"|following\s+disagreements?"
        r"|certain\s+disagreements?"
        r"|(?:a|the)\s+disagreement\s+(?:with|between|regarding|concerning|over|about)"
        r"|disagreements?\s+(?:arose|existed|occurred|related|involved|concerned)"
        r"|disagreements?\s+(?:as|described|set\s+forth|regarding)",
        re.IGNORECASE,
    )
    has_affirmative = bool(_yes_disagree.search(text))
    disagreements = has_affirmative

    # Auditor out: firm name near departure language
    out_match = re.search(
        r"(" + AUDITOR_PATTERN.pattern + r")"
        r"[^.]{0,200}?"
        r"(dismiss|terminat|replac|resign|former|previous|was\s+the)",
        text, re.IGNORECASE,
    )
    auditor_out = out_match.group(1).strip() if out_match else None

    # Auditor in: firm name near engagement language
    in_match = re.search(
        r"(engag|appoint|retain|select|hired|has\s+been\s+selected)"
        r"[^.]{0,150}?"
        r"(" + AUDITOR_PATTERN.pattern + r")",
        text, re.IGNORECASE,
    )
    # Also try reverse order: firm name then engagement verb
    if in_match is None:
        in_match = re.search(
            r"(" + AUDITOR_PATTERN.pattern + r")"
            r"[^.]{0,150}?"
            r"(engag|appoint|retain|select|hired|will\s+serve|as\s+its)",
            text, re.IGNORECASE,
        )
        auditor_in = in_match.group(1).strip() if in_match else None
    else:
        auditor_in = in_match.group(2).strip() if in_match else None

    return {
        "auditor_out":   auditor_out,
        "auditor_in":    auditor_in,
        "reason":        reason,
        "disagreements": disagreements,
    }


def parse_one_filing(row: pd.Series) -> dict:
    """Download, parse, and extract Item 4.01 for a single filing."""
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
        # parse_8K.py calls sys.exit("No Items Found") when it can't parse —
        # catch it so one bad filing doesn't kill the whole run.
        return {**base, "parse_status": "no_items_found",
                "item401_text": None, "auditor_out": None,
                "auditor_in": None, "reason": None, "disagreements": None}
    except Exception as exc:
        log.debug("edgarParser error for %s: %s", url, exc)
        return {**base, "parse_status": "parser_error",
                "item401_text": None, "auditor_out": None,
                "auditor_in": None, "reason": None, "disagreements": None}

    item_row = extract_item401_row(filing_df)
    if item_row is None:
        return {**base, "parse_status": "no_item401",
                "item401_text": None, "auditor_out": None,
                "auditor_in": None, "reason": None, "disagreements": None}

    text    = item_row.get("itemText", "") or ""   # column name in parse_8K.py
    fields  = parse_item401_text(text)

    return {
        **base,
        "parse_status": "ok",
        "item401_text": text[:4000],   # cap at 4K chars for storage
        **fields,
    }


# ── Step 5: Clean and write output ────────────────────────────────────────────

def classify_big4(name: str | None) -> bool | None:
    if name is None:
        return None
    return bool(BIG4_PATTERN.search(name))


def build_quality_direction(out_big4: bool | None, in_big4: bool | None) -> str:
    if out_big4 is None or in_big4 is None:
        return "unknown"
    if out_big4 and not in_big4:
        return "Big4_to_nonBig4"
    if not out_big4 and in_big4:
        return "nonBig4_to_Big4"
    if out_big4 and in_big4:
        return "Big4_to_Big4"
    return "nonBig4_to_nonBig4"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== 01_build_edgar_event_file.py  start ===")
    log.info("Sample period: %d–%d", START_YEAR, END_YEAR)

    # Step 1: EFTS pre-filter
    log.info("Step 1: EFTS pre-filter")
    efts_set = build_efts_candidate_set()

    # Step 2: Quarterly index
    log.info("Step 2: Download quarterly indexes")
    index = build_quarterly_index()

    # Step 3: Intersect
    log.info("Step 3: Intersect EFTS candidates with index")
    candidates = intersect_with_index(efts_set, index)

    # Step 4: Parse filings — with checkpointing so interrupted runs can resume
    log.info("Step 4: Parse %d candidate filings", len(candidates))

    # Load any previously completed results
    if CHECKPOINT_FILE.exists():
        completed = pd.read_csv(CHECKPOINT_FILE)
        done_accs = set(completed["acc_nodash"].astype(str))
        results   = completed.to_dict("records")
        log.info("Resuming from checkpoint: %d already parsed", len(results))
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
            warnings.simplefilter("ignore")   # suppress XMLParsedAsHTMLWarning
            results.append(parse_one_filing(row))

        # Save checkpoint periodically
        if (i + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False)

    # Final checkpoint save
    pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False)

    if not results:
        log.error(
            "No filings parsed. Check EFTS sample and index acc_nodash sample "
            "in the log above — the _id formats may not be matching."
        )
        return

    all_results = pd.DataFrame(results)

    # Write diagnostic log
    all_results.to_csv(LOG_FILE, index=False)
    log.info("Parse log written: %s", LOG_FILE)
    log.info("Status breakdown:\n%s", all_results["parse_status"].value_counts().to_string())

    # Step 5: Clean and write output
    clean = (
        all_results[all_results["parse_status"] == "ok"]
        .copy()
        .assign(
            date_filed      = lambda d: pd.to_datetime(d["date_filed"]),
            filing_year     = lambda d: d["date_filed"].dt.year,
            out_big4        = lambda d: d["auditor_out"].map(classify_big4),
            in_big4         = lambda d: d["auditor_in"].map(classify_big4),
            is_amendment    = lambda d: d["form_type"] == "8-K/A",
        )
        .assign(
            quality_direction = lambda d: [
                build_quality_direction(ob, ib)
                for ob, ib in zip(d["out_big4"], d["in_big4"])
            ]
        )
        .sort_values("date_filed")
        .reset_index(drop=True)
    )

    # Write parquet
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(OUT_FILE, index=False)
    log.info("Event file written: %s", OUT_FILE)
    log.info("Rows: %d", len(clean))
    if len(clean) == 0:
        log.error("Zero events with parse_status=ok. Check EFTS intersection and parser.")
        return
    log.info(
        "Date range: %s — %s",
        clean["date_filed"].min().date(),
        clean["date_filed"].max().date(),
    )

    # Summary by year and quality direction
    summary = (
        clean.groupby(["filing_year", "quality_direction"])
        .size()
        .unstack(fill_value=0)
    )
    log.info("Annual breakdown:\n%s", summary.to_string())
    log.info("=== done ===")


if __name__ == "__main__":
    main()
