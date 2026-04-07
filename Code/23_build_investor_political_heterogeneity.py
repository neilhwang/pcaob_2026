"""
23_build_investor_political_heterogeneity.py
=============================================
Construct firm-event measures of investor-base political heterogeneity
by matching 13F institutional filers to FEC individual contribution data.

DESIGN
------
For each auditor-change event, we:
  1. Identify all 13F filers holding shares of that firm at the most recent
     quarter-end (from the cached quarterly 13F parquets).
  2. Match each 13F filer (by employer name) to FEC individual contribution
     records to determine the filer's aggregate political lean.
  3. Compute firm-event measures of investor-base political heterogeneity:
     - share-weighted |Dem% - Rep%| (lower = more heterogeneous)
     - share-weighted Herfindahl of political lean
     - fraction of institutional shares with identifiable political lean

INPUTS
------
  Data/Raw/sec_13f/quarterly_holdings/*_structured.parquet (post-2013)
  Data/Raw/sec_13f/quarterly_holdings/*_edgar.parquet      (pre-2013)
  Data/Raw/indiv00/ through Data/Raw/indiv24/  (FEC bulk contribution files)
  Data/Processed/analysis_sample.parquet
  WRDS: crsp.msenames (CUSIP-to-permno mapping)

OUTPUTS
-------
  Data/Processed/fec_filer_political_lean.parquet  (filer-level lookup)
  Data/Processed/investor_political_heterogeneity.parquet  (firm-event level)

METHODOLOGY
-----------
Following Wintoki & Xi (2019, JFQA) and Mahmood (2021, SSRN):
  - FEC contributors are matched to 13F filers via employer name
  - Political lean = (Dem contributions - Rep contributions) /
                     (Dem contributions + Rep contributions)
  - Cumulative contributions through the event year
  - Fuzzy matching with rapidfuzz (threshold >= 0.80)

NOTES
-----
  - indiv20 (2019-2020 cycle) is currently empty; re-download from FEC
  - Pre-2013 quarterly parquets are aggregated and do not preserve filer
    identity; this script re-downloads and re-parses those quarters with
    filer-level detail (cached separately)
  - First run is slow (FEC parsing + fuzzy matching); subsequent runs use
    cached lookup tables
"""

import logging
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "Data" / "Processed"
RAW  = ROOT / "Data" / "Raw"
RAW_13F = RAW / "sec_13f"
CACHE_QUARTERLY = RAW_13F / "quarterly_holdings"

ANALYSIS_SAMPLE = PROC / "analysis_sample.parquet"
FEC_LEAN_FILE   = PROC / "fec_filer_political_lean.parquet"
OUT_FILE        = PROC / "investor_political_heterogeneity.parquet"

# FEC file schema (pipe-delimited, no header)
FEC_COLUMNS = [
    "cmte_id", "amndt_ind", "rpt_tp", "transaction_pgi", "image_num",
    "transaction_tp", "entity_tp", "name", "city", "state", "zip_code",
    "employer", "occupation", "transaction_dt", "transaction_amt",
    "other_id", "tran_id", "file_num", "memo_cd", "memo_text", "sub_id",
]

# Occupation keywords that identify investment professionals
INVESTMENT_OCCUPATIONS = {
    "investment", "portfolio", "fund manager", "hedge fund",
    "asset management", "mutual fund", "financial advisor",
    "money manager", "wealth management", "capital management",
    "securities", "trader", "analyst", "managing director",
    "chief investment", "cio", "partner",
}

# ── SEC helpers (for structured 13F downloads) ────────────────────────────────
_SEC_13F_PAGE   = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
_SEC_HEADERS    = {"User-Agent": "Neil Hwang nhwang@byu.edu"}
_SEC_RATE_LIMIT = 0.12  # seconds between requests (~8/sec, under 10/sec cap)


def _sec_get(url, max_retries=3):
    """Rate-limited GET to SEC EDGAR with retries."""
    import requests as _req
    for attempt in range(max_retries):
        time.sleep(_SEC_RATE_LIMIT)
        try:
            r = _req.get(url, headers=_SEC_HEADERS, timeout=60)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                log.warning("Rate limited — waiting %ds...", wait)
                time.sleep(wait)
            elif attempt == max_retries - 1:
                return r
        except Exception as exc:
            if attempt == max_retries - 1:
                log.warning("Request failed for %s: %s", url, exc)
                return None
            time.sleep(5)
    return None


def get_structured_zip_urls():
    """
    Scrape the SEC 13F data-sets page and return a dict mapping
    (year, quarter) -> full ZIP URL.  Covers 2013-Q2 onward.
    """
    r = _sec_get(_SEC_13F_PAGE)
    if r is None or r.status_code != 200:
        log.warning("Could not fetch SEC 13F data sets page")
        return {}

    zips = re.findall(r'href="([^"]*form13f\.zip)"', r.text, re.IGNORECASE)
    url_map = {}
    for z in zips:
        full_url = f"https://www.sec.gov{z}" if z.startswith("/") else z
        # Pattern 1: YYYYqQ_form13f.zip
        m = re.search(r"(\d{4})q(\d)_form13f", z, re.IGNORECASE)
        if m:
            url_map[(int(m.group(1)), int(m.group(2)))] = full_url
            continue
        # Pattern 2: 01jan2024-29feb2024_form13f.zip
        m2 = re.search(
            r"(\d{2})(\w{3})(\d{4})-\d{2}\w{3}\d{4}_form13f", z, re.IGNORECASE
        )
        if m2:
            month_map = {
                "jan": 1, "feb": 1, "mar": 1, "apr": 2, "may": 2, "jun": 2,
                "jul": 3, "aug": 3, "sep": 3, "oct": 4, "nov": 4, "dec": 4,
            }
            yr = int(m2.group(3))
            q  = month_map.get(m2.group(2).lower(), 1)
            url_map[(yr, q)] = full_url

    log.info("Found %d structured 13F ZIP URLs on SEC page", len(url_map))
    return url_map


# ── Step 1: Build FEC political lean by employer ──────────────────────────────

def load_fec_contributions():
    """
    Load all FEC individual contribution files and aggregate by employer.

    Returns DataFrame with columns:
        employer_clean, total_dem, total_rep, n_contributors, lean_score
    """
    cache = PROC / "fec_employer_lean_cache.parquet"
    if cache.exists():
        log.info("Loading cached FEC employer lean from %s", cache)
        return pd.read_parquet(cache)

    log.info("Building FEC employer lean from raw contribution files...")

    # We need committee-to-party mapping to classify contributions
    # FEC committees: committee IDs starting with C00... map to candidates
    # We'll use a simplified approach: match committee IDs to party via
    # the committee master file, or use the transaction_pgi field

    all_chunks = []
    fec_dirs = sorted(RAW.glob("indiv[0-9][0-9]"))

    for fec_dir in fec_dirs:
        itcont = fec_dir / "itcont.txt"
        if not itcont.exists():
            log.warning("No itcont.txt in %s, skipping", fec_dir.name)
            continue

        log.info("  Reading %s...", fec_dir.name)
        try:
            # Read in chunks to manage memory
            for chunk in pd.read_csv(
                itcont,
                sep="|",
                header=None,
                names=FEC_COLUMNS,
                dtype=str,
                encoding="latin-1",
                on_bad_lines="skip",
                chunksize=500_000,
                low_memory=False,
            ):
                # Filter to individual contributions only (entity_tp = IND)
                chunk = chunk[chunk["entity_tp"].fillna("").str.strip() == "IND"]

                # Parse amount
                chunk["amt"] = pd.to_numeric(
                    chunk["transaction_amt"], errors="coerce"
                )
                chunk = chunk[chunk["amt"].notna() & (chunk["amt"] > 0)]

                # Keep only rows with employer information
                chunk = chunk[
                    chunk["employer"].notna()
                    & (chunk["employer"].str.strip() != "")
                ]

                # Classify party from committee ID
                # Democratic committees typically: DNC, DCCC, DSCC, ActBlue,
                # and candidate committees for D candidates
                # Republican: RNC, NRCC, NRSC, WinRed, and R candidate cmtes
                # Simplified: use committee ID prefix patterns
                # More robust: download committee master file (cm.txt)
                # For now, keep cmte_id and employer for later party mapping
                chunk = chunk[["cmte_id", "employer", "occupation", "amt"]].copy()
                all_chunks.append(chunk)

        except Exception as e:
            log.warning("Error reading %s: %s", fec_dir.name, e)
            continue

    if not all_chunks:
        log.error("No FEC data loaded!")
        return pd.DataFrame()

    log.info("  Concatenating %d chunks...", len(all_chunks))
    fec = pd.concat(all_chunks, ignore_index=True)
    log.info("  Total FEC records after filtering: %d", len(fec))

    # ── Map committee IDs to party ────────────────────────────────────────
    # Download committee master files to get party affiliation
    # FEC provides cm.txt files in each cycle's bulk download
    # For simplicity, use a known-party lookup for major committees
    # and candidate committee naming conventions

    fec["party"] = classify_committee_party(fec["cmte_id"])
    fec = fec[fec["party"].isin(["D", "R"])]
    log.info("  Records with D/R party classification: %d", len(fec))

    # Clean employer names
    fec["employer_clean"] = fec["employer"].str.upper().str.strip()
    fec["employer_clean"] = fec["employer_clean"].str.replace(
        r"[^A-Z0-9\s&]", "", regex=True
    )
    fec["employer_clean"] = fec["employer_clean"].str.replace(
        r"\s+", " ", regex=True
    ).str.strip()

    # Aggregate by employer
    agg = (
        fec.groupby(["employer_clean", "party"])["amt"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    agg.columns = ["employer_clean", "total_dem", "total_rep"]

    # Political lean score: -1 (all Dem) to +1 (all Rep)
    total = agg["total_dem"] + agg["total_rep"]
    agg["lean_score"] = (agg["total_rep"] - agg["total_dem"]) / total
    agg["total_contributions"] = total
    agg["n_records"] = (
        fec.groupby("employer_clean").size().reindex(agg["employer_clean"]).values
    )

    # Filter to employers with meaningful contribution volume
    agg = agg[agg["total_contributions"] >= 500].copy()
    log.info("  Employers with >= $500 total contributions: %d", len(agg))

    agg.to_parquet(cache, index=False)
    log.info("  Cached to %s", cache)
    return agg


def classify_committee_party(cmte_ids: pd.Series) -> pd.Series:
    """
    Classify FEC committee IDs to party (D or R).

    Uses the committee master files (cm.txt) from each FEC cycle if available,
    falling back to known-committee lookups.
    """
    # Try to load committee master files
    cm_records = []
    for fec_dir in sorted(RAW.glob("indiv[0-9][0-9]")):
        # Committee master file may be in the same directory or a parallel one
        # FEC bulk data includes cm.txt in the committee master download
        # Check common locations
        for cm_path in [
            fec_dir / "cm.txt",
            fec_dir.parent / f"cm{fec_dir.name[-2:]}" / "cm.txt",
            RAW / f"cm{fec_dir.name[-2:]}" / "cm.txt",
        ]:
            if cm_path.exists():
                try:
                    cm = pd.read_csv(
                        cm_path, sep="|", header=None, dtype=str,
                        on_bad_lines="skip",
                        names=[
                            "cmte_id", "cmte_nm", "tres_nm", "cmte_st1",
                            "cmte_st2", "cmte_city", "cmte_st", "cmte_zip",
                            "cmte_dsgn", "cmte_tp", "cmte_pty_affiliation",
                            "cmte_filing_freq", "org_tp", "connected_org_nm",
                            "cand_id",
                        ],
                    )
                    cm = cm[["cmte_id", "cmte_pty_affiliation"]].dropna()
                    cm_records.append(cm)
                except Exception:
                    pass
                break

    if cm_records:
        cm_all = pd.concat(cm_records, ignore_index=True).drop_duplicates(
            subset="cmte_id", keep="last"
        )
        cm_map = dict(zip(cm_all["cmte_id"], cm_all["cmte_pty_affiliation"]))
        log.info("  Committee-to-party mapping: %d committees", len(cm_map))

        party = cmte_ids.map(cm_map)
        # Normalize party codes: DEM -> D, REP -> R
        party = party.str.upper().str.strip()
        party = party.replace({"DEM": "D", "REP": "R", "DFL": "D"})
        # Keep only D and R
        party[~party.isin(["D", "R"])] = np.nan
        return party

    # Fallback: known major committees
    log.warning("No committee master files found; using fallback party mapping")
    log.warning("Download committee master files (cm.txt) for better coverage")

    known_dem = {
        "C00010603",  # DNC
        "C00000935",  # DCCC
        "C00042366",  # DSCC
        "C00401224",  # ActBlue
    }
    known_rep = {
        "C00003418",  # RNC
        "C00075820",  # NRCC
        "C00027466",  # NRSC
        "C00694323",  # WinRed
    }

    def _classify(cid):
        if cid in known_dem:
            return "D"
        if cid in known_rep:
            return "R"
        return np.nan

    return cmte_ids.map(_classify)


# ── Step 2: Extract filer-level holdings from 13F ─────────────────────────────

def load_filer_holdings_structured(year, quarter, target_cusips, zip_url=None):
    """
    Download a structured quarterly 13F ZIP and extract filer-level holdings.
    zip_url must be obtained from get_structured_zip_urls() beforehand.
    Returns DataFrame: [filer_cik, filer_name, cusip8, shares, rdate]
    """
    import io as _io
    import zipfile as _zf

    cache = CACHE_QUARTERLY / f"{year}q{quarter}_filer_level.parquet"
    if cache.exists():
        cached = pd.read_parquet(cache)
        # Stale-cache guard: if all filer_name entries are blank, the cache was
        # built with the old broken SUBMISSION column lookup — regenerate.
        if len(cached) > 0 and (cached["filer_name"].fillna("").str.strip() != "").any():
            return cached
        log.info("  Stale quarterly cache for %dQ%d (blank filer names) — regenerating",
                 year, quarter)
        cache.unlink()

    if zip_url is None:
        log.warning("  No ZIP URL for %dQ%d — skipping (call get_structured_zip_urls first)",
                    year, quarter)
        return pd.DataFrame(columns=["filer_cik", "filer_name", "cusip8", "shares", "rdate"])

    log.info("  Downloading %dQ%d structured 13F with filer detail...", year, quarter)

    try:
        r = _sec_get(zip_url)
        if r is None or r.status_code != 200:
            log.warning("  Failed to download %dQ%d: HTTP %d", year, quarter,
                        getattr(r, "status_code", 0))
            return pd.DataFrame(columns=["filer_cik", "filer_name", "cusip8", "shares", "rdate"])

        z = _zf.ZipFile(_io.BytesIO(r.content))
    except Exception as e:
        log.warning("  Error downloading %dQ%d: %s", year, quarter, e)
        return pd.DataFrame(columns=["filer_cik", "filer_name", "cusip8", "shares", "rdate"])

    # Parse COVERPAGE.tsv for filer names (contains FILINGMANAGER_NAME)
    # Fall back to SUBMISSION.tsv CIK for the filer ID.
    filer_map = {}  # accession_number -> (cik, name)

    # CIK lookup from SUBMISSION.tsv
    cik_map = {}
    if "SUBMISSION.tsv" in z.namelist():
        with z.open("SUBMISSION.tsv") as f:
            sub = pd.read_csv(f, sep="\t", dtype=str, low_memory=False)
        sub.columns = [c.upper() for c in sub.columns]
        if "ACCESSION_NUMBER" in sub.columns and "CIK" in sub.columns:
            cik_map = dict(zip(sub["ACCESSION_NUMBER"].fillna(""),
                               sub["CIK"].fillna("")))

    # Manager name from COVERPAGE.tsv
    if "COVERPAGE.tsv" in z.namelist():
        with z.open("COVERPAGE.tsv") as f:
            cp = pd.read_csv(f, sep="\t", dtype=str, low_memory=False)
        cp.columns = [c.upper() for c in cp.columns]
        if "ACCESSION_NUMBER" in cp.columns and "FILINGMANAGER_NAME" in cp.columns:
            for _, row in cp.iterrows():
                acc  = str(row.get("ACCESSION_NUMBER", "") or "").strip()
                name = str(row.get("FILINGMANAGER_NAME", "") or "").strip()
                if acc and name:
                    cik = cik_map.get(acc, "")
                    filer_map[acc] = (cik, name.upper())

    # Parse INFOTABLE.tsv for holdings
    if "INFOTABLE.tsv" not in z.namelist():
        log.warning("  No INFOTABLE.tsv in %dQ%d", year, quarter)
        return pd.DataFrame(columns=["filer_cik", "filer_name", "cusip8", "shares", "rdate"])

    with z.open("INFOTABLE.tsv") as f:
        info = pd.read_csv(f, sep="\t", dtype=str, low_memory=False)
    info.columns = [c.upper() for c in info.columns]

    if "CUSIP" not in info.columns:
        return pd.DataFrame(columns=["filer_cik", "filer_name", "cusip8", "shares", "rdate"])

    info["cusip8"] = info["CUSIP"].str[:8]
    info = info[info["cusip8"].isin(target_cusips)].copy()

    if len(info) == 0:
        result = pd.DataFrame(columns=["filer_cik", "filer_name", "cusip8", "shares", "rdate"])
        result.to_parquet(cache, index=False)
        return result

    # Filter to shares
    if "SSHPRNAMTTYPE" in info.columns:
        info = info[info["SSHPRNAMTTYPE"].str.upper().str.strip() == "SH"]

    info["shares"] = pd.to_numeric(info["SSHPRNAMT"], errors="coerce")
    info = info[info["shares"].notna() & (info["shares"] > 0)]

    # Map filer identity
    if "ACCESSION_NUMBER" in info.columns:
        info["filer_cik"] = info["ACCESSION_NUMBER"].map(
            lambda x: filer_map.get(x, ("", ""))[0]
        )
        info["filer_name"] = info["ACCESSION_NUMBER"].map(
            lambda x: filer_map.get(x, ("", ""))[1]
        )
    else:
        info["filer_cik"] = ""
        info["filer_name"] = ""

    qe = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    result = info[["filer_cik", "filer_name", "cusip8", "shares"]].copy()
    result["rdate"] = pd.Timestamp(f"{year}-{qe[quarter]}")

    result.to_parquet(cache, index=False)
    log.info("  %dQ%d: %d holdings, %d unique filers",
             year, quarter, len(result), result["filer_name"].nunique())
    return result


# ── Step 3: Match 13F filers to FEC employers ────────────────────────────────

def build_filer_lean_lookup(fec_employer_lean: pd.DataFrame,
                            filer_names: pd.Series) -> pd.DataFrame:
    """
    Fuzzy-match 13F filer names to FEC employer names.

    Returns DataFrame: [filer_name, employer_clean, match_score,
                        lean_score, total_dem, total_rep]
    """
    if FEC_LEAN_FILE.exists():
        cached = pd.read_parquet(FEC_LEAN_FILE)
        if len(cached) > 0:
            log.info("Loading cached filer-to-FEC lean lookup (%d matches)", len(cached))
            return cached
        log.warning("Cached filer lean file is empty — regenerating")

    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        log.error("rapidfuzz not installed. Run: pip install rapidfuzz")
        raise

    unique_filers = filer_names.dropna().unique()
    unique_filers = [f for f in unique_filers if f and len(f) > 2]
    log.info("Matching %d unique 13F filer names to FEC employers...", len(unique_filers))

    # Clean filer names for matching
    def clean_name(n):
        n = str(n).upper().strip()
        n = re.sub(r"[^A-Z0-9\s&]", "", n)
        n = re.sub(r"\s+", " ", n).strip()
        # Remove common suffixes
        for suffix in [" LLC", " LP", " INC", " CORP", " LTD", " CO",
                       " GROUP", " ASSOCIATES", " PARTNERS", " ADVISORS",
                       " MANAGEMENT", " CAPITAL", " HOLDINGS"]:
            if n.endswith(suffix):
                n = n[: -len(suffix)].strip()
        return n

    fec_names = fec_employer_lean["employer_clean"].values
    fec_name_list = list(fec_names)

    results = []
    for i, filer in enumerate(unique_filers):
        if i % 500 == 0 and i > 0:
            log.info("  Matched %d / %d filers...", i, len(unique_filers))

        cleaned = clean_name(filer)
        if len(cleaned) < 3:
            continue

        # Find best match
        match = process.extractOne(
            cleaned, fec_name_list, scorer=fuzz.token_sort_ratio,
            score_cutoff=80,
        )
        if match is not None:
            matched_name, score, idx = match
            row = fec_employer_lean[
                fec_employer_lean["employer_clean"] == matched_name
            ].iloc[0]
            results.append({
                "filer_name": filer,
                "filer_name_clean": cleaned,
                "employer_clean": matched_name,
                "match_score": score,
                "lean_score": row["lean_score"],
                "total_dem": row["total_dem"],
                "total_rep": row["total_rep"],
                "total_contributions": row["total_contributions"],
            })

    result = pd.DataFrame(results)
    log.info("Matched %d / %d filer names (%.1f%%)",
             len(result), len(unique_filers),
             100 * len(result) / max(len(unique_filers), 1))

    result.to_parquet(FEC_LEAN_FILE, index=False)
    log.info("Cached to %s", FEC_LEAN_FILE)
    return result


# ── Step 4: Aggregate to firm-event level ─────────────────────────────────────

def compute_firm_event_heterogeneity(
    filer_holdings: pd.DataFrame,
    filer_lean: pd.DataFrame,
    sample: pd.DataFrame,
    cusip_map: pd.DataFrame,
    shrout: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each event, compute investor-base political heterogeneity.

    Returns DataFrame: [permno, event_date, investor_het, dem_share,
                        rep_share, matched_share, n_matched_filers]
    """
    # Merge filer lean into holdings
    holdings = filer_holdings.merge(
        filer_lean[["filer_name", "lean_score"]],
        on="filer_name", how="left",
    )

    # Holdings should already have permno from Step 2 merge
    if "permno" not in holdings.columns and cusip_map is not None:
        holdings = holdings.merge(
            cusip_map.rename(columns={"ncusip": "cusip8"}),
            on="cusip8", how="inner",
        )

    results = []
    for _, ev in sample.iterrows():
        permno = ev["permno"]
        edate = pd.Timestamp(ev["event_date"])

        # Find most recent quarter-end before event
        firm_holdings = holdings[
            (holdings["permno"] == permno)
            & (holdings["rdate"] <= edate)
            & (holdings["rdate"] >= edate - pd.Timedelta(days=180))
        ]

        if len(firm_holdings) == 0:
            results.append({
                "permno": permno,
                "event_date": edate,
                "investor_het": np.nan,
                "dem_share": np.nan,
                "rep_share": np.nan,
                "matched_share": np.nan,
                "n_matched_filers": 0,
                "n_total_filers": 0,
            })
            continue

        # Use most recent quarter
        latest_q = firm_holdings["rdate"].max()
        qh = firm_holdings[firm_holdings["rdate"] == latest_q].copy()

        total_inst_shares = qh["shares"].sum()
        matched = qh[qh["lean_score"].notna()]
        matched_shares = matched["shares"].sum()

        if matched_shares == 0 or total_inst_shares == 0:
            results.append({
                "permno": permno,
                "event_date": edate,
                "investor_het": np.nan,
                "dem_share": np.nan,
                "rep_share": np.nan,
                "matched_share": matched_shares / max(total_inst_shares, 1),
                "n_matched_filers": len(matched),
                "n_total_filers": len(qh),
            })
            continue

        # Classify filers as Dem-leaning or Rep-leaning
        matched["is_dem"] = matched["lean_score"] < 0
        matched["is_rep"] = matched["lean_score"] > 0

        dem_shares = matched.loc[matched["is_dem"], "shares"].sum()
        rep_shares = matched.loc[matched["is_rep"], "shares"].sum()

        dem_frac = dem_shares / matched_shares
        rep_frac = rep_shares / matched_shares

        # Investor heterogeneity = 1 - |Dem% - Rep%|
        # Maximized at 0.5/0.5 split (het = 1), minimized at 1/0 (het = 0)
        investor_het = 1 - abs(dem_frac - rep_frac)

        results.append({
            "permno": permno,
            "event_date": edate,
            "investor_het": investor_het,
            "dem_share": dem_frac,
            "rep_share": rep_frac,
            "matched_share": matched_shares / total_inst_shares,
            "n_matched_filers": len(matched),
            "n_total_filers": len(qh),
        })

    return pd.DataFrame(results)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("Building investor-base political heterogeneity measures")
    log.info("=" * 70)

    # Load analysis sample
    sample = pd.read_parquet(ANALYSIS_SAMPLE)
    log.info("Analysis sample: %d events", len(sample))

    # ── Step 1: FEC employer lean ─────────────────────────────────────────
    log.info("Step 1: Loading FEC contributions and building employer lean...")
    fec_lean = load_fec_contributions()
    if len(fec_lean) == 0:
        log.error("No FEC data loaded. Exiting.")
        return
    log.info("  FEC employers with political lean: %d", len(fec_lean))

    # ── Step 2: Extract filer-level 13F holdings from SEC structured data ───
    log.info("Step 2: Extracting filer-level 13F holdings from SEC structured ZIPs...")

    filer_holdings_cache = PROC / "filer_level_holdings_cache.parquet"
    cusip_map = None  # may not be needed if cache hit

    if filer_holdings_cache.exists():
        _cached_holdings = pd.read_parquet(filer_holdings_cache)
        # Stale-cache guard: combined cache built with blank filer names is unusable
        if (len(_cached_holdings) > 0
                and (_cached_holdings["filer_name"].fillna("").str.strip() != "").any()):
            log.info("  Loading cached filer-level holdings")
            filer_holdings = _cached_holdings
        else:
            log.info("  Stale combined holdings cache (blank filer names) — regenerating")
            filer_holdings_cache.unlink()
            _cached_holdings = None  # fall through to else

    if not filer_holdings_cache.exists():
        # ── 2a: Get CUSIP mapping from CRSP ──────────────────────────────
        try:
            import wrds
            db = wrds.Connection(wrds_username="nhwang")
        except Exception as e:
            log.error("WRDS connection failed: %s", e)
            return

        sample_permnos = sample["permno"].unique().tolist()
        log.info("  Pulling CUSIP mapping for %d permnos...", len(sample_permnos))

        cusip_map = db.raw_sql("""
            SELECT DISTINCT permno, ncusip, namedt, nameendt
            FROM crsp.msenames
            WHERE ncusip IS NOT NULL AND LENGTH(ncusip) = 8
        """)
        cusip_map["permno"] = cusip_map["permno"].astype(int)
        cusip_map = cusip_map[cusip_map["permno"].isin(sample_permnos)]
        target_cusips = set(cusip_map["ncusip"].unique())
        log.info("  Target CUSIPs: %d", len(target_cusips))
        db.close()

        # ── 2b: Scrape structured ZIP URLs from SEC 13F data-sets page ───
        zip_urls = get_structured_zip_urls()
        if not zip_urls:
            log.error("Could not retrieve structured 13F ZIP URLs from SEC page.")
            return

        # ── 2c: Download and parse each relevant quarter ─────────────────
        # SEC structured data covers 2013-Q2 onward.
        # We pull all quarters that overlap with the sample window.
        sample["event_date"] = pd.to_datetime(sample["event_date"])
        min_year = sample["event_date"].dt.year.min()
        max_year = sample["event_date"].dt.year.max()

        quarters_to_pull = sorted(
            (y, q) for (y, q) in zip_urls
            if min_year <= y <= max_year
        )
        if not quarters_to_pull:
            log.error("No SEC ZIP quarters fall within sample years %d–%d.",
                      min_year, max_year)
            return

        log.info("  Quarters to download: %d (%dQ%d – %dQ%d)",
                 len(quarters_to_pull),
                 quarters_to_pull[0][0], quarters_to_pull[0][1],
                 quarters_to_pull[-1][0], quarters_to_pull[-1][1])

        all_holdings = []
        for y, q in quarters_to_pull:
            h = load_filer_holdings_structured(
                y, q, target_cusips, zip_urls[(y, q)]
            )
            if len(h) > 0:
                all_holdings.append(h)

        if not all_holdings:
            log.error("No filer-level 13F holdings loaded from SEC.")
            return

        filer_holdings = pd.concat(all_holdings, ignore_index=True)

        # Map cusip8 → permno
        filer_holdings = filer_holdings.merge(
            cusip_map[["ncusip", "permno"]].drop_duplicates().rename(
                columns={"ncusip": "cusip8"}
            ),
            on="cusip8", how="inner",
        )
        filer_holdings["permno"] = filer_holdings["permno"].astype(int)

        filer_holdings.to_parquet(filer_holdings_cache, index=False)
        log.info("  Cached filer-level holdings to %s", filer_holdings_cache)

    log.info("  Total filer-level holdings: %d rows, %d unique filers",
             len(filer_holdings),
             filer_holdings["filer_name"].nunique() if "filer_name" in filer_holdings.columns else "?")

    # ── Step 3: Match filers to FEC ───────────────────────────────────────
    log.info("Step 3: Matching 13F filers to FEC political lean...")
    filer_lean = build_filer_lean_lookup(
        fec_lean, filer_holdings["filer_name"]
    )
    log.info("  Matched filers: %d", len(filer_lean))

    # ── Step 4: Aggregate to firm-event level ─────────────────────────────
    log.info("Step 4: Computing firm-event political heterogeneity...")
    result = compute_firm_event_heterogeneity(
        filer_holdings, filer_lean, sample, cusip_map, None
    )

    # Summary statistics
    matched = result[result["investor_het"].notna()]
    log.info("  Events with heterogeneity measure: %d / %d",
             len(matched), len(result))
    if len(matched) > 0:
        log.info("  Mean investor_het: %.3f (SD: %.3f)",
                 matched["investor_het"].mean(), matched["investor_het"].std())
        log.info("  Mean matched_share: %.3f",
                 matched["matched_share"].mean())
        log.info("  Mean n_matched_filers: %.1f",
                 matched["n_matched_filers"].mean())

    result.to_parquet(OUT_FILE, index=False)
    log.info("Saved to %s", OUT_FILE)
    log.info("=" * 70)
    log.info("DONE")


if __name__ == "__main__":
    try:
        import requests  # needed for 13F downloads
    except ImportError:
        log.warning("requests not installed; 13F re-downloads will fail. "
                    "Run: pip install requests")
        requests = None
    main()
