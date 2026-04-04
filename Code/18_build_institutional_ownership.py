"""
18_build_institutional_ownership.py
====================================
Builds institutional ownership fractions for each auditor-change event
by downloading and parsing 13F filings from SEC EDGAR.

PURPOSE
-------
Build a retail ownership moderator for testing whether the polarization
effect on |CAR| is stronger among firms with higher retail ownership.

DATA SOURCES
------------
1. SEC EDGAR structured 13F data sets (2013-Q2 through 2023-Q4):
   Quarterly ZIP files from sec.gov containing INFOTABLE.tsv with
   individual 13F filer holdings. We aggregate shares across all filers
   per CUSIP x quarter.

2. SEC EDGAR individual 13F-HR filings (2000-Q4 through 2013-Q1):
   Downloaded from EDGAR Archives. We parse the plain-text or XML
   info tables embedded in each filing to extract CUSIP-level holdings.

3. WRDS CRSP (crsp.msf): Monthly shares outstanding for normalization,
   and crsp.msenames for CUSIP-to-permno mapping.

INPUTS
------
  Data/Processed/analysis_sample.parquet  -- permno, gvkey, cik, event_date
  SEC EDGAR 13F data (downloaded and cached in Data/Raw/sec_13f/)
  WRDS: crsp.msf, crsp.msenames

OUTPUTS
-------
  Data/Processed/institutional_ownership.parquet
      permno, event_date, inst_own_pct, retail_pct, io_source, rdate_matched

  Data/Raw/sec_13f/quarterly_holdings/  (cached parsed holdings)

METHODOLOGY
-----------
For each event, find the most recent 13F quarter-end BEFORE the event
date (within 180 days). Institutional ownership = total shares held by
all 13F filers / CRSP shares outstanding.

NOTES
-----
- WRDS username: nhwang
- SEC EDGAR rate limit: 10 requests/second (we throttle to ~8/sec)
- Institutional ownership is capped at [0, 1]
- First run downloads and parses SEC data (cached for subsequent runs)
- Pre-2013 downloads individual filings from EDGAR; expect 2-4 hours
- Post-2013 downloads quarterly ZIP files (~70MB each); ~30 min total
"""

from pathlib import Path
import io
import re
import time
import warnings
import zipfile

import numpy as np
import pandas as pd
import requests
import wrds

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
PROC     = ROOT / "Data" / "Processed"
RAW_13F  = ROOT / "Data" / "Raw" / "sec_13f"
CACHE_QUARTERLY = RAW_13F / "quarterly_holdings"

ANALYSIS_SAMPLE = PROC / "analysis_sample.parquet"
OUT_PATH = PROC / "institutional_ownership.parquet"

WRDS_USERNAME = "nhwang"
MAX_LOOKBACK_DAYS = 180

# SEC EDGAR requires a User-Agent with contact info
SEC_HEADERS = {"User-Agent": "PCAOB Research nhwang@byu.edu"}
SEC_RATE_LIMIT = 0.12  # seconds between requests (~8/sec, under 10/sec limit)

# URL for the SEC structured 13F data page
SEC_13F_PAGE = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"


# ── Helpers ──────────────────────────────────────────────────────────────────

def batch_list(lst, size=500):
    """Yield successive chunks of a list."""
    lst = list(lst)
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def sql_in_int(values):
    """Format a list of integers for a SQL IN clause."""
    return "(" + ", ".join(str(v) for v in values) + ")"


def quarter_end_dates(start_year, start_q, end_year, end_q):
    """Generate (year, quarter, date_string) from start to end."""
    qe = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    results = []
    y, q = start_year, start_q
    while (y, q) <= (end_year, end_q):
        results.append((y, q, f"{y}-{qe[q]}"))
        q += 1
        if q > 4:
            q = 1
            y += 1
    return results


def sec_get(url, max_retries=3):
    """GET request to SEC with rate limiting and retries."""
    for attempt in range(max_retries):
        time.sleep(SEC_RATE_LIMIT)
        try:
            r = requests.get(url, headers=SEC_HEADERS, timeout=60)
            if r.status_code == 200:
                return r
            elif r.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                if attempt == max_retries - 1:
                    return r
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"    Request failed: {e}")
                return None
            time.sleep(5)
    return None


# ── Step 1: Load sample and get CUSIPs from WRDS ────────────────────────────

def load_sample_and_cusips():
    """
    Load the analysis sample and build CUSIP-to-permno mapping from CRSP.
    Also pull CRSP monthly shares outstanding for normalization.
    """
    cols = ["gvkey", "permno", "cik", "event_date"]
    sample = pd.read_parquet(ANALYSIS_SAMPLE, columns=cols)
    sample["event_date"] = pd.to_datetime(sample["event_date"])
    sample["permno"] = sample["permno"].astype(int)
    sample = sample.drop_duplicates(subset=["permno", "event_date"]).reset_index(drop=True)
    print(f"  Events: {len(sample):,}  ({sample['permno'].nunique()} unique permnos)")

    permnos = sample["permno"].unique().tolist()
    print(f"  Pulling CUSIP mapping and shrout from CRSP...")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)

    # CUSIP-to-permno mapping with date ranges
    records = []
    for chunk in batch_list(permnos, 300):
        sql = f"""
            SELECT DISTINCT permno, ncusip, namedt, nameendt
            FROM crsp.msenames
            WHERE permno IN {sql_in_int(chunk)}
              AND ncusip IS NOT NULL
              AND LENGTH(ncusip) = 8
        """
        records.append(db.raw_sql(sql))

    cusip_map = pd.concat(records, ignore_index=True)
    cusip_map["namedt"] = pd.to_datetime(cusip_map["namedt"])
    cusip_map["nameendt"] = pd.to_datetime(cusip_map["nameendt"])
    cusip_map["permno"] = cusip_map["permno"].astype(int)

    # CRSP monthly shares outstanding
    date_start = (sample["event_date"].min()
                  - pd.Timedelta(days=MAX_LOOKBACK_DAYS + 90)).strftime("%Y-%m-%d")
    date_end = sample["event_date"].max().strftime("%Y-%m-%d")

    shr_records = []
    for chunk in batch_list(permnos, 300):
        sql = f"""
            SELECT permno, date, shrout
            FROM crsp.msf
            WHERE permno IN {sql_in_int(chunk)}
              AND date BETWEEN '{date_start}' AND '{date_end}'
              AND shrout IS NOT NULL
              AND shrout > 0
        """
        shr_records.append(db.raw_sql(sql))

    shrout = pd.concat(shr_records, ignore_index=True)
    shrout["date"] = pd.to_datetime(shrout["date"])
    shrout["permno"] = shrout["permno"].astype(int)
    print(f"  CRSP shrout rows: {len(shrout):,}")

    db.close()

    target_cusips = set(cusip_map["ncusip"].unique())
    print(f"  Target CUSIPs (8-digit): {len(target_cusips)}")

    return sample, cusip_map, shrout, target_cusips


# ── Step 2: SEC Structured Data (2013-Q2+) ──────────────────────────────────

def get_structured_zip_urls():
    """
    Scrape the SEC 13F data sets page to get URLs for quarterly ZIP files.
    Returns dict mapping (year, quarter) to full URL.
    """
    r = sec_get(SEC_13F_PAGE)
    if r is None or r.status_code != 200:
        print(f"  WARNING: Could not fetch SEC 13F data sets page")
        return {}

    zips = re.findall(r'href="([^"]*form13f\.zip)"', r.text, re.IGNORECASE)
    url_map = {}

    for z in zips:
        full_url = f"https://www.sec.gov{z}" if z.startswith("/") else z

        # YYYYqQ_form13f.zip format
        m = re.search(r'(\d{4})q(\d)_form13f', z, re.IGNORECASE)
        if m:
            url_map[(int(m.group(1)), int(m.group(2)))] = full_url
            continue

        # Date-range format: 01jan2024-29feb2024_form13f.zip
        m2 = re.search(r'(\d{2})(\w{3})(\d{4})-\d{2}\w{3}\d{4}_form13f',
                        z, re.IGNORECASE)
        if m2:
            month_map = {"jan": 1, "feb": 1, "mar": 1, "apr": 2, "may": 2,
                         "jun": 2, "jul": 3, "aug": 3, "sep": 3, "oct": 4,
                         "nov": 4, "dec": 4}
            yr = int(m2.group(3))
            q = month_map.get(m2.group(2).lower(), 1)
            url_map[(yr, q)] = full_url

    return url_map


def process_structured_quarter(year, quarter, zip_url, target_cusips):
    """
    Download a SEC structured 13F quarterly ZIP and extract aggregate
    holdings per CUSIP (filtered to target CUSIPs).

    Returns DataFrame: [cusip8, rdate, total_shares]
    """
    cache_file = CACHE_QUARTERLY / f"{year}q{quarter}_structured.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    print(f"    Downloading {year}Q{quarter} structured data...")
    r = sec_get(zip_url)
    if r is None or r.status_code != 200:
        print(f"    WARNING: Failed to download {zip_url}")
        return pd.DataFrame(columns=["cusip8", "rdate", "total_shares"])

    try:
        z = zipfile.ZipFile(io.BytesIO(r.content))
    except zipfile.BadZipFile:
        print(f"    WARNING: Bad ZIP file for {year}Q{quarter}")
        return pd.DataFrame(columns=["cusip8", "rdate", "total_shares"])

    if "INFOTABLE.tsv" not in z.namelist():
        print(f"    WARNING: No INFOTABLE.tsv in {year}Q{quarter}")
        return pd.DataFrame(columns=["cusip8", "rdate", "total_shares"])

    with z.open("INFOTABLE.tsv") as f:
        info = pd.read_csv(f, sep="\t", dtype=str, low_memory=False)

    info.columns = [c.upper() for c in info.columns]

    if "CUSIP" not in info.columns:
        print(f"    WARNING: No CUSIP column in {year}Q{quarter}")
        return pd.DataFrame(columns=["cusip8", "rdate", "total_shares"])

    info["cusip8"] = info["CUSIP"].str[:8]
    info = info[info["cusip8"].isin(target_cusips)].copy()

    if len(info) == 0:
        result = pd.DataFrame(columns=["cusip8", "rdate", "total_shares"])
        CACHE_QUARTERLY.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_file, index=False)
        return result

    # Filter to shares (not principal amount)
    if "SSHPRNAMTTYPE" in info.columns:
        info = info[info["SSHPRNAMTTYPE"].str.upper().str.strip() == "SH"]

    info["shares"] = pd.to_numeric(info["SSHPRNAMT"], errors="coerce")
    info = info[info["shares"].notna() & (info["shares"] > 0)]

    # Use canonical quarter-end date
    qe = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    canonical_rdate = pd.Timestamp(f"{year}-{qe[quarter]}")

    result = (
        info.groupby("cusip8")["shares"]
        .sum()
        .reset_index()
        .rename(columns={"shares": "total_shares"})
    )
    result["rdate"] = canonical_rdate

    CACHE_QUARTERLY.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_file, index=False)
    n_filers = info["ACCESSION_NUMBER"].nunique() if "ACCESSION_NUMBER" in info.columns else "?"
    print(f"    {year}Q{quarter}: {len(result)} CUSIPs, {n_filers} filers")
    return result


# ── Step 3: Pre-2013 EDGAR 13F filings ──────────────────────────────────────

def get_13f_filing_urls(year, quarter):
    """
    Download the EDGAR full-index for a given quarter and extract
    all 13F-HR filing URLs (excluding amendments).
    """
    url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/company.idx"
    r = sec_get(url)
    if r is None or r.status_code != 200:
        print(f"    WARNING: Could not fetch index for {year}Q{quarter}")
        return []

    filings = []
    for line in r.text.split("\n"):
        # Match 13F-HR but not 13F-HR/A
        if "13F-HR " in line and "13F-HR/A" not in line:
            parts = line.strip().split()
            filename = parts[-1] if parts else ""
            if filename.startswith("edgar/"):
                filings.append(f"https://www.sec.gov/Archives/{filename}")
    return filings


def parse_13f_filing(text, target_cusips):
    """
    Parse a 13F-HR filing (text or XML) to extract holdings for
    target CUSIPs.

    In text format, the CUSIP appears mid-line after the issuer name
    and class. The typical column order is:
        NAME, CLASS, CUSIP(9), SHARES, VALUE(x1000), SH/PRN, ...

    In XML format (some post-2009 filings), we extract from
    <infoTable> elements.

    Returns dict: cusip8 -> total shares held by this filer.
    """
    holdings = {}

    # ── Try XML format first ──
    if ("<informationtable" in text.lower()
            or "<ns1:informationtable" in text.lower()):
        entries = re.findall(
            r'<(?:ns1:)?infoTable>(.*?)</(?:ns1:)?infoTable>',
            text, re.DOTALL | re.IGNORECASE
        )
        for entry in entries:
            m_cusip = re.search(
                r'<(?:ns1:)?cusip>(.*?)</(?:ns1:)?cusip>',
                entry, re.IGNORECASE
            )
            if not m_cusip:
                continue
            cusip8 = m_cusip.group(1).strip()[:8]
            if cusip8 not in target_cusips:
                continue

            # Only count shares, not principal amount
            m_type = re.search(
                r'<(?:ns1:)?sshPrnamtType>(.*?)</(?:ns1:)?sshPrnamtType>',
                entry, re.IGNORECASE
            )
            if m_type and m_type.group(1).strip().upper() != "SH":
                continue

            m_shares = re.search(
                r'<(?:ns1:)?sshPrnamt>(.*?)</(?:ns1:)?sshPrnamt>',
                entry, re.IGNORECASE
            )
            if not m_shares:
                continue

            try:
                shares = int(float(m_shares.group(1).strip().replace(",", "")))
            except ValueError:
                continue

            if shares > 0:
                holdings[cusip8] = holdings.get(cusip8, 0) + shares

        return holdings

    # ── Plain text format ──
    # Column order varies across filers. Two common layouts:
    #   (A) NAME, CLASS, CUSIP, SHARES, VALUE(x$1000), SH/PRN, ...
    #   (B) NAME, CLASS, CUSIP, VALUE(x$1000), SHARES, SH/PRN, ...
    #
    # Robust heuristic: for virtually all stocks with price < $1000,
    # shares > value_in_thousands. We always take max(num1, num2) as
    # shares. This bypasses fragile header-column-order detection.

    lines = text.split("\n")

    for line in lines:
        for m_cusip in re.finditer(r'([A-Z0-9]{9})', line):
            cusip9 = m_cusip.group(1)
            cusip8 = cusip9[:8]
            if cusip8 not in target_cusips:
                continue

            after = line[m_cusip.end():]

            # Extract the two numbers after CUSIP
            nums = re.findall(r'(\d[\d,]*)', after)
            if len(nums) < 2:
                continue

            try:
                num1 = int(nums[0].replace(",", ""))
                num2 = int(nums[1].replace(",", ""))
            except ValueError:
                continue

            # Take the larger number as shares. For stocks with
            # price < $1000: shares > value/1000 always holds.
            shares = max(num1, num2)

            if shares > 0:
                holdings[cusip8] = holdings.get(cusip8, 0) + shares

    return holdings


def process_pre2013_quarter(year, quarter, target_cusips):
    """
    Download all 13F-HR filings for a quarter from EDGAR,
    parse each one, and aggregate holdings per CUSIP.

    Returns DataFrame: [cusip8, rdate, total_shares]
    """
    cache_file = CACHE_QUARTERLY / f"{year}q{quarter}_edgar.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    print(f"    Processing {year}Q{quarter} from EDGAR individual filings...")

    filing_urls = get_13f_filing_urls(year, quarter)
    print(f"    Found {len(filing_urls)} 13F-HR filings")

    if len(filing_urls) == 0:
        result = pd.DataFrame(columns=["cusip8", "rdate", "total_shares"])
        CACHE_QUARTERLY.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_file, index=False)
        return result

    # Download and parse each filing, accumulating total shares per CUSIP
    all_holdings = {}  # cusip8 -> total shares across all filers
    n_parsed = 0
    n_errors = 0

    for i, url in enumerate(filing_urls):
        if (i + 1) % 500 == 0:
            print(f"      {i+1}/{len(filing_urls)} filings processed "
                  f"({n_parsed} parsed, {n_errors} errors, "
                  f"{len(all_holdings)} CUSIPs found)")

        r = sec_get(url)
        if r is None or r.status_code != 200:
            n_errors += 1
            continue

        try:
            holdings = parse_13f_filing(r.text, target_cusips)
            n_parsed += 1
            for cusip8, shares in holdings.items():
                all_holdings[cusip8] = all_holdings.get(cusip8, 0) + shares
        except Exception:
            n_errors += 1

    # Build result DataFrame
    qe = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    canonical_rdate = pd.Timestamp(f"{year}-{qe[quarter]}")

    if all_holdings:
        result = pd.DataFrame([
            {"cusip8": c, "total_shares": s, "rdate": canonical_rdate}
            for c, s in all_holdings.items()
        ])
    else:
        result = pd.DataFrame(columns=["cusip8", "rdate", "total_shares"])

    CACHE_QUARTERLY.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_file, index=False)
    print(f"    {year}Q{quarter}: {len(result)} CUSIPs from "
          f"{n_parsed} filings ({n_errors} errors)")
    return result


# ── Step 4: Build quarterly ownership dataset ───────────────────────────────

def build_all_quarterly_holdings(target_cusips, needed_quarters):
    """
    Download and parse 13F data for all needed quarters.
    Uses structured data for 2013-Q2+, individual EDGAR filings for earlier.

    Returns concatenated DataFrame: [cusip8, rdate, total_shares]
    """
    print("\n  Fetching SEC structured 13F data URLs...")
    structured_urls = get_structured_zip_urls()
    print(f"  Found {len(structured_urls)} structured quarterly files")
    for yq in sorted(structured_urls.keys())[:3]:
        print(f"    {yq}")
    print(f"    ...")

    CACHE_QUARTERLY.mkdir(parents=True, exist_ok=True)

    all_holdings = []
    n_cached = 0
    n_download = 0

    for year, quarter in sorted(needed_quarters):
        # Check if cached
        cache_file_struct = CACHE_QUARTERLY / f"{year}q{quarter}_structured.parquet"
        cache_file_edgar = CACHE_QUARTERLY / f"{year}q{quarter}_edgar.parquet"
        if cache_file_struct.exists() or cache_file_edgar.exists():
            n_cached += 1
        else:
            n_download += 1

    print(f"  Quarters to process: {len(needed_quarters)} "
          f"({n_cached} cached, {n_download} to download)")

    for year, quarter in sorted(needed_quarters):
        if (year, quarter) in structured_urls:
            df = process_structured_quarter(
                year, quarter, structured_urls[(year, quarter)], target_cusips
            )
        else:
            df = process_pre2013_quarter(year, quarter, target_cusips)

        if len(df) > 0:
            all_holdings.append(df)

    if all_holdings:
        return pd.concat(all_holdings, ignore_index=True)
    else:
        return pd.DataFrame(columns=["cusip8", "rdate", "total_shares"])


# ── Step 5: Map CUSIPs to permnos and compute ownership ────────────────────

def compute_ownership(holdings, cusip_map, shrout):
    """
    Map 8-digit CUSIPs to permnos using CRSP msenames date ranges,
    then compute inst_own_pct = total_shares / (shrout * 1000).
    """
    if len(holdings) == 0:
        return pd.DataFrame(columns=["permno", "rdate", "inst_own_pct"])

    # Join on cusip8 = ncusip, with rdate falling within namedt-nameendt
    merged = holdings.merge(
        cusip_map, left_on="cusip8", right_on="ncusip", how="inner"
    )
    merged = merged[
        (merged["rdate"] >= merged["namedt"])
        & (merged["rdate"] <= merged["nameendt"])
    ]

    if len(merged) == 0:
        return pd.DataFrame(columns=["permno", "rdate", "inst_own_pct"])

    # Aggregate to (permno, rdate) — a firm may have multiple CUSIPs
    agg = merged.groupby(["permno", "rdate"])["total_shares"].sum().reset_index()

    # Merge with CRSP shrout (nearest month-end within 45 days)
    # Standardize datetime resolution to avoid merge errors
    agg["rdate"] = pd.to_datetime(agg["rdate"]).dt.as_unit("ns")
    agg = agg.sort_values("rdate")
    shrout_sorted = shrout.copy()
    shrout_sorted = shrout_sorted.rename(columns={"date": "crsp_date"})
    shrout_sorted["crsp_date"] = pd.to_datetime(
        shrout_sorted["crsp_date"]
    ).dt.as_unit("ns")
    shrout_sorted = shrout_sorted.sort_values("crsp_date")

    own = pd.merge_asof(
        agg, shrout_sorted,
        by="permno",
        left_on="rdate",
        right_on="crsp_date",
        direction="nearest",
        tolerance=pd.Timedelta(days=45)
    )

    # shrout is in thousands; total_shares is actual shares
    mask = own["shrout"].notna() & (own["shrout"] > 0)
    own["inst_own_pct"] = np.where(
        mask,
        own["total_shares"] / (own["shrout"] * 1000.0),
        np.nan
    )

    n_valid = mask.sum()
    print(f"  Ownership computed for {n_valid} (permno, rdate) pairs")
    print(f"  Unique permnos: {own[mask]['permno'].nunique()}")

    return own[["permno", "rdate", "inst_own_pct"]].copy()


# ── Step 6: Match ownership to events ───────────────────────────────────────

def match_ownership_to_events(sample, ownership):
    """
    For each (permno, event_date), find the most recent 13F quarter-end
    strictly before the event date (within MAX_LOOKBACK_DAYS).
    """
    results = []
    ownership = ownership.sort_values(["permno", "rdate"]).reset_index(drop=True)

    for _, row in sample.iterrows():
        permno = row["permno"]
        event_date = row["event_date"]

        base = dict(
            permno=permno,
            event_date=event_date,
            inst_own_pct=np.nan,
            retail_pct=np.nan,
            io_source="no_match",
            rdate_matched=pd.NaT,
        )

        firm_own = ownership[ownership["permno"] == permno]
        if len(firm_own) == 0:
            results.append(base)
            continue

        cutoff = event_date - pd.Timedelta(days=1)
        lookback = event_date - pd.Timedelta(days=MAX_LOOKBACK_DAYS)
        pre_own = firm_own[
            (firm_own["rdate"] <= cutoff)
            & (firm_own["rdate"] >= lookback)
            & (firm_own["inst_own_pct"].notna())
        ]

        if len(pre_own) == 0:
            results.append(base)
            continue

        latest = pre_own.sort_values("rdate").iloc[-1]
        iop = np.clip(latest["inst_own_pct"], 0.0, 1.0)

        base["inst_own_pct"] = iop
        base["retail_pct"] = 1.0 - iop
        base["io_source"] = "sec_edgar"
        base["rdate_matched"] = latest["rdate"]
        results.append(base)

    return pd.DataFrame(results)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== 18_build_institutional_ownership.py ===")

    # Step 1: Load sample and CRSP data
    print("\n[1] Loading sample and CRSP data...")
    sample, cusip_map, shrout, target_cusips = load_sample_and_cusips()

    # Determine needed quarters
    # For each event, we need the quarter ending before the event date
    # (within 180 days). Cover 6 months before earliest to latest event.
    earliest = sample["event_date"].min() - pd.Timedelta(days=MAX_LOOKBACK_DAYS)
    latest = sample["event_date"].max()

    start_year = earliest.year
    start_q = (earliest.month - 1) // 3 + 1
    end_year = latest.year
    end_q = (latest.month - 1) // 3 + 1

    needed_quarters = [
        (y, q)
        for y, q, _ in quarter_end_dates(start_year, start_q, end_year, end_q)
    ]
    print(f"\n  Need quarters: {needed_quarters[0]} to {needed_quarters[-1]}")
    print(f"  Total quarters: {len(needed_quarters)}")

    # Step 2: Download and parse 13F data
    print("\n[2] Downloading and parsing 13F data...")
    holdings = build_all_quarterly_holdings(target_cusips, needed_quarters)
    print(f"\n  Total holdings rows: {len(holdings):,}")

    # Step 3: Compute ownership ratios
    print("\n[3] Computing institutional ownership ratios...")
    ownership = compute_ownership(holdings, cusip_map, shrout)

    # Step 4: Match to events
    print("\n[4] Matching ownership to events...")
    result = match_ownership_to_events(sample, ownership)

    # Summary
    n_total = len(result)
    n_matched = result["inst_own_pct"].notna().sum()

    print(f"\n  Events processed    : {n_total:,}")
    print(f"  With ownership data : {n_matched:,} ({n_matched/n_total:.1%})")
    print(f"  Missing             : {n_total - n_matched:,}")

    if n_matched > 0:
        matched = result[result["inst_own_pct"].notna()]
        print(f"\n  inst_own_pct summary:")
        print(matched["inst_own_pct"].describe().to_string())
        print(f"\n  retail_pct summary:")
        print(matched["retail_pct"].describe().to_string())

        # Sanity check
        median_io = matched["inst_own_pct"].median()
        if median_io < 0.10:
            print(f"\n  WARNING: Median inst_own_pct = {median_io:.4f} "
                  f"seems low. Check data quality.")
        elif median_io > 0.95:
            print(f"\n  WARNING: Median inst_own_pct = {median_io:.4f} "
                  f"seems high. Check data quality.")

    # Save
    out_cols = ["permno", "event_date", "inst_own_pct", "retail_pct",
                "io_source", "rdate_matched"]
    result = result[out_cols].copy()
    result.to_parquet(OUT_PATH, index=False)
    print(f"\n  Saved: {OUT_PATH}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
