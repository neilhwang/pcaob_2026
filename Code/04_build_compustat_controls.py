"""
04_build_compustat_controls.py
==============================
Pull firm-year financial controls from Compustat via WRDS.

INPUT:  WRDS (comp.funda — Compustat annual fundamentals)
OUTPUT: Data/Processed/compustat_controls.parquet
        Columns: gvkey, fyear, size, leverage, roa, btm, loss,
                 sales_growth, current_ratio, state, sic2

VARIABLES:
    size         = log(total assets)          [log(at)]
    leverage     = total debt / total assets  [(dltt + dlc) / at]
    roa          = net income / total assets  [ni / at]
    btm          = book equity / market equity [(ceq) / (prcc_f * csho)]
    loss         = 1 if ni < 0
    sales_growth = (sale_t - sale_{t-1}) / sale_{t-1}
    current_ratio= current assets / current liabilities [act / lct]
    state        = HQ state (for merging with polarization measure)
    sic2         = 2-digit SIC code (for industry fixed effects)

SAMPLE FILTERS:
    - Domestic firms (fic == 'USA')
    - Non-financial, non-utility (SIC not 6000-6999, 4900-4999)
    - Positive total assets and sales
    - Fiscal years 1999-2023 (one extra year for sales_growth lag)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

OUT_FILE = Path(__file__).resolve().parent.parent / "Data/Processed/compustat_controls.parquet"
WRDS_USERNAME = ""   # leave blank to be prompted

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def connect_wrds():
    import wrds
    kwargs = {"wrds_username": WRDS_USERNAME} if WRDS_USERNAME else {}
    conn = wrds.Connection(**kwargs)
    log.info("Connected to WRDS.")
    return conn


def pull_compustat(conn) -> pd.DataFrame:
    log.info("Pulling Compustat annual fundamentals ...")
    df = conn.raw_sql("""
        SELECT gvkey, cik, datadate, fyear,
               at, dltt, dlc, ni, ceq, prcc_f, csho,
               sale, act, lct,
               state, sic, fic
        FROM comp.funda
        WHERE indfmt  = 'INDL'
          AND datafmt  = 'STD'
          AND popsrc   = 'D'
          AND consol   = 'C'
          AND fyear BETWEEN 1999 AND 2023
          AND fic = 'USA'
    """)
    log.info("Raw Compustat rows: %d", len(df))
    return df


def clean_and_construct(df: pd.DataFrame) -> pd.DataFrame:
    # Coerce numeric columns
    num_cols = ["at", "dltt", "dlc", "ni", "ceq", "prcc_f", "csho",
                "sale", "act", "lct"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop financials and utilities
    df["sic"] = pd.to_numeric(df["sic"], errors="coerce")
    df = df[~df["sic"].between(6000, 6999)]
    df = df[~df["sic"].between(4900, 4999)]

    # Require positive assets and sales
    df = df[(df["at"] > 0) & (df["sale"] > 0)]

    # 2-digit SIC
    df["sic2"] = (df["sic"] // 100).astype("Int64")

    # Core controls
    df["size"]     = np.log(df["at"])
    df["leverage"] = (df["dltt"].fillna(0) + df["dlc"].fillna(0)) / df["at"]
    df["roa"]      = df["ni"] / df["at"]
    df["loss"]     = (df["ni"] < 0).astype(int)
    mktcap         = df["prcc_f"] * df["csho"]
    df["btm"]      = np.where(mktcap > 0, df["ceq"] / mktcap, np.nan)
    df["current_ratio"] = np.where(
        df["lct"] > 0, df["act"] / df["lct"], np.nan
    )

    # Sales growth: requires prior-year sale within same gvkey
    df = df.sort_values(["gvkey", "fyear"])
    df["sale_lag"] = df.groupby("gvkey")["sale"].shift(1)
    df["sales_growth"] = np.where(
        df["sale_lag"] > 0,
        (df["sale"] - df["sale_lag"]) / df["sale_lag"],
        np.nan,
    )

    # Winsorize continuous controls at 1/99
    winsor_cols = ["size", "leverage", "roa", "btm", "sales_growth", "current_ratio"]
    for c in winsor_cols:
        lo = df[c].quantile(0.01)
        hi = df[c].quantile(0.99)
        df[c] = df[c].clip(lo, hi)

    # Keep fyear >= 2000 after computing lagged sales growth
    df = df[df["fyear"] >= 2000]

    out = df[[
        "gvkey", "cik", "fyear", "sic2", "state",
        "size", "leverage", "roa", "btm", "loss",
        "sales_growth", "current_ratio",
    ]].reset_index(drop=True)

    log.info("Clean Compustat rows: %d", len(out))
    log.info("Summary:\n%s", out[winsor_cols].describe().to_string())
    return out


def main() -> None:
    log.info("=== 04_build_compustat_controls.py  start ===")
    conn = connect_wrds()
    raw  = pull_compustat(conn)
    conn.close()
    log.info("WRDS connection closed.")

    clean = clean_and_construct(raw)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(OUT_FILE, index=False)
    log.info("Controls written: %s", OUT_FILE)
    log.info("=== done ===")


if __name__ == "__main__":
    main()
