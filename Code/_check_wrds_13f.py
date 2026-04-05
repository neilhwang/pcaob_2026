"""
Quick check: which 13F / institutional ownership tables are accessible on WRDS?
Run: python Code/_check_wrds_13f.py
"""

import wrds

db = wrds.Connection()

tables_to_check = [
    # Thomson-Reuters S34 (legacy 13F)
    ("tfn", "s34"),
    ("tfn", "s34type1"),
    ("tfn", "s34type2"),
    # WRDS ownership research tables
    ("wrdsapps", "ownership_13f"),
    ("wrdsapps", "ownership_stddate"),
    # Thomson-Reuters Mutual Fund (s12)
    ("tfn", "s12"),
    ("tfn", "s12type1"),
    # Newer WRDS 13F
    ("wrdssec", "ownership_13f"),
    ("wrdssec", "institutional_13f"),
    # SEC 13F via WRDS
    ("sec", "holding_13f"),
    # WRDS sample tables (often available on basic)
    ("wrdsapps", "ibcrsphist"),
    ("fssamp", "fssamp"),
]

print(f"{'Schema':<20} {'Table':<25} {'Accessible?'}")
print("-" * 60)

for schema, table in tables_to_check:
    try:
        result = db.raw_sql(
            f"SELECT COUNT(*) AS n FROM {schema}.{table} LIMIT 1"
        )
        n = result["n"].iloc[0]
        print(f"{schema:<20} {table:<25} YES  (rows sampled: {n})")
    except Exception as e:
        msg = str(e).split("\n")[0][:60]
        print(f"{schema:<20} {table:<25} NO   ({msg})")

# Also list all schemas we have access to that might contain ownership data
print("\n\nSearching for any tables with '13f' or 'ownership' in the name...")
try:
    all_libs = db.raw_sql(
        "SELECT DISTINCT table_schema, table_name "
        "FROM information_schema.tables "
        "WHERE lower(table_name) LIKE '%13f%' "
        "   OR lower(table_name) LIKE '%ownership%' "
        "   OR lower(table_name) LIKE '%holding%' "
        "ORDER BY table_schema, table_name"
    )
    if len(all_libs):
        for _, row in all_libs.iterrows():
            print(f"  {row['table_schema']}.{row['table_name']}")
    else:
        print("  (none found)")
except Exception as e:
    print(f"  Schema search failed: {e}")

db.close()
