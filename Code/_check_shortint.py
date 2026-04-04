import wrds
db = wrds.Connection(wrds_username='nhwang')
for table in ['comp.sec_shortint', 'crsp.msi_shortint', 'comp.shortint']:
    try:
        r = db.raw_sql(f'SELECT * FROM {table} LIMIT 5')
        print(f'{table}: OK')
        print(f'  columns: {r.columns.tolist()}')
    except Exception as e:
        msg = str(e).split('\n')[0]
        print(f'{table}: BLOCKED — {msg}')
db.close()
