# export_gp_prices.py
import os, pyodbc, pandas as pd

SERVER   = r"DYNAMICSGP"   # or "SERVER\\INSTANCE"
DATABASE = "CDI"
TRUSTED  = True            # False => set UID/PWD
UID, PWD = "", ""
OUT_CSV  = "chemicals.csv"  # date,symbol,price

# Load last exported date to pull only new months
last_date = None
if os.path.exists(OUT_CSV):
    try:
        last_date = pd.read_csv(OUT_CSV, parse_dates=["date"])["date"].max().date()
    except Exception:
        last_date = None

where_incremental = ""
params = []
if last_date:
    where_incremental = "AND MonthEnd > ?"
    params.append(str(last_date))

SQL = fr"""
WITH s AS (
    SELECT
        l.ITEMNMBR, l.LOCNCODE,
        h.DOCDATE AS DocDate, EOMONTH(h.DOCDATE) AS MonthEnd,
        l.UOFM,
        CAST(l.QTYSHPPD AS decimal(18,6)) AS QtyOnLine,
        CAST(l.XTNDPRCE AS decimal(18,6)) AS ExtPrice
    FROM dbo.SOP30300 l
    JOIN dbo.SOP30200 h
      ON h.SOPTYPE = l.SOPTYPE AND h.SOPNUMBE = l.SOPNUMBE
    WHERE l.SOPTYPE = 3 AND l.QTYSHPPD > 0 AND h.DOCDATE >= '2015-01-01'
),
s2 AS (
    SELECT s.*,
        CASE
            WHEN s.UOFM LIKE '%250%'       THEN 250.0
            WHEN s.UOFM LIKE '%180%'       THEN 180.0
            WHEN s.UOFM LIKE '%55%'        THEN 55.0
            WHEN s.UOFM LIKE '%5%'         THEN 5.0
            WHEN RIGHT(s.ITEMNMBR,2)='02'  THEN 180.0
            WHEN RIGHT(s.ITEMNMBR,3)='250' THEN 250.0
            ELSE NULL
        END AS GalPerUnit
    FROM s
),
priced AS (
    SELECT
        ITEMNMBR, MonthEnd,
        CAST(ExtPrice / NULLIF(QtyOnLine * GalPerUnit, 0) AS decimal(18,6)) AS Price_per_Gallon
    FROM s2
)
SELECT MonthEnd AS date, ITEMNMBR AS symbol, AVG(Price_per_Gallon) AS price
FROM priced
WHERE Price_per_Gallon IS NOT NULL {where_incremental}
GROUP BY MonthEnd, ITEMNMBR
ORDER BY MonthEnd, ITEMNMBR;
"""

if TRUSTED:
    cn = pyodbc.connect(
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;Encrypt=no"
    )
else:
    cn = pyodbc.connect(
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={SERVER};DATABASE={DATABASE};UID={UID};PWD={PWD};Encrypt=no"
    )

new_df = pd.read_sql(SQL, cn, params=params)
new_df["date"] = pd.to_datetime(new_df["date"]).dt.date

if os.path.exists(OUT_CSV):
    old = pd.read_csv(OUT_CSV, parse_dates=["date"])
    merged = (pd.concat([old, new_df], ignore_index=True)
                .drop_duplicates(subset=["date","symbol"])
                .sort_values(["date","symbol"]))
else:
    merged = new_df

merged.to_csv(OUT_CSV, index=False)
print(f"✅ chemicals.csv rows: {len(merged):,} (added {len(new_df):,})")
