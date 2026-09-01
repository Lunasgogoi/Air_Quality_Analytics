import sqlite3
import pandas as pd

con = sqlite3.connect("data/airlens.db")

query = """
SELECT *
FROM air_quality
LIMIT 5;
"""

result = pd.read_sql_query(query, con)

print(result)

con.close()