import pandas as pd
import sqlite3

df = pd.read_csv("data/processed/air_quality_clean.csv")

con = sqlite3.connect("data/airlens.db")

df.to_sql(
    "air_quality",
    con,
    if_exists="replace",
    index=False
)

con.close()

print("Database created successfully")