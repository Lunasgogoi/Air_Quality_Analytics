"""Shared file access, SQL analytics and small dashboard helpers."""

from contextlib import closing
from pathlib import Path
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "airlens.db"
ENGINEERED_PATH = ROOT / "data" / "processed" / "engineered_air_quality.csv"
ANOMALY_PATH = ROOT / "data" / "processed" / "anomaly_results.csv"
MODEL_PATH = ROOT / "models" / "aqi_forecasting_model.pkl"

CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
CATEGORY_COLORS = dict(zip(CATEGORIES, ["#21865b", "#84a332", "#e1b739", "#e58336", "#ce4545", "#822b51"]))
SEASONS = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]
TEAL = "#0f766e"


class DashboardError(Exception):
    """An actionable data or model problem that can be displayed in the app."""


def require_file(path):
    if not Path(path).is_file():
        raise DashboardError(f"Required file is missing: {Path(path).relative_to(ROOT) if Path(path).is_relative_to(ROOT) else path}")


def require_columns(frame, columns, source):
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise DashboardError(f"{source} is missing columns: {', '.join(missing)}.")


def validate_dates(frame, source):
    require_columns(frame, ["City", "Date"], source)
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    if frame["Date"].isna().any() or frame["City"].isna().any():
        raise DashboardError(f"{source} contains missing cities or invalid dates.")
    if frame.duplicated(["City", "Date"]).any():
        raise DashboardError(f"{source} contains duplicate city/date observations.")
    return frame.sort_values(["City", "Date"]).reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def query_database(sql, params=()):
    """Read the existing database without creating or modifying any objects."""
    require_file(DB_PATH)
    try:
        with closing(sqlite3.connect(DB_PATH.as_uri() + "?mode=ro", uri=True)) as con:
            return pd.read_sql_query(sql, con, params=params)
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        raise DashboardError(
            "Could not read data/airlens.db. The air_quality_validated view must exist; "
            "its setup SQL is in sql/analytics_queries.sql."
        ) from exc


@st.cache_data(ttl=300, show_spinner=False)
def load_validated():
    frame = query_database("SELECT * FROM air_quality_validated ORDER BY City, Date")
    require_columns(frame, ["Valid_AQI", "AQI_Outlier", *POLLUTANTS], "air_quality_validated")
    if not frame["Valid_AQI"].dropna().between(0, 500).all():
        raise DashboardError("air_quality_validated contains Valid_AQI outside 0–500. Check the existing view.")
    return validate_dates(frame, "air_quality_validated")


def city_summary():
    # Same validated coverage and reliable-city criteria as SQL sections 15–16.
    return query_database("""
        SELECT City, COUNT(*) AS total_records, COUNT(Valid_AQI) AS valid_days,
               100.0 * COUNT(Valid_AQI) / COUNT(*) AS coverage,
               AVG(Valid_AQI) AS average_aqi, MIN(Valid_AQI) AS minimum_aqi,
               MAX(Valid_AQI) AS maximum_aqi
        FROM air_quality_validated GROUP BY City ORDER BY average_aqi DESC
    """)


def reliable_summary():
    summary = city_summary()
    return summary[(summary["valid_days"] >= 500) & (summary["coverage"] >= 80)].copy()


def monthly_summary():
    frame = query_database("""
        SELECT City, strftime('%Y-%m-01', Date) AS Date,
               AVG(Valid_AQI) AS average_aqi, COUNT(Valid_AQI) AS valid_days
        FROM air_quality_validated GROUP BY City, strftime('%Y-%m', Date)
        ORDER BY City, Date
    """)
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame


def seasonal_summary():
    # Matches the four-season mapping in SQL section 20 and notebook 03.
    return query_database("""
        SELECT City,
            CASE
                WHEN CAST(strftime('%m', Date) AS INTEGER) IN (12, 1, 2) THEN 'Winter'
                WHEN CAST(strftime('%m', Date) AS INTEGER) IN (3, 4, 5) THEN 'Summer'
                WHEN CAST(strftime('%m', Date) AS INTEGER) IN (6, 7, 8, 9) THEN 'Monsoon'
                ELSE 'Post-Monsoon'
            END AS Season,
            AVG(Valid_AQI) AS average_aqi, COUNT(Valid_AQI) AS valid_days
        FROM air_quality_validated WHERE Valid_AQI IS NOT NULL
        GROUP BY City, Season ORDER BY City
    """)


def calendar_trends(frame):
    """Keep calendar gaps; each window includes today and the preceding 6/29 days."""
    parts = []
    for city, group in frame.groupby("City"):
        daily = group.set_index("Date")["Valid_AQI"].sort_index().asfreq("D")
        result = daily.to_frame()
        result["City"] = city
        for days in (7, 30):
            window = daily.rolling(f"{days}D", min_periods=1, closed="right")
            result[f"AQI_MA{days}"] = window.mean()
            result[f"Valid_Days{days}"] = window.count()
        parts.append(result.reset_index())
    columns = ["Date", "Valid_AQI", "City", "AQI_MA7", "Valid_Days7", "AQI_MA30", "Valid_Days30"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def prepare_anomalies(frame):
    """Validate the export schema shown in notebook 07, without inferring any flags."""
    require_columns(frame, ["City", "Date", "Valid_AQI", "Is_Anomaly"], "anomaly_results.csv")
    frame = validate_dates(frame, "anomaly_results.csv")
    flags = frame["Is_Anomaly"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}
    )
    if flags.isna().any():
        raise DashboardError("anomaly_results.csv has missing or invalid Is_Anomaly flags. Expected True/False or 1/0.")
    frame["Is_Anomaly"] = flags.astype(bool)
    values = pd.to_numeric(frame["Valid_AQI"], errors="coerce")
    frame["Valid_AQI"] = values.where(values.between(0, 500))
    return frame


@st.cache_data(ttl=300, show_spinner=False)
def load_anomalies():
    require_file(ANOMALY_PATH)
    try:
        return prepare_anomalies(pd.read_csv(ANOMALY_PATH))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DashboardError("Could not read data/processed/anomaly_results.csv.") from exc


def anomaly_summary(frame):
    summary = frame.groupby("City")["Is_Anomaly"].agg(Anomaly_Days="sum", Evaluated_Days="size").reset_index()
    summary["Anomaly_Percent"] = summary["Anomaly_Days"] / summary["Evaluated_Days"] * 100
    return summary.sort_values("Anomaly_Days", ascending=False)


def category_counts(frame):
    valid = pd.to_numeric(frame["Valid_AQI"], errors="coerce")
    categories = pd.cut(valid.where(valid.between(0, 500)), [0, 50, 100, 200, 300, 400, 500],
                        labels=CATEGORIES, include_lowest=True)
    return categories.value_counts(sort=False).reindex(CATEGORIES, fill_value=0).rename_axis("Category").reset_index(name="Observations")


def category_chart(frame, title):
    counts = category_counts(frame)
    if not counts["Observations"].sum():
        st.info("No validated AQI observations are available for this selection.")
        return
    plot(px.bar(counts, x="Category", y="Observations", color="Category", title=title,
                color_discrete_map=CATEGORY_COLORS, category_orders={"Category": CATEGORIES}), legend=False)


def plot(fig, legend=True):
    fig.update_layout(margin=dict(l=12, r=12, t=55, b=20), showlegend=legend,
                      legend_title_text="", font=dict(size=13))
    st.plotly_chart(fig, width="stretch")


def number(value, decimals=1):
    return "Unavailable" if pd.isna(value) else f"{value:,.{decimals}f}"


def default_cities(cities):
    preferred = [city for city in ["Delhi", "Bengaluru", "Hyderabad"] if city in cities]
    return preferred or list(cities)[:1]


def select_city(cities, key, label="City"):
    cities = sorted(cities)
    if not cities:
        st.info("No cities are available in this dataset.")
        st.stop()
    return st.selectbox(label, cities, index=cities.index("Delhi") if "Delhi" in cities else 0, key=key)
