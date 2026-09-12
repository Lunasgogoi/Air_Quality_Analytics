import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (POLLUTANTS, TEAL, category_chart, city_summary, default_cities,
                   load_validated, monthly_summary, number, plot, select_city)

st.title("City Analytics")
st.write("Inspect a city's AQI coverage, pollutant measurements and monthly patterns.")
data = load_validated()
city = select_city(data["City"].unique(), "analytics_city")
selected = data[data["City"] == city]
summary = city_summary()
stats = summary[summary["City"] == city].iloc[0]
for col, label, value in zip(st.columns(5),
    ["Average AQI", "Minimum AQI", "Maximum AQI", "Valid AQI days", "AQI coverage"],
    [number(stats["average_aqi"]), number(stats["minimum_aqi"]), number(stats["maximum_aqi"]),
     number(stats["valid_days"], 0), f"{stats['coverage']:.1f}%"]):
    col.metric(label, value, border=True)
st.caption(f"{selected['Date'].min():%d %b %Y} – {selected['Date'].max():%d %b %Y}. Coverage = validated AQI days ÷ {int(stats['total_records']):,} recorded city-days. Days absent from the source are not in this denominator.")

category_chart(selected, f"{city} · AQI category distribution")
st.subheader("Average pollutant levels")
pollutants = pd.DataFrame({"Pollutant": POLLUTANTS,
                          "Average": selected[POLLUTANTS].mean().values,
                          "Observations": selected[POLLUTANTS].count().values}).dropna(subset=["Average"])
for col, mask, title in zip(st.columns([3, 1]),
    [pollutants["Pollutant"] != "CO", pollutants["Pollutant"] == "CO"],
    ["Pollutants excluding CO", "Carbon monoxide"]):
    with col:
        values = pollutants[mask]
        if values.empty:
            st.info("No pollutant measurements available.")
        else:
            plot(px.bar(values, x="Pollutant", y="Average", title=title, hover_data=["Observations"],
                        labels={"Average": "Mean concentration (source units)"}, color_discrete_sequence=[TEAL]))
st.caption("Each mean uses that pollutant's available measurements, including days with missing AQI. CO is shown separately because its scale differs. Heights across pollutants do not measure relative health impact.")

monthly = monthly_summary()
monthly = monthly[monthly["City"] == city]
if monthly["average_aqi"].notna().any():
    monthly = monthly.set_index("Date").asfreq("MS").reset_index()
    plot(px.line(monthly, x="Date", y="average_aqi", title=f"{city} · monthly average AQI",
                 hover_data=["valid_days"], labels={"average_aqi": "Average AQI"}, color_discrete_sequence=[TEAL]))
else:
    st.info("No validated AQI is available for the monthly trend.")

st.subheader("Compare cities")
cities = sorted(data["City"].unique())
defaults = list(dict.fromkeys([city, *default_cities(cities)]))
comparison = st.multiselect("Cities to compare", cities, default=defaults, key="comparison_cities")
values = summary[summary["City"].isin(comparison)].dropna(subset=["average_aqi"])
if values.empty:
    st.info("Select cities with validated AQI to see a comparison.")
else:
    plot(px.bar(values, x="City", y="average_aqi", hover_data=["coverage", "valid_days"],
                title="Average AQI by city", labels={"average_aqi": "Average AQI", "coverage": "Coverage (%)"},
                color_discrete_sequence=[TEAL]))
st.caption("City averages use each city's available historical dates. Compare coverage and reporting periods alongside the averages.")
