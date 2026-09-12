import plotly.express as px
import streamlit as st

from utils import calendar_trends, default_cities, load_validated, plot

st.title("Pollution Trends")
st.write("Follow daily measurements and longer-term patterns across cities.")
data = load_validated()
cities = sorted(data["City"].unique())
selected = st.multiselect("Cities", cities, default=default_cities(cities), key="trend_cities")
if not selected:
    st.info("Select at least one city to view trends.")
    st.stop()
daily = calendar_trends(data[data["City"].isin(selected)])
if not daily["Valid_AQI"].notna().any():
    st.info("No validated AQI observations are available for these cities.")
    st.stop()
plot(px.line(daily, x="Date", y="Valid_AQI", color="City", title="Daily AQI",
             labels={"Valid_AQI": "AQI"}))
monthly = daily.set_index("Date").groupby("City")["Valid_AQI"].resample("MS").mean().reset_index()
plot(px.line(monthly, x="Date", y="Valid_AQI", color="City", title="Monthly average AQI",
             labels={"Valid_AQI": "Average AQI"}))
st.subheader("Calendar moving averages")
st.caption("Each window includes today and the preceding 6 or 29 calendar days. Missing AQI is not filled, and missing dates remain gaps. Averages use available readings; hover to see the count in each window.")
for days in (7, 30):
    plot(px.line(daily, x="Date", y=f"AQI_MA{days}", color="City",
                 hover_data={f"Valid_Days{days}": True}, title=f"{days}-day moving average",
                 labels={f"AQI_MA{days}": "Average AQI", f"Valid_Days{days}": "Valid days in window"}))
