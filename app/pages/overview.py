import plotly.express as px
import streamlit as st

from utils import TEAL, category_chart, load_validated, number, plot, reliable_summary

st.title("AirLens")
st.subheader("Intelligent Air Quality Analytics & Forecasting Platform")
st.write("Explore historical air quality across Indian cities, inspect the inputs that influence the forecasting model, and estimate next-day AQI.")

data = load_validated()
if data.empty:
    st.info("The validated database contains no observations.")
    st.stop()

st.caption(f"Historical dataset: {data['Date'].min():%d %b %Y} – {data['Date'].max():%d %b %Y}. AQI analytics use Valid_AQI (0–500).")
reliable = reliable_summary()
cols = st.columns(3)
cols[0].metric("Total cities", data["City"].nunique(), border=True)
cols[1].metric("Validated AQI observations", f"{data['Valid_AQI'].count():,}", border=True)
cols[2].metric("Average AQI", number(data["Valid_AQI"].mean()), border=True)

if not reliable.empty:
    high, low = reliable.iloc[0], reliable.iloc[-1]
    cols = st.columns(2)
    cols[0].metric("Highest average · reliable cities", high["City"], border=True)
    cols[0].caption(f"Average AQI: {high['average_aqi']:.1f}")
    cols[1].metric("Lowest average · reliable cities", low["City"], border=True)
    cols[1].caption(f"Average AQI: {low['average_aqi']:.1f}")
    plot(px.bar(reliable.sort_values("average_aqi"), x="average_aqi", y="City", orientation="h",
                title="Average AQI by reliable city", labels={"average_aqi": "Average AQI"},
                color_discrete_sequence=[TEAL], height=460))
else:
    st.info("No cities meet the reliability criteria for this dataset.")
st.caption("Reliable cities have at least 500 validated observations and at least 80% AQI coverage. Averages pool each city's available dates; reporting periods differ.")
category_chart(data, "AQI category distribution · all cities")
st.caption("Counts represent city-date observations. Missing and out-of-range AQI are excluded; categories are derived from Valid_AQI.")

st.subheader("Explore with AirLens")
for col, heading, description in zip(st.columns(4),
    ["What happened?", "Why?", "What happens next?", "Is anything unusual?"],
    ["Historical analytics", "Pollutant and model explainability analysis", "Next-day AQI forecasting", "Anomaly detection"]):
    with col:
        st.markdown(f"**{heading}**")
        st.write(description)
