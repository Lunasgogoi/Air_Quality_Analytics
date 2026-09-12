import plotly.express as px
import streamlit as st

from utils import ANOMALY_PATH, TEAL, anomaly_summary, load_anomalies, plot, select_city

st.title("Anomaly Detection")
st.write("An anomaly represents a pollutant pattern that differs significantly from that city's historical behaviour. It does not automatically mean the day was dangerous.")

if not ANOMALY_PATH.is_file():
    st.info("Anomaly results are unavailable. Place the final export at data/processed/anomaly_results.csv, then select Refresh data in the sidebar.")
    st.caption("The dashboard reads saved city-wise Isolation Forest results and never fits an anomaly detector. The requested final method uses PM2.5, NO2, CO, SO2 and O3, excluding PM10.")
    st.stop()

data = load_anomalies()
if data.empty:
    st.info("The anomaly export has no evaluated observations.")
    st.stop()
city = select_city(data["City"].unique(), "anomaly_city")
selected = data[data["City"] == city].copy()
anomalies = selected[selected["Is_Anomaly"]]
cols = st.columns(3)
cols[0].metric("Total anomaly days", len(anomalies), border=True)
cols[1].metric("Anomaly percentage", f"{len(anomalies) / len(selected) * 100:.1f}%", border=True)
cols[2].metric("Evaluated days", f"{len(selected):,}", border=True)
st.caption(f"{selected['Date'].min():%d %b %Y} – {selected['Date'].max():%d %b %Y}. Percentage uses only days present in the saved anomaly export, including days with missing AQI.")

if anomalies.empty:
    st.info(f"No anomalies were detected for {city} in the saved results.")
else:
    st.subheader("Anomaly observations")
    st.dataframe(anomalies.sort_values("Date", ascending=False), hide_index=True, width="stretch")

if selected["Valid_AQI"].notna().any():
    daily = selected.set_index("Date")[["Valid_AQI"]].asfreq("D").reset_index()
    timeline = px.line(daily, x="Date", y="Valid_AQI", title=f"{city} · AQI with detected anomalies",
                       labels={"Valid_AQI": "AQI"}, color_discrete_sequence=[TEAL])
    visible = anomalies.dropna(subset=["Valid_AQI"])
    if not visible.empty:
        timeline.add_scatter(x=visible["Date"], y=visible["Valid_AQI"], mode="markers", name="Anomaly",
                             marker={"color": "#ce4545", "size": 10, "symbol": "diamond"})
    missing = anomalies[anomalies["Valid_AQI"].isna()]
    for date in missing["Date"]:
        timeline.add_shape(type="line", x0=date, x1=date, y0=0, y1=1, yref="paper",
                           line={"color": "#ce4545", "dash": "dot", "width": 1})
    plot(timeline)
    if not missing.empty:
        st.caption(f"{len(missing)} anomaly day(s) have no validated AQI. Dotted vertical lines mark their dates; the table retains their flags.")
else:
    st.info("No validated AQI values are available for this timeline. Anomaly counts and table still include all saved flags.")

summary = anomaly_summary(data)
plot(px.bar(summary, x="City", y="Anomaly_Days", title="Detected Pollution Anomalies by City",
            hover_data=["Evaluated_Days", "Anomaly_Percent"], labels={"Anomaly_Days": "Anomaly days"},
            color_discrete_sequence=[TEAL]))
st.caption("Cities with zero detected anomalies remain in the comparison. Counts depend on how many days were evaluated; a missing city has no exported results.")
