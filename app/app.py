"""Run from the project root: streamlit run app/app.py."""

import streamlit as st

from utils import DashboardError

st.set_page_config(page_title="AirLens", page_icon="🌍", layout="wide")

with st.sidebar:
    st.title("🌍 AirLens")
    st.caption("Intelligent Air Quality Analytics & Forecasting Platform")
    st.caption("Historical observations · India")
    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

page = st.navigation([
    st.Page("pages/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("pages/city_analytics.py", title="City Analytics", icon=":material/location_city:"),
    st.Page("pages/pollution_trends.py", title="Pollution Trends", icon=":material/show_chart:"),
    st.Page("pages/seasonal_analysis.py", title="Seasonal Analysis", icon=":material/calendar_month:"),
    st.Page("pages/forecast.py", title="AQI Forecast", icon=":material/online_prediction:"),
    st.Page("pages/explainability.py", title="Model Explainability", icon=":material/search:"),
    st.Page("pages/anomaly_detection.py", title="Anomaly Detection", icon=":material/troubleshoot:"),
    st.Page("pages/model_performance.py", title="Model Performance", icon=":material/assessment:"),
], expanded=True)

try:
    page.run()
except DashboardError as exc:
    st.error(str(exc))
    st.caption("Check the project files, then use Refresh data in the sidebar.")
