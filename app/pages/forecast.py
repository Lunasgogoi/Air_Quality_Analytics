import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model_utils import (aqi_category, load_engineered, load_model, model_input,
                         predict_rows, select_observation)
from utils import CATEGORIES, CATEGORY_COLORS, number, plot

st.title("AQI Forecast")
st.write("Estimate tomorrow's AQI from an existing observation and its saved historical features.")
model = load_model()
data = load_engineered()
row = select_observation(data, "forecast")
observation = row.iloc[0]
forecast_date = observation["Date"] + pd.Timedelta(days=1)
st.caption(f"{observation['City']} · observation: {observation['Date']:%d %b %Y} · forecast for: {forecast_date:%d %b %Y}")
st.info("This is a historical forecast demonstration. ‘Tomorrow’ means the calendar day after the selected observation; the dataset is not a live feed.")

predicted = predict_rows(model, row)[0]
current = observation["Valid_AQI"]
difference = predicted - current
category = aqi_category(predicted)
cols = st.columns(3)
cols[0].metric("Current AQI", number(current), border=True)
cols[1].metric("Predicted Tomorrow AQI", f"{predicted:.1f}", border=True)
cols[2].metric("Predicted change", f"{difference:+.1f}" if pd.notna(difference) else "Unavailable", border=True)
st.subheader(f"Predicted AQI category: {category}")

if 0 <= predicted <= 500:
    bounds = [0, 50, 100, 200, 300, 400, 500]
    gauge = go.Figure(go.Indicator(mode="gauge+number", value=predicted,
        number={"valueformat": ".1f"}, title={"text": f"AQI forecast · {forecast_date:%d %b %Y}"},
        gauge={"axis": {"range": [0, 500]}, "bar": {"color": "#172b36", "thickness": 0.25},
               "steps": [{"range": [bounds[i], bounds[i + 1]], "color": CATEGORY_COLORS[name]}
                         for i, name in enumerate(CATEGORIES)]}))
    gauge.update_layout(height=330)
    plot(gauge)
else:
    st.warning("The model output is outside 0–500, so no standard AQI category or gauge is assigned. The prediction is displayed without clipping.")
st.caption("Good: 0–50 · Satisfactory: >50–100 · Moderate: >100–200 · Poor: >200–300 · Very Poor: >300–400 · Severe: >400–500. Categories use the unrounded prediction.")

inputs = model_input(model, row)
if inputs.isna().any(axis=None):
    st.caption("Some inputs are missing. The saved pipeline applies its fitted imputation; no new values or features are fitted here.")
with st.expander("View the model inputs"):
    st.write(f"These {len(inputs.columns)} inputs follow the saved pipeline's training column order.")
    st.dataframe(inputs.T.rename(columns={inputs.index[0]: "Value"}).astype(str), width="stretch")
    st.caption("Lag and rolling features come directly from the engineered dataset. Target_AQI and Date are excluded from model inputs.")
