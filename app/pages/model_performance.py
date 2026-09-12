import pandas as pd
import plotly.express as px
import streamlit as st

from model_utils import performance_table, test_evaluation
from utils import TEAL, number, plot

st.title("Model Performance")
st.subheader("Final Test Model: XGBoost")
evaluation = test_evaluation()
metrics = performance_table(evaluation)
model = metrics.iloc[0]
baseline = metrics.iloc[1]
for col, metric, decimals in zip(st.columns(3), ["MAE", "RMSE", "R²"], [2, 2, 3]):
    col.metric(metric, number(model[metric], decimals), border=True)
if pd.notna(baseline["MAE"]) and baseline["MAE"] > 0:
    improvement = (baseline["MAE"] - model["MAE"]) / baseline["MAE"] * 100
    if improvement >= 0:
        st.success(f"Final model achieved approximately {improvement:.0f}% lower MAE than the persistence baseline.")
    else:
        st.info(f"The saved model has {-improvement:.1f}% higher MAE than the persistence baseline on these files.")
st.caption(f"Calculated from the saved model and the original test split: observation dates {evaluation['Date'].min():%d %b %Y} – {evaluation['Date'].max():%d %b %Y}. No model is retrained.")
st.dataframe(metrics.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.3f}"}), hide_index=True, width="stretch")
st.caption(f"Matches notebook 06: XGBoost uses {int(model['Observations']):,} labelled test rows; persistence uses {int(baseline['Observations']):,} rows with both current and next-day AQI. The model imputes missing inputs; the baseline cannot predict without current AQI. The reported percentage uses these original evaluation sets.")
st.caption("MAE is average absolute error in AQI points; RMSE gives larger errors more weight. Lower is better for both. Higher R² indicates a better fit.")
errors = metrics.melt(id_vars="Model", value_vars=["MAE", "RMSE"], var_name="Metric", value_name="Error")
plot(px.bar(errors, x="Metric", y="Error", color="Model", barmode="group", title="Model comparison · test error"))

scatter = px.scatter(evaluation, x="Target_AQI", y="Predicted_AQI", color="City", opacity=0.55,
                     hover_data=["Forecast_Date"], title="Actual vs Predicted AQI · test observations",
                     labels={"Target_AQI": "Actual next-day AQI", "Predicted_AQI": "Predicted next-day AQI"})
scatter.add_shape(type="line", x0=0, y0=0, x1=500, y1=500, line={"dash": "dash", "color": "#64748b"})
plot(scatter)

delhi = evaluation[evaluation["City"] == "Delhi"].copy()
if not delhi.empty:
    delhi = delhi.set_index("Forecast_Date")[["Target_AQI", "Predicted_AQI"]].asfreq("D").reset_index()
    delhi = delhi.rename(columns={"Target_AQI": "Actual AQI", "Predicted_AQI": "Predicted AQI"})
    plot(px.line(delhi, x="Forecast_Date", y=["Actual AQI", "Predicted AQI"],
                 title="Delhi · actual vs predicted next-day AQI", labels={"Forecast_Date": "Forecast date", "value": "AQI"}))
else:
    st.info("Delhi observations are not available in the test split.")

city_errors = evaluation.groupby("City").agg(MAE=("Absolute_Error", "mean"), Observations=("Absolute_Error", "count")).reset_index()
plot(px.bar(city_errors.sort_values("MAE"), x="MAE", y="City", orientation="h", height=460,
            title="City-wise test MAE", hover_data=["Observations"], color_discrete_sequence=[TEAL]))
