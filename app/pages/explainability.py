import pandas as pd
import plotly.express as px
import streamlit as st

from model_utils import (feature_importance, load_engineered, load_model,
                         local_contributions, select_observation)
from utils import DashboardError, TEAL, plot

st.title("Model Explainability")
st.write("Feature importance shows which inputs the forecasting model relies on most.")
model = load_model()
importance = feature_importance(model).head(15).sort_values("Importance")
plot(px.bar(importance, x="Importance", y="Feature", orientation="h", height=530,
            title="Top 15 features · XGBoost global importance", color_discrete_sequence=[TEAL]))
st.caption("Importance is normalized average gain from the saved XGBoost model. Encoded categories are shown separately. This indicates model reliance, not the direction of an effect or a causal relationship.")

st.subheader("Explain one forecast")
st.write("See which inputs raise or lower a prediction relative to the model's reference value.")
row = select_observation(load_engineered(), "explanation")
if st.button("Explain this forecast", type="primary"):
    try:
        contributions, reference, prediction = local_contributions(model, row)
    except (DashboardError, ValueError, RuntimeError) as exc:
        st.info(f"A local explanation could not be calculated: {exc}")
    else:
        cols = st.columns(2)
        cols[0].metric("Model reference AQI", f"{reference:.1f}", border=True)
        cols[1].metric("Predicted AQI", f"{prediction:.1f}", border=True)
        # Keep the strongest contributions in both directions.
        positive = contributions[contributions["Contribution"] > 0].nlargest(6, "Contribution")
        negative = contributions[contributions["Contribution"] < 0].nsmallest(6, "Contribution")
        top = pd.concat([negative, positive]).sort_values("Contribution")
        top["Direction"] = top["Contribution"].map(lambda value: "Raises prediction" if value > 0 else "Lowers prediction")
        plot(px.bar(top, x="Contribution", y="Feature", color="Direction", orientation="h", height=500,
                    color_discrete_map={"Raises prediction": "#ce4545", "Lowers prediction": TEAL},
                    title="Top positive and negative SHAP contributions",
                    labels={"Contribution": "Contribution to predicted AQI"}))
        remaining = contributions["Contribution"].sum() - top["Contribution"].sum()
        st.caption(f"Reference {reference:.2f} + displayed contributions {top['Contribution'].sum():+.2f} + remaining inputs {remaining:+.2f} = forecast {prediction:.2f}.")
        st.caption("Tree SHAP contributions are computed with XGBoost on the saved pipeline's transformed inputs. They explain this model prediction and do not establish causality.")
