"""Inference and explanations using the saved pipeline. Nothing is fitted here."""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import (CATEGORIES, ENGINEERED_PATH, MODEL_PATH, DashboardError,
                   require_columns, require_file, select_city, validate_dates)

# Original feature-date split from notebooks/06_model_evaluation.ipynb.
TEST_START = pd.Timestamp("2020-01-01")


@st.cache_resource(show_spinner="Loading the saved forecasting pipeline…")
def load_model():
    require_file(MODEL_PATH)
    try:
        model = joblib.load(MODEL_PATH)
        if not {"preprocessor", "model"}.issubset(model.named_steps):
            raise ValueError("Expected the saved preprocessor + model pipeline.")
        training_features(model)
        return model
    except Exception as exc:
        raise DashboardError(
            "Could not load models/aqi_forecasting_model.pkl. Install the model package "
            f"versions in requirements.txt. Details: {exc}"
        ) from exc


def training_features(model):
    names = list(model.feature_names_in_)
    if not names or len(names) != len(set(names)) or {"Date", "AQI", "Target_AQI", "AQI_Bucket"}.intersection(names):
        raise DashboardError("The saved pipeline has an unexpected forecasting input schema.")
    if names != list(model.named_steps["preprocessor"].feature_names_in_):
        raise DashboardError("Pipeline and preprocessor input columns do not match.")
    return names


@st.cache_data(ttl=300, show_spinner=False)
def load_engineered():
    require_file(ENGINEERED_PATH)
    try:
        frame = pd.read_csv(ENGINEERED_PATH)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DashboardError("Could not read data/processed/engineered_air_quality.csv.") from exc
    require_columns(frame, ["City", "Date", "Valid_AQI"], ENGINEERED_PATH.name)
    return validate_dates(frame, ENGINEERED_PATH.name)


def model_input(model, rows):
    """Select the exact saved training columns, in order; leave imputation to the pipeline."""
    features = training_features(model)
    require_columns(rows, features, ENGINEERED_PATH.name)
    inputs = rows.loc[:, features].copy()
    if np.isinf(inputs.select_dtypes(include="number").to_numpy(dtype=float)).any():
        raise DashboardError("The selected model inputs contain infinite values.")
    return inputs


def predict_rows(model, rows):
    try:
        predictions = np.asarray(model.predict(model_input(model, rows)), dtype=float)
    except DashboardError:
        raise
    except Exception as exc:
        raise DashboardError(f"The saved pipeline could not predict these observations: {exc}") from exc
    if not np.isfinite(predictions).all():
        raise DashboardError("The saved pipeline returned a non-finite prediction.")
    return predictions


def aqi_category(value):
    if not np.isfinite(value) or value < 0 or value > 500:
        return "Outside validated AQI range"
    for upper, category in zip([50, 100, 200, 300, 400, 500], CATEGORIES):
        if value <= upper:
            return category


def select_observation(frame, key):
    city = select_city(frame["City"].unique(), f"{key}_city")
    observations = frame[frame["City"] == city].sort_values("Date", ascending=False)
    dates = observations["Date"].tolist()
    date = st.selectbox("Observation date (latest first)", dates,
                        format_func=lambda value: value.strftime("%d %b %Y"), key=f"{key}_date_{city}")
    return observations.loc[observations["Date"] == date].iloc[[0]].copy()


def feature_importance(model):
    names = model.named_steps["preprocessor"].get_feature_names_out()
    values = model.named_steps["model"].feature_importances_
    return pd.DataFrame({"Feature": [name.split("__", 1)[-1] for name in names],
                         "Importance": values}).sort_values("Importance", ascending=False)


def local_contributions(model, row):
    """Use XGBoost's native Tree SHAP on the pipeline's own transformed inputs."""
    preprocessor = model.named_steps["preprocessor"]
    transformed = preprocessor.transform(model_input(model, row))
    contributions = model.named_steps["model"].get_booster().predict(
        xgb.DMatrix(transformed), pred_contribs=True
    )[0]
    prediction = predict_rows(model, row)[0]
    if not np.isclose(contributions.sum(), prediction, atol=0.001):
        raise DashboardError("Feature contributions do not reconcile to this prediction.")
    names = preprocessor.get_feature_names_out()
    return pd.DataFrame({"Feature": [name.split("__", 1)[-1] for name in names],
                         "Contribution": contributions[:-1]}), float(contributions[-1]), prediction


@st.cache_data(ttl=300, show_spinner="Evaluating the saved model on the original test split…")
def test_evaluation():
    data = load_engineered()
    require_columns(data, ["Target_AQI"], ENGINEERED_PATH.name)
    test = data[(data["Date"] >= TEST_START) & data["Target_AQI"].notna()].copy()
    if test.empty:
        raise DashboardError("No labelled observations are available in the original test split (from 1 January 2020).")
    if not test["Target_AQI"].between(0, 500).all():
        raise DashboardError("The test data contains Target_AQI outside the validated 0–500 range.")
    test["Predicted_AQI"] = predict_rows(load_model(), test)
    test["Forecast_Date"] = test["Date"] + pd.Timedelta(days=1)
    test["Absolute_Error"] = (test["Target_AQI"] - test["Predicted_AQI"]).abs()
    return test


def performance_table(evaluation):
    baseline = evaluation.dropna(subset=["Valid_AQI", "Target_AQI"])
    rows = []
    for label, data, prediction in [("XGBoost", evaluation, "Predicted_AQI"),
                                     ("Persistence baseline", baseline, "Valid_AQI")]:
        count = len(data)
        rows.append({"Model": label, "Observations": count,
                     "MAE": mean_absolute_error(data["Target_AQI"], data[prediction]) if count else np.nan,
                     "RMSE": np.sqrt(mean_squared_error(data["Target_AQI"], data[prediction])) if count else np.nan,
                     "R²": r2_score(data["Target_AQI"], data[prediction]) if count > 1 else np.nan})
    return pd.DataFrame(rows)
