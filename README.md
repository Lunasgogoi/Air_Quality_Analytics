# AirLens

**Intelligent Air Quality Analytics & Forecasting Platform**

A Streamlit + Plotly dashboard for the existing Indian city air quality project.
It reads the SQLite analytics view and processed artifacts, and runs inference with
the saved sklearn/XGBoost pipeline. It does not train models or modify the data.

## Run the dashboard

From the project root, activate an environment with the dependencies installed:

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

This workspace also has a tested Python 3.14 environment named `.venv314`.
The original `.venv` could not start because its Python 3.13 executable was
inaccessible; it has been preserved. On Windows, run the dashboard directly with:

```powershell
.\.venv314\Scripts\python.exe -m streamlit run app/app.py
```

Or activate it and use the requested entry point:

```powershell
.\.venv314\Scripts\Activate.ps1
streamlit run app/app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.
Paths are resolved from the application files, with no hardcoded Windows paths.

## Deploy on Streamlit Community Cloud

Use repository `Lunasgogoi/Air_Quality_Analytics`, branch `main`, and main file path
`app/app.py`. Select **Python 3.13** in Advanced settings. The runtime package pins
require Python 3.12 or newer; Python 3.13 matches the original training environment.

The app has a small UTF-8 dependency file at `app/requirements.txt`. Community
Cloud [checks the entrypoint directory first](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies),
so it installs this file instead of the root notebook environment. The root
`requirements.txt` is preserved, including notebook tools and Windows-only
`pywinpty`, which cannot be installed on the Linux deployment host. Keep the saved
model's scikit-learn and XGBoost versions pinned to avoid pickle incompatibilities.

Push these runtime artifacts along with the application code:

- `data/airlens.db` (already tracked)
- `data/processed/engineered_air_quality.csv`
- `models/aqi_forecasting_model.pkl`
- `data/processed/anomaly_results.csv` when the final export is available

The `.gitignore` exceptions allow only these processed/model artifacts; raw data
and other generated files remain excluded. Missing model/data files will prevent
the corresponding dashboard pages from working after dependency installation.

After committing and pushing the deployment changes, let Community Cloud rebuild
the app. If installation still fails, inspect the first `ERROR` above the final
"installer returned a non-zero exit code" message; the final message alone does
not identify the failed dependency.

## Dashboard pages

| Page | Contents |
| --- | --- |
| Overview | Dataset metrics, reliable-city ranking, validated AQI categories |
| City Analytics | AQI coverage, pollutant means, monthly trends, city comparisons |
| Pollution Trends | Daily/monthly AQI and calendar-based 7/30-day moving averages |
| Seasonal Analysis | Seasonal comparison, city/season heatmap, highest-average season |
| AQI Forecast | City/date selection, next-day estimate, category and AQI gauge |
| Model Explainability | Saved XGBoost feature importance and optional local SHAP contributions |
| Anomaly Detection | Saved flags, city counts, anomaly table and marked AQI timeline |
| Model Performance | Original test metrics, baseline comparison, prediction plots and city MAE |

## Required existing artifacts

| Path | Use |
| --- | --- |
| `data/airlens.db` | Existing `air_quality_validated` view for historical analytics |
| `data/processed/engineered_air_quality.csv` | Original forecasting inputs and evaluation targets |
| `models/aqi_forecasting_model.pkl` | Complete fitted preprocessing + XGBoost pipeline |
| `data/processed/anomaly_results.csv` | Completed city-wise anomaly results |

Missing or incompatible artifacts produce an explanatory message on the relevant
page. Database connections are read-only. Dataset/query caches expire after five
minutes; the pipeline uses `st.cache_resource`. **Refresh data** clears both caches
after replacing an artifact or restoring a missing export.

**Anomaly export currently missing:** `anomaly_results.csv` was not present during
implementation. The saved notebook 07 contains an older run that includes
`Valid_AQI` and `PM10`, and it has no CSV export cell. Those results were not used
as a substitute for the requested final method, which uses PM2.5, NO2, CO, SO2 and
O3. Restore the already-completed final export at the path above; the dashboard
does not fit or rerun Isolation Forest.

The anomaly loader expects the result fields shown in notebook 07: `City`, `Date`,
`Valid_AQI` and `Is_Anomaly`. `Anomaly` and `Anomaly_Score` are optional and appear
in the table when present. Boolean flags may be True/False or 1/0. Missing or
unrecognized flags and duplicate city/date records are rejected. Zero-anomaly
cities remain visible, and flagged days without AQI remain in counts and tables.

## How the dashboard reuses the project

- Historical AQI analytics use `Valid_AQI` from the existing SQLite view, including
  the SQL notebook's reliability rule: at least 500 validated observations and at
  least 80% coverage. Coverage is valid observations divided by recorded city-days.
- AQI categories are derived from values in 0–500, rather than the source bucket.
  City comparisons pool each city's available dates; coverage and time spans differ.
- Trend windows retain missing dates and missing AQI. A 7-day average includes the
  selected date and preceding six calendar days, using only available readings.
  Hover counts reveal sparse windows. These display calculations do not change
  the model's saved lag/rolling features.
- Forecast rows come directly from `engineered_air_quality.csv`. The 28 input
  names and their order come from `pipeline.feature_names_in_`; `Date`, raw `AQI`
  and `Target_AQI` are excluded. The saved pipeline handles imputation, scaling and
  encoding. No preprocessing is fitted or reconstructed.
- Forecasts are historical demonstrations: the latest engineered observation is
  30 June 2020. “Tomorrow” is the day after the selected observation, not the
  current calendar day. Continuous predictions use category bounds of ≤50,
  ≤100, ≤200, ≤300, ≤400 and ≤500. Out-of-range outputs are displayed without
  clipping and receive no standard category.
- Global importance uses the saved model's normalized gain. Local explanations
  use [XGBoost's native SHAP contributions](https://xgboost.readthedocs.io/en/stable/prediction.html)
  after the existing preprocessor. No separate `shap` dependency is needed for
  the dashboard. Contributions describe model behavior and do not establish causality.
- Performance is recalculated by inference on the original notebook 06 split,
  using observation dates from 1 January 2020. The XGBoost result reproduces MAE
  **14.32**, RMSE **23.03**, R² **0.868** on **2,170** rows. Persistence reproduces
  **17.25**, **29.36**, **0.785** on **2,163** rows with current AQI available. The
  approximately 17% MAE reduction uses those original evaluation sets. The app
  displays actual computed results rather than hardcoding these metrics.

## Structure

- `data/raw/`: Source datasets
- `data/processed/`: Cleaned and transformed datasets
- `notebooks/`: Analysis workflow notebooks
- `src/`: Reusable Python source code
- `sql/`: SQL queries and database scripts
- `models/`: Trained model artifacts
- `app/app.py`: Page configuration, sidebar navigation and friendly error handling
- `app/utils.py`: Read-only SQL/data loading, calendar trends and shared chart helpers
- `app/model_utils.py`: Cached pipeline loading, inference, evaluation and explanations
- `app/pages/`: One small Streamlit script for each dashboard page
- `tests/`: Automated tests

## Notebook workflow

1. Data understanding
2. Data cleaning
3. Exploratory data analysis
4. Feature engineering
5. Model training
6. Model evaluation
7. Explainability and anomaly analysis

## Verification

```bash
python -m compileall -q app tests
python -m unittest discover -s tests -p "test_*.py" -v
```

The tests use real artifacts for pipeline/schema checks, the original performance
metrics, all city filters, forecast date changes and Streamlit page navigation.
Small in-memory/temporary fixtures test calendar gaps, category boundaries, missing
files and zero/missing-AQI anomaly cases; they never replace project datasets.

All 16 tests passed during implementation. Streamlit started successfully and its
health endpoint returned HTTP 200. Browser visual QA was unavailable because no
browser was connected to the session.
