"""Run: python -m unittest discover -s tests -p 'test_*.py' -v.

Small in-memory fixtures cover edge cases only; dashboard integration uses real artifacts.
"""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import model_utils
import utils


class DataSemanticsTests(unittest.TestCase):
    def test_calendar_windows_do_not_join_distant_observations(self):
        frame = pd.DataFrame({"City": ["Test city"] * 4,
                              "Date": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-08", "2020-02-01"]),
                              "Valid_AQI": [100.0, np.nan, 200.0, 50.0]})
        daily = utils.calendar_trends(frame).set_index("Date")
        self.assertEqual(len(daily), 32)
        self.assertTrue(pd.isna(daily.loc["2020-01-02", "Valid_AQI"]))
        self.assertEqual(daily.loc["2020-01-08", "AQI_MA7"], 200)
        self.assertEqual(daily.loc["2020-01-08", "Valid_Days7"], 1)
        self.assertEqual(daily.loc["2020-01-08", "AQI_MA30"], 150)
        self.assertEqual(daily.loc["2020-02-01", "AQI_MA30"], 125)
        self.assertTrue(pd.isna(daily.loc["2020-01-20", "AQI_MA7"]))

    def test_aqi_categories_exclude_invalid_readings(self):
        data = pd.DataFrame({"Valid_AQI": [-1, 0, 50, 50.1, 100, 101, 201, 301, 401, 500, 501, np.nan]})
        counts = utils.category_counts(data).set_index("Category")["Observations"]
        self.assertEqual(counts.tolist(), [2, 2, 1, 1, 1, 2])
        for value, expected in [(0, "Good"), (50.1, "Satisfactory"), (100.1, "Moderate"),
                                (200.1, "Poor"), (300.1, "Very Poor"), (400.1, "Severe"), (500, "Severe")]:
            self.assertEqual(model_utils.aqi_category(value), expected)
        self.assertEqual(model_utils.aqi_category(501), "Outside validated AQI range")
        self.assertEqual(model_utils.aqi_category(-1), "Outside validated AQI range")

    def test_anomaly_flags_and_zero_counts(self):
        frame = pd.DataFrame({"City": ["Zero city", "Zero city", "Flagged city"],
                              "Date": ["2020-01-01", "2020-01-02", "2020-01-01"],
                              "Valid_AQI": [50, np.nan, np.nan], "Is_Anomaly": ["False", "0", "True"]})
        clean = utils.prepare_anomalies(frame)
        summary = utils.anomaly_summary(clean).set_index("City")
        self.assertEqual(summary.loc["Zero city", "Anomaly_Days"], 0)
        self.assertEqual(summary.loc["Flagged city", "Anomaly_Percent"], 100)
        frame.loc[0, "Is_Anomaly"] = "unknown"
        with self.assertRaises(utils.DashboardError):
            utils.prepare_anomalies(frame)

    def test_missing_database_is_not_created(self):
        with tempfile.TemporaryDirectory(prefix="airlens-test-") as directory:
            missing = Path(directory) / "missing.db"
            with patch.object(utils, "DB_PATH", missing):
                with self.assertRaises(utils.DashboardError):
                    utils.query_database.__wrapped__("SELECT * FROM air_quality_validated")
            self.assertFalse(missing.exists())


class RealArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = model_utils.load_model()
        cls.engineered = model_utils.load_engineered()

    def test_exact_feature_schema_excludes_target_and_preserves_missing_inputs(self):
        rows = self.engineered.head(3)
        inputs = model_utils.model_input(self.model, rows)
        self.assertEqual(list(inputs.columns), list(self.model.feature_names_in_))
        self.assertEqual(len(inputs.columns), 28)
        self.assertNotIn("Target_AQI", inputs)
        self.assertNotIn("Date", inputs)
        pd.testing.assert_frame_equal(inputs, rows[list(self.model.feature_names_in_)])
        with self.assertRaisesRegex(utils.DashboardError, "PM2.5"):
            model_utils.model_input(self.model, rows.drop(columns=["PM2.5"]))
        changed_target = rows.assign(Target_AQI=9999)
        np.testing.assert_array_equal(model_utils.predict_rows(self.model, rows),
                                      model_utils.predict_rows(self.model, changed_target))

    def test_latest_forecasts_for_every_model_city(self):
        rows = self.engineered.groupby("City").tail(1)
        predictions = model_utils.predict_rows(self.model, rows)
        self.assertEqual(len(predictions), 12)
        self.assertTrue(np.isfinite(predictions).all())

    def test_original_evaluation_results(self):
        evaluation = model_utils.test_evaluation()
        result = model_utils.performance_table(evaluation)
        self.assertEqual(result["Observations"].tolist(), [2170, 2163])
        for i, expected in enumerate([(14.32, 23.03, 0.868), (17.25, 29.36, 0.785)]):
            for metric, value, places in zip(["MAE", "RMSE", "R²"], expected, [2, 2, 3]):
                self.assertEqual(round(result.iloc[i][metric], places), value)

    def test_shap_contributions_reconcile(self):
        row = self.engineered[self.engineered["City"] == "Delhi"].tail(1)
        contributions, reference, prediction = model_utils.local_contributions(self.model, row)
        self.assertAlmostEqual(reference + contributions["Contribution"].sum(), prediction, places=3)
        self.assertTrue((contributions["Contribution"] > 0).any())
        self.assertTrue((contributions["Contribution"] < 0).any())

    def test_validated_database_and_reliability(self):
        frame = utils.load_validated()
        self.assertEqual(frame["City"].nunique(), 26)
        self.assertEqual(frame["Valid_AQI"].count(), 24307)
        self.assertTrue(frame["Valid_AQI"].dropna().between(0, 500).all())
        self.assertEqual(len(utils.reliable_summary()), 12)
        self.assertEqual(utils.reliable_summary().iloc[0]["City"], "Delhi")


class DashboardInteractionTests(unittest.TestCase):
    def page(self, filename):
        return AppTest.from_file(str(ROOT / "app" / "pages" / filename), default_timeout=30).run()

    def assert_clean(self, app):
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertFalse(app.error, [e.value for e in app.error])

    def test_all_pages_through_navigation(self):
        app = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=30).run()
        self.assert_clean(app)
        for path in sorted((ROOT / "app" / "pages").glob("*.py")):
            with self.subTest(page=path.name):
                app.switch_page(f"pages/{path.name}").run()
                self.assert_clean(app)

    def test_city_filters_for_all_cities(self):
        app = self.page("city_analytics.py")
        for city in sorted(utils.load_validated()["City"].unique()):
            with self.subTest(city=city):
                app.selectbox[0].select(city).run()
                self.assert_clean(app)
                expected = utils.city_summary().set_index("City").loc[city, "valid_days"]
                self.assertEqual(app.metric[3].value, f"{expected:,}")
        app.multiselect[0].set_value([]).run()
        self.assert_clean(app)

    def test_empty_multiselects(self):
        for page in ["pollution_trends.py", "seasonal_analysis.py"]:
            app = self.page(page)
            app.multiselect[0].set_value([]).run()
            self.assert_clean(app)
            self.assertTrue(app.info)
            self.assertEqual(len(app.get("plotly_chart")), 0)

    def test_forecast_city_and_date_changes(self):
        app = self.page("forecast.py")
        for city in sorted(model_utils.load_engineered()["City"].unique()):
            app.selectbox[0].select(city).run()
            self.assert_clean(app)
            dates = model_utils.load_engineered().query("City == @city")["Date"]
            app.selectbox[1].set_value(dates.min()).run()
            self.assert_clean(app)
            self.assertEqual(len(app.metric), 3)

    def test_explanation_button(self):
        app = self.page("explainability.py")
        app.button[0].click().run()
        self.assert_clean(app)
        self.assertEqual(len(app.get("plotly_chart")), 2)
        self.assertEqual(len(app.metric), 2)

    def test_anomaly_page_missing_export(self):
        with tempfile.TemporaryDirectory(prefix="airlens-test-") as directory:
            with patch.object(utils, "ANOMALY_PATH", Path(directory) / "missing.csv"):
                app = self.page("anomaly_detection.py")
                self.assert_clean(app)
                self.assertIn("unavailable", app.info[0].value)

    def test_anomaly_page_zero_and_missing_aqi(self):
        # Synthetic fixture stays in a temporary directory; no project data is replaced.
        fixture = pd.DataFrame({"City": ["Zero city", "Zero city", "Flagged city", "Flagged city"],
                                "Date": ["2020-01-01", "2020-01-03", "2020-01-01", "2020-01-03"],
                                "Valid_AQI": [40, np.nan, 80, np.nan],
                                "Is_Anomaly": [False, False, False, True]})
        with tempfile.TemporaryDirectory(prefix="airlens-test-") as directory:
            path = Path(directory) / "anomalies.csv"
            fixture.to_csv(path, index=False)
            utils.load_anomalies.clear()
            try:
                with patch.object(utils, "ANOMALY_PATH", path):
                    app = self.page("anomaly_detection.py")
                    app.selectbox[0].select("Zero city").run()
                    self.assert_clean(app)
                    self.assertEqual(app.metric[0].value, "0")
                    self.assertTrue(any("No anomalies" in info.value for info in app.info))
                    app.selectbox[0].select("Flagged city").run()
                    self.assert_clean(app)
                    self.assertEqual(app.metric[0].value, "1")
                    self.assertEqual(app.metric[1].value, "50.0%")
                    self.assertEqual(len(app.dataframe[0].value), 1)
            finally:
                utils.load_anomalies.clear()


if __name__ == "__main__":
    unittest.main()
