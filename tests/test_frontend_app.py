"""Focused, dependency-light tests for the Streamlit presentation layer."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app.py"


def _load_app_with_stubs():
    """Load app helpers without requiring third-party packages in the test runner."""

    streamlit_stub = types.ModuleType("streamlit")
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = lambda data, columns=None: {"data": data, "columns": columns}
    pipeline_stub = types.ModuleType("src.pipeline")
    pipeline_stub.run_pipeline = lambda **_kwargs: {}

    previous = {
        name: sys.modules.get(name)
        for name in ("streamlit", "pandas", "src.pipeline", "crop_frontend_under_test")
    }
    sys.modules["streamlit"] = streamlit_stub
    sys.modules["pandas"] = pandas_stub
    sys.modules["src.pipeline"] = pipeline_stub

    spec = importlib.util.spec_from_file_location("crop_frontend_under_test", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load app.py for testing.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    for name, old_module in previous.items():
        if old_module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old_module
    return module


class _Column:
    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        return False


class _StreamlitRecorder:
    def __init__(self):
        self.subheaders = []
        self.metrics = []
        self.dataframes = []
        self.captions = []
        self.warnings = []
        self.images = []
        self.markdown_calls = []
        self.info_calls = []

    def divider(self):
        return None

    def subheader(self, value):
        self.subheaders.append(value)

    def columns(self, count):
        return [_Column() for _ in range(count)]

    def metric(self, label, value):
        self.metrics.append((label, value))

    def caption(self, value):
        self.captions.append(value)

    def dataframe(self, value, **_kwargs):
        self.dataframes.append(value)

    def info(self, value):
        self.info_calls.append(value)

    def image(self, value, **_kwargs):
        self.images.append(value)

    def warning(self, value):
        self.warnings.append(value)

    def markdown(self, value, **_kwargs):
        self.markdown_calls.append(value)


class FrontendAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _load_app_with_stubs()

    def test_pipeline_is_not_called_before_submission(self):
        calls = []

        def fake_pipeline(**values):
            calls.append(values)
            return {"ok": True}

        result = self.app._run_if_submitted(False, {"N": 90}, fake_pipeline)

        self.assertIsNone(result)
        self.assertEqual(calls, [])

    def test_submission_passes_all_seven_manual_inputs_to_pipeline(self):
        inputs = {
            "N": 90,
            "P": 42,
            "K": 43,
            "temperature": 25.0,
            "humidity": 80.0,
            "ph": 6.5,
            "rainfall": 200.0,
        }
        calls = []

        def fake_pipeline(**values):
            calls.append(values)
            return {"predicted_crop": "rice"}

        result = self.app._run_if_submitted(True, inputs, fake_pipeline)

        self.assertEqual(result, {"predicted_crop": "rice"})
        self.assertEqual(calls, [inputs])

    def test_probability_is_derived_from_pipeline_value(self):
        self.assertEqual(self.app._format_probability(0.9), "90.00%")
        self.assertEqual(self.app._format_probability(0), "0.00%")
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            self.app._format_probability(1.01)

    def test_top_three_rows_are_ranked_by_actual_probability(self):
        rows = self.app._top_three_rows(
            [
                {"rank": 3, "crop": "jute", "probability": 0.10},
                {"rank": 1, "crop": "rice", "probability": 0.85},
                {"rank": 2, "crop": "coffee", "probability": 0.05},
            ]
        )

        self.assertEqual([row["Crop"] for row in rows], ["Rice", "Jute", "Coffee"])
        self.assertEqual([row["Rank"] for row in rows], [1, 2, 3])
        self.assertEqual([row["Probability"] for row in rows], ["85.00%", "10.00%", "5.00%"])

        with self.assertRaisesRegex(ValueError, "exactly three"):
            self.app._top_three_rows(rows[:2])

    def test_renderer_shows_complete_result_and_verification_warning(self):
        recorder = _StreamlitRecorder()
        original_streamlit = self.app.st
        self.app.st = recorder
        try:
            result = {
                "predicted_crop": "rice",
                "prediction_probability": 0.85,
                "top_3": [
                    {"rank": 1, "crop": "rice", "probability": 0.85},
                    {"rank": 2, "crop": "jute", "probability": 0.10},
                    {"rank": 3, "crop": "coffee", "probability": 0.05},
                ],
                "feature_importance": [
                    {"feature": "rainfall", "importance": 0.4, "importance_percent": 40.0}
                ],
                "soil_assessment": {
                    "nitrogen_status": "Needs verification",
                    "phosphorus_status": "Needs verification",
                    "potassium_status": "Needs verification",
                    "ph_status": "Needs verification",
                    "overall_assessment": "Needs verification",
                    "thresholds_verified": False,
                    "threshold_source": "Provisional configuration",
                },
            }

            self.app._render_results(
                result,
                importance_image=PROJECT_ROOT / "results" / "does-not-exist.png",
            )
        finally:
            self.app.st = original_streamlit

        self.assertIn("Recommended Crop", recorder.subheaders)
        self.assertIn("Top-3 Crop Recommendations", recorder.subheaders)
        self.assertIn("Why this crop?", recorder.subheaders)
        self.assertIn("Overall Model Feature Importance", recorder.subheaders)
        self.assertIn("Indicative Soil Parameter Assessment", recorder.subheaders)
        self.assertIn(("Recommended crop", "Rice"), recorder.metrics)
        self.assertIn(("Prediction probability", "85.00%"), recorder.metrics)
        self.assertEqual(len(recorder.dataframes), 2)
        self.assertTrue(any("require verification" in warning for warning in recorder.warnings))
        self.assertTrue(any("temporarily unavailable" in message for message in recorder.info_calls))
        self.assertFalse(any("accuracy" in caption.lower() for caption in recorder.captions[:2]))

    def test_frontend_does_not_contain_training_implementation(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertNotIn("RandomForestClassifier", source)
        self.assertNotIn("train_test_split", source)
        self.assertNotIn(".fit(", source)
        self.assertIn("from src.pipeline import run_pipeline", source)


if __name__ == "__main__":
    unittest.main()
