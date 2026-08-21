"""Top-3 presentation and read-only research-dashboard regression tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.app_support import (
    CLOSE_RESULT_MARGIN_THRESHOLD,
    ResearchArtifactError,
    close_result_message,
    load_research_dashboard_data,
    ranked_top_three,
    top_prediction_margin,
    top_three_display_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TopThreePresentationTests(unittest.TestCase):
    def test_descending_full_precision_and_zero_probability_are_preserved(self) -> None:
        source = [
            {"crop": "zero_crop", "probability": 0.0},
            {"crop": "second", "probability": 0.123456789},
            {"crop": "first", "probability": 0.876543211},
        ]
        ranked = ranked_top_three(source)
        rows = top_three_display_rows(source)

        self.assertEqual([item["crop"] for item in ranked], ["first", "second", "zero_crop"])
        self.assertEqual(ranked[0]["probability"], 0.876543211)
        self.assertEqual(ranked[-1]["probability"], 0.0)
        self.assertEqual(rows[-1]["Probability"], "0.00%")

    def test_equal_probabilities_keep_model_class_order(self) -> None:
        rows = ranked_top_three(
            [
                {"crop": "class_b", "probability": 0.4},
                {"crop": "class_a", "probability": 0.4},
                {"crop": "class_c", "probability": 0.2},
            ]
        )
        self.assertEqual([row["crop"] for row in rows], ["class_b", "class_a", "class_c"])

    def test_margin_and_close_message_use_documented_threshold(self) -> None:
        close = [
            {"crop": "a", "probability": 0.50},
            {"crop": "b", "probability": 0.45},
            {"crop": "c", "probability": 0.05},
        ]
        far = [
            {"crop": "a", "probability": 0.80},
            {"crop": "b", "probability": 0.15},
            {"crop": "c", "probability": 0.05},
        ]
        self.assertAlmostEqual(top_prediction_margin(close), CLOSE_RESULT_MARGIN_THRESHOLD)
        self.assertIn("similar support", close_result_message(close))
        self.assertIsNone(close_result_message(far))


class ResearchDashboardLoaderTests(unittest.TestCase):
    def test_checked_in_summaries_are_loaded_without_recalculation(self) -> None:
        dashboard = load_research_dashboard_data(project_root=PROJECT_ROOT)
        self.assertTrue(dashboard["available"])
        self.assertAlmostEqual(dashboard["baseline"]["held_out_accuracy"], 0.9954545454545455)
        self.assertEqual(dashboard["baseline"]["held_out_samples"], 440)
        self.assertEqual(dashboard["top_k"]["top_2_accuracy"], 1.0)
        self.assertFalse(dashboard["research"]["calibration"]["improved"])
        self.assertGreaterEqual(len(dashboard["charts"]), 5)

    def test_all_optional_research_files_may_be_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dashboard = load_research_dashboard_data(project_root=directory)
        self.assertFalse(dashboard["available"])
        self.assertIsNone(dashboard["baseline"])
        self.assertIsNone(dashboard["top_k"])
        self.assertGreaterEqual(len(dashboard["warnings"]), 3)

    def test_invalid_numeric_summary_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results" / "research").mkdir(parents=True)
            evaluation = {
                "held_out_test_metrics": {"accuracy": "not-a-number", "macro_f1": 0.9},
                "split": {"test_samples": 10},
                "cross_validation": {"mean_accuracy": 0.9},
            }
            with (root / "results" / "evaluation_results.json").open("w", encoding="utf-8") as handle:
                json.dump(evaluation, handle)
            with self.assertRaises(ResearchArtifactError):
                load_research_dashboard_data(project_root=root)

    def test_app_does_not_hardcode_benchmark_accuracy_as_probability(self) -> None:
        source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("99.55%", source)
        self.assertNotIn("Guaranteed Confidence", source)
        self.assertIn("Prediction Probability", source)


if __name__ == "__main__":
    unittest.main()
