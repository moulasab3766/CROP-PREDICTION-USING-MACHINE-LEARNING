"""Fast tests for tuning, ablation, robustness, disagreement, and error helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.preprocessing import FEATURE_NAMES
from src.research.error_analysis import grouped_error_descriptives
from src.research.feature_ablation import ablation_configurations, add_ablation_deltas
from src.research.model_disagreement import consensus_agreement
from src.research.robustness_analysis import (
    apply_controlled_change,
    perturbation_specification,
    stratified_sample_positions,
)
from src.research.tune_random_forest import format_search_results, parameter_search_space


class TuningUtilityTests(unittest.TestCase):
    def test_search_space_contains_all_required_rf_parameters(self) -> None:
        space = parameter_search_space()
        self.assertEqual(
            set(space),
            {
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "max_features",
                "class_weight",
            },
        )
        self.assertIn(None, space["class_weight"])
        self.assertIn("balanced", space["class_weight"])

    def test_search_result_formatting_preserves_params_and_ranks(self) -> None:
        frame = format_search_results(
            {
                "params": [{"n_estimators": 100}, {"n_estimators": 200}],
                "mean_test_macro_f1": np.array([0.8, 0.9]),
                "std_test_macro_f1": np.array([0.1, 0.05]),
                "rank_test_macro_f1": np.array([2, 1]),
                "mean_test_accuracy": np.array([0.81, 0.91]),
                "std_test_accuracy": np.array([0.09, 0.04]),
                "rank_test_accuracy": np.array([2, 1]),
                "mean_fit_time": np.array([1.0, 2.0]),
            }
        )
        self.assertEqual(frame.iloc[0]["param_n_estimators"], 200)
        self.assertEqual(frame.iloc[0]["rank_cv_macro_f1"], 1)


class AblationUtilityTests(unittest.TestCase):
    def test_configurations_are_baseline_plus_exactly_one_removal_each(self) -> None:
        configurations = ablation_configurations()
        self.assertEqual(len(configurations), 8)
        self.assertEqual(configurations["All 7 Features"], FEATURE_NAMES)
        for name, features in list(configurations.items())[1:]:
            self.assertTrue(name.startswith("Without "))
            self.assertEqual(len(features), 6)

    def test_signed_deltas_use_all_feature_baseline(self) -> None:
        rows = add_ablation_deltas(
            [
                {"configuration": "All 7 Features", "accuracy": 0.9, "macro_f1": 0.8},
                {"configuration": "Without N", "accuracy": 0.85, "macro_f1": 0.75},
            ]
        )
        self.assertAlmostEqual(rows[1]["accuracy_delta"], -0.05)
        self.assertAlmostEqual(rows[1]["macro_f1_delta"], -0.05)


class RobustnessUtilityTests(unittest.TestCase):
    def test_every_feature_has_symmetric_changes_and_original(self) -> None:
        specification = perturbation_specification()
        self.assertEqual(tuple(specification), FEATURE_NAMES)
        for feature, values in specification.items():
            changes = values["changes"]
            self.assertIn(0.0, changes, feature)
            self.assertAlmostEqual(changes[0], -changes[-1])
            self.assertAlmostEqual(changes[1], -changes[-2])

    def test_changes_clip_to_documented_bounds_and_report_clips(self) -> None:
        values, clipped = apply_controlled_change(
            np.array([1.0, 5.0]), -3.0, kind="absolute", minimum=0.0, maximum=10.0
        )
        np.testing.assert_array_equal(values, [0.0, 2.0])
        np.testing.assert_array_equal(clipped, [True, False])

    def test_relative_change_and_stratified_selection(self) -> None:
        values, clipped = apply_controlled_change(
            np.array([100.0]), 0.1, kind="relative", minimum=0.0, maximum=200.0
        )
        self.assertAlmostEqual(values[0], 110.0)
        self.assertFalse(clipped[0])
        positions = stratified_sample_positions(np.array([0, 0, 1, 1, 1]), 2)
        np.testing.assert_array_equal(positions, [0, 1, 2, 3])


class AgreementAndErrorUtilityTests(unittest.TestCase):
    def test_consensus_count_and_tie_are_deterministic(self) -> None:
        self.assertEqual(consensus_agreement(["rice"] * 6), (6, "rice"))
        self.assertEqual(
            consensus_agreement(["rice", "rice", "maize", "maize", "apple", "apple"]),
            (2, "apple"),
        )

    def test_error_descriptives_keep_correct_and_incorrect_groups(self) -> None:
        frame = pd.DataFrame(
            [
                {"correct": True, "top_1_probability": 0.9, "top_1_vs_top_2_margin": 0.8, "agreement_count": 6},
                {"correct": False, "top_1_probability": 0.5, "top_1_vs_top_2_margin": 0.1, "agreement_count": 3},
            ]
        )
        rows = grouped_error_descriptives(frame)
        self.assertEqual({row["group"] for row in rows}, {"correct", "incorrect"})
        incorrect = next(row for row in rows if row["group"] == "incorrect")
        self.assertEqual(incorrect["mean_agreement_count"], 3.0)


if __name__ == "__main__":
    unittest.main()
