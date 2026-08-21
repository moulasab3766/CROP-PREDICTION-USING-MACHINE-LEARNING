"""Deterministic tests for shared research and Top-K utilities."""

from __future__ import annotations

import unittest

import numpy as np

from src.preprocessing import FEATURE_NAMES
from src.research.common import (
    expected_calibration_error,
    load_research_split,
    probability_metrics,
    top_label_calibration_bins,
    validate_probability_matrix,
)
from src.research.evaluate_topk import calculate_top_k_summary, top_k_correctness


class ResearchSplitTests(unittest.TestCase):
    def test_canonical_split_sizes_order_and_stratification(self) -> None:
        split = load_research_split()
        self.assertEqual(split.X_train.shape, (1760, 7))
        self.assertEqual(split.X_test.shape, (440, 7))
        self.assertEqual(tuple(split.X_train.columns), FEATURE_NAMES)
        self.assertTrue(np.all(np.bincount(split.y_train) == 80))
        self.assertTrue(np.all(np.bincount(split.y_test) == 20))


class ProbabilityValidationTests(unittest.TestCase):
    def test_probability_zero_one_and_rows_summing_to_one_are_valid(self) -> None:
        matrix = validate_probability_matrix([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_equal(matrix, [[1.0, 0.0], [0.0, 1.0]])

    def test_nan_negative_above_one_and_bad_sum_are_rejected(self) -> None:
        for matrix in (
            [[np.nan, np.nan]],
            [[-0.1, 1.1]],
            [[0.2, 1.2]],
            [[0.2, 0.2]],
        ):
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    validate_probability_matrix(matrix)

    def test_top_label_bins_keep_empty_bins_and_equal_confidences(self) -> None:
        probabilities = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
        bins = top_label_calibration_bins(
            np.array([0, 1, 0]), probabilities, np.array([0, 1]), n_bins=5
        )
        self.assertEqual(len(bins), 5)
        self.assertEqual(sum(int(row["count"]) for row in bins), 3)
        populated = [row for row in bins if row["count"]]
        self.assertEqual(len(populated), 1)
        self.assertAlmostEqual(float(populated[0]["observed_accuracy"]), 2 / 3)
        self.assertAlmostEqual(expected_calibration_error(bins), abs(2 / 3 - 0.8))

    def test_probability_metrics_are_computed_from_true_columns(self) -> None:
        metrics = probability_metrics(
            np.array([0, 1]),
            np.array([[0.9, 0.1], [0.2, 0.8]]),
            np.array([0, 1]),
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["mean_true_class_probability"], 0.85)
        self.assertGreater(metrics["log_loss"], 0.0)


class TopKLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classes = np.array([0, 1, 2, 3])
        self.probabilities = np.array(
            [
                [0.7, 0.2, 0.1, 0.0],
                [0.5, 0.4, 0.1, 0.0],
                [0.5, 0.3, 0.2, 0.0],
                [0.5, 0.3, 0.19, 0.01],
            ]
        )
        self.truth = np.array([0, 1, 2, 3])

    def test_rank_one_two_three_and_outside_top_three_cases(self) -> None:
        np.testing.assert_array_equal(
            top_k_correctness(self.probabilities, self.truth, self.classes, k=1),
            [True, False, False, False],
        )
        np.testing.assert_array_equal(
            top_k_correctness(self.probabilities, self.truth, self.classes, k=2),
            [True, True, False, False],
        )
        np.testing.assert_array_equal(
            top_k_correctness(self.probabilities, self.truth, self.classes, k=3),
            [True, True, True, False],
        )

    def test_summary_is_monotonic_and_counts_incremental_corrections(self) -> None:
        summary = calculate_top_k_summary(
            self.probabilities, self.truth, self.classes
        )
        self.assertEqual(summary["top_1_accuracy"], 0.25)
        self.assertEqual(summary["top_2_accuracy"], 0.50)
        self.assertEqual(summary["top_3_accuracy"], 0.75)
        self.assertEqual(summary["top_5_accuracy"], 1.0)
        self.assertEqual(summary["additional_correct_top_1_to_2"], 1)
        self.assertEqual(summary["additional_correct_top_2_to_3"], 1)

    def test_invalid_k_is_rejected(self) -> None:
        for k in (0, 5, True):
            with self.subTest(k=k):
                with self.assertRaises(ValueError):
                    top_k_correctness(self.probabilities, self.truth, self.classes, k=k)


if __name__ == "__main__":
    unittest.main()
