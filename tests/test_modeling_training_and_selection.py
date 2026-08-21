"""Focused tests for split safety, model definitions, metrics, and selection."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.compare_models import build_model_candidates, select_model_from_results
from src.preprocessing import FEATURE_NAMES
from src.train import (
    FeatureLeakageError,
    build_random_forest,
    calculate_classification_metrics,
    create_train_test_split,
    feature_overlap_report,
    require_no_feature_overlap,
)


def _metric_row(model: str, accuracy: float, macro_f1: float) -> dict[str, float | str]:
    return {
        "model": model,
        "accuracy": accuracy,
        "macro_precision": macro_f1,
        "macro_recall": macro_f1,
        "macro_f1": macro_f1,
        "weighted_precision": macro_f1,
        "weighted_recall": macro_f1,
        "weighted_f1": macro_f1,
    }


class SplitAndMetricTests(unittest.TestCase):
    def test_split_is_reproducible_stratified_80_20(self) -> None:
        values = np.arange(220 * len(FEATURE_NAMES), dtype=float).reshape(220, 7)
        X = pd.DataFrame(values, columns=FEATURE_NAMES)
        y = np.repeat(np.arange(22), 10)
        first = create_train_test_split(X, y)
        second = create_train_test_split(X, y)
        X_train, X_test, y_train, y_test = first
        self.assertEqual((len(X_train), len(X_test)), (176, 44))
        self.assertEqual(np.bincount(y_train).tolist(), [8] * 22)
        self.assertEqual(np.bincount(y_test).tolist(), [2] * 22)
        self.assertEqual(X_train.index.tolist(), second[0].index.tolist())
        self.assertEqual(X_test.index.tolist(), second[1].index.tolist())

    def test_exact_seven_feature_overlap_is_blocking(self) -> None:
        train = pd.DataFrame([[1, 2, 3, 4, 5, 6, 7]], columns=FEATURE_NAMES)
        test = pd.DataFrame([[1, 2, 3, 4, 5, 6, 7]], columns=FEATURE_NAMES)
        report = feature_overlap_report(train, test)
        self.assertTrue(report["overlap_detected"])
        self.assertEqual(report["overlapping_unique_feature_vectors"], 1)
        with self.assertRaises(FeatureLeakageError):
            require_no_feature_overlap(train, test)

    def test_complete_metric_set_uses_computed_predictions(self) -> None:
        metrics = calculate_classification_metrics(
            np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])
        )
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["correct_predictions"], 3)
        for name in (
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
        ):
            self.assertIn(name, metrics)

    def test_random_forest_configuration_is_exact(self) -> None:
        model = build_random_forest()
        self.assertEqual(model.n_estimators, 100)
        self.assertEqual(model.random_state, 42)
        self.assertEqual(model.n_jobs, -1)


class ComparisonDefinitionTests(unittest.TestCase):
    def test_exact_six_models_and_training_only_scaler_pipelines(self) -> None:
        models = build_model_candidates()
        self.assertEqual(
            list(models),
            [
                "Logistic Regression",
                "Decision Tree",
                "K-Nearest Neighbors",
                "Support Vector Machine",
                "Gaussian Naive Bayes",
                "Random Forest",
            ],
        )
        for name in (
            "Logistic Regression",
            "K-Nearest Neighbors",
            "Support Vector Machine",
        ):
            self.assertIsInstance(models[name], Pipeline)
            self.assertIsInstance(models[name].named_steps["scaler"], StandardScaler)
        self.assertNotIsInstance(models["Decision Tree"], Pipeline)
        self.assertNotIsInstance(models["Gaussian Naive Bayes"], Pipeline)
        self.assertNotIsInstance(models["Random Forest"], Pipeline)

    def test_accuracy_tie_is_resolved_by_macro_f1(self) -> None:
        rows = [
            _metric_row("Random Forest", 0.99, 0.991),
            _metric_row("Gaussian Naive Bayes", 0.99, 0.990),
            _metric_row("SVM", 0.98, 0.980),
        ]
        selection = select_model_from_results(rows)
        self.assertEqual(selection["highest_accuracy_models"], ["Random Forest", "Gaussian Naive Bayes"])
        self.assertEqual(selection["selected_model"], "Random Forest")
        self.assertEqual(selection["tie_break_trace"][0]["metric"], "macro_f1")

    def test_unresolved_tie_does_not_claim_a_unique_best_model(self) -> None:
        rows = [
            _metric_row("Model A", 0.99, 0.99),
            _metric_row("Model B", 0.99, 0.99),
        ]
        selection = select_model_from_results(rows)
        self.assertEqual(selection["status"], "unresolved_metric_tie")
        self.assertIsNone(selection["selected_model"])
        self.assertIn("No unique best model", selection["reason"])


if __name__ == "__main__":
    unittest.main()
