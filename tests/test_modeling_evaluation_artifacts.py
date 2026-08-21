"""Integration test for reliability artifacts using an in-memory fixture."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import src.evaluate as evaluation
from src.preprocessing import EXPECTED_CROP_CLASSES, FEATURE_NAMES
from src.train import build_random_forest, create_train_test_split


def _small_balanced_fixture() -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    labels = np.repeat(np.asarray(EXPECTED_CROP_CLASSES), 10)
    encoder = LabelEncoder().fit(labels)
    y = encoder.transform(labels)
    # Every row is unique and crop clusters are deliberately easy to separate.
    values = []
    for encoded_class in y:
        offset = len(values) % 10
        base = float(encoded_class * 100 + offset)
        values.append([base + column / 10.0 for column in range(7)])
    X = pd.DataFrame(values, columns=FEATURE_NAMES)
    return X, y, encoder


class ReliabilityArtifactTests(unittest.TestCase):
    def test_evaluation_saves_all_required_files_and_training_only_cv(self) -> None:
        X, y, encoder = _small_balanced_fixture()
        X_train, _, y_train, _ = create_train_test_split(X, y)
        model = build_random_forest().fit(X_train, y_train)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_path = root / "random_forest_crop.joblib"
            encoder_path = root / "label_encoder.joblib"
            results_dir = root / "results"
            joblib.dump(model, model_path)
            joblib.dump(encoder, encoder_path)

            with patch.object(
                evaluation,
                "preprocess_data",
                return_value=(X, y, encoder),
            ):
                result = evaluation.evaluate_random_forest(
                    Path("unused-dataset.csv"),
                    model_path=model_path,
                    encoder_path=encoder_path,
                    results_dir=results_dir,
                    verbose=False,
                )

            for filename in (
                "confusion_matrix.png",
                "confusion_matrix.csv",
                "classification_report.txt",
                "evaluation_results.json",
            ):
                artifact = results_dir / filename
                self.assertTrue(artifact.is_file(), filename)
                self.assertGreater(artifact.stat().st_size, 0)

            self.assertEqual(
                result["cross_validation"]["partition"],
                "training_partition_only",
            )
            self.assertEqual(len(result["cross_validation"]["fold_accuracies"]), 5)
            self.assertEqual(
                result["cross_validation"]["training_sample_count"], len(X_train)
            )
            self.assertEqual(
                result["leakage_check"]["overlapping_unique_feature_vectors"], 0
            )
            saved = json.loads(
                (results_dir / "evaluation_results.json").read_text(encoding="utf-8")
            )
            self.assertIn("held_out_test_metrics", saved)
            self.assertEqual(len(saved["class_names"]), 22)


if __name__ == "__main__":
    unittest.main()
