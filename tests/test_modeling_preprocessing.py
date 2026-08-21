"""Focused tests for strict dataset inspection and preprocessing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import (
    EXPECTED_COLUMNS,
    EXPECTED_CROP_CLASSES,
    FEATURE_NAMES,
    DatasetValidationError,
    inspect_dataframe,
    load_dataset,
    preprocess_data,
    validate_dataset,
)


def valid_crop_dataframe() -> pd.DataFrame:
    """Create an in-memory schema fixture; it is never a project dataset."""

    rows: list[dict[str, float | int | str]] = []
    for crop_index, crop in enumerate(EXPECTED_CROP_CLASSES):
        for sample_index in range(100):
            unique = crop_index * 100 + sample_index
            rows.append(
                {
                    "N": unique,
                    "P": unique + 1,
                    "K": unique + 2,
                    "temperature": 10.0 + unique / 1000.0,
                    "humidity": 20.0 + unique / 1000.0,
                    "ph": 4.0 + unique / 10000.0,
                    "rainfall": 50.0 + unique / 100.0,
                    "label": crop,
                }
            )
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


class DatasetValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = valid_crop_dataframe()

    def test_valid_dataframe_has_exact_required_facts(self) -> None:
        report = inspect_dataframe(self.frame)
        self.assertTrue(report["valid"], report["issues"])
        self.assertEqual(report["shape"], (2200, 8))
        self.assertEqual(report["columns"], list(EXPECTED_COLUMNS))
        self.assertEqual(report["duplicate_rows"], 0)
        self.assertEqual(report["unique_crop_count"], 22)
        self.assertTrue(report["is_class_balanced"])
        self.assertTrue(all(count == 100 for count in report["class_counts"].values()))

    def test_column_order_is_strict_and_data_is_not_mutated(self) -> None:
        wrong_order = self.frame.loc[
            :, ("P", "N", "K", "temperature", "humidity", "ph", "rainfall", "label")
        ].copy()
        before = wrong_order.copy(deep=True)
        with self.assertRaises(DatasetValidationError) as caught:
            validate_dataset(wrong_order)
        self.assertIn("exact order", str(caught.exception))
        pd.testing.assert_frame_equal(wrong_order, before)

    def test_missing_duplicate_and_class_count_problems_are_all_reported(self) -> None:
        invalid = self.frame.copy(deep=True)
        invalid.iloc[-1] = invalid.iloc[0]
        invalid.loc[1, "temperature"] = np.nan
        report = inspect_dataframe(invalid)
        self.assertFalse(report["valid"])
        joined = "\n".join(report["issues"])
        self.assertIn("missing values", joined)
        self.assertIn("duplicate rows", joined)
        self.assertIn("mismatched counts", joined)

    def test_csv_loading_preserves_feature_order_and_encodes_only_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "Crop_recommendation.csv"
            self.frame.to_csv(path, index=False)
            loaded = load_dataset(path)
            X, y, encoder = preprocess_data(path)

        pd.testing.assert_frame_equal(loaded, self.frame)
        self.assertEqual(tuple(X.columns), FEATURE_NAMES)
        self.assertEqual(X.shape, (2200, 7))
        self.assertEqual(y.shape, (2200,))
        self.assertEqual(set(encoder.classes_), set(EXPECTED_CROP_CLASSES))
        self.assertTrue(all(np.issubdtype(dtype, np.number) for dtype in X.dtypes))


if __name__ == "__main__":
    unittest.main()
