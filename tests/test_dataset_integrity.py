"""Regression checks that guard the unmodified Kaggle dataset contract."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "Crop_recommendation.csv"
EXPECTED_COLUMNS = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
    "label",
]
EXPECTED_CROPS = {
    "rice",
    "maize",
    "chickpea",
    "kidneybeans",
    "pigeonpeas",
    "mothbeans",
    "mungbean",
    "blackgram",
    "lentil",
    "pomegranate",
    "banana",
    "mango",
    "grapes",
    "watermelon",
    "muskmelon",
    "apple",
    "orange",
    "papaya",
    "coconut",
    "cotton",
    "jute",
    "coffee",
}
EXPECTED_SHA256 = "54a5a6e5408668e668667efc50de2fc867c1b875e0431b4f54dd331b0a109a4e"


class DatasetIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataframe = pd.read_csv(DATASET_PATH)

    def test_source_file_checksum_is_unchanged(self) -> None:
        actual = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
        self.assertEqual(actual, EXPECTED_SHA256)

    def test_shape_and_column_order(self) -> None:
        self.assertEqual(self.dataframe.shape, (2200, 8))
        self.assertEqual(list(self.dataframe.columns), EXPECTED_COLUMNS)

    def test_data_quality_contract(self) -> None:
        self.assertEqual(int(self.dataframe.isna().sum().sum()), 0)
        self.assertEqual(int(self.dataframe.duplicated().sum()), 0)
        self.assertTrue(
            all(pd.api.types.is_numeric_dtype(self.dataframe[name]) for name in EXPECTED_COLUMNS[:-1])
        )

    def test_classes_are_exactly_balanced(self) -> None:
        counts = self.dataframe["label"].value_counts()
        self.assertEqual(set(counts.index), EXPECTED_CROPS)
        self.assertEqual(set(counts.tolist()), {100})


if __name__ == "__main__":
    unittest.main()

