from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics import confusion_matrix, per_class_accuracy  # noqa: E402


class MetricsTests(unittest.TestCase):
    def test_confusion_matrix_counts_true_and_predicted_labels(self) -> None:
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 0, 2])

        matrix = confusion_matrix(y_true, y_pred, num_classes=3)

        np.testing.assert_array_equal(
            matrix,
            np.array(
                [
                    [1, 1, 0],
                    [0, 2, 0],
                    [1, 0, 1],
                ]
            ),
        )

    def test_per_class_accuracy_handles_empty_classes(self) -> None:
        matrix = np.array(
            [
                [2, 0],
                [0, 0],
            ]
        )

        np.testing.assert_allclose(per_class_accuracy(matrix), np.array([1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
