from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cnn_mnist import SimpleMNISTCNN, prepare_mnist_images, train_cnn, CNNConfig  # noqa: E402


class SimpleMNISTCNNTests(unittest.TestCase):
    def test_forward_outputs_probabilities(self) -> None:
        X = np.random.default_rng(1).random((4, 1, 28, 28))
        model = SimpleMNISTCNN(filters=2, kernel_size=3, learning_rate=0.01, seed=2)

        probs = model.predict_proba(X)

        self.assertEqual(probs.shape, (4, 10))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(4), atol=1e-10)

    def test_fit_records_initial_and_final_metrics(self) -> None:
        rng = np.random.default_rng(3)
        X = rng.random((8, 1, 28, 28))
        y = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        model = SimpleMNISTCNN(filters=2, kernel_size=3, learning_rate=0.01, seed=4)

        history = model.fit(X, y, epochs=1, batch_size=4, val_data=(X, y), verbose=False)

        self.assertEqual(len(history["loss"]), 2)
        self.assertEqual(len(history["val_accuracy"]), 2)


class CNNDataTests(unittest.TestCase):
    def test_prepare_mnist_images_returns_nchw_scaled_images(self) -> None:
        X, y = prepare_mnist_images("train", limit=8)

        self.assertEqual(X.shape, (8, 1, 28, 28))
        self.assertEqual(y.shape, (8,))
        self.assertGreaterEqual(float(X.min()), 0.0)
        self.assertLessEqual(float(X.max()), 1.0)

    def test_train_cnn_smoke_run(self) -> None:
        metrics = train_cnn(
            CNNConfig(filters=2, epochs=1, batch_size=4, train_limit=8, val_limit=8),
            verbose=False,
        )

        self.assertEqual(metrics["model"], "simple_cnn")
        self.assertEqual(len(metrics["history"]["accuracy"]), 2)


if __name__ == "__main__":
    unittest.main()
