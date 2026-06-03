from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from classification_nn import (  # noqa: E402
    MLPClassifier,
    evaluate_checkpoint,
    prepare_iris_split,
    prepare_mnist_split,
    save_checkpoint,
    train_dataset,
)


class MLPClassifierTests(unittest.TestCase):
    def test_mlp_learns_simple_three_class_problem(self) -> None:
        X = np.array(
            [
                [-2.0, -1.0],
                [-1.5, -1.2],
                [0.0, 2.0],
                [0.4, 1.7],
                [2.0, -1.0],
                [1.7, -1.3],
            ],
            dtype=np.float64,
        )
        y = np.array([0, 0, 1, 1, 2, 2])

        model = MLPClassifier(
            input_dim=2,
            hidden_dim=8,
            output_dim=3,
            learning_rate=0.2,
            seed=7,
        )
        history = model.fit(X, y, epochs=250, batch_size=3)

        self.assertLess(history["loss"][-1], history["loss"][0])
        self.assertGreaterEqual(model.accuracy(X, y), 0.95)

    def test_fit_records_initial_metrics_before_training(self) -> None:
        X = np.array(
            [
                [-1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        )
        y = np.array([0, 1, 2])
        model = MLPClassifier(
            input_dim=2,
            hidden_dim=4,
            output_dim=3,
            learning_rate=0.1,
            seed=3,
        )

        initial_accuracy = model.accuracy(X, y)
        history = model.fit(X, y, epochs=1, batch_size=3)

        self.assertEqual(len(history["accuracy"]), 2)
        self.assertEqual(history["accuracy"][0], initial_accuracy)

    def test_train_dataset_allows_zero_epochs_for_baseline_only(self) -> None:
        metrics = train_dataset(
            "iris",
            epochs=0,
            hidden_dim=4,
            learning_rate=0.01,
            batch_size=16,
            verbose=False,
        )

        self.assertEqual(metrics["config"]["epochs"], 0)
        self.assertEqual(metrics["epochs_trained"], 0)
        self.assertFalse(metrics["stopped_early"])
        self.assertEqual(len(metrics["history"]["accuracy"]), 1)

    def test_train_dataset_records_l2_regularization_strength(self) -> None:
        metrics = train_dataset(
            "iris",
            epochs=1,
            hidden_dim=4,
            learning_rate=0.01,
            batch_size=16,
            l2=0.001,
            verbose=False,
        )

        self.assertEqual(metrics["config"]["l2"], 0.001)

    def test_train_dataset_records_training_strategy_options(self) -> None:
        metrics = train_dataset(
            "iris",
            epochs=2,
            hidden_dim=4,
            learning_rate=0.02,
            batch_size=16,
            optimizer="momentum",
            momentum=0.8,
            lr_decay=0.1,
            early_stopping_patience=2,
            verbose=False,
        )

        self.assertEqual(metrics["config"]["optimizer"], "momentum")
        self.assertEqual(metrics["config"]["momentum"], 0.8)
        self.assertEqual(metrics["config"]["lr_decay"], 0.1)
        self.assertEqual(metrics["config"]["early_stopping_patience"], 2)
        self.assertIn("learning_rate", metrics["history"])
        self.assertEqual(metrics["epochs_trained"], len(metrics["history"]["loss"]) - 1)
        self.assertIn("stopped_early", metrics)

    def test_saved_checkpoint_loads_and_matches_validation_metrics(self) -> None:
        metrics = train_dataset(
            "iris",
            epochs=5,
            hidden_dim=4,
            learning_rate=0.05,
            batch_size=16,
            verbose=False,
            return_artifacts=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "iris_model.npz"
            save_checkpoint(
                metrics["model"],
                dataset="iris",
                meta=metrics["meta"],
                config=metrics["config"],
                path=checkpoint,
            )
            loaded_metrics = evaluate_checkpoint(checkpoint, split="val")

        self.assertEqual(loaded_metrics["dataset"], "iris")
        self.assertEqual(loaded_metrics["split"], "val")
        self.assertAlmostEqual(
            loaded_metrics["accuracy"],
            metrics["final_val_accuracy"],
            places=12,
        )
        self.assertAlmostEqual(
            loaded_metrics["loss"],
            metrics["history"]["val_loss"][-1],
            places=12,
        )


class DataPreparationTests(unittest.TestCase):
    def test_prepare_iris_split_standardizes_and_encodes_labels(self) -> None:
        X_train, y_train, meta = prepare_iris_split("train")
        X_val, y_val, _ = prepare_iris_split("val", meta=meta)

        self.assertEqual(X_train.shape, (120, 4))
        self.assertEqual(y_train.shape, (120,))
        self.assertEqual(X_val.shape, (30, 4))
        self.assertEqual(y_val.shape, (30,))
        self.assertEqual(set(y_train.tolist()), {0, 1, 2})
        np.testing.assert_allclose(X_train.mean(axis=0), np.zeros(4), atol=1e-8)

    def test_prepare_mnist_split_flattens_scales_and_limits_samples(self) -> None:
        X, y, _ = prepare_mnist_split("train", limit=32)

        self.assertEqual(X.shape, (32, 784))
        self.assertEqual(y.shape, (32,))
        self.assertGreaterEqual(float(X.min()), 0.0)
        self.assertLessEqual(float(X.max()), 1.0)
        self.assertTrue(np.issubdtype(y.dtype, np.integer))

    def test_prepare_mnist_test_split_loads_official_t10k_samples(self) -> None:
        X, y, _ = prepare_mnist_split("test", limit=64)

        self.assertEqual(X.shape, (64, 784))
        self.assertEqual(y.shape, (64,))
        self.assertGreaterEqual(float(X.min()), 0.0)
        self.assertLessEqual(float(X.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
