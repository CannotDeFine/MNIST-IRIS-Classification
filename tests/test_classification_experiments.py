from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from classification_experiments import (  # noqa: E402
    ExperimentConfig,
    SoftmaxRegressionClassifier,
    build_experiment_configs,
    run_experiment,
)


class SoftmaxRegressionClassifierTests(unittest.TestCase):
    def test_linear_softmax_learns_simple_problem(self) -> None:
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

        model = SoftmaxRegressionClassifier(
            input_dim=2,
            output_dim=3,
            learning_rate=0.2,
            seed=11,
        )
        history = model.fit(X, y, epochs=200, batch_size=3)

        self.assertLess(history["loss"][-1], history["loss"][0])
        self.assertGreaterEqual(model.accuracy(X, y), 0.95)


class ExperimentRunnerTests(unittest.TestCase):
    def test_build_configs_includes_current_and_ablation_variants(self) -> None:
        configs = build_experiment_configs("mnist", profile="quick")
        names = [config.name for config in configs]

        self.assertIn("mnist_linear_softmax", names)
        self.assertIn("mnist_current_mlp", names)
        self.assertIn("mnist_no_pixel_scaling", names)
        self.assertIn("mnist_small_normal_init", names)

    def test_run_experiment_reports_initial_and_final_metrics(self) -> None:
        config = ExperimentConfig(
            name="mnist_test_smoke",
            dataset="mnist",
            model_type="linear",
            description="tiny smoke run",
            ablates="hidden ReLU layer",
            epochs=1,
            learning_rate=0.1,
            batch_size=16,
            train_limit=32,
            val_limit=32,
            scale_inputs=True,
            seed=5,
        )

        result = run_experiment(config, verbose=False)

        self.assertEqual(result["name"], "mnist_test_smoke")
        self.assertEqual(result["config"]["epochs"], 1)
        self.assertEqual(len(result["history"]["accuracy"]), 2)
        self.assertIn("initial_val_accuracy", result)
        self.assertIn("final_val_accuracy", result)


if __name__ == "__main__":
    unittest.main()
