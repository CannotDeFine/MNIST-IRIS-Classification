from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from classification_experiments import (  # noqa: E402
    ExperimentConfig,
    SigmoidMSEMLPClassifier,
    SoftmaxRegressionClassifier,
    build_experiment_configs,
    build_l2_sweep_configs,
    build_learning_rate_sweep_configs,
    build_loss_comparison_configs,
    build_training_strategy_configs,
    build_width_sweep_configs,
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


class SigmoidMSEMLPClassifierTests(unittest.TestCase):
    def test_sigmoid_mse_mlp_learns_simple_problem(self) -> None:
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

        model = SigmoidMSEMLPClassifier(
            input_dim=2,
            hidden_dim=8,
            output_dim=3,
            learning_rate=0.2,
            seed=13,
        )
        history = model.fit(X, y, epochs=250, batch_size=3)

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
        self.assertIn("mnist_l2_regularization", names)

    def test_build_l2_sweep_configs_uses_multiple_strengths(self) -> None:
        configs = build_l2_sweep_configs("mnist", profile="quick")
        strengths = [config.l2 for config in configs]

        self.assertEqual(strengths, [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        self.assertTrue(all(config.model_type == "mlp" for config in configs))

    def test_build_lr_and_width_sweeps_use_expected_values(self) -> None:
        lr_configs = build_learning_rate_sweep_configs("iris", profile="quick")
        width_configs = build_width_sweep_configs("mnist", profile="quick")

        self.assertEqual([config.learning_rate for config in lr_configs], [0.001, 0.01, 0.05, 0.1])
        self.assertEqual([config.hidden_dim for config in width_configs], [32, 64, 128, 256])

    def test_build_training_strategy_configs_includes_momentum(self) -> None:
        configs = build_training_strategy_configs("mnist", profile="quick")
        names = [config.name for config in configs]

        self.assertIn("mnist_strategy_momentum", names)
        self.assertTrue(any(config.optimizer == "momentum" for config in configs))

    def test_build_loss_comparison_configs_includes_both_losses(self) -> None:
        configs = build_loss_comparison_configs("iris", profile="quick")
        names = [config.name for config in configs]
        model_types = [config.model_type for config in configs]

        self.assertIn("iris_sigmoid_mse", names)
        self.assertIn("iris_softmax_cross_entropy", names)
        self.assertEqual(model_types, ["mlp_mse", "mlp"])

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
            l2=0.001,
        )

        result = run_experiment(config, verbose=False)

        self.assertEqual(result["name"], "mnist_test_smoke")
        self.assertEqual(result["config"]["epochs"], 1)
        self.assertEqual(result["config"]["l2"], 0.001)
        self.assertEqual(result["epochs_trained"], 1)
        self.assertFalse(result["stopped_early"])
        self.assertEqual(len(result["history"]["accuracy"]), 2)
        self.assertIn("initial_val_accuracy", result)
        self.assertIn("final_val_accuracy", result)


if __name__ == "__main__":
    unittest.main()
