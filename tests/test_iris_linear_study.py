from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iris_linear_study import (  # noqa: E402
    MulticlassPerceptron,
    decision_boundary_grid,
    run_study,
)


class MulticlassPerceptronTests(unittest.TestCase):
    def _separable_three_class(self) -> tuple[np.ndarray, np.ndarray]:
        # Three well-separated 2D clusters -> linearly separable.
        rng = np.random.default_rng(0)
        centers = np.array([[-4.0, 0.0], [4.0, 0.0], [0.0, 5.0]])
        X = np.vstack([c + 0.2 * rng.standard_normal((20, 2)) for c in centers])
        y = np.repeat([0, 1, 2], 20)
        return X, y

    def test_learns_linearly_separable_data(self) -> None:
        X, y = self._separable_three_class()
        model = MulticlassPerceptron(input_dim=2, output_dim=3, seed=1)
        history = model.fit(X, y, epochs=50)
        self.assertEqual(model.accuracy(X, y), 1.0)
        # Convergence: final mistake count is zero and it dropped from the start.
        self.assertEqual(history["errors"][-1], 0)
        self.assertGreaterEqual(history["errors"][0], history["errors"][-1])

    def test_no_update_when_prediction_correct(self) -> None:
        model = MulticlassPerceptron(input_dim=2, output_dim=2, learning_rate=1.0, seed=0)
        # Make class 1 strictly preferred for this sample.
        model.W = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        x = np.array([2.0, 2.0])
        before = model.W.copy()
        updated = model._update_sample(x, 1)
        self.assertFalse(updated)
        np.testing.assert_array_equal(model.W, before)

    def test_update_moves_scores_toward_true_class(self) -> None:
        model = MulticlassPerceptron(input_dim=2, output_dim=2, learning_rate=1.0, seed=0)
        x = np.array([1.0, 1.0])
        # Zero init -> tie broken toward class 0, so true class 1 is "wrong".
        score_true_before = model.decision_function(x.reshape(1, -1))[0, 1]
        updated = model._update_sample(x, 1)
        score_true_after = model.decision_function(x.reshape(1, -1))[0, 1]
        self.assertTrue(updated)
        self.assertGreater(score_true_after, score_true_before)


class DecisionBoundaryTests(unittest.TestCase):
    def test_grid_shapes_match(self) -> None:
        X = np.array([[-1.0, -1.0], [1.0, 1.0], [0.0, 2.0], [2.0, 0.0]])
        model = MulticlassPerceptron(input_dim=2, output_dim=3, seed=0)
        model.fit(X, np.array([0, 1, 2, 1]), epochs=5)
        xx, yy, Z = decision_boundary_grid(model, X, resolution=30)
        self.assertEqual(xx.shape, yy.shape)
        self.assertEqual(Z.shape, xx.shape)
        self.assertEqual(xx.shape, (30, 30))


class RunStudyTests(unittest.TestCase):
    def test_run_study_writes_json_and_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "iris_linear_study.json"
            figure_dir = tmp_path / "figures"
            paths = run_study(
                output=output,
                figure_dir=figure_dir,
                seed=42,
                epochs_perceptron=30,
                epochs_linear=80,
                epochs_mlp=80,
                hidden_dim=16,
                quiet=True,
            )
            self.assertTrue(output.exists())
            for name in (
                "iris_linear_vs_mlp_accuracy.png",
                "iris_linear_decision_boundary.png",
                "iris_linear_convergence.png",
            ):
                self.assertTrue((figure_dir / name).exists(), f"missing {name}")
            self.assertIn(output, paths)

            items = json.loads(output.read_text(encoding="utf-8"))
            names = {item["name"] for item in items}
            self.assertEqual(
                names,
                {"iris_perceptron", "iris_linear_softmax", "iris_mlp"},
            )
            softmax = next(i for i in items if i["name"] == "iris_linear_softmax")
            self.assertGreaterEqual(softmax["final_val_accuracy"], 0.9)


if __name__ == "__main__":
    unittest.main()
