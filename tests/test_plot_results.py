from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plot_results import plot_metric_comparison  # noqa: E402


class PlotResultsTests(unittest.TestCase):
    def test_plot_metric_comparison_writes_summary_and_figures(self) -> None:
        payload = [
            {
                "name": "model_a",
                "dataset": "mnist",
                "final_val_accuracy": 0.91,
                "final_train_accuracy": 0.93,
                "final_val_loss": 0.2,
                "final_train_loss": 0.1,
                "epochs_trained": 5,
                "stopped_early": False,
            },
            {
                "name": "model_b",
                "dataset": "mnist",
                "final_val_accuracy": 0.94,
                "final_train_accuracy": 0.96,
                "final_val_loss": 0.15,
                "final_train_loss": 0.08,
                "history": {"loss": [1.0, 0.5, 0.3]},
                "stopped_early": True,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metrics_path = tmp / "metrics.json"
            metrics_path.write_text(json.dumps(payload), encoding="utf-8")

            outputs = plot_metric_comparison(
                metrics_paths=[metrics_path],
                output_dir=tmp,
                output_prefix="comparison",
                dataset="mnist",
            )

            self.assertEqual(len(outputs), 3)
            for path in outputs:
                self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
