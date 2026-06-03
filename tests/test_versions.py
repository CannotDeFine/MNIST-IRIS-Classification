from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from versions.common import run_version_configs, selected_ablation_configs  # noqa: E402


class VersionRunnerTests(unittest.TestCase):
    def test_selected_ablation_configs_filters_by_name(self) -> None:
        configs = selected_ablation_configs(
            dataset="mnist",
            profile="quick",
            names=["mnist_untrained_mlp", "mnist_linear_softmax"],
        )

        self.assertEqual([config.name for config in configs], ["mnist_untrained_mlp", "mnist_linear_softmax"])

    def test_run_version_configs_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            results = run_version_configs(
                "v0_untrained_random",
                ["mnist_untrained_mlp"],
                dataset="mnist",
                profile="quick",
                output_dir=output_dir,
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["name"], "mnist_untrained_mlp")
            self.assertTrue((output_dir / "quick.json").exists())
            self.assertFalse((output_dir / "quick.md").exists())


if __name__ == "__main__":
    unittest.main()
