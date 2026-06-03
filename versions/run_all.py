"""Run all roadmap-version experiment groups."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from versions.common import run_component_suites, run_version_configs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run roadmap-version experiment groups.")
    parser.add_argument("--dataset", choices=["iris", "mnist", "all"], default="all")
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_version_configs(
        "v0_untrained_random",
        ["iris_untrained_mlp", "mnist_untrained_mlp"],
        dataset=args.dataset,
        profile=args.profile,
    )
    run_version_configs(
        "v1_linear_softmax",
        ["iris_linear_softmax", "mnist_linear_softmax"],
        dataset=args.dataset,
        profile=args.profile,
    )
    run_version_configs(
        "v2_small_mlp",
        ["iris_small_hidden", "mnist_small_hidden"],
        dataset=args.dataset,
        profile=args.profile,
    )
    run_version_configs(
        "v3_stable_mlp",
        [
            "iris_no_standardization",
            "iris_small_normal_init",
            "iris_current_mlp",
            "mnist_no_pixel_scaling",
            "mnist_full_batch",
            "mnist_small_normal_init",
            "mnist_current_mlp",
        ],
        dataset=args.dataset,
        profile=args.profile,
    )
    run_component_suites(
        "v4_component_checks",
        ["loss-comparison", "l2-sweep", "lr-sweep", "width-sweep", "training-strategies"],
        dataset=args.dataset,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()
