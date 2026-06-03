"""V3: stable MLP training components."""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from versions.common import build_version_parser, run_version_configs  # noqa: E402


EXPERIMENTS = [
    "iris_no_standardization",
    "iris_small_normal_init",
    "iris_current_mlp",
    "mnist_no_pixel_scaling",
    "mnist_full_batch",
    "mnist_small_normal_init",
    "mnist_current_mlp",
]


def main() -> None:
    parser = build_version_parser("Run V3 stable-MLP experiments.")
    args = parser.parse_args()
    run_version_configs(
        "v3_stable_mlp",
        EXPERIMENTS,
        dataset=args.dataset,
        profile=args.profile,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

