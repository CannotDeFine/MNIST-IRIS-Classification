"""V1: linear softmax baseline."""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from versions.common import build_version_parser, run_version_configs  # noqa: E402


EXPERIMENTS = ["iris_linear_softmax", "mnist_linear_softmax"]


def main() -> None:
    parser = build_version_parser("Run V1 linear softmax experiments.")
    args = parser.parse_args()
    run_version_configs(
        "v1_linear_softmax",
        EXPERIMENTS,
        dataset=args.dataset,
        profile=args.profile,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

