"""V2: small one-hidden-layer MLP."""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from versions.common import build_version_parser, run_version_configs  # noqa: E402


EXPERIMENTS = ["iris_small_hidden", "mnist_small_hidden"]


def main() -> None:
    parser = build_version_parser("Run V2 small-MLP experiments.")
    args = parser.parse_args()
    run_version_configs(
        "v2_small_mlp",
        EXPERIMENTS,
        dataset=args.dataset,
        profile=args.profile,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

