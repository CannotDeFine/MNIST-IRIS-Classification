"""V0: untrained random-network baseline."""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from versions.common import build_version_parser, run_version_configs  # noqa: E402


EXPERIMENTS = ["iris_untrained_mlp", "mnist_untrained_mlp"]


def main() -> None:
    parser = build_version_parser("Run V0 untrained random-network experiments.")
    args = parser.parse_args()
    run_version_configs(
        "v0_untrained_random",
        EXPERIMENTS,
        dataset=args.dataset,
        profile=args.profile,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

