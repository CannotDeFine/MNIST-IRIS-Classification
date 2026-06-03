"""V4: focused component checks."""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from versions.common import build_version_parser, run_component_suites  # noqa: E402


SUITES = ["loss-comparison", "l2-sweep", "lr-sweep", "width-sweep", "training-strategies"]


def main() -> None:
    parser = build_version_parser("Run V4 component-check suites.")
    parser.add_argument(
        "--suite",
        action="append",
        choices=SUITES,
        help="Run only one suite. Can be passed multiple times.",
    )
    args = parser.parse_args()
    run_component_suites(
        "v4_component_checks",
        args.suite or SUITES,
        dataset=args.dataset,
        profile=args.profile,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
