"""Final selected MNIST MLP."""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from versions.common import build_final_parser, run_final_mnist  # noqa: E402


def main() -> None:
    parser = build_final_parser()
    args = parser.parse_args()
    run_final_mnist(args)


if __name__ == "__main__":
    main()

