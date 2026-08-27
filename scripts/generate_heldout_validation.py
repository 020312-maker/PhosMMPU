"""Generate held-out validation figures from completed seed runs."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.multimodal.result_validation import generate_validation_report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate five-seed held-out validation figures")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=[11, 23, 37, 51, 73])
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    run_root = Path(args.run_root)
    output = generate_validation_report(
        run_root,
        args.expected_seeds,
        Path(args.output_dir) if args.output_dir else run_root / "results_validation",
    )
    print(output)


if __name__ == "__main__":
    main()
