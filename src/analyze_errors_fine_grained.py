"""Print saved scores, gold entities, true positives, and entity errors."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fine_grained_evaluation import find_results_file, print_saved_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "sarvam"))
    parser.add_argument("--sentences", type=int, help="Select results by sentence count")
    parser.add_argument("--file", type=Path, help="Explicit results JSON path")
    args = parser.parse_args()
    print_saved_summary(
        args.file or find_results_file(args.provider, args.sentences)
    )


if __name__ == "__main__":
    main()
