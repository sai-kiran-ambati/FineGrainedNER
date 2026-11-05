"""Run a detailed token-wise matching report for saved fine-grained results."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.matching_analysis import main_for


if __name__ == "__main__":
    main_for("token_wise")
