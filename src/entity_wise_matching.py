"""Run a detailed exact entity-wise matching report."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.matching_analysis import main_for


if __name__ == "__main__":
    main_for("entity_wise")
