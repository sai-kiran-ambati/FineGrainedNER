"""MultiCoNER v2 fine labels and their TeluguNER coarse projections."""

import json
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MULTICONER_LABELS_FILE = PROJECT_ROOT / "prompts" / "multiconer_v2_fine_labels.json"

# These TeluguNER classes have no corresponding MultiCoNER v2 fine label.
# They remain available as direct output labels so existing gold entities
# can still be represented and evaluated without forcing a false subtype.
TELUGU_ONLY_LABELS = {"CARDINAL", "TIME", "ORGANISM", "OTH"}


@lru_cache(maxsize=1)
def load_fine_labels():
    with MULTICONER_LABELS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def all_output_labels():
    return frozenset(load_fine_labels()) | frozenset(TELUGU_ONLY_LABELS)


def map_to_telugu_coarse(label):
    label = label.upper()
    if label in TELUGU_ONLY_LABELS:
        return label
    definition = load_fine_labels().get(label)
    if definition is None:
        return None
    return definition["telugu_coarse"]


def multiconer_coarse(label):
    definition = load_fine_labels().get(label.upper())
    return definition["multiconer_coarse"] if definition else None
