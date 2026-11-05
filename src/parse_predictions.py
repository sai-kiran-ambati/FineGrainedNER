import re

from src.fine_grained_taxonomy import (
    all_output_labels,
    map_to_telugu_coarse,
    multiconer_coarse,
)


def build_candidate_lookup(candidates):
    lookup = {}

    for i, candidate in enumerate(candidates, start=1):
        lookup[f"C{i:03d}"] = candidate

    return lookup


def parse_gpt_output(response_text, candidates):
    """
    Parse PromptNER output from a chat model.

    Supported format:

    | Candidate | Entity | Type | Explanation |
    |---|---|---|---|
    | C001 | True | OTH | ... |
    | C002 | False | NONE | ... |

    Also supports the older format:

    Candidate ID: C001
    True/False: True
    Type: LOC
    """

    candidate_lookup = build_candidate_lookup(candidates)

    predictions = []

    # ==============================================================
    # FORMAT 1: Markdown table
    # ==============================================================
    # Accept both Markdown table rows and plain pipe-delimited rows. Sarvam
    # often omits the leading/trailing pipes and Markdown header separator.
    table_pattern = re.compile(
        r"^\s*\|?\s*(C\d+)\s*\|\s*"
        r"(True|False)\s*\|\s*"
        r"(NONE|[A-Z_]+)\s*\|",
        re.IGNORECASE | re.MULTILINE,
    )

    table_matches = table_pattern.findall(response_text)

    for candidate_id, decision, label in table_matches:

        candidate_id = candidate_id.upper()
        decision = decision.lower()
        label = label.upper()

        # Only keep positive entity predictions
        if decision != "true":
            continue

        if label == "NONE":
            continue

        # Ignore hallucinated candidate IDs
        if candidate_id not in candidate_lookup:
            continue

        candidate = candidate_lookup[candidate_id]

        predictions.append({
            "candidate_id": candidate_id,
            "text": candidate["text"],
            "label": label,
            "start_token": candidate["start_token"],
            "end_token": candidate["end_token"],
        })

    # ==============================================================
    # FORMAT 2: Candidate ID / True-False / Type format
    # ==============================================================
    if not table_matches:

        pattern1 = re.compile(
            r"Candidate ID:\s*(C\d+)"
            r".*?"
            r"True/False:\s*(True|False)"
            r".*?"
            r"Type:\s*([A-Z_]+)",
            re.IGNORECASE | re.DOTALL,
        )

        matches = pattern1.findall(response_text)

        for candidate_id, decision, label in matches:

            candidate_id = candidate_id.upper()
            decision = decision.lower()
            label = label.upper()

            if decision != "true":
                continue

            if label == "NONE":
                continue

            if candidate_id not in candidate_lookup:
                continue

            candidate = candidate_lookup[candidate_id]

            predictions.append({
                "candidate_id": candidate_id,
                "text": candidate["text"],
                "label": label,
                "start_token": candidate["start_token"],
                "end_token": candidate["end_token"],
            })

    return predictions


def parse_fine_grained_output(response_text, candidates):
    """Parse fine-label rows and attach their two coarse-level mappings."""
    candidate_lookup = build_candidate_lookup(candidates)
    allowed_labels = all_output_labels()
    pattern = re.compile(
        r"^\s*\|?\s*(C\d+)\s*\|\s*"
        r"(True|False)\s*\|\s*"
        r"([A-Z_]+|NONE)(?:\s*\||\s*$)",
        re.IGNORECASE | re.MULTILINE,
    )

    predictions = []
    for candidate_id, decision, fine_label in pattern.findall(response_text):
        candidate_id = candidate_id.upper()
        fine_label = fine_label.upper()

        if decision.lower() != "true" or fine_label == "NONE":
            continue
        if candidate_id not in candidate_lookup or fine_label not in allowed_labels:
            continue

        candidate = candidate_lookup[candidate_id]
        predictions.append({
            "candidate_id": candidate_id,
            "text": candidate["text"],
            "fine_label": fine_label,
            "multiconer_coarse": multiconer_coarse(fine_label),
            "label": map_to_telugu_coarse(fine_label),
            "start_token": candidate["start_token"],
            "end_token": candidate["end_token"],
        })
    return predictions
