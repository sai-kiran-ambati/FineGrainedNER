from pathlib import Path
import json

from src.fine_grained_taxonomy import (
    load_fine_labels,
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFINITIONS_FILE = PROJECT_ROOT / "prompts" / "telugu_definitions.txt"
EXAMPLES_FILE = PROJECT_ROOT / "prompts" / "few_shot_examples.json"
TELUGU_FINE_LABELS_FILE = PROJECT_ROOT / "prompts" / "multiconer_v2_fine_labels_te.json"


# --------------------------------------------------
# Load definitions
# --------------------------------------------------

def load_definitions():
    with open(DEFINITIONS_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


# --------------------------------------------------
# Load few-shot examples
# --------------------------------------------------

def load_examples():
    with open(EXAMPLES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------
# Format one few-shot example
# --------------------------------------------------

def format_example(example):
    lines = []

    lines.append(f"Text: {example['text']}")
    lines.append("Entities:")

    for entity in example["entities"]:
        lines.append(
            f"- {entity['text']} -> {entity['label']}"
        )

    return "\n".join(lines)


# --------------------------------------------------
# Build PromptNER prompt
# --------------------------------------------------

def build_prompt(sentence, candidates):
    definitions = load_definitions()
    examples = load_examples()

    prompt_parts = []

    # ----------------------------------------------
    # Task
    # ----------------------------------------------

    prompt_parts.append(
        """You are an expert Named Entity Recognition system for Telugu.

Your task is to identify which candidate phrases in a Telugu sentence
are named entities and assign the correct entity type.

Use the entity definitions and examples provided below.

For every candidate:
1. Decide whether it is an entity.
2. If it is an entity, assign its entity type.
3. Give a short explanation for the decision.

Do not invent entities that are not present in the candidate list.
"""
    )

    # ----------------------------------------------
    # Entity definitions
    # ----------------------------------------------

    prompt_parts.append(
        "\nENTITY DEFINITIONS\n"
        "===================\n"
        + definitions
    )

    # ----------------------------------------------
    # Few-shot demonstrations
    # ----------------------------------------------

    prompt_parts.append(
        "\nFEW-SHOT EXAMPLES\n"
        "=================="
    )

    for i, example in enumerate(examples, start=1):
        prompt_parts.append(
            f"\nExample {i}\n"
            f"---------\n"
            + format_example(example)
        )

    # ----------------------------------------------
    # Current sentence
    # ----------------------------------------------

    prompt_parts.append(
        "\nINPUT SENTENCE\n"
        "==============\n"
        f"{sentence}"
    )

    # ----------------------------------------------
    # Candidate entities
    # ----------------------------------------------

    prompt_parts.append(
        "\nCANDIDATE ENTITIES\n"
        "=================="
    )

    for i, candidate in enumerate(candidates, start=1):

        if isinstance(candidate, dict):

            prompt_parts.append(
                f"C{i:03d} | "
                f"{candidate['text']} | "
                f"tokens {candidate['start_token']}-"
                f"{candidate['end_token']}"
            )

        else:

            prompt_parts.append(
                f"C{i:03d} | {candidate}"
            )


    # --------------------------------------------------
    # Required output format
    # --------------------------------------------------

    prompt_parts.append(
        """You are an expert Named Entity Recognition system for Telugu.

    Your task is to identify the exact candidate spans that correspond to
    named entities in the Telugu sentence and assign the correct entity type.

    Use the entity definitions and few-shot examples provided below.

    For every candidate:

    1. Decide whether the candidate is an entity.
    2. If it is an entity, assign exactly one entity type.
    3. Give a short explanation for the decision.
    4. If it is not an entity, output False and Type NONE.

    IMPORTANT RULES FOR ENTITY SPANS:

    - NER requires the exact entity span, not just a phrase that contains
    an entity.
    - When candidate spans overlap, do not mark multiple overlapping
    candidates as True unless they represent genuinely separate entities.
    - Prefer the candidate that represents the complete entity span.
    - Do not extend an entity with surrounding ordinary words.
    - Do not shorten an entity when the complete entity is present as a
    candidate.
    - Use the context of the complete sentence to determine the correct
    boundary.
    - The Candidate ID identifies the exact candidate span.
    - Do not invent entities that are not present in the candidate list.

    """
    )

    return "\n".join(prompt_parts)


def build_fine_grained_prompt(
    sentence, candidates, *, definition_language="english", include_examples=True
):
    """Build PromptNER input with MultiCoNER v2 fine labels.

    The existing TeluguNER definitions and BIO-derived few-shot examples are
    retained. Fine labels project to TeluguNER classes for evaluation; labels
    without a MultiCoNER counterpart remain direct TeluguNER outputs.
    """
    telugu_definitions = load_definitions()
    examples = load_examples() if include_examples else []
    fine_labels = load_fine_labels()
    if definition_language == "telugu":
        with TELUGU_FINE_LABELS_FILE.open("r", encoding="utf-8") as f:
            fine_definitions_te = json.load(f)
    elif definition_language == "english":
        fine_definitions_te = None
    else:
        raise ValueError("definition_language must be 'english' or 'telugu'.")

    parts = [
        "You are a fine-grained Named Entity Recognition system for Telugu.\n"
        "Identify exact named-entity spans among the candidate phrases. Use the\n"
        "definitions below. Choose one fine label for each entity. For candidates\n"
        "that are not entities, choose NONE. Never invent a candidate or label.\n"
        "\nMULTICONER V2 FINE-GRAINED LABELS\n"
        "=================================",
    ]
    for label, item in fine_labels.items():
        definition = (
            fine_definitions_te[label]
            if fine_definitions_te is not None
            else item["definition"]
        )
        parts.append(
            f"{label}: {definition} "
            f"(MultiCoNER coarse: {item['multiconer_coarse']}; "
            f"TeluguNER coarse: {item['telugu_coarse']})"
        )

    parts.extend([
        "\nTELUGUNER COARSE LABEL DEFINITIONS\n"
        "===================================\n" + telugu_definitions,
        "\nTELUGUNER CLASSES WITHOUT A MULTICONER V2 FINE SUBTYPE\n"
        "=====================================================\n"
        "CARDINAL: A number or count expression annotated CARDINAL.\n"
        "TIME: A date, year, day, month, clock time, or duration annotated TIME.\n"
        "ORGANISM: A named living organism, animal, plant, or other living being.\n"
        "OTH: An important named entity that fits none of the fine labels above.\n"
        "Use these direct TeluguNER labels only when no MultiCoNER fine label fits.",
    ])
    if include_examples:
        parts.append(
            "\nFEW-SHOT EXAMPLES (coarse-label demonstrations)\n"
            "================================================\n"
            "These examples demonstrate Telugu span boundaries and coarse class "
            "semantics only. Do not copy their label names into the output; choose "
            "a fine label from the MultiCoNER taxonomy whenever one fits."
        )
        parts.extend(
            f"Example {i}\nText: {example['text']}\nEntities:\n" +
            "\n".join(f"- {entity['text']} -> {entity['label']}" for entity in example["entities"])
            for i, example in enumerate(examples, start=1)
        )
    parts.extend([
        "\nINPUT SENTENCE\n==============\n" + sentence,
        "\nCANDIDATE ENTITIES\n==================",
    ])
    for i, candidate in enumerate(candidates, start=1):
        parts.append(
            f"C{i:03d} | {candidate['text']} | tokens "
            f"{candidate['start_token']}-{candidate['end_token']}"
        )
    parts.append(
        "\nOUTPUT FORMAT\n============="
        "\nReturn exactly one row for every candidate, in this format:"
        "\n| Candidate | Entity | Fine label |"
        "\n| C001 | True | POLITICIAN |"
        "\n| C002 | False | NONE |"
        "\nUse only the 33 listed MultiCoNER fine labels plus CARDINAL, TIME, "
        "ORGANISM, OTH, and NONE. Do not return MultiCoNER coarse labels."
        "\nFor overlapping candidates, mark only the exact complete entity span."
    )
    return "\n\n".join(parts)


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    test_sentence = (
        "తిరుమలలో సోమవారం కొత్త కార్యక్రమం ప్రారంభమైంది."
    )

    test_candidates = [
        "తిరుమల",
        "సోమవారం",
        "కొత్త కార్యక్రమం",
        "కార్యక్రమం"
    ]

    prompt = build_prompt(
        test_sentence,
        test_candidates
    )

    print("=" * 80)
    print(prompt)
    print("=" * 80)
