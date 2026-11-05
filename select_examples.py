import json
import random


TRAIN_FILE = "data/train.json"
OUTPUT_FILE = "data/fewshot_5.json"

SEED = 42

TARGET_TYPES = [
    "PER",
    "ORG",
    "LOC",
    "CARDINAL",
    "TIME"
]


def get_entity_types(item):
    return set(
        entity["label"]
        for entity in item["entities"]
    )


def score_example(item, target_type):

    types = get_entity_types(item)

    if target_type not in types:
        return -1

    num_entities = len(item["entities"])

    # Prefer reasonably short demonstrations
    if num_entities <= 1:
        entity_score = 1
    elif num_entities <= 5:
        entity_score = 10
    elif num_entities <= 8:
        entity_score = 5
    else:
        entity_score = 1

    # Prefer sentences containing the target
    target_score = 10

    # Slight preference for examples
    # containing more than one type
    diversity_score = min(len(types), 4) * 2

    return (
        target_score
        + entity_score
        + diversity_score
    )


def main():

    with open(
        TRAIN_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        train_data = json.load(f)

    random.seed(SEED)

    selected = []
    used_sentence_ids = set()

    for target_type in TARGET_TYPES:

        candidates = []

        for item in train_data:

            if item["sentence_id"] in used_sentence_ids:
                continue

            score = score_example(
                item,
                target_type
            )

            if score >= 0:
                candidates.append(
                    (score, item)
                )

        if not candidates:

            print(
                f"WARNING: "
                f"No example found for {target_type}"
            )

            continue

        candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        # Look at the best 50 candidates
        top_candidates = candidates[:50]

        _, selected_item = random.choice(
            top_candidates
        )

        selected.append(selected_item)

        used_sentence_ids.add(
            selected_item["sentence_id"]
        )

    # --------------------------------------------
    # Print examples
    # --------------------------------------------

    print("=" * 70)
    print("SELECTED 5-SHOT EXAMPLES")
    print("=" * 70)

    for i, item in enumerate(
        selected,
        start=1
    ):

        print(f"\nExample {i}")

        print(
            "Sentence ID:",
            item["sentence_id"]
        )

        print("\nSentence:")
        print(item["text"])

        print("\nEntities:")

        for entity in item["entities"]:

            print(
                f"  {entity['text']} "
                f"→ {entity['label']}"
            )

        print("\nEntity types:")

        print(
            sorted(
                get_entity_types(item)
            )
        )

        print(
            "\nNumber of entities:",
            len(item["entities"])
        )

        print("-" * 70)

    # --------------------------------------------
    # Save
    # --------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            selected,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print(
        "Number of demonstrations:",
        len(selected)
    )


if __name__ == "__main__":
    main()