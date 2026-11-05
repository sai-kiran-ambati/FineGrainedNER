import pandas as pd
import json
from pathlib import Path


INPUT_FILE = "anonymous_telner/dataset/dataset-generic.csv"
OUTPUT_FILE = "data/teluguner_generic.json"


def reconstruct_sentence(words):
    """
    Reconstruct a readable sentence from TeluguNER tokens.
    """

    text = ""

    no_space_before = {
        ".", ",", ":", ";", "!", "?", ")", "]", "}",
        "।", "॥"
    }

    no_space_after = {
        "(", "[", "{"
    }

    for word in words:

        if not text:
            text = word

        elif word in no_space_before:
            text += word

        elif text[-1:] in no_space_after:
            text += word

        else:
            text += " " + word

    return text


def extract_entities(rows):

    entities = []

    current_tokens = []
    current_label = None
    start_token = None
    last_token = None

    for row in rows:

        word = row["word"]
        tag = row["tag"]
        word_index = int(row["word_index"])

        # -----------------------------------------
        # Beginning of a new entity
        # -----------------------------------------

        if tag.startswith("B-"):

            # Save previous entity
            if current_tokens:

                entities.append({
                    "text": " ".join(current_tokens),
                    "label": current_label,
                    "start_token": start_token,
                    "end_token": last_token
                })

            current_tokens = [word]
            current_label = tag[2:]
            start_token = word_index
            last_token = word_index

        # -----------------------------------------
        # Continuation of current entity
        # -----------------------------------------

        elif (
            tag.startswith("I-")
            and current_tokens
            and current_label == tag[2:]
        ):

            current_tokens.append(word)
            last_token = word_index

        # -----------------------------------------
        # Malformed I-X
        # -----------------------------------------

        elif tag.startswith("I-"):

            # Save previous entity
            if current_tokens:

                entities.append({
                    "text": " ".join(current_tokens),
                    "label": current_label,
                    "start_token": start_token,
                    "end_token": last_token
                })

            # Treat malformed I-X as a new entity
            current_tokens = [word]
            current_label = tag[2:]
            start_token = word_index
            last_token = word_index

        # -----------------------------------------
        # Outside entity
        # -----------------------------------------

        else:

            if current_tokens:

                entities.append({
                    "text": " ".join(current_tokens),
                    "label": current_label,
                    "start_token": start_token,
                    "end_token": last_token
                })

                current_tokens = []
                current_label = None
                start_token = None
                last_token = None

    # -----------------------------------------
    # Save final entity
    # -----------------------------------------

    if current_tokens:

        entities.append({
            "text": " ".join(current_tokens),
            "label": current_label,
            "start_token": start_token,
            "end_token": last_token
        })

    return entities


def prepare_dataset():

    df = pd.read_csv(INPUT_FILE)

    dataset = []

    for sentence_id, group in df.groupby("sentence_id"):

        group = group.sort_values("word_index")

        words = group["word"].tolist()

        # Reconstruct readable sentence
        text = reconstruct_sentence(words)

        # Extract entities directly from original rows
        entities = extract_entities(
            group.to_dict("records")
        )

        # Preserve original tokens
        tokens = words

        dataset.append({
            "sentence_id": int(sentence_id),
            "text": text,
            "tokens": tokens,
            "entities": entities
        })

    return dataset


if __name__ == "__main__":

    Path("data").mkdir(exist_ok=True)

    dataset = prepare_dataset()

    # ---------------------------------------------
    # Check for missing labels
    # ---------------------------------------------

    missing_labels = []

    for item in dataset:

        for entity in item["entities"]:

            if entity["label"] is None:

                missing_labels.append({
                    "sentence_id": item["sentence_id"],
                    "entity": entity
                })

    print("\nDataset validation")
    print("=" * 60)

    print("Total sentences:", len(dataset))
    print("Missing-label entities:", len(missing_labels))

    # ---------------------------------------------
    # Save
    # ---------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dataset,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nDataset prepared successfully.")
    print("Saved to:", OUTPUT_FILE)

    # ---------------------------------------------
    # Display first 3 examples
    # ---------------------------------------------

    print("\nFirst 3 examples:\n")

    for item in dataset[:3]:

        print("=" * 70)

        print(
            "Sentence ID:",
            item["sentence_id"]
        )

        print("Text:", item["text"])

        print("Entities:")

        for entity in item["entities"]:

            print(
                f"  {entity['text']} → "
                f"{entity['label']}"
            )