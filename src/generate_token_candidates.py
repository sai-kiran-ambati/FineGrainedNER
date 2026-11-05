from pathlib import Path
import pandas as pd
import unicodedata
import re


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_FILE = (
    PROJECT_ROOT
    / "anonymous_telner"
    / "dataset"
    / "dataset-generic.csv"
)


def normalize_for_matching(text):
    """
    Normalize text only for matching gold entities
    with generated candidates.

    Original dataset text is never modified.
    """

    text = unicodedata.normalize("NFC", text)

    invisible_chars = [
        "\u200b",
        "\u200c",
        "\u200d",
        "\u2060",
        "\ufeff"
    ]

    for char in invisible_chars:
        text = text.replace(char, "")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_data():
    return pd.read_csv(
        CSV_FILE,
        dtype={
            "sentence_id": int,
            "word_index": int,
            "word": str,
            "tag": str
        }
    )


def get_sentence(df, sentence_id):

    sentence = df[
        df["sentence_id"] == sentence_id
    ].sort_values("word_index")

    return sentence


def generate_token_candidates(
    sentence_df,
    max_n=5
):
    """
    Generate candidates directly from the
    original TeluguNER tokens.

    Each candidate stores:
        text
        start_token
        end_token
    """

    tokens = sentence_df[
        "word"
    ].tolist()

    candidates = []

    for start in range(len(tokens)):

        for n in range(1, max_n + 1):

            end = start + n

            if end > len(tokens):
                break

            span = tokens[start:end]

            candidate_text = " ".join(span)

            candidates.append({
                "text": candidate_text,
                "start_token": start + 1,
                "end_token": end
            })

    return candidates


def extract_gold_entities(sentence_df):
    entities = []

    current_tokens = []
    current_label = None
    start_token = None
    last_token = None

    for _, row in sentence_df.iterrows():
        word = row["word"]
        tag = row["tag"]
        index = int(row["word_index"])

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
            start_token = index
            last_token = index

        elif (
            tag.startswith("I-")
            and current_tokens
            and current_label == tag[2:]
        ):
            current_tokens.append(word)
            last_token = index

        elif tag.startswith("I-"):
            # Malformed BIO: treat I-X as a new entity
            if current_tokens:
                entities.append({
                    "text": " ".join(current_tokens),
                    "label": current_label,
                    "start_token": start_token,
                    "end_token": last_token
                })

            current_tokens = [word]
            current_label = tag[2:]
            start_token = index
            last_token = index

        else:
            # O tag
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

    # Save final entity
    if current_tokens:
        entities.append({
            "text": " ".join(current_tokens),
            "label": current_label,
            "start_token": start_token,
            "end_token": last_token
        })

    return entities


def candidate_recall(
    gold_entities,
    candidates
):

    candidate_spans = {
        (
            c["start_token"],
            c["end_token"]
        )
        for c in candidates
    }

    found = 0

    for entity in gold_entities:

        gold_span = (
            entity["start_token"],
            entity["end_token"]
        )

        if gold_span in candidate_spans:
            found += 1

    total = len(gold_entities)

    recall = (
        found / total
        if total > 0
        else 1.0
    )

    return found, total, recall


if __name__ == "__main__":

    df = load_data()

    # Use the official test split
    TEST_FILE = PROJECT_ROOT / "data" / "test.json"

    import json

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print("=" * 80)
    print("FULL TEST-SET CANDIDATE RECALL")
    print("=" * 80)

    total_gold_entities = 0
    total_found_entities = 0

    total_candidates = 0
    total_sentences = 0

    sentences_with_missing = 0
    missing_entities = []

    max_candidates = 0
    max_candidates_sentence_id = None

    # ---------------------------------------------------------
    # Evaluate every test sentence
    # ---------------------------------------------------------

    for item in test_data:

        sentence_id = int(item["sentence_id"])

        sentence_df = get_sentence(
            df,
            sentence_id
        )

        # Gold entities from original TeluguNER CSV
        gold_entities = extract_gold_entities(
            sentence_df
        )

        # Generate 1-5 token candidates
        candidates = generate_token_candidates(
            sentence_df,
            max_n=5
        )

        # Candidate spans
        candidate_spans = {
            (
                c["start_token"],
                c["end_token"]
            )
            for c in candidates
        }

        # Evaluate gold entities
        for entity in gold_entities:

            gold_span = (
                entity["start_token"],
                entity["end_token"]
            )

            total_gold_entities += 1

            if gold_span in candidate_spans:
                total_found_entities += 1
            else:
                sentences_with_missing += 1

                missing_entities.append({
                    "sentence_id": sentence_id,
                    "text": entity["text"],
                    "label": entity["label"],
                    "start_token": entity["start_token"],
                    "end_token": entity["end_token"]
                })

        # Candidate statistics
        num_candidates = len(candidates)

        total_candidates += num_candidates
        total_sentences += 1

        if num_candidates > max_candidates:
            max_candidates = num_candidates
            max_candidates_sentence_id = sentence_id

    # ---------------------------------------------------------
    # Final statistics
    # ---------------------------------------------------------

    if total_gold_entities > 0:
        recall = (
            total_found_entities /
            total_gold_entities
        )
    else:
        recall = 1.0

    average_candidates = (
        total_candidates /
        total_sentences
        if total_sentences > 0
        else 0
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(
        f"Test sentences:              "
        f"{total_sentences:,}"
    )

    print(
        f"Total gold entities:         "
        f"{total_gold_entities:,}"
    )

    print(
        f"Gold entities found:         "
        f"{total_found_entities:,}"
    )

    print(
        f"Gold entities missed:        "
        f"{total_gold_entities - total_found_entities:,}"
    )

    print(
        f"Candidate recall:            "
        f"{recall:.2%}"
    )

    print(
        f"Total candidate spans:       "
        f"{total_candidates:,}"
    )

    print(
        f"Average candidates/sentence: "
        f"{average_candidates:.2f}"
    )

    print(
        f"Maximum candidates/sentence: "
        f"{max_candidates:,}"
    )

    print(
        f"Max candidate sentence ID:   "
        f"{max_candidates_sentence_id}"
    )

    # ---------------------------------------------------------
    # Missing entities
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("MISSED GOLD ENTITIES")
    print("=" * 80)

    if not missing_entities:
        print("None.")
    else:
        print(
            f"Total missed: "
            f"{len(missing_entities)}"
        )

        # Show first 20 only
        for entity in missing_entities[:20]:
            print(
                f"\nSentence ID: {entity['sentence_id']}"
            )

            print(
                f"Span: "
                f"{entity['start_token']}-"
                f"{entity['end_token']}"
            )

            print(
                f"Entity: {entity['text']}"
            )

            print(
                f"Label: {entity['label']}"
            )

        if len(missing_entities) > 20:
            print(
                f"\n... and "
                f"{len(missing_entities) - 20} more."
            )