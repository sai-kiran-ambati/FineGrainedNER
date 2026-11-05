from pathlib import Path
import json
import re
import unicodedata


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_FILE = PROJECT_ROOT / "data" / "test.json"


def load_test_data():
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize_telugu(sentence):
    """
    Simple whitespace-based tokenizer.

    We intentionally start simple because the TeluguNER data
    is already tokenized at the dataset level.
    """
    return sentence.split()


def generate_candidates(sentence, max_n=5):
    """
    Generate contiguous n-gram candidates.

    Generates spans of 1 to max_n tokens.
    Punctuation-only tokens are skipped.
    Punctuation attached to the beginning/end of a
    candidate is removed.
    """

    tokens = tokenize_telugu(sentence)

    candidates = []

    punctuation = {
        ".", ",", ":", ";", "!", "?",
        "।", "॥",
        ")", "]", "}",
        "(", "[", "{"
    }

    for n in range(1, max_n + 1):

        for i in range(len(tokens) - n + 1):

            span_tokens = tokens[i:i+n]

            candidate = " ".join(span_tokens)

            # Remove punctuation from beginning/end
            candidate = candidate.strip(
                ".,:;!?।॥()[]{}"
            )

            candidate = clean_candidate(candidate)

            if not candidate:
                continue

            if candidate in punctuation:
                continue

            candidates.append(candidate)

    return candidates


def remove_duplicates(candidates):

    seen = set()
    unique = []

    for candidate in candidates:

        normalized = normalize_for_matching(
            candidate
        )

        if normalized not in seen:

            seen.add(normalized)
            unique.append(candidate)

    return unique


def clean_candidate(candidate):
    """
    Normalize Telugu text for candidate/gold matching.

    This is used ONLY for candidate recall matching.
    We keep the original text unchanged for final evaluation.
    """

    candidate = unicodedata.normalize(
        "NFC",
        candidate
    )

    # Remove zero-width / invisible formatting characters
    invisible_chars = [
        "\u200b",  # ZERO WIDTH SPACE
        "\u200c",  # ZERO WIDTH NON-JOINER
        "\u200d",  # ZERO WIDTH JOINER
        "\u2060",  # WORD JOINER
        "\ufeff"   # ZERO WIDTH NO-BREAK SPACE
    ]

    for char in invisible_chars:
        candidate = candidate.replace(char, "")

    # Normalize whitespace
    candidate = re.sub(
        r"\s+",
        " ",
        candidate
    )

    return candidate.strip()


def filter_candidates(candidates):
    """
    Remove obviously useless candidates.

    This is deliberately conservative.
    We don't want to accidentally remove real entities.
    """

    filtered = []

    punctuation_only = {
        ".", ",", ":", ";", "!", "?",
        "।", "॥", ")", "]", "}", "(",
        "[", "{"
    }

    for candidate in candidates:

        candidate = clean_candidate(candidate)

        if not candidate:
            continue

        # Remove single punctuation candidates
        if candidate in punctuation_only:
            continue

        filtered.append(candidate)

    return filtered


def get_sentence_text(item):
    """
    Handle the sentence representation in our JSON.

    Our current JSON stores the sentence under 'text'.
    """

    return item["text"]


def candidate_recall(gold_entities, candidates):

    candidate_set = {
        clean_candidate(c).lower()
        for c in candidates
    }

    found = 0

    for entity in gold_entities:

        gold_text = clean_candidate(
            entity["text"]
        ).lower()

        if gold_text in candidate_set:
            found += 1

    total = len(gold_entities)

    if total == 0:
        return 1.0

    return found / total



def normalize_for_matching(text):
    """
    Normalize text ONLY for comparing gold entities
    with generated candidates.

    Original text is never modified.
    """

    text = unicodedata.normalize("NFC", text)

    # Remove zero-width/invisible formatting characters
    for char in [
        "\u200b",  # Zero Width Space
        "\u200c",  # Zero Width Non-Joiner
        "\u200d",  # Zero Width Joiner
        "\u2060",  # Word Joiner
        "\ufeff"   # Zero Width No-Break Space
    ]:
        text = text.replace(char, "")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def unicode_debug(text):
    return " ".join(
        f"U+{ord(c):04X}"
        for c in text
    )


def evaluate_candidate_recall(data, num_sentences=20):

    total_gold = 0
    total_found = 0

    print("\n" + "=" * 80)
    print("CANDIDATE RECALL DIAGNOSTIC")
    print("=" * 80)

    for item in data[:num_sentences]:

        sentence = get_sentence_text(item)

        candidates = generate_candidates(
            sentence,
            max_n=3
        )

        candidates = filter_candidates(candidates)

        candidate_set = {
            normalize_for_matching(c)
            for c in candidates
        }

        for entity in item.get("entities", []):

            gold_original = entity["text"]

            gold_normalized = normalize_for_matching(
                gold_original
            )

            total_gold += 1

            if gold_normalized in candidate_set:

                total_found += 1

            else:

                print("\n" + "-" * 80)
                print(
                    f"Sentence ID: "
                    f"{item.get('sentence_id')}"
                )

                print(
                    f"Gold: "
                    f"{gold_original} -> "
                    f"{entity['label']}"
                )

                print(
                    f"Normalized gold: "
                    f"{gold_normalized}"
                )

                print(
                    f"Gold Unicode:\n"
                    f"{unicode_debug(gold_original)}"
                )

                # Find candidates that look similar
                similar = []

                for candidate in candidates:

                    candidate_normalized = (
                        normalize_for_matching(candidate)
                    )

                    if (
                        gold_normalized in candidate_normalized
                        or
                        candidate_normalized in gold_normalized
                    ):
                        similar.append(candidate)

                if similar:

                    print("Similar candidates:")

                    for candidate in similar:
                        print(
                            f"  {candidate}"
                        )

                else:

                    print(
                        "No similar candidate found."
                    )

    recall = (
        total_found / total_gold
        if total_gold > 0
        else 0
    )

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)

    print(f"Gold entities: {total_gold}")
    print(f"Found entities: {total_found}")
    print(
        f"Missing entities: "
        f"{total_gold - total_found}"
    )
    print(
        f"Candidate recall: "
        f"{recall:.2%}"
    )

def find_missing_entities(item, candidates):

    candidate_set = {
        clean_candidate(c).lower()
        for c in candidates
    }

    missing = []

    for entity in item.get("entities", []):

        gold = clean_candidate(
            entity["text"]
        ).lower()

        if gold not in candidate_set:
            missing.append(entity)


            print(
                f"  GOLD: {entity['text']}"
            )

            print(
                f"  GOLD Unicode: "
                f"{unicode_debug(entity['text'])}"
            )


    return missing

def show_unicode(text):
    print(f"\nText: {text}")
    print("Characters:")

    for char in text:
        print(
            f"  {char!r} -> U+{ord(char):04X} "
            f"{unicodedata.name(char, 'UNKNOWN')}"
        )


def unicode_debug(text):
    return " ".join(
        f"U+{ord(c):04X}"
        for c in text
    )


def find_similar_candidates(gold, candidates):

    gold_clean = clean_candidate(gold)

    results = []

    for candidate in candidates:

        candidate_clean = clean_candidate(candidate)

        if (
            gold_clean in candidate_clean
            or candidate_clean in gold_clean
        ):
            results.append(candidate)

    return results



if __name__ == "__main__":

    data = load_test_data()

    # Take the first test sentence
    item = data[0]

    sentence = get_sentence_text(item)

    print("=" * 80)
    print("TEST SENTENCE")
    print("=" * 80)

    print(sentence)

    print("\n" + "=" * 80)
    print("GOLD ENTITIES")
    print("=" * 80)

    for entity in item.get("entities", []):
        print(
            f"{entity['text']} -> {entity['label']}"
        )

    candidates = generate_candidates(
        sentence,
        max_n=5
    )

    candidates = filter_candidates(candidates)
    candidates = remove_duplicates(candidates)

    print("\n" + "=" * 80)
    print("CANDIDATES")
    print("=" * 80)

    for i, candidate in enumerate(candidates, start=1):
        print(f"{i:3}. {candidate}")

    print("\n" + "=" * 80)
    print(f"Number of candidates: {len(candidates)}")
    print("=" * 80)


    recall = candidate_recall(
        item.get("entities", []),
        candidates
    )

    print("\n" + "=" * 80)
    print("CANDIDATE GENERATION EVALUATION")
    print("=" * 80)

    print(
        f"Gold entities: {len(item.get('entities', []))}"
    )

    print(
        f"Candidates: {len(candidates)}"
    )

    print(
        f"Candidate recall: {recall:.2%}"
    )    


    evaluate_candidate_recall(
        data,
        num_sentences=20
    )


    similar = find_similar_candidates(
        entity["text"],
        candidates
    )

    print(
        f"  GOLD: {entity['text']} -> {entity['label']}"
    )

    if similar:
        print("  Similar candidates:")
        for candidate in similar:
            print(f"    {candidate}")
    else:
        print("  No similar candidate found.")