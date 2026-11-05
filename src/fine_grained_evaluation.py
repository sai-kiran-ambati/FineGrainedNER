"""Evaluation views for fine labels mapped to TeluguNER coarse classes."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate_predictions import evaluate_predictions


def find_results_file(provider=None, sentence_count=None):
    """Find matching results, defaulting to the most recently updated run."""
    results_dir = PROJECT_ROOT / "results"
    provider_part = provider or "*"
    count_part = str(sentence_count) if sentence_count is not None else "*"
    pattern = f"promptner_{provider_part}_fine_grained{count_part}_results.json"
    candidates = list(results_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No fine-grained result files match {pattern} in {results_dir}. "
            "Run a fine-grained sentence runner first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def print_saved_summary(results_file):
    with results_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    summary = data.get("summary", {})
    print(f"Results: {results_file}")
    print(f"Provider: {summary.get('provider', 'unknown')} | Model: {summary.get('model', 'unknown')}")
    scores = summary.get("matching_approaches", {})
    if not scores:
        print("No matching_approaches summary found. Re-run the fine-grained runner.")
        return
    for name, score in scores.items():
        print(
            f"{name}: P={score['precision']:.4f} "
            f"R={score['recall']:.4f} F1={score['f1']:.4f} "
            f"(TP={score['true_positives']} FP={score['false_positives']} "
            f"FN={score['false_negatives']})"
        )
    print_entity_details(data)


def _entity_key(entity):
    return (
        int(entity["start_token"]),
        int(entity["end_token"]),
        entity["label"],
    )


def print_entity_details(data):
    """Show gold, exact true positives, false positives, and false negatives."""
    for row in data.get("sentences", []):
        gold = row.get("gold_entities", [])
        predictions = row.get("predictions", [])
        gold_by_key = {_entity_key(entity): entity for entity in gold}
        pred_by_key = {_entity_key(pred): pred for pred in predictions}
        true_positives = sorted(gold_by_key.keys() & pred_by_key.keys())
        false_positives = sorted(pred_by_key.keys() - gold_by_key.keys())
        false_negatives = sorted(gold_by_key.keys() - pred_by_key.keys())

        print(f"\nSentence {row.get('sentence_id')}: {row.get('sentence', '')}")
        print("  Ground entities:")
        if gold:
            for entity in gold:
                print(
                    f"    {entity['text']} -> {entity['label']} "
                    f"[{entity['start_token']}-{entity['end_token']}]"
                )
        else:
            print("    (none)")

        print("  False positives:")
        if false_positives:
            for match in false_positives:
                pred = pred_by_key[match]
                print(
                    f"    {pred['text']} -> {pred.get('fine_label', pred['label'])} "
                    f"(maps to {pred['label']}) [{pred['start_token']}-{pred['end_token']}]"
                )
        else:
            print("    (none)")

        print("  False negatives:")
        if false_negatives:
            for match in false_negatives:
                entity = gold_by_key[match]
                print(
                    f"    {entity['text']} -> {entity['label']} "
                    f"[{entity['start_token']}-{entity['end_token']}]"
                )
        else:
            print("    (none)")

        print("  True positives (exact entity-wise):")
        if true_positives:
            for match in true_positives:
                pred = pred_by_key[match]
                print(
                    f"    {pred['text']} -> {pred.get('fine_label', pred['label'])} "
                    f"(maps to {pred['label']}) [{pred['start_token']}-{pred['end_token']}]"
                )
        else:
            print("    (none)")


def _metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def tokenwise_tag_sequences(sentence_df, predictions):
    """Build aligned gold/predicted BIO tags, resolving prediction overlaps."""
    ordered = sentence_df.sort_values("word_index")
    gold_tags = ordered["tag"].astype(str).tolist()
    predicted_tags = ["O"] * len(gold_tags)

    # Resolve overlapping model spans deterministically, preferring the
    # longest predicted entity and then the leftmost span.
    ordered_predictions = sorted(
        predictions,
        key=lambda pred: (
            -(pred["end_token"] - pred["start_token"] + 1),
            pred["start_token"],
            pred["end_token"],
            pred.get("candidate_id", ""),
        ),
    )
    occupied = set()
    for pred in ordered_predictions:
        start = int(pred["start_token"])
        end = int(pred["end_token"])
        label = pred["label"]
        token_positions = set(range(start, end + 1))
        if start < 1 or end > len(predicted_tags) or token_positions & occupied:
            continue
        occupied |= token_positions
        predicted_tags[start - 1] = f"B-{label}"
        for token_index in range(start, end):
            predicted_tags[token_index] = f"I-{label}"
    return ordered, gold_tags, predicted_tags


def evaluate_token_wise(sentence_df, predictions):
    """Compare BIO tags token by token; O tokens are excluded from P/R/F1."""
    _, gold_tags, predicted_tags = tokenwise_tag_sequences(
        sentence_df, predictions
    )

    tp = fp = fn = 0
    correct_tokens = 0
    for gold_tag, predicted_tag in zip(gold_tags, predicted_tags):
        if gold_tag == predicted_tag:
            correct_tokens += 1
            if gold_tag != "O":
                tp += 1
        else:
            if predicted_tag != "O":
                fp += 1
            if gold_tag != "O":
                fn += 1

    result = _metrics(tp, fp, fn)
    result["token_accuracy"] = correct_tokens / len(gold_tags) if gold_tags else 0.0
    result["tokens"] = len(gold_tags)
    return result


def evaluate_entity_wise(gold_entities, predictions):
    """Exact span and mapped TeluguNER coarse-label match."""
    return evaluate_predictions(gold_entities, predictions)


def partial_match_pairs(gold_entities, predictions):
    """Return one-to-one partial matches as (prediction index, gold index).

    A match requires inclusive token-span overlap and equal TeluguNER coarse
    labels. Each prediction and gold entity can appear in at most one pair.
    """
    adjacency = []
    for pred in predictions:
        p_start, p_end = int(pred["start_token"]), int(pred["end_token"])
        adjacency.append([
            idx for idx, gold in enumerate(gold_entities)
            if (
                pred["label"] == gold["label"]
                and p_start <= int(gold["end_token"])
                and int(gold["start_token"]) <= p_end
            )
        ])

    gold_to_prediction = {}

    def assign(pred_idx, visited):
        for gold_idx in adjacency[pred_idx]:
            if gold_idx in visited:
                continue
            visited.add(gold_idx)
            if gold_idx not in gold_to_prediction or assign(
                gold_to_prediction[gold_idx], visited
            ):
                gold_to_prediction[gold_idx] = pred_idx
                return True
        return False

    for pred_idx in range(len(predictions)):
        assign(pred_idx, set())
    return sorted(
        (pred_idx, gold_idx)
        for gold_idx, pred_idx in gold_to_prediction.items()
    )


def evaluate_partial(gold_entities, predictions):
    """One-to-one partial span match with matching mapped coarse labels."""
    matched = len(partial_match_pairs(gold_entities, predictions))
    return _metrics(
        matched,
        len(predictions) - matched,
        len(gold_entities) - matched,
    )


def evaluate_all_matching(sentence_df, gold_entities, predictions):
    """Return all three matching approaches for one sentence."""
    return {
        "token_wise": evaluate_token_wise(sentence_df, predictions),
        "entity_wise": evaluate_entity_wise(gold_entities, predictions),
        "partial": evaluate_partial(gold_entities, predictions),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Display the saved fine-grained NER matching scores."
    )
    parser.add_argument("--provider", choices=("openai", "sarvam"))
    parser.add_argument("--sentences", type=int, help="Select results by sentence count")
    parser.add_argument("--file", type=Path, help="Explicit result JSON path")
    args = parser.parse_args()
    results_file = args.file or find_results_file(args.provider, args.sentences)
    print_saved_summary(results_file)


if __name__ == "__main__":
    main()
