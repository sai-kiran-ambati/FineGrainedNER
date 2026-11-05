"""Detailed reports for one saved fine-grained matching approach."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fine_grained_evaluation import (
    evaluate_entity_wise,
    evaluate_partial,
    evaluate_token_wise,
    find_results_file,
    partial_match_pairs,
    tokenwise_tag_sequences,
)
from src.generate_token_candidates import get_sentence, load_data


def _key(entity):
    return (
        int(entity["start_token"]),
        int(entity["end_token"]),
        entity["label"],
    )


def _print_gold(gold):
    print("  Ground entities:")
    if not gold:
        print("    (none)")
    for entity in gold:
        print(
            f"    {entity['text']} -> {entity['label']} "
            f"[{entity['start_token']}-{entity['end_token']}]"
        )


def run_matching_report(mode, results_file=None, provider=None):
    results_file = results_file or find_results_file(provider)
    data = json.loads(results_file.read_text(encoding="utf-8"))
    token_data = load_data() if mode == "token_wise" else None
    total_tp = total_fp = total_fn = 0
    total_tokens = total_correct_tokens = 0

    for row in data.get("sentences", []):
        gold = row.get("gold_entities", [])
        predictions = row.get("predictions", [])
        print(f"\nSentence {row['sentence_id']}: {row['sentence']}")
        _print_gold(gold)

        if mode == "entity_wise":
            scores = evaluate_entity_wise(gold, predictions)
            gold_by_key = {_key(entity): entity for entity in gold}
            pred_by_key = {_key(pred): pred for pred in predictions}
            tp_keys = gold_by_key.keys() & pred_by_key.keys()
            fp_keys = pred_by_key.keys() - gold_by_key.keys()
            fn_keys = gold_by_key.keys() - pred_by_key.keys()
            print("  True positives:")
            for key in sorted(tp_keys):
                pred = pred_by_key[key]
                print(
                    f"    {pred['text']} -> {pred.get('fine_label', pred['label'])} "
                    f"(maps to {pred['label']}) [{key[0]}-{key[1]}]"
                )
            _print_entity_errors(fp_keys, pred_by_key, "False positives")
            _print_entity_errors(fn_keys, gold_by_key, "False negatives")

        elif mode == "partial":
            scores = evaluate_partial(gold, predictions)
            pairs = partial_match_pairs(gold, predictions)
            matched_preds = {p for p, _ in pairs}
            matched_gold = {g for _, g in pairs}
            print("  True positives (overlap + mapped class):")
            for pred_idx, gold_idx in pairs:
                pred, entity = predictions[pred_idx], gold[gold_idx]
                print(
                    f"    predicted {pred['text']} -> {pred.get('fine_label', pred['label'])} "
                    f"(maps to {pred['label']}) [{pred['start_token']}-{pred['end_token']}] "
                    f"matches gold {entity['text']} -> {entity['label']} "
                    f"[{entity['start_token']}-{entity['end_token']}]"
                )
            _print_unmatched_entities(
                [p for i, p in enumerate(predictions) if i not in matched_preds],
                "False positives",
            )
            _print_unmatched_entities(
                [g for i, g in enumerate(gold) if i not in matched_gold],
                "False negatives",
            )

        else:
            sentence_df = get_sentence(token_data, row["sentence_id"])
            ordered, gold_tags, pred_tags = tokenwise_tag_sequences(
                sentence_df, predictions
            )
            scores = evaluate_token_wise(sentence_df, predictions)
            words = ordered["word"].astype(str).tolist()
            tp_pos = [i for i, (g, p) in enumerate(zip(gold_tags, pred_tags)) if g != "O" and g == p]
            fp_pos = [i for i, (g, p) in enumerate(zip(gold_tags, pred_tags)) if p != "O" and g != p]
            fn_pos = [i for i, (g, p) in enumerate(zip(gold_tags, pred_tags)) if g != "O" and g != p]
            print("  True-positive tokens:")
            _print_token_positions(tp_pos, words, gold_tags)
            print("  False-positive tokens:")
            _print_token_positions(fp_pos, words, pred_tags)
            print("  False-negative tokens:")
            _print_token_positions(fn_pos, words, gold_tags)
            total_tokens += scores["tokens"]
            total_correct_tokens += round(scores["token_accuracy"] * scores["tokens"])

        total_tp += scores["true_positives"]
        total_fp += scores["false_positives"]
        total_fn += scores["false_negatives"]
        print(
            f"  {mode}: P={scores['precision']:.4f} R={scores['recall']:.4f} "
            f"F1={scores['f1']:.4f}"
        )

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"\nResults file: {results_file}")
    print(f"Overall {mode}: P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
    print(f"TP={total_tp} FP={total_fp} FN={total_fn}")
    if mode == "token_wise":
        accuracy = total_correct_tokens / total_tokens if total_tokens else 0.0
        print(f"Token accuracy={accuracy:.4f} ({total_correct_tokens}/{total_tokens})")


def _print_entity_errors(indices, entities, heading):
    print(f"  {heading}:")
    if not indices:
        print("    (none)")
    for key in sorted(indices):
        entity = entities[key]
        fine_label = entity.get("fine_label")
        label = fine_label or entity["label"]
        mapping = f" (maps to {entity['label']})" if fine_label else ""
        print(f"    {entity['text']} -> {label}{mapping} [{key[0]}-{key[1]}]")


def _print_unmatched_entities(entities, heading):
    print(f"  {heading}:")
    if not entities:
        print("    (none)")
    for entity in entities:
        label = entity.get("fine_label", entity["label"])
        mapping = f" (maps to {entity['label']})" if "fine_label" in entity else ""
        print(
            f"    {entity['text']} -> {label}{mapping} "
            f"[{entity['start_token']}-{entity['end_token']}]"
        )


def _print_token_positions(indices, words, tags):
    if not indices:
        print("    (none)")
    for i in indices:
        print(f"    token {i + 1}: {words[i]} -> {tags[i]}")


def main_for(mode):
    parser = argparse.ArgumentParser(description=f"Detailed {mode} matching report")
    parser.add_argument("--provider", choices=("openai", "sarvam"))
    parser.add_argument("--sentences", type=int, help="Select results by sentence count")
    parser.add_argument("--file", type=Path, help="Explicit results JSON path")
    args = parser.parse_args()
    selected_file = args.file or find_results_file(args.provider, args.sentences)
    run_matching_report(mode, selected_file)
