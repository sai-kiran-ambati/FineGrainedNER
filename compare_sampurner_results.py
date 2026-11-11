"""Recompute all SampurNER matching scores from a saved prediction run."""

import argparse
import json
from pathlib import Path

from src.sampurner_evaluation import evaluate_all


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def aggregate(per_sentence, name):
    scores = [row[name] for row in per_sentence]
    tp = sum(score["true_positives"] for score in scores)
    fp = sum(score["false_positives"] for score in scores)
    fn = sum(score["false_negatives"] for score in scores)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    result = {"true_positives": tp, "false_positives": fp, "false_negatives": fn,
              "precision": precision, "recall": recall,
              "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    if name == "token_wise":
        token_count = sum(score.get("tokens", 0) for score in scores)
        correct = sum(round(score.get("token_accuracy", 0.0) * score.get("tokens", 0)) for score in scores)
        result["tokens"] = token_count
        result["token_accuracy"] = correct / token_count if token_count else 0.0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Saved PromptNER SampurNER result JSON")
    parser.add_argument("--output", type=Path, help="Optional output path for the comparison JSON")
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    summary = source.get("summary", {})
    level = summary.get("level", "fine")
    split_file = ROOT / "data" / "sampurner" / f"{summary.get('split', 'test')}.json"
    token_counts = {}
    if split_file.exists():
        prepared = json.loads(split_file.read_text(encoding="utf-8"))
        token_counts = {row["sentence_id"]: len(row["tokens"]) for row in prepared}
    sentence_scores = []
    for row in source.get("sentences", []):
        tokens = row.get("tokens")
        n_tokens = (
            len(tokens) if tokens is not None
            else token_counts.get(row.get("sentence_id"), len(row.get("sentence", "").split()))
        )
        scores = evaluate_all(
            row.get("gold_entities", []), row.get("predictions", []), n_tokens, level
        )
        sentence_scores.append({
            "sentence_id": row.get("sentence_id"),
            "matching_evaluations": scores,
        })
    approaches = {
        name: aggregate([row["matching_evaluations"] for row in sentence_scores], name)
        for name in ("token_wise", "entity_wise", "partial")
    }
    comparison = {
        "summary": {
            **source.get("summary", {}),
            "matching_approaches": approaches,
            "scoring": "Exact span+label, BIO token-wise, and one-to-one partial overlap+label",
        },
        "sentences": sentence_scores,
    }
    output_path = args.output or RESULTS_DIR / f"sampurner_matching_comparison_{args.input.stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, score in approaches.items():
        print(f"{name}: P={score['precision']:.4f} R={score['recall']:.4f} F1={score['f1']:.4f}")
    print(f"Saved comparison to {output_path}")


if __name__ == "__main__":
    main()
