
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

TEST_FILE = PROJECT_ROOT / "data" / "test.json"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "promptner_gpt20_results.json"


# ---------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------
sys.path.insert(0, str(SRC_DIR))

from src.generate_token_candidates import (  # noqa: E402
    load_data,
    get_sentence,
    generate_token_candidates,
    extract_gold_entities,
)
from src.filter_candidates import filter_candidates  # noqa: E402
from src.build_prompt import build_prompt  # noqa: E402
from src.parse_predictions import parse_gpt_output  # noqa: E402
from src.evaluate_predictions import evaluate_predictions  # noqa: E402


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
MODEL = "gpt-5.6-luna"
MAX_N = 5

# Keep this deterministic: always use the first 20 entries in test.json.
NUM_SENTENCES = 20


def load_test_ids(limit=NUM_SENTENCES):
    """Load sentence IDs from test.json."""
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support either:
    #   [13192, 12756, ...]
    # or:
    #   {"sentence_ids": [13192, 12756, ...]}
    # or a list of objects containing sentence_id.
    if isinstance(data, dict):
        if "sentence_ids" in data:
            ids = data["sentence_ids"]
        elif "ids" in data:
            ids = data["ids"]
        elif "data" in data:
            data = data["data"]
            ids = []
            for item in data:
                if isinstance(item, dict):
                    ids.append(item["sentence_id"])
                else:
                    ids.append(item)
        else:
            raise ValueError(
                "test.json must contain 'sentence_ids', 'ids', or 'data'."
            )
    elif isinstance(data, list):
        ids = []
        for item in data:
            if isinstance(item, dict):
                if "sentence_id" not in item:
                    raise ValueError(
                        "An object in test.json is missing 'sentence_id'."
                    )
                ids.append(item["sentence_id"])
            else:
                ids.append(item)
    else:
        raise ValueError("Unsupported test.json format.")

    return [int(x) for x in ids[:limit]]


def reconstruct_sentence(sentence_df):
    """Reconstruct the sentence exactly as stored token-by-token."""
    return " ".join(sentence_df["word"].tolist())


def aggregate_counts(results):
    tp = sum(r["evaluation"]["true_positives"] for r in results)
    fp = sum(r["evaluation"]["false_positives"] for r in results)
    fn = sum(r["evaluation"]["false_negatives"] for r in results)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    # -------------------------------------------------------------
    # 1. Load API key
    # -------------------------------------------------------------
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not found. Put it in your .env file."
        )

    client = OpenAI()

    # -------------------------------------------------------------
    # 2. Load data and deterministic test IDs
    # -------------------------------------------------------------
    df = load_data()
    test_ids = load_test_ids()

    if len(test_ids) < NUM_SENTENCES:
        raise RuntimeError(
            f"Expected at least {NUM_SENTENCES} test sentence IDs, "
            f"but found {len(test_ids)}."
        )

    print(f"Model: {MODEL}")
    print(f"Sentences: {len(test_ids)}")
    print(f"Candidate strategy: contiguous 1-{MAX_N} original tokens")
    print("-" * 70)

    all_results = []

    # -------------------------------------------------------------
    # 3. Process each sentence
    # -------------------------------------------------------------
    for position, sentence_id in enumerate(test_ids, start=1):
        print(
            f"\n[{position:02d}/{len(test_ids)}] "
            f"Processing sentence_id={sentence_id}"
        )

        sentence_df = get_sentence(df, sentence_id)

        if sentence_df.empty:
            print("  WARNING: sentence not found; skipping.")
            continue

        sentence = reconstruct_sentence(sentence_df)

        # Gold entities come directly from the original dataset.
        gold_entities = extract_gold_entities(sentence_df)

        # ---------------------------------------------------------
        # Candidate generation: FROZEN 1-5 token strategy
        # ---------------------------------------------------------
        candidates_before = generate_token_candidates(
            sentence_df,
            max_n=MAX_N,
        )

        candidates = filter_candidates(candidates_before)

        print(
            f"  Gold entities: {len(gold_entities)} | "
            f"Candidates: {len(candidates_before)} -> {len(candidates)}"
        )

        # ---------------------------------------------------------
        # Prompt construction
        # ---------------------------------------------------------
        prompt = build_prompt(
            sentence=sentence,
            candidates=candidates,
        )

        # ---------------------------------------------------------
        # GPT call
        # ---------------------------------------------------------
        try:
            response = client.responses.create(
                model=MODEL,
                input=prompt,
            )
            raw_output = response.output_text
        except Exception as exc:
            print(f"  ERROR during API call: {exc}")
            all_results.append(
                {
                    "sentence_position": position,
                    "sentence_id": sentence_id,
                    "sentence": sentence,
                    "gold_entities": gold_entities,
                    "candidate_count_before_filter": len(candidates_before),
                    "candidate_count_after_filter": len(candidates),
                    "candidates": candidates,
                    "prompt": prompt,
                    "raw_output": None,
                    "predictions": [],
                    "evaluation": {
                        "true_positives": 0,
                        "false_positives": 0,
                        "false_negatives": len(gold_entities),
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                    },
                    "error": str(exc),
                }
            )
            continue

        # ---------------------------------------------------------
        # Parse Candidate IDs -> exact text + token span + label
        # ---------------------------------------------------------
        predictions = parse_gpt_output(
            raw_output,
            candidates,
        )

        # ---------------------------------------------------------
        # Exact entity-level evaluation:
        # (start_token, end_token, label)
        # ---------------------------------------------------------
        evaluation = evaluate_predictions(
            gold_entities,
            predictions,
        )

        print(
            f"  Predicted: {len(predictions)} | "
            f"TP={evaluation['true_positives']} "
            f"FP={evaluation['false_positives']} "
            f"FN={evaluation['false_negatives']} | "
            f"F1={evaluation['f1']:.4f}"
        )

        all_results.append(
            {
                "sentence_position": position,
                "sentence_id": sentence_id,
                "sentence": sentence,
                "gold_entities": gold_entities,
                "candidate_count_before_filter": len(candidates_before),
                "candidate_count_after_filter": len(candidates),
                "candidates": candidates,
                "prompt": prompt,
                "raw_output": raw_output,
                "predictions": predictions,
                "evaluation": evaluation,
            }
        )

    # -------------------------------------------------------------
    # 4. Aggregate Micro-P/R/F1
    # -------------------------------------------------------------
    aggregate = aggregate_counts(all_results)

    # Candidate recall is measured independently from the model.
    # We check whether every gold span has an exact candidate span.
    candidate_recall_tp = 0
    candidate_gold = 0

    for result in all_results:
        gold_spans = {
            (e["start_token"], e["end_token"])
            for e in result["gold_entities"]
        }
        candidate_spans = {
            (c["start_token"], c["end_token"])
            for c in result["candidates"]
        }

        candidate_recall_tp += len(gold_spans & candidate_spans)
        candidate_gold += len(gold_spans)

    candidate_recall = (
        candidate_recall_tp / candidate_gold
        if candidate_gold
        else 0.0
    )

    summary = {
        "model": MODEL,
        "num_sentences_requested": NUM_SENTENCES,
        "num_sentences_processed": len(all_results),
        "candidate_strategy": (
            "contiguous spans of 1-5 original TeluguNER tokens, "
            "followed by punctuation-edge filtering"
        ),
        "candidate_recall": {
            "found_gold_spans": candidate_recall_tp,
            "gold_spans": candidate_gold,
            "recall": candidate_recall,
        },
        "micro_evaluation": aggregate,
    }

    output = {
        "summary": summary,
        "sentences": all_results,
    }

    # -------------------------------------------------------------
    # 5. Save complete reproducibility artifact
    # -------------------------------------------------------------
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -------------------------------------------------------------
    # 6. Print final result
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL MICRO RESULTS")
    print("=" * 70)
    print(f"TP:        {aggregate['true_positives']}")
    print(f"FP:        {aggregate['false_positives']}")
    print(f"FN:        {aggregate['false_negatives']}")
    print(f"Precision: {aggregate['precision']:.4f}")
    print(f"Recall:    {aggregate['recall']:.4f}")
    print(f"F1:        {aggregate['f1']:.4f}")

    print("\nCANDIDATE RECALL")
    print("-" * 70)
    print(
        f"Found: {candidate_recall_tp}/{candidate_gold} "
        f"({candidate_recall:.4%})"
    )

    print(f"\nFull results saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
