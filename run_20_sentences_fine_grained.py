"""Run fine-grained PromptNER and score via TeluguNER coarse labels."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from run_20_sentences import (
    PROJECT_ROOT,
    load_test_ids,
    reconstruct_sentence,
    MAX_N,
    NUM_SENTENCES,
)
from src.build_prompt import build_fine_grained_prompt
from src.fine_grained_evaluation import evaluate_all_matching
from src.filter_candidates import filter_candidates
from src.generate_token_candidates import (
    extract_gold_entities,
    generate_token_candidates,
    get_sentence,
    load_data,
)
from src.parse_predictions import parse_fine_grained_output


def make_client():
    load_dotenv(PROJECT_ROOT / ".env")
    provider = os.getenv("NER_PROVIDER", "openai").strip().lower()
    if provider == "sarvam":
        api_key = os.getenv("SARVAM_API_KEY") or os.getenv("SARVAM_API_SUBSCRIPTION_KEY")
        if not api_key:
            raise RuntimeError("Set SARVAM_API_KEY in .env to use Sarvam.")
        model = os.getenv("SARVAM_MODEL", "sarvam-105b")
        client = OpenAI(api_key=api_key, base_url="https://api.sarvam.ai/v2", timeout=180.0, max_retries=1)
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in .env to use OpenAI.")
        model = os.getenv("GPT5_MODEL", "gpt-5.6-luna")
        client = OpenAI(api_key=api_key, timeout=180.0, max_retries=1)
    else:
        raise ValueError("NER_PROVIDER must be 'openai' or 'sarvam'.")
    return provider, model, client


def main(num_sentences=NUM_SENTENCES, ablation="baseline"):
    ablation_settings = {
        "baseline": {"definition_language": "english", "include_examples": True},
        "telugu_definitions": {"definition_language": "telugu", "include_examples": True},
        "zero_shot": {"definition_language": "english", "include_examples": False},
        "few_shot": {"definition_language": "english", "include_examples": True},
    }
    if ablation not in ablation_settings:
        raise ValueError(f"Unknown ablation condition: {ablation}")
    prompt_settings = ablation_settings[ablation]
    provider, model, client = make_client()
    default_workers = "3" if provider == "sarvam" else "1"
    max_workers = max(1, int(os.getenv("NER_MAX_CONCURRENCY", default_workers)))
    output_file = (
        PROJECT_ROOT / "results" /
        f"promptner_{provider}_{ablation}_fine_grained{num_sentences}_results.json"
        if ablation != "baseline"
        else PROJECT_ROOT / "results" /
        f"promptner_{provider}_fine_grained{num_sentences}_results.json"
    )
    df = load_data()
    test_ids = load_test_ids(num_sentences)
    if len(test_ids) < num_sentences:
        raise RuntimeError(f"Expected {num_sentences} test IDs, got {len(test_ids)}.")

    print(f"Provider: {provider} | Model: {model}")
    print(f"Fine-grained taxonomy: MultiCoNER v2 + Telugu-only fallback labels")
    print(f"Ablation: {ablation}")
    print(f"Request concurrency: {max_workers}")
    results = []

    def aggregate_matching(name):
        scores = [r["matching_evaluations"][name] for r in results]
        tp = sum(score["true_positives"] for score in scores)
        fp = sum(score["false_positives"] for score in scores)
        fn = sum(score["false_negatives"] for score in scores)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        aggregate = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if name == "token_wise":
            token_count = sum(score.get("tokens", 0) for score in scores)
            correct_tokens = sum(
                round(score.get("token_accuracy", 0.0) * score.get("tokens", 0))
                for score in scores
            )
            aggregate["token_accuracy"] = (
                correct_tokens / token_count if token_count else 0.0
            )
            aggregate["tokens"] = token_count
        return aggregate

    jobs = []
    for position, sentence_id in enumerate(test_ids, start=1):
        sentence_df = get_sentence(df, sentence_id)
        if sentence_df.empty:
            print(f"[{position:02d}/{len(test_ids)}] sentence_id={sentence_id} not found; skipping.")
            continue

        sentence = reconstruct_sentence(sentence_df)
        gold = extract_gold_entities(sentence_df)
        candidates_before = generate_token_candidates(sentence_df, max_n=MAX_N)
        candidates = filter_candidates(candidates_before)
        prompt = build_fine_grained_prompt(
            sentence, candidates, **prompt_settings
        )
        jobs.append({
            "position": position,
            "sentence_id": sentence_id,
            "sentence_df": sentence_df,
            "sentence": sentence,
            "gold": gold,
            "candidates_before": candidates_before,
            "candidates": candidates,
            "prompt": prompt,
        })

    def request(prompt):
        response = client.responses.create(model=model, input=prompt, store=False)
        return response.output_text or ""

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(request, job["prompt"]) for job in jobs]
        for job, future in zip(jobs, futures):
            position = job["position"]
            sentence_id = job["sentence_id"]
            sentence_df = job["sentence_df"]
            sentence = job["sentence"]
            gold = job["gold"]
            candidates_before = job["candidates_before"]
            candidates = job["candidates"]
            prompt = job["prompt"]
            print(f"\n[{position:02d}/{len(test_ids)}] sentence_id={sentence_id}")
            try:
                raw_output = future.result()
                predictions = parse_fine_grained_output(raw_output, candidates)
                matching_evaluations = evaluate_all_matching(
                    sentence_df, gold, predictions
                )
                evaluation = matching_evaluations["entity_wise"]
                error = None
            except Exception as exc:
                raw_output = None
                predictions = []
                matching_evaluations = evaluate_all_matching(sentence_df, gold, [])
                error = str(exc)
                print(f"  API error: {error}")
            evaluation = matching_evaluations["entity_wise"]

            print(
                f"  Fine predictions: {len(predictions)} | "
                f"F1 token={matching_evaluations['token_wise']['f1']:.4f} "
                f"entity={matching_evaluations['entity_wise']['f1']:.4f} "
                f"partial={matching_evaluations['partial']['f1']:.4f}"
            )
            results.append({
                "sentence_position": position,
                "sentence_id": sentence_id,
                "sentence": sentence,
                "gold_entities": gold,
                "candidate_count_before_filter": len(candidates_before),
                "candidate_count_after_filter": len(candidates),
                "candidates": candidates,
                "prompt": prompt,
                "raw_output": raw_output,
                "predictions": predictions,
                "evaluation": evaluation,
                "matching_evaluations": matching_evaluations,
                **({"error": error} if error else {}),
            })

    matching_approaches = {
        name: aggregate_matching(name)
        for name in ("token_wise", "entity_wise", "partial")
    }
    aggregate = matching_approaches["entity_wise"]
    output = {
        "summary": {
            "provider": provider,
            "model": model,
            "ablation": ablation,
            "definition_language": prompt_settings["definition_language"],
            "few_shot_examples_included": prompt_settings["include_examples"],
            "num_sentences_requested": num_sentences,
            "num_sentences_processed": len(results),
            "request_concurrency": max_workers,
            "scoring": "Exact span + mapped TeluguNER coarse label",
            "fine_taxonomy": "MultiCoNER v2 33 labels plus direct TeluguNER labels",
            "micro_evaluation": aggregate,
            "matching_approaches": matching_approaches,
        },
        "sentences": results,
    }
    output_file.parent.mkdir(exist_ok=True)
    output_file.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nMicro F1: {aggregate['f1']:.4f}")
    for name, scores in matching_approaches.items():
        print(
            f"{name}: P={scores['precision']:.4f} R={scores['recall']:.4f} "
            f"F1={scores['f1']:.4f}"
        )
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
