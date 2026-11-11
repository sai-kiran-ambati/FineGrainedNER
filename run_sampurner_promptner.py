"""Run PromptNER on SampurNER Telugu test sentences (coarse or fine labels)."""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from src.sampurner_evaluation import evaluate_all


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "sampurner" / "test.json"
RESULTS_DIR = ROOT / "results"


def coarse_label(fine):
    """FewNERD/SampurNER hierarchy parent: the component before the first hyphen."""
    return fine.split("-", 1)[0].upper()


def candidates_for(tokens, max_n):
    return [
        {"text": " ".join(tokens[start:end]), "start_token": start + 1, "end_token": end}
        for start in range(len(tokens))
        for end in range(start + 1, min(len(tokens), start + max_n) + 1)
    ]


def make_prompt(sentence, candidates, labels, level, examples):
    label_text = "\n".join(f"- {label}" for label in labels)
    rows = "\n".join(
        f"C{i:03d} | {c['text']} | tokens {c['start_token']}-{c['end_token']}"
        for i, c in enumerate(candidates, 1)
    )
    demo = ""
    if examples:
        demo = "\nExamples:\n" + "\n".join(
            f"Text: {x['text']}\nEntities: " + ", ".join(f"{e['text']} -> {e['label']}" for e in x['entities'])
            for x in examples
        )
    return f"""You are a Telugu named entity recognition system. Identify the exact named entity spans among the candidates.
Use the {level}-grained SampurNER labels below. Choose only one label from the list for each entity. If it is not an entity, output False and NONE.

Allowed labels:
{label_text}
{demo}

Sentence: {sentence}

Candidates:
{rows}

Return one line per candidate in exactly this format:
| Candidate ID | True or False | Label or NONE |
Do not add candidate IDs and do not mark multiple overlapping spans for the same entity. Prefer the complete exact span.
"""


def parse_output(output, candidates, allowed):
    lookup = {f"C{i:03d}": c for i, c in enumerate(candidates, 1)}
    found = {}
    pattern = re.compile(r"^\s*\|?\s*(C\d+)\s*\|\s*(True|False)\s*\|\s*([A-Za-z][A-Za-z0-9_/-]*|NONE)", re.I | re.M)
    for cid, decision, label in pattern.findall(output):
        cid, label = cid.upper(), label.lower()
        if decision.lower() != "true" or cid not in lookup or label not in allowed:
            continue
        c = lookup[cid]
        found[(c["start_token"], c["end_token"])] = {
            **c, "label": label, "candidate_id": cid,
        }
    return list(found.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("coarse", "fine"), default="fine")
    parser.add_argument("--sentences", type=int, default=20)
    parser.add_argument("--max-span-tokens", type=int, default=5)
    parser.add_argument("--few-shot", type=int, default=3)
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    args = parser.parse_args()

    data_file = DATA_FILE.with_name(f"{args.split}.json")
    if not data_file.exists():
        raise SystemExit(f"Missing {data_file}. Run: python prepare_sampurner.py")
    data = json.loads(data_file.read_text(encoding="utf-8"))
    if args.sentences <= 0:
        raise SystemExit("--sentences must be positive")
    samples = data[:args.sentences]
    labels = sorted({e["label"] for row in data for e in row["entities"]})
    if args.level == "coarse":
        labels = sorted({coarse_label(x) for x in labels})
    allowed = {label.lower() for label in labels}
    examples = json.loads(DATA_FILE.with_name("train.json").read_text(encoding="utf-8"))[:args.few_shot] if args.few_shot else []
    if args.level == "coarse":
        examples = [
            {**row, "entities": [{**entity, "label": coarse_label(entity["label"])}
                                  for entity in row["entities"]]}
            for row in examples
        ]
    load_dotenv(ROOT / ".env")
    provider = os.getenv("NER_PROVIDER", "openai").strip().lower()
    if provider == "sarvam":
        key = os.getenv("SARVAM_API_KEY") or os.getenv("SARVAM_API_SUBSCRIPTION_KEY")
        if not key:
            raise RuntimeError("Set SARVAM_API_KEY to use Sarvam.")
        model = os.getenv("SARVAM_MODEL", "sarvam-105b")
        client = OpenAI(api_key=key, base_url="https://api.sarvam.ai/v2", timeout=180.0, max_retries=1)
    elif provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Set OPENAI_API_KEY to use OpenAI.")
        model = os.getenv("GPT5_MODEL", "gpt-5.6-luna")
        client = OpenAI(api_key=key, timeout=180.0, max_retries=1)
    else:
        raise ValueError("NER_PROVIDER must be openai or sarvam")
    default_workers = "3" if provider == "sarvam" else "1"
    max_workers = max(1, int(os.getenv("NER_MAX_CONCURRENCY", default_workers)))
    print(f"Request concurrency: {max_workers}")

    results = []
    jobs = []
    for position, row in enumerate(samples, 1):
        tokens, gold = row["tokens"], row["entities"]
        candidates = candidates_for(tokens, args.max_span_tokens)
        prompt = make_prompt(row["text"], candidates, labels, args.level, examples)
        jobs.append((position, row, gold, candidates, prompt))

    def request(prompt):
        response = client.responses.create(model=model, input=prompt, store=False)
        return response.output_text or ""

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(request, job[4]) for job in jobs]
        for (position, row, gold, candidates, prompt), future in zip(jobs, futures):
            try:
                raw = future.result()
                predictions = parse_output(raw, candidates, allowed)
                matching_evaluations = evaluate_all(
                    gold, predictions, len(row["tokens"]), args.level
                )
                error = None
            except Exception as exc:
                raw, predictions, error = None, [], str(exc)
                matching_evaluations = evaluate_all(
                    gold, [], len(row["tokens"]), args.level
                )
            evaluation = matching_evaluations["entity_wise"]
            print(f"[{position}/{len(samples)}] id={row['sentence_id']} F1={evaluation['f1']:.4f}")
            results.append({"sentence_position": position, "sentence_id": row["sentence_id"],
                            "sentence": row["text"], "tokens": row["tokens"],
                            "gold_entities": gold, "predictions": predictions,
                            "evaluation": evaluation, "matching_evaluations": matching_evaluations,
                            "raw_output": raw, **({"error": error} if error else {})})

    matching_approaches = {}
    for name in ("token_wise", "entity_wise", "partial"):
        scores = [row["matching_evaluations"][name] for row in results]
        tp = sum(item["true_positives"] for item in scores)
        fp = sum(item["false_positives"] for item in scores)
        fn = sum(item["false_negatives"] for item in scores)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        aggregate = {"true_positives": tp, "false_positives": fp, "false_negatives": fn,
                     "precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0.0}
        if name == "token_wise":
            tokens = sum(item["tokens"] for item in scores)
            correct_tokens = sum(round(item["token_accuracy"] * item["tokens"]) for item in scores)
            aggregate["token_accuracy"] = correct_tokens / tokens if tokens else 0.0
            aggregate["tokens"] = tokens
        matching_approaches[name] = aggregate
    aggregate = matching_approaches["entity_wise"]
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / f"promptner_sampurner_{provider}_{args.level}_{args.split}{len(samples)}.json"
    output = {"summary": {"dataset": "SampurNER", "language": "te",
        "provider": provider, "model": model, "level": args.level, "split": args.split,
        "num_sentences": len(results), "request_concurrency": max_workers,
        "labels": labels, "scoring": "Exact, BIO token-wise, and one-to-one partial matching",
        "micro_evaluation": aggregate, "matching_approaches": matching_approaches},
        "sentences": results}
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    comparison_path = RESULTS_DIR / f"sampurner_matching_comparison_{provider}_{args.level}_{args.split}{len(samples)}.json"
    comparison = {
        "summary": output["summary"],
        "sentences": [{
            "sentence_id": row["sentence_id"],
            "matching_evaluations": row["matching_evaluations"],
        } for row in results],
    }
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, score_result in matching_approaches.items():
        print(f"{name}: P={score_result['precision']:.4f} R={score_result['recall']:.4f} F1={score_result['f1']:.4f}")
    print(f"Results saved to {output_path}")
    print(f"Matching comparison saved to {comparison_path}")


if __name__ == "__main__":
    main()
