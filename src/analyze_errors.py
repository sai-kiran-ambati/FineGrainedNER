import json
from pathlib import Path


RESULT_FILE = (
    Path(__file__).resolve().parent.parent
    / "results"
    / "promptner_gpt20_results.json"
)


def entity_key(entity):
    return (
        entity["start_token"],
        entity["end_token"],
        entity["label"],
    )


def main():

    with open(RESULT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    print("=" * 80)
    print("PROMPTNER ERROR ANALYSIS")
    print("=" * 80)

    for result in data["sentences"]:

        gold = result["gold_entities"]
        predictions = result["predictions"]

        gold_set = {
            entity_key(e)
            for e in gold
        }

        pred_set = {
            entity_key(p)
            for p in predictions
        }

        tp = gold_set & pred_set
        fp = pred_set - gold_set
        fn = gold_set - pred_set

        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)

        if not fp and not fn:
            continue

        print("\n" + "=" * 80)
        print(f"SENTENCE ID: {result['sentence_id']}")
        print("=" * 80)

        print("\nSentence:")
        print(result["sentence"])

        print("\nGOLD:")
        for entity in gold:
            marker = "✓" if entity_key(entity) in tp else "✗"
            print(
                f"  {marker} "
                f"{entity['text']} "
                f"→ {entity['label']} "
                f"[tokens {entity['start_token']}-{entity['end_token']}]"
            )

        print("\nPREDICTIONS:")
        for prediction in predictions:
            marker = "✓" if entity_key(prediction) in tp else "✗"
            print(
                f"  {marker} "
                f"{prediction['text']} "
                f"→ {prediction['label']} "
                f"[tokens {prediction['start_token']}-"
                f"{prediction['end_token']}]"
            )

        if fp:
            print("\nFALSE POSITIVES:")
            for start, end, label in sorted(fp):
                matching = [
                    p for p in predictions
                    if (
                        p["start_token"],
                        p["end_token"],
                        p["label"],
                    ) == (start, end, label)
                ]

                if matching:
                    print(
                        f"  {matching[0]['text']} "
                        f"→ {label} "
                        f"[tokens {start}-{end}]"
                    )

        if fn:
            print("\nFALSE NEGATIVES:")
            for start, end, label in sorted(fn):
                matching = [
                    g for g in gold
                    if (
                        g["start_token"],
                        g["end_token"],
                        g["label"],
                    ) == (start, end, label)
                ]

                if matching:
                    print(
                        f"  {matching[0]['text']} "
                        f"→ {label} "
                        f"[tokens {start}-{end}]"
                    )

    print("\n" + "=" * 80)
    print("TOTALS")
    print("=" * 80)

    print(f"TP: {total_tp}")
    print(f"FP: {total_fp}")
    print(f"FN: {total_fn}")

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp
        else 0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")


if __name__ == "__main__":
    main()
