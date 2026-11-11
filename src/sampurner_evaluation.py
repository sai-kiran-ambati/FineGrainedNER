"""Token-wise, exact entity-wise, and partial matching for SampurNER runs."""

from collections import Counter


def _label(label, level):
    label = label.lower()
    return label.split("-", 1)[0] if level == "coarse" else label


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


def evaluate_entity_wise(gold, predictions, level="fine"):
    """Exact inclusive span and label match."""
    gold_keys = Counter(
        (int(x["start_token"]), int(x["end_token"]), _label(x["label"], level))
        for x in gold
    )
    pred_keys = Counter(
        (int(x["start_token"]), int(x["end_token"]), _label(x["label"], level))
        for x in predictions
    )
    tp = sum((gold_keys & pred_keys).values())
    return _metrics(tp, sum((pred_keys - gold_keys).values()), sum((gold_keys - pred_keys).values()))


def _tags(entities, n_tokens, level, *, resolve_overlaps=False):
    tags = ["O"] * n_tokens
    ordered = list(entities)
    if resolve_overlaps:
        ordered.sort(key=lambda x: (
            -(int(x["end_token"]) - int(x["start_token"]) + 1),
            int(x["start_token"]),
            int(x["end_token"]),
        ))
    occupied = set()
    for entity in ordered:
        start, end = int(entity["start_token"]), int(entity["end_token"])
        if start < 1 or end > n_tokens or start > end:
            continue
        positions = set(range(start, end + 1))
        if resolve_overlaps and positions & occupied:
            continue
        label = _label(entity["label"], level)
        tags[start - 1] = f"B-{label}"
        for pos in range(start + 1, end + 1):
            tags[pos - 1] = f"I-{label}"
        occupied.update(positions)
    return tags


def evaluate_token_wise(gold, predictions, n_tokens, level="fine"):
    """BIO token comparison; outside tokens are excluded from P/R/F1."""
    gold_tags = _tags(gold, n_tokens, level)
    pred_tags = _tags(predictions, n_tokens, level, resolve_overlaps=True)
    tp = fp = fn = correct = 0
    for gold_tag, pred_tag in zip(gold_tags, pred_tags):
        if gold_tag == pred_tag:
            correct += 1
            if gold_tag != "O":
                tp += 1
        else:
            if pred_tag != "O":
                fp += 1
            if gold_tag != "O":
                fn += 1
    result = _metrics(tp, fp, fn)
    result["token_accuracy"] = correct / n_tokens if n_tokens else 0.0
    result["tokens"] = n_tokens
    return result


def _partial_pairs(gold, predictions, level):
    adjacency = []
    for pred in predictions:
        p_start, p_end = int(pred["start_token"]), int(pred["end_token"])
        adjacency.append([
            gi for gi, target in enumerate(gold)
            if _label(pred["label"], level) == _label(target["label"], level)
            and p_start <= int(target["end_token"])
            and int(target["start_token"]) <= p_end
        ])
    gold_to_pred = {}

    def assign(pi, visited):
        for gi in adjacency[pi]:
            if gi in visited:
                continue
            visited.add(gi)
            if gi not in gold_to_pred or assign(gold_to_pred[gi], visited):
                gold_to_pred[gi] = pi
                return True
        return False

    for pi in range(len(predictions)):
        assign(pi, set())
    return len(gold_to_pred)


def evaluate_partial(gold, predictions, level="fine"):
    """One-to-one span-overlap match requiring equal fine/coarse labels."""
    matched = _partial_pairs(gold, predictions, level)
    return _metrics(matched, len(predictions) - matched, len(gold) - matched)


def evaluate_all(gold, predictions, n_tokens, level="fine"):
    return {
        "token_wise": evaluate_token_wise(gold, predictions, n_tokens, level),
        "entity_wise": evaluate_entity_wise(gold, predictions, level),
        "partial": evaluate_partial(gold, predictions, level),
    }
