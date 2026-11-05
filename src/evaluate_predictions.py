def evaluate_predictions(gold_entities, predictions):
    """
    Exact entity-level evaluation.

    A prediction is correct only when:

        start_token matches
        end_token matches
        label matches
    """

    gold_set = {
        (
            entity["start_token"],
            entity["end_token"],
            entity["label"]
        )
        for entity in gold_entities
    }

    prediction_set = {
        (
            prediction["start_token"],
            prediction["end_token"],
            prediction["label"]
        )
        for prediction in predictions
    }

    true_positives = len(
        gold_set & prediction_set
    )

    false_positives = len(
        prediction_set - gold_set
    )

    false_negatives = len(
        gold_set - prediction_set
    )

    precision = (
        true_positives /
        (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0.0
    )

    recall = (
        true_positives /
        (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }