from src.evaluate_predictions import evaluate_predictions


# --------------------------------------------------
# Gold entities
# --------------------------------------------------

gold_entities = [
    {
        "text": "వీఎంసీ",
        "label": "ORG",
        "start_token": 6,
        "end_token": 6
    },
    {
        "text": "25",
        "label": "CARDINAL",
        "start_token": 12,
        "end_token": 12
    }
]


# --------------------------------------------------
# Predictions
# --------------------------------------------------

predictions = [
    {
        "candidate_id": "C015",
        "text": "వీఎంసీ",
        "label": "ORG",
        "start_token": 6,
        "end_token": 6
    },
    {
        "candidate_id": "C039",
        "text": "25",
        "label": "CARDINAL",
        "start_token": 12,
        "end_token": 12
    },
    {
        "candidate_id": "C040",
        "text": "25 మంది",
        "label": "CARDINAL",
        "start_token": 12,
        "end_token": 13
    }
]


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

results = evaluate_predictions(
    gold_entities,
    predictions
)


# --------------------------------------------------
# Print
# --------------------------------------------------

print("=" * 80)
print("EVALUATION")
print("=" * 80)

print(f"True Positives:  {results['true_positives']}")
print(f"False Positives: {results['false_positives']}")
print(f"False Negatives: {results['false_negatives']}")

print(f"Precision:       {results['precision']:.4f}")
print(f"Recall:          {results['recall']:.4f}")
print(f"F1:              {results['f1']:.4f}")