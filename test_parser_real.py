from src.parse_predictions import parse_gpt_output


candidates = [
    {
        "text": "టాక్సీవాలా",
        "start_token": 1,
        "end_token": 1,
    },
    {
        "text": "టాక్సీవాలా ఆ జాబితాలో",
        "start_token": 1,
        "end_token": 3,
    },
    {
        "text": "మాళవిక",
        "start_token": 4,
        "end_token": 4,
    },
]


response = """
| Candidate | Entity | Type | Explanation |
|---|---|---|---|
| C001 | True | OTH | సినిమా పేరు |
| C002 | False | NONE | విస్తృత span |
| C003 | True | PER | వ్యక్తి పేరు |
"""

predictions = parse_gpt_output(response, candidates)

print("Parsed predictions:")
for prediction in predictions:
    print(prediction)