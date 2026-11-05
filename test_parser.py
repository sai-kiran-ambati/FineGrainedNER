from src.parse_predictions import parse_gpt_output


candidates = [
    {
        "text": "విజ‌య‌వాడ",
        "start_token": 1,
        "end_token": 1
    },
    {
        "text": "విజ‌య‌వాడ న‌గ‌ర‌పాల‌క",
        "start_token": 1,
        "end_token": 2
    },
    {
        "text": "విజ‌య‌వాడ న‌గ‌ర‌పాల‌క సంస్థ",
        "start_token": 1,
        "end_token": 3
    },
    {
        "text": "వీఎంసీ",
        "start_token": 6,
        "end_token": 6
    }
]


fake_response = """
Candidate ID: C001
True/False: False
Explanation: Not an entity.
Type: NONE

Candidate ID: C004
True/False: True
Explanation: Organization abbreviation.
Type: ORG
"""


predictions = parse_gpt_output(
    fake_response,
    candidates
)


print("=" * 80)
print("PARSED PREDICTIONS")
print("=" * 80)

for prediction in predictions:
    print(prediction)