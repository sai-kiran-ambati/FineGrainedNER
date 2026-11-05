from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_FILE = PROJECT_ROOT / "data" / "test.json"


with open(TEST_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


for item in data[:5]:

    print("=" * 80)
    print("Sentence ID:", item["sentence_id"])
    print(item["text"])

    print("\nEntities:")

    for entity in item["entities"]:

        print(
            f"{entity['start_token']}-"
            f"{entity['end_token']} | "
            f"{entity['text']} -> "
            f"{entity['label']}"
        )