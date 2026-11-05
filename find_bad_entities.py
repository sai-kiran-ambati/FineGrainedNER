import json


FILES = {
    "TRAIN": "data/train.json",
    "VALIDATION": "data/validation.json",
    "TEST": "data/test.json"
}


for split_name, file in FILES.items():

    print("=" * 70)
    print(split_name)

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    bad_count = 0

    for item in data:

        for entity in item["entities"]:

            if entity.get("label") is None:

                bad_count += 1

                print("\nBAD ENTITY")
                print("Sentence ID:", item["sentence_id"])
                print("Sentence:", item["text"])
                print("Entity:", entity)

                if bad_count >= 10:
                    break

        if bad_count >= 10:
            break

    print("\nBad entities found:", bad_count)