import json


FILES = {
    "TRAIN": "data/train.json",
    "VALIDATION": "data/validation.json",
    "TEST": "data/test.json"
}


def get_entity_types(data):
    types = set()

    for item in data:
        for entity in item["entities"]:
            # Only add the label if it exists and is not None
            if entity.get("label") is not None:
                types.add(entity["label"])
            else:
                # Optional: tracking or flagging data errors
                types.add("MISSING_LABEL") 

    return sorted(types)



for name, file in FILES.items():

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    types = get_entity_types(data)

    print("=" * 60)
    print(name)

    print("Sentences:", len(data))
    print("Entity types:", types)