import json
import random
from pathlib import Path


INPUT_FILE = "data/teluguner_generic.json"

TRAIN_FILE = "data/train.json"
VAL_FILE = "data/validation.json"
TEST_FILE = "data/test.json"

SEED = 42


def main():

    # ---------------------------------------------
    # Load dataset
    # ---------------------------------------------

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Total sentences:", len(data))


    # ---------------------------------------------
    # Shuffle reproducibly
    # ---------------------------------------------

    random.seed(SEED)

    random.shuffle(data)


    # ---------------------------------------------
    # Calculate split sizes
    # ---------------------------------------------

    total = len(data)

    train_end = int(total * 0.80)
    val_end = train_end + int(total * 0.10)


    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]


    # ---------------------------------------------
    # Save
    # ---------------------------------------------

    Path("data").mkdir(exist_ok=True)

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(VAL_FILE, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    with open(TEST_FILE, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)


    # ---------------------------------------------
    # Print statistics
    # ---------------------------------------------

    print("\nSplit completed.")

    print("Train:", len(train_data))
    print("Validation:", len(val_data))
    print("Test:", len(test_data))

    print("\nRandom seed:", SEED)


if __name__ == "__main__":
    main()