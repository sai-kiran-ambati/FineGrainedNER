import pandas as pd


FILE = "anonymous_telner/dataset/dataset-generic.csv"

df = pd.read_csv(FILE)

invalid = []

for sentence_id, group in df.groupby("sentence_id"):

    group = group.sort_values("word_index")

    current_type = None

    for _, row in group.iterrows():

        tag = row["tag"]

        if tag == "O":
            current_type = None

        elif tag.startswith("B-"):
            current_type = tag[2:]

        elif tag.startswith("I-"):

            entity_type = tag[2:]

            if current_type is None:
                invalid.append(
                    (
                        sentence_id,
                        row["word"],
                        tag,
                        "I-tag without B-tag"
                    )
                )

            elif entity_type != current_type:
                invalid.append(
                    (
                        sentence_id,
                        row["word"],
                        tag,
                        f"Expected I-{current_type}"
                    )
                )

            current_type = entity_type


print("Invalid BIO transitions:", len(invalid))

print("\nFirst 20:")

for item in invalid[:20]:
    print(item)