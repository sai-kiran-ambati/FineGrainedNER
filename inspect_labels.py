import pandas as pd

files = [
    "anonymous_telner/dataset/dataset-combined.csv",
    "anonymous_telner/dataset/medical-dataset.csv",
    "anonymous_telner/dataset/dataset-generic.csv"
]

for file in files:
    print("\n" + "="*70)
    print("File :", file)
    print("="*70)

    df = pd.read_csv(file)

    # counting all unique tags
    print("\nAll tage :")
    print(df['tag'].value_counts())

    # extracting entity types from B-/I- tags
    entity_types = sorted(
        set(
            tag.split("-", 1)[1]
            for tag in df['tag'].dropna().unique()
            if tag != 'O' and "-" in tag
        )
    )

    print("\nEntity types :")
    print(entity_types)

    print("\nNumber of Entity types :", len(entity_types))