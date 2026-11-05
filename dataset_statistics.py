import pandas as pd


FILE = "anonymous_telner/dataset/dataset-generic.csv"

df = pd.read_csv(FILE)


# --------------------------------------------------
# 1. Number of sentences
# --------------------------------------------------

num_sentences = df["sentence_id"].nunique()

print("=" * 60)
print("DATASET STATISTICS")
print("=" * 60)

print("\nNumber of sentences:", num_sentences)
print("Number of tokens:", len(df))


# --------------------------------------------------
# 2. Entity type distribution
# --------------------------------------------------

entity_df = df[df["tag"] != "O"].copy()

entity_types = (
    entity_df["tag"]
    .str.replace(r"^[BI]-", "", regex=True)
)

print("\nEntity token distribution:")
print(entity_types.value_counts())


# --------------------------------------------------
# 3. Number of entity starts
# --------------------------------------------------

begin_entities = df[df["tag"].str.startswith("B-", na=False)]

print("\nNumber of entities:", len(begin_entities))

print("\nEntity distribution:")
print(
    begin_entities["tag"]
    .str.replace("B-", "", regex=False)
    .value_counts()
)


# --------------------------------------------------
# 4. Number of sentences containing each entity type
# --------------------------------------------------

print("\nSentences containing each entity type:")

sentence_entity_counts = (
    df[df["tag"].str.startswith("B-", na=False)]
    .assign(
        entity_type=lambda x:
        x["tag"].str.replace("B-", "", regex=False)
    )
    .groupby("entity_type")["sentence_id"]
    .nunique()
    .sort_values(ascending=False)
)

print(sentence_entity_counts)