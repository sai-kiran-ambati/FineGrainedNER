from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_FILE = (
    PROJECT_ROOT
    / "anonymous_telner"
    / "dataset"
    / "dataset-generic.csv"
)


sentence_id = 12735

df = pd.read_csv(CSV_FILE)

sentence = df[df["sentence_id"] == sentence_id]

print("=" * 80)
print(f"SENTENCE ID: {sentence_id}")
print("=" * 80)

for _, row in sentence.iterrows():

    print(
        f"word_index={row['word_index']:>3} | "
        f"word={row['word']!r:<30} | "
        f"tag={row['tag']}"
    )