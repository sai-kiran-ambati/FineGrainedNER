import pandas as pd

files = [
    "anonymous_telner/dataset/dataset-combined.csv",
    "anonymous_telner/dataset/medical-dataset.csv",
    "anonymous_telner/dataset/dataset-generic.csv"
]

for file in files:
    print("\n" + "="*80)
    print("File :", file)
    print("="*80)

    df = pd.read_csv(file)

    print("\nShape:")
    print(df.shape)

    print("\nColumns")
    print(df.columns.tolist())

    print("\nFirst 5 rows :")
    print(df.head())

    print("\nData types :")
    print(df.dtypes)