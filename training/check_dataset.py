import pandas as pd
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "tasks.csv"

print("=" * 60)
print("TASK DATASET CHECK")
print("=" * 60)

# Load dataset
df = pd.read_csv(DATASET_PATH)

print(f"\nDataset: {DATASET_PATH}")
print(f"Total examples: {len(df)}")

# Columns
print("\nColumns:")
for column in df.columns:
    print(f"  - {column}")

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Duplicate rows
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Categories
print("\nCATEGORY DISTRIBUTION:")
print(df["category"].value_counts())

# Priorities
print("\nPRIORITY DISTRIBUTION:")
print(df["priority"].value_counts())

# Category + priority combinations
print("\nCATEGORY × PRIORITY:")
print(pd.crosstab(df["category"], df["priority"]))

# Duplicate titles
print(f"\nDuplicate titles: {df['title'].duplicated().sum()}")

# Empty strings
print("\nEmpty values:")
for column in ["title", "description", "category", "priority"]:
    empty = (df[column].astype(str).str.strip() == "").sum()
    print(f"  {column}: {empty}")

print("\n" + "=" * 60)
print("CHECK COMPLETE")
print("=" * 60)