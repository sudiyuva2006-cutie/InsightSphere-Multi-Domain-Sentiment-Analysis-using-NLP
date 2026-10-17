import pandas as pd
import os

datasets = {
    "Food": "datasets/food_reviews.csv",
    "Books": "datasets/book_reviews.csv",
    "Products": "datasets/product_reviews.csv"
}

print("=" * 70)
print("INSIGHTSPHERE - NEW DOMAIN DATASET INSPECTION")
print("=" * 70)

for domain, path in datasets.items():

    print(f"\n{'=' * 20} {domain.upper()} {'=' * 20}")

    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        continue

    try:
        df = pd.read_csv(path)

        print(f"\nRows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")

        print("\nColumn Names:")
        for col in df.columns:
            print(f"  - {col}")

        print("\nMissing Values:")
        print(df.isnull().sum())

        print("\nFirst 3 Rows:")
        print(df.head(3).to_string())

    except Exception as e:
        print(f"❌ Error reading file: {e}")

print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)