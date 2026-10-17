import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOOD_PATH = os.path.join(BASE_DIR, "datasets", "food_reviews.xlsx")

print("=" * 70)
print("INSIGHTSPHERE - FOOD DATASET INSPECTION")
print("=" * 70)

if not os.path.exists(FOOD_PATH):
    print("❌ food_reviews.xlsx not found.")
else:
    try:
        food = pd.read_excel(FOOD_PATH)

        print("\n✓ Food Excel file opened successfully.")

        print(f"\nRows: {food.shape[0]}")
        print(f"Columns: {food.shape[1]}")

        print("\nColumn Names:")
        for col in food.columns:
            print(f"  - {col}")

        print("\nFirst 5 Rows:")
        print(food.head(5).to_string())

        print("\nMissing Values:")
        print(food.isnull().sum())

    except Exception as e:
        print("\n❌ ERROR:")
        print(e)

print("\n" + "=" * 70)
print("FOOD INSPECTION COMPLETED")
print("=" * 70)