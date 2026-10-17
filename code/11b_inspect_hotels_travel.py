import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

files = ["hotels.csv", "travel.csv"]

print("=" * 70)
print("INSIGHTSPHERE - HOTEL & TRAVEL DATASET INSPECTION")
print("=" * 70)

for filename in files:

    path = os.path.join(DATASET_DIR, filename)

    print("\n" + "=" * 25)
    print(filename.upper())
    print("=" * 25)

    if not os.path.exists(path):
        print("❌ File not found")
        continue

    # Try common encodings
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding)
            print(f"✓ Encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"❌ Error: {e}")
            break

    if df is None:
        print("❌ Could not read this CSV with the available encodings.")
        continue

    print(f"\nRows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn Names:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nFirst 3 Rows:")
    print(df.head(3).to_string())

print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)