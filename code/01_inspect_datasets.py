import pandas as pd

print("=" * 60)
print("INSIGHTSPHERE - DATASET INSPECTION")
print("=" * 60)

# -------------------------
# MOVIES
# -------------------------
print("\n🎬 MOVIES DATASET")

movies = pd.read_csv("datasets/movies.csv")

print("Shape:", movies.shape)
print("Columns:")
print(movies.columns.tolist())

print("\nFirst 3 rows:")
print(movies.head(3))


# -------------------------
# HOTELS
# -------------------------
print("\n🏨 HOTELS DATASET")

hotels = pd.read_csv("datasets/hotels.csv")

print("Shape:", hotels.shape)
print("Columns:")
print(hotels.columns.tolist())

print("\nFirst 3 rows:")
print(hotels.head(3))


# -------------------------
# TRAVEL
# -------------------------
print("\n✈️ TRAVEL DATASET")

travel = pd.read_csv("datasets/travel.csv", encoding="latin1")

print("Shape:", travel.shape)
print("Columns:")
print(travel.columns.tolist())

print("\nFirst 3 rows:")
print(travel.head(3))


print("\n" + "=" * 60)
print("DATASET INSPECTION COMPLETED")
print("=" * 60)