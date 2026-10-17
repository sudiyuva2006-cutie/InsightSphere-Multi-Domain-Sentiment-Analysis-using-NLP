import pandas as pd

print("=" * 70)
print("INSIGHTSPHERE - MULTI-DOMAIN SENTIMENT ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD CLEANED DATASETS
# ============================================================

print("\nLoading datasets...")

movies = pd.read_csv("outputs/movies_cleaned.csv")
hotels = pd.read_csv("outputs/hotels_cleaned.csv")
travel = pd.read_csv("outputs/travel_cleaned.csv")

print("Movies:", movies.shape)
print("Hotels:", hotels.shape)
print("Travel:", travel.shape)


# ============================================================
# SELECT COMMON COLUMNS
# ============================================================

movies_final = movies[
    ["domain", "review", "cleaned_review", "sentiment"]
].copy()


hotels_final = hotels[
    ["domain", "review", "cleaned_review", "sentiment"]
].copy()


travel_final = travel[
    ["domain", "Review", "cleaned_review", "sentiment"]
].copy()

# Rename Travel review column
travel_final.rename(
    columns={"Review": "review"},
    inplace=True
)


# ============================================================
# COMBINE ALL DOMAINS
# ============================================================

print("\nCombining datasets...")

final_data = pd.concat(
    [
        movies_final,
        hotels_final,
        travel_final
    ],
    ignore_index=True
)


# Remove empty reviews
final_data = final_data.dropna(
    subset=["review", "sentiment"]
)


# ============================================================
# SENTIMENT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SENTIMENT SUMMARY")
print("=" * 70)

summary = (
    final_data
    .groupby(["domain", "sentiment"])
    .size()
    .reset_index(name="count")
)


# Calculate percentage within each domain
summary["percentage"] = (
    summary.groupby("domain")["count"]
    .transform(lambda x: (x / x.sum()) * 100)
)


summary["percentage"] = summary["percentage"].round(2)


print("\nDomain-wise Sentiment:")

print(summary.to_string(index=False))


# ============================================================
# SAVE FINAL DATASET
# ============================================================

final_data.to_csv(
    "outputs/final_sentiment_data.csv",
    index=False
)


# Save summary for Power BI
summary.to_csv(
    "outputs/sentiment_summary.csv",
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("MULTI-DOMAIN ANALYSIS COMPLETED!")
print("=" * 70)

print("\nFinal dataset:")
print("outputs/final_sentiment_data.csv")

print("\nDashboard summary:")
print("outputs/sentiment_summary.csv")

print("\nFinal combined shape:")
print(final_data.shape)

print("\nDomains:")
print(final_data["domain"].unique())

print("\nSentiments:")
print(final_data["sentiment"].unique())

print("\n" + "=" * 70)