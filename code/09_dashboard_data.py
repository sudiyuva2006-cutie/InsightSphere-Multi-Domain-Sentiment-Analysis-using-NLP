import pandas as pd
import os


# ============================================================
# INSIGHTSPHERE - DASHBOARD DATA PREPARATION
# ============================================================

print("=" * 60)
print("INSIGHTSPHERE - DASHBOARD DATA PREPARATION")
print("=" * 60)


# ============================================================
# LOAD FINAL SENTIMENT DATA
# ============================================================

input_file = "outputs/final_sentiment_data.csv"

if not os.path.exists(input_file):
    print("\nERROR: final_sentiment_data.csv not found.")
    print("Please run 08_multi_domain_analysis.py first.")
    exit()


data = pd.read_csv(input_file)


print("\nFinal dataset loaded successfully!")
print("Shape:", data.shape)

print("\nColumns:")
print(data.columns.tolist())


# ============================================================
# SENTIMENT SUMMARY
# ============================================================

sentiment_summary = (
    data["sentiment"]
    .value_counts()
    .reset_index()
)

sentiment_summary.columns = [
    "sentiment",
    "count"
]


print("\nOverall Sentiment Summary:")
print(sentiment_summary)


# ============================================================
# DOMAIN + SENTIMENT SUMMARY
# ============================================================

domain_summary = (
    data
    .groupby(["domain", "sentiment"])
    .size()
    .reset_index(name="count")
)


print("\nDomain-wise Sentiment Summary:")
print(domain_summary)


# ============================================================
# DOMAIN TOTALS
# ============================================================

domain_totals = (
    data["domain"]
    .value_counts()
    .reset_index()
)

domain_totals.columns = [
    "domain",
    "total_reviews"
]


print("\nDomain Totals:")
print(domain_totals)


# ============================================================
# SAVE DASHBOARD FILES
# ============================================================

os.makedirs("dashboard", exist_ok=True)


sentiment_summary.to_csv(
    "dashboard/sentiment_summary_dashboard.csv",
    index=False
)


domain_summary.to_csv(
    "dashboard/domain_sentiment_dashboard.csv",
    index=False
)


domain_totals.to_csv(
    "dashboard/domain_totals_dashboard.csv",
    index=False
)


print("\nDashboard files created successfully:")

print("dashboard/sentiment_summary_dashboard.csv")
print("dashboard/domain_sentiment_dashboard.csv")
print("dashboard/domain_totals_dashboard.csv")


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("DASHBOARD DATA PREPARATION COMPLETED")
print("=" * 60)