import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


print("=" * 60)
print("INSIGHTSPHERE - K-MEANS CLUSTERING")
print("=" * 60)


# --------------------------------------------------
# LOAD MOVIE DATASET
# --------------------------------------------------

data = pd.read_csv("outputs/movies_cleaned.csv")

reviews = data["cleaned_review"].fillna("")


# --------------------------------------------------
# TF-IDF FEATURE EXTRACTION
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.95
)

X_tfidf = vectorizer.fit_transform(reviews)


print("\nTF-IDF Matrix Shape:", X_tfidf.shape)


# --------------------------------------------------
# K-MEANS MODEL
# --------------------------------------------------

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = model.fit_predict(X_tfidf)


# --------------------------------------------------
# ADD CLUSTER TO DATASET
# --------------------------------------------------

data["cluster"] = clusters


# --------------------------------------------------
# DISPLAY CLUSTER INFORMATION
# --------------------------------------------------

print("\nCluster Distribution:")

print(data["cluster"].value_counts().sort_index())


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

data.to_csv(
    "outputs/movies_kmeans_results.csv",
    index=False
)


print("\nClustered dataset saved to:")
print("outputs/movies_kmeans_results.csv")


print("\n" + "=" * 60)
print("K-MEANS CLUSTERING COMPLETED!")
print("=" * 60)