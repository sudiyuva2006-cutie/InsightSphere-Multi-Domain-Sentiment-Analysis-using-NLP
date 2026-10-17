import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

print("=" * 60)
print("INSIGHTSPHERE - TF-IDF FEATURE EXTRACTION")
print("=" * 60)


# --------------------------------------------------
# FUNCTION TO APPLY TF-IDF
# --------------------------------------------------

def apply_tfidf(file_path, domain_name):

    print(f"\nProcessing {domain_name}...")

    data = pd.read_csv(file_path)

    # Take the cleaned review text
    reviews = data["cleaned_review"].fillna("")

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.95
    )

    # Convert text into numerical features
    tfidf_matrix = vectorizer.fit_transform(reviews)

    print("Number of reviews:", len(reviews))
    print("Number of TF-IDF features:", len(vectorizer.get_feature_names_out()))
    print("TF-IDF matrix shape:", tfidf_matrix.shape)

    return vectorizer, tfidf_matrix


# --------------------------------------------------
# MOVIES
# --------------------------------------------------

movies_vectorizer, movies_tfidf = apply_tfidf(
    "outputs/movies_cleaned.csv",
    "Movies"
)


# --------------------------------------------------
# HOTELS
# --------------------------------------------------

hotels_vectorizer, hotels_tfidf = apply_tfidf(
    "outputs/hotels_cleaned.csv",
    "Hotels"
)


# --------------------------------------------------
# TRAVEL
# --------------------------------------------------

travel_vectorizer, travel_tfidf = apply_tfidf(
    "outputs/travel_cleaned.csv",
    "Travel"
)


print("\n" + "=" * 60)
print("TF-IDF FEATURE EXTRACTION COMPLETED!")
print("=" * 60)