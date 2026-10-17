import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))


# --------------------------------------------------
# TEXT CLEANING FUNCTION
# --------------------------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)       # Remove HTML tags
    text = re.sub(r"[^a-zA-Z\s]", " ", text) # Remove punctuation/numbers
    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# --------------------------------------------------
# MOVIES
# --------------------------------------------------

print("\nProcessing Movies...")

movies = pd.read_csv("datasets/movies.csv")

movies = movies.dropna(subset=["review", "sentiment"])
movies = movies.drop_duplicates(subset=["review"])

movies["cleaned_review"] = movies["review"].apply(clean_text)
movies["domain"] = "Movies"

movies = movies[[
    "domain",
    "review",
    "cleaned_review",
    "sentiment"
]]

movies.to_csv(
    "outputs/movies_cleaned.csv",
    index=False
)

print("Movies completed:", movies.shape)


# --------------------------------------------------
# HOTELS
# --------------------------------------------------

print("\nProcessing Hotels...")

hotels = pd.read_csv(
    "datasets/hotels.csv",
    encoding="latin1"
)

hotels = hotels.dropna(
    subset=[
        "Positive_Review",
        "Negative_Review",
        "Reviewer_Score"
    ]
)

# Combine positive and negative review text
hotels["review"] = (
    hotels["Positive_Review"].fillna("")
    + " "
    + hotels["Negative_Review"].fillna("")
)

# Create sentiment from reviewer score
def hotel_sentiment(score):

    if score >= 7:
        return "positive"
    elif score <= 4:
        return "negative"
    else:
        return "neutral"


hotels["sentiment"] = hotels["Reviewer_Score"].apply(
    hotel_sentiment
)

hotels["cleaned_review"] = hotels["review"].apply(
    clean_text
)

hotels["domain"] = "Hotels"

hotels = hotels[[
    "domain",
    "review",
    "cleaned_review",
    "sentiment",
    "Reviewer_Score",
    "Hotel_Name"
]]

# Use a manageable sample for ML processing
hotels = hotels.sample(
    n=min(50000, len(hotels)),
    random_state=42
)

hotels.to_csv(
    "outputs/hotels_cleaned.csv",
    index=False
)

print("Hotels completed:", hotels.shape)


# --------------------------------------------------
# TRAVEL
# --------------------------------------------------

print("\nProcessing Travel...")

travel = pd.read_csv(
    "datasets/travel.csv",
    encoding="latin1"
)

travel = travel.dropna(
    subset=["Review", "Rating"]
)

travel = travel.drop_duplicates(
    subset=["Review"]
)


# Convert rating into sentiment
def travel_sentiment(rating):

    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"


travel["sentiment"] = travel["Rating"].apply(
    travel_sentiment
)

travel["cleaned_review"] = travel["Review"].apply(
    clean_text
)

travel["domain"] = "Travel"

travel = travel[[
    "domain",
    "Review",
    "cleaned_review",
    "sentiment",
    "Rating"
]]

travel.to_csv(
    "outputs/travel_cleaned.csv",
    index=False
)

print("Travel completed:", travel.shape)


# --------------------------------------------------
# FINAL MESSAGE
# --------------------------------------------------

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nCreated files:")
print("1. outputs/movies_cleaned.csv")
print("2. outputs/hotels_cleaned.csv")
print("3. outputs/travel_cleaned.csv")