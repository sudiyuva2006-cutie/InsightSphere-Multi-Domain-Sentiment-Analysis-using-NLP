import pandas as pd
import os
import re

# ============================================================
# INSIGHTSPHERE - FINAL SIX DOMAIN PREPROCESSING
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 75)
print("INSIGHTSPHERE - SIX DOMAIN SENTIMENT PREPROCESSING")
print("=" * 75)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_csv_safely(path):
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

    for encoding in encodings:
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                engine="python",
                on_bad_lines="skip"
            )
            print(f"   Encoding: {encoding}")
            return df
        except Exception:
            continue

    return None


def sentiment_from_rating(rating):
    try:
        rating = float(rating)
    except:
        return None

    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"


all_data = []


# ============================================================
# 1. MOVIES
# ============================================================

print("\n[1/6] Processing Movies...")

movies_path = os.path.join(DATASET_DIR, "movies.csv")

movies = read_csv_safely(movies_path)

if movies is not None:

    movies["review"] = movies["review"].apply(clean_text)

    movies["sentiment"] = (
        movies["sentiment"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    movies = movies[
        movies["review"].ne("") &
        movies["sentiment"].isin(
            ["positive", "negative", "neutral"]
        )
    ]

    movies_final = movies[
        ["review", "sentiment"]
    ].copy()

    movies_final.insert(0, "domain", "Movies")

    all_data.append(movies_final)

    print(
        f"   Movies processed: {len(movies_final)} rows"
    )


# ============================================================
# 2. HOTELS
# ============================================================

print("\n[2/6] Processing Hotels...")

hotels_path = os.path.join(DATASET_DIR, "hotels.csv")

hotels = read_csv_safely(hotels_path)

if hotels is not None:

    hotel_data = []

    for _, row in hotels.iterrows():

        positive = clean_text(row["Positive_Review"])
        negative = clean_text(row["Negative_Review"])

        review = ""

        if positive:
            review += positive

        if negative:
            if review:
                review += " "

            review += negative

        if not review:
            continue

        try:
            score = float(row["Reviewer_Score"])
        except:
            continue

        if score <= 4:
            sentiment = "negative"
        elif score < 7:
            sentiment = "neutral"
        else:
            sentiment = "positive"

        hotel_data.append(
            {
                "domain": "Hotels",
                "review": review,
                "sentiment": sentiment
            }
        )

    hotels_final = pd.DataFrame(hotel_data)

    all_data.append(hotels_final)

    print(
        f"   Hotels processed: {len(hotels_final)} rows"
    )


# ============================================================
# 3. TRAVEL
# ============================================================

print("\n[3/6] Processing Travel...")

travel_path = os.path.join(DATASET_DIR, "travel.csv")

travel = read_csv_safely(travel_path)

if travel is not None:

    travel_data = []

    for _, row in travel.iterrows():

        review = clean_text(row["Review"])

        if not review:
            continue

        sentiment = sentiment_from_rating(row["Rating"])

        if sentiment is None:
            continue

        travel_data.append(
            {
                "domain": "Travel",
                "review": review,
                "sentiment": sentiment
            }
        )

    travel_final = pd.DataFrame(travel_data)

    all_data.append(travel_final)

    print(
        f"   Travel processed: {len(travel_final)} rows"
    )


# ============================================================
# 4. FOOD
# ============================================================

print("\n[4/6] Processing Food...")

food_path = os.path.join(
    DATASET_DIR,
    "food_reviews.xlsx"
)

if os.path.exists(food_path):

    try:

        food = pd.read_excel(food_path)

        print("   Food Excel file loaded successfully.")

        food_data = []

        for _, row in food.iterrows():

            review = clean_text(row["Text"])

            if not review:
                continue

            sentiment = sentiment_from_rating(
                row["Rating"]
            )

            if sentiment is None:
                continue

            food_data.append(
                {
                    "domain": "Food",
                    "review": review,
                    "sentiment": sentiment
                }
            )

        food_final = pd.DataFrame(food_data)

        all_data.append(food_final)

        print(
            f"   Food processed: {len(food_final)} rows"
        )

    except Exception as e:

        print(
            f"   WARNING: Food processing failed: {e}"
        )

else:

    print("   WARNING: food_reviews.xlsx not found.")


# ============================================================
# 5. BOOKS
# ============================================================

print("\n[5/6] Processing Books...")

books_path = os.path.join(
    DATASET_DIR,
    "book_reviews.csv"
)

books = read_csv_safely(books_path)

if books is not None:

    books_data = []

    for _, row in books.iterrows():

        review = clean_text(row["ReviewText"])

        if not review:
            continue

        try:
            sentiment_value = int(row["Sentiment"])
        except:
            continue

        if sentiment_value == 0:
            sentiment = "negative"
        elif sentiment_value == 1:
            sentiment = "neutral"
        elif sentiment_value == 2:
            sentiment = "positive"
        else:
            continue

        books_data.append(
            {
                "domain": "Books",
                "review": review,
                "sentiment": sentiment
            }
        )

    books_final = pd.DataFrame(books_data)

    all_data.append(books_final)

    print(
        f"   Books processed: {len(books_final)} rows"
    )


# ============================================================
# 6. PRODUCTS
# ============================================================

print("\n[6/6] Processing Products...")

products_path = os.path.join(
    DATASET_DIR,
    "product_reviews.csv"
)

products = read_csv_safely(products_path)

if products is not None:

    products_data = []

    for _, row in products.iterrows():

        review = clean_text(row["reviews.text"])

        if not review:
            continue

        sentiment = sentiment_from_rating(
            row["reviews.rating"]
        )

        if sentiment is None:
            continue

        products_data.append(
            {
                "domain": "Products",
                "review": review,
                "sentiment": sentiment
            }
        )

    products_final = pd.DataFrame(products_data)

    all_data.append(products_final)

    print(
        f"   Products processed: {len(products_final)} rows"
    )


# ============================================================
# COMBINE ALL SIX DOMAINS
# ============================================================

print("\n" + "=" * 75)
print("COMBINING SIX DOMAINS")
print("=" * 75)

if len(all_data) != 6:

    print(
        f"\nWARNING: Only {len(all_data)} of 6 domains were processed."
    )

else:

    print("\nAll six domains processed successfully.")


final_data = pd.concat(
    all_data,
    ignore_index=True
)


# ============================================================
# FINAL CLEANING
# ============================================================

final_data["review"] = final_data["review"].apply(clean_text)

final_data = final_data[
    final_data["review"].ne("") &
    final_data["sentiment"].isin(
        ["positive", "negative", "neutral"]
    )
]

final_data = final_data[
    ["domain", "review", "sentiment"]
]


# ============================================================
# SAVE FINAL DATASET
# ============================================================

final_output = os.path.join(
    OUTPUT_DIR,
    "final_sentiment_data_six_domains.csv"
)

final_data.to_csv(
    final_output,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"\nFinal dataset saved:"
)

print(final_output)

print(
    f"\nTotal rows: {len(final_data)}"
)


# ============================================================
# SENTIMENT SUMMARY
# ============================================================

sentiment_summary = (
    final_data
    .groupby("sentiment")
    .size()
    .reset_index(name="count")
)

sentiment_output = os.path.join(
    OUTPUT_DIR,
    "sentiment_summary_six_domains.csv"
)

sentiment_summary.to_csv(
    sentiment_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# DOMAIN SUMMARY
# ============================================================

domain_summary = (
    final_data
    .groupby("domain")
    .size()
    .reset_index(name="count")
)

domain_output = os.path.join(
    OUTPUT_DIR,
    "domain_summary_six_domains.csv"
)

domain_summary.to_csv(
    domain_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# DOMAIN + SENTIMENT SUMMARY
# ============================================================

domain_sentiment_summary = (
    final_data
    .groupby(
        ["domain", "sentiment"]
    )
    .size()
    .reset_index(name="count")
)

domain_sentiment_output = os.path.join(
    OUTPUT_DIR,
    "domain_sentiment_summary_six_domains.csv"
)

domain_sentiment_summary.to_csv(
    domain_sentiment_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("FINAL SIX-DOMAIN VALIDATION")
print("=" * 75)

print("\nDomains:")
print(
    final_data["domain"].unique().tolist()
)

print("\nSentiments:")
print(
    final_data["sentiment"].unique().tolist()
)

print("\nRows by domain:")
print(
    final_data["domain"]
    .value_counts()
    .to_string()
)

print("\nRows by sentiment:")
print(
    final_data["sentiment"]
    .value_counts()
    .to_string()
)

print("\nDomain + Sentiment:")
print(
    domain_sentiment_summary.to_string(
        index=False
    )
)

expected_domains = {
    "Movies",
    "Hotels",
    "Travel",
    "Food",
    "Books",
    "Products"
}

actual_domains = set(
    final_data["domain"].unique()
)

if actual_domains == expected_domains:

    print("\n" + "=" * 75)
    print("SUCCESS: ALL SIX DOMAINS ARE PRESENT")
    print("SIX-DOMAIN PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 75)

else:

    missing = expected_domains - actual_domains

    print("\nWARNING: Missing domains:")
    print(missing)

print("\nDone.")