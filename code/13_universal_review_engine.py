# ============================================================
# INSIGHTSPHERE - UNIVERSAL WEB REVIEW ENGINE
# Step 13
# ============================================================

import os
import re
import time
import joblib
import pandas as pd

from dotenv import load_dotenv
from tavily import TavilyClient


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "naive_bayes_model_six_domains.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_vectorizer_six_domains.pkl"
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(os.path.join(BASE_DIR, ".env"))

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


if not TAVILY_API_KEY:
    raise RuntimeError(
        "\nTAVILY_API_KEY was not found.\n"
        "Create a .env file inside the InsightSphere folder and add:\n\n"
        "TAVILY_API_KEY=YOUR_NEW_TAVILY_KEY\n"
    )


# ============================================================
# CHECK MODEL FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}\n"
    )

if not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(
        f"\nTF-IDF vectorizer not found:\n{VECTORIZER_PATH}\n"
    )


# ============================================================
# LOAD ML MODEL
# ============================================================

print("\nLoading InsightSphere ML model...")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("ML model loaded successfully.")
print("TF-IDF vectorizer loaded successfully.")


# ============================================================
# TAVILY CLIENT
# ============================================================

tavily = TavilyClient(api_key=TAVILY_API_KEY)


# ============================================================
# SEARCH QUERIES
# ============================================================

def build_search_queries(item):

    return [
        f'"{item}" reviews',
        f'"{item}" customer reviews',
        f'"{item}" user reviews',
        f'"{item}" ratings reviews',
        f'"{item}" pros cons',
        f'"{item}" experience review'
    ]


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = str(text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# CHECK WHETHER TEXT LOOKS LIKE REVIEW CONTENT
# ============================================================

def looks_like_review(text):

    text_lower = text.lower()

    review_words = [
        "review",
        "rating",
        "rated",
        "stars",
        "experience",
        "liked",
        "love",
        "loved",
        "dislike",
        "disliked",
        "recommend",
        "recommended",
        "worth",
        "quality",
        "service",
        "good",
        "great",
        "bad",
        "poor",
        "excellent",
        "terrible",
        "amazing",
        "awful",
        "pros",
        "cons"
    ]

    return any(word in text_lower for word in review_words)


# ============================================================
# SPLIT TEXT INTO SENTENCES
# ============================================================

def split_sentences(text):

    text = clean_text(text)

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    cleaned = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence) < 25:
            continue

        if len(sentence) > 500:
            continue

        cleaned.append(sentence)

    return cleaned


# ============================================================
# SEARCH WEB
# ============================================================

def search_web(item):

    print("\nSearching the web...")
    print(f"Item: {item}")

    queries = build_search_queries(item)

    results = []

    seen_urls = set()

    for query in queries:

        print(f"  Searching: {query}")

        try:

            response = tavily.search(
                query=query,
                search_depth="advanced",
                max_results=5,
                include_answer=False,
                include_raw_content=True
            )

            search_results = response.get(
                "results",
                []
            )

            for result in search_results:

                url = result.get("url", "").strip()

                if not url:
                    continue

                if url in seen_urls:
                    continue

                seen_urls.add(url)

                results.append({
                    "title": result.get(
                        "title",
                        ""
                    ),
                    "url": url,
                    "content": result.get(
                        "raw_content",
                        ""
                    ) or result.get(
                        "content",
                        ""
                    )
                })

        except Exception as error:

            print(
                f"  Search warning: {error}"
            )

    return results


# ============================================================
# EXTRACT REVIEW SENTENCES
# ============================================================

def extract_review_sentences(results):

    review_sentences = []

    seen_sentences = set()

    for result in results:

        content = result.get(
            "content",
            ""
        )

        sentences = split_sentences(content)

        for sentence in sentences:

            normalized = sentence.lower().strip()

            if normalized in seen_sentences:
                continue

            if not looks_like_review(sentence):
                continue

            seen_sentences.add(normalized)

            review_sentences.append({
                "text": sentence,
                "source": result.get(
                    "title",
                    "Web source"
                ),
                "url": result.get(
                    "url",
                    ""
                )
            })

    return review_sentences


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def analyze_sentiment(review_sentences):

    if not review_sentences:

        return {
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "total": 0,
            "confidence": 0,
            "details": []
        }

    texts = [
        item["text"]
        for item in review_sentences
    ]

    predictions = model.predict(
        vectorizer.transform(texts)
    )

    probabilities = model.predict_proba(
        vectorizer.transform(texts)
    )

    positive = 0
    neutral = 0
    negative = 0

    confidence_values = []

    details = []

    for item, prediction, probability in zip(
        review_sentences,
        predictions,
        probabilities
    ):

        sentiment = str(prediction).lower()

        confidence = float(
            max(probability) * 100
        )

        confidence_values.append(
            confidence
        )

        if sentiment == "positive":
            positive += 1

        elif sentiment == "negative":
            negative += 1

        else:
            neutral += 1

        details.append({
            "text": item["text"],
            "sentiment": sentiment,
            "confidence": round(
                confidence,
                2
            ),
            "source": item["source"],
            "url": item["url"]
        })

    total = len(predictions)

    average_confidence = (
        sum(confidence_values)
        / len(confidence_values)
    )

    return {
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "total": total,
        "confidence": round(
            average_confidence,
            2
        ),
        "details": details
    }


# ============================================================
# OVERALL SENTIMENT
# ============================================================

def get_overall_sentiment(analysis):

    positive = analysis["positive"]
    neutral = analysis["neutral"]
    negative = analysis["negative"]

    if positive == 0 and negative == 0 and neutral == 0:
        return "Insufficient Review Data"

    if positive >= negative and positive >= neutral:
        return "Positive"

    if negative >= positive and negative >= neutral:
        return "Negative"

    return "Neutral"


# ============================================================
# CREATE SUMMARY
# ============================================================

def create_summary(item, analysis):

    total = analysis["total"]

    if total == 0:

        return (
            f"InsightSphere could not find enough "
            f"review-related information about "
            f"{item} to produce a reliable sentiment summary."
        )

    positive_percentage = (
        analysis["positive"] / total
    ) * 100

    negative_percentage = (
        analysis["negative"] / total
    ) * 100

    neutral_percentage = (
        analysis["neutral"] / total
    ) * 100

    overall = get_overall_sentiment(
        analysis
    )

    return (
        f"Overall sentiment for {item} is {overall}. "
        f"Based on {total} review-related text segments, "
        f"{positive_percentage:.1f}% were classified as positive, "
        f"{negative_percentage:.1f}% as negative, and "
        f"{neutral_percentage:.1f}% as neutral."
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    item,
    analysis,
    results,
    processing_time
):

    overall = get_overall_sentiment(
        analysis
    )

    print("\n")
    print("=" * 65)
    print("                 INSIGHTSPHERE")
    print("          UNIVERSAL REVIEW ANALYSIS")
    print("=" * 65)

    print(f"\nItem: {item}")

    print(
        f"\nOverall Sentiment: {overall}"
    )

    print(
        f"Model Confidence: "
        f"{analysis['confidence']}%"
    )

    print(
        f"\nReview Evidence Analyzed: "
        f"{analysis['total']}"
    )

    print(
        f"\nPositive: {analysis['positive']}"
    )

    print(
        f"Neutral:  {analysis['neutral']}"
    )

    print(
        f"Negative: {analysis['negative']}"
    )

    print("\nSUMMARY")
    print("-" * 65)

    print(
        create_summary(
            item,
            analysis
        )
    )

    print("\nWEB SOURCES")
    print("-" * 65)

    displayed_urls = set()

    for result in results:

        url = result.get(
            "url",
            ""
        )

        if not url:
            continue

        if url in displayed_urls:
            continue

        displayed_urls.add(url)

        print(
            f"- {result.get('title', 'Source')}"
        )

        print(
            f"  {url}"
        )

        if len(displayed_urls) >= 10:
            break

    print(
        f"\nProcessing time: "
        f"{processing_time:.2f} seconds"
    )

    print("=" * 65)


# ============================================================
# MAIN UNIVERSAL SEARCH
# ============================================================

def analyze_item(item):

    item = clean_text(item)

    if not item:

        print(
            "\nPlease enter an item name."
        )

        return

    start_time = time.time()

    results = search_web(item)

    print(
        f"\nWeb pages collected: "
        f"{len(results)}"
    )

    review_sentences = (
        extract_review_sentences(
            results
        )
    )

    print(
        f"Review text segments found: "
        f"{len(review_sentences)}"
    )

    # Limit analysis to prevent extremely
    # large web pages from overwhelming the model.

    review_sentences = review_sentences[:150]

    analysis = analyze_sentiment(
        review_sentences
    )

    processing_time = (
        time.time() - start_time
    )

    display_results(
        item=item,
        analysis=analysis,
        results=results,
        processing_time=processing_time
    )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 65)
    print("        INSIGHTSPHERE UNIVERSAL REVIEW ENGINE")
    print("=" * 65)

    print(
        "\nEnter anything you want to analyze:"
    )

    print(
        "Examples:"
    )

    print(
        "  Interstellar"
    )

    print(
        "  Atomic Habits"
    )

    print(
        "  Ooty"
    )

    print(
        "  Biryani"
    )

    print(
        "  Cetaphil Moisturizer"
    )

    print(
        "  iPhone"
    )

    print(
        "  Taj Coromandel"
    )

    print(
        "\nType 'exit' to stop."
    )

    while True:

        item = input(
            "\nEnter item/topic: "
        ).strip()

        if item.lower() == "exit":
            print(
                "\nInsightSphere closed."
            )
            break

        analyze_item(item)