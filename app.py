from flask import Flask, render_template, request, jsonify
import os
import re
import time
import joblib
from dotenv import load_dotenv
from tavily import TavilyClient

# ============================================================
# INSIGHTSPHERE
# Universal Web Review Intelligence
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

ENV_PATH = os.path.join(
    BASE_DIR,
    ".env"
)

# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv(ENV_PATH)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("InsightSphere ML model loaded successfully.")
    print("Six-domain TF-IDF vectorizer loaded successfully.")

except Exception as error:

    print("MODEL ERROR:", error)

    model = None
    vectorizer = None


# ============================================================
# TAVILY
# ============================================================

if TAVILY_API_KEY:

    try:
        tavily = TavilyClient(
            api_key=TAVILY_API_KEY
        )

        print(
            "Tavily web search connected successfully."
        )

    except Exception as error:

        print(
            "TAVILY ERROR:",
            error
        )

        tavily = None

else:

    print(
        "WARNING: TAVILY_API_KEY not found."
    )

    tavily = None


# ============================================================
# SOURCE QUALITY
# ============================================================

HIGH_VALUE_DOMAINS = [

    "rottentomatoes.com",
    "imdb.com",
    "metacritic.com",
    "rogerebert.com",

    "tripadvisor.com",
    "booking.com",
    "agoda.com",
    "makemytrip.com",

    "amazon.com",
    "amazon.in",
    "flipkart.com",

    "goodreads.com",

    "yelp.com"
]


LOW_VALUE_DOMAINS = [

    "wikipedia.org",
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "pinterest.com",
    "reddit.com"
]


def source_quality(url):

    url_lower = url.lower()

    for domain in HIGH_VALUE_DOMAINS:

        if domain in url_lower:
            return 3

    for domain in LOW_VALUE_DOMAINS:

        if domain in url_lower:
            return 0

    return 2


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SEARCH QUERIES
# ============================================================

def build_queries(item):

    return [

        f'"{item}" reviews',

        f'"{item}" customer reviews',

        f'"{item}" user reviews',

        f'"{item}" ratings',

        f'"{item}" pros cons',

        f'"{item}" review experience',

        f'"{item}" what people say'

    ]


# ============================================================
# WEB SEARCH
# ============================================================

def search_web(item):

    if tavily is None:
        return []

    results = []
    seen_urls = set()

    for query in build_queries(item):

        try:

            response = tavily.search(

                query=query,

                search_depth="advanced",

                max_results=5,

                include_answer=False,

                include_raw_content=True
            )

            for result in response.get(
                "results",
                []
            ):

                url = clean_text(
                    result.get(
                        "url",
                        ""
                    )
                )

                if not url:
                    continue

                if url in seen_urls:
                    continue

                seen_urls.add(url)

                content = (

                    result.get(
                        "raw_content"
                    )

                    or

                    result.get(
                        "content"
                    )

                    or ""

                )

                results.append({

                    "title": clean_text(
                        result.get(
                            "title",
                            "Web Source"
                        )
                    ),

                    "url": url,

                    "content": clean_text(
                        content
                    ),

                    "quality": source_quality(
                        url
                    )

                })

        except Exception as error:

            print(
                "Search warning:",
                error
            )

    # Highest quality sources first

    results.sort(
        key=lambda x: x["quality"],
        reverse=True
    )

    return results


# ============================================================
# REVIEW SENTENCE DETECTION
# ============================================================

REVIEW_TERMS = [

    "review",
    "rating",
    "rated",
    "stars",
    "experience",
    "recommend",
    "recommended",
    "worth",
    "quality",
    "service",
    "liked",
    "love",
    "loved",
    "dislike",
    "disliked",
    "good",
    "great",
    "bad",
    "poor",
    "excellent",
    "terrible",
    "amazing",
    "awful",
    "fantastic",
    "boring",
    "beautiful",
    "comfortable",
    "expensive",
    "cheap",
    "value",
    "taste",
    "delicious",
    "disappointing",
    "satisfied",
    "satisfaction"
]


def looks_like_review(text):

    text_lower = text.lower()

    matches = 0

    for term in REVIEW_TERMS:

        if term in text_lower:
            matches += 1

    return matches >= 1


# ============================================================
# SENTENCE EXTRACTION
# ============================================================

def extract_review_sentences(results):

    evidence = []

    seen = set()

    # Prefer quality sources

    for result in results:

        content = result.get(
            "content",
            ""
        )

        if not content:
            continue

        parts = re.split(
            r"(?<=[.!?])\s+",
            content
        )

        for sentence in parts:

            sentence = clean_text(
                sentence
            )

            if len(sentence) < 30:
                continue

            if len(sentence) > 500:
                continue

            if not looks_like_review(
                sentence
            ):
                continue

            key = sentence.lower()

            if key in seen:
                continue

            seen.add(key)

            evidence.append({

                "text": sentence,

                "source": result[
                    "title"
                ],

                "url": result[
                    "url"
                ],

                "quality": result[
                    "quality"
                ]

            })

    # Highest quality evidence first

    evidence.sort(
        key=lambda x: x["quality"],
        reverse=True
    )

    return evidence


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def analyze_reviews(
    review_sentences
):

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

    transformed = vectorizer.transform(
        texts
    )

    predictions = model.predict(
        transformed
    )

    probabilities = model.predict_proba(
        transformed
    )

    positive = 0
    neutral = 0
    negative = 0

    confidence_values = []

    details = []

    for (

        item,
        prediction,
        probability

    ) in zip(

        review_sentences,
        predictions,
        probabilities

    ):

        sentiment = str(
            prediction
        ).lower()

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

            "text": item[
                "text"
            ],

            "sentiment": sentiment,

            "confidence": round(
                confidence,
                2
            ),

            "source": item[
                "source"
            ],

            "url": item[
                "url"
            ]

        })

    total = len(
        predictions
    )

    average_confidence = (

        sum(confidence_values)

        /

        len(confidence_values)

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

def get_overall_sentiment(
    analysis
):

    if analysis[
        "total"
    ] == 0:

        return "Insufficient Data"

    values = {

        "Positive":
            analysis["positive"],

        "Neutral":
            analysis["neutral"],

        "Negative":
            analysis["negative"]

    }

    return max(
        values,
        key=values.get
    )


# ============================================================
# ASPECT DETECTION
# ============================================================

ASPECTS = {

    "Story & Content": [

        "story",
        "plot",
        "content",
        "script",
        "writing",
        "narrative"

    ],

    "Acting & Performance": [

        "acting",
        "actor",
        "actress",
        "performance",
        "cast",
        "character"

    ],

    "Quality": [

        "quality",
        "build",
        "material",
        "durability",
        "design"

    ],

    "Price & Value": [

        "price",
        "expensive",
        "cheap",
        "value",
        "worth",
        "cost"

    ],

    "Service": [

        "service",
        "staff",
        "support",
        "delivery",
        "customer service"

    ],

    "Location": [

        "location",
        "place",
        "area",
        "view",
        "destination"

    ],

    "Comfort": [

        "comfortable",
        "comfort",
        "room",
        "bed",
        "clean"

    ],

    "Taste & Food": [

        "taste",
        "food",
        "flavour",
        "flavor",
        "delicious",
        "spicy",
        "meal"

    ]

}


def detect_aspects(details):

    aspect_data = {}

    for aspect, keywords in ASPECTS.items():

        positive = 0
        negative = 0
        neutral = 0

        for detail in details:

            text = detail[
                "text"
            ].lower()

            if any(
                keyword in text
                for keyword in keywords
            ):

                sentiment = detail[
                    "sentiment"
                ]

                if sentiment == "positive":

                    positive += 1

                elif sentiment == "negative":

                    negative += 1

                else:

                    neutral += 1

        total = (
            positive
            + negative
            + neutral
        )

        if total > 0:

            if positive >= negative:

                aspect_sentiment = (
                    "Positive"
                )

            else:

                aspect_sentiment = (
                    "Negative"
                )

            aspect_data[aspect] = {

                "sentiment":
                    aspect_sentiment,

                "positive":
                    positive,

                "negative":
                    negative,

                "neutral":
                    neutral,

                "mentions":
                    total

            }

    return aspect_data


# ============================================================
# KEY INSIGHTS
# ============================================================

def extract_key_insights(
    details
):

    positive_points = []

    negative_points = []

    for detail in details:

        text = detail[
            "text"
        ]

        sentiment = detail[
            "sentiment"
        ]

        if sentiment == "positive":

            if len(
                positive_points
            ) < 5:

                positive_points.append(
                    text
                )

        elif sentiment == "negative":

            if len(
                negative_points
            ) < 5:

                negative_points.append(
                    text
                )

    return {

        "positive": positive_points,

        "negative": negative_points

    }


# ============================================================
# SUMMARY
# ============================================================

def create_summary(
    item,
    analysis
):

    total = analysis[
        "total"
    ]

    if total == 0:

        return (

            f"InsightSphere could not "
            f"find enough review-related "
            f"information about {item}."

        )

    positive_pct = (

        analysis["positive"]
        / total

    ) * 100

    neutral_pct = (

        analysis["neutral"]
        / total

    ) * 100

    negative_pct = (

        analysis["negative"]
        / total

    ) * 100

    overall = get_overall_sentiment(
        analysis
    )

    return (

        f"{item} has an overall "
        f"{overall.lower()} sentiment "
        f"based on {total} review-related "
        f"text segments collected from "
        f"web sources. "

        f"{positive_pct:.1f}% were classified "
        f"as positive, "

        f"{neutral_pct:.1f}% as neutral, "
        f"and "

        f"{negative_pct:.1f}% as negative."

    )


# ============================================================
# UNIVERSAL ANALYSIS
# ============================================================

def analyze_item(item):

    item = clean_text(item)

    if not item:

        return {

            "success": False,

            "error":
                "Please enter an item or topic."

        }

    if model is None:

        return {

            "success": False,

            "error":
                "ML model could not be loaded."

        }

    if tavily is None:

        return {

            "success": False,

            "error":
                "Web search is unavailable. "
                "Please check your Tavily API key."

        }

    start_time = time.time()

    print()
    print("=" * 65)

    print(
        f"Analyzing: {item}"
    )

    print("=" * 65)

    # --------------------------------------------------------
    # WEB SEARCH
    # --------------------------------------------------------

    print(
        "Searching high-quality web sources..."
    )

    web_results = search_web(
        item
    )

    print(
        f"Web pages collected: "
        f"{len(web_results)}"
    )

    # --------------------------------------------------------
    # REVIEW EVIDENCE
    # --------------------------------------------------------

    review_sentences = (
        extract_review_sentences(
            web_results
        )
    )

    print(
        f"Review evidence found: "
        f"{len(review_sentences)}"
    )

    # Keep strongest evidence

    review_sentences = (
        review_sentences[:150]
    )

    # --------------------------------------------------------
    # ML
    # --------------------------------------------------------

    analysis = analyze_reviews(
        review_sentences
    )

    # --------------------------------------------------------
    # ASPECTS
    # --------------------------------------------------------

    aspects = detect_aspects(
        analysis["details"]
    )

    # --------------------------------------------------------
    # KEY INSIGHTS
    # --------------------------------------------------------

    insights = extract_key_insights(
        analysis["details"]
    )

    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    overall = get_overall_sentiment(
        analysis
    )

    processing_time = (
        time.time() - start_time
    )

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    sources = []

    seen_sources = set()

    # Prioritize quality sources

    for result in web_results:

        url = result.get(
            "url",
            ""
        )

        if not url:
            continue

        if url in seen_sources:
            continue

        seen_sources.add(url)

        sources.append({

            "title":
                result.get(
                    "title",
                    "Web Source"
                ),

            "url": url,

            "quality":
                result.get(
                    "quality",
                    2
                )

        })

        if len(sources) >= 10:
            break

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    result = {

        "success": True,

        "item": item,

        "overall_sentiment":
            overall,

        "confidence":
            analysis["confidence"],

        "total_reviews_analyzed":
            analysis["total"],

        "positive":
            analysis["positive"],

        "neutral":
            analysis["neutral"],

        "negative":
            analysis["negative"],

        "summary":
            create_summary(
                item,
                analysis
            ),

        "aspects":
            aspects,

        "key_insights":
            insights,

        "sources":
            sources,

        "processing_time":
            round(
                processing_time,
                2
            ),

        "details":
            analysis[
                "details"
            ][:20]

    }

    print(
        f"Overall sentiment: {overall}"
    )

    print(
        f"Confidence: "
        f"{analysis['confidence']}%"
    )

    print(
        f"Processing time: "
        f"{processing_time:.2f} seconds"
    )

    print("=" * 65)

    return result


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# ANALYZE API
# ============================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    try:

        data = request.get_json(
            silent=True
        )

        if not data:

            return jsonify({

                "success": False,

                "error":
                    "No data received."

            }), 400

        item = data.get(
            "item",
            ""
        )

        result = analyze_item(
            item
        )

        return jsonify(
            result
        )

    except Exception as error:

        print(
            "Analysis error:",
            error
        )

        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 65)

    print(
        "              INSIGHTSPHERE"
    )

    print(
        "       UNIVERSAL REVIEW INTELLIGENCE"
    )

    print("=" * 65)

    print()
    print(
        "Server starting..."
    )

    print(
        "Open: http://127.0.0.1:5000"
    )

    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )