# ============================================================
# STEP 12 - TRAIN MACHINE LEARNING MODELS
# INSIGHTSPHERE: MULTI-DOMAIN SENTIMENT ANALYSIS
# ============================================================

import os
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "final_sentiment_data_six_domains.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

RANDOM_STATE = 42

# Maximum number of reviews taken from each domain.
# This prevents Hotels from dominating the training data.
MAX_PER_DOMAIN = 50000

TEST_SIZE = 0.20

# TF-IDF settings
MAX_FEATURES = 30000


# ============================================================
# 3. START
# ============================================================

print("=" * 70)
print("INSIGHTSPHERE - STEP 12")
print("SIX-DOMAIN MACHINE LEARNING MODEL TRAINING")
print("=" * 70)

start_time = time.time()


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\n[1/9] Loading six-domain dataset...")

if not os.path.exists(DATA_PATH):
    print("\nERROR: Dataset not found!")
    print(DATA_PATH)
    raise FileNotFoundError(DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Total rows:", len(df))
print("Columns:", list(df.columns))


# ============================================================
# 5. BASIC CLEANING
# ============================================================

print("\n[2/9] Cleaning dataset...")

required_columns = ["domain", "review", "sentiment"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' is missing from the dataset."
        )

df = df[required_columns].copy()

# Remove missing values
df = df.dropna(subset=["review", "sentiment", "domain"])

# Convert review text to string
df["review"] = df["review"].astype(str)

# Remove empty reviews
df = df[df["review"].str.strip() != ""]

# Normalize sentiment
df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()

# Keep only valid sentiment classes
valid_sentiments = ["positive", "negative", "neutral"]

df = df[df["sentiment"].isin(valid_sentiments)]

print("Rows after cleaning:", len(df))


# ============================================================
# 6. BALANCE THE DOMAINS
# ============================================================

print("\n[3/9] Preparing balanced multi-domain training data...")

print("\nOriginal rows by domain:")
print(df["domain"].value_counts())

balanced_parts = []

for domain in df["domain"].unique():

    domain_data = df[df["domain"] == domain].copy()

    # If domain has more than the limit,
    # randomly select MAX_PER_DOMAIN rows.
    if len(domain_data) > MAX_PER_DOMAIN:
        domain_data = domain_data.sample(
            n=MAX_PER_DOMAIN,
            random_state=RANDOM_STATE
        )

    balanced_parts.append(domain_data)

df_balanced = pd.concat(
    balanced_parts,
    ignore_index=True
)

# Shuffle
df_balanced = df_balanced.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

print("\nTraining dataset prepared.")

print("\nRows by domain:")
print(df_balanced["domain"].value_counts())

print("\nRows by sentiment:")
print(df_balanced["sentiment"].value_counts())

print("\nTotal training rows:", len(df_balanced))


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

print("\n[4/9] Splitting data into training and testing sets...")

X = df_balanced["review"]
y = df_balanced["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 8. TF-IDF FEATURE EXTRACTION
# ============================================================

print("\n[5/9] Creating TF-IDF features...")
print("Please wait...")

tfidf_vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    dtype=np.float32
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

X_test_tfidf = tfidf_vectorizer.transform(X_test)

print("TF-IDF transformation completed.")

print("Training matrix shape:", X_train_tfidf.shape)
print("Testing matrix shape:", X_test_tfidf.shape)

print(
    "Number of TF-IDF features:",
    len(tfidf_vectorizer.get_feature_names_out())
)


# ============================================================
# 9. SAVE TF-IDF VECTORIZER
# ============================================================

tfidf_path = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer_six_domains.pkl"
)

joblib.dump(
    tfidf_vectorizer,
    tfidf_path
)

print("\nTF-IDF vectorizer saved:")
print(tfidf_path)


# ============================================================
# 10. NAIVE BAYES
# ============================================================

print("\n[6/9] Training Naive Bayes model...")
print("Please wait...")

nb_model = MultinomialNB()

nb_model.fit(
    X_train_tfidf,
    y_train
)

nb_predictions = nb_model.predict(X_test_tfidf)

nb_accuracy = accuracy_score(
    y_test,
    nb_predictions
)

nb_precision = precision_score(
    y_test,
    nb_predictions,
    average="weighted",
    zero_division=0
)

nb_recall = recall_score(
    y_test,
    nb_predictions,
    average="weighted",
    zero_division=0
)

nb_f1 = f1_score(
    y_test,
    nb_predictions,
    average="weighted",
    zero_division=0
)

print("\nNaive Bayes Results:")
print("Accuracy :", round(nb_accuracy * 100, 2), "%")
print("Precision:", round(nb_precision * 100, 2), "%")
print("Recall   :", round(nb_recall * 100, 2), "%")
print("F1 Score :", round(nb_f1 * 100, 2), "%")

print("\nNaive Bayes Classification Report:")
print(
    classification_report(
        y_test,
        nb_predictions,
        zero_division=0
    )
)

nb_model_path = os.path.join(
    MODEL_DIR,
    "naive_bayes_model_six_domains.pkl"
)

joblib.dump(
    nb_model,
    nb_model_path
)

print("Naive Bayes model saved:")
print(nb_model_path)


# ============================================================
# 11. LOGISTIC REGRESSION
# ============================================================

print("\n[7/9] Training Logistic Regression model...")
print("This may take some time. Please wait...")

lr_model = LogisticRegression(
    max_iter=1000,
    solver="saga",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

lr_model.fit(
    X_train_tfidf,
    y_train
)

lr_predictions = lr_model.predict(X_test_tfidf)

lr_accuracy = accuracy_score(
    y_test,
    lr_predictions
)

lr_precision = precision_score(
    y_test,
    lr_predictions,
    average="weighted",
    zero_division=0
)

lr_recall = recall_score(
    y_test,
    lr_predictions,
    average="weighted",
    zero_division=0
)

lr_f1 = f1_score(
    y_test,
    lr_predictions,
    average="weighted",
    zero_division=0
)

print("\nLogistic Regression Results:")
print("Accuracy :", round(lr_accuracy * 100, 2), "%")
print("Precision:", round(lr_precision * 100, 2), "%")
print("Recall   :", round(lr_recall * 100, 2), "%")
print("F1 Score :", round(lr_f1 * 100, 2), "%")

print("\nLogistic Regression Classification Report:")
print(
    classification_report(
        y_test,
        lr_predictions,
        zero_division=0
    )
)

lr_model_path = os.path.join(
    MODEL_DIR,
    "logistic_regression_model_six_domains.pkl"
)

joblib.dump(
    lr_model,
    lr_model_path
)

print("Logistic Regression model saved:")
print(lr_model_path)


# ============================================================
# 12. RANDOM FOREST
# ============================================================

print("\n[8/9] Preparing features for Random Forest...")
print("Applying Truncated SVD to reduce dimensionality...")

# Random Forest is not efficient with tens of thousands
# of sparse TF-IDF features.
#
# SVD converts the TF-IDF matrix into a smaller dense matrix.

SVD_COMPONENTS = 50

svd = TruncatedSVD(
    n_components=SVD_COMPONENTS,
    random_state=RANDOM_STATE
)

X_train_svd = svd.fit_transform(X_train_tfidf)

X_test_svd = svd.transform(X_test_tfidf)

print("SVD completed.")

print("Reduced training shape:", X_train_svd.shape)
print("Reduced testing shape:", X_test_svd.shape)


# ============================================================
# 13. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest model...")
print("Please wait...")

rf_model = RandomForestClassifier(
    n_estimators=75,
    max_depth=25,
    min_samples_split=4,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_model.fit(
    X_train_svd,
    y_train
)

rf_predictions = rf_model.predict(X_test_svd)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

rf_precision = precision_score(
    y_test,
    rf_predictions,
    average="weighted",
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_predictions,
    average="weighted",
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_predictions,
    average="weighted",
    zero_division=0
)

print("\nRandom Forest Results:")
print("Accuracy :", round(rf_accuracy * 100, 2), "%")
print("Precision:", round(rf_precision * 100, 2), "%")
print("Recall   :", round(rf_recall * 100, 2), "%")
print("F1 Score :", round(rf_f1 * 100, 2), "%")

print("\nRandom Forest Classification Report:")
print(
    classification_report(
        y_test,
        rf_predictions,
        zero_division=0
    )
)


# ============================================================
# 14. SAVE SVD
# ============================================================

svd_path = os.path.join(
    MODEL_DIR,
    "random_forest_svd_six_domains.pkl"
)

joblib.dump(
    svd,
    svd_path
)

print("\nSVD transformer saved:")
print(svd_path)


# ============================================================
# 15. SAVE RANDOM FOREST
# ============================================================

rf_model_path = os.path.join(
    MODEL_DIR,
    "random_forest_model_six_domains.pkl"
)

joblib.dump(
    rf_model,
    rf_model_path
)

print("Random Forest model saved:")
print(rf_model_path)


# ============================================================
# 16. MODEL COMPARISON
# ============================================================

print("\n[9/9] Creating model comparison report...")

results = pd.DataFrame({
    "Model": [
        "Naive Bayes",
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        nb_accuracy,
        lr_accuracy,
        rf_accuracy
    ],
    "Precision": [
        nb_precision,
        lr_precision,
        rf_precision
    ],
    "Recall": [
        nb_recall,
        lr_recall,
        rf_recall
    ],
    "F1 Score": [
        nb_f1,
        lr_f1,
        rf_f1
    ]
})

# Convert values to percentages
results["Accuracy"] = results["Accuracy"] * 100
results["Precision"] = results["Precision"] * 100
results["Recall"] = results["Recall"] * 100
results["F1 Score"] = results["F1 Score"] * 100

# Round values
results = results.round(2)

comparison_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison_six_domains.csv"
)

results.to_csv(
    comparison_path,
    index=False
)

print("\nMODEL COMPARISON")
print("=" * 70)
print(results.to_string(index=False))

print("\nModel comparison saved:")
print(comparison_path)


# ============================================================
# 17. FIND BEST MODEL
# ============================================================

best_model_row = results.loc[
    results["F1 Score"].idxmax()
]

print("\nBest model based on F1 Score:")
print(best_model_row["Model"])

print(
    "Best F1 Score:",
    best_model_row["F1 Score"],
    "%"
)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

elapsed_time = time.time() - start_time

print("\n")
print("=" * 70)
print("STEP 12 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nSix domains used:")
for domain in sorted(df_balanced["domain"].unique()):
    print(" -", domain)

print("\nModels trained:")
print(" - Naive Bayes")
print(" - Logistic Regression")
print(" - Random Forest")

print("\nSaved model files:")
print(" - tfidf_vectorizer_six_domains.pkl")
print(" - naive_bayes_model_six_domains.pkl")
print(" - logistic_regression_model_six_domains.pkl")
print(" - random_forest_svd_six_domains.pkl")
print(" - random_forest_model_six_domains.pkl")

print("\nSaved evaluation file:")
print(" - model_comparison_six_domains.csv")

print(
    "\nTotal execution time:",
    round(elapsed_time / 60, 2),
    "minutes"
)

print("\nNEXT STEP:")
print("Update the Flask application to use the new six-domain")
print("Logistic Regression model and TF-IDF vectorizer.")

print("\n" + "=" * 70)
print("INSIGHTSPHERE MODEL TRAINING FINISHED")
print("=" * 70)