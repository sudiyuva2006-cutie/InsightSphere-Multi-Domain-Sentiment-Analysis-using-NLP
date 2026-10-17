import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


print("=" * 60)
print("INSIGHTSPHERE - RANDOM FOREST")
print("=" * 60)


# Load cleaned movie dataset
data = pd.read_csv("outputs/movies_cleaned.csv")

X = data["cleaned_review"].fillna("")
y = data["sentiment"]


# TF-IDF Feature Extraction
vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.95
)

X_tfidf = vectorizer.fit_transform(X)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)


# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nRandom Forest Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(model, "models/random_forest_model.pkl")

print("\nModel saved successfully:")
print("✓ models/random_forest_model.pkl")


print("=" * 60)
print("RANDOM FOREST COMPLETED")
print("=" * 60)