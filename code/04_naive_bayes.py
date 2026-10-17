import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


print("=" * 60)
print("INSIGHTSPHERE - NAIVE BAYES")
print("=" * 60)


# --------------------------------------------------
# LOAD MOVIE DATA
# --------------------------------------------------

data = pd.read_csv("outputs/movies_cleaned.csv")

X = data["cleaned_review"].fillna("")
y = data["sentiment"]


# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.95
)

X_tfidf = vectorizer.fit_transform(X)


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# --------------------------------------------------
# NAIVE BAYES MODEL
# --------------------------------------------------

model = MultinomialNB()

model.fit(X_train, y_train)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nNaive Bayes Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------------------------
# SAVE MODEL AND VECTORIZER
# --------------------------------------------------

joblib.dump(vectorizer, "models/naive_bayes_tfidf_vectorizer.pkl")
joblib.dump(model, "models/naive_bayes_model.pkl")

print("\nModel files saved successfully:")
print("✓ models/naive_bayes_tfidf_vectorizer.pkl")
print("✓ models/naive_bayes_model.pkl")


print("=" * 60)
print("NAIVE BAYES COMPLETED")
print("=" * 60)