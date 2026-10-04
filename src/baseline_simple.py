"""
Simple baseline (not trivial, not the full AI system).

The idea: instead of guessing the same answer every time (trivial baseline),
let's use a basic, classic method: turn each message into numbers based on
which words appear in it (TF-IDF), then train a simple model (Logistic
Regression) to learn which words point to which category.

This is still "dumb" compared to using an LLM, but it actually reads the
message, so it should clearly beat the trivial baseline. If it doesn't,
that's an important thing to know.

Note: the golden set only has 200 examples, and a few categories are rare
(as few as 2 examples). Splitting into train/test means some rare
categories may barely appear in one side or the other. This is a real
limitation worth mentioning in the report, not something to hide.

Usage:
    python -u src/baseline_simple.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

GOLDEN_SET_PATH = "data/processed/golden_set.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.25  # 25% held out for testing


def run_classifier(df, label_column, label_name):
    print(f"\n=== Simple baseline: {label_name} ===")

    X = df["text_customer"]
    y = df[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=2000,
        ngram_range=(1, 2),  # single words + two-word phrases
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")
    print("\nFull report:")
    print(classification_report(y_test, predictions, zero_division=0))

    return accuracy, macro_f1


def main():
    df = pd.read_csv(GOLDEN_SET_PATH)
    print(f"Loaded {len(df)} labeled examples from {GOLDEN_SET_PATH}")

    run_classifier(df, "final_intent", "Intent classification")
    run_classifier(df, "auto_or_escalate", "Auto vs escalate decision")


if __name__ == "__main__":
    main()
