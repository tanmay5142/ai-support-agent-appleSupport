"""
Trivial baseline.

The idea: what if the "AI system" was actually dumb and just guessed the
most common answer every single time, no matter what the message said?
This gives you a floor to beat. If your real system barely beats this,
that's a big red flag worth knowing about early.

What this does:
1. Looks at the golden set and finds the single most common intent, and
   the single most common auto/escalate decision.
2. Pretends to "predict" that same answer for every row.
3. Compares those guesses to the real labels and prints accuracy.

Usage:
    python -u src/baseline_trivial.py
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

GOLDEN_SET_PATH = "data/processed/golden_set.csv"


def main():
    df = pd.read_csv(GOLDEN_SET_PATH)

    print(f"Loaded {len(df)} labeled examples from {GOLDEN_SET_PATH}\n")

    # ---- Intent baseline ----
    most_common_intent = df["final_intent"].value_counts().idxmax()
    print(f"Most common intent overall: '{most_common_intent}'")

    intent_predictions = [most_common_intent] * len(df)
    intent_accuracy = accuracy_score(df["final_intent"], intent_predictions)
    intent_f1 = f1_score(df["final_intent"], intent_predictions, average="macro", zero_division=0)

    print(f"\n--- Intent classification: trivial baseline ---")
    print(f"Always guessing '{most_common_intent}'")
    print(f"Accuracy: {intent_accuracy:.3f}")
    print(f"Macro F1: {intent_f1:.3f}  (this will be low on purpose — it's a weak baseline)")

    # ---- Auto/escalate baseline ----
    most_common_decision = df["auto_or_escalate"].value_counts().idxmax()
    print(f"\nMost common decision overall: '{most_common_decision}'")

    decision_predictions = [most_common_decision] * len(df)
    decision_accuracy = accuracy_score(df["auto_or_escalate"], decision_predictions)
    decision_f1 = f1_score(df["auto_or_escalate"], decision_predictions, average="macro", zero_division=0)

    print(f"\n--- Auto/escalate decision: trivial baseline ---")
    print(f"Always guessing '{most_common_decision}'")
    print(f"Accuracy: {decision_accuracy:.3f}")
    print(f"Macro F1: {decision_f1:.3f}")

    # Full breakdown, useful to paste into your report later
    print("\n--- Full intent classification report (trivial baseline) ---")
    print(classification_report(df["final_intent"], intent_predictions, zero_division=0))


if __name__ == "__main__":
    main()
