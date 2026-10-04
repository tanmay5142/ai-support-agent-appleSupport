"""
Scores the already-saved predictions file against the golden labels.
Doesn't call the LLM at all - just reads the CSV that evaluate_pipeline.py
and resume_pipeline_errors.py already filled in, so you can re-check your
numbers anytime without spending API credit again.

Usage:
    python -u src/score_predictions.py
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

PREDICTIONS_PATH = "data/processed/pipeline_predictions.csv"


def main():
    df = pd.read_csv(PREDICTIONS_PATH)

    valid = df[(df["predicted_intent"] != "ERROR") & (df["predicted_intent"].notna())]
    print(f"Scoring {len(valid)} of {len(df)} total rows (excluding any still-failed ones)\n")

    print("=== Real pipeline: Intent classification ===")
    acc = accuracy_score(valid["final_intent"], valid["predicted_intent"])
    f1 = f1_score(valid["final_intent"], valid["predicted_intent"], average="macro", zero_division=0)
    print(f"Accuracy: {acc:.3f}")
    print(f"Macro F1: {f1:.3f}")
    print(classification_report(valid["final_intent"], valid["predicted_intent"], zero_division=0))

    print("\n=== Real pipeline: Auto vs escalate decision ===")
    acc = accuracy_score(valid["auto_or_escalate"], valid["predicted_decision"])
    f1 = f1_score(valid["auto_or_escalate"], valid["predicted_decision"], average="macro", zero_division=0)
    print(f"Accuracy: {acc:.3f}")
    print(f"Macro F1: {f1:.3f}")
    print(classification_report(valid["auto_or_escalate"], valid["predicted_decision"], zero_division=0))


if __name__ == "__main__":
    main()