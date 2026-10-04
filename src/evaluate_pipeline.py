"""
Runs the real AI pipeline (classify + retrieve + generate + decide) on
every message in the golden set, and scores it the same way the trivial
and simple baselines were scored - so the three can be compared fairly.

This calls the LLM once per message for classification and once for the
decision (reply generation also costs a call, but isn't scored here -
that's what the LLM-as-judge step will do separately). With 200 messages,
expect this to take a few minutes and use a small amount of API credit.

Usage:
    python -u src/evaluate_pipeline.py
"""

import time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

from retrieval import Retriever
from intents import classify_intent
from generate import generate_reply
from policy import decide_escalation

GOLDEN_SET_PATH = "data/processed/golden_set.csv"
RESULTS_OUT_PATH = "data/processed/pipeline_predictions.csv"


def main():
    df = pd.read_csv(GOLDEN_SET_PATH)
    print(f"Loaded {len(df)} labeled examples from {GOLDEN_SET_PATH}")

    retriever = Retriever()  # build the similarity index once, reuse for every message

    predicted_intents = []
    predicted_decisions = []
    draft_replies = []
    errors = 0

    for i, row in df.iterrows():
        message = row["text_customer"]
        try:
            intent_result = classify_intent(message)
            intent = intent_result["intent"]

            reply_result = generate_reply(message, retriever=retriever)
            draft_reply = reply_result["draft_reply"]

            decision_result = decide_escalation(message, intent, draft_reply)
            decision = decision_result["decision"]

        except Exception as e:
            # Don't let one bad response (e.g. malformed JSON) kill the whole run.
            print(f"Row {i} failed: {e}")
            intent = "ERROR"
            decision = "ERROR"
            draft_reply = ""
            errors += 1

        predicted_intents.append(intent)
        predicted_decisions.append(decision)
        draft_replies.append(draft_reply)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(df)}...")

        time.sleep(0.2)  # small pause to avoid hitting rate limits

    df["predicted_intent"] = predicted_intents
    df["predicted_decision"] = predicted_decisions
    df["draft_reply"] = draft_replies
    df.to_csv(RESULTS_OUT_PATH, index=False)
    print(f"\nSaved all predictions + draft replies to {RESULTS_OUT_PATH}")

    if errors:
        print(f"\n{errors} rows had errors (marked as 'ERROR') - worth a look before trusting scores below.")

    # Score against the same golden labels, same way the baselines were scored
    valid = df[df["predicted_intent"] != "ERROR"]

    print("\n=== Real pipeline: Intent classification ===")
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
