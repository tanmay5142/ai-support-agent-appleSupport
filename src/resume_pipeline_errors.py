"""
Resumes a pipeline evaluation run that got cut short by a rate limit.
Only re-processes rows marked "ERROR" - doesn't waste API calls redoing
rows that already worked.

Usage:
    python -u src/resume_pipeline_errors.py
"""

import time
import pandas as pd

from retrieval import Retriever
from intents import classify_intent
from generate import generate_reply
from policy import decide_escalation

PREDICTIONS_PATH = "data/processed/pipeline_predictions.csv"

MAX_RETRIES = 3
WAIT_BETWEEN_ROWS = 1.0     # seconds, be gentler on the rate limit than before
WAIT_AFTER_RATE_LIMIT = 60  # seconds, pause if we hit a rate limit again


def process_one(message, retriever):
    intent_result = classify_intent(message)
    intent = intent_result["intent"]

    reply_result = generate_reply(message, retriever=retriever)
    draft_reply = reply_result["draft_reply"]

    decision_result = decide_escalation(message, intent, draft_reply)
    decision = decision_result["decision"]

    return intent, decision, draft_reply


def main():
    df = pd.read_csv(PREDICTIONS_PATH)
    error_rows = df[df["predicted_intent"] == "ERROR"]
    print(f"Found {len(error_rows)} rows to retry.")

    if len(error_rows) == 0:
        print("Nothing to retry - all rows already have predictions.")
        return

    retriever = Retriever()
    fixed = 0
    still_failing = 0

    for i, row in error_rows.iterrows():
        message = row["text_customer"]

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                intent, decision, draft_reply = process_one(message, retriever)
                df.loc[i, "predicted_intent"] = intent
                df.loc[i, "predicted_decision"] = decision
                df.loc[i, "draft_reply"] = draft_reply
                fixed += 1
                break
            except Exception as e:
                is_rate_limit = "rate_limit" in str(e).lower() or "429" in str(e)
                if is_rate_limit and attempt < MAX_RETRIES:
                    print(f"Row {i}: rate limited, waiting {WAIT_AFTER_RATE_LIMIT}s before retry {attempt+1}/{MAX_RETRIES}...")
                    time.sleep(WAIT_AFTER_RATE_LIMIT)
                else:
                    print(f"Row {i}: still failing after {attempt} attempt(s): {e}")
                    still_failing += 1

        time.sleep(WAIT_BETWEEN_ROWS)

    df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"\nFixed {fixed} rows. {still_failing} rows still failing (probably still rate-limited).")
    print(f"Saved back to {PREDICTIONS_PATH}")

    if still_failing > 0:
        print("If rows are still failing, your daily quota probably hasn't reset yet.")
        print("Try again in a few hours, or tomorrow.")


if __name__ == "__main__":
    main()
