"""
Runs the LLM judge on every drafted reply from the pipeline run, and
scores it against the real historical reply.

Saves after every single row (not just at the end) - so if this gets
interrupted (rate limit, closed terminal, whatever), just run it again
and it will pick up where it left off instead of starting over or
re-spending API credit on rows already scored.

Usage:
    python -u src/run_judge.py
"""

import os
import time
import pandas as pd

from judge import judge_reply

PREDICTIONS_PATH = "data/processed/pipeline_predictions.csv"
SCORES_PATH = "data/processed/judge_scores.csv"

SCORE_COLUMNS = ["relevance", "tone", "approach_match", "overall", "justification"]


def main():
    predictions = pd.read_csv(PREDICTIONS_PATH)

    # If a scores file already exists from a previous (possibly interrupted)
    # run, keep using it. Otherwise start fresh from the predictions file.
    if os.path.exists(SCORES_PATH):
        df = pd.read_csv(SCORES_PATH)
        print(f"Found existing {SCORES_PATH}, resuming from where it left off.")
    else:
        df = predictions.copy()
        for col in SCORE_COLUMNS:
            df[col] = pd.NA
        print("Starting a fresh judging run.")

    already_done = df["overall"].notna().sum()
    print(f"{already_done} of {len(df)} rows already scored.\n")

    for i, row in df.iterrows():
        if pd.notna(row["overall"]):
            continue  # already scored, skip

        message = row["text_customer"]
        real_reply = row["text_reply"]
        draft_reply = row["draft_reply"]

        if pd.isna(draft_reply) or draft_reply == "" or row.get("predicted_intent") == "ERROR":
            # nothing to judge if there was no draft reply
            continue

        try:
            result = judge_reply(message, draft_reply, real_reply)
            for col in SCORE_COLUMNS:
                df.loc[i, col] = result.get(col)
        except Exception as e:
            print(f"Row {i} failed: {e}")
            # leave it blank, will be retried next run

        # Save after EVERY row - this is the important bit for resumability
        df.to_csv(SCORES_PATH, index=False)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(df)}...")

        time.sleep(1.0)

    scored = df["overall"].notna().sum()
    print(f"\nDone. {scored} of {len(df)} rows have judge scores.")
    print(f"Saved to {SCORES_PATH}")

    if scored > 0:
        print("\nAverage scores across all judged replies:")
        for col in ["relevance", "tone", "approach_match", "overall"]:
            avg = pd.to_numeric(df[col], errors="coerce").mean()
            print(f"  {col}: {avg:.2f} / 5")


if __name__ == "__main__":
    main()