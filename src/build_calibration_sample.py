"""
Picks a sample of already-judged replies for YOU to score by hand, so we
can check whether the LLM judge's scores actually mean anything or if
it's just being lenient/generous across the board.

Usage:
    python -u src/build_calibration_sample.py
"""

import pandas as pd

SCORES_PATH = "data/processed/judge_scores.csv"
OUT_PATH = "data/processed/calibration_sample_to_score.csv"

SAMPLE_SIZE = 30
RANDOM_STATE = 11


def main():
    df = pd.read_csv(SCORES_PATH)
    df = df[df["overall"].notna()]  # only rows that actually got judged

    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)

    # Keep the judge's scores HIDDEN in a separate column name so you score
    # blind, without being influenced by what the AI judge already said.
    sample = sample.rename(columns={
        "relevance": "judge_relevance",
        "tone": "judge_tone",
        "approach_match": "judge_approach_match",
        "overall": "judge_overall",
        "justification": "judge_justification",
    })

    # Empty columns for YOUR scores - same 1-5 scale, same meaning as the judge's
    sample["your_relevance"] = ""
    sample["your_tone"] = ""
    sample["your_approach_match"] = ""
    sample["your_overall"] = ""

    cols = [
        "text_customer", "text_reply", "draft_reply",
        "your_relevance", "your_tone", "your_approach_match", "your_overall",
        "judge_relevance", "judge_tone", "judge_approach_match", "judge_overall", "judge_justification",
    ]
    sample[cols].to_csv(OUT_PATH, index=False)

    print(f"Saved {len(sample)} rows to {OUT_PATH}")
    print("\nIMPORTANT: score the 'your_*' columns WITHOUT looking at the 'judge_*' columns.")
    print("Open the file, hide/ignore the judge_ columns, and fill in your own 1-5 scores")
    print("for relevance, tone, approach_match, and overall - just like the judge did.")
    print("Once you're done, we'll compare your scores to the judge's.")


if __name__ == "__main__":
    main()