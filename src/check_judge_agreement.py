"""
Checks how well the LLM judge's scores actually agree with your own,
row by row - not just whether the averages happen to look similar.

Two scores can have the same average while disagreeing on almost every
individual row (e.g. judge is too high on some, too low on others, and
it cancels out). This checks the real thing: for each row, how close
was the judge to what you actually thought?

Usage:
    python -u src/check_judge_agreement.py
"""

import pandas as pd
from scipy.stats import pearsonr

SCORED_PATH = "data/processed/calibration_sample_scored.csv"

DIMENSIONS = ["relevance", "tone", "approach_match", "overall"]


def main():
    df = pd.read_csv(SCORED_PATH)

    print(f"Comparing {len(df)} rows of your scores vs. the judge's scores.\n")

    for dim in DIMENSIONS:
        your_col = f"your_{dim}"
        judge_col = f"judge_{dim}"

        your_scores = df[your_col].astype(float)
        judge_scores = df[judge_col].astype(float)

        diffs = (your_scores - judge_scores).abs()

        exact_match_pct = (diffs == 0).mean() * 100
        within_1_pct = (diffs <= 1).mean() * 100
        mean_abs_diff = diffs.mean()

        correlation, _ = pearsonr(your_scores, judge_scores)

        print(f"--- {dim} ---")
        print(f"  Exact match:        {exact_match_pct:.0f}% of rows")
        print(f"  Within 1 point:     {within_1_pct:.0f}% of rows")
        print(f"  Average difference: {mean_abs_diff:.2f} points")
        print(f"  Correlation:        {correlation:.2f}  (1.0 = perfect agreement, 0 = no relationship)")
        print()

    # Flag the rows where you and the judge disagreed the most - these are
    # worth reading by hand, they're your best real examples for a
    # "where the judge breaks down" write-up.
    df["overall_diff"] = (df["your_overall"].astype(float) - df["judge_overall"].astype(float)).abs()
    biggest_disagreements = df.sort_values("overall_diff", ascending=False).head(5)

    print("=== Biggest disagreements (worth looking at by hand) ===")
    for _, row in biggest_disagreements.iterrows():
        print(f"\nCustomer message: {row['text_customer']}")
        print(f"Draft reply: {row['draft_reply']}")
        print(f"You said overall={row['your_overall']}, judge said overall={row['judge_overall']}")
        print(f"Judge's reasoning: {row['judge_justification']}")


if __name__ == "__main__":
    main()