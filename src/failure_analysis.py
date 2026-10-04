"""
Digs through the real pipeline predictions to find concrete examples of
where the system got things wrong - organized by failure type, with real
quotes you can paste straight into your README.

Usage:
    python -u src/failure_analysis.py
"""

import pandas as pd

PREDICTIONS_PATH = "data/processed/pipeline_predictions.csv"


def main():
    df = pd.read_csv(PREDICTIONS_PATH)
    valid = df[df["predicted_intent"] != "ERROR"].copy()

    print("=" * 70)
    print("1. WHERE EACH INTENT GETS CONFUSED (most common wrong guess)")
    print("=" * 70)
    for true_intent in sorted(valid["final_intent"].unique()):
        subset = valid[valid["final_intent"] == true_intent]
        wrong = subset[subset["predicted_intent"] != true_intent]
        if len(wrong) == 0:
            continue
        top_confusion = wrong["predicted_intent"].value_counts().idxmax()
        top_count = wrong["predicted_intent"].value_counts().max()
        print(f"  {true_intent}: {len(wrong)}/{len(subset)} wrong -> most often called '{top_confusion}' ({top_count}x)")

    print("\n" + "=" * 70)
    print("2. MOST IMPORTANT FAILURE: auto-handled something that should've been escalated")
    print("   (this is the costly mistake direction for a real support system)")
    print("=" * 70)
    risky = valid[(valid["auto_or_escalate"] == "escalate") & (valid["predicted_decision"] == "auto")]
    print(f"  {len(risky)} of {len(valid)} rows ({len(risky)/len(valid)*100:.1f}%)\n")
    for _, row in risky.head(5).iterrows():
        print(f"  Customer: {row['text_customer'][:100]}")
        print(f"  Intent: {row['final_intent']} | Should have been: escalate | AI decided: auto")
        print(f"  AI's draft reply: {row['draft_reply'][:120]}")
        print()

    print("=" * 70)
    print("3. language_redirect MISSES (non-English messages the AI didn't catch)")
    print("=" * 70)
    lang_misses = valid[(valid["final_intent"] == "language_redirect") & (valid["predicted_intent"] != "language_redirect")]
    print(f"  {len(lang_misses)} missed\n")
    for _, row in lang_misses.head(4).iterrows():
        print(f"  \"{row['text_customer'][:100]}\"")
        print(f"  -> AI called this: {row['predicted_intent']}\n")

    print("=" * 70)
    print("4. complaint_vent FALSE ALARMS (real issues wrongly called 'just venting')")
    print("=" * 70)
    vent_fp = valid[(valid["predicted_intent"] == "complaint_vent") & (valid["final_intent"] != "complaint_vent")]
    print(f"  {len(vent_fp)} false alarms\n")
    for _, row in vent_fp.head(4).iterrows():
        print(f"  \"{row['text_customer'][:100]}\"")
        print(f"  -> Real intent: {row['final_intent']}\n")

    print("=" * 70)
    print("5. Draft replies with possible text glitches (missing spaces, odd joins)")
    print("=" * 70)
    # Specifically: punctuation immediately followed by a CAPITAL letter with
    # no space (e.g. "correct?Let us know"). Must be uppercase only - allowing
    # lowercase too falsely matched every link like "https://t.co/..." (a
    # period followed by lowercase "c" in "t.co" is not a glitch).
    glitch_pattern = r"[.?!,][A-Z]"
    glitches = valid[valid["draft_reply"].str.contains(glitch_pattern, regex=True, na=False)]
    print(f"  {len(glitches)} possible glitches found (quick heuristic, check these by eye)\n")
    for _, row in glitches.head(5).iterrows():
        print(f"  \"{row['draft_reply'][:140]}\"\n")


if __name__ == "__main__":
    main()