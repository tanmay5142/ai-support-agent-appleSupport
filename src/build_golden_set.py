"""
Build the golden evaluation set sample.

What this does:
- Loads the paired customer/reply data.
- Uses simple keyword rules to give each row a rough "guess intent" —
  just to help spread out the sample across categories. This is NOT the
  final label. You will still manually check and correct every row.
- Picks a mix of rows: some from each rough category, plus some fully
  random rows, so the final set isn't 90% one category and isn't just
  luck-of-the-draw either.
- Saves everything to one CSV with empty columns ready for you to fill in
  by hand: final_intent, auto_or_escalate, notes.

Usage:
    python -u src/build_golden_set.py
"""

import pandas as pd

PAIRS_PATH = "data/processed/apple_pairs.csv"
OUT_PATH = "data/processed/golden_set_to_label.csv"

TARGET_TOTAL = 200          # aim for 200 examples (inside the 150-250 range)
PER_CATEGORY_TARGET = 18    # aim for roughly this many per rough category
RANDOM_STATE = 42           # fixed seed so this is reproducible

# Very rough keyword rules — just to help spread the sample across intents.
# Not meant to be accurate. You will manually fix the real label for every row.
ROUGH_RULES = {
    "battery_performance": ["battery", "drain", "slow", "overheat", "hot", "lag"],
    "bug_after_update": ["bug", "glitch", "broken", "freeze", "freezing", "crash", "stuck"],
    "howto_settings": ["how do i", "how to", "settings", "brightness", "wifi", "wi-fi"],
    "device_troubleshoot_dm": ["not working", "won't", "wont", "doesn't work", "issue", "problem"],
    "update_compatibility": ["support", "compatible", "does it work", "ios 1", "update"],
    "warranty_repair": ["warranty", "repair", "broken screen", "replace", "defect"],
    "language_redirect": [],  # handled separately below (non-ASCII check)
    "praise_thanks": ["thank", "thanks", "great", "love", "appreciate"],
}


def guess_intent(text: str) -> str:
    text_lower = str(text).lower()

    # crude check for non-English text: if it has a notable share of
    # non-ASCII letters, guess it's a language-redirect case
    non_ascii_count = sum(1 for ch in text if ord(ch) > 127)
    if non_ascii_count > 5:
        return "language_redirect"

    for label, keywords in ROUGH_RULES.items():
        for kw in keywords:
            if kw in text_lower:
                return label

    return "other_unclear"


def main():
    pairs = pd.read_csv(PAIRS_PATH)
    pairs["rough_guess_intent"] = pairs["text_customer"].apply(guess_intent)

    print("Rough category counts across full dataset:")
    print(pairs["rough_guess_intent"].value_counts())

    picked_chunks = []
    for label in pairs["rough_guess_intent"].unique():
        subset = pairs[pairs["rough_guess_intent"] == label]
        n = min(PER_CATEGORY_TARGET, len(subset))
        picked_chunks.append(subset.sample(n=n, random_state=RANDOM_STATE))

    stratified = pd.concat(picked_chunks)

    # top up with plain random rows (not already picked) to reach the target
    remaining_needed = max(0, TARGET_TOTAL - len(stratified))
    leftover_pool = pairs.drop(stratified.index)
    topup = leftover_pool.sample(n=min(remaining_needed, len(leftover_pool)), random_state=RANDOM_STATE)

    golden = pd.concat([stratified, topup]).drop_duplicates(subset=["tweet_id_customer"])
    golden = golden.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)  # shuffle order

    # empty columns for you to fill in by hand
    golden["final_intent"] = ""
    golden["auto_or_escalate"] = ""
    golden["notes"] = ""

    cols = [
        "tweet_id_customer", "text_customer", "text_reply",
        "rough_guess_intent", "final_intent", "auto_or_escalate", "notes",
    ]
    golden[cols].to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(golden)} rows to {OUT_PATH}")
    print("Open this file in Excel/Sheets and fill in: final_intent, auto_or_escalate, notes")


if __name__ == "__main__":
    main()
