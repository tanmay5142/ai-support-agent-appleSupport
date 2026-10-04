"""
Retrieval: given a new customer message, find past messages that sound
similar, and show how Apple actually replied to those. This is the
"grounding" piece — instead of the AI making up a reply from nothing,
it looks at real historical examples first.

How it works, simply:
- Every past customer message gets turned into a list of "important
  words" (TF-IDF - same idea as the simple baseline you already ran).
- A new message gets turned into the same kind of list.
- We compare the new message to every past one and pick the closest
  matches (cosine similarity - a standard way to measure "how similar
  are these two lists of words").

Usage (standalone test):
    python -u src/retrieval.py "my battery is draining so fast after the update"
"""

import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PAIRS_PATH = "data/processed/apple_pairs.csv"


class Retriever:
    def __init__(self, pairs_path: str = PAIRS_PATH):
        self.pairs = pd.read_csv(pairs_path)
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),
        )
        # Fit once on all historical customer messages
        self.message_vectors = self.vectorizer.fit_transform(
            self.pairs["text_customer"].fillna("")
        )

    def find_similar(self, message: str, k: int = 3):
        """Return the top-k most similar historical (customer, reply) pairs."""
        query_vector = self.vectorizer.transform([message])
        similarities = cosine_similarity(query_vector, self.message_vectors)[0]

        top_indices = similarities.argsort()[::-1][:k]

        results = []
        for idx in top_indices:
            results.append({
                "customer_message": self.pairs.iloc[idx]["text_customer"],
                "brand_reply": self.pairs.iloc[idx]["text_reply"],
                "similarity": round(float(similarities[idx]), 3),
            })
        return results


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "my battery drains so fast since the update"

    retriever = Retriever()
    matches = retriever.find_similar(query, k=3)

    print(f"Query: {query}\n")
    for i, match in enumerate(matches, 1):
        print(f"--- Match {i} (similarity: {match['similarity']}) ---")
        print(f"Past customer message: {match['customer_message']}")
        print(f"Apple's reply:         {match['brand_reply']}")
        print()