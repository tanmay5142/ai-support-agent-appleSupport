"""
Drafts a reply to a new customer message, grounded in how Apple actually
replied to similar past messages (from retrieval.py).

Usage (standalone test):
    python -u src/generate.py "my battery is draining fast since the update"
"""

import sys
from llm_client import ask_llm
from retrieval import Retriever


def build_prompt(message: str, examples: list) -> str:
    examples_text = "\n\n".join(
        f'Past customer message: "{ex["customer_message"]}"\n'
        f'How Apple replied: "{ex["brand_reply"]}"'
        for ex in examples
    )

    return f"""You are AppleSupport's customer service team, replying on Twitter.

Here are some real past examples of similar customer messages and how
Apple actually replied to them:

{examples_text}

Now write a reply to this new customer message, in the same style, tone,
and approach Apple used above (short, friendly, and following the same
kind of next step - e.g. asking for a DM if the past examples did that,
or pointing to a general fix if the past examples did that):

New customer message: "{message}"

Reply with ONLY the reply text, nothing else. Keep it under 280 characters,
matching Twitter's style.
"""


def generate_reply(message: str, retriever: Retriever = None) -> dict:
    if retriever is None:
        retriever = Retriever()

    examples = retriever.find_similar(message, k=3)
    prompt = build_prompt(message, examples)
    draft_reply = ask_llm(prompt, max_tokens=350).strip()

    return {
        "draft_reply": draft_reply,
        "grounded_on": examples,
    }


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "my battery is draining fast since the update"
    result = generate_reply(query)

    print(f"Customer message: {query}\n")
    print(f"Draft reply: {result['draft_reply']}\n")
    print("Grounded on these past examples:")
    for ex in result["grounded_on"]:
        print(f"  - (similarity {ex['similarity']}) {ex['customer_message'][:80]}...")