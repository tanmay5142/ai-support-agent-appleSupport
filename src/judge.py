"""
Judges the quality of an AI-drafted reply by comparing it to the real
historical reply the brand actually sent for that same message.

This is more concrete than just asking "is this a good reply?" in a
vacuum - it's asking "did the AI do roughly what the real support team
did?", which is a fair question since the whole point of this project
is replies grounded in real historical resolutions.

Usage (standalone test):
    python -u src/judge.py
"""

from llm_client import ask_llm_json


def build_prompt(message: str, draft_reply: str, real_reply: str) -> str:
    return f"""You are evaluating an AI-drafted customer support reply by
comparing it to how the brand's real support team actually replied to
this exact message in the past.

Customer message: "{message}"

Real reply (what the brand's support team actually said): "{real_reply}"

AI-drafted reply (the one to evaluate): "{draft_reply}"

Rate the AI-drafted reply on each of these, using a 1-5 scale (1 = poor, 5 = excellent):
- relevance: does it address the same core issue as the real reply?
- tone: is it polite, professional, and appropriately toned for customer support?
- approach_match: does it take a similar kind of next step as the real reply
  (e.g. both ask for more info, both point to a fix, both redirect to DM) -
  even if the exact wording is different?
- overall: overall quality as a customer support reply

Respond with ONLY a JSON object in this exact shape, nothing else:
{{"relevance": <1-5>, "tone": <1-5>, "approach_match": <1-5>, "overall": <1-5>, "justification": "<one short sentence>"}}
"""


def judge_reply(message: str, draft_reply: str, real_reply: str) -> dict:
    prompt = build_prompt(message, draft_reply, real_reply)
    return ask_llm_json(prompt, max_tokens=350)


if __name__ == "__main__":
    # quick manual test with made-up example
    message = "my battery is draining fast since the update"
    real_reply = "We'd like to help. Send us a DM with your device and iOS version."
    draft_reply = "Sorry to hear that! Could you let us know which device and iOS version you're on? DM us and we'll help from there."

    result = judge_reply(message, draft_reply, real_reply)
    print(f"Message: {message}")
    print(f"Real reply: {real_reply}")
    print(f"Draft reply: {draft_reply}\n")
    print(f"Scores: {result}")