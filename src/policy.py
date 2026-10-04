"""
Decides whether a message should be auto-handled or sent to a human,
following the same rule written in policy.md - and gives a one-line
reason, since the assignment specifically asks for a stated reason.

Usage (standalone test):
    python -u src/policy.py "thanks so much, that fixed it!" praise_thanks
"""

import sys
from llm_client import ask_llm_json

POLICY_TEXT = """
Auto-handle when:
- The customer is just saying thanks, no action needed.
- The customer is writing in a different language (just needs a language redirect).
- It's a simple how-to/settings question with one clear public answer.
- It's a general compatibility/availability question, not tied to their personal account.

Escalate to a human when:
- The issue needs personal or account-specific details to solve (this covers
  most device troubleshooting, since Apple's own team usually asks for this).
- It's about warranty, repair, a broken/defective device, or a billing/charge dispute.
- The customer is angry or venting with no clear, specific, resolvable request.
- The message is too vague/short to tell what's actually needed.
"""


def build_prompt(message: str, intent: str, draft_reply: str) -> str:
    return f"""Apply this support policy to decide if a message should be
auto-handled by an AI, or escalated to a human agent:

{POLICY_TEXT}

Customer message: "{message}"
Predicted intent: {intent}
Draft AI reply: "{draft_reply}"

Respond with ONLY a JSON object in this exact shape, nothing else:
{{"decision": "auto" or "escalate", "reason": "<one short sentence, referencing the policy>"}}
"""


def decide_escalation(message: str, intent: str, draft_reply: str) -> dict:
    prompt = build_prompt(message, intent, draft_reply)
    return ask_llm_json(prompt, max_tokens=350)


if __name__ == "__main__":
    message = sys.argv[1] if len(sys.argv) > 1 else "thanks so much, that fixed it!"
    intent = sys.argv[2] if len(sys.argv) > 2 else "praise_thanks"
    result = decide_escalation(message, intent, draft_reply="You're welcome!")
    print(f"Message: {message}")
    print(f"Decision: {result['decision']}")
    print(f"Reason: {result['reason']}")