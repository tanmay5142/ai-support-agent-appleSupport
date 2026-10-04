"""
The full agent pipeline for one customer message:
1. Classify the intent
2. Retrieve similar historical examples
3. Draft a grounded reply
4. Decide auto-handle vs escalate, with a reason

Usage:
    python -u src/pipeline.py "my battery is draining fast since the update"
"""

import sys
import json
from intents import classify_intent
from generate import generate_reply
from policy import decide_escalation
from retrieval import Retriever


def run_pipeline(message: str, retriever: Retriever = None) -> dict:
    if retriever is None:
        retriever = Retriever()

    intent_result = classify_intent(message)
    intent = intent_result["intent"]

    reply_result = generate_reply(message, retriever=retriever)
    draft_reply = reply_result["draft_reply"]

    decision_result = decide_escalation(message, intent, draft_reply)

    return {
        "message": message,
        "intent": intent,
        "intent_reason": intent_result["reason"],
        "draft_reply": draft_reply,
        "grounded_on": reply_result["grounded_on"],
        "decision": decision_result["decision"],
        "decision_reason": decision_result["reason"],
    }


if __name__ == "__main__":
    message = sys.argv[1] if len(sys.argv) > 1 else "my battery is draining fast since the update"

    result = run_pipeline(message)

    print(f"MESSAGE:  {result['message']}\n")
    print(f"INTENT:   {result['intent']}")
    print(f"          (why: {result['intent_reason']})\n")
    print(f"DRAFT REPLY: {result['draft_reply']}\n")
    print(f"DECISION: {result['decision'].upper()}")
    print(f"          (why: {result['decision_reason']})\n")
    print("Grounded on:")
    for ex in result["grounded_on"]:
        print(f"  - {ex['customer_message'][:70]}...")