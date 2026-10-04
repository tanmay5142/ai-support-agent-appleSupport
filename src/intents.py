"""
Classifies a customer message into one of our 8 intent categories,
using the LLM with the definitions we worked out from real data.

Usage (standalone test):
    python -u src/intents.py "thanks so much, that fixed it!"
"""

import sys
from llm_client import ask_llm_json

# These are the 11 categories that actually came out of hand-labeling the
# golden set (the original 8 from early data exploration, plus 3 more real
# patterns - billing, pure venting, and unclear/fragment messages - that
# only showed up once we labeled a larger sample). Using the SAME list here
# as in the golden set is important: if the classifier used a different set
# of categories, scoring it against your hand labels wouldn't line up.
INTENT_DEFINITIONS = {
    "battery_performance": "Complaints about battery drain, device running hot, or slowness.",
    "bug_after_update": "A bug, crash, freeze, or glitch that started after an iOS/software update.",
    "howto_settings": "A general 'how do I do X' or settings question, answerable with public info.",
    "device_troubleshoot_dm": "A device issue that needs personal/account-specific diagnosis (Apple would normally ask for a DM or device details).",
    "update_compatibility": "Questions about whether something is supported, available, or compatible.",
    "warranty_repair": "Warranty, repair, hardware defect, or a physically broken/defective device.",
    "billing_complaint": "Complaints about being charged incorrectly or money-related disputes.",
    "language_redirect": "Message written in a language other than English.",
    "praise_thanks": "A thank-you or positive feedback message, no action needed.",
    "complaint_vent": "Angry or frustrated venting with no clear, specific, resolvable request.",
    "unclear_needs_more_info": "Too short or vague to tell what the customer actually needs.",
}


def build_prompt(message: str) -> str:
    categories_list = "\n".join(
        f'- "{name}": {desc}' for name, desc in INTENT_DEFINITIONS.items()
    )
    return f"""You are classifying a customer support tweet directed at AppleSupport.

Pick exactly ONE of these categories:
{categories_list}

Customer message:
\"\"\"{message}\"\"\"

Respond with ONLY a JSON object in this exact shape, nothing else:
{{"intent": "<category_name>", "reason": "<one short sentence why>"}}
"""


def classify_intent(message: str) -> dict:
    prompt = build_prompt(message)
    result = ask_llm_json(prompt, max_tokens=350)
    return result


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "thanks so much, that fixed it!"
    result = classify_intent(query)
    print(f"Message: {query}")
    print(f"Predicted intent: {result['intent']}")
    print(f"Reason: {result['reason']}")