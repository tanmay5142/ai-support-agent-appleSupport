# Escalation Policy — When the AI replies on its own vs. sends to a human

This rule is based on what Apple's own support team actually does in real replies
(seen in the paired dataset). The AI copies that same behavior.

## Auto-handle (AI can reply on its own) when:

- Customer just says thanks — no action needed.
- Customer is writing in a different language — Apple just points them to
  language-specific support, so the AI can do the same.
- Customer asks a simple "how do I do X" question that has one clear public
  answer (a settings toggle, a known article).
- Customer asks a general question like "does X work on Y device" — not tied
  to their personal account or personal data.

## Escalate to a human when:

- The issue needs personal details to solve (Apple itself asks for a DM,
  serial number, or account info for this kind of issue — so the AI copies
  that and hands off instead of guessing).
- It's about warranty or a broken/defective device — money and liability are
  involved, needs a real person.
- The customer is angry or venting and it's not clear what they actually want
  fixed.
- (Once the real classifier is built) the classifier isn't confident about
  the intent — safer to hand off than guess.

## Why this rule

It's not a random guess — it copies what Apple's own support team already
does for each type of issue. That makes it easy to defend later: "why did you
escalate this one?" can always be answered with "because that's what the
brand itself does for this kind of issue."
