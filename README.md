# AI Customer Support Agent — AppleSupport (Twitter)

An AI agent that reads a real customer support message, figures out what
it's about, drafts a reply grounded in how the brand has actually resolved
similar issues in the past, and decides whether it's safe to send
automatically or whether it needs a human.

Built on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset (Kaggle), focused on **AppleSupport**.

> Originally started as a take-home exercise (Hiver SDE Intern), finished
> independently as a personal project after deciding not to submit it.

---

## What it does

1. **Classifies** an incoming message into one of 11 intent categories.
2. **Drafts a reply**, grounded in real similar past customer messages and
   how AppleSupport actually replied to them.
3. **Decides** auto-handle vs. escalate-to-human, with a stated reason.

---

## Why AppleSupport

Picked after comparing a few high-volume brands (Amazon, Apple, Delta) in
the raw data. AppleSupport stood out for being mostly English, and for
having a small number of clearly repeating issue types (battery drain,
bugs after an iOS update, DM-based troubleshooting) with visibly templated
replies — which makes "grounded in historical resolution" an honest claim
to make, not just a nice-sounding one.

---

## How it works

```
Customer message
      |
      v
 [intents.py]  -----> predicted intent (11 categories, LLM-based)
      |
      v
[retrieval.py] -----> finds 3 most similar past messages (TF-IDF similarity)
      |                and how AppleSupport actually replied to them
      v
[generate.py]  -----> drafts a reply in the same style/approach as those
      |                real historical replies
      v
 [policy.py]   -----> decides auto-handle vs escalate, with a reason,
                       using the same rule written out in policy.md
```

Retrieval is intentionally **not** based on intent labels — it just finds
similar-sounding past messages using TF-IDF (same technique as the simple
baseline below), so it works across all 77,714 paired messages without
needing every one of them hand-labeled.

---

## The 11 intent categories

Started from 8 categories found by skimming the data early on. Labeling
the golden set by hand surfaced 3 more real, recurring patterns
(`billing_complaint`, `complaint_vent`, `unclear_needs_more_info`) that
weren't obvious from a quick skim. Final list:

`battery_performance`, `bug_after_update`, `howto_settings`,
`device_troubleshoot_dm`, `update_compatibility`, `warranty_repair`,
`billing_complaint`, `language_redirect`, `praise_thanks`,
`complaint_vent`, `unclear_needs_more_info`

## Escalation policy

See [`policy.md`](policy.md). In short: auto-handle anything answerable
with public information (thanks, language redirects, generic how-to
questions); escalate anything needing personal/account details, money
(warranty, billing), or where the customer is upset with no clear
resolvable ask. The rule is modeled directly on what AppleSupport's own
team actually does in the real data — e.g. they ask for a DM before
handling most device-specific issues, so the agent escalates those too.

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install pandas scikit-learn openai python-dotenv scipy
```

Download the dataset (needs a free Kaggle account + API token):

```bash
kaggle datasets download -d thoughtvector/customer-support-on-twitter
# unzip into data/raw/ so the CSV ends up at data/raw/twcs/twcs.csv
```

Create a `.env` file in the project root with a [Groq](https://console.groq.com) API key:

```
GROQ_API_KEY=your_key_here
```

(Groq is used via its OpenAI-compatible endpoint, model `openai/gpt-oss-20b`
— cheap and fast enough for this project's scale. Swapping to OpenAI or
Anthropic instead only requires changing `src/llm_client.py`; nothing
else in the pipeline needs to change.)

## Running it

```bash
# 1. Build the paired dataset (customer message -> AppleSupport's reply)
python -u src/pair_conversations.py

# 2. Try the agent on one message
python -u src/pipeline.py "my battery is draining fast since the update"

# 3. Baselines (for comparison, no API calls needed)
python -u src/baseline_trivial.py
python -u src/baseline_simple.py

# 4. Run the real system across the full golden set (200 hand-labeled examples)
python -u src/evaluate_pipeline.py
# if it gets rate-limited partway through, just run:
python -u src/resume_pipeline_errors.py
python -u src/score_predictions.py

# 5. Score reply quality with an LLM judge
python -u src/run_judge.py

# 6. Check the judge against real human judgment
python -u src/build_calibration_sample.py   # then hand-score the sample yourself
python -u src/check_judge_agreement.py

# 7. Pull concrete failure examples
python -u src/failure_analysis.py
```

The golden set (`data/processed/golden_set.csv`, 200 examples) was built
by sampling a mix of rows across rough keyword-based categories plus some
fully random ones (`src/build_golden_set.py`), then hand-labeling every
row's real intent, auto/escalate decision, and a short note.

---

## Results

### Intent classification

| | Trivial baseline | Simple baseline (TF-IDF + LogReg) | **Real system (LLM)** |
|---|---|---|---|
| Accuracy | 32.0% | 42.0% | **59.0%** |
| Macro F1 | 0.044 | 0.277 | **0.634** |

### Auto-handle vs. escalate decision

| | Trivial baseline | Simple baseline | **Real system** |
|---|---|---|---|
| Accuracy | 77.5% | 78.0% | **81.0%** |
| Macro F1 | 0.437 | 0.569 | **0.703** |
| Recall on "auto" class | 0% | 18% | **47%** |

The trivial baseline's 77.5% "accuracy" is misleading on its own — it's
just always guessing "escalate," which is useless in practice since it
never auto-handles anything. The "auto" recall row is the more honest
number to watch, and it improves clearly at every stage.

### Reply quality (LLM judge, 1-5 scale, compared against the real
historical reply for each message)

| | Relevance | Tone | Approach match | Overall |
|---|---|---|---|---|
| Judge average | 4.63 | 4.77 | 4.54 | 4.56 |

### Judge vs. human agreement (30 manually re-scored examples)

| | Exact match | Within 1 point | Correlation |
|---|---|---|---|
| Relevance | 73% | 93% | 0.89 |
| Tone | 90% | 100% | 0.47* |
| Approach match | 60% | 93% | 0.71 |
| Overall | 70% | 100% | 0.78 |

\* Tone's correlation looks low only because almost every reply scored 4
or 5 — correlation needs variation to be meaningful, so this isn't a real
disagreement, just a side effect of most replies genuinely having a fine
tone.

**Takeaway:** the judge is reasonably well-calibrated, not just lenient.
Reply quality and intent accuracy measure different things — a reply can
still sound reasonable even when the underlying intent label was wrong,
since many confused categories (see below) call for similar-sounding
replies anyway.

---

## Failure analysis

**1. Three categories blur together: `bug_after_update`, `device_troubleshoot_dm`,
and `unclear_needs_more_info`.** This is the single biggest source of
error (26/64 and 32/50 wrong respectively). Likely cause: these aren't
cleanly separable from the customer's message alone — almost every device
bug genuinely is "after an update," and whether it needs personal
troubleshooting often only becomes clear from the brand's *reply*, not
the complaint itself. This same ambiguity showed up while hand-labeling
the golden set, so it's a real property of the data, not just a model
weakness.

**2. The costly failure direction: 7% (14/200) of messages were
auto-handled when they should have been escalated.** Notably, the
*drafted replies* in these cases often still asked for a DM or personal
details — meaning the reply-writing step got it right, but the
escalation-decision step didn't pick up on its own input correctly, even
though it's given the draft reply to look at.

**3. `language_redirect` misses most non-English messages (5/8 missed),**
usually mistaking them for `bug_after_update`. Hypothesis: the classifier
seems to key off recognizable English tech terms embedded in the text
(`#iOS11`, `Safari`, `@AppleSupport`) rather than checking whether the
overall message is in English.

**4. `complaint_vent` has 11 false alarms** — real, specific issues
(a billing dispute, a connectivity problem, a legitimate popup question)
wrongly dismissed as "just venting." These all share angry tone or
profanity. Hypothesis: the model conflates emotional tone with "no
actionable request," when frustrated customers often do have a clear
underlying issue.

**5. Minor: occasional missing-space typos in generated replies**
(e.g. "try upgrading andsee if that improves"). Rare, caught by manual
review rather than any automated check — word-gluing typos like this
don't follow a reliable pattern worth building a detector for at this
scale.

---

## What's misleading about the headline numbers

- **The golden set is small and imbalanced** (200 examples, with some
  categories as rare as 2-5 examples). Scores for those rare categories
  swing a lot from a single example and shouldn't be over-trusted.
- **"81% escalate accuracy" sounds strong but started from a 77.5%
  trivial floor** — the real improvement is better seen in the "auto"
  recall number (0% -> 18% -> 47%), which is still well under half.
- **The judge compares against the real historical reply**, which itself
  is sometimes just a generic "DM us" deflection rather than an actual
  resolution (see note below on pairs) — so a high judge score doesn't
  always mean the issue was truly solved, only that the AI matched the
  brand's own (sometimes non-committal) style.
- **A "pair" is the customer's message and AppleSupport's *first* reply**,
  not necessarily the final resolution — many real conversations continue
  into DMs that aren't in this dataset. The grounding and judging are
  both based on that first-reply pattern, not the eventual outcome.

---

## Key decisions (non-obvious ones)

- Chose AppleSupport over Amazon/Delta for being mostly English and
  having clearly repeating, templated issue types.
- Retrieval is based on message similarity (TF-IDF), not intent labels —
  works across the full dataset without needing everything hand-labeled.
- Taxonomy grew from 8 to 11 categories after hand-labeling revealed real
  patterns a quick skim missed.
- Escalation policy is modeled directly on the brand's own real behavior
  (if Apple asks for a DM for issue type X, the agent escalates X too)
  rather than an arbitrary rule.
- The LLM judge compares against the *real historical reply* for the same
  message, instead of scoring reply quality in the abstract.
- Switched LLM providers twice (Anthropic -> OpenAI -> Groq) with only
  `src/llm_client.py` needing to change each time, since all LLM calls
  are isolated behind one shared module.
- Built evaluation/judging scripts to save progress after every row
  (not just at the end), after hitting a free-tier rate limit partway
  through a run — re-running just picks up where it left off instead of
  wasting API calls redoing completed work.

---

## What I'd do with more time

- Use embeddings instead of TF-IDF for retrieval — would likely find
  better-matched grounding examples, especially for shorter messages.
- Fix the escalation-decision step to react more reliably to its own
  drafted reply (failure mode #2 above).
- Add a lightweight language-detection step before classification, to
  fix the `language_redirect` recall problem directly instead of relying
  on the LLM to infer it from context.
- Grow the golden set, especially for the rare categories
  (`billing_complaint`, `update_compatibility`) where current scores are
  based on very few examples.
- Try to follow conversations past the first reply, to judge against real
  final resolutions rather than first-response patterns.

---

## Tech stack

Python, pandas, scikit-learn (TF-IDF + Logistic Regression baseline),
Groq API (`openai/gpt-oss-20b`, OpenAI-compatible client).
