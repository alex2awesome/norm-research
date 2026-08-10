"""Prompts for the articulation-STaR loop.

Design constraints:
  - Generator NEVER sees the true label.
  - Generator output has POSITIVE_ASPECTS + NEGATIVE_ASPECTS blocks, NO verdict.
  - Each aspect must reference specific content from the artifact (format
    discipline = soft block on style-only rationales).
  - Judge sees ONLY the rationale (no artifact). Predicts label by weighing.
"""

GEN_SYSTEM = (
    "You are a careful reviewer. You list specific reasons grounded in the "
    "content of the artifact. You never state an overall verdict; you only "
    "enumerate aspects on each side and let the reader weigh them."
)

GEN_USER_TEMPLATE = """\
Below is a {text_type}. Two readers will use your output to decide whether \
this artifact would be {positive_label} or {negative_label}. Your job is to \
articulate the positive and negative aspects that a member of the relevant \
community would weigh -- not to give a verdict.

ARTIFACT
--------
{text}

INSTRUCTIONS
------------
1. Produce TWO sections, exactly in this format:

   POSITIVE_ASPECTS:
   - [quoted or paraphrased specific element from the artifact] — [the normative principle it satisfies]
   - ...

   NEGATIVE_ASPECTS:
   - [quoted or paraphrased specific element from the artifact] — [the normative principle it violates or fails]
   - ...

2. Each bullet MUST have two parts separated by " — ":
   (a) a concrete element actually present (or notably absent) in the artifact,
   (b) the community norm or principle it triggers.

3. List 3-6 bullets per section. Each section must contain at least 2 bullets, \
even when the artifact strongly leans one way -- a fair reviewer can always \
find aspects on each side.

4. DO NOT state a verdict. DO NOT say "therefore", "overall", "I recommend", \
"on balance", or any equivalent. Stop after the last NEGATIVE_ASPECTS bullet.

5. Be specific to THIS artifact. Generic statements that could apply to any \
{text_type} ("the writing is clear", "the methodology is sound") are not useful.
"""


JUDGE_SYSTEM = (
    "You are a meta-evaluator. You read another reviewer's notes on an "
    "artifact you have not seen, and predict which decision the community "
    "would reach. You weigh the substance of each listed aspect; you do not "
    "merely count items. /no_think"
)

JUDGE_USER_TEMPLATE = """\
Another reviewer wrote the notes below about a {text_type}. You have not seen \
the artifact -- only these notes. Based on the substance of the positive and \
negative aspects they listed, predict which decision the community would \
reach: would the artifact be {positive_label} or {negative_label}?

REVIEWER NOTES
--------------
{rationale}

Respond with EXACTLY one of these two answers on a single line, lowercase, \
no punctuation, no quotes, no extra words:

ANSWER: {positive_label}

OR

ANSWER: {negative_label}

Do not show any reasoning. Just the single ANSWER line. /no_think
"""


def render_gen(text: str, text_type: str, pos: str, neg: str) -> list[dict]:
    return [
        {"role": "system", "content": GEN_SYSTEM},
        {"role": "user", "content": GEN_USER_TEMPLATE.format(
            text=text, text_type=text_type,
            positive_label=pos, negative_label=neg,
        )},
    ]


# ── Label-conditioned ("hint") generation ─────────────────────────────
#
# Drop the blind-extraction constraint and tell the model what the community
# decided. The model's job becomes articulating *why* — grounded reasons
# that explain the decision, not predicting it. Classical STaR
# rationalization in spirit.
#
# Used as a fallback when blind extraction produces upbeat both-sides
# rationales that the bottleneck judge can't separate from substance.
# The contrastive weak/strong judge still applies as a quality filter:
# we keep rationales whose label is decodable from substance, not from
# the LLM having been told the label and parroting it.

GEN_HINT_SYSTEM = (
    "You are a careful reviewer. You articulate specific reasons grounded "
    "in the content of an artifact, drawn from community norms of quality "
    "in this domain."
)

GEN_HINT_USER_TEMPLATE = """\
A community of readers decided that the following {text_type} was \
**{decision_label}**. Your job is to articulate the specific aspects of \
this artifact that explain that decision -- positive aspects that pulled \
toward the actual outcome, and negative aspects that pulled against it. \
A fair reviewer would name both sides even when the verdict is clear.

ARTIFACT
--------
{text}

INSTRUCTIONS
------------
1. Produce TWO sections, exactly in this format:

   POSITIVE_ASPECTS:
   - [quoted or paraphrased specific element from the artifact] — [the community norm it satisfies]
   - ...

   NEGATIVE_ASPECTS:
   - [quoted or paraphrased specific element from the artifact] — [the community norm it violates or fails]
   - ...

2. Each bullet MUST have two parts separated by " — ":
   (a) a concrete element actually present (or notably absent) in the artifact,
   (b) the community norm or principle it triggers.

3. List 3-6 bullets per section. Each section must contain at least 2 bullets.

4. DO NOT state a verdict. DO NOT say "therefore", "overall", "I recommend", \
"on balance", or any equivalent. Do NOT mention the decision the community \
reached -- only articulate the aspects.

5. Be specific to THIS artifact. Generic statements that could apply to any \
{text_type} are not useful.
"""


def render_gen_hint(text: str, text_type: str, decision_label: str) -> list[dict]:
    """Label-conditioned generation prompt — the model knows the decision."""
    return [
        {"role": "system", "content": GEN_HINT_SYSTEM},
        {"role": "user", "content": GEN_HINT_USER_TEMPLATE.format(
            text=text, text_type=text_type, decision_label=decision_label,
        )},
    ]


def render_judge(rationale: str, text_type: str, pos: str, neg: str) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
            rationale=rationale, text_type=text_type,
            positive_label=pos, negative_label=neg,
        )},
    ]
