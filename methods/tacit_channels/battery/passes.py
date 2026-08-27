"""Pass planner — bundle every W1 probe's row-needs into single engine loads.

The battery's cost law: cost ~ O(engine loads), never O(probes). Probes declare row VARIANTS;
this module builds all variant rows for a (model, adapter, domain) engine key so one
score run serves exclusion (P-INT-1), negation/composition (P-COMP-1), and the tf baseline —
and a second generation+tf stage serves reason-first (P-INTF-1/P-TOK-1) and verbalized
confidence (P-STAT-1 v1).

Single-stage variants (pure prompt transforms; ride ONE teacher-forced batch):
  tf         — the frozen baseline template (identical to the scorer's)
  exclusion  — instruct the model to judge AGAINST the criterion (Jacoby PDP; leak-through
               of the installed policy = internalized). Readout unchanged (P(YES)).
  negated    — the anti-construct (cheapest memorization test; knowing-using NOT operator)
  composed   — "A AND B" construct pairs (composition ladder; needs a matching target pass)

Two-stage variants (generate first, then teacher-forced):
  reason_first — model writes its reasoning, THEN answers; contrast vs tf gives BOTH the
                 CoT-delta (P-TOK-1) and explain-then-score degradation (P-INTF-1)
  confidence   — after judging, the model states confidence 0-100 (verbalized; upgrades the
                 log-odds proxy per prereg P-B5)

Frozen wording constants below are part of the prereg'd instrument once the battery
confirmatory wave starts — change only via dated prereg addendum.
"""
from __future__ import annotations

EXCLUSION_PREFIX = (
    "IMPORTANT INVERSION INSTRUCTION: Deliberately judge AGAINST the criterion below. "
    "Answer YES only if the item FAILS the criterion, and NO if it satisfies it.\n\n")

NEGATED_WRAPPER = (
    "The criterion is the ABSENCE of the following: the item should NOT exhibit it.\n\n{content}")

COMPOSED_WRAPPER = "{content_a}\n\nAND SIMULTANEOUSLY:\n\n{content_b}"

REASON_FIRST_INSTR = (
    "Before answering, briefly explain the criterion-relevant considerations for this item "
    "(2-3 sentences). Then answer.")

CONFIDENCE_QUESTION = (
    "How confident are you in that judgment? Reply with a single integer 0-100.")

HOLISTIC_PROMPT = (
    "Judge the overall quality of this item as an instance of its genre, all things "
    "considered. Use your full sense of what makes such an item good — every consideration "
    "you would actually bring to bear, whether or not that consideration has a name.")

# ---- W1c wordings — ORACLE-CERTIFIED on the synthetic calibration suite (glm-4.7,
# cycles 1-2, 2026-07-24; outputs/tacit_channels/battery_calibration/). These replace the
# cycle-1-defective forms for all future passes; the W1a forms above stay frozen for the
# already-scored grids' provenance. Change only via dated prereg addendum + recalibration.

# exclusion: same prefix, but the template's final question is REPLACED with this
# (the deployed straight question cost a measured .24 accuracy on mechanical rules):
EXCLUSION_FIXED_QUESTION = (
    "Per the inversion instruction above: answer YES if the text FAILS the stated "
    "criterion, NO if it satisfies it. Answer with exactly one word: YES or NO.")

# negation: explicit-mapping wrapper + consistent question (cycle-1 wrapper compliance was
# construct-dependent, .15-1.0; this form calibrates .88-1.0 on all oracled constructs and
# beats hand-negated predicates on ambiguous ones):
NEG_FX_WRAPPER = ("INVERTED CRITERION: the criterion for this evaluation is the ABSENCE "
                  "of the following property.\n\nProperty: {content}")
NEG_FX_QUESTION = (
    "Is the property ABSENT from the text? Answer YES if the property is absent, NO if "
    "it is present. Answer with exactly one word: YES or NO.")

# holistic: graded elicitation replaces the floor-collapsing YES/NO form (calibrated:
# parse 1.0, rho +.83 vs a known mixture, full scale use on no-oracle constructs):
HOLISTIC_GRADED_TEMPLATE = (
    "You are rating a text.\n\n{rubric}\n\nText:\n{text}\n\n"
    "Reply with a single integer 0-10.")


def name_form_prompt(cell: dict, form_id: str) -> str | None:
    for arm in cell["arms"]:
        if arm["id"] == "name":
            for form in arm["forms"]:
                if form["id"] == form_id:
                    return form["prompt"]
    return None


def build_single_stage_rows(cells: dict, variants: tuple, composed_pairs: tuple = (),
                            forms: tuple = ("canonical", "question", "boilerplate")) -> list:
    """Rows for ONE teacher-forced batch: [{cell_id, variant, form, content, meta...}].
    `content` is what goes into readout_template.format(rubric=..., text=item)."""
    rows = []
    for cell_id, cell in sorted(cells.items()):
        for form_id in forms:
            base = name_form_prompt(cell, form_id)
            if base is None:
                continue
            if "tf" in variants:
                rows.append({"cell_id": cell_id, "variant": "tf", "form": form_id,
                             "content": base})
            if "exclusion" in variants:
                rows.append({"cell_id": cell_id, "variant": "exclusion", "form": form_id,
                             "content": EXCLUSION_PREFIX + base})
            if "negated" in variants:
                rows.append({"cell_id": cell_id, "variant": "negated", "form": form_id,
                             "content": NEGATED_WRAPPER.format(content=base)})
    if "composed" in variants:
        for (cell_a, cell_b) in composed_pairs:
            for form_id in forms:
                a = name_form_prompt(cells[cell_a], form_id)
                b = name_form_prompt(cells[cell_b], form_id)
                if a is None or b is None:
                    continue
                rows.append({"cell_id": f"{cell_a}&&{cell_b}", "variant": "composed",
                             "form": form_id,
                             "content": COMPOSED_WRAPPER.format(content_a=a, content_b=b),
                             "pair": [cell_a, cell_b]})
    return rows


def build_holistic_row(domain: str) -> dict:
    """The unnamed-residual instrument: one domain-level all-things-considered row.
    Reference vectors for the residual R^2 are the domain's named target policies."""
    return {"cell_id": f"HOLISTIC::{domain}", "variant": "holistic", "form": "canonical",
            "content": HOLISTIC_PROMPT}


def build_reason_first_prompts(template: str, rows: list, texts: list,
                               max_text_chars: int) -> list:
    """Stage 1 of reason_first: generation prompts (one per row x item)."""
    out = []
    for i, row in enumerate(rows):
        for j, text in enumerate(texts):
            base = template.format(rubric=row["content"], text=text[:max_text_chars])
            base = base.replace("Answer with exactly one word: YES or NO.",
                                REASON_FIRST_INSTR)
            out.append({"row_index": i, "item_index": j, "prompt": base})
    return out


def assemble_reason_first_tf(template: str, row: dict, text: str, rationale: str,
                             max_text_chars: int) -> str:
    """Stage 2: the teacher-forced prompt with the model's own rationale interposed."""
    base = template.format(rubric=row["content"], text=text[:max_text_chars])
    return (base.replace("Answer with exactly one word: YES or NO.",
                         f"Reasoning: {rationale.strip()}\n\n"
                         "Answer with exactly one word: YES or NO."))


def confidence_prompt(judgment_prompt: str, answer: str) -> str:
    """Verbalized-confidence elicitation given the model's own answer."""
    return f"{judgment_prompt}\n\nYour answer was: {answer}\n\n{CONFIDENCE_QUESTION}"


def plan_summary(rows: list) -> dict:
    from collections import Counter
    return dict(Counter(r["variant"] for r in rows))
