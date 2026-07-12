"""Rich, deterministic feature extraction over a rubric/prompt body — "all aspects of the
prompt" for the scaling-axis discovery analysis (Stage 0, 2026-06-18).

`artifact.measure_prompt` gives the three budget-cap features (instruction_tokens, clause_count,
n_fewshots). This module is the superset: every prompt property we can read *without an LLM and
deterministically*, grouped so the analysis can ask which of them (a) actually VARY across a
population of optimized prompts and (b) co-vary with recovery — i.e. which lend themselves to a
scaling-law axis vs which are dead.

Pure stdlib (re + math). All regexes are simple word-boundary alternations (no nested optional
quantifiers) so they cannot backtrack pathologically [[feedback_regex_backtracking_sigalrm]].
The three budget keys are taken verbatim from `artifact.measure_prompt` so they stay consistent
with the optimizer's caps.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List

from .artifact import measure_prompt

# ---- compiled patterns (kept simple & anchored) -----------------------------------------

_WORD = re.compile(r"[A-Za-z']+")
_SENT_END = re.compile(r"[.!?](?:\s|$)")
_BULLET = re.compile(r"(?:^|\n)\s*(?:[-*•]|\d+[.)])\s")
_NUMBERED = re.compile(r"(?:^|\n)\s*\d+[.)]\s")
_NUMERIC = re.compile(r"\b\d+(?:\.\d+)?\b")
_QUOTED = re.compile(r"[\"“'][^\"”']{2,80}[\"”']")

_CONDITIONAL = re.compile(r"\b(?:if|when|unless|except|whenever|in case|provided that)\b", re.I)
_AGGREGATION = re.compile(
    r"\b(?:aggregat\w*|average|combine|sum of|overall score|final score|weighted|"
    r"total score|add up)\b", re.I)
_DECOMP = re.compile(
    r"\b(?:step\s*\d|sub-?question|sub-?criteri\w*|first,|then,|next,|finally,|"
    r"break (?:this|it) (?:down|into)|for each (?:of the )?(?:following|criteri))\b", re.I)
_EDGE = re.compile(r"\b(?:edge case|corner case|exception|special case|ambiguous|borderline)\b", re.I)
_DEFINITION = re.compile(r"\b(?:defined as|is defined|means that|refers to|i\.e\.|that is,|"
                         r"in other words)\b", re.I)
_SCALE_ANCHOR = re.compile(
    r"(?:\b0\.0\b|\b0\.5\b|\b1\.0\b|\bscore of\b|\b0 to 1\b|\[0,\s*1\]|"
    r"\b(?:assign|give|score)\b[^.\n]{0,30}\b(?:0|1|0\.5)\b)", re.I)

_MODAL = re.compile(r"\b(?:must|should|shall|need to|required to|has to|ought to)\b", re.I)
_HEDGE = re.compile(r"\b(?:may|might|could|generally|usually|often|tends? to|somewhat|"
                    r"fairly|relatively|approximately|roughly|more or less|in general)\b", re.I)
_VAGUE_QUALITY = re.compile(
    r"\b(?:good|bad|appropriate|adequate|reasonable|sufficient|nice|fine|quality|"
    r"well[- ]written|clear|strong|weak|poor|excellent|decent|solid)\b", re.I)
_INTENSIFIER = re.compile(r"\b(?:very|clearly|highly|extremely|truly|particularly|"
                          r"especially|significantly|substantially)\b", re.I)
_NEGATION = re.compile(r"\b(?:not|no|never|none|cannot|can't|don't|doesn't|without|"
                       r"neither|nor)\b", re.I)
_PROCEDURAL = re.compile(r"\b(?:step|first|second|third|then|next|finally|begin by|"
                         r"start by|proceed)\b", re.I)

# imperative openers: a clause that starts with one of these verbs reads as an instruction
_IMPERATIVE_VERBS = {
    "score", "rate", "assess", "evaluate", "judge", "determine", "decide", "check",
    "verify", "identify", "count", "consider", "look", "compare", "examine", "give",
    "assign", "award", "penalize", "reward", "read", "mark", "find", "list", "ensure",
    "describe", "explain", "output", "return", "respond", "rank",
}
_CLAUSE_SPLIT = re.compile(r"(?:^|\n|[.!?]\s+)")


def _syllables(word: str) -> int:
    w = word.lower()
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def _flesch(words: List[str], n_sentences: int) -> float:
    """Flesch reading ease (higher = easier). Returns NaN for empty input."""
    nw = len(words)
    if nw == 0 or n_sentences == 0:
        return float("nan")
    syl = sum(_syllables(w) for w in words)
    return 206.835 - 1.015 * (nw / n_sentences) - 84.6 * (syl / nw)


def extract_prompt_features(body: str) -> Dict[str, float]:
    """Flat dict of deterministic prompt features. Keys are stable; see FEATURE_GROUPS for
    grouping. Counts that scale with length also get a per-100-word *density* twin so the
    analysis can separate "uses more conditionals" from "is just longer"."""
    body = body or ""
    words = _WORD.findall(body)
    nw = max(len(words), 1)
    n_sent = max(len(_SENT_END.findall(body)), 1)
    lines = [ln for ln in body.splitlines() if ln.strip()]
    lower = [w.lower() for w in words]

    base = measure_prompt(body)                       # instruction_tokens, clause_count, n_fewshots
    bullets = len(_BULLET.findall(body))
    numbered = len(_NUMBERED.findall(body))
    n_numeric = len(_NUMERIC.findall(body))
    n_quoted = len(_QUOTED.findall(body))
    n_cond = len(_CONDITIONAL.findall(body))
    n_modal = len(_MODAL.findall(body))
    n_hedge = len(_HEDGE.findall(body))
    n_vague = len(_VAGUE_QUALITY.findall(body))
    n_neg = len(_NEGATION.findall(body))
    n_proc = len(_PROCEDURAL.findall(body))
    n_q = body.count("?")

    # imperative clauses: clause starts with an instruction verb
    n_imper = 0
    for clause in _CLAUSE_SPLIT.split(body):
        m = _WORD.search(clause)
        if m and m.group(0).lower() in _IMPERATIVE_VERBS:
            n_imper += 1

    sent_lens = [len(_WORD.findall(s)) for s in _SENT_END.split(body) if s.strip()]
    ttr = len(set(lower)) / nw
    mean_word_len = sum(len(w) for w in words) / nw
    pct_long_words = sum(1 for w in words if len(w) > 6) / nw

    def dens(x):                                       # count per 100 words (length-normalized)
        return 100.0 * x / nw

    feats = {
        # ---- size ----
        "char_count": float(len(body)),
        "word_count": float(len(words)),
        "instruction_tokens": float(base["instruction_tokens"]),
        "sentence_count": float(len(_SENT_END.findall(body))),
        "line_count": float(len(lines)),
        "mean_sentence_len": float(sum(sent_lens) / len(sent_lens)) if sent_lens else 0.0,
        "max_sentence_len": float(max(sent_lens)) if sent_lens else 0.0,
        # ---- structure ----
        "clause_count": float(base["clause_count"]),
        "bullet_count": float(bullets),
        "numbered_item_count": float(numbered),
        "n_fewshots": float(base["n_fewshots"]),
        "has_worked_example": float(base["n_fewshots"] > 0),
        "has_scale_anchors": float(bool(_SCALE_ANCHOR.search(body))),
        "has_aggregation_rule": float(bool(_AGGREGATION.search(body))),
        "is_decomposed": float(bool(_DECOMP.search(body)) or numbered >= 2),
        "n_subquestions": float(n_q),
        "n_conditionals": float(n_cond),
        "cond_density": dens(n_cond),
        "has_edge_case": float(bool(_EDGE.search(body)) or n_cond > 0),
        "has_definition": float(bool(_DEFINITION.search(body))),
        "n_imperatives": float(n_imper),
        # ---- specificity ----
        "n_numeric_mentions": float(n_numeric),
        "numeric_density": dens(n_numeric),
        "n_quoted_phrases": float(n_quoted),
        "n_modals": float(n_modal),
        "modal_density": dens(n_modal),
        "specificity_ratio": float((n_numeric + n_quoted) / n_sent),
        # ---- vagueness (the inverse axis) ----
        "n_hedges": float(n_hedge),
        "hedge_density": dens(n_hedge),
        "n_vague_quality": float(n_vague),
        "vague_density": dens(n_vague),
        "n_intensifiers": float(len(_INTENSIFIER.findall(body))),
        "n_negations": float(n_neg),
        # ---- linguistic / readability ----
        "type_token_ratio": float(ttr),
        "mean_word_len": float(mean_word_len),
        "pct_long_words": float(pct_long_words),
        "flesch_reading_ease": float(_flesch(words, n_sent)),
        "procedural_marker_count": float(n_proc),
        "procedural_density": dens(n_proc),
    }
    return feats


FEATURE_GROUPS: Dict[str, List[str]] = {
    "size": ["char_count", "word_count", "instruction_tokens", "sentence_count",
             "line_count", "mean_sentence_len", "max_sentence_len"],
    "structure": ["clause_count", "bullet_count", "numbered_item_count", "n_fewshots",
                  "has_worked_example", "has_scale_anchors", "has_aggregation_rule",
                  "is_decomposed", "n_subquestions", "n_conditionals", "cond_density",
                  "has_edge_case", "has_definition", "n_imperatives"],
    "specificity": ["n_numeric_mentions", "numeric_density", "n_quoted_phrases", "n_modals",
                    "modal_density", "specificity_ratio"],
    "vagueness": ["n_hedges", "hedge_density", "n_vague_quality", "vague_density",
                  "n_intensifiers", "n_negations"],
    "linguistic": ["type_token_ratio", "mean_word_len", "pct_long_words",
                   "flesch_reading_ease", "procedural_marker_count", "procedural_density"],
}

# provenance covariates (logged by the registry, not extracted from text) that the analysis
# treats alongside the intrinsic features as candidate scaling axes.
COVARIATE_COLUMNS = ["data_budget_n", "judge_model", "operator", "optimizer",
                     "optimizer_round", "lineage_depth", "token_cap"]


def all_feature_names() -> List[str]:
    return [f for group in FEATURE_GROUPS.values() for f in group]


if __name__ == "__main__":      # smoke test: eyeball features on contrasting rubrics
    import json
    samples = {
        "vague": "Score whether the solution is good. Give a high score if it is well-written "
                 "and clear, otherwise a low score.",
        "specific_decomposed": (
            "Evaluate the solution in 3 steps. Step 1: check whether every function has a "
            "docstring (0 or 1). Step 2: count comments that explain WHY (not what). Step 3: "
            "if the comment density exceeds 0.2, score 1.0; between 0.1 and 0.2, score 0.5; "
            "below 0.1, score 0.0. Aggregate: average the three sub-scores.\n"
            "EXAMPLE 1:\n```\n# binary search to avoid O(n)\n```\nScore: 1.0"),
    }
    for name, body in samples.items():
        print(f"\n=== {name} ===")
        print(json.dumps(extract_prompt_features(body), indent=1))
