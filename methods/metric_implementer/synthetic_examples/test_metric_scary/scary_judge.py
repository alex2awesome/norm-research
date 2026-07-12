"""Deterministic, zero-spend planted judge for the `is_scary` world.

``ScaryFakeBackend`` is a drop-in for ``LLMBackend`` (same ``generate`` / ``generate_batch``
surface) that routes by prompt content and answers every role the scorecard + optimizer use:

  * JUDGE         — scores an item by the planted rule: a cue category counts iff it BOTH
                    fires in the text AND is named by the rubric (``cues.planted_score``). So
                    a rubric that articulates more cue categories recovers more of the planted
                    signal — the optimizer has something real to improve.
  * RECONSTRUCTOR — reads the shown (text, score) pairs and articulates the categories that
                    drove the high scores it observed, as a rubric — a faithful read of the
                    judge's behaviour, not the ground truth.
  * GRADER        — rates how well an induced rule matches the canonical description by cue
                    overlap (1-5).
  * GENERATOR     — makes counterfactual edits: inject a not-yet-present cue (raise scariness),
                    strip all cues (lower it), or change an unrelated detail (suspect cell).
  * REVISER       — advances the rubric by articulating the next uncovered cue category, with
                    an explicit operator (ANCHOR / EDGE / DECOMPOSE / CLARIFY).
  * ATTRIBUTE     — tags counterfactual misses as AMBIGUOUS_PROMPT (the rubric under-specifies).

Everything is a pure function of the prompt text, so runs are reproducible and need no GPU,
no network, and no API key. The judge tier is tagged ``fake/scary-oracle`` so the Stage-0
analysis flags it as planted/smoke data.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Set

from ...backends import CallStats
from ...backends import Roles
from . import cues
from . import scary_metric as SM

JUDGE_TAG = "fake/scary-oracle"

# Human rubric lines per category; each line contains that category's KEYWORDS, so a rubric
# built from these lines is "covered" by cues.coverage().
_CATEGORY_LINE: Dict[str, str] = {
    "SOUND": "ominous sounds — a creak, footsteps, a scream, a scrape, sudden silence",
    "PRESENCE": "an unseen presence — a figure, a shadow, the sense of being watched or followed",
    "DREAD": "explicit fear, dread, terror, or an ominous feeling that something is wrong",
    "BODY": "bodily fear — a pounding heart, caught breath, a shiver up the spine, trembling",
    "DANGER": "signs of danger or threat — a knife, blood, an attack, harm",
}

_FENCE = re.compile(r"```(?:\w*\n)?(.*?)```", re.DOTALL)
_SCORE_PAIR = re.compile(r"\[score=([0-9.]+)\]\s*```\n(.*?)\n```", re.DOTALL)


def _render_rubric(categories: List[str]) -> str:
    """A rubric naming exactly ``categories`` (so cues.coverage() recovers them)."""
    ordered = [c for c in cues.CATEGORIES if c in set(categories)]
    lines = [f"- {c}: {_CATEGORY_LINE[c]}." for c in ordered]
    return ("Score how scary the story is, in [0,1]. Look for these cues and score higher "
            "when more appear:\n" + "\n".join(lines) +
            "\nScore 0.0 if none are present (a calm, ordinary scene); 0.5 for one cue; "
            "1.0 when several cues co-occur.")


def _strip_scary(text: str) -> str:
    out = text
    for ms in cues.MARKERS.values():
        for m in ms:
            out = re.sub(re.escape(m), "everything stayed quiet", out, flags=re.IGNORECASE)
    return out


def _inject(text: str, category: str) -> str:
    marker = cues.MARKERS[category][0]
    return text.rstrip() + " " + marker[0].upper() + marker[1:] + "."


def _pick_new_category(text: str) -> str:
    present = cues.fires(text)
    for c in cues.CATEGORIES:
        if c != "DREAD" and c not in present:   # probe a cue beyond the obvious one
            return c
    for c in cues.CATEGORIES:
        if c not in present:
            return c
    return "DANGER"


class ScaryFakeBackend:
    """Deterministic stand-in for ``LLMBackend`` over the planted `is_scary` world."""

    def __init__(self, model: str = JUDGE_TAG, role: str = "fake"):
        self.model, self.role = model, role
        self.stats = CallStats()

    # -- route table ---------------------------------------------------------------------
    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 600,
                 validate=None, temperature: Optional[float] = None) -> str:
        self.stats.n_calls += 1
        if "could a careful but limited reader" in prompt:        # ATTRIBUTE
            return json.dumps({"attributions": ["AMBIGUOUS_PROMPT"] * 8})
        if "improving the RUBRIC" in prompt:                      # REVISER (prompt kind)
            return self._revise(prompt)
        if "improving a Python implementation" in prompt:         # REVISER (code kind; unused)
            return json.dumps({"operator": "CODE_REVISE",
                               "code": "def score(text):\n    return 0.5\n",
                               "rationale": "n/a"})
        if "articulate the single rule" in prompt or "hidden evaluator" in prompt:  # RECON
            return self._reconstruct(prompt)
        if "RULE A" in prompt and "RULE B" in prompt:             # GRADER
            return self._grade(prompt)
        if '{"critique"' in prompt:                               # ProTeGi textual gradient
            return json.dumps({"critique": "the rubric names too few concrete scary cue "
                                           "categories, so it misses scary stories"})
        if 'Respond ONLY with JSON: {"rubric"' in prompt:         # EvoPrompt/ProTeGi/APE propose
            return self._advance_rubric(prompt)
        if "Rewrite the following" in prompt:                     # GENERATOR (counterfactual)
            return self._counterfactual(prompt)
        return self._judge(prompt)                                # JUDGE (incl. oracle)

    def generate_batch(self, prompts: List[str], system: Optional[str] = None,
                       max_tokens: int = 600, validate=None,
                       temperature: Optional[float] = None) -> List[str]:
        return [self.generate(p, system, max_tokens, validate, temperature) for p in prompts]

    # -- per-role logic ------------------------------------------------------------------
    def _judge(self, prompt: str) -> str:
        fence = _FENCE.search(prompt)
        text = fence.group(1) if fence else prompt
        rubric_region = prompt.split("RUBRIC:", 1)[-1].split("```", 1)[0]
        score = cues.planted_score(rubric_region, text)
        return json.dumps({"score": round(score, 3), "applicable": True})

    def _reconstruct(self, prompt: str) -> str:
        pairs = _SCORE_PAIR.findall(prompt)
        high: Set[str] = set()
        if pairs:
            scores = [float(s) for s, _ in pairs]
            thr = max(scores) if max(scores) > 0 else 0.0
            for s, txt in pairs:
                if float(s) >= thr and thr > 0:
                    high |= cues.fires(txt)
        cats = sorted(high) or ["DREAD"]
        names = ", ".join(c.lower() for c in cats)
        rule = (f"the evaluator gives higher scores to stories that show {names} cues; "
                f"calm, ordinary scenes score 0")
        return json.dumps({"rule": rule, "rubric": _render_rubric(cats)})

    def _grade(self, prompt: str) -> str:
        rule_b = prompt.split("RULE B:", 1)[-1].split("Respond ONLY", 1)[0]
        cov = cues.coverage(rule_b)
        match = 1 + round(4 * len(cov) / len(cues.CATEGORIES))
        return json.dumps({"match": int(max(1, min(5, match))),
                           "difference": "cue coverage" if len(cov) < 5 else "same criterion"})

    def _counterfactual(self, prompt: str) -> str:
        fence = _FENCE.search(prompt)
        text = (fence.group(1) if fence else "").strip()
        if "UNRELATED aspect" in prompt:                  # suspect: change something neutral
            return text + " It was an ordinary Tuesday."
        if "LACKS the property" in prompt:                # lower scariness: strip all cues
            return _strip_scary(text)
        if "EXHIBITS the property" in prompt:             # raise scariness: add a new cue
            return _inject(text, _pick_new_category(text))
        return text

    def _advance_rubric(self, prompt: str) -> str:
        """Generic proposal for EvoPrompt/ProTeGi/APE: cover one more cue category than the
        rubric(s) embedded in the prompt. The canonical description is stripped first so the
        proposal reflects the CURRENT rubric's coverage, not the full construct definition."""
        p = prompt.replace(SM.CANONICAL_DESCRIPTION, "")
        cov = cues.coverage(p)
        nxt = [c for c in cues.CATEGORIES if c not in cov]
        cats = sorted(cov | ({nxt[0]} if nxt else set())) or ["DREAD", "SOUND"]
        return json.dumps({"rubric": _render_rubric(cats)})

    def _revise(self, prompt: str) -> str:
        body = prompt.split("CURRENT RUBRIC:", 1)[-1].split("MEASURED PROBLEMS", 1)[0]
        cov = cues.coverage(body)
        uncovered = [c for c in cues.CATEGORIES if c not in cov]
        if not uncovered:                                  # already names every cue: converged
            return json.dumps({"operator": "CLARIFY", "rubric": body.strip(),
                               "rationale": "all cue categories already articulated"})
        add = uncovered[0]
        operator = "ANCHOR" if len(cov) <= 1 else ("DECOMPOSE" if len(cov) >= 3 else "EDGE")
        new_cats = sorted(cov | {add})
        return json.dumps({
            "operator": operator,
            "rubric": _render_rubric(new_cats),
            "rationale": f"articulate the {add.lower()} cue the rubric was missing"})


def scary_roles() -> Roles:
    """All eight roles served by the deterministic planted backend (zero spend)."""
    f = ScaryFakeBackend
    return Roles(judge=f(role="judge"), reviser=f(role="reviser"),
                 reconstructor=f(role="reconstructor"),
                 acceptance_reconstructor=f(role="acceptance_reconstructor"),
                 generator=f(role="generator"),
                 acceptance_generator=f(role="acceptance_generator"),
                 grader=f(role="grader"), oracle=f(role="oracle"))
