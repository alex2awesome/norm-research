"""Offline oracle: a deterministic stand-in for the LLM proposer and judge.

Lets the whole infilling loop run in CI with no model, while still exercising the *real*
contrast→feature→materialize→reinsert path:

- ``oracle_proposer`` reads the WRONG-positive vs WRONG-negative examples out of the proposer
  prompt and returns the single attribute that best separates them (which, thanks to the
  residualized contrast, is whichever tacit norm the known metrics missed). This mirrors what
  a real LLM does — articulate the distinguishing property — but deterministically.
- ``oracle_judge_scorer`` materializes a proposed feature by detecting its target attribute in
  each dossier (via the shared :func:`world.detect`).

A live run swaps these for ``make_proposer`` / ``make_vllm_judge_scorer`` (the real LLM reads
the prose and the synonym variety makes it a genuine articulation test). Each oracle rubric
also carries a ``[[oracle:attr=value]]`` tag so the oracle judge can score it deterministically;
a real judge simply ignores the tag and follows the prose rubric.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

import numpy as np

from methods.metrics_tree_infilling.io_metrics import MetricSpec
from . import world

_TAG = re.compile(r"\[\[oracle:(\w+)=(\w+)\]\]")


# --------------------------------------------------------------------------------------
# Proposer
# --------------------------------------------------------------------------------------

def _parse_prompt(prompt: str) -> Tuple[List[str], List[str]]:
    """Recover the (positives, negatives) example texts from the proposer prompt.

    Each example block runs from its header (``POSITIVES (label 1):`` / ``NEGATIVES (label
    0):``) to the next blank line, so parsing does NOT depend on the exact instruction text that
    follows the examples (the block end must not be coupled to a phrase like 'Identify exactly
    ONE', which the proposer prompt may legitimately reword)."""
    def block(header: str) -> List[str]:
        i = prompt.find(header)
        if i == -1:
            return []
        rest = prompt[i + len(header):]
        end = rest.find("\n\n")
        chunk = (rest if end == -1 else rest[:end]).strip()
        return [s.strip() for s in chunk.split("\n---\n") if s.strip()]
    pos = block("POSITIVES (label 1):\n")
    neg = block("NEGATIVES (label 0):\n")
    return pos, neg


def _best_separating_attr(pos: List[str], neg: List[str]) -> Optional[Tuple[str, str, float]]:
    best = None
    for attr, values in world.ATTRIBUTES.items():
        if attr in world.KNOWN_ATTRS:
            # the proposer is instructed to find a property NOT already covered by the
            # published Code; a real LLM honors that, so the oracle skips the known attributes.
            continue
        v0 = next(iter(values))
        p_pos = np.mean([world.detect(t, attr) == v0 for t in pos]) if pos else 0.0
        p_neg = np.mean([world.detect(t, attr) == v0 for t in neg]) if neg else 0.0
        sep = abs(p_pos - p_neg)
        value = v0 if p_pos >= p_neg else [v for v in values if v != v0][0]
        if best is None or sep > best[2]:
            best = (attr, value, sep)
    return best


def oracle_proposer(prompt: str) -> str:
    pos, neg = _parse_prompt(prompt)
    if not pos or not neg:
        return ""
    best = _best_separating_attr(pos, neg)
    if best is None or best[2] < 0.15:     # no attribute meaningfully separates them
        return ""
    attr, value, _ = best
    phrasings = world.ATTRIBUTES[attr][value][:3]
    gloss = "; or ".join(phrasings)
    return json.dumps({
        "name": f"{attr}_{value}",
        "description": f"Whether the creature {gloss}.",
        "rubric": (f"Return 1 if the creature is described as follows: {gloss}; "
                   f"otherwise return 0. [[oracle:{attr}={value}]]"),
    })


# --------------------------------------------------------------------------------------
# Judge scorer
# --------------------------------------------------------------------------------------

def _target_of(metric: MetricSpec) -> Optional[Tuple[str, str]]:
    m = _TAG.search(metric.guidance or "") or _TAG.search(metric.name or "")
    if m:
        return m.group(1), m.group(2)
    # fall back: infer from the rubric prose by scanning known phrasings
    text = f"{metric.name} {metric.guidance} {metric.description}".lower()
    for attr, values in world.ATTRIBUTES.items():
        for value, phrasings in values.items():
            if any(ph.lower() in text for ph in phrasings):
                return attr, value
    return None


def oracle_judge_scorer(metrics: List[MetricSpec], texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    n, M = len(texts), len(metrics)
    levels = np.full((n, M), np.nan)
    applicable = np.zeros((n, M), dtype=bool)
    for j, metric in enumerate(metrics):
        target = _target_of(metric)
        if target is None:
            continue
        attr, value = target
        for i, t in enumerate(texts):
            detected = world.detect(t, attr)
            if detected is not None:
                levels[i, j] = 1.0 if detected == value else 0.0
                applicable[i, j] = True
    return levels, applicable
