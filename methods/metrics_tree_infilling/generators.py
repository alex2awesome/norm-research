"""Proposal generators — the arms of the metric-discovery comparison.

Each generator produces candidate rubrics; ALL candidates flow through the SAME acceptance
gate (`global_infill.run_global_infill` via `proposal_fn`), so arms are comparable on
accepted-bits-per-proposal, and each arm doubles as a capture-recapture list for the flux
upper bound (MCC §3 Wrap 2 / §9.1 — a single arm gives an anti-conservatively low flux).

Arms:
  residual        — the engine's own: WRONG/RIGHT residual contrast (targeted; default)
  unconditional   — autorubric-style: sees ONLY the existing metric descriptions, proposes what
                    else evaluators of this genre care about (no items, no labels, no residual)
  label_contrast  — example-grounded autorubric: random pos vs neg items by RAW LABEL (not
                    residual) — what naive rubric-mining from data looks like
  autometrics_iterative — port of methods/autometrics iterative_refinement: failure-PAIR
                    conditioning + iteration memory + self-critique filter (superficial vs
                    substantive) + hash-dedup. Prompt language from ContrastiveRubricSignature.
  metric_tree     — port of methods/metric_tree discriminative gap-fill: partition the corpus
                    by the label-informative bank metrics, propose metrics WITHIN mixed-base-
                    rate cells ("all these look identical to the bank, yet some pass and some
                    fail"). Prompt language from PartitionMetricProposer._build_discriminative_prompt.

The two ports keep the EXECUTOR fixed (the shared ``proposer(prompt)`` callable): an arm is a
conditioning strategy, not a model — running each method's own LLM backend would confound
generator strategy with executor family, and the certificate is executor-indexed (A*_E).
Both ports force binary yes/no rubrics (metric_tree does the same binary-forcing to the same
base proposer in-source); semantic dedup beyond the hash/blocklist is delegated to the gate's
redundancy-R^2 guard, which measures it rather than asking an LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from .contrast import Contrast
from .feature_gen import ProposedFeature, propose_feature


# Anti-surface instruction appended to every arm's prompt under content_only_guard. Surface
# features (format/length/markdown/pronoun/digit counts) leak venue/virality artifacts and, at
# thin signal scale, out-compete real content metrics — "Manual Markdown Formatting" won on
# peer-review. Content = a property of what the text SAYS or DOES, not how it looks.
CONTENT_ONLY_INSTRUCTION = (
    "\n\nIMPORTANT — CONTENT ONLY: propose a property of WHAT THE TEXT SAYS OR DOES (its claims, "
    "reasoning, narrative, evidence, craft). Do NOT propose surface features: text length, word/"
    "sentence/paragraph counts, markdown or formatting, presence of code blocks or headers, "
    "capitalization, punctuation counts, readability scores, or the mere presence of digits/"
    "pronouns. A reader should not be able to score it from formatting alone.")

# HARD surface tokens: unambiguously about form/format — always surface, no content override
# (e.g. "markdown characters" must not be rescued by the fiction sense of "character").
_HARD_SURFACE_RE = re.compile(
    r"\bmarkdown\b|\bformatting\b|\bword count\b|\bcharacter count\b|\btext length\b|"
    r"\breadability (score|index)\b|\bwhitespace\b|\bline break|\ball[- ]caps\b|"
    r"\bbullet( point)?s?\b|\bheaders?\b|\bheadings?\b|"
    r"\bnumber of (words|sentences|paragraphs|characters|lines)\b|\bcode block", re.I)
# SOFT surface tokens: surface UNLESS a genuine content word co-occurs (a metric can mention
# "paragraph" or "punctuation" while being about content).
_SOFT_SURFACE_RE = re.compile(
    r"\bsentence length\b|\bparagraph(s)? (count|number)\b|\bcapitaliz|\bpunctuation\b|"
    r"\bexclamation (mark|point)|\bfirst[- ]person pronoun|\bpresence of (a )?(digit|number)s?\b",
    re.I)
_CONTENT_OVERRIDE_RE = re.compile(
    r"\bargument|evidence|claim|reason|narrat|\bplot\b|theme|dialog|imager|metaphor|"
    r"novel|contribut|method|analys|proof|rigor|coheren|persuas|insight|original|voice|tone", re.I)


def is_surface_only(name: str, rubric: str) -> bool:
    """True when a proposal is defined by a surface/format property with no content anchor."""
    blob = f"{name} {rubric}"
    if _HARD_SURFACE_RE.search(blob):
        return True
    return bool(_SOFT_SURFACE_RE.search(blob)) and not _CONTENT_OVERRIDE_RE.search(blob)


def _guard(prompt: str, cfg) -> str:
    """Append the content-only instruction when the guard is on."""
    return prompt + (CONTENT_ONLY_INSTRUCTION if getattr(cfg, "content_only_guard", False) else "")


def _drop_surface(props: List["Proposal"], cfg) -> List["Proposal"]:
    """Post-filter: remove surface-only proposals when the guard is on (belt-and-suspenders to
    the prompt instruction — the acceptance gate also carries the drop, but filtering here keeps
    the flux capture-recapture list content-only)."""
    if not getattr(cfg, "content_only_guard", False):
        return props
    return [p for p in props if not is_surface_only(p.name, p.rubric)]


@dataclass
class Proposal:
    name: str
    description: str
    rubric: str
    generator: str
    n_examples: int = 0


def _parse_candidates(resp: Optional[str]) -> List[dict]:
    if not resp:
        return []
    s = resp.strip()
    lo, hi = s.find("{"), s.rfind("}")
    if lo < 0 or hi <= lo:
        return []
    try:
        doc = json.loads(s[lo:hi + 1])
    except Exception:
        return []
    cands = doc.get("candidates") if isinstance(doc, dict) else None
    if isinstance(cands, list):
        return [c for c in cands if isinstance(c, dict) and c.get("name")]
    if isinstance(doc, dict) and doc.get("name"):
        return [doc]
    return []


# --------------------------------------------------------------------------------------
# Arm: unconditional (autorubric-style, no data)
# --------------------------------------------------------------------------------------

_UNCONDITIONAL_PROMPT = """You are extending an evaluation rubric bank for the task: {task_hint}.

Known criteria already in the bank:
{known}

Propose {k} NEW evaluation criteria that expert evaluators of this kind of work plausibly use
but that are NOT covered by any known criterion above. Each must be a concrete, checkable
YES/NO property of a single text (not a comparison, not a score). Return JSON only:
{{"candidates": [{{"name": "...", "description": "...", "rubric": "YES if ... ; NO otherwise."}}]}}"""


def unconditional_generator(task_hint: str, k: int = 4) -> Callable:
    """No items, no labels, no residual — the pure-prior baseline."""

    def gen(contrast, known_descriptions: List[str], cfg, proposer) -> List[Proposal]:
        prompt = _UNCONDITIONAL_PROMPT.format(
            task_hint=task_hint,
            known="\n".join(f"- {d}" for d in known_descriptions[-80:]) or "(none)", k=k)
        out = []
        for c in _parse_candidates(proposer(_guard(prompt, cfg)))[:k]:
            out.append(Proposal(name=str(c.get("name", "")).strip(),
                                description=str(c.get("description", "")).strip(),
                                rubric=str(c.get("rubric", "")).strip() or str(c.get("description", "")),
                                generator="unconditional", n_examples=0))
        return _drop_surface([p for p in out if p.name and p.rubric], cfg)

    return gen


# --------------------------------------------------------------------------------------
# Arm: label-contrast (example-grounded autorubric — raw labels, not residuals)
# --------------------------------------------------------------------------------------

_LABEL_PROMPT = """Below are text excerpts the community ACCEPTED and excerpts it REJECTED.

Known criteria (do NOT re-propose these):
{known}

ACCEPTED:
{pos}

REJECTED:
{neg}

Propose ONE new evaluation criterion, not covered by the known criteria, that best separates
the accepted from the rejected excerpts. It must be a concrete YES/NO property of a single
text. Return JSON only:
{{"candidates": [{{"name": "...", "description": "...", "rubric": "YES if ...; NO otherwise."}}]}}"""


def label_contrast_generator(texts: Sequence[str], y: np.ndarray, seed: int = 0) -> Callable:
    """Random accepted-vs-rejected examples by RAW label. Differs from the residual arm in one
    bit only — conditioning on y instead of |y - p_bank| — so the comparison isolates the value
    of residual targeting."""
    rng = np.random.default_rng(seed)

    def gen(contrast, known_descriptions: List[str], cfg, proposer) -> List[Proposal]:
        k = int(getattr(cfg, "proposer_max_examples", 6))
        max_chars = int(getattr(cfg, "contrast_max_chars", 4000))
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        if len(pos_idx) < 2 or len(neg_idx) < 2:
            return []
        pos = [str(texts[i])[:max_chars] for i in rng.choice(pos_idx, min(k, len(pos_idx)), replace=False)]
        neg = [str(texts[i])[:max_chars] for i in rng.choice(neg_idx, min(k, len(neg_idx)), replace=False)]
        prompt = _LABEL_PROMPT.format(
            known="\n".join(f"- {d}" for d in known_descriptions[-80:]) or "(none)",
            pos="\n---\n".join(pos), neg="\n---\n".join(neg))
        cands = _parse_candidates(proposer(_guard(prompt, cfg)))[:1]
        return _drop_surface([Proposal(name=str(c.get("name", "")).strip(),
                         description=str(c.get("description", "")).strip(),
                         rubric=str(c.get("rubric", "")).strip() or str(c.get("description", "")),
                         generator="label_contrast", n_examples=len(pos) + len(neg))
                for c in cands if c.get("name")], cfg)

    return gen


# --------------------------------------------------------------------------------------
# Arm: residual (wraps the engine's own proposer into the common interface)
# --------------------------------------------------------------------------------------

def residual_generator() -> Callable:
    def gen(contrast: Optional[Contrast], known_descriptions: List[str], cfg, proposer) -> List[Proposal]:
        if contrast is None:
            return []
        pf: Optional[ProposedFeature] = propose_feature(contrast, known_descriptions, cfg, proposer)
        if pf is None:
            return []
        n_ex = (min(len(contrast.wrong_pos), cfg.proposer_max_examples)
                + min(len(contrast.wrong_neg), cfg.proposer_max_examples))
        return [Proposal(name=pf.name, description=pf.description, rubric=pf.rubric,
                         generator="residual", n_examples=n_ex)]

    return gen


# --------------------------------------------------------------------------------------
# Arm: autometrics_iterative (port of methods/autometrics iterative_refinement)
# --------------------------------------------------------------------------------------

# Instruction text ported from ContrastiveRubricProposer.ContrastiveRubricSignature docstring
# (methods/autometrics/.../generator/ContrastiveRubricProposer.py) with binary forcing.
_AMI_PROMPT = """Propose metrics and rubrics that distinguish positive vs negative examples. \
Metrics must capture substantive, content-level distinctions (e.g. evidence quality, domain \
relevance, specificity of claims) rather than surface-level features (e.g. text length, \
formatting, readability, word choice). Every proposed metric must plausibly distinguish between \
items a domain expert would rate differently. Each metric must have a DETAILED rubric with \
specific, descriptive scoring criteria.

TASK: {task_hint}

CURRENT METRICS (already scored on every item — do NOT re-propose or reword these):
{known}
{memory}
CONTRASTIVE FAILURE PAIRS — matched pairs the current metrics score SIMILARLY yet the community
judged OPPOSITELY (POS = accepted, NEG = rejected). Whatever separates them is what the current
metrics miss:

{pairs}

Propose {k} NEW binary metrics that explain these failure pairs. Return JSON only — a list of
metric objects, each with keys {{"name", "rubric", "scale"}} where scale is "binary" and rubric
is a dict with keys "yes"/"no", each a 2-3 sentence description with concrete, observable
criteria. Metric names should be specific and descriptive (not generic like "Quality Score")."""

_AMI_CRITIQUE = """You proposed the following candidate evaluation metrics for the task: {task_hint}.

{cands}

For each, verdict whether it is "substantive" (captures a content-level distinction a domain
expert would recognize) or "superficial" (surface/formatting/length proxy, or so generic that
virtually every item satisfies it). Return JSON only:
{{"verdicts": ["substantive" | "superficial", ...]}} in the same order."""


def _parse_metric_list(resp: Optional[str]) -> List[dict]:
    """Parse the autometrics-style JSON list output (list | {"metrics": []} | {"candidates": []})."""
    if not resp:
        return []
    s = resp.strip()
    lo = s.find("[")
    if lo >= 0 and s.rfind("]") > lo:
        try:
            doc = json.loads(s[lo:s.rfind("]") + 1])
        except Exception:
            doc = None
        if isinstance(doc, list):
            return [c for c in doc if isinstance(c, dict) and c.get("name")]
    return _parse_candidates(resp)


def _rubric_text(c: dict) -> str:
    r = c.get("rubric")
    if isinstance(r, dict):
        yes = str(r.get("yes", "")).strip()
        no = str(r.get("no", "")).strip()
        if yes:
            return f"YES if {yes} NO otherwise" + (f": {no}" if no else ".")
    return str(r or c.get("description", "")).strip()


def autometrics_iterative_generator(task_hint: str, k: int = 4, self_critique: bool = True) -> Callable:
    """AutoMetrics-Iterative conditioning: failure pairs + iteration memory + self-critique.

    The gate's bank plays the role of the method's fitted metric head, so the residual
    ``contrast.pairs`` ARE its failure pairs (high-|residual| matched pos/neg). Iteration
    memory (this arm's own prior proposals) persists across rounds via closure — the port of
    the runner's trajectory summary + hash-dedup."""
    proposed_before: List[str] = []

    def gen(contrast, known_descriptions: List[str], cfg, proposer) -> List[Proposal]:
        if contrast is None or not contrast.pairs:
            return []
        max_chars = int(getattr(cfg, "contrast_max_chars", 4000)) // 2
        pairs = "\n\n".join(
            f"PAIR\nPOS: {str(p)[:max_chars]}\nNEG: {str(n)[:max_chars]}"
            for p, n in contrast.pairs)
        memory = ""
        if proposed_before:
            memory = ("\nPREVIOUSLY PROPOSED BY YOU (rejected or already tried — do not repeat, "
                      "reason about what they all MISSED and go elsewhere):\n"
                      + "\n".join(f"- {n}" for n in proposed_before[-20:]) + "\n")
        prompt = _AMI_PROMPT.format(
            task_hint=task_hint,
            known="\n".join(f"- {d}" for d in known_descriptions[-80:]) or "(none)",
            memory=memory, pairs=pairs, k=k)
        cands = _parse_metric_list(proposer(_guard(prompt, cfg)))[:k]
        seen = {n.lower() for n in proposed_before}
        cands = [c for c in cands if str(c.get("name", "")).strip().lower() not in seen]
        if cands and self_critique:
            listing = "\n".join(
                f"{i+1}. {c.get('name')}: {_rubric_text(c)[:300]}" for i, c in enumerate(cands))
            resp = proposer(_AMI_CRITIQUE.format(task_hint=task_hint, cands=listing))
            try:
                s = (resp or "").strip()
                verdicts = json.loads(s[s.find("{"):s.rfind("}") + 1]).get("verdicts", [])
                kept = [c for c, v in zip(cands, verdicts)
                        if "superficial" not in str(v).lower()]
                cands = kept or cands       # a garbled critique must not zero the round
            except Exception:
                pass
        out = []
        for c in cands:
            name = str(c.get("name", "")).strip()
            rub = _rubric_text(c)
            if name and rub:
                proposed_before.append(name)
                out.append(Proposal(name=name, description=str(c.get("description", name)).strip(),
                                    rubric=rub, generator="autometrics_iterative",
                                    n_examples=2 * len(contrast.pairs)))
        return out

    return gen


# --------------------------------------------------------------------------------------
# Arm: metric_tree (port of methods/metric_tree discriminative gap-fill)
# --------------------------------------------------------------------------------------

# Language ported from PartitionMetricProposer._build_discriminative_prompt
# (methods/metric_tree/proposer.py).
_MT_PROMPT = """{task_hint}

POPULATION: {n_cell:,} examples in this partition.
ACCEPTANCE RATE: {pos_rate:.0%} accepted, {neg_rate:.0%} rejected.
Why this matters: even after filtering through all our existing features, {neg_rate:.0%} of
items in this group are STILL rejected. Our existing features capture what these items have in
common, but they completely miss whatever causes {neg_rate:.0%} to fail. Your job is to find
THAT missing signal.

=== WHAT THIS PARTITION IS ===
{partition_context}

Given that ALL items in this group share the above characteristics, your features must be
SPECIFIC to this type of item. Generic quality criteria are useless — they score YES for 95%+
of items and fail to discriminate.

=== METRICS ALREADY IN USE (DO NOT RE-PROPOSE) ===
BLOCKLIST: {blocklist}
Also avoid synonyms, rewordings, or slight variations of the above.

ACCEPTED items in this partition:
{pos}

REJECTED items in this partition:
{neg}

Your goal: propose {k} binary (yes/no) features that distinguish accepted from rejected items
WITHIN this group that our existing features completely miss.

CRITICAL REQUIREMENTS:
  1. SPECIFICITY: write YES criteria that are HARD TO SATISFY — only 20-80% of items should
     qualify. If a mediocre item would trivially satisfy the criterion, it is too generic.
  2. PARTITION-SPECIFIC: what are the UNIQUE failure modes for this subpopulation, not
     generic quality issues?
  3. NO OVERLAP with the blocklist or existing features.

Return JSON only — a list of metric objects, each with keys {{"name", "rubric", "scale"}} where
scale is "binary" and rubric is a dict with keys "yes"/"no", each a detailed multi-sentence
description."""


def metric_tree_generator(task_hint: str, texts: Sequence[str], y: np.ndarray,
                          bank_levels: np.ndarray, metric_names: List[str],
                          k: int = 2, depth: int = 2, seed: int = 0) -> Callable:
    """Metric-tree conditioning: partition by the label-informative bank metrics, then run the
    discriminative gap-fill prompt inside mixed-base-rate cells (restructure._gap_fill port:
    cells with base rate in [0.2, 0.8] and >=2 of each class are the proposable ones).

    Partitioning stand-in for the tree's NA-aware greedy rebuild: binarize each bank metric at
    its mean, take the ``depth`` metrics with highest |corr(m, y)|, cells = joint assignments.
    Closure round-robins over viable cells across gate rounds."""
    rng = np.random.default_rng(seed)
    B = np.asarray(bank_levels, float)
    yv = np.asarray(y, float)
    cors = []
    for j in range(B.shape[1]):
        col = B[:, j]
        m = np.isfinite(col)
        if m.sum() < 20 or np.nanstd(col[m]) == 0 or np.std(yv[m]) == 0:
            cors.append(0.0); continue
        cors.append(abs(float(np.corrcoef(col[m], yv[m])[0, 1])))
    top = np.argsort(cors)[::-1][:depth]
    binar = np.zeros((len(yv), len(top)), int)
    for a, j in enumerate(top):
        col = B[:, j]
        binar[:, a] = (np.nan_to_num(col, nan=np.nanmean(col)) >= np.nanmean(col)).astype(int)
    cells = {}
    for i in range(len(yv)):
        cells.setdefault(tuple(binar[i]), []).append(i)
    viable = []
    for key, idx in cells.items():
        rate = yv[idx].mean()
        if 0.2 <= rate <= 0.8 and (yv[idx] == 1).sum() >= 2 and (yv[idx] == 0).sum() >= 2:
            viable.append((key, idx))
    state = {"cursor": 0}

    def gen(contrast, known_descriptions: List[str], cfg, proposer) -> List[Proposal]:
        if not viable:
            return []
        key, idx = viable[state["cursor"] % len(viable)]
        state["cursor"] += 1
        idx = np.array(idx)
        rate = float(yv[idx].mean())
        n_show = int(getattr(cfg, "proposer_max_examples", 6))
        max_chars = int(getattr(cfg, "contrast_max_chars", 4000))
        pos_i = idx[yv[idx] == 1]; neg_i = idx[yv[idx] == 0]
        pos = [str(texts[i])[:max_chars] for i in rng.choice(pos_i, min(n_show, len(pos_i)), replace=False)]
        neg = [str(texts[i])[:max_chars] for i in rng.choice(neg_i, min(n_show, len(neg_i)), replace=False)]
        pctx = "\n".join(
            f"- {metric_names[j]}: {'HIGH' if key[a] else 'LOW'}" for a, j in enumerate(top))
        prompt = _MT_PROMPT.format(
            task_hint=task_hint, n_cell=len(idx), pos_rate=rate, neg_rate=1 - rate,
            partition_context=pctx,
            blocklist=", ".join(f"'{d[:60]}'" for d in known_descriptions[-40:]) or "(none)",
            pos="\n---\n".join(pos), neg="\n---\n".join(neg), k=k)
        out = []
        for c in _parse_metric_list(proposer(_guard(prompt, cfg)))[:k]:
            name = str(c.get("name", "")).strip()
            rub = _rubric_text(c)
            if name and rub:
                out.append(Proposal(name=name, description=str(c.get("description", name)).strip(),
                                    rubric=rub, generator="metric_tree",
                                    n_examples=len(pos) + len(neg)))
        return _drop_surface(out, cfg)

    return gen


GENERATOR_FACTORIES = {
    "residual": lambda **kw: residual_generator(),
    "unconditional": lambda task_hint="this domain", k=4, **kw: unconditional_generator(task_hint, k),
    "label_contrast": lambda texts=None, y=None, seed=0, **kw: label_contrast_generator(texts, y, seed),
    "autometrics_iterative": lambda task_hint="this domain", k=4, **kw:
        autometrics_iterative_generator(task_hint, k),
    "metric_tree": lambda task_hint="this domain", texts=None, y=None, bank_levels=None,
        metric_names=None, k=2, seed=0, **kw:
        metric_tree_generator(task_hint, texts, y, bank_levels, metric_names, k=k, seed=seed),
}
