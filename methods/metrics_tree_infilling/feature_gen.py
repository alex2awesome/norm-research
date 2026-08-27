"""Generate the missing feature as a reusable scorer + distillation (spec §4).

The proposer is shown the WRONG positives/negatives (a residualized contrast) and the
descriptions of the known metrics, and asked for ONE distinguishing property that is *not*
already covered. It returns ``{name, description, rubric}``. The rubric is wrapped as a
judge :class:`~.io_metrics.MetricSpec` (the distilled, reproducible scorer): a frozen,
temperature-0 rubric applied per item — the same family as the existing judge metrics, but
now reproducible. We also estimate the scorer's reliability (test-retest agreement), which
feeds the redundancy check and the depth discount (spec §7).
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .contrast import Contrast
from .io_metrics import JudgeScorer, MetricSpec, _stable_id

# proposer(prompt) -> raw text
Proposer = Callable[[str], str]


@dataclass
class ProposedFeature:
    name: str
    description: str
    rubric: str
    raw: str = ""

    def to_metric(self) -> MetricSpec:
        # role="feature": a discovered feature enters the within-node model (X) but not the
        # splitting covariates (z), so the tree never fragments regions on its raw value.
        return MetricSpec(
            metric_id=_stable_id("new", self.name, self.description),
            name=self.name, description=self.description, kind="judge", guidance=self.rubric,
            role="feature",
        )


_PROMPT = """These items were labeled by experts. The POSITIVES were labeled 1 and the \
NEGATIVES labeled 0, but a set of known criteria FAILED to predict that for these particular \
items — so whatever separates them is NOT captured by the known criteria.

KNOWN CRITERIA — these are ALREADY MEASURED. Your answer MUST NOT be, re-name, re-state, paraphrase, \
or be a sub- or super-case of ANY of these:
{known}

POSITIVES (label 1):
{pos}

NEGATIVES (label 0):
{neg}

Propose up to {k} DISTINCT candidate properties that each distinguish the positives from the \
negatives AND are NOT captured by any known criterion above. Favor qualities the known criteria \
ignore — domain-specific practices, argument patterns, and community conventions particular to \
this kind of text, as well as aesthetic, tonal, or structural properties — anything visible in \
the text itself. Before listing each candidate, confirm it is not a restatement of a known criterion. \
Return ONLY a JSON object of the form:
{{"candidates": [{{"name": "<short name>", "description": "<one sentence>", "rubric": "<a \
scoring rubric precise enough to apply to any single item and return a number in [0,1]>"}}]}}"""


def make_proposer(cfg) -> Proposer:
    """A proposer backed by ``LLMClient`` (defaults to Claude for quality, per config)."""
    from verification_library.client import LLMClient

    if cfg.proposer_backend == "anthropic":
        client = LLMClient.from_anthropic(model=cfg.proposer_model, concurrency=cfg.llm_concurrency)
    else:
        import os
        client = LLMClient.from_openai_compatible(
            model=cfg.proposer_model,
            base_url=cfg.openai_base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
            concurrency=cfg.llm_concurrency,
        )

    def proposer(prompt: str) -> str:
        return asyncio.run(client.generate(prompt, max_tokens=900, temperature=cfg.llm_temperature))

    return proposer


def propose_feature(
    contrast: Contrast, known_metric_descriptions: List[str], cfg, proposer: Proposer,
    max_examples: Optional[int] = None,
) -> Optional[ProposedFeature]:
    """Run the proposer on a residualized contrast; parse + de-duplicate vs known criteria.

    Asks for up to ``cfg.proposer_k_candidates`` (default 4) distinct candidate properties that
    are NOT covered by the known criteria, then returns the first candidate that is not
    token-redundant with a known criterion. Falls back to the first candidate if all are
    redundant — the post-materialization redundancy guard (loop, on scored levels) is the
    precise backstop. This fixes the observed failure mode (a live proposer re-deriving a known
    criterion like "diet / herbivore vs carnivore" instead of the tacit norm).

    Accepts either the multi-candidate ``{"candidates":[...]}`` form or a legacy single-object
    ``{"name",...}`` (e.g. the offline oracle proposer), so the no-LLM tests are unaffected.
    """
    if max_examples is None:
        max_examples = int(getattr(cfg, "proposer_max_examples", 6))
    pos = contrast.wrong_pos[:max_examples]
    neg = contrast.wrong_neg[:max_examples]
    if not pos or not neg:
        return None
    known = "\n".join(f"- {d}" for d in known_metric_descriptions) or "(none)"
    k = int(getattr(cfg, "proposer_k_candidates", 4))
    prompt = _PROMPT.format(
        known=known,
        pos="\n---\n".join(pos),
        neg="\n---\n".join(neg),
        k=k,
    )
    try:
        raw = proposer(prompt)
    except Exception as e:                       # transient API failure -> no proposal, not a crash
        _log_proposer_error(e)
        return None
    cands = _parse_json_candidates(raw)
    if not cands:
        return None
    known_sets = [_content_tokens(d) for d in known_metric_descriptions]
    chosen = next((c for c in cands if not _redundant_with_known(c, known_sets)), None)
    if chosen is None:
        chosen = cands[0]
    if not str(chosen.get("name", "")).strip() or not str(chosen.get("rubric", "")).strip():
        return None
    return ProposedFeature(
        name=str(chosen["name"]).strip(),
        description=str(chosen.get("description", "")).strip(),
        rubric=str(chosen["rubric"]).strip(),
        raw=raw,
    )


@dataclass
class ProposedComposite:
    """A composite feature: two primitives combined by a boolean rule (§9 fix).

    The ``rule`` is a *candidate* from the proposer; the loop re-fits the best rule on data
    (``interactions.best_combination``) before reinsertion, so a wrong stated rule is corrected.
    """

    primitives: List[ProposedFeature]
    rule: Optional[str] = None
    raw: str = ""

    def to_metric(self) -> MetricSpec:
        names = " / ".join(p.name for p in self.primitives)
        rule = self.rule or "best-fit"
        return MetricSpec(
            metric_id=_stable_id("cmp", names, rule),
            name=f"composite[{rule}]({names})", kind="code",
            description=f"boolean {rule} of [{names}]",
            code_fn=lambda t: 0.0,          # placeholder; loop materializes via the primitives
            role="feature",
        )


_COMPOSITE_PROMPT = """These items were labeled by experts. The POSITIVES (label 1) and NEGATIVES
(label 0) below are items a set of known criteria FAILED to predict, so whatever separates them is
NOT captured by the known criteria. Importantly, NO single property may distinguish them on its
own — it may take a COMBINATION of two properties (A-but-not-B, A-and-B, exactly-one-of-A/B).

KNOWN CRITERIA — already measured; your primitives MUST NOT restate these:
{known}

POSITIVES (label 1):
{pos}

NEGATIVES (label 0):
{neg}

Name exactly TWO distinct primitive properties (each visible in the text, not a known criterion)
and a boolean rule combining them that separates positives from negatives. Return ONLY JSON:
{{"primitives": [{{"name": "<name>", "description": "<one sentence>", "rubric": "<scoring rubric>"}}, {{"name": "...", "description": "...", "rubric": "..."}}],
  "rule": "and | or | xor | a_not_b | b_not_a"}}"""


def propose_composite_feature(
    contrast: Contrast, known_metric_descriptions: List[str], cfg, proposer: Proposer,
    max_examples: Optional[int] = None,
) -> Optional[ProposedComposite]:
    """Propose a 2-primitive composite for a gap no single feature closes (§9 / root-XOR case)."""
    if max_examples is None:
        max_examples = int(getattr(cfg, "proposer_max_examples", 6))
    pos = contrast.wrong_pos[:max_examples]
    neg = contrast.wrong_neg[:max_examples]
    if not pos or not neg:
        return None
    known = "\n".join(f"- {d}" for d in known_metric_descriptions) or "(none)"
    prompt = _COMPOSITE_PROMPT.format(
        known=known, pos="\n---\n".join(pos), neg="\n---\n".join(neg))
    try:
        return _parse_composite(proposer(prompt))
    except Exception as e:                       # transient API failure -> no composite, not a crash
        _log_proposer_error(e)
        return None


def _parse_composite(resp: Optional[str]) -> Optional[ProposedComposite]:
    if not resp:
        return None
    s = resp.strip()
    lo, hi = s.find("{"), s.rfind("}")
    if lo == -1 or hi == -1 or hi <= lo:
        return None
    try:
        obj = json.loads(s[lo:hi + 1])
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    prims = obj.get("primitives") if isinstance(obj.get("primitives"), list) else None
    if not prims:
        return None
    feats: List[ProposedFeature] = []
    for p in prims:
        if not isinstance(p, dict):
            continue
        name, rubric = str(p.get("name", "")).strip(), str(p.get("rubric", "")).strip()
        if name and rubric:
            feats.append(ProposedFeature(
                name=name, description=str(p.get("description", "")).strip(), rubric=rubric))
    if len(feats) < 2:
        return None
    rule = str(obj.get("rule", "")).strip().lower() or None
    if rule not in ("and", "or", "xor", "a_not_b", "b_not_a"):
        rule = None
    return ProposedComposite(primitives=feats[:2], rule=rule, raw=resp)


def estimate_reliability(
    metric: MetricSpec, texts: List[str], judge_scorer: JudgeScorer,
    sample_size: int = 100, rng: Optional[np.random.Generator] = None,
) -> float:
    """Test-retest agreement of the distilled scorer on a held-out sample (spec §4).

    Scores the sample twice and returns Pearson correlation of the levels among items
    applicable in both passes (1.0 == perfectly reproducible). Used as a reliability discount
    in :mod:`guards`.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if len(texts) == 0:
        return 1.0
    sel = rng.choice(len(texts), size=min(sample_size, len(texts)), replace=False)
    sample = [texts[i] for i in sel]
    lv1, ap1 = judge_scorer([metric], sample)
    lv2, ap2 = judge_scorer([metric], sample)
    both = ap1[:, 0] & ap2[:, 0]
    if both.sum() < 5:
        return 1.0
    a, b = lv1[both, 0], lv2[both, 0]
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float(np.mean(np.abs(a - b) < 1e-6))
    return float(np.clip(np.corrcoef(a, b)[0, 1], 0.0, 1.0))


_proposer_err_count = 0


def _log_proposer_error(e: Exception, limit: int = 6) -> None:
    """Throttled notice that a proposer API call failed (and was skipped, not crashed)."""
    global _proposer_err_count
    _proposer_err_count += 1
    if _proposer_err_count <= limit:
        print(f"[proposer] API failure ({type(e).__name__}: {str(e)[:90]}); skipping this proposal",
              flush=True)


def _parse_json_object(resp: Optional[str]) -> Optional[dict]:
    if not resp:
        return None
    s = resp.strip()
    lo, hi = s.find("{"), s.rfind("}")
    if lo == -1 or hi == -1 or hi <= lo:
        return None
    try:
        obj = json.loads(s[lo:hi + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _parse_json_candidates(resp: Optional[str]) -> List[dict]:
    """Parse a proposer response into a list of candidate dicts.

    Accepts ``{"candidates":[...]}`` or a legacy single ``{"name":...}`` object (the offline
    oracle proposer emits the single-object form).
    """
    if not resp:
        return []
    s = resp.strip()
    # strip markdown fences, then try a top-level ARRAY envelope before the object envelope —
    # the offline 70B proposer often returns `[{...},{...}]`, and slicing first-{ to last-}
    # mangles that into invalid JSON (the residual-arm 0-proposals bug, 2026-07-05)
    s = re.sub(r"```(?:json)?", "", s).strip()
    obj = None
    a_lo, a_hi = s.find("["), s.rfind("]")
    o_lo, o_hi = s.find("{"), s.rfind("}")
    for lo, hi in [(a_lo, a_hi), (o_lo, o_hi)]:
        if lo != -1 and hi != -1 and hi > lo:
            try:
                obj = json.loads(s[lo:hi + 1])
                break
            except Exception:
                obj = None
    if obj is None:
        return []
    if isinstance(obj, dict):
        cands = obj.get("candidates") if isinstance(obj.get("candidates"), list) else None
        if cands is not None:
            return [c for c in cands if isinstance(c, dict)]
        if any(k in obj for k in ("name", "rubric")):
            return [obj]
        return []
    if isinstance(obj, list):
        return [c for c in obj if isinstance(c, dict)]
    return []


_STOP = set("""a an the of to in on at for and or but is are was were be been being this that
these those it its their his her our your you we they he she them us i as with from by into
than then so such not no yes if when while which who whom whose what item items text creature
creatures whether rather vs versus either neither both each all none most more less one two
property criterion criteria feature""".split())


def _content_tokens(text: str) -> set:
    return {w for w in re.sub(r"[^a-z0-9 ]", " ", (text or "").lower()).split()
            if len(w) > 2 and w not in _STOP}


def _redundant_with_known(cand: dict, known_sets: List[set], threshold: float = 0.6) -> bool:
    """Coarse token-containment gate: is this candidate a restatement of a known criterion?

    max over known criteria of (candidate content-tokens also in that criterion) / |candidate
    tokens|. Dependency-free; intentionally conservative so it does not reject genuine tacit
    norms (e.g. "glow"/"song") that merely share generic words. The post-materialization
    redundancy guard in the loop (on scored levels) is the precise backstop.
    """
    cand_tokens = _content_tokens(f"{cand.get('name', '')} {cand.get('description', '')}")
    if not cand_tokens or not known_sets:
        return False
    best = 0.0
    for ks in known_sets:
        if not ks:
            continue
        best = max(best, len(cand_tokens & ks) / len(cand_tokens))
    return best >= threshold
