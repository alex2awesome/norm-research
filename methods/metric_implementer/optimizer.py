"""GEPA-style improvement loop over EITHER kind of artifact (prompt or code).

Fidelity-only objective (the scalarization in measures.py cannot see predictive
performance). The reviser reads the scorecard plus *textual* failure artifacts —
reconstruction mismatch (names the misreading), missed counterfactual pairs, retest flips,
invariance violations, and (cross-kind) the code<->judge disagreement queue — and proposes
mutations. Counterfactual edits are regenerated fresh for every scoring pass (anti-Goodhart).
Acceptance runs once on cross-family roles. Every candidate is persisted to the Registry
with its lineage, complexity, and the data budget visible when it was produced.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .artifact import MetricArtifact
from .backends import Roles, parse_json_obj
from .config import BudgetCaps
from .measures import compute_scorecard, fidelity_scalar
from .registry import Registry

_OPERATORS = ["CLARIFY", "FEWSHOT+", "ANCHOR", "EDGE", "PRUNE", "DECOMPOSE", "MECHANIZE"]

_ATTRIBUTE_PROMPT = """A rubric is executed by a WEAKER model than you. For each failure \
below, decide: could a careful but limited reader score it correctly from the rubric alone \
(then the failure is the rubric's fault: AMBIGUOUS_PROMPT), or does correct scoring require \
judgment the rubric cannot transmit to a weak reader as written (JUDGE_LIMITATION)?

RUBRIC:
{body}

FAILURES:
{failures}

Respond ONLY with JSON: {{"attributions": ["AMBIGUOUS_PROMPT"|"JUDGE_LIMITATION", ...] \
(one per failure, in order)}}"""

_REVISE_PROMPT_KIND = """You are improving the RUBRIC of an automated evaluator (an LLM \
judge that scores one competitive-programming solution at a time in [0,1]).

CURRENT RUBRIC:
{body}

MEASURED PROBLEMS (from validity testing — these are real failures, not hypotheticals):
{failures}

Produce ONE revised rubric that fixes the most damaging problem. Pick exactly one mutation \
operator and apply it:
- CLARIFY: sharpen the definition against the confusion named above
- FEWSHOT+: add ONE worked example (use a failed counterfactual pair if shown: it has a \
known correct answer)
- ANCHOR: define precisely what scores of 0.0, 0.5, 1.0 mean
- EDGE: add an explicit edge-case clause for an item the judge flip-flopped on
- PRUNE: remove text that adds instability without adding meaning
- DECOMPOSE: split into 2-3 sub-questions plus an explicit aggregation rule
- MECHANIZE: rewrite judgment-language as explicit step-by-step checks a much weaker \
reader could follow mechanically, with an explicit aggregation rule

The rubric will be executed by a WEAKER model than you: failures tagged JUDGE_LIMITATION \
need DECOMPOSE or MECHANIZE (transmit less judgment, more procedure); failures tagged \
AMBIGUOUS_PROMPT need CLARIFY, ANCHOR, or EDGE.

HARD CONSTRAINT: the revised rubric must stay within {token_cap} tokens (~{word_cap} words) \
and at most {fewshot_cap} worked examples.

Respond ONLY with JSON: {{"operator": "<one of {ops}>", "rubric": "<the full revised \
rubric>", "rationale": "<one sentence>"}}"""

_REVISE_CODE_KIND = """You are improving a Python implementation of a metric over \
competitive-programming solutions. It must expose score(text) -> float in [0,1] or None \
(not applicable). Standard library only. It receives the raw solution source as `text`.

METRIC INTENT: {description}

CURRENT IMPLEMENTATION:
```python
{body}
```

MEASURED PROBLEMS (real failures from validity testing):
{failures}

Produce ONE revised implementation fixing the most damaging problem (e.g., the items where \
this code most disagrees with a careful reading of the intent). Keep it deterministic and \
fast. HARD CONSTRAINT: at most {ast_cap} AST nodes (roughly {loc_cap} lines).

Respond ONLY with JSON: {{"operator": "CODE_REVISE", "code": "<the full revised module>", \
"rationale": "<one sentence>"}}"""


def _format_failures(card: dict, convergence_queue: Optional[List[dict]] = None) -> str:
    out = []
    rec = card.get("reconstruction", {})
    if rec.get("induced_rule") and np.isfinite(rec.get("semantic", np.nan)) \
            and rec["semantic"] < 0.99:
        out.append(f"- RECONSTRUCTION: an independent observer watching this evaluator's "
                   f"scores described its rule as: \"{rec['induced_rule']}\" "
                   f"(intended rule differs: {rec.get('difference', '')})")
    for f in card.get("counterfactual", {}).get("failures", [])[:3]:
        out.append(f"- COUNTERFACTUAL MISS [{f['cell']}]: after an edit that should have "
                   f"moved the score in a known direction, delta={f['delta']:+.2f}. "
                   f"Edited excerpt:\n  {f['edited_excerpt'][:300]}")
    for f in card.get("reliability", {}).get("retest_flips", [])[:2]:
        out.append(f"- RETEST FLIP: item #{f['idx']} scored differently across passes "
                   f"(delta={f['delta']:.2f}) — the rubric underdetermines it.")
    for v in card.get("consistency", {}).get("violations", [])[:2]:
        out.append(f"- INVARIANCE VIOLATION: a {v['invariance']} change (which should not "
                   f"matter) moved the score by {v['delta']:.2f}.")
    for q in (convergence_queue or [])[:3]:
        out.append(f"- CODE<->JUDGE DISAGREEMENT: judge={q['judge']:.2f} vs "
                   f"code={q['code']:.2f} on:\n  {q['text_excerpt'][:300]}")
    return "\n".join(out) if out else "- (no failures recorded; tighten precision anyway)"


@dataclass
class Candidate:
    artifact: MetricArtifact
    version_id: str
    scorecard: Optional[dict] = None
    fidelity: float = float("nan")
    operator: str = "INIT"


def _attribute_failures(cand: Candidate, roles: Roles, failures: str) -> str:
    """Strong-reviser diagnosis: is each failure the prompt's fault (AMBIGUOUS_PROMPT) or
    beyond the weak judge's capability as written (JUDGE_LIMITATION)? Routes the operator
    choice; JUDGE_LIMITATION -> DECOMPOSE/MECHANIZE."""
    if cand.artifact.kind != "prompt" or "COUNTERFACTUAL MISS" not in failures:
        return failures
    raw = roles.reviser.generate(
        _ATTRIBUTE_PROMPT.format(body=cand.artifact.body, failures=failures),
        max_tokens=200, validate=lambda s: parse_json_obj(s) is not None)
    obj = parse_json_obj(raw) or {}
    attrs = obj.get("attributions") or []
    if not attrs:
        return failures
    tagged, k = [], 0
    for line in failures.split("\n- "):
        if line.startswith("COUNTERFACTUAL MISS") and k < len(attrs):
            line = f"[{attrs[k]}] " + line
            k += 1
        tagged.append(line)
    return "\n- ".join(tagged)


def _mutate(cand: Candidate, roles: Roles, cfg, caps: BudgetCaps,
            convergence_queue: Optional[List[dict]]
            ) -> Tuple[List[Tuple[str, MetricArtifact, str]], str]:
    """Returns (mutations, attributed_failures): each mutation carries the reviser's own
    one-sentence rationale, and the attribution-tagged failure text is returned so the
    caller can persist the full decision trail (operator choice is not a black box)."""
    failures = _format_failures(cand.scorecard or {}, convergence_queue)
    failures = _attribute_failures(cand, roles, failures)
    art = cand.artifact
    out = []
    if art.kind == "prompt":
        token_cap = caps.instruction_tokens or 100000
        prompt = _REVISE_PROMPT_KIND.format(
            body=art.body, failures=failures, token_cap=token_cap,
            word_cap=int(token_cap * 0.75),
            fewshot_cap=caps.n_fewshots if caps.n_fewshots is not None else 8,
            ops=_OPERATORS)
        key, field_ = "rubric", "rubric"
    else:
        ast_cap = caps.code_ast_nodes or 100000
        prompt = _REVISE_CODE_KIND.format(
            description=art.description, body=art.body, failures=failures,
            ast_cap=ast_cap, loc_cap=max(10, ast_cap // 8))
        key, field_ = "code", "code"
    raws = roles.reviser.generate_batch([prompt] * cfg.n_mutations, max_tokens=1800,
                                        validate=lambda s: bool(
                                            (parse_json_obj(s) or {}).get(key)))
    for raw in raws:
        obj = parse_json_obj(raw)
        if not obj or not obj.get(field_):
            continue
        body = str(obj[field_])
        out.append((str(obj.get("operator", "?")),
                    MetricArtifact(metric_id=art.metric_id, kind=art.kind, body=body,
                                   name=art.name, description=art.description,
                                   invariances=art.invariances),
                    str(obj.get("rationale", ""))))
    return out, failures


def improve(
    seed_artifact: MetricArtifact, texts: List[str], roles: Roles, cfg,
    registry: Registry, *, caps: Optional[BudgetCaps] = None, rounds: Optional[int] = None,
    run_id: Optional[str] = None, data_ids: Optional[List[str]] = None,
    convergence_queue: Optional[List[dict]] = None, log=print,
) -> dict:
    """Budget-capped GEPA loop for one artifact. Returns the run summary (and persists
    everything: versions, scorecards, ledger events, acceptance verdict)."""
    caps = caps or BudgetCaps()
    rounds = rounds if rounds is not None else (caps.optimizer_rounds or cfg.default_rounds)
    run_id = run_id or f"improve_{seed_artifact.metric_id}_{int(time.time())}"
    rng = np.random.default_rng(cfg.random_seed)
    data_ids = data_ids or [str(i) for i in range(len(texts))]
    if caps.data_budget is not None and len(texts) > caps.data_budget:
        sel = rng.choice(len(texts), size=caps.data_budget, replace=False)
        texts = [texts[i] for i in sel]
        data_ids = [data_ids[i] for i in sel]
    models = {"judge": cfg.judge_model, "reviser": cfg.reviser_model,
              "reconstructor": cfg.reconstructor_model}

    registry.register_metric(seed_artifact.metric_id, seed_artifact.name,
                             seed_artifact.description)
    registry.log("RUN_STARTED", run_id=run_id, metric_id=seed_artifact.metric_id,
                 kind=seed_artifact.kind, caps=caps.as_dict(), rounds=rounds,
                 n_texts=len(texts))

    def persist(art, operator, parent, rnd, notes="") -> str:
        return registry.create_version(
            art, operator=operator, parent_version=parent, run_id=run_id,
            optimizer_round=rnd, data_budget_ids=data_ids,
            caps_in_force=caps.as_dict(), models=models, notes=notes)

    oracle_cache: Dict[str, tuple] = {}     # oracle pass is candidate-independent

    def score(c: Candidate, rnd: int) -> None:
        c.scorecard = compute_scorecard(c.artifact, texts, roles, cfg, rng,
                                        oracle_cache=oracle_cache)
        c.fidelity = c.scorecard["fidelity_scalar"]
        registry.save_scorecard(c.artifact.metric_id, c.version_id, c.artifact.kind,
                                {**c.scorecard, "round": rnd, "run_id": run_id},
                                eval_config_hash=f"r{rnd}")
        log(f"  [{run_id} r{rnd}] {c.version_id} ({c.operator}) "
            f"fidelity={c.fidelity:.3f} "
            f"(cf={c.scorecard['counterfactual'].get('cf_validity', float('nan')):.2f} "
            f"recon={c.scorecard['reconstruction'].get('behavioral', float('nan')):.2f} "
            f"rho={c.scorecard['reliability'].get('rho', float('nan')):.2f})")

    seed_vid = registry.head(seed_artifact.metric_id, seed_artifact.kind)
    if seed_vid is None:
        seed_vid = persist(seed_artifact, "INIT", None, 0)
    seed = Candidate(seed_artifact, seed_vid, operator="INIT")
    score(seed, 0)
    pool = [seed]

    # Rank candidates promotable-first: a high-fidelity but unpromotable candidate (e.g.
    # reliability below floor) must not become the lineage parent — round 2 of the first
    # live run mutated exactly such a candidate and wasted the round.
    def rank_key(c: Candidate):
        promotable = bool((c.scorecard or {}).get("promotable", False))
        return (promotable, np.nan_to_num(c.fidelity, nan=-1))

    for rnd in range(1, rounds + 1):
        best = max(pool, key=rank_key)
        muts, attributed = _mutate(best, roles, cfg, caps, convergence_queue)
        registry.log("FAILURES_ATTRIBUTED", run_id=run_id, round=rnd,
                     parent=best.version_id, attributed_failures=attributed)
        fresh: List[Candidate] = []
        for op, art, rationale in muts:
            bad = art.violates(caps)
            if bad:
                registry.log("VERSION_REJECTED_BUDGET", run_id=run_id,
                             metric_id=art.metric_id, operator=op, exceeded=bad,
                             rationale=rationale, complexity=art.complexity())
                log(f"  [r{rnd}] mutation ({op}) rejected: exceeds {bad}")
                continue
            vid = persist(art, op, best.version_id, rnd,
                          notes=f"reviser_rationale: {rationale}")
            cand = Candidate(art, vid, operator=op)
            score(cand, rnd)        # fresh CF edits inside -> regenerated per scoring pass
            fresh.append(cand)
        pool = sorted(pool + fresh, key=rank_key, reverse=True)[: cfg.pool_keep_k]

    # acceptance: cross-family reconstructor + generator, fresh draws
    best = max(pool, key=rank_key)
    acc_best = compute_scorecard(best.artifact, texts, roles, cfg, rng, acceptance=True,
                                 oracle_cache=oracle_cache)
    acc_seed = compute_scorecard(seed.artifact, texts, roles, cfg, rng, acceptance=True,
                                 oracle_cache=oracle_cache)
    accepted = (best.version_id != seed.version_id
                and np.nan_to_num(acc_best["fidelity_scalar"], nan=-1)
                > np.nan_to_num(acc_seed["fidelity_scalar"], nan=-1)
                and acc_best.get("promotable", False))
    if accepted:
        # Prompts are optimized FOR this judge tier; HEAD is per-(kind, tier) so tiers
        # never overwrite each other's current-best lineage. Code kind is judge-free.
        tier = roles.judge.model if seed_artifact.kind == "prompt" else None
        registry.set_head(best.artifact.metric_id, best.artifact.kind,
                          best.version_id, run_id, judge_tier=tier)
    summary = {
        "run_id": run_id, "metric_id": seed_artifact.metric_id,
        "kind": seed_artifact.kind, "caps": caps.as_dict(), "rounds": rounds,
        "seed_version": seed.version_id, "best_version": best.version_id,
        "seed_fidelity_inloop": seed.fidelity, "best_fidelity_inloop": best.fidelity,
        "seed_fidelity_acceptance": acc_seed["fidelity_scalar"],
        "best_fidelity_acceptance": acc_best["fidelity_scalar"],
        "accepted": bool(accepted),
        "cost_usd_so_far": roles.total_cost(),
    }
    registry.log("RUN_FINISHED", **summary, role_stats=roles.stats())
    log(f"  [{run_id}] done: accepted={accepted} "
        f"seed={summary['seed_fidelity_acceptance']:.3f} -> "
        f"best={summary['best_fidelity_acceptance']:.3f} "
        f"(cost so far ${summary['cost_usd_so_far']:.3f})")
    return summary
