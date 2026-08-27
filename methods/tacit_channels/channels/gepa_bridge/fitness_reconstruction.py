"""C(c1) - GEPA with reconstruction-rho fitness: learned articulation as the RL channel.

The candidate's single component is an articulation text; fitness = how well a FROZEN
executor, prompted with that articulation, reconstructs the target's name-invoked judgment
vector on GEPA-dev items (disjoint from eval items - stable-hash split upstream).

This is the bridge that makes parts 1-2 and part 3 one pipeline: the optimizer PRODUCES the
explicit text the articulation program evaluates.

Wraps the pip `gepa` package exactly like m_omega_gepa.py (adapter pattern at its L347-380):
per-instance scores for GEPA's Pareto frontier, pool-level Spearman as the reported
objective, and the seed-only guard (gepa swallows backend errors with raise_on_exception=
False and silently returns the seed).

GPU (executor scoring) + API (reflection LM). Import of `gepa` is lazy.
"""
from __future__ import annotations

import json

import numpy as np

from methods.tacit_channels.channels.common import _rankdata, spearman

COMPONENT = "articulation"


def render_prompts(articulation: str, construct: str, texts: list[str], template: str,
                   max_text_chars: int) -> list[str]:
    rubric = f"{construct}\n\n{articulation}".strip()
    return [template.format(rubric=rubric, text=t[:max_text_chars]) for t in texts]


def per_instance_scores(pyes: np.ndarray, target: np.ndarray) -> list[float]:
    """Per-item contribution to rank agreement: negative |rank residual|, scaled to [0,1].

    GEPA needs per-instance scores for its Pareto frontier; the pool objective (Spearman) is
    rank-equivalent to maximizing the mean of these (same discipline as m_omega_gepa's
    MAD-decomposed per-instance signal)."""
    n = len(target)
    if n < 2:
        return [0.0] * n
    r_hat = _rankdata(np.asarray(pyes, float))
    r_tgt = _rankdata(np.asarray(target, float))
    resid = np.abs(r_hat - r_tgt) / (n - 1)
    return [float(1.0 - r) for r in resid]


def make_adapter(executor_backend, texts: list[str], target: np.ndarray, construct: str,
                 template: str, max_text_chars: int, label_token_ids: dict,
                 score_fn=None):
    """Build the GEPAAdapter subclass instance (lazy gepa import)."""
    from gepa.core.adapter import EvaluationBatch, GEPAAdapter

    from methods.tacit_channels.channels.eval.teacher_forced_lora import (
        score_declared_binary_lora,
    )
    score_fn = score_fn or score_declared_binary_lora

    class ReconstructionAdapter(GEPAAdapter):
        """Fitness = executor-with-candidate-articulation reconstructing the target."""

        def __init__(self):
            self._cache: dict[str, np.ndarray] = {}

        def _pyes(self, articulation: str) -> np.ndarray:
            key = articulation.strip()
            if key not in self._cache:
                prompts = render_prompts(key, construct, texts, template, max_text_chars)
                self._cache[key] = np.asarray(score_fn(
                    executor_backend, prompts, pos="YES", neg="NO",
                    expected_token_ids=label_token_ids, seed=20260722), float)
            return self._cache[key]

        def evaluate(self, batch, candidate, capture_traces=False):
            articulation = candidate[COMPONENT]
            pyes = self._pyes(articulation)
            scores = per_instance_scores(pyes, target)
            pool_rho = spearman(pyes, target)
            outputs = [{"pyes": float(p), "target": float(t)}
                       for p, t in zip(pyes, target)]
            trajectories = None
            if capture_traces:
                trajectories = [{"pool_rho": pool_rho, "item": i,
                                 "pyes": float(pyes[i]), "target": float(target[i])}
                                for i in range(len(texts))]
            return EvaluationBatch(outputs=outputs, scores=scores,
                                   trajectories=trajectories)

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            order = np.argsort(eval_batch.scores)  # worst rank-residual items first
            worst = order[:8]
            records = []
            for i in worst:
                records.append({
                    "Inputs": texts[i][:800],
                    "Generated Outputs": json.dumps(eval_batch.outputs[i]),
                    "Feedback": (
                        "The executor's judgment of this item disagrees most with the target "
                        "policy's rank. Revise the articulation so the stated criterion "
                        "captures whatever distinction the target policy applies here, "
                        "without naming the item."),
                })
            return {COMPONENT: records}

    return ReconstructionAdapter()


def optimize(executor_backend, reflection_fn, seed_articulation: str, construct: str,
             texts: list[str], target: np.ndarray, template: str, max_text_chars: int,
             label_token_ids: dict, rounds: int = 6, n_mutations: int = 4) -> dict:
    """Run gepa.optimize with the reconstruction adapter. Returns winner + trajectory."""
    import gepa

    adapter = make_adapter(executor_backend, texts, target, construct, template,
                           max_text_chars, label_token_ids)
    budget = rounds * n_mutations * len(texts)
    result = gepa.optimize(
        seed_candidate={COMPONENT: seed_articulation},
        trainset=list(range(len(texts))), valset=list(range(len(texts))),
        adapter=adapter, reflection_lm=reflection_fn,
        max_metric_calls=budget, run_dir=None, seed=0,
        raise_on_exception=False)
    best = result.best_candidate[COMPONENT]
    pyes = adapter._pyes(best)
    out = {
        "optimized_articulation": best,
        "rho": spearman(pyes, target),
        "seed_rho": spearman(adapter._pyes(seed_articulation), target),
        "n_candidates_evaluated": len(adapter._cache),
    }
    # seed-only guard (m_omega_gepa L401-409 discipline): gepa swallows backend errors and
    # can silently return the seed - make that failure loud.
    if out["n_candidates_evaluated"] <= 1:
        out["warning"] = ("GEPA evaluated only the seed candidate - reflection or executor "
                          "backend failed silently; result is NOT an optimization outcome.")
    return out
