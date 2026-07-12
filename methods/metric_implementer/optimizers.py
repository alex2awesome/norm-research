"""Distinct-mechanism prompt optimizers over the UNSUPERVISED fidelity objective.

GEPA (``optimizer.improve``) is the reflective, failure-attributed operator loop. This module
adds three optimizers with genuinely different *search* mechanisms so a generated prompt
population is not all-GEPA (needed to make "which prompt features predict recovery" generalize
across mechanisms, and to give the scaling sweeps optimizer diversity):

  * ``evoprompt``  — evolutionary search (Guo et al.): a POPULATION of rubrics, advanced each
                     round by mutation + crossover, truncation-selected by fidelity.
  * ``protegi``    — textual-gradient / APO (Pryzant et al.): summarize WHY a rubric fails (the
                     "gradient"), edit to address it, BEAM-search over edits.
  * ``ape``        — instruction induction / OPRO (Zhou et al. / Yang et al.): resample N fresh
                     rubrics from a meta-prompt conditioned on the best-so-far + their scores.

All share the objective (``measures.fidelity_scalar`` via ``compute_scorecard`` — label-free),
the role backends, budget caps, and registry/scorecard persistence; each tags its versions with
its ``optimizer`` name. N (``caps.data_budget``) and K (``caps.n_fewshots``) are swept caps, so
data-budget and few-shot axes are exercised exactly as in GEPA. Summaries match
``optimizer.improve`` so the same drivers/analysis consume them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .artifact import MetricArtifact
from .backends import Roles, parse_json_obj
from .config import BudgetCaps
from .measures import compute_scorecard
from .optimizer import _format_failures
from .registry import Registry

_WORD_RATIO = 0.75


@dataclass
class _Cand:
    artifact: MetricArtifact
    version_id: str
    scorecard: Optional[dict] = None
    fidelity: float = float("nan")
    operator: str = "INIT"


def _rank_key(c: _Cand):
    promotable = bool((c.scorecard or {}).get("promotable", False))
    return (promotable, np.nan_to_num(c.fidelity, nan=-1.0))


def _art(seed: MetricArtifact, body: str) -> MetricArtifact:
    return MetricArtifact(metric_id=seed.metric_id, kind="prompt", body=str(body),
                          name=seed.name, description=seed.description,
                          invariances=seed.invariances)


class _Engine:
    """Shared scaffold: data-budget (N) sampling, seed, persist+score, acceptance, summary."""

    def __init__(self, optimizer: str, seed: MetricArtifact, texts: List[str], roles: Roles,
                 cfg, registry: Registry, *, caps: BudgetCaps, rounds: int,
                 run_id: Optional[str], data_ids: Optional[List[str]], log):
        self.optimizer = optimizer
        self.seed = seed
        self.roles = roles
        self.cfg = cfg
        self.registry = registry
        self.caps = caps
        self.rounds = rounds
        self.log = log
        self.rng = np.random.default_rng(cfg.random_seed)
        self.run_id = run_id or f"{optimizer}_{seed.metric_id}_{int(time.time())}"
        data_ids = data_ids or [str(i) for i in range(len(texts))]
        if caps.data_budget is not None and len(texts) > caps.data_budget:   # N axis
            sel = self.rng.choice(len(texts), size=caps.data_budget, replace=False)
            texts = [texts[i] for i in sel]
            data_ids = [data_ids[i] for i in sel]
        self.texts, self.data_ids = texts, data_ids
        self.models = {"judge": cfg.judge_model, "reviser": cfg.reviser_model,
                       "reconstructor": cfg.reconstructor_model, "optimizer": optimizer}
        self.oracle_cache: Dict[str, tuple] = {}
        registry.register_metric(seed.metric_id, seed.name, seed.description)
        registry.log("RUN_STARTED", run_id=self.run_id, metric_id=seed.metric_id,
                     kind="prompt", optimizer=optimizer, caps=caps.as_dict(),
                     rounds=rounds, n_texts=len(texts))

    def word_cap(self) -> int:
        return int((self.caps.instruction_tokens or 100000) * _WORD_RATIO)

    def fewshot_cap(self) -> int:
        return self.caps.n_fewshots if self.caps.n_fewshots is not None else 8

    def persist_score(self, art: MetricArtifact, operator: str, parent: Optional[str],
                      rnd: int, rationale: str = "") -> Optional[_Cand]:
        bad = art.violates(self.caps)
        if bad:
            self.registry.log("VERSION_REJECTED_BUDGET", run_id=self.run_id,
                              metric_id=art.metric_id, operator=operator,
                              optimizer=self.optimizer, exceeded=bad,
                              complexity=art.complexity())
            return None
        vid = self.registry.create_version(
            art, operator=operator, optimizer=self.optimizer, parent_version=parent,
            run_id=self.run_id, optimizer_round=rnd, data_budget_ids=self.data_ids,
            caps_in_force=self.caps.as_dict(), models=self.models,
            notes=f"rationale: {rationale}" if rationale else "")
        card = compute_scorecard(art, self.texts, self.roles, self.cfg, self.rng,
                                 oracle_cache=self.oracle_cache)
        self.registry.save_scorecard(art.metric_id, vid, "prompt",
                                     {**card, "round": rnd, "run_id": self.run_id},
                                     eval_config_hash=f"r{rnd}")
        c = _Cand(art, vid, card, card["fidelity_scalar"], operator)
        self.log(f"  [{self.run_id} r{rnd}] {vid} ({operator}) "
                 f"fidelity={c.fidelity:.3f}")
        return c

    def seed_cand(self) -> _Cand:
        vid = self.registry.head(self.seed.metric_id, "prompt")
        if vid is None:
            vid = self.registry.create_version(
                self.seed, operator="INIT", optimizer=self.optimizer, parent_version=None,
                run_id=self.run_id, optimizer_round=0, data_budget_ids=self.data_ids,
                caps_in_force=self.caps.as_dict(), models=self.models)
        card = compute_scorecard(self.seed, self.texts, self.roles, self.cfg, self.rng,
                                 oracle_cache=self.oracle_cache)
        self.registry.save_scorecard(self.seed.metric_id, vid, "prompt",
                                     {**card, "round": 0, "run_id": self.run_id}, "r0")
        return _Cand(self.seed, vid, card, card["fidelity_scalar"], "INIT")

    def finalize(self, best: _Cand, seed: _Cand) -> dict:
        acc_best = compute_scorecard(best.artifact, self.texts, self.roles, self.cfg,
                                     self.rng, acceptance=True, oracle_cache=self.oracle_cache)
        acc_seed = compute_scorecard(seed.artifact, self.texts, self.roles, self.cfg,
                                     self.rng, acceptance=True, oracle_cache=self.oracle_cache)
        accepted = (best.version_id != seed.version_id
                    and np.nan_to_num(acc_best["fidelity_scalar"], nan=-1)
                    > np.nan_to_num(acc_seed["fidelity_scalar"], nan=-1)
                    and acc_best.get("promotable", False))
        if accepted:
            self.registry.set_head(best.artifact.metric_id, "prompt", best.version_id,
                                   self.run_id, judge_tier=self.roles.judge.model)
        summary = {
            "run_id": self.run_id, "metric_id": self.seed.metric_id, "kind": "prompt",
            "optimizer": self.optimizer, "caps": self.caps.as_dict(), "rounds": self.rounds,
            "seed_version": seed.version_id, "best_version": best.version_id,
            "seed_fidelity_inloop": seed.fidelity, "best_fidelity_inloop": best.fidelity,
            "seed_fidelity_acceptance": acc_seed["fidelity_scalar"],
            "best_fidelity_acceptance": acc_best["fidelity_scalar"],
            "accepted": bool(accepted), "cost_usd_so_far": self.roles.total_cost(),
        }
        self.registry.log("RUN_FINISHED", **summary, role_stats=self.roles.stats())
        self.log(f"  [{self.run_id}] done ({self.optimizer}): accepted={accepted} "
                 f"seed={summary['seed_fidelity_acceptance']:.3f} -> "
                 f"best={summary['best_fidelity_acceptance']:.3f}")
        return summary


def _propose(roles: Roles, prompt: str, n: int, key: str = "rubric") -> List[str]:
    raws = roles.reviser.generate_batch(
        [prompt] * n, max_tokens=1800,
        validate=lambda s: bool((parse_json_obj(s) or {}).get(key)))
    out = []
    for raw in raws:
        obj = parse_json_obj(raw)
        if obj and obj.get(key):
            out.append(str(obj[key]))
    return out


# ---- prompts (real-LLM-facing; the planted judge routes them by their key phrases) -------

_FEWSHOT_CLAUSE = ("Stay within {word_cap} words and at most {fewshot_cap} worked examples "
                   "(label any example 'EXAMPLE 1:', 'EXAMPLE 2:').")

_MUT = ("You are improving a rubric an LLM judge uses to score {item_noun} in [0,1] for the "
        "property:\n{description}\n\nCURRENT RUBRIC:\n{body}\n\nRewrite it into a DIFFERENT, "
        "higher-quality rubric for the SAME property — vary the wording and structure to make "
        "it a more reliable and valid evaluator. " + _FEWSHOT_CLAUSE +
        "\nRespond ONLY with JSON: {{\"rubric\": \"<the full rubric>\"}}")

_XOVER = ("Two rubrics, A and B, are used by an LLM judge to score {item_noun} for the "
          "property:\n{description}\n\nRUBRIC A:\n{a}\n\nRUBRIC B:\n{b}\n\nWrite ONE new rubric "
          "that combines the best of both A and B into a more reliable, valid evaluator. "
          + _FEWSHOT_CLAUSE +
          "\nRespond ONLY with JSON: {{\"rubric\": \"<the full rubric>\"}}")

_GRAD = ("A rubric is used by an LLM judge. Its measured failures:\n{failures}\n\nIn one or two "
         "sentences, state the single most important reason this rubric fails — the key fix it "
         "needs.\nRespond ONLY with JSON: {{\"critique\": \"<the reason>\"}}")

_APO_EDIT = ("Improve this rubric an LLM judge uses to score {item_noun} for the property:\n"
             "{description}\n\nCURRENT RUBRIC:\n{body}\n\nCRITIQUE (the main problem to fix):\n"
             "{critique}\n\nWrite an improved rubric addressing this critique. " + _FEWSHOT_CLAUSE +
             "\nRespond ONLY with JSON: {{\"rubric\": \"<the full rubric>\"}}")

_APE = ("You are writing the RUBRIC an LLM judge will use to score {item_noun} in [0,1] for the "
        "property:\n{description}\n\n{trajectory}Write a NEW rubric that will score HIGHER on "
        "reliability and validity than any above. Be concrete about what to look for. "
        + _FEWSHOT_CLAUSE +
        "\nRespond ONLY with JSON: {{\"rubric\": \"<the full rubric>\"}}")


def _fmt(tmpl: str, eng: _Engine, **kw) -> str:
    return tmpl.format(item_noun=getattr(eng.cfg, "item_noun", "item"),
                       description=eng.seed.description or eng.seed.name,
                       word_cap=eng.word_cap(), fewshot_cap=eng.fewshot_cap(), **kw)


# ---- the three optimizers ----------------------------------------------------------------

def evoprompt(seed, texts, roles, cfg, registry, *, caps=None, rounds=None, run_id=None,
              data_ids=None, pop_size=4, log=print) -> dict:
    """Evolutionary search: a population advanced by mutation + crossover, truncation-selected."""
    caps = caps or BudgetCaps()
    rounds = rounds if rounds is not None else (caps.optimizer_rounds or cfg.default_rounds)
    eng = _Engine("evoprompt", seed, texts, roles, cfg, registry, caps=caps, rounds=rounds,
                  run_id=run_id, data_ids=data_ids, log=log)
    s = eng.seed_cand()
    pop = [s]
    for b in _propose(roles, _fmt(_MUT, eng, body=seed.body), pop_size - 1):
        c = eng.persist_score(_art(seed, b), "EVO_MUT", s.version_id, 0)
        if c:
            pop.append(c)
    pop = sorted(pop, key=_rank_key, reverse=True)[:pop_size]
    for rnd in range(1, rounds + 1):
        offspring = []
        a, b = pop[0], pop[min(1, len(pop) - 1)]
        for body in _propose(roles, _fmt(_XOVER, eng, a=a.artifact.body, b=b.artifact.body), 1):
            c = eng.persist_score(_art(seed, body), "EVO_XOVER", a.version_id, rnd)
            if c:
                offspring.append(c)
        for parent in pop[:2]:
            for body in _propose(roles, _fmt(_MUT, eng, body=parent.artifact.body), 1):
                c = eng.persist_score(_art(seed, body), "EVO_MUT", parent.version_id, rnd)
                if c:
                    offspring.append(c)
        pop = sorted(pop + offspring, key=_rank_key, reverse=True)[:pop_size]
    return eng.finalize(max(pop, key=_rank_key), s)


def protegi(seed, texts, roles, cfg, registry, *, caps=None, rounds=None, run_id=None,
            data_ids=None, beam_width=2, edits_per=2, log=print) -> dict:
    """Textual-gradient (APO): critique a rubric's failures, edit to fix, beam-search."""
    caps = caps or BudgetCaps()
    rounds = rounds if rounds is not None else (caps.optimizer_rounds or cfg.default_rounds)
    eng = _Engine("protegi", seed, texts, roles, cfg, registry, caps=caps, rounds=rounds,
                  run_id=run_id, data_ids=data_ids, log=log)
    s = eng.seed_cand()
    beam = [s]
    for rnd in range(1, rounds + 1):
        cands = []
        for member in beam:
            failures = _format_failures(member.scorecard or {})
            crit = _propose(roles, _GRAD.format(failures=failures), 1, key="critique")
            critique = crit[0] if crit else "the rubric is too vague to apply consistently"
            for body in _propose(roles, _fmt(_APO_EDIT, eng, body=member.artifact.body,
                                             critique=critique), edits_per):
                c = eng.persist_score(_art(seed, body), "APO_EDIT", member.version_id, rnd,
                                      rationale=critique[:120])
                if c:
                    cands.append(c)
        beam = sorted(beam + cands, key=_rank_key, reverse=True)[:beam_width]
    return eng.finalize(max(beam, key=_rank_key), s)


def ape(seed, texts, roles, cfg, registry, *, caps=None, rounds=None, run_id=None,
        data_ids=None, n_candidates=4, log=print) -> dict:
    """Instruction induction / OPRO: resample fresh rubrics from a meta-prompt conditioned on
    the best-so-far rubrics and their fidelity scores."""
    caps = caps or BudgetCaps()
    rounds = rounds if rounds is not None else (caps.optimizer_rounds or cfg.default_rounds)
    eng = _Engine("ape", seed, texts, roles, cfg, registry, caps=caps, rounds=rounds,
                  run_id=run_id, data_ids=data_ids, log=log)
    s = eng.seed_cand()
    best_seen = [s]
    for rnd in range(1, rounds + 1):
        top = sorted(best_seen, key=_rank_key, reverse=True)[:3]
        traj = ""
        if any(np.isfinite(c.fidelity) for c in top):
            lines = "\n".join(f"[fidelity={c.fidelity:.2f}]\n{c.artifact.body[:600]}"
                              for c in top if np.isfinite(c.fidelity))
            traj = ("Previous rubrics and their fidelity scores (higher = better):\n"
                    f"{lines}\n\n")
        for body in _propose(roles, _fmt(_APE, eng, trajectory=traj), n_candidates):
            c = eng.persist_score(_art(seed, body), "APE_GEN", s.version_id, rnd)
            if c:
                best_seen.append(c)
    return eng.finalize(max(best_seen, key=_rank_key), s)


OPTIMIZERS = {"evoprompt": evoprompt, "protegi": protegi, "ape": ape}
