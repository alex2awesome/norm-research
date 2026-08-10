"""GEPA-M_ω: optimize the metric prompt for discriminative M_ω (2026-06-25;
official-GEPA migration 2026-07-19).

Problem: M_ω = the executor X's verdict on the metric prompt. Currently we score the raw R2
`merged_description`, which is often skewed or near-constant (capitalization → 93% YES; PRISMA →
100% NO). A near-constant M_ω caps recovery at ~0 — there's nothing to reconstruct.

Solution: Use GEPA to GENERATE a discriminative M_ω — optimize the metric prompt so the executor's
verdict actually varies across items (balanced base-rate ≈ 0.5, high std). This is the proof's
`OPT = sup_p R(p)` search over prompts, with transmission/discrimination as the objective.

Algorithm (OFFICIAL github `gepa` since 2026-07-19 — the hand-rolled rounds/mutations loop is
archived VERBATIM in `archive/inhouse_m_omega_gepa_deprecated.py`; user directive D1 in
notes/2026-07-19__gepa-consolidation-and-momega-upgrade-plan.md):
  1. Seed candidate = seed_body (the R2 merged_description).
  2. `gepa.optimize` drives the search through a `GEPAAdapter`:
     - Executor (X) scores each candidate's M_ω on items via SAMPLED YES/NO (generate "YES"/"NO",
       parse — mirror recon_channel._sampled_binary). Unchanged primitive `_score_binary_sampled`.
     - Per-instance search signal = the mean-absolute-deviation decomposition of the pool objective
       (`_per_instance_discrimination`), which is RANK-EQUIVALENT to `_discrimination_score` for
       binary verdicts — GEPA needs per-instance scores for its Pareto frontier.
     - Reflection LM (the `reviser` backend) proposes rewritten criteria from feedback built out of
       the same near-0.5 failure items (`_select_failures`) + `_mutation_prompt` intent as before.
  3. Objective (REPORTED, unchanged) = discrimination: std(pyes) - 0.5*abs(mean(pyes)-0.5)
     (high std, base-rate near 0.5), computed post-hoc over the full pool for the seed and every
     distinct candidate.
  4. Return optimized prompt, M_ω (pyes vector), mean, std, base_rate, trajectory.

Test harness: Compare raw merged_description M_ω vs GEPA-optimized M_ω on creative-writing metrics.

CAVEAT (memory `project_a_bank_degeneracy_audit`): "variance-revival != information-revival". A
discrimination-maximizing objective can produce high-variance but UNINFORMATIVE criteria (mined
banks ran 54-68% degenerate). This module preserves that pre-existing objective across the
optimizer migration; it does NOT endorse it as a recovery or certification target.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.metric_implementer.backends import LLMBackend
from methods.metric_implementer.batch_scoring import _YESNO_TEMPLATE
from methods.metric_implementer.config import ImplementerConfig, apply_task_preset, REPO_ROOT
from methods.metric_implementer.experiments.mine_clusters import r2_groups


@dataclass
class GEPAStats:
    """Statistics for one candidate prompt."""
    prompt: str
    pyes: np.ndarray  # binary vector (0/1, NaN for parse failures)
    mean: float
    std: float
    base_rate: float
    discrimination: float  # objective = std - 0.5*abs(mean - 0.5)


def _score_binary_sampled(backend: LLMBackend, rubric: str, texts: List[str],
                           max_chars: int, temperature: float = 0.7,
                           seed: int = 0) -> np.ndarray:
    """Sampled YES/NO verdict per item (mirror recon_channel._sampled_binary)."""
    prompts = [_YESNO_TEMPLATE.format(rubric=rubric, text=t[:max_chars]) for t in texts]
    outs = backend.generate_batch(prompts, system=None, max_tokens=4,
                                  temperature=temperature, seed=seed)
    v = np.full(len(texts), np.nan)
    for i, o in enumerate(outs):
        t = (o or "").strip().lower()
        if t.startswith("yes"):
            v[i] = 1.0
        elif t.startswith("no"):
            v[i] = 0.0
    return v


def _discrimination_score(pyes: np.ndarray) -> float:
    """Objective = high std + base-rate near 0.5. Returns std - 0.5*abs(mean - 0.5)."""
    if np.all(np.isnan(pyes)):
        return -np.inf
    valid = pyes[~np.isnan(pyes)]
    if len(valid) == 0:
        return -np.inf
    mean_val = float(np.mean(valid))
    std_val = float(np.std(valid))
    # Reward high std, penalize deviation from 0.5 base-rate
    return std_val - 0.5 * abs(mean_val - 0.5)


def _compute_stats(pyes: np.ndarray, prompt: str) -> GEPAStats:
    """Compute full statistics for a candidate prompt."""
    valid = pyes[~np.isnan(pyes)]
    mean_val = float(np.mean(valid)) if len(valid) > 0 else 0.0
    std_val = float(np.std(valid)) if len(valid) > 0 else 0.0
    return GEPAStats(
        prompt=prompt,
        pyes=pyes,
        mean=mean_val,
        std=std_val,
        base_rate=mean_val,
        discrimination=_discrimination_score(pyes)
    )


def _mutation_prompt(noun: str, current_prompt: str, round_idx: int,
                    failure_examples: str) -> str:
    """Generate the reviser's mutation request."""
    return f"""You are refining an evaluation criterion for {noun} excerpts.

Your task: PARAPHRASE, SHARPEN, or NARROW THE SCOPE of the criterion below so that a fresh
evaluator would give MORE VARIED verdicts across different {noun} examples (mix of YES and NO,
NOT near-constant). Keep the SAME underlying property — just reword it to help the evaluator
discriminate better.

Current criterion (round {round_idx}):
{current_prompt}

Examples where the current criterion struggles (near-ambiguous / wrong base-rate):
{failure_examples}

Propose ONE reworded criterion. Reply with ONLY the criterion text (no preamble, no JSON,
no explanation)."""


def _per_instance_discrimination(pyes: np.ndarray) -> List[float]:
    """PER-INSTANCE search signal for the official-GEPA adapter (Pareto selection needs one score
    per instance, but `_discrimination_score` is a POOL-level statistic).

    Mean-absolute-deviation decomposition of the pool objective:

        s_i = |p_i - p_bar| - 0.5 * |p_bar - 0.5|,   p_bar = mean over VALID verdicts in the batch

    so that  mean_i(s_i) = MAD(p) - 0.5*|p_bar - 0.5|,  versus the canonical
             D(p)        = std(p) - 0.5*|p_bar - 0.5|.

    WHY THIS IS FAITHFUL (rank-equivalence, exact for binary verdicts): `_score_binary_sampled`
    emits p_i in {0, 1}, so with u = p_bar*(1 - p_bar),

        MAD = p_bar*(1 - p_bar) + (1 - p_bar)*p_bar = 2u        std = sqrt(u)

    Both are strictly increasing in u, and u = 0.25 - d^2 where d = |p_bar - 0.5|. Hence
    mean_i(s_i) = 0.5 - 2d^2 - 0.5d and D = sqrt(0.25 - d^2) - 0.5d are BOTH strictly decreasing
    functions of the same scalar d, so they induce the SAME ordering over candidate prompts. The
    search signal is therefore rank-equivalent to `_discrimination_score`, and GEPA's full-valset
    aggregate selects exactly the candidate the canonical objective would select.

    NaN verdicts (parse failures) score -1.0 and are excluded from p_bar.
    """
    arr = np.asarray(pyes, dtype=float)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return [-1.0] * len(arr)
    p_bar = float(np.mean(valid))
    penalty = 0.5 * abs(p_bar - 0.5)
    return [-1.0 if np.isnan(p) else float(abs(float(p) - p_bar) - penalty) for p in arr]


def _select_failures(texts: List[str], pyes: np.ndarray, k: int = 10,
                    max_chars: int = 600) -> str:
    """Select k items where M_ω is near-ambiguous (pyes near 0.5) for reviser feedback."""
    if np.all(np.isnan(pyes)):
        return "(no valid scores to select from)"

    # Items with scores closest to 0.5 (most ambiguous)
    valid_mask = ~np.isnan(pyes)
    if not np.any(valid_mask):
        return "(no valid scores)"

    valid_pyes = pyes[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    # Sort by distance from 0.5
    distances = np.abs(valid_pyes - 0.5)
    sorted_order = np.argsort(distances)
    k_select = min(k, len(sorted_order))
    selected_indices = valid_indices[sorted_order[:k_select]]

    examples = []
    for idx in selected_indices:
        score = pyes[idx]
        examples.append(f"[score={score:.2f}]\n```\n{texts[idx][:max_chars]}\n```")

    return "\n\n".join(examples) if examples else "(no failure examples)"


# Output constraint appended to every reflective-feedback string. The in-house loop enforced this
# inside its mutation request ("Reply with ONLY the criterion text"); official GEPA's reflection LM
# writes the component itself, so the constraint has to travel in the feedback it reads.
_M_OMEGA_CONSTRAINT = (
    "HARD CONSTRAINTS for the rewritten criterion: it must remain ONE self-contained evaluation "
    "criterion that a fresh evaluator can apply to a single excerpt and answer YES or NO; do NOT "
    "copy in examples, exemplars, or excerpt text; do NOT add meta-commentary, preamble, rationale, "
    "or headings — emit the criterion text only."
)


def _reflective_feedback(noun: str, prompt: str, texts: List[str], pyes: np.ndarray,
                         depth: int, max_chars: int = 600) -> str:
    """Feedback string handed to GEPA's reflection LM.

    Reuses the in-house loop's ingredients: `_select_failures` (the items whose verdict sits nearest
    0.5 — where M_ω is least decisive) and `_mutation_prompt`'s framing (reword / sharpen / narrow so
    verdicts VARY while the underlying property stays the same). Adds the measured base-rate/std of
    the current criterion and the explicit output constraint.
    """
    arr = np.asarray(pyes, dtype=float)
    stats = _compute_stats(arr, prompt)
    failures = _select_failures(texts, arr, k=10, max_chars=max_chars)
    measured = (
        f"Measured on this batch: base_rate={stats.base_rate:.3f} (target 0.5), "
        f"std={stats.std:.3f} (higher is better), discrimination={stats.discrimination:.3f}."
    )
    return (f"{_mutation_prompt(noun, prompt, depth, failures)}\n\n"
            f"{measured}\n\n{_M_OMEGA_CONSTRAINT}")


def gepa_discriminative_m_omega(executor: LLMBackend, reviser: LLMBackend,
                                seed_body: str, texts: List[str], noun: str,
                                *, rounds: int = 3, n_mutations: int = 4,
                                max_chars: int = 600,
                                mutation_mode: str = "reword",
                                fewshot_examples=None,
                                max_metric_calls: Optional[int] = None) -> dict:
    """Official-GEPA search for a discriminative M_ω prompt.

    DEPRECATION NOTE (2026-07-19, user directive D1): the hand-rolled rounds/mutations loop that used
    to live here is archived VERBATIM in `archive/inhouse_m_omega_gepa_deprecated.py`. Official
    github `gepa` is the only sanctioned optimizer for reconstruction experiments. The SIGNATURE and
    the RETURN CONTRACT are unchanged, so `run_r2_recovery.py --gepa-m-omega` needs no edit.

    THE ESTIMAND IS UNCHANGED. The executor primitive is still `_score_binary_sampled` (sampled
    YES/NO), and the REPORTED objective is still the canonical pool-level `_discrimination_score`
    = std(pyes) - 0.5*|mean(pyes) - 0.5|.

    SEARCH SIGNAL vs REPORTED STATISTIC. GEPA requires PER-INSTANCE scores for its Pareto frontier,
    but the native objective is pool-level. The adapter therefore scores instances with
    `_per_instance_discrimination` (the mean-absolute-deviation decomposition), which is
    RANK-EQUIVALENT to `_discrimination_score` for binary verdicts — proof in that function's
    docstring. Because the valset is the whole pool, GEPA's aggregate selection therefore picks the
    same candidate the canonical objective would, PROVIDED parse-failure rates are equal across
    candidates: NaN verdicts score -1.0 in the search signal but are EXCLUDED from the canonical
    statistic, so the search actually maximizes (discrimination among valid verdicts) minus (a
    parse-failure penalty). A criterion that discriminates well among the items it parses but fails
    to parse often can rank below a weaker-but-cleaner one. The seed is candidate #0 in GEPA's own
    evaluation, so the returned prompt is never worse than the seed on the search signal. The
    canonical statistic is then recomputed POST-HOC over the full pool for the seed, the winner, and
    every distinct candidate (same discipline as `run_v14_value_campaign.run_decoder_tuning`).

    BUDGET MAPPING. The legacy `rounds`/`n_mutations` knobs (CLI flags `--gepa-rounds`,
    `--gepa-mutations` in run_r2_recovery.py) no longer name loop structure — official GEPA schedules
    its own iterations — so they are mapped to a metric-call budget:

        max_metric_calls = rounds * n_mutations * len(texts)

    i.e. the same number of item-level executor evaluations the old loop would have spent on its
    mutations. Pass `max_metric_calls` to override directly.

    INERT KWARGS. `mutation_mode` / `fewshot_examples` drove the in-house code-appended few-shot
    mutation operator, which has no equivalent under GEPA's reflective proposer (GEPA writes the
    candidate itself). They are still ACCEPTED for signature compatibility but ignored, with a
    warning if a caller asks for a non-default value. Note the operator was not recommended anyway:
    the 2026-06-25 A/B found code-selected few-shot HURT discrimination (0.448 -> 0.292).

    CAVEAT (memory `project_a_bank_degeneracy_audit`): "variance-revival != information-revival" — a
    discrimination-maximizing objective can produce high-variance but UNINFORMATIVE criteria. This
    migration preserves the pre-existing objective and does not endorse it.

    Args:
        executor: LLMBackend for scoring (X, e.g., glm-4.7)
        reviser: LLMBackend used as GEPA's reflection LM (e.g., glm-5.2)
        seed_body: Initial prompt (R2 merged_description) = GEPA's seed candidate
        texts: List of item texts to score
        noun: Item noun (e.g., "story", "paper")
        rounds: Legacy knob; folded into the metric-call budget (default 3)
        n_mutations: Legacy knob; folded into the metric-call budget (default 4)
        max_chars: Max chars per text (default 600)
        mutation_mode: Accepted for compatibility, ignored (see INERT KWARGS)
        fewshot_examples: Accepted for compatibility, ignored (see INERT KWARGS)
        max_metric_calls: Optional direct budget override

    Returns:
        dict with (IDENTICAL to the in-house loop):
            - 'optimized_prompt': best prompt found
            - 'pyes': binary verdict vector (n_items)
            - 'mean': mean of pyes
            - 'std': std of pyes
            - 'base_rate': mean of pyes
            - 'discrimination': discrimination score
            - 'trajectory': list of (index, prompt, std, base_rate, discrimination). Index 0 is the
              seed and the LAST row is the prompt actually returned, as before; the rows between are
              now every DISTINCT candidate GEPA evaluated (the in-house loop logged the running best
              once per round, which GEPA has no analogue for).
    """
    n_items = len(texts)
    if n_items == 0:
        raise ValueError("gepa_discriminative_m_omega requires a non-empty text pool")
    if mutation_mode != "reword" or fewshot_examples:
        warnings.warn(
            "m_omega_gepa: mutation_mode/fewshot_examples are inert under official GEPA "
            "(the code-appended few-shot mutation operator was archived 2026-07-19 in "
            "archive/inhouse_m_omega_gepa_deprecated.py); proceeding with reflective rewording.",
            RuntimeWarning, stacklevel=2,
        )

    import gepa
    from gepa.core.adapter import EvaluationBatch, GEPAAdapter

    budget = (int(max_metric_calls) if max_metric_calls is not None
              else int(rounds) * int(n_mutations) * n_items)
    budget = max(budget, n_items)  # at least one full-pool pass (the seed candidate)
    component = "m_omega_criterion"
    seed_prompt = str(seed_body)
    instances = [{"index": index, "text": texts[index]} for index in range(n_items)]

    # (prompt -> {item index -> verdict}). One cache for the whole search, so a candidate GEPA
    # evaluated on the full valset costs nothing to re-report post-hoc, and a candidate seen only on
    # a reflection minibatch is merely topped up. `_score_binary_sampled` stays the sole primitive.
    # Each (prompt, item) is therefore drawn ONCE and the same draw backs both the search score and
    # the reported statistic — deliberate: it is cheaper than rescoring and it guarantees the report
    # describes the verdicts the search actually saw.
    verdicts: dict = {}
    proposals: List[str] = [seed_prompt]

    def _cached_verdicts(prompt: str, indices) -> np.ndarray:
        cache = verdicts.setdefault(prompt, {})
        todo = [index for index in indices if index not in cache]
        if todo:
            scored = _score_binary_sampled(executor, prompt, [texts[i] for i in todo], max_chars)
            for slot, index in enumerate(todo):
                cache[index] = float(scored[slot])
        return np.asarray([cache[index] for index in indices], dtype=float)

    def _pool_stats(prompt: str) -> GEPAStats:
        return _compute_stats(_cached_verdicts(prompt, list(range(n_items))), prompt)

    class MOmegaAdapter(GEPAAdapter):
        def evaluate(self, batch, candidate, capture_traces=False):
            prompt = str(next(iter(candidate.values()))).strip()
            if len(prompt) < 10:
                # Mirrors the in-house loop's "skip empty/too-short mutations", but as a scored
                # rejection so GEPA learns the constraint instead of silently losing the iteration.
                trajs = ([{"data": item, "full_assistant_response": "",
                           "feedback": "REJECTED before scoring: the criterion was empty or under "
                                       f"10 characters. {_M_OMEGA_CONSTRAINT}"}
                          for item in batch] if capture_traces else None)
                return EvaluationBatch(outputs=[{}] * len(batch), scores=[-1.0] * len(batch),
                                       trajectories=trajs)
            if prompt not in proposals:
                proposals.append(prompt)
            indices = [int(item["index"]) for item in batch]
            pyes = _cached_verdicts(prompt, indices)
            trajs = None
            if capture_traces:
                feedback = _reflective_feedback(noun, prompt, [texts[i] for i in indices], pyes,
                                                depth=len(proposals), max_chars=max_chars)
                trajs = [{"data": item, "full_assistant_response": "", "feedback": feedback}
                         for item in batch]
            return EvaluationBatch(outputs=[{}] * len(batch),
                                   scores=_per_instance_discrimination(pyes), trajectories=trajs)

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            target = components_to_update[0]
            items = [{
                "Inputs": f"one {noun} excerpt from the design pool (withheld — the criterion must "
                          "stay a general rule, not a description of these items)",
                "Generated Outputs": "(sampled YES/NO verdict from the executor)",
                "Feedback": trajectory["feedback"],
            } for trajectory in (eval_batch.trajectories or [])]
            return {target: items}

    def reflection_lm(prompt):
        text = prompt if isinstance(prompt, str) else json.dumps(prompt)
        return str(reviser.generate_batch(
            [text], system=None, max_tokens=1200, temperature=0.9,
        )[0])

    print(f"[gepa_discriminative_m_omega] official-gepa n_items={n_items} budget={budget} "
          f"(rounds={rounds} x n_mutations={n_mutations})", flush=True)
    result = gepa.optimize(
        seed_candidate={component: seed_prompt},
        trainset=instances, valset=instances,
        adapter=MOmegaAdapter(), reflection_lm=reflection_lm,
        max_metric_calls=budget, run_dir=None, seed=0,
        display_progress_bar=False, raise_on_exception=False,
    )
    best_candidate = dict(getattr(result, "best_candidate", None) or {})
    winner = str(next(iter(best_candidate.values()))).strip() if best_candidate else seed_prompt
    if len(winner) < 10:  # degenerate/empty winner — fall back to the seed rather than ship it
        winner = seed_prompt
    if len(proposals) <= 1 and winner == seed_prompt:
        # raise_on_exception=False means a down executor/reflection backend degrades to "return
        # the seed" with no error; make that failure mode loud instead of silent.
        warnings.warn(
            "m_omega_gepa: official GEPA returned the seed with no other candidate evaluated — "
            "the search may have silently failed (backend errors are swallowed by "
            "raise_on_exception=False). Check executor/reviser health before trusting this run.",
            RuntimeWarning, stacklevel=2,
        )

    # POST-HOC canonical report: per-instance signal drove the SEARCH, the canonical pool-level
    # `_discrimination_score` drives the REPORT, for the seed and every distinct candidate.
    reported = list(proposals)
    if winner not in reported:
        reported.append(winner)
    trajectory = []
    for index, prompt in enumerate(reported):
        stats = _pool_stats(prompt)
        trajectory.append((index, prompt, stats.std, stats.base_rate, stats.discrimination))
        label = "seed" if index == 0 else f"candidate {index}"
        print(f"  {label}: std={stats.std:.3f}, base_rate={stats.base_rate:.3f}, "
              f"disc={stats.discrimination:.3f}")

    best_stats = _pool_stats(winner)
    # Last row = the prompt actually returned (preserves the in-house trajectory[-1] contract).
    trajectory.append((len(reported), winner, best_stats.std, best_stats.base_rate,
                       best_stats.discrimination))
    print(f"  *** WINNER: std={best_stats.std:.3f}, base_rate={best_stats.base_rate:.3f}, "
          f"disc={best_stats.discrimination:.3f}")

    return {
        "optimized_prompt": winner,
        "pyes": best_stats.pyes,
        "mean": best_stats.mean,
        "std": best_stats.std,
        "base_rate": best_stats.base_rate,
        "discrimination": best_stats.discrimination,
        "trajectory": trajectory
    }


# --------------------------------------------------------------------------------------------
# M_ω v2 — unit recombination on top of official GEPA (D2 reconstruction side, 2026-07-20)
# --------------------------------------------------------------------------------------------
_UNIT_SPLIT_RE = r"(?<=[.!?])\s+|\n+|^[-*•]\s*"


def _clause_units(prompts: List[str], lo: int = 25, hi: int = 400) -> List[str]:
    """Split prompts into candidate instruction clauses (mirrors the benchmark-side miner)."""
    import re as _re
    units, seen = [], set()
    for text in prompts:
        for clause in _re.split(_UNIT_SPLIT_RE, str(text)):
            c = clause.strip().strip("-*• ").rstrip()
            key = " ".join(c.lower().split())
            if lo <= len(c) <= hi and key not in seen:
                seen.add(key)
                units.append(c)
    return units


def _suggest_units_llm(reviser: LLMBackend, noun: str, init_prompt: str, ambiguous: str,
                       n: int) -> List[str]:
    """D2 unit source (c): novel clauses proposed by the reviser from near-ambiguous items."""
    ask = (
        f"An evaluation criterion for {noun} excerpts is being refined by adding short, "
        f"standalone facet clauses.\n\nCURRENT CRITERION:\n{init_prompt}\n\nEXCERPTS WHERE ITS "
        f"VERDICT IS LEAST DECISIVE:\n{ambiguous}\n\nPropose {n} DISTINCT candidate facet "
        "clauses. Each must be one or two self-contained sentences a fresh evaluator could "
        "apply to a single excerpt, must sharpen what counts as YES vs NO, and must never "
        "reference the excerpts above. Reply with ONLY a JSON array of strings."
    )
    try:
        raw = str(reviser.generate_batch([ask], system=None, max_tokens=1200,
                                         temperature=1.0)[0])
        arr = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
        return [str(x).strip() for x in arr if 25 <= len(str(x).strip()) <= 400][:n]
    except Exception as exc:  # noqa: BLE001 — degraded pool, not a fatal error
        warnings.warn(f"unit_recombination_m_omega: LLM unit suggestion failed ({exc}); "
                      "continuing with trajectory+children units only", RuntimeWarning)
        return []


def _compile_units(init: str, units: List[str]) -> str:
    if not units:
        return init
    return (init + "\n\nWhen judging, also apply these refinements to the same criterion:\n"
            + "\n".join(f"- {u}" for u in units))


def unit_recombination_m_omega(executor: LLMBackend, reviser: LLMBackend, seed_body,
                               texts: List[str], noun: str, *,
                               children: Optional[List[str]] = None,
                               rounds: int = 3, n_mutations: int = 4, max_chars: int = 600,
                               max_metric_calls: Optional[int] = None,
                               top_k: int = 8, n_llm_units: int = 8, max_units: int = 40,
                               confirm_frac: float = 0.25, seed: int = 0) -> dict:
    """M_ω v2 (D2, reconstruction side): official-GEPA best candidate + greedy unit recombination.

    GEOMETRY (the UNIT_BOTTLENECK_DIAGNOSIS-supported design, ported from the benchmark side):
      1. Split the design texts into SELECT and CONFIRM slices (deterministic, seeded).
      2. Run official GEPA (`gepa_discriminative_m_omega`) on the SELECT slice -> init prompt.
         The compile INITIALIZES FROM GEPA's winner, so M_ω starts as a superset of GEPA's answer.
      3. Unit pool = (a) clauses mined from GEPA's own trajectory candidates, (b) THE CHILDREN
         METRICS — certified R1 child criteria when judging an R2 (the D2 source with no
         benchmark analog), (c) reviser-suggested novel clauses from near-ambiguous items.
      4. Paired marginal pass: every unit appended alone to init, scored on the SAME select
         slice; rank by canonical-discrimination delta. NO cheap screen.
      5. Cumulative-prefix sweep over the top-k positives + a drop-one pass.
      6. NO-REGRET GUARD on the disjoint confirm slice: ship init unless the compile beats it
         there. So M_ω >= GEPA-winner by construction, up to confirmation noise.

    ESTIMAND UNCHANGED: selection signal and report are the canonical `_discrimination_score`
    over `_score_binary_sampled` verdicts. The D6 caveat applies to this objective exactly as it
    does to `gepa_discriminative_m_omega` (a capacity objective; revisiting it is open decision
    3 and needs sign-off — this function deliberately does NOT change the objective).

    RETURN: the same 7-key contract as `gepa_discriminative_m_omega` (drop-in), plus a "units"
    dict (pool sizes per source, marginals, compiled units, guard outcome). Trajectory keeps the
    5-tuple rows; row 0 = seed, last row = the prompt actually shipped; unit-phase probes are
    appended between GEPA's rows and the final row. All final stats are over the FULL design
    pool passed in (select + confirm), like the wrapped function.
    """
    n_items = len(texts)
    if n_items == 0:
        raise ValueError("unit_recombination_m_omega requires a non-empty text pool")
    order = np.random.default_rng(seed).permutation(n_items)
    n_confirm = max(1, int(round(confirm_frac * n_items))) if n_items >= 8 else 0
    confirm_idx = [int(i) for i in order[:n_confirm]]
    select_idx = [int(i) for i in order[n_confirm:]]
    select_texts = [texts[i] for i in select_idx]
    confirm_texts = [texts[i] for i in confirm_idx]

    # ---- 1-2. official GEPA on the select slice only (confirm slice never seen by the search)
    g = gepa_discriminative_m_omega(executor, reviser, seed_body, select_texts, noun,
                                    rounds=rounds, n_mutations=n_mutations, max_chars=max_chars,
                                    max_metric_calls=max_metric_calls)
    init = str(g["optimized_prompt"])

    # ---- verdict cache for the unit phase (same one-draw-per-(prompt,item) discipline)
    verdicts: dict = {}

    def _verdicts(prompt: str, idx_list: List[int]) -> np.ndarray:
        cache = verdicts.setdefault(prompt, {})
        todo = [i for i in idx_list if i not in cache]
        if todo:
            scored = _score_binary_sampled(executor, prompt, [texts[i] for i in todo], max_chars)
            for slot, i in enumerate(todo):
                cache[i] = float(scored[slot])
        return np.asarray([cache[i] for i in idx_list], dtype=float)

    def _disc(prompt: str, idx_list: List[int]) -> float:
        return _discrimination_score(_verdicts(prompt, idx_list))

    init_disc = _disc(init, select_idx)

    # ---- 3. unit pool: trajectory clauses + children metrics + LLM suggestions
    traj_prompts = [row[1] for row in (g.get("trajectory") or [])]
    traj_units = _clause_units(traj_prompts)
    child_units = [c.strip() for c in (children or []) if c and len(c.strip()) > 4]
    ambiguous = _select_failures(select_texts, _verdicts(init, select_idx), k=6,
                                 max_chars=max_chars)
    llm_units = _suggest_units_llm(reviser, noun, init, ambiguous, n_llm_units)
    pool, seen = [], set()
    for u in child_units + llm_units + traj_units:   # children first: the D2-novel source
        key = " ".join(u.lower().split())
        if key not in seen:
            seen.add(key)
            pool.append(u)
    pool = pool[:max_units]
    print(f"[unit_recombination_m_omega] units: {len(child_units)} children + {len(llm_units)} "
          f"LLM + {len(traj_units)} trajectory -> {len(pool)} used | select={len(select_idx)} "
          f"confirm={len(confirm_idx)} | init disc={init_disc:.3f}", flush=True)

    # ---- 4. paired marginal pass on the select slice
    marginals = []
    for u in pool:
        cand = _compile_units(init, [u])
        marginals.append({"unit": u, "delta": _disc(cand, select_idx) - init_disc,
                          "source": ("children" if u in child_units else
                                     "llm" if u in llm_units else "trajectory")})
    marginals.sort(key=lambda r: -r["delta"])

    # ---- 5. cumulative prefix + drop-one
    ordered = [r["unit"] for r in marginals if r["delta"] > 0][:top_k]
    best_set: List[str] = []
    best_disc = init_disc
    probe_rows = []
    cum: List[str] = []
    for u in ordered:
        cum = cum + [u]
        cand = _compile_units(init, cum)
        d = _disc(cand, select_idx)
        stats = _compute_stats(_verdicts(cand, select_idx), cand)
        probe_rows.append((cand, stats))
        if d > best_disc:
            best_set, best_disc = list(cum), d
    for u in list(best_set):
        trial = [x for x in best_set if x != u]
        cand = _compile_units(init, trial)
        d = _disc(cand, select_idx)
        if d >= best_disc:
            best_set, best_disc = trial, d

    compiled = _compile_units(init, best_set)

    # ---- 6. no-regret guard on the disjoint confirm slice
    fell_back = False
    if best_set and confirm_idx:
        if _disc(compiled, confirm_idx) < _disc(init, confirm_idx):
            compiled, fell_back = init, True
    print(f"[unit_recombination_m_omega] compiled {len(best_set)} units "
          f"(select disc {init_disc:.3f} -> {best_disc:.3f})"
          + ("; NO-REGRET GUARD fired -> shipping GEPA winner" if fell_back else ""), flush=True)

    # ---- report: canonical stats over the FULL design pool
    full_idx = list(range(n_items))
    final_stats = _compute_stats(_verdicts(compiled, full_idx), compiled)
    trajectory = [(row[0], row[1], row[2], row[3], row[4]) for row in (g.get("trajectory") or [])]
    base_len = len(trajectory)
    for offset, (cand, stats) in enumerate(probe_rows):
        trajectory.append((base_len + offset, cand, stats.std, stats.base_rate,
                           stats.discrimination))
    trajectory.append((len(trajectory), compiled, final_stats.std, final_stats.base_rate,
                       final_stats.discrimination))

    return {
        "optimized_prompt": compiled,
        "pyes": final_stats.pyes,
        "mean": final_stats.mean,
        "std": final_stats.std,
        "base_rate": final_stats.base_rate,
        "discrimination": final_stats.discrimination,
        "trajectory": trajectory,
        "units": {
            "n_children": len(child_units), "n_llm": len(llm_units),
            "n_trajectory": len(traj_units), "n_pool_used": len(pool),
            "n_compiled": len(best_set), "compiled_units": list(best_set),
            "marginals_top10": marginals[:10],
            "init_is_gepa_winner": True, "init_select_discrimination": float(init_disc),
            "compiled_select_discrimination": float(best_disc),
            "fell_back_to_init": bool(fell_back),
            "n_select": len(select_idx), "n_confirm": len(confirm_idx),
        },
    }


def _load_creative_writing_pool() -> tuple[List[str], List[str]]:
    """Load creative-writing pool from methods/metric_implementer/trial/pool_creative_writing.jsonl.gz

    Returns:
        (texts, ids) tuples
    """
    pool_path = REPO_ROOT / "methods/metric_implementer/trial/pool_creative_writing.jsonl.gz"
    texts, ids = [], []

    with gzip.open(pool_path, 'rt') as f:
        for line in f:
            try:
                obj = json.loads(line)
                texts.append(obj["text"])
                ids.append(obj.get("candidate_id", f"idx_{len(texts)}"))
            except Exception:
                continue

    return texts, ids


def main():
    """Test harness: compare raw M_ω vs GEPA-optimized M_ω on creative-writing metrics."""
    parser = argparse.ArgumentParser(
        description="GEPA-M_ω: optimize metric prompts for discriminative M_ω")
    parser.add_argument("--task", default="creative-writing",
                       choices=["creative-writing", "peer-review"],
                       help="Task to test on")
    parser.add_argument("--bucket", default="general",
                       choices=["general", "specific", "hyper_specific"],
                       help="R2 bucket to sample metrics from")
    parser.add_argument("--n-metrics", type=int, default=3,
                       help="Number of metrics to test")
    parser.add_argument("--rounds", type=int, default=3,
                       help="GEPA rounds per metric")
    parser.add_argument("--n-mutations", type=int, default=4,
                       help="Mutations per GEPA round")
    parser.add_argument("--executor-model", default="glm-4.7",
                       help="Executor model (X)")
    parser.add_argument("--reviser-model", default="glm-5.2",
                       help="Reviser model (GLM)")
    parser.add_argument("--backend", default="zai_anthropic",
                       choices=["zai_anthropic", "openrouter"],
                       help="Backend for API calls")
    parser.add_argument("--max-text-chars", type=int, default=600,
                       help="Max characters per text item")
    parser.add_argument("--n-items", type=int, default=60,
                       help="Number of items to score (subset of pool)")
    args = parser.parse_args()

    print("=" * 80)
    print("GEPA-M_ω: Discriminative Metric Prompt Optimization")
    print("=" * 80)

    # Load config
    cfg = ImplementerConfig()
    cfg.backend = args.backend
    apply_task_preset(cfg, args.task)

    # Create backends
    executor = LLMBackend(args.executor_model, "executor", cfg,
                         temperature=cfg.judge_temperature)
    reviser = LLMBackend(args.reviser_model, "reviser", cfg,
                        temperature=cfg.other_temperature)

    print(f"\nBackend: {args.backend}")
    print(f"Executor: {args.executor_model}")
    print(f"Reviser: {args.reviser_model}")
    print(f"Task: {args.task}, Bucket: {args.bucket}")

    # Load pool
    print(f"\nLoading {args.task} pool...")
    if args.task == "creative-writing":
        texts, ids = _load_creative_writing_pool()
    else:
        # For peer-review or other tasks, add path here
        pool_path = REPO_ROOT / f"methods/metric_implementer/trial/pool_{args.task.replace('-', '_')}.jsonl.gz"
        texts, ids = [], []
        if pool_path.exists():
            with gzip.open(pool_path, 'rt') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        texts.append(obj["text"])
                        ids.append(obj.get("candidate_id", f"idx_{len(texts)}"))
                    except Exception:
                        continue
        else:
            print(f"  ERROR: Pool not found at {pool_path}")
            return

    # Subsample if needed
    n_items = min(args.n_items, len(texts))
    if n_items < len(texts):
        import random
        idx = random.sample(range(len(texts)), n_items)
        texts = [texts[i] for i in idx]
        ids = [ids[i] for i in idx]

    print(f"  Loaded {len(texts)} items (max {args.max_text_chars} chars each)")

    # Load R2 metrics
    print(f"\nLoading R2 metrics for {args.task}/{args.bucket}...")
    r2_metrics = r2_groups(args.task, args.bucket)
    print(f"  Found {len(r2_metrics)} R2 metrics")

    # Select top metrics by leaf count
    r2_metrics_sorted = sorted(r2_metrics, key=lambda m: m["total_leaf_rubrics"], reverse=True)
    n_metrics = min(args.n_metrics, len(r2_metrics_sorted))
    selected = r2_metrics_sorted[:n_metrics]

    print(f"\nSelected {n_metrics} metrics to test:")
    for i, m in enumerate(selected):
        print(f"  {i+1}. [{m['group_idx']}] {m['merged_name']} "
              f"({m['total_leaf_rubrics']} leaves)")

    # Run GEPA on each metric
    print("\n" + "=" * 80)
    print("Running GEPA optimization...")
    print("=" * 80)

    results = []
    for i, metric in enumerate(selected):
        print(f"\n{'=' * 20} Metric {i+1}/{n_metrics}: {metric['merged_name']} {'=' * 20}")
        print(f"Seed description:\n  {metric['merged_description'][:200]}...")

        try:
            # Raw M_ω (seed)
            print("\n--- RAW M_ω (seed prompt) ---")
            raw_pyes = _score_binary_sampled(executor, metric['merged_description'],
                                            texts, args.max_text_chars)
            raw_stats = _compute_stats(raw_pyes, metric['merged_description'])

            print(f"Raw: std={raw_stats.std:.3f}, base_rate={raw_stats.base_rate:.3f}, "
                  f"disc={raw_stats.discrimination:.3f}")

            # GEPA-optimized M_ω
            print("\n--- GEPA-OPTIMIZED M_ω ---")
            gepa_result = gepa_discriminative_m_omega(
                executor, reviser, metric['merged_description'],
                texts, cfg.item_noun,
                rounds=args.rounds,
                n_mutations=args.n_mutations,
                max_chars=args.max_text_chars
            )

            print(f"\nGEPA: std={gepa_result['std']:.3f}, "
                  f"base_rate={gepa_result['base_rate']:.3f}, "
                  f"disc={gepa_result['discrimination']:.3f}")

            # Summary
            delta_std = gepa_result['std'] - raw_stats.std
            delta_br = abs(gepa_result['base_rate'] - 0.5) - abs(raw_stats.base_rate - 0.5)

            print(f"\nDelta: Δstd={delta_std:+.3f}, Δ|base_rate-0.5|={delta_br:+.3f}")

            results.append({
                "metric_name": metric['merged_name'],
                "metric_idx": metric['group_idx'],
                "raw_std": raw_stats.std,
                "raw_base_rate": raw_stats.base_rate,
                "raw_disc": raw_stats.discrimination,
                "gepa_std": gepa_result['std'],
                "gepa_base_rate": gepa_result['base_rate'],
                "gepa_disc": gepa_result['discrimination'],
                "delta_std": delta_std,
                "delta_br": delta_br,
                "optimized_prompt": gepa_result['optimized_prompt'],
                "trajectory": gepa_result['trajectory']
            })

        except Exception as e:
            print(f"\nERROR on metric {metric['merged_name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Raw vs GEPA-optimized M_ω")
    print("=" * 80)
    print(f"{'Metric':<40} | {'Raw std':>8} | {'Raw BR':>8} | {'GEPA std':>8} | "
          f"{'GEPA BR':>8} | {'Δstd':>7} | {'Δ|BR-0.5|':>9}")
    print("-" * 80)

    for r in results:
        metric_short = r['metric_name'][:37] + "..." if len(r['metric_name']) > 40 else r['metric_name']
        print(f"{metric_short:<40} | {r['raw_std']:8.3f} | {r['raw_base_rate']:8.3f} | "
              f"{r['gepa_std']:8.3f} | {r['gepa_base_rate']:8.3f} | "
              f"{r['delta_std']:+7.3f} | {r['delta_br']:+9.3f}")

    # Overall statistics
    n_improved_std = sum(1 for r in results if r['delta_std'] > 0)
    n_improved_br = sum(1 for r in results if r['delta_br'] < 0)  # negative = closer to 0.5

    print("-" * 80)
    print(f"Improved std: {n_improved_std}/{len(results)}")
    print(f"Improved base-rate (closer to 0.5): {n_improved_br}/{len(results)}")

    # Show optimized prompts for inspection
    print("\n" + "=" * 80)
    print("OPTIMIZED PROMPTS (for manual inspection)")
    print("=" * 80)

    for i, r in enumerate(results):
        print(f"\n--- Metric {i+1}: {r['metric_name']} ---")
        print(f"Raw: {selected[i]['merged_description'][:200]}...")
        print(f"\nGEPA-optimized:\n{r['optimized_prompt']}")
        print(f"\nStats: std={r['gepa_std']:.3f}, base_rate={r['gepa_base_rate']:.3f}")

    print("\n" + "=" * 80)
    print("Done. Cost summary:")
    print(f"  Executor ({args.executor_model}): {executor.stats.as_dict()}")
    print(f"  Reviser ({args.reviser_model}): {reviser.stats.as_dict()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
