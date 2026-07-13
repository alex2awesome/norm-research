"""Anchor-free reconstruction measurement I_V(x -> m_recovered) (2026-06-16).

Unlike the consistency channel (re-run a fixed rubric; near-degenerate because pass noise is tiny),
this routes the metric through a real bottleneck:

    m's behavior on a TRAIN split  ->  a reconstructor induces a rule  ->  a FRESH executor applies
    the induced rule to HELD-OUT x  ->  I_V on the held-out verdicts.

The recovery instances are R stochastic reconstruction draws conditional on one shared training split
(the channel's real stochasticity is which rule gets induced). Two reconstruction modes:
  * free -- the reconstructor free-generates a rubric from hi/lo (text, m-verdict) examples.
  * mcq  -- the reconstructor PICKS one rule from a fixed candidate list after an exact contrastive
            teaching-set design. Target-option probability is the per-metric readout; I(J;Jhat) is
            the panel identity channel. Canonical-body replay is secondary. Neither mode is an
            upper-bound estimator or a ground-truth correctness test.

Scoring uses the CAMPAIGN protocol (forced YES/NO), not a [0,1] float rubric: the float rubric +
0.5 threshold collapses on lenient executors, whereas YES/NO discriminates (and keeps the numbers
comparable to the cached sampled long table). m's behavior + metric selection use the continuous
logprob P(YES); recovery/consistency passes use SAMPLED YES/NO (temperature, per-seed) -> binary
matrices -> the validated binary I_V estimator.

One resident vLLM model, ONE GPU. Reuses judges/vllm_backend/manifest/config/batch_scoring + vinfo.

Run (sk3, 1 GPU):
  CUDA_VISIBLE_DEVICES=6 VLLM_GPU_MEM_UTIL=0.3 HOME=/lfs/skampere3/0/alexspan \
    python -m methods.metric_implementer.recon_channel --tasks math_se,peer_review \
    --n-scan 12 --n-metrics 3 --R 5 --mode free
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import time
from typing import List

import numpy as np

from . import vinfo
from .backends import parse_json_obj
from .batch_scoring import _YESNO_TEMPLATE
from .config import ImplementerConfig, apply_task_preset
from .manifest import full_manifest, load_corpus, load_metrics
from .vllm_backend import make_judge_backend

CLONE_CAP = 0.97   # design-split behavioral clones are not identifiable MCQ distractors

_RECON_FREE = (
    "You are reverse-engineering a hidden evaluation criterion applied to {noun}.\n"
    "Below are excerpts, each with the binary score (1 = satisfies, 0 = does not) the hidden "
    "criterion assigned.\n\n{examples}\n\n"
    "{detail} The rule must let a fresh evaluator reproduce these scores on a NEW excerpt. "
    "Reply with ONLY the rule (no preamble, no JSON)."
)


def _detail_directive(L: int) -> str:
    """Map the articulation budget L (tokens) to a DETAIL level. A bare max_tokens cap doesn't bind
    (the model self-limits to ~40-token rules), so L must drive how much the rule articulates: more
    budget -> more specific conditions / edge cases, not just permission to ramble."""
    w = max(8, int(L * 0.75))
    if L <= 50:
        return "State the SINGLE property the criterion measures as ONE concise sentence."
    if L <= 150:
        return (f"State the property the criterion measures as a rule of about {w} words, with "
                "2-3 specific, discriminating conditions.")
    return (f"State the property as a DETAILED rubric of about {w} words: several specific "
            "conditions, the edge cases that flip the score, and what separates satisfying from "
            "non-satisfying excerpts.")

_RECON_MCQ = (
    "A hidden evaluation criterion assigned these binary scores (1/0) to {noun} excerpts.\n\n"
    "{examples}\n\n"
    "Which ONE of the following candidate criteria best explains the scores?\n\n{choices}\n\n"
    'Return ONLY JSON: {{"choice": <integer 1-{m}>}}'
)

_RECON_MCQ_LOGITS = (
    "A hidden evaluation criterion assigned these binary scores (1/0) to {noun} excerpts.\n\n"
    "{examples}\n\n"
    "Which ONE of the following candidate criteria best explains the scores?\n\n{choices}\n\n"
    "Answer with ONLY one digit from {labels}."
)

_RECON_MCQ_NO_DEMOS = "(No scored examples are provided.)"


_NORM_DESC = (
    "Rewrite this evaluation criterion for {noun} excerpts as ONE neutral sentence beginning "
    "'Whether the {noun} ', at most 16 words, plain register, no examples, no markdown, no rationale.\n\n"
    "Criterion: {desc}\n\nRewritten:"
)


def _normalize_descs(backend, noun, descs) -> list:
    """Paraphrase each option description into a uniform-register one-liner so MCQ identification
    can't exploit phrasing distinctiveness (vivid vs generic labels). Greedy/deterministic. Falls
    back to the truncated original if the rewrite is empty/too short."""
    prompts = [_NORM_DESC.format(noun=noun, desc=(d or "")[:400]) for d in descs]
    outs = backend.generate_batch(prompts, system=None, max_tokens=40, temperature=0.0, seed=0)
    res = []
    for d, o in zip(descs, outs):
        t = (o or "").strip().splitlines()[0].strip() if o else ""
        res.append(t if len(t) >= 8 else (d or "")[:120])
    return res


def _pyes(backend, rubric, texts, max_chars, *, template=None) -> np.ndarray:
    """Continuous P(YES) per item via 1-token logprobs (campaign continuous readout). NaN = neither
    token in top logprobs. ``template`` (a str.format template over ``{rubric}``/``{text}``) overrides
    the default rubric-first ``_YESNO_TEMPLATE`` — pass the TEXT-FIRST form so the shared probe text
    sits at the token prefix and vLLM prefix-caches it across criteria (efficiency lever #1)."""
    tpl = template if template is not None else _YESNO_TEMPLATE
    prompts = [tpl.format(rubric=rubric, text=t[:max_chars]) for t in texts]
    return np.array(backend.score_binary(prompts, pos="YES", neg="NO"), dtype=float)


def _sampled_binary(backend, rubric, texts, max_chars, temperature, seed) -> np.ndarray:
    """One SAMPLED YES/NO verdict per item (campaign sampled protocol). Binary, NaN if unparsed."""
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


def _hi_lo_examples(texts, pyes, k=4, max_chars=600) -> str:
    """Demonstrate the criterion with the k highest- and k lowest-P(YES) items (guarantees contrast
    even when the base rate is skewed)."""
    order = np.argsort(np.nan_to_num(pyes, nan=0.5))
    lo, hi = order[:k], order[-k:]
    rows = [(i, 0) for i in lo] + [(i, 1) for i in hi]
    return "\n\n".join(f"[score={s}]\n```\n{texts[i][:max_chars]}\n```" for i, s in rows)


def _hi_lo_examples_wide(texts, pyes, n_examples=30, max_chars=600) -> str:
    """Demonstrate the criterion with ~n_examples total (half highest-P(YES), half lowest-P(YES)).
    Shows more evidence to the reconstructor to reduce prior collapse."""
    k = max(1, n_examples // 2)
    order = np.argsort(np.nan_to_num(pyes, nan=0.5))
    lo, hi = order[:k], order[-k:]
    rows = [(i, 0) for i in lo] + [(i, 1) for i in hi]
    return "\n\n".join(f"[score={s}]\n```\n{texts[i][:max_chars]}\n```" for i, s in rows)


def _balanced_examples(texts, pyes, k=30, max_chars=600) -> str:
    """Demonstrate the criterion with balanced YES/NO examples. Oversamples the minority class
    to maximize contrast even when M_ω is skewed (e.g., 93% YES). Shows up to k//2 minority
    items (all of them if fewer exist) + fills the rest with majority items (prefer the most
    extreme majority items). Each row is labeled with its true binary score.

    For skewed metrics like capitalization (93% YES on 60 items = ~4 NO), this shows all 4 NO
    + 26 YES for maximum contrast, helping GLM avoid collapsing to a generic prior.
    """
    n = len(texts)
    if n == 0:
        return "(no examples)"

    # Binarize at 0.5 threshold
    binary = (pyes > 0.5).astype(int)

    # Count each class
    n_yes = int((binary == 1).sum())
    n_no = int((binary == 0).sum())

    # Edge case: only one class present
    if n_yes == 0 or n_no == 0:
        # Show k items with the single label, note it
        label = 1 if n_yes > 0 else 0
        k_show = min(k, n)
        order = np.argsort(pyes) if label == 1 else np.argsort(-pyes)
        indices = order[:k_show]
        rows = [(int(i), label) for i in indices]
        note = f"\n\nNOTE: All {n} items have score={label} (metric is constant)"
        return "\n\n".join(f"[score={s}]\n```\n{texts[i][:max_chars]}\n```" for i, s in rows) + note

    # Normal case: both classes present
    minority_class = 0 if n_no < n_yes else 1
    majority_class = 1 - minority_class
    n_minority = n_no if minority_class == 0 else n_yes

    # Take ALL minority items if fewer than k//2, otherwise take k//2
    k_minority = min(k // 2, n_minority)
    k_majority = k - k_minority

    # Get minority indices (any order - we take all available)
    minority_mask = (binary == minority_class)
    minority_indices = np.where(minority_mask)[0][:k_minority]

    # For majority, prefer most extreme items (highest P(YES) if majority=1, lowest if majority=0)
    majority_mask = (binary == majority_class)
    majority_candidates = np.where(majority_mask)[0]

    if majority_class == 1:
        # Sort by descending P(YES)
        majority_sorted = majority_candidates[np.argsort(-pyes[majority_candidates])]
    else:
        # Sort by ascending P(YES) (most confident NOs)
        majority_sorted = majority_candidates[np.argsort(pyes[majority_candidates])]

    majority_indices = majority_sorted[:k_majority]

    # Combine and label with TRUE binary scores
    indices = np.concatenate([minority_indices, majority_indices])
    rows = [(int(i), int(binary[i])) for i in indices]

    # Sort by score for cleaner display (all 0s then all 1s)
    rows.sort(key=lambda x: x[1])

    output = "\n\n".join(f"[score={s}]\n```\n{texts[i][:max_chars]}\n```" for i, s in rows)

    # Add a note if minority class is very small
    if n_minority < k // 2:
        output += f"\n\nNOTE: Only {n_minority} {minority_class}-score items available (showing all)"

    return output


def _format_example_records(records, max_chars=600) -> str:
    """Render a frozen list of ``(item_index, text, binary_score)`` records."""
    if not records:
        return "(no examples)"
    return "\n\n".join(
        f"[score={int(score)}]\n```\n{text[:max_chars]}\n```"
        for _, text, score in records
    )


def _permuted_label_examples(records, seed: int, max_chars=600) -> str:
    """Break text/score association while preserving texts and the score marginal.

    This is a reconstruction-prior control, not a replacement target.  The permutation is
    deterministic from ``seed`` and is forced to differ from the observed assignment whenever
    more than one distinct rearrangement exists.
    """
    if len(records) < 2:
        return _format_example_records(records, max_chars)
    scores = np.asarray([int(r[2]) for r in records], dtype=int)
    rng = np.random.default_rng(seed)
    permuted = scores[rng.permutation(len(scores))]
    if np.array_equal(permuted, scores) and np.unique(scores).size > 1:
        for shift in range(1, len(scores)):
            candidate = np.roll(scores, shift)
            if not np.array_equal(candidate, scores):
                permuted = candidate
                break
    shuffled = [(idx, text, int(score))
                for (idx, text, _), score in zip(records, permuted)]
    return _format_example_records(shuffled, max_chars)


def _exact_contrastive_example_indices(
    target_pyes: np.ndarray,
    distractor_pyes: list[np.ndarray],
    *,
    n_examples: int,
    min_disagreements: int,
    require_target_balance: bool = True,
) -> tuple[np.ndarray, dict]:
    """Exactly choose a max-min separating MCQ teaching set.

    Each distractor is a behavioral hypothesis.  An informative target annotation is one on
    which that hypothesis's hard verdict differs from the target's hard verdict.  The mixed-
    integer program lexicographically maximizes: (1) the minimum number of demonstrated
    disagreements against any distractor, (2) target-label balance, and (3) total separation and
    executor confidence.  Only calibration/design values may be passed here.

    The optimization affects power, not the subsequent evaluation estimand.  Nevertheless, the
    function fails closed unless the returned examples meet the declared minimum against every
    distractor and, when ``require_target_balance`` is true, contain both target labels. Bulk prompt
    audits set it false so degenerate candidates are measured rather than selectively dropped.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    target = np.asarray(target_pyes, dtype=float)
    alternatives = [np.asarray(v, dtype=float) for v in distractor_pyes]
    if target.ndim != 1 or not alternatives:
        raise ValueError("contrastive MCQ design needs one target and at least one distractor")
    if any(v.shape != target.shape for v in alternatives):
        raise ValueError("target and distractor design vectors must be aligned")
    if n_examples < 2 or min_disagreements < 0:
        raise ValueError("n_examples must be >=2 and min_disagreements must be >=0")

    finite = np.isfinite(target)
    for vec in alternatives:
        finite &= np.isfinite(vec)
    valid_idx = np.flatnonzero(finite)
    if len(valid_idx) < n_examples:
        raise ValueError(
            f"only {len(valid_idx)} complete design items for {n_examples} requested examples")

    q = target[valid_idx]
    alt = np.vstack([v[valid_idx] for v in alternatives])
    y = (q > 0.5).astype(int)
    z_alt = (alt > 0.5).astype(int)
    disagreement = (z_alt != y[None, :]).astype(float)
    if require_target_balance and np.unique(y).size < 2:
        raise ValueError("target is constant on the design split; MCQ annotations are uninformative")
    available = disagreement.sum(axis=1).astype(int)
    if np.any(available < min_disagreements):
        bad = np.flatnonzero(available < min_disagreements).tolist()
        raise ValueError(
            "distractors lack the declared design-split separation: "
            f"indices={bad}, available={available.tolist()}")

    n = len(valid_idx)
    k = int(n_examples)
    # Stage 1: maximize the minimum demonstrated disagreement u.
    # Variables are n binary item selectors followed by integer u.
    c1 = np.r_[np.zeros(n), -1.0]
    rows, lower, upper = [], [], []
    rows.append(np.r_[np.ones(n), 0.0]); lower.append(k); upper.append(k)
    for d in disagreement:
        rows.append(np.r_[d, -1.0]); lower.append(0.0); upper.append(np.inf)
    if require_target_balance:
        rows.append(np.r_[y, 0.0]); lower.append(1.0); upper.append(np.inf)
        rows.append(np.r_[1 - y, 0.0]); lower.append(1.0); upper.append(np.inf)
    result1 = milp(
        c1,
        integrality=np.ones(n + 1, dtype=int),
        bounds=Bounds(np.zeros(n + 1), np.r_[np.ones(n), k]),
        constraints=LinearConstraint(np.asarray(rows), np.asarray(lower), np.asarray(upper)),
    )
    if not result1.success:
        raise RuntimeError(f"contrastive example optimization failed at separation stage: {result1.message}")
    min_sep = int(np.floor(result1.x[-1] + 1e-7))

    # Stage 2: at the optimal separation, maximize v=min(# target NO, # target YES).
    c2 = np.r_[np.zeros(n), -1.0]
    rows, lower, upper = [], [], []
    rows.append(np.r_[np.ones(n), 0.0]); lower.append(k); upper.append(k)
    for d in disagreement:
        rows.append(np.r_[d, 0.0]); lower.append(min_sep); upper.append(np.inf)
    rows.append(np.r_[y, -1.0]); lower.append(0.0); upper.append(np.inf)
    rows.append(np.r_[1 - y, -1.0]); lower.append(0.0); upper.append(np.inf)
    result2 = milp(
        c2,
        integrality=np.ones(n + 1, dtype=int),
        bounds=Bounds(np.zeros(n + 1), np.r_[np.ones(n), k]),
        constraints=LinearConstraint(np.asarray(rows), np.asarray(lower), np.asarray(upper)),
    )
    if not result2.success:
        raise RuntimeError(f"contrastive example optimization failed at balance stage: {result2.message}")
    min_class = int(np.floor(result2.x[-1] + 1e-7))

    # Stage 3: break remaining ties by total option separation, then a fixed item-order
    # priority.  The selection rule intentionally depends only on hard annotations.  This
    # makes the reconstruction value a function of the annotation behavior actually shown
    # to the reconstructor, rather than of hidden soft-score confidence.
    priority = (n - np.arange(n, dtype=float)) / (n * (n + 1.0))
    c3 = -(disagreement.sum(axis=0) + priority)
    rows, lower, upper = [], [], []
    rows.append(np.ones(n)); lower.append(k); upper.append(k)
    for d in disagreement:
        rows.append(d); lower.append(min_sep); upper.append(np.inf)
    rows.append(y); lower.append(min_class); upper.append(np.inf)
    rows.append(1 - y); lower.append(min_class); upper.append(np.inf)
    result3 = milp(
        c3,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(np.asarray(rows), np.asarray(lower), np.asarray(upper)),
    )
    if not result3.success:
        raise RuntimeError(f"contrastive example optimization failed at tie-break stage: {result3.message}")

    chosen_local = np.flatnonzero(result3.x > 0.5)
    chosen = valid_idx[chosen_local]
    achieved = disagreement[:, chosen_local].sum(axis=1).astype(int)
    chosen_y = y[chosen_local]
    if (len(chosen) != k or np.any(achieved < min_disagreements)
            or (require_target_balance and np.unique(chosen_y).size < 2)):
        raise RuntimeError("contrastive optimizer returned a teaching set that violates its contract")
    return chosen.astype(int), {
        "solver": "scipy.optimize.milp; exact three-stage lexicographic design",
        "n_complete_design_items": int(n),
        "n_examples": int(k),
        "min_disagreements_required": int(min_disagreements),
        "per_distractor_disagreements_available": available.tolist(),
        "per_distractor_disagreements_demonstrated": achieved.tolist(),
        "minimum_demonstrated_separation": int(achieved.min()),
        "target_example_counts": {
            "0": int(np.sum(chosen_y == 0)),
            "1": int(np.sum(chosen_y == 1)),
        },
        "target_is_degenerate_on_design": bool(np.unique(y).size < 2),
        "target_balance_required": bool(require_target_balance),
        "tie_break": "fixed design-item order; no hidden soft-score dependence",
    }


def _fixed_teaching_panel_design(
    target_pyes: np.ndarray, distractor_pyes: list[np.ndarray]
) -> dict:
    """Describe an already-frozen panel without rerunning candidate-specific MILPs."""
    target = np.asarray(target_pyes, dtype=float)
    alternatives = [np.asarray(value, dtype=float) for value in distractor_pyes]
    if (target.ndim != 1 or not alternatives
            or any(value.shape != target.shape for value in alternatives)
            or np.any(~np.isfinite(target))
            or any(np.any(~np.isfinite(value)) for value in alternatives)):
        raise ValueError("fixed teaching panel requires aligned finite option behaviors")
    target_hard = (target > 0.5).astype(int)
    disagreement = np.vstack([
        (value > 0.5).astype(int) != target_hard for value in alternatives
    ])
    achieved = disagreement.sum(axis=1).astype(int)
    return {
        "solver": "frozen ordered teaching panel; no per-candidate optimization",
        "n_complete_design_items": int(len(target)),
        "n_examples": int(len(target)),
        "min_disagreements_required": 0,
        "per_distractor_disagreements_available": achieved.tolist(),
        "per_distractor_disagreements_demonstrated": achieved.tolist(),
        "minimum_demonstrated_separation": int(achieved.min()),
        "target_example_counts": {
            "0": int(np.sum(target_hard == 0)),
            "1": int(np.sum(target_hard == 1)),
        },
        "target_is_degenerate_on_design": bool(np.unique(target_hard).size < 2),
        "target_balance_required": False,
        "tie_break": "not applicable; item order was frozen before prompt search",
    }


def induce_free(backend, noun, examples, seed, max_tokens=450) -> dict:
    # seed MUST be forwarded: generate() defaults seed=0, so without this every reconstruction is
    # identical (no recovery diversity). generate_batch threads the seed -> R distinct inductions.
    # max_tokens = the articulation budget L, which now drives the DETAIL DIRECTIVE (not just a cap --
    # a bare cap doesn't bind). Hard generation cap sits above L so the targeted detail isn't truncated.
    detail = _detail_directive(max_tokens)
    hard_cap = int(max_tokens * 1.4) + 24
    raw = backend.generate_batch([_RECON_FREE.format(noun=noun, examples=examples, detail=detail)],
                                 system=None, max_tokens=hard_cap, temperature=0.9, seed=seed)[0]
    rule = (raw or "").strip()
    return {"rule": rule, "rubric": rule}


_GEPA_REFINE = (
    "You are refining a rule meant to reproduce a hidden criterion's binary scores on {noun} "
    "excerpts.\n\nCURRENT RULE:\n{rule}\n\nIt got these excerpts WRONG (each shown with the correct "
    "hidden score it should have produced):\n\n{mistakes}\n\nRewrite the rule so it ALSO gets these "
    "right without breaking the rest. {detail} The rule must let a fresh evaluator apply it to a new "
    "excerpt. Reply with ONLY the revised rule, no preamble, no JSON."
)


def induce_gepa(backend, noun, train_texts, train_pyes, m_train, max_chars, l_cap, *,
                pop=5, rounds=2, val_frac=0.4, seed0=0) -> dict:
    """Budget-capped GEPA reconstruction: the L-axis estimator of the SUP over length-L rules.
    Propose `pop` diverse rules (free-gen, capped at l_cap = L) -> SELECT by agreement-to-m on a
    held-from-train validation split -> REFINE the best on its mistakes (still <= L) -> repeat
    `rounds`. Objective IS the articulability target (predict m's verdicts), unlike production
    improve() (composite fidelity). Returns the best length-L rule found."""
    n = len(train_texts)
    rng = np.random.default_rng(13 + seed0)
    order = rng.permutation(n)
    n_val = max(6, int(val_frac * n))
    val_idx, ex_idx = order[:n_val], order[n_val:]
    examples = _hi_lo_examples([train_texts[i] for i in ex_idx], train_pyes[ex_idx], max_chars=max_chars)
    val_texts = [train_texts[i] for i in val_idx]
    val_m = m_train[val_idx]

    def val_agree(rule):
        vv = _verdict_vec(_pyes(backend, rule, val_texts, max_chars))
        ok = np.isfinite(vv) & np.isfinite(val_m)
        return float(np.mean(vv[ok] == val_m[ok])) if ok.any() else float("nan"), vv

    cands = [induce_free(backend, noun, examples, seed0 * 100 + s, max_tokens=l_cap)["rubric"]
             for s in range(pop)]
    cands = [c for c in cands if c]
    best, best_score, best_vv = "", -1.0, None
    for rnd in range(rounds):
        scored = []
        for c in cands:
            a, vv = val_agree(c)
            scored.append((a if np.isfinite(a) else -1.0, c, vv))
        scored.sort(key=lambda x: -x[0])
        if scored and scored[0][0] > best_score:
            best_score, best, best_vv = scored[0][0], scored[0][1], scored[0][2]
        if rnd == rounds - 1 or not scored:
            break
        # refine the leader on the items it misclassified
        top, tvv = scored[0][1], scored[0][2]
        wrong = [i for i in range(len(val_texts))
                 if np.isfinite(tvv[i]) and np.isfinite(val_m[i]) and tvv[i] != val_m[i]]
        mistakes = "\n\n".join(f"[correct score={int(val_m[i])}]\n{val_texts[i][:400]}"
                               for i in wrong[:6]) or "(none — try to sharpen the rule)"
        refined = backend.generate_batch(
            [_GEPA_REFINE.format(noun=noun, rule=top, mistakes=mistakes, detail=_detail_directive(l_cap))],
            system=None, max_tokens=int(l_cap * 1.4) + 24, temperature=0.7, seed=rnd + 1)[0]
        survivors = [s[1] for s in scored[: max(2, pop // 2)]]
        if refined and refined.strip():
            survivors.append(refined.strip())
        survivors.append(induce_free(backend, noun, examples, 777 + rnd, max_tokens=l_cap)["rubric"])
        cands = [c for c in survivors if c]
    rule = best or (cands[0] if cands else "")
    return {"rule": rule, "rubric": rule, "val_agree": best_score}


# --------------------------------------------------------------------------------------------
# Data-driven reconstruction (Codex fix #4, 2026-06-25): the free reconstructor collapses to a generic
# "novelty/quality" prior even when M_ω discriminates (capitalization→"problem formulation", etc. — see
# recovery_trial/REPORT.md "word count recovered as completeness"). Fix: force the reconstructor to test
# cheap surface features against the labels BEFORE writing prose. If a feature dominates, the criterion IS
# that feature; only then fall back to a (specific, non-generic) semantic read.
# --------------------------------------------------------------------------------------------

def _simple_features(texts):
    """Cheap task-agnostic surface features per item (the kinds of spurious drivers a weak executor keys
    on: length, capitalization, code, digits, lexical diversity). Returns (feat_names, X[n_items,n_feat])."""
    import re as _re
    names = ["char_len", "n_words", "n_sent", "avg_word_len", "titlecase_ratio", "digit_ratio",
             "has_url", "has_code", "punct_ratio", "ttr"]
    X = np.zeros((len(texts), len(names)), dtype=float)
    for i, t in enumerate(texts):
        t = t or ""
        words = _re.findall(r"\b\w+\b", t)
        nw = max(1, len(words))
        X[i, 0] = len(t)
        X[i, 1] = len(words)
        X[i, 2] = max(1, len(_re.findall(r"[.!?]+", t)))
        X[i, 3] = float(np.mean([len(w) for w in words])) if words else 0.0
        X[i, 4] = sum(1 for w in words if w[:1].isupper()) / nw          # capitalization signal
        X[i, 5] = sum(c.isdigit() for c in t) / max(1, len(t))
        X[i, 6] = 1.0 if _re.search(r"https?://|www\.", t) else 0.0
        X[i, 7] = 1.0 if ("```" in t or _re.search(r"\b(def|function|return|import)\b|</", t)) else 0.0
        X[i, 8] = sum(c in ".,;:!?-" for c in t) / max(1, len(t))
        X[i, 9] = len(set(w.lower() for w in words)) / nw
    return names, X


def _point_biserial(x, y):
    """Point-biserial r between continuous x and binary y (nan-safe, 0 if degenerate)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4 or len(set(y[m].tolist())) < 2:
        return 0.0
    x, y = x[m], y[m]
    m1 = y == 1
    if not (m1.any() and (~m1).any()):
        return 0.0
    return float((x[m1].mean() - x[~m1].mean()) / (x.std() + 1e-9)
                 * np.sqrt(m1.mean() * (1 - m1.mean())))


def _feat_corr_table(train_texts, m_train):
    """feature↔label correlations (point-biserial), sorted by |r|, as a readable table string."""
    names, X = _simple_features(train_texts)
    rows = sorted(((nm, _point_biserial(X[:, j], m_train)) for j, nm in enumerate(names)),
                  key=lambda r: -abs(r[1]))
    tbl = "\n".join(f"  {nm:18s} r={c:+.2f}" for nm, c in rows)
    return tbl, rows


_RECON_DATADRIVEN = (
    "You are reverse-engineering a hidden evaluation criterion applied to {noun}. You get (excerpt, "
    "label) pairs (1 = satisfies, 0 = does not) AND a table of precomputed surface-feature ↔ label "
    "correlations.\n\n"
    "FEATURE ↔ LABEL CORRELATIONS (point-biserial r; |r|>0.4 = strong):\n{feat_tbl}\n\n"
    "EXAMPLES:\n{examples}\n\n"
    "Follow these steps EXACTLY — do NOT jump to a generic 'quality/novelty' answer:\n"
    "1. If any feature has |r| > 0.4, the criterion is largely THAT surface feature — name it explicitly "
    "(e.g. 'score by length', 'score by capitalization rate').\n"
    "2. If NO feature exceeds 0.4, the criterion is semantic — read the label=1 vs label=0 examples and "
    "name the SPECIFIC separating property (the actual property, not 'quality'/'novelty'/'completeness').\n"
    "3. If labels match no feature and no semantic pattern, say the labels look random.\n"
    "{detail} Reply with ONLY the rule (no preamble, no JSON)."
)


def induce_free_dd(backend, noun, examples, feat_tbl, seed, max_tokens=450):
    """Data-driven reconstruction (less prior-prone than induce_free). `feat_tbl` from _feat_corr_table."""
    detail = _detail_directive(max_tokens)
    hard_cap = int(max_tokens * 1.4) + 24
    raw = backend.generate_batch([_RECON_DATADRIVEN.format(noun=noun, examples=examples,
                                                           feat_tbl=feat_tbl, detail=detail)],
                                 system=None, max_tokens=hard_cap, temperature=0.9, seed=seed)[0]
    rule = (raw or "").strip()
    return {"rule": rule, "rubric": rule}


# --------------------------------------------------------------------------------------------
# Iterative reasoning reconstruction (2026-06-25): multi-turn critique/refine loop to push GLM past
# generic priors by forcing it to confront specific labeled evidence and its own errors.
# --------------------------------------------------------------------------------------------

_REASONING_PROPOSE = (
    "You are reverse-engineering a hidden evaluation criterion applied to {noun}.\n\n"
    "Below are excerpts, each with the binary score (1 = satisfies, 0 = does not) the hidden "
    "criterion assigned.\n\n{examples}\n\n"
    "Propose a specific rule that explains these scores. Do NOT give a generic answer like "
    "'quality' or 'novelty' — look at the ACTUAL pattern in the examples.\n\n"
    "{detail} Reply with ONLY the rule (no preamble, no JSON)."
)


_REASONING_CRITIQUE = (
    "You are refining a rule meant to reproduce a hidden criterion's binary scores on {noun} "
    "excerpts.\n\n"
    "CURRENT RULE:\n{rule}\n\n"
    "Here are the SAME excerpts with their correct scores:\n\n{examples}\n\n"
    "Find ONE excerpt from the list above where your CURRENT RULE would give the WRONG score "
    "(i.e., the rule says score=1 but the correct score is 0, or vice versa). Then revise the rule "
    "to fix that specific mistake.\n\n"
    "If you cannot find any mistake, say 'NO MISTAKE' and repeat the rule unchanged.\n\n"
    "Reply with EITHER:\n"
    "1. The revised rule (if you found a mistake)\n"
    "2. 'NO MISTAKE' followed by the unchanged rule (if the rule already fits)\n\n"
    "Do NOT include preamble or JSON."
)


def induce_reasoning(backend, noun, examples, seed, max_tokens=450, rounds=3) -> dict:
    """Multi-turn iterative reasoning reconstruction: GLM proposes a hypothesis, then is shown the
    SAME examples and asked to find a case it gets WRONG and revise, repeating for several rounds.
    This forces GLM to confront specific labeled evidence rather than defaulting to a generic prior.

    Args:
        backend: LLMBackend (GLM via zai_anthropic)
        noun: item type (e.g., 'review')
        examples: formatted (text,score) pairs from _hi_lo_examples_wide
        seed: random seed for generation
        max_tokens: articulation budget L
        rounds: number of critique/refine iterations (default 3)

    Returns:
        dict with 'rule' (final hypothesis) and 'rubric' (same)
    """
    detail = _detail_directive(max_tokens)
    hard_cap = int(max_tokens * 1.4) + 24

    # Round 1: initial hypothesis
    prompt = _REASONING_PROPOSE.format(noun=noun, examples=examples, detail=detail)
    raw = backend.generate_batch([prompt], system=None, max_tokens=hard_cap,
                                 temperature=0.7, seed=seed)[0]
    rule = (raw or "").strip()

    # Rounds 2+: critique/refine loop
    for rnd in range(1, rounds):
        prompt = _REASONING_CRITIQUE.format(noun=noun, rule=rule, examples=examples)
        raw = backend.generate_batch([prompt], system=None, max_tokens=hard_cap,
                                     temperature=0.7, seed=seed + rnd)[0]
        response = (raw or "").strip()

        # Check if GLM claims "NO MISTAKE"
        if "NO MISTAKE" in response.upper():
            # Extract the rule after "NO MISTAKE"
            parts = response.split("NO MISTAKE")
            if len(parts) > 1:
                rule = parts[-1].strip()
            # else: keep previous rule
        else:
            # Response is the revised rule
            rule = response

        if not rule or len(rule) < 10:
            # Degenerate response - stop iterating
            break

    return {"rule": rule, "rubric": rule}


def induce_mcq(backend, noun, examples, cand_descs, seed) -> dict:
    choices = "\n".join(f"{i + 1}. {d[:200]}" for i, d in enumerate(cand_descs))
    raw = backend.generate_batch([_RECON_MCQ.format(noun=noun, examples=examples, choices=choices,
                                                    m=len(cand_descs))], system=None, max_tokens=20,
                                 temperature=0.9, seed=seed,
                                 validate=lambda s: parse_json_obj(s) is not None)[0]
    obj = parse_json_obj(raw) or {}
    try:
        ch = int(obj.get("choice")) - 1
    except (TypeError, ValueError):
        ch = -1
    return {"choice": ch}


OPTION_ORDER_DESIGN_SCHEMA = "reconstruction-mcq-option-order-block-v1"


def _balanced_option_permutations(n_options: int, n_draws: int, seed: int = 7) -> list[np.ndarray]:
    """Construct a deterministic option-order block.

    When the requested block has exactly ``n_options!`` rows, retain the historical
    seeded cyclic block as its prefix, then append every remaining permutation once in
    lexicographic order. This preserves existing cache keys for the prefix while making
    the complete block exactly factorial. Other sizes retain the historical schedule.
    """
    if n_options < 2 or n_draws < 1:
        raise ValueError("MCQ needs at least two options and one reconstruction draw")
    rng = np.random.default_rng(seed)
    if n_draws == math.factorial(n_options):
        base = rng.permutation(n_options)
        prefix = [tuple(np.roll(base, offset).astype(int)) for offset in range(n_options)]
        seen = set(prefix)
        orders = [*prefix, *(
            permutation for permutation in itertools.permutations(range(n_options))
            if permutation not in seen
        )]
        return [np.asarray(permutation, dtype=int) for permutation in orders]
    permutations = []
    for start in range(0, n_draws, n_options):
        base = rng.permutation(n_options)
        block_size = min(n_options, n_draws - start)
        permutations.extend(np.roll(base, offset) for offset in range(block_size))
    return [np.asarray(p, dtype=int) for p in permutations]


def mcq_option_order_design(
    n_options: int, n_draws: int, seed: int = 7,
) -> dict:
    """Return the complete, auditable design record for an MCQ option-order block."""
    permutations = _balanced_option_permutations(n_options, n_draws, seed=seed)
    orders = [permutation.astype(int).tolist() for permutation in permutations]
    if any(sorted(order) != list(range(n_options)) for order in orders):
        raise RuntimeError("MCQ option-order generator produced a non-permutation")
    unique_orders = {tuple(order) for order in orders}
    factorial_size = math.factorial(n_options)
    full_factorial = (
        len(orders) == factorial_size
        and len(unique_orders) == factorial_size
        and unique_orders == set(itertools.permutations(range(n_options)))
    )
    position_counts = [
        [sum(order[position] == option for order in orders) for position in range(n_options)]
        for option in range(n_options)
    ]
    position_counterbalanced = all(
        len(set(counts)) == 1 for counts in position_counts
    )
    order_sha256 = hashlib.sha256(json.dumps(
        orders, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    core = {
        "schema": OPTION_ORDER_DESIGN_SCHEMA,
        "n_options": int(n_options),
        "n_draws": int(n_draws),
        "factorial_size": int(factorial_size),
        "exact_full_factorial": bool(full_factorial),
        "each_permutation_exactly_once": bool(full_factorial),
        "position_counterbalanced": bool(position_counterbalanced),
        "n_unique_orders": int(len(unique_orders)),
        "generation_rule": (
            "seeded_cyclic_prefix_then_lexicographic_full_factorial"
            if full_factorial else "seeded_cyclic_position_blocks"
        ),
        "generation_seed": int(seed),
        "cache_compatible_cyclic_prefix_size": int(n_options if full_factorial else 0),
        "canonical_option_position_counts": position_counts,
        "canonical_option_orders": orders,
        "orders_sha256": order_sha256,
        "query_seeds_by_condition": {
            "annotations": [10_000 + draw for draw in range(n_draws)],
            "no_demonstrations": [20_000 + draw for draw in range(n_draws)],
            "shuffled_labels": [30_000 + draw for draw in range(n_draws)],
        },
        "shuffled_label_permutation_seeds": [
            91_000 + draw for draw in range(n_draws)
        ],
    }
    return {**core, "design_sha256": hashlib.sha256(json.dumps(
        core, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()}


def mcq_no_demo_choice_probabilities(
    reconstructor,
    *,
    noun: str,
    option_descriptions,
    n_draws: int,
    query_batch_size: int = 512,
) -> dict:
    """Evaluate the blind, position-counterbalanced MCQ menu prior.

    No target identity or candidate-prompt annotation is rendered. Canonical option zero
    is tracked only after scoring so callers can diagnose the target prior without
    disclosing it to the reconstructor.
    """
    descriptions = [str(value) for value in option_descriptions]
    n_options = len(descriptions)
    if n_options < 2 or n_draws < n_options or n_draws % n_options != 0:
        raise ValueError("no-demo calibration requires complete option-position blocks")
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")
    option_order_design = mcq_option_order_design(n_options, n_draws)
    permutations = [
        np.asarray(order, dtype=int)
        for order in option_order_design["canonical_option_orders"]
    ]
    prompts = []
    for permutation in permutations:
        displayed = [descriptions[i] for i in permutation]
        choices = "\n".join(
            f"{i + 1}. {description[:200]}" for i, description in enumerate(displayed))
        prompts.append(_RECON_MCQ_LOGITS.format(
            noun=noun,
            examples=_RECON_MCQ_NO_DEMOS,
            choices=choices,
            labels=", ".join(str(i + 1) for i in range(n_options)),
        ))
    seeds = option_order_design["query_seeds_by_condition"]["no_demonstrations"]
    displayed_rows = []
    choice_labels = [str(i + 1) for i in range(n_options)]
    for start in range(0, len(prompts), query_batch_size):
        stop = min(len(prompts), start + query_batch_size)
        batch = np.asarray(reconstructor.score_choices(
            prompts[start:stop],
            choice_labels,
            system=None,
            seed=seeds[start:stop],
        ), dtype=float)
        if (batch.shape != (stop - start, n_options)
                or np.any(~np.isfinite(batch)) or np.any(batch < 0.0)
                or np.any(batch.sum(axis=1) <= 0.0)):
            raise RuntimeError("invalid no-demo choice-probability batch")
        batch = batch / batch.sum(axis=1, keepdims=True)
        displayed_rows.extend(batch.tolist())

    displayed_probabilities = np.asarray(displayed_rows, dtype=float)
    canonical_probabilities = np.empty_like(displayed_probabilities)
    for draw, permutation in enumerate(permutations):
        canonical_probabilities[draw, permutation] = displayed_probabilities[draw]
    prior = canonical_probabilities.mean(axis=0)
    prior = prior / prior.sum()
    positive = prior > 0.0
    entropy = float(-np.sum(prior[positive] * np.log2(prior[positive])))
    return {
        "canonical_choice_probabilities": canonical_probabilities.tolist(),
        "displayed_choice_probabilities": displayed_probabilities.tolist(),
        "canonical_mean_prior": prior.tolist(),
        "target_probability": float(prior[0]),
        "maximum_option_probability": float(np.max(prior)),
        "normalized_entropy": float(entropy / np.log2(n_options)),
        "total_variation_from_uniform": float(
            0.5 * np.abs(prior - 1.0 / n_options).sum()),
        "n_draws": int(n_draws),
        "position_counterbalanced": bool(option_order_design["position_counterbalanced"]),
        "option_order_design": option_order_design,
        "query_sha256": [hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                         for prompt in prompts],
    }


def _verdict_vec(pyes) -> np.ndarray:
    """P(YES) -> binarized verdict vector (NaN preserved). The unit transmission is computed on,
    so distractor similarity is measured here too (behavioral, not semantic)."""
    pyes = np.asarray(pyes, dtype=float)
    v = (pyes > 0.5).astype(float)
    v[~np.isfinite(pyes)] = np.nan
    return v


def _kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa between two binarized verdict vectors, over items where both are finite.
    Chance-corrected so two near-constant (e.g. all-NO) rubrics don't read as 'similar' for free --
    raw agreement is base-rate-inflated; kappa nets out the marginal."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4:
        return float("nan")
    a, b = a[m], b[m]
    po = float(np.mean(a == b))
    pa, pb = float(np.mean(a)), float(np.mean(b))
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else 0.0


def _candidate_sims(target_vec, pool_pyes, target_id, *, design_indices=None):
    """Behavioral distractor statistics on a declared design split only.

    ``pool_pyes`` remains aligned to the complete item panel so a selected option can later be
    evaluated on a lockbox.  No statistic in this function may inspect outside ``design_indices``.
    """
    target_full = np.asarray(target_vec, dtype=float)
    idx = (np.arange(len(target_full), dtype=int) if design_indices is None
           else np.asarray(design_indices, dtype=int))
    if idx.ndim != 1 or np.any(idx < 0) or np.any(idx >= len(target_full)):
        raise ValueError("design_indices must index target_vec")
    target = target_full[idx]
    sims = []
    for mm, pp in pool_pyes:
        if mm.metric_id == target_id:
            continue
        pp_full = np.asarray(pp, dtype=float)
        if pp_full.shape != target_full.shape:
            continue
        v = _verdict_vec(pp_full)[idx]
        finite = np.isfinite(target) & np.isfinite(v)
        target_hard = target[finite].astype(int)
        candidate_hard = v[finite].astype(int)
        disagree = target_hard != candidate_hard
        sims.append({
            "met": mm,
            "pyes": pp_full,
            "kappa": _kappa(target, v),
            "rate": float(np.nanmean(v)),
            "n_design_complete": int(finite.sum()),
            "n_disagree": int(disagree.sum()),
            "disagreement_rate": float(disagree.mean()) if disagree.size else float("nan"),
            "n_target1_candidate0": int(np.sum((target_hard == 1) & (candidate_hard == 0))),
            "n_target0_candidate1": int(np.sum((target_hard == 0) & (candidate_hard == 1))),
        })
    return [s for s in sims if np.isfinite(s["kappa"])]


def _select_distractors(sims, mode, n, target_rate, *, band=None, rng=None):
    """Pick n distractors from `sims` per a selection rule:
      random            -- uniform (the 'too easy' baseline: distractors likely far from m -> ceiling)
      hard              -- top-n by kappa-to-target (behavioral near-misses -> identification is real)
      matched_base_rate -- closest base rate to m (kills the base-rate giveaway leaked by the demo)
      band              -- n distractors whose kappa is closest to `band` (the graded difficulty dial)
      contrastive       -- nearest non-clones having enough calibration disagreements; demonstration
                           selection separately guarantees those differences are actually shown
    """
    if not sims:
        return []
    if mode == "random":
        idx = rng.permutation(len(sims))[:n]
        return [sims[i] for i in idx]
    if mode == "hard":
        return sorted(sims, key=lambda s: -s["kappa"])[:n]
    if mode == "contrastive":
        return sorted(sims, key=lambda s: (-s["kappa"], -s["n_disagree"], s["met"].metric_id))[:n]
    if mode == "matched_base_rate":
        return sorted(sims, key=lambda s: abs(s["rate"] - target_rate))[:n]
    if mode == "band":
        return sorted(sims, key=lambda s: abs(s["kappa"] - band))[:n]
    raise ValueError(f"unknown distractor mode: {mode}")


def _mcq_recon(backend, noun, examples, example_records, options, gold_idx, held_texts,
               max_chars, R, temperature, desc_map=None, recon=None, run_controls=True,
               reconstruction_temperature=0.9, choice_readout="auto"):
    """Run the selection MCQ, then separately perform optional behavioral replay.

    The reconstructor sees only target ``(text, score)`` demonstrations and option descriptions. It
    never sees candidate-option behaviors. Option positions are counterbalanced. The no-demonstration
    and shuffled-label conditions reuse the identical option order on every draw, isolating semantic/
    position priors from information supplied by the annotations.

    The chosen canonical body is subsequently re-executed on held-out items. That matrix is a separate
    behavioral-equivalence readout; it is not how the MCQ selection is made.
    """
    recon = recon or backend
    if desc_map:
        canonical_descs = [desc_map.get(o.metric_id) or (o.description or o.name) for o in options]
    else:
        canonical_descs = [(o.description or o.name) for o in options]
    n_options = len(options)
    option_order_design = mcq_option_order_design(n_options, R)
    permutations = [
        np.asarray(order, dtype=int)
        for order in option_order_design["canonical_option_orders"]
    ]
    conditions = ["annotations"]
    if run_controls:
        conditions.extend(["no_demonstrations", "shuffled_labels"])

    if choice_readout not in ("auto", "sampled", "logits"):
        raise ValueError("choice_readout must be auto, sampled, or logits")
    use_logits = choice_readout == "logits" or (
        choice_readout == "auto" and callable(getattr(recon, "score_choices", None)))

    def _build_specs(for_logits):
        built = []
        for r, permutation in enumerate(permutations):
            displayed = [canonical_descs[i] for i in permutation]
            choices = "\n".join(f"{i + 1}. {d[:200]}" for i, d in enumerate(displayed))
            condition_examples = {
                "annotations": examples,
                "no_demonstrations": _RECON_MCQ_NO_DEMOS,
                "shuffled_labels": _permuted_label_examples(
                    example_records,
                    seed=option_order_design["shuffled_label_permutation_seeds"][r],
                    max_chars=max_chars),
            }
            for condition in conditions:
                if for_logits:
                    prompt = _RECON_MCQ_LOGITS.format(
                        noun=noun,
                        examples=condition_examples[condition],
                        choices=choices,
                        labels=", ".join(str(i + 1) for i in range(n_options)),
                    )
                else:
                    prompt = _RECON_MCQ.format(
                        noun=noun,
                        examples=condition_examples[condition],
                        choices=choices,
                        m=n_options,
                    )
                built.append((condition, r, permutation, prompt))
        return built

    def _valid_choice(raw):
        obj = parse_json_obj(raw) or {}
        try:
            choice = int(obj.get("choice"))
        except (TypeError, ValueError):
            return False
        return 1 <= choice <= n_options

    picks_by_condition = {condition: np.full(R, -1, dtype=int) for condition in conditions}
    display_picks_by_condition = {condition: np.full(R, -1, dtype=int)
                                  for condition in conditions}
    probabilities_by_condition = {
        condition: np.full((R, n_options), np.nan, dtype=float) for condition in conditions}
    raw_by_condition = {condition: [""] * R for condition in conditions}
    fallback_reason = None

    if use_logits:
        specs = _build_specs(True)
        prompts = [spec[3] for spec in specs]
        seeds = [
            option_order_design["query_seeds_by_condition"][spec[0]][spec[1]]
            for spec in specs
        ]
        try:
            probability_rows = np.asarray(recon.score_choices(
                prompts,
                [str(i + 1) for i in range(n_options)],
                system=None,
                seed=seeds,
            ), dtype=float)
            expected_shape = (len(specs), n_options)
            if probability_rows.shape != expected_shape or np.any(~np.isfinite(probability_rows)):
                raise RuntimeError(
                    f"choice-logit backend returned shape {probability_rows.shape}; "
                    f"expected complete {expected_shape}")
            for (condition, r, permutation, _), displayed_probs in zip(specs, probability_rows):
                displayed_probs = displayed_probs / displayed_probs.sum()
                display_choice = int(np.argmax(displayed_probs))
                canonical_choice = int(permutation[display_choice])
                canonical_probs = np.empty(n_options, dtype=float)
                canonical_probs[permutation] = displayed_probs
                display_picks_by_condition[condition][r] = display_choice
                picks_by_condition[condition][r] = canonical_choice
                probabilities_by_condition[condition][r] = canonical_probs
        except Exception as error:
            if choice_readout == "logits":
                raise
            fallback_reason = repr(error)
            use_logits = False

    if not use_logits:
        specs = _build_specs(False)
        prompts = [spec[3] for spec in specs]
        seeds = [
            option_order_design["query_seeds_by_condition"][spec[0]][spec[1]]
            for spec in specs
        ]
        raw_outputs = recon.generate_batch(
            prompts,
            system=None,
            max_tokens=20,
            temperature=reconstruction_temperature,
            seed=seeds,
            validate=_valid_choice,
        )
        for (condition, r, permutation, _), raw in zip(specs, raw_outputs):
            obj = parse_json_obj(raw) or {}
            try:
                display_choice = int(obj.get("choice")) - 1
            except (TypeError, ValueError):
                display_choice = -1
            canonical_choice = (int(permutation[display_choice])
                                if 0 <= display_choice < n_options else -1)
            display_picks_by_condition[condition][r] = display_choice
            picks_by_condition[condition][r] = canonical_choice
            if canonical_choice >= 0:
                probabilities_by_condition[condition][r] = 0.0
                probabilities_by_condition[condition][r, canonical_choice] = 1.0
            else:
                probabilities_by_condition[condition][r] = 0.0
            raw_by_condition[condition][r] = str(raw or "")[:200]

    main_picks = picks_by_condition["annotations"]
    recon_cols, induced = [], []
    for r, canonical_choice in enumerate(main_picks):
        permutation = permutations[r]
        display_choice = int(display_picks_by_condition["annotations"][r])
        gold_position = int(np.flatnonzero(permutation == gold_idx)[0])
        if 0 <= canonical_choice < n_options:
            induced.append({
                "draw": int(r),
                "choice": int(canonical_choice),
                "display_choice": display_choice,
                "name": options[canonical_choice].name,
                "is_gold": bool(canonical_choice == gold_idx),
                "gold_display_position": gold_position,
                "option_order": [str(options[i].metric_id) for i in permutation],
            })
            recon_cols.append(_sampled_binary(
                backend, options[canonical_choice].body, held_texts, max_chars,
                temperature, r + 1))
        else:
            induced.append({
                "draw": int(r),
                "choice": -1,
                "display_choice": display_choice,
                "name": "?",
                "is_gold": False,
                "gold_display_position": gold_position,
                "option_order": [str(options[i].metric_id) for i in permutation],
            })
            recon_cols.append(np.full(len(held_texts), np.nan))
    recon_mat = np.column_stack(recon_cols)

    def _condition_report(condition):
        picks = picks_by_condition[condition]
        valid = (picks >= 0) & (picks < n_options)
        correct = picks == gold_idx
        probabilities = probabilities_by_condition[condition]
        return {
            "accuracy": float(np.mean(correct)),
            "mean_target_probability": float(np.mean(probabilities[:, gold_idx])),
            "valid_rate": float(np.mean(valid)),
            "accuracy_given_valid": float(np.mean(correct[valid])) if np.any(valid) else None,
            "canonical_choices": picks.tolist(),
            "display_choices": display_picks_by_condition[condition].tolist(),
            "canonical_choice_probabilities": probabilities.tolist(),
            "raw_outputs": raw_by_condition[condition],
        }

    condition_reports = {condition: _condition_report(condition) for condition in conditions}
    main_acc = condition_reports["annotations"]["accuracy"]
    no_demo_acc = (condition_reports.get("no_demonstrations") or {}).get("accuracy")
    shuffled_acc = (condition_reports.get("shuffled_labels") or {}).get("accuracy")
    main_score = condition_reports["annotations"]["mean_target_probability"]
    no_demo_score = (condition_reports.get("no_demonstrations") or {}).get(
        "mean_target_probability")
    shuffled_score = (condition_reports.get("shuffled_labels") or {}).get(
        "mean_target_probability")
    available_controls = [x for x in (no_demo_score, shuffled_score) if x is not None]
    strongest_control = max(available_controls) if available_controls else None
    target_positions = [int(np.flatnonzero(p == gold_idx)[0]) for p in permutations]
    position_counts = {str(pos): int(np.sum(np.asarray(target_positions) == pos))
                       for pos in range(n_options)}
    report = {
        "primary_readout": ("mean normalized target-option probability over counterbalanced prompts"
                            if use_logits else
                            "MCQ option selection accuracy over counterbalanced stochastic draws"),
        "readout_kind": "normalized_choice_logits" if use_logits else "sampled_choice",
        "identification_score": float(main_score),
        "identification_acc": float(main_acc),
        "chance_accuracy": float(1.0 / n_options),
        "no_demonstration_score": no_demo_score,
        "shuffled_label_score": shuffled_score,
        "no_demonstration_acc": no_demo_acc,
        "shuffled_label_acc": shuffled_acc,
        "annotation_lift_over_no_demonstration": (
            float(main_score - no_demo_score) if no_demo_score is not None else None),
        "annotation_lift_over_shuffled_labels": (
            float(main_score - shuffled_score) if shuffled_score is not None else None),
        "annotation_lift_over_strongest_control": (
            float(main_score - strongest_control) if strongest_control is not None else None),
        "conditions": condition_reports,
        "position_counterbalanced": bool(option_order_design["position_counterbalanced"]),
        "option_order_design": option_order_design,
        "target_position_counts": position_counts,
        "n_draws": int(R),
        "behavioral_replay_choice_rule": (
            "argmax normalized choice probability" if use_logits else "sampled MCQ choice"),
        "choice_logit_fallback_reason": fallback_reason,
        "option_codebook": [
            {
                "metric_id": str(option.metric_id),
                "name": str(option.name),
                "description_sha256": hashlib.sha256(
                    str(canonical_descs[i]).encode("utf-8")).hexdigest(),
                "canonical_body_sha256": hashlib.sha256(
                    str(option.body).encode("utf-8")).hexdigest(),
            }
            for i, option in enumerate(options)
        ],
    }
    return recon_mat, induced, report


def mcq_identity_channel(rows, condition="annotations") -> dict:
    """Aggregate per-target MCQ posteriors into the identity channel ``I(J; Jhat)``.

    A single metric has fixed identity, so its target-option probability is a conditional recovery
    score rather than mutual information. Across a declared set of target metrics, this function
    assigns targets equal prior weight and constructs the full channel from the stored canonical
    option probabilities. No external task labels enter. Invalid sampled outputs receive an explicit
    ``__INVALID__`` outcome instead of being silently dropped.
    """
    grouped = collections.defaultdict(list)
    for row in rows:
        target_id = str(row.get("metric_id") or "")
        identification = row.get("identification") or {}
        condition_row = (identification.get("conditions") or {}).get(condition) or {}
        option_ids = [str(x.get("metric_id")) for x in identification.get("option_codebook") or []]
        probabilities = np.asarray(
            condition_row.get("canonical_choice_probabilities") or [], dtype=float)
        if (not target_id or probabilities.ndim != 2 or not option_ids
                or probabilities.shape[1] != len(option_ids)):
            continue
        if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            continue
        grouped[target_id].append((option_ids, probabilities.mean(axis=0)))
    targets = sorted(grouped)
    if len(targets) < 2:
        return {"valid": False, "error": "identity MI requires at least two target metrics"}

    outcome_ids = sorted({option_id for values in grouped.values()
                          for option_ids, _ in values for option_id in option_ids})
    invalid_id = "__INVALID__"
    outcome_ids.append(invalid_id)
    outcome_index = {option_id: i for i, option_id in enumerate(outcome_ids)}
    channel = np.zeros((len(targets), len(outcome_ids)), dtype=float)
    for target_index, target_id in enumerate(targets):
        target_rows = []
        for option_ids, probabilities in grouped[target_id]:
            row = np.zeros(len(outcome_ids), dtype=float)
            for option_id, probability in zip(option_ids, probabilities):
                row[outcome_index[option_id]] += float(probability)
            row[outcome_index[invalid_id]] = max(0.0, 1.0 - row.sum())
            target_rows.append(row)
        channel[target_index] = np.mean(target_rows, axis=0)
        total = channel[target_index].sum()
        if total <= 0.0:
            return {"valid": False, "error": f"target {target_id} has no choice probability mass"}
        channel[target_index] /= total

    joint = channel / len(targets)
    product = np.outer(joint.sum(axis=1), joint.sum(axis=0))
    keep = joint > 0.0
    mi = float(np.sum(joint[keep] * np.log2(joint[keep] / product[keep])))
    target_entropy = float(np.log2(len(targets)))
    target_recovery = float(np.mean([
        channel[i, outcome_index[target_id]] if target_id in outcome_index else 0.0
        for i, target_id in enumerate(targets)
    ]))
    return {
        "valid": True,
        "condition": condition,
        "estimand": "I(J; Jhat) for an equal-weight randomized target metric J",
        "mutual_information_bits": mi,
        "target_entropy_bits": target_entropy,
        "normalized_mutual_information": mi / target_entropy if target_entropy > 0.0 else None,
        "mean_target_recovery_probability": target_recovery,
        "target_ids": targets,
        "outcome_ids": outcome_ids,
        "channel_rows": channel.tolist(),
        "n_targets": len(targets),
        "scope": (
            "closed-codebook identity recovery under the stored target-specific MCQ designs; "
            "not behavioral replay MI and not external validity"),
    }


def mcq_value_from_precomputed_behavior(
    reconstructor,
    *,
    noun: str,
    candidate_prompt_text: str,
    target_metric_id: str,
    target_description: str,
    target_scores: np.ndarray,
    probe_texts: list[str],
    distractors: list[object],
    design_indices: np.ndarray,
    codebook_frozen_before_prompt_search: bool,
    item_ids: list[str] | None = None,
    n_examples: int = 8,
    n_reconstruction_draws: int = 4,
    max_chars: int = 600,
    choice_readout: str = "auto",
    fixed_teaching_panel: bool = False,
) -> dict:
    """Value one prompt from a precomputed executor signature, with no external labels.

    The option codebook is fixed across prompt candidates. Each distractor supplies ``metric_id``,
    ``description``, and an aligned ``scores``/``pyes`` vector; an optional ``body`` is hashed for
    provenance. The candidate's own executor scores generate the only annotations shown to the
    reconstructor. Behaviorally collinear or constant candidates remain in the audit and receive a
    prior-controlled measured value rather than being rejected after seeing their behavior.

    The CR-3 value mark is annotation-attributable lift over the stronger of the no-demonstration and
    shuffled-label controls, clipped to ``[0,1]``. Raw target-option probability is retained alongside
    it. This is a fixed-design metric-level reconstruction value; bank-level identity MI is computed by
    :func:`mcq_identity_channel` across randomized target metrics.
    """
    from types import SimpleNamespace

    if not codebook_frozen_before_prompt_search:
        raise ValueError("MCQ codebook must be frozen before prompt search")
    scores = np.asarray(target_scores, dtype=float)
    if scores.ndim != 1 or len(scores) != len(probe_texts):
        raise ValueError("target_scores must align with probe_texts")
    idx = np.asarray(design_indices, dtype=int)
    if (idx.ndim != 1 or len(idx) < n_examples or len(set(map(int, idx))) != len(idx)
            or np.any(idx < 0) or np.any(idx >= len(scores))):
        raise ValueError("design_indices must be unique valid probe indices covering n_examples")
    if item_ids is None:
        item_ids = [hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:20]
                    for text in probe_texts]
    if len(item_ids) != len(probe_texts):
        raise ValueError("item_ids must align with probe_texts")

    def _get(obj, name, default=None):
        return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)

    options = [SimpleNamespace(
        metric_id=str(target_metric_id),
        name=str(target_description),
        description=str(target_description),
        body=str(candidate_prompt_text),
        meta={},
    )]
    distractor_scores = []
    for position, distractor in enumerate(distractors):
        metric_id = str(_get(distractor, "metric_id", ""))
        description = str(_get(distractor, "description", ""))
        vector = _get(distractor, "scores", _get(distractor, "pyes"))
        vector = np.asarray(vector, dtype=float)
        if not metric_id or not description or vector.shape != scores.shape:
            raise ValueError(f"invalid distractor at position {position}")
        options.append(SimpleNamespace(
            metric_id=metric_id,
            name=description,
            description=description,
            body=str(_get(distractor, "body", description)),
            meta={},
        ))
        distractor_scores.append(vector)
    if len(options) < 2:
        raise ValueError("the frozen MCQ codebook needs at least one distractor")
    if (n_reconstruction_draws < len(options)
            or n_reconstruction_draws % len(options) != 0):
        raise ValueError(
            "n_reconstruction_draws must be a positive multiple of option count")

    if fixed_teaching_panel and len(idx) != n_examples:
        raise ValueError("fixed teaching panel must contain exactly n_examples ordered items")
    if fixed_teaching_panel:
        selected_global = idx
        teaching_design = _fixed_teaching_panel_design(
            scores[idx], [vector[idx] for vector in distractor_scores])
    else:
        selected_local, teaching_design = _exact_contrastive_example_indices(
            scores[idx],
            [vector[idx] for vector in distractor_scores],
            n_examples=n_examples,
            min_disagreements=0,
            require_target_balance=False,
        )
        selected_global = idx[selected_local]
    records = [
        (int(i), str(probe_texts[i]), int(scores[i] > 0.5))
        for i in selected_global
    ]
    if not fixed_teaching_panel:
        order_payload = [
            (str(item_ids[int(i)]), int(scores[int(i)] > 0.5))
            for i in selected_global
        ]
        order_seed = int(hashlib.sha256(json.dumps(
            order_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()[:8], 16)
        order = np.random.default_rng(order_seed).permutation(len(records))
        records = [records[i] for i in order]
    rendered_examples = _format_example_records(records, max_chars=max_chars)

    class _NoReplayExecutor:
        @staticmethod
        def generate_batch(prompts, **_kwargs):
            return [""] * len(prompts)

    _, _, identification = _mcq_recon(
        _NoReplayExecutor(),
        noun,
        rendered_examples,
        records,
        options,
        0,
        [],
        max_chars,
        n_reconstruction_draws,
        0.0,
        recon=reconstructor,
        run_controls=True,
        choice_readout=choice_readout,
    )
    raw_value = float(identification["identification_score"])
    no_demo_value = identification.get("no_demonstration_score")
    if no_demo_value is None or not 0.0 <= float(no_demo_value) <= 1.0:
        raise RuntimeError("Reconstruction-MCQ value requires a valid no-demonstration control")
    # The no-demonstration query contains only the unlabeled, position-counterbalanced
    # frozen codebook. The evaluator tracks which canonical option is the target, but that
    # identity is not disclosed in the query. It is independent of candidate p, so
    # q_annotations <= 1 gives a genuine all-prompt cap on annotation-attributable lift.
    global_value_cap = float(1.0 - float(no_demo_value))
    lift = float(identification["annotation_lift_over_strongest_control"])
    attributable_value = float(np.clip(lift, 0.0, global_value_cap))
    transcript = [
        {"item_id": str(item_ids[record[0]]), "score": int(record[2])}
        for record in records
    ]
    transcript_sha256 = hashlib.sha256(json.dumps(
        transcript, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return {
        "schema": "reconstruction-mcq-prompt-value-v2",
        "metric_id": str(target_metric_id),
        "candidate_prompt_sha256": hashlib.sha256(
            str(candidate_prompt_text).encode("utf-8")).hexdigest(),
        "value_mark": attributable_value,
        "value_name": "annotation-attributable Reconstruction-MCQ target-option lift",
        "value_unit": "probability",
        "value_cap": global_value_cap,
        "all_finite_prompt_cap_reason": "1 - frozen no-demonstration target-option probability",
        "raw_target_option_probability": raw_value,
        "annotation_lift_unclipped": lift,
        "identification": identification,
        "design": {
            "item_ids_in_prompt_order": [str(item_ids[record[0]]) for record in records],
            "indices_in_prompt_order": [int(record[0]) for record in records],
            "scores_in_prompt_order": [int(record[2]) for record in records],
            "teaching_transcript_sha256": transcript_sha256,
            "teaching_set": teaching_design,
            "codebook_frozen_before_prompt_search": True,
            "uses_external_labels": False,
            "behavioral_vectors_shown_to_reconstructor": False,
            "candidate_prompt_text_affects_teaching_order": False,
            "fixed_ordered_teaching_panel": bool(fixed_teaching_panel),
        },
        "scope": (
            "fixed executor signature panel, fixed MCQ codebook, fixed reconstructor and design rule; "
            "no claim about substantive correctness or external validity"),
    }


def mcq_logit_values_from_precomputed_behaviors(
    reconstructor,
    *,
    noun: str,
    candidate_prompt_texts: list[str],
    target_metric_id: str,
    target_description: str,
    target_score_rows: np.ndarray,
    probe_texts: list[str],
    distractors: list[object],
    design_indices: np.ndarray,
    codebook_frozen_before_prompt_search: bool,
    item_ids: list[str] | None = None,
    n_examples: int = 8,
    n_reconstruction_draws: int = 4,
    max_chars: int = 600,
    query_batch_size: int = 512,
    fixed_no_demo_canonical_probabilities: np.ndarray | None = None,
    fixed_teaching_panel: bool = False,
) -> list[dict]:
    """Batched deterministic-logit version of the CR-3 MCQ value instrument.

    It is definitionally identical to :func:`mcq_value_from_precomputed_behavior` with
    ``choice_readout='logits'``. The only execution change is batching: annotation and
    shuffled-control queries from many candidates share one backend call, while the
    candidate-independent no-demonstration queries are evaluated once per metric.
    """
    if not callable(getattr(reconstructor, "score_choices", None)):
        raise ValueError("batched MCQ logit values require score_choices")
    if not codebook_frozen_before_prompt_search:
        raise ValueError("MCQ codebook must be frozen before prompt search")
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")
    rows = np.asarray(target_score_rows, dtype=float)
    texts = [str(text) for text in candidate_prompt_texts]
    if (rows.ndim != 2 or rows.shape != (len(texts), len(probe_texts))
            or np.any(~np.isfinite(rows))):
        raise ValueError("target_score_rows must be a finite candidate-by-probe matrix")
    idx = np.asarray(design_indices, dtype=int)
    if (idx.ndim != 1 or len(idx) < n_examples or len(set(map(int, idx))) != len(idx)
            or np.any(idx < 0) or np.any(idx >= len(probe_texts))):
        raise ValueError("design_indices must be unique valid probe indices covering n_examples")
    if item_ids is None:
        item_ids = [hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:20]
                    for text in probe_texts]
    if len(item_ids) != len(probe_texts):
        raise ValueError("item_ids must align with probe_texts")

    def _get(obj, name, default=None):
        return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)

    option_ids = [str(target_metric_id)]
    option_descriptions = [str(target_description)]
    option_bodies = [None]
    distractor_scores = []
    for position, distractor in enumerate(distractors):
        metric_id = str(_get(distractor, "metric_id", ""))
        description = str(_get(distractor, "description", ""))
        vector = np.asarray(_get(distractor, "scores", _get(distractor, "pyes")), dtype=float)
        if (not metric_id or not description or vector.shape != (len(probe_texts),)
                or np.any(~np.isfinite(vector))):
            raise ValueError(f"invalid distractor at position {position}")
        option_ids.append(metric_id)
        option_descriptions.append(description)
        option_bodies.append(str(_get(distractor, "body", description)))
        distractor_scores.append(vector)
    n_options = len(option_ids)
    if n_options < 2:
        raise ValueError("the frozen MCQ codebook needs at least one distractor")
    if (n_reconstruction_draws < n_options
            or n_reconstruction_draws % n_options != 0):
        raise ValueError("n_reconstruction_draws must be a positive multiple of option count")
    option_order_design = mcq_option_order_design(n_options, n_reconstruction_draws)
    permutations = [
        np.asarray(order, dtype=int)
        for order in option_order_design["canonical_option_orders"]
    ]

    if fixed_teaching_panel and len(idx) != n_examples:
        raise ValueError("fixed teaching panel must contain exactly n_examples ordered items")
    prepared = []
    for candidate_text, scores in zip(texts, rows):
        if fixed_teaching_panel:
            selected_global = idx
            teaching_design = _fixed_teaching_panel_design(
                scores[idx], [vector[idx] for vector in distractor_scores])
        else:
            selected_local, teaching_design = _exact_contrastive_example_indices(
                scores[idx],
                [vector[idx] for vector in distractor_scores],
                n_examples=n_examples,
                min_disagreements=0,
                require_target_balance=False,
            )
            selected_global = idx[selected_local]
        records = [
            (int(i), str(probe_texts[i]), int(scores[i] > 0.5))
            for i in selected_global
        ]
        if not fixed_teaching_panel:
            order_payload = [
                (str(item_ids[int(i)]), int(scores[int(i)] > 0.5))
                for i in selected_global
            ]
            order_seed = int(hashlib.sha256(json.dumps(
                order_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest()[:8], 16)
            order = np.random.default_rng(order_seed).permutation(len(records))
            records = [records[i] for i in order]
        transcript = [
            {"item_id": str(item_ids[record[0]]), "score": int(record[2])}
            for record in records
        ]
        prepared.append({
            "candidate_text": candidate_text,
            "records": records,
            "examples": _format_example_records(records, max_chars=max_chars),
            "teaching_design": teaching_design,
            "transcript": transcript,
            "transcript_sha256": hashlib.sha256(json.dumps(
                transcript, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest(),
        })

    query_prompts: list[str] = []
    query_seeds: list[int] = []
    query_specs: list[tuple[int | None, str, int, np.ndarray]] = []

    def _add_query(candidate_index, condition, draw, permutation, examples):
        displayed = [option_descriptions[i] for i in permutation]
        choices = "\n".join(f"{i + 1}. {description[:200]}"
                            for i, description in enumerate(displayed))
        query_prompts.append(_RECON_MCQ_LOGITS.format(
            noun=noun,
            examples=examples,
            choices=choices,
            labels=", ".join(str(i + 1) for i in range(n_options)),
        ))
        query_seeds.append(
            option_order_design["query_seeds_by_condition"][condition][draw])
        query_specs.append((candidate_index, condition, draw, permutation))

    # The no-demonstration query is independent of candidate p. Bootstrap evaluates it
    # once; subsequent phases may supply that immutable canonical channel verbatim.
    if fixed_no_demo_canonical_probabilities is None:
        for draw, permutation in enumerate(permutations):
            _add_query(None, "no_demonstrations", draw, permutation, _RECON_MCQ_NO_DEMOS)
    for candidate_index, case in enumerate(prepared):
        for draw, permutation in enumerate(permutations):
            _add_query(candidate_index, "annotations", draw, permutation, case["examples"])
            shuffled = _permuted_label_examples(
                case["records"],
                seed=option_order_design["shuffled_label_permutation_seeds"][draw],
                max_chars=max_chars)
            _add_query(candidate_index, "shuffled_labels", draw, permutation, shuffled)

    displayed_probability_rows = []
    choice_labels = [str(i + 1) for i in range(n_options)]
    for start in range(0, len(query_prompts), query_batch_size):
        stop = min(len(query_prompts), start + query_batch_size)
        batch = np.asarray(reconstructor.score_choices(
            query_prompts[start:stop],
            choice_labels,
            system=None,
            seed=query_seeds[start:stop],
        ), dtype=float)
        if batch.shape != (stop - start, n_options) or np.any(~np.isfinite(batch)):
            raise RuntimeError("choice-logit backend returned an incomplete batched MCQ result")
        displayed_probability_rows.extend(batch.tolist())

    conditions = ("annotations", "no_demonstrations", "shuffled_labels")
    canonical = [
        {condition: np.full((n_reconstruction_draws, n_options), np.nan, dtype=float)
         for condition in conditions}
        for _ in prepared
    ]
    displayed = [
        {condition: np.full((n_reconstruction_draws, n_options), np.nan, dtype=float)
         for condition in conditions}
        for _ in prepared
    ]
    common_no_demo_canonical = np.full((n_reconstruction_draws, n_options), np.nan, dtype=float)
    common_no_demo_displayed = np.full((n_reconstruction_draws, n_options), np.nan, dtype=float)
    if fixed_no_demo_canonical_probabilities is not None:
        fixed = np.asarray(fixed_no_demo_canonical_probabilities, dtype=float)
        if (fixed.shape != common_no_demo_canonical.shape or np.any(~np.isfinite(fixed))
                or np.any(fixed < 0.0) or np.any(fixed.sum(axis=1) <= 0.0)):
            raise ValueError("invalid frozen no-demonstration choice channel")
        common_no_demo_canonical = fixed.copy()
        for draw, permutation in enumerate(permutations):
            common_no_demo_displayed[draw] = common_no_demo_canonical[draw, permutation]
    for spec, raw_probability in zip(query_specs, displayed_probability_rows):
        candidate_index, condition, draw, permutation = spec
        row = np.asarray(raw_probability, dtype=float)
        total = float(row.sum())
        if total <= 0.0:
            raise RuntimeError("choice-logit row has no declared-option probability mass")
        row = row / total
        canonical_row = np.empty(n_options, dtype=float)
        canonical_row[permutation] = row
        if candidate_index is None:
            common_no_demo_canonical[draw] = canonical_row
            common_no_demo_displayed[draw] = row
        else:
            canonical[candidate_index][condition][draw] = canonical_row
            displayed[candidate_index][condition][draw] = row
    for candidate_index in range(len(prepared)):
        canonical[candidate_index]["no_demonstrations"] = common_no_demo_canonical.copy()
        displayed[candidate_index]["no_demonstrations"] = common_no_demo_displayed.copy()

    def _condition_report(canonical_probabilities, displayed_probabilities):
        canonical_picks = np.argmax(canonical_probabilities, axis=1)
        display_picks = np.argmax(displayed_probabilities, axis=1)
        return {
            "accuracy": float(np.mean(canonical_picks == 0)),
            "mean_target_probability": float(np.mean(canonical_probabilities[:, 0])),
            "valid_rate": 1.0,
            "accuracy_given_valid": float(np.mean(canonical_picks == 0)),
            "canonical_choices": canonical_picks.astype(int).tolist(),
            "display_choices": display_picks.astype(int).tolist(),
            "canonical_choice_probabilities": canonical_probabilities.tolist(),
            "raw_outputs": [""] * n_reconstruction_draws,
        }

    target_positions = [int(np.flatnonzero(permutation == 0)[0])
                        for permutation in permutations]
    position_counts = {str(position): int(np.sum(np.asarray(target_positions) == position))
                       for position in range(n_options)}
    results = []
    for candidate_index, case in enumerate(prepared):
        condition_reports = {
            condition: _condition_report(
                canonical[candidate_index][condition], displayed[candidate_index][condition])
            for condition in conditions
        }
        main_score = condition_reports["annotations"]["mean_target_probability"]
        no_demo_score = condition_reports["no_demonstrations"]["mean_target_probability"]
        shuffled_score = condition_reports["shuffled_labels"]["mean_target_probability"]
        strongest_control = max(no_demo_score, shuffled_score)
        report = {
            "primary_readout": "mean normalized target-option probability over counterbalanced prompts",
            "readout_kind": "normalized_choice_logits",
            "identification_score": float(main_score),
            "identification_acc": condition_reports["annotations"]["accuracy"],
            "chance_accuracy": float(1.0 / n_options),
            "no_demonstration_score": float(no_demo_score),
            "shuffled_label_score": float(shuffled_score),
            "no_demonstration_acc": condition_reports["no_demonstrations"]["accuracy"],
            "shuffled_label_acc": condition_reports["shuffled_labels"]["accuracy"],
            "annotation_lift_over_no_demonstration": float(main_score - no_demo_score),
            "annotation_lift_over_shuffled_labels": float(main_score - shuffled_score),
            "annotation_lift_over_strongest_control": float(main_score - strongest_control),
            "conditions": condition_reports,
            "position_counterbalanced": bool(option_order_design["position_counterbalanced"]),
            "option_order_design": option_order_design,
            "target_position_counts": position_counts,
            "n_draws": int(n_reconstruction_draws),
            "behavioral_replay_choice_rule": "argmax normalized choice probability",
            "choice_logit_fallback_reason": None,
            "batched_choice_queries": True,
            "option_codebook": [
                {
                    "metric_id": option_ids[i],
                    "name": option_descriptions[i],
                    "description_sha256": hashlib.sha256(
                        option_descriptions[i].encode("utf-8")).hexdigest(),
                    "canonical_body_sha256": hashlib.sha256(
                        (case["candidate_text"] if i == 0 else option_bodies[i]).encode(
                            "utf-8")).hexdigest(),
                }
                for i in range(n_options)
            ],
        }
        global_value_cap = float(1.0 - no_demo_score)
        lift = float(report["annotation_lift_over_strongest_control"])
        value = float(np.clip(lift, 0.0, global_value_cap))
        records = case["records"]
        results.append({
            "schema": "reconstruction-mcq-prompt-value-v2",
            "metric_id": str(target_metric_id),
            "candidate_prompt_sha256": hashlib.sha256(
                case["candidate_text"].encode("utf-8")).hexdigest(),
            "value_mark": value,
            "value_name": "annotation-attributable Reconstruction-MCQ target-option lift",
            "value_unit": "probability",
            "value_cap": global_value_cap,
            "all_finite_prompt_cap_reason": (
                "1 - frozen no-demonstration target-option probability"),
            "raw_target_option_probability": float(main_score),
            "annotation_lift_unclipped": lift,
            "identification": report,
            "design": {
                "item_ids_in_prompt_order": [str(item_ids[record[0]]) for record in records],
                "indices_in_prompt_order": [int(record[0]) for record in records],
                "scores_in_prompt_order": [int(record[2]) for record in records],
                "teaching_transcript_sha256": case["transcript_sha256"],
                "teaching_set": case["teaching_design"],
                "codebook_frozen_before_prompt_search": True,
                "uses_external_labels": False,
                "behavioral_vectors_shown_to_reconstructor": False,
                "candidate_prompt_text_affects_teaching_order": False,
                "fixed_ordered_teaching_panel": bool(fixed_teaching_panel),
            },
            "scope": (
                "fixed executor signature panel, fixed MCQ codebook, fixed reconstructor and "
                "design rule; no claim about substantive correctness or external validity"),
        })
    return results


def _ivs(recon_mat, m_verdict_held, cons_iv, metric, pyes, mode, *, extra=None) -> dict:
    """Common I_V readout block for a recovered-verdict matrix (free or one MCQ option set)."""
    iv_recon = vinfo.iv_from_reconstruction(recon_mat, orig_verdicts=m_verdict_held)
    iv_trans = vinfo.iv_transmission(recon_mat, m_verdict_held)   # I(m; m_recovered): articulability
    fixed_target = vinfo.fixed_target_channel_certificate(m_verdict_held, recon_mat)
    row = {
        "metric_id": metric.metric_id, "name": metric.name, "mode": mode,
        "n_held": int(recon_mat.shape[0]), "R": int(recon_mat.shape[1]),
        # ARTICULABILITY = transmission I(m; m_recovered), the capability-robust signal
        "iv_transmission": iv_trans.get("iv_mm"),
        "iv_transmission_ci": [iv_trans.get("ci_lo"), iv_trans.get("ci_hi")],
        "transmission_norm": iv_trans.get("transmission_norm"),
        # I(x -> m_recovered): the recovered rule's discrimination (capability-bound covariate)
        "iv_recon": iv_recon.get("iv_mm"), "iv_recon_ci": [iv_recon.get("ci_lo"), iv_recon.get("ci_hi")],
        "fidelity_to_m_pearson": iv_recon.get("agree_orig_pearson"),
        "iv_consistency_held": cons_iv,
        # Same-f, same-heldout fixed-target DPI object. This is the certificate payload; the
        # percentile intervals above remain descriptive estimator intervals.
        "fixed_target_certificate": fixed_target,
        "tvd_recovery": fixed_target.get("tvd", {}).get("R"),
        "tvd_target_ceiling": fixed_target.get("tvd", {}).get("T_target"),
        "m_pyes_mean": float(np.nanmean(pyes)), "m_pyes_std": float(np.nanstd(pyes)),
        "subtask_breadth": (metric.meta or {}).get("subtask_breadth"),
        "estimand_role": "achieved reconstruction-channel value; not a prompt-space upper bound",
    }
    if extra:
        row.update(extra)
    if mode == "mcq":
        row["behavioral_replay_mi_bits"] = iv_trans.get("iv_mm")
        row["behavioral_replay_mi_ci"] = [iv_trans.get("ci_lo"), iv_trans.get("ci_hi")]
        row["estimand_role"] = (
            "MCQ identification is the primary reconstruction measure; behavioral replay MI is "
            "a secondary equivalence readout, not the selection mechanism or an upper bound")
    return row


def run_metric(backend, noun, metric, all_metrics, texts, pyes, *, R, n_train, mode,
               max_chars, temperature=0.7, l_cap=450, distractor="contrastive", pool_pyes=None,
               n_options=4, graded_bands=(0.25, 0.5, 0.75, 0.95), desc_map=None,
               induce="free", gepa_pop=5, gepa_rounds=2, n_examples=4, example_mode="balanced",
               recon_backend=None, mcq_n_examples=8, mcq_min_design_disagreements=2,
               mcq_min_demo_disagreements=2, mcq_controls=True, split_seed=0,
               item_ids=None, mcq_choice_readout="auto") -> list:
    """Recover `metric` through the articulation bottleneck. Returns a LIST of result rows:
    one row for free / a single pinned MCQ option set, or one row per difficulty band for
    distractor='graded'.

    Two-backend split (prompt-optimality real-test architecture): ``backend`` is the EXECUTOR
    (the target model X, vLLM) and does ALL scoring — ``_pyes``/``_sampled_binary``/``cons_mat``
    (the metric's verdicts + re-execution of the induced rule). ``recon_backend`` (default =
    ``backend``) is the RECONSTRUCTOR — a strong model (GLM via z.ai) that induces the rule via
    ``induce_free``/``induce_gepa``/``induce_mcq``/``induce_reasoning``. So GLM articulates, X executes — recovery R =
    I(M_X; exec(M̂_GLM)) is X re-expressing what GLM read from X's labels, with no anchor M.

    Args:
        example_mode: "balanced" (default, oversamples minority class for maximum contrast on skewed
                     metrics) or "hi_lo" (legacy, shows k highest + k lowest P(YES) items)."""
    recon = recon_backend or backend          # default: single-backend (dev path)
    n = len(texts)
    if not 1 <= n_train < n:
        raise ValueError("n_train must leave nonempty design and held-out splits")
    if len(pyes) != n:
        raise ValueError("pyes must align with texts")
    if item_ids is None:
        item_ids = [hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:20] for text in texts]
    if len(item_ids) != n:
        raise ValueError("item_ids must align with texts")
    m_verdict = (pyes > 0.5).astype(float)
    m_verdict[~np.isfinite(pyes)] = np.nan
    rng = np.random.default_rng(split_seed)
    order = rng.permutation(n)
    train_idx, held_idx = order[:n_train], order[n_train:]
    held_texts = [texts[i] for i in held_idx]
    m_held = m_verdict[held_idx]
    train_texts = [texts[i] for i in train_idx]
    m_train = m_verdict[train_idx]

    examples = None
    if mode == "free":
        # Free reconstruction retains its historical example builders. MCQ examples are selected
        # only after its option set is frozen, using the contrastive design below.
        if example_mode == "balanced":
            examples = _balanced_examples(
                train_texts, pyes[train_idx], k=n_examples, max_chars=max_chars)
        elif induce == "reasoning":
            examples = _hi_lo_examples_wide(
                train_texts, pyes[train_idx], n_examples=n_examples, max_chars=max_chars)
        else:
            examples = _hi_lo_examples(
                train_texts, pyes[train_idx], k=n_examples, max_chars=max_chars)

    # consistency baseline on the SAME held items (seed rubric, R sampled passes) -- computed once.
    cons_mat = np.column_stack([_sampled_binary(backend, metric.body, held_texts, max_chars,
                                                temperature, r + 1) for r in range(R)])
    cons_iv = vinfo.cell_report(cons_mat, n_boot=300, min_items=0).get("iv_mm")

    if mode == "free":
        feat_tbl = None
        if induce == "free_dd":                       # data-driven: feature↔label table (Codex fix #4)
            feat_tbl, _ = _feat_corr_table(train_texts, m_train)
        recon_cols, induced = [], []
        for r in range(R):
            if induce == "gepa":
                ind = induce_gepa(recon, noun, train_texts, pyes[train_idx], m_train,
                                  max_chars, l_cap, pop=gepa_pop, rounds=gepa_rounds, seed0=r)
                induced.append({"rule": ind["rule"][:200], "rubric_len": len(ind["rubric"]),
                                "val_agree": ind.get("val_agree")})
            elif induce == "free_dd":
                ind = induce_free_dd(recon, noun, examples, feat_tbl, r, max_tokens=l_cap)
                induced.append({"rule": ind["rule"][:200], "rubric_len": len(ind["rubric"])})
            elif induce == "reasoning":
                ind = induce_reasoning(recon, noun, examples, r, max_tokens=l_cap, rounds=3)
                induced.append({"rule": ind["rule"][:200], "rubric_len": len(ind["rubric"])})
            else:
                ind = induce_free(recon, noun, examples, r, max_tokens=l_cap)
                induced.append({"rule": ind["rule"][:200], "rubric_len": len(ind["rubric"])})
            if not ind["rubric"]:
                recon_cols.append(np.full(len(held_idx), np.nan)); continue
            recon_cols.append(_sampled_binary(backend, ind["rubric"], held_texts, max_chars,
                                              temperature, r + 1))
        recon_mat = np.column_stack(recon_cols)
        return [_ivs(recon_mat, m_held, cons_iv, metric, pyes, "free",
                     extra={"induced": induced, "l_cap": l_cap, "induce": induce})]

    # ---- MCQ: design on train/calibration only; freeze before held-out replay ----
    if n_options < 2:
        raise ValueError("MCQ requires at least two options")
    if mcq_n_examples < mcq_min_demo_disagreements * (n_options - 1) + 1:
        raise ValueError(
            "mcq_n_examples is too small for the worst-case separation contract; require "
            "min_demo_disagreements*(n_options-1)+1")
    rng_d = np.random.default_rng(7)
    all_sims = _candidate_sims(
        m_verdict, pool_pyes or [], metric.metric_id, design_indices=train_idx)
    nonclones = [s for s in all_sims if s["kappa"] < CLONE_CAP]
    sims = [s for s in nonclones
            if s["n_disagree"] >= mcq_min_design_disagreements]
    n_clones = len(all_sims) - len(nonclones)
    n_underidentified = len(nonclones) - len(sims)
    if len(sims) < n_options - 1:
        raise ValueError(
            f"only {len(sims)} identifiable distractors for {n_options - 1} required "
            f"(clones={n_clones}, below separation floor={n_underidentified})")
    target_rate = float(np.nanmean(m_verdict[train_idx]))

    def _one_set(sel, tag):
        if len(sel) != n_options - 1:
            raise ValueError(f"MCQ option construction returned {len(sel)} distractors; "
                             f"expected {n_options - 1}")
        # Canonical order is target first. Display order is counterbalanced independently inside
        # _mcq_recon and recorded draw by draw.
        options = [metric] + [s["met"] for s in sel]
        gold_idx = 0
        selected_local, teaching_design = _exact_contrastive_example_indices(
            np.asarray(pyes, dtype=float)[train_idx],
            [np.asarray(s["pyes"], dtype=float)[train_idx] for s in sel],
            n_examples=mcq_n_examples,
            min_disagreements=mcq_min_demo_disagreements,
        )
        selected_global = train_idx[selected_local]
        records = [
            (int(i), str(texts[i]), int(m_verdict[i]))
            for i in selected_global
        ]
        # Randomize presentation order after exact set selection. This seed depends only on frozen
        # identifiers, not on held-out behavior.
        order_seed = int(hashlib.sha256(
            (str(metric.metric_id) + "|" + tag).encode("utf-8")).hexdigest()[:8], 16)
        record_order = np.random.default_rng(order_seed).permutation(len(records))
        records = [records[i] for i in record_order]
        examples_mcq = _format_example_records(records, max_chars=max_chars)

        # S remains a descriptive option-set difficulty statistic, now computed only on design data.
        S = max((s["kappa"] for s in sel), default=float("nan"))
        recon_mat, induced, identification = _mcq_recon(
            backend, noun, examples_mcq, records, options, gold_idx,
            held_texts, max_chars, R, temperature, desc_map=desc_map,
            recon=recon, run_controls=mcq_controls, choice_readout=mcq_choice_readout)
        id_acc = identification["identification_acc"]
        id_score = identification["identification_score"]
        M = len(options)
        id_cc = (id_acc - 1.0 / M) / (1.0 - 1.0 / M) if (M > 1 and np.isfinite(id_acc)) else float("nan")
        distractor_stats = [
            {k: v for k, v in s.items() if k not in ("met", "pyes")}
            | {"metric_id": str(s["met"].metric_id), "name": str(s["met"].name)}
            for s in sel
        ]
        recovered_raw = [[None if not np.isfinite(v) else int(v) for v in row]
                         for row in recon_mat]
        validity_warnings = []
        if not identification["position_counterbalanced"]:
            validity_warnings.append(
                "R is not a multiple of n_options; target positions differ by at most one draw")
        if not mcq_controls:
            validity_warnings.append("semantic-prior controls were explicitly disabled")
        return _ivs(recon_mat, m_held, cons_iv, metric, pyes, "mcq",
                    extra={"distractor": tag, "n_options": M, "option_set_S": S,
                           "identification_score": id_score,
                           "identification_acc": id_acc, "identification_cc": id_cc,
                           "identification": identification,
                           "normalized": bool(desc_map), "n_clones_excluded": int(n_clones),
                           "n_underidentified_excluded": int(n_underidentified),
                           "mcq_bound_grade": False,
                           "mcq_measurement_grade": bool(mcq_controls),
                           "mcq_role": (
                               "anchor-free closed-codebook identification measurement; "
                               "behavioral replay is a separate secondary readout"),
                           "primary_reconstruction_score": id_score,
                           "primary_reconstruction_readout": identification["primary_readout"],
                           "annotation_attributable_lift": identification.get(
                               "annotation_lift_over_strongest_control"),
                           "mcq_validity_warnings": validity_warnings,
                           "mcq_design": {
                               "estimand": "active contrastive teaching-set reconstruction",
                               "behavior_use": (
                                   "design-split option screening and example selection only"),
                               "split_seed": int(split_seed),
                               "design_item_ids": [str(item_ids[i]) for i in train_idx],
                               "heldout_item_ids": [str(item_ids[i]) for i in held_idx],
                               "example_item_ids_in_prompt_order": [
                                   str(item_ids[record[0]]) for record in records],
                               "example_indices_in_prompt_order": [
                                   int(record[0]) for record in records],
                               "teaching_set": teaching_design,
                               "distractors": distractor_stats,
                               "min_design_disagreements": int(mcq_min_design_disagreements),
                               "heldout_used_for_design": False,
                               "options_frozen_before_heldout_replay": True,
                           },
                           "mcq_raw": {
                               "heldout_target_verdicts": [
                                   None if not np.isfinite(v) else int(v) for v in m_held],
                               "heldout_recovered_verdicts": recovered_raw,
                           },
                           "induced": induced})

    if distractor != "graded":
        sel = _select_distractors(sims, distractor, n_options - 1, target_rate, rng=rng_d)
        return [_one_set(sel, distractor)]

    # graded: sweep band centers across the available kappa range -> transmission-vs-difficulty curve
    ks = [s["kappa"] for s in sims]
    rows = []
    for q in graded_bands:
        center = float(np.quantile(ks, q))
        sel = _select_distractors(sims, "band", n_options - 1, target_rate, band=center, rng=rng_d)
        rows.append(_one_set(sel, f"band@q{int(q * 100):02d}"))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--tasks", default="math_se,peer_review")
    ap.add_argument("--n-metrics", type=int, default=3, help="discriminating metrics to reconstruct")
    ap.add_argument("--n-scan", type=int, default=12, help="seed pool to P(YES)-scan")
    ap.add_argument("--n-items", type=int, default=60)
    ap.add_argument("--n-train", type=int, default=30)
    ap.add_argument("--R", type=int, default=20,
                    help="reconstruction draws; for MCQ use a multiple of --n-options")
    ap.add_argument("--mode", default="free", choices=["free", "mcq"])
    ap.add_argument("--distractor", default="contrastive",
                    choices=["contrastive", "random", "hard", "matched_base_rate", "graded"],
                    help="MCQ option-set selection: contrastive (default) = nearest design-split "
                         "non-clones with guaranteed teaching examples; legacy modes remain "
                         "diagnostics and are also restricted to the design split")
    ap.add_argument("--n-options", type=int, default=4, help="MCQ options per item (= 1 true + n-1 distractors); chance=1/n")
    ap.add_argument("--mcq-examples", type=int, default=8,
                    help="contrastively selected scored demonstrations shown in MCQ mode")
    ap.add_argument("--mcq-min-disagreements", type=int, default=2,
                    help="minimum target/distractor disagreements both available and demonstrated")
    ap.add_argument("--mcq-choice-readout", default="auto",
                    choices=["auto", "logits", "sampled"],
                    help="auto uses normalized choice logits when the reconstructor exposes them")
    ap.add_argument("--normalize-options", action="store_true",
                    help="paraphrase MCQ option descriptions to a uniform register (kills the "
                         "phrasing-distinctiveness confound; exploratory because the paraphrase is not "
                         "the executable rubric body)")
    ap.add_argument("--recon-max-tokens", type=int, default=450,
                    help="articulation budget L: token cap on the induced rule (sweep for L-axis)")
    ap.add_argument("--std-floor", type=float, default=0.12, help="min P(YES) std to count as discriminating")
    ap.add_argument("--metric-ids", default="", help="comma-sep metric_ids to FIX across tiers "
                    "(transmission(E) needs the same metrics at every E); overrides std-selection")
    ap.add_argument("--induce", default="free", choices=["free", "gepa"],
                    help="free = single-shot reconstruction; gepa = budget-capped propose/select/"
                         "refine loop (estimates the SUP over length-L rules -> the L-axis)")
    ap.add_argument("--gepa-pop", type=int, default=5, help="GEPA candidates proposed per round")
    ap.add_argument("--gepa-rounds", type=int, default=2, help="GEPA refine rounds")
    ap.add_argument("--l-sweep", default="", help="comma-sep L (token) budgets -> one row per (metric,L); "
                    "the articulation-budget axis I_{V_{L,E}}(X->m). Overrides --recon-max-tokens.")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--out", default="/lfs/skampere3/0/alexspan/tmp_vinfo/recon_results.json")
    # two-backend split (real-test architecture): executor X (--model, vLLM) scores; the reconstructor
    # (--reconstructor-model via --reconstructor-backend, default GLM/z.ai) induces the rule. Default
    # None = single-backend (the original dev path: X does both).
    ap.add_argument("--reconstructor-backend", default=None,
                    choices=[None, "openrouter", "zai", "zai_anthropic"],
                    help="API backend for the reconstructor (None = use the executor for both; "
                         "'zai_anthropic' = glm-5 via z.ai subscription (free); 'zai' = PaaS prepaid)")
    ap.add_argument("--reconstructor-model", default="glm-5",
                    help="reconstructor model slug on the API backend (e.g. glm-5)")
    args = ap.parse_args(argv)

    man = full_manifest(metrics_per_task=max(args.n_scan, args.n_metrics), metric_files_cap=400)
    by_name = {e.name: e for e in man.datasets}
    cfg0 = ImplementerConfig()
    if args.fake:
        cfg0.vllm_fake = True
    backend = make_judge_backend(args.model, cfg0, temperature=None)  # executor X (vLLM), resident
    recon_backend = None
    if args.reconstructor_backend:
        from .backends import LLMBackend
        rcfg = ImplementerConfig(); rcfg.backend = args.reconstructor_backend
        recon_backend = LLMBackend(args.reconstructor_model, "reconstructor", rcfg)
        print(f"reconstructor: {args.reconstructor_model} via {args.reconstructor_backend} "
              f"(executor X = {args.model} via vLLM)")

    results = []
    for tname in args.tasks.split(","):
        entry = by_name.get(tname) or next((e for e in man.datasets if tname in e.name), None)
        if entry is None:
            print("SKIP unknown task", tname); continue
        cfg = ImplementerConfig(); apply_task_preset(cfg, entry.task)
        noun = getattr(cfg, "item_noun", entry.task.replace("_", " "))
        max_chars = getattr(cfg, "max_text_chars", 4000)
        full = load_metrics(entry)
        idmap = {m.metric_id: m for m in full}
        pool = full[: args.n_scan]                       # MCQ candidate pool
        texts, corpus_ids = load_corpus(entry, args.n_items, seed=7)
        selection_idx = np.random.default_rng(0).permutation(len(texts))[:args.n_train]

        want = [i for i in args.metric_ids.split(",") if i in idmap] if args.metric_ids else []
        # pool_pyes: behavioral verdict vectors for the whole candidate pool -> MCQ distractor mining.
        pool_pyes = None
        if want:
            # FIXED metric set across tiers (valid transmission(E)). Measure all; flag degenerate.
            print(f"\n=== {tname} ({entry.task}) — FIXED {len(want)} metrics, {len(texts)} items ===")
            chosen = []
            for i in want:
                py = _pyes(backend, idmap[i].body, texts, max_chars)
                chosen.append((idmap[i], py, float(np.nanmean(py[selection_idx])),
                               float(np.nanstd(py[selection_idx]))))
            print("  P(YES) mean/std:", [f"{s[2]:.2f}/{s[3]:.2f}" for s in chosen])
            if args.mode == "mcq":   # need a distractor pool too; scan it
                scan = [(met, _pyes(backend, met.body, texts, max_chars)) for met in pool]
                pool_pyes = [(met, py) for met, py in scan]
        else:
            # P(YES) pre-scan: a metric DISCRIMINATES iff P(YES) varies across items (std).
            print(f"\n=== {tname} ({entry.task}) — scanning {len(pool)} seeds, {len(texts)} items ===")
            scan = [(met, py, float(np.nanmean(py[selection_idx])),
                     float(np.nanstd(py[selection_idx])))
                    for met in pool for py in [_pyes(backend, met.body, texts, max_chars)]]
            print("  P(YES) mean/std:", [f"{s[2]:.2f}/{s[3]:.2f}" for s in scan])
            disc = sorted([s for s in scan if np.isfinite(s[3]) and s[3] >= args.std_floor
                           and 0.1 <= s[2] <= 0.9], key=lambda s: -s[3])
            print(f"  -> {len(disc)}/{len(scan)} discriminating (std>={args.std_floor})")
            chosen = disc[: args.n_metrics]
            pool_pyes = [(s[0], s[1]) for s in scan]

        # Normalize option descriptions ONCE per task (uniform register) for the realistic MCQ bound.
        desc_map = None
        if args.mode == "mcq" and args.normalize_options and pool_pyes:
            uniq = {}
            for met, _ in pool_pyes:
                uniq[met.metric_id] = met.description or met.name
            for met, *_ in chosen:
                uniq[met.metric_id] = met.description or met.name
            ids = list(uniq)
            normed = _normalize_descs(backend, noun, [uniq[i] for i in ids])
            desc_map = dict(zip(ids, normed))
            print(f"  normalized {len(desc_map)} option descriptions (uniform register)")

        l_grid = [int(x) for x in args.l_sweep.split(",") if x.strip()] or [args.recon_max_tokens]
        for met, py, mean, std in chosen:
            for L in l_grid:
                t0 = time.time()
                rows = run_metric(backend, noun, met, pool, texts, py, R=args.R, n_train=args.n_train,
                                  mode=args.mode, max_chars=max_chars, l_cap=L,
                                  distractor=args.distractor, pool_pyes=pool_pyes,
                                  n_options=args.n_options, desc_map=desc_map,
                                  induce=args.induce, gepa_pop=args.gepa_pop, gepa_rounds=args.gepa_rounds,
                                  recon_backend=recon_backend,
                                  mcq_n_examples=args.mcq_examples,
                                  mcq_min_design_disagreements=args.mcq_min_disagreements,
                                  mcq_min_demo_disagreements=args.mcq_min_disagreements,
                                  item_ids=corpus_ids,
                                  mcq_choice_readout=args.mcq_choice_readout)
                nz = lambda x: float("nan") if x is None else x
                for r in rows:
                    r["task"] = tname
                    results.append(r)
                    tr = nz(r["iv_transmission"]); trc = [nz(c) for c in r["iv_transmission_ci"]]
                    tag = (f" [{r['distractor']} S={nz(r.get('option_set_S')):.2f} "
                           f"id={nz(r.get('identification_acc')):.2f} idcc={nz(r.get('identification_cc')):.2f} "
                           f"m={r.get('n_options')}]"
                           if r["mode"] == "mcq" else f" [L={L} {args.induce}]")
                    print(f"  [{met.metric_id[:24]:24s}]{tag} I(m;m^)={tr:.3f} ci=[{trc[0]:.2f},{trc[1]:.2f}] "
                          f"Tnorm={nz(r.get('transmission_norm')):.2f} | pYES={mean:.2f}±{std:.2f} "
                          f"({time.time() - t0:.0f}s)")

    json.dump(results, open(args.out, "w"), indent=1)
    print("\nwrote", args.out, "| backend:", backend.stats.as_dict())

    print("\n=== SUMMARY (bits) — TRANSMISSION I(m;m_recovered) = articulability; "
          "I_V_recon = capability-bound discrimination ===")
    agg = collections.defaultdict(lambda: {"trans": [], "recon": [], "fid": []})
    for r in results:
        agg[r["task"]]["trans"].append(r["iv_transmission"])
        agg[r["task"]]["recon"].append(r["iv_recon"])
        if r["fidelity_to_m_pearson"] is not None and np.isfinite(r["fidelity_to_m_pearson"]):
            agg[r["task"]]["fid"].append(r["fidelity_to_m_pearson"])
    for t, d in agg.items():
        f = np.mean(d["fid"]) if d["fid"] else float("nan")
        print(f"  {t:12s} TRANSMIT={np.nanmean(d['trans']):.3f}  I_V_recon={np.nanmean(d['recon']):.3f}  "
              f"fid_pearson={f:.2f}  (n={len(d['trans'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
