"""Reconstruction-channel I_V(x -> m_recovered): the GENUINE articulation bottleneck (2026-06-16).

Unlike the consistency channel (re-run a fixed rubric; near-degenerate because pass noise is tiny),
this routes the metric through a real bottleneck:

    m's behavior on a TRAIN split  ->  a reconstructor induces a rule  ->  a FRESH executor applies
    the induced rule to HELD-OUT x  ->  I_V on the held-out verdicts.

The recovery instances are R INDEPENDENT reconstructions (the channel's real stochasticity is which
rule gets induced). Two reconstruction modes:
  * free -- the reconstructor free-generates a rubric from hi/lo (text, m-verdict) examples.
  * mcq  -- the reconstructor PICKS one rule from a fixed candidate list (the task's seed metrics);
            a discrete codebook -> recovery bounded by log2(M) bits and identification accuracy.

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
import json
import time
from typing import List

import numpy as np

from . import vinfo
from .backends import parse_json_obj
from .batch_scoring import _YESNO_TEMPLATE
from .config import ImplementerConfig, apply_task_preset
from .manifest import full_manifest, load_corpus, load_metrics
from .vllm_backend import make_judge_backend

CLONE_CAP = 0.97   # distractors with kappa-to-target >= this are behavioral clones, excluded from selection

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


def _candidate_sims(target_vec, pool_pyes, target_id):
    """For each pool metric != target, its behavioral kappa-to-target and base rate. The basis for
    all distractor selection. pool_pyes: list of (metric, pyes_vector)."""
    sims = []
    for mm, pp in pool_pyes:
        if mm.metric_id == target_id:
            continue
        v = _verdict_vec(pp)
        sims.append({"met": mm, "kappa": _kappa(target_vec, v),
                     "rate": float(np.nanmean(pp > 0.5))})
    return [s for s in sims if np.isfinite(s["kappa"])]


def _select_distractors(sims, mode, n, target_rate, *, band=None, rng=None):
    """Pick n distractors from `sims` per a selection rule:
      random            -- uniform (the 'too easy' baseline: distractors likely far from m -> ceiling)
      hard              -- top-n by kappa-to-target (behavioral near-misses -> identification is real)
      matched_base_rate -- closest base rate to m (kills the base-rate giveaway leaked by the demo)
      band              -- n distractors whose kappa is closest to `band` (the graded difficulty dial)
    """
    if not sims:
        return []
    if mode == "random":
        idx = rng.permutation(len(sims))[:n]
        return [sims[i] for i in idx]
    if mode == "hard":
        return sorted(sims, key=lambda s: -s["kappa"])[:n]
    if mode == "matched_base_rate":
        return sorted(sims, key=lambda s: abs(s["rate"] - target_rate))[:n]
    if mode == "band":
        return sorted(sims, key=lambda s: abs(s["kappa"] - band))[:n]
    raise ValueError(f"unknown distractor mode: {mode}")


def _mcq_recon(backend, noun, examples, options, gold_idx, held_texts, max_chars, R, temperature,
               desc_map=None, recon=None):
    """Run R independent MCQ reconstructions over a fixed option set. Returns the held-out recovery
    matrix, the per-pass picks, and identification accuracy (fraction picking the true metric).
    `desc_map` (metric_id -> normalized description) presents options in a uniform register so the
    pick can't cheat on phrasing distinctiveness (vivid vs generic labels). ``recon`` (default
    ``backend``) is the reconstructor that PICKS the option (the strong model); ``backend`` is the
    executor that re-scores the picked option's body on held-out items (X)."""
    recon = recon or backend
    if desc_map:
        cand_descs = [desc_map.get(o.metric_id) or (o.description or o.name) for o in options]
    else:
        cand_descs = [(o.description or o.name) for o in options]
    recon_cols, induced, picks = [], [], []
    for r in range(R):
        pick = induce_mcq(recon, noun, examples, cand_descs, r)
        ch = pick["choice"]
        picks.append(ch)
        if 0 <= ch < len(options):
            induced.append({"choice": ch, "name": options[ch].name, "is_gold": ch == gold_idx})
            recon_cols.append(_sampled_binary(backend, options[ch].body, held_texts, max_chars,
                                              temperature, r + 1))
        else:
            induced.append({"choice": ch, "name": "?", "is_gold": False})
            recon_cols.append(np.full(len(held_texts), np.nan))
    recon_mat = np.column_stack(recon_cols)
    valid = [p for p in picks if 0 <= p < len(options)]
    id_acc = float(np.mean([p == gold_idx for p in valid])) if valid else float("nan")
    return recon_mat, induced, id_acc


def _ivs(recon_mat, m_verdict_held, cons_iv, metric, pyes, mode, *, extra=None) -> dict:
    """Common I_V readout block for a recovered-verdict matrix (free or one MCQ option set)."""
    iv_recon = vinfo.iv_from_reconstruction(recon_mat, orig_verdicts=m_verdict_held)
    iv_trans = vinfo.iv_transmission(recon_mat, m_verdict_held)   # I(m; m_recovered): articulability
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
        "m_pyes_mean": float(np.nanmean(pyes)), "m_pyes_std": float(np.nanstd(pyes)),
        "subtask_breadth": (metric.meta or {}).get("subtask_breadth"),
    }
    if extra:
        row.update(extra)
    return row


def run_metric(backend, noun, metric, all_metrics, texts, pyes, *, R, n_train, mode,
               max_chars, temperature=0.7, l_cap=450, distractor="hard", pool_pyes=None,
               n_options=4, graded_bands=(0.25, 0.5, 0.75, 0.95), desc_map=None,
               induce="free", gepa_pop=5, gepa_rounds=2, n_examples=4, example_mode="balanced",
               recon_backend=None) -> list:
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
    m_verdict = (pyes > 0.5).astype(float)
    m_verdict[~np.isfinite(pyes)] = np.nan
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    train_idx, held_idx = order[:n_train], order[n_train:]
    held_texts = [texts[i] for i in held_idx]
    m_held = m_verdict[held_idx]
    train_texts = [texts[i] for i in train_idx]
    m_train = m_verdict[train_idx]

    # Select example builder based on example_mode
    if example_mode == "balanced":
        # Balanced mode: oversample minority class for maximum contrast
        examples = _balanced_examples(train_texts, pyes[train_idx], k=n_examples, max_chars=max_chars)
    elif induce == "reasoning":
        # Reasoning mode: wide hi_lo (more examples, still not balanced)
        examples = _hi_lo_examples_wide(train_texts, pyes[train_idx], n_examples=n_examples, max_chars=max_chars)
    else:
        # Legacy mode: standard hi_lo with k examples
        examples = _hi_lo_examples(train_texts, pyes[train_idx], k=n_examples, max_chars=max_chars)

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

    # ---- MCQ: build option set(s) from BEHAVIORAL distractor selection ----
    rng_d = np.random.default_rng(7)
    all_sims = _candidate_sims(m_verdict, pool_pyes or [], metric.metric_id)
    # Exclude behavioral CLONES (kappa >= CLONE_CAP): a near-duplicate rubric isn't a distractor, it's
    # a relabeled copy of m -> would inflate 'hard' transmission for free and isn't a real near-miss.
    sims = [s for s in all_sims if s["kappa"] < CLONE_CAP]
    n_clones = len(all_sims) - len(sims)
    target_rate = float(np.nanmean(m_verdict))

    def _one_set(sel, tag):
        options = [metric] + [s["met"] for s in sel]
        perm = rng_d.permutation(len(options))
        options = [options[i] for i in perm]
        gold_idx = int(np.where(perm == 0)[0][0])
        # S = realized difficulty of the set: how close the BEST distractor behaves to m.
        S = max((s["kappa"] for s in sel), default=float("nan"))
        recon_mat, induced, id_acc = _mcq_recon(backend, noun, examples, options, gold_idx,
                                                held_texts, max_chars, R, temperature, desc_map=desc_map,
                                                recon=recon)
        # chance-corrected identification = (id - 1/M)/(1 - 1/M): the realistic UPPER-bound signal
        M = len(options)
        id_cc = (id_acc - 1.0 / M) / (1.0 - 1.0 / M) if (M > 1 and np.isfinite(id_acc)) else float("nan")
        return _ivs(recon_mat, m_held, cons_iv, metric, pyes, "mcq",
                    extra={"distractor": tag, "n_options": M, "option_set_S": S,
                           "identification_acc": id_acc, "identification_cc": id_cc,
                           "normalized": bool(desc_map), "n_clones_excluded": int(n_clones),
                           "induced": induced})

    if distractor != "graded":
        sel = _select_distractors(sims, distractor, n_options - 1, target_rate, rng=rng_d)
        return [_one_set(sel, distractor)]

    # graded: sweep band centers across the available kappa range -> transmission-vs-difficulty curve
    if not sims:
        return [_one_set([], "band@nan")]
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
    ap.add_argument("--R", type=int, default=5)
    ap.add_argument("--mode", default="free", choices=["free", "mcq"])
    ap.add_argument("--distractor", default="hard",
                    choices=["random", "hard", "matched_base_rate", "graded"],
                    help="MCQ option-set selection: random (easy ceiling), hard (behavioral "
                         "near-misses), matched_base_rate (kill base-rate tell), graded (sweep "
                         "difficulty S -> transmission-vs-difficulty curve)")
    ap.add_argument("--n-options", type=int, default=4, help="MCQ options per item (= 1 true + n-1 distractors); chance=1/n")
    ap.add_argument("--normalize-options", action="store_true",
                    help="paraphrase MCQ option descriptions to a uniform register (kills the "
                         "phrasing-distinctiveness confound) -> the REALISTIC identification upper bound")
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
        texts, _ = load_corpus(entry, args.n_items, seed=7)

        want = [i for i in args.metric_ids.split(",") if i in idmap] if args.metric_ids else []
        # pool_pyes: behavioral verdict vectors for the whole candidate pool -> MCQ distractor mining.
        pool_pyes = None
        if want:
            # FIXED metric set across tiers (valid transmission(E)). Measure all; flag degenerate.
            print(f"\n=== {tname} ({entry.task}) — FIXED {len(want)} metrics, {len(texts)} items ===")
            chosen = []
            for i in want:
                py = _pyes(backend, idmap[i].body, texts, max_chars)
                chosen.append((idmap[i], py, float(np.nanmean(py)), float(np.nanstd(py))))
            print("  P(YES) mean/std:", [f"{s[2]:.2f}/{s[3]:.2f}" for s in chosen])
            if args.mode == "mcq":   # need a distractor pool too; scan it
                scan = [(met, _pyes(backend, met.body, texts, max_chars)) for met in pool]
                pool_pyes = [(met, py) for met, py in scan]
        else:
            # P(YES) pre-scan: a metric DISCRIMINATES iff P(YES) varies across items (std).
            print(f"\n=== {tname} ({entry.task}) — scanning {len(pool)} seeds, {len(texts)} items ===")
            scan = [(met, py, float(np.nanmean(py)), float(np.nanstd(py)))
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
                                  recon_backend=recon_backend)
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
