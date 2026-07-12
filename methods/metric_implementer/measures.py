"""Construct-fidelity measures (README §scorecard). All label-free.

Implemented for the trial: reliability (1), counterfactual validity (2), reconstruction
validity (3), consistency/invariance (4), inter-implementation & code<->judge convergence
(7+8). Silver alignment (5) and predictive contribution (6) are deliberately absent from
this module — (5) needs commentary corpora, (6) is evaluation-only and computed by consumer
pipelines, never here, so the optimizer *cannot* see it (the evaluate-never-gate invariant
enforced structurally).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .artifact import MetricArtifact
from .backends import Roles, parse_json_obj
from .judges import CodeJudge, PromptJudge, make_judge
from .vinfo import tvd_mi, tvd_mi_passes, tvd_transmission


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
        # degenerate: fall back to exact-agreement rate
        return float(np.mean(np.abs(a[m] - b[m]) < 0.05)) if m.sum() else float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
        # degenerate (e.g. a collapsed judge outputs a constant): no rank correlation exists
        return float("nan")
    r = spearmanr(a[m], b[m]).statistic
    return float(r) if np.isfinite(r) else float("nan")


def format_perturb(body: str, seed: int) -> str:
    """Meaning-preserving format perturbation of a rubric: reorder UNORDERED bullet lines,
    normalize whitespace, and prepend a neutral framing line. Numbered/step lists are left in
    order (only '-'/'*'/'•' bullets move). Used to decorrelate format/position bias between two
    judge passes; any residual score shift across formats is real format-sensitivity, not signal."""
    rng = np.random.default_rng(seed)
    lines = body.splitlines()
    idx = [i for i, ln in enumerate(lines) if re.match(r"\s*[-*•]\s", ln)]
    if len(idx) >= 2:
        moved = [lines[idx[p]] for p in rng.permutation(len(idx))]
        for j, i in enumerate(idx):
            lines[i] = moved[j]
    out = "\n".join(lines)
    if seed % 2 == 0:
        out = re.sub(r"[ ]{2,}", " ", out)
    framings = ["Apply the rubric below carefully.", "Use the following rubric.",
                "Score using the rubric below."]
    return framings[seed % len(framings)] + "\n" + out


# ---- 1. reliability ---------------------------------------------------------------------

def reliability(artifact: MetricArtifact, texts: List[str], roles: Roles, cfg) -> dict:
    """Test-retest: k passes at judge temperature; mean pairwise Pearson.
    Code artifacts are deterministic -> reliability 1.0 by construction."""
    if artifact.kind == "code":
        return {"rho": 1.0, "passes": 0, "n": len(texts), "deterministic": True,
                "tvd_mi": None}
    judge = PromptJudge(roles.judge, cfg)
    # optionally format-perturb each pass after the first, so the two "views" differ in layout
    # (not just sampling noise) -> reliability TVD-MI decorrelates shared format/position bias.
    perturb = getattr(cfg, "reliability_format_perturb", False)
    passes = []
    for i in range(cfg.reliability_passes):
        a = artifact
        if perturb and i > 0:
            a = MetricArtifact(metric_id=artifact.metric_id, kind="prompt",
                               body=format_perturb(artifact.body, i), name=artifact.name,
                               description=artifact.description, invariances=artifact.invariances)
        passes.append(judge.score(a, texts)[0])
    tvd = tvd_mi_passes(np.column_stack(passes)) if len(passes) >= 2 else float("nan")
    rhos, flips = [], []
    for i in range(len(passes)):
        for j in range(i + 1, len(passes)):
            rhos.append(_pearson(passes[i], passes[j]))
            d = np.abs(passes[i] - passes[j])
            flips += [{"idx": int(k), "delta": float(d[k])}
                      for k in np.argsort(-np.nan_to_num(d))[:3] if d[k] > 0.3]
    rho = float(np.nanmean(rhos)) if rhos else float("nan")
    return {"rho": rho, "passes": cfg.reliability_passes, "n": len(texts),
            "retest_flips": flips[:6], "deterministic": False, "tvd_mi": tvd,
            "first_pass_scores": passes[0] if passes else None}


# ---- 4. consistency (declared invariances) ------------------------------------------------

_TRANSFORMS = {}


def _transform(name):
    def deco(fn):
        _TRANSFORMS[name] = fn
        return fn
    return deco


@_transform("blank_lines")
def _t_blank(code: str, rng) -> str:
    lines = code.splitlines()
    out = []
    for ln in lines:
        out.append(ln)
        if rng.random() < 0.25:
            out.append("")
    return "\n".join(out)


@_transform("comment_rewording")
def _t_comment(code: str, rng) -> str:
    # deterministic-ish rewording: prefix comments with an equivalent marker
    return re.sub(r"#\s*", lambda m: "# note: ", code)


@_transform("identifier_rename")
def _t_rename(code: str, rng) -> str:
    """Rename identifiers to OTHER meaningful names (semantic-preserving for metrics that
    do not target identifier quality). Crude token-safe renames of common names."""
    mapping = {"result": "outcome", "count": "tally", "total": "aggregate",
               "data": "records", "arr": "items", "ans": "answer", "res": "outcome2",
               "temp": "holder", "num": "value"}
    out = code
    for a, b in mapping.items():
        out = re.sub(rf"\b{a}\b", b, out)
    return out


def consistency(artifact: MetricArtifact, texts: List[str], roles: Roles, cfg,
                rng: np.random.Generator) -> dict:
    """Score stability under the artifact's *declared* invariances. A probe neutral for one
    metric is the target of another, so probes are per-metric (artifact.invariances)."""
    judge = make_judge(artifact, roles, cfg)
    base, base_ap = judge.score(artifact, texts)
    per: Dict[str, float] = {}
    tvd_per: Dict[str, float] = {}
    worst: List[dict] = []
    for inv in artifact.invariances:
        fn = _TRANSFORMS.get(inv)
        if fn is None:
            continue
        tx = [fn(t, rng) for t in texts]
        s, ap = judge.score(artifact, tx)
        m = base_ap & ap & np.isfinite(base) & np.isfinite(s)
        if m.sum() < 3:
            per[inv] = float("nan")
            continue
        delta = np.abs(s[m] - base[m])
        per[inv] = float(np.mean(delta <= 0.15))     # stability rate
        # TVD-MI(base, transformed): an INVARIANT metric keeps its item-signal under the
        # nuisance change -> high shared information; leakage -> low. (A collapsed judge -> 0.)
        tvd_per[inv] = tvd_mi(base[m], s[m])
        k = int(np.argmax(delta))
        if delta[k] > 0.3:
            worst.append({"invariance": inv, "idx": int(np.where(m)[0][k]),
                          "delta": float(delta[k])})
    vals = [v for v in per.values() if np.isfinite(v)]
    tvd_vals = [v for v in tvd_per.values() if np.isfinite(v)]
    return {"mean": float(np.mean(vals)) if vals else float("nan"),
            "per_invariance": per, "violations": worst[:6],
            "tvd_mi": float(np.mean(tvd_vals)) if tvd_vals else float("nan"),
            "tvd_per_invariance": tvd_per}


# ---- 3. reconstruction ---------------------------------------------------------------------

_RECON_PROMPT = """You are shown {item_phrase} with the score a hidden evaluator \
gave each one (0 = low, 1 = high). Your job: articulate the single rule the evaluator is \
applying. Do not guess what a good evaluator SHOULD do — describe what THIS one is doing.

{pairs}

Respond ONLY with JSON: {{"rule": "<one-sentence description of the rule>", \
"rubric": "<a rubric precise enough for someone else to score new items in [0,1]>"}}"""

_GRADE_PROMPT = """Rate how closely RULE B describes the same evaluation criterion as \
RULE A, on a 1-5 scale (5 = same criterion, 3 = overlapping but different emphasis, \
1 = different criterion).

RULE A: {a}

RULE B: {b}

Respond ONLY with JSON: {{"match": <int 1-5>, "difference": "<one short phrase>"}}"""


def reconstruct(artifact: MetricArtifact, texts: List[str], roles: Roles, cfg,
                rng: np.random.Generator, acceptance: bool = False) -> dict:
    """Judge labels items -> reconstructor articulates the observed rule -> semantic match
    (graded vs the artifact's own description) + behavioral match (induced rubric re-scored
    by the judge backend, Spearman vs the original scores on held-out items)."""
    recon_llm = roles.acceptance_reconstructor if acceptance else roles.reconstructor
    judge = make_judge(artifact, roles, cfg)

    # leave room for behavioral held-out items: labeling the whole pool starves the
    # behavioral match (observed 2026-06-12: n_pool == n_label -> behavioral always NaN).
    # Clamp to len(texts) so tiny pools (dry-runs) don't oversample.
    n_label = min(cfg.n_reconstruct_label_items, len(texts),
                  max(4, len(texts) - cfg.n_reconstruct_behavioral))
    idx = rng.choice(len(texts), size=n_label, replace=False)
    sample = [texts[i] for i in idx]
    scores, ap = judge.score(artifact, sample)

    ok = np.where(ap & np.isfinite(scores))[0]
    if len(ok) < 8:
        return {"semantic": float("nan"), "behavioral": float("nan"),
                "induced_rule": "", "error": "too_few_applicable"}
    order = ok[np.argsort(scores[ok])]
    k = min(cfg.n_reconstruct_shown // 2, len(order) // 2)
    show = list(order[:k]) + list(order[-k:])
    pairs = "\n\n".join(
        f"[score={scores[i]:.2f}]\n```\n{sample[i][:1200]}\n```" for i in show)

    raw = recon_llm.generate(_RECON_PROMPT.format(pairs=pairs, item_phrase=getattr(cfg, "recon_item_phrase", "code-solution excerpts")), max_tokens=500,
                             validate=lambda s: parse_json_obj(s) is not None)
    obj = parse_json_obj(raw) or {}
    rule, rubric = str(obj.get("rule", "")), str(obj.get("rubric", ""))
    if not rubric:
        return {"semantic": float("nan"), "behavioral": float("nan"),
                "induced_rule": rule, "error": "no_rubric"}

    # semantic match: grader compares induced rule to the artifact's own description/body
    target_desc = artifact.description or artifact.body[:600]
    graw = roles.grader.generate(_GRADE_PROMPT.format(a=target_desc, b=rule),
                                 max_tokens=120,
                                 validate=lambda s: parse_json_obj(s) is not None)
    gobj = parse_json_obj(graw) or {}
    try:
        semantic = (float(gobj.get("match")) - 1.0) / 4.0
    except (TypeError, ValueError):
        semantic = float("nan")

    # behavioral match: induced rubric scored on held-out items
    held = [i for i in range(len(texts)) if i not in set(idx)]
    rng.shuffle(held)
    held = held[: cfg.n_reconstruct_behavioral]
    held_texts = [texts[i] for i in held]
    induced = MetricArtifact(metric_id=artifact.metric_id + "_induced", kind="prompt",
                             body=rubric, description=rule)
    # Re-execute the induced rubric with a DIFFERENT-FAMILY same-tier executor when available, so
    # TVD-MI(original, induced) does not count bias the original judge shares with itself. Falls
    # back to the original judge (in-loop / 1-GPU default).
    executor = getattr(roles, "cross_executor", None) or roles.judge
    pj = PromptJudge(executor, cfg)
    s_orig, ap_o = judge.score(artifact, held_texts)
    s_ind, ap_i = pj.score(induced, held_texts)
    # A (near-)constant judge is degenerate, not unmeasurable: there is no behavior to
    # reconstruct. Score it 0 so the optimizer cannot profit from collapsing the judge
    # to a constant (renormalizing a NaN away would silently reward exactly that).
    if np.nanstd(np.where(ap_o, s_orig, np.nan)) < 0.05:
        return {"semantic": semantic, "behavioral": 0.0, "tvd_mi": 0.0,
                "induced_rule": rule, "induced_rubric": rubric,
                "degenerate_constant_judge": True,
                "difference": gobj.get("difference", "")}
    behavioral = _spearman(np.where(ap_o, s_orig, np.nan), np.where(ap_i, s_ind, np.nan))
    # TVD-MI(original, induced): the gaming-robust I(m; m̂) for this channel -- bits the induced
    # rubric transmits about the original's verdicts (and 0 for a collapsed judge, unlike Spearman).
    tvd = tvd_mi(np.where(ap_o, s_orig, np.nan), np.where(ap_i, s_ind, np.nan))
    return {"semantic": semantic, "behavioral": behavioral, "tvd_mi": tvd,
            "induced_rule": rule, "induced_rubric": rubric,
            "difference": gobj.get("difference", "")}


# ---- 2. counterfactual validity ------------------------------------------------------------

_CF_EDIT_PROMPT = """Rewrite the following {noun} so that it {direction_clause}, while \
changing NOTHING else of substance: {keep_clause}, and keep {hold_clause} \
exactly as similar as possible. Output ONLY the rewritten {noun}, no commentary.

```
{text}
```"""


def counterfactual_validity(artifact: MetricArtifact, texts: List[str], roles: Roles, cfg,
                            rng: np.random.Generator, suspects: Optional[List[dict]] = None,
                            acceptance: bool = False) -> dict:
    """Generator produces minimal pairs (raise-T / lower-T edits with everything else held);
    validity = judge tracks the planted direction. Suspect-control cells (vary S, hold T)
    check the judge is NOT tracking a named confound."""
    gen = roles.acceptance_generator if acceptance else roles.generator
    judge = make_judge(artifact, roles, cfg)
    target = artifact.description or artifact.name

    base_idx = rng.choice(len(texts), size=min(cfg.n_cf_base_texts, len(texts)),
                          replace=False)
    hold_default = getattr(cfg, "cf_hold_default", "comments, identifier style, and length")
    cells = [("t_up", f"clearly EXHIBITS the property: {target}", hold_default),
             ("t_down", f"clearly LACKS the property: {target}", hold_default)]
    for s in (suspects or [])[:2]:
        cells.append((f"s_vary:{s['name']}",
                      f"changes this UNRELATED aspect: {s['edit']} — while the property "
                      f"'{target}' must stay UNCHANGED",
                      "the target property"))

    prompts, meta = [], []
    for i in base_idx:
        for cell, direction, hold in cells:
            prompts.append(_CF_EDIT_PROMPT.format(
                direction_clause=direction, hold_clause=hold, text=texts[i][:2500],
                noun=getattr(cfg, "item_noun", "solution"),
                keep_clause=getattr(cfg, "cf_keep_clause", "keep the algorithm and structure")))
            meta.append({"base": int(i), "cell": cell})
    edited = gen.generate_batch(prompts, max_tokens=1400)
    edited = [re.sub(r"^```\w*\n?|```$", "", e.strip(), flags=re.M) for e in edited]

    all_texts = [texts[i] for i in base_idx] + edited
    s_all, ap_all = judge.score(artifact, all_texts)
    base_scores = {int(b): s_all[k] for k, b in enumerate(base_idx)}
    res: Dict[str, List[float]] = {}
    failures: List[dict] = []
    for k, m in enumerate(meta):
        s_edit = s_all[len(base_idx) + k]
        s_base = base_scores[m["base"]]
        if not (np.isfinite(s_edit) and np.isfinite(s_base)):
            continue
        delta = s_edit - s_base
        if m["cell"] == "t_up":
            hit = float(delta > 0.05)
        elif m["cell"] == "t_down":
            hit = float(delta < -0.05)
        else:                                   # suspect cell: should NOT move
            hit = float(abs(delta) <= 0.15)
        res.setdefault(m["cell"], []).append(hit)
        if not hit:
            failures.append({**m, "delta": float(delta),
                             "edited_excerpt": edited[k][:400]})
    per_cell = {c: float(np.mean(v)) for c, v in res.items() if v}
    t_cells = [v for c, v in per_cell.items() if c.startswith("t_")]
    return {"cf_validity": float(np.mean(t_cells)) if t_cells else float("nan"),
            "per_cell": per_cell, "failures": failures[:6]}


# ---- 7+8. agreement / convergence ----------------------------------------------------------

def convergence(prompt_art: MetricArtifact, code_art: MetricArtifact, texts: List[str],
                roles: Roles, cfg) -> dict:
    """Code<->judge convergence on a shared sample + the disagreement queue (the worst
    |code - judge| items), which is the improvement queue for the optimizer."""
    pj = PromptJudge(roles.judge, cfg)
    cj = CodeJudge()
    sp, ap_p = pj.score(prompt_art, texts)
    sc, ap_c = cj.score(code_art, texts)
    both = ap_p & ap_c & np.isfinite(sp) & np.isfinite(sc)
    rho = _spearman(np.where(both, sp, np.nan), np.where(both, sc, np.nan))
    diffs = np.where(both, np.abs(sp - sc), -1)
    order = np.argsort(-diffs)[: cfg.n_disagreement_keep]
    queue = [{"idx": int(i), "judge": float(sp[i]), "code": float(sc[i]),
              "abs_diff": float(diffs[i]), "text_excerpt": texts[i][:400]}
             for i in order if diffs[i] > 0]
    return {"spearman": rho, "n_both_applicable": int(both.sum()),
            "applicability_prompt": float(ap_p.mean()),
            "applicability_code": float(ap_c.mean()),
            "disagreement_queue": queue}


# ---- oracle agreement (strong-proposer / weak-judge asymmetry) -----------------------------

_ORACLE_RUBRIC = """Evaluate the property: {description}
Score 1.0 when the {noun} clearly exhibits it, 0.0 when it clearly lacks it, intermediate
values for partial cases. Apply your own expert judgment of the property as described —
do not invent additional criteria."""


def oracle_agreement(artifact: MetricArtifact, texts: List[str], roles: Roles, cfg,
                     oracle_cache: Optional[dict] = None) -> dict:
    """Spearman between this implementation's scores and a STRONG model scoring the metric's
    canonical *description* (not the candidate body) on a fixed subsample.

    The oracle is a stable distillation target: it does not move as candidates mutate, so
    optimizing against it asks "does (weak judge + this prompt) reproduce the strong model's
    reading of the construct?" — one point on the judge-capability scaling axis. The oracle
    pass is cached per metric (it is candidate-independent).
    """
    if roles.oracle is None or cfg.n_oracle_items <= 0:
        return {"spearman": float("nan"), "n": 0, "skipped": True}
    sub = texts[: cfg.n_oracle_items]
    key = artifact.metric_id
    if oracle_cache is not None and key in oracle_cache:
        s_oracle, ap_o = oracle_cache[key]
    else:
        canon = MetricArtifact(metric_id=artifact.metric_id + "_oracle", kind="prompt",
                               body=_ORACLE_RUBRIC.format(
                                   description=artifact.description or artifact.name,
                                   noun=getattr(cfg, "item_noun", "solution")),
                               description=artifact.description)
        oj = PromptJudge(roles.oracle, cfg)
        s_oracle, ap_o = oj.score(canon, sub, temperature=0.3)
        if oracle_cache is not None:
            oracle_cache[key] = (s_oracle, ap_o)
    judge = make_judge(artifact, roles, cfg)
    s_weak, ap_w = judge.score(artifact, sub)
    both = ap_o & ap_w
    rho = _spearman(np.where(both, s_oracle, np.nan), np.where(both, s_weak, np.nan))
    return {"spearman": rho, "n": int(both.sum()), "oracle_model": roles.oracle.model,
            "skipped": False}


# ---- the scorecard ------------------------------------------------------------------------

def discrimination(first_pass_scores, cfg) -> dict:
    """The per-item SIGNAL a prompt carries — the thing the leniency attractor destroys.

    A prompt that scores ~constant across items (the GEPA leniency fixed point: a mechanical
    checklist every item passes) has near-zero discrimination, and so caps every downstream MI
    quantity at ~0 by the entropy bound — regardless of LLM resilience. We measure the prompt's
    soft spread as the TVD transmission of its [0,1] verdicts (treating them as per-item P(YES)):
    T = mean|p_i - p̄|, range [0, ½]. `disc_norm` = min(1, T / T_healthy) folds it onto [0,1]
    for the fidelity objective (T_healthy default 0.10 — a prompt that varies by ~0.1 MAD is
    healthily discriminating; one near the 0.02 noise floor is degenerate). NO extra judge call:
    reuses reliability's first pass. Code artifacts → None (they're deterministic, scored separately)."""
    if first_pass_scores is None:
        return {"T": None, "disc_norm": None, "collapsed": True}
    s = np.asarray(first_pass_scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 4 or np.nanstd(s) < 1e-6:
        return {"T": 0.0, "disc_norm": 0.0, "collapsed": True}
    T = float(tvd_transmission(s, debias=False)["tvd_t"])
    T_healthy = getattr(cfg, "disc_healthy", 0.10)
    return {"T": T, "disc_norm": float(min(1.0, T / T_healthy)),
            "collapsed": bool(T < getattr(cfg, "disc_floor", 0.02))}


def fidelity_scalar(card: dict, cfg) -> float:
    """The optimizer objective. Predictive contribution is structurally absent from this
    module, so it CANNOT leak in. NaN components are skipped with weight renormalization."""
    parts = [(cfg.w_cf, card.get("counterfactual", {}).get("cf_validity")),
             (cfg.w_recon, card.get("reconstruction", {}).get("behavioral")),
             (cfg.w_reliability, card.get("reliability", {}).get("rho")),
             (cfg.w_consistency, card.get("consistency", {}).get("mean")),
             (getattr(cfg, "w_oracle", 0.0),
              card.get("oracle_agreement", {}).get("spearman")),
             (getattr(cfg, "w_disc", 0.0),
              card.get("discrimination", {}).get("disc_norm"))]
    num = den = 0.0
    for w, v in parts:
        if v is not None and np.isfinite(v):
            num += w * float(np.clip(v, 0.0, 1.0))
            den += w
    return num / den if den > 0 else float("nan")


def compute_scorecard(artifact: MetricArtifact, texts: List[str], roles: Roles, cfg,
                      rng: np.random.Generator, suspects: Optional[List[dict]] = None,
                      acceptance: bool = False,
                      oracle_cache: Optional[dict] = None) -> dict:
    rel = reliability(artifact, texts, roles, cfg)
    rec = reconstruct(artifact, texts, roles, cfg, rng, acceptance=acceptance)
    # reconstruction names the suspect: if the induced rule mismatches, it becomes a
    # counterfactual control cell ("the judge appears to track THIS instead")
    auto_suspects = list(suspects or [])
    if rec.get("induced_rule") and np.isfinite(rec.get("semantic", np.nan)) \
            and rec["semantic"] < 0.75:
        auto_suspects.append({"name": "reconstructed_reading",
                              "edit": rec["induced_rule"]})
    cf = counterfactual_validity(artifact, texts, roles, cfg, rng,
                                 suspects=auto_suspects, acceptance=acceptance)
    con = consistency(artifact, texts[: cfg.n_consistency_items], roles, cfg, rng)
    orc = oracle_agreement(artifact, texts, roles, cfg, oracle_cache)
    disc = discrimination(rel.get("first_pass_scores"), cfg)       # reuse reliability's first pass
    card = {"metric_id": artifact.metric_id, "kind": artifact.kind,
            "body_hash": artifact.body_hash,
            "reliability": rel, "reconstruction": rec, "counterfactual": cf,
            "consistency": con, "oracle_agreement": orc, "discrimination": disc,
            "complexity": artifact.complexity()}
    # Gaming-robust recovery: mean TVD-MI over the three peer/perturbation channels
    # (reliability, consistency, reconstruction). Recorded ALONGSIDE fidelity_scalar, not
    # folded into it -- so we can compare whether an optimizer inflates one but not the other.
    _peer_tvd = [rel.get("tvd_mi"), con.get("tvd_mi"), rec.get("tvd_mi")]
    _peer_tvd = [v for v in _peer_tvd if v is not None and np.isfinite(v)]
    card["tvd_recovery"] = float(np.mean(_peer_tvd)) if _peer_tvd else float("nan")
    card["fidelity_scalar"] = fidelity_scalar(card, cfg)
    # promotable: reliable AND not collapsed to a leniency attractor (discrimination gate).
    # A degenerate constant-YES prompt has rho~1 (perfectly stable) but carries no signal —
    # the disc term blocks it from becoming the GEPA lineage parent.
    card["promotable"] = bool(np.isfinite(rel.get("rho", np.nan))
                              and rel["rho"] >= cfg.reliability_floor
                              and not disc.get("collapsed", False))
    return card
