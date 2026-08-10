"""Outer discover -> detect -> generate -> materialize -> reinsert -> guard loop (spec §8).

    repeat:
        fit gap-detecting tree on items_discover (current metric set)
        gap_nodes = terminal nodes flagged on items_test (§2.1)
        if no gaps: STOP
        for each gap node (or pooled cluster, §6):
            build residualized contrast (§3) -> propose feature (§4)
            materialize over the whole corpus + reinsert (§5)
            apply guards on items_test (§7); if kept, add to the metric set
        if nothing kept this round, or budget exhausted: STOP

Materialization is incremental: only the newly proposed feature is scored each round; existing
columns are carried forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .contrast import Contrast, build_contrast, pool_contrasts
from .depth_dial import find_poolable_clusters
from .feature_gen import (
    ProposedComposite, Proposer, estimate_reliability, propose_composite_feature, propose_feature,
)
from .gaps import flag_gap_nodes
from .interactions import best_combination, combine_with_rule
from .guards import (
    gap_closure_check,
    measured_importance,
    redundancy_check,
    subset_deviance,
)
from .io_metrics import (
    JudgeScorer,
    MetricSpec,
    ScoreMatrix,
    make_design,
)
from .mob.glmtree import GapTree


@dataclass
class FeatureRecord:
    name: str
    description: str
    rubric: str
    origin: str                     # "leaf:<node>" | "pooled:<nodes>"
    status: str                     # "kept" | "dropped:<guard>"
    reliability: float = float("nan")
    redundancy_r2: float = float("nan")
    gap_drop_fraction: float = float("nan")
    minimal_depth: int = -1
    corrected_importance: float = float("nan")
    coverage: float = float("nan")     # fraction of the population the closed gap(s) span
    gap_depth: int = -1                # depth of the gap node (or min depth of pooled cluster)


@dataclass
class InfillResult:
    tree: GapTree
    metrics: List[MetricSpec]
    records: List[FeatureRecord] = field(default_factory=list)
    rounds: int = 0
    final_gap_count: int = 0
    # Final materialized matrices (base + kept features), discover + test sides. Use these for
    # power measurement instead of re-materializing ``metrics``: kept COMPOSITES have only a
    # placeholder code_fn (their levels came from scoring the two primitives), so a fresh
    # materialize() would turn them into constant-zero columns (Codex #3).
    sm_discover: Optional[ScoreMatrix] = None
    sm_test: Optional[ScoreMatrix] = None
    # When a 3-way split is used, sm_holdout is the FINAL (untouched) test side — base + kept
    # features, materialized but never used for any keep/drop decision. None for 2-way callers.
    sm_holdout: Optional[ScoreMatrix] = None
    # Guard-split bank AUC after each outer round (the selection-side power trajectory; the
    # honest TEST trajectory is measured by the caller on sm_holdout, which the loop never sees).
    guard_auc_trajectory: List[Tuple[int, float]] = field(default_factory=list)


# --------------------------------------------------------------------------------------

def _append_column(sm: ScoreMatrix, levels_col: np.ndarray, applicable_col: np.ndarray,
                   metric: MetricSpec) -> ScoreMatrix:
    return ScoreMatrix(
        levels=np.column_stack([sm.levels, levels_col]),
        applicable=np.column_stack([sm.applicable, applicable_col]),
        metric_ids=sm.metric_ids + [metric.metric_id],
        metric_names=sm.metric_names + [metric.name],
        roles=(sm.roles or ["both"] * (sm.levels.shape[1])) + [metric.role],
    )


def _score_one(metric: MetricSpec, texts: List[str], judge_scorer: JudgeScorer
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Materialize a single new metric over ``texts`` -> (levels (n,), applicable (n,))."""
    if metric.kind == "code":
        lv = np.full(len(texts), np.nan)
        ap = np.zeros(len(texts), dtype=bool)
        for i, t in enumerate(texts):
            v = metric.code_fn(t)
            if v is not None and np.isfinite(v):
                lv[i], ap[i] = v, True
        return lv, ap
    lv, ap = judge_scorer([metric], texts)
    return lv[:, 0], ap[:, 0]


def _materialize_composite(comp: ProposedComposite, node_disc_idx, texts_d: List[str],
                           texts_t: List[str], y_d: np.ndarray, judge_scorer: JudgeScorer,
                           cfg, rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, Optional[str]]:
    """Materialize a 2-primitive composite: score each primitive over discover+test, fit the best
    boolean rule on the gap node's discover items, apply it to both. Returns the combined level
    columns + their applicability (AND of the primitives') + reliability (min of the two)."""
    p = [pr.to_metric() for pr in comp.primitives]
    ad, aap = _score_one(p[0], texts_d, judge_scorer)
    bd, bap = _score_one(p[1], texts_d, judge_scorer)
    at, aap_t = _score_one(p[0], texts_t, judge_scorer)
    bt, bap_t = _score_one(p[1], texts_t, judge_scorer)
    ndi = node_disc_idx if node_disc_idx is not None else np.arange(len(y_d))
    # Fit the boolean rule ONLY on rows where both primitives are applicable. ``binarize`` maps
    # N/A -> 0, so fitting on the full node would select the rule under N/A-as-0 semantics while
    # materialization masks those same rows to NaN — a train/apply mismatch that biases the chosen
    # op (Codex #5). If too few applicable rows, treat as no genuine interaction.
    fit_mask = np.zeros(len(y_d), dtype=bool)
    fit_mask[ndi] = True
    fit_mask &= aap & bap
    if int(fit_mask.sum()) >= 20:
        rule, _ = best_combination(ad[fit_mask], bd[fit_mask], y_d[fit_mask])
    else:
        rule = None
    ap_d, ap_t = aap & bap, aap_t & bap_t
    if rule is None:
        # no genuine interaction fit -> mark non-applicable everywhere so the gap-closure guard
        # drops it. Do NOT insert a spurious marginal (that would violate the §9 interaction gate).
        nan_d = np.full_like(ad, np.nan, dtype=float)
        return nan_d, np.zeros_like(ap_d), np.full_like(at, np.nan, dtype=float), \
            np.zeros_like(ap_t), 0.0, None
    # non-applicable rows -> NaN so make_design imputes them (not treated as real 0s)
    lv_d = np.where(ap_d, combine_with_rule(ad, bd, rule), np.nan).astype(float)
    lv_t = np.where(ap_t, combine_with_rule(at, bt, rule), np.nan).astype(float)
    rel = min(estimate_reliability(p[0], texts_t, judge_scorer, cfg.reliability_sample_size, rng),
              estimate_reliability(p[1], texts_t, judge_scorer, cfg.reliability_sample_size, rng))
    return lv_d, ap_d, lv_t, ap_t, float(rel), rule


@dataclass
class _Proposal:
    feature: object                 # ProposedFeature | ProposedComposite
    contrast: Contrast
    target_test_idx: np.ndarray     # held-out items whose gap this should close
    origin: str
    gap_depth: int                  # depth of the gap node (min depth across a pooled cluster)
    composite: Optional[ProposedComposite] = None   # set for the §9 composite path
    node_disc_idx: Optional[np.ndarray] = None      # gap node's discover rows (to fit the rule)


def _assemble_proposals(
    gaps, contrasts, cfg, proposer: Proposer, known_desc: List[str],
    composite_proposer: Optional[Proposer] = None,
) -> List[_Proposal]:
    """Per-gap-node proposals, with poolable clusters re-proposed once (depth dial, §6)."""
    per_node = []
    for g, c in zip(gaps, contrasts):
        if c is None:
            continue
        pf = propose_feature(c, known_desc, cfg, proposer)
        if pf is not None:
            per_node.append((g, c, pf))
    if not per_node and not getattr(cfg, "enable_composite_proposer", False):
        return []   # a gap with no single proposal is exactly the root-XOR case composites handle

    descriptions = [pf.description or pf.name for (_, _, pf) in per_node]
    clusters = find_poolable_clusters(descriptions, cfg)
    clustered = set().union(*clusters) if clusters else set()

    proposals: List[_Proposal] = []
    for cluster in clusters:
        members = [per_node[i] for i in cluster]
        pooled_c = pool_contrasts([c for (_, c, _) in members])
        pf = propose_feature(pooled_c, known_desc, cfg, proposer)
        if pf is None:
            continue
        target = np.concatenate([g.test_indices for (g, _, _) in members])
        origin = "pooled:" + "+".join(g.node.node_id for (g, _, _) in members)
        gap_depth = min(g.node.depth for (g, _, _) in members)
        proposals.append(_Proposal(pf, pooled_c, target, origin, gap_depth))

    for i, (g, c, pf) in enumerate(per_node):
        if i in clustered:
            continue
        proposals.append(_Proposal(pf, c, g.test_indices, f"leaf:{g.node.node_id}", g.node.depth))

    # §9 composite path: for each gap, also propose a 2-primitive boolean composite (the case no
    # single feature can close — a root XOR of absent primitives). Gated by cfg.
    if getattr(cfg, "enable_composite_proposer", False):
        cprop = composite_proposer or proposer
        for g, c in zip(gaps, contrasts):
            if c is None:
                continue
            comp = propose_composite_feature(c, known_desc, cfg, cprop)
            if comp is not None:
                proposals.append(_Proposal(
                    comp, c, g.test_indices, f"composite:{g.node.node_id}", g.node.depth,
                    composite=comp, node_disc_idx=g.node.indices))
    return proposals


# --------------------------------------------------------------------------------------

def _bank_auc(X_disc: np.ndarray, y_disc: np.ndarray,
              X_eval: np.ndarray, y_eval: np.ndarray) -> float:
    """Guard-side bank AUC: fit label ~ metric-levels LR on discover, eval on the guard split,
    using only non-degenerate columns. This is the direct north-star power criterion."""
    if len(np.unique(y_disc)) < 2 or len(np.unique(y_eval)) < 2:
        return float("nan")
    keep = np.where(X_disc.std(axis=0) > 1e-6)[0]
    if len(keep) < 1:
        return float("nan")
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(X_disc[:, keep], y_disc)
    return float(roc_auc_score(y_eval, lr.predict_proba(X_eval[:, keep])[:, 1]))


def _accept(mode: str, gap_closed: bool, auc_gain: float, min_gain: float) -> bool:
    """Combine the gap-closure guard and the direct guard-AUC gate per cfg.acceptance_mode."""
    has_auc = bool(np.isfinite(auc_gain)) and auc_gain >= min_gain
    if mode == "auc":
        return has_auc
    if mode == "either":
        return gap_closed or has_auc
    if mode == "both":
        return gap_closed and has_auc
    return gap_closed                       # default "gap_closure"


def run_infill(
    df_discover, df_test,
    metrics: List[MetricSpec], sm_discover: ScoreMatrix, sm_test: ScoreMatrix,
    cfg, proposer: Proposer, judge_scorer: JudgeScorer,
    log=print, composite_proposer: Optional[Proposer] = None,
    df_holdout=None, sm_holdout: Optional[ScoreMatrix] = None,
) -> InfillResult:
    """Run the full infilling loop. ``sm_*`` are the materialized base metric sets.

    Two-way callers pass ``df_test``/``sm_test`` (the eval side drives every decision). For a
    genuine holdout, pass ``df_holdout``/``sm_holdout`` (the untouched confirm split): features
    are materialized there too but NO decision consults it, so its AUC is a clean readout.
    Acceptance is governed by ``cfg.acceptance_mode`` (gap-closure and/or direct guard AUC).
    """
    rng = np.random.default_rng(cfg.random_seed)
    y_d = df_discover[cfg.label_column].to_numpy(float)
    y_e = df_test[cfg.label_column].to_numpy(float)          # "eval" = guard side; decisions here
    texts_d = df_discover[cfg.text_column].astype(str).tolist()
    texts_e = df_test[cfg.text_column].astype(str).tolist()
    have_holdout = df_holdout is not None and sm_holdout is not None
    texts_h = df_holdout[cfg.text_column].astype(str).tolist() if have_holdout else []

    metrics = list(metrics)
    records: List[FeatureRecord] = []

    X_d, fn_d, Z_d, spec = make_design(sm_discover, df_discover, cfg)
    X_e, _, Z_e, _ = make_design(sm_test, df_test, cfg, spec=spec)
    tree = GapTree(cfg).fit(X_d, y_d, Z_d, fn_d)
    cur_auc_e = _bank_auc(X_d, y_d, X_e, y_e)                 # current bank AUC on the guard split
    traj: List[Tuple[int, float]] = []

    rounds = 0
    for rnd in range(cfg.max_outer_rounds):
        rounds = rnd + 1
        gaps = flag_gap_nodes(tree, X_e, y_e, Z_e, cfg)
        log(f"[round {rounds}] tree terminals={len(tree.terminal_nodes())} gap_nodes={len(gaps)} "
            f"guard_auc={cur_auc_e:.3f}")
        traj.append((rounds, cur_auc_e))
        if not gaps:
            break

        contrasts = [build_contrast(tree, g, df_discover, X_d, y_d, cfg, rng) for g in gaps]
        known_desc = [(m.description or m.name) for m in metrics][:40]
        proposals = _assemble_proposals(gaps, contrasts, cfg, proposer, known_desc, composite_proposer)
        log(f"[round {rounds}] proposals={len(proposals)}")

        kept_this_round = 0
        for prop in proposals[: cfg.max_features_per_round]:
            metric = prop.feature.to_metric()
            metric.role = cfg.discovered_feature_role   # "feature" (X only) or "both" (X + splittable)

            # old held-out deviance at the targeted items (current tree, guard side)
            old_dev, _ = subset_deviance(tree, X_e, y_e, Z_e, prop.target_test_idx)

            # materialize the new feature over discover + guard
            if prop.composite is not None:
                lv_d, ap_d, lv_e, ap_e, reliability, fitted_rule = _materialize_composite(
                    prop.composite, prop.node_disc_idx, texts_d, texts_e, y_d,
                    judge_scorer, cfg, rng)
                if fitted_rule is None:
                    records.append(_rec(prop, metric, "dropped:no_interaction", len(y_e),
                                        reliability=reliability))
                    log(f"  drop[no_interaction] composite {metric.name!r}")
                    continue
            else:
                lv_d, ap_d = _score_one(metric, texts_d, judge_scorer)
                lv_e, ap_e = _score_one(metric, texts_e, judge_scorer)
                reliability = estimate_reliability(
                    metric, texts_e, judge_scorer, cfg.reliability_sample_size, rng)

            # guard 2: redundancy vs existing metric columns (guard set)
            red = redundancy_check(lv_e, X_e, ap_e, reliability, cfg.tau_redundant)
            if red.redundant:
                records.append(_rec(prop, metric, "dropped:redundant", len(y_e),
                                    reliability=reliability, redundancy_r2=red.r2))
                log(f"  drop[redundant] {metric.name!r} R2={red.r2:.2f}")
                continue

            # reinsert: refit with the new feature
            sm_d2 = _append_column(sm_discover, lv_d, ap_d, metric)
            sm_e2 = _append_column(sm_test, lv_e, ap_e, metric)
            X_d2, fn_d2, Z_d2, spec2 = make_design(sm_d2, df_discover, cfg)
            X_e2, _, Z_e2, _ = make_design(sm_e2, df_test, cfg, spec=spec2)
            tree2 = GapTree(cfg).fit(X_d2, y_d, Z_d2, fn_d2)
            new_flagged = {g.node.node_id for g in flag_gap_nodes(tree2, X_e2, y_e, Z_e2, cfg)}

            # guard 3: gap-closure  +  direct guard-AUC gate (cfg.acceptance_mode)
            closure = gap_closure_check(
                old_dev, tree2, X_e2, y_e, Z_e2, prop.target_test_idx, new_flagged, cfg)
            cand_auc_e = _bank_auc(X_d2, y_d, X_e2, y_e)
            auc_gain = ((cand_auc_e - cur_auc_e)
                        if (np.isfinite(cand_auc_e) and np.isfinite(cur_auc_e)) else float("nan"))
            if not _accept(cfg.acceptance_mode, closure.closed, auc_gain, cfg.min_auc_gain):
                reason = "no_closure" if not closure.closed else "no_auc"
                records.append(_rec(prop, metric, f"dropped:{reason}", len(y_e),
                                    reliability=reliability, redundancy_r2=red.r2,
                                    gap_drop_fraction=closure.drop_fraction))
                log(f"  drop[{reason}] {metric.name!r} drop_frac={closure.drop_fraction:.2f} "
                    f"auc_gain={auc_gain:+.3f} lvl_std={np.nanstd(lv_e):.3f} "
                    f"lvl_uniq={len(np.unique(np.round(lv_e[ap_e], 2)))}")
                continue

            # KEEP: also materialize on the untouched holdout (final power read), then commit
            if have_holdout:
                if prop.composite is not None:
                    ph = [pr.to_metric() for pr in prop.composite.primitives]
                    ah, aap_h = _score_one(ph[0], texts_h, judge_scorer)
                    bh, bap_h = _score_one(ph[1], texts_h, judge_scorer)
                    ap_h = aap_h & bap_h
                    lv_h = np.where(ap_h, combine_with_rule(ah, bh, fitted_rule),
                                    np.nan).astype(float)
                else:
                    lv_h, ap_h = _score_one(metric, texts_h, judge_scorer)
                sm_holdout = _append_column(sm_holdout, lv_h, ap_h, metric)

            imp = measured_importance(tree2, metric.name, reliability)
            metrics.append(metric)
            sm_discover, sm_test = sm_d2, sm_e2
            X_d, fn_d, Z_d, spec = X_d2, fn_d2, Z_d2, spec2
            X_e, Z_e, tree = X_e2, Z_e2, tree2
            cur_auc_e = cand_auc_e if np.isfinite(cand_auc_e) else cur_auc_e
            records.append(_rec(prop, metric, "kept", len(y_e), reliability=reliability,
                                redundancy_r2=red.r2, gap_drop_fraction=closure.drop_fraction,
                                minimal_depth=imp.minimal_depth,
                                corrected_importance=imp.corrected_importance))
            kept_this_round += 1
            log(f"  KEEP {metric.name!r} depth={imp.minimal_depth} drop_frac={closure.drop_fraction:.2f} "
                f"auc_gain={auc_gain:+.3f} rel={reliability:.2f}")

        if kept_this_round == 0:
            break

    # Recompute coverage on the FINAL tree: the fraction of the population in leaves where the
    # feature is actually active (its standardized coefficient is non-negligible). This is the
    # faithful generality readout, independent of which gap node first surfaced the feature.
    kept_names = {r.name for r in records if r.status == "kept"}
    for r in records:
        if r.name in kept_names:
            cov = tree.feature_active_coverage(r.name, X_e, Z_e, len(y_e))
            if np.isfinite(cov):
                r.coverage = cov

    final_gaps = flag_gap_nodes(tree, X_e, y_e, Z_e, cfg)
    return InfillResult(tree=tree, metrics=metrics, records=records,
                        rounds=rounds, final_gap_count=len(final_gaps),
                        sm_discover=sm_discover, sm_test=sm_test,
                        sm_holdout=sm_holdout if have_holdout else None,
                        guard_auc_trajectory=traj)


def _rec(prop: _Proposal, metric: MetricSpec, status: str, n_test: int, **kw) -> FeatureRecord:
    coverage = len(prop.target_test_idx) / max(n_test, 1)
    return FeatureRecord(
        name=metric.name, description=metric.description, rubric=metric.guidance,
        origin=prop.origin, status=status, coverage=coverage, gap_depth=prop.gap_depth, **kw,
    )
