"""Global (tree-free) metric infilling — the simple boosting-style sibling of ``loop.run_infill``.

No MOB tree, no partitioning: fit the bank GLM on ALL discover rows, form the WRONG/RIGHT
residual contrast GLOBALLY, ask the proposer for one corpus-wide metric per round, accept iff
guard-split bank-AUC improves by ``min_auc_gain``. Repeat until ``patience`` consecutive
rejections or ``max_rounds``.

Where this sits in the theory (notes/2026-07-01__metric-count-certificates.md §5): the global
contrast targets the residual trichotomy's (iii) uniform component and the corpus-wide part of
(ii); it is blind BY DESIGN to (i) moderation-shaped residual — a metric whose sign flips
across subpopulations nets to ~zero in a global contrast. The count certificates are
partition-free and apply verbatim: the accepted sequence IS the greedy chain, so
``N_delta <= (U - V(S_g)) / delta`` and the Minoux stopping read come directly off the ledger.

Per-metric ledger (the three requested tracks):
  1. data-to-develop  — ``n_proposal_examples`` (contrast items the proposer saw) plus a
     ``data_curve``: guard-AUC gain when the bank(+metric) GLM is trained on 25/50/100% of
     discover rows -> ``min_train_frac`` = smallest fraction achieving half the full gain.
  2. applicability    — fraction of discover rows where the judge marks the metric applicable
     (and same for guard).
  3. reconstruction   — articulability R of the metric itself: a reconstructor LLM re-derives
     the rubric from (text, verdict) pairs (never seeing the rubric), the re-derived rubric is
     re-executed on held-out items, and we report agreement/AUC vs the original verdicts.
     ``reconstruct_n_tries`` best-of-k on a dev slice is a cheap stand-in for GEPA; a full
     GEPA-optimized reconstruction plugs in via ``reconstructor_fn`` (see AGENT_PLAYBOOK).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .contrast import Contrast
from .feature_gen import propose_feature
from .io_metrics import MetricSpec, ScoreMatrix, make_design
from .loop import _append_column, _score_one


# --------------------------------------------------------------------------------------

@dataclass
class MetricLedger:
    """Everything tracked about one proposed metric (accepted or not)."""

    name: str
    description: str
    rubric: str
    round: int
    status: str                                   # "kept" | "dropped:<reason>"
    # -- track 1: data needed to develop --------------------------------------------
    n_proposal_examples: int = 0                  # contrast items shown to the proposer
    data_curve: dict = field(default_factory=dict)   # train_frac -> guard AUC gain
    min_train_frac: float = float("nan")          # smallest frac reaching >= half the full gain
    # -- track 2: applicability ------------------------------------------------------
    applicability_discover: float = float("nan")
    applicability_guard: float = float("nan")
    # -- track 3: articulability of the metric itself --------------------------------
    reconstruction_agreement: float = float("nan")  # held-out verdict agreement (balanced acc)
    reconstruction_auc: float = float("nan")        # held-out AUC of rederived vs original
    reconstruction_rubric: str = ""                 # what the reconstructor articulated
    # -- value ------------------------------------------------------------------------
    guard_auc_before: float = float("nan")
    guard_auc_after: float = float("nan")
    auc_gain: float = float("nan")
    guard_bits_before: float = float("nan")     # V_bits: guard log-loss reduction vs base rate
    guard_bits_after: float = float("nan")
    bits_gain: float = float("nan")             # delta in bits/item — the certificate currency
    redundancy_r2: float = float("nan")
    generator: str = "residual"                 # which proposal arm produced this candidate
    # -- confirm stage (winner's-curse control, MCC §4.4) ------------------------------
    confirm_auc_gain: float = float("nan")      # fresh-seed repeated-CV mean gain
    confirm_bits_gain: float = float("nan")
    confirm_p_auc: float = float("nan")         # NB-corrected one-sided p vs 0
    confirm_p_bits: float = float("nan")
    confirm_m: int = 0                          # Bonferroni divisor applied (0 = stage off)
    # -- judge reliability (measurement-floor disambiguation, 2026-07-05) ---------------------
    retest_spearman: float = float("nan")       # per-metric test-retest rank corr (temp>0, indep)
    reliability_applicability_agree: float = float("nan")
    attenuation_flag: bool = False              # retest < min_reliability: a ~0 gain is a FLOOR
    # operationalization (GEPA-style rubric iteration against label-free diagnostics)
    op_iterations: int = 0
    op_retest: float = float("nan")
    op_recovery: float = float("nan")


@dataclass
class GlobalInfillResult:
    metrics: List[MetricSpec]                     # base + kept
    ledgers: List[MetricLedger]
    guard_auc_trajectory: List[float]             # after each round (index 0 = base bank)
    guard_bits_trajectory: List[float] = field(default_factory=list)   # V_bits, same indexing
    sm_discover: Optional[ScoreMatrix] = None
    sm_guard: Optional[ScoreMatrix] = None
    rounds: int = 0

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        with open(out / "global_infill_ledger.json", "w") as f:
            json.dump({
                "guard_auc_trajectory": self.guard_auc_trajectory,
                "guard_bits_trajectory": self.guard_bits_trajectory,
                "rounds": self.rounds,
                "ledgers": [asdict(l) for l in self.ledgers],
            }, f, indent=2)


# --------------------------------------------------------------------------------------

def _bank_eval(X_tr: np.ndarray, y_tr: np.ndarray, X_ev: np.ndarray, y_ev: np.ndarray,
               train_frac: float = 1.0, seed: int = 0) -> tuple:
    """(AUC, V_bits) of the bank GLM on the eval split, optionally sub-training.

    ``V_bits`` = held-out log-loss reduction vs the base-rate model, in bits/item — the plug-in
    estimate of I(Y; M_S) that the count certificates compose over (MCC §1). AUC is the familiar
    secondary readout.
    """
    n = len(y_tr)
    if train_frac < 1.0:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n, size=max(20, int(round(train_frac * n))), replace=False)
        X_tr, y_tr = X_tr[keep], y_tr[keep]
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_ev)) < 2:
        return float("nan"), float("nan")
    lr = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    p = np.clip(lr.predict_proba(X_ev)[:, 1], 1e-9, 1 - 1e-9)
    auc = float(roc_auc_score(y_ev, p))
    q = np.clip(y_tr.mean(), 1e-9, 1 - 1e-9)          # base-rate model from TRAIN side
    ll_model = np.mean(y_ev * np.log2(p) + (1 - y_ev) * np.log2(1 - p))
    ll_base = np.mean(y_ev * np.log2(q) + (1 - y_ev) * np.log2(1 - q))
    return auc, float(ll_model - ll_base)


def _bank_auc(X_tr, y_tr, X_ev, y_ev, train_frac: float = 1.0, seed: int = 0) -> float:
    return _bank_eval(X_tr, y_tr, X_ev, y_ev, train_frac, seed)[0]


def _paired_cv_diffs(X_base: np.ndarray, X_aug: np.ndarray, y: np.ndarray,
                     n_folds: int = 5, seed: int = 0) -> tuple:
    """Per-fold paired (AUC, bits) gains of the augmented bank over the base bank."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    d_auc, d_bits = [], []
    for tr, ev in skf.split(X_base, y):
        a0, b0 = _bank_eval(X_base[tr], y[tr], X_base[ev], y[ev])
        a1, b1 = _bank_eval(X_aug[tr], y[tr], X_aug[ev], y[ev])
        if np.isfinite(a0) and np.isfinite(a1):
            d_auc.append(a1 - a0); d_bits.append(b1 - b0)
    return np.array(d_auc), np.array(d_bits)


def _cv_gain(X_base: np.ndarray, X_aug: np.ndarray, y: np.ndarray,
             n_folds: int = 5, seed: int = 0) -> tuple:
    """PAIRED k-fold CV gain of the augmented bank over the base bank on pooled rows.

    Both models are trained/evaluated on identical folds, so the per-fold gains are paired —
    the split-noise that swamps a single small guard split largely cancels. Returns
    (auc_gain_mean, bits_gain_mean, auc_gain_se) over folds.
    """
    d_auc, d_bits = _paired_cv_diffs(X_base, X_aug, y, n_folds, seed)
    if len(d_auc) == 0:
        return float("nan"), float("nan"), float("nan")
    se = float(d_auc.std(ddof=1) / np.sqrt(len(d_auc))) if len(d_auc) > 1 else float("nan")
    return float(d_auc.mean()), float(d_bits.mean()), se


def _nb_corrected_p(diffs: np.ndarray, n_folds: int) -> float:
    """One-sided p (gain > 0) from repeated-CV paired fold gains, with the Nadeau-Bengio
    (2003) variance correction for train-set overlap: Var(mean) = (1/(R*K) + n_ev/n_tr) * s^2,
    with n_ev/n_tr = 1/(K-1) for K-fold. The naive SE understates — fold train sets share
    (K-2)/(K-1) of their rows, so the R*K gains are positively correlated."""
    d = diffs[np.isfinite(diffs)]
    if len(d) < 4:
        return float("nan")
    s2 = float(d.var(ddof=1))
    if s2 <= 0:
        return 0.0 if d.mean() > 0 else 1.0
    var_mean = (1.0 / len(d) + 1.0 / (n_folds - 1)) * s2
    t = float(d.mean()) / math.sqrt(var_mean)
    from scipy import stats
    return float(stats.t.sf(t, df=len(d) - 1))


def _confirm_stage(X_base: np.ndarray, X_aug: np.ndarray, y: np.ndarray,
                   n_repeats: int, n_folds: int = 5, base_seed: int = 0) -> dict:
    """Winner's-curse confirm stage (MCC §4.4): re-estimate the primary-gate gain with
    ``n_repeats`` FRESH fold partitions (seeds disjoint from the primary gate's), pool the
    R*K paired fold gains, and return fresh-mean gains + NB-corrected one-sided p-values.
    The caller applies the Bonferroni-adjusted alpha; this function only measures."""
    all_auc, all_bits = [], []
    for r in range(n_repeats):
        d_auc, d_bits = _paired_cv_diffs(X_base, X_aug, y, n_folds,
                                         seed=base_seed + 7919 * (r + 1))
        all_auc.append(d_auc); all_bits.append(d_bits)
    d_auc = np.concatenate(all_auc) if all_auc else np.array([])
    d_bits = np.concatenate(all_bits) if all_bits else np.array([])
    return {
        "auc_gain": float(d_auc.mean()) if len(d_auc) else float("nan"),
        "bits_gain": float(d_bits.mean()) if len(d_bits) else float("nan"),
        "p_auc": _nb_corrected_p(d_auc, n_folds),
        "p_bits": _nb_corrected_p(d_bits, n_folds),
        "n_diffs": int(len(d_auc)),
    }


def _global_contrast(X: np.ndarray, y: np.ndarray, texts: Sequence[str], cfg,
                     rng: np.random.Generator) -> Optional[Contrast]:
    """WRONG/RIGHT residual contrast over the WHOLE discover split (per class, as in §3)."""
    if len(np.unique(y)) < 2:
        return None
    lr = LogisticRegression(max_iter=2000).fit(X, y)
    p = lr.predict_proba(X)[:, 1]
    # bank-quality guard: the residual is only meaningful if the bank predicts y. Against a
    # garbage/contaminated bank (CW medoids) |y-p| is random and the "WRONG" set is noise that
    # cues surface features — skip rather than propose against noise.
    min_bank_auc = float(getattr(cfg, "min_bank_auc_for_residual", 0.0))
    if min_bank_auc > 0:
        try:
            bank_auc = roc_auc_score(y, p)
        except ValueError:
            bank_auc = float("nan")
        if not np.isfinite(bank_auc) or bank_auc < min_bank_auc:
            print(f"[global_contrast] bank AUC {bank_auc:.3f} < {min_bank_auc} — residual "
                  "contrast uninformative, skipping (use a data/label-grounded arm instead)",
                  flush=True)
            return None
    abs_resid = np.abs(y - p)
    max_chars = int(getattr(cfg, "contrast_max_chars", 4000))

    def top_wrong(mask):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return []
        order = idx[np.argsort(-abs_resid[idx])]
        q = max(2, int(round(len(order) * (1 - cfg.wrong_resid_quantile))))
        return [str(texts[i])[:max_chars] for i in order[:q]]

    wrong_pos = top_wrong(y == 1)
    wrong_neg = top_wrong(y == 0)
    if not wrong_pos or not wrong_neg:
        return None
    k = min(cfg.contrastive_pairs_k, len(wrong_pos), len(wrong_neg))
    pairs = [(wrong_pos[i], wrong_neg[i]) for i in range(k)]
    right_idx = np.argsort(abs_resid)[: max(2, int(round(len(y) * cfg.right_resid_quantile)))]
    return Contrast(node_id="GLOBAL", wrong_pos=wrong_pos, wrong_neg=wrong_neg, pairs=pairs,
                    wrong_disc_idx=np.argsort(-abs_resid)[: len(wrong_pos) + len(wrong_neg)],
                    right_disc_idx=right_idx, n_wrong=len(wrong_pos) + len(wrong_neg))


# --------------------------------------------------------------------------------------

_RECON_PROMPT = """You are reverse-engineering an evaluation criterion from examples.
Below are text excerpts with the VERDICT an unknown evaluation rubric assigned to each
(1 = satisfies the criterion, 0 = does not). Articulate the single most likely criterion
as a reusable YES/NO rubric. Return JSON only: {{"rubric": "<one-paragraph rubric>"}}

{examples}"""


def reconstruction_accuracy(
    metric: MetricSpec, levels: np.ndarray, applicable: np.ndarray, texts: Sequence[str],
    judge_scorer, proposer: Callable[[str], Optional[str]], cfg,
    n_show: int = 24, n_eval: int = 60, n_tries: int = 2, seed: int = 0,
) -> tuple:
    """Articulability R of one metric: rederive its rubric from (text, verdict) pairs, re-execute
    on held-out items, compare to the original verdicts. Returns (agreement, auc, best_rubric).

    ``n_tries`` best-of-k (selected on the show-slice) is the cheap GEPA stand-in; it LOWER-bounds
    the GEPA-optimized R. The reconstructor never sees the rubric text (anchor-free recovery).
    """
    rng = np.random.default_rng(seed)
    app_idx = np.where(applicable)[0]
    if len(app_idx) < n_show + 10:
        return float("nan"), float("nan"), ""
    # mean-split, not median-split: for near-binary levels (e.g. 0/1 with a 40% rate) the
    # median collapses to an endpoint and median-binarization goes constant
    binar = (levels[app_idx] >= np.nanmean(levels[app_idx])).astype(int)
    if len(np.unique(binar)) < 2:
        return float("nan"), float("nan"), ""
    perm = rng.permutation(len(app_idx))
    show, hold = perm[:n_show], perm[n_show: n_show + n_eval]
    ex = "\n---\n".join(
        f"VERDICT={binar[i]}\n{str(texts[app_idx[i]])[:1200]}" for i in show)

    best = (float("nan"), float("nan"), "")
    for t in range(n_tries):
        resp = proposer(_RECON_PROMPT.format(examples=ex))
        if not resp:
            continue
        s = resp.strip(); lo, hi = s.find("{"), s.rfind("}")
        try:
            rubric = json.loads(s[lo:hi + 1]).get("rubric", "").strip()
        except Exception:
            continue
        if not rubric:
            continue
        recon = MetricSpec(metric_id=f"{metric.metric_id}_recon{t}", name=f"recon:{metric.name}",
                           description=rubric, kind="judge", guidance=rubric)
        hold_texts = [str(texts[app_idx[i]]) for i in hold]
        lv, ap = _score_one(recon, hold_texts, judge_scorer)
        m = ap & np.isfinite(lv)
        if m.sum() < 10:
            continue
        pred = (lv[m] >= np.nanmean(lv[m])).astype(int)
        truth = binar[hold][m]
        if len(np.unique(truth)) < 2:
            continue
        agree = float(((pred == truth)[truth == 1].mean() + (pred == truth)[truth == 0].mean()) / 2)
        try:
            auc = float(roc_auc_score(truth, lv[m]))
        except ValueError:
            auc = float("nan")
        if not np.isfinite(best[0]) or agree > best[0]:
            best = (agree, auc, rubric)
    return best


# --------------------------------------------------------------------------------------

def run_global_infill(
    sm_discover: ScoreMatrix, df_discover, y_discover: np.ndarray,
    sm_guard: ScoreMatrix, df_guard, y_guard: np.ndarray,
    metrics: List[MetricSpec], cfg, judge_scorer, proposer,
    max_rounds: int = 6, patience: int = 2, measure_reconstruction: bool = True,
    reconstructor_fn: Optional[Callable] = None,
    proposal_fn: Optional[Callable] = None,
) -> GlobalInfillResult:
    """The global loop. ``proposer(prompt)->str`` and ``judge_scorer`` as in loop.run_infill.

    ``reconstructor_fn`` overrides the built-in best-of-k reconstruction (signature of
    :func:`reconstruction_accuracy`) — this is the GEPA plug-point.
    ``proposal_fn(contrast, known_descriptions, cfg, proposer) -> List[generators.Proposal]``
    selects the proposal ARM (see generators.py); default = the residual arm. All arms flow
    through the identical viability/redundancy/AUC/bits gate, so ledgers are comparable and
    each arm is a capture-recapture list.
    """
    rng = np.random.default_rng(cfg.random_seed)
    texts_d = df_discover[cfg.text_column].astype(str).tolist()
    texts_g = df_guard[cfg.text_column].astype(str).tolist()
    metrics = list(metrics)
    ledgers: List[MetricLedger] = []
    recon = reconstructor_fn or reconstruction_accuracy

    def design(sm, df, spec=None):
        return make_design(sm, df, cfg, spec)

    X_d, fn, _, spec = design(sm_discover, df_discover)
    X_g, _, _, _ = design(sm_guard, df_guard, spec)
    auc0, bits0 = _bank_eval(X_d, y_discover, X_g, y_guard)
    trajectory = [auc0]
    bits_traj = [bits0]
    min_bits_gain = float(getattr(cfg, "min_bits_gain", 0.0))   # 0 => AUC gate only (legacy)
    misses = 0

    if proposal_fn is None:
        from .generators import residual_generator
        proposal_fn = residual_generator()
    queue: List = []                    # proposals awaiting evaluation (multi-candidate arms)

    for rnd in range(1, max_rounds + 1):
        if misses >= patience:
            break
        # feedback distinguishes "already covered" (bank) from "empirically dead" (drops with
        # their measured gain) — without the annotation the proposer cannot learn which
        # directions were tried and failed, and convergent re-proposals burn gate slots.
        known = [m.description or m.name for m in metrics] + [
            f"{l.description or l.name} (TRIED round {l.round}: "
            f"measured {'%+.4f' % l.bits_gain if l.bits_gain is not None else 'no'} bits — "
            f"{l.status.split(':')[-1]})"
            for l in ledgers if l.status.startswith("dropped")]
        if not queue:
            contrast = _global_contrast(X_d, y_discover, texts_d, cfg, rng)
            # a proposer/API failure is an EMPTY ROUND, never a crashed arm (the 2026-07-05
            # overnight died mid-arm on an exhausted-retries RuntimeError from the label_contrast
            # arm's direct proposer call — arms must be failure-isolated for the flux read)
            try:
                queue = list(proposal_fn(contrast, known, cfg, proposer) or [])
            except Exception as e:
                print(f"[global_infill] proposal_fn failure ({type(e).__name__}: {e}) — "
                      "empty round", flush=True)
                queue = []
        if not queue:
            misses += 1
            continue
        prop = queue.pop(0)
        cand = MetricSpec(metric_id=f"g{rnd}_{abs(hash(prop.name)) % 99991}", name=prop.name,
                          description=prop.description, kind="judge", guidance=prop.rubric)
        led = MetricLedger(
            name=prop.name, description=prop.description, rubric=prop.rubric, round=rnd,
            status="pending", guard_auc_before=trajectory[-1], guard_bits_before=bits_traj[-1],
            n_proposal_examples=int(getattr(prop, "n_examples", 0)))
        led.generator = getattr(prop, "generator", "residual")

        # content-only guard: drop surface/format proposals ex-ante (thin-signal leakage)
        if getattr(cfg, "content_only_guard", False):
            from .generators import is_surface_only
            if is_surface_only(prop.name, prop.rubric):
                led.status = "dropped:surface"
                ledgers.append(led); misses += 1; continue

        # duplicate guard: convergent re-proposals (Clear Resolution x5, Tonal Clarity x3)
        # evade name-dedup via paraphrase; an embedding match against THIS LEG's prior
        # proposals skips the materialization + gate slot (status distinguishes convergence —
        # itself a signal stage-2 replication uses — from novelty).
        if getattr(cfg, "dedup_proposals", True) and ledgers:
            try:
                from sentence_transformers import SentenceTransformer
                global _DEDUP_MODEL
                if "_DEDUP_MODEL" not in globals() or _DEDUP_MODEL is None:
                    _DEDUP_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
                prior = [f"{l.name}. {l.description}" for l in ledgers]
                embs = _DEDUP_MODEL.encode(prior + [f"{prop.name}. {prop.description}"],
                                           normalize_embeddings=True)
                sim = float((embs[:-1] @ embs[-1]).max())
                if sim > float(getattr(cfg, "dedup_cosine", 0.86)):
                    led.status = f"dropped:duplicate(cos={sim:.2f})"
                    ledgers.append(led); misses += 1; continue
            except Exception:
                pass

        # OPERATIONALIZE (2026-07-09): GEPA-style iteration of the rubric against label-free
        # instrument diagnostics (retest, MI-recovery, distribution) BEFORE gate scoring —
        # every proposed metric, every arm. The gate still owns all label contact.
        if getattr(cfg, "operationalize_proposals", False):
            try:
                from .operationalize import operationalize_rubric
                cal = texts_d[: min(120, len(texts_d))]
                op = operationalize_rubric(prop.name, prop.description, prop.rubric,
                                           cal, judge_scorer, proposer, cfg)
                if op.rubric != prop.rubric:
                    cand.guidance = op.rubric
                    led.rubric = op.rubric
                led.op_iterations = op.iterations
                led.op_retest = op.retest
                led.op_recovery = op.recovery
            except Exception as e:
                print(f"[operationalize] skipped ({type(e).__name__}: {e})", flush=True)

        lv_d, ap_d = _score_one(cand, texts_d, judge_scorer)
        led.applicability_discover = float(ap_d.mean())
        # viability on discover (same thresholds as the tree path)
        if ap_d.mean() < cfg.viability_min_applicability or \
                np.nanstd(lv_d[ap_d]) < cfg.viability_min_std:
            led.status = "dropped:viability"
            ledgers.append(led); misses += 1; continue
        # redundancy vs current bank (R^2 of new levels on existing X, applicable rows)
        from sklearn.linear_model import LinearRegression
        m = ap_d & np.isfinite(lv_d)
        if m.sum() >= 20:
            r2 = LinearRegression().fit(X_d[m], lv_d[m]).score(X_d[m], lv_d[m])
            led.redundancy_r2 = float(r2)
            if r2 > cfg.tau_redundant:
                led.status = "dropped:redundant"
                ledgers.append(led); misses += 1; continue

        # judge reliability (measurement-floor disambiguation): a per-metric test-retest stamped
        # on EVERY scored proposal — so a ~0 gain reads as genuine-null (reliable but non-
        # predictive) vs executor-can't-apply (low retest -> attenuation). Diagnostic, never a
        # drop; a tacit/hard-to-apply metric is still a legitimate member of the articulable class.
        if getattr(cfg, "measure_reliability", False):
            try:
                from .reliability import judge_test_retest
                rel = judge_test_retest(cand, texts_d, cfg,
                                        temperature=getattr(cfg, "reliability_temperature", 0.6),
                                        seed=cfg.random_seed)
                led.retest_spearman = rel["retest_spearman"]
                led.reliability_applicability_agree = rel["applicability_agree"]
                led.attenuation_flag = rel["attenuation_flag"]
            except Exception as e:
                print(f"[global_infill] reliability read failed ({type(e).__name__})", flush=True)

        lv_g, ap_g = _score_one(cand, texts_g, judge_scorer)
        led.applicability_guard = float(ap_g.mean())
        sm_d2 = _append_column(sm_discover, lv_d, ap_d, cand)
        sm_g2 = _append_column(sm_guard, lv_g, ap_g, cand)
        X_d2, _, _, spec2 = design(sm_d2, df_discover)
        X_g2, _, _, _ = design(sm_g2, df_guard, spec2)
        auc1, bits1 = _bank_eval(X_d2, y_discover, X_g2, y_guard)
        led.guard_auc_after, led.guard_bits_after = auc1, bits1
        if getattr(cfg, "acceptance_eval", "guard") == "cv":
            # paired k-fold CV over pooled discover+guard rows: split-noise cancels, so the
            # gate compares gain to a much smaller SE than one small guard split affords
            Xp0 = np.vstack([X_d, X_g]); Xp1 = np.vstack([X_d2, X_g2])
            yp = np.concatenate([y_discover, y_guard])
            led.auc_gain, led.bits_gain, _ = _cv_gain(Xp0, Xp1, yp, seed=cfg.random_seed)
        else:
            led.auc_gain = auc1 - trajectory[-1]
            led.bits_gain = bits1 - bits_traj[-1]

        # track 1: data curve — gain at 25/50/100% of the training rows, measured with the SAME
        # instrument as the acceptance gate (CV mode: paired-fold gain on a row subsample;
        # guard mode: single-split delta) so the curve is comparable to the gate
        full_gain = led.auc_gain
        if getattr(cfg, "acceptance_eval", "guard") == "cv":
            Xp0 = np.vstack([X_d, X_g]); Xp1 = np.vstack([X_d2, X_g2])
            yp = np.concatenate([y_discover, y_guard])
            rng_c = np.random.default_rng(cfg.random_seed)
            for frac in (0.25, 0.5, 1.0):
                keep = rng_c.choice(len(yp), size=max(60, int(round(frac * len(yp)))), replace=False)
                g_auc, _, _ = _cv_gain(Xp0[keep], Xp1[keep], yp[keep], seed=cfg.random_seed)
                led.data_curve[str(frac)] = float(g_auc)
        else:
            for frac in (0.25, 0.5, 1.0):
                a_b = _bank_auc(X_d, y_discover, X_g, y_guard, frac, cfg.random_seed)
                a_a = _bank_auc(X_d2, y_discover, X_g2, y_guard, frac, cfg.random_seed)
                led.data_curve[str(frac)] = float(a_a - a_b)
        if np.isfinite(full_gain) and full_gain > 0:
            for frac in (0.25, 0.5, 1.0):
                if led.data_curve[str(frac)] >= 0.5 * full_gain:
                    led.min_train_frac = frac
                    break

        # acceptance: AUC gate always; bits gate additionally when min_bits_gain > 0
        # (the count certificates only compose over bits — MCC §4.3)
        if not np.isfinite(auc1) or led.auc_gain < cfg.min_auc_gain:
            led.status = f"dropped:auc_gain<{cfg.min_auc_gain}"
            ledgers.append(led); misses += 1; continue
        if min_bits_gain > 0 and (not np.isfinite(led.bits_gain) or led.bits_gain < min_bits_gain):
            led.status = f"dropped:bits_gain<{min_bits_gain}"
            ledgers.append(led); misses += 1; continue

        # confirm stage (winner's-curse control, MCC §4.4): the primary gate alone is
        # anti-conservative under selection over many proposals — fresh-seed repeated CV must
        # REPLICATE the gain (same delta floor) and clear Bonferroni-adjusted significance.
        n_rep = int(getattr(cfg, "confirm_n_repeats", 0))
        if n_rep > 0 and getattr(cfg, "acceptance_eval", "guard") == "cv":
            Xp0 = np.vstack([X_d, X_g]); Xp1 = np.vstack([X_d2, X_g2])
            yp = np.concatenate([y_discover, y_guard])
            m_bonf = int(getattr(cfg, "gate_bonferroni_m", 0)) or max_rounds
            conf = _confirm_stage(Xp0, Xp1, yp, n_repeats=n_rep, base_seed=cfg.random_seed)
            led.confirm_auc_gain, led.confirm_bits_gain = conf["auc_gain"], conf["bits_gain"]
            led.confirm_p_auc, led.confirm_p_bits = conf["p_auc"], conf["p_bits"]
            led.confirm_m = m_bonf
            alpha_adj = float(getattr(cfg, "gate_alpha", 0.05)) / max(1, m_bonf)
            ok = (np.isfinite(conf["auc_gain"]) and conf["auc_gain"] >= cfg.min_auc_gain
                  and np.isfinite(conf["p_auc"]) and conf["p_auc"] <= alpha_adj)
            if min_bits_gain > 0:
                ok = ok and (np.isfinite(conf["bits_gain"]) and conf["bits_gain"] >= min_bits_gain
                             and np.isfinite(conf["p_bits"]) and conf["p_bits"] <= alpha_adj)
            if not ok:
                led.status = (f"dropped:confirm(p_auc={conf['p_auc']:.4g},"
                              f"p_bits={conf['p_bits']:.4g},alpha_adj={alpha_adj:.4g})")
                ledgers.append(led); misses += 1; continue

        # track 3: articulability of the accepted metric
        if measure_reconstruction:
            try:
                agree, rauc, rrub = recon(cand, lv_d, ap_d, texts_d, judge_scorer, proposer, cfg)
                led.reconstruction_agreement, led.reconstruction_auc = agree, rauc
                led.reconstruction_rubric = rrub
            except Exception as e:
                print(f"[global_infill] reconstruction failure ({type(e).__name__}) — "
                      "leaving R fields NaN", flush=True)

        led.status = "kept"
        ledgers.append(led)
        metrics.append(cand)
        sm_discover, sm_guard = sm_d2, sm_g2
        X_d, X_g = X_d2, X_g2
        trajectory.append(auc1)
        bits_traj.append(bits1)
        misses = 0

    return GlobalInfillResult(metrics=metrics, ledgers=ledgers,
                              guard_auc_trajectory=trajectory, guard_bits_trajectory=bits_traj,
                              sm_discover=sm_discover, sm_guard=sm_guard, rounds=len(trajectory) - 1)
