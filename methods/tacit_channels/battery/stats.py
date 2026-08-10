"""Battery statistics — the corrected, unit-tested statistical core for W1+ tallies.

Every statistic here encodes a lesson from the 2026-07-23 audit
(notes/2026-07-23__w1-code-results-audit.md):
  - leak_stats returns the CROSS-CELL NULL companion (leak_specific = same - cross), never
    the raw same-cell correlation alone;
  - holistic_residual chooses ridge strength by inner split and FAILS CLOSED on degenerate
    targets (returns a verdict, not a garbage number);
  - confidence_scale_valid is the scale-use gate as code (the parse-rate gate alone missed
    constant/binary collapse);
  - chance levels are explicit constants, never implied .5.

Pure numpy; no I/O. Tallies import from here; tests attack each statistic with synthetic
agents of KNOWN profile.
"""
from __future__ import annotations

import numpy as np

from methods.tacit_channels.channels.common import _rankdata, spearman

ITEM_AGREEMENT_CHANCE = 2.0 / 3.0   # E[1 - |r_a - r_b|/(n-1)] for independent rankings


def zrank(v: np.ndarray) -> np.ndarray:
    r = _rankdata(np.asarray(v, float))
    s = r.std()
    return (r - r.mean()) / (s if s > 0 else 1.0)


def leak_stats(excl_by_form: dict, tf_by_form: dict, cross_tf: list,
               mask: np.ndarray) -> dict | None:
    """Exclusion leak with the cross-cell null companion.

    excl_by_form / tf_by_form: {form: vector} for ONE cell; cross_tf: tf vectors of OTHER
    cells (any form) as the generic-similarity null. leak_specific = same - cross is the
    reportable statistic; leak_self alone is NEVER a headline (audit finding #2).
    """
    same = [spearman(np.asarray(e)[mask], np.asarray(tf_by_form[f])[mask])
            for f, e in excl_by_form.items() if f in tf_by_form]
    same = [s for s in same if not np.isnan(s)]
    if not same:
        return None
    cross = []
    for f, e in excl_by_form.items():
        for other in cross_tf:
            s = spearman(np.asarray(e)[mask], np.asarray(other)[mask])
            if not np.isnan(s):
                cross.append(s)
    leak_self = float(np.median(same))
    if not cross:  # fail closed: without a cross-cell null, leak_specific would silently
        return {"leak_self": leak_self, "leak_cross": None,   # equal the forbidden
                "leak_specific": None, "n_cross": 0}          # leak_self headline
    leak_cross = float(np.median(cross))
    return {"leak_self": leak_self, "leak_cross": leak_cross,
            "leak_specific": leak_self - leak_cross, "n_cross": len(cross)}


def not_gap(tf_vecs: list, neg_vecs: list, t_ref: np.ndarray,
            mask: np.ndarray) -> dict | None:
    """Knowing-using NOT: adverse tf rho vs target MINUS adverse negated rho vs REVERSED
    target. gap 0 = perfect explicit NOT; gap ~= 2*tf_rho = negation fully ignored."""
    def adverse(vecs, ref):
        vals = [spearman(np.asarray(v)[mask], ref[mask]) for v in vecs]
        vals = [x for x in vals if not np.isnan(x)]
        return float(min(vals)) if vals else None
    tf_rho = adverse(tf_vecs, t_ref)
    neg_rho = adverse(neg_vecs, -t_ref)
    if tf_rho is None or neg_rho is None:
        return None
    return {"tf_rho": tf_rho, "neg_rho_vs_reversed": neg_rho,
            "not_gap": tf_rho - neg_rho}


def composition_rho(comp_vecs: list, member_refs: list, mask: np.ndarray,
                    mode: str = "min_z") -> float | None:
    """Composed-judgment match against a blend reference (v0: elementwise min of rank-z
    member references = soft AND). v1 replaces the blend with the target's own composed
    vector — same function, member_refs=[target_composed]."""
    if mode == "min_z" and len(member_refs) >= 2:
        ref = np.minimum.reduce([zrank(m) for m in member_refs])
    else:
        ref = np.asarray(member_refs[0], float)
    vals = [spearman(np.asarray(v)[mask], ref[mask]) for v in comp_vecs]
    vals = [x for x in vals if not np.isnan(x)]
    return float(min(vals)) if vals else None


def holistic_residual(y: np.ndarray, X: np.ndarray, fit_mask: np.ndarray,
                      eval_mask: np.ndarray, y_std_floor: float = 0.05,
                      alphas: tuple = (1.0, 10.0, 100.0, 1000.0)) -> dict:
    """Unnamed-share estimator with degeneracy guard + inner-split ridge selection.

    Returns {"verdict": "ok"|"degenerate", "oos_r2": float|None, ...}. NEVER returns a
    number for a floor-collapsed y (audit finding: v0 reported R2 = -30 on a near-constant
    vector)."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if not np.isfinite(X).all():  # one NaN would poison its whole z-scored column
        bad = int((~np.isfinite(X)).sum())
        raise ValueError(f"holistic_residual: {bad} non-finite entries in X — "
                         "complete or drop predictors upstream")
    s_fit = float(y[fit_mask].std()) if np.count_nonzero(fit_mask) else float("nan")
    s_ev = float(y[eval_mask].std()) if np.count_nonzero(eval_mask) else float("nan")
    # degenerate iff: empty/NaN side, below floor, or EXACTLY constant (catches floor=0)
    if (not np.isfinite(s_fit) or not np.isfinite(s_ev)
            or s_fit < y_std_floor or s_ev < y_std_floor
            or s_fit == 0.0 or s_ev == 0.0):
        return {"verdict": "degenerate", "oos_r2": None,
                "y_std": float(y.std()) if y.size else None}
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    fit_idx = np.where(fit_mask)[0]
    inner_fit, inner_val = fit_idx[::2], fit_idx[1::2]

    def ridge_pred(train_idx, alpha, at_idx):
        A = Xz[train_idx]
        w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ y[train_idx])
        return Xz[at_idx] @ w

    best_alpha, best = None, -np.inf
    for a in alphas:
        r = y[inner_val] - ridge_pred(inner_fit, a, inner_val)
        r2 = 1 - r.var() / y[inner_val].var()
        if r2 > best:
            best, best_alpha = r2, a
    resid = y[eval_mask] - ridge_pred(fit_idx, best_alpha, np.where(eval_mask)[0])
    oos = 1 - resid.var() / y[eval_mask].var()
    return {"verdict": "ok", "oos_r2": float(oos), "alpha": best_alpha,
            # share clipped to [0,1] for reporting; oos_r2 stays raw (can be < 0)
            "unnamed_share": float(np.clip(1 - oos, 0.0, 1.0)),
            "y_std": float(y.std())}


def confidence_scale_valid(conf: np.ndarray, min_unique: int = 8,
                           min_cell_std: float = 5.0) -> dict:
    """Scale-use gate (0-100 verbalized confidence). Parse rate alone is insufficient —
    a load whose values are one constant or a {0,100} binary is VOID."""
    finite = conf[np.isfinite(conf)]
    n_unique = len(np.unique(finite)) if finite.size else 0
    cell_stds = [np.nanstd(row[np.isfinite(row)]) for row in conf
                 if np.isfinite(row).sum() >= 10]
    med_std = float(np.median(cell_stds)) if cell_stds else 0.0
    return {"n_unique": n_unique, "median_cell_std": med_std,
            "valid": bool(n_unique >= min_unique and med_std >= min_cell_std)}


def conf_acc_stats(conf: np.ndarray, agreement: np.ndarray) -> dict | None:
    """Dienes statistics for one cell: confidence-accuracy correlation + bottom-quartile
    agreement, reported WITH the chance level (audit finding #3)."""
    ok = np.isfinite(conf) & np.isfinite(agreement)
    if ok.sum() < 50:
        return None
    c, a = conf[ok], agreement[ok]
    if c.std() == 0:
        return {"conf_acc_corr": None, "guess_agreement": None,
                "degenerate_confidence": True, "degenerate_agreement": False}
    if a.std() == 0:  # constant agreement -> corr undefined; None, never a bare NaN
        return {"conf_acc_corr": None, "guess_agreement": None,
                "degenerate_confidence": False, "degenerate_agreement": True}
    q = np.quantile(c, 0.25)
    return {"conf_acc_corr": spearman(c, a),
            "guess_agreement": float(a[c <= q].mean()),
            "guess_agreement_minus_chance": float(a[c <= q].mean()
                                                  - ITEM_AGREEMENT_CHANCE),
            "degenerate_confidence": False, "degenerate_agreement": False}
