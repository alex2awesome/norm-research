#!/usr/bin/env python3
"""Closed-form LEACE eraser (LEAst-squares Concept Erasure), implemented from the
equations of Belrose et al. 2023 ("LEACE: Perfect linear concept erasure in
closed form", NeurIPS 2023), Theorem 4.3.

Given representations X in R^d and concept labels Z in R^k, the LEACE eraser is
the affine map

    r(x) = x - W^+ P_{W Sigma_XZ} W (x - mu_X)

where
    Sigma_XX  = Cov(X)                      (d x d)
    W         = Sigma_XX^{-1/2}             (whitening; pseudo-inverse sqrt)
    W^+       = Sigma_XX^{+1/2}             (unwhitening)
    Sigma_XZ  = Cov(X, Z)                   (d x k)
    P_{M}     = orthogonal projection onto colspace(M), here M = W Sigma_XZ.

Sanity of the closed form: Cov(r(X), Z) = Sigma_XZ - W^+ P W Sigma_XZ
= W^+ (I - P)(W Sigma_XZ) = 0 because P projects onto col(W Sigma_XZ).

CERTIFICATE FORM (corrected 2026-08-10 after the lit review,
notes/2026-08-10__litreview_spurious_debiasing.md R1a):
  * For CATEGORICAL one-hot Z, zero cross-covariance is equivalent to linear
    guardedness -- no linear classifier beats the constant predictor under ANY
    convex loss (Belrose et al. Def 2.3 / Thm 3.1).
  * For CONTINUOUS Z, the guarantee is ONLY with respect to the ordinary
    least-squares loss (their S4.3); the any-convex-loss form does NOT provably
    extend.  The eraser math below is concept-agnostic -- the certificate you
    hold is determined by how the caller ENCODES Z.  Battery discipline: bin
    continuous nuisances to train-decile one-hot before fitting (see
    run_battery_leace.onehot_bins), so the strong categorical form applies and
    any coarsening of the bins (e.g. a median-split probe target, the median
    being one of the decile edges) is a linear function of the erased Z and
    inherits the certificate.
  * Nothing is guaranteed about NONLINEAR readers under either form (their own
    stated non-guarantee), which is exactly what the V2 nonlinear-residue
    readout quantifies; and even categorical guardedness binds only
    binary-linear readers -- a multiclass softmax can recover Z despite it
    (Ravfogel et al. ACL 2023, Thm 3.4), the ceiling on any claim made here.

Everything is fit on TRAIN rows only and applied everywhere (concept labels of
eval/test rows never touch the transformation).

Self-check: `python leace.py` runs the synthetic hand-check (planted linear
concept; verifies machine-precision cross-covariance kill, chance-level linear
probe, surgical distortion) and exits nonzero on failure.
"""
from __future__ import annotations

import numpy as np

RCOND_COV = 1e-9      # relative eigenvalue cutoff for the covariance pseudo-sqrt
RCOND_RANK = 1e-6     # relative singular-value cutoff for rank(W Sigma_XZ)


class LeaceEraser:
    """Fit on (X_train, Z_train); apply to any X. All math in float64."""

    def __init__(self, rcond_cov: float = RCOND_COV, rcond_rank: float = RCOND_RANK):
        self.rcond_cov = rcond_cov
        self.rcond_rank = rcond_rank

    def fit(self, X: np.ndarray, Z: np.ndarray, eig=None):
        """eig: optional (evals, evecs) of Cov(X) from precompute_eig(X), so many
        erasers over the same X pay for one eigendecomposition."""
        X = np.asarray(X, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z[:, None]
        n, d = X.shape
        assert Z.shape[0] == n and n > d / 2, f"n={n}, d={d}: need n comfortably > 0"
        self.mu_x = X.mean(0)
        Xc = X - self.mu_x
        Zc = Z - Z.mean(0)

        # covariance pseudo-sqrt via eigendecomposition (Sigma is symmetric PSD)
        if eig is None:
            Sigma = (Xc.T @ Xc) / (n - 1)
            evals, evecs = np.linalg.eigh(Sigma)
        else:
            evals, evecs = eig
        cut = evals.max() * self.rcond_cov
        keep = evals > cut
        self.cov_rank = int(keep.sum())
        s = evals[keep]
        V = evecs[:, keep]
        # W = V diag(s^-1/2) V^T ; W^+ = V diag(s^+1/2) V^T   (restricted to range)
        self._V = V
        self._s_isqrt = 1.0 / np.sqrt(s)
        self._s_sqrt = np.sqrt(s)

        Sxz = (Xc.T @ Zc) / (n - 1)                     # d x k
        M = V @ (self._s_isqrt[:, None] * (V.T @ Sxz))  # W Sigma_XZ, d x k
        U, sing, _ = np.linalg.svd(M, full_matrices=False)
        rk = int((sing > (sing.max() if sing.size else 0.0) * self.rcond_rank).sum())
        self.proj_rank = rk
        Ur = U[:, :rk]                                  # d x rk, orthonormal
        # eraser: x - W^+ Ur Ur^T W (x - mu).  Store B = W^+ Ur (d x rk) and
        # C = Ur^T W (rk x d) so apply() never forms a d x d matrix.
        self._B = V @ (self._s_sqrt[:, None] * (V.T @ Ur))
        self._C = (V @ (self._s_isqrt[:, None] * (V.T @ Ur))).T
        return self

    def apply(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Xc = X - self.mu_x
        return (X - Xc @ self._C.T @ self._B.T).astype(np.float64)

    # -- diagnostics ---------------------------------------------------------
    def crosscov_max(self, X: np.ndarray, Z: np.ndarray, standardize: bool = False) -> float:
        """max |Cov(r(X), Z)| entry, on any row set.  standardize=True returns the
        max |corr| instead (scale-free; the right units for a holdout readout)."""
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z[:, None]
        R = self.apply(X)
        Rc = R - R.mean(0)
        Zc = Z - Z.mean(0)
        C = Rc.T @ Zc / (len(R) - 1)
        if standardize:
            sx = Rc.std(0)
            sz = Zc.std(0)
            C = C / np.outer(np.where(sx < 1e-12, 1.0, sx), np.where(sz < 1e-12, 1.0, sz))
        return float(np.abs(C).max())

    def distortion(self, X: np.ndarray) -> float:
        """mean ||r(x) - x|| / mean ||x - mu||  (how surgical the edit is)."""
        X = np.asarray(X, dtype=np.float64)
        R = self.apply(X)
        return float(np.linalg.norm(R - X, axis=1).mean()
                     / max(np.linalg.norm(X - self.mu_x, axis=1).mean(), 1e-12))


def precompute_eig(X: np.ndarray):
    """One eigendecomposition of Cov(X) reusable across erasers on the same X."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(0)
    Sigma = (Xc.T @ Xc) / (len(X) - 1)
    return np.linalg.eigh(Sigma)


def _selfcheck() -> int:
    """Synthetic hand-check with a known planted linear concept."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    n, d = 4000, 64
    z = (rng.random(n) < 0.4).astype(float)            # binary concept
    g = rng.normal(size=n)                             # continuous 2nd concept
    X = rng.normal(size=(n, d))
    X[:, 0] += 3.0 * z                                 # axis-aligned plant
    X[:, 3] -= 2.0 * z                                 # second direction
    X[:, 5] += 1.0 * g
    X = X @ rng.normal(size=(d, d)) * 0.3 + X          # mix the axes
    tr = np.arange(n) < n // 2
    Z = np.column_stack([z, g])

    er = LeaceEraser().fit(X[tr], Z[tr])
    R = er.apply(X)

    cc_tr = er.crosscov_max(X[tr], Z[tr])               # raw units: machine zero
    cc_te = er.crosscov_max(X[~tr], Z[~tr], standardize=True)   # holdout: max |corr|
    lin = LogisticRegression(max_iter=2000).fit(R[tr], z[tr].astype(int))
    auc_tr = roc_auc_score(z[tr], lin.decision_function(R[tr]))
    auc_te = roc_auc_score(z[~tr], lin.decision_function(R[~tr]))
    lin0 = LogisticRegression(max_iter=2000).fit(X[tr], z[tr].astype(int))
    auc_raw = roc_auc_score(z[~tr], lin0.decision_function(X[~tr]))
    dist = er.distortion(X)

    print(f"selfcheck: raw linear AUC (holdout)      = {auc_raw:.4f}  (should be ~1)")
    print(f"selfcheck: post-LEACE linear AUC train   = {auc_tr:.4f}  (should be ~.5)")
    print(f"selfcheck: post-LEACE linear AUC holdout = {auc_te:.4f}  (should be ~.5)")
    print(f"selfcheck: max |crosscov| train (raw) = {cc_tr:.2e} ; holdout max |corr| = {cc_te:.4f}")
    print(f"selfcheck: distortion = {dist:.4f}  proj_rank = {er.proj_rank} (expect 2)")

    ok = (auc_raw > 0.95 and cc_tr < 1e-10 and abs(auc_tr - 0.5) < 0.02
          and abs(auc_te - 0.5) < 0.05 and cc_te < 0.10
          and er.proj_rank == 2 and dist < 0.5)
    print("selfcheck:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
