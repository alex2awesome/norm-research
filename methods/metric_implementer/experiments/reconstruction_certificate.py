"""Auxiliary behavioral-replay certificates after reconstruction.

This is not the production CR-3 value functional and does not bound its primary
MCQ identification-lift readout. It is retained as a separately scoped secondary
measurement for experiments that re-execute a frozen MCQ choice on a fresh item
lockbox. It requires no anchor, silver label, human judgment, or external target.
A prompt produces its own hard verdict ``M_p``. A frozen reconstructor sees
only training ``(item, M_p)`` pairs, emits one or more prompts, and those prompts are
re-executed on a fresh iid item lockbox to produce ``Mhat_p``.

The auxiliary behavioral-replay objective is polarity-invariant mutual information
after MCQ identification and canonical re-execution:

    R_E(p) = I(M_p ; Mhat_p).

For a frozen finite codebook ``C_b``, the reconstructed channel is a convex
mixture of canonical option channels.  Data processing and channel convexity give

    R_E(p) <= max_{c in C_b} I(X; E(c, X)).

This is an estimator-specific upper bound for that secondary replay channel. The universal binary caps
(``1`` Shannon bit and ``1/2`` TVD) remain valid fallbacks.  The Shannon fallback
gap has the exact decomposition

    1 - R_shannon = (1 - H(M_p)) + H(M_p | Mhat_p),

so balanced use of the verdict range and reconstructability are both necessary.
Neither term says that the behavior is substantively correct; that is external
validity and is outside this certificate.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.stats import beta as _beta


SCHEMA_VERSION = "reconstruction-global-prompt-v1"
_EPS = 1e-12


def _h_binary(p: float) -> float:
    p = float(np.clip(p, 0.0, 1.0))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def _cp_interval(successes: int, n: int, alpha: float) -> tuple[float, float]:
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("invalid binomial count")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    lo = 0.0 if successes == 0 else float(
        _beta.ppf(alpha / 2.0, successes, n - successes + 1))
    hi = 1.0 if successes == n else float(
        _beta.ppf(1.0 - alpha / 2.0, successes + 1, n - successes))
    return lo, hi


def _entropy_interval(probability_interval: tuple[float, float]) -> tuple[float, float]:
    lo, hi = probability_interval
    h_lo = min(_h_binary(lo), _h_binary(hi))
    h_hi = 1.0 if lo <= 0.5 <= hi else max(_h_binary(lo), _h_binary(hi))
    return float(h_lo), float(h_hi)


def _polarity_invariant_error_upper(
    mismatch_mean: float,
    radius: float,
) -> tuple[float, tuple[float, float]]:
    lo = max(0.0, float(mismatch_mean) - float(radius))
    hi = min(1.0, float(mismatch_mean) + float(radius))
    if lo <= 0.5 <= hi:
        upper = 0.5
    elif hi < 0.5:
        upper = hi
    else:
        upper = 1.0 - lo
    return float(upper), (float(lo), float(hi))


def _joint_from_mixture(target: np.ndarray, reconstructed_probability: np.ndarray) -> np.ndarray:
    y = np.asarray(target, float)
    q = np.asarray(reconstructed_probability, float)
    return np.asarray([
        [np.mean((1.0 - y) * (1.0 - q)), np.mean((1.0 - y) * q)],
        [np.mean(y * (1.0 - q)), np.mean(y * q)],
    ], float)


def _shannon_mi(joint: np.ndarray) -> float:
    prod = np.outer(joint.sum(axis=1), joint.sum(axis=0))
    keep = joint > 0.0
    return float(max(0.0, np.sum(joint[keep] * np.log2(joint[keep] / prod[keep]))))


def mcq_reconstruction_certificate(
    prompt_verdicts: np.ndarray,
    option_verdicts: np.ndarray,
    reconstruction_choices: np.ndarray,
    reconstruction_draw_assignment: np.ndarray,
    *,
    option_ids: list[str] | tuple[str, ...],
    codebook_frozen_before_candidate_optimization: bool,
    choices_frozen_before_lockbox: bool,
    assignments_iid_uniform_and_predeclared: bool,
    lockbox_unused_for_optimization: bool,
    alpha: float = 0.05,
    epsilon_bits: float = 0.05,
    scope: Mapping[str, object] | None = None,
) -> dict:
    """Certify Reconstruction-MCQ recovery under one frozen canonical codebook.

    ``prompt_verdicts`` is the candidate prompt's own binary annotation on each
    fresh item. ``option_verdicts[k]`` is executor ``E`` re-executing canonical
    option ``k`` on the same items. The reconstructor makes one codebook choice per
    reconstruction draw using training annotations only. Before lockbox access,
    each item is assigned iid-uniformly to one reconstruction draw; its recovered
    verdict is the corresponding chosen option's verdict.

    This matched design produces iid binary pairs and permits exact binomial
    confidence components. No external or silver label is an input.
    """
    premises = {
        "codebook_frozen_before_candidate_optimization":
            codebook_frozen_before_candidate_optimization,
        "choices_frozen_before_lockbox": choices_frozen_before_lockbox,
        "assignments_iid_uniform_and_predeclared": assignments_iid_uniform_and_predeclared,
        "lockbox_unused_for_optimization": lockbox_unused_for_optimization,
    }
    failed = [name for name, holds in premises.items() if not holds]
    if failed:
        raise ValueError("certificate premise failed: " + ", ".join(failed))
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if not np.isfinite(epsilon_bits) or not 0.0 <= epsilon_bits <= 1.0:
        raise ValueError("epsilon_bits must lie in [0, 1]")

    y_raw = np.asarray(prompt_verdicts, float)
    options_raw = np.asarray(option_verdicts, float)
    choices_raw = np.asarray(reconstruction_choices)
    assignment_raw = np.asarray(reconstruction_draw_assignment)
    if y_raw.ndim != 1 or len(y_raw) < 2:
        raise ValueError("prompt_verdicts must be a one-dimensional vector with at least two items")
    if options_raw.ndim != 2 or options_raw.shape[1] != len(y_raw) or options_raw.shape[0] < 2:
        raise ValueError("option_verdicts must have shape (at least 2 options, n_items)")
    if choices_raw.ndim != 1 or len(choices_raw) < 1:
        raise ValueError("reconstruction_choices must be a nonempty one-dimensional vector")
    if assignment_raw.ndim != 1 or len(assignment_raw) != len(y_raw):
        raise ValueError("reconstruction_draw_assignment must have one entry per lockbox item")
    if len(option_ids) != options_raw.shape[0] or len(set(map(str, option_ids))) != len(option_ids):
        raise ValueError("option_ids must be unique and match option_verdicts rows")
    if (np.any(~np.isfinite(y_raw)) or np.any(~np.isfinite(options_raw))
            or np.any((y_raw != 0.0) & (y_raw != 1.0))
            or np.any((options_raw != 0.0) & (options_raw != 1.0))):
        raise ValueError("bound-grade prompt and option verdicts must be complete hard binary arrays")
    if (not np.issubdtype(choices_raw.dtype, np.integer)
            or np.any(choices_raw < 0) or np.any(choices_raw >= options_raw.shape[0])):
        raise ValueError("every reconstruction choice must index a canonical option")
    if (not np.issubdtype(assignment_raw.dtype, np.integer)
            or np.any(assignment_raw < 0) or np.any(assignment_raw >= len(choices_raw))):
        raise ValueError("every item assignment must index a reconstruction draw")

    y = y_raw.astype(np.uint8)
    options = options_raw.astype(np.uint8)
    choices = choices_raw.astype(int)
    assignment = assignment_raw.astype(int)
    selected_option = choices[assignment]
    recovered = options[selected_option, np.arange(len(y))]
    n = len(y)

    joint = np.zeros((2, 2), float)
    for a in (0, 1):
        for b in (0, 1):
            joint[a, b] = float(np.mean((y == a) & (recovered == b)))
    recovery_hat = _shannon_mi(joint)
    target_entropy_hat = _h_binary(float(y.mean()))
    option_entropy_hat = np.asarray([_h_binary(float(row.mean())) for row in options])
    codebook_upper_hat = float(np.max(option_entropy_hat))

    # The codebook upper and candidate lower each receive half the familywise
    # error budget. Option prevalence intervals are simultaneous by Bonferroni.
    alpha_upper = alpha / 2.0
    alpha_lower = alpha / 2.0
    option_component_alpha = alpha_upper / len(options)
    option_intervals = [
        _cp_interval(int(row.sum()), n, option_component_alpha) for row in options
    ]
    option_entropy_ucb = np.asarray([
        _entropy_interval(interval)[1] for interval in option_intervals
    ], float)
    codebook_upper_ucb = float(np.max(option_entropy_ucb))

    target_interval = _cp_interval(int(y.sum()), n, alpha_lower / 2.0)
    target_entropy_lcb = _entropy_interval(target_interval)[0]
    mismatches = int(np.sum(y != recovered))
    mismatch_interval = _cp_interval(mismatches, n, alpha_lower / 2.0)
    mismatch_lo, mismatch_hi = mismatch_interval
    if mismatch_lo <= 0.5 <= mismatch_hi:
        polarity_error_ucb = 0.5
    elif mismatch_hi < 0.5:
        polarity_error_ucb = mismatch_hi
    else:
        polarity_error_ucb = 1.0 - mismatch_lo
    conditional_entropy_ucb = (1.0 if polarity_error_ucb >= 0.5
                               else _h_binary(polarity_error_ucb))
    recovery_lcb = float(max(0.0, target_entropy_lcb - conditional_entropy_ucb))
    gap_ucb = float(max(0.0, codebook_upper_ucb - recovery_lcb))

    best_option = int(np.argmax(option_entropy_ucb))
    return {
        "schema": "reconstruction-mcq-codebook-v1",
        "estimand": {
            "name": "auxiliary executor-specific MCQ behavioral-replay prompt optimality",
            "definition": "R_{b,E}(p)=I(M_p; Mhat_p) after MCQ choice and canonical re-execution",
            "candidate_prompt_class": "all finite prompts Sigma*; no prompt-length budget",
            "reconstruction_output_class": "the frozen canonical MCQ codebook C_b",
            "polarity": "invariant",
            "input_contract": "prompt self-annotations only; no external labels",
        },
        "empirical": {
            "recovery_bits": recovery_hat,
            "target_entropy_bits": target_entropy_hat,
            "codebook_transmission_upper_bits": codebook_upper_hat,
            "codebook_option_entropy_bits": option_entropy_hat.tolist(),
            "raw_mismatch_rate": mismatches / n,
            "polarity_invariant_mismatch_rate": min(mismatches / n, 1.0 - mismatches / n),
            "n_lockbox_items": n,
            "n_options": int(len(options)),
            "n_reconstruction_draws": int(len(choices)),
        },
        "certified": {
            "simultaneous_confidence": float(1.0 - alpha),
            "codebook_prompt_optimum_UCB_bits": codebook_upper_ucb,
            "candidate_recovery_LCB_bits": recovery_lcb,
            "global_optimization_gap_UCB_bits": gap_ucb,
            "target_epsilon_bits": float(epsilon_bits),
            "status": ("PROVABLY_EPSILON_OPTIMAL"
                       if gap_ucb <= epsilon_bits else
                       "GLOBAL_GAP_EXCEEDS_TARGET_EPSILON"),
            "best_upper_option_id": str(option_ids[best_option]),
            "per_option": {
                str(option_id): {
                    "prevalence_interval": list(interval),
                    "transmission_UCB_bits": float(entropy_u),
                }
                for option_id, interval, entropy_u in zip(
                    option_ids, option_intervals, option_entropy_ucb)
            },
            "target_prevalence_interval": list(target_interval),
            "raw_mismatch_probability_interval": list(mismatch_interval),
            "polarity_invariant_error_UCB": float(polarity_error_ucb),
            "conditional_entropy_UCB_bits": float(conditional_entropy_ucb),
            "alpha_allocation": {
                "total_alpha": float(alpha),
                "codebook_upper": float(alpha_upper),
                "per_option_upper_component": float(option_component_alpha),
                "candidate_lower": float(alpha_lower),
                "candidate_target_prevalence": float(alpha_lower / 2.0),
                "candidate_mismatch": float(alpha_lower / 2.0),
            },
        },
        "proof": {
            "chain": (
                "I(M_p;Mhat_p) <= I(X;Mhat_p) <= "
                "max_{c in C_b} I(X;E(c,X))"
            ),
            "why": [
                "MCQ choice is made from reconstruction-training annotations and is independent of lockbox X",
                "Mhat_p is a convex mixture of frozen canonical option channels",
                "mutual information is convex in the channel for fixed item distribution",
                "binary option transmission is at most its verdict entropy",
            ],
        },
        "scope": {
            **dict(scope or {}),
            "production_cr3_role": "secondary behavioral replay; not the primary MCQ identification value",
            "option_ids": [str(x) for x in option_ids],
            "premises": premises,
            "required_inputs": [
                "candidate prompt's own lockbox annotations",
                "canonical codebook option annotations",
                "MCQ reconstruction choices made from training annotations",
                "predeclared iid-uniform draw assignment",
            ],
            "explicitly_not_required": [
                "silver labels",
                "ground-truth labels",
                "human annotations",
                "archival outcomes",
            ],
            "does_not_certify": [
                "substantive correctness or external validity",
                "a free-form reconstructor whose outputs are outside C_b",
                "a different codebook, executor, item distribution, or readout",
            ],
        },
    }


def reconstruction_global_certificate(
    prompt_verdicts: np.ndarray,
    reconstructed_verdicts: np.ndarray,
    *,
    prompt_frozen_before_lockbox: bool,
    reconstructions_frozen_before_lockbox: bool,
    lockbox_unused_for_optimization: bool,
    alpha: float = 0.05,
    epsilon_shannon_bits: float = 0.05,
    epsilon_tvd: float = 0.025,
    scope: Mapping[str, object] | None = None,
) -> dict:
    """Certify one prompt against global binary reconstruction-channel caps.

    ``prompt_verdicts`` is the candidate prompt's own hard verdict on each fresh
    lockbox item.  ``reconstructed_verdicts`` is ``(n_items, n_reconstructions)``;
    each column is one prompt reconstructed without lockbox access and re-executed by
    the declared executor.  A uniformly selected frozen reconstruction column defines
    the recovered channel.  Confidence is over iid lockbox items, conditional on that
    finite reconstruction ensemble.

    The function fails closed if prompt selection, reconstruction, or optimization
    touched the lockbox.  It has deliberately no parameter for external labels.
    """
    if not prompt_frozen_before_lockbox:
        raise ValueError("candidate prompt must be frozen before lockbox access")
    if not reconstructions_frozen_before_lockbox:
        raise ValueError("reconstructed prompts must be frozen before lockbox access")
    if not lockbox_unused_for_optimization:
        raise ValueError("lockbox must be unused for optimization, selection, and stopping")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if not np.isfinite(epsilon_shannon_bits) or not 0.0 <= epsilon_shannon_bits <= 1.0:
        raise ValueError("epsilon_shannon_bits must lie in [0, 1]")
    if not np.isfinite(epsilon_tvd) or not 0.0 <= epsilon_tvd <= 0.5:
        raise ValueError("epsilon_tvd must lie in [0, 0.5]")

    y_raw = np.asarray(prompt_verdicts, float)
    recovered = np.asarray(reconstructed_verdicts, float)
    if y_raw.ndim != 1 or recovered.ndim != 2 or recovered.shape[0] != len(y_raw):
        raise ValueError("expected prompt_verdicts (n,) and reconstructed_verdicts (n, R)")
    if len(y_raw) < 2 or recovered.shape[1] < 1:
        raise ValueError("at least two lockbox items and one reconstruction are required")
    if np.any(~np.isfinite(y_raw)) or np.any(~np.isfinite(recovered)):
        raise ValueError("bound-grade lockbox arrays must be complete and finite")
    if np.any((y_raw != 0.0) & (y_raw != 1.0)):
        raise ValueError("prompt_verdicts must be hard binary values")
    if np.any((recovered < 0.0) | (recovered > 1.0)):
        raise ValueError("reconstructed_verdicts must lie in [0, 1]")

    y = y_raw.astype(np.uint8)
    q = recovered.mean(axis=1)
    n = len(y)
    joint = _joint_from_mixture(y, q)
    shannon_hat = _shannon_mi(joint)
    prod = np.outer(joint.sum(axis=1), joint.sum(axis=0))
    tvd_hat = float(0.5 * np.abs(joint - prod).sum())
    target_entropy_hat = _h_binary(float(y.mean()))
    conditional_entropy_hat = float(max(0.0, target_entropy_hat - shannon_hat))

    # Two primary lower-bound claims (Shannon and TVD) share alpha.  The Shannon
    # claim further shares its budget between target prevalence and the bounded
    # mismatch loss.  This makes both reported global-gap UCBs simultaneous.
    alpha_measure = alpha / 2.0
    alpha_prevalence = alpha_measure / 2.0
    alpha_mismatch = alpha_measure / 2.0
    prevalence_interval = _cp_interval(int(y.sum()), n, alpha_prevalence)
    target_entropy_interval = _entropy_interval(prevalence_interval)

    mismatch_loss = y * (1.0 - q) + (1.0 - y) * q
    mismatch_hat = float(np.mean(mismatch_loss))
    mismatch_radius = float(np.sqrt(np.log(2.0 / alpha_mismatch) / (2.0 * n)))
    e_pm_u, mismatch_interval = _polarity_invariant_error_upper(
        mismatch_hat, mismatch_radius)
    conditional_entropy_u = 1.0 if e_pm_u >= 0.5 else _h_binary(e_pm_u)
    shannon_lcb = float(max(0.0, target_entropy_interval[0] - conditional_entropy_u))
    shannon_gap_ucb = float(1.0 - shannon_lcb)

    # TVD MI for binary variables is 2*|Cov(M_p, Mhat_p)|.  Simultaneous
    # Hoeffding intervals for E[M*q], E[M], and E[q] give an interval for the
    # covariance and hence a rigorous polarity-invariant lower bound.
    tvd_radius = float(np.sqrt(np.log(6.0 / alpha_measure) / (2.0 * n)))
    a_hat = float(np.mean(y * q))
    b_hat = float(np.mean(y))
    c_hat = float(np.mean(q))
    a_lo, a_hi = max(0.0, a_hat - tvd_radius), min(1.0, a_hat + tvd_radius)
    b_lo, b_hi = max(0.0, b_hat - tvd_radius), min(1.0, b_hat + tvd_radius)
    c_lo, c_hi = max(0.0, c_hat - tvd_radius), min(1.0, c_hat + tvd_radius)
    cov_lo = a_lo - b_hi * c_hi
    cov_hi = a_hi - b_lo * c_lo
    if cov_lo <= 0.0 <= cov_hi:
        abs_cov_lcb = 0.0
    else:
        abs_cov_lcb = min(abs(cov_lo), abs(cov_hi))
    tvd_lcb = float(np.clip(2.0 * abs_cov_lcb, 0.0, 0.5))
    tvd_gap_ucb = float(0.5 - tvd_lcb)

    return {
        "schema": SCHEMA_VERSION,
        "estimand": {
            "name": "auxiliary executor-specific reconstruction-only prompt optimality",
            "shannon": "R_E(p) = I(M_p; Mhat_p)",
            "tvd": "R^TVD_E(p) = D_TV(P[M_p,Mhat_p], P[M_p]P[Mhat_p])",
            "optimization_class": "all finite prompts Sigma*; no prompt-length budget",
            "target_source": "the candidate prompt's own annotations; no external labels",
            "polarity": "invariant",
            "conditioning": "the frozen finite reconstruction ensemble",
        },
        "empirical": {
            "shannon_recovery_bits": shannon_hat,
            "tvd_recovery": tvd_hat,
            "target_entropy_bits": target_entropy_hat,
            "conditional_entropy_bits": conditional_entropy_hat,
            "raw_mismatch_probability": mismatch_hat,
            "polarity_invariant_mismatch_probability": min(mismatch_hat, 1.0 - mismatch_hat),
            "n_lockbox_items": n,
            "n_frozen_reconstructions": int(recovered.shape[1]),
        },
        "certified": {
            "simultaneous_confidence": float(1.0 - alpha),
            "shannon": {
                "global_upper_bound_bits": 1.0,
                "candidate_recovery_LCB_bits": shannon_lcb,
                "global_optimization_gap_UCB_bits": shannon_gap_ucb,
                "target_epsilon_bits": float(epsilon_shannon_bits),
                "status": ("PROVABLY_EPSILON_OPTIMAL"
                           if shannon_gap_ucb <= epsilon_shannon_bits else
                           "GLOBAL_GAP_EXCEEDS_TARGET_EPSILON"),
                "target_prevalence_interval": list(prevalence_interval),
                "target_entropy_interval_bits": list(target_entropy_interval),
                "raw_mismatch_probability_interval": list(mismatch_interval),
                "polarity_invariant_error_UCB": e_pm_u,
                "conditional_entropy_UCB_bits": conditional_entropy_u,
                "gap_decomposition": (
                    "1-R = (1-H(M_p)) + H(M_p|Mhat_p); both terms are reconstruction-only"
                ),
            },
            "tvd": {
                "global_upper_bound": 0.5,
                "candidate_recovery_LCB": tvd_lcb,
                "global_optimization_gap_UCB": tvd_gap_ucb,
                "target_epsilon": float(epsilon_tvd),
                "status": ("PROVABLY_EPSILON_OPTIMAL"
                           if tvd_gap_ucb <= epsilon_tvd else
                           "GLOBAL_GAP_EXCEEDS_TARGET_EPSILON"),
                "covariance_interval": [cov_lo, cov_hi],
            },
            "alpha_allocation": {
                "total_alpha": float(alpha),
                "per_measure": float(alpha_measure),
                "shannon_prevalence": float(alpha_prevalence),
                "shannon_mismatch": float(alpha_mismatch),
                "tvd_three_mean_bundle": float(alpha_measure),
            },
        },
        "scope": {
            **dict(scope or {}),
            "required_inputs": [
                "candidate prompt's own hard lockbox verdicts",
                "frozen reconstructed prompts' lockbox verdicts",
            ],
            "explicitly_not_required": [
                "silver labels",
                "ground-truth labels",
                "human annotations",
                "archival outcomes",
                "a second metric oracle",
            ],
            "assumptions": [
                "iid lockbox items from the claimed item distribution",
                "candidate prompt frozen before lockbox access",
                "reconstructed prompts induced without lockbox access and then frozen",
                "lockbox unused for prompt optimization, selection, stopping, or revision",
                "the finite reconstruction ensemble is the declared reconstruction estimator",
            ],
            "does_not_certify": [
                "substantive correctness of the reconstructed behavior",
                "alignment with any unavailable external construct",
                "generalization over a different reconstructor distribution",
            ],
        },
    }
