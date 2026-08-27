"""Metric-count certificates — MCC theory (notes/2026-07-01__metric-count-certificates.md §3-4)
computed from run artifacts.

Inputs are deliberately minimal: a bits trajectory (V_bits after each accepted metric), the
per-proposal ledger, and optional wrap ingredients (dense ceiling + its dominance evidence).
Outputs are the value bracket and the two count certificates, each tagged with its assumptions
so a report can never silently promote a conditional bound to an unconditional one.

The certificates:
  N_lower  = ceil(V_bits / log2 K)          [assumption-free: DPI + entropy counting, MCC §4.1]
             (optionally V_bits / T_max under per-metric-call scoring — caller-supplied T_max)
  N_upper  = |S_kept| + (U - V_bits) / delta_bits    [conditional on the wrap U, MCC §4.2a]
  Minoux stopping read                      [monotone V: within-pool epsilon-optimality, §4.2b]
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


# --------------------------------------------------------------------------------------

@dataclass
class ValueBracket:
    """floor <= A*_E <= wrap (or right-censored)."""

    floor_bits: float                       # achieved V(S_g) on guard/test, bits/item
    wrap_bits: Optional[float]              # U (dense-stack) if dominance gate passed, else None
    wrap_source: str = "none"               # "dense_stack" | "flux" | "none"
    right_censored: bool = True             # True when no valid wrap exists (the CW case)
    dominance_evidence: dict = field(default_factory=dict)


@dataclass
class CountCertificates:
    n_kept: int
    # necessity (lower bound on lexicon size) — assumption-free leg
    n_lower_logK: int                       # ceil(V_bits / log2 K)
    n_lower_Tmax: Optional[int]             # ceil(V_bits / T_max) if T_max supplied (see MCC §4.1
    #                                         implementation flag: needs per-metric-call scoring)
    # sufficiency / collectibility (upper bound on further delta-useful metrics)
    delta_bits: float
    n_upper: Optional[float]                # |S_kept| + (U - V)/delta ; None when right-censored
    # Minoux stopping read (within-pool epsilon-optimality of the greedy chain)
    minoux_tail_bits: float                 # sum of top-k residual marginal gains (rejected pool)
    minoux_epsilon: Optional[float]         # tail / gamma_hat if gamma supplied
    assumptions: List[str] = field(default_factory=list)


@dataclass
class CertificateReport:
    task: str
    executor: str
    K: int
    bracket: ValueBracket
    counts: CountCertificates
    notes: List[str] = field(default_factory=list)

    def save(self, out_path: str | Path) -> None:
        p = Path(out_path); p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump({"task": self.task, "executor": self.executor, "K": self.K,
                       "bracket": asdict(self.bracket), "counts": asdict(self.counts),
                       "notes": self.notes}, f, indent=2)

    def render(self) -> str:
        b, c = self.bracket, self.counts
        wrap = (f"{b.wrap_bits:.4f} bits [{b.wrap_source}]" if b.wrap_bits is not None
                else "RIGHT-CENSORED (no valid wrap)")
        nup = f"{c.n_upper:.1f}" if c.n_upper is not None else "— (right-censored)"
        lines = [
            f"== Metric-count certificate — {self.task} / {self.executor} (K={self.K}) ==",
            f"value bracket:  {b.floor_bits:.4f} bits  <=  A*_E  <=  {wrap}",
            f"N_lower (free): {c.n_lower_logK}"
            + (f"   N_lower (T_max): {c.n_lower_Tmax}" if c.n_lower_Tmax is not None else ""),
            f"N_upper (delta={c.delta_bits} bits): {nup}   (|S_kept|={c.n_kept})",
            f"Minoux tail: {c.minoux_tail_bits:.4f} bits"
            + (f" -> epsilon={c.minoux_epsilon:.4f}" if c.minoux_epsilon is not None else ""),
            "assumptions: " + ("; ".join(c.assumptions) or "(none beyond plug-in estimation)"),
        ]
        lines += [f"note: {n}" for n in self.notes]
        return "\n".join(lines)


# --------------------------------------------------------------------------------------

def dense_stack_wrap(
    dense_bits: float, bank_bits: float, stack_bits: Optional[float] = None,
    dense_plateaued: bool = False, margin_bits: float = 0.02,
) -> ValueBracket:
    """MCC §3 Wrap 1. U_dense is a valid upper wrap on the articulable ceiling ONLY under the
    dominance gate: the dense scaling curve has plateaued AND the bank sits below dense by a
    margin. Otherwise the ceiling is right-censored (the honest CW read).

    ``stack_bits`` (dense model given the bank outputs as extra features) supersedes
    ``dense_bits`` when provided — the stack makes the dominance check partially internal.
    """
    u = stack_bits if stack_bits is not None else dense_bits
    ev = {"dense_bits": dense_bits, "bank_bits": bank_bits, "stack_bits": stack_bits,
          "dense_plateaued": dense_plateaued, "margin_bits": margin_bits}
    ok = dense_plateaued and (bank_bits <= u - margin_bits)
    if ok:
        return ValueBracket(floor_bits=bank_bits, wrap_bits=float(u),
                            wrap_source="dense_stack", right_censored=False,
                            dominance_evidence=ev)
    return ValueBracket(floor_bits=bank_bits, wrap_bits=None, wrap_source="none",
                        right_censored=True, dominance_evidence=ev)


def flux_wrap(
    bank_bits: float, flux_tail_bits: float, gamma_hat: float = 1.0,
) -> ValueBracket:
    """MCC §3 Wrap 2 (process-relative): A* <= V(S_g) + (1/gamma) * flux_tail. Caller supplies the
    Good-Toulmin horizon estimate ``flux_tail_bits`` (D1-D3 on the supervised value spectrum);
    this function only assembles the bracket and refuses nonsense inputs."""
    if not (flux_tail_bits >= 0 and 0 < gamma_hat <= 1):
        return ValueBracket(floor_bits=bank_bits, wrap_bits=None, wrap_source="none",
                            right_censored=True)
    return ValueBracket(floor_bits=bank_bits, wrap_bits=float(bank_bits + flux_tail_bits / gamma_hat),
                        wrap_source="flux", right_censored=False,
                        dominance_evidence={"flux_tail_bits": flux_tail_bits, "gamma_hat": gamma_hat})


def count_certificates(
    bracket: ValueBracket, n_kept: int, delta_bits: float, K: int = 2,
    rejected_gains_bits: Optional[List[float]] = None, budget_k: Optional[int] = None,
    gamma_hat: Optional[float] = None, T_max_bits: Optional[float] = None,
    per_metric_call_scoring: bool = False,
) -> CountCertificates:
    """The two count certificates + the Minoux stopping read, from a finished run.

    - ``rejected_gains_bits``: measured marginal V_bits gains of proposals NOT kept (the delta(e)
      of the Minoux bound); top-``budget_k`` of them form the within-pool optimality tail.
    - ``T_max_bits`` tightens N_lower ONLY when ``per_metric_call_scoring`` (the §4.1 conditional-
      independence requirement); silently ignored otherwise.
    """
    V = max(0.0, bracket.floor_bits)
    assumptions = ["plug-in MI (finite-sample); use CIs on V_bits"]

    n_lower_logK = math.ceil(V / math.log2(K)) if V > 0 else 0
    n_lower_Tmax = None
    if T_max_bits and T_max_bits > 0:
        if per_metric_call_scoring:
            n_lower_Tmax = math.ceil(V / T_max_bits)
            assumptions.append("T_max leg: per-metric-call scoring (cond. indep. given X)")
        else:
            assumptions.append("T_max supplied but IGNORED: shared-call scoring breaks the "
                               "conditional-independence step (MCC §4.1)")

    n_upper = None
    if bracket.wrap_bits is not None and delta_bits > 0:
        n_upper = n_kept + max(0.0, bracket.wrap_bits - V) / delta_bits
        assumptions.append(f"N_upper: wrap={bracket.wrap_source} + quotient distinctness + "
                           "per-metric prompt optimality")

    tail = sorted((g for g in (rejected_gains_bits or []) if g > 0), reverse=True)
    k = budget_k if budget_k is not None else len(tail)
    minoux_tail = float(sum(tail[:k]))
    minoux_eps = minoux_tail / gamma_hat if (gamma_hat and gamma_hat > 0) else None
    if minoux_eps is not None:
        assumptions.append(f"Minoux epsilon: measured gamma_hat={gamma_hat} (weak submodularity)")

    return CountCertificates(n_kept=n_kept, n_lower_logK=n_lower_logK, n_lower_Tmax=n_lower_Tmax,
                             delta_bits=delta_bits, n_upper=n_upper,
                             minoux_tail_bits=minoux_tail, minoux_epsilon=minoux_eps,
                             assumptions=assumptions)


def report_from_ledger(
    ledger_path: str | Path, task: str, executor: str, delta_bits: float, K: int = 2,
    dense_bits: Optional[float] = None, stack_bits: Optional[float] = None,
    dense_plateaued: bool = False, gamma_hat: Optional[float] = None,
) -> CertificateReport:
    """Assemble the full certificate report from a saved global_infill_ledger.json."""
    return report_from_ledgers([ledger_path], task=task, executor=executor,
                               delta_bits=delta_bits, K=K, dense_bits=dense_bits,
                               stack_bits=stack_bits, dense_plateaued=dense_plateaued,
                               gamma_hat=gamma_hat)


def report_from_ledgers(
    ledger_paths: List[str | Path], task: str, executor: str, delta_bits: float, K: int = 2,
    dense_bits: Optional[float] = None, stack_bits: Optional[float] = None,
    dense_plateaued: bool = False, gamma_hat: Optional[float] = None,
    flux_tail_bits: Optional[float] = None, flux_gamma: float = 1.0,
    flux_meta: Optional[dict] = None,
) -> CertificateReport:
    """UNION certificate over multiple generator-arm ledgers (MCC §2a: the certified artifact
    is the union ledger; per-arm reports always carry the single-arm honesty note).

    The arms run independently from the same starting bank, so the honest union floor is the
    MAX of the per-arm final V_bits, not their sum — a joint refit stacking every kept metric
    could exceed it, and a note says so whenever >1 arm kept metrics.

    ``flux_tail_bits``/``flux_meta`` come from :func:`flux.flux_from_ledgers`; when both the
    dense and flux wraps are valid the bracket takes the TIGHTER one, and the flux wrap always
    carries its process-relative scope note (it bounds what THESE arms articulate by horizon
    (1+c)N, not the unconditional ceiling)."""
    all_ledgers, floors, kept_per_arm = [], [], {}
    for p in ledger_paths:
        with open(p) as f:
            data = json.load(f)
        bits_traj = data.get("guard_bits_trajectory") or []
        if not bits_traj:
            raise ValueError(f"{p}: no guard_bits_trajectory — rerun with the bits readout")
        floors.append(float(bits_traj[-1]))
        ledgers = data.get("ledgers", [])
        all_ledgers.extend(ledgers)
        for l in ledgers:
            if l.get("status") == "kept":
                kept_per_arm.setdefault(l.get("generator", "residual"), []).append(l)
    floor = max(floors)
    kept = [l for l in all_ledgers if l.get("status") == "kept"]
    rejected_gains = [l.get("bits_gain") for l in all_ledgers
                      if l.get("status") != "kept" and isinstance(l.get("bits_gain"), (int, float))]

    if dense_bits is not None:
        bracket = dense_stack_wrap(dense_bits, floor, stack_bits, dense_plateaued)
    else:
        bracket = ValueBracket(floor_bits=floor, wrap_bits=None, right_censored=True)

    notes = ["floor from the guard/CV trajectory (run-stage artifact) — the untouched-test "
             "read (MCC §4.5 step 5) supersedes it for any published bracket"]
    if flux_tail_bits is not None:
        fb = flux_wrap(floor, float(flux_tail_bits), flux_gamma)
        if not fb.right_censored:
            notes.append(
                f"U_flux = {fb.wrap_bits:.4f} bits (tail {flux_tail_bits:.4f}/gamma {flux_gamma})"
                " — PROCESS-RELATIVE: bounds what THESE generator arms articulate by horizon "
                "(1+c)N, not the unconditional ceiling (MCC §3 Wrap 2)")
            if bracket.right_censored or (bracket.wrap_bits is not None
                                          and fb.wrap_bits < bracket.wrap_bits):
                fb.dominance_evidence.update({"superseded_dense": bracket.wrap_bits})
                bracket = fb
    if flux_meta:
        if flux_meta.get("singleton_regime"):
            notes.append(f"flux read in the SINGLETON regime (f1/species = "
                         f"{flux_meta.get('f1_over_species', float('nan')):.2f}) — spectrum "
                         "still growing; horizon mass dominated by w_1 (degeneracy lemma)")
        ns, na = flux_meta.get("n_species"), flux_meta.get("n_species_audit_tau")
        if ns and na and abs(ns - na) / max(ns, na) > 0.25:
            notes.append(f"merge-precision audit DIVERGES ({ns} species @tau="
                         f"{flux_meta.get('tau')} vs {na} @tau={flux_meta.get('audit_tau')}) — "
                         "quotient fragile; treat N_upper as indicative")
        if flux_meta.get("n_arms", 0) < 2:
            notes.append("flux over a SINGLE arm — anti-conservative (MCC §9.1)")

    counts = count_certificates(bracket, n_kept=len(kept), delta_bits=delta_bits, K=K,
                                rejected_gains_bits=rejected_gains, gamma_hat=gamma_hat)
    if bracket.right_censored:
        notes.append("articulable ceiling RIGHT-CENSORED: no wrap passed its gate "
                     "(dense not plateaued / no dominance evidence) — N_upper unavailable")
    arms = {l.get("generator", "residual") for l in all_ledgers}
    if len(arms) < 2:
        notes.append(f"single generator arm ({arms or '{}'}): flux/coverage reads are "
                     "anti-conservative — add a second proposal family (MCC §9.1)")
    else:
        notes.append(f"union certificate over {len(arms)} generator arms "
                     f"({sorted(arms)}) — MCC §2a satisfied")
    if len(ledger_paths) > 1 and len(kept_per_arm) > 1:
        notes.append("floor = max over per-arm banks (arms ran independently); a joint refit "
                     "of all kept metrics could exceed it")
    kept_no_confirm = [l for l in kept if not l.get("confirm_m")]
    if kept and kept_no_confirm:
        notes.append(f"{len(kept_no_confirm)}/{len(kept)} kept metric(s) NOT confirm-staged "
                     "(no fresh-seed replication / Bonferroni) — winner's-curse exposure; "
                     "the 2026-07-02 runs showed unconfirmed CV survivors die on test (MCC §4.4)")
    elif kept:
        ms = sorted({int(l.get("confirm_m", 0)) for l in kept})
        notes.append(f"all kept metrics confirm-staged (fresh-seed repeated CV, NB-corrected, "
                     f"Bonferroni m={ms})")
    return CertificateReport(task=task, executor=executor, K=K, bracket=bracket,
                             counts=counts, notes=notes)
