"""cleanliness_gate — the channel-cleanliness certificate (validity layer; theory §6.10 / scorecard V1–V3).

PRUNE-help is a LEAKAGE ALARM, not a structural claim (§6.10): if removing criterion ``e`` from Ω RAISES
R(C(Ω)), then ``e`` was carrying confounded/spurious signal — a side-channel that helped alone (strong
marginal) but interfered in combination (double-counting/contradiction). A criterion STAYS in Ω iff it
passes the screen; cleaning removes the spurious (leakage-driven) non-monotonicity, so Ω becomes
near-monotone and the cheap γ guarantees (§6.2/§6.6) are RE-ENABLED AS A BONUS of validity — never assumed
or proven standalone (the §6.10 convergence).

This module's ``prune_help_from_rdict`` is the GPU-FREE core: it operates on the certificate's per-subset
R-dict (exact mode, all 2^K−1 subsets scored), so it needs no rescoring. The full gate adds two legs that
need E's scored signals (wired in the GPU sweep):
  (a) ``adversarial_saturation``: I(X_probe; M | X_Ω) ≈ 0 for side-channel probes (length/format/OOD) —
      the conditional-independence test that Ω BLOCKS the channel (assumption-free ignorability, §6.7c leg 2);
  (b) ``counterfactual_validity`` (measures.py): ``e`` tracks a PLANTED direction, not a confound.

CLI (zero-GPU, on a saved omega_certificate npz):
  python -m methods.metric_implementer.experiments.cleanliness_gate <npz>
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Sequence

import numpy as np


def _rdict_from_arrays(subset_keys: Sequence[str], subset_R: Sequence[float]) -> Dict[tuple, float]:
    """Reconstruct {sorted-criterion-tuple -> R} from the npz's subset_keys ('1,2,3') + subset_R."""
    out = {}
    for k, r in zip(subset_keys, subset_R):
        try:
            key = tuple(sorted(int(x) for x in str(k).split(",") if x.strip() != ""))
        except ValueError:
            continue
        out[key] = float(r)
    return out


def prune_help_from_rdict(subset_keys: Sequence[str], subset_R: Sequence[float], K: int,
                          *, eps: float = 1e-9) -> dict:
    """Per-criterion PRUNE-help from the exact R-dict (the GPU-free leakage alarm).

    For each criterion ``e`` in 0..K-1: ``prune_help[e] = R(Ω∖{e}) − R(Ω)``.
      * ``> 0``  ⟹  removing ``e`` RAISES recovery ⟹ ``e`` is a LEAKAGE SUSPECT (drop it);
      * ``≤ 0``  ⟹  ``e`` contributes (or is neutral) ⟹ keep.
    ``marginal[e] = R(Ω) − R(Ω∖{e}) = −prune_help`` is the standard marginal contribution (``< 0`` ⟹ the
    non-monotonicity/PRUNE of §6.1). Returns the cleaned Ω (suspects dropped) and whether Ω is already
    near-monotone (no suspects) — the validity statement that the bound is not propped up by a spurious
    correlate.
    """
    R = _rdict_from_arrays(subset_keys, subset_R)
    full = tuple(range(K))
    if full not in R:
        return {"error": "full-set Ω not scored in R-dict (exact-mode certificate required)",
                "K": K, "has_full": False}
    r_full = R[full]
    per: Dict[int, dict] = {}
    for e in range(K):
        minus = tuple(sorted(i for i in range(K) if i != e))
        r_minus = R.get(minus)
        ph = (float(r_minus) - r_full) if r_minus is not None else float("nan")
        per[e] = {"prune_help": ph, "marginal": -ph if np.isfinite(ph) else float("nan"),
                  "R_without_e": (float(r_minus) if r_minus is not None else float("nan")),
                  "R_full": r_full, "leakage_suspect": bool(np.isfinite(ph) and ph > eps)}
    suspects = [e for e in range(K) if per[e]["leakage_suspect"]]
    cleaned = sorted(e for e in range(K) if e not in suspects)
    return {"K": K, "R_full": r_full, "per_criterion": per,
            "leakage_suspects": suspects, "n_suspects": len(suspects),
            "cleaned_omega": cleaned, "near_monotone": len(suspects) == 0,
            "max_prune_help": float(max((per[e]["prune_help"] for e in range(K)),
                                        key=lambda v: (v if np.isfinite(v) else -np.inf)))}


def report(res: dict, crits: Sequence[str] | None = None) -> str:
    """Human-readable cleanliness readout (mirrors the certificate's print style)."""
    if "error" in res:
        return f"[cleanliness] {res['error']}"
    lines = [f"[cleanliness] R(Ω)={res['R_full']:.4f} over K={res['K']} criteria"]
    for e in range(res["K"]):
        p = res["per_criterion"][e]
        flag = "  ⚠ LEAKAGE SUSPECT (drop)" if p["leakage_suspect"] else ""
        nm = (crits[e][:52] + "…") if crits and e < len(crits) and len(crits[e]) > 53 else \
             (crits[e][:53] if crits and e < len(crits) else "")
        lines.append(f"  e={e}  prune_help={p['prune_help']:+.4f}  marginal={p['marginal']:+.4f}"
                     f"  R(Ω∖e)={p['R_without_e']:.4f}{flag}  {nm}")
    if res["near_monotone"]:
        lines.append(f"  → Ω is NEAR-MONOTONE (no PRUNE-help): validity holds, bound not propped up by a "
                     f"spurious correlate; cheap γ guarantees legitimate.")
    else:
        lines.append(f"  → {res['n_suspects']} leakage suspect(s) {res['leakage_suspects']}: drop them → "
                     f"cleaned Ω={res['cleaned_omega']}; re-certify on the cleaned set (removes spurious "
                     f"non-monotonicity, restores near-submodularity, §6.10).")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cleanliness_gate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", help="a saved omega_certificate npz (subset_keys/subset_R/crits/K)")
    a = ap.parse_args(argv)
    d = np.load(a.npz, allow_pickle=True)
    K = int(d["K"]) if "K" in d else len(set(int(x) for k in d["subset_keys"] for x in str(k).split(",") if x))
    crits = list(d["crits"]) if "crits" in d else None
    res = prune_help_from_rdict(d["subset_keys"], d["subset_R"], K)
    print(report(res, crits))


if __name__ == "__main__":
    main()
