"""§12.3 VALUE-CENSUS driver — PER METRIC, anchor-free (M_i, never the aggregate Y). Theory §12.3.

This driver is descriptive only. Its α_V-Heaps readout DEGENERATES when the spectrum is
singleton-dominated (f₁/N→1 ⇒ α_V ≡ α ≡ 1 mechanically — Lemma 12.6.0); read it only alongside
``f1_over_N``, and treat the §12.3-E verdicts here as the descriptive census, not the depth verdict.

Loops over a directory of per-metric checkpoints (``…_metric{idx}_sigs.npz`` written by the metric
α-probe, each storing the criterion signatures AND M_i — the metric's OWN verdict). For each metric it
runs BOTH the behavior census (``alpha_probe``, CMI skipped — we only need α_i) and the value census
(``value_census``) on the IDENTICAL signatures at the IDENTICAL τ, so α_i and α_{V,i} read the SAME
species partition (GV4), and prints the headline trio — α_i (behavior), α_{V,i} (value), the breadth gap
α_i − α_{V,i} — plus singleton flux, additive/conditional-greedy diagnostics, the conditional-gain sum
divided by H(M_i), and top standalone-value criteria. The gain ratio is not a recovered-information fraction.

CPU-ONLY, anchor-free: no GPU, no executor, no new draws, **no Y anywhere** — M_i comes from the
checkpoint (computed once during the behavior run via ``metric_verdict``).

No saturation decision is issued here. The species/value map is estimated from the
same sample, mutual information need not be submodular, and Hill is descriptive.

Example (CPU, after a metric α-probe run):
    python -m methods.metric_implementer.experiments.run_value_census --ckpt-dir /…/alpha_probe_metric
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from .. import vinfo
from . import alpha_probe as ap
from . import value_census as vc


def _value_decision(alpha_i: float, alpha_V: float, gap: float, mv0: float, hill: float, *,
                    eps_alpha_v: float = 0.3, eps_gap_track: float = 0.10, eps_mv: float = 0.01) -> dict:
    """Retained API slot; deliberately refuses to promote diagnostics to a bound."""
    return {"verdict": "DESCRIPTIVE",
            "note": "alpha_V, singleton flux, and Hill describe this realized sample only; "
                    "they do not certify saturation or a prompt ceiling."}


def _run_one(ckpt: str, a) -> dict | None:
    z = np.load(ckpt, allow_pickle=True)
    sigs = np.asarray(z["sigs"], float)
    tags = list(z["tags"])
    prompts = list(z["prompts"]) if "prompts" in z else None
    tau = float(a.tau) if a.tau is not None else float(z["tau"])
    name = str(z["name"]) if "name" in z else os.path.basename(ckpt)
    if "M_i" not in z:
        print(f"  SKIP {name[:50]!r}: checkpoint has no M_i (re-run the metric α-probe to embed it)")
        return None
    M_i = (np.asarray(z["M_i"], float) > 0.5).astype(int)          # the metric's OWN verdict (anchor-free)

    beh = ap.alpha_probe(sigs, tags, tau=tau, delta=a.delta, compute_cmi=False)
    alpha_i = float(beh["alpha_terminal"])
    rep = vc.value_census(sigs, M_i, tau=tau, delta=a.delta, submodular_top_k=a.submodular_top_k,
                          prompts=prompts)
    alpha_V = rep["alpha_V_terminal"]
    gap = alpha_i - alpha_V
    h_mi = float(vinfo._h_bits(float(np.mean(M_i))))                # M_i entropy — the recovery ceiling
    gain_ratio = float(rep["submodular"]["R_full"] / h_mi) if h_mi > 1e-9 else 0.0
    dec = _value_decision(alpha_i, alpha_V, gap, rep["MV0"], rep["hill_tail_index"])
    if gain_ratio < 0.05 or h_mi < 0.05:
        # Degenerate M_i (near-constant or near-zero conditional-gain sum): the alpha_V read is an artifact of a
        # flat target, NOT "a few criteria suffice." Relabel so it isn't mistaken for SATURATED.
        dec = {"verdict": "DEGENERATE",
               "note": "M_i near-constant (H(M_i)≈0) or near-zero conditional-gain sum; the "
                       "α_V≈0 / MV0≈0 read is a flat-target artifact, not saturation."}
    rep.update({"name": name, "ckpt": ckpt, "alpha_i": alpha_i, "alpha_V_terminal": alpha_V,
                "breadth_gap": float(gap), "H_M_i": h_mi,
                "conditional_gain_sum_over_H": gain_ratio, "decision": dec})
    return rep


def main(argv=None):
    ap_ = argparse.ArgumentParser(prog="run_value_census", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--ckpt-dir", default=None, help="dir of …_metric{idx}_sigs.npz (per-metric loop)")
    ap_.add_argument("--checkpoint", default=None, help="single per-metric checkpoint (alternative to --ckpt-dir)")
    ap_.add_argument("--tau", type=float, default=None, help="collide τ (default: each checkpoint's τ — GV4)")
    ap_.add_argument("--delta", type=float, default=0.05)
    ap_.add_argument("--submodular-top-k", type=int, default=150)
    ap_.add_argument("--out-dir", default="/tmp/value_census")
    a = ap_.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)

    ckpts = ([a.checkpoint] if a.checkpoint
             else sorted(glob.glob(os.path.join(a.ckpt_dir, "*_metric*_sigs.npz"))) if a.ckpt_dir else [])
    if not ckpts:
        raise SystemExit("pass --ckpt-dir (per-metric loop) or --checkpoint (single)")
    print(f"value census (per metric, anchor-free): {len(ckpts)} checkpoints")

    results = []
    for ck in ckpts:
        rep = _run_one(ck, a)
        if rep is None:
            continue
        results.append(rep)
        sub = rep["submodular"]
        print(f"  α_i={rep['alpha_i']:.3f} α_V={rep['alpha_V_terminal']:.3f} gap={rep['breadth_gap']:+.3f} "
              f"MV0={rep['MV0']:.4f} gain_sum/H={rep['conditional_gain_sum_over_H']:.2f} "
              f"{rep['decision']['verdict']:8s} {rep['name'][:40]}")

    out_json = os.path.join(a.out_dir, "value_census_summary.json")
    with open(out_json, "w") as fout:
        json.dump(_jsonable({"n_metrics": len(results), "results": results}), fout, indent=2)
    if results:
        done = [r for r in results if np.isfinite(r["alpha_V_terminal"])]
        print("\n" + "=" * 72)
        for r in sorted(done, key=lambda r: r["breadth_gap"])[-8:]:
            print(f"  gap={r['breadth_gap']:+.3f} α_i={r['alpha_i']:.3f} α_V={r['alpha_V_terminal']:.3f} "
                  f"gain/H={r['conditional_gain_sum_over_H']:.2f} "
                  f"{r['decision']['verdict']:8s} {r['name'][:40]}")
        print(f"summary → {out_json}")


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return o


if __name__ == "__main__":
    main()
