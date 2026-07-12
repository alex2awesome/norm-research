"""Validate the shared-Ω retarget confound on EXISTING 8B/3B native certificates (CPU, no GPU).

The retarget design reuses one executor's signatures (8B) with another's target (70B/Qwen).
The core confound (per the 2026-07-02 adversarial review): does the SOURCE of signatures
change the greedy HEAD (OPT_Ω criterion selection)? If barely, sig-reuse is empirically benign;
if a lot, OPT_Ω(70B) is a poor proxy for 70B's native ceiling.

We can't score 70B's own sigs, but we have TWO native executors (8B, 3B) on the SAME criteria +
probes, so we bound the confound directly:

  (A) NATIVE overlap   head(8B-sigs, 8B-M) vs head(3B-sigs, 3B-M)   [both sigs+target differ]
  (B) SIGS-source effect, 8B target held fixed:
        head(8B-sigs, 8B-M) vs head(3B-sigs, 8B-M)   [only the signature source differs]
      -> this IS the retarget operation (foreign sigs + native target). (B)>0.7 benign, <0.4 strong confound.

Jaccard is over the SELECTED CRITERIA (by prompt text), so it is invariant to row ordering.
Run on sk3 where the checkpoints live:
    python -m methods.metric_implementer.experiments.validate_retarget_overlap
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

from .value_certificate import greedy_head

# Use the _v2 dirs: v1 aligned_*_orbit have WRONG M_i (level-dispatch bug, pre-fix).
DIR8 = "/lfs/skampere3/0/alexspan/outputs/r3_cw/aligned_8b_orbit_v2"
DIR3 = "/lfs/skampere3/0/alexspan/outputs/r3_cw/aligned_3b_orbit_v2"


def _load(d: str) -> dict:
    out = {}
    for f in glob.glob(os.path.join(d, "*_sigs.npz")):
        z = np.load(f, allow_pickle=True)
        if "M_i" not in z.files:
            continue
        out[os.path.basename(f)] = {
            "sigs": np.asarray(z["sigs"], float),
            "M": np.asarray(z["M_i"], float),
            "prompts": [str(p) for p in z["prompts"]],
        }
    return out


def _head(sigs: np.ndarray, M: np.ndarray):
    """Mirror value_certificate.certificate preprocessing, return full greedy_head dict."""
    B = (np.nan_to_num(sigs, nan=0.5) > 0.5).astype(int)
    m = (M > 0.5).astype(int)
    return greedy_head(B, m)


def _head_prompts(sigs: np.ndarray, M: np.ndarray, prompts: list[str]) -> set[str]:
    sel = _head(sigs, M)["selected"]
    idx = [s for s in sel if isinstance(s, (int, np.integer))]
    return {prompts[int(i)] for i in idx}


def _jacc(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def main() -> None:
    d8 = _load(DIR8)
    d3 = _load(DIR3)
    common = sorted(set(d8) & set(d3))
    print(f"8B ckpts={len(d8)}  3B ckpts={len(d3)}  common={len(common)}")
    rows = []
    for k in common:
        p8, p3 = d8[k]["prompts"], d3[k]["prompts"]
        if set(p8) != set(p3) or d8[k]["M"].shape[0] != d3[k]["sigs"].shape[1]:
            continue
        h88 = _head(d8[k]["sigs"], d8[k]["M"])   # 8B-native
        h33 = _head(d3[k]["sigs"], d3[k]["M"])   # 3B-native
        h38 = _head(d3[k]["sigs"], d8[k]["M"])   # 3B-sigs + 8B-target  (retarget sig-swap)
        h83 = _head(d8[k]["sigs"], d3[k]["M"])   # 8B-sigs + 3B-target  (target-swap)
        def sel(h, pp):
            return {pp[i] for i in h["selected"] if isinstance(i, (int, np.integer))}
        rows.append({"k": k, "j_nat": _jacc(sel(h88, p8), sel(h33, p3)), "j_sigs": _jacc(sel(h88, p8), sel(h38, p3)),
                     "o88": h88["opt_omega_bits"], "o33": h33["opt_omega_bits"],
                     "o38": h38["opt_omega_bits"], "o83": h83["opt_omega_bits"],
                     "f88": h88["frac_H"], "f38": h38["frac_H"], "f83": h83["frac_H"]})
    if not rows:
        print("no aligned metric pairs found.")
        return

    def stats(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        c = np.corrcoef(a, b)[0, 1] if a.std() > 1e-9 and b.std() > 1e-9 else float("nan")
        return float(np.mean(np.abs(a - b))), float(c), float(a.mean()), float(b.mean())

    jn = np.array([r["j_nat"] for r in rows]); js = np.array([r["j_sigs"] for r in rows])
    o88 = np.array([r["o88"] for r in rows]); o33 = np.array([r["o33"] for r in rows])
    o38 = np.array([r["o38"] for r in rows]); o83 = np.array([r["o83"] for r in rows])
    print(f"\nHEAD-Jaccard (expected ~0 with the redundant ~775-freegen pool -> NOT decisive):")
    print(f"  (A) native 8Bv3B   mean={jn.mean():.3f} median={np.median(jn):.3f}")
    print(f"  (B) sigs-swap@8B-M mean={js.mean():.3f} median={np.median(js):.3f}")
    mad_sigs, c_sigs, m88, m38 = stats(o88, o38)   # (C) sigs-source effect @ 8B target  = CONFOUND
    mad_tgt, c_tgt, _, m83 = stats(o88, o83)       # (D) target effect @ 8B sigs          = GENUINE SCALING SIGNAL
    mad_nat, c_nat, _, _ = stats(o88, o33)         # (E) native (both sigs+target differ)
    print(f"\nOPT_Ω (bits) — DECISIVE:")
    print(f"  (C) SIGS-source  OPT(8Bsigs,8Bm) vs OPT(3Bsigs,8Bm): mean|Δ|={mad_sigs:.4f} corr={c_sigs:.3f} (means {m88:.4f} vs {m38:.4f})  <- CONFOUND")
    print(f"  (D) TARGET       OPT(8Bsigs,8Bm) vs OPT(8Bsigs,3Bm): mean|Δ|={mad_tgt:.4f} corr={c_tgt:.3f} (means {m88:.4f} vs {m83:.4f})  <- SCALING SIGNAL")
    print(f"  (E) native       OPT(8Bsigs,8Bm) vs OPT(3Bsigs,3Bm): mean|Δ|={mad_nat:.4f} corr={c_nat:.3f}")
    print(f"\nVERDICT: (C)≪(D) => OPT_Ω driven by executor M_i (target), not borrowed sigs => RETARGET SOUND.")
    print(f"         (C)≈(D) or (C)>(D) => sigs matter as much as executor => CONFOUND.")
    print(f"  here (C)={mad_sigs:.4f} (D)={mad_tgt:.4f}  D/C={mad_tgt/max(mad_sigs,1e-9):.2f}")
    print("\nper-metric (first 14): o88 | o38(sigs-swap) | o83(target-swap) | frac_H 88/38/83")
    for r in rows[:14]:
        print(f"  {r['k'][:34]:34s} {r['o88']:.3f} | {r['o38']:.3f} | {r['o83']:.3f} | {r['f88']:.2f}/{r['f38']:.2f}/{r['f83']:.2f}")


if __name__ == "__main__":
    sys.exit(main())
