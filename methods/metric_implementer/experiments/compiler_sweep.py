"""Compiler sweep — does a more faithful C(Ω) raise I(M, M_ω)?  (2026-06-23)

The decomposition gap I(M, M_ω) compares the prose prompt p̂'s verdict (P) to the all-criteria
compilation C(Ω)'s verdict (M_ω). The default compiler is a flat CONJUNCTION, which is a poor
formal match to a weighted-sum p̂ — so part of a low I(M, M_ω) could be a COMPILER artifact, not a
real decomposition loss. This sweep isolates that: it loads a saved omega_certificate npz (which
persists P_cont, P_bin, prose_prompt, crits) and re-scores ONLY the all-criteria verdict M_cont
under each `_compile` variant, recomputing I(M, M_ω). One prompt × N items per compiler — cheap.

DECISIVE READ:
  * If I ≈ T_prose across ALL compilers, the prose channel's spread (T_prose) is the BINDING cap,
    not the compiler framing → raise T_prose (discriminative GEPA), don't fiddle with C(Ω).
  * If a faithful compiler (weighted_sum / echo_prose) lifts I well above the conjunction's value
    AND toward T_prose, the compiler WAS losing headroom → ship the faithful compiler.

The headline `tvd_recovery` is median-split-coarse (2π(1−π)|group-mean-diff|) and saturates at
T_prose; we ALSO report the finer Spearman(M_cont, P_cont) and 4-bin tvd_mi(M_cont, P_cont),
where formal compiler differences show up even when the coarse headline does not move. The
`echo_prose` row (M_cont := P_cont) is the SELF-recovery ceiling — I → T_prose by construction.

sk3 (1 GPU):  HOME=/lfs; CUDA_VISIBLE_DEVICES=<free> $PY -m ...compiler_sweep \
    --npz tmp_vinfo/real_test3/creative-writing.npz \
    --pool <pool.jsonl.gz> --text-col text --task creative-writing \
    --model meta-llama/Llama-3.1-8B-Instruct --out /tmp/compiler_sweep.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .. import config as cfgmod
from .. import vinfo
from .real_gamma import _signal
from .small_omega_brute_force import _compile


def _load_texts(pool, text_col, n_items, max_chars_check=50):
    if pool.endswith(".parquet"):
        df = pd.read_parquet(pool)
    elif pool.endswith(".csv") or pool.endswith(".csv.gz"):
        df = pd.read_csv(pool)
    else:
        df = pd.read_json(pool, lines=True,
                          compression="gzip" if pool.endswith(".gz") else None)
    df = df[df[text_col].astype(str).str.len() > max_chars_check]
    return df[text_col].astype(str).tolist()[:n_items]


def main(argv=None):
    import sys
    ap = argparse.ArgumentParser(prog="compiler_sweep", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--task", default="creative-writing")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--out", default="/tmp/compiler_sweep.json")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args(argv)

    d = np.load(a.npz, allow_pickle=True)
    crits = list(d["crits"])
    P_cont = np.asarray(d["P_cont"], float)
    P_bin = np.asarray(d["P_bin"], float)
    prose = str(d["prose_prompt"])
    K = int(d["K"])
    N = int(d["n_items"])
    print(f"loaded npz: K={K} criteria, N={N} items; prose p̂ {len(prose)} chars")
    print(f"  crits: {crits}")

    texts = _load_texts(a.pool, a.text_col, N)
    assert len(texts) == N, f"pool gave {len(texts)} items, npz has {N} — pool mismatch"
    print(f"reloaded {len(texts)} texts from {a.pool}")

    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    cfg.vllm_fake = a.fake
    from ..vllm_backend import make_judge_backend
    backend = make_judge_backend(a.model, cfg)

    # constant across compilers (the prose channel's own spread = the DPI cap on I)
    T_prose = float(vinfo.tvd_transmission(P_cont, debias=False)["tvd_t"])
    I_self = float(vinfo.tvd_recovery(P_cont, P_bin)["tvd_recovery"])   # echo_prose ceiling
    print(f"\nT_prose (DPI cap) = {T_prose:.4f}   echo_prose self-ceiling I = {I_self:.4f}\n")

    compilers = ["conjunction", "weighted_sum", "prose_join", "echo_prose"]
    rows = []
    for comp in compilers:
        if comp == "echo_prose":
            M_cont = P_cont                       # p̂ verbatim: no re-score needed
        else:
            rubric = _compile(crits, tuple(range(K)), compiler=comp)
            M_cont = _signal(backend, rubric, texts, a.max_chars)
        I_MMw = float(vinfo.tvd_recovery(M_cont, P_bin)["tvd_recovery"])
        T_om = float(vinfo.tvd_transmission(M_cont, debias=False)["tvd_t"])
        # finer agreement measures (where formal compiler differences show up)
        m = np.isfinite(M_cont) & np.isfinite(P_cont)
        from scipy.stats import spearmanr
        sp = float(spearmanr(M_cont[m], P_cont[m]).statistic) if m.sum() >= 4 else float("nan")
        tmi4 = float(vinfo.tvd_mi(M_cont, P_cont, n_bins=4, debias=True, n_perm=32))
        rows.append({"compiler": comp, "I_M_Momega": I_MMw, "T_omega": T_om,
                     "I_over_Tprose": I_MMw / T_prose if T_prose > 1e-9 else float("nan"),
                     "spearman": sp, "tvd_mi_4bin": tmi4,
                     "M_base_rate": float(np.mean(M_cont))})
        print(f"  {comp:14s}  I(M,M_ω)={I_MMw:.4f}  T_ω={T_om:.4f}  "
              f"I/T_prose={I_MMw/T_prose:.2f}  spearman={sp:.3f}  tvd_mi4={tmi4:.3f}")

    out = {"npz": a.npz, "K": K, "N": N, "T_prose": T_prose, "echo_ceiling_I": I_self,
           "crits": crits, "rows": rows}
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    print("\n===== VERDICT =====")
    conj = next(r for r in rows if r["compiler"] == "conjunction")["I_M_Momega"]
    best = max(rows, key=lambda r: r["I_M_Momega"])
    if best["I_M_Momega"] - conj < 0.01:
        print(f"compiler INVARIANT: best ({best['compiler']}, I={best['I_M_Momega']:.4f}) ≈ "
              f"conjunction ({conj:.4f}). T_prose={T_prose:.4f} is the BINDING cap → raise T_prose.")
    else:
        print(f"compiler MATTERS: {best['compiler']} lifts I {conj:.4f}→{best['I_M_Momega']:.4f} "
              f"(cap T_prose={T_prose:.4f}). Ship the faithful compiler.")


if __name__ == "__main__":
    main()
