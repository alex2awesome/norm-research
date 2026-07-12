"""Pipeline validity audit (CPU, on saved checkpoints) — tests the three open trust questions.

1. EXECUTOR CONFOUND: is B_E tracking executor discrimination (capacity artifact) or metric structure?
   Same metrics across executors (llama3b/8b/qwen122b): do B_E RANKS agree (metric-intrinsic) or
   shuffle (executor noise)? Correlate B_E with signature discrimination (mean_between_sig_l1,
   pyes_global_std, frac_near_constant_sig).
2. PROPOSER COVERAGE: do qwen/llama/haiku find DISTINCT species (coverage growing) or redundant ones
   (shared blind spots -> B_E under-estimated)? diversity_gap + per-family unique-species fraction.
3. PROBE SUFFICIENCY: where does D_obs plateau vs probe count?

Uses Chao1 (closed-form) as the B_E point proxy to avoid the slow bootstrap — fine for cross-executor
comparison. Run on saved rubric-first checkpoints.
"""
from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict

import numpy as np

from . import alpha_probe as ap


def _ckpt_basics(f):
    z = np.load(f, allow_pickle=True)
    S = np.asarray(z["sigs"], float); tags = list(z["tags"])
    gi = int(z["r2_idx"]) if "r2_idx" in z.files else -1
    name = str(z["name"]) if "name" in z.files else os.path.basename(f)
    return S, tags, gi, name


def _be_proxy(S, tags, th=0.15):
    """Fast B_E/D_obs via conditional_species + Chao1 (no bootstrap)."""
    lab = ap.conditional_species(S, cmi_thresh=th)
    v = lab >= 0
    sp = ap.spectrum(lab[v], [tags[i] for i in range(len(lab)) if v[i]])
    return sp["D"], ap.chao1(sp["f"], sp["D"]), sp["f"]


def main(argv=None):
    p = argparse.ArgumentParser(prog="pipeline_audit", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="/lfs/skampere3/0/alexspan/outputs/crc_scaling")
    p.add_argument("--executors", default="llama3b,llama8b,qwen122b")
    p.add_argument("--n-metrics", type=int, default=8)
    a = p.parse_args(argv)

    execs = a.executors.split(",")
    per_exec = defaultdict(dict)   # exec -> gi -> {D, chao1, diag...}
    for ex in execs:
        files = sorted(glob.glob(os.path.join(a.root, ex, "*_sigs.npz")))[: a.n_metrics]
        for f in files:
            S, tags, gi, name = _ckpt_basics(f)
            D, ch, fsp = _be_proxy(S, tags)
            diag = ap.signature_diagnostics(S)
            per_exec[ex][gi] = {"name": name, "D": D, "chao1": ch,
                                "disc_l1": diag["mean_between_sig_l1"],
                                "pyes_std": diag["pyes_global_std"],
                                "frac_const": diag["frac_near_constant_sig"]}

    # ---- (1a) executor discrimination profile ----
    print("=== (1) EXECUTOR CONFOUND: does B_E track discrimination (capacity) or metric structure? ===")
    print("%-10s %6s %7s %8s %8s %8s"%("executor","n","B_E(ch)","Dobs","discL1","pYesStd"))
    for ex in execs:
        rs = list(per_exec[ex].values())
        if not rs: continue
        print("%-10s %6d %7.1f %7.1f %8.3f %8.3f"%(
            ex, len(rs), float(np.mean([r["chao1"] for r in rs])), float(np.mean([r["D"] for r in rs])),
            float(np.mean([r["disc_l1"] for r in rs])), float(np.mean([r["pyes_std"] for r in rs]))))
    print("  -> if B_E and discL1/pYesStd rise together across executors, B_E measures CAPACITY (confound).")

    # ---- (1b) B_E rank agreement: PAIRWISE intersections (not the thin 3-way) ----
    print("\n  rank agreement (pairwise B_E Spearman on shared metrics):")
    from scipy.stats import spearmanr
    for i in range(len(execs)):
        for j in range(i + 1, len(execs)):
            gi_sh = sorted(set(per_exec[execs[i]]) & set(per_exec[execs[j]]))
            if len(gi_sh) < 3:
                print("    %s vs %s: only %d shared — skip" % (execs[i], execs[j], len(gi_sh))); continue
            vals_i = [per_exec[execs[i]][g]["chao1"] for g in gi_sh]
            vals_j = [per_exec[execs[j]][g]["chao1"] for g in gi_sh]
            r, _ = spearmanr(vals_i, vals_j)
            print("    %s vs %s (n=%d): Spearman = %+.2f  (high=metric-intrinsic, ~0=executor noise)" % (
                execs[i], execs[j], len(gi_sh), r))
            for g in gi_sh:
                print("       gi=%-4d %-16s %s=%-5.1f %s=%-5.1f" % (
                    g, per_exec[execs[i]][g]["name"][:16], execs[i],
                    per_exec[execs[i]][g]["chao1"], execs[j], per_exec[execs[j]][g]["chao1"]))

    # ---- (2) proposer coverage on the CONDITIONAL partition, free-gen families only ----
    print("\n=== (2) PROPOSER COVERAGE (conditional partition, free-gen families only) ===")
    ex0 = execs[1] if len(execs) > 1 else execs[0]
    files = sorted(glob.glob(os.path.join(a.root, ex0, "*_sigs.npz")))[: a.n_metrics]
    for f in files:
        S, tags, gi, name = _ckpt_basics(f)
        fg = [i for i, t in enumerate(tags) if str(t) not in ("children", "gepa")]
        lab = ap.conditional_species(S[fg], cmi_thresh=0.15)
        v = lab >= 0
        sp = ap.spectrum(lab[v], [tags[fg[i]] for i in range(len(lab)) if v[i]])
        cap = sp["capture_table"]                       # {family-subset: #species}
        tot = sum(cap.values()) or 1
        n1 = sum(val for k, val in cap.items() if len(k.split(",")) == 1)   # caught by exactly 1 list
        n3 = sum(val for k, val in cap.items() if len(k.split(",")) >= 3)   # caught by all 3 lists
        print("  %-22s D_obs(fg)=%-3d | 1-list=%-3d(%2d%%)  all-3=%-3d(%2d%%) | lists=%s" % (
            name[:22], sp["D"], n1, 100 * n1 // tot, n3, 100 * n3 // tot, ",".join(sp["families"])))
    print("  -> some all-3 overlap is NEEDED (it drives the recapture estimate);")
    print("     high 1-list% = lists complementary (good space coverage); high all-3% = redundant.")

    # ---- (3) probe sufficiency ----
    print("\n=== (3) PROBE SUFFICIENCY: D_obs vs probe count (plateau?) ===")
    for f in files[:4]:
        S, tags, gi, name = _ckpt_basics(f); npr = S.shape[1]
        row = []
        for n in [300, 240, 180, 120]:
            idx = list(range(npr)) if n == npr else sorted(np.random.default_rng(0).choice(npr, n, replace=False))
            D, _, _ = _be_proxy(S[:, idx], tags)
            row.append(D)
        print("  %-22s  300=%-3d 240=%-3d 180=%-3d 120=%-3d"%tuple([name[:22]] + row))


if __name__ == "__main__":
    main()
