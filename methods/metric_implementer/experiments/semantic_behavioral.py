"""Pre-semantic vs post-behavioral species agreement (validity of the dedup).

Does a SEMANTIC judgement ("are these two criteria the same paraphrase?", GLM-4.7) match the
BEHAVIORAL judgement ("do they induce the same YES/NO verdict pattern?", conditional_species)?

The checkpoint stores both inputs: `prompts` (criteria text -> semantic side, via GLM) and `sigs`
(soft P(YES) signatures -> behavioral side, via conditional_species). We sample criterion PAIRS from
three strata — within-species, between-species, and borderline (high MI, different species) — ask GLM
SAME/DIFFERENT, and build the confusion matrix. Agreement => the behavioral dedup is semantically
meaningful; the mismatch cells are the informative ones:

  * behavioral-same & semantic-DIFFERENT: two distinct criteria that behave identically (redundant
    criteria the executor can't tell apart — an executor-resolution ceiling).
  * behavioral-different & semantic-SAME : paraphrases that behave differently (wording-sensitive —
    the form-fragility signal, directly observed at the pair level).
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np

from .. import backends, config as cfgmod
from . import alpha_probe as ap


def _pairs_within_between(lab, tags, n_within, n_between, seed=0):
    rng = np.random.default_rng(seed)
    species = {}
    for i, l in enumerate(lab):
        if l >= 0:
            species.setdefault(int(l), []).append(i)
    iid = [i for i, t in enumerate(tags) if str(t) not in ("children", "gepa")]
    within = []
    for members in species.values():
        members = [m for m in members if m in set(iid)]
        if len(members) >= 2:
            idx = rng.choice(len(members), size=min(len(members), 8), replace=False)
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    within.append((members[idx[a]], members[idx[b]]))
    rng.shuffle(within); within = within[:n_within]
    between = []
    guard = 0
    pool = iid
    while len(between) < n_between and guard < 20 * n_between and len(species) >= 2:
        guard += 1
        i, j = int(rng.choice(pool)), int(rng.choice(pool))
        if i != j and lab[i] != lab[j]:
            between.append((i, j))
    return within, between


def main(argv=None):
    p = argparse.ArgumentParser(prog="semantic_behavioral", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-within", type=int, default=50)
    p.add_argument("--n-between", type=int, default=50)
    p.add_argument("--n-borderline", type=int, default=30)
    p.add_argument("--cmi-thresh", type=float, default=0.15)
    p.add_argument("--glm-model", default="glm-4.7")
    args = p.parse_args(argv)

    z = np.load(args.ckpt, allow_pickle=True)
    S = np.asarray(z["sigs"], float); tags = list(z["tags"]); prompts = list(z["prompts"])
    name = str(z["name"]) if "name" in z.files else os.path.basename(args.ckpt)
    lab = ap.conditional_species(S, cmi_thresh=args.cmi_thresh)
    from collections import Counter
    cnt = Counter(int(x) for x in lab)
    nsp = len([k for k in cnt if k >= 0])
    print(f"=== semantic vs behavioral: {name[:40]} (n_crit={len(tags)}, species={nsp}, "
          f"dropped(const)={cnt.get(-1, 0)}, top sizes={sorted((v for k, v in cnt.items() if k >= 0), reverse=True)[:6]}) ===")

    within, between = _pairs_within_between(lab, tags, args.n_within, args.n_between, seed=0)
    # borderline: highest pairwise MI among DIFFERENT-species FREE-GEN pairs (the threshold-edge cases)
    from .be_variance import mi_matrix_and_h
    iid = [i for i, t in enumerate(tags) if str(t) not in ("children", "gepa")]
    Bfg = (np.nan_to_num(S[iid], nan=0.5) > 0.5).astype(int)   # free-gen criteria (rows) x probes
    MIfg, _, _ = mi_matrix_and_h(Bfg)                          # criteria x criteria MI
    cross = []
    for a in range(len(iid)):
        for b in range(a + 1, len(iid)):
            if lab[iid[a]] >= 0 and lab[iid[b]] >= 0 and lab[iid[a]] != lab[iid[b]]:
                cross.append((float(MIfg[a, b]), iid[a], iid[b]))
    cross.sort(key=lambda x: -x[0])
    borderline = [(a, b) for _, a, b in cross[: args.n_borderline]]

    pairs = [(i, j, "within") for i, j in within] + \
            [(i, j, "between") for i, j in between] + \
            [(i, j, "borderline") for i, j in borderline]
    if not pairs:
        print("  no pairs to test"); return
    print(f"  testing {len(within)} within + {len(between)} between + {len(borderline)} borderline pairs via {args.glm_model}")

    cfg = cfgmod.ImplementerConfig(backend="zai_anthropic")
    glm = backends.LLMBackend(args.glm_model, "semcheck", cfg)
    sysp = "You decide whether two evaluation criteria test the same thing. Consider only whether they " \
           "evaluate the SAME dimension/aspect of the item, ignoring wording style or specificity level."
    qprompts = [f"Criterion A: {prompts[i]}\n\nCriterion B: {prompts[j]}\n\nDo A and B evaluate the SAME "
                f"aspect of the item (one is a paraphrase of the other), or DIFFERENT aspects? "
                f"Reply with one word: SAME or DIFFERENT." for i, j, _ in pairs]
    outs = glm.generate_batch(qprompts, system=sysp, max_tokens=5, temperature=0.0, seed=0)
    sem = []
    for o in outs:
        t = (o or "").strip().upper()
        sem.append(1 if re.search(r"\bSAME\b", t) and not re.search(r"\bDIFFERENT\b", t) else 0)

    # behavioral label: within=1 (same species), between/borderline=0 (different species)
    beh = [1 if k == "within" else 0 for _, _, k in pairs]
    # confusion over within+between (borderline is a separate diagnostic stratum, not in the base matrix)
    base = [m for m in range(len(pairs)) if pairs[m][2] != "borderline"]
    from collections import Counter
    cm = Counter((beh[m], sem[m]) for m in base)
    tot = len(base) or 1
    print(f"  confusion (behavioral x semantic), n={len(base)}:")
    print(f"    beh-SAME  & sem-SAME  (paraphrase, behaves same): {cm[(1,1)]:3d}  ({100*cm[(1,1)]/tot:.0f}%)")
    print(f"    beh-SAME  & sem-DIFF  (distinct but redundant):    {cm[(1,0)]:3d}  ({100*cm[(1,0)]/tot:.0f}%)  <- executor can't separate")
    print(f"    beh-DIFF  & sem-SAME  (paraphrase, behaves diff): {cm[(0,1)]:3d}  ({100*cm[(0,1)]/tot:.0f}%)  <- wording-sensitive")
    print(f"    beh-DIFF  & sem-DIFF  (distinct, behaves diff):   {cm[(0,0)]:3d}  ({100*cm[(0,0)]/tot:.0f}%)")
    agree = (cm[(1, 1)] + cm[(0, 0)]) / tot
    # precision of behavioral-merge as a paraphrase detector, and of behavioral-split
    merge_den = (cm[(1, 1)] + cm[(1, 0)]) or 1
    print(f"  agreement = {agree:.2f} | P(sem-SAME | beh-SAME) = {cm[(1,1)]/merge_den:.2f} (merge precision)")
    # borderline stratum: of high-MI cross-species pairs, how many are semantic-SAME?
    bl_same = sum(sem[m] for m in range(len(pairs)) if pairs[m][2] == "borderline")
    bl_n = sum(1 for m in range(len(pairs)) if pairs[m][2] == "borderline")
    if bl_n:
        print(f"  borderline (high-MI, different species): {bl_same}/{bl_n} semantic-SAME "
              f"({100*bl_same/bl_n:.0f}%) <- high => many near-merges are real paraphrases (threshold sane)")


if __name__ == "__main__":
    main()
