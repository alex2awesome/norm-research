#!/usr/bin/env python
"""
Cross-task concept-sharing matrix with permutation significance (2026-07-04, v2).

Runs at R3-METRIC level — the same units as the certificates, grids, and the
CW×humor judged matching — using outputs/hierarchy/<task>_general_r3_expanded.json
(merged_name + merged_description per group). v1 used norm_embed leaf artifacts and
was abandoned (leaf keys not row-aligned with the embedding files; and leaf level is
below the level every other analysis lives at).

For every task pair (A,B): do A's R3 criteria have unusually close counterparts among
B's criteria, beyond generic cross-task similarity?

Statistic (threshold-free): coverage(A->B) = mean over A-metrics of max cosine to any
B-metric (bge-large embeddings of "name: description"); stat = symmetrized mean.
Null: size-matched random metric sets drawn from the pool of all OTHER tasks' metrics
(excluding A and B) -> pair-specific affinity, not "criteria all sound alike".
BH-FDR across the pairs.

Embedding similarity is FILTER-grade (within-task calibration: raw BGE cos ~91%
precise only at 0.95): the matrix ranks/tests pair AFFINITY; individual top pairs are
exported for judge verification (crosstask_match.py protocol) before being called
matches.

Usage (sk3 or laptop; CPU):
  python -m methods.metric_implementer.experiments.crosstask_sharing_matrix \
      --hierarchy-dir outputs/hierarchy --bucket general \
      --min-groups 10 --n-perm 2000 \
      --out notebooks/data/two_faces_20260702/crosstask/sharing_matrix_r3.json
"""
import argparse
import glob
import json
import os
import re

import numpy as np


def load_r3(hierarchy_dir, bucket):
    tasks = {}
    for f in sorted(glob.glob(os.path.join(hierarchy_dir, f"*_{bucket}_r3_expanded.json"))):
        task = os.path.basename(f).replace(f"_{bucket}_r3_expanded.json", "")
        d = json.load(open(f))
        groups = d.get("merged_groups", d)
        if isinstance(groups, dict):
            groups = list(groups.values())
        items = []
        for g in groups:
            name = g.get("merged_name") or ""
            desc = g.get("merged_description") or ""
            if not (name or desc):
                continue
            items.append({"gi": g.get("group_idx"), "name": name,
                          "text": f"{name}: {desc}".strip(": "),
                          "n_leaves": g.get("total_leaf_rubrics")})
        if items:
            tasks[task] = items
    return tasks


def embed(texts, model_name, device):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    return model.encode(texts, batch_size=64, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)


def coverage(Va, Vb):
    S = Va @ Vb.T
    return float(np.mean(S.max(axis=1))), S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hierarchy-dir", required=True)
    ap.add_argument("--bucket", default="general")
    ap.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-groups", type=int, default=10,
                    help="skip tasks with fewer R3 groups (too thin to test)")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--top-pairs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tasks = load_r3(a.hierarchy_dir, a.bucket)
    skipped = {t: len(v) for t, v in tasks.items() if len(v) < a.min_groups}
    tasks = {t: v for t, v in tasks.items() if len(v) >= a.min_groups}
    names = sorted(tasks)
    print(f"tasks kept ({len(names)}): " +
          ", ".join(f"{t}({len(tasks[t])})" for t in names))
    if skipped:
        print(f"SKIPPED (fewer than {a.min_groups} R3 groups): {skipped}")

    all_texts, spans = [], {}
    for t in names:
        spans[t] = (len(all_texts), len(all_texts) + len(tasks[t]))
        all_texts += [it["text"] for it in tasks[t]]
    print(f"embedding {len(all_texts)} R3 criteria with {a.model} on {a.device} ...")
    E = embed(all_texts, a.model, a.device)
    V = {t: E[s:e] for t, (s, e) in spans.items()}

    rng = np.random.default_rng(a.seed)
    results = []
    for i, ta in enumerate(names):
        for tb in names[i + 1:]:
            Va, Vb = V[ta], V[tb]
            cov_ab, Sab = coverage(Va, Vb)
            cov_ba, _ = coverage(Vb, Va)
            stat = (cov_ab + cov_ba) / 2

            pool = np.vstack([V[t] for t in names if t not in (ta, tb)])
            null = np.empty(a.n_perm)
            for p in range(a.n_perm):
                Rb = pool[rng.choice(len(pool), size=len(Vb),
                                     replace=len(Vb) > len(pool))]
                Ra = pool[rng.choice(len(pool), size=len(Va),
                                     replace=len(Va) > len(pool))]
                n_ab, _ = coverage(Va, Rb)
                n_ba, _ = coverage(Vb, Ra)
                null[p] = (n_ab + n_ba) / 2
            z = float((stat - null.mean()) / max(null.std(ddof=1), 1e-9))
            p_perm = float((np.sum(null >= stat) + 1) / (a.n_perm + 1))

            flat = np.argsort(-Sab, axis=None)[:a.top_pairs]
            tops = []
            for fidx in flat:
                ia, ib = np.unravel_index(fidx, Sab.shape)
                tops.append({"cos": round(float(Sab[ia, ib]), 4),
                             "gi_a": tasks[ta][ia]["gi"], "name_a": tasks[ta][ia]["name"],
                             "gi_b": tasks[tb][ib]["gi"], "name_b": tasks[tb][ib]["name"]})
            results.append({"task_a": ta, "task_b": tb, "stat": round(stat, 4),
                            "null_mean": round(float(null.mean()), 4),
                            "null_sd": round(float(null.std(ddof=1)), 4),
                            "z": round(z, 2), "p_perm": round(p_perm, 5),
                            "cov_ab": round(cov_ab, 4), "cov_ba": round(cov_ba, 4),
                            "top_pairs": tops})
            print(f"{ta} x {tb}: stat={stat:.4f} null={null.mean():.4f}"
                  f"±{null.std(ddof=1):.4f} z={z:+.2f} p={p_perm:.4f}")

    ps = np.array([r["p_perm"] for r in results])
    order = np.argsort(ps)
    m = len(ps)
    q = np.empty(m)
    prev = 1.0
    for pos in range(m - 1, -1, -1):
        idx = order[pos]
        prev = min(prev, ps[idx] * m / (pos + 1))
        q[idx] = prev
    for r, qv in zip(results, q):
        r["q_bh"] = round(float(qv), 5)

    out = {"bucket": a.bucket, "model": a.model, "min_groups": a.min_groups,
           "n_perm": a.n_perm, "seed": a.seed,
           "note": "R3-level; stat = symmetrized mean max-cos coverage; null = "
                   "size-matched metric sets from other tasks (pair-specific "
                   "affinity). top_pairs are embedding-grade — judge before "
                   "calling any single pair a match.",
           "n_groups": {t: len(tasks[t]) for t in names},
           "skipped_thin_tasks": skipped, "pairs": results}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)

    sig = sorted([r for r in results if r["q_bh"] <= 0.05], key=lambda r: -r["z"])
    print(f"\n{len(sig)}/{len(results)} pairs significant at q<=0.05:")
    for r in sig:
        print(f"  {r['task_a']} x {r['task_b']}: z={r['z']:+.2f} q={r['q_bh']}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
