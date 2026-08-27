#!/usr/bin/env python3
"""Is the optimal claim-decomposition unit rejection-type-dependent? (offline, cached scores)

Semantic-delta beat union/lexical-delta overall but §102 slightly DROPPED while §103 jumped.
Hypothesis (patent law): §102 anticipation needs ONE reference to disclose ALL elements ->
full-element (union) unit; §103 obviousness turns on the ADDED limitation -> delta unit.
This reuses the pipeline's own split-half softmin aggregation on the cached element scores of
three units, splits AUC by rejection type with bootstrap CIs, and reports whether a
rejection-type-aware unit (union for §102, semdelta for §103) beats any single unit.

Run ON sk3 (CPU): python3 scripts/patents_unit_by_rejtype.py
"""
import json, sys
import numpy as np
sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research/scripts")
from score_gold_mixture import claim_scores_with_T
from sklearn.metrics import roc_auc_score

TB = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/processed/gold_mixture_testbed_v1"
ARMS = {"union": f"{TB}/units_ab12k_elscores.json",
        "lexical_delta": f"{TB}/units_ab12k_armC_elscores.json",
        "semantic_delta": f"{TB}/units_semdelta_elscores.json"}
COND = "retrieved_only"
RNG = np.random.default_rng(0)


def rejtype_map():
    m = {}
    for ln in open(f"{TB}/units_ab12k.jsonl"):
        r = json.loads(ln)
        k = f"{r['app_id']}||{r['ifw_number']}||{r['claim_num']}"
        m[k] = str(r.get("oa_primary_rejection_type"))
    return m


def auc_oriented(y, s):
    # score = softmin of element-disclosure; HIGH = all elements disclosed = fell -> AUC of (+score)
    return roc_auc_score(y, list(s))


def boot_ci(y, s, n=1000):
    y, s = np.asarray(y), np.asarray(s)
    idx = np.arange(len(y))
    accs = []
    for _ in range(n):
        b = RNG.choice(idx, len(idx), replace=True)
        if len(set(y[b].tolist())) < 2:
            continue
        accs.append(auc_oriented(y[b], s[b]))
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return lo, hi


def main():
    rt = rejtype_map()
    # per arm: key -> (score, label)
    scores = {}
    for name, path in ARMS.items():
        d = json.load(open(path))[COND]
        per_claim = {k: (v[0], v[1], tuple(v[2]), v[3]) for k, v in d.items() if v[1]}
        cs, bestT = claim_scores_with_T(per_claim)
        scores[name] = {k: (cs[k], per_claim[k][0]) for k in cs}
        print(f"[{name}] {len(cs)} claims, split-half T={bestT}", flush=True)

    print(f"\n{'unit':16s} {'rejtype':8s} {'n':>5s} {'AUC':>7s}  95% CI")
    keyset = set.intersection(*[set(scores[a]) for a in ARMS])
    per_type_best = {}
    for name in ARMS:
        for typ in ("102", "103", "all"):
            ks = [k for k in keyset if (typ == "all" or rt.get(k) == typ)]
            y = [scores[name][k][1] for k in ks]
            s = [scores[name][k][0] for k in ks]
            if len(set(y)) < 2 or len(y) < 30:
                continue
            a = auc_oriented(y, s)
            lo, hi = boot_ci(y, s)
            print(f"{name:16s} {typ:8s} {len(y):5d} {a:7.4f}  [{lo:.3f},{hi:.3f}]", flush=True)
            per_type_best.setdefault(typ, []).append((a, name))

    # rejection-type-aware oracle: best unit per type, applied per claim
    print("\n=== rejection-type-aware unit (best per type) vs single best ===", flush=True)
    best_unit = {}
    for typ in ("102", "103"):
        best = max(per_type_best[typ])  # max true AUC now
        best_unit[typ] = best[1]
        print(f"  §{typ}: best single unit = {best[1]} ({best[0]:.4f})", flush=True)
    # hybrid: best unit per rejection type, semantic_delta for anything else
    ks = sorted(keyset)
    y = [scores["semantic_delta"][k][1] for k in ks]
    hyb, semd = [], []
    for k in ks:
        t = rt.get(k)
        src = best_unit.get(t, "semantic_delta")
        hyb.append(scores[src][k][0])
        semd.append(scores["semantic_delta"][k][0])
    a_h, a_s = auc_oriented(y, hyb), auc_oriented(y, semd)
    lo, hi = boot_ci(y, hyb)
    print(f"  HYBRID (union@102 + semdelta@103/other) overall AUC={a_h:.4f} [{lo:.3f},{hi:.3f}] "
          f"vs pure semantic_delta {a_s:.4f}  delta={a_h - a_s:+.4f}", flush=True)
    print("UNIT_BY_REJTYPE_DONE", flush=True)


if __name__ == "__main__":
    main()
