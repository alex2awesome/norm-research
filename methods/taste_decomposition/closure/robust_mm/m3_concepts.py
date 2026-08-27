#!/usr/bin/env python3
"""M3 step 1 -- concept census of the peer-verdict A bank + stratified holdout design.

The bank ships 154 delivered criteria but only 95 distinct concepts; after the
frozen column-degeneracy screen (learned on FIT+MINE only) 79 columns survive
carrying 54 distinct concepts (notes/2026-08-06__closure-swap-and-missing-mass.md
Sec 2.4).  M3 holds out sets of K=8 of those 54 concepts, stratified by alone-AUC,
and asks whether a sealed multi-proposer fleet rediscovers them.

Design decisions recorded here so the holdouts are reproducible:
  * concept = distinct A criterion NAME among surviving columns.
  * alone-AUC is computed on FIT+MINE ONLY (MONITOR is never read for a design
    decision).  MONITOR alone-AUC is reported descriptively afterwards.
  * strata = terciles of |alone-AUC - .5| (informativeness), 18/18/18.
  * 3 replicates x (3 high + 3 mid + 2 low) = 24 distinct concepts, no concept in
    two replicates.  Assignment is a stable sha256 sort over a fixed salt, never a
    seeded shuffle (stable-hash-splits rule).
  * bit-identical duplicate columns are located explicitly (by value, not only by
    name) so depletion removes the concept's whole footprint.

CPU only, seconds.  Usage: python m3_concepts.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
CLOSURE = HERE.parent
sys.path.insert(0, str(CLOSURE))
import closure_lib as L  # noqa: E402

SALT = "m3-leaveout-holdout-v1"
K = 8
N_REP = 3
PER_REP = {"high": 3, "mid": 3, "low": 2}


def hash_key(s: str) -> str:
    return hashlib.sha256(f"{SALT}|{s}".encode()).hexdigest()


def main():
    pop = L.load_population()
    _, split, dsplit, mining = L.load_splits()
    y, nt, A = pop["y"], pop["ntitle"], pop["A"]
    names = pop["a_names"]
    fitm = split == "fit_mine"
    monm = split == "monitor"

    keepA, medA = L.clean_fit(A[fitm])
    kept_names = [names[j] for j in keepA]
    concepts = sorted(set(kept_names))
    print(f"A columns {A.shape[1]} -> surviving {len(keepA)} -> distinct concepts {len(concepts)}")

    # --- exact-duplicate footprint (by VALUE, across the whole A matrix) ---------
    # so depletion removes bit-identical copies even if they carry a different name.
    sig_to_cols = defaultdict(list)
    for j in range(A.shape[1]):
        col = A[:, j].astype(float)
        sig = hashlib.sha256(np.ascontiguousarray(np.nan_to_num(col, nan=-999.0)).tobytes()).hexdigest()
        sig_to_cols[sig].append(j)

    concept_cols = {}          # concept -> all A column indices in its footprint
    for c in concepts:
        by_name = [j for j in range(A.shape[1]) if names[j] == c]
        foot = set(by_name)
        for j in by_name:
            col = A[:, j].astype(float)
            sig = hashlib.sha256(np.ascontiguousarray(np.nan_to_num(col, nan=-999.0)).tobytes()).hexdigest()
            foot |= set(sig_to_cols[sig])
        concept_cols[c] = sorted(foot)

    cross_name_dupes = {c: [names[j] for j in v if names[j] != c]
                        for c, v in concept_cols.items()
                        if any(names[j] != c for j in v)}

    # --- alone-AUC on FIT+MINE (design) and MONITOR (descriptive) ---------------
    rows = []
    for c in concepts:
        j = [k for k in keepA if names[k] == c][0]
        med = float(medA[list(keepA).index(j)])
        col = np.where(np.isnan(A[:, j]), med, A[:, j])
        a_fit = float(roc_auc_score(y[fitm], col[fitm]))
        a_mon = float(roc_auc_score(y[monm], col[monm]))
        rows.append({
            "concept": c,
            "n_columns_in_footprint": len(concept_cols[c]),
            "n_surviving_columns": int(sum(1 for k in keepA if names[k] == c)),
            "alone_auc_fitmine": a_fit,
            "alone_auc_monitor": a_mon,
            "informativeness_fitmine": abs(a_fit - 0.5),
            "nonnull_frac": float(np.mean(~np.isnan(A[:, j]))),
        })

    rows.sort(key=lambda r: -r["informativeness_fitmine"])
    n = len(rows)
    for i, r in enumerate(rows):
        r["rank_by_informativeness"] = i + 1
        r["stratum"] = "high" if i < n // 3 else ("mid" if i < 2 * (n // 3) else "low")

    # --- stratified, non-overlapping replicate assignment -----------------------
    by_stratum = defaultdict(list)
    for r in rows:
        by_stratum[r["stratum"]].append(r)
    replicates = {f"rep{i+1}": [] for i in range(N_REP)}
    for s, want in PER_REP.items():
        pool = sorted(by_stratum[s], key=lambda r: hash_key(r["concept"]))
        need = want * N_REP
        assert len(pool) >= need, f"stratum {s}: {len(pool)} < {need}"
        for i in range(N_REP):
            replicates[f"rep{i+1}"] += [r["concept"] for r in pool[i * want:(i + 1) * want]]

    allsel = [c for v in replicates.values() for c in v]
    assert len(allsel) == len(set(allsel)) == K * N_REP, "overlap in replicate assignment"

    out = {
        "design": {
            "salt": SALT, "K": K, "n_replicates": N_REP, "per_replicate_strata": PER_REP,
            "alone_auc_population": "FIT+MINE (MONITOR never read for a design decision)",
            "stratum_rule": "terciles of |alone-AUC - .5| over the 54 distinct concepts",
        },
        "census": {
            "n_A_columns_delivered": int(A.shape[1]),
            "n_A_columns_surviving_screen": int(len(keepA)),
            "n_distinct_concepts_surviving": len(concepts),
            "concepts_with_cross_name_bit_identical_columns": cross_name_dupes,
        },
        "concepts": rows,
        "replicates": {k: sorted(v) for k, v in replicates.items()},
        "replicate_detail": {
            k: [{"concept": c,
                 "stratum": next(r["stratum"] for r in rows if r["concept"] == c),
                 "alone_auc_fitmine": next(r["alone_auc_fitmine"] for r in rows if r["concept"] == c),
                 "alone_auc_monitor": next(r["alone_auc_monitor"] for r in rows if r["concept"] == c),
                 "footprint_columns": concept_cols[c]}
                for c in sorted(v, key=lambda c: -next(r["informativeness_fitmine"]
                                                       for r in rows if r["concept"] == c))]
            for k, v in replicates.items()
        },
        "concept_footprints": {c: concept_cols[c] for c in concepts},
    }
    (HERE / "m3_concepts.json").write_text(json.dumps(out, indent=1))

    print(f"cross-name bit-identical footprints: {len(cross_name_dupes)}")
    for k, v in replicates.items():
        det = out["replicate_detail"][k]
        print(f"\n{k}: " + ", ".join(f"{d['stratum']}/{d['alone_auc_fitmine']:.3f}" for d in det))
        for d in det:
            print(f"   [{d['stratum']:4s}] AUC_fit {d['alone_auc_fitmine']:.3f} "
                  f"AUC_mon {d['alone_auc_monitor']:.3f} ({len(d['footprint_columns'])}col) {d['concept'][:70]}")
    print("\nwrote", HERE / "m3_concepts.json")


if __name__ == "__main__":
    main()
