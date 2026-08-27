#!/usr/bin/env python3
"""B-SIDE census -- the spurious-track mirror of m3_concepts.py.

Builds the ~40-channel peer nuisance (Track-B) census for the peer-review verdict
cell and a stratified K=6 x 3-replicate holdout design, exactly mirroring the A-side
M3 (methods/taste_decomposition/closure/robust_mm/m3_concepts.py) design decisions:

  * channel = a Track-B blind_id (round-tagged Pxx, e.g. "r4:P01") that survived BOTH
    the misrouting audit (final_route == "B") and the collapse gate (round4_results.json
    collapse_gate_dropped).  40 channels total across rounds 1-4, matching
    round4_results.json track_B.n_B_criteria_total = 40 exactly (audited below).
  * alone-AUC for the DESIGN decision (stratification) is computed on FIT+MINE ONLY,
    from the raw round{r}_scores.npz score columns -- MONITOR is never read for a
    design decision (standing rule).  MONITOR alone-AUC is cross-checked against the
    already-published round4_results.json numbers and reported descriptively.
  * strata = terciles of |alone-AUC - .5| over the 40 channels: high/mid/low.
  * 3 replicates x (2 high + 2 mid + 2 low) = K=6 per replicate, 18 distinct channels,
    no channel in two replicates.  Assignment is a stable sha256 sort over a fixed
    salt (never a seeded shuffle), per the stable-hash-splits rule.
  * unlike the A-side, NO depletion/refit step follows: Track-B channels never enter
    VA_nl (they are declared nuisances, not scored bank members), so removing one from
    "the declared set" has no footprint in any score matrix and no effect on the
    disagreement slice.  The censused channels are held out only in the sense that they
    are the DETECTION TARGETS a sealed fleet is asked to rediscover; the fleet never
    sees the declared set regardless (sealed by construction, per M1's contract).

CPU only, seconds.  Usage: python bside_census.py
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
ROBUST_MM = HERE.parent
CLOSURE = ROBUST_MM.parent
sys.path.insert(0, str(CLOSURE))
import closure_lib as L  # noqa: E402

SALT = "bside-leaveout-holdout-v1"
K = 6
N_REP = 3
PER_REP = {"high": 2, "mid": 2, "low": 2}
ROUNDS = (1, 2, 3, 4)


def hash_key(s: str) -> str:
    return hashlib.sha256(f"{SALT}|{s}".encode()).hexdigest()


def load_round_b(r, y, fitm, monm):
    """Round-r B score columns, split by that round's final routing, collapse-gated.

    Mirrors stage4_round4.load_round_blocks() but keeps only the B side and returns
    per-column records (name, instruction, alone-AUC on FIT+MINE and MONITOR).
    """
    z = np.load(CLOSURE / f"round{r}_scores.npz", allow_pickle=True)
    cids = [str(s) for s in z["crit_ids"]]
    X = z["X"]
    routing = json.loads((CLOSURE / f"round{r}_routing_final.json").read_text())
    rep_path = CLOSURE / f"round{r}_score_report.json"
    collapsed = set()
    if rep_path.exists():
        rep = json.loads(rep_path.read_text())
        collapsed = {k for k, v in rep["per_criterion"].items() if v.get("collapsed")}
    # merge BOTH track files: some B-routed channels were originally PROPOSED on
    # track A and reclassified "incidental" by the misrouting audit, so their
    # instruction/rationale text lives in round{r}_track_a.json, not track_b.json.
    track_a = json.loads((CLOSURE / f"round{r}_track_a.json").read_text())
    track_b = json.loads((CLOSURE / f"round{r}_track_b.json").read_text())
    src_meta = {c["id"]: c for c in track_a["criteria"]}
    src_meta.update({c["id"]: c for c in track_b["criteria"]})

    recs = []
    for x in routing["final"]:
        if x["final_route"] != "B" or x["blind_id"] in collapsed:
            continue
        bid = x["blind_id"]
        j = cids.index(bid)
        col = X[:, j].astype(float)
        med = float(np.nanmedian(col[fitm]))
        colf = np.where(np.isnan(col), med, col)
        a_fit = float(roc_auc_score(y[fitm], colf[fitm]))
        a_mon = float(roc_auc_score(y[monm], colf[monm]))
        src = x.get("src_id", "")
        meta = src_meta.get(src, {})
        recs.append({
            "channel": f"r{r}:{bid}",
            "round": r,
            "blind_id": bid,
            "src_id": src,
            "name": x.get("name", meta.get("name", "")),
            "instruction": meta.get("instruction", ""),
            "upstream_parent": meta.get("upstream_parent"),
            "mixed": meta.get("mixed"),
            "alone_auc_fitmine": a_fit,
            "alone_auc_monitor": a_mon,
            "nonnull_frac": float(np.mean(~np.isnan(col))),
        })
    return recs


def main():
    pop = L.load_population()
    _, split, dsplit, mining = L.load_splits()
    y = pop["y"]
    fitm = split == "fit_mine"
    monm = split == "monitor"

    rows = []
    for r in ROUNDS:
        rows += load_round_b(r, y, fitm, monm)

    print(f"B channels: {len(rows)} (audit target: 40, matches "
          f"round4_results.json track_B.n_B_criteria_total)")

    # cross-check against the already-published MONITOR alone-AUCs
    published = json.loads((CLOSURE / "round4_results.json").read_text())[
        "track_B"]["per_feature_alone_AUC_MONITOR"]
    mismatches = []
    for r in rows:
        pub = published.get(r["channel"])
        if pub is None:
            mismatches.append((r["channel"], "MISSING from published"))
        elif abs(pub - r["alone_auc_monitor"]) > 1e-9:
            mismatches.append((r["channel"], pub, r["alone_auc_monitor"]))
    print(f"cross-check vs round4_results.json: {len(mismatches)} mismatches "
          f"(0 expected)")
    if mismatches:
        for m in mismatches[:10]:
            print("  MISMATCH", m)

    for r in rows:
        r["informativeness_fitmine"] = abs(r["alone_auc_fitmine"] - 0.5)
    rows.sort(key=lambda r: -r["informativeness_fitmine"])
    n = len(rows)
    for i, r in enumerate(rows):
        r["rank_by_informativeness"] = i + 1
        r["stratum"] = "high" if i < n // 3 else ("mid" if i < 2 * (n // 3) else "low")

    by_stratum = defaultdict(list)
    for r in rows:
        by_stratum[r["stratum"]].append(r)
    replicates = {f"rep{i+1}": [] for i in range(N_REP)}
    for s, want in PER_REP.items():
        pool = sorted(by_stratum[s], key=lambda r: hash_key(r["channel"]))
        need = want * N_REP
        assert len(pool) >= need, f"stratum {s}: {len(pool)} < {need}"
        for i in range(N_REP):
            replicates[f"rep{i+1}"] += [r["channel"] for r in pool[i * want:(i + 1) * want]]

    allsel = [c for v in replicates.values() for c in v]
    assert len(allsel) == len(set(allsel)) == K * N_REP, "overlap in replicate assignment"

    by_channel = {r["channel"]: r for r in rows}

    out = {
        "design": {
            "salt": SALT, "K": K, "n_replicates": N_REP, "per_replicate_strata": PER_REP,
            "alone_auc_population": "FIT+MINE (MONITOR never read for a design decision)",
            "stratum_rule": "terciles of |alone-AUC - .5| over the 40 Track-B channels",
            "mirrors": "m3_concepts.py (A-side), K=8/3+3+2 there vs K=6/2+2+2 here",
        },
        "census": {
            "n_B_channels_total": len(rows),
            "n_rounds": len(ROUNDS),
            "cross_check_mismatches_vs_round4_results": len(mismatches),
            "strata_sizes": {s: len(v) for s, v in by_stratum.items()},
        },
        "channels": rows,
        "replicates": {k: sorted(v) for k, v in replicates.items()},
        "replicate_detail": {
            k: [{"channel": c, **{kk: by_channel[c][kk] for kk in
                                   ("round", "src_id", "name", "stratum",
                                    "alone_auc_fitmine", "alone_auc_monitor",
                                    "upstream_parent", "mixed")}}
                for c in sorted(v, key=lambda c: -by_channel[c]["informativeness_fitmine"])]
            for k, v in replicates.items()
        },
    }
    (HERE / "bside_census.json").write_text(json.dumps(out, indent=1))

    for k, v in replicates.items():
        det = out["replicate_detail"][k]
        print(f"\n{k}: " + ", ".join(f"{d['stratum']}/{d['alone_auc_fitmine']:.3f}" for d in det))
        for d in det:
            print(f"   [{d['stratum']:4s}] AUC_fit {d['alone_auc_fitmine']:.3f} "
                  f"AUC_mon {d['alone_auc_monitor']:.3f} {d['channel']:8s} {d['name'][:65]}")
    print("\nwrote", HERE / "bside_census.json")


if __name__ == "__main__":
    main()
