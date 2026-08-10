#!/usr/bin/env python3
"""Build data.csv + COMPANY-GROUPED stable-hash 80/10/10 split for the STANDARDIZED
dense arm (Llama-3.1-8B LoRA via methods/dense/train_reward_model.py) on the
press-release editorial-pickup VERDICT cell (k>=3 outlets; task D7 job 2, 2026-08-07).

WHY THIS RERUN: methods/taste_decomposition/results/press_verdict_layer1.json's
T_provisional=.679 is NOT a same-rows rescore on this cell's exact 2,956-row population --
it is "the audit's own correction of an earlier dense (bge-m3) number" computed on a
DIFFERENT (72k-row, k>=1-label) population (see press_verdict_layer1.py's own
T_PROVISIONAL_SOURCE string). This script builds an honest, same-population,
company-grouped dense-standard run so a real T can be compared against VA_lin=.6712/
VA_nl=.7011 -- this decides whether press's "bank>=dense" standing survives an
apples-to-apples dense.

Population/text/group EXACTLY mirror methods/taste_decomposition/press_verdict_layer1.py's
own population loader (load_population() / build_v_matrix()) -- the SAME 2,956-row k>=3
population the current V/A/VA Layer-1 numbers are computed on:
  - ids/y/company from methods/taste_decomposition/results/press_verdict_pr_A_k3_scores_CACHE.npz
    (scp'd from sk3's run_A_layer_k3.py output; n=2956, pos=1478/1478, companies=556).
  - text = clean_text(id2text[id]) from datasets/press-releases/press_release_deconfounded.parquet
    (id -> text join; clean_text = null-byte strip only, verbatim
    press_verdict_v_features_recon.py's clean_text -- the SAME text the V-feature bank read).
  - judgement = y (1 iff editorial pickup by >=3 distinct tracked outlets, else a topic-matched
    n_out==0 negative).
  - group = company (comp array; 556 distinct).

Split: deterministic greedy + hill-climb bucket assignment targeting 80/10/10 by row count
AND matched pos-rate per bucket (stable hash tiebreak, no seeded shuffle) -- verbatim
stable_hash_bucket_map from datasets/humor/hashtagwars/build_dense_standard.py.

Usage (CPU only, local or sk3):
  python3 build_dense_standard_k3.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "dense_standard_k3"
A_CACHE_CANDIDATES = [
    HERE.resolve().parents[1] / "methods" / "taste_decomposition" / "results" / "press_verdict_pr_A_k3_scores_CACHE.npz",
    Path("/lfs/skampere3/0/alexspan/norm-research/methods/taste_decomposition/results/press_verdict_pr_A_k3_scores_CACHE.npz"),
]
PARQUET_CANDIDATES = [
    HERE / "press_release_deconfounded.parquet",
    Path("/lfs/skampere3/0/alexspan/norm-research/datasets/press-releases/press_release_deconfounded.parquet"),
]

EXPECTED_N = 2956
EXPECTED_POS_RATE = 0.5
EXPECTED_N_GROUPS = 556


def sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()


def clean_text(s):
    if not isinstance(s, str):
        return ""
    return s.replace("\x00", " ")


def stable_hash_bucket_map(y_by_group: dict, targets=None, lam: float = 2.5) -> dict:
    """Verbatim from datasets/humor/hashtagwars/build_dense_standard.py."""
    targets = targets or {"train": .8, "eval": .1, "test": .1}
    sizes = {g: len(v) for g, v in y_by_group.items()}
    pos = {g: sum(v) for g, v in y_by_group.items()}
    total = sum(sizes.values())
    overall_rate = sum(pos.values()) / total
    order = sorted(sizes, key=lambda g: (-sizes[g], sha1(g)))
    filled = {b: 0 for b in targets}
    filled_pos = {b: 0 for b in targets}
    bmap = {}

    def obj():
        o = sum((filled[b] / total - targets[b]) ** 2 for b in targets)
        o += lam * sum(((filled_pos[b] / max(filled[b], 1)) - overall_rate) ** 2 for b in targets)
        return o

    for g in order:
        best_b, best_o = None, None
        for b in targets:
            filled[b] += sizes[g]; filled_pos[b] += pos[g]
            o = obj()
            if best_o is None or o < best_o:
                best_o, best_b = o, b
            filled[b] -= sizes[g]; filled_pos[b] -= pos[g]
        bmap[g] = best_b
        filled[best_b] += sizes[g]; filled_pos[best_b] += pos[g]

    improved = True
    n_iter = 0
    while improved and n_iter < 20:
        improved = False
        n_iter += 1
        for g in order:
            cur = bmap[g]
            best_b, best_o = cur, obj()
            for b in targets:
                if b == cur:
                    continue
                filled[cur] -= sizes[g]; filled_pos[cur] -= pos[g]
                filled[b] += sizes[g]; filled_pos[b] += pos[g]
                o = obj()
                if o < best_o - 1e-12:
                    best_b, best_o = b, o
                filled[b] -= sizes[g]; filled_pos[b] -= pos[g]
                filled[cur] += sizes[g]; filled_pos[cur] += pos[g]
            if best_b != cur:
                filled[cur] -= sizes[g]; filled_pos[cur] -= pos[g]
                filled[best_b] += sizes[g]; filled_pos[best_b] += pos[g]
                bmap[g] = best_b
                improved = True
    return bmap


def main():
    a_cache = next((p for p in A_CACHE_CANDIDATES if p.exists()), None)
    parquet = next((p for p in PARQUET_CANDIDATES if p.exists()), None)
    assert a_cache is not None, f"none of {A_CACHE_CANDIDATES} exist"
    assert parquet is not None, f"none of {PARQUET_CANDIDATES} exist"
    print(f"reading {a_cache}\nreading {parquet}")

    d = np.load(a_cache, allow_pickle=True)
    ids, y, comp = d["ids"].astype(str), d["y"].astype(int), d["comp"].astype(str)

    df = pd.read_parquet(parquet, columns=["id", "text"])
    df["id"] = df["id"].astype(str)
    id2text = dict(zip(df["id"], df["text"].fillna("")))
    missing = [i for i in ids if i not in id2text]
    assert not missing, f"{len(missing)} ids missing from parquet"

    n = len(ids)
    pos_rate = float(y.mean())
    n_groups = len(set(comp))
    print(f"n={n} pos_rate={pos_rate!r} n_groups={n_groups}")
    assert n == EXPECTED_N, f"n mismatch: {n} != {EXPECTED_N}"
    assert abs(pos_rate - EXPECTED_POS_RATE) < 1e-9, f"pos_rate mismatch: {pos_rate!r} != {EXPECTED_POS_RATE!r}"
    assert n_groups == EXPECTED_N_GROUPS, f"n_groups mismatch: {n_groups} != {EXPECTED_N_GROUPS}"
    print("ASSERTION PASS: rows are exactly the press_verdict_layer1.json k>=3 population "
          "(n, pos_rate, n_groups all match to float precision)")

    out_rows = []
    for i, g, yy in zip(ids, comp, y):
        out_rows.append({"text": clean_text(id2text[i]), "judgement": int(yy), "group": g, "row_id": i})

    y_by_group = defaultdict(list)
    for r in out_rows:
        y_by_group[r["group"]].append(r["judgement"])
    by_split = {"train": [], "eval": [], "test": []}
    bmap = stable_hash_bucket_map(y_by_group)
    for r in out_rows:
        by_split[bmap[r["group"]]].append(r)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split").mkdir(exist_ok=True)
    cols = ["text", "judgement", "group", "row_id"]
    with open(OUT / "data.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    for split in ("train", "eval", "test"):
        with open(OUT / "split" / f"{split}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(by_split[split])

    manifest = {
        "cell": "press_verdict (k>=3 outlets)",
        "source_a_cache": str(a_cache),
        "source_parquet": str(parquet),
        "population_recipe": "verbatim methods/taste_decomposition/press_verdict_layer1.py load_population()"
                              "/build_v_matrix() population (ids/y/company from "
                              "press_verdict_pr_A_k3_scores_CACHE.npz, text joined from "
                              "press_release_deconfounded.parquet, clean_text=null-byte-strip only)",
        "layer1_reference": "methods/taste_decomposition/results/press_verdict_layer1.json",
        "n": n, "pos_rate": pos_rate, "n_groups": n_groups,
        "group_column": "company",
        "y_definition": "1 iff editorial pickup by >=3 distinct tracked outlets, else topic-matched n_out==0 negative",
        "split_group_counts": {s: len(set(r["group"] for r in by_split[s])) for s in by_split},
        "split_row_counts": {s: len(by_split[s]) for s in by_split},
        "split_pos_rates": {s: (sum(r["judgement"] for r in by_split[s]) / max(len(by_split[s]), 1))
                             for s in by_split},
        "split_fractions": {s: len(by_split[s]) / n for s in by_split},
        "assertion_rows_subseteq_layer1_population": (
            f"n={n} == {EXPECTED_N}, pos_rate matches to float precision, "
            f"n_groups={n_groups} == {EXPECTED_N_GROUPS} -- PASS"
        ),
        "recipe": "Llama-3.1-8B LoRA r16/a32, lr5e-5, batch16, max_len1024, 2 epochs, "
                  "gradient-checkpointing, select-on-eval (dense-standard, no deviation), 3 seeds (42,1,2; small n)",
    }
    with open(OUT / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps(manifest, indent=2))
    print("BUILD_DONE")


if __name__ == "__main__":
    main()
