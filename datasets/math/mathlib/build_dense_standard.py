#!/usr/bin/env python3
"""Build data.csv + AREA-GROUPED stable-hash 80/10/10 split for the STANDARDIZED
dense arm (Llama-3.1-8B LoRA via methods/dense/train_reward_model.py) on the
mathlib4 accept/reject VERDICT cell (task D7 job 1, 2026-08-07).

WHY THIS RERUN: the old headline T=.770 is retired -- notes/2026-08-06__gap_closer_batch.md
job 2 traced it to a DIFFERENT, pre-de-confounding, 35,796-row population (title+diff,
test-split, checkpoint-selection-optimistic) that predates and does not match the current
canonical de-confounded n=7,956 slice methods/taste_decomposition/results/mathlib_verdict_layer1.json's
V'/A/VA numbers are computed on. That cell ALSO uses a single fixed, label-STRATIFIED-not-
GROUPED train/eval split (parquet's own 'split' column) -- not area-disjoint. This script
fixes both problems for a fresh rerun: (1) exact same population as the Layer-1 cell
(n=7956, pos_rate=0.9428104575163399, n_groups=31, group_column=area), (2) an AREA-GROUPED
stable-hash split (no train/eval area leakage) instead of the label-stratified one.

Population/text/group EXACTLY mirror methods/taste_decomposition/mathlib_verdict_data/
mathlib_remeasure2.py's C-leg (the cell's own "de-confounded C (author-stripped TF-IDF)"
computation, VAT_CLOSURE.md's C=.750 raw / .736 topic-resid):
  - source: accept_reject_clean.parquet (n=7,956; identical rows to accept_reject_clean_deconf.parquet,
    which merely adds derived columns -- both verified byte-identical on 'number'/'diff'/'judgement').
  - text = diff_noauth (diff with Copyright/Authors/License/maintainer lines stripped --
    save_deconf.py's own finding: "kills the dense-overfit surface at zero cost", C .742->.750
    raw). NO title concatenation: the canonical C definition for THIS de-confounded population
    is diff-only (mathlib_remeasure2.py's Cpred / mathlib_top_tfidf.py / mathlib_authorstrip_topic.py
    all use diff/diff_noauth alone, never title+diff -- title+diff was only ever used in the
    OLD, retired, pre-de-confounding 35,796-row population).
  - judgement = judgement (0=reject/closed-unmerged, 1=accept/merged).
  - group = area (top-level Mathlib/<Area>/ path most touched by the diff; regex identical to
    mathlib_remeasure2.py/save_deconf.py; 31 distinct incl. NONE).

Split: deterministic greedy + hill-climb bucket assignment targeting 80/10/10 by row count
AND matched pos-rate per bucket (stable hash tiebreak, no seeded shuffle) -- verbatim
stable_hash_bucket_map from datasets/humor/hashtagwars/build_dense_standard.py (same
degenerate-pos-rate-if-naive concern applies here: area size correlates weakly with reject
rate -- Tactic/GroupTheory/NONE/Computability run 86-91% vs 94-99% typical elsewhere).

Usage (CPU only, local or sk3):
  python3 build_dense_standard.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "dense_standard"
# local mirror of sk3's /lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib/accept_reject_clean.parquet
SRC_CANDIDATES = [
    HERE / "accept_reject_clean.parquet",
    HERE.resolve().parents[2] / "methods" / "taste_decomposition" / "mathlib_verdict_data" / "accept_reject_clean.parquet",
    Path("/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib/accept_reject_clean.parquet"),
]

EXPECTED_N = 7956
EXPECTED_POS_RATE = 0.9428104575163399
EXPECTED_N_GROUPS = 31

AUTHOR_RE = re.compile(
    r"Copyright|Authors:|Released under|Apache 2\.0|described in the file LICENSE|SPDX-License|maintainer",
    re.I,
)
AREA_RE = re.compile(r"(?:a|b)/Mathlib/([A-Za-z0-9_]+)/")


def sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()


def strip_author(d: str) -> str:
    return "\n".join(l for l in str(d).split("\n") if not AUTHOR_RE.search(l))


def area(d: str) -> str:
    ms = AREA_RE.findall(str(d))
    return Counter(ms).most_common(1)[0][0] if ms else "NONE"


def stable_hash_bucket_map(y_by_group: dict, targets=None, lam: float = 2.5) -> dict:
    """Verbatim from datasets/humor/hashtagwars/build_dense_standard.py -- deterministic
    greedy + hill-climb bin-packing of groups into train/eval/test buckets targeting
    80/10/10 BY ROW COUNT *AND* matched per-bucket pos-rate. No seeded shuffle."""
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
    src = next((p for p in SRC_CANDIDATES if p.exists()), None)
    assert src is not None, f"none of {SRC_CANDIDATES} exist -- copy accept_reject_clean.parquet first"
    print(f"reading {src}")
    df = pd.read_parquet(src).reset_index(drop=True)

    df["diff_noauth"] = df["diff"].astype(str).map(strip_author)
    df["area"] = df["diff"].astype(str).map(area)

    n = len(df)
    pos_rate = float(df["judgement"].mean())
    n_groups = df["area"].nunique()
    print(f"n={n} pos_rate={pos_rate!r} n_groups={n_groups}")
    assert n == EXPECTED_N, f"n mismatch: {n} != {EXPECTED_N} (Layer-1 population)"
    assert abs(pos_rate - EXPECTED_POS_RATE) < 1e-9, \
        f"pos_rate mismatch: {pos_rate!r} != {EXPECTED_POS_RATE!r} (Layer-1 population)"
    assert n_groups == EXPECTED_N_GROUPS, f"n_groups mismatch: {n_groups} != {EXPECTED_N_GROUPS}"
    print("ASSERTION PASS: rows are exactly the mathlib_verdict_layer1.json population "
          "(n, pos_rate, n_groups all match to float precision)")

    out_rows = []
    for _, r in df.iterrows():
        out_rows.append({
            "text": r["diff_noauth"],
            "judgement": int(r["judgement"]),
            "group": r["area"],
            "row_id": str(r["number"]),
        })

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
        "cell": "mathlib_verdict",
        "source": str(src),
        "population_recipe": "verbatim methods/taste_decomposition/mathlib_verdict_data/mathlib_remeasure2.py "
                              "C-leg population (accept_reject_clean.parquet); text=diff_noauth (author-stripped, "
                              "no title, matching the cell's own canonical C definition, VAT_CLOSURE.md .750/.736)",
        "layer1_reference": "methods/taste_decomposition/results/mathlib_verdict_layer1.json",
        "n": n, "pos_rate": pos_rate, "n_groups": n_groups,
        "group_column": "area (top-level Mathlib/<Area>/ path most touched by the diff; 31 distinct incl. NONE)",
        "y_definition": "1=accept(merged) 0=reject(closed-unmerged), judgement column verbatim",
        "split_group_counts": {s: len(set(r["group"] for r in by_split[s])) for s in by_split},
        "split_row_counts": {s: len(by_split[s]) for s in by_split},
        "split_pos_rates": {s: (sum(r["judgement"] for r in by_split[s]) / max(len(by_split[s]), 1))
                             for s in by_split},
        "split_fractions": {s: len(by_split[s]) / n for s in by_split},
        "assertion_rows_subseteq_layer1_population": (
            f"n={n} == {EXPECTED_N}, pos_rate matches to float precision, "
            f"n_groups={n_groups} == {EXPECTED_N_GROUPS} -- PASS"
        ),
        "deviation_from_original_pipeline": "original mathlib_remeasure2.py used a SINGLE fixed "
                                             "label-stratified-not-grouped split (parquet's own 'split' "
                                             "column); this rerun uses an AREA-GROUPED stable-hash 80/10/10 "
                                             "split instead (task D7 job 1 spec, notes/2026-08-06__gap_closer_batch.md job 2).",
        "recipe": "Llama-3.1-8B LoRA r16/a32, lr5e-5, batch16, max_len1024, 2 epochs, "
                  "gradient-checkpointing, select-on-eval (dense-standard, no deviation), 3 seeds (42,1,2; small n)",
    }
    with open(OUT / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps(manifest, indent=2))
    print("BUILD_DONE")


if __name__ == "__main__":
    main()
