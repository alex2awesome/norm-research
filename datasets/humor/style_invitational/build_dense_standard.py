#!/usr/bin/env python3
"""Build data.csv + stable-hash grouped 80/10/10 split for the STANDARDIZED dense
arm (Llama-3.1-8B LoRA via methods/dense/train_reward_model.py) on the Style
Invitational top-tier cell (V4 remaining-cells task, 2026-08-06).

Population/y/group EXACTLY reproduce datasets/va_gemma_banks/score_va_gemma_banks.py
build_style_invitational() (the same population the Layer-1 nonlinear-stack matrix
outputs/va_gemma_banks/style_invitational_shard*.npz was built from, verified
against methods/taste_decomposition/results/style_inv_toptier_layer1.json:
n=9637, pos_rate=0.16073466846529003, n_groups=316, group_column="week_id"):
  - ALL 9,637 rows of style_invitational.jsonl (ALL weeks, no subsampling).
  - y (top_tier) = 1 iff tier in {"winner", "runnerup"}, else 0.
  - group = str(week_id).

Text = the same CONTEST PROMPT / ENTRY context block the Gemma A-judge was given
(ctx() in score_va_gemma_banks.py).

Split: stable-hash deterministic greedy + hill-climb bucket assignment targeting
80/10/10 by row count AND matched pos-rate per bucket (see
stable_hash_bucket_map; same helper as the HashtagWars build script, fixed
2026-08-06 to also balance pos-rate -- see that script's docstring for why).
No seeded shuffle.

Usage (CPU only):
  python3 build_dense_standard.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "dense_standard"
JL = HERE / "style_invitational.jsonl"

EXPECTED_N = 9637
EXPECTED_POS_RATE = 0.16073466846529003
EXPECTED_N_GROUPS = 316


def sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()


def stable_hash_bucket_map(y_by_group: dict, targets=None, lam: float = 2.5) -> dict:
    """Deterministic greedy + hill-climb repair bin-packing of groups into
    train/eval/test buckets targeting 80/10/10 BY ROW COUNT *AND* matched
    per-bucket pos-rate. No seeded shuffle. Verbatim pattern from
    hashtagwars/build_dense_standard.py (see its docstring for why the
    pos-rate term is needed)."""
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
    rows = [json.loads(l) for l in open(JL) if l.strip()]
    out_rows = []
    for r in rows:
        y = 1 if r["tier"] in ("winner", "runnerup") else 0
        group = str(r["week_id"])
        text = f'CONTEST PROMPT: {r["contest_prompt"]}\n\nENTRY: "{r["entry_text"]}"'
        out_rows.append({"text": text, "judgement": y, "group": group, "tier": r["tier"]})

    n = len(out_rows)
    pos_rate = sum(r["judgement"] for r in out_rows) / n
    n_groups = len(set(r["group"] for r in out_rows))
    print(f"n={n} pos_rate={pos_rate!r} n_groups={n_groups}")

    assert n == EXPECTED_N, f"n mismatch: {n} != {EXPECTED_N} (Layer-1 population)"
    assert abs(pos_rate - EXPECTED_POS_RATE) < 1e-9, \
        f"pos_rate mismatch: {pos_rate!r} != {EXPECTED_POS_RATE!r} (Layer-1 population)"
    assert n_groups == EXPECTED_N_GROUPS, f"n_groups mismatch: {n_groups} != {EXPECTED_N_GROUPS}"
    print("ASSERTION PASS: rows are exactly the style_inv_toptier_layer1.json population "
          "(n, pos_rate, n_groups all match to float precision)")

    from collections import defaultdict
    y_by_group = defaultdict(list)
    for r in out_rows:
        y_by_group[r["group"]].append(r["judgement"])
    bmap = stable_hash_bucket_map(y_by_group)
    by_split = {"train": [], "eval": [], "test": []}
    for r in out_rows:
        by_split[bmap[r["group"]]].append(r)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split").mkdir(exist_ok=True)
    cols = ["text", "judgement", "group", "tier"]
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
        "cell": "style_inv_toptier",
        "source": "datasets/humor/style_invitational/style_invitational.jsonl",
        "population_recipe": "verbatim datasets/va_gemma_banks/score_va_gemma_banks.py build_style_invitational()",
        "layer1_reference": "methods/taste_decomposition/results/style_inv_toptier_layer1.json",
        "n": n,
        "pos_rate": pos_rate,
        "n_groups": n_groups,
        "group_column": "week_id",
        "y_definition": '1 iff tier in {"winner","runnerup"}, else 0 (top_tier)',
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
                  "gradient-checkpointing, select-on-eval (dense-standard, no deviation)",
    }
    with open(OUT / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps(manifest, indent=2))
    print("BUILD_DONE")


if __name__ == "__main__":
    main()
