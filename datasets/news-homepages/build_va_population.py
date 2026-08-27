#!/usr/bin/env python3
"""Freeze the homepage-curation A/V population (OUTLET-HELD-OUT) and build the
dense-standard arm on exactly that population.

Cell: journalism x curation. The registry line is
"collected (homepage placement; outlet-held-out census .593 / llm .531;
T .824 groupsplit provisional) - integrate + stack": the data exist, an A bank
was scored once on a 4,400-row sample by a 70B judge, and no V+A stack was ever
fitted. This build integrates them into one instrument on one population.

WEAK-INSTRUMENT FLAG (carried into every downstream number): y here is
homepage SPATIAL PLACEMENT -- top half vs bottom half of the rendered top-30%
zone of a homepage capture -- which is an editorial prominence decision made
jointly with layout constraints, ad slots, image availability and template
rules. It is the weakest preference signal in the grid, and outlet identity
alone predicts it strongly (BBC ~34% top vs Washington Post ~73% top), which is
exactly why the design is outlet-held-out.

Population source: homepage_newsworthiness_clean_v9.csv.gz (133,130 rows /
13,666 snapshots, columns text/judgement/snapshot_id) joined to the recovered
outlet map (analysis/recover_outlet_map.py; 97.8% of rows resolved at a median
per-snapshot coverage share of 1.000). Rows whose snapshot is `ambiguous` are
dropped.

Sampling: per outlet, whole SNAPSHOTS in stable-hash order
sha256("homepage-va-v1|" + snapshot_id) until that outlet's cap is met. Capping
per outlet (rather than a uniform prefix) is what makes an outlet-held-out fold
meaningful: the natural mix is 37% CNN.

Dense standard: OUTLET-grouped 80/10/10 (train 6 outlets / eval 1 / test 1) via
the frozen stable_hash_bucket_map bin-packer -- the registry's outlet-held-out
specification, so T and the V/A readouts are computed against the same
generalisation demand.

Usage (CPU only, run on sk3):
  python3 build_va_population.py --pop <clean_v9.csv.gz> --map <outlet_map_v9.csv.gz> \
      --outdir datasets/news-homepages/va
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

SALT = "homepage-va-v1|"
PER_OUTLET_CAP = 1700


def sha256(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8")).hexdigest()


def sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()


def stable_hash_bucket_map(y_by_group: dict, targets=None, lam: float = 2.5) -> dict:
    """Verbatim from datasets/humor/hashtagwars/build_dense_standard.py."""
    targets = targets or {"train": .8, "eval": .1, "test": .1}
    sizes = {g: len(v) for g, v in y_by_group.items()}
    pos = {g: sum(v) for g, v in y_by_group.items()}
    total = sum(sizes.values())
    overall_rate = sum(pos.values()) / total
    order = sorted(sizes, key=lambda g: (-sizes[g], sha1(str(g))))
    filled = {b: 0 for b in targets}
    filled_pos = {b: 0 for b in targets}
    bmap = {}

    def obj():
        o = sum((filled[b] / total - targets[b]) ** 2 for b in targets)
        o += lam * sum(((filled_pos[b] / max(filled[b], 1)) - overall_rate) ** 2
                       for b in targets)
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

    improved, n_iter = True, 0
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cap", type=int, default=PER_OUTLET_CAP)
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.pop)
    om = pd.read_csv(a.map)[["snapshot_id", "outlet", "top_share", "margin"]]
    df = df.merge(om, on="snapshot_id", how="left")
    n0 = len(df)
    df = df[df["outlet"].notna() & (df["outlet"] != "ambiguous")].copy()
    print(f"joined outlet: kept {len(df):,} / {n0:,} "
          f"({(n0-len(df))/n0:.3%} ambiguous or unmapped)")
    print(df.groupby("outlet")["judgement"].agg(["size", "mean"]).to_string())

    df["row_id"] = [sha1(f"{s}|{t}")[:20] for s, t in
                    zip(df["snapshot_id"], df["text"])]
    dup = int(df["row_id"].duplicated().sum())
    if dup:
        print(f"WARNING: {dup} duplicate row_ids -- deduplicated, first kept")
        df = df[~df["row_id"].duplicated()].reset_index(drop=True)

    parts = []
    for outlet, sub in df.groupby("outlet"):
        order = sorted(sub["snapshot_id"].unique(), key=lambda s: sha256(SALT + str(s)))
        sizes = sub.groupby("snapshot_id").size().to_dict()
        take, n = [], 0
        for s in order:
            take.append(s)
            n += sizes[s]
            if n >= a.cap:
                break
        parts.append(sub[sub["snapshot_id"].isin(set(take))])
    pop = pd.concat(parts).sort_values(["outlet", "snapshot_id", "row_id"],
                                       kind="mergesort").reset_index(drop=True)
    pop["group"] = pop["outlet"]
    print(f"population n={len(pop):,} outlets={pop['outlet'].nunique()} "
          f"snapshots={pop['snapshot_id'].nunique():,} pos={pop['judgement'].mean():.4f}")
    print(pop.groupby("outlet")["judgement"].agg(["size", "mean"]).to_string())

    cols = ["row_id", "snapshot_id", "outlet", "group", "text", "judgement"]
    pop[cols].to_csv(out / "population.csv.gz", index=False)

    dense = out / "dense_standard"
    (dense / "split").mkdir(parents=True, exist_ok=True)
    out_rows = [{"text": r.text, "judgement": int(r.judgement), "group": r.group,
                 "row_id": r.row_id} for r in pop.itertuples()]
    y_by_group = defaultdict(list)
    for r in out_rows:
        y_by_group[r["group"]].append(r["judgement"])
    bmap = stable_hash_bucket_map(y_by_group)
    by_split = {"train": [], "eval": [], "test": []}
    for r in out_rows:
        by_split[bmap[r["group"]]].append(r)
    dcols = ["text", "judgement", "group", "row_id"]
    with open(dense / "data.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=dcols); w.writeheader(); w.writerows(out_rows)
    for s in ("train", "eval", "test"):
        with open(dense / "split" / f"{s}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=dcols); w.writeheader(); w.writerows(by_split[s])

    nn = len(out_rows)
    man = {
        "cell": "homepage_curation",
        "title": "journalism homepage curation (spatial placement, outlet-held-out)",
        "source": a.pop, "outlet_map": a.map,
        "sampling": f'per outlet, whole snapshots in sha256("{SALT}" + snapshot_id) '
                    f'order until >= {a.cap} rows',
        "n": nn, "pos_rate": sum(r["judgement"] for r in out_rows) / nn,
        "n_groups": len(set(r["group"] for r in out_rows)),
        "group_column": "outlet (OUTLET-HELD-OUT)",
        "secondary_group_column": "snapshot_id",
        "y_definition": ("1 = link rendered in the TOP half of the homepage capture's "
                         "top-30% zone, 0 = bottom half of that zone"),
        "weak_instrument_flag": (
            "y is spatial placement, jointly determined with layout/ad/image "
            "constraints, not a clean editorial preference; outlet identity alone "
            "predicts it strongly (BBC ~34% top vs WaPo ~73% top). Every number from "
            "this cell carries this flag."),
        "outlet_split": {s: sorted(set(r["group"] for r in by_split[s])) for s in by_split},
        "split_row_counts": {s: len(by_split[s]) for s in by_split},
        "split_pos_rates": {s: sum(r["judgement"] for r in by_split[s]) / max(len(by_split[s]), 1)
                            for s in by_split},
        "split_fractions": {s: len(by_split[s]) / nn for s in by_split},
        "recipe": ("Llama-3.1-8B LoRA r16/a32, lr5e-5, batch16, max_len1024, 2 epochs, "
                   "gradient-checkpointing, select-on-eval (dense-standard, no deviation)"),
    }
    (dense / "manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps(man, indent=2))
    print("HOMEPAGE_VA_POPULATION_DONE")


if __name__ == "__main__":
    main()
