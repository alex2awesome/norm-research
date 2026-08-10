#!/usr/bin/env python3
"""Freeze the reddit-jokes COMMUNITY cell population (label-blind sampling) and
build the dense-standard CSVs on exactly that population.

Cell: humor x community/crowd in the VAT grid -- r/Jokes posts, y = crowd
upvote verdict (top 25% vs bottom 25% inside a length_bin x format x topic
stratum, mid 50% dropped; see datasets/humor/build_reddit_humor_dataset.py).
The registry currently carries only the May-2026 floor-harness numbers
(V .574 / VA .564 dagger); this build supplies the mature-bank population that
the Gemma-4-31B A bank and the dense standard are both run on.

Source (canonical, identical on sk3 as reddit_humor_modeling_with_topics.csv.gz):
  datasets/humor/reddit_humor_with_topics.csv.gz  -- 383,786 rows, columns
  text / judgement / topic, verified row-for-row identical to the canonical
  reddit_humor_modeling_dedup.csv.gz plus the LDA topic column.

Sampling: STABLE HASH, never a seeded shuffle.
  row_id = sha1(text)[:20]  (verified unique over all 383,786 rows)
  order  = sha256("jokes-va-v1|" + row_id)
  take   = first N_SAMPLE rows of that order.
A uniform stable-hash prefix is used rather than whole-group draws because the
grouping unit here (LDA topic, 50 of them) is a stratification variable of the
labeller, not a container: every topic must stay represented for the grouped
readout to have folds. Pos-rate is ~.50 inside every topic by construction
(range .422-.566 over the 50 topics), so group identity carries almost no label
information -- the grouped split is a lexical-domain control, not a leak fix.

Dense standard: same population, topic-grouped 80/10/10 via the frozen
stable_hash_bucket_map bin-packer (row-count AND pos-rate balanced), ported
verbatim from datasets/humor/hashtagwars/build_dense_standard.py.

Usage (CPU only):
  python3 datasets/humor/reddit_jokes/build_population.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SRC = REPO / "datasets/humor/reddit_humor_with_topics.csv.gz"
SALT = "jokes-va-v1|"
N_SAMPLE = 16000


def sha256(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8")).hexdigest()


def sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()


def stable_hash_bucket_map(y_by_group: dict, targets=None, lam: float = 2.5) -> dict:
    """Verbatim from datasets/humor/hashtagwars/build_dense_standard.py --
    deterministic greedy + hill-climb bin-packing of groups into train/eval/test
    targeting 80/10/10 BY ROW COUNT *AND* matched per-bucket pos-rate."""
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


def load_population(n_sample: int = N_SAMPLE) -> pd.DataFrame:
    df = pd.read_csv(SRC)
    df["row_id"] = [sha1(t)[:20] for t in df["text"]]
    assert not df["row_id"].duplicated().any(), "row_id collision -- sampling unsafe"
    df["group"] = ["t%02d" % t for t in df["topic"]]
    df["_ord"] = [sha256(SALT + r) for r in df["row_id"]]
    df = df.sort_values("_ord", kind="mergesort").head(n_sample).drop(columns="_ord")
    return df.sort_values("row_id", kind="mergesort").reset_index(drop=True)


def main():
    df = load_population()
    n = len(df)
    pos_rate = float(df["judgement"].mean())
    n_groups = int(df["group"].nunique())
    print(f"n={n} pos_rate={pos_rate!r} n_groups={n_groups}")
    print(f"chars: median={df['text'].str.len().median():.0f} "
          f"p95={df['text'].str.len().quantile(.95):.0f} max={df['text'].str.len().max()}")

    outdir = HERE / "va"
    outdir.mkdir(parents=True, exist_ok=True)
    df[["row_id", "group", "topic", "text", "judgement"]].to_csv(
        outdir / "population.csv.gz", index=False)

    # ---- dense standard on EXACTLY this population --------------------------
    dense = HERE / "dense_standard"
    (dense / "split").mkdir(parents=True, exist_ok=True)
    out_rows = [{"text": r.text, "judgement": int(r.judgement), "group": r.group,
                 "row_id": r.row_id} for r in df.itertuples()]
    y_by_group = defaultdict(list)
    for r in out_rows:
        y_by_group[r["group"]].append(r["judgement"])
    bmap = stable_hash_bucket_map(y_by_group)
    by_split = {"train": [], "eval": [], "test": []}
    for r in out_rows:
        by_split[bmap[r["group"]]].append(r)

    cols = ["text", "judgement", "group", "row_id"]
    with open(dense / "data.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(out_rows)
    for split in ("train", "eval", "test"):
        with open(dense / "split" / f"{split}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
            w.writerows(by_split[split])

    manifest = {
        "cell": "jokes_community",
        "title": "reddit-jokes community (r/Jokes crowd upvotes)",
        "source": str(SRC.relative_to(REPO)),
        "sampling": f'stable hash: first {N_SAMPLE} rows under sha256("{SALT}" + sha1(text)[:20])',
        "n": n, "pos_rate": pos_rate, "n_groups": n_groups,
        "group_column": "LDA topic (50)",
        "y_definition": ("1 = top 25% by score inside its (length_bin x format x topic) "
                         "stratum, 0 = bottom 25%; middle 50% dropped upstream "
                         "(build_reddit_humor_dataset.py)"),
        "split_group_counts": {s: len(set(r["group"] for r in by_split[s])) for s in by_split},
        "split_row_counts": {s: len(by_split[s]) for s in by_split},
        "split_pos_rates": {s: (sum(r["judgement"] for r in by_split[s])
                                / max(len(by_split[s]), 1)) for s in by_split},
        "split_fractions": {s: len(by_split[s]) / n for s in by_split},
        "recipe": ("Llama-3.1-8B LoRA r16/a32, lr5e-5, batch16, max_len1024, 2 epochs, "
                   "gradient-checkpointing, select-on-eval (dense-standard, no deviation)"),
    }
    (dense / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print("BUILD_DONE")


if __name__ == "__main__":
    main()
