#!/usr/bin/env python3
"""Rebuild ONLY the homepage dense-standard split so it satisfies the frozen
trainer's 80/10/10 ratio check, without touching the scored A/V population.

Why this is needed: `methods/dense/train_reward_model.py::get_or_create_fixed_split`
hard-requires train/eval/test fractions within +-2 percentage points of
80/10/10. The homepage cell's grouping unit is the OUTLET and there are only 8 of
them at ~1,700 rows each, so any whole-outlet 6/1/1 assignment lands at roughly
.74/.13/.13 and is rejected (observed .7370/.1317/.1313).

Fix: keep the outlet-held-out design and keep every row inside the already-scored
12,998-row population; simply take a SUBSET of it for the dense arm, sized so the
ratios come out right.

  eval  = one held-out outlet, trimmed to E rows
  test  = a second held-out outlet, trimmed to E rows
  train = the remaining six outlets, trimmed to 8E rows in total

The two held-out outlets are chosen deterministically by
sha256("homepage-va-heldout|" + outlet) among the seven outlets that have enough
rows; WSJ is excluded from being held out because only 61 of its snapshots
resolve (paywalled/JS-rendered captures) and it is the thinnest fold. Trimming
removes WHOLE SNAPSHOTS in stable-hash order, never individual rows, so the
snapshot container is never split.

Every dense row is a scored row, so T's eval rows remain a subset of the A/V
population (FREEZE CHANGE 2 satisfied).

Usage (CPU only, run on sk3):
  python3 fix_dense_split.py --pop datasets/news-homepages/va/population.csv.gz \\
      --outdir datasets/news-homepages/va/dense_standard
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

SALT_SNAP = "homepage-va-v1|"
SALT_HELD = "homepage-va-heldout|"
EXCLUDE_FROM_HELDOUT = {"wsj"}


def sha256(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8")).hexdigest()


def take_snapshots(sub: pd.DataFrame, target: int) -> pd.DataFrame:
    """Whole snapshots in stable-hash order until >= target rows."""
    order = sorted(sub["snapshot_id"].astype(str).unique(),
                   key=lambda s: sha256(SALT_SNAP + s))
    sizes = sub.groupby(sub["snapshot_id"].astype(str)).size().to_dict()
    take, n = [], 0
    for s in order:
        if n >= target:
            break
        take.append(s)
        n += sizes[s]
    return sub[sub["snapshot_id"].astype(str).isin(set(take))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    out = Path(a.outdir)
    (out / "split").mkdir(parents=True, exist_ok=True)

    pop = pd.read_csv(a.pop)
    pop["snapshot_id"] = pop["snapshot_id"].astype(str)
    counts = pop.groupby("outlet").size().to_dict()
    eligible = sorted([o for o in counts if o not in EXCLUDE_FROM_HELDOUT],
                      key=lambda o: sha256(SALT_HELD + o))
    eval_outlet, test_outlet = eligible[0], eligible[1]
    train_outlets = [o for o in counts if o not in (eval_outlet, test_outlet)]
    train_capacity = sum(counts[o] for o in train_outlets)

    # 8E <= train_capacity and E <= min(held-out outlet sizes)
    E = min(train_capacity // 8, counts[eval_outlet], counts[test_outlet])
    per_train = E * 8 // len(train_outlets)
    print(f"held-out: eval={eval_outlet} ({counts[eval_outlet]}), "
          f"test={test_outlet} ({counts[test_outlet]}); train outlets={sorted(train_outlets)} "
          f"capacity={train_capacity}; E={E}, per-train-outlet target={per_train}")

    parts = {"eval": take_snapshots(pop[pop.outlet == eval_outlet], E),
             "test": take_snapshots(pop[pop.outlet == test_outlet], E)}
    tr = []
    # small outlets contribute everything; the shortfall is spread over the rest
    small = [o for o in train_outlets if counts[o] <= per_train]
    big = [o for o in train_outlets if counts[o] > per_train]
    got_small = sum(counts[o] for o in small)
    need_big = max(E * 8 - got_small, 0)
    per_big = need_big // max(len(big), 1)
    for o in train_outlets:
        target = counts[o] if o in small else per_big
        tr.append(take_snapshots(pop[pop.outlet == o], target))
    parts["train"] = pd.concat(tr)

    total = sum(len(v) for v in parts.values())
    fr = {k: len(v) / total for k, v in parts.items()}
    print("row counts:", {k: len(v) for k, v in parts.items()},
          "fractions:", {k: round(v, 4) for k, v in fr.items()})
    assert abs(fr["train"] - .80) <= .02 and abs(fr["eval"] - .10) <= .02 \
        and abs(fr["test"] - .10) <= .02, f"ratios still out of tolerance: {fr}"

    cols = ["text", "judgement", "group", "row_id"]
    allrows = pd.concat([parts["train"], parts["eval"], parts["test"]])
    allrows = allrows.rename(columns={"outlet": "_o"})
    allrows["group"] = allrows["_o"]
    with open(out / "data.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        w.writerows(allrows[cols].to_dict("records"))
    for s in ("train", "eval", "test"):
        d = parts[s].rename(columns={"outlet": "_o"})
        d["group"] = d["_o"]
        with open(out / "split" / f"{s}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
            w.writerows(d[cols].to_dict("records"))

    man = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").exists() else {}
    man.update({
        "split_rebuilt": ("outlet-held-out subset of the scored population, sized to the "
                          "trainer's 80/10/10 +-2pp requirement; whole snapshots only"),
        "eval_outlet": eval_outlet, "test_outlet": test_outlet,
        "train_outlets": sorted(train_outlets),
        "heldout_excluded": sorted(EXCLUDE_FROM_HELDOUT),
        "n_dense": int(total),
        "split_row_counts": {k: int(len(v)) for k, v in parts.items()},
        "split_fractions": {k: float(v) for k, v in fr.items()},
        "split_pos_rates": {k: float(v["judgement"].mean()) for k, v in parts.items()},
        "dense_rows_subset_of_scored_population": True,
    })
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps({k: man[k] for k in ("eval_outlet", "test_outlet", "n_dense",
                                          "split_row_counts", "split_fractions",
                                          "split_pos_rates")}, indent=2))
    print("HOMEPAGE_SPLIT_FIXED")


if __name__ == "__main__":
    main()
