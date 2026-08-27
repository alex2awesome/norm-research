"""EXP-GTK-1 builder — deterministic design manifest + the three training datasets.

Implements the FROZEN design of notes/2026-07-22__exp-gtk-1-prereg.md (sha256 recorded in the
design manifest). Selection is fully deterministic: constructs ordered by sha256(cell_id),
stratified by level; items split by stable hash (salt `exp_gtk1`).

Outputs under --out-dir:
  design_manifest.json                     (A/B sets, arms, hashes — the frozen design)
  dataset_real.jsonl                       (target's true policies, item-half-1)
  dataset_shuffled.jsonl                   (item-shuffled labels; seed from design)
  dataset_construct_permuted.jsonl         (true policies, deranged construct assignment)

CPU-only. Run from repo root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from methods.tacit_channels.channels.common import read_jsonl, stable_split, write_jsonl

PREREG = "notes/2026-07-22__exp-gtk-1-prereg.md"
SALT = "exp_gtk1"
EXECUTOR_JOB = "qwen25_7b_executor"
DOMAIN = "humor"
N_A, N_B1, N_B5 = 16, 20, 4


def _ordered(cells: list[dict]) -> list[dict]:
    return sorted(cells, key=lambda r: hashlib.sha256(r["cell_id"].encode()).hexdigest())


def _stratified_take(cells: list[dict], n: int) -> list[dict]:
    """Proportional-by-level deterministic take from sha-ordered cells."""
    by_level = defaultdict(list)
    for c in _ordered(cells):
        by_level[c.get("level") or "?"].append(c)
    total = len(cells)
    take, remainder = [], []
    for level in sorted(by_level):
        k = round(n * len(by_level[level]) / total)
        take.extend(by_level[level][:k])
        remainder.extend(by_level[level][k:])
    for c in _ordered(remainder):  # fill/trim to exactly n
        if len(take) >= n:
            break
        take.append(c)
    return take[:n]


def select_sets(rescues_path: str) -> dict:
    rows = [r for r in read_jsonl(rescues_path)
            if r["executor_job"] == EXECUTOR_JOB and r["domain"] == DOMAIN]
    if not rows:
        raise SystemExit(f"no rows for {EXECUTOR_JOB}/{DOMAIN} in {rescues_path}")
    failure = [r for r in rows if (r.get("gap") or 0) > .10
               and (r.get("best_rho") is None or r["best_rho"] < .70)]
    success = [r for r in rows if r.get("best_rho") is not None and r["best_rho"] >= .70]
    a_set = _stratified_take(failure, N_A)
    a_ids = {r["cell_id"] for r in a_set}
    rest = [r for r in failure if r["cell_id"] not in a_ids]
    b1 = _stratified_take(rest, N_B1)
    b1_ids = {r["cell_id"] for r in b1}
    rest2 = [r for r in rest if r["cell_id"] not in b1_ids]
    b5 = _stratified_take(rest2, N_B5)
    return {
        "A": sorted(a_ids),
        "B1": sorted(b1_ids),
        "B2_success": sorted(r["cell_id"] for r in success),
        "B5": sorted(r["cell_id"] for r in b5),
        "n_failure_set": len(failure), "n_success_set": len(success),
    }


def derive_arm_datasets(real_rows: list[dict], a_cells: list[str], seed: int = 20260722):
    """shuffled: permute labels within construct; construct-permuted: derange the mapping
    cell->label-vector (labels stay REAL target policies, attached to the wrong construct)."""
    rng = random.Random(seed)
    by_cell = defaultdict(list)
    for r in real_rows:
        by_cell[r["cell_id"]].append(r)

    shuffled = []
    for cell, rows in sorted(by_cell.items()):
        labels = [r["p_yes"] for r in rows]
        rng.shuffle(labels)
        shuffled.extend({**r, "p_yes": lab, "arm": "shuffled"}
                        for r, lab in zip(rows, labels))

    cells = sorted(by_cell)
    perm = cells[:]
    while True:  # derangement
        rng.shuffle(perm)
        if all(a != b for a, b in zip(cells, perm)):
            break
    mapping = dict(zip(cells, perm))
    permuted = []
    for cell, rows in sorted(by_cell.items()):
        donor_rows = {r["item_sha256"]: r["p_yes"] for r in by_cell[mapping[cell]]}
        for r in rows:
            if r["item_sha256"] in donor_rows:
                permuted.append({**r, "p_yes": donor_rows[r["item_sha256"]],
                                 "arm": "construct_permuted",
                                 "label_donor_cell": mapping[cell]})
    return shuffled, permuted, mapping


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescues", required=True)
    ap.add_argument("--real-dataset", required=True,
                    help="output of build_policy_distill_dataset for the A cells "
                         "(item-half-1 rows only are used)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    sets = select_sets(args.rescues)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # keep only A-cell, item-half-1 rows for training (item-disjointness rule)
    real_all = read_jsonl(args.real_dataset)
    a_ids = set(sets["A"])
    real_rows = [r for r in real_all if r["cell_id"] in a_ids
                 and stable_split(r["item_sha256"], 0.5, salt=SALT) == "train"]
    if not real_rows:
        raise SystemExit("no item-half-1 rows for the A cells — check the real dataset")
    for r in real_rows:
        r["arm"] = "real"
    shuffled, permuted, mapping = derive_arm_datasets(real_rows, sets["A"])

    write_jsonl(str(out / "dataset_real.jsonl"), real_rows)
    write_jsonl(str(out / "dataset_shuffled.jsonl"), shuffled)
    write_jsonl(str(out / "dataset_construct_permuted.jsonl"), permuted)

    prereg_sha = hashlib.sha256(Path(PREREG).read_bytes()).hexdigest() \
        if Path(PREREG).exists() else None
    manifest = {
        "schema": "exp_gtk1_design/v1", "prereg": PREREG, "prereg_sha256": prereg_sha,
        "executor_job": EXECUTOR_JOB, "domain": DOMAIN, "item_salt": SALT,
        **sets, "construct_permutation": mapping,
        "n_train_rows": len(real_rows),
        "datasets_sha256": {
            name: hashlib.sha256((out / name).read_bytes()).hexdigest()
            for name in ("dataset_real.jsonl", "dataset_shuffled.jsonl",
                         "dataset_construct_permuted.jsonl")},
    }
    (out / "design_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: (len(v) if isinstance(v, list) else v)
                      for k, v in sets.items()}, indent=2))
    print(f"design + 3 datasets -> {out} ({len(real_rows)} train rows/arm)")


if __name__ == "__main__":
    main()
