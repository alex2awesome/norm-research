"""Phase 0.1 — extract the articulation text of every (conditioned) rescue.

Walks the interim family score grids, recomputes the gap-conditioned rescue verdict per cell,
and joins each cell's best substantive arm back to its articulation text in the arm bank.
Non-rescued cells are emitted too (rescued=false) so the lexicon probe has a contrast set.

CPU-only; reads local npz + the arm bank. Run from the repo root.

Usage:
  python -m methods.tacit_channels.channels.frontier_probe.extract_rescue_articulations \
      --bank notebooks/data/two_faces_20260702/tacit_breadth_arm_bank_v3.json \
      --base notebooks/data/two_faces_20260702 \
      --out outputs/tacit_channels/frontier_probe/rescue_articulations.jsonl
"""
from __future__ import annotations

import argparse
import os

from methods.tacit_channels.channels.common import (
    DOMAINS, FAMILIES, arm_form_prompt, cell_stats, is_conditioned_rescue,
    load_grid, parse_bank_cells, write_jsonl,
)


def extract(bank_path: str, base: str, families: dict | None = None,
            domains=DOMAINS) -> list[dict]:
    cells = parse_bank_cells(bank_path)
    rows = []
    for family, (subdir, target_job) in (families or FAMILIES).items():
        root = os.path.join(base, subdir)
        if not os.path.isdir(root):
            continue
        executor_jobs = sorted(
            j for j in os.listdir(root)
            if j.endswith("_executor") and os.path.isdir(os.path.join(root, j)))
        for job in executor_jobs:
            for domain in domains:
                tgt, _ = load_grid(root, target_job, domain)
                exe, emeta = load_grid(root, job, domain)
                if not tgt or not exe:
                    continue
                for cell_id in sorted({c for (c, a, f) in tgt}):
                    stats = cell_stats(tgt, exe, emeta, cell_id)
                    if stats is None or stats["best_arm"] is None:
                        continue
                    bank_cell = cells.get(cell_id)
                    text = (arm_form_prompt(bank_cell, stats["best_arm"], "canonical")
                            if bank_cell else None)
                    name_text = (arm_form_prompt(bank_cell, "name", "canonical")
                                 if bank_cell else None)
                    rows.append({
                        "family": family, "executor_job": job, "domain": domain,
                        **stats,
                        "rescued": is_conditioned_rescue(stats),
                        "construct": (bank_cell or {}).get("construct"),
                        "level": (bank_cell or {}).get("level"),
                        "construct_name_text": name_text,
                        "articulation_text": text,
                    })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--base", default="notebooks/data/two_faces_20260702")
    ap.add_argument("--out", required=True)
    ap.add_argument("--domains", default=",".join(DOMAINS))
    args = ap.parse_args()
    rows = extract(args.bank, args.base, domains=tuple(args.domains.split(",")))
    n = write_jsonl(args.out, rows)
    rescued = sum(1 for r in rows if r["rescued"])
    print(f"wrote {n} rows ({rescued} conditioned rescues) -> {args.out}")


if __name__ == "__main__":
    main()
