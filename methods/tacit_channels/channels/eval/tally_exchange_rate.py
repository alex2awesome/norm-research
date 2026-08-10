"""The 2-D exchange-rate tally: adverse-rho over (articulation dose x intervention dose).

Reads (a) the target job's frozen npz grids, (b) one or more executor grids per intervention
tag (base / lora_n8 / lora_n32 / ...; produced by score_with_adapter or the frozen scorer),
and emits the surface: per (cell, arm, intervention) adverse-rho, plus iso-rho contours —
for each rho level, the cheapest articulation arm (by added content words) and the smallest
N reaching it.

CPU-only. Interim/point-estimate readout — same caveats as the family ladder tallies.

Layout convention:
  --exec-root/
      base/grid_<domain>_*.npz            (intervention tag = dir name)
      lora_n8/grid_<domain>_*.npz
      lora_n32/...

Usage:
  python -m methods.tacit_channels.channels.eval.tally_exchange_rate \
      --target-root notebooks/data/two_faces_20260702/family_scores_qwen25 \
      --target-job qwen25_72b_name_target --exec-root outputs/tacit_channels/family_scores/q7b \
      --domain humor --out outputs/tacit_channels/exchange_rate_q7b_humor.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from methods.tacit_channels.channels.common import (
    load_grid, spearman, write_jsonl,
)


def n_from_tag(tag: str) -> int:
    """Intervention dose: lora_n<k> -> k; base -> 0."""
    if tag.startswith("lora_n"):
        try:
            return int(tag.split("lora_n")[1])
        except ValueError:
            return -1
    return 0


def tally_surface(target_root: str, target_job: str, exec_root: str, domain: str):
    tgt, _tmeta = load_grid(target_root, target_job, domain)
    if not tgt:
        raise SystemExit(f"no target grids for {domain} under {target_root}/{target_job}")
    tags = sorted(d for d in os.listdir(exec_root)
                  if os.path.isdir(os.path.join(exec_root, d)))
    rows = []
    for tag in tags:
        exe, emeta = load_grid(exec_root, tag, domain)
        if not exe:
            continue
        for cell in sorted({c for (c, a, f) in tgt}):
            t_forms = [v for (c, a, f), v in tgt.items() if c == cell and a == "name"]
            if not t_forms:
                continue
            t = np.mean(t_forms, axis=0)
            arms = {a for (c, a, f) in exe if c == cell}
            for arm in sorted(arms):
                forms = [(f, v) for (c, a, f), v in exe.items() if c == cell and a == arm]
                if not forms:
                    continue
                adverse = min(spearman(v, t) for _f, v in forms)
                any_meta = emeta[(cell, arm, forms[0][0])]
                rows.append({
                    "domain": domain, "cell_id": cell, "intervention": tag,
                    "n_examples": n_from_tag(tag), "arm_id": arm,
                    "added_words": any_meta.get("added_content_word_count") or 0,
                    "is_control": any_meta.get("control_for") is not None,
                    "adverse_rho": None if np.isnan(adverse) else round(float(adverse), 4),
                })
    return rows


def iso_rho_contours(rows: list[dict], levels=(0.5, 0.6, 0.7, 0.8)) -> list[dict]:
    """Per cell x rho-level: cheapest articulation-only dose vs smallest intervention-only N."""
    by_cell = defaultdict(list)
    for r in rows:
        if not r["is_control"] and r["adverse_rho"] is not None:
            by_cell[r["cell_id"]].append(r)
    contours = []
    for cell, cell_rows in sorted(by_cell.items()):
        for level in levels:
            # articulation axis: intervention == base (N=0), any arm
            art = [r["added_words"] for r in cell_rows
                   if r["n_examples"] == 0 and r["adverse_rho"] >= level]
            # intervention axis: name arm only (no articulation), any N
            inter = [r["n_examples"] for r in cell_rows
                     if r["arm_id"] == "name" and r["n_examples"] > 0
                     and r["adverse_rho"] >= level]
            contours.append({
                "cell_id": cell, "rho_level": level,
                "min_articulation_words": min(art) if art else None,
                "min_intervention_n": min(inter) if inter else None,
            })
    return contours


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-root", required=True)
    ap.add_argument("--target-job", required=True)
    ap.add_argument("--exec-root", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = tally_surface(args.target_root, args.target_job, args.exec_root, args.domain)
    contours = iso_rho_contours(rows)
    write_jsonl(args.out, rows)
    write_jsonl(args.out.replace(".jsonl", "_contours.jsonl"), contours)

    print(f"{len(rows)} surface rows; contour summary:")
    reachable = defaultdict(lambda: {"articulation": 0, "intervention": 0, "both": 0, "n": 0})
    for c in contours:
        b = reachable[c["rho_level"]]
        b["n"] += 1
        a_ok = c["min_articulation_words"] is not None
        i_ok = c["min_intervention_n"] is not None
        if a_ok and i_ok:
            b["both"] += 1
        elif a_ok:
            b["articulation"] += 1
        elif i_ok:
            b["intervention"] += 1
    print(json.dumps(reachable, indent=2, default=int))


if __name__ == "__main__":
    main()
