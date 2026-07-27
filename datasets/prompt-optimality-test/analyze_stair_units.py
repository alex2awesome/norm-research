"""Retained-unit analysis for the M_ω staircase (envelope H-i/H-ii: absorption & specificity).

Which frozen-pool units survive into each scale's winning candidate? Stair proposals rows carry
only hashes+scores, so retention is recovered by TEXT CONTAINMENT: a pool unit counts as retained
iff its text appears verbatim in the shipped candidate's instructions (composition is literal
concatenation, so exact containment is faithful; whitespace-normalized to be safe).

Usage: python analyze_stair_units.py <bench> [--pool pools/<bench>_Qwen3-8B_frozen.json]
Scans runs_paperexact/<bench>/*/unitrecomb_stair/result.json (local mirror).
"""
import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).parent


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--pool", default=None)
    a = ap.parse_args()

    pool_path = Path(a.pool) if a.pool else HERE / "pools" / f"{a.bench}_Qwen3-8B_frozen.json"
    pool = json.loads(pool_path.read_text())
    units = pool["units"] if isinstance(pool, dict) and "units" in pool else pool
    texts = [u["unit"] if isinstance(u, dict) else str(u) for u in units]

    print(f"pool: {pool_path.name} ({len(texts)} units)")
    print(f"{'scale':12} {'seed':>6} {'best':>6} {'retained':>8} {'frac':>6} "
          f"{'cand_chars':>10} {'mean_unit_chars':>15}")
    for rj in sorted(HERE.glob(f"runs_paperexact/{a.bench}/*/unitrecomb_stair/result.json")):
        r = json.loads(rj.read_text())
        cand = r.get("best_candidate", {})
        blob = norm(" ".join(cand.values())) if isinstance(cand, dict) else norm(str(cand))
        # Net-of-init: the pool is mined from 8B trajectories, so GEPA's own winner (the stair's
        # init) trivially "contains" its own fragments — only units ABSENT from init count as
        # selected. Requires the same scale's official/result.json alongside.
        init_blob = ""
        off_rj = rj.parent.parent / "official" / "result.json"
        if off_rj.exists():
            oc = json.loads(off_rj.read_text()).get("best_candidate", {})
            init_blob = norm(" ".join(oc.values())) if isinstance(oc, dict) else norm(str(oc))
        kept = [t for t in texts if norm(t) in blob]
        net = [t for t in kept if norm(t) not in init_blob]
        identical = bool(init_blob) and blob == init_blob
        scale = rj.parent.parent.name
        mean_len = sum(len(t) for t in net) / len(net) if net else 0
        print(f"{scale:12} {r.get('seed_test'):>6} {r.get('best_test'):>6} "
              f"{len(net):>8} {len(net)/len(texts):>6.2f} {len(blob):>10} {mean_len:>15.0f}"
              f"{'  [== init: guard fell back, deltas = re-measurement noise]' if identical else ''}"
              f"{'' if init_blob else '  [no official result: GROSS count, not net]'}")


if __name__ == "__main__":
    main()
