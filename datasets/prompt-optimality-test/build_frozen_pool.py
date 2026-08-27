"""Build the FROZEN unit pool for the OSL staircase (design v2, 2026-07-22).

The frozen pool = the union of every unit the 8B unitrecomb runs ACTUALLY evaluated (each
result.json's units.marginals lists the full pool with paired deltas), deduped at tier-1
(module + normalized string). Deterministic — no re-mining, no LLM calls — and it contains
every past winner by construction (the hover-v5 lesson). INVALID_* run dirs are excluded.

Usage: python3 build_frozen_pool.py <bench> [--lm Qwen3-8B] -> pools/<bench>_<lm>_frozen.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--lm", default="Qwen3-8B")
    a = ap.parse_args()

    lm_dir = HERE / "runs_paperexact" / a.bench / a.lm
    units, seen, provenance = [], set(), []
    for rdir in sorted(lm_dir.glob("unitrecomb*")):
        res = rdir / "result.json"
        if not res.exists():
            continue
        r = json.loads(res.read_text())
        u = r.get("units") or {}
        kept = {(cu["module"], " ".join(cu["unit"].lower().split()))
                for cu in u.get("compiled_units") or []}
        n_new = 0
        for m in u.get("marginals") or []:
            key = (m["module"], " ".join(m["unit"].lower().split()))
            if key in seen:
                continue
            seen.add(key)
            n_new += 1
            units.append({"module": m["module"], "unit": m["unit"],
                          "source": m.get("source", "trajectory"),
                          "from_run": rdir.name,
                          "delta_8b": m.get("delta"),
                          "won_8b": key in kept})
        provenance.append({"run": rdir.name, "n_new_units": n_new,
                           "best_test": r.get("best_test")})
    out = HERE / "pools"
    out.mkdir(exist_ok=True)
    path = out / f"{a.bench}_{a.lm}_frozen.json"
    path.write_text(json.dumps({"bench": a.bench, "lm": a.lm, "n_units": len(units),
                                "provenance": provenance, "units": units}, indent=1))
    n_win = sum(1 for u in units if u["won_8b"])
    print(f"{path}  n_units={len(units)}  past-winners={n_win}  runs={len(provenance)}")


if __name__ == "__main__":
    main()
