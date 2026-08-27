"""THE canonical upper-bounds rollup — one file to answer "where are our upper bounds?".

Run: python3 analyze_upper_bounds_rollup.py   ->  runs/UPPER_BOUNDS.md (+ .json)

Collects, per (bench x task-LM) with a rescore pool: best shipped / pool max / union oracle
(computed fresh from the pools), the EVT endpoint (from bounds_evt_summary*.json if present),
and the missing-value projected ceiling (from unit_missing_mass_summary.json, panel-scale).
Certificate strengths are printed in the header so the table can't be over-read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RUNS = HERE / "runs_paperexact"
BENCHES = ("aime", "hover", "hotpot", "ifbench", "livebench", "pupa")
LMS = ("Qwen3-8B", "glm-5.2")

HEADER = """# UPPER BOUNDS — canonical rollup (regenerate: python3 analyze_upper_bounds_rollup.py)

HEADLINE BOUNDS (user framing 2026-07-21: the sampling-theoretic pair — two levels of one
species-sampling question, both process-conditional, both tied to the OSL/unseen-value machinery):
- **EVT endpoint** (PROMPT level): estimated upper endpoint of the search's own draw
  distribution = what infinite search from THIS process achieves. Estimate + CI, never a
  certificate (adaptive search breaks i.i.d.); "unstable" = the two estimators disagree beyond
  the noise floor — quote the interval, not a point. Undershoots the all-prompt truth (gestalt).
- **missing-value ceiling** (UNIT level): value of the unit generator at Chao-projected species
  richness (Good-Turing missing mass x value-saturation replay). PANEL scale; extrapolation
  banned from certificates.

DIAGNOSTICS (not headline bounds):
- **pool max**: best OBSERVED single prompt; (pool max - shipped) = measured selection headroom.
- **union oracle**: per-item-oracle router ceiling, pool-certified but noise-inflated; coverage
  sanity check only.
- **certified all-prompt**: 1.0 — PROVABLY VACUOUS on deterministic-label tasks (theorem, not a
  gap); the non-trivial certified caps live in the noisy-label (norm-research) tasks.
"""


def arm_dirs(bench, lm):
    """official/inhouse plus every unitrecomb variant (unitrecomb, unitrecomb_v5sk2, ...)."""
    base = RUNS / bench / lm
    out = [base / "official", base / "inhouse"]
    out += sorted(d for d in base.glob("unitrecomb*") if d.is_dir())
    return out


def pool(bench, lm):
    rows, seen = [], set()
    for d in arm_dirs(bench, lm):
        p = d / "rescore.jsonl"
        if p.exists():
            for line in open(p):
                r = json.loads(line)
                if r["hash"] not in seen:
                    seen.add(r["hash"])
                    rows.append(r["item_scores"])
    return np.array(rows) if rows else None


def shipped(bench, lm):
    out = {}
    for d in arm_dirs(bench, lm):
        p = d / "result.json"
        if p.exists():
            r = json.loads(p.read_text())
            out[d.name] = (r.get("seed_test"), r.get("best_test"), r.get("regression_flag"))
    return out


def evt_lookup():
    out = {}
    for p in (HERE / "runs").glob("bounds_evt_summary_paperexact_*.json"):
        data = json.loads(p.read_text())
        for d in data.get("datasets", []):
            tag = d["dataset"].split(" ")[0]          # "hover:Qwen3-8B"
            bench, lm = tag.split(":")
            gpd = d["endpoint_point_kmedian"]["gpd"]
            ci = d["endpoint_boot_ci_2p5_97p5"]["gpd"]
            stable = d.get("estimators_agree_within_noise", False)
            out[(bench, lm)] = (gpd, ci, stable)
    return out


def mm_lookup():
    p = HERE / "runs" / "unit_missing_mass_summary.json"
    out = {}
    if p.exists():
        for r in json.loads(p.read_text()).get("results", []):
            if r["pool"].startswith("paperexact/"):
                _, bench, lm = r["pool"].split("/")
                out[(bench, lm)] = r.get("projected_unit_ceiling_at_chao")
    return out


def main():
    evt, mm = evt_lookup(), mm_lookup()
    lines = [HEADER]
    for lm in LMS:
        lines += [f"## {lm} column" + ("  (paper-exact)" if lm == "Qwen3-8B" else
                  "  (modern substitute column — user decision 2026-07-21)"), "",
                  "| bench | GEPA shipped | M_ω shipped | **EVT endpoint** | "
                  "**missing-value ceiling** (panel scale) | pool max (diag) | "
                  "union oracle (diag) |",
                  "|---|---|---|---|---|---|---|"]
        for bench in BENCHES:
            M = pool(bench, lm)
            ships = shipped(bench, lm)
            def s(arm):
                v = ships.get(arm)
                if not v or v[1] is None:
                    return "—"
                flag = " ⚠" if v[2] else ""
                return f"{v[1]:.3f}{flag}"
            def s_momega():
                # best guarded unitrecomb VARIANT (unitrecomb, unitrecomb_v5sk2, ...)
                ur = [(k, v) for k, v in ships.items()
                      if k.startswith("unitrecomb") and v[1] is not None]
                if not ur:
                    return "—"
                k, v = max(ur, key=lambda kv: kv[1][1])
                flag = " ⚠" if v[2] else ""
                tag = "" if k == "unitrecomb" else f" ({k.split('_', 1)[1]})"
                return f"{v[1]:.3f}{flag}{tag}"
            if M is None:
                pm = uo = "pending"
            else:
                pm, uo = f"{M.mean(1).max():.3f}", f"{M.max(0).mean():.3f}"
            e = evt.get((bench, lm))
            es = (f"**{e[0]:.3f} [{e[1][0]:.3f},{e[1][1]:.3f}]**" + ("" if e[2] else " ⚠interval-only")
                  ) if e and np.isfinite(e[0]) else "—"
            mv = mm.get((bench, lm))
            mvs = f"{mv:.3f}" if isinstance(mv, (int, float)) else "—"
            lines.append(f"| {bench} | {s('official')} | {s_momega()} | {es} | {mvs} | "
                         f"{pm} | {uo} |")
        lines.append("")
    lines += ["⚠ = regression_flag (review for outage corruption before quoting). GLM-column "
              "cells inherit the outage caveats logged in the plan note until uniform rescores "
              "complete.", ""]
    (HERE / "runs" / "UPPER_BOUNDS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
