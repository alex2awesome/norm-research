"""Profile assembly + convergence analysis for the Tacitness Battery.

Runs every runnable probe at or below --wave on the given slice, writes long-format rows to
the profile store, then computes the convergence readout the whole program is for: across
constructs, do the probes' primary statistics agree on WHICH constructs are tacit?

Convergence method: per (probe, rung) primary statistic -> sign so higher = more tacit ->
rank across constructs -> probe x probe Spearman on shared constructs + PC1 share of the
probe correlation matrix. Cluster-bootstrap CIs deferred to the confirmatory run.

Usage:
  python -m methods.tacit_channels.battery.profile --wave 0 --family qwen25 --domain humor \
      --run-tag w0_v1
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from methods.tacit_channels.battery.artifacts import ArtifactContext, write_profile_rows
from methods.tacit_channels.battery.registry import all_probes
from methods.tacit_channels.channels.common import _rankdata

# probe -> (primary statistic, sign so that HIGHER = MORE TACIT-ish)
PRIMARY = {
    "P-CHAN-core": ("articulation_gain", -1.0),        # low gain (with gap) = tacit-candidate
    "P-STAT-1": ("conf_acc_corr", -1.0),               # low conf-accuracy corr = tacit
    "P-GT-1": ("target_form_consistency", -1.0),       # low reference coherence (diagnostic)
    "P-GT-3": ("dist_qq_corr", -1.0),
    "P-GEN-1": ("typicality_agreement_corr", 1.0),     # steeper decay off-core = rule-bound
    "P-SCAL-1": ("max_articulation_gain", -1.0),
    "P-SCAL-2": ("divergence_slope", 1.0),   # divergence GROWS with capability (Dreyfus)
    "P-CEIL-1": ("cap_oos", -1.0),                     # low articulation ceiling = tacit
}


def run_battery(ctx: ArtifactContext, wave: int, run_tag: str) -> tuple[list[dict], dict]:
    rows = []
    executed = []
    for spec in all_probes(wave=wave, runnable_only=True):
        try:
            out = spec.compute(ctx)
            rows.extend(out)
            executed.append((spec.id, len(out)))
            print(f"  {spec.id:14s} {spec.title[:58]:58s} -> {len(out)} rows")
        except Exception as e:  # a broken probe never kills the battery; it is REPORTED
            executed.append((spec.id, f"ERROR {e}"))
            print(f"  {spec.id:14s} ERROR: {e}")
    for r in rows:
        r.setdefault("run_tag", run_tag)
        r.setdefault("family", ctx.family)
    return rows, dict(executed)


def convergence(rows: list[dict]) -> dict:
    """Probe x probe agreement on which constructs are tacit (rank-standardized, signed)."""
    per_probe: dict[tuple, dict] = defaultdict(dict)   # (probe, rung) -> {construct: signed}
    for r in rows:
        spec = PRIMARY.get(r["probe"])
        if spec is None or r["statistic"] != spec[0] or r["construct"].startswith("_AGG"):
            continue
        key = (r["probe"], r["rung"] if r["rung"] != "_LADDER_" else "_LADDER_")
        per_probe[key][r["construct"]] = spec[1] * r["value"]

    # collapse rungs: mean signed value per (probe, construct) across rungs
    collapsed: dict[str, dict] = defaultdict(dict)
    for (probe, _rung), d in per_probe.items():
        for c, v in d.items():
            collapsed[probe].setdefault(c, []).append(v)
    series = {p: {c: float(np.mean(vs)) for c, vs in d.items()}
              for p, d in collapsed.items()}

    probes = sorted(series)
    mat = np.full((len(probes), len(probes)), np.nan)
    for i, p in enumerate(probes):
        for j, q in enumerate(probes):
            shared = sorted(set(series[p]) & set(series[q]))
            if len(shared) < 10:
                continue
            a = _rankdata(np.array([series[p][c] for c in shared]))
            b = _rankdata(np.array([series[q][c] for c in shared]))
            mat[i, j] = np.corrcoef(a, b)[0, 1]
    # PC1 share over the probe-correlation matrix (nan -> 0 off-diagonal)
    C = np.nan_to_num(mat, nan=0.0)
    np.fill_diagonal(C, 1.0)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    return {
        "probes": probes,
        "matrix": [[None if np.isnan(v) else round(float(v), 3) for v in row]
                   for row in mat],
        "pc1_share": round(float(ev[0] / len(probes)), 4) if len(probes) else None,
        "n_probes": len(probes),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=0)
    ap.add_argument("--family", default="qwen25")
    ap.add_argument("--domain", default="humor")
    ap.add_argument("--exec-jobs", default="", help="comma list; empty = all present")
    ap.add_argument("--run-tag", required=True)
    args = ap.parse_args()

    ctx = ArtifactContext(
        family=args.family, domain=args.domain,
        exec_jobs=tuple(j for j in args.exec_jobs.split(",") if j))
    print(f"Tacitness Battery: wave<={args.wave} {args.family}/{args.domain} "
          f"executors={ctx.executors()}")
    rows, executed = run_battery(ctx, args.wave, args.run_tag)
    path = write_profile_rows(rows, args.run_tag, domain=args.domain)
    conv = convergence(rows)

    summary = {"run_tag": args.run_tag, "family": args.family, "domain": args.domain,
               "wave": args.wave, "n_rows": len(rows), "executed": executed,
               "convergence": conv, "profile_path": path}
    out_dir = "notebooks/data/two_faces_20260702/tacit_profile"
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, f"battery_summary_{args.domain}_{args.run_tag}.json")
    json.dump(summary, open(summary_path, "w"), indent=2)

    print(f"\nprofile rows -> {path}")
    print(f"summary      -> {summary_path}")
    print(f"\nCONVERGENCE (probe x probe rank agreement over constructs):")
    print("  probes:", ", ".join(conv["probes"]))
    for p, row in zip(conv["probes"], conv["matrix"]):
        print(f"  {p:14s}", " ".join("  .  " if v is None else f"{v:+.2f}" for v in row))
    print(f"  PC1 share (general tacitness factor): {conv['pc1_share']}")


if __name__ == "__main__":
    main()
