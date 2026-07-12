"""Production-GEPA lineage on metrics STRATIFIED by subtask_breadth (specificity class).

For each chosen metric we run optimizer.improve() at one or more token budgets (L) and read the
registry lineage back: every minted version carries its GEPA operator (INIT/CLARIFY/MECHANIZE/
DECOMPOSE/FEWSHOT+/ANCHOR/EDGE/PRUNE), its optimizer round, its parent, and its measured length
(instruction_tokens / clause_count). Scorecards give fidelity per version. Join with the metric's
specificity class to answer:
  - which specificity classes need LONGER rubrics (length at convergence) vs tap out short,
  - which operators fire in which class,
  - whether each operator GROWS or SHRINKS the prompt (child - parent length), and helps fidelity.

ONE resident model plays every role (judge/reviser/reconstructor) -> 1 GPU. Cheap scorecard
(oracle off, 2 reliability passes, small data budget).
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np

from .config import BudgetCaps, ImplementerConfig, apply_task_preset
from .manifest import full_manifest, load_corpus, load_metrics
from .optimizer import improve
from .registry import Registry
from .scale import _vllm_roles

CLASSES = ["very_narrow", "narrow", "moderate", "broad", "very_broad"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--tasks", default="creative_writing,math_se")
    ap.add_argument("--per-class", type=int, default=2, help="metrics per breadth class per task")
    ap.add_argument("--caps", default="120,600", help="token budgets L (one improve() run each)")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-items", type=int, default=24)
    ap.add_argument("--registry-dir", default="/lfs/skampere3/0/alexspan/tmp_vinfo/gepa_registry")
    ap.add_argument("--out", default="/lfs/skampere3/0/alexspan/tmp_vinfo/gepa_lineage.json")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args(argv)

    man = full_manifest(metrics_per_task=400, metric_files_cap=1200)
    caps_list = [int(x) for x in args.caps.split(",")]
    reg = Registry(Path(args.registry_dir))

    cfg0 = ImplementerConfig()
    if args.fake:
        cfg0.vllm_fake = True
    cfg0.n_oracle_items = 0
    cfg0.reliability_passes = 2
    roles = _vllm_roles(cfg0, args.model, args.model)   # one resident model, every role

    selected = []
    for tname in args.tasks.split(","):
        entry = next((e for e in man.datasets if e.name == tname), None)
        if entry is None:
            print("skip", tname); continue
        cfg = ImplementerConfig(); apply_task_preset(cfg, entry.task)
        cfg.vllm_fake = args.fake; cfg.n_oracle_items = 0; cfg.reliability_passes = 2
        texts, ids = load_corpus(entry, args.n_items, seed=7)
        by = collections.defaultdict(list)
        for m in load_metrics(entry):
            b = (m.meta or {}).get("subtask_breadth")
            if b:
                by[b].append(m)
        chosen = [m for c in CLASSES for m in by.get(c, [])[: args.per_class]]
        print(f"{tname}: " + ", ".join(f"{c}={min(len(by.get(c, [])), args.per_class)}" for c in CLASSES)
              + f"  -> {len(chosen)} metrics")
        for m in chosen:
            for cap in caps_list:
                bc = BudgetCaps(instruction_tokens=cap, n_fewshots=max(0, cap // 150),
                                optimizer_rounds=args.rounds, data_budget=args.n_items)
                rid = f"lin_{tname}_{m.metric_id[:24]}_t{cap}"
                t0 = time.time()
                try:
                    improve(m, texts, roles, cfg, reg, caps=bc, rounds=args.rounds,
                            run_id=rid, data_ids=ids, log=lambda *a, **k: None)
                    print(f"  ok {rid} ({time.time() - t0:.0f}s)", flush=True)
                except Exception as e:
                    print(f"  FAIL {rid}: {type(e).__name__}: {str(e)[:140]}", flush=True)
            selected.append((tname, m))

    # read lineage back from the registry + join breadth + fidelity
    rows = []
    for tname, m in selected:
        b = (m.meta or {}).get("subtask_breadth")
        # fidelity per version: scorecard files are named "{version_id}__{kind}__{hash}.json"
        scdir = reg._mdir(m.metric_id) / "scorecards"
        fid = {}
        if scdir.exists():
            for p in scdir.glob("*__prompt__*.json"):
                try:
                    fid[p.name.split("__")[0]] = json.loads(p.read_text()).get("fidelity_scalar")
                except Exception:
                    pass
        for v in reg.versions(m.metric_id, kind="prompt"):
            lin = v.get("lineage", {})        # parent_version / operator / optimizer_round
            bud = v.get("budget", {})         # instruction_tokens / clause_count / n_fewshots
            vid = v.get("version_id")
            rows.append({
                "task": tname, "metric_id": m.metric_id, "name": m.name, "breadth": b,
                "version_id": vid, "operator": lin.get("operator"),
                "round": lin.get("optimizer_round"), "parent": lin.get("parent_version"),
                "instruction_tokens": bud.get("instruction_tokens"),
                "clause_count": bud.get("clause_count"), "n_fewshots": bud.get("n_fewshots"),
                "fidelity": fid.get(vid)})
    json.dump(rows, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}: {len(rows)} versions over {len(selected)} (metric,task)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
