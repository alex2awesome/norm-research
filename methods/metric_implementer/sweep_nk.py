"""Multi-optimizer N×K sweep on REAL data (sk3 vLLM, 1 GPU) — the corpus generator for the
N/K scaling-law analysis.

Mirrors ``scale.improve_all`` but runs FOUR optimizers — GEPA (reflective) + EvoPrompt
(population GA) + ProTeGi (textual-gradient beam) + APE/OPRO (induction) — over a data-budget
(N) × few-shot (K) grid at ONE resident judge tier (E fixed). Every version is tagged in the
registry by ``optimizer``, by N (``budget.data_budget_n``), and by K
(``budget.caps_in_force.n_fewshots``), so the resulting population varies optimizer × N × K at
fixed E — exactly the inputs to (a) N- and K-axis scaling curves and (b) mechanism-diverse
"which prompt features predict recovery".

One resident model serves all roles -> 1 GPU [[feedback_gpu_usage]]. Resumable: a (run_id)
that already logged RUN_FINISHED is skipped, so a killed job restarts where it stopped.
sk3-only; run under nohup with HOME pinned to /lfs [[feedback_sk3_afs_tokens]].

    # GPU-free dry-run of the whole driver (validate plumbing)
    python -m methods.metric_implementer.sweep_nk --fake --tasks law --n-metrics 1 \
        --n-grid 15,60 --k-grid 0 --out-root /tmp/nk_fake

    # real run, one resident 7B judge, one GPU
    HOME=/lfs/skampere3/0/alexspan CUDA_VISIBLE_DEVICES=6 \
        python -m methods.metric_implementer.sweep_nk --tasks law,humor --n-metrics 2 \
        --model Qwen/Qwen2.5-7B-Instruct --out-root outputs/metric_implementer_scale/nksweep
"""

from __future__ import annotations

import argparse
import json
import time
from typing import List, Optional

from .config import BudgetCaps, ImplementerConfig, apply_task_preset
from .manifest import full_manifest, load_corpus, load_metrics
from .optimizer import improve
from .optimizers import ape, evoprompt, protegi
from .registry import Registry
from .scale import _vllm_roles


def _gepa(seed, texts, roles, cfg, reg, *, caps, rounds, run_id, data_ids, log):
    return improve(seed, texts, roles, cfg, reg, caps=caps, rounds=rounds, run_id=run_id,
                   data_ids=data_ids, log=log)


OPTS = {"gepa": _gepa, "evoprompt": evoprompt, "protegi": protegi, "ape": ape}


def _tune_cfg(cfg, fake: bool, model: str) -> None:
    """Bounded, cheap scorecard sizes for the sweep; one resident model plays every role."""
    cfg.vllm_fake = fake
    cfg.judge_model = model
    cfg.n_reliability_items = 20
    cfg.reliability_passes = 2
    cfg.n_reconstruct_label_items = 30
    cfg.n_reconstruct_shown = 12
    cfg.n_reconstruct_behavioral = 20
    cfg.n_cf_base_texts = 6
    cfg.n_consistency_items = 10
    cfg.n_oracle_items = 20            # self-oracle via the SAME resident model (no extra GPU)
    cfg.n_mutations = 2


def _done(reg: Registry, run_id: str) -> bool:
    try:
        return any(e.get("event") == "RUN_FINISHED" and e.get("run_id") == run_id
                   for e in reg.ledger())
    except Exception:
        return False


def sweep(*, tasks: Optional[List[str]] = None, n_metrics: int = 2, n_items: int = 60,
          model: str = "Qwen/Qwen2.5-7B-Instruct", n_grid=(15, 30, 60), k_grid=(0, 2),
          rounds: int = 2, token_cap: int = 400, optimizers: Optional[List[str]] = None,
          out_root: Optional[str] = None, run_id: Optional[str] = None, fake: bool = False,
          log=print) -> dict:
    run_id = run_id or f"nksweep_{int(time.time())}"
    optimizers = optimizers or list(OPTS)
    datasets = full_manifest(metrics_per_task=n_metrics).datasets
    if tasks:
        datasets = [e for e in datasets if e.task in set(tasks)
                    or any(t in e.name for t in tasks)]
    plan = len(datasets) * n_metrics * len(n_grid) * len(k_grid) * len(optimizers)
    log(f"[sweep {run_id}] model={model} tasks={[e.task for e in datasets]} "
        f"N={list(n_grid)} K={list(k_grid)} optimizers={optimizers} "
        f"-> up to {plan} optimizer runs")
    summary = {"run_id": run_id, "model": model, "planned": plan,
               "runs": 0, "skipped": 0, "failed": 0}

    for entry in datasets:
        cfg = ImplementerConfig()
        apply_task_preset(cfg, entry.task)
        _tune_cfg(cfg, fake, model)
        if out_root:
            cfg.output_dir = out_root
        reg = Registry(cfg.registry_dir())
        roles = _vllm_roles(cfg, model, model)          # one resident model = 1 GPU
        texts, ids = load_corpus(entry, n_items, seed=0)
        for art in load_metrics(entry)[:n_metrics]:
            reg.register_metric(art.metric_id, art.name, art.description)
            for N in n_grid:
                for K in k_grid:
                    for name in optimizers:
                        rid = f"{run_id}__{entry.name}__{art.metric_id}__N{N}_K{K}__{name}"
                        if _done(reg, rid):
                            summary["skipped"] += 1
                            continue
                        caps = BudgetCaps(instruction_tokens=token_cap, n_fewshots=K,
                                          optimizer_rounds=rounds, data_budget=N)
                        try:
                            OPTS[name](art, texts, roles, cfg, reg, caps=caps, rounds=rounds,
                                       run_id=rid, data_ids=ids, log=log)
                            summary["runs"] += 1
                        except Exception as e:           # one cell failing must not kill the sweep
                            log(f"[FAIL] {rid}: {type(e).__name__}: {e}")
                            summary["failed"] += 1
        log(f"[sweep] {entry.name}: done")
    log(f"[sweep {run_id}] {json.dumps(summary)}")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default=None, help="comma list of task names (default: all 7)")
    ap.add_argument("--n-metrics", type=int, default=2)
    ap.add_argument("--n-items", type=int, default=60)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-grid", default="15,30,60", help="data_budget (N) grid")
    ap.add_argument("--k-grid", default="0,2", help="n_fewshots (K) grid")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--token-cap", type=int, default=400)
    ap.add_argument("--optimizers", default=None, help="comma subset of gepa,evoprompt,protegi,ape")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--fake", action="store_true", help="FakeVLLM dry-run (no GPU)")
    args = ap.parse_args(argv)
    if args.fake and not args.out_root:
        args.out_root = f"/tmp/nk_fake/{int(time.time())}"
    res = sweep(
        tasks=[t.strip() for t in args.tasks.split(",")] if args.tasks else None,
        n_metrics=args.n_metrics, n_items=args.n_items, model=args.model,
        n_grid=tuple(int(x) for x in args.n_grid.split(",")),
        k_grid=tuple(int(x) for x in args.k_grid.split(",")),
        rounds=args.rounds, token_cap=args.token_cap,
        optimizers=[o.strip() for o in args.optimizers.split(",")] if args.optimizers else None,
        out_root=args.out_root, run_id=args.run_id, fake=args.fake)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
