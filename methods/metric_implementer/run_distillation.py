"""Batch GEPA distillation of a task's rubric metrics into optimized prompts.

Pre-task stage for ``metrics_tree_infilling``: for each rubric metric in
``datasets/<task>/online-rubrics``, run :func:`metric_implementer.optimizer.improve` (the GEPA loop,
fidelity-only objective) with a **Gemma 4** judge (+ cross-family Qwen/Llama acceptance), and
persist the optimized prompts to the registry. Pair with ``export.py`` (and ``--rubrics-dir`` in
``metrics_tree_infilling/run.py``) to feed the tree cheap, optimized scorers so it can run at scale
off OpenRouter instead of the rate-limited z.ai proxy.

The seed for each metric is a ``kind="prompt"`` :class:`MetricArtifact` whose body is the rubric.
``improve()`` mutates prompts only here (no code kind), so ``convergence_queue`` is left empty.

Example
-------
    PYTHONPATH=methods python -m metric_implementer.run_distillation \\
        --task peer-review --max-metrics 5 --rounds 2 --sample-items 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .artifact import MetricArtifact
from .backends import make_roles
from .config import BudgetCaps, ImplementerConfig, apply_task_preset
from .measures import compute_scorecard
from .optimizer import improve
from .registry import Registry

GEMMA4 = "google/gemma-4-31b-it"   # OpenRouter; tracked in manifest.py


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="dataset/task name, e.g. peer-review")
    p.add_argument("--max-metrics", type=int, default=5, help="cap # rubric metrics to distill")
    p.add_argument("--sample-items", type=int, default=200, help="# corpus texts passed to GEPA")
    p.add_argument("--rounds", type=int, default=2, help="GEPA rounds per metric")
    p.add_argument("--judge-model", default=GEMMA4)
    p.add_argument("--reviser-model", default=GEMMA4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-registry", default=None, help="override registry dir (defaults to cfg)")
    return p


def _load_rubrics(task: str, limit: int):
    """Base rubric metrics for the task, via the infilling loader."""
    from metrics_tree_infilling.io_metrics import load_rubric_metrics
    return load_rubric_metrics(task, limit=limit)


def _load_texts(task: str, n: int, seed: int) -> List[str]:
    """Sample ``n`` corpus texts for the task (reads the dataset CSV directly)."""
    from metrics_tree_infilling.run import DATASET_CONFIGS
    from metrics_tree_infilling.io_metrics import REPO_ROOT
    dcfg = DATASET_CONFIGS[task]
    base = REPO_ROOT / dcfg["split"]
    fp = next((Path(str(base) + ext) for ext in (".csv.gz", ".csv") if Path(str(base) + ext).exists()), None)
    if fp is None:
        raise FileNotFoundError(f"dataset split not found near {base}")
    df = pd.read_csv(fp, low_memory=False)
    col = dcfg["text"]
    s = df[col].dropna().astype(str)
    if len(s) > n:
        s = s.sample(n=n, random_state=seed)
    return s.tolist()


def _seed_artifact(rubric, task: str) -> MetricArtifact:
    """Build the prompt seed for one rubric metric."""
    slug = "".join(c if c.isalnum() else "_" for c in rubric.name.lower())[:40] or "metric"
    mid = f"distill_{task}_{slug}_{rubric.metric_id[:8]}"
    return MetricArtifact(
        metric_id=mid, kind="prompt",
        body=rubric.rubric_text, name=rubric.name, description=rubric.description,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.time()

    cfg = ImplementerConfig()
    try:
        apply_task_preset(cfg, args.task)          # picks up task-specific cfg if known
    except KeyError:
        pass                                       # unknown task: keep defaults
    cfg.task = args.task                           # registry_dir/runs key on the ACTUAL task
    cfg.random_seed = args.seed
    cfg.judge_model = args.judge_model
    cfg.reviser_model = args.reviser_model
    # acceptance roles keep their Qwen/Llama defaults -> cross-family vs Gemma judge

    rubrics = _load_rubrics(args.task, args.max_metrics)
    texts = _load_texts(args.task, args.sample_items, args.seed)
    print(f"[distill] task={args.task} rubrics={len(rubrics)} texts={len(texts)} "
          f"judge={cfg.judge_model} reviser={cfg.reviser_model}")

    registry = Registry(cfg.registry_dir())
    roles = make_roles(cfg)
    rng = np.random.default_rng(cfg.random_seed)

    rows = []
    for i, r in enumerate(rubrics, 1):
        seed = _seed_artifact(r, args.task)
        registry.register_metric(seed.metric_id, seed.name, seed.description)
        seed_card = compute_scorecard(seed, texts, roles, cfg, np.random.default_rng(cfg.random_seed))
        seed_fid = seed_card["fidelity_scalar"]
        print(f"\n[distill] ({i}/{len(rubrics)}) {r.name!r:48} seed_fidelity={seed_fid:.3f}")

        summary = improve(
            seed, texts, roles, cfg, registry,
            caps=BudgetCaps(optimizer_rounds=args.rounds),
            rounds=args.rounds,
            data_ids=[str(j) for j in range(len(texts))],
            run_id=f"distill_{seed.metric_id}",
            log=print,
        )
        head_vid = registry.head(seed.metric_id, "prompt")
        head_body = registry.get_version(seed.metric_id, head_vid, "prompt")["body"] if head_vid else None
        head_fid = summary.get("fidelity") if isinstance(summary, dict) else None
        rows.append({
            "metric_id": seed.metric_id, "name": r.name,
            "seed_fidelity": round(seed_fid, 3), "head_fidelity": head_fid,
            "head_version": head_vid, "improved": head_body != seed.body if head_body else False,
        })
        print(f"[distill]       head_version={head_vid} head_fidelity={head_fid} "
              f"changed={'yes' if head_body != seed.body else 'no'}")

    out = {
        "task": args.task, "judge_model": cfg.judge_model, "rounds": args.rounds,
        "n_rubrics": len(rubrics), "n_texts": len(texts),
        "wall_s": round(time.time() - t0, 1), "cost_usd": round(roles.total_cost(), 4),
        "metrics": rows,
    }
    out_dir = Path(cfg.registry_dir()) / "distillation_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.task}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[distill] DONE in {out['wall_s']}s cost=${out['cost_usd']:.4f}")
    for row in rows:
        print(f"   {row['name'][:44]:44} seed={row['seed_fidelity']} head={row['head_fidelity']} "
              f"changed={row['improved']}")
    print(f"[distill] summary -> {out_dir / (args.task + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
