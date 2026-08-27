"""GEPA prompt-optimization scoped to VIABLE rubrics (the goal's "GEPA + reconstruction accuracy" step).

run_distillation.py distills EVERY rubric, but inapplicable ones collapse and GEPA can't fix a
concept that doesn't apply. This pre-filters to rubrics carrying signal, then runs
metric_implementer.optimizer.improve on each. improve()'s objective is fidelity_scalar, which
weights reconstruction accuracy (w_recon) + reliability + counterfactual + discrimination.

Plumbing note: the viability probe uses metric_implementer's OWN LLMBackend (sync urllib via
generate_batch) — NOT the ctree's httpx make_vllm_judge_scorer — so it doesn't fire
"Event loop is closed" when mixed with metric_implementer's flow.

    python gepa_viable.py creative-writing 40 40 1 5   # task n_metrics n_probe rounds max_viable
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "methods"))

from metric_implementer.artifact import MetricArtifact
from metric_implementer.backends import make_roles
from metric_implementer.config import BudgetCaps, ImplementerConfig, apply_task_preset
from metric_implementer.measures import compute_scorecard
from metric_implementer.optimizer import improve
from metric_implementer.registry import Registry

from metrics_tree_infilling.io_metrics import REPO_ROOT, load_rubric_metrics
from metrics_tree_infilling.run import DATASET_CONFIGS

GEMMA4 = "google/gemma-4-31b-it"


def _load_texts(task, n, seed):
    dcfg = DATASET_CONFIGS[task]
    base = REPO_ROOT / dcfg["split"]
    fp = base if base.is_file() else next(
        (Path(str(base) + e) for e in (".csv.gz", ".csv") if Path(str(base) + e).exists()), base)
    df = pd.read_csv(fp, low_memory=False)
    s = df[dcfg["text"]].dropna().astype(str)
    return s.sample(n=min(n, len(s)), random_state=seed).tolist()


def _judge_prompt(rubrics, text, max_chars):
    crit = "\n".join(f"{k}. {m.name}: {m.rubric_text or m.description}" for k, m in enumerate(rubrics))
    return ("Score the TEXT on each criterion. For each, decide if the criterion is applicable to "
            "this text, and if so give a score in [0,1] (1 = fully satisfies). Return ONLY a JSON "
            'array of objects {"index": int, "applicable": bool, "score": number}.\n\n'
            f"CRITERIA:\n{crit}\n\nTEXT:\n{text[:max_chars]}")


def _parse_arr(resp):
    if not resp:
        return []
    s = resp.strip()
    lo, hi = s.find("["), s.rfind("]")
    if lo == -1 or hi == -1 or hi <= lo:
        return []
    try:
        return [o for o in json.loads(s[lo:hi + 1]) if isinstance(o, dict)]
    except Exception:
        return []


def _viable_roles(rubrics, texts, roles, max_chars):
    """Cheap no-httpx viability probe: one judge call per item scores all rubrics (sync urllib)."""
    prompts = [_judge_prompt(rubrics, t, max_chars) for t in texts]
    resps = roles.judge.generate_batch(prompts, max_tokens=max(1800, 70 * len(rubrics)))
    n, M = len(texts), len(rubrics)
    lv = np.full((n, M), np.nan)
    ap = np.zeros((n, M), bool)
    for i, r in enumerate(resps):
        for o in _parse_arr(r):
            j = o.get("index")
            if isinstance(j, int) and 0 <= j < M and o.get("applicable", True) and o.get("score") is not None:
                lv[i, j] = float(np.clip(o["score"], 0, 1))
                ap[i, j] = True
    return [rubrics[j] for j in range(M)
            if ap[:, j].mean() > 0.3 and np.std(lv[ap[:, j], j]) > 0.1]


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else "creative-writing"
    n_metrics = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    n_probe = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    max_viable = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    t0 = time.time()

    cfg = ImplementerConfig()
    apply_task_preset(cfg, task)   # correct per-task judge framing (CW=story, not code-solution)
    cfg.task = task
    cfg.random_seed = 7
    cfg.judge_model = GEMMA4
    cfg.reviser_model = GEMMA4
    roles = make_roles(cfg)

    rubrics = load_rubric_metrics(task, limit=n_metrics)
    probe_texts = _load_texts(task, n_probe, 7)
    viable = _viable_roles(rubrics, probe_texts, roles, getattr(cfg, "max_text_chars", 2000))
    if max_viable:
        viable = viable[:max_viable]
    print(f"[gepa-viable] task={task} rubrics={len(rubrics)} viable={len(viable)} rounds={rounds}",
          flush=True)
    if not viable:
        print("[gepa-viable] no viable rubrics; abort.", flush=True)
        return 1

    registry = Registry(cfg.registry_dir())
    texts = probe_texts
    rng = np.random.default_rng(cfg.random_seed)
    rows = []
    for i, r in enumerate(viable, 1):
        slug = "".join(c if c.isalnum() else "_" for c in r.name.lower())[:40] or "metric"
        mid = f"gepav2_{task}_{slug}_{r.metric_id[:8]}"
        seed = MetricArtifact(metric_id=mid, kind="prompt", body=r.rubric_text,
                              name=r.name, description=r.description)
        registry.register_metric(seed.metric_id, seed.name, seed.description)
        seed_card = compute_scorecard(seed, texts, roles, cfg, np.random.default_rng(cfg.random_seed))
        seed_fid = seed_card["fidelity_scalar"]
        summary = improve(seed, texts, roles, cfg, registry, caps=BudgetCaps(optimizer_rounds=rounds),
                          rounds=rounds, data_ids=[str(j) for j in range(len(texts))],
                          run_id=f"gepav2_{mid}", log=lambda *a, **k: None)
        # Authoritative acceptance: improve() gates on a cross-family acceptance scorecard and
        # writes a tiered HEAD (prompt@judge_model). Reading the un-tiered HEAD misreports 'no'
        # even when a mutant was accepted (Codex #1); use the returned summary's `accepted` flag.
        accepted = bool(summary.get("accepted")) if summary else False
        best_acc_fid = (summary or {}).get("best_fidelity_acceptance")
        acc_s = f"{best_acc_fid:.3f}" if isinstance(best_acc_fid, (int, float)) else "  na"
        print(f"  ({i}/{len(viable)}) {r.name[:44]:46} seed_fid={seed_fid:.3f} "
              f"acc_fid={acc_s} accepted={'yes' if accepted else 'no'}", flush=True)
        rows.append((r.name, round(seed_fid, 3), accepted, mid))

    improved = sum(1 for _, _, c, _ in rows if c)
    print(f"\n[gepa-viable] DONE in {time.time() - t0:.0f}s | viable={len(rows)} "
          f"accepted={improved} cost=${roles.total_cost():.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
