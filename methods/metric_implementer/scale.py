"""Scaled E7 orchestrator — many metrics × datasets × models, vLLM offline batch on sk3.

Architecture (decouples SEARCH from MEASUREMENT so exactly ONE large model is resident at a
time — fits the 1-GPU cluster rule [[feedback_gpu_usage]], [[feedback_minimize_gpus]]):

  Phase SEARCH (improve_all):  one resident STRONG model serves reviser/reconstructor/oracle/
      judge. For every (dataset × metric × seed × token_cap) it runs the GEPA loop, minting a
      lineage of prompt iterations — INIT -> CLARIFY / MECHANIZE / FEWSHOT+ / ANCHOR / EDGE /
      PRUNE / DECOMPOSE — each persisted IMMUTABLY to the Registry (every prompt + iteration
      saved, the user's requirement).

  Phase MEASURE (score_all):  for each judge TIER (capability axis), load it ONCE, then
      MEGA-BATCH every (metric × prompt-version × item × pass) across all datasets through a
      single vLLM model and stream the long table — keyed by (judge_model, item, version
      [=prompt], round [=iteration]) + the GEPA operator that minted the prompt. Thousands of
      prompts per generate call ([[feedback_vllm_batch_size]]); checkpointed per flush.

The long table (batch_scoring.LONGTABLE_COLUMNS) is the deliverable; the per-tier articulability
frontier and E7 brackets are computed from it offline (e7_brackets.py, extended).

sk3-only ([[feedback_metric_implementer_sk3_only]]): all roles are vLLM-resident open models;
no OpenRouter. Run under nohup with HOME pinned to /lfs ([[feedback_sk3_afs_tokens]]).

CLI:
  python -m methods.metric_implementer.scale search   [--manifest pilot] [--fake]
  python -m methods.metric_implementer.scale measure   [--manifest pilot] [--fake]
  python -m methods.metric_implementer.scale all       [--manifest pilot] [--fake]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .artifact import MetricArtifact
from .batch_scoring import (LongTableWriter, ScoreJob, batch_score_many,
                            score_continuous_many, score_sampled_many,
                            vacuous_baseline_artifact)
from .config import ImplementerConfig, apply_task_preset
from .manifest import RunManifest, load_corpus, load_metrics, pilot_manifest, full_manifest
from .registry import Registry
from .vllm_backend import make_judge_backend


# ---- role wiring (vLLM-resident, duck-compatible with backends.Roles) ----------------------

def _vllm_roles(cfg, judge_model: str, strong_model: str):
    """A Roles-shaped object whose members are vLLM backends. For SEARCH the strong model
    plays every smart role (and judges at the anchor tier); cheap to build, one resident model
    when judge_model == strong_model."""
    from .backends import Roles
    jb = make_judge_backend(judge_model, cfg, getattr(cfg, "judge_temperature", 0.7))
    sb = make_judge_backend(strong_model, cfg, getattr(cfg, "other_temperature", 0.7))
    ob = make_judge_backend(strong_model, cfg, 0.3) if getattr(cfg, "n_oracle_items", 0) else None
    return Roles(judge=jb, reviser=sb, reconstructor=sb, acceptance_reconstructor=sb,
                 generator=sb, acceptance_generator=sb, grader=sb, oracle=ob)


def _cfg_for(task: str, manifest: RunManifest, fake: bool) -> ImplementerConfig:
    cfg = ImplementerConfig()
    apply_task_preset(cfg, task)
    cfg.vllm_fake = fake
    cfg.n_oracle_items = manifest.n_oracle_items
    cfg.reliability_passes = manifest.passes
    return cfg


# ---- SEARCH: mint prompt iterations into the registry --------------------------------------

def improve_all(manifest: RunManifest, *, fake: bool = False,
                out_root: Optional[str] = None, log=print) -> dict:
    """Run GEPA per (dataset × metric × seed × token_cap) under the resident strong model.
    Every iteration is persisted to the registry by the optimizer. ``out_root`` must match the
    value passed to ``score_all`` so MEASURE sees the iterations SEARCH minted."""
    from .optimizer import improve
    from .config import BudgetCaps

    strong = manifest.strong_model
    minted = 0
    for entry in manifest.datasets:
        cfg = _cfg_for(entry.task, manifest, fake)
        if out_root:
            cfg.output_dir = out_root
        registry = Registry(cfg.registry_dir())
        roles = _vllm_roles(cfg, strong, strong)        # anchor-tier optimization, 1 model
        metrics = load_metrics(entry)
        for seed in manifest.seeds:
            cfg.random_seed = seed
            texts, ids = load_corpus(entry, manifest.n_items, seed=seed)
            for art in metrics:
                for cap in manifest.token_caps:
                    caps = BudgetCaps(instruction_tokens=cap,
                                      n_fewshots=max(0, cap // 150),
                                      optimizer_rounds=manifest.gepa_rounds)
                    rid = f"{manifest.run_id}_search_{entry.name}_{art.metric_id}_s{seed}_t{cap}"
                    try:
                        improve(art, texts, roles, cfg, registry, caps=caps,
                                rounds=manifest.gepa_rounds, run_id=rid, data_ids=ids, log=log)
                        minted += 1
                    except Exception as e:        # one metric failing must not kill the sweep
                        log(f"  [SEARCH FAIL] {rid}: {type(e).__name__}: {e}")
        log(f"[SEARCH] {entry.name}: done ({len(metrics)} metrics x {len(manifest.seeds)} "
            f"seeds x {len(manifest.token_caps)} caps)")
    return {"run_id": manifest.run_id, "search_runs": minted}


# ---- MEASURE: mega-batch every prompt-version across every tier -> long table --------------

def _all_prompt_versions(registry: Registry, metric_id: str) -> List[dict]:
    return registry.versions(metric_id, kind="prompt")


def score_all(manifest: RunManifest, *, fake: bool = False, include_history: bool = True,
              out_root: Optional[str] = None, continuous: bool = False,
              sampled: bool = False, temperature: float = 0.7,
              baseline: bool = True, baseline_only: bool = False, log=print) -> dict:
    """For each tier (resident once), mega-batch score every prompt version of every metric
    over the corpus, ``passes`` times, streaming the long table. include_history=False scores
    only HEAD/seed versions (cheaper). ``out_root`` overrides the output base (testability +
    custom run dirs); registries live under ``<out_root>/<task>/registry``."""
    base = out_root or ImplementerConfig().output_dir
    out_dir = Path(base) / "longtable"
    writer = LongTableWriter(out_dir=out_dir, run_id=manifest.run_id, flush_every=4000)
    (out_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (out_dir / "manifests" / f"{manifest.run_id}.json").write_text(
        json.dumps(manifest.to_json(), indent=2))

    per_tier_counts = {}
    for tier in manifest.tiers:
        cfg0 = ImplementerConfig()
        cfg0.vllm_fake = fake
        backend = make_judge_backend(tier, cfg0, getattr(cfg0, "judge_temperature", 0.7))
        jobs: List[ScoreJob] = []
        for entry in manifest.datasets:
            cfg = _cfg_for(entry.task, manifest, fake)
            if out_root:
                cfg.output_dir = out_root
            registry = Registry(cfg.registry_dir())
            texts, ids = load_corpus(entry, manifest.n_items, seed=0)   # fixed eval corpus
            # empty-rubric H_V(Y|X) baseline: one criteria-free CONTROL job per dataset (per
            # tier) — NOT a target M. Subtract it from any rubric for conditional V-info /
            # substitutive-vs-complementary tacit split. --baseline-only backfills it cheaply for existing runs.
            if baseline or baseline_only:
                jobs.append(ScoreJob(
                    artifact=vacuous_baseline_artifact(entry.task),
                    version_id=f"baseline::{entry.name}", texts=texts, item_ids=ids,
                    operator="BASELINE", round=-1, token_cap=0,
                    dataset=entry.name, task=entry.task))
            if baseline_only:
                continue
            for art in load_metrics(entry):
                registry.register_metric(art.metric_id, art.name, art.description)
                versions = (_all_prompt_versions(registry, art.metric_id) if include_history
                            else [])
                if not versions:        # ensure at least the seed exists as a version
                    vid = registry.head(art.metric_id, "prompt") or registry.create_version(
                        art, operator="INIT", parent_version=None, run_id=manifest.run_id,
                        optimizer_round=0, data_budget_ids=ids, caps_in_force=None,
                        models={"judge": tier})
                    versions = [registry.get_version(art.metric_id, vid, "prompt")]
                for v in versions:
                    a = registry.artifact_from_version(v)
                    jobs.append(ScoreJob(
                        artifact=a, version_id=v["version_id"], texts=texts, item_ids=ids,
                        operator=(v.get("lineage") or {}).get("operator", "INIT"),
                        round=(v.get("lineage") or {}).get("optimizer_round") or 0,
                        token_cap=((v.get("budget") or {}).get("caps_in_force") or {}
                                   ).get("instruction_tokens"),
                        dataset=entry.name, task=entry.task))
        log(f"[MEASURE] tier={tier}: {len(jobs)} (metric×version) jobs × {manifest.n_items} "
            f"items × {manifest.passes} passes = "
            f"{sum(len(j.texts) for j in jobs)*manifest.passes} prompts")
        # mega-batch all jobs for this resident tier in chunks (cap prompts/flush for memory)
        CHUNK = 256       # jobs per generate flush; each job = n_items*passes prompts
        for i in range(0, len(jobs), CHUNK):
            chunk = jobs[i:i + CHUNK]
            if sampled:
                score_sampled_many(chunk, backend, cfg0, run_id=manifest.run_id,
                                   passes=manifest.passes, temperature=temperature, writer=writer)
            elif continuous:
                score_continuous_many(chunk, backend, cfg0, run_id=manifest.run_id,
                                      passes=manifest.passes, writer=writer)
            else:
                batch_score_many(chunk, backend, cfg0, run_id=manifest.run_id,
                                 passes=manifest.passes, writer=writer)
        writer.flush()
        per_tier_counts[tier] = len(jobs)
        log(f"[MEASURE] tier={tier} done; backend stats {backend.stats.as_dict()}")
    writer.flush()
    return {"run_id": manifest.run_id, "long_rows": writer.n_written,
            "per_tier_jobs": per_tier_counts, "longtable_dir": str(out_dir)}


# ---- CLI -----------------------------------------------------------------------------------

def _manifest(name: str) -> RunManifest:
    if name == "pilot":
        return pilot_manifest()
    if name == "full" or name.startswith("full:"):
        mpt = int(name.split(":", 1)[1]) if ":" in name else 40
        return full_manifest(metrics_per_task=mpt)
    p = Path(name)
    if p.exists():
        raise SystemExit("custom manifest JSON loading: TODO (use 'pilot'/'full')")
    raise SystemExit(f"unknown manifest {name!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["search", "measure", "all"])
    ap.add_argument("--manifest", default="pilot")
    ap.add_argument("--fake", action="store_true",
                    help="FakeVLLM (no GPU) — dry-run the full pipeline + long table")
    ap.add_argument("--no-history", action="store_true",
                    help="measure only seed/HEAD versions, not every iteration")
    ap.add_argument("--out-root", default=None,
                    help="output base for registries + long table (default: cfg.output_dir); "
                         "SEARCH and MEASURE must share it")
    ap.add_argument("--tier", default=None,
                    help="restrict to ONE judge tier (one model per process -> clean GPU teardown)")
    ap.add_argument("--continuous", action="store_true",
                    help="continuous-fidelity readout: score = P(YES) via logprobs (bypasses JSON)")
    ap.add_argument("--sampled", action="store_true",
                    help="no-abstention multi-pass SAMPLED YES/NO (temp>0, per-pass seed) -> "
                         "pass-rate = response, across-pass agreement = consistency")
    ap.add_argument("--temp", type=float, default=0.7, help="temperature for --sampled")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the empty-rubric H_V(Y|X) control baseline jobs")
    ap.add_argument("--baseline-only", action="store_true",
                    help="score ONLY the empty-rubric baseline (cheap backfill of the "
                         "conditional-V-info baseline column for an existing run)")
    ap.add_argument("--passes", type=int, default=None, help="override manifest passes")
    ap.add_argument("--run-id", default=None,
                    help="override manifest run_id (unique long-table part names per process)")
    args = ap.parse_args(argv)
    m = _manifest(args.manifest)
    if args.tier:
        m.tiers = [args.tier]
    if args.passes:
        m.passes = args.passes
    if args.run_id:
        m.run_id = args.run_id
    # safety: a --fake dry-run must never write into the real registries / long table.
    # Default it to a scratch dir unless the caller chose one explicitly.
    if args.fake and not args.out_root:
        args.out_root = f"/tmp/e7scale_fake/{m.run_id}"
        print(f"[scale] --fake without --out-root: writing to scratch {args.out_root}")
    t0 = time.time()
    if args.mode in ("search", "all"):
        print(json.dumps(improve_all(m, fake=args.fake, out_root=args.out_root), indent=2))
    if args.mode in ("measure", "all"):
        print(json.dumps(score_all(m, fake=args.fake, out_root=args.out_root,
                                   include_history=not args.no_history,
                                   continuous=args.continuous, sampled=args.sampled,
                                   temperature=args.temp, baseline=not args.no_baseline,
                                   baseline_only=args.baseline_only), indent=2))
    print(f"[scale {args.mode}] wall {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
