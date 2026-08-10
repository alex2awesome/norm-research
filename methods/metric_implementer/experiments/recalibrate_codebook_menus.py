"""Recalibrate an existing MCQ codebook's menus for blind-prior balance (centralness).

The Tier-B codebooks were mined with ``selection_method="behavioral_hardness_only"``:
distractors are the highest-kappa (most behaviorally confusable) sibling metrics, but the
menu is not balanced against the *blind* menu prior. A mid-size reconstructor can then
identify the target from its description text alone, so the blind prior sits well above
chance (observed 0.35-0.41 vs 0.25 for four options) and demonstration lift measured
against that prior goes negative.

This module re-selects distractors so the blind menu prior is flat (no option pickable
without demonstrations), reusing the already-tested centralness functions from
``cr3_reconstruction_values`` in the same order ``run_cr3_mining_loop`` uses them:

    build_task_centralness_reference_plan  ->  reference plan (CPU, descriptions only)
    score_task_centralness_reference       ->  centralness calibration (GPU, blind priors)
    build_codebook_panel_plan              ->  hard + centralness-fallback candidate menus
    score_codebook_panel_priors            ->  per-panel blind priors (GPU, the big pass)
    select_prior_balanced_panels           ->  hardest menu passing the prior-balance gates
    build_frozen_codebook_manifest(panel_selections=...) -> recalibrated codebook

Nothing here inspects labels or demonstration behavior: menu selection is a function of
option descriptions and the blind reconstructor only. The recalibrated codebook is written
to a new path; the original is never mutated.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from ..config import ImplementerConfig
from ..vllm_backend import make_judge_backend, model_revision_id
from .cr3_reconstruction_values import (
    build_codebook_panel_plan,
    build_frozen_codebook_manifest,
    build_task_centralness_reference_plan,
    score_codebook_panel_priors,
    score_task_centralness_reference,
    select_prior_balanced_panels,
)

# Frozen prior-balance gates (identical defaults to the mining loop / library).
MAX_OPTION_PROBABILITY = 0.35
TARGET_PROBABILITY_TOLERANCE = 0.10
MIN_NORMALIZED_ENTROPY = 0.90


def _reconstructor(model: str, *, fake: bool):
    cfg = ImplementerConfig()
    cfg.vllm_fake = bool(fake)
    cfg.vllm_gpu_mem_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.90"))
    cfg.vllm_tp_size = int(os.environ.get("VLLM_TP_SIZE", "1"))
    cfg.vllm_max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192"))
    if os.environ.get("METRIC_IMPLEMENTER_LFS_HOME"):
        cfg.vllm_lfs_home = os.environ["METRIC_IMPLEMENTER_LFS_HOME"]
    overrides = json.loads(os.environ.get("V14_MODEL_PATH_OVERRIDES_JSON", "{}"))
    runtime_model = str(overrides.get(str(model), str(model)))
    backend = make_judge_backend(runtime_model, cfg, 0.0)
    revision = str(model) if fake else model_revision_id(runtime_model)
    return backend, revision


def _bootstrap_paths(codebook: dict) -> list[str]:
    metrics = codebook["metrics"]
    return [str(metrics[key]["bootstrap_path"]) for key in sorted(metrics)]


def recalibrate_codebook(
    codebook: dict,
    reconstructor,
    *,
    reconstructor_model: str,
    reconstructor_revision: str,
    bootstrap_paths: Sequence[str] | None = None,
    n_draws: int = 24,
    query_batch_size: int = 512,
    centralness_candidate_pool_size: int = 32,
    centralness_fallback_panels_per_target: int = 64,
    candidate_pool_size: int = 16,
    max_panels_per_target: int = 256,
) -> dict:
    """Return a recalibrated codebook manifest plus a per-target balance report."""
    n_options = int(codebook["n_options"])
    design_size = len(list(codebook["design_indices"]))
    seed = int(codebook["design_seed"])
    min_dis = int(codebook["min_design_disagreements"])
    noun = str(codebook["reconstruction_noun"])
    max_chars = int(codebook["reconstruction_max_chars"])
    paths = list(bootstrap_paths) if bootstrap_paths is not None else _bootstrap_paths(codebook)
    targets = [k for k, e in codebook["entries"].items() if e.get("valid")]

    reference_plan = build_task_centralness_reference_plan(paths, seed=seed)
    centralness_calibration = score_task_centralness_reference(
        reconstructor, reference_plan=reference_plan, noun=noun,
        query_batch_size=query_batch_size,
        reconstructor_model=reconstructor_model,
        reconstructor_revision=reconstructor_revision,
    )
    panel_plan = build_codebook_panel_plan(
        paths,
        centralness_reference_plan=reference_plan,
        centralness_calibration=centralness_calibration,
        target_metric_keys=targets,
        n_options=n_options, design_size=design_size,
        min_design_disagreements=min_dis, seed=seed,
        candidate_pool_size=candidate_pool_size,
        max_panels_per_target=max_panels_per_target,
        centralness_candidate_pool_size=centralness_candidate_pool_size,
        centralness_fallback_panels_per_target=centralness_fallback_panels_per_target,
    )
    prior_calibration = score_codebook_panel_priors(
        reconstructor, panel_plan=panel_plan, noun=noun, n_draws=n_draws,
        query_batch_size=query_batch_size,
        reconstructor_model=reconstructor_model,
        reconstructor_revision=reconstructor_revision,
    )
    selections = select_prior_balanced_panels(
        panel_plan, prior_calibration,
        maximum_option_probability=MAX_OPTION_PROBABILITY,
        target_probability_tolerance=TARGET_PROBABILITY_TOLERANCE,
        minimum_normalized_entropy=MIN_NORMALIZED_ENTROPY,
    )
    recalibrated = build_frozen_codebook_manifest(
        paths, n_options=n_options, design_size=design_size,
        min_design_disagreements=min_dis, seed=seed,
        panel_selections=selections, freeze_teaching_panels=True,
        reconstruction_noun=noun, reconstruction_max_chars=max_chars,
    )
    report = _balance_report(selections, targets)
    return {"codebook": recalibrated, "selections": selections, "report": report}


def _balance_report(selections: dict, targets: Sequence[str]) -> dict:
    passing = 0
    per_target = {}
    for key in targets:
        sel = selections.get(key, {})
        pc = sel.get("prior_calibration", {})
        prior = pc.get("prior", {})
        target_prob = None
        canonical = prior.get("canonical_mean_prior")
        if isinstance(canonical, (list, tuple)) and canonical:
            target_prob = float(canonical[0])
        ok = bool(pc.get("passes_prior_balance"))
        passing += int(ok)
        per_target[key] = {
            "passes_prior_balance": ok,
            "blind_target_probability": target_prob,
            "distractor_metric_keys": list(sel.get("distractor_metric_keys", [])),
        }
    return {
        "n_targets": len(targets),
        "n_passing_prior_balance": passing,
        "per_target": per_target,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebook", required=True, help="path to the source MCQ codebook JSON")
    parser.add_argument("--out", required=True, help="path for the recalibrated codebook JSON")
    parser.add_argument("--reconstructor-model", required=True)
    parser.add_argument("--fake-backend", action="store_true")
    parser.add_argument("--n-draws", type=int, default=24)
    parser.add_argument("--query-batch-size", type=int, default=512)
    args = parser.parse_args(argv)

    codebook = json.loads(Path(args.codebook).read_text())
    reconstructor, revision = _reconstructor(args.reconstructor_model, fake=args.fake_backend)
    result = recalibrate_codebook(
        codebook, reconstructor,
        reconstructor_model=args.reconstructor_model,
        reconstructor_revision=revision,
        n_draws=args.n_draws, query_batch_size=args.query_batch_size,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result["codebook"], sort_keys=True, separators=(",", ":")))
    report_path = out.with_suffix(".balance_report.json")
    report_path.write_text(json.dumps(result["report"], indent=2))
    rep = result["report"]
    print(f"recalibrated {args.codebook}")
    print(f"  targets: {rep['n_targets']}  passing prior-balance: {rep['n_passing_prior_balance']}")
    print(f"  codebook -> {out}")
    print(f"  report   -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
