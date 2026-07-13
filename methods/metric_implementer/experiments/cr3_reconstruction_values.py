"""Frozen-codebook Reconstruction-MCQ value marks for CR-3 prompt audits.

This module is the bridge between executor behavior discovery and the primary
anchor-free reconstruction objective. It has two responsibilities:

1. use bootstrap-only executor behavior to freeze one target-containing metric
   codebook per metric, before prompt-value search;
2. value every row in a scored CR-3 pool/audit artifact with the same repaired
   Reconstruction-MCQ protocol.

No anchor, silver label, human label, outcome, or archival proxy is accepted by any
interface in this module. The resulting bounded values feed
``cr_audit.prompt_articulation_certificate`` as supplied value marks. Behavior
signatures continue to define capture species. The deterministic-logit path additionally
enforces that MCQ value is a function of the exact hard annotation behavior; sampled
fallbacks retain the all-draw finite-horizon bound without that support-level promotion.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
import sqlite3
from typing import Mapping, Sequence

import numpy as np

from ..recon_channel import (
    CLONE_CAP,
    _kappa,
    mcq_no_demo_choice_probabilities,
    mcq_logit_values_from_precomputed_behaviors,
    mcq_value_from_precomputed_behavior,
)
CODEBOOK_SCHEMA = "cr3-reconstruction-codebook-v3"
PANEL_PLAN_SCHEMA = "cr3-reconstruction-panel-plan-v1"
PRIOR_CALIBRATION_SCHEMA = "cr3-reconstruction-prior-calibration-v1"
VALUE_SCHEMA = "cr3-reconstruction-values-v4"


class CachedChoiceReconstructor:
    """Persistent exact cache for normalized MCQ choice-probability queries.

    vLLM kernels can differ at the last floating-point bits across processes. The
    operational reconstruction functional must nevertheless assign one immutable value
    to one frozen teaching transcript. This cache keys the full rendered query, seed,
    declared choices, model revision, and protocol, then reuses the first finite row.
    """

    SCHEMA = "cr3-choice-probability-cache-v2"

    def __init__(self, backend, path: str | Path, *, model: str, revision: str):
        self.backend = backend
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model = str(model)
        self.revision = str(revision)
        self.choice_readout_id = str(getattr(backend, "choice_readout_id", ""))
        if not self.choice_readout_id:
            raise ValueError("choice-probability cache requires an explicit backend readout id")
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS choice_rows ("
            "cache_key TEXT PRIMARY KEY, probabilities_json TEXT NOT NULL) WITHOUT ROWID")
        self.connection.commit()
        self.rows = {
            str(key): json.loads(str(payload))
            for key, payload in self.connection.execute(
                "SELECT cache_key, probabilities_json FROM choice_rows")
        }

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def _key(self, prompt: str, choices: Sequence[str], system: str | None, seed: int) -> str:
        payload = {
            "schema": self.SCHEMA,
            "model": self.model,
            "revision": self.revision,
            "choice_readout_id": self.choice_readout_id,
            "prompt": str(prompt),
            "choices": [str(choice) for choice in choices],
            "system": system,
            "seed": int(seed),
        }
        return _payload_sha256(payload)

    def score_choices(self, prompts, choices, system=None, seed=0):
        prompts = [str(prompt) for prompt in prompts]
        if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
            seeds = [int(value) for value in seed]
            if len(seeds) != len(prompts):
                raise ValueError("choice-cache seed count does not match prompt count")
        else:
            seeds = [int(seed)] * len(prompts)
        keys = [self._key(prompt, choices, system, item_seed)
                for prompt, item_seed in zip(prompts, seeds)]
        missing_keys = []
        missing_prompts = []
        missing_seeds = []
        seen_missing = set()
        for key, prompt, item_seed in zip(keys, prompts, seeds):
            if key not in self.rows and key not in seen_missing:
                seen_missing.add(key)
                missing_keys.append(key)
                missing_prompts.append(prompt)
                missing_seeds.append(item_seed)
        if missing_keys:
            observed = np.asarray(self.backend.score_choices(
                missing_prompts, choices, system=system, seed=missing_seeds), float)
            if (observed.shape != (len(missing_keys), len(choices))
                    or np.any(~np.isfinite(observed)) or np.any(observed < 0.0)
                    or np.any(observed.sum(axis=1) <= 0.0)):
                raise RuntimeError("cannot cache an invalid MCQ choice-probability batch")
            observed = observed / observed.sum(axis=1, keepdims=True)
            inserts = []
            for key, row in zip(missing_keys, observed):
                values = row.tolist()
                self.rows[key] = values
                inserts.append((key, json.dumps(values, separators=(",", ":"))))
            self.connection.executemany(
                "INSERT OR IGNORE INTO choice_rows(cache_key, probabilities_json) VALUES (?, ?)",
                inserts,
            )
            self.connection.commit()
            # A concurrent worker may have won INSERT OR IGNORE. Reload the committed rows so
            # every process uses the same first-writer value rather than its private observation.
            for key in missing_keys:
                stored = self.connection.execute(
                    "SELECT probabilities_json FROM choice_rows WHERE cache_key = ?", (key,)
                ).fetchone()
                if stored is None:
                    raise RuntimeError("choice-probability cache commit lost a requested row")
                values = np.asarray(json.loads(str(stored[0])), float)
                if (values.shape != (len(choices),) or np.any(~np.isfinite(values))
                        or np.any(values < 0.0) or values.sum() <= 0.0):
                    raise RuntimeError("choice-probability cache contains an invalid committed row")
                self.rows[key] = (values / values.sum()).tolist()
        return [list(self.rows[key]) for key in keys]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _bootstrap(path: str | Path) -> dict:
    source = Path(path).resolve()
    z = np.load(source, allow_pickle=True)
    required = {
        "sigs", "texts", "target", "metric_description", "probe_texts",
        "probe_sha256", "executor_model", "executor_model_revision", "readout_id",
    }
    missing = sorted(required.difference(z.files))
    if missing:
        raise ValueError(f"bootstrap {source} lacks {missing}")
    target = np.asarray(z["target"], float)
    probes = [str(value) for value in z["probe_texts"]]
    if target.ndim != 1 or len(target) != len(probes) or np.any(~np.isfinite(target)):
        raise ValueError(f"bootstrap {source} has an invalid target/probe panel")
    key = source.parents[1].name if source.parent.name == "bootstrap" else source.stem
    return {
        "key": key,
        "path": str(source),
        "sha256": file_sha256(source),
        "description": str(z["metric_description"]),
        "target": target,
        "probe_texts": probes,
        "probe_sha256": str(z["probe_sha256"]),
        "executor_model": str(z["executor_model"]),
        "executor_model_revision": str(z["executor_model_revision"]),
        "readout_id": str(z["readout_id"]),
    }


def build_frozen_codebook_manifest(
    bootstrap_paths: Sequence[str | Path],
    *,
    n_options: int = 4,
    design_size: int = 120,
    min_design_disagreements: int = 2,
    seed: int = 0,
    panel_selections: Mapping[str, Mapping[str, object]] | None = None,
) -> dict:
    """Freeze related, empirically distinguishable options before prompt-value search.

    ``panel_selections`` may supply a no-demo-calibrated panel chosen solely from the
    bootstrap-derived panel plan. It cannot depend on any candidate prompt behavior.
    """
    views = [_bootstrap(path) for path in bootstrap_paths]
    if len(views) < n_options or n_options < 2:
        raise ValueError("need at least n_options bootstrap metrics and n_options >= 2")
    if len({view["key"] for view in views}) != len(views):
        raise ValueError("bootstrap metric keys must be unique")
    namespaces = {
        (view["probe_sha256"], view["executor_model_revision"], view["readout_id"])
        for view in views
    }
    if len(namespaces) != 1:
        raise ValueError("all metrics in one codebook manifest must share probe/executor/readout")
    n_probes = len(views[0]["target"])
    if any(len(view["target"]) != n_probes for view in views):
        raise ValueError("bootstrap targets are not aligned")
    if not 2 <= design_size <= n_probes:
        raise ValueError("design_size must lie in [2, n_probes]")
    if min_design_disagreements < 1:
        raise ValueError("min_design_disagreements must be positive")
    design_indices = np.random.default_rng(seed).permutation(n_probes)[:design_size]

    entries = {}
    for target in views:
        target_hard = (target["target"][design_indices] > 0.5).astype(float)
        candidates = []
        for candidate in views:
            if candidate["key"] == target["key"]:
                continue
            candidate_hard = (candidate["target"][design_indices] > 0.5).astype(float)
            disagreement = target_hard != candidate_hard
            kappa = _kappa(target_hard, candidate_hard)
            if not np.isfinite(kappa):
                continue
            candidates.append({
                "metric_key": candidate["key"],
                "kappa": float(kappa),
                "n_disagree": int(disagreement.sum()),
                "design_yes_rate": float(np.mean(candidate_hard)),
                "n_target1_candidate0": int(np.sum(
                    (target_hard == 1) & (candidate_hard == 0))),
                "n_target0_candidate1": int(np.sum(
                    (target_hard == 0) & (candidate_hard == 1))),
            })
        eligible = [candidate for candidate in candidates
                    if candidate["kappa"] < CLONE_CAP
                    and candidate["n_disagree"] >= min_design_disagreements]
        eligible.sort(key=lambda candidate: (
            -candidate["kappa"], -candidate["n_disagree"], candidate["metric_key"]))
        selection = ((panel_selections or {}).get(target["key"]) or {})
        selected_keys = [str(value) for value in selection.get("distractor_metric_keys", [])]
        if selected_keys:
            by_key = {candidate["metric_key"]: candidate for candidate in eligible}
            if (len(selected_keys) != n_options - 1 or len(set(selected_keys)) != len(selected_keys)
                    or any(key not in by_key for key in selected_keys)):
                raise ValueError(f"invalid calibrated panel selection for {target['key']}")
            selected = [by_key[key] for key in selected_keys]
        else:
            selected = eligible[:n_options - 1]
        prior_calibration = selection.get("prior_calibration")
        entries[target["key"]] = {
            "target_metric_key": target["key"],
            "target_description": target["description"],
            "target_design_yes_rate": float(np.mean(target_hard)),
            "distractor_metric_keys": [candidate["metric_key"] for candidate in selected],
            "distractor_design_statistics": selected,
            "eligible_distractor_statistics": eligible,
            "selected_distractor_kappa_min": (
                float(min(candidate["kappa"] for candidate in selected)) if selected else None),
            "selected_distractor_kappa_mean": (
                float(np.mean([candidate["kappa"] for candidate in selected]))
                if selected else None),
            "selected_distractor_disagreements_min": (
                int(min(candidate["n_disagree"] for candidate in selected)) if selected else None),
            "selection_method": (
                "blind_no_demo_prior_balance_then_behavioral_hardness"
                if selected_keys else "behavioral_hardness_only"),
            "prior_calibration": prior_calibration,
            "valid": len(selected) == n_options - 1,
            "failure": (None if len(selected) == n_options - 1 else
                        f"only {len(selected)} eligible distractors for {n_options - 1} required"),
        }

    core = {
        "schema": CODEBOOK_SCHEMA,
        "n_options": int(n_options),
        "design_indices": design_indices.astype(int).tolist(),
        "design_seed": int(seed),
        "min_design_disagreements": int(min_design_disagreements),
        "probe_sha256": views[0]["probe_sha256"],
        "executor_model": views[0]["executor_model"],
        "executor_model_revision": views[0]["executor_model_revision"],
        "readout_id": views[0]["readout_id"],
        "metrics": {
            view["key"]: {
                "bootstrap_path": view["path"],
                "bootstrap_sha256": view["sha256"],
                "description": view["description"],
            }
            for view in views
        },
        "entries": entries,
        "premises": {
            "built_from_bootstrap_only": True,
            "frozen_before_prompt_value_search": True,
            "prior_calibration_used_no_candidate_prompt_annotations": bool(panel_selections),
            "uses_external_labels": False,
        },
    }
    return {**core, "manifest_sha256": _payload_sha256(core)}


def build_codebook_panel_plan(
    bootstrap_paths: Sequence[str | Path],
    *,
    target_metric_keys: Sequence[str],
    n_options: int = 4,
    design_size: int = 120,
    min_design_disagreements: int = 2,
    seed: int = 0,
    candidate_pool_size: int = 16,
    max_panels_per_target: int = 256,
) -> dict:
    """Enumerate hard candidate menus using bootstrap behavior only."""
    if candidate_pool_size < n_options - 1 or max_panels_per_target < 1:
        raise ValueError("panel search needs enough candidates and a positive panel budget")
    base = build_frozen_codebook_manifest(
        bootstrap_paths,
        n_options=n_options,
        design_size=design_size,
        min_design_disagreements=min_design_disagreements,
        seed=seed,
    )
    targets = [str(value) for value in target_metric_keys]
    if len(set(targets)) != len(targets):
        raise ValueError("target_metric_keys must be unique")
    panels = {}
    for target_key in targets:
        if target_key not in base["entries"]:
            raise ValueError(f"panel target {target_key!r} is absent from the candidate bank")
        entry = base["entries"][target_key]
        eligible = list(entry["eligible_distractor_statistics"])[:candidate_pool_size]
        combinations = []
        for combo in itertools.combinations(eligible, n_options - 1):
            keys = [str(candidate["metric_key"]) for candidate in combo]
            behavior = {
                "selected_distractor_kappa_min": float(min(c["kappa"] for c in combo)),
                "selected_distractor_kappa_mean": float(np.mean([c["kappa"] for c in combo])),
                "selected_distractor_disagreements_min": int(
                    min(c["n_disagree"] for c in combo)),
            }
            combinations.append({
                "distractor_metric_keys": keys,
                "option_metric_keys": [target_key, *keys],
                "option_descriptions": [
                    base["metrics"][key]["description"] for key in [target_key, *keys]
                ],
                "behavioral_panel_statistics": behavior,
            })
        combinations.sort(key=lambda panel: (
            -panel["behavioral_panel_statistics"]["selected_distractor_kappa_min"],
            -panel["behavioral_panel_statistics"]["selected_distractor_kappa_mean"],
            -panel["behavioral_panel_statistics"]["selected_distractor_disagreements_min"],
            panel["distractor_metric_keys"],
        ))
        kept = combinations[:max_panels_per_target]
        for panel in kept:
            panel["panel_id"] = _payload_sha256({
                "target_metric_key": target_key,
                "distractor_metric_keys": panel["distractor_metric_keys"],
            })
        panels[target_key] = kept

    core = {
        "schema": PANEL_PLAN_SCHEMA,
        "base_codebook_manifest_sha256": base["manifest_sha256"],
        "n_options": int(n_options),
        "target_metric_keys": targets,
        "candidate_pool_size": int(candidate_pool_size),
        "max_panels_per_target": int(max_panels_per_target),
        "panels": panels,
        "premises": {
            "uses_bootstrap_behavior_only": True,
            "uses_candidate_prompt_annotations": False,
            "uses_external_labels": False,
        },
    }
    return {**core, "plan_sha256": _payload_sha256(core)}


def score_codebook_panel_priors(
    reconstructor,
    *,
    panel_plan: Mapping[str, object],
    noun: str,
    n_draws: int = 4,
    query_batch_size: int = 512,
    reconstructor_model: str,
    reconstructor_revision: str,
) -> dict:
    """Measure blind no-demo menu priors; candidate annotations are unavailable here."""
    plan = dict(panel_plan)
    observed_sha = str(plan.pop("plan_sha256", ""))
    if plan.get("schema") != PANEL_PLAN_SCHEMA or observed_sha != _payload_sha256(plan):
        raise ValueError("invalid or mutated MCQ panel plan")
    rows = {}
    for target_key in plan["target_metric_keys"]:
        target_rows = []
        for panel in plan["panels"][target_key]:
            prior = mcq_no_demo_choice_probabilities(
                reconstructor,
                noun=str(noun),
                option_descriptions=panel["option_descriptions"],
                n_draws=n_draws,
                query_batch_size=query_batch_size,
            )
            target_rows.append({
                "panel_id": panel["panel_id"],
                "target_metric_key": target_key,
                "distractor_metric_keys": panel["distractor_metric_keys"],
                "behavioral_panel_statistics": panel["behavioral_panel_statistics"],
                "prior": prior,
            })
        rows[target_key] = target_rows
    core = {
        "schema": PRIOR_CALIBRATION_SCHEMA,
        "panel_plan_sha256": observed_sha,
        "reconstructor_model": str(reconstructor_model),
        "reconstructor_revision": str(reconstructor_revision),
        "choice_readout_id": str(getattr(reconstructor, "choice_readout_id", "unverified")),
        "noun": str(noun),
        "n_draws": int(n_draws),
        "rows": rows,
        "premises": {
            "blind_unlabeled_menu": True,
            "position_counterbalanced": True,
            "uses_candidate_prompt_annotations": False,
            "uses_external_labels": False,
        },
    }
    return {**core, "calibration_sha256": _payload_sha256(core)}


def select_prior_balanced_panels(
    panel_plan: Mapping[str, object],
    prior_calibration: Mapping[str, object],
    *,
    maximum_option_probability: float = 0.35,
    target_probability_tolerance: float = 0.10,
    minimum_normalized_entropy: float = 0.90,
) -> dict[str, dict]:
    """Choose the hardest panel satisfying predeclared full-posterior prior gates.

    If no panel passes, retain the least-violating panel for a formal-only result and
    mark the failure explicitly. This keeps the full denominator without laundering an
    instrument failure into an articulability conclusion.
    """
    plan = dict(panel_plan)
    plan_sha = str(plan.pop("plan_sha256", ""))
    calibration = dict(prior_calibration)
    calibration_sha = str(calibration.pop("calibration_sha256", ""))
    if (plan.get("schema") != PANEL_PLAN_SCHEMA or plan_sha != _payload_sha256(plan)
            or calibration.get("schema") != PRIOR_CALIBRATION_SCHEMA
            or calibration_sha != _payload_sha256(calibration)
            or calibration.get("panel_plan_sha256") != plan_sha):
        raise ValueError("panel plan and prior calibration are invalid or mismatched")
    if (not 0.0 < maximum_option_probability <= 1.0
            or not 0.0 <= target_probability_tolerance <= 1.0
            or not 0.0 <= minimum_normalized_entropy <= 1.0):
        raise ValueError("invalid prior-balance thresholds")
    chance = 1.0 / int(plan["n_options"])
    thresholds = {
        "maximum_option_probability": float(maximum_option_probability),
        "target_probability_interval": [
            float(max(0.0, chance - target_probability_tolerance)),
            float(min(1.0, chance + target_probability_tolerance)),
        ],
        "minimum_normalized_entropy": float(minimum_normalized_entropy),
    }
    selections = {}
    for target_key in plan["target_metric_keys"]:
        rows = [dict(row) for row in calibration["rows"].get(target_key, [])]
        if not rows:
            raise ValueError(f"no prior calibration rows for {target_key}")
        for row in rows:
            prior = row["prior"]
            target_probability = float(prior["target_probability"])
            max_probability = float(prior["maximum_option_probability"])
            entropy = float(prior["normalized_entropy"])
            violations = {
                "maximum_option_probability_excess": float(max(
                    0.0, max_probability - maximum_option_probability)),
                "target_probability_distance_excess": float(max(
                    0.0, abs(target_probability - chance) - target_probability_tolerance)),
                "normalized_entropy_shortfall": float(max(
                    0.0, minimum_normalized_entropy - entropy)),
            }
            row["prior_balance_violations"] = violations
            row["passes_prior_balance"] = all(value <= 1e-12 for value in violations.values())
            row["total_prior_balance_violation"] = float(sum(violations.values()))

        passing = [row for row in rows if row["passes_prior_balance"]]
        candidates = passing or rows
        candidates.sort(key=lambda row: (
            0.0 if row["passes_prior_balance"] else row["total_prior_balance_violation"],
            -row["behavioral_panel_statistics"]["selected_distractor_kappa_min"],
            -row["behavioral_panel_statistics"]["selected_distractor_kappa_mean"],
            -row["behavioral_panel_statistics"]["selected_distractor_disagreements_min"],
            row["panel_id"],
        ))
        chosen = candidates[0]
        selections[target_key] = {
            "distractor_metric_keys": list(chosen["distractor_metric_keys"]),
            "prior_calibration": {
                "panel_id": chosen["panel_id"],
                "passes_prior_balance": bool(chosen["passes_prior_balance"]),
                "prior": chosen["prior"],
                "violations": chosen["prior_balance_violations"],
                "thresholds": thresholds,
                "n_panels_evaluated": len(rows),
                "n_panels_passing": len(passing),
                "calibration_sha256": calibration_sha,
            },
        }
    return selections


def validate_codebook_manifest(manifest: Mapping[str, object]) -> None:
    payload = dict(manifest)
    observed = str(payload.pop("manifest_sha256", ""))
    if payload.get("schema") != CODEBOOK_SCHEMA or observed != _payload_sha256(payload):
        raise ValueError("invalid or mutated Reconstruction-MCQ codebook manifest")
    premises = payload.get("premises") or {}
    if not premises.get("built_from_bootstrap_only") or not premises.get(
            "frozen_before_prompt_value_search") or premises.get("uses_external_labels"):
        raise ValueError("codebook manifest does not satisfy the anchor-free freezing contract")


def _load_scored_rows(path: str | Path, *, expected_probe_sha256: str) -> dict:
    source = Path(path).resolve()
    z = np.load(source, allow_pickle=True)
    if "sigs" not in z or "texts" not in z or "probe_sha256" not in z:
        raise ValueError(f"scored artifact {source} lacks sigs/texts/probe hash")
    signatures = np.asarray(z["sigs"], float)
    texts = [str(value) for value in z["texts"]]
    if (signatures.ndim != 2 or len(signatures) != len(texts)
            or np.any(~np.isfinite(signatures))):
        raise ValueError(f"scored artifact {source} has invalid rows")
    if str(z["probe_sha256"]) != expected_probe_sha256:
        raise ValueError("scored artifact probe panel differs from frozen codebook")
    return {
        "path": str(source),
        "sha256": file_sha256(source),
        "signatures": signatures,
        "texts": texts,
        "families": ([str(value) for value in z["families"]]
                     if "families" in z else None),
    }


def evaluate_scored_prompt_values(
    reconstructor,
    *,
    codebook_manifest: Mapping[str, object],
    target_metric_key: str,
    scored_path: str | Path,
    noun: str,
    n_examples: int = 8,
    n_reconstruction_draws: int = 4,
    max_chars: int = 600,
    choice_readout: str = "auto",
    query_batch_size: int = 512,
    fixed_no_demo_canonical_probabilities: np.ndarray | None = None,
    choice_probabilities_content_cached: bool = False,
) -> dict:
    """Measure every scored prompt row; no row is dropped for low/degenerate behavior."""
    validate_codebook_manifest(codebook_manifest)
    target_metric_key = str(target_metric_key)
    entries = codebook_manifest["entries"]
    if target_metric_key not in entries or not entries[target_metric_key]["valid"]:
        raise ValueError(f"target metric {target_metric_key!r} lacks a valid frozen codebook")
    metric_meta = codebook_manifest["metrics"]
    target_bootstrap = _bootstrap(metric_meta[target_metric_key]["bootstrap_path"])
    if target_bootstrap["sha256"] != metric_meta[target_metric_key]["bootstrap_sha256"]:
        raise ValueError("target bootstrap changed after codebook freezing")
    entry = entries[target_metric_key]
    distractors = []
    for distractor_key in entry["distractor_metric_keys"]:
        metadata = metric_meta[distractor_key]
        view = _bootstrap(metadata["bootstrap_path"])
        if view["sha256"] != metadata["bootstrap_sha256"]:
            raise ValueError("distractor bootstrap changed after codebook freezing")
        distractors.append({
            "metric_id": distractor_key,
            "description": view["description"],
            "body": view["description"],
            "scores": view["target"],
        })
    scored = _load_scored_rows(
        scored_path, expected_probe_sha256=str(codebook_manifest["probe_sha256"]))
    if scored["signatures"].shape[1] != len(target_bootstrap["probe_texts"]):
        raise ValueError("scored prompt signatures are not aligned to bootstrap probes")

    cache = {}
    row_details = None
    design_indices = np.asarray(codebook_manifest["design_indices"], int)
    use_batched_logits = choice_readout == "logits" or (
        choice_readout == "auto" and callable(getattr(reconstructor, "score_choices", None)))
    if use_batched_logits:
        try:
            row_details = mcq_logit_values_from_precomputed_behaviors(
                reconstructor,
                noun=noun,
                candidate_prompt_texts=scored["texts"],
                target_metric_id=target_metric_key,
                target_description=entry["target_description"],
                target_score_rows=scored["signatures"],
                probe_texts=target_bootstrap["probe_texts"],
                distractors=distractors,
                design_indices=design_indices,
                codebook_frozen_before_prompt_search=True,
                n_examples=n_examples,
                n_reconstruction_draws=n_reconstruction_draws,
                max_chars=max_chars,
                query_batch_size=query_batch_size,
                fixed_no_demo_canonical_probabilities=(
                    fixed_no_demo_canonical_probabilities),
            )
        except Exception:
            if choice_readout == "logits":
                raise
            row_details = None
    if row_details is None:
        row_details = []
        for text, signature in zip(scored["texts"], scored["signatures"]):
            key = (text, signature.tobytes())
            if key not in cache:
                cache[key] = mcq_value_from_precomputed_behavior(
                    reconstructor,
                    noun=noun,
                    candidate_prompt_text=text,
                    target_metric_id=target_metric_key,
                    target_description=entry["target_description"],
                    target_scores=signature,
                    probe_texts=target_bootstrap["probe_texts"],
                    distractors=distractors,
                    design_indices=design_indices,
                    codebook_frozen_before_prompt_search=True,
                    n_examples=n_examples,
                    n_reconstruction_draws=n_reconstruction_draws,
                    max_chars=max_chars,
                    choice_readout=choice_readout,
                )
            row_details.append(cache[key])
    row_values = [float(detail["value_mark"]) for detail in row_details]
    values = np.asarray(row_values, float)
    if values.shape != (len(scored["texts"]),) or np.any(~np.isfinite(values)):
        raise RuntimeError("Reconstruction-MCQ value evaluation did not cover every prompt row")
    caps = np.asarray([detail["value_cap"] for detail in row_details], float)
    no_demo_scores = np.asarray([
        detail["identification"]["no_demonstration_score"] for detail in row_details
    ], float)
    if (np.any(~np.isfinite(caps)) or np.any(~np.isfinite(no_demo_scores))
            or not np.allclose(caps, caps[0], rtol=0.0, atol=1e-12)
            or not np.allclose(no_demo_scores, no_demo_scores[0], rtol=0.0, atol=1e-12)):
        raise RuntimeError(
            "the frozen no-demonstration control changed across prompt candidates")
    value_cap = float(caps[0])
    if np.any(values > value_cap + 1e-12):
        raise RuntimeError("Reconstruction-MCQ values exceed their frozen-control global cap")
    readout_kinds = {
        str(detail["identification"]["readout_kind"]) for detail in row_details
    }
    value_determined_by_exact_behavior = (
        readout_kinds == {"normalized_choice_logits"}
        and bool(choice_probabilities_content_cached))
    if value_determined_by_exact_behavior:
        by_behavior: dict[bytes, float] = {}
        for signature, value in zip(scored["signatures"], values):
            behavior = (np.asarray(signature, float) > 0.5).astype(np.uint8).tobytes()
            previous = by_behavior.setdefault(behavior, float(value))
            if not np.isclose(previous, value, rtol=0.0, atol=1e-12):
                raise RuntimeError(
                    "identical hard annotation behavior produced inconsistent deterministic MCQ values")
    return {
        "schema": VALUE_SCHEMA,
        "target_metric_key": target_metric_key,
        "source_scored_path": scored["path"],
        "source_scored_sha256": scored["sha256"],
        "codebook_manifest_sha256": codebook_manifest["manifest_sha256"],
        "values": values,
        "raw_target_option_probability": np.asarray([
            detail["raw_target_option_probability"] for detail in row_details], float),
        "details": row_details,
        "families": scored["families"],
        "value_name": "annotation-attributable Reconstruction-MCQ target-option lift",
        "value_unit": "probability",
        "value_cap": value_cap,
        "choice_readout_id": str(getattr(reconstructor, "choice_readout_id", "unverified")),
        "no_demonstration_target_probability": float(no_demo_scores[0]),
        "fixed_no_demo_canonical_choice_probabilities": np.asarray(
            row_details[0]["identification"]["conditions"]["no_demonstrations"][
                "canonical_choice_probabilities"],
            float,
        ),
        "n_rows": len(values),
        "n_unique_prompt_behaviors_valued": len({
            (text, signature.tobytes())
            for text, signature in zip(scored["texts"], scored["signatures"])
        }),
        "batched_logit_path": bool(use_batched_logits and row_details
                                    and row_details[0]["identification"].get(
                                        "batched_choice_queries")),
        "premises": {
            "every_scored_row_valued": True,
            "codebook_frozen_before_prompt_search": True,
            "uses_external_labels": False,
            "no_demonstration_control_is_prompt_independent": True,
            "global_cap_is_one_minus_no_demonstration_probability": True,
            "value_determined_by_exact_behavior": value_determined_by_exact_behavior,
            "deterministic_choice_readout": value_determined_by_exact_behavior,
            "choice_probabilities_content_cached": bool(
                choice_probabilities_content_cached),
        },
    }


def write_value_artifact(path: str | Path, payload: Mapping[str, object], *,
                         reconstructor_model: str, reconstructor_revision: str) -> None:
    """Atomically persist a complete, content-addressed value transaction."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable value artifact {target}")
    if payload.get("schema") != VALUE_SCHEMA:
        raise ValueError("unexpected value payload schema")
    target.parent.mkdir(parents=True, exist_ok=True)
    details_json = json.dumps(payload["details"], sort_keys=True, separators=(",", ":"))
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(
            tmp,
            schema=np.asarray(VALUE_SCHEMA),
            target_metric_key=np.asarray(payload["target_metric_key"]),
            source_scored_path=np.asarray(payload["source_scored_path"]),
            source_scored_sha256=np.asarray(payload["source_scored_sha256"]),
            codebook_manifest_sha256=np.asarray(payload["codebook_manifest_sha256"]),
            values=np.asarray(payload["values"], float),
            raw_target_option_probability=np.asarray(
                payload["raw_target_option_probability"], float),
            details_json=np.asarray(details_json),
            families=np.asarray(payload["families"] or [], object),
            value_name=np.asarray(payload["value_name"]),
            value_unit=np.asarray(payload["value_unit"]),
            value_cap=np.asarray(payload["value_cap"], float),
            choice_readout_id=np.asarray(payload["choice_readout_id"]),
            no_demonstration_target_probability=np.asarray(
                payload["no_demonstration_target_probability"], float),
            fixed_no_demo_canonical_choice_probabilities=np.asarray(
                payload["fixed_no_demo_canonical_choice_probabilities"], float),
            reconstructor_model=np.asarray(str(reconstructor_model)),
            reconstructor_revision=np.asarray(str(reconstructor_revision)),
            premises_json=np.asarray(json.dumps(payload["premises"], sort_keys=True)),
        )
        with tmp.open("rb") as source:
            os.fsync(source.fileno())
        os.replace(tmp, target)
        directory_fd = os.open(str(target.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_value_artifact(
    path: str | Path,
    *,
    expected_source_scored_sha256: str | None = None,
    expected_codebook_manifest_sha256: str | None = None,
    expected_choice_readout_id: str | None = None,
    expected_reconstructor_model: str | None = None,
    expected_reconstructor_revision: str | None = None,
) -> dict:
    source = Path(path).resolve()
    z = np.load(source, allow_pickle=True)
    if str(z["schema"]) != VALUE_SCHEMA:
        raise ValueError(f"unexpected value artifact schema in {source}")
    choice_readout_id = str(z["choice_readout_id"])
    if (expected_choice_readout_id is not None
            and choice_readout_id != str(expected_choice_readout_id)):
        raise ValueError(f"unexpected Reconstruction-MCQ choice readout in {source}")
    reconstructor_model = str(z["reconstructor_model"])
    reconstructor_revision = str(z["reconstructor_revision"])
    if (expected_reconstructor_model is not None
            and reconstructor_model != str(expected_reconstructor_model)):
        raise ValueError(f"unexpected Reconstruction-MCQ reconstructor model in {source}")
    if (expected_reconstructor_revision is not None
            and reconstructor_revision != str(expected_reconstructor_revision)):
        raise ValueError(f"unexpected Reconstruction-MCQ reconstructor revision in {source}")
    source_sha = str(z["source_scored_sha256"])
    codebook_sha = str(z["codebook_manifest_sha256"])
    if expected_source_scored_sha256 is not None and source_sha != expected_source_scored_sha256:
        raise ValueError("value artifact does not match its scored behavior artifact")
    if expected_codebook_manifest_sha256 is not None and codebook_sha != expected_codebook_manifest_sha256:
        raise ValueError("value artifact does not match the frozen codebook")
    values = np.asarray(z["values"], float)
    raw = np.asarray(z["raw_target_option_probability"], float)
    cap = float(z["value_cap"])
    if (values.ndim != 1 or raw.shape != values.shape or np.any(~np.isfinite(values))
            or np.any(values < -1e-12) or np.any(values > cap + 1e-12)):
        raise ValueError(f"invalid bounded values in {source}")
    premises = json.loads(str(z["premises_json"]))
    if (not premises.get("every_scored_row_valued")
            or premises.get("uses_external_labels")
            or not premises.get("no_demonstration_control_is_prompt_independent")
            or not premises.get("global_cap_is_one_minus_no_demonstration_probability")):
        raise ValueError("value artifact violates the anchor-free all-draw contract")
    no_demo = float(z["no_demonstration_target_probability"])
    fixed_no_demo = np.asarray(z["fixed_no_demo_canonical_choice_probabilities"], float)
    if (not 0.0 <= no_demo <= 1.0
            or fixed_no_demo.ndim != 2 or np.any(~np.isfinite(fixed_no_demo))
            or np.any(fixed_no_demo < 0.0)
            or not np.allclose(fixed_no_demo.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
            or not np.isclose(cap, 1.0 - no_demo, rtol=0.0, atol=1e-12)
            or not np.isclose(
                no_demo, float(np.mean(fixed_no_demo[:, 0])), rtol=0.0, atol=1e-12)):
        raise ValueError("value artifact has an invalid frozen-control global cap")
    return {
        "path": str(source),
        "sha256": file_sha256(source),
        "target_metric_key": str(z["target_metric_key"]),
        "source_scored_sha256": source_sha,
        "codebook_manifest_sha256": codebook_sha,
        "values": values,
        "raw_target_option_probability": raw,
        "details": json.loads(str(z["details_json"])),
        "families": [str(value) for value in z["families"]],
        "value_name": str(z["value_name"]),
        "value_unit": str(z["value_unit"]),
        "value_cap": cap,
        "choice_readout_id": choice_readout_id,
        "no_demonstration_target_probability": no_demo,
        "fixed_no_demo_canonical_choice_probabilities": fixed_no_demo,
        "reconstructor_model": reconstructor_model,
        "reconstructor_revision": reconstructor_revision,
        "premises": premises,
    }
