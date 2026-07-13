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
    _exact_contrastive_example_indices,
    _kappa,
    mcq_no_demo_choice_probabilities,
    mcq_logit_values_from_precomputed_behaviors,
    mcq_value_from_precomputed_behavior,
)
LEGACY_CODEBOOK_SCHEMA = "cr3-reconstruction-codebook-v3"
CODEBOOK_SCHEMA = "cr3-reconstruction-codebook-v4"
PANEL_PLAN_SCHEMA = "cr3-reconstruction-panel-plan-v1"
PRIOR_CALIBRATION_SCHEMA = "cr3-reconstruction-prior-calibration-v1"
VALUE_SCHEMA = "cr3-reconstruction-values-v4"
FINITE_STATE_SCORED_SCHEMA = "cr3-reconstruction-finite-state-scored-v1"
FINITE_STATE_ENVELOPE_SCHEMA = "cr3-reconstruction-finite-state-envelope-v1"
FIXED_TEACHING_SIZE = 8


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
        # Keep only rows touched by this process. Full panel searches can create millions
        # of rows; eagerly materializing the whole SQLite cache is an avoidable OOM risk.
        self.rows = {}

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
        uncached = list(dict.fromkeys(key for key in keys if key not in self.rows))
        for start in range(0, len(uncached), 500):
            chunk = uncached[start:start + 500]
            placeholders = ",".join("?" for _ in chunk)
            for key, payload in self.connection.execute(
                    f"SELECT cache_key, probabilities_json FROM choice_rows "
                    f"WHERE cache_key IN ({placeholders})", chunk):
                values = np.asarray(json.loads(str(payload)), float)
                if (values.shape != (len(choices),) or np.any(~np.isfinite(values))
                        or np.any(values < 0.0)
                        or not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-12)):
                    raise RuntimeError("choice-probability cache contains an invalid row")
                self.rows[str(key)] = values.tolist()
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
                        or np.any(values < 0.0)
                        or not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-12)):
                    raise RuntimeError("choice-probability cache contains an invalid committed row")
                self.rows[key] = values.tolist()
        result = [list(self.rows[key]) for key in keys]
        # SQLite is the persistent cache. Retaining every row in this process would
        # eventually reproduce the eager-load OOM during a full panel sweep.
        self.rows.clear()
        return result


def import_choice_probability_cache(
    source_path: str | Path, destination_path: str | Path
) -> dict:
    """Transactionally merge an immutable cache without sharing a writable DB file."""
    source_path = Path(source_path).resolve()
    destination_path = Path(destination_path).resolve()
    if source_path == destination_path or not source_path.is_file():
        raise ValueError("choice-cache import requires distinct existing source and destination paths")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    destination = sqlite3.connect(destination_path)
    destination.execute("PRAGMA journal_mode=WAL")
    destination.execute("PRAGMA synchronous=FULL")
    destination.execute(
        "CREATE TABLE IF NOT EXISTS choice_rows ("
        "cache_key TEXT PRIMARY KEY, probabilities_json TEXT NOT NULL) WITHOUT ROWID")
    semantic = hashlib.sha256()
    n_rows = 0
    try:
        destination.execute("BEGIN IMMEDIATE")
        for key, payload in source.execute(
                "SELECT cache_key, probabilities_json FROM choice_rows ORDER BY cache_key"):
            values = np.asarray(json.loads(str(payload)), float)
            if (values.ndim != 1 or len(values) < 2 or np.any(~np.isfinite(values))
                    or np.any(values < 0.0)
                    or not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-12)):
                raise ValueError("source choice cache contains an invalid probability row")
            normalized = json.dumps(values.tolist(), separators=(",", ":"))
            semantic.update(str(key).encode("ascii"))
            semantic.update(b"\0")
            semantic.update(normalized.encode("ascii"))
            semantic.update(b"\n")
            existing = destination.execute(
                "SELECT probabilities_json FROM choice_rows WHERE cache_key=?", (str(key),)
            ).fetchone()
            if existing is None:
                destination.execute(
                    "INSERT INTO choice_rows(cache_key, probabilities_json) VALUES (?, ?)",
                    (str(key), normalized),
                )
            else:
                previous = np.asarray(json.loads(str(existing[0])), float)
                if not np.array_equal(previous, values):
                    raise ValueError(f"choice-cache collision disagrees for key {key}")
            n_rows += 1
        destination.commit()
    except Exception:
        destination.rollback()
        raise
    finally:
        source.close()
        destination.close()
    return {
        "schema": "cr3-choice-cache-import-v1",
        "source_path": str(source_path),
        "source_file_sha256": file_sha256(source_path),
        "source_rows": n_rows,
        "source_semantic_sha256": semantic.hexdigest(),
        "destination_path": str(destination_path),
        "writable_database_shared": False,
    }


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


def _binary_state_rows(k: int) -> np.ndarray:
    if k < 1 or k > 20:
        raise ValueError("finite-state enumeration requires 1 <= k <= 20")
    states = np.arange(1 << k, dtype=np.uint32)
    shifts = np.arange(k - 1, -1, -1, dtype=np.uint32)
    return ((states[:, None] >> shifts[None, :]) & 1).astype(np.uint8)


def _binary_state_integer(bits: Sequence[int]) -> int:
    value = 0
    for bit in bits:
        if int(bit) not in (0, 1):
            raise ValueError("binary state contains a non-binary entry")
        value = (value << 1) | int(bit)
    return value


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
    freeze_teaching_panels: bool = True,
    teaching_size: int = FIXED_TEACHING_SIZE,
    reconstruction_noun: str | None = None,
    reconstruction_max_chars: int = 600,
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
    if freeze_teaching_panels and teaching_size != FIXED_TEACHING_SIZE:
        raise ValueError(
            f"the bound-grade protocol freezes teaching_size={FIXED_TEACHING_SIZE}")
    if freeze_teaching_panels and (not str(reconstruction_noun or "").strip()
                                   or reconstruction_max_chars <= 0):
        raise ValueError("bound-grade codebooks require a frozen noun and positive max_chars")
    design_indices = np.random.default_rng(seed).permutation(n_probes)[:design_size]
    design_set = set(map(int, design_indices))
    teaching_candidate_indices = np.asarray(
        [index for index in range(n_probes) if index not in design_set], dtype=int)
    if freeze_teaching_panels and len(teaching_candidate_indices) < teaching_size:
        raise ValueError(
            "the complement of the panel-design split is too small for the fixed teaching panel")

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
        entry = {
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
                "blind_no_demo_prior_balance_then_finite_state_capability"
                if selection.get("state_envelope_selection") else
                "blind_no_demo_prior_balance_then_behavioral_hardness"
                if selected_keys else "behavioral_hardness_only"),
            "prior_calibration": prior_calibration,
            "state_envelope_selection": selection.get("state_envelope_selection"),
            "valid": len(selected) == n_options - 1,
            "failure": (None if len(selected) == n_options - 1 else
                        f"only {len(selected)} eligible distractors for {n_options - 1} required"),
        }
        if freeze_teaching_panels and entry["valid"]:
            selected_views = {
                candidate["key"]: candidate for candidate in views
                if candidate["key"] in entry["distractor_metric_keys"]
            }
            distractor_vectors = [
                selected_views[key]["target"][teaching_candidate_indices]
                for key in entry["distractor_metric_keys"]
            ]
            chosen_local, teaching_design = _exact_contrastive_example_indices(
                target["target"][teaching_candidate_indices],
                distractor_vectors,
                n_examples=teaching_size,
                min_disagreements=0,
                require_target_balance=False,
            )
            fixed_indices = teaching_candidate_indices[chosen_local]
            target_bits = (target["target"][fixed_indices] > 0.5).astype(np.uint8)
            item_ids = [hashlib.sha256(
                str(target["probe_texts"][index]).encode("utf-8")
            ).hexdigest()[:20] for index in fixed_indices]
            transcript = [
                {"item_id": item_id, "score": int(bit)}
                for item_id, bit in zip(item_ids, target_bits)
            ]
            entry.update({
                "fixed_teaching_indices": fixed_indices.astype(int).tolist(),
                "fixed_teaching_item_ids": item_ids,
                "fixed_teaching_target_scores": target_bits.astype(int).tolist(),
                "fixed_teaching_target_state": _binary_state_integer(target_bits),
                "fixed_teaching_target_transcript_sha256": hashlib.sha256(json.dumps(
                    transcript, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")).hexdigest(),
                "fixed_teaching_design": teaching_design,
                "fixed_teaching_selection_rule": (
                    "exact max-min contrastive design on canonical target/distractor executor "
                    "behaviors in the disjoint panel-design complement"),
            })
        entries[target["key"]] = entry

    core = {
        "schema": CODEBOOK_SCHEMA if freeze_teaching_panels else LEGACY_CODEBOOK_SCHEMA,
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
    if freeze_teaching_panels:
        core["n_probes"] = int(n_probes)
        core["reconstruction_noun"] = str(reconstruction_noun)
        core["reconstruction_max_chars"] = int(reconstruction_max_chars)
        core["fixed_teaching_protocol"] = {
            "teaching_size": int(teaching_size),
            "candidate_indices": teaching_candidate_indices.astype(int).tolist(),
            "candidate_split": "sorted complement of design_indices",
            "hard_annotation_rule": "1 iff normalized pYES > 0.5; ties map to 0",
            "same_ordered_items_for_every_prompt_candidate": True,
            "all_binary_transcripts_enumerable": True,
        }
        core["premises"].update({
            "fixed_teaching_panels_use_canonical_bootstrap_behaviors_only": True,
            "teaching_panels_disjoint_from_panel_design_split": True,
        })
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
        freeze_teaching_panels=False,
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


def prior_balanced_panel_rows(
    panel_plan: Mapping[str, object],
    prior_calibration: Mapping[str, object],
    *,
    maximum_option_probability: float = 0.35,
    target_probability_tolerance: float = 0.10,
    minimum_normalized_entropy: float = 0.90,
) -> dict:
    """Validate and annotate every candidate row with the frozen blind-prior gates."""
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
    annotated = {}
    for target_key in plan["target_metric_keys"]:
        rows = [dict(row) for row in calibration["rows"].get(target_key, [])]
        if not rows:
            raise ValueError(f"no prior calibration rows for {target_key}")
        for row in rows:
            prior = row["prior"]
            probabilities = _validated_probability_matrix(
                prior.get("canonical_choice_probabilities"),
                label=f"blind prior for {target_key}/{row.get('panel_id')}",
                n_options=int(plan["n_options"]),
            )
            mean_prior = probabilities.mean(axis=0)
            target_probability = float(mean_prior[0])
            max_probability = float(mean_prior.max())
            positive = mean_prior > 0.0
            entropy = float(
                -np.sum(mean_prior[positive] * np.log2(mean_prior[positive]))
                / np.log2(len(mean_prior)))
            total_variation = float(
                0.5 * np.abs(mean_prior - 1.0 / len(mean_prior)).sum())
            prior_checks = {
                "target_probability": target_probability,
                "maximum_option_probability": max_probability,
                "normalized_entropy": entropy,
                "total_variation_from_uniform": total_variation,
            }
            if (int(prior.get("n_draws", -1)) != probabilities.shape[0]
                    or not prior.get("position_counterbalanced")
                    or not np.allclose(
                        np.asarray(prior.get("canonical_mean_prior"), float), mean_prior,
                        rtol=0.0, atol=1e-12)
                    or any(not np.isclose(
                        float(prior.get(field, np.nan)), expected,
                        rtol=0.0, atol=1e-12)
                        for field, expected in prior_checks.items())):
                raise ValueError(
                    f"blind-prior summaries disagree with their posterior matrix for "
                    f"{target_key}/{row.get('panel_id')}")
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

        annotated[target_key] = rows
    return {
        "rows": annotated,
        "thresholds": thresholds,
        "calibration_sha256": calibration_sha,
    }


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
    ranked = prior_balanced_panel_rows(
        panel_plan,
        prior_calibration,
        maximum_option_probability=maximum_option_probability,
        target_probability_tolerance=target_probability_tolerance,
        minimum_normalized_entropy=minimum_normalized_entropy,
    )
    selections = {}
    for target_key, original_rows in ranked["rows"].items():
        rows = [dict(row) for row in original_rows]

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
                "thresholds": ranked["thresholds"],
                "n_panels_evaluated": len(rows),
                "n_panels_passing": len(passing),
                "calibration_sha256": ranked["calibration_sha256"],
            },
        }
    return selections


def select_state_capable_panels(
    panel_plan: Mapping[str, object],
    prior_calibration: Mapping[str, object],
    panel_envelopes: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    maximum_option_probability: float = 0.35,
    target_probability_tolerance: float = 0.10,
    minimum_normalized_entropy: float = 0.90,
) -> dict[str, dict]:
    """Choose maximum-range live panels after blind-prior filtering.

    Every prior-passing panel must have an exhaustive state envelope. Panels are live
    when at least one envelope maximizer has positive lift and uniquely identifies the
    target. Selection maximizes ``U_state`` among live panels, then uses behavioral
    hardness and panel ID only as deterministic tie-breaks. If no live panel exists,
    the maximum-range prior-passing panel is retained for a formal-only estimand.
    """
    ranked = prior_balanced_panel_rows(
        panel_plan,
        prior_calibration,
        maximum_option_probability=maximum_option_probability,
        target_probability_tolerance=target_probability_tolerance,
        minimum_normalized_entropy=minimum_normalized_entropy,
    )
    fallback = select_prior_balanced_panels(
        panel_plan,
        prior_calibration,
        maximum_option_probability=maximum_option_probability,
        target_probability_tolerance=target_probability_tolerance,
        minimum_normalized_entropy=minimum_normalized_entropy,
    )
    selections = {}
    for target_key, rows in ranked["rows"].items():
        passing = [dict(row) for row in rows if row["passes_prior_balance"]]
        if not passing:
            selection = dict(fallback[target_key])
            selection["state_envelope_selection"] = {
                "method": "no_prior_passing_panel; least_prior_violation_fallback",
                "passes_state_capability": False,
                "n_prior_passing_panels": 0,
                "n_live_panels": 0,
                "candidate_envelopes": [],
            }
            selections[target_key] = selection
            continue
        target_envelopes = panel_envelopes.get(target_key) or {}
        candidates = []
        for row in passing:
            panel_id = str(row["panel_id"])
            if panel_id not in target_envelopes:
                raise ValueError(
                    f"prior-passing panel {panel_id} for {target_key} lacks a state envelope")
            envelope = dict(target_envelopes[panel_id])
            if (envelope.get("target_metric_key") != target_key
                    or envelope.get("prior_panel_id") != panel_id
                    or list(envelope.get("distractor_metric_keys") or [])
                    != list(row["distractor_metric_keys"])):
                raise ValueError(f"state envelope is bound to the wrong panel for {target_key}")
            capability = envelope.get("state_envelope_capability") or {}
            candidates.append({
                **row,
                "finite_state_upper_bound": float(envelope["finite_state_upper_bound"]),
                "passes_state_capability": bool(
                    capability.get("has_positive_unique_target_maximizer")),
                "envelope_summary_sha256": str(envelope["summary_sha256"]),
                "state_function_semantic_sha256": str(
                    envelope["state_function_semantic_sha256"]),
            })
        live = [row for row in candidates if row["passes_state_capability"]]
        eligible = live or candidates
        eligible.sort(key=lambda row: (
            -row["finite_state_upper_bound"],
            -row["behavioral_panel_statistics"]["selected_distractor_kappa_min"],
            -row["behavioral_panel_statistics"]["selected_distractor_kappa_mean"],
            -row["behavioral_panel_statistics"]["selected_distractor_disagreements_min"],
            row["panel_id"],
        ))
        chosen = eligible[0]
        selections[target_key] = {
            "distractor_metric_keys": list(chosen["distractor_metric_keys"]),
            "prior_calibration": {
                "panel_id": chosen["panel_id"],
                "passes_prior_balance": True,
                "prior": chosen["prior"],
                "violations": chosen["prior_balance_violations"],
                "thresholds": ranked["thresholds"],
                "n_panels_evaluated": len(rows),
                "n_panels_passing": len(passing),
                "calibration_sha256": ranked["calibration_sha256"],
            },
            "state_envelope_selection": {
                "method": (
                    "max_U_state_among_live_prior_passing_panels; "
                    "behavioral_hardness_then_panel_id_tie_break"),
                "passes_state_capability": bool(live),
                "chosen_panel_id": chosen["panel_id"],
                "chosen_finite_state_upper_bound": chosen["finite_state_upper_bound"],
                "chosen_envelope_summary_sha256": chosen["envelope_summary_sha256"],
                "chosen_state_function_semantic_sha256": chosen[
                    "state_function_semantic_sha256"],
                "n_prior_passing_panels": len(passing),
                "n_live_panels": len(live),
                "candidate_envelopes": [{
                    "panel_id": row["panel_id"],
                    "finite_state_upper_bound": row["finite_state_upper_bound"],
                    "passes_state_capability": row["passes_state_capability"],
                    "envelope_summary_sha256": row["envelope_summary_sha256"],
                    "state_function_semantic_sha256": row[
                        "state_function_semantic_sha256"],
                } for row in sorted(candidates, key=lambda row: row["panel_id"])],
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
    if (not premises.get("fixed_teaching_panels_use_canonical_bootstrap_behaviors_only")
            or not premises.get("teaching_panels_disjoint_from_panel_design_split")):
        raise ValueError("codebook manifest lacks the fixed teaching-panel contract")
    n_probes = int(payload.get("n_probes", -1))
    protocol = payload.get("fixed_teaching_protocol") or {}
    k = int(protocol.get("teaching_size", -1))
    if (k != FIXED_TEACHING_SIZE
            or protocol.get("hard_annotation_rule")
            != "1 iff normalized pYES > 0.5; ties map to 0"
            or not protocol.get("same_ordered_items_for_every_prompt_candidate")
            or not protocol.get("all_binary_transcripts_enumerable")):
        raise ValueError("codebook manifest has an invalid fixed teaching protocol")
    if (not str(payload.get("reconstruction_noun", "")).strip()
            or int(payload.get("reconstruction_max_chars", 0)) <= 0):
        raise ValueError("codebook manifest lacks its frozen reconstruction rendering contract")
    design = [int(value) for value in payload.get("design_indices", [])]
    candidates = [int(value) for value in protocol.get("candidate_indices", [])]
    expected_candidates = [index for index in range(n_probes) if index not in set(design)]
    if (n_probes < k or len(set(design)) != len(design)
            or any(index < 0 or index >= n_probes for index in design)
            or candidates != expected_candidates):
        raise ValueError("codebook design and teaching-candidate splits are not disjoint/exhaustive")
    candidate_set = set(candidates)
    for key, entry in payload.get("entries", {}).items():
        if not entry.get("valid"):
            continue
        indices = [int(value) for value in entry.get("fixed_teaching_indices", [])]
        item_ids = [str(value) for value in entry.get("fixed_teaching_item_ids", [])]
        scores = [int(value) for value in entry.get("fixed_teaching_target_scores", [])]
        if (len(indices) != k or len(set(indices)) != k or len(item_ids) != k
                or len(scores) != k or not set(indices).issubset(candidate_set)
                or any(score not in (0, 1) for score in scores)):
            raise ValueError(f"metric {key} has an invalid fixed teaching panel")
        transcript = [
            {"item_id": item_id, "score": score}
            for item_id, score in zip(item_ids, scores)
        ]
        transcript_sha = hashlib.sha256(json.dumps(
            transcript, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        if (_binary_state_integer(scores) != int(entry.get("fixed_teaching_target_state", -1))
                or transcript_sha != entry.get("fixed_teaching_target_transcript_sha256")):
            raise ValueError(f"metric {key} has a mutated fixed target transcript")


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
    if (str(noun) != codebook_manifest["reconstruction_noun"]
            or int(max_chars) != int(codebook_manifest["reconstruction_max_chars"])):
        raise ValueError("value query rendering differs from the frozen codebook contract")
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
    design_indices = np.asarray(entry["fixed_teaching_indices"], int)
    if n_examples != len(design_indices):
        raise ValueError(
            "n_examples must equal the frozen codebook teaching size; changing it changes the estimand")
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
                fixed_teaching_panel=True,
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
                    fixed_teaching_panel=True,
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
            behavior = (np.asarray(signature, float)[design_indices] > 0.5).astype(
                np.uint8).tobytes()
            previous = by_behavior.setdefault(behavior, float(value))
            if not np.isclose(previous, value, rtol=0.0, atol=1e-12):
                raise RuntimeError(
                    "identical hard annotation behavior produced inconsistent deterministic MCQ values")
    expected_indices = design_indices.astype(int).tolist()
    for detail in row_details:
        design = detail.get("design") or {}
        if (design.get("indices_in_prompt_order") != expected_indices
                or not design.get("fixed_ordered_teaching_panel")):
            raise RuntimeError("prompt value did not use the frozen ordered teaching panel")
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
        "n_unique_fixed_binary_transcripts_valued": len({
            (np.asarray(signature, float)[design_indices] > 0.5).astype(np.uint8).tobytes()
            for signature in scored["signatures"]
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
            "value_is_function_of_fixed_binary_teaching_transcript": bool(
                value_determined_by_exact_behavior),
        },
    }


def _finite_state_instrument_sha256(
    codebook_manifest: Mapping[str, object], target_metric_key: str
) -> str:
    entry = codebook_manifest["entries"][str(target_metric_key)]
    return _payload_sha256({
        "schema": FINITE_STATE_SCORED_SCHEMA,
        "codebook_manifest_sha256": codebook_manifest["manifest_sha256"],
        "target_metric_key": str(target_metric_key),
        "fixed_teaching_indices": entry["fixed_teaching_indices"],
        "state_encoding": "unsigned big-endian binary over the stored teaching order",
        "n_states": 1 << len(entry["fixed_teaching_indices"]),
        "value_schema": VALUE_SCHEMA,
    })


def write_finite_state_scored_artifact(
    path: str | Path,
    *,
    codebook_manifest: Mapping[str, object],
    target_metric_key: str,
) -> dict:
    """Write the exhaustive binary transcript population for one frozen instrument."""
    validate_codebook_manifest(codebook_manifest)
    target_metric_key = str(target_metric_key)
    entry = codebook_manifest["entries"].get(target_metric_key)
    if not entry or not entry.get("valid"):
        raise ValueError(f"target metric {target_metric_key!r} lacks a valid codebook entry")
    bootstrap = _bootstrap(
        codebook_manifest["metrics"][target_metric_key]["bootstrap_path"])
    teaching_indices = np.asarray(entry["fixed_teaching_indices"], dtype=int)
    state_bits = _binary_state_rows(len(teaching_indices))
    state_integers = np.arange(len(state_bits), dtype=np.uint32)
    signatures = np.zeros((len(state_bits), len(bootstrap["probe_texts"])), dtype=np.uint8)
    signatures[:, teaching_indices] = state_bits
    texts = np.asarray([
        f"finite-state transcript {value:0{len(teaching_indices)}b}"
        for value in state_integers
    ], object)
    instrument_sha = _finite_state_instrument_sha256(
        codebook_manifest, target_metric_key)
    target = Path(path)
    if target.exists():
        return load_finite_state_scored_artifact(
            target,
            codebook_manifest=codebook_manifest,
            target_metric_key=target_metric_key,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(
            tmp,
            schema=np.asarray(FINITE_STATE_SCORED_SCHEMA),
            codebook_manifest_sha256=np.asarray(codebook_manifest["manifest_sha256"]),
            instrument_sha256=np.asarray(instrument_sha),
            target_metric_key=np.asarray(target_metric_key),
            teaching_indices=teaching_indices,
            state_integers=state_integers,
            state_bits=state_bits,
            sigs=signatures,
            texts=texts,
            families=np.asarray(["exhaustive_binary_state"] * len(state_bits), object),
            probe_sha256=np.asarray(codebook_manifest["probe_sha256"]),
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
    return load_finite_state_scored_artifact(
        target,
        codebook_manifest=codebook_manifest,
        target_metric_key=target_metric_key,
    )


def load_finite_state_scored_artifact(
    path: str | Path,
    *,
    codebook_manifest: Mapping[str, object],
    target_metric_key: str,
) -> dict:
    """Fail closed unless an artifact is the complete, ordered binary state table."""
    validate_codebook_manifest(codebook_manifest)
    source = Path(path).resolve()
    z = np.load(source, allow_pickle=True)
    target_metric_key = str(target_metric_key)
    entry = codebook_manifest["entries"][target_metric_key]
    teaching_indices = np.asarray(entry["fixed_teaching_indices"], dtype=int)
    expected_bits = _binary_state_rows(len(teaching_indices))
    expected_integers = np.arange(len(expected_bits), dtype=np.uint32)
    state_bits = np.asarray(z["state_bits"], dtype=np.uint8)
    state_integers = np.asarray(z["state_integers"], dtype=np.uint32)
    signatures = np.asarray(z["sigs"], dtype=float)
    if (str(z["schema"]) != FINITE_STATE_SCORED_SCHEMA
            or str(z["codebook_manifest_sha256"]) != codebook_manifest["manifest_sha256"]
            or str(z["target_metric_key"]) != target_metric_key
            or str(z["probe_sha256"]) != codebook_manifest["probe_sha256"]
            or str(z["instrument_sha256"]) != _finite_state_instrument_sha256(
                codebook_manifest, target_metric_key)
            or not np.array_equal(np.asarray(z["teaching_indices"], int), teaching_indices)
            or not np.array_equal(state_bits, expected_bits)
            or not np.array_equal(state_integers, expected_integers)
            or signatures.shape != (len(expected_bits), int(codebook_manifest["n_probes"]))
            or not np.array_equal(
                (signatures[:, teaching_indices] > 0.5).astype(np.uint8), expected_bits)):
        raise ValueError("invalid, incomplete, or mutated finite-state scored artifact")
    non_teaching = np.ones(signatures.shape[1], dtype=bool)
    non_teaching[teaching_indices] = False
    if np.any(signatures[:, non_teaching] != 0.0):
        raise ValueError("finite-state artifact has noncanonical values outside its teaching panel")
    return {
        "path": str(source),
        "sha256": file_sha256(source),
        "instrument_sha256": str(z["instrument_sha256"]),
        "target_metric_key": target_metric_key,
        "teaching_indices": teaching_indices,
        "state_integers": state_integers,
        "state_bits": state_bits,
        "n_states": len(state_bits),
    }


def build_finite_state_envelope(
    *,
    codebook_manifest: Mapping[str, object],
    target_metric_key: str,
    state_scored_path: str | Path,
    value_payload: Mapping[str, object],
) -> dict:
    """Certify the exact upper envelope over every hard transcript of fixed T_8."""
    state_table = load_finite_state_scored_artifact(
        state_scored_path,
        codebook_manifest=codebook_manifest,
        target_metric_key=target_metric_key,
    )
    target_metric_key = str(target_metric_key)
    if (value_payload.get("target_metric_key") != target_metric_key
            or value_payload.get("source_scored_sha256") != state_table["sha256"]
            or value_payload.get("codebook_manifest_sha256") != codebook_manifest["manifest_sha256"]
            or not (value_payload.get("premises") or {}).get(
                "value_is_function_of_fixed_binary_teaching_transcript")
            or not (value_payload.get("premises") or {}).get(
                "value_determined_by_exact_behavior")):
        raise ValueError("finite-state values are not bound to the exact deterministic instrument")
    values = np.asarray(value_payload["values"], dtype=float)
    details = list(value_payload["details"])
    if (values.shape != (state_table["n_states"],) or len(details) != len(values)
            or np.any(~np.isfinite(values))):
        raise ValueError("finite-state value table is incomplete")
    transcript_hashes = []
    expected_indices = state_table["teaching_indices"].astype(int).tolist()
    entry = codebook_manifest["entries"][target_metric_key]
    expected_item_ids = [str(value) for value in entry["fixed_teaching_item_ids"]]
    raw_values = np.asarray(value_payload["raw_target_option_probability"], dtype=float)
    fixed_no_demo = _validated_probability_matrix(
        value_payload["fixed_no_demo_canonical_choice_probabilities"],
        label="finite-state frozen no-demonstration block",
        n_options=int(codebook_manifest["n_options"]),
    )
    if raw_values.shape != values.shape or np.any(~np.isfinite(raw_values)):
        raise ValueError("finite-state raw-value table is incomplete")
    semantic_rows = []
    for state_integer, bits, value, raw_value, detail in zip(
            state_table["state_integers"], state_table["state_bits"],
            values, raw_values, details):
        design = detail.get("design") or {}
        if (design.get("indices_in_prompt_order") != expected_indices
                or design.get("item_ids_in_prompt_order") != expected_item_ids
                or design.get("scores_in_prompt_order") != bits.astype(int).tolist()
                or not design.get("fixed_ordered_teaching_panel")):
            raise ValueError(f"finite state {int(state_integer)} used a different teaching panel")
        transcript = [
            {"item_id": item_id, "score": int(bit)}
            for item_id, bit in zip(expected_item_ids, bits)
        ]
        expected_transcript_sha = hashlib.sha256(json.dumps(
            transcript, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        if design.get("teaching_transcript_sha256") != expected_transcript_sha:
            raise ValueError(f"finite state {int(state_integer)} has a forged transcript hash")
        derived = _validate_value_detail(
            detail,
            expected_value=float(value),
            expected_raw=float(raw_value),
            expected_no_demo=fixed_no_demo,
            expected_n_options=int(codebook_manifest["n_options"]),
            label=f"finite state {int(state_integer)}",
        )
        transcript_hashes.append(expected_transcript_sha)
        semantic_rows.append({
            "state": int(state_integer),
            "bits": bits.astype(int).tolist(),
            "transcript_sha256": expected_transcript_sha,
            "value": float(derived["value"]),
            "raw_target_option_probability": float(derived["means"]["annotations"]),
            "condition_choice_probabilities": {
                condition: derived["matrices"][condition].tolist()
                for condition in ("annotations", "no_demonstrations", "shuffled_labels")
            },
        })
    if len(set(transcript_hashes)) != state_table["n_states"]:
        raise ValueError("finite-state value table does not contain one unique transcript per state")

    upper_bound = float(np.max(values))
    coarse_cap = float(value_payload["value_cap"])
    if upper_bound > coarse_cap + 1e-12:
        raise ValueError("finite-state envelope exceeds the coarse no-demo range cap")
    maximizing_states = state_table["state_integers"][
        np.isclose(values, upper_bound, rtol=0.0, atol=1e-12)].astype(int).tolist()

    def _state_posterior(state_integer: int) -> np.ndarray:
        probabilities = np.asarray(
            details[state_integer]["identification"]["conditions"]["annotations"][
                "canonical_choice_probabilities"], dtype=float)
        if (probabilities.ndim != 2
                or probabilities.shape[1] != codebook_manifest["n_options"]
                or np.any(~np.isfinite(probabilities))):
            raise ValueError(f"finite state {state_integer} lacks a valid annotation posterior")
        return probabilities.mean(axis=0)

    maximizing_state_diagnostics = []
    for state_integer in maximizing_states:
        posterior = _state_posterior(state_integer)
        maximizing_state_diagnostics.append({
            "state": state_integer,
            "value": float(values[state_integer]),
            "mean_annotation_option_posterior": posterior.tolist(),
            "positive_annotation_lift": bool(float(values[state_integer]) > 1e-12),
            "unique_target_posterior_argmax": bool(
                posterior[0] > float(np.max(posterior[1:])) + 1e-12),
        })
    qualifying_maximizers = [
        row["state"] for row in maximizing_state_diagnostics
        if row["positive_annotation_lift"] and row["unique_target_posterior_argmax"]
    ]

    canonical_state = int(entry["fixed_teaching_target_state"])
    canonical_posterior = _state_posterior(canonical_state)
    canonical_positive_lift = bool(float(values[canonical_state]) > 1e-12)
    canonical_unique_argmax = bool(
        canonical_posterior[0] > float(np.max(canonical_posterior[1:])) + 1e-12)
    semantic_function = {
        "schema": "cr3-reconstruction-state-function-semantic-v1",
        "target_metric_key": target_metric_key,
        "option_metric_keys": [target_metric_key, *entry["distractor_metric_keys"]],
        "option_descriptions": [
            codebook_manifest["metrics"][key]["description"]
            for key in [target_metric_key, *entry["distractor_metric_keys"]]
        ],
        "reconstruction_noun": codebook_manifest["reconstruction_noun"],
        "reconstruction_max_chars": codebook_manifest["reconstruction_max_chars"],
        "fixed_teaching_indices": expected_indices,
        "fixed_teaching_item_ids": expected_item_ids,
        "choice_readout_id": str(value_payload["choice_readout_id"]),
        "reconstructor_model": str(value_payload["reconstructor_model"]),
        "reconstructor_revision": str(value_payload["reconstructor_revision"]),
        "state_rows": semantic_rows,
    }
    state_function_semantic_sha256 = _payload_sha256(semantic_function)
    core = {
        "schema": FINITE_STATE_ENVELOPE_SCHEMA,
        "target_metric_key": target_metric_key,
        "prior_panel_id": str((entry.get("prior_calibration") or {}).get("panel_id", "")),
        "distractor_metric_keys": list(entry["distractor_metric_keys"]),
        "codebook_manifest_sha256": codebook_manifest["manifest_sha256"],
        "instrument_sha256": state_table["instrument_sha256"],
        "state_scored_path": state_table["path"],
        "state_scored_sha256": state_table["sha256"],
        "state_value_path": str(value_payload["path"]),
        "state_value_sha256": str(value_payload["sha256"]),
        "state_function_semantic_sha256": state_function_semantic_sha256,
        "choice_readout_id": str(value_payload["choice_readout_id"]),
        "reconstructor_model": str(value_payload["reconstructor_model"]),
        "reconstructor_revision": str(value_payload["reconstructor_revision"]),
        "no_demonstration_target_probability": float(
            value_payload["no_demonstration_target_probability"]),
        "no_demonstration_channel_sha256": _payload_sha256({
            "canonical_choice_probabilities": np.asarray(
                value_payload["fixed_no_demo_canonical_choice_probabilities"], float).tolist(),
        }),
        "n_teaching_items": len(expected_indices),
        "n_states": state_table["n_states"],
        "state_encoding": "unsigned big-endian binary over the stored teaching order",
        "finite_state_upper_bound": upper_bound,
        "coarse_no_demo_range_cap": coarse_cap,
        "maximizing_states": maximizing_states,
        "state_envelope_capability": {
            "maximizing_state_diagnostics": maximizing_state_diagnostics,
            "qualifying_maximizing_states": qualifying_maximizers,
            "has_positive_unique_target_maximizer": bool(qualifying_maximizers),
        },
        "operational_target_diagnostic": {
            "state": canonical_state,
            "transcript_sha256": transcript_hashes[canonical_state],
            "value": float(values[canonical_state]),
            "mean_annotation_option_posterior": canonical_posterior.tolist(),
            "positive_annotation_lift": canonical_positive_lift,
            "unique_target_posterior_argmax": canonical_unique_argmax,
            "is_headline_gate": False,
            "note": (
                "replay of the frozen operational target/orbit behavior; neither ground truth, "
                "necessarily realizable by one prompt, nor an oracle-capability requirement"),
        },
        "bound_identity": (
            "for every finite executable prompt p, V(p)=v(s_T(p)) "
            "<= max_{s in {0,1}^8} v(s) <= 1-q0"),
        "premises": {
            "all_binary_transcripts_enumerated_exactly_once": True,
            "fixed_ordered_teaching_panel": True,
            "deterministic_content_cached_choice_readout": True,
            "uses_external_labels": False,
            "maximizing_state_need_not_be_prompt_reachable": True,
        },
    }
    return {**core, "summary_sha256": _payload_sha256(core)}


def validate_finite_state_envelope(
    summary: Mapping[str, object],
    *,
    codebook_manifest: Mapping[str, object],
    target_metric_key: str,
    state_scored_path: str | Path,
    value_payload: Mapping[str, object],
) -> None:
    expected = build_finite_state_envelope(
        codebook_manifest=codebook_manifest,
        target_metric_key=target_metric_key,
        state_scored_path=state_scored_path,
        value_payload=value_payload,
    )
    if dict(summary) != expected:
        raise ValueError("invalid or mutated finite-state envelope summary")


def lookup_scored_prompt_values(
    *,
    codebook_manifest: Mapping[str, object],
    target_metric_key: str,
    scored_path: str | Path,
    state_scored_path: str | Path,
    state_value_payload: Mapping[str, object],
    envelope_summary: Mapping[str, object],
) -> dict:
    """Value candidate signatures by exact lookup in the immutable 2^8 table."""
    validate_finite_state_envelope(
        envelope_summary,
        codebook_manifest=codebook_manifest,
        target_metric_key=target_metric_key,
        state_scored_path=state_scored_path,
        value_payload=state_value_payload,
    )
    target_metric_key = str(target_metric_key)
    entry = codebook_manifest["entries"][target_metric_key]
    teaching_indices = np.asarray(entry["fixed_teaching_indices"], dtype=int)
    scored = _load_scored_rows(
        scored_path, expected_probe_sha256=str(codebook_manifest["probe_sha256"]))
    if scored["signatures"].shape[1] != int(codebook_manifest["n_probes"]):
        raise ValueError("candidate signatures are not aligned to the finite-state instrument")
    bits = (scored["signatures"][:, teaching_indices] > 0.5).astype(np.uint8)
    weights = (1 << np.arange(len(teaching_indices) - 1, -1, -1)).astype(np.uint32)
    state_integers = np.asarray(bits @ weights, dtype=int)
    table_values = np.asarray(state_value_payload["values"], dtype=float)
    table_details = list(state_value_payload["details"])
    values = table_values[state_integers]
    details = []
    for text, state_integer in zip(scored["texts"], state_integers):
        detail = json.loads(json.dumps(table_details[int(state_integer)]))
        candidate_sha = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
        detail["candidate_prompt_sha256"] = candidate_sha
        detail["finite_state_lookup"] = {
            "state": int(state_integer),
            "candidate_prompt_sha256": candidate_sha,
            "target_body_not_rendered_in_mcq_query": True,
            "instrument_sha256": envelope_summary["instrument_sha256"],
            "state_value_sha256": envelope_summary["state_value_sha256"],
            "envelope_summary_sha256": envelope_summary["summary_sha256"],
        }
        details.append(detail)
    raw = np.asarray([detail["raw_target_option_probability"] for detail in details], float)
    coarse_cap = float(state_value_payload["value_cap"])
    if np.any(values > float(envelope_summary["finite_state_upper_bound"]) + 1e-12):
        raise RuntimeError("finite-state lookup produced a value above its own envelope")
    return {
        "schema": VALUE_SCHEMA,
        "target_metric_key": target_metric_key,
        "source_scored_path": scored["path"],
        "source_scored_sha256": scored["sha256"],
        "codebook_manifest_sha256": codebook_manifest["manifest_sha256"],
        "values": values,
        "raw_target_option_probability": raw,
        "details": details,
        "families": scored["families"],
        "value_name": state_value_payload["value_name"],
        "value_unit": state_value_payload["value_unit"],
        "value_cap": coarse_cap,
        "choice_readout_id": state_value_payload["choice_readout_id"],
        "no_demonstration_target_probability": state_value_payload[
            "no_demonstration_target_probability"],
        "fixed_no_demo_canonical_choice_probabilities": np.asarray(
            state_value_payload["fixed_no_demo_canonical_choice_probabilities"], float),
        "n_rows": len(values),
        "n_unique_prompt_behaviors_valued": len({
            (text, signature.tobytes())
            for text, signature in zip(scored["texts"], scored["signatures"])
        }),
        "n_unique_fixed_binary_transcripts_valued": len(set(map(int, state_integers))),
        "batched_logit_path": False,
        "finite_state_lookup": True,
        "premises": {
            "every_scored_row_valued": True,
            "codebook_frozen_before_prompt_search": True,
            "uses_external_labels": False,
            "no_demonstration_control_is_prompt_independent": True,
            "global_cap_is_one_minus_no_demonstration_probability": True,
            "value_determined_by_exact_behavior": True,
            "deterministic_choice_readout": True,
            "choice_probabilities_content_cached": True,
            "value_is_function_of_fixed_binary_teaching_transcript": True,
            "exhaustive_finite_state_lookup": True,
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


def _validated_probability_matrix(
    value: object, *, label: str, n_options: int | None = None
) -> np.ndarray:
    probabilities = np.asarray(value, dtype=float)
    if (probabilities.ndim != 2 or probabilities.shape[0] < 1
            or probabilities.shape[1] < 2
            or (n_options is not None and probabilities.shape[1] != int(n_options))
            or np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0)
            or not np.allclose(
                probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)):
        raise ValueError(f"{label} is not a finite normalized choice-probability matrix")
    return probabilities


def _validate_value_detail(
    detail: Mapping[str, object],
    *,
    expected_value: float,
    expected_raw: float,
    expected_no_demo: np.ndarray,
    expected_n_options: int,
    label: str,
) -> dict:
    """Recompute one value mark from the stored full posterior matrices."""
    identification = detail.get("identification") or {}
    conditions = identification.get("conditions") or {}
    matrices = {}
    means = {}
    for condition in ("annotations", "no_demonstrations", "shuffled_labels"):
        report = conditions.get(condition) or {}
        matrix = _validated_probability_matrix(
            report.get("canonical_choice_probabilities"),
            label=f"{label} {condition}",
            n_options=expected_n_options,
        )
        if matrix.shape != expected_no_demo.shape:
            raise ValueError(f"{label} condition blocks do not share the frozen query shape")
        mean = float(matrix[:, 0].mean())
        if not np.isclose(
                float(report.get("mean_target_probability", np.nan)), mean,
                rtol=0.0, atol=1e-12):
            raise ValueError(f"{label} has a scalar summary inconsistent with its matrix")
        matrices[condition] = matrix
        means[condition] = mean
    if not np.allclose(
            matrices["no_demonstrations"], expected_no_demo, rtol=0.0, atol=1e-12):
        raise ValueError(f"{label} changed the frozen no-demonstration matrix")

    main = means["annotations"]
    no_demo = means["no_demonstrations"]
    shuffled = means["shuffled_labels"]
    cap = 1.0 - no_demo
    lift = main - max(no_demo, shuffled)
    value = float(np.clip(lift, 0.0, cap))
    scalar_checks = {
        "identification_score": main,
        "no_demonstration_score": no_demo,
        "shuffled_label_score": shuffled,
        "annotation_lift_over_no_demonstration": main - no_demo,
        "annotation_lift_over_shuffled_labels": main - shuffled,
        "annotation_lift_over_strongest_control": lift,
    }
    for field, expected in scalar_checks.items():
        if not np.isclose(
                float(identification.get(field, np.nan)), expected,
                rtol=0.0, atol=1e-12):
            raise ValueError(f"{label} has inconsistent {field}")
    detail_checks = {
        "raw_target_option_probability": main,
        "value_cap": cap,
        "annotation_lift_unclipped": lift,
        "value_mark": value,
    }
    for field, expected in detail_checks.items():
        if not np.isclose(
                float(detail.get(field, np.nan)), expected, rtol=0.0, atol=1e-12):
            raise ValueError(f"{label} has inconsistent {field}")
    if (not np.isclose(expected_raw, main, rtol=0.0, atol=1e-12)
            or not np.isclose(expected_value, value, rtol=0.0, atol=1e-12)):
        raise ValueError(f"{label} disagrees with the persisted value arrays")
    return {"matrices": matrices, "means": means, "cap": cap, "lift": lift, "value": value}


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
            or np.any(~np.isfinite(raw)) or np.any(raw < -1e-12) or np.any(raw > 1.0 + 1e-12)
            or not np.isfinite(cap) or not 0.0 <= cap <= 1.0
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
    details = json.loads(str(z["details_json"]))
    families = [str(value) for value in z["families"]]
    if (not isinstance(details, list) or len(details) != len(values)
            or (families and len(families) != len(values))):
        raise ValueError(f"value artifact row metadata is incomplete in {source}")
    for index, (value, raw_value, detail) in enumerate(zip(values, raw, details)):
        if not isinstance(detail, Mapping):
            raise ValueError(f"value artifact detail {index} is not a mapping")
        _validate_value_detail(
            detail,
            expected_value=float(value),
            expected_raw=float(raw_value),
            expected_no_demo=fixed_no_demo,
            expected_n_options=fixed_no_demo.shape[1],
            label=f"value artifact row {index}",
        )
    return {
        "path": str(source),
        "sha256": file_sha256(source),
        "target_metric_key": str(z["target_metric_key"]),
        "source_scored_sha256": source_sha,
        "codebook_manifest_sha256": codebook_sha,
        "values": values,
        "raw_target_option_probability": raw,
        "details": details,
        "families": families,
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
