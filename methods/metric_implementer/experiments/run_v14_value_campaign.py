"""Resumable v14 decoder-instrument campaign launcher.

GPU stages are deliberately split by resident model: ``constructor`` fills MCQ cells
and behavioral inductions for one decoder family, ``executor`` fills rule/probe cells
with the frozen 8B executor, and ``aggregate`` is CPU-only.  The split makes physical
GPU authorization explicit and preserves completed cells across process restarts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..config import ImplementerConfig
from ..vllm_backend import (
    CR3_BINARY_READOUT_ID,
    make_judge_backend,
    model_revision_id,
    release_resident_engines,
)
from .cr3_evidence_store import EvidenceCellStore, file_sha256
from .cr3_reconstruction_values import _bootstrap
from .cr3_sampled_value_certify import _codebook_menu
from .run_v13_value_campaign import (
    FIXED_EXECUTOR,
    _load_codebook_for_entry,
    _resolve_path,
    load_candidate_population,
    load_metrics_manifest,
    select_metric_entries,
    _target_entropy_for_entry,
)
from .v14_behavioral_channel import (
    BEHAVIORAL_ARMS,
    DEFAULT_TEMPLATE as DEFAULT_BEHAVIORAL_TEMPLATE,
    evaluate_behavioral_state_tables_v14,
)
from .v14_audit import (
    AUDIT_FAMILIES,
    propose_family_audit,
    score_audit_ledger,
)
from .v14_mcq_channel import DEFAULT_MCQ_TEMPLATE, evaluate_mcq_state_tables_v14
from .v14_decoder_tuning import (
    propose_mutations,
    select_dev_metrics,
    tune_shared_template_batched,
)
from .v14_panel_design import (
    build_panel_design,
    canonical_sha256,
    freeze_probe_split,
    validate_panel_design,
    validate_probe_split,
)
from .v14_value_bound import (
    aggregate_state_tables,
    classify_status,
    dkw_expected_best_gain_bound,
    fidelity_legibility_diagnostic,
    novelty_collapse_curves,
    record_rank_gain_bound,
    signatures_to_states,
    split_sample_cp_gain_bound,
    validate_state_tables,
    plugin_binary_mutual_information,
)
from .v14_tuning_evaluator import (
    aggregate_template_fitness,
    cached_probe_embeddings,
    freeze_reference_set,
    induce_behavioral_reference_templates,
    score_behavioral_reference_templates,
    score_mcq_reference_templates,
)
from .v14_preregistration import (
    build_preregistration,
    build_production_freeze,
    choose_qualified_decoder,
    evaluate_decoder_qualification,
    evaluate_sentinel_liveness,
)
from .v14_release_report import audit_release, write_release_outputs
from .v14_liveness_controls import (
    finish_liveness_executor_controls,
    run_liveness_constructor_controls,
)


CAMPAIGN_SCHEMA = "cr3-v14-campaign-v1"
DESIGN_SCHEMA = "cr3-v14-metric-design-v1"
TEMPLATE_FREEZE_SCHEMA = "cr3-v14-template-freeze-v1"
FORBIDDEN_SK3_GPUS = {1, 2, 3, 4}
DEFAULT_DECODER_MODELS = {
    "qwen": "Qwen/Qwen2.5-14B-Instruct",
    "llama": "meta-llama/Llama-3.3-70B-Instruct",
    "mistral": "mistralai/Mistral-Small-24B-Instruct-2501",
}


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", str(value)).strip("_")


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def parse_physical_gpu_ids(values: Sequence[str] | str | None) -> tuple[int, ...]:
    if values is None:
        raw = os.environ.get("V14_PHYSICAL_GPUS", "")
        parts = raw.split(",") if raw else []
    elif isinstance(values, str):
        parts = values.split(",")
    else:
        parts = [part for value in values for part in str(value).split(",")]
    try:
        ids = tuple(int(part.strip()) for part in parts if part.strip())
    except ValueError as exc:
        raise ValueError("physical GPU IDs must be comma-separated integers") from exc
    if len(ids) != len(set(ids)) or any(value < 0 for value in ids):
        raise ValueError("physical GPU IDs must be unique nonnegative integers")
    return ids


def assert_gpu_authorized(
    physical_gpu_ids: Sequence[int], *, hostname: str | None = None, fake_backends: bool = False,
) -> None:
    if fake_backends:
        return
    host = str(hostname or socket.gethostname()).lower()
    ids = set(map(int, physical_gpu_ids))
    if not ids:
        raise RuntimeError("GPU phases require explicit physical IDs via --physical-gpus")
    if host.startswith("sk3") and ids.intersection(FORBIDDEN_SK3_GPUS):
        raise RuntimeError(
            "hard safety stop: sk3 physical GPUs 1,2,3,4 are permanently forbidden for v14"
        )
    visible = tuple(
        int(value) for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip().isdigit()
    )
    if visible and tuple(map(int, physical_gpu_ids)) != visible:
        raise RuntimeError(
            f"declared physical GPUs {tuple(physical_gpu_ids)} do not match "
            f"CUDA_VISIBLE_DEVICES={visible}"
        )


def _backend(model: str, *, fake: bool):
    cfg = ImplementerConfig()
    cfg.vllm_fake = bool(fake)
    cfg.vllm_gpu_mem_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.90"))
    cfg.vllm_tp_size = int(os.environ.get("VLLM_TP_SIZE", "1"))
    cfg.vllm_max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192"))
    if os.environ.get("METRIC_IMPLEMENTER_LFS_HOME"):
        cfg.vllm_lfs_home = os.environ["METRIC_IMPLEMENTER_LFS_HOME"]
    overrides = json.loads(os.environ.get("V14_MODEL_PATH_OVERRIDES_JSON", "{}"))
    runtime_model = str(overrides.get(str(model), str(model)))
    result = make_judge_backend(runtime_model, cfg, 0.0)
    revision = str(model) if fake else model_revision_id(runtime_model)
    return result, revision


def _probe_ids(probe_texts: Sequence[str]) -> list[str]:
    return [
        hashlib.sha256(f"{index}\x1f{text}".encode("utf-8")).hexdigest()
        for index, text in enumerate(probe_texts)
    ]


def _certification_probe_embeddings(
    probe_texts: Sequence[str], *, cache_path: str | Path, fake_backends: bool,
) -> np.ndarray:
    if not fake_backends:
        return cached_probe_embeddings(probe_texts, cache_path=cache_path)
    rows = []
    for text in map(str, probe_texts):
        digest = hashlib.sha256(("fake-bge\x1f" + text).encode("utf-8")).digest()
        vector = np.frombuffer(digest, dtype=np.uint8).astype(float) - 127.5
        rows.append(vector / np.linalg.norm(vector))
    return np.vstack(rows)


def _codebook_signatures(codebook: Mapping[str, object]) -> tuple[list[str], np.ndarray]:
    keys = sorted(map(str, codebook["metrics"]))
    rows = []
    for key in keys:
        bootstrap = _bootstrap(codebook["metrics"][key]["bootstrap_path"])
        rows.append(np.asarray(bootstrap["target"], dtype=float))
    matrix = np.vstack(rows)
    if matrix.ndim != 2 or np.any(~np.isfinite(matrix)):
        raise RuntimeError("codebook bootstrap targets are invalid")
    return keys, matrix


def _design_path(out_root: Path, metric_key: str) -> Path:
    return out_root / "designs" / _safe(metric_key) / "design_manifest.json"


def build_designs(
    *, metrics_manifest_path: str | Path, out_root: str | Path, run_sha: str,
    metric_keys: Sequence[str] | None = None,
) -> list[dict]:
    """CPU-only v14 split/panel freeze over the existing 300-probe assets."""
    manifest, base = load_metrics_manifest(metrics_manifest_path)
    entries = select_metric_entries(manifest, base)
    requested = None if metric_keys is None else set(map(str, metric_keys))
    if requested is not None:
        entries = [entry for entry in entries if str(entry["metric_key"]) in requested]
        if {str(entry["metric_key"]) for entry in entries} != requested:
            raise ValueError("requested metric key is absent from the selected manifest")
    out = Path(out_root).resolve()
    results = []
    for entry in entries:
        metric_key = str(entry["metric_key"])
        destination = _design_path(out, metric_key)
        if destination.is_file():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            validate_metric_design(existing)
            results.append(existing)
            continue
        codebook = _load_codebook_for_entry(entry, base)
        target_bootstrap = _bootstrap(codebook["metrics"][metric_key]["bootstrap_path"])
        probe_texts = list(map(str, target_bootstrap["probe_texts"]))
        if len(probe_texts) != 300:
            raise ValueError(f"v14 requires the 300-probe bank for {metric_key}")
        ids = _probe_ids(probe_texts)
        split = freeze_probe_split(ids, run_sha=run_sha, metric_key=metric_key)
        codebook_keys, signatures = _codebook_signatures(codebook)
        target_index = codebook_keys.index(metric_key)
        panel = build_panel_design(
            signatures, target_index=target_index,
            teaching_indices=split["teaching"]["indices"], run_sha=run_sha,
            metric_key=metric_key, probe_ids=ids,
            decoder_families=("qwen", "llama", "mistral"),
        )
        materialized_entry = dict(entry)
        for field in ("codebook_path", "assets_root", "candidate_bank_path"):
            if materialized_entry.get(field):
                materialized_entry[field] = str(_resolve_path(materialized_entry[field], base))
        if materialized_entry.get("candidate_sources"):
            materialized_entry["candidate_sources"] = [
                {
                    **dict(source),
                    "path": str(_resolve_path(source["path"], base)),
                }
                for source in materialized_entry["candidate_sources"]
            ]
        materialized_entry.pop("candidate_source_set", None)
        core = {
            "schema": DESIGN_SCHEMA,
            "run_sha": str(run_sha),
            "metric_key": metric_key,
            "task": str(entry["task"]),
            "level": str(entry["level"]),
            "metric": str(entry["metric"]),
            "source_metrics_manifest": str(Path(metrics_manifest_path).resolve()),
            "source_metrics_manifest_sha256": file_sha256(metrics_manifest_path),
            "entry": materialized_entry,
            "probe_split": split,
            "panel_design": panel,
            "codebook_metric_keys": codebook_keys,
            "codebook_signatures_sha256": hashlib.sha256(
                np.ascontiguousarray((signatures > 0.5).astype(np.uint8)).tobytes()
            ).hexdigest(),
            "target_entropy_on_h_bits": _binary_entropy(
                np.asarray(target_bootstrap["target"])[split["heldout"]["indices"]] > 0.5
            ),
            "executor": {
                "model": str(target_bootstrap.get("executor_model", FIXED_EXECUTOR)),
                "revision": str(target_bootstrap.get("executor_model_revision", "unknown")),
                "readout_id": str(target_bootstrap.get("readout_id", CR3_BINARY_READOUT_ID)),
            },
        }
        core["design_manifest_sha256"] = canonical_sha256(core)
        _atomic_json(destination, core)
        results.append(core)
    index = {
        "schema": "cr3-v14-design-index-v1",
        "run_sha": str(run_sha),
        "metrics": [{
            "metric_key": row["metric_key"],
            "path": str(_design_path(out, row["metric_key"])),
            "sha256": row["design_manifest_sha256"],
        } for row in results],
    }
    index["index_sha256"] = canonical_sha256(index)
    _atomic_json(out / "design_index.json", index)
    return results


def _binary_entropy(bits: Sequence[int]) -> float:
    vector = np.asarray(bits, dtype=np.uint8)
    probability = float(np.mean(vector))
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return float(-probability * np.log2(probability) - (1.0 - probability) * np.log2(1.0 - probability))


def validate_metric_design(payload: Mapping[str, object]) -> None:
    if payload.get("schema") != DESIGN_SCHEMA:
        raise ValueError("unsupported v14 metric design")
    validate_probe_split(payload["probe_split"])
    validate_panel_design(payload["panel_design"])
    core = dict(payload)
    observed = str(core.pop("design_manifest_sha256", ""))
    if observed != canonical_sha256(core):
        raise ValueError("v14 metric design checksum mismatch")
    teaching = set(payload["probe_split"]["teaching"]["indices"])
    if teaching != set(payload["panel_design"]["teaching_indices"]):
        raise ValueError("panel design does not use the frozen teaching split")


def load_designs(out_root: str | Path, metric_keys: Sequence[str] | None = None) -> list[dict]:
    root = Path(out_root).resolve()
    index = json.loads((root / "design_index.json").read_text(encoding="utf-8"))
    requested = None if metric_keys is None else set(map(str, metric_keys))
    rows = []
    for entry in index["metrics"]:
        if requested is not None and str(entry["metric_key"]) not in requested:
            continue
        payload = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
        validate_metric_design(payload)
        rows.append(payload)
    if requested is not None and {row["metric_key"] for row in rows} != requested:
        raise ValueError("requested design is missing")
    return rows


def prepare_development_population(
    *, certified_out_root: str | Path, dev_metrics_manifest_path: str | Path,
    run_sha: str,
) -> dict:
    """Freeze eight metric-held-out development metrics and sparse GEPA references."""
    certified = load_designs(certified_out_root)
    certified_keys = [str(row["metric_key"]) for row in certified]
    manifest, base = load_metrics_manifest(dev_metrics_manifest_path)
    candidates = [dict(row) for row in manifest["metrics"]]
    by_task = {}
    for row in candidates:
        row["target_entropy_bits"] = _target_entropy_for_entry(row, base)
        by_task.setdefault(str(row["task"]), []).append(row)
    for task_rows in by_task.values():
        ranked = sorted(task_rows, key=lambda row: (
            float(row["target_entropy_bits"]), str(row["metric_key"]),
        ))
        denominator = max(1, len(ranked) - 1)
        for position, row in enumerate(ranked):
            row["target_entropy_quintile"] = min(4, int(5 * position / (denominator + 1)))
    selected = select_dev_metrics(
        candidates, certified_metric_keys=certified_keys, run_sha=run_sha, n_dev=8,
    )
    for row in selected:
        for field in ("codebook_path", "assets_root", "candidate_bank_path"):
            if row.get(field):
                row[field] = str(_resolve_path(row[field], base))
        if row.get("candidate_sources"):
            for source in row["candidate_sources"]:
                source["path"] = str(_resolve_path(source["path"], base))
    dev_root = Path(certified_out_root).resolve() / "development"
    selected_manifest = {
        "schema": "cr3-value-bound-metrics-v13.1",
        "release": "v14-development-only-no-claims",
        "selection_provenance": "seven tasks plus one additional metric; disjoint from certified keys",
        "metrics": selected,
    }
    selected_path = dev_root / "dev_metrics.json"
    _atomic_json(selected_path, selected_manifest)
    designs = build_designs(
        metrics_manifest_path=selected_path, out_root=dev_root, run_sha=f"{run_sha}:dev",
    )
    reference_rows = []
    for design in designs:
        context = _metric_context(design)
        ddec = list(design["probe_split"]["decoder_development"]["indices"])
        target = (np.asarray(context["target"]["target"], dtype=float)[ddec] > 0.5).astype(int)
        signatures = np.asarray(context["population"]["signatures"], dtype=float)
        reference_values = np.asarray([
            plugin_binary_mutual_information(target, row[ddec] > 0.5)
            for row in signatures
        ], dtype=float)
        # Near/far transfer needs only teaching examples and D_dec.  Do not even
        # embed certification H during adaptive development: H is first touched
        # after the template freeze.
        teaching = list(map(int, design["probe_split"]["teaching"]["indices"]))
        embedding_indices = sorted(set(teaching).union(ddec))
        embedding_texts = [context["probe_texts"][index] for index in embedding_indices]
        embedding_path = dev_root / "embeddings" / f"{_safe(design['metric_key'])}.npz"
        subset_embeddings = cached_probe_embeddings(
            embedding_texts, cache_path=embedding_path,
        )
        embeddings = np.zeros(
            (len(context["probe_texts"]), subset_embeddings.shape[1]), dtype=float,
        )
        embeddings[np.asarray(embedding_indices, dtype=int)] = subset_embeddings
        reference = freeze_reference_set(
            metric_key=str(design["metric_key"]),
            panel_design=design["panel_design"],
            candidate_signatures=signatures,
            target_signature=context["target"]["target"],
            decoder_development_indices=ddec,
            candidate_reference_values=reference_values,
            probe_embeddings=embeddings,
        )
        path = dev_root / "references" / f"{_safe(design['metric_key'])}.json"
        _atomic_json(path, reference)
        reference_rows.append({
            "metric_key": str(design["metric_key"]), "path": str(path),
            "sha256": canonical_sha256(reference),
        })
    index = {
        "schema": "cr3-v14-development-reference-index-v1",
        "certified_metric_keys": certified_keys,
        "development_metric_keys": [str(row["metric_key"]) for row in designs],
        "metric_level_disjoint": not bool(
            set(certified_keys).intersection(row["metric_key"] for row in designs)
        ),
        "selected_manifest": str(selected_path),
        "selected_manifest_sha256": file_sha256(selected_path),
        "references": reference_rows,
    }
    if not index["metric_level_disjoint"]:
        raise RuntimeError("development and certified metric populations overlap")
    index["index_sha256"] = canonical_sha256(index)
    _atomic_json(dev_root / "reference_index.json", index)
    return index


def _development_contexts(out_root: str | Path) -> list[dict]:
    root = Path(out_root).resolve() / "development"
    designs = load_designs(root)
    contexts = []
    for design in designs:
        context = _metric_context(design)
        reference = json.loads(
            (root / "references" / f"{_safe(design['metric_key'])}.json").read_text()
        )
        menu = context["menu"]
        contexts.append({
            "metric_key": str(design["metric_key"]),
            "noun": str(context["codebook"]["reconstruction_noun"]),
            "probe_texts": context["probe_texts"],
            "target_signature": np.asarray(context["target"]["target"], dtype=float),
            "target_description": str(menu["entry"]["target_description"]),
            "distractors": menu["distractors"],
            "reference_set": reference,
            "executor_revision": str(design["executor"]["revision"]),
            "readout_id": str(design["executor"]["readout_id"]),
        })
    return contexts


def run_decoder_tuning(
    *, out_root: str | Path, channel: str, arm: str,
    decoder_models: Mapping[str, str], proposer_model: str,
    fake_backends: bool, physical_gpu_ids: Sequence[int], query_batch_size: int = 2048,
) -> dict:
    """Run one finite shared-template GEPA search with model-resident round batches."""
    assert_gpu_authorized(physical_gpu_ids, fake_backends=fake_backends)
    if channel not in {"mcq", "behavioral"}:
        raise ValueError("tuning channel must be mcq or behavioral")
    if channel == "behavioral" and arm not in BEHAVIORAL_ARMS:
        raise ValueError("behavioral tuning requires one declared arm")
    contexts = _development_contexts(out_root)
    if len(contexts) != 8:
        raise RuntimeError("v14 tuning requires exactly eight development metrics")
    store_path = Path(out_root) / "development" / "tuning_cells.sqlite"
    seed_template = DEFAULT_MCQ_TEMPLATE if channel == "mcq" else DEFAULT_BEHAVIORAL_TEMPLATE
    required_fields = (
        ("noun", "examples", "choices", "labels") if channel == "mcq"
        else ("noun", "feature_table", "examples", "arm_instruction")
    )
    forbidden = []
    for context in contexts:
        forbidden.extend([context["metric_key"], context["target_description"]])
        forbidden.extend(
            str(item.get("description", "") if isinstance(item, Mapping) else item.description)
            for item in context["distractors"]
        )

    def propose_fn(incumbent, feedback, round_index, count):
        proposer, _revision = _backend(proposer_model, fake=fake_backends)
        try:
            return propose_mutations(
                proposer, channel=channel, arm=arm, incumbent=incumbent,
                feedback=feedback, round_index=round_index, count=count,
            )
        finally:
            release_resident_engines()

    def evaluate_batch(templates):
        rows = []
        behavioral_inductions = {}
        with EvidenceCellStore(store_path) as store:
            for family, model in decoder_models.items():
                decoder, decoder_revision = _backend(model, fake=fake_backends)
                try:
                    if channel == "mcq":
                        rows.extend(score_mcq_reference_templates(
                            decoder, templates=templates, contexts=contexts,
                            decoder_family=family, constructor_revision=decoder_revision,
                            store=store, query_batch_size=query_batch_size,
                        ))
                    else:
                        behavioral_inductions[family] = induce_behavioral_reference_templates(
                            decoder, templates=templates, arm=arm, contexts=contexts,
                            decoder_family=family, decoder_revision=decoder_revision,
                            store=store,
                        )
                finally:
                    release_resident_engines()
            if channel == "behavioral":
                executor, executor_revision = _backend(FIXED_EXECUTOR, fake=fake_backends)
                try:
                    for family in decoder_models:
                        rows.extend(score_behavioral_reference_templates(
                            executor, templates=templates, arm=arm, contexts=contexts,
                            induction_rows=behavioral_inductions[family],
                            executor_revision=executor_revision,
                            readout_id=CR3_BINARY_READOUT_ID, store=store,
                            query_batch_size=query_batch_size, decoder_family=family,
                        ))
                finally:
                    release_resident_engines()
        reports = aggregate_template_fitness(rows)
        return reports

    result = tune_shared_template_batched(
        propose_fn, evaluate_batch, seed_template=seed_template,
        channel=channel, arm=arm, forbidden_strings=forbidden,
        required_fields=required_fields,
    )
    name = "mcq" if channel == "mcq" else f"behavioral__{arm}"
    _atomic_json(Path(out_root) / "development" / "tuning" / f"{name}.json", result)
    return result


def _qualification_panel(design: Mapping[str, object], family: str) -> dict:
    source = min(
        design["panel_design"]["panels"], key=lambda row: str(row["panel_sha256"])
    )
    row = dict(source)
    row["original_trial"] = int(source["trial"])
    row["decoder_family"] = str(family)
    return {
        **dict(design["panel_design"]),
        "panels": [row], "n_panels": 1,
        "design_sha256": canonical_sha256({
            "purpose": "decoder-mini-qualification", "family": family,
            "parent": design["panel_design"]["design_sha256"],
            "panel": source["panel_sha256"],
        }),
    }


def prepare_sentinel_population(
    *, out_root: str | Path, sentinel_metrics_manifest_path: str | Path,
    run_sha: str, sentinel_metric_keys: Sequence[str],
) -> list[dict]:
    if len(sentinel_metric_keys) != 6:
        raise ValueError("v14 sentinel population must contain exactly six metrics")
    return build_designs(
        metrics_manifest_path=sentinel_metrics_manifest_path,
        out_root=Path(out_root) / "sentinel", run_sha=f"{run_sha}:sentinel",
        metric_keys=sentinel_metric_keys,
    )


def run_qualification_constructor(
    *, out_root: str | Path, decoder_family: str, decoder_model: str,
    fake_backends: bool, physical_gpu_ids: Sequence[int], query_batch_size: int = 2048,
) -> dict:
    assert_gpu_authorized(physical_gpu_ids, fake_backends=fake_backends)
    designs = load_designs(Path(out_root) / "sentinel")
    if len(designs) != 6:
        raise RuntimeError("decoder mini-qualification requires six sentinel designs")
    backend, revision = _backend(decoder_model, fake=fake_backends)
    model_key = _safe(decoder_model)
    root = Path(out_root) / "sentinel" / "qualification" / decoder_family / model_key
    rows = []
    try:
        with EvidenceCellStore(Path(out_root) / "sentinel" / "qualification_cells.sqlite") as store:
            for design in designs:
                context = _metric_context(design)
                subset = _qualification_panel(design, decoder_family)
                marker = root / _safe(design["metric_key"]) / "induction.json"
                if marker.exists():
                    row = json.loads(marker.read_text())
                else:
                    row = evaluate_behavioral_state_tables_v14(
                        backend, None, design_manifest=subset,
                        probe_texts=context["probe_texts"],
                        heldout_indices=design["probe_split"]["heldout"]["indices"],
                        heldout_target=(np.asarray(context["target"]["target"])[
                            design["probe_split"]["heldout"]["indices"]
                        ] > 0.5).astype(int),
                        noun=str(context["codebook"]["reconstruction_noun"]),
                        decoder_revision=revision,
                        executor_revision=str(design["executor"]["revision"]),
                        readout_id=str(design["executor"]["readout_id"]), store=store,
                        templates={arm: DEFAULT_BEHAVIORAL_TEMPLATE for arm in BEHAVIORAL_ARMS},
                        query_batch_size=query_batch_size, induction_only=True,
                    )
                    _atomic_json(marker, row)
                rows.append({"metric_key": design["metric_key"], **row})
    finally:
        release_resident_engines()
    payload = {
        "schema": "cr3-v14-qualification-constructor-v1", "family": decoder_family,
        "model": decoder_model, "revision": revision, "rows": rows,
    }
    _atomic_json(root / "constructor.json", payload)
    return payload


def run_qualification_executor(
    *, out_root: str | Path, decoder_family: str, decoder_model: str,
    fake_backends: bool, physical_gpu_ids: Sequence[int], query_batch_size: int = 4096,
) -> dict:
    assert_gpu_authorized(physical_gpu_ids, fake_backends=fake_backends)
    designs = load_designs(Path(out_root) / "sentinel")
    model_key = _safe(decoder_model)
    root = Path(out_root) / "sentinel" / "qualification" / decoder_family / model_key
    constructor = json.loads((root / "constructor.json").read_text())
    executor, revision = _backend(FIXED_EXECUTOR, fake=fake_backends)
    rows = []
    try:
        with EvidenceCellStore(Path(out_root) / "sentinel" / "qualification_cells.sqlite") as store:
            for design in designs:
                if not fake_backends and revision != str(design["executor"]["revision"]):
                    raise RuntimeError(
                        "qualification executor revision differs from the frozen design"
                    )
                context = _metric_context(design)
                subset = _qualification_panel(design, decoder_family)
                result = evaluate_behavioral_state_tables_v14(
                    _CacheOnlyConstructor(), executor, design_manifest=subset,
                    probe_texts=context["probe_texts"],
                    heldout_indices=design["probe_split"]["heldout"]["indices"],
                    heldout_target=(np.asarray(context["target"]["target"])[
                        design["probe_split"]["heldout"]["indices"]
                    ] > 0.5).astype(int),
                    noun=str(context["codebook"]["reconstruction_noun"]),
                    decoder_revision=str(constructor["revision"]),
                    executor_revision=str(design["executor"]["revision"]),
                    readout_id=str(design["executor"]["readout_id"]), store=store,
                    templates={arm: DEFAULT_BEHAVIORAL_TEMPLATE for arm in BEHAVIORAL_ARMS},
                    query_batch_size=query_batch_size,
                )
                panel = subset["panels"][0]
                state = int("".join(map(str, panel["target_state_bits"])), 2)
                arm_row = result["arms"]["unconstrained"]
                shuffled_lift = max(
                    0.0, float(arm_row["shuffled_mi"][0, state]) - float(arm_row["blind_mi"])
                )
                rows.append({
                    "metric_key": str(design["metric_key"]),
                    "canonical_lift_bits": float(arm_row["clipped_value"][0, state]),
                    "canonical_raw_lift_bits": float(arm_row["raw_lift"][0, state]),
                    "shuffled_lift_bits": shuffled_lift,
                })
    finally:
        release_resident_engines()
    qualification = evaluate_decoder_qualification(rows)
    qualification.update({
        "family": decoder_family, "model": decoder_model,
        "revision": str(constructor["revision"]), "executor_revision": revision,
    })
    _atomic_json(root / "qualification.json", qualification)
    return qualification


def freeze_production_instrument(
    *, out_root: str | Path, qualification_specs: Sequence[str], release_commit: str,
) -> dict:
    root = Path(out_root).resolve()
    selections = []
    for spec in qualification_specs:
        # family=primary_path[,fallback_path]
        if "=" not in spec:
            raise ValueError("qualification spec must be family=primary[,fallback]")
        family, paths = spec.split("=", 1)
        candidates = paths.split(",")
        primary = json.loads(Path(candidates[0]).read_text())
        fallback = json.loads(Path(candidates[1]).read_text()) if len(candidates) > 1 else None
        selections.append(choose_qualified_decoder(
            family=family, primary=primary, fallback=fallback,
        ))
    tuning_root = root / "development" / "tuning"
    traces = {
        "mcq": json.loads((tuning_root / "mcq.json").read_text()),
        "behavioral_unconstrained": json.loads(
            (tuning_root / "behavioral__unconstrained.json").read_text()
        ),
        "behavioral_no_verbatim": json.loads(
            (tuning_root / "behavioral__no_verbatim_examples.json").read_text()
        ),
    }
    design_index = json.loads((root / "design_index.json").read_text())
    forbidden = []
    for design in [*load_designs(root), *load_designs(root / "development")]:
        context = _metric_context(design)
        menu = context["menu"]
        forbidden.extend([
            str(design["metric_key"]), str(menu["entry"]["target_description"]),
        ])
        forbidden.extend(
            str(item.get("description", "") if isinstance(item, Mapping) else item.description)
            for item in menu["distractors"]
        )
    freeze = build_production_freeze(
        design_index=design_index, decoder_selections=selections,
        mcq_trace=traces["mcq"],
        behavioral_unconstrained_trace=traces["behavioral_unconstrained"],
        behavioral_no_verbatim_trace=traces["behavioral_no_verbatim"],
        forbidden_strings=forbidden, release_commit=release_commit,
        out_path=root / "template_freeze.json",
    )
    first_design = load_designs(root)[0]
    prereg = build_preregistration(
        template_freeze=freeze, design_index=design_index,
        metrics_manifest_sha256=str(first_design["source_metrics_manifest_sha256"]),
        out_path=root / "preregistration.json",
    )
    return {"template_freeze": freeze, "preregistration": prereg}


def run_liveness_constructor_stage(
    *, out_root: str | Path, decoder_family: str, fake_backends: bool,
    physical_gpu_ids: Sequence[int], query_batch_size: int = 1024,
) -> dict:
    assert_gpu_authorized(physical_gpu_ids, fake_backends=fake_backends)
    root = Path(out_root).resolve()
    freeze = load_template_freeze(root / "template_freeze.json")
    selection = next(
        row for row in freeze["decoder_panel"] if str(row["family"]) == decoder_family
    )
    backend, revision = _backend(str(selection["model"]), fake=fake_backends)
    if not fake_backends and revision != str(selection["revision"]):
        raise RuntimeError("liveness decoder revision differs from the production freeze")
    rows = {}
    try:
        with EvidenceCellStore(root / "sentinel" / "liveness_cells.sqlite") as store:
            for instrument, templates in freeze["instruments"].items():
                rows[instrument] = run_liveness_constructor_controls(
                    backend, decoder_family=decoder_family, decoder_revision=revision,
                    templates=templates, store=store, query_batch_size=query_batch_size,
                )
    finally:
        release_resident_engines()
    payload = {
        "schema": "cr3-v14-liveness-constructor-stage-v1",
        "decoder_family": decoder_family, "decoder_model": selection["model"],
        "decoder_revision": revision, "instruments": rows,
    }
    _atomic_json(root / "sentinel" / f"liveness_constructor_{decoder_family}.json", payload)
    return payload


def run_liveness_executor_stage(
    *, out_root: str | Path, fake_backends: bool,
    physical_gpu_ids: Sequence[int], query_batch_size: int = 2048,
) -> dict:
    assert_gpu_authorized(physical_gpu_ids, fake_backends=fake_backends)
    root = Path(out_root).resolve()
    sentinel_results = root / "sentinel" / "results.parquet"
    if not sentinel_results.is_file():
        raise RuntimeError("six-metric frozen-instrument sentinel evaluation is incomplete")
    sentinel_frame = pd.read_parquet(sentinel_results)
    if len(sentinel_frame) != 6 * 2 * 3 or sentinel_frame["metric_key"].nunique() != 6:
        raise RuntimeError("six-metric sentinel result artifact is structurally incomplete")
    executor, revision = _backend(FIXED_EXECUTOR, fake=fake_backends)
    rows = []
    try:
        with EvidenceCellStore(root / "sentinel" / "liveness_cells.sqlite") as store:
            for family in DEFAULT_DECODER_MODELS:
                constructor = json.loads(
                    (root / "sentinel" / f"liveness_constructor_{family}.json").read_text()
                )
                for instrument, result in constructor["instruments"].items():
                    observed = finish_liveness_executor_controls(
                        executor, constructor_result=result, executor_revision=revision,
                        readout_id=CR3_BINARY_READOUT_ID, store=store,
                        query_batch_size=query_batch_size,
                    )
                    rows.extend({**row, "instrument": instrument} for row in observed)
    finally:
        release_resident_engines()
    payload = {
        "schema": "cr3-v14-liveness-control-rows-v1",
        "executor_revision": revision, "rows": rows,
    }
    control_path = root / "sentinel" / "liveness_control_rows.json"
    _atomic_json(control_path, payload)
    gate = apply_sentinel_gate(control_rows_path=control_path, out_root=root)
    return {"controls": payload, "gate": gate}


def load_template_freeze(path: str | Path) -> dict:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != TEMPLATE_FREEZE_SCHEMA:
        raise ValueError("unsupported v14 template freeze")
    core = dict(payload)
    observed = str(core.pop("freeze_sha256", ""))
    if observed != canonical_sha256(core):
        raise ValueError("template freeze checksum mismatch")
    return payload


def _require_live_sentinel(path: str | Path | None, *, fake_backends: bool) -> None:
    if fake_backends:
        return
    if path is None:
        raise RuntimeError("production fan-out requires --sentinel-report")
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != "cr3-v14-sentinel-liveness-v1" or not bool(
        payload.get("passed", False)
    ):
        raise RuntimeError("sentinel liveness did not pass; production fan-out is blocked")


def apply_sentinel_gate(*, control_rows_path: str | Path, out_root: str | Path) -> dict:
    payload = json.loads(Path(control_rows_path).read_text())
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("sentinel control artifact must contain a rows list")
    result = evaluate_sentinel_liveness(rows)
    _atomic_json(Path(out_root) / "sentinel_report.json", result)
    if not result["passed"]:
        raise RuntimeError("sentinel detected structural failure or control-defined instrument death")
    return result


def write_seed_template_freeze(path: str | Path) -> dict:
    """Write the untuned instrument only; tuned entries must later come from GEPA."""
    core = {
        "schema": TEMPLATE_FREEZE_SCHEMA,
        "instruments": {
            "untuned": {
                "mcq": DEFAULT_MCQ_TEMPLATE,
                "behavioral": {arm: DEFAULT_BEHAVIORAL_TEMPLATE for arm in BEHAVIORAL_ARMS},
            },
        },
        "shared_across_metrics_and_decoder_families": True,
        "searched_per_family_variation": False,
        "study_alpha": 0.05,
        "process_bound_cell_alpha": 0.05,
        "alpha_scope": "seed-only smoke freeze; production preregistration must replace this",
    }
    core["freeze_sha256"] = canonical_sha256(core)
    _atomic_json(Path(path), core)
    return core


def _subset_panel_design(design: Mapping[str, object], family: str) -> dict:
    panel = dict(design["panel_design"])
    selected = [
        dict(row) for row in panel["panels"] if str(row["decoder_family"]) == str(family)
    ]
    if not selected:
        raise ValueError(f"design has no panels assigned to decoder family {family}")
    for row in selected:
        row["original_trial"] = int(row["trial"])
    panel["panels"] = selected
    panel["n_panels"] = len(selected)
    panel["design_sha256"] = canonical_sha256({
        "parent": design["panel_design"]["design_sha256"],
        "family": family,
        "trials": [row["original_trial"] for row in selected],
    })
    return panel


class _CacheOnlyConstructor:
    def generate_batch(self, *_args, **_kwargs):
        raise RuntimeError("behavioral induction cache is incomplete")


def _metric_context(design: Mapping[str, object]) -> dict:
    entry = design["entry"]
    codebook = _load_codebook_for_entry(entry, Path("/"))
    metric_key = str(design["metric_key"])
    target = _bootstrap(codebook["metrics"][metric_key]["bootstrap_path"])
    probe_texts = list(map(str, target["probe_texts"]))
    population = load_candidate_population(
        entry, Path("/"), n_probes=len(probe_texts), probe_sha256=str(target["probe_sha256"]),
    )
    return {
        "codebook": codebook, "target": target, "probe_texts": probe_texts,
        "population": population, "menu": _codebook_menu(codebook, metric_key),
    }


def run_constructor_stage(
    *, out_root: str | Path, template_freeze_path: str | Path,
    decoder_family: str, decoder_model: str, metric_keys: Sequence[str] | None,
    fake_backends: bool, physical_gpu_ids: Sequence[int], channels: Sequence[str],
    query_batch_size: int, sentinel_report_path: str | Path | None = None,
    require_sentinel: bool = True,
) -> list[dict]:
    if require_sentinel:
        _require_live_sentinel(sentinel_report_path, fake_backends=fake_backends)
    assert_gpu_authorized(physical_gpu_ids, fake_backends=fake_backends)
    designs = load_designs(out_root, metric_keys)
    freeze = load_template_freeze(template_freeze_path)
    selections = [
        row for row in freeze.get("decoder_panel", [])
        if str(row["family"]) == str(decoder_family)
    ]
    if selections:
        if len(selections) != 1:
            raise RuntimeError(f"freeze has an invalid {decoder_family} decoder selection")
        frozen_model = str(selections[0]["model"])
        if str(decoder_model) != frozen_model:
            raise RuntimeError(
                f"constructor model {decoder_model!r} differs from frozen "
                f"{decoder_family} model {frozen_model!r}"
            )
    backend, revision = _backend(decoder_model, fake=fake_backends)
    if selections and not fake_backends and revision != str(selections[0]["revision"]):
        raise RuntimeError("constructor revision differs from the production freeze")
    root = Path(out_root).resolve()
    store_path = root / "evidence_cells.sqlite"
    reports = []
    try:
        with EvidenceCellStore(store_path) as store:
            for design in designs:
                metric_key = str(design["metric_key"])
                context = _metric_context(design)
                subset = _subset_panel_design(design, decoder_family)
                metric_root = root / "state_chunks" / _safe(metric_key) / decoder_family
                for instrument, templates in freeze["instruments"].items():
                    instrument_root = metric_root / str(instrument)
                    if "mcq" in channels:
                        path = instrument_root / "mcq_state_tables.npz"
                        if not path.exists():
                            menu = context["menu"]
                            result = evaluate_mcq_state_tables_v14(
                                backend, design_manifest=subset,
                                noun=str(context["codebook"]["reconstruction_noun"]),
                                target_metric_id=metric_key,
                                target_description=str(menu["entry"]["target_description"]),
                                distractors=menu["distractors"],
                                probe_texts=context["probe_texts"],
                                constructor_revision=revision, store=store,
                                template=str(templates["mcq"]),
                                max_chars=int(context["codebook"]["reconstruction_max_chars"]),
                                n_reconstruction_draws=8,
                                query_batch_size=int(query_batch_size),
                            )
                            _atomic_npz(
                                path, raw_lift=result["raw_lift"],
                                clipped_value=result["clipped_value"],
                                normalized_lift=result["normalized_lift"],
                                raw_target_probability=result["raw_target_probability"],
                                shuffled_target_probability=result["shuffled_target_probability"],
                                annotation_accuracy=result["annotation_accuracy"],
                            )
                            _atomic_json(instrument_root / "mcq_metadata.json", {
                                key: value for key, value in result.items()
                                if not isinstance(value, np.ndarray)
                            })
                    if "behavioral" in channels:
                        marker = instrument_root / "behavioral_induction.json"
                        if not marker.exists():
                            result = evaluate_behavioral_state_tables_v14(
                                backend, None, design_manifest=subset,
                                probe_texts=context["probe_texts"],
                                heldout_indices=design["probe_split"]["heldout"]["indices"],
                                heldout_target=(np.asarray(context["target"]["target"])[
                                    design["probe_split"]["heldout"]["indices"]
                                ] > 0.5).astype(int),
                                noun=str(context["codebook"]["reconstruction_noun"]),
                                decoder_revision=revision,
                                executor_revision=str(design["executor"]["revision"]),
                                readout_id=str(design["executor"]["readout_id"]),
                                store=store, templates=templates["behavioral"],
                                max_chars=int(context["codebook"]["reconstruction_max_chars"]),
                                query_batch_size=int(query_batch_size), induction_only=True,
                            )
                            _atomic_json(marker, result)
                reports.append({
                    "metric_key": metric_key, "decoder_family": decoder_family,
                    "decoder_model": decoder_model, "decoder_revision": revision,
                    "n_panels": len(subset["panels"]),
                })
    finally:
        release_resident_engines()
    _atomic_json(root / "constructor_stage" / f"{decoder_family}.json", {
        "schema": "cr3-v14-constructor-stage-v1", "rows": reports,
    })
    return reports


def run_executor_stage(
    *, out_root: str | Path, template_freeze_path: str | Path,
    metric_keys: Sequence[str] | None, fake_backends: bool,
    physical_gpu_ids: Sequence[int], query_batch_size: int,
    sentinel_report_path: str | Path | None = None,
    require_sentinel: bool = True,
) -> list[dict]:
    if require_sentinel:
        _require_live_sentinel(sentinel_report_path, fake_backends=fake_backends)
    assert_gpu_authorized(physical_gpu_ids, fake_backends=fake_backends)
    designs = load_designs(out_root, metric_keys)
    freeze = load_template_freeze(template_freeze_path)
    backend, revision = _backend(FIXED_EXECUTOR, fake=fake_backends)
    root = Path(out_root).resolve()
    reports = []
    try:
        with EvidenceCellStore(root / "evidence_cells.sqlite") as store:
            for design in designs:
                context = _metric_context(design)
                metric_key = str(design["metric_key"])
                if not fake_backends and revision != str(design["executor"]["revision"]):
                    raise RuntimeError("fixed executor revision differs from the frozen design")
                probe_embeddings = _certification_probe_embeddings(
                    context["probe_texts"],
                    cache_path=(
                        root / "certification_embeddings" / f"{_safe(metric_key)}.npz"
                    ),
                    fake_backends=fake_backends,
                )
                for family in DEFAULT_DECODER_MODELS:
                    constructor_report_path = root / "constructor_stage" / f"{family}.json"
                    if not constructor_report_path.exists():
                        raise RuntimeError(f"constructor stage is incomplete for {family}")
                    subset = _subset_panel_design(design, family)
                    family_root = root / "state_chunks" / _safe(metric_key) / family
                    for instrument, templates in freeze["instruments"].items():
                        instrument_root = family_root / str(instrument)
                        induction = json.loads(
                            (instrument_root / "behavioral_induction.json").read_text()
                        )
                        path = instrument_root / "behavioral_state_tables.npz"
                        if path.exists():
                            continue
                        result = evaluate_behavioral_state_tables_v14(
                            _CacheOnlyConstructor(), backend, design_manifest=subset,
                            probe_texts=context["probe_texts"],
                            heldout_indices=design["probe_split"]["heldout"]["indices"],
                            heldout_target=(np.asarray(context["target"]["target"])[
                                design["probe_split"]["heldout"]["indices"]
                            ] > 0.5).astype(int),
                            noun=str(context["codebook"]["reconstruction_noun"]),
                            decoder_revision=str(induction["decoder_revision"]),
                            executor_revision=str(design["executor"]["revision"]),
                            readout_id=str(design["executor"]["readout_id"]),
                            store=store, templates=templates["behavioral"],
                            max_chars=int(context["codebook"]["reconstruction_max_chars"]),
                            query_batch_size=int(query_batch_size),
                            probe_embeddings=probe_embeddings,
                        )
                        arrays = {}
                        metadata = {
                            key: value for key, value in result.items()
                            if key != "arms"
                        }
                        metadata["arms"] = {}
                        for arm, row in result["arms"].items():
                            for field, value in row.items():
                                if isinstance(value, np.ndarray):
                                    arrays[f"{arm}__{field}"] = value
                                else:
                                    metadata["arms"].setdefault(arm, {})[field] = value
                        _atomic_npz(path, **arrays)
                        _atomic_json(instrument_root / "behavioral_metadata.json", metadata)
                        reports.append({
                            "metric_key": metric_key, "decoder_family": family,
                            "instrument": instrument,
                            "n_panels": len(subset["panels"]),
                        })
    finally:
        release_resident_engines()
    _atomic_json(root / "executor_stage.json", {
        "schema": "cr3-v14-executor-stage-v1", "executor_revision": revision,
        "rows": reports,
    })
    return reports


def run_audit_proposer_stage(
    *, out_root: str | Path, audit_family: str, metric_keys: Sequence[str] | None,
    fake_backends: bool, physical_gpu_ids: Sequence[int], total_budget: int = 400,
    sentinel_report_path: str | Path | None = None,
) -> list[dict]:
    _require_live_sentinel(sentinel_report_path, fake_backends=fake_backends)
    assert_gpu_authorized(physical_gpu_ids, fake_backends=fake_backends)
    if audit_family not in AUDIT_FAMILIES:
        raise ValueError(f"unknown frozen audit family {audit_family!r}")
    designs = load_designs(out_root, metric_keys)
    model = AUDIT_FAMILIES[audit_family]
    backend, revision = _backend(model, fake=fake_backends)
    reports = []
    try:
        for design in designs:
            context = _metric_context(design)
            entry = context["codebook"]["entries"][str(design["metric_key"])]
            reports.append(propose_family_audit(
                backend, out_root=out_root, metric_key=str(design["metric_key"]),
                metric_name=str(entry.get("target_name") or design["metric_key"]),
                metric_description=str(entry["target_description"]),
                family=audit_family, model=model, model_revision=revision,
                total_budget=int(total_budget),
            ))
    finally:
        release_resident_engines()
    _atomic_json(Path(out_root) / "audit" / f"proposer_stage_{audit_family}.json", {
        "schema": "cr3-v14-audit-proposer-stage-v1", "family": audit_family,
        "model": model, "model_revision": revision, "rows": reports,
    })
    return reports


def run_audit_score_stage(
    *, out_root: str | Path, metric_keys: Sequence[str] | None,
    fake_backends: bool, physical_gpu_ids: Sequence[int], total_budget: int = 400,
    query_batch_size: int = 4096, sentinel_report_path: str | Path | None = None,
) -> list[dict]:
    _require_live_sentinel(sentinel_report_path, fake_backends=fake_backends)
    assert_gpu_authorized(physical_gpu_ids, fake_backends=fake_backends)
    designs = load_designs(out_root, metric_keys)
    executor, revision = _backend(FIXED_EXECUTOR, fake=fake_backends)
    reports = []
    try:
        with EvidenceCellStore(Path(out_root) / "evidence_cells.sqlite") as store:
            for design in designs:
                context = _metric_context(design)
                if not fake_backends and revision != str(design["executor"]["revision"]):
                    raise RuntimeError("audit executor revision differs from the frozen design")
                reports.append(score_audit_ledger(
                    executor, out_root=out_root, metric_key=str(design["metric_key"]),
                    probe_texts=context["probe_texts"], executor_revision=revision,
                    readout_id=str(design["executor"]["readout_id"]), store=store,
                    total_budget=int(total_budget),
                    max_chars=int(context["codebook"]["reconstruction_max_chars"]),
                    query_batch_size=int(query_batch_size),
                ))
    finally:
        release_resident_engines()
    _atomic_json(Path(out_root) / "audit" / "score_stage.json", {
        "schema": "cr3-v14-audit-score-stage-v1", "executor_revision": revision,
        "rows": reports,
    })
    return reports


def _combine_chunks(
    root: Path, design: Mapping[str, object], instrument: str, channel: str,
    arm: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.empty((50, 256), dtype=float)
    clipped = np.empty_like(raw)
    filled = np.zeros(50, dtype=bool)
    for family in DEFAULT_DECODER_MODELS:
        subset = _subset_panel_design(design, family)
        path = (
            root / "state_chunks" / _safe(design["metric_key"]) / family / instrument /
            ("mcq_state_tables.npz" if channel == "mcq" else "behavioral_state_tables.npz")
        )
        with np.load(path, allow_pickle=False) as artifact:
            if channel == "mcq":
                part_raw = np.asarray(artifact["raw_lift"], dtype=float)
                part_clipped = np.asarray(artifact["clipped_value"], dtype=float)
            else:
                part_raw = np.asarray(artifact[f"{arm}__raw_lift"], dtype=float)
                part_clipped = np.asarray(artifact[f"{arm}__clipped_value"], dtype=float)
        for position, panel in enumerate(subset["panels"]):
            trial = int(panel["original_trial"])
            raw[trial] = part_raw[position]
            clipped[trial] = part_clipped[position]
            filled[trial] = True
    if not np.all(filled):
        raise RuntimeError("combined v14 table lacks one or more frozen trials")
    validate_state_tables(raw, clipped)
    return raw, clipped


def _combine_behavioral_transfer_chunks(
    root: Path, design: Mapping[str, object], instrument: str, arm: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fields = ("near_raw_lift", "near_clipped_value", "far_raw_lift", "far_clipped_value")
    combined = {field: np.empty((50, 256), dtype=float) for field in fields}
    filled = np.zeros(50, dtype=bool)
    for family in DEFAULT_DECODER_MODELS:
        subset = _subset_panel_design(design, family)
        path = (
            root / "state_chunks" / _safe(design["metric_key"]) / family / instrument /
            "behavioral_state_tables.npz"
        )
        with np.load(path, allow_pickle=False) as artifact:
            parts = {
                field: np.asarray(artifact[f"{arm}__{field}"], dtype=float)
                for field in fields
            }
        for position, panel in enumerate(subset["panels"]):
            trial = int(panel["original_trial"])
            for field in fields:
                combined[field][trial] = parts[field][position]
            filled[trial] = True
    if not np.all(filled):
        raise RuntimeError("combined behavioral transfer table lacks frozen trials")
    validate_state_tables(combined["near_raw_lift"], combined["near_clipped_value"])
    validate_state_tables(combined["far_raw_lift"], combined["far_clipped_value"])
    return tuple(combined[field] for field in fields)


def _complete_historical_population(context: Mapping[str, object]) -> dict:
    """Restore provenance-valid frozen suffixes as pre-audit achieved evidence."""
    population = context["population"]
    signatures = [np.asarray(population["signatures"], dtype=float)]
    prompts = list(map(str, population["texts"]))
    families = list(map(str, population["families"]))
    streams = population.get("process_streams")
    if streams is not None:
        for stream_position, stream in enumerate(streams):
            provenance = stream["provenance"]
            source_path = Path(provenance["source_path"])
            indices = list(map(
                int, provenance["source_row_indices_never_absorbed_audit_suffix"],
            ))
            with np.load(source_path, allow_pickle=True) as artifact:
                field = "texts" if "texts" in artifact.files else "prompts"
                source_texts = [str(artifact[field][index]) for index in indices]
            suffix = np.asarray(stream["audit_suffix_signatures"], dtype=float)
            if suffix.shape != (len(source_texts), signatures[0].shape[1]):
                raise RuntimeError("historical audit suffix text/signature rows are misaligned")
            signatures.append(suffix)
            prompts.extend(source_texts)
            families.extend([str(stream["family"])] * len(source_texts))
    matrix = np.vstack(signatures)
    ids = [
        f"historical:{index}:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        for index, text in enumerate(prompts)
    ]
    return {
        "signatures": matrix, "prompts": prompts, "prompt_ids": ids,
        "families": families, "record_rank_provenance_valid": streams is not None,
        "n_harvested_draws": len(prompts),
    }


def _control_diagnostics(
    root: Path, design: Mapping[str, object], instrument: str, channel: str,
    arm: str | None,
) -> dict:
    blind = []
    best = []
    explanation = []
    for family in DEFAULT_DECODER_MODELS:
        subset = _subset_panel_design(design, family)
        source = (
            root / "state_chunks" / _safe(design["metric_key"]) / family / instrument
        )
        if channel == "mcq":
            metadata = json.loads((source / "mcq_metadata.json").read_text())
            with np.load(source / "mcq_state_tables.npz", allow_pickle=False) as artifact:
                signal = np.asarray(artifact["raw_target_probability"], dtype=float)
            blind.extend([float(metadata["blind"]["target_probability"])] * len(subset["panels"]))
            best.extend(np.max(signal, axis=1).astype(float).tolist())
            if metadata.get("best_explanation_rate") is not None:
                explanation.append(float(metadata["best_explanation_rate"]))
        else:
            metadata = json.loads((source / "behavioral_metadata.json").read_text())
            with np.load(source / "behavioral_state_tables.npz", allow_pickle=False) as artifact:
                signal = np.asarray(artifact[f"{arm}__raw_mi"], dtype=float)
            blind.extend([
                float(metadata["arms"][str(arm)]["blind_mi"])
            ] * len(subset["panels"]))
            best.extend(np.max(signal, axis=1).astype(float).tolist())
    blind_mean = float(np.mean(blind))
    best_mean = float(np.mean(best))
    return {
        "blind_signal": blind_mean,
        "best_annotated_signal": best_mean,
        "best_minus_blind_signal": best_mean - blind_mean,
        "best_explanation_rate": (
            float(np.mean(explanation)) if explanation else None
        ),
    }


def aggregate_campaign(
    *, out_root: str | Path, template_freeze_path: str | Path,
    metric_keys: Sequence[str] | None, channels: Sequence[str],
) -> pd.DataFrame:
    root = Path(out_root).resolve()
    designs = load_designs(root, metric_keys)
    freeze = load_template_freeze(template_freeze_path)
    result_rows = []
    value_tables = {}
    for design in designs:
        context = _metric_context(design)
        metric_key = str(design["metric_key"])
        historical = _complete_historical_population(context)
        prompts = historical["prompts"]
        prompt_ids = historical["prompt_ids"]
        signatures = historical["signatures"]
        panels = [row["indices"] for row in design["panel_design"]["panels"]]
        families = [row["decoder_family"] for row in design["panel_design"]["panels"]]
        heldout = design["probe_split"]["heldout"]["indices"]
        target_h = (np.asarray(context["target"]["target"])[heldout] > 0.5).astype(int)
        for instrument in freeze["instruments"]:
            channel_arms = []
            if "mcq" in channels:
                channel_arms.append(("mcq", None))
            if "behavioral" in channels:
                channel_arms.extend(("behavioral", arm) for arm in BEHAVIORAL_ARMS)
            for channel, arm in channel_arms:
                raw, clipped = _combine_chunks(root, design, instrument, channel, arm)
                control = _control_diagnostics(
                    root, design, str(instrument), channel, arm,
                )
                discovery = aggregate_state_tables(
                    raw_lift=raw, clipped_value=clipped,
                    prompt_signatures=signatures, panels=panels,
                    prompt_ids=prompt_ids, decoder_families=families,
                )
                audit_path = root / "audit" / "signatures" / f"{metric_key}.npz"
                audit = None
                process_bounds = None
                novelty_rows = None
                all_signatures = signatures
                all_ids = prompt_ids
                all_prompts = prompts
                if audit_path.is_file():
                    with np.load(audit_path, allow_pickle=True) as artifact:
                        audit_signatures = np.asarray(artifact["sigs"], dtype=float)
                        audit_prompts = list(map(str, artifact["prompts"]))
                        audit_families = list(map(str, artifact["families"]))
                    audit_ids = [
                        f"audit:{index}:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
                        for index, text in enumerate(audit_prompts)
                    ]
                    audit = aggregate_state_tables(
                        raw_lift=raw, clipped_value=clipped,
                        prompt_signatures=audit_signatures, panels=panels,
                        prompt_ids=audit_ids, decoder_families=families,
                    )
                    all_signatures = np.vstack((signatures, audit_signatures))
                    all_ids = [*prompt_ids, *audit_ids]
                    all_prompts = [*prompts, *audit_prompts]
                aggregate = aggregate_state_tables(
                    raw_lift=raw, clipped_value=clipped,
                    prompt_signatures=all_signatures, panels=panels,
                    prompt_ids=all_ids, decoder_families=families,
                )
                transfer = None
                transfer_tables = None
                if channel == "behavioral":
                    transfer_tables = _combine_behavioral_transfer_chunks(
                        root, design, str(instrument), str(arm),
                    )
                    near_aggregate = aggregate_state_tables(
                        raw_lift=transfer_tables[0], clipped_value=transfer_tables[1],
                        prompt_signatures=all_signatures, panels=panels,
                        prompt_ids=all_ids, decoder_families=families,
                    )
                    far_aggregate = aggregate_state_tables(
                        raw_lift=transfer_tables[2], clipped_value=transfer_tables[3],
                        prompt_signatures=all_signatures, panels=panels,
                        prompt_ids=all_ids, decoder_families=families,
                    )
                    representative = str(
                        aggregate["legibility_argmax"]["canonical_representative"]
                    )
                    representative_position = all_ids.index(representative)
                    near_value = float(near_aggregate["prompt_value"][representative_position])
                    far_value = float(far_aggregate["prompt_value"][representative_position])
                    transfer = {
                        "embedding_model": "BAAI/bge-large-en-v1.5",
                        "near_value_at_legibility_representative": near_value,
                        "far_value_at_legibility_representative": far_value,
                        "far_near_ratio": (
                            float(far_value / near_value) if near_value > 1e-12 else None
                        ),
                        "near_independent_cap": near_aggregate["free_recombination_cap"],
                        "far_independent_cap": far_aggregate["free_recombination_cap"],
                        "diagnostic_only_not_primary_value": True,
                    }
                value_tables[(metric_key, str(instrument), channel, arm)] = np.asarray(
                    aggregate["prompt_value"], dtype=float
                )
                if audit is not None:
                    cell_alpha = float(freeze.get("process_bound_cell_alpha", 0.05))
                    stochastic_bound_alpha = cell_alpha / 2.0
                    process_bounds = {}
                    for horizon in (100, 300):
                        rows = {
                            "record_rank": record_rank_gain_bound(
                                n=(
                                    len(audit["prompt_value"])
                                    + (
                                        len(discovery["prompt_value"])
                                        if historical["record_rank_provenance_valid"] else 0
                                    )
                                ), m=horizon,
                                achieved=aggregate["achieved_value"],
                                cap=aggregate["free_recombination_cap"],
                            ),
                            "split_cp": split_sample_cp_gain_bound(
                                discovery_achieved=discovery["achieved_value"],
                                audit_values=audit["prompt_value"],
                                current_achieved=aggregate["achieved_value"],
                                cap=aggregate["free_recombination_cap"],
                                future_horizon=horizon, alpha=stochastic_bound_alpha,
                            ),
                            "dkw": dkw_expected_best_gain_bound(
                                observed_values=audit["prompt_value"],
                                achieved=aggregate["achieved_value"],
                                cap=aggregate["free_recombination_cap"],
                                future_horizon=horizon, alpha=stochastic_bound_alpha,
                            ),
                        }
                        per_family = {}
                        for audit_family in sorted(set(audit_families)):
                            mask = np.asarray(audit_families) == audit_family
                            family_values = np.asarray(audit["prompt_value"])[mask]
                            family_rows = {
                                "record_rank": record_rank_gain_bound(
                                    n=len(family_values), m=horizon,
                                    achieved=aggregate["achieved_value"],
                                    cap=aggregate["free_recombination_cap"],
                                ),
                                "split_cp": split_sample_cp_gain_bound(
                                    discovery_achieved=discovery["achieved_value"],
                                    audit_values=family_values,
                                    current_achieved=aggregate["achieved_value"],
                                    cap=aggregate["free_recombination_cap"],
                                    future_horizon=horizon, alpha=stochastic_bound_alpha,
                                ),
                                "dkw": dkw_expected_best_gain_bound(
                                    observed_values=family_values,
                                    achieved=aggregate["achieved_value"],
                                    cap=aggregate["free_recombination_cap"],
                                    future_horizon=horizon, alpha=stochastic_bound_alpha,
                                ),
                            }
                            per_family[audit_family] = {
                                "diagnostic_only": True, "bounds": family_rows,
                                "minimum_gain_upper": min(
                                    float(row["gain_upper"]) for row in family_rows.values()
                                ),
                            }
                        process_bounds[str(horizon)] = {
                            "bounds": rows,
                            "headline_gain_upper": min(
                                float(row["gain_upper"]) for row in rows.values()
                            ),
                            "headline_rule": "minimum of all premise-valid declared bounds",
                            "stochastic_bound_alpha": stochastic_bound_alpha,
                            "within_claim_alpha_split": "equal split across split-CP and DKW",
                            "per_family_diagnostics": per_family,
                        }
                    discovery_behavior_hashes = [
                        hashlib.sha256(np.ascontiguousarray(row).view(np.uint8)).hexdigest()
                        for row in signatures
                    ]
                    discovery_codes = signatures_to_states(signatures, panels)
                    discovery_code_hashes = [
                        hashlib.sha256(np.ascontiguousarray(row).view(np.uint8)).hexdigest()
                        for row in discovery_codes
                    ]
                    novelty_rows = novelty_collapse_curves(
                        full_signatures=audit_signatures,
                        joint_codes=signatures_to_states(audit_signatures, panels),
                        values=audit["prompt_value"],
                        frozen_incumbent=discovery["achieved_value"],
                        discovery_signature_hashes=discovery_behavior_hashes,
                        discovery_code_hashes=discovery_code_hashes,
                        families=audit_families,
                    )
                distortion = fidelity_legibility_diagnostic(
                    prompt_signatures_on_h=all_signatures[:, heldout], target_on_h=target_h,
                    legibility_values=aggregate["prompt_value"], prompt_ids=all_ids,
                )
                status = classify_status(
                    achieved=aggregate["achieved_value"],
                    cap=aggregate["free_recombination_cap"],
                    raw_panel_caps=aggregate["raw_panel_caps"],
                    future_gain_bound=(
                        None if process_bounds is None
                        else process_bounds["100"]["headline_gain_upper"]
                    ),
                    blind_value=control["blind_signal"],
                    best_annotated_value=control["best_annotated_signal"],
                )
                canonical_states = [
                    int("".join(map(str, panel["target_state_bits"])), 2)
                    for panel in design["panel_design"]["panels"]
                ]
                canonical_raw_lift = float(np.mean([
                    raw[position, state] for position, state in enumerate(canonical_states)
                ]))
                canonical_value = float(np.mean([
                    clipped[position, state] for position, state in enumerate(canonical_states)
                ]))
                artifact_root = (
                    root / "certificates" / _safe(metric_key) / str(instrument) /
                    (channel if arm is None else f"{channel}__{arm}")
                )
                state_arrays = {"raw_lift": raw, "clipped_value": clipped}
                if transfer_tables is not None:
                    state_arrays.update(dict(zip((
                        "near_raw_lift", "near_clipped_value",
                        "far_raw_lift", "far_clipped_value",
                    ), transfer_tables)))
                _atomic_npz(artifact_root / "state_tables.npz", **state_arrays)
                _atomic_json(artifact_root / "design_manifest.json", design)
                frame = pd.DataFrame({
                    "prompt_id": all_ids, "prompt_text": all_prompts,
                    "raw_lift": aggregate["prompt_raw_lift"],
                    "value": aggregate["prompt_value"],
                    "fidelity_bits": distortion["fidelity"],
                })
                _atomic_parquet(artifact_root / "prompt_values.parquet", frame)
                if novelty_rows is not None:
                    _atomic_parquet(
                        artifact_root / "novelty_curves.parquet", pd.DataFrame(novelty_rows)
                    )
                certificate = {
                    "schema": "cr3-v14-certificate-v1",
                    "task": design["task"], "level": design["level"],
                    "metric": design["metric"], "metric_key": metric_key,
                    "instrument": instrument, "channel": channel, "arm": arm,
                    "executor": design["executor"],
                    "design_manifest_sha256": design["design_manifest_sha256"],
                    "achieved_value": aggregate["achieved_value"],
                    "free_recombination_cap": aggregate["free_recombination_cap"],
                    "structural_gap": aggregate["structural_gap"],
                    "annotated_canonical_raw_lift": canonical_raw_lift,
                    "annotated_canonical_value": canonical_value,
                    "control_diagnostics": control,
                    "near_far_transfer_diagnostic": transfer,
                    "panel_caps": aggregate["panel_caps"].tolist(),
                    "decoder_family_rows": aggregate.get("decoder_family_rows"),
                    "decoder_family_achieved_variance": aggregate.get(
                        "decoder_family_achieved_variance"
                    ),
                    "legibility_argmax": aggregate["legibility_argmax"],
                    "fidelity_argmax": distortion["fidelity_argmax"],
                    "oracle_omega_star": {
                        "definition": "decoder_free_argmax_of_behavioral_fidelity_on_H",
                        "tie_class": distortion["fidelity_argmax"],
                    },
                    "decoder_specific_optimum_instance": {
                        "definition": "reader_relative_argmax_of_reconstruction_legibility",
                        "tie_class": aggregate["legibility_argmax"],
                        "ranking_claim_allowed": distortion[
                            "optimal_prompt_ranking_allowed"
                        ],
                        "reporting_label": (
                            "decoder_specific_optimum_instance"
                            if distortion["optimal_prompt_ranking_allowed"]
                            else "tie_class_with_canonical_representative_no_optimal_claim"
                        ),
                    },
                    "distortion": {
                        key: value for key, value in distortion.items()
                        if key != "fidelity"
                    },
                    "status": status,
                    "behavioral_ceiling_contract": (
                        None if channel == "mcq" else {
                            "target_entropy_on_H_bits": design["target_entropy_on_h_bits"],
                            "identification_mi_is_not_a_behavioral_ceiling": True,
                        }
                    ),
                    "identification_diagnostic": {
                        "mean_I_z_M_bits": float(np.mean([
                            row["identification_mi_bits"]
                            for row in design["panel_design"]["panels"]
                        ])),
                        "mean_target_margin": float(np.mean([
                            row["target_margin"] for row in design["panel_design"]["panels"]
                        ])),
                        "n_target_uniqueness_exceptions": int(sum(
                            not row["target_unique"]
                            for row in design["panel_design"]["panels"]
                        )),
                        "scope": "identification_ceiling_and_panel_quality_diagnostic_only",
                        "behavioral_ceiling": False,
                    },
                    "ceiling_table": {
                        "instrument_free_recombination_cap": {
                            "value": aggregate["free_recombination_cap"],
                            "moves_with_decoder_tuning": True,
                            "role": "descriptive_ruler_not_headline",
                        },
                        "identification_I_z_M_bits": {
                            "value": float(np.mean([
                                row["identification_mi_bits"]
                                for row in design["panel_design"]["panels"]
                            ])),
                            "moves_with_decoder_tuning": False,
                            "role": "identification_only_not_behavioral_ceiling",
                        },
                        "target_entropy_on_H_bits": {
                            "value": design["target_entropy_on_h_bits"],
                            "moves_with_decoder_tuning": False,
                            "role": "behavioral_total_information_ceiling",
                        },
                    },
                    "gold_fidelity": design["entry"].get("gold_fidelity"),
                    "gold_fidelity_available": design["entry"].get("gold_fidelity") is not None,
                    "record_rank_harvested_provenance_valid": historical[
                        "record_rank_provenance_valid"
                    ],
                    "n_harvested_draws": historical["n_harvested_draws"],
                    "discovery_achieved_value": discovery["achieved_value"],
                    "audit_achieved_value": None if audit is None else audit["achieved_value"],
                    "process_relative_bounds": (
                        "pending_fresh_audit_stage" if process_bounds is None else process_bounds
                    ),
                }
                certificate["certificate_sha256"] = canonical_sha256(certificate)
                _atomic_json(artifact_root / "certificate.json", certificate)
                result_rows.append({
                    "task": design["task"], "level": design["level"],
                    "metric": design["metric"], "metric_key": metric_key,
                    "instrument": instrument, "channel": channel, "arm": arm,
                    "executor": design["executor"]["model"],
                    "decoder_panel": "qwen14+llama70+mistral24",
                    "achieved_value": aggregate["achieved_value"],
                    "structural_cap": aggregate["free_recombination_cap"],
                    "structural_gap": aggregate["structural_gap"],
                    "decoder_family_variance": aggregate.get("decoder_family_achieved_variance"),
                    "gold_fidelity": design["entry"].get("gold_fidelity"),
                    "blind_to_best_gap": control["best_minus_blind_signal"],
                    "far_near_transfer_ratio": (
                        None if transfer is None else transfer["far_near_ratio"]
                    ),
                    "fidelity_legibility_spearman": distortion["spearman_rho"],
                    "ranking_allowed": distortion["optimal_prompt_ranking_allowed"],
                    "status": status["status"],
                    "certificate_path": str(artifact_root / "certificate.json"),
                })
    for row in result_rows:
        key = (str(row["metric_key"]), str(row["instrument"]))
        mcq = value_tables.get((*key, "mcq", None))
        behavioral = value_tables.get((*key, "behavioral", "unconstrained"))
        if mcq is None or behavioral is None or len(mcq) < 2:
            correlation = None
        elif np.all(mcq == mcq[0]) or np.all(behavioral == behavioral[0]):
            correlation = None
        else:
            observed = float(spearmanr(mcq, behavioral).statistic)
            correlation = observed if np.isfinite(observed) else None
        row["mcq_behavioral_spearman"] = correlation
        behavioral_rows = {
            str(candidate["arm"]): float(candidate["achieved_value"])
            for candidate in result_rows
            if str(candidate["metric_key"]) == key[0]
            and str(candidate["instrument"]) == key[1]
            and candidate["channel"] == "behavioral"
        }
        row["exemplar_vs_rule_achieved_gap_bits"] = (
            behavioral_rows.get("unconstrained", 0.0)
            - behavioral_rows.get("no_verbatim_examples", 0.0)
            if set(behavioral_rows) == set(BEHAVIORAL_ARMS) else None
        )
    frame = pd.DataFrame(result_rows)
    _atomic_parquet(root / "results.parquet", frame)
    _atomic_json(root / "campaign_manifest.json", {
        "schema": CAMPAIGN_SCHEMA,
        "n_metrics": len(designs), "n_results": len(frame),
        "channels": list(channels), "template_freeze": str(Path(template_freeze_path).resolve()),
        "results_path": str(root / "results.parquet"),
        "gpu_restriction": "sk3 physical GPUs 1,2,3,4 forbidden",
        "scientific_gate_applied": False,
        "control_liveness_gate_applied": (root / "sentinel_report.json").is_file(),
        "sentinel_report": (
            str(root / "sentinel_report.json")
            if (root / "sentinel_report.json").is_file() else None
        ),
        "fresh_audit_complete": all(
            (root / "audit" / "signatures" / f"{design['metric_key']}.npz").is_file()
            for design in designs
        ),
    })
    return frame


def _decoder_models(arguments: Sequence[str] | None) -> dict[str, str]:
    if not arguments:
        return dict(DEFAULT_DECODER_MODELS)
    output = {}
    for value in arguments:
        if "=" not in value:
            raise ValueError("--decoder-model entries must be family=model")
        family, model = value.split("=", 1)
        output[family] = model
    if set(output) != set(DEFAULT_DECODER_MODELS):
        raise ValueError("decoder models must define qwen, llama, and mistral")
    return output


def _selected_constructor_model(
    template_freeze_path: str | Path, decoder_family: str,
    arguments: Sequence[str] | None,
) -> str:
    freeze = load_template_freeze(template_freeze_path)
    frozen = [
        row for row in freeze.get("decoder_panel", [])
        if str(row["family"]) == str(decoder_family)
    ]
    if not frozen:
        return _decoder_models(arguments)[decoder_family]
    if len(frozen) != 1:
        raise RuntimeError(f"freeze has an invalid {decoder_family} decoder selection")
    model = str(frozen[0]["model"])
    if arguments:
        requested = _decoder_models(arguments)[decoder_family]
        if requested != model:
            raise RuntimeError(
                f"requested constructor {requested!r} differs from frozen model {model!r}"
            )
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=[
        "design", "prepare-dev", "prepare-sentinel", "qualify-constructor",
        "qualify-executor", "tune", "freeze", "seed-freeze", "constructor", "executor",
        "sentinel-constructor", "sentinel-executor", "sentinel-aggregate",
        "liveness-constructor", "liveness-executor", "sentinel-gate",
        "audit-proposer", "audit-score", "aggregate", "report",
    ])
    parser.add_argument("--metrics-manifest")
    parser.add_argument("--dev-metrics-manifest")
    parser.add_argument("--sentinel-metrics-manifest")
    parser.add_argument("--sentinel-metric-keys", nargs="+", default=[
        "humor_R3_metric0", "humor_R3_metric10", "humor_R3_metric11",
        "humor_R3_metric12", "humor_R3_metric34", "humor_R3_metric50",
    ])
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--run-sha", default="v14.0")
    parser.add_argument("--metric-keys", nargs="+")
    parser.add_argument("--template-freeze")
    parser.add_argument("--decoder-family", choices=sorted(DEFAULT_DECODER_MODELS))
    parser.add_argument("--audit-family", choices=sorted(AUDIT_FAMILIES))
    parser.add_argument("--decoder-model", action="append")
    parser.add_argument("--tuning-channel", choices=["mcq", "behavioral"])
    parser.add_argument("--tuning-arm", default="unconstrained")
    parser.add_argument("--proposer-model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--qualification", action="append")
    parser.add_argument("--release-commit")
    parser.add_argument("--sentinel-controls")
    parser.add_argument("--sentinel-report")
    parser.add_argument("--channels", nargs="+", choices=["mcq", "behavioral"],
                        default=["mcq", "behavioral"])
    parser.add_argument("--query-batch-size", type=int, default=2048)
    parser.add_argument("--audit-budget", type=int, default=400)
    parser.add_argument("--physical-gpus", nargs="*")
    parser.add_argument("--fake-backends", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.out_root).resolve()
    if args.phase == "design":
        if not args.metrics_manifest:
            raise ValueError("design phase requires --metrics-manifest")
        build_designs(
            metrics_manifest_path=args.metrics_manifest, out_root=root,
            run_sha=args.run_sha, metric_keys=args.metric_keys,
        )
    elif args.phase == "prepare-dev":
        if not args.dev_metrics_manifest:
            raise ValueError("prepare-dev phase requires --dev-metrics-manifest")
        prepare_development_population(
            certified_out_root=root,
            dev_metrics_manifest_path=args.dev_metrics_manifest,
            run_sha=args.run_sha,
        )
    elif args.phase == "prepare-sentinel":
        if not args.sentinel_metrics_manifest:
            raise ValueError("prepare-sentinel requires --sentinel-metrics-manifest")
        prepare_sentinel_population(
            out_root=root,
            sentinel_metrics_manifest_path=args.sentinel_metrics_manifest,
            run_sha=args.run_sha, sentinel_metric_keys=args.sentinel_metric_keys,
        )
    elif args.phase == "qualify-constructor":
        if not args.decoder_family:
            raise ValueError("qualify-constructor requires --decoder-family")
        model = _decoder_models(args.decoder_model)[args.decoder_family]
        run_qualification_constructor(
            out_root=root, decoder_family=args.decoder_family, decoder_model=model,
            fake_backends=args.fake_backends,
            physical_gpu_ids=parse_physical_gpu_ids(args.physical_gpus),
            query_batch_size=args.query_batch_size,
        )
    elif args.phase == "qualify-executor":
        if not args.decoder_family:
            raise ValueError("qualify-executor requires --decoder-family")
        model = _decoder_models(args.decoder_model)[args.decoder_family]
        run_qualification_executor(
            out_root=root, decoder_family=args.decoder_family, decoder_model=model,
            fake_backends=args.fake_backends,
            physical_gpu_ids=parse_physical_gpu_ids(args.physical_gpus),
            query_batch_size=args.query_batch_size,
        )
    elif args.phase == "tune":
        if not args.tuning_channel:
            raise ValueError("tune phase requires --tuning-channel")
        run_decoder_tuning(
            out_root=root, channel=args.tuning_channel, arm=args.tuning_arm,
            decoder_models=_decoder_models(args.decoder_model),
            proposer_model=args.proposer_model, fake_backends=args.fake_backends,
            physical_gpu_ids=parse_physical_gpu_ids(args.physical_gpus),
            query_batch_size=args.query_batch_size,
        )
    elif args.phase == "freeze":
        if not args.qualification or not args.release_commit:
            raise ValueError("freeze requires three --qualification specs and --release-commit")
        freeze_production_instrument(
            out_root=root, qualification_specs=args.qualification,
            release_commit=args.release_commit,
        )
    elif args.phase == "sentinel-gate":
        if not args.sentinel_controls:
            raise ValueError("sentinel-gate requires --sentinel-controls")
        apply_sentinel_gate(
            control_rows_path=args.sentinel_controls, out_root=root,
        )
    elif args.phase == "liveness-constructor":
        if not args.decoder_family:
            raise ValueError("liveness-constructor requires --decoder-family")
        run_liveness_constructor_stage(
            out_root=root, decoder_family=args.decoder_family,
            fake_backends=args.fake_backends,
            physical_gpu_ids=parse_physical_gpu_ids(args.physical_gpus),
            query_batch_size=args.query_batch_size,
        )
    elif args.phase == "liveness-executor":
        run_liveness_executor_stage(
            out_root=root, fake_backends=args.fake_backends,
            physical_gpu_ids=parse_physical_gpu_ids(args.physical_gpus),
            query_batch_size=args.query_batch_size,
        )
    elif args.phase == "seed-freeze":
        destination = Path(args.template_freeze or root / "template_freeze.json")
        if destination.exists():
            raise FileExistsError(destination)
        write_seed_template_freeze(destination)
    elif args.phase == "constructor":
        if not args.template_freeze or not args.decoder_family:
            raise ValueError("constructor phase requires --template-freeze and --decoder-family")
        model = _selected_constructor_model(
            args.template_freeze, args.decoder_family, args.decoder_model,
        )
        run_constructor_stage(
            out_root=root, template_freeze_path=args.template_freeze,
            decoder_family=args.decoder_family,
            decoder_model=model, metric_keys=args.metric_keys,
            fake_backends=args.fake_backends,
            physical_gpu_ids=parse_physical_gpu_ids(args.physical_gpus),
            channels=args.channels, query_batch_size=args.query_batch_size,
            sentinel_report_path=args.sentinel_report,
        )
    elif args.phase == "sentinel-constructor":
        if not args.template_freeze or not args.decoder_family:
            raise ValueError(
                "sentinel-constructor requires --template-freeze and --decoder-family"
            )
        model = _selected_constructor_model(
            args.template_freeze, args.decoder_family, args.decoder_model,
        )
        run_constructor_stage(
            out_root=root / "sentinel", template_freeze_path=args.template_freeze,
            decoder_family=args.decoder_family, decoder_model=model,
            metric_keys=args.metric_keys, fake_backends=args.fake_backends,
            physical_gpu_ids=parse_physical_gpu_ids(args.physical_gpus),
            channels=args.channels, query_batch_size=args.query_batch_size,
            require_sentinel=False,
        )
    elif args.phase == "executor":
        if not args.template_freeze:
            raise ValueError("executor phase requires --template-freeze")
        run_executor_stage(
            out_root=root, template_freeze_path=args.template_freeze,
            metric_keys=args.metric_keys, fake_backends=args.fake_backends,
            physical_gpu_ids=parse_physical_gpu_ids(args.physical_gpus),
            query_batch_size=args.query_batch_size,
            sentinel_report_path=args.sentinel_report,
        )
    elif args.phase == "sentinel-executor":
        if not args.template_freeze:
            raise ValueError("sentinel-executor requires --template-freeze")
        run_executor_stage(
            out_root=root / "sentinel", template_freeze_path=args.template_freeze,
            metric_keys=args.metric_keys, fake_backends=args.fake_backends,
            physical_gpu_ids=parse_physical_gpu_ids(args.physical_gpus),
            query_batch_size=args.query_batch_size, require_sentinel=False,
        )
    elif args.phase == "sentinel-aggregate":
        if not args.template_freeze:
            raise ValueError("sentinel-aggregate requires --template-freeze")
        aggregate_campaign(
            out_root=root / "sentinel", template_freeze_path=args.template_freeze,
            metric_keys=args.metric_keys, channels=args.channels,
        )
    elif args.phase == "audit-proposer":
        if not args.audit_family:
            raise ValueError("audit-proposer phase requires --audit-family")
        run_audit_proposer_stage(
            out_root=root, audit_family=args.audit_family, metric_keys=args.metric_keys,
            fake_backends=args.fake_backends,
            physical_gpu_ids=parse_physical_gpu_ids(args.physical_gpus),
            total_budget=args.audit_budget,
            sentinel_report_path=args.sentinel_report,
        )
    elif args.phase == "audit-score":
        run_audit_score_stage(
            out_root=root, metric_keys=args.metric_keys,
            fake_backends=args.fake_backends,
            physical_gpu_ids=parse_physical_gpu_ids(args.physical_gpus),
            total_budget=args.audit_budget, query_batch_size=args.query_batch_size,
            sentinel_report_path=args.sentinel_report,
        )
    elif args.phase == "aggregate":
        if not args.template_freeze:
            raise ValueError("aggregate phase requires --template-freeze")
        aggregate_campaign(
            out_root=root, template_freeze_path=args.template_freeze,
            metric_keys=args.metric_keys, channels=args.channels,
        )
    elif args.phase == "report":
        report = audit_release(root)
        write_release_outputs(root, report)
        if not report["complete"]:
            raise RuntimeError("v14 release completion audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
