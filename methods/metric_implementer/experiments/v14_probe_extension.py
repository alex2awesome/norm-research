"""Append-only 90-probe extension for the v14.1 120/30/240 split."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..batch_scoring import _YESNO_TEMPLATE
from .cr3_reconstruction_values import _bootstrap
from .v14_panel_design import canonical_sha256


SCHEMA = "cr3-v14-probe-extension-v1"
N_EXTENSION_PROBES = 90


def load_task_candidates(spec: Mapping[str, object], *, base: str | Path) -> list[str]:
    path = Path(str(spec["path"]))
    if not path.is_absolute():
        path = Path(base) / path
    field = str(spec.get("text_field", "text"))
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        values = pd.read_parquet(path, columns=[field])[field].tolist()
    elif suffix in {".jsonl", ".ndjson"}:
        values = [json.loads(line)[field] for line in path.read_text().splitlines() if line.strip()]
    elif suffix == ".json":
        payload = json.loads(path.read_text())
        rows = payload if isinstance(payload, list) else payload[str(spec.get("rows_field", "rows"))]
        values = [row[field] if isinstance(row, Mapping) else row for row in rows]
    else:
        values = [line for line in path.read_text().splitlines() if line.strip()]
    return list(map(str, values))


def normalize_probe_text(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def select_extension_texts(
    candidates: Sequence[str], *, existing_texts: Sequence[str], task: str,
    run_sha: str, n: int = N_EXTENSION_PROBES,
) -> list[str]:
    """Stable-hash sample unused, normalized-distinct texts from the same task corpus."""
    excluded = {normalize_probe_text(value) for value in existing_texts}
    unique = {}
    for value in map(str, candidates):
        normalized = normalize_probe_text(value)
        if normalized and normalized not in excluded:
            unique.setdefault(normalized, value)
    ranked = sorted(
        unique.items(),
        key=lambda row: (
            hashlib.sha256(
                f"{run_sha}\x1f{task}\x1fprobe-extension\x1f{row[0]}".encode()
            ).hexdigest(),
            row[0],
        ),
    )
    if len(ranked) < int(n):
        raise ValueError(f"task {task} has only {len(ranked)} unused distinct probes; need {n}")
    return [value for _, value in ranked[:int(n)]]


def score_extension_codebook(
    executor, *, codebook: Mapping[str, object], extension_texts: Sequence[str],
    executor_revision: str, readout_id: str, query_batch_size: int = 2048,
) -> dict:
    """Score exact frozen description-form orbits on only the appended probes."""
    texts = list(map(str, extension_texts))
    keys = sorted(map(str, codebook["metrics"]))
    scores = np.empty((len(keys), len(texts)), dtype=np.float32)
    forms_by_key = {}
    for row, key in enumerate(keys):
        metric = codebook["metrics"][key]
        bootstrap = _bootstrap(metric["bootstrap_path"])
        forms = [str(value).strip() for value in bootstrap.get(
            "target_form_texts", [metric["description"]]
        ) if str(value).strip()]
        forms = list(dict.fromkeys(forms))
        forms_by_key[key] = forms
        form_scores = []
        for form in forms:
            values = []
            for start in range(0, len(texts), int(query_batch_size)):
                batch = texts[start:start + int(query_batch_size)]
                prompts = [_YESNO_TEMPLATE.format(rubric=form, text=text) for text in batch]
                values.extend(executor.score_binary_constrained(
                    prompts, system=None, pos="YES", neg="NO", seed=0,
                ))
            form_scores.append(np.asarray(values, dtype=float))
        scores[row] = np.mean(np.vstack(form_scores), axis=0).astype(np.float32)
    if np.any(~np.isfinite(scores)):
        raise RuntimeError("probe-extension executor scores are non-finite")
    return {
        "schema": SCHEMA, "metric_keys": keys, "texts": texts, "scores": scores,
        "forms_sha256": canonical_sha256(forms_by_key),
        "executor_revision": str(executor_revision), "readout_id": str(readout_id),
    }


def write_extension(path: str | Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    core = {
        "schema": payload["schema"], "metric_keys": list(payload["metric_keys"]),
        "texts": list(payload["texts"]), "forms_sha256": payload["forms_sha256"],
        "executor_revision": payload["executor_revision"], "readout_id": payload["readout_id"],
    }
    core["sha256"] = canonical_sha256(core)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(
        temporary, metadata=np.asarray(json.dumps(core, sort_keys=True)),
        scores=np.asarray(payload["scores"], dtype=np.float32),
    )
    os.replace(temporary, destination)


def load_extension(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as artifact:
        metadata = json.loads(str(artifact["metadata"]))
        scores = np.asarray(artifact["scores"], dtype=float)
    observed = str(metadata.pop("sha256"))
    if canonical_sha256(metadata) != observed:
        raise RuntimeError(f"probe extension checksum mismatch: {path}")
    if (metadata.get("schema") != SCHEMA or len(metadata["texts"]) != N_EXTENSION_PROBES
            or scores.shape != (len(metadata["metric_keys"]), N_EXTENSION_PROBES)
            or np.any(~np.isfinite(scores))):
        raise RuntimeError(f"invalid probe extension: {path}")
    return {**metadata, "sha256": observed, "scores": scores}


def append_extension_to_split(base_split: Mapping[str, object], all_probe_ids: Sequence[str]) -> dict:
    """Append indices 300..389 to H without perturbing the frozen first-300 split."""
    import copy

    result = copy.deepcopy(dict(base_split))
    ids = list(map(str, all_probe_ids))
    old_n = int(result["n_probes"])
    if old_n != 300 or len(ids) != 390:
        raise ValueError("v14.1 append requires a 300-probe base and 390 combined IDs")
    heldout = result["heldout"]
    heldout["indices"] = [*map(int, heldout["indices"]), *range(old_n, len(ids))]
    heldout["indices"] = sorted(heldout["indices"])
    heldout["probe_ids"] = [ids[index] for index in heldout["indices"]]
    heldout["sha256"] = canonical_sha256(heldout["probe_ids"])
    result["n_probes"] = len(ids)
    result["probe_ids_sha256"] = canonical_sha256(ids)
    result["probe_extension"] = {
        "append_only": True, "base_n": old_n, "extension_n": len(ids) - old_n,
        "extension_indices": list(range(old_n, len(ids))),
    }
    result.pop("split_sha256", None)
    result["split_sha256"] = canonical_sha256(result)
    return result
