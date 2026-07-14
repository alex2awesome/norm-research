#!/usr/bin/env python
"""Score frozen source-only name arms on public fresh partitions; gate lockbox execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from methods.codability.experiments.build_fresh_item_partitions import sha256_file, text_sha256
from methods.codability.experiments.score_adaptive_ostensive_orbits import (
    score_declared_binary,
)
from methods.codability.experiments.score_fresh_target_views import load_domain_items
from methods.codability.experiments.policy_data import (
    _resolve_declared_path,
    atomic_savez_compressed,
    atomic_write_text,
    authorize_policy_partition,
    load_partition_source_groups,
    selection_required_for_phase,
    validate_additional_artifacts,
    validate_frozen_environment,
    validate_frozen_implementation,
)
from methods.codability.experiments.validate_fresh_item_partitions import validate_packet
from methods.metric_implementer import config as cfgmod
from methods.metric_implementer.vllm_backend import make_judge_backend


MANIFEST_PATH = Path(__file__).with_name("fresh_name_execution_manifest_v1.json")


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict:
    return json.loads(Path(path).read_text())


def _by_id(rows: list[dict]) -> dict[str, dict]:
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate id")
    return result


def filter_items(items: dict, partitions: list[str]) -> dict:
    keep = [i for i, partition in enumerate(items["partitions"]) if partition in partitions]
    if not keep:
        raise ValueError(f"none of the requested partitions exist: {partitions}")
    hashes = [items["hashes"][i] for i in keep]
    return {
        "rows": [items["rows"][i] for i in keep],
        "texts": [items["texts"][i] for i in keep], "hashes": hashes,
        "partitions": [items["partitions"][i] for i in keep],
        "item_set_sha256": hashlib.sha256("\n".join(hashes).encode()).hexdigest(),
        "partition_files": [row for row in items["partition_files"]
                            if Path(row["path"]).stem in partitions],
    }


def _model_tag(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", Path(model.rstrip("/")).name)


def selected_arms(cell: dict, policy: str) -> list[dict]:
    if policy == "all":
        return cell["arms"]
    if policy == "name_only":
        return [arm for arm in cell["arms"] if arm["id"] == "name"]
    raise ValueError(f"unknown arm policy {policy!r}")


def _validate_teacher_forced_row_batch_size(
        value: int | None, *, binary_readout: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("teacher-forced row batch size must be a positive integer")
    if binary_readout != "teacher_forced_declared_labels":
        raise ValueError(
            "teacher-forced row batching requires teacher_forced_declared_labels readout"
        )
    return value


def selection_scope_for_manifest(
    manifest: dict,
    *,
    phase: str,
    selection_path: str | Path,
) -> tuple[str, str | None]:
    """Return the partition/phase scope expected by the selection's frozen schema.

    The H49 v2 artifact historically binds the eventual lockbox even when reused during its
    calibration phase.  Generic breadth selections instead bind the current phase's sole
    partition.  Keeping the distinction here prevents generic validation from changing H49.
    """
    selection = json.loads(Path(selection_path).read_text())
    if selection.get("schema") == "policy_articulation_selection/v1":
        partitions = manifest.get("phases", {}).get(phase, [])
        if len(partitions) != 1:
            raise ValueError(
                "policy-articulation selection requires one partition in its current phase"
            )
        return partitions[0], phase
    lockbox_partitions = manifest.get("phases", {}).get("lockbox", [])
    if len(lockbox_partitions) != 1:
        raise ValueError("legacy lockbox selection requires one declared lockbox partition")
    return lockbox_partitions[0], None


def load_lockbox_selection(path: str | Path | dict, *, arm_bank_sha256: str,
                           packet_manifest_sha256: str,
                           expected_partition: str | None = None,
                           expected_phase: str | None = None,
                           arm_bank: dict | None = None) -> dict[str, set[str]]:
    selection = path if isinstance(path, dict) else json.loads(Path(path).read_text())
    if selection.get("schema") == "policy_articulation_selection/v1":
        if selection.get("status") != (
                "frozen-after-search-before-validation-scoring"):
            raise ValueError("policy-articulation selection is not frozen")
        if (expected_phase is None or selection.get("selected_phase") != expected_phase
                or expected_partition is None
                or selection.get("selected_partition") != expected_partition):
            raise ValueError(
                "selection selected phase/partition mismatch: "
                f"{selection.get('selected_phase')!r}/"
                f"{selection.get('selected_partition')!r} != "
                f"{expected_phase!r}/{expected_partition!r}"
            )
        if selection.get("arm_bank_sha256") != arm_bank_sha256:
            raise ValueError("selection/arm-bank hash mismatch")
        if selection.get("packet_manifest_sha256") != packet_manifest_sha256:
            raise ValueError("selection/packet hash mismatch")
        if arm_bank is None:
            raise ValueError(
                "policy-articulation selection validation requires the exact arm bank"
            )
        bank_rows = arm_bank.get("cells", [])
        bank_cells = {cell.get("id"): cell for cell in bank_rows}
        if None in bank_cells or len(bank_cells) != len(bank_rows):
            raise ValueError("arm bank contains missing or duplicate cell ids")
        required_provenances = {
            "inert_length_control", "wrong_construct_control",
        }
        allowed = {}
        rows = selection.get("cells", [])
        if not isinstance(rows, list) or not rows:
            raise ValueError("selection contains no cells")
        for row in rows:
            cell_id = row.get("cell_id")
            if not isinstance(cell_id, str) or not cell_id or cell_id in allowed:
                raise ValueError("selection contains missing or duplicate cell ids")
            cell = bank_cells.get(cell_id)
            if cell is None:
                raise ValueError(f"selection cell {cell_id!r} is absent from arm bank")
            bank_arm_rows = cell.get("arms", [])
            bank_arms = {arm.get("id"): arm for arm in bank_arm_rows}
            if None in bank_arms or len(bank_arms) != len(bank_arm_rows):
                raise ValueError(f"arm bank cell {cell_id!r} has invalid arm ids")
            values_list = row.get("allowed_arm_ids", [])
            candidate_list = row.get("candidate_arm_ids", [])
            control_list = row.get("control_ids", [])
            provenance_list = row.get("required_control_provenances", [])
            if any(not isinstance(values, list) for values in (
                    values_list, candidate_list, control_list, provenance_list)):
                raise ValueError(f"selection {cell_id} arm panels must be lists")
            if any(not isinstance(value, str) for value in (
                    *values_list, *candidate_list, *control_list)):
                raise ValueError(f"selection {cell_id} has invalid arm ids")
            values = set(values_list)
            candidates = set(candidate_list)
            controls = set(control_list)
            if (len(values) != len(values_list)
                    or len(candidates) != len(candidate_list)
                    or len(controls) != len(control_list)):
                raise ValueError(f"selection {cell_id} contains duplicate arm ids")
            declared_provenances = set(provenance_list)
            if (not candidates or len(controls) != 2 * len(candidates)
                    or "name" not in values
                    or values != {"name", *candidates, *controls}
                    or declared_provenances != required_provenances
                    or len(declared_provenances) != len(provenance_list)):
                raise ValueError(
                    f"selection {cell_id} omits exact matched controls/provenances "
                    "or has an incomplete or hidden arm panel"
                )
            if not values <= set(bank_arms):
                raise ValueError(f"selection {cell_id} names arms absent from bank")
            if bank_arms["name"].get("provenance") != "construct_name":
                raise ValueError(f"selection {cell_id} baseline is not the construct name")
            for source_id in candidates:
                source = bank_arms[source_id]
                if (source.get("control_for") is not None
                        or source.get("provenance") in required_provenances
                        or source.get("provenance") == "construct_name"):
                    raise ValueError(
                        f"selection candidate {source_id!r} is not a content arm"
                    )
                source_controls = [
                    bank_arms[control_id]
                    for control_id in controls
                    if bank_arms[control_id].get("control_for") == source_id
                ]
                provenances = [control.get("provenance") for control in source_controls]
                if (len(source_controls) != 2
                        or set(provenances) != required_provenances
                        or len(set(provenances)) != len(provenances)):
                    raise ValueError(
                        f"selection candidate {source_id!r} lacks exactly one control "
                        "of each required provenance"
                    )
                source_forms = source.get("forms", [])
                source_form_ids = [form.get("id") for form in source_forms]
                if (not source_forms or None in source_form_ids
                        or len(source_form_ids) != len(set(source_form_ids))):
                    raise ValueError(
                        f"selection candidate {source_id!r} has an invalid form orbit"
                    )
                for control in source_controls:
                    if control.get("channel") != source.get("channel"):
                        raise ValueError(
                            f"selection control {control['id']!r} changes the channel"
                        )
                    for count_key in (
                            "semantic_content_word_count", "added_content_word_count"):
                        source_count = source.get(count_key)
                        control_count = control.get(count_key)
                        if (not isinstance(source_count, int)
                                or isinstance(source_count, bool)
                                or source_count < 0
                                or control_count != source_count):
                            raise ValueError(
                                f"selection control {control['id']!r} is not exact "
                                f"{count_key.replace('_', '-')} matched"
                            )
                    control_forms = control.get("forms", [])
                    if [form.get("id") for form in control_forms] != source_form_ids:
                        raise ValueError(
                            f"selection control {control['id']!r} changes the form orbit"
                        )
                    for source_form, control_form in zip(source_forms, control_forms):
                        source_total = source_form.get("total_word_count")
                        if (not isinstance(source_total, int)
                                or isinstance(source_total, bool)
                                or source_total < 0
                                or control_form.get("total_word_count") != source_total):
                            raise ValueError(
                                f"selection control {control['id']!r} is not exact "
                                "prompt-word-count matched"
                            )
            matched_control_ids = {
                control["id"]
                for source_id in candidates
                for control in (
                    bank_arms[control_id] for control_id in controls
                )
                if control.get("control_for") == source_id
            }
            if matched_control_ids != controls:
                raise ValueError(
                    f"selection {cell_id} includes a control for an unselected candidate"
                )
            allowed[cell_id] = values
        if set(allowed) != set(bank_cells):
            raise ValueError(
                "policy-articulation selection cell panel differs from the arm bank"
            )
        return allowed
    if selection.get("schema") == "policy_isomorphism_lockbox_selection/v2":
        if selection.get("status") != (
                "frozen-before-declared-lockbox-target-or-executor-scoring"):
            raise ValueError("policy-isomorphism lockbox selection is not frozen")
        if selection.get("arm_bank_sha256") != arm_bank_sha256:
            raise ValueError("selection/arm-bank hash mismatch")
        if selection.get("packet_manifest_sha256") != packet_manifest_sha256:
            raise ValueError("selection/packet hash mismatch")
        partition = selection.get("lockbox_partition")
        if expected_partition is None or partition != expected_partition:
            raise ValueError(
                f"selection lockbox partition mismatch: {partition!r} != "
                f"{expected_partition!r}"
            )
        allowed = {}
        required_provenances = {
            "inert_length_control", "wrong_construct_control",
        }
        bank_cells = (
            {cell["id"]: cell for cell in arm_bank.get("cells", [])}
            if arm_bank is not None else None
        )
        for row in selection.get("cells", []):
            values = set(row.get("allowed_arm_ids", []))
            controls = set(row.get("control_ids", []))
            candidates = set(row.get("candidate_arm_ids", []))
            declared_provenances = set(row.get("required_control_provenances", []))
            if (not candidates or "name" not in values or len(controls) < 2
                    or values != {"name", *candidates, *controls}
                    or declared_provenances != required_provenances):
                raise ValueError(
                    f"selection {row.get('cell_id')} omits required controls/provenances "
                    "or has an incomplete or hidden arm panel"
                )
            if bank_cells is not None:
                cell = bank_cells.get(row.get("cell_id"))
                if cell is None:
                    raise ValueError(
                        f"selection cell {row.get('cell_id')!r} is absent from arm bank"
                    )
                bank_arms = {arm["id"]: arm for arm in cell.get("arms", [])}
                if not values <= set(bank_arms):
                    raise ValueError(
                        f"selection {row.get('cell_id')} names arms absent from bank"
                    )
                for source_id in candidates:
                    source = bank_arms[source_id]
                    if (source.get("control_for") is not None
                            or source.get("provenance") in required_provenances
                            or source.get("provenance") == "construct_name"):
                        raise ValueError(
                            f"selection candidate {source_id!r} is not a content arm"
                        )
                    source_controls = [bank_arms[control_id] for control_id in controls
                                       if bank_arms[control_id].get("control_for") == source_id]
                    provenances = {control.get("provenance")
                                   for control in source_controls}
                    if provenances != required_provenances or len(source_controls) != 2:
                        raise ValueError(
                            f"selection candidate {source_id!r} lacks exactly one control "
                            "of each required provenance"
                        )
                    source_forms = [form.get("id") for form in source.get("forms", [])]
                    for control in source_controls:
                        if (control.get("semantic_content_word_count")
                                != source.get("semantic_content_word_count")):
                            raise ValueError(
                                f"selection control {control['id']!r} is not word-count matched"
                            )
                        if [form.get("id") for form in control.get("forms", [])] != source_forms:
                            raise ValueError(
                                f"selection control {control['id']!r} changes the form orbit"
                            )
                        source_totals = [form.get("total_word_count")
                                         for form in source.get("forms", [])]
                        control_totals = [form.get("total_word_count")
                                          for form in control.get("forms", [])]
                        if control_totals != source_totals:
                            raise ValueError(
                                f"selection control {control['id']!r} is not prompt-count matched"
                            )
                routes = row.get("fiber_routes", {})
                fiber_routes = set(routes.values())
                if (set(routes) != {"declarative", "procedural"}
                        or fiber_routes != candidates):
                    raise ValueError(
                        f"selection {row.get('cell_id')} fiber routes differ from candidates"
                    )
                for channel, source_id in routes.items():
                    if bank_arms[source_id].get("channel") != channel:
                        raise ValueError(
                            f"selection fiber route {source_id!r} is not {channel!r}"
                        )
            allowed[row["cell_id"]] = values
        if not allowed:
            raise ValueError("selection contains no cells")
        return allowed
    if selection.get("schema") == "policy_isomorphism_lockbox_selection/v1":
        if selection.get("status") != "frozen-before-residual-lockbox-target-or-executor-scoring":
            raise ValueError("policy-isomorphism lockbox selection is not frozen")
        if selection.get("arm_bank_sha256") != arm_bank_sha256:
            raise ValueError("selection/arm-bank hash mismatch")
        if selection.get("packet_manifest_sha256") != packet_manifest_sha256:
            raise ValueError("selection/packet hash mismatch")
        if (selection.get("lockbox_partition") != "residual_lockbox"
                or expected_partition not in (None, "residual_lockbox")):
            raise ValueError("unexpected policy-isomorphism lockbox partition")
        allowed = {}
        for row in selection.get("cells", []):
            values = set(row.get("allowed_arm_ids", []))
            if "name" not in values or not row.get("control_ids"):
                raise ValueError(f"selection {row.get('cell_id')} omits required controls")
            allowed[row["cell_id"]] = values
        if not allowed:
            raise ValueError("selection contains no cells")
        return allowed
    if selection.get("schema") != "fresh_name_arm_selection/v1":
        raise ValueError("unsupported lockbox selection schema")
    if selection.get("arm_bank_sha256") != arm_bank_sha256:
        raise ValueError("selection/arm-bank hash mismatch")
    if selection.get("packet_manifest_sha256") != packet_manifest_sha256:
        raise ValueError("selection/packet hash mismatch")
    allowed = {}
    for row in selection.get("cells", []):
        if not row.get("cuf_status"):
            raise ValueError(f"selection {row.get('cell_id')} omits CUF status")
        selected = row.get("selected_arm_id")
        controls = set(row.get("matched_control_ids", []))
        suffix = selected.removeprefix("source_") if isinstance(selected, str) else ""
        required = {f"control_wrong_{suffix}", f"control_inert_{suffix}"}
        if not selected or not required <= controls:
            raise ValueError(f"selection {row.get('cell_id')} omits matched controls")
        allowed.setdefault(row["cell_id"], {"name"}).update({selected, *controls})
    if not allowed:
        raise ValueError("selection contains no cells")
    return allowed


def score_domain(*, backend, model_job: dict, domain: str, cells: list[dict], items: dict,
                 readout_template: str, arm_bank_sha256: str, packet_manifest_sha256: str,
                 execution_manifest_sha256: str, phase: str, out_dir: str | Path,
                 repetition: int, allowed_arm_ids: dict[str, set[str]] | None = None,
                 binary_readout: str = "legacy_top20_normalized",
                 label_token_ids: dict[str, int] | None = None,
                 teacher_forced_row_batch_size: int | None = None,
                 overwrite: bool = False) -> dict:
    model = model_job["model"]
    backend_class = backend.__class__.__name__
    fake_backend = backend_class == "FakeVLLM"
    teacher_forced_row_batch_size = _validate_teacher_forced_row_batch_size(
        teacher_forced_row_batch_size, binary_readout=binary_readout)
    out = (Path(out_dir) / model_job["id"] /
           f"grid_{domain}_{phase}_{_model_tag(model)}_rep{repetition}.npz")
    if out.exists() and not overwrite:
        sidecar = out.with_suffix(".json")
        if not sidecar.is_file():
            raise ValueError(f"stale output sidecar at {sidecar}")
        try:
            saved_report = json.loads(sidecar.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"stale output sidecar at {sidecar}") from exc
        with np.load(out, allow_pickle=True) as saved:
            required_fields = {"arm_bank_sha256", "packet_manifest_sha256",
                               "probe_set_sha256", "execution_manifest_sha256",
                               "readout_template_sha256", "binary_readout",
                               "label_token_ids", "backend_class", "fake_backend"}
            if not required_fields <= set(saved.files):
                raise ValueError(f"stale output at {out}")
            if (str(saved["arm_bank_sha256"]) != arm_bank_sha256
                    or str(saved["packet_manifest_sha256"]) != packet_manifest_sha256
                    or str(saved["probe_set_sha256"]) != items["item_set_sha256"]
                    or str(saved["execution_manifest_sha256"])
                    != execution_manifest_sha256
                    or str(saved["readout_template_sha256"])
                    != text_sha256(readout_template)
                    or str(saved["binary_readout"]) != binary_readout
                    or str(saved["label_token_ids"]) != (
                        json.dumps(label_token_ids, sort_keys=True)
                        if label_token_ids else ""
                    )
                    or str(saved["backend_class"]) != backend_class
                    or bool(saved["fake_backend"]) != fake_backend):
                raise ValueError(f"stale output at {out}")
        expected_report = {
            "schema": "fresh_name_arm_scores/v1",
            "model_job_id": model_job["id"],
            "model": model,
            "role": model_job["role"],
            "phase": phase,
            "domain": domain,
            "repetition": repetition,
            "probe_set_sha256": items["item_set_sha256"],
            "arm_bank_sha256": arm_bank_sha256,
            "packet_manifest_sha256": packet_manifest_sha256,
            "execution_manifest_sha256": execution_manifest_sha256,
            "readout_template_sha256": text_sha256(readout_template),
            "binary_readout": binary_readout,
            "label_token_ids": label_token_ids,
            "backend_class": backend_class,
            "fake_backend": fake_backend,
            "score_artifact_sha256": sha256_file(out),
        }
        changed_report = sorted(
            key for key, value in expected_report.items()
            if saved_report.get(key) != value
        )
        if changed_report:
            raise ValueError(
                f"stale output sidecar at {sidecar}: {changed_report}"
            )
        return {"domain": domain, "status": "already_complete", "path": str(out)}
    tasks = {cell.get("task") for cell in cells}
    if None in tasks:
        # Backward compatibility for the frozen v1 bank. New banks must declare task explicitly.
        legacy = {"humor": "humor", "cw": "creative-writing", "pr": "press-releases"}
        if domain not in legacy or tasks != {None}:
            raise ValueError(f"{domain}: cells omit or disagree on canonical task")
        task = legacy[domain]
    elif len(tasks) == 1:
        task = next(iter(tasks))
    else:
        raise ValueError(f"{domain}: cells span multiple canonical tasks: {sorted(tasks)}")
    task_cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), task)
    scores, meta = [], []
    pending_rows: list[tuple[list[str], int]] = []

    def flush_teacher_forced_rows() -> None:
        if not pending_rows:
            return
        n_items = len(items["texts"])
        flat_prompts = [
            prompt
            for prompts, _seed in pending_rows
            for prompt in prompts
        ]
        flat_seeds = [
            row_seed
            for prompts, row_seed in pending_rows
            for _prompt in prompts
        ]
        expected_shape = (len(pending_rows) * n_items,)
        if len(flat_prompts) != expected_shape[0] or len(flat_seeds) != expected_shape[0]:
            raise ValueError("teacher-forced row batch has inconsistent prompt/seed dimensions")
        flat_scores = np.asarray(score_declared_binary(
            backend,
            flat_prompts,
            pos="YES",
            neg="NO",
            expected_token_ids=label_token_ids,
            seed=flat_seeds,
        ), float)
        if flat_scores.shape != expected_shape:
            raise ValueError(
                "teacher-forced row batch returned an invalid output shape: "
                f"observed={flat_scores.shape}, expected={expected_shape}"
            )
        scores.extend(flat_scores.reshape(len(pending_rows), n_items))
        pending_rows.clear()

    for cell in cells:
        arms = selected_arms(cell, model_job["arm_policy"])
        if allowed_arm_ids is not None:
            if cell["id"] not in allowed_arm_ids:
                continue
            arms = [arm for arm in arms if arm["id"] in allowed_arm_ids[cell["id"]]]
        for arm in arms:
            for form in arm["forms"]:
                if binary_readout == "teacher_forced_declared_labels":
                    prompts = [
                        readout_template.format(
                            rubric=form["prompt"], text=text[:task_cfg.max_text_chars])
                        for text in items["texts"]
                    ]
                    row_seed = (
                        repetition * 1_000_003
                        + len(meta) * 1009
                        + 20260713
                    )
                    if teacher_forced_row_batch_size is None:
                        scores.append(score_declared_binary(
                            backend,
                            prompts,
                            pos="YES",
                            neg="NO",
                            expected_token_ids=label_token_ids,
                            seed=row_seed,
                        ))
                    else:
                        pending_rows.append((prompts, row_seed))
                        if len(pending_rows) == teacher_forced_row_batch_size:
                            flush_teacher_forced_rows()
                elif binary_readout == "legacy_top20_normalized":
                    # Keep the historical readout available without importing the unrelated
                    # alpha-probe/reconstruction stack into the breadth teacher-forced process.
                    from methods.metric_implementer.experiments.alpha_probe import signature
                    scores.append(signature(
                        backend, form["prompt"], items["texts"],
                        task_cfg.max_text_chars, template=readout_template))
                else:
                    raise ValueError(f"unsupported binary readout {binary_readout!r}")
                meta.append({"cell_id": cell["id"], "domain": domain, "task": task,
                             "level": cell.get("level"), "bucket": cell.get("bucket"),
                             "metric_id": cell.get("metric_id"),
                             "node_id": cell.get("node_id"), "gi": cell.get("gi"),
                             "construct": cell["construct"], "arm_id": arm["id"],
                             "channel": arm["channel"], "provenance": arm["provenance"],
                             "control_for": arm["control_for"],
                             "semantic_content_word_count": arm["semantic_content_word_count"],
                             "added_content_word_count": arm.get(
                                 "added_content_word_count"),
                             "n_address_units": arm.get("n_address_units"),
                             "form": form["id"], "prompt_sha256": form["prompt_sha256"]})
    flush_teacher_forced_rows()
    matrix = np.vstack(scores)
    report = {
        "schema": "fresh_name_arm_scores/v1", "model_job_id": model_job["id"],
        "model": model, "role": model_job["role"], "phase": phase,
        "domain": domain, "task": task, "repetition": repetition,
        "n_items": len(items["texts"]),
        "n_score_rows": len(meta), "nan_rate": float(np.isnan(matrix).mean()),
        "probe_set_sha256": items["item_set_sha256"],
        "arm_bank_sha256": arm_bank_sha256,
        "packet_manifest_sha256": packet_manifest_sha256,
        "execution_manifest_sha256": execution_manifest_sha256,
        "readout_template_sha256": text_sha256(readout_template),
        "binary_readout": binary_readout,
        "label_token_ids": label_token_ids,
        "backend_class": backend_class,
        "fake_backend": fake_backend,
        "partition_files": items["partition_files"],
        "runtime_packet_validation": items.get("runtime_packet_validation"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_savez_compressed(
        out, scores=matrix,
        meta=np.asarray([json.dumps(row, sort_keys=True) for row in meta], dtype=object),
        probe_sha256=np.asarray(items["hashes"]),
        probe_partition=np.asarray(items["partitions"]),
        probe_set_sha256=items["item_set_sha256"], reader=model,
        model_job_id=model_job["id"], role=model_job["role"], phase=phase,
        repetition=repetition, arm_bank_sha256=arm_bank_sha256,
        packet_manifest_sha256=packet_manifest_sha256,
        execution_manifest_sha256=execution_manifest_sha256,
        readout_template_sha256=text_sha256(readout_template),
        binary_readout=binary_readout,
        label_token_ids=np.asarray(
            json.dumps(label_token_ids, sort_keys=True) if label_token_ids else ""),
        backend_class=backend_class,
        fake_backend=fake_backend,
    )
    report["score_artifact_sha256"] = sha256_file(out)
    sidecar = out.with_suffix(".json")
    atomic_write_text(sidecar, json.dumps(report, indent=1) + "\n")
    return {"domain": domain, "status": "complete", "path": str(out),
            "report": str(sidecar), "n_items": len(items["texts"]),
            "n_score_rows": len(meta), "nan_rate": report["nan_rate"]}


def run(*, model_job_id: str, phase: str, arm_bank_path: str, packet_root: str,
        packet_manifest: str, target_manifest_path: str, out_dir: str,
        execution_manifest_path: str | Path = MANIFEST_PATH, repetition: int = 0,
        selection_artifact: str | None = None, fake: bool = False,
        lockbox_release_artifact: str | None = None,
        domains: tuple[str, ...] | None = None,
        overwrite: bool = False) -> dict:
    execution_manifest_path = Path(execution_manifest_path).resolve()
    manifest = load_manifest(execution_manifest_path)
    if phase not in manifest.get("phases", {}):
        raise ValueError(f"phase {phase!r} is not declared in the execution manifest")
    partitions = manifest["phases"][phase]
    if not partitions:
        raise ValueError(f"phase {phase!r} contains no partitions")
    is_frozen_v2 = manifest.get("schema") == "fresh_name_execution_manifest/v2"
    if is_frozen_v2 and len(partitions) != 1:
        raise ValueError("frozen v2 scoring requires exactly one partition per phase")
    declared_domains = list(manifest.get("domains", ("humor", "cw", "pr")))
    if (not declared_domains or len(declared_domains) != len(set(declared_domains))
            or any(not isinstance(domain, str) or not domain for domain in declared_domains)):
        raise ValueError("execution manifest has invalid or duplicate domains")
    requested_domains = list(domains) if domains is not None else declared_domains
    if (not requested_domains or len(requested_domains) != len(set(requested_domains))
            or not set(requested_domains) <= set(declared_domains)):
        raise ValueError(
            f"requested domains must be a nonempty unique subset of {declared_domains}"
        )
    execution_domains = [
        domain for domain in declared_domains if domain in set(requested_domains)
    ]

    # This is the validation-fold I/O barrier.  For a frozen v2 run, authenticate the exact
    # partition (and, for the sealed lockbox phase, its required release) before resolving any
    # caller-supplied packet paths, validating the packet, or touching partition item files.
    # Open/public and legacy-v1 runs retain their existing authorization semantics.
    if selection_artifact is not None:
        selection_artifact = str(_resolve_declared_path(
            selection_artifact, manifest_path=execution_manifest_path))
    partition_authorizations = (
        [
            authorize_policy_partition(
                partition,
                operation=f"fresh name-arm {phase} scoring",
                execution_manifest_path=execution_manifest_path,
                selection_artifact_path=selection_artifact,
                lockbox_release_artifact_path=lockbox_release_artifact,
            )
            for partition in partitions
        ]
        if is_frozen_v2 else []
    )

    # Everything below the barrier may authenticate or open packet-related artifacts.
    arm_bank_path = _resolve_declared_path(
        str(arm_bank_path), manifest_path=execution_manifest_path)
    packet_manifest = _resolve_declared_path(
        str(packet_manifest), manifest_path=execution_manifest_path)
    packet_root = _resolve_declared_path(
        str(packet_root), manifest_path=execution_manifest_path)
    target_manifest_path = _resolve_declared_path(
        str(target_manifest_path), manifest_path=execution_manifest_path)
    additional_artifact_validation = validate_additional_artifacts(
        manifest, manifest_path=execution_manifest_path)
    if is_frozen_v2:
        validate_frozen_implementation(
            manifest,
            manifest_path=execution_manifest_path,
            section="scoring",
        )
        if not fake:
            validate_frozen_environment(manifest, require_accelerator=True)
    protocol_path = (
        _resolve_declared_path(
            manifest["protocol_manifest_path"],
            manifest_path=execution_manifest_path,
        )
        if manifest.get("protocol_manifest_path") else None
    )
    if (protocol_path and manifest.get("protocol_manifest_sha256")
            and sha256_file(protocol_path) != manifest["protocol_manifest_sha256"]):
        raise ValueError("item protocol manifest differs from execution manifest")
    packet_integrity = validate_packet(
        packet_manifest,
        **({"protocol_path": protocol_path} if protocol_path else {}),
        domains=set(execution_domains),
        partitions=set(partitions),
    )
    if not packet_integrity["valid"]:
        raise ValueError(packet_integrity["errors"])
    if sha256_file(arm_bank_path) != manifest["arm_bank_sha256"]:
        raise ValueError("arm bank differs from frozen execution manifest")
    if sha256_file(packet_manifest) != manifest["packet_manifest_sha256"]:
        raise ValueError("packet manifest differs from frozen execution manifest")
    if selection_required_for_phase(manifest, phase) and not selection_artifact:
        raise ValueError(f"{phase} phase requires a frozen selection artifact")
    if selection_artifact and not Path(selection_artifact).exists():
        raise ValueError("selection artifact does not exist")
    if (selection_artifact and manifest.get("selection_artifact_sha256")
            and sha256_file(selection_artifact) != manifest["selection_artifact_sha256"]):
        raise ValueError("selection artifact differs from execution manifest")
    jobs = _by_id(manifest["model_jobs"])
    model_job = jobs[model_job_id]
    required_repetitions = set(model_job.get("required_repetitions", [repetition]))
    if repetition not in required_repetitions:
        raise ValueError(
            f"repetition {repetition} is outside frozen set {sorted(required_repetitions)}"
        )
    tensor_parallel_size = int(model_job.get("tensor_parallel_size", 1))
    maximum_gpus = int(manifest.get("resource_policy", {}).get(
        "maximum_gpus_for_any_job", 4))
    if tensor_parallel_size < 1 or tensor_parallel_size > maximum_gpus:
        raise ValueError(
            f"model job tensor parallel size {tensor_parallel_size} exceeds resource policy"
        )
    model_path = Path(model_job["model"])
    if not fake and model_job.get("snapshot_revision"):
        if (not model_path.is_dir()
                or model_path.name != model_job["snapshot_revision"]):
            raise ValueError("model path does not match frozen snapshot revision")
        for filename, key in (
                ("config.json", "config_sha256"),
                ("tokenizer_config.json", "tokenizer_config_sha256")):
            if (not (model_path / filename).is_file()
                    or sha256_file(model_path / filename) != model_job.get(key)):
                raise ValueError(f"model {filename} differs from execution manifest")
    arm_bank = json.loads(Path(arm_bank_path).read_text())
    expected_target_manifest_sha256 = manifest.get("target_prompt_manifest_sha256")
    if (expected_target_manifest_sha256 is not None
            and sha256_file(target_manifest_path) != expected_target_manifest_sha256):
        raise ValueError("target prompt/readout manifest differs from execution manifest")
    target_manifest = json.loads(Path(target_manifest_path).read_text())
    readout_template = target_manifest["readout_template"]
    binary_readout = manifest.get("binary_readout", "legacy_top20_normalized")
    teacher_forced_row_batch_size = _validate_teacher_forced_row_batch_size(
        manifest.get("teacher_forced_row_batch_size"),
        binary_readout=binary_readout,
    )
    if (manifest.get("readout_template_sha256") is not None
            and text_sha256(readout_template) != manifest["readout_template_sha256"]):
        raise ValueError("readout template differs from frozen execution manifest")
    if (binary_readout == "teacher_forced_declared_labels"
            and manifest.get("label_support") != ["YES", "NO"]):
        raise ValueError("teacher-forced binary readout requires frozen YES/NO labels")
    label_token_ids = None
    if binary_readout == "teacher_forced_declared_labels":
        validation = manifest.get("teacher_forced_label_validation", {})
        label_token_ids = {
            label: int(validation[f"{label}_token_id"])
            for label in manifest["label_support"]
        }
    bank_cells_by_domain = {
        domain: [cell for cell in arm_bank["cells"] if cell["domain"] == domain]
        for domain in declared_domains
    }
    domain_tasks = {}
    legacy_tasks = {"humor": "humor", "cw": "creative-writing", "pr": "press-releases"}
    for domain, cells in bank_cells_by_domain.items():
        if not cells:
            raise ValueError(f"arm bank contains no cells for declared domain {domain!r}")
        tasks = {cell.get("task") for cell in cells}
        if tasks == {None} and domain in legacy_tasks:
            domain_tasks[domain] = legacy_tasks[domain]
        elif None not in tasks and len(tasks) == 1:
            domain_tasks[domain] = next(iter(tasks))
        else:
            raise ValueError(f"{domain}: arm bank has ambiguous canonical tasks: {tasks}")
    frozen_domain_tasks = manifest.get("domain_tasks")
    if frozen_domain_tasks is not None and frozen_domain_tasks != domain_tasks:
        raise ValueError("arm-bank domain/task registry differs from execution manifest")

    # Deep selection validation is intentionally before backend construction: a malformed
    # artifact must fail without loading an 8B/70B model onto the server.
    selection_scope = (
        selection_scope_for_manifest(
            manifest, phase=phase, selection_path=selection_artifact)
        if selection_artifact else None
    )
    allowed_arm_ids = (
        load_lockbox_selection(
            selection_artifact,
            arm_bank_sha256=sha256_file(arm_bank_path),
            packet_manifest_sha256=sha256_file(packet_manifest),
            expected_partition=selection_scope[0],
            expected_phase=selection_scope[1],
            arm_bank=arm_bank,
        )
        if selection_artifact else None
    )

    first_task = domain_tasks[execution_domains[0]]
    config = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), first_task)
    expected_max_chars = manifest.get("item_text_max_chars")
    expected_by_task = manifest.get("item_text_max_chars_by_task")
    for task in sorted(set(domain_tasks.values())):
        task_max_chars = int(
            cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), task).max_text_chars)
        if expected_by_task is not None:
            if int(expected_by_task.get(task, -1)) != task_max_chars:
                raise ValueError(
                    f"{task}: text truncation differs from frozen execution manifest")
        elif expected_max_chars is not None and int(expected_max_chars) != task_max_chars:
            raise ValueError(
                f"{task}: text truncation differs from frozen execution manifest")
    if fake:
        config.vllm_fake = True
    config.vllm_tp_size = tensor_parallel_size
    if model_job.get("dtype") is not None:
        config.vllm_dtype = model_job["dtype"]
    if model_job.get("gpu_memory_utilization") is not None:
        config.vllm_gpu_mem_util = float(model_job["gpu_memory_utilization"])
    if model_job.get("max_model_len") is not None:
        config.vllm_max_model_len = int(model_job["max_model_len"])
    backend = make_judge_backend(model_job["model"], config, temperature=None)
    results = []
    for domain in execution_domains:
        cells = bank_cells_by_domain[domain]
        # Load only the explicitly authorized files.  In particular, calibration execution never
        # opens a lockbox file and then filters it out after the fact.
        items = load_domain_items(packet_root, domain, partitions=partitions)
        if is_frozen_v2:
            packet_binding = load_partition_source_groups(
                packet_root,
                packet_manifest,
                domain=domain,
                partition=partitions[0],
                item_hashes=items["hashes"],
            )
            items["runtime_packet_validation"] = packet_binding["validation"]
        results.append(score_domain(
            backend=backend, model_job=model_job, domain=domain, cells=cells, items=items,
            readout_template=readout_template, arm_bank_sha256=sha256_file(arm_bank_path),
            packet_manifest_sha256=sha256_file(packet_manifest),
            execution_manifest_sha256=sha256_file(execution_manifest_path),
            phase=phase, out_dir=out_dir, repetition=repetition,
            allowed_arm_ids=allowed_arm_ids, binary_readout=binary_readout,
            label_token_ids=label_token_ids,
            teacher_forced_row_batch_size=teacher_forced_row_batch_size,
            overwrite=overwrite))
    return {"schema": "fresh_name_arm_execution/v1", "model_job_id": model_job_id,
            "phase": phase, "repetition": repetition, "results": results,
            "requested_domains": execution_domains,
            "partition_authorizations": partition_authorizations,
            "additional_artifact_validation": additional_artifact_validation,
            "selection_artifact_sha256": (sha256_file(selection_artifact)
                                           if selection_artifact else None)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-job", required=True)
    parser.add_argument("--phase", required=True,
                        help="phase id declared by --execution-manifest")
    parser.add_argument("--arm-bank", required=True)
    parser.add_argument("--packet-root", required=True)
    parser.add_argument("--packet-manifest", required=True)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--execution-manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--selection-artifact", default=None)
    parser.add_argument("--lockbox-release-artifact", default=None)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument(
        "--domains", default=None,
        help="optional comma-separated subset of manifest domains for resumable parallel scoring",
    )
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run(
        model_job_id=args.model_job, phase=args.phase, arm_bank_path=args.arm_bank,
        packet_root=args.packet_root, packet_manifest=args.packet_manifest,
        target_manifest_path=args.target_manifest, out_dir=args.out_dir,
        execution_manifest_path=args.execution_manifest,
        repetition=args.repetition, selection_artifact=args.selection_artifact,
        lockbox_release_artifact=args.lockbox_release_artifact,
        domains=(tuple(
            domain.strip() for domain in args.domains.split(",") if domain.strip()
        ) if args.domains else None),
        fake=args.fake, overwrite=args.overwrite)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
