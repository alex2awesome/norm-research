"""Filter the unscored code-review prompt batch after static cross-audit.

The v2 batch remains immutable.  This builder copies only jobs for the 18
corrected heldout-ready mappings and re-freezes the same salted, within-level,
different-program wrong-relation control over that corrected scope.  It never
calls a model and never reads a prompt response, candidate score, reference,
outcome, correlation, or reconstruction result.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

from methods.metric_seam.hierarchy_code_review_corrected_funnel import (
    SCHEMA as FUNNEL_SCHEMA,
    validate_corrected_funnel,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"
DEFAULT_SOURCE_MANIFEST = BASE / "code_review_reconstruction_prompt_manifest_v2.json"
DEFAULT_SOURCE_JOBS = BASE / "code_review_reconstruction_prompt_jobs_v2.jsonl.gz"
DEFAULT_CORRECTED_FUNNEL = BASE / "code_review_corrected_funnel_v1.json"
DEFAULT_FIDELITY = BASE / "code_review_construct_fidelity_v2.json"
DEFAULT_PANEL = BASE / "panel_v3.json"
DEFAULT_CROSS_AUDIT = BASE / "code_review_construct_fidelity_independent_cross_audit_v1.json"
DEFAULT_TRAIN_GATE = BASE / "code_review_train_gate_v1.json"
DEFAULT_HELDOUT = BASE / "code_review_heldout_readiness_v1.json"
DEFAULT_PREVALENCE = BASE / "code_review_witness_prevalence_v3.json"
DEFAULT_OUTPUT_MANIFEST = BASE / "code_review_reconstruction_prompt_manifest_v3.json"
DEFAULT_OUTPUT_JOBS = BASE / "code_review_reconstruction_prompt_jobs_v3.jsonl.gz"

SOURCE_SCHEMA = "metric-seam.hierarchy-reconstruction-prompt-batch.v2"
SCHEMA = "metric-seam.hierarchy-reconstruction-prompt-batch.v3"
WRONG_RELATION_SALT = "metric-seam-wrong-relation-control-v1"
CHANNELS = (
    "source_only_whole_construct",
    "source_only_subrelation",
    "implementation_disclosed",
)
PASSES = (1, 2)


class PromptSubsetError(ValueError):
    """Raised when the source batch cannot be filtered without scope drift."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromptSubsetError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PromptSubsetError(f"{path}: expected a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _validate_source_manifest(manifest: Mapping[str, Any]) -> set[str]:
    if manifest.get("schema") != SOURCE_SCHEMA or manifest.get("status") != "compiled_unscored":
        raise PromptSubsetError("expected the frozen compiled-unscored v2 prompt manifest")
    for flag in (
        "external_ground_truth_used",
        "candidate_scores_read_or_embedded",
        "prompt_outputs_used",
        "outcome_labels_used",
    ):
        if manifest.get(flag) is not False:
            raise PromptSubsetError(f"source prompt manifest violates sealed flag {flag}")
    if manifest.get("n_cells") != 21 or manifest.get("n_jobs") != 15750:
        raise PromptSubsetError("source prompt manifest does not describe the 21-cell v2 batch")
    if manifest.get("n_items_per_cell") != 125 or manifest.get("n_channels") != 3:
        raise PromptSubsetError("source prompt batch dimensions drifted")
    if manifest.get("passes") != [1, 2] or set(manifest.get("channels", {})) != set(CHANNELS):
        raise PromptSubsetError("source channels/passes drifted")
    cell_ids = manifest.get("cell_ids")
    if not isinstance(cell_ids, list) or len(cell_ids) != 21 or len(set(cell_ids)) != 21:
        raise PromptSubsetError("source manifest has invalid cell IDs")
    return {str(cell_id) for cell_id in cell_ids}


def _surviving_cells(
    manifest_cells: set[str], corrected_funnel: Mapping[str, Any]
) -> tuple[set[str], list[dict[str, Any]]]:
    if corrected_funnel.get("schema") != FUNNEL_SCHEMA or corrected_funnel.get("status") != (
        "corrected_static_gate_propagated_without_reexecution"
    ):
        raise PromptSubsetError("unexpected corrected funnel artifact")
    removed = corrected_funnel.get("removed_mappings", {}).get("heldout_confirmatory")
    if not isinstance(removed, list) or len(removed) != 3:
        raise PromptSubsetError("corrected funnel does not identify exactly three heldout removals")
    removed_ids = {str(row.get("cell_id")) for row in removed if isinstance(row, Mapping)}
    expected_removed = {
        "TB::code-review::general::R1::merged_tree::171::33b7ed9b7e4e601644ef",
        "TB::code-review::general::R2::merged_group::131::43ed2014b9a1669be3ca",
        "TB::code-review::general::R3::grandparent::3::681c2abce3bef33e3781",
    }
    if removed_ids != expected_removed or not removed_ids <= manifest_cells:
        raise PromptSubsetError("heldout removal IDs differ from the guarded static audit")
    survivors = manifest_cells - removed_ids
    if len(survivors) != 18:
        raise PromptSubsetError("corrected prompt scope is not 18 cells")
    return survivors, copy.deepcopy(removed)


def _wrong_relation_controls(
    survivors: set[str], fidelity: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows_raw = fidelity.get("rows")
    if not isinstance(rows_raw, list) or len(rows_raw) != 90:
        raise PromptSubsetError("construct fidelity does not contain 90 rows")
    rows = {str(row.get("cell_id")): row for row in rows_raw if isinstance(row, Mapping)}
    selected = []
    for cell_id in survivors:
        row = rows.get(cell_id)
        candidate = row.get("candidate") if isinstance(row, Mapping) else None
        if not isinstance(candidate, Mapping):
            raise PromptSubsetError(f"surviving cell lacks a candidate: {cell_id}")
        selected.append(
            {
                "cell_id": cell_id,
                "aspect_id": str(candidate["aspect_id"]),
                "level": str(row["level"]),
            }
        )
    controls = []
    for level in ("R1", "R2", "R3"):
        level_rows = [row for row in selected if row["level"] == level]
        ordered = sorted(
            level_rows,
            key=lambda row: _digest_json(
                {"salt": WRONG_RELATION_SALT, "cell_id": row["cell_id"]}
            ),
        )
        valid_shift = None
        for shift in range(1, len(ordered)):
            if all(
                row["aspect_id"] != ordered[(index + shift) % len(ordered)]["aspect_id"]
                for index, row in enumerate(ordered)
            ):
                valid_shift = shift
                break
        if valid_shift is None:
            raise PromptSubsetError(f"{level}: no different-program control assignment")
        for index, row in enumerate(ordered):
            control = ordered[(index + valid_shift) % len(ordered)]
            controls.append(
                {
                    "cell_id": row["cell_id"],
                    "code_vector_aspect_id": row["aspect_id"],
                    "control_prompt_cell_id": control["cell_id"],
                    "control_prompt_aspect_id": control["aspect_id"],
                    "level": level,
                    "construction": (
                        "salted within-level circular shift with different aspect_id"
                    ),
                }
            )
    controls.sort(key=lambda row: row["cell_id"])
    if {row["cell_id"] for row in controls} != survivors:
        raise PromptSubsetError("wrong-relation controls do not cover corrected cells")
    if any(row["control_prompt_cell_id"] not in survivors for row in controls):
        raise PromptSubsetError("wrong-relation control points outside corrected scope")
    return controls


def _control_reassignments(
    source_manifest: Mapping[str, Any], controls: Sequence[Mapping[str, Any]]
) -> list[dict[str, str]]:
    try:
        old_rows = source_manifest["analysis_preregistration"]["wrong_relation_control"][
            "rows"
        ]
    except (KeyError, TypeError) as exc:
        raise PromptSubsetError("source wrong-relation control table is missing") from exc
    if not isinstance(old_rows, list):
        raise PromptSubsetError("source wrong-relation control table is malformed")
    old = {
        str(row["cell_id"]): str(row["control_prompt_cell_id"])
        for row in old_rows
        if isinstance(row, Mapping)
    }
    changes = []
    for row in controls:
        cell_id = str(row["cell_id"])
        new_cell = str(row["control_prompt_cell_id"])
        if old.get(cell_id) != new_cell:
            changes.append(
                {
                    "cell_id": cell_id,
                    "old_control_prompt_cell_id": old.get(cell_id, ""),
                    "new_control_prompt_cell_id": new_cell,
                    "reason": "same salted rule re-frozen over corrected 18-cell scope",
                }
            )
    changes.sort(key=lambda row: row["cell_id"])
    if len(changes) != 4:
        raise PromptSubsetError(f"expected four corrected control assignments; found {len(changes)}")
    return changes


def _open_gzip_writer(path: Path) -> tuple[BinaryIO, gzip.GzipFile]:
    raw = path.open("xb")
    try:
        compressed = gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0)
    except Exception:
        raw.close()
        raise
    return raw, compressed


def filter_jobs(
    source_jobs: Path,
    survivors: set[str],
    *,
    output_jobs: Path | None = None,
    reject_non_survivors: bool = False,
) -> dict[str, Any]:
    """Stream the exact surviving JSONL lines and return dimension guards."""

    if output_jobs is not None and output_jobs.exists():
        raise PromptSubsetError(f"refusing to overwrite additive jobs: {output_jobs}")
    count = 0
    ids: set[str] = set()
    per_cell = Counter()
    per_channel = Counter()
    per_pass = Counter()
    per_item_by_cell: dict[str, set[str]] = defaultdict(set)
    digest = hashlib.sha256()
    raw_out: BinaryIO | None = None
    gzip_out: gzip.GzipFile | None = None
    try:
        if output_jobs is not None:
            output_jobs.parent.mkdir(parents=True, exist_ok=True)
            raw_out, gzip_out = _open_gzip_writer(output_jobs)
        with gzip.open(source_jobs, "rb") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    job = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PromptSubsetError(
                        f"source jobs line {line_number} is not JSON"
                    ) from exc
                if not isinstance(job, Mapping):
                    raise PromptSubsetError(f"source jobs line {line_number} is not an object")
                metadata = job.get("audit_metadata")
                if not isinstance(metadata, Mapping):
                    raise PromptSubsetError(f"source jobs line {line_number} lacks audit metadata")
                cell_id = str(metadata.get("cell_id"))
                if cell_id not in survivors:
                    if reject_non_survivors:
                        raise PromptSubsetError(
                            f"out-of-scope cell retained on line {line_number}: {cell_id}"
                        )
                    continue
                request_id = job.get("request_id")
                if not isinstance(request_id, str) or not request_id.startswith(f"{cell_id}::"):
                    raise PromptSubsetError(f"request/cell identity drift on line {line_number}")
                if request_id in ids:
                    raise PromptSubsetError(f"duplicate surviving request ID: {request_id}")
                channel = str(metadata.get("channel"))
                pass_id = metadata.get("pass_id")
                item_key = str(metadata.get("item_key"))
                if channel not in CHANNELS or pass_id not in PASSES or not item_key:
                    raise PromptSubsetError(f"invalid job dimensions on line {line_number}")
                ids.add(request_id)
                count += 1
                per_cell[cell_id] += 1
                per_channel[channel] += 1
                per_pass[str(pass_id)] += 1
                per_item_by_cell[cell_id].add(item_key)
                digest.update(line)
                if gzip_out is not None:
                    gzip_out.write(line)
    finally:
        if gzip_out is not None:
            gzip_out.close()
        if raw_out is not None:
            raw_out.close()
    if count != 13500 or set(per_cell) != survivors or set(per_cell.values()) != {750}:
        raise PromptSubsetError(
            f"filtered job dimensions drifted: count={count}, cells={len(per_cell)}"
        )
    if per_channel != Counter({channel: 4500 for channel in CHANNELS}):
        raise PromptSubsetError(f"filtered channel counts drifted: {per_channel}")
    if per_pass != Counter({"1": 6750, "2": 6750}):
        raise PromptSubsetError(f"filtered pass counts drifted: {per_pass}")
    if {len(items) for items in per_item_by_cell.values()} != {125}:
        raise PromptSubsetError("filtered cells do not each cover 125 items")
    return {
        "n_jobs": count,
        "n_cells": len(per_cell),
        "jobs_per_cell": 750,
        "n_items_per_cell": 125,
        "n_channels": 3,
        "n_passes": 2,
        "channel_counts": dict(sorted(per_channel.items())),
        "pass_counts": dict(sorted(per_pass.items())),
        "decompressed_jsonl_sha256": digest.hexdigest(),
    }


def _filtered_vector_clusters(
    source_manifest: Mapping[str, Any], survivors: set[str]
) -> list[dict[str, Any]]:
    try:
        clusters = source_manifest["clustered_inference"]["vector_clusters"]
    except (KeyError, TypeError) as exc:
        raise PromptSubsetError("source vector clusters are missing") from exc
    if not isinstance(clusters, list):
        raise PromptSubsetError("source vector clusters are malformed")
    filtered = []
    covered: set[str] = set()
    for cluster in clusters:
        if not isinstance(cluster, Mapping):
            raise PromptSubsetError("source vector cluster is malformed")
        kept = sorted(set(map(str, cluster.get("cell_ids", []))) & survivors)
        if not kept:
            continue
        if covered & set(kept):
            raise PromptSubsetError("filtered vector clusters overlap")
        covered.update(kept)
        filtered.append({**copy.deepcopy(dict(cluster)), "cell_ids": kept})
    if covered != survivors or len(filtered) != 10:
        raise PromptSubsetError("filtered vector clusters do not cover 18 cells in 10 clusters")
    return sorted(filtered, key=lambda row: row["vector_cluster_id"])


def build_subset_manifest(
    source_manifest: Mapping[str, Any],
    corrected_funnel: Mapping[str, Any],
    fidelity: Mapping[str, Any],
    job_summary: Mapping[str, Any],
    *,
    sources: Mapping[str, Any],
) -> dict[str, Any]:
    source_cells = _validate_source_manifest(source_manifest)
    survivors, removed = _surviving_cells(source_cells, corrected_funnel)
    controls = _wrong_relation_controls(survivors, fidelity)
    control_changes = _control_reassignments(source_manifest, controls)
    if job_summary.get("n_jobs") != 13500 or job_summary.get("n_cells") != 18:
        raise PromptSubsetError("job scan does not match corrected scope")

    manifest = copy.deepcopy(dict(source_manifest))
    manifest["schema"] = SCHEMA
    manifest["status"] = "compiled_unscored_static_cross_audit_filtered"
    manifest["sources"] = copy.deepcopy(dict(sources))
    manifest["scope_statements"]["selected_construct_fidelity_verdict_counts"] = {
        "partial": 18
    }
    manifest["scope_statements"]["whole_construct_limit"] = (
        "All 18 corrected selected code mappings have construct-fidelity "
        "verdict=partial. The source-only whole-construct channel therefore measures "
        "scope loss and cannot establish whole-construct isomorphism."
    )
    manifest["source_only_subrelation_selection"]["rows"] = [
        row
        for row in manifest["source_only_subrelation_selection"]["rows"]
        if row.get("cell_id") in survivors
    ]
    prereg = manifest["analysis_preregistration"]
    prereg["wrong_relation_control"]["rows"] = controls
    prereg["wrong_relation_control"]["scope_refreeze"] = (
        "The same v2 salted within-level different-aspect rule was re-frozen over "
        "the corrected 18-cell scope before any prompt execution."
    )
    prereg["wrong_relation_control"]["reassignments_from_v2"] = control_changes
    prereg["multiplicity"] = (
        "Report all 18 cell estimates and BH-FDR-adjusted two-sided p-values within "
        "each channel; emphasize effect sizes and support counts."
    )
    vector_clusters = _filtered_vector_clusters(source_manifest, survivors)
    manifest["clustered_inference"]["n_relation_mappings"] = 18
    manifest["clustered_inference"]["n_unique_program_vectors"] = 10
    manifest["clustered_inference"]["vector_clusters"] = vector_clusters
    manifest["passes"] = [1, 2]
    manifest["n_cells"] = 18
    manifest["n_unique_program_vectors"] = 10
    manifest["n_items_per_cell"] = 125
    manifest["n_channels"] = 3
    manifest["n_jobs"] = 13500
    manifest["expected_n_jobs"] = 13500
    manifest["cell_ids"] = sorted(survivors)
    manifest["static_cross_audit_filter"] = {
        "rule": (
            "v3 cells = v2 compiled cells intersect corrected heldout-confirmatory "
            "scope; filtered jobs are exact decompressed JSONL lines from v2"
        ),
        "n_source_cells": 21,
        "n_surviving_cells": 18,
        "n_excluded_cells": 3,
        "excluded_mappings": removed,
        "job_filter_summary": copy.deepcopy(dict(job_summary)),
        "old_batch_disposition": (
            "v2 is retained for provenance but invalid for future corrected-scope "
            "prompt execution unless these three cell IDs are excluded"
        ),
        "prompt_execution_performed": False,
    }
    return manifest


def validate_prompt_subset(
    artifact: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    corrected_funnel: Mapping[str, Any],
    fidelity: Mapping[str, Any],
    source_jobs: Path,
    output_jobs: Path,
) -> None:
    source_cells = _validate_source_manifest(source_manifest)
    survivors, _ = _surviving_cells(source_cells, corrected_funnel)
    expected_scan = filter_jobs(source_jobs, survivors)
    actual_scan = filter_jobs(output_jobs, survivors, reject_non_survivors=True)
    if expected_scan != actual_scan:
        raise PromptSubsetError("v3 jobs are not the exact surviving v2 JSONL subsequence")
    sources = artifact.get("sources")
    if not isinstance(sources, Mapping):
        raise PromptSubsetError("v3 manifest sources are missing")
    expected = build_subset_manifest(
        source_manifest,
        corrected_funnel,
        fidelity,
        expected_scan,
        sources=sources,
    )
    if artifact != expected:
        raise PromptSubsetError("v3 prompt manifest differs from guarded rebuild")
    jobs_binding = sources.get("filtered_prompt_jobs")
    if not isinstance(jobs_binding, Mapping) or jobs_binding.get("sha256") != _sha256_file(
        output_jobs
    ):
        raise PromptSubsetError("v3 compressed jobs binding drifted")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--source-jobs", type=Path, default=DEFAULT_SOURCE_JOBS)
    parser.add_argument("--corrected-funnel", type=Path, default=DEFAULT_CORRECTED_FUNNEL)
    parser.add_argument("--fidelity", type=Path, default=DEFAULT_FIDELITY)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--cross-audit", type=Path, default=DEFAULT_CROSS_AUDIT)
    parser.add_argument("--train-gate", type=Path, default=DEFAULT_TRAIN_GATE)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--prevalence", type=Path, default=DEFAULT_PREVALENCE)
    parser.add_argument("--out-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--out-jobs", type=Path, default=DEFAULT_OUTPUT_JOBS)
    args = parser.parse_args(argv)
    if args.out_manifest.exists() or args.out_jobs.exists():
        raise FileExistsError("refusing to overwrite additive v3 prompt artifacts")

    panel = _load(args.panel)
    fidelity = _load(args.fidelity)
    cross_audit = _load(args.cross_audit)
    train_gate = _load(args.train_gate)
    heldout = _load(args.heldout)
    prevalence = _load(args.prevalence)
    corrected_funnel = _load(args.corrected_funnel)
    try:
        validate_corrected_funnel(
            corrected_funnel,
            panel,
            fidelity,
            cross_audit,
            train_gate,
            heldout,
            prevalence,
        )
    except ValueError as exc:
        raise PromptSubsetError(f"corrected funnel failed validation: {exc}") from exc

    source_manifest = _load(args.source_manifest)
    source_cells = _validate_source_manifest(source_manifest)
    survivors, _ = _surviving_cells(source_cells, corrected_funnel)
    job_summary = filter_jobs(args.source_jobs, survivors, output_jobs=args.out_jobs)
    sources = {
        **copy.deepcopy(source_manifest["sources"]),
        "source_prompt_manifest_v2": {
            "path": _relative(args.source_manifest),
            "sha256": _sha256_file(args.source_manifest),
        },
        "source_prompt_jobs_v2": {
            "path": _relative(args.source_jobs),
            "sha256": _sha256_file(args.source_jobs),
        },
        "corrected_funnel": {
            "path": _relative(args.corrected_funnel),
            "sha256": _sha256_file(args.corrected_funnel),
        },
        "independent_cross_audit": {
            "path": _relative(args.cross_audit),
            "sha256": _sha256_file(args.cross_audit),
        },
        "filtered_prompt_jobs": {
            "path": _relative(args.out_jobs),
            "sha256": _sha256_file(args.out_jobs),
        },
    }
    artifact = build_subset_manifest(
        source_manifest,
        corrected_funnel,
        fidelity,
        job_summary,
        sources=sources,
    )
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "n_cells": artifact["n_cells"],
                "n_unique_program_vectors": artifact["n_unique_program_vectors"],
                "n_jobs": artifact["n_jobs"],
                "n_wrong_relation_control_reassignments": len(
                    artifact["analysis_preregistration"]["wrong_relation_control"][
                        "reassignments_from_v2"
                    ]
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
