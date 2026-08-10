#!/usr/bin/env python3
"""Bind the frozen Science addressed arm to the additive hierarchy subset.

This is a CPU-only, pre-prompt receipt.  It does not compile a new prompt,
execute a prepared prompt, load a model response, or use an outcome/reference
target.  It establishes four narrower facts:

* every one of the 300 additive hierarchy items maps uniquely to the frozen
  2,400-paper source corpus;
* the selected items map to 235 existing v8 prepared requests and 65 existing
  missing-body structural abstentions (124/26 train and 111/39 heldout);
* the frozen v9 addressed-code result agrees item-by-item on status and output
  counts, and exactly in aggregate, with the hierarchy continuous-text code
  execution; and
* all relevant item, request, result, implementation, and replay artifacts are
  content-bound in one additive receipt.

The v8 prompt renders addressed JSONL rather than the hierarchy ``ctext`` bytes.
The receipt therefore licenses a same-evidence addressed transport control, not
an exact-ctext isomorphism claim.  With zero prompt responses, it also measures
neither prompt articulability nor prompt/code reconstruction.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import addressed_code_comparator_v9 as v9
from . import addressed_pipeline_v8 as v8
from . import audit_addressed_v9_replay as v9_replay
from . import core_relation_strict as strict
from methods.metric_seam import hierarchy_science_fullarticle_runner as hierarchy


ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "metric-seam.science-fullarticle-addressed-subset-binding.v1"

DEFAULT_SOURCE = ROOT / "datasets/peer-review/peer_review_cv_evidence.jsonl"
DEFAULT_V8_BUNDLE = (
    ROOT / "outputs/metric_seam_pilot/science_articulability_v8_hardened_prepared"
)
DEFAULT_V9_BUNDLE = (
    ROOT
    / "outputs/metric_seam_pilot/science_verifiability_v9_relation_strict_addressed"
)
DEFAULT_V9_REPLAY = (
    ROOT
    / "outputs/metric_seam_pilot/"
    "science_verifiability_v9_relation_strict_addressed_replay_v1.json"
)
DEFAULT_HIERARCHY_BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"
DEFAULT_ITEMS = DEFAULT_HIERARCHY_BASE / "items_v3/peer-review-fullarticle"
DEFAULT_OUT = (
    DEFAULT_HIERARCHY_BASE
    / "peer_review_science_addressed_subset_binding_v1.json"
)

SPLITS = {
    "compiler_train": {
        "items_name": "compiler_train.json",
        "execution_name": "peer_review_science_fullarticle_compiler_train_v1.json",
        "phase": "compiler_train",
        "expected_requests": 124,
        "expected_structural_abstentions": 26,
    },
    "sealed_heldout": {
        "items_name": "sealed_heldout.json",
        "execution_name": (
            "peer_review_science_fullarticle_heldout_pre_reference_v1.json"
        ),
        "phase": "heldout_pre_reference",
        "expected_requests": 111,
        "expected_structural_abstentions": 39,
    },
}

AGREEMENT_FIELDS = {
    "verifier_status": "status",
    "claim_count": "claim_count",
    "certificate_count": "certificate_count",
    "evidence_link_count": "evidence_link_count",
    "decision_counts": "decision_counts",
}


class ScienceSubsetBindingError(ValueError):
    """Raised when a frozen Science input or crosswalk has drifted."""


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_file(path: Path) -> dict[str, Any]:
    return {
        "path": _display(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ScienceSubsetBindingError(
                    f"{_display(path)} line {line_number} is not an object"
                )
            yield row


def _assert_manifest_file(
    bundle: Path,
    file_record: Mapping[str, Any],
    *,
    expected_count: int,
) -> Path:
    path = bundle / str(file_record.get("path"))
    if file_record.get("count") != expected_count:
        raise ScienceSubsetBindingError(f"recorded count drifted for {_display(path)}")
    if file_record.get("sha256") != _sha256(path):
        raise ScienceSubsetBindingError(f"recorded hash drifted for {_display(path)}")
    return path


def _load_selected_v8_rows(
    *,
    bundle: Path,
    manifest: Mapping[str, Any],
    selected_indices: set[int],
    source_rows: list[dict[str, str]],
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[str, Path]]:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ScienceSubsetBindingError("v8 manifest has no file ledger")
    requests_path = _assert_manifest_file(
        bundle,
        files.get("requests", {}),
        expected_count=1957,
    )
    abstentions_path = _assert_manifest_file(
        bundle,
        files.get("structural_abstentions", {}),
        expected_count=443,
    )
    crosswalk_path = _assert_manifest_file(
        bundle,
        files.get("source_crosswalk", {}),
        expected_count=2400,
    )

    request_indices: set[int] = set()
    selected_requests: dict[int, dict[str, Any]] = {}
    request_count = 0
    for row in _read_jsonl(requests_path):
        request_count += 1
        source_index = row.get("source_index")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index in request_indices
        ):
            raise ScienceSubsetBindingError("v8 requests contain an invalid source index")
        request_indices.add(source_index)
        if source_index in selected_indices:
            selected_requests[source_index] = row
    if request_count != 1957:
        raise ScienceSubsetBindingError("v8 request file line count drifted")

    abstention_indices: set[int] = set()
    selected_abstentions: dict[int, dict[str, Any]] = {}
    abstention_count = 0
    for row in _read_jsonl(abstentions_path):
        abstention_count += 1
        source_index = row.get("source_index")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index in abstention_indices
        ):
            raise ScienceSubsetBindingError(
                "v8 abstentions contain an invalid source index"
            )
        abstention_indices.add(source_index)
        if source_index in selected_indices:
            selected_abstentions[source_index] = row
    if abstention_count != 443:
        raise ScienceSubsetBindingError("v8 abstention file line count drifted")
    if request_indices & abstention_indices:
        raise ScienceSubsetBindingError("v8 request and abstention strata overlap")
    if request_indices | abstention_indices != set(range(len(source_rows))):
        raise ScienceSubsetBindingError("v8 strata no longer cover the corpus exactly")

    for source_index, row in selected_requests.items():
        paper = source_rows[source_index]
        if row.get("schema_version") != v8.REQUEST_SCHEMA:
            raise ScienceSubsetBindingError("selected v8 request schema drifted")
        if row.get("paper_input") != paper:
            raise ScienceSubsetBindingError("selected v8 request source projection drifted")
        if row.get("paper_input_sha256") != v8.hash_value(paper):
            raise ScienceSubsetBindingError("selected v8 paper-input hash drifted")
        material = {key: row.get(key) for key in v8._REQUEST_MATERIAL_KEYS}
        request_sha = v8.hash_value(material)
        expected_id = f"science_v8_addressed_{source_index:04d}_{request_sha[:16]}"
        if row.get("request_sha256") != request_sha or row.get("request_id") != expected_id:
            raise ScienceSubsetBindingError("selected v8 request material binding drifted")

    abstention_schema_keys = {
        "schema_version",
        "source_index",
        "paper_id",
        "paper_input_sha256",
        "source_map_sha256",
        "source_crosswalk_sha256",
        "v7_request_id",
        "v7_request_sha256",
        "prompt_eligible",
        "status",
        "reason",
        "api_call_required",
        "abstention_sha256",
    }
    for source_index, row in selected_abstentions.items():
        paper = source_rows[source_index]
        if set(row) != abstention_schema_keys:
            raise ScienceSubsetBindingError("selected v8 abstention schema drifted")
        if (
            row.get("schema_version") != v8.STRUCTURAL_ABSTENTION_SCHEMA
            or row.get("paper_id") != paper["paper_id"]
            or row.get("paper_input_sha256") != v8.hash_value(paper)
            or row.get("prompt_eligible") is not False
            or row.get("status") != "structural_abstention"
            or row.get("reason") != "missing_fullpaper_body"
            or row.get("api_call_required") is not False
        ):
            raise ScienceSubsetBindingError("selected v8 abstention contract drifted")
        material = {
            key: value
            for key, value in row.items()
            if key not in {"schema_version", "abstention_sha256"}
        }
        if row.get("abstention_sha256") != v8.hash_value(material):
            raise ScienceSubsetBindingError("selected v8 abstention hash drifted")

    return selected_requests, selected_abstentions, {
        "requests": requests_path,
        "source_crosswalk": crosswalk_path,
        "structural_abstentions": abstentions_path,
    }


def _load_selected_v9_rows(
    *,
    path: Path,
    selected_indices: set[int],
    source_rows: list[dict[str, str]],
) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    seen: set[int] = set()
    for row in _read_jsonl(path):
        source_index = row.get("source_index")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index in seen
        ):
            raise ScienceSubsetBindingError("v9 result has an invalid source index")
        seen.add(source_index)
        if (
            row.get("schema_version")
            != "science-verifiability-addressed-result-v9"
            or row.get("paper_id") != source_rows[source_index]["paper_id"]
            or not isinstance(row.get("result"), Mapping)
        ):
            raise ScienceSubsetBindingError("v9 result source binding drifted")
        if source_index in selected_indices:
            selected[source_index] = row
    if seen != set(range(len(source_rows))):
        raise ScienceSubsetBindingError("v9 results do not cover the corpus exactly")
    if set(selected) != selected_indices:
        raise ScienceSubsetBindingError("v9 selected subset coverage drifted")
    return selected


def _v9_execution_summary(
    items: list[dict[str, str]],
    item_to_index: Mapping[str, int],
    v9_rows: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    for item in items:
        result = v9_rows[item_to_index[item["item_key"]]]["result"]
        status = result.get("status")
        if status == "abstain":
            measurement_state = "abstained"
        elif status in hierarchy._MEASURED_VERIFIER_STATUSES:
            measurement_state = "measured"
        else:
            raise ScienceSubsetBindingError(
                f"selected v9 result has unsupported status {status!r}"
            )
        rows.append(
            {
                "item_key": item["item_key"],
                "measurement_state": measurement_state,
                "verifier_status": status,
                "certificate_count": result.get("certificate_count"),
            }
        )
    return hierarchy._summarize_execution(rows, n_relations=6)


def audit(
    *,
    source_path: Path = DEFAULT_SOURCE,
    v8_bundle: Path = DEFAULT_V8_BUNDLE,
    v9_bundle: Path = DEFAULT_V9_BUNDLE,
    v9_replay_path: Path = DEFAULT_V9_REPLAY,
    hierarchy_base: Path = DEFAULT_HIERARCHY_BASE,
    items_dir: Path = DEFAULT_ITEMS,
) -> dict[str, Any]:
    """Build and validate the deterministic hierarchy-subset receipt."""

    item_manifest_path = items_dir / "manifest.json"
    item_manifest = _load_json(item_manifest_path)
    if (
        item_manifest.get("schema") != hierarchy.ITEM_SCHEMA
        or item_manifest.get("status")
        != "additive_noncanonical_fullarticle_section_split_frozen"
        or item_manifest.get("selection", {}).get("selected_n") != 300
        or item_manifest.get("selection", {}).get("compiler_train_body_nonempty_n")
        != 124
        or item_manifest.get("selection", {}).get("sealed_heldout_body_nonempty_n")
        != 111
    ):
        raise ScienceSubsetBindingError("hierarchy full-article item manifest drifted")

    source_rows = hierarchy.load_outcome_blind_source(source_path)
    if len(source_rows) != 2400:
        raise ScienceSubsetBindingError("science source corpus no longer has 2,400 rows")
    source_indices_by_ctext: dict[str, list[int]] = defaultdict(list)
    for source_index, paper in enumerate(source_rows):
        source_indices_by_ctext[
            hierarchy.article_ctext(paper["abstract"], paper["body"])
        ].append(source_index)

    split_state: dict[str, dict[str, Any]] = {}
    selected_indices: set[int] = set()
    for split, contract in SPLITS.items():
        items_path = items_dir / contract["items_name"]
        items = hierarchy.validate_items(
            _load_json(items_path),
            expected_prefix=(
                "science_train_" if split == "compiler_train" else "science_heldout_"
            ),
        )
        if len(items) != 150:
            raise ScienceSubsetBindingError(f"{split} no longer has 150 items")
        recorded_fingerprint = item_manifest.get("projection_fingerprints", {}).get(split)
        recomputed_fingerprint = hierarchy._fingerprint(items)
        if recorded_fingerprint != recomputed_fingerprint:
            raise ScienceSubsetBindingError(f"{split} projection fingerprint drifted")

        item_to_index: dict[str, int] = {}
        for item in items:
            matches = source_indices_by_ctext.get(item["ctext"], [])
            if len(matches) != 1:
                raise ScienceSubsetBindingError(
                    f"{item['item_key']} has {len(matches)} exact source projections"
                )
            source_index = matches[0]
            if source_index in selected_indices:
                raise ScienceSubsetBindingError("hierarchy splits reuse a source row")
            selected_indices.add(source_index)
            item_to_index[item["item_key"]] = source_index

        execution_path = hierarchy_base / contract["execution_name"]
        execution = _load_json(execution_path)
        if (
            execution.get("schema") != hierarchy.EXECUTION_SCHEMA
            or execution.get("status")
            != "execution_complete_pre_prompt_pre_reference"
            or execution.get("phase") != contract["phase"]
        ):
            raise ScienceSubsetBindingError(f"{split} hierarchy execution drifted")
        execution_rows = hierarchy._validate_execution_rows(execution.get("rows"))
        if [row["item_key"] for row in execution_rows] != [
            item["item_key"] for item in items
        ]:
            raise ScienceSubsetBindingError(f"{split} execution item order drifted")
        if execution.get("summary") != hierarchy._summarize_execution(
            execution_rows,
            n_relations=6,
        ):
            raise ScienceSubsetBindingError(f"{split} hierarchy summary drifted")
        split_state[split] = {
            "items": items,
            "items_path": items_path,
            "item_to_index": item_to_index,
            "execution": execution,
            "execution_path": execution_path,
            "execution_by_item": {
                row["item_key"]: row for row in execution_rows
            },
            "recorded_fingerprint": recorded_fingerprint,
            "recomputed_fingerprint": recomputed_fingerprint,
        }
    if len(selected_indices) != 300:
        raise ScienceSubsetBindingError("hierarchy selected-index union drifted")

    v8_manifest_path = v8_bundle / "manifest.json"
    v8_manifest = _load_json(v8_manifest_path)
    if (
        v8_manifest.get("schema_version") != v8.MANIFEST_SCHEMA
        or v8_manifest.get("status") != "prepared_not_run_no_api_calls"
        or v8_manifest.get("execution_policy", {}).get("api_calls_made_by_prepare")
        != 0
        or v8_manifest.get("execution_policy", {}).get("gpu_used") is not False
        or v8_manifest.get("isomorphism_scope", {}).get("same_evidence_content")
        is not True
        or v8_manifest.get("isomorphism_scope", {}).get("same_input_representation")
        is not False
        or v8_manifest.get("isomorphism_scope", {}).get("full_isomorphism_licensed")
        is not False
    ):
        raise ScienceSubsetBindingError("v8 manifest contract drifted")
    if (
        v8_manifest.get("input", {}).get("record_count") != len(source_rows)
        or v8_manifest.get("input", {}).get("source_file_sha256")
        != _sha256(source_path)
    ):
        raise ScienceSubsetBindingError("v8 source-file binding drifted")

    selected_requests, selected_abstentions, v8_files = _load_selected_v8_rows(
        bundle=v8_bundle,
        manifest=v8_manifest,
        selected_indices=selected_indices,
        source_rows=source_rows,
    )
    if set(selected_requests) | set(selected_abstentions) != selected_indices:
        raise ScienceSubsetBindingError("selected v8 strata do not cover the subset")

    v9_manifest_path = v9_bundle / "manifest.json"
    v9_results_path = v9_bundle / "code_results.jsonl"
    v9_manifest = _load_json(v9_manifest_path)
    if (
        v9_manifest.get("schema_version")
        != "science-verifiability-addressed-v9-relation-strict"
        or v9_manifest.get("status") != "completed_cpu_no_api_no_gpu"
        or v9_manifest.get("summary", {}).get("records") != 2400
        or v9_manifest.get("summary", {}).get("certificates") != 100
    ):
        raise ScienceSubsetBindingError("v9 manifest contract drifted")
    selected_v9 = _load_selected_v9_rows(
        path=v9_results_path,
        selected_indices=selected_indices,
        source_rows=source_rows,
    )

    replay = _load_json(v9_replay_path)
    if (
        replay.get("status") != "byte_exact_cpu_replay_complete"
        or replay.get("replay", {}).get("byte_exact_all_outputs") is not True
        or replay.get("prompt_plane", {}).get("prompt_responses_in_current_v8_bundle")
        != 0
    ):
        raise ScienceSubsetBindingError("v9 replay freeze is incomplete")
    replay_outputs = replay.get("archived_outputs", {})
    if (
        replay_outputs.get("manifest.json", {}).get("sha256")
        != _sha256(v9_manifest_path)
        or replay_outputs.get("code_results.jsonl", {}).get("sha256")
        != _sha256(v9_results_path)
    ):
        raise ScienceSubsetBindingError("v9 replay freeze no longer binds current outputs")

    crosswalk: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    total_transport = Counter()
    total_reason_agreement = Counter()
    for split, contract in SPLITS.items():
        state = split_state[split]
        transport = Counter()
        reason_agreement = Counter()
        agreement_count = 0
        for item in state["items"]:
            item_key = item["item_key"]
            source_index = state["item_to_index"][item_key]
            source = source_rows[source_index]
            abstract, body = hierarchy.parse_article_ctext(item["ctext"])
            if abstract != source["abstract"].strip() or body != source["body"].strip():
                raise ScienceSubsetBindingError(
                    f"{item_key} no longer has the same source evidence content"
                )

            if source_index in selected_requests:
                request = selected_requests[source_index]
                transport_status = "compiled_unscored_request"
                request_id = request["request_id"]
                request_sha = request["request_sha256"]
                abstention_sha = None
            else:
                abstention = selected_abstentions[source_index]
                transport_status = "structural_abstention_no_remote_call"
                request_id = None
                request_sha = None
                abstention_sha = abstention["abstention_sha256"]
            transport[transport_status] += 1
            total_transport[transport_status] += 1

            v9_result = selected_v9[source_index]["result"]
            hierarchy_result = state["execution_by_item"][item_key]
            field_agreement = {
                hierarchy_field: hierarchy_result.get(hierarchy_field)
                == v9_result.get(v9_field)
                for hierarchy_field, v9_field in AGREEMENT_FIELDS.items()
            }
            if not all(field_agreement.values()):
                raise ScienceSubsetBindingError(
                    f"{item_key} v9/hierarchy output fields diverged"
                )
            agreement_count += 1
            same_reason = hierarchy_result.get("reason") == v9_result.get("reason")
            reason_agreement["exact" if same_reason else "representation_specific"] += 1
            total_reason_agreement[
                "exact" if same_reason else "representation_specific"
            ] += 1
            if not same_reason and (
                hierarchy_result.get("reason"),
                v9_result.get("reason"),
            ) != ("missing_fullpaper_body", "missing_fullpaper_body_addresses"):
                raise ScienceSubsetBindingError(
                    f"{item_key} has an unexpected reason-label divergence"
                )

            crosswalk.append(
                {
                    "split": split,
                    "item_key": item_key,
                    "item_ctext_sha256": hashlib.sha256(
                        item["ctext"].encode("utf-8")
                    ).hexdigest(),
                    "source_index": source_index,
                    "source_projection_sha256": v8.hash_value(source),
                    "v8_prompt_transport": {
                        "status": transport_status,
                        "request_id": request_id,
                        "request_sha256": request_sha,
                        "structural_abstention_sha256": abstention_sha,
                    },
                    "v9_addressed_code": {
                        "status": v9_result.get("status"),
                        "reason": v9_result.get("reason"),
                        "claim_count": v9_result.get("claim_count"),
                        "certificate_count": v9_result.get("certificate_count"),
                        "evidence_link_count": v9_result.get("evidence_link_count"),
                        "decision_counts": v9_result.get("decision_counts"),
                        "result_sha256": _fingerprint(v9_result),
                    },
                    "hierarchy_continuous_code": {
                        "status": hierarchy_result.get("verifier_status"),
                        "reason": hierarchy_result.get("reason"),
                        "claim_count": hierarchy_result.get("claim_count"),
                        "certificate_count": hierarchy_result.get("certificate_count"),
                        "evidence_link_count": hierarchy_result.get(
                            "evidence_link_count"
                        ),
                        "decision_counts": hierarchy_result.get("decision_counts"),
                    },
                    "agreement": {
                        "status_and_output_counts_exact": True,
                        "reason_label_exact": same_reason,
                    },
                }
            )

        expected_transport = {
            "compiled_unscored_request": contract["expected_requests"],
            "structural_abstention_no_remote_call": contract[
                "expected_structural_abstentions"
            ],
        }
        if dict(transport) != expected_transport:
            raise ScienceSubsetBindingError(f"{split} v8 subset counts drifted")
        v9_summary = _v9_execution_summary(
            state["items"],
            state["item_to_index"],
            selected_v9,
        )
        hierarchy_summary = state["execution"]["summary"]
        if v9_summary != hierarchy_summary:
            raise ScienceSubsetBindingError(f"{split} v9 aggregate diverged")
        split_summaries[split] = {
            "items": len(state["items"]),
            "prompt_transport": expected_transport,
            "v9_hierarchy_item_field_agreement": {
                "agree": agreement_count,
                "total": len(state["items"]),
                "fields": list(AGREEMENT_FIELDS),
            },
            "reason_label_agreement": dict(reason_agreement),
            "v9_recomputed_summary": v9_summary,
            "hierarchy_archived_summary": hierarchy_summary,
            "aggregate_exact_dict_agreement": True,
        }

    expected_total_transport = {
        "compiled_unscored_request": 235,
        "structural_abstention_no_remote_call": 65,
    }
    if dict(total_transport) != expected_total_transport:
        raise ScienceSubsetBindingError("combined v8 subset counts drifted")
    if dict(total_reason_agreement) != {
        "exact": 235,
        "representation_specific": 65,
    }:
        raise ScienceSubsetBindingError("v9/hierarchy reason-label accounting drifted")

    crosswalk.sort(
        key=lambda row: (
            0 if row["split"] == "compiler_train" else 1,
            row["item_key"],
        )
    )
    if len(crosswalk) != 300:
        raise ScienceSubsetBindingError("subset crosswalk no longer has 300 rows")

    prompt_spec_path = ROOT / v8_manifest["prompt_spec"]["path"]
    model_manifest_path = ROOT / v8_manifest["model_manifest"]["path"]
    continuous_path = ROOT / "outputs/metric_seam_pilot/science_claims_v2_relation_strict_v23/results.json"
    receipt = {
        "schema_version": SCHEMA,
        "status": "cpu_only_subset_binding_complete_pre_prompt",
        "objective": "unsupervised_relation_local_prompt_code_reconstruction_scaffold",
        "method_origin": "manually_constructed_retrospective_pipeline_seed",
        "task": "peer-review",
        "representation_contract": {
            "same_evidence_content": True,
            "same_source_address_inventory_for_v8_prompt_and_v9_code": True,
            "same_input_representation": False,
            "exact_hierarchy_ctext_rendered_to_prompt": False,
            "full_isomorphism_licensed": False,
            "transport_class": "same_evidence_source_addressed_not_exact_ctext",
            "reason": v8_manifest["isomorphism_scope"]["reason"],
        },
        "temporal_disposition": {
            "current_subset_role": "instrument_development_exploratory_unscored",
            "fresh_split_required_for_confirmatory_prompt_code_claim": True,
            "current_heldout_may_support_exploratory_reconstruction_only": True,
            "reason": (
                "The addressed prompt contract and manually fixed code decomposition were "
                "developed before this binding receipt; the current split is not a fresh "
                "temporally blind confirmation split."
            ),
        },
        "prompt_plane": {
            "selected_items": 300,
            "distinct_prepared_unscored_request_records": 235,
            "structural_abstentions_without_remote_call": 65,
            "prompt_responses": 0,
            "prompt_articulability_measured": False,
            "prompt_code_reconstruction_measured": False,
            "planned_stateless_passes": 2,
            "planned_two_pass_prompt_jobs_if_executed": 470,
            "two_pass_jobs_materialized_as_separate_requests": False,
            "six_relation_mappings_share_one_result_vector": True,
        },
        "split_summaries": split_summaries,
        "combined_summary": {
            "items": 300,
            "prompt_transport": expected_total_transport,
            "v9_hierarchy_item_field_agreement": {
                "agree": 300,
                "total": 300,
                "fields": list(AGREEMENT_FIELDS),
            },
            "reason_label_agreement": dict(total_reason_agreement),
            "v9_hierarchy_aggregate_exact_for_both_splits": True,
        },
        "bound_inputs": {
            "receipt_builder": _bound_file(Path(__file__)),
            "outcome_masked_source": _bound_file(source_path),
            "v8": {
                "manifest": _bound_file(v8_manifest_path),
                "requests": _bound_file(v8_files["requests"]),
                "source_crosswalk": _bound_file(v8_files["source_crosswalk"]),
                "structural_abstentions": _bound_file(
                    v8_files["structural_abstentions"]
                ),
                "prompt_spec": _bound_file(prompt_spec_path),
                "model_manifest": _bound_file(model_manifest_path),
                "implementation": _bound_file(Path(v8.__file__)),
            },
            "v9": {
                "manifest": _bound_file(v9_manifest_path),
                "code_results": _bound_file(v9_results_path),
                "continuous_code_input": _bound_file(continuous_path),
                "implementation": _bound_file(Path(v9.__file__)),
                "strict_relation_implementation": _bound_file(Path(strict.__file__)),
                "full_corpus_replay_receipt": _bound_file(v9_replay_path),
                "full_corpus_replay_audit": _bound_file(Path(v9_replay.__file__)),
            },
            "hierarchy": {
                "item_manifest": _bound_file(item_manifest_path),
                "runner_implementation": _bound_file(Path(hierarchy.__file__)),
                "splits": {
                    split: {
                        "items": _bound_file(state["items_path"]),
                        "execution": _bound_file(state["execution_path"]),
                        "recorded_projection_fingerprint": state[
                            "recorded_fingerprint"
                        ],
                        "recomputed_projection_fingerprint": state[
                            "recomputed_fingerprint"
                        ],
                    }
                    for split, state in split_state.items()
                },
            },
        },
        "execution_policy": {
            "prompt_jobs_executed": False,
            "prompt_responses_loaded": False,
            "reference_values_loaded": False,
            "outcome_values_loaded": False,
            "external_supervision_used": False,
            "models_or_apis_called": False,
            "accelerators_used": False,
            "whole_review_score_emitted": False,
        },
        "claim_boundary": (
            "This receipt binds a same-evidence addressed prompt scaffold to a "
            "relation-local code result and proves addressed/continuous code agreement "
            "on the selected items. It does not measure prompt articulability, exact-ctext "
            "isomorphism, prompt/code reconstruction, whole-criterion codability, or "
            "external scientific truth."
        ),
        "crosswalk": crosswalk,
    }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--v8-bundle", type=Path, default=DEFAULT_V8_BUNDLE)
    parser.add_argument("--v9-bundle", type=Path, default=DEFAULT_V9_BUNDLE)
    parser.add_argument("--v9-replay", type=Path, default=DEFAULT_V9_REPLAY)
    parser.add_argument("--hierarchy-base", type=Path, default=DEFAULT_HIERARCHY_BASE)
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    payload = audit(
        source_path=args.source.resolve(),
        v8_bundle=args.v8_bundle.resolve(),
        v9_bundle=args.v9_bundle.resolve(),
        v9_replay_path=args.v9_replay.resolve(),
        hierarchy_base=args.hierarchy_base.resolve(),
        items_dir=args.items.resolve(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selected_items": payload["prompt_plane"]["selected_items"],
                "prepared_unscored_requests": payload["prompt_plane"][
                    "distinct_prepared_unscored_request_records"
                ],
                "structural_abstentions": payload["prompt_plane"][
                    "structural_abstentions_without_remote_call"
                ],
                "planned_two_pass_jobs": payload["prompt_plane"][
                    "planned_two_pass_prompt_jobs_if_executed"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
