#!/usr/bin/env python3
"""Join active a104 diffs to frozen repository-execution telemetry.

This is a typed capability-augmentation receipt, not a scorer.  It keeps the
diff-only static/AST arm separate from behavioral evidence produced earlier in
repository environments.  The current run only classifies stored telemetry;
it does not execute repositories, read the LLM reference, or use the corpus's
accept/reject field.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ITEMS = ROOT / "outputs/metric_seam_pilot/tasks/code_review/items.json"
DEFAULT_TELEMETRY = (
    ROOT
    / "datasets/code-review/pr_test_execution/outputs/"
    "transplant_consolidated_2026_07_13_canonical.csv"
)
DEFAULT_A104_ANCHOR = (
    ROOT / "outputs/metric_seam_pilot/tasks/code_review/a104_cpu_sealed_eval_v4.json"
)
DEFAULT_OUT = (
    ROOT
    / "outputs/metric_seam_pilot/tasks/code_review/"
    "a104_execution_telemetry_bridge_v1.json"
)

CERTIFICATE_LABELS = frozenset({"pinned", "partial_pinned", "vacuous"})
NO_CERTIFICATE_LABELS = frozenset({"none"})
INDETERMINATE_LABELS = frozenset({"indeterminate"})
TELEMETRY_FIELDS = (
    "row_id",
    "batch",
    "paper_id",
    "language",
    "file_source",
    "transplant_pr_label",
    "n_assertion_fail",
    "n_vacuous_pass",
    "n_compile_fail",
    "n_setup_fail",
    "n_uncollected",
    "test_byte_ratio",
    "n_files_total",
    "n_files_test",
    "test_only_ratio",
    "n_lines_added",
    "n_lines_deleted",
    "image_tag",
    "base_sha",
    "era_id",
    "phys_id",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _repo_key(value: Any) -> str:
    return str(value).strip().split("/")[-1].lower()


def _pr_key(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def project_items(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Project only join and shared-input fields; labels are never indexed."""

    projected = []
    for row in rows:
        projected.append({
            "datapoint_id": str(row["datapoint_id"]),
            "repo_key": _repo_key(row["repo"]),
            "pr_key": _pr_key(row["pr_number"]),
            "ctext": str(row["ctext"]),
        })
    return projected


def _float_or_none(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    return float(text)


def _int_or_none(value: Any) -> int | None:
    number = _float_or_none(value)
    return None if number is None else int(number)


def project_telemetry(row: Mapping[str, Any]) -> dict[str, Any]:
    """Retain execution evidence only, excluding accept/reject columns."""

    label = str(row["transplant_pr_label"])
    if label in CERTIFICATE_LABELS:
        state = "finite_execution_certificate"
    elif label in NO_CERTIFICATE_LABELS:
        state = "no_finite_certificate"
    elif label in INDETERMINATE_LABELS:
        state = "indeterminate"
    else:
        state = "execution_error_or_apply_failure"
    return {
        "row_id": str(row["row_id"]),
        "repo_key": _repo_key(row["batch"]),
        "pr_key": _pr_key(row["paper_id"]),
        "language": str(row["language"]),
        "file_source": str(row["file_source"]),
        "transplant_pr_label": label,
        "typed_state": state,
        "certificate_positive": label in CERTIFICATE_LABELS,
        "signals": {
            "n_assertion_fail": _int_or_none(row["n_assertion_fail"]),
            "n_vacuous_pass": _int_or_none(row["n_vacuous_pass"]),
            "n_compile_fail": _int_or_none(row["n_compile_fail"]),
            "n_setup_fail": _int_or_none(row["n_setup_fail"]),
            "n_uncollected": _int_or_none(row["n_uncollected"]),
            "test_byte_ratio": _float_or_none(row["test_byte_ratio"]),
            "n_files_total": _int_or_none(row["n_files_total"]),
            "n_files_test": _int_or_none(row["n_files_test"]),
            "test_only_ratio": _float_or_none(row["test_only_ratio"]),
            "n_lines_added": _int_or_none(row["n_lines_added"]),
            "n_lines_deleted": _int_or_none(row["n_lines_deleted"]),
        },
        "environment_receipt": {
            "image_tag": str(row["image_tag"] or ""),
            "base_sha": str(row["base_sha"] or ""),
            "era_id": str(row["era_id"] or ""),
            "phys_id": str(row["phys_id"] or ""),
        },
    }


def _count_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(row["transplant_pr_label"] for row in rows)
    states = Counter(row["typed_state"] for row in rows)
    return {
        "rows": len(rows),
        "label_counts": dict(sorted(labels.items())),
        "typed_state_counts": dict(sorted(states.items())),
        "finite_execution_certificates": sum(
            row["certificate_positive"] for row in rows
        ),
    }


def build(
    *,
    items_path: Path = DEFAULT_ITEMS,
    telemetry_path: Path = DEFAULT_TELEMETRY,
    anchor_path: Path = DEFAULT_A104_ANCHOR,
) -> dict[str, Any]:
    raw_items = json.loads(items_path.read_text(encoding="utf-8"))
    if not isinstance(raw_items, list) or len(raw_items) != 250:
        raise ValueError("expected the frozen 250-item active code-review corpus")
    items = project_items(raw_items)
    if len({row["datapoint_id"] for row in items}) != len(items):
        raise ValueError("active a104 item IDs are not unique")
    if len({(row["repo_key"], row["pr_key"]) for row in items}) != len(items):
        raise ValueError("active a104 repository/PR join keys are not unique")

    with telemetry_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set(TELEMETRY_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("execution telemetry schema drifted")
        telemetry = [project_telemetry(row) for row in reader]
    if len(telemetry) != 800:
        raise ValueError("expected the frozen 800-row execution replay corpus")
    telemetry_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in telemetry:
        telemetry_groups.setdefault((row["repo_key"], row["pr_key"]), []).append(row)
    ambiguous_telemetry_keys = sorted(
        key for key, rows in telemetry_groups.items() if len(rows) != 1
    )
    telemetry_by_key = {
        key: rows[0] for key, rows in telemetry_groups.items() if len(rows) == 1
    }
    if ambiguous_telemetry_keys != [("chia-blockchain", "496")]:
        raise ValueError("execution telemetry duplicate-key disclosure drifted")

    ids = sorted(row["datapoint_id"] for row in items)
    random.Random(7).shuffle(ids)
    train = set(ids[:150])
    joined = []
    for item in items:
        telemetry_row = telemetry_by_key.get((item["repo_key"], item["pr_key"]))
        if telemetry_row is None:
            continue
        joined.append({
            "datapoint_id": item["datapoint_id"],
            "split": "train" if item["datapoint_id"] in train else "heldout",
            "repo_key": item["repo_key"],
            "pr_key": item["pr_key"],
            **telemetry_row,
        })
    joined.sort(key=lambda row: row["datapoint_id"])

    overall = _count_rows(joined)
    by_split = {
        split: _count_rows([row for row in joined if row["split"] == split])
        for split in ("train", "heldout")
    }
    if (
        overall["rows"] != 32
        or overall["finite_execution_certificates"] != 1
        or overall["label_counts"]
        != {
            "error_no_diff_in_manifest": 1,
            "indeterminate": 22,
            "indeterminate_apply_fail": 1,
            "none": 7,
            "pinned": 1,
        }
        or by_split["train"]["rows"] != 18
        or by_split["heldout"]["rows"] != 14
        or by_split["heldout"]["finite_execution_certificates"] != 1
    ):
        raise ValueError("active a104 execution overlap drifted")
    positive = [row for row in joined if row["certificate_positive"]]
    if (
        positive[0]["transplant_pr_label"] != "pinned"
        or (positive[0]["signals"]["n_assertion_fail"] or 0) <= 0
    ):
        raise ValueError("the sole finite execution certificate is malformed")

    return {
        "schema_version": "metric-seam.active-code-a104-execution-bridge.v1",
        "status": "stored_execution_telemetry_join_complete",
        "lane": "active_code_review_census_criterion_local_augmentation",
        "criterion": "a104",
        "objective": "unsupervised_typed_capability_augmentation_pre_reconstruction",
        "bound_inputs": {
            "active_items": _bound(items_path),
            "stored_execution_telemetry": _bound(telemetry_path),
            "a104_v4_code_depth_anchor_hash_only": _bound(anchor_path),
        },
        "projection_contract": {
            "item_fields_indexed": ["datapoint_id", "repo", "pr_number", "ctext"],
            "telemetry_fields_indexed": list(TELEMETRY_FIELDS),
            "source_columns_explicitly_not_indexed": ["judgement", "judgement_source"],
            "trusted_item_loader_deserializes_full_rows": True,
            "label_inaccessible_corpus_construction_claimed": False,
            "reference_or_outcome_values_used_for_join_or_classification": False,
            "ambiguous_source_keys_excluded_from_join": [
                {"repo_key": repo, "pr_key": pr}
                for repo, pr in ambiguous_telemetry_keys
            ],
        },
        "execution_provenance": {
            "stored_telemetry_from_prior_repository_execution": True,
            "repositories_or_tests_executed_in_this_bridge": False,
            "models_or_apis_called": False,
            "accelerators_used": False,
            "discovery_mode": "replay_of_manual_execution_pipeline_seed",
            "relation_depth": 4,
            "relation_depth_label": "environment_or_world_execution",
        },
        "summary": {
            "active_items": len(items),
            "exact_repository_pr_overlap": overall["rows"],
            "overlap_rate": overall["rows"] / len(items),
            "finite_execution_certificates": overall[
                "finite_execution_certificates"
            ],
            "finite_certificate_rate_conditional_overlap": overall[
                "finite_execution_certificates"
            ]
            / overall["rows"],
            "finite_certificate_rate_over_active_items": overall[
                "finite_execution_certificates"
            ]
            / len(items),
            "overall": overall,
            "by_split": by_split,
        },
        "rows": joined,
        "axis_status": {
            "prompt_articulability": "not_measured",
            "diff_only_code_verifiability": "measured_elsewhere_a104_v4",
            "environment_execution_verifiability": (
                "one_finite_positive_certificate_on_exact_overlap"
            ),
            "reconstruction_agreement": "not_estimated_by_this_bridge",
            "isomorphism": "not_established_extra_environment_evidence",
            "codability": "not_estimated",
        },
        "representation_boundary": {
            "active_candidate_input": "presented_diff_ctext",
            "execution_evidence_input": "repository_checkout_and_test_environment",
            "same_input_representation": False,
            "classification": "capability_augmentation_not_isomorphic_substitution",
        },
        "temporal_disposition": {
            "heldout_use": "exploratory_post_heldout_bridge",
            "positive_certificate_is_heldout": True,
            "no_tuning_or_scalar_score_permitted": True,
            "fresh_split_required_for_confirmatory_augmentation_claim": True,
        },
        "claim_boundary": (
            "One of 32 exact active-item overlaps carries a finite depth-4 execution "
            "certificate. The remaining rows are no-certificate, indeterminate, or "
            "execution failures, never correctness negatives. This is sparse feasibility "
            "evidence for a non-isomorphic capability extension, not reconstruction, "
            "prompt articulability, a whole-criterion score, or a codability rate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--telemetry", type=Path, default=DEFAULT_TELEMETRY)
    parser.add_argument("--a104-anchor", type=Path, default=DEFAULT_A104_ANCHOR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    payload = build(
        items_path=args.items,
        telemetry_path=args.telemetry,
        anchor_path=args.a104_anchor,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
