#!/usr/bin/env python3
"""Freeze an audited exact-truth release for one clean explicit GEPA role."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .adjudicate_gemma import CONFIDENCES, DECISIONS as ADJUDICATION_DECISIONS
from .common import read_jsonl, sha256_file
from .finalize_exact_multi_pass_truth import _decision_key, _winner
from .make_calibration import split_group_for


RELEASE_SCHEMA = "silver-match-v3-clean-gepa-exact-truth-release-v2"
RELEASE_STATUS = "FROZEN_EXACT_TRUTH_RELEASE_AUDITED"
ROLE_FREEZE_SCHEMA = "silver-match-v3-clean-gepa-panel-freeze-v1"
CONSENSUS_SCHEMA = "silver-match-v3-exact-multi-pass-truth-report-v1"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
TRUTH_DECISIONS = frozenset(ADJUDICATION_DECISIONS)


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    uids = {str(row.get("norm_uid") or "") for row in rows}
    if not rows or "" in uids or len(uids) != len(rows):
        raise ValueError(f"empty, missing, or duplicate truth UIDs: {path}")
    return rows


def _localize(recorded: str, panel_root: Path) -> Path:
    path = Path(recorded).resolve()
    if path.is_file():
        return path
    candidate = panel_root / Path(recorded).parent.name / Path(recorded).name
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"cannot localize consensus input: {recorded}")


def _localize_dir(recorded: str, panel_root: Path) -> Path:
    path = Path(recorded).resolve()
    if path.is_dir():
        return path
    candidate = panel_root / Path(recorded).name
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(f"cannot localize independent pass root: {recorded}")


def _verify_ref(ref: dict[str, Any], panel_root: Path, label: str) -> Path:
    path = _localize(str(ref.get("path") or ""), panel_root)
    if sha256_file(path) != str(ref.get("sha256") or ""):
        raise ValueError(f"{label} reference hash drift: {path}")
    return path


def _same_ref(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        bool(left.get("path"))
        and bool(right.get("path"))
        and Path(str(left["path"])).resolve() == Path(str(right["path"])).resolve()
        and str(left.get("sha256") or "") == str(right.get("sha256") or "")
    )


def _load_task_norms(
    manifest_path: Path,
    task: str,
    wanted_uids: set[str],
    *,
    panel_root: Path,
) -> dict[str, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    norms: dict[str, dict[str, Any]] = {}
    for corpus, meta in (manifest.get("corpora") or {}).items():
        if meta.get("task") != task:
            continue
        path = Path(str(meta.get("path") or ""))
        if not path.is_absolute():
            path = manifest_path.parent / path
        if not path.is_file():
            path = _localize(str(path), panel_root)
        for row in read_jsonl(path.resolve()):
            uid = str(row["norm_uid"])
            if uid not in wanted_uids:
                continue
            if uid in norms or row.get("task") != task or row.get("corpus") != corpus:
                raise ValueError(f"canonical manifest task/corpus/UID drift: {uid}")
            norms[uid] = row
    if set(norms) != wanted_uids:
        raise ValueError(
            f"manifest lacks canonical task norms: {sorted(wanted_uids - set(norms))[:3]}"
        )
    return norms


def _expected_hydrated_item(canonical: dict[str, Any], *, role: str) -> dict[str, Any]:
    group = split_group_for(canonical)
    return {
        **canonical,
        "gepa_role": role,
        "permanently_excluded_from_mi_and_outcome_estimation": True,
        "permanently_excluded_from_retriever_gradients": True,
        "predeclared_split": "train",
        "source_group": group,
        "split": "train",
        "split_group": group,
        "truth_hidden": True,
    }


def _verified_ref_identity(
    ref: dict[str, Any], panel_root: Path, label: str
) -> tuple[Path, str]:
    path = _verify_ref(ref, panel_root, label).resolve()
    return path, sha256_file(path)


def _verify_resolver_lineage(
    *,
    name: str,
    validation: dict[str, Any],
    panel_root: Path,
    candidate_release_path: Path,
    candidate_release_sha: str,
    current_unresolved: set[str],
    initial_label_refs: set[tuple[Path, str]],
    task: str,
) -> None:
    """Prove a later pass was built only from the prior blind frontier."""
    schema = str(validation.get("schema_version") or "")
    inputs = validation.get("inputs") or {}
    source_path, source_sha = _verified_ref_identity(
        inputs.get("source_pack_validation") or {},
        panel_root,
        f"{name} resolver source pack",
    )
    if (
        validation.get("task") != task
        or validation.get("truth_hidden") is not True
        or validation.get("prior_decisions_and_metric_ids_hidden") is not True
        or source_path != candidate_release_path.resolve()
        or source_sha != candidate_release_sha
    ):
        raise ValueError(f"resolver source-pack/blinding lineage drift: {name}")

    if schema == "silver-match-v3-semantic-resolver-pack-v1":
        selection = validation.get("selection_rule") or {}
        semantic_ref = _verified_ref_identity(
            inputs.get("semantic_labels") or {},
            panel_root,
            f"{name} semantic resolver labels",
        )
        strict_ref = _verified_ref_identity(
            inputs.get("strict_key") or {},
            panel_root,
            f"{name} semantic resolver strict key",
        )
        if (
            selection.get("mode") != "exact_disagreements_only"
            or selection.get("all_exact_strict_key_mismatches") is not True
            or {semantic_ref, strict_ref} != initial_label_refs
        ):
            raise ValueError(f"semantic resolver prior-pass lineage drift: {name}")
        return

    if schema == "silver-match-v3-exact-unresolved-resolver-pack-v1":
        if (
            validation.get("selection_rule")
            != "all_and_only_current_exact_consensus_unresolved_uids"
        ):
            raise ValueError(f"exact resolver selection-rule drift: {name}")
        unresolved_path = _verify_ref(
            inputs.get("unresolved") or {}, panel_root, f"{name} prior unresolved"
        )
        unresolved_rows = _rows(unresolved_path)
        unresolved_uids = {str(row["norm_uid"]) for row in unresolved_rows}
        if unresolved_uids != current_unresolved or any(
            row.get("task") != task for row in unresolved_rows
        ):
            raise ValueError(f"exact resolver prior-unresolved lineage drift: {name}")
        return

    raise ValueError(f"unsupported resolver pack schema: {name}/{schema}")


def _named_paths(values: list[str], flag: str) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{flag} must be NAME=PATH: {value!r}")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path or name in output:
            raise ValueError(f"invalid or duplicate {flag}: {value!r}")
        output[name] = Path(raw_path).resolve()
    return output


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    task, role = args.task, args.role
    truth_path = Path(args.truth).resolve()
    report_path = Path(args.consensus_report).resolve()
    role_freeze_path = Path(args.role_freeze).resolve()
    identities_path = Path(args.identities).resolve()
    independence_path = Path(args.independence_audit).resolve()
    candidate_release_path = Path(args.candidate_release).resolve()
    raw_transcript_guides = args.transcript_guide
    if isinstance(raw_transcript_guides, (str, Path)):
        raw_transcript_guides = [raw_transcript_guides]
    transcript_guide_paths = [
        Path(value).resolve() for value in raw_transcript_guides
    ]
    output = Path(args.output).resolve()
    if not transcript_guide_paths or len(set(transcript_guide_paths)) != len(
        transcript_guide_paths
    ):
        raise ValueError("one or more distinct transcript guides are required")
    for transcript_guide_path in transcript_guide_paths:
        if not transcript_guide_path.is_file():
            raise FileNotFoundError(transcript_guide_path)
    transcript_guide_shas = [
        sha256_file(path) for path in transcript_guide_paths
    ]
    if len(set(transcript_guide_shas)) != len(transcript_guide_shas):
        raise ValueError("transcript guides must have distinct immutable content")
    if output.exists():
        raise FileExistsError(output)
    truth = _rows(truth_path)
    truth_uids = {str(row["norm_uid"]) for row in truth}
    panel_root = truth_path.parents[1]

    role_freeze = json.loads(role_freeze_path.read_text(encoding="utf-8"))
    frozen_identities = (role_freeze.get("outputs") or {}).get("identities") or {}
    if (
        role_freeze.get("schema_version") != ROLE_FREEZE_SCHEMA
        or role_freeze.get("status") != "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES"
        or role_freeze.get("task") != task
        or role_freeze.get("role") != role
        or role_freeze.get("required_upstream_split") != "train"
        or int(role_freeze.get("selected_count") or -1) != len(truth)
        or frozen_identities.get("sha256") != sha256_file(identities_path)
    ):
        raise ValueError("role identity freeze does not bind truth release universe")
    identity_rows = _rows(identities_path)
    identity_uids = {str(row["norm_uid"]) for row in identity_rows}
    if identity_uids != truth_uids:
        raise ValueError("truth release and frozen identities differ")

    candidate_release = json.loads(candidate_release_path.read_text(encoding="utf-8"))
    candidate_inputs = candidate_release.get("inputs") or {}
    candidate_outputs = candidate_release.get("outputs") or {}
    candidate_sha = str((candidate_outputs.get("candidates") or {}).get("sha256") or "")
    if (
        candidate_release.get("schema_version")
        != "silver-match-v3-clean-gepa-label-pack-v1"
        or candidate_release.get("status") != "FROZEN_TRUTH_HIDDEN_BEFORE_LABELING"
        or candidate_release.get("task") != task
        or candidate_release.get("gepa_role") != role
        or int(candidate_release.get("count") or -1) != len(truth)
        or not candidate_sha
    ):
        raise ValueError("candidate release is not the frozen truth-hidden role pack")
    if (candidate_inputs.get("identities") or {}).get("sha256") != sha256_file(
        identities_path
    ) or (candidate_inputs.get("identity_freeze") or {}).get("sha256") != sha256_file(
        role_freeze_path
    ):
        raise ValueError("candidate release does not bind role identities/freeze")
    required_source_refs = (
        "manifest",
        "bank_source",
        "candidate_source",
        "identities",
        "identity_freeze",
        "upstream_role_freeze",
    )
    verified_source_paths = {}
    for key in required_source_refs:
        verified_source_paths[key] = _verify_ref(
            candidate_inputs.get(key) or {}, panel_root, f"candidate {key}"
        )
    candidate_path = _verify_ref(
        candidate_outputs.get("candidates") or {}, panel_root, "candidate output"
    )
    if sha256_file(candidate_path) != candidate_sha:
        raise ValueError("candidate output hash differs from candidate release")
    canonical_norms = _load_task_norms(
        verified_source_paths["manifest"],
        task,
        truth_uids,
        panel_root=panel_root,
    )
    for row in truth:
        uid = str(row["norm_uid"])
        canonical = canonical_norms[uid]
        if (
            row.get("corpus") != canonical.get("corpus")
            or row.get("row") != canonical.get("row")
            or row.get("source_group") != split_group_for(canonical)
        ):
            raise ValueError(f"truth identity differs from canonical norm: {uid}")
    bank_payload = json.loads(
        verified_source_paths["bank_source"].read_text(encoding="utf-8")
    )
    bank_rows = bank_payload.get("metrics") or bank_payload.get("bank") or []
    bank_ids = {str(row.get("metric_id") or "") for row in bank_rows}
    if not bank_ids or "" in bank_ids or len(bank_ids) != len(bank_rows):
        raise ValueError("candidate release bank has missing or duplicate metric IDs")
    current_bank_source_sha256 = str(candidate_release.get("bank_source_sha256") or "")
    if (
        not current_bank_source_sha256
        or str(bank_payload.get("source_sha256") or "") != current_bank_source_sha256
    ):
        raise ValueError("candidate release and canonical bank source hash differ")
    for row in truth:
        decision = str(row.get("decision") or "")
        metric_id = row.get("metric_id")
        if (
            row.get("task") != task
            or row.get("gepa_role") != role
            or row.get("split") != "train"
            or row.get("current_bank_source_sha256") != current_bank_source_sha256
            or decision not in TRUTH_DECISIONS
            or (decision == "MATCH" and str(metric_id) not in bank_ids)
            or (decision != "MATCH" and metric_id is not None)
        ):
            raise ValueError(
                f"truth row has invalid task/role/decision/current-bank leaf: "
                f"{row.get('norm_uid')}"
            )

    independence = json.loads(independence_path.read_text(encoding="utf-8"))
    required_independence = (
        "distinct_bank_order",
        "distinct_item_order",
        "distinct_seeds",
        "pass_predictions_mutually_visible",
        "prior_truth_or_predictions_exposed_to_either_pass",
        "same_bank_leaf_set",
        "same_canonical_item_content_by_uid",
        "same_frozen_source_pack",
        "same_uid_set",
    )
    if (
        independence.get("schema_version")
        != "silver-match-v3-independent-pack-view-audit-v1"
        or independence.get("status")
        != "FROZEN_MUTUALLY_PREDICTION_HIDDEN_BEFORE_LABELING"
        or independence.get("task") != task
        or int(independence.get("count") or -1) != len(truth)
        or any(
            independence.get(key) is not expected
            for key, expected in (
                ("distinct_bank_order", True),
                ("distinct_item_order", True),
                ("distinct_seeds", True),
                ("pass_predictions_mutually_visible", False),
                ("prior_truth_or_predictions_exposed_to_either_pass", False),
                ("same_bank_leaf_set", True),
                ("same_canonical_item_content_by_uid", True),
                ("same_frozen_source_pack", True),
                ("same_uid_set", True),
            )
        )
    ):
        raise ValueError(
            "independence audit does not certify frozen mutually hidden passes: "
            + ",".join(required_independence)
        )

    # Resolve the pass roots named by the independence audit and bind every
    # claimed validation/bank/items hash to the actual artifact.  Each pass
    # validation must in turn bind the exact source label-pack validation and
    # the same manifest/identity/freeze provenance as the candidate release.
    independence_passes = independence.get("passes") or {}
    if set(independence_passes) != {"A", "B"}:
        raise ValueError("independence audit must bind exactly passes A and B")
    independent_artifacts: dict[str, dict[str, Any]] = {}
    candidate_release_sha = sha256_file(candidate_release_path)
    for name in ("A", "B"):
        meta = independence_passes[name]
        pass_root = _localize_dir(str(meta.get("root") or ""), panel_root)
        validation_path = pass_root / "validation.json"
        bank_path = pass_root / "bank.json"
        items_path = pass_root / "items.jsonl"
        if (
            not validation_path.is_file()
            or not bank_path.is_file()
            or not items_path.is_file()
            or sha256_file(validation_path) != str(meta.get("validation_sha256") or "")
            or sha256_file(bank_path) != str(meta.get("bank_sha256") or "")
            or sha256_file(items_path) != str(meta.get("items_sha256") or "")
        ):
            raise ValueError(
                f"independence pass {name} artifacts are missing or drifted"
            )
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        validation_inputs = validation.get("inputs") or {}
        validation_outputs = validation.get("outputs") or {}
        source_pack = validation.get("source_pack") or {}
        source_pack_path = _localize_dir(
            str(source_pack.get("path") or ""), panel_root
        )
        if (
            validation.get("schema_version")
            != "silver-match-v3-permuted-independent-teacher-pack-v1"
            or validation.get("status") != "FROZEN_TRUTH_HIDDEN_BEFORE_LABELING"
            or validation.get("truth_hidden") is not True
            or validation.get(
                "prior_decisions_proposals_predictions_and_outcomes_hidden"
            )
            is not True
            or validation.get("task") != task
            or validation.get("gepa_role") != role
            or int(validation.get("count") or -1) != len(truth)
            or int(validation.get("seed") or -1) != int(meta.get("seed") or -2)
            or source_pack_path != candidate_release_path.parent.resolve()
            or str(source_pack.get("validation_sha256") or "") != candidate_release_sha
            or not _same_ref(
                validation_outputs.get("bank") or {},
                {"path": str(bank_path), "sha256": sha256_file(bank_path)},
            )
            or not _same_ref(
                validation_outputs.get("items") or {},
                {"path": str(items_path), "sha256": sha256_file(items_path)},
            )
            or any(
                not _same_ref(
                    validation_inputs.get(key) or {}, candidate_inputs.get(key) or {}
                )
                for key in required_source_refs
            )
        ):
            raise ValueError(f"independence pass {name} validation provenance drift")
        item_uids = {str(row.get("norm_uid") or "") for row in read_jsonl(items_path)}
        if item_uids != truth_uids:
            raise ValueError(
                f"independence pass {name} items differ from truth universe"
            )
        independent_artifacts[name] = {
            "root": str(pass_root),
            "validation": {
                "path": str(validation_path),
                "sha256": sha256_file(validation_path),
            },
            "bank": {"path": str(bank_path), "sha256": sha256_file(bank_path)},
            "items": {"path": str(items_path), "sha256": sha256_file(items_path)},
            "source_pack_validation_sha256": candidate_release_sha,
        }

    report = json.loads(report_path.read_text(encoding="utf-8"))
    resolved = (report.get("outputs") or {}).get("resolved") or {}
    unresolved = (report.get("outputs") or {}).get("unresolved") or {}
    if (
        report.get("schema_version") != CONSENSUS_SCHEMA
        or report.get("complete") is not True
        or report.get("task") != task
        or report.get("gepa_role") != role
        or int(report.get("source_count") or -1) != len(truth)
        or int(report.get("resolved_count") or -1) != len(truth)
        or int(report.get("unresolved_count", -1)) != 0
        or resolved.get("sha256") != sha256_file(truth_path)
        or unresolved.get("sha256") != EMPTY_SHA256
    ):
        raise ValueError("exact consensus report does not release this complete truth")

    passes = []
    replay_passes: list[
        tuple[
            str,
            dict[str, Any],
            dict[str, dict[str, Any]],
            dict[str, Any],
            Path,
        ]
    ] = []
    report_passes = (report.get("inputs") or {}).get("passes") or {}
    if len(report_passes) < 2:
        raise ValueError("truth release requires at least two exact-consensus passes")
    report_rounds = report.get("rounds") or []
    round_names = [str(round_meta.get("pass") or "") for round_meta in report_rounds]
    if (
        len(report_rounds) != len(report_passes)
        or "" in round_names
        or len(set(round_names)) != len(round_names)
        or set(round_names) != set(report_passes)
    ):
        raise ValueError("consensus rounds do not define each bound pass exactly once")
    ordered_report_passes = [(name, report_passes[name]) for name in round_names]
    external_transcript_audits = _named_paths(
        list(getattr(args, "transcript_audit", None) or []), "--transcript-audit"
    )
    if external_transcript_audits and set(external_transcript_audits) != set(
        report_passes
    ):
        raise ValueError(
            "external strict transcript audits must cover every consensus pass"
        )
    full_source_passes = [
        (name, meta)
        for name, meta in ordered_report_passes
        if int(meta.get("count") or -1) == len(truth)
    ]
    if len(full_source_passes) != 2 or [name for name, _ in full_source_passes] != [
        name for name, _ in ordered_report_passes[:2]
    ]:
        raise ValueError(
            "consensus report must begin with exactly two full-source independent passes"
        )
    expected_independent_sets = {
        (
            artifact["validation"]["sha256"],
            artifact["bank"]["sha256"],
            artifact["items"]["sha256"],
        )
        for artifact in independent_artifacts.values()
    }
    report_full_source_sets = {
        (
            str((meta.get("pack_validation") or {}).get("sha256") or ""),
            str(meta.get("pack_bank_sha256") or ""),
            str(meta.get("pack_items_sha256") or ""),
        )
        for _, meta in full_source_passes
    }
    if (
        len(expected_independent_sets) != 2
        or len(report_full_source_sets) != 2
        or report_full_source_sets != expected_independent_sets
    ):
        raise ValueError(
            "full-source consensus pass artifacts differ from independence audit A/B"
        )
    seen_label_shas: set[str] = set()
    seen_pack_validation_shas: set[str] = set()
    for name, meta in ordered_report_passes:
        labels_meta = meta.get("labels") or {}
        pack_meta = meta.get("pack_validation") or {}
        label_path = _localize(str(labels_meta.get("path") or ""), panel_root)
        pack_validation_path = _localize(str(pack_meta.get("path") or ""), panel_root)
        label_validation_path = label_path.parent / "labels.validation.json"
        if (
            sha256_file(label_path) != labels_meta.get("sha256")
            or sha256_file(pack_validation_path) != pack_meta.get("sha256")
            or not label_validation_path.is_file()
            or (
                name in independent_artifacts
                and sha256_file(pack_validation_path)
                != independent_artifacts[name]["validation"]["sha256"]
            )
            or (
                name in independent_artifacts
                and str(meta.get("pack_bank_sha256") or "")
                != independent_artifacts[name]["bank"]["sha256"]
            )
            or (
                name in independent_artifacts
                and str(meta.get("pack_items_sha256") or "")
                != independent_artifacts[name]["items"]["sha256"]
            )
        ):
            raise ValueError(f"consensus pass artifact drift: {name}")
        pack_validation = json.loads(pack_validation_path.read_text(encoding="utf-8"))
        pack_outputs = pack_validation.get("outputs") or {}
        pack_bank_path = _verify_ref(
            pack_outputs.get("bank") or {}, panel_root, f"{name} pack bank"
        )
        pack_items_path = _verify_ref(
            pack_outputs.get("items") or {}, panel_root, f"{name} pack items"
        )
        if str(meta.get("pack_bank_sha256") or "") != sha256_file(
            pack_bank_path
        ) or str(meta.get("pack_items_sha256") or "") != sha256_file(pack_items_path):
            raise ValueError(f"consensus pass bank/items drift: {name}")
        pack_item_rows = _rows(pack_items_path)
        pack_item_uids = {str(row["norm_uid"]) for row in pack_item_rows}
        label_rows = _rows(label_path)
        label_by_uid = {str(row["norm_uid"]): row for row in label_rows}
        if (
            set(label_by_uid) != pack_item_uids
            or not pack_item_uids <= truth_uids
            or pack_validation.get("task") != task
            or str(pack_validation.get("bank_source_sha256") or "")
            != current_bank_source_sha256
            or int(meta.get("count") or -1) != len(label_by_uid)
        ):
            raise ValueError(f"consensus pass label/pack universe drift: {name}")
        for item in pack_item_rows:
            uid = str(item["norm_uid"])
            if item != _expected_hydrated_item(canonical_norms[uid], role=role):
                raise ValueError(
                    f"pass item differs from canonical hydrated content: {name}/{uid}"
                )
        label_sha = sha256_file(label_path)
        pack_validation_sha = sha256_file(pack_validation_path)
        if (
            label_sha in seen_label_shas
            or pack_validation_sha in seen_pack_validation_shas
        ):
            raise ValueError(f"consensus reuses a label or pass pack: {name}")
        seen_label_shas.add(label_sha)
        seen_pack_validation_shas.add(pack_validation_sha)
        for uid, row in label_by_uid.items():
            decision, metric_id = _decision_key(row)
            confidence = str(row.get("confidence") or "").lower()
            if (
                row.get("task") != task
                or row.get("current_bank_source_sha256") != current_bank_source_sha256
                or decision not in TRUTH_DECISIONS
                or confidence not in CONFIDENCES
                or (decision == "MATCH" and metric_id not in bank_ids)
                or (decision != "MATCH" and row.get("metric_id") is not None)
            ):
                raise ValueError(f"invalid exact-consensus label: {name}/{uid}")
        replay_passes.append(
            (name, meta, label_by_uid, pack_validation, pack_validation_path)
        )
        validation = json.loads(label_validation_path.read_text(encoding="utf-8"))
        if (
            validation.get("schema_version")
            != "silver-match-v3-independent-label-validation-v1"
            or validation.get("complete") is not True
            or validation.get("task") != task
            or int(validation.get("count") or -1) != int(meta.get("count") or -2)
            or ((validation.get("output") or {}).get("sha256"))
            != sha256_file(label_path)
            or ((validation.get("pack_validation") or {}).get("sha256"))
            != sha256_file(pack_validation_path)
            or (
                validation.get("retrieval_candidate_sha256")
                not in (None, candidate_sha)
            )
        ):
            raise ValueError(f"validated-label release drift: {name}")
        embedded_transcript_ref = validation.get("transcript_audit") or {}
        external_transcript_path = external_transcript_audits.get(name)
        if external_transcript_path is not None:
            if not external_transcript_path.is_file():
                raise FileNotFoundError(external_transcript_path)
            transcript_path = external_transcript_path
            transcript_source = "external_source_workspace_hash_equivalent"
        elif embedded_transcript_ref:
            transcript_path = _verify_ref(
                embedded_transcript_ref, panel_root, f"{name} embedded transcript audit"
            )
            transcript_source = "embedded_pass_workspace"
        else:
            raise ValueError(f"strict transcript audit is required for pass: {name}")
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        raw_chunks = validation.get("raw_chunks") or {}
        transcript_chunks = transcript.get("chunks") or []
        transcript_by_chunk = {
            str(row.get("chunk") or ""): row for row in transcript_chunks
        }
        chunk_refs = pack_outputs.get("chunks") or {}
        chunk_sha_by_id = {
            Path(str(recorded)).stem: str(expected)
            for recorded, expected in chunk_refs.items()
        }
        audit_guides = transcript.get("guides") or []
        transcript_schema = transcript.get("schema_version")
        common_transcript_invalid = (
            transcript.get("status") != "PASS"
            or transcript.get("complete") is not True
            or transcript.get("violations") != []
            or (transcript.get("bank") or {}).get("sha256")
            != sha256_file(pack_bank_path)
            or (transcript.get("items") or {}).get("sha256")
            != sha256_file(pack_items_path)
            or (transcript.get("pack_validation") or {}).get("sha256")
            != sha256_file(pack_validation_path)
            or int(transcript.get("expected_chunks") or -1) != len(raw_chunks)
            or int(transcript.get("audited_chunks") or -1) != len(raw_chunks)
            or len(transcript_by_chunk) != len(transcript_chunks)
            or set(transcript_by_chunk) != set(raw_chunks)
            or set(chunk_sha_by_id) != set(raw_chunks)
        )
        if transcript_schema == "silver-match-v3-isolated-labeler-transcript-audit-v1":
            observed_guide_shas = [
                str(row.get("sha256") or "") for row in audit_guides
            ]
            backend_transcript_valid = (
                transcript.get("full_pack_artifact_binding") is True
                and len(observed_guide_shas) == len(transcript_guide_shas)
                and len(set(observed_guide_shas)) == len(observed_guide_shas)
                and set(observed_guide_shas) == set(transcript_guide_shas)
            )
        elif transcript_schema == "silver-match-v3-claude-labeler-transcript-audit-v1":
            contract = transcript.get("contract") or {}
            execution_freeze = transcript.get("execution_freeze") or {}
            execution_freeze_path = _localize(
                str(execution_freeze.get("path") or ""), panel_root
            )
            execution_freeze_payload = json.loads(
                execution_freeze_path.read_text(encoding="utf-8")
            )
            frozen_implementation = {
                str(row.get("sha256") or "")
                for row in execution_freeze_payload.get("implementation") or []
            }
            backend_transcript_valid = (
                transcript.get("truth_hidden") is True
                and execution_freeze.get("sha256")
                == sha256_file(execution_freeze_path)
                and set(transcript_guide_shas) <= frozen_implementation
                and contract.get("only_read_and_structured_output_tools_observed")
                is True
                and contract.get(
                    "every_chunk_read_only_its_guides_bank_and_assigned_items"
                )
                is True
                and contract.get(
                    "sample_keys_predictions_proposals_mi_outcomes_and_gemma_absent"
                )
                is True
                and contract.get("network_and_mcp_use_absent") is True
                and contract.get("final_payload_exactly_bound_to_raw_labels") is True
            )
        else:
            backend_transcript_valid = False
        if common_transcript_invalid or not backend_transcript_valid:
            raise ValueError(f"strict transcript audit drift: {name}")
        for chunk, raw_meta in raw_chunks.items():
            audited = transcript_by_chunk[chunk]
            chunk_path = pack_validation_path.parent / "chunks" / f"{chunk}.jsonl"
            if transcript_schema == "silver-match-v3-isolated-labeler-transcript-audit-v1":
                raw_path = label_path.parent / "raw_labels" / f"{chunk}.json"
                log_path = label_path.parent / "logs" / f"{chunk}.log"
                backend_chunk_valid = (
                    raw_path.is_file()
                    and log_path.is_file()
                    and str(audited.get("log_sha256") or "")
                    == sha256_file(log_path)
                    and int(audited.get("command_count") or 0) >= 1
                )
            else:
                raw_path = Path(str(audited.get("raw_label_path") or "")).resolve()
                transcript_artifact = Path(
                    str(audited.get("transcript_path") or "")
                ).resolve()
                stderr_artifact = Path(str(audited.get("stderr_path") or "")).resolve()
                backend_chunk_valid = (
                    raw_path == (label_path.parent / "raw_labels" / f"{chunk}.json").resolve()
                    and raw_path.is_file()
                    and transcript_artifact.is_file()
                    and stderr_artifact.is_file()
                    and str(audited.get("transcript_sha256") or "")
                    == sha256_file(transcript_artifact)
                    and str(audited.get("stderr_sha256") or "")
                    == sha256_file(stderr_artifact)
                    and int(audited.get("event_count") or 0) >= 1
                )
            if (
                not chunk_path.is_file()
                or not backend_chunk_valid
                or sha256_file(chunk_path) != chunk_sha_by_id[chunk]
                or str(audited.get("chunk_sha256") or "") != sha256_file(chunk_path)
                or str(raw_meta.get("raw_sha256") or "") != sha256_file(raw_path)
                or str(audited.get("raw_label_sha256") or "") != sha256_file(raw_path)
            ):
                raise ValueError(
                    f"strict transcript chunk/raw/log binding drift: {name}/{chunk}"
                )
        transcript_audit = {
            "mode": "strict_isolation_audit",
            "source": transcript_source,
            "path": str(transcript_path),
            "sha256": sha256_file(transcript_path),
            "source_workspace_pack_root": str(transcript.get("pack_root") or ""),
            "full_pack_artifact_binding": True,
            "artifact_equivalence_verified": True,
            "guide_sha256": transcript_guide_shas[0],
            "guide_sha256s": transcript_guide_shas,
            "schema_version": transcript_schema,
        }
        passes.append(
            {
                "name": name,
                "labels": {"path": str(label_path), "sha256": sha256_file(label_path)},
                "label_validation": {
                    "path": str(label_validation_path),
                    "sha256": sha256_file(label_validation_path),
                },
                "pack_validation": {
                    "path": str(pack_validation_path),
                    "sha256": sha256_file(pack_validation_path),
                },
                "transcript_audit": transcript_audit,
            }
        )

    # Recompute the deterministic consensus from the hash-bound label passes.
    # This makes a jointly edited report/resolved file insufficient to forge a
    # release: the first two passes must cover the full source and every later
    # resolver must cover exactly the still-unresolved set in chronological
    # report order.
    if len(report_rounds) != len(replay_passes):
        raise ValueError("consensus report round count differs from bound passes")
    votes: dict[str, list[tuple[str, tuple[str, str | None], dict[str, Any]]]] = {
        uid: [] for uid in truth_uids
    }
    current_unresolved = set(truth_uids)
    initial_label_refs = {
        (
            _localize(str((meta.get("labels") or {}).get("path") or ""), panel_root),
            str((meta.get("labels") or {}).get("sha256") or ""),
        )
        for _, meta, _, _, _ in replay_passes[:2]
    }
    for ordinal, (
        name,
        meta,
        label_by_uid,
        pack_validation,
        _pack_validation_path,
    ) in enumerate(replay_passes, start=1):
        observed = set(label_by_uid)
        expected = truth_uids if ordinal <= 2 else current_unresolved
        if observed != expected:
            raise ValueError(
                f"consensus pass does not cover exact unresolved frontier: {name}"
            )
        if ordinal > 2:
            _verify_resolver_lineage(
                name=name,
                validation=pack_validation,
                panel_root=panel_root,
                candidate_release_path=candidate_release_path,
                candidate_release_sha=candidate_release_sha,
                current_unresolved=current_unresolved,
                initial_label_refs=initial_label_refs,
                task=task,
            )
        before = len(current_unresolved)
        for uid, row in label_by_uid.items():
            votes[uid].append((name, _decision_key(row), row))
        current_unresolved = {
            uid
            for uid in truth_uids
            if _winner([key for _, key, _ in votes[uid]]) is None
        }
        round_meta = report_rounds[ordinal - 1]
        expected_round_core = {
            "pass": name,
            "ordinal": ordinal,
            "labeled_count": len(observed),
            "unresolved_before": before,
            "newly_resolved": before - len(current_unresolved),
            "unresolved_after": len(current_unresolved),
            "count": len(label_by_uid),
            "labels": meta.get("labels"),
            "pack_validation": meta.get("pack_validation"),
            "pack_items_sha256": meta.get("pack_items_sha256"),
            "pack_bank_sha256": meta.get("pack_bank_sha256"),
        }
        if any(
            round_meta.get(key) != value for key, value in expected_round_core.items()
        ):
            raise ValueError(f"consensus report round metadata drift: {name}")
    if current_unresolved:
        raise ValueError("bound pass replay leaves unresolved truth rows")
    truth_by_uid = {str(row["norm_uid"]): row for row in truth}
    for uid, available in votes.items():
        winner = _winner([key for _, key, _ in available])
        if winner is None:
            raise ValueError(f"bound pass replay has no unique exact winner: {uid}")
        supporters = [name for name, key, _ in available if key == winner]
        supporter_rows = [row for _, key, row in available if key == winner]
        confidence = (
            "high"
            if sum(
                str(row.get("confidence") or "").lower() == "high"
                for row in supporter_rows
            )
            >= 2
            else "medium"
        )
        released = truth_by_uid[uid]
        if (
            (str(released.get("decision") or ""), released.get("metric_id")) != winner
            or str(released.get("confidence") or "").lower() != confidence
            or list(released.get("agreement_sources") or []) != supporters
        ):
            raise ValueError(f"released truth differs from bound pass consensus: {uid}")

    payload = {
        "schema_version": RELEASE_SCHEMA,
        "status": RELEASE_STATUS,
        "task": task,
        "role": role,
        "count": len(truth),
        "truth": {"path": str(truth_path), "sha256": sha256_file(truth_path)},
        "consensus_report": {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
        },
        "role_freeze": {
            "path": str(role_freeze_path),
            "sha256": sha256_file(role_freeze_path),
        },
        "identities": {
            "path": str(identities_path),
            "sha256": sha256_file(identities_path),
        },
        "independence_audit": {
            "path": str(independence_path),
            "sha256": sha256_file(independence_path),
            "passes": independent_artifacts,
        },
        "candidate_release": {
            "path": str(candidate_release_path),
            "sha256": sha256_file(candidate_release_path),
            "candidate_sha256": candidate_sha,
        },
        "consensus_replay": {
            "algorithm": (
                "first two full-source passes; each later pass exact prior unresolved; "
                "unique decision-and-leaf winner with at least two votes"
            ),
            "pass_order": [name for name, _, _, _, _ in replay_passes],
            "round_count": len(replay_passes),
            "resolved_count": len(truth),
            "unresolved_count": 0,
            "round_metadata_verified": True,
            "released_decision_metric_confidence_supporters_exact": True,
        },
        "passes": passes,
        "scientific_contract": {
            "exact_decision_and_leaf_consensus_complete": True,
            "consensus_recomputed_from_bound_pass_labels": True,
            "all_pass_labels_and_validations_hash_bound": True,
            "transcripts_hash_bound_and_leakage_audited": True,
            "strict_transcript_pass_required_for_every_consensus_pass": True,
            "cross_workspace_artifacts_hash_equivalent": True,
            "legacy_transcripts_allowed": False,
            "truth_may_be_used_only_for_declared_gepa_role": True,
        },
    }
    if bool(getattr(args, "check_only", False)):
        return {**payload, "output": None, "output_sha256": None, "check_only": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**payload, "output": str(output), "output_sha256": sha256_file(output)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--role", choices=("optimize", "select"), required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--consensus-report", required=True)
    parser.add_argument("--role-freeze", required=True)
    parser.add_argument("--identities", required=True)
    parser.add_argument("--independence-audit", required=True)
    parser.add_argument("--candidate-release", required=True)
    parser.add_argument(
        "--transcript-guide",
        action="append",
        required=True,
        help=(
            "Immutable guide read by every labeler chunk; repeat for every guide "
            "that the strict transcript audit binds."
        ),
    )
    parser.add_argument(
        "--transcript-audit",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Strict PASS audit for each consensus pass; repeat for every pass name.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate every release invariant without writing the output artifact.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(freeze(parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
