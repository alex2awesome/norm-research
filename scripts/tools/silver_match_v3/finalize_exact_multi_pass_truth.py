#!/usr/bin/env python3
"""Resolve truth only after two independent passes agree on decision and leaf.

The first two passes must each cover the complete frozen source pack.  Every
later pass must cover exactly the rows still unresolved after all earlier
passes.  This makes each resolver round disagreement-only and prevents labels
from being spent on, or leaking into, already resolved examples.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .adjudicate_gemma import CONFIDENCES, DECISIONS
from .common import read_jsonl, sha256_file, write_jsonl


SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    output = {str(row.get("norm_uid") or ""): row for row in rows}
    if "" in output or len(output) != len(rows):
        raise ValueError(f"missing or duplicate norm_uid values: {path}")
    return output


def _parse_specs(values: list[str], flag: str) -> list[tuple[str, Path]]:
    output: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"{flag} must be NAME=PATH: {value!r}")
        name, raw_path = value.split("=", 1)
        if not SAFE_NAME.fullmatch(name) or name in seen or not raw_path:
            raise ValueError(f"invalid or duplicate {flag}: {value!r}")
        output.append((name, Path(raw_path).resolve()))
        seen.add(name)
    return output


def _decision_key(row: dict[str, Any]) -> tuple[str, str | None]:
    decision = str(row.get("decision") or "").upper()
    metric_id = str(row["metric_id"]) if decision == "MATCH" else None
    return decision, metric_id


def _winner(votes: list[tuple[str, str | None]]) -> tuple[str, str | None] | None:
    counts = Counter(votes)
    winners = [key for key, count in counts.items() if count >= 2]
    return winners[0] if len(winners) == 1 else None


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision": row.get("decision"),
        "metric_id": row.get("metric_id"),
        "confidence": row.get("confidence"),
        "reason": row.get("reason"),
        "label_source": row.get("label_source"),
        "annotator": row.get("annotator"),
    }


def _validate_role_freeze(
    freeze_path: Path,
    source_validation: dict[str, Any],
    requested_role: str,
) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    legacy_identity_sha = str(
        (source_validation.get("input_hashes") or {}).get("items") or ""
    )
    modern_inputs = source_validation.get("inputs") or {}
    modern_identity_sha = str(
        (modern_inputs.get("identities") or {}).get("sha256") or ""
    )
    expected_identity_sha = legacy_identity_sha or modern_identity_sha
    modern_freeze_sha = str(
        (modern_inputs.get("identity_freeze") or {}).get("sha256") or ""
    )
    output = (freeze.get("outputs") or {}).get("identities") or {}
    contract = freeze.get("content_contract") or {}
    if (
        freeze.get("schema_version") != "silver-match-v3-clean-gepa-panel-freeze-v1"
        or freeze.get("status") != "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES"
        or freeze.get("task") != source_validation.get("task")
        or freeze.get("role") != requested_role
        or freeze.get("required_upstream_split") != "train"
        or str(output.get("sha256") or "") != expected_identity_sha
        or int(freeze.get("selected_count") or -1) != int(source_validation.get("count") or -2)
        or contract.get("selection_uses_identity_and_source_group_only") is not True
        or any(
            contract.get(key) is not False
            for key in (
                "downstream_outcomes_read",
                "metric_ids_read",
                "model_prediction_fields_read",
                "truth_fields_read",
            )
        )
        or (
            not legacy_identity_sha
            and (
                not modern_freeze_sha
                or sha256_file(freeze_path) != modern_freeze_sha
            )
        )
    ):
        raise ValueError("GEPA role freeze does not bind the clean source pack")
    return {
        "path": str(freeze_path),
        "sha256": sha256_file(freeze_path),
        "role": requested_role,
        "identity_sha256": expected_identity_sha,
        "selected_count": int(freeze["selected_count"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument(
        "--label-pass",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Validated label pass, in chronological order; repeat at least twice.",
    )
    parser.add_argument(
        "--pass-pack",
        action="append",
        default=[],
        metavar="NAME=ROOT",
        help="Truth-hidden pack used by the correspondingly named label pass.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--unresolved-output", required=True)
    parser.add_argument("--disagreements-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--gepa-role",
        choices=(
            "evaluation",
            "optimize",
            "select",
            "adjudicator_dev",
            "verifier_dev",
        ),
        default="evaluation",
        help=(
            "Declare whether resolved truth is evaluation-only, may mutate prompts "
            "(optimize), or may only choose among already frozen prompts (select)."
        ),
    )
    parser.add_argument(
        "--gepa-role-freeze",
        help=(
            "Clean identity-only GEPA panel FREEZE.json. Required for optimize/select "
            "when the truth-hidden audit pack predates embedded usage-contract fields."
        ),
    )
    args = parser.parse_args()

    label_specs = _parse_specs(args.label_pass, "--label-pass")
    pack_specs = dict(_parse_specs(args.pass_pack, "--pass-pack"))
    names = [name for name, _ in label_specs]
    if len(label_specs) < 2:
        parser.error("at least two --label-pass values are required")
    if set(names) != set(pack_specs):
        raise ValueError("--label-pass and --pass-pack names must match exactly")

    source = Path(args.pack_root).resolve()
    output = Path(args.output).resolve()
    unresolved_output = Path(args.unresolved_output).resolve()
    disagreements_output = Path(args.disagreements_output).resolve()
    report_path = Path(args.report).resolve()
    if any(
        path.exists()
        for path in (output, unresolved_output, disagreements_output, report_path)
    ):
        raise FileExistsError("refusing to overwrite exact-consensus truth outputs")

    source_validation_path = source / "validation.json"
    source_validation = json.loads(source_validation_path.read_text(encoding="utf-8"))
    role_freeze_metadata = None
    panel_role = (
        args.gepa_role
        if args.gepa_role in {"adjudicator_dev", "verifier_dev"}
        else None
    )
    usage_role = "select" if panel_role is not None else args.gepa_role
    if panel_role is not None:
        if source_validation.get("gepa_role") != panel_role:
            raise ValueError("requested GEPA panel role differs from frozen source pack")
        if not args.gepa_role_freeze:
            raise ValueError(
                "adjudicator/verifier selection panels require --gepa-role-freeze"
            )
        role_freeze_metadata = _validate_role_freeze(
            Path(args.gepa_role_freeze).resolve(),
            source_validation,
            panel_role,
        )
    elif usage_role != "evaluation":
        embedded_role = source_validation.get("gepa_role")
        if embedded_role is not None:
            if embedded_role != usage_role:
                raise ValueError("requested GEPA role differs from frozen source pack")
            contract = source_validation.get("usage_contract") or {}
            expected_flag = (
                "optimize_may_mutate_prompts"
                if usage_role == "optimize"
                else "select_may_choose_only_predeclared_variants"
            )
            if contract.get(expected_flag) is not True:
                raise ValueError(f"frozen source pack does not authorize {usage_role}")
        else:
            if not args.gepa_role_freeze:
                raise ValueError(
                    "legacy truth-hidden pack requires --gepa-role-freeze for optimize/select"
                )
            role_freeze_metadata = _validate_role_freeze(
                Path(args.gepa_role_freeze).resolve(),
                source_validation,
                usage_role,
            )
    elif args.gepa_role_freeze:
        raise ValueError("--gepa-role-freeze is only valid for optimize/select")
    source_items_path, source_bank_path = source / "items.jsonl", source / "bank.json"
    if sha256_file(source_items_path) != source_validation["outputs"]["items"]["sha256"]:
        raise ValueError("source items hash mismatch")
    if sha256_file(source_bank_path) != source_validation["outputs"]["bank"]["sha256"]:
        raise ValueError("source bank hash mismatch")
    items = list(read_jsonl(source_items_path))
    item_by_uid = {str(row["norm_uid"]): row for row in items}
    if len(item_by_uid) != len(items):
        raise ValueError("source pack contains duplicate UIDs")
    source_uids = set(item_by_uid)
    task = str(source_validation["task"])
    bank_hash = str(source_validation["bank_source_sha256"])
    bank = json.loads(source_bank_path.read_text(encoding="utf-8"))
    bank_ids = {str(row["metric_id"]) for row in bank["metrics"]}

    labels: dict[str, dict[str, dict[str, Any]]] = {}
    pass_metadata: dict[str, dict[str, Any]] = {}
    validation_hashes: set[str] = set()
    label_hashes: set[str] = set()
    for name, label_path in label_specs:
        pack_root = pack_specs[name]
        validation_path = pack_root / "validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        items_path, bank_path = pack_root / "items.jsonl", pack_root / "bank.json"
        if validation.get("truth_hidden") is not True:
            raise ValueError(f"pass pack is not truth-hidden: {name}")
        if sha256_file(items_path) != validation["outputs"]["items"]["sha256"]:
            raise ValueError(f"pass-pack items hash mismatch: {name}")
        if sha256_file(bank_path) != validation["outputs"]["bank"]["sha256"]:
            raise ValueError(f"pass-pack bank hash mismatch: {name}")
        if (
            str(validation.get("task")) != task
            or str(validation.get("bank_source_sha256")) != bank_hash
        ):
            raise ValueError(f"pass-pack task/bank mismatch: {name}")
        pass_items = _index(items_path)
        if not set(pass_items).issubset(source_uids):
            raise ValueError(f"pass pack contains UIDs outside source: {name}")
        pass_labels = _index(label_path)
        if set(pass_labels) != set(pass_items):
            raise ValueError(f"labels do not exactly cover their truth-hidden pack: {name}")
        for uid, row in pass_labels.items():
            decision, metric_id = _decision_key(row)
            confidence = str(row.get("confidence") or "").lower()
            if (
                str(row.get("task")) != task
                or str(row.get("current_bank_source_sha256")) != bank_hash
                or decision not in DECISIONS
                or confidence not in CONFIDENCES
            ):
                raise ValueError(f"invalid task/bank/decision/confidence in {name}: {uid}")
            if decision == "MATCH" and metric_id not in bank_ids:
                raise ValueError(f"MATCH leaf absent from current bank in {name}: {uid}")
            if decision != "MATCH" and row.get("metric_id") is not None:
                raise ValueError(f"non-MATCH carries a metric ID in {name}: {uid}")
        label_sha = sha256_file(label_path)
        validation_sha = sha256_file(validation_path)
        if label_sha in label_hashes or validation_sha in validation_hashes:
            raise ValueError("label passes must use distinct labels and truth-hidden packs")
        label_hashes.add(label_sha)
        validation_hashes.add(validation_sha)
        labels[name] = pass_labels
        pass_metadata[name] = {
            "labels": {"path": str(label_path), "sha256": label_sha},
            "pack_validation": {
                "path": str(validation_path),
                "sha256": validation_sha,
            },
            "pack_items_sha256": sha256_file(items_path),
            "pack_bank_sha256": sha256_file(bank_path),
            "count": len(pass_labels),
        }

    if set(labels[names[0]]) != source_uids or set(labels[names[1]]) != source_uids:
        raise ValueError("the first two independent passes must each cover the source pack")

    votes: dict[str, list[tuple[str, tuple[str, str | None], dict[str, Any]]]] = {
        uid: [] for uid in source_uids
    }
    rounds: list[dict[str, Any]] = []
    current_unresolved = set(source_uids)
    for index, name in enumerate(names):
        observed = set(labels[name])
        expected = source_uids if index < 2 else current_unresolved
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                f"pass {name} must cover exactly the rows unresolved before it; "
                f"missing={missing[:3]} extra={extra[:3]}"
            )
        before = len(current_unresolved)
        for uid, row in labels[name].items():
            votes[uid].append((name, _decision_key(row), row))
        current_unresolved = {
            uid
            for uid in source_uids
            if _winner([key for _, key, _ in votes[uid]]) is None
        }
        rounds.append(
            {
                "pass": name,
                "ordinal": index + 1,
                "labeled_count": len(observed),
                "unresolved_before": before,
                "newly_resolved": before - len(current_unresolved),
                "unresolved_after": len(current_unresolved),
                **pass_metadata[name],
            }
        )

    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    supporter_patterns: Counter[str] = Counter()
    first_labels, second_labels = labels[names[0]], labels[names[1]]
    initial_decision_agreement = sum(
        first_labels[uid].get("decision") == second_labels[uid].get("decision")
        for uid in source_uids
    )
    initial_exact_agreement = sum(
        _decision_key(first_labels[uid]) == _decision_key(second_labels[uid])
        for uid in source_uids
    )
    initial_both_match = [
        uid
        for uid in source_uids
        if first_labels[uid].get("decision") == "MATCH"
        and second_labels[uid].get("decision") == "MATCH"
    ]
    initial_match_leaf_agreement = sum(
        first_labels[uid].get("metric_id") == second_labels[uid].get("metric_id")
        for uid in initial_both_match
    )
    for item in items:
        uid = str(item["norm_uid"])
        available = votes[uid]
        keys = [key for _, key, _ in available]
        winner = _winner(keys)
        provenance = {name: _compact(row) for name, _, row in available}
        distinct = len(set(keys)) > 1
        if distinct:
            disagreements.append(
                {
                    "norm_uid": uid,
                    "task": task,
                    "corpus": item.get("corpus"),
                    "row": item.get("row"),
                    "source_group": item.get("source_group") or item.get("split_group"),
                    "source_predictions": provenance,
                }
            )
        if winner is None:
            unresolved.append(
                {
                    "norm_uid": uid,
                    "task": task,
                    "corpus": item.get("corpus"),
                    "row": item.get("row"),
                    "source_group": item.get("source_group") or item.get("split_group"),
                    "unresolved_reason": "no unique exact decision-and-leaf agreement from two independent passes",
                    "source_predictions": provenance,
                }
            )
            continue
        supporters = [name for name, key, _ in available if key == winner]
        supporter_rows = [row for _, key, row in available if key == winner]
        supporter_patterns["+".join(supporters)] += 1
        decision_counts[winner[0]] += 1
        confidence = (
            "high"
            if sum(str(row.get("confidence")) == "high" for row in supporter_rows) >= 2
            else "medium"
        )
        resolved.append(
            {
                "schema_version": "silver-match-v3-exact-multi-pass-resolved-truth-v1",
                "norm_uid": uid,
                "task": task,
                "corpus": item.get("corpus"),
                "row": item.get("row"),
                "source_group": item.get("source_group") or item.get("split_group"),
                "split": "train" if usage_role == "optimize" else "dev",
                "gepa_role": usage_role,
                "gepa_panel_role": panel_role,
                "decision": winner[0],
                "metric_id": winner[1],
                "confidence": confidence,
                "reason": "Exact decision and metric leaf agreed by at least two independently configured full-bank passes.",
                "current_bank_source_sha256": bank_hash,
                "label_source": "independent_exact_multi_pass_resolution",
                "evaluation_only": usage_role != "optimize",
                "training_eligible": False,
                "prompt_gradient_eligible": usage_role == "optimize",
                "prompt_selection_eligible": usage_role == "select",
                "agreement_sources": supporters,
                "source_predictions": provenance,
            }
        )

    write_jsonl(output, resolved)
    write_jsonl(unresolved_output, unresolved)
    write_jsonl(disagreements_output, disagreements)
    report = {
        "schema_version": "silver-match-v3-exact-multi-pass-truth-report-v1",
        "task": task,
        "gepa_role": usage_role,
        "gepa_panel_role": panel_role,
        "policy": (
            "first two passes cover all rows; each later pass covers exactly the "
            "remaining exact disagreements; resolve only with a unique decision-and-leaf "
            "key supported by at least two independent passes"
        ),
        "source_count": len(items),
        "resolved_count": len(resolved),
        "unresolved_count": len(unresolved),
        "complete": not unresolved,
        "resolved_decision_counts": dict(sorted(decision_counts.items())),
        "resolved_match_count": decision_counts["MATCH"],
        "resolved_typed_nonmatch_count": len(resolved) - decision_counts["MATCH"],
        "disagreement_count": len(disagreements),
        "initial_pair_agreement": {
            "first": names[0],
            "second": names[1],
            "decision_agreement_count": initial_decision_agreement,
            "decision_agreement_rate": initial_decision_agreement / len(source_uids),
            "exact_decision_and_leaf_agreement_count": initial_exact_agreement,
            "exact_decision_and_leaf_agreement_rate": initial_exact_agreement
            / len(source_uids),
            "both_match_count": len(initial_both_match),
            "leaf_agreement_when_both_match_count": initial_match_leaf_agreement,
            "leaf_agreement_when_both_match_rate": (
                initial_match_leaf_agreement / len(initial_both_match)
                if initial_both_match
                else None
            ),
        },
        "supporter_patterns": dict(sorted(supporter_patterns.items())),
        "rounds": rounds,
        "permanent_blind_rows_in_source": int(
            source_validation.get("permanent_blind_rows_in_source", 0)
        ),
        "inputs": {
            "source_pack_validation": {
                "path": str(source_validation_path),
                "sha256": sha256_file(source_validation_path),
            },
            "gepa_role_freeze": role_freeze_metadata,
            "passes": pass_metadata,
        },
        "outputs": {
            "resolved": {"path": str(output), "sha256": sha256_file(output)},
            "unresolved": {
                "path": str(unresolved_output),
                "sha256": sha256_file(unresolved_output),
            },
            "disagreements": {
                "path": str(disagreements_output),
                "sha256": sha256_file(disagreements_output),
            },
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
