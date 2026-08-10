#!/usr/bin/env python3
"""Audit every explicit-role prompt-dev adjudicator error without new labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    output = {str(row.get("norm_uid") or ""): row for row in rows}
    if not rows or "" in output or len(output) != len(rows):
        raise ValueError(f"empty, missing, or duplicate norm_uid: {path}")
    return output


def _ref_path(ref: dict[str, Any], label: str) -> Path:
    path = Path(str(ref.get("path") or "")).resolve()
    if not path.is_file() or sha256_file(path) != str(ref.get("sha256") or ""):
        raise ValueError(f"{label} is missing or hash-drifted: {path}")
    return path


def _output_arg(command: dict[str, Any]) -> Path:
    argv = list(command.get("argv") or [])
    if argv.count("--output") != 1:
        raise ValueError("planned command must contain exactly one --output")
    return Path(argv[argv.index("--output") + 1]).resolve()


def _key(row: dict[str, Any]) -> tuple[str, str | None]:
    return str(row.get("decision") or ""), (
        str(row["metric_id"]) if row.get("metric_id") is not None else None
    )


def _category(
    truth: dict[str, Any], prediction: dict[str, Any], candidate_ids: set[str]
) -> str | None:
    if _key(truth) == _key(prediction):
        return None
    truth_decision = str(truth.get("decision") or "")
    predicted_decision = str(prediction.get("decision") or "")
    if truth_decision == "MATCH" and str(truth.get("metric_id")) not in candidate_ids:
        return "candidate_miss"
    if truth_decision == "MATCH_FAMILY_ONLY":
        return "family_only_boundary_error"
    if truth_decision == "MATCH":
        return (
            "wrong_leaf_candidate_present"
            if predicted_decision == "MATCH"
            else "false_abstain_on_exact_match"
        )
    return (
        f"false_match_on_{truth_decision.lower()}"
        if predicted_decision == "MATCH"
        else f"wrong_typed_abstention_truth_{truth_decision.lower()}"
    )


def audit(plan_path: Path, freeze_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan_path = plan_path.resolve()
    freeze_path = freeze_path.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    plan_sha = sha256_file(plan_path)
    task = str(plan.get("task") or "")
    if (
        plan.get("schema_version")
        != "silver-match-v3-explicit-role-task-local-gepa-plan-v1"
        or freeze.get("schema_version")
        != "silver-match-v3-explicit-role-task-local-gepa-freeze-v1"
        or freeze.get("task") != task
        or (freeze.get("command_plan") or {}).get("sha256") != plan_sha
    ):
        raise ValueError("plan/FREEZE contract is missing or hash-drifted")

    inputs = plan.get("inputs") or {}
    role = (inputs.get("explicit_roles") or {}).get("select") or {}
    truth_path = _ref_path(role.get("truth") or {}, "prompt-dev truth")
    candidates_path = _ref_path(role.get("candidates") or {}, "prompt-dev candidates")
    manifest_path = _ref_path(inputs.get("manifest") or {}, "manifest")
    bank_path = _ref_path(inputs.get("bank") or {}, "bank")
    truth = _index(truth_path)
    candidates = _index(candidates_path)
    if set(truth) != set(candidates):
        raise ValueError("prompt-dev truth and candidates have different UID coverage")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    norms: dict[str, dict[str, Any]] = {}
    norm_sources = []
    for corpus, meta in sorted((manifest.get("corpora") or {}).items()):
        if meta.get("task") != task:
            continue
        path = Path(str(meta["path"])).resolve()
        indexed = _index(path)
        overlap = set(norms) & set(indexed)
        if overlap:
            raise ValueError(f"duplicate task norm UID across corpora: {sorted(overlap)[:3]}")
        norms.update(indexed)
        norm_sources.append({"corpus": corpus, "path": str(path), "sha256": sha256_file(path)})
    if not set(truth) <= set(norms):
        raise ValueError("prompt-dev truth contains noncanonical task UIDs")
    bank_payload = json.loads(bank_path.read_text(encoding="utf-8"))
    cards = {str(row["metric_id"]): row for row in bank_payload.get("metrics") or []}

    outputs: dict[str, dict[str, Path]] = {}
    consensus: dict[str, Path] = {}
    scores: dict[str, Path] = {}
    for cell in plan.get("commands") or []:
        if cell.get("role") != "prompt_dev":
            continue
        variant = str(cell.get("variant") or "")
        if cell.get("stage") == "adjudicator":
            outputs.setdefault(variant, {})[str(cell["order"])] = _output_arg(
                cell["direct_batch_command"]
            )
        elif cell.get("stage") == "adjudicator_consensus":
            consensus[variant] = _output_arg(cell["command"])
        elif cell.get("stage") == "adjudicator_score":
            scores[variant] = _output_arg(cell["command"])
    declared = [str(row["name"]) for row in plan.get("adjudicator_variants") or []]
    if any(
        set(outputs.get(name) or {}) != {"original", "hashed"}
        or name not in consensus
        or name not in scores
        for name in declared
    ):
        raise ValueError("plan lacks a complete prompt-dev adjudicator audit cell")

    indexed_outputs: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    artifact_refs = []
    for variant in declared:
        indexed_outputs[variant] = {
            order: _index(path) for order, path in outputs[variant].items()
        }
        indexed_outputs[variant]["consensus"] = _index(consensus[variant])
        for label, path in [*outputs[variant].items(), ("consensus", consensus[variant]), ("score", scores[variant])]:
            if not path.is_file():
                raise ValueError(f"planned prompt-dev artifact is missing: {path}")
            artifact_refs.append(
                {"variant": variant, "kind": label, "path": str(path), "sha256": sha256_file(path)}
            )

    per_variant_order: dict[str, dict[str, Counter[str]]] = {
        name: {order: Counter() for order in ("original", "hashed", "consensus")}
        for name in declared
    }
    audit_rows = []
    for uid in sorted(truth):
        candidate_rows = list(candidates[uid].get("candidates") or [])
        candidate_ids = {str(row["metric_id"]) for row in candidate_rows}
        candidate_rank = {str(row["metric_id"]): int(row["rank"]) for row in candidate_rows}
        gold = truth[uid]
        views = {}
        any_error = False
        for variant in declared:
            variant_views = {}
            for order in ("original", "hashed", "consensus"):
                prediction = indexed_outputs[variant][order].get(uid)
                if prediction is None:
                    variant_views[order] = {"present": False, "error_category": "not_retained_in_consensus" if order == "consensus" else "missing_output"}
                    if order != "consensus":
                        any_error = True
                    continue
                category = _category(gold, prediction, candidate_ids)
                if category:
                    per_variant_order[variant][order][category] += 1
                    any_error = True
                predicted_id = (
                    str(prediction["metric_id"])
                    if prediction.get("metric_id") is not None
                    else None
                )
                variant_views[order] = {
                    "present": True,
                    "decision": prediction.get("decision"),
                    "metric_id": predicted_id,
                    "confidence": prediction.get("confidence"),
                    "reason": prediction.get("reason"),
                    "error_category": category,
                    "predicted_candidate_rank": candidate_rank.get(predicted_id),
                    "predicted_metric_card": cards.get(predicted_id),
                }
            views[variant] = variant_views
        if not any_error:
            continue
        gold_id = str(gold["metric_id"]) if gold.get("metric_id") is not None else None
        audit_rows.append(
            {
                "schema_version": "silver-match-v3-explicit-role-adjudicator-error-audit-v1",
                "task": task,
                "norm_uid": uid,
                "corpus": gold.get("corpus"),
                "canonical_norm": norms[uid],
                "truth": gold,
                "truth_metric_card": cards.get(gold_id),
                "truth_candidate_present": gold_id in candidate_ids if gold_id else None,
                "truth_candidate_rank": candidate_rank.get(gold_id),
                "candidate_ids": [str(row["metric_id"]) for row in candidate_rows],
                "variant_views": views,
                "audit_scope": "prompt_dev_evaluation_only_no_gradient",
            }
        )

    report = {
        "schema_version": "silver-match-v3-explicit-role-adjudicator-error-audit-report-v1",
        "task": task,
        "scope": {
            "role": "prompt_dev",
            "evaluation_only": True,
            "gradient_eligible": False,
            "test_or_blind_audit_used": False,
            "production_used": False,
            "outcomes_or_mi_used": False,
        },
        "panel_count": len(truth),
        "audit_uid_count": len(audit_rows),
        "per_variant_order_error_categories": {
            variant: {
                order: dict(sorted(counts.items()))
                for order, counts in orders.items()
            }
            for variant, orders in per_variant_order.items()
        },
        "inputs": {
            "command_plan": {"path": str(plan_path), "sha256": plan_sha},
            "freeze": {"path": str(freeze_path), "sha256": sha256_file(freeze_path)},
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "bank": {"path": str(bank_path), "sha256": sha256_file(bank_path)},
            "truth": {"path": str(truth_path), "sha256": sha256_file(truth_path)},
            "candidates": {"path": str(candidates_path), "sha256": sha256_file(candidates_path)},
            "norm_sources": norm_sources,
            "adjudicator_artifacts": artifact_refs,
        },
    }
    return audit_rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--freeze")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    plan = Path(args.plan).resolve()
    output = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite adjudicator error audit")
    rows, report = audit(
        plan,
        Path(args.freeze).resolve() if args.freeze else plan.with_name("FREEZE.json"),
    )
    write_jsonl(output, rows)
    report["output"] = {
        "path": str(output),
        "sha256": sha256_file(output),
        "count": len(rows),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "report_sha256": sha256_file(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
