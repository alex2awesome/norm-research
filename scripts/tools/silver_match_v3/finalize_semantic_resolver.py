#!/usr/bin/env python3
"""Finalize exact two-source semantic truth and audit a strict consensus key."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl
from .score_verifier_calibration import safe_rate, wilson_interval


SOURCE_ORDER = ("semantic_pass1", "strict_three_pass", "resolver_pass2")


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    output = {str(row["norm_uid"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate norm_uid values: {path}")
    return output


def _decision_key(row: dict[str, Any]) -> tuple[str, str | None]:
    decision = str(row.get("decision") or "")
    return decision, str(row["metric_id"]) if decision == "MATCH" else None


def _render_key(key: tuple[str, str | None]) -> str:
    return key[0] + (f":{key[1]}" if key[1] is not None else "")


def _compact(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "decision": row.get("decision"),
        "metric_id": row.get("metric_id"),
        "confidence": row.get("confidence"),
        "reason": row.get("reason"),
        "label_source": row.get("label_source"),
        "annotator": row.get("annotator"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--semantic-labels", required=True)
    parser.add_argument("--strict-key", required=True)
    parser.add_argument("--resolver-pack-root", required=True)
    parser.add_argument("--resolver-labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--unresolved-output", required=True)
    parser.add_argument("--disagreements-output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    paths = {
        "semantic_labels": Path(args.semantic_labels).resolve(),
        "strict_key": Path(args.strict_key).resolve(),
        "resolver_labels": Path(args.resolver_labels).resolve(),
    }
    pack_root = Path(args.pack_root).resolve()
    resolver_pack_root = Path(args.resolver_pack_root).resolve()
    output = Path(args.output).resolve()
    unresolved_output = Path(args.unresolved_output).resolve()
    disagreements_output = Path(args.disagreements_output).resolve()
    report_path = Path(args.report).resolve()
    if any(path.exists() for path in (output, unresolved_output, disagreements_output, report_path)):
        raise FileExistsError("refusing to overwrite semantic resolver outputs")

    validation_path = pack_root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    items_path, bank_path = pack_root / "items.jsonl", pack_root / "bank.json"
    if sha256_file(items_path) != validation["outputs"]["items"]["sha256"]:
        raise ValueError("source items hash mismatch")
    if sha256_file(bank_path) != validation["outputs"]["bank"]["sha256"]:
        raise ValueError("source bank hash mismatch")
    items = list(read_jsonl(items_path))
    item_by_uid = {str(row["norm_uid"]): row for row in items}
    if len(item_by_uid) != len(items):
        raise ValueError("source items contain duplicate UIDs")
    semantic, strict, resolver = (_index(paths[name]) for name in paths)
    if set(semantic) != set(item_by_uid):
        raise ValueError("semantic labels must cover source items exactly")
    if not set(strict).issubset(item_by_uid) or not set(resolver).issubset(item_by_uid):
        raise ValueError("strict/resolver labels contain UIDs outside source items")
    resolver_validation_path = resolver_pack_root / "validation.json"
    resolver_validation = json.loads(resolver_validation_path.read_text(encoding="utf-8"))
    resolver_items_path = resolver_pack_root / "items.jsonl"
    if sha256_file(resolver_items_path) != resolver_validation["outputs"]["items"]["sha256"]:
        raise ValueError("resolver pack items hash mismatch")
    resolver_uids = {str(row["norm_uid"]) for row in read_jsonl(resolver_items_path)}
    if set(resolver) != resolver_uids:
        raise ValueError("resolver labels do not exactly cover the hidden resolver pack")

    task = str(validation["task"])
    bank_hash = str(validation["bank_source_sha256"])
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    bank_by_id = {str(row["metric_id"]): row for row in bank["metrics"]}
    for name, rows in (("semantic", semantic), ("strict", strict), ("resolver", resolver)):
        for uid, row in rows.items():
            if row.get("task") != task or row.get("current_bank_source_sha256") != bank_hash:
                raise ValueError(f"{name} task/bank mismatch: {uid}")

    input_hashes = {name: sha256_file(path) for name, path in paths.items()}
    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    resolved_decisions: Counter[str] = Counter()
    unresolved_patterns: Counter[str] = Counter()
    disagreement_patterns: Counter[str] = Counter()
    agreement_source_patterns: Counter[str] = Counter()
    strict_stats: Counter[str] = Counter()

    for item in items:
        uid = str(item["norm_uid"])
        source_rows = {
            "semantic_pass1": semantic[uid],
            "strict_three_pass": strict.get(uid),
            "resolver_pass2": resolver.get(uid),
        }
        available = {name: row for name, row in source_rows.items() if row is not None}
        vote_counts = Counter(_decision_key(row) for row in available.values())
        winning = [key for key, count in vote_counts.items() if count >= 2]
        if len(winning) > 1:
            raise AssertionError(f"multiple exact majorities are impossible: {uid}")
        rendered_pattern = "|".join(
            f"{name}={_render_key(_decision_key(row))}"
            for name, row in available.items()
        )
        distinct = len(vote_counts) > 1
        if distinct:
            disagreement_patterns[rendered_pattern] += 1
        provenance = {
            name: _compact(source_rows[name]) for name in SOURCE_ORDER
        }
        cards = {
            name: (
                bank_by_id.get(str(row.get("metric_id")))
                if row is not None and row.get("decision") == "MATCH"
                else None
            )
            for name, row in source_rows.items()
        }
        detail = {
            "norm_uid": uid,
            "task": task,
            "corpus": item.get("corpus"),
            "row": item.get("row"),
            "source_group": item.get("source_group") or item.get("split_group"),
            "norm": item.get("norm"),
            "context": item.get("context"),
            "source_predictions": provenance,
            "source_metric_cards": cards,
            "pattern": rendered_pattern,
        }
        if distinct:
            disagreements.append(detail)

        if not winning:
            availability = "+".join(available)
            unresolved_patterns[f"{availability}:{len(vote_counts)}_distinct"] += 1
            unresolved.append(
                {
                    **detail,
                    "unresolved_reason": "no exact decision-and-leaf agreement from two independent sources",
                }
            )
        else:
            key = winning[0]
            supporters = [
                name for name, row in available.items() if _decision_key(row) == key
            ]
            dissenters = [name for name in available if name not in supporters]
            high_support = sum(
                available[name].get("confidence") == "high" for name in supporters
            )
            confidence = "high" if high_support >= 2 else "medium"
            agreement_source_patterns["+".join(supporters)] += 1
            resolved_decisions[key[0]] += 1
            resolved.append(
                {
                    "schema_version": "silver-match-v3-semantic-resolved-truth-v1",
                    "norm_uid": uid,
                    "task": task,
                    "corpus": item.get("corpus"),
                    "row": item.get("row"),
                    "source_group": item.get("source_group") or item.get("split_group"),
                    "split": "dev",
                    "decision": key[0],
                    "metric_id": key[1],
                    "confidence": confidence,
                    "reason": (
                        "Exact decision and leaf independently agreed by "
                        + ", ".join(supporters)
                        + (f"; dissent from {', '.join(dissenters)}." if dissenters else ".")
                    ),
                    "current_bank_source_sha256": bank_hash,
                    "label_source": "independent_semantic_exact_two_source_resolution",
                    "evaluation_only": True,
                    "training_eligible": False,
                    "agreement_sources": supporters,
                    "dissenting_sources": dissenters,
                    "source_predictions": provenance,
                    "input_label_sha256": input_hashes,
                }
            )

        if uid in strict:
            strict_stats["n"] += 1
            semantic_agrees = _decision_key(semantic[uid]) == _decision_key(strict[uid])
            strict_stats["semantic_exact_agreement"] += int(semantic_agrees)
            strict_stats["semantic_decision_agreement"] += int(
                semantic[uid].get("decision") == strict[uid].get("decision")
            )
            resolver_row = resolver.get(uid)
            if resolver_row is not None:
                strict_stats["resolver_available"] += 1
                strict_stats["resolver_exact_agreement"] += int(
                    _decision_key(resolver_row) == _decision_key(strict[uid])
                )
                if not semantic_agrees:
                    strict_stats["correction_opportunities"] += 1
                    strict_stats["resolver_supports_strict_correction"] += int(
                        _decision_key(resolver_row) == _decision_key(strict[uid])
                    )
                    strict_stats["resolver_supports_semantic_challenge"] += int(
                        _decision_key(resolver_row) == _decision_key(semantic[uid])
                    )

    write_jsonl(output, resolved)
    write_jsonl(unresolved_output, unresolved)
    write_jsonl(disagreements_output, disagreements)
    strict_n = strict_stats["n"]
    opportunities = strict_stats["correction_opportunities"]
    report = {
        "schema_version": "silver-match-v3-semantic-resolver-finalization-v1",
        "task": task,
        "source_count": len(items),
        "resolved_count": len(resolved),
        "resolved_coverage": safe_rate(len(resolved), len(items)),
        "resolved_decision_counts": dict(sorted(resolved_decisions.items())),
        "resolved_match_count": resolved_decisions["MATCH"],
        "resolved_typed_nonmatch_count": len(resolved) - resolved_decisions["MATCH"],
        "exact_leaf_coverage_of_source": safe_rate(resolved_decisions["MATCH"], len(items)),
        "unresolved_count": len(unresolved),
        "unresolved_patterns": dict(sorted(unresolved_patterns.items())),
        "disagreement_count": len(disagreements),
        "disagreement_patterns": dict(
            sorted(disagreement_patterns.items(), key=lambda value: (-value[1], value[0]))
        ),
        "agreement_source_patterns": dict(sorted(agreement_source_patterns.items())),
        "strict78_audit": {
            "n": strict_n,
            "semantic_exact_agreement": strict_stats["semantic_exact_agreement"],
            "semantic_exact_agreement_rate": safe_rate(
                strict_stats["semantic_exact_agreement"], strict_n
            ),
            "semantic_exact_agreement_wilson_95": wilson_interval(
                strict_stats["semantic_exact_agreement"], strict_n
            ),
            "semantic_decision_agreement_rate": safe_rate(
                strict_stats["semantic_decision_agreement"], strict_n
            ),
            "resolver_available": strict_stats["resolver_available"],
            "resolver_exact_agreement_rate_when_available": safe_rate(
                strict_stats["resolver_exact_agreement"],
                strict_stats["resolver_available"],
            ),
            "correction_opportunities": opportunities,
            "resolver_supports_strict_correction": strict_stats[
                "resolver_supports_strict_correction"
            ],
            "strict_correction_precision": safe_rate(
                strict_stats["resolver_supports_strict_correction"], opportunities
            ),
            "resolver_supports_semantic_challenge": strict_stats[
                "resolver_supports_semantic_challenge"
            ],
            "semantic_challenge_precision": safe_rate(
                strict_stats["resolver_supports_semantic_challenge"], opportunities
            ),
        },
        "permanent_blind_rows_in_source": int(
            validation.get("permanent_blind_rows_in_source", 0)
        ),
        "inputs": {
            "source_pack_validation": {
                "path": str(validation_path),
                "sha256": sha256_file(validation_path),
            },
            "resolver_pack_validation": {
                "path": str(resolver_validation_path),
                "sha256": sha256_file(resolver_validation_path),
            },
            **{
                name: {"path": str(path), "sha256": input_hashes[name]}
                for name, path in paths.items()
            },
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
