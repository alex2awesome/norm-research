#!/usr/bin/env python3
"""Freeze a failed optimize adjudicator baseline and its exact confusion audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file
from .score_verifier_calibration import wilson_interval


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _named_paths(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--truth-audit must be NAME=PATH: {value!r}")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path or name in output:
            raise ValueError(f"invalid or duplicate truth audit: {value!r}")
        output[name] = Path(raw_path).resolve()
    return output


def _index(path: Path, kind: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = list(read_jsonl(path))
    indexed = {str(row.get("norm_uid") or ""): row for row in rows}
    if not rows or "" in indexed or len(indexed) != len(rows):
        raise ValueError(f"{kind} has missing or duplicate norm_uid values")
    return rows, indexed


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--truth-consensus-report", required=True)
    parser.add_argument(
        "--unresolved-exclusions",
        help=(
            "Exact unresolved ledger excluded after the allowed resolver rounds. "
            "Required when the consensus report is incomplete."
        ),
    )
    parser.add_argument("--truth-audit", action="append", default=[])
    parser.add_argument("--original", required=True)
    parser.add_argument("--hashed", required=True)
    parser.add_argument("--score", required=True)
    parser.add_argument("--false-abstention-original", required=True)
    parser.add_argument("--false-abstention-hashed", required=True)
    parser.add_argument("--minimum-exact-precision", type=float, default=0.90)
    parser.add_argument("--minimum-wilson-lower", type=float, default=0.80)
    parser.add_argument("--minimum-support", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = {
        "truth": Path(args.truth).resolve(),
        "truth_consensus_report": Path(args.truth_consensus_report).resolve(),
        "original": Path(args.original).resolve(),
        "hashed": Path(args.hashed).resolve(),
        "score": Path(args.score).resolve(),
        "false_abstention_original": Path(args.false_abstention_original).resolve(),
        "false_abstention_hashed": Path(args.false_abstention_hashed).resolve(),
    }
    if args.unresolved_exclusions:
        paths["unresolved_exclusions"] = Path(args.unresolved_exclusions).resolve()
    truth_audits = _named_paths(args.truth_audit)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not 0.0 <= args.minimum_exact_precision <= 1.0:
        raise ValueError("minimum exact precision must be in [0, 1]")
    if not 0.0 <= args.minimum_wilson_lower <= 1.0:
        raise ValueError("minimum Wilson lower bound must be in [0, 1]")
    if args.minimum_support < 1:
        raise ValueError("minimum support must be positive")

    truth_rows, truth = _index(paths["truth"], "truth")
    _, original = _index(paths["original"], "original predictions")
    _, hashed = _index(paths["hashed"], "hashed predictions")
    uids = set(truth)
    if set(original) != uids or set(hashed) != uids:
        raise ValueError("prediction universes do not exactly cover truth")

    report = json.loads(paths["truth_consensus_report"].read_text(encoding="utf-8"))
    report_resolved = (report.get("outputs") or {}).get("resolved") or {}
    report_unresolved = (report.get("outputs") or {}).get("unresolved") or {}
    unresolved_count = int(report.get("unresolved_count", -1))
    unresolved_rows: list[dict[str, Any]] = []
    if "unresolved_exclusions" in paths:
        unresolved_rows, unresolved_by_uid = _index(
            paths["unresolved_exclusions"], "unresolved exclusions"
        )
        overlap = set(unresolved_by_uid) & uids
        if overlap:
            raise ValueError(
                f"resolved truth overlaps unresolved exclusions: {sorted(overlap)[:3]}"
            )
    if (
        report.get("schema_version")
        != "silver-match-v3-exact-multi-pass-truth-report-v1"
        or int(report.get("resolved_count", -1)) != len(truth_rows)
        or report_resolved.get("sha256") != sha256_file(paths["truth"])
    ):
        raise ValueError("truth is not the exact resolved consensus bound by the report")
    if unresolved_count == 0:
        if (
            report.get("complete") is not True
            or unresolved_rows
            or report_unresolved.get("sha256")
            != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ):
            raise ValueError("complete truth report has inconsistent unresolved state")
    else:
        rounds = list(report.get("rounds") or [])
        if (
            report.get("complete") is not False
            or not unresolved_rows
            or unresolved_count != len(unresolved_rows)
            or report_unresolved.get("sha256")
            != sha256_file(paths["unresolved_exclusions"])
            or int(report.get("source_count", -1))
            != len(truth_rows) + len(unresolved_rows)
            or len(rounds) != 4
            or int((rounds[-1] or {}).get("unresolved_after", -1))
            != unresolved_count
        ):
            raise ValueError(
                "unresolved exclusions are not the exact residual ledger after two resolver rounds"
            )
    report_pass_metadata = ((report.get("inputs") or {}).get("passes") or {})
    report_passes = set(report_pass_metadata)
    if set(truth_audits) != report_passes:
        raise ValueError("truth audits must cover every exact-consensus pass")
    for name, path in truth_audits.items():
        audit = json.loads(path.read_text(encoding="utf-8"))
        pass_meta = report_pass_metadata[name]
        pack_validation_ref = pass_meta.get("pack_validation") or {}
        pack_validation_path = Path(str(pack_validation_ref.get("path") or "")).resolve()
        audit_pack_validation = audit.get("pack_validation") or {}
        if (
            audit.get("status") != "PASS"
            or audit.get("complete") is not True
            or audit.get("violations") != []
            or (
                audit.get("pass_key") is not None
                and audit.get("pass_key") != name
            )
            or Path(str(audit.get("pack_root") or "")).resolve()
            != pack_validation_path.parent
            or audit_pack_validation.get("sha256")
            != pack_validation_ref.get("sha256")
        ):
            raise ValueError(f"truth transcript audit is not a strict PASS: {name}")

    score = json.loads(paths["score"].read_text(encoding="utf-8"))
    score_inputs = score.get("inputs") or {}
    for name in ("truth", "original", "hashed"):
        if (score_inputs.get(name) or {}).get("sha256") != sha256_file(paths[name]):
            raise ValueError(f"score does not bind {name}")
    if (
        score.get("schema_version")
        != "silver-match-v3-optimize-two-order-adjudicator-score-v1"
        or score.get("role") != "optimize_prompt_gradient_only"
        or score.get("prompt_or_model_selection_performed") is not False
        or score.get("scientific_evaluation_claim_allowed") is not False
    ):
        raise ValueError("unexpected optimize baseline score contract")

    false_abstention = {}
    for name in ("original", "hashed"):
        path = paths[f"false_abstention_{name}"]
        audit = json.loads(path.read_text(encoding="utf-8"))
        if (
            audit.get("schema_version") != "silver-match-v3-false-abstention-audit-v1"
            or int(audit.get("joined_rows") or -1) != len(truth_rows)
            or list((audit.get("gold_inputs") or {}).values())
            != [sha256_file(paths["truth"])]
            or list((audit.get("prediction_inputs") or {}).values())
            != [sha256_file(paths[name])]
        ):
            raise ValueError(f"false-abstention audit does not bind {name}")
        exclusion_inputs = (audit.get("analysis_exclusions") or {}).get("inputs") or {}
        expected_exclusion_hashes = (
            [sha256_file(paths["unresolved_exclusions"])]
            if unresolved_rows
            else []
        )
        if (
            list(exclusion_inputs.values()) != expected_exclusion_hashes
            or int((audit.get("analysis_exclusions") or {}).get("count", -1))
            != len(unresolved_rows)
        ):
            raise ValueError(f"false-abstention audit exclusion ledger drift: {name}")
        false_abstention[name] = audit["overall"]

    decision_confusion: dict[str, Counter[str]] = {
        "original": Counter(),
        "hashed": Counter(),
    }
    by_truth_decision: dict[str, dict[str, Counter[str]]] = {
        "original": defaultdict(Counter),
        "hashed": defaultdict(Counter),
    }
    strict_terminal: Counter[str] = Counter()
    strict_by_truth: dict[str, Counter[str]] = defaultdict(Counter)
    order_exact_confusion: Counter[str] = Counter()
    by_metric: dict[str, Counter[str]] = defaultdict(Counter)

    for uid, gold in truth.items():
        gold_decision = str(gold["decision"])
        gold_metric = gold.get("metric_id")
        left, right = original[uid], hashed[uid]
        for name, prediction in (("original", left), ("hashed", right)):
            predicted_decision = str(prediction["decision"])
            predicted_metric = prediction.get("metric_id")
            decision_confusion[name][f"{gold_decision}->{predicted_decision}"] += 1
            bucket = by_truth_decision[name][gold_decision]
            bucket["count"] += 1
            bucket["predicted_match"] += predicted_decision == "MATCH"
            bucket["exact_label_correct"] += (
                predicted_decision,
                predicted_metric,
            ) == (gold_decision, gold_metric)
            if gold_decision == "MATCH":
                metric_bucket = by_metric[str(gold_metric)]
                metric_bucket["truth_match_count"] += name == "original"
                metric_bucket[f"{name}_predicted_match"] += predicted_decision == "MATCH"
                metric_bucket[f"{name}_exact_id_correct"] += (
                    predicted_decision == "MATCH" and predicted_metric == gold_metric
                )

        left_key = (str(left["decision"]), left.get("metric_id"))
        right_key = (str(right["decision"]), right.get("metric_id"))
        gold_key = (gold_decision, gold_metric)
        if left_key != right_key:
            terminal = "ORDER_DISAGREEMENT"
            order_exact_confusion[f"{left_key[0]}->{right_key[0]}"] += 1
        elif left_key[0] == "MATCH" and left_key == gold_key:
            terminal = "MATCH_EXACT_CORRECT"
        elif left_key[0] == "MATCH" and gold_decision == "MATCH":
            terminal = "MATCH_WRONG_LEAF"
        elif left_key[0] == "MATCH":
            terminal = "MATCH_FALSE_POSITIVE"
        elif left_key == gold_key:
            terminal = "TYPED_ABSTENTION_EXACT_CORRECT"
        else:
            terminal = "TYPED_ABSTENTION_WRONG"
        strict_terminal[terminal] += 1
        strict_by_truth[gold_decision][terminal] += 1
        if gold_decision == "MATCH" and left_key == right_key == gold_key:
            by_metric[str(gold_metric)]["strict_exact_id_correct"] += 1

    strict = (score.get("metrics") or {}).get("strict_consensus") or {}
    support = int(strict.get("confirmed_match_count") or 0)
    correct = int(strict.get("correct_exact_id_count") or 0)
    precision = strict.get("exact_id_precision")
    interval = wilson_interval(correct, support)
    gate_checks = {
        "minimum_support": support >= args.minimum_support,
        "minimum_exact_precision": precision is not None
        and float(precision) >= args.minimum_exact_precision,
        "minimum_wilson_lower": interval is not None
        and interval[0] >= args.minimum_wilson_lower,
    }
    if all(gate_checks.values()):
        raise ValueError("negative-gate recorder called on an eligible baseline")

    decision_counts = Counter(str(row["decision"]) for row in truth_rows)
    result = {
        "schema_version": "silver-match-v3-optimize-adjudicator-negative-gate-v1",
        "status": "FROZEN_COMPLETE_TRUTH_BASELINE_REJECTED_NO_PRODUCTION_LAUNCH",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": report.get("task"),
        "role": "optimize_prompt_gradient_evidence_only",
        "truth": {
            "count": len(truth_rows),
            "decision_counts": dict(sorted(decision_counts.items())),
            "match_rate": _rate(decision_counts["MATCH"], len(truth_rows)),
            "typed_nonmatch_rate": _rate(
                len(truth_rows) - decision_counts["MATCH"], len(truth_rows)
            ),
            "resolved_exact_consensus": True,
            "complete_exact_consensus": not unresolved_rows,
            "unresolved_count": len(unresolved_rows),
            "unresolved_rows_excluded_after_two_allowed_resolver_rounds": bool(
                unresolved_rows
            ),
            "source_count": len(truth_rows) + len(unresolved_rows),
        },
        "production_gate": {
            "thresholds": {
                "minimum_exact_id_precision": args.minimum_exact_precision,
                "minimum_exact_id_precision_wilson_95_lower": args.minimum_wilson_lower,
                "minimum_confirmed_match_support": args.minimum_support,
                "thresholds_lowered": False,
            },
            "observed": {
                "confirmed_match_support": support,
                "correct_exact_id_count": correct,
                "exact_id_precision": precision,
                "exact_id_precision_wilson_95": interval,
                "exact_id_recall_of_truth_matches": strict.get(
                    "exact_id_recall_of_truth_matches"
                ),
                "strict_typed_abstention_accuracy": strict.get(
                    "strict_typed_abstention_accuracy"
                ),
                "strict_exact_label_accuracy": strict.get("strict_exact_label_accuracy"),
            },
            "checks": gate_checks,
            "eligible": False,
        },
        "confusion_audit": {
            "decision_confusion_by_order": {
                name: dict(sorted(counts.items()))
                for name, counts in decision_confusion.items()
            },
            "per_truth_decision_by_order": {
                name: {
                    decision: dict(sorted(counts.items()))
                    for decision, counts in sorted(table.items())
                }
                for name, table in by_truth_decision.items()
            },
            "strict_terminal_counts": dict(sorted(strict_terminal.items())),
            "strict_terminal_rates": {
                key: value / len(truth_rows)
                for key, value in sorted(strict_terminal.items())
            },
            "strict_terminal_by_truth_decision": {
                decision: dict(sorted(counts.items()))
                for decision, counts in sorted(strict_by_truth.items())
            },
            "order_disagreement_decision_pairs": dict(
                sorted(order_exact_confusion.items())
            ),
            "truth_match_by_metric": {
                metric_id: {
                    **dict(sorted(counts.items())),
                    "strict_exact_recall": _rate(
                        counts["strict_exact_id_correct"], counts["truth_match_count"]
                    ),
                }
                for metric_id, counts in sorted(by_metric.items())
            },
            "false_abstention_audits": false_abstention,
        },
        "diagnosis": {
            "dominant_failure": "overmatching_and_exact_leaf_confusion",
            "single_order_match_counts": {
                "original": int(score["metrics"]["original"]["predicted_match_count"]),
                "hashed": int(score["metrics"]["hashed"]["predicted_match_count"]),
                "truth": decision_counts["MATCH"],
            },
            "production_launch_allowed": False,
            "prompt_gradient_evidence_allowed": True,
            "additional_peer_prompt_or_pipeline_iteration_performed": False,
        },
        "inputs": {
            **{name: _artifact(path) for name, path in sorted(paths.items())},
            "truth_transcript_audits": {
                name: _artifact(path) for name, path in sorted(truth_audits.items())
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {**result, "output": str(output), "sha256": sha256_file(output)},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
