#!/usr/bin/env python3
"""Finalize two-order Gemma verification into positives and contrast ledgers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .common import read_jsonl, sha256_file, write_jsonl


def index_unique(rows: Sequence[dict[str, Any]], label: str) -> dict[str, dict]:
    output = {str(row["norm_uid"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate UID in {label}")
    return output


def finalize(
    proposals: Sequence[dict[str, Any]],
    first: Sequence[dict[str, Any]],
    second: Sequence[dict[str, Any]],
    *,
    require_one_high: bool,
    retrieval_injected: dict[str, bool] | None = None,
    selected_prompt_sha256: str | None = None,
    requires_independent_audit: bool = True,
    calibration_power_status: str = "not_recorded",
) -> tuple[list[dict], list[dict], list[dict], dict[str, Any]]:
    proposal_by_uid = index_unique(proposals, "proposals")
    first_by_uid = index_unique(first, "first verification")
    second_by_uid = index_unique(second, "second verification")
    if proposal_by_uid.keys() != first_by_uid.keys() or proposal_by_uid.keys() != second_by_uid.keys():
        raise ValueError("proposal and verification UID sets differ")
    retained, contrasts, rejected = [], [], []
    counts: Counter[str] = Counter()
    metric_counts: Counter[str] = Counter()
    retrieval_injected = retrieval_injected or {}
    for uid in sorted(proposal_by_uid):
        proposal, left, right = proposal_by_uid[uid], first_by_uid[uid], second_by_uid[uid]
        proposed = str(proposal["metric_id"])
        for label, verification in (("first", left), ("second", right)):
            recorded_primary = verification.get("primary_metric_id")
            if recorded_primary is not None and str(recorded_primary) != proposed:
                raise ValueError(f"{label} verification primary mismatch for {uid}")
        orders = {left.get("order_mode"), right.get("order_mode")}
        orders.discard(None)
        if len(orders) != 2:
            raise ValueError(f"verification orders are not independent for {uid}: {orders}")
        prompt_hashes = {left.get("prompt_sha256"), right.get("prompt_sha256")}
        prompt_hashes.discard(None)
        if len(prompt_hashes) != 1:
            raise ValueError(f"verification prompt hashes differ for {uid}: {prompt_hashes}")
        if selected_prompt_sha256 and prompt_hashes != {selected_prompt_sha256}:
            raise ValueError(
                f"verification prompt was not the dev-selected prompt for {uid}"
            )
        if left.get("decision") == right.get("decision"):
            counts["decision_order_agreement"] += 1
        if (left.get("decision"), left.get("metric_id")) == (
            right.get("decision"),
            right.get("metric_id"),
        ):
            counts["decision_metric_order_agreement"] += 1
        counts[f"pair:{left.get('decision')}|{right.get('decision')}"] += 1
        was_injected = retrieval_injected.get(uid)
        retrieval_status = (
            "injected_for_verification"
            if was_injected is True
            else "natural_top_k"
            if was_injected is False
            else "not_recorded"
        )
        left_confirm = left.get("decision") == "CONFIRM_MATCH" and left.get("metric_id") == proposed
        right_confirm = right.get("decision") == "CONFIRM_MATCH" and right.get("metric_id") == proposed
        confidences = {str(left.get("confidence")), str(right.get("confidence"))}
        confidence_ok = confidences <= {"high", "medium"} and (
            not require_one_high or "high" in confidences
        )
        if left_confirm and right_confirm and confidence_ok:
            row = dict(proposal)
            row.update(
                {
                    "label_source": "sonnet_plus_gemma4_order_stable",
                    "supervision_strength": "strong_order_stable_distillation",
                    "verification_prompt_sha256": left.get("prompt_sha256"),
                    "verification_orders": sorted(
                        {str(left.get("order_mode")), str(right.get("order_mode"))}
                    ),
                    "verification_confidences": sorted(confidences),
                    "proposal_retrieval_status": retrieval_status,
                    "gradient_eligible": not requires_independent_audit,
                }
            )
            retained.append(row)
            counts["retained"] += 1
            metric_counts[f"retained:{proposed}"] += 1
            counts[f"retained_retrieval:{retrieval_status}"] += 1
            continue
        stable_better = (
            left.get("decision") == right.get("decision") == "BETTER_CANDIDATE"
            and left.get("metric_id")
            and left.get("metric_id") == right.get("metric_id")
        )
        if stable_better:
            contrasts.append(
                {
                    "norm_uid": uid,
                    "task": proposal["task"],
                    "corpus": proposal["corpus"],
                    "proposed_metric_id": proposed,
                    "preferred_metric_id": left["metric_id"],
                    "label_source": "gemma4_order_stable_contrast",
                    "proposal_retrieval_status": retrieval_status,
                    "first_reason": left.get("reason"),
                    "second_reason": right.get("reason"),
                }
            )
            counts["stable_better_contrast"] += 1
            metric_counts[f"stable_better_from:{proposed}"] += 1
            metric_counts[f"stable_better_to:{left['metric_id']}"] += 1
        else:
            counts["other_rejection"] += 1
        rejected.append(
            {
                "norm_uid": uid,
                "task": proposal["task"],
                "corpus": proposal["corpus"],
                "proposed_metric_id": proposed,
                "first_decision": left.get("decision"),
                "first_metric_id": left.get("metric_id"),
                "first_confidence": left.get("confidence"),
                "second_decision": right.get("decision"),
                "second_metric_id": right.get("metric_id"),
                "second_confidence": right.get("confidence"),
                "proposal_retrieval_status": retrieval_status,
            }
        )
        counts[f"rejected_retrieval:{retrieval_status}"] += 1
    report = {
        "count": len(proposals),
        "require_one_high": require_one_high,
        "counts": dict(sorted(counts.items())),
        "retained_fraction": len(retained) / len(proposals) if proposals else None,
        "retrieval_status_recorded": bool(retrieval_injected),
        "selected_prompt_sha256": selected_prompt_sha256,
        "calibration_power_status": calibration_power_status,
        "requires_independent_audit_before_gradient_use": requires_independent_audit,
        "order_stability": {
            "decision_agreement": counts["decision_order_agreement"] / len(proposals)
            if proposals
            else None,
            "decision_metric_agreement": counts["decision_metric_order_agreement"]
            / len(proposals)
            if proposals
            else None,
        },
        "metric_counts": dict(sorted(metric_counts.items())),
        "proposal_retrieval_counts": dict(
            sorted(
                Counter(
                    "injected_for_verification" if value else "natural_top_k"
                    for value in retrieval_injected.values()
                ).items()
            )
        ),
    }
    return retained, contrasts, rejected, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument(
        "--candidates",
        help=(
            "the compact verification slate; records whether the proposal was "
            "naturally retrieved or injected for judging"
        ),
    )
    parser.add_argument(
        "--selection-record",
        help="dev-only GEPA verifier selection record; enforces its chosen prompt hash",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--allow-both-medium", action="store_true")
    args = parser.parse_args()
    paths = {
        key: Path(getattr(args, key)).resolve()
        for key in ("proposals", "first", "second")
    }
    if args.candidates:
        paths["candidates"] = Path(args.candidates).resolve()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite verification finalizer: {output}")
    output.mkdir(parents=True, exist_ok=True)
    retrieval_injected = None
    if "candidates" in paths:
        candidate_rows = list(read_jsonl(paths["candidates"]))
        candidate_by_uid = index_unique(candidate_rows, "verification candidates")
        if candidate_by_uid.keys() != {
            str(row["norm_uid"]) for row in read_jsonl(paths["proposals"])
        }:
            raise ValueError("proposal and verification candidate UID sets differ")
        retrieval_injected = {}
        for uid, row in candidate_by_uid.items():
            values = list(row.get("candidates") or [])
            if not values:
                raise ValueError(f"empty verification candidate slate: {uid}")
            retrieval_injected[uid] = bool(
                row.get("primary_was_injected")
                or values[0].get("injected_primary")
            )
    selection_path = Path(args.selection_record).resolve() if args.selection_record else None
    selected_prompt_sha256 = None
    requires_independent_audit = True
    calibration_power_status = "not_recorded"
    if selection_path:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("selection_split") != "dev":
            raise ValueError("verifier selection record was not selected on dev")
        selected_prompt_sha256 = str(selection.get("chosen", {}).get("prompt_sha256") or "")
        if not selected_prompt_sha256:
            raise ValueError("selection record lacks chosen prompt_sha256")
        requires_independent_audit = bool(
            selection.get("requires_independent_audit_before_gradient_use", True)
        )
        calibration_power_status = str(
            selection.get("calibration_power_status") or "not_recorded"
        )
    retained, contrasts, rejected, report = finalize(
        list(read_jsonl(paths["proposals"])),
        list(read_jsonl(paths["first"])),
        list(read_jsonl(paths["second"])),
        require_one_high=not args.allow_both_medium,
        retrieval_injected=retrieval_injected,
        selected_prompt_sha256=selected_prompt_sha256,
        requires_independent_audit=requires_independent_audit,
        calibration_power_status=calibration_power_status,
    )
    outputs = {
        "retained": output / "retained_teachers.jsonl",
        "contrasts": output / "hard_contrasts.jsonl",
        "rejected": output / "rejected_proposals.jsonl",
    }
    write_jsonl(outputs["retained"], retained)
    write_jsonl(outputs["contrasts"], contrasts)
    write_jsonl(outputs["rejected"], rejected)
    report["input_hashes"] = {key: sha256_file(path) for key, path in paths.items()}
    if selection_path:
        report["selection_record"] = str(selection_path)
        report["selection_record_sha256"] = sha256_file(selection_path)
    report["output_hashes"] = {key: sha256_file(path) for key, path in outputs.items()}
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
