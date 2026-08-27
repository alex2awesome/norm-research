#!/usr/bin/env python3
"""Finalize blind full-bank labels from independent Codex and Gemma votes.

Gemma's two orderings count as one model vote.  A release label requires a
unique decision-key winner with at least two independent votes.  Rows without
that evidence remain unresolved for another truth-hidden Codex pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adjudicate_gemma import CONFIDENCES, DECISIONS
from .common import read_jsonl, sha256_file, write_jsonl


SCHEMA_VERSION = "silver-match-v3-full-bank-multi-vote-consensus-v1"
LABEL_SOURCE = "independent_full_bank_multi_vote_consensus"
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def _index(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    output = {str(row.get("norm_uid") or ""): row for row in rows}
    if "" in output or len(output) != len(rows):
        raise ValueError(f"{name} contains missing or duplicate norm_uid values")
    return output


def _decision_key(row: dict[str, Any], bank_ids: set[str], name: str) -> tuple[str, str | None]:
    decision = str(row.get("decision") or "")
    metric_id = row.get("metric_id")
    metric_id = None if metric_id is None else str(metric_id)
    confidence = str(row.get("confidence") or "")
    if decision not in DECISIONS or confidence not in CONFIDENCES:
        raise ValueError(f"invalid {name} decision/confidence")
    if decision == "MATCH":
        if metric_id not in bank_ids:
            raise ValueError(f"{name} MATCH metric is absent from the bank")
    elif metric_id is not None:
        raise ValueError(f"{name} abstention carries a metric")
    return decision, metric_id


def _validate_pack(pack_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], set[str]]:
    validation_path = pack_root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    items_path, bank_path = pack_root / "items.jsonl", pack_root / "bank.json"
    if sha256_file(items_path) != validation["outputs"]["items"]["sha256"]:
        raise ValueError("pack items hash mismatch")
    if sha256_file(bank_path) != validation["outputs"]["bank"]["sha256"]:
        raise ValueError("pack bank hash mismatch")
    if validation.get("truth_hidden") is not True:
        raise ValueError("label pack was not truth-hidden")
    if not (
        validation.get("system_key_excluded_from_label_pack") is True
        or validation.get("prior_decisions_and_metric_ids_hidden") is True
    ):
        raise ValueError("label pack did not hide prior decisions and metric IDs")
    items = _index(list(read_jsonl(items_path)), f"pack {pack_root}")
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    bank_ids = {str(row["metric_id"]) for row in bank["metrics"]}
    if len(bank_ids) != len(bank["metrics"]):
        raise ValueError("pack bank contains duplicate metric IDs")
    if str(bank.get("source_sha256") or "") != str(validation["bank_source_sha256"]):
        raise ValueError("pack bank identity mismatch")
    return validation, items, bank_ids


def _validate_codex_pass(
    *,
    pack_root: Path,
    labels_path: Path,
    validation_path: Path,
    original_items: dict[str, dict[str, Any]],
    original_bank_ids: set[str],
    bank_sha: str,
) -> dict[str, dict[str, Any]]:
    pack_validation, pass_items, pass_bank_ids = _validate_pack(pack_root)
    if pass_bank_ids != original_bank_ids or str(pack_validation["bank_source_sha256"]) != bank_sha:
        raise ValueError("Codex pass used a different bank")
    if not set(pass_items) <= set(original_items):
        raise ValueError("Codex resolver pass contains UIDs outside the original pack")
    for uid, item in pass_items.items():
        original = original_items[uid]
        if any(item.get(field) != original.get(field) for field in ("task", "corpus", "row")):
            raise ValueError(f"Codex resolver item identity changed: {uid}")

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        validation.get("complete") is not True
        or int(validation.get("count", -1)) != len(pass_items)
        or (validation.get("output") or {}).get("sha256") != sha256_file(labels_path)
        or (validation.get("pack_validation") or {}).get("sha256")
        != sha256_file(pack_root / "validation.json")
        or str(validation.get("bank_source_sha256") or "") != bank_sha
    ):
        raise ValueError("Codex validation is incomplete or not bound to its pack/output")
    transcript = validation.get("transcript_audit") or {}
    transcript_status = str(transcript.get("status") or "")
    if transcript_status == "PASS":
        transcript_path = Path(str(transcript.get("path") or ""))
        if (
            not transcript_path.is_file()
            or sha256_file(transcript_path) != transcript.get("sha256")
        ):
            raise ValueError("Codex PASS transcript audit is missing or hash-drifted")
    elif transcript_status == "PASS_COMPOSITE_TRANSCRIPT_CLEAN":
        inputs = validation.get("inputs") or {}
        if (
            validation.get("schema_version")
            != "silver-match-v3-composite-transcript-clean-label-promotion-v1"
            or validation.get("selection_used_label_content") is not False
        ):
            raise ValueError("Codex composite transcript promotion lacks fail-closed provenance")
        for name in ("failed_source_audit", "repair_audit"):
            ref = inputs.get(name) or {}
            path = Path(str(ref.get("path") or ""))
            if not path.is_file() or sha256_file(path) != ref.get("sha256"):
                raise ValueError(f"Codex composite transcript input changed: {name}")
    else:
        raise ValueError("Codex labels lack a complete transcript-isolation audit")
    labels = _index(list(read_jsonl(labels_path)), f"Codex labels {labels_path}")
    if set(labels) != set(pass_items):
        raise ValueError("Codex labels do not exactly cover their frozen pack")
    for uid, row in labels.items():
        item = original_items[uid]
        _decision_key(row, original_bank_ids, f"Codex label {uid}")
        if (
            row.get("label_source") != "independent_codex_full_bank"
            or str(row.get("current_bank_source_sha256") or "") != bank_sha
            or any(row.get(field) != item.get(field) for field in ("task", "corpus", "row"))
        ):
            raise ValueError(f"Codex label lacks blind full-bank provenance: {uid}")
    return labels


def _combined_prompt_sha(freeze: dict[str, Any]) -> str:
    refs = [freeze["inputs"]["prompt"], freeze["inputs"]["prompt_addon"]]
    texts = []
    for ref in refs:
        path = Path(str(ref["path"]))
        if not path.is_absolute():
            path = Path.cwd() / path
        if sha256_file(path) != ref["sha256"]:
            raise ValueError(f"Gemma prompt component changed: {path}")
        texts.append(path.read_text(encoding="utf-8").rstrip())
    return hashlib.sha256(("\n\n".join(texts) + "\n").encode("utf-8")).hexdigest()


def _validate_gemma_inputs(
    *,
    pack_validation_path: Path,
    candidates_path: Path,
    candidate_freeze_path: Path,
    gemma_freeze_path: Path,
    retry_freeze_path: Path,
    original_path: Path,
    hashed_path: Path,
    items: dict[str, dict[str, Any]],
    bank_ids: set[str],
    bank_sha: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    candidate_freeze = json.loads(candidate_freeze_path.read_text(encoding="utf-8"))
    gemma_freeze = json.loads(gemma_freeze_path.read_text(encoding="utf-8"))
    retry = json.loads(retry_freeze_path.read_text(encoding="utf-8"))
    contract = gemma_freeze.get("scientific_contract") or {}
    if not (
        candidate_freeze.get("status") == "FROZEN_BEFORE_INFERENCE"
        and candidate_freeze.get("truth_hidden") is True
        and int(candidate_freeze.get("count", -1)) == len(items)
        and int(candidate_freeze.get("candidate_depth", -1)) == len(bank_ids)
        and ((candidate_freeze.get("inputs") or {}).get("pack_validation") or {}).get("sha256")
        == sha256_file(pack_validation_path)
        and (candidate_freeze.get("output") or {}).get("sha256") == sha256_file(candidates_path)
    ):
        raise ValueError("full-bank candidate freeze is invalid or unlinked")
    if not (
        gemma_freeze.get("status") == "FROZEN_BEFORE_INFERENCE"
        and contract.get("gemma_outputs_may_not_directly_enter_the_release") is True
        and contract.get("gemma_two_order_exact_consensus_counts_as_one_model_vote") is True
        and contract.get("promotion_requires_exact_agreement_with_an_independent_codex_full_bank_label_or_further_resolver") is True
        and contract.get("disagreements_remain_unresolved") is True
        and ((gemma_freeze.get("inputs") or {}).get("candidate_freeze") or {}).get("sha256")
        == sha256_file(candidate_freeze_path)
        and ((gemma_freeze.get("inputs") or {}).get("candidates") or {}).get("sha256")
        == sha256_file(candidates_path)
    ):
        raise ValueError("Gemma secondary-vote freeze is invalid or unlinked")
    if not (
        retry.get("status") == "FROZEN_BEFORE_RETRY_INFERENCE"
        and retry.get("unchanged_inputs_and_scientific_contract") is True
        and int((retry.get("retry_change") or {}).get("candidate_depth", -1)) == len(bank_ids)
        and Path(str((retry.get("outputs") or {}).get("original"))).name == original_path.name
        and Path(str((retry.get("outputs") or {}).get("hashed"))).name == hashed_path.name
    ):
        raise ValueError("Gemma retry freeze is invalid or points to other outputs")

    candidates = _index(list(read_jsonl(candidates_path)), "full-bank candidates")
    if set(candidates) != set(items):
        raise ValueError("full-bank candidates do not cover the label pack")
    for uid, row in candidates.items():
        ids = [str(value["metric_id"]) for value in row.get("candidates") or []]
        if len(ids) != len(bank_ids) or set(ids) != bank_ids:
            raise ValueError(f"candidate row is not the complete bank: {uid}")
        if str(row.get("bank_source_sha256") or "") != bank_sha:
            raise ValueError(f"candidate bank identity mismatch: {uid}")

    first = _index(list(read_jsonl(original_path)), "Gemma original")
    second = _index(list(read_jsonl(hashed_path)), "Gemma hashed")
    if set(first) != set(items) or set(second) != set(items):
        raise ValueError("Gemma outputs do not exactly cover the original pack")
    expected_model = str((gemma_freeze.get("runtime") or {}).get("model") or "")
    expected_prompt = _combined_prompt_sha(gemma_freeze)
    for output_path, order in ((original_path, "original"), (hashed_path, "hashed")):
        meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("output_sha256") != sha256_file(output_path)
            or meta.get("input_candidates_sha256") != sha256_file(candidates_path)
            or str(meta.get("model") or "") != expected_model
            or str(meta.get("prompt_sha256") or "") != expected_prompt
            or meta.get("order_mode") != order
        ):
            raise ValueError(f"Gemma completion metadata is invalid or unlinked: {order}")
    for uid in items:
        for row, order in ((first[uid], "original"), (second[uid], "hashed")):
            item = items[uid]
            if (
                row.get("order_mode") != order
                or str(row.get("model") or "") != expected_model
                or str(row.get("prompt_sha256") or "") != expected_prompt
                or str(row.get("candidate_bank_source_sha256") or "") != bank_sha
                or len(row.get("candidate_ids") or []) != len(bank_ids)
                or set(map(str, row.get("candidate_ids") or [])) != bank_ids
                or any(row.get(field) != item.get(field) for field in ("task", "corpus", "row"))
            ):
                raise ValueError(f"Gemma output provenance mismatch: {uid}/{order}")
            if row.get("decision") != "INVALID_OUTPUT":
                _decision_key(row, bank_ids, f"Gemma {order} {uid}")
    return first, second


def finalize_consensus(
    *,
    original_items: dict[str, dict[str, Any]],
    bank_ids: set[str],
    bank_sha: str,
    codex_passes: list[tuple[str, dict[str, dict[str, Any]], str]],
    gemma_original: dict[str, dict[str, Any]],
    gemma_hashed: dict[str, dict[str, Any]],
    min_gemma_confidence: str = "medium",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not codex_passes:
        raise ValueError("at least one independent Codex pass is required")
    threshold = CONFIDENCE_RANK[min_gemma_confidence]
    accepted: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    counts = Counter()
    for uid, item in original_items.items():
        votes: list[dict[str, Any]] = []
        for source, labels, source_sha in codex_passes:
            if uid not in labels:
                continue
            row = labels[uid]
            key = _decision_key(row, bank_ids, f"Codex vote {uid}")
            votes.append(
                {
                    "source": source,
                    "family": "codex",
                    "key": key,
                    "confidence": row["confidence"],
                    "input_sha256": source_sha,
                }
            )
        left, right = gemma_original[uid], gemma_hashed[uid]
        gemma_status = "order_disagreement"
        if left.get("decision") == "INVALID_OUTPUT" or right.get("decision") == "INVALID_OUTPUT":
            gemma_status = "invalid_output"
        else:
            left_key = _decision_key(left, bank_ids, f"Gemma original vote {uid}")
            right_key = _decision_key(right, bank_ids, f"Gemma hashed vote {uid}")
            if left_key == right_key:
                minimum = min(
                    CONFIDENCE_RANK[str(left["confidence"])],
                    CONFIDENCE_RANK[str(right["confidence"])],
                )
                if minimum >= threshold:
                    gemma_status = "exact_order_consensus_vote"
                    votes.append(
                        {
                            "source": "gemma4_gepa_two_order",
                            "family": "gemma4_gepa",
                            "key": left_key,
                            "confidence": min(
                                (str(left["confidence"]), str(right["confidence"])),
                                key=CONFIDENCE_RANK.get,
                            ),
                            "input_sha256": None,
                        }
                    )
                else:
                    gemma_status = "exact_order_consensus_below_confidence_floor"

        vote_counts = Counter(vote["key"] for vote in votes)
        if not vote_counts:
            reason = "no_eligible_independent_votes"
        else:
            best = max(vote_counts.values())
            winners = [key for key, value in vote_counts.items() if value == best]
            if best < 2:
                reason = "insufficient_exact_independent_votes"
            elif len(winners) != 1:
                reason = "independent_vote_tie"
            else:
                winner = winners[0]
                winning_votes = [vote for vote in votes if vote["key"] == winner]
                raw_confidence = min(
                    (str(vote["confidence"]) for vote in winning_votes),
                    key=CONFIDENCE_RANK.get,
                )
                accepted.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "norm_uid": uid,
                        "corpus": item["corpus"],
                        "task": item["task"],
                        "row": item["row"],
                        "decision": winner[0],
                        "metric_id": winner[1],
                        "confidence": raw_confidence,
                        "reason": "Unique exact decision-key consensus from at least two independent full-bank votes.",
                        "label_source": LABEL_SOURCE,
                        "current_bank_source_sha256": bank_sha,
                        "candidate_bank_source_sha256": bank_sha,
                        "candidate_ids": [winner[1]] if winner[0] == "MATCH" else [],
                        "consensus_vote_count": len(winning_votes),
                        "consensus_total_eligible_votes": len(votes),
                        "consensus_vote_sources": [vote["source"] for vote in winning_votes],
                        "consensus_vote_families": sorted({vote["family"] for vote in winning_votes}),
                        "source_confidences_preserved": {
                            vote["source"]: vote["confidence"] for vote in winning_votes
                        },
                        "gemma_vote_status": gemma_status,
                        "permanently_excluded_from_gradients": True,
                        "training_eligible_preverification": False,
                    }
                )
                counts[f"accepted:{winner[0]}"] += 1
                continue
        counts[f"unresolved:{reason}"] += 1
        counts[f"gemma:{gemma_status}"] += 1
        unresolved.append(
            {
                "schema_version": "silver-match-v3-full-bank-multi-vote-unresolved-v1",
                "norm_uid": uid,
                "corpus": item["corpus"],
                "task": item["task"],
                "row": item["row"],
                "unresolved_reason": reason,
                "bank_source_sha256": bank_sha,
            }
        )
    report = {
        "input_count": len(original_items),
        "accepted_count": len(accepted),
        "unresolved_count": len(unresolved),
        "complete": not unresolved,
        "min_gemma_confidence": min_gemma_confidence,
        "policy": {
            "gemma_original_and_hashed_count_as_one_vote_only_on_exact_decision_key_agreement": True,
            "minimum_independent_votes": 2,
            "winner_must_be_unique": True,
            "further_isolated_codex_passes_may_resolve_prior_disagreement_by_strict_vote_majority": True,
            "source_confidences_are_preserved_not_upgraded": True,
        },
        "counts": dict(sorted(counts.items())),
    }
    return accepted, unresolved, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--candidate-freeze", required=True)
    parser.add_argument("--gemma-freeze", required=True)
    parser.add_argument("--gemma-retry-freeze", required=True)
    parser.add_argument("--gemma-original", required=True)
    parser.add_argument("--gemma-hashed", required=True)
    parser.add_argument("--codex-pack-root", action="append", required=True)
    parser.add_argument("--codex-labels", action="append", required=True)
    parser.add_argument("--codex-validation", action="append", required=True)
    parser.add_argument("--min-gemma-confidence", choices=("medium", "high"), default="medium")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if not (
        len(args.codex_pack_root) == len(args.codex_labels) == len(args.codex_validation)
    ):
        parser.error("Codex pack, label, and validation arguments must have equal counts")

    pack_root = Path(args.pack_root).resolve()
    pack_validation, items, bank_ids = _validate_pack(pack_root)
    bank_sha = str(pack_validation["bank_source_sha256"])
    codex_passes = []
    seen_label_hashes = set()
    codex_input_meta = []
    for index, (pack, labels, validation) in enumerate(
        zip(args.codex_pack_root, args.codex_labels, args.codex_validation), 1
    ):
        pack_path, labels_path, validation_path = map(
            lambda value: Path(value).resolve(), (pack, labels, validation)
        )
        label_sha = sha256_file(labels_path)
        if label_sha in seen_label_hashes:
            raise ValueError("duplicate Codex pass supplied more than once")
        seen_label_hashes.add(label_sha)
        rows = _validate_codex_pass(
            pack_root=pack_path,
            labels_path=labels_path,
            validation_path=validation_path,
            original_items=items,
            original_bank_ids=bank_ids,
            bank_sha=bank_sha,
        )
        source = f"codex_isolated_pass_{index:02d}"
        codex_passes.append((source, rows, label_sha))
        codex_input_meta.append(
            {
                "source": source,
                "pack_validation_sha256": sha256_file(pack_path / "validation.json"),
                "labels_sha256": label_sha,
                "validation_sha256": sha256_file(validation_path),
                "count": len(rows),
            }
        )

    candidates_path = Path(args.candidates).resolve()
    candidate_freeze_path = Path(args.candidate_freeze).resolve()
    gemma_freeze_path = Path(args.gemma_freeze).resolve()
    retry_freeze_path = Path(args.gemma_retry_freeze).resolve()
    original_path = Path(args.gemma_original).resolve()
    hashed_path = Path(args.gemma_hashed).resolve()
    gemma_original, gemma_hashed = _validate_gemma_inputs(
        pack_validation_path=pack_root / "validation.json",
        candidates_path=candidates_path,
        candidate_freeze_path=candidate_freeze_path,
        gemma_freeze_path=gemma_freeze_path,
        retry_freeze_path=retry_freeze_path,
        original_path=original_path,
        hashed_path=hashed_path,
        items=items,
        bank_ids=bank_ids,
        bank_sha=bank_sha,
    )
    accepted, unresolved, report = finalize_consensus(
        original_items=items,
        bank_ids=bank_ids,
        bank_sha=bank_sha,
        codex_passes=codex_passes,
        gemma_original=gemma_original,
        gemma_hashed=gemma_hashed,
        min_gemma_confidence=args.min_gemma_confidence,
    )
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite consensus output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    labels_path = output_root / "labels.jsonl"
    unresolved_path = output_root / "unresolved.jsonl"
    write_jsonl(labels_path, accepted)
    write_jsonl(unresolved_path, unresolved)
    final_report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        **report,
        "count": len(accepted),
        "task": pack_validation["task"],
        "bank_source_sha256": bank_sha,
        "pack_validation": {
            "path": str(pack_root / "validation.json"),
            "sha256": sha256_file(pack_root / "validation.json"),
        },
        "inputs": {
            "candidates_sha256": sha256_file(candidates_path),
            "candidate_freeze_sha256": sha256_file(candidate_freeze_path),
            "gemma_freeze_sha256": sha256_file(gemma_freeze_path),
            "gemma_retry_freeze_sha256": sha256_file(retry_freeze_path),
            "gemma_original_sha256": sha256_file(original_path),
            "gemma_hashed_sha256": sha256_file(hashed_path),
            "codex_passes": codex_input_meta,
        },
        "output": {"path": str(labels_path), "sha256": sha256_file(labels_path)},
        "unresolved": {
            "path": str(unresolved_path),
            "sha256": sha256_file(unresolved_path),
            "count": len(unresolved),
        },
    }
    report_path = output_root / "validation.json"
    report_path.write_text(
        json.dumps(final_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(final_report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
