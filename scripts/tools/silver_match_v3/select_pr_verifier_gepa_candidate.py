#!/usr/bin/env python3
"""Freeze the Codex-judged PR verifier candidate before any fresh audit draw."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any


AUTHOR_SCHEMA = "silver-match-v3-pr-gemma-verifier-gepa-author-output-v1"
SCORE_SCHEMA = "silver-match-v3-pr-verifier-gepa-optimize92-batch-score-v1"
EXPECTED_VARIANTS = {"leaf_contrast", "predicate_first", "proof_obligations"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_selection(
    *,
    author_output_path: Path,
    score_report_path: Path,
    chosen_name: str,
    judge_id: str,
    rationale: str,
    output_dir: Path,
) -> dict[str, Any]:
    author_output_path = author_output_path.resolve()
    score_report_path = score_report_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    author = _load_json(author_output_path)
    report = _load_json(score_report_path)
    if (
        author.get("schema_version") != AUTHOR_SCHEMA
        or author.get("status") != "FROZEN_THREE_GEMMA4_CANDIDATES_OPTIMIZE_ONLY"
        or author.get("fresh_test_drawn_or_read") is not False
        or set(author.get("candidates") or {}) != EXPECTED_VARIANTS
    ):
        raise ValueError("author output is not the frozen optimize-only three-candidate universe")
    if (
        report.get("schema_version") != SCORE_SCHEMA
        or report.get("task") != "press-releases"
        or report.get("role") != "optimize"
        or report.get("status") != "SCORED_ALL_CANDIDATES_OPTIMIZE_ONLY_NOT_SELECTED"
        or report.get("policy") != "two_order_exact_high"
        or report.get("select_test_mi_outcomes_opened") is not False
        or report.get("selection_performed") is not False
        or set(report.get("scores") or {}) != EXPECTED_VARIANTS
        or set(report.get("outputs") or {}) != EXPECTED_VARIANTS
    ):
        raise ValueError("score report is not the complete, unselected optimize-only comparison")
    score_freeze_ref = report.get("input_freeze") or {}
    score_freeze_path = Path(str(score_freeze_ref.get("path") or "")).resolve()
    if (
        not score_freeze_path.is_file()
        or sha256_file(score_freeze_path) != score_freeze_ref.get("sha256")
    ):
        raise ValueError("scoring input freeze is missing or hash-drifted")
    score_freeze = _load_json(score_freeze_path)
    author_ref = (score_freeze.get("inputs") or {}).get("author_output_freeze") or {}
    if (
        Path(str(author_ref.get("path") or "")).resolve() != author_output_path
        or author_ref.get("sha256") != sha256_file(author_output_path)
        or (score_freeze.get("contracts") or {}).get("optimize_only") is not True
        or (score_freeze.get("contracts") or {}).get("select_test_mi_outcomes_opened")
        is not False
        or score_freeze.get("orders") != ["original", "hashed"]
        or int(score_freeze.get("paired_count") or 0) != 92
        or int(score_freeze.get("total_prompt_count") or 0) != 552
    ):
        raise ValueError("scoring freeze does not bind the optimize-only author universe")
    for name in sorted(EXPECTED_VARIANTS):
        prompt_ref = (author["candidates"][name] or {}).get("prompt") or {}
        prompt_path = Path(str(prompt_ref.get("path") or "")).resolve()
        if (
            not prompt_path.is_file()
            or sha256_file(prompt_path) != prompt_ref.get("sha256")
            or (score_freeze.get("prompts") or {}).get(name) != prompt_ref
        ):
            raise ValueError(f"prompt is missing, drifted, or not scoring-bound: {name}")
        for order in ("original", "hashed"):
            ref = ((report.get("outputs") or {}).get(name) or {}).get(order) or {}
            path = Path(str(ref.get("path") or "")).resolve()
            if not path.is_file() or sha256_file(path) != ref.get("sha256"):
                raise ValueError(f"scored output is missing or hash-drifted: {name}/{order}")
    if chosen_name not in EXPECTED_VARIANTS:
        raise ValueError("chosen candidate is outside the frozen universe")
    scores = report["scores"]
    chosen_score = scores[chosen_name]
    # The isolated Codex judgment is accepted only when its choice is not
    # dominated on the deployed precision-first objective. This prevents a
    # free-form judge record from selecting a demonstrably weaker candidate.
    chosen_key = (
        float(chosen_score["retained_precision"]),
        float(chosen_score["retained_recall_of_correct_proposals"]),
        float(chosen_score["wrong_proposal_rejection_rate"]),
    )
    better = {
        name: (
            float(score["retained_precision"]),
            float(score["retained_recall_of_correct_proposals"]),
            float(score["wrong_proposal_rejection_rate"]),
        )
        for name, score in scores.items()
        if (
            float(score["retained_precision"]),
            float(score["retained_recall_of_correct_proposals"]),
            float(score["wrong_proposal_rejection_rate"]),
        )
        > chosen_key
    }
    if better:
        raise ValueError(f"Codex choice is dominated under the frozen objective: {better}")
    prompt_source = Path(author["candidates"][chosen_name]["prompt"]["path"]).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        selected_prompt = stage / "selected.prompt.txt"
        shutil.copyfile(prompt_source, selected_prompt)
        if sha256_file(selected_prompt) != sha256_file(prompt_source):
            raise ValueError("selected prompt copy differs from frozen source")
        payload = {
            "schema_version": "silver-match-v3-pr-verifier-gepa-codex-selection-v1",
            "task": "press-releases",
            "status": "SELECTED_FOR_FRESH_INDEPENDENT_AUDIT_NOT_PRODUCTION_PROMOTED",
            "selection_role": "optimize_gepa_codex_judge",
            "selection_split": "optimize",
            "objective": (
                "precision-first exact-leaf retention; maximize retained precision, "
                "then recall and wrong-proposal rejection"
            ),
            "chosen": {
                "name": chosen_name,
                "prompt": {
                    "path": str((output_dir / selected_prompt.name).resolve()),
                    "sha256": sha256_file(selected_prompt),
                },
                "source_prompt": _artifact(prompt_source),
                "optimize_metrics": chosen_score,
            },
            "judge": {
                "identity": judge_id,
                "rationale": rationale,
                "determination": (
                    "The chosen design uniquely leads retained precision and recall, "
                    "while adding no false retains versus either alternative."
                ),
            },
            "candidates": {
                name: {
                    "prompt": author["candidates"][name]["prompt"],
                    "optimize_metrics": scores[name],
                }
                for name in sorted(EXPECTED_VARIANTS)
            },
            "validated_topology": {
                "model": score_freeze["model"],
                "proposal_count": 1,
                "maximum_strongest_alternatives": int(
                    score_freeze["inference"]["max_alternatives"]
                ),
                "orders": ["original", "hashed"],
                "retention_policy": "two_order_exact_high",
                "retain_iff": (
                    "both orders return CONFIRM_MATCH/high with the same proposed metric_id"
                ),
                "typed_abstentions": [
                    "AMBIGUOUS_MATCH",
                    "BETTER_CANDIDATE",
                    "NO_EXPLICIT_CRITERION",
                    "CONTEXT_NEEDED",
                    "GENERIC_VERDICT",
                    "NO_CANDIDATE_FITS",
                    "NOISE",
                ],
                "inference": score_freeze["inference"],
                "row_schema": "silver-match-v3.0",
                "required_output_fields": [
                    "decision",
                    "metric_id",
                    "confidence",
                    "reason",
                ],
            },
            "inputs": {
                "author_output": _artifact(author_output_path),
                "score_report": _artifact(score_report_path),
                "scoring_input_freeze": _artifact(score_freeze_path),
            },
            "isolation": {
                "fresh_test_drawn_or_read": False,
                "select_or_blind_audit_opened": False,
                "mi_or_outcomes_opened": False,
                "production_promotion_allowed_before_fresh_audit": False,
                "prompt_iteration_after_this_freeze_allowed": False,
            },
        }
        selection = stage / "SELECTION.json"
        selection.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        stage.rename(output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-output", required=True)
    parser.add_argument("--score-report", required=True)
    parser.add_argument("--chosen", required=True)
    parser.add_argument("--judge-id", required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    payload = build_selection(
        author_output_path=Path(args.author_output),
        score_report_path=Path(args.score_report),
        chosen_name=args.chosen,
        judge_id=args.judge_id,
        rationale=args.rationale,
        output_dir=Path(args.output_dir),
    )
    output = Path(args.output_dir).resolve() / "SELECTION.json"
    print(json.dumps({**payload, "selection_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
