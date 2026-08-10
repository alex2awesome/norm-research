import hashlib
import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.select_pr_verifier_gepa_candidate import (
    build_selection,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    names = ("leaf_contrast", "predicate_first", "proof_obligations")
    author_candidates = {}
    outputs = {}
    for name in names:
        prompt = tmp_path / f"{name}.txt"
        prompt.write_text(name, encoding="utf-8")
        author_candidates[name] = {
            "prompt": {"path": str(prompt), "sha256": _sha(prompt)}
        }
        outputs[name] = {}
        for order in ("original", "hashed"):
            output = tmp_path / name / f"{order}.jsonl"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("{}\n", encoding="utf-8")
            outputs[name][order] = {"path": str(output), "sha256": _sha(output)}
    author_path = tmp_path / "AUTHOR.json"
    _write(
        author_path,
        {
            "schema_version": "silver-match-v3-pr-gemma-verifier-gepa-author-output-v1",
            "status": "FROZEN_THREE_GEMMA4_CANDIDATES_OPTIMIZE_ONLY",
            "fresh_test_drawn_or_read": False,
            "candidates": author_candidates,
        },
    )
    freeze_path = tmp_path / "FREEZE.json"
    _write(
        freeze_path,
        {
            "inputs": {
                "author_output_freeze": {
                    "path": str(author_path),
                    "sha256": _sha(author_path),
                }
            },
            "contracts": {
                "optimize_only": True,
                "select_test_mi_outcomes_opened": False,
            },
            "orders": ["original", "hashed"],
            "paired_count": 92,
            "total_prompt_count": 552,
            "prompts": {name: author_candidates[name]["prompt"] for name in names},
            "model": "gemma4",
            "inference": {"max_alternatives": 15},
        },
    )
    scores = {
        "leaf_contrast": {
            "retained_precision": 0.83,
            "retained_recall_of_correct_proposals": 0.32,
            "wrong_proposal_rejection_rate": 0.93,
        },
        "predicate_first": {
            "retained_precision": 0.84,
            "retained_recall_of_correct_proposals": 0.34,
            "wrong_proposal_rejection_rate": 0.93,
        },
        "proof_obligations": {
            "retained_precision": 0.86,
            "retained_recall_of_correct_proposals": 0.41,
            "wrong_proposal_rejection_rate": 0.93,
        },
    }
    report_path = tmp_path / "REPORT.json"
    _write(
        report_path,
        {
            "schema_version": "silver-match-v3-pr-verifier-gepa-optimize92-batch-score-v1",
            "task": "press-releases",
            "role": "optimize",
            "status": "SCORED_ALL_CANDIDATES_OPTIMIZE_ONLY_NOT_SELECTED",
            "policy": "two_order_exact_high",
            "select_test_mi_outcomes_opened": False,
            "selection_performed": False,
            "scores": scores,
            "outputs": outputs,
            "input_freeze": {"path": str(freeze_path), "sha256": _sha(freeze_path)},
        },
    )
    return author_path, report_path


def test_freezes_dominant_candidate_for_fresh_audit(tmp_path: Path) -> None:
    author, report = _fixture(tmp_path)
    out = tmp_path / "selected"
    payload = build_selection(
        author_output_path=author,
        score_report_path=report,
        chosen_name="proof_obligations",
        judge_id="codex-isolated",
        rationale="Unique precision and recall leader with unchanged false-retain count.",
        output_dir=out,
    )
    assert payload["chosen"]["name"] == "proof_obligations"
    assert payload["status"].endswith("NOT_PRODUCTION_PROMOTED")
    assert payload["isolation"]["fresh_test_drawn_or_read"] is False
    assert _sha(out / "selected.prompt.txt") == payload["chosen"]["prompt"]["sha256"]


def test_rejects_dominated_judge_choice(tmp_path: Path) -> None:
    author, report = _fixture(tmp_path)
    with pytest.raises(ValueError, match="dominated"):
        build_selection(
            author_output_path=author,
            score_report_path=report,
            chosen_name="predicate_first",
            judge_id="codex-isolated",
            rationale="bad choice",
            output_dir=tmp_path / "selected",
        )


def test_rejects_test_exposure_flag(tmp_path: Path) -> None:
    author, report = _fixture(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["select_test_mi_outcomes_opened"] = True
    _write(report, payload)
    with pytest.raises(ValueError, match="optimize-only"):
        build_selection(
            author_output_path=author,
            score_report_path=report,
            chosen_name="proof_obligations",
            judge_id="codex-isolated",
            rationale="would otherwise win",
            output_dir=tmp_path / "selected",
        )
