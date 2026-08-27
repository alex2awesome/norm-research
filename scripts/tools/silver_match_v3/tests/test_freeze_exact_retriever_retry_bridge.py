import json
import sys
from pathlib import Path

from scripts.tools.silver_match_v3.common import write_jsonl
from scripts.tools.silver_match_v3.freeze_exact_retriever_retry_bridge import main


def _row(uid: str, role: str, decision: str = "MATCH") -> dict:
    return {
        "norm_uid": uid,
        "task": "demo",
        "corpus": "c",
        "split_group": f"c:source:{uid}",
        "current_bank_source_sha256": "bank",
        "decision": decision,
        "metric_id": "m1" if decision == "MATCH" else None,
        "gepa_role": role,
        "prompt_gradient_eligible": role == "optimize",
        "prompt_selection_eligible": role == "select",
        "training_eligible": False,
        "agreement_sources": ["A", "B"],
    }


def test_filters_prior_nontrain_and_excludes_select(tmp_path: Path, monkeypatch) -> None:
    prior = tmp_path / "prior.jsonl"
    optimize = tmp_path / "optimize.jsonl"
    select = tmp_path / "select.jsonl"
    roles = tmp_path / "roles.jsonl"
    output, report = tmp_path / "teachers.jsonl", tmp_path / "report.json"
    write_jsonl(
        prior,
        [
            {**_row("p_train", ""), "label_source": "independent_subagent"},
            {**_row("p_dev", ""), "label_source": "independent_subagent"},
        ],
    )
    write_jsonl(optimize, [_row("o", "optimize")])
    write_jsonl(select, [_row("s", "select")])
    write_jsonl(
        roles,
        [
            {
                "norm_uid": uid,
                "task": "demo",
                "corpus": "c",
                "source_group": f"c:source:{uid}",
                "retriever_source_group": f"demo\\u001fc\\u001fsource\\u001f{uid}",
                "split": split,
            }
            for uid, split in (("p_train", "train"), ("p_dev", "dev"), ("o", "train"), ("s", "train"))
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_exact_retriever_retry_bridge",
            "--task",
            "demo",
            "--prior-labels",
            str(prior),
            "--upstream-roles",
            str(roles),
            "--optimize-truth",
            str(optimize),
            "--select-truth",
            str(select),
            "--output",
            str(output),
            "--report",
            str(report),
            "--expected-prior-input",
            "2",
            "--expected-prior-train",
            "1",
            "--expected-optimize-input",
            "1",
            "--expected-optimize-matches",
            "1",
            "--expected-select-input",
            "1",
        ],
    )
    main()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert {row["norm_uid"] for row in rows} == {"p_train", "o"}
    assert all(row["training_eligible"] is True for row in rows)
    audit = json.loads(report.read_text())
    assert audit["counts"]["prior_authoritative_roles"] == {"dev": 1, "train": 1}
    assert audit["counts"]["select_rows_permanently_excluded"] == 1
