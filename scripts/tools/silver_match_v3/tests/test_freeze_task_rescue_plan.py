import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.freeze_task_rescue_plan import freeze_plan


def _dump(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value)
    else:
        path.write_text(json.dumps(value) + "\n")
    return path


def _fixture(tmp_path: Path):
    bank = _dump(tmp_path / "bank.json", {"metrics": [{"metric_id": "a1"}, {"metric_id": "a2"}]})
    manifest = _dump(
        tmp_path / "manifest.json",
        {
            "banks": {"t": {"path": str(bank), "count": 2, "source_sha256": "bank-sha"}},
            "corpora": {"c1": {"task": "t", "count": 1}, "c2": {"task": "t", "count": 1}},
        },
    )
    plan = _dump(
        tmp_path / "plan.json",
        {
            "status": "FROZEN_READY_FOR_UNLABELED_PRODUCTION",
            "task": "t",
            "corpora": ["c1", "c2"],
            "expected_count": 2,
            "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
            "adjudicator": {"candidate_depth": 50, "model": "/model", "selection": {"path": "/adj"}},
            "verifier": {"selection": {"path": "/verify"}, "production_policy": {"path": "/policy"}},
        },
    )
    finals = {}
    for corpus in ("c1", "c2"):
        output = _dump(tmp_path / f"{corpus}.jsonl", f'{{"corpus":"{corpus}"}}\n')
        final_report = _dump(
            tmp_path / f"{corpus}.jsonl.report.json",
            {
                "complete": True,
                "strict_production": True,
                "task": "t",
                "corpus": corpus,
                "output_sha256": sha256_file(output),
                "production_plan": {"sha256": sha256_file(plan)},
            },
        )
        finals[corpus] = {
            "output": {"path": str(output), "sha256": sha256_file(output)},
            "report": {"path": str(final_report), "sha256": sha256_file(final_report)},
        }
    report = _dump(
        tmp_path / "production.report.json",
        {
            "schema_version": "silver-match-v3-task-production-run-v1",
            "status": "COMPLETE_PRE_RESCUE_ONLY",
            "task": "t",
            "candidate_count": 2,
            "plan": {"sha256": sha256_file(plan)},
            "final_pre_rescue": finals,
        },
    )
    system_values = []
    audits = []
    for system_index, system in enumerate(("base", "adapter")):
        for corpus_index, corpus in enumerate(("c1", "c2")):
            candidate = _dump(
                tmp_path / f"{corpus}.{system}.jsonl",
                f'{{"system":"{system}","corpus":"{corpus}","salt":{system_index * 2 + corpus_index}}}\n',
            )
            meta = _dump(tmp_path / f"{corpus}.{system}.jsonl.meta.json", {"ok": True, "system": system})
            audit = _dump(
                tmp_path / f"{corpus}.{system}.audit.json",
                {
                    "schema_version": "silver-match-v3-production-candidate-audit-v1",
                    "complete": True,
                    "task": "t",
                    "corpus": corpus,
                    "manifest_sha256": sha256_file(manifest),
                    "bank_source_sha256": "bank-sha",
                    "bank_count": 2,
                    "materialized_k": 2,
                    "expected_k": 2,
                    "observed_count": 1,
                    "candidate_inputs": {
                        str(candidate): {
                            "sha256": sha256_file(candidate),
                            "meta": str(meta),
                            "meta_sha256": sha256_file(meta),
                        }
                    },
                },
            )
            system_values.append(f"{system}={candidate}")
            audits.append(audit)
    return plan, report, system_values, audits


def test_freezes_two_system_exact_full_bank_rescue(tmp_path: Path) -> None:
    plan, report, systems, audits = _fixture(tmp_path)
    exclusion = _dump(tmp_path / "exclusions.jsonl", '{"norm_uid":"excluded"}\n')
    result = freeze_plan(
        production_plan_path=plan,
        production_report_path=report,
        candidate_system_values=systems,
        candidate_audit_paths=audits,
        repo_root=Path(".").resolve(),
        abstention_prompt_path=Path("scripts/tools/silver_match_v3/prompts/verify_abstention_v1.txt"),
        blind_audit_exclusion_paths=[exclusion],
    )
    assert result["status"] == "FROZEN_READY_FOR_REPEATED_FULL_BANK_RESCUE"
    assert set(result["candidate_systems"]) == {"adapter", "base"}
    assert result["rescue_policy"]["coverage_repeats"] == 2
    assert result["rescue_policy"]["reinclude_primary"] is True
    assert set(result["primary_final_pre_rescue"]) == {"c1", "c2"}


def test_rejects_system_missing_one_task_corpus(tmp_path: Path) -> None:
    plan, report, systems, audits = _fixture(tmp_path)
    exclusion = _dump(tmp_path / "exclusions.jsonl", '{"norm_uid":"excluded"}\n')
    systems = [value for value in systems if not value.startswith("base=") or "c2.base" not in value]
    audits = [path for path in audits if "c2.base" not in path.name]
    with pytest.raises(ValueError, match="lacks exact corpus coverage"):
        freeze_plan(
            production_plan_path=plan,
            production_report_path=report,
            candidate_system_values=systems,
            candidate_audit_paths=audits,
            repo_root=Path(".").resolve(),
            abstention_prompt_path=Path("scripts/tools/silver_match_v3/prompts/verify_abstention_v1.txt"),
            blind_audit_exclusion_paths=[exclusion],
        )


def test_preserves_selected_three_order_verifier_in_rescue(tmp_path: Path) -> None:
    plan, report, systems, audits = _fixture(tmp_path)
    plan_payload = json.loads(plan.read_text())
    plan_payload["verifier"]["orders"] = ["original", "hashed", "reverse"]
    plan.write_text(json.dumps(plan_payload) + "\n")
    plan_sha = sha256_file(plan)
    report_payload = json.loads(report.read_text())
    for values in report_payload["final_pre_rescue"].values():
        report_path = Path(values["report"]["path"])
        final_payload = json.loads(report_path.read_text())
        final_payload["production_plan"]["sha256"] = plan_sha
        report_path.write_text(json.dumps(final_payload) + "\n")
        values["report"]["sha256"] = sha256_file(report_path)
    report_payload.update(
        {
            "schema_version": "silver-match-v3-task-production-run-v2",
            "plan": {"sha256": plan_sha},
            "strict_verification": {
                "orders": ["original", "hashed", "reverse"]
            },
        }
    )
    report.write_text(json.dumps(report_payload) + "\n")
    exclusion = _dump(tmp_path / "exclusions.jsonl", '{"norm_uid":"excluded"}\n')
    result = freeze_plan(
        production_plan_path=plan,
        production_report_path=report,
        candidate_system_values=systems,
        candidate_audit_paths=audits,
        repo_root=Path(".").resolve(),
        abstention_prompt_path=Path(
            "scripts/tools/silver_match_v3/prompts/verify_abstention_v1.txt"
        ),
        blind_audit_exclusion_paths=[exclusion],
    )
    assert result["schema_version"] == "silver-match-v3-task-rescue-plan-v3"
    assert result["rescue_policy"]["contrastive_verification_orders"] == [
        "original",
        "hashed",
        "reverse",
    ]
    assert result["rescue_policy"][
        "strict_all_selected_order_contrastive_verification"
    ] is True
