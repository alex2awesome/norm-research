from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.audit_notice_rescue_sealed_artifacts import (
    EXPECTED_TRIALS,
    _assert_same_jsonl,
    _validate_artifact_lock,
    _validate_bank_binding,
    _validate_capture_universe,
    _validate_typed_output,
    _validate_verifier_output,
)
from scripts.tools.silver_match_v3.common import sha256_file, write_jsonl
from scripts.tools.silver_match_v3.continue_notice_rescue_from_sealed_artifacts import (
    _snapshot_dependencies,
    _validate_blind_bank_binding,
    _verify_dependency_snapshot,
)


def _coverage_fixture() -> tuple[
    set[str], dict[str, dict], dict[str, dict[str, dict]]
]:
    bank = {f"a{index}" for index in range(88)}
    primary = {
        "u": {
            "corpus": "notice_and_comment",
            "row": 7,
            "candidate_ids": [f"a{index}" for index in range(50)],
        }
    }
    halves = [
        [f"a{index}" for index in range(44)],
        [f"a{index}" for index in range(44, 88)],
        [f"a{index}" for index in range(44)],
        [f"a{index}" for index in range(44, 88)],
    ]
    systems = ["adapter", "bge", "bge", "nemotron_base"]
    captures = {
        name: {
            "u": {
                "corpus": "notice_and_comment",
                "row": 7,
                "candidate_ids": halves[index],
                "primary_candidate_ids": [f"a{position}" for position in range(50)],
                "rescue_system": systems[index],
                "rescue_lane": f"{systems[index]}:rank",
                "rescue_capture": 0 if index < 2 else 1,
            }
        }
        for index, name in enumerate(EXPECTED_TRIALS)
    }
    return bank, primary, captures


def test_full_bank_coverage_and_primary_reinclude_fail_closed() -> None:
    bank, primary, captures = _coverage_fixture()
    report = _validate_capture_universe(
        captures, primary=primary, bank_ids=bank, coverage_repeats=2
    )
    assert report["bank_metrics_per_uid"] == 88
    captures["trial-003"]["u"]["candidate_ids"].pop()
    with pytest.raises(ValueError, match="coverage mismatch"):
        _validate_capture_universe(
            captures, primary=primary, bank_ids=bank, coverage_repeats=2
        )


def test_cross_trial_routing_and_distinct_systems_fail_closed() -> None:
    bank, primary, captures = _coverage_fixture()
    captures["trial-002"]["u"]["row"] = 8
    with pytest.raises(ValueError, match="routing mismatch"):
        _validate_capture_universe(
            captures, primary=primary, bank_ids=bank, coverage_repeats=2
        )
    _, primary, captures = _coverage_fixture()
    for rows in captures.values():
        rows["u"]["rescue_system"] = "bge"
    with pytest.raises(ValueError, match="distinct rescue systems"):
        _validate_capture_universe(
            captures, primary=primary, bank_ids=bank, coverage_repeats=2
        )


def test_manifest_bank_source_artifact_binding_fails_on_mutation(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("source\n", encoding="utf-8")
    source_sha = sha256_file(source)
    bank = tmp_path / "bank.json"
    bank.write_text(
        json.dumps(
            {
                "source_sha256": source_sha,
                "metrics": [{"metric_id": f"a{index}"} for index in range(88)],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "banks": {
            "notice-and-comment": {
                "path": str(bank),
                "source_path": str(source),
                "source_sha256": source_sha,
                "count": 88,
            }
        }
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _, _, _, ids, observed_sha = _validate_bank_binding(manifest_path, manifest)
    assert len(ids) == 88 and observed_sha == source_sha
    source.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest-bound"):
        _validate_bank_binding(manifest_path, manifest)


def test_artifact_lock_binds_bank_bytes_and_task_norms(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("source\n", encoding="utf-8")
    source_sha = sha256_file(source)
    bank = tmp_path / "bank.json"
    bank_bytes = (
        json.dumps(
            {
                "source_sha256": source_sha,
                "metrics": [{"metric_id": f"a{index}"} for index in range(88)],
            }
        )
        + "\n"
    )
    bank.write_text(bank_bytes, encoding="utf-8")
    norms = {}
    for corpus in ("notice_and_comment", "nc_public_comments"):
        path = tmp_path / f"{corpus}.jsonl"
        write_jsonl(path, [{"norm_uid": f"{corpus}-u"}])
        norms[corpus] = path
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "banks": {
            "notice-and-comment": {
                "path": str(bank),
                "source_path": str(source),
                "source_sha256": source_sha,
                "count": 88,
            }
        },
        "corpora": {
            corpus: {
                "path": str(path),
                "count": 1,
                "task": "notice-and-comment",
            }
            for corpus, path in norms.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    lock_path = tmp_path / "artifact_lock.json"
    lock = {
        "schema_version": "silver-match-v3.0",
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "banks": {
            "notice-and-comment": {
                "path": str(bank),
                "count": 88,
                "sha256": sha256_file(bank),
            }
        },
        "norms": {
            corpus: {"path": str(path), "count": 1, "sha256": sha256_file(path)}
            for corpus, path in norms.items()
        },
    }
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    binding = _validate_artifact_lock(
        artifact_lock_path=lock_path,
        manifest_path=manifest_path,
        manifest=manifest,
        bank_path=bank,
    )
    assert binding["bank"]["sha256"] == sha256_file(bank)
    bank.write_text(bank_bytes + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact lock"):
        _validate_artifact_lock(
            artifact_lock_path=lock_path,
            manifest_path=manifest_path,
            manifest=manifest,
            bank_path=bank,
        )
    bank.write_text(bank_bytes, encoding="utf-8")
    lock["banks"]["notice-and-comment"]["sha256"] = "0" * 64
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact lock"):
        _validate_artifact_lock(
            artifact_lock_path=lock_path,
            manifest_path=manifest_path,
            manifest=manifest,
            bank_path=bank,
        )


def test_recomputed_aggregate_or_combine_content_mutation_is_rejected(
    tmp_path: Path,
) -> None:
    left, right = tmp_path / "left.jsonl", tmp_path / "right.jsonl"
    write_jsonl(left, [{"norm_uid": "u", "decision": "MATCH"}])
    write_jsonl(right, [{"norm_uid": "u", "decision": "NOISE"}])
    with pytest.raises(ValueError, match="recomputation differs"):
        _assert_same_jsonl(left, right, "mutation")


def test_verifier_decision_and_metric_schema_mutation_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "verify.jsonl"
    base = {
        "norm_uid": "u",
        "corpus": "notice_and_comment",
        "task": "notice-and-comment",
        "row": 1,
        "candidate_bank_source_sha256": "bank",
        "order_mode": "original",
        "prompt_sha256": "7f8fb51b43bf367ed96dd3cf5e1b871e87a4dced87c2198035992aeb751c696d",
        "primary_prompt_sha256": "c839a28c4c452de8faa937e064d2ad4824d06dfe65730ce29263815d355cd111",
        "primary_metric_id": "a0",
        "alternative_ids": ["a1"],
        "decision": "CONFIRM_MATCH",
        "metric_id": "a1",
        "confidence": "high",
        "parse_error": None,
    }
    write_jsonl(path, [base])
    original = {
        "u": {
            "candidate_ids": ["a0", "a1"],
            "metric_id": "a0",
        }
    }
    with pytest.raises(ValueError, match="decision/metric schema"):
        _validate_verifier_output(
            path,
            order="original",
            expected_uids={"u"},
            original=original,
            bank_sha="bank",
        )


def test_typed_decision_schema_mutation_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "typed.jsonl"
    write_jsonl(
        path,
        [
            {
                "norm_uid": "u",
                "corpus": "notice_and_comment",
                "task": "notice-and-comment",
                "row": 1,
                "bank_source_sha256": "bank",
                "rescue_bank_count": 88,
                "order_mode": "original",
                "prompt_sha256": "bb7ce2f492d6e933242b9650eae98eb588fd66b7403ccbb6eac19c3181b32b5d",
                "rescue_coverage_repeats": 2,
                "rescue_reincludes_primary": True,
                "decision": "MADE_UP",
                "confirmed_decision": None,
                "metric_id": None,
                "confidence": "high",
                "parse_error": None,
            }
        ],
    )
    with pytest.raises(ValueError, match="decision schema"):
        _validate_typed_output(
            path,
            order="original",
            expected={"u": ("notice_and_comment", 1)},
            bank_sha="bank",
        )


def test_recursive_dependency_snapshot_detects_mutation(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    package = repo / "scripts/tools/silver_match_v3"
    package.mkdir(parents=True)
    for path in [repo / "scripts/__init__.py", repo / "scripts/tools/__init__.py", package / "__init__.py"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    entry, dependency = package / "a.py", package / "b.py"
    entry.write_text("from .b import VALUE\n", encoding="utf-8")
    dependency.write_text("VALUE = 1\n", encoding="utf-8")
    _, inventory = _snapshot_dependencies(
        entrypoints=[entry], repo=repo, snapshot_root=tmp_path / "snapshot"
    )
    _verify_dependency_snapshot(entrypoints=[entry], repo=repo, inventory=inventory)
    dependency.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dependency drift"):
        _verify_dependency_snapshot(entrypoints=[entry], repo=repo, inventory=inventory)


def test_blind_bank_binding_has_no_self_fallback(tmp_path: Path) -> None:
    bank = tmp_path / "bank.json"
    bank.write_text("{\"metrics\": []}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "banks": {
                    "notice-and-comment": {
                        "path": str(bank),
                        "source_sha256": "authoritative",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "blind"
    copied = root / "banks/notice-and-comment.json"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(bank.read_bytes())
    (root / "sample_report.json").write_text(
        json.dumps(
            {
                "manifest_sha256": sha256_file(manifest),
                "bank_outputs": {
                    "notice-and-comment": {
                        "path": str(copied),
                        "sha256": sha256_file(copied),
                        "source_sha256": "wrong",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    pack = root / "task__notice-and-comment.label_pack"
    pack.mkdir()
    (pack / "validation.json").write_text(
        json.dumps(
            {
                "bank_metric_count": 88,
                "bank_source_sha256": "wrong",
                "inputs": {
                    "canonical_bank": {
                        "source_sha256": "wrong",
                        "sha256": sha256_file(bank),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="authoritative bank mismatch"):
        _validate_blind_bank_binding(
            manifest_path=manifest,
            blind_root=root,
            task="notice-and-comment",
            expected_count=88,
        )
