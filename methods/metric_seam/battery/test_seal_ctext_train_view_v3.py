"""Tests for the additive immutable TRAIN-only ctext projection."""

from __future__ import annotations

import json
from pathlib import Path
import stat
from tempfile import TemporaryDirectory

import pytest

from methods.metric_seam.battery.audit_ctext_compiler_view_v1 import build_receipt
from methods.metric_seam.battery.audit_full_corpus_sanitizer_v1 import (
    build_replay_receipt,
)
from methods.metric_seam.battery.seal_ctext_train_view_v3 import (
    REDACTION_TOKEN,
    credential_pattern_counts,
    prepare_train_view,
    sanitize_ctext,
)


def _inputs(root: Path) -> tuple[Path, Path]:
    source = root / "items.json"
    source.write_text(
        json.dumps(
            [
                {
                    "datapoint_id": f"private-{index}",
                    "ctext": f"diff body {index}",
                    "judgement": index % 2,
                    "repo": "private/repo",
                    "raw_text": f"secret raw {index}",
                }
                for index in range(8)
            ]
        )
    )
    contract = root / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema": "projected-contract-test-v1",
                "criterion_id": "a-test",
                "articulated_construct": "identifier relations",
            }
        )
    )
    return source, contract


def test_bundle_has_only_opaque_alias_and_train_ctext_rows():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source, contract = _inputs(root)
        bundle_path, manifest_path = prepare_train_view(
            source=source,
            contract_path=contract,
            out_dir=root / "prepared",
            task="code_review",
            criterion_id="a-test",
            train_count=5,
            split_seed=7,
            dependency_files={"test_dependency": Path(__file__)},
            dependency_packages={"tree-sitter", "definitely-not-installed-package"},
        )
        bundle = json.loads(bundle_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        assert len(bundle["train_items"]) == 5
        assert [row["item_key"] for row in bundle["train_items"]] == [
            f"train_{index:04d}" for index in range(1, 6)
        ]
        assert all(set(row) == {"ctext", "item_key"} for row in bundle["train_items"])
        encoded = bundle_path.read_text()
        assert "private-" not in encoded
        assert "judgement" not in encoded
        assert "secret raw" not in encoded
        assert manifest["partition"]["heldout_count"] == 3
        assert manifest["partition"]["heldout_rows_materialized"] is False
        assert manifest["projection"]["outcome_values_recorded_in_manifest"] is False
        assert manifest["credential_redaction"]["applied_before_partition"] is True
        assert manifest["environment"]["packages"]["tree-sitter"]
        assert manifest["environment"]["packages"]["definitely-not-installed-package"] is None
        assert stat.S_IMODE(bundle_path.stat().st_mode) & stat.S_IWUSR == 0
        assert stat.S_IMODE(manifest_path.stat().st_mode) & stat.S_IWUSR == 0


def test_sanitizer_replaces_only_credential_value_and_preserves_normal_literals():
    fake_value = "synthetic-not-a-real-secret-12345"
    raw = (
        f'client_secret = "{fake_value}"\n'
        'timeout_seconds = 30\n'
        'endpoint = "https://example.invalid/v1"\n'
        'password = get_from_environment()\n'
    )
    sanitized, counts = sanitize_ctext(raw)
    assert fake_value not in sanitized
    assert f'client_secret = "{REDACTION_TOKEN}"' in sanitized
    assert "timeout_seconds = 30" in sanitized
    assert 'endpoint = "https://example.invalid/v1"' in sanitized
    assert "password = get_from_environment()" in sanitized
    assert counts["credential_assignment_long_literal"] == 1
    assert sum(credential_pattern_counts(sanitized).values()) == 0


def test_every_credential_pattern_is_deterministic_and_scans_clean_afterward():
    # Build synthetic surfaces by concatenation so the test source itself does
    # not look like a live credential file to repository scanners.
    cases = {
        "private_key_block": (
            "-----BEGIN " + "PRIVATE KEY-----\nSYNTHETICONLY\n-----END PRIVATE KEY-----"
        ),
        "aws_access_key": "AK" + "IA" + "A" * 16,
        "github_token": "gh" + "p_" + "A" * 40,
        "google_api_key": "AI" + "za" + "A" * 35,
        "openai_style_key": "s" + "k-" + "A" * 24,
        "slack_token": "xo" + "xb-" + "A" * 12,
        "jwt_compact": "ey" + "J" + "A" * 12 + "." + "B" * 12 + "." + "C" * 12,
    }
    for category, synthetic in cases.items():
        raw = f"prefix {synthetic} suffix"
        first, counts = sanitize_ctext(raw)
        second, second_counts = sanitize_ctext(raw)
        assert first == second
        assert counts == second_counts
        assert synthetic not in first
        assert REDACTION_TOKEN in first
        assert counts[category] == 1
        assert sum(credential_pattern_counts(first).values()) == 0


def test_manifest_counts_do_not_record_values_or_source_ids_and_audit_is_zero():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source, contract = _inputs(root)
        rows = json.loads(source.read_text())
        fake_value = "synthetic-not-a-real-secret-67890"
        rows[0]["ctext"] = f'password = "{fake_value}"'
        source.write_text(json.dumps(rows))
        bundle_path, manifest_path = prepare_train_view(
            source=source,
            contract_path=contract,
            out_dir=root / "prepared",
            task="code_review",
            criterion_id="a-test",
            train_count=5,
            split_seed=7,
        )
        manifest_text = manifest_path.read_text()
        manifest = json.loads(manifest_text)
        assert fake_value not in manifest_text
        assert not any(f"private-{index}" in manifest_text for index in range(8))
        assert manifest["credential_redaction"]["full"]["changed_row_count"] == 1
        assert manifest["credential_redaction"]["full"]["total_matches"] == 1
        receipt = build_receipt(
            bundle_path=bundle_path,
            manifest_path=manifest_path,
            source_path=source,
        )
        assert receipt["credential_scan"]["total_matches"] == 0
        assert receipt["interface_audit"]["source_identifier_occurrence_count"] == 0
        assert receipt["interface_audit"]["structural_outcome_key_count"] == 0
        assert receipt["compiler_handoff_envelope"]["contains_steward_manifest"] is False
        assert receipt["compiler_handoff_allowed"] is True
        replay = build_replay_receipt(
            source_path=source,
            bundle_path=bundle_path,
            manifest_path=manifest_path,
        )
        assert replay["post_sanitization_full_corpus_clean"] is True
        assert replay["compiler_train_rows_equal_replayed_sanitized_train"] is True
        assert replay["recorded_manifest_counts_match"] is True
        assert replay["replay_passed"] is True


def test_partition_is_deterministic_without_exposing_source_ids():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source, contract = _inputs(root)
        first, _ = prepare_train_view(
            source=source,
            contract_path=contract,
            out_dir=root / "first",
            task="code_review",
            criterion_id="a-test",
            train_count=5,
            split_seed=7,
        )
        second, _ = prepare_train_view(
            source=source,
            contract_path=contract,
            out_dir=root / "second",
            task="code_review",
            criterion_id="a-test",
            train_count=5,
            split_seed=7,
        )
        assert first.read_bytes() == second.read_bytes()


def test_refuses_overwrite_and_outcome_bearing_contract_key():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source, contract = _inputs(root)
        out = root / "prepared"
        prepare_train_view(
            source=source,
            contract_path=contract,
            out_dir=out,
            task="code_review",
            criterion_id="a-test",
            train_count=5,
        )
        with pytest.raises(FileExistsError):
            prepare_train_view(
                source=source,
                contract_path=contract,
                out_dir=out,
                task="code_review",
                criterion_id="a-test",
                train_count=5,
            )

        contaminated = root / "contaminated.json"
        contaminated.write_text(json.dumps({"criterion_id": "a-test", "score": 0.9}))
        with pytest.raises(ValueError, match="forbidden key"):
            prepare_train_view(
                source=source,
                contract_path=contaminated,
                out_dir=root / "contaminated-output",
                task="code_review",
                criterion_id="a-test",
                train_count=5,
            )


def test_rejected_active_code_bundle_is_absent():
    root = Path(__file__).resolve().parents[3]
    rejected = (
        root
        / "outputs/metric_seam_pilot/reconstruction_v2/blind_code_a407_prepare_001/"
        "compiler_bundle.json"
    )
    assert not rejected.exists()
