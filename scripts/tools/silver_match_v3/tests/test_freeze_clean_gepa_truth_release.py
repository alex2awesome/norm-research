import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import sha256_file, write_jsonl
from scripts.tools.silver_match_v3.adjudicate_gemma import DECISIONS
from scripts.tools.silver_match_v3.freeze_clean_gepa_truth_release import (
    TRUTH_DECISIONS,
    _expected_hydrated_item,
    _load_task_norms,
    _verify_resolver_lineage,
    freeze,
)
from scripts.tools.silver_match_v3.make_calibration import split_group_for


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _fixture(tmp_path: Path) -> Namespace:
    task, role = "code-review", "optimize"
    panel = tmp_path / "panel"
    consensus = panel / "consensus"
    label_pack = panel / "label_pack"
    consensus.mkdir(parents=True)
    label_pack.mkdir()
    guide_path = tmp_path / "INDEPENDENT_LABELING_GUIDE.md"
    guide_path.write_text("strict immutable labeling guide\n")
    canonical = [
        {
            "norm_uid": "u1",
            "task": task,
            "corpus": "c",
            "source_id": "source-1",
            "row": 1,
            "norm": "criterion one",
            "context": "human context one",
        },
        {
            "norm_uid": "u2",
            "task": task,
            "corpus": "c",
            "source_id": "source-2",
            "row": 2,
            "norm": "criterion two",
            "context": "human context two",
        },
    ]
    truth = [
        {
            "norm_uid": "u1",
            "task": task,
            "corpus": "c",
            "row": 1,
            "source_group": split_group_for(canonical[0]),
            "gepa_role": role,
            "split": "train",
            "decision": "MATCH",
            "metric_id": "a0",
            "confidence": "high",
            "current_bank_source_sha256": "bank-source-hash",
            "agreement_sources": ["A", "B"],
        },
        {
            "norm_uid": "u2",
            "task": task,
            "corpus": "c",
            "row": 2,
            "source_group": split_group_for(canonical[1]),
            "gepa_role": role,
            "split": "train",
            "decision": "NOISE",
            "metric_id": None,
            "confidence": "high",
            "current_bank_source_sha256": "bank-source-hash",
            "agreement_sources": ["A", "B"],
        },
    ]
    truth_path = consensus / "resolved.jsonl"
    unresolved_path = consensus / "unresolved.jsonl"
    identities_path = panel / "identities.jsonl"
    write_jsonl(truth_path, truth)
    unresolved_path.write_text("")
    write_jsonl(
        identities_path,
        (
            {
                "norm_uid": row["norm_uid"],
                "source_group": split_group_for(canonical[index]),
            }
            for index, row in enumerate(truth)
        ),
    )
    role_freeze_path = panel / "FREEZE.json"
    role_freeze_path.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-clean-gepa-panel-freeze-v1",
                "status": "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES",
                "task": task,
                "role": role,
                "required_upstream_split": "train",
                "selected_count": 2,
                "outputs": {"identities": _ref(identities_path)},
            }
        )
    )

    manifest_path = tmp_path / "manifest.json"
    norms_path = tmp_path / "norms.jsonl"
    bank_source_path = tmp_path / "bank_source.json"
    candidate_source_path = tmp_path / "candidate_source.jsonl"
    upstream_freeze_path = tmp_path / "upstream_freeze.json"
    write_jsonl(norms_path, canonical)
    manifest_path.write_text(
        json.dumps(
            {
                "corpora": {
                    "c": {
                        "task": task,
                        "path": str(norms_path),
                        "count": len(canonical),
                    }
                }
            }
        )
        + "\n"
    )
    bank_source_path.write_text(
        '{"source_sha256": "bank-source-hash", "metrics": '
        '[{"metric_id": "a0"}, {"metric_id": "a1"}]}\n'
    )
    write_jsonl(candidate_source_path, ({"norm_uid": row["norm_uid"]} for row in truth))
    upstream_freeze_path.write_text("{}\n")
    source_refs = {
        "manifest": _ref(manifest_path),
        "bank_source": _ref(bank_source_path),
        "candidate_source": _ref(candidate_source_path),
        "identities": _ref(identities_path),
        "identity_freeze": _ref(role_freeze_path),
        "upstream_role_freeze": _ref(upstream_freeze_path),
    }
    candidates_path = label_pack / "candidates.top2.jsonl"
    write_jsonl(
        candidates_path,
        (
            {"norm_uid": row["norm_uid"], "candidates": [{"metric_id": "a0"}]}
            for row in truth
        ),
    )
    candidate_release_path = label_pack / "validation.json"
    candidate_release_path.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-clean-gepa-label-pack-v1",
                "status": "FROZEN_TRUTH_HIDDEN_BEFORE_LABELING",
                "truth_hidden": True,
                "task": task,
                "gepa_role": role,
                "count": 2,
                "candidate_k": 2,
                "bank_source_sha256": "bank-source-hash",
                "inputs": source_refs,
                "outputs": {"candidates": _ref(candidates_path)},
            }
        )
    )

    independence_passes = {}
    report_passes = {}
    hydrated = [_expected_hydrated_item(row, role=role) for row in canonical]
    for name, seed in (("A", 11), ("B", 29)):
        pass_root = panel / f"independent_pass_{name}"
        pass_root.mkdir()
        bank_path = pass_root / "bank.json"
        items_path = pass_root / "items.jsonl"
        validation_path = pass_root / "validation.json"
        labels_path = pass_root / "labels.validated.jsonl"
        labels_validation_path = pass_root / "labels.validation.json"
        transcript_path = pass_root / "TRANSCRIPT_ISOLATION_AUDIT.json"
        chunks_root = pass_root / "chunks"
        raw_root = pass_root / "raw_labels"
        logs_root = pass_root / "logs"
        chunks_root.mkdir()
        raw_root.mkdir()
        logs_root.mkdir()
        chunk_path = chunks_root / "part-000.jsonl"
        raw_path = raw_root / "part-000.json"
        log_path = logs_root / "part-000.log"
        write_jsonl(chunk_path, hydrated)
        raw_path.write_text("{}\n")
        log_path.write_text("sandbox: read-only\napproval: never\n")
        bank_path.write_text(
            json.dumps(
                {
                    "metrics": (
                        [{"metric_id": "a0"}, {"metric_id": "a1"}]
                        if name == "A"
                        else [{"metric_id": "a1"}, {"metric_id": "a0"}]
                    )
                }
            )
        )
        write_jsonl(items_path, hydrated if name == "A" else reversed(hydrated))
        validation_path.write_text(
            json.dumps(
                {
                    "schema_version": "silver-match-v3-permuted-independent-teacher-pack-v1",
                    "status": "FROZEN_TRUTH_HIDDEN_BEFORE_LABELING",
                    "truth_hidden": True,
                    "prior_decisions_proposals_predictions_and_outcomes_hidden": True,
                    "task": task,
                    "gepa_role": role,
                    "count": 2,
                    "seed": seed,
                    "candidate_k": 2,
                    "bank_source_sha256": "bank-source-hash",
                    "inputs": source_refs,
                    "outputs": {
                        "bank": _ref(bank_path),
                        "items": _ref(items_path),
                        "chunks": {str(chunk_path): sha256_file(chunk_path)},
                    },
                    "source_pack": {
                        "path": str(label_pack),
                        "validation_sha256": sha256_file(candidate_release_path),
                    },
                }
            )
        )
        write_jsonl(
            labels_path,
            ({**row, "annotator": f"independent-{name}"} for row in truth),
        )
        transcript_path.write_text(
            json.dumps(
                {
                    "schema_version": "silver-match-v3-isolated-labeler-transcript-audit-v1",
                    "status": "PASS",
                    "complete": True,
                    "violations": [],
                    "pack_root": str(pass_root),
                    "bank": _ref(bank_path),
                    "items": _ref(items_path),
                    "pack_validation": _ref(validation_path),
                    "full_pack_artifact_binding": True,
                    "guides": [_ref(guide_path)],
                    "expected_chunks": 1,
                    "audited_chunks": 1,
                    "chunks": [
                        {
                            "chunk": "part-000",
                            "chunk_sha256": sha256_file(chunk_path),
                            "raw_label_sha256": sha256_file(raw_path),
                            "log_sha256": sha256_file(log_path),
                            "command_count": 1,
                        }
                    ],
                }
            )
        )
        labels_validation_path.write_text(
            json.dumps(
                {
                    "schema_version": "silver-match-v3-independent-label-validation-v1",
                    "complete": True,
                    "task": task,
                    "count": 2,
                    "output": _ref(labels_path),
                    "pack_validation": _ref(validation_path),
                    "retrieval_candidate_sha256": sha256_file(candidates_path),
                    "transcript_audit": _ref(transcript_path),
                    "raw_chunks": {
                        "part-000": {
                            "count": 2,
                            "raw_sha256": sha256_file(raw_path),
                        }
                    },
                }
            )
        )
        independence_passes[name] = {
            "root": str(pass_root),
            "seed": seed,
            "validation_sha256": sha256_file(validation_path),
            "bank_sha256": sha256_file(bank_path),
            "items_sha256": sha256_file(items_path),
        }
        report_passes[name] = {
            "count": 2,
            "labels": _ref(labels_path),
            "pack_validation": _ref(validation_path),
            "pack_bank_sha256": sha256_file(bank_path),
            "pack_items_sha256": sha256_file(items_path),
        }

    independence_path = panel / "INDEPENDENCE_AUDIT.json"
    independence_path.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-independent-pack-view-audit-v1",
                "status": "FROZEN_MUTUALLY_PREDICTION_HIDDEN_BEFORE_LABELING",
                "task": task,
                "count": 2,
                "distinct_bank_order": True,
                "distinct_item_order": True,
                "distinct_seeds": True,
                "pass_predictions_mutually_visible": False,
                "prior_truth_or_predictions_exposed_to_either_pass": False,
                "same_bank_leaf_set": True,
                "same_canonical_item_content_by_uid": True,
                "same_frozen_source_pack": True,
                "same_uid_set": True,
                "passes": independence_passes,
            }
        )
    )
    report_path = consensus / "report.json"
    rounds = [
        {
            **report_passes["A"],
            "pass": "A",
            "ordinal": 1,
            "labeled_count": 2,
            "unresolved_before": 2,
            "newly_resolved": 0,
            "unresolved_after": 2,
        },
        {
            **report_passes["B"],
            "pass": "B",
            "ordinal": 2,
            "labeled_count": 2,
            "unresolved_before": 2,
            "newly_resolved": 2,
            "unresolved_after": 0,
        },
    ]
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-exact-multi-pass-truth-report-v1",
                "complete": True,
                "task": task,
                "gepa_role": role,
                "source_count": 2,
                "resolved_count": 2,
                "unresolved_count": 0,
                "inputs": {"passes": report_passes},
                "rounds": rounds,
                "outputs": {
                    "resolved": _ref(truth_path),
                    "unresolved": _ref(unresolved_path),
                },
            }
        )
    )
    return Namespace(
        task=task,
        role=role,
        truth=str(truth_path),
        consensus_report=str(report_path),
        role_freeze=str(role_freeze_path),
        identities=str(identities_path),
        independence_audit=str(independence_path),
        candidate_release=str(candidate_release_path),
        transcript_guide=str(guide_path),
        transcript_audit=[],
        output=str(panel / "EXACT_TRUTH_RELEASE.json"),
    )


def _rebind_changed_pass_validation(args: Namespace, name: str = "A") -> None:
    independence_path = Path(args.independence_audit)
    independence = json.loads(independence_path.read_text())
    validation_path = Path(independence["passes"][name]["root"]) / "validation.json"
    new_sha = sha256_file(validation_path)
    independence["passes"][name]["validation_sha256"] = new_sha
    independence_path.write_text(json.dumps(independence))
    report_path = Path(args.consensus_report)
    report = json.loads(report_path.read_text())
    report["inputs"]["passes"][name]["pack_validation"]["sha256"] = new_sha
    for round_meta in report["rounds"]:
        if round_meta["pass"] == name:
            round_meta["pack_validation"]["sha256"] = new_sha
    report_path.write_text(json.dumps(report))
    labels_validation_path = validation_path.parent / "labels.validation.json"
    labels_validation = json.loads(labels_validation_path.read_text())
    labels_validation["pack_validation"]["sha256"] = new_sha
    labels_validation_path.write_text(json.dumps(labels_validation))


def _rebind_changed_pass_items(args: Namespace, name: str = "A") -> None:
    independence_path = Path(args.independence_audit)
    independence = json.loads(independence_path.read_text())
    pass_root = Path(independence["passes"][name]["root"])
    validation_path = pass_root / "validation.json"
    items_path = pass_root / "items.jsonl"
    items_sha = sha256_file(items_path)
    validation = json.loads(validation_path.read_text())
    validation["outputs"]["items"]["sha256"] = items_sha
    validation_path.write_text(json.dumps(validation))
    independence["passes"][name]["items_sha256"] = items_sha
    independence_path.write_text(json.dumps(independence))
    report_path = Path(args.consensus_report)
    report = json.loads(report_path.read_text())
    report["inputs"]["passes"][name]["pack_items_sha256"] = items_sha
    for round_meta in report["rounds"]:
        if round_meta["pass"] == name:
            round_meta["pack_items_sha256"] = items_sha
    report_path.write_text(json.dumps(report))
    _rebind_changed_pass_validation(args, name)


def test_truth_release_binds_independent_pass_artifacts(tmp_path):
    args = _fixture(tmp_path)
    result = freeze(args)
    assert set(result["independence_audit"]["passes"]) == {"A", "B"}
    assert result["candidate_release"]["candidate_sha256"]


def test_task_norm_loader_localizes_an_unmounted_manifest_path(tmp_path):
    panel_root = tmp_path / "panel"
    norms_root = panel_root / "norms"
    norms_root.mkdir(parents=True)
    norms_path = norms_root / "humor.jsonl"
    write_jsonl(
        norms_path,
        [
            {
                "norm_uid": "u1",
                "task": "humor",
                "corpus": "humor_multi",
                "norm": "setup should support the punchline",
            }
        ],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "corpora": {
                    "humor_multi": {
                        "task": "humor",
                        "path": "/unmounted/sk3/norms/humor.jsonl",
                    }
                }
            }
        )
    )
    loaded = _load_task_norms(
        manifest_path, "humor", {"u1"}, panel_root=panel_root
    )
    assert loaded["u1"]["norm"] == "setup should support the punchline"


def test_truth_release_check_only_writes_nothing(tmp_path):
    args = _fixture(tmp_path)
    args.check_only = True
    result = freeze(args)
    assert result["check_only"] is True
    assert result["output"] is None
    assert not Path(args.output).exists()


def test_truth_release_accepts_complete_external_strict_audit_mapping(tmp_path):
    args = _fixture(tmp_path)
    audit = json.loads(Path(args.independence_audit).read_text())
    args.transcript_audit = []
    for name in ("A", "B"):
        pass_root = Path(audit["passes"][name]["root"])
        labels_validation_path = pass_root / "labels.validation.json"
        labels_validation = json.loads(labels_validation_path.read_text())
        transcript_path = Path(labels_validation.pop("transcript_audit")["path"])
        labels_validation_path.write_text(json.dumps(labels_validation))
        args.transcript_audit.append(f"{name}={transcript_path}")
    result = freeze(args)
    assert {row["transcript_audit"]["source"] for row in result["passes"]} == {
        "external_source_workspace_hash_equivalent"
    }


def test_truth_release_binds_every_isolated_labeler_guide(tmp_path):
    args = _fixture(tmp_path)
    second_guide = tmp_path / "ISOLATED_LABELER_NO_DISCOVERY_GUIDE.md"
    second_guide.write_text("strict immutable no-discovery guide\n")
    args.transcript_guide = [args.transcript_guide, str(second_guide)]
    audit = json.loads(Path(args.independence_audit).read_text())
    for name in ("A", "B"):
        pass_root = Path(audit["passes"][name]["root"])
        transcript_path = pass_root / "TRANSCRIPT_ISOLATION_AUDIT.json"
        transcript = json.loads(transcript_path.read_text())
        transcript["guides"].append(_ref(second_guide))
        transcript_path.write_text(json.dumps(transcript))
        labels_validation_path = pass_root / "labels.validation.json"
        labels_validation = json.loads(labels_validation_path.read_text())
        labels_validation["transcript_audit"] = _ref(transcript_path)
        labels_validation_path.write_text(json.dumps(labels_validation))
    result = freeze(args)
    expected = [sha256_file(Path(path)) for path in args.transcript_guide]
    assert all(
        row["transcript_audit"]["guide_sha256s"] == expected
        for row in result["passes"]
    )


def test_truth_release_rejects_unbound_extra_isolated_labeler_guide(tmp_path):
    args = _fixture(tmp_path)
    second_guide = tmp_path / "ISOLATED_LABELER_NO_DISCOVERY_GUIDE.md"
    second_guide.write_text("strict immutable no-discovery guide\n")
    audit = json.loads(Path(args.independence_audit).read_text())
    for name in ("A", "B"):
        pass_root = Path(audit["passes"][name]["root"])
        transcript_path = pass_root / "TRANSCRIPT_ISOLATION_AUDIT.json"
        transcript = json.loads(transcript_path.read_text())
        transcript["guides"].append(_ref(second_guide))
        transcript_path.write_text(json.dumps(transcript))
        labels_validation_path = pass_root / "labels.validation.json"
        labels_validation = json.loads(labels_validation_path.read_text())
        labels_validation["transcript_audit"] = _ref(transcript_path)
        labels_validation_path.write_text(json.dumps(labels_validation))
    with pytest.raises(ValueError, match="strict transcript audit drift"):
        freeze(args)


def test_truth_release_rejects_missing_strict_transcript_pass(tmp_path):
    args = _fixture(tmp_path)
    audit = json.loads(Path(args.independence_audit).read_text())
    pass_root = Path(audit["passes"]["A"]["root"])
    labels_validation_path = pass_root / "labels.validation.json"
    labels_validation = json.loads(labels_validation_path.read_text())
    labels_validation.pop("transcript_audit")
    labels_validation_path.write_text(json.dumps(labels_validation))
    with pytest.raises(ValueError, match="strict transcript audit is required"):
        freeze(args)


def test_truth_release_taxonomy_reuses_canonical_decisions_and_accepts_noise(tmp_path):
    assert TRUTH_DECISIONS == frozenset(DECISIONS)
    assert "NOISE" in TRUTH_DECISIONS
    args = _fixture(tmp_path)
    assert freeze(args)["count"] == 2


def test_truth_release_rejects_unknown_decision_taxonomy(tmp_path):
    args = _fixture(tmp_path)
    truth_path = Path(args.truth)
    rows = [json.loads(line) for line in truth_path.read_text().splitlines()]
    rows[1]["decision"] = "UNKNOWN_NEW_DECISION"
    write_jsonl(truth_path, rows)
    with pytest.raises(ValueError, match="invalid task/role/decision"):
        freeze(args)


def test_truth_release_rejects_duplicate_identity_rows(tmp_path):
    args = _fixture(tmp_path)
    identities_path = Path(args.identities)
    first = identities_path.read_text().splitlines()[0]
    with identities_path.open("a") as handle:
        handle.write(first + "\n")
    freeze_path = Path(args.role_freeze)
    role_freeze = json.loads(freeze_path.read_text())
    role_freeze["outputs"]["identities"]["sha256"] = sha256_file(identities_path)
    freeze_path.write_text(json.dumps(role_freeze))
    with pytest.raises(ValueError, match="duplicate truth UIDs"):
        freeze(args)


def test_truth_release_rejects_self_consistent_forged_truth_and_report(tmp_path):
    args = _fixture(tmp_path)
    truth_path = Path(args.truth)
    rows = [json.loads(line) for line in truth_path.read_text().splitlines()]
    rows[0]["metric_id"] = "a1"
    write_jsonl(truth_path, rows)
    report_path = Path(args.consensus_report)
    report = json.loads(report_path.read_text())
    report["outputs"]["resolved"]["sha256"] = sha256_file(truth_path)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="differs from bound pass consensus"):
        freeze(args)


def test_truth_release_full_source_pass_binding_is_name_independent(tmp_path):
    args = _fixture(tmp_path)
    path = Path(args.consensus_report)
    report = json.loads(path.read_text())
    passes = report["inputs"]["passes"]
    report["inputs"]["passes"] = {"X": passes["A"], "Y": passes["B"]}
    report["rounds"][0]["pass"] = "X"
    report["rounds"][1]["pass"] = "Y"
    truth_path = Path(args.truth)
    truth = [json.loads(line) for line in truth_path.read_text().splitlines()]
    for row in truth:
        row["agreement_sources"] = ["X", "Y"]
    write_jsonl(truth_path, truth)
    report["outputs"]["resolved"]["sha256"] = sha256_file(truth_path)
    path.write_text(json.dumps(report))
    assert freeze(args)["count"] == 2


def test_truth_release_rejects_name_substituted_full_source_pack(tmp_path):
    args = _fixture(tmp_path)
    path = Path(args.consensus_report)
    report = json.loads(path.read_text())
    passes = report["inputs"]["passes"]
    passes["A"]["pack_items_sha256"] = passes["B"]["pack_items_sha256"]
    report["inputs"]["passes"] = {"X": passes["A"], "Y": passes["B"]}
    report["rounds"][0]["pass"] = "X"
    report["rounds"][1]["pass"] = "Y"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="differ from independence audit"):
        freeze(args)


def test_truth_release_uses_rounds_as_authoritative_pass_order(tmp_path):
    args = _fixture(tmp_path)
    path = Path(args.consensus_report)
    report = json.loads(path.read_text())
    passes = report["inputs"]["passes"]
    report["inputs"]["passes"] = {"B": passes["B"], "A": passes["A"]}
    path.write_text(json.dumps(report))
    assert freeze(args)["consensus_replay"]["pass_order"] == ["A", "B"]


def test_truth_release_rejects_self_consistent_altered_norm_same_uid(tmp_path):
    args = _fixture(tmp_path)
    audit = json.loads(Path(args.independence_audit).read_text())
    items_path = Path(audit["passes"]["A"]["root"]) / "items.jsonl"
    rows = [json.loads(line) for line in items_path.read_text().splitlines()]
    rows[0]["norm"] = "altered content under the same immutable UID"
    write_jsonl(items_path, rows)
    _rebind_changed_pass_items(args)
    with pytest.raises(ValueError, match="differs from canonical hydrated content"):
        freeze(args)


def test_resolver_lineage_binds_initial_labels_and_prior_unresolved(tmp_path):
    candidate = tmp_path / "label_pack" / "validation.json"
    candidate.parent.mkdir()
    candidate.write_text("{}\n")
    labels_a = tmp_path / "independent_pass_A" / "labels.validated.jsonl"
    labels_b = tmp_path / "independent_pass_B" / "labels.validated.jsonl"
    labels_a.parent.mkdir()
    labels_b.parent.mkdir()
    write_jsonl(labels_a, [{"norm_uid": "u1"}])
    write_jsonl(labels_b, [{"norm_uid": "u1"}])
    initial = {
        (labels_a.resolve(), sha256_file(labels_a)),
        (labels_b.resolve(), sha256_file(labels_b)),
    }
    source_ref = _ref(candidate)
    semantic = {
        "schema_version": "silver-match-v3-semantic-resolver-pack-v1",
        "task": "code-review",
        "truth_hidden": True,
        "prior_decisions_and_metric_ids_hidden": True,
        "selection_rule": {
            "mode": "exact_disagreements_only",
            "all_exact_strict_key_mismatches": True,
        },
        "inputs": {
            "source_pack_validation": source_ref,
            "semantic_labels": _ref(labels_a),
            "strict_key": _ref(labels_b),
        },
    }
    _verify_resolver_lineage(
        name="R1",
        validation=semantic,
        panel_root=tmp_path,
        candidate_release_path=candidate,
        candidate_release_sha=sha256_file(candidate),
        current_unresolved={"u1"},
        initial_label_refs=initial,
        task="code-review",
    )

    unresolved = tmp_path / "consensus_r1" / "unresolved.jsonl"
    unresolved.parent.mkdir()
    write_jsonl(unresolved, [{"norm_uid": "u1", "task": "code-review"}])
    exact = {
        "schema_version": "silver-match-v3-exact-unresolved-resolver-pack-v1",
        "task": "code-review",
        "truth_hidden": True,
        "prior_decisions_and_metric_ids_hidden": True,
        "selection_rule": "all_and_only_current_exact_consensus_unresolved_uids",
        "inputs": {
            "source_pack_validation": source_ref,
            "unresolved": _ref(unresolved),
        },
    }
    _verify_resolver_lineage(
        name="R2",
        validation=exact,
        panel_root=tmp_path,
        candidate_release_path=candidate,
        candidate_release_sha=sha256_file(candidate),
        current_unresolved={"u1"},
        initial_label_refs=initial,
        task="code-review",
    )
    with pytest.raises(ValueError, match="prior-unresolved lineage drift"):
        _verify_resolver_lineage(
            name="R2",
            validation=exact,
            panel_root=tmp_path,
            candidate_release_path=candidate,
            candidate_release_sha=sha256_file(candidate),
            current_unresolved={"u2"},
            initial_label_refs=initial,
            task="code-review",
        )


def test_truth_release_rejects_unrelated_strict_pass_audit(tmp_path):
    args = _fixture(tmp_path)
    audit = json.loads(Path(args.independence_audit).read_text())
    pass_root = Path(audit["passes"]["A"]["root"])
    transcript_path = pass_root / "TRANSCRIPT_ISOLATION_AUDIT.json"
    transcript = json.loads(transcript_path.read_text())
    transcript["items"]["sha256"] = "0" * 64
    transcript_path.write_text(json.dumps(transcript))
    labels_validation_path = pass_root / "labels.validation.json"
    labels_validation = json.loads(labels_validation_path.read_text())
    labels_validation["transcript_audit"]["sha256"] = sha256_file(transcript_path)
    labels_validation_path.write_text(json.dumps(labels_validation))
    with pytest.raises(ValueError, match="strict transcript audit drift"):
        freeze(args)


def test_truth_release_rejects_strict_chunk_file_drift(tmp_path):
    args = _fixture(tmp_path)
    audit = json.loads(Path(args.independence_audit).read_text())
    pass_root = Path(audit["passes"]["A"]["root"])
    with (pass_root / "chunks" / "part-000.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="chunk/raw/log binding drift"):
        freeze(args)


def test_truth_release_rejects_invalid_current_bank_truth_leaf(tmp_path):
    args = _fixture(tmp_path)
    truth_path = Path(args.truth)
    rows = [json.loads(line) for line in truth_path.read_text().splitlines()]
    rows[0]["metric_id"] = "outside-bank"
    write_jsonl(truth_path, rows)
    with pytest.raises(ValueError, match="current-bank leaf"):
        freeze(args)


@pytest.mark.parametrize("field", ["validation_sha256", "bank_sha256", "items_sha256"])
def test_truth_release_rejects_independence_artifact_hash_drift(tmp_path, field):
    args = _fixture(tmp_path)
    path = Path(args.independence_audit)
    audit = json.loads(path.read_text())
    audit["passes"]["A"][field] = "0" * 64
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="artifacts are missing or drifted"):
        freeze(args)


@pytest.mark.parametrize(
    "kind", ["source_pack", "manifest", "identities", "identity_freeze"]
)
def test_truth_release_rejects_independent_pass_source_provenance_drift(tmp_path, kind):
    args = _fixture(tmp_path)
    audit = json.loads(Path(args.independence_audit).read_text())
    validation_path = Path(audit["passes"]["A"]["root"]) / "validation.json"
    validation = json.loads(validation_path.read_text())
    if kind == "source_pack":
        validation["source_pack"]["validation_sha256"] = "0" * 64
    else:
        validation["inputs"][kind]["sha256"] = "0" * 64
    validation_path.write_text(json.dumps(validation))
    _rebind_changed_pass_validation(args)
    with pytest.raises(ValueError, match="validation provenance drift"):
        freeze(args)
