import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.aggregate_nemotron_ce_seed_consensus import (
    CONSENSUS_REPORT_SCHEMA,
    CONSENSUS_SCHEMA,
)
from scripts.tools.silver_match_v3.audit_final_outputs import audit_outputs
from scripts.tools.silver_match_v3.common import sha256_file, write_jsonl
from scripts.tools.silver_match_v3.freeze_postinference_analysis_release import (
    freeze_release,
    parse_final_bindings,
    resolve_task,
)
from scripts.tools.silver_match_v3.silver_mi_validation_v3 import run_validation


def _dump(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _fixture(tmp_path: Path):
    bank = _dump(
        tmp_path / "bank.json",
        {
            "task": "humor",
            "source_sha256": "bank-sha",
            "metrics": [
                {
                    "metric_id": f"a{index}",
                    "metric_index": index,
                    "name": f"metric {index}",
                    "leaf_count": index + 1,
                }
                for index in range(4)
            ],
        },
    )
    norms = tmp_path / "norms.jsonl"
    canonical = [
        {
            "norm_uid": "u1",
            "row": 0,
            "task": "humor",
            "corpus": "humor-corpus",
            "source_group": "g1",
            "split": "production",
            "norm": "criterion one",
        },
        {
            "norm_uid": "u2",
            "row": 1,
            "task": "humor",
            "corpus": "humor-corpus",
            "source_group": "g2",
            "split": "production",
            "norm": "criterion two",
        },
    ]
    write_jsonl(norms, canonical)
    manifest = _dump(
        tmp_path / "manifest.json",
        {
            "banks": {
                "humor": {
                    "path": str(bank),
                    "source_sha256": "bank-sha",
                    "count": 4,
                }
            },
            "corpora": {
                "humor-corpus": {
                    "path": str(norms),
                    "task": "humor",
                    "count": 2,
                }
            },
        },
    )
    plan = _dump(
        tmp_path / "plan.json",
        {
            "status": "FROZEN_READY_FOR_UNLABELED_PRODUCTION",
            "task": "humor",
            "manifest": {"sha256": sha256_file(manifest)},
            "bank_source_sha256": "bank-sha",
        },
    )

    def state(metric, passes):
        return {
            "top_metric_id": metric,
            "top_predicted_relation": "EXACT" if passes else "REJECT",
            "top_exact_probability": 0.9 if passes else 0.1,
            "second_exact_probability": 0.1,
            "top_exact_margin": 0.8 if passes else 0.0,
            "score_threshold": 0.7,
            "top_margin_threshold": 0.2,
            "passes_frozen_gate": passes,
            "has_family_argmax_candidate": False,
            "all_candidates_reject": not passes,
        }

    ce = tmp_path / "ce.jsonl"
    ce_rows = [
        {
            "schema_version": CONSENSUS_SCHEMA,
            "norm_uid": "u1",
            "task": "humor",
            "corpus": "humor-corpus",
            "source_group": "g1",
            "split": "production",
            "decision": "MATCH",
            "routing_category": "MATCH",
            "automatic_match": True,
            "metric_id": "a1",
            "candidate_count": 2,
            "candidates": [{"metric_id": "a1"}, {"metric_id": "a2"}],
            "seed_decisions": {"s1": state("a1", True), "s2": state("a1", True)},
            "provisional_routing_only": False,
            "human_abstention_subtype_assigned": False,
        },
        {
            "schema_version": CONSENSUS_SCHEMA,
            "norm_uid": "u2",
            "task": "humor",
            "corpus": "humor-corpus",
            "source_group": "g2",
            "split": "production",
            "decision": "ROUTE_TO_ADJUDICATION",
            "routing_category": "CE_REJECT_BOTH",
            "automatic_match": False,
            "metric_id": None,
            "candidate_count": 2,
            "candidates": [{"metric_id": "a2"}, {"metric_id": "a3"}],
            "seed_decisions": {"s1": state("a2", False), "s2": state("a2", False)},
            "provisional_routing_only": True,
            "human_abstention_subtype_assigned": False,
        },
    ]
    write_jsonl(ce, ce_rows)
    ce_report = _dump(
        tmp_path / "ce.report.json",
        {
            "schema_version": CONSENSUS_REPORT_SCHEMA,
            "status": "COMPLETE",
            "output": str(ce),
            "output_sha256": sha256_file(ce),
            "norm_count": 2,
            "validation": {
                "all_norms_preserved": True,
                "all_thresholds_from_checkpoint_dev": True,
                "seed_norm_candidate_source_split_universes_identical": True,
                "test_threshold_tuning_performed": False,
            },
        },
    )
    gemma = tmp_path / "gemma.jsonl"
    gemma_rows = [
        {
            "norm_uid": "u1",
            "task": "humor",
            "corpus": "humor-corpus",
            "source_group": "g1",
            "decision": "MATCH",
            "metric_id": "a1",
            "confidence": "high",
            "candidate_bank_source_sha256": "bank-sha",
        },
        {
            "norm_uid": "u2",
            "task": "humor",
            "corpus": "humor-corpus",
            "source_group": "g2",
            "decision": "NO_CANDIDATE_FITS",
            "metric_id": None,
            "confidence": "high",
            "candidate_bank_source_sha256": "bank-sha",
        },
    ]
    write_jsonl(gemma, gemma_rows)
    gemma_report = _dump(
        tmp_path / "gemma.report.json",
        {
            "status": "SELECTED_FOR_PRODUCTION",
            "output": {"path": str(gemma), "sha256": sha256_file(gemma)},
        },
    )
    final = tmp_path / "final.jsonl"
    final_rows = [
        {
            "norm_uid": "u1",
            "row": 0,
            "task": "humor",
            "corpus": "humor-corpus",
            "source_group": "g1",
            "decision": "MATCH",
            "metric_id": "a1",
            "confidence": "high",
            "verification_status": "two_seed_same_leaf_dev_gated",
            "rescue_status": "NOT_APPLICABLE_PRIMARY_MATCH",
            "bank_source_sha256": "bank-sha",
        },
        {
            "norm_uid": "u2",
            "row": 1,
            "task": "humor",
            "corpus": "humor-corpus",
            "source_group": "g2",
            "decision": "NO_CANDIDATE_FITS",
            "metric_id": None,
            "confidence": "high",
            "verification_status": "rescued_repeated_full_bank_typed_abstention",
            "rescue_status": "EXHAUSTIVE_RESCUE_RESOLVED",
            "bank_source_sha256": "bank-sha",
            "pre_rescue": {
                "decision": "NO_CANDIDATE_FITS",
                "metric_id": None,
            },
            "rescue_resolution": {"coverage_repeats": 2},
        },
    ]
    write_jsonl(final, final_rows)
    final_audit = _dump(
        tmp_path / "final.audit.json",
        audit_outputs(manifest, [final], tasks={"humor"}, corpora={"humor-corpus"}),
    )
    rescue_report = _dump(
        tmp_path / "rescue.report.json",
        {
            "schema_version": "silver-match-v3-rescue-merge-v1",
            "strict_production": True,
            "unresolved_rows": 0,
            "output": str(final),
            "output_sha256": sha256_file(final),
        },
    )
    exclusion = tmp_path / "exclude.jsonl"
    write_jsonl(exclusion, [{"norm_uid": "u2"}])
    risk = _dump(tmp_path / "risk.json", {"placeholder": True})
    certificate = _dump(
        tmp_path / "mi.json",
        {
            "schema_version": "silver-match-v3-extracted-mi-certificate-v1",
            "task": "humor",
            "table": [
                {
                    "file": f"x_metric{index}_z.json",
                    "name": f"metric {index}",
                    "opt_omega_bits": index + 0.1,
                    "H_M": index + 0.2,
                }
                for index in range(4)
            ],
        },
    )
    return {
        "manifest_path": manifest,
        "task": "humor",
        "plan_path": plan,
        "final_paths": {"humor-corpus": final},
        "final_audit_path": final_audit,
        "ce_path": ce,
        "ce_report_path": ce_report,
        "gemma_path": gemma,
        "gemma_report_path": gemma_report,
        "rescue_report_path": rescue_report,
        "risk_release_path": risk,
        "analysis_exclusion_paths": [exclusion],
        "mi_certificate_path": certificate,
        "expected_rows": 2,
        "expected_bank_metrics": 4,
    }


def _risk(kwargs):
    exclusion = kwargs["analysis_exclusion_paths"][0].resolve()
    return {
        "task": kwargs["task"],
        "status": "PASS",
        "complete": True,
        "production_final_blind_audit": True,
        "gates": {"precision": True, "false_abstention": True},
        "thresholds": {"alpha_one_sided": 0.05},
        "analysis_exclusions": {str(exclusion): sha256_file(exclusion)},
        "match_audit": {"statistics": {"audited_rows": 60}},
        "abstention_audit": {"statistics": {"audited_rows": 60}},
    }


def _multicorpus_fixture(tmp_path: Path, *, task: str, corpus_count: int):
    bank = _dump(
        tmp_path / "bank.json",
        {
            "task": task,
            "source_sha256": f"{task}-bank-sha",
            "metrics": [
                {
                    "metric_id": f"a{index}",
                    "metric_index": index,
                    "name": f"{task} metric {index}",
                    "leaf_count": index + 1,
                }
                for index in range(4)
            ],
        },
    )
    # Deliberately non-lexical names make preservation of manifest insertion
    # order observable.
    corpora = [f"{task}-part-{index:02d}" for index in reversed(range(corpus_count))]
    corpus_meta = {}
    canonical = []
    final_paths = {}
    final_by_corpus = {}
    ce_rows = []
    gemma_rows = []

    def state(metric, passes):
        return {
            "top_metric_id": metric,
            "top_predicted_relation": "EXACT" if passes else "REJECT",
            "top_exact_probability": 0.9 if passes else 0.1,
            "second_exact_probability": 0.1,
            "top_exact_margin": 0.8 if passes else 0.0,
            "score_threshold": 0.7,
            "top_margin_threshold": 0.2,
            "passes_frozen_gate": passes,
            "has_family_argmax_candidate": False,
            "all_candidates_reject": not passes,
        }

    for position, corpus in enumerate(corpora):
        uid = f"u{position:02d}"
        source_group = f"g{position:02d}"
        norm = {
            "norm_uid": uid,
            "row": 100 + position,
            "task": task,
            "corpus": corpus,
            "source_group": source_group,
            "split": "production",
            "norm": f"explicit criterion {position}",
        }
        norm_path = tmp_path / f"{corpus}.norms.jsonl"
        write_jsonl(norm_path, [norm])
        corpus_meta[corpus] = {"path": str(norm_path), "task": task, "count": 1}
        canonical.append(norm)
        automatic = position == 0
        ce_rows.append(
            {
                "schema_version": CONSENSUS_SCHEMA,
                "norm_uid": uid,
                "task": task,
                "corpus": corpus,
                "source_group": source_group,
                "split": "production",
                "decision": "MATCH" if automatic else "ROUTE_TO_ADJUDICATION",
                "routing_category": "MATCH" if automatic else "CE_REJECT_BOTH",
                "automatic_match": automatic,
                "metric_id": "a1" if automatic else None,
                "candidate_count": 2,
                "candidates": [{"metric_id": "a1"}, {"metric_id": "a2"}],
                "seed_decisions": {
                    "s1": state("a1", automatic),
                    "s2": state("a1", automatic),
                },
                "provisional_routing_only": not automatic,
                "human_abstention_subtype_assigned": False,
            }
        )
        gemma_decision = "MATCH" if automatic else "NO_CANDIDATE_FITS"
        gemma_rows.append(
            {
                "norm_uid": uid,
                "task": task,
                "corpus": corpus,
                "source_group": source_group,
                "decision": gemma_decision,
                "metric_id": "a1" if automatic else None,
                "confidence": "high",
                "candidate_bank_source_sha256": f"{task}-bank-sha",
            }
        )
        final = {
            "norm_uid": uid,
            "row": 100 + position,
            "task": task,
            "corpus": corpus,
            "source_group": source_group,
            "decision": "MATCH" if automatic else "NO_CANDIDATE_FITS",
            "metric_id": "a1" if automatic else None,
            "confidence": "high",
            "verification_status": (
                "two_seed_same_leaf_dev_gated"
                if automatic
                else "rescued_repeated_full_bank_typed_abstention"
            ),
            "rescue_status": (
                "NOT_APPLICABLE_PRIMARY_MATCH"
                if automatic
                else "EXHAUSTIVE_RESCUE_RESOLVED"
            ),
            "bank_source_sha256": f"{task}-bank-sha",
        }
        if not automatic:
            final["pre_rescue"] = {
                "decision": "NO_CANDIDATE_FITS",
                "metric_id": None,
            }
            final["rescue_resolution"] = {"coverage_repeats": 2}
        final_path = tmp_path / f"{corpus}.final.jsonl"
        write_jsonl(final_path, [final])
        final_paths[corpus] = final_path
        final_by_corpus[corpus] = final

    manifest = _dump(
        tmp_path / "manifest.json",
        {
            "banks": {
                task: {
                    "path": str(bank),
                    "source_sha256": f"{task}-bank-sha",
                    "count": 4,
                }
            },
            "corpora": corpus_meta,
        },
    )
    corpora = list(json.loads(manifest.read_text())["corpora"])
    plan = _dump(
        tmp_path / "plan.json",
        {
            "status": "FROZEN_READY_FOR_UNLABELED_PRODUCTION",
            "task": task,
            "manifest": {"sha256": sha256_file(manifest)},
            "bank_source_sha256": f"{task}-bank-sha",
        },
    )
    ce = tmp_path / "ce.jsonl"
    write_jsonl(ce, list(reversed(ce_rows)))
    ce_report = _dump(
        tmp_path / "ce.report.json",
        {
            "schema_version": CONSENSUS_REPORT_SCHEMA,
            "status": "COMPLETE",
            "output": str(ce),
            "output_sha256": sha256_file(ce),
            "norm_count": corpus_count,
            "validation": {
                "all_norms_preserved": True,
                "all_thresholds_from_checkpoint_dev": True,
                "seed_norm_candidate_source_split_universes_identical": True,
                "test_threshold_tuning_performed": False,
            },
        },
    )
    gemma = tmp_path / "gemma.jsonl"
    write_jsonl(gemma, list(reversed(gemma_rows)))
    gemma_report = _dump(
        tmp_path / "gemma.report.json",
        {
            "status": "SELECTED_FOR_PRODUCTION",
            "output": {"path": str(gemma), "sha256": sha256_file(gemma)},
        },
    )
    merged = tmp_path / "task.final.merged.jsonl"
    write_jsonl(merged, [final_by_corpus[corpus] for corpus in corpora])
    rescue_report = _dump(
        tmp_path / "rescue.report.json",
        {
            "schema_version": "silver-match-v3-rescue-merge-v1",
            "strict_production": True,
            "unresolved_rows": 0,
            "output": str(merged),
            "output_sha256": sha256_file(merged),
        },
    )
    final_audit = _dump(
        tmp_path / "final.audit.json",
        audit_outputs(
            manifest,
            [final_paths[corpus] for corpus in corpora],
            tasks={task},
            corpora=set(corpora),
        ),
    )
    exclusion = tmp_path / "exclude.jsonl"
    write_jsonl(exclusion, [{"norm_uid": canonical[-1]["norm_uid"]}])
    risk = _dump(tmp_path / "risk.json", {"placeholder": True})
    certificate = _dump(
        tmp_path / "mi.json",
        {
            "schema_version": "silver-match-v3-extracted-mi-certificate-v1",
            "task": task,
            "table": [
                {
                    "file": f"x_metric{index}_z.json",
                    "name": f"{task} metric {index}",
                    "opt_omega_bits": index + 0.1,
                    "H_M": index + 0.2,
                }
                for index in range(4)
            ],
        },
    )
    return {
        "manifest_path": manifest,
        "task": task,
        "plan_path": plan,
        "final_paths": final_paths,
        "final_audit_path": final_audit,
        "ce_path": ce,
        "ce_report_path": ce_report,
        "gemma_path": gemma,
        "gemma_report_path": gemma_report,
        "rescue_report_path": rescue_report,
        "risk_release_path": risk,
        "analysis_exclusion_paths": [exclusion],
        "mi_certificate_path": certificate,
        "merged_final_path": merged,
        "expected_rows": corpus_count,
        "expected_bank_metrics": 4,
        "corpora": corpora,
    }


def test_freezes_complete_nine_status_release_and_mi_handoff(tmp_path, monkeypatch):
    kwargs = _fixture(tmp_path)
    monkeypatch.setattr(
        "scripts.tools.silver_match_v3.freeze_postinference_analysis_release.verify_task_final_risk_release",
        lambda *args, **kw: _risk(kwargs),
    )
    release, handoff = freeze_release(**kwargs)
    assert release["status"] == "TASK_FROZEN_ANALYSIS_READY"
    assert release["rates"]["denominator_all_canonical_norms"] == 2
    assert set(release["rates"]["nine_status_counts"]) == {
        "MATCH",
        "MATCH_FAMILY_ONLY",
        "NO_EXPLICIT_CRITERION",
        "CONTEXT_NEEDED",
        "GENERIC_VERDICT",
        "NO_CANDIDATE_FITS",
        "NOISE",
        "UNSTABLE_MATCH",
        "INVALID_OUTPUT",
    }
    assert release["rates"]["match_rate"] == 0.5
    assert release["rates"]["rescue_rate"] == 0.5
    assert release["coverage"]["mi_certificate_metrics"] == 4
    assert handoff["denominators"]["analysis_eligible_rows"] == 1
    assert handoff["command_module"].endswith("silver_mi_validation_v3")
    release_path = _dump(tmp_path / "analysis.release.json", release)
    result = run_validation(
        release_path=release_path,
        certificate_path=kwargs["mi_certificate_path"],
        n_permutations=2,
        n_bootstrap=2,
        seed=7,
    )
    assert result["rows"] == 2
    assert result["analysis_eligible_rows"] == 1
    assert result["decision_counts"] == {"MATCH": 1, "NO_CANDIDATE_FITS": 1}


def test_rejects_ce_match_when_one_seed_did_not_pass_gate(tmp_path, monkeypatch):
    kwargs = _fixture(tmp_path)
    rows = [json.loads(line) for line in kwargs["ce_path"].read_text().splitlines()]
    rows[0]["seed_decisions"]["s2"]["passes_frozen_gate"] = False
    write_jsonl(kwargs["ce_path"], rows)
    report = json.loads(kwargs["ce_report_path"].read_text())
    report["output_sha256"] = sha256_file(kwargs["ce_path"])
    _dump(kwargs["ce_report_path"], report)
    monkeypatch.setattr(
        "scripts.tools.silver_match_v3.freeze_postinference_analysis_release.verify_task_final_risk_release",
        lambda *args, **kw: _risk(kwargs),
    )
    with pytest.raises(ValueError, match="same-leaf/two-gate"):
        freeze_release(**kwargs)


def test_rejects_non_ce_row_that_bypassed_repeated_full_bank_rescue(
    tmp_path, monkeypatch
):
    kwargs = _fixture(tmp_path)
    final_path = kwargs["final_paths"]["humor-corpus"]
    rows = [json.loads(line) for line in final_path.read_text().splitlines()]
    rows[1]["rescue_status"] = "NOT_APPLICABLE"
    write_jsonl(final_path, rows)
    # Keep the outer artifacts honestly rebound so the semantic check is reached.
    _dump(
        kwargs["final_audit_path"],
        audit_outputs(
            kwargs["manifest_path"],
            [final_path],
            tasks={"humor"},
            corpora={"humor-corpus"},
        ),
    )
    report = json.loads(kwargs["rescue_report_path"].read_text())
    report["output_sha256"] = sha256_file(final_path)
    _dump(kwargs["rescue_report_path"], report)
    monkeypatch.setattr(
        "scripts.tools.silver_match_v3.freeze_postinference_analysis_release.verify_task_final_risk_release",
        lambda *args, **kw: _risk(kwargs),
    )
    with pytest.raises(ValueError, match="bypassed exhaustive rescue"):
        freeze_release(**kwargs)


def test_rejects_risk_release_without_all_gates_passing(tmp_path, monkeypatch):
    kwargs = _fixture(tmp_path)
    failed = _risk(kwargs)
    failed["status"] = "FAIL"
    failed["complete"] = False
    failed["gates"]["false_abstention"] = False
    monkeypatch.setattr(
        "scripts.tools.silver_match_v3.freeze_postinference_analysis_release.verify_task_final_risk_release",
        lambda *args, **kw: failed,
    )
    with pytest.raises(ValueError, match="risk release.*PASS"):
        freeze_release(**kwargs)


def test_code_two_corpora_freezes_one_task_handoff(tmp_path, monkeypatch):
    kwargs = _multicorpus_fixture(tmp_path, task="code", corpus_count=2)
    monkeypatch.setattr(
        "scripts.tools.silver_match_v3.freeze_postinference_analysis_release.verify_task_final_risk_release",
        lambda *args, **kw: _risk(kwargs),
    )
    release, handoff = freeze_release(
        **{key: value for key, value in kwargs.items() if key != "corpora"}
    )
    assert release["corpora"] == kwargs["corpora"]
    assert len(release["final_outputs"]) == 2
    assert release["coverage"]["canonical_rows_by_corpus"] == {
        corpus: 1 for corpus in kwargs["corpora"]
    }
    assert release["rates"]["rescue_rate"] == 0.5
    assert release["rates"]["macro_over_corpora"]["groups"] == 2
    assert handoff["denominators"]["canonical_rows"] == 2
    assert handoff["denominators"]["corpora"] == 2


def test_legal_ten_corpora_preserves_manifest_order_and_macro_rates(
    tmp_path, monkeypatch
):
    kwargs = _multicorpus_fixture(tmp_path, task="legal", corpus_count=10)
    monkeypatch.setattr(
        "scripts.tools.silver_match_v3.freeze_postinference_analysis_release.verify_task_final_risk_release",
        lambda *args, **kw: _risk(kwargs),
    )
    release, handoff = freeze_release(
        **{key: value for key, value in kwargs.items() if key != "corpora"}
    )
    assert release["corpora"] == kwargs["corpora"]
    assert release["coverage"]["manifest_corpus_order"] == kwargs["corpora"]
    assert release["coverage"]["corpora_audited"] == 10
    assert release["rates"]["macro_over_corpora"]["groups"] == 10
    assert release["rates"]["rescue_count"] == 9
    assert release["rates"]["rescue_rate"] == 0.9
    assert handoff["denominators"]["canonical_rows"] == 10
    assert list(handoff["denominators"]["canonical_rows_by_corpus"]) == kwargs[
        "corpora"
    ]


def test_multicorpus_final_bindings_fail_on_missing_or_duplicate_corpus(tmp_path):
    kwargs = _multicorpus_fixture(tmp_path, task="code", corpus_count=2)
    corpora = kwargs["corpora"]
    finals = kwargs["final_paths"]
    with pytest.raises(ValueError, match="coverage mismatch"):
        parse_final_bindings(
            [f"{corpora[0]}={finals[corpora[0]]}"],
            kwargs["manifest_path"],
            "code",
        )
    with pytest.raises(ValueError, match="duplicate final corpus"):
        parse_final_bindings(
            [
                f"{corpora[0]}={finals[corpora[0]]}",
                f"{corpora[0]}={finals[corpora[0]]}",
            ],
            kwargs["manifest_path"],
            "code",
        )


def test_task_is_derived_only_for_single_task_manifest(tmp_path):
    kwargs = _multicorpus_fixture(tmp_path, task="code", corpus_count=2)
    assert resolve_task(kwargs["manifest_path"], None) == "code"
    payload = json.loads(kwargs["manifest_path"].read_text())
    payload["banks"]["legal"] = payload["banks"]["code"]
    _dump(kwargs["manifest_path"], payload)
    with pytest.raises(ValueError, match="--task is required"):
        resolve_task(kwargs["manifest_path"], None)
