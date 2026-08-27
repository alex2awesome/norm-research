from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_science_fullarticle_runner import (
    BLOCKER_SCHEMA,
    BODY_HEADER,
    DEFAULT_BASE,
    DEFAULT_CANONICAL_ITEMS,
    DEFAULT_FIDELITY,
    DEFAULT_ITEMS,
    DEFAULT_SEED,
    DEFAULT_SOURCE,
    EXECUTION_SCHEMA,
    GATE_SCHEMA,
    ITEM_SCHEMA,
    ScienceExecutionError,
    _concise_verifier_result,
    _summarize_execution,
    article_ctext,
    build_additive_items,
    build_canonical_representation_blocker,
    build_heldout_execution,
    build_train_gate,
    execute_items,
    load_outcome_blind_source,
    parse_article_ctext,
    validate_items,
)
from methods.metric_seam.science_claims_v2.core_relation_strict import verify_document


BLOCKER = DEFAULT_BASE / "peer_review_science_canonical_representation_blocker_v1.json"
TRAIN = DEFAULT_BASE / "peer_review_science_fullarticle_compiler_train_v1.json"
GATE = DEFAULT_BASE / "peer_review_science_fullarticle_train_gate_v1.json"
HELDOUT = DEFAULT_BASE / "peer_review_science_fullarticle_heldout_pre_reference_v1.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs():
    return _load(DEFAULT_SEED), _load(DEFAULT_FIDELITY)


def test_source_outcome_is_masked_before_decode_and_cannot_change_projection(tmp_path):
    template = (
        '{{"paper_id": "paper-1", "y": {outcome}, '
        '"abstract": "We show a 12% gain.", "body": "Table 2 shows a 12% gain."}}\n'
    )
    zero = tmp_path / "zero.jsonl"
    one = tmp_path / "one.jsonl"
    zero.write_text(template.format(outcome=0), encoding="utf-8")
    one.write_text(template.format(outcome=1), encoding="utf-8")
    expected = [
        {
            "paper_id": "paper-1",
            "abstract": "We show a 12% gain.",
            "body": "Table 2 shows a 12% gain.",
        }
    ]
    assert load_outcome_blind_source(zero) == expected
    assert load_outcome_blind_source(one) == expected


def test_source_reader_fails_closed_if_outcome_field_is_not_maskable(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"paper_id": "paper-1", "abstract": "a", "body": "b"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ScienceExecutionError, match="outcome-mask contract"):
        load_outcome_blind_source(path)


def test_canonical_blocker_quantifies_exact_join_without_forcing_body_evidence():
    source = load_outcome_blind_source(DEFAULT_SOURCE)
    result = build_canonical_representation_blocker(
        _load(DEFAULT_CANONICAL_ITEMS / "compiler_train.json"),
        _load(DEFAULT_CANONICAL_ITEMS / "sealed_heldout.json"),
        source,
    )
    assert result == _load(BLOCKER)
    assert result["schema"] == BLOCKER_SCHEMA
    assert result["status"] == "canonical_execution_blocked_by_representation_mismatch"
    assert result["coverage_audit"]["compiler_train"]["join_state_counts"] == {
        "ambiguous_exact_abstract_join": 0,
        "exact_join_missing_body": 4,
        "exact_join_with_body": 2,
        "no_exact_abstract_join": 144,
    }
    assert result["coverage_audit"]["sealed_heldout"]["join_state_counts"] == {
        "ambiguous_exact_abstract_join": 0,
        "exact_join_missing_body": 2,
        "exact_join_with_body": 4,
        "no_exact_abstract_join": 144,
    }
    assert result["coverage_audit"]["pooled"] == {
        "n_items": 300,
        "n_exact_abstract_joins": 12,
        "n_exact_joins_with_nonempty_body": 6,
    }
    assert result["execution"]["performed"] is False
    assert result["disposition"]["forced_join_permitted"] is False


def test_additive_split_is_outcome_blind_same_bytes_and_checked_in_exactly():
    source = load_outcome_blind_source(DEFAULT_SOURCE)
    manifest, train, heldout = build_additive_items(source)
    observed_manifest = _load(DEFAULT_ITEMS / "manifest.json")
    observed_train = _load(DEFAULT_ITEMS / "compiler_train.json")
    observed_heldout = _load(DEFAULT_ITEMS / "sealed_heldout.json")
    assert train == observed_train
    assert heldout == observed_heldout
    for key in ("schema", "status", "task", "representation", "selection", "policy"):
        assert manifest[key] == observed_manifest[key]
    assert observed_manifest["schema"] == ITEM_SCHEMA
    assert observed_manifest["selection"] == {
        "compiler_train_body_nonempty_n": 124,
        "compiler_train_n": 150,
        "conditioned_on_body_availability": False,
        "current_stage_outcome_blind": True,
        "outcome_or_reference_values_used": False,
        "rule": (
            "stable SHA-256 order of the permitted abstract+body projection; first 300; "
            "first 150 compiler-train and next 150 sealed heldout"
        ),
        "salt": "metric-seam-science-fullarticle-shared-items-v1",
        "sealed_heldout_body_nonempty_n": 111,
        "sealed_heldout_n": 150,
        "selected_body_nonempty_n": 235,
        "selected_n": 300,
        "source_rows_scanned": 2400,
        "unique_projected_rows": 2400,
        "upstream_2400_paper_corpus_historically_outcome_stratified": True,
    }
    assert observed_manifest["representation"][
        "same_ctext_bytes_required_for_future_prompt_and_code"
    ] is True
    assert observed_manifest["representation"]["complete_pdf_claimed"] is False
    assert observed_manifest["comparability"]["canonical_hierarchy_items"] is False
    assert observed_manifest["source"]["outcome_value_masked_before_json_decode"] is True
    assert "balanced accept/reject strata" in observed_manifest["source"][
        "upstream_sampling_provenance"
    ]
    assert observed_manifest["policy"] == {
        "accelerators_used": False,
        "compiler_receives_heldout_text": False,
        "external_supervision_used_by_this_split_builder": False,
        "models_or_apis_called": False,
        "outcome_fields_emitted": False,
        "reference_fields_emitted": False,
    }
    assert all(set(item) == {"item_key", "ctext"} for item in train + heldout)
    assert len({item["ctext"] for item in train + heldout}) == 300
    assert all(BODY_HEADER in item["ctext"] for item in train + heldout)


def test_article_representation_round_trips_and_rejects_extra_item_fields():
    ctext = article_ctext(" Abstract. ", " Body. ")
    assert parse_article_ctext(ctext) == ("Abstract.", "Body.")
    with pytest.raises(ScienceExecutionError, match="exactly item_key and ctext"):
        validate_items([{"item_key": "x", "ctext": ctext, "y": 1}])


def test_train_artifact_retains_three_states_and_only_six_audited_relations():
    result = _load(TRAIN)
    assert result["schema"] == EXECUTION_SCHEMA
    assert result["phase"] == "compiler_train"
    assert result["summary"] == {
        "measured_coverage": 0.786667,
        "measured_verifier_status_counts": {
            "evidence_link": 14,
            "insufficient": 97,
            "supported": 7,
        },
        "n_distinct_measured_verifier_statuses": 3,
        "n_items_with_relation_certificate": 7,
        "n_mapping_item_applications": 900,
        "n_relation_certificates": 7,
        "n_relation_mappings": 6,
        "n_unique_item_executions": 150,
        "three_state_totals_mapping_applications": {
            "abstained": 192,
            "failed": 0,
            "measured": 708,
        },
        "three_state_totals_unique_items": {
            "abstained": 32,
            "failed": 0,
            "measured": 118,
        },
    }
    assert len(result["relation_mappings"]) == 6
    assert {row["effective_code_depth"] for row in result["relation_mappings"]} == {3}
    assert set(result["execution_policy"].values()) == {
        False,
        True,
    }
    for field in (
        "reference_values_loaded",
        "outcome_values_loaded",
        "prompt_or_reconstruction_outputs_loaded",
        "external_supervision_used",
        "models_or_apis_called",
        "accelerators_used",
    ):
        assert result["execution_policy"][field] is False
    assert all(result["metamorphic_self_check"].values())
    assert all("ctext" not in row and "y" not in row for row in result["rows"])
    abstentions = [row for row in result["rows"] if row["measurement_state"] == "abstained"]
    assert {row["reason"] for row in abstentions} == {
        "missing_fullpaper_body",
        "no_executable_claim_relation",
        "no_retrievable_evidence",
    }


def test_train_gate_uses_only_measurability_and_freezes_all_six_mappings():
    train = _load(TRAIN)
    gate = build_train_gate(train)
    assert gate == _load(GATE)
    assert gate["schema"] == GATE_SCHEMA
    assert gate["selected"] is True
    assert gate["summary"] == {
        "n_candidate_relation_mappings": 6,
        "n_selected_relation_mappings": 6,
        "n_train_abstained_items": 32,
        "n_train_failed_items": 0,
        "n_train_measured_items": 118,
    }
    assert set(gate["forbidden_selection_inputs"].values()) == {False}
    assert all(value["passes"] for value in gate["criteria"].values())


def test_gate_fails_closed_on_execution_failure_or_forbidden_reference():
    train = copy.deepcopy(_load(TRAIN))
    measured = next(
        row for row in train["rows"] if row["measurement_state"] == "measured"
    )
    measured.update(
        {
            "measurement_state": "failed",
            "verifier_status": "execution_error",
            "reason": "trusted_verifier_exception",
            "error_type": "RuntimeError",
            "claim_count": 0,
            "certificate_count": 0,
            "evidence_link_count": 0,
            "decision_counts": {},
            "graph": None,
            "relation_certificates": [],
        }
    )
    train["summary"] = _summarize_execution(
        train["rows"], n_relations=len(train["relation_mappings"])
    )
    assert build_train_gate(train)["selected"] is False

    train = copy.deepcopy(_load(TRAIN))
    train["execution_policy"]["reference_values_loaded"] = True
    with pytest.raises(ScienceExecutionError, match="forbidden input"):
        build_train_gate(train)


def test_heldout_is_pre_reference_and_preserves_abstention_and_certificate_counts():
    result = _load(HELDOUT)
    assert result["schema"] == EXECUTION_SCHEMA
    assert result["phase"] == "heldout_pre_reference"
    assert result["train_gate"] == {
        "n_selected_relation_mappings": 6,
        "schema": GATE_SCHEMA,
        "selected": True,
        "selection_used_heldout": False,
    }
    assert result["summary"] == {
        "measured_coverage": 0.72,
        "measured_verifier_status_counts": {
            "evidence_link": 21,
            "insufficient": 78,
            "supported": 9,
        },
        "n_distinct_measured_verifier_statuses": 3,
        "n_items_with_relation_certificate": 9,
        "n_mapping_item_applications": 900,
        "n_relation_certificates": 10,
        "n_relation_mappings": 6,
        "n_unique_item_executions": 150,
        "three_state_totals_mapping_applications": {
            "abstained": 252,
            "failed": 0,
            "measured": 648,
        },
        "three_state_totals_unique_items": {
            "abstained": 42,
            "failed": 0,
            "measured": 108,
        },
    }
    assert all(
        result["execution_policy"][field] is False
        for field in (
            "reference_values_loaded",
            "outcome_values_loaded",
            "prompt_or_reconstruction_outputs_loaded",
            "external_supervision_used",
            "models_or_apis_called",
            "accelerators_used",
        )
    )
    assert {row["reason"] for row in result["rows"] if row["measurement_state"] == "abstained"} == {
        "missing_fullpaper_body",
        "no_executable_claim_relation",
    }
    certificates = [
        certificate
        for row in result["rows"]
        for certificate in row["relation_certificates"]
    ]
    assert len(certificates) == 10
    assert {certificate["decision"] for certificate in certificates} == {"supported"}
    assert {certificate["witness_kind"] for certificate in certificates} == {
        "relation_certificate"
    }


def test_checked_in_execution_rows_recompute_for_representative_items():
    for execution_path, items_path in (
        (TRAIN, DEFAULT_ITEMS / "compiler_train.json"),
        (HELDOUT, DEFAULT_ITEMS / "sealed_heldout.json"),
    ):
        execution = _load(execution_path)
        items = {item["item_key"]: item for item in _load(items_path)}
        statuses = ("supported", "insufficient", "abstain")
        for status in statuses:
            observed = next(
                row for row in execution["rows"] if row["verifier_status"] == status
            )
            abstract, body = parse_article_ctext(items[observed["item_key"]]["ctext"])
            raw = verify_document(observed["item_key"], abstract, body)
            if status == "abstain":
                assert raw["status"] == "abstain"
                assert raw["reason"] == observed["reason"]
            else:
                expected = {
                    "item_key": observed["item_key"],
                    "measurement_state": "measured",
                    **_concise_verifier_result(raw),
                }
                expected = json.loads(json.dumps(expected, allow_nan=False))
                assert observed == expected


def test_three_state_runner_retains_abstention_and_exception_without_crashing():
    seed, fidelity = _inputs()
    items = [
        {
            "item_key": f"science_train_{index:04d}",
            "ctext": article_ctext("We show a result.", "Table 2 shows a result."),
        }
        for index in range(1, 151)
    ]

    def fake_verifier(item_key: str, _abstract: str, _body: str):
        if item_key.endswith("0001"):
            raise RuntimeError("deliberate")
        if item_key.endswith("0002"):
            return {
                "status": "abstain",
                "reason": "no_executable_claim_relation",
                "claim_count": 0,
                "certificates": [],
            }
        return {
            "status": "insufficient",
            "reason": "retrieved_without_relation_certificate",
            "claim_count": 1,
            "certificate_count": 0,
            "evidence_link_count": 0,
            "decision_counts": {"insufficient": 1},
            "certificates": [],
            "graph": {"claim_nodes": 1},
        }

    result = execute_items(
        items,
        seed,
        fidelity,
        phase="compiler_train",
        items_path="synthetic.json",
        verifier=fake_verifier,
        self_check=lambda: {"synthetic": True},
    )
    assert result["summary"]["three_state_totals_unique_items"] == {
        "measured": 148,
        "abstained": 1,
        "failed": 1,
    }
    assert result["rows"][0]["error_type"] == "RuntimeError"


def test_heldout_gate_mismatch_fails_before_any_execution():
    seed, fidelity = _inputs()
    gate = copy.deepcopy(_load(GATE))
    gate["selected_relation_mappings"][0]["cell_id"] = "wrong"
    with pytest.raises(ScienceExecutionError, match="mappings drifted"):
        build_heldout_execution(
            _load(DEFAULT_ITEMS / "sealed_heldout.json"),
            seed,
            fidelity,
            gate,
            items_path="sealed_heldout.json",
        )
