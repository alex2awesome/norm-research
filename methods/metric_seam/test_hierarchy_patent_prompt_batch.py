from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_patent_prompt_batch import (
    CANONICAL_ITEMS_ROOT,
    OUTPUT_CONTRACTS,
    POST_CODE_ARM_ID,
    SOURCE_ARM_IDS,
    CompiledPatentPromptBatch,
    PatentPromptBatchError,
    _post_code_prompt,
    _post_code_response_schema,
    _source_prompt,
    _source_specs,
    _validate_arm_bank,
    _validate_fidelity,
    _validate_items,
    _write_jobs,
    compile_prompt_batch,
    load_bound_items,
    validate_post_code_response,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"
FIDELITY = BASE / "patents_claim_structure_construct_fidelity_v1.json"
BANK = BASE / "prompt_arm_bank_v3.json"
GATE = BASE / "patents_claim_structure_train_gate_v1.json"
TRAIN_MANIFEST = BASE / "patents_prompt_train_manifest_v3.json"
TRAIN_JOBS = BASE / "patents_prompt_train_jobs_v3.jsonl.gz"
HELDOUT_MANIFEST = BASE / "patents_prompt_heldout_fixed_manifest_v3.json"
HELDOUT_JOBS = BASE / "patents_prompt_heldout_fixed_jobs_v3.jsonl.gz"
SUPERSESSION = BASE / "patents_prompt_v3_supersession_receipt.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def real_inputs():
    fidelity = _load(FIDELITY)
    bank = _load(BANK)
    gate = _load(GATE)
    fidelity_rows = _validate_fidelity(fidelity)
    patent_cells = _validate_arm_bank(bank, fidelity)
    return fidelity, gate, fidelity_rows, patent_cells


@pytest.fixture(scope="module")
def train_batch(real_inputs) -> CompiledPatentPromptBatch:
    fidelity, gate, _rows, _cells = real_inputs
    items, path, _manifest = load_bound_items(
        CANONICAL_ITEMS_ROOT, "compiler_train"
    )
    return compile_prompt_batch(
        fidelity,
        gate,
        _load(BANK),
        items,
        items_source=str(path),
    )


@pytest.fixture(scope="module")
def heldout_batch(real_inputs) -> CompiledPatentPromptBatch:
    fidelity, gate, _rows, _cells = real_inputs
    items, path, _manifest = load_bound_items(
        CANONICAL_ITEMS_ROOT, "heldout_pre_reference"
    )
    return compile_prompt_batch(
        fidelity,
        gate,
        _load(BANK),
        items,
        phase="heldout_pre_reference",
        items_source=str(path),
    )


def test_frozen_five_are_exact_relation_local_operational_outputs(real_inputs):
    _fidelity, _gate, rows, _cells = real_inputs
    assert len(OUTPUT_CONTRACTS) == 5
    assert set(OUTPUT_CONTRACTS) <= set(rows)
    assert {rows[cell_id]["verdict"] for cell_id in OUTPUT_CONTRACTS} == {
        "partial_relation_local"
    }
    assert all(
        rows[cell_id]["exact_whole_construct_fidelity"] is False
        for cell_id in OUTPUT_CONTRACTS
    )
    assert sorted(contract["depth"] for contract in OUTPUT_CONTRACTS.values()) == [
        1,
        1,
        1,
        1,
        2,
    ]


def test_source_channel_is_exact_canonical_quartet_with_matched_controls(real_inputs):
    _fidelity, _gate, _rows, cells = real_inputs
    for cell_id in OUTPUT_CONTRACTS:
        specs = _source_specs(cells[cell_id])
        assert [spec["arm_id"] for spec in specs] == list(SOURCE_ARM_IDS)
        assert {spec["form_id"] for spec in specs} == {"canonical"}
        assert specs[1]["control_for"] is None
        assert specs[2]["control_for"] == "source_definition_rules"
        assert specs[3]["control_for"] == "source_definition_rules"
        assert specs[2]["role"] == specs[3]["role"] == "source_bank_control"


def test_source_and_post_code_channels_are_visibly_separate(real_inputs):
    _fidelity, _gate, rows, cells = real_inputs
    cell_id = next(iter(OUTPUT_CONTRACTS))
    ctext = "ABSTRACT:\nA short disclosure.\n\nCLAIMS:\n1. A machine."
    source = _source_prompt(_source_specs(cells[cell_id])[1]["prompt"], ctext)
    post_code = _post_code_prompt(
        rows[cell_id],
        OUTPUT_CONTRACTS[cell_id],
        ctext,
        at_declared_character_cap=False,
    )
    assert source.count(ctext) == 1
    assert post_code.count(ctext) == 1
    assert "FROZEN SOURCE ARTICULATION CHANNEL" in source
    assert "POST-CODE RELATION-DISCLOSED CHANNEL" not in source
    assert "POST-CODE RELATION-DISCLOSED CHANNEL" in post_code
    assert "verifiability = the frozen code program" in post_code
    assert "isomorphism" not in post_code.lower()
    for relation_id in OUTPUT_CONTRACTS[cell_id]["relation_ids"]:
        assert relation_id in post_code


def test_official_train_items_are_required_byte_for_byte():
    items, path, _manifest = load_bound_items(CANONICAL_ITEMS_ROOT, "compiler_train")
    normalized, official_path, _ = _validate_items(
        items,
        phase="compiler_train",
        items_source=str(path),
    )
    assert normalized == items
    assert official_path == path
    contaminated = deepcopy(items)
    contaminated[0]["ctext"] += " "
    with pytest.raises(PatentPromptBatchError, match="official patent ctext bytes"):
        _validate_items(
            contaminated,
            phase="compiler_train",
            items_source=str(path),
        )
    with pytest.raises(PatentPromptBatchError, match="official split"):
        _validate_items(
            items,
            phase="compiler_train",
            items_source=str(path.parent / "wrong.json"),
        )


def test_fidelity_and_arm_bank_fail_closed_on_relation_or_control_drift(real_inputs):
    fidelity, _gate, _rows, _cells = real_inputs
    broken_fidelity = deepcopy(fidelity)
    target = next(
        row
        for row in broken_fidelity["rows"]
        if row["cell_id"] in OUTPUT_CONTRACTS
    )
    target["matched_relations"][0]["relation_id"] = "easier_proxy"
    with pytest.raises(PatentPromptBatchError, match="mapping drift"):
        _validate_fidelity(broken_fidelity)

    # The bank's own content identity catches any control mutation before use.
    broken_bank = deepcopy(_load(BANK))
    target_cell = next(
        cell for cell in broken_bank["cells"] if cell["id"] in OUTPUT_CONTRACTS
    )
    wrong = next(
        arm
        for arm in target_cell["arms"]
        if arm["id"] == "control_wrong_definition_rules"
    )
    wrong["control_for"] = None
    with pytest.raises(PatentPromptBatchError, match="identity mismatch"):
        _validate_arm_bank(broken_bank, fidelity)


def test_gzip_writer_is_deterministic_and_refuses_overwrite(tmp_path):
    jobs = [{"request_id": "a"}, {"request_id": "b"}]
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    assert _write_jobs(first, jobs, 2) == 2
    assert _write_jobs(second, jobs, 2) == 2
    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rt", encoding="utf-8") as handle:
        assert [json.loads(line) for line in handle] == jobs
    with pytest.raises(FileExistsError):
        _write_jobs(first, jobs, 2)


def test_train_compiler_counts_only_five_operational_outputs(train_batch):
    assert train_batch.manifest["status"] == "compiled_unscored"
    assert train_batch.manifest["phase"] == "compiler_train"
    assert train_batch.manifest["summary"] == {
        "n_cells": 5,
        "n_cells_by_level": {"R1": 0, "R2": 1, "R3": 4},
        "n_cells_by_depth": {"1": 4, "2": 1},
        "n_items": 150,
        "n_passes": 2,
        "n_source_prompt_specs": 20,
        "n_post_code_structured_specs": 5,
        "n_prompt_specs": 25,
        "n_jobs": 7500,
        "n_prompt_responses": 0,
        "n_reconstruction_estimates": 0,
        "n_isomorphism_adjudications": 0,
    }
    assert set(train_batch.manifest["forbidden_inputs"].values()) == {False}
    assert train_batch.manifest["gate_use_disclosure"] == {
        "compiler_train_gate_consumed": True,
        "gate_selected_operationally_variable_relation_outputs": True,
        "source_arm_quartet_selected_from_gate_statistics": False,
        "item_level_train_code_values_available_to_prompt_compiler": False,
        "heldout_information_used_to_change_gate_or_prompt_specs": False,
        "current_phase_heldout_ctext_packaged": False,
        "investigator_level_heldout_blindness_claimed": False,
    }


def test_jobs_embed_exact_ctext_once_and_keep_metadata_model_invisible(train_batch):
    source_job = next(train_batch.iter_jobs())
    ctext = train_batch.items[0]["ctext"]
    assert set(source_job) == {
        "request_id",
        "request",
        "executor_metadata",
        "audit_metadata",
    }
    assert set(source_job["request"]) == {"system", "user"}
    assert source_job["request"]["user"].count(ctext) == 1
    assert "ctext" not in source_job["audit_metadata"]
    assert source_job["audit_metadata"]["ctext_sha256"]
    assert source_job["audit_metadata"]["arm_role"] == "source_name_baseline"
    assert source_job["executor_metadata"]["stateless_separate_call"] is True
    assert "::contract=v3::" in source_job["request_id"]
    assert "::arm=" in source_job["request_id"]

    post_code = next(
        job
        for job in train_batch.iter_jobs()
        if job["audit_metadata"]["arm_id"] == POST_CODE_ARM_ID
    )
    assert post_code["request"]["user"].count(ctext) == 1
    assert "POST-CODE RELATION-DISCLOSED CHANNEL" in post_code["request"]["user"]
    assert "mode=positive_marker_certificates_plus_below_cap" in post_code[
        "request"
    ]["user"]
    assert post_code["audit_metadata"]["arm_role"] == (
        "post_code_relation_disclosure"
    )
    assert post_code["audit_metadata"]["response_contract_id"].startswith(
        "patent_"
    )
    assert post_code["executor_metadata"]["semantic_response_validator"] == (
        "validate_post_code_response.v3"
    )


def test_gate_cell_or_relation_drift_fails_closed(real_inputs):
    fidelity, gate, _rows, _cells = real_inputs
    items, path, _manifest = load_bound_items(
        CANONICAL_ITEMS_ROOT, "compiler_train"
    )
    broken_gate = deepcopy(gate)
    broken_gate["selected_operational_cells"][0]["relations"][0][
        "relation_id"
    ] = "easier_proxy"
    with pytest.raises(PatentPromptBatchError, match="gate output relation drift"):
        compile_prompt_batch(
            fidelity,
            broken_gate,
            _load(BANK),
            items,
            items_source=str(path),
        )


def test_v3_request_identity_cannot_collide_with_superseded_packs(train_batch):
    request_ids = [job["request_id"] for _, job in zip(range(40), train_batch.iter_jobs())]
    assert len(request_ids) == len(set(request_ids))
    assert all("::contract=v3::" in request_id for request_id in request_ids)
    v1_style = {request_id.replace("::contract=v3", "") for request_id in request_ids}
    v2_style = {
        request_id.replace("::contract=v3", "::contract=v2")
        for request_id in request_ids
    }
    assert not set(request_ids) & v1_style
    assert not set(request_ids) & v2_style
    assert train_batch.manifest["independent_pass_execution_contract"][
        "sampling_seed_salt"
    ] == "metric-seam-patent-prompt-pass-v3"


def test_cap_specialized_schema_forbids_functional_scalar(train_batch):
    post_jobs = (
        job
        for job in train_batch.iter_jobs()
        if job["audit_metadata"]["response_contract_id"]
        == "patent_functional_marker_incidence.v3"
    )
    cap_job = next(
        job
        for job in post_jobs
        if job["audit_metadata"]["ctext_at_declared_character_cap"]
    )
    scalar = cap_job["executor_metadata"]["response_schema"]["properties"][
        "presented_claim_incidence"
    ]
    assert scalar == {"type": "null", "const": None}
    assert "maxItems" not in cap_job["executor_metadata"]["response_schema"][
        "properties"
    ]["presented_active_claim_numbers"]
    below = _post_code_response_schema(
        "patent_functional_marker_incidence.v3",
        at_declared_character_cap=False,
    )
    assert below["properties"]["presented_claim_incidence"]["type"] == [
        "number",
        "null",
    ]
    architecture = _post_code_response_schema(
        "patent_architecture_finite_witnesses.v3",
        at_declared_character_cap=True,
    )
    assert not {
        "dependency_presented_text_score",
        "layering_presented_text_value",
    } & set(architecture["properties"])


def test_architecture_validator_matches_v13_certificate_unions():
    payload = {
        "measurement_status": "measured",
        "dependency_certificates": [
            {
                "relation": "claim_dependency_well_formedness",
                "kind": "positive_witness",
                "child_claim": 2,
                "parent_claim": 1,
            },
            {
                "relation": "claim_dependency_well_formedness",
                "kind": "counter_witness",
                "child": 3,
                "parent": 4,
                "reasons": ["referenced_claim_number_is_not_lower"],
            },
            {
                "relation": "claim_dependency_well_formedness",
                "kind": "counter_witness",
                "claim": 5,
                "surface": "5 through 3",
                "reason": "descending_dependency_range",
            },
        ],
        "layering_witnesses": [
            {
                "relation": "claim_set_layering",
                "kind": "positive_witness",
                "independent_claim": 1,
                "dependent_claim": 2,
                "parent_claim": 1,
            }
        ],
        "rationale": "Finite presented-text witnesses.",
    }
    assert validate_post_code_response(
        payload,
        contract_id="patent_architecture_finite_witnesses.v3",
        at_declared_character_cap=True,
    ) == payload
    wrong_order = deepcopy(payload)
    wrong_order["dependency_certificates"][0]["parent_claim"] = 3
    with pytest.raises(PatentPromptBatchError, match="parent < child"):
        validate_post_code_response(
            wrong_order,
            contract_id="patent_architecture_finite_witnesses.v3",
            at_declared_character_cap=False,
        )
    empty_measured = {
        "measurement_status": "measured",
        "dependency_certificates": [],
        "layering_witnesses": [],
        "rationale": "No output.",
    }
    with pytest.raises(PatentPromptBatchError, match="requires a finite witness"):
        validate_post_code_response(
            empty_measured,
            contract_id="patent_architecture_finite_witnesses.v3",
            at_declared_character_cap=True,
        )
    duplicate = deepcopy(payload)
    duplicate["dependency_certificates"].append(
        deepcopy(duplicate["dependency_certificates"][0])
    )
    with pytest.raises(PatentPromptBatchError, match="duplicate dependency"):
        validate_post_code_response(
            duplicate,
            contract_id="patent_architecture_finite_witnesses.v3",
            at_declared_character_cap=True,
        )
    duplicate_layering = deepcopy(payload)
    duplicate_layering["layering_witnesses"].append(
        deepcopy(duplicate_layering["layering_witnesses"][0])
    )
    with pytest.raises(PatentPromptBatchError, match="duplicate layering"):
        validate_post_code_response(
            duplicate_layering,
            contract_id="patent_architecture_finite_witnesses.v3",
            at_declared_character_cap=True,
        )


def test_category_validator_enforces_v13_span_and_prompt_boundaries(real_inputs):
    _fidelity, _gate, rows, _cells = real_inputs
    contract = next(
        value
        for value in OUTPUT_CONTRACTS.values()
        if value["response_contract_id"]
        == "patent_category_positive_certificates.v3"
    )
    prompt = _post_code_prompt(
        rows[next(cell_id for cell_id, value in OUTPUT_CONTRACTS.items() if value is contract)],
        contract,
        "CLAIMS:\n1. A device comprising a processor.",
        at_declared_character_cap=False,
    )
    for phrase in (
        "first 240 claim-text characters",
        "non-transitory medium",
        "device",
        "Choose the rightmost category-bearing match",
    ):
        assert phrase in prompt
    payload = {
        "measurement_status": "measured",
        "category_certificates": [
            {
                "relation": "statutory_category_surface_coverage",
                "kind": "positive_witness",
                "claim": 1,
                "category": "machine_or_apparatus",
                "surface": "device",
                "span": [2, 8],
            }
        ],
        "rationale": "Positive surface.",
    }
    validate_post_code_response(
        payload,
        contract_id="patent_category_positive_certificates.v3",
        at_declared_character_cap=True,
    )
    payload["category_certificates"][0]["span"] = [8, 2]
    with pytest.raises(PatentPromptBatchError, match="invalid category certificate"):
        validate_post_code_response(
            payload,
            contract_id="patent_category_positive_certificates.v3",
            at_declared_character_cap=True,
        )
    duplicate = {
        "measurement_status": "measured",
        "category_certificates": [
            {
                "relation": "statutory_category_surface_coverage",
                "kind": "positive_witness",
                "claim": 1,
                "category": "machine_or_apparatus",
                "surface": "device",
                "span": [2, 8],
            }
        ]
        * 2,
        "rationale": "Duplicate surfaces must fail.",
    }
    with pytest.raises(PatentPromptBatchError, match="duplicate category"):
        validate_post_code_response(
            duplicate,
            contract_id="patent_category_positive_certificates.v3",
            at_declared_character_cap=True,
        )


def test_functional_validator_enforces_cap_and_incidence_arithmetic():
    payload = {
        "measurement_status": "measured",
        "presented_active_claim_numbers": [1, 2],
        "functional_marker_certificates": [
            {
                "relation": "functional_limitation_incidence",
                "kind": "positive_witness",
                "claim": 2,
                "surface": "configured to",
            }
        ],
        "presented_claim_incidence": 0.5,
        "rationale": "One of two presented active claims.",
    }
    validate_post_code_response(
        payload,
        contract_id="patent_functional_marker_incidence.v3",
        at_declared_character_cap=False,
    )
    inconsistent = deepcopy(payload)
    inconsistent["presented_claim_incidence"] = 0.25
    with pytest.raises(PatentPromptBatchError, match="incidence is inconsistent"):
        validate_post_code_response(
            inconsistent,
            contract_id="patent_functional_marker_incidence.v3",
            at_declared_character_cap=False,
        )
    cap = deepcopy(payload)
    cap["presented_claim_incidence"] = None
    validate_post_code_response(
        cap,
        contract_id="patent_functional_marker_incidence.v3",
        at_declared_character_cap=True,
    )
    cap["presented_claim_incidence"] = 0.5
    with pytest.raises(PatentPromptBatchError, match="cap-contact"):
        validate_post_code_response(
            cap,
            contract_id="patent_functional_marker_incidence.v3",
            at_declared_character_cap=True,
        )
    no_marker_at_cap = {
        "measurement_status": "applicable_abstain",
        "presented_active_claim_numbers": [1, 2],
        "functional_marker_certificates": [],
        "presented_claim_incidence": None,
        "rationale": "No positive marker certificate in presented bytes.",
    }
    validate_post_code_response(
        no_marker_at_cap,
        contract_id="patent_functional_marker_incidence.v3",
        at_declared_character_cap=True,
    )
    wrong_cap_status = deepcopy(no_marker_at_cap)
    wrong_cap_status["measurement_status"] = "not_applicable"
    with pytest.raises(PatentPromptBatchError, match="finite-support policy"):
        validate_post_code_response(
            wrong_cap_status,
            contract_id="patent_functional_marker_incidence.v3",
            at_declared_character_cap=True,
        )
    no_claims_at_cap = deepcopy(no_marker_at_cap)
    no_claims_at_cap["measurement_status"] = "not_applicable"
    no_claims_at_cap["presented_active_claim_numbers"] = []
    validate_post_code_response(
        no_claims_at_cap,
        contract_id="patent_functional_marker_incidence.v3",
        at_declared_character_cap=True,
    )


def test_abstract_validator_requires_status_count_consistency():
    payload = {
        "measurement_status": "measured",
        "abstract_word_count": 42,
        "rationale": "Counted the closed named span.",
    }
    validate_post_code_response(
        payload,
        contract_id="patent_presented_abstract_word_count.v3",
        at_declared_character_cap=True,
    )
    payload["abstract_word_count"] = None
    with pytest.raises(PatentPromptBatchError, match="status/count mismatch"):
        validate_post_code_response(
            payload,
            contract_id="patent_presented_abstract_word_count.v3",
            at_declared_character_cap=True,
        )


def test_heldout_compiles_fixed_form_orbits_without_any_results(heldout_batch):
    assert heldout_batch.manifest["phase"] == "heldout_pre_reference"
    assert heldout_batch.manifest["summary"] == {
        "n_cells": 5,
        "n_cells_by_level": {"R1": 0, "R2": 1, "R3": 4},
        "n_cells_by_depth": {"1": 4, "2": 1},
        "n_items": 150,
        "n_passes": 2,
        "n_source_prompt_specs": 60,
        "n_post_code_structured_specs": 5,
        "n_prompt_specs": 65,
        "n_jobs": 19500,
        "n_prompt_responses": 0,
        "n_reconstruction_estimates": 0,
        "n_isomorphism_adjudications": 0,
    }
    assert heldout_batch.manifest["source_arm_contract"]["forms"] == [
        "canonical",
        "question",
        "boilerplate",
    ]
    assert heldout_batch.manifest["gate_use_disclosure"][
        "current_phase_heldout_ctext_packaged"
    ] is True
    assert heldout_batch.manifest["batch_role"] == (
        "fixed_after_train_gate_exploratory_pre_reference"
    )
    temporal = heldout_batch.manifest["temporal_provenance"]
    assert temporal["absence_of_human_influence_certified"] is False
    assert temporal["fresh_confirmatory_split_required_for_temporal_preregistration"] is True
    assert "exploratory" in temporal["current_heldout_disposition"]
    assert set(heldout_batch.manifest["forbidden_inputs"].values()) == {False}


@pytest.mark.parametrize(
    ("manifest_path", "jobs_path", "expected"),
    [
        (TRAIN_MANIFEST, TRAIN_JOBS, 7500),
        (HELDOUT_MANIFEST, HELDOUT_JOBS, 19500),
    ],
)
def test_checked_in_gzip_batches_match_unscored_manifests(
    manifest_path, jobs_path, expected
):
    manifest = _load(manifest_path)
    assert manifest["status"] == "compiled_unscored"
    assert manifest["summary"]["n_jobs"] == expected
    assert manifest["summary"]["n_prompt_responses"] == 0
    assert manifest["summary"]["n_reconstruction_estimates"] == 0
    assert manifest["summary"]["n_isomorphism_adjudications"] == 0
    jobs_artifact = manifest["jobs_artifact"]
    assert jobs_artifact["n_jobs"] == expected
    assert jobs_artifact["model_api_or_gpu_calls_performed"] is False
    assert jobs_artifact["sha256"] == hashlib.sha256(jobs_path.read_bytes()).hexdigest()
    request_ids = set()
    count = 0
    with gzip.open(jobs_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            count += 1
            request_ids.add(row["request_id"])
            assert set(row["request"]) == {"system", "user"}
    assert count == expected
    assert len(request_ids) == expected


def test_supersession_receipt_preserves_prior_packs_and_binds_v3():
    receipt = _load(SUPERSESSION)
    assert receipt["status"] == "v3-repaired-compiled-unscored"
    assert "do not execute" in receipt["v1_disposition"]
    assert receipt["v1_cross_audit"][
        "blocking_or_material_findings_repaired_in_v2"
    ] == ["P1", "P2", "P3", "P4", "P5", "P6"]
    for artifact in [
        *receipt["superseded_v1_artifacts"],
        *receipt["superseded_v2_artifacts"],
        *receipt["current_v3_artifacts"],
    ]:
        path = ROOT / artifact["path"]
        assert path.is_file()
        assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    temporal = receipt["temporal_disposition"]
    assert temporal["temporally_predeclared_or_confirmatory"] is False
    assert temporal["fresh_confirmatory_split_required"] is True
    assert set(receipt["execution"].values()) <= {0, False}
