from __future__ import annotations

import copy
import json
from pathlib import Path

from methods.metric_seam.hierarchy_science_claim_seed_mapper import (
    CAPABILITY_ID,
    DESIGN_SCOPE,
    build_capability_inventory,
    build_seed_map,
    read_source_module,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"
SCIENCE = ROOT / "methods/metric_seam/science_claims_v2"


def _inputs():
    panel = json.loads((BASE / "panel_v3.json").read_text(encoding="utf-8"))
    capability = build_capability_inventory(
        SCIENCE / "core.py",
        SCIENCE / "core_relation_strict.py",
        repo_root=ROOT,
    )
    return panel, capability


def test_static_reader_does_not_import_or_execute_source(tmp_path):
    marker = tmp_path / "executed"
    module = tmp_path / "manual_source.py"
    module.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('bad')\n"
        "def verify_document(paper_id, abstract, body):\n"
        "    return {'status': 'bad'}\n",
        encoding="utf-8",
    )
    inventory = read_source_module(module, role="test")
    assert inventory.functions == ("verify_document",)
    assert not marker.exists()


def test_real_source_inventory_is_manual_pure_code_and_relation_deep():
    _, capability = _inputs()
    assert capability["capability_id"] == CAPABILITY_ID
    assert capability["historical_construction"] == (
        "retrospective manually designed pipeline seed"
    )
    assert capability["automatic_discovery"] is False
    assert capability["channel"] == "pure_code"
    assert capability["maximum_relation_chain_depth"] == 3
    assert [step["effective_depth"] for step in capability["relation_chain"]] == [
        2,
        3,
        2,
        2,
    ]
    assert all(
        not module["forbidden_import_roots"]
        and not module["dynamic_execution_calls"]
        for module in capability["source_modules"]
    )
    assert capability["external_scientific_truth_established"] is False
    assert capability["whole_peer_review_judgement_established"] is False


def test_source_only_inventory_is_narrow_with_exact_counts():
    panel, capability = _inputs()
    result = build_seed_map(panel, capability)
    assert result["design_scope"] == DESIGN_SCOPE
    assert result["task"] == "peer-review"
    assert result["n_cells"] == 90
    assert result["n_historical_capability_families"] == 1
    assert result["summary"]["decision_counts"] == {
        "abstain": 81,
        "candidate_seed_pending_independent_construct_fidelity_audit": 9,
    }
    assert result["summary"]["by_level"] == {
        "R1": {"n_cells": 30, "n_candidate_seeds": 2, "n_abstentions": 28},
        "R2": {"n_cells": 30, "n_candidate_seeds": 2, "n_abstentions": 28},
        "R3": {"n_cells": 30, "n_candidate_seeds": 5, "n_abstentions": 25},
    }
    assert result["summary"]["relation_local_fidelity_established"] == 0
    assert result["summary"]["execution_witnesses_established"] == 0
    assert result["summary"]["external_scientific_truth_claims"] == 0


def test_expected_source_relations_are_the_only_retrieved_candidates():
    panel, capability = _inputs()
    result = build_seed_map(panel, capability)
    observed = {
        row["metric_name"]
        for row in result["rows"]
        if row["selected_seed"] is not None
    }
    assert observed == {
        "Alignment of claims, evidence strength, and tone",
        "Front matter quality (title and abstract)",
        "Claim–evidence alignment and causal caution",
        "Claim support and inference rigor",
        "Citation practice quality, coverage, and ethics",
        "Causal inference and generalization claims rigor",
        "Discussion and conclusions — interpretation, balance, and implications",
        "Title, abstract, and plain‑language summaries — clarity and fidelity",
    }
    # One construct occurs independently at R2 and R3.
    assert sum(
        row["metric_name"] == "Claim–evidence alignment and causal caution"
        and row["selected_seed"] is not None
        for row in result["rows"]
    ) == 2
    selected = [row["selected_seed"] for row in result["rows"] if row["selected_seed"]]
    assert all(seed["capability_id"] == CAPABILITY_ID for seed in selected)
    assert all(seed["channel"] == "pure_code" for seed in selected)
    assert all(seed["maximum_relation_chain_depth"] == 3 for seed in selected)


def test_related_science_words_do_not_expand_the_one_capability():
    panel, capability = _inputs()
    result = build_seed_map(panel, capability)
    for name in (
        "Novelty and significance of contribution",
        "Quality of the Discussion: evidence-based, contextualized, and balanced",
        "Formal analysis (statistical/mathematical/computational)",
        "Quantitative study design and analysis rigor",
        "External validity and generalizability",
        "Title clarity and accuracy",
    ):
        rows = [row for row in result["rows"] if row["metric_name"] == name]
        assert rows, name
        assert all(row["decision"] == "abstain" for row in rows)
        assert all(row["selected_seed"] is None for row in rows)


def test_unknown_outcome_fields_cannot_change_retrieval():
    panel, capability = _inputs()
    baseline = build_seed_map(panel, capability)
    poisoned = copy.deepcopy(panel)
    for index, cell in enumerate(poisoned["cells"]):
        cell["acceptance_label"] = index % 2
        cell["reviewer_score"] = float(index)
        cell["program_output"] = {"status": "supported", "certificate_count": 999}
        cell["correlation"] = 0.999
        cell["reconstruction_result"] = {"rho": 0.999, "isomorphic": True}
    assert build_seed_map(poisoned, capability) == baseline


def test_checked_in_seed_artifact_is_exact_builder_output():
    panel, capability = _inputs()
    expected = build_seed_map(panel, capability)
    observed = json.loads(
        (BASE / "peer_review_science_claim_seed_map_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert observed == expected
