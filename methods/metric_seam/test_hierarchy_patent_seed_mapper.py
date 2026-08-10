from __future__ import annotations

import copy
import json
from pathlib import Path

from methods.metric_seam.hierarchy_patent_seed_mapper import (
    DESIGN_SCOPE,
    build_seed_map,
    load_program_seeds,
    read_program_seed,
)


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "outputs/metric_seam_pilot/hierarchy_r123/panel_v3.json"
PROGRAMS = ROOT / "methods/metric_seam/f2p_mock/programs_pa"


def _inputs():
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    programs = load_program_seeds(PROGRAMS, repo_root=ROOT)
    return panel, programs


def test_static_reader_does_not_import_or_execute_program(tmp_path):
    marker = tmp_path / "executed"
    module = tmp_path / "a999_h0.py"
    module.write_text(
        '"""a999 -- Novelty and public disclosure (hybrid)."""\n'
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('bad')\n"
        "LLM_FIELDS = {'closest_art': 'prompt'}\n"
        "def score(text, extracted, ops, dpid=None):\n"
        "    return ops.prior_art(dpid)\n",
        encoding="utf-8",
    )
    seed = read_program_seed(module)
    assert seed.heading == "Novelty and public disclosure"
    assert seed.invoked_ops == ("prior_art",)
    assert not marker.exists()


def test_real_inventory_is_narrow_and_provenance_honest():
    panel, programs = _inputs()
    result = build_seed_map(panel, programs)
    assert result["design_scope"] == DESIGN_SCOPE
    assert result["n_cells"] == 90
    assert result["n_historical_program_families"] == 4
    assert result["summary"]["decision_counts"] == {
        "abstain": 84,
        "candidate_seed_pending_independent_construct_fidelity_audit": 6,
    }
    assert result["summary"]["by_level"]["R1"]["n_candidate_seeds"] == 2
    assert result["summary"]["by_level"]["R2"]["n_candidate_seeds"] == 1
    assert result["summary"]["by_level"]["R3"]["n_candidate_seeds"] == 3
    assert result["summary"]["relation_local_fidelity_established"] == 0
    assert result["summary"]["pure_code_witnesses_established"] == 0
    selected = [row["selected_seed"] for row in result["rows"] if row["selected_seed"]]
    assert selected
    assert all(seed["depth_provenance"]["derived_program_depth"] == 3 for seed in selected)
    assert all(seed["provenance"]["evidence_candidate_pool"] == "examiner/oracle conditioned" for seed in selected)
    assert all(not seed["provenance"]["pure_code_witness"] for seed in selected)


def test_unknown_outcome_fields_cannot_change_retrieval():
    panel, programs = _inputs()
    baseline = build_seed_map(panel, programs)
    poisoned = copy.deepcopy(panel)
    for index, cell in enumerate(poisoned["cells"]):
        cell["judgement"] = index % 2
        cell["heldout_label"] = "grant" if index % 3 else "reject"
        cell["program_output"] = {"score": index / 1000}
        cell["correlation"] = 0.999
        cell["reconstruction_result"] = {"rho": 0.999, "isomorphic": True}
    assert build_seed_map(poisoned, programs) == baseline


def test_known_patent_relations_retrieve_expected_families():
    panel, programs = _inputs()
    result = build_seed_map(panel, programs)
    expected = {
        "Novelty requirement (statutory)": "a34",
        "Patentability prerequisites — novelty, inventive step, industrial applicability": "a35",
        "Novelty and public-disclosure bars": "a34",
        "Claim charting and prior‑art differentiation rigor": "a60",
        "Utility/industrial applicability": "a35",
        "Novelty and pre‑filing disclosures/grace periods": "a34",
    }
    observed = {
        row["metric_name"]: row["selected_seed"]["aspect_id"]
        for row in result["rows"]
        if row["selected_seed"] is not None
    }
    assert observed == expected


def test_related_words_do_not_expand_tiny_bank_beyond_its_relations():
    panel, programs = _inputs()
    result = build_seed_map(panel, programs)
    for name in (
        "Background and problem framing in the specification",
        "Written Description — possession at filing across claimed breadth",
        "Eligibility — judicial exceptions and practical application/inventive concept",
        "PTAB AIA trial petitions/briefs — particularity, mapping, and content",
    ):
        row = next(item for item in result["rows"] if item["metric_name"] == name)
        assert row["decision"] == "abstain"
        assert row["selected_seed"] is None

