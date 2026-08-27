from __future__ import annotations

import copy
import json
from pathlib import Path

from methods.metric_seam.hierarchy_math_seed_mapper import (
    DESIGN_SCOPE,
    build_seed_map,
    load_program_families,
    read_capability_catalog,
    read_program_variant,
)


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "outputs/metric_seam_pilot/hierarchy_r123/panel_v3.json"
PROGRAMS = ROOT / "methods/metric_seam/hybrids/programs_math"
OPS_MATH = ROOT / "methods/metric_seam/hybrids/ops_math.py"


def _inputs():
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    catalog = read_capability_catalog(OPS_MATH)
    families = load_program_families(
        PROGRAMS,
        capability_catalog=catalog,
        repo_root=ROOT,
    )
    return panel, catalog, families


def test_static_reader_does_not_import_or_execute_historical_program(tmp_path):
    marker = tmp_path / "executed"
    module = tmp_path / "a999_h0.py"
    module.write_text(
        '"""a999: Structural proof organization.\n\n'
        'rho=1.0 outcome material that must not enter source heading."""\n'
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('bad')\n"
        "LLM_FIELDS = {'semantic_gap': 'prompt'}\n"
        "def score(text, extracted, ops):\n"
        "    return len(ops.proof_skeleton(text))\n",
        encoding="utf-8",
    )
    variant = read_program_variant(
        module,
        capability_names={"proof_skeleton"},
    )
    assert variant.source_heading == "Structural proof organization"
    assert variant.deep_math_ops == ("proof_skeleton",)
    assert not marker.exists()


def test_real_math_map_is_30_by_level_and_provenance_honest():
    panel, catalog, families = _inputs()
    result = build_seed_map(panel, families, capability_catalog=catalog)
    assert result["design_scope"] == DESIGN_SCOPE
    assert result["n_cells"] == 90
    assert result["n_historical_program_families"] == 35
    assert result["n_historical_program_variants"] == 40
    assert all(stats["n_cells"] == 30 for stats in result["summary"]["by_level"].values())
    assert result["summary"]["exact_whole_construct_code_fidelity_established"] == 0
    assert result["summary"]["relation_local_code_fidelity_established"] == 0
    selected = [row["selected_seed"] for row in result["rows"] if row["selected_seed"]]
    assert selected
    assert all(seed["depth_provenance"]["derived_code_depth"] >= 2 for seed in selected)
    assert all(
        "manual historical hybrid" in seed["hybrid_provenance"]["historical_construction"]
        for seed in selected
    )
    assert all(
        row["interpretation"].startswith("retrospective candidate retrieval only")
        for row in result["rows"]
    )


def test_unknown_judgment_and_outcome_fields_cannot_change_retrieval():
    panel, catalog, families = _inputs()
    baseline = build_seed_map(panel, families, capability_catalog=catalog)
    poisoned = copy.deepcopy(panel)
    for index, cell in enumerate(poisoned["cells"]):
        cell["reference_judgment"] = 1.0 if index % 2 else 0.0
        cell["heldout_label"] = "accept" if index % 3 else "reject"
        cell["program_output"] = {"score": index / 1000}
        cell["correlation"] = 0.999
        cell["reconstruction_result"] = {"rho": 0.999, "isomorphic": True}
    observed = build_seed_map(poisoned, families, capability_catalog=catalog)
    assert observed == baseline


def test_known_source_semantics_retrieve_math_program_families():
    panel, catalog, families = _inputs()
    result = build_seed_map(panel, families, capability_catalog=catalog)

    notation = next(
        row
        for row in result["rows"]
        if row["level"] == "R3"
        and row["metric_name"].startswith("Notation and terminology")
    )
    assert notation["selected_seed"] is not None
    assert notation["selected_seed"]["aspect_id"] in {"a156", "a168", "a180", "a210"}

    visual = next(
        row
        for row in result["rows"]
        if row["level"] == "R2"
        and row["metric_name"] == "Visual and diagrammatic reasoning in proofs"
    )
    assert visual["selected_seed"] is not None
    assert visual["selected_seed"]["aspect_id"] == "a42"


def test_unrepresented_social_science_constructs_abstain():
    panel, catalog, families = _inputs()
    result = build_seed_map(panel, families, capability_catalog=catalog)
    for metric_name in (
        "Kuhnian paradigm theory constructs",
        "Affective and self‑regulatory dimensions in learning",
    ):
        row = next(item for item in result["rows"] if item["metric_name"] == metric_name)
        assert row["decision"] == "abstain"
        assert row["selected_seed"] is None


def test_latest_revision_policy_is_source_order_only():
    _, _, families = _inputs()
    by_id = {family.aspect_id: family for family in families}
    assert [variant.revision for variant in by_id["a144"].variants] == [0, 1]
    assert by_id["a144"].selected_variant.revision == 1
    assert by_id["a12"].selected_variant.revision == 0
