from __future__ import annotations

import copy
import json
from pathlib import Path

from methods.metric_seam.hierarchy_seed_mapper import (
    DESIGN_SCOPE,
    build_seed_map,
    load_program_library,
    read_program_metadata,
)


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "outputs/metric_seam_pilot/hierarchy_r123/panel_v3.json"
METRICS = ROOT / "methods/existing_metrics_runner/coded/metrics"


def _inputs():
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    programs = load_program_library(METRICS, repo_root=ROOT)
    return panel, programs


def test_static_metadata_reader_does_not_import_metric_module(tmp_path):
    marker = tmp_path / "executed"
    module = tmp_path / "a999_probe.py"
    module.write_text(
        '"""Probe metric.\n\nNo runtime metadata needed."""\n'
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('bad')\n"
        "ASPECT_ID = 'a999'\nASPECT_NAME = 'Probe metric'\nTIER = 2\n"
        "TOOLS = ['tree-sitter-python']\nAPPLIES_TO_LANGS = ['Python']\n"
        "CLASSIFICATION = 'THIN'\n"
        "def applies(text): return True\ndef score(text): return 1.0\n",
        encoding="utf-8",
    )
    metadata = read_program_metadata(module)
    assert metadata.aspect_id == "a999"
    assert metadata.program_shape == "parsed_structure"
    assert not marker.exists()


def test_real_code_review_map_is_30_by_each_level_and_provenance_honest():
    panel, programs = _inputs()
    result = build_seed_map(panel, programs)
    assert result["design_scope"] == DESIGN_SCOPE
    assert result["hierarchy_frame"] == panel["hierarchy_frame"]
    assert result["n_cells"] == 90
    assert {row["level"] for row in result["rows"]} == {"R1", "R2", "R3"}
    assert all(stats["n_cells"] == 30 for stats in result["summary"]["by_level"].values())
    assert all(
        row["interpretation"].startswith("retrospective candidate retrieval only")
        for row in result["rows"]
    )
    selected = [row["selected_seed"] for row in result["rows"] if row["selected_seed"]]
    assert selected
    assert all(seed["tool_provenance"]["declared_classification"] != "THICK" for seed in selected)
    assert all(seed["depth_provenance"]["declared_tool_tier"] >= 2 for seed in selected)
    assert all(seed["depth_provenance"]["preferred_nonlexical"] for seed in selected)


def test_unknown_judgment_and_outcome_fields_cannot_change_retrieval():
    panel, programs = _inputs()
    baseline = build_seed_map(panel, programs)
    poisoned = copy.deepcopy(panel)
    for index, cell in enumerate(poisoned["cells"]):
        cell["reference_judgment"] = 1.0 if index % 2 else 0.0
        cell["heldout_label"] = "accept" if index % 3 else "reject"
        cell["reconstruction_result"] = {"rho": 0.99}
    observed = build_seed_map(poisoned, programs)
    assert observed == baseline


def test_known_source_semantics_retrieve_matching_structural_programs():
    panel, programs = _inputs()
    result = build_seed_map(panel, programs)

    naming = next(
        row for row in result["rows"]
        if row["level"] == "R1" and row["metric_name"].startswith("Intention-revealing")
    )
    assert naming["selected_seed"] is not None
    assert naming["selected_seed"]["aspect_id"] in {"a43", "a70", "a164", "a407"}

    control_flow = next(
        row for row in result["rows"]
        if row["level"] == "R3" and "cognitive complexity" in row["metric_name"]
    )
    assert control_flow["selected_seed"] is not None
    assert control_flow["selected_seed"]["aspect_id"] in {"a0", "a501", "a502"}


def test_weak_visual_design_construct_abstains_instead_of_claiming_a_program():
    panel, programs = _inputs()
    result = build_seed_map(panel, programs)
    visual = next(
        row for row in result["rows"]
        if row["level"] == "R3" and row["metric_name"] == "Design system and visual coherence"
    )
    assert visual["decision"] == "abstain"
    assert visual["selected_seed"] is None
