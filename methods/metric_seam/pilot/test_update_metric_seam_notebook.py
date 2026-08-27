from __future__ import annotations

import ast
import json
from pathlib import Path
import shutil

from methods.metric_seam.pilot import update_metric_seam_notebook as updater


def _cell(notebook: dict, cell_id: str) -> dict:
    return next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)


def test_updater_makes_optional_task_tables_explicit_and_is_idempotent(
    tmp_path: Path, monkeypatch
) -> None:
    notebook_path = tmp_path / "metric-seam.ipynb"
    shutil.copyfile(updater.NOTEBOOK, notebook_path)
    monkeypatch.setattr(updater, "NOTEBOOK", notebook_path)

    updater.update_notebook()
    once = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell_id in ("dde45571", "8174418c", "humor-legal-code"):
        cell = _cell(once, cell_id)
        cell["execution_count"] = 101
        cell["outputs"] = [{"name": "stdout", "output_type": "stream", "text": "ok\n"}]
    notebook_path.write_text(json.dumps(once), encoding="utf-8")
    updater.update_notebook()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    survey = "".join(_cell(notebook, "dde45571")["source"])
    assert "survey_task_tables(OUTD)" in survey
    assert "unavailable_missing_artifact" not in survey  # supplied by the tested loader
    assert "excluded, not counted as zero" in survey
    assert 'f"tasks/{task}/seam_table.json"' not in survey
    assert "def _survey_table_path" not in survey
    ast.parse(survey)

    comparison = "".join(_cell(notebook, "8174418c")["source"])
    assert "survey_task_tables(OUTD)" in comparison
    assert "missing artifact; excluded, not counted as zero" in comparison
    assert "json.load(open" not in comparison
    ast.parse(comparison)

    legal = "".join(_cell(notebook, "humor-legal-code")["source"])
    assert "optional_seam_table" in legal
    assert "missing artifact; excluded, not counted as zero" in legal
    ast.parse(legal)

    for cell_id in ("dde45571", "8174418c", "humor-legal-code"):
        cell = _cell(notebook, cell_id)
        assert cell["execution_count"] == 101
        assert cell["outputs"] == [
            {"name": "stdout", "output_type": "stream", "text": "ok\n"}
        ]

    hierarchy = "".join(
        _cell(notebook, updater.CELL_PREFIX + "hierarchy-funnel")["source"]
    )
    assert "math_hierarchy_static_funnel" in hierarchy
    assert "math_hierarchy_symbolic_capability_sensitivity" in hierarchy
    assert "math_hierarchy_operational_funnel" in hierarchy
    assert "Math prompt-articulability batches:" in hierarchy
    assert "technical_coverage" in hierarchy
    assert "Corrected outcome perturbation ranges recomputed:" in hierarchy
    assert "code_review_representation_family_sensitivity" in hierarchy
    assert "Code representation-family anchor:" in hierarchy
    assert "code_review_additive_unused_program_funnel" in hierarchy
    assert "Additive code-review extension:" in hierarchy
    assert "additive held-out confirmatory-ready" in hierarchy
    assert "science_hierarchy_static_funnel" in hierarchy
    assert "science_hierarchy_fullarticle_operational_funnel" in hierarchy
    assert "Science addressed prompt/code scaffold:" in hierarchy
    assert "distinct prepared requests" in hierarchy
    assert "Science exact-ctext prompt instrument:" in hierarchy
    assert "exact decoded payload replay" in hierarchy
    assert 'science_projection["code_projection_summary"]' in hierarchy
    assert 'science_projection["reconstruction_decisions"]' in hierarchy
    assert "patent_claim_structure_hierarchy_static_funnel" in hierarchy
    assert "patent_claim_structure_hierarchy_operational_funnel" in hierarchy
    assert "pure-code patent claim-structure stage" in hierarchy
    assert "witnesses_by_audited_depth" in hierarchy
    ast.parse(hierarchy)

    technical = "".join(
        _cell(notebook, updater.CELL_PREFIX + "technical")["source"]
    )
    assert "active_code_a104_supplemental" in technical
    assert "Code a104 input-projection sensitivity" in technical
    assert "Code a104 repository-execution augmentation" in technical
    assert "Code 10-program representation family" in technical
    ast.parse(technical)

    generated_ids = [
        cell.get("id")
        for cell in notebook["cells"]
        if str(cell.get("id", "")).startswith(updater.CELL_PREFIX)
    ]
    assert len(generated_ids) == len(set(generated_ids))


def test_current_notebook_setup_and_survey_cells_execute_without_optional_files() -> None:
    """Regression for the missing ``tasks/code_review/seam_table.json`` failure."""

    notebook = json.loads(updater.NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {}
    setup = "".join(_cell(notebook, "e3ced72c")["source"])
    survey = "".join(_cell(notebook, "dde45571")["source"])
    exec(compile(setup, "<metric-seam-notebook-setup>", "exec"), namespace)
    exec(compile(survey, "<metric-seam-notebook-survey>", "exec"), namespace)

    bundle = namespace["survey_bundle"]
    assert isinstance(bundle, dict)
    assert bundle["unavailable_tasks"] == []
    assert len(namespace["rows"]) == 246
