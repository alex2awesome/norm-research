from methods.metric_seam.family_scale.study_spec import compile_study


def _panel():
    cells = []
    for task in ("math-stackexchange", "code-review", "peer-review", "patents"):
        for level in ("R1", "R2", "R3"):
            for i in range(7):
                cells.append({
                    "id": f"{task}:{level}:{i}", "task": task, "level": level,
                    "construct": f"construct {i}", "description": f"description {i}",
                    "children": ["must not leak"], "outcome": 0.9,
                })
    return {"status": "frozen-outcome-blind-hierarchy-sample", "cells": cells}


def test_compile_is_deterministic_blind_and_balanced():
    a = compile_study(_panel(), per_domain_level=2)
    b = compile_study(_panel(), per_domain_level=2)
    assert a == b
    assert len(a["cells"]) == 24
    assert all(set(row["metric_text"]) == {"construct", "description"} for row in a["cells"])
    assert "children" not in str(a) and "must not leak" not in str(a)
    assert a["prompt_execution"]["unbatched_calibration_fraction"] == 0.10
    assert a["occasion"]["shared_proposer_required"] is True


def test_rejects_nonfrozen_panel():
    panel = _panel()
    panel["status"] = "draft"
    try:
        compile_study(panel)
    except ValueError as exc:
        assert "frozen" in str(exc)
    else:
        raise AssertionError("expected ValueError")
