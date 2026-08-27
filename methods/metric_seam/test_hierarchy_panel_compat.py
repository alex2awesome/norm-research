from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import methods.metric_seam.hierarchy_panel_compat as compat


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "outputs/metric_seam_pilot/hierarchy_r123/panel_v3.json"


def _load_panel() -> dict:
    return json.loads(PANEL.read_text(encoding="utf-8"))


def _rebind(panel: dict) -> None:
    core = {key: value for key, value in panel.items() if key != "panel_content_sha256"}
    panel["panel_content_sha256"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_frozen_v1_panel_passes_its_own_complete_contract():
    assert compat.validate_frozen_v1_panel(_load_panel()) == []


def test_frozen_v1_validator_rejects_missing_dependency_metadata_after_rebinding():
    panel = copy.deepcopy(_load_panel())
    panel["cells"][0].pop("dependency_component_id")
    _rebind(panel)
    errors = compat.validate_frozen_v1_panel(panel)
    assert any("missing frozen v1 fields" in error for error in errors)


def test_non_v1_panels_still_dispatch_to_the_live_strict_validator(monkeypatch):
    called = []

    def fake_live_validator(panel: dict) -> list[str]:
        called.append(panel)
        return ["live validator result"]

    monkeypatch.setattr(compat, "validate_metric_panel", fake_live_validator)
    panel = {"schema": "tacit_breadth_metric_panel/v3"}
    assert compat.validate_hierarchy_panel(panel) == ["live validator result"]
    assert called == [panel]
