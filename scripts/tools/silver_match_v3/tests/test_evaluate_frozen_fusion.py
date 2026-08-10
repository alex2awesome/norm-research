import json
from pathlib import Path

import numpy as np
import pytest

from scripts.tools.silver_match_v3.evaluate_frozen_fusion import (
    model_fingerprint,
    sliced_metrics,
    validate_frozen_model,
)


def test_sliced_metrics_keeps_panel_breakdown():
    labels = [
        {"norm_uid": "a", "metric_id": "m1", "human_panel": "old", "corpus": "c"},
        {"norm_uid": "b", "metric_id": "m2", "human_panel": "new", "corpus": "c"},
    ]
    result = sliced_metrics(np.asarray([1, 60]), labels, 100)
    assert result["all"]["recall_at_50"] == 0.5
    assert result["by_human_panel"]["old"]["recall_at_50"] == 1.0
    assert result["by_human_panel"]["new"]["recall_at_50"] == 0.0


def test_model_fingerprint_mismatch_is_rejected(tmp_path: Path):
    dev = tmp_path / "dev.jsonl"
    test = tmp_path / "test.jsonl"
    dev.write_text("")
    test.write_text("")
    base = {
        "manifest_sha256": "m", "encoder": "base", "adapter_hashes": None,
        "query_format": "nemotron", "dense_query_instruction": True,
        "query_views": "evidence+statement", "component_k": 100, "output_k": 100,
    }
    dev.with_suffix(".jsonl.meta.json").write_text(json.dumps(base))
    changed = {**base, "encoder": "other"}
    test.with_suffix(".jsonl.meta.json").write_text(json.dumps(changed))
    fusion = {"candidate_inputs": {str(dev): "sha"}}
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_frozen_model(test, fusion, require_test_marker=False)
    assert model_fingerprint(base)["encoder"] == "base"
