import numpy as np
import json
from pathlib import Path
import pytest

from scripts.tools.silver_match_v3.retrieve_calibration import (
    hydrate_items,
    reserve_frozen_test,
    task_candidates,
    uses_nemotron_query_format,
)


def test_hydrate_is_noop_when_norm_present():
    rows = [{"norm_uid": "u", "norm": "be clear"}]
    assert hydrate_items(rows, {"corpora": {}}) == rows


def test_task_candidates_applies_query_formatter_to_dense_queries_only():
    class DummyModel:
        def __init__(self):
            self.calls = []

        def encode(self, texts, **kwargs):
            self.calls.append(list(texts))
            # One-dimensional normalized embeddings are enough for this routing test.
            return np.ones((len(texts), 1), dtype=np.float32)

    model = DummyModel()
    rows = [{
        "schema_version": "v",
        "norm_uid": "a" * 64,
        "corpus": "c",
        "task": "t",
        "row": 0,
        "norm": "Prefer precise wording",
        "context": "Prefer precise wording in the conclusion.",
    }]
    metrics = [{
        "metric_id": "m1",
        "name": "Precision",
        "description": "Uses precise wording.",
        "examples": [],
    }]
    output = task_candidates(
        model,
        rows,
        metrics,
        component_k=1,
        output_k=1,
        query_formatter=lambda query: "INSTRUCTED: " + query,
    )
    assert not model.calls[0][0].startswith("INSTRUCTED:")  # metric card
    assert model.calls[1][0].startswith("INSTRUCTED:")
    assert output[0]["candidates"][0]["metric_id"] == "m1"
    assert output[0]["candidates"][0]["metric_index"] == 0


def test_nemotron_query_format_auto_detection():
    assert uses_nemotron_query_format("auto", "/models/llama-embed-nemotron-8b", None)
    assert uses_nemotron_query_format("auto", "/models/anything", "/adapter")
    assert not uses_nemotron_query_format("raw", "/models/nemotron", "/adapter")
    assert uses_nemotron_query_format("nemotron", "/models/bge", None)


def test_task_candidates_applies_frozen_component_weights():
    class DummyModel:
        def encode(self, texts, **kwargs):
            if len(texts) == 2:  # two bank cards
                return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            return np.asarray([[0.0, 1.0]], dtype=np.float32)

    rows = [
        {
            "schema_version": "v",
            "norm_uid": "b" * 64,
            "corpus": "c",
            "task": "t",
            "row": 0,
            "norm": "Prefer the second construct",
        }
    ]
    metrics = [
        {"metric_id": "m1", "name": "First", "description": "first", "examples": []},
        {"metric_id": "m2", "name": "Second", "description": "second", "examples": []},
    ]
    weights = {
        "dense_rank": 1.0,
        "dense_statement_rank": 0.0,
        "word_rank": 0.0,
        "word_statement_rank": 0.0,
        "char_rank": 0.0,
        "char_statement_rank": 0.0,
    }
    output = task_candidates(
        DummyModel(),
        rows,
        metrics,
        component_k=2,
        output_k=2,
        component_weights=weights,
    )
    assert [row["metric_id"] for row in output[0]["candidates"]] == ["m2", "m1"]


def test_frozen_test_reservation_is_atomic_and_records_hashes(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    items = tmp_path / "items.jsonl"
    selection = tmp_path / "selection.json"
    manifest.write_text("{}")
    items.write_text("{}\n")
    selection.write_text("{}")
    marker = tmp_path / "test.started.json"
    output = tmp_path / "candidates.jsonl"
    payload = reserve_frozen_test(
        marker,
        output_path=output,
        manifest_path=manifest,
        items_path=items,
        selection_record_path=selection,
        adapter_path=None,
    )
    assert json.loads(marker.read_text())["inputs"]["items"]["sha256"]
    assert payload["status"] == "STARTED_TEST_INPUT_CONSUMED"
    with pytest.raises(FileExistsError, match="already consumed"):
        reserve_frozen_test(
            marker,
            output_path=output,
            manifest_path=manifest,
            items_path=items,
            selection_record_path=selection,
            adapter_path=None,
        )
