from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from methods.metric_seam.verifiers.cuf_snapshot import (
    CufValidationError,
    build_code_review_join,
    normalize_metric_name,
    snapshot_bank,
    validate_bank_bytes,
    write_new_json,
)


def _unit(node_id: int = 0, *, span: str = "a load-bearing unit") -> dict:
    return {
        "node_id": node_id,
        "level": 1,
        "span": span,
        "delta_free": 0.3,
        "p_free": 0.001,
        "delta_M": 0.2,
        "p_M": 0.002,
        "sign_stability": 1.0,
        "kappa": 0.1,
        "eps_ctx": 0.01,
        "verdict": "CERTIFIED-UNIT",
        "atom": "ATOM",
        "detect_free": True,
        "detect_M": True,
        "certified_lo": 0.2,
    }


def _bank_bytes(rows: list[dict]) -> bytes:
    return ("\n".join(json.dumps(row) for row in rows) + "\n").encode()


def _bank_row(metric: str, k: int) -> dict:
    return {"metric": metric, "k": k, "rows": [_unit(k)], "meta": {"seed": 0}}


def test_normalize_metric_name_is_conservative() -> None:
    assert normalize_metric_name("  Intention‑Revealing\nNaming ") == (
        normalize_metric_name("intention-revealing naming")
    )
    assert normalize_metric_name("Comments: clear, intentional, why-focused") != (
        normalize_metric_name("Comments: clear, intentional, and why-focused")
    )


def test_snapshot_is_content_addressed_exact_and_idempotent(tmp_path: Path) -> None:
    data = _bank_bytes([_bank_row("Metric One", 0), _bank_row("Metric Two", 1)])
    root = tmp_path / "snapshots"
    manifest = snapshot_bank(
        source="-",
        source_label="ssh://sk3/example/bank_units.jsonl",
        snapshot_root=root,
        task="code-review",
        executor="llama8b",
        stdin=io.BytesIO(data),
    )
    bank_path = Path(manifest["snapshot"]["bank_path"])
    assert bank_path.read_bytes() == data
    assert manifest["validation"]["bank_rows"] == 2
    assert manifest["validation"]["unit_rows"] == 2
    assert manifest["validation"]["certified_unit_rows"] == 2
    assert manifest["source"]["locator"] == "ssh://sk3/example/bank_units.jsonl"
    assert manifest["executor"] == "llama8b"
    assert len((root / "index.jsonl").read_text().splitlines()) == 1

    repeated = snapshot_bank(
        source="-",
        source_label="a-different-transport-label-for-the-same-bytes",
        snapshot_root=root,
        task="code-review",
        executor="llama8b",
        stdin=io.BytesIO(data),
    )
    assert repeated == manifest
    assert len((root / "index.jsonl").read_text().splitlines()) == 1


def test_validation_rejects_missing_unit_fields_before_snapshot(tmp_path: Path) -> None:
    bad = _bank_row("Metric", 0)
    del bad["rows"][0]["detect_M"]
    data = _bank_bytes([bad])
    with pytest.raises(CufValidationError, match="detect_M"):
        validate_bank_bytes(data)
    with pytest.raises(CufValidationError, match="detect_M"):
        snapshot_bank(
            source="-",
            source_label="fixture",
            snapshot_root=tmp_path / "snapshots",
            task="code-review",
            executor="llama8b",
            stdin=io.BytesIO(data),
        )
    assert not (tmp_path / "snapshots").exists()


def test_certified_count_uses_exact_bank_verdict_not_detector_flags() -> None:
    certified = _unit(0)
    detected_only = _unit(1)
    detected_only["verdict"] = "NOT-CERTIFIED-UNIT"
    rows, counts = validate_bank_bytes(
        _bank_bytes(
            [{"metric": "Metric", "k": 0, "rows": [certified, detected_only], "meta": {}}]
        )
    )
    assert len(rows) == 1
    assert counts["unit_rows"] == 2
    assert counts["certified_unit_rows"] == 1


def _write_program(path: Path, aspect_id: str, aspect_name: str) -> None:
    path.write_text(
        f"ASPECT_ID = {aspect_id!r}\nASPECT_NAME = {aspect_name!r}\n",
        encoding="utf-8",
    )


def test_join_auto_accepts_only_unique_exact_normalized_names(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    programs = repo / "programs"
    programs.mkdir()
    _write_program(programs / "a1.py", "a1", "Exact‑Name")
    _write_program(programs / "a2.py", "a2", "Ambiguous Name")
    _write_program(programs / "a3.py", "a3", "Semantic Drift")

    bank_data = _bank_bytes(
        [
            _bank_row("exact-name", 0),
            _bank_row("Ambiguous Name", 1),
            _bank_row(" ambiguous   name ", 2),
            _bank_row("A Different Semantic Name", 3),
        ]
    )
    manifest = snapshot_bank(
        source="-",
        source_label="fixture",
        snapshot_root=repo / "snapshots",
        task="code-review",
        executor="llama8b",
        stdin=io.BytesIO(bank_data),
    )
    snapshot_manifest = Path(manifest["snapshot"]["manifest_path"])

    cell_ids = ["cell-1", "cell-2", "cell-3"]
    cell_manifest = repo / "cells.json"
    cell_manifest.write_text(json.dumps({"cell_ids": cell_ids}), encoding="utf-8")
    fidelity = repo / "fidelity.json"
    fidelity.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "cell_id": cell_id,
                        "level": "R1",
                        "metric_name": f"Parent {index}",
                        "candidate": {
                            "aspect_id": f"a{index}",
                            "source_path": f"programs/a{index}.py",
                        },
                    }
                    for index, cell_id in enumerate(cell_ids, 1)
                ]
            }
        ),
        encoding="utf-8",
    )

    result = build_code_review_join(
        snapshot_manifest=snapshot_manifest,
        cell_manifest=cell_manifest,
        construct_fidelity=fidelity,
        repo_root=repo,
        expected_cells=3,
    )
    assert result["summary"]["status_counts"] == {
        "auto_accepted_exact_normalized_unique": 1,
        "queued_ambiguous_exact_normalized_for_sonnet": 1,
        "queued_unmatched_for_sonnet": 1,
    }
    by_id = {row["cell_id"]: row for row in result["rows"]}
    assert by_id["cell-1"]["selected_bank_metric"]["metric"] == "exact-name"
    assert by_id["cell-2"]["selected_bank_metric"] is None
    assert len(by_id["cell-2"]["exact_normalized_bank_candidates"]) == 2
    assert by_id["cell-3"]["selected_bank_metric"] is None
    assert by_id["cell-3"]["exact_normalized_bank_candidates"] == []
    assert len(result["review_queue"]) == 2
    assert result["join_policy"]["fuzzy_matching_performed"] is False
    assert result["join_policy"]["semantic_adjudication_performed"] is False


def test_write_new_json_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    write_new_json(path, {"v": 1})
    with pytest.raises(FileExistsError):
        write_new_json(path, {"v": 2})
    assert json.loads(path.read_text()) == {"v": 1}
