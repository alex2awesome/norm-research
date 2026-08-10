from __future__ import annotations

import json
from pathlib import Path

from methods.metric_seam.battery.seal_ctext_items_v2 import canonical_bytes, sha256
from methods.metric_seam.battery.technical_entry_v1 import normalize_depth
from methods.metric_seam.pilot.audit_math_a12_relation_depth_multiview_v1 import (
    AUDIT_SCHEMA,
    CRITERION_ID,
    MANIFEST_SCHEMA,
    RELATION_ID,
    audit_relation_depth,
)
from methods.metric_seam.pilot.replay_math_a12_pair_certificates_v1 import (
    MANIFEST_SCHEMA as PROJECTION_MANIFEST_SCHEMA,
)


ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / "methods/metric_seam/pilot/_math_a12_pair_certificate_worker_v1.py"
REPLAY = ROOT / "methods/metric_seam/pilot/replay_math_a12_pair_certificates_v1.py"


def _write(path: Path, value: object) -> None:
    payload = value if isinstance(value, bytes) else canonical_bytes(value)
    path.write_bytes(payload)


def _analysis(*, candidates: int, parsed: int, positive: int) -> dict[str, object]:
    return {
        "pair_candidate_count": candidates,
        "parsed_rational_pair_count": parsed,
        "positive_code_witness_count": positive,
        "abstained": parsed == 0,
    }


def _projection_fixture(tmp_path: Path) -> Path:
    projection = tmp_path / "projection"
    projection.mkdir()
    rows = {
        "rows": [
            {
                "datapoint_id": "no_pair",
                "certificate_count": 0,
                "analysis": _analysis(candidates=0, parsed=0, positive=0),
            },
            {
                "datapoint_id": "parse_failure",
                "certificate_count": 1,
                "analysis": _analysis(candidates=1, parsed=0, positive=0),
            },
            {
                "datapoint_id": "positive",
                "certificate_count": 1,
                "analysis": _analysis(candidates=1, parsed=1, positive=1),
            },
        ]
    }
    certificates = [
        {
            "datapoint_id": "parse_failure",
            "status": "parse_noncoverage",
            "positive_code_witness": False,
        },
        {
            "datapoint_id": "positive",
            "status": "verified_rational_identity",
            "positive_code_witness": True,
        },
    ]
    old_depth = {
        "scale": "metric-seam.relation-depth.v1",
        "criterion_id": CRITERION_ID,
        "relation_id": RELATION_ID,
        "candidate_sha256": "a" * 64,
        "universe_sha256": (
            "534386b17d742d55ae3848e440908812d8890f7806436ee059430837d5df4a0c"
        ),
        "nodes": [
            {
                "node_id": "parser",
                "implementation": "code",
                "relation_depth": 1,
                "contributes_to_output": True,
            },
            {
                "node_id": "formal",
                "implementation": "code",
                "relation_depth": 3,
                "contributes_to_output": True,
            },
        ],
        "static_max_relation_depth": 3,
        "longest_path_edges": 1,
        "dynamic_contributing_depth_histogram": {"1": 2, "3": 1},
    }
    summary = {
        "schema": "metric-seam.math-a12-pair-certificate-projection.v1",
        "heldout_count": 3,
        "pair_certificate_count": 2,
    }
    _write(projection / "pair_certificates.jsonl", b"".join(canonical_bytes(v) for v in certificates))
    _write(projection / "row_projection.json", rows)
    _write(projection / "relation_depth.json", old_depth)
    _write(projection / "projection_summary.json", summary)
    artifacts = {
        name: {"sha256": sha256(projection / name)}
        for name in (
            "pair_certificates.jsonl",
            "row_projection.json",
            "relation_depth.json",
            "projection_summary.json",
        )
    }
    manifest = {
        "schema": PROJECTION_MANIFEST_SCHEMA,
        "reference_loaded_or_used": False,
        "artifacts": artifacts,
        "execution": {
            "worker": {"path": str(WORKER), "sha256": sha256(WORKER)},
            "replay": {"path": str(REPLAY), "sha256": sha256(REPLAY)},
        },
    }
    _write(projection / "projection_manifest.json", manifest)
    return projection


def test_multiview_depth_separates_attempt_from_positive_evidence(
    tmp_path: Path,
) -> None:
    projection = _projection_fixture(tmp_path)
    output = tmp_path / "audit"
    audit_relation_depth(projection_dir=projection, output_dir=output)

    summary = json.loads((output / "audit_summary.json").read_text())
    assert summary["schema"] == AUDIT_SCHEMA
    assert summary["depth_views"]["deepest_attempted"]["histogram"] == {
        "1": 1,
        "3": 2,
    }
    assert summary["depth_views"]["deepest_decision_contributing"][
        "histogram"
    ] == {"1": 1, "3": 2}
    assert summary["depth_views"]["positive_relation_evidence"] == {
        "histogram": {"3": 1},
        "evidence_rows": 1,
        "no_positive_evidence_rows": 2,
        "semantics": "depth only for rows with at least one positive code witness",
    }

    depth = json.loads((output / "relation_depth_multiview.json").read_text())
    assert depth["dynamic_contributing_depth_histogram"] == {"1": 1, "3": 2}
    normalized = normalize_depth(
        depth,
        heldout_count=3,
        candidate_sha256="a" * 64,
        criterion_id=CRITERION_ID,
        relation_id=RELATION_ID,
        universe_sha256=depth["universe_sha256"],
    )
    assert normalized["dynamic_contributing_depth_histogram"] == {
        "0": 0,
        "1": 1,
        "2": 0,
        "3": 2,
        "4": 0,
    }
    manifest = json.loads((output / "audit_manifest.json").read_text())
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["reference_loaded_or_used"] is False


def test_multiview_depth_rejects_projection_certificate_drift(tmp_path: Path) -> None:
    projection = _projection_fixture(tmp_path)
    (projection / "pair_certificates.jsonl").write_text("{}\n", encoding="utf-8")
    try:
        audit_relation_depth(projection_dir=projection, output_dir=tmp_path / "audit")
    except ValueError as exc:
        assert "artifact mismatch" in str(exc)
    else:
        raise AssertionError("projection drift should have been rejected")
