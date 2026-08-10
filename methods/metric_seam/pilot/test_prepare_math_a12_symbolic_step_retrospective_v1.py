"""Privacy and semantic tests for reproducible math-a12 TRAIN preparation."""

import json
from pathlib import Path

import pytest

from methods.metric_seam.pilot.prepare_math_a12_symbolic_step_retrospective_v1 import (
    AUTHORSHIP_STATUS,
    prepare,
)


def test_prepare_emits_only_sanitized_opaque_train_and_unscoped_nonidentity(tmp_path: Path):
    source = tmp_path / "items.json"
    source.write_text(
        json.dumps(
            [
                {
                    "datapoint_id": f"source_{index}",
                    "ctext": text,
                    "judgement": index % 2,
                }
                for index, text in enumerate(
                    [
                        "Question: Expand.\n\nAnswer: $(x+1)^2=x^2+2x+1$.",
                        "Question: Solve.\n\nAnswer: $x=1$.",
                        "Question: Why?\n\nAnswer: By compactness.",
                        r"Question: Check.\n\nAnswer: $\sin^2x+\cos^2x=1$.",
                        r"Question: Cancel.\n\nAnswer: $\frac{x^2-1}{x-1}=x+1$.",
                        "Question: Add.\n\nAnswer: $2+2=4$.",
                    ]
                )
            ]
        ),
        encoding="utf-8",
    )
    construct = tmp_path / "construct.json"
    construct.write_text(
        json.dumps(
            {
                "construct_definition": "Eliminate ambiguity and algebraic errors.",
                "cf_probes": [],
                "boundary_notes": "Whole-proof rigor is broader.",
            }
        ),
        encoding="utf-8",
    )
    relation = tmp_path / "relation.json"
    relation.write_text(json.dumps({"relation_id": "rational_identity"}))

    out_dir = tmp_path / "prepared"
    bundle_path, manifest_path, summary_path, report_path = prepare(
        source=source,
        construct_contract=construct,
        relation_contract=relation,
        out_dir=out_dir,
        train_count=5,
        split_seed=7,
    )
    bundle_text = bundle_path.read_text(encoding="utf-8")
    bundle = json.loads(bundle_text)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")

    assert all(set(row) == {"ctext", "item_key"} for row in bundle["train_items"])
    assert [row["item_key"] for row in bundle["train_items"]] == [
        f"train_{index:04d}" for index in range(1, 6)
    ]
    assert "source_" not in bundle_text
    assert "judgement" not in bundle_text
    assert manifest["partition"]["heldout_rows_materialized"] is False
    assert manifest["environment"]["packages"]["sympy"]
    assert manifest["environment"]["packages"]["lark"]
    assert set(manifest["implementation"]) == {
        "preparer",
        "relation_contract",
        "symbolic_operation",
    }
    assert summary["authorship_status"] == AUTHORSHIP_STATUS
    assert summary["coverage"]["universal_identity_counterexample_count"] == 0
    assert summary["coverage"]["exact_nonidentity_witness_count"] >= 1
    assert summary["coverage"]["criterion_defect_witness_count"] == 0
    assert summary["reference_accessed"] is False
    assert summary["heldout_accessed"] is False
    assert "ctext" not in summary
    assert "selected_retrospective_seed_with_aggregate_train_summary_exposure" in report
    assert "No executable pair means abstention" in report

    with pytest.raises(FileExistsError):
        prepare(
            source=source,
            construct_contract=construct,
            relation_contract=relation,
            out_dir=out_dir,
            train_count=5,
            split_seed=7,
        )
