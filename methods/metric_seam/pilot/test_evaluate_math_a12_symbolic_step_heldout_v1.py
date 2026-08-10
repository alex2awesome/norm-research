from __future__ import annotations

import json
from pathlib import Path
import random

import pytest

from methods.metric_seam.battery.seal_ctext_train_view_v3 import prepare_train_view
from methods.metric_seam.pilot.evaluate_math_a12_symbolic_step_heldout_v1 import (
    HeldoutIntegrityError,
    execute_heldout,
    finalize_reference,
)


ROOT = Path(__file__).resolve().parents[3]
OPERATION = ROOT / "methods/metric_seam/hybrids/ops_symbolic_steps_v1.py"
RELATION_CONTRACT = (
    ROOT
    / "methods/metric_seam/contracts/math_a12_symbolic_step_relation_contract_v1.json"
)


def _prepare_fixture(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    source = tmp_path / "items.json"
    rows = [
        {
            "datapoint_id": f"d{index:02d}",
            "judgement": index % 2,
            "text": "forbidden raw source field",
            "ctext": text,
        }
        for index, text in enumerate(
            [
                "Question: q\nAnswer: $$x+x=2x$$",
                "Question: q\nAnswer: $$x+x=x$$",
                "Question: q\nAnswer: prose only",
                "Question: q\nAnswer: $$\\frac{x}{x}=1$$",
            ]
        )
    ]
    source.write_text(json.dumps(rows), encoding="utf-8")
    construct = tmp_path / "construct.json"
    construct.write_text(
        json.dumps({"construct_definition": "Precision and rigor"}),
        encoding="utf-8",
    )
    preparation = tmp_path / "preparation"
    prepare_train_view(
        source=source,
        contract_path=construct,
        out_dir=preparation,
        task="math",
        criterion_id="a12",
        train_count=2,
        split_seed=7,
        dependency_files={
            "relation_contract": RELATION_CONTRACT,
            "symbolic_operation": OPERATION,
        },
        dependency_packages=("sympy", "lark"),
    )
    ids = sorted(row["datapoint_id"] for row in rows)
    random.Random(7).shuffle(ids)
    return preparation, source, sorted(ids[2:])


def test_execute_precedes_reference_and_emits_no_parent_scalar(tmp_path: Path) -> None:
    preparation, _, heldout_ids = _prepare_fixture(tmp_path)
    execution_dir = tmp_path / "execution"
    execute_heldout(preparation_dir=preparation, output_dir=execution_dir)
    execution = json.loads((execution_dir / "candidate_execution.json").read_text())
    assert execution["reference_accessed"] is False
    assert execution["whole_criterion_scalar"] is None
    assert execution["candidate_reference_correlation"] is None
    assert {row["datapoint_id"] for row in execution["rows"]} == set(heldout_ids)
    assert execution["summary"]["criterion_defect_witness_count"] == 0

    reference = tmp_path / "reference.jsonl"
    reference.write_text(
        "".join(
            json.dumps(
                {
                    "aspect_id": "a12",
                    "datapoint_id": datapoint_id,
                    "channel": channel,
                    "score": score,
                }
            )
            + "\n"
            for datapoint_id in heldout_ids
            for channel, score in (("pass1", 4), ("pass2", 6))
        ),
        encoding="utf-8",
    )
    final_dir = tmp_path / "final"
    finalize_reference(
        execution_dir=execution_dir,
        reference_path=reference,
        output_dir=final_dir,
    )
    final = json.loads((final_dir / "finalization.json").read_text())
    assert final["prompt_reference"]["candidate_completed_before_reference_load"] is True
    assert final["candidate_parent_scalar"] is None
    assert final["candidate_reference_correlation"] is None
    assert final["whole_criterion_reconstruction"] == "NOT_ESTIMATED"
    assert final["isomorphism"] == "NOT_ESTIMATED"


def test_finalize_rejects_tampered_execution(tmp_path: Path) -> None:
    preparation, _, _ = _prepare_fixture(tmp_path)
    execution_dir = tmp_path / "execution"
    execute_heldout(preparation_dir=preparation, output_dir=execution_dir)
    execution_path = execution_dir / "candidate_execution.json"
    execution_path.chmod(0o644)
    execution = json.loads(execution_path.read_text())
    execution["whole_criterion_scalar"] = 0.5
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    reference = tmp_path / "reference.jsonl"
    reference.write_text("", encoding="utf-8")
    with pytest.raises(HeldoutIntegrityError, match="changed after sealing"):
        finalize_reference(
            execution_dir=execution_dir,
            reference_path=reference,
            output_dir=tmp_path / "final",
        )
