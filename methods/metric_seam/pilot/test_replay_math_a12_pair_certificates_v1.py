from __future__ import annotations

import json
from pathlib import Path

import pytest

from methods.metric_seam.battery.seal_ctext_train_view_v3 import prepare_train_view
from methods.metric_seam.battery.technical_entry_v1 import normalize_depth
from methods.metric_seam.pilot._math_a12_pair_certificate_worker_v1 import (
    project_document,
)
from methods.metric_seam.pilot.evaluate_math_a12_symbolic_step_heldout_v1 import (
    execute_heldout,
)
from methods.metric_seam.pilot.replay_math_a12_pair_certificates_v1 import (
    replay_pair_certificates,
)


ROOT = Path(__file__).resolve().parents[3]
OPERATION = ROOT / "methods/metric_seam/hybrids/ops_symbolic_steps_v1.py"
RELATION_CONTRACT = (
    ROOT
    / "methods/metric_seam/contracts/math_a12_symbolic_step_relation_contract_v1.json"
)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "items.json"
    source.write_text(
        json.dumps(
            [
                {
                    "datapoint_id": f"d{index:02d}",
                    "judgement": index % 2,
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
        ),
        encoding="utf-8",
    )
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps({"construct_definition": "Precision and rigor"}),
        encoding="utf-8",
    )
    preparation = tmp_path / "preparation"
    prepare_train_view(
        source=source,
        contract_path=contract,
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
    execution = tmp_path / "execution"
    execute_heldout(preparation_dir=preparation, output_dir=execution)
    return preparation, execution


def test_project_document_retains_identity_and_counterexample() -> None:
    projected = project_document(
        "Question: q\nAnswer: "
        "$$\\frac{1}{x}+1=\\frac{x+1}{x} \\\\ x+x=x$$"
    )
    certificates = projected["certificates"]
    statuses = {certificate["status"] for certificate in certificates}
    assert "verified_rational_identity" in statuses
    assert "exact_nonidentity_witness" in statuses
    identity = next(
        certificate
        for certificate in certificates
        if certificate["status"] == "verified_rational_identity"
    )
    nonidentity = next(
        certificate
        for certificate in certificates
        if certificate["status"] == "exact_nonidentity_witness"
    )
    assert identity["expression_pair"]["lhs_sympy_canonical"] is not None
    assert identity["domain_nonzero_obligations"] == ["x != 0"]
    assert nonidentity["counterexample_assignment"] is not None
    assert nonidentity["criterion_defect_witness"] is False


@pytest.mark.xfail(
    strict=False,
    reason=(
        "historical frozen-v1 integration uses a 0.5s ITIMER_REAL parse budget; "
        "cold SymPy/Lark initialization can exceed it on a loaded CPU"
    ),
)
def test_projection_matches_sealed_execution_and_depth_schema(tmp_path: Path) -> None:
    preparation, execution = _fixture(tmp_path)
    output = tmp_path / "projection"
    replay_pair_certificates(
        preparation_dir=preparation,
        execution_dir=execution,
        output_dir=output,
    )
    summary = json.loads((output / "projection_summary.json").read_text())
    assert summary["sealed_v1_row_classifications_exact"] is True
    assert summary["sealed_v1_aggregate_exact"] is True
    assert summary["reference_loaded_or_used_by_replay"] is False
    assert summary["temporal_status"] == "post_reference_projection_replay"
    depth = json.loads((output / "relation_depth.json").read_text())
    normalized = normalize_depth(
        depth,
        heldout_count=summary["heldout_count"],
        candidate_sha256=depth["candidate_sha256"],
        criterion_id="math__a12",
        relation_id="explicit_rational_equality_preservation",
        universe_sha256=depth["universe_sha256"],
    )
    assert normalized["static_max_relation_depth"] == 3
    assert sum(normalized["dynamic_contributing_depth_histogram"].values()) == 2

    certificates = [
        json.loads(line)
        for line in (output / "pair_certificates.jsonl").read_text().splitlines()
    ]
    assert len(certificates) == summary["pair_certificate_count"]
    assert all("pair_sha256" in certificate for certificate in certificates)
