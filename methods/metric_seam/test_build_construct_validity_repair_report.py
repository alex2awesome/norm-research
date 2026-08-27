import json
from pathlib import Path

from methods.metric_seam.build_construct_validity_repair_report import build_summary


ROOT = Path(__file__).resolve().parents[2]


def _load(relative: str):
    return json.loads((ROOT / relative).read_text())


def test_actual_repair_artifacts_have_three_distinct_stop_stages() -> None:
    value = build_summary(
        a12=_load("outputs/metric_seam_pilot/verifier_pipeline_v2/math.a12.rigor.contextual_equation_use/construct_counterexample.json"),
        math_agreement=_load("outputs/metric_seam_pilot/hierarchy_r123/results/math_a12_train_agreement_v1/readout.json"),
        code_review=_load("outputs/metric_seam_pilot/hierarchy_r123/results/code_review_ast_train_v2/readout.json"),
        patent_probe=_load("outputs/metric_seam_pilot/verifier_pipeline_v2/patents.antecedent_basis.bounded_claim_graph/transport_v2/base_rate_probe.json"),
        patent_code=_load("outputs/metric_seam_pilot/verifier_pipeline_v2/patents.antecedent_basis.bounded_claim_graph/imported_code_gate.json"),
        sources={},
    )
    assert value["results"]["math_a12"]["construct_correct"] == 0
    assert value["results"]["code_review"]["natural_gate_passed"] == 0
    patent = value["results"]["patent_antecedent"]
    assert patent["pre_authoring_prompt_probe"]["passed"] is True
    assert patent["imported_binary_code_train"]["passes_max_90_percent_violation_gate"] is False
    assert patent["heldout_accessed"] is False
