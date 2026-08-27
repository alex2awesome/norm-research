from __future__ import annotations

import json

from methods.metric_seam.analyze_math_a12_train_agreement import analyze
from methods.metric_seam.verifiers.math_a12_llm_contract import (
    RationalExpressionPair,
    compile_request,
    validate_response_envelope,
)
from methods.metric_seam.verifiers.schema import Span, Verdict


def test_analysis_reports_train_proxy_and_two_pass_reliability() -> None:
    span = Span("answer.md", 2, 2)
    pair = RationalExpressionPair("train_0001.pair-1", "x+x", "2x", span, span)
    requests = [
        compile_request(pair=pair, pass_index=index, model="sonnet", split="compiler_train")
        for index in (1, 2)
    ]
    raw = json.dumps(
        {"applies": True, "violated": False, "witnesses": [span.to_json_value()]}
    )
    responses = []
    for index, request in enumerate(requests):
        envelope = {
            "request_sha256": request["request_sha256"],
            "request_index": index,
            "status": "valid",
            "raw_response": raw,
        }
        envelope["validated_response"] = validate_response_envelope(envelope, request)
        responses.append(envelope)
    symbolic = {
        "schema": "metric-seam.math-a12-symbolic-train-verifier.v1",
        "split": "compiler_train",
        "heldout_accessed": False,
        "natural_pairs": [
            {
                "item_key": "train_0001",
                "pair_id": "pair-1",
                "verdict": Verdict(True, False, (Span("answer.md", 2, 2, node_id="pair-1"),)).to_json_value(),
            }
        ]
    }
    result = analyze(symbolic=symbolic, requests=requests, responses=responses)
    assert result["status"] == "complete"
    assert result["by_pass"]["1"]["witness_readout"].startswith("not_estimated")
    assert result["by_pass"]["1"]["ablation_readout"] == "not_run"
    assert result["by_pass"]["1"]["applicability_matrix"]["both_apply"] == 1
    assert result["by_pass"]["1"]["jointly_applicable_polarity_matrix"]["both_satisfied"] == 1
    # No violated rows makes the ablation and kappa certificate intentionally undefined.
    assert result["v_llm_pass_reliability"]["n"] == 1
