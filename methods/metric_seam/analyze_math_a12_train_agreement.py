#!/usr/bin/env python3
"""Read out Math-a12 V_symbolic/V_llm agreement on compiler TRAIN pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from methods.metric_seam.verifiers.certificates import (
    applies_agreement,
    jointly_applicable_polarity_agreement,
)
from methods.metric_seam.verifiers.math_a12_llm_contract import (
    PARSER_VERSION,
    REQUEST_SCHEMA,
    validate_response_envelope,
)
from methods.metric_seam.verifiers.schema import Verdict


def _stats(value: Any) -> dict[str, object]:
    return {
        "n": value.n,
        "agreements": value.agreements,
        "observed_agreement": value.observed_agreement,
        "kappa": value.kappa,
    }


def analyze(
    *,
    symbolic: Mapping[str, Any],
    requests: Sequence[dict],
    responses: Sequence[dict],
) -> dict[str, Any]:
    if symbolic.get("schema") != "metric-seam.math-a12-symbolic-train-verifier.v1":
        raise ValueError("unexpected symbolic TRAIN schema")
    if symbolic.get("split") != "compiler_train" or symbolic.get("heldout_accessed") is not False:
        raise ValueError("symbolic readout is not compiler TRAIN only")
    ast_by_pair: dict[str, Verdict] = {}
    for row in symbolic["natural_pairs"]:
        pair_id = f"{row['item_key']}.{row['pair_id']}"
        ast_by_pair[pair_id] = Verdict.from_json(row["verdict"])
    request_by_digest = {row["request_sha256"]: row for row in requests}
    if len(request_by_digest) != len(requests):
        raise ValueError("duplicate request digest")
    if not requests or any(
        row.get("schema") != REQUEST_SCHEMA
        or row.get("split") != "compiler_train"
        or row.get("relation_id") != "explicit_rational_equality_preservation"
        or row.get("response_contract", {}).get("parser_version") != PARSER_VERSION
        for row in requests
    ):
        raise ValueError("request bundle schema/split/relation/parser drift")
    pass_indices = sorted({row["pass_index"] for row in requests})
    request_pairs_by_pass = {
        pass_index: {row["pair"]["pair_id"] for row in requests if row["pass_index"] == pass_index}
        for pass_index in pass_indices
    }
    if any(pairs != set(ast_by_pair) for pairs in request_pairs_by_pass.values()):
        raise ValueError("request pair universe differs from the symbolic pair universe")
    if len(requests) != len(ast_by_pair) * len(pass_indices):
        raise ValueError("request bundle is not exactly one request per pair/pass")
    response_by_digest: dict[str, dict] = {}
    invalid_response_count = 0
    response_status_counts: dict[str, int] = {}
    parse_mode_counts: dict[str, int] = {}
    recovery_kind_counts: dict[str, int] = {}
    for row in responses:
        digest = row.get("request_sha256")
        if digest not in request_by_digest or digest in response_by_digest:
            raise ValueError("unknown or duplicate response digest")
        status = str(row.get("status"))
        response_status_counts[status] = response_status_counts.get(status, 0) + 1
        if status != "valid":
            invalid_response_count += 1
            continue
        validated = validate_response_envelope(row, request_by_digest[digest])
        if row.get("validated_response") != validated:
            raise ValueError("retained response does not replay")
        mode = str(validated["parse_mode"])
        parse_mode_counts[mode] = parse_mode_counts.get(mode, 0) + 1
        recovery = row.get("recovery")
        if isinstance(recovery, dict) and isinstance(recovery.get("kind"), str):
            kind = recovery["kind"]
            recovery_kind_counts[kind] = recovery_kind_counts.get(kind, 0) + 1
        response_by_digest[digest] = validated

    pair_pass: dict[int, dict[str, Verdict]] = {1: {}, 2: {}}
    for digest, validated in response_by_digest.items():
        request = request_by_digest[digest]
        pair_pass[request["pass_index"]][request["pair"]["pair_id"]] = Verdict.from_json(
            validated["verdict"]
        )

    by_pass: dict[str, Any] = {}
    for pass_index in pass_indices:
        common = sorted(set(ast_by_pair) & set(pair_pass[pass_index]))
        if not common:
            by_pass[str(pass_index)] = {"n": 0, "status": "no_valid_responses"}
            continue
        ast = [ast_by_pair[pair_id] for pair_id in common]
        llm = [pair_pass[pass_index][pair_id] for pair_id in common]
        applicability = applies_agreement(ast, llm)
        polarity = jointly_applicable_polarity_agreement(ast, llm)
        applicability_matrix = {
            "both_not_applicable": sum(not a.applies and not b.applies for a, b in zip(ast, llm)),
            "symbolic_only_applies": sum(a.applies and not b.applies for a, b in zip(ast, llm)),
            "llm_only_applies": sum(not a.applies and b.applies for a, b in zip(ast, llm)),
            "both_apply": sum(a.applies and b.applies for a, b in zip(ast, llm)),
        }
        polarity_matrix = {
            "both_satisfied": sum(
                a.applies and b.applies and not a.violated and not b.violated
                for a, b in zip(ast, llm)
            ),
            "both_violated": sum(
                a.applies and b.applies and a.violated and b.violated
                for a, b in zip(ast, llm)
            ),
            "symbolic_satisfied_llm_violated": sum(
                a.applies and b.applies and not a.violated and b.violated
                for a, b in zip(ast, llm)
            ),
            "symbolic_violated_llm_satisfied": sum(
                a.applies and b.applies and a.violated and not b.violated
                for a, b in zip(ast, llm)
            ),
        }
        by_pass[str(pass_index)] = {
            "n": len(common),
            "status": "complete" if len(common) == len(ast_by_pair) else "partial",
            "applicability_agreement": _stats(applicability),
            "jointly_applicable_polarity_agreement": _stats(polarity),
            "witness_readout": "not_estimated_shared_span_bound_by_construction",
            "ablation_readout": "not_run",
            "applicability_matrix": applicability_matrix,
            "jointly_applicable_polarity_matrix": polarity_matrix,
            "v_symbolic_state_counts": {
                state: sum(value.state == state for value in ast)
                for state in ("not_applicable", "satisfied", "violated")
            },
            "v_llm_state_counts": {
                state: sum(value.state == state for value in llm)
                for state in ("not_applicable", "satisfied", "violated")
            },
        }

    common_passes = (
        sorted(set(pair_pass[1]) & set(pair_pass[2]))
        if set(pass_indices) == {1, 2}
        else []
    )
    reliability = None
    if common_passes:
        one = [pair_pass[1][pair_id] for pair_id in common_passes]
        two = [pair_pass[2][pair_id] for pair_id in common_passes]
        reliability = {
            "n": len(common_passes),
            "applicability": _stats(applies_agreement(one, two)),
            "jointly_applicable_polarity": _stats(
                jointly_applicable_polarity_agreement(one, two)
            ),
        }
    expected = len(requests)
    return {
        "schema": "metric-seam.math-a12-train-agreement.v1",
        "status": (
            "complete"
            if len(response_by_digest) == expected and invalid_response_count == 0
            else "partial"
        ),
        "split": "compiler_train",
        "relation_id": "explicit_rational_equality_preservation",
        "measurement_unit": "extracted adjacent equality pair",
        "expected_request_count": expected,
        "valid_response_count": len(response_by_digest),
        "invalid_response_count": invalid_response_count,
        "response_status_counts": dict(sorted(response_status_counts.items())),
        "parse_mode_counts": dict(sorted(parse_mode_counts.items())),
        "recovery_kind_counts": dict(sorted(recovery_kind_counts.items())),
        "parser_version": PARSER_VERSION,
        "model": requests[0]["model"],
        "pass_indices": pass_indices,
        "symbolic_pair_count": len(ast_by_pair),
        "by_pass": by_pass,
        "v_llm_pass_reliability": reliability,
        "claim_limits": [
            "This is exploratory model-specific TRAIN agreement, not a held-out verifiability certificate.",
            "Both arms receive equality pairs proposed by the same structural extractor; independent relation discovery is not measured.",
            "Witness agreement is structurally bound by the supplied pair span and is not estimated.",
            "Witness ablation was not run and no ablation claim is made.",
            "The operational target is rational-expression identity/nonidentity on an inferred algebraic domain, not a document-declared domain.",
            "The target is not whole-proof rigor, whole-document reconstruction, or whole-metric isomorphism.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbolic-readout", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--parser-calibration-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    symbolic = json.loads(args.symbolic_readout.read_text(encoding="utf-8"))
    requests = [json.loads(line) for line in args.requests.read_text(encoding="utf-8").splitlines()]
    responses = [json.loads(line) for line in args.responses.read_text(encoding="utf-8").splitlines()]
    value = analyze(symbolic=symbolic, requests=requests, responses=responses)
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    value["sources"] = {
        "symbolic_readout": {"path": str(args.symbolic_readout), "sha256": digest(args.symbolic_readout)},
        "requests": {"path": str(args.requests), "sha256": digest(args.requests)},
        "responses": {"path": str(args.responses), "sha256": digest(args.responses)},
        "parser_calibration_receipt": {
            "path": str(args.parser_calibration_receipt),
            "sha256": digest(args.parser_calibration_receipt),
        },
    }
    calibration = json.loads(args.parser_calibration_receipt.read_text(encoding="utf-8"))
    if (
        calibration.get("schema")
        != "metric-seam.math-a12-post-smoke-parser-calibration.v1"
        or calibration.get("revision", {}).get("parser_version") != PARSER_VERSION
    ):
        raise ValueError("parser calibration receipt does not bind the active parser")
    value["post_smoke_parser_calibration"] = {
        "status": calibration["status"],
        "original_smoke_status_counts": calibration["original_smoke"]["status_counts"],
        "replayed_valid": calibration["revision"]["replayed_valid"],
        "prompt_changed": calibration["revision"]["prompt_changed"],
        "model_calls_repeated": calibration["revision"]["model_calls_repeated"],
        "request_sha256_unchanged": calibration["digest_boundary"]["request_sha256_unchanged"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
