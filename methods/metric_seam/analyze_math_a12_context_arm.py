#!/usr/bin/env python3
"""Analyze the preregistered 443-pair Math-a12 full-context arm."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from methods.metric_seam.verifiers.certificates import (
    applies_agreement, jointly_applicable_polarity_agreement)
from methods.metric_seam.verifiers.math_a12_context_contract import validate_response_envelope
from methods.metric_seam.verifiers.schema import Span, Verdict

ROOT = Path(__file__).resolve().parents[2]
SYMBOLIC = ROOT / "outputs/metric_seam_pilot/hierarchy_r123/results/math_a12_symbolic_train_v1/readout.json"
REQ = ROOT / "outputs/metric_seam_pilot/hierarchy_r123/requests/math_a12_context_train_v1/requests.jsonl"
RESP = ROOT / "outputs/metric_seam_pilot/hierarchy_r123/results/math_a12_context_train_v1/responses.jsonl"
OUT = ROOT / "outputs/metric_seam_pilot/hierarchy_r123/results/math_a12_context_train_v1/readout.json"


def stat(x):
    return {"n": x.n, "agreements": x.agreements,
            "observed_agreement": x.observed_agreement, "kappa": x.kappa}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbolic", type=Path, default=SYMBOLIC)
    ap.add_argument("--requests", type=Path, default=REQ)
    ap.add_argument("--responses", type=Path, default=RESP)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    symbolic = json.loads(args.symbolic.read_text())
    requests = [json.loads(x) for x in args.requests.read_text().splitlines()]
    responses = [json.loads(x) for x in args.responses.read_text().splitlines()]
    if len(requests) != 443 or len({r["pair_id"] for r in requests}) != 443:
        raise ValueError("context request universe is not the frozen 443 pairs")
    by_digest = {r["request_sha256"]: r for r in requests}
    sym = {}
    for row in symbolic["natural_pairs"]:
        sym[f"{row['item_key']}.{row['pair_id']}"] = Verdict.from_json(row["verdict"])
    if set(sym) != {r["pair_id"] for r in requests}:
        raise ValueError("context and symbolic pair universes differ")

    valid, invalid = {}, []
    for envelope in responses:
        digest = envelope.get("request_sha256")
        if digest not in by_digest or digest in valid:
            raise ValueError("unknown or duplicate response digest")
        if not envelope.get("valid"):
            invalid.append({"request_sha256": digest,
                            "error": envelope.get("validation_error")})
            continue
        replay = validate_response_envelope(envelope, by_digest[digest])
        if replay != envelope.get("validated"):
            raise ValueError("retained validation does not replay")
        valid[replay["pair_id"]] = replay

    common = sorted(set(sym) & set(valid))
    sv = [sym[p] for p in common]
    cv = [Verdict(
        valid[p]["applies"], valid[p]["violated"],
        tuple(Span.from_json_value(w) for w in valid[p]["witnesses"])
        if valid[p]["applies"] else (),
    ) for p in common]
    app = applies_agreement(sv, cv)
    pol = jointly_applicable_polarity_agreement(sv, cv)
    roles = Counter(valid[p]["role"] for p in common)
    role_x_symbolic = defaultdict(Counter)
    for pair_id in common:
        role_x_symbolic[valid[pair_id]["role"]][sym[pair_id].state] += 1
    # The individuation gap is the number/rate of code-applicable pairs that
    # full context identifies as a different occasion.  It is descriptive and
    # role-resolved; no supervised external truth is introduced.
    code_app = [p for p in common if sym[p].applies]
    reclassified = [p for p in code_app if valid[p]["role"] != "asserted_identity_step"]
    gap_roles = Counter(valid[p]["role"] for p in reclassified)
    matrix = {
        "both_not_applicable": sum(not a.applies and not b.applies for a, b in zip(sv, cv)),
        "symbolic_only_applies": sum(a.applies and not b.applies for a, b in zip(sv, cv)),
        "context_only_applies": sum(not a.applies and b.applies for a, b in zip(sv, cv)),
        "both_apply": sum(a.applies and b.applies for a, b in zip(sv, cv)),
    }
    polarity_matrix = {
        "both_satisfied": sum(a.applies and b.applies and not a.violated and not b.violated for a,b in zip(sv,cv)),
        "both_violated": sum(a.applies and b.applies and a.violated and b.violated for a,b in zip(sv,cv)),
        "symbolic_satisfied_context_violated": sum(a.applies and b.applies and not a.violated and b.violated for a,b in zip(sv,cv)),
        "symbolic_violated_context_satisfied": sum(a.applies and b.applies and a.violated and not b.violated for a,b in zip(sv,cv)),
    }
    out = {
        "schema": "metric-seam.math-a12-context-readout.v1",
        "status": "complete" if len(valid) == 443 else "partial",
        "split": "compiler_train", "measurement_unit": "(lhs,rhs,document occasion)",
        "model": requests[0]["model"], "expected": 443,
        "valid": len(valid), "invalid": len(invalid), "invalid_rows": invalid,
        "role_distribution": dict(sorted(roles.items())),
        "role_by_symbolic_state": {k: dict(v) for k,v in sorted(role_x_symbolic.items())},
        "role_conditioned_individuation_gap": {
            "symbolic_applicable_n": len(code_app),
            "reclassified_non_asserted_n": len(reclassified),
            "reclassified_non_asserted_rate": len(reclassified) / len(code_app) if code_app else None,
            "reclassified_roles": dict(sorted(gap_roles.items())),
        },
        "applicability_agreement": stat(app),
        "applicability_matrix": matrix,
        "joint_asserted_identity_polarity_agreement": stat(pol),
        "joint_asserted_identity_polarity_matrix": polarity_matrix,
        "old_pair_only_control": {
            "valid_n": 437, "applicability_kappa": 0.44475223497899535,
            "joint_polarity_kappa": 1.0,
            "reading": "context-free symbolic-identity agreement; retracted as a construct-validity result",
        },
        "claim_limits": [
            "Role is an unsupervised Sonnet reconstruction, not an external ground-truth annotation.",
            "The shared structural extractor still proposes all pairs; whole-document relation discovery is not measured.",
            "Polarity agreement is interpreted only on jointly asserted_identity_step pairs.",
            "This repairs occasion individuation but does not establish whole-proof rigor or whole-metric isomorphism.",
        ],
        "sources": {"symbolic": {"path": str(args.symbolic), "sha256": sha(args.symbolic)},
                    "requests": {"path": str(args.requests), "sha256": sha(args.requests)},
                    "responses": {"path": str(args.responses), "sha256": sha(args.responses)}},
    }
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    report = args.output.with_name("report.md")
    report.write_text(
        "# Math a12 full-context role arm\n\n"
        f"Status: **{out['status']}** ({len(valid)}/443 valid).\n\n"
        f"Roles: `{dict(roles)}`.\n\n"
        f"The frozen symbolic verifier called {len(code_app)} pairs applicable; full context "
        f"reclassified {len(reclassified)} ({out['role_conditioned_individuation_gap']['reclassified_non_asserted_rate']:.1%}) "
        "as definitions, hypotheses, equations-to-solve, or other occasions.\n\n"
        f"Applicability kappa: `{app.kappa}`. Joint asserted-step polarity kappa: `{pol.kappa}`.\n\n"
        "These are unsupervised reconstruction statistics, not externally anchored validity.\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
