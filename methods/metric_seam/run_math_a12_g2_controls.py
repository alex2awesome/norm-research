#!/usr/bin/env python3
"""Run canonical G2 proxy traps against the frozen Math-a12 symbolic verifier."""
from __future__ import annotations

import json
from pathlib import Path

from methods.metric_seam.killswitch.plants import validity_controls
from methods.metric_seam.verifiers.math_a12_symbolic import verify_document_pairs

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/metric_seam_pilot/hierarchy_r123/results/math_a12_g2_validity_v1"
UNIT = "math.a12.rigor.contextual_equation_use"


def states(text: str, key: str) -> list[str]:
    return [verdict.state for _, verdict in verify_document_pairs(text, item_key=key)]


def main() -> int:
    rows = []
    for index, control in enumerate(validity_controls(UNIT), 1):
        pos = states(control["text_pos"], f"g2_{index:02d}_pos")
        neg = states(control["text_neg"], f"g2_{index:02d}_neg")
        pos_violation, neg_violation = "violated" in pos, "violated" in neg
        if control["polarity"] == "positive":
            passed = neg_violation and not pos_violation
            outcome = "SEPARATES_TRUE_VIOLATION" if passed else "MISSES_TRUE_VIOLATION"
        else:
            fired = neg_violation and not pos_violation
            passed = not fired
            outcome = "PROXY_FIRE" if fired else "NO_PROXY_FIRE"
        rows.append({**control, "text_pos_states": pos, "text_neg_states": neg,
                     "outcome": outcome, "passed": passed})
    pminus = [r for r in rows if r["polarity"] == "negative_proxy_trap"]
    pplus = [r for r in rows if r["polarity"] == "positive"]
    result = {
        "schema": "metric-seam.g2-construct-validity-controls.v1",
        "unit_id": UNIT, "verifier": "frozen Math-a12 symbolic identity verifier",
        "negative_proxy_traps": {"passed": sum(r["passed"] for r in pminus), "n": len(pminus)},
        "positive_true_violations": {"passed": sum(r["passed"] for r in pplus), "n": len(pplus)},
        "g2_pass": all(r["passed"] for r in rows),
        "rows": rows,
        "reading": (
            "Failure means symbolic nonidentity is not a valid proxy for rigor across occasions; "
            "it does not negate the SymPy capability on asserted identity steps."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "readout.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (OUT / "report.md").write_text(
        "# Math a12 G2 construct-validity controls\n\n"
        f"G2 verdict: **{'PASS' if result['g2_pass'] else 'FAIL'}**. "
        f"Negative proxy traps passed {result['negative_proxy_traps']['passed']}/{len(pminus)}; "
        f"true-violation controls passed {result['positive_true_violations']['passed']}/{len(pplus)}.\n\n"
        "This retracts the old symbolic-identity agreement as a rigor-validity result while retaining "
        "the bounded algebra capability for contextually asserted identity steps.\n")
    print(OUT / "readout.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
