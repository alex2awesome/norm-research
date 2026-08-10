#!/usr/bin/env python3
"""Run the imported Patent graph on natural TRAIN and frozen construct controls."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Sequence

from methods.metric_seam.verifiers.construct_probe import PATENT_ANTECEDENT_PROPOSAL
from methods.metric_seam.verifiers.patent_antecedent import construct_controls, verify_antecedent_basis


def build_readout(rows: Sequence[dict[str, str]]) -> dict[str, object]:
    natural = []
    counts: Counter[str] = Counter()
    errors = 0
    for row in rows:
        try:
            verdict = verify_antecedent_basis(row["ctext"])
            counts[verdict.state] += 1
            natural.append({"item_key": row["item_key"], "state": verdict.state, "error": None})
        except Exception as exc:
            errors += 1
            natural.append({"item_key": row["item_key"], "state": None, "error": f"{type(exc).__name__}: {exc}"})
    challenges = []
    for control in construct_controls():
        verdict = verify_antecedent_basis(control.ctext)
        challenges.append({
            "control_id": control.control_id,
            "ctext": control.ctext,
            "expected_construct_state": control.expected_state,
            "proxy_triggered": control.proxy_triggered,
            "rationale": control.rationale,
            "code_state": verdict.state,
            "code_matches_expected_construct": verdict.state == control.expected_state,
        })
    return {
        "schema": "metric-seam.patent-antecedent-imported-draft-gate.v1",
        "status": "imported_retrospective_draft_executed",
        "proposal": PATENT_ANTECEDENT_PROPOSAL.to_json_value(),
        "draft_provenance": {
            "mode": "preexisting_manual_deep_code_imported_after_probe_pass",
            "automatic_discovery_claimed": False,
            "source": "methods/metric_seam/patent_claim_graph_additive_v1.py",
        },
        "natural_train": {
            "n": len(rows), "execution_errors": errors,
            "state_counts": {state: counts[state] for state in ("not_applicable", "satisfied", "violated")},
            "rows": natural,
        },
        "construct_challenge": {
            "status": "code_executed_blind_two_pass_construct_adjudication_pending",
            "code_correct": sum(row["code_matches_expected_construct"] for row in challenges),
            "n": len(challenges), "controls": challenges,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if "heldout" in str(args.train).casefold():
        raise ValueError("construct gate refuses held-out paths")
    source = json.loads(args.train.read_text(encoding="utf-8"))
    rows = source if isinstance(source, list) else source.get("train_items") or source.get("items")
    if not isinstance(rows, list):
        raise ValueError("TRAIN bundle has no rows")
    result = build_readout(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
