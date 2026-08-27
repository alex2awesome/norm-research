#!/usr/bin/env python3
"""Replay audited relation counterexamples against frozen v1 and corrected v2 ops.

This is a code-verifiability readout, not an LLM-reference reconstruction test.  The
expected outcomes are executable/metamorphic invariants authored during the independent
audit, so provenance is ``manual`` and the comparison is ``replay`` rather than automatic
capability discovery.

Usage: python -m methods.metric_seam.hybrids.eval_ops_capability_v2 --check
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

METRIC_SEAM = Path(__file__).resolve().parent.parent

if __package__:
    from . import ops_capability as v1
    # The frozen v2.1 source intentionally retains its historical bare v1 import.
    # Bind that exact dependency name during package import without modifying the
    # SHA-pinned scientific source.
    prior_v1_alias = sys.modules.get("ops_capability")
    sys.modules["ops_capability"] = v1
    try:
        from . import ops_capability_v2 as v2
    finally:
        if prior_v1_alias is None:
            sys.modules.pop("ops_capability", None)
        else:
            sys.modules["ops_capability"] = prior_v1_alias
    from ..environment_v2 import environment_fingerprint
else:  # Preserve the historical direct-script entrypoint.
    HYBRIDS = Path(__file__).resolve().parent
    for path in (HYBRIDS, METRIC_SEAM):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import ops_capability as v1  # type: ignore[no-redef]  # noqa: E402
    import ops_capability_v2 as v2  # type: ignore[no-redef]  # noqa: E402
    from environment_v2 import environment_fingerprint  # type: ignore[no-redef]  # noqa: E402


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _invalid_date_status(op):
    rows = op.date_chain("The notice arrived April 31, 1982.")
    return rows[0].get("parse_status", "UNMARKED") if rows else "DROPPED"


def _has_attribution_mode(op, text, mode):
    return any(r.get("attribution_mode") == mode for r in op.attributions(text))


def _action_beat_speakers(op, text):
    return [
        r.get("speaker_span")
        for r in op.attributions(text)
        if r.get("attribution_mode") == "adjacent_named_action_beat"
    ]


def _refrain_class(op, text):
    rows = op.is_refrain(text)
    return rows[0].get("is_refrain") if rows else "NO_ROW"


def _ownership_values(op, text):
    return [r.get("speaker_is_first_person_org") for r in op.attributions(text)]


def cases():
    """The complete named v2.1 defect ledger, not a favorable-case sample."""

    return [
        {
            "id": "date.missing_year_frozen_epoch",
            "relation": "date_parse",
            "expected": "2000-04-02",
            "run": lambda op: op.date_chain("The notice arrived April 2.")[0]["date"],
        },
        {
            "id": "date.invalid_calendar_surface_retained",
            "relation": "date_parse",
            "historical_defect": "April-31 date silently dropped",
            "expected": "INVALID",
            "run": _invalid_date_status,
        },
        {
            "id": "date.negative_gap_rejected",
            "relation": "temporal_order",
            "expected": False,
            "run": lambda op: op.deadline_satisfied("2020-02-01", "2020-01-01", 90),
        },
        {
            "id": "number.direction_preserved",
            "relation": "signed_percent_change",
            "expected": False,
            "run": lambda op: op.number_consistency(
                "It decreased from 100 to 50, a 50% increase."
            )[0]["consistent"],
        },
        {
            "id": "statistics.lower_bound_abstains",
            "relation": "p_value_bound",
            "expected": None,
            "run": lambda op: op.stat_consistency("The estimate was z = 2.58, p > .001.")[0][
                "decision_inconsistent"
            ],
        },
        {
            "id": "statistics.upper_bound_abstains",
            "relation": "p_value_bound",
            "expected": None,
            "run": lambda op: op.stat_consistency("The estimate was z = 2.58, p < .10.")[0][
                "decision_inconsistent"
            ],
        },
        {
            "id": "attribution.repeated_span_abstains",
            "relation": "span_identity",
            "expected": None,
            "run": lambda op: op.self_attributed(
                "Repeated claim. Repeated claim.", "Repeated claim."
            ),
        },
        {
            "id": "attribution.out_of_cap_span_abstains",
            "relation": "span_identity",
            "expected": None,
            "run": lambda op: op.self_attributed(
                "x" * 8100 + " unique tail", "unique tail"
            ),
        },
        {
            "id": "attribution.multi_org_ownership_abstains",
            "relation": "document_voice_ownership",
            "expected": [None, None, None],
            "run": lambda op: _ownership_values(
                op,
                "Apple announced that sales grew. Google said that demand would grow. "
                "Microsoft noted that prices fell.",
            ),
        },
        {
            "id": "attribution.missing_issuer_abstains",
            "relation": "document_voice_ownership",
            "expected": [None],
            "run": lambda op: _ownership_values(
                op, '\"We are proud,\" said Maria Chen at Acme Biotech.'
            ),
        },
        {
            "id": "attribution.conjunct_inherits_shared_subject",
            "relation": "predicate_subject_binding",
            "historical_defect": "conjunct reporting verb loses shared subject",
            "expected": True,
            "run": lambda op: _has_attribution_mode(
                op,
                "Wren starved the city, let drought do its work, and told the council "
                "exactly what she'd done.",
                "reporting_verb_shared_subject",
            ),
        },
        {
            "id": "attribution.named_action_beat_anchors_quote",
            "relation": "fiction_action_beat_speaker_binding",
            "historical_defect": "action-beat-then-quote attribution blindness",
            "expected": ["Bren the smith", "Old Yara", "Young Cass"],
            "run": lambda op: _action_beat_speakers(
                op,
                "'The bridge toll doubles this spring,' the reeve announced. "
                "'Speak now.'\n\n"
                "Bren the smith stepped forward first. 'Double it and I lose every "
                "customer on the far bank. Cap it, don't raise it.'\n\n"
                "Old Yara leaned on her cane. 'I don't care about the toll. I want the "
                "ferry running again — my knees can't take the bridge steps anymore.'\n\n"
                "Young Cass didn't wait to be called on. 'None of that matters if the "
                "bridge collapses first. Fund the repair, then argue about the toll.'",
            ),
        },
        {
            "id": "refrain.one_word_callback_eligible",
            "relation": "short_refrain_detection",
            "historical_defect": "three-word clustering floor",
            "expected": True,
            "run": lambda op: _refrain_class(
                op, "Never. The river climbed and swallowed the road. Never."
            ),
        },
        {
            "id": "refrain.historical_two_word_callback_eligible",
            "relation": "short_refrain_detection",
            "historical_defect": "three-word clustering floor (a270 exact probe surface)",
            "scope": "local_relation_only_not_document_level_a270_score",
            "expected": True,
            "run": lambda op: _refrain_class(
                op,
                "The outer wall fell. They fought. The docks burned. They fought. "
                "By the time only the harbor was left, they were still fighting.",
            ),
        },
        {
            "id": "refrain.adjacent_variation_not_craft",
            "relation": "refrain_progression",
            "expected": False,
            "run": lambda op: _refrain_class(
                op, "We will win today. We really will win today. A final sentence."
            ),
        },
        {
            "id": "position.invalid_offset_abstains",
            "relation": "discourse_position",
            "expected": None,
            "run": lambda op: op.discourse_position("First. Middle. Last.", (999, 1000)),
        },
        {
            "id": "position.ambiguous_string_abstains",
            "relation": "discourse_position",
            "expected": None,
            "run": lambda op: op.discourse_position(
                "Same line. Middle line. Same line.", "Same line."
            ),
        },
    ]


def evaluate():
    rows = []
    for spec in cases():
        row = {k: v for k, v in spec.items() if k != "run"}
        for name, module in (("frozen_v1", v1), ("corrected_v2", v2)):
            try:
                actual = spec["run"](module)
                row[name] = {"actual": actual, "pass": actual == spec["expected"]}
            except Exception as exc:
                row[name] = {
                    "actual": None,
                    "pass": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        rows.append(row)
    return {
        "schema": "metric-seam.capability-counterexample-replay.v2.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "code_verifiability",
        "reference_isomorphism_evaluated": False,
        "articulability_evaluated": False,
        "discovery_mode": "replay",
        "expected_outcome_provenance": "manual_independent_audit",
        "coverage": {
            "policy": "complete_named_defect_ledger",
            "note": (
                "Includes every behavior claimed by the additive capability wrapper, "
                "including all four historical defect families and prior conservative "
                "abstention/safeguard changes."
            ),
            "historical_defect_families": [
                "date parsing (missing year and invalid calendar surface)",
                "attribution (conjunct/shared subject and named action beat)",
                "short refrain eligibility",
                "deadline arithmetic",
            ],
        },
        "versions": {"frozen": v1.VERSION, "corrected": v2.VERSION},
        "implementation": {
            "runner_sha256": _sha256(__file__),
            "frozen_v1_sha256": _sha256(v1.__file__),
            "corrected_v2_sha256": _sha256(v2.__file__),
            "environment": environment_fingerprint(),
        },
        "summary": {
            "n": len(rows),
            "frozen_v1_pass": sum(r["frozen_v1"]["pass"] for r in rows),
            "corrected_v2_pass": sum(r["corrected_v2"]["pass"] for r in rows),
        },
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs/metric_seam_pilot/reconstruction_v2/"
            "capability_counterexamples_v2_1.json"
        ),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = evaluate()
    print(json.dumps(result["summary"], indent=2))
    if not args.check:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    return 0 if result["summary"]["corrected_v2_pass"] == result["summary"]["n"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
