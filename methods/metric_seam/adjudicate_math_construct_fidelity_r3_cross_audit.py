"""Guarded cross-adjudication overlay for the frozen math R3 audit.

This module records a second, adversarial static reading of every retrieved
candidate in ``math_stackexchange_construct_fidelity_R3_v1.json``.  It is an
overlay, not a replacement: the source audit remains unchanged, and each
authored change is bound to the source row and selected program identity.

The review is outcome-blind.  It reads only the source audit, ``ops_math.py``,
and selected program source modules as text.  It does not import or execute a
candidate and does not read items, references, labels, outputs, correlations,
reconstruction results, or model/API data.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_AUDIT = Path(
    "outputs/metric_seam_pilot/hierarchy_r123/"
    "math_stackexchange_construct_fidelity_R3_v1.json"
)
DEFAULT_OUT = Path(
    "outputs/metric_seam_pilot/hierarchy_r123/"
    "math_stackexchange_construct_fidelity_R3_cross_adjudication_v1.json"
)

EXPECTED_SOURCE_AUDIT_SHA256 = (
    "53fda24960dcf7fc8874206ff53b2dfbdaec2bc70b52400e75d6215c8c0f91cb"
)
EXPECTED_OPS_MATH_SHA256 = (
    "4b4fcea356a9e602ae7f9bab858a14cd442bb1a6811db142180b2fb172e704ab"
)

ALLOWED_PATCH_FIELDS = {
    "verdict",
    "scope",
    "eligible_for_relation_local_execution",
    "audited_depth",
    "implemented_relations",
    "residual_construct",
    "polarity_aggregation_applicability_caveats",
    "justification",
}

A66_CLAMP_CAVEAT = (
    "A field-preserving clamp always passes a nonempty extracted mapping, even "
    "when both values are empty strings. That bypasses the program's `if not "
    "extracted` length-decay/connective fallback; only the downstream length, "
    "equation-wall, long-QED, and deflection modulation survives. The empty-map "
    "static projection and fixed-field conditional slices are distinct instruments."
)

A198_SENTINEL_CAVEAT = (
    "A nonempty constant cited_source clamp can interact with the input through "
    "the grounding guard: values such as YES or PRESENT are treated as a named "
    "source when that same word occurs in the answer. Empty and NONE avoid this "
    "collision, but the other sentinel slices are not semantically neutral."
)

A168_IMPLEMENTED_RELATIONS = [
    "isolate the Answer segment and compute LaTeX delimiter health",
    "parse each math span and penalize spans whose entire content is a bare logical or quantifier command, while exempting standard direction labels",
    "reward a fixed set of prose quantifier/connective idioms and short relational-chain shapes",
]

A168_DEPTH_CAVEAT = (
    "The document-level congruence/mod co-occurrence penalty is executable but is "
    "a notation-domain proxy, not a requested formula-to-prose grammar relation; "
    "it receives no matched-relation depth credit."
)

A168_EMPTY_FIELD_CAVEAT = (
    "With both fields absent, notation_flaw is interpreted as no detected flaw "
    "and contributes 0.2125 after weighting, while the unknown claim_quality path "
    "contributes a fixed 0.10. Those constants remain in the static projection."
)

A168_JUSTIFICATION = (
    "Bare logical-command spans and fixed prose/connective forms implement a narrow "
    "formula-integration subrelation. They use parsed-span structure (depth 1); the "
    "only document-wide code relation is the unrelated congruence/mod proxy, so it "
    "cannot raise matched-relation depth. Full grammar and semantic readability "
    "remain outside code."
)

A36_IMPLEMENTED_RELATIONS = [
    "over the full normalized Question-plus-Answer input, count numbered equations and relate repeated tag or parenthesized-number tokens as potential backreferences",
    "over that full input, count proof-skeleton cases, formal theorem/definition/proof/QED markers, and logical-connective density",
    "score sentence and average equation-token transparency and LaTeX delimiter health over the full input",
    "penalize very long full inputs with zero counted structural scaffolding",
]

A36_CHANNEL_CAVEAT = (
    "The program never isolates the Answer segment, so question-side tags, cases, "
    "formal markers, connectives, sentence shape, and delimiter defects all alter "
    "the scalar for an answer-scoped proof-structure construct."
)

A36_NONE_CAVEAT = (
    "The STAGE_LABELS prompt prescribes NONE when no labels exist, but "
    "_count_labels('NONE') returns one. Only an empty-string field clamp preserves "
    "zero added stage labels."
)

A180_IMPLEMENTED_RELATIONS = [
    "parse LaTeX delimiter health over the full normalized Question-plus-Answer input and normalize issue count by number of math spans",
    "build a notation census and compare bare versus command spellings for a fixed catalog of mathematical functions across the full input",
    "apply a mild penalty when average math-span token length over the full input exceeds sixty",
]

A180_CHANNEL_CAVEAT = (
    "The program does not isolate Answer:. Question-side notation and cross-question/"
    "answer spelling mixtures therefore contribute to what the source audit had "
    "described as consistency across the answer."
)

A210_CHANNEL_CAVEAT = (
    "The program does not isolate the Answer segment; question-side markup defects "
    "and question/answer Unicode-versus-command mixtures contribute to the score."
)

A210_NONE_CAVEAT = (
    "The notation_issue prompt prescribes NONE when no issue exists, but every "
    "nonempty value, including NONE, incurs the 0.08 issue penalty and suppresses "
    "the additional 0.04 clean bonus. Relative to an empty field, a compliant NONE "
    "therefore lowers an otherwise code-clean score by 0.12."
)


# Authored change specs. ``append_caveat(s)`` is resolved against the frozen
# row. Unchanged reviewed candidates are intentionally absent.
CHANGE_SPECS: dict[str, dict[str, Any]] = {
    "TB::math-stackexchange::general::R3::grandparent::10::2cdcd5d981d447ebf94c": {
        "kind": "field_clamp_control_flow_disclosure",
        "append_caveat": A66_CLAMP_CAVEAT,
        "reason": (
            "The source audit correctly describes score(text, {}, ops), but the "
            "field-preserving clamp grid cannot reach that mapping-truthiness branch."
        ),
    },
    "TB::math-stackexchange::general::R3::grandparent::4::573bc6a1b44aee4f6236": {
        "kind": "field_clamp_grounding_collision",
        "append_caveat": A198_SENTINEL_CAVEAT,
        "reason": (
            "A constant nonempty field value is conditionally accepted by a lexical "
            "input-grounding test, creating an input-dependent sentinel collision."
        ),
    },
    "TB::math-stackexchange::general::R3::merged_group::4::7a187ce24529c9c48ce1": {
        "kind": "field_clamp_grounding_collision",
        "append_caveat": A198_SENTINEL_CAVEAT,
        "reason": (
            "A constant nonempty field value is conditionally accepted by a lexical "
            "input-grounding test, creating an input-dependent sentinel collision."
        ),
    },
    "TB::math-stackexchange::general::R3::merged_group::16::bedf9a334646bda9ae1e": {
        "kind": "matched_relation_depth_and_empty_field_correction",
        "after": {
            "audited_depth": 1,
            "implemented_relations": A168_IMPLEMENTED_RELATIONS,
            "justification": A168_JUSTIFICATION,
        },
        "append_caveats": [A168_DEPTH_CAVEAT, A168_EMPTY_FIELD_CAVEAT],
        "reason": (
            "The only depth-2 branch is a notation-domain proxy outside the requested "
            "formula/prose grammar relation; the matched surviving operations are "
            "parsed-span and within-unit relations at depth 1."
        ),
    },
    "TB::math-stackexchange::general::R3::merged_group::14::da47b04ffaa9bf294ae9": {
        "kind": "answer_channel_and_field_interface_disclosure",
        "after": {"implemented_relations": A36_IMPLEMENTED_RELATIONS},
        "append_caveats": [A36_CHANNEL_CAVEAT, A36_NONE_CAVEAT],
        "reason": (
            "Static tracing shows every code signal is computed over the unsplit input, "
            "and the prompt's negative sentinel is parsed as one positive label."
        ),
    },
    "TB::math-stackexchange::general::R3::merged_group::3::e7911c7b707a53bacba4": {
        "kind": "answer_channel_scope_correction",
        "after": {"implemented_relations": A180_IMPLEMENTED_RELATIONS},
        "append_caveat": A180_CHANNEL_CAVEAT,
        "reason": (
            "The implementation performs no Answer split, so its census and hygiene "
            "relations are full-input rather than answer-only."
        ),
    },
    "TB::math-stackexchange::general::R3::grandparent::16::570eed33fe5f1ce2a120": {
        "kind": "answer_channel_and_field_interface_disclosure",
        "append_caveats": [A210_CHANNEL_CAVEAT, A210_NONE_CAVEAT],
        "reason": (
            "The implementation is full-input and treats its prompt-prescribed negative "
            "response as a positive issue signal."
        ),
    },
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(payload)


def _load_source(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = _sha256_bytes(raw)
    if digest != EXPECTED_SOURCE_AUDIT_SHA256:
        raise ValueError(
            f"source audit changed: expected {EXPECTED_SOURCE_AUDIT_SHA256}, got {digest}"
        )
    return json.loads(raw), digest


def _candidate_guard(row: dict[str, Any]) -> dict[str, str]:
    candidate = row["candidate"]
    return {
        "aspect_id": candidate["aspect_id"],
        "source_path": candidate["source_path"],
        "program_sha256": candidate["program_sha256"],
    }


def _review_set_digest(rows: list[dict[str, Any]]) -> str:
    identities = [
        {"cell_id": row["cell_id"], **_candidate_guard(row)}
        for row in rows
        if row.get("candidate") is not None
    ]
    identities.sort(key=lambda value: value["cell_id"])
    return _sha256_json(identities)


def _resolve_after(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    after = copy.deepcopy(spec.get("after", {}))
    appended = []
    if "append_caveat" in spec:
        appended.append(spec["append_caveat"])
    appended.extend(spec.get("append_caveats", []))
    if appended:
        if "polarity_aggregation_applicability_caveats" in after:
            raise ValueError("cannot both replace and append caveats")
        caveats = list(row["polarity_aggregation_applicability_caveats"])
        for caveat in appended:
            if caveat not in caveats:
                caveats.append(caveat)
        after["polarity_aggregation_applicability_caveats"] = caveats
    if not after:
        raise ValueError(f"empty change for {row['cell_id']}")
    return after


def _apply_changes(
    rows: list[dict[str, Any]], changes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {change["cell_id"]: change for change in changes}
    patched = []
    for source_row in rows:
        row = copy.deepcopy(source_row)
        change = by_id.get(row["cell_id"])
        if change:
            row.update(copy.deepcopy(change["after"]))
        patched.append(row)
    return patched


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    retrieved = [row for row in rows if row.get("candidate") is not None]
    eligible = [row for row in retrieved if row["eligible_for_relation_local_execution"]]
    return {
        "retrieved_verdicts": dict(
            sorted(Counter(row["verdict"] for row in retrieved).items())
        ),
        "retrieved_depths": dict(
            sorted(Counter(str(row["audited_depth"]) for row in retrieved).items())
        ),
        "eligible_depths": dict(
            sorted(Counter(str(row["audited_depth"]) for row in eligible).items())
        ),
        "eligible_for_relation_local_execution": len(eligible),
    }


def build(source_path: Path = DEFAULT_SOURCE_AUDIT) -> dict[str, Any]:
    source_path = REPO_ROOT / source_path if not source_path.is_absolute() else source_path
    source, source_sha = _load_source(source_path)
    rows = source["rows"]
    by_id = {row["cell_id"]: row for row in rows}
    retrieved = [row for row in rows if row.get("candidate") is not None]

    changes = []
    for cell_id, spec in CHANGE_SPECS.items():
        row = by_id[cell_id]
        after = _resolve_after(row, spec)
        before = {field: copy.deepcopy(row[field]) for field in after}
        changes.append(
            {
                "cell_id": cell_id,
                "candidate_guard": _candidate_guard(row),
                "change_kind": spec["kind"],
                "before": before,
                "after": after,
                "reason": spec["reason"],
            }
        )
    changes.sort(key=lambda value: value["cell_id"])
    patched_rows = _apply_changes(rows, changes)

    artifact = {
        "schema": "metric-seam.math-static-construct-fidelity-cross-adjudication.v1",
        "status": "complete_guarded_static_cross_audit",
        "task": "math-stackexchange",
        "levels": ["R3"],
        "design_scope": "outcome_blind_static_code_only_cross_adjudication",
        "source_audit": str(DEFAULT_SOURCE_AUDIT),
        "source_audit_sha256": source_sha,
        "source_audit_schema": source["schema"],
        "ops_math_source": source["ops_math_source"],
        "ops_math_sha256": source["ops_math_sha256"],
        "review_policy": (
            "Credit only a decision-contributing requested code relation. Presence "
            "is not function or quality. Depth is the deepest matched code relation, "
            "not the deepest unrelated program branch. Trace empty fields, constant "
            "clamps, channel selection, and interface defaults explicitly."
        ),
        "forbidden_inputs": source["forbidden_inputs"],
        "forbidden_inputs_used": False,
        "candidate_execution_performed": False,
        "candidate_import_performed": False,
        "items_loaded": False,
        "model_or_api_calls_performed": False,
        "accelerators_used": False,
        "review_coverage": {
            "source_rows": len(rows),
            "retrieved_candidates_reviewed": len(retrieved),
            "retrieved_candidate_set_sha256": _review_set_digest(rows),
            "changed_rows": len(changes),
            "unchanged_retrieved_rows": len(retrieved) - len(changes),
            "all_retrieved_candidates_reviewed": True,
        },
        "before_counts": _counts(rows),
        "after_counts_if_overlay_applied": _counts(patched_rows),
        "change_kind_counts": dict(
            sorted(Counter(change["change_kind"] for change in changes).items())
        ),
        "instrument_disclosures": [
            (
                "score(text, {}, ops) and score(text, {all_fields: ''}, ops) are "
                "not equivalent for every historical hybrid; the latter is a "
                "field-preserving conditional slice, not a pure-code projection."
            ),
            (
                "a36, a180, and a210 do not isolate the Answer channel; their narrow "
                "witnesses are contaminated by question-side structure."
            ),
            (
                "a36 and a210 mishandle their prompt-prescribed NONE response, and "
                "a198 can ground a constant nonempty sentinel against input text."
            ),
            (
                "These disclosures describe static control flow only. No real program "
                "or conditional slice was executed in this cross-audit."
            ),
        ],
        "changes": changes,
        "interpretation": (
            "The overlay leaves every R3 verdict unchanged, corrects one matched depth, "
            "and narrows instrument claims where channel or clamp semantics differ. "
            "Partial remains relation-local. Mismatch and non-discovery do not establish "
            "tacitness, inarticulability, or universal non-verifiability."
        ),
    }
    validate(artifact, source)
    return artifact


def validate(artifact: dict[str, Any], source: dict[str, Any]) -> None:
    if artifact["schema"] != "metric-seam.math-static-construct-fidelity-cross-adjudication.v1":
        raise ValueError("unexpected overlay schema")
    if source["schema"] != "metric-seam.math-static-construct-fidelity.v1":
        raise ValueError("unexpected source-audit schema")
    if source.get("execution_performed") is not False:
        raise ValueError("source audit is not static")
    for field in (
        "forbidden_inputs_used",
        "candidate_execution_performed",
        "candidate_import_performed",
        "items_loaded",
        "model_or_api_calls_performed",
        "accelerators_used",
    ):
        if artifact.get(field) is not False:
            raise ValueError(f"cross-audit violated {field}")

    rows = source["rows"]
    if len(rows) != 30 or len({row["cell_id"] for row in rows}) != 30:
        raise ValueError("source audit must contain 30 unique R3 rows")
    retrieved = [row for row in rows if row.get("candidate") is not None]
    if len(retrieved) != 19:
        raise ValueError("source audit must contain 19 retrieved candidates")
    by_id = {row["cell_id"]: row for row in rows}

    coverage = artifact["review_coverage"]
    if coverage["source_rows"] != 30:
        raise ValueError("incorrect source-row coverage")
    if coverage["retrieved_candidates_reviewed"] != 19:
        raise ValueError("incorrect retrieved-candidate coverage")
    if coverage["retrieved_candidate_set_sha256"] != _review_set_digest(rows):
        raise ValueError("retrieved-candidate set guard mismatch")
    if coverage["all_retrieved_candidates_reviewed"] is not True:
        raise ValueError("cross-audit is not marked complete")

    changes = artifact["changes"]
    if len(changes) != len(CHANGE_SPECS):
        raise ValueError("wrong number of changes")
    if len({change["cell_id"] for change in changes}) != len(changes):
        raise ValueError("duplicate change cell")
    if {change["cell_id"] for change in changes} != set(CHANGE_SPECS):
        raise ValueError("change set differs from authored specs")
    if coverage["changed_rows"] != len(changes):
        raise ValueError("incorrect changed-row count")
    if coverage["unchanged_retrieved_rows"] != 19 - len(changes):
        raise ValueError("incorrect unchanged-row count")

    for change in changes:
        cell_id = change["cell_id"]
        row = by_id.get(cell_id)
        if row is None or row.get("candidate") is None:
            raise ValueError(f"change is not bound to a retrieved row: {cell_id}")
        if change["candidate_guard"] != _candidate_guard(row):
            raise ValueError(f"candidate guard mismatch: {cell_id}")
        fields = set(change["after"])
        if not fields or not fields <= ALLOWED_PATCH_FIELDS:
            raise ValueError(f"invalid patch fields: {cell_id}")
        if set(change["before"]) != fields:
            raise ValueError(f"before/after field mismatch: {cell_id}")
        if change["before"] == change["after"]:
            raise ValueError(f"no-op patch: {cell_id}")
        for field in fields:
            if change["before"][field] != row[field]:
                raise ValueError(f"stale before guard for {cell_id}:{field}")
        if not isinstance(change.get("reason"), str) or not change["reason"].strip():
            raise ValueError(f"missing reason: {cell_id}")

        source_file = REPO_ROOT / change["candidate_guard"]["source_path"]
        actual_program_sha = _sha256_bytes(source_file.read_bytes())
        if actual_program_sha != change["candidate_guard"]["program_sha256"]:
            raise ValueError(f"selected program changed: {cell_id}")

    ops_path = REPO_ROOT / artifact["ops_math_source"]
    actual_ops_sha = _sha256_bytes(ops_path.read_bytes())
    if actual_ops_sha != EXPECTED_OPS_MATH_SHA256:
        raise ValueError("ops_math changed")
    if artifact["ops_math_sha256"] != actual_ops_sha:
        raise ValueError("ops_math artifact guard mismatch")

    patched = _apply_changes(rows, changes)
    for row in patched:
        verdict = row["verdict"]
        if verdict in {"exact", "partial"}:
            if row["scope"] == "none" or not row["eligible_for_relation_local_execution"]:
                raise ValueError(f"eligible verdict has inconsistent scope: {row['cell_id']}")
        elif verdict in {"mismatch", "no_candidate_bounded_non_discovery"}:
            if row["scope"] != "none" or row["eligible_for_relation_local_execution"]:
                raise ValueError(f"ineligible verdict has inconsistent scope: {row['cell_id']}")
        else:
            raise ValueError(f"unknown verdict: {verdict}")
        depth = row["audited_depth"]
        if depth is not None and (
            isinstance(depth, bool) or not isinstance(depth, int) or not 0 <= depth <= 4
        ):
            raise ValueError(f"invalid depth: {row['cell_id']}")

    if artifact["before_counts"] != _counts(rows):
        raise ValueError("before counts do not reproduce")
    if artifact["after_counts_if_overlay_applied"] != _counts(patched):
        raise ValueError("after counts do not reproduce")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", type=Path, default=DEFAULT_SOURCE_AUDIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    artifact = build(args.source_audit)
    out = REPO_ROOT / args.out if not args.out.is_absolute() else args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "out": str(out),
                "reviewed": artifact["review_coverage"]["retrieved_candidates_reviewed"],
                "changes": artifact["review_coverage"]["changed_rows"],
                "after": artifact["after_counts_if_overlay_applied"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
