"""Guarded cross-adjudication overlay for the frozen math R1/R2 audit.

This module records a second, adversarial static reading of every retrieved
candidate in ``math_stackexchange_construct_fidelity_R1_R2_v1.json``.  It is
an overlay, not a replacement: the source audit remains immutable, and every
change is guarded by the source-artifact hash plus the selected program's
identity and byte hash.

The review is deliberately outcome-blind.  It reads the source audit,
``ops_math.py``, and the selected program source modules as text.  It does not
import or execute a candidate and does not read items, references, outcomes,
splits, outputs, correlations, reconstruction results, or model/API data.
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
    "math_stackexchange_construct_fidelity_R1_R2_v1.json"
)
DEFAULT_OUT = Path(
    "outputs/metric_seam_pilot/hierarchy_r123/"
    "math_stackexchange_construct_fidelity_R1_R2_cross_adjudication_v1.json"
)

# This is intentionally a byte-level freeze.  A revised source audit requires
# a new cross-audit, not an implicit rebase of these authored judgments.
EXPECTED_SOURCE_AUDIT_SHA256 = (
    "6d1eb12b9c150aa9b6da92a72588b261d8136eefce4d49a9ba57a9a143e89052"
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

A42_IMPLEMENTED_RELATIONS = [
    "Assign a fixed 0.05 base to normalized texts of at least 30 characters when the excluded example and visual gates are empty.",
    "Penalize four or more high-precision decimal tokens by 0.15.",
    "Subtract a small penalty when delimiter_health reports at least two issue categories.",
]

A42_CAVEATS = [
    "The marker, case, and numeric-density elaboration routine is not decision-contributing in code-only mode: it is called only after an excluded LLM example or visual gate is positive.",
    "The code-only projection is nearly constant at 0.05 before negative decimal and delimiter penalties; it cannot positively detect an example, figure, diagram, table, caption, or visual role.",
]

A42_JUSTIFICATION = (
    "In the code-only projection, the positive elaboration branch is unreachable. "
    "The remaining fixed base plus negative decimal and LaTeX penalties implement no "
    "requested visual, figure, caption, or multimedia relation. The mismatch verdict "
    "therefore remains correct, but the earlier operation description over-credited "
    "gated code."
)

A12_CAVEAT = (
    "With the excluded fields empty, unjustified_step defaults to no detected gap "
    "(justification_score=1.0), while concrete_result defaults to non-substantive and "
    "caps every otherwise non-thin code-only score at 0.18; thin math-free answers "
    "are capped at 0.15."
)

A198_CAVEAT = (
    "MathOps.equation_stats returns a tuple, but this program reads n_numbered only "
    "when the return value is a dict, so that library-derived numbering path is "
    "silently zero; the program's direct regex token detections still survive."
)

A174_CAVEAT = (
    "With the excluded derivation_steps field empty, llm_none is true and the code "
    "unconditionally forgives one detected bad layout unit; code-only bad-unit burden "
    "is therefore biased downward by one."
)


# Authored change specs.  ``append_caveat`` is resolved against the frozen row
# by build(); all other entries are literal replacements.  No unchanged row is
# represented here.
CHANGE_SPECS: dict[str, dict[str, Any]] = {
    # a42: all five source rows over-credit _elaboration_score even though its
    # positive branch is unreachable when LLM fields are excluded.
    "TB::math-stackexchange::general::R1::parented_tree::357::27284082b353b1e934c9": {
        "kind": "decision_path_correction",
        "after": {
            "implemented_relations": A42_IMPLEMENTED_RELATIONS,
            "justification": A42_JUSTIFICATION,
        },
        "append_caveats": A42_CAVEATS,
        "reason": "The source audit described a gated elaboration routine as code-only behavior; static control flow shows only a fixed base and two negative penalties survive.",
    },
    "TB::math-stackexchange::general::R1::merged_tree::238::fc4cf188a1bf9f46c92e": {
        "kind": "decision_path_correction",
        "after": {
            "implemented_relations": A42_IMPLEMENTED_RELATIONS,
            "justification": A42_JUSTIFICATION,
        },
        "append_caveats": A42_CAVEATS,
        "reason": "The source audit described a gated elaboration routine as code-only behavior; static control flow shows only a fixed base and two negative penalties survive.",
    },
    "TB::math-stackexchange::general::R2::merged_group::9::080f2cb25e085f5fe3ea": {
        "kind": "decision_path_correction",
        "after": {
            "implemented_relations": A42_IMPLEMENTED_RELATIONS,
            "justification": A42_JUSTIFICATION,
        },
        "append_caveats": A42_CAVEATS,
        "reason": "The source audit described a gated elaboration routine as code-only behavior; static control flow shows only a fixed base and two negative penalties survive.",
    },
    "TB::math-stackexchange::general::R2::grandparent::7::311f48a4043dc1ab3d3e": {
        "kind": "decision_path_correction",
        "after": {
            "implemented_relations": A42_IMPLEMENTED_RELATIONS,
            "justification": A42_JUSTIFICATION,
        },
        "append_caveats": A42_CAVEATS,
        "reason": "The source audit described a gated elaboration routine as code-only behavior; static control flow shows only a fixed base and two negative penalties survive.",
    },
    "TB::math-stackexchange::general::R2::grandparent::26::7025d317ae10b00048de": {
        "kind": "decision_path_correction",
        "after": {
            "implemented_relations": A42_IMPLEMENTED_RELATIONS,
            "justification": A42_JUSTIFICATION,
        },
        "append_caveats": A42_CAVEATS,
        "reason": "The source audit described a gated elaboration routine as code-only behavior; static control flow shows only a fixed base and two negative penalties survive.",
    },

    # a12: four rows omit the code-only aggregation clamp induced by the
    # excluded concrete_result field.  Verdicts remain relation-local/mismatch.
    "TB::math-stackexchange::general::R1::parented_tree::81::79313ce4ce283e86ebe7": {
        "kind": "aggregation_caveat",
        "append_caveat": A12_CAVEAT,
        "reason": "The empty-field semantics force a universal 0.18 code-only ceiling (0.15 for thin math-free answers), which materially narrows what its surviving structure signals can express.",
    },
    "TB::math-stackexchange::general::R1::merged_tree::99::27a8afde78689275c8e2": {
        "kind": "aggregation_caveat",
        "append_caveat": A12_CAVEAT,
        "reason": "The empty-field semantics force a universal 0.18 code-only ceiling (0.15 for thin math-free answers), which materially narrows what its surviving structure signals can express.",
    },
    "TB::math-stackexchange::general::R1::parented_tree::36::1fd33b611482f069f2db": {
        "kind": "aggregation_caveat",
        "append_caveat": A12_CAVEAT,
        "reason": "The empty-field semantics force a universal 0.18 code-only ceiling (0.15 for thin math-free answers), which materially narrows what its surviving structure signals can express.",
    },
    "TB::math-stackexchange::general::R2::merged_group::56::fde63c9dd735984587fe": {
        "kind": "aggregation_caveat",
        "append_caveat": A12_CAVEAT,
        "reason": "The empty-field semantics force a universal 0.18 code-only ceiling (0.15 for thin math-free answers), which materially narrows what its surviving structure signals can express.",
    },

    # a198: direct token regexes survive, but the MathOps numbering branch is
    # dead because of a tuple/dict interface mismatch.
    "TB::math-stackexchange::general::R1::parented_tree::207::3d526700c07efc57e086": {
        "kind": "capability_interface_caveat",
        "append_caveat": A198_CAVEAT,
        "reason": "Static interface tracing shows equation_stats returns a tuple while a198_h1 accepts only a dict for n_numbered; direct tag/ref regexes keep the verdict partial.",
    },
    "TB::math-stackexchange::general::R2::grandparent::36::b25ef38532b6b0948927": {
        "kind": "capability_interface_caveat",
        "append_caveat": A198_CAVEAT,
        "reason": "Static interface tracing shows equation_stats returns a tuple while a198_h1 accepts only a dict for n_numbered; direct tag/ref regexes keep the verdict partial.",
    },

    # a174: excluding the semantic field is not neutral; its empty value
    # actively subtracts one detected bad unit.
    "TB::math-stackexchange::general::R2::merged_group::10::a8db4920911b2f373015": {
        "kind": "aggregation_caveat",
        "append_caveat": A174_CAVEAT,
        "reason": "The code-only projection interprets the absent LLM field as ZERO/no-calculation and removes one bad layout unit; direct layout relations still justify partial/depth 2.",
    },

    # a168: a document-wide relation between congruence notation and modular
    # language is a depth-2 co-occurrence relation under the frozen vocabulary.
    "TB::math-stackexchange::general::R1::merged_tree::87::01c3ffc70f1d857bb128": {
        "kind": "depth_correction",
        "after": {"audited_depth": 2},
        "reason": "The cong/mod penalty relates two document occurrences; it is not only a parsed-span or within-unit tally and therefore reaches depth 2.",
    },

    # a126 plausible-quality: the program identifies analogy/exploration form
    # but never tests the requested warrant, credibility, or constraint.
    "TB::math-stackexchange::general::R1::parented_tree::151::a3d74e378f643f073486": {
        "kind": "presence_function_verdict_correction",
        "after": {
            "verdict": "mismatch",
            "scope": "none",
            "eligible_for_relation_local_execution": False,
            "justification": (
                "The operative construct is epistemic quality: whether an analogy or "
                "induction is warranted, credible, and meaningfully constrained. The "
                "program detects analogy- or exploration-shaped text but evaluates none "
                "of those functional quality relations; presence is only an applicability cue."
            ),
        },
        "reason": "This is a presence-versus-function collision: discovery-form cues cannot decide whether non-deductive reasoning is warranted, credible, or constrained.",
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
            f"source audit hash changed: expected {EXPECTED_SOURCE_AUDIT_SHA256}, got {digest}"
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
    identities.sort(key=lambda x: x["cell_id"])
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
    patched: list[dict[str, Any]] = []
    for source_row in rows:
        row = copy.deepcopy(source_row)
        change = by_id.get(row["cell_id"])
        if change:
            row.update(copy.deepcopy(change["after"]))
        patched.append(row)
    return patched


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    retrieved = [row for row in rows if row.get("candidate") is not None]
    return {
        "retrieved_verdicts": dict(sorted(Counter(r["verdict"] for r in retrieved).items())),
        "retrieved_depths": dict(
            sorted(Counter(str(r["audited_depth"]) for r in retrieved).items())
        ),
        "eligible_for_relation_local_execution": sum(
            bool(r["eligible_for_relation_local_execution"]) for r in retrieved
        ),
    }


def build(source_path: Path = DEFAULT_SOURCE_AUDIT) -> dict[str, Any]:
    source_path = REPO_ROOT / source_path if not source_path.is_absolute() else source_path
    source, source_sha = _load_source(source_path)
    rows = source["rows"]
    by_id = {row["cell_id"]: row for row in rows}
    retrieved = [row for row in rows if row.get("candidate") is not None]

    changes: list[dict[str, Any]] = []
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

    changes.sort(key=lambda x: x["cell_id"])
    patched_rows = _apply_changes(rows, changes)

    artifact = {
        "schema": "metric-seam.math-static-construct-fidelity-cross-adjudication.v1",
        "status": "complete_guarded_static_cross_audit",
        "task": "math-stackexchange",
        "levels": ["R1", "R2"],
        "design_scope": "outcome_blind_static_code_only_cross_adjudication",
        "source_audit": str(DEFAULT_SOURCE_AUDIT),
        "source_audit_sha256": source_sha,
        "source_audit_schema": source["schema"],
        "ops_math_source": source["ops_math_source"],
        "ops_math_sha256": source["ops_math_sha256"],
        "review_policy": (
            "Credit only a decision-contributing requested code relation. Presence of a "
            "topic, analogy, proof marker, or artifact is not its function or quality. "
            "Trace excluded-field defaults through aggregation rather than assuming that "
            "removing an LLM field is neutral. Depth is the deepest surviving code-only "
            "operation under the frozen source-audit vocabulary."
        ),
        "forbidden_inputs": source["forbidden_inputs"],
        "forbidden_inputs_used": False,
        "candidate_execution_performed": False,
        "candidate_import_performed": False,
        "model_or_api_calls_performed": False,
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
            sorted(Counter(c["change_kind"] for c in changes).items())
        ),
        "changes": changes,
        "interpretation": (
            "This overlay corrects static relation descriptions, aggregation caveats, one "
            "depth, and one presence-versus-function verdict. Unchanged retrieved rows "
            "retain the source audit's bounded claims. Neither mismatch nor non-discovery "
            "establishes tacitness, inarticulability, or universal non-verifiability."
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
    if artifact.get("forbidden_inputs_used") is not False:
        raise ValueError("forbidden inputs were used")
    if artifact.get("candidate_execution_performed") is not False:
        raise ValueError("candidate execution was performed")
    if artifact.get("candidate_import_performed") is not False:
        raise ValueError("candidate import was performed")
    if artifact.get("model_or_api_calls_performed") is not False:
        raise ValueError("model/API calls were performed")

    rows = source["rows"]
    if len(rows) != 60 or len({r["cell_id"] for r in rows}) != 60:
        raise ValueError("source audit must contain 60 unique rows")
    retrieved = [r for r in rows if r.get("candidate") is not None]
    if len(retrieved) != 28:
        raise ValueError("source audit must contain 28 retrieved candidates")
    by_id = {row["cell_id"]: row for row in rows}

    coverage = artifact["review_coverage"]
    if coverage["source_rows"] != 60:
        raise ValueError("incorrect source-row coverage")
    if coverage["retrieved_candidates_reviewed"] != 28:
        raise ValueError("incorrect retrieved-candidate coverage")
    if coverage["retrieved_candidate_set_sha256"] != _review_set_digest(rows):
        raise ValueError("retrieved-candidate set guard mismatch")
    if coverage["all_retrieved_candidates_reviewed"] is not True:
        raise ValueError("cross-audit is not marked complete")

    changes = artifact["changes"]
    if len(changes) != len(CHANGE_SPECS):
        raise ValueError("wrong number of changes")
    if len({c["cell_id"] for c in changes}) != len(changes):
        raise ValueError("duplicate change cell")
    if {c["cell_id"] for c in changes} != set(CHANGE_SPECS):
        raise ValueError("change set differs from authored specs")
    if coverage["changed_rows"] != len(changes):
        raise ValueError("incorrect changed-row count")
    if coverage["unchanged_retrieved_rows"] != 28 - len(changes):
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
            raise ValueError(f"selected program hash changed: {cell_id}")

    ops_path = REPO_ROOT / artifact["ops_math_source"]
    actual_ops_sha = _sha256_bytes(ops_path.read_bytes())
    if actual_ops_sha != EXPECTED_OPS_MATH_SHA256:
        raise ValueError("ops_math hash changed")
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
        if depth is not None and (not isinstance(depth, int) or not 0 <= depth <= 4):
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
