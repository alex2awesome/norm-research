"""Independently cross-audit the frozen code-review construct-fidelity table.

This is a source-only, pre-execution instrument.  It reviews every retrieved
candidate in the canonical 90-cell code-review table and emits an additive
guarded overlay.  It never loads task items, prompt values, references,
outcomes, program outputs, or reconstruction results.

The audit makes two distinctions that the original table could blur:

* a structural correlate is not automatically an implemented sub-relation;
* the depth of the whole candidate program is not automatically the depth of
  the branch that actually matches the requested relation.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    ROOT
    / "outputs/metric_seam_pilot/hierarchy_r123/"
    "code_review_construct_fidelity_v2.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/metric_seam_pilot/hierarchy_r123/"
    "code_review_construct_fidelity_independent_cross_audit_v1.json"
)

SCHEMA = "metric-seam.code-review-construct-fidelity-independent-cross-audit.v1"
SOURCE_SCHEMA = "metric-seam.code-review-construct-fidelity-merged.v1"
LEVELS = ("R1", "R2", "R3")
ROW_STATE_FIELDS = (
    "verdict",
    "scope",
    "eligible_for_relation_local_execution",
    "audited_depth",
)


class CrossAuditError(ValueError):
    """Raised when the frozen source or an emitted overlay fails a guard."""


# This mapping freezes the complete review population.  It prevents a future
# candidate from being silently certified by a default branch and binds each
# reviewed cell to the source module that was actually inspected.
EXPECTED_RETRIEVED: dict[str, str] = {
    "TB::code-review::general::R1::merged_tree::124::fefb361c177a9fde3493": "a15",
    "TB::code-review::general::R1::merged_tree::125::ed5014bc2e3f88db5238": "a347",
    "TB::code-review::general::R1::merged_tree::137::ce1fd00b757dcbeaa064": "a88",
    "TB::code-review::general::R1::merged_tree::154::3b3db2a49261cf7e9891": "a18",
    "TB::code-review::general::R1::merged_tree::171::33b7ed9b7e4e601644ef": "a8",
    "TB::code-review::general::R1::merged_tree::19::a3ed15b08f82c1c27732": "a5",
    "TB::code-review::general::R1::merged_tree::230::ffe4d6661ee873dc0b7f": "a112",
    "TB::code-review::general::R1::merged_tree::29::31fc25952aa634846b1a": "a18",
    "TB::code-review::general::R1::merged_tree::31::f239fc227b096b1638ef": "a118",
    "TB::code-review::general::R1::merged_tree::39::8978ed2489b7ca126ec1": "a148",
    "TB::code-review::general::R1::merged_tree::43::54a9882d31bda2c084dc": "a47",
    "TB::code-review::general::R1::merged_tree::55::742ea284277cb2d01283": "a175",
    "TB::code-review::general::R1::merged_tree::83::81345133d1681ebfa23c": "a43",
    "TB::code-review::general::R1::merged_tree::94::c54443cc24d466bbfb18": "a70",
    "TB::code-review::general::R1::merged_tree::98::7c4db3822e45f99c6bce": "a43",
    "TB::code-review::general::R1::parented_tree::109::b2ca69bf6976fb8fb447": "a5",
    "TB::code-review::general::R1::parented_tree::127::5149406452a2761d4772": "a175",
    "TB::code-review::general::R1::parented_tree::130::9c920f5714978ff6c008": "a18",
    "TB::code-review::general::R1::parented_tree::21::e1c3bc938276569dc4bc": "a43",
    "TB::code-review::general::R1::parented_tree::240::89d88f11d74be4c924ab": "a104",
    "TB::code-review::general::R1::parented_tree::30::2f57addd702abf9b3d91": "a401",
    "TB::code-review::general::R1::parented_tree::315::2b3b13a351cb7821665c": "a112",
    "TB::code-review::general::R1::parented_tree::412::1f8ddc80a33f3d98f456": "a3",
    "TB::code-review::general::R1::parented_tree::88::aea0edd640d1ededc9b5": "a65",
    "TB::code-review::general::R2::grandparent::1::a2e3d89942e44f992d0a": "a5",
    "TB::code-review::general::R2::grandparent::31::cfb58f445e87c9098808": "a38",
    "TB::code-review::general::R2::grandparent::39::4929457fac99c8364d2b": "a92",
    "TB::code-review::general::R2::grandparent::43::12b78f2174bf884b965b": "a96",
    "TB::code-review::general::R2::grandparent::44::504809954bb08b978c5b": "a99",
    "TB::code-review::general::R2::grandparent::59::be3cff63def799193078": "a47",
    "TB::code-review::general::R2::grandparent::76::ed0c59341c122f14110b": "a175",
    "TB::code-review::general::R2::merged_group::102::7f3dcaecef8a8e60788e": "a104",
    "TB::code-review::general::R2::merged_group::118::ba8ef295035d8737c698": "a118",
    "TB::code-review::general::R2::merged_group::131::43ed2014b9a1669be3ca": "a131",
    "TB::code-review::general::R2::merged_group::31::e2b7345652fae91aabf6": "a401",
    "TB::code-review::general::R2::merged_group::3::9f17fdc8189a12428995": "a3",
    "TB::code-review::general::R2::merged_group::40::46a2dd9793d0881557cd": "a43",
    "TB::code-review::general::R2::merged_group::44::a248fa197f79e5e94f7d": "a118",
    "TB::code-review::general::R2::merged_group::4::aa14dfc5afe2bd7133bc": "a3",
    "TB::code-review::general::R2::merged_group::51::f8198da2d53d4f4219b2": "a52",
    "TB::code-review::general::R2::merged_group::74::76f22465ebfa26f20094": "a65",
    "TB::code-review::general::R2::merged_group::82::cca24c32a9b4560be92c": "a130",
    "TB::code-review::general::R2::merged_group::87::734883234b9f607cb9a6": "a38",
    "TB::code-review::general::R2::merged_group::88::623b4cfaade2e0ff389b": "a88",
    "TB::code-review::general::R3::grandparent::0::9c815cf7273820f4f7f1": "a37",
    "TB::code-review::general::R3::grandparent::14::38c3dcb57d12d9cbd880": "a112",
    "TB::code-review::general::R3::grandparent::15::7868422a985d046dcdc2": "a47",
    "TB::code-review::general::R3::grandparent::16::9a8efffaad0ee10facbb": "a280",
    "TB::code-review::general::R3::grandparent::20::b295cf2b241026f56f13": "a20",
    "TB::code-review::general::R3::grandparent::21::3420365609caef197413": "a184",
    "TB::code-review::general::R3::grandparent::22::5e63e1115f2d0f8b88f2": "a410",
    "TB::code-review::general::R3::grandparent::3::681c2abce3bef33e3781": "a131",
    "TB::code-review::general::R3::grandparent::6::451817f945294fa4abd8": "a76",
    "TB::code-review::general::R3::merged_group::0::7e7105b23885efd5e5ff": "a43",
    "TB::code-review::general::R3::merged_group::13::b71a08674e6794d89a82": "a52",
    "TB::code-review::general::R3::merged_group::14::f57a574fa387748b99c2": "a130",
    "TB::code-review::general::R3::merged_group::17::5b66a1627e1e35eb9b9b": "a191",
    "TB::code-review::general::R3::merged_group::19::40c516351670d980a895": "a3",
    "TB::code-review::general::R3::merged_group::1::d6f5bd2d4ef0a792711a": "a15",
    "TB::code-review::general::R3::merged_group::20::55d11b5d73aa3ced65cd": "a148",
    "TB::code-review::general::R3::merged_group::21::bdce02e573b7f97faf5d": "a92",
    "TB::code-review::general::R3::merged_group::25::af86a763745e1d7c2f83": "a47",
    "TB::code-review::general::R3::merged_group::28::66cb8ac9cbc2fd756a34": "a0",
    "TB::code-review::general::R3::merged_group::30::c5836627124ef9cf4b51": "a1",
    "TB::code-review::general::R3::merged_group::31::4b956308225f91ac3a64": "a76",
    "TB::code-review::general::R3::merged_group::3::2e8ab4cdcc604febf558": "a88",
    "TB::code-review::general::R3::merged_group::4::fd97c9079d16266eb063": "a38",
    "TB::code-review::general::R3::merged_group::9::c500175a8ff2b991a5d4": "a410",
}


# Corrections are deliberately sparse and guarded by the full before state.
# A mismatch keeps the depth of the inspected candidate for provenance, but
# matched_relation_depth becomes null and it leaves the eligible depth count.
VERDICT_CORRECTIONS: dict[str, str] = {
    "TB::code-review::general::R1::merged_tree::171::33b7ed9b7e4e601644ef": (
        "Size, file count, and top-directory spread never relate hunks by "
        "logical behavior or test whether concerns are independently splittable. "
        "They are scope correlates, not an executed atomicity sub-relation."
    ),
    "TB::code-review::general::R1::merged_tree::31::f239fc227b096b1638ef": (
        "The Rust rules score documentation, constructors, feature gates, macros, "
        "and re-exports, but never count/minimize the public surface or compare API "
        "versions. Neighboring API hygiene cannot substitute for smallness or "
        "conservative evolution."
    ),
    "TB::code-review::general::R2::grandparent::43::12b78f2174bf884b965b": (
        "The program counts formal-file, import, annotation, and CI-token presence. "
        "It neither runs a verifier nor reads a finding/result, so it cannot execute "
        "the requested produce-actionable-defect-results relation."
    ),
    "TB::code-review::general::R2::grandparent::44::504809954bb08b978c5b": (
        "Builder signals and wide-construction candidates are pooled globally and "
        "never linked to the same class or call. A gratuitous builder with no wide "
        "constructor is rewarded, so the requested conditional relation and its "
        "polarity are not implemented."
    ),
    "TB::code-review::general::R2::merged_group::131::43ed2014b9a1669be3ca": (
        "Any feature file and selected framework imports/call names receive credit, "
        "without parsing a requirement-to-scenario, action-to-assertion, or "
        "test-to-implementation relation and without execution. Style presence does "
        "not implement user-visible behavior verification."
    ),
    "TB::code-review::general::R3::grandparent::3::681c2abce3bef33e3781": (
        "Boundary-driver/BDD imports and call-name counts are not related to an "
        "intended requirement, public interface, observable assertion, or real "
        "interaction. The requested behavior correspondence is absent."
    ),
}

DEPTH_CORRECTIONS: dict[str, tuple[int, str]] = {
    "TB::code-review::general::R2::merged_group::51::f8198da2d53d4f4219b2": (
        1,
        "For Documentation IA, the matching branch counts parsed headings, links, "
        "lists, and code blocks within each added document. The candidate's depth-2 "
        "document-to-source co-change branch measures freshness, not IA, examples, "
        "or navigability, and therefore cannot raise matched-relation depth.",
    )
}

BANNED_IMPORT_ROOTS = {
    "anthropic",
    "httpx",
    "openai",
    "requests",
    "transformers",
    "urllib",
}
SENSITIVE_FIELD_NAMES = {
    "ground_truth",
    "judgment",
    "label",
    "llm",
    "outcome",
    "reference",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CrossAuditError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CrossAuditError(f"{path}: expected a JSON object")
    return value


def _row_state(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: copy.deepcopy(row.get(field)) for field in ROW_STATE_FIELDS}


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                try:
                    return ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    return None
    return None


def _program_source_audit(candidate: Mapping[str, Any]) -> dict[str, Any]:
    aspect_id = candidate.get("aspect_id")
    source_path = candidate.get("source_path")
    expected_sha = candidate.get("source_sha256")
    if not all(isinstance(value, str) and value for value in (aspect_id, source_path, expected_sha)):
        raise CrossAuditError(f"malformed candidate identity: {candidate!r}")
    path = (ROOT / source_path).resolve()
    if not path.is_file():
        raise CrossAuditError(f"missing candidate source: {source_path}")
    raw = path.read_bytes()
    observed_sha = _sha256_bytes(raw)
    if observed_sha != expected_sha:
        raise CrossAuditError(f"candidate source drift for {aspect_id}: {source_path}")
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=source_path)
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise CrossAuditError(f"cannot parse candidate source {source_path}: {exc}") from exc

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.lstrip(".").split(".", 1)[0])
    banned_imports = sorted(imports & BANNED_IMPORT_ROOTS)
    if banned_imports:
        raise CrossAuditError(f"{aspect_id}: model/network imports present: {banned_imports}")

    score_defs = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "score"
    ]
    if len(score_defs) != 1:
        raise CrossAuditError(f"{aspect_id}: expected exactly one module-level score()")
    score_def = score_defs[0]
    score_parameters = [arg.arg for arg in score_def.args.posonlyargs + score_def.args.args]
    if (
        score_parameters != ["diff_text"]
        or score_def.args.vararg is not None
        or score_def.args.kwarg is not None
        or score_def.args.kwonlyargs
    ):
        raise CrossAuditError(
            f"{aspect_id}: score input channel is not exactly score(diff_text)"
        )

    sensitive_accesses: list[str] = []
    for node in ast.walk(score_def):
        key: str | None = None
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            key = node.slice.value if isinstance(node.slice.value, str) else None
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            key = node.args[0].value
        if key and key.lower() in SENSITIVE_FIELD_NAMES:
            sensitive_accesses.append(key)
    if sensitive_accesses:
        raise CrossAuditError(
            f"{aspect_id}: score() accesses prohibited channel fields {sensitive_accesses}"
        )

    declared_aspect = _literal_assignment(tree, "ASPECT_ID")
    if declared_aspect != aspect_id:
        raise CrossAuditError(
            f"{aspect_id}: source declares ASPECT_ID={declared_aspect!r}"
        )
    return {
        "aspect_id": aspect_id,
        "source_path": source_path,
        "source_sha256": observed_sha,
        "score_parameters": score_parameters,
        "llm_or_judgment_field_accesses": [],
        "model_or_network_imports": [],
        "declared_tools": _literal_assignment(tree, "TOOLS"),
        "declared_tier": _literal_assignment(tree, "TIER"),
        "channel_conclusion": (
            "The candidate exposes score(diff_text) only. No L-channel, prompt, "
            "reference, outcome, model, or network input is available to score()."
        ),
    }


def _validate_source(source: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if source.get("schema") != SOURCE_SCHEMA:
        raise CrossAuditError(f"unexpected source schema: {source.get('schema')!r}")
    if source.get("status") != "static_construct_fidelity_complete_pre_execution":
        raise CrossAuditError("source audit is not the frozen pre-execution table")
    if source.get("task") != "code-review":
        raise CrossAuditError("source task is not code-review")
    for flag in (
        "execution_performed",
        "reference_values_loaded",
        "outcome_labels_loaded",
        "external_supervision",
    ):
        if source.get(flag) is not False:
            raise CrossAuditError(f"source violates sealed flag {flag}")
    rows = source.get("rows")
    if not isinstance(rows, list) or len(rows) != 90:
        raise CrossAuditError("source must contain exactly 90 rows")
    retrieved = [row for row in rows if isinstance(row, Mapping) and row.get("candidate")]
    observed: dict[str, str] = {}
    for row in retrieved:
        cell_id = row.get("cell_id")
        candidate = row.get("candidate")
        if not isinstance(cell_id, str) or not isinstance(candidate, Mapping):
            raise CrossAuditError("malformed retrieved source row")
        aspect_id = candidate.get("aspect_id")
        if not isinstance(aspect_id, str):
            raise CrossAuditError(f"{cell_id}: missing candidate aspect")
        if cell_id in observed:
            raise CrossAuditError(f"duplicate source cell: {cell_id}")
        observed[cell_id] = aspect_id
    if observed != EXPECTED_RETRIEVED:
        missing = sorted(set(EXPECTED_RETRIEVED) - set(observed))
        extra = sorted(set(observed) - set(EXPECTED_RETRIEVED))
        wrong = sorted(
            key
            for key in set(observed) & set(EXPECTED_RETRIEVED)
            if observed[key] != EXPECTED_RETRIEVED[key]
        )
        raise CrossAuditError(
            f"retrieved population drift: missing={missing[:3]}, extra={extra[:3]}, "
            f"wrong_candidate={wrong[:3]}"
        )
    return rows


def _summarize(states: list[Mapping[str, Any]], rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    verdicts = Counter(state["verdict"] for state in states)
    eligible_depths = Counter(
        str(state["audited_depth"])
        for state in states
        if state["eligible_for_relation_local_execution"]
    )
    by_level: dict[str, Any] = {}
    for level in LEVELS:
        indexes = [index for index, row in enumerate(rows) if row["level"] == level]
        local_states = [states[index] for index in indexes]
        local_verdicts = Counter(state["verdict"] for state in local_states)
        local_depths = Counter(
            str(state["audited_depth"])
            for state in local_states
            if state["eligible_for_relation_local_execution"]
        )
        by_level[level] = {
            "n_metrics": len(indexes),
            "verdict_counts": dict(sorted(local_verdicts.items())),
            "relation_local_static_fidelity_count": sum(
                bool(state["eligible_for_relation_local_execution"])
                for state in local_states
            ),
            "audited_depth_counts_eligible": dict(sorted(local_depths.items())),
        }
    eligible_programs = {
        row["candidate"]["aspect_id"]
        for row, state in zip(rows, states)
        if state["eligible_for_relation_local_execution"] and row.get("candidate")
    }
    return {
        "n_metrics": len(states),
        "verdict_counts": dict(sorted(verdicts.items())),
        "relation_local_static_fidelity_count": sum(
            bool(state["eligible_for_relation_local_execution"]) for state in states
        ),
        "whole_construct_exact_count": verdicts["exact"],
        "audited_depth_counts_eligible": dict(sorted(eligible_depths.items())),
        "n_unique_eligible_programs": len(eligible_programs),
        "by_level": by_level,
    }


def _diagnostics(cell_id: str, row: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, str]:
    aspect_id = row["candidate"]["aspect_id"]
    if cell_id in VERDICT_CORRECTIONS:
        if aspect_id in {"a96", "a131"}:
            presence = "presence_or_style_signal_without_requested_function"
        elif aspect_id == "a99":
            presence = "conditional_relation_not_linked"
        elif aspect_id == "a8":
            presence = "structural_correlate_without_logical_relation"
        else:
            presence = "neighboring_api_rules_without_size_or_evolution_relation"
    elif after["verdict"] == "mismatch":
        presence = "no_requested_relation_executed"
    else:
        presence = "named_relation_local_operation_only; residual_construct_unimplemented"

    if aspect_id == "a76" and after["eligible_for_relation_local_execution"]:
        polarity = (
            "correct only for modal handler-style consistency; uniformly swallowed "
            "failures can score high, so whole-construct robustness polarity is not claimed"
        )
    elif cell_id in VERDICT_CORRECTIONS:
        polarity = "not valid for the requested relation"
    elif after["eligible_for_relation_local_execution"]:
        polarity = "certified only for the named matched sub-relation"
    else:
        polarity = "not applicable because the candidate is a mismatch"

    if cell_id in DEPTH_CORRECTIONS:
        aggregation = "unrelated depth-2 freshness branch excluded from matched-relation depth"
    elif aspect_id == "a99" and cell_id in VERDICT_CORRECTIONS:
        aggregation = "unlinked builder and wide-constructor counts cannot implement the conditional"
    elif after["eligible_for_relation_local_execution"]:
        aggregation = "only decision-contributing operations named in implemented_relations are credited"
    else:
        aggregation = "candidate aggregation is not credited to the requested relation"

    caveats = row.get("dependency_applicability_caveats")
    applicability = caveats[0] if isinstance(caveats, list) and caveats else "No caveat recorded."
    return {
        "presence_vs_function": presence,
        "polarity": polarity,
        "aggregation": aggregation,
        "applicability": applicability,
        "l_channel": "absent from the candidate score interface and source imports",
    }


def build_cross_audit(
    source: Mapping[str, Any], *, source_name: str = str(DEFAULT_SOURCE.relative_to(ROOT))
) -> dict[str, Any]:
    """Return the complete 68-row additive cross-audit overlay."""

    all_rows = _validate_source(source)
    retrieved_rows = [row for row in all_rows if row.get("candidate")]

    program_candidates: dict[str, Mapping[str, Any]] = {}
    for row in retrieved_rows:
        candidate = row["candidate"]
        aspect_id = candidate["aspect_id"]
        prior = program_candidates.get(aspect_id)
        if prior is not None and prior != candidate:
            raise CrossAuditError(f"inconsistent candidate identity for {aspect_id}")
        program_candidates[aspect_id] = candidate
    program_audits = [
        _program_source_audit(program_candidates[aspect_id])
        for aspect_id in sorted(program_candidates, key=lambda value: int(value[1:]))
    ]

    reviews: list[dict[str, Any]] = []
    before_states: list[dict[str, Any]] = []
    after_states: list[dict[str, Any]] = []
    for row in retrieved_rows:
        cell_id = row["cell_id"]
        before = _row_state(row)
        after = copy.deepcopy(before)
        if cell_id in VERDICT_CORRECTIONS:
            if before != {
                "verdict": "partial",
                "scope": "subrelation_only",
                "eligible_for_relation_local_execution": True,
                "audited_depth": before["audited_depth"],
            }:
                raise CrossAuditError(f"{cell_id}: verdict-correction before state drift")
            after.update(
                verdict="mismatch",
                scope="none",
                eligible_for_relation_local_execution=False,
            )
        if cell_id in DEPTH_CORRECTIONS:
            new_depth, _ = DEPTH_CORRECTIONS[cell_id]
            if before["verdict"] != "partial" or before["audited_depth"] != 2:
                raise CrossAuditError(f"{cell_id}: depth-correction before state drift")
            after["audited_depth"] = new_depth

        changed_fields = [field for field in ROW_STATE_FIELDS if before[field] != after[field]]
        if cell_id in VERDICT_CORRECTIONS:
            reason = VERDICT_CORRECTIONS[cell_id]
        elif cell_id in DEPTH_CORRECTIONS:
            reason = DEPTH_CORRECTIONS[cell_id][1]
        elif after["verdict"] == "partial":
            reason = (
                "Certified only as the already named relation-local sub-relation after "
                "independent source inspection. " + str(row.get("rationale") or "")
            )
        else:
            reason = (
                "Certified as a retrieved mismatch after independent source inspection. "
                + str(row.get("rationale") or "")
            )
        reviews.append(
            {
                "cell_id": cell_id,
                "level": row["level"],
                "metric_name": row["metric_name"],
                "candidate": copy.deepcopy(row["candidate"]),
                "before": before,
                "after": after,
                "changed_fields": changed_fields,
                "matched_relation_depth": (
                    after["audited_depth"]
                    if after["eligible_for_relation_local_execution"]
                    else None
                ),
                "relation_diagnostics": _diagnostics(cell_id, row, after),
                "reason": reason,
                "review_status": "independently_source_reviewed",
            }
        )
        before_states.append(before)
        after_states.append(after)

    all_before = [_row_state(row) for row in all_rows]
    all_after = copy.deepcopy(all_before)
    all_index = {row["cell_id"]: index for index, row in enumerate(all_rows)}
    for review in reviews:
        all_after[all_index[review["cell_id"]]] = copy.deepcopy(review["after"])
    before_summary = _summarize(all_before, all_rows)
    after_summary = _summarize(all_after, all_rows)

    changes = [
        {
            "cell_id": review["cell_id"],
            "candidate_aspect_id": review["candidate"]["aspect_id"],
            "before": review["before"],
            "after": review["after"],
            "changed_fields": review["changed_fields"],
            "reason": review["reason"],
        }
        for review in reviews
        if review["changed_fields"]
    ]
    if len(reviews) != 68 or len(changes) != 7:
        raise CrossAuditError(
            f"unexpected review/change coverage: reviews={len(reviews)}, changes={len(changes)}"
        )

    return {
        "schema": SCHEMA,
        "status": "complete_independent_static_cross_audit_pre_execution",
        "task": "code-review",
        "design_scope": "blind_static_source_only_construct_fidelity_cross_audit",
        "source_construct_fidelity": source_name,
        "source_construct_fidelity_sha256": _sha256_bytes(
            json.dumps(source, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "sealed_inputs": {
            "items_loaded": False,
            "candidate_execution_performed": False,
            "prompt_or_model_calls_performed": False,
            "llm_judgments_loaded": False,
            "references_loaded": False,
            "outcomes_loaded": False,
            "program_outputs_loaded": False,
            "correlations_loaded": False,
            "reconstruction_results_loaded": False,
            "external_supervision_used": False,
            "gpu_used": False,
        },
        "review_policy": {
            "relation_match": (
                "Partial requires a decision-contributing operation for an explicitly "
                "requested sub-relation. Topic overlap, a generic correlate, tool/style "
                "presence where function is requested, or an unexecuted conditional does "
                "not qualify."
            ),
            "depth": (
                "Matched-relation depth is the deepest decision-contributing operation "
                "for the matched sub-relation. Unrelated deeper branches cannot raise it."
            ),
            "negative_result": (
                "Mismatch and abstention are bounded non-discovery within the frozen "
                "historical program bank, never evidence of tacitness."
            ),
        },
        "provenance": {
            "program_bank": (
                "retrospective historical code-review A-bank; programs were manually "
                "constructed and original authorship is not uniformly encoded"
            ),
            "candidate_placement": (
                "deterministic automatic source-metadata retrieval in the frozen seed map"
            ),
            "automatic_program_discovery_claimed": False,
            "cross_audit": "manual independent source adjudication encoded by this builder",
        },
        "coverage": {
            "n_panel_cells": 90,
            "n_retrieved_reviewed": len(reviews),
            "n_previously_accepted_reviewed": sum(
                review["before"]["verdict"] == "partial" for review in reviews
            ),
            "n_previously_mismatch_reviewed": sum(
                review["before"]["verdict"] == "mismatch" for review in reviews
            ),
            "n_unique_program_sources_reviewed": len(program_audits),
            "complete": True,
        },
        "before_summary": before_summary,
        "after_summary": after_summary,
        "headline_correction": {
            "relation_local_static_fidelity_count_before": before_summary[
                "relation_local_static_fidelity_count"
            ],
            "relation_local_static_fidelity_count_after": after_summary[
                "relation_local_static_fidelity_count"
            ],
            "balanced_panel_fraction_before": round(
                before_summary["relation_local_static_fidelity_count"] / 90, 6
            ),
            "balanced_panel_fraction_after": round(
                after_summary["relation_local_static_fidelity_count"] / 90, 6
            ),
            "eligible_depth_counts_before": before_summary[
                "audited_depth_counts_eligible"
            ],
            "eligible_depth_counts_after": after_summary[
                "audited_depth_counts_eligible"
            ],
            "interpretation": (
                "This corrects a static source-fidelity gate only. It is not a codability, "
                "execution, reconstruction, isomorphism, articulability, verifiability, "
                "or tacitness estimate."
            ),
        },
        "n_guarded_changes": len(changes),
        "changes": changes,
        "program_source_audits": program_audits,
        "reviews": reviews,
    }


def validate_cross_audit(
    artifact: Mapping[str, Any], source: Mapping[str, Any]
) -> None:
    """Validate an artifact by rebuilding it from the frozen source."""

    expected = build_cross_audit(
        source, source_name=str(artifact.get("source_construct_fidelity") or "")
    )
    if artifact != expected:
        raise CrossAuditError("cross-audit artifact differs from guarded rebuild")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite additive artifact: {args.out}")
    source = _load(args.source)
    try:
        source_name = str(args.source.resolve().relative_to(ROOT))
    except ValueError:
        source_name = str(args.source.resolve())
    artifact = build_cross_audit(source, source_name=source_name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact["headline_correction"], indent=2))


if __name__ == "__main__":
    main()
