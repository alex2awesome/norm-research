#!/usr/bin/env python3
"""Inventory unused historical code programs for corrected code-review gaps.

This is a source-only developmental proposal.  It inspects the frozen
hierarchy construct table, its independent static overlay, and literal source
metadata from the existing metric library.  It does not import or execute a
candidate and never reads task items, program outputs, outcomes, references,
prompt responses, or reconstruction results.

Every corrected gap receives exactly one conservative disposition:

* ``propose_partial_mapping``: a named requested sub-relation is actually
  implemented, pending an independent source audit and later train replay;
* ``near_match_reject``: executable code is adjacent but misses a required
  relation, channel, applicability gate, or aggregation contract;
* ``nonexecutable_catalog_only``: a semantically relevant historical entry
  exists but is deliberately Tier 0 / THICK and supplies no code witness;
* ``bounded_non_discovery``: this unused library contains no defensible seed.

Existing/manual programs are allowed retrospective pipeline seeds.  Nothing
in this inventory is described as automatically discovered or certified.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from methods.metric_seam.hierarchy_seed_mapper import (
    ProgramMetadata,
    load_program_library,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"
METRICS = ROOT / "methods/existing_metrics_runner/coded/metrics"
FIDELITY = BASE / "code_review_construct_fidelity_v2.json"
CROSS_AUDIT = BASE / "code_review_construct_fidelity_independent_cross_audit_v1.json"
SEED_MAP = BASE / "code_review_seed_map_v3.json"
CORRECTED_FUNNEL = BASE / "code_review_corrected_funnel_v1.json"
DEFAULT_OUT = BASE / "code_review_unused_program_feasibility_inventory_v1.json"

SCHEMA = "metric-seam.code-review-unused-program-feasibility-inventory.v1"
DECISIONS = {
    "propose_partial_mapping",
    "near_match_reject",
    "nonexecutable_catalog_only",
    "bounded_non_discovery",
}
DEPTH_VOCABULARY = {
    0: "surface lexical operation",
    1: "parsed document/code structure",
    2: "cross-span, cross-file, or cross-section relation",
    3: "formal solver or evidence-graph execution",
    4: "environment or test execution",
}
BANNED_IMPORT_ROOTS = {
    "anthropic", "httpx", "openai", "requests", "transformers", "urllib",
}


def _proposal(
    candidate: str,
    implemented_relation: str,
    depth: int,
    applicability: str,
    abstention: str,
    rationale: str,
) -> dict[str, Any]:
    return {
        "decision": "propose_partial_mapping",
        "candidate": candidate,
        "implemented_relation": implemented_relation,
        "proposed_matched_relation_depth": depth,
        "applicability": applicability,
        "abstention": abstention,
        "rationale": rationale,
    }


def _near(
    candidate: str,
    implemented_relation: str,
    applicability: str,
    abstention: str,
    rationale: str,
) -> dict[str, Any]:
    return {
        "decision": "near_match_reject",
        "candidate": candidate,
        "implemented_relation": implemented_relation,
        "proposed_matched_relation_depth": None,
        "applicability": applicability,
        "abstention": abstention,
        "rationale": rationale,
    }


def _catalog(candidate: str, relation: str, rationale: str) -> dict[str, Any]:
    return {
        "decision": "nonexecutable_catalog_only",
        "candidate": candidate,
        "implemented_relation": relation,
        "proposed_matched_relation_depth": None,
        "applicability": "No executable applies()/score() contract: Tier 0 / THICK catalog entry.",
        "abstention": "Always non-executable in the current program class.",
        "rationale": rationale,
    }


def _bounded(rationale: str) -> dict[str, Any]:
    return {
        "decision": "bounded_non_discovery",
        "candidate": None,
        "implemented_relation": None,
        "proposed_matched_relation_depth": None,
        "applicability": None,
        "abstention": "No source-faithful unused program seed found in this frozen library inventory.",
        "rationale": rationale,
    }


# Frozen only after the read-only feasibility pass.  The keys are the complete
# corrected 40-gap population, preventing permissive default acceptance.
PROPOSALS: dict[str, dict[str, Any]] = {
    "TB::code-review::general::R1::merged_tree::125::ed5014bc2e3f88db5238": _near(
        "a202",
        "Run pydocstyle-compatible Ruff rules over existing added Python docstrings and score formatting violations per documented definition.",
        "Added Python code containing at least one documented definition and an available Ruff executable.",
        "No Python, no documented definition, unavailable Ruff, or no measurable reconstructed fragment.",
        "Docstring formatting is not MDN content-authoring guidance and never compares alternatives or the usefulness/consistency of prose.",
    ),
    "TB::code-review::general::R1::merged_tree::171::33b7ed9b7e4e601644ef": _bounded(
        "Unused programs expose size, test-file, comment, or refactoring correlates but none relates all hunks by one logical behavior or detects independently splittable concerns."
    ),
    "TB::code-review::general::R1::merged_tree::31::f239fc227b096b1638ef": _proposal(
        "a35",
        "Within added supported-language classes, relate public mutable fields to accessor presence and public return annotations to concrete collection implementations, then score implementation leakage versus stable abstraction.",
        2,
        "Added Python, JavaScript, TypeScript, or Java class/interface bodies with classifiable fields, methods, accessors, or public return annotations.",
        "No supported class/interface, empty marker classes, or no classifiable public-member relation.",
        "This executes the stable-abstraction/client-coupling sub-relation only; it does not count total API size or compare API versions, so scope remains partial.",
    ),
    "TB::code-review::general::R1::merged_tree::43::54a9882d31bda2c084dc": _near(
        "a232",
        "Detect five class-level Fowler refactoring signatures across Python/Java pre/post fragments.",
        "Python or Java class changes containing both additions and removals.",
        "No supported class-level pre/post change or no parseable class relation.",
        "Generic OO refactorings neither establish security effectiveness nor derive or extend a security-specific reusable pattern."
    ),
    "TB::code-review::general::R1::merged_tree::55::742ea284277cb2d01283": _proposal(
        "a72",
        "Reconstruct added supported-language file fragments, invoke the language's canonical formatter in check mode, and aggregate formatter conformance weighted by added lines.",
        2,
        "Added Python, Go, JavaScript, TypeScript, or Java files for which the corresponding formatter can be invoked.",
        "No supported file, missing formatter, or every reconstructed fragment is unmeasurable because of truncation/parse/tool failure.",
        "Canonical formatting is an executable coding-standard sub-relation; project-local naming, idioms, policy authority, and justified deviations remain outside the program."
    ),
    "TB::code-review::general::R1::merged_tree::83::81345133d1681ebfa23c": _near(
        "a178",
        "Relate one-line Python/Go accessor bodies to their method names and score avoidance of get_/Get prefixes.",
        "Added Python or Go accessor-shaped methods.",
        "No accessor-shaped method or no parseable supported code.",
        "The accessor relation is usage-adjacent, but the program does not establish that a method is public API or whether its wording communicates caller-visible semantics."
    ),
    "TB::code-review::general::R1::parented_tree::21::e1c3bc938276569dc4bc": _proposal(
        "a72",
        "Reconstruct added supported-language file fragments, invoke canonical ecosystem formatters, and aggregate formatter conformance weighted by added lines.",
        2,
        "Added Python, Go, JavaScript, TypeScript, or Java files with an available formatter.",
        "No supported file, missing formatter, or all reconstructed fragments are unmeasurable.",
        "This certifies only the ecosystem-formatting convention sub-relation; it does not compare against repository-local code or decide whether a deviation improves health."
    ),
    "TB::code-review::general::R1::parented_tree::240::89d88f11d74be4c924ab": _near(
        "a89",
        "Parse supported test files and compute interaction-assertion versus state-assertion call ratios.",
        "Supported test files containing recognized assertion calls.",
        "No recognized test path, no parseable test code, or no recognized assertions.",
        "Assertion-call style does not implement test-level selection, organization, independence, case right-sizing, boundary coverage, or dependency appropriateness."
    ),
    "TB::code-review::general::R1::parented_tree::292::9715d7b9a882ac04b634": _near(
        "a72",
        "Run canonical language formatters over reconstructed added code and score conformance.",
        "Added supported-language files with an available formatter.",
        "No supported/measurable formatted fragment.",
        "Formatter conformance is one rule application, but the construct requires guideline authority, applicable scope, canonical selection, and explicit exception rationale."
    ),
    "TB::code-review::general::R1::parented_tree::296::0c319d9296a0d7738e18": _near(
        "a80",
        "Compare Python pre/post fragments for Extract/Inline/Rename/Guard-Clause/Introduce-Parameter-Object signatures.",
        "Python files with parseable added and removed function-level fragments.",
        "No supported pre/post function relation or insufficient parseable context.",
        "Named refactoring presence does not identify a legacy system, establish invariant preservation, or handle isolated migration, backport, and maintenance constraints."
    ),
    "TB::code-review::general::R1::parented_tree::329::62f59cd542c0650fea94": _bounded(
        "No unused executable program identifies user hints, on-demand availability, speech frequency, or interaction complexity."
    ),
    "TB::code-review::general::R1::parented_tree::412::1f8ddc80a33f3d98f456": _near(
        "a35",
        "Relate class fields, accessors, and concrete return types to information-hiding failures.",
        "Added supported-language classes with classifiable member relations.",
        "No supported class/member relation.",
        "Information hiding is adjacent to responsibility boundaries, but the program never relates module parts through shared data or assigns one actor/reason to change."
    ),
    "TB::code-review::general::R1::parented_tree::424::b66c601f666efa93f31f": _bounded(
        "No unused executable program parses research citations, evidentiary claims, limitations, or cross-project transfer arguments."
    ),
    "TB::code-review::general::R1::parented_tree::467::917dcff0ea9a54fa869a": _catalog(
        "a45",
        "Catalog description covers resource-oriented API design and simplicity.",
        "The semantically relevant entry is deliberately THICK/Tier 0 and provides no executable resource-hierarchy or embedding relation."
    ),
    "TB::code-review::general::R1::parented_tree::500::deeedeec921e759fcacf": _bounded(
        "No unused executable program parses responsive layout, spacing, motion, typography, component-overview, or reusable visual-asset relations."
    ),
    "TB::code-review::general::R1::parented_tree::88::aea0edd640d1ededc9b5": _near(
        "a305",
        "Parse added HTML/JSX media elements and relate audio/video nodes to caption, subtitle, track, or transcript-link structures.",
        "Added supported HTML/JSX-family audio or video elements.",
        "No media element or no available parser for the relevant fragment.",
        "A caption/transcript artifact is a narrow accessibility implementation; it does not establish an accessibility-first design process, mindset, or remediation strategy."
    ),
    "TB::code-review::general::R2::grandparent::14::4bdf25d641991adb5485": _catalog(
        "a78",
        "Catalog description covers change/commit/PR communication quality.",
        "The exact communication-oriented entry is THICK/Tier 0; no unused executable program combines change size, structure, intent, scope, and impact for reviewer understanding."
    ),
    "TB::code-review::general::R2::grandparent::35::9f1c0bad3ec0de67775b": _bounded(
        "No unused executable program identifies concurrency/async models and relates lifecycle, synchronization, observability, or debuggability."
    ),
    "TB::code-review::general::R2::grandparent::3::2e742c97fa237c1e3aa6": _proposal(
        "a400",
        "Parse added Python functions and relate loop nesting, recursion/self-call multiplicity, memoization, binary-search shape, and sorted-plus-loop structure to a heuristic worst-case Big-O class.",
        2,
        "Added parseable Python functions.",
        "No added Python function or no parseable supported fragment.",
        "This executes the algorithmic-complexity sub-relation only; evidence from profiles, API design, correctness trade-offs, and scalability justification remains unimplemented."
    ),
    "TB::code-review::general::R2::grandparent::40::6baf94558fe45a9add63": _bounded(
        "No unused executable program extracts user-facing messages and evaluates clarity, actionability, recovery guidance, frequency, or security/privacy guidance."
    ),
    "TB::code-review::general::R2::grandparent::43::12b78f2174bf884b965b": _proposal(
        "a181",
        "Reconstruct added Python files, execute Ruff static analysis, parse actual diagnostic counts, and normalize lint-finding density by added lines.",
        1,
        "Added Python code with Ruff available and at least one measurable reconstructed fragment.",
        "No added Python, unavailable Ruff, or every reconstructed fragment is unparseable/unmeasurable.",
        "Unlike the rejected machinery-presence seed, this consumes real static-analysis findings. It remains a Python/Ruff defect-detection slice and does not run formal proofs, dynamic debugging, or repair."
    ),
    "TB::code-review::general::R2::grandparent::44::504809954bb08b978c5b": _near(
        "a80",
        "Detect several function-level Fowler patterns, including a weak Introduce Parameter Object signature.",
        "Python pre/post function fragments.",
        "No supported pre/post function relation.",
        "The score pools unrelated refactorings and never conditionally links a wide constructor to the introduced parameter object, so the requested ergonomic relation and polarity remain absent."
    ),
    "TB::code-review::general::R2::grandparent::57::8c59fd0f3eb33cc1c1b4": _catalog(
        "a85",
        "Catalog description covers resilience, availability, and fault tolerance.",
        "The relevant entry is THICK/Tier 0; no unused executable candidate relates distributed failure detection, isolation, recovery, and sustained availability."
    ),
    "TB::code-review::general::R2::grandparent::58::72af4fe863b520ca3df9": _near(
        "a400",
        "Classify added Python function shapes into heuristic asymptotic-complexity buckets.",
        "Added parseable Python functions.",
        "No parseable Python function.",
        "Asymptotic source shape does not execute load, latency, throughput, throttling, caching, or data-pipeline responsiveness relations."
    ),
    "TB::code-review::general::R2::grandparent::81::d5aaae5996a19a393c23": _bounded(
        "No unused executable program reads repository history or relates VCS/workflow capabilities to team size, distribution, collaboration, or scale."
    ),
    "TB::code-review::general::R2::grandparent::83::1bb2184d821d43157c61": _bounded(
        "No unused executable program reads roadmaps, enterprise features, trust collateral, feasibility, or stakeholder-confidence evidence."
    ),
    "TB::code-review::general::R2::merged_group::102::7f3dcaecef8a8e60788e": _catalog(
        "a102",
        "Catalog description exactly covers tests driving design and refactoring.",
        "The exact-title entry is THICK/Tier 0; executable test/refactoring neighbors do not establish that tests causally shaped design or enabled refactoring over time."
    ),
    "TB::code-review::general::R2::merged_group::116::927550bd426a03b2e337": _near(
        "a202",
        "Run pydocstyle-compatible structural/formatting rules on existing added Python docstrings.",
        "Added Python definitions with docstrings and available Ruff.",
        "No documented Python definition or no measurable Ruff result.",
        "Docstring capitalization, whitespace, and section formatting do not measure helpful voice, procedural tone, worldwide comprehensibility, or audience fit."
    ),
    "TB::code-review::general::R2::merged_group::123::a2a43bff0666769d0822": _proposal(
        "a309",
        "Parse changed file paths and relate each non-test source filename to a plausibly corresponding changed test filename, then score the matched source-file fraction.",
        2,
        "A diff touching at least one identifiable supported source-code file.",
        "No identifiable source-code file or no usable path relation.",
        "This executes the explicitly named tests-included baseline only; coding-practice quality, debt reduction, actual test behavior, and integration readiness remain unimplemented."
    ),
    "TB::code-review::general::R2::merged_group::131::43ed2014b9a1669be3ca": _near(
        "a127",
        "Parse test names for English/verb-like behavior specification and penalize pixel/snapshot-fidelity assertions.",
        "Supported test files with recognized test declarations or pixel/snapshot calls.",
        "No recognized test path/declaration or no parseable supported test fragment.",
        "Readable test names and snapshot avoidance do not relate a requirement to a scenario, observable assertion, implementation, or executed acceptance result."
    ),
    "TB::code-review::general::R2::merged_group::40::46a2dd9793d0881557cd": _proposal(
        "a72",
        "Reconstruct added supported-language fragments, run canonical ecosystem formatters, and aggregate formatter conformance weighted by added lines.",
        2,
        "Added supported-language files with an available formatter.",
        "No supported file, missing formatter, or all fragments unmeasurable.",
        "This executes ecosystem formatting adherence only; repository-local comparison and the justification/benefit of deviations remain outside scope."
    ),
    "TB::code-review::general::R3::grandparent::11::2c562440277fde02d164": _proposal(
        "a25",
        "Identify added configuration files, parse/lint their structure, scan committed secrets, and penalize large configuration bursts to score managed/minimal configuration.",
        1,
        "Added non-lock configuration files in supported YAML/JSON/TOML/INI/env families.",
        "No supported configuration change; unavailable sub-tools become omitted/default-neutral branches rather than evidence.",
        "The requested construct explicitly names minimal and explicit configuration. This implements the minimal/managed configuration slice, not general simplicity, magic/metaprogramming, or speculative abstraction."
    ),
    "TB::code-review::general::R3::grandparent::12::090af4e25717489168bb": _proposal(
        "a400",
        "Parse added Python functions and relate loop nesting, recursion, memoization, binary-search shape, and sorted-plus-loop structure to a heuristic worst-case Big-O class.",
        2,
        "Added parseable Python functions.",
        "No added Python function or no parseable fragment.",
        "This implements the efficient-algorithm source-shape sub-relation only; profiles, workloads, resource constraints, latency, and empirical trade-off justification remain absent."
    ),
    "TB::code-review::general::R3::grandparent::18::baf83bcbdb28b38c6317": _bounded(
        "No unused executable program consumes design tokens, component usage, rendered UI, brand guidance, or visual-coherence evidence."
    ),
    "TB::code-review::general::R3::grandparent::19::65db7483b9b1675f88f0": _catalog(
        "a268",
        "Catalog description covers component and project structure conventions.",
        "The architecture-adjacent entry is THICK/Tier 0; no executable unused program compares a change to established architecture and operating context."
    ),
    "TB::code-review::general::R3::grandparent::3::681c2abce3bef33e3781": _near(
        "a89",
        "Parse test bodies and score interaction-assertion versus state-assertion call ratios.",
        "Supported test files with recognized assertions.",
        "No recognized test/assertion relation.",
        "Assertion style is not related to intended requirements, a public interface, real user interactions, or safe-refactoring evidence, so the prior a131 mismatch is not repaired."
    ),
    "TB::code-review::general::R3::grandparent::7::5d7903455895f9dde690": _catalog(
        "a45",
        "Catalog description covers resource-oriented external API design and simplicity.",
        "The semantically relevant entry is THICK/Tier 0 and supplies no executable domain-resource or internal-schema-boundary relation."
    ),
    "TB::code-review::general::R3::grandparent::8::ca1ea174680697aba810": _near(
        "a202",
        "Run pydocstyle-compatible formatting rules over existing added Python docstrings.",
        "Added Python definitions with docstrings and available Ruff.",
        "No documented Python definition or no measurable Ruff result.",
        "The program neither identifies public API declarations nor measures documentation coverage, content correctness, ergonomics, safety, representative use, or whether clients can use the API without reading implementation."
    ),
    "TB::code-review::general::R3::merged_group::13::b71a08674e6794d89a82": _near(
        "a202",
        "Run pydocstyle-compatible capitalization, whitespace, section-order, and placement checks on added Python docstrings.",
        "Added Python definitions with docstrings and available Ruff.",
        "No documented Python definition or no measurable Ruff result.",
        "Formatting and capitalization rules do not implement voice, tone, plain-language clarity, locale/audience fit, or ambiguity detection."
    ),
    "TB::code-review::general::R3::merged_group::17::5b66a1627e1e35eb9b9b": _near(
        "a35",
        "Relate public fields to accessors and public concrete collection returns to stable interface abstractions in added class bodies.",
        "Added supported-language classes with classifiable public-member relations.",
        "No supported class/member relation.",
        "Added-code abstraction leakage is relevant to future stability but never compares prior API versions, clients, deprecations, versioning, additive evolution, or backward compatibility."
    ),
}


class InventoryError(ValueError):
    """Raised when the proposed source-only inventory is incomplete or unsafe."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InventoryError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return sorted(roots)


def _corrected_gaps(fidelity: Mapping, cross_audit: Mapping) -> list[dict]:
    if fidelity.get("schema") != "metric-seam.code-review-construct-fidelity-merged.v1":
        raise InventoryError("unexpected fidelity schema")
    if cross_audit.get("schema") != (
        "metric-seam.code-review-construct-fidelity-independent-cross-audit.v1"
    ):
        raise InventoryError("unexpected independent cross-audit schema")
    reviews = {row["cell_id"]: row for row in cross_audit["reviews"]}
    gaps = []
    for row in fidelity["rows"]:
        review = reviews.get(row["cell_id"])
        state = review["after"] if review is not None else {
            key: row.get(key) for key in (
                "verdict", "scope", "eligible_for_relation_local_execution", "audited_depth"
            )
        }
        if state.get("eligible_for_relation_local_execution") is not True:
            gaps.append({**row, "corrected_before_state": state})
    if len(gaps) != 40 or Counter(row["level"] for row in gaps) != Counter(
        {"R1": 16, "R2": 15, "R3": 9}
    ):
        raise InventoryError("corrected gap population is not R1=16/R2=15/R3=9")
    return gaps


def _old_bank(seed_map: Mapping) -> set[str]:
    if seed_map.get("schema") != "metric-seam.hierarchy-code-review-seed-map.v1":
        raise InventoryError("unexpected seed-map schema")
    used = {
        row["selected_seed"]["aspect_id"]
        for row in seed_map["rows"] if row.get("selected_seed") is not None
    }
    if len(used) != 33:
        raise InventoryError("historical program bank is not 33 unique programs")
    return used


def _program_record(program: ProgramMetadata) -> dict[str, Any]:
    path = ROOT / program.path
    imports = _imports(path)
    banned = sorted(set(imports) & BANNED_IMPORT_ROOTS)
    if banned:
        raise InventoryError(f"candidate {program.aspect_id} imports banned roots: {banned}")
    return {
        "aspect_id": program.aspect_id,
        "aspect_name": program.aspect_name,
        "source_path": program.path,
        "source_sha256": _sha256(path),
        "declared_tool_tier": program.declared_tier,
        "declared_tool_tier_meaning": (
            "non-executable catalog entry" if program.declared_tier == 0
            else "historical module-local tool tier; distinct from matched-relation depth"
        ),
        "declared_classification": program.classification,
        "declared_tools": list(program.tools),
        "declared_languages": list(program.languages),
        "derived_program_shape": program.program_shape,
        "imported_module_roots": imports,
        "model_or_network_import_detected": False,
        "execution_performed": False,
        "provenance": (
            "retrospective reuse of an existing/manual historical metric program; "
            "not automatic program discovery"
        ),
    }


def build_inventory() -> dict[str, Any]:
    fidelity = _load(FIDELITY)
    cross = _load(CROSS_AUDIT)
    seed = _load(SEED_MAP)
    corrected = _load(CORRECTED_FUNNEL)
    gaps = _corrected_gaps(fidelity, cross)
    old_bank = _old_bank(seed)
    programs = load_program_library(METRICS, repo_root=ROOT)
    by_id = {program.aspect_id: program for program in programs}
    deep_executable = {
        program.aspect_id for program in programs
        if program.executable and program.declared_tier >= 2
    }
    unused_deep = deep_executable - old_bank
    if len(programs) != 154 or len(deep_executable) != 134 or len(unused_deep) != 101:
        raise InventoryError("program-library inventory drifted")
    gap_ids = {row["cell_id"] for row in gaps}
    if set(PROPOSALS) != gap_ids:
        raise InventoryError(
            f"proposal population drift: missing={sorted(gap_ids-set(PROPOSALS))}, "
            f"extra={sorted(set(PROPOSALS)-gap_ids)}"
        )

    output_rows = []
    for row in gaps:
        spec = PROPOSALS[row["cell_id"]]
        if spec["decision"] not in DECISIONS:
            raise InventoryError(f"invalid decision for {row['cell_id']}")
        candidate_id = spec["candidate"]
        candidate = None
        if candidate_id is not None:
            if candidate_id in old_bank:
                raise InventoryError(f"candidate {candidate_id} was already in the 33-program bank")
            if candidate_id not in by_id:
                raise InventoryError(f"unknown candidate {candidate_id}")
            candidate = _program_record(by_id[candidate_id])
        decision = spec["decision"]
        if decision in {"propose_partial_mapping", "near_match_reject"}:
            if candidate_id not in unused_deep:
                raise InventoryError(f"{decision} candidate {candidate_id} is not unused Tier>=2")
        if decision == "nonexecutable_catalog_only":
            if candidate is None or candidate["declared_tool_tier"] != 0:
                raise InventoryError("catalog-only decision requires an unused Tier-0 module")
        depth = spec["proposed_matched_relation_depth"]
        if decision == "propose_partial_mapping":
            if depth not in DEPTH_VOCABULARY or depth == 0:
                raise InventoryError("proposed mapping requires a nonlexical matched depth")
            scope = "subrelation_only"
            eligible = True
        else:
            if depth is not None:
                raise InventoryError("rejected/nonexecuted row cannot claim matched depth")
            scope = "none"
            eligible = False
        output_rows.append({
            "cell_id": row["cell_id"],
            "level": row["level"],
            "metric_name": row["metric_name"],
            "metric_description": row["metric_description"],
            "requested_relation": row["requested_relation"],
            "corrected_before_state": row["corrected_before_state"],
            "decision": decision,
            "candidate": candidate,
            "implemented_relation": spec["implemented_relation"],
            "proposed_matched_relation_depth": depth,
            "proposed_matched_relation_depth_meaning": (
                DEPTH_VOCABULARY[depth] if depth is not None else None
            ),
            "proposed_scope": scope,
            "eligible_for_future_independent_source_audit": eligible,
            "applicability": spec["applicability"],
            "abstention": spec["abstention"],
            "rationale": spec["rationale"],
            "program_execution_performed": False,
        })

    decisions = Counter(row["decision"] for row in output_rows)
    proposed = [row for row in output_rows if row["decision"] == "propose_partial_mapping"]
    proposed_by_level = Counter(row["level"] for row in proposed)
    current_stages = corrected["corrected_readout"]["by_level"]
    projected = {}
    for level in ("R1", "R2", "R3"):
        current = current_stages[level]["relation_local_static_fidelity"][
            "balanced_panel"
        ]["n_positive"]
        addition = proposed_by_level[level]
        projected[level] = {
            "current_corrected_static": current,
            "proposed_pending_independent_audit": addition,
            "upper_bound_if_every_proposal_survives": current + addition,
            "remaining_to_30_under_that_upper_bound": 30 - current - addition,
        }
    return {
        "schema": SCHEMA,
        "status": "developmental_source_only_proposals_pending_independent_audit",
        "task": "code-review",
        "objective": (
            "Find source-faithful relation-local seeds absent from the historical "
            "33-program bank for the corrected 40-cell static-fidelity gap population."
        ),
        "sources": {
            str(path.relative_to(ROOT)): {"sha256": _sha256(path)}
            for path in (FIDELITY, CROSS_AUDIT, SEED_MAP, CORRECTED_FUNNEL)
        },
        "sealed_inputs": {
            "task_items_loaded": False,
            "candidate_programs_imported_or_executed": False,
            "program_outputs_loaded": False,
            "references_loaded": False,
            "outcomes_loaded": False,
            "prompt_responses_loaded": False,
            "models_or_apis_called": False,
            "gpu_used": False,
            "external_supervision_used": False,
        },
        "library_inventory": {
            "n_metric_modules": len(programs),
            "n_executable_modules": sum(program.executable for program in programs),
            "n_declared_tier_ge_2_executable": len(deep_executable),
            "historical_selected_bank_unique_programs": len(old_bank),
            "unused_declared_tier_ge_2_executable": len(unused_deep),
            "existing_manual_programs_allowed_as_retrospective_seeds": True,
        },
        "summary": {
            "n_corrected_gaps": len(output_rows),
            "corrected_gap_by_level": dict(sorted(Counter(
                row["level"] for row in output_rows
            ).items())),
            "decision_counts": dict(sorted(decisions.items())),
            "n_proposed_partial_mappings": len(proposed),
            "n_unique_proposed_programs": len({
                row["candidate"]["aspect_id"] for row in proposed
            }),
            "proposed_by_level": dict(sorted(proposed_by_level.items())),
            "proposed_matched_depth_counts": dict(sorted(Counter(
                str(row["proposed_matched_relation_depth"]) for row in proposed
            ).items())),
            "projected_static_upper_bound_if_all_survive": projected,
        },
        "claim_limits": [
            "A proposal is not a certified mapping and does not modify the canonical 50/90 corrected static result.",
            "Independent source adjudication is required before any execution or static-fidelity update.",
            "Later train-only operational measurement and heldout readiness gates remain required.",
            "Near matches, Tier-0 catalog entries, and bounded non-discoveries contribute zero proposed static coverage.",
            "This inventory does not measure codability, reconstruction, prompt articulability, isomorphism, whole-construct verifiability, or tacitness.",
        ],
        "rows": output_rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    payload = build_inventory()
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.out.is_file() or args.out.read_text(encoding="utf-8") != serialized:
            raise InventoryError(f"checked artifact differs from source inventory: {args.out}")
        print(f"PASS {args.out}")
        return
    args.out.write_text(serialized, encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
