"""Run the additive press-release hierarchy relation lane on exact ``ctext``.

The lane has two deliberately separate products:

* a static, exhaustive source map for all 90 press-release R1/R2/R3 cells;
* item-local relation witnesses from parser/graph/arithmetic programs.

Compiler-train execution writes a freeze receipt before any heldout item is
loaded.  Heldout execution verifies the source map, implementation, panel, and
train artifact against that receipt *before* opening the heldout JSON.  The
news-homepage task may reuse the frozen relation program as a secondary source-
family check, but does not receive a hierarchy-cell map of its own.

No command reads references, outcomes, prompt outputs, external truth, URLs, or
corpus state.  No result is a prompt-articulability, reconstruction,
isomorphism, codability, or whole-criterion measurement.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

from methods.metric_seam.press_release_relations_v1 import (
    PROGRAM_ID,
    RELATION_SPECS,
    SCHEMA as RELATION_SCHEMA,
    analyze_press_release_ctext,
    implementation_dependencies,
)


SCHEMA = "metric-seam.hierarchy-press-release-local-relations-execution.v1"
SOURCE_MAP_SCHEMA = "metric-seam.hierarchy-press-release-static-source-map.v1"
FREEZE_SCHEMA = "metric-seam.hierarchy-press-release-train-freeze.v1"
TASK = "press-releases"
SECONDARY_TASK = "news-homepages"
TASKS = {TASK, SECONDARY_TASK}
PHASES = {"compiler_train", "heldout_pre_reference"}
LEVELS = ("R1", "R2", "R3")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL = REPO_ROOT / "outputs/metric_seam_pilot/hierarchy_r123/panel_v3.json"
DEFAULT_ITEMS_ROOT = REPO_ROOT / "outputs/metric_seam_pilot/hierarchy_r123/items_v2"


class PressReleaseHierarchyError(ValueError):
    """Raised when the sealed text-only hierarchy contract is violated."""


# Manually adjudicated against panel_v3 construct and description text only.
# Keys are native level/source-kind/source-index identities, not fuzzy name matches.
# Empty tuples are bounded non-discoveries, not evidence that the construct is tacit.
CELL_RELATIONS: dict[str, tuple[str, ...]] = {
    # R1
    "R1:merged_tree:53": (
        "claim_evidence_alignment",
        "attribution_scoped_claim_language",
    ),
    "R1:merged_tree:146": ("commitment_action_binding",),
    "R1:merged_tree:115": ("commitment_action_binding",),
    "R1:parented_tree:78": (),
    "R1:parented_tree:286": (
        "event_logistics_binding",
        "date_quantity_internal_consistency",
    ),
    "R1:parented_tree:695": (
        "quote_integration_structure",
        "attribution_claim_binding",
    ),
    "R1:merged_tree:38": (),
    "R1:merged_tree:325": (
        "entity_evidence_graph",
        "claim_evidence_alignment",
        "date_quantity_internal_consistency",
    ),
    "R1:merged_tree:302": (),
    "R1:parented_tree:154": (
        "uncertainty_claim_scope_binding",
        "claim_evidence_alignment",
        "date_quantity_internal_consistency",
        "attribution_scoped_claim_language",
    ),
    "R1:parented_tree:272": ("url_role_clause_binding",),
    "R1:parented_tree:318": (),
    "R1:merged_tree:9": (),
    "R1:merged_tree:218": (),
    "R1:merged_tree:386": (),
    "R1:parented_tree:92": (
        "attribution_claim_binding",
        "url_role_clause_binding",
        "claim_evidence_alignment",
    ),
    "R1:parented_tree:667": (),
    "R1:parented_tree:550": (
        "url_role_clause_binding",
        "claim_evidence_alignment",
        "entity_evidence_graph",
    ),
    "R1:merged_tree:539": (),
    "R1:merged_tree:270": (
        "attribution_scoped_claim_language",
        "claim_evidence_alignment",
    ),
    "R1:merged_tree:371": (),
    "R1:parented_tree:62": (),
    "R1:parented_tree:581": (
        "cta_resource_binding",
        "significance_comparison_binding",
    ),
    "R1:parented_tree:707": (),
    "R1:merged_tree:75": (),
    "R1:merged_tree:142": ("sentence_dependency_readability",),
    "R1:merged_tree:515": (
        "date_quantity_internal_consistency",
        "significance_comparison_binding",
    ),
    "R1:parented_tree:165": (
        "boilerplate_contact_structure",
        "url_role_clause_binding",
    ),
    "R1:parented_tree:635": (
        "event_logistics_binding",
        "opening_information_graph_alignment",
        "date_quantity_internal_consistency",
        "significance_comparison_binding",
        "opening_locality_binding",
    ),
    "R1:parented_tree:64": (
        "attribution_claim_binding",
        "url_role_clause_binding",
    ),
    # R2
    "R2:grandparent:6": ("uncertainty_claim_scope_binding",),
    "R2:grandparent:51": (
        "date_quantity_internal_consistency",
        "entity_evidence_graph",
    ),
    "R2:grandparent:61": (),
    "R2:merged_group:32": (),
    "R2:merged_group:73": (),
    "R2:merged_group:95": ("sentence_dependency_readability",),
    "R2:grandparent:22": ("commitment_action_binding",),
    "R2:grandparent:67": (
        "uncertainty_claim_scope_binding",
        "claim_evidence_alignment",
        "url_role_clause_binding",
        "date_quantity_internal_consistency",
    ),
    "R2:grandparent:52": (),
    "R2:merged_group:208": (),
    "R2:merged_group:84": ("url_role_clause_binding",),
    "R2:merged_group:127": ("event_logistics_binding",),
    "R2:grandparent:4": (),
    "R2:grandparent:100": (),
    "R2:grandparent:69": (
        "event_logistics_binding",
        "date_quantity_internal_consistency",
    ),
    "R2:merged_group:138": (
        "opening_information_graph_alignment",
        "event_logistics_binding",
    ),
    "R2:merged_group:136": (
        "boilerplate_contact_structure",
        "url_role_clause_binding",
        "section_scannability_structure",
        "cta_resource_binding",
    ),
    "R2:merged_group:23": (
        "date_quantity_internal_consistency",
        "url_role_clause_binding",
        "claim_evidence_alignment",
        "commitment_action_binding",
    ),
    "R2:grandparent:0": (
        "attribution_claim_binding",
        "entity_evidence_graph",
        "claim_evidence_alignment",
        "url_role_clause_binding",
        "uncertainty_claim_scope_binding",
        "date_quantity_internal_consistency",
        "commitment_action_binding",
    ),
    "R2:grandparent:50": (),
    "R2:grandparent:99": (
        "quote_integration_structure",
        "attribution_claim_binding",
    ),
    "R2:merged_group:15": (),
    "R2:merged_group:38": (
        "attribution_claim_binding",
        "quote_integration_structure",
    ),
    "R2:merged_group:210": ("opening_locality_binding",),
    "R2:grandparent:14": (
        "boilerplate_contact_structure",
        "url_role_clause_binding",
        "attribution_claim_binding",
        "quote_integration_structure",
    ),
    "R2:grandparent:32": (
        "date_quantity_internal_consistency",
        "entity_evidence_graph",
        "attribution_claim_binding",
        "opening_information_graph_alignment",
    ),
    "R2:grandparent:98": (),
    "R2:merged_group:31": (
        "attribution_scoped_claim_language",
        "claim_evidence_alignment",
    ),
    "R2:merged_group:43": ("cta_resource_binding",),
    "R2:merged_group:68": ("uncertainty_claim_scope_binding",),
    # R3
    "R3:grandparent:3": (
        "sentence_dependency_readability",
        "section_scannability_structure",
    ),
    "R3:grandparent:22": (),
    "R3:grandparent:13": (
        "sentence_dependency_readability",
        "claim_evidence_alignment",
        "uncertainty_claim_scope_binding",
        "url_role_clause_binding",
    ),
    "R3:merged_group:1": (
        "attribution_claim_binding",
        "entity_evidence_graph",
        "claim_evidence_alignment",
        "url_role_clause_binding",
        "uncertainty_claim_scope_binding",
        "date_quantity_internal_consistency",
    ),
    "R3:merged_group:10": (
        "sentence_dependency_readability",
        "section_scannability_structure",
        "opening_information_graph_alignment",
    ),
    "R3:merged_group:29": (
        "boilerplate_contact_structure",
        "url_role_clause_binding",
    ),
    "R3:grandparent:11": (
        "cta_resource_binding",
        "significance_comparison_binding",
    ),
    "R3:grandparent:16": (
        "url_role_clause_binding",
        "boilerplate_contact_structure",
    ),
    "R3:grandparent:21": (
        "section_scannability_structure",
        "sentence_dependency_readability",
    ),
    "R3:merged_group:0": (
        "claim_evidence_alignment",
        "attribution_scoped_claim_language",
        "attribution_claim_binding",
        "commitment_action_binding",
    ),
    "R3:merged_group:28": ("cta_resource_binding",),
    "R3:merged_group:16": (),
    "R3:grandparent:15": (
        "attribution_claim_binding",
        "quote_integration_structure",
        "entity_evidence_graph",
        "date_quantity_internal_consistency",
    ),
    "R3:grandparent:20": ("opening_information_graph_alignment",),
    "R3:grandparent:14": (
        "event_logistics_binding",
        "date_quantity_internal_consistency",
        "entity_evidence_graph",
        "opening_information_graph_alignment",
        "opening_locality_binding",
    ),
    "R3:merged_group:12": (
        "event_logistics_binding",
        "date_quantity_internal_consistency",
        "opening_information_graph_alignment",
        "significance_comparison_binding",
        "opening_locality_binding",
    ),
    "R3:merged_group:14": (
        "boilerplate_contact_structure",
        "url_role_clause_binding",
        "section_scannability_structure",
        "quote_integration_structure",
    ),
    "R3:merged_group:11": ("section_scannability_structure",),
    "R3:grandparent:10": (),
    "R3:grandparent:17": (
        "url_role_clause_binding",
        "boilerplate_contact_structure",
    ),
    "R3:grandparent:19": (
        "significance_comparison_binding",
        "entity_evidence_graph",
        "date_quantity_internal_consistency",
        "claim_evidence_alignment",
    ),
    "R3:merged_group:26": (),
    "R3:merged_group:8": ("sentence_dependency_readability",),
    "R3:merged_group:2": (
        "attribution_scoped_claim_language",
        "claim_evidence_alignment",
    ),
    "R3:grandparent:5": (),
    "R3:grandparent:12": (
        "claim_evidence_alignment",
        "entity_evidence_graph",
        "url_role_clause_binding",
        "attribution_scoped_claim_language",
    ),
    "R3:grandparent:18": (),
    "R3:merged_group:5": (
        "attribution_claim_binding",
        "quote_integration_structure",
    ),
    "R3:merged_group:36": (),
    "R3:merged_group:27": ("significance_comparison_binding",),
}


DIRECT_RELATION_PAIRS = {
    ("R1:merged_tree:146", "commitment_action_binding"),
    ("R1:parented_tree:695", "quote_integration_structure"),
    ("R1:merged_tree:325", "entity_evidence_graph"),
    ("R1:parented_tree:92", "attribution_claim_binding"),
    ("R1:parented_tree:550", "url_role_clause_binding"),
    ("R1:parented_tree:581", "cta_resource_binding"),
    ("R1:merged_tree:142", "sentence_dependency_readability"),
    ("R1:parented_tree:165", "boilerplate_contact_structure"),
    ("R1:parented_tree:64", "attribution_claim_binding"),
    ("R2:grandparent:6", "uncertainty_claim_scope_binding"),
    ("R2:merged_group:95", "sentence_dependency_readability"),
    ("R2:grandparent:67", "uncertainty_claim_scope_binding"),
    ("R2:merged_group:127", "event_logistics_binding"),
    ("R2:grandparent:69", "event_logistics_binding"),
    ("R2:merged_group:138", "opening_information_graph_alignment"),
    ("R2:grandparent:99", "quote_integration_structure"),
    ("R2:merged_group:38", "attribution_claim_binding"),
    ("R2:grandparent:22", "commitment_action_binding"),
    ("R2:merged_group:23", "commitment_action_binding"),
    ("R2:merged_group:210", "opening_locality_binding"),
    ("R2:merged_group:68", "uncertainty_claim_scope_binding"),
    ("R3:grandparent:3", "sentence_dependency_readability"),
    ("R3:merged_group:1", "attribution_claim_binding"),
    ("R3:merged_group:29", "boilerplate_contact_structure"),
    ("R3:merged_group:28", "cta_resource_binding"),
    ("R3:grandparent:20", "opening_information_graph_alignment"),
    ("R3:merged_group:11", "section_scannability_structure"),
    ("R3:merged_group:8", "sentence_dependency_readability"),
    ("R3:merged_group:5", "attribution_claim_binding"),
    ("R3:merged_group:27", "significance_comparison_binding"),
}


# Optional cell-local gates narrow a general relation program to the exact requested
# subrelation.  They are declarative source-map contracts, not post-hoc item tuning;
# this lane does not aggregate them into criterion scores or prevalence estimates.
CELL_RELATION_FILTERS: dict[tuple[str, str], str] = {
    ("R1:merged_tree:146", "commitment_action_binding"): (
        "witness.action_class in {'remedial_commitment','public_reporting_commitment'}"
    ),
    ("R1:merged_tree:115", "commitment_action_binding"): (
        "witness.action_class == 'public_reporting_commitment'"
    ),
    ("R1:parented_tree:272", "url_role_clause_binding"): (
        "witness.bound_role == 'social'"
    ),
    ("R1:parented_tree:550", "url_role_clause_binding"): (
        "witness.bound_role == 'source'"
    ),
    ("R1:merged_tree:515", "date_quantity_internal_consistency"): (
        "summary.rederived_arithmetic_relations > 0"
    ),
    ("R1:merged_tree:515", "significance_comparison_binding"): (
        "witness contains a bound quantity comparison"
    ),
    ("R1:parented_tree:64", "url_role_clause_binding"): (
        "witness.bound_role == 'source'"
    ),
    ("R2:grandparent:22", "commitment_action_binding"): (
        "witness.action_class == 'remedial_commitment'"
    ),
    ("R2:merged_group:84", "url_role_clause_binding"): (
        "witness.bound_role in {'action','organization','social'}"
    ),
    ("R2:merged_group:23", "commitment_action_binding"): (
        "witness.action_class == 'public_reporting_commitment'"
    ),
    ("R2:grandparent:0", "url_role_clause_binding"): (
        "witness.bound_role == 'source'"
    ),
    ("R2:grandparent:0", "commitment_action_binding"): (
        "witness.action_class == 'public_reporting_commitment'"
    ),
    ("R2:grandparent:14", "url_role_clause_binding"): (
        "witness.bound_role in {'asset','contact','organization'}"
    ),
    ("R3:merged_group:1", "url_role_clause_binding"): (
        "witness.bound_role == 'source'"
    ),
    ("R3:merged_group:29", "url_role_clause_binding"): (
        "witness.bound_role == 'organization'"
    ),
    ("R3:grandparent:16", "url_role_clause_binding"): (
        "witness.bound_role == 'organization'"
    ),
    ("R3:merged_group:0", "commitment_action_binding"): (
        "witness.action_class in {'remedial_commitment','public_reporting_commitment'}"
    ),
    ("R3:merged_group:14", "url_role_clause_binding"): (
        "witness.bound_role in {'asset','contact','organization'}"
    ),
    ("R3:grandparent:17", "url_role_clause_binding"): (
        "witness.bound_role == 'asset'"
    ),
    ("R3:grandparent:12", "url_role_clause_binding"): (
        "witness.bound_role == 'source'"
    ),
}


# Machine-executable counterparts to the human-readable contracts above.  Keeping
# the two ledgers separate makes the source map readable while ensuring execution
# cannot silently substitute a broader relation-level count for a cell-local gate.
CELL_RELATION_FILTER_CODES: dict[tuple[str, str], str] = {
    ("R1:merged_tree:146", "commitment_action_binding"): "commitment_remedial_or_reporting",
    ("R1:merged_tree:115", "commitment_action_binding"): "commitment_public_reporting",
    ("R1:parented_tree:272", "url_role_clause_binding"): "url_social",
    ("R1:parented_tree:550", "url_role_clause_binding"): "url_source",
    ("R1:merged_tree:515", "date_quantity_internal_consistency"): "date_arithmetic",
    ("R1:merged_tree:515", "significance_comparison_binding"): "significance_quantity",
    ("R1:parented_tree:64", "url_role_clause_binding"): "url_source",
    ("R2:grandparent:22", "commitment_action_binding"): "commitment_remedial",
    ("R2:merged_group:84", "url_role_clause_binding"): "url_action_org_social",
    ("R2:merged_group:23", "commitment_action_binding"): "commitment_public_reporting",
    ("R2:grandparent:0", "url_role_clause_binding"): "url_source",
    ("R2:grandparent:0", "commitment_action_binding"): "commitment_public_reporting",
    ("R2:grandparent:14", "url_role_clause_binding"): "url_asset_contact_org",
    ("R3:merged_group:1", "url_role_clause_binding"): "url_source",
    ("R3:merged_group:29", "url_role_clause_binding"): "url_org",
    ("R3:grandparent:16", "url_role_clause_binding"): "url_org",
    ("R3:merged_group:0", "commitment_action_binding"): "commitment_remedial_or_reporting",
    ("R3:merged_group:14", "url_role_clause_binding"): "url_asset_contact_org",
    ("R3:grandparent:17", "url_role_clause_binding"): "url_asset",
    ("R3:grandparent:12", "url_role_clause_binding"): "url_source",
}


NO_CANDIDATE_REASONS: dict[str, str] = {
    "R1:parented_tree:78": "respect, representation, and accessibility require semantic or missing-format evidence",
    "R1:merged_tree:38": "impartiality and partisan balance require viewpoint/source adjudication",
    "R1:merged_tree:302": "cultural and historical sensitivity requires semantic/community context",
    "R1:parented_tree:318": "distribution cadence, hosting, translation, and access metadata are absent",
    "R1:merged_tree:9": "the normative free-flow/public-decision relation is not locally executable",
    "R1:merged_tree:218": "target audience and channel fit are not identified in ctext",
    "R1:merged_tree:386": "outlet/journalist target context is absent",
    "R1:parented_tree:667": "repurposing and downstream distribution actions are absent from ctext",
    "R1:merged_tree:539": "search ranking and metadata are absent from ctext",
    "R1:merged_tree:371": "cultural offensiveness requires semantic/community context",
    "R1:parented_tree:62": "mission/action authenticity requires organizational ground context",
    "R1:parented_tree:707": "coverage pickup and causal predictors require external outcomes",
    "R1:merged_tree:75": (
        "issue time, event-relative response latency, and update cadence are not available"
    ),
    "R2:grandparent:61": "device, multimedia, region, and performance observations are absent",
    "R2:merged_group:32": "neutrality and public-interest orientation are semantic judgments",
    "R2:merged_group:73": "empathy and tone-deafness are semantic judgments",
    "R2:grandparent:52": "copyright, licensing, data protection, and plagiarism require external evidence",
    "R2:merged_group:208": "outlet, beat, and reporter target context is absent",
    "R2:grandparent:4": "values/actions authenticity requires external organizational context",
    "R2:grandparent:100": "the source items are not a sealed pitch-email representation",
    "R2:grandparent:50": "exclusivity, embargo, event staging, and pickup odds require process metadata",
    "R2:merged_group:15": "message-to-action authenticity requires external organizational context",
    "R2:grandparent:98": "local norms, translation quality, and cultural sensitivity are semantic/contextual",
    "R3:grandparent:22": "visual artifacts are absent from the exact ctext projection",
    "R3:merged_group:16": "routing, timing, and cross-channel versions are absent",
    "R3:grandparent:10": "cross-channel messages and real actions are absent",
    "R3:merged_group:26": "priority-public identity and tailoring target are absent",
    "R3:grandparent:5": "outlet/reporter target, routing, and algorithmic distribution state are absent",
    "R3:grandparent:18": "segment and outlet identities needed for customization are absent",
    "R3:merged_group:36": "neutrality and public-interest tone are semantic judgments",
}


RELATION_SEEDS: dict[str, tuple[str, ...]] = {
    "attribution_claim_binding": (
        "methods/metric_seam/hybrids/programs_v2/a2_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a31_h0.py",
        "methods/metric_seam/hybrids/ops_capability.py",
    ),
    "quote_integration_structure": (
        "methods/metric_seam/hybrids/programs_v2/a87_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a2_h0.py",
    ),
    "entity_evidence_graph": (
        "methods/metric_seam/hybrids/programs_v2/a28_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a66_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a97_h0.py",
        "methods/metric_seam/hybrids/ops_capability.py",
    ),
    "claim_evidence_alignment": (
        "methods/metric_seam/hybrids/programs_v2/a25_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a28_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a66_h0.py",
    ),
    "date_quantity_internal_consistency": (
        "methods/metric_seam/hybrids/programs_v2/a115_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a97_h0.py",
        "methods/metric_seam/hybrids/ops_capability.py",
    ),
    "url_role_clause_binding": (
        "methods/metric_seam/hybrids/programs_v2/a2_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a41_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a81_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a111_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a112_h0.py",
    ),
    "opening_information_graph_alignment": (
        "methods/metric_seam/hybrids/programs_v2/a75_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a76_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a103_h0.py",
    ),
    "sentence_dependency_readability": (
        "methods/metric_seam/hybrids/programs_v2/a103_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a104_h0.py",
    ),
    "cta_resource_binding": (
        "methods/metric_seam/hybrids/programs_v2/a42_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a64_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a111_h0.py",
    ),
    "boilerplate_contact_structure": (
        "methods/metric_seam/hybrids/programs_v2/a41_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a81_h0.py",
    ),
    "section_scannability_structure": (
        "methods/metric_seam/hybrids/programs_v2/a104_h0.py",
    ),
    "event_logistics_binding": (
        "methods/metric_seam/hybrids/programs_v2/a75_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a76_h0.py",
    ),
    "attribution_scoped_claim_language": (
        "methods/metric_seam/hybrids/programs_v2/a25_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a31_h0.py",
        "methods/metric_seam/hybrids/ops_capability.py",
    ),
    "significance_comparison_binding": (
        "methods/metric_seam/hybrids/programs_v2/a64_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a65_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a115_h0.py",
    ),
    "uncertainty_claim_scope_binding": (
        "methods/metric_seam/hybrids/programs_v2/a25_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a31_h0.py",
    ),
    "commitment_action_binding": (
        "methods/metric_seam/hybrids/programs_v2/a119_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a97_h0.py",
    ),
    "opening_locality_binding": (
        "methods/metric_seam/hybrids/programs_v2/a65_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a75_h0.py",
        "methods/metric_seam/hybrids/programs_v2/a76_h0.py",
    ),
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    value = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(value)


def _serialized_cell_relation_filters() -> dict[str, str]:
    return {
        f"{cell_key}|{relation_id}": value
        for (cell_key, relation_id), value in sorted(CELL_RELATION_FILTERS.items())
    }


def _serialized_cell_relation_filter_codes() -> dict[str, str]:
    return {
        f"{cell_key}|{relation_id}": value
        for (cell_key, relation_id), value in sorted(CELL_RELATION_FILTER_CODES.items())
    }


def _cell_key(cell: Mapping[str, Any]) -> str:
    return f"{cell['level']}:{cell['source_kind']}:{cell['source_index']}"


def _validate_panel(panel: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if panel.get("schema") != "tacit_breadth_metric_panel/v1":
        raise PressReleaseHierarchyError("unexpected hierarchy panel schema")
    cells = [cell for cell in panel.get("cells", []) if cell.get("task") == TASK]
    if len(cells) != 90:
        raise PressReleaseHierarchyError("press-release panel must contain 90 cells")
    if Counter(cell.get("level") for cell in cells) != {level: 30 for level in LEVELS}:
        raise PressReleaseHierarchyError("press-release panel must contain 30 cells per level")
    keys = [_cell_key(cell) for cell in cells]
    if len(set(keys)) != 90:
        raise PressReleaseHierarchyError("press-release panel identities are not unique")
    if set(keys) != set(CELL_RELATIONS):
        missing = sorted(set(keys) - set(CELL_RELATIONS))
        extra = sorted(set(CELL_RELATIONS) - set(keys))
        raise PressReleaseHierarchyError(
            f"static source map identity drift; missing={missing}, extra={extra}"
        )
    empty_keys = {key for key, relations in CELL_RELATIONS.items() if not relations}
    if set(NO_CANDIDATE_REASONS) != empty_keys:
        raise PressReleaseHierarchyError("bounded non-discovery reasons are not exhaustive")
    for key, relation_ids in CELL_RELATIONS.items():
        if len(relation_ids) != len(set(relation_ids)):
            raise PressReleaseHierarchyError(f"duplicate relation mapping for {key}")
        unknown = set(relation_ids) - set(RELATION_SPECS)
        if unknown:
            raise PressReleaseHierarchyError(f"unknown relations for {key}: {unknown}")
    invalid_filters = [
        pair
        for pair in CELL_RELATION_FILTERS
        if pair[0] not in CELL_RELATIONS or pair[1] not in CELL_RELATIONS[pair[0]]
    ]
    if invalid_filters:
        raise PressReleaseHierarchyError(
            f"cell-local filters do not map to selected relations: {invalid_filters}"
        )
    if set(CELL_RELATION_FILTER_CODES) != set(CELL_RELATION_FILTERS):
        raise PressReleaseHierarchyError(
            "cell-local executable filter codes and descriptions do not have identical keys"
        )
    return cells


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if len(node.targets) == 1 else None
        )
        if isinstance(target, ast.Name) and target.id == name:
            try:
                return ast.literal_eval(node.value)
            except (TypeError, ValueError):
                return None
    return None


def _inspect_seed(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    fields = _literal_assignment(tree, "LLM_FIELDS")
    regex_names = []
    invoked_ops = set()
    functions = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.add(node.name)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if any(
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == "re"
                    and child.func.attr == "compile"
                    for child in ast.walk(node.value)
                ):
                    regex_names.append(target.id)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"ops", "_cap"}
        ):
            invoked_ops.add(node.func.attr)
    return {
        "path": _relative(path),
        "sha256": _sha256_path(path),
        "ast_node_count": sum(1 for _node in ast.walk(tree)),
        "module_docstring": ast.get_docstring(tree),
        "score_function_present": "score" in functions,
        "llm_field_names": sorted(fields) if isinstance(fields, dict) else [],
        "regex_constant_names": sorted(regex_names),
        "invoked_ops": sorted(invoked_ops),
        "imported_or_executed": False,
    }


def build_source_map(panel: Mapping[str, Any], *, panel_path: Path) -> dict[str, Any]:
    cells = _validate_panel(panel)
    seed_paths = sorted({path for paths in RELATION_SEEDS.values() for path in paths})
    seeds = {
        path: _inspect_seed(REPO_ROOT / path)
        for path in seed_paths
    }
    rows = []
    for cell in cells:
        key = _cell_key(cell)
        relation_ids = CELL_RELATIONS[key]
        candidates = []
        for relation_id in relation_ids:
            spec = RELATION_SPECS[relation_id]
            candidates.append(
                {
                    "relation_id": relation_id,
                    "subrelation_match": (
                        "direct_named_subrelation"
                        if (key, relation_id) in DIRECT_RELATION_PAIRS
                        else "adjacent_partial_subrelation"
                    ),
                    "implemented_relation": spec["implemented_relation"],
                    "program_relation_depth_ceiling": spec["matched_depth"],
                    "matched_relation_depth": None,
                    "matched_depth_status": "requires_item_local_runtime_witness",
                    "cell_local_applicability_filter": {
                        "code": CELL_RELATION_FILTER_CODES.get(
                            (key, relation_id), "positive_relation"
                        ),
                        "description": CELL_RELATION_FILTERS.get(
                            (key, relation_id),
                            "any positive local relation witness under the relation contract",
                        ),
                    },
                    "depth_meaning": spec["depth_meaning"],
                    "does_not_establish": spec["does_not_establish"],
                    "historical_seed_paths": list(RELATION_SEEDS[relation_id]),
                }
            )
        rows.append(
            {
                "cell_id": cell["id"],
                "static_cell_key": key,
                "task": TASK,
                "level": cell["level"],
                "source_kind": cell["source_kind"],
                "source_index": cell["source_index"],
                "requested_construct": cell["construct"],
                "requested_description": cell["description"],
                "requested_source_text_sha256": _sha256_bytes(
                    (cell["construct"] + "\0" + cell["description"]).encode("utf-8")
                ),
                "decision": (
                    "relation_local_candidate"
                    if candidates
                    else "bounded_non_discovery_in_frozen_program_class"
                ),
                "implemented_candidates": candidates,
                "whole_construct_match_established": False,
                "bounded_non_discovery_reason": (
                    None if candidates else NO_CANDIDATE_REASONS[key]
                ),
                "unimplemented_scope": (
                    "Only the explicitly listed relation-local candidates are implemented; "
                    "every other component of the requested construct remains unimplemented."
                    if candidates
                    else "No relation-local candidate was selected from this frozen program class."
                ),
            }
        )

    decision_counts = Counter(row["decision"] for row in rows)
    candidate_counts = Counter(
        candidate["relation_id"]
        for row in rows
        for candidate in row["implemented_candidates"]
    )
    direct_count = sum(
        candidate["subrelation_match"] == "direct_named_subrelation"
        for row in rows
        for candidate in row["implemented_candidates"]
    )
    return {
        "schema": SOURCE_MAP_SCHEMA,
        "task": TASK,
        "status": "static_requested_vs_implemented_adjudication",
        "panel": {
            "path": _relative(panel_path),
            "sha256": _sha256_path(panel_path),
            "panel_content_sha256": panel["panel_content_sha256"],
        },
        "design_scope": {
            "source_text_used": ["construct", "description", "native cell identity"],
            "items_loaded": False,
            "heldout_text_loaded": False,
            "references_or_outcomes_loaded": False,
            "prompt_outputs_loaded": False,
            "programs_imported_or_executed": False,
            "historical_program_sources_inspected_by_ast": True,
            "selection_origin": (
                "manual retrospective relation adjudication using existing h0 programs and "
                "capability ops as disclosed seeds"
            ),
        },
        "depth_contract": {
            "D2": "parser-backed structural measurement",
            "D3": (
                "positive graph, within-document retrieval, dependency-binding, or arithmetic relation"
            ),
            "depth_applies_to": "implemented relation only, never the whole requested criterion",
            "static_source_map_depth": (
                "program ceiling only; matched_relation_depth remains null until an item-local "
                "runtime witness realizes D2 or D3"
            ),
        },
        "interpretation": {
            "permitted": (
                "exhaustive static inventory of requested cells, selected local relations, "
                "matched depths, and bounded non-discoveries"
            ),
            "forbidden": [
                "prompt articulability",
                "reconstruction",
                "isomorphism",
                "codability",
                "whole-criterion verifiability",
                "tacitness from a negative result",
            ],
        },
        "summary": {
            "n_cells": len(rows),
            "cells_by_level": dict(sorted(Counter(row["level"] for row in rows).items())),
            "decision_counts": dict(sorted(decision_counts.items())),
            "candidate_applications": sum(candidate_counts.values()),
            "unique_relation_programs": len(candidate_counts),
            "candidate_applications_by_relation": dict(sorted(candidate_counts.items())),
            "direct_named_subrelation_applications": direct_count,
            "adjacent_partial_subrelation_applications": sum(candidate_counts.values())
            - direct_count,
            "whole_construct_matches_established": 0,
        },
        "relation_programs": {
            relation_id: {
                **RELATION_SPECS[relation_id],
                "historical_seed_paths": list(RELATION_SEEDS[relation_id]),
            }
            for relation_id in sorted(RELATION_SPECS)
        },
        "historical_seed_source_inspection": seeds,
        "rows": rows,
    }


def validate_items(items: Sequence[Mapping[str, Any]], *, phase: str) -> None:
    if phase not in PHASES:
        raise PressReleaseHierarchyError(f"unsupported phase: {phase}")
    if not isinstance(items, list) or not items:
        raise PressReleaseHierarchyError("items must be a nonempty JSON list")
    prefix = "train_" if phase == "compiler_train" else "heldout_"
    seen = set()
    for index, row in enumerate(items):
        if not isinstance(row, Mapping) or set(row) != {"item_key", "ctext"}:
            raise PressReleaseHierarchyError(
                f"item {index} must expose exactly item_key and ctext"
            )
        item_key = row["item_key"]
        if not isinstance(item_key, str) or not item_key.startswith(prefix):
            raise PressReleaseHierarchyError(f"item {index} has invalid opaque split key")
        if item_key in seen:
            raise PressReleaseHierarchyError(f"duplicate item key: {item_key}")
        seen.add(item_key)
        if not isinstance(row["ctext"], str) or not row["ctext"].strip():
            raise PressReleaseHierarchyError(f"item {index} has invalid ctext")


def validate_manifest(
    manifest: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    *,
    task: str,
    phase: str,
) -> int:
    if manifest.get("schema") != "metric-seam.hierarchy-shared-items.v1":
        raise PressReleaseHierarchyError("unexpected shared-item manifest schema")
    if manifest.get("task") != task:
        raise PressReleaseHierarchyError("shared-item manifest task mismatch")
    representation = manifest.get("representation", {})
    if (
        representation.get("field") != "ctext"
        or representation.get("same_bytes_required_for_prompt_and_code") is not True
    ):
        raise PressReleaseHierarchyError("manifest does not require exact shared ctext")
    max_chars = representation.get("max_chars")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise PressReleaseHierarchyError("manifest max_chars is invalid")
    policy = manifest.get("policy", {})
    if (
        policy.get("outcome_columns_emitted") is not False
        or policy.get("external_supervision_used") is not False
    ):
        raise PressReleaseHierarchyError("manifest violates outcome-blind policy")
    expected_count = (
        manifest.get("selection", {}).get("train_n")
        if phase == "compiler_train"
        else manifest.get("selection", {}).get("heldout_n")
    )
    if expected_count != len(items):
        raise PressReleaseHierarchyError("manifest split count mismatch")
    if any(len(row["ctext"]) > max_chars for row in items):
        raise PressReleaseHierarchyError("item exceeds declared character cap")
    return max_chars


def execute_split(
    items: Sequence[Mapping[str, Any]],
    *,
    task: str,
    phase: str,
    max_chars: int,
    source_map_sha256: str,
    freeze_sha256: str | None = None,
) -> dict[str, Any]:
    validate_items(items, phase=phase)
    rows = []
    failures = Counter()
    relation_statuses: dict[str, Counter[str]] = {
        relation_id: Counter() for relation_id in RELATION_SPECS
    }
    relation_witnesses = Counter()
    realized_depths: dict[str, Counter[str]] = {
        relation_id: Counter() for relation_id in RELATION_SPECS
    }
    for item in items:
        text = item["ctext"]
        at_cap = len(text) == max_chars
        try:
            result = analyze_press_release_ctext(text)
        except Exception as error:  # fail-closed receipt on unvetted text
            failures[type(error).__name__] += 1
            rows.append(
                {
                    "item_key": item["item_key"],
                    "ctext_sha256": _sha256_bytes(text.encode("utf-8")),
                    "ctext_chars": len(text),
                    "at_declared_character_cap": at_cap,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "result": None,
                }
            )
            continue
        for relation_id, relation in result["relations"].items():
            relation_statuses[relation_id][relation["status"]] += 1
            relation_witnesses[relation_id] += int(relation["witness_count"])
            realized_depths[relation_id][str(relation["realized_depth"])] += 1
        rows.append(
            {
                "item_key": item["item_key"],
                "ctext_sha256": _sha256_bytes(text.encode("utf-8")),
                "ctext_utf8_bytes": len(text.encode("utf-8")),
                "ctext_chars": len(text),
                "at_declared_character_cap": at_cap,
                "possibly_truncated_by_declared_projection": at_cap,
                "absence_inference_permitted": False,
                "status": "measured_with_possible_truncation" if at_cap else "measured",
                "error_type": None,
                "result": result,
            }
        )
    status_counts = Counter(row["status"] for row in rows)
    return {
        "schema": SCHEMA,
        "relation_program_schema": RELATION_SCHEMA,
        "program_id": PROGRAM_ID,
        "task": task,
        "source_family_role": (
            "primary_press_release_hierarchy_lane"
            if task == TASK
            else "secondary_news_homepage_source_family_check"
        ),
        "phase": phase,
        "status": "executed_local_relations_no_criterion_verdicts",
        "chronology": {
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
            "heldout_text_loaded": phase == "heldout_pre_reference",
            "freeze_verified_before_heldout_open": phase == "heldout_pre_reference",
            "references_loaded_after_execution": False,
        },
        "bindings": {
            "source_map_sha256": source_map_sha256,
            "train_freeze_sha256": freeze_sha256,
        },
        "design": {
            "input_fields": ["item_key", "ctext"],
            "exact_ctext_sha256_recorded": True,
            "outcomes_or_references_loaded": False,
            "prompt_outputs_loaded": False,
            "external_truth_loaded": False,
            "corpus_state_or_retrieval_loaded": False,
            "network_or_url_resolution_used": False,
            "api_or_llm_calls_used": False,
            "local_cpu_statistical_parser_used": True,
            "gpu_used": False,
            "local_cpu_spacy_pipeline_used": True,
            "criterion_scalar_scores_emitted": False,
            "whole_criterion_verdicts_emitted": False,
            "absence_certificates_emitted": False,
            "declared_character_cap": max_chars,
        },
        "interpretation": {
            "permitted": (
                "local relation witnesses and operation-status frequencies on the exact "
                "presented ctext; not construct prevalence"
            ),
            "forbidden": [
                "prompt articulability",
                "reconstruction",
                "isomorphism",
                "codability",
                "whole-criterion verifiability",
                "negative-result evidence of tacitness",
            ],
        },
        "dependencies": implementation_dependencies(),
        "summary": {
            "n_items": len(rows),
            "status_counts": dict(sorted(status_counts.items())),
            "failure_types": dict(sorted(failures.items())),
            "items_at_declared_character_cap": sum(
                bool(row.get("at_declared_character_cap")) for row in rows
            ),
            "relation_status_counts": {
                relation_id: dict(sorted(counts.items()))
                for relation_id, counts in sorted(relation_statuses.items())
            },
            "relation_witness_counts": dict(sorted(relation_witnesses.items())),
            "realized_depth_counts": {
                relation_id: dict(sorted(counts.items()))
                for relation_id, counts in sorted(realized_depths.items())
            },
        },
        "rows": rows,
    }


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _implementation_bindings() -> dict[str, dict[str, Any]]:
    paths = [
        Path(__file__).resolve(),
        REPO_ROOT / "methods/metric_seam/press_release_relations_v1.py",
    ]
    return {
        _relative(path): {"bytes": path.stat().st_size, "sha256": _sha256_path(path)}
        for path in paths
    }


def build_freeze_receipt(
    *,
    source_map_path: Path,
    train_output_path: Path,
    panel_path: Path,
) -> dict[str, Any]:
    return {
        "schema": FREEZE_SCHEMA,
        "task": TASK,
        "status": "frozen_after_compiler_train_before_any_heldout_load",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "chronology": {
            "compiler_train_executed": True,
            "press_release_heldout_text_loaded": False,
            "news_homepage_heldout_text_loaded": False,
            "references_or_outcomes_loaded": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "dependencies": implementation_dependencies(),
            "cpu_only_required": True,
        },
        "implementation": _implementation_bindings(),
        "source_map": {
            "path": _relative(source_map_path),
            "bytes": source_map_path.stat().st_size,
            "sha256": _sha256_path(source_map_path),
        },
        "compiler_train_artifact": {
            "path": _relative(train_output_path),
            "bytes": train_output_path.stat().st_size,
            "sha256": _sha256_path(train_output_path),
        },
        "panel": {
            "path": _relative(panel_path),
            "bytes": panel_path.stat().st_size,
            "sha256": _sha256_path(panel_path),
        },
        "mapping_contract_sha256": _canonical_sha(
            {
                "cell_relations": CELL_RELATIONS,
                "direct_pairs": sorted(DIRECT_RELATION_PAIRS),
                "non_discovery_reasons": NO_CANDIDATE_REASONS,
                "relation_seeds": RELATION_SEEDS,
                "cell_relation_filters": _serialized_cell_relation_filters(),
            }
        ),
    }


def verify_freeze(
    freeze: Mapping[str, Any],
    *,
    freeze_path: Path,
    source_map_path: Path,
    panel_path: Path,
) -> str:
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("task") != TASK:
        raise PressReleaseHierarchyError("unexpected train freeze receipt")
    if freeze.get("status") != "frozen_after_compiler_train_before_any_heldout_load":
        raise PressReleaseHierarchyError("train freeze status is invalid")
    chronology = freeze.get("chronology", {})
    if (
        chronology.get("compiler_train_executed") is not True
        or chronology.get("press_release_heldout_text_loaded") is not False
        or chronology.get("news_homepage_heldout_text_loaded") is not False
        or chronology.get("references_or_outcomes_loaded") is not False
    ):
        raise PressReleaseHierarchyError("train freeze chronology is invalid")
    expected_implementation = freeze.get("implementation")
    if expected_implementation != _implementation_bindings():
        raise PressReleaseHierarchyError("implementation drifted after train freeze")
    runtime = freeze.get("runtime", {})
    if (
        runtime.get("python") != platform.python_version()
        or runtime.get("dependencies") != implementation_dependencies()
        or runtime.get("cpu_only_required") is not True
    ):
        raise PressReleaseHierarchyError("runtime drifted after train freeze")
    source_binding = freeze.get("source_map", {})
    if (
        source_binding.get("path") != _relative(source_map_path)
        or source_binding.get("sha256") != _sha256_path(source_map_path)
    ):
        raise PressReleaseHierarchyError("source map drifted after train freeze")
    panel_binding = freeze.get("panel", {})
    if (
        panel_binding.get("path") != _relative(panel_path)
        or panel_binding.get("sha256") != _sha256_path(panel_path)
    ):
        raise PressReleaseHierarchyError("panel drifted after train freeze")
    train_binding = freeze.get("compiler_train_artifact", {})
    train_path = REPO_ROOT / train_binding.get("path", "")
    if not train_path.is_file() or train_binding.get("sha256") != _sha256_path(train_path):
        raise PressReleaseHierarchyError("compiler-train artifact drifted after freeze")
    expected_mapping_sha = _canonical_sha(
        {
            "cell_relations": CELL_RELATIONS,
            "direct_pairs": sorted(DIRECT_RELATION_PAIRS),
            "non_discovery_reasons": NO_CANDIDATE_REASONS,
            "relation_seeds": RELATION_SEEDS,
            "cell_relation_filters": _serialized_cell_relation_filters(),
        }
    )
    if freeze.get("mapping_contract_sha256") != expected_mapping_sha:
        raise PressReleaseHierarchyError("mapping contract drifted after train freeze")
    return _sha256_path(freeze_path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args(argv)

    panel = _load_json(args.panel)
    source_map_sha: str
    freeze_sha: str | None = None
    if args.task == TASK and args.phase == "compiler_train":
        if args.freeze is None:
            raise PressReleaseHierarchyError("compiler-train requires --freeze output")
        source_map = build_source_map(panel, panel_path=args.panel)
        _write_new(args.source_map, source_map)
        source_map_sha = _sha256_path(args.source_map)
    else:
        if args.freeze is None:
            raise PressReleaseHierarchyError("frozen execution requires --freeze")
        # This check is intentionally complete before items or manifest are opened.
        freeze = _load_json(args.freeze)
        freeze_sha = verify_freeze(
            freeze,
            freeze_path=args.freeze,
            source_map_path=args.source_map,
            panel_path=args.panel,
        )
        source_map_sha = _sha256_path(args.source_map)

    # For heldout, this is the first line that opens the sealed items file.
    items = _load_json(args.items)
    manifest = _load_json(args.manifest)
    validate_items(items, phase=args.phase)
    max_chars = validate_manifest(
        manifest,
        items,
        task=args.task,
        phase=args.phase,
    )
    result = execute_split(
        items,
        task=args.task,
        phase=args.phase,
        max_chars=max_chars,
        source_map_sha256=source_map_sha,
        freeze_sha256=freeze_sha,
    )
    result["input_artifacts"] = {
        "items": {
            "path": _relative(args.items),
            "bytes": args.items.stat().st_size,
            "sha256": _sha256_path(args.items),
            "opened_after_freeze_verification": args.phase == "heldout_pre_reference",
        },
        "manifest": {
            "path": _relative(args.manifest),
            "bytes": args.manifest.stat().st_size,
            "sha256": _sha256_path(args.manifest),
            "opened_after_freeze_verification": args.phase == "heldout_pre_reference",
        },
    }
    _write_new(args.output, result)
    if args.task == TASK and args.phase == "compiler_train":
        freeze = build_freeze_receipt(
            source_map_path=args.source_map,
            train_output_path=args.output,
            panel_path=args.panel,
        )
        _write_new(args.freeze, freeze)
        freeze_sha = _sha256_path(args.freeze)
    print(
        json.dumps(
            {
                "task": args.task,
                "phase": args.phase,
                "output": str(args.output),
                "source_map_sha256": source_map_sha,
                "freeze_sha256": freeze_sha,
                **result["summary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
