"""Build a denominator-safe ledger for the current technical metric-seam evidence.

The ledger is deliberately a *typed union*, not a pooled scorecard.  It keeps
three empirically different objects separate:

``criterion_scalar_reconstruction``
    Agreement between an executable score and an articulated prompt/LLM score.
``relation_instance_verification``
    Executable witnesses or abstentions for a named sub-relation.
``program_structure_descriptor``
    Structure of an implementation under a named, local measurement scale.

In particular, a relation witness yield is not a criterion-level codability
rate, and an evidence-graph edge count is not control-flow depth.  The builder
fails closed on missing inputs and validates every fraction's denominator.
It only reads existing artifacts and runs no model, API, or GPU workload.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "metric-seam.technical-evidence-ledger.v1"
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = (
    ROOT / "outputs/metric_seam_pilot/reconstruction_v2/technical_evidence_ledger_v1"
)

STRATA = {
    "criterion_scalar_reconstruction",
    "relation_instance_verification",
    "program_structure_descriptor",
}
READOUT_STATUSES = {"observed", "unavailable", "unopened", "not_run"}
NULL_STATUSES = {"unavailable", "unopened", "not_run"}


class LedgerError(RuntimeError):
    """Raised when an input or ledger invariant is violated."""


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise LedgerError(f"required artifact is missing: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LedgerError(f"source receipt path is missing: {path.relative_to(ROOT)}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _readout(
    metric: str,
    estimate: int | float | bool | None,
    *,
    status: str = "observed",
    numerator: int | None = None,
    denominator: int | None = None,
    support_n: int | None = None,
    conditioning: str,
    inference_status: str = "descriptive",
    recomputable: bool = True,
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "metric": metric,
        "estimate": estimate,
        "status": status,
        "numerator": numerator,
        "denominator": denominator,
        "support_n": support_n,
        "conditioning": conditioning,
        "inference_status": inference_status,
        "recomputable": recomputable,
        "note": note,
    }


def _selection(
    mode: str,
    timing: str,
    *,
    representative: bool,
    note: str,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "timing": timing,
        "representative_of_domain": representative,
        "note": note,
    }


def _units(
    kind: str,
    *,
    population_n: int | None = None,
    train_n: int | None = None,
    heldout_n: int | None = None,
    eligible_n: int | None = None,
    candidate_n: int | None = None,
    reference_n: int | None = None,
    common_n: int | None = None,
    conditioning: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "population_n": population_n,
        "train_n": train_n,
        "heldout_n": heldout_n,
        "eligible_n": eligible_n,
        "candidate_n": candidate_n,
        "reference_n": reference_n,
        "common_n": common_n,
        "conditioning": conditioning,
    }


def _permissions(**overrides: bool) -> dict[str, bool]:
    permissions = {
        "may_claim_prompt_articulability": False,
        "may_claim_code_verifiability": False,
        "may_claim_descriptive_reconstruction": False,
        "may_claim_confirmatory_isomorphism": False,
        "may_claim_constructive_extension": False,
        "may_claim_domain_codability": False,
        "may_claim_tacitness": False,
    }
    unknown = set(overrides) - set(permissions)
    if unknown:
        raise LedgerError(f"unknown claim permission(s): {sorted(unknown)}")
    permissions.update(overrides)
    return permissions


def _base_record(
    *,
    record_id: str,
    stratum: str,
    domain: str,
    criterion_id: str | None,
    relation_id: str | None,
    selection: Mapping[str, Any],
    units: Mapping[str, Any],
    channels: Mapping[str, Any],
    readouts: Mapping[str, Any],
    fidelity: Mapping[str, Any],
    claim_permissions: Mapping[str, Any],
    sources: Iterable[Mapping[str, Any]],
    claim_boundary: str,
    nonrecomputable_claims: Iterable[str] = (),
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "stratum": stratum,
        "domain": domain,
        "criterion_id": criterion_id,
        "relation_id": relation_id,
        "selection": dict(selection),
        "units": dict(units),
        "channels": dict(channels),
        "readouts": dict(readouts),
        "fidelity": dict(fidelity),
        "claim_permissions": dict(claim_permissions),
        "claim_boundary": claim_boundary,
        "sources": list(sources),
        "nonrecomputable_claims": list(nonrecomputable_claims),
    }


def _math_records() -> list[dict[str, Any]]:
    base = ROOT / "outputs/metric_seam_pilot/reconstruction_v2"
    a144_dir = base / "blind_math_a144_001"
    a144_record_path = a144_dir / "reconstruction_record.json"
    a144_metrics_path = a144_dir / "sealed_eval_002/metrics.json"
    a144_adversary_path = a144_dir / "adversary_001/RESULTS.json"
    a144_record = _read_json(a144_record_path)
    a144_metrics = _read_json(a144_metrics_path)
    a144_adversary = _read_json(a144_adversary_path)

    candidate = a144_metrics["candidate"]
    reference = a144_metrics["reference"]
    if a144_record["outcome"] != "proxy_mismatch":
        raise LedgerError("math a144 no longer has the audited proxy_mismatch outcome")

    records = [
        _base_record(
            record_id="math.a144.blind_scalar",
            stratum="criterion_scalar_reconstruction",
            domain="math",
            criterion_id="a144",
            relation_id=None,
            selection=_selection(
                "blind_agentic",
                "candidate frozen before held-out reference access",
                representative=False,
                note="One selected criterion; not a random sample of math metrics.",
            ),
            units=_units(
                "criterion_x_item_scalar",
                heldout_n=int(candidate["heldout_count"]),
                candidate_n=int(candidate["n_scoreable"]),
                reference_n=int(candidate["reference_available_count"]),
                common_n=int(candidate["common_support_n"]),
                conditioning="Held-out items with both candidate code score and frozen two-pass prompt/LLM reference.",
            ),
            channels={
                "candidate": "code",
                "reference": "prompt_llm_two_pass",
                "external_supervised_anchor": None,
            },
            readouts={
                "reconstruction_spearman": _readout(
                    "spearman",
                    float(candidate["spearman_reconstruction"]),
                    support_n=int(candidate["common_support_n"]),
                    conditioning="Common held-out support only.",
                    inference_status="descriptive_blind_heldout",
                ),
                "reference_repeatability": _readout(
                    "spearman",
                    float(reference["two_pass_spearman_reliability"]),
                    support_n=int(reference["n_two_pass"]),
                    conditioning="Items with both prompt/LLM passes.",
                    inference_status="instrument_reliability",
                ),
                "candidate_coverage": _readout(
                    "fraction",
                    float(candidate["candidate_coverage_all_heldout"]),
                    numerator=int(candidate["n_scoreable"]),
                    denominator=int(candidate["heldout_count"]),
                    conditioning="All held-out items.",
                ),
            },
            fidelity={
                "construct_fidelity": "fail",
                "program_fidelity": "candidate_not_equivalent_to_reference_instrument",
                "input_fidelity": "pass_on_common_support",
                "outcome": "proxy_mismatch",
            },
            claim_permissions=_permissions(may_claim_descriptive_reconstruction=True),
            sources=(
                _receipt(a144_record_path),
                _receipt(a144_metrics_path),
                _receipt(a144_adversary_path),
            ),
            claim_boundary="A blind held-out reconstruction estimate for one criterion; construct failure pre-empts any isomorphism or code-verifiability claim.",
        ),
        _base_record(
            record_id="math.a144.construct_adversary",
            stratum="relation_instance_verification",
            domain="math",
            criterion_id="a144",
            relation_id="explicit_witness_verification_and_scope",
            selection=_selection(
                "frozen_construct_adversary",
                "suite frozen before execution",
                representative=False,
                note="Authored counterexamples test known construct boundaries; they are not corpus prevalence estimates.",
            ),
            units=_units(
                "adversarial_relation_case",
                population_n=int(a144_adversary["summary"]["range_total"]),
                eligible_n=int(a144_adversary["summary"]["ordering_total"]),
                conditioning="Frozen authored adversary suite.",
            ),
            channels={
                "candidate": "code",
                "reference": "frozen_expected_relation_behavior",
            },
            readouts={
                "ordering_pass_rate": _readout(
                    "fraction",
                    a144_adversary["summary"]["ordering_passes"]
                    / a144_adversary["summary"]["ordering_total"],
                    numerator=int(a144_adversary["summary"]["ordering_passes"]),
                    denominator=int(a144_adversary["summary"]["ordering_total"]),
                    conditioning="Paired adversary orderings.",
                ),
                "expected_range_pass_rate": _readout(
                    "fraction",
                    a144_adversary["summary"]["range_passes"]
                    / a144_adversary["summary"]["range_total"],
                    numerator=int(a144_adversary["summary"]["range_passes"]),
                    denominator=int(a144_adversary["summary"]["range_total"]),
                    conditioning="Individual adversary score ranges.",
                ),
                "suite_pass": _readout(
                    "boolean",
                    bool(a144_adversary["decision"] == "ACCEPT"),
                    conditioning="All frozen suite gates, including category floors.",
                ),
            },
            fidelity={"construct_fidelity": "fail", "outcome": "proxy_mismatch"},
            claim_permissions=_permissions(),
            sources=(_receipt(a144_adversary_path),),
            claim_boundary="Failure is bounded non-discovery in this program class and budget, not evidence of tacitness.",
        ),
    ]

    a216_dir = base / "blind_math_a216_001"
    a216_record_path = a216_dir / "reconstruction_record.json"
    a216_adversary_path = a216_dir / "adversary_001/result.json"
    a216 = _read_json(a216_record_path)
    _read_json(a216_adversary_path)
    summary = a216["adversary_summary"]
    if a216["heldout_reference_opened"] is not False:
        raise LedgerError("math a216 held-out reference access status changed")
    records.append(
        _base_record(
            record_id="math.a216.construct_adversary",
            stratum="relation_instance_verification",
            domain="math",
            criterion_id="a216",
            relation_id="equation_reference_graph_and_semantic_target",
            selection=_selection(
                "blind_agentic",
                "construct gate run before held-out reference access",
                representative=False,
                note="One selected criterion; held-out scalar reference remains unopened.",
            ),
            units=_units(
                "adversarial_relation_case",
                population_n=int(summary["pair_cases"] + summary["range_cases"]),
                eligible_n=int(summary["pair_cases"]),
                conditioning="Frozen authored pair and range adversary cases.",
            ),
            channels={
                "candidate": "code",
                "reference": "frozen_expected_relation_behavior",
                "heldout_prompt_llm_reference": "unopened",
            },
            readouts={
                "pair_pass_rate": _readout(
                    "fraction",
                    float(summary["pair_pass_rate"]),
                    numerator=round(summary["pair_pass_rate"] * summary["pair_cases"]),
                    denominator=int(summary["pair_cases"]),
                    conditioning="All paired adversary cases; aggregate pass rate does not override category floors.",
                ),
                "range_pass_rate": _readout(
                    "fraction",
                    float(summary["range_pass_rate"]),
                    numerator=round(
                        summary["range_pass_rate"] * summary["range_cases"]
                    ),
                    denominator=int(summary["range_cases"]),
                    conditioning="All adversary expected-range cases.",
                ),
                "minimum_category_pass_rate": _readout(
                    "fraction",
                    float(summary["minimum_pair_category_pass_rate"]),
                    numerator=0,
                    denominator=1,
                    conditioning="Frozen category-floor minimum; semantic-target category failed 0/1.",
                ),
                "heldout_reconstruction_spearman": _readout(
                    "spearman",
                    None,
                    status="unopened",
                    conditioning="Held-out reference deliberately unopened after construct failure.",
                ),
            },
            fidelity={"construct_fidelity": "fail", "outcome": "proxy_mismatch"},
            claim_permissions=_permissions(),
            sources=(_receipt(a216_record_path), _receipt(a216_adversary_path)),
            claim_boundary="Construct failure pre-empted scalar evaluation; an unopened estimate is null, not zero.",
        )
    )

    a12_dir = base / "math_a12_symbolic_step_retrospective_prepare_001"
    a12_summary_path = a12_dir / "train_symbolic_step_summary.json"
    a12_manifest_path = a12_dir / "prepare_manifest.json"
    a12_compiler_path = a12_dir / "compiler_bundle.json"
    a12_source_path = (
        ROOT
        / "methods/metric_seam/pilot/prepare_math_a12_symbolic_step_retrospective_v1.py"
    )
    a12 = _read_json(a12_summary_path)
    coverage = a12["coverage"]
    if not a12["train_only"] or a12["heldout_accessed"]:
        raise LedgerError("math a12 is no longer the audited TRAIN-only preparation")
    records.append(
        _base_record(
            record_id="math.a12.train_symbolic_step",
            stratum="relation_instance_verification",
            domain="math",
            criterion_id="a12",
            relation_id="exact_rational_identity_or_nonidentity_between_extracted_steps",
            selection=_selection(
                "retrospective_seed",
                "TRAIN-only procedural preparation",
                representative=False,
                note="Historical rationale and one aggregate TRAIN statistic were visible; no held-out or per-item reference was accessed.",
            ),
            units=_units(
                "extracted_equation_pair",
                train_n=int(a12["train_row_count"]),
                eligible_n=int(coverage["parsed_rational_pair_count"]),
                conditioning="TRAIN rows and parsed rational equation pairs only.",
            ),
            channels={
                "candidate": "code_symbolic_rational_arithmetic",
                "criterion_reference": "unavailable",
                "heldout_reference": "not_accessed",
            },
            readouts={
                "rows_with_executable_pair": _readout(
                    "fraction",
                    coverage["rows_with_executable_pair"] / coverage["rows"],
                    numerator=int(coverage["rows_with_executable_pair"]),
                    denominator=int(coverage["rows"]),
                    conditioning="All TRAIN rows; a row may contain multiple parsed pairs.",
                ),
                "parsed_pair_positive_witness_rate": _readout(
                    "fraction",
                    coverage["positive_code_witness_count"]
                    / coverage["parsed_rational_pair_count"],
                    numerator=int(coverage["positive_code_witness_count"]),
                    denominator=int(coverage["parsed_rational_pair_count"]),
                    conditioning="Parsed rational pairs; identity and exact nonidentity witnesses are disjoint pair outcomes.",
                ),
                "verified_identity_count": _readout(
                    "count",
                    int(coverage["verified_rational_identity_count"]),
                    conditioning="Parsed rational pairs on TRAIN.",
                ),
                "exact_nonidentity_count": _readout(
                    "count",
                    int(coverage["exact_nonidentity_witness_count"]),
                    conditioning="Parsed rational pairs on TRAIN; not automatically criterion defects.",
                ),
                "whole_criterion_scalar": _readout(
                    "spearman",
                    None,
                    status="unavailable",
                    conditioning="No whole-criterion reference was accessed or scored.",
                ),
            },
            fidelity={
                "relation_fidelity": "parser_scoped",
                "whole_criterion_fidelity": "unavailable",
                "universal_scope_determination": "not_implemented",
            },
            claim_permissions=_permissions(may_claim_code_verifiability=True),
            sources=(
                _receipt(a12_summary_path),
                _receipt(a12_manifest_path),
                _receipt(a12_compiler_path),
                _receipt(a12_source_path),
            ),
            claim_boundary="TRAIN relation-local exact arithmetic witnesses only. Nonidentity is not a criterion defect without a separately frozen universal-scope rule.",
        )
    )

    a12_heldout_dir = base / "math_a12_symbolic_step_heldout_001"
    a12_final_path = a12_heldout_dir / "finalization/finalization.json"
    a12_execution_path = a12_heldout_dir / "execution/candidate_execution.json"
    a12_execution_manifest_path = a12_heldout_dir / "execution/execution_manifest.json"
    a12_final_report_path = a12_heldout_dir / "finalization/REPORT.md"
    a12_heldout = _read_json(a12_final_path)
    a12_execution = _read_json(a12_execution_path)
    heldout_summary = a12_heldout["candidate_execution_summary"]
    if (
        a12_heldout["schema"] != "metric-seam.math-a12-symbolic-heldout-finalization.v1"
        or a12_heldout["candidate_parent_scalar"] is not None
        or a12_heldout["candidate_reference_correlation"] is not None
        or a12_heldout["isomorphism"] != "NOT_ESTIMATED"
        or a12_execution["reference_accessed"] is not False
        or not a12_heldout["prompt_reference"][
            "candidate_completed_before_reference_load"
        ]
    ):
        raise LedgerError("math a12 held-out claim boundary or seal changed")
    records.append(
        _base_record(
            record_id="math.a12.heldout_symbolic_step",
            stratum="relation_instance_verification",
            domain="math",
            criterion_id="a12",
            relation_id="explicit_rational_equality_preservation",
            selection=_selection(
                "sealed_heldout_execution",
                "candidate completed before stored prompt reference load",
                representative=False,
                note="Held-out relation-instance execution of the frozen TRAIN-prepared program; no parent scalar was defined.",
            ),
            units=_units(
                "extracted_equation_pair",
                heldout_n=int(a12_heldout["heldout_count"]),
                eligible_n=int(heldout_summary["parsed_rational_pair_count"]),
                reference_n=int(
                    a12_heldout["prompt_reference"]["available_both_passes"]
                ),
                conditioning="Seed-7 held-out rows and parsed rational equation pairs; prompt reference is instrument context, not a relation-matched target.",
            ),
            channels={
                "candidate": "code_symbolic_rational_arithmetic",
                "stored_reference": "prompt_llm_two_pass_not_relation_matched",
                "external_supervised_anchor": None,
            },
            readouts={
                "rows_with_executable_pair": _readout(
                    "fraction",
                    heldout_summary["rows_with_executable_pair"]
                    / a12_heldout["heldout_count"],
                    numerator=int(heldout_summary["rows_with_executable_pair"]),
                    denominator=int(a12_heldout["heldout_count"]),
                    conditioning="All held-out rows; a row may contain multiple parsed pairs.",
                ),
                "row_abstention_rate": _readout(
                    "fraction",
                    heldout_summary["rows_abstained"] / a12_heldout["heldout_count"],
                    numerator=int(heldout_summary["rows_abstained"]),
                    denominator=int(a12_heldout["heldout_count"]),
                    conditioning="All held-out rows; abstention is neither negative evidence nor tacitness evidence.",
                ),
                "verified_identity_pair_rate": _readout(
                    "fraction",
                    heldout_summary["verified_rational_identity_count"]
                    / heldout_summary["parsed_rational_pair_count"],
                    numerator=int(heldout_summary["verified_rational_identity_count"]),
                    denominator=int(heldout_summary["parsed_rational_pair_count"]),
                    conditioning="Parsed rational pairs on held-out; exact identity witnesses only.",
                ),
                "exact_nonidentity_pair_rate": _readout(
                    "fraction",
                    heldout_summary["exact_nonidentity_witness_count"]
                    / heldout_summary["parsed_rational_pair_count"],
                    numerator=int(heldout_summary["exact_nonidentity_witness_count"]),
                    denominator=int(heldout_summary["parsed_rational_pair_count"]),
                    conditioning="Parsed rational pairs on held-out; nonidentity is not automatically a criterion defect.",
                ),
                "prompt_reference_availability": _readout(
                    "fraction",
                    a12_heldout["prompt_reference"]["available_both_passes"]
                    / a12_heldout["heldout_count"],
                    numerator=int(
                        a12_heldout["prompt_reference"]["available_both_passes"]
                    ),
                    denominator=int(a12_heldout["heldout_count"]),
                    conditioning="Held-out rows with both stored prompt/LLM passes.",
                ),
                "prompt_reference_repeatability": _readout(
                    "spearman",
                    float(a12_heldout["prompt_reference"]["two_pass_spearman"]),
                    support_n=int(
                        a12_heldout["prompt_reference"]["available_both_passes"]
                    ),
                    conditioning="Stored prompt/LLM reference pass1-versus-pass2 reliability.",
                    inference_status="instrument_reliability",
                ),
                "whole_criterion_reconstruction": _readout(
                    "spearman",
                    None,
                    status="unavailable",
                    conditioning="No parent scalar or candidate/reference correlation was defined.",
                ),
            },
            fidelity={
                "relation_fidelity": "parser_scoped_exact_rational_relation",
                "whole_criterion_fidelity": "unavailable",
                "isomorphism": "not_estimated",
                "candidate_before_reference_load": True,
            },
            claim_permissions=_permissions(may_claim_code_verifiability=True),
            sources=(
                _receipt(a12_final_path),
                _receipt(a12_execution_path),
                _receipt(a12_execution_manifest_path),
                _receipt(a12_final_report_path),
            ),
            claim_boundary="Sealed held-out relation-instance code witnesses only. No parent scalar, correlation, whole-criterion reconstruction, or isomorphism was estimated.",
        )
    )

    projection_dir = a12_heldout_dir / "pair_certificate_projection_replay_001"
    projection_summary_path = projection_dir / "projection_summary.json"
    projection_pairs_path = projection_dir / "pair_certificates.jsonl"
    projection_rows_path = projection_dir / "row_projection.json"
    projection_manifest_path = projection_dir / "projection_manifest.json"
    projection = _read_json(projection_summary_path)
    pair_counts = projection["pair_status_counts"]
    if (
        projection["schema"] != "metric-seam.math-a12-pair-certificate-projection.v1"
        or projection["temporal_status"] != "post_reference_projection_replay"
        or projection["reference_loaded_or_used_by_replay"]
        or projection["new_blind_result"]
        or projection["new_isomorphism_result"]
        or projection["new_reconstruction_result"]
        or not projection["sealed_v1_aggregate_exact"]
        or not projection["sealed_v1_row_classifications_exact"]
        or sum(pair_counts.values()) != projection["pair_certificate_count"]
    ):
        raise LedgerError("math a12 pair projection claim boundary changed")
    records.append(
        _base_record(
            record_id="math.a12.post_reference_pair_projection",
            stratum="relation_instance_verification",
            domain="math",
            criterion_id="a12",
            relation_id="explicit_rational_equality_preservation",
            selection=_selection(
                "post_reference_projection_replay",
                "inspectability projection created after reference access; replay itself did not load the reference",
                representative=False,
                note="Faithful materialization of frozen v1 classifications, not a new blind run or a new empirical result.",
            ),
            units=_units(
                "pair_candidate_projection",
                population_n=int(projection["pair_certificate_count"]),
                heldout_n=int(projection["heldout_count"]),
                eligible_n=int(
                    pair_counts["verified_rational_identity"]
                    + pair_counts["exact_nonidentity_witness"]
                ),
                conditioning="All projected pair candidates emitted from the frozen 100-row held-out execution.",
            ),
            channels={
                "candidate": "post_reference_inspectability_projection_of_frozen_code_output",
                "reference_used_by_replay": None,
                "external_supervised_anchor": None,
            },
            readouts={
                "verified_identity_share_of_pair_candidates": _readout(
                    "fraction",
                    pair_counts["verified_rational_identity"]
                    / projection["pair_certificate_count"],
                    numerator=int(pair_counts["verified_rational_identity"]),
                    denominator=int(projection["pair_certificate_count"]),
                    conditioning="All projected pair candidates; exact identity tier.",
                ),
                "exact_nonidentity_share_of_pair_candidates": _readout(
                    "fraction",
                    pair_counts["exact_nonidentity_witness"]
                    / projection["pair_certificate_count"],
                    numerator=int(pair_counts["exact_nonidentity_witness"]),
                    denominator=int(projection["pair_certificate_count"]),
                    conditioning="All projected pair candidates; nonidentity is not a document defect without universal scope.",
                ),
                "parse_noncoverage_share_of_pair_candidates": _readout(
                    "fraction",
                    pair_counts["parse_noncoverage"]
                    / projection["pair_certificate_count"],
                    numerator=int(pair_counts["parse_noncoverage"]),
                    denominator=int(projection["pair_certificate_count"]),
                    conditioning="All projected pair candidates; noncoverage is abstention, not negative evidence.",
                ),
                "sealed_aggregate_exact": _readout(
                    "boolean",
                    bool(projection["sealed_v1_aggregate_exact"]),
                    conditioning="Projection aggregate compared byte-for-value with the sealed v1 summary.",
                ),
                "sealed_row_classifications_exact": _readout(
                    "boolean",
                    bool(projection["sealed_v1_row_classifications_exact"]),
                    conditioning="Projection row classifications compared with the sealed v1 execution.",
                ),
            },
            fidelity={
                "temporal_status": projection["temporal_status"],
                "representation_policy": projection["representation_policy"],
                "domain_obligation_boundary": projection["domain_obligation_boundary"],
                "new_blind_result": False,
                "new_reconstruction_result": False,
                "new_isomorphism_result": False,
                "embedded_depth_interpretation": "superseded_by_relation_depth_multiview_audit_002",
            },
            claim_permissions=_permissions(may_claim_code_verifiability=True),
            sources=(
                _receipt(projection_summary_path),
                _receipt(projection_pairs_path),
                _receipt(projection_rows_path),
                _receipt(projection_manifest_path),
            ),
            claim_boundary="Inspectable post-reference projection of already-frozen relation-local classifications only; it adds no blind, reconstruction, or isomorphism result.",
        )
    )

    depth_dir = a12_heldout_dir / "relation_depth_multiview_audit_002"
    depth_path = depth_dir / "relation_depth_multiview.json"
    depth_report_path = depth_dir / "AUDIT_REPORT.md"
    depth = _read_json(depth_path)
    attempted = depth["depth_views"]["deepest_attempted"]
    contributing = depth["depth_views"]["deepest_decision_contributing"]
    positive = depth["depth_views"]["positive_relation_evidence"]
    attempted_depth_1 = int(attempted["histogram"]["1"])
    attempted_depth_3 = int(attempted["histogram"]["3"])
    positive_depth_3 = int(positive["histogram"]["3"])
    formal_parse_noncoverage_rows = attempted_depth_3 - positive_depth_3
    if (
        depth["scale"] != "metric-seam.relation-depth.v1"
        or attempted["accounted_rows"] != a12_heldout["heldout_count"]
        or contributing["accounted_rows"] != a12_heldout["heldout_count"]
        or attempted["histogram"] != contributing["histogram"]
        or attempted_depth_1 + attempted_depth_3 != a12_heldout["heldout_count"]
        or positive["evidence_rows"] != heldout_summary["rows_with_executable_pair"]
        or positive["no_positive_evidence_rows"] != heldout_summary["rows_abstained"]
        or formal_parse_noncoverage_rows != 39
        or depth["supersedes_interpretation_of"]["prior_dynamic_histogram"]
        != {"1": 74, "3": 26}
    ):
        raise LedgerError("math a12 relation-depth multi-view audit changed")
    records.append(
        _base_record(
            record_id="math.a12.relation_depth_multiview",
            stratum="program_structure_descriptor",
            domain="math",
            criterion_id="a12",
            relation_id="explicit_rational_equality_preservation",
            selection=_selection(
                "post_reference_multiview_audit",
                "depth interpretation audited after frozen execution and reference access",
                representative=False,
                note="Supersedes only the old 74@depth1/26@depth3 dynamic interpretation; frozen classifications and evidence counts are unchanged.",
            ),
            units=_units(
                "heldout_row_relation_depth_view",
                heldout_n=int(a12_heldout["heldout_count"]),
                population_n=int(a12_heldout["heldout_count"]),
                eligible_n=attempted_depth_3,
                conditioning="Three separately named views over the same 100 held-out rows.",
            ),
            channels={
                "runtime": "code",
                "reference_used_by_audit": None,
                "external_supervised_anchor": None,
            },
            readouts={
                "static_max_relation_depth": _readout(
                    "ordinal_relation_depth",
                    int(depth["static_max_relation_depth"]),
                    conditioning="Maximum operation tier on metric-seam.relation-depth.v1; not edge count.",
                ),
                "longest_path_edges": _readout(
                    "count",
                    int(depth["longest_path_edges"]),
                    conditioning="Dependency edge count between the two declared relation-program nodes; distinct from ordinal relation depth.",
                ),
                "attempted_depth_1_rate": _readout(
                    "fraction",
                    attempted_depth_1 / attempted["accounted_rows"],
                    numerator=attempted_depth_1,
                    denominator=int(attempted["accounted_rows"]),
                    conditioning="Deepest attempted operation: rows stopping at document-structure parsing.",
                ),
                "attempted_depth_3_rate": _readout(
                    "fraction",
                    attempted_depth_3 / attempted["accounted_rows"],
                    numerator=attempted_depth_3,
                    denominator=int(attempted["accounted_rows"]),
                    conditioning="Deepest attempted operation: rows sent to the formal verification path, including parse failures.",
                ),
                "decision_contributing_depth_1_rate": _readout(
                    "fraction",
                    int(contributing["histogram"]["1"])
                    / contributing["accounted_rows"],
                    numerator=int(contributing["histogram"]["1"]),
                    denominator=int(contributing["accounted_rows"]),
                    conditioning="Deepest decision-contributing operation: structure-only abstentions.",
                ),
                "decision_contributing_depth_3_rate": _readout(
                    "fraction",
                    int(contributing["histogram"]["3"])
                    / contributing["accounted_rows"],
                    numerator=int(contributing["histogram"]["3"]),
                    denominator=int(contributing["accounted_rows"]),
                    conditioning="Deepest decision-contributing operation: formal witnesses or formal parse-noncoverage abstentions.",
                ),
                "positive_evidence_depth_3_rate": _readout(
                    "fraction",
                    positive_depth_3 / a12_heldout["heldout_count"],
                    numerator=positive_depth_3,
                    denominator=int(a12_heldout["heldout_count"]),
                    conditioning="Rows with at least one positive depth-3 code witness; this is narrower than attempted depth.",
                ),
                "formal_path_positive_evidence_rate": _readout(
                    "fraction",
                    positive_depth_3 / attempted_depth_3,
                    numerator=positive_depth_3,
                    denominator=attempted_depth_3,
                    conditioning="Rows reaching the formal path; at least one positive relation witness.",
                ),
                "formal_path_parse_noncoverage_rate": _readout(
                    "fraction",
                    formal_parse_noncoverage_rows / attempted_depth_3,
                    numerator=formal_parse_noncoverage_rows,
                    denominator=attempted_depth_3,
                    conditioning="Rows reaching the formal path but producing only parse noncoverage; abstention, not negative evidence.",
                ),
            },
            fidelity={
                "descriptor_kind": "relation_depth_program",
                "scale_id": depth["scale"],
                "view_separation": [
                    "deepest_attempted",
                    "deepest_decision_contributing",
                    "positive_relation_evidence",
                ],
                "supersedes_prior_dynamic_interpretation": True,
                "frozen_classifications_changed": False,
                "whole_criterion_semantic_depth": "not_established",
            },
            claim_permissions=_permissions(),
            sources=(
                _receipt(depth_path),
                _receipt(depth_report_path),
                _receipt(projection_summary_path),
                _receipt(projection_rows_path),
            ),
            claim_boundary="Post-reference multi-view structure audit only. A depth-3 attempt is not a successful solver result, positive evidence, construct fidelity, or reconstruction; positive evidence is reported separately.",
        )
    )

    replay_path = ROOT / "outputs/metric_seam_pilot/technical_replay_v2/results.json"
    replay = _read_json(replay_path)
    a150_case = next(
        row
        for row in replay["cases"]
        if row["case_id"] == "math_a150_sympy_scope_replay"
    )
    measurements = a150_case["axes"]["verifiability"]["measurements"]
    a150_meta_path = (
        ROOT
        / "outputs/metric_seam_pilot/battery/effort_ladder/e2l/math__a150/meta.json"
    )
    a150_adv_path = (
        ROOT
        / "outputs/metric_seam_pilot/battery/effort_ladder/e2l/math__a150/self_adversary_results.json"
    )
    records.append(
        _base_record(
            record_id="math.a150.sympy_scope_replay",
            stratum="relation_instance_verification",
            domain="math",
            criterion_id="a150",
            relation_id="hypothesis_licenses_subsequent_operation",
            selection=_selection(
                "retrospective_seed",
                "adaptive TRAIN replay",
                representative=False,
                note="Historical replay, not blind automatic selection.",
            ),
            units=_units(
                "structurally_located_relation_occurrence",
                train_n=150,
                eligible_n=int(measurements["structural_occurrences"]),
                conditioning="Real TRAIN since/because licensing occurrences after noise guards.",
            ),
            channels={
                "candidate": "code_sympy",
                "criterion_reference": "seen_train_llm",
            },
            readouts={
                "real_sympy_checkable_rate": _readout(
                    "fraction",
                    measurements["sympy_checkable_occurrences"]
                    / measurements["structural_occurrences"],
                    numerator=int(measurements["sympy_checkable_occurrences"]),
                    denominator=int(measurements["structural_occurrences"]),
                    conditioning="Real structurally located TRAIN relation occurrences.",
                ),
                "synthetic_verified_count": _readout(
                    "count",
                    int(
                        a150_case["axes"]["constructive_extension"]["measurements"][
                            "verified_synthetic_consequents"
                        ]
                    ),
                    conditioning="Hand-built synthetic mechanism probes only.",
                ),
                "synthetic_rejected_wrong_count": _readout(
                    "count",
                    int(
                        a150_case["axes"]["constructive_extension"]["measurements"][
                            "rejected_wrong_synthetic_consequents"
                        ]
                    ),
                    conditioning="Hand-built synthetic mechanism probes only.",
                ),
            },
            fidelity={
                "relation_match": "fail_scope_mismatch",
                "mechanism_probe": "clean_on_two_synthetic_cases",
                "corpus_coverage": "zero_of_twenty",
            },
            claim_permissions=_permissions(),
            sources=(
                _receipt(replay_path),
                _receipt(a150_meta_path),
                _receipt(a150_adv_path),
            ),
            claim_boundary="SymPy distinguishes two synthetic equation-consequence probes but covers 0/20 real licensing occurrences; this is a relation mismatch, not a math-domain codability rate.",
        )
    )
    return records


def _active_code_records() -> list[dict[str, Any]]:
    base = ROOT / "outputs/metric_seam_pilot/reconstruction_v2"
    family_path = base / "code_depth_full_panel_retrospective_002/results.json"
    family = _read_json(family_path)
    if family["schema"] != "metric-seam.code-depth-full-panel-retrospective.v2":
        raise LedgerError("active-code depth family is not the corrected v2 artifact")
    family_source = ROOT / "methods/metric_seam/pilot/code_depth_retrospective.py"
    records: list[dict[str, Any]] = []
    heldout_n = int(family["split"]["heldout_count"])

    for row in family["criteria"]:
        criterion_id = str(row["criterion_id"])
        deep = row["deep_only_heldout"]
        comparison = row.get("heldout_comparison")
        reference = row["heldout_reference"]
        if comparison is None:
            shallow_rho = _readout(
                "spearman",
                None,
                status="unavailable",
                conditioning="No TRAIN-selected shallow executable comparator was available.",
            )
            delta = _readout(
                "delta_spearman",
                None,
                status="unavailable",
                conditioning="No paired deep-versus-shallow comparison.",
            )
            p_value = _readout(
                "paired_randomization_p_value",
                None,
                status="unavailable",
                conditioning="No inferential comparison.",
            )
            q_value = _readout(
                "benjamini_hochberg_q_value",
                None,
                status="unavailable",
                conditioning="No inferential comparison.",
            )
            common_n = int(deep["n"])
        else:
            common_n = int(comparison["n_paired"])
            shallow_rho = _readout(
                "spearman",
                comparison["rho_shallow"],
                support_n=common_n,
                conditioning="Paired held-out support shared by deep code, shallow code, and prompt/LLM reference.",
                inference_status="descriptive_retrospective",
            )
            delta = _readout(
                "delta_spearman",
                comparison["delta_spearman"],
                support_n=common_n,
                conditioning="Deep code minus TRAIN-selected shallow code on paired held-out support.",
                inference_status=(
                    "multiplicity_controlled_family_test"
                    if comparison["inferential_eligible"]
                    else "descriptive_ineligible"
                ),
            )
            if comparison["inferential_eligible"]:
                p_value = _readout(
                    "paired_randomization_p_value",
                    comparison["paired_randomization"]["p_value"],
                    support_n=common_n,
                    conditioning="One-sided paired swap test for deep correlation greater than shallow.",
                    inference_status="inferential_retrospective",
                )
                q_value = _readout(
                    "benjamini_hochberg_q_value",
                    comparison["bh_q_value"],
                    support_n=common_n,
                    conditioning="BH family of four eligible active criteria.",
                    inference_status="multiplicity_controlled_family_test",
                )
            else:
                p_value = _readout(
                    "paired_randomization_p_value",
                    None,
                    status="unavailable",
                    conditioning="Comparison failed frozen support/coverage/variance eligibility.",
                )
                q_value = _readout(
                    "benjamini_hochberg_q_value",
                    None,
                    status="unavailable",
                    conditioning="Ineligible comparisons are not members of the BH family.",
                )

        records.append(
            _base_record(
                record_id=f"code.active_depth.{criterion_id}",
                stratum="criterion_scalar_reconstruction",
                domain="code_review",
                criterion_id=criterion_id,
                relation_id=None,
                selection=_selection(
                    "active_panel_retrospective",
                    "deep program pre-existing; shallow comparator selected on TRAIN",
                    representative=True,
                    note="Complete active 18-criterion coding panel, but the analysis is retrospective rather than preregistered prospectively.",
                ),
                units=_units(
                    "criterion_x_item_scalar",
                    population_n=18,
                    train_n=int(family["split"]["train_count"]),
                    heldout_n=heldout_n,
                    reference_n=int(reference["count"]),
                    common_n=common_n,
                    conditioning="Frozen held-out split; support varies by criterion and program coverage.",
                ),
                channels={
                    "candidate": "deep_executable_code",
                    "comparator": (
                        "train_selected_shallow_executable_code"
                        if comparison is not None
                        else "unavailable"
                    ),
                    "reference": "prompt_llm_two_pass",
                    "important_guard": "This is code-versus-code authoring depth, not prompt-versus-code.",
                },
                readouts={
                    "deep_reconstruction_spearman": _readout(
                        "spearman",
                        deep["rho"],
                        support_n=int(deep["n"]),
                        conditioning="Held-out support shared by deep code and two-pass prompt/LLM reference.",
                        inference_status="descriptive_retrospective",
                    ),
                    "shallow_reconstruction_spearman": shallow_rho,
                    "deep_minus_shallow_delta": delta,
                    "paired_randomization_p_value": p_value,
                    "bh_q_value": q_value,
                    "reference_repeatability": _readout(
                        "spearman",
                        reference["pass1_pass2_spearman"],
                        support_n=int(reference["reliability_n"]),
                        conditioning="Held-out items with both prompt/LLM passes.",
                        inference_status="instrument_reliability",
                    ),
                    "deep_coverage": _readout(
                        "fraction",
                        float(deep["coverage_over_heldout"]),
                        numerator=round(
                            float(deep["coverage_over_heldout"]) * heldout_n
                        ),
                        denominator=heldout_n,
                        conditioning="All held-out items; this is candidate coverage, while rho support additionally requires the reference.",
                    ),
                },
                fidelity={
                    "construct_fidelity": "not_independently_established",
                    "program_comparison": "deep_manual_code_vs_shallow_prompt_generated_code",
                    "status": row["status"],
                },
                claim_permissions=_permissions(
                    may_claim_descriptive_reconstruction=True
                ),
                sources=(_receipt(family_path), _receipt(family_source)),
                claim_boundary="Per-criterion reconstruction and a bounded full-family deep-vs-shallow comparison; not a fraction of all code metrics that are codable.",
            )
        )

    a407_eval_path = base / "a407_sealed_historical_eval_001/evaluation.json"
    a407_coverage_path = (
        base / "a407_dual_channel_prepare_002_clean/code_coverage_summary.json"
    )
    a407_prompt_path = (
        base / "a407_matched_prompt_prepare_003_blind/preparation_manifest.json"
    )
    a407_source = (
        ROOT / "methods/metric_seam/pilot/evaluate_code_review_a407_historical_v1.py"
    )
    a407 = _read_json(a407_eval_path)
    coverage = _read_json(a407_coverage_path)
    prompt = _read_json(a407_prompt_path)
    primary = a407["primary_code_vs_historical_composite"]
    records.append(
        _base_record(
            record_id="code.a407.structural_partial_historical",
            stratum="criterion_scalar_reconstruction",
            domain="code_review",
            criterion_id="a407",
            relation_id=None,
            selection=_selection(
                "retrospective_seed",
                "historical reference opened after candidate preparation",
                representative=False,
                note="Focused partial-aggregate study; separate from the active-panel a407 deep program.",
            ),
            units=_units(
                "criterion_x_item_scalar",
                heldout_n=int(coverage["heldout_count"]),
                eligible_n=int(primary["eligible_exact_input_count"]),
                reference_n=int(primary["historical_reference_available_count"]),
                candidate_n=int(primary["code_covered_count"]),
                common_n=int(primary["available_pair_count"]),
                conditioning="99 exact-input rows; nullable code coverage and historical reference common support.",
            ),
            channels={
                "candidate": "code_structural_partial_aggregate",
                "reference": "historical_prompt_llm_two_pass_holistic",
                "matched_raw_prompt_arm": "prepared_not_run",
                "matched_hybrid_prompt_arm": "prepared_not_run",
            },
            readouts={
                "partial_reconstruction_spearman": _readout(
                    "spearman",
                    float(primary["spearman"]),
                    support_n=int(primary["available_pair_count"]),
                    conditioning="Common support for the structural partial aggregate and historical holistic reference.",
                    inference_status="descriptive_only",
                ),
                "partial_code_coverage": _readout(
                    "fraction",
                    primary["code_covered_count"]
                    / primary["eligible_exact_input_count"],
                    numerator=int(primary["code_covered_count"]),
                    denominator=int(primary["eligible_exact_input_count"]),
                    conditioning="Exact-input primary rows only; neutral no-declaration rows are noncoverage.",
                ),
                "matched_raw_prompt_reconstruction": _readout(
                    "spearman",
                    None,
                    status="not_run",
                    conditioning=f"{prompt['raw_fact_null_count']} matched requests prepared; zero executed.",
                ),
                "matched_hybrid_reconstruction": _readout(
                    "spearman",
                    None,
                    status="not_run",
                    conditioning=f"{prompt['hybrid_fact_present_count']} matched requests prepared; zero executed.",
                ),
            },
            fidelity={
                "construct_fidelity": "unavailable",
                "program_fidelity": "fail_missing_semantic_context_fit",
                "whole_criterion_fidelity": "unavailable",
                "outcome": "descriptive_reconstruction_only",
            },
            claim_permissions=_permissions(may_claim_descriptive_reconstruction=True),
            sources=(
                _receipt(a407_eval_path),
                _receipt(a407_coverage_path),
                _receipt(a407_prompt_path),
                _receipt(a407_source),
            ),
            claim_boundary="Descriptive partial reconstruction only; neither whole-criterion isomorphism nor prompt-channel effects are estimable.",
        )
    )

    events = a407["coverage_and_noncoverage"]["event_witness_policy"]["all_100"]
    all_rows = a407["coverage_and_noncoverage"]["all_100_rows"]
    records.append(
        _base_record(
            record_id="code.a407.relation_witnesses",
            stratum="relation_instance_verification",
            domain="code_review",
            criterion_id="a407",
            relation_id="identifier_declaration_scope_use_placeholder_collision_relations",
            selection=_selection(
                "retrospective_seed",
                "held-out structural analysis",
                representative=False,
                note="One focused criterion; positive and negative witnesses have different eligibility rules.",
            ),
            units=_units(
                "code_diff_row",
                heldout_n=int(coverage["heldout_count"]),
                eligible_n=int(coverage["declaration_covered_count"]),
                conditioning="All 100 held-out rows for positive events; strict-complete subset for negative absence claims.",
            ),
            channels={
                "candidate": "code_ast_scope_graph",
                "reference": "relation_local_structural_checks",
            },
            readouts={
                "declaration_coverage": _readout(
                    "fraction",
                    coverage["declaration_covered_count"] / coverage["heldout_count"],
                    numerator=int(coverage["declaration_covered_count"]),
                    denominator=int(coverage["heldout_count"]),
                    conditioning="All held-out rows.",
                ),
                "strict_complete_coverage": _readout(
                    "fraction",
                    all_rows["strict_complete_covered_count"]
                    / coverage["heldout_count"],
                    numerator=int(all_rows["strict_complete_covered_count"]),
                    denominator=int(coverage["heldout_count"]),
                    conditioning="Rows with declarations and no parse/truncation/file-support defects.",
                ),
                "placeholder_positive_event_count": _readout(
                    "count",
                    int(events["placeholder_positive_event_witness_count"]),
                    conditioning="Positive events may be licensed on partial parses.",
                ),
                "collision_or_shadow_positive_event_count": _readout(
                    "count",
                    int(events["collision_or_shadowing_positive_event_witness_count"]),
                    conditioning="Positive events may be licensed on partial parses; rows may overlap other event types.",
                ),
                "placeholder_negative_support_count": _readout(
                    "count",
                    int(events["placeholder_strict_complete_negative_support_count"]),
                    conditioning="Negative absence support is licensed only on strict-complete rows.",
                ),
                "collision_or_shadow_negative_support_count": _readout(
                    "count",
                    int(
                        events[
                            "collision_or_shadowing_strict_complete_negative_support_count"
                        ]
                    ),
                    conditioning="Negative absence support is licensed only on strict-complete rows.",
                ),
                "semantic_context_fit_coverage": _readout(
                    "fraction",
                    0.0,
                    numerator=0,
                    denominator=int(coverage["heldout_count"]),
                    conditioning="All held-out rows; the relation is explicitly unavailable.",
                ),
            },
            fidelity={
                "relation_fidelity": "parser_scoped",
                "whole_criterion_fidelity": "unavailable",
                "positive_negative_evidence_symmetry": "not_assumed",
            },
            claim_permissions=_permissions(may_claim_code_verifiability=True),
            sources=(_receipt(a407_eval_path), _receipt(a407_coverage_path)),
            claim_boundary="Relation-local events and abstentions only. Event counts overlap and must not be summed into unique-row codability.",
        )
    )

    summary = family["summary"]
    records.append(
        _base_record(
            record_id="code.active_panel.authored_depth_class",
            stratum="program_structure_descriptor",
            domain="code_review",
            criterion_id=None,
            relation_id=None,
            selection=_selection(
                "complete_active_panel",
                "retrospective inventory",
                representative=True,
                note="All 18 active coding criteria.",
            ),
            units=_units(
                "criterion_program",
                population_n=int(summary["active_criteria"]),
                conditioning="Active code-review panel only.",
            ),
            channels={"runtime": "code_for_both_depth_classes"},
            readouts={
                "deep_program_count": _readout(
                    "count",
                    int(summary["criteria_with_deep_program"]),
                    conditioning="Pre-existing manually engineered coding-A programs.",
                ),
                "shallow_comparator_count": _readout(
                    "count",
                    int(summary["criteria_with_train_selected_shallow_comparator"]),
                    conditioning="Prompt-generated executable programs selected on TRAIN.",
                ),
            },
            fidelity={
                "descriptor_kind": "authored_depth_class",
                "scale_id": "active-code.manual-deep-vs-prompt-generated-shallow.v1",
                "numeric_control_flow_depth_available": False,
            },
            claim_permissions=_permissions(),
            sources=(_receipt(family_path), _receipt(family_source)),
            claim_boundary="Deep/shallow is an authoring class, not a standardized node or path depth; compare only within this scale.",
        )
    )

    structure_path = base / "code_program_structure_retrospective_001/results.json"
    structure_source = (
        ROOT / "methods/metric_seam/pilot/code_program_structure_retrospective.py"
    )
    structure_result = _read_json(structure_path)
    paired = structure_result["paired_summary"]
    associations = structure_result["association_sensitivity"]
    if (
        structure_result["schema"]
        != "metric-seam.code-program-structure-retrospective.v1"
        or structure_result["scope"]["deep_programs"] != 18
        or structure_result["scope"]["train_selected_shallow_programs"] != 15
        or len(structure_result["pairs"]) != 15
    ):
        raise LedgerError("active-code source-structure family changed unexpectedly")
    records.append(
        _base_record(
            record_id="code.active_panel.entry_module_source_structure",
            stratum="program_structure_descriptor",
            domain="code_review",
            criterion_id=None,
            relation_id=None,
            selection=_selection(
                "retrospective_full_family",
                "source inventory after reconstruction results were visible",
                representative=True,
                note="All 15 active deep/shallow pairs with a TRAIN-selected shallow executable comparator.",
            ),
            units=_units(
                "deep_shallow_entry_module_pair",
                population_n=int(paired["ast_nodes"]["pair_n"]),
                eligible_n=int(associations["comparison_support_eligible"]["n"]),
                conditioning="Python entry modules only; shared parser and library internals are excluded.",
            ),
            channels={
                "deep": "manually_engineered_executable_code_entry_module",
                "shallow": "train_selected_prompt_generated_executable_code_entry_module",
                "runtime": "code_for_both_arms",
            },
            readouts={
                "ast_nodes_deep_greater_rate": _readout(
                    "fraction",
                    paired["ast_nodes"]["deep_greater_count"]
                    / paired["ast_nodes"]["pair_n"],
                    numerator=int(paired["ast_nodes"]["deep_greater_count"]),
                    denominator=int(paired["ast_nodes"]["pair_n"]),
                    conditioning="All 15 paired entry modules.",
                ),
                "nonblank_lines_deep_greater_rate": _readout(
                    "fraction",
                    paired["nonblank_noncomment_lines"]["deep_greater_count"]
                    / paired["nonblank_noncomment_lines"]["pair_n"],
                    numerator=int(
                        paired["nonblank_noncomment_lines"]["deep_greater_count"]
                    ),
                    denominator=int(paired["nonblank_noncomment_lines"]["pair_n"]),
                    conditioning="All 15 paired entry modules.",
                ),
                "condensed_call_path_deep_greater_rate": _readout(
                    "fraction",
                    paired["condensed_longest_path_edges"]["deep_greater_count"]
                    / paired["condensed_longest_path_edges"]["pair_n"],
                    numerator=int(
                        paired["condensed_longest_path_edges"]["deep_greater_count"]
                    ),
                    denominator=int(paired["condensed_longest_path_edges"]["pair_n"]),
                    conditioning="Scope-qualified lexical call graphs after SCC condensation; two ties.",
                ),
                "deep_median_ast_nodes": _readout(
                    "median_count",
                    float(paired["ast_nodes"]["deep_median"]),
                    support_n=int(paired["ast_nodes"]["pair_n"]),
                    conditioning="Deep entry modules across 15 paired criteria.",
                ),
                "shallow_median_ast_nodes": _readout(
                    "median_count",
                    float(paired["ast_nodes"]["shallow_median"]),
                    support_n=int(paired["ast_nodes"]["pair_n"]),
                    conditioning="Shallow entry modules across 15 paired criteria.",
                ),
                "deep_median_condensed_call_path_edges": _readout(
                    "median_count",
                    float(paired["condensed_longest_path_edges"]["deep_median"]),
                    support_n=int(paired["condensed_longest_path_edges"]["pair_n"]),
                    conditioning="Scope-qualified lexical entry-module call graphs.",
                ),
                "shallow_median_condensed_call_path_edges": _readout(
                    "median_count",
                    float(paired["condensed_longest_path_edges"]["shallow_median"]),
                    support_n=int(paired["condensed_longest_path_edges"]["pair_n"]),
                    conditioning="Scope-qualified lexical entry-module call graphs.",
                ),
                "call_path_association_all_defined": _readout(
                    "spearman",
                    float(
                        associations["all_defined"]["metrics"][
                            "condensed_longest_path_edges"
                        ]["spearman_structure_delta_vs_reconstruction_delta"]
                    ),
                    support_n=int(associations["all_defined"]["n"]),
                    conditioning="Pairs with defined reconstruction delta; post-hoc descriptive association only.",
                    inference_status="descriptive_post_hoc_no_p_value",
                ),
                "call_path_association_minimum_common_n_20": _readout(
                    "spearman",
                    float(
                        associations["minimum_common_n_20"]["metrics"][
                            "condensed_longest_path_edges"
                        ]["spearman_structure_delta_vs_reconstruction_delta"]
                    ),
                    support_n=int(associations["minimum_common_n_20"]["n"]),
                    conditioning="Pairs with reconstruction common n at least 20; post-hoc descriptive association only.",
                    inference_status="descriptive_post_hoc_no_p_value",
                ),
                "call_path_association_comparison_eligible": _readout(
                    "spearman",
                    float(
                        associations["comparison_support_eligible"]["metrics"][
                            "condensed_longest_path_edges"
                        ]["spearman_structure_delta_vs_reconstruction_delta"]
                    ),
                    support_n=int(associations["comparison_support_eligible"]["n"]),
                    conditioning="Four reconstruction-comparison-eligible pairs; post-hoc descriptive association only.",
                    inference_status="descriptive_post_hoc_no_p_value",
                ),
            },
            fidelity={
                "descriptor_kind": structure_result["descriptor_semantics"]["kind"],
                "scale_id": "code-program-structure.entry-module-ast-and-scope-qualified-callgraph.v1",
                "call_graph_semantics": structure_result["descriptor_semantics"][
                    "call_graph"
                ],
                "transitive_dependencies_included": structure_result[
                    "descriptor_semantics"
                ]["transitive_dependencies_included"],
                "semantic_relation_depth": "not_measured",
                "dynamic_execution_depth": "not_measured",
                "association_interpretation": structure_result[
                    "association_interpretation"
                ],
                "association_upstream_note": "The source-structure artifact was computed against the numerically identical historical v1 reconstruction artifact; the ledger's current scalar records and added receipt use corrected v2 claim language.",
            },
            claim_permissions=_permissions(),
            sources=(
                _receipt(structure_path),
                _receipt(structure_source),
                _receipt(family_path),
            ),
            claim_boundary="Entry-module syntax and a conservative lexical call graph only. Sign-changing post-hoc associations do not support a directional structure-signal claim and are never semantic depth.",
        )
    )
    return records


def _science_relation_counts(path: Path) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                matches = row["result"]["matches"]
            except (KeyError, TypeError) as exc:
                raise LedgerError(
                    f"invalid science result on line {line_number}"
                ) from exc
            for match in matches:
                counts[(str(match["claim"]["relation"]), str(match["decision"]))] += 1
    return counts


def _science_records() -> list[dict[str, Any]]:
    code_dir = (
        ROOT
        / "outputs/metric_seam_pilot/science_verifiability_v9_relation_strict_addressed"
    )
    manifest_path = code_dir / "manifest.json"
    results_path = code_dir / "code_results.jsonl"
    source_path = (
        ROOT / "methods/metric_seam/science_claims_v2/addressed_code_comparator_v9.py"
    )
    continuous_results_path = (
        ROOT
        / "outputs/metric_seam_pilot/science_claims_v2_relation_strict_v23/results.json"
    )
    strict_core_path = (
        ROOT / "methods/metric_seam/science_claims_v2/core_relation_strict.py"
    )
    addressed_v8_path = (
        ROOT / "methods/metric_seam/science_claims_v2/addressed_code_comparator_v8.py"
    )
    prompt_manifest_path = (
        ROOT
        / "outputs/metric_seam_pilot/science_articulability_v8_hardened_prepared/manifest.json"
    )
    manifest = _read_json(manifest_path)
    continuous = _read_json(continuous_results_path)
    prompt = _read_json(prompt_manifest_path)
    summary = manifest["summary"]
    representation = manifest["representation_comparison"]
    strong_normalized = representation["strong_whitespace_normalized_text"]
    supported_sets = representation["supported_paper_sets"]
    weak_normalized = representation["weak_whitespace_normalized_text"]
    archived_manifest_has_bindings = any(
        key in manifest
        for key in ("inputs", "implementation_dependencies", "source_bindings")
    )
    if (
        continuous["schema_version"] != "science-claims-v2.3-relation-strict"
        or len(continuous["records"]) != summary["records"]
        or archived_manifest_has_bindings
        or strong_normalized
        != {
            "addressed": 100,
            "addressed_only": 0,
            "continuous": 100,
            "continuous_only": 0,
            "intersection": 100,
        }
        or supported_sets
        != {
            "addressed": 95,
            "addressed_only": 0,
            "continuous": 95,
            "continuous_only": 0,
            "intersection": 95,
        }
        or representation["paper_status_agreement"] != 2396
        or representation["paper_status_total"] != 2400
        or weak_normalized
        != {
            "addressed": 430,
            "addressed_only": 1,
            "continuous": 434,
            "continuous_only": 5,
            "intersection": 429,
        }
    ):
        raise LedgerError("science v9 representation comparison changed unexpectedly")
    relation_counts = _science_relation_counts(results_path)
    expected_decisions = {
        "supported": int(summary["decision_counts"]["supported"]),
        "evidence_link": int(summary["decision_counts"]["evidence_link"]),
        "insufficient": int(summary["decision_counts"]["insufficient"]),
    }
    observed_decisions = Counter()
    for (relation, decision), count in relation_counts.items():
        del relation
        observed_decisions[decision] += count
    if dict(observed_decisions) != expected_decisions:
        raise LedgerError(
            f"science relation decisions disagree with manifest: {observed_decisions}"
        )

    matched = summary["matched_relation_counts"]
    selected = summary["selected_relation_counts"]
    records = [
        _base_record(
            record_id="science.v9.document_local_relations",
            stratum="relation_instance_verification",
            domain="science",
            criterion_id="document_local_claim_evidence_relation",
            relation_id="numeric_comparative_and_surface_evidence_relations",
            selection=_selection(
                "manually_constructed_retrospective_seed",
                "full-corpus CPU replay",
                representative=False,
                note="Program decomposition was manually supplied; corpus is the available 2,400-paper peer-review set, not a census of science metrics.",
            ),
            units=_units(
                "selected_claim_evidence_match",
                population_n=int(summary["records"]),
                eligible_n=int(summary["selected_claim_addresses"]),
                candidate_n=int(summary["matched_claim_addresses"]),
                conditioning="Selected abstract claim addresses and one-to-one matched body evidence addresses.",
            ),
            channels={
                "candidate": "code_document_local_relation_parser",
                "prompt_articulability": "prepared_not_run",
                "external_scientific_truth": None,
                "representation_comparison_left": "same_strict_code_program_over_continuous_text",
                "representation_comparison_right": "same_strict_code_program_over_exact_addressed_spans",
                "representation_comparison_type": "code_to_code_same_program_input_representation_robustness",
            },
            readouts={
                "strong_numeric_witness_rate": _readout(
                    "fraction",
                    relation_counts[("numeric", "supported")] / matched["numeric"],
                    numerator=int(relation_counts[("numeric", "supported")]),
                    denominator=int(matched["numeric"]),
                    conditioning="Matched numeric claim-evidence instances only.",
                ),
                "strong_comparative_witness_rate": _readout(
                    "fraction",
                    relation_counts[("comparative", "supported")]
                    / matched["comparative"],
                    numerator=int(relation_counts[("comparative", "supported")]),
                    denominator=int(matched["comparative"]),
                    conditioning="Matched comparative claim-evidence instances only.",
                ),
                "strong_witness_rate_all_matched": _readout(
                    "fraction",
                    summary["certificates"] / summary["matched_claim_addresses"],
                    numerator=int(summary["certificates"]),
                    denominator=int(summary["matched_claim_addresses"]),
                    conditioning="All matched relation instances; strong parser certificates only.",
                ),
                "weak_theoretical_link_rate": _readout(
                    "fraction",
                    relation_counts[("theoretical", "evidence_link")]
                    / matched["theoretical"],
                    numerator=int(relation_counts[("theoretical", "evidence_link")]),
                    denominator=int(matched["theoretical"]),
                    conditioning="Matched theoretical relation instances; weak evidence-link tier only.",
                ),
                "weak_empirical_link_rate": _readout(
                    "fraction",
                    relation_counts[("empirical", "evidence_link")]
                    / matched["empirical"],
                    numerator=int(relation_counts[("empirical", "evidence_link")]),
                    denominator=int(matched["empirical"]),
                    conditioning="Matched empirical relation instances; weak evidence-link tier only.",
                ),
                "weak_qualitative_link_rate": _readout(
                    "fraction",
                    relation_counts[("qualitative", "evidence_link")]
                    / matched["qualitative"],
                    numerator=int(relation_counts[("qualitative", "evidence_link")]),
                    denominator=int(matched["qualitative"]),
                    conditioning="Matched qualitative relation instances; weak evidence-link tier only.",
                ),
                "prompt_articulability_output_rate": _readout(
                    "fraction",
                    None,
                    status="not_run",
                    conditioning=f"{prompt['files']['requests']['count']} body-present requests prepared and zero model calls made.",
                ),
                "same_program_strong_normalized_overlap_given_continuous": _readout(
                    "fraction",
                    strong_normalized["intersection"] / strong_normalized["continuous"],
                    numerator=int(strong_normalized["intersection"]),
                    denominator=int(strong_normalized["continuous"]),
                    conditioning="Whitespace-normalized strong certificate identities from the continuous-text code run recovered by the exact-address code run; same strict executable relation program, code-to-code only.",
                ),
                "same_program_strong_normalized_overlap_given_addressed": _readout(
                    "fraction",
                    strong_normalized["intersection"] / strong_normalized["addressed"],
                    numerator=int(strong_normalized["intersection"]),
                    denominator=int(strong_normalized["addressed"]),
                    conditioning="Whitespace-normalized strong certificate identities from the exact-address code run recovered by the continuous-text code run; same strict executable relation program, code-to-code only.",
                ),
                "same_program_strong_normalized_intersection_count": _readout(
                    "count",
                    int(strong_normalized["intersection"]),
                    conditioning="Shared whitespace-normalized strong certificate identities across the two code input representations.",
                ),
                "same_program_supported_set_overlap_given_continuous": _readout(
                    "fraction",
                    supported_sets["intersection"] / supported_sets["continuous"],
                    numerator=int(supported_sets["intersection"]),
                    denominator=int(supported_sets["continuous"]),
                    conditioning="Supported-paper set from the continuous-text code run recovered by the exact-address code run.",
                ),
                "same_program_supported_set_overlap_given_addressed": _readout(
                    "fraction",
                    supported_sets["intersection"] / supported_sets["addressed"],
                    numerator=int(supported_sets["intersection"]),
                    denominator=int(supported_sets["addressed"]),
                    conditioning="Supported-paper set from the exact-address code run recovered by the continuous-text code run.",
                ),
                "same_program_supported_set_intersection_count": _readout(
                    "count",
                    int(supported_sets["intersection"]),
                    conditioning="Shared supported-paper set across continuous and exact-address code representations.",
                ),
                "same_program_paper_status_agreement": _readout(
                    "fraction",
                    representation["paper_status_agreement"]
                    / representation["paper_status_total"],
                    numerator=int(representation["paper_status_agreement"]),
                    denominator=int(representation["paper_status_total"]),
                    conditioning="Paper-level status equality across the same strict code program under continuous and exact-address input representations.",
                ),
                "same_program_weak_normalized_overlap_given_continuous": _readout(
                    "fraction",
                    weak_normalized["intersection"] / weak_normalized["continuous"],
                    numerator=int(weak_normalized["intersection"]),
                    denominator=int(weak_normalized["continuous"]),
                    conditioning="Whitespace-normalized weak evidence links from the continuous-text code run recovered by the exact-address code run; weak tier remains separate.",
                ),
                "same_program_weak_normalized_overlap_given_addressed": _readout(
                    "fraction",
                    weak_normalized["intersection"] / weak_normalized["addressed"],
                    numerator=int(weak_normalized["intersection"]),
                    denominator=int(weak_normalized["addressed"]),
                    conditioning="Whitespace-normalized weak evidence links from the exact-address code run recovered by the continuous-text code run; weak tier remains separate.",
                ),
                "same_program_weak_normalized_intersection_count": _readout(
                    "count",
                    int(weak_normalized["intersection"]),
                    conditioning="Shared whitespace-normalized weak evidence links across the two code input representations.",
                ),
                "same_program_weak_continuous_only_count": _readout(
                    "count",
                    int(weak_normalized["continuous_only"]),
                    conditioning="Whitespace-normalized weak links emitted only by the continuous-text code representation.",
                ),
                "same_program_weak_addressed_only_count": _readout(
                    "count",
                    int(weak_normalized["addressed_only"]),
                    conditioning="Whitespace-normalized weak links emitted only by the exact-address code representation.",
                ),
            },
            fidelity={
                "certificate_scope": manifest["certificate_scope"],
                "relation_fidelity": "strict_parser_scoped",
                "input_representation_fidelity": "whitespace_normalized_strong_set_100_of_100",
                "whole_science_metric_fidelity": "not_established",
                "external_truth": "not_claimed",
                "representation_comparison_scope": "same_program_code_to_code_input_representation_robustness",
                "prompt_to_code_isomorphism_from_overlap": "not_licensed",
                "domain_codability_from_overlap": "not_licensed",
                "archived_v9_manifest_source_bindings": "absent",
                "ledger_source_receipts_status": "additive_bindings_only_not_retroactive_manifest_repair",
            },
            claim_permissions=_permissions(may_claim_code_verifiability=True),
            sources=(
                _receipt(manifest_path),
                _receipt(results_path),
                _receipt(continuous_results_path),
                _receipt(strict_core_path),
                _receipt(addressed_v8_path),
                _receipt(source_path),
                _receipt(prompt_manifest_path),
            ),
            claim_boundary="Document-local parser witnesses plus same-program code-to-code input-representation robustness, not external scientific truth, prompt-to-code isomorphism, or science-domain codability. Strong and weak tiers are separate and must not be added. Source receipts are additive ledger bindings because the archived v9 manifest itself lacks dependency bindings.",
        ),
        _base_record(
            record_id="science.v9.instance_evidence_graph",
            stratum="program_structure_descriptor",
            domain="science",
            criterion_id="document_local_claim_evidence_relation",
            relation_id=None,
            selection=_selection(
                "full_corpus_execution",
                "retrospective structure inventory",
                representative=False,
                note="Graph counts describe runtime evidence instances, not the comparator's control flow.",
            ),
            units=_units(
                "evidence_graph_instance",
                population_n=int(summary["records"]),
                conditioning="All corpus papers processed by science v9.",
            ),
            channels={"runtime": "code"},
            readouts={
                "candidate_edge_count": _readout(
                    "count",
                    int(summary["graph_edges"]),
                    conditioning="All candidate claim-to-body evidence edges.",
                ),
                "selected_claim_count": _readout(
                    "count",
                    int(summary["selected_claim_addresses"]),
                    conditioning="Selected abstract claim addresses.",
                ),
                "matched_edge_count": _readout(
                    "count",
                    int(summary["matched_claim_addresses"]),
                    conditioning="One-to-one matched claim-evidence edges.",
                ),
            },
            fidelity={
                "descriptor_kind": "instance_evidence_graph",
                "scale_id": "science-v9.addressed-claim-evidence-graph.v1",
                "control_flow_depth_available": False,
            },
            claim_permissions=_permissions(),
            sources=(_receipt(manifest_path), _receipt(results_path)),
            claim_boundary="Instance evidence-graph size is not executable program depth and cannot be compared numerically with typed-DAG path length.",
        ),
    ]
    # Selected counts are used only as a cross-check; the ledger denominator is
    # matched instances because decisions exist only after matching.
    if (
        sum(int(value) for value in selected.values())
        != summary["selected_claim_addresses"]
    ):
        raise LedgerError(
            "science selected relation counts do not sum to selected claims"
        )
    return records


def _load_dag_program(path: Path, module_name: str) -> Mapping[str, Any]:
    """Load an already-executed WS4 source artifact to inspect its frozen PROG."""

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise LedgerError(f"cannot load DAG program: {path.relative_to(ROOT)}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    program = getattr(module, "PROG", None)
    if not isinstance(program, Mapping):
        raise LedgerError(f"DAG program has no PROG mapping: {path.relative_to(ROOT)}")
    return program


def _dag_structure(program: Mapping[str, Any]) -> dict[str, Any]:
    nodes = {str(node["id"]): node for node in program["nodes"]}
    if len(nodes) != len(program["nodes"]):
        raise LedgerError("DAG contains duplicate node IDs")
    visiting: set[str] = set()
    memo: dict[str, int] = {}

    def depth(node_id: str) -> int:
        if node_id in memo:
            return memo[node_id]
        if node_id in visiting:
            raise LedgerError("DAG contains a cycle")
        visiting.add(node_id)
        needs = nodes[node_id]["needs"]
        value = 0 if not needs else 1 + max(depth(str(parent)) for parent in needs)
        visiting.remove(node_id)
        memo[node_id] = value
        return value

    out = str(program["out"])
    deepest = depth(out)
    l_nodes = [node for node in nodes.values() if node["impl"] == "L"]
    return {
        "n_nodes": len(nodes),
        "n_code_nodes": sum(node["impl"] == "C" for node in nodes.values()),
        "n_llm_nodes": len(l_nodes),
        "n_evidence_nodes": sum(
            node["op_class"] == "evidence" for node in nodes.values()
        ),
        "deepest_output_path_edges": deepest,
        "llm_node_graph_depths": sorted(depth(str(node["id"])) for node in l_nodes),
        "llm_node_abstraction_levels": sorted(int(node["level"]) for node in l_nodes),
    }


def _patent_records() -> list[dict[str, Any]]:
    task_dir = ROOT / "outputs/metric_seam_pilot/tasks/patents_pa"
    report_path = task_dir / "ws3_eval_report.json"
    evidence_rows_path = task_dir / "ws3_evidence_results.jsonl"
    source_path = ROOT / "methods/metric_seam/f2p_mock/ws3_eval_evidence.py"
    family_path = (
        ROOT
        / "outputs/metric_seam_pilot/reconstruction_v2/patent_ws3_family_retrospective_001/results.json"
    )
    family_source_path = (
        ROOT / "methods/metric_seam/pilot/patent_ws3_family_retrospective.py"
    )
    report = _read_json(report_path)
    family = _read_json(family_path)
    if (
        family["schema"] != "metric-seam.patent-ws3-family-retrospective.v1"
        or family["summary"]["registered_criteria"] != 4
        or family["summary"]["bh_family_size"] != 4
    ):
        raise LedgerError("patent WS3 retrospective family changed unexpectedly")
    family_by_id = {row["criterion_id"]: row for row in family["criteria"]}
    records: list[dict[str, Any]] = []
    reliable = {"a26", "a34", "a35"}

    for criterion_id in ("a26", "a34", "a35", "a60"):
        legacy_evidence = report[criterion_id]["evidence"]
        evidence = family_by_id[criterion_id]
        is_reliable = criterion_id in reliable
        if (
            evidence["heldout_n"] != legacy_evidence["n_test"]
            or not math.isclose(
                evidence["rho_full_evidence_operation"],
                legacy_evidence["rho_full"],
                abs_tol=0.001,
            )
            or not math.isclose(
                evidence["rho_null_operation"],
                legacy_evidence["rho_null"],
                abs_tol=0.001,
            )
        ):
            raise LedgerError(f"patent {criterion_id} retrospective disagrees with WS3")
        bootstrap = evidence["paired_bootstrap"]
        if bootstrap["status"] == "available":
            bootstrap_lower = _readout(
                "delta_spearman_ci_bound",
                float(bootstrap["interval"][0]),
                support_n=int(evidence["reference_common_n"]),
                conditioning="Lower bound of the retrospective paired-item percentile bootstrap interval.",
                inference_status="retrospective_descriptive_interval",
            )
            bootstrap_upper = _readout(
                "delta_spearman_ci_bound",
                float(bootstrap["interval"][1]),
                support_n=int(evidence["reference_common_n"]),
                conditioning="Upper bound of the retrospective paired-item percentile bootstrap interval.",
                inference_status="retrospective_descriptive_interval",
            )
        else:
            bootstrap_lower = _readout(
                "delta_spearman_ci_bound",
                None,
                status="unavailable",
                conditioning=f"Retrospective paired bootstrap unavailable: {bootstrap['reason']}.",
            )
            bootstrap_upper = _readout(
                "delta_spearman_ci_bound",
                None,
                status="unavailable",
                conditioning=f"Retrospective paired bootstrap unavailable: {bootstrap['reason']}.",
            )
        records.append(
            _base_record(
                record_id=f"patents.ws3.{criterion_id}.evidence_arm",
                stratum="criterion_scalar_reconstruction",
                domain="patents",
                criterion_id=criterion_id,
                relation_id=None,
                selection=_selection(
                    "retrospective_full_four_criterion_family",
                    "family inference computed after programs and aggregate outcomes were visible",
                    representative=False,
                    note="Complete registered WS3 family, but outside the 159-criterion census panel and not prospectively confirmatory.",
                ),
                units=_units(
                    "criterion_x_patent_application_scalar",
                    population_n=4,
                    train_n=150,
                    heldout_n=int(evidence["heldout_n"]),
                    reference_n=int(evidence["reference_common_n"]),
                    common_n=int(evidence["reference_common_n"]),
                    conditioning="Frozen evidence-aware prompt/LLM reference on the 100-item seed-7 held-out split.",
                ),
                channels={
                    "candidate": "hybrid_code_retrieval_and_llm_extracted_fields",
                    "reference": "evidence_aware_prompt_llm",
                    "ablation": "same_program_with_null_prior_art_ops",
                    "oracle_caveat": "retrieval features depend on an upstream LLM/oracle-conditioned disclosure artifact",
                },
                readouts={
                    "full_reconstruction_spearman": _readout(
                        "spearman",
                        float(evidence["rho_full_evidence_operation"]),
                        support_n=int(evidence["reference_common_n"]),
                        conditioning="Evidence-aware test arm.",
                        inference_status="retrospective_descriptive",
                    ),
                    "null_reconstruction_spearman": _readout(
                        "spearman",
                        float(evidence["rho_null_operation"]),
                        support_n=int(evidence["reference_common_n"]),
                        conditioning="Same test arm with prior-art operation nulled.",
                        inference_status="retrospective_descriptive",
                    ),
                    "operator_marginal_delta": _readout(
                        "delta_spearman",
                        float(evidence["delta_spearman"]),
                        support_n=int(evidence["reference_common_n"]),
                        conditioning="Full minus null prior-art operation on the evidence-aware arm.",
                        inference_status="retrospective_family_analysis",
                    ),
                    "reference_repeatability": _readout(
                        "spearman",
                        float(evidence["reference_two_pass_spearman"]),
                        support_n=int(evidence["reference_common_n"]),
                        conditioning="Prompt/LLM reference repeatability on the evidence-aware arm.",
                        inference_status="instrument_reliability",
                    ),
                    "paired_randomization_p_value": _readout(
                        "paired_randomization_p_value",
                        float(evidence["paired_randomization"]["p_value"]),
                        support_n=int(evidence["reference_common_n"]),
                        conditioning="One-sided paired swap test for full-operation correlation greater than null-operation correlation.",
                        inference_status="retrospective_family_analysis",
                    ),
                    "bh_q_value": _readout(
                        "benjamini_hochberg_q_value",
                        float(evidence["bh_q_value"]),
                        support_n=int(evidence["reference_common_n"]),
                        conditioning="BH correction across the complete four-criterion WS3 family.",
                        inference_status="multiplicity_controlled_retrospective",
                    ),
                    "bh_fdr_reject": _readout(
                        "boolean",
                        bool(evidence["bh_fdr_reject"]),
                        conditioning="Retrospective BH-FDR alpha=.05 across all four registered criteria.",
                        inference_status="multiplicity_controlled_retrospective",
                    ),
                    "threshold_and_fdr_screens_met": _readout(
                        "boolean",
                        bool(evidence["threshold_and_fdr_screens_met"]),
                        conditioning="Reference reliability, absolute rho, minimum effect, and BH-FDR screens.",
                        inference_status="retrospective_family_analysis",
                    ),
                    "effect_precision_characterized": _readout(
                        "boolean",
                        bool(evidence["effect_precision_characterized"]),
                        conditioning="Threshold/FDR screens pass and a finite paired bootstrap interval is available with lower bound above zero.",
                        inference_status="retrospective_family_analysis",
                    ),
                    "null_score_modal_fraction": _readout(
                        "fraction",
                        float(evidence["null_score_modal_fraction"]),
                        numerator=round(
                            evidence["null_score_modal_fraction"]
                            * evidence["reference_common_n"]
                        ),
                        denominator=int(evidence["reference_common_n"]),
                        conditioning="Null-operation held-out scores; values at or above .95 trigger a near-degeneracy warning.",
                    ),
                    "paired_bootstrap_ci_lower": bootstrap_lower,
                    "paired_bootstrap_ci_upper": bootstrap_upper,
                },
                fidelity={
                    "reference_reliability": (
                        "usable_historical"
                        if is_reliable
                        else "too_low_for_target_quality_claim"
                    ),
                    "oracle_provenance": "disclosed",
                    "construct_fidelity": "not_independently_established",
                    "confirmation_status": family["confirmation_status"],
                    "null_rank_support_warning": evidence["null_rank_support_warning"],
                    "null_score_unique_values": evidence["null_score_unique_values"],
                    "bootstrap_status": bootstrap["status"],
                },
                claim_permissions=_permissions(
                    may_claim_descriptive_reconstruction=is_reliable
                ),
                sources=(
                    _receipt(family_path),
                    _receipt(family_source_path),
                    _receipt(report_path),
                    _receipt(evidence_rows_path),
                    _receipt(source_path),
                ),
                claim_boundary=(
                    "Retrospective full-family reconstruction with BH-FDR and an oracle-conditioned retrieval caveat; not confirmatory evidence or patent-domain codability."
                    if is_reliable
                    else "Target reference reliability is too low for a reconstruction conclusion; retain only as an instrument-quality negative."
                ),
                nonrecomputable_claims=(
                    "The exact upstream oracle-injection rate cannot be recomputed locally because datasets/patents/processed/option3_claims_gemma_scale.jsonl is absent.",
                ),
            )
        )

    ws4_root = ROOT / "outputs/metric_seam_pilot/battery/effort_ladder/ws4"
    for criterion_id in ("a26", "a34", "a35"):
        cell_dir = ws4_root / f"patents_pa__{criterion_id}"
        dag_path = cell_dir / "dag_program.py"
        readouts_path = cell_dir / "readouts.json"
        meta_path = cell_dir / "meta.json"
        readouts = _read_json(readouts_path)
        meta = _read_json(meta_path)
        structure = _dag_structure(
            _load_dag_program(dag_path, f"metric_seam_ledger_patent_{criterion_id}")
        )
        declared = meta["structure"]
        if (
            structure["n_nodes"] != declared["n_nodes"]
            or structure["n_code_nodes"] != declared["impl_counts"]["C"]
            or structure["n_llm_nodes"] != declared["impl_counts"]["L"]
            or structure["n_evidence_nodes"] != declared["op_class_counts"]["evidence"]
        ):
            raise LedgerError(
                f"patent {criterion_id} source structure disagrees with meta"
            )
        retrieval = next(
            row
            for row in readouts["per_node_own_transformation_ablation_marginals"]
            if row["node"] == "prior_art_lookup"
        )
        records.append(
            _base_record(
                record_id=f"patents.ws4.{criterion_id}.typed_dag",
                stratum="program_structure_descriptor",
                domain="patents",
                criterion_id=criterion_id,
                relation_id=None,
                selection=_selection(
                    "retrospective_refactor",
                    "bit-exact typed-DAG refactor of frozen h0",
                    representative=False,
                    note="Three preselected patent criteria; DAG refactor preserves historical program behavior and bugs.",
                ),
                units=_units(
                    "typed_dag_program",
                    train_n=int(readouts["n_train"]),
                    conditioning="One frozen program and its TRAIN ablation readouts.",
                ),
                channels={
                    "runtime": "hybrid_code_and_llm_extracted_fields",
                    "retrieval": "code_lookup_over_oracle_conditioned_prior_art_features",
                },
                readouts={
                    "node_count": _readout(
                        "count", structure["n_nodes"], conditioning="Frozen typed DAG."
                    ),
                    "code_node_count": _readout(
                        "count",
                        structure["n_code_nodes"],
                        conditioning="Nodes with impl=C.",
                    ),
                    "llm_node_count": _readout(
                        "count",
                        structure["n_llm_nodes"],
                        conditioning="Nodes with impl=L.",
                    ),
                    "evidence_node_count": _readout(
                        "count",
                        structure["n_evidence_nodes"],
                        conditioning="Nodes tagged op_class=evidence; overlaps code/LLM implementation classes.",
                    ),
                    "deepest_output_path_edges": _readout(
                        "count",
                        structure["deepest_output_path_edges"],
                        conditioning="Longest dependency path from a root node to the declared output.",
                    ),
                    "prior_art_lookup_ablation_delta_rho": _readout(
                        "delta_spearman",
                        float(retrieval["delta_rho"]),
                        support_n=int(retrieval["n_ablated_scored"]),
                        conditioning="TRAIN judge support; own-transformation ablation of the retrieval node.",
                        inference_status="descriptive_train_ablation",
                    ),
                },
                fidelity={
                    "descriptor_kind": "typed_dag_control_flow",
                    "scale_id": "ws4.typed-dag-dependency-edge-depth.v1",
                    "equivalence_to_h0": "bit_exact_pass",
                    "llm_node_graph_depths": structure["llm_node_graph_depths"],
                    "llm_node_abstraction_levels": structure[
                        "llm_node_abstraction_levels"
                    ],
                    "level_and_graph_depth_are_distinct": True,
                },
                claim_permissions=_permissions(),
                sources=(
                    _receipt(dag_path),
                    _receipt(readouts_path),
                    _receipt(meta_path),
                ),
                claim_boundary="Control-flow structure of one bit-exact historical hybrid program; runtime code share does not imply end-to-end pure-code verification.",
            )
        )
    return records


def validate_ledger(ledger: Mapping[str, Any]) -> None:
    """Validate type, denominator, null, and non-pooling invariants."""

    if ledger.get("schema") != SCHEMA:
        raise LedgerError(f"unexpected schema: {ledger.get('schema')!r}")
    if ledger.get("external_supervised_ground_truth_used") is not False:
        raise LedgerError(
            "ledger must preserve the unsupervised reconstruction objective"
        )
    records = ledger.get("records")
    if not isinstance(records, list) or not records:
        raise LedgerError("records must be a non-empty list")
    record_ids: set[str] = set()
    for record in records:
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise LedgerError("every record needs a non-empty record_id")
        if record_id in record_ids:
            raise LedgerError(f"duplicate record_id: {record_id}")
        record_ids.add(record_id)
        stratum = record.get("stratum")
        if stratum not in STRATA:
            raise LedgerError(f"{record_id}: invalid stratum {stratum!r}")
        if record.get("claim_permissions", {}).get("may_claim_domain_codability"):
            raise LedgerError(f"{record_id}: domain codability is not licensed")

        if stratum == "program_structure_descriptor":
            fidelity = record.get("fidelity", {})
            if not fidelity.get("descriptor_kind") or not fidelity.get("scale_id"):
                raise LedgerError(
                    f"{record_id}: structure descriptor needs kind and scale_id"
                )

        units = record.get("units")
        if not isinstance(units, Mapping) or not units.get("kind"):
            raise LedgerError(f"{record_id}: missing unit kind")
        for key in (
            "population_n",
            "train_n",
            "heldout_n",
            "eligible_n",
            "candidate_n",
            "reference_n",
            "common_n",
        ):
            value = units.get(key)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise LedgerError(f"{record_id}: invalid units.{key}={value!r}")

        readouts = record.get("readouts")
        if not isinstance(readouts, Mapping) or not readouts:
            raise LedgerError(f"{record_id}: readouts must be a non-empty object")
        for name, readout in readouts.items():
            prefix = f"{record_id}.{name}"
            if not isinstance(readout, Mapping):
                raise LedgerError(f"{prefix}: readout must be an object")
            status = readout.get("status")
            if status not in READOUT_STATUSES:
                raise LedgerError(f"{prefix}: invalid status {status!r}")
            estimate = readout.get("estimate")
            numerator = readout.get("numerator")
            denominator = readout.get("denominator")
            support_n = readout.get("support_n")
            if status in NULL_STATUSES and estimate is not None:
                raise LedgerError(f"{prefix}: null-status estimate must be null")
            if status == "observed" and estimate is None:
                # Undefined correlations are real observed outcomes, not missing
                # artifacts; they must say so through the metric name.
                if readout.get("metric") not in {"spearman", "delta_spearman"}:
                    raise LedgerError(f"{prefix}: observed estimate is null")
            for count_name, count in (
                ("numerator", numerator),
                ("denominator", denominator),
                ("support_n", support_n),
            ):
                if count is not None and (
                    not isinstance(count, int) or isinstance(count, bool) or count < 0
                ):
                    raise LedgerError(f"{prefix}: invalid {count_name}={count!r}")
            if readout.get("metric") == "fraction":
                if status == "observed":
                    if numerator is None or denominator is None or denominator <= 0:
                        raise LedgerError(
                            f"{prefix}: fraction needs positive denominator"
                        )
                    if numerator > denominator:
                        raise LedgerError(f"{prefix}: numerator exceeds denominator")
                    if not math.isclose(
                        float(estimate),
                        numerator / denominator,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ):
                        raise LedgerError(
                            f"{prefix}: fraction estimate disagrees with n/d"
                        )
                elif numerator is not None or denominator is not None:
                    raise LedgerError(
                        f"{prefix}: null-status fractions must not encode a synthetic 0/n"
                    )
            elif numerator is not None or denominator is not None:
                raise LedgerError(
                    f"{prefix}: numerator/denominator are reserved for fractions"
                )
            if (
                estimate is not None
                and not isinstance(estimate, bool)
                and not _finite(estimate)
            ):
                raise LedgerError(f"{prefix}: non-finite estimate")
            if (
                not isinstance(readout.get("conditioning"), str)
                or not readout["conditioning"]
            ):
                raise LedgerError(f"{prefix}: conditioning is required")

    summary = ledger.get("summary", {})
    if summary.get("record_count") != len(records):
        raise LedgerError("summary record_count disagrees with records")
    if summary.get("domain_codability_estimates_emitted") != 0:
        raise LedgerError("no domain codability estimate is licensed")
    if summary.get("cross_stratum_pooled_estimates_emitted") != 0:
        raise LedgerError("cross-stratum pooling is forbidden")


def _family_summaries(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    code_rows = [
        row for row in records if row["record_id"].startswith("code.active_depth.")
    ]
    eligible = [
        row
        for row in code_rows
        if row["readouts"]["bh_q_value"]["status"] == "observed"
    ]
    improvements = [
        row
        for row in eligible
        if row["readouts"]["bh_q_value"]["estimate"] <= 0.05
        and row["readouts"]["deep_minus_shallow_delta"]["estimate"] >= 0.02
    ]
    patent_rows = [
        row for row in records if row["record_id"].startswith("patents.ws3.")
    ]
    return {
        "active_code_depth_family": {
            "active_criteria": len(code_rows),
            "criteria_with_deep_program": len(code_rows),
            "criteria_with_shallow_comparator": sum(
                row["readouts"]["shallow_reconstruction_spearman"]["status"]
                == "observed"
                for row in code_rows
            ),
            "inferentially_eligible": len(eligible),
            "bh_fdr_and_minimum_effect_improvements": {
                "numerator": len(improvements),
                "denominator": len(eligible),
            },
            "claim": "No multiplicity-controlled deep-over-shallow improvement in the four eligible active criteria; this is not zero code codability.",
        },
        "blind_math_construct_family": {
            "selected_blind_criteria": 2,
            "construct_fidelity_passes": {"numerator": 0, "denominator": 2},
            "claim": "Both selected blind candidates were proxy mismatches; failure is bounded to the frozen program classes and budgets.",
        },
        "patent_historical_selected_family": {
            "selected_criteria": len(patent_rows),
            "bh_fdr_rejections": {
                "numerator": sum(
                    bool(row["readouts"]["bh_fdr_reject"]["estimate"])
                    for row in patent_rows
                ),
                "denominator": len(patent_rows),
            },
            "threshold_and_fdr_screens_met": {
                "numerator": sum(
                    bool(row["readouts"]["threshold_and_fdr_screens_met"]["estimate"])
                    for row in patent_rows
                ),
                "denominator": len(patent_rows),
            },
            "effect_precision_characterized": {
                "numerator": sum(
                    bool(row["readouts"]["effect_precision_characterized"]["estimate"])
                    for row in patent_rows
                ),
                "denominator": len(patent_rows),
            },
            "reference_usable_for_historical_description": {
                "numerator": sum(
                    row["fidelity"]["reference_reliability"] == "usable_historical"
                    for row in patent_rows
                ),
                "denominator": len(patent_rows),
            },
            "claim": "The retrospective full family has 2/4 BH-FDR rejections and threshold/FDR screen passes (a34, a35), but only a35 has characterized positive effect precision; a34's bootstrap interval is unavailable, a26 misses FDR at q=.0568, a60 fails target reliability/absolute fit, and every result remains oracle-conditioned and non-confirmatory.",
        },
    }


def build_ledger() -> dict[str, Any]:
    records = (
        _math_records()
        + _active_code_records()
        + _science_records()
        + _patent_records()
    )
    by_stratum = Counter(row["stratum"] for row in records)
    by_domain = Counter(row["domain"] for row in records)
    ledger = {
        "schema": SCHEMA,
        "objective": "unsupervised_reconstruction_of_articulated_prompt_llm_judgement",
        "external_supervised_ground_truth_used": False,
        "generated_by": "methods/metric_seam/pilot/build_technical_evidence_ledger_v1.py",
        "aggregation_guards": [
            "criterion scalar reconstruction, relation-instance verification, and program structure are separate strata and are never pooled",
            "strong and weak science evidence tiers are reported separately and are never summed",
            "structure values are comparable only when descriptor_kind and scale_id are identical",
            "entry-module source syntax, typed-DAG dependency depth, ordinal relation depth, and instance evidence-graph size are different structure scales",
            "attempted depth, decision-contributing depth, and positive-evidence depth are named views and are never substituted for one another",
            "unavailable, unopened, and not-run values are null rather than zero",
            "every fraction carries numerator, denominator, and conditioning",
            "selected-family rates are not domain codability rates",
            "negative discovery results bound a frozen program class/capability/representation/budget and do not establish tacitness",
            "post-reference inspectability projections and audits do not become new blind, reconstruction, or isomorphism results",
            "articulability is prompt-based; verifiability is code-based; isomorphism is a separate comparison",
        ],
        "records": records,
        "family_summaries": _family_summaries(records),
        "summary": {
            "record_count": len(records),
            "by_stratum": dict(sorted(by_stratum.items())),
            "by_domain": dict(sorted(by_domain.items())),
            "domain_codability_estimates_emitted": 0,
            "cross_stratum_pooled_estimates_emitted": 0,
            "explicitly_nonpoolable": True,
        },
        "known_absences": {
            "math_a12_heldout_scalar": "unavailable; the held-out artifact contains relation-instance witnesses but deliberately defines no parent scalar or correlation",
            "science_prompt_outputs": "not run",
            "code_a407_matched_prompt_outputs": "not run",
        },
    }
    validate_ledger(ledger)
    return ledger


def _pct(readout: Mapping[str, Any]) -> str:
    if readout["status"] != "observed" or readout["estimate"] is None:
        return readout["status"]
    if readout["metric"] != "fraction":
        return str(readout["estimate"])
    return (
        f"{100 * float(readout['estimate']):.1f}% "
        f"({readout['numerator']}/{readout['denominator']})"
    )


def render_report(ledger: Mapping[str, Any]) -> str:
    records = {row["record_id"]: row for row in ledger["records"]}
    lines = [
        "# Technical evidence ledger v1",
        "",
        "This is a CPU-only, source-bound snapshot of the current Math, active Code, Science, and Patent evidence. It preserves the unsupervised reconstruction objective: the reference, when present, is articulated prompt/LLM judgement rather than supervised external truth.",
        "",
        "The central guard is non-pooling. Criterion-level scalar reconstruction, relation-instance verification, and program structure answer different questions. No percentage below is a domain-wide codability estimate.",
        "",
        "## Bounded family results",
        "",
        "| Family | Bounded result | Licensed reading |",
        "|---|---:|---|",
    ]
    families = ledger["family_summaries"]
    code = families["active_code_depth_family"]
    code_result = code["bh_fdr_and_minimum_effect_improvements"]
    math_family = families["blind_math_construct_family"]
    math_result = math_family["construct_fidelity_passes"]
    patents = families["patent_historical_selected_family"]
    patent_result = patents["bh_fdr_rejections"]
    lines.extend(
        [
            f"| Active Code depth family | {code_result['numerator']}/{code_result['denominator']} BH-FDR + minimum-effect improvements | {code['claim']} |",
            f"| Blind Math construct family | {math_result['numerator']}/{math_result['denominator']} construct passes | {math_family['claim']} |",
            f"| Retrospective Patent WS3 family | {patent_result['numerator']}/{patent_result['denominator']} BH-FDR rejections | {patents['claim']} |",
            "",
            "## Relation-instance verification",
            "",
            "| Domain / record | Readout | Result | Conditioning |",
            "|---|---|---:|---|",
        ]
    )
    relation_keys = {
        "math.a144.construct_adversary": ("ordering_pass_rate", "ordering pass"),
        "math.a216.construct_adversary": (
            "pair_pass_rate",
            "pair pass; category floor failed",
        ),
        "math.a12.train_symbolic_step": (
            "parsed_pair_positive_witness_rate",
            "positive exact pair witness",
        ),
        "math.a12.heldout_symbolic_step": (
            "rows_with_executable_pair",
            "held-out row with executable pair",
        ),
        "math.a12.post_reference_pair_projection": (
            "parse_noncoverage_share_of_pair_candidates",
            "projected pair parse noncoverage",
        ),
        "math.a150.sympy_scope_replay": (
            "real_sympy_checkable_rate",
            "real SymPy-checkable occurrence",
        ),
        "code.a407.relation_witnesses": (
            "declaration_coverage",
            "declaration-covered row",
        ),
        "science.v9.document_local_relations": (
            "strong_numeric_witness_rate",
            "strong numeric witness",
        ),
    }
    for record_id, (key, label) in relation_keys.items():
        row = records[record_id]
        readout = row["readouts"][key]
        lines.append(
            f"| {row['domain']} / `{record_id}` | {label} | {_pct(readout)} | {readout['conditioning']} |"
        )
    science = records["science.v9.document_local_relations"]
    for key, label in (
        ("strong_comparative_witness_rate", "strong comparative witness"),
        (
            "strong_witness_rate_all_matched",
            "strong witness across all matched relations",
        ),
        ("weak_theoretical_link_rate", "weak theoretical link"),
        ("weak_empirical_link_rate", "weak empirical link"),
        ("weak_qualitative_link_rate", "weak qualitative link"),
    ):
        readout = science["readouts"][key]
        lines.append(
            f"| science / `science.v9.document_local_relations` | {label} | {_pct(readout)} | {readout['conditioning']} |"
        )

    lines.extend(
        [
            "",
            "Math a12 now has a sealed relation-local held-out execution in addition to its TRAIN preparation: 26/100 held-out rows contain an executable pair, with 11 exact identity and 54 exact nonidentity pair classifications across 65 parsed pairs; 74/100 rows abstain. A later post-reference replay materializes all 277 pair candidates (212 parse noncoverage, 11 identity, 54 nonidentity) and exactly preserves the sealed row and aggregate classifications, but creates no new blind, reconstruction, or isomorphism result. The corrected multi-view depth audit shows 35/100 rows stop at depth 1 and 65/100 attempt and receive a decision from the depth-3 formal path; among those 65, 26 have positive witnesses and 39 end in parse-noncoverage abstention. Positive-evidence depth therefore remains 26@depth3 plus 74 with no positive evidence. The stored prompt reference is available on 99/100 rows (two-pass rho=0.835), but no parent scalar or relation-matched prompt output exists. An exact nonidentity is not a criterion defect without a separate universal-scope rule.",
            "",
            "Science strong and weak rows are deliberately separate tiers. A same-program code-to-code representation audit finds all 100/100 whitespace-normalized strong certificates in both the continuous and exact-address runs (intersection 100), identical supported-paper sets of 95/95 (intersection 95), and paper-status agreement on 2,396/2,400. Weak normalized overlap is 429/434 from the continuous side and 429/430 from the addressed side, with 5 continuous-only and 1 addressed-only. This measures robustness of the same strict executable relation program to two code input representations; it is not prompt-to-code isomorphism or codability. The archived v9 manifest lacks dependency bindings, so the ledger's receipts for the continuous v2.3 result and strict/v8/v9 code are additive provenance rather than a retroactive manifest repair. The prompt articulability arm has 1,957 body-present requests prepared but no outputs, so code-prompt isomorphism remains unavailable.",
            "",
            "## Active Code full family",
            "",
            "Both candidate columns below are executable code. “Deep” is a manually engineered authoring class; “shallow” is a TRAIN-selected prompt-generated executable class. It is not a prompt-versus-code comparison.",
            "",
            "| Criterion | Deep rho (n) | Shallow rho (n) | Delta | Status |",
            "|---|---:|---:|---:|---|",
        ]
    )
    active_rows = sorted(
        (
            row
            for row in ledger["records"]
            if row["record_id"].startswith("code.active_depth.")
        ),
        key=lambda row: int(row["criterion_id"][1:]),
    )
    for row in active_rows:
        deep = row["readouts"]["deep_reconstruction_spearman"]
        shallow = row["readouts"]["shallow_reconstruction_spearman"]
        delta = row["readouts"]["deep_minus_shallow_delta"]
        deep_text = (
            f"{deep['estimate']:.3f} ({deep['support_n']})"
            if deep["estimate"] is not None
            else f"undefined ({deep['support_n']})"
        )
        shallow_text = (
            f"{shallow['estimate']:.3f} ({shallow['support_n']})"
            if shallow["status"] == "observed" and shallow["estimate"] is not None
            else shallow["status"]
            if shallow["status"] != "observed"
            else "undefined"
        )
        delta_text = (
            f"{delta['estimate']:+.3f}"
            if delta["status"] == "observed" and delta["estimate"] is not None
            else delta["status"]
            if delta["status"] != "observed"
            else "undefined"
        )
        lines.append(
            f"| {row['criterion_id']} | {deep_text} | {shallow_text} | {delta_text} | {row['fidelity']['status']} |"
        )

    a407 = records["code.a407.structural_partial_historical"]
    partial = a407["readouts"]["partial_reconstruction_spearman"]
    code_structure = records["code.active_panel.entry_module_source_structure"]
    lines.extend(
        [
            "",
            f"The focused a407 structural partial aggregate is separate from the family program: rho={partial['estimate']:.3f} on n={partial['support_n']}, with {_pct(a407['readouts']['partial_code_coverage'])} coverage of exact-input rows. Its matched raw and hybrid prompt arms are prepared but not run.",
            "",
            f"The source-structure audit confirms that the authoring labels correspond to larger entry modules: deep has more AST nodes on {_pct(code_structure['readouts']['ast_nodes_deep_greater_rate'])} pairs and more nonblank lines on {_pct(code_structure['readouts']['nonblank_lines_deep_greater_rate'])}; its SCC-condensed lexical call path is longer on {_pct(code_structure['readouts']['condensed_call_path_deep_greater_rate'])}. This is source syntax, not semantic relation depth. Its call-path association with reconstruction delta changes from {code_structure['readouts']['call_path_association_all_defined']['estimate']:+.3f} (n={code_structure['readouts']['call_path_association_all_defined']['support_n']}) to {code_structure['readouts']['call_path_association_comparison_eligible']['estimate']:+.3f} (n={code_structure['readouts']['call_path_association_comparison_eligible']['support_n']}) across support strata, so no directional structure-signal claim is licensed.",
            "",
            "## Patent WS3 full retrospective family",
            "",
            "| Criterion | Full rho | Null rho | Delta | BH q | Bootstrap CI | Guard |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for criterion_id in ("a26", "a34", "a35", "a60"):
        row = records[f"patents.ws3.{criterion_id}.evidence_arm"]
        readouts = row["readouts"]
        lower = readouts["paired_bootstrap_ci_lower"]
        upper = readouts["paired_bootstrap_ci_upper"]
        interval = (
            f"[{lower['estimate']:.3f}, {upper['estimate']:.3f}]"
            if lower["status"] == "observed"
            else "unavailable"
        )
        warning = row["fidelity"]["null_rank_support_warning"] or "none"
        lines.append(
            f"| {criterion_id} | {readouts['full_reconstruction_spearman']['estimate']:.3f} | {readouts['null_reconstruction_spearman']['estimate']:.3f} | {readouts['operator_marginal_delta']['estimate']:+.3f} | {readouts['bh_q_value']['estimate']:.4f} | {interval} | {warning} |"
        )
    lines.extend(
        [
            "",
            "The family has retrospective BH-FDR rejections for a34 and a35 only. a26 misses at q=.0568; a60 fails reference-reliability and absolute-fit screens. a34's paired bootstrap interval is unavailable because too many resamples had undefined rank correlation. Null-operation scores are near-degenerate for a26, a34, and a60. All four programs remain manually seeded and oracle-conditioned by force-included examiner-cited prior art, so these are not confirmatory autonomous-retrieval or patent-truth results.",
            "",
            "## Program structure (non-comparable scales)",
            "",
            "| Record | Descriptor kind / scale | Structural readout |",
            "|---|---|---|",
        ]
    )
    for row in ledger["records"]:
        if row["stratum"] != "program_structure_descriptor":
            continue
        fidelity = row["fidelity"]
        kind_scale = f"`{fidelity['descriptor_kind']}` / `{fidelity['scale_id']}`"
        if fidelity["descriptor_kind"] == "typed_dag_control_flow":
            value = (
                f"{row['readouts']['node_count']['estimate']} nodes; "
                f"{row['readouts']['code_node_count']['estimate']} C / "
                f"{row['readouts']['llm_node_count']['estimate']} L; "
                f"deepest output path {row['readouts']['deepest_output_path_edges']['estimate']} edges"
            )
        elif fidelity["descriptor_kind"] == "authored_depth_class":
            value = (
                f"{row['readouts']['deep_program_count']['estimate']} deep; "
                f"{row['readouts']['shallow_comparator_count']['estimate']} shallow comparators; no numeric path depth"
            )
        elif fidelity["descriptor_kind"] == "instance_evidence_graph":
            value = (
                f"{row['readouts']['candidate_edge_count']['estimate']} candidate edges; "
                f"{row['readouts']['matched_edge_count']['estimate']} matched; not control-flow depth"
            )
        elif fidelity["descriptor_kind"] == "python_entry_module_source_structure":
            value = (
                f"deep median {row['readouts']['deep_median_ast_nodes']['estimate']:.0f} vs shallow "
                f"{row['readouts']['shallow_median_ast_nodes']['estimate']:.0f} AST nodes; "
                "scope-qualified lexical graph, not semantic depth"
            )
        elif fidelity["descriptor_kind"] == "relation_depth_program":
            value = (
                f"static maximum relation depth {row['readouts']['static_max_relation_depth']['estimate']}; "
                f"dependency path {row['readouts']['longest_path_edges']['estimate']} edge; "
                "not source size or semantic whole-criterion depth"
            )
        else:
            raise LedgerError(
                f"report has no renderer for descriptor kind {fidelity['descriptor_kind']}"
            )
        lines.append(f"| `{row['record_id']}` | {kind_scale} | {value} |")

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The current artifacts support bounded claims: executable witnesses exist for named relations; some code scores reconstruct articulated prompt/LLM scores on explicit support; and several historical programs have measurable structure. They do not yet support a percentage of Math, Code, Science, or Patent metrics that are codable. A defensible domain percentage needs a frozen criterion sampling frame, a common eligibility rule, a common success gate, and completed prompt and code channels.",
            "",
            "The ledger therefore emits zero domain-codability estimates. Negative discovery outcomes establish only bounded non-discovery under the frozen program class, capabilities, representation, and budget; they do not establish tacitness.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(ledger: Mapping[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ledger.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "REPORT.md").write_text(render_report(ledger), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate that existing outputs exactly match a fresh in-memory build.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ledger = build_ledger()
    if args.check:
        ledger_path = args.out / "ledger.json"
        report_path = args.out / "REPORT.md"
        if _read_json(ledger_path) != ledger:
            raise LedgerError(f"stale ledger output: {ledger_path}")
        if report_path.read_text(encoding="utf-8") != render_report(ledger):
            raise LedgerError(f"stale report output: {report_path}")
    else:
        write_outputs(ledger, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
