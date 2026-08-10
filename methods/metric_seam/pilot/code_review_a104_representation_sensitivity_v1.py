#!/usr/bin/env python3
"""Additive a104 sensitivity to the hierarchy's 4,000-character diff prefix.

This instrument does not replace the sealed a104 V4 result.  It replays the
historical deep checker with the exact parser implementation recorded by the
V3 independent audit, verifies that replay against all 250 frozen scores, then
changes only the code input projection from the historical head/tail ``ctext``
to the hierarchy's ``ctext[:4000]`` string.  The historical two-pass LLM
reference remains fixed on head/tail ``ctext``; this is therefore a one-sided
sensitivity analysis, not a same-input prefix prompt/code comparison.

The analysis is post-hoc and exploratory.  It performs no model call, program
selection, tuning, repository execution, or GPU work, and it defines no gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Callable, Iterator, Mapping, Sequence

import whatthepatch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TASK = ROOT / "outputs/metric_seam_pilot/tasks/code_review"
HIERARCHY_ITEMS = (
    ROOT / "outputs/metric_seam_pilot/hierarchy_r123/items_v2/code-review"
)

ITEMS = TASK / "items.json"
CODE_SCORES = TASK / "code_scores.json"
RESULTS = TASK / "results.jsonl"
V4 = TASK / "a104_cpu_sealed_eval_v4.json"
V3_AUDIT = TASK / "a104_cpu_v3_independent_audit_v1.json"
CANDIDATE = (
    ROOT / "methods/existing_metrics_runner/coded/metrics/a104_test_presence.py"
)
MANIFEST = HIERARCHY_ITEMS / "manifest.json"
TRAIN_ITEMS = HIERARCHY_ITEMS / "compiler_train.json"
HELDOUT_ITEMS = HIERARCHY_ITEMS / "sealed_heldout.json"
DEFAULT_OUT = TASK / "a104_representation_sensitivity_v1.json"

SCHEMA = "metric-seam.code-review-a104-representation-sensitivity.v1"
EXPECTED_FROZEN_SANDBOX_SHA256 = (
    "f1f7723a3934a71e42a0e564735429c1fc508993e376585ffeaca86657c471e6"
)
FROZEN_SANDBOX_GIT_PROVENANCE = {
    "commit": "e6018339153dfecf17dae9a51d3bea8c7c8257c2",
    "sandbox_blob_sha1": "c867caae1ae119f982803eaf785e7dcade04253f",
    "path": "methods/existing_metrics_runner/coded/sandbox.py",
}

MEASUREMENT_INPUT_SHA256 = {
    ITEMS: "676e6bc906eb82c4cab5917231951de8ea29dad62626dbcc95721c3e90884e94",
    CODE_SCORES: "11e50e0ae8d3676a5e6cca0ec17a78f6b503e540bbf7d8666840678ceac71875",
    CANDIDATE: "0463400110726908b4c16fcff3cf9e7083b20b64f01277e81229b6b92f284c1e",
    MANIFEST: "a31803c3474147247b0df75d7b8e5c81bade7c83af433bf7a9a20d0b98fe05d5",
    TRAIN_ITEMS: "0e6f68619dd1405b15ec99899edea50b539d2f12663fc7c0ff35abd2e5038167",
    HELDOUT_ITEMS: "bda8ad98f69f79ff41d3fc0261d0b66c678c24945ce9b35b3b1433836b6584f4",
}
REFERENCE_INPUT_SHA256 = {
    RESULTS: "f15543fff913c0d37f9985864119be19954e5fa95c5ba213912b61200fe42084",
    V4: "8c43b22dd12425f3f523d0fbb4a6765d3c943c10d9ecd945a952c1c904c3d909",
    V3_AUDIT: "ccaad1b876e3c5b5acdd9a9544154bbbff452260db4aa9806252355b927038c8",
}

EXPECTED_DEPENDENCY_VERSIONS = {
    "whatthepatch": "1.0.7",
    "tree-sitter": "0.25.2",
    "tree-sitter-python": "0.25.0",
    "tree-sitter-javascript": "0.25.0",
    "tree-sitter-typescript": "0.23.2",
    "tree-sitter-java": "0.23.5",
    "tree-sitter-go": "0.25.0",
}

# Exact historical function text from the sandbox whose full-file SHA-256 is
# EXPECTED_FROZEN_SANDBOX_SHA256.  Executing this literal in an isolated
# namespace binds behavior to the audited dependency without reverting or
# importing the live parser implementation.
FROZEN_PARSER_SOURCE = '''def parse_diff_added_by_file(diff_text: str) -> Dict[str, str]:
    """Walk a unified diff; return {filepath: added_lines_concatenated}.

    Uses whatthepatch (line-oriented unified-diff parser, tolerant of
    truncation — important because the dense PR text format token-truncates
    the diff). Skips the preamble (## PR Title / ## Description) by jumping
    to the first `diff --git`.
    """
    idx = diff_text.find("diff --git")
    if idx == -1:
        return {}
    out: Dict[str, List[str]] = {}
    try:
        diffs = whatthepatch.parse_patch(diff_text[idx:])
    except Exception:
        return {}
    for d in diffs:
        if d is None:
            continue
        path = (d.header.new_path or d.header.old_path or "")
        if path.startswith("b/"):
            path = path[2:]
        if not path or path == "/dev/null":
            continue
        added = [ch.line for ch in (d.changes or [])
                 if ch.old is None and ch.new is not None and ch.line is not None]
        if added:
            out.setdefault(path, []).extend(added)
    return {f: "\\n".join(ls) for f, ls in out.items() if ls}
'''

# Filled from the literal above, not from the mutable live sandbox.
EXPECTED_FROZEN_PARSER_SOURCE_SHA256 = (
    "7543c918448595ead187ee569df2e3b7b83ff47f705e82b3d9f1dd68b858fb93"
)

EXPECTED_READOUT = {
    "frozen_replay_mismatches": 0,
    "prefix_crosswalk_n": 250,
    "common_heldout_n": 93,
    "head_tail_rho_on_common": 0.6453742826894445,
    "prefix_rho_on_common": 0.5144084535611719,
    "delta_prefix_minus_head_tail": -0.1309658291282726,
    "program_vector_rho": 0.7777488599159648,
    "applicability_status_changes": 12,
    "value_changes_on_common_scored": 118,
}
EXPECTED_MEASUREMENT_FREEZE_SHA256 = (
    "2bd879152465e1098dd7b13ad6a16c37265931bcf7605ffca3087663292980b3"
)


class SensitivityInputError(ValueError):
    """Raised when an input or replay no longer matches the frozen contract."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_paths(expected_sha256: Mapping[Path, str]) -> dict[str, str]:
    observed = {str(path.relative_to(ROOT)): sha256(path) for path in expected_sha256}
    for path, expected in expected_sha256.items():
        actual = observed[str(path.relative_to(ROOT))]
        if actual != expected:
            raise SensitivityInputError(
                f"frozen input changed: {path.relative_to(ROOT)} {actual} != {expected}"
            )
    return observed


def build_frozen_parser() -> Callable[[str], dict[str, str]]:
    """Compile the exact audited parser function in memory."""

    observed = _source_sha256(FROZEN_PARSER_SOURCE)
    if observed != EXPECTED_FROZEN_PARSER_SOURCE_SHA256:
        raise SensitivityInputError(
            f"vendored parser source changed: {observed}"
        )
    namespace = {
        "Dict": dict,
        "List": list,
        "whatthepatch": whatthepatch,
    }
    exec(  # noqa: S102 - fixed hash-verified local source, never external input
        compile(FROZEN_PARSER_SOURCE, "<frozen-a104-parser-v1>", "exec"),
        namespace,
    )
    parser = namespace.get("parse_diff_added_by_file")
    if not callable(parser):
        raise SensitivityInputError("vendored parser did not compile to a callable")
    return parser


@contextmanager
def bind_frozen_parser(candidate_module) -> Iterator[None]:
    """Temporarily bind the candidate's parser global and always restore it."""

    original = candidate_module.parse_diff_added_by_file
    candidate_module.parse_diff_added_by_file = build_frozen_parser()
    try:
        yield
    finally:
        candidate_module.parse_diff_added_by_file = original


def _validate_source_items(payload: object) -> list[dict[str, str]]:
    if not isinstance(payload, list) or len(payload) != 250:
        raise SensitivityInputError("source items must be a 250-row list")
    projected = []
    seen = set()
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise SensitivityInputError(f"source item {index} is not an object")
        key, ctext = row.get("datapoint_id"), row.get("ctext")
        if not isinstance(key, str) or not key or key in seen:
            raise SensitivityInputError(f"source item {index} has invalid/duplicate id")
        if not isinstance(ctext, str) or not ctext.startswith("diff --git"):
            raise SensitivityInputError(f"source item {index} has invalid ctext")
        seen.add(key)
        # Deliberately project away judgement, repository, and PR metadata.
        projected.append({"datapoint_id": key, "ctext": ctext})
    return projected


def _validate_hierarchy_items(payload: object, *, expected_prefix: str) -> list[dict[str, str]]:
    if not isinstance(payload, list) or len(payload) != 125:
        raise SensitivityInputError("each hierarchy split must contain 125 rows")
    rows = []
    seen = set()
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping) or set(row) != {"item_key", "ctext"}:
            raise SensitivityInputError(f"hierarchy item {index} violates two-field schema")
        key, ctext = row["item_key"], row["ctext"]
        if not isinstance(key, str) or not key.startswith(expected_prefix) or key in seen:
            raise SensitivityInputError(f"hierarchy item {index} has invalid/duplicate key")
        if not isinstance(ctext, str) or not ctext.startswith("diff --git"):
            raise SensitivityInputError(f"hierarchy item {index} has invalid ctext")
        seen.add(key)
        rows.append({"item_key": key, "ctext": ctext})
    return rows


def exact_prefix_crosswalk(
    source_items: Sequence[Mapping[str, str]],
    hierarchy_items: Sequence[Mapping[str, str]],
    *,
    max_chars: int,
) -> dict[str, str]:
    """Return source-id -> hierarchy-key after exact unique string matching."""

    projected = [row["ctext"][:max_chars] for row in source_items]
    hierarchy_texts = [row["ctext"] for row in hierarchy_items]
    if len(set(projected)) != len(projected):
        raise SensitivityInputError("source prefix projection is not unique")
    if len(set(hierarchy_texts)) != len(hierarchy_texts):
        raise SensitivityInputError("hierarchy ctext is not unique")
    if Counter(projected) != Counter(hierarchy_texts):
        raise SensitivityInputError("hierarchy ctext is not the exact source-prefix multiset")
    key_by_text = {row["ctext"]: row["item_key"] for row in hierarchy_items}
    return {
        row["datapoint_id"]: key_by_text[row["ctext"][:max_chars]]
        for row in source_items
    }


def _score_program(candidate_module, rows: Sequence[Mapping[str, str]]) -> tuple[dict, list]:
    scores: dict[str, float | None] = {}
    errors = []
    for row in rows:
        key, text = row["datapoint_id"], row["ctext"]
        try:
            if not candidate_module.applies(text):
                scores[key] = None
                continue
            value = candidate_module.score(text)
            scores[key] = None if value is None else float(value)
        except Exception as exc:  # historical runner converted item errors to NA
            scores[key] = None
            errors.append({"datapoint_id": key, "error_type": type(exc).__name__})
    return scores, errors


def _split_ids(rows: Sequence[Mapping[str, str]]) -> tuple[set[str], set[str]]:
    identifiers = sorted(row["datapoint_id"] for row in rows)
    random.Random(7).shuffle(identifiers)
    return set(identifiers[:150]), set(identifiers[150:])


def _midranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        rank = (index + stop) / 2 + 1
        for position in range(index, stop + 1):
            ranks[order[position]] = rank
        index = stop + 1
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left_ranks, right_ranks = _midranks(left), _midranks(right)
    left_mean, right_mean = statistics.mean(left_ranks), statistics.mean(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks)
    )
    left_norm = math.sqrt(sum((value - left_mean) ** 2 for value in left_ranks))
    right_norm = math.sqrt(sum((value - right_mean) ** 2 for value in right_ranks))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else float("nan")


def _load_a104_judge(path: Path) -> dict[str, float]:
    """Load the frozen two-pass LLM reconstruction reference only after scoring."""

    pass1: dict[str, int] = {}
    pass2: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("aspect_id") != "a104" or not isinstance(row.get("score"), int):
            continue
        if row.get("channel") == "pass1":
            pass1[row["datapoint_id"]] = row["score"]
        elif row.get("channel") == "pass2":
            pass2[row["datapoint_id"]] = row["score"]
    common = set(pass1) & set(pass2)
    if len(common) < 30:
        raise SensitivityInputError("a104 two-pass reference has insufficient support")
    return {key: (pass1[key] + pass2[key]) / 20.0 for key in common}


def _assert_expected_readout(readout: Mapping[str, float | int]) -> None:
    for key, expected in EXPECTED_READOUT.items():
        observed = readout.get(key)
        if isinstance(expected, float):
            if not isinstance(observed, (int, float)) or not math.isclose(
                float(observed), expected, rel_tol=0.0, abs_tol=5e-15
            ):
                raise SensitivityInputError(
                    f"readout drift for {key}: {observed!r} != {expected!r}"
                )
        elif observed != expected:
            raise SensitivityInputError(
                f"readout drift for {key}: {observed!r} != {expected!r}"
            )


def build_result() -> dict:
    # No outcome-bearing reference file is opened in this phase.  The source
    # item serialization contains merge judgement, but only its id/ctext
    # projection is indexed (the same bounded caveat recorded by V4).
    measurement_receipts = _verify_paths(MEASUREMENT_INPUT_SHA256)
    dependency_versions = {
        name: importlib.metadata.version(name) for name in EXPECTED_DEPENDENCY_VERSIONS
    }
    if dependency_versions != EXPECTED_DEPENDENCY_VERSIONS:
        raise SensitivityInputError(
            f"dependency-version drift: {dependency_versions!r}"
        )

    manifest = _load_json(MANIFEST)
    representation = manifest.get("representation", {})
    if representation.get("max_chars") != 4000 or not representation.get(
        "same_bytes_required_for_prompt_and_code"
    ):
        raise SensitivityInputError("hierarchy representation contract changed")

    source_items = _validate_source_items(_load_json(ITEMS))
    hierarchy_train = _validate_hierarchy_items(
        _load_json(TRAIN_ITEMS), expected_prefix="train_"
    )
    hierarchy_heldout = _validate_hierarchy_items(
        _load_json(HELDOUT_ITEMS), expected_prefix="heldout_"
    )
    hierarchy_all = [*hierarchy_train, *hierarchy_heldout]
    crosswalk = exact_prefix_crosswalk(
        source_items, hierarchy_all, max_chars=representation["max_chars"]
    )

    frozen_columns = _load_json(CODE_SCORES)
    frozen_head_tail = frozen_columns.get("a104_coded_checker")
    if not isinstance(frozen_head_tail, Mapping) or set(frozen_head_tail) != {
        row["datapoint_id"] for row in source_items
    }:
        raise SensitivityInputError("frozen a104 score map has the wrong item set")

    # Candidate scoring and exact score replay happen before RESULTS is opened.
    from methods.existing_metrics_runner.coded.metrics import a104_test_presence

    prefix_rows = [
        {
            "datapoint_id": row["datapoint_id"],
            "ctext": row["ctext"][: representation["max_chars"]],
        }
        for row in source_items
    ]
    live_parser = a104_test_presence.parse_diff_added_by_file
    with bind_frozen_parser(a104_test_presence):
        replay_head_tail, head_tail_errors = _score_program(
            a104_test_presence, source_items
        )
        prefix_scores, prefix_errors = _score_program(a104_test_presence, prefix_rows)
    if a104_test_presence.parse_diff_added_by_file is not live_parser:
        raise SensitivityInputError("live parser global was not restored")

    replay_mismatches = sum(
        replay_head_tail.get(key) != frozen_head_tail.get(key)
        for key in set(replay_head_tail) | set(frozen_head_tail)
    )
    if replay_mismatches:
        raise SensitivityInputError(
            f"frozen parser failed exact a104 replay on {replay_mismatches}/250 rows"
        )

    train_ids, heldout_ids = _split_ids(source_items)
    if (len(train_ids), len(heldout_ids)) != (150, 100):
        raise SensitivityInputError("V4 split reconstruction failed")

    measurement_freeze_sha256 = _canonical_sha256(
        {
            "criterion": "a104",
            "candidate_sha256": MEASUREMENT_INPUT_SHA256[CANDIDATE],
            "parser_source_sha256": EXPECTED_FROZEN_PARSER_SOURCE_SHA256,
            "crosswalk": crosswalk,
            "head_tail_scores": frozen_head_tail,
            "prefix4000_scores": prefix_scores,
            "train_ids": sorted(train_ids),
            "heldout_ids": sorted(heldout_ids),
        }
    )
    if measurement_freeze_sha256 != EXPECTED_MEASUREMENT_FREEZE_SHA256:
        raise SensitivityInputError(
            "measurement freeze drift: "
            f"{measurement_freeze_sha256} != {EXPECTED_MEASUREMENT_FREEZE_SHA256}"
        )

    # The candidate, projection, scores, and split are now fixed.  Only now
    # open/hash the outcome-bearing reference files, then parse their values.
    reference_receipts = _verify_paths(REFERENCE_INPUT_SHA256)
    audit = _load_json(V3_AUDIT)
    if (
        audit.get("source_hashes_at_audit", {}).get(
            "deep_checker_sandbox_dependency"
        )
        != EXPECTED_FROZEN_SANDBOX_SHA256
    ):
        raise SensitivityInputError("V3 audit does not bind the expected sandbox")
    v4 = _load_json(V4)
    judge = _load_a104_judge(RESULTS)
    source_receipts = {**measurement_receipts, **reference_receipts}
    common_heldout = sorted(
        key
        for key in heldout_ids
        if key in judge
        and frozen_head_tail.get(key) is not None
        and prefix_scores.get(key) is not None
    )
    full_head_tail_heldout = sorted(
        key
        for key in heldout_ids
        if key in judge and frozen_head_tail.get(key) is not None
    )
    prefix_heldout = sorted(
        key
        for key in heldout_ids
        if key in judge and prefix_scores.get(key) is not None
    )
    all_common_scored = sorted(
        key
        for key in frozen_head_tail
        if frozen_head_tail.get(key) is not None and prefix_scores.get(key) is not None
    )

    head_tail_common_rho = spearman(
        [frozen_head_tail[key] for key in common_heldout],
        [judge[key] for key in common_heldout],
    )
    prefix_common_rho = spearman(
        [prefix_scores[key] for key in common_heldout],
        [judge[key] for key in common_heldout],
    )
    vector_rho = spearman(
        [frozen_head_tail[key] for key in common_heldout],
        [prefix_scores[key] for key in common_heldout],
    )
    full_head_tail_rho = spearman(
        [frozen_head_tail[key] for key in full_head_tail_heldout],
        [judge[key] for key in full_head_tail_heldout],
    )

    readout = {
        "frozen_replay_mismatches": replay_mismatches,
        "prefix_crosswalk_n": len(crosswalk),
        "common_heldout_n": len(common_heldout),
        "head_tail_rho_on_common": head_tail_common_rho,
        "prefix_rho_on_common": prefix_common_rho,
        "delta_prefix_minus_head_tail": prefix_common_rho - head_tail_common_rho,
        "program_vector_rho": vector_rho,
        "applicability_status_changes": sum(
            (frozen_head_tail.get(key) is None) != (prefix_scores.get(key) is None)
            for key in frozen_head_tail
        ),
        "value_changes_on_common_scored": sum(
            frozen_head_tail[key] != prefix_scores[key]
            for key in all_common_scored
        ),
    }
    _assert_expected_readout(readout)

    v4_headline = v4.get("heldout_rhos_common_intersection", {}).get(
        "preexisting_deep_coded_checker"
    )
    if len(full_head_tail_heldout) != v4.get("common_heldout_n") or not math.isclose(
        full_head_tail_rho, v4_headline, rel_tol=0.0, abs_tol=5e-15
    ):
        raise SensitivityInputError("full-support head/tail replay does not recover V4")

    head_tail_scored = {key for key, value in frozen_head_tail.items() if value is not None}
    prefix_scored = {key for key, value in prefix_scores.items() if value is not None}
    capped = sum(len(row["ctext"]) == representation["max_chars"] for row in hierarchy_all)
    source_head_tail_capped = sum(len(row["ctext"]) == 7507 for row in source_items)

    return {
        "schema": SCHEMA,
        "status": "complete_posthoc_exploratory",
        "task": "code-review",
        "criterion": "a104",
        "objective": (
            "unsupervised one-sided sensitivity of frozen code-to-LLM "
            "reconstruction agreement to the code input projection"
        ),
        "design_scope": "post_hoc_representation_sensitivity_no_new_gate",
        "claim_status": {
            "post_hoc": True,
            "exploratory": True,
            "new_gate": False,
            "program_selection": "none_single_frozen_candidate",
            "criterion_selected_for_sensitivity_post_outcome": True,
            "supersedes_prior_artifact": False,
            "canonical_v4_modified": False,
        },
        "compute": {
            "cpu_only": True,
            "model_or_api_calls": False,
            "gpu_used": False,
            "repository_or_under_review_test_execution": False,
        },
        "sources": {
            path: {"sha256": digest}
            for path, digest in sorted(source_receipts.items())
        },
        "frozen_dependency_binding": {
            "origin_sandbox_sha256": EXPECTED_FROZEN_SANDBOX_SHA256,
            "origin_git_provenance": FROZEN_SANDBOX_GIT_PROVENANCE,
            "vendored_parser_source_sha256": EXPECTED_FROZEN_PARSER_SOURCE_SHA256,
            "binding": (
                "hash-verified historical parser source compiled in memory and bound "
                "only to the a104 module global inside a restoring context manager"
            ),
            "live_sandbox_file_modified_by_instrument": False,
            "live_parser_global_restored": True,
            "dependency_versions": dependency_versions,
        },
        "blindness_and_reference_order": {
            "candidate_input_fields": ["datapoint_id", "ctext"],
            "serialized_source_contains_merge_judgement": True,
            "merge_judgement_indexed_by_candidate": False,
            "classification": "label_unreferenced_not_label_inaccessible",
            "candidate_scores_and_split_frozen_before_llm_reference_values_parsed": True,
            "measurement_freeze_sha256": measurement_freeze_sha256,
            "reference_files_opened_only_after_measurement_freeze": True,
            "outcomes_used_for_program_selection_or_tuning": False,
            "external_ground_truth_used": False,
            "reference": "pre-existing two-pass LLM judgment used by sealed a104 V4",
            "reference_representation": "historical_head_tail_ctext",
            "prefix_candidate_matches_reference_input": False,
            "prefix_arm": "one_sided_representation_mismatch_sensitivity",
            "direct_same_input_prefix_prompt_code_test": False,
        },
        "crosswalk": {
            "source_rows": len(source_items),
            "hierarchy_rows": len(hierarchy_all),
            "exact_unique_prefix_matches": len(crosswalk),
            "prefix_max_chars": representation["max_chars"],
            "hierarchy_rows_at_cap": capped,
            "hierarchy_fraction_at_cap": capped / len(hierarchy_all),
            "source_head_tail_rows_at_7507_chars": source_head_tail_capped,
            "same_bytes_prompt_code_contract": representation[
                "same_bytes_required_for_prompt_and_code"
            ],
            "matching_rule": (
                "exact Python string equality; therefore exact UTF-8 bytes when encoded"
            ),
            "v4_split_reused_for_statistics": {"seed": 7, "train": 150, "heldout": 100},
            "hierarchy_split_used_for_statistics": False,
            "split_note": (
                "hierarchy bytes are crosswalked exactly, but statistics retain V4's "
                "frozen 150/100 split rather than the hierarchy's distinct 125/125 split"
            ),
        },
        "frozen_replay": {
            "n_items": len(source_items),
            "score_mismatches": replay_mismatches,
            "head_tail_item_errors": head_tail_errors,
            "requirement": "exact 0/250 match to frozen a104_coded_checker",
        },
        "score_support_all_250": {
            "head_tail_scored": len(head_tail_scored),
            "prefix_scored": len(prefix_scored),
            "common_scored": len(all_common_scored),
            "head_tail_only": len(head_tail_scored - prefix_scored),
            "prefix_only": len(prefix_scored - head_tail_scored),
            "both_missing": len(frozen_head_tail) - len(head_tail_scored | prefix_scored),
            "applicability_status_changes": readout[
                "applicability_status_changes"
            ],
            "value_changes_on_common_scored": readout[
                "value_changes_on_common_scored"
            ],
            "prefix_item_errors": prefix_errors,
        },
        "heldout_readout": {
            "common_support_n": len(common_heldout),
            "head_tail_rho_on_common": head_tail_common_rho,
            "prefix4000_rho_on_common": prefix_common_rho,
            "delta_prefix_minus_head_tail": prefix_common_rho - head_tail_common_rho,
            "head_tail_prefix_program_vector_rho": vector_rho,
            "historical_head_tail_own_support": {
                "n": len(full_head_tail_heldout),
                "rho": full_head_tail_rho,
                "canonical_v4_rho": v4_headline,
            },
            "prefix4000_own_support": {
                "n": len(prefix_heldout),
                "rho": spearman(
                    [prefix_scores[key] for key in prefix_heldout],
                    [judge[key] for key in prefix_heldout],
                ),
            },
        },
        "interpretation": {
            "result": (
                "On this fixed 93-row V4-heldout common support, the observed a104 "
                "reconstruction rho is lower for the first-4000-character prefix "
                "than for the historical head/tail representation."
            ),
            "isomorphism": (
                "Future prompt/code byte equality is necessary but does not make a "
                "truncated projection complete for the construct. Here the reference "
                "remains on historical head/tail ctext while only the code arm receives "
                "the prefix, so this measures one-sided representation sensitivity."
            ),
            "not_permitted": [
                "new confirmatory gate or promotion",
                "whole-code-review codability estimate",
                "direct prompt-articulability versus code-verifiability comparison",
                "direct same-input prefix prompt/code isomorphism test",
                "isomorphism certification from rho alone",
                "external-ground-truth or correctness claim",
                "tacitness claim from any failure",
            ],
            "next_use": (
                "treat prefix versus head/tail/full-diff representation as a frozen "
                "factor in future untouched coding criteria before adding more h0s"
            ),
        },
    }


def expected_artifact() -> bytes:
    return (
        json.dumps(build_result(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = expected_artifact()
    if args.check:
        if not args.out.exists() or args.out.read_bytes() != payload:
            raise SystemExit(f"artifact mismatch: {args.out}")
        print(f"ok {args.out}")
        return 0
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(payload)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
