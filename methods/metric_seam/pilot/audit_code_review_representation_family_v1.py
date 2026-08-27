#!/usr/bin/env python3
"""Audit frozen code-review program outputs across three text projections.

This developmental instrument is deliberately outcome/reference inaccessible.
It recovers the historical head/tail input only from prompt *requests*, maps
those request IDs to local raw diff paths without a label table, and refuses
to read the outcome-bearing active ``items.json`` file.  No prompt response,
reconstruction result, model, API, accelerator, or promotion gate is used.

The first-4,000-character arm must reproduce every frozen train and heldout
program row before any representation-sensitivity readout is emitted.  The
historical diff parser is compiled from fixed local source and bound in worker
memory; the live, user-modified sandbox module is never edited.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import whatthepatch


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"
TASK = ROOT / "outputs/metric_seam_pilot/tasks/code_review"
ITEM_ROOT = BASE / "items_v2/code-review"
RAW_ROOT = ROOT / "datasets/code-review/pr_test_execution/batch_runs"

CONTRACT = BASE / "code_review_representation_family_analysis_contract_v1.json"
PROMPT_REQUESTS = TASK / "prompts.jsonl"
MANIFEST = ITEM_ROOT / "manifest.json"
TRAIN_ITEMS = ITEM_ROOT / "compiler_train.json"
HELDOUT_ITEMS = ITEM_ROOT / "sealed_heldout.json"
FIDELITY = BASE / "code_review_construct_fidelity_v2.json"
CROSS_AUDIT = BASE / "code_review_construct_fidelity_independent_cross_audit_v1.json"
TRAIN_GATE = BASE / "code_review_train_gate_v1.json"
HELDOUT_READINESS = BASE / "code_review_heldout_readiness_v1.json"
TRAIN_EXECUTION = BASE / "code_review_train_execution_v2.json"
HELDOUT_EXECUTION = BASE / "code_review_heldout_execution_v1.json"
CORRECTED_FUNNEL = BASE / "code_review_corrected_funnel_v1.json"
DEFAULT_OUT = BASE / "code_review_representation_family_sensitivity_v1.json"

SCHEMA = "metric-seam.code-review-representation-family-sensitivity.v1"
CONTRACT_SCHEMA = "metric-seam.code-review-representation-family-analysis-contract.v1"
EXPECTED_INPUT_SHA256 = {
    CONTRACT: "dd9b264a5f8294f706ee8ffade8217f9f567ab4301f292aaa694549253421f08",
    PROMPT_REQUESTS: "936b8660d9e8630001ccc612671bd8f5622cdba4205ceebf35d47fe33083456a",
    MANIFEST: "a31803c3474147247b0df75d7b8e5c81bade7c83af433bf7a9a20d0b98fe05d5",
    TRAIN_ITEMS: "0e6f68619dd1405b15ec99899edea50b539d2f12663fc7c0ff35abd2e5038167",
    HELDOUT_ITEMS: "bda8ad98f69f79ff41d3fc0261d0b66c678c24945ce9b35b3b1433836b6584f4",
    FIDELITY: "d5e9118de1147877da31b76bec907c037a7e08be589f653a1e22f3e89929078f",
    CROSS_AUDIT: "96333b43d43bc0c69edc3e3c3aaa8b442012144248710c9cb3952b3e5c5c5dc7",
    TRAIN_GATE: "c280b6951c3618abe99e64b7f24afee112e318e128c5308ffa578d7c4689eab3",
    HELDOUT_READINESS: "0472519b45a4c4e0140a142a4474ac25bdc55d0b49e804e4d3c54d7fbb980463",
    TRAIN_EXECUTION: "b18f34254319a35d140c38cddadd85cee09f83f800272d07aede8b7a9dd18054",
    HELDOUT_EXECUTION: "9b0042a92bd25249743d5f915d61c658f968e83a91dfc7e1bbc49fc01e70a4c7",
    CORRECTED_FUNNEL: "b6fe4e80d9d4c52fff7a66e05f15b11c651c2246f0cddd5928af13d530e38f66",
}

# This is the exact corrected parser function used by the canonical hierarchy
# train-v2 and heldout executions.  The incident record invalidates train-v1,
# which used the earlier lazy-generator parser.  The corrected function is
# frozen here so the live, user-modified sandbox.py never needs to be changed.
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
    # ``whatthepatch.parse_patch`` is lazy: malformed/binary blocks can raise
    # while iterating, outside a try around generator construction.  Parse each
    # file block independently so a binary attachment cannot discard valid text
    # hunks elsewhere in the same PR.
    chunks = diff_text[idx:].split("\\ndiff --git ")
    blocks = [chunks[0], *("diff --git " + chunk for chunk in chunks[1:])]
    for block in blocks:
        if "\\nGIT binary patch\\n" in block or "\\nBinary files " in block:
            continue
        try:
            diffs = list(whatthepatch.parse_patch(block))
        except Exception:
            continue
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
EXPECTED_FROZEN_PARSER_SOURCE_SHA256 = (
    "1683190826205cea9ffafb44c192e01191d983e21f5425a6c1fee67f56e26494"
)

PROJECTIONS = (
    "P0_prefix4000",
    "P1_head5000_tail2500",
    "P2_raw_diff_capped300k",
)
PAIRS = (
    ("P0_prefix4000", "P1_head5000_tail2500"),
    ("P1_head5000_tail2500", "P2_raw_diff_capped300k"),
    ("P0_prefix4000", "P2_raw_diff_capped300k"),
)
FORBIDDEN_ACTIVE_ITEMS = TASK / "items.json"
PROMPT_ROW_FIELDS = {"channel", "aspect_id", "datapoint_id", "prompt"}


class FamilyAuditError(ValueError):
    """Raised when a frozen input, crosswalk, or replay contract fails."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_frozen_inputs() -> dict[str, str]:
    observed = {}
    for path, expected in EXPECTED_INPUT_SHA256.items():
        actual = sha256(path)
        if actual != expected:
            raise FamilyAuditError(
                f"frozen input drift: {path.relative_to(ROOT)} {actual} != {expected}"
            )
        observed[str(path.relative_to(ROOT))] = actual
    contract = _load_json(CONTRACT)
    if contract.get("schema") != CONTRACT_SCHEMA or contract.get("status") != (
        "frozen_developmental_before_family_execution"
    ):
        raise FamilyAuditError("analysis contract is not the frozen developmental contract")
    if FORBIDDEN_ACTIVE_ITEMS in EXPECTED_INPUT_SHA256:
        raise FamilyAuditError("outcome-bearing items.json entered the input allowlist")
    return observed


def build_frozen_parser():
    observed = hashlib.sha256(FROZEN_PARSER_SOURCE.encode("utf-8")).hexdigest()
    if observed != EXPECTED_FROZEN_PARSER_SOURCE_SHA256:
        raise FamilyAuditError("vendored historical parser source drifted")
    namespace = {"Dict": dict, "List": list, "whatthepatch": whatthepatch}
    exec(  # noqa: S102 - fixed, hash-checked repository source only
        compile(FROZEN_PARSER_SOURCE, "<frozen-family-parser-v1>", "exec"),
        namespace,
    )
    parser = namespace.get("parse_diff_added_by_file")
    if not callable(parser):
        raise FamilyAuditError("historical parser did not compile")
    return parser


def _extract_document(prompt: str) -> str:
    opening, closing = "<document>\n", "\n</document>"
    if prompt.count("<document>") != 1 or prompt.count("</document>") != 1:
        raise FamilyAuditError("prompt request does not have exactly one document block")
    try:
        return prompt.split(opening, 1)[1].split(closing, 1)[0]
    except IndexError as exc:
        raise FamilyAuditError("prompt document delimiters drifted") from exc


def load_head_tail_requests(path: Path = PROMPT_REQUESTS) -> tuple[dict[str, str], dict]:
    """Recover P1 from request serialization without reading any response."""

    by_pair: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    channels = Counter()
    aspects = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if set(row) != PROMPT_ROW_FIELDS:
                raise FamilyAuditError(f"prompt request row {line_number} fields drifted")
            channel = row["channel"]
            channels[channel] += 1
            if channel not in {"pass1", "pass2"}:
                continue
            aspect_id, datapoint_id = row["aspect_id"], row["datapoint_id"]
            if not all(isinstance(value, str) and value for value in (
                aspect_id, datapoint_id, row["prompt"]
            )):
                raise FamilyAuditError(f"prompt request row {line_number} is malformed")
            key = (datapoint_id, aspect_id)
            if channel in by_pair[key]:
                raise FamilyAuditError(f"duplicate {channel} request for {key}")
            by_pair[key][channel] = _extract_document(row["prompt"])
            aspects.add(aspect_id)

    if channels != Counter({"pass1": 4500, "pass2": 4500, "scope": 250}):
        raise FamilyAuditError(f"prompt channel inventory drifted: {dict(channels)}")
    if len(aspects) != 18 or len(by_pair) != 4500:
        raise FamilyAuditError("prompt request aspect/item inventory drifted")

    per_id: dict[str, set[str]] = defaultdict(set)
    for key, serialized in by_pair.items():
        if set(serialized) != {"pass1", "pass2"}:
            raise FamilyAuditError(f"request channels incomplete for {key}")
        if serialized["pass1"] != serialized["pass2"]:
            raise FamilyAuditError(f"request-channel document mismatch for {key}")
        per_id[key[0]].add(serialized["pass1"])
    if len(per_id) != 250 or any(len(documents) != 1 for documents in per_id.values()):
        raise FamilyAuditError("aspects do not serialize one identical P1 per datapoint")
    documents = {key: next(iter(values)) for key, values in per_id.items()}
    return documents, {
        "n_request_rows": sum(channels.values()),
        "channel_counts": dict(sorted(channels.items())),
        "n_aspects_two_pass": len(aspects),
        "n_datapoints": len(documents),
        "pass1_pass2_byte_agreement_pairs": len(by_pair),
        "unique_documents_per_datapoint": 1,
        "response_fields_loaded": False,
    }


def historical_canonical(text: str) -> str:
    if len(text) <= 5000 + 2500 + 500:
        return text
    return text[:5000] + "\n[...]\n" + text[-2500:]


def read_raw_capped(path: Path, cap: int = 300_000) -> str:
    size = path.stat().st_size
    if size <= cap:
        return path.open(encoding="utf-8", errors="replace").read()
    half = cap // 2
    with path.open("rb") as handle:
        head = handle.read(half)
        handle.seek(max(size - half, half), os.SEEK_SET)
        tail = handle.read()
    return (
        head.decode("utf-8", "replace")
        + "\n[...RAW-TRUNCATED...]\n"
        + tail.decode("utf-8", "replace")
    )


def resolve_raw_diffs(datapoint_ids: set[str], raw_root: Path = RAW_ROOT) -> dict[str, Path]:
    hits: dict[str, list[Path]] = defaultdict(list)
    pattern = re.compile(r"^pr_(\d+)\.diff$")
    for path in raw_root.glob("*/diffs/pr_*.diff"):
        match = pattern.match(path.name)
        if not match:
            continue
        repo, pr_number = path.parents[1].name, int(match.group(1))
        digest = hashlib.sha256(f"{repo}#{pr_number}".encode("utf-8")).hexdigest()
        datapoint_id = "crb" + digest[:10]
        if datapoint_id in datapoint_ids:
            hits[datapoint_id].append(path)
    bad = {key: values for key, values in hits.items() if len(values) != 1}
    missing = sorted(datapoint_ids - set(hits))
    if bad or missing or len(hits) != 250:
        raise FamilyAuditError(
            f"P2 path crosswalk is not one-to-one: bad={len(bad)}, missing={len(missing)}"
        )
    return {key: values[0] for key, values in hits.items()}


def _validate_p0_rows(rows: object, prefix: str) -> list[dict[str, str]]:
    if not isinstance(rows, list) or len(rows) != 125:
        raise FamilyAuditError(f"{prefix} hierarchy arm is not a 125-row list")
    result = []
    for index, row in enumerate(rows, 1):
        expected_key = f"{prefix}_{index:04d}"
        if (
            not isinstance(row, Mapping)
            or set(row) != {"item_key", "ctext"}
            or row.get("item_key") != expected_key
            or not isinstance(row.get("ctext"), str)
            or not row["ctext"]
            or len(row["ctext"]) > 4000
        ):
            raise FamilyAuditError(f"invalid P0 row {expected_key}")
        result.append(dict(row))
    return result


def build_representations() -> tuple[dict[str, list[dict[str, str]]], dict]:
    manifest = _load_json(MANIFEST)
    representation = manifest.get("representation", {})
    selection = manifest.get("selection", {})
    if (
        manifest.get("schema") != "metric-seam.hierarchy-shared-items.v1"
        or manifest.get("task") != "code-review"
        or representation.get("max_chars") != 4000
        or representation.get("projection")
        != "source_text[:max_chars] before exact deduplication"
        or selection.get("outcome_or_reference_values_used") is not False
    ):
        raise FamilyAuditError("P0 hierarchy manifest drifted")
    train = _validate_p0_rows(_load_json(TRAIN_ITEMS), "train")
    heldout = _validate_p0_rows(_load_json(HELDOUT_ITEMS), "heldout")
    p0 = train + heldout

    head_tail, prompt_receipt = load_head_tail_requests()
    prefix_to_ids: dict[str, list[str]] = defaultdict(list)
    for datapoint_id, text in head_tail.items():
        prefix_to_ids[text[:4000]].append(datapoint_id)
    item_to_datapoint = {}
    for row in p0:
        matches = prefix_to_ids.get(row["ctext"], [])
        if len(matches) != 1:
            raise FamilyAuditError(
                f"P0/P1 prefix crosswalk is not one-to-one for {row['item_key']}"
            )
        item_to_datapoint[row["item_key"]] = matches[0]
    if len(set(item_to_datapoint.values())) != 250:
        raise FamilyAuditError("P0/P1 crosswalk does not cover 250 distinct datapoints")

    raw_paths = resolve_raw_diffs(set(head_tail))
    raw_text = {key: read_raw_capped(path) for key, path in raw_paths.items()}
    canonical_mismatches = [
        key for key in sorted(head_tail)
        if historical_canonical(raw_text[key]) != head_tail[key]
    ]
    if canonical_mismatches:
        raise FamilyAuditError(
            f"P2 canonicalization failed for {len(canonical_mismatches)}/250 rows"
        )

    p1 = [
        {"item_key": row["item_key"], "ctext": head_tail[item_to_datapoint[row["item_key"]]]}
        for row in p0
    ]
    p2 = [
        {"item_key": row["item_key"], "ctext": raw_text[item_to_datapoint[row["item_key"]]]}
        for row in p0
    ]
    arms = dict(zip(PROJECTIONS, (p0, p1, p2), strict=True))
    lengths = {
        projection: {
            "min_chars": min(len(row["ctext"]) for row in rows),
            "median_chars": sorted(len(row["ctext"]) for row in rows)[125],
            "max_chars": max(len(row["ctext"]) for row in rows),
            "n_over_4000_chars": sum(len(row["ctext"]) > 4000 for row in rows),
        }
        for projection, rows in arms.items()
    }
    return arms, {
        "prompt_request_serialization": prompt_receipt,
        "P0_P1_exact_prefix_crosswalk_n": 250,
        "P1_P2_exact_canonicalization_n": 250,
        "P2_local_path_crosswalk_n": 250,
        "outcome_bearing_items_json_loaded": False,
        "representation_lengths": lengths,
        "representation_vector_sha256": {
            key: canonical_digest(rows) for key, rows in arms.items()
        },
        "raw_path_identity_sha256": canonical_digest({
            item_key: str(raw_paths[datapoint].relative_to(ROOT))
            for item_key, datapoint in sorted(item_to_datapoint.items())
        }),
    }


def _identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["aspect_id"]), str(row["source_path"]), str(row["source_sha256"])
    )


def load_populations() -> tuple[list[dict], list[dict], dict[str, list[dict]], dict]:
    fidelity = _load_json(FIDELITY)
    cross = _load_json(CROSS_AUDIT)
    train_gate = _load_json(TRAIN_GATE)
    readiness = _load_json(HELDOUT_READINESS)
    corrected = _load_json(CORRECTED_FUNNEL)
    if fidelity.get("schema") != "metric-seam.code-review-construct-fidelity-merged.v1":
        raise FamilyAuditError("construct-fidelity source drifted")
    if cross.get("schema") != (
        "metric-seam.code-review-construct-fidelity-independent-cross-audit.v1"
    ):
        raise FamilyAuditError("independent cross-audit source drifted")
    if train_gate.get("selection_basis") != "compiler_train_outputs_only":
        raise FamilyAuditError("secondary fleet was not selected on train outputs only")
    for field in ("reference_values_used", "outcome_labels_used", "heldout_items_or_outputs_used"):
        if train_gate.get(field) is not False:
            raise FamilyAuditError(f"train selection violates {field}")
    for field in ("reference_values_used", "outcome_labels_used", "prompt_outputs_used"):
        if readiness.get(field) is not False:
            raise FamilyAuditError(f"heldout readiness violates {field}")

    audit_rows = {row["cell_id"]: row for row in fidelity["rows"]}
    after = {
        review["cell_id"]: review
        for review in cross["reviews"]
    }
    secondary = [dict(row) for row in train_gate["selected_programs"]]
    if len(secondary) != 16 or len({_identity(row) for row in secondary}) != 16:
        raise FamilyAuditError("secondary train-selected population is not 16 unique programs")

    primary = []
    criterion_rows: dict[str, list[dict]] = defaultdict(list)
    for program in readiness["confirmatory_programs"]:
        surviving = []
        for cell_id in program["cell_ids"]:
            base = audit_rows[cell_id]
            review = after.get(cell_id)
            if review is None or review.get("before") != {
                key: base[key]
                for key in ("verdict", "scope", "eligible_for_relation_local_execution", "audited_depth")
            }:
                raise FamilyAuditError(f"cross-audit state drift for {cell_id}")
            state = review["after"]
            if state.get("eligible_for_relation_local_execution") is not True:
                continue
            candidate = base["candidate"]
            if _identity(program) != (
                candidate["aspect_id"], candidate["source_path"], candidate["source_sha256"]
            ):
                raise FamilyAuditError(f"candidate identity drift for {cell_id}")
            relation = {
                "cell_id": cell_id,
                "level": base["level"],
                "metric_name": base["metric_name"],
                "candidate_aspect_id": candidate["aspect_id"],
                "matched_relation_depth": state["audited_depth"],
                "construct_fidelity_verdict": state["verdict"],
                "scope": state["scope"],
                "requested_relation": base["requested_relation"],
                "implemented_relations": base["implemented_relations"],
            }
            surviving.append(cell_id)
            criterion_rows[candidate["aspect_id"]].append(relation)
        if surviving:
            selected = dict(program)
            selected["cell_ids"] = surviving
            primary.append(selected)

    if len(primary) != 10 or sum(map(len, criterion_rows.values())) != 18:
        raise FamilyAuditError("corrected primary population is not 10 programs / 18 mappings")
    if not {_identity(row) for row in primary} <= {_identity(row) for row in secondary}:
        raise FamilyAuditError("primary program family is not nested in the secondary family")
    corrected_stages = corrected.get("corrected_readout", {}).get("stages", {})
    if corrected_stages.get("heldout_confirmatory_reconstruction_evaluable", {}).get(
        "balanced_panel", {}
    ).get("n_positive") != 18:
        raise FamilyAuditError("corrected-funnel 18-mapping anchor drifted")
    return primary, secondary, criterion_rows, {
        "primary_unique_programs": len(primary),
        "primary_relation_mappings": sum(map(len, criterion_rows.values())),
        "secondary_unique_programs": len(secondary),
        "primary_is_subset_of_secondary": True,
        "selection_loaded_outcomes_or_references": False,
    }


def _worker_environment(home: Path) -> dict[str, str]:
    from methods.metric_seam.hierarchy_code_runner import worker_environment

    return worker_environment(home)


def _run_program_worker(
    program: Mapping[str, Any], items: Sequence[Mapping[str, str]], *, timeout: float
) -> dict:
    with tempfile.TemporaryDirectory(prefix="metric-seam-family-") as directory:
        home = Path(directory)
        out = home / "result.json"
        command = [
            sys.executable,
            "-m", "methods.metric_seam.pilot.audit_code_review_representation_family_v1",
            "worker",
            "--source-path", str(program["source_path"]),
            "--aspect-id", str(program["aspect_id"]),
            "--source-sha256", str(program["source_sha256"]),
            "--out", str(out),
        ]
        try:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=_worker_environment(home),
                input=json.dumps(list(items), ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise FamilyAuditError(f"program timeout: {program['aspect_id']}") from exc
        if process.returncode or not out.is_file():
            raise FamilyAuditError(
                f"program worker failed: {program['aspect_id']} exit={process.returncode}"
            )
        result = _load_json(out)
        if (
            result.get("aspect_id") != program["aspect_id"]
            or result.get("source_path") != program["source_path"]
            or result.get("worker_status") != "completed"
            or len(result.get("rows", [])) != len(items)
        ):
            raise FamilyAuditError(f"program worker result drift for {program['aspect_id']}")
        return result


def worker(source_path: str, aspect_id: str, source_sha256: str, out: Path) -> None:
    from methods.existing_metrics_runner.coded import sandbox
    from methods.metric_seam.hierarchy_code_runner import execute_one_program, validate_items

    items = validate_items(json.loads(sys.stdin.read()))
    sandbox.parse_diff_added_by_file = build_frozen_parser()
    result = execute_one_program(source_path, aspect_id, source_sha256, items)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _frozen_rows(path: Path, aspects: set[str]) -> dict[str, list[dict]]:
    payload = _load_json(path)
    rows = {
        program["aspect_id"]: program["rows"]
        for program in payload["programs"]
        if program["aspect_id"] in aspects
    }
    if set(rows) != aspects:
        raise FamilyAuditError(f"frozen execution lacks selected programs: {path.name}")
    return rows


def require_exact_p0_replay(outputs: Mapping[str, Mapping[str, Any]], aspects: set[str]) -> dict:
    train = _frozen_rows(TRAIN_EXECUTION, aspects)
    heldout = _frozen_rows(HELDOUT_EXECUTION, aspects)
    mismatch_counts = {}
    for aspect_id in sorted(aspects):
        observed = {row["item_key"]: row for row in outputs[aspect_id]["rows"]}
        expected = train[aspect_id] + heldout[aspect_id]
        mismatches = sum(observed.get(row["item_key"]) != row for row in expected)
        mismatch_counts[aspect_id] = mismatches
    total = sum(mismatch_counts.values())
    if total:
        raise FamilyAuditError(f"P0 does not exactly replay frozen outputs: {total} rows")
    return {
        "required_before_sensitivity_readout": True,
        "train_rows_exact": 16 * 125,
        "heldout_rows_exact": 16 * 125,
        "total_rows_exact": 16 * 250,
        "mismatch_counts_by_program": mismatch_counts,
        "total_mismatches": total,
    }


def compare_rows(left_rows: Sequence[Mapping], right_rows: Sequence[Mapping]) -> dict:
    left = {row["item_key"]: row for row in left_rows}
    right = {row["item_key"]: row for row in right_rows}
    if len(left) != 250 or set(left) != set(right):
        raise FamilyAuditError("representation outputs do not share a 250-row item universe")
    transitions = Counter()
    status_agree = applicability_changes = error_detail_changes = 0
    common_scored = exact_values = 0
    absolute_shifts = []
    exact_rows = 0
    for key in sorted(left):
        before, after = left[key], right[key]
        transitions[f"{before['status']} -> {after['status']}"] += 1
        status_agree += before["status"] == after["status"]
        applicability_changes += before.get("applies") != after.get("applies")
        error_detail_changes += before.get("error_type") != after.get("error_type")
        if before["status"] == after["status"] == "scored":
            common_scored += 1
            shift = abs(float(after["score"]) - float(before["score"]))
            absolute_shifts.append(shift)
            exact_values += shift == 0.0
        exact_rows += before == after
    if exact_rows == 250:
        sensitivity_class = "exact_stable"
    elif status_agree < 250 or applicability_changes or error_detail_changes:
        sensitivity_class = "status_or_applicability_sensitive"
    elif common_scored and exact_values < common_scored:
        sensitivity_class = "value_sensitive_only"
    else:
        sensitivity_class = "not_measured"
    return {
        "n_items": 250,
        "status_transition_counts": dict(sorted(transitions.items())),
        "n_exact_rows": exact_rows,
        "exact_row_agreement": exact_rows / 250,
        "n_status_agree": status_agree,
        "status_agreement": status_agree / 250,
        "n_applicability_changes": applicability_changes,
        "n_error_detail_changes": error_detail_changes,
        "n_common_scored": common_scored,
        "n_exact_values_on_common_scored": exact_values,
        "exact_value_agreement_on_common_scored": (
            exact_values / common_scored if common_scored else None
        ),
        "mean_absolute_score_shift_on_common_scored": (
            sum(absolute_shifts) / len(absolute_shifts) if absolute_shifts else None
        ),
        "max_absolute_score_shift_on_common_scored": (
            max(absolute_shifts) if absolute_shifts else None
        ),
        "sensitivity_class": sensitivity_class,
    }


def _macro(programs: Sequence[Mapping[str, Any]]) -> dict:
    result = {}
    for left, right in PAIRS:
        pair = f"{left} -> {right}"
        rows = [program["pairwise_sensitivity"][pair] for program in programs]
        value_rows = [
            row for row in rows
            if row["exact_value_agreement_on_common_scored"] is not None
        ]
        result[pair] = {
            "aggregation_unit": "unique_program",
            "n_programs": len(rows),
            "sensitivity_class_counts": dict(sorted(Counter(
                row["sensitivity_class"] for row in rows
            ).items())),
            "mean_program_status_agreement": sum(
                row["status_agreement"] for row in rows
            ) / len(rows),
            "mean_program_exact_row_agreement": sum(
                row["exact_row_agreement"] for row in rows
            ) / len(rows),
            "mean_program_applicability_change_rate": sum(
                row["n_applicability_changes"] / row["n_items"] for row in rows
            ) / len(rows),
            "n_programs_with_common_scored_rows": len(value_rows),
            "mean_program_exact_value_agreement_on_common_scored": (
                sum(row["exact_value_agreement_on_common_scored"] for row in value_rows)
                / len(value_rows) if value_rows else None
            ),
            "pooled_criterion_mapping_estimate_emitted": False,
        }
    return result


def _run_projection(
    projection: str,
    programs: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, str]],
    *,
    timeout: float,
    max_workers: int = 4,
) -> dict[str, dict]:
    """Run at most four isolated workers and return program-order results."""

    if not 1 <= max_workers <= 4:
        raise FamilyAuditError("representation audit permits 1-4 CPU workers")
    completed: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_program_worker, program, items, timeout=timeout): program
            for program in programs
        }
        for count, future in enumerate(as_completed(futures), 1):
            program = futures[future]
            aspect_id = str(program["aspect_id"])
            completed[aspect_id] = future.result()
            print(
                f"[{projection}] completed {count:02d}/{len(programs)} {aspect_id}",
                flush=True,
            )
    return {
        str(program["aspect_id"]): completed[str(program["aspect_id"])]
        for program in programs
    }


def build(*, timeout: float = 900.0) -> dict:
    input_hashes = verify_frozen_inputs()
    live_sandbox = ROOT / "methods/existing_metrics_runner/coded/sandbox.py"
    live_sandbox_sha_before = sha256(live_sandbox)
    arms, representation_receipt = build_representations()
    primary, secondary, criterion_by_aspect, population_receipt = load_populations()
    primary_aspects = {row["aspect_id"] for row in primary}

    outputs: dict[str, dict[str, dict]] = {
        "P0_prefix4000": _run_projection(
            "P0_prefix4000", secondary, arms["P0_prefix4000"], timeout=timeout
        )
    }
    replay = require_exact_p0_replay(outputs["P0_prefix4000"], {
        row["aspect_id"] for row in secondary
    })
    for projection in PROJECTIONS[1:]:
        outputs[projection] = _run_projection(
            projection, secondary, arms[projection], timeout=timeout
        )
    if sha256(live_sandbox) != live_sandbox_sha_before:
        raise FamilyAuditError("live sandbox.py changed during the audit")

    secondary_results = []
    for program in secondary:
        aspect_id = program["aspect_id"]
        source = ROOT / program["source_path"]
        if sha256(source) != program["source_sha256"]:
            raise FamilyAuditError(f"candidate source drift for {aspect_id}")
        pairwise = {}
        for left, right in PAIRS:
            pairwise[f"{left} -> {right}"] = compare_rows(
                outputs[left][aspect_id]["rows"], outputs[right][aspect_id]["rows"]
            )
        secondary_results.append({
            "aspect_id": aspect_id,
            "source_path": program["source_path"],
            "primary_corrected_family": aspect_id in primary_aspects,
            "primary_relation_mapping_count": len(criterion_by_aspect.get(aspect_id, [])),
            "primary_matched_relation_depths": sorted({
                row["matched_relation_depth"] for row in criterion_by_aspect.get(aspect_id, [])
            }),
            "projection_measurement_status": {
                projection: outputs[projection][aspect_id]["measurement_status"]
                for projection in PROJECTIONS
            },
            "projection_status_counts": {
                projection: outputs[projection][aspect_id]["summary"]["status_counts"]
                for projection in PROJECTIONS
            },
            "pairwise_sensitivity": pairwise,
        })
    primary_results = [
        row for row in secondary_results if row["primary_corrected_family"]
    ]
    if len(primary_results) != 10:
        raise FamilyAuditError("primary result join lost a program")

    program_by_aspect = {row["aspect_id"]: row for row in primary_results}
    criterion_join = []
    for aspect_id in sorted(criterion_by_aspect):
        for criterion in sorted(criterion_by_aspect[aspect_id], key=lambda row: row["cell_id"]):
            program = program_by_aspect[aspect_id]
            criterion_join.append({
                **criterion,
                "pairwise_sensitivity_class": {
                    pair: readout["sensitivity_class"]
                    for pair, readout in program["pairwise_sensitivity"].items()
                },
                "program_result_reference": aspect_id,
            })
    if len(criterion_join) != 18:
        raise FamilyAuditError("typed criterion join is not 18 rows")

    dependency_versions = {}
    for package in (
        "whatthepatch", "tree-sitter", "tree-sitter-python",
        "tree-sitter-javascript", "tree-sitter-typescript", "tree-sitter-java",
        "tree-sitter-go", "lizard", "radon", "bandit",
    ):
        try:
            dependency_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[package] = None

    return {
        "schema": SCHEMA,
        "status": "developmental_exploratory_family_audit_complete",
        "task": "code-review",
        "analysis_contract": {
            "path": str(CONTRACT.relative_to(ROOT)),
            "sha256": input_hashes[str(CONTRACT.relative_to(ROOT))],
            "status": "frozen_before_family_execution",
            "confirmatory_preregistration": False,
        },
        "blindness_and_channel": {
            "outcome_bearing_items_json_loaded": False,
            "prompt_requests_used_as_input_serialization_only": True,
            "prompt_responses_loaded": False,
            "llm_judgments_loaded": False,
            "outcomes_loaded": False,
            "references_loaded": False,
            "reconstruction_results_or_correlations_loaded": False,
            "models_or_apis_called": False,
            "external_supervision_used": False,
            "gpu_or_accelerator_used": False,
        },
        "source_bindings": input_hashes,
        "execution_environment": {
            "candidate_worker_process_isolated": True,
            "maximum_concurrent_cpu_workers": 4,
            "worker_filesystem_isolated": False,
            "worker_network_isolated": False,
            "credentials_inherited_by_worker": False,
            "accelerators_visible_to_worker": False,
            "canonical_corrected_parser_bound_in_worker_memory": True,
            "canonical_corrected_parser_source_sha256": (
                EXPECTED_FROZEN_PARSER_SOURCE_SHA256
            ),
            "canonical_corrected_sandbox_full_source_sha256": (
                "eaf948ca39ae4028142ec7dbf6195d25fe00dd394f1756367b4c20b7d5f2fab7"
            ),
            "live_sandbox_file_modified": False,
            "live_sandbox_sha256_observed": live_sandbox_sha_before,
            "dependency_versions": dependency_versions,
        },
        "representation_crosswalk": representation_receipt,
        "population": population_receipt,
        "P0_exact_frozen_replay": replay,
        "primary_program_macro": _macro(primary_results),
        "secondary_program_macro": _macro(secondary_results),
        "primary_program_results": primary_results,
        "secondary_program_results": secondary_results,
        "typed_primary_criterion_join": {
            "n_relation_mappings": len(criterion_join),
            "aggregation_performed": False,
            "reason": (
                "Multiple criterion mappings can share one program; rows are a typed join, "
                "not independent observations."
            ),
            "rows": criterion_join,
        },
        "interpretation": {
            "positive_scope": (
                "Representation robustness or sensitivity of frozen relation-local code "
                "measurements under three fixed projections."
            ),
            "not_measured": [
                "prompt articulability", "reconstruction", "isomorphism", "codability",
                "whole-construct verifiability", "tacitness",
            ],
            "upstream_frame_caveat": (
                "This audit never loads an outcome or reference, but the upstream active "
                "corpus builder restricted its sampling frame to PRs with known accepted/"
                "rejected status."
            ),
            "developmental_caveat": (
                "The machine-readable contract was frozen before this family execution but "
                "during instrument development, so this is exploratory/developmental rather "
                "than a confirmatory preregistration."
            ),
            "secondary_scope": (
                "The all-16 summary is secondary coverage. The primary macro uses only the "
                "10 unique programs supporting the corrected 18-mapping family."
            ),
            "promotion_gate_defined": False,
        },
    }


def check(path: Path = DEFAULT_OUT) -> dict:
    verify_frozen_inputs()
    result = _load_json(path)
    if result.get("schema") != SCHEMA or result.get("status") != (
        "developmental_exploratory_family_audit_complete"
    ):
        raise FamilyAuditError("checked artifact has an unexpected schema/status")
    if result.get("analysis_contract", {}).get("sha256") != sha256(CONTRACT):
        raise FamilyAuditError("checked artifact is not bound to the frozen contract")
    if result.get("P0_exact_frozen_replay", {}).get("total_mismatches") != 0:
        raise FamilyAuditError("checked artifact did not exactly replay P0")
    if result.get("population") != {
        "primary_unique_programs": 10,
        "primary_relation_mappings": 18,
        "secondary_unique_programs": 16,
        "primary_is_subset_of_secondary": True,
        "selection_loaded_outcomes_or_references": False,
    }:
        raise FamilyAuditError("checked artifact population drifted")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    build_parser.add_argument("--timeout", type=float, default=900.0)
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--artifact", type=Path, default=DEFAULT_OUT)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--source-path", required=True)
    worker_parser.add_argument("--aspect-id", required=True)
    worker_parser.add_argument("--source-sha256", required=True)
    worker_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in {None, "build"}:
        out = args.out if args.command == "build" else DEFAULT_OUT
        timeout = args.timeout if args.command == "build" else 900.0
        result = build(timeout=timeout)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    elif args.command == "check":
        check(args.artifact)
        print(f"PASS {args.artifact}")
    elif args.command == "worker":
        worker(args.source_path, args.aspect_id, args.source_sha256, args.out)


if __name__ == "__main__":
    main()
