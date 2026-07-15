#!/usr/bin/env python3
"""One-call semantic alignment for a completed three-fleet decomposition.

The Sonnet prompt contains only the metric text and blinded relation units.  It
never contains corpus items, programs, executions, labels, or outcomes.  Each
semantic cluster may contain at most one unit from each opaque source set;
units without a construct-equivalent counterpart remain unmatched.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Literal, Mapping, Sequence

from .decomposition_stability import (
    DecompositionSchemaError,
    FleetDecomposition,
    SCHEMA as DECOMPOSITION_SCHEMA,
)


REQUEST_SCHEMA = "metric-seam.three-fleet-semantic-alignment-request.v1"
REPORT_SCHEMA = "metric-seam.three-fleet-semantic-alignment-report.v1"
PARSER_VERSION = "metric-seam.three-fleet-semantic-alignment-parser.v1"

_FENCE = re.compile(
    r"\A\s*```(?:json)?\s*\n?(.*?)\n?```\s*\Z", re.DOTALL | re.IGNORECASE
)
_OPAQUE_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")

SYSTEM_PROMPT = """You are a semantic alignment adjudicator.
You receive one metric and three opaque source sets of independently proposed executable relation units. Cluster units only when they express the same construct-level relation, not merely because they share an operation class, witness type, topic, or vocabulary. A cluster must contain two or three unit IDs and at most one unit from each source set. Leave any unit unmatched when construct equivalence is not supported.
Return exactly one JSON object with exactly these keys:
{"clusters": [{"cluster_id": string, "unit_ids": [string]}], "unmatched_unit_ids": [string]}
Every supplied unit ID must occur exactly once, either in one cluster or in unmatched_unit_ids. Do not emit confidence, scores, rationales, Markdown, or additional keys."""


class AlignmentError(ValueError):
    """Raised when the closed-world input or alignment response is invalid."""


@dataclass(frozen=True)
class ParsedAlignment:
    clusters: tuple[dict[str, object], ...]
    unmatched_unit_ids: tuple[str, ...]
    parser_version: str
    parse_mode: Literal["strict_json", "fence_unwrapped"]


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AlignmentError("value is not finite JSON") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _hash_rank(seed: str, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode("utf-8")).hexdigest()


def _exact_keys(value: Mapping[str, object], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise AlignmentError(
            f"{path}: key mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _nonempty_text(value: object, path: str, *, maximum: int = 20000) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or any(ord(char) < 32 and char not in "\n\t" for char in value)
    ):
        raise AlignmentError(f"{path}: expected bounded nonempty text")
    return value


def _load_closed_submission(
    value: object,
) -> tuple[dict[str, str], tuple[FleetDecomposition, ...]]:
    if not isinstance(value, dict):
        raise AlignmentError("submission must be an object")
    _exact_keys(value, {"schema", "metric", "fleets"}, "submission")
    if value["schema"] != DECOMPOSITION_SCHEMA:
        raise AlignmentError("submission schema is not a decomposition submission")
    metric = value["metric"]
    if not isinstance(metric, dict):
        raise AlignmentError("submission.metric must be an object")
    _exact_keys(metric, {"name", "text"}, "submission.metric")
    metric_value = {
        "name": _nonempty_text(metric["name"], "submission.metric.name", maximum=1000),
        "text": _nonempty_text(metric["text"], "submission.metric.text"),
    }
    fleets_raw = value["fleets"]
    if not isinstance(fleets_raw, list) or len(fleets_raw) != 3:
        raise AlignmentError("submission must contain exactly three fleets")
    try:
        fleets = tuple(
            FleetDecomposition.from_value(row, f"submission.fleets[{index}]")
            for index, row in enumerate(fleets_raw)
        )
    except DecompositionSchemaError as exc:
        raise AlignmentError("submission contains an invalid fleet") from exc
    if len({fleet.fleet_id for fleet in fleets}) != 3:
        raise AlignmentError("fleet IDs must be unique")
    return metric_value, fleets


def compile_alignment_request(
    submission: object,
    *,
    randomization_seed: str,
    model: str = "sonnet",
) -> dict[str, object]:
    """Compile the single blinded alignment call licensed for one metric."""

    metric, fleets = _load_closed_submission(submission)
    seed = _nonempty_text(randomization_seed, "randomization_seed", maximum=1000)
    model = _nonempty_text(model, "model", maximum=256)
    if "sonnet" not in model.casefold():
        raise AlignmentError("semantic alignment requires a Sonnet model")

    # Opaque set labels are assigned by a seeded permutation.  The public
    # prompt carries set membership only to enforce one-to-one matching; the
    # private map retains the original fleet identity for analysis.
    fleet_order = sorted(
        fleets,
        key=lambda fleet: (
            _hash_rank(seed, "fleet", fleet.fleet_id),
            fleet.fleet_id,
        ),
    )
    source_set_for_fleet = {
        fleet.fleet_id: f"source_{index + 1}" for index, fleet in enumerate(fleet_order)
    }

    public_units: list[dict[str, str]] = []
    private_units: list[dict[str, object]] = []
    used_ids: set[str] = set()
    for fleet in fleets:
        for index, relation in enumerate(fleet.relations):
            nonce = 0
            while True:
                opaque = "u_" + _hash_rank(
                    seed,
                    "unit",
                    f"{fleet.fleet_id}\0{index}\0{nonce}\0{_canonical_json(relation.to_value())}",
                )[:16]
                if opaque not in used_ids:
                    break
                nonce += 1
            used_ids.add(opaque)
            source_set_id = source_set_for_fleet[fleet.fleet_id]
            public_units.append(
                {
                    "unit_id": opaque,
                    "source_set_id": source_set_id,
                    **relation.to_value(),
                }
            )
            private_units.append(
                {
                    "unit_id": opaque,
                    "source_set_id": source_set_id,
                    "fleet_id": fleet.fleet_id,
                    "relation": relation.to_value(),
                }
            )

    public_units.sort(
        key=lambda row: (
            _hash_rank(seed, "public_order", row["unit_id"]),
            row["unit_id"],
        )
    )
    private_units.sort(key=lambda row: str(row["unit_id"]))
    prompt_input = {"metric": metric, "relation_units": public_units}
    user_prompt = "ALIGNMENT_INPUT_JSON:\n" + _canonical_json(prompt_input)
    identity: dict[str, object] = {
        "schema": REQUEST_SCHEMA,
        "status": "compiled_for_exactly_one_alignment_call",
        "model": model,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "metric": metric,
        "public_relation_units": public_units,
        "private_unit_map": private_units,
        "randomization_seed_sha256": _sha256(seed),
        "submission_sha256": _sha256(submission),
        "response_contract": {
            "parser_version": PARSER_VERSION,
            "top_level_keys": ["clusters", "unmatched_unit_ids"],
            "cluster_size": [2, 3],
            "maximum_per_source_set_per_cluster": 1,
            "all_units_partitioned_exactly_once": True,
            "floats_allowed": False,
        },
    }
    return {**identity, "request_sha256": _sha256(identity)}


def _validated_request(request: object) -> dict[str, object]:
    if not isinstance(request, dict):
        raise AlignmentError("compiled request must be an object")
    expected = {
        "schema",
        "status",
        "model",
        "system_prompt",
        "user_prompt",
        "metric",
        "public_relation_units",
        "private_unit_map",
        "randomization_seed_sha256",
        "submission_sha256",
        "response_contract",
        "request_sha256",
    }
    _exact_keys(request, expected, "request")
    if request["schema"] != REQUEST_SCHEMA:
        raise AlignmentError("unsupported request schema")
    if request["status"] != "compiled_for_exactly_one_alignment_call":
        raise AlignmentError("request is not call-ready")
    if request["system_prompt"] != SYSTEM_PROMPT:
        raise AlignmentError("system prompt changed after compilation")
    model = request["model"]
    if not isinstance(model, str) or "sonnet" not in model.casefold():
        raise AlignmentError("request is not pinned to Sonnet")
    identity = {key: request[key] for key in expected - {"request_sha256"}}
    if request["request_sha256"] != _sha256(identity):
        raise AlignmentError("request digest mismatch")
    public = request["public_relation_units"]
    private = request["private_unit_map"]
    if not isinstance(public, list) or not isinstance(private, list):
        raise AlignmentError("request relation maps must be arrays")
    public_ids = {row.get("unit_id") for row in public if isinstance(row, dict)}
    private_ids = {row.get("unit_id") for row in private if isinstance(row, dict)}
    if len(public_ids) != len(public) or public_ids != private_ids:
        raise AlignmentError("public and private unit maps do not align")
    return request


def parse_alignment_response(
    raw: str, *, request: Mapping[str, object]
) -> ParsedAlignment:
    """Parse the strict partition and enforce fleetwise one-to-one clusters."""

    validated_request = _validated_request(dict(request))
    if not isinstance(raw, str) or not raw.strip():
        raise AlignmentError("empty alignment response")
    mode: Literal["strict_json", "fence_unwrapped"] = "strict_json"
    candidate = raw
    fenced = _FENCE.match(raw)
    if fenced:
        mode = "fence_unwrapped"
        candidate = fenced.group(1)

    def reject_float(_: str) -> None:
        raise AlignmentError("floating-point values are forbidden")

    try:
        value = json.loads(
            candidate,
            parse_float=reject_float,
            parse_constant=reject_float,
        )
    except AlignmentError:
        raise
    except json.JSONDecodeError as exc:
        raise AlignmentError("alignment response is not valid JSON") from exc
    if not isinstance(value, dict):
        raise AlignmentError("alignment response must be an object")
    _exact_keys(value, {"clusters", "unmatched_unit_ids"}, "response")
    clusters_raw = value["clusters"]
    unmatched_raw = value["unmatched_unit_ids"]
    if not isinstance(clusters_raw, list) or not isinstance(unmatched_raw, list):
        raise AlignmentError("clusters and unmatched_unit_ids must be arrays")

    private_rows = validated_request["private_unit_map"]
    assert isinstance(private_rows, list)
    private_by_id = {
        str(row["unit_id"]): row for row in private_rows if isinstance(row, dict)
    }
    known_ids = set(private_by_id)
    seen_ids: set[str] = set()
    seen_clusters: set[str] = set()
    clusters: list[dict[str, object]] = []
    unit_cluster: dict[str, str] = {}
    for index, cluster in enumerate(clusters_raw):
        if not isinstance(cluster, dict):
            raise AlignmentError(f"response.clusters[{index}] must be an object")
        _exact_keys(cluster, {"cluster_id", "unit_ids"}, f"response.clusters[{index}]")
        cluster_id = cluster["cluster_id"]
        unit_ids = cluster["unit_ids"]
        if not isinstance(cluster_id, str) or not _OPAQUE_ID.fullmatch(cluster_id):
            raise AlignmentError(f"response.clusters[{index}].cluster_id is invalid")
        if cluster_id in seen_clusters:
            raise AlignmentError("cluster IDs must be unique")
        seen_clusters.add(cluster_id)
        if not isinstance(unit_ids, list) or not 2 <= len(unit_ids) <= 3:
            raise AlignmentError("each semantic cluster must contain two or three units")
        if not all(isinstance(unit_id, str) and unit_id in known_ids for unit_id in unit_ids):
            raise AlignmentError("semantic cluster contains an unknown unit")
        if len(unit_ids) != len(set(unit_ids)) or seen_ids & set(unit_ids):
            raise AlignmentError("each unit may occur only once")
        source_sets = [private_by_id[unit_id]["source_set_id"] for unit_id in unit_ids]
        if len(source_sets) != len(set(source_sets)):
            raise AlignmentError("a cluster may contain at most one unit per fleet")
        seen_ids.update(unit_ids)
        for unit_id in unit_ids:
            unit_cluster[unit_id] = cluster_id
        clusters.append({"cluster_id": cluster_id, "unit_ids": list(unit_ids)})

    if not all(isinstance(unit_id, str) and unit_id in known_ids for unit_id in unmatched_raw):
        raise AlignmentError("unmatched_unit_ids contains an unknown unit")
    unmatched = tuple(unmatched_raw)
    if len(unmatched) != len(set(unmatched)) or seen_ids & set(unmatched):
        raise AlignmentError("each unit may occur only once")
    seen_ids.update(unmatched)
    if seen_ids != known_ids:
        raise AlignmentError("clusters and unmatched units must partition every supplied unit")

    # Exact normalized relation identity is the declared mechanical lower
    # bound.  The semantic adjudicator may add alignments but may not erase an
    # exact pair that was already identified without semantic judgment.
    for left_index, left in enumerate(private_rows):
        assert isinstance(left, dict)
        for right in private_rows[left_index + 1 :]:
            assert isinstance(right, dict)
            if (
                left["fleet_id"] != right["fleet_id"]
                and left["relation"] == right["relation"]
                and (
                    unit_cluster.get(str(left["unit_id"])) is None
                    or unit_cluster.get(str(left["unit_id"]))
                    != unit_cluster.get(str(right["unit_id"]))
                )
            ):
                raise AlignmentError(
                    "semantic response omitted a lexical-exact lower-bound match"
                )

    return ParsedAlignment(
        tuple(clusters), unmatched, PARSER_VERSION, mode
    )


def _ratio(numerator: int, denominator: int) -> dict[str, object] | None:
    if denominator == 0:
        return None
    return {
        "numerator": numerator,
        "denominator": denominator,
        "decimal": round(numerator / denominator, 6),
    }


def build_alignment_report(
    *, request: Mapping[str, object], raw_response: str
) -> dict[str, object]:
    """Build pairwise semantic and lexical-lower-bound overlap statistics."""

    parsed = parse_alignment_response(raw_response, request=request)
    validated_request = _validated_request(dict(request))
    private_rows = validated_request["private_unit_map"]
    assert isinstance(private_rows, list)
    private_by_id = {
        str(row["unit_id"]): row for row in private_rows if isinstance(row, dict)
    }
    cluster_for: dict[str, str] = {}
    for cluster in parsed.clusters:
        for unit_id in cluster["unit_ids"]:  # type: ignore[union-attr]
            cluster_for[str(unit_id)] = str(cluster["cluster_id"])

    fleet_ids = sorted({str(row["fleet_id"]) for row in private_rows if isinstance(row, dict)})
    units_by_fleet = {
        fleet_id: [
            row for row in private_rows
            if isinstance(row, dict) and row["fleet_id"] == fleet_id
        ]
        for fleet_id in fleet_ids
    }
    pairwise: list[dict[str, object]] = []
    for left_index, fleet_a in enumerate(fleet_ids):
        for fleet_b in fleet_ids[left_index + 1 :]:
            units_a = units_by_fleet[fleet_a]
            units_b = units_by_fleet[fleet_b]
            semantic_pairs = [
                (str(left["unit_id"]), str(right["unit_id"]))
                for left in units_a
                for right in units_b
                if cluster_for.get(str(left["unit_id"])) is not None
                and cluster_for.get(str(left["unit_id"]))
                == cluster_for.get(str(right["unit_id"]))
            ]
            lexical_pairs = [
                (str(left["unit_id"]), str(right["unit_id"]))
                for left in units_a
                for right in units_b
                if left["relation"] == right["relation"]
            ]
            semantic_n = len(semantic_pairs)
            lexical_n = len(lexical_pairs)
            n_a = len(units_a)
            n_b = len(units_b)
            pairwise.append(
                {
                    "fleet_a": fleet_a,
                    "fleet_b": fleet_b,
                    "n_a": n_a,
                    "n_b": n_b,
                    "semantic_match_count": semantic_n,
                    "semantic_precision_a_to_b": _ratio(semantic_n, n_a),
                    "semantic_recall_a_to_b": _ratio(semantic_n, n_b),
                    "semantic_jaccard": _ratio(semantic_n, n_a + n_b - semantic_n),
                    "lexical_exact_lower_bound": {
                        "match_count": lexical_n,
                        "precision_a_to_b": _ratio(lexical_n, n_a),
                        "recall_a_to_b": _ratio(lexical_n, n_b),
                        "jaccard": _ratio(lexical_n, n_a + n_b - lexical_n),
                    },
                    "semantic_gain_over_lexical_exact": semantic_n - lexical_n,
                    "semantic_matched_unit_pairs": [list(pair) for pair in semantic_pairs],
                    "lexical_exact_unit_pairs": [list(pair) for pair in lexical_pairs],
                }
            )

    def disclosed_unit(unit_id: str) -> dict[str, object]:
        row = private_by_id[unit_id]
        return {
            "unit_id": unit_id,
            "fleet_id": row["fleet_id"],
            **row["relation"],  # type: ignore[arg-type]
        }

    return {
        "schema": REPORT_SCHEMA,
        "status": "one_call_semantic_alignment_complete",
        "input_scope": "metric_text_and_three_fleet_relation_descriptions_only",
        "metric": validated_request["metric"],
        "request_sha256": validated_request["request_sha256"],
        "response_sha256": hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
        "model": validated_request["model"],
        "alignment_call_count": 1,
        "parser": {
            "version": parsed.parser_version,
            "mode": parsed.parse_mode,
        },
        "semantic_clusters": [
            {
                "cluster_id": cluster["cluster_id"],
                "units": [
                    disclosed_unit(str(unit_id))
                    for unit_id in cluster["unit_ids"]  # type: ignore[union-attr]
                ],
            }
            for cluster in parsed.clusters
        ],
        "unmatched_units": [disclosed_unit(unit_id) for unit_id in parsed.unmatched_unit_ids],
        "pairwise": pairwise,
        "lexical_exact_retained_as_lower_bound": True,
        "scope_guards": {
            "corpus_accessed": False,
            "programs_accessed": False,
            "program_outputs_accessed": False,
            "scores_labels_or_outcomes_accessed": False,
        },
    }


def run_alignment_call(
    request: Mapping[str, object],
    *,
    executable: str = "claude",
    timeout_seconds: float = 300.0,
) -> dict[str, object]:
    """Execute exactly one fresh local Sonnet call and return its report."""

    validated = _validated_request(dict(request))
    argv = [
        executable,
        "--model",
        str(validated["model"]),
        "--output-format",
        "text",
        "--system-prompt",
        SYSTEM_PROMPT,
        "-p",
        str(validated["user_prompt"]),
    ]
    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise AlignmentError(
            f"local Sonnet alignment call failed with exit code {completed.returncode}"
        )
    return build_alignment_report(request=validated, raw_response=completed.stdout)


def _write_json(path: Path | None, value: object) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(rendered, end="")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile", help="compile one blinded request")
    compile_parser.add_argument("submission", type=Path)
    compile_parser.add_argument("--seed", required=True)
    compile_parser.add_argument("--model", default="sonnet")
    compile_parser.add_argument("--output", type=Path)

    score_parser = subparsers.add_parser("score", help="score one retained response")
    score_parser.add_argument("request", type=Path)
    score_parser.add_argument("response", type=Path)
    score_parser.add_argument("--output", type=Path)

    run_parser = subparsers.add_parser("run", help="make exactly one local Sonnet call")
    run_parser.add_argument("request", type=Path)
    run_parser.add_argument("--executable", default="claude")
    run_parser.add_argument("--timeout-seconds", type=float, default=300.0)
    run_parser.add_argument("--output", type=Path)

    args = parser.parse_args(argv)
    if args.command == "compile":
        submission = json.loads(args.submission.read_text(encoding="utf-8"))
        result = compile_alignment_request(
            submission, randomization_seed=args.seed, model=args.model
        )
    elif args.command == "score":
        request = json.loads(args.request.read_text(encoding="utf-8"))
        raw_response = args.response.read_text(encoding="utf-8")
        result = build_alignment_report(request=request, raw_response=raw_response)
    else:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        result = run_alignment_call(
            request,
            executable=args.executable,
            timeout_seconds=args.timeout_seconds,
        )
    _write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
