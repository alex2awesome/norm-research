"""Compare code and prompt applicability/abstention on the same 18×125 cells."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

from methods.metric_seam.hierarchy_prompt_batch import validate_prompt_response
from methods.metric_seam.run_hierarchy_prompt_jobs import (
    iter_selected_jobs,
    preflight_jobs,
    request_sha256,
)


SCHEMA = "metric-seam.code-review-applicability-ladder.v1"
IMPLEMENTATION_CHANNEL = "implementation_disclosed"
CEILING_CHANNEL = "full_executable_contract"
EXPECTED_CELLS = 18
EXPECTED_ITEMS = 125
EXPECTED_PROMPT_ROWS = EXPECTED_CELLS * EXPECTED_ITEMS * 2


class ApplicabilityAnalysisError(ValueError):
    """Raised when the diagnostic inputs do not share the frozen support."""


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ApplicabilityAnalysisError(f"expected an object in {path}")
    return value


def _load_jobs(path: Path, channel: str) -> tuple[dict[str, dict], dict[str, dict]]:
    preflight_jobs(path, channel=channel, expected_jobs=EXPECTED_PROMPT_ROWS)
    by_request: dict[str, dict] = {}
    cells: dict[str, dict] = {}
    slots: set[tuple[str, str, int]] = set()
    for row in iter_selected_jobs(path, channel):
        metadata = row["audit_metadata"]
        pass_id = metadata.get("pass_id", metadata.get("pass_index"))
        slot = (metadata["cell_id"], metadata["item_key"], pass_id)
        if slot in slots:
            raise ApplicabilityAnalysisError(f"duplicate prompt slot: {slot}")
        slots.add(slot)
        by_request[row["request_id"]] = row
        cell = cells.setdefault(
            metadata["cell_id"],
            {
                "aspect_id": metadata["aspect_id"],
                "level": metadata["level"],
                "item_keys": set(),
            },
        )
        if cell["aspect_id"] != metadata["aspect_id"]:
            raise ApplicabilityAnalysisError("cell aspect identity changed across jobs")
        cell["item_keys"].add(metadata["item_key"])
    return by_request, cells


def summarize_status_counts(counts: Mapping[str, int], expected: int) -> dict:
    """Return explicit expected- and valid-row denominators for one prompt arm."""

    valid = int(counts.get("valid", 0))
    measurement = {
        state: int(counts.get(state, 0))
        for state in ("not_applicable", "applicable_abstain", "scored")
    }
    return {
        "expected_rows": expected,
        "valid_rows": valid,
        "contract_error_rows": int(counts.get("contract_error", 0)),
        "transport_error_rows": int(counts.get("transport_error", 0)),
        "measurement_status_counts": measurement,
        "rates_over_expected_rows": {
            state: value / expected for state, value in measurement.items()
        },
        "rates_over_valid_rows": {
            state: (value / valid if valid else None)
            for state, value in measurement.items()
        },
        "unscored_rate_over_expected_rows": (
            measurement["not_applicable"] + measurement["applicable_abstain"]
        )
        / expected,
        "unscored_rate_over_valid_rows": (
            (
                measurement["not_applicable"]
                + measurement["applicable_abstain"]
            )
            / valid
            if valid
            else None
        ),
    }


def _load_prompt_arm(
    responses_path: Path,
    jobs: Mapping[str, dict],
) -> tuple[dict, dict[str, dict], dict[tuple[str, str], dict[int, str]]]:
    global_counts: Counter[str] = Counter()
    per_cell: dict[str, Counter[str]] = defaultdict(Counter)
    item_pass_states: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
    observed: set[str] = set()
    with responses_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ApplicabilityAnalysisError(
                    f"invalid response JSON at {responses_path}:{line_number}"
                ) from exc
            request_id = response.get("request_id") if isinstance(response, dict) else None
            if not isinstance(request_id, str) or request_id not in jobs:
                raise ApplicabilityAnalysisError(
                    f"unexpected response request_id at line {line_number}"
                )
            if request_id in observed:
                raise ApplicabilityAnalysisError(f"duplicate response: {request_id}")
            observed.add(request_id)
            job = jobs[request_id]
            if response.get("request_sha256") != request_sha256(job["request"]):
                raise ApplicabilityAnalysisError(f"request hash mismatch: {request_id}")
            metadata = job["audit_metadata"]
            cell_id = metadata["cell_id"]
            pass_id = metadata.get("pass_id", metadata.get("pass_index"))
            status = response.get("status")
            global_counts[str(status)] += 1
            per_cell[cell_id][str(status)] += 1
            if status == "valid":
                parsed = validate_prompt_response(response.get("parsed_response"))
                state = parsed["measurement_status"]
                global_counts[state] += 1
                per_cell[cell_id][state] += 1
                item_pass_states[(cell_id, metadata["item_key"])][pass_id] = state
    missing = set(jobs) - observed
    if missing:
        raise ApplicabilityAnalysisError(
            f"response artifact is incomplete: {len(missing)} requests missing"
        )
    aggregate = summarize_status_counts(global_counts, len(jobs))
    cells = {
        cell_id: summarize_status_counts(counts, EXPECTED_ITEMS * 2)
        for cell_id, counts in per_cell.items()
    }
    return aggregate, cells, item_pass_states


def _prompt_pass_consensus(
    item_pass_states: Mapping[tuple[str, str], Mapping[int, str]],
) -> dict:
    counts: Counter[str] = Counter()
    for states in item_pass_states.values():
        if set(states) != {1, 2}:
            counts["missing_or_invalid_pass"] += 1
            continue
        pair = (states[1], states[2])
        if pair[0] == pair[1]:
            counts[f"both_{pair[0]}"] += 1
        else:
            counts["discordant_valid_states"] += 1
    expected = EXPECTED_CELLS * EXPECTED_ITEMS
    return {
        "expected_cell_items": expected,
        "counts": dict(sorted(counts.items())),
        "rates": {key: value / expected for key, value in sorted(counts.items())},
    }


def _code_rows_by_cell(
    execution: Mapping[str, object],
    cells: Mapping[str, dict],
) -> tuple[dict, dict[str, dict]]:
    programs = execution.get("programs")
    if not isinstance(programs, list):
        raise ApplicabilityAnalysisError("code execution has no programs")
    programs_by_aspect = {program["aspect_id"]: program for program in programs}
    total: Counter[str] = Counter()
    per_cell = {}
    for cell_id, metadata in cells.items():
        program = programs_by_aspect.get(metadata["aspect_id"])
        if not isinstance(program, Mapping):
            raise ApplicabilityAnalysisError(f"missing code program for {cell_id}")
        rows = program.get("rows")
        if not isinstance(rows, list) or len(rows) != EXPECTED_ITEMS:
            raise ApplicabilityAnalysisError(f"invalid code rows for {cell_id}")
        row_items = {row["item_key"] for row in rows}
        if row_items != metadata["item_keys"]:
            raise ApplicabilityAnalysisError(f"code/prompt item mismatch for {cell_id}")
        counts = Counter(row["status"] for row in rows)
        if set(counts) - {"scored", "abstained", "not_applicable"}:
            raise ApplicabilityAnalysisError(f"unknown code status for {cell_id}")
        total.update(counts)
        per_cell[cell_id] = {
            "aspect_id": metadata["aspect_id"],
            "level": metadata["level"],
            "expected_items": EXPECTED_ITEMS,
            "status_counts": dict(sorted(counts.items())),
            "rates": {
                state: counts[state] / EXPECTED_ITEMS
                for state in ("not_applicable", "abstained", "scored")
            },
            "unscored_rate": (
                counts["not_applicable"] + counts["abstained"]
            )
            / EXPECTED_ITEMS,
        }
    expected = EXPECTED_CELLS * EXPECTED_ITEMS
    return {
        "expected_mapping_items": expected,
        "status_counts": dict(sorted(total.items())),
        "rates": {
            state: total[state] / expected
            for state in ("not_applicable", "abstained", "scored")
        },
        "unscored_rate": (
            total["not_applicable"] + total["abstained"]
        )
        / expected,
    }, per_cell


def build_comparison(code: Mapping, implementation: Mapping, ceiling: Mapping) -> dict:
    code_not_applicable = code["rates"]["not_applicable"]
    code_unscored = code["unscored_rate"]
    implementation_not_applicable = implementation["rates_over_valid_rows"][
        "not_applicable"
    ]
    ceiling_not_applicable = ceiling["rates_over_valid_rows"]["not_applicable"]
    implementation_unscored = implementation["unscored_rate_over_valid_rows"]
    ceiling_unscored = ceiling["unscored_rate_over_valid_rows"]
    return {
        "primary_denominators": (
            "code rates use 18 mapping-weighted cells × 125 items; prompt rates use "
            "strict-valid pass-level responses, with expected-row rates also reported"
        ),
        "not_applicable_rate": {
            "code": code_not_applicable,
            "implementation_disclosed": implementation_not_applicable,
            "full_executable_contract": ceiling_not_applicable,
        },
        "total_unscored_rate": {
            "code": code_unscored,
            "implementation_disclosed": implementation_unscored,
            "full_executable_contract": ceiling_unscored,
        },
        "differences_from_code": {
            "implementation_not_applicable_minus_code": (
                implementation_not_applicable - code_not_applicable
            ),
            "ceiling_not_applicable_minus_code": (
                ceiling_not_applicable - code_not_applicable
            ),
            "implementation_unscored_minus_code": (
                implementation_unscored - code_unscored
            ),
            "ceiling_unscored_minus_code": ceiling_unscored - code_unscored,
        },
        "absolute_not_applicable_distance_from_code": {
            "implementation_disclosed": abs(
                implementation_not_applicable - code_not_applicable
            ),
            "full_executable_contract": abs(
                ceiling_not_applicable - code_not_applicable
            ),
        },
        "ceiling_is_closer_to_code_not_applicability_than_implementation": (
            abs(ceiling_not_applicable - code_not_applicable)
            < abs(implementation_not_applicable - code_not_applicable)
        ),
        "interpretation_limit": (
            "No numerical threshold for 'approximately code' was preregistered. "
            "Distances and per-cell rates are reported; the substantive reading must "
            "not be created after seeing them."
        ),
    }


def analyze(
    *,
    implementation_jobs_path: Path,
    implementation_responses_path: Path,
    ceiling_jobs_path: Path,
    ceiling_responses_path: Path,
    code_execution_path: Path,
) -> dict:
    implementation_jobs, implementation_cells = _load_jobs(
        implementation_jobs_path, IMPLEMENTATION_CHANNEL
    )
    ceiling_jobs, ceiling_cells = _load_jobs(ceiling_jobs_path, CEILING_CHANNEL)
    if set(implementation_cells) != set(ceiling_cells):
        raise ApplicabilityAnalysisError("implementation and ceiling cells differ")
    for cell_id in implementation_cells:
        left = implementation_cells[cell_id]
        right = ceiling_cells[cell_id]
        if (
            left["aspect_id"] != right["aspect_id"]
            or left["level"] != right["level"]
            or left["item_keys"] != right["item_keys"]
        ):
            raise ApplicabilityAnalysisError(f"arm support mismatch for {cell_id}")

    implementation, implementation_per_cell, implementation_pairs = _load_prompt_arm(
        implementation_responses_path, implementation_jobs
    )
    ceiling, ceiling_per_cell, ceiling_pairs = _load_prompt_arm(
        ceiling_responses_path, ceiling_jobs
    )
    code, code_per_cell = _code_rows_by_cell(
        _load_json(code_execution_path), ceiling_cells
    )
    per_mapping = []
    for cell_id in sorted(ceiling_cells):
        per_mapping.append(
            {
                "cell_id": cell_id,
                "aspect_id": ceiling_cells[cell_id]["aspect_id"],
                "level": ceiling_cells[cell_id]["level"],
                "code": code_per_cell[cell_id],
                "implementation_disclosed": implementation_per_cell[cell_id],
                "full_executable_contract": ceiling_per_cell[cell_id],
            }
        )
    return {
        "schema": SCHEMA,
        "scope": {
            "relation_mappings": EXPECTED_CELLS,
            "items_per_mapping": EXPECTED_ITEMS,
            "prompt_passes": 2,
            "implementation_channel": IMPLEMENTATION_CHANNEL,
            "ceiling_channel": CEILING_CHANNEL,
        },
        "code": code,
        "implementation_disclosed": implementation,
        "full_executable_contract": ceiling,
        "implementation_pass_consensus": _prompt_pass_consensus(implementation_pairs),
        "ceiling_pass_consensus": _prompt_pass_consensus(ceiling_pairs),
        "comparison": build_comparison(code, implementation, ceiling),
        "per_mapping": per_mapping,
        "claim_limits": [
            "The code has distinct not_applicable and abstained states; neither is silently collapsed in this diagnostic.",
            "Prompt-arm primary rates condition on strict-valid responses, and expected-row rates are reported separately.",
            "Cells sharing a program remain separate relation mappings, matching the reconstruction estimand.",
            "This diagnostic localizes support loss; it does not establish specificity, tacitness, or external correctness.",
        ],
    }


def _pct(value: float | None) -> str:
    return "undefined" if value is None else f"{100.0 * value:.1f}%"


def render_report(readout: Mapping[str, object]) -> str:
    comparison = readout["comparison"]
    code = readout["code"]
    implementation = readout["implementation_disclosed"]
    ceiling = readout["full_executable_contract"]
    not_applicable = comparison["not_applicable_rate"]
    unscored = comparison["total_unscored_rate"]
    lines = [
        (
            f"**Not-applicable rates: code {_pct(not_applicable['code'])}; "
            f"full-contract ceiling {_pct(not_applicable['full_executable_contract'])}; "
            f"implementation summary {_pct(not_applicable['implementation_disclosed'])}.**"
        ),
        "",
        "# Code-review applicability and abstention ladder",
        "",
        (
            f"Total unscored rates (keeping code abstention distinct in the artifact): "
            f"code {_pct(unscored['code'])}; full-contract ceiling "
            f"{_pct(unscored['full_executable_contract'])}; implementation summary "
            f"{_pct(unscored['implementation_disclosed'])}."
        ),
        "",
        (
            "Code status breakdown: "
            f"{_pct(code['rates']['scored'])} scored, "
            f"{_pct(code['rates']['not_applicable'])} not_applicable, and "
            f"{_pct(code['rates']['abstained'])} applicable-but-abstained. "
            "The full-contract prompt used no applicable_abstain response, so its "
            "not_applicable rate must not be treated as an exact state match to code."
        ),
        "",
        (
            "The previously cited 84.1% implementation-summary abstention figure is "
            "not reproduced by the frozen primary denominators. Exact rates are "
            f"{_pct(implementation['rates_over_valid_rows']['not_applicable'])} "
            "not_applicable over strict-valid rows and "
            f"{_pct(implementation['rates_over_expected_rows']['not_applicable'])} "
            "over expected rows; total unscored rates are "
            f"{_pct(implementation['unscored_rate_over_valid_rows'])} and "
            f"{_pct(implementation['unscored_rate_over_expected_rows'])}, respectively. "
            "The corresponding full-contract total-unscored rates are "
            f"{_pct(ceiling['unscored_rate_over_valid_rows'])} and "
            f"{_pct(ceiling['unscored_rate_over_expected_rows'])}."
        ),
        "",
        (
            "The full-contract arm is closer to code not-applicability than the "
            "implementation summary: "
            f"**{comparison['ceiling_is_closer_to_code_not_applicability_than_implementation']}**. "
            "This is descriptive because no numerical equivalence threshold was frozen."
        ),
        "",
        "## Claim limits",
        "",
    ]
    lines.extend(f"- {limit}" for limit in readout["claim_limits"])
    lines.append("")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation-jobs", required=True, type=Path)
    parser.add_argument("--implementation-responses", required=True, type=Path)
    parser.add_argument("--ceiling-jobs", required=True, type=Path)
    parser.add_argument("--ceiling-responses", required=True, type=Path)
    parser.add_argument("--code-execution", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    readout = analyze(
        implementation_jobs_path=args.implementation_jobs,
        implementation_responses_path=args.implementation_responses,
        ceiling_jobs_path=args.ceiling_jobs,
        ceiling_responses_path=args.ceiling_responses,
        code_execution_path=args.code_execution,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(readout, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path = args.output.with_suffix(".md")
    report_path.write_text(render_report(readout), encoding="utf-8")
    print(json.dumps(readout["comparison"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
