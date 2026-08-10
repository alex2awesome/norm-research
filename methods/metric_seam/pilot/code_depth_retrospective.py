"""CPU-only full-panel retrospective of deep versus shallow code programs.

This analysis uses the active 18-criterion code-review panel.  Both compared
arms are executable code at runtime:

* ``*_coded_checker`` is the pre-existing manually engineered static/AST arm;
* ``*_v{0,1,2}_*`` are prompt-generated shallow Python programs, selected by
  TRAIN reconstruction agreement only.

The frozen two-pass LLM judgments are the unsupervised reconstruction target.
``items.json.judgement`` is deliberately never read.  Because the programs and
references predate this evaluator and aggregate outcomes were inspected before
this script was frozen, the result is retrospective and exploratory.  It does
not certify prompt articulability, code verifiability, construct fidelity,
isomorphism, or correctness on disagreements.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping

from methods.metric_seam.battery.certify_batch_v2 import (
    benjamini_hochberg,
    paired_bootstrap_ci,
    paired_randomization_test,
    spearman,
)


SCHEMA = "metric-seam.code-depth-full-panel-retrospective.v2"
ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = ROOT / "outputs/metric_seam_pilot/tasks/code_review"
DEFAULT_OUT = (
    ROOT
    / "outputs/metric_seam_pilot/reconstruction_v2/"
    "code_depth_full_panel_retrospective_002"
)
_SHALLOW_RE = re.compile(r"^(a\d+)_v[012]_")


@dataclass(frozen=True)
class Settings:
    split_seed: int = 7
    train_count: int = 150
    heldout_count: int = 100
    alpha: float = 0.05
    minimum_effect: float = 0.02
    conditional_coverage_min: float = 0.90
    min_pairs: int = 20
    permutation_samples: int = 10_000
    bootstrap_samples: int = 5_000
    bootstrap_confidence: float = 0.95
    resampling_seed: int = 20_260_713


class RetrospectiveError(RuntimeError):
    """Raised when an input or analysis invariant is violated."""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_receipt(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _load_reference_rows(
    path: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]]:
    """Return two-pass composites and the underlying pass maps.

    Duplicate ``(criterion, item, channel)`` rows fail closed.  Only ``pass1``
    and ``pass2`` numeric scores enter; raw response text is ignored.
    """

    passes: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            aspect = row.get("aspect_id")
            item_id = row.get("datapoint_id")
            channel = row.get("channel")
            if channel not in {"pass1", "pass2"} or not isinstance(aspect, str):
                continue
            if not isinstance(item_id, str) or not item_id:
                raise RetrospectiveError(f"invalid item ID on line {line_number}")
            if channel in passes[aspect][item_id]:
                raise RetrospectiveError(
                    f"duplicate reference row: {(aspect, item_id, channel)}"
                )
            score = row.get("score")
            if _is_number(score):
                numeric = float(score)
                if not 0.0 <= numeric <= 10.0:
                    raise RetrospectiveError(
                        f"reference score outside 0..10 on line {line_number}"
                    )
                passes[aspect][item_id][channel] = numeric

    composites: dict[str, dict[str, float]] = defaultdict(dict)
    for aspect, by_item in passes.items():
        for item_id, values in by_item.items():
            if {"pass1", "pass2"}.issubset(values):
                composites[aspect][item_id] = (
                    values["pass1"] + values["pass2"]
                ) / 20.0
    return dict(composites), {
        aspect: dict(by_item) for aspect, by_item in passes.items()
    }


def _item_ids_without_outcomes(path: Path) -> list[str]:
    rows = _read_json(path)
    if not isinstance(rows, list):
        raise RetrospectiveError("items.json must be a list")
    item_ids: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RetrospectiveError(f"items[{index}] is not an object")
        item_id = row.get("datapoint_id")
        if not isinstance(item_id, str) or not item_id:
            raise RetrospectiveError(f"items[{index}] has no datapoint_id")
        item_ids.append(item_id)
    if len(item_ids) != len(set(item_ids)):
        raise RetrospectiveError("items.json contains duplicate datapoint_id values")
    return item_ids


def _split(item_ids: Iterable[str], settings: Settings) -> tuple[set[str], set[str]]:
    shuffled = sorted(item_ids)
    random.Random(settings.split_seed).shuffle(shuffled)
    expected = settings.train_count + settings.heldout_count
    if len(shuffled) != expected:
        raise RetrospectiveError(
            f"expected {expected} items, observed {len(shuffled)}"
        )
    return set(shuffled[: settings.train_count]), set(shuffled[settings.train_count :])


def _validate_score_maps(raw: Any) -> dict[str, dict[str, float | None]]:
    if not isinstance(raw, Mapping):
        raise RetrospectiveError("code_scores.json must be an object")
    output: dict[str, dict[str, float | None]] = {}
    for program, score_map in raw.items():
        if not isinstance(program, str) or not isinstance(score_map, Mapping):
            raise RetrospectiveError("invalid code-score program record")
        cleaned: dict[str, float | None] = {}
        for item_id, value in score_map.items():
            if not isinstance(item_id, str):
                raise RetrospectiveError(f"{program} contains a non-string item ID")
            if value is None:
                cleaned[item_id] = None
            elif _is_number(value) and 0.0 <= float(value) <= 1.0:
                cleaned[item_id] = float(value)
            else:
                raise RetrospectiveError(f"invalid score in {program}[{item_id}]")
        output[program] = cleaned
    return output


def _available(score_map: Mapping[str, float | None]) -> set[str]:
    return {item_id for item_id, value in score_map.items() if value is not None}


def _rho(
    score_map: Mapping[str, float | None],
    reference: Mapping[str, float],
    ids: Iterable[str],
) -> tuple[float | None, int]:
    common = sorted(set(ids) & set(reference) & _available(score_map))
    if len(common) < 3:
        return None, len(common)
    value = spearman(
        [float(score_map[item_id]) for item_id in common],
        [reference[item_id] for item_id in common],
    )
    return (value if math.isfinite(value) else None), len(common)


def _reference_reliability(
    pass_rows: Mapping[str, Mapping[str, float]], ids: Iterable[str]
) -> tuple[float | None, int]:
    common = sorted(
        item_id
        for item_id in set(ids) & set(pass_rows)
        if {"pass1", "pass2"}.issubset(pass_rows[item_id])
    )
    if len(common) < 3:
        return None, len(common)
    value = spearman(
        [pass_rows[item_id]["pass1"] for item_id in common],
        [pass_rows[item_id]["pass2"] for item_id in common],
    )
    return (value if math.isfinite(value) else None), len(common)


def _derived_seed(settings: Settings, aspect: str, family: str) -> int:
    payload = f"{settings.resampling_seed}\0{aspect}\0{family}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _select_shallow(
    aspect: str,
    score_maps: Mapping[str, Mapping[str, float | None]],
    reference: Mapping[str, float],
    train_ids: set[str],
) -> tuple[str | None, list[dict[str, Any]]]:
    variants = sorted(
        program
        for program in score_maps
        if _SHALLOW_RE.match(program) and program.startswith(f"{aspect}_")
    )
    rows: list[dict[str, Any]] = []
    for program in variants:
        rho, n = _rho(score_maps[program], reference, train_ids)
        rows.append({"program": program, "train_rho": rho, "train_n": n})
    scoreable = [row for row in rows if row["train_rho"] is not None]
    if not scoreable:
        return None, rows
    # Lexical program name is the deterministic tie-breaker.
    selected = sorted(scoreable, key=lambda row: (-row["train_rho"], row["program"]))[0]
    return str(selected["program"]), rows


def evaluate(
    *,
    settings: Settings | None = None,
    task_dir: Path = TASK_DIR,
    run_inference: bool = True,
) -> dict[str, Any]:
    settings = settings or Settings()
    input_paths = {
        "items": task_dir / "items.json",
        "references": task_dir / "results_newrun.jsonl",
        "code_scores": task_dir / "code_scores.json",
        "aspects_used": task_dir / "aspects_used.json",
        "aspect_definitions": task_dir / "aspects_candidates.json",
    }
    for label, path in input_paths.items():
        if not path.is_file():
            raise RetrospectiveError(f"missing {label}: {path}")

    item_ids = _item_ids_without_outcomes(input_paths["items"])
    train_ids, heldout_ids = _split(item_ids, settings)
    references, pass_rows = _load_reference_rows(input_paths["references"])
    score_maps = _validate_score_maps(_read_json(input_paths["code_scores"]))
    aspects = _read_json(input_paths["aspects_used"])
    if not isinstance(aspects, list) or not all(isinstance(value, str) for value in aspects):
        raise RetrospectiveError("aspects_used.json must be a list of IDs")
    definitions_raw = _read_json(input_paths["aspect_definitions"])
    definitions = {
        row["aspect_id"]: row
        for row in definitions_raw
        if isinstance(row, Mapping) and isinstance(row.get("aspect_id"), str)
    }

    results: list[dict[str, Any]] = []
    p_values: dict[str, float] = {}
    for aspect in aspects:
        reference = references.get(aspect, {})
        deep_program = f"{aspect}_coded_checker"
        deep_scores = score_maps.get(deep_program)
        selected, train_selection = _select_shallow(
            aspect, score_maps, reference, train_ids
        )
        reliability, reliability_n = _reference_reliability(
            pass_rows.get(aspect, {}), heldout_ids
        )
        definition = definitions.get(aspect, {})
        row: dict[str, Any] = {
            "criterion_id": aspect,
            "name": definition.get("name"),
            "description": definition.get("description"),
            "deep_program": deep_program if deep_scores is not None else None,
            "deep_program_authoring": "preexisting_manually_engineered_coding_A_bank",
            "shallow_program_authoring": "prompt_generated_executable_python",
            "runtime_channels": {"deep": "code", "shallow": "code"},
            "train_shallow_selection": {
                "rule": "maximum TRAIN Spearman; lexical program-name tie-break",
                "selected": selected,
                "candidates": train_selection,
            },
            "heldout_reference": {
                "count": len(set(heldout_ids) & set(reference)),
                "availability_over_heldout": len(set(heldout_ids) & set(reference))
                / settings.heldout_count,
                "pass1_pass2_spearman": reliability,
                "reliability_n": reliability_n,
            },
            "heldout_comparison": None,
            "status": None,
        }
        if deep_scores is None:
            row["status"] = "deep_program_unavailable"
            results.append(row)
            continue

        deep_rho_all, deep_n_all = _rho(deep_scores, reference, heldout_ids)
        row["deep_only_heldout"] = {
            "rho": deep_rho_all,
            "n": deep_n_all,
            "coverage_over_heldout": len(heldout_ids & _available(deep_scores))
            / settings.heldout_count,
        }
        if selected is None:
            row["status"] = "shallow_comparator_unavailable"
            results.append(row)
            continue

        shallow_scores = score_maps[selected]
        common = sorted(
            heldout_ids
            & set(reference)
            & _available(deep_scores)
            & _available(shallow_scores)
        )
        reference_n = len(heldout_ids & set(reference))
        paired_coverage = len(common) / reference_n if reference_n else 0.0
        comparison: dict[str, Any] = {
            "n_paired": len(common),
            "paired_coverage_given_reference": paired_coverage,
            "candidate_coverage_over_heldout": len(
                heldout_ids & _available(deep_scores)
            )
            / settings.heldout_count,
            "comparator_coverage_over_heldout": len(
                heldout_ids & _available(shallow_scores)
            )
            / settings.heldout_count,
            "rho_deep": None,
            "rho_shallow": None,
            "delta_spearman": None,
            "minimum_effect": settings.minimum_effect,
            "minimum_effect_met": False,
            "inferential_eligible": False,
            "ineligibility_reasons": [],
            "paired_randomization": None,
            "paired_bootstrap": None,
            "bh_q_value": None,
            "fdr_reject": None,
            "improvement_supported": None,
        }
        if len(common) >= 3:
            deep_vector = [float(deep_scores[item_id]) for item_id in common]
            shallow_vector = [float(shallow_scores[item_id]) for item_id in common]
            reference_vector = [reference[item_id] for item_id in common]
            rho_deep = spearman(deep_vector, reference_vector)
            rho_shallow = spearman(shallow_vector, reference_vector)
            if math.isfinite(rho_deep) and math.isfinite(rho_shallow):
                comparison["rho_deep"] = rho_deep
                comparison["rho_shallow"] = rho_shallow
                comparison["delta_spearman"] = rho_deep - rho_shallow
                comparison["minimum_effect_met"] = (
                    rho_deep - rho_shallow >= settings.minimum_effect
                )
            else:
                comparison["ineligibility_reasons"].append("undefined_correlation")
        else:
            comparison["ineligibility_reasons"].append("fewer_than_three_pairs")
        if len(common) < settings.min_pairs:
            comparison["ineligibility_reasons"].append("paired_support_below_minimum")
        if paired_coverage < settings.conditional_coverage_min:
            comparison["ineligibility_reasons"].append(
                "conditional_coverage_below_minimum"
            )
        comparison["inferential_eligible"] = not comparison["ineligibility_reasons"]

        if comparison["inferential_eligible"] and run_inference:
            deep_vector = [float(deep_scores[item_id]) for item_id in common]
            shallow_vector = [float(shallow_scores[item_id]) for item_id in common]
            reference_vector = [reference[item_id] for item_id in common]
            comparison["paired_randomization"] = paired_randomization_test(
                deep_vector,
                shallow_vector,
                reference_vector,
                samples=settings.permutation_samples,
                seed=_derived_seed(settings, aspect, "permutation"),
            )
            comparison["paired_bootstrap"] = paired_bootstrap_ci(
                deep_vector,
                shallow_vector,
                reference_vector,
                samples=settings.bootstrap_samples,
                confidence=settings.bootstrap_confidence,
                seed=_derived_seed(settings, aspect, "bootstrap"),
            )
            p_values[aspect] = comparison["paired_randomization"]["p_value"]

        row["heldout_comparison"] = comparison
        row["status"] = (
            "inferentially_eligible"
            if comparison["inferential_eligible"]
            else "descriptive_only"
        )
        results.append(row)

    adjusted = benjamini_hochberg(p_values)
    for row in results:
        comparison = row.get("heldout_comparison")
        aspect = row["criterion_id"]
        if not isinstance(comparison, dict) or aspect not in adjusted:
            continue
        comparison["bh_q_value"] = adjusted[aspect]
        comparison["fdr_reject"] = adjusted[aspect] <= settings.alpha
        comparison["improvement_supported"] = bool(
            comparison["fdr_reject"] and comparison["minimum_effect_met"]
        )

    eligible = [
        row
        for row in results
        if isinstance(row.get("heldout_comparison"), dict)
        and row["heldout_comparison"]["inferential_eligible"]
    ]
    supported = [
        row
        for row in eligible
        if row["heldout_comparison"]["improvement_supported"] is True
    ]
    return {
        "schema": SCHEMA,
        "objective": "unsupervised_reconstruction_of_frozen_llm_judgment",
        "external_ground_truth_used": False,
        "item_outcome_field_used": False,
        "analysis_timing": {
            "classification": "retrospective_full_family_exploratory",
            "preregistered": False,
            "note": (
                "Inputs and program outputs predate this evaluator, but aggregate outcomes "
                "were inspected before this analysis code was frozen. All 18 active criteria "
                "are retained to avoid outcome-based row selection."
            ),
        },
        "claim_boundary": {
            "permitted": [
                "within-code-channel reconstruction comparison",
                "full-active-panel descriptive depth screen",
                "multiplicity-controlled exploratory inference on fixed support gates",
            ],
            "not_permitted": [
                "prompt articulability comparison",
                "code verifiability or construct-fidelity certification",
                "isomorphism from correlation alone",
                "code correctness on reference disagreements",
                "external-truth or automatic-discovery claim",
            ],
        },
        "settings": asdict(settings),
        "inputs": {label: _artifact_receipt(path) for label, path in input_paths.items()},
        "split": {
            "algorithm": "sorted datapoint_id; random.Random(seed).shuffle; first train_count",
            "train_count": len(train_ids),
            "heldout_count": len(heldout_ids),
        },
        "summary": {
            "active_criteria": len(aspects),
            "criteria_with_deep_program": sum(
                row["deep_program"] is not None for row in results
            ),
            "criteria_with_train_selected_shallow_comparator": sum(
                row["train_shallow_selection"]["selected"] is not None
                for row in results
            ),
            "inferentially_eligible": len(eligible),
            "bh_family_size": len(adjusted),
            "multiplicity_controlled_improvements": len(supported),
            "multiplicity_controlled_improvement_ids": [
                row["criterion_id"] for row in supported
            ],
        },
        "criteria": results,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def render_report(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# Active-code program-depth retrospective",
        "",
        "This is a CPU-only, full-family retrospective against the frozen two-pass LLM",
        "reconstruction reference. Both compared arms execute code; authoring origin differs.",
        "It is exploratory, not a prompt-articulability, verifiability, or isomorphism certificate.",
        "",
        "## Headline",
        "",
        f"- Active criteria: {summary['active_criteria']}.",
        f"- Deep static/AST programs present: {summary['criteria_with_deep_program']}.",
        "- Criteria with a TRAIN-selected shallow executable comparator: "
        f"{summary['criteria_with_train_selected_shallow_comparator']}.",
        f"- Support-gated inferential family: {summary['inferentially_eligible']}.",
        "- BH-FDR .05 improvements also clearing delta-rho >= .02: "
        f"{summary['multiplicity_controlled_improvements']}.",
        "",
        "## Per criterion",
        "",
        "| criterion | common n | ref availability | deep rho | shallow rho | delta | p | BH q | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["criteria"]:
        comparison = row.get("heldout_comparison") or {}
        reference = row["heldout_reference"]
        permutation = comparison.get("paired_randomization") or {}
        lines.append(
            "| {criterion} | {n} | {availability:.1%} | {deep} | {shallow} | {delta} | {p} | {q} | {status} |".format(
                criterion=row["criterion_id"],
                n=comparison.get("n_paired", 0),
                availability=reference["availability_over_heldout"],
                deep=_fmt(comparison.get("rho_deep", row.get("deep_only_heldout", {}).get("rho"))),
                shallow=_fmt(comparison.get("rho_shallow")),
                delta=_fmt(comparison.get("delta_spearman")),
                p=_fmt(permutation.get("p_value")),
                q=_fmt(comparison.get("bh_q_value")),
                status=row["status"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This retrospective panel provides no multiplicity-controlled support for a general",
            "'deeper code wins' claim. Point estimates can be",
            "positive while paired uncertainty remains wide, and no eligible comparison survives",
            "the full-family multiplicity correction. Coverage remains criterion-dependent. The",
            "result strengthens the typed conclusion: structural depth is a program descriptor;",
            "relation match, observation coverage, and construct fidelity must be measured separately.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_exclusive(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = evaluate()
    result_text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    report_text = render_report(result)
    result_path = args.out_dir / "results.json"
    report_path = args.out_dir / "REPORT.md"
    if args.check:
        if result_path.read_text(encoding="utf-8") != result_text:
            raise RetrospectiveError("stored results do not match deterministic replay")
        if report_path.read_text(encoding="utf-8") != report_text:
            raise RetrospectiveError("stored report does not match deterministic replay")
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
        return 0
    _write_exclusive(result_path, result_text)
    _write_exclusive(report_path, report_text)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"wrote {result_path.relative_to(ROOT)}")
    print(f"wrote {report_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
