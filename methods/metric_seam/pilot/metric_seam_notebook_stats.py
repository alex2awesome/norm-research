"""Recompute the adjudicated metric-seam notebook statistics.

This module is deliberately about the scientific readouts rather than artifact
provenance.  It keeps four denominators separate:

* the 159-criterion census panel;
* the 142 panel criteria whose modern contracts tag every counterfactual probe
  as CODE or L;
* the currently materialized census cells and promotion queue; and
* the nine WS4 typed-DAG programs.

Historical ``CAM`` keys are read for compatibility, but the returned names use
``code_reconstruction`` and ``hybrid_reconstruction``.  The latter is not
prompt articulability or pure-code codability.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/metric_seam_pilot"
EFFORT = BASE / "battery/effort_ladder"
HIERARCHY = BASE / "hierarchy_r123"
DOMAIN_ORDER = (
    "press_releases",
    "creative_writing",
    "math",
    "humor",
    "legal_title_vii",
    "peer_review",
    "legal_ss_disability",
)
THRESHOLDS = (0.30, 0.50, 0.80)

# These eleven cells completed but were not placed in PROMOTION_QUEUE.json.
# Values are the final independently adjudicated SEP counts, not intermediate
# self-authored or h0 checks.  Keeping the reconciliation here makes the
# notebook's 43-cell denominator inspectable.
_NONQUEUE_FINAL_CELLS = {
    ("creative_writing", "a18"): (3, 5, False),
    ("creative_writing", "a198"): (3, 5, False),
    ("creative_writing", "a279"): (4, 5, False),
    ("creative_writing", "a315"): (2, 6, False),
    ("creative_writing", "a333"): (6, 6, True),
    ("creative_writing", "a54"): (4, 6, False),
    ("creative_writing", "a72"): (1, 5, False),
    ("creative_writing", "a9"): (3, 5, False),
    ("humor", "a306"): (4, 6, False),
    ("math", "a48"): (1, 4, False),
    ("peer_review", "a0"): (2, 5, False),
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


_TASK_SURVEY_TABLES = (
    ("math", "survey", ("tasks/math/seam_table.json",)),
    (
        "code_review",
        "survey",
        (
            # The comments-only historical survey moved when the active coding
            # lane began.  Prefer the archive, but retain the old location as a
            # portability fallback for pre-migration checkouts.
            "tasks/code_review/archive_pre_e2ladder/seam_table.json",
            "tasks/code_review/seam_table.json",
        ),
    ),
    ("patents", "survey", ("tasks/patents/seam_table.json",)),
    (
        "code_review_diffs",
        "follow-up",
        ("tasks/code_review_diffs/seam_table.json",),
    ),
    (
        "code_competition",
        "follow-up",
        ("tasks/code_competition/seam_table.json",),
    ),
    ("pr_exec", "follow-up", ("tasks/pr_exec/seam_table.json",)),
)


def optional_seam_table(path: Path) -> dict[str, Any]:
    """Load a seam table without converting absence into an empty result.

    Missing artifacts are a normal state while technical lanes are being built.
    ``row_count=None`` and ``rows=None`` distinguish that state from a present,
    legitimately empty table.  Existing malformed artifacts still fail loudly.
    """

    path = Path(path)
    if not path.is_file():
        return {
            "status": "unavailable_missing_artifact",
            "source_path": None,
            "expected_path": str(path),
            "row_count": None,
            "rows": None,
        }
    payload = _read_json(path)
    rows = payload.get("table") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"malformed seam table: {path}")
    return {
        "status": "available",
        "source_path": str(path),
        "expected_path": str(path),
        "row_count": len(rows),
        "rows": rows,
    }


def survey_task_tables(base: Path = BASE) -> dict[str, Any]:
    """Load the historical task surveys and retain explicit missingness.

    The returned ``rows`` contain only rows from present artifacts.  Every
    expected task remains represented in ``sources``; unavailable tasks have a
    null row count, never a fabricated zero.  Parse/schema failures are not
    treated as missing artifacts.
    """

    base = Path(base)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for task, wave, relative_candidates in _TASK_SURVEY_TABLES:
        candidates = [base / relative for relative in relative_candidates]
        selected = next((path for path in candidates if path.is_file()), None)
        if selected is None:
            sources.append(
                {
                    "task": task,
                    "wave": wave,
                    "status": "unavailable_missing_artifact",
                    "row_count": None,
                    "source_path": None,
                    "expected_paths": [str(path) for path in candidates],
                }
            )
            continue
        loaded = optional_seam_table(selected)
        table_rows = loaded["rows"]
        if table_rows is None:  # pragma: no cover - selected was checked above
            raise RuntimeError(f"selected seam table disappeared: {selected}")
        sources.append(
            {
                "task": task,
                "wave": wave,
                "status": "available",
                "row_count": len(table_rows),
                "source_path": str(selected),
                "expected_paths": [str(path) for path in candidates],
            }
        )
        for raw_row in table_rows:
            row = dict(raw_row)
            row["task"] = task
            row["wave"] = wave
            rows.append(row)
    unavailable = [source["task"] for source in sources if source["status"] != "available"]
    return {
        "expected_task_count": len(_TASK_SURVEY_TABLES),
        "available_task_count": len(_TASK_SURVEY_TABLES) - len(unavailable),
        "unavailable_task_count": len(unavailable),
        "unavailable_tasks": unavailable,
        "sources": sources,
        "rows": rows,
    }


def _pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 1) if denominator else float("nan")


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("Wilson interval requires 0 <= successes <= positive total")
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    radius = (
        z
        * (
            proportion * (1 - proportion) / total
            + z**2 / (4 * total**2)
        )
        ** 0.5
        / denominator
    )
    return [center - radius, center + radius]


def _fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value using the probability-ordering definition."""

    if min(a, b, c, d) < 0:
        raise ValueError("contingency counts must be nonnegative")
    first_total = a + b
    second_total = c + d
    success_total = a + c
    population = first_total + second_total
    denominator = math.comb(population, first_total)

    def probability(value: int) -> float:
        return (
            math.comb(success_total, value)
            * math.comb(population - success_total, first_total - value)
            / denominator
        )

    lower = max(0, first_total - (population - success_total))
    upper = min(first_total, success_total)
    observed = probability(a)
    return min(
        1.0,
        sum(
            probability(value)
            for value in range(lower, upper + 1)
            if probability(value) <= observed + 1e-15
        ),
    )


def _midranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("rank vectors differ in length")
    if len(left) < 3 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    x = _midranks(left)
    y = _midranks(right)
    mx = statistics.mean(x)
    my = statistics.mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    denominator = (
        sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)
    ) ** 0.5
    return numerator / denominator if denominator else None


def _contract_path(task: str, aspect: str) -> tuple[str, Path]:
    stem = f"{task}__{aspect}.json"
    choices = (
        ("v3", EFFORT / "contracts_v3" / stem),
        ("v2", EFFORT / "contracts_v2" / stem),
        ("v1", EFFORT / "contracts" / stem),
    )
    for generation, path in choices:
        if path.exists():
            return generation, path
    raise FileNotFoundError(stem)


def panel_rows() -> list[dict[str, Any]]:
    """Join the frozen 159-criterion panel to historical code/hybrid readouts."""

    panel = _read_json(EFFORT / "panel_v3_census.json")["census"]
    cam = _read_json(BASE / "cam_profile.json")
    readout = {
        (task, row["aspect"]): row
        for task, task_record in cam.items()
        for row in task_record.get("per_criterion", [])
    }
    rows: list[dict[str, Any]] = []
    for task in DOMAIN_ORDER:
        for panel_row in panel[task]:
            aspect = panel_row["aspect"]
            metric = readout[(task, aspect)]
            generation, contract_path = _contract_path(task, aspect)
            contract = _read_json(contract_path)
            channels = [probe.get("channel") for probe in contract.get("cf_probes", [])]
            fully_tagged = bool(channels) and all(value in {"CODE", "L"} for value in channels)
            rows.append(
                {
                    "task": task,
                    "aspect": aspect,
                    "band": panel_row["band"],
                    "code_reconstruction": float(metric["r_base"]),
                    "hybrid_reconstruction": float(metric["r_hyb"]),
                    "contract_generation": generation,
                    "probe_count": len(channels),
                    "code_probe_count": sum(value == "CODE" for value in channels),
                    "l_probe_count": sum(value == "L" for value in channels),
                    "fully_channel_tagged": fully_tagged,
                    "code_probe_share": (
                        sum(value == "CODE" for value in channels) / len(channels)
                        if fully_tagged
                        else None
                    ),
                }
            )
    if len(rows) != 159:
        raise ValueError(f"expected 159 panel rows, found {len(rows)}")
    return rows


def _threshold_counts(rows: Iterable[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows]
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        **{
            f"ge_{threshold:.2f}": {
                "count": sum(value >= threshold for value in values),
                "pct": _pct(sum(value >= threshold for value in values), len(values)),
            }
            for threshold in THRESHOLDS
        },
    }


def codability_by_domain() -> list[dict[str, Any]]:
    """Return domain summaries with code and hybrid estimands kept distinct."""

    rows = panel_rows()
    output: list[dict[str, Any]] = []
    for task in (*DOMAIN_ORDER, "ALL"):
        selected = rows if task == "ALL" else [row for row in rows if row["task"] == task]
        code = _threshold_counts(selected, "code_reconstruction")
        hybrid = _threshold_counts(selected, "hybrid_reconstruction")
        output.append(
            {
                "task": task,
                "n": len(selected),
                "code_mean": code["mean"],
                "code_median": code["median"],
                "code_ge_30_n": code["ge_0.30"]["count"],
                "code_ge_30_pct": code["ge_0.30"]["pct"],
                "code_ge_50_n": code["ge_0.50"]["count"],
                "code_ge_50_pct": code["ge_0.50"]["pct"],
                "code_ge_80_n": code["ge_0.80"]["count"],
                "code_ge_80_pct": code["ge_0.80"]["pct"],
                "hybrid_mean": hybrid["mean"],
                "hybrid_ge_30_n": hybrid["ge_0.30"]["count"],
                "hybrid_ge_30_pct": hybrid["ge_0.30"]["pct"],
                "hybrid_ge_50_n": hybrid["ge_0.50"]["count"],
                "hybrid_ge_50_pct": hybrid["ge_0.50"]["pct"],
                "hybrid_ge_80_n": hybrid["ge_0.80"]["count"],
                "hybrid_ge_80_pct": hybrid["ge_0.80"]["pct"],
            }
        )
    return output


def channel_contract_summary() -> dict[str, Any]:
    """Summarize articulated CODE/L hypotheses, not empirical successes."""

    rows = panel_rows()
    tagged = [row for row in rows if row["fully_channel_tagged"]]
    per_domain: list[dict[str, Any]] = []
    for task in DOMAIN_ORDER:
        selected = [row for row in tagged if row["task"] == task]
        n_code = sum(row["code_probe_count"] for row in selected)
        n_l = sum(row["l_probe_count"] for row in selected)
        per_domain.append(
            {
                "task": task,
                "tagged_criteria": len(selected),
                "tagged_probes": n_code + n_l,
                "code_probes": n_code,
                "l_probes": n_l,
                "code_probe_pct": _pct(n_code, n_code + n_l),
                "code_reconstruction_mean": round(
                    statistics.mean(row["code_reconstruction"] for row in selected), 3
                ),
            }
        )
    criterion_classes = Counter()
    for row in tagged:
        if row["code_probe_count"] == row["probe_count"]:
            criterion_classes["all_CODE"] += 1
        elif row["l_probe_count"] == row["probe_count"]:
            criterion_classes["all_L"] += 1
        else:
            criterion_classes["mixed_CODE_L"] += 1
    n_code = sum(row["code_probe_count"] for row in tagged)
    n_l = sum(row["l_probe_count"] for row in tagged)
    per_band: list[dict[str, Any]] = []
    for band in ("floor", "mid", "control"):
        selected = [row for row in rows if row["band"] == band]
        band_code = sum(row["code_probe_count"] for row in selected)
        band_l = sum(row["l_probe_count"] for row in selected)
        untyped = sum(
            row["probe_count"] - row["code_probe_count"] - row["l_probe_count"]
            for row in selected
        )
        per_band.append(
            {
                "band": band,
                "code_probes": band_code,
                "typed_probes": band_code + band_l,
                "untyped_probes": untyped,
                "code_probe_pct": _pct(band_code, band_code + band_l),
            }
        )
    pooled_rho = _spearman(
        [float(row["code_probe_share"]) for row in tagged],
        [row["code_reconstruction"] for row in tagged],
    )
    domain_rho = _spearman(
        [row["code_probe_pct"] for row in per_domain],
        [row["code_reconstruction_mean"] for row in per_domain],
    )
    return {
        "panel_criteria": len(rows),
        "fully_tagged_criteria": len(tagged),
        "legacy_untagged_criteria": len(rows) - len(tagged),
        "tagged_probes": n_code + n_l,
        "code_probes": n_code,
        "l_probes": n_l,
        "code_probe_pct": _pct(n_code, n_code + n_l),
        "criterion_classes": dict(criterion_classes),
        "code_tag_share_vs_code_reconstruction": {
            "criterion_level_n": len(tagged),
            "criterion_level_spearman": round(float(pooled_rho), 3),
            "domain_level_n": len(per_domain),
            "domain_level_spearman": round(float(domain_rho), 3),
            "interpretation": (
                "The strong domain association and weak pooled criterion association are "
                "descriptive and post hoc; tags were not authored independently of domain knowledge."
            ),
        },
        "per_domain": per_domain,
        "per_band": per_band,
    }


def census_progress() -> list[dict[str, Any]]:
    """Count materialized cells and train-only contract-pass queue entries."""

    panel = _read_json(EFFORT / "panel_v3_census.json")["census"]
    queue = _read_json(EFFORT / "census/PROMOTION_QUEUE.json")["queue"]
    queued_by_task = Counter(row["task"] for row in queue)
    output: list[dict[str, Any]] = []
    for task in DOMAIN_ORDER:
        attempted = sum(
            (EFFORT / "census" / f"{task}__{row['aspect']}" / "meta.json").exists()
            for row in panel[task]
        )
        output.append(
            {
                "task": task,
                "attempted": attempted,
                "panel_n": len(panel[task]),
                "attempted_pct": _pct(attempted, len(panel[task])),
                "train_contract_queue_n": queued_by_task[task],
                "queue_pct_of_attempted": _pct(queued_by_task[task], attempted),
            }
        )
    output.append(
        {
            "task": "ALL",
            "attempted": sum(row["attempted"] for row in output),
            "panel_n": sum(row["panel_n"] for row in output),
            "attempted_pct": _pct(
                sum(row["attempted"] for row in output),
                sum(row["panel_n"] for row in output),
            ),
            "train_contract_queue_n": len(queue),
            "queue_pct_of_attempted": _pct(
                len(queue), sum(row["attempted"] for row in output)
            ),
        }
    )
    return output


def census_outcome_summary() -> dict[str, Any]:
    """Reconcile the queue with final outcomes for all 43 completed cells."""

    import re

    queue = _read_json(EFFORT / "census/PROMOTION_QUEUE.json")["queue"]
    queued_separations: list[tuple[int, int]] = []
    for row in queue:
        match = re.search(r"(\d+)\s*/\s*(\d+)", row["contract"])
        if match is None:
            raise ValueError(f"missing SEP fraction in queue row: {row}")
        queued_separations.append((int(match.group(1)), int(match.group(2))))
    final_passes = len(queue) + sum(
        passed for _, _, passed in _NONQUEUE_FINAL_CELLS.values()
    )
    separation_n = sum(value for value, _ in queued_separations) + sum(
        value for value, _, _ in _NONQUEUE_FINAL_CELLS.values()
    )
    separation_den = sum(value for _, value in queued_separations) + sum(
        value for _, value, _ in _NONQUEUE_FINAL_CELLS.values()
    )
    return {
        "attempted_cells": len(queue) + len(_NONQUEUE_FINAL_CELLS),
        "final_contract_passes": final_passes,
        "final_contract_pass_pct": _pct(
            final_passes, len(queue) + len(_NONQUEUE_FINAL_CELLS)
        ),
        "final_separations": separation_n,
        "final_separation_opportunities": separation_den,
        "final_separation_pct": _pct(separation_n, separation_den),
        "queued_passes": len(queue),
        "nonqueued_pass_ids": [
            f"{task}__{aspect}"
            for (task, aspect), (_, _, passed) in _NONQUEUE_FINAL_CELLS.items()
            if passed
        ],
        "queue_note": (
            "creative_writing__a333 passed 6/6 but was not enqueued; the artifacts do "
            "not explicitly record why."
        ),
    }


@lru_cache(maxsize=1)
def census_probe_channel_replay() -> dict[str, Any]:
    """Freshly replay all 43 code-only candidates against their synthetic probes.

    This is intentionally separate from ``census_outcome_summary``: it loads and
    executes candidate programs, so it is slower.  The readout diagnoses the
    contract instrument.  In particular, success on an authored-L probe is a
    code proxy separating a synthetic pair, not evidence that the L relation
    has become generally codable.
    """

    battery_dir = ROOT / "methods/metric_seam/battery"
    sys.path.insert(0, str(battery_dir))
    from battery_common import load_ctx, load_mod  # type: ignore[import-not-found]
    from contract_check import probe_score  # type: ignore[import-not-found]

    contexts: dict[str, Any] = {}
    counts: Counter[tuple[str, str]] = Counter()
    candidate_paths = sorted((EFFORT / "census").glob("*/candidate.py"))
    for candidate_path in candidate_paths:
        task, aspect = candidate_path.parent.name.split("__", 1)
        if task not in contexts:
            contexts[task] = load_ctx(task)
        contract = _read_json(EFFORT / "contracts" / f"{task}__{aspect}.json")
        module = load_mod(candidate_path)
        for probe in contract["cf_probes"]:
            positive, _ = probe_score(
                module.score, probe["text_pos"], contexts[task]["ops"]
            )
            negative, _ = probe_score(
                module.score, probe["text_neg"], contexts[task]["ops"]
            )
            outcome = (
                "NONE"
                if positive is None or negative is None
                else "SEP"
                if positive > negative
                else "INV"
                if positive < negative
                else "TIE"
            )
            counts[(str(probe.get("channel", "MISSING")).upper(), outcome)] += 1

    if len(candidate_paths) != 43:
        raise ValueError(f"expected 43 census candidates, found {len(candidate_paths)}")
    by_channel: dict[str, dict[str, Any]] = {}
    for channel in ("CODE", "L"):
        outcomes = {
            outcome: counts[(channel, outcome)]
            for outcome in ("SEP", "TIE", "INV", "NONE")
        }
        total = sum(outcomes.values())
        by_channel[channel] = {
            **outcomes,
            "total": total,
            "separation_pct": _pct(outcomes["SEP"], total),
        }
    return {
        "candidate_count": len(candidate_paths),
        "by_authored_channel": by_channel,
        "interpretation": (
            "Synthetic probe separation is train/probe-local. L-probe separation often "
            "reflects a code proxy and must not be relabeled as observed relation codability."
        ),
    }


def creative_writing_heldout_adjudication() -> dict[str, Any]:
    """Return both the historical exploratory labels and the corrected inference."""

    report = _read_json(EFFORT / "census/cw_heldout_report.json")
    rows = {key: value for key, value in report.items() if not key.startswith("_")}
    addendum = report["_multiplicity_and_threshold_addendum"]
    exploratory = sorted(
        key for key, value in rows.items() if value.get("verdict") == "PROMOTED"
    )
    both_bars = sorted(
        key
        for key, value in rows.items()
        if value.get("verdict") == "PROMOTED" and value.get("G1_verdict") == "PASS"
    )
    unambiguous = [
        value for value in rows.values() if not value.get("low_judge_coverage", False)
    ]
    bh = addendum["multiplicity_BH_FDR_0.10"]
    return {
        "candidate_count": len(rows),
        "unambiguous_count": len(unambiguous),
        "exploratory_pairwise_count": len(exploratory),
        "exploratory_pairwise_ids": exploratory,
        "g1_and_pairwise_count": len(both_bars),
        "g1_and_pairwise_ids": both_bars,
        "bh_test_count": int(bh["n_tests"]),
        "bh_survivor_count": len(bh["pass"]),
        "bh_survivor_ids": list(bh["pass"]),
        "threshold_was_preregistered": False,
    }


def _load_program(path: Path) -> dict[str, Any]:
    name = "metric_seam_ws4_" + path.parent.name.replace("__", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PROG


def ws4_depth_rows() -> list[dict[str, Any]]:
    """Compute graph depth from the nine executable typed DAGs."""

    rows: list[dict[str, Any]] = []
    for path in sorted((EFFORT / "ws4").glob("*/dag_program.py")):
        program = _load_program(path)
        nodes = {node["id"]: node for node in program["nodes"]}

        @lru_cache(maxsize=None)
        def graph_depth(node_id: str) -> int:
            needs = nodes[node_id]["needs"]
            return 0 if not needs else 1 + max(graph_depth(parent) for parent in needs)

        children = {node_id: [] for node_id in nodes}
        for node in nodes.values():
            for parent in node["needs"]:
                children[parent].append(node["id"])

        @lru_cache(maxsize=None)
        def ancestors(node_id: str) -> frozenset[str]:
            result: set[str] = set()
            for parent in nodes[node_id]["needs"]:
                result.add(parent)
                result.update(ancestors(parent))
            return frozenset(result)

        @lru_cache(maxsize=None)
        def longest_to_output(node_id: str) -> int | None:
            if node_id == program["out"]:
                return 0
            downstream = [
                distance
                for child in children[node_id]
                if (distance := longest_to_output(child)) is not None
            ]
            return 1 + max(downstream) if downstream else None

        l_nodes = [node for node in program["nodes"] if node["impl"] == "L"]
        l_ids = [node["id"] for node in l_nodes]
        l_frontier = [
            node_id
            for node_id in l_ids
            if not any(other in ancestors(node_id) for other in l_ids)
        ]
        cell = path.parent.name
        task = cell.split("__", 1)[0]
        rows.append(
            {
                "cell": cell,
                "task": task,
                "n_nodes": len(nodes),
                "n_code_nodes": sum(node["impl"] == "C" for node in program["nodes"]),
                "n_l_nodes": len(l_nodes),
                "longest_path_edges": max(graph_depth(node_id) for node_id in nodes),
                "output_depth_edges": graph_depth(program["out"]),
                "l_graph_depths": [graph_depth(node["id"]) for node in l_nodes],
                "l_abstraction_levels": [int(node["level"]) for node in l_nodes],
                "retrieval_nodes": sum(
                    node["ntype"] == "R" for node in program["nodes"]
                ),
                "evidence_nodes": sum(
                    node["op_class"] == "evidence" for node in program["nodes"]
                ),
                "l_frontier_to_output_longest_edges": [
                    longest_to_output(node_id) for node_id in l_frontier
                ],
            }
        )
    if len(rows) != 9:
        raise ValueError(f"expected nine WS4 programs, found {len(rows)}")
    return rows


def ws4_depth_summary() -> dict[str, Any]:
    rows = ws4_depth_rows()
    total_nodes = sum(row["n_nodes"] for row in rows)
    total_l = sum(row["n_l_nodes"] for row in rows)
    l_depths = [depth for row in rows for depth in row["l_graph_depths"]]
    l_levels = [level for row in rows for level in row["l_abstraction_levels"]]
    frontier_distances = [
        distance
        for row in rows
        for distance in row["l_frontier_to_output_longest_edges"]
        if distance is not None
    ]
    return {
        "programs": len(rows),
        "nodes": total_nodes,
        "median_nodes": statistics.median(row["n_nodes"] for row in rows),
        "node_range": [min(row["n_nodes"] for row in rows), max(row["n_nodes"] for row in rows)],
        "l_nodes": total_l,
        "l_node_pct": _pct(total_l, total_nodes),
        "median_longest_path_edges": statistics.median(
            row["longest_path_edges"] for row in rows
        ),
        "longest_path_range": [
            min(row["longest_path_edges"] for row in rows),
            max(row["longest_path_edges"] for row in rows),
        ],
        "l_nodes_at_graph_root": sum(depth == 0 for depth in l_depths),
        "l_level_counts": dict(Counter(l_levels)),
        "retrieval_nodes": sum(row["retrieval_nodes"] for row in rows),
        "evidence_nodes": sum(row["evidence_nodes"] for row in rows),
        "mean_l_frontier_to_output_longest_edges": round(
            statistics.mean(frontier_distances), 2
        ),
        "median_l_frontier_to_output_longest_edges": statistics.median(
            frontier_distances
        ),
        "l_frontier_to_output_range": [
            min(frontier_distances),
            max(frontier_distances),
        ],
        "rows": rows,
    }


def active_code_depth_retrospective() -> dict[str, Any]:
    """Read the full-family deep-vs-shallow active-code comparison.

    This is a retrospective comparison of two executable-code arms.  It is not
    a prompt-articulability test and it is deliberately kept separate from the
    WS4 graph descriptors: source/program depth and reconstruction signal are
    different estimands.
    """

    path = (
        BASE
        / "reconstruction_v2/code_depth_full_panel_retrospective_002/results.json"
    )
    result = _read_json(path)
    if result.get("schema") != "metric-seam.code-depth-full-panel-retrospective.v2":
        raise ValueError(f"unexpected active-code depth schema: {result.get('schema')}")
    rows = result.get("criteria")
    if not isinstance(rows, list) or len(rows) != 18:
        raise ValueError(f"expected 18 active-code criteria, found {len(rows or [])}")

    eligible = [
        row
        for row in rows
        if (row.get("heldout_comparison") or {}).get("inferential_eligible") is True
    ]
    improved = [
        row
        for row in eligible
        if (row.get("heldout_comparison") or {}).get("improvement_supported") is True
    ]
    positive_descriptive = [
        row
        for row in rows
        if isinstance((row.get("heldout_comparison") or {}).get("delta_spearman"), (int, float))
        and row["heldout_comparison"]["delta_spearman"] > 0
    ]
    output_rows = []
    for row in rows:
        comparison = row.get("heldout_comparison") or {}
        deep_only = row.get("deep_only_heldout") or {}
        output_rows.append(
            {
                "criterion_id": row["criterion_id"],
                "name": row.get("name", ""),
                "status": row.get("status"),
                "n_paired": comparison.get("n_paired"),
                "reference_availability": (
                    row.get("heldout_reference") or {}
                ).get("availability_over_heldout"),
                "deep_rho": comparison.get("rho_deep", deep_only.get("rho")),
                "shallow_rho": comparison.get("rho_shallow"),
                "delta_spearman": comparison.get("delta_spearman"),
                "p_value": (comparison.get("paired_randomization") or {}).get("p_value"),
                "bh_q_value": comparison.get("bh_q_value"),
                "ci_low": (comparison.get("paired_bootstrap") or {}).get("interval", [None, None])[0],
                "ci_high": (comparison.get("paired_bootstrap") or {}).get("interval", [None, None])[1],
            }
        )

    return {
        **result["summary"],
        "descriptive_positive_delta_count": len(positive_descriptive),
        "descriptive_comparison_count": sum(
            row["delta_spearman"] is not None for row in output_rows
        ),
        "multiplicity_controlled_improvements": len(improved),
        "rows": output_rows,
        "interpretation": (
            "Retrospective full-family code-versus-code comparison. No support-gated "
            "criterion survived BH-FDR; program depth is not itself reconstruction evidence."
        ),
    }


def active_code_a104_supplemental() -> dict[str, Any]:
    """Return criterion-local representation and execution supplements for a104.

    Neither supplement changes the frozen V4 result or the 18-criterion family.
    The representation comparison is one-sided because the historical LLM
    reference remains on head/tail text while only code receives the prefix.
    The execution bridge uses stored repository telemetry that is unavailable
    in presented diff ``ctext``, so it is capability augmentation rather than
    an isomorphic substitution.
    """

    sensitivity = _read_json(
        BASE / "tasks/code_review/a104_representation_sensitivity_v1.json"
    )
    bridge = _read_json(
        BASE / "tasks/code_review/a104_execution_telemetry_bridge_v1.json"
    )
    if (
        sensitivity.get("schema")
        != "metric-seam.code-review-a104-representation-sensitivity.v1"
        or sensitivity.get("status") != "complete_posthoc_exploratory"
        or sensitivity.get("criterion") != "a104"
    ):
        raise ValueError("unexpected a104 representation-sensitivity artifact")
    if (
        bridge.get("schema_version")
        != "metric-seam.active-code-a104-execution-bridge.v1"
        or bridge.get("status") != "stored_execution_telemetry_join_complete"
        or bridge.get("criterion") != "a104"
    ):
        raise ValueError("unexpected a104 execution-telemetry bridge")

    crosswalk = sensitivity["crosswalk"]
    replay = sensitivity["frozen_replay"]
    support = sensitivity["score_support_all_250"]
    heldout = sensitivity["heldout_readout"]
    reference_order = sensitivity["blindness_and_reference_order"]
    if (
        crosswalk.get("exact_unique_prefix_matches") != 250
        or crosswalk.get("hierarchy_rows_at_cap") != 205
        or crosswalk.get("hierarchy_fraction_at_cap") != 0.82
        or replay.get("score_mismatches") != 0
        or heldout.get("common_support_n") != 93
        or support.get("applicability_status_changes") != 12
        or support.get("value_changes_on_common_scored") != 118
        or reference_order.get("prefix_candidate_matches_reference_input") is not False
        or reference_order.get("direct_same_input_prefix_prompt_code_test") is not False
        or reference_order.get("prefix_arm")
        != "one_sided_representation_mismatch_sensitivity"
    ):
        raise ValueError("a104 representation-sensitivity boundary drifted")
    expected_rhos = {
        "head_tail_rho_on_common": 0.6453742826894445,
        "prefix4000_rho_on_common": 0.5144084535611719,
        "delta_prefix_minus_head_tail": -0.1309658291282726,
        "head_tail_prefix_program_vector_rho": 0.7777488599159648,
    }
    if any(
        not math.isclose(
            float(heldout.get(key)), expected, rel_tol=0.0, abs_tol=5e-15
        )
        for key, expected in expected_rhos.items()
    ):
        raise ValueError("a104 representation-sensitivity readout drifted")

    summary = bridge["summary"]
    representation = bridge["representation_boundary"]
    provenance = bridge["execution_provenance"]
    if (
        summary.get("active_items") != 250
        or summary.get("exact_repository_pr_overlap") != 32
        or summary.get("finite_execution_certificates") != 1
        or summary.get("finite_certificate_rate_conditional_overlap") != 0.03125
        or summary.get("finite_certificate_rate_over_active_items") != 0.004
        or representation.get("same_input_representation") is not False
        or representation.get("classification")
        != "capability_augmentation_not_isomorphic_substitution"
        or provenance.get("relation_depth") != 4
        or provenance.get("stored_telemetry_from_prior_repository_execution")
        is not True
        or provenance.get("repositories_or_tests_executed_in_this_bridge")
        is not False
    ):
        raise ValueError("a104 execution-telemetry boundary drifted")

    return {
        "criterion": "a104",
        "representation_sensitivity": {
            "common_heldout_n": heldout["common_support_n"],
            "historical_head_tail_rho": heldout["head_tail_rho_on_common"],
            "prefix4000_rho": heldout["prefix4000_rho_on_common"],
            "delta_prefix_minus_head_tail": heldout[
                "delta_prefix_minus_head_tail"
            ],
            "code_vector_rho": heldout[
                "head_tail_prefix_program_vector_rho"
            ],
            "applicability_status_changes_all_250": support[
                "applicability_status_changes"
            ],
            "value_changes_on_common_scored_all_250": support[
                "value_changes_on_common_scored"
            ],
            "hierarchy_prefix_rows_at_cap": crosswalk[
                "hierarchy_rows_at_cap"
            ],
            "hierarchy_rows": crosswalk["hierarchy_rows"],
            "one_sided_not_same_input_prompt_code": True,
            "post_hoc_exploratory": True,
        },
        "execution_augmentation": {
            "active_items": summary["active_items"],
            "exact_repository_pr_overlap": summary[
                "exact_repository_pr_overlap"
            ],
            "overlap_rate": summary["overlap_rate"],
            "finite_execution_certificates": summary[
                "finite_execution_certificates"
            ],
            "finite_certificate_rate_conditional_overlap": summary[
                "finite_certificate_rate_conditional_overlap"
            ],
            "finite_certificate_rate_over_active_items": summary[
                "finite_certificate_rate_over_active_items"
            ],
            "relation_depth": provenance["relation_depth"],
            "stored_prior_execution": True,
            "same_input_representation": False,
            "capability_augmentation_not_isomorphic_substitution": True,
        },
        "interpretation": (
            "The first-4000 projection is materially lossy for this frozen checker, "
            "but the comparison is one-sided and post-hoc. A sparse stored execution "
            "join supplies one finite depth-4 witness; it is extra-environment evidence, "
            "not prompt/code isomorphism or a correctness/codability denominator."
        ),
    }


def code_review_representation_family_sensitivity() -> dict[str, Any]:
    """Summarize label/reference-free input-projection sensitivity.

    The primary aggregation unit is the ten unique programs supporting the
    corrected 18-mapping family.  Criterion mappings are a typed join only;
    they are never pooled as independent observations.  This is code/code
    representation robustness, not prompt articulability or reconstruction.
    """

    result = _read_json(
        HIERARCHY / "code_review_representation_family_sensitivity_v1.json"
    )
    contract = _read_json(
        HIERARCHY / "code_review_representation_family_analysis_contract_v1.json"
    )
    if (
        result.get("schema")
        != "metric-seam.code-review-representation-family-sensitivity.v1"
        or result.get("status")
        != "developmental_exploratory_family_audit_complete"
        or contract.get("schema")
        != "metric-seam.code-review-representation-family-analysis-contract.v1"
        or contract.get("status")
        != "frozen_developmental_before_family_execution"
    ):
        raise ValueError("code-review representation-family audit is incomplete")

    replay = result["P0_exact_frozen_replay"]
    population = result["population"]
    crosswalk = result["representation_crosswalk"]
    blindness = result["blindness_and_channel"]
    execution = result["execution_environment"]
    typed_join = result["typed_primary_criterion_join"]
    if replay != {
        "heldout_rows_exact": 2000,
        "mismatch_counts_by_program": {key: 0 for key in (
            "a0", "a1", "a112", "a131", "a15", "a18", "a37", "a38",
            "a401", "a43", "a47", "a52", "a70", "a76", "a8", "a92",
        )},
        "required_before_sensitivity_readout": True,
        "total_mismatches": 0,
        "total_rows_exact": 4000,
        "train_rows_exact": 2000,
    }:
        raise ValueError("code-review representation anchor replay drifted")
    if population != {
        "primary_is_subset_of_secondary": True,
        "primary_relation_mappings": 18,
        "primary_unique_programs": 10,
        "secondary_unique_programs": 16,
        "selection_loaded_outcomes_or_references": False,
    }:
        raise ValueError("code-review representation population drifted")
    if (
        crosswalk.get("P0_P1_exact_prefix_crosswalk_n") != 250
        or crosswalk.get("P1_P2_exact_canonicalization_n") != 250
        or crosswalk.get("P2_local_path_crosswalk_n") != 250
        or crosswalk.get("outcome_bearing_items_json_loaded") is not False
        or typed_join.get("aggregation_performed") is not False
        or typed_join.get("n_relation_mappings") != 18
        or len(typed_join.get("rows", [])) != 18
        or any(blindness.get(key) is not False for key in (
            "external_supervision_used",
            "gpu_or_accelerator_used",
            "llm_judgments_loaded",
            "models_or_apis_called",
            "outcome_bearing_items_json_loaded",
            "outcomes_loaded",
            "prompt_responses_loaded",
            "reconstruction_results_or_correlations_loaded",
            "references_loaded",
        ))
        or execution.get("live_sandbox_file_modified") is not False
        or execution.get("maximum_concurrent_cpu_workers") != 4
    ):
        raise ValueError("code-review representation channel boundary drifted")

    pair_order = (
        "P0_prefix4000 -> P1_head5000_tail2500",
        "P1_head5000_tail2500 -> P2_raw_diff_capped300k",
        "P0_prefix4000 -> P2_raw_diff_capped300k",
    )
    expected_macro = {
        pair_order[0]: (0.7068, 0.8556, 0.0708, 0.7280995460063534),
        pair_order[1]: (0.5556, 0.8108, 0.078, 0.604142312774589),
        pair_order[2]: (0.4588, 0.6956, 0.1488, 0.550178916821058),
    }
    primary_macro = result["primary_program_macro"]
    macro_rows = []
    for pair in pair_order:
        row = primary_macro[pair]
        expected = expected_macro[pair]
        observed = (
            row["mean_program_exact_row_agreement"],
            row["mean_program_status_agreement"],
            row["mean_program_applicability_change_rate"],
            row["mean_program_exact_value_agreement_on_common_scored"],
        )
        if (
            row.get("aggregation_unit") != "unique_program"
            or row.get("n_programs") != 10
            or row.get("pooled_criterion_mapping_estimate_emitted") is not False
            or row.get("sensitivity_class_counts")
            != {"status_or_applicability_sensitive": 10}
            or any(
                not math.isclose(value, target, rel_tol=0.0, abs_tol=5e-15)
                for value, target in zip(observed, expected)
            )
        ):
            raise ValueError(f"code-review representation macro drifted: {pair}")
        macro_rows.append({
            "comparison": pair,
            "programs": 10,
            "exact_row_agreement": observed[0],
            "status_agreement": observed[1],
            "applicability_change_rate": observed[2],
            "exact_value_agreement_on_common_scored": observed[3],
            "status_or_applicability_sensitive_programs": 10,
        })

    program_rows = result["primary_program_results"]
    if len(program_rows) != 10:
        raise ValueError("code-review primary program rows drifted")
    by_depth: dict[int, list[dict[str, Any]]] = {1: [], 2: []}
    for row in program_rows:
        depths = row.get("primary_matched_relation_depths")
        if not isinstance(depths, list) or not depths or max(depths) not in by_depth:
            raise ValueError("code-review primary program depth drifted")
        by_depth[max(depths)].append(row)
    if {depth: len(rows) for depth, rows in by_depth.items()} != {1: 7, 2: 3}:
        raise ValueError("code-review primary depth counts drifted")
    depth_descriptive = {}
    for depth, rows in by_depth.items():
        depth_descriptive[depth] = {
            "programs": len(rows),
            "comparisons": {
                pair: {
                    "mean_exact_row_agreement": statistics.mean(
                        row["pairwise_sensitivity"][pair]["exact_row_agreement"]
                        for row in rows
                    ),
                    "mean_exact_value_agreement_on_common_scored": statistics.mean(
                        row["pairwise_sensitivity"][pair][
                            "exact_value_agreement_on_common_scored"
                        ]
                        for row in rows
                    ),
                }
                for pair in pair_order
            },
        }

    join_rows = typed_join["rows"]
    return {
        "primary_unique_programs": population["primary_unique_programs"],
        "primary_relation_mappings": population["primary_relation_mappings"],
        "secondary_unique_programs": population["secondary_unique_programs"],
        "P0_exact_replay_rows": replay["total_rows_exact"],
        "P0_exact_replay_mismatches": replay["total_mismatches"],
        "crosswalk_rows": 250,
        "primary_program_macro": macro_rows,
        "primary_program_depth_counts": {
            depth: len(rows) for depth, rows in by_depth.items()
        },
        "primary_program_depth_descriptive": depth_descriptive,
        "typed_mapping_level_counts": dict(Counter(
            row["level"] for row in join_rows
        )),
        "typed_mapping_depth_counts": dict(Counter(
            int(row["matched_relation_depth"]) for row in join_rows
        )),
        "axes": {
            "prompt_articulability": "not_measured",
            "code_verifiability": "representation_sensitivity_measured_relation_local",
            "reconstruction_agreement": "not_measured",
            "isomorphism": "not_measured_code_code_projection_audit_only",
            "codability": "not_estimated",
        },
        "interpretation": (
            "All ten primary programs change status or applicability under every "
            "projection contrast. Input representation is therefore a family-wide "
            "property of these code measurements, not an a104-only anomaly. The "
            "depth-1 versus depth-2 split is exploratory with n=7 versus n=3."
        ),
    }


def active_code_source_structure() -> dict[str, Any]:
    """Return common Python-source descriptors for deep and shallow code arms."""

    path = (
        BASE
        / "reconstruction_v2/code_program_structure_retrospective_001/results.json"
    )
    result = _read_json(path)
    if result.get("schema") != "metric-seam.code-program-structure-retrospective.v1":
        raise ValueError(f"unexpected source-structure schema: {result.get('schema')}")
    scope = result.get("scope") or {}
    if scope.get("deep_programs") != 18 or scope.get("train_selected_shallow_programs") != 15:
        raise ValueError(f"unexpected source-structure scope: {scope}")
    return result


def math_a12_relation_generalization() -> dict[str, Any]:
    """Compare TRAIN and sealed held-out coverage for the exact symbolic relation."""

    train = _read_json(
        BASE
        / "reconstruction_v2/math_a12_symbolic_step_retrospective_prepare_001/"
        "train_symbolic_step_summary.json"
    )
    heldout = _read_json(
        BASE
        / "reconstruction_v2/math_a12_symbolic_step_heldout_001/finalization/"
        "finalization.json"
    )
    if train.get("schema") != "metric-seam.math-a12-symbolic-step-train-summary.v1":
        raise ValueError("unexpected Math a12 TRAIN summary schema")
    if heldout.get("schema") != "metric-seam.math-a12-symbolic-heldout-finalization.v1":
        raise ValueError("unexpected Math a12 held-out finalization schema")
    train_counts = train["coverage"]
    heldout_counts = heldout["candidate_execution_summary"]
    train_n = int(train["train_row_count"])
    heldout_n = int(heldout["heldout_count"])
    train_covered = int(train_counts["rows_with_executable_pair"])
    heldout_covered = int(heldout_counts["rows_with_executable_pair"])
    return {
        "train": {
            "rows": train_n,
            "covered_rows": train_covered,
            "coverage": train_covered / train_n,
            "coverage_wilson_95": _wilson_interval(train_covered, train_n),
            "identity_classifications": int(
                train_counts["verified_rational_identity_count"]
            ),
            "nonidentity_classifications": int(
                train_counts["exact_nonidentity_witness_count"]
            ),
        },
        "heldout": {
            "rows": heldout_n,
            "covered_rows": heldout_covered,
            "coverage": heldout_covered / heldout_n,
            "coverage_wilson_95": _wilson_interval(heldout_covered, heldout_n),
            "identity_classifications": int(
                heldout_counts["verified_rational_identity_count"]
            ),
            "nonidentity_classifications": int(
                heldout_counts["exact_nonidentity_witness_count"]
            ),
            "abstained_rows": int(heldout_counts["rows_abstained"]),
        },
        "heldout_minus_train_coverage": (
            heldout_covered / heldout_n - train_covered / train_n
        ),
        "coverage_fisher_exact_two_sided_p": _fisher_two_sided(
            heldout_covered,
            heldout_n - heldout_covered,
            train_covered,
            train_n - train_covered,
        ),
        "prompt_reference": {
            "available_both_passes": int(
                heldout["prompt_reference"]["available_both_passes"]
            ),
            "two_pass_spearman": float(
                heldout["prompt_reference"]["two_pass_spearman"]
            ),
        },
        "whole_criterion_reconstruction": heldout["whole_criterion_reconstruction"],
        "isomorphism": heldout["isomorphism"],
        "interpretation": (
            "Coverage generalizes descriptively, but coverage is not whole-criterion "
            "codability. Pair-level payloads were materialized only by a later "
            "post-reference, code-only projection replay."
        ),
    }


def math_a12_pair_projection_depth() -> dict[str, Any]:
    """Return inspectability counts and the corrected multi-view depth accounting.

    The pair projection is post-reference and therefore cannot create a new blind
    reconstruction result.  Its additive depth audit separates formal execution
    attempts from successful relation evidence so failed parsing is not silently
    credited as either shallow execution or positive verification.
    """

    base = BASE / "reconstruction_v2/math_a12_symbolic_step_heldout_001"
    projection = _read_json(
        base / "pair_certificate_projection_replay_001/projection_summary.json"
    )
    audit = _read_json(
        base / "relation_depth_multiview_audit_002/audit_summary.json"
    )
    if projection.get("schema") != "metric-seam.math-a12-pair-certificate-projection.v1":
        raise ValueError("unexpected Math a12 pair-projection schema")
    if audit.get("schema") != "metric-seam.math-a12-relation-depth-multiview-audit.v1":
        raise ValueError("unexpected Math a12 relation-depth audit schema")
    if projection.get("sealed_v1_row_classifications_exact") is not True:
        raise ValueError("Math a12 projection does not reproduce sealed row results")
    if projection.get("sealed_v1_aggregate_exact") is not True:
        raise ValueError("Math a12 projection does not reproduce sealed aggregates")
    if audit.get("source_projection_verified") is not True:
        raise ValueError("Math a12 depth audit did not verify its source projection")
    if audit.get("source_row_and_aggregate_results_unchanged") is not True:
        raise ValueError("Math a12 depth audit changed frozen classifications")

    return {
        "heldout_count": int(projection["heldout_count"]),
        "pair_certificate_count": int(projection["pair_certificate_count"]),
        "pair_status_counts": projection["pair_status_counts"],
        "row_category_counts": audit["row_category_counts"],
        "depth_views": audit["depth_views"],
        "formal_path_positive_evidence_rate": (
            audit["row_category_counts"]["formal_positive_relation_evidence"]
            / (
                audit["row_category_counts"]["formal_positive_relation_evidence"]
                + audit["row_category_counts"][
                    "formal_parse_noncoverage_abstention"
                ]
            )
        ),
        "temporal_status": projection["temporal_status"],
        "new_blind_result": projection["new_blind_result"],
        "new_reconstruction_result": projection["new_reconstruction_result"],
        "new_isomorphism_result": projection["new_isomorphism_result"],
        "interpretation": (
            "Post-reference inspectability replay. Depth 3 can mean attempted formal "
            "execution or positive relation evidence; those views are reported separately."
        ),
    }


def science_relation_witness_summary() -> dict[str, Any]:
    """Return strict full-article witness rates and code-representation replay.

    The continuous and exact-address arms execute the same manually constructed
    relation program.  Their overlap is a representation-robustness diagnostic,
    not prompt/code isomorphism and not external scientific truth.
    """

    continuous = _read_json(BASE / "science_claims_v2_relation_strict_v23/results.json")
    addressed = _read_json(
        BASE / "science_verifiability_v9_relation_strict_addressed/manifest.json"
    )
    prompt = _read_json(
        BASE / "science_articulability_v8_hardened_prepared/manifest.json"
    )
    replay = _read_json(
        BASE
        / "science_verifiability_v9_relation_strict_addressed_replay_v1.json"
    )
    if continuous.get("schema_version") != "science-claims-v2.3-relation-strict":
        raise ValueError("unexpected strict continuous-science schema")
    if addressed.get("schema_version") != "science-verifiability-addressed-v9-relation-strict":
        raise ValueError("unexpected strict addressed-science schema")
    continuous_summary = continuous["summary"]
    addressed_summary = addressed["summary"]
    comparison = addressed["representation_comparison"]
    if continuous_summary["certificate_decisions"] != {"supported": 100}:
        raise ValueError("unexpected continuous-science certificate count")
    if addressed_summary["certificates"] != 100:
        raise ValueError("unexpected addressed-science certificate count")
    if comparison["strong_whitespace_normalized_text"] != {
        "addressed": 100,
        "addressed_only": 0,
        "continuous": 100,
        "continuous_only": 0,
        "intersection": 100,
    }:
        raise ValueError("strict science strong witnesses are not representation-invariant")
    if (
        prompt.get("schema_version")
        != "science-articulability-addressed-bundle-v8"
        or prompt.get("status") != "prepared_not_run_no_api_calls"
        or prompt.get("files", {}).get("requests", {}).get("count") != 1957
        or prompt.get("files", {}).get("structural_abstentions", {}).get("count")
        != 443
        or prompt.get("execution_policy", {}).get("api_calls_made_by_prepare") != 0
        or prompt.get("execution_policy", {}).get("gpu_used") is not False
        or prompt.get("isomorphism_scope", {}).get("same_evidence_content") is not True
        or prompt.get("isomorphism_scope", {}).get("same_input_representation")
        is not False
    ):
        raise ValueError("unexpected current addressed-science prompt bundle")
    if (
        replay.get("schema_version")
        != "metric-seam.science-addressed-v9-replay-freeze.v1"
        or replay.get("status") != "byte_exact_cpu_replay_complete"
        or replay.get("replay", {}).get("byte_exact_all_outputs") is not True
        or replay.get("replay", {}).get("records") != 2400
        or replay.get("replay", {}).get("strong_relation_witnesses") != 100
        or replay.get("prompt_plane", {}).get("compiled_unscored_jobs") != 1957
        or replay.get("prompt_plane", {}).get("prompt_responses_in_current_v8_bundle")
        != 0
        or replay.get("temporal_disposition", {}).get(
            "fresh_split_required_for_confirmatory_prompt_code_claim"
        )
        is not True
    ):
        raise ValueError("unexpected strict addressed-science replay receipt")

    matched = continuous_summary["matched_edge_relations"]
    certificates = continuous_summary["certificate_relations"]
    relations = {}
    for relation in ("numeric", "comparative"):
        numerator = int(certificates[relation])
        denominator = int(matched[relation])
        relations[relation] = {
            "numerator": numerator,
            "denominator": denominator,
            "rate": numerator / denominator,
            "wilson_95": _wilson_interval(numerator, denominator),
        }
    all_matched = sum(int(value) for value in matched.values())
    strong = int(continuous_summary["certificate_decisions"]["supported"])
    supported_documents = int(continuous_summary["status_counts"]["supported"])
    papers = int(continuous_summary["papers"])
    return {
        "relation_witnesses": relations,
        "all_matched_relations": {
            "numerator": strong,
            "denominator": all_matched,
            "rate": strong / all_matched,
            "wilson_95": _wilson_interval(strong, all_matched),
        },
        "supported_documents": {
            "numerator": supported_documents,
            "denominator": papers,
            "rate": supported_documents / papers,
            "wilson_95": _wilson_interval(supported_documents, papers),
        },
        "representation_replay": {
            "strong_exact_text_intersection": int(
                comparison["strong_exact_text"]["intersection"]
            ),
            "strong_witness_intersection": int(
                comparison["strong_whitespace_normalized_text"]["intersection"]
            ),
            "strong_witness_continuous": int(
                comparison["strong_whitespace_normalized_text"]["continuous"]
            ),
            "strong_witness_addressed": int(
                comparison["strong_whitespace_normalized_text"]["addressed"]
            ),
            "supported_document_intersection": int(
                comparison["supported_paper_sets"]["intersection"]
            ),
            "supported_document_continuous": int(
                comparison["supported_paper_sets"]["continuous"]
            ),
            "supported_document_addressed": int(
                comparison["supported_paper_sets"]["addressed"]
            ),
            "paper_status_agreement": int(comparison["paper_status_agreement"]),
            "paper_status_total": int(comparison["paper_status_total"]),
            "weak_witness_intersection": int(
                comparison["weak_whitespace_normalized_text"]["intersection"]
            ),
            "weak_witness_continuous": int(
                comparison["weak_whitespace_normalized_text"]["continuous"]
            ),
            "weak_witness_addressed": int(
                comparison["weak_whitespace_normalized_text"]["addressed"]
            ),
            "archived_outputs_byte_exact_on_cpu_replay": True,
        },
        "prompt_articulability_status": "compiled_unscored_not_measured",
        "prompt_batch": {
            "corpus_records": int(prompt["strata"]["observed"]["corpus_records"]),
            "compiled_unscored_jobs": int(prompt["files"]["requests"]["count"]),
            "structural_abstentions_without_remote_call": int(
                prompt["files"]["structural_abstentions"]["count"]
            ),
            "prompt_responses": 0,
            "same_evidence_content": True,
            "same_input_representation_as_historical_continuous_code": False,
            "same_source_address_scaffold_bound_to_addressed_v9_code": True,
            "semantic_prompt_code_comparison_measured": False,
            "temporal_status": "instrument_development_exploratory_unscored",
            "fresh_split_required_for_confirmatory_prompt_code_claim": True,
        },
        "method_origin": continuous["method_origin"],
        "certificate_scope": continuous["certificate_scope"],
        "interpretation": (
            "The same strict executable relation is robust to continuous versus "
            "exact-address representations, and the exact-address output is byte-replay "
            "bound. The current prompt jobs remain unscored, so this is a source-address "
            "scaffold rather than prompt/code semantic isomorphism."
        ),
    }


def patent_ws3_family_retrospective() -> dict[str, Any]:
    """Return the full four-criterion multiplicity-aware WS3 retrospective."""

    result = _read_json(
        BASE
        / "reconstruction_v2/patent_ws3_family_retrospective_001/results.json"
    )
    if result.get("schema") != "metric-seam.patent-ws3-family-retrospective.v1":
        raise ValueError("unexpected patent WS3 family schema")
    if result.get("summary", {}).get("registered_criteria") != 4:
        raise ValueError("patent WS3 family must retain all four criteria")
    return result


def technical_evidence_ledger_summary() -> dict[str, Any]:
    """Read the typed, explicitly non-poolable technical evidence ledger."""

    result = _read_json(
        BASE / "reconstruction_v2/technical_evidence_ledger_v1/ledger.json"
    )
    if result.get("schema") != "metric-seam.technical-evidence-ledger.v1":
        raise ValueError("unexpected technical evidence ledger schema")
    summary = result.get("summary") or {}
    if summary.get("record_count") != len(result.get("records") or []):
        raise ValueError("technical ledger record count is inconsistent")
    if summary.get("explicitly_nonpoolable") is not True:
        raise ValueError("technical ledger must remain explicitly non-poolable")
    if summary.get("cross_stratum_pooled_estimates_emitted") != 0:
        raise ValueError("technical ledger emitted a cross-stratum pooled estimate")
    if summary.get("domain_codability_estimates_emitted") != 0:
        raise ValueError("technical ledger emitted a domain codability estimate")
    return {
        "summary": summary,
        "family_summaries": result["family_summaries"],
        "known_absences": result["known_absences"],
        "aggregation_guards": result["aggregation_guards"],
    }


def code_review_hierarchy_corrected_funnel() -> dict[str, Any]:
    """Read the independently corrected code-review availability funnel.

    This propagates the static construct-fidelity cross-audit through the
    already completed train and held-out availability gates.  It does not
    re-execute programs or read score vectors, prompt outputs, references, or
    outcomes.
    """

    audit = _read_json(
        HIERARCHY
        / "code_review_construct_fidelity_independent_cross_audit_v1.json"
    )
    funnel = _read_json(HIERARCHY / "code_review_corrected_funnel_v1.json")
    if audit.get("schema") != (
        "metric-seam.code-review-construct-fidelity-independent-cross-audit.v1"
    ):
        raise ValueError("unexpected code-review cross-audit schema")
    if audit.get("status") != "complete_independent_static_cross_audit_pre_execution":
        raise ValueError("code-review cross-audit is incomplete")
    if funnel.get("schema") != "metric-seam.code-review-corrected-funnel.v1":
        raise ValueError("unexpected corrected code-review funnel schema")
    if funnel.get("status") != (
        "corrected_static_gate_propagated_without_reexecution"
    ):
        raise ValueError("corrected code-review funnel is incomplete")
    if audit.get("task") != "code-review" or funnel.get("task") != "code-review":
        raise ValueError("corrected code-review task binding drifted")
    if set(audit.get("sealed_inputs", {}).values()) != {False}:
        raise ValueError("code-review cross-audit crossed a sealed input boundary")
    if set(funnel.get("sealed_inputs", {}).values()) != {False}:
        raise ValueError("corrected code-review funnel crossed a sealed input boundary")
    after = audit.get("after_summary", {})
    if (
        after.get("relation_local_static_fidelity_count") != 50
        or after.get("whole_construct_exact_count") != 0
        or after.get("audited_depth_counts_eligible") != {"1": 25, "2": 25}
        or audit.get("n_guarded_changes") != 7
    ):
        raise ValueError("code-review cross-audit counts drifted")
    corrected = funnel.get("corrected_readout", {})
    stage_counts = {
        stage: corrected["stages"][stage]["balanced_panel"]["n_positive"]
        for stage in (
            "retrieved_candidate",
            "relation_local_static_fidelity",
            "train_operational_relation_witness",
            "heldout_confirmatory_reconstruction_evaluable",
        )
    }
    if stage_counts != {
        "retrieved_candidate": 68,
        "relation_local_static_fidelity": 50,
        "train_operational_relation_witness": 27,
        "heldout_confirmatory_reconstruction_evaluable": 18,
    }:
        raise ValueError("corrected code-review stage counts drifted")
    expected_depths = {
        "relation_local_static_fidelity": {"1": 25, "2": 25},
        "train_operational_relation_witness": {"1": 19, "2": 8},
        "heldout_confirmatory_reconstruction_evaluable": {"1": 12, "2": 6},
    }
    depth_counts = {
        stage: {
            depth: values["n_positive"]
            for depth, values in corrected["by_depth"][stage].items()
        }
        for stage in expected_depths
    }
    if depth_counts != expected_depths:
        raise ValueError("corrected code-review depth counts drifted")
    if (
        len(funnel.get("removed_mappings", {}).get("static", [])) != 6
        or len(funnel.get("removed_mappings", {}).get("train_operational", [])) != 3
        or len(
            funnel.get("removed_mappings", {}).get("heldout_confirmatory", [])
        )
        != 3
        or len(funnel.get("depth_corrections", [])) != 1
    ):
        raise ValueError("corrected code-review removal ledger drifted")

    return {
        "task": "code-review",
        "panel_cells": 90,
        "stage_counts": stage_counts,
        "corrected_readout": corrected,
        "historical_readout": funnel["historical_readout"],
        "removed_mappings": funnel["removed_mappings"],
        "depth_corrections": funnel["depth_corrections"],
        "cross_audit": {
            "retrieved_rows_reviewed": audit["coverage"]["n_retrieved_reviewed"],
            "program_sources_reviewed": audit["coverage"][
                "n_unique_program_sources_reviewed"
            ],
            "guarded_changes": audit["n_guarded_changes"],
            "complete": audit["coverage"]["complete"],
        },
        "axes": {
            "prompt_articulability": "not_measured",
            "code_verifiability": (
                "relation_local_candidate_measurements_available; "
                "whole_construct_verifiability_not_established"
            ),
            "reconstruction_agreement": "not_estimated",
            "isomorphism": "not_estimated",
            "codability": "not_estimated",
        },
        "interpretation": funnel["interpretation"],
    }


def code_review_hierarchy_reconstruction_funnel() -> dict[str, Any]:
    """Summarize the frozen R1/R2/R3 lane before prompt scoring.

    Every count in this funnel precedes the prompt/reference comparison.  The
    stages therefore describe candidate availability, static relation match,
    and executable score support; they are not reconstruction, codability, or
    isomorphism estimates.
    """

    fidelity = _read_json(HIERARCHY / "code_review_construct_fidelity_v2.json")
    train = _read_json(HIERARCHY / "code_review_train_gate_v1.json")
    heldout = _read_json(HIERARCHY / "code_review_heldout_readiness_v1.json")
    prompt = _read_json(
        HIERARCHY / "code_review_reconstruction_prompt_manifest_v3.json"
    )
    incident = _read_json(
        HIERARCHY / "code_review_binary_diff_parser_incident_v1.json"
    )
    prevalence = _read_json(HIERARCHY / "code_review_witness_prevalence_v3.json")
    corrected = code_review_hierarchy_corrected_funnel()

    expected_schemas = {
        "fidelity": "metric-seam.code-review-construct-fidelity-merged.v1",
        "train": "metric-seam.hierarchy-code-train-gate.v1",
        "heldout": "metric-seam.hierarchy-code-heldout-readiness.v1",
        "prompt": "metric-seam.hierarchy-reconstruction-prompt-batch.v3",
        "incident": "metric-seam.code-review-parser-incident.v1",
        "prevalence": "metric-seam.hierarchy-witness-prevalence.v2",
    }
    artifacts = {
        "fidelity": fidelity,
        "train": train,
        "heldout": heldout,
        "prompt": prompt,
        "incident": incident,
        "prevalence": prevalence,
    }
    for name, schema in expected_schemas.items():
        if artifacts[name].get("schema") != schema:
            raise ValueError(f"unexpected code-review hierarchy {name} schema")

    fsum = fidelity["summary"]
    tsum = train["summary"]
    hsum = heldout["summary"]
    panel_n = int(fsum["n_metrics"])
    exact_n = int(fsum["whole_construct_exact_count"])
    historical_counts = (
        panel_n,
        int(fsum["relation_local_static_fidelity_count"]),
        exact_n,
        int(tsum["n_selected_relation_mappings"]),
        int(hsum["n_confirmatory_relation_mappings"]),
    )
    if historical_counts != (90, 56, 0, 30, 21):
        raise ValueError("historical code-review funnel counts drifted")
    stage_counts = corrected["stage_counts"]
    static_n = int(stage_counts["relation_local_static_fidelity"])
    train_n = int(stage_counts["train_operational_relation_witness"])
    heldout_n = int(
        stage_counts["heldout_confirmatory_reconstruction_evaluable"]
    )
    if (static_n, train_n, heldout_n) != (50, 27, 18):
        raise ValueError("corrected code-review funnel counts drifted")
    if (
        prompt.get("status")
        != "compiled_unscored_static_cross_audit_filtered"
        or prompt.get("n_cells") != heldout_n
        or prompt.get("n_jobs") != 13500
    ):
        raise ValueError("prompt manifest is not the corrected unscored 18-cell set")
    if prompt.get("candidate_scores_read_or_embedded") is not False:
        raise ValueError("prompt manifest crossed the frozen code/reference boundary")
    if heldout.get("prompt_outputs_used") is not False:
        raise ValueError("heldout readiness was not frozen before prompt scoring")
    if incident.get("status") != "closed_by_additive_rerun":
        raise ValueError("binary-diff parser incident is not closed")
    if train.get("training_execution_source") != incident.get("canonical_replacement"):
        raise ValueError("train gate does not use the canonical parser replay")
    if prevalence.get("status") != "pre_reconstruction_code_witness_prevalence":
        raise ValueError("code-review witness prevalence is not canonical")
    if prevalence.get("panel_content_sha256") != fidelity.get("panel_content_sha256"):
        raise ValueError("code-review prevalence/fidelity panel binding drifted")

    corrected_readout = corrected["corrected_readout"]
    by_level = []
    for level in ("R1", "R2", "R3"):
        level_readout = corrected_readout["by_level"][level]
        static_level = int(
            level_readout["relation_local_static_fidelity"]["balanced_panel"][
                "n_positive"
            ]
        )
        train_level = int(
            level_readout["train_operational_relation_witness"]["balanced_panel"][
                "n_positive"
            ]
        )
        heldout_level = int(
            level_readout["heldout_confirmatory_reconstruction_evaluable"][
                "balanced_panel"
            ]["n_positive"]
        )
        by_level.append(
            {
                "level": level,
                "panel_n": int(fsum["by_level"][level]["n_metrics"]),
                "static_relation_local_n": static_level,
                "train_operational_n": train_level,
                "heldout_code_score_ready_n": heldout_level,
                "heldout_pct_of_panel": _pct(heldout_level, 30),
                "heldout_pct_of_static": _pct(heldout_level, static_level),
            }
        )

    by_depth = []
    for depth in ("1", "2"):
        static_depth = int(
            corrected_readout["by_depth"]["relation_local_static_fidelity"][depth][
                "n_positive"
            ]
        )
        train_depth = int(
            corrected_readout["by_depth"]["train_operational_relation_witness"][
                depth
            ]["n_positive"]
        )
        heldout_depth = int(
            corrected_readout["by_depth"][
                "heldout_confirmatory_reconstruction_evaluable"
            ][depth]["n_positive"]
        )
        by_depth.append(
            {
                "depth": int(depth),
                "depth_meaning": fidelity["depth_vocabulary"][depth],
                "static_relation_local_n": static_depth,
                "train_operational_n": train_depth,
                "heldout_code_score_ready_n": heldout_depth,
                "heldout_pct_of_static": _pct(heldout_depth, static_depth),
            }
        )

    if sum(row["heldout_code_score_ready_n"] for row in by_level) != heldout_n:
        raise ValueError("heldout level counts do not sum to the funnel total")
    if sum(row["heldout_code_score_ready_n"] for row in by_depth) != heldout_n:
        raise ValueError("heldout depth counts do not sum to the funnel total")

    return {
        "task": "code-review",
        "hierarchy_frame": fidelity["hierarchy_frame"],
        "stages": [
            {
                "stage": "static relation-local fidelity",
                "n": static_n,
                "denominator": panel_n,
                "pct": _pct(static_n, panel_n),
            },
            {
                "stage": "TRAIN-operational mapping",
                "n": train_n,
                "denominator": panel_n,
                "pct": _pct(train_n, panel_n),
            },
            {
                "stage": "held-out code-score ready",
                "n": heldout_n,
                "denominator": panel_n,
                "pct": _pct(heldout_n, panel_n),
            },
            {
                "stage": "prompt jobs compiled (unscored)",
                "n": int(prompt["n_cells"]),
                "denominator": panel_n,
                "pct": _pct(int(prompt["n_cells"]), panel_n),
            },
        ],
        "whole_construct_exact": {
            "n": exact_n,
            "denominator": panel_n,
            "pct": _pct(exact_n, panel_n),
        },
        "by_level": by_level,
        "by_depth": by_depth,
        "heldout_min_code_scores": int(
            heldout["thresholds"]["confirmatory_min_paired_scores"]
        ),
        "prompt_manifest": {
            "status": prompt["status"],
            "cells": int(prompt["n_cells"]),
            "items_per_cell": int(prompt["n_items_per_cell"]),
            "channels": list(prompt["channels"]),
            "passes": list(prompt["passes"]),
            "jobs": int(prompt["n_jobs"]),
            "unique_program_vectors": int(prompt["n_unique_program_vectors"]),
            "scope_statements": prompt["scope_statements"],
            "external_ground_truth_used": bool(prompt["external_ground_truth_used"]),
            "candidate_scores_read_or_embedded": bool(
                prompt["candidate_scores_read_or_embedded"]
            ),
            "control_reassignments_after_scope_filter": len(
                prompt["analysis_preregistration"]["wrong_relation_control"][
                    "reassignments_from_v2"
                ]
            ),
            "old_batch_disposition": prompt["static_cross_audit_filter"][
                "old_batch_disposition"
            ],
        },
        "axes": {
            "prompt_articulability": "not_measured_jobs_compiled_unscored",
            "code_verifiability": (
                "relation_local_candidate_measurements_available; "
                "whole_construct_verifiability_not_established"
            ),
            "reconstruction_agreement": "not_estimated",
            "isomorphism": "not_estimated",
            "codability": "not_estimated",
        },
        "parser_incident": {
            "status": incident["status"],
            "invalidated_artifact": incident["invalidated_artifact"],
            "canonical_replacement": incident["canonical_replacement"],
            "invalidated_v1_excluded": True,
        },
        "prevalence": {
            "estimand": (
                "conditional stratum expansion over eligible action-node records under "
                "hash-as-random within-stratum exchangeability"
            ),
            "sampling_frame": prevalence["sampling_frame"],
            "estimated_population_nodes": 1128,
            "pooled": {
                outcome: corrected_readout["stages"][outcome][
                    "conditional_eligible_inventory_expansion"
                ]
                for outcome in (
                    "retrieved_candidate",
                    "relation_local_static_fidelity",
                    "train_operational_relation_witness",
                    "heldout_confirmatory_reconstruction_evaluable",
                )
            },
            "by_level": {
                level: {
                    outcome: corrected_readout["by_level"][level][outcome][
                        "conditional_eligible_inventory_expansion"
                    ]["rate"]
                    for outcome in (
                        "relation_local_static_fidelity",
                        "train_operational_relation_witness",
                        "heldout_confirmatory_reconstruction_evaluable",
                    )
                }
                for level in ("R1", "R2", "R3")
            },
            "terminal_frontier": prevalence["sensitivities"]
            ["tightest_first_terminal_frontier"],
            "dependence_diagnostics": prevalence["sensitivities"]
            ["dependence_component_diagnostics"],
            "corrected_outcome_perturbation_ranges_recomputed": False,
            "historical_pre_cross_audit_point_readout": corrected[
                "historical_readout"
            ],
            "outstanding_sensitivities": [
                *prevalence["outstanding_sensitivities"],
                (
                    "Recompute dependency/provenance perturbation ranges and merged-only "
                    "sensitivity after the independent construct-fidelity correction."
                ),
            ],
            "supersedes": {
                "historical_funnel": "56 static -> 30 train -> 21 heldout",
                "corrected_funnel": "50 static -> 27 train -> 18 heldout",
                "historical_prompt_manifest": (
                    "code_review_reconstruction_prompt_manifest_v2.json"
                ),
                "corrected_prompt_manifest": (
                    "code_review_reconstruction_prompt_manifest_v3.json"
                ),
            },
            "claim_limits": prevalence["claim_limits"],
        },
        "construct_fidelity_cross_audit": corrected["cross_audit"],
        "corrected_gate_propagation": {
            "removed_static": len(corrected["removed_mappings"]["static"]),
            "removed_train_operational": len(
                corrected["removed_mappings"]["train_operational"]
            ),
            "removed_heldout_ready": len(
                corrected["removed_mappings"]["heldout_confirmatory"]
            ),
            "depth_corrections": len(corrected["depth_corrections"]),
            "programs_reexecuted": False,
        },
        "interpretation": (
            "This is a pre-reconstruction engineering funnel. Its numerators cannot "
            "be reported as metric codability, prompt articulability, reconstruction, "
            "or isomorphism rates."
        ),
    }


def code_review_additive_unused_program_funnel() -> dict[str, Any]:
    """Summarize the repaired nine-mapping additive code-review extension."""

    cross = _read_json(
        HIERARCHY / "code_review_unused_program_construct_cross_audit_v1.json"
    )
    fidelity = _read_json(
        HIERARCHY / "code_review_construct_fidelity_additive_unused_programs_v1.json"
    )
    train = _read_json(
        HIERARCHY / "code_review_additive59_compiler_train_execution_v1.json"
    )
    gate = _read_json(HIERARCHY / "code_review_additive59_train_gate_v1.json")
    heldout = _read_json(
        HIERARCHY / "code_review_additive59_heldout_pre_reference_execution_v1.json"
    )
    readiness = _read_json(
        HIERARCHY / "code_review_additive59_heldout_readiness_v1.json"
    )
    if (
        cross.get("schema")
        != "metric-seam.code-review-unused-program-construct-cross-audit.v1"
        or cross.get("status")
        != "independent_static_construct_audit_complete_pre_execution"
        or fidelity.get("schema")
        != "metric-seam.code-review-construct-fidelity-merged.v1"
        or fidelity.get("status")
        != "additive_static_construct_fidelity_complete_pre_execution"
    ):
        raise ValueError("code-review additive static audit is incomplete")
    for phase, artifact in (
        ("compiler_train", train),
        ("heldout_pre_reference", heldout),
    ):
        if (
            artifact.get("schema") != "metric-seam.hierarchy-code-execution.v1"
            or artifact.get("status") != "worker_replay_complete"
            or artifact.get("phase") != phase
            or artifact.get("reference_fields_passed_to_worker") is not False
            or artifact.get("outcome_fields_passed_to_worker") is not False
            or artifact.get("credentials_inherited_by_worker") is not False
            or artifact.get("accelerators_visible_to_worker") is not False
        ):
            raise ValueError(f"code-review additive {phase} execution is invalid")
    if (
        gate.get("schema") != "metric-seam.hierarchy-code-train-gate.v1"
        or gate.get("status") != "frozen_before_heldout_program_execution"
        or gate.get("selection_basis") != "compiler_train_outputs_only"
        or gate.get("reference_values_used") is not False
        or gate.get("outcome_labels_used") is not False
        or gate.get("heldout_items_or_outputs_used") is not False
        or readiness.get("schema")
        != "metric-seam.hierarchy-code-heldout-readiness.v1"
        or readiness.get("status") != "frozen_before_prompt_reference_scoring"
        or readiness.get("reference_values_used") is not False
        or readiness.get("outcome_labels_used") is not False
        or readiness.get("prompt_outputs_used") is not False
    ):
        raise ValueError("code-review additive train/heldout gate is invalid")

    static_summary = fidelity["summary"]
    cross_summary = cross["summary"]
    train_summary = train["summary"]
    gate_summary = gate["summary"]
    heldout_summary = heldout["summary"]
    ready_summary = readiness["summary"]
    observed = {
        "canonical_corrected_static_unchanged": cross_summary[
            "canonical_corrected_static_unchanged"
        ],
        "additive_static_union": static_summary[
            "relation_local_static_fidelity_count"
        ],
        "additive_static_by_level": {
            level: static_summary["by_level"][level][
                "relation_local_static_fidelity_count"
            ]
            for level in ("R1", "R2", "R3")
        },
        "new_static_mappings": cross_summary["n_accepted_partial_relation_local"],
        "new_static_by_depth": cross_summary["accepted_by_audited_depth"],
        "train_unique_programs": train_summary["n_unique_programs"],
        "train_nondegenerate_mappings": train_summary[
            "n_relation_mappings_with_nondegenerate_measurement"
        ],
        "train_selected_mappings": gate_summary["n_selected_relation_mappings"],
        "train_selected_by_level": gate_summary[
            "selected_relation_mappings_by_level"
        ],
        "train_selected_by_depth": gate_summary[
            "selected_relation_mappings_by_depth"
        ],
        "heldout_selected_mappings": heldout_summary["n_planned_relation_mappings"],
        "heldout_nondegenerate_mappings": heldout_summary[
            "n_relation_mappings_with_nondegenerate_measurement"
        ],
        "heldout_confirmatory_mappings": ready_summary[
            "n_confirmatory_relation_mappings"
        ],
        "heldout_confirmatory_by_level": ready_summary[
            "confirmatory_relation_mappings_by_level"
        ],
        "heldout_confirmatory_by_depth": ready_summary[
            "confirmatory_relation_mappings_by_depth"
        ],
        "heldout_readiness_counts": ready_summary["relation_readiness_counts"],
    }
    if observed != {
        "canonical_corrected_static_unchanged": 50,
        "additive_static_union": 59,
        "additive_static_by_level": {"R1": 17, "R2": 19, "R3": 23},
        "new_static_mappings": 9,
        "new_static_by_depth": {"2": 4, "4": 5},
        "train_unique_programs": 32,
        "train_nondegenerate_mappings": 42,
        "train_selected_mappings": 35,
        "train_selected_by_level": {"R1": 11, "R2": 10, "R3": 14},
        "train_selected_by_depth": {"1": 18, "2": 12, "4": 5},
        "heldout_selected_mappings": 35,
        "heldout_nondegenerate_mappings": 35,
        "heldout_confirmatory_mappings": 19,
        "heldout_confirmatory_by_level": {"R1": 7, "R2": 5, "R3": 7},
        "heldout_confirmatory_by_depth": {"1": 12, "2": 7},
        "heldout_readiness_counts": {
            "confirmatory_reconstruction_evaluable": 19,
            "exploratory_sparse": 12,
            "insufficient_paired_support": 4,
        },
    }:
        raise ValueError("code-review additive funnel counts drifted")
    new_aspects = {"a25", "a35", "a72", "a181", "a309", "a400"}
    gate_rows = {
        row["aspect_id"]: row
        for row in gate["programs"]
        if row["aspect_id"] in new_aspects
    }
    ready_rows = {
        row["aspect_id"]: row
        for row in readiness["programs"]
        if row["aspect_id"] in new_aspects
    }
    return {
        **observed,
        "new_program_train_gate": {
            aspect: {
                "decision": gate_rows[aspect]["decision"],
                "n_scored": gate_rows[aspect]["n_scored"],
                "n_relation_mappings": gate_rows[aspect]["n_relation_mappings"],
            }
            for aspect in sorted(gate_rows)
        },
        "new_program_heldout_readiness": {
            aspect: {
                "readiness": ready_rows[aspect]["readiness"],
                "n_scored": ready_rows[aspect]["n_scored"],
                "n_relation_mappings": ready_rows[aspect]["n_relation_mappings"],
            }
            for aspect in sorted(ready_rows)
        },
        "axes": {
            "prompt_articulability": "not_measured",
            "code_verifiability": "relation_local_static_train_and_heldout_measured",
            "reconstruction": "not_measured",
            "isomorphism": "not_measured",
            "whole_construct_verifiability": 0,
            "external_supervised_anchor_used": False,
        },
        "interpretation": (
            "This additive lane increases static relation-local coverage from 50 to 59 "
            "without changing the canonical artifact. Thirty-five mappings survive the "
            "train-only gate and execute nondegenerately heldout; only nineteen meet the "
            "pre-registered thirty-pair confirmatory-readiness threshold."
        ),
    }


def patent_hierarchy_static_funnel() -> dict[str, Any]:
    """Summarize the narrow static patent hierarchy lane.

    The bank contains four retrospective manual hybrids.  Their prior-art
    operation is examiner/oracle conditioned and uses precomputed reading-model
    disclosure relations.  Consequently, depth-3 relation matches are an
    evidence-channel descriptor, not pure-code verifiability.
    """

    seeds = _read_json(HIERARCHY / "patents_seed_map_v1.json")
    fidelity = _read_json(HIERARCHY / "patents_construct_fidelity_v1.json")
    prevalence = _read_json(HIERARCHY / "patents_static_witness_prevalence_v1.json")
    expected = {
        "seeds": "metric-seam.hierarchy-patent-seed-map.v1",
        "fidelity": "metric-seam.hierarchy-patent-construct-fidelity.v1",
        "prevalence": "metric-seam.patent-static-witness-prevalence.v1",
    }
    for name, artifact in (
        ("seeds", seeds),
        ("fidelity", fidelity),
        ("prevalence", prevalence),
    ):
        if artifact.get("schema") != expected[name]:
            raise ValueError(f"unexpected patent hierarchy {name} schema")
    if fidelity.get("status") != "static-relation-local-adjudication-complete":
        raise ValueError("patent hierarchy fidelity audit is incomplete")
    if prevalence.get("status") != "static_descriptive_rates_complete":
        raise ValueError("patent hierarchy prevalence is incomplete")
    if (
        seeds.get("panel_content_sha256")
        != fidelity.get("source_panel_content_sha256")
        or seeds.get("panel_content_sha256") != prevalence.get("panel_content_sha256")
    ):
        raise ValueError("patent hierarchy artifacts use different panels")
    summary = fidelity["summary"]
    if (
        summary.get("n_retrieved"),
        summary.get("n_partial_relation_local"),
        summary.get("n_exact_whole_construct"),
        summary.get("n_pure_code_witnesses"),
    ) != (6, 6, 0, 0):
        raise ValueError("patent hierarchy static counts drifted")

    pooled = prevalence["pooled_eligible_action_nodes"]
    expansion = pooled["eligible_inventory_stratum_expansion"]
    return {
        "task": "patents",
        "historical_program_families": int(seeds["n_historical_program_families"]),
        "panel_cells": int(fidelity["n_cells"]),
        "retrieved_candidates": int(summary["n_retrieved"]),
        "relation_local_static_fidelity": int(summary["n_partial_relation_local"]),
        "whole_construct_exact": int(summary["n_exact_whole_construct"]),
        "pure_code_witnesses": int(summary["n_pure_code_witnesses"]),
        "balanced_panel": {
            outcome: pooled["balanced_panel"][outcome]
            for outcome in (
                "retrieved_candidate",
                "relation_local_static_fidelity",
                "depth3_evidence_relation",
                "pure_code_witness",
                "whole_construct_exact",
            )
        },
        "conditional_eligible_inventory": {
            "population_nodes": int(
                prevalence["sampling_frame"]["n_eligible_action_node_records"]
            ),
            **{
                outcome: expansion[outcome]
                for outcome in (
                    "retrieved_candidate",
                    "relation_local_static_fidelity",
                    "depth3_evidence_relation",
                    "pure_code_witness",
                    "whole_construct_exact",
                )
            },
        },
        "by_level": summary["by_level"],
        "maximum_matching_relation_depth_counts": summary[
            "maximum_matching_relation_depth_counts"
        ],
        "channel_provenance": prevalence["channel_provenance"],
        "axes": {
            "prompt_articulability": "not_measured",
            "code_verifiability": "not_measured_static_relation_fidelity_only",
            "reconstruction_agreement": "not_estimated",
            "isomorphism": "not_estimated",
            "codability": "not_estimated",
        },
        "claim_limits": prevalence["claim_limits"],
        "interpretation": (
            "This is coverage of relation-local static witnesses in a four-program historical "
            "bank. It is not a patent-metric codability estimate. Depth 3 denotes the "
            "oracle-conditioned external evidence operation, not pure code."
        ),
    }


def patent_claim_structure_hierarchy_static_funnel() -> dict[str, Any]:
    """Summarize the additive pure-code patent claim-structure instrument.

    Unlike :func:`patent_hierarchy_static_funnel`, this lane uses no prior-art,
    examiner, reading-model, prompt, reference, or outcome channel.  It parses
    the exact shared ``ctext`` into a claim graph and emits only scoped relation
    witnesses.  Static fidelity and train variation are reported separately:
    a formatter-constant relation can be a faithful executable articulation of
    a narrow relation without being useful for reconstruction on this sample.
    """

    fidelity = _read_json(
        HIERARCHY / "patents_claim_structure_construct_fidelity_v1.json"
    )
    train = _read_json(HIERARCHY / "patents_claim_structure_compiler_train_v14.json")
    if (
        fidelity.get("schema")
        != "metric-seam.hierarchy-patent-claim-structure-fidelity.v1"
        or fidelity.get("status") != "conservative-static-adjudication-complete"
        or fidelity.get("task") != "patents"
        or fidelity.get("n_cells") != 90
    ):
        raise ValueError("unexpected patent claim-structure fidelity artifact")
    if (
        train.get("schema")
        != "metric-seam.hierarchy-patent-claim-structure-execution.v3"
        or train.get("program_schema") != "metric-seam.patent-claim-structure.v13"
        or train.get("phase") != "compiler_train"
        or fidelity.get("train_receipt_schema") != train.get("schema")
        or fidelity.get("program_schema") != train.get("program_schema")
    ):
        raise ValueError("unexpected patent claim-structure TRAIN receipt")
    forbidden_design_fields = (
        "outcome_or_reference_values_loaded",
        "prompt_outputs_loaded",
        "prior_art_or_examiner_evidence_loaded",
        "external_supervision_used",
        "whole_patent_score_emitted",
    )
    if any(train.get("design", {}).get(key) is not False for key in forbidden_design_fields):
        raise ValueError("patent claim-structure TRAIN receipt loaded a forbidden channel")

    rows = fidelity.get("rows")
    if not isinstance(rows, list) or len(rows) != 90:
        raise ValueError("patent claim-structure audit does not cover 90 cells")
    accepted = [row for row in rows if row.get("verdict") == "partial_relation_local"]
    near_misses = [
        row for row in rows if row.get("verdict") == "sensitivity_near_miss_not_accepted"
    ]
    rejected = [row for row in rows if row.get("verdict") == "no_faithful_relation"]
    if (len(accepted), len(near_misses), len(rejected)) != (8, 4, 78):
        raise ValueError("patent claim-structure verdict funnel drifted")

    operational = []
    static_only = []
    for row in accepted:
        relations = row.get("matched_relations")
        if not isinstance(relations, list) or not relations:
            raise ValueError("accepted patent claim-structure row has no relation")
        classifications = {
            relation.get("train_operational_applicability", {}).get("classification")
            for relation in relations
            if isinstance(relation, dict)
        }
        if classifications == {"measured_but_constant_non_operational"}:
            static_only.append(row)
        elif "measured_but_constant_non_operational" in classifications:
            raise ValueError("patent claim-structure cell mixes static and operational relations")
        else:
            operational.append(row)
    if len(static_only) != 3 or len(operational) != 5:
        raise ValueError("patent claim-structure TRAIN applicability drifted")

    summary = fidelity.get("summary", {})
    additive = summary.get("additive_union_with_historical", {})
    weighted = summary.get("posthoc_design_weighted_conditional_sensitivity", {})
    train_summary = train.get("summary", {})
    if (
        summary.get("n_partial_relation_local_cells") != 8
        or summary.get("n_exact_whole_construct_cells") != 0
        or summary.get("maximum_matching_relation_depth_counts") != {"1": 7, "2": 1}
        or additive.get("n_additive_union_cells") != 14
        or additive.get("n_overlapping_cells") != 0
        or train_summary.get("n_items") != 150
        or train_summary.get("failure_types") != {}
        or train_summary.get("items_at_declared_character_cap") != 119
    ):
        raise ValueError("patent claim-structure summary drifted")

    return {
        "task": "patents",
        "instrument": "pure_code_claim_structure_v13",
        "panel_cells": 90,
        "relation_local_static_fidelity": 8,
        "train_operational_candidates_pre_frozen_gate": 5,
        "static_only_formatter_constant_cells": 3,
        "sensitivity_near_misses_not_credited": 4,
        "no_faithful_relation": 78,
        "whole_construct_exact": 0,
        "balanced_panel": {
            "relation_local_static_fidelity": {"n": 8, "denominator": 90, "rate": 8 / 90},
            "train_operational_candidates_pre_frozen_gate": {
                "n": 5,
                "denominator": 90,
                "rate": 5 / 90,
            },
            "static_only_formatter_constant": {
                "n": 3,
                "denominator": 90,
                "rate": 3 / 90,
            },
            "exact_whole_construct": {"n": 0, "denominator": 90, "rate": 0.0},
        },
        "by_level": summary["by_level"],
        "maximum_matching_relation_depth_counts": summary[
            "maximum_matching_relation_depth_counts"
        ],
        "relation_cell_counts": summary["relation_cell_counts"],
        "train": {
            "items": int(train_summary["n_items"]),
            "status_counts": train_summary["status_counts"],
            "items_at_declared_character_cap": int(
                train_summary["items_at_declared_character_cap"]
            ),
            "relation_measurement": train_summary["relation_measurement"],
            "certificate_counts": train_summary["certificate_counts"],
        },
        "historical_comparison": {
            "manual_oracle_conditioned_cells": int(additive["n_historical_partial_cells"]),
            "overlap": int(additive["n_overlapping_cells"]),
            "descriptive_union_cells": int(additive["n_additive_union_cells"]),
            "descriptive_union_depth_counts": additive[
                "maximum_matching_relation_depth_counts"
            ],
            "provenance_warning": additive["provenance_warning"],
        },
        "posthoc_design_weighted_conditional_sensitivity": weighted,
        "cap_rule": fidelity["cap_rule"],
        "axes": {
            "prompt_articulability": "not_measured",
            "code_verifiability": "relation_local_execution_pre_heldout",
            "reconstruction_agreement": "not_estimated",
            "isomorphism": "shared_ctext_only_other_axes_not_estimated",
            "codability": "not_estimated",
        },
        "interpretation": (
            "Eight of 90 cells have a conservative partial pure-code relation match. "
            "Five vary on TRAIN and are candidates for a separately frozen gate; three "
            "section-presence witnesses are faithful but formatter-constant. These are "
            "static/execution precursors, not prompt articulability, reconstruction, "
            "isomorphism, whole-construct verifiability, or codability estimates."
        ),
    }


def patent_claim_structure_hierarchy_operational_funnel() -> dict[str, Any]:
    """Summarize the frozen-gate patent heldout code execution.

    This is a code-side relation-output readout only.  It intentionally stops
    before loading or estimating any prompt, reference, reconstruction, or
    outcome quantity.
    """

    gate = _read_json(HIERARCHY / "patents_claim_structure_train_gate_v1.json")
    heldout = _read_json(
        HIERARCHY / "patents_claim_structure_heldout_pre_reference_v1.json"
    )
    operational = _read_json(
        HIERARCHY / "patents_claim_structure_operational_summary_v1.json"
    )
    prompt_train = _read_json(HIERARCHY / "patents_prompt_train_manifest_v3.json")
    prompt_heldout = _read_json(
        HIERARCHY / "patents_prompt_heldout_fixed_manifest_v3.json"
    )
    prompt_v1_audit = _read_json(HIERARCHY / "patents_prompt_v1_cross_audit.json")
    prompt_supersession = _read_json(
        HIERARCHY / "patents_prompt_v3_supersession_receipt.json"
    )
    prompt_validator = _read_json(
        HIERARCHY / "patents_prompt_v3_validator_freeze.json"
    )
    if (
        gate.get("schema")
        != "metric-seam.hierarchy-patent-claim-structure-train-gate.v1"
        or gate.get("status") != "frozen-before-heldout-pre-reference-execution"
        or gate.get("task") != "patents"
    ):
        raise ValueError("unexpected patent claim-structure TRAIN gate")
    boundaries = gate.get("channel_boundaries")
    if (
        not isinstance(boundaries, dict)
        or boundaries.get("input_fields") != ["item_key", "ctext"]
        or any(value is not False for key, value in boundaries.items() if key != "input_fields")
    ):
        raise ValueError("patent claim-structure gate crossed a forbidden channel")
    gate_summary = gate.get("summary", {})
    selected = gate.get("selected_operational_cells")
    static_only = gate.get("static_only_cells")
    if (
        not isinstance(selected, list)
        or not isinstance(static_only, list)
        or len(selected) != 5
        or len(static_only) != 3
        or gate_summary.get("n_selected_operational_cells") != 5
        or gate_summary.get("n_static_only_constant_cells") != 3
        or gate_summary.get("n_whole_construct_cells") != 0
        or gate_summary.get("prompt_scored_cells") != 0
        or gate_summary.get("reconstruction_evaluable_cells") != 0
        or gate_summary.get("isomorphism_evaluable_cells") != 0
    ):
        raise ValueError("patent claim-structure gate summary drifted")
    operational_summary = operational.get("stage_summary", {})
    operational_cells = operational.get("heldout_operational_cells")
    if (
        operational.get("schema")
        != "metric-seam.hierarchy-patent-claim-structure-operational-summary.v1"
        or operational.get("status")
        != "heldout-relation-measurement-complete-pre-reference"
        or operational.get("task") != "patents"
        or not isinstance(operational_cells, list)
        or {row.get("cell_id") for row in operational_cells} != {
            row.get("cell_id") for row in selected
        }
        or operational_summary.get("n_static_relation_local_cells") != 8
        or operational_summary.get("n_train_operational_cells") != 5
        or operational_summary.get("n_heldout_relation_measurable_cells") != 5
        or operational_summary.get("n_prompt_articulability_measured_cells") != 0
        or operational_summary.get("n_reference_reconstruction_measured_cells") != 0
        or operational_summary.get("n_prompt_code_isomorphism_evaluable_cells") != 0
        or operational_summary.get("n_whole_criterion_codability_established_cells")
        != 0
    ):
        raise ValueError("patent claim-structure operational summary drifted")
    operational_boundaries = operational.get("channel_boundaries", {})
    if any(
        operational_boundaries.get(key) is not False
        for key in (
            "reference_or_prompt_values_loaded",
            "outcomes_loaded",
            "prior_art_or_examiner_evidence_loaded",
            "external_supervision_loaded",
            "models_or_apis_called",
            "whole_patent_score_emitted",
        )
    ):
        raise ValueError("patent operational summary crossed a forbidden channel")
    for manifest, phase, expected_jobs, expected_specs in (
        (prompt_train, "compiler_train", 7_500, 25),
        (prompt_heldout, "heldout_pre_reference", 19_500, 65),
    ):
        if (
            manifest.get("schema") != "metric-seam.patent-prompt-articulability-batch.v3"
            or manifest.get("status") != "compiled_unscored"
            or manifest.get("task") != "patents"
            or manifest.get("phase") != phase
            or set(manifest.get("forbidden_inputs", {}).values()) != {False}
            or manifest.get("summary", {}).get("n_cells") != 5
            or manifest.get("summary", {}).get("n_prompt_specs") != expected_specs
            or manifest.get("summary", {}).get("n_jobs") != expected_jobs
            or manifest.get("summary", {}).get("n_prompt_responses") != 0
            or manifest.get("summary", {}).get("n_reconstruction_estimates") != 0
            or manifest.get("summary", {}).get("n_isomorphism_adjudications") != 0
            or manifest.get("jobs_artifact", {}).get("n_jobs") != expected_jobs
            or manifest.get("jobs_artifact", {}).get("model_api_or_gpu_calls_performed")
            is not False
        ):
            raise ValueError(f"patent {phase} prompt manifest drifted")
        temporal = manifest.get("temporal_provenance", {})
        projection = manifest.get("model_input_projection_contract", {})
        if (
            temporal.get("absence_of_human_influence_certified") is not False
            or temporal.get("mechanical_consumption_of_heldout_code_or_summary")
            is not False
            or projection.get("post_code_schema_is_cap_specialized_per_job") is not True
            or projection.get("post_code_semantic_validator_required")
            != "validate_post_code_response.v3"
        ):
            raise ValueError(f"patent {phase} prompt contract is not v2 fail-closed")
    if (
        prompt_heldout.get("batch_role")
        != "fixed_after_train_gate_exploratory_pre_reference"
        or prompt_heldout.get("temporal_provenance", {}).get(
            "fresh_confirmatory_split_required_for_temporal_preregistration"
        )
        is not True
        or prompt_v1_audit.get("schema")
        != "metric-seam.patent-prompt-v1-cross-audit.v1"
        or prompt_v1_audit.get("status") != "complete-v1-superseded-not-executable"
        or prompt_v1_audit.get("disposition", {}).get("v1_prompt_manifests_and_jobs")
        != "superseded exploratory receipts; do not execute model calls from these packs"
        or prompt_supersession.get("schema")
        != "metric-seam.patent-prompt-supersession.v2"
        or prompt_supersession.get("status") != "v3-repaired-compiled-unscored"
        or prompt_supersession.get("v2_disposition")
        != (
            "superseded unexecuted receipt; duplicate-certificate and cap-status "
            "semantic holes repaired in v3"
        )
        or prompt_supersession.get("temporal_disposition", {}).get(
            "fresh_confirmatory_split_required"
        )
        is not True
        or prompt_validator.get("schema")
        != "metric-seam.patent-prompt-validator-freeze.v1"
        or prompt_validator.get("status")
        != "frozen-unscored-before-any-prompt-execution"
        or prompt_validator.get("validator_id") != "validate_post_code_response.v3"
        or prompt_validator.get("execution_contract", {}).get(
            "v3_post_code_responses_must_pass_this_exact_validator"
        )
        is not True
        or prompt_validator.get("execution_contract", {}).get(
            "prompt_responses_observed_before_freeze"
        )
        != 0
        or prompt_validator.get("tests", {}).get("combined_patent_stack_passed")
        != 77
    ):
        raise ValueError("patent prompt chronology/supersession drifted")

    if (
        heldout.get("schema")
        != "metric-seam.hierarchy-patent-claim-structure-execution.v3"
        or heldout.get("program_schema") != "metric-seam.patent-claim-structure.v13"
        or heldout.get("phase") != "heldout_pre_reference"
    ):
        raise ValueError("unexpected patent claim-structure heldout receipt")
    heldout_design = heldout.get("design", {})
    for key in (
        "outcome_or_reference_values_loaded",
        "prompt_outputs_loaded",
        "prior_art_or_examiner_evidence_loaded",
        "external_supervision_used",
        "whole_patent_score_emitted",
        "absence_certificate_permitted",
    ):
        if heldout_design.get(key) is not False:
            raise ValueError(f"patent heldout execution violated {key}")
    heldout_summary = heldout.get("summary", {})
    if (
        heldout_summary.get("n_items") != 150
        or heldout_summary.get("failure_types") != {}
        or heldout_summary.get("items_at_declared_character_cap") != 123
        or heldout_summary.get("status_counts")
        != {
            "measured": 27,
            "measured_with_possible_truncation": 122,
            "relation_abstained": 1,
        }
    ):
        raise ValueError("patent heldout execution summary drifted")

    relation_measurement = heldout_summary.get("relation_measurement", {})
    selected_relation_ids = {
        relation_id
        for cell in selected
        for relation_id in cell.get("ordered_relation_ids", [])
    }
    expected_relation_ids = {
        "claim_dependency_well_formedness",
        "claim_set_layering",
        "statutory_category_surface_coverage",
        "functional_limitation_incidence",
        "abstract_word_count",
    }
    if selected_relation_ids != expected_relation_ids:
        raise ValueError("patent heldout relation set drifted")
    if any(
        relation_measurement.get(relation_id, {}).get("n_measured", 0) <= 0
        or relation_measurement.get(relation_id, {}).get("nonconstant") is not True
        for relation_id in selected_relation_ids
    ):
        raise ValueError("a frozen patent output is not heldout measurable")
    certificates = heldout_summary.get("certificate_counts", {})
    for relation_id in (
        "claim_dependency_well_formedness",
        "statutory_category_surface_coverage",
        "functional_limitation_incidence",
    ):
        if certificates.get(relation_id, 0) <= 0:
            raise ValueError("a finite patent certificate channel is empty on heldout")

    return {
        "task": "patents",
        "instrument": "pure_code_claim_structure_v13",
        "panel_cells": 90,
        "static_relation_local_cells": 8,
        "train_gate_selected_cells": 5,
        "heldout_relation_measurable_cells": 5,
        "static_only_formatter_constant_cells": 3,
        "whole_construct_exact": 0,
        "static_to_operational_fraction": 5 / 8,
        "selected_mean_maximum_depth": 1.2,
        "balanced_panel": {
            "static_relation_local": {"n": 8, "denominator": 90, "rate": 8 / 90},
            "train_gate_selected": {"n": 5, "denominator": 90, "rate": 5 / 90},
            "heldout_relation_measurable": {
                "n": 5,
                "denominator": 90,
                "rate": 5 / 90,
            },
            "whole_construct_exact": {"n": 0, "denominator": 90, "rate": 0.0},
        },
        "selected_cells_by_level": gate_summary["selected_cells_by_level"],
        "selected_cells_by_maximum_depth": gate_summary[
            "selected_cells_by_maximum_depth"
        ],
        "selected_output_contracts": selected,
        "cap_policy": gate["cap_policy"],
        "heldout_pre_reference": {
            "items": int(heldout_summary["n_items"]),
            "status_counts": heldout_summary["status_counts"],
            "items_at_declared_character_cap": int(
                heldout_summary["items_at_declared_character_cap"]
            ),
            "relation_measurement": {
                relation_id: relation_measurement[relation_id]
                for relation_id in sorted(selected_relation_ids)
            },
            "finite_certificate_counts": {
                relation_id: certificates.get(relation_id, 0)
                for relation_id in (
                    "claim_dependency_well_formedness",
                    "statutory_category_surface_coverage",
                    "functional_limitation_incidence",
                )
            },
        },
        "prompt_batches_compiled_unscored": {
            "compiler_train": {
                "jobs": int(prompt_train["summary"]["n_jobs"]),
                "prompt_specs": int(prompt_train["summary"]["n_prompt_specs"]),
                "source_prompt_specs": int(
                    prompt_train["summary"]["n_source_prompt_specs"]
                ),
                "post_code_structured_specs": int(
                    prompt_train["summary"]["n_post_code_structured_specs"]
                ),
            },
            "heldout_pre_reference": {
                "jobs": int(prompt_heldout["summary"]["n_jobs"]),
                "prompt_specs": int(prompt_heldout["summary"]["n_prompt_specs"]),
                "source_prompt_specs": int(
                    prompt_heldout["summary"]["n_source_prompt_specs"]
                ),
                "post_code_structured_specs": int(
                    prompt_heldout["summary"]["n_post_code_structured_specs"]
                ),
            },
            "prompt_responses": 0,
            "reconstruction_estimates": 0,
            "isomorphism_adjudications": 0,
            "v1_packs": "superseded_not_executable",
            "v2_packs": "superseded_unexecuted",
            "v3_heldout_temporal_status": (
                "fixed_after_train_gate_exploratory_pre_reference"
            ),
            "semantic_validator": "validate_post_code_response.v3_frozen",
            "fresh_split_required_for_confirmatory_temporal_claim": True,
        },
        "axes": {
            "prompt_articulability": "jobs_compiled_unscored_not_measured",
            "code_verifiability": "heldout_relation_local_outputs_executed",
            "reconstruction_agreement": "not_estimated",
            "isomorphism": "shared_ctext_confirmed_other_axes_not_estimated",
            "codability": "not_estimated",
        },
        "interpretation": (
            "Five frozen partial-relation output contracts produced measurable heldout "
            "code-side variation or finite witnesses. This is positive relation-local "
            "verifiability on the presented bytes, not whole-criterion verification, "
            "prompt articulability, reconstruction, isomorphism, codability, patent "
            "validity, or external truth."
        ),
    }


def patent_claim_graph_additive_cross_audited_funnel() -> dict[str, Any]:
    """Summarize the independently audited deep claim-graph extension.

    The original proposal remains visible as a discovery artifact.  Current
    executable coverage counts only mappings whose emitted certificates survive
    exact replay and finite adversarial probes; conceptual mappings backed by a
    defective parser or an over-broad linkage rule remain quarantined.
    """

    audit = _read_json(
        HIERARCHY / "patents_claim_graph_additive_cross_audit_v1.json"
    )
    if (
        audit.get("schema")
        != "metric-seam.patent-claim-graph-additive-cross-audit.v1"
        or audit.get("status")
        != "independent_additive_cross_audit_complete_with_quarantines"
        or audit.get("task") != "patents"
    ):
        raise ValueError("unexpected additive patent claim-graph cross-audit")
    design = audit.get("design", {})
    if (
        design.get("canonical_artifacts_modified") is not False
        or design.get("external_supervision_used") is not False
        or design.get("models_apis_or_accelerators_used") is not False
        or design.get("prompt_outputs_loaded") is not False
        or design.get("references_or_outcomes_loaded") is not False
    ):
        raise ValueError("additive patent claim-graph audit crossed a forbidden channel")
    summary = audit.get("summary", {})
    union = audit.get("descriptive_union_check", {})
    expected = {
        "n_original_additive_cells": 8,
        "n_original_additive_mappings": 11,
        "n_construct_mappings_retained_as_relation_local": 11,
        "n_current_executable_cells_after_cross_audit": 5,
        "n_current_executable_mappings_after_cross_audit": 5,
        "n_quarantined_cells": 3,
        "n_quarantined_mappings": 6,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("additive patent claim-graph audit counts drifted")
    if (
        summary.get("all_pipeline_replays_exact") is not True
        or summary.get("current_executable_cell_depth_counts") != {"2": 3, "3": 2}
        or summary.get("current_executable_cells_by_level") != {"R2": 3, "R3": 2}
        or union.get("trusted_three_lane_union") != 19
        or union.get("original_three_lane_union") != 22
    ):
        raise ValueError("additive patent claim-graph cross-audit summary drifted")
    limits = audit.get("claim_limits", {})
    if (
        limits.get("whole_construct_cells") != 0
        or limits.get("codability_claim_permitted") is not False
        or limits.get("prompt_articulability_measured") is not False
        or limits.get("reference_reconstruction_measured") is not False
        or limits.get("isomorphism_measured") is not False
    ):
        raise ValueError("additive patent claim-graph claim limits drifted")

    return {
        "task": "patents",
        "instrument": "pure_code_claim_graph_additive_v1_cross_audited",
        "panel_cells": 90,
        "original_relation_local_cells": int(summary["n_original_additive_cells"]),
        "original_relation_local_mappings": int(
            summary["n_original_additive_mappings"]
        ),
        "conceptual_relation_local_mappings_retained": int(
            summary["n_construct_mappings_retained_as_relation_local"]
        ),
        "current_certificate_safe_cells": int(
            summary["n_current_executable_cells_after_cross_audit"]
        ),
        "current_certificate_safe_mappings": int(
            summary["n_current_executable_mappings_after_cross_audit"]
        ),
        "quarantined_cells": int(summary["n_quarantined_cells"]),
        "quarantined_mappings": int(summary["n_quarantined_mappings"]),
        "quarantined_relations": list(summary["quarantined_relations"]),
        "certificate_safe_by_level": summary["current_executable_cells_by_level"],
        "certificate_safe_by_depth": summary["current_executable_cell_depth_counts"],
        "heldout_markush_certificates": {
            "original": int(summary["heldout_markush_original_certificates"]),
            "retained_after_truncation_filter": int(
                summary["heldout_markush_retained_certificates"]
            ),
            "retained_items": int(summary["heldout_markush_retained_items"]),
        },
        "balanced_panel": {
            "original_relation_local": {"n": 8, "denominator": 90, "rate": 8 / 90},
            "current_certificate_safe": {"n": 5, "denominator": 90, "rate": 5 / 90},
            "whole_construct_exact": {"n": 0, "denominator": 90, "rate": 0.0},
        },
        "descriptive_three_lane_union": {
            "historical_cells": int(union["historical_cells"]),
            "canonical_cells": int(union["canonical_cells"]),
            "original_additive_cells": int(union["original_additive_cells"]),
            "trusted_current_additive_cells": int(
                union["trusted_current_additive_cells"]
            ),
            "original_union": int(union["original_three_lane_union"]),
            "trusted_current_union": int(union["trusted_three_lane_union"]),
            "interpretation": union["interpretation"],
        },
        "axes": {
            "prompt_articulability": "not_measured",
            "code_verifiability": "five_relation_local_certificate_classes_cross_audited",
            "reconstruction_agreement": "not_estimated",
            "isomorphism": "not_estimated",
            "codability": "not_estimated",
        },
        "interpretation": (
            "Five of 90 additive claim-graph cells currently have adversarially safe "
            "relation-local certificate classes (three depth 2 and two depth 3). Six "
            "other mappings remain conceptually relation-local but their numeric/formula "
            "implementations are quarantined. The provenance-separated 19-cell union is "
            "descriptive coverage, not codability, articulability, reconstruction, or "
            "isomorphism."
        ),
    }


def math_hierarchy_static_funnel() -> dict[str, Any]:
    """Summarize the cross-audited retrospective math static-witness lane.

    This reads source-only construct-fidelity adjudication and its descriptive
    expansion.  The historical programs have not been executed in this lane,
    so none of the returned rates measures code verifiability, reconstruction,
    isomorphism, codability, or a hierarchy trend.
    """

    fidelity = _read_json(
        HIERARCHY / "math_stackexchange_construct_fidelity_merged_v1.json"
    )
    prevalence = _read_json(
        HIERARCHY / "math_stackexchange_static_witness_prevalence_v1.json"
    )
    if fidelity.get("schema") != "metric-seam.math-construct-fidelity-merged.v1":
        raise ValueError("unexpected math hierarchy fidelity schema")
    if prevalence.get("schema") != "metric-seam.math-static-witness-prevalence.v1":
        raise ValueError("unexpected math hierarchy prevalence schema")
    if fidelity.get("status") != "static_construct_fidelity_complete_pre_execution":
        raise ValueError("math hierarchy fidelity audit is incomplete")
    if prevalence.get("status") != "static_descriptive_rates_cross_audited":
        raise ValueError("math hierarchy prevalence is incomplete")
    if (
        fidelity.get("task") != "math-stackexchange"
        or prevalence.get("task") != "math-stackexchange"
    ):
        raise ValueError("math hierarchy task binding drifted")
    if fidelity.get("panel_content_sha256") != prevalence.get("panel_content_sha256"):
        raise ValueError("math hierarchy artifacts use different panels")
    if fidelity.get("cross_audit") != {
        "status": "complete",
        "n_guarded_changes": 21,
        "provisional_until_complete": False,
    }:
        raise ValueError("math hierarchy cross-audit binding drifted")
    if prevalence.get("cross_audit") != fidelity.get("cross_audit"):
        raise ValueError("math hierarchy prevalence predates the cross-audit")
    forbidden_loads = (
        "execution_performed",
        "items_loaded",
        "reference_values_loaded",
        "outcome_labels_loaded",
        "program_outputs_loaded",
        "external_supervision",
    )
    if any(fidelity.get(field) is not False for field in forbidden_loads):
        raise ValueError("math hierarchy static audit loaded a forbidden execution channel")

    summary = fidelity["summary"]
    if (
        summary.get("n_cells"),
        summary.get("n_retrieved_candidates"),
        summary.get("eligible_for_relation_local_execution"),
        summary.get("whole_construct_exact_count"),
        summary.get("eligible_audited_depths"),
    ) != (90, 47, 33, 0, {"1": 10, "2": 23}):
        raise ValueError("math hierarchy static counts drifted")
    level_witnesses = {
        level: int(summary["by_level"][level]["eligible_for_relation_local_execution"])
        for level in ("R1", "R2", "R3")
    }
    if level_witnesses != {"R1": 12, "R2": 6, "R3": 15}:
        raise ValueError("math hierarchy level counts drifted")

    pooled = prevalence["pooled_eligible_action_nodes"]
    balanced = pooled["balanced_panel"]
    expansion = pooled["eligible_inventory_stratum_expansion"]
    if balanced["relation_local_static_fidelity"]["rate"] != 0.366667:
        raise ValueError("math balanced-panel point estimate drifted")
    if expansion["relation_local_static_fidelity"]["rate"] != 0.361266:
        raise ValueError("math eligible-inventory expansion drifted")

    return {
        "task": "math-stackexchange",
        "historical_program_families": int(summary["n_unique_eligible_programs"]),
        "panel_cells": int(summary["n_cells"]),
        "retrieved_candidates": int(summary["n_retrieved_candidates"]),
        "relation_local_static_witnesses": int(
            summary["eligible_for_relation_local_execution"]
        ),
        "whole_construct_exact": int(summary["whole_construct_exact_count"]),
        "balanced_panel": {
            outcome: balanced[outcome]
            for outcome in (
                "retrieved_candidate",
                "relation_local_static_fidelity",
                "whole_construct_exact",
            )
        },
        "eligible_inventory_stratum_expansion": {
            "population_nodes": int(
                prevalence["sampling_frame"]["n_eligible_action_node_records"]
            ),
            **{
                outcome: expansion[outcome]
                for outcome in (
                    "retrieved_candidate",
                    "relation_local_static_fidelity",
                    "whole_construct_exact",
                )
            },
        },
        "witnesses_by_level": level_witnesses,
        "witnesses_by_audited_depth": {
            int(depth): int(count)
            for depth, count in summary["eligible_audited_depths"].items()
        },
        "cross_audit": fidelity["cross_audit"],
        "axes": {
            "prompt_articulability": "not_measured",
            "code_verifiability": "not_measured_static_source_audit_only",
            "reconstruction_agreement": "not_estimated",
            "isomorphism": "not_estimated",
            "codability": "not_estimated",
            "hierarchy_trend": "not_estimated",
        },
        "claim_limits": prevalence["claim_limits"],
        "interpretation": (
            "Retrospective relation-local static witness coverage in a manual historical "
            "hybrid bank. No candidate program was executed, and the R1/R2/R3 points do "
            "not establish a hierarchy trend."
        ),
    }


def math_hierarchy_symbolic_capability_sensitivity() -> dict[str, Any]:
    """Summarize the additive, source-only SymPy capability sensitivity.

    The canonical 33-cell math result is left unchanged.  This readout asks
    how many additional panel cells have a narrowly adjudicated presented-step
    rational-equality relation in a previously manual SymPy/Lark capability.
    Neither the capability nor any item is executed here.
    """

    fidelity = _read_json(
        HIERARCHY
        / "math_stackexchange_symbolic_capability_construct_fidelity_v1.json"
    )
    prevalence = _read_json(
        HIERARCHY
        / "math_stackexchange_symbolic_capability_expansion_prevalence_v1.json"
    )
    if fidelity.get("schema") != (
        "metric-seam.math-symbolic-capability-construct-fidelity.v1"
    ):
        raise ValueError("unexpected math symbolic fidelity schema")
    if prevalence.get("schema") != (
        "metric-seam.math-symbolic-capability-expansion-prevalence.v1"
    ):
        raise ValueError("unexpected math symbolic prevalence schema")
    if fidelity.get("status") != (
        "static_five_dimension_adjudication_complete_pre_execution"
    ):
        raise ValueError("math symbolic fidelity audit is incomplete")
    if prevalence.get("status") != (
        "static_additive_sensitivity_complete_pre_execution"
    ):
        raise ValueError("math symbolic sensitivity is incomplete")
    if (
        fidelity.get("task") != "math-stackexchange"
        or prevalence.get("task") != "math-stackexchange"
        or fidelity.get("panel_content_sha256")
        != prevalence.get("panel_content_sha256")
    ):
        raise ValueError("math symbolic artifacts use a different task or panel")
    for field in (
        "programs_or_items_executed",
        "certificate_counts_loaded",
        "prompt_outputs_loaded",
        "reference_values_loaded",
        "outcome_labels_loaded",
        "correlations_or_reconstruction_loaded",
        "models_apis_or_gpus_used",
    ):
        if fidelity.get(field) is not False:
            raise ValueError(f"math symbolic audit crossed forbidden boundary: {field}")
    if (
        prevalence.get("program_or_item_execution_emitted") is not False
        or prevalence.get("prompt_reference_outcome_or_reconstruction_stages_emitted")
        is not False
        or prevalence.get("canonical_artifact_modified") is not False
    ):
        raise ValueError("math symbolic prevalence crossed a frozen boundary")

    summary = fidelity["summary"]
    expected_summary = {
        "n_cells": 90,
        "n_retrieved_candidates": 15,
        "n_relation_local_static_matches": 7,
        "n_retrieved_relation_mismatches": 8,
        "n_newly_covered_cells": 5,
        "n_existing_cells_adding_formal_symbolic_relation": 2,
        "canonical_relation_local_cells_unchanged": 33,
        "additive_sensitivity_union_cells": 38,
        "n_whole_construct_exact": 0,
        "accepted_by_level": {"R1": 1, "R2": 3, "R3": 3},
        "newly_covered_by_level": {"R2": 2, "R3": 3},
    }
    if summary != expected_summary:
        raise ValueError("math symbolic sensitivity counts drifted")
    depth = prevalence.get("relation_depth_receipt", {})
    if (
        depth.get("depth") != 3
        or depth.get("formal_symbolic_matched_cells") != 7
        or depth.get("newly_covered_at_depth3") != 5
        or depth.get("isolation_or_test_execution_adds_depth") is not False
    ):
        raise ValueError("math symbolic relation-depth receipt drifted")
    pooled = prevalence["pooled_eligible_action_nodes"]
    balanced = pooled["balanced_panel"]
    expansion = pooled["eligible_inventory_stratum_expansion"]
    if (
        balanced["canonical_relation_local_unchanged"]["rate"] != 0.366667
        or balanced["additive_sensitivity_union_relation_local"]["rate"]
        != 0.422222
        or expansion["canonical_relation_local_unchanged"]["rate"] != 0.361266
        or expansion["additive_sensitivity_union_relation_local"]["rate"]
        != 0.376231
    ):
        raise ValueError("math symbolic sensitivity rates drifted")

    return {
        "task": "math-stackexchange",
        "panel_cells": 90,
        "canonical_relation_local_cells": 33,
        "retrieved_candidates": 15,
        "formal_symbolic_relation_local_cells": 7,
        "newly_covered_cells": 5,
        "existing_cells_adding_formal_symbolic_relation": 2,
        "additive_union_cells": 38,
        "whole_construct_exact": 0,
        "balanced_panel": balanced,
        "eligible_inventory_stratum_expansion": {
            "population_nodes": int(
                prevalence["sampling_frame"]["n_eligible_action_node_records"]
            ),
            **expansion,
        },
        "accepted_by_level": summary["accepted_by_level"],
        "newly_covered_by_level": summary["newly_covered_by_level"],
        "matched_relation_depth": depth,
        "axes": {
            "prompt_articulability": "not_measured",
            "code_verifiability": "not_measured_static_source_audit_only",
            "reconstruction_agreement": "not_estimated",
            "isomorphism": "not_estimated",
            "codability": "not_estimated",
        },
        "claim_limits": prevalence["claim_limits"],
        "interpretation": (
            "A previously manual formal-symbolic capability adds five relation-local "
            "panel cells in a static source sensitivity. The canonical result remains "
            "33/90; 38/90 is the additive union, not executed codability."
        ),
    }


def math_hierarchy_operational_funnel() -> dict[str, Any]:
    """Summarize the executed, target-free constant-L math slices.

    The operational object is ``g_c(x)=f(x,c)`` with every historical LLM
    field fixed to a sentinel chosen only by train measurability.  It is not
    the original hybrid, a pure-code rewrite, or a reconstruction score.
    """

    result = _read_json(
        HIERARCHY / "math_stackexchange_lclamp_operational_prevalence_v1.json"
    )
    prompt_train = _read_json(
        HIERARCHY / "math_stackexchange_prompt_train_manifest_v1.json"
    )
    prompt_heldout = _read_json(
        HIERARCHY / "math_stackexchange_prompt_heldout_fixed_manifest_v1.json"
    )
    if result.get("schema") != "metric-seam.math-lclamp-operational-prevalence.v1":
        raise ValueError("unexpected math L-clamp operational schema")
    if result.get("status") != (
        "complete_static_train_and_pre_reference_heldout_funnel"
    ):
        raise ValueError("math L-clamp operational funnel is incomplete")
    if result.get("task") != "math-stackexchange":
        raise ValueError("math L-clamp operational task binding drifted")
    for prompt, phase, jobs in (
        (prompt_train, "compiler_train", 295200),
        (prompt_heldout, "heldout_pre_reference", 128700),
    ):
        if (
            prompt.get("schema")
            != "metric-seam.math-prompt-articulability-batch.v1"
            or prompt.get("status") != "compiled_unscored"
            or prompt.get("task") != "math-stackexchange"
            or prompt.get("phase") != phase
            or prompt.get("summary", {}).get("n_jobs") != jobs
            or prompt.get("summary", {}).get("n_prompt_responses") != 0
            or prompt.get("summary", {}).get("n_reconstruction_estimates") != 0
            or prompt.get("summary", {}).get("n_isomorphism_adjudications") != 0
            or set(prompt.get("forbidden_inputs", {}).values()) != {False}
            or prompt.get("jobs_artifact", {}).get("model_or_api_calls_performed")
            is not False
        ):
            raise ValueError(f"math {phase} prompt manifest drifted")
    if prompt_train.get("construct_fidelity_fingerprint") != prompt_heldout.get(
        "construct_fidelity_fingerprint"
    ):
        raise ValueError("math prompt phases use different construct audits")
    contract = result.get("channel_contract", {})
    if contract != {
        "program_execution_outputs_read": True,
        "item_text_loaded": False,
        "reference_values_loaded": False,
        "outcome_labels_loaded": False,
        "prompt_or_llm_values_loaded": False,
        "models_or_apis_called": False,
        "accelerators_used": False,
        "score_direction_or_target_used_for_selection": False,
    }:
        raise ValueError("math L-clamp channel contract drifted")
    validation = result.get("validation", {})
    expected_counts = {
        "static_relation_local_witness": 33,
        "train_operational_constant_l_slice": 33,
        "heldout_measurable_constant_l_slice": 33,
    }
    if validation.get("stage_relation_mapping_counts") != expected_counts:
        raise ValueError("math L-clamp stage counts drifted")
    if validation.get("compiler_train", {}).get("three_state_totals") != {
        "measured": 36000,
        "abstained": 0,
        "failed": 0,
    }:
        raise ValueError("math L-clamp train measurements drifted")
    if validation.get("heldout_pre_reference", {}).get("three_state_totals") != {
        "measured": 2400,
        "abstained": 0,
        "failed": 0,
    }:
        raise ValueError("math L-clamp heldout measurements drifted")
    pooled = result["pooled_eligible_action_nodes"]
    balanced = pooled["balanced_panel"]
    expansion = pooled["eligible_inventory_stratum_expansion"]
    if any(balanced[stage]["rate"] != 0.366667 for stage in expected_counts):
        raise ValueError("math L-clamp balanced rates drifted")
    if any(expansion[stage]["rate"] != 0.361266 for stage in expected_counts):
        raise ValueError("math L-clamp expanded rates drifted")
    sensitivity = result["unsupervised_sentinel_sensitivity"]
    if (
        sensitivity.get("used_for_train_gate_selection") is not False
        or sensitivity.get("used_for_heldout_decisions") is not False
        or sensitivity.get("reference_values_used") is not False
        or sensitivity.get("outcome_labels_used") is not False
        or sensitivity.get("score_direction_or_target_used") is not False
    ):
        raise ValueError("math sentinel sensitivity crossed a selection boundary")

    return {
        "task": "math-stackexchange",
        "panel_cells": 90,
        "unique_programs": int(
            validation["construct_fidelity"]["n_unique_programs"]
        ),
        "stage_relation_mapping_counts": expected_counts,
        "balanced_panel": balanced,
        "eligible_inventory_stratum_expansion": {
            "population_nodes": int(
                result["sampling_frame"]["n_eligible_action_node_records"]
            ),
            **expansion,
        },
        "stage_retention": pooled["stage_retention"],
        "compiler_train": validation["compiler_train"],
        "heldout_pre_reference": validation["heldout_pre_reference"],
        "by_level": result["by_level"],
        "by_audited_depth": result["by_audited_depth"],
        "sentinel_sensitivity": sensitivity,
        "prompt_batches": {
            "compiler_train": {
                "status": prompt_train["status"],
                **prompt_train["summary"],
            },
            "heldout_pre_reference": {
                "status": prompt_heldout["status"],
                **prompt_heldout["summary"],
            },
            "raw_signed_heldout_primary": prompt_heldout[
                "heldout_analysis_preregistration"
            ]["primary_reconstruction"],
            "isomorphism_polarity_gate": prompt_heldout[
                "heldout_analysis_preregistration"
            ]["isomorphism_polarity_gate"],
        },
        "scientific_object": result["scientific_object"],
        "axes": {
            "prompt_articulability": "not_measured_jobs_compiled_unscored",
            "code_verifiability": (
                "relation_local_constant_l_conditional_variation_established; "
                "original_hybrid_and_whole_construct_not_established"
            ),
            "reconstruction_agreement": "not_estimated",
            "isomorphism": "not_estimated",
            "codability": "not_estimated",
            "hierarchy_trend": "not_estimated",
        },
        "claim_limits": result["claim_limits"],
        "interpretation": (
            "All 33 audited relation mappings retained nonconstant, failure-free scores "
            "under train-selected constant-L slices on heldout text. This establishes "
            "conditional code-side variation only."
        ),
    }


def science_hierarchy_static_funnel() -> dict[str, Any]:
    """Summarize the full-article science claim verifier's static mappings.

    The single historical capability is manual, pure code, and document
    internal.  This pass did not execute it or load articles, outputs, prompt
    references, outcomes, or external scientific-truth supervision.
    """

    fidelity = _read_json(
        HIERARCHY / "peer_review_science_claim_construct_fidelity_v1.json"
    )
    prevalence = _read_json(
        HIERARCHY / "peer_review_science_claim_static_prevalence_v1.json"
    )
    if (
        fidelity.get("schema")
        != "metric-seam.hierarchy-science-claim-construct-fidelity.v1"
    ):
        raise ValueError("unexpected science hierarchy fidelity schema")
    if (
        prevalence.get("schema")
        != "metric-seam.science-claim-static-witness-prevalence.v1"
    ):
        raise ValueError("unexpected science hierarchy prevalence schema")
    if (
        fidelity.get("status")
        != "static-relation-local-adjudication-complete-pre-execution"
    ):
        raise ValueError("science hierarchy fidelity audit is incomplete")
    if prevalence.get("status") != "static_descriptive_rates_complete_pre_execution":
        raise ValueError("science hierarchy prevalence is incomplete")
    if fidelity.get("task") != "peer-review" or prevalence.get("task") != "peer-review":
        raise ValueError("science hierarchy task binding drifted")
    if (
        fidelity.get("source_panel_content_sha256")
        != prevalence.get("panel_content_sha256")
    ):
        raise ValueError("science hierarchy artifacts use different panels")

    fidelity_false_flags = (
        "articles_or_items_loaded",
        "execution_performed",
        "external_supervision_loaded_for_this_audit",
        "historical_certificates_or_program_outputs_loaded",
        "outcome_labels_loaded",
        "prompt_or_reconstruction_outputs_loaded",
        "reference_values_loaded",
    )
    prevalence_false_flags = (
        "execution_or_outcome_stages_emitted",
        "prompt_or_model_stages_emitted",
        "reconstruction_or_isomorphism_stages_emitted",
        "uncertainty_intervals_emitted",
    )
    if any(fidelity.get(field) is not False for field in fidelity_false_flags):
        raise ValueError("science static audit loaded a forbidden execution channel")
    if any(prevalence.get(field) is not False for field in prevalence_false_flags):
        raise ValueError("science prevalence emitted a forbidden downstream stage")

    summary = fidelity["summary"]
    if (
        fidelity.get("n_cells"),
        summary.get("n_retrieved"),
        summary.get("n_partial_relation_local"),
        summary.get("n_relation_mismatch"),
        summary.get("n_exact_whole_construct"),
        summary.get("n_execution_witnesses"),
        summary.get("n_external_scientific_truth_claims"),
        summary.get("maximum_matching_relation_depth_counts"),
    ) != (90, 9, 6, 3, 0, 0, 0, {"3": 6}):
        raise ValueError("science hierarchy static counts drifted")
    level_witnesses = {
        level: int(summary["by_level"][level]["n_partial_relation_local"])
        for level in ("R1", "R2", "R3")
    }
    if level_witnesses != {"R1": 2, "R2": 2, "R3": 2}:
        raise ValueError("science hierarchy level counts drifted")

    pooled = prevalence["pooled_eligible_action_nodes"]
    balanced = pooled["balanced_panel"]
    expansion = pooled["eligible_inventory_stratum_expansion"]
    if balanced["relation_local_static_fidelity"]["rate"] != 0.066667:
        raise ValueError("science balanced-panel point estimate drifted")
    if expansion["relation_local_static_fidelity"]["rate"] != 0.055407:
        raise ValueError("science eligible-inventory expansion drifted")

    return {
        "task": "peer-review",
        "historical_program_families": 1,
        "panel_cells": int(fidelity["n_cells"]),
        "retrieved_candidates": int(summary["n_retrieved"]),
        "relation_local_static_witnesses": int(summary["n_partial_relation_local"]),
        "relation_mismatches": int(summary["n_relation_mismatch"]),
        "whole_construct_exact": int(summary["n_exact_whole_construct"]),
        "balanced_panel": {
            outcome: balanced[outcome]
            for outcome in (
                "retrieved_candidate",
                "relation_local_static_fidelity",
                "depth3_relation_local_static_fidelity",
                "whole_construct_exact",
            )
        },
        "eligible_inventory_stratum_expansion": {
            "population_nodes": int(
                prevalence["sampling_frame"]["n_eligible_action_node_records"]
            ),
            **{
                outcome: expansion[outcome]
                for outcome in (
                    "retrieved_candidate",
                    "relation_local_static_fidelity",
                    "depth3_relation_local_static_fidelity",
                    "whole_construct_exact",
                )
            },
        },
        "witnesses_by_level": level_witnesses,
        "witnesses_by_audited_depth": {3: int(summary["n_with_depth3_matching_relation"])},
        "channel_provenance": prevalence["channel_provenance"],
        "axes": {
            "prompt_articulability": "not_measured",
            "code_verifiability": "not_measured_static_source_audit_only",
            "reconstruction_agreement": "not_estimated",
            "isomorphism": "not_estimated",
            "codability": "not_estimated",
            "external_scientific_truth": "not_estimated_document_internal_only",
            "hierarchy_trend": "not_estimated",
        },
        "claim_limits": prevalence["claim_limits"],
        "interpretation": (
            "Retrospective static coverage of numeric/comparative document-internal "
            "claim consistency by one manual full-article pure-code verifier. No program "
            "execution or external scientific-truth adjudication occurred in this pass."
        ),
    }


def science_hierarchy_fullarticle_operational_funnel() -> dict[str, Any]:
    """Summarize the canonical blocker and additive full-article CPU run.

    The canonical hierarchy items are abstract-only, so they cannot supply the
    distinct body evidence required by the historical verifier while keeping
    prompt and code bytes identical.  The executed result therefore uses a
    separately frozen, outcome-blind abstract+body sample and is explicitly
    non-comparable to canonical hierarchy item execution.
    """

    blocker = _read_json(
        HIERARCHY / "peer_review_science_canonical_representation_blocker_v1.json"
    )
    result = _read_json(
        HIERARCHY
        / "peer_review_science_fullarticle_operational_prevalence_v1.json"
    )
    addressed_binding = _read_json(
        HIERARCHY / "peer_review_science_addressed_subset_binding_v1.json"
    )
    exact_prompt_dir = HIERARCHY / "science_exact_ctext_prompt_v1"
    exact_prompt = _read_json(exact_prompt_dir / "manifest.json")
    exact_prompt_receipt = _read_json(exact_prompt_dir / "audit_receipt.json")
    if blocker.get("schema") != (
        "metric-seam.science-canonical-representation-blocker.v1"
    ):
        raise ValueError("unexpected science representation-blocker schema")
    if blocker.get("status") != (
        "canonical_execution_blocked_by_representation_mismatch"
    ):
        raise ValueError("canonical science representation blocker is incomplete")
    if result.get("schema") != (
        "metric-seam.science-fullarticle-operational-prevalence.v1"
    ):
        raise ValueError("unexpected science full-article operational schema")
    if result.get("status") != (
        "additive_representation_static_train_heldout_funnel_complete"
    ):
        raise ValueError("science full-article operational funnel is incomplete")
    if (
        addressed_binding.get("schema_version")
        != "metric-seam.science-fullarticle-addressed-subset-binding.v1"
        or addressed_binding.get("status")
        != "cpu_only_subset_binding_complete_pre_prompt"
        or addressed_binding.get("task") != "peer-review"
    ):
        raise ValueError("science addressed subset binding is incomplete")
    if (
        exact_prompt.get("schema_version")
        != "metric-seam.science-exact-ctext-prompt-bundle.v1"
        or exact_prompt.get("status")
        != "compiled_unscored_zero_calls_exact_shared_payload"
        or exact_prompt.get("task") != "peer-review"
        or exact_prompt_receipt.get("schema_version")
        != "metric-seam.science-exact-ctext-prompt-receipt.v1"
        or exact_prompt_receipt.get("status")
        != "cpu_only_exact_payload_replay_validated_zero_calls"
    ):
        raise ValueError("science exact-ctext prompt instrument is incomplete")
    if blocker.get("task") != "peer-review" or result.get("task") != "peer-review":
        raise ValueError("science operational task binding drifted")
    if set(blocker.get("forbidden_inputs", {}).values()) != {False}:
        raise ValueError("science blocker crossed a forbidden input boundary")
    if blocker.get("execution") != {
        "performed": False,
        "three_state_outputs": {"abstained": 0, "failed": 0, "measured": 0},
        "why_not": (
            "supplementing canonical ctext with a separately joined body would give "
            "the code arm evidence not present in the shared prompt/code bytes"
        ),
    }:
        raise ValueError("canonical science execution was not kept blocked")
    pooled_join = blocker["coverage_audit"]["pooled"]
    if pooled_join != {
        "n_exact_abstract_joins": 12,
        "n_exact_joins_with_nonempty_body": 6,
        "n_items": 300,
    }:
        raise ValueError("canonical science representation audit drifted")
    expected_contract = {
        "accelerators_used": False,
        "external_supervision_used": False,
        "item_text_loaded_by_summary": False,
        "models_or_apis_called": False,
        "outcome_values_loaded": False,
        "program_execution_outputs_read": True,
        "prompt_or_reconstruction_outputs_loaded": False,
        "reference_values_loaded": False,
    }
    if result.get("channel_contract") != expected_contract:
        raise ValueError("science full-article channel contract drifted")
    validation = result["validation"]
    expected_stages = {
        "static_relation_local_witness": 6,
        "train_operational_fullarticle_section_verifier": 6,
        "heldout_measurable_fullarticle_section_verifier": 6,
    }
    if validation.get("stage_relation_mapping_counts") != expected_stages:
        raise ValueError("science full-article stage counts drifted")
    expected_train = {
        "measured": 118,
        "abstained": 32,
        "failed": 0,
    }
    expected_heldout = {
        "measured": 108,
        "abstained": 42,
        "failed": 0,
    }
    train = result["item_execution"]["compiler_train"]
    heldout = result["item_execution"]["heldout_pre_reference"]
    if train.get("three_state_totals_unique_items") != expected_train:
        raise ValueError("science full-article train totals drifted")
    if heldout.get("three_state_totals_unique_items") != expected_heldout:
        raise ValueError("science full-article heldout totals drifted")
    if (
        train.get("n_relation_certificates") != 7
        or heldout.get("n_relation_certificates") != 10
        or heldout.get("n_items_with_relation_certificate") != 9
    ):
        raise ValueError("science relation-certificate counts drifted")
    representation = result["representation"]
    if (
        representation.get("canonical_hierarchy_items") is not False
        or representation.get("direct_comparison_to_canonical_abstract_only_execution")
        is not False
        or representation.get("same_bytes_for_future_prompt_and_current_code")
        is not True
        or representation.get("complete_pdf_claimed") is not False
    ):
        raise ValueError("science full-article representation scope drifted")
    pooled = result["pooled_eligible_action_nodes"]
    balanced = pooled["balanced_panel"]
    expansion = pooled["eligible_inventory_stratum_expansion"]
    if any(balanced[stage]["rate"] != 0.066667 for stage in expected_stages):
        raise ValueError("science full-article balanced rates drifted")
    if any(expansion[stage]["rate"] != 0.055407 for stage in expected_stages):
        raise ValueError("science full-article expanded rates drifted")

    prompt_plane = addressed_binding["prompt_plane"]
    representation_contract = addressed_binding["representation_contract"]
    agreement = addressed_binding["combined_summary"]
    temporal = addressed_binding["temporal_disposition"]
    execution_policy = addressed_binding["execution_policy"]
    if prompt_plane != {
        "distinct_prepared_unscored_request_records": 235,
        "planned_stateless_passes": 2,
        "planned_two_pass_prompt_jobs_if_executed": 470,
        "prompt_articulability_measured": False,
        "prompt_code_reconstruction_measured": False,
        "prompt_responses": 0,
        "selected_items": 300,
        "six_relation_mappings_share_one_result_vector": True,
        "structural_abstentions_without_remote_call": 65,
        "two_pass_jobs_materialized_as_separate_requests": False,
    }:
        raise ValueError("science addressed prompt-plane counts drifted")
    if (
        representation_contract.get("same_evidence_content") is not True
        or representation_contract.get(
            "same_source_address_inventory_for_v8_prompt_and_v9_code"
        )
        is not True
        or representation_contract.get("same_input_representation") is not False
        or representation_contract.get("exact_hierarchy_ctext_rendered_to_prompt")
        is not False
        or representation_contract.get("full_isomorphism_licensed") is not False
    ):
        raise ValueError("science addressed representation boundary drifted")
    if (
        agreement.get("v9_hierarchy_item_field_agreement", {}).get("agree") != 300
        or agreement.get("v9_hierarchy_item_field_agreement", {}).get("total") != 300
        or agreement.get("v9_hierarchy_aggregate_exact_for_both_splits") is not True
        or agreement.get("prompt_transport")
        != {
            "compiled_unscored_request": 235,
            "structural_abstention_no_remote_call": 65,
        }
        or temporal.get("fresh_split_required_for_confirmatory_prompt_code_claim")
        is not True
        or execution_policy.get("models_or_apis_called") is not False
        or execution_policy.get("accelerators_used") is not False
    ):
        raise ValueError("science addressed subset agreement drifted")

    exact_summary = exact_prompt["summary"]
    exact_representation = exact_prompt["representation_contract"]
    exact_policy = exact_prompt["execution_policy"]
    exact_loaded = exact_prompt["loaded_input_policy"]
    exact_chronology = exact_prompt["chronology"]
    exact_transport = exact_prompt["transport_control_inventory"]
    exact_target = exact_prompt["future_comparison_target"]
    exact_validation = exact_prompt_receipt["validation"]
    exact_claim_boundary = exact_prompt_receipt["claim_boundary"]
    if exact_summary != {
        "articulability_measurements": 0,
        "compiled_prompt_pass_records": 470,
        "mapping_record_applications_if_executed": 2820,
        "n_relation_mappings": 6,
        "pass_expanded_result_slots": 600,
        "pass_expanded_structural_no_call_outcomes": 130,
        "planned_stateless_passes": 2,
        "prompt_eligible_unique_items": 235,
        "prompt_responses": 0,
        "reconstruction_measurements": 0,
        "structural_abstention_unique_items": 65,
        "unique_items": 300,
    }:
        raise ValueError("science exact-ctext prompt counts drifted")
    if (
        exact_representation.get("class")
        != "exact_shared_ctext_payload_with_prompt_scaffolding"
        or exact_representation.get("same_frozen_ctext_payload_bytes_as_current_code")
        is not True
        or exact_representation.get(
            "decoded_model_visible_user_content_contains_ctext_once"
        )
        is not True
        or exact_representation.get("raw_jsonl_or_provider_wire_byte_identity_claimed")
        is not False
        or exact_representation.get("whole_request_identity_claimed") is not False
        or exact_representation.get("full_semantic_isomorphism_licensed") is not False
        or exact_representation.get("all_nonstandard_transport_controls_preserved")
        is not True
        or exact_representation.get("provider_transport_compatibility_tested")
        is not False
        or exact_policy.get("model_calls_made") != 0
        or exact_policy.get("api_calls_made") != 0
        or exact_policy.get("remote_calls_made") != 0
        or exact_policy.get("gpu_or_accelerator_used") is not False
        or exact_loaded.get("item_level_code_outputs_or_results_loaded") is not False
        or exact_loaded.get("outcomes_or_reference_values_loaded") is not False
        or exact_chronology.get(
            "fresh_split_required_for_confirmatory_reconstruction_or_isomorphism"
        )
        is not True
        or exact_validation.get("decoded_exact_payload_records") != 470
        or exact_validation.get("payload_mismatches") != 0
        or exact_validation.get("payload_multiple_occurrences") != 0
        or exact_transport.get("eligible_unique_items") != 36
        or exact_transport.get("compiled_prompt_pass_records") != 72
        or exact_transport.get("nul_u0000_eligible_unique_items") != 22
        or exact_transport.get("nul_u0000_compiled_prompt_pass_records") != 44
        or exact_transport.get("line_separator_u2028_eligible_unique_items") != 1
        or exact_transport.get("line_separator_u2028_compiled_prompt_pass_records")
        != 2
        or exact_target.get("name")
        != "relation_local_numeric_comparative_projection"
        or exact_target.get("whole_frozen_code_vector") is not False
        or exact_target.get("code_projection_compiled_and_replay_bound") is not True
        or exact_target.get("reconstruction_decisions")
        != ["contradicted", "insufficient", "supported"]
        or exact_target.get("code_projection_summary")
        != {
            "decision_counts": {"insufficient": 141, "supported": 17},
            "evidence_link_decisions": 0,
            "items": 300,
            "selected_claims": 158,
        }
        or exact_target.get("evidence_link_in_reconstruction_target") is not False
        or exact_claim_boundary.get("response_validation_checks_relation_truth")
        is not False
        or exact_claim_boundary.get("response_validation_checks_decision_correctness")
        is not False
    ):
        raise ValueError("science exact-ctext representation boundary drifted")

    return {
        "task": "peer-review",
        "canonical_representation_blocker": {
            "status": blocker["status"],
            "canonical_items": pooled_join["n_items"],
            "exact_abstract_joins": pooled_join["n_exact_abstract_joins"],
            "exact_joins_with_nonempty_body": pooled_join[
                "n_exact_joins_with_nonempty_body"
            ],
            "execution_performed": blocker["execution"]["performed"],
            "reason": blocker["execution"]["why_not"],
        },
        "representation": representation,
        "panel_cells": 90,
        "stage_relation_mapping_counts": expected_stages,
        "balanced_panel": balanced,
        "eligible_inventory_stratum_expansion": {
            "population_nodes": int(
                result["sampling_frame"]["n_eligible_action_node_records"]
            ),
            **expansion,
        },
        "stage_retention": pooled["stage_retention"],
        "compiler_train": train,
        "heldout_pre_reference": heldout,
        "additive_addressed_prompt_overlay": {
            "prompt_plane": prompt_plane,
            "representation_contract": representation_contract,
            "code_replay_agreement": agreement[
                "v9_hierarchy_item_field_agreement"
            ],
            "code_aggregate_exact_for_both_splits": agreement[
                "v9_hierarchy_aggregate_exact_for_both_splits"
            ],
            "split_prompt_transport": {
                split: addressed_binding["split_summaries"][split][
                    "prompt_transport"
                ]
                for split in ("compiler_train", "sealed_heldout")
            },
            "temporal_disposition": temporal,
            "method_origin": addressed_binding["method_origin"],
            "claim_boundary": addressed_binding["claim_boundary"],
        },
        "exact_ctext_prompt_instrument": {
            "summary": exact_summary,
            "by_phase": exact_prompt["by_phase"],
            "representation_contract": exact_representation,
            "transport_control_inventory": exact_transport,
            "future_comparison_target": exact_target,
            "chronology": exact_chronology,
            "validation": exact_validation,
            "claim_boundary": exact_claim_boundary,
            "method_origin": exact_prompt["method_origin"],
            "interpretation": exact_prompt["interpretation"],
        },
        "by_level": result["by_level"],
        "scientific_object": result["scientific_object"],
        "axes": {
            "prompt_articulability": "not_measured",
            "code_verifiability": (
                "relation_local_document_internal_execution_established_on_additive_"
                "fullarticle_section_representation"
            ),
            "reconstruction_agreement": "not_estimated",
            "isomorphism": "not_estimated_future_same_bytes_contract_frozen",
            "codability": "not_estimated",
            "external_scientific_truth": False,
        },
        "claim_limits": result["claim_limits"],
        "interpretation": (
            "The canonical abstract-only hierarchy execution remains blocked. On a "
            "separate outcome-blind full-article-section sample, all six approved "
            "relation mappings pass train and held-out measurability without failures; "
            "this is document-internal code execution, not reconstruction or truth."
        ),
    }


def all_statistics() -> dict[str, Any]:
    return {
        "codability_by_domain": codability_by_domain(),
        "channel_contracts": channel_contract_summary(),
        "census_progress": census_progress(),
        "census_outcomes": census_outcome_summary(),
        "census_probe_channel_replay": census_probe_channel_replay(),
        "creative_writing_heldout": creative_writing_heldout_adjudication(),
        "ws4_depth": ws4_depth_summary(),
        "active_code_depth_retrospective": active_code_depth_retrospective(),
        "active_code_a104_supplemental": active_code_a104_supplemental(),
        "code_review_representation_family_sensitivity": (
            code_review_representation_family_sensitivity()
        ),
        "active_code_source_structure": active_code_source_structure(),
        "math_a12_relation_generalization": math_a12_relation_generalization(),
        "math_a12_pair_projection_depth": math_a12_pair_projection_depth(),
        "science_relation_witness_summary": science_relation_witness_summary(),
        "patent_ws3_family_retrospective": patent_ws3_family_retrospective(),
        "technical_evidence_ledger_summary": technical_evidence_ledger_summary(),
        "code_review_hierarchy_reconstruction_funnel": (
            code_review_hierarchy_reconstruction_funnel()
        ),
        "code_review_additive_unused_program_funnel": (
            code_review_additive_unused_program_funnel()
        ),
        "math_hierarchy_static_funnel": math_hierarchy_static_funnel(),
        "science_hierarchy_static_funnel": science_hierarchy_static_funnel(),
        "science_hierarchy_fullarticle_operational_funnel": (
            science_hierarchy_fullarticle_operational_funnel()
        ),
        "patent_hierarchy_static_funnel": patent_hierarchy_static_funnel(),
        "patent_claim_structure_hierarchy_static_funnel": (
            patent_claim_structure_hierarchy_static_funnel()
        ),
        "patent_claim_structure_hierarchy_operational_funnel": (
            patent_claim_structure_hierarchy_operational_funnel()
        ),
        "patent_claim_graph_additive_cross_audited_funnel": (
            patent_claim_graph_additive_cross_audited_funnel()
        ),
    }


if __name__ == "__main__":
    print(json.dumps(all_statistics(), indent=2, sort_keys=True))
