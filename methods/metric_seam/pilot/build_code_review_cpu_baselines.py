"""Build outcome-blind CPU baselines for the active ``code_review`` census.

This is not a legacy execution replay.  It runs frozen text-to-score programs
and the existing coding A-bank checkers on the active 250 ``ctext`` diffs.
Only ``datapoint_id`` and ``ctext`` are read from ``items.json``; merge labels
and judge results are neither loaded nor used.  The output has the standard
``code_scores.json`` flavor keys needed by the task-generic h0 harness, plus
``*_coded_checker`` keys that preserve the deeper AST/static checker pole.

CPU only; no model runner, repository checkout, tests, network, or GPU.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "outputs/metric_seam_pilot/tasks/code_review"
CODEGEN = ROOT / "runs/validity_full/v2/code_review/codegen_claude"
METRICS = ROOT / "methods/existing_metrics_runner/coded/metrics"
FLAVORS = ("v0_keyword", "v1_structure", "v2_holistic")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_file(path: Path):
    spec = importlib.util.spec_from_file_location(f"active_cr_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metric_module(aid: str) -> tuple[str, Path] | None:
    matches = sorted(METRICS.glob(f"{aid}_*.py"))
    if len(matches) != 1:
        return None
    path = matches[0]
    return f"methods.existing_metrics_runner.coded.metrics.{path.stem}", path


def _safe_score(module, text: str, check_applies: bool = False):
    try:
        if check_applies and hasattr(module, "applies") and not module.applies(text):
            return None
        value = module.score(text)
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def main() -> None:
    out_path = TASK / "code_scores.json"
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {out_path}")

    # Deliberately project away the supervised field before any programs run.
    raw_items = json.loads((TASK / "items.json").read_text())
    items = [(row["datapoint_id"], row["ctext"]) for row in raw_items]
    aspect_ids = json.loads((TASK / "aspects_used.json").read_text())
    scores: dict[str, dict[str, float | None]] = {}
    sources: dict[str, dict] = {}

    for aid in aspect_ids:
        for flavor in FLAVORS:
            path = CODEGEN / f"{aid}_{flavor}.py"
            if not path.exists():
                continue
            module = _load_file(path)
            key = f"{aid}_{flavor}"
            scores[key] = {dpid: _safe_score(module, text) for dpid, text in items}
            sources[key] = {"kind": "frozen_text_program", "path": str(path.relative_to(ROOT)),
                            "sha256": _sha(path)}

        metric = _metric_module(aid)
        if metric is not None:
            module_name, path = metric
            module = importlib.import_module(module_name)
            key = f"{aid}_coded_checker"
            scores[key] = {
                dpid: _safe_score(module, text, check_applies=True)
                for dpid, text in items
            }
            sources[key] = {"kind": "coding_a_bank_checker", "path": str(path.relative_to(ROOT)),
                            "sha256": _sha(path)}

    out_path.write_text(json.dumps(scores, indent=1, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "metric-seam-active-code-review-cpu-baselines-v1",
        "lane": "active_code_review_census",
        "legacy_replay": False,
        "execution": "CPU-only score functions over frozen ctext; no repository/test execution",
        "design_scope": "outcome_blind",
        "input_fields_read": ["datapoint_id", "ctext"],
        "input_items_sha256": _sha(TASK / "items.json"),
        "n_items": len(items),
        "n_aspects": len(aspect_ids),
        "n_score_columns": len(scores),
        "missing_text_flavors": [
            f"{aid}_{flavor}" for aid in aspect_ids for flavor in FLAVORS
            if not (CODEGEN / f"{aid}_{flavor}.py").exists()
        ],
        "sources": sources,
        "output_sha256": _sha(out_path),
    }
    (TASK / "code_scores_cpu_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({k: manifest[k] for k in (
        "lane", "n_items", "n_aspects", "n_score_columns", "missing_text_flavors"
    )}, indent=2))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()

