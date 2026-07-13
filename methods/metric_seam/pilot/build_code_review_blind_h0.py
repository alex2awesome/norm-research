"""Freeze the active a104 h0 scores/profiles before outcome evaluation.

Reads no merge labels or judge results.  The generated artifact records the
structural evidence available from each active census ``ctext`` diff, making
later comparison to the prompt judgment a sealed-evaluator step rather than a
program-design input.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "outputs/metric_seam_pilot/tasks/code_review"
OUT = TASK / "blind_h0_cpu_v2"
OPS_SOURCE = ROOT / "methods/metric_seam/hybrids/ops_code.py"
PROGRAM_SOURCE = ROOT / "methods/metric_seam/hybrids/programs_code_review/a104_h0.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _program():
    spec = importlib.util.spec_from_file_location("active_code_review_a104_h0", PROGRAM_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(PROGRAM_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from methods.metric_seam.hybrids.ops_code import CodeOps

    if OUT.exists():
        raise SystemExit(f"refusing to overwrite frozen output {OUT}")
    OUT.mkdir(parents=True)
    raw_items = json.loads((TASK / "items.json").read_text())
    items = [(row["datapoint_id"], row["ctext"]) for row in raw_items]
    program, ops = _program(), CodeOps()
    scores: dict[str, float] = {}
    profile_path = OUT / "a104_profiles.jsonl"
    with profile_path.open("w") as handle:
        for dpid, text in items:
            profile = ops.test_design_profile(text)
            scores[dpid] = program.score(text, {}, ops)
            handle.write(json.dumps({"datapoint_id": dpid, "profile": profile},
                                    sort_keys=True) + "\n")
    scores_path = OUT / "a104_scores.json"
    scores_path.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "metric-seam-active-code-review-blind-h0-v2",
        "lane": "active_code_review_census",
        "legacy_replay": False,
        "criterion": "a104",
        "design_scope": "outcome_blind_unsupervised_reconstruction",
        "input_fields_read": ["datapoint_id", "ctext"],
        "n_items": len(items),
        "program_sha256": _sha(PROGRAM_SOURCE),
        "ops_sha256": _sha(OPS_SOURCE),
        "items_sha256": _sha(TASK / "items.json"),
        "scores_sha256": _sha(scores_path),
        "profiles_sha256": _sha(profile_path),
        "subrelations": program.SUBRELATIONS,
        "claim_boundary": (
            "Code verifies structural test relations; it does not establish behavioural "
            "intent, oracle validity, or executed test success."
        ),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"lane": manifest["lane"], "criterion": "a104",
                      "n_items": len(items), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
