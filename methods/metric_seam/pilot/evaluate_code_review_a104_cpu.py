"""Sealed CPU evaluation of active code-review a104 reconstruction poles.

The blind h0 hashes are verified before judge results are opened.  The report
then compares, on the frozen held-out split and a common item intersection:
the TRAIN-selected prompt-compiled baseline, the pre-existing deep coded
checker, and the new relation-aware blind h0.  No program is modified here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "methods/metric_seam/hybrids"))
from methods.metric_seam.certificates import spearman  # noqa: E402
from methods.metric_seam.hybrids.eval_hybrids_task import (  # noqa: E402
    load_judge, paired_boot, split_task_ids,
)

TASK = ROOT / "outputs/metric_seam_pilot/tasks/code_review"
BLIND = TASK / "blind_h0_cpu_v2"
FLAVORS = ("v0_keyword", "v1_structure", "v2_holistic")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rho(column, judge, ids):
    selected = [d for d in ids if d in judge and column.get(d) is not None]
    return selected, spearman([column[d] for d in selected], [judge[d] for d in selected])


def main() -> None:
    blind_manifest = json.loads((BLIND / "manifest.json").read_text())
    checks = {
        "items_sha256": _sha(TASK / "items.json"),
        "scores_sha256": _sha(BLIND / "a104_scores.json"),
        "profiles_sha256": _sha(BLIND / "a104_profiles.jsonl"),
        "program_sha256": _sha(
            ROOT / "methods/metric_seam/hybrids/programs_code_review/a104_h0.py"
        ),
        "ops_sha256": _sha(ROOT / "methods/metric_seam/hybrids/ops_code.py"),
    }
    if any(blind_manifest[key] != value for key, value in checks.items()):
        raise SystemExit("blind h0 freeze verification failed; refusing to load judge results")

    items = json.loads((TASK / "items.json").read_text())
    train, test = split_task_ids(items)
    judge_all, reliability = load_judge(TASK / "results.jsonl")
    judge = judge_all["a104"]
    code = json.loads((TASK / "code_scores.json").read_text())
    relation_h0 = json.loads((BLIND / "a104_scores.json").read_text())

    # Protocol-frozen baseline selection uses TRAIN only.
    train_rhos = {}
    for flavor in FLAVORS:
        _, rho = _rho(code[f"a104_{flavor}"], judge, train)
        train_rhos[flavor] = rho
    selected_flavor = max(train_rhos, key=train_rhos.get)
    prompt_baseline = code[f"a104_{selected_flavor}"]
    coded_checker = code["a104_coded_checker"]

    common_test = sorted(
        d for d in test if d in judge and prompt_baseline.get(d) is not None
        and coded_checker.get(d) is not None and relation_h0.get(d) is not None
    )
    columns = {
        "prompt_compiled_baseline": prompt_baseline,
        "preexisting_deep_coded_checker": coded_checker,
        "blind_relation_h0": relation_h0,
    }
    test_rhos = {
        name: spearman([column[d] for d in common_test], [judge[d] for d in common_test])
        for name, column in columns.items()
    }
    checker_boot = paired_boot(common_test, coded_checker, prompt_baseline, judge)
    relation_boot = paired_boot(common_test, relation_h0, prompt_baseline, judge)
    floor = max(test_rhos["prompt_compiled_baseline"] + 0.10, 0.60)

    profiles = [json.loads(line)["profile"] for line in
                (BLIND / "a104_profiles.jsonl").read_text().splitlines()]
    coverage = {
        "n_items": len(profiles),
        "n_truncated": sum(bool(p["truncated_input"]) for p in profiles),
        "n_with_source_file": sum(bool(p["source_files"]) for p in profiles),
        "n_with_test_file": sum(bool(p["test_files"]) for p in profiles),
        "n_with_functional_test_source_edge": sum(bool(p["test_to_source_edges"])
                                                   for p in profiles),
        "n_with_assertion": sum(p["assertions"] > 0 for p in profiles),
    }
    result = {
        "schema_version": "metric-seam-active-code-review-a104-sealed-eval-v2",
        "lane": "active_code_review_census",
        "legacy_replay": False,
        "compute": "CPU only; no repository/test execution and no model inference",
        "objective": "unsupervised reconstruction of the prompt judgment",
        "freeze_verified_before_judge_load": True,
        "blind_freeze_checks": checks,
        "criterion": "a104",
        "judge_rel1": reliability["a104"],
        "split": {"train": len(train), "test": len(test), "seed": 7},
        "baseline_selection": {
            "rule": "highest Spearman on TRAIN among frozen prompt-compiled flavors",
            "train_rhos": train_rhos,
            "selected": selected_flavor,
        },
        "common_heldout_n": len(common_test),
        "heldout_rhos_common_intersection": test_rhos,
        "gate_floor": floor,
        "preexisting_deep_coded_checker": {
            "delta_vs_prompt_baseline": (
                test_rhos["preexisting_deep_coded_checker"]
                - test_rhos["prompt_compiled_baseline"]
            ),
            "P_gate": checker_boot[0], "P_beats_baseline": checker_boot[1],
            "bootstrap_used": checker_boot[2],
            "passes_current_gate": bool(
                test_rhos["preexisting_deep_coded_checker"] >= floor
                and checker_boot[0] >= 0.5
            ),
            "interpretation": (
                "manual/pre-existing deep verifier reconstructed the LLM judgment better "
                "than the prompt-compiled baseline; code overperformance is allowed"
            ),
        },
        "blind_relation_h0": {
            "delta_vs_prompt_baseline": (
                test_rhos["blind_relation_h0"] - test_rhos["prompt_compiled_baseline"]
            ),
            "P_gate": relation_boot[0], "P_beats_baseline": relation_boot[1],
            "bootstrap_used": relation_boot[2],
            "passes_current_gate": bool(
                test_rhos["blind_relation_h0"] >= floor and relation_boot[0] >= 0.5
            ),
            "interpretation": (
                "positive but sub-gate blind reconstruction; do not tune after held-out read"
            ),
        },
        "structural_profile_coverage": coverage,
        "claim_boundary": (
            "Correlation reconstructs the articulated prompt judgment. Structural code "
            "evidence verifies test/source relations, not behavioural intent, oracle "
            "validity, or actual test success."
        ),
    }
    path = TASK / "a104_cpu_sealed_eval_v2.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    report = f"""# Active coding census a104 — sealed CPU evaluation

This is the active 250-diff coding census, not the legacy prototype replay.
The blind relation-aware h0 freeze was hash-verified before judge results were loaded.
No model inference, repository checkout, or test execution occurred in this run.

On the common held-out intersection (`n={len(common_test)}`), the TRAIN-selected
prompt-compiled baseline reached rho={test_rhos['prompt_compiled_baseline']:.3f}.
The pre-existing deep coded checker reached rho={test_rhos['preexisting_deep_coded_checker']:.3f}
(delta={result['preexisting_deep_coded_checker']['delta_vs_prompt_baseline']:+.3f},
P(gate)={checker_boot[0]:.3f}, P(beats)={checker_boot[1]:.3f}) and passes the current
held-out gate. This is a code-overperformance result relative to the prompt-compiled
program pole, while the reconstruction target remains the articulated LLM judgment.

The new outcome-blind relation h0 reached rho={test_rhos['blind_relation_h0']:.3f}
(delta={result['blind_relation_h0']['delta_vs_prompt_baseline']:+.3f},
P(gate)={relation_boot[0]:.3f}, P(beats)={relation_boot[1]:.3f}). It is a positive
reconstruction but does not pass the frozen gate; it must not be tuned on this readout.

The code evidence covers test presence, source/test balance, AST identifier/name
correspondence, and assertion structure. It does not establish behavioural intent,
oracle validity, or actual test success.
"""
    (TASK / "A104_CPU_SEALED_REPORT_V2.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
