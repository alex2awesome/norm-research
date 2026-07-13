"""Publish the provenance-corrected active code-review a104 V3 record.

V2's numerical evaluation is reproducible, but its ``blind`` terminology was
stronger than its evidence.  The h0 was manually authored after the judge file
already existed.  Its scorer does not reference the outcome field, and the
evaluator verifies the frozen h0 before loading judge results, but authoring was
not performed inside the mechanically sealed compiler used by reconstruction-v2.

This additive correction preserves V2 byte-for-byte.  It verifies V2's recorded
hashes against the current frozen inputs, copies its numerical readout, and adds
the narrower provenance classification and data/model lineage required to quote
the result accurately.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "outputs/metric_seam_pilot/tasks/code_review"
V2_JSON = TASK / "a104_cpu_sealed_eval_v2.json"
V3_JSON = TASK / "a104_cpu_sealed_eval_v3.json"
V3_REPORT = TASK / "A104_CPU_SEALED_REPORT_V3.md"
BLIND = TASK / "blind_h0_cpu_v2"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_v2_inputs(v2: dict) -> dict[str, str]:
    checks = {
        "items_sha256": _sha(TASK / "items.json"),
        "scores_sha256": _sha(BLIND / "a104_scores.json"),
        "profiles_sha256": _sha(BLIND / "a104_profiles.jsonl"),
        "program_sha256": _sha(
            ROOT / "methods/metric_seam/hybrids/programs_code_review/a104_h0.py"
        ),
        "ops_sha256": _sha(ROOT / "methods/metric_seam/hybrids/ops_code.py"),
    }
    if checks != v2["blind_freeze_checks"]:
        raise SystemExit("V2 inputs no longer match its recorded hashes; refusing correction")
    return checks


def _verify_comparison_inputs() -> dict:
    manifest_path = TASK / "code_scores_cpu_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    code_scores_path = TASK / "code_scores.json"
    if manifest["input_items_sha256"] != _sha(TASK / "items.json"):
        raise SystemExit("CPU comparison manifest has an unexpected items hash")
    if manifest["output_sha256"] != _sha(code_scores_path):
        raise SystemExit("CPU comparison scores no longer match their manifest")

    relevant = (
        "a104_v0_keyword", "a104_v1_structure", "a104_v2_holistic",
        "a104_coded_checker",
    )
    source_receipts = {}
    for key in relevant:
        record = manifest["sources"][key]
        path = ROOT / record["path"]
        actual = _sha(path)
        if actual != record["sha256"]:
            raise SystemExit(f"comparison source no longer matches manifest: {key}")
        source_receipts[key] = record
    return {
        "verification_time_semantics": (
            "verified while publishing the additive V3 correction; V2 did not itself "
            "hash-check these comparison inputs before loading judge results"
        ),
        "code_scores_sha256": _sha(code_scores_path),
        "code_scores_cpu_manifest_sha256": _sha(manifest_path),
        "sources": source_receipts,
    }


def main() -> None:
    if V3_JSON.exists() or V3_REPORT.exists():
        raise SystemExit("refusing to overwrite existing V3 correction")

    v2 = json.loads(V2_JSON.read_text())
    checks = _verify_v2_inputs(v2)
    comparison_receipt = _verify_comparison_inputs()
    items = json.loads((TASK / "items.json").read_text())
    profile_rows = [
        json.loads(line)["profile"]
        for line in (BLIND / "a104_profiles.jsonl").read_text().splitlines()
    ]
    edge_keys = [
        (edge["source"], edge["test"], tuple(edge["evidence"]))
        for profile in profile_rows
        for edge in profile["test_to_source_edges"]
    ]
    v3 = dict(v2)
    retrospective_h0 = dict(v3.pop("blind_relation_h0"))
    retrospective_h0["interpretation"] = (
        "positive but sub-gate retrospective manual/mock reconstruction; "
        "do not tune after held-out read"
    )
    heldout_rhos = dict(v3["heldout_rhos_common_intersection"])
    heldout_rhos["retrospective_relation_h0"] = heldout_rhos.pop("blind_relation_h0")
    v3["heldout_rhos_common_intersection"] = heldout_rhos
    v3["retrospective_relation_h0"] = retrospective_h0
    v3["h0_freeze_checks"] = v3.pop("blind_freeze_checks")
    v3.update({
        "schema_version": "metric-seam-active-code-review-a104-sealed-eval-v3",
        "supersedes": "a104_cpu_sealed_eval_v2.json",
        "correction_scope": "provenance terminology only; numerical results unchanged",
        "h0_discovery_provenance": "manual_mock_retrospective_seed",
        "h0_authoring_certification": "not_mechanically_blind",
        "freeze_verified_before_judge_load": True,
        "h0_freeze_checks": checks,
        "execution_blindness": {
            "classification": "label_unreferenced_not_label_inaccessible",
            "scorer_fields_indexed": ["datapoint_id", "ctext"],
            "serialized_input_contains_merge_judgement": True,
            "merge_judgement_referenced_by_scorer": False,
            "judge_results_loaded_by_h0_builder": False,
            "limitation": (
                "The builder deserializes items.json, which contains judgement, before "
                "projecting rows to datapoint_id and ctext. The manual authoring process "
                "also lacked a sealed transcript. Therefore label-unreferenced execution "
                "is verified, but label-inaccessible authoring is not."
            ),
        },
        "data_provenance": {
            "active_item_representation": "head/tail-canonicalized unified PR diff (ctext)",
            "source_diff_directory": "datasets/code-review/pr_test_execution/batch_runs",
            "uses_legacy_f2p_mock_program_or_output": False,
            "uses_prior_test_execution_telemetry_or_test_outcome": False,
            "repository_checkout_or_test_execution_in_this_run": False,
            "qualification": (
                "The raw diff files reside under the pr_test_execution corpus. This run "
                "does not replay its test-execution pipeline or consume its execution "
                "telemetry; corpus residence must not be described as total independence "
                "from pr_test_execution."
            ),
        },
        "model_provenance": {
            "model_or_gpu_inference_in_this_cpu_run": False,
            "reference_judgement": "pre-existing two-pass model-produced results.jsonl",
            "prompt_compiled_baselines": (
                "pre-existing frozen Claude-produced text-to-score programs"
            ),
            "deep_coded_checker": (
                "pre-existing coding A-bank static/AST checker; not newly discovered h0"
            ),
            "new_relation_h0": (
                "manually authored retrospective seed over new CodeOps relation library"
            ),
        },
        "comparison_inputs_verified_at_correction": comparison_receipt,
        "correction_input_receipt": {
            "v2_json_sha256": _sha(V2_JSON),
            "v2_report_sha256": _sha(TASK / "A104_CPU_SEALED_REPORT_V2.md"),
        },
        "structural_profile_audit": {
            "n_items": len(profile_rows),
            "n_ctext_head_tail_truncated": sum(
                "\n[...]\n" in row["ctext"] for row in items
            ),
            "n_items_with_source_file": sum(bool(p["source_files"]) for p in profile_rows),
            "n_items_with_test_file": sum(bool(p["test_files"]) for p in profile_rows),
            "n_items_with_ast_edge": sum(
                bool(p["test_to_source_edges"]) for p in profile_rows
            ),
            "n_ast_edges_raw": len(edge_keys),
            "n_ast_edges_unique_qualified_triples": len(set(edge_keys)),
            "n_items_with_assertion": sum(p["assertions"] > 0 for p in profile_rows),
            "n_assertions": sum(p["assertions"] for p in profile_rows),
            "qualified_name_collision_limitation": (
                "CodeSymbol qualified_name is path::name and omits lexical class/receiver "
                "scope. Three of 44 emitted edges duplicate a qualified triple. The "
                "frozen h0 remains unchanged after held-out evaluation; future untouched "
                "criteria should use scope-aware symbol identities."
            ),
            "runtime_limitation": (
                "AST evidence requires the tree-sitter core and language grammar packages. "
                "The library degrades to empty AST evidence when they are unavailable, so "
                "future manifests should freeze dependency versions as well as source hashes."
            ),
        },
        "objective": "unsupervised reconstruction of the articulated prompt judgment",
    })
    V3_JSON.write_text(json.dumps(v3, indent=2, sort_keys=True) + "\n")

    rhos = v3["heldout_rhos_common_intersection"]
    deep = v3["preexisting_deep_coded_checker"]
    h0 = v3["retrospective_relation_h0"]
    report = f"""# Active coding census a104 — provenance-corrected sealed CPU evaluation (V3)

## Correction notice

This record supersedes the framing, but not the numbers, in
`A104_CPU_SEALED_REPORT_V2.md` and `a104_cpu_sealed_eval_v2.json`. All V2
hashes and numerical results reproduce. V2 called the new relation h0
"outcome-blind"; that was too strong for a manually authored program whose
source was created after the judge file existed. The accurate classification
is **manual/mock retrospective seed with label-unreferenced execution**. Its
authoring was not mechanically label-inaccessible.

The evaluator does provide a narrower, real seal: it hash-verifies the h0
program, operations, profiles, scores, and items before it loads the articulated
LLM judgment. The h0 builder never references `judgement` or `results.jsonl`,
but it deserializes `items.json`, which physically contains a merge-judgement
field, and then projects each row to `datapoint_id` and `ctext`.

## Three distinct program poles

1. The **prompt-compiled baseline** is a frozen, pre-existing Claude-produced
   text-to-score program, selected among three frozen flavors using TRAIN only.
2. The **pre-existing deep coded checker** is the coding A-bank's static/AST
   checker. It was not discovered by the new h0 process.
3. The **new relation h0** is the retrospective manual/mock seed over `CodeOps`.
   It is useful evidence for the proposed relation decomposition, but not a
   blind-compiler discovery claim.

On the common held-out intersection (`n={v3['common_heldout_n']}`), the
prompt-compiled baseline reached rho={rhos['prompt_compiled_baseline']:.3f}.
The pre-existing deep checker reached rho={rhos['preexisting_deep_coded_checker']:.3f}
(delta={deep['delta_vs_prompt_baseline']:+.3f}, P(gate)={deep['P_gate']:.3f},
P(beats)={deep['P_beats_baseline']:.3f}) and passes the current gate. This is
code overperformance relative to the prompt-compiled program pole while the
reconstruction target remains the articulated LLM judgment.

The retrospective relation h0 reached rho={rhos['retrospective_relation_h0']:.3f}
(delta={h0['delta_vs_prompt_baseline']:+.3f}, P(gate)={h0['P_gate']:.3f},
P(beats)={h0['P_beats_baseline']:.3f}). It is a positive, sub-gate
reconstruction and must not be tuned on this held-out readout.

## Data and compute provenance

The active inputs are head/tail-canonicalized unified PR diffs. Their raw diff
files reside under `datasets/code-review/pr_test_execution/batch_runs`; this is
therefore not total corpus independence from `pr_test_execution`. No legacy
`f2p_mock` program/output, test-execution telemetry, or per-PR test outcome was
used by these scorers, and no repository checkout or test execution occurred.

No GPU or model inference occurred in this CPU run. The evaluator necessarily
uses a pre-existing model-produced `results.jsonl` as its reconstruction target,
and the prompt-compiled baselines are frozen model-produced artifacts. "No
inference in this run" must not be expanded into "no model artifact was used."

At correction time, V3 also verified `code_scores.json`, its CPU manifest, and
the four a104 comparison-program hashes. This confirms the current comparison
inputs. V2 itself did not hash-check those baseline inputs before opening the
judge, so this is a correction-time integrity receipt, not a retroactive claim
about V2's evaluator ordering.

The code evidence covers test presence, source/test balance, AST identifier/name
correspondence, and assertion structure. It does not establish behavioural
intent, oracle validity, or actual test success.

The frozen profile contains 168/250 head/tail-truncated `ctext` inputs, 209
items with a source file, 159 with a test file, 18 with at least one AST
test-to-source edge, 44 raw edges (41 unique qualified triples), and 33 items
with at least one assertion (114 assertions total). Three duplicate edge triples
expose a bounded identity limitation: `path::name` omits class/receiver scope.
This frozen h0 is not repaired after the held-out read; the next untouched code
criterion should use scope-aware symbol identities. AST extraction also depends
on installed tree-sitter grammars, which future manifests should version-freeze.
"""
    V3_REPORT.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
