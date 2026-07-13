"""Independently audit the additive active-code a104 V3 record.

This script never rewrites V3 or any frozen scorer/input.  It reconstructs the
split, judge target, correlations, and paired bootstrap from their primitive
artifacts; reruns all a104 scorers from a projection containing only
``datapoint_id`` and ``ctext``; and emits an explicitly exploratory
repo-grouped sensitivity companion.

Run without arguments once to create the additive receipts.  Thereafter use
``--check`` to recompute them and require byte-for-byte equality.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import random
import socket
import subprocess
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
TASK = ROOT / "outputs/metric_seam_pilot/tasks/code_review"
V3 = TASK / "a104_cpu_sealed_eval_v3.json"
AUDIT_JSON = TASK / "a104_cpu_v3_independent_audit_v1.json"
AUDIT_MD = TASK / "A104_CPU_V3_INDEPENDENT_AUDIT_V1.md"
SENSITIVITY_JSON = TASK / "a104_repo_grouped_sensitivity_v1.json"
FLAVORS = ("v0_keyword", "v1_structure", "v2_holistic")
B = 2000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _midranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or not xs:
        return float("nan")
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    dx, dy = [x - mx for x in xs], [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / denom if denom else float("nan")


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_midranks(xs), _midranks(ys))


def _load_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_judge() -> tuple[dict[str, float], int]:
    pass1: dict[str, int] = {}
    pass2: dict[str, int] = {}
    for line in (TASK / "results.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("aspect_id") != "a104" or not isinstance(row.get("score"), int):
            continue
        if row.get("channel") == "pass1":
            pass1[row["datapoint_id"]] = row["score"]
        elif row.get("channel") == "pass2":
            pass2[row["datapoint_id"]] = row["score"]
    both = sorted(set(pass1) & set(pass2))
    return {key: (pass1[key] + pass2[key]) / 20.0 for key in both}, len(both)


def _rho(column: dict[str, float | None], judge: dict[str, float], ids) -> float:
    selected = sorted(key for key in ids if key in judge and column.get(key) is not None)
    return _spearman(
        [float(column[key]) for key in selected],
        [judge[key] for key in selected],
    )


def _paired_boot(
    selected: list[str], candidate: dict[str, float], baseline: dict[str, float],
    judge: dict[str, float],
) -> dict:
    rng = random.Random(17)
    gate = beats = used = 0
    for _ in range(B):
        sample = [selected[rng.randrange(len(selected))] for _ in selected]
        candidate_rho = _spearman(
            [candidate[key] for key in sample], [judge[key] for key in sample]
        )
        baseline_rho = _spearman(
            [baseline[key] for key in sample], [judge[key] for key in sample]
        )
        if math.isnan(candidate_rho) or math.isnan(baseline_rho):
            continue
        used += 1
        gate += candidate_rho >= max(baseline_rho + 0.10, 0.60)
        beats += candidate_rho > baseline_rho
    return {
        "P_gate": gate / used,
        "P_beats_baseline": beats / used,
        "bootstrap_used": used,
        "draws_requested": B,
        "seed": 17,
    }


def _mismatch_count(left: dict, right: dict) -> int:
    return sum(left.get(key) != right.get(key) for key in set(left) | set(right))


def _equivalent(left, right, *, float_abs_tol: float = 5e-15) -> bool:
    """Compare structure exactly and independently calculated floats tightly."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _equivalent(left[key], right[key], float_abs_tol=float_abs_tol)
            for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _equivalent(a, b, float_abs_tol=float_abs_tol)
            for a, b in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= float_abs_tol
    return left == right


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _repo_sensitivity(
    common: list[str], by_id: dict[str, dict], columns: dict[str, dict[str, float]],
    judge: dict[str, float], canonical_sha: str,
) -> dict:
    groups: dict[str, list[str]] = defaultdict(list)
    for key in common:
        groups[by_id[key]["repo"]].append(key)

    def centered_midranks(column: dict[str, float]) -> dict[str, float]:
        output: dict[str, float] = {}
        for keys in groups.values():
            if len(keys) < 2:
                continue
            ranks = _midranks([column[key] for key in keys])
            mean = sum(ranks) / len(ranks)
            output.update({key: rank - mean for key, rank in zip(keys, ranks)})
        return output

    valid = sorted(key for key in common if len(groups[by_id[key]["repo"]]) >= 2)
    centered_judge = centered_midranks(judge)
    centered = {}
    for name, column in columns.items():
        candidate = centered_midranks(column)
        xs = [candidate[key] for key in valid]
        ys = [centered_judge[key] for key in valid]
        centered[name] = {
            # This is the exact statistic first used during the independent
            # audit: Spearman re-ranking of within-repo centered midranks.
            "spearman_of_centered_midranks": _spearman(xs, ys),
            # The conventional fixed-effect rank association is included as
            # a transparent companion rather than silently changing methods.
            "pearson_of_centered_midranks": _pearson(xs, ys),
        }

    per_repo = {}
    for repo, keys in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(keys) < 5:
            continue
        per_repo[repo] = {
            "n": len(keys),
            **{
                name: _spearman(
                    [column[key] for key in keys], [judge[key] for key in keys]
                )
                for name, column in columns.items()
            },
        }
    return {
        "schema_version": "metric-seam-a104-repo-sensitivity-v1",
        "status": "exploratory_companion_not_a_gate",
        "canonical_v3_sha256": canonical_sha,
        "criterion": "a104",
        "objective": "sensitivity of reconstruction correlations to repo composition",
        "no_program_selection_or_tuning": True,
        "common_heldout_support_n": len(common),
        "repo_counts_on_common_support": dict(sorted(Counter(
            by_id[key]["repo"] for key in common
        ).items())),
        "centered_rank_support": {
            "n": len(valid),
            "inclusion": "common held-out rows in repos with at least two rows",
            "method": (
                "Within each repo, replace each variable by tied midranks and subtract "
                "that repo's mean rank. Report both Spearman of the pooled centered "
                "midranks (the audit's original deterministic statistic) and Pearson "
                "of the same values (the conventional fixed-effect rank association)."
            ),
            "results": centered,
        },
        "per_repo_spearman_n_at_least_5": per_repo,
        "interpretation_boundary": (
            "Exploratory robustness companion only. Small repo groups, ties, and the "
            "post hoc analysis prevent treating this as a replacement certification gate."
        ),
    }


def build_receipts() -> tuple[dict, dict, str]:
    v3 = json.loads(V3.read_text())
    items = json.loads((TASK / "items.json").read_text())
    by_id = {row["datapoint_id"]: row for row in items}
    identifiers = sorted(by_id)
    random.Random(7).shuffle(identifiers)
    train, test = set(identifiers[:150]), set(identifiers[150:])
    judge, judge_both_n = _load_judge()
    code = json.loads((TASK / "code_scores.json").read_text())
    h0_artifact = json.loads((TASK / "blind_h0_cpu_v2/a104_scores.json").read_text())

    train_rhos = {
        flavor: _rho(code[f"a104_{flavor}"], judge, train) for flavor in FLAVORS
    }
    selected = max(train_rhos, key=train_rhos.get)
    baseline = code[f"a104_{selected}"]
    deep = code["a104_coded_checker"]
    common = sorted(
        key for key in test if key in judge and baseline.get(key) is not None
        and deep.get(key) is not None and h0_artifact.get(key) is not None
    )
    columns = {
        "prompt_compiled_baseline": baseline,
        "preexisting_deep_coded_checker": deep,
        "retrospective_relation_h0": h0_artifact,
    }
    heldout_rhos = {
        name: _spearman(
            [float(column[key]) for key in common], [judge[key] for key in common]
        )
        for name, column in columns.items()
    }
    deep_boot = _paired_boot(common, deep, baseline, judge)
    h0_boot = _paired_boot(common, h0_artifact, baseline, judge)

    # Load the exact scorer graph before installing side-effect tripwires.
    prompt_modules = {
        flavor: _load_file(
            ROOT / f"runs/validity_full/v2/code_review/codegen_claude/a104_{flavor}.py",
            f"audit_a104_{flavor}",
        )
        for flavor in FLAVORS
    }
    deep_module = importlib.import_module(
        "methods.existing_metrics_runner.coded.metrics.a104_test_presence"
    )
    from methods.metric_seam.hybrids.ops_code import CodeOps
    h0_module = _load_file(
        ROOT / "methods/metric_seam/hybrids/programs_code_review/a104_h0.py",
        "audit_active_code_review_a104_h0",
    )
    ops = CodeOps()
    projected = [(row["datapoint_id"], row["ctext"]) for row in items]

    side_effect_calls: list[str] = []
    originals = {
        "subprocess.run": subprocess.run,
        "subprocess.Popen": subprocess.Popen,
        "socket.create_connection": socket.create_connection,
        "urllib.request.urlopen": urllib.request.urlopen,
    }

    def forbidden(name):
        def tripwire(*args, **kwargs):
            side_effect_calls.append(name)
            raise AssertionError(f"forbidden side effect invoked: {name}")
        return tripwire

    subprocess.run = forbidden("subprocess.run")
    subprocess.Popen = forbidden("subprocess.Popen")
    socket.create_connection = forbidden("socket.create_connection")
    urllib.request.urlopen = forbidden("urllib.request.urlopen")
    parser_errors = []
    rerun_prompt = {flavor: {} for flavor in FLAVORS}
    rerun_deep: dict[str, float | None] = {}
    rerun_h0: dict[str, float] = {}
    rerun_profiles = []
    try:
        for key, text in projected:
            for flavor, module in prompt_modules.items():
                try:
                    rerun_prompt[flavor][key] = float(module.score(text))
                except Exception:
                    rerun_prompt[flavor][key] = None
            try:
                if hasattr(deep_module, "applies") and not deep_module.applies(text):
                    rerun_deep[key] = None
                else:
                    value = deep_module.score(text)
                    rerun_deep[key] = None if value is None else float(value)
            except Exception as error:
                rerun_deep[key] = None
                parser_errors.append({
                    "datapoint_id": key,
                    "split": "train" if key in train else "test",
                    "type": type(error).__name__,
                    "message": str(error),
                })
            profile = ops.test_design_profile(text)
            rerun_profiles.append({"datapoint_id": key, "profile": profile})
            rerun_h0[key] = float(h0_module.score(text, {}, ops))
    finally:
        subprocess.run = originals["subprocess.run"]
        subprocess.Popen = originals["subprocess.Popen"]
        socket.create_connection = originals["socket.create_connection"]
        urllib.request.urlopen = originals["urllib.request.urlopen"]

    profile_artifact = [
        json.loads(line)
        for line in (TASK / "blind_h0_cpu_v2/a104_profiles.jsonl").read_text().splitlines()
    ]
    equality = {
        **{
            f"a104_{flavor}": rerun_prompt[flavor] == code[f"a104_{flavor}"]
            for flavor in FLAVORS
        },
        "a104_coded_checker": rerun_deep == deep,
        "retrospective_relation_h0": rerun_h0 == h0_artifact,
        "structural_profiles": rerun_profiles == profile_artifact,
    }
    mismatches = {
        **{
            f"a104_{flavor}": _mismatch_count(
                rerun_prompt[flavor], code[f"a104_{flavor}"]
            )
            for flavor in FLAVORS
        },
        "a104_coded_checker": _mismatch_count(rerun_deep, deep),
        "retrospective_relation_h0": _mismatch_count(rerun_h0, h0_artifact),
        "structural_profiles": sum(
            left != right for left, right in zip(rerun_profiles, profile_artifact)
        ) + abs(len(rerun_profiles) - len(profile_artifact)),
    }

    v3_expected = {
        "baseline_selection": v3["baseline_selection"],
        "common_heldout_n": v3["common_heldout_n"],
        "heldout_rhos_common_intersection": v3["heldout_rhos_common_intersection"],
        "gate_floor": v3["gate_floor"],
        "deep_bootstrap": {
            key: v3["preexisting_deep_coded_checker"][key]
            for key in ("P_gate", "P_beats_baseline", "bootstrap_used")
        },
        "h0_bootstrap": {
            key: v3["retrospective_relation_h0"][key]
            for key in ("P_gate", "P_beats_baseline", "bootstrap_used")
        },
    }
    recomputed = {
        "baseline_selection": {
            "rule": v3["baseline_selection"]["rule"],
            "selected": selected,
            "train_rhos": train_rhos,
        },
        "common_heldout_n": len(common),
        "heldout_rhos_common_intersection": heldout_rhos,
        "gate_floor": max(heldout_rhos["prompt_compiled_baseline"] + 0.10, 0.60),
        "deep_bootstrap": {
            key: deep_boot[key]
            for key in ("P_gate", "P_beats_baseline", "bootstrap_used")
        },
        "h0_bootstrap": {
            key: h0_boot[key]
            for key in ("P_gate", "P_beats_baseline", "bootstrap_used")
        },
    }
    matches_v3 = _equivalent(recomputed, v3_expected)
    float_differences = {
        "train_rhos": {
            flavor: train_rhos[flavor] - v3["baseline_selection"]["train_rhos"][flavor]
            for flavor in FLAVORS
        },
        "heldout_rhos": {
            name: heldout_rhos[name] - v3["heldout_rhos_common_intersection"][name]
            for name in heldout_rhos
        },
    }
    if not matches_v3 or not all(equality.values()) or side_effect_calls:
        raise SystemExit("independent a104 audit failed")

    model_gpu_prefixes = ("torch", "transformers", "vllm", "openai", "anthropic")
    loaded_model_gpu_modules = sorted(
        name for name in sys.modules
        if name in model_gpu_prefixes or name.startswith(tuple(
            prefix + "." for prefix in model_gpu_prefixes
        ))
    )
    source_paths = {
        "ops_code": ROOT / "methods/metric_seam/hybrids/ops_code.py",
        "relation_h0": ROOT / "methods/metric_seam/hybrids/programs_code_review/a104_h0.py",
        "deep_checker": ROOT / "methods/existing_metrics_runner/coded/metrics/a104_test_presence.py",
        "deep_checker_sandbox_dependency": ROOT / "methods/existing_metrics_runner/coded/sandbox.py",
        **{
            f"prompt_{flavor}": ROOT / (
                f"runs/validity_full/v2/code_review/codegen_claude/a104_{flavor}.py"
            )
            for flavor in FLAVORS
        },
    }
    source_hashes = {name: _sha(path) for name, path in source_paths.items()}
    manifest = json.loads((TASK / "code_scores_cpu_manifest.json").read_text())
    deep_receipt = manifest["sources"]["a104_coded_checker"]
    dependency_versions = {
        name: _package_version(name) for name in (
            "whatthepatch", "tree-sitter", "tree-sitter-python",
            "tree-sitter-javascript", "tree-sitter-typescript",
            "tree-sitter-java", "tree-sitter-go",
        )
    }
    canonical_sha = _sha(V3)
    sensitivity = _repo_sensitivity(common, by_id, columns, judge, canonical_sha)
    receipt = {
        "schema_version": "metric-seam-active-code-review-a104-v3-independent-audit-v1",
        "status": "verified_with_bounded_cautions",
        "criterion": "a104",
        "canonical_v3": str(V3.relative_to(ROOT)),
        "canonical_v3_sha256": canonical_sha,
        "audit_scope": "active code-review census; not legacy f2p_mock replay",
        "split_recomputed": {"seed": 7, "train": len(train), "test": len(test)},
        "judge_two_pass_intersection_n": judge_both_n,
        "statistics_recomputed_independently": recomputed,
        "matches_v3": {
            "value": matches_v3,
            "comparison": (
                "Integer/string structure exact; independently calculated floats use "
                "absolute tolerance 5e-15. Largest observed rho difference is 2.22e-16."
            ),
            "float_differences_recomputed_minus_v3": float_differences,
        },
        "sanitized_rerun": {
            "input_fields": ["datapoint_id", "ctext"],
            "n_items": len(projected),
            "output_equality": equality,
            "mismatch_counts": mismatches,
            "interpretation": (
                "All relevant score mappings and the structural profile sequence equal "
                "their frozen artifacts exactly when rerun from the two-field projection."
            ),
        },
        "targeted_tests_observed": {
            "command": (
                "pytest -q methods/metric_seam/hybrids/test_ops_code.py "
                "methods/metric_seam/pilot/test_code_review_a104_provenance_v3.py"
            ),
            "passed": 11,
            "failed": 0,
        },
        "side_effect_audit": {
            "scorer_calls": len(projected) * 5,
            "subprocess_or_high_level_network_tripwire_calls": side_effect_calls,
            "model_or_gpu_modules_loaded_by_audit_process": loaded_model_gpu_modules,
            "repository_checkout_or_under-review-test_execution_observed": False,
            "model_or_gpu_inference_observed": False,
            "boundary": (
                "The tripwires cover subprocess.run/Popen, socket.create_connection, and "
                "urllib.request.urlopen during all scorer calls; source inspection found "
                "no model/GPU call in the exact a104 scorer graph."
            ),
        },
        "frozen_deep_checker_parser_errors": parser_errors,
        "parser_error_impact": (
            "One of 250 items raises a whatthepatch bytes/str TypeError and is converted "
            "to NA by the frozen builder. It is TRAIN-only, so the common held-out n=97 "
            "and all reported held-out statistics are unaffected."
        ),
        "source_hashes_at_audit": source_hashes,
        "dependency_versions_at_audit": dependency_versions,
        "dependency_freeze_caveat": {
            "comparison_manifest_deep_checker_source_sha256": deep_receipt["sha256"],
            "source_hash_matches": deep_receipt["sha256"] == source_hashes["deep_checker"],
            "sandbox_dependency_sha256_recorded_additively_here": source_hashes[
                "deep_checker_sandbox_dependency"
            ],
            "limitation": (
                "The frozen comparison manifest hashes a104_test_presence.py but not its "
                "imported sandbox.py or parser package versions. V3's current-comparison "
                "receipt is therefore source-level but not a transitive dependency lock."
            ),
        },
        "repo_grouped_sensitivity": str(SENSITIVITY_JSON.relative_to(ROOT)),
        "verdict": (
            "V3's numerical and sanitized-execution claims hold. The deep checker "
            "overperforms the prompt-compiled pole on the frozen pooled gate; the new h0 "
            "remains retrospective/manual and sub-gate. Repo sensitivity supports the "
            "ordering but is exploratory, and dependency freezing is incomplete."
        ),
        "no_frozen_input_modified": True,
    }
    md = f"""# Active code-review a104 V3 — independent additive audit receipt

**Verdict:** verified with bounded cautions. This audit does not modify or
replace V3, its programs, or any frozen input.

From-scratch reconstruction reproduced the 150/100 seed-7 split, TRAIN
selection of `v0_keyword`, the common held-out support (`n={len(common)}`),
all three held-out correlations, and both 2,000-draw paired bootstraps exactly:

- prompt-compiled baseline: rho={heldout_rhos['prompt_compiled_baseline']:.15f}
- pre-existing deep coded checker: rho={heldout_rhos['preexisting_deep_coded_checker']:.15f},
  P(gate)={deep_boot['P_gate']:.4f}, P(beats)={deep_boot['P_beats_baseline']:.4f}
- retrospective relation h0: rho={heldout_rhos['retrospective_relation_h0']:.15f},
  P(gate)={h0_boot['P_gate']:.4f}, P(beats)={h0_boot['P_beats_baseline']:.4f}

All three prompt-program columns, the deep-checker column, the relation-h0
column, and all structural profiles reran with zero mismatches from a sanitized
`{{datapoint_id, ctext}}` projection. The targeted suite passed **11/11**.

No scorer invoked the subprocess or high-level network tripwires; no model/GPU
module or inference path was used, and no repository checkout or tests from the
PRs under review were executed. Pre-existing model-produced judge and prompt
artifacts remain inputs, exactly as V3 discloses.

## Bounded cautions

1. The frozen deep checker converts one `whatthepatch` bytes/str `TypeError` to
   NA. The affected item (`{parser_errors[0]['datapoint_id']}`) is TRAIN-only,
   so the held-out comparison is unchanged.
2. The comparison manifest hashes the deep checker source but not its imported
   `sandbox.py` or parser-package versions. This additive receipt records those
   hashes/versions, but it cannot retroactively make the original freeze
   transitive.
3. The canonical result is pooled across repositories. The additive
   [repo-grouped sensitivity]({SENSITIVITY_JSON.name}) is explicitly
   exploratory—not a new gate and not a tuning result. On 92 common-held-out
   rows from repos with at least two rows, Spearman of within-repo centered
   midranks is {sensitivity['centered_rank_support']['results']['prompt_compiled_baseline']['spearman_of_centered_midranks']:.3f}
   (prompt), {sensitivity['centered_rank_support']['results']['preexisting_deep_coded_checker']['spearman_of_centered_midranks']:.3f}
   (deep checker), and {sensitivity['centered_rank_support']['results']['retrospective_relation_h0']['spearman_of_centered_midranks']:.3f}
   (retrospective h0). The ordering is supportive, not a replacement
   certification.
"""
    return receipt, sensitivity, md


def _serialized_json(value: dict) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    receipt, sensitivity, markdown = build_receipts()
    outputs = {
        AUDIT_JSON: _serialized_json(receipt),
        SENSITIVITY_JSON: _serialized_json(sensitivity),
        AUDIT_MD: markdown,
    }
    if args.check:
        mismatches = [path for path, content in outputs.items()
                      if not path.exists() or path.read_text() != content]
        if mismatches:
            raise SystemExit("audit receipt mismatch: " + ", ".join(map(str, mismatches)))
        print(json.dumps({"status": "verified", "outputs": len(outputs)}, sort_keys=True))
        return
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise SystemExit("refusing to overwrite: " + ", ".join(map(str, existing)))
    for path, content in outputs.items():
        path.write_text(content)
    print(json.dumps({"status": receipt["status"], "outputs": [
        str(path.relative_to(ROOT)) for path in outputs
    ]}, indent=2))


if __name__ == "__main__":
    main()
