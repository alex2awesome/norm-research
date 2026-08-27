"""Multiplicity-aware retrospective of the four patent WS3 evidence operations.

WS3 already compared each evidence-aware hybrid with the same program under a
null prior-art operation.  This additive CPU analysis retains all four selected
criteria, reconstructs the exact seed-7 held-out score maps, and applies paired
randomization, paired bootstrap intervals, and BH-FDR as one fixed descriptive
family.

The frozen two-pass evidence-aware LLM judgment is the unsupervised reference.
No ``items.json.judgement`` value is read.  This is retrospective: criteria,
programs, and aggregate results were already visible, so multiplicity control
narrows the historical evidence but does not create a confirmatory certificate.
Examiner-cited prior art was force-included upstream; all results remain
oracle-conditioned and say nothing about autonomous retrieval or patent truth.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import random
from collections import Counter
from typing import Any, Mapping

from methods.metric_seam.battery.certify_batch_v2 import (
    benjamini_hochberg,
    paired_bootstrap_ci,
    paired_randomization_test,
    spearman,
)
from methods.metric_seam.f2p_mock import ws3_eval_evidence as legacy


SCHEMA = "metric-seam.patent-ws3-family-retrospective.v1"
ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = ROOT / "outputs/metric_seam_pilot/tasks/patents_pa"
DEFAULT_OUT = (
    ROOT
    / "outputs/metric_seam_pilot/reconstruction_v2/"
    "patent_ws3_family_retrospective_001"
)
ASPECTS = ("a26", "a34", "a60", "a35")


class PatentFamilyError(RuntimeError):
    """Raised if the historical WS3 family cannot be reconstructed exactly."""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed(base: int, aspect: str, purpose: str) -> int:
    import hashlib

    payload = f"{base}\0{aspect}\0{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _load_program(aspect: str):
    path = ROOT / "methods/metric_seam/f2p_mock/programs_pa" / f"{aspect}_h0.py"
    spec = importlib.util.spec_from_file_location(f"patent_ws3_{aspect}", path)
    if spec is None or spec.loader is None:
        raise PatentFamilyError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def _validate_score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PatentFamilyError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise PatentFamilyError(f"{label} is outside [0, 1]")
    return number


def evaluate(
    *,
    task_dir: Path = TASK_DIR,
    split_seed: int = 7,
    test_fraction: float = 0.40,
    resampling_seed: int = 20_260_713,
    permutation_samples: int = 10_000,
    bootstrap_samples: int = 5_000,
    bootstrap_confidence: float = 0.95,
    alpha: float = 0.05,
    minimum_effect: float = 0.05,
    absolute_rho_min: float = 0.30,
    reference_reliability_min: float = 0.30,
    minimum_pairs: int = 40,
) -> dict[str, Any]:
    if task_dir.resolve() != legacy.OUT.resolve():
        raise PatentFamilyError("legacy loader is bound to the canonical patents_pa task")
    _, _, _, fields = legacy.load_all()
    evidence = legacy.load_ws3()["evidence"]
    items = _read_json(task_dir / "items.json")
    if not isinstance(items, list) or len(items) != 250:
        raise PatentFamilyError("expected the frozen 250-item patent corpus")
    texts: dict[str, str] = {}
    for row in items:
        # Deliberately project only the identifier and scoring text.  The source
        # outcome/``judgement`` value never enters this evaluator.
        item_id = row.get("datapoint_id")
        ctext = row.get("ctext")
        if not isinstance(item_id, str) or not isinstance(ctext, str):
            raise PatentFamilyError("patent item lacks datapoint_id or ctext")
        texts[item_id] = ctext
    ids = sorted(texts)
    test_n = int(test_fraction * len(ids))
    heldout_ids = sorted(random.Random(split_seed).sample(ids, test_n))
    if len(heldout_ids) != 100:
        raise PatentFamilyError("expected the historical 100-item held-out split")

    ops_full = legacy.PriorArtOps(task_dir / "pa_features.json")
    ops_null = legacy.NullPriorArtOps(task_dir / "pa_features.json")
    rows: list[dict[str, Any]] = []
    p_values: dict[str, float] = {}
    for aspect in ASPECTS:
        module, program_path = _load_program(aspect)
        extracted = {
            item_id: {
                field_name.split("__", 1)[1]: field_values.get(item_id, "")
                for field_name, field_values in fields.items()
                if field_name.startswith(aspect + "__")
            }
            for item_id in heldout_ids
        }
        full = {
            item_id: _validate_score(
                module.score(texts[item_id], extracted[item_id], ops_full, dpid=item_id),
                f"{aspect}.full[{item_id}]",
            )
            for item_id in heldout_ids
        }
        null = {
            item_id: _validate_score(
                module.score(texts[item_id], extracted[item_id], ops_null, dpid=item_id),
                f"{aspect}.null[{item_id}]",
            )
            for item_id in heldout_ids
        }
        reference = evidence["judge"].get(aspect, {})
        common = sorted(set(heldout_ids) & set(reference))
        if len(common) < minimum_pairs:
            raise PatentFamilyError(f"{aspect} has only {len(common)} reference-common rows")
        candidate_values = [full[item_id] for item_id in common]
        null_values = [null[item_id] for item_id in common]
        reference_values = [float(reference[item_id]) for item_id in common]
        rho_full = spearman(candidate_values, reference_values)
        rho_null = spearman(null_values, reference_values)
        if not all(math.isfinite(value) for value in (rho_full, rho_null)):
            raise PatentFamilyError(f"{aspect} has an undefined observed correlation")
        delta = rho_full - rho_null
        try:
            permutation = paired_randomization_test(
                candidate_values,
                null_values,
                reference_values,
                samples=permutation_samples,
                seed=_seed(resampling_seed, aspect, "permutation"),
            )
        except ValueError as exc:
            permutation = {
                "method": "unavailable",
                "p_value": 1.0,
                "reason": str(exc),
                "family_value_policy": "p=1 retains the registered criterion",
            }
        try:
            bootstrap = paired_bootstrap_ci(
                candidate_values,
                null_values,
                reference_values,
                samples=bootstrap_samples,
                confidence=bootstrap_confidence,
                seed=_seed(resampling_seed, aspect, "bootstrap"),
            )
            bootstrap["status"] = "available"
        except ValueError as exc:
            bootstrap = {
                "status": "unavailable",
                "method": "paired_item_percentile_bootstrap",
                "confidence": bootstrap_confidence,
                "interval": None,
                "reason": str(exc),
            }
        rel1 = float(evidence["rel1"][aspect])
        null_counts = Counter(null_values)
        null_modal_count = max(null_counts.values())
        p_values[aspect] = float(permutation["p_value"])
        rows.append(
            {
                "criterion_id": aspect,
                "program": program_path.relative_to(ROOT).as_posix(),
                "provenance": "retrospective_manual_oracle_conditioned_seed",
                "heldout_n": len(heldout_ids),
                "reference_common_n": len(common),
                "reference_availability": len(common) / len(heldout_ids),
                "reference_two_pass_spearman": rel1,
                "reference_reliability_floor": reference_reliability_min,
                "reference_reliability_floor_met": rel1 >= reference_reliability_min,
                "rho_full_evidence_operation": rho_full,
                "rho_null_operation": rho_null,
                "null_score_unique_values": len(null_counts),
                "null_score_modal_fraction": null_modal_count / len(null_values),
                "null_rank_support_warning": (
                    "near_degenerate_null_score_distribution"
                    if null_modal_count / len(null_values) >= 0.95
                    else None
                ),
                "delta_spearman": delta,
                "minimum_effect": minimum_effect,
                "minimum_effect_met": delta >= minimum_effect,
                "absolute_rho_min": absolute_rho_min,
                "absolute_floor_met": rho_full >= absolute_rho_min,
                "paired_randomization": permutation,
                "paired_bootstrap": bootstrap,
                "bh_q_value": None,
                "bh_fdr_reject": None,
                "threshold_and_fdr_screens_met": None,
                "effect_precision_characterized": None,
            }
        )

    q_values = benjamini_hochberg(p_values)
    for row in rows:
        aspect = row["criterion_id"]
        row["bh_q_value"] = q_values[aspect]
        row["bh_fdr_reject"] = q_values[aspect] <= alpha
        row["threshold_and_fdr_screens_met"] = all(
            (
                row["reference_reliability_floor_met"],
                row["minimum_effect_met"],
                row["absolute_floor_met"],
                row["bh_fdr_reject"],
            )
        )
        row["effect_precision_characterized"] = all(
            (
                row["threshold_and_fdr_screens_met"],
                row["paired_bootstrap"]["interval"] is not None,
                row["null_rank_support_warning"] is None,
            )
        )

    return {
        "schema": SCHEMA,
        "objective": "unsupervised reconstruction of a frozen evidence-aware LLM reference",
        "selection": "retrospective full four-criterion WS3 family",
        "confirmation_status": "descriptive_retrospective_not_confirmatory",
        "oracle_conditioning": (
            "examiner-cited prior art was force-included in candidate sets; autonomous "
            "retrieval and patent-truth claims are unavailable"
        ),
        "settings": {
            "split_seed": split_seed,
            "test_fraction": test_fraction,
            "heldout_n": test_n,
            "resampling_seed": resampling_seed,
            "permutation_samples": permutation_samples,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_confidence": bootstrap_confidence,
            "alpha": alpha,
            "minimum_effect": minimum_effect,
            "absolute_rho_min": absolute_rho_min,
            "reference_reliability_min": reference_reliability_min,
            "minimum_pairs": minimum_pairs,
        },
        "summary": {
            "registered_criteria": len(rows),
            "bh_family_size": len(q_values),
            "bh_fdr_rejections": sum(row["bh_fdr_reject"] for row in rows),
            "threshold_and_fdr_screens_met": sum(
                row["threshold_and_fdr_screens_met"] for row in rows
            ),
            "threshold_and_fdr_screen_ids": [
                row["criterion_id"]
                for row in rows
                if row["threshold_and_fdr_screens_met"]
            ],
            "effect_precision_characterized": sum(
                row["effect_precision_characterized"] for row in rows
            ),
            "effect_precision_characterized_ids": [
                row["criterion_id"]
                for row in rows
                if row["effect_precision_characterized"]
            ],
        },
        "criteria": rows,
        "input_policy": {
            "items_fields_read": ["datapoint_id", "ctext"],
            "items_judgement_read": False,
            "model_calls": False,
            "gpu_used": False,
        },
    }


def render_report(result: Mapping[str, Any]) -> str:
    table_rows = []
    for row in result["criteria"]:
        interval = row["paired_bootstrap"]["interval"]
        interval_text = (
            f"[{interval[0]:+.3f}, {interval[1]:+.3f}]"
            if interval is not None
            else "unavailable"
        )
        table_rows.append(
            f"| {row['criterion_id']} | {row['reference_two_pass_spearman']:.3f} | "
            f"{row['rho_full_evidence_operation']:.3f} | {row['rho_null_operation']:.3f} | "
            f"{row['delta_spearman']:+.3f} | {interval_text} | "
            f"{row['paired_randomization']['p_value']:.4f} | {row['bh_q_value']:.4f} | "
            f"{'yes' if row['threshold_and_fdr_screens_met'] else 'no'} |"
        )
    table = "\n".join(table_rows)
    summary = result["summary"]
    return f"""# Patent WS3 full-family retrospective inference

All four historically selected evidence-dominant criteria are retained. The comparison is
the same hybrid program with the prior-art operation present versus nulled, against the
frozen two-pass evidence-aware LLM reconstruction reference.

| criterion | ref rel | full rho | null rho | delta | paired 95% CI | p | BH q | threshold+FDR screens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

BH-FDR rejects {summary['bh_fdr_rejections']}/{summary['registered_criteria']} paired nulls;
{summary['threshold_and_fdr_screens_met']}/{summary['registered_criteria']} also meet the
retrospectively declared reliability, absolute-rho, and effect-size screens. These are
descriptive multiplicity-aware results, not confirmatory certifications: the family and
aggregate outcomes were already known before this evaluator was frozen.

The nulled program is nearly rank-degenerate for a26, a34, and a60 (the modal score occurs
on at least 95% of held-out rows). Their null rho and bootstrap difference are correspondingly
fragile; unavailable intervals are reported as unavailable rather than repaired with jitter or
selective resampling. The paired randomization family is still shown, but this null-support
diagnostic prevents the large point marginals from being mistaken for precise effect sizes.
Only {summary['effect_precision_characterized']}/{summary['registered_criteria']} result(s)
({', '.join(summary['effect_precision_characterized_ids']) or 'none'}) both clear those
screens and have a non-degenerate null plus an available paired interval.

Examiner-cited art was force-included upstream. The result supports an oracle-conditioned
representation/relation match for the prior-art operation; it does not establish autonomous
retrieval, patent correctness, external ground truth, or population patent-metric codability.
"""


def write_result(result: Mapping[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "REPORT.md").write_text(render_report(result), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = evaluate()
    if args.check:
        existing = _read_json(args.out_dir / "results.json")
        if existing != result:
            raise PatentFamilyError("stored patent family result differs from rerun")
    else:
        write_result(result, args.out_dir)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
