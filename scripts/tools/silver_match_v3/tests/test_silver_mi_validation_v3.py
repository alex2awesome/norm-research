import json

import numpy as np

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.silver_mi_validation_v3 import (
    _permutation_p,
    run_validation,
)


def test_permutation_test_draws_exactly_declared_replicates():
    class CountingRng:
        def __init__(self):
            self.calls = 0

        def permutation(self, values):
            self.calls += 1
            return np.roll(values, self.calls)

    rng = CountingRng()
    rho, p_value = _permutation_p(
        np.arange(6, dtype=float),
        np.arange(6, dtype=float),
        7,
        rng,
    )
    assert rng.calls == 7
    assert rho == 1.0
    assert 0.0 < p_value <= 1.0


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _artifact(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def test_v3_mi_validation_uses_only_frozen_exact_matches_and_source_groups(tmp_path):
    bank = tmp_path / "bank.json"
    metrics = [
        {"metric_id": f"a{i}", "metric_index": i, "name": f"Metric {i}", "leaf_count": i + 1}
        for i in range(4)
    ]
    bank.write_text(json.dumps({"source_sha256": "b" * 64, "metrics": metrics}))
    norms = tmp_path / "norms.jsonl"
    _jsonl(
        norms,
        [
            {"norm_uid": str(i), "source_id": f"s{i // 2}", "polarity": "positive", "kind": "norm"}
            for i in range(10)
        ],
    )
    final = tmp_path / "final.jsonl"
    decisions = [
        ("MATCH", "a0"),
        ("MATCH", "a0"),
        ("MATCH", "a1"),
        ("MATCH", "a2"),
        ("NO_CANDIDATE_FITS", None),
        ("MATCH", "a3"),
        ("NO_EXPLICIT_CRITERION", None),
        ("MATCH", "a3"),
        ("NOISE", None),
        ("NO_CANDIDATE_FITS", None),
    ]
    _jsonl(
        final,
        [
            {"norm_uid": str(i), "corpus": "corpus", "decision": d, "metric_id": m}
            for i, (d, m) in enumerate(decisions)
        ],
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "banks": {"task": {"path": str(bank), "source_sha256": "b" * 64}},
                "corpora": {"corpus": {"task": "task", "count": 10, "path": str(norms)}},
            }
        )
    )
    plan, final_audit, risk = tmp_path / "plan", tmp_path / "audit", tmp_path / "risk"
    for path in (plan, final_audit, risk):
        path.write_text(path.name)
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps(
            {
                "status": "TASK_FROZEN_ANALYSIS_READY",
                "task": "task",
                "corpora": ["corpus"],
                "expected_rows": 10,
                "manifest": _artifact(manifest),
                "production_plan": _artifact(plan),
                "final_audit": _artifact(final_audit),
                "blind_risk_audit": _artifact(risk),
                "final_outputs": [_artifact(final)],
                "bank_source_sha256": "b" * 64,
                "blind_risk": {"audited_rows": 100},
                "analysis_exclusions": {"count": 1, "norm_uids": ["8"]},
                "precision_claim_supported": True,
                "false_abstention_claim_supported": False,
                "analysis_firewall": {
                    "task_matcher_is_immutable": True,
                    "may_tune_this_or_other_task_matchers_from_results": False,
                },
            }
        )
    )
    certificate = tmp_path / "cert.json"
    certificate.write_text(
        json.dumps(
            [
                {
                    "file": f"task_R2_metric{i}_sigs.npz",
                    "name": f"Metric {i}",
                    "opt_omega_bits": float(i),
                    "H_M": float(i + 1),
                    "gains": [float(i) / 2],
                    "eps_bits_adv": 0.1,
                }
                for i in range(4)
            ]
        )
    )
    result = run_validation(
        release_path=release,
        certificate_path=certificate,
        n_permutations=10,
        n_bootstrap=10,
        seed=1,
    )
    assert result["rows"] == 10
    assert result["analysis_excluded_rows"] == 1
    assert result["analysis_eligible_rows"] == 9
    assert result["exact_matches"] == 6
    assert result["source_groups"] == 5
    assert result["certificate"]["join_counts"] == {"metric_index": 4}
    assert result["primary_estimand"] == "source_presence.OPT.spearman_rho"
    assert result["results"]["source_presence"]["OPT"]["n_metrics"] == 4
    assert len(result["per_metric"]) == 4
    assert result["per_metric"][0] == {
        "metric_id": "a0",
        "metric_index": 0,
        "metric_name": "Metric 0",
        "leaf_count": 1.0,
        "mi_scores": {"EPS_ADV": 0.1, "G1": 0.0, "OPT": 0.0, "T_HM": 1.0},
        "silver_salience": {
            "equal_corpus_share": 1 / 3,
            "micro_norm": 2.0,
            "negative_micro_norm": 0.0,
            "positive_micro_norm": 2.0,
            "source_presence": 1.0,
        },
    }
    assert result["analysis_units"]["metric_unit_key"] == "bank metric_id"
