import json

from scripts.tools.silver_match_v3.aggregate_mi_validation_v3 import aggregate


def _report(path, task, rho, n, supported):
    path.write_text(
        json.dumps(
            {
                "status": "TASK_FROZEN_ANALYSIS",
                "task": task,
                "certificate": {"bank_coverage": 1.0},
                "exact_matches": 10,
                "exact_match_rate": .5,
                "precision_claim_supported": supported,
                "false_abstention_claim_supported": supported,
                "results": {
                    "source_presence": {
                        "OPT": {
                            "spearman_rho": rho,
                            "source_group_bootstrap_rho_95": [rho - .1, rho + .1],
                            "permutation_p_two_sided": .05,
                            "partial_rho_given_log_leaf_count_and_HM": rho / 2,
                            "n_metrics": n,
                        }
                    }
                },
            }
        )
    )
    return path


def test_meta_reports_all_and_blind_supported_subsets(tmp_path):
    paths = [
        _report(tmp_path / "a.json", "a", .2, 20, True),
        _report(tmp_path / "b.json", "b", .4, 100, True),
        _report(tmp_path / "c.json", "c", -.1, 30, False),
    ]
    result = aggregate(paths)
    assert result["all_released_tasks_meta"]["tasks"] == 3
    assert result["both_blind_claims_supported_meta"]["tasks"] == 2
    assert result["both_blind_claims_supported_tasks"] == ["a", "b"]
