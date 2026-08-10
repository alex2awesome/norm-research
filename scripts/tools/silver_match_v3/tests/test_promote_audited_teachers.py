from scripts.tools.silver_match_v3.promote_audited_teachers import gate


def test_audit_gate_requires_locked_teachers_and_precision():
    teachers = [{"gradient_eligible": False} for _ in range(2)]
    audit = {
        "retained_exact_precision": {"n": 35},
        "retained_exact_precision_design_weighted": {
            "estimate": 0.95,
            "approximate_wilson_95": [0.82, 0.99],
        },
    }
    result = gate(
        teachers,
        audit,
        min_retained_audit_n=30,
        min_point_precision=0.9,
        min_ci_lower=0.8,
    )
    assert result["passed"] is True


def test_audit_gate_rejects_underpowered_sample():
    result = gate(
        [{"gradient_eligible": False}],
        {
            "retained_exact_precision": {"n": 2},
            "retained_exact_precision_design_weighted": {
                "estimate": 1.0,
                "approximate_wilson_95": [0.34, 1.0],
            },
        },
        min_retained_audit_n=30,
        min_point_precision=0.9,
        min_ci_lower=0.8,
    )
    assert result["passed"] is False
