import json

from scripts.tools.silver_match_v3.select_predeclared_verifier_policy import main


def test_selects_higher_wilson_eligible_policy(tmp_path, monkeypatch):
    freeze = {
        "task": "humor",
        "candidate_policies": [{"name": "two_order_exact_high"}, {"name": "all_three_order_exact_high"}],
        "eligibility_gate": {"minimum_retained": 20, "minimum_point_precision": 0.9, "minimum_wilson_95_lower": 0.8},
    }
    two_policy = {"retained": 30, "retained_precision": 0.95, "retained_precision_wilson_95": [0.81, 0.99]}
    three_policy = {"retained": 22, "retained_precision": 1.0, "retained_precision_wilson_95": [0.85, 1.0]}
    values = {
        "freeze.json": freeze,
        "two.json": {"selection_split": "dev", "policies": {"high_only": two_policy}},
        "three.json": {"selection_split": "dev", "policy": three_policy},
    }
    for name, value in values.items():
        (tmp_path / name).write_text(json.dumps(value))
    output = tmp_path / "selected.json"
    monkeypatch.setattr("sys.argv", ["select", "--policy-freeze", str(tmp_path / "freeze.json"), "--two-order-score", str(tmp_path / "two.json"), "--three-order-score", str(tmp_path / "three.json"), "--output", str(output)])
    main()
    assert json.loads(output.read_text())["chosen"]["name"] == "all_three_order_exact_high"
