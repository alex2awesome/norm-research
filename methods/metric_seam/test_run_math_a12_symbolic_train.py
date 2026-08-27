from __future__ import annotations

from methods.metric_seam.run_math_a12_symbolic_train import run_train


def test_train_gate_uses_natural_pairs_and_probes_do_not_rescue_base_rate() -> None:
    rows = [
        {
            "item_key": "train_0001",
            "ctext": "Answer:\n$$x=x$$\n$$x=2x$$\n$$f(x)=x$$",
        },
        {
            "item_key": "train_0002",
            "ctext": "Answer:\n$$y+y=2y$$\n$$y+y=3y$$\n$$g(y)=y$$",
        },
    ]
    result = run_train(rows, probe_cap=2)
    assert result["natural_state_counts"] == {
        "not_applicable": 2,
        "satisfied": 2,
        "violated": 2,
    }
    assert result["gate"]["natural_pair_candidates"] == 6
    assert result["gate"]["natural_applies"] == 4
    assert result["gate"]["probe_total"] == 2
    assert result["gate"]["probe_correct"] == 2
    assert result["model_or_gpu_used"] is False
