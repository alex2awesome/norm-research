from methods.metric_seam.verifiers.lifecycle import (
    ConstructControl,
    UnitProposal,
    evaluate_construct_challenge,
    evaluate_pre_authoring_probe,
    select_after_construct_gate,
    stable_train_sample,
)
from methods.metric_seam.verifiers.schema import Span, Verdict


def _proposal() -> UnitProposal:
    return UnitProposal(
        unit_id="task.a1.unit", task="task", criterion_id="a1",
        construct_text="A construct.", relation="A bounded relation.", occasion="An occasion.",
        satisfied_when="It holds.", violated_when="It fails.", required_context="Full document.",
        non_goals=("whole metric",), proxy_risks=("surface token",),
    )


def _verdict(state: str) -> Verdict:
    if state == "not_applicable":
        return Verdict(False, False)
    return Verdict(True, state == "violated", (Span("document.txt", 1, 1),))


def test_stable_sample_and_pre_authoring_stop() -> None:
    rows = [{"item_key": f"i{index}", "ctext": "x"} for index in range(40)]
    sample = stable_train_sample(rows, salt="frozen", sample_size=8)
    assert sample == stable_train_sample(list(reversed(rows)), salt="frozen", sample_size=8)
    ids = [row["item_key"] for row in sample]
    verdicts = {item_id: _verdict("satisfied") for item_id in ids}
    result = evaluate_pre_authoring_probe(_proposal(), ids, verdicts)
    assert result["passed"] is False
    assert result["detector_authorship_permitted"] is False
    assert "too_few_violated_items" in result["failed_checks"]


def test_construct_challenge_preempts_agreement() -> None:
    controls = tuple(
        [ConstructControl(f"on{i}", "text", "satisfied", True, "counterproxy") for i in range(4)]
        + [ConstructControl(f"off{i}", "text", "violated", False, "counterproxy") for i in range(4)]
    )
    judges = {control.control_id: [control.expected_state, control.expected_state] for control in controls}
    verifier = {
        control.control_id: _verdict("violated" if control.proxy_triggered else "not_applicable")
        for control in controls
    }
    challenge = evaluate_construct_challenge(_proposal(), controls, judges, verifier)
    assert challenge["passed"] is False
    selection = select_after_construct_gate(
        _proposal(), base_rate_probe={"passed": True}, natural_train_gate={"passed": True},
        construct_challenge=challenge, authorship_rounds=1,
    )
    assert selection["selected"] is False
    assert selection["agreement_computed"] is False
