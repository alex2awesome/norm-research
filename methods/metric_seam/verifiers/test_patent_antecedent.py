from methods.metric_seam.verifiers.patent_antecedent import construct_controls, verify_antecedent_basis


def test_construct_controls_cross_the_proxy_and_construct() -> None:
    controls = construct_controls()
    assert sum(c.proxy_triggered and c.expected_state == "satisfied" for c in controls) == 4
    assert sum((not c.proxy_triggered) and c.expected_state == "violated" for c in controls) == 4
    assert all(verify_antecedent_basis(c.ctext).applies for c in controls)
