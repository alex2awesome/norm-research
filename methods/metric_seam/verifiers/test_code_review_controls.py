from methods.metric_seam.verifiers.code_review_controls import CONTROLS, controls_by_id


def test_control_registry_is_unique_and_has_required_placements():
    by_id = controls_by_id()
    assert len(by_id) == len(CONTROLS)
    assert by_id["pcr901"].expected == "certify"
    assert by_id["pcr902"].expected == "certify"
    assert by_id["pcr906"].expected == "do_not_certify"
    assert by_id["pcr905"].expected == "directional_only"


def test_positive_controls_have_parseable_ordinary_go_sources():
    for control in CONTROLS:
        if control.expected != "certify":
            continue
        source = "\n".join(control.planted_source)
        assert source.startswith("package internal")
        assert "metric_seam" not in source
        assert control.extension == "go"


def test_null_and_directional_controls_do_not_manufacture_variance():
    for control_id in ("pcr905", "pcr906"):
        control = controls_by_id()[control_id]
        assert control.mutation_kind is None
        assert control.planted_source == ()
