from methods.metric_seam.run_construct_validity_repair import build_readout


def test_a12_proxy_misconstrual_is_executable() -> None:
    readout = build_readout()
    assert readout["status"] == "proxy_misconstrual_executably_demonstrated"
    summary = readout["summary"]
    assert summary["proxy_on_construct_satisfied"] >= 4
    assert summary["proxy_on_called_violated"] >= 4
    assert summary["old_verifier_construct_correct"] < summary["controls"]
    assert readout["disposition"]["a12_rigor_unit"] == "rejected_before_freeze_construct_misconstrual"
