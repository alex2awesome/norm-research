"""Certificates module: bracket gating, count bounds, and the ledger->report path."""

import json
import math

from metrics_tree_infilling.certificates import (
    count_certificates, dense_stack_wrap, flux_wrap, report_from_ledger,
    report_from_ledgers)


def test_dominance_gate_right_censors_cw_case():
    # dense still climbing -> no wrap, no N_upper, regardless of margins
    b = dense_stack_wrap(dense_bits=0.20, bank_bits=0.10, dense_plateaued=False)
    assert b.right_censored and b.wrap_bits is None
    c = count_certificates(b, n_kept=3, delta_bits=0.01)
    assert c.n_upper is None
    # gate passes -> wrap live
    b2 = dense_stack_wrap(dense_bits=0.20, bank_bits=0.10, dense_plateaued=True)
    assert not b2.right_censored and b2.wrap_bits == 0.20


def test_count_bounds_arithmetic():
    b = dense_stack_wrap(dense_bits=0.30, bank_bits=0.12, dense_plateaued=True)
    c = count_certificates(b, n_kept=4, delta_bits=0.02, K=2,
                           rejected_gains_bits=[0.004, 0.001, -0.002], gamma_hat=0.8)
    assert c.n_lower_logK == math.ceil(0.12 / 1.0) == 1
    assert abs(c.n_upper - (4 + (0.30 - 0.12) / 0.02)) < 1e-9      # 13.0
    assert abs(c.minoux_tail_bits - 0.005) < 1e-12                  # negatives excluded
    assert abs(c.minoux_epsilon - 0.005 / 0.8) < 1e-12


def test_Tmax_leg_requires_per_metric_call_scoring():
    b = dense_stack_wrap(dense_bits=0.30, bank_bits=0.12, dense_plateaued=True)
    c_shared = count_certificates(b, 4, 0.02, T_max_bits=0.05, per_metric_call_scoring=False)
    assert c_shared.n_lower_Tmax is None            # shared-call: leg not licensed
    c_ok = count_certificates(b, 4, 0.02, T_max_bits=0.05, per_metric_call_scoring=True)
    assert c_ok.n_lower_Tmax == math.ceil(0.12 / 0.05) == 3


def test_flux_wrap_rejects_bad_gamma():
    assert flux_wrap(0.1, 0.05, gamma_hat=0.0).right_censored
    w = flux_wrap(0.1, 0.05, gamma_hat=0.5)
    assert abs(w.wrap_bits - 0.2) < 1e-12


def test_report_from_ledger_roundtrip(tmp_path):
    ledger = {
        "guard_bits_trajectory": [0.05, 0.08, 0.11],
        "rounds": 2,
        "ledgers": [
            {"status": "kept", "bits_gain": 0.03, "generator": "residual"},
            {"status": "kept", "bits_gain": 0.03, "generator": "residual"},
            {"status": "dropped:auc", "bits_gain": 0.002, "generator": "residual"},
        ],
    }
    p = tmp_path / "global_infill_ledger.json"
    p.write_text(json.dumps(ledger))
    rep = report_from_ledger(p, task="toy", executor="glm-5.2", delta_bits=0.01, K=2,
                             dense_bits=0.30, dense_plateaued=True)
    assert rep.bracket.floor_bits == 0.11
    assert rep.counts.n_kept == 2
    assert abs(rep.counts.n_upper - (2 + (0.30 - 0.11) / 0.01)) < 1e-9
    assert any("single generator arm" in n for n in rep.notes)     # honesty note fires
    assert "N_upper" in rep.render() and "value bracket" in rep.render()


def test_report_from_ledgers_union(tmp_path):
    """MCC §2a: union over arms — floor = max of per-arm floors, kept/rejected pooled,
    single-arm note suppressed at >=2 arms, max-not-stacked note when >1 arm kept."""
    def _write(name, traj, ledgers):
        p = tmp_path / name
        p.write_text(json.dumps({"guard_bits_trajectory": traj, "ledgers": ledgers}))
        return p

    p1 = _write("a.json", [0.05, 0.09], [
        {"status": "kept", "bits_gain": 0.04, "generator": "residual"},
        {"status": "dropped:auc", "bits_gain": 0.003, "generator": "residual"}])
    p2 = _write("b.json", [0.05, 0.07], [
        {"status": "kept", "bits_gain": 0.02, "generator": "label_contrast"},
        {"status": "dropped:auc", "bits_gain": -0.001, "generator": "label_contrast"}])

    rep = report_from_ledgers([p1, p2], task="toy", executor="glm-5.2", delta_bits=0.01)
    assert rep.bracket.floor_bits == 0.09                       # max, not sum
    assert rep.counts.n_kept == 2                               # pooled across arms
    assert abs(rep.counts.minoux_tail_bits - 0.003) < 1e-12     # rejected pooled, negatives out
    assert not any("single generator arm" in n for n in rep.notes)
    assert any("union certificate over 2 generator arms" in n for n in rep.notes)
    assert any("max over per-arm banks" in n for n in rep.notes)
    # single-path call still fires the single-arm note (delegation preserved)
    rep1 = report_from_ledger(p1, task="toy", executor="glm-5.2", delta_bits=0.01)
    assert any("single generator arm" in n for n in rep1.notes)
