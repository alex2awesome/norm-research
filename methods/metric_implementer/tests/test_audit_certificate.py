"""Regression tests for the certificate audit disciplines (born 2026-07-02 audit)."""
from methods.metric_implementer.experiments import audit_certificate as ac


def test_degenerate_filter_and_spurious_codifiable():
    rows = [
        {"name": "good", "verdict": "CODIFIABLE", "H_M": 0.8, "form_invariant": True},
        {"name": "vacuous", "verdict": "CODIFIABLE", "H_M": 0.02, "form_invariant": True},  # degen
        {"name": "fd", "verdict": "FORM-DOMINATED", "H_M": 0.6, "form_invariant": False},
        {"name": "null_hm", "verdict": "UNDERSAMPLED", "H_M": None, "form_invariant": False},  # degen
    ]
    r = ac.audit_cert(rows, floor=0.15)
    assert r["n_degenerate"] == 2 and r["n_keep"] == 2
    assert r["verdicts_kept"] == {"CODIFIABLE": 1, "FORM-DOMINATED": 1}
    assert r["spurious_codifiable"] == ["vacuous"]
    assert r["form_gate_pass_kept"] == 1               # only 'good' passes among kept


def test_clean_reader_gaps_excludes_reference_executor():
    # three readers; 8B is the reference executor and must be dropped from gaps
    report = {
        "Llama-3.2-1B": {"m0": {"name": {"bal_acc": 0.50}, "definition": {"bal_acc": 0.55}}},
        "Llama-3.2-3B": {"m0": {"name": {"bal_acc": 0.60}, "definition": {"bal_acc": 0.70}}},
        "Llama-3.1-8B": {"m0": {"name": {"bal_acc": 0.90}, "definition": {"bal_acc": 0.95}}},
    }
    g = ac.clean_reader_gaps(report, "8B")
    assert g["excluded_reference_readers"] == ["Llama-3.1-8B"]
    assert g["readers_used"] == ["Llama-3.2-1B", "Llama-3.2-3B"]
    # only the 3B−1B pair remains; name gap .60−.50=.10, definition .70−.55=.15
    assert g["gaps"]["3B−1B"]["name"] == 0.10
    assert g["gaps"]["3B−1B"]["definition"] == 0.15
