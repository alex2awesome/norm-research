"""Flux read (U_flux) from union ledgers: species quotient -> value spectrum -> D1-D3.
Embeddings are injected so tests exercise the bridge, not sentence-transformers."""

import json

import numpy as np
import pytest

from metrics_tree_infilling.certificates import report_from_ledgers
from metrics_tree_infilling.flux import _binary_entropy, flux_from_ledgers, species_from_proposals

# deterministic "embeddings": keyword -> orthogonal unit vector
_KEYS = ["zephyr", "dialogue", "tempo", "imagery"]


def embed_fn(texts):
    out = []
    for t in texts:
        v = np.zeros(len(_KEYS))
        for i, k in enumerate(_KEYS):
            if k in t.lower():
                v[i] = 1.0
        if v.sum() == 0:
            v[-1] = 1.0
        out.append(v / np.linalg.norm(v))
    return np.array(out)


def test_species_quotient_merges_paraphrases():
    texts = ["zephyr wind marker", "mentions a zephyr", "uses dialogue"]
    labels = species_from_proposals(texts, tau=0.92, embed_fn=embed_fn)
    assert labels[0] == labels[1] != labels[2]
    assert len(set(labels)) == 2


def _write_ledger(path, entries, floor=0.05):
    with open(path, "w") as f:
        json.dump({"guard_bits_trajectory": [floor], "rounds": len(entries),
                   "ledgers": entries}, f)


def _entry(name, rubric, gain, gen, status="dropped:auc_gain<0.02"):
    return {"name": name, "rubric": rubric, "description": name, "round": 1,
            "status": status, "bits_gain": gain, "generator": gen}


@pytest.fixture
def ledgers(tmp_path):
    a = tmp_path / "arm_a.json"
    b = tmp_path / "arm_b.json"
    _write_ledger(a, [_entry("Zephyr", "mentions a zephyr wind", 0.02, "residual"),
                      _entry("Dialogue", "uses dialogue well", 0.005, "residual")],
                  floor=0.05)
    _write_ledger(b, [_entry("Zephyr marker", "names the zephyr", 0.01, "metric_tree"),
                      _entry("Tempo", "controls tempo", None, "metric_tree")],
                  floor=0.04)
    return [a, b]


def test_flux_from_ledgers_spectrum_and_tail(ledgers):
    fx = flux_from_ledgers(ledgers, base_rate=0.4, c=1.0, embed_fn=embed_fn)
    assert fx["n_draws"] == 4 and fx["n_arms"] == 2
    assert fx["n_species"] == 3                       # zephyr x2 merged; dialogue; tempo
    assert fx["f1"] == 2
    # species values: zephyr max(0.02, 0.01)=0.02 (n=2); dialogue 0.005 (n=1); tempo nan->0 (n=1)
    assert fx["value_spectrum"] == {"1": 0.005, "2": 0.02}
    assert fx["floor_bits"] == pytest.approx(0.05)    # max over arm floors
    # T1 cap: H(0.4) - 0.05 >> empirical max 0.02
    assert fx["B_cap"] == pytest.approx(_binary_entropy(0.4) - 0.05)
    # G(c=1) = w1 - w2 = 0.005 - 0.02 < 0 -> clamped; tail = slack only
    assert fx["good_toulmin_Gc"] == pytest.approx(0.005 - 0.02)
    assert fx["flux_tail_bits"] == pytest.approx(fx["mcdiarmid_slack"])
    assert fx["delta_eff"] < fx["delta"]              # anytime spending is strictly costlier


def test_flux_positive_horizon_when_singletons_carry_value(tmp_path):
    a = tmp_path / "a.json"
    _write_ledger(a, [_entry("Zephyr", "zephyr wind", 0.03, "residual"),
                      _entry("Dialogue", "dialogue quality", 0.02, "unconditional"),
                      _entry("Imagery", "imagery density", 0.01, "unconditional")])
    fx = flux_from_ledgers([a], base_rate=0.5, embed_fn=embed_fn)
    assert fx["n_species"] == 3 and fx["f1"] == 3 and fx["singleton_regime"]
    assert fx["good_toulmin_Gc"] == pytest.approx(0.06)   # all singletons: G(1) = w1
    assert fx["flux_tail_bits"] == pytest.approx(0.06 + fx["mcdiarmid_slack"])


def test_union_report_takes_flux_wrap_when_dense_censored(ledgers):
    fx = flux_from_ledgers(ledgers, base_rate=0.4, embed_fn=embed_fn)
    rep = report_from_ledgers(ledgers, task="t", executor="e", delta_bits=0.01,
                              flux_tail_bits=fx["flux_tail_bits"], flux_meta=fx)
    assert not rep.bracket.right_censored
    assert rep.bracket.wrap_source == "flux"
    assert rep.counts.n_upper is not None
    text = rep.render()
    assert "PROCESS-RELATIVE" in text


def test_union_report_flux_honesty_notes():
    meta = {"singleton_regime": True, "f1_over_species": 1.0, "n_species": 10,
            "n_species_audit_tau": 4, "tau": 0.92, "audit_tau": 0.85, "n_arms": 1}
    import json as _json
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "l.json"
        _write_ledger(p, [_entry("A", "a", 0.01, "residual")])
        rep = report_from_ledgers([p], task="t", executor="e", delta_bits=0.01,
                                  flux_tail_bits=0.05, flux_meta=meta)
    joined = " ".join(rep.notes)
    assert "SINGLETON regime" in joined
    assert "merge-precision audit DIVERGES" in joined
    assert "SINGLE arm" in joined


def test_union_report_prefers_tighter_wrap(ledgers):
    # dense wrap valid at 0.30; flux tail tiny -> flux wrap = floor + tail is tighter
    rep = report_from_ledgers(ledgers, task="t", executor="e", delta_bits=0.01,
                              dense_bits=0.30, dense_plateaued=True,
                              flux_tail_bits=0.01)
    assert rep.bracket.wrap_source == "flux"
    assert rep.bracket.wrap_bits == pytest.approx(0.05 + 0.01)
    # and the other way: huge flux tail -> dense stays
    rep2 = report_from_ledgers(ledgers, task="t", executor="e", delta_bits=0.01,
                               dense_bits=0.30, dense_plateaued=True,
                               flux_tail_bits=5.0)
    assert rep2.bracket.wrap_source == "dense_stack"
