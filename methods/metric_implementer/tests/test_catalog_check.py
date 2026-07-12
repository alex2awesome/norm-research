"""The catalog checker must catch exactly the class of mistake that produced the 2026-07-02
level-dispatch bug: a checkpoint whose stored identity does not match the hierarchy entry its
filename points at."""
import json

import numpy as np

from methods.metric_implementer.experiments import catalog_check as cc

GROUPS = {"R3": [{"merged_name": f"r3-concept-{i}"} for i in range(10)],
          "R2": [{"merged_name": f"r2-concept-{i}"} for i in range(10)]}


def _groups_for(level):
    return GROUPS[level]


def _ckpt(tmp_path, fname, **fields):
    p = tmp_path / fname
    np.savez(p, **fields)
    return str(p)


def test_c1_catches_level_identity_mismatch(tmp_path):
    good = _ckpt(tmp_path, "t_R3_metric2_sigs.npz", name="r3-concept-2",
                 M_i=np.zeros(5), sigs=np.zeros((3, 5)))
    bad = _ckpt(tmp_path, "t_R3_metric4_sigs.npz", name="r2-concept-4",   # WRONG hierarchy's name
                M_i=np.zeros(5), sigs=np.zeros((3, 5)))
    assert cc.check_ckpt(good, _groups_for)["fails"] == []
    fails = cc.check_ckpt(bad, _groups_for)["fails"]
    assert len(fails) == 1 and "C1 name_match" in fails[0]


def test_c2_c3_c5_shape_stale_orbit(tmp_path):
    p = _ckpt(tmp_path, "t_R3_metric1_sigs.npz", name="r3-concept-1",
              M_i=np.zeros(5), sigs=np.zeros((3, 7)),                     # C2: 5 != 7
              tau0=0.05,                                                  # C3: stale literal
              orbit_forms=4)                                              # C5: orbit fields absent
    r = cc.check_ckpt(p, _groups_for)
    assert any("C2 shape" in f for f in r["fails"])
    assert any("C5 orbit" in f for f in r["fails"])
    assert any("C3" in w for w in r["warns"])


def test_c4_forminv_status_and_dir_rollup(tmp_path):
    _ckpt(tmp_path, "t_R3_metric2_sigs.npz", name="r3-concept-2",
          M_i=np.zeros(5), sigs=np.zeros((3, 5)))
    (tmp_path / "t_R3_metric2_forminv.json").write_text(
        json.dumps({"binary_flip_rate": 0.1, "pairs": [{"form": "question"}]}))
    _ckpt(tmp_path, "t_R3_metric3_sigs.npz", name="r3-concept-3",
          M_i=np.zeros(5), sigs=np.zeros((3, 5)))
    d = cc.check_dir(str(tmp_path), _groups_for)
    assert d["forminv"] == {"pairs": 1, "summary-only": 0, "MISSING": 1}
    assert d["status"] == "OK" and d["n_ckpts"] == 2


def test_unparseable_filename_fails(tmp_path):
    p = _ckpt(tmp_path, "weird_sigs.npz", name="x")
    assert cc.check_ckpt(p, _groups_for)["fails"]
