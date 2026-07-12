"""Planted-world tests for form_decompose: the decomposition must (a) calibrate away a PURE
STRICTNESS SHIFT (form main effect), (b) NOT calibrate away a PERMUTATION (item-x-form
entanglement), (c) read main-effect share off pair records correctly, (d) degrade gracefully on
pre-2026-07-02 artifacts."""
import numpy as np

from methods.metric_implementer.experiments import form_decompose as fd


def _npz(tmp_path, name="m0", **fields):
    p = tmp_path / f"{name}_sigs.npz"
    np.savez(p, **fields)
    return np.load(p, allow_pickle=True)


def test_pure_shift_world_calibrates_away(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.uniform(0.25, 0.75, 400)                       # margin so shifts never clip
    forms = np.vstack([base, base + 0.12, base - 0.10, base + 0.05])
    z = _npz(tmp_path, M_i_forms=forms,
             M_i_form_names=np.array(["canonical", "question", "boilerplate", "suffix"],
                                     dtype=object))
    ts = fd.target_seat(z)
    assert ts["flip_raw"] > 0.05                              # shifts DO flip verdicts raw...
    assert ts["flip_cal"] < 0.02                              # ...and calibration removes them
    # in a shift world the flips live at the decision boundary
    assert ts["boundary_dist_flipped"] < ts["boundary_dist_stable"]


def test_permutation_world_survives_calibration(tmp_path):
    rng = np.random.default_rng(1)
    base = rng.uniform(0.05, 0.95, 400)
    forms = [base]
    for s in (2, 3, 4):
        f = base.copy()
        idx = np.random.default_rng(s).choice(400, 80, replace=False)
        f[idx] = 1.0 - f[idx]                                 # sign-balanced rank scramble
        forms.append(f)
    ts = fd.target_seat(_npz(tmp_path, M_i_forms=np.vstack(forms)))
    assert ts["flip_raw"] > 0.05
    assert ts["flip_cal"] > 0.5 * ts["flip_raw"]              # reordering is NOT calibratable


def test_instrument_seat_main_effect_share_extremes():
    mk = lambda b, d: {"criterion": 0, "form": "question", "drift": d, "flip": 0.1,
                       "bias": b, "yes_shift": 0.0}
    uniform = fd.instrument_seat({"pairs": [mk(0.10, 0.10), mk(-0.08, 0.08)]})
    assert uniform["question"]["main_effect_share"] == 1.0    # |bias| == drift: pure shift
    entangled = fd.instrument_seat({"pairs": [mk(0.001, 0.20), mk(0.0, 0.15)]})
    assert entangled["question"]["main_effect_share"] < 0.05  # drift with no net direction


def test_old_format_graceful(tmp_path):
    z = _npz(tmp_path, M_i=np.zeros(10), sigs=np.zeros((3, 10)))
    assert fd.target_seat(z) is None                          # no M_i_forms saved
    assert fd.instrument_seat({"median_drift": 0.1}) is None  # no pairs recorded
