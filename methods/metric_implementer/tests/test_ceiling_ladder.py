import json
from pathlib import Path

import numpy as np
import pandas as pd

from methods.metric_implementer.experiments.ceiling_ladder import (
    PLANTED_CRITERIA,
    _binary_mi_matrix,
    _permuted_ladder_values,
    _planted_text,
    _sha,
    dawid_skene_binary,
    fleiss_kappa,
    miller_madow_mutual_information,
    permutation_pvalue,
    reference_reliability_gate,
    run_constructor,
    run_executor,
)
from methods.metric_implementer.experiments.v14_value_bound import (
    plugin_binary_mutual_information,
)


def test_mechanical_planted_labels_are_exact():
    for index in range(96):
        observed = [int(row["truth"](_planted_text(index))) for row in PLANTED_CRITERIA]
        expected = [(index >> shift) & 1 for shift in range(5, -1, -1)]
        assert observed == expected


def test_reference_reliability_and_attenuation_are_finite():
    labels = np.vstack([
        np.ones((20, 3)), np.zeros((20, 3)),
        np.tile([1, 1, 0], (10, 1)), np.tile([0, 0, 1], (10, 1)),
    ]).astype(np.uint8)
    assert -1.0 <= fleiss_kappa(labels) <= 1.0
    fitted = dawid_skene_binary(labels)
    posterior = np.asarray(fitted["posterior_probability"])
    assert posterior.shape == (60,)
    assert np.all((posterior >= 0.0) & (posterior <= 1.0))


def test_reference_reliability_gate_uses_chance_adjusted_lower_bound():
    truth = np.asarray([0, 1] * 60, dtype=np.uint8)
    reliable = np.vstack([truth, truth, truth])
    accepted = reference_reliability_gate(reliable, seed=7, n_bootstraps=300)
    assert accepted["passed"] is True
    rng = np.random.default_rng(4)
    chance = rng.integers(0, 2, size=(3, 120), dtype=np.uint8)
    rejected = reference_reliability_gate(chance, seed=7, n_bootstraps=300)
    assert rejected["passed"] is False


def test_permutation_null_preserves_controls_and_is_reproducible():
    target = np.asarray([0, 1] * 30, dtype=np.uint8)
    prediction = target.copy()
    blind = np.zeros(60, dtype=np.uint8)
    shuffled = np.roll(target, 1)
    first = _permuted_ladder_values(
        target, [prediction], [blind], [shuffled], n_permutations=100, seed=9,
    )
    second = _permuted_ladder_values(
        target, [prediction], [blind], [shuffled], n_permutations=100, seed=9,
    )
    assert np.array_equal(first, second)
    assert permutation_pvalue(1.0, first)["p_greater_equal"] <= 2 / 101


def test_miller_madow_full_table_reduces_plugin_bias():
    target = np.asarray([0, 0, 1, 1] * 15, dtype=np.uint8)
    prediction = np.asarray([0, 1, 0, 1] * 15, dtype=np.uint8)
    assert miller_madow_mutual_information(target, prediction) < 0.0


def test_vectorized_permutation_mi_matches_scalar_definition():
    rng = np.random.default_rng(2)
    targets = rng.integers(0, 2, (7, 60), dtype=np.uint8)
    predictions = rng.integers(0, 2, (11, 60), dtype=np.uint8)
    expected = np.asarray([
        [plugin_binary_mutual_information(target, prediction) for prediction in predictions]
        for target in targets
    ])
    np.testing.assert_allclose(_binary_mi_matrix(targets, predictions), expected, atol=1e-15)


def test_fake_constructor_and_executor_cover_both_ladder_channels(tmp_path: Path):
    root = tmp_path / "ladder"
    metric_key = "task_R3_metric0"
    menu = [metric_key, *[f"task_R3_metric{i}" for i in range(1, 11)]]
    panel_core = {
        "indices": list(range(8)), "texts": [f"demo {i}" for i in range(8)],
        "target_labels": [0, 1] * 4, "target_quantized_labels": [0, 3] * 4,
    }
    panel = {**panel_core, "panel_sha256": _sha(panel_core)}
    payload = {
        "schema": "cr3-independent-ceiling-ladder-v1", "metric_key": metric_key,
        "task": "task", "level": "R3", "metric": "metric0", "noun": "text",
        "max_chars": 1024, "probe_sha256": "probe", "heldout_indices": list(range(60)),
        "heldout_texts": [f"heldout {i}" for i in range(60)],
        "reference_probe_texts": [f"reference {i}" for i in range(300)],
        "operational_target": [0, 1] * 30, "panels": [panel], "menu_keys": menu,
        "size11_keys": menu, "menu_descriptions": {key: f"criterion {key}" for key in menu},
        "forms_by_key": {key: [f"criterion {key}"] for key in menu},
        "target_description_payload_sha256": "description", "source_bootstrap_sha256": "source",
        "design_manifest_sha256": "design", "design_scientific_sha256": "scientific",
    }
    payload["freeze_sha256"] = _sha(payload)
    design_path = root / "designs" / metric_key / "ladder_design.json"
    design_path.parent.mkdir(parents=True)
    design_path.write_text(json.dumps(payload))
    pd.DataFrame([{
        "metric_key": metric_key, "task": "task",
        "path": str(design_path.relative_to(root)), "freeze_sha256": payload["freeze_sha256"],
        "menu_size": len(menu),
    }]).to_parquet(root / "design_index.parquet", index=False)

    constructor = run_constructor(root, metric_keys=[metric_key], fake=True)
    executor = run_executor(root, metric_keys=[metric_key], fake=True)
    assert constructor["n_metrics"] == executor["n_metrics"] == 1
    built = json.loads((root / "constructor" / metric_key / "constructor.json").read_text())
    assert set(built["c1"]) == {"full_task_bank", "size11"}
    assert {row["condition"] for row in built["c2_rules"]} == {
        "canonical", "shuffled", "blind",
    }
    assert {row["condition"] for row in built["c3_rules"]} == {"canonical", "shuffled"}
