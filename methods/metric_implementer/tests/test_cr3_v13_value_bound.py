"""CPU/fake-backend checks for the frozen CR-3 v13.1 two-channel engine."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from methods.metric_implementer.config import ImplementerConfig
from methods.metric_implementer.experiments.behavioral_value_channel import (
    BEHAVIORAL_ARMS,
    _redact_verbatim_example_shingles,
    contains_verbatim_example,
    evaluate_behavioral_state_tables,
)
from methods.metric_implementer.experiments.assemble_v13_release_results import (
    assemble_release_results,
)
from methods.metric_implementer.experiments.cr3_sampled_value_certify import (
    VALUE_BOUND_DESIGN_SCHEMA,
    build_value_bound_design_manifest,
    enumerate_exact_pool_values,
    evaluate_mcq_state_tables_v13_1,
    fixed_prefix_capture_recapture,
)
from methods.metric_implementer.experiments.cr3_sampled_value_certify import (
    _load_production_codebook,
    _pool_positive_quotas,
)
from methods.metric_implementer.experiments.consolidate_v13_value_lanes import (
    consolidate_lane_roots,
)
from methods.metric_implementer.experiments.run_v13_value_campaign import (
    main as campaign_main,
    select_tier_a_upgrades,
)
from methods.metric_implementer.experiments.select_v13_tier_a_upgrades import (
    freeze_tier_a_upgrades,
)
from methods.metric_implementer.experiments.v13_value_cache import ValueCache, cache_key
from methods.metric_implementer.vllm_backend import FakeVLLM


FIXTURE = (
    Path(__file__).parent / "fixtures" / "cr3_v12_humor_metric50_subset"
)
METRIC = "humor_R3_metric50"


def test_no_verbatim_fallback_removes_only_copied_spans():
    example = "one two three four five six seven eight nine ten eleven twelve thirteen"
    rule = f"Prefer concise setups. {example}. Reward a surprising final contrast."
    cleaned = _redact_verbatim_example_shingles(rule, [example])
    assert not contains_verbatim_example(cleaned, [example])
    assert "Prefer concise setups" in cleaned
    assert "Reward a surprising final contrast" in cleaned
    assert "example-specific phrase omitted" in cleaned


def _codebook():
    manifest, _ = _load_production_codebook(
        FIXTURE / "mcq_codebooks" / "humor.json", assets_root=FIXTURE
    )
    return manifest


def _one_panel_design(channel: str):
    design = build_value_bound_design_manifest(
        _codebook(), target_metric_key=METRIC, heldout_size=60
    )
    design = copy.deepcopy(design)
    design["tiers"]["B"]["active_pool_ids"] = [design["pools"][0]["pool_id"]]
    if channel == "mcq":
        design["tiers"]["B"]["mcq_panels_per_pool"] = 1
    else:
        design["tiers"]["B"]["behavioral_panels_per_pool"] = 1
    return design


def _fake_backend(model="fake"):
    cfg = ImplementerConfig()
    cfg.vllm_fake = True
    return FakeVLLM(model, "judge", cfg, 0.0)


def test_shared_design_freezes_six_disjoint_pools_and_heldout():
    codebook = _codebook()
    first = build_value_bound_design_manifest(codebook, target_metric_key=METRIC)
    second = build_value_bound_design_manifest(codebook, target_metric_key=METRIC)
    assert first == second
    assert first["schema"] == VALUE_BOUND_DESIGN_SCHEMA
    assert len(first["pools"]) == 6
    pool_indices = [index for pool in first["pools"] for index in pool["indices"]]
    assert len(pool_indices) == len(set(pool_indices)) == 72
    assert set(pool_indices).isdisjoint(first["heldout"]["indices"])
    for pool in first["pools"]:
        assert len(pool["indices"]) == 12
        assert len(pool["mcq_panels"]) == 12
        assert len(pool["behavioral_panels"]) == 4
        assert all(len(panel["fixed_teaching_indices"]) == 8 for panel in pool["mcq_panels"])
        assert all(
            len(panel["fixed_teaching_indices"]) == 6
            for panel in pool["behavioral_panels"]
        )


def test_pool_freezing_does_not_qualify_on_target_balance():
    assert _pool_positive_quotas(10, 110) == [1, 1, 1, 1, 1, 1]
    assert _pool_positive_quotas(120, 0) == [12, 12, 12, 12, 12, 12]


def test_exact_pool_enumeration_cap_dominates_every_observed_value():
    design = _one_panel_design("behavioral")
    state_values = np.linspace(0.0, 0.9, 64, dtype=float)[None, :]
    rng = np.random.default_rng(13)
    signatures = rng.random((25, design["n_probes"]))
    result = enumerate_exact_pool_values(
        design, channel="behavioral", tier="B",
        state_values=state_values, signatures=signatures,
    )
    assert result["pool_pattern_values"].shape == (1, 4096)
    assert result["exact_structural_cap"] == pytest.approx(0.9)
    assert result["exact_structural_cap"] >= np.max(result["mean_prompt_value"])
    assert result["exact_structural_gap"] == pytest.approx(
        result["exact_structural_cap"] - result["achieved_value"]
    )


def test_fixed_prefix_cp_does_not_use_sequential_first_contacts():
    design = _one_panel_design("behavioral")
    aggregation = enumerate_exact_pool_values(
        design, channel="behavioral", tier="B",
        state_values=np.linspace(0.0, 1.0, 64)[None, :],
        signatures=np.zeros((2, design["n_probes"])),
    )
    prefix = np.zeros((5, design["n_probes"]), dtype=float)
    # The same unseen species occurs three times.  Relative to the fixed prefix all three
    # are Bernoulli successes; sequential first-contact counting would incorrectly report one.
    audit = np.ones((3, design["n_probes"]), dtype=float)
    report = fixed_prefix_capture_recapture(
        aggregation,
        process_streams=[{
            "family": "f", "discovery_prefix_signatures": prefix,
            "audit_suffix_signatures": audit,
        }],
    )
    row = report["family_rows"]["f"]
    assert row["per_pool"][0]["n_audit_draws_unseen_relative_to_fixed_prefix"] == 3
    assert row["joint_pattern"]["n_audit_draws_unseen_relative_to_fixed_prefix"] == 3
    assert report["premises"]["sequential_first_contacts_not_treated_as_binomial"] is True
    assert set(report["horizons"]) == {"100", "300"}


def test_value_cache_repeated_keys_must_be_identical(tmp_path):
    key = cache_key("cell", {"x": 1})
    with ValueCache(tmp_path / "cache.sqlite") as cache:
        assert cache.get(key) is None
        cache.put(key, "cell", {"value": 0.25})
        assert cache.put(key, "cell", {"value": 0.25}) == {"value": 0.25}
        with pytest.raises(RuntimeError, match="non-identical"):
            cache.put(key, "cell", {"value": 0.50})


def test_mcq_fake_backend_one_panel_is_complete_and_non_disclosing(tmp_path):
    codebook = _codebook()
    design = _one_panel_design("mcq")
    with ValueCache(tmp_path / "mcq.sqlite") as cache:
        result = evaluate_mcq_state_tables_v13_1(
            _fake_backend(), codebook_manifest=codebook, design_manifest=design,
            target_metric_key=METRIC, tier="B", constructor_revision="fake-rev",
            cache=cache, query_batch_size=128,
        )
        repeated = evaluate_mcq_state_tables_v13_1(
            _fake_backend(), codebook_manifest=codebook, design_manifest=design,
            target_metric_key=METRIC, tier="B", constructor_revision="fake-rev",
            cache=cache, query_batch_size=128,
        )
    assert result["state_values"].shape == (1, 256)
    assert np.all(np.isfinite(result["state_values"]))
    assert repeated["cache_misses"] == 0 and repeated["cache_hits"] == 256
    assert result["non_disclosure"]["candidate_prompt_text_passed_to_query_builder"] is False


class _PlantedConstructor:
    def __init__(self, target_by_text):
        self.target_by_text = target_by_text

    def generate_batch(self, prompts, **_kwargs):
        outputs = []
        for prompt in prompts:
            examples = re.findall(r"\[label=(\d)\]\n```\n(.*?)\n```", prompt, re.DOTALL)
            if examples and all(
                int(label) == self.target_by_text[text] for label, text in examples
            ):
                outputs.append("TARGET_RULE")
            else:
                outputs.append("CONSTANT_RULE")
        return outputs


class _PlantedExecutor:
    def __init__(self, target_by_text):
        self.target_by_text = target_by_text

    def score_binary_constrained(self, prompts, **_kwargs):
        values = []
        for prompt in prompts:
            rubric = re.search(r"Criterion:\n(.*?)\n\nText:", prompt, re.DOTALL).group(1)
            text = re.search(r"\n\nText:\n(.*?)\n\nDoes the text", prompt, re.DOTALL).group(1)
            values.append(float(self.target_by_text[text]) if rubric == "TARGET_RULE" else 0.0)
        return values


def test_behavioral_fake_induce_execute_planted_rule_and_both_arms(tmp_path):
    codebook = _codebook()
    design = _one_panel_design("behavioral")
    target = np.load(
        codebook["metrics"][METRIC]["bootstrap_path"], allow_pickle=True
    )
    target_by_text = {
        str(text)[:int(codebook["reconstruction_max_chars"])]: int(score > 0.5)
        for text, score in zip(target["probe_texts"], target["target"])
    }
    with ValueCache(tmp_path / "behavioral.sqlite") as cache:
        result = evaluate_behavioral_state_tables(
            _PlantedConstructor(target_by_text), _PlantedExecutor(target_by_text),
            codebook_manifest=codebook, design_manifest=design,
            target_metric_key=METRIC, tier="B", constructor_revision="constructor-rev",
            executor_revision="executor-rev", executor_readout_id="binary-v3",
            cache=cache, query_batch_size=512,
        )
    assert set(result["arms"]) == set(BEHAVIORAL_ARMS)
    panel = result["active_design"]["panels"][0]
    target_state = int(panel["fixed_teaching_target_state"])
    for arm in BEHAVIORAL_ARMS:
        table = result["arms"][arm]["state_values"]
        assert table.shape == (1, 64) and np.all(np.isfinite(table))
        assert table[0, target_state] == pytest.approx(result["target_entropy_bits"])
        assert result["arms"][arm]["blind_mutual_information_bits"] == pytest.approx(0.0)
    assert result["non_disclosure"]["candidate_prompt_text_passed_to_inducer"] is False


def test_unified_launcher_fake_both_channels_writes_frozen_outputs(tmp_path):
    metrics_manifest = {
        "schema": "cr3-value-bound-metrics-v13.1",
        "metrics": [{
            "task": "humor", "level": "R3", "metric": "50", "metric_key": METRIC,
            "codebook_path": str(FIXTURE / "mcq_codebooks" / "humor.json"),
            "codebook_layout": "production", "assets_root": str(FIXTURE),
            "candidate_bank_path": str(
                FIXTURE / METRIC / "historical" / "scored.npz"
            ),
        }],
    }
    manifest_path = tmp_path / "metrics.json"
    manifest_path.write_text(json.dumps(metrics_manifest), encoding="utf-8")
    out_root = tmp_path / "campaign"
    assert campaign_main([
        "--channels", "mcq", "behavioral",
        "--constructor-models", "fake-constructor",
        "--metrics-manifest", str(manifest_path),
        "--tier", "B", "--out-root", str(out_root), "--fake-backends",
        "--query-batch-size", "1024", "--preflight-one-panel",
    ]) == 0
    results = pd.read_parquet(out_root / "results.parquet")
    assert len(results) == 2 and set(results["channel"]) == {"mcq", "behavioral"}
    for channel in ("mcq", "behavioral"):
        directory = (
            out_root / "tier_B" / f"humor__R3__{METRIC}" /
            "fake-constructor" / channel
        )
        assert (directory / "design_manifest.json").exists()
        assert (directory / "state_tables.npz").exists()
        assert (directory / "certificate.json").exists()
        assert (directory / "prompt_values.parquet").exists()
        certificate = json.loads((directory / "certificate.json").read_text())
        assert certificate["exact_structural_cap"] >= certificate["achieved_value"]
    campaign = json.loads((out_root / "campaign_manifest.json").read_text())
    assert campaign["n_results"] == 2 and campaign["preflight_one_panel"] is True


def test_tier_a_upgrade_selection_keeps_channel_units_separate():
    entries = [
        {"task": "task", "level": "R3", "metric": str(index),
         "metric_key": f"metric-{index}"}
        for index in range(12)
    ]
    rows = []
    for index, entry in enumerate(entries):
        for constructor in ("c1", "c2"):
            for channel in ("mcq", "behavioral"):
                rows.append({
                    **entry, "constructor": constructor, "channel": channel,
                    "exact_structural_gap": (
                        float(index) if channel == "behavioral" else 1000.0 - index
                    ),
                    "cross_channel_spearman": float(index) / 12.0,
                })
    chosen, provenance = select_tier_a_upgrades(rows, entries)
    keys = [entry["metric_key"] for entry in chosen]
    assert keys[:5] == ["metric-11", "metric-10", "metric-9", "metric-8", "metric-7"]
    assert keys[5:] == ["metric-0", "metric-1", "metric-2", "metric-3", "metric-4"]
    assert provenance["channels_numerically_combined"] is False


def test_lane_consolidation_requires_complete_unique_matrix(tmp_path):
    lane_roots = []
    for lane_index, constructor in enumerate(("c1", "c2")):
        root = tmp_path / f"lane-{lane_index}"
        root.mkdir()
        rows = [{
            "schema": "cr3-value-bound-results-v13.1", "tier": "A",
            "task": "humor", "level": "R3", "metric_key": "m0",
            "constructor": constructor, "channel": channel,
        } for channel in ("mcq", "behavioral")]
        pd.DataFrame(rows).to_parquet(root / "results.parquet", index=False)
        (root / "campaign_manifest.json").write_text(json.dumps({
            "schema": "cr3-value-bound-campaign-v13.1",
        }), encoding="utf-8")
        lane_roots.append(root)
    combined = consolidate_lane_roots(
        lane_roots, out_root=tmp_path / "combined",
        expected_constructors=["c1", "c2"],
    )
    assert len(combined) == 4
    assert (tmp_path / "combined" / "results.parquet").exists()
    assert set(combined["artifact_root"]) == set(map(str, lane_roots))


def test_cross_lane_tier_a_upgrade_manifest_is_selected_once(tmp_path):
    entries = [
        {
            "task": "task", "level": "R3", "metric": str(index),
            "metric_key": f"metric-{index}",
            "codebook_path": "unused-codebook.json",
        }
        for index in range(12)
    ]
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({
        "schema": "cr3-value-bound-metrics-v13.1", "metrics": entries,
    }), encoding="utf-8")
    rows = []
    for index, entry in enumerate(entries):
        for constructor in ("c1", "c2"):
            for channel in ("mcq", "behavioral"):
                rows.append({
                    **entry, "schema": "cr3-value-bound-results-v13.1", "tier": "B",
                    "constructor": constructor, "channel": channel,
                    "exact_structural_gap": float(index),
                    "cross_channel_spearman": float(index) / 12.0,
                })
    results_path = tmp_path / "results.parquet"
    pd.DataFrame(rows).to_parquet(results_path, index=False)
    report = freeze_tier_a_upgrades(
        results_path=results_path, metrics_manifest_path=metrics_path,
        out_root=tmp_path / "upgrade",
    )
    upgrade = json.loads(Path(report["metrics_manifest_path"]).read_text())
    assert report["n_selected"] == 10
    assert len(upgrade["metrics"]) == 10
    assert upgrade["auto_upgrade_tier_a"] is False
    assert upgrade["selection_provenance"]["channels_numerically_combined"] is False


def test_release_result_assembly_keeps_stages_and_channel_units_separate(tmp_path):
    stages = {}
    for stage, tier in (("wave1", "A"), ("breadth", "B")):
        root = tmp_path / stage
        root.mkdir()
        pd.DataFrame([{
            "schema": "cr3-value-bound-results-v13.1", "tier": tier,
            "task": "humor", "level": "R3", "metric_key": "m",
            "constructor": "c", "channel": channel,
            "achieved_value": float(index),
        } for index, channel in enumerate(("mcq", "behavioral"))]).to_parquet(
            root / "results.parquet", index=False
        )
        stages[stage] = root
    combined = assemble_release_results(
        stages, out_root=tmp_path / "release", expected_results=4,
    )
    assert len(combined) == 4
    assert set(combined["campaign_stage"]) == {"wave1", "breadth"}
    manifest = json.loads((tmp_path / "release/campaign_manifest.json").read_text())
    assert manifest["mcq_and_behavioral_values_numerically_combined"] is False
