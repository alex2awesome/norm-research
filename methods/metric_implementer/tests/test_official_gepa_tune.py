"""Schema/plumbing test for the official-GEPA decoder-tuning path (--phase tune).

The in-house bounded GEPA loop was deprecated 2026-07-19; ``run_decoder_tuning`` now wraps
official ``gepa.optimize``. This test exercises the wrapper with ``fake_backends=True`` and a
monkeypatched ``gepa.optimize`` (plus fake dev-contexts + fake reference scorer, so no GPU and
no network) and asserts the frozen output JSON has the ``v14-tune-official-gepa-v1`` schema and
the keys downstream ``build_production_freeze`` consumes.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import gepa

from methods.metric_implementer.experiments import run_v14_value_campaign as campaign
from methods.metric_implementer.experiments.run_v14_value_campaign import run_decoder_tuning


def _fake_contexts():
    return [{
        "metric_key": f"dev_metric_{index}",
        "noun": "item",
        "target_description": f"development target criterion {index}",
        "distractors": [
            {"description": f"distractor {index}a"},
            {"description": f"distractor {index}b"},
        ],
    } for index in range(8)]


def _fake_mcq_scores(reconstructor, *, templates, contexts, decoder_family,
                     constructor_revision, store=None, query_batch_size=1024):
    from methods.metric_implementer.experiments.v14_decoder_tuning import template_sha256
    rows = []
    for context in contexts:
        for template in templates:
            sha = template_sha256(template)
            for split, state, fitness, canonical in (
                ("search", 0, 0.5, True), ("heldout_prompt", 1, 0.4, False),
            ):
                rows.append({
                    "template_sha256": sha, "metric_key": str(context["metric_key"]),
                    "decoder_family": str(decoder_family), "trial": 0, "state": state,
                    "reference_split": split, "raw_lift": fitness, "blind_probability": 0.1,
                    "normalized_fitness": fitness,
                    "target_metric_id": str(context["metric_key"]),
                    "predicted_metric_id": str(context["metric_key"]),
                    "identification_ceiling_bits": 1.0,
                    "is_canonical_state": canonical,
                })
    return rows


def test_run_decoder_tuning_writes_official_gepa_schema(tmp_path, monkeypatch):
    winner = "WINNER decode using {noun} over {examples} {choices} {labels} contrastively"

    def fake_optimize(**kwargs):
        # Sanity: the wrapper must hand official gepa the expected optimize surface.
        assert set(kwargs["seed_candidate"]) == {"mcq_template"}
        assert kwargs["adapter"] is not None and callable(kwargs["reflection_lm"])
        assert kwargs["max_metric_calls"] == 4
        return SimpleNamespace(best_candidate={"mcq_template": winner})

    monkeypatch.setattr(gepa, "optimize", fake_optimize)
    monkeypatch.setattr(campaign, "_development_contexts", lambda _out_root: _fake_contexts())
    monkeypatch.setattr(campaign, "_backend", lambda model, *, fake: (object(), "rev"))
    monkeypatch.setattr(campaign, "release_resident_engines", lambda *a, **k: None)
    monkeypatch.setattr(campaign, "score_mcq_reference_templates", _fake_mcq_scores)

    result = run_decoder_tuning(
        out_root=tmp_path, channel="mcq", arm="unconstrained",
        decoder_models={"qwen": "fake-qwen"}, proposer_model="zai_anthropic:glm-5.2",
        fake_backends=True, physical_gpu_ids=[], query_batch_size=8, max_metric_calls=4,
    )

    assert result["schema"] == "v14-tune-official-gepa-v1"
    assert result["optimizer"] == "official-gepa"
    assert result["winner_template"] == winner
    # Keys build_production_freeze reads off each tuning trace must be present + valid.
    assert result["shared_across_decoder_families"] is True
    assert result["seed_template"] and result["freeze_sha256"]
    assert result["winner_template_sha256"] and isinstance(result["winner_report"], dict)
    assert result["reports"] and result["winner_template_sha256"] in result["reports"]

    out_path = Path(tmp_path) / "development" / "tuning" / "mcq.json"
    assert out_path.is_file()
    written = json.loads(out_path.read_text())
    assert written["schema"] == "v14-tune-official-gepa-v1"
    assert written["winner_template"] == winner
    assert written["winner_template_sha256"] in written["reports"]
    # proposals.jsonl log dir is provisioned in the output tree.
    assert (Path(tmp_path) / "development" / "tuning" / "mcq_official_gepa").is_dir()
