import json
from types import SimpleNamespace

from scripts.tools.silver_match_v3.run_typed_lora_frozen_eval import (
    INFERENCE_SCHEMA,
    PREDICTION_SCHEMA,
    _artifact,
    freeze,
    score,
    sha256_file,
)


def _write_json(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_freeze_then_score_keeps_inference_prompt_only(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    selection = {
        "status": "SELECTED_ON_DEV_ONLY",
        "selection_split": "dev",
        "test_or_blind_data_read": False,
        "chosen_cumulative_exposure": 100,
        "chosen_dev_report": {
            "confidence_gate": {
                "minimum_confidence": "medium",
                "gate_feasible": True,
            }
        },
    }
    _write_json(adapter / "DEV_SELECTION.json", selection)
    report = {
        "status": "COMPLETE_DEV_SELECTED_ADAPTER_FRESH_RELOAD_VERIFIED",
        "selection": selection,
        "adapter": {
            "directory": str(adapter),
            "content": {
                "files": [
                    {
                        "relative_path": name,
                        "sha256": sha256_file(adapter / name),
                    }
                    for name in ("adapter_config.json", "adapter_model.safetensors")
                ]
            },
        },
    }
    report_path = tmp_path / "TRAINING_REPORT.json"
    _write_json(report_path, report)

    datasets = {}
    for role in ("test", "blind"):
        path = tmp_path / f"{role}.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "task": "humor",
                    "corpus": "c",
                    "norm_uid": f"{role}-1",
                    "source_group": f"g-{role}",
                    "split": role,
                    "view": "retrieval_order",
                    "gradient_eligible": False,
                    "decision": "MATCH",
                    "metric_id": "a1",
                    "truth_decision": "MATCH",
                    "target_relation": "EXACT",
                    "candidate_metric_ids": ["a1", "a2"],
                    "messages": [
                        {"role": "user", "content": "prompt"},
                        {"role": "assistant", "content": "secret gold rationale"},
                    ],
                }
            ],
        )
        datasets[role] = path
    root = tmp_path / "frozen"
    freeze(
        SimpleNamespace(
            test_dataset=str(datasets["test"]),
            test_sha256=sha256_file(datasets["test"]),
            blind_dataset=str(datasets["blind"]),
            blind_sha256=sha256_file(datasets["blind"]),
            adapter=str(adapter),
            training_report=str(report_path),
            output_root=str(root),
        )
    )
    prompt_text = (root / "test.prompts.jsonl").read_text(encoding="utf-8")
    assert "secret gold rationale" not in prompt_text
    assert '"role": "assistant"' not in prompt_text

    predictions_path = tmp_path / "predictions.jsonl"
    prediction_rows = []
    for role in ("test", "blind"):
        prediction = {
            "decision": "MATCH",
            "metric_id": "a1",
            "confidence": "high",
            "reason": "fits",
            "parse_error": None,
            "raw_response": None,
        }
        prediction_rows.append(
            {
                "schema_version": PREDICTION_SCHEMA,
                "split": role,
                "norm_uid": f"{role}-1",
                "base": prediction,
                "lora": prediction,
            }
        )
    _write_jsonl(predictions_path, prediction_rows)
    freeze_path = root / "FREEZE.json"
    meta_path = tmp_path / "INFERENCE_META.json"
    _write_json(
        meta_path,
        {
            "schema_version": INFERENCE_SCHEMA,
            "status": "COMPLETE_TRUTH_BLIND_PAIRED_INFERENCE",
            "test_or_blind_gold_read": False,
            "freeze": _artifact(freeze_path),
            "predictions": {**_artifact(predictions_path), "count": 2},
        },
    )
    output = tmp_path / "score.json"
    result = score(
        SimpleNamespace(
            freeze=str(freeze_path),
            inference_meta=str(meta_path),
            output=str(output),
        )
    )
    assert result["systems"]["lora"]["pooled"]["dev_selected_match_gate"] == {
        "minimum_confidence": "medium",
        "accepted_count": 2,
        "correct_exact_leaf_count": 2,
        "gold_match_count": 2,
        "exact_precision": 1.0,
        "exact_precision_wilson_95_lower": result["systems"]["lora"]["pooled"][
            "dev_selected_match_gate"
        ]["exact_precision_wilson_95_lower"],
        "exact_recall": 1.0,
        "exact_f_beta_0_5": 1.0,
        "abstention_rate": 0.0,
    }
