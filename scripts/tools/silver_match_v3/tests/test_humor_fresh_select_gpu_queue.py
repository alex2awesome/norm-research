import json
import sys
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.freeze_humor_fresh_select_gpu_queue import (
    main as freeze_main,
)
from scripts.tools.silver_match_v3.run_humor_fresh_select_gpu_queue import (
    validate_queue,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_single_gpu_queue_is_sequential_backend_pure_and_hash_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    python = tmp_path / "python"
    python.write_text("frozen-python\n", encoding="utf-8")
    prompt, addon = tmp_path / "adjudicator.txt", tmp_path / "adjudicator-addon.txt"
    verifier_prompts = [tmp_path / f"verifier-{index}.txt" for index in range(2)]
    for index, path in enumerate([prompt, addon, *verifier_prompts]):
        path.write_text(f"prompt-{index}\n", encoding="utf-8")

    identity = tmp_path / "identity.json"
    _write_json(
        identity,
        {
            "task": "humor",
            "role": "select",
            "status": "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES",
        },
    )
    pack = tmp_path / "pack"
    pack.mkdir()
    candidates = pack / "candidates.top50.jsonl"
    with candidates.open("w", encoding="utf-8") as handle:
        for row in range(300):
            handle.write(
                json.dumps(
                    {
                        "norm_uid": f"u{row:03d}",
                        "candidates": [
                            {"metric_id": f"a{metric}"} for metric in range(50)
                        ],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    validation = pack / "validation.json"
    _write_json(
        validation,
        {
            "task": "humor",
            "gepa_role": "select",
            "truth_hidden": True,
            "count": 300,
            "candidate_k": 50,
            "outputs": {"candidates": {"sha256": sha256_file(candidates)}},
            "inputs": {"identity_freeze": {"sha256": sha256_file(identity)}},
        },
    )
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "truth_or_label_fields_in_manifest": False,
            "source_pack": {"validation_sha256": sha256_file(validation)},
        },
    )
    selection = tmp_path / "selection.json"
    _write_json(
        selection,
        {
            "task": "humor",
            "adjudicator_test_consumed": False,
            "chosen": {
                "name": "r1",
                "prompt_component_sha256": {
                    str(prompt): sha256_file(prompt),
                    str(addon): sha256_file(addon),
                },
            },
        },
    )
    r5 = tmp_path / "r5" / "FREEZE.json"
    _write_json(
        r5,
        {
            "status": "FROZEN_BEFORE_VERIFIER_GEPA_INFERENCE",
            "permanent_blind_consumed": False,
            "prompt": {
                "components": [
                    {"path": str(path), "sha256": sha256_file(path)}
                    for path in verifier_prompts
                ]
            },
        },
    )
    model = tmp_path / "model" / "snapshot-revision"
    model.mkdir(parents=True)
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ):
        (model / name).write_text(name + "\n", encoding="utf-8")
    _write_json(
        model / "model.safetensors.index.json",
        {
            "weight_map": {
                "layer.0": "model-00001-of-00002.safetensors",
                "layer.1": "model-00002-of-00002.safetensors",
            }
        },
    )
    output = tmp_path / "outputs"
    queue = tmp_path / "queue.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_humor_fresh_select_gpu_queue",
            "--repo",
            str(repo),
            "--python",
            str(python),
            "--pack-root",
            str(pack),
            "--execution-manifest",
            str(manifest),
            "--identity-freeze",
            str(identity),
            "--adjudicator-selection",
            str(selection),
            "--adjudicator-prompt",
            str(prompt),
            "--adjudicator-addon",
            str(addon),
            "--r5-freeze",
            str(r5),
            "--verifier-prompt",
            str(verifier_prompts[0]),
            "--verifier-prompt",
            str(verifier_prompts[1]),
            "--model-snapshot",
            str(model),
            "--gpu-id",
            "7",
            "--output-root",
            str(output),
            "--queue-output",
            str(queue),
        ],
    )
    freeze_main()
    payload = json.loads(queue.read_text(encoding="utf-8"))
    validate_queue(payload)
    assert payload["gpu_policy"] == {
        "physical_gpu_ids": [7],
        "maximum_concurrent_gpus": 1,
        "global_gpu_count_gate_applied": False,
    }
    assert len(payload["stages"]) == 6
    assert all(not stage["parallel"] for stage in payload["stages"])
    assert all(len(stage["cells"]) == 1 for stage in payload["stages"])
    assert all(
        cell["module"].startswith("scripts.tools.silver_match_v3.")
        for stage in payload["stages"]
        for cell in stage["cells"]
    )
    assert {
        cell["cuda_visible_devices"]
        for stage in payload["stages"]
        for cell in stage["cells"]
        if cell["cuda_visible_devices"] is not None
    } == {"7"}

    prompt.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_queue(payload)
