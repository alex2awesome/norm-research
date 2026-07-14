#!/usr/bin/env python
"""Score the frozen intact articulation bank with Llama-3.2-1B on public folds."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from methods.codability.experiments.build_fresh_item_partitions import sha256_file, text_sha256
from methods.codability.experiments.policy_data import (
    PUBLIC_DEVELOPMENT_PARTITIONS,
    require_partition,
)
from methods.metric_implementer import config as cfgmod
from methods.metric_implementer.experiments import alpha_probe as ap
from methods.metric_implementer.vllm_backend import make_judge_backend


MODEL = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_JOB = "llama1_adjacent"


def _items(packet_root: str | Path, partition: str) -> list[dict]:
    path = Path(packet_root) / "humor" / "items" / f"{partition}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _model_tag(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", Path(model.rstrip("/")).name)


def run(*, bank_path: str, packet_root: str, target_manifest_path: str,
        out_root: str, model: str = MODEL, model_job: str = MODEL_JOB,
        fake: bool = False) -> dict:
    bank = json.loads(Path(bank_path).read_text())
    allowed_status = {
        "frozen-before-1b-executor-public-scoring",
        "frozen-before-8b-executor-public-scoring",
    }
    if bank.get("status") not in allowed_status:
        raise ValueError("within-family scale-pair bank is not frozen")
    if len(bank["partitions"]) != len(set(bank["partitions"])):
        raise ValueError("scale-pair partitions must be unique")
    for partition in bank["partitions"]:
        require_partition(
            partition,
            allowed=PUBLIC_DEVELOPMENT_PARTITIONS,
            operation="within-family scale-pair scoring",
        )
    target_manifest = json.loads(Path(target_manifest_path).read_text())
    readout_template = target_manifest["readout_template"]
    config = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), "humor")
    if fake:
        config.vllm_fake = True
    backend = make_judge_backend(model, config, temperature=None)
    cell = bank["cells"][0]
    outputs = []
    for partition in bank["partitions"]:
        items = _items(packet_root, partition)
        hashes = [row["text_sha256"] for row in items]
        texts = [row["text"] for row in items]
        score_rows, meta = [], []
        for arm in cell["arms"]:
            for form in arm["forms"]:
                scores = np.asarray(ap.signature(
                    backend, form["prompt"], texts, config.max_text_chars,
                    template=readout_template), float)
                if not np.isfinite(scores).all():
                    raise ValueError(f"non-finite 1B score for {partition}/{arm['id']}/{form['id']}")
                score_rows.append(scores)
                meta.append({
                    "cell_id": cell["id"], "domain": "humor", "gi": cell["gi"],
                    "construct": cell["construct"], "arm_id": arm["id"],
                    "channel": arm["channel"], "provenance": arm["provenance"],
                    "control_for": arm["control_for"],
                    "semantic_content_word_count": arm["semantic_content_word_count"],
                    "form": form["id"], "prompt_sha256": form["prompt_sha256"],
                })
        directory = Path(out_root) / partition / model_job
        directory.mkdir(parents=True, exist_ok=True)
        out = directory / f"grid_humor_adjacent_{_model_tag(model)}_rep0.npz"
        np.savez_compressed(
            out,
            scores=np.asarray(score_rows),
            meta=np.asarray([json.dumps(row, sort_keys=True) for row in meta], dtype=object),
            probe_sha256=np.asarray(hashes),
            probe_partition=np.asarray([partition] * len(hashes)),
            reader=model,
            model_job_id=model_job,
            role="small",
            phase="adjacent_public",
            repetition=0,
            source_artifact_sha256=sha256_file(bank_path),
            isolated_partition=partition,
            readout_template_sha256=text_sha256(readout_template),
        )
        sidecar = out.with_suffix(".json")
        sidecar.write_text(json.dumps({
            "schema": "adjacent_scale_isomorphism_scores/v1",
            "status": "public-development-only",
            "partition": partition,
            "model": model,
            "model_family": bank["model_family"],
            "target": bank["target_policy"],
            "n_items": len(hashes),
            "n_arms": len(cell["arms"]),
            "n_forms": 3,
            "bank_sha256": sha256_file(bank_path),
            "readout_template_sha256": text_sha256(readout_template),
            "lockbox_status": "not read or authorized",
        }, indent=1))
        outputs.append({"partition": partition, "npz": str(out),
                        "sidecar": str(sidecar), "sha256": sha256_file(out)})
    return {"schema": "adjacent_scale_isomorphism_execution/v1",
            "bank_sha256": sha256_file(bank_path), "model": model,
            "model_job_id": model_job, "outputs": outputs,
            "lockbox_status": "not read or authorized"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bank", required=True)
    parser.add_argument("--packet-root", required=True)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--model-job", default=MODEL_JOB)
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()
    result = run(bank_path=args.bank, packet_root=args.packet_root,
                 target_manifest_path=args.target_manifest, out_root=args.out_root,
                 model=args.model, model_job=args.model_job, fake=args.fake)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
