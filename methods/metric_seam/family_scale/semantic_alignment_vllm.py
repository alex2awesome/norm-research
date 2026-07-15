#!/usr/bin/env python3
"""Run the frozen family-scale semantic alignments with an offline vLLM model.

This is an independent-instrument replication of the Sonnet alignment pass.
It consumes already compiled, metric-text-only requests and writes raw model
responses plus a provenance manifest. Scoring remains a separate CPU step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time


DEFAULT_MODEL = (
    "/lfs/skampere3/0/alexspan/merged_models/"
    "Llama-3.3-70B-FP8-with-tokenizer"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    # macOS tar archives can contain AppleDouble ``._*`` sidecars.  They are
    # metadata, not frozen requests, and must never enter the model batch.
    paths = sorted(
        path
        for path in args.requests_dir.glob("*_request.json")
        if not path.name.startswith("._")
    )
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit("no alignment request JSON files found")
    requests = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for path, request in zip(paths, requests):
        if request.get("schema") != "metric-seam.three-fleet-semantic-alignment-request.v1":
            raise ValueError(f"unexpected request schema: {path}")
        if request.get("status") != "compiled_for_exactly_one_alignment_call":
            raise ValueError(f"request is not frozen pre-call: {path}")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="auto",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=False,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    conversations = [
        [
            {"role": "system", "content": str(request["system_prompt"])},
            {"role": "user", "content": str(request["user_prompt"])},
        ]
        for request in requests
    ]
    started = time.time()
    outputs = llm.chat(conversations, sampling, use_tqdm=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, request, output in zip(paths, requests, outputs):
        raw = output.outputs[0].text if output.outputs else ""
        out_path = args.output_dir / path.name.replace("_request.json", "_raw.txt")
        out_path.write_text(raw, encoding="utf-8")
        rows.append({
            "request_file": path.name,
            "request_sha256": request["request_sha256"],
            "source_request_model": request["model"],
            "raw_file": out_path.name,
            "raw_sha256": sha256(out_path),
            "finish_reason": output.outputs[0].finish_reason if output.outputs else None,
        })
    manifest = {
        "schema": "metric-seam.semantic-alignment-vllm-run.v1",
        "status": "raw_responses_complete",
        "host": socket.gethostname(),
        "model": args.model,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "n": len(rows),
        "elapsed_seconds": time.time() - started,
        "rows": rows,
        "claim_limits": [
            "This is an independent open-model semantic-alignment instrument, not external ground truth.",
            "Raw outputs are scored later against the same frozen structural contract.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": manifest["status"], "n": len(rows), "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
