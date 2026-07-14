#!/usr/bin/env python3
"""Hash-freeze the exact sk2 Gemma model and similarity runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    host = socket.gethostname().split(".", 1)[0].lower()
    if host not in {"sk2", "skampere2"} and not host.startswith("skampere2-"):
        raise RuntimeError(f"model inventory is sk2-only; refusing {socket.gethostname()}")
    model = Path(args.model).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if "skampere2" not in str(model):
        raise ValueError(f"model path is not on sk2 storage: {model}")
    files = sorted(
        path for path in model.iterdir()
        if path.is_file() and path.name in {
            "config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "model.safetensors.index.json", "chat_template.jinja",
        } or path.is_file() and path.suffix == ".safetensors"
    )
    if not files or not any(path.suffix == ".safetensors" for path in files):
        raise ValueError("model snapshot is incomplete")
    versions = subprocess.check_output(
        [args.python, "-c", "import torch,transformers,peft,accelerate; print(torch.__version__,transformers.__version__,peft.__version__,accelerate.__version__)"],
        text=True,
    ).strip().split()
    payload = {
        "schema_version": "gemma4-similarity-sk2-model-inventory-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "model": str(model),
        "runtime": dict(zip(("torch", "transformers", "peft", "accelerate"), versions)),
        "files": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in files
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "files": len(files), "bytes": sum(path.stat().st_size for path in files)}))


if __name__ == "__main__":
    main()
