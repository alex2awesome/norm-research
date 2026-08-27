"""Merge-and-score fallback: fold a LoRA adapter into base weights as a full checkpoint.

Used only if LoRA x prompt_logprobs (teacher-forced) misbehaves on the installed vLLM —
the merged checkpoint then goes through the completely unmodified scoring path.

Includes the index-vs-shard verification (broken-merged-checkpoint landmine,
reference_fp8_vllm_sk3: a merged dir whose safetensors index references missing/truncated
shards loads into garbage '!' output instead of failing).

Usage:
  python -m methods.tacit_channels.channels.distill.merge_adapter \
      --base Qwen/Qwen2.5-7B-Instruct --adapter outputs/.../adapters/humor_x_n32 \
      --out /lfs/.../merged/humor_x_n32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify_checkpoint(out_dir: Path) -> None:
    """Every shard named by the safetensors index must exist and be non-trivially sized."""
    index_path = out_dir / "model.safetensors.index.json"
    if not index_path.exists():
        single = out_dir / "model.safetensors"
        if not single.exists() or single.stat().st_size < 1024:
            raise SystemExit(f"merged checkpoint invalid: no usable safetensors in {out_dir}")
        return
    index = json.loads(index_path.read_text())
    shards = sorted(set(index["weight_map"].values()))
    problems = []
    for shard in shards:
        p = out_dir / shard
        if not p.exists():
            problems.append(f"missing shard {shard}")
        elif p.stat().st_size < 1024:
            problems.append(f"suspiciously small shard {shard} ({p.stat().st_size} bytes)")
    if problems:
        raise SystemExit("merged checkpoint FAILED index-vs-shard verification:\n  "
                         + "\n  ".join(problems))
    print(f"index-vs-shard OK: {len(shards)} shards")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    print(f"loading base {args.base} ({args.dtype}, CPU merge)")
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    print("merging adapter into base weights")
    model = model.merge_and_unload()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(out_dir)

    prov = Path(args.adapter) / "adapter_provenance.json"
    if prov.exists():
        merged_prov = json.loads(prov.read_text())
        merged_prov["merged_from_adapter"] = str(args.adapter)
        (out_dir / "adapter_provenance.json").write_text(json.dumps(merged_prov, indent=2))

    verify_checkpoint(out_dir)
    print(f"merged checkpoint -> {out_dir}")


if __name__ == "__main__":
    main()
