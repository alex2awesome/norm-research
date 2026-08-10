"""Embed canonical rubric forms on sk3 with nvidia/llama-embed-nemotron-8b.

nemotron-embed is instruction-aware: an instruction prefix tells it what notion
of similarity to use. We give it a CLUSTERING instruction so that rubrics
measuring the same underlying property land close together. The same
instruction is prepended to every rubric (symmetric clustering, not retrieval).

Format (from the model card): f"Instruct: {instruction}\\nQuery: {text}".

One .npy per (bucket, task); row order matches the input jsonl.

Usage:
  python sk3_embed_nemotron.py --input canon_all_real_forms.jsonl
"""
from __future__ import annotations

import os

os.environ["HOME"] = "/lfs/skampere3/0/alexspan"
# Force-set (not setdefault): the sk3 shell preloads HF_HOME pointing at the
# READ-ONLY shared cache, which breaks trust_remote_code dynamic-module writes.
os.environ["HF_HOME"] = "/lfs/skampere3/0/alexspan/hf_cache"
os.environ["HF_MODULES_CACHE"] = "/lfs/skampere3/0/alexspan/hf_cache/modules"
os.environ["TRANSFORMERS_CACHE"] = "/lfs/skampere3/0/alexspan/hf_cache"
os.environ["XDG_CACHE_HOME"] = "/lfs/skampere3/0/alexspan/.cache"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = ("/lfs/skampere3/0/shared_hf_cache/models--nvidia--llama-embed-nemotron-8b"
         "/snapshots/1acaf42b890bafa464ef9a58d1c0db0dd26120d4")

# Clustering instruction: group rubrics by the underlying property they measure.
INSTRUCTION = ("Represent this evaluation rubric by the underlying property it "
               "measures, so that rubrics that measure the same property are "
               "close together and rubrics that measure different properties "
               "are far apart.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="jsonl {task,bucket,idx,canonical}")
    ap.add_argument("--outdir", default="/lfs/skampere3/0/alexspan/norm_embed/out")
    args = ap.parse_args()

    groups: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for line in open(args.input):
        r = json.loads(line)
        groups.setdefault((r.get("bucket", "general"), r["task"]), []).append(
            (r["idx"], r["canonical"]))

    print(f"loading {MODEL.split('/')[-3]} ...", flush=True)
    model = SentenceTransformer(
        MODEL, trust_remote_code=True,
        model_kwargs={"attn_implementation": "eager", "torch_dtype": "bfloat16"},
        tokenizer_kwargs={"padding_side": "left"})

    outd = Path(args.outdir)
    outd.mkdir(parents=True, exist_ok=True)
    for (bucket, task), items in sorted(groups.items()):
        items.sort()
        texts = [f"Instruct: {INSTRUCTION}\nQuery: {c if c else ' '}"
                 for _, c in items]
        emb = model.encode(texts, batch_size=64, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True)
        np.save(outd / f"emb_nemo_{bucket}_{task}.npy", emb.astype(np.float32))
        print(f"  {bucket}/{task}: {emb.shape}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
