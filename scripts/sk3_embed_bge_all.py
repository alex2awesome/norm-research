"""Embed canonical rubric forms on sk3 with bge-large-en-v1.5, per (bucket,task).

Replaces the nemotron embedding: nemotron-embed with a long prepended
instruction collapsed (unrelated rubrics landed at cosine ~0.96 -- the constant
instruction dominates its avg-pooling). bge-large is properly discriminative
(random-pair cosine ~0.6). No instruction; plain symmetric embedding.

Output: emb_bge_<bucket>_<task>.npy, row order matching canon_all_real_forms.jsonl
"""
from __future__ import annotations

import os

os.environ["HOME"] = "/lfs/skampere3/0/alexspan"
os.environ["HF_HOME"] = "/lfs/skampere3/0/alexspan/hf_cache"
os.environ["HF_MODULES_CACHE"] = "/lfs/skampere3/0/alexspan/hf_cache/modules"
os.environ["XDG_CACHE_HOME"] = "/lfs/skampere3/0/alexspan/.cache"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

WORK = Path("/lfs/skampere3/0/alexspan/norm_embed")
INP = WORK / "canon_all_real_forms.jsonl"
OUTD = WORK / "out"
MODEL = ("/lfs/skampere3/0/shared_hf_cache/models--BAAI--bge-large-en-v1.5"
         "/snapshots")


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    model_path = next((MODEL + "/" + d) for d in __import__("os").listdir(MODEL))
    groups: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for line in INP.open():
        r = json.loads(line)
        groups.setdefault((r["bucket"], r["task"]), []).append(
            (r["idx"], r["canonical"]))

    print(f"loading bge-large from {model_path}", flush=True)
    model = SentenceTransformer(model_path, device="cuda")
    for (bucket, task), items in sorted(groups.items()):
        items.sort()
        texts = [t if t else " " for _, t in items]
        emb = model.encode(texts, batch_size=256, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True)
        np.save(OUTD / f"emb_bge_{bucket}_{task}.npy", emb.astype(np.float32))
        print(f"  {bucket}/{task}: {emb.shape}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
