"""Re-embed rubric-cluster + subtask texts with a local model on sk3.

Cross-check for the dispersion dynamic-range question: if a different embedding
family reproduces the per-task ordering found with text-embedding-3-small +
pca_top30, the ordering is a real signal rather than an OpenAI-model artifact.

Runs on ONE GPU (set CUDA_VISIBLE_DEVICES). Writes one .npy per (level, task),
row order matching scripts/export_cluster_texts.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

WORK = Path("/lfs/skampere3/0/alexspan/norm_embed")
INP = WORK / "_sk3_embed_input.jsonl"
OUTD = WORK / "out"
MODEL = "BAAI/bge-large-en-v1.5"


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for line in INP.open():
        r = json.loads(line)
        groups.setdefault((r["level"], r["task"]), []).append((r["idx"], r["text"]))

    print(f"loading {MODEL} ...", flush=True)
    model = SentenceTransformer(MODEL, device="cuda")

    for (level, task), items in sorted(groups.items()):
        items.sort()
        texts = [t if t else " " for _, t in items]
        emb = model.encode(texts, batch_size=128, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True)
        np.save(OUTD / f"emb_bge_{level}_{task}.npy", emb.astype(np.float32))
        print(f"  {level}/{task}: {emb.shape}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
