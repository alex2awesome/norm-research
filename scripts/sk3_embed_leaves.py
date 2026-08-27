"""Embed leaf-rubric names on sk3 with bge-large. One .npy per task, row order
matching scripts/export_leaf_names.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

WORK = Path("/lfs/skampere3/0/alexspan/norm_embed")
INP = WORK / "_sk3_leaf_input.jsonl"
OUTD = WORK / "out"
MODEL = "BAAI/bge-large-en-v1.5"


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[tuple[int, str]]] = {}
    for line in INP.open():
        r = json.loads(line)
        groups.setdefault(r["task"], []).append((r["idx"], r["name"]))

    print(f"loading {MODEL} ...", flush=True)
    model = SentenceTransformer(MODEL, device="cuda")
    for task, items in sorted(groups.items()):
        items.sort()
        texts = [t if t else " " for _, t in items]
        emb = model.encode(texts, batch_size=256, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True)
        np.save(OUTD / f"emb_bge_leafname_{task}.npy", emb.astype(np.float32))
        print(f"  {task}: {emb.shape}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
