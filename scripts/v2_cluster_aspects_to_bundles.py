"""Cluster the 218 R2 aspects into ~30 bundles of 5-8 aspects each.

Used for multi-rubric batched judging — related aspects scored in one call.
Embeds aspect name + description, runs k-means with k=30.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import openai
from sklearn.cluster import KMeans


def embed_texts(texts, batch=128, model="text-embedding-3-small"):
    client = openai.OpenAI()
    out = []
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=model, input=texts[i:i+batch])
        out.extend([d.embedding for d in resp.data])
    return np.array(out)


def main():
    v2 = Path("runs/validity_full/full_v2")
    aspects = json.loads((Path("runs/validity_full/full_v1") /
                          "r2_aspects.json").read_text())

    texts = [f"{a['name']}: {a['description']}" for a in aspects]
    print(f"embedding {len(texts)} aspects...")
    embs = embed_texts(texts)
    print(f"  shape: {embs.shape}")

    # Aim for ~6-7 aspects per bundle → k=32
    k = 32
    print(f"k-means k={k}...")
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(embs)
    labels = km.labels_

    # Build bundles
    bundles = defaultdict(list)
    for asp, lab in zip(aspects, labels):
        bundles[int(lab)].append(asp["aspect_id"])

    # Rebalance: split bundles >10, merge bundles <3
    final = []
    for bid, ids in bundles.items():
        if len(ids) <= 10:
            final.append(ids)
        else:
            # split into halves
            for i in range(0, len(ids), 8):
                final.append(ids[i:i+8])

    # Merge tiny bundles (<3) with smallest neighbor
    aspect_to_bundle = {}
    for i, ids in enumerate(final):
        for aid in ids:
            aspect_to_bundle[aid] = i

    print(f"  {len(final)} bundles")
    sizes = sorted([len(b) for b in final])
    print(f"  bundle sizes: min={sizes[0]} median={sizes[len(sizes)//2]} max={sizes[-1]}")

    aspect_by_id = {a["aspect_id"]: a for a in aspects}
    out_bundles = []
    for bid, ids in enumerate(final):
        out_bundles.append({
            "bundle_id": f"b{bid}",
            "aspect_ids": ids,
            "aspects": [{"aspect_id": aid,
                          "name": aspect_by_id[aid]["name"],
                          "description": aspect_by_id[aid]["description"]}
                         for aid in ids],
        })

    out_path = v2 / "judge_bundles.json"
    out_path.write_text(json.dumps(out_bundles, indent=1))
    print(f"wrote {out_path}")
    # Also save aspect->bundle index
    (v2 / "aspect_to_bundle.json").write_text(json.dumps(aspect_to_bundle, indent=1))

    # Print sample bundle
    print("\nsample bundle (b0):")
    for a in out_bundles[0]["aspects"]:
        print(f"  {a['aspect_id']}: {a['name']}")


if __name__ == "__main__":
    main()
