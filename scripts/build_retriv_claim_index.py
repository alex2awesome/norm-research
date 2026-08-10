#!/usr/bin/env python3
"""Build a retriv dense index over all patent claim-1 texts.

Uses the fine-tuned BGE-M3 anticipation model. Indexes claims from
patents_dataset.jsonl.gz (4.7M pre-grant pubs) + can be extended to
granted patents later.

Stores under retriv collection 'patent_claims_v1'.
"""
import argparse
import gzip
import json
import os
import sys

# retriv uses HOME dir; pin to /lfs
os.environ.setdefault("HOME", "/lfs/skampere3/0/alexspan")

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
JSONL = f"{BASE}/patents_dataset.jsonl.gz"
MODEL = "/lfs/skampere3/0/alexspan/norm-research/models/bge-m3-anticipation"
COLLECTION = "patent_claims_v1"


def yield_claims():
    """Stream JSONL → {id, text} dicts, one per app, claims field."""
    n = 0
    with gzip.open(JSONL, "rt") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            pid = str(d.get("pgpub_id", "")).strip()
            claims = (d.get("pg_claims") or "").strip()
            abstract = (d.get("pg_abstract") or "").strip()
            if not pid or not claims:
                continue
            yield {
                "id": pid,
                "text": claims[:4000],
                "abstract": abstract[:1000],
            }
            n += 1
            if n % 500_000 == 0:
                print(f"  yielded {n:,}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--collection", default=COLLECTION)
    p.add_argument("--model", default=MODEL)
    args = p.parse_args()

    from retriv import DenseRetriever
    print(f"Initializing DenseRetriever with model {args.model} ...")
    dr = DenseRetriever(
        index_name=args.collection,
        model=args.model,
        normalize=True,
        max_length=512,
        use_ann=True,
    )
    print("Indexing all claims ...")
    dr.index(
        collection=yield_claims(),
        show_progress=True,
        batch_size=128,
    )
    print(f"\nDone. Saved under retriv collection '{args.collection}'.")


if __name__ == "__main__":
    main()
