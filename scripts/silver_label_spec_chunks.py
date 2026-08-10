#!/usr/bin/env python3
"""LLM silver labeling for (claim, spec_chunk) pairs.

Given clean §102 training pairs and the spec chunk FAISS index:
  1. For each pair (rejected_claim, cited_pgpub_id):
     a. Look up ALL chunks belonging to that cited_pgpub_id
     b. Ask LLM (vLLM Qwen3.5 or Claude API): "does this chunk anticipate the
        claim?" — return YES/NO + 1-sentence reason
     c. Save labeled triples
  2. Also generates hard-negative labels: top-K chunks retrieved via FAISS
     from OTHER patents, score them against the same claim.

Output: processed/silver_labels.jsonl.gz
  {anchor_pgpub_id, anchor_text, candidate_source: "g"|"pg",
   candidate_doc_id, candidate_chunk_idx, candidate_text,
   relation: "positive"|"negative", llm_label: "yes"|"no"|"partial",
   llm_reason: str}

Use these labels to fine-tune the spec-aware bi-encoder.
"""
import argparse
import gzip
import json
import os
import sys
import time
from collections import defaultdict

import requests

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
CLEAN_102 = f"{BASE}/processed/clean_102_pairs.jsonl.gz"
META_FILE = f"{BASE}/processed/spec_embeddings/meta.jsonl"
OUT = f"{BASE}/processed/silver_labels.jsonl.gz"

VLLM_URL = "http://localhost:8001/v1/chat/completions"
MODEL_NAME = "qwen3.5-122b-a10b-fp8"


PROMPT = """You are a USPTO patent examiner. Determine whether the passage from a prior-art document anticipates an examined claim.

§102 anticipation requires every limitation of the examined claim to be disclosed in the prior art (in this single passage).

EXAMINED CLAIM:
{claim}

CANDIDATE PRIOR ART PASSAGE:
{passage}

Respond in this exact JSON format:
{{"label": "yes" | "partial" | "no", "reason": "<1-sentence explanation>"}}

label="yes" only if the passage discloses EVERY limitation of the claim.
label="partial" if the passage discloses some but not all limitations.
label="no" if the passage doesn't disclose the claim's invention at all."""


def llm_label(claim: str, passage: str) -> dict:
    body = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": PROMPT.format(claim=claim[:3000], passage=passage[:1500])}],
        "max_tokens": 200,
        "temperature": 0.0,
    }
    r = requests.post(VLLM_URL, json=body, timeout=120)
    if r.status_code != 200:
        return {"label": "error", "reason": f"http {r.status_code}"}
    txt = r.json()["choices"][0]["message"]["content"]
    # Extract JSON
    import re
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return {"label": "error", "reason": "no json"}
    try:
        out = json.loads(m.group(0))
        return out
    except Exception:
        return {"label": "error", "reason": "parse"}


def load_doc_chunks():
    """Read meta.jsonl, build doc_id → list of chunk positions index."""
    doc_chunks = defaultdict(list)
    with open(META_FILE) as f:
        for line_idx, line in enumerate(f):
            r = json.loads(line)
            doc_chunks[(r["source"], r["doc_id"])].append(line_idx)
    return doc_chunks


def read_chunk_text(doc_chunks_positions):
    """Given line indices in meta, look up source parquets and return texts."""
    # This is a simplified version: we expect the chunks parquet is
    # iterable in the same order as meta.jsonl. We load all parquets, build
    # an in-memory list (will work for tens of millions of chunks at ~500 bytes
    # each = ~10 GB, fits in memory on sk3).
    import pyarrow.parquet as pq
    import glob
    chunk_dir = f"{BASE}/processed/spec_chunks"
    all_texts = []
    for f in sorted(glob.glob(f"{chunk_dir}/*.parquet")):
        for row in pq.read_table(f, columns=["text"]).to_pylist():
            all_texts.append(row["text"])
    return all_texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_pairs", type=int, default=10000,
                   help="How many §102 pairs to label")
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    print("Loading meta.jsonl ...", file=sys.stderr)
    doc_chunks = load_doc_chunks()
    print(f"  {len(doc_chunks):,} (source, doc_id) → chunks", file=sys.stderr)

    print("Loading chunk texts in order ...", file=sys.stderr)
    all_texts = read_chunk_text(doc_chunks)
    print(f"  {len(all_texts):,} chunk texts loaded", file=sys.stderr)

    print(f"Streaming clean 102 pairs from {CLEAN_102} ...", file=sys.stderr)
    n_labeled = 0
    n_yes = 0; n_partial = 0; n_no = 0; n_err = 0
    t0 = time.time()
    with gzip.open(CLEAN_102, "rt") as fin, gzip.open(args.out, "wt") as fout:
        for line in fin:
            if n_labeled >= args.n_pairs:
                break
            pair = json.loads(line)
            anchor = pair["anchor_text"]
            cited_pid = pair["positive_pgpub_id"]
            # Try both source variants
            chunks_g = doc_chunks.get(("g", cited_pid), [])
            chunks_pg = doc_chunks.get(("pg", cited_pid), [])
            chunk_positions = chunks_g + chunks_pg
            if not chunk_positions:
                continue
            for pos in chunk_positions[:20]:  # cap per doc for speed
                passage = all_texts[pos]
                lbl = llm_label(anchor, passage)
                fout.write(json.dumps({
                    "anchor_pgpub_id": pair["anchor_pgpub_id"],
                    "anchor_text": anchor[:500],
                    "candidate_doc_id": cited_pid,
                    "candidate_chunk_idx": pos,
                    "candidate_text": passage[:500],
                    "relation": "positive",
                    "llm_label": lbl.get("label"),
                    "llm_reason": lbl.get("reason"),
                }) + "\n")
                n_labeled += 1
                v = lbl.get("label")
                if v == "yes": n_yes += 1
                elif v == "partial": n_partial += 1
                elif v == "no": n_no += 1
                else: n_err += 1
                if n_labeled % 100 == 0:
                    rate = n_labeled / max(1, time.time() - t0)
                    print(f"  labeled {n_labeled:,}  yes={n_yes} partial={n_partial} no={n_no} err={n_err}  "
                          f"{rate:.1f}/s", file=sys.stderr, flush=True)
    print(f"\nDone. {n_labeled:,} pairs labeled.")
    print(f"  yes={n_yes}  partial={n_partial}  no={n_no}  err={n_err}")


if __name__ == "__main__":
    main()
