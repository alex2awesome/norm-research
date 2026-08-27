#!/usr/bin/env python3
"""peer_revealed RECOGNITION PROBE (GPU leg of the memorization audit, 2026-08-12).

Measures per-paper "seen-ness" with the UNTRAINED base model (meta-llama/Llama-3.1-8B,
the same base the dense T fine-tunes): mean per-token NLL of the E-row text
(title+abstract, truncated 1024 tokens). Papers well represented in pretraining sit
at lower NLL. Output is one NLL per E-row; the discriminating readout (does the
d-c residual concentrate in recognized rows WITHIN year bands?) runs on CPU
afterwards (recognition_readout.py).

Descriptive instrument: NLL is confounded with era style and text regularity, which
is why the readout conditions on year band and reports both directions.

Run on sk3 (one GPU, poll-claim runner). Writes peer_recognition_nll.jsonl.
"""
import gzip
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "fusion/t0_rows/peer_revealed.texts.jsonl.gz"
OUT = HERE / "peer_recognition_nll.jsonl"
MODEL = "meta-llama/Llama-3.1-8B"
MAXTOK = 1024

rows = [json.loads(l) for l in gzip.open(SRC, "rt")]
done = set()
if OUT.exists():
    done = {json.loads(l)["uid"] for l in open(OUT)}
todo = [r for r in rows if r["uid"] not in done]
print(f"{len(rows)} rows, {len(done)} done, {len(todo)} to go", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda")
model.eval()

with open(OUT, "a") as fh, torch.no_grad():
    for i, r in enumerate(todo):
        ids = tok(r["text"], return_tensors="pt", truncation=True,
                  max_length=MAXTOK).input_ids.cuda()
        out = model(ids, labels=ids)
        n_tok = int(ids.shape[1])
        fh.write(json.dumps({"uid": r["uid"], "ntitle": r["id"],
                             "mean_nll": float(out.loss), "n_tokens": n_tok}) + "\n")
        if (i + 1) % 50 == 0:
            fh.flush()
            print(f"{i+1}/{len(todo)}", flush=True)
print("RECOGNITION_PROBE_DONE", flush=True)
