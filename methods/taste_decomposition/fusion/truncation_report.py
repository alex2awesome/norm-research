#!/usr/bin/env python3
"""Truncation-rate report for the Direction-3 augmented caption datasets
(max_length=1024, Llama-3.1-8B tokenizer) plus the moredata/original splits."""
import glob
import json
import os

import pandas as pd
from transformers import AutoTokenizer

FUS = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(FUS, "dense_data")
MAXLEN = 1024

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
out = {}
for d in sorted(glob.glob(f"{DATA}/*/")):
    name = os.path.basename(d.rstrip("/"))
    df = pd.read_csv(os.path.join(d, "data.csv"))
    lens = [len(tok(t, truncation=False)["input_ids"]) for t in df.text.astype(str)]
    s = pd.Series(lens)
    out[name] = {"n": int(len(s)), "max_tokens": int(s.max()),
                 "p99_tokens": int(s.quantile(.99)), "mean_tokens": float(s.mean()),
                 "truncated_at_1024": int((s > MAXLEN).sum()),
                 "truncation_rate": float((s > MAXLEN).mean())}
    print(name, out[name], flush=True)
with open(os.path.join(DATA, "truncation_report.json"), "w") as f:
    json.dump(out, f, indent=2)
print("wrote", os.path.join(DATA, "truncation_report.json"))
