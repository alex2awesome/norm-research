#!/usr/bin/env python3
"""E2E smoke of claim_verification on 3 press releases (Gemma 8006 on sk3).
Run ON sk3: python -m methods.claim_verification.smoke_pr (repo root)."""
import sys, json
sys.path.insert(0, "methods")
import pandas as pd
from claim_verification.core import doc_metrics, Cache, split_head_body

CFG = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma",
       "doc_kind": "press release", "max_claims": 4}

df = pd.read_parquet("datasets/press-releases/press_release_deconfounded.parquet")
df = df[df.text.str.len().between(1500, 6000)].sample(3, random_state=3)
cache = Cache("outputs/claimverify_smoke/cache.jsonl")
for _, row in df.iterrows():
    m, detail = doc_metrics(row.text, CFG, cache)
    head, _ = split_head_body(row.text)
    print("=" * 70)
    print("HEAD:", head[:180].replace("\n", " "))
    for c, v in zip(detail["claims"], detail["verdicts"]):
        print(f"  CLAIM [{c.get('kind','?'):11}] {c['claim'][:90]}")
        print(f"    -> {v['verdict']:7} ev={v.get('evidence_type')} grounded={v['grounded']} span={v['span'][:70]!r}")
    print("  CENSUS:", detail["census"])
    print("  METRICS:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if not k.startswith("ev_")})
print("SMOKE_DONE")
