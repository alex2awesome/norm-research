#!/usr/bin/env python3
"""Build (element, paragraph) cross-encoder training set from V2 judge labels.

Sources (all on sk3, CPU-only):
  truecite_testbed_v1/v2_pairs.jsonl           element x top-3 paragraph TEXT
  truecite_testbed_v1/v2_pair_judgments.jsonl  Qwen-122B disclosed+confidence
  (optional) v2_pairs_v61.jsonl / v2_pair_judgments_v61.jsonl  -- appended
              automatically once tonight's v6.1a ablation lands

These pairs come from the ACTUAL retrieval distribution (v6a top-3), i.e. the
hard cases small judges fail on -- exactly what to distill.  Soft label =
confidence if disclosed else 100-confidence, rescaled to [0,1].

Split: GROUP by app_id (80/10/10) so no application straddles splits.

Output: datasets/patents/processed/element_para_xenc_v1/{train,val,test}.jsonl.gz
"""
import glob
import gzip
import hashlib
import json
import os

BASE = os.path.expanduser(
    "~/norm-research/datasets/patents/processed/truecite_testbed_v1")
OUT = os.path.expanduser(
    "~/norm-research/datasets/patents/processed/element_para_xenc_v1")
os.makedirs(OUT, exist_ok=True)

SOURCES = [("v6a", f"{BASE}/v2_pairs.jsonl", f"{BASE}/v2_pair_judgments.jsonl")]
if os.path.exists(f"{BASE}/v2_pair_judgments_v61.jsonl"):
    SOURCES.append(("v61", f"{BASE}/v2_pairs_v61.jsonl",
                    f"{BASE}/v2_pair_judgments_v61.jsonl"))


def key(j):
    return (j["app_id"], j["ifw"], str(j["claim_num"]), int(j["el_idx"]),
            int(j["p_idx"]))


records, seen = [], set()
for src, pairs_f, judg_f in SOURCES:
    # (key) -> (element, pgpub_id, para_key, para_text)
    paras = {}
    for line in open(pairs_f):
        r = json.loads(line)
        for p_idx, p in enumerate(r["paras"]):
            paras[key({**r, "p_idx": p_idx})] = (
                r["element"], p[0], p[1], p[2])
    n_src = 0
    for line in open(judg_f):
        j = json.loads(line)
        if j.get("disclosed") is None:
            continue
        k = key(j)
        if k not in paras or k in seen:
            continue
        seen.add(k)
        element, pgpub, pkey, ptext = paras[k]
        conf = int(j["confidence"])
        soft = (conf if j["disclosed"] else 100 - conf) / 100.0
        records.append({
            "app_id": j["app_id"], "ifw": j["ifw"],
            "claim_num": str(j["claim_num"]), "el_idx": int(j["el_idx"]),
            "element": element, "pgpub_id": pgpub, "para_key": pkey,
            "paragraph": ptext, "disclosed": bool(j["disclosed"]),
            "soft_label": round(soft, 3), "judge_src": src,
        })
        n_src += 1
    print(f"source {src}: +{n_src:,} labeled pairs")

print(f"total: {len(records):,} pairs, "
      f"disclosed rate {sum(r['disclosed'] for r in records)/len(records):.3f}")

# group split by app_id via stable hash
def split_of(app_id):
    h = int(hashlib.md5(app_id.encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


outs = {s: gzip.open(f"{OUT}/{s}.jsonl.gz", "wt") for s in
        ("train", "val", "test")}
counts = {s: 0 for s in outs}
for r in records:
    s = split_of(r["app_id"])
    outs[s].write(json.dumps(r) + "\n")
    counts[s] += 1
for f in outs.values():
    f.close()

stats = {"counts": counts, "n_apps": len({r['app_id'] for r in records}),
         "disclosed_rate": round(
             sum(r["disclosed"] for r in records) / len(records), 4),
         "sources": [s for s, _, _ in SOURCES]}
json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2))
