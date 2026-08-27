#!/usr/bin/env python3
"""Consolidate Gemma-parsed claim-matching guideline metrics into a deduplicated bank.

Per BEST-PRACTICES [metric inference]: gather all extracted rubric metrics, dedup by embedding
(cos>0.86 = same metric reworded), then coverage-select medoids (KMeans) so the bank is fixed and
comparable. Preserve source domain (journalism/academic/patents) on each kept metric. Output one
bank JSONL: {metric_id, name, description, guidance, domain, dup_count}.

Run on sk3 (needs sentence_transformers + torch): use the miniconda python.
  /lfs/skampere3/0/alexspan/miniconda3/bin/python scripts/claim_matching_bank.py
"""
import json, glob, os
import numpy as np

BASE = "/lfs/skampere3/0/alexspan/norm-research"
# read whichever parse backend produced output (Gemma parse was GPU-blocked -> Sonnet subagents);
# invented metrics live alongside, tagged domain="invented"
PARSED_DIRS = [f"{BASE}/datasets/claim-matching/guidelines/sonnet-parsed",
               f"{BASE}/datasets/claim-matching/guidelines/gemma-parsed"]
OUT = f"{BASE}/datasets/claim-matching/claim_matching_bank.jsonl"
DUP_TAU = 0.86
TARGET_MEDOIDS = 60


def main():
    raw = []
    files = []
    for pd in PARSED_DIRS:
        files += glob.glob(f"{pd}/*.json")
    for fp in sorted(files):
        if os.path.basename(fp).startswith("_"):
            continue
        d = json.load(open(fp))
        dom = d.get("domain") or d.get("extracted", {}).get("domain") or "unknown"
        for m in d.get("extracted", {}).get("rubrics_metrics", []) or []:
            name = (m.get("name") or "").strip()
            desc = (m.get("description") or "").strip()
            if len(name) < 4 or len(desc) < 20:
                continue
            raw.append({"name": name, "description": desc,
                        "guidance": (m.get("guidance") or "").strip(),
                        "domain": dom, "src": os.path.basename(fp)})
    print(f"[bank] {len(raw)} raw metrics from {len(set(r['src'] for r in raw))} docs", flush=True)
    dd = {}
    for r in raw:
        dd.setdefault(r["domain"], 0); dd[r["domain"]] += 1
    print(f"[bank] raw by domain: {dd}", flush=True)
    if not raw:
        print("no metrics parsed — check parse output"); return

    # CPU-friendly dedup embedder (GPUs are often saturated by other jobs; ~300 short texts is fast)
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 6e9 else "cpu"
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
    print(f"[bank] dedup embedder on {dev}", flush=True)
    txt = [f"{r['name']}. {r['description']}" for r in raw]
    E = st.encode(txt, batch_size=64, convert_to_numpy=True, normalize_embeddings=True,
                  show_progress_bar=False)

    # greedy dedup at cos>DUP_TAU (keep first, count dups)
    keep, kept_vecs = [], []
    for i, r in enumerate(raw):
        v = E[i]
        hit = None
        for j, kv in enumerate(kept_vecs):
            if float(v @ kv) > DUP_TAU:
                hit = j; break
        if hit is None:
            r = dict(r); r["dup_count"] = 1
            keep.append(r); kept_vecs.append(v)
        else:
            keep[hit]["dup_count"] += 1
    print(f"[bank] after dedup(cos>{DUP_TAU}): {len(keep)} unique metrics", flush=True)

    # PER-DOMAIN medoid selection: keep ALL invented (the novel arm, already distinct); coverage-
    # select the gathered doctrine per domain so the bank is domain-balanced and the invented arm is
    # measurable (global KMeans collapsed invented to 1 — that would kill the invented-vs-gathered test)
    from sklearn.cluster import KMeans
    kv = np.array(kept_vecs)
    CAPS = {"journalism": 18, "academic": 18, "patents": 22}  # invented: keep all
    final = []
    for dom in sorted(set(r["domain"] for r in keep)):
        idx = [i for i, r in enumerate(keep) if r["domain"] == dom]
        cap = CAPS.get(dom)
        if cap is None or len(idx) <= cap:
            final += idx
            print(f"[bank] {dom}: kept all {len(idx)}", flush=True)
        else:
            sub = kv[idx]
            km = KMeans(n_clusters=cap, n_init=10, random_state=0).fit(sub)
            for c in range(cap):
                ci = [idx[j] for j in np.where(km.labels_ == c)[0]]
                cen = km.cluster_centers_[c]
                final.append(max(ci, key=lambda i: float(kv[i] @ cen)))
            print(f"[bank] {dom}: {len(idx)} -> {cap} medoids", flush=True)
    keep = [keep[i] for i in sorted(set(final))]
    print(f"[bank] domain-stratified -> {len(keep)} metrics", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        for i, r in enumerate(keep):
            r["metric_id"] = f"cm{i:03d}"
            fh.write(json.dumps(r) + "\n")
    fin = {}
    for r in keep:
        fin[r["domain"]] = fin.get(r["domain"], 0) + 1
    print(f"[bank] wrote {len(keep)} metrics -> {OUT}", flush=True)
    print(f"[bank] final by domain: {fin}", flush=True)
    print("BANK_DONE", flush=True)


if __name__ == "__main__":
    main()
