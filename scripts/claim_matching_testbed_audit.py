#!/usr/bin/env python3
"""Confound audit for the pair-grain claim-matching testbed (localize_results_scale_gemma.jsonl).

pos = (claim-element, reference-span) pair the examiner cited as a match; neg = a not-cited pair.
Before this becomes the scoring bed for claim-matching metrics, check the lexical/length confounds
a text-only metric could exploit trivially (per BEST-PRACTICES content-guard + dataset-first):
  - is pos separable by span/element LENGTH alone?
  - is pos separable by lexical OVERLAP (element∩span) alone? -> the baseline metrics must beat.
  - where do neg spans come from (same doc as element's field? different topic?) -> topicality leak?
Prints length/overlap AUCs, base rate, within-doc structure, 8+8 samples. Run on sk3 (CPU)."""
import json, re, hashlib, collections
import numpy as np
from sklearn.metrics import roc_auc_score

BASE = "/lfs/skampere3/0/alexspan/norm-research"
LOC = f"{BASE}/datasets/patents/processed/localize_results_scale_gemma.jsonl"
WORD = re.compile(r"[a-z]{3,}")


def toks(s):
    return set(WORD.findall((s or "").lower()))


def join_spans(sp):
    if isinstance(sp, list):
        sp = " ".join(x for x in sp if isinstance(x, str))
    return (sp or "").strip()


def main():
    rows = []
    for ln in open(LOC):
        r = json.loads(ln)
        el = (r.get("element") or "").strip()
        sp = join_spans(r.get("spans"))
        if len(el) < 10 or not sp or el == "(no per-element breakdown)":
            continue
        et, st = toks(el), toks(sp)
        rows.append({
            "y": 1 if r["label"] == "pos" else 0,
            "uid": str(r.get("uid")), "doc_id": str(r.get("doc_id")),
            "el_len": len(el.split()), "sp_len": len(sp.split()),
            "jacc": len(et & st) / max(1, len(et | st)),
            "contain": len(et & st) / max(1, len(et)),
            "disc": str(r.get("discloses")) == "True",
            "el": el, "sp": sp,
        })
    y = np.array([r["y"] for r in rows])
    print(f"[n] {len(rows)} (element,span) pairs, pos rate {y.mean():.3f}", flush=True)
    print(f"[disclose] Gemma-said-discloses rate: pos {np.mean([r['disc'] for r in rows if r['y']==1]):.3f} "
          f"neg {np.mean([r['disc'] for r in rows if r['y']==0]):.3f}", flush=True)

    print("\n[confound: can a trivial feature separate pos/neg?]", flush=True)
    for name in ("el_len", "sp_len", "jacc", "contain"):
        v = np.array([r[name] for r in rows], float)
        a = roc_auc_score(y, v)
        mp, mn = v[y == 1].mean(), v[y == 0].mean()
        print(f"  {name:9s} AUC={a:.4f}  (pos_mean {mp:.3f} vs neg_mean {mn:.3f})", flush=True)

    # topicality / doc structure: are neg spans from a different doc than pos?
    docs = collections.Counter(r["doc_id"] for r in rows)
    print(f"\n[doc structure] {len(docs)} distinct doc_ids; "
          f"top doc has {docs.most_common(1)[0][1]} pairs", flush=True)
    # element reuse across pos/neg (same element, both labels?)
    byel = collections.defaultdict(set)
    for r in rows:
        byel[r["el"][:120]].add(r["y"])
    mixed = sum(1 for v in byel.values() if len(v) == 2)
    print(f"[element reuse] {len(byel)} distinct elements; {mixed} appear as BOTH pos&neg "
          f"({mixed/max(1,len(byel)):.1%}) = the within-element discriminable set", flush=True)

    print("\n[POS samples] (jacc, contain | element -> span)", flush=True)
    for r in [r for r in rows if r["y"] == 1][:6]:
        print(f"  j={r['jacc']:.2f} c={r['contain']:.2f} | {r['el'][:70]} -> {r['sp'][:70]}", flush=True)
    print("[NEG samples]", flush=True)
    for r in [r for r in rows if r["y"] == 0][:6]:
        print(f"  j={r['jacc']:.2f} c={r['contain']:.2f} | {r['el'][:70]} -> {r['sp'][:70]}", flush=True)
    print("TESTBED_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
