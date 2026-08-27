#!/usr/bin/env python3
"""Testbed v2: identical to v1 EXCEPT the multiple-gold bug fix (Codex audit finding #1) — refs with
is_gold=True NEVER enter the filler pool. In v1, a claim with 2+ examiner-cited refs could draw a GOLD
span as its "negative" (measured: 7.94% of claims had a gold in the pool; 1.06% of picked negatives
WERE gold -> unwinnable pairs). Everything else (element filter, deterministic pseudo-random pick,
app-hash split, schema) is byte-identical logic to v1.

Outputs:
  datasets/claim-matching/testbed/pair_testbed_v2.jsonl      full v2 testbed
  datasets/claim-matching/testbed/v2_changed_negs.jsonl      (uid,y=0) rows that DIFFER from v1,
                                                             restricted to the standard 800-claim probe
                                                             (for targeted rescoring via --pairs-file)
  python scripts/claim_matching_build_testbed_v2.py
"""
import json, re, hashlib
import numpy as np
from sklearn.metrics import roc_auc_score

BASE = "/lfs/skampere3/0/alexspan/norm-research"
SCALE = f"{BASE}/datasets/patents/processed/option3_claims_gemma_scale.jsonl"
V1 = f"{BASE}/datasets/claim-matching/testbed/pair_testbed.jsonl"
OUT = f"{BASE}/datasets/claim-matching/testbed/pair_testbed_v2.jsonl"
CHANGED = f"{BASE}/datasets/claim-matching/testbed/v2_changed_negs.jsonl"
WORD = re.compile(r"[a-z]{3,}")


def toks(s):
    return set(WORD.findall((s or "").lower()))


def join_spans(sp):
    if isinstance(sp, list):
        sp = " ".join(x for x in sp if isinstance(x, str))
    return (sp or "").strip()


def split_bucket(app):
    return int(hashlib.md5(f"split::{app}".encode()).hexdigest(), 16) % 1000


def main():
    rows = []
    n_claim = n_skip = n_hadgold = 0
    for ln in open(SCALE):
        r = json.loads(ln)
        if r["label"] != "pos":
            continue
        n_claim += 1
        el = (r["element"] or "").strip()
        if len(el) < 12:
            n_skip += 1; continue
        gold, fillers, had_gold_in_pool = None, [], False
        for ref in r["refs"]:
            sp = join_spans(ref.get("spans"))
            if not sp:
                continue
            if str(ref.get("is_gold")) == "True":
                if gold is None:
                    gold = sp
                else:
                    had_gold_in_pool = True   # v1 would have put this in fillers — v2 DROPS it
            else:
                fillers.append(sp)
        if gold is None or not fillers:
            n_skip += 1; continue
        n_hadgold += had_gold_in_pool
        pick = int(hashlib.md5(f"neg::{r['uid']}".encode()).hexdigest(), 16) % len(fillers)
        neg = fillers[pick]
        base = {"uid": str(r["uid"]), "app_id": str(r["app_id"]),
                "claim_num": str(r["claim_num"]), "element": el,
                "rejection_type": str(r.get("rejection_type"))}
        rows.append({**base, "y": 1, "span": gold})
        rows.append({**base, "y": 0, "span": neg})
    print(f"[v2] {n_claim} pos-claims, {n_skip} skipped -> {len(rows)} pairs "
          f"({len(rows)//2} claims); {n_hadgold} claims had extra gold(s) excluded from pool",
          flush=True)

    # confound audit (same as v1: want ~.5)
    y = np.array([r["y"] for r in rows])
    el_t = [toks(r["element"]) for r in rows]
    sp_t = [toks(r["span"]) for r in rows]
    contain = np.array([len(e & s) / max(1, len(e)) for e, s in zip(el_t, sp_t)])
    splen = np.array([len(r["span"].split()) for r in rows], float)
    print(f"[confound] span_len AUC={roc_auc_score(y, splen):.4f}  "
          f"lexical AUC={roc_auc_score(y, contain):.4f}", flush=True)

    for r in rows:
        r["split"] = "test" if split_bucket(r["app_id"]) < 200 else "train"
    with open(OUT, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    # diff vs v1 -> changed negatives, restricted to the standard 800-claim probe
    v1neg = {}
    byuid = {}
    import collections as C
    byu = C.defaultdict(list)
    for ln in open(V1):
        r = json.loads(ln)
        byu[r["uid"]].append(r)
        if r["y"] == 0:
            v1neg[r["uid"]] = r["span"]
    uids = [u for u, v in byu.items() if len(v) == 2 and {x["y"] for x in v} == {0, 1}]
    uids.sort(key=lambda u: hashlib.md5(f"probe::{u}".encode()).hexdigest())
    probe = set(uids[:800])

    changed = [r for r in rows if r["y"] == 0 and r["uid"] in v1neg and r["span"] != v1neg[r["uid"]]]
    changed_probe = [r for r in changed if r["uid"] in probe]
    with open(CHANGED, "w") as fh:
        for r in changed_probe:
            fh.write(json.dumps(r) + "\n")
    print(f"[v2] changed negatives: {len(changed)} total; {len(changed_probe)} inside the 800-claim "
          f"probe -> {CHANGED}", flush=True)
    print("BUILD_V2_DONE", flush=True)


if __name__ == "__main__":
    main()
