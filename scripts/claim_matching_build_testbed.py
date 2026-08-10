#!/usr/bin/env python3
"""Build the clean within-claim claim-matching testbed (element-length controlled, lexically adversarial).

From option3_claims_gemma_scale.jsonl, for each examiner-rejected claim:
  pos = (element, span-of-the-EXAMINER-CITED (gold) reference)          -> a true claim-match
  neg = (element, span-of-the-HARDEST filler reference)                 -> a same-field non-match
where the hardest filler = the non-gold candidate whose span has the HIGHEST lexical containment of
the element (the strongest lexical distractor). Same element on both sides -> controls the element-
length/breadth confound that contaminated both the claim-level and localize testbeds. Because the
examiner's gold ref is lexically LESS similar than fillers ~96% of the time (memory: position-leak
audit), a lexical-overlap baseline is expected to point the WRONG way here — so any recovery is
genuine semantic claim-matching, not surface overlap. Label Y = is-the-examiner's-cited-reference.

Reconstruction-only: metrics later score (element, span) match WITHOUT seeing Y; Y enters only at
the recovery readout. Split by app-hash (salted, mod-1000). Run on sk3 (CPU).
Prints the confound table (length/overlap AUC — want near .5 or inverted) then writes the testbed."""
import json, re, hashlib
import numpy as np
from sklearn.metrics import roc_auc_score

BASE = "/lfs/skampere3/0/alexspan/norm-research"
SCALE = f"{BASE}/datasets/patents/processed/option3_claims_gemma_scale.jsonl"
OUT = f"{BASE}/datasets/claim-matching/testbed/pair_testbed.jsonl"
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
    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    n_claim = n_skip = 0
    for ln in open(SCALE):
        r = json.loads(ln)
        if r["label"] != "pos":
            continue
        n_claim += 1
        el = (r["element"] or "").strip()
        if len(el) < 12:
            n_skip += 1; continue
        et = toks(el)
        gold, fillers = None, []
        for ref in r["refs"]:
            sp = join_spans(ref.get("spans"))
            if not sp:
                continue
            (gold and 0)  # noop
            if str(ref.get("is_gold")) == "True" and gold is None:
                gold = sp
            else:
                fillers.append(sp)
        if gold is None or not fillers:
            n_skip += 1; continue
        # deterministic pseudo-random filler (NOT hardest — avoid engineering lexical anti-correlation
        # into the label; lexical overlap is controlled as a covariate in the recovery readout instead)
        pick = int(hashlib.md5(f"neg::{r['uid']}".encode()).hexdigest(), 16) % len(fillers)
        neg = fillers[pick]
        base = {"uid": str(r["uid"]), "app_id": str(r["app_id"]),
                "claim_num": str(r["claim_num"]), "element": el,
                "rejection_type": str(r.get("rejection_type"))}
        rows.append({**base, "y": 1, "span": gold})
        rows.append({**base, "y": 0, "span": neg})
    print(f"[build] {n_claim} pos-claims, {n_skip} skipped (no gold/filler/short) -> "
          f"{len(rows)} pairs ({len(rows)//2} claims x pos+neg)", flush=True)

    # confound table — length/overlap should NOT separate y (and lexical should point wrong way)
    y = np.array([r["y"] for r in rows])
    def feat(fn):
        return np.array([fn(r) for r in rows], float)
    el_t = [toks(r["element"]) for r in rows]
    sp_t = [toks(r["span"]) for r in rows]
    contain = np.array([len(e & s) / max(1, len(e)) for e, s in zip(el_t, sp_t)])
    splen = feat(lambda r: len(r["span"].split()))
    print("\n[confound: does a trivial feature separate gold-match from filler? want ~.5]", flush=True)
    print(f"  span_len      AUC={roc_auc_score(y, splen):.4f}", flush=True)
    print(f"  lexical_contain AUC={roc_auc_score(y, contain):.4f}  "
          f"(gold {contain[y==1].mean():.3f} vs filler {contain[y==0].mean():.3f})", flush=True)

    # split + write
    for r in rows:
        r["split"] = "test" if split_bucket(r["app_id"]) < 200 else "train"
    ntr = sum(1 for r in rows if r["split"] == "train")
    with open(OUT, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\n[split] train {ntr} / test {len(rows)-ntr}  (app-hash, salted mod-1000, <200=test)",
          flush=True)
    print(f"[write] {OUT}", flush=True)
    print("BUILD_TESTBED_DONE", flush=True)


if __name__ == "__main__":
    main()
