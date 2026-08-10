#!/usr/bin/env python3
"""Mine reviewer NOVELTY complaints ("this was already done", "not novel", "incremental") —
the second expert-revealed ground truth, validating the prior-art checker the way support
complaints validate the adequacy checker. Same machinery as run_reviewer_flags (regex ->
Gemma confirm-judge with blinded anchors). Output: outputs/reviewer_flags/novelty_flags.jsonl
Run on sk3: python -m methods.claim_verification.run_novelty_flags"""
import json, os, re, sqlite3, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np, pandas as pd
from claim_verification.core import Cache, _post, _parse_json, _key, _sentences

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
PDF_DB = f"{ROOT}/datasets/peer-review/peer_review_pdfs.db"
CFG = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma"}

CAND = re.compile(
    r"(not\s+(?:\w+ly\s+)?novel|lacks?\s+novelty|(?:limited|marginal|incremental|questionable|unclear|little)\s+novelty"
    r"|novelty\s+(?:is|seems?|appears?)\s+(?:limited|marginal|incremental|unclear|low|questionable|thin)"
    r"|already\s+(?:been\s+)?(?:done|proposed|studied|explored|shown|introduced|established|known)"
    r"|(?:has|have|was|were)\s+(?:already\s+)?been\s+(?:proposed|done|studied|explored|shown|introduced)"
    r"|well[-\s]?known\s+(?:technique|method|idea|result|trick)"
    r"|(?:very|highly|too)\s+similar\s+to|closely\s+(?:resembles|follows|related\s+to)\s+(?:\[|\w+\s+et\s+al)"
    r"|prior\s+work\s+(?:has\s+)?already|incremental\s+(?:contribution|improvement|extension|work)"
    r"|straightforward\s+(?:extension|application|combination)|simple\s+combination\s+of)", re.I)

JUDGE = """A peer reviewer wrote this (from a review of a machine-learning paper):

"{sent}"

Is the reviewer asserting that the paper's contribution or a specific idea in it LACKS NOVELTY
— i.e., that it was already done, proposed, or known in prior work, or is only an incremental
variation of prior work? (Not: praising novelty, describing the paper's own related-work
section, or generic comments.)

Return ONLY JSON:
{{"flags_not_novel": true/false, "claimed_novel_thing": "<paraphrase of what the paper claims as novel that the reviewer disputes, or empty>", "prior_work_named": "<the prior work the reviewer points to, or empty>"}}"""

ANCHORS = [
    ("The novelty is limited: contrastive pre-training with hard negatives was already proposed by Chen et al. 2020.", True),
    ("This is a straightforward combination of two well-known techniques (LoRA and distillation) with no new insight.", True),
    ("The core idea of using attention over graph edges has already been done in GAT; the paper does not discuss the difference.", True),
    ("Adaptive learning rates for this setting are well-known tricks; the contribution is incremental.", True),
    ("The paper proposes a novel and elegant approach that I have not seen before.", False),
    ("The related work section covers prior work on distillation thoroughly.", False),
    ("The authors compare against well-known baselines, which strengthens the evaluation.", False),
    ("It would be interesting to see whether this has already been tried on larger models — a discussion would help.", False),
]

def judge_sent(sent, cache):
    k = _key("novflag", CFG["model"], sent[:400])
    hit = cache.get(k)
    if hit is not None: return hit
    raw = _post(CFG["base_url"], CFG["model"], JUDGE.format(sent=sent[:600]), max_tokens=180)
    obj = _parse_json(raw) or {}
    out = {"flag": bool(obj.get("flags_not_novel")),
           "claim": str(obj.get("claimed_novel_thing") or "")[:300],
           "prior": str(obj.get("prior_work_named") or "")[:200]}
    cache.put(k, out)
    return out

def main():
    t = pd.read_csv(f"{ROOT}/outputs/tiered_peer/tiered_metrics.csv")
    forums = {i.replace("iclr_", ""): i for i in t.id.astype(str)}
    con = sqlite3.connect(PDF_DB)
    q = ",".join("?" * len(forums))
    revs = con.execute(f"SELECT paper_id, review_text, is_meta_review FROM reviews "
                       f"WHERE paper_id IN ({q})", list(forums)).fetchall()
    cands = []
    for pid, txt, meta in revs:
        if not txt or meta: continue
        for s in _sentences(str(txt)):
            if 40 < len(s) < 700 and CAND.search(s):
                cands.append((forums[pid], s))
    print(f"[novflag] {len(cands)} candidate sentences "
          f"({len(set(p for p,_ in cands))} papers)", flush=True)
    cache = Cache(f"{ROOT}/outputs/reviewer_flags/nov_cache.jsonl")
    lock = Lock(); results = []; anchor_out = {}
    items = [("ANCHOR", s, lab) for s, lab in ANCHORS] + [(p, s, None) for p, s in cands]
    def work(item):
        p, s, lab = item
        try: r = judge_sent(s, cache)
        except Exception: return
        with lock:
            if p == "ANCHOR": anchor_out[s[:50]] = (r["flag"], lab)
            else: results.append({"paper": p, "sent": s, **r})
            if len(results) % 300 == 0 and results:
                print(f"[novflag] {len(results)}/{len(cands)}", flush=True)
    with ThreadPoolExecutor(max_workers=24) as ex:
        list(ex.map(work, items))
    ok = sum(1 for f, l in anchor_out.values() if f == l)
    print(f"[novflag] ANCHORS: {ok}/{len(anchor_out)} correct; judge flag rate "
          f"{np.mean([r['flag'] for r in results]):.3f}", flush=True)
    with open(f"{ROOT}/outputs/reviewer_flags/novelty_flags.jsonl", "w") as f:
        for r in results: f.write(json.dumps(r) + "\n")
    F = pd.DataFrame(results)
    conf = F[F.flag]
    per = conf.groupby("paper").size()
    print(f"[novflag] confirmed complaints: {len(conf)} over {len(per)} papers "
          f"({len(per)/len(forums):.3f} of sample)", flush=True)
    print("NOVELTY_FLAGS_DONE", flush=True)

if __name__ == "__main__":
    main()
