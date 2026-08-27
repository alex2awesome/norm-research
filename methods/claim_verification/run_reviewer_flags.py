#!/usr/bin/env python3
"""Expert-revealed validation of the claim-verification metric (user: reviewers often say
"this isn't supported" — can our metric find those papers/claims?).

Pipeline on the 600 tiered_peer papers (t1_support already computed):
 1. Mine all their reviews for support-complaint candidate sentences (regex).
 2. Gemma confirm-judge each candidate (+BLINDED ANCHORS per standing rule): does the reviewer
    assert a specific claim lacks support/evidence? If yes, quote the challenged claim.
 3. Analyses:
    A. paper-level: does LOW t1_support predict reviewer-flagged papers? (AUC, extreme groups)
    B. claim-level (sharpest): match reviewer-challenged claims to our extracted claims;
       our verifier's verdict on matched claims vs all claims (does it say NONE more often?).
    C. prevalence + tail examples for the report.
Run on sk3: python -m methods.claim_verification.run_reviewer_flags"""
import json, os, re, sqlite3, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np, pandas as pd
from claim_verification.core import Cache, _post, _parse_json, _key, _sentences, verify_claim
from claim_verification.seam_metrics import _toks

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
EB = os.path.join(ROOT, "datasets/evidence_bases")
PDF_DB = f"{ROOT}/datasets/peer-review/peer_review_pdfs.db"
CFG = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma", "doc_kind": "scientific paper"}

CAND = re.compile(
    r"(not?\s+(?:\w+ly\s+)?(?:supported|substantiated|justified|backed|validated|evidenced)"
    r"|unsupported|unsubstantiated|over-?claim|over-?stat"
    r"|no\s+(?:\w+\s+)?(?:evidence|support|justification)\s+(?:for|to|that)"
    r"|lacks?\s+(?:\w+\s+)?(?:evidence|support|justification)"
    r"|insufficient\s+(?:evidence|support)"
    r"|do(?:es)?\s+not\s+(?:support|justify|substantiate)"
    r"|claims?\s+(?:is|are)\s+(?:too\s+strong|not)"
    r"|not\s+borne\s+out|cannot\s+conclude)", re.I)

JUDGE = """A peer reviewer wrote this (from a review of a scientific paper):

"{sent}"

Is the reviewer asserting that a SPECIFIC claim, conclusion, or statement made by the paper
lacks adequate support or evidence? (Not: asking for clarification, praising support,
discussing related work, or generic weakness comments.)

Return ONLY JSON:
{{"flags_unsupported_claim": true/false, "challenged_claim": "<paraphrase of the paper claim being challenged, or empty>"}}"""

# blinded anchors (known labels) — injected into every judging run, per standing rule
ANCHORS = [
    ("The central claim that the method generalizes to unseen domains is not supported by the experiments, which only use synthetic data.", True),
    ("There is no evidence for the claimed 10x speedup; Table 3 shows at most 2x.", True),
    ("The claim in the abstract that this is the first such method is unsubstantiated and overclaims novelty.", True),
    ("The authors claim robustness to noise but provide no justification or ablation for this.", True),
    ("The claims are well supported by extensive experiments on five benchmarks.", False),
    ("The authors support their claims with a new large-scale benchmark, which I appreciate.", False),
    ("Could the authors clarify how the loss is normalized? This was not clear to me.", False),
    ("Prior work does not support batch sizes this large, so this contribution is timely.", False),
]

def judge_sent(sent, cache):
    k = _key("revflag", CFG["model"], sent[:400])
    hit = cache.get(k)
    if hit is not None: return hit
    raw = _post(CFG["base_url"], CFG["model"], JUDGE.format(sent=sent[:600]), max_tokens=150)
    obj = _parse_json(raw) or {}
    out = {"flag": bool(obj.get("flags_unsupported_claim")),
           "claim": str(obj.get("challenged_claim") or "")[:300]}
    cache.put(k, out)
    return out

def jaccard(a, b):
    sa, sb = set(_toks(a)), set(_toks(b))
    return len(sa & sb) / max(len(sa | sb), 1)

def main():
    t = pd.read_csv(f"{ROOT}/outputs/tiered_peer/tiered_metrics.csv")
    t["id"] = t.id.astype(str)
    forums = {i.replace("iclr_", ""): i for i in t.id}
    con = sqlite3.connect(PDF_DB)
    q = ",".join("?" * len(forums))
    revs = con.execute(f"SELECT paper_id, review_text, is_meta_review FROM reviews "
                       f"WHERE paper_id IN ({q})", list(forums)).fetchall()
    print(f"[revflag] {len(revs)} reviews over {len(forums)} papers", flush=True)
    # 1. candidates
    cands = []   # (paper_id, sentence)
    for pid, txt, meta in revs:
        if not txt or meta: continue
        for s in _sentences(str(txt)):
            if 40 < len(s) < 700 and CAND.search(s):
                cands.append((forums[pid], s))
    print(f"[revflag] candidate sentences: {len(cands)} "
          f"({len(set(p for p,_ in cands))} papers with >=1)", flush=True)
    # 2. judge (+anchors blinded into the batch)
    cache = Cache(f"{ROOT}/outputs/reviewer_flags/cache.jsonl")
    os.makedirs(f"{ROOT}/outputs/reviewer_flags", exist_ok=True)
    lock = Lock(); results = []; anchor_out = {}
    items = [("ANCHOR", s, lab) for s, lab in ANCHORS] + [(p, s, None) for p, s in cands]
    def work(item):
        p, s, lab = item
        try: r = judge_sent(s, cache)
        except Exception: return
        with lock:
            if p == "ANCHOR": anchor_out[s[:50]] = (r["flag"], lab)
            else: results.append({"paper": p, "sent": s, **r})
            if len(results) % 300 == 0 and results: print(f"[revflag] {len(results)}/{len(cands)}", flush=True)
    with ThreadPoolExecutor(max_workers=24) as ex:
        list(ex.map(work, items))
    ok = sum(1 for f, l in anchor_out.values() if f == l)
    print(f"[revflag] ANCHORS: {ok}/{len(anchor_out)} correct "
          f"(judge flag rate on candidates: {np.mean([r['flag'] for r in results]):.3f})", flush=True)
    with open(f"{ROOT}/outputs/reviewer_flags/flags.jsonl", "w") as f:
        for r in results: f.write(json.dumps(r) + "\n")
    # 3A. paper-level
    F = pd.DataFrame(results)
    conf = F[F.flag]
    per = conf.groupby("paper").size().rename("n_flags")
    M = t.merge(per, left_on="id", right_index=True, how="left")
    M["n_flags"] = M.n_flags.fillna(0)
    M["flagged"] = (M.n_flags >= 1).astype(int)
    print(f"\n[revflag] prevalence: {M.flagged.mean():.3f} of papers flagged >=1; "
          f"flag-count dist: {M.n_flags.value_counts().sort_index().head(8).to_dict()}", flush=True)
    from sklearn.metrics import roc_auc_score
    print("\n[revflag] === A. does our metric find reviewer-flagged papers? ===", flush=True)
    for c in ("t1_support", "t1_echo", "t3_support", "t4_support", "novelty"):
        v = M[c].values.astype(float); mk = ~np.isnan(v)
        if mk.sum() > 100 and M.flagged[mk].nunique() == 2:
            a = roc_auc_score(M.flagged[mk], -v[mk])   # LOW support -> flagged
            print(f"  low {c:12} -> flagged  AUC={a:.4f} (n={int(mk.sum())})", flush=True)
    hi = M[M.n_flags >= 2]; lo = M[M.n_flags == 0]
    if len(hi) > 20:
        print(f"  extreme groups: t1_support flagged>=2 {hi.t1_support.mean():.3f} (n={len(hi)}) "
              f"vs unflagged {lo.t1_support.mean():.3f} (n={len(lo)})", flush=True)
        yy = np.r_[np.ones(len(hi)), np.zeros(len(lo))]
        vv = np.r_[hi.t1_support.values, lo.t1_support.values]
        mk = ~np.isnan(vv)
        print(f"  extreme-group AUC (low t1 -> flagged): {roc_auc_score(yy[mk], -vv[mk]):.4f}", flush=True)
    # also: does flagging predict the OUTCOME? (sanity: reviewers' complaints matter)
    a = roc_auc_score(1 - M.y, M.n_flags)
    print(f"  n_flags -> REJECT AUC={a:.4f} (sanity: complaints should track rejection)", flush=True)
    # 3B. claim-level: reviewer-challenged claim vs our extracted claims + verdicts
    print("\n[revflag] === B. claim-level localization ===", flush=True)
    claims = {}
    for ln in open(os.path.join(EB, "claims_peerintro.jsonl")):
        try:
            r = json.loads(ln)
            if r.get("claims"):
                claims[str(r["doc_id"])] = [c["claim"] if isinstance(c, dict) else str(c)
                                            for c in r["claims"]][:4]
        except Exception: pass
    from claim_verification.run_tiered_peer import PaperDB
    pdb = PaperDB()
    vcache = Cache(f"{ROOT}/outputs/tiered_peer/cache.jsonl")
    matched, verdicts_matched, verdicts_rest = [], [], []
    conf_by_paper = conf.groupby("paper").claim.apply(list).to_dict()
    def verdict_of(pid, cl):
        pool = pdb.internals(pid.replace("iclr_", ""))
        if not pool: return None
        return verify_claim(cl, pool, CFG, vcache)["verdict"]
    todo = []
    for pid, challenged in conf_by_paper.items():
        ours = claims.get(pid, [])
        for ch in challenged:
            if not ch: continue
            best = max(ours, key=lambda c: jaccard(ch, c), default=None)
            if best and jaccard(ch, best) > 0.25:
                todo.append((pid, best, ch, True))
    # control: same papers, their OTHER extracted claims
    ctrl_pool = [(pid, c) for pid in conf_by_paper for c in claims.get(pid, [])]
    rng = np.random.default_rng(0)
    ctrl_idx = rng.choice(len(ctrl_pool), size=min(len(todo) * 2, len(ctrl_pool)), replace=False)
    todo += [(ctrl_pool[i][0], ctrl_pool[i][1], "", False) for i in ctrl_idx]
    print(f"  matched reviewer-challenged->extracted claims: "
          f"{sum(1 for x in todo if x[3])} (+{sum(1 for x in todo if not x[3])} control)", flush=True)
    def vwork(item):
        pid, cl, ch, is_m = item
        try: v = verdict_of(pid, cl)
        except Exception: return
        if v is None: return
        with lock:
            (verdicts_matched if is_m else verdicts_rest).append(v)
            if is_m: matched.append({"paper": pid, "our_claim": cl, "challenged": ch, "verdict": v})
    with ThreadPoolExecutor(max_workers=20) as ex:
        list(ex.map(vwork, todo))
    def dist(vs):
        n = max(len(vs), 1)
        return {k: round(sum(1 for v in vs if v == k) / n, 3) for k in ("FULL", "PARTIAL", "NONE")}
    print(f"  verdicts on REVIEWER-CHALLENGED claims (n={len(verdicts_matched)}): "
          f"{dist(verdicts_matched)}", flush=True)
    print(f"  verdicts on control claims same papers (n={len(verdicts_rest)}): "
          f"{dist(verdicts_rest)}", flush=True)
    with open(f"{ROOT}/outputs/reviewer_flags/matched_claims.jsonl", "w") as f:
        for r in matched: f.write(json.dumps(r) + "\n")
    # 3C. examples for the report
    print("\n[revflag] === C. examples (reviewer flag + our claim + our verdict) ===", flush=True)
    for r in matched[:6]:
        print(f"  [{r['verdict']}] {r['paper']}\n    reviewer: {r['challenged'][:140]}\n"
              f"    ours:     {r['our_claim'][:140]}", flush=True)
    M.to_csv(f"{ROOT}/outputs/reviewer_flags/paper_level.csv", index=False)
    print("REVIEWER_FLAGS_DONE", flush=True)

if __name__ == "__main__":
    main()
