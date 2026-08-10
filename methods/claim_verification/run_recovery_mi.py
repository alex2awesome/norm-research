#!/usr/bin/env python3
"""Recovery-MI for claim EXTRACTION (canonical recovery leg, per recovery_reconstruction_no_anchor):
Can a STRONG recoverer (GLM-5.2) infer the extraction rule from k exemplars and reproduce the
extractor's outputs on held-out docs?

Protocol per domain:
  1. Sample k=6 (doc-head, Gemma-claims) exemplar pairs + m=30 held-out docs.
  2. GLM sees the exemplars ONLY (not the prompt), infers the rule, and states it.
  3. GLM applies its inferred rule to each held-out head -> claims_hat.
  4. Agreement = symmetric token-Jaccard claim matching (same matcher as GEPA reliability):
     recovery = mean over docs of match(claims_gemma, claims_glm).
  5. Baselines: (a) floor = agreement between Gemma-claims of doc_i and GLM-claims of doc_j
     (mismatched pairs; controls for generic claim-extraction overlap);
     (b) ceiling = Gemma re-run reliability (from GEPA: pr .97 / news .99 / peer .96).
  recovery >> floor and near ceiling => extraction rule is RECOVERABLE = high I(M;M_hat).
Run on sk3: python -m methods.claim_verification.run_recovery_mi"""
import json, os, sys, time, random
sys.path.insert(0, "methods")
import numpy as np
from claim_verification.core import _parse_json, _sentences
from claim_verification.seam_metrics import _toks
from claim_verification.gepa_extract import glm   # GLM-5.2 via z.ai (proposer creds)

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
EB = os.path.join(ROOT, "datasets/evidence_bases")
K_EX, M_HELD = 6, 30

def _match(a, b):
    """Symmetric claim-set agreement via token-Jaccard>0.5 matching (GEPA matcher)."""
    if not a or not b: return 0.0
    def one(x, ys):
        yt = [set(_toks(s)) for s in ys]
        hit = 0
        for s in x:
            st = set(_toks(s))
            if any(len(st & y) / max(len(st | y), 1) > 0.5 for y in yt): hit += 1
        return hit / max(len(x), 1)
    return (one(a, b) + one(b, a)) / 2

INFER = """Here are {k} examples of an unknown text-analysis procedure. Each example shows a
document excerpt (INPUT) and the procedure's output (OUTPUT).

{exemplars}

Infer the procedure's rule. State it in 2-3 sentences. Return ONLY JSON:
{{"rule": "<your inferred rule>"}}"""

APPLY = """You inferred this text-analysis rule:
RULE: {rule}

Apply it to this document excerpt. Produce the output in the same JSON format as the examples
(a list under key "claims", each item a short sentence string).

INPUT:
{head}

Return ONLY JSON: {{"claims": ["<...>", ...]}}"""

def head_of(doc_text, domain):
    sents = _sentences(doc_text) if domain != "pr" else None
    if domain == "pr":
        from claim_verification.core import split_head_body
        h, _ = split_head_body(doc_text)
        return h[:1500]
    return " ".join((sents or [""])[:5])[:1500]

def load_pairs(domain, src_texts):
    """(head, claims) pairs from the claims files, joined to doc text for the head."""
    claims_file = {"pr": "claims_pr.jsonl", "newsfull": "claims_newsfull.jsonl",
                   "peerintro": "claims_peerintro.jsonl"}[domain]
    pairs = []
    for ln in open(os.path.join(EB, claims_file)):
        try:
            r = json.loads(ln)
            if not r.get("claims"): continue
            t = src_texts.get(str(r["doc_id"]))
            if not t or len(t) < 600: continue
            cl = [c["claim"] if isinstance(c, dict) else str(c) for c in r["claims"]][:4]
            pairs.append((head_of(t, domain), cl))
        except Exception: pass
        if len(pairs) >= 400: break
    return pairs

def source_texts(domain):
    out = {}
    if domain == "pr":
        import pandas as pd
        d = pd.read_parquet(f"{ROOT}/datasets/press-releases/press_release_deconfounded.parquet")
        out = {str(r.id): str(r.text) for r in d.itertuples()}
    elif domain == "peerintro":
        import sqlite3
        con = sqlite3.connect(f"{ROOT}/datasets/peer-review/peer_review_pdfs.db")
        for pid, sections in con.execute("SELECT paper_id, sections FROM pdf_versions WHERE version=0"):
            try: s = json.loads(sections) if sections else {}
            except Exception: continue
            t = (s.get("abstract", "") or "") + "\n\n" + (s.get("introduction", "") or "")
            if len(t) > 600: out[f"iclr_{pid}"] = t
    else:  # newsfull
        import glob as _g
        for path in _g.glob(f"{ROOT}/datasets/news-homepages/fulltext/fulltext_v2_shard*.jsonl") + \
                    [f"{ROOT}/datasets/news-homepages/fulltext/fulltext.jsonl"]:
            if not os.path.exists(path): continue
            for ln in open(path):
                try:
                    r = json.loads(ln)
                    if r.get("route") != "FAIL" and r.get("text_len", 0) > 600:
                        out[r["url"]] = r["text"]
                except Exception: pass
            if len(out) > 60000: break
    return out

def run_domain(domain, seed=13):
    rng = random.Random(seed)
    texts = source_texts(domain)
    pairs = load_pairs(domain, texts)
    if len(pairs) < K_EX + M_HELD:
        print(f"[recov-{domain}] insufficient pairs ({len(pairs)})", flush=True); return None
    rng.shuffle(pairs)
    ex, held = pairs[:K_EX], pairs[K_EX:K_EX + M_HELD]
    ex_txt = "\n\n".join(f"INPUT:\n{h}\n\nOUTPUT:\n" + json.dumps({"claims": c})
                          for h, c in ex)
    raw = glm(INFER.format(k=K_EX, exemplars=ex_txt), max_tokens=500)
    rule = (( _parse_json(raw) or {}).get("rule") or "").strip()
    print(f"[recov-{domain}] inferred rule: {rule[:180]}", flush=True)
    if not rule: return None
    agree, floor = [], []
    held_claims_glm = []
    for h, c_gemma in held:
        try:
            raw2 = glm(APPLY.format(rule=rule, head=h), max_tokens=600)
            c_glm = [(str(x)) for x in ((_parse_json(raw2) or {}).get("claims") or [])][:4]
        except Exception:
            c_glm = []
        held_claims_glm.append(c_glm)
        agree.append(_match(c_gemma, c_glm))
        time.sleep(0.5)
    # floor: mismatched pairing (doc_i gemma vs doc_j glm)
    for i in range(len(held)):
        j = (i + 1) % len(held)
        floor.append(_match(held[i][1], held_claims_glm[j]))
    rec, flo = float(np.mean(agree)), float(np.mean(floor))
    print(f"[recov-{domain}] RECOVERY={rec:.3f}  floor={flo:.3f}  (n={len(held)})", flush=True)
    return {"domain": domain, "recovery": rec, "floor": flo, "rule": rule,
            "n_held": len(held), "per_doc": agree}

if __name__ == "__main__":
    out = []
    for dom in ("pr", "newsfull", "peerintro"):
        r = run_domain(dom)
        if r: out.append(r)
        with open(f"{ROOT}/outputs/recovery_mi_extraction.json", "w") as f:
            json.dump(out, f, indent=1)
    print("\n[recov] === SUMMARY (ceiling = Gemma re-run reliability from GEPA) ===", flush=True)
    ceil = {"pr": 0.97, "newsfull": 0.99, "peerintro": 0.96}
    for r in out:
        print(f"  {r['domain']:10} recovery={r['recovery']:.3f} floor={r['floor']:.3f} "
              f"ceiling~{ceil.get(r['domain'], float('nan')):.2f}", flush=True)
    print("RECOVERY_MI_DONE", flush=True)
