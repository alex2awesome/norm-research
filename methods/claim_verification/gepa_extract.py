#!/usr/bin/env python3
"""GEPA-tune the Gemma claim-EXTRACTION prompt per domain (press releases, news articles,
peer review). Extraction needs its own objective (metric-fidelity doesn't apply):

  ext_fidelity = 0.30*reliability   (claims stable across 2 runs; token-set Jaccard matching)
              + 0.30*groundedness   (mean retrieval top1 of claims vs the doc's own body —
                                     extracted claims should be locatable, not hallucinated)
              + 0.25*coverage       (frac of head's checkable tokens — numbers/entities —
                                     covered by the claims' union)
              + 0.15*yield          (frac docs producing >=2 parseable claims)

GEPA loop: seed prompt -> evaluate -> GLM-5.2 proposes K mutations from failure summaries ->
evaluate each on probe docs -> keep best if it beats current by margin -> iterate R rounds.
Judge/extractor = Gemma (:8006); proposer = GLM-5.2 (z.ai anthropic endpoint).
Run on sk3: python -m methods.claim_verification.gepa_extract [--domains pr,news,peer]"""
import sys, os, json, re, time, argparse, random, urllib.request
sys.path.insert(0, "methods")
import numpy as np, pandas as pd
from claim_verification.core import (Cache, _post, _parse_json, split_head_body,
                                     make_passages, _key)
from claim_verification.seam_metrics import _toks, _overlap
from claim_verification.prompts import CLAIM_EXTRACT

GEMMA = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma"}
K_MUT, ROUNDS, N_PROBE, MARGIN = 4, 3, 24, 0.02

def glm(prompt, max_tokens=2000, tries=7):  # z.ai returns transient 529s under load; backoff up to ~5min
    """GLM-5.2 via z.ai anthropic endpoint (proposer)."""
    key = open(os.path.expanduser("~/.z-ai-api-key.txt")).read().strip()
    body = {"model": "glm-5.2", "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    for t in range(tries):
        try:
            req = urllib.request.Request("https://api.z.ai/api/anthropic/v1/messages",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "x-api-key": key,
                         "anthropic-version": "2023-06-01"})
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.loads(r.read())
            return "".join(b.get("text", "") for b in out.get("content", []))
        except Exception:
            if t == tries - 1: raise
            time.sleep(min(15 * (t + 1), 60))

def extract_with(prompt_tmpl, head, doc_kind, max_claims, seed_tag=""):
    p = prompt_tmpl.format(doc_kind=doc_kind, head=head[:3000], max_claims=max_claims)
    if seed_tag: p += f"\n<!-- {seed_tag} -->"   # cache-busting for reliability re-run
    raw = _post(GEMMA["base_url"], GEMMA["model"], p, max_tokens=700)
    obj = _parse_json(raw) or {}
    return [str(c.get("claim", "")) for c in obj.get("claims", [])
            if isinstance(c, dict) and c.get("claim")][:max_claims]

def _jaccard_sets(a, b):
    """Match claims across runs by token-Jaccard>0.5; return matched frac."""
    if not a or not b: return 0.0
    bt = [set(_toks(x)) for x in b]
    hit = 0
    for x in a:
        xt = set(_toks(x))
        if any(len(xt & y) / max(len(xt | y), 1) > 0.5 for y in bt): hit += 1
    return hit / max(len(a), 1)

CHECKABLE = re.compile(r"\b\d[\d,.]*%?\b|\b[A-Z][a-z]{2,}\b")

def evaluate_prompt(prompt_tmpl, docs, doc_kind, max_claims=4):
    rel, grd, cov, yld, fails = [], [], [], [], []
    for d in docs:
        head, body = split_head_body(d)
        passages = make_passages(body)
        try:
            c1 = extract_with(prompt_tmpl, head, doc_kind, max_claims)
            c2 = extract_with(prompt_tmpl, head, doc_kind, max_claims, seed_tag="rerun")
        except Exception as e:
            fails.append(f"API error: {str(e)[:60]}"); continue
        yld.append(len(c1) >= 2)
        if not c1:
            fails.append(f"ZERO claims for head: {head[:100]!r}"); continue
        rel.append((_jaccard_sets(c1, c2) + _jaccard_sets(c2, c1)) / 2)
        if passages:
            ptoks = [_toks(p) for p in passages]
            tops = [max((_overlap(_toks(c), pt) for pt in ptoks), default=0.0) for c in c1]
            grd.append(float(np.mean(tops)))
            for c, t in zip(c1, tops):
                if t < 0.25: fails.append(f"UNGROUNDED claim {c[:90]!r}")
        head_check = set(m.group(0).lower() for m in CHECKABLE.finditer(head))
        claim_toks = set(w.lower() for c in c1 for w in c.split())
        if head_check:
            cov.append(sum(1 for w in head_check if w in claim_toks) / len(head_check))
    def m(x): return float(np.mean(x)) if x else 0.0
    score = 0.30 * m(rel) + 0.30 * m(grd) + 0.25 * m(cov) + 0.15 * m(yld)
    return score, {"rel": m(rel), "grd": m(grd), "cov": m(cov), "yld": m(yld)}, fails[:12]

MUTATE = """You are optimizing a CLAIM-EXTRACTION prompt for {doc_kind}. The prompt template
(placeholders {{doc_kind}}, {{head}}, {{max_claims}} MUST be preserved) is:

---CURRENT PROMPT---
{prompt}
---END---

Scores (0-1): reliability={rel:.2f} (same claims on re-run), groundedness={grd:.2f} (claims
locatable in the document body), coverage={cov:.2f} (claims capture the head's numbers/entities),
yield={yld:.2f} (docs producing >=2 claims). Failure examples:
{fails}

Propose {k} REVISED prompt templates that fix the weakest scores. Keep the JSON output contract
(claims list with claim+kind). Return JSON: {{"prompts": ["<template1>", "<template2>", ...]}}"""

def gepa_domain(name, docs, doc_kind):
    print(f"\n===== GEPA-extract: {name} ({len(docs)} probe docs) =====", flush=True)
    cur = CLAIM_EXTRACT
    cur_score, comps, fails = evaluate_prompt(cur, docs, doc_kind)
    print(f"  seed: score={cur_score:.3f} {comps}", flush=True)
    history = [{"round": 0, "score": cur_score, **comps}]
    for r in range(1, ROUNDS + 1):
        try:
            raw = glm(MUTATE.format(doc_kind=doc_kind, prompt=cur, k=K_MUT,
                                    fails="\n".join("- " + f for f in fails) or "- (none)", **comps))
            obj = _parse_json(raw) or {}
            cands = [p for p in obj.get("prompts", []) if "{head}" in p and "{max_claims}" in p][:K_MUT]
        except Exception as e:
            print(f"  r{r}: proposer error {str(e)[:60]}", flush=True); continue
        best_c, best_s, best_comps, best_fails = None, cur_score, comps, fails
        for ci, cand in enumerate(cands):
            s, cc, ff = evaluate_prompt(cand, docs, doc_kind)
            print(f"  r{r} cand{ci}: {s:.3f} {cc}", flush=True)
            if s > best_s: best_c, best_s, best_comps, best_fails = cand, s, cc, ff
        if best_c is not None and best_s >= cur_score + MARGIN:
            cur, cur_score, comps, fails = best_c, best_s, best_comps, best_fails
            print(f"  r{r}: ACCEPT {cur_score:.3f}", flush=True)
        else:
            print(f"  r{r}: keep {cur_score:.3f}", flush=True)
        history.append({"round": r, "score": cur_score, **comps})
    return {"domain": name, "best_prompt": cur, "score": cur_score,
            "components": comps, "history": history}

def load_docs(domain, n=N_PROBE, seed=11):
    if domain == "pr":
        d = pd.read_parquet("datasets/press-releases/press_release_deconfounded.parquet")
        d = d[d.text.str.len().between(1200, 8000)]
        return d.text.sample(n, random_state=seed).tolist(), "press release"
    if domain == "news":
        import csv as _csv
        _csv.field_size_limit(sys.maxsize)
        texts = []
        for r in _csv.DictReader(open("datasets/press-releases/news_articles.csv")):
            t = (r.get("news_article_text") or "").strip()
            if 1200 < len(t) < 12000: texts.append(t)
            if len(texts) >= n * 3: break
        random.Random(seed).shuffle(texts)
        return texts[:n], "news article"
    if domain == "peer":
        d = pd.read_csv("datasets/peer-review/peer_review_modeling_dataset.csv.gz", compression="gzip")
        d = d[d.text.str.len().between(1200, 12000)]
        return d.text.sample(n, random_state=seed).tolist(), "scientific paper (abstract and introduction)"
    raise ValueError(domain)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="pr,news,peer")
    args = ap.parse_args()
    out = []
    for dom in args.domains.split(","):
        docs, kind = load_docs(dom.strip())
        out.append(gepa_domain(dom.strip(), docs, kind))
        with open("outputs/gepa_extract_results.jsonl", "w") as f:
            for r in out: f.write(json.dumps(r) + "\n")
    print("\nGEPA_EXTRACT_DONE", flush=True)
