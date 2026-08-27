#!/usr/bin/env python3
"""PLANTED POSITIVE CONTROL for news claim verification (closes the range-restriction
argument): credible-outlet news sits at the support ceiling (t1_support median 1.0), so
chance-level outcome prediction could mean "no variance" OR "pipeline blind". Here we
MANUFACTURE variance: corrupt real head claims (number-swap, entity-swap) and re-verify
against the SAME article. If the pipeline works, corrupted claims drop FULL -> PARTIAL/
NONE at high rate while originals stay FULL. Paired design + 4 synthetic anchors.
Run on sk3: python -m methods.claim_verification.run_planted_news"""
import glob, hashlib, json, re, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np, pandas as pd
from claim_verification.core import (Cache, extract_claims, verify_claim,
                                     split_head_body, make_passages)
from claim_verification.evidence_api import clean_evidence_text

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
NH = f"{ROOT}/datasets/news-homepages"
CFG = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma",
       "doc_kind": "news article", "max_claims": 6}
N_DOCS = 250

def corrupt_number(c):
    """Perturb every number in the claim (x -> ~2.7x+13, keeps plausibility)."""
    def rep(m):
        try: v = float(m.group(0).replace(",", ""))
        except Exception: return m.group(0)
        nv = int(round(v * 2.7 + 13))
        return str(nv)
    out = re.sub(r"\d[\d,]*(?:\.\d+)?", rep, c)
    return out if out != c else None

ENT = re.compile(r"(?<!^)(?<![.!?] )\b([A-Z][a-z]{2,}(?: [A-Z][a-z]{2,}){0,2})\b")

def corrupt_entity(c, pool):
    """Swap the first mid-sentence proper-noun span with an entity from another doc."""
    m = ENT.search(c)
    if not m: return None
    repl = next((e for e in pool if e != m.group(1) and e not in c), None)
    if not repl: return None
    return c[:m.start(1)] + repl + c[m.end(1):]

ANCHORS = [
    # (claim, corrupted, article, expected orig, expected corrupted)
    ("The council approved a 12 million dollar budget for road repairs.",
     "The council approved a 45 million dollar budget for road repairs.",
     "City leaders met Tuesday night. The council approved a 12 million dollar budget "
     "for road repairs after months of debate. Work begins in the spring, officials said, "
     "starting with the downtown corridor.", "FULL", "NONE"),
    ("Riverton Hospital opened a new pediatric wing on Monday.",
     "Lakeside Clinic opened a new pediatric wing on Monday.",
     "A ribbon-cutting ceremony drew hundreds. Riverton Hospital opened a new pediatric "
     "wing on Monday, doubling its capacity for young patients. The wing includes forty "
     "beds and a dedicated imaging suite, administrators said.", "FULL", "NONE"),
]

def main():
    M = pd.read_csv(f"{ROOT}/outputs/multi_y_news/metrics_audited.csv")
    urls = set(M.url)
    text = {}
    for p in glob.glob(f"{NH}/fulltext/fulltext_v2_shard*.jsonl"):
        for ln in open(p):
            try: r = json.loads(ln)
            except Exception: continue
            if r.get("url") in urls and len(r.get("text") or "") > 400:
                text[r["url"]] = clean_evidence_text(r["text"])
    # stable-hash doc selection
    docs = sorted(text.items(), key=lambda kv: hashlib.md5(kv[0].encode()).hexdigest())[:N_DOCS]
    print(f"[plant] {len(docs)} docs selected", flush=True)
    cache = Cache(f"{ROOT}/outputs/attrib_adequacy/cache.jsonl")  # reuse: extraction cached
    vcache = Cache(f"{ROOT}/outputs/planted_news/cache.jsonl")
    import os; os.makedirs(f"{ROOT}/outputs/planted_news", exist_ok=True)
    # entity pool from other docs' claims
    pool_ents = []
    tasks, lock, rows = [], Lock(), []
    plan = []
    for u, t in docs:
        head, body = split_head_body(t)
        try: claims = extract_claims(head, CFG, cache)
        except Exception: continue
        cls = [c["claim"] if isinstance(c, dict) else str(c) for c in claims][:4]
        for c in cls:
            m = ENT.search(c)
            if m: pool_ents.append(m.group(1))
        plan.append((u, cls, make_passages(body)))
    rng = np.random.default_rng(0); rng.shuffle(pool_ents)
    n_num = n_ent = n_skip = 0
    for u, cls, passages in plan:
        for c in cls:
            cc = corrupt_number(c); kind = "number"
            if cc is None:
                cc = corrupt_entity(c, pool_ents); kind = "entity"
            if cc is None:
                n_skip += 1; continue
            if kind == "number": n_num += 1
            else: n_ent += 1
            tasks.append((u, c, cc, kind, passages))
    print(f"[plant] pairs: {len(tasks)} (number {n_num}, entity {n_ent}, skipped {n_skip})", flush=True)
    def work(t):
        u, c, cc, kind, passages = t
        try:
            vo = verify_claim(c, passages, CFG, vcache)
            vc = verify_claim(cc, passages, CFG, vcache)
        except Exception:
            return
        with lock:
            rows.append({"url": u, "kind": kind, "claim": c[:110], "corrupted": cc[:110],
                         "v_orig": vo["verdict"], "v_corr": vc["verdict"]})
            if len(rows) % 100 == 0: print(f"[plant] {len(rows)}/{len(tasks)}", flush=True)
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(work, tasks))
    # anchors
    ok = 0
    for c, cc, art, e1, e2 in ANCHORS:
        _, body = split_head_body(art)
        ps = make_passages(body) or [art]
        v1 = verify_claim(c, ps, CFG, vcache)["verdict"]
        v2 = verify_claim(cc, ps, CFG, vcache)["verdict"]
        ok += int(v1 == e1) + int(v2 in (e2, "PARTIAL"))
    print(f"[plant] ANCHORS {ok}/4", flush=True)
    F = pd.DataFrame(rows)
    F.to_csv(f"{ROOT}/outputs/planted_news/results.csv", index=False)
    print("\n[plant] verdict distributions (paired):", flush=True)
    for col in ("v_orig", "v_corr"):
        print(f"  {col}: {F[col].value_counts(normalize=True).round(3).to_dict()}", flush=True)
    for kind, g in F.groupby("kind"):
        fo = g[g.v_orig == "FULL"]
        det = (fo.v_corr != "FULL").mean() if len(fo) else float("nan")
        none_r = (fo.v_corr == "NONE").mean() if len(fo) else float("nan")
        print(f"  {kind:7} n={len(g):4d} orig FULL {(g.v_orig=='FULL').mean():.3f} | "
              f"DETECTION (FULL->not-FULL) {det:.3f} | FULL->NONE {none_r:.3f} (n_pairs={len(fo)})", flush=True)
    fo = F[F.v_orig == "FULL"]
    print(f"\n[plant] OVERALL detection: {(fo.v_corr != 'FULL').mean():.3f} "
          f"(FULL->NONE {(fo.v_corr == 'NONE').mean():.3f}, n={len(fo)}); "
          f"orig stays FULL {(F.v_orig == 'FULL').mean():.3f}", flush=True)
    ex_ = fo[fo.v_corr == "FULL"].head(3)
    for _, r in ex_.iterrows():
        print(f"  MISSED [{r.kind}]: '{r.corrupted[:90]}'", flush=True)
    print("PLANTED_NEWS_DONE", flush=True)

if __name__ == "__main__":
    main()
