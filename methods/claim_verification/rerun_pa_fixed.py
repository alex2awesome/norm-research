#!/usr/bin/env python3
"""Re-run ONLY the prior-art stage of the paper pilot with echo-indexed verdicts.

The pilot's positional verdicts array proved misaligned (self-detect .483 positional vs .856 by
the judge's own best_idx; 175 provable shifts). Claims are read back from the pilot's detail
JSONL; candidates are rebuilt deterministically (same TF-IDF pool, same stable-hash planting).

Run ON sk3 (gemma4 env, 1 GPU): python -m methods.claim_verification.rerun_pa_fixed
"""
import json, pathlib, sqlite3, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from claim_verification.paper_adapter import (PRIOR_ART_VERIFY, stable_pos, parse_prior_art,
                                              prior_art_metrics)

BASE = pathlib.Path("/lfs/skampere3/0/alexspan/norm-research")
DB = BASE / "datasets/peer-review/peer_review_pdfs.db"
EV2 = BASE / "datasets/peer-review/peer_review_cv_evidence_v2.jsonl"
DETAIL = BASE / "outputs/claimverify_paper/paper_pilot_detail.jsonl"
OUTD = BASE / "outputs/claimverify_paper"
GEMMA4 = "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb"
K_PA = 8


def main():
    detail = [json.loads(l) for l in open(DETAIL)]
    ev = {r["paper_id"]: r for r in (json.loads(l) for l in open(EV2))}
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("SELECT paper_id, year, title, abstract FROM papers "
                "WHERE venue='ICLR' AND abstract IS NOT NULL AND LENGTH(abstract)>100")
    pool = [{"pid": p, "year": int(y) if y else None, "title": t or "", "abstract": a}
            for p, y, t, a in cur.fetchall() if y]
    forums = [d["paper_id"][5:] if d["paper_id"].startswith("iclr_") else d["paper_id"] for d in detail]
    years = {}
    for chunk in range(0, len(forums), 900):
        q = forums[chunk:chunk + 900]
        cur.execute(f"SELECT paper_id, year FROM papers WHERE paper_id IN ({','.join('?' * len(q))})", q)
        years.update({p: int(y) for p, y in cur.fetchall() if y})
    con.close()

    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    vec = TfidfVectorizer(max_features=60000, stop_words="english", sublinear_tf=True)
    P = vec.fit_transform([p["title"] + " " + p["abstract"] for p in pool])
    pool_years = np.array([p["year"] for p in pool])

    pareqs = []
    for i, (d, forum) in enumerate(zip(detail, forums)):
        year = years.get(forum)
        abstract = ev[d["paper_id"]]["abstract"]
        if not year:
            continue
        elig = np.where(pool_years < year)[0]
        if len(elig) < 20:
            continue
        for j, c in enumerate(d["claims"]):
            q = vec.transform([c["claim"] + " " + abstract[:400]])
            sims = (P[elig] @ q.T).toarray().ravel()
            top = elig[np.argsort(-sims)[:K_PA - 2]]
            cands = [{"title": pool[t]["title"], "abstract": pool[t]["abstract"]} for t in top]
            self_c = {"title": "", "abstract": abstract}
            foreign = pool[int(elig[stable_pos(d["paper_id"] + str(j), len(elig))])]
            cands.append({"title": foreign["title"], "abstract": foreign["abstract"], "kind": "foreign"})
            si = stable_pos(d["paper_id"] + str(j) + "self", len(cands) + 1)
            cands.insert(si, self_c)
            fi = next(ii for ii, cd in enumerate(cands) if cd.get("kind") == "foreign")
            pareqs.append((i, j, cands, si, fi))
    print(f"[pa2] {len(pareqs)} prior-art reqs rebuilt for {len(detail)} papers", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=8192, enable_prefix_caching=True, trust_remote_code=True)
    convs = []
    for i, j, cands, si, fi in pareqs:
        ctxt = "\n\n".join(f"[{k}] {c['title'][:150]}\n{c['abstract'][:900]}" for k, c in enumerate(cands))
        convs.append([{"role": "user", "content": PRIOR_ART_VERIFY.format(
            claim=detail[i]["claims"][j]["claim"], candidates=ctxt)}])
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=420))

    from collections import Counter
    pa = {}
    for (i, j, cands, si, fi), o in zip(pareqs, outs):
        res = parse_prior_art(o.outputs[0].text, len(cands))
        res["self_idx"], res["foreign_idx"] = si, fi
        pa.setdefault(i, {})[j] = res
    h = Counter(v for byj in pa.values() for r in byj.values() for v in r["verdicts"])
    pf = sum(1 for byj in pa.values() for r in byj.values() if not r["parsed"])
    print(f"[pa2] verdicts={dict(h)} parse_fail={pf}", flush=True)

    import pandas as pd
    from sklearn.metrics import roc_auc_score
    recs = []
    for i, d in enumerate(detail):
        rows = [pa[i][j] for j in sorted(pa.get(i, {}))]
        m = {"paper_id": d["paper_id"], "y": d["y"], "n_pa": len(rows)}
        m.update(prior_art_metrics(rows))
        recs.append(m)
        d["pa_v2"] = [{k: v for k, v in r.items() if k != "parsed"} for r in rows]
    M = pd.DataFrame(recs)
    M.to_csv(OUTD / "paper_pilot_pa_v2.csv", index=False)
    with open(OUTD / "paper_pilot_detail_pa_v2.jsonl", "w") as fh:
        for d in detail:
            fh.write(json.dumps(d) + "\n")

    print("\n=== PA v2 (echo-indexed) INSTRUMENT ===", flush=True)
    print(f"  pa_self_detect={M['pa_self_detect'].mean():.3f} (positional run: .483; best_idx bound: .856)",
          flush=True)
    print(f"  pa_foreign_distinct={M['pa_foreign_distinct'].mean():.3f}", flush=True)
    y = M["y"].values
    for c in ("pa_anticipated_rate", "pa_partial_rate"):
        v = M[c].values.astype(float)
        mk = ~np.isnan(v)
        if mk.sum() > 60:
            print(f"  {c:22s} AUC={roc_auc_score(y[mk], v[mk]):.3f} (n={mk.sum()})", flush=True)
    print("PA_V2_DONE", flush=True)


if __name__ == "__main__":
    main()
