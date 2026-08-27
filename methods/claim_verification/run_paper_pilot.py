#!/usr/bin/env python3
"""Peer-review claim-verification pilot (patents-shaped instrument, offline-batch vLLM).

Stages:
  0 CPU  rebuild SUBTRACTIVE bodies for all 2400 ICLR papers from the PDF DB (fixes the 18.5%
         empty-body bug of the additive whitelist) -> peer_review_cv_evidence_v2.jsonl; load the
         prior-art pool (all ICLR abstracts w/ year); stable-hash pilot sample (--n per run).
  1 GPU  claim extraction from abstracts (batch).
  2 CPU  per-claim passage retrieval (own body), null-twin foreign pools (hash-order derangement),
         number-perturbed claim twins, prior-art candidates (TF-IDF top-6 earlier-year + planted
         SELF abstract + deterministic foreign candidate).
  3 GPU  localize-then-verify batch: real + null + perturbed arms in ONE llm.chat call.
  4 GPU  prior-art anticipation batch.
  5 CPU  per-paper arms, instrument-validity readouts (planted controls), univariate + stacked
         AUCs (threshold-free), save metrics CSV + detail JSONL.

Run ON sk3 (gemma4 env, 1 GPU), from the repo root:
  CUDA_VISIBLE_DEVICES=2 python -m methods.claim_verification.run_paper_pilot --n 300
Old-cv dilution check afterwards (same evidence file, zero new code):
  python datasets/peer-review/extract_and_score_cv.py --evidence .../peer_review_cv_evidence_v2.jsonl
"""
import argparse, hashlib, json, pathlib, sqlite3, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from claim_verification.paper_adapter import (
    PAPER_CLAIM_EXTRACT, PAPER_LOCALIZE_VERIFY, PRIOR_ART_VERIFY,
    subtractive_body, paragraphs, select_passages, perturb_numbers, stable_pos,
    parse_claims, parse_verify, parse_prior_art,
    support_metrics, retrieval_metrics, prior_art_metrics)

BASE = pathlib.Path("/lfs/skampere3/0/alexspan/norm-research")
DB = BASE / "datasets/peer-review/peer_review_pdfs.db"
SRC = BASE / "datasets/peer-review/peer_review_cv_evidence.jsonl"
EV2 = BASE / "datasets/peer-review/peer_review_cv_evidence_v2.jsonl"
OUTD = BASE / "outputs/claimverify_paper"
GEMMA4 = "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb"
MAX_CLAIMS = 5


def shash(s):
    return hashlib.sha1(s.encode()).hexdigest()


def stage0(n_pilot):
    con = sqlite3.connect(DB)
    cur = con.cursor()
    if EV2.exists() and EV2.stat().st_size > 10_000_000:
        # reuse the cached rebuild — keeps the GPU-selection-to-engine-init race window short
        rows = [json.loads(l) for l in open(EV2) if l.strip()]
        print(f"[s0] reusing cached evidence v2: n={len(rows)} ({EV2})", flush=True)
    else:
        rows = [json.loads(l) for l in open(SRC) if l.strip()]
        cur.execute("SELECT paper_id, sections, full_text FROM pdf_versions WHERE version=0")
        raw = {pid: (sec, ft) for pid, sec, ft in cur.fetchall()}
        srcs = {"sections": 0, "fulltext_fallback": 0, "none": 0}
        for r in rows:
            forum = r["paper_id"][5:] if r["paper_id"].startswith("iclr_") else r["paper_id"]
            sec, ft = raw.get(forum, (None, None))
            body, src = subtractive_body(sec, ft)
            r["body"], srcs[src] = body, srcs[src] + 1
        with open(EV2, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"[s0] evidence v2: n={len(rows)} body_src={srcs} -> {EV2}", flush=True)

    cur.execute("SELECT paper_id, year, title, abstract FROM papers "
                "WHERE venue='ICLR' AND abstract IS NOT NULL AND LENGTH(abstract)>100")
    pool = [{"pid": p, "year": int(y) if y else None, "title": t or "", "abstract": a}
            for p, y, t, a in cur.fetchall() if y]
    years = {}
    forums = [r["paper_id"][5:] if r["paper_id"].startswith("iclr_") else r["paper_id"] for r in rows]
    for chunk in range(0, len(forums), 900):
        q = forums[chunk:chunk + 900]
        cur.execute(f"SELECT paper_id, year FROM papers WHERE paper_id IN ({','.join('?' * len(q))})", q)
        years.update({p: int(y) for p, y in cur.fetchall() if y})
    con.close()
    for r, f in zip(rows, forums):
        r["year"] = years.get(f)
        r["forum"] = f

    # stable-hash pilot sample, balanced per class (never seeded-shuffle a growing list)
    usable = [r for r in rows if len(r["body"]) > 2000]
    pos = sorted((r for r in usable if r["y"] == 1), key=lambda r: shash(r["paper_id"]))
    neg = sorted((r for r in usable if r["y"] == 0), key=lambda r: shash(r["paper_id"]))
    k = n_pilot // 2 if n_pilot else min(len(pos), len(neg))
    pilot = pos[:k] + neg[:k]
    print(f"[s0] usable={len(usable)}/{len(rows)} pilot={len(pilot)} "
          f"({sum(r['y'] for r in pilot)} pos) pool={len(pool)} w/year={sum(1 for r in pilot if r['year'])}",
          flush=True)
    return pilot, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--util", type=float, default=0.90)
    ap.add_argument("--k-passages", type=int, default=8)
    ap.add_argument("--k-pa", type=int, default=8)
    ap.add_argument("--evidence-only", action="store_true")
    a = ap.parse_args()

    OUTD.mkdir(parents=True, exist_ok=True)
    pilot, pool = stage0(a.n)
    if a.evidence_only:
        print("EVIDENCE_ONLY_DONE", flush=True)
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=a.util,
              max_model_len=8192, enable_prefix_caching=True, trust_remote_code=True)

    # ---- stage 1: claim extraction ----
    convs = [[{"role": "user", "content": PAPER_CLAIM_EXTRACT.format(
        head=r["abstract"][:5000], max_claims=MAX_CLAIMS)}] for r in pilot]
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=500))
    for r, o in zip(pilot, outs):
        r["claims"] = parse_claims(o.outputs[0].text, MAX_CLAIMS)
    n_claims = sum(len(r["claims"]) for r in pilot)
    print(f"[s1] {n_claims} claims from {len(pilot)} abstracts "
          f"(0-claim papers: {sum(1 for r in pilot if not r['claims'])})", flush=True)

    # ---- stage 2: pools, twins, prior-art candidates ----
    for r in pilot:
        r["paras"] = paragraphs(r["body"])
    order = sorted(range(len(pilot)), key=lambda i: shash(pilot[i]["paper_id"]))
    partner = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}  # derangement

    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    vec = TfidfVectorizer(max_features=60000, stop_words="english", sublinear_tf=True)
    P = vec.fit_transform([p["title"] + " " + p["abstract"] for p in pool])
    pool_years = np.array([p["year"] for p in pool])

    vreqs, pareqs = [], []  # (paper_idx, claim_idx, arm, passages) / (paper_idx, claim_idx, cands, self_idx, foreign_idx)
    for i, r in enumerate(pilot):
        foreign_paras = pilot[partner[i]]["paras"]
        for j, c in enumerate(r["claims"]):
            ps = select_passages(c["claim"], r["paras"], k=a.k_passages)
            vreqs.append((i, j, "real", ps))
            vreqs.append((i, j, "null", select_passages(c["claim"], foreign_paras, k=a.k_passages)))
            pert = perturb_numbers(c["claim"])
            if pert:
                vreqs.append((i, j, "pert", ps, pert))
            if r["year"]:
                elig = np.where(pool_years < r["year"])[0]
                if len(elig) >= 20:
                    q = vec.transform([c["claim"] + " " + r["abstract"][:400]])
                    sims = (P[elig] @ q.T).toarray().ravel()
                    top = elig[np.argsort(-sims)[:a.k_pa - 2]]
                    cands = [{"title": pool[t]["title"], "abstract": pool[t]["abstract"], "kind": "real"}
                             for t in top]
                    self_c = {"title": "", "abstract": r["abstract"], "kind": "self"}
                    foreign = pool[int(elig[stable_pos(r["paper_id"] + str(j), len(elig))])]
                    cands.append({"title": foreign["title"], "abstract": foreign["abstract"], "kind": "foreign"})
                    si = stable_pos(r["paper_id"] + str(j) + "self", len(cands) + 1)
                    cands.insert(si, self_c)
                    fi = next(ii for ii, cd in enumerate(cands) if cd["kind"] == "foreign")
                    pareqs.append((i, j, cands, si, fi))
    print(f"[s2] verify reqs={len(vreqs)} prior-art reqs={len(pareqs)}", flush=True)

    # ---- stage 3: verify (real + null + pert in one batch) ----
    convs = []
    for req in vreqs:
        i, j, arm, ps = req[0], req[1], req[2], req[3]
        claim = req[4] if arm == "pert" else pilot[i]["claims"][j]["claim"]
        ptxt = "\n".join(f"[{k}] {p}" for k, p in enumerate(ps))
        convs.append([{"role": "user", "content": PAPER_LOCALIZE_VERIFY.format(claim=claim, passages=ptxt)}])
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=320))
    verd = {}
    for req, o in zip(vreqs, outs):
        i, j, arm, ps = req[0], req[1], req[2], req[3]
        verd[(i, j, arm)] = parse_verify(o.outputs[0].text, ps)
    from collections import Counter
    for arm in ("real", "null", "pert"):
        h = Counter(v["verdict"] for k, v in verd.items() if k[2] == arm)
        pf = sum(1 for k, v in verd.items() if k[2] == arm and not v["parsed"])
        print(f"[s3] {arm:5s} verdicts={dict(h)} parse_fail={pf}", flush=True)

    # ---- stage 4: prior-art ----
    convs = []
    for i, j, cands, si, fi in pareqs:
        ctxt = "\n\n".join(f"[{k}] {c['title'][:150]}\n{c['abstract'][:900]}" for k, c in enumerate(cands))
        convs.append([{"role": "user", "content": PRIOR_ART_VERIFY.format(
            claim=pilot[i]["claims"][j]["claim"], candidates=ctxt)}])
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=260))
    pa = {}
    for (i, j, cands, si, fi), o in zip(pareqs, outs):
        res = parse_prior_art(o.outputs[0].text, len(cands))
        res["self_idx"], res["foreign_idx"] = si, fi
        pa[(i, j)] = res
    h = Counter(v for r in pa.values() for v in r["verdicts"])
    print(f"[s4] pa verdicts={dict(h)} parse_fail={sum(1 for r in pa.values() if not r['parsed'])}", flush=True)

    # ---- stage 5: aggregate + readouts ----
    import pandas as pd
    recs, details = [], []
    for i, r in enumerate(pilot):
        cl = [c["claim"] for c in r["claims"]]
        real = [verd[(i, j, "real")] for j in range(len(cl)) if (i, j, "real") in verd]
        null = [verd[(i, j, "null")] for j in range(len(cl)) if (i, j, "null") in verd]
        pert = [verd[(i, j, "pert")] for j in range(len(cl)) if (i, j, "pert") in verd]
        pa_rows = [pa[(i, j)] for j in range(len(cl)) if (i, j) in pa]
        m = {"paper_id": r["paper_id"], "y": r["y"], "year": r["year"],
             "n_claims": len(cl), "n_pert": len(pert), "n_pa": len(pa_rows)}
        m.update(support_metrics(real, "s_"))
        m.update({("null_" + k): v for k, v in support_metrics(null, "s_").items()})
        m.update({("pert_" + k): v for k, v in support_metrics(pert, "s_").items()})
        m.update(retrieval_metrics(cl, r["paras"], prefix="r_"))
        m.update({("null_" + k): v for k, v in
                  retrieval_metrics(cl, pilot[partner[i]]["paras"], prefix="r_").items()})
        m.update(prior_art_metrics(pa_rows))
        recs.append(m)
        details.append({"paper_id": r["paper_id"], "y": r["y"], "claims": r["claims"],
                        "real": real, "null": null, "pert": pert,
                        "pa": [{k: v for k, v in p.items() if k != "parsed"} for p in pa_rows]})
    M = pd.DataFrame(recs)
    M.to_csv(OUTD / "paper_pilot_metrics.csv", index=False)
    with open(OUTD / "paper_pilot_detail.jsonl", "w") as fh:
        for d in details:
            fh.write(json.dumps(d) + "\n")

    import numpy as np
    y = M["y"].values
    print("\n=== INSTRUMENT VALIDITY (planted controls) ===", flush=True)
    for real_c, null_c in (("s_support_rate", "null_s_support_rate"),
                           ("r_mean_top1", "null_r_mean_top1")):
        vr, vn = M[real_c].values, M[null_c].values
        mk = ~(np.isnan(vr) | np.isnan(vn))
        print(f"  {real_c:20s} real={np.nanmean(vr):.3f} null={np.nanmean(vn):.3f} "
              f"gap={np.nanmean(vr[mk] - vn[mk]):+.3f} "
              f"(papers w/ real>null: {np.mean(vr[mk] > vn[mk]):.2%})", flush=True)
    mk = M["n_pert"].values > 0
    print(f"  pert twin (n={mk.sum()} papers): FULL real={M.loc[mk, 's_support_rate'].mean():.3f} "
          f"perturbed={M.loc[mk, 'pert_s_support_rate'].mean():.3f}", flush=True)
    print(f"  pa_self_detect={M['pa_self_detect'].mean():.3f} (want ~1) "
          f"pa_foreign_distinct={M['pa_foreign_distinct'].mean():.3f} (want ~1)", flush=True)

    print("\n=== ARM AUCs (univariate, threshold-free) ===", flush=True)
    from sklearn.metrics import roc_auc_score
    for c in [c for c in M.columns if c[0] in "rsnp" and c not in ("paper_id", "year")]:
        v = M[c].values.astype(float)
        mk = ~np.isnan(v)
        if mk.sum() > 60 and len(set(y[mk])) == 2 and np.nanstd(v) > 1e-9:
            print(f"  {c:26s} AUC={roc_auc_score(y[mk], v[mk]):.3f} (n={mk.sum()})", flush=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    print("\n=== STACKS (logistic, 5-fold CV) ===", flush=True)
    for label, cols in (("s_* support", [c for c in M.columns if c.startswith("s_")]),
                        ("r_* retrieval", [c for c in M.columns if c.startswith("r_")]),
                        ("pa_* prior-art", ["pa_anticipated_rate", "pa_partial_rate"]),
                        ("FULL stack", [c for c in M.columns if c.startswith(("s_", "r_", "pa_anticipated", "pa_partial"))])):
        X = M[cols].values.astype(float)
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=3000, class_weight="balanced"))
        auc = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              scoring="roc_auc").mean()
        print(f"  {label:16s} AUC={auc:.3f} ({len(cols)} feats)", flush=True)
    print("\nrefs: V_regex=.635 V_code_fullpaper=.682 A_judge=.676 old-cv all ~.50 (18.5% empty bodies)",
          flush=True)
    print("PAPER_PILOT_DONE", flush=True)


if __name__ == "__main__":
    main()
