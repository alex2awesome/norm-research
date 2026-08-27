#!/usr/bin/env python3
"""Pilot follow-up, two experiments in one engine session (same 300 papers / 1,378 claims):

  G — GRADED verdicts (0-4 support scale) on the body leg, real + null-twin + perturbed arms.
      Tests whether the binary verifier's FULL-conservatism (5.9% FULL) hides discrimination
      signal. If graded mean-support is still ~.50 AUC, the null is about the construct.
  F — FULL-TEXT prior-art verification, per-ref calls (patents-faithful: candidate evidence is
      excerpts of the prior paper's BODY, not its abstract; one candidate per call so there is
      no verdict-array alignment to get wrong). Planted SELF (own body) + foreign controls.

Reads claims back from the pilot detail JSONL. Run ON sk3 (gemma4 env, 1 GPU):
  python -m methods.claim_verification.run_graded_ft
"""
import hashlib, json, pathlib, sqlite3, sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from claim_verification.paper_adapter import (
    GRADED_VERIFY, PRIOR_ART_VERIFY_FT, subtractive_body, paragraphs, select_passages,
    perturb_numbers, stable_pos, parse_graded, parse_pa_single, graded_metrics, ft_pa_metrics)

BASE = pathlib.Path("/lfs/skampere3/0/alexspan/norm-research")
DB = BASE / "datasets/peer-review/peer_review_pdfs.db"
EV2 = BASE / "datasets/peer-review/peer_review_cv_evidence_v2.jsonl"
DETAIL = BASE / "outputs/claimverify_paper/paper_pilot_detail.jsonl"
OUTD = BASE / "outputs/claimverify_paper"
GEMMA4 = "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb"
K_REAL_FT = 4


def shash(s):
    return hashlib.sha1(s.encode()).hexdigest()


def main():
    detail = [json.loads(l) for l in open(DETAIL)]
    ev = {r["paper_id"]: r for r in (json.loads(l) for l in open(EV2))}
    forums = [d["paper_id"][5:] if d["paper_id"].startswith("iclr_") else d["paper_id"] for d in detail]

    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("SELECT p.paper_id, p.year, p.title, p.abstract FROM papers p "
                "JOIN pdf_versions v ON p.paper_id=v.paper_id AND v.version=0 "
                "WHERE p.venue='ICLR' AND p.abstract IS NOT NULL AND LENGTH(p.abstract)>100 "
                "AND p.year IS NOT NULL")
    pool = [{"pid": p, "year": int(y), "title": t or "", "abstract": a} for p, y, t, a in cur.fetchall()]
    years = {}
    for chunk in range(0, len(forums), 900):
        q = forums[chunk:chunk + 900]
        cur.execute(f"SELECT paper_id, year FROM papers WHERE paper_id IN ({','.join('?' * len(q))})", q)
        years.update({p: int(y) for p, y in cur.fetchall() if y})
    print(f"[gf] pool(w/ fulltext)={len(pool)} pilot={len(detail)}", flush=True)

    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=60000, stop_words="english", sublinear_tf=True)
    P = vec.fit_transform([p["title"] + " " + p["abstract"] for p in pool])
    pool_years = np.array([p["year"] for p in pool])

    # own paragraph pools + derangement partners (same scheme as the pilot)
    for d in detail:
        d["paras"] = paragraphs(ev[d["paper_id"]]["body"])
    order = sorted(range(len(detail)), key=lambda i: shash(detail[i]["paper_id"]))
    partner = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}

    # ---- plan FT candidates, collect candidate forums needing bodies ----
    ftreqs = []  # (paper_i, claim_j, kind, pool_idx or None)
    need_forums = set()
    for i, (d, forum) in enumerate(zip(detail, forums)):
        year = years.get(forum)
        if not year:
            continue
        elig = np.where(pool_years < year)[0]
        if len(elig) < 20:
            continue
        for j, c in enumerate(d["claims"]):
            q = vec.transform([c["claim"] + " " + ev[d["paper_id"]]["abstract"][:400]])
            sims = (P[elig] @ q.T).toarray().ravel()
            for t in elig[np.argsort(-sims)[:K_REAL_FT]]:
                ftreqs.append((i, j, "real", int(t)))
                need_forums.add(pool[int(t)]["pid"])
            ftreqs.append((i, j, "self", None))
            fidx = int(elig[stable_pos(d["paper_id"] + str(j) + "ft", len(elig))])
            ftreqs.append((i, j, "foreign", fidx))
            need_forums.add(pool[fidx]["pid"])
    print(f"[gf] FT reqs={len(ftreqs)} over {len(need_forums)} candidate papers", flush=True)

    cand_paras = {}
    need = sorted(need_forums)
    for chunk in range(0, len(need), 400):
        q = need[chunk:chunk + 400]
        cur.execute(f"SELECT paper_id, sections, full_text FROM pdf_versions "
                    f"WHERE version=0 AND paper_id IN ({','.join('?' * len(q))})", q)
        for pid, sec, ft in cur.fetchall():
            body, _ = subtractive_body(sec, ft)
            cand_paras[pid] = paragraphs(body) if len(body) > 2000 else None
    con.close()
    print(f"[gf] candidate bodies: {sum(1 for v in cand_paras.values() if v)}/{len(cand_paras)} usable",
          flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=8192, enable_prefix_caching=True, trust_remote_code=True)

    # ---- batch G: graded verify (real + null + pert) ----
    greqs, convs = [], []
    for i, d in enumerate(detail):
        for j, c in enumerate(d["claims"]):
            real_ps = select_passages(c["claim"], d["paras"], k=8)
            null_ps = select_passages(c["claim"], detail[partner[i]]["paras"], k=8)
            arms = [("real", c["claim"], real_ps), ("null", c["claim"], null_ps)]
            pert = perturb_numbers(c["claim"])
            if pert:
                arms.append(("pert", pert, real_ps))
            for arm, claim, ps in arms:
                greqs.append((i, j, arm, ps))
                ptxt = "\n".join(f"[{k}] {p}" for k, p in enumerate(ps))
                convs.append([{"role": "user", "content": GRADED_VERIFY.format(claim=claim, passages=ptxt)}])
    print(f"[gf] graded reqs={len(greqs)}", flush=True)
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=320))
    gv = {}
    for (i, j, arm, ps), o in zip(greqs, outs):
        gv[(i, j, arm)] = parse_graded(o.outputs[0].text, ps)
    for arm in ("real", "null", "pert"):
        h = Counter(v["support"] for k, v in gv.items() if k[2] == arm)
        pf = sum(1 for k, v in gv.items() if k[2] == arm and not v["parsed"])
        print(f"[gf] graded {arm:5s} support-hist={dict(sorted(h.items()))} parse_fail={pf}", flush=True)

    # ---- batch F: full-text prior-art, per-ref ----
    convs, kept = [], []
    for i, j, kind, t in ftreqs:
        claim = detail[i]["claims"][j]["claim"]
        if kind == "self":
            title, paras = "(candidate paper)", detail[i]["paras"]
        else:
            title, paras = pool[t]["title"][:150], cand_paras.get(pool[t]["pid"])
        if not paras:
            continue
        ex = select_passages(claim, paras, k=3)
        kept.append((i, j, kind, t))
        convs.append([{"role": "user", "content": PRIOR_ART_VERIFY_FT.format(
            claim=claim, title=title, excerpts="\n".join(f"- {p}" for p in ex))}])
    print(f"[gf] FT calls={len(convs)} (dropped {len(ftreqs) - len(convs)} bodyless)", flush=True)
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=220))
    ft = {}
    for (i, j, kind, t), o in zip(kept, outs):
        ft.setdefault((i, j), {"real": [], "self": None, "foreign": None})
        res = parse_pa_single(o.outputs[0].text)
        if kind == "real":
            ft[(i, j)]["real"].append(res["verdict"])
        else:
            ft[(i, j)][kind] = res["verdict"]
    h = Counter(v for r in ft.values() for v in r["real"])
    print(f"[gf] FT real verdicts={dict(h)}", flush=True)

    # ---- aggregate ----
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    recs = []
    for i, d in enumerate(detail):
        nj = len(d["claims"])
        m = {"paper_id": d["paper_id"], "y": d["y"], "n_claims": nj}
        m.update(graded_metrics([gv[(i, j, "real")] for j in range(nj) if (i, j, "real") in gv], "g_"))
        m.update({("null_" + k): v for k, v in graded_metrics(
            [gv[(i, j, "null")] for j in range(nj) if (i, j, "null") in gv], "g_").items()})
        m.update({("pert_" + k): v for k, v in graded_metrics(
            [gv[(i, j, "pert")] for j in range(nj) if (i, j, "pert") in gv], "g_").items()})
        m.update(ft_pa_metrics([ft[(i, j)] for j in range(nj) if (i, j) in ft]))
        recs.append(m)
    M = pd.DataFrame(recs)
    M.to_csv(OUTD / "paper_pilot_graded_ft.csv", index=False)
    with open(OUTD / "paper_pilot_graded_ft_detail.jsonl", "w") as fh:
        for i, d in enumerate(detail):
            fh.write(json.dumps({
                "paper_id": d["paper_id"], "y": d["y"], "claims": d["claims"],
                "graded": {arm: [gv.get((i, j, arm)) for j in range(len(d["claims"]))]
                           for arm in ("real", "null", "pert")},
                "ft": [ft.get((i, j)) for j in range(len(d["claims"]))]}) + "\n")

    y = M["y"].values
    print("\n=== GRADED INSTRUMENT ===", flush=True)
    print(f"  g_mean_support  real={M['g_mean_support'].mean():.3f} "
          f"null={M['null_g_mean_support'].mean():.3f} pert={M['pert_g_mean_support'].mean():.3f}", flush=True)
    print(f"  g_frac_ge3      real={M['g_frac_ge3'].mean():.3f} "
          f"null={M['null_g_frac_ge3'].mean():.3f} pert={M['pert_g_frac_ge3'].mean():.3f}", flush=True)
    print("\n=== FT PRIOR-ART INSTRUMENT ===", flush=True)
    print(f"  ft_self_detect={M['ft_self_detect'].mean():.3f} (abstract-PA v2: .702) "
          f"ft_foreign_distinct={M['ft_foreign_distinct'].mean():.3f}", flush=True)
    print(f"  ft_anticipated_rate mean={M['ft_anticipated_rate'].mean():.3f} (abstract-PA: judge-only)", flush=True)
    print("\n=== AUCs (univariate) ===", flush=True)
    for c in M.columns:
        if c in ("paper_id", "y"):
            continue
        v = M[c].values.astype(float)
        mk = ~np.isnan(v)
        if mk.sum() > 60 and len(set(y[mk])) == 2 and np.nanstd(v) > 1e-9:
            print(f"  {c:26s} AUC={roc_auc_score(y[mk], v[mk]):.3f} (n={mk.sum()})", flush=True)
    print("GRADED_FT_DONE", flush=True)


if __name__ == "__main__":
    main()
