#!/usr/bin/env python3
"""ROUND-0 CONCEPT CENSUS of the incoming 198-rubric N&C A-bank.

FREEZE requirement: "concept census of the incoming bank at round 0" +
"Deduplicate and REGISTER-MATCH the incoming bank at round 0" (missing-mass
robustification note, PART 4 recommendation 3).  The peer bank went 154 delivered
-> 95 distinct names -> 79 surviving columns -> 54 effective concepts; the same
ladder is computed here.

Instrument stack (cheapest decisive test first):
  L0 delivered rubrics                       198
  L1 distinct NAMES                          (exact, case/whitespace-normalised)
  L2 columns surviving the frozen degeneracy screen, fit on FIT+MINE only
  L3 VALUE-identical / near-identical columns collapsed
       (|Pearson r| >= .98 on FIT+MINE, single-linkage) -> effective measurement rank
  L4 SEMANTIC concepts: embedding-shortlisted candidate pairs adjudicated blind.
       Embedding is used ONLY to shortlist (never to decide), and only WITHIN one
       register (bank rubric vs bank rubric), which is exactly the case the
       missing-mass note licenses; identity decisions come from a blind pairwise
       judge pass (m3-style) written to `census_pairs_blind.json`.

Also reports the REGISTER of the bank vs the corpus (the silent governor of every
similarity measurement downstream) as a descriptive note.

CPU only; the embedding step reuses the Layer-2 bge cache mechanism.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import nc_closure_lib as L

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RUBRICS = REPO / "datasets" / "notice-and-comment" / "v4" / "nc_rubrics.jsonl"
CACHE = HERE / "census_embed_cache.npz"

R_CORR = 0.98
SHORTLIST_TAU = 0.80   # shortlist only; identity decided by the blind judge pass
TOP_K = 4


def norm_name(s: str) -> str:
    s = s.lower().replace("‑", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def embed(texts):
    """bge-large CLS embeddings via plain transformers (Layer-2 recipe)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    cached = {}
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        cached = {str(k): v for k, v in zip(z["keys"], z["vecs"])}
    keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
    need = [t for t, k in zip(texts, keys) if k not in cached]
    if need:
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        mod = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5").to(dev).eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(need), 32):
                b = tok(need[i:i + 32], padding=True, truncation=True, max_length=512,
                        return_tensors="pt").to(dev)
                h = mod(**b).last_hidden_state[:, 0]
                h = torch.nn.functional.normalize(h, dim=-1)
                out.append(h.cpu().numpy())
        out = np.vstack(out)
        for t, v in zip(need, out):
            cached[hashlib.sha1(t.encode()).hexdigest()] = v
        np.savez_compressed(CACHE, keys=np.array(list(cached), dtype=object),
                            vecs=np.array(list(cached.values())))
    return np.array([cached[k] for k in keys])


def main():
    pop = L.load_population()
    summary, split, dsplit, mining, monitor_full = L.load_splits()
    fm = split == "fit_mine"

    rub = [json.loads(l) for l in open(RUBRICS) if l.strip()]
    names = pop["a_names"]
    assert len(rub) == len(names) == 198
    by_name = {r["name"]: r for r in rub}
    descs = [by_name[n]["description"] if n in by_name else "" for n in names]

    A = pop["A"]
    res = {"cell": "nc_responded", "L0_delivered": len(names)}

    # ---- L1 distinct names ---------------------------------------------------
    nn = [norm_name(n) for n in names]
    dupname = defaultdict(list)
    for i, k in enumerate(nn):
        dupname[k].append(i)
    res["L1_distinct_names"] = len(dupname)
    res["L1_duplicate_name_groups"] = [
        {"name": names[v[0]], "idx": v} for v in dupname.values() if len(v) > 1
    ]

    # ---- L2 surviving columns (screen fit on FIT+MINE only) ------------------
    keep, meds = L.clean_fit(A[fm])
    res["L2_surviving_columns"] = int(len(keep))
    res["L2_dropped"] = [names[j] for j in range(A.shape[1]) if j not in set(keep.tolist())]

    Ak = L.clean_apply(A, keep, meds)
    Afm = Ak[fm]
    kept_names = [names[j] for j in keep]
    kept_descs = [descs[j] for j in keep]

    # ---- L3 value-level near-duplicates -> effective measurement rank ---------
    C = np.corrcoef(Afm.T)
    C = np.nan_to_num(C)
    m = len(keep)
    parent = list(range(m))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    pairs_val = []
    for i in range(m):
        for j in range(i + 1, m):
            if abs(C[i, j]) >= R_CORR:
                union(i, j)
                pairs_val.append({"a": kept_names[i], "b": kept_names[j], "r": float(C[i, j])})
    clus = defaultdict(list)
    for i in range(m):
        clus[find(i)].append(i)
    res["L3_value_clusters"] = len(clus)
    res["L3_r_threshold"] = R_CORR
    res["L3_merged_pairs"] = pairs_val
    res["L3_multi_member_clusters"] = [
        [kept_names[i] for i in v] for v in clus.values() if len(v) > 1
    ]
    # descriptive: how much of the correlation mass sits high
    off = C[np.triu_indices(m, 1)]
    res["L3_corr_distribution"] = {
        "mean_abs": float(np.abs(off).mean()),
        "p90": float(np.quantile(np.abs(off), 0.90)),
        "p99": float(np.quantile(np.abs(off), 0.99)),
        "max": float(np.abs(off).max()),
        "frac_ge_0.90": float((np.abs(off) >= 0.90).mean()),
        "frac_ge_0.80": float((np.abs(off) >= 0.80).mean()),
    }

    # ---- alone-AUC per surviving column (FIT+MINE only; MONITOR never read) ---
    from sklearn.metrics import roc_auc_score

    yfm = pop["y"][fm]
    alone = {}
    for i in range(m):
        try:
            alone[kept_names[i]] = float(roc_auc_score(yfm, Afm[:, i]))
        except ValueError:
            alone[kept_names[i]] = float("nan")
    res["alone_auc_fitmine"] = alone
    vals = np.array([v for v in alone.values() if not np.isnan(v)])
    res["alone_auc_summary"] = {
        "max": float(vals.max()), "min": float(vals.min()),
        "median": float(np.median(vals)),
        "median_abs_dev_from_half": float(np.median(np.abs(vals - 0.5))),
        "n_ge_0.55": int((vals >= 0.55).sum()), "n_le_0.45": int((vals <= 0.45).sum()),
    }

    # ---- L4 semantic shortlist (embedding = SHORTLIST ONLY) ------------------
    texts = [f"{n}. {d}" for n, d in zip(kept_names, kept_descs)]
    E = embed(texts)
    S = E @ E.T
    np.fill_diagonal(S, -1)
    cand = []
    for i in range(m):
        order = np.argsort(-S[i])[:TOP_K]
        for j in order:
            if j > i and S[i, j] >= SHORTLIST_TAU:
                cand.append((i, int(j), float(S[i, j])))
    cand = sorted({(min(a, b), max(a, b), s) for a, b, s in cand}, key=lambda t: -t[2])
    res["L4_shortlist_tau"] = SHORTLIST_TAU
    res["L4_n_candidate_pairs"] = len(cand)
    res["L4_cos_distribution"] = {
        "max_offdiag": float(S.max()),
        "p99": float(np.quantile(S[np.triu_indices(m, 1)], 0.99)),
        "median": float(np.median(S[np.triu_indices(m, 1)])),
    }

    blind = []
    for n, (i, j, s) in enumerate(cand):
        # X/Y order randomised by stable hash so the judge sees no provenance order
        flip = int(hashlib.sha256(f"census|{i}|{j}".encode()).hexdigest(), 16) % 2
        a, b = (i, j) if not flip else (j, i)
        blind.append({
            "pair_id": f"CP{n+1:03d}",
            "X": {"name": kept_names[a], "description": kept_descs[a]},
            "Y": {"name": kept_names[b], "description": kept_descs[b]},
            "_key": {"i": i, "j": j, "cos": s, "flipped": bool(flip)},
        })
    (HERE / "census_pairs_blind.json").write_text(json.dumps(
        {"n_pairs": len(blind), "pairs": [{k: v for k, v in p.items() if k != "_key"} for p in blind]},
        indent=1))
    (HERE / "census_pairs_key.json").write_text(json.dumps(
        {"pairs": [{"pair_id": p["pair_id"], **p["_key"]} for p in blind]}, indent=1))

    # ---- register note -------------------------------------------------------
    res["register_note"] = {
        "bank_register": "federal regulatory-analysis language (RIA/E.O. 12866/CFR framing), "
                         "authored as agency-facing comment-quality rubrics",
        "corpus_register": "public comments on proposed federal rules (individual, org, "
                           "law-firm and form-letter submissions)",
        "why_it_matters": "cross-register embedding thresholds do not transfer "
                          "(missing-mass note s2.3); all identity decisions in this campaign "
                          "use blind full-recall adjudication, never a cosine threshold.",
    }

    (HERE / "concept_census.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items()
                      if k in ("L0_delivered", "L1_distinct_names", "L2_surviving_columns",
                               "L3_value_clusters", "L3_corr_distribution",
                               "L4_n_candidate_pairs", "L4_cos_distribution",
                               "alone_auc_summary")}, indent=2))
    print(f"blind pairs -> census_pairs_blind.json ({len(blind)})")


if __name__ == "__main__":
    main()
