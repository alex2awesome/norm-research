#!/usr/bin/env python3
"""GEPA-style revival of DEGENERATE peer-review A-rubrics (label-free objective).

Input: the 104 degenerate rubrics from outputs/a_bank_audit/a_metric_audit_sk3.csv
(DEAD / NEAR / UNUSABLE_NA on the ICLR-2400 scoring). For each, Gemma-4 proposes K=3
re-articulations given DISTRIBUTION feedback only (mode share, NA rate — never labels):
keep the underlying construct, re-target it so it is applicable + discriminative on ML
conference abstracts. Variants are scored on a stable-hash dev slice (n=300) with the
ORIGINAL instrument (same SYS prompt / one-token 1.0|0.5|0.0|NA protocol as
score_va_gemma.py). Metrics with no alive variant get a round-2 proposal with round-1
feedback. Best alive variant per metric is rescored on the full 2400.

Objective = ALIVE-ness (NA<.5 and mode-share<.90 on dev) — variance/applicability only.
Labels are used ONLY in the final eval readout (A of alive-bank +/- revived), as in every
V/A row. Outputs: peer_review_revived.npz + gepa_revive_report.json.

Run (gemma4 env, 1 GPU):
  CUDA_VISIBLE_DEVICES=1 /lfs/skampere3/0/alexspan/envs/gemma4/bin/python \
      datasets/peer-review/gepa_revive_dead.py
"""
import csv, gzip, hashlib, json, random, re
from pathlib import Path
import numpy as np

BASE = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review")
AUDIT = "/lfs/skampere3/0/alexspan/norm-research/outputs/a_bank_audit/a_metric_audit_sk3.csv"
RUBRICS = Path("/lfs/skampere3/0/alexspan/data/peer_review/rubrics.jsonl")
GEMMA4 = "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb"
OUTDIR = Path("/lfs/skampere3/0/alexspan/norm-research/outputs/a_bank_audit")

SYS = ("You are an expert academic peer reviewer. You are given a paper's ABSTRACT and ONE "
       "quality criterion. Decide how strongly the abstract, on its own evidence, satisfies "
       "that criterion. Answer with EXACTLY ONE token:\n"
       "  1.0 = clearly satisfies the criterion\n"
       "  0.5 = partially / weakly / borderline\n"
       "  0.0 = fails / cuts against the criterion\n"
       "  NA = the abstract gives no evidence bearing on this criterion\n"
       "Judge the paper's quality, not whether it will be accepted. Output only the token.")

PROPOSE = """A quality criterion was scored on 2,400 machine-learning conference (ICLR) paper
ABSTRACTS with a 1.0/0.5/0.0/NA protocol, and came out DEGENERATE: {feedback}.

CRITERION NAME: {name}
DESCRIPTION: {desc}

Rewrite this criterion so that it (a) preserves the same underlying quality construct as
closely as the domain allows, and (b) is APPLICABLE to ML conference abstracts and likely to
DISCRIMINATE quality among them (real spread across 1.0/0.5/0.0, low NA). If the criterion is
domain-inapplicable (e.g. a clinical-trial reporting rule), map its SPIRIT to the nearest
ML-research analog. Never mention acceptance, reviews, or outcomes. Make this rewrite
substantively DIFFERENT from these earlier attempts (if any): {prior}

Return ONLY JSON: {{"name": "<concise criterion name>", "description": "<2-3 sentence description>"}}"""

def parse_tok(t):
    t = (t or "").strip().lower()
    if t.startswith("na") or "n/a" in t: return np.nan
    if "0.5" in t: return 0.5
    if t.startswith("1"): return 1.0
    if t.startswith("0"): return 0.0
    return np.nan

def metric_block(name, desc):
    return f"CRITERION: {name}\nDESCRIPTION: {desc}\n\nAnswer with one token:"

def alive_stats(scores):
    v = np.asarray(scores, float)
    na = float(np.mean(np.isnan(v)))
    nv = v[~np.isnan(v)]
    if len(nv) == 0: return na, 1.0, np.nan
    vals, cnt = np.unique(nv, return_counts=True)
    return na, float(cnt.max() / len(nv)), float(vals[cnt.argmax()])

def main():
    import pandas as pd
    from vllm import LLM, SamplingParams

    # -- degenerate targets --
    A = pd.read_csv(AUDIT)
    A = A[A.task.str.startswith("peer review") & A.cls.isin(["DEAD", "NEAR", "UNUSABLE_NA"])]
    targets = []
    for _, r in A.iterrows():
        idx = int(str(r.metric).split("|")[0])
        fb = (f"NA rate {r.na:.0%}, and of the scored cases {r.mode_share:.0%} received the "
              f"same value ({r['mode']})")
        targets.append(dict(idx=idx, feedback=fb))
    rub = [json.loads(l) for l in open(RUBRICS) if l.strip() and json.loads(l).get("name")]
    print(f"[revive] {len(targets)} degenerate targets of {len(rub)} rubrics", flush=True)

    # -- same 2400 rows as the V/A run --
    d = np.load(BASE / "peer_review_scores_iclr.npz", allow_pickle=True)
    ids = [str(i) for i in d["ids"]]; y = d["y"].astype(int)
    X_orig = d["X"].astype(float)
    texts = {}
    with gzip.open(BASE / "splits" / "train.csv.gz", "rt", errors="ignore") as fh:
        for row in csv.DictReader(fh):
            if row.get("id") in set(ids): texts[row["id"]] = row.get("text") or ""
    abstracts = [texts[i] for i in ids]
    hs = sorted(range(len(ids)), key=lambda i: hashlib.sha1(ids[i].encode()).hexdigest())
    dev = hs[:300]
    print(f"[revive] rows {len(abstracts)}, dev {len(dev)}", flush=True)

    llm = LLM(model=GEMMA4, gpu_memory_utilization=0.85, max_model_len=4096, enforce_eager=False)
    sp_score = SamplingParams(temperature=0.0, max_tokens=4)
    sp_gen = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=380)

    def chat(batch, sp):
        msgs = [[{"role": "system", "content": s}, {"role": "user", "content": u}] for s, u in batch]
        outs = llm.chat(msgs, sp)
        return [o.outputs[0].text for o in outs]

    def propose(round_targets, prior_map):
        reqs, keys = [], []
        for t in round_targets:
            m = rub[t["idx"]]
            prior = prior_map.get(t["idx"], "none")
            for k in range(3):
                reqs.append(("You rewrite evaluation rubrics. Output only JSON.",
                             PROPOSE.format(feedback=t["feedback"], name=m["name"],
                                            desc=m.get("description", ""), prior=prior)))
                keys.append((t["idx"], k))
        outs = chat(reqs, sp_gen)
        var = {}
        for (idx, k), o in zip(keys, outs):
            mt = re.search(r"\{.*\}", o, re.S)
            if not mt: continue
            try: j = json.loads(mt.group(0))
            except Exception: continue
            if j.get("name") and j.get("description"):
                var.setdefault(idx, []).append(dict(name=str(j["name"])[:120],
                                                    description=str(j["description"])[:700]))
        return var

    def score_variants(var, row_idx):
        reqs, keys = [], []
        for idx, vs in var.items():
            for vi, v in enumerate(vs):
                blk = metric_block(v["name"], v["description"])
                for ri in row_idx:
                    reqs.append((SYS, f"ABSTRACT:\n{abstracts[ri][:6000]}\n\n{blk}"))
                    keys.append((idx, vi, ri))
        print(f"[revive] scoring {len(reqs)} calls", flush=True)
        outs = chat(reqs, sp_score)
        sc = {}
        for (idx, vi, ri), o in zip(keys, outs):
            sc.setdefault((idx, vi), {})[ri] = parse_tok(o)
        return sc

    # ---- round 1 ----
    var1 = propose(targets, {})
    sc1 = score_variants(var1, dev)
    best, need_r2, prior_map = {}, [], {}
    for t in targets:
        idx = t["idx"]; cands = []
        for vi, v in enumerate(var1.get(idx, [])):
            s = [sc1.get((idx, vi), {}).get(ri, np.nan) for ri in dev]
            na, ms, mode = alive_stats(s)
            cands.append((ms, na, vi, v))
        alive = [c for c in cands if c[1] < .5 and c[0] < .90]
        if alive:
            best[idx] = min(alive, key=lambda c: (c[0], c[1]))[3] | {"round": 1}
        else:
            need_r2.append(t)
            prior_map[idx] = " | ".join(
                f"'{v['name']}' -> NA {na:.0%}, mode-share {ms:.0%}"
                for ms, na, vi, v in cands) or "none parsed"
    print(f"[revive] round1 alive {len(best)}/{len(targets)}; round2 for {len(need_r2)}", flush=True)

    # ---- round 2 (diversity pass with round-1 feedback) ----
    if need_r2:
        var2 = propose(need_r2, prior_map)
        sc2 = score_variants(var2, dev)
        for t in need_r2:
            idx = t["idx"]; cands = []
            for vi, v in enumerate(var2.get(idx, [])):
                s = [sc2.get((idx, vi), {}).get(ri, np.nan) for ri in dev]
                na, ms, mode = alive_stats(s)
                cands.append((ms, na, vi, v))
            alive = [c for c in cands if c[1] < .5 and c[0] < .90]
            if alive: best[idx] = min(alive, key=lambda c: (c[0], c[1]))[3] | {"round": 2}
    print(f"[revive] total revived on dev: {len(best)}/{len(targets)}", flush=True)

    # ---- full 2400 rescore of revived variants ----
    full_var = {idx: [v] for idx, v in best.items()}
    scF = score_variants(full_var, list(range(len(abstracts))))
    R = np.full((len(abstracts), len(best)), np.nan)
    names = []
    for j, (idx, v) in enumerate(sorted(best.items())):
        names.append(f"{idx:03d}R|{v['name'][:60]}")
        for ri in range(len(abstracts)):
            R[ri, j] = scF.get((idx, 0), {}).get(ri, np.nan)
    # final alive check at full scale
    keep = []
    for j in range(R.shape[1]):
        na, ms, mode = alive_stats(R[:, j])
        keep.append(na < .5 and ms < .90)
    keep = np.array(keep)
    print(f"[revive] alive at full scale: {int(keep.sum())}/{R.shape[1]}", flush=True)
    np.savez(OUTDIR / "peer_review_revived.npz", R=R, names=np.array(names),
             keep=keep, ids=np.array(ids),
             variants=json.dumps({str(k): v for k, v in best.items()}))

    # ---- eval readout (labels enter HERE only) ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    def auc_of(M):
        pipe = make_pipeline(SimpleImputer(strategy="constant", fill_value=0.5), StandardScaler(),
                             LogisticRegression(max_iter=3000, class_weight="balanced"))
        return float(cross_val_score(pipe, M, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                     scoring="roc_auc").mean())
    na_o = np.isnan(X_orig).mean(0)
    ms_o = np.array([alive_stats(X_orig[:, j])[1] for j in range(X_orig.shape[1])])
    alive_o = (na_o < .95) & (ms_o < .95)
    Rk = R[:, keep]
    rep = dict(n_targets=len(targets), revived_dev=len(best), revived_full=int(keep.sum()),
               A_alive=auc_of(X_orig[:, alive_o]),
               A_alive_plus_revived=auc_of(np.hstack([X_orig[:, alive_o], Rk])) if keep.sum() else None,
               A_revived_only=auc_of(Rk) if keep.sum() else None,
               revived_uni=[])
    from sklearn.metrics import roc_auc_score
    kn = [n for n, k in zip(names, keep) if k]
    for j in range(Rk.shape[1]):
        col = Rk[:, j]; mk = ~np.isnan(col)
        if mk.sum() > 100 and len(set(y[mk])) == 2:
            rep["revived_uni"].append((kn[j], round(roc_auc_score(y[mk], col[mk]), 3), int(mk.sum())))
    rep["revived_uni"].sort(key=lambda t: -abs(t[1] - .5))
    json.dump(rep, open(OUTDIR / "gepa_revive_report.json", "w"), indent=2)
    print(json.dumps({k: rep[k] for k in ["n_targets", "revived_dev", "revived_full",
                                          "A_alive", "A_alive_plus_revived", "A_revived_only"]}, indent=2))
    print("top revived uni:", rep["revived_uni"][:8], flush=True)
    print("GEPA_REVIVE_DONE", flush=True)

if __name__ == "__main__":
    main()
