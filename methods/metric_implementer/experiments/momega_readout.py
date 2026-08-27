"""Final m_omega-proper readout (prereg 29fe4b35d). Runs on sk3 once mo_*_hat8b.json land.
Selections (label-free arms use PROBES only): m_omega = argmax rank-agreement(hat scores,
M_i_8B); m_llm = argmax critic rank-agreement of candidate probe scores; m_desc = C0;
skyline = argmax mention-AUC on OBJECTIVE-half corpus. Eval: mention-AUC on EVAL-half."""
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

MD = Path("/lfs/skampere3/0/alexspan/mention_auc")
OC = Path("/lfs/skampere3/0/alexspan/outputs/objective_comparison_v1")
BANKS = Path("/lfs/skampere3/0/alexspan/outputs/ecert_slice_v1")
YFILE = {"humor": "humor_ypos.json", "cw": "variant_ypos_cw.json", "peer": "peer_y_pos.json"}


def obj_half(task, doc):
    return int(hashlib.md5(f"ocsplit:{task}:{doc}".encode()).hexdigest(), 16) % 2 == 0


def auc(y, p):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    if not n1 or not n0:
        return None
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rankagree(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return -2.0
    return float(np.corrcoef(ra, rb)[0, 1])


def ymap_load(yf):
    raw = json.load(open(MD / yf))
    k = next(iter(raw))
    out = defaultdict(set)
    if re.fullmatch(r"a\d+", k):
        for m, docs in raw.items():
            for d in docs:
                out[d].add(m)
    else:
        for d, ms in raw.items():
            out[d] = set(ms)
    return out


cands = json.load(open(BANKS / "momega_candidates.json"))
crit = defaultdict(dict)
for line in open(OC / "critic_all_results.jsonl"):
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r.get("score") is not None:
        crit[r["aspect_id"]][r["datapoint_id"]] = float(r["score"])

rows = []
for task in ("humor", "peer", "cw"):
    enc = json.load(open(MD / f"mo_{task}_probes8b.json"))
    hat = json.load(open(MD / f"mo_{task}_hat8b.json"))
    cor = json.load(open(MD / f"mo_{task}_corpus8b.json"))
    mi_d = json.load(open(MD / f"ocl_llama8b_{task}_probes.json"))
    pids = enc["post_ids"]
    assert hat["post_ids"] == pids and mi_d["post_ids"] == pids
    cids = cor["post_ids"]
    ym = ymap_load(YFILE[task])
    for r0 in [c for c in cands if c["task"] == task]:
        mid = r0["metric"]
        mk = f"{mid}__-1"
        if mk not in mi_d["scores"]:
            continue
        Mi = np.asarray(mi_d["scores"][mk], float)
        cr = crit.get(mid, {})
        cr_al = np.array([cr.get(x, np.nan) for x in pids])
        rec_scores, llm_scores = {}, {}
        for j in range(6):
            hk = f"{mid}__{800 + j}"
            ek = f"{mid}__{700 + j}"
            if hk in hat["scores"]:
                v = np.asarray(hat["scores"][hk], float)
                fin = np.isfinite(v) & np.isfinite(Mi)
                if fin.sum() >= 100 and v[fin].std() > 0 and Mi[fin].std() > 0:
                    rec_scores[j] = rankagree(v[fin], Mi[fin])
            if ek in enc["scores"]:
                v = np.asarray(enc["scores"][ek], float)
                fin = np.isfinite(v) & np.isfinite(cr_al)
                if fin.sum() >= 100 and v[fin].std() > 0 and cr_al[fin].std() > 0:
                    llm_scores[j] = rankagree(v[fin], cr_al[fin])
        if not rec_scores or not llm_scores:
            continue
        yv = np.array([1 if mid in ym.get(x, ()) else 0 for x in cids])
        oh = np.array([obj_half(task, x) for x in cids])
        cand_auc_obj, cand_auc_eval = {}, {}
        for j in range(6):
            ck = f"{mid}__{700 + j}"
            if ck not in cor["scores"]:
                continue
            v = np.asarray(cor["scores"][ck], float)
            for tag, mask in (("obj", oh), ("eval", ~oh)):
                fin = np.isfinite(v) & mask
                if fin.sum() < 60 or yv[fin].sum() < 5 or yv[fin].sum() > fin.sum() - 5:
                    continue
                a = auc(yv[fin], v[fin])
                (cand_auc_obj if tag == "obj" else cand_auc_eval)[j] = a
        if 0 not in cand_auc_eval or len(cand_auc_eval) < 4:
            continue
        j_omega = max(rec_scores, key=rec_scores.get)
        j_llm = max(llm_scores, key=llm_scores.get)
        j_sky = max(cand_auc_obj, key=cand_auc_obj.get) if cand_auc_obj else None
        row = {"task": task, "metric": mid, "name": r0["name"][:40],
               "j_omega": j_omega, "j_llm": j_llm, "j_sky": j_sky,
               "auc_desc": round(cand_auc_eval[0], 4),
               "auc_omega": round(cand_auc_eval.get(j_omega, np.nan), 4),
               "auc_llm": round(cand_auc_eval.get(j_llm, np.nan), 4),
               "auc_sky": round(cand_auc_eval.get(j_sky, np.nan), 4) if j_sky is not None else None,
               "auc_oracle_eval": round(max(cand_auc_eval.values()), 4)}
        if not any(np.isnan(x) for x in (row["auc_omega"], row["auc_llm"])):
            rows.append(row)

print(f"metrics evaluable: {len(rows)}")
for r in rows:
    print(f"{r['task']:6s} {r['metric']:6s} desc {r['auc_desc']:.3f} | omega[{r['j_omega']}] "
          f"{r['auc_omega']:.3f} | llm[{r['j_llm']}] {r['auc_llm']:.3f} | "
          f"sky {r['auc_sky']} | oracle {r['auc_oracle_eval']:.3f}  {r['name']}")
rng = random.Random(0)


def pboot(d):
    n = len(d)
    obs = float(np.mean(d))
    boots = sorted(float(np.mean([d[rng.randrange(n)] for _ in range(n)]))
                   for _ in range(20000))
    return obs, boots[500], boots[19499]


if len(rows) >= 8:
    for name, d in (("Q1 omega - desc", [r["auc_omega"] - r["auc_desc"] for r in rows]),
                    ("Q2 omega - llm", [r["auc_omega"] - r["auc_llm"] for r in rows]),
                    ("Q3 skyline - desc", [r["auc_sky"] - r["auc_desc"] for r in rows
                                           if r["auc_sky"] is not None]),
                    ("oracle - desc", [r["auc_oracle_eval"] - r["auc_desc"] for r in rows])):
        if len(d) < 8:
            continue
        o, lo, hi = pboot(d)
        w = sum(1 for x in d if x > 0); l_ = sum(1 for x in d if x < 0)
        print(f"{name:18s} mean {o:+.4f} [{lo:+.4f},{hi:+.4f}] +/-: {w}/{l_}")
json.dump(rows, open(MD / "momega_readout_v1.json", "w"), indent=1)
print("saved -> momega_readout_v1.json")
