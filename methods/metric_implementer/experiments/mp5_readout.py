"""EXP-MP-1c readout (prereg 74d99a18c). Runs on sk3."""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MD = Path("/lfs/skampere3/0/alexspan/mention_auc")
enc = json.load(open(MD / "mp5_peer_probes8b.json"))
hats = [json.load(open(MD / f"mp5_peer_hat8b_k{K}.json")) for K in range(3)]
cor = json.load(open(MD / "mp5_peer_corpus8b.json"))
labels = json.load(open(MD / "mp5_labels.json"))
pids = enc["post_ids"]
for h in hats:
    assert h["post_ids"] == pids
cids = cor["post_ids"]
cidx = {d: i for i, d in enumerate(cids)}
crit = defaultdict(dict)
for f in ("mp2_critic_results.jsonl", "mp5_critic_results.jsonl"):
    for line in open(MD / f):
        r = json.loads(line)
        if r.get("score") is not None:
            crit[r["aspect_id"]][r["datapoint_id"]] = float(r["score"])


def rankagree(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    if not n1 or not n0:
        return None
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


rows = []
for m, lab in sorted(labels.items(), key=lambda kv: -len(kv[1]["pos"])):
    Mi = np.asarray(enc["scores"][f"{m}__700"], float)
    cr = np.array([crit[m].get(d, np.nan) for d in pids])
    rec_s, llm_s = {}, {}
    for j in range(12):
        vals = []
        for h in hats:
            hv = np.asarray(h["scores"].get(f"{m}__{800 + j}", []), float)
            if len(hv):
                fin = np.isfinite(hv) & np.isfinite(Mi)
                if fin.sum() >= 100 and hv[fin].std() > 0:
                    ra = rankagree(hv[fin], Mi[fin])
                    if ra is not None:
                        vals.append(ra)
        if len(vals) >= 2:
            rec_s[j] = float(np.mean(vals))
        ev = np.asarray(enc["scores"][f"{m}__{700 + j}"], float)
        fin = np.isfinite(ev) & np.isfinite(cr)
        if fin.sum() >= 100 and ev[fin].std() > 0 and cr[fin].std() > 0:
            ra = rankagree(ev[fin], cr[fin])
            if ra is not None:
                llm_s[j] = ra
    if not rec_s or not llm_s:
        print(f"{m}: selection failed")
        continue
    j_om, j_ll = max(rec_s, key=rec_s.get), max(llm_s, key=llm_s.get)
    arms = {"desc": 700, "omega": 700 + j_om, "llm": 700 + j_ll}
    row = {"metric": m, "j_omega": j_om, "j_llm": j_ll, "n_pos": len(lab["pos"])}
    for tier, key in (("corr", "pos"), ("single", "single")):
        P = [d for d in lab[key] if d in cidx]
        Ng = [d for d in lab["neg"] if d in cidx]
        docs = P + Ng
        y = np.array([1] * len(P) + [0] * len(Ng))
        sc = {}
        keep = np.ones(len(docs), bool)
        for a, fi in arms.items():
            v = np.asarray(cor["scores"][f"{m}__{fi}"], float)
            s = np.array([v[cidx[d]] for d in docs])
            sc[a] = s
            keep &= np.isfinite(s)
        for a in arms:
            v = auc(y[keep], sc[a][keep])
            row[f"{tier}_{a}"] = round(v, 4) if v is not None else None
    rows.append(row)

print(f"{'metric':6s} npos | corr: desc  omega  llm  | j_om j_ll")
for r in rows:
    print(f"{r['metric']:6s} {r['n_pos']:4d} | {r['corr_desc']:.3f}  {r['corr_omega']:.3f}"
          f"  {r['corr_llm']:.3f} | C{r['j_omega']}  C{r['j_llm']}")
print("j_omega:", dict(Counter(r["j_omega"] for r in rows)),
      "j_llm:", dict(Counter(r["j_llm"] for r in rows)),
      "| divergent selections:", sum(1 for r in rows if r["j_omega"] != r["j_llm"]), "/", len(rows))
rng = random.Random(0)


def pb(d):
    n = len(d)
    boots = sorted(float(np.mean([d[rng.randrange(n)] for _ in range(n)]))
                   for _ in range(20000))
    return float(np.mean(d)), boots[500], boots[19499], sum(1 for b in boots if b <= 0) / 20000


d_c = [r["corr_omega"] - r["corr_llm"] for r in rows]
d_s = [r["single_omega"] - r["single_llm"] for r in rows]
o, lo, hi, p1 = pb(d_c)
print(f"H1c omega-llm (corroborated) mean {o:+.4f} [{lo:+.4f},{hi:+.4f}] "
      f"+/-: {sum(1 for x in d_c if x>0)}/{sum(1 for x in d_c if x<0)} p1~{p1:.4f}")
o2, lo2, hi2, _ = pb([r["corr_omega"] - r["corr_desc"] for r in rows])
print(f"omega-desc (descriptive)     mean {o2:+.4f} [{lo2:+.4f},{hi2:+.4f}]")
dd = [a - b for a, b in zip(d_c, d_s)]
o3, lo3, hi3, p3 = pb(dd)
print(f"dose-response (corr - single gap) {o3:+.4f} [{lo3:+.4f},{hi3:+.4f}] p1~{p3:.3f}")
for tag, cond in (("omega>=.65", lambda r: r["corr_omega"] >= .65),
                  ("llm>=.65 SYMMETRIC", lambda r: r["corr_llm"] >= .65),
                  ("unconditional", lambda r: True)):
    sub = [r for r in rows if cond(r)]
    if not sub:
        print(f"stratum {tag}: EMPTY"); continue
    d = [r["corr_omega"] - r["corr_llm"] for r in sub]
    print(f"stratum {tag}: n={len(sub)} mean omega "
          f"{np.mean([r['corr_omega'] for r in sub]):.3f} llm "
          f"{np.mean([r['corr_llm'] for r in sub]):.3f} gap {np.mean(d):+.4f}")
json.dump(rows, open(MD / "mp5_readout_v1.json", "w"), indent=1)
print("saved -> mp5_readout_v1.json")
