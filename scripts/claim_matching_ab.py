#!/usr/bin/env python3
"""Joint A/B forced-choice claim-matching instrument (Codex round-2, finding 3/8 top lever).

Instead of rating each (element, span) independently on 0-4 and reconstructing the ranking, show
BOTH spans in ONE prompt per metric and force a choice: which passage better satisfies this
criterion for this element? Removes separate-call calibration noise (pair diff variance 2sigma^2
-> shared context), kills 5-point tie mass, and aligns the instrument with the paired endpoint.

Reconstruction-only: the model NEVER sees the label; passage order is randomized by a
label-independent hash of (uid, metric_id); which slot held gold is stored for the readout only.
Truncation widened per Codex #5: element[:1200], span[:2000].

  # pilot (200 claims, BOTH orders -> order-consistency audit):
  python scripts/claim_matching_ab.py --model <path> --tag ab12b_pilot --pairs-file <core_pairs> \
      --uids-limit 200 --both-orders
  # full core (single randomized order):
  python scripts/claim_matching_ab.py --model <path> --tag ab12b --pairs-file <core_pairs>
  # readout (CPU):
  python scripts/claim_matching_ab.py --readout outputs/claim_matching/ab_scores_<tag>.jsonl
"""
import argparse, json, re, hashlib, os, collections

BASE = "/lfs/skampere3/0/alexspan/norm-research"
BANK = f"{BASE}/datasets/claim-matching/claim_matching_bank.jsonl"
TESTBED = f"{BASE}/datasets/claim-matching/testbed/pair_testbed_v2.jsonl"
OUTDIR = f"{BASE}/outputs/claim_matching"

SYS = ("You compare a patent CLAIM ELEMENT against two candidate prior-art passages using ONE "
       "specific matching CRITERION. Exactly one passage is from the reference a patent examiner "
       "actually cited against this claim. Judge which passage better satisfies the criterion in "
       "substance — surface word overlap is not a match.")

_CHOICE = re.compile(r'"?choice"?\s*[:=]\s*"?([AB])"?', re.I)
_CONF = re.compile(r'"?confidence"?\s*[:=]\s*([0-3])')


def horder(uid, mid):
    """label-independent deterministic order: True -> gold in slot A."""
    return int(hashlib.md5(f"ab::{uid}::{mid}".encode()).hexdigest(), 16) % 2 == 0


def build_jobs(pairs_file, uids_limit, both_orders):
    byu = collections.defaultdict(dict)
    for ln in open(pairs_file):
        r = json.loads(ln)
        byu[r["uid"]][r["y"]] = r
    uids = sorted((u for u, d in byu.items() if 1 in d and 0 in d),
                  key=lambda u: hashlib.md5(f"abpilot::{u}".encode()).hexdigest())
    if uids_limit:
        uids = uids[:uids_limit]
    bank = [json.loads(l) for l in open(BANK)]
    jobs = []
    for u in uids:
        g, f = byu[u][1], byu[u][0]
        for m in bank:
            orders = [horder(u, m["metric_id"])]
            if both_orders:
                orders = [True, False]
            for gold_first in orders:
                a, b = (g["span"], f["span"]) if gold_first else (f["span"], g["span"])
                crit = f"{m['name']}: {m['description']}"
                guid = (m.get("guidance") or "")[:400]
                user = (f"CRITERION:\n{crit}\n{guid}\n\nCLAIM ELEMENT:\n{g['element'][:1200]}\n\n"
                        f"PASSAGE A:\n{a[:2000]}\n\nPASSAGE B:\n{b[:2000]}\n\n"
                        'Which passage better satisfies the criterion for this claim element? '
                        'Reply ONE JSON only: {"choice": "A" or "B", "confidence": 0-3}.')
                jobs.append({"uid": u, "metric_id": m["metric_id"], "domain": m.get("domain"),
                             "gold_slot": "A" if gold_first else "B", "prompt": user})
    return jobs


def run(a):
    from vllm import LLM, SamplingParams
    jobs = build_jobs(a.pairs_file, a.uids_limit, a.both_orders)
    print(f"[ab:{a.tag}] {len(jobs)} forced-choice judgments "
          f"({'both orders' if a.both_orders else 'single order'})", flush=True)
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem_util,
              max_model_len=4096, trust_remote_code=True)
    msgs = [[{"role": "system", "content": SYS}, {"role": "user", "content": j["prompt"]}]
            for j in jobs]
    sp = SamplingParams(temperature=0.0, max_tokens=40)
    outs = llm.chat(msgs, sp)
    rows, retry_idx = [], []
    for j, o in zip(jobs, outs):
        txt = o.outputs[0].text if o.outputs else ""
        mc, mf = _CHOICE.search(txt or ""), _CONF.search(txt or "")
        rows.append({**{k: j[k] for k in ("uid", "metric_id", "domain", "gold_slot")},
                     "choice": mc.group(1).upper() if mc else None,
                     "conf": int(mf.group(1)) if mf else None})
    n_unparsed = sum(1 for r in rows if r["choice"] is None)
    if n_unparsed:
        retry_idx = [i for i, r in enumerate(rows) if r["choice"] is None]
        sp2 = SamplingParams(temperature=0.5, max_tokens=40, seed=7)
        outs2 = llm.chat([msgs[i] for i in retry_idx], sp2)
        for i, o in zip(retry_idx, outs2):
            txt = o.outputs[0].text if o.outputs else ""
            mc, mf = _CHOICE.search(txt or ""), _CONF.search(txt or "")
            if mc:
                rows[i]["choice"] = mc.group(1).upper()
                rows[i]["conf"] = int(mf.group(1)) if mf else None
    fp = f"{OUTDIR}/ab_scores_{a.tag}.jsonl"
    with open(fp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    cov = 1 - sum(1 for r in rows if r["choice"] is None) / max(1, len(rows))
    print(f"[ab:{a.tag}] coverage {cov:.3f} -> {fp}", flush=True)
    if cov < 0.95:
        print(f"AB_COVERAGE_LOW {cov:.3f}", flush=True)
    print("AB_SCORED", flush=True)


def _wacc(v):
    import numpy as np
    return float(np.mean(np.where(v > 0, 1.0, np.where(v == 0, 0.5, 0.0))))


def readout(fp):
    """Codex round-2 readout: honest naming (conf-weighted signed vote, NOT majority), sign-only
    ablation, flip-zeroing of order-inconsistent dual judgments, nested shrinkage-trimmed vote
    (app-balanced reliabilities shrunk w=max(0,2a-1), inner app-CV over top-k and transform),
    and a serialized artifact (ab_readout_<tag>.json)."""
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    rows = [json.loads(l) for l in open(fp)]
    u2app = {}
    for ln in open(TESTBED):
        r = json.loads(ln)
        u2app[r["uid"]] = str(r.get("app_id") or r["uid"])
    byum = collections.defaultdict(list)
    for r in rows:
        byum[(r["uid"], r["metric_id"])].append(r)
    dual = [v for v in byum.values() if len(v) == 2]
    audit = {}
    if dual:
        cons = sum(1 for v in dual
                   if v[0]["choice"] and v[1]["choice"]
                   and (v[0]["choice"] == v[0]["gold_slot"]) == (v[1]["choice"] == v[1]["gold_slot"]))
        posA = sum(1 for v in dual for x in v if x["choice"] == "A") / max(1, 2 * len(dual))
        audit = {"order_consistency": cons / len(dual), "p_choose_A": posA, "n_dual": len(dual)}
        print(f"[audit] order-consistency {cons/len(dual):.3f} (n={len(dual)} dual-scored); "
              f"P(choose A) {posA:.3f} (position-bias check, want ~.5)", flush=True)

    # per (uid, metric): signed score toward gold; unparsed -> 0; dual-order DISAGREEMENT -> 0
    # (Codex round-2 #3: exact criterion-level abstention, not confidence-difference residue)
    sgn = {}
    for (u, m), v in byum.items():
        dirs = [(x["choice"] == x["gold_slot"]) if x["choice"] else None for x in v]
        if len(v) == 2 and None not in dirs and dirs[0] != dirs[1]:
            sgn[(u, m)] = 0.0
            continue
        vals = []
        for x in v:
            if x["choice"] is None:
                vals.append(0.0); continue
            s = 1.0 if x["choice"] == x["gold_slot"] else -1.0
            vals.append(s * (0.5 + (x["conf"] if x["conf"] is not None else 1)))
        sgn[(u, m)] = float(np.mean(vals))
    mids = sorted({m for _, m in sgn})
    uids = sorted({u for u, _ in sgn})
    X = np.array([[sgn.get((u, m), 0.0) for m in mids] for u in uids])
    apps = np.array([u2app.get(u, u) for u in uids])
    folds = np.array([int(hashlib.md5(f"cv::{a}".encode()).hexdigest(), 16) % 5 for a in apps])

    perm = sorted(((m, _wacc(X[:, j])) for j, m in enumerate(mids)), key=lambda t: -t[1])
    print("[top metrics by A/B within-claim — GLOBAL rank, optimistically selected]", flush=True)
    for m, w in perm[:8]:
        print(f"  {m} within={w:.3f}", flush=True)

    res = {"n_claims": len(uids), "audit": audit,
           "per_metric": {m: w for m, w in perm}}
    res["conf_weighted_vote"] = _wacc(np.sign(X.sum(axis=1)))
    res["sign_only_majority"] = _wacc(np.sign(np.sign(X).sum(axis=1)))
    print(f"[A/B conf-weighted signed vote] within-claim acc={res['conf_weighted_vote']:.3f} "
          f"(n={len(uids)})", flush=True)
    print(f"[A/B sign-only majority]        within-claim acc={res['sign_only_majority']:.3f}",
          flush=True)

    # nested shrinkage-trimmed vote (Codex round-2 action #1)
    KS = [1, 3, 5, 10, 20, len(mids)]
    TRANSFORMS = {"sign": np.sign, "clip1.5": lambda Z: np.clip(Z, -1.5, 1.5), "raw": lambda Z: Z}
    def balanced_w(tr_idx):
        # app-balanced per-criterion accuracy -> shrunk weight max(0, 2a-1)
        byapp = collections.defaultdict(list)
        for i in tr_idx:
            byapp[apps[i]].append(i)
        accs = []
        for app_rows in byapp.values():
            sub = X[app_rows]
            accs.append([np.mean(np.where(sub[:, j] > 0, 1.0,
                                          np.where(sub[:, j] == 0, 0.5, 0.0)))
                         for j in range(len(mids))])
        a_bal = np.mean(np.array(accs), axis=0)
        return np.maximum(0.0, 2 * a_bal - 1)
    correct_st = np.zeros(len(uids))
    fold_sel = {}
    for f in range(5):
        te = np.where(folds == f)[0]; tr = np.where(folds != f)[0]
        if len(te) == 0 or len(tr) < 20:
            continue
        w = balanced_w(tr)
        order = np.argsort(-w)
        # inner app-grouped 4-fold selection of (k, transform)
        infold = np.array([int(hashlib.md5(f"icv::{apps[i]}".encode()).hexdigest(), 16) % 4
                           for i in tr])
        best, best_acc = (KS[-1], "raw"), -1.0
        for k in KS:
            cols = order[:k]
            for tname, tf in TRANSFORMS.items():
                accs = []
                for g in range(4):
                    ite = tr[infold == g]
                    if len(ite) == 0:
                        continue
                    itr = tr[infold != g]
                    wi = balanced_w(itr)
                    ci = np.argsort(-wi)[:k]
                    v = (tf(X[ite][:, ci]) * wi[ci]).sum(axis=1)
                    accs.append(_wacc(np.sign(v)))
                if accs and np.mean(accs) > best_acc:
                    best_acc, best = np.mean(accs), (k, tname)
        k, tname = best
        cols = order[:k]
        v = (TRANSFORMS[tname](X[te][:, cols]) * w[cols]).sum(axis=1)
        correct_st[te] = np.where(v > 0, 1.0, np.where(v == 0, 0.5, 0.0))
        fold_sel[f] = {"k": int(k), "transform": tname, "inner_acc": round(best_acc, 3)}
    res["shrink_trim_vote"] = float(correct_st.mean())
    res["shrink_trim_fold_selection"] = fold_sel
    print(f"[A/B shrinkage-trimmed vote (nested app-CV)] within-claim acc={res['shrink_trim_vote']:.3f}  "
          f"fold picks={fold_sel}", flush=True)

    # symmetrized logistic (reference arm)
    correct = np.zeros(len(uids))
    for f in range(5):
        te = folds == f; tr = ~te
        if te.sum() == 0 or tr.sum() < 10:
            continue
        Xtr = np.vstack([X[tr], -X[tr]]); ytr = np.r_[np.ones(tr.sum()), np.zeros(tr.sum())]
        keep = Xtr.std(axis=0) > 0
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(Xtr[:, keep], ytr)
        p = clf.predict_proba(X[te][:, keep])[:, 1]
        correct[te] = np.where(p > 0.5, 1.0, np.where(p == 0.5, 0.5, 0.0))
    res["logistic"] = float(correct.mean())
    print(f"[A/B combined (symmetrized logistic, app-folds)] within-claim acc={res['logistic']:.3f}",
          flush=True)

    out = fp.replace("ab_scores_", "ab_readout_").replace(".jsonl", ".json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"AB_READOUT_DONE -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--tag")
    ap.add_argument("--pairs-file", dest="pairs_file")
    ap.add_argument("--uids-limit", type=int, default=0)
    ap.add_argument("--both-orders", action="store_true")
    ap.add_argument("--gpu-mem-util", type=float, default=0.35)
    ap.add_argument("--readout", default=None)
    a = ap.parse_args()
    if a.readout:
        readout(a.readout)
    else:
        assert a.model and a.tag and a.pairs_file, "--model/--tag/--pairs-file required"
        run(a)


if __name__ == "__main__":
    main()
