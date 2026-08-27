"""Tier-3 wave-2 convergence analysis (2026-08-15). Order per protocol:
  1. CERTIFICATION GATE: wave-2 sealed anchors (void wave if <80%); v1-vs-v2 agreement on
     the 1,080 re-arbitrated items (arbiter reliability under protocol change).
  2. PURIFIED LABELS y*: per (task, metric, doc) — v2 verdict preferred over v1 where both
     exist; two variants: ALL (applies bool) and CONFIDENT (score<=2 -> 0, >=8 -> 1, else drop).
  3. TIER-3 TABLE (task #33): AUC of the canonical judge vs y* per task + label confusion
     (mention-y vs y*) with stratum counts.
  4. TIER-A ARMS (task #34, prereg P1-P3): per metric, AUC of each arm's selected form
     (m_recon / m_fb from selections.json; m_desc = ocdef corpus) vs y* on arbitrated docs;
     paired bootstrap on mean delta + sign test; moderator = form_sensitivity.
Peer note: arm-eval uses peer_within_scores form shards — n per metric may be thin
(arbitrated docs ∩ shard); reported per cell, never hidden.
Runs on sk3. Artifacts: mention_auc/t3_final_readout.json.
"""
import glob
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

MD = Path("/lfs/skampere3/0/alexspan/mention_auc")
OC = Path("/lfs/skampere3/0/alexspan/outputs/objective_comparison_v1")


def auc(y, p):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    if not n1 or not n0:
        return None
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_verdicts(d):
    V = {}
    for f in glob.glob(str(MD / d / "*.json")):
        for v in json.load(open(f)):
            V[v["item_id"]] = v
    return V


def main():
    it1 = {i["item_id"]: i for i in json.load(open(MD / "t3_items_v2.json"))}
    it2 = {i["item_id"]: i for i in json.load(open(MD / "t3_items_wave2.json"))}
    V1 = load_verdicts("verdicts")
    V2 = load_verdicts("verdicts_w2")
    out = {}

    # ---- 1. certification ----
    key = {(k.get("task", "peer"), k["metric"], k["doc"]): k["truth"]
           for k in json.load(open(MD / "t3_anchor_key_SEALED.json"))}
    for label, items, V in (("wave1", it1, V1), ("wave2", it2, V2)):
        ok = tot = 0
        for iid, v in V.items():
            it = items.get(iid)
            if not it or it["stratum"] != "anchor":
                continue
            t = key.get((it.get("task", "peer"), it["metric"], it["doc"]))
            if t is None:
                continue
            tot += 1
            ok += int(bool(v["applies"]) == bool(t))
        out[f"anchors_{label}"] = f"{ok}/{tot}"
        print(f"CERT {label}: anchors {ok}/{tot}")
        if tot and ok / tot < 0.8:
            print(f"!! {label} FAILS certification — its verdicts are VOID")
            out[f"{label}_void"] = True
    # v1-v2 agreement on redo items
    agree = tot = 0
    for iid2, it in it2.items():
        if it.get("wave") != "v2redo" or not it.get("item_id_v1"):
            continue
        v2 = V2.get(iid2); v1 = V1.get(it["item_id_v1"])
        if v1 and v2:
            tot += 1
            agree += int(bool(v1["applies"]) == bool(v2["applies"]))
    out["v1v2_agreement"] = f"{agree}/{tot}" + (f" ({agree/tot:.2f})" if tot else "")
    print(f"v1-vs-v2 agreement on re-arbitrated items: {out['v1v2_agreement']}")

    # ---- 2. purified labels ----
    ystar = {}                                     # (task, metric, doc) -> dict
    for items, V, wave in ((it1, V1, 1), (it2, V2, 2)):
        for iid, v in V.items():
            it = items.get(iid)
            if not it or it["stratum"] == "anchor":
                continue
            k = (it.get("task", "peer"), it["metric"], it["doc"])
            if k in ystar and ystar[k]["wave"] > wave:
                continue
            sc = v.get("score")
            conf = 1 if (sc is not None and sc >= 8) else (0 if (sc is not None and sc <= 2)
                                                          else None)
            ystar[k] = {"applies": 1 if v["applies"] else 0, "conf": conf, "wave": wave,
                        "mention": 1 if it["stratum"].startswith("y1") else 0}
    out["n_purified"] = len(ystar)
    print(f"purified labels: {len(ystar)} (task,metric,doc) triples")

    # ---- score loaders ----
    def corpus_forms(task):
        f = {"peer": "peer_within_scores.json"}.get(task, f"{task}_scores_g4.json")
        d = json.load(open(MD / f))
        return d["post_ids"], d["scores"]

    def canonical(task):
        if task == "peer":
            d = json.load(open(MD / "peer_p_scores.json"))
            return d["post_ids"], d["scores"]
        man = json.load(open(MD / f"{task}_forms_manifest.json"))
        d = json.load(open(MD / f"{task}_scores_g4.json"))
        best = {}
        for e in man:
            m = e["metric_id"]
            if m not in best or e.get("mi_form", 0) > best[m][1]:
                best[m] = (e["form_idx"], e.get("mi_form", 0))
        return d["post_ids"], {m: d["scores"][f"{m}__{fi}"]
                               for m, (fi, _) in best.items() if f"{m}__{fi}" in d["scores"]}

    # ---- 3. tier-3 table ----
    print("\n=== TIER-3: canonical judge vs purified y* ===")
    print(f"{'task':7s} {'variant':9s} {'n':>5s} {'AUC':>6s} {'n_conf':>6s} {'AUC_conf':>8s}")
    t3 = {}
    for task in ("peer", "cw", "pr", "humor", "crx"):
        try:
            ids, S = canonical(task) if task != "crx" else (
                json.load(open(MD / "crx_p_scores.json"))["post_ids"],
                json.load(open(MD / "crx_p_scores.json"))["scores"])
        except FileNotFoundError:
            continue
        idx = {d: i for i, d in enumerate(ids)}
        rows_all, rows_conf = [], []
        for (t, m, doc), y in ystar.items():
            if t != task or m not in S or doc not in idx:
                continue
            sc = float(np.asarray(S[m], float)[idx[doc]])
            if not np.isfinite(sc):
                continue
            rows_all.append((sc, y["applies"]))
            if y["conf"] is not None:
                rows_conf.append((sc, y["conf"]))
        res = {}
        for tag, rows in (("all", rows_all), ("conf", rows_conf)):
            if len(rows) < 30:
                res[tag] = None
                continue
            p = np.array([r[0] for r in rows]); yv = np.array([r[1] for r in rows])
            res[tag] = {"n": len(rows), "auc": round(auc(yv, p), 4) if auc(yv, p) else None}
        t3[task] = res
        a = res.get("all") or {}
        c = res.get("conf") or {}
        print(f"{task:7s} {'':9s} {a.get('n', 0):5d} {str(a.get('auc')):>6s} "
              f"{c.get('n', 0):6d} {str(c.get('auc')):>8s}")
    out["tier3"] = t3

    # ---- 4. tier-A arms ----
    print("\n=== TIER-A: arm AUCs vs purified y* (confident labels) ===")
    sel = json.load(open(OC / "selections.json"))
    sens = json.load(open(OC / "form_sensitivity.json"))
    arm_rows = []
    for task, mm in sel.items():
        ids_f, Sf = corpus_forms(task)
        idxf = {d: i for i, d in enumerate(ids_f)}
        dd = json.load(open(MD / f"ocdef_{task}_corpus_g4.json"))
        ids_d, Sd = dd["post_ids"], dd["scores"]
        idxd = {d: i for i, d in enumerate(ids_d)}
        for mid, arms in mm.items():
            docs = [(doc, y) for (t, m, doc), y in ystar.items()
                    if t == task and m == mid and y["conf"] is not None]
            if len(docs) < 12:
                continue
            row = {"task": task, "metric": mid, "n": len(docs),
                   "sens": sens.get(task, {}).get(mid)}
            okrow = True
            for arm, fkey in (("m_recon", (arms.get("m_recon") or [None])[0]),
                              ("m_fb", (arms.get("m_fb") or [None])[0]),
                              ("m_desc", f"{mid}__-1")):
                src = Sd if arm == "m_desc" else Sf
                idx = idxd if arm == "m_desc" else idxf
                if fkey is None or fkey not in src:
                    okrow = False
                    break
                vec = np.asarray(src[fkey], float)
                pts = [(float(vec[idx[doc]]), y["conf"]) for doc, y in docs
                       if doc in idx and np.isfinite(vec[idx[doc]])]
                ys_ = [y for _, y in pts]
                if len(pts) < 12 or min(ys_.count(0), ys_.count(1)) < 4:
                    okrow = False
                    break
                p = np.array([x[0] for x in pts]); yv = np.array([x[1] for x in pts])
                row[arm] = round(auc(yv, p), 4)
                row[f"n_{arm}"] = len(pts)
            if okrow:
                arm_rows.append(row)
    out["tierA_rows"] = arm_rows
    if arm_rows:
        rec = np.array([r["m_recon"] for r in arm_rows])
        fb = np.array([r["m_fb"] for r in arm_rows])
        de = np.array([r["m_desc"] for r in arm_rows])
        rng = random.Random(0)

        def pboot(d, B=20000):
            obs = float(np.mean(d)); n = len(d)
            cnt = 0; lo_hi = []
            for _ in range(B):
                s = [d[rng.randrange(n)] for _ in range(n)]
                m = float(np.mean(s)); lo_hi.append(m)
            lo_hi.sort()
            return obs, lo_hi[int(.025 * B)], lo_hi[int(.975 * B)]

        for name, d in (("recon_minus_fb", rec - fb), ("recon_minus_desc", rec - de),
                        ("fb_minus_desc", fb - de)):
            obs, lo, hi = pboot(list(d))
            w = int((d > 0).sum()); l_ = int((d < 0).sum())
            out[f"P_{name}"] = {"mean": round(obs, 4), "ci": [round(lo, 4), round(hi, 4)],
                                "wins": f"{w}/{w + l_}"}
            print(f"{name}: mean {obs:+.4f} [{lo:+.4f},{hi:+.4f}] wins {w}/{w + l_} "
                  f"(n={len(d)})")
        ss = np.array([r["sens"] if r["sens"] is not None else np.nan for r in arm_rows])
        okm = np.isfinite(ss)
        if okm.sum() >= 10:
            from scipy.stats import spearmanr
            rho, pv = spearmanr(ss[okm], (rec - fb)[okm])
            out["P2_moderator"] = {"spearman": round(float(rho), 3), "p": round(float(pv), 4),
                                   "n": int(okm.sum())}
            print(f"P2 moderator rho(form-sensitivity, recon-fb delta) = {rho:+.3f} "
                  f"(p={pv:.4f}, n={okm.sum()})")
    json.dump(out, open(MD / "t3_final_readout.json", "w"), indent=1)
    print("\nsaved -> t3_final_readout.json")


if __name__ == "__main__":
    main()
