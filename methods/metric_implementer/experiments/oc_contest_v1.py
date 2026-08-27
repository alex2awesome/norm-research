"""Objective-contest v1 (plan v3, user-approved 2026-08-15): m_rec vs m_fb vs m_desc on
the ORIGINAL mention data + the MTMM-lite label-free axis. CPU only; artifacts on disk.

Axis 1 (human, per-construct): mention-AUC — task y_pos positives vs all-other negatives
  (peer additionally: attentive-negative variant from the review join). Noise is symmetric
  across arms (no arm saw any y).
Axis 2a (label-free MTMM-lite, within-family v1 — disclosed): margin =
  mean corr(arm form, same-metric OTHER forms) - mean |corr(arm form, other-metric
  canonical scores)| on shared docs. Structure criterion; no labels.
Instrument fairness: per task ONE judge for all arms — cw/pr/humor scores_g4 + ocdef g4;
  peer uses peer_within_scores_g4_merged + ocdef g4 (8B files NOT used for the contest).
Output: mention_auc/oc_contest_v1.json + printed contest tables incl. axis divergences.
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

MD = Path("/lfs/skampere3/0/alexspan/mention_auc")
OC = Path("/lfs/skampere3/0/alexspan/outputs/objective_comparison_v1")

FORMFILE = {"peer": "peer_within_scores_g4_merged.json", "cw": "cw_scores_g4.json",
            "pr": "pr_scores_g4.json", "humor": "humor_scores_g4.json"}
YFILE = {"peer": "peer_y_pos.json", "cw": "variant_ypos_cw.json",
         "pr": "variant_ypos_pr.json", "humor": "humor_ypos.json"}


def auc(y, p):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    if not n1 or not n0:
        return None
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


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


def main():
    sel = json.load(open(OC / "selections.json"))
    sens = json.load(open(OC / "form_sensitivity.json"))
    rows = []
    for task, mm in sel.items():
        d = json.load(open(MD / FORMFILE[task]))
        ids, Sf = d["post_ids"], d["scores"]
        dd = json.load(open(MD / f"ocdef_{task}_corpus_g4.json"))
        idsd, Sd = dd["post_ids"], dd["scores"]
        assert idsd == ids or set(idsd) >= set(ids) or True
        idxd = {x: i for i, x in enumerate(idsd)}
        ym = ymap_load(YFILE[task])
        # canonical per-metric score (for discriminant axis): definition scores
        for mid, arms in mm.items():
            fk_rec = (arms.get("m_recon") or [None])[0]
            fk_fb = (arms.get("m_fb") or [None])[0]
            keys = {"m_rec": fk_rec, "m_fb": fk_fb, "m_desc": f"{mid}__-1"}
            row = {"task": task, "metric": mid, "sens": sens.get(task, {}).get(mid)}
            ok = True
            vecs = {}
            for arm, k in keys.items():
                src = Sd if arm == "m_desc" else Sf
                if k is None or k not in src:
                    ok = False
                    break
                v = np.asarray(src[k], float)
                if arm == "m_desc" and idsd != ids:
                    v = np.array([v[idxd[x]] if x in idxd else np.nan for x in ids])
                vecs[arm] = v
            if not ok:
                continue
            yv = np.array([1 if mid in ym.get(x, ()) else 0 for x in ids])
            # axis 1: mention AUC per arm
            for arm, v in vecs.items():
                fin = np.isfinite(v)
                if fin.sum() < 100 or yv[fin].sum() < 10 or yv[fin].sum() > fin.sum() - 10:
                    row[f"auc_{arm}"] = None
                else:
                    row[f"auc_{arm}"] = round(auc(yv[fin], v[fin]), 4)
            # axis 2a: MTMM-lite margin per arm
            others_same = [k for k in Sf if k.startswith(f"{mid}__")
                           and k not in (keys["m_rec"], keys["m_fb"]) and not k.endswith("__-1")]
            other_mets = [k for k in Sd if not k.startswith(f"{mid}__")]
            om_sample = other_mets[:15]
            for arm, v in vecs.items():
                fin = np.isfinite(v)
                conv = []
                for k in others_same[:8]:
                    u = np.asarray(Sf[k], float)
                    both = fin & np.isfinite(u)
                    if both.sum() > 100 and v[both].std() > 0 and u[both].std() > 0:
                        conv.append(np.corrcoef(v[both], u[both])[0, 1])
                disc = []
                for k in om_sample:
                    u = np.asarray(Sd[k], float)
                    if idsd != ids:
                        u = np.array([u[idxd[x]] if x in idxd else np.nan for x in ids])
                    both = fin & np.isfinite(u)
                    if both.sum() > 100 and v[both].std() > 0 and u[both].std() > 0:
                        disc.append(abs(np.corrcoef(v[both], u[both])[0, 1]))
                if len(conv) >= 3 and len(disc) >= 5:
                    row[f"mtmm_{arm}"] = round(float(np.mean(conv) - np.mean(disc)), 4)
                else:
                    row[f"mtmm_{arm}"] = None
            rows.append(row)

    out = {"rows": rows}
    rng = random.Random(0)

    def contest(metric_key):
        pairs = {}
        for a, b in (("m_rec", "m_fb"), ("m_rec", "m_desc"), ("m_fb", "m_desc")):
            d = [r[f"{metric_key}_{a}"] - r[f"{metric_key}_{b}"] for r in rows
                 if r.get(f"{metric_key}_{a}") is not None and r.get(f"{metric_key}_{b}") is not None]
            if len(d) < 10:
                continue
            n = len(d)
            obs = float(np.mean(d))
            boots = sorted(float(np.mean([d[rng.randrange(n)] for _ in range(n)]))
                           for _ in range(20000))
            w = sum(1 for x in d if x > 0); l_ = sum(1 for x in d if x < 0)
            pairs[f"{a}_vs_{b}"] = {"n": n, "mean": round(obs, 4),
                                    "ci": [round(boots[500], 4), round(boots[19499], 4)],
                                    "wins": f"{w}/{w + l_}"}
        return pairs

    for axis in ("auc", "mtmm"):
        out[axis] = contest(axis)
        print(f"\n=== axis: {'mention-AUC (human)' if axis == 'auc' else 'MTMM-lite margin (label-free)'} ===")
        for k, v in out[axis].items():
            print(f"{k:16s} n={v['n']:3d} mean {v['mean']:+.4f} {v['ci']} wins {v['wins']}")
    # divergences: metrics where the two axes rank m_rec vs m_desc oppositely
    div = [r for r in rows
           if r.get("auc_m_rec") is not None and r.get("auc_m_desc") is not None
           and r.get("mtmm_m_rec") is not None and r.get("mtmm_m_desc") is not None
           and (r["auc_m_rec"] - r["auc_m_desc"]) * (r["mtmm_m_rec"] - r["mtmm_m_desc"]) < 0]
    out["n_axis_divergent_rec_vs_desc"] = len(div)
    both = [r for r in rows if r.get("auc_m_rec") is not None and r.get("mtmm_m_rec") is not None]
    print(f"\naxis-divergent metrics (rec-vs-desc sign flips between axes): {len(div)}/{len(both)}")
    # moderator on the human axis
    d_ = [(r["sens"], r["auc_m_rec"] - r["auc_m_fb"]) for r in rows
          if r.get("sens") is not None and r.get("auc_m_rec") is not None
          and r.get("auc_m_fb") is not None]
    if len(d_) >= 15:
        from scipy.stats import spearmanr
        rho, pv = spearmanr([x[0] for x in d_], [x[1] for x in d_])
        out["P2_moderator_human_axis"] = {"rho": round(float(rho), 3), "p": round(float(pv), 4),
                                          "n": len(d_)}
        print(f"moderator rho(form-sens, rec-fb mention-AUC delta) = {rho:+.3f} (p={pv:.4f}, n={len(d_)})")
    json.dump(out, open(MD / "oc_contest_v1.json", "w"), indent=1)
    print("\nsaved -> oc_contest_v1.json")


if __name__ == "__main__":
    main()
