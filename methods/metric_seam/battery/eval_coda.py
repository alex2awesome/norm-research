"""CODA probe evaluation (priors note §3.1, pre-registered 2026-07-05).

Anchor validation -> per-feature Spearman vs y_code/y_fm/ratio -> LOTO rank-ridge
(features F1-F8 -> y_code, leave-one-task-out) vs baselines (zero-shot codability,
rel1-alone) -> CODIF mediation join (phrasing -> program composition -> outcome).

-> outputs/metric_seam_pilot/battery/coda_eval.json
"""
import json, math, pathlib, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/metric_seam_pilot"
CODA = BASE / "battery/coda"
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import certificates  # noqa: E402
spearman = certificates.spearman

FEATS = [f"F{i}" for i in range(1, 9)]
TASKS = ["press_releases", "creative_writing", "math", "humor", "legal_title_vii"]


def load_annotations():
    ann = {}
    for p in sorted(CODA.glob("annotations_*.json")):
        ann.update(json.load(open(p)))
    zs = {}
    for p in sorted(CODA.glob("zeroshot_*.json")):
        zs.update(json.load(open(p)))
    return ann, zs


def validate_anchors(manifest, ann):
    """Hard anchors must land in their keyed bands, per batch (batch = id//26)."""
    report, fails = {}, 0
    for cid, m in manifest.items():
        if m["kind"] != "anchor" or not m.get("hard"):
            continue
        got = ann.get(cid)
        if got is None:
            report[cid] = {"anchor": m["anchor_name"], "status": "MISSING"}
            fails += 1
            continue
        bad = {f: got.get(f) for f, allowed in m["checks"].items()
               if got.get(f) not in allowed}
        if bad:
            fails += 1
        report[cid] = {"anchor": m["anchor_name"],
                       "status": "PASS" if not bad else f"FAIL {bad}",
                       "batch": int(cid[1:]) // 26}
    return report, fails


def rel_for(task):
    from judge_rep_task import gemma_judge
    from battery_common import load_ctx
    ctx = load_ctx(task)
    _, rel = gemma_judge(task, ctx["outdir"])
    return rel


def rank(v):
    order = np.argsort(v)
    r = np.empty(len(v))
    r[order] = np.arange(len(v), dtype=float)
    # midranks for ties
    vv = np.asarray(v)
    for u in np.unique(vv):
        m = vv == u
        r[m] = r[m].mean()
    return r


def loto_ridge(panel, xcols, ycol, alpha=1.0):
    """Rank-ridge trained on 4 tasks, predict the 5th; Spearman within held-out."""
    per_task, pooled_pred, pooled_y = {}, [], []
    for hold in TASKS:
        tr = [r for r in panel if r["task"] != hold and r[ycol] is not None]
        te = [r for r in panel if r["task"] == hold and r[ycol] is not None]
        if len(te) < 8 or len(tr) < 20:
            continue
        Xtr = np.array([[r[c] for c in xcols] for r in tr], float)
        ytr = rank([r[ycol] for r in tr])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr = (Xtr - mu) / sd
        w = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(len(xcols)), Xtr.T @ ytr)
        Xte = (np.array([[r[c] for c in xcols] for r in te], float) - mu) / sd
        pred = Xte @ w
        ys = [r[ycol] for r in te]
        per_task[hold] = {"n": len(te), "spearman": round(spearman(list(pred), ys), 3)}
        # normalize within-task ranks to [0,1]: raw 0..n-1 ranks pooled across
        # unequal folds share a fold-size signal (perm-null pooled rho ~ +.1-.2)
        pooled_pred += [float(x) / max(len(pred) - 1, 1) for x in rank(pred)]
        pooled_y += [float(x) / max(len(ys) - 1, 1) for x in rank(ys)]
    pooled = spearman(pooled_pred, pooled_y) if pooled_pred else float("nan")
    return per_task, round(pooled, 3), pooled_pred, pooled_y


def boot_ci(pred, y, B=2000, seed=7):
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    pred, y = np.asarray(pred), np.asarray(y)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        stats.append(spearman(list(pred[idx]), list(y[idx])))
    return round(float(np.percentile(stats, 2.5)), 3), \
        round(float(np.percentile(stats, 97.5)), 3)


def main():
    manifest = json.load(open(CODA / "coda_manifest.json"))
    ann, zs = load_annotations()
    areport, afails = validate_anchors(manifest, ann)
    n_hard = sum(1 for v in areport.values())

    # outcome panel
    cam = json.load(open(BASE / "cam_profile.json"))
    fb = json.load(open(BASE / "battery/fleet_boundness.json"))["per_aspect"]
    rels = {t: rel_for(t) for t in TASKS}
    panel = []
    for cid, m in manifest.items():
        if m["kind"] != "real" or cid not in ann:
            continue
        t, aid = m["task"], m["aid"]
        row_cam = next((r for r in cam[t]["per_criterion"] if r["aspect"] == aid), None)
        if row_cam is None:
            continue
        f = fb.get(t, {}).get(aid, {})
        rl, rq = f.get("ratio_llama"), f.get("ratio_qwen")
        row = {"task": t, "aid": aid, "y_code": row_cam["r_base"],
               "y_hyb": row_cam["r_hyb"],
               "y_fm": f.get("fm"), "certified": f.get("certified_gate"),
               "ratio_mean": (rl + rq) / 2 if rl is not None and rq is not None else None,
               "rel1": rels[t].get(aid), "zeroshot": zs.get(cid)}
        row.update({k: ann[cid].get(k) for k in FEATS})
        if any(row[k] is None for k in FEATS):
            continue
        panel.append(row)

    # 1. per-feature Spearmans (pooled ranks within task to kill task-level confound,
    #    plus raw pooled for reference)
    per_feature = {}
    for fcol in FEATS + ["zeroshot", "rel1"]:
        entry = {}
        for ycol in ["y_code", "y_fm", "ratio_mean"]:
            rows = [r for r in panel if r[ycol] is not None and r[fcol] is not None]
            if len(rows) < 20:
                continue
            entry[ycol] = round(spearman([r[fcol] for r in rows],
                                         [r[ycol] for r in rows]), 3)
            # within-task pooled (rank within each task first)
            wp, wy = [], []
            for t in TASKS:
                tr = [r for r in rows if r["task"] == t]
                if len(tr) < 8:
                    continue
                wp += list(rank([r[fcol] for r in tr]))
                wy += list(rank([r[ycol] for r in tr]))
            entry[ycol + "_withintask"] = round(spearman(wp, wy), 3) if wp else None
        per_feature[fcol] = entry

    # 2. LOTO rank-ridge on F1-F8 -> y_code, vs baselines
    per_task, pooled, pp, py = loto_ridge(panel, FEATS, "y_code")
    lo, hi = boot_ci(pp, py)
    zs_per_task = {}
    zs_pool_p, zs_pool_y = [], []
    for t in TASKS:
        tr = [r for r in panel if r["task"] == t and r["zeroshot"] is not None]
        if len(tr) >= 8:
            zs_per_task[t] = {"n": len(tr),
                              "spearman": round(spearman([r["zeroshot"] for r in tr],
                                                         [r["y_code"] for r in tr]), 3)}
            zs_pool_p += list(rank([r["zeroshot"] for r in tr]))
            zs_pool_y += list(rank([r["y_code"] for r in tr]))
    zs_pooled = round(spearman(zs_pool_p, zs_pool_y), 3)
    zlo, zhi = boot_ci(zs_pool_p, zs_pool_y)
    rel_rows = [r for r in panel if r["rel1"] is not None]
    rel_alone = round(spearman([r["rel1"] for r in rel_rows],
                               [r["y_code"] for r in rel_rows]), 3)

    # 2b. LOTO for y_fm (the corollary target)
    per_task_fm, pooled_fm, _, _ = loto_ridge(
        [r for r in panel if r["y_fm"] is not None], FEATS, "y_fm")

    def within(rows, xcol, ycol):
        wp, wy = [], []
        for t in TASKS:
            tr = [r for r in rows if r["task"] == t and r.get(xcol) is not None
                  and r.get(ycol) is not None]
            if len(tr) < 8:
                continue
            wp += list(rank([r[xcol] for r in tr]))
            wy += list(rank([r[ycol] for r in tr]))
        return round(spearman(wp, wy), 3) if len(wp) >= 20 else None

    # 3. CODIF mediation join: phrasing features vs program composition
    codif = {}
    for line in open(BASE / "battery/codif/codif_merged.jsonl"):
        d = json.loads(line)
        codif[(d["task"], d["aid"])] = d
    med = {}
    C8SH = {"NONE": 0, "LOW": 0, "MED": 1, "HIGH": 2}
    joined = []
    for r in panel:
        c = codif.get((r["task"], r["aid"]))
        if not c:
            continue
        tags = c.get("tags", {})
        jr = dict(r)
        for tg in ["C2", "C3", "C4", "C5", "C6", "C7"]:
            jr[tg] = 1 if tags.get(tg, {}).get("present") else 0
        share = c.get("c8_share") or c.get("C8_share")
        if share is None and "C8" in tags:
            share = tags["C8"].get("share")
        jr["c8_share"] = C8SH.get(share) if isinstance(share, str) else None
        joined.append(jr)
    for fcol in FEATS:
        entry = {}
        for tg in ["C3", "C4", "C5", "C6", "c8_share"]:
            rows = [r for r in joined if r.get(tg) is not None]
            if len(rows) >= 30:
                entry[tg] = round(spearman([r[fcol] for r in rows],
                                           [r[tg] for r in rows]), 3)
                entry[tg + "_withintask"] = within(rows, fcol, tg)
        med[fcol] = entry

    # 4. Empirical separators: which battery coordinates distinguish coded from
    #    prompted criteria WITHIN task (the ex-post complement to CODA's ex-ante)
    conv = {(c["task"], c["aid"]): c["conv"]
            for c in json.load(open(BASE / "battery/artic_eval.json"))["criteria"]}
    e7 = json.load(open(BASE / "battery/e7_provenance_grid.json"))["rows"]
    dmax = {}
    for r in e7:
        k = (r["task"], r["aid"])
        if r.get("distill") is not None:
            dmax[k] = max(dmax.get(k, -9), r["distill"])
    for r in joined:
        k = (r["task"], r["aid"])
        r["conv"] = conv.get(k)
        r["distill_max"] = dmax.get(k)
        r["cert01"] = 1 if r.get("certified") else 0
    separators = {}
    for xcol in ["y_fm", "ratio_mean", "rel1", "conv", "distill_max", "zeroshot",
                 "c8_share", "C3", "C4", "C5", "C6"]:
        separators[xcol] = {"y_code": within(joined, xcol, "y_code"),
                            "y_hyb": within(joined, xcol, "y_hyb"),
                            "certified": within(joined, xcol, "cert01"),
                            "n": sum(1 for r in joined if r.get(xcol) is not None)}

    out = {"n_panel": len(panel),
           "anchor_report": areport,
           "anchor_hard_fails": f"{afails}/{n_hard}",
           "per_feature_spearman": per_feature,
           "loto_ridge_ycode": {"per_task": per_task, "pooled": pooled,
                                "pooled_ci95": [lo, hi]},
           "loto_ridge_yfm": {"per_task": per_task_fm, "pooled": pooled_fm},
           "baseline_zeroshot": {"per_task": zs_per_task, "pooled": zs_pooled,
                                 "pooled_ci95": [zlo, zhi]},
           "baseline_rel1_alone": rel_alone,
           "codif_mediation": med,
           "empirical_separators": separators,
           "panel": panel}
    json.dump(out, open(BASE / "battery/coda_eval.json", "w"), indent=1)

    print(f"panel n={len(panel)}  anchors: {afails}/{n_hard} hard fails")
    print("\nper-feature Spearman (pooled / within-task):")
    print(f"{'feat':9s} {'y_code':>16s} {'y_fm':>16s} {'ratio':>16s}")
    for fcol in FEATS + ["zeroshot", "rel1"]:
        e = per_feature[fcol]
        def fmt(y):
            a, b = e.get(y), e.get(y + "_withintask")
            return f"{a if a is not None else '—':>7} /{b if b is not None else '—':>7}"
        print(f"{fcol:9s} {fmt('y_code')} {fmt('y_fm')} {fmt('ratio_mean')}")
    print(f"\nLOTO ridge -> y_code pooled {pooled} CI {[lo, hi]}  per-task "
          f"{ {k: v['spearman'] for k, v in per_task.items()} }")
    print(f"zero-shot baseline pooled {zs_pooled} CI {[zlo, zhi]}  per-task "
          f"{ {k: v['spearman'] for k, v in zs_per_task.items()} }")
    print(f"rel1-alone vs y_code: {rel_alone}")
    print(f"LOTO ridge -> y_fm pooled {pooled_fm}  per-task "
          f"{ {k: v['spearman'] for k, v in per_task_fm.items()} }")
    print("\ncodif mediation (pooled / within-task):")
    for fcol in FEATS:
        cells = "  ".join(f"{tg}:{med[fcol].get(tg)}/{med[fcol].get(tg + '_withintask')}"
                          for tg in ["C3", "C5", "C6", "c8_share"])
        print(f" {fcol}: {cells}")
    print("\nempirical separators (within-task Spearman):")
    print(f"{'coord':11s} {'y_code':>7s} {'y_hyb':>7s} {'cert':>7s} {'n':>4s}")
    for xcol, e in separators.items():
        print(f"{xcol:11s} {str(e['y_code']):>7s} {str(e['y_hyb']):>7s} "
              f"{str(e['certified']):>7s} {e['n']:>4d}")
    print(f"-> {BASE / 'battery/coda_eval.json'}")


if __name__ == "__main__":
    main()
