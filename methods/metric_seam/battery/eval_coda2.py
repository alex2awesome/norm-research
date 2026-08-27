"""CODA-2: predictability on the EXPANDED panel (2026-07-08).

Merges the CODA-1 fleet panel (141 criteria, battery/coda_eval.json) with the 3 new
tasks (peer_review, legal_ss_disability, humor_units; coda2 annotations + cam_profile +
new_task_boundness). 8 tasks total. Re-runs LOTO rank-ridge for TWO targets to answer:
does phrasing predict the CONTINUOUS seam position (y_fm = borrowed-judgment marginal)
better than the BINARY code floor (y_code)?  Plus a budget-depth validation on the 12
ladder criteria.

-> outputs/metric_seam_pilot/battery/coda2_eval.json
"""
import json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import certificates  # noqa: E402
spearman = certificates.spearman
BASE = pathlib.Path(__file__).resolve().parents[3] / "outputs/metric_seam_pilot"

FEATS = [f"F{i}" for i in range(1, 9)]
TASKS = ["press_releases", "creative_writing", "math", "humor", "legal_title_vii",
         "peer_review", "legal_ss_disability", "humor_units"]


def rank(v):
    v = np.asarray(v, float)
    r = np.empty(len(v))
    r[np.argsort(v)] = np.arange(len(v))
    for u in np.unique(v):
        m = v == u
        r[m] = r[m].mean()
    return r


def nrank(v):
    # Raw within-fold ranks span 0..n-1, so concatenating unequal folds puts a
    # shared fold-size signal in BOTH vectors (perm-null pooled rho ~ +.16-.19
    # on this panel). Normalize to [0,1] so pooling carries no size signal.
    return rank(v) / max(len(v) - 1, 1)


def within(rows, x, y):
    wp, wy = [], []
    for t in TASKS:
        tr = [r for r in rows if r["task"] == t and r.get(x) is not None
              and r.get(y) is not None]
        if len(tr) < 8:
            continue
        wp += list(nrank([r[x] for r in tr]))
        wy += list(nrank([r[y] for r in tr]))
    return (round(spearman(wp, wy), 3), len(wp)) if len(wp) >= 20 else (None, len(wp))


def loto(panel, xcols, ycol, alpha=1.0):
    per, folds = {}, []
    for hold in TASKS:
        tr = [r for r in panel if r["task"] != hold and r.get(ycol) is not None
              and all(r.get(c) is not None for c in xcols)]
        te = [r for r in panel if r["task"] == hold and r.get(ycol) is not None
              and all(r.get(c) is not None for c in xcols)]
        if len(te) < 8 or len(tr) < 20:
            continue
        X = np.array([[r[c] for c in xcols] for r in tr], float)
        y = rank([r[ycol] for r in tr])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        X = (X - mu) / sd
        w = np.linalg.solve(X.T @ X + alpha * np.eye(len(xcols)), X.T @ y)
        Xte = (np.array([[r[c] for c in xcols] for r in te], float) - mu) / sd
        pred = Xte @ w
        ys = [r[ycol] for r in te]
        per[hold] = round(spearman(list(pred), ys), 3)
        folds.append((list(pred), ys))
    pp = [x for pred, ys in folds for x in nrank(pred)]
    py = [x for pred, ys in folds for x in nrank(ys)]
    return per, (round(spearman(pp, py), 3) if pp else None), pp, py, folds


def perm_p(folds, observed, B=2000, seed=11):
    """One-sided p: within-fold shuffle of predictions -> pooled null."""
    if not folds or observed is None:
        return None
    rng = np.random.default_rng(seed)
    py = [x for _, ys in folds for x in nrank(ys)]
    ge = 0
    for _ in range(B):
        pp = [x for pred, _ in folds
              for x in nrank(np.asarray(pred)[rng.permutation(len(pred))])]
        ge += spearman(pp, py) >= observed
    return round((ge + 1) / (B + 1), 4)


def boot_ci(pp, py, B=2000, seed=7):
    if not pp:
        return [None, None]
    rng = np.random.default_rng(seed)
    pp, py = np.asarray(pp), np.asarray(py)
    s = [spearman(list(pp[i]), list(py[i]))
         for i in (rng.integers(0, len(py), len(py)) for _ in range(B))]
    return [round(float(np.percentile(s, 2.5)), 3), round(float(np.percentile(s, 97.5)), 3)]


def zeroshot_pooled(panel, ycol):
    pp, py, folds = [], [], []
    for t in TASKS:
        tr = [r for r in panel if r["task"] == t and r.get("zeroshot") is not None
              and r.get(ycol) is not None]
        if len(tr) >= 8:
            zs = [r["zeroshot"] for r in tr]
            ys = [r[ycol] for r in tr]
            pp += list(nrank(zs))
            py += list(nrank(ys))
            folds.append((zs, ys))
    return (round(spearman(pp, py), 3) if pp else None), pp, py, folds


def main():
    panel = list(json.load(open(BASE / "battery/coda_eval.json"))["panel"])
    for r in panel:
        r["y_seampos"] = None  # computed below for all

    # new criteria
    manifest = json.load(open(BASE / "battery/coda2/coda2_manifest.json"))
    ann = {}
    for p in sorted((BASE / "battery/coda2").glob("annotations_*.json")):
        ann.update(json.load(open(p)))
    zs = json.load(open(BASE / "battery/coda2/zeroshot.json"))
    cam = json.load(open(BASE / "cam_profile.json"))
    nb = json.load(open(BASE / "battery/new_task_boundness.json"))
    # anchor validation
    afail = 0
    for cid, m in manifest.items():
        if m.get("kind") == "anchor" and m.get("hard"):
            g = ann.get(cid, {})
            for f, allowed in m["checks"].items():
                if g.get(f) not in allowed:
                    afail += 1
                    break
    n_added = 0
    for cid, m in manifest.items():
        if m["kind"] != "real" or cid not in ann:
            continue
        t, aid = m["task"], m["aid"]
        rc = next((x for x in cam[t]["per_criterion"] if x["aspect"] == aid), None)
        b = nb.get(f"{t}.{aid}")
        if rc is None or b is None:
            continue
        row = {"task": t, "aid": aid, "y_code": rc["r_base"], "y_hyb": rc["r_hyb"],
               "y_fm": b["fm"], "certified": b["certified"], "rel1": b["rel1"],
               "ratio_mean": None, "zeroshot": zs.get(cid)}
        row.update({k: ann[cid].get(k) for k in FEATS})
        if any(row[k] is None for k in FEATS):
            continue
        panel.append(row)
        n_added += 1

    # seam position: fraction of hybrid signal carried by borrowed judgment.
    # y_hyb (ceiling-normed hybrid) and y_fm (raw marginal); use fm / (fm + max(code_raw,0))
    # with code_raw approximated by y_code on the same ceiling-normed scale as y_hyb, and
    # fm rescaled by ceiling so both live on the ceiling-normed scale.
    for r in panel:
        yc, yh, fm, rel = r.get("y_code"), r.get("y_hyb"), r.get("y_fm"), r.get("rel1")
        if yh is None or yc is None or fm is None:
            continue
        # ceiling-normed field gain = y_hyb - y_code (both ceiling-normed); guard >=0
        gain = max(yh - yc, 0.0)
        base = max(yc, 0.0)
        r["y_seampos"] = round(gain / (gain + base), 3) if (gain + base) > 0.05 else None

    # analyses
    targets = ["y_code", "y_fm", "y_seampos"]
    out = {"n_panel": len(panel), "n_new": n_added, "anchor_hard_fails": afail,
           "pooling": "v2 2026-07-08: within-fold ranks normalized to [0,1] before "
                      "concatenation + within-fold permutation p. v1 pooled raw "
                      "0..n-1 ranks across unequal folds -> fold-size artifact "
                      "(perm-null pooled rho +.16-.19 on this panel); v1 numbers "
                      "retracted in the note's PREDICTABILITY-2 correction.",
           "per_feature_withintask": {}, "loto": {}, "zeroshot": {}}
    for fcol in FEATS + ["rel1", "zeroshot"]:
        out["per_feature_withintask"][fcol] = {
            y: within(panel, fcol, y)[0] for y in targets}
    for y in targets:
        per, pooled, pp, py, folds = loto(panel, FEATS, y)
        zper, zpp, zpy, zfolds = zeroshot_pooled(panel, y)
        out["loto"][y] = {"pooled": pooled, "ci95": boot_ci(pp, py), "per_task": per,
                          "perm_p": perm_p(folds, pooled)}
        out["zeroshot"][y] = {"pooled": zper, "perm_p": perm_p(zfolds, zper)}

    # budget-depth validation (12 ladder criteria): CODA features vs actual seam depth
    ladder = json.load(open(BASE / "battery/budget_ladder.json"))
    fmap = {(r["task"], r["aid"]): r for r in panel}

    def depth(row):
        pts = [(b, row.get(f"b{b}_test")) for b in (0, 1, 2, 4)]
        pts = [(b, v) for b, v in pts if v is not None]
        if not pts:
            return None
        top = max(v for _, v in pts)
        if top <= 0:
            return None
        for b, v in pts:
            if v >= 0.95 * top:
                return b
        return pts[-1][0]
    drows = []
    for k, v in ladder.items():
        t, aid = k.split(".")
        d = depth(v)
        pr = fmap.get((t, aid))
        if d is not None and pr:
            drows.append({**pr, "seam_depth": d})
    bd = {}
    for fcol in FEATS + ["y_fm", "zeroshot"]:
        xs = [r for r in drows if r.get(fcol) is not None]
        if len(xs) >= 8:
            bd[fcol] = round(spearman([r[fcol] for r in xs],
                                      [r["seam_depth"] for r in xs]), 3)
    out["budget_depth_validation"] = {"n": len(drows), "spearman_vs_depth": bd}

    json.dump(out, open(BASE / "battery/coda2_eval.json", "w"), indent=1)

    print(f"panel n={len(panel)} (+{n_added} new)  anchor hard fails={afail}")
    print("\nLOTO rank-ridge (F1-F8), 8-task leave-one-task-out:")
    print(f"{'target':12s} {'pooled':>8s} {'perm_p':>7s} {'ci95':>16s}  "
          f"{'zshot':>6s} {'z_p':>7s}")
    for y in targets:
        L, Z = out["loto"][y], out["zeroshot"][y]
        print(f"{y:12s} {str(L['pooled']):>8s} {str(L['perm_p']):>7s} "
              f"{str(L['ci95']):>16s}  {str(Z['pooled']):>6s} {str(Z['perm_p']):>7s}")
    print("\nper-feature within-task Spearman:")
    print(f"{'feat':6s} {'y_code':>8s} {'y_fm':>8s} {'y_seampos':>10s}")
    for fcol in FEATS + ["rel1", "zeroshot"]:
        e = out["per_feature_withintask"][fcol]
        print(f"{fcol:6s} {str(e['y_code']):>8s} {str(e['y_fm']):>8s} {str(e['y_seampos']):>10s}")
    print(f"\nbudget-depth validation (n={out['budget_depth_validation']['n']}): "
          f"{out['budget_depth_validation']['spearman_vs_depth']}")
    print(f"-> {BASE / 'battery/coda2_eval.json'}")


if __name__ == "__main__":
    main()
