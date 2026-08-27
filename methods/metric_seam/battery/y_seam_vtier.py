"""y-seam V-tier + confound controls: complete the X->y ladder started by y_seam_extend.

Tiers per task (all evaluated as held-out TEST AUC vs the real binary outcome y):
  V  (code)     : every certified h0 program run with an EMPTY LLM-field map — pure
                  code + deterministic/evidence ops, no LLM reads. Median AUC across
                  programs (no post-hoc selection) + a train-fit logistic combination
                  ("V-combined", the linear-combination arm of the manual VAT).
  H  (hybrid)   : same programs with the cached LLM field extractions (where they exist).
  G  (language) : the y_seam_extend forecast-prompt column (already scored).
  len (floor)   : char-length, orientation-free (confound floor).

Confound control (user-flagged: length is a confounder to BALANCE, not just report):
  every AUC is also reported LENGTH-STRATIFIED — test items are split into length
  quintiles and concordant pairs are only counted WITHIN a stratum (a stratified
  c-index: sum_s U_s / sum_s (pos_s*neg_s)). Under stratification the char-length
  floor collapses toward .5 by construction; whatever signal survives is
  length-independent.

Usage: python3 y_seam_vtier.py <task>          (all local CPU; no GPU, no API)
Writes y_seam/y_seam_<task>_vtier.json
"""
import json, sys, pathlib
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam/battery"))
import battery_common as bc
from battery_common import load_mod, run_prog
from y_seam_extend import items_with_y, auc

bc.PROGDIR.update({"legal_title_vii": "programs_legal", "peer_review": "programs_peer",
                   "legal_ss_disability": "programs_ssdis",
                   "code_review": "programs_code_review", "patents_pa": "programs_pa"})
OUT = HERE / "y_seam"; OUT.mkdir(exist_ok=True)
BASE = ROOT / "outputs/metric_seam_pilot"


def prog_dir(task):
    if task == "patents_pa":
        return ROOT / "methods/metric_seam/f2p_mock/programs_pa"
    return ROOT / "methods/metric_seam/hybrids" / bc.PROGDIR[task]


def task_ops(task, ctx):
    if task == "patents_pa":
        sys.path.insert(0, str(ROOT / "methods/metric_seam/f2p_mock"))
        from ops_pa import PriorArtOps
        return PriorArtOps(BASE / "tasks/patents_pa/pa_features.json",
                           corpus_path=str(BASE / "tasks/patents_pa/items.json"))
    return ctx["ops"]


def stratified_auc(scores, labels, lens, n_strata=5):
    """c-index counting concordant pairs only within length quintiles."""
    order = sorted(range(len(lens)), key=lambda i: lens[i])
    strata = [order[(k * len(order)) // n_strata:((k + 1) * len(order)) // n_strata]
              for k in range(n_strata)]
    num = den = 0.0
    for idx in strata:
        s = [scores[i] for i in idx]; l = [labels[i] for i in idx]
        npos = sum(l); nneg = len(l) - npos
        if npos == 0 or nneg == 0:
            continue
        a, _, _ = auc(s, l)
        num += a * npos * nneg; den += npos * nneg
    return num / den if den else float("nan")


def main(task):
    ctx = bc.load_ctx(task)
    iy = items_with_y(task)
    ops = task_ops(task, ctx)
    train_ids = sorted(ctx["train"]); test_ids = sorted(ctx["test"])
    items = ctx["items"]

    # --- run every h0 program: V (empty fields) and H (cached fields) columns ---
    progs = sorted(prog_dir(task).glob("a*_h0.py"))
    cols_V, cols_H, failed = {}, {}, []
    for p in progs:
        aid = p.stem[:-3]
        try:
            mod = load_mod(p)
            cols_V[aid] = run_prog(mod.score, items, {}, ops)              # pure mechanism
            fmap = ctx["f_orig"].get(aid, {})
            if fmap:
                cols_H[aid] = run_prog(mod.score, items, fmap, ops)        # hybrid
        except Exception as e:
            failed.append(f"{aid}:{type(e).__name__}")
    print(f"{task}: ran {len(cols_V)}/{len(progs)} programs (V), {len(cols_H)} with fields (H)"
          + (f"  FAILED: {failed}" if failed else ""))

    # --- G column from the existing y-seam scoring ---
    g = {}
    res_path = OUT / f"y_seam_{task}_results.jsonl"
    for line in open(res_path):
        r = json.loads(line)
        if r.get("aspect_id", "").endswith(".Y.final") and isinstance(r.get("score"), int):
            g[r["datapoint_id"]] = r["score"]

    sel = [d for d in test_ids if d in g and iy.get(d, ("", None))[1] in (0, 1)]
    y = [iy[d][1] for d in sel]
    lens = [len(iy[d][0]) for d in sel]

    def col_auc(col):
        vals = [col.get(d) for d in sel]
        if any(v is None for v in vals):
            keep = [i for i, v in enumerate(vals) if v is not None]
            if len(keep) < 0.8 * len(sel):
                return None
            a, _, _ = auc([vals[i] for i in keep], [y[i] for i in keep])
            sa = stratified_auc([vals[i] for i in keep], [y[i] for i in keep],
                                [lens[i] for i in keep])
            return a, sa
        a, _, _ = auc(vals, y)
        return a, stratified_auc(vals, y, lens)

    import statistics as st
    per_prog = {}
    for aid, col in cols_V.items():
        r = col_auc(col)
        if r:
            per_prog[aid] = dict(auc=round(r[0], 4), auc_len_strat=round(r[1], 4))
    V_med = st.median(v["auc"] for v in per_prog.values()) if per_prog else float("nan")
    V_med_s = st.median(v["auc_len_strat"] for v in per_prog.values()) if per_prog else float("nan")

    H_med = H_med_s = None
    if cols_H:
        hs = [col_auc(c) for c in cols_H.values()]
        hs = [h for h in hs if h]
        if hs:
            H_med = st.median(h[0] for h in hs); H_med_s = st.median(h[1] for h in hs)

    # --- V-combined: logistic over all V columns, TRAIN-fit, TEST AUC ---
    V_comb = V_comb_s = None
    if len(per_prog) >= 2:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        aids = sorted(per_prog)
        def matrix(ids):
            X, yy, ll = [], [], []
            for d in ids:
                lab = iy.get(d, ("", None))[1]
                if lab not in (0, 1):
                    continue
                row = [cols_V[a].get(d) for a in aids]
                if any(v is None for v in row):
                    continue
                X.append(row); yy.append(lab); ll.append(len(iy[d][0]))
            return np.array(X, float), np.array(yy), ll
        Xtr, ytr, _ = matrix(train_ids)
        Xte, yte, lte = matrix(test_ids)
        if len(set(ytr)) == 2 and len(Xte) >= 40:
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            clf = LogisticRegression(max_iter=2000, C=1.0).fit((Xtr - mu) / sd, ytr)
            p = clf.predict_proba((Xte - mu) / sd)[:, 1]
            a, _, _ = auc(list(p), list(yte))
            V_comb = round(a, 4)
            V_comb_s = round(stratified_auc(list(p), list(yte), lte), 4)

    # --- G + length floor ---
    gv = [g[d] for d in sel]
    G_auc, _, _ = auc(gv, y)
    G_s = stratified_auc(gv, y, lens)
    L_auc, _, _ = auc(lens, y)
    L_abs = max(L_auc, 1 - L_auc)
    L_s = stratified_auc(lens, y, lens)  # ~.5 by construction (sanity)

    res = dict(task=task, n_test=len(sel), n_pos=sum(y), n_neg=len(y) - sum(y),
               n_programs=len(per_prog), programs_failed=failed,
               V_median_auc=round(V_med, 4), V_median_auc_len_strat=round(V_med_s, 4),
               V_combined_auc=V_comb, V_combined_auc_len_strat=V_comb_s,
               H_median_auc=round(H_med, 4) if H_med is not None else None,
               H_median_auc_len_strat=round(H_med_s, 4) if H_med_s is not None else None,
               G_auc=round(G_auc, 4), G_auc_len_strat=round(G_s, 4),
               charlen_auc_abs=round(L_abs, 4), charlen_auc_len_strat=round(L_s, 4),
               per_program=per_prog)
    json.dump(res, open(OUT / f"y_seam_{task}_vtier.json", "w"), indent=1)
    print(f"{task}: V_med={V_med:.3f} (strat {V_med_s:.3f}) | "
          f"V_comb={V_comb} (strat {V_comb_s}) | "
          f"H_med={H_med if H_med is None else round(H_med,3)} | "
          f"G={G_auc:.3f} (strat {G_s:.3f}) | len_abs={L_abs:.3f} (strat {L_s:.3f})")


if __name__ == "__main__":
    main(sys.argv[1])
