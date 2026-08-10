"""E7 SEL pilot — TF-IDF arm (label-clean distill-the-field design, seam note §6).

For each near-categorical LLM field (<=8 distinct normalized values covering >=90% of
train): train a TF-IDF logistic-regression selector S on (x_train, F_gemma(x_train)) —
the certified field's OWN train outputs, never judge scores. Then on held-out test:
  agree(S,F)        — surface-distillability of the field
  fm_S vs fm_F      — plug S's predicted values into the frozen program; judge scores
                      touched ONLY here, in the final rho readout (as in every arm)
  frac_distilled    — fm_S / fm_F (when |fm_F| > .05)
Cross per criterion with E6 transport ratio (inventory) — provenance grid of §6.

Usage: python3 e7_sel_pilot.py [task ...]   (default 4 core tasks)
-> outputs/metric_seam_pilot/battery/e7_sel_pilot.json
"""
import json, pathlib, sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx, load_mod, run_prog, BASE, ROOT  # noqa: E402
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
from certificates import spearman  # noqa: E402

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

SELECTOR = "bge" if "--bge" in sys.argv else "tfidf"
if SELECTOR == "bge":
    sys.argv.remove("--bge")
    from sentence_transformers import SentenceTransformer
    _BGE = SentenceTransformer("BAAI/bge-small-en-v1.5")

TASKS = ["press_releases", "creative_writing", "math", "humor"]
MAXVAL = 8
COVER = 0.90


def norm(x):
    return (x or "").strip().strip('."’”').lower()


def rho_on(sel, col, judge):
    s = [d for d in sel if col.get(d) is not None]
    return spearman([col[d] for d in s], [judge[d] for d in s]) if len(s) >= 20 else float("nan")


def med(xs):
    xs = sorted(x for x in xs if x is not None and x == x)
    return round(xs[len(xs) // 2], 3) if xs else None


def eval_task(task, inv):
    ctx = load_ctx(task)
    train = sorted(ctx["train"])
    test = sorted(ctx["test"])
    emb = None
    if SELECTOR == "bge":  # one embedding pass per task, reused across fields
        ids = [d for d in ctx["items"]]
        vecs = _BGE.encode([ctx["items"][d][:2000] for d in ids],
                           batch_size=64, show_progress_bar=False,
                           normalize_embeddings=True)
        emb = dict(zip(ids, vecs))
    rows = {}
    for aid in sorted(inv):
        f_aid = ctx["f_orig"].get(aid)
        if not f_aid:
            continue
        prog = ctx["hyb"] / f"{aid}_h0.py"
        if not prog.exists():
            continue
        mod = load_mod(prog)
        judge = ctx["judge"].get(aid, {})
        fields = sorted({fn for d in f_aid for fn in f_aid[d]})
        for field in fields:
            tr = [(d, norm(f_aid[d].get(field))) for d in train
                  if d in ctx["items"] and f_aid.get(d, {}).get(field) is not None]
            if len(tr) < 60:
                continue
            vals = Counter(v for _, v in tr)
            top = [v for v, _ in vals.most_common(MAXVAL)]
            if sum(vals[v] for v in top) / len(tr) < COVER or len(vals) < 2:
                continue  # free-text field, not distill-classifiable
            tr = [(d, v) for d, v in tr if v in top]
            if len(Counter(v for _, v in tr)) < 2:
                continue
            te = [d for d in test if d in ctx["items"]]
            clf = LogisticRegression(max_iter=2000, C=1.0)
            if SELECTOR == "bge":
                clf.fit([emb[d] for d, _ in tr], [v for _, v in tr])
                pred = dict(zip(te, clf.predict([emb[d] for d in te])))
            else:
                vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                      sublinear_tf=True, min_df=2)
                Xtr = vec.fit_transform([ctx["items"][d] for d, _ in tr])
                clf.fit(Xtr, [v for _, v in tr])
                pred = dict(zip(te, clf.predict(vec.transform(
                    [ctx["items"][d] for d in te]))))
            # agreement with the real field on test
            both = [d for d in te if f_aid.get(d, {}).get(field) is not None]
            agree = (sum(norm(f_aid[d][field]) == pred[d] for d in both) / len(both)
                     if len(both) >= 30 else None)
            # program-level: swap ONLY this field for S's prediction
            fmap_F = {d: dict(f_aid.get(d, {})) for d in te}
            fmap_S = {d: {**f_aid.get(d, {}), field: pred[d]} for d in te}
            col_F = run_prog(mod.score, ctx["items"], fmap_F, ctx["ops"])
            col_S = run_prog(mod.score, ctx["items"], fmap_S, ctx["ops"])
            col_B = run_prog(mod.score, ctx["items"], {}, ctx["ops"])
            tsel = [d for d in te if d in judge
                    and col_F.get(d) is not None and col_S.get(d) is not None
                    and col_B.get(d) is not None]
            if len(tsel) < 30:
                continue
            rF = rho_on(tsel, col_F, judge)
            rS = rho_on(tsel, col_S, judge)
            rB = rho_on(tsel, col_B, judge)
            if not (rF == rF and rS == rS and rB == rB):
                continue
            fm_F, fm_S = rF - rB, rS - rB
            iv = inv.get(aid, {})
            rows[f"{aid}__{field}"] = {
                "n_train": len(tr), "n_test": len(tsel), "agree": (round(agree, 3)
                                                                   if agree is not None else None),
                "fm_F": round(fm_F, 3), "fm_S": round(fm_S, 3),
                "frac_distilled": (round(fm_S / fm_F, 3) if abs(fm_F) > 0.05 else None),
                "ratio_llama": iv.get("ratio_llama")}
            r = rows[f"{aid}__{field}"]
            print(f"{task} {aid}.{field}: agree={r['agree']} fm_F={r['fm_F']} "
                  f"fm_S={r['fm_S']} distilled={r['frac_distilled']} "
                  f"transport_ratio={r['ratio_llama']}")
    return rows


def main():
    inv_all = json.load(open(BASE / "battery/inventory.json"))
    out = {}
    for task in (sys.argv[1:] or TASKS):
        try:
            rows = eval_task(task, inv_all.get(task, {}))
        except Exception as e:
            out[task] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{task}: ERROR {e}")
            continue
        summ = {"n_fields": len(rows),
                "median_agree": med([r["agree"] for r in rows.values()]),
                "median_frac_distilled": med([r["frac_distilled"]
                                              for r in rows.values()])}
        out[task] = {"fields": rows, "summary": summ}
        print(f"{task} summary: {json.dumps(summ)}")
    path = BASE / ("battery/e7_sel_pilot.json" if SELECTOR == "tfidf"
                   else "battery/e7_sel_pilot_bge.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
