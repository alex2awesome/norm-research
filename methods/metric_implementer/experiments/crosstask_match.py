#!/usr/bin/env python
"""
Cross-task concept matching — Stage A of the iso-morphism scale-out (2026-07-04).

CLAIM UNDER TEST ("iso-morphism between task pairs"): the same human criterion recurs
across tasks, and where it recurs, its ARTICULABILITY PROFILE (per-rung self-readout
recovery from isomorphism_census.py) is correlated — tacitness is concept-intrinsic,
not task-idiosyncratic.

Pipeline (cascade, mirroring the within-task recipe: cheap filter -> strong verify):
  candidates : TF-IDF (word 1-2grams + char 3-5grams) cosine over name+definition+rubric,
               reciprocal top-K union -> candidate pair list (JSON) for judging.
               Raw-embedding matching alone is NOT trusted (within-task calibration:
               BGE cos>=0.95 is only 91% precise at 14% recall) — judged verification
               is mandatory before any pair is called a match.
  score      : join judged pairs (score 0/1/2, double-judged) with each domain's census
               per-metric profiles -> matched-pair profile correlation + permutation null.

Reconstruction-only: matching uses metric TEXT only; profiles are own-verdict recovery.
No task labels anywhere.

Usage:
  python -m methods.metric_implementer.experiments.crosstask_match candidates \
      --grid-a notebooks/data/two_faces_20260702/r3_cw/grid_cw_v1/messages.json --task-a creative-writing \
      --grid-b notebooks/data/two_faces_20260702/r3_humor/grid_humor_v1/messages.json --task-b humor \
      --k 5 --out notebooks/data/two_faces_20260702/crosstask/cw_humor_candidates.json

  python -m methods.metric_implementer.experiments.crosstask_match score \
      --judged notebooks/data/two_faces_20260702/crosstask/cw_humor_judged.json \
      --census notebooks/data/two_faces_20260702/isomorphism_census.json \
      --tags notebooks/data/two_faces_20260702/concept_tags.json \
      --out notebooks/data/two_faces_20260702/crosstask/cw_humor_isomorphism.json
"""
import argparse
import json
import os

import numpy as np


def load_grid_texts(path):
    """Load metric texts from either a grid messages.json (gi -> {name, rubric, rungs})
    or an R3 hierarchy *_r3_expanded.json (merged_groups -> merged_name/description).
    Auto-detected, so non-grid tasks can be matched before their grids exist."""
    m = json.load(open(path))
    out = {}
    if isinstance(m, dict) and "merged_groups" in m:          # hierarchy file
        groups = m["merged_groups"]
        if isinstance(groups, dict):
            groups = list(groups.values())
        for pos, g in enumerate(groups):
            name = g.get("merged_name") or ""
            desc = g.get("merged_description") or ""
            if not (name or desc):
                continue
            # group_idx is often None in the expanded files; the sweep assigns gi by
            # list order ("sweeping N in-order from 0"), so position is the gi.
            gidx = g.get("group_idx")
            gi = str(gidx if gidx is not None else pos)
            out[gi] = {"name": name, "definition": "", "rubric": desc,
                       "text": f"{name} {desc}".strip()}
        return out
    for gi, v in m.items():                                    # grid messages.json
        rungs = v.get("rungs", {})
        text = " ".join(str(x) for x in [
            v.get("name", ""), rungs.get("definition", ""), v.get("rubric", "")])
        out[gi] = {"name": v.get("name", ""), "definition": rungs.get("definition", ""),
                   "rubric": v.get("rubric", ""), "text": text}
    return out


def cmd_candidates(a):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    A = load_grid_texts(a.grid_a)
    B = load_grid_texts(a.grid_b)
    ka, kb = sorted(A, key=int), sorted(B, key=int)
    corpus = [A[g]["text"] for g in ka] + [B[g]["text"] for g in kb]

    sims = []
    for params in [dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                        stop_words="english"),
                   dict(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)]:
        X = TfidfVectorizer(**params).fit_transform(corpus)
        sims.append(cosine_similarity(X[:len(ka)], X[len(ka):]))
    S = np.mean(sims, axis=0)  # word/char blend

    pairs = set()
    K = a.k
    for i in range(len(ka)):                      # top-K per A metric
        for j in np.argsort(-S[i])[:K]:
            pairs.add((i, int(j)))
    for j in range(len(kb)):                      # reciprocal top-K per B metric
        for i in np.argsort(-S[:, j])[:K]:
            pairs.add((int(i), j))
    if a.floor is not None:                       # plus anything above the floor
        for i, j in zip(*np.where(S >= a.floor)):
            pairs.add((int(i), int(j)))

    cand = []
    for i, j in sorted(pairs, key=lambda p: -S[p[0], p[1]]):
        cand.append({
            "gi_a": ka[i], "gi_b": kb[j], "tfidf": round(float(S[i, j]), 4),
            "a": {k: A[ka[i]][k] for k in ("name", "definition", "rubric")},
            "b": {k: B[kb[j]][k] for k in ("name", "definition", "rubric")},
        })
    out = {"task_a": a.task_a, "task_b": a.task_b, "k": K, "floor": a.floor,
           "n_a": len(ka), "n_b": len(kb), "n_candidates": len(cand),
           "note": "judged verification REQUIRED; tfidf is a filter, never a verdict",
           "candidates": cand}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{len(cand)} candidates ({a.task_a} n={len(ka)} x {a.task_b} n={len(kb)}) -> {a.out}")
    print("tfidf dist of candidates:",
          {q: round(float(np.percentile([c['tfidf'] for c in cand], q)), 3)
           for q in (10, 50, 90, 99)})


PROFILE_KEYS = ["name", "definition", "dossier", "gap_def_name", "gap_dossier_name"]


def cmd_score(a):
    from scipy.stats import spearmanr

    judged = json.load(open(a.judged))
    census = json.load(open(a.census))
    tags = None
    if a.tags:
        raw = json.load(open(a.tags))["tags"]
        tags = {}
        for t in raw:
            tags.setdefault(t["domain"], {})[str(t["gi"])] = t["label"]

    task_a, task_b = judged["task_a"], judged["task_b"]
    pm_a = census["domains"][task_a]["per_metric"]
    pm_b = census["domains"][task_b]["per_metric"]

    matches = [p for p in judged["pairs"] if p.get("final_score") == 2]
    related = [p for p in judged["pairs"] if p.get("final_score") == 1]

    # greedy 1-1 subset (dependence control: one profile per metric per side);
    # ties broken by tfidf, which is independent of the profiles under test
    used_a, used_b, one2one = set(), set(), []
    for p in sorted(matches, key=lambda x: -x.get("tfidf", 0.0)):
        if p["gi_a"] in used_a or p["gi_b"] in used_b:
            continue
        used_a.add(p["gi_a"]); used_b.add(p["gi_b"]); one2one.append(p)

    def build_rows(pair_list):
        rows = []
        for p in pair_list:
            va, vb = pm_a.get(p["gi_a"], {}), pm_b.get(p["gi_b"], {})
            if va.get("skip") or vb.get("skip") or not va or not vb:
                continue
            row = {"gi_a": p["gi_a"], "gi_b": p["gi_b"], "name_a": p["a"]["name"],
                   "name_b": p["b"]["name"]}
            for k in PROFILE_KEYS:
                row[f"a_{k}"], row[f"b_{k}"] = va.get(k), vb.get(k)
            if tags:
                row["label_a"] = tags.get(task_a, {}).get(p["gi_a"])
                row["label_b"] = tags.get(task_b, {}).get(p["gi_b"])
            rows.append(row)
        return rows

    def run_tests(rows, rng):
        tests = {}
        for k in PROFILE_KEYS:
            xa = np.array([r[f"a_{k}"] for r in rows], dtype=float)
            xb = np.array([r[f"b_{k}"] for r in rows], dtype=float)
            ok = np.isfinite(xa) & np.isfinite(xb)
            if ok.sum() < 5:
                tests[k] = {"n": int(ok.sum()), "note": "too few"}
                continue
            rho = spearmanr(xa[ok], xb[ok]).statistic
            null = np.empty(a.n_perm)
            xb_ok = xb[ok].copy()
            for t in range(a.n_perm):
                rng.shuffle(xb_ok)
                null[t] = spearmanr(xa[ok], xb_ok).statistic
            p_perm = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (a.n_perm + 1))
            tests[k] = {"n": int(ok.sum()), "spearman_rho": round(float(rho), 3),
                        "p_perm_two_sided": round(p_perm, 4)}
        lab_test = None
        if tags:
            lab = [(r["label_a"], r["label_b"]) for r in rows
                   if r.get("label_a") and r.get("label_b")]
            if lab:
                agree = sum(x == y for x, y in lab)
                la = [x for x, _ in lab]
                lb_arr = np.array([y for _, y in lab])
                null_agree = np.empty(a.n_perm)
                for t in range(a.n_perm):
                    rng.shuffle(lb_arr)
                    null_agree[t] = np.mean(np.array(la) == lb_arr)
                p_lab = float((np.sum(null_agree >= agree / len(lab)) + 1) / (a.n_perm + 1))
                lab_test = {"n": len(lab), "agree": agree,
                            "rate": round(agree / len(lab), 3),
                            "p_perm_one_sided": round(p_lab, 4)}
        return tests, lab_test

    rows_all = build_rows(matches)
    rows_121 = build_rows(one2one)
    rng = np.random.default_rng(0)
    tests_all, lab_all = run_tests(rows_all, rng)
    tests_121, lab_121 = run_tests(rows_121, rng)

    out = {"task_a": task_a, "task_b": task_b, "n_judged": len(judged["pairs"]),
           "n_match2": len(matches), "n_related1": len(related),
           "n_scored_all": len(rows_all), "n_scored_one2one": len(rows_121),
           "profiles": rows_all,
           "profile_tests_all_pairs": tests_all, "label_agreement_all_pairs": lab_all,
           "profile_tests_one2one": tests_121, "label_agreement_one2one": lab_121}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[{task_a} x {task_b}] judged={len(judged['pairs'])} match2={len(matches)} "
          f"related1={len(related)} scored: all={len(rows_all)} 1-1={len(rows_121)}")
    for tag, tests, lab in [("ALL-PAIRS", tests_all, lab_all),
                            ("ONE-TO-ONE", tests_121, lab_121)]:
        print(f" == {tag} ==")
        for k, v in tests.items():
            print(f"  {k:18s} {v}")
        if lab:
            print(f"  label agreement: {lab}")
    print(f"-> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("candidates")
    c.add_argument("--grid-a", required=True)
    c.add_argument("--task-a", required=True)
    c.add_argument("--grid-b", required=True)
    c.add_argument("--task-b", required=True)
    c.add_argument("--k", type=int, default=5)
    c.add_argument("--floor", type=float, default=None)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_candidates)

    s = sub.add_parser("score")
    s.add_argument("--judged", required=True)
    s.add_argument("--census", required=True)
    s.add_argument("--tags", default=None)
    s.add_argument("--n-perm", type=int, default=10000)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_score)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
