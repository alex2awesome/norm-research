#!/usr/bin/env python3
"""Layer-1 "nonlinear stack" of the taste-residual decomposition, mathlib_verdict cell
(mathlib4 PR accept/reject -- "PR merged" verdict).

Design: notes/2026-08-05__taste-decomposition-design.md (S0 ledger, S1 protocol).
Pilot precedent + protocol notes: notes/2026-08-05__layer1_peer_verdict_pilot.md.
Structural template: methods/taste_decomposition/nc_layer1_stack.py (seeds {0,1,2}
mean+spread; PRIMARY group-level cluster bootstrap on Delta_interact).

SPECIAL RULE FOR THIS CELL (task instruction): T=.770 for mathlib is UNVERIFIED-split.
This script NEVER computes/reports Delta_total or Delta_beyond -- ledger stops at
Delta_interact. No T claim is made or hunted for.

--------------------------------------------------------------------------------
DISCOVERY LOG (how the V / A / VA ambiguity was resolved)
--------------------------------------------------------------------------------
Local search confirmed the raw parquet + aggregation scripts do NOT exist under
datasets/math/mathlib/ locally (only friction_dataset / mathlib_rescore authoring
artifacts, neither of which produced the V.68-ish/VA.67-ish numbers quoted in the
design doc). All of the following were found ONLY on sk3 under
/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib/ and scp'd down
(read-only on sk3; nothing modified/deleted there) into
methods/taste_decomposition/mathlib_verdict_data/ (kept OUTSIDE datasets/ per this
task's hard rule "only write new files under methods/taste_decomposition/"):
  accept_reject_clean.parquet   (n=7956, the CANONICAL clean accept/reject slice,
                                  VAT_CLOSURE.md 2026-06-25; has a pre-existing
                                  train/eval/test "split" column, judgement=accept)
  mathlib_diff_v_features.parquet (22 deterministic V features, join key "number")
  a_metric_verdicts_mathlib.jsonl (10 LLM-judged review-norm criteria m01-m10,
                                  Qwen3.5-122B, join key "number")
  mathlib_remeasure2.py         (THE script that produced the closure-table numbers)
  mathlib_tactic_decomp2.py, mathlib_authorstrip_topic.py, save_deconf.py,
  finalize_slice.py, clean_ladder.py, mathlib_top_tfidf.py, what_is_abandon.py
    (upstream/sibling scripts -- read for context, not re-run)

Ambiguity resolution (V vs VA, and raw vs topic-residualized):
mathlib_remeasure2.py's own final print table has FIVE named rows: "V (orig)",
"V' (V+tactic)", "C (no-auth TFIDF)", "A (m01-10)", "A + V'" -- each with a RAW
and a topic-residualized column. Two independently-plausible readings of the
task brief's "V .684/VA .680" existed (both close to numbers in this table but
neither an exact match); running the ACTUAL script locally (this repo, same
parquet) reproduces the table to the printed precision:
    V (orig)          raw 0.6488  resid 0.635
    V' (V+tactic)      raw 0.6827  resid 0.680
    A (m01-10)         raw 0.4568  resid 0.478
    A + V'             raw 0.6683  resid 0.666
The standard VA = V+A convention used everywhere else in this framework settles
which row is "VA": it MUST be the row that literally concatenates A onto the V
block -- i.e. "A + V'" -- not "V' resid" (which never touches A and was one of
the two candidate misreadings). This also matches
notes/2026-07-27__vat-run-registry.md line 66 ("mathlib maintainer accept: V
.680/VA .668/T .770 (split unverified -- V4 verify)") almost exactly on the VA
side (.6683 vs quoted .668, diff .0003) -- the registry entry is therefore
taken as the closer paraphrase of the same underlying script, and RAW (not
topic-residualized) is used as the Layer-1 gate target, because:
  (a) raw is the plain, unadjusted linear-aggregation AUC -- the ordinary
      meaning of "_lin" everywhere else in this framework;
  (b) raw VA = 0.6683 matches the registry's .668 almost exactly, while
      neither raw nor resid VA is close to the design doc's approximate ".680";
  (c) topic-residualization is a confound-control diagnostic (conceptually a
      Layer-2(b) nuisance-stratification check -- "area" is exactly the kind
      of topic covariate Layer 2(b) stratifies on), out of scope for a Layer-1
      V/A/VA gate.
So: GATE TARGET = V' raw 0.6827 (~design doc's "V .684", off by .0013,
registry's "V .680" off by .0027 -- both within tolerance) and VA raw 0.6683
(design doc's "VA .680" is OFF by .012, outside tolerance and evidently the
mis-transcription; registry's "VA .668" matches to .0003). "V" in this
ledger = V' (V + 33 regex tactic-idiom counts), matching what mathlib_remeasure2.py
itself concatenates A onto to form "VA" -- NOT the smaller un-enriched "V (orig)"
block. Both design-doc and registry targets are reported in the gate table below
for transparency; the registry's is closer and is treated as authoritative
because it (uniquely) reproduces the VA row to <0.001.

Group unit: the ORIGINAL script (mathlib_remeasure2.py) uses NO GroupKFold and
NO CV at all -- linear aggregation is a single fixed train/eval split read off
the parquet's own "split" column (stratified, not grouped: pos rate ~.94 in
every split, near-identical across train/eval/test; NOT date-ordered, NOT
author-grouped -- there is no author/contributor id column in the clean slice
at all, only free-text copyright lines inside the diff). Per the task's own
instruction ("use whichever the ORIGINAL script actually used, don't invent a
new grouping scheme"), the outer "fold" here COLLAPSES TO ONE FOLD (train fits,
eval reads out) -- this is an explicit, documented deviation from the
GroupKFold(5) outer-fold structure used in every other cell so far, forced by
this cell's own published methodology having no CV. A genuine group unit is
still required for (i) the FROZEN inner GroupKFold(3) grid-selection step and
(ii) the FROZEN PRIMARY group-level cluster bootstrap on Delta_interact. The
one group covariate that is ALREADY part of the original pipeline (used for its
own topic-confound residualization step) is "area" -- the top-level Mathlib/
<Area>/ path most frequently touched by the diff (regex identical to
mathlib_authorstrip_topic.py / mathlib_remeasure2.py's own area() function;
31 distinct values incl. "NONE" for diffs matching no Mathlib/<Area>/ path,
train n_areas=30 min-count 3, eval n_areas=28). This is used rather than
inventing a new grouping scheme (e.g. PR author, which has no clean id column
here) or leaving the cell fully ungrouped.

CPU only. No new judging. Usage:
  python mathlib_verdict_layer1.py
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
ML = Path(__file__).resolve().parent / "mathlib_verdict_data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------- published targets -
# design doc (notes/2026-08-05__taste-decomposition-design.md L151): approx
# "V .684/VA .680" -- traced to a mis-transcription (see discovery-log docstring).
DESIGN_DOC_APPROX = {"V": 0.684, "VA": 0.680}
# notes/2026-07-27__vat-run-registry.md L66 (closer paraphrase of the same script)
REGISTRY_APPROX = {"V": 0.680, "VA": 0.668, "T": 0.770}
GATE_TOL = 0.005

TACTICS = [
    "grind", "aesop", "simp", "simpa", "fun_prop", "funprop", "cat_disch",
    "catdisch", "decide", "norm_num", "ring", "nlinarith", "linarith", "omega",
    "intro", "apply", "have", "unfold", "rw", "rewrite", "cases", "induction",
    "exact", "refine", "rwa", "trans", "calc", "change", "ext", "constructor",
    "congr", "simps", "obtain",
]
_TAC_PAT = {t: re.compile(r"\b" + t + r"\b") for t in TACTICS}

GRID = [
    {"max_leaf_nodes": 15, "learning_rate": 0.06, "max_iter": 400},
    {"max_leaf_nodes": 31, "learning_rate": 0.06, "max_iter": 400},
]
N_INNER = 3
GBM_SEEDS = (0, 1, 2)


# ---------------------------------------------------------------- data -----
def area_of(diff_text: str) -> str:
    ms = re.findall(r"(?:a|b)/Mathlib/([A-Za-z0-9_]+)/", str(diff_text))
    return Counter(ms).most_common(1)[0][0] if ms else "NONE"


def tactic_counts(diffs: pd.Series) -> np.ndarray:
    M = np.zeros((len(diffs), len(TACTICS)))
    for i, d in enumerate(diffs.astype(str)):
        for j, t in enumerate(TACTICS):
            M[i, j] = len(_TAC_PAT[t].findall(d))
    return M


def clean_cols(M, names):
    """EXACT copy of nc_layer1_stack.clean_cols (== aggregate_nc_multiy.clean_cols):
    drop degenerate cols, median-impute NA. Verified to drop exactly the two
    all-zero regex artifacts ('funprop','catdisch' -- the underscore-free
    variants never match real Lean text) and nothing else; dropping a
    zero-variance column changes NO LogisticRegression prediction (StandardScaler
    zeroes it identically whether kept or dropped), so this guard does not
    perturb the gate reproduction below."""
    keep, out = [], []
    for j in range(M.shape[1]):
        col = M[:, j].astype(float)
        nonna = col[~np.isnan(col)]
        if len(nonna) == 0:
            continue
        med = np.median(nonna)
        c = np.where(np.isnan(col), med, col)
        vals, counts = np.unique(c, return_counts=True)
        offmodal = len(c) - counts.max()
        if offmodal < 5 or c.std() == 0:
            continue
        keep.append(j)
        out.append(c)
    if not out:
        return np.zeros((M.shape[0], 0)), []
    return np.column_stack(out), [names[j] for j in keep]


def load_data():
    df = pd.read_parquet(ML / "accept_reject_clean.parquet").reset_index(drop=True)
    df["area"] = df["diff"].astype(str).map(area_of)

    Vraw = pd.read_parquet(ML / "mathlib_diff_v_features.parquet")
    vf = [c for c in Vraw.columns if c != "number"]
    Vm_full = (df.merge(Vraw[["number"] + vf], on="number", how="left")[vf]
               .apply(pd.to_numeric, errors="coerce").values.astype(float))
    Vm_full = np.where(np.isnan(Vm_full), 0, Vm_full)
    tac_full = tactic_counts(df["diff"])
    Vprime_full = np.hstack([Vm_full, tac_full])
    vprime_names = vf + TACTICS

    av = pd.read_json(ML / "a_metric_verdicts_mathlib.jsonl", lines=True)
    anorms = [c for c in av.columns if c.startswith("m") and c[1:3].isdigit()]
    av = av[["number"] + anorms].dropna(subset=anorms).drop_duplicates("number")
    adf = df.merge(av, on="number", how="inner").reset_index(drop=True)
    Am = adf[anorms].values.astype(float)
    Am = np.where(np.isnan(Am), 0, Am)

    VmA = (adf.merge(Vraw[["number"] + vf], on="number", how="left")[vf]
           .apply(pd.to_numeric, errors="coerce").values.astype(float))
    VmA = np.where(np.isnan(VmA), 0, VmA)
    tacA = tactic_counts(adf["diff"])
    VprimeA = np.hstack([VmA, tacA])

    return dict(df=df, adf=adf, Vprime_full=Vprime_full, vprime_names=vprime_names,
                Am=Am, anorms=anorms, VprimeA=VprimeA)


# -------------------------------------------------------- linear (gate) ----
def fit_predict_linear(Xtr, ytr, Xte):
    pipe = make_pipeline(StandardScaler(),
                          LogisticRegression(class_weight="balanced", max_iter=3000))
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xte)[:, 1]


# ------------------------------------------------------------- nonlinear ---
def _fit_gbm(params, seed):
    return HistGradientBoostingClassifier(
        max_leaf_nodes=params["max_leaf_nodes"], learning_rate=params["learning_rate"],
        max_iter=params["max_iter"], early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=seed)


def gbm_single_split(Xtr, ytr, Xte, yte, groups_tr, seed, verbose=False):
    """Single outer 'fold' (train fits, eval reads out) mirroring this cell's own
    published single-split methodology; grid picked by inner GroupKFold(3) on
    area WITHIN train only -- no eval-row information ever enters selection."""
    n_inner = min(N_INNER, len(np.unique(groups_tr)))
    inner = list(GroupKFold(n_splits=n_inner).split(np.zeros(len(ytr)), groups=groups_tr))
    scores = []
    for params in GRID:
        aucs = []
        for itr, ite in inner:
            m = _fit_gbm(params, seed)
            m.fit(Xtr[itr], ytr[itr])
            aucs.append(roc_auc_score(ytr[ite], m.predict_proba(Xtr[ite])[:, 1]))
        scores.append(float(np.mean(aucs)))
    best = int(np.argmax(scores))
    m = _fit_gbm(GRID[best], seed)
    m.fit(Xtr, ytr)
    pred = m.predict_proba(Xte)[:, 1]
    train_auc = float(roc_auc_score(ytr, m.predict_proba(Xtr)[:, 1]))
    if verbose:
        print(f"    seed {seed}: pick leaves={GRID[best]['max_leaf_nodes']} "
              f"inner={scores} train_auc={train_auc:.4f} eval_auc={roc_auc_score(yte, pred):.4f}")
    return {"pred": pred, "pick": GRID[best]["max_leaf_nodes"], "inner_auc": scores,
            "train_auc": train_auc, "eval_auc": float(roc_auc_score(yte, pred))}


# --------------------------------------------------------------- bootstrap -
def bootstrap_delta_interact_area(y, lin_pred, nl_pred, groups, n_boot=2000, seed=0):
    """PRIMARY: group-level (area) cluster bootstrap over the EVAL rows.
    'area' is the ONE group covariate present in the original pipeline (used
    there for its own topic-confound residualization); generalizes
    nc_layer1_stack.bootstrap_delta_interact_docket to this cell's own group
    unit, per the frozen requirement that group-level is PRIMARY."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    deltas = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in draw])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        deltas.append(float(roc_auc_score(yb, nl_pred[idx]) - roc_auc_score(yb, lin_pred[idx])))
    deltas = np.array(deltas)
    return {"n_boot_used": int(len(deltas)), "n_groups_resampled": int(len(uniq)),
            "mean": float(deltas.mean()),
            "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            "p_gt_0": float((deltas > 0).mean())}


def bootstrap_delta_interact_row(y, lin_pred, nl_pred, n_boot=2000, seed=0):
    """Secondary row-level diagnostic (pilot's original method)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        deltas.append(float(roc_auc_score(yb, nl_pred[idx]) - roc_auc_score(yb, lin_pred[idx])))
    deltas = np.array(deltas)
    return {"n_boot_used": int(len(deltas)), "mean": float(deltas.mean()),
            "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            "p_gt_0": float((deltas > 0).mean())}


# ----------------------------------------------------------------- shap ----
def shap_interactions(X, y, names, seed=0, n_sub=300, top_k=15):
    import shap
    m = _fit_gbm(GRID[1], seed)
    m.fit(X, y)
    ex = shap.TreeExplainer(m)
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(y), size=min(n_sub, len(y)), replace=False)
    sv = ex.shap_values(X[sub])
    if isinstance(sv, list):
        sv = sv[-1]
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    imp = np.abs(sv).mean(0)
    top = np.argsort(-imp)[:top_k]
    top_names = [names[j] for j in top]
    m2 = _fit_gbm(GRID[1], seed)
    m2.fit(X[:, top], y)
    ex2 = shap.TreeExplainer(m2)
    iv = ex2.shap_interaction_values(X[sub][:, top])
    if isinstance(iv, list):
        iv = iv[-1]
    if iv.ndim == 4:
        iv = iv[:, :, :, -1]
    M = np.abs(iv).mean(0)
    pairs = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            pairs.append((top_names[i], top_names[j], float(M[i, j] + M[j, i])))
    pairs.sort(key=lambda t: -t[2])
    diag = [(top_names[i], float(M[i, i])) for i in range(len(top))]
    off_frac = float((M.sum() - np.trace(M)) / M.sum())
    return {"top_features": [{"name": n, "mean_abs_shap": float(imp[j])} for n, j in zip(top_names, top)],
            "top_pairs": [{"a": a, "b": b, "mean_abs_interaction": v} for a, b, v in pairs[:10]],
            "main_effects": [{"name": n, "mean_abs_main": v} for n, v in diag],
            "offdiagonal_mass_fraction": off_frac, "n_subsample": int(len(sub))}


# ----------------------------------------------------------------- main ----
def main():
    t0 = time.time()
    d = load_data()
    df, adf = d["df"], d["adf"]

    tr_full = (df["split"] == "train").values
    ev_full = (df["split"] == "eval").values
    trA = (adf["split"] == "train").values
    evA = (adf["split"] == "eval").values

    y_full = df["judgement"].values.astype(int)
    y_A = adf["judgement"].values.astype(int)
    groups_full = df["area"].values
    groups_A = adf["area"].values

    print(f"V-pool (full clean slice) n={len(df)} train={tr_full.sum()} eval={ev_full.sum()} "
          f"pos_rate={y_full.mean():.4f} n_areas={len(np.unique(groups_full))}")
    print(f"A/VA-pool (A-scored subset) n={len(adf)} train={trA.sum()} eval={evA.sum()} "
          f"pos_rate={y_A.mean():.4f} n_areas={len(np.unique(groups_A))}")

    res = {
        "cell": "mathlib_verdict", "title": "mathlib4 PR accept/reject (maintainer merge verdict)",
        "sklearn_version": sklearn.__version__,
        "note_scope": "SPECIAL RULE: T=.770 is UNVERIFIED-split for this cell. "
                       "Delta_total / Delta_beyond are NOT computed or reported here.",
        "n_pool_V": int(len(df)), "n_pool_VA": int(len(adf)),
        "n_eval_V": int(ev_full.sum()), "n_eval_VA": int(evA.sum()),
        "pos_rate_pool": float(y_full.mean()),
        "group_column": "area (top-level Mathlib/<Area>/ path most touched by the diff; "
                         "31 distinct incl. NONE; the ONLY group covariate already present "
                         "in the original pipeline -- no author/contributor id column exists "
                         "in the clean slice)",
        "n_groups_V_pool": int(len(np.unique(groups_full))),
        "n_groups_VA_pool": int(len(np.unique(groups_A))),
        "outer_fold_structure": "SINGLE fixed train/eval split (parquet's own 'split' column, "
                                 "stratified NOT grouped) -- the ORIGINAL published pipeline "
                                 "(mathlib_remeasure2.py) uses no CV/GroupKFold at all, so the "
                                 "'outer fold' collapses to one fold here (explicit deviation "
                                 "from the 5-fold structure used in other cells, forced by this "
                                 "cell's own methodology).",
        "matrix": str(ML),
        "a_bank": "10 LLM-judged review-norm criteria (m01-m10), Qwen3.5-122B "
                  "(a_metric_verdicts_mathlib.jsonl) -- pre-dates the current A-bank-build "
                  "rule (GEPA + Gemma-4-31B); reused as-is per 'no new judging' rule.",
    }

    # ---- feature blocks (post degeneracy guard) -----------------------------
    V_full_c, v_full_names = clean_cols(d["Vprime_full"], d["vprime_names"])
    A_c, a_names = clean_cols(d["Am"], d["anorms"])
    VprimeA_c, vprimeA_names = clean_cols(d["VprimeA"], d["vprime_names"])
    VA_c = np.column_stack([VprimeA_c, A_c])
    VA_names = vprimeA_names + a_names
    res["n_features"] = {"V": int(V_full_c.shape[1]), "A": int(A_c.shape[1]), "VA": int(VA_c.shape[1])}
    res["dropped_degenerate_cols"] = {
        "V_dropped": sorted(set(d["vprime_names"]) - set(v_full_names)),
        "VA_V_component_dropped": sorted(set(d["vprime_names"]) - set(vprimeA_names)),
    }
    print(f"post-guard feature counts: V={V_full_c.shape[1]} A={A_c.shape[1]} VA={VA_c.shape[1]} "
          f"(dropped {res['dropped_degenerate_cols']['V_dropped']})")

    # ---- linear gate + ledger (StandardScaler + LR(class_weight=balanced)) --
    lin_pred = {}
    lin_auc = {}
    for key, X, tr, ev, y in [
        ("V", V_full_c, tr_full, ev_full, y_full),
        ("A", A_c, trA, evA, y_A),
        ("VA", VA_c, trA, evA, y_A),
    ]:
        p = fit_predict_linear(X[tr], y[tr], X[ev])
        lin_pred[key] = p
        lin_auc[key] = float(roc_auc_score(y[ev], p))
        print(f"  linear {key:2s}: {lin_auc[key]:.4f}  (n_eval={ev.sum()})")
    res["linear"] = lin_auc

    # ---- gate (reproduce this cell's OWN published linear numbers) ----------
    res["gate"] = {
        "V": {"design_doc_approx": DESIGN_DOC_APPROX["V"], "registry_approx": REGISTRY_APPROX["V"],
              "script_own_exact_raw": 0.682667, "reproduced": lin_auc["V"],
              "abs_diff_vs_script_own": abs(lin_auc["V"] - 0.682667),
              "abs_diff_vs_design_doc": abs(lin_auc["V"] - DESIGN_DOC_APPROX["V"]),
              "abs_diff_vs_registry": abs(lin_auc["V"] - REGISTRY_APPROX["V"]),
              "pass": abs(lin_auc["V"] - 0.682667) <= GATE_TOL},
        "VA": {"design_doc_approx": DESIGN_DOC_APPROX["VA"], "registry_approx": REGISTRY_APPROX["VA"],
               "script_own_exact_raw": 0.668279, "reproduced": lin_auc["VA"],
               "abs_diff_vs_script_own": abs(lin_auc["VA"] - 0.668279),
               "abs_diff_vs_design_doc": abs(lin_auc["VA"] - DESIGN_DOC_APPROX["VA"]),
               "abs_diff_vs_registry": abs(lin_auc["VA"] - REGISTRY_APPROX["VA"]),
               "pass": abs(lin_auc["VA"] - 0.668279) <= GATE_TOL},
        "A": {"vat_closure_range_raw": "~0.46-0.56 (chance, split-noisy per VAT_CLOSURE.md)",
              "script_own_exact_raw": 0.456794, "reproduced": lin_auc["A"],
              "abs_diff_vs_script_own": abs(lin_auc["A"] - 0.456794),
              "pass": abs(lin_auc["A"] - 0.456794) <= GATE_TOL,
              "note": "no single published point target for A; gated against this repo's own "
                      "exact re-derivation of the script instead"},
    }
    gate_pass = res["gate"]["V"]["pass"] and res["gate"]["VA"]["pass"]
    res["gate_pass"] = gate_pass
    print("GATE:", "PASS" if gate_pass else "FAIL",
          json.dumps({k: round(v["abs_diff_vs_script_own"], 6) for k, v in res["gate"].items()}))
    if not gate_pass:
        res["runtime_sec"] = time.time() - t0
        (RESULTS_DIR / "mathlib_verdict_layer1.json").write_text(json.dumps(res, indent=2, default=str))
        print("GATE FAILED -- stopping, no nonlinear stack run.")
        return

    # ---- nonlinear: V, A, VA -- seeds 0/1/2, single train/eval split --------
    nl_pred_by_key_seed = {}
    nl_info = {"V": {}, "A": {}, "VA": {}}
    for key, X, tr, ev, y, groups in [
        ("V", V_full_c, tr_full, ev_full, y_full, groups_full),
        ("A", A_c, trA, evA, y_A, groups_A),
        ("VA", VA_c, trA, evA, y_A, groups_A),
    ]:
        for seed in GBM_SEEDS:
            r = gbm_single_split(X[tr], y[tr], X[ev], y[ev], groups[tr], seed, verbose=(seed == 0))
            nl_pred_by_key_seed[(key, seed)] = r["pred"]
            nl_info[key][str(seed)] = {k: v for k, v in r.items() if k != "pred"}
            print(f"  gbm {key:2s} seed {seed}: eval_auc={r['eval_auc']:.4f} train_auc={r['train_auc']:.4f} pick={r['pick']}")

    nl_auc = {}
    nl_spread = {}
    nl_mean_prob = {}
    for key in ["V", "A", "VA"]:
        aucs = [nl_info[key][str(s)]["eval_auc"] for s in GBM_SEEDS]
        nl_auc[key] = float(np.mean(aucs))
        nl_spread[key] = float(max(aucs) - min(aucs))
        nl_mean_prob[key] = np.mean([nl_pred_by_key_seed[(key, s)] for s in GBM_SEEDS], axis=0)
    res["nonlinear"] = {
        k: {"seed_aucs": {str(s): nl_info[k][str(s)]["eval_auc"] for s in GBM_SEEDS},
            "mean_auc": nl_auc[k], "spread": nl_spread[k],
            "train_auc_seed0": nl_info[k]["0"]["train_auc"], "picks": {str(s): nl_info[k][str(s)]["pick"] for s in GBM_SEEDS}}
        for k in ["V", "A", "VA"]
    }

    np.save(RESULTS_DIR / "mathlib_verdict_va_nl_oof_seed0.npy", nl_pred_by_key_seed[("VA", 0)])
    np.save(RESULTS_DIR / "mathlib_verdict_va_nl_oof_mean3.npy", nl_mean_prob["VA"])

    ledger = {
        "V_lin": lin_auc["V"], "V_nl_mean": nl_auc["V"], "V_nl_spread": nl_spread["V"],
        "A_lin": lin_auc["A"], "A_nl_mean": nl_auc["A"], "A_nl_spread": nl_spread["A"],
        "VA_lin": lin_auc["VA"], "VA_nl_mean": nl_auc["VA"], "VA_nl_spread": nl_spread["VA"],
        "Delta_interact": nl_auc["VA"] - lin_auc["VA"],
        "V_interact": nl_auc["V"] - lin_auc["V"],
        "note": "No T / Delta_total / Delta_beyond per this cell's special rule "
                "(T=.770 is UNVERIFIED-split).",
    }
    res["ledger"] = ledger
    res["overfit_gap"] = {k: nl_info[k]["0"]["train_auc"] - nl_info[k]["0"]["eval_auc"] for k in ["V", "A", "VA"]}
    print(json.dumps(ledger, indent=2))

    # ---- bootstrap CI on Delta_interact (nl = mean-of-3-seeds probs) --------
    res["delta_interact_bootstrap_area_PRIMARY"] = bootstrap_delta_interact_area(
        y_A[evA], lin_pred["VA"], nl_mean_prob["VA"], groups_A[evA])
    res["delta_interact_bootstrap_row_secondary"] = bootstrap_delta_interact_row(
        y_A[evA], lin_pred["VA"], nl_mean_prob["VA"])
    b = res["delta_interact_bootstrap_area_PRIMARY"]
    print(f"Delta_interact PRIMARY (area-cluster, n_groups={b['n_groups_resampled']}): "
          f"mean={b['mean']:.4f} CI95={b['ci95']} P(>0)={b['p_gt_0']:.2f}")

    # ---- SHAP screen (descriptive only) --------------------------------------
    try:
        print("  shap (VA, full A-pool train+eval) ...")
        VA_all = np.vstack([VA_c[trA], VA_c[evA]])
        y_all = np.concatenate([y_A[trA], y_A[evA]])
        res["shap"] = shap_interactions(VA_all, y_all, VA_names, seed=0)
    except Exception as e:  # pragma: no cover
        res["shap"] = {"error": repr(e)}
        print("  shap FAILED:", e)

    res["protocol_notes"] = [
        "Group unit = 'area' (top-level Mathlib/<Area>/ path), the only group covariate "
        "already present in the original pipeline; no PR-author id column exists in the "
        "clean slice (author confound is handled via free-text copyright-line stripping, "
        "not a groupable id).",
        "Outer fold structure is a SINGLE train/eval split (this cell's own published "
        "methodology has no CV at all), not GroupKFold(5) -- documented deviation from "
        "other cells, forced by fidelity to the original pipeline.",
        "V ('V-prime') pool (n=7956) is larger than the A/VA pool (n=7921, A-scored "
        "subset only) -- mirrors mathlib_remeasure2.py's own population choice per block, "
        "not invented here. Delta_interact only ever compares VA_nl vs VA_lin on the "
        "IDENTICAL VA-pool eval rows, so this population split does not confound it.",
        "No Delta_total / Delta_beyond / T claim in this JSON, per this cell's special rule.",
    ]

    res["runtime_sec"] = time.time() - t0
    (RESULTS_DIR / "mathlib_verdict_layer1.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {RESULTS_DIR / 'mathlib_verdict_layer1.json'}  ({res['runtime_sec']:.1f}s)")


if __name__ == "__main__":
    main()
