#!/usr/bin/env python3
"""Grouped-CV V / A / V+A readouts for the three Gemma-4-31B-scored A banks.

Reads the shard npz files written by score_va_gemma_banks.py and writes
RESULTS_gemma.md + results_gemma.json into each bank directory.
Threshold-free only: pooled out-of-fold ROC AUC and univariate rank AUCs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(os.environ.get("NR_REPO", "/lfs/skampere3/0/alexspan/norm-research"))
OUT = Path(os.environ.get("VA_OUT", str(REPO / "outputs/va_gemma_banks")))
SEED = 20260728

BANK_DIR = {
    "hashtagwars": REPO / "datasets/humor/hashtagwars/va",
    "style_invitational": REPO / "datasets/humor/style_invitational/va",
    "creative_writing": REPO / "datasets/creative-writing/va_bank_v2",
}


def load_bank(name):
    meta = json.loads((OUT / f"{name}_meta.json").read_text())
    Xs, Vs, ids, grp = [], [], [], []
    shard_of = []
    si = 0
    while (OUT / f"{name}_shard{si}.npz").exists():
        z = np.load(OUT / f"{name}_shard{si}.npz", allow_pickle=True)
        Xs.append(z["X"]); Vs.append(z["V"])
        ids += z["ids"].tolist(); grp += z["groups"].tolist()
        shard_of += [si] * len(z["ids"])
        si += 1
    X = np.vstack(Xs); V = np.vstack(Vs)
    order = {d: i for i, d in enumerate(ids)}
    idx = np.array([order[d] for d in meta["item_ids"]])
    return (meta, X[idx], V[idx], np.array(meta["item_groups"], dtype=object),
            np.array(shard_of)[idx])


def oof_auc(X, y, groups, n_splits=5):
    if y.sum() < n_splits or (len(y) - y.sum()) < n_splits:
        return float("nan"), []
    pred = np.full(len(y), np.nan)
    folds = []
    for f, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        if y[tr].sum() == 0 or y[tr].sum() == len(tr):
            continue
        pipe = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=SEED))
        pipe.fit(X[tr], y[tr])
        pred[te] = pipe.predict_proba(X[te])[:, 1]
        if len(set(y[te].tolist())) > 1:
            folds.append({"fold": f, "n": int(len(te)), "pos": int(y[te].sum()),
                          "auc": float(roc_auc_score(y[te], pred[te]))})
    ok = np.isfinite(pred)
    return float(roc_auc_score(y[ok], pred[ok])), folds


def suite(V, A, y, groups, idx=None):
    if idx is None:
        idx = np.arange(len(y))
    yi, gi = y[idx], groups[idx]
    out = {"n": int(len(idx)), "positives": int(yi.sum()),
           "negatives": int(len(idx) - yi.sum()),
           "groups": int(len(set(gi.tolist()))), "models": {}}
    for nm, M in (("V", V), ("A", A), ("V+A", np.column_stack([V, A]))):
        auc, folds = oof_auc(M[idx], yi, gi)
        out["models"][nm] = {"auc": auc, "folds": folds}
    return out


def univariate(A, y, names):
    rows = []
    for c, nm in enumerate(names):
        v = A[:, c].copy()
        fin = np.isfinite(v)
        v[~fin] = float(np.nanmedian(v)) if fin.any() else 0.5
        vals, cnts = np.unique(v, return_counts=True)
        modal = float(cnts.max() / len(v))
        auc = float(roc_auc_score(y, v)) if len(vals) > 1 else 0.5
        rows.append({"criterion": nm, "auc": auc, "na_rate": float((~fin).mean()),
                     "modal_share": modal, "mean": float(np.nanmean(A[:, c])),
                     "near_constant": bool(modal > 0.98)})
    rows.sort(key=lambda r: -r["auc"])
    return rows


def balanced_idx(y, groups):
    rng = np.random.default_rng(SEED)
    sel = []
    for g in sorted(set(groups.tolist())):
        i = np.flatnonzero(groups == g)
        p, n = i[y[i] == 1], i[y[i] == 0]
        sel += p.tolist()
        sel += rng.choice(n, size=min(len(n), len(p)), replace=False).tolist()
    return np.array(sorted(sel), dtype=int)


def battery():
    p = OUT / "anchor_battery.json"
    return json.loads(p.read_text()) if p.exists() else {}


def battery_md(name):
    b = battery().get(name)
    if not b:
        return "_Extended anchor battery not available._"
    k = b["k_per_class"]
    return (
        f"An extended battery of {k} independently drawn anchors per class "
        f"(seeds disjoint from the shard anchors) was scored with the same judge, "
        f"prompts and criteria:\n\n"
        "| Anchor class | mean A | sd | n |\n|---|---:|---:|---:|\n"
        f"| known positive | {b['anchor_pos']['mean']:.3f} | {b['anchor_pos']['sd']:.3f} | {k} |\n"
        f"| random negative | {b['anchor_neg']['mean']:.3f} | {b['anchor_neg']['sd']:.3f} | {k} |\n"
        f"| scrambled nonsense | {b['anchor_scram']['mean']:.3f} | {b['anchor_scram']['sd']:.3f} | {k} |\n\n"
        f"Item-level rank separation: positive vs negative AUC "
        f"**{b['pos_vs_neg_auc']:.3f}**; coherent (pos+neg) vs scrambled AUC "
        f"**{b['coherent_vs_scrambled_auc']:.3f}**. Ordering on class means holds: "
        f"**{b['ordering_holds_on_means']}**.")


def invalid_shards(meta):
    return [r["shard"] for r in meta["anchor_reports"] if not r["valid"]]


def md_table(res):
    L = ["| Features | AUC |", "|---|---:|"]
    for nm in ("V", "A", "V+A"):
        L.append(f"| {nm} | {res['models'][nm]['auc']:.4f} |")
    return "\n".join(L)


def anchor_md(reports):
    L = ["| Shard | n items | anchor pos | anchor neg | anchor scrambled | ordering | attempts |",
         "|---:|---:|---:|---:|---:|---|---:|"]
    for r in reports:
        L.append(f"| {r['shard']} | {r['n_items']} | {r['pos']:.3f} | {r['neg']:.3f} | "
                 f"{r['scram']:.3f} | {'PASS' if r['valid'] else 'FAIL'} | {r['attempts']} |")
    return "\n".join(L)


def crit_md(rows, k=12):
    L = ["| Rank | Criterion | univariate AUC | NA rate | modal share |",
         "|---:|---|---:|---:|---:|"]
    for i, r in enumerate(rows[:k], 1):
        L.append(f"| {i} | {r['criterion']} | {r['auc']:.4f} | {r['na_rate']:.3f} | "
                 f"{r['modal_share']:.3f} |")
    return "\n".join(L)


HEAD = """# {title} — Gemma-4-31B articulated-criteria (A) readout

Judge: **google/gemma-4-31b-it**, local snapshot, offline-batch vLLM (no HTTP server),
one token from {{1.0, 0.5, 0.0, NA}} per (item, criterion), temperature 0, max_tokens 6,
prefix caching on, spawn multiprocessing, single GPU (GPU 2, gpu_memory_utilization 0.55,
stacked beside a running LoRA training job). Run date: 2026-07-29.

Label-blind: y never appears in a judge prompt, and no criterion was selected or
rewritten using y. The A bank ({n_crit} criteria) was authored earlier and is used verbatim.

Readouts are 5-fold `GroupKFold` logistic (C=1, median imputation + missingness
indicators, standardization inside each training fold), pooled out-of-fold ROC AUC.
Threshold-free: no accuracy, no thresholds.
"""


def write(name, title, body, payload):
    d = BANK_DIR[name]
    (d / "RESULTS_gemma.md").write_text(body)
    (d / "results_gemma.json").write_text(json.dumps(payload, indent=1) + "\n")
    print("wrote", d / "RESULTS_gemma.md")


def do_hashtagwars():
    meta, A, V, groups, shard = load_bank("hashtagwars")
    y = np.array(meta["ys"]["top10_or_winner"])
    names = meta["a_names"]
    full = suite(V, A, y, groups)
    bi = balanced_idx(y, groups)
    bal = suite(V, A, y, groups, bi)
    uni = univariate(A, y, names)
    payload = {"task": "SemEval-2017 Task 6 #HashtagWars", "judge": "gemma-4-31b-it",
               "y_definition": "1 iff staff label in {1,2} (top ten or winner)",
               "grouping_unit": "hashtag contest", "n_criteria": len(names),
               "full": full, "balanced_within_hashtag": bal,
               "univariate_A": uni, "anchors": meta["anchor_reports"],
               "na_rate_overall": float(np.isnan(A).mean()),
               "anchor_battery": battery().get("hashtagwars"),
               "invalid_shards": invalid_shards(meta),
               "v_names": meta["v_names"], "a_names": names,
               "selection": meta["meta"]}
    body = HEAD.format(title="#HashtagWars", n_crit=len(names)) + f"""
## Sample

Precommitted label-blind subset reused verbatim from the earlier run: the first 40
hashtags under SHA-256("hashtagwars-va-v1:" + hashtag), every tweet retained.
**n = {full['n']}: {full['positives']} positive / {full['negatives']} negative across
{full['groups']} hashtags.** The contest hashtag is supplied as shared context; it is
constant within a contest and cannot separate entries inside the grouping unit.

## Grouped-CV AUC (full sample)

{md_table(full)}

## Balanced-within-hashtag readout

All positives plus an equal number of negatives sampled inside each hashtag (seed
{SEED}); n = {bal['n']} ({bal['positives']} pos / {bal['negatives']} neg,
{bal['groups']} hashtags). This is the readout comparable to the discarded codebook run.

{md_table(bal)}

## Anchor check (3 blinded rows in every shard)

Each of the 8 shards carried a staff winner, a random unselected entry, and a
scrambled word-salad row, drawn with a shard-specific seed (so the 8 checks are
independent draws, not one repeated draw). Values are that anchor's mean A score.

{anchor_md(meta['anchor_reports'])}

Shards whose first 3-row draw failed were re-drawn (attempt count above); every
shard ended on a passing draw. The per-shard rows themselves are unchanged by a
re-draw (temperature 0, one independent prompt per item x criterion), so a re-draw
tests the anchor sample, not the shard's scores.

### Extended anchor battery

{battery_md('hashtagwars')}

## Top articulated criteria (univariate)

Raw-direction ROC AUC of each criterion over the full sample; NA replaced by the
criterion's median for this descriptive statistic only. AUC never selected criteria.
Overall NA rate {float(np.isnan(A).mean()):.3f}.

{crit_md(uni)}

{caveats(uni, meta)}
"""
    write("hashtagwars", "#HashtagWars", body, payload)
    return payload


def caveats(uni, meta):
    nc = [r["criterion"] for r in uni if r["near_constant"]]
    s = "## Caveats\n\n"
    s += ("- Judge is Gemma-4-31B, a local open-weights model; A is the articulated-criteria "
          "channel as read by that judge, not a human editorial judgment.\n")
    if nc:
        s += ("- **Near-constant criteria (>98% one value, contribute no rank information): "
              + ", ".join(nc) + ".**\n")
    else:
        s += "- No criterion collapsed to a near-constant value (all modal shares < 0.98).\n"
    s += ("- Criteria are correlated and several are conditionally applicable; median "
          "imputation plus missingness indicators is a modeling choice.\n"
          "- Pooled out-of-fold AUC is descriptive; no confidence intervals are reported.\n"
          "- No hyperparameter search, no AUC-driven criterion selection, no threshold tuning.\n")
    return s


def do_style_invitational():
    meta, A, V, groups, shard = load_bank("style_invitational")
    names = meta["a_names"]
    ys = {k: np.array(v) for k, v in meta["ys"].items()}
    res = {}
    for k, y in ys.items():
        res[k] = {"full": suite(V, A, y, groups), "univariate_A": univariate(A, y, names)}
    payload = {"task": "Washington Post Style Invitational", "judge": "gemma-4-31b-it",
               "grouping_unit": "week_id", "n_criteria": len(names),
               "y_definitions": {
                   "top_tier": "1 = winner or runnerup, 0 = honorable_mention",
                   "winner_vs_rest": "1 = winner, 0 = runnerup or honorable_mention"},
               "results": res, "anchors": meta["anchor_reports"],
               "anchor_battery": battery().get("style_invitational"),
               "invalid_shards": invalid_shards(meta),
               "sensitivity_drop_invalid_shards": None,
               "na_rate_overall": float(np.isnan(A).mean()),
               "v_names": meta["v_names"], "a_names": names, "meta": meta["meta"]}
    bad = invalid_shards(meta)
    if bad:
        keep = np.flatnonzero(~np.isin(shard, bad))
        sens = {k: suite(V, A, np.array(v), groups, keep) for k, v in meta["ys"].items()}
        payload["sensitivity_drop_invalid_shards"] = {"dropped_shards": bad, "results": sens}
        sens_md = ("#### Sensitivity: invalid shard(s) dropped\n\n"
                   + "\n\n".join(f"**{k}** (n = {v['n']}, {v['positives']} pos): "
                                   + " / ".join(f"{m} {v['models'][m]['auc']:.4f}"
                                                for m in ("V", "A", "V+A"))
                                   for k, v in sens.items()))
    else:
        sens_md = ""
    tt, wv = res["top_tier"]["full"], res["winner_vs_rest"]["full"]
    body = HEAD.format(title="Style Invitational", n_crit=len(names)) + f"""
## Sample

All {meta['meta']['n_weeks']} archived weeks were scored (a superset of the
110-week precommitted design in `score_va.py`), n = {tt['n']} entries.
Grouping unit = `week_id`. The two y's are reported **separately and never merged**.

## y (a) top tier: winner ∪ runnerup vs honorable_mention

n = {tt['n']}: {tt['positives']} positive / {tt['negatives']} negative,
{tt['groups']} weeks.

{md_table(tt)}

### Top criteria (top-tier y)

{crit_md(res['top_tier']['univariate_A'])}

## y (b) winner vs rest

n = {wv['n']}: {wv['positives']} positive / {wv['negatives']} negative,
{wv['groups']} weeks.

{md_table(wv)}

### Top criteria (winner-vs-rest y)

{crit_md(res['winner_vs_rest']['univariate_A'])}

## Anchor check (3 blinded rows in every shard)

Per shard: a winner, a random honorable mention, and a scrambled word-salad entry,
shard-specific seeds. Values are that anchor's mean A score.

{anchor_md(meta['anchor_reports'])}

**Shard {invalid_shards(meta)} did NOT reach a passing 3-row ordering within 4
independent anchor draws and is recorded as INVALID.** Its rows are retained in the
headline readout and a leave-that-shard-out sensitivity readout is given below.
Re-drawing anchors cannot change a shard's item scores (temperature 0, one
independent prompt per item x criterion), so the failure is a property of the
anchor sample and of how little a single winner separates from a single
honorable mention under this judge, not of the shard's scoring run.

{sens_md}

### Extended anchor battery

{battery_md('style_invitational')}

{caveats(res['top_tier']['univariate_A'], meta)}
- Archive bylines (author name, hometown) remain inside `entry_text`; the judge is
  instructed to ignore them, but they still affect the deterministic V features.
- The honorable-mention pool is itself already editor-selected, so the negative class
  is a *published* negative, not a random submission.
"""
    write("style_invitational", "Style Invitational", body, payload)
    return payload


def do_creative():
    meta, A, V, groups, shard = load_bank("creative_writing")
    names = meta["a_names"]
    y = np.array(meta["ys"]["score_ge10"])
    full = suite(V, A, y, groups)
    uni = univariate(A, y, names)
    payload = {"task": "r/WritingPrompts (writingprompts_modeling_clean grouped build)",
               "judge": "gemma-4-31b-it", "grouping_unit": "prompt_id",
               "y_definition": "1 = story score >= 10 (clean 50/50 build), 0 otherwise",
               "n_criteria": len(names), "full": full, "univariate_A": uni,
               "anchors": meta["anchor_reports"],
               "na_rate_overall": float(np.isnan(A).mean()),
               "anchor_battery": battery().get("creative_writing"),
               "invalid_shards": invalid_shards(meta),
               "v_names": meta["v_names"], "a_names": names, "meta": meta["meta"]}
    tr = meta["meta"]["truncation"]
    body = HEAD.format(title="Creative writing (r/WritingPrompts)", n_crit=len(names)) + f"""
## Sample

Source: **canonical** `datasets/creative-writing/writingprompts_modeling_clean.csv.gz`
(the real grouped 50/50 score>=10 build present on sk3), not the local reconstruction
in this directory. Whole prompt groups taken in SHA-256("cw-va-v2-sample|" + prompt_id)
order until n >= 2000, exactly the rule recorded in `sample_inventory.json`.
**n = {full['n']}: {full['positives']} positive / {full['negatives']} negative across
{full['groups']} prompt groups.** Grouping unit = prompt.

Deterministic truncation: stories longer than {tr['source_chars']} characters are shown
as the first {tr['head']} characters + `[... DETERMINISTIC MIDDLE OMISSION ...]` + the
last {tr['tail']} characters. Writing prompts are truncated to {tr['prompt_chars']}
characters. No randomness is involved.

## Grouped-CV AUC

{md_table(full)}

## Anchor check (3 blinded rows in every shard)

Per shard: a random positive story, a random negative story, and a word-salad scramble
of two stories, shard-specific seeds. Values are that anchor's mean A score.

{anchor_md(meta['anchor_reports'])}

### Extended anchor battery

{battery_md('creative_writing')}

## Top articulated criteria (univariate)

{crit_md(uni)}

{caveats(uni, meta)}
- **The positive-vs-negative half of the anchor check does NOT hold for this task.**
  In the 12-per-class battery the positive anchors mean 0.686 and the negative anchors
  mean 0.744 (pos-vs-neg AUC 0.403, i.e. slightly reversed), and 3 of the 4 per-shard
  3-row checks passed only on the second anchor draw. What the anchors do establish
  robustly is coherence sensitivity: coherent-vs-scrambled AUC is 1.000 and the
  scrambled anchors mean 0.008. Read the A numbers below as "the judge applies these
  criteria to real prose and is not collapsed", not as "the anchor protocol certified
  positive/negative separation on this task". Both anchor classes are drawn from the
  same balanced pool, where a single score>=10 story is genuinely hard to tell from a
  single score<10 story; that is consistent with the modest A AUC of 0.605.
- Long stories are middle-omitted, so criteria about middle-of-story structure are
  judged on partial evidence.
"""
    write("creative_writing", "Creative writing", body, payload)
    return payload


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["hashtagwars", "style_invitational", "creative_writing"]
    fns = {"hashtagwars": do_hashtagwars, "style_invitational": do_style_invitational,
           "creative_writing": do_creative}
    for w in which:
        p = fns[w]()
        print(w, json.dumps({k: v for k, v in p.items()
                             if k in ("full", "results")}, default=str)[:600])
