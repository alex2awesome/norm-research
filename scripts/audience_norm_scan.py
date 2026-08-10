"""Scan existing aspects.json across tasks for audience-appeal-type norms.

For each task, categorize aspects via keyword matching on name+description into:
  - HOOK / OPENING / ATTENTION
  - VOICE / VOICE-IDENTITY / PERSONALITY
  - NOVELTY / ORIGINALITY / SURPRISE
  - SHARE / VIRAL / SHAREABILITY / MEMORABILITY
  - EMOTIONAL / RESONANCE / IMPACT
  - AUDIENCE / READER / READABILITY
  - PAYOFF / PUNCH / TWIST / CLIMAX
  - HUMOR_MECHANICS (only for humor task: setup, callback, timing, surprise)

For each match, report: aspect_id, name, applicability rate, lift delta, lift p,
and feature-importance rank under the RF (if among top 100).
"""
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
TASKS = [
    "peer_review", "math", "notice_and_comment", "press_releases",
    "humor", "news_homepages", "patents", "code_review", "creative_writing",
]
JUDGES = ["qwen_thinking_fp8", "claude"]
RANDOM_SEED = 42

CATEGORIES = {
    "hook/opening/attention": [
        r"\bhook\b", r"opening", r"attention", r"grab", r"first line",
        r"first paragraph", r"compelling start",
    ],
    "voice/personality": [
        r"\bvoice\b", r"narrative voice", r"authorial", r"persona",
        r"\bstyle\b.*identity", r"distinctive style", r"signature",
    ],
    "novelty/originality/surprise": [
        r"novel", r"original", r"\bfresh\b", r"surprise", r"unexpected",
        r"twist", r"subver", r"unconventional", r"innovative",
    ],
    "share/viral/memorable": [
        r"shareab", r"viral", r"memorab", r"sticky", r"contagious",
        r"go.viral", r"spread",
    ],
    "emotional/resonance/impact": [
        r"emotional", r"resonance", r"impact", r"poignan", r"moving",
        r"heart", r"empathy.*reader", r"engagement",
    ],
    "audience/reader-experience": [
        r"audience.*experien", r"reader.*experien", r"engag.*reader",
        r"reader.*engag", r"reader.*invest", r"reader.*reaction",
    ],
    "payoff/twist/climax": [
        r"payoff", r"punchline", r"\bpunch\b", r"climax", r"\btwist\b",
        r"reversal", r"reveal", r"\bturn\b", r"peripeteia",
    ],
    "humor_mechanics": [
        r"setup.*punch", r"callback", r"timing", r"incongru", r"absurd",
        r"\bsubvert", r"\bjoke\b.*structure", r"comedic timing",
    ],
}


def scan_aspects(task):
    p = REPO / f"runs/validity_full/v2/{task}/aspects.json"
    arr = json.loads(p.read_text())
    matches = {cat: [] for cat in CATEGORIES}
    for a in arr:
        text = (a.get("name", "") + " " + a.get("description", "")).lower()
        for cat, pats in CATEGORIES.items():
            for pat in pats:
                if re.search(pat, text):
                    matches[cat].append(a)
                    break
    return arr, matches


def task_rf_and_lift(task):
    """Train RF and compute per-aspect lift. Return: feat_importance dict,
    per-aspect delta & p, and a aspect→rank-in-importance map."""
    labels_p = REPO / f"runs/validity_full/v2/{task}/datapoints.json"
    dps_raw = json.loads(labels_p.read_text())
    labels = pd.Series(
        {d["datapoint_id"]: int(d["judgement"])
         for d in dps_raw if d.get("judgement") is not None}
    )
    score_dfs, appl_dfs = [], []
    for j in JUDGES:
        f = REPO / f"outputs/v2_db/cells_v1/task={task}/judge={j}/data.parquet"
        if not f.exists():
            continue
        c = pd.read_parquet(f)
        c["score_num"] = c["score"].where(c["applicable"], np.nan).astype(float)
        sc = (c.groupby(["datapoint_id", "aspect_id"])["score_num"]
                .mean().unstack("aspect_id"))
        ap = (c.groupby(["datapoint_id", "aspect_id"])["applicable"]
                .max().unstack("aspect_id").fillna(False).astype(int))
        score_dfs.append(sc); appl_dfs.append(ap)
    if not score_dfs:
        return None
    all_dps = sorted(set().union(*(df.index for df in score_dfs)))
    all_asp = sorted(set().union(*(df.columns for df in score_dfs)))

    def reidx(dfs, fill=np.nan):
        return [df.reindex(index=all_dps, columns=all_asp, fill_value=fill)
                for df in dfs]

    sc = reidx(score_dfs); ap = reidx(appl_dfs, fill=0)
    score = pd.concat([s.stack(dropna=False) for s in sc], axis=1).mean(
        axis=1, skipna=True).unstack().reindex(index=all_dps, columns=all_asp)
    appl = sum(ap); appl = (appl > 0).astype(int)
    common = sorted(set(score.index) & set(labels.index))
    if len(common) < 200:
        return None
    score = score.loc[common]; appl = appl.loc[common]
    y = labels.loc[common].values

    X = np.concatenate([
        score[all_asp].fillna(0).values,
        appl[all_asp].values.astype(float),
    ], axis=1)
    feat_names = ([f"score_{a}" for a in all_asp]
                  + [f"appl_{a}" for a in all_asp])

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED)
    rf = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED,
    )
    rf.fit(Xtr, ytr)
    imp = rf.feature_importances_

    # Aggregate per-aspect: importance of score_a + appl_a
    aspect_imp = {}
    for i, fname in enumerate(feat_names):
        a = fname.split("_", 1)[1]
        aspect_imp[a] = aspect_imp.get(a, 0.0) + float(imp[i])
    ranked = sorted(aspect_imp.items(), key=lambda x: -x[1])
    rank_of = {a: r for r, (a, _) in enumerate(ranked)}

    # per-aspect lift
    from scipy.stats import ttest_ind
    lifts = {}
    appls = {}
    for a in all_asp:
        col = score[a]; mask = col.notna()
        if mask.sum() < 10: continue
        s0 = col[mask][y[mask] == 0]; s1 = col[mask][y[mask] == 1]
        if len(s0) < 5 or len(s1) < 5: continue
        delta = s1.mean() - s0.mean()
        try: _, p = ttest_ind(s1, s0, equal_var=False)
        except: p = np.nan
        lifts[a] = (delta, p)
        appls[a] = mask.sum() / len(col)

    return {"aspect_imp": aspect_imp, "rank_of": rank_of,
            "lifts": lifts, "appls": appls, "n_aspects": len(all_asp)}


def main():
    print("=" * 92)
    print("AUDIENCE-APPEAL NORM SCAN — per task: count of matches per category,")
    print("then for each matched aspect: RF importance rank + lift delta")
    print("=" * 92)
    for task in TASKS:
        arr, matches = scan_aspects(task)
        n_total = len(arr)
        print(f"\n=== {task}  (total aspects: {n_total}) ===")
        # Pre-train RF for ranking
        meta = task_rf_and_lift(task)

        for cat, asps in matches.items():
            if not asps:
                continue
            print(f"\n  [{cat}]  {len(asps)} aspects matched:")
            # Sort by lift |delta| if meta available else by name
            decorated = []
            for a in asps:
                aid = a["aspect_id"]
                if meta:
                    delta, pval = meta["lifts"].get(aid, (None, None))
                    rank = meta["rank_of"].get(aid, None)
                    appl = meta["appls"].get(aid, None)
                else:
                    delta = pval = rank = appl = None
                decorated.append((a, delta, pval, rank, appl))
            decorated.sort(key=lambda x: (-(x[3] is None), x[3] if x[3] is not None else 999999))
            for a, delta, pval, rank, appl in decorated[:8]:
                d_s = f"delta={delta:+.3f}" if delta is not None else "delta=?    "
                p_s = f"p={pval:.2e}" if pval is not None else "p=?     "
                r_s = f"rank={rank+1:>3d}/{meta['n_aspects']}" if rank is not None and meta else "rank=?  "
                ap_s = f"appl={appl:.0%}" if appl is not None else "appl=?"
                desc = a.get('description', '')[:90]
                print(f"    {a['aspect_id']:<5} {r_s}  {d_s} {p_s}  {ap_s}")
                print(f"          [{a['name'][:70]}]")
                print(f"          {desc}")


if __name__ == "__main__":
    main()
