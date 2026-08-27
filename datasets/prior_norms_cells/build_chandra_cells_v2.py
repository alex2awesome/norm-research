#!/usr/bin/env python3
"""Chandrasekharan pooled removal cells — GATED BUILD v2 (2026-08-24).
Same discipline as v1 build (matched renderer both classes, %0A decode, kept rows
text-matching any removal EXCLUDED, fragment drop, 1:1 within-subreddit, artifact
probes) with the leak-audit-driven changes:
  1. kept side = kept_v2_<sub>.jsonl.gz (era-UNIFORM 22-strata collection over
     2016-05-01..2017-03-31; v1 covered only the last 1-6 days per sub).
  2. mod/AutoModerator removal-notice rows stripped from BOTH classes
     (strict pattern; audit found 0.7-1.7% contamination each side).
  3. per-row kept created_utc recorded in the population (ts column; removed side
     has no timestamps — log carries body+subreddit only) for era-stratified readouts.
  4. per-row kept author recorded as author_hash (sha1[:16] of Arctic Shift author;
     [deleted] -> NA) for author-disjoint splits / author-grouped folds. AUTHOR
     CHANNEL UNTESTABLE ON REMOVED SIDE — Chandrasekharan corpus is anonymized
     (removal log = body+subreddit; released macro-norm CSVs are bare text).
Outputs chandra_{humor,cw}_v2_* ALONGSIDE v1 (v1 untouched). Probes reported in
BOTH configs: v1 gate config (char_wb 2-4, 30k, 70/30 within-sub) for the direct
v1 .806/.834 comparison, and audit config (word 1-2 / char_wb 3-5, 50k, 5-fold)."""
import gzip, hashlib, json, sys, zlib
from pathlib import Path
import numpy as np
import pandas as pd

NR = Path("/lfs/skampere3/0/alexspan/norm-research")
sys.path.insert(0, str(NR / "datasets/humor/reddit_jokes"))
from build_removal_v2_normalized import norm as _norm
import re as _re

def norm(t):
    return _norm(_re.sub(r"%0[aAdD]", " ", str(t)))

NOTICE = _re.compile(
    r"(?i)(?:has been (?:temporarily )?removed|breaks the rules of|"
    r"your (?:post|submission|comment|story) (?:has|was)|message the mods|"
    r"thank you for submitting to|following formatting issues|violates our rules|"
    r"i am a bot|this action was performed automatically)")

HUMOR = ["funny", "Showerthoughts", "tifu", "nottheonion", "me_irl"]
CW = ["nosleep", "books", "gameofthrones", "asoiaf"]
D = NR / "datasets/prior_norms_cells"

print("loading removal log...", flush=True)
rl = pd.read_csv(NR / "datasets/prior_norms/reddit-removal-log.csv")
rl = rl[rl.subreddit.isin(HUMOR + CW)].copy()
rl["text"] = rl.body.astype(str).map(norm)
rl = rl[rl.text.str.len() >= 25]
n0 = len(rl)
rl = rl[~rl.text.str.contains(NOTICE)]
print(f"removals after render+len: {n0}; after notice-strip: {len(rl)} (-{n0-len(rl)})", flush=True)
rem_hashes = set(hashlib.sha1(t.encode()).hexdigest() for t in rl.text)

kept_rows = []
notice_kept = 0
for sub in HUMOR + CW:
    f = D / f"kept_v2_{sub}.jsonl.gz"
    n_x = 0
    fh = gzip.open(f, "rt")
    while True:
        try:
            line = fh.readline()
        except (EOFError, OSError, zlib.error):
            print(f"[{sub}] torn gzip tail tolerated", flush=True)
            break
        if not line:
            break
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = norm(str(r["body"]))
        if len(t) < 25:
            continue
        if NOTICE.search(t):
            notice_kept += 1
            continue
        if hashlib.sha1(t.encode()).hexdigest() in rem_hashes:
            n_x += 1
            continue
        a = r.get("author")
        ah = (hashlib.sha1(a.encode()).hexdigest()[:16]
              if a and a != "[deleted]" else None)
        kept_rows.append({"subreddit": sub, "text": t, "id": r.get("id"),
                          "ts": r.get("created_utc"), "author_hash": ah})
    print(f"[{sub}] kept_v2 loaded; removal-matches excluded: {n_x}", flush=True)
print(f"kept notice-strip total: {notice_kept}", flush=True)
kp = pd.DataFrame(kept_rows).drop_duplicates(subset=["text"])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

def within_sub_v1cfg(df, subs):
    """v1 gate probe: char_wb 2-4, 30k, min_df 5; per-sub 70/30 split, seed 3."""
    ws = {}
    for sub in subs:
        m = df[df.group == sub]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=30000, min_df=5)
        Xi = vec.fit_transform(m.text)
        yi = m.judgement.values
        cut = int(.7 * Xi.shape[0])
        idx = np.random.default_rng(3).permutation(Xi.shape[0])
        clf = LogisticRegression(max_iter=1500).fit(Xi[idx[:cut]], yi[idx[:cut]])
        ws[sub] = roc_auc_score(yi[idx[cut:]], clf.predict_proba(Xi[idx[cut:]])[:, 1])
    return ws

def within_sub_5fold(df, subs, vf):
    per, ns = {}, {}
    for sub in subs:
        m = df[df.group == sub]
        if len(m) < 60:
            continue
        X = vf().fit_transform(m.text)
        yi = m.judgement.values
        oof = np.zeros(len(yi))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=3).split(X, yi):
            clf = LogisticRegression(max_iter=2000).fit(X[tr], yi[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        per[sub] = roc_auc_score(yi, oof)
        ns[sub] = len(yi)
    w = np.array([ns[s] for s in per]); a = np.array([per[s] for s in per])
    return float((w * a).sum() / w.sum()), per

W12 = lambda: TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=50000, min_df=5, sublinear_tf=True)
C35 = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=50000, min_df=5, sublinear_tf=True)

for name, subs in (("chandra_humor_v2", HUMOR), ("chandra_cw_v2", CW)):
    parts = []
    for sub in subs:
        r_s = rl[rl.subreddit == sub]
        k_s = kp[kp.subreddit == sub]
        n = min(len(r_s), len(k_s))
        r_i = r_s.sample(n=n, random_state=7) if len(r_s) > n else r_s
        k_i = k_s.sample(n=n, random_state=7) if len(k_s) > n else k_s
        parts.append(pd.DataFrame({"text": r_i.text.values, "judgement": 1, "group": sub,
                                   "ts": np.nan, "author_hash": None}))
        parts.append(pd.DataFrame({"text": k_i.text.values, "judgement": 0, "group": sub,
                                   "ts": k_i.ts.values, "author_hash": k_i.author_hash.values}))
    df = pd.concat(parts).reset_index(drop=True)
    df["row_id"] = [f"{name}:{i}" for i in range(len(df))]
    y = df.judgement.values
    g = df.group.values
    # grouped-OOF (v1 config) for continuity
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=30000, min_df=5)
    X = vec.fit_transform(df.text)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(min(5, len(subs))).split(X, groups=g):
        clf = LogisticRegression(max_iter=1500)
        clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
    probe = roc_auc_score(y, oof)
    ws_v1 = within_sub_v1cfg(df, subs)
    w_auc, w_per = within_sub_5fold(df, subs, W12)
    c_auc, c_per = within_sub_5fold(df, subs, C35)
    ts = pd.to_numeric(df[df.judgement == 0].ts, errors="coerce").dropna()
    df.to_csv(D / f"{name}_population.csv.gz", index=False, compression="gzip")
    man = {"cell": name, "n": len(df), "pos_rate": float(y.mean()),
           "per_sub": df.groupby("group").judgement.agg(["count", "mean"]).round(3).to_dict(),
           "kept_ts_coverage": {"frac_with_ts": round(float(df[df.judgement == 0].ts.notna().mean()), 4),
                                "min": str(pd.to_datetime(ts.min(), unit="s").date()),
                                "med": str(pd.to_datetime(ts.median(), unit="s").date()),
                                "max": str(pd.to_datetime(ts.max(), unit="s").date()),
                                "distinct_days": int(pd.to_datetime(ts, unit="s").dt.date.nunique())},
           "artifact_probe_charngram_groupedOOF_v1cfg": round(float(probe), 4),
           "artifact_probe_within_sub_mean_v1cfg": round(float(np.mean(list(ws_v1.values()))), 4),
           "artifact_probe_within_sub_per_sub_v1cfg": {k: round(v, 4) for k, v in ws_v1.items()},
           "probe_within_sub_word12_5fold": {"mean": round(w_auc, 4), "per_sub": {k: round(v, 4) for k, v in w_per.items()}},
           "probe_within_sub_char35_5fold": {"mean": round(c_auc, 4), "per_sub": {k: round(v, 4) for k, v in c_per.items()}},
           "v1_comparison": {"within_sub_v1cfg_v1": .806 if "humor" in name else .8337,
                             "groupedOOF_v1": .6524 if "humor" in name else .6052},
           "author_channel": {
               "kept_frac_with_author": round(float(df[df.judgement == 0].author_hash.notna().mean()), 4),
               "kept_unique_authors": int(df[df.judgement == 0].author_hash.nunique()),
               "removed": "author channel untestable — removal corpus anonymized "
                          "(reddit-removal-log.csv = body+subreddit only; released "
                          "macro-norm CSVs are bare text; no author or hash)"},
           "changes_vs_v1": ["era-uniform kept collection (22 strata)",
                             "mod/AutoMod notice rows stripped BOTH classes",
                             "kept created_utc recorded (ts col)",
                             "kept author_hash recorded (sha1[:16]); removed side N/A"]}
    (D / f"{name}_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(f"[{name}] n={len(df)} groupedOOF(v1cfg) {probe:.4f} within-sub(v1cfg) {np.mean(list(ws_v1.values())):.4f} "
          f"word12 {w_auc:.4f} char35 {c_auc:.4f}", flush=True)
    print(f"  per-sub v1cfg: { {k: round(v,3) for k,v in ws_v1.items()} }", flush=True)
print("CHANDRA_BUILD_V2_DONE")
