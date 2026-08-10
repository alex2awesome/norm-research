#!/usr/bin/env python
"""Build the Reddit lay-crowd newsworthiness dataset (news-C cell).

Unit: a link post to r/news or r/worldnews = one headline judged by the crowd.
X = headline title (presentation-normalized; raw_title kept).
y = score percentile within (subreddit x posting-day UTC) cell:
    top quartile (pct >= 0.75) -> 1, bottom quartile (pct <= 0.25) -> 0,
    middle dropped.  Percentile is average-rank based, so the huge score==1
    tie block (never-voted / removed posts) falls in the middle and is
    dropped automatically; negatives are therefore actively-downvoted posts.

Deconfounding (mandated; mirrors math/stackexchange/build_v3_position_matched.py
philosophy of matching label classes on structural confounds): fit a
text-free propensity model P(y=1 | hour, weekday, post age, domain, author
post count, subreddit), bin into deciles, and exactly balance pos/neg within
each decile by downsampling the majority class.

Splits: stable hash of post id -> md5 % 10; 0-7 train, 8 eval, 9 test.

Outputs (under --out-dir):
  built/train.csv.gz, built/eval.csv.gz, built/test.csv.gz
  manifest.json (all stage counts, audits, TF-IDF floor results)
"""

import argparse
import gzip
import hashlib
import html
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

REDDIT_INTERNAL = {"reddit.com", "redd.it", "i.redd.it", "v.redd.it",
                   "old.reddit.com", "np.reddit.com", "new.reddit.com"}

CHAR_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-",
    "…": "...", " ": " ",
}

TRAILING_SEP_RE = re.compile(r"\s[|\-–—]\s")


def normalize_title(t):
    """Presentation normalization: NFKC, curly->straight quotes, dashes,
    ellipsis, collapse whitespace."""
    t = html.unescape(html.unescape(t))  # twice: dumps contain &amp;#x27; double-escapes
    t = unicodedata.normalize("NFKC", t)
    for k, v in CHAR_MAP.items():
        t = t.replace(k, v)
    return re.sub(r"\s+", " ", t).strip()


def title_key(t):
    """Near-dup clustering key: lowercase alnum tokens of normalized title."""
    return " ".join(re.findall(r"[a-z0-9]+", normalize_title(t).lower()))


def domain_of(u):
    host = urlparse(u.strip()).netloc.lower()
    return re.sub(r"^(www\.|m\.|amp\.)", "", host)


def url_key(u):
    pr = urlparse(u.strip())
    host = re.sub(r"^(www\.|m\.|amp\.)", "", pr.netloc.lower())
    return host + pr.path.rstrip("/")


def trailing_segment(t):
    """Return (segment, head) for the last ' | ' / ' - ' style separator if
    the trailing segment is short (<=6 words); else (None, t)."""
    ms = list(TRAILING_SEP_RE.finditer(t))
    if not ms:
        return None, t
    last = ms[-1]
    seg = t[last.end():].strip()
    head = t[:last.start()].strip()
    if seg and len(seg.split()) <= 6 and head:
        return seg, head
    return None, t


def build_boiler_set(titles, domains, min_count=30, top3_share=0.6):
    """Outlet-boilerplate trailing segments: frequent (>=min_count) AND
    concentrated in few domains (top-3 domains cover >=top3_share of the
    segment's occurrences).  The concentration guard keeps cross-outlet
    headline conventions (e.g. trailing 'report') out of the strip list."""
    seg_count = Counter()
    seg_dom = defaultdict(Counter)
    for t, d in zip(titles, domains):
        seg, _ = trailing_segment(t)
        if seg:
            k = seg.lower()
            seg_count[k] += 1
            seg_dom[k][d] += 1
    boiler = {}
    for k, c in seg_count.items():
        if c < min_count:
            continue
        dc = seg_dom[k]
        share = sum(n for _, n in dc.most_common(3)) / c
        if share >= top3_share:
            boiler[k] = (c, round(share, 3), dc.most_common(1)[0][0])
    return boiler


def strip_boiler(t, boiler, max_iter=3):
    for _ in range(max_iter):
        seg, head = trailing_segment(t)
        if seg is None or seg.lower() not in boiler:
            return t
        t = head
    return t


def split_of(post_id):
    h = int(hashlib.md5(post_id.encode()).hexdigest(), 16) % 10
    return "train" if h <= 7 else ("eval" if h == 8 else "test")


def tfidf_floor(df, text_col, label):
    """TF-IDF + LR floor: train on train split, AUC on eval/test, top feats."""
    tr = df[df.split == "train"]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000,
                          sublinear_tf=True, lowercase=True)
    Xtr = vec.fit_transform(tr[text_col])
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(Xtr, tr.judgement.values)
    out = {"label": label}
    for sp in ("eval", "test"):
        d = df[df.split == sp]
        out[f"auc_{sp}"] = round(roc_auc_score(
            d.judgement.values, lr.predict_proba(vec.transform(d[text_col]))[:, 1]), 4)
    te = df[df.split == "test"]
    for sub in sorted(df.subreddit.unique()):
        d = te[te.subreddit == sub]
        out[f"auc_test_{sub}"] = round(roc_auc_score(
            d.judgement.values, lr.predict_proba(vec.transform(d[text_col]))[:, 1]), 4)
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(lr.coef_[0])
    out["top20_neg"] = [(names[i], round(float(lr.coef_[0][i]), 3)) for i in order[:20]]
    out["top20_pos"] = [(names[i], round(float(lr.coef_[0][i]), 3)) for i in order[-20:][::-1]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", type=Path,
                    default=Path("/lfs/skampere3/0/alexspan/norm-research/datasets/taxonomy_dumps"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/lfs/skampere3/0/alexspan/norm-research/datasets/news-homepages/reddit_newsworthiness"))
    ap.add_argument("--min-cell", type=int, default=20)
    ap.add_argument("--maturity-days", type=float, default=3.0,
                    help="drop posts younger than this at dump time (scores not mature)")
    ap.add_argument("--dedup-window-days", type=float, default=3.0)
    ap.add_argument("--top-domains", type=int, default=100,
                    help="number of domain one-hots in the propensity model")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    manifest = {"build_date": datetime.now().isoformat(timespec="seconds"),
                "args": {k: str(v) for k, v in vars(args).items()}}
    stages = {}

    # ---------- load ----------
    rows = []
    author_count = Counter()
    n_meta = n_2nd = 0
    for fn in ("news_posts.jsonl", "worldnews_posts.jsonl"):
        with open(args.dump_dir / fn) as f:
            for line in f:
                d = json.loads(line)
                author_count[d.get("author")] += 1
                if "_meta" in d:
                    n_meta += 1
                    if "retrieved_2nd_on" in (d["_meta"] or {}):
                        n_2nd += 1
                rows.append(d)
    stages["raw"] = len(rows)
    manifest["meta_coverage"] = {"rows_with_meta": n_meta, "rows_with_retrieved_2nd_on": n_2nd}
    print(f"raw rows: {len(rows)};  _meta: {n_meta};  retrieved_2nd_on: {n_2nd}")

    max_ts = max(r["created_utc"] for r in rows)
    manifest["dump_max_created_utc"] = datetime.fromtimestamp(max_ts, tz=timezone.utc).isoformat()

    # ---------- basic filters ----------
    kept, drop = [], Counter()
    for r in rows:
        t = (r.get("title") or "").strip()
        if t in ("", "[removed]", "[deleted]"):
            drop["bad_title"] += 1; continue
        if r.get("over_18"):
            drop["over_18"] += 1; continue
        if (r.get("selftext") or "").strip():
            drop["self_post_nonempty_selftext"] += 1; continue
        u = (r.get("url") or "").strip()
        if not u:
            drop["no_url"] += 1; continue
        dom = domain_of(u)
        if dom in REDDIT_INTERNAL or not dom:
            drop["reddit_internal_url"] += 1; continue
        if max_ts - r["created_utc"] < args.maturity_days * 86400:
            drop["too_recent_immature_score"] += 1; continue
        r["domain"] = dom
        kept.append(r)
    stages["after_basic_filters"] = len(kept)
    manifest["basic_filter_drops"] = dict(drop)
    print(f"after basic filters: {len(kept)}  drops: {dict(drop)}")

    # ---------- dedup: earliest per normalized-title / url cluster, 3-day window ----------
    kept.sort(key=lambda r: r["created_utc"])
    last_seen = {}
    deduped, ddrop = [], Counter()
    win = args.dedup_window_days * 86400
    for r in kept:
        tk = title_key(r["title"])
        uk = url_key(r["url"])
        keys = [("t", tk)] if tk else []
        keys.append(("u", uk))
        dup = any(k in last_seen and r["created_utc"] - last_seen[k] <= win for k in keys)
        for k in keys:           # extend chains: repeated reposting stays dead
            last_seen[k] = r["created_utc"]
        if dup:
            ddrop["dup_title_or_url_3d"] += 1
        else:
            deduped.append(r)
    stages["after_dedup"] = len(deduped)
    manifest["dedup_drops"] = dict(ddrop)
    print(f"after dedup: {len(deduped)}  ({dict(ddrop)})")

    # ---------- cells + percentile labels ----------
    df = pd.DataFrame(deduped)[["id", "subreddit", "title", "url", "domain",
                                "score", "num_comments", "created_utc",
                                "author", "link_flair_text"]]
    dt = pd.to_datetime(df.created_utc, unit="s", utc=True)
    df["day"] = dt.dt.date.astype(str)
    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.weekday
    df["age_days"] = (max_ts - df.created_utc) / 86400.0
    cell_sizes = df.groupby(["subreddit", "day"])["id"].transform("size")
    n_small = int((cell_sizes < args.min_cell).sum())
    df = df[cell_sizes >= args.min_cell].copy()
    stages["after_min_cell"] = len(df)
    print(f"dropped {n_small} rows in cells <{args.min_cell}; remaining {len(df)}")

    df["percentile"] = df.groupby(["subreddit", "day"])["score"].rank(pct=True, method="average")
    df["judgement"] = np.where(df.percentile >= 0.75, 1,
                               np.where(df.percentile <= 0.25, 0, -1))
    n_mid = int((df.judgement == -1).sum())
    df = df[df.judgement >= 0].copy()
    stages["middle_dropped"] = n_mid
    stages["labeled"] = len(df)
    pre_counts = df.groupby(["subreddit", "judgement"]).size().to_dict()
    manifest["label_counts_before_balancing"] = {f"{k[0]}_y{k[1]}": int(v) for k, v in pre_counts.items()}
    print(f"labeled: {len(df)}  (middle dropped: {n_mid})")
    print("label counts before balancing:", manifest["label_counts_before_balancing"])

    # ---------- presentation normalization ----------
    df["raw_title"] = df.title
    df["norm_title"] = df.title.map(normalize_title)

    # ---------- spot-check ----------
    print("\n===== SPOT-CHECK: 10 random POS =====")
    for _, r in df[df.judgement == 1].sample(10, random_state=args.seed).iterrows():
        print(f"  [{r.subreddit} {r.day} score={r.score} pct={r.percentile:.2f} "
              f"nc={r.num_comments} {r.domain}] {r.norm_title[:130]}")
    print("===== SPOT-CHECK: 10 random NEG =====")
    for _, r in df[df.judgement == 0].sample(10, random_state=args.seed).iterrows():
        print(f"  [{r.subreddit} {r.day} score={r.score} pct={r.percentile:.2f} "
              f"nc={r.num_comments} {r.domain}] {r.norm_title[:130]}")

    # ---------- domain stats ----------
    top20 = df.domain.value_counts().head(20)
    dom_stats = []
    for d, c in top20.items():
        sub = df[df.domain == d]
        dom_stats.append({"domain": d, "share": round(c / len(df), 4),
                          "p_y1": round(float(sub.judgement.mean()), 3), "n": int(c)})
    manifest["top20_domains"] = dom_stats
    print("\ntop-20 domains (share, P(y=1)):")
    for s in dom_stats:
        print(f"  {s['n']:6d}  share={s['share']:.3f}  P(y=1)={s['p_y1']:.3f}  {s['domain']}")

    # ---------- propensity model (confounds only, NO text) ----------
    top_doms = list(df.domain.value_counts().head(args.top_domains).index)
    dom_idx = {d: i for i, d in enumerate(top_doms)}
    n = len(df)
    F = np.zeros((n, 24 + 7 + len(top_doms) + 4), dtype=np.float32)
    F[np.arange(n), df.hour.values] = 1.0                       # hour one-hot
    F[np.arange(n), 24 + df.weekday.values] = 1.0               # weekday one-hot
    di = df.domain.map(lambda d: dom_idx.get(d, -1)).values
    has_dom = di >= 0
    F[np.where(has_dom)[0], 31 + di[has_dom]] = 1.0             # domain one-hot
    base = 31 + len(top_doms)
    F[:, base] = (~has_dom).astype(np.float32)                  # other-domain flag
    F[:, base + 1] = np.log1p(df.age_days.values)               # post age at retrieval
    F[:, base + 2] = np.log1p(df.author.map(author_count).fillna(0).values)  # author activity
    F[:, base + 3] = (df.subreddit == "worldnews").astype(np.float32)
    y = df.judgement.values

    lr = LogisticRegression(max_iter=2000, C=1.0)
    cv_proba = cross_val_predict(lr, F, y, method="predict_proba",
                                 cv=StratifiedKFold(5, shuffle=True, random_state=args.seed))[:, 1]
    prop_auc = roc_auc_score(y, cv_proba)
    manifest["propensity_auc_cv5"] = round(float(prop_auc), 4)
    print(f"\npropensity model 5-fold CV AUC (confounds only): {prop_auc:.4f}")

    lr.fit(F, y)
    feat_names = ([f"hour_{h}" for h in range(24)] + [f"wd_{w}" for w in range(7)]
                  + [f"dom_{d}" for d in top_doms]
                  + ["dom_other", "log_age_days", "log_author_posts", "is_worldnews"])
    coefs = sorted(zip(feat_names, lr.coef_[0]), key=lambda x: -abs(x[1]))[:15]
    manifest["propensity_top_coefs"] = [(f, round(float(c), 3)) for f, c in coefs]
    print("top propensity coefs:", manifest["propensity_top_coefs"][:10])
    df["propensity"] = lr.predict_proba(F)[:, 1]

    # ---------- decile balancing (exact pos/neg match within decile) ----------
    edges = np.unique(np.quantile(df.propensity.values, np.linspace(0, 1, 11)))
    df["pdecile"] = np.clip(np.searchsorted(edges[1:-1], df.propensity.values, side="right"),
                            0, len(edges) - 2)
    keep_idx = []
    for dec, g in df.groupby("pdecile"):
        pos = list(g.index[g.judgement == 1])
        neg = list(g.index[g.judgement == 0])
        m = min(len(pos), len(neg))
        keep_idx += rng.sample(pos, m) + rng.sample(neg, m)
    bal = df.loc[sorted(keep_idx)].copy()
    stages["after_decile_balancing"] = len(bal)
    post_counts = bal.groupby(["subreddit", "judgement"]).size().to_dict()
    manifest["label_counts_after_balancing"] = {f"{k[0]}_y{k[1]}": int(v) for k, v in post_counts.items()}
    print(f"\nafter decile balancing: {len(bal)}")
    print("label counts after balancing:", manifest["label_counts_after_balancing"])
    bal_auc = roc_auc_score(bal.judgement.values, bal.propensity.values)
    manifest["propensity_auc_after_balancing"] = round(float(bal_auc), 4)
    print(f"propensity AUC on balanced set (should be ~0.5 within-decile): {bal_auc:.4f}")

    # ---------- outlet boilerplate stripping ----------
    boiler = build_boiler_set(bal.norm_title.tolist(), bal.domain.tolist())
    manifest["boiler_segments"] = {k: v for k, v in sorted(boiler.items(), key=lambda x: -x[1][0])[:40]}
    print(f"\nboilerplate trailing segments to strip: {len(boiler)}; top 15:")
    for k, (c, share, d) in sorted(boiler.items(), key=lambda x: -x[1][0])[:15]:
        print(f"  {c:5d}  top3share={share:.2f}  ({d})  '{k}'")
    bal["text"] = bal.norm_title.map(lambda t: strip_boiler(t, boiler))
    n_stripped = int((bal.text != bal.norm_title).sum())
    manifest["titles_with_boiler_stripped"] = n_stripped
    print(f"titles changed by stripping: {n_stripped} / {len(bal)}")

    # ---------- splits ----------
    bal["split"] = bal.id.map(split_of)
    manifest["splits"] = {s: {"rows": int((bal.split == s).sum()),
                              "pos": int(bal[bal.split == s].judgement.sum())}
                          for s in ("train", "eval", "test")}
    print("\nsplits:", manifest["splits"])

    # ---------- TF-IDF/LR floor, before vs after stripping ----------
    print("\n===== TF-IDF floor (normalized title, BEFORE boiler strip) =====")
    res_a = tfidf_floor(bal, "norm_title", "before_strip")
    print(json.dumps(res_a, indent=1, default=str))
    print("\n===== TF-IDF floor (text, AFTER boiler strip) =====")
    res_b = tfidf_floor(bal, "text", "after_strip")
    print(json.dumps(res_b, indent=1, default=str))
    manifest["tfidf_floor_before_strip"] = res_a
    manifest["tfidf_floor_after_strip"] = res_b

    # ---------- length confound ----------
    tl = bal.text.str.len().values
    r_len = float(np.corrcoef(tl, bal.judgement.values)[0, 1])
    manifest["title_len_label_corr"] = round(r_len, 4)
    manifest["title_len_means"] = {"pos": round(float(tl[bal.judgement == 1].mean()), 1),
                                   "neg": round(float(tl[bal.judgement == 0].mean()), 1)}
    print(f"\ntitle length vs label corr: {r_len:.4f}  means: {manifest['title_len_means']}")

    # ---------- write ----------
    out_built = args.out_dir / "built"
    out_built.mkdir(parents=True, exist_ok=True)
    cols = ["text", "judgement", "raw_title", "score", "percentile", "num_comments",
            "subreddit", "domain", "post_id", "author", "created_utc",
            "link_flair_text", "propensity"]
    bal = bal.rename(columns={"id": "post_id"})
    for sp in ("train", "eval", "test"):
        d = bal[bal.split == sp][cols].sample(frac=1.0, random_state=args.seed)
        p = out_built / f"{sp}.csv.gz"
        d.to_csv(p, index=False, compression="gzip")
        print(f"wrote {len(d):6d} rows -> {p}")

    manifest["stage_counts"] = stages
    with open(args.out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nwrote manifest -> {args.out_dir / 'manifest.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
