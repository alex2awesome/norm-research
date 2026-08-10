#!/usr/bin/env python3
"""
Build the BBC Most-Read dataset (news-C cell: lay crowd, REVEALED ATTENTION).

Input : raw/captures.jsonl  (from scrape_bbc_mostread.py)
Output: built/{train,eval,test}.csv.gz  +  manifest.json + build_log.txt

Label semantics
---------------
y = 1 : headline appeared on the BBC "Most Read" list for that capture (with rank).
y = 0 : same-day BBC headline NOT on that day's most-read list (control pool).
Both classes come from the SAME capture pipeline (Wayback HTML), so there is no
cross-source typography leak.  X = headline text (presentation-normalized; raw kept).

This is a LAY-CROWD, REVEALED signal (what readers actually clicked / read),
complementing the Reddit cell (lay crowd, revealed votes) and contrasting with
news_homepages (EXPERT editors, revealed placement).

CONFOUND (this cell is flagged YELLOW): most-read reflects PLACEMENT/promotion as
much as intrinsic interest -- a story splashed at the top of the homepage gets more
clicks.  We record rank + capture timestamp.  The control pool (same-day other
headlines from the same captures) partially controls for day & topic-mix.  The
separate news_homepages dataset (homepage spatial position) could later instrument
placement directly.

Dedup: keep earliest capture-occurrence of each normalized headline (a story that
is most-read on several consecutive days, or appears as control then later most-read,
is collapsed to its first appearance with its FIRST observed label decided by a
"most-read wins if ever most-read in window" rule -- documented below).

Splits: stable hash of headline id -> md5 % 10; 0-7 train, 8 eval, 9 test.
"""
import argparse
import gzip
import hashlib
import html as htmllib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

CHAR_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-",
    "…": "...", " ": " ",
}


def normalize(t):
    t = htmllib.unescape(htmllib.unescape(t or ""))
    t = unicodedata.normalize("NFKC", t)
    for k, v in CHAR_MAP.items():
        t = t.replace(k, v)
    return re.sub(r"\s+", " ", t).strip()


def tkey(t):
    return " ".join(re.findall(r"[a-z0-9]+", normalize(t).lower()))


# --- presentation cleaners, applied SYMMETRICALLY to positives and controls ---
# The most-read pages give clean full headlines; the OLD (2014-16) homepage index
# carousel gives anchor text with a leading "N:" rank, a trailing "Watch"/duration,
# or a "Full article" prefix. If we did NOT strip these from controls, the classes
# would differ by FORMAT (a cross-source typography leak) rather than by content.
# So both classes pass through the same clean_headline().
LEAD_RANK_RE = re.compile(r"^\d{1,2}:\s+")          # "1: Giant rubber duck ..."
LEAD_DUR_RE = re.compile(r"^\d{1,2}:\d{2}\s+")       # "3:14 Migrants ..." video clip
FULL_ART_RE = re.compile(r"^full article[:\s-]*", re.I)
PREFIX_RE = re.compile(r"^(video|watch|live|audio|listen|in pictures|in pics|"
                       r"podcast|gallery)\b[:\s-]*", re.I)
# media-duration prefixes "Video 1 minute 38 seconds " / "1 min 14 secs "
DUR_RE = re.compile(r"^(video\s+)?\d+\s+(hour|minute|min|second|sec)s?"
                    r"(\s+\d+\s+(minute|min|second|sec)s?)?\s*", re.I)
# trailing video markers: "... Watch", "... Watch 06:08", "... 06:08", "... Video"
TRAIL_WATCH_RE = re.compile(r"\s+(watch|video|listen|live)(\s+\d{1,2}:\d{2})?\s*$", re.I)
TRAIL_DUR_RE = re.compile(r"\s+\d{1,2}:\d{2}\s*$")
# trailing card metadata "  7 hrs ago Section" (React backstop)
TRAIL_META_RE = re.compile(r"\s+\d+\s+(hrs?|hours?|mins?|minutes?|days?)\s+ago"
                           r"(\s+[A-Z][\w &]+)?\s*$", re.I)
# a control is a VIDEO CLIP (drop it) if it has a clip duration anywhere, or a
# Watch/Video/Listen marker at the START or END (not mid-headline, to avoid
# dropping legit headlines like "Watchdog probes ..." / "... video game sales").
VIDEO_MARK_RE = re.compile(r"(^|\s)\d{1,2}:\d{2}(\s|$)|"
                           r"^(watch|video|listen)\b|\b(watch|video|listen)$", re.I)


def strip_prefix(t):
    """Symmetric presentation cleaner (name kept for back-compat)."""
    prev = None
    while prev != t:
        prev = t
        t = LEAD_DUR_RE.sub("", t).strip()
        t = LEAD_RANK_RE.sub("", t).strip()
        t = FULL_ART_RE.sub("", t).strip()
        t = PREFIX_RE.sub("", t).strip()
        t = DUR_RE.sub("", t).strip()
        t = TRAIL_WATCH_RE.sub("", t).strip()
        t = TRAIL_DUR_RE.sub("", t).strip()
        t = TRAIL_META_RE.sub("", t).strip()
    return t


def hid(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def split_of(h):
    v = int(hashlib.md5(h.encode()).hexdigest(), 16) % 10
    return "train" if v <= 7 else ("eval" if v == 8 else "test")


def day_of(ts):
    return ts[:8]


def section_of(href):
    m = re.search(r"/(news|sport|business|world|uk|technology|science|health|"
                  r"entertainment|education|politics)/", href or "")
    return m.group(1) if m else "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default="raw/captures.jsonl")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--min-mostread", type=int, default=5,
                    help="require >= this many most-read items in a capture to use it")
    ap.add_argument("--min-controls", type=int, default=5,
                    help="require >= this many control headlines on a day")
    ap.add_argument("--min-ctrl-len", type=int, default=20,
                    help="min cleaned-headline length for a control (drops teasers)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    manifest = {"build_date": datetime.now().isoformat(timespec="seconds")}
    log = []

    def emit(*a):
        s = " ".join(str(x) for x in a)
        print(s); log.append(s)

    # ---------- load captures ----------
    caps = []
    with open(args.captures) as f:
        for line in f:
            try:
                caps.append(json.loads(line))
            except Exception:
                continue
    emit(f"loaded {len(caps)} capture records")
    by_parser = Counter(c.get("parser") for c in caps if c.get("n_mostread", 0) > 0)
    emit("captures-with-mostread by parser:", dict(by_parser))

    # ---------- assemble positive (most-read) rows ----------
    # one row per (capture, rank). dedup happens later by normalized headline.
    pos_rows = []
    days_with_mr = set()
    # day -> set of normalized most-read keys (for control exclusion)
    day_mr_keys = defaultdict(set)
    for c in caps:
        items = c.get("most_read") or []
        if len(items) < args.min_mostread:
            continue
        day = day_of(c["timestamp"])
        days_with_mr.add(day)
        for it in items:
            head = strip_prefix(normalize(it["headline"]))
            if len(head) < 8:
                continue
            k = tkey(head)
            if not k:
                continue
            day_mr_keys[day].add(k)
            pos_rows.append({
                "headline": head, "tkey": k, "rank": it.get("rank"),
                "href": it.get("href", ""), "timestamp": c["timestamp"],
                "day": day, "channel": c.get("channel"),
                "parser": c.get("parser"), "y": 1,
            })
    emit(f"positive (most-read) raw rows: {len(pos_rows)} "
         f"across {len(days_with_mr)} days")

    # ---------- assemble control (non-most-read) rows ----------
    # use channel-B captures' "others"; only keep controls on days that HAVE a
    # most-read list (so the day/topic-mix is matched).  Exclude any headline
    # whose key is in that day's most-read set.
    ctrl_rows = []
    for c in caps:
        others = c.get("others") or []
        if not others:
            continue
        day = day_of(c["timestamp"])
        if day not in days_with_mr:
            continue
        mrk = day_mr_keys[day]
        for o in others:
            raw = normalize(o["headline"])
            # Drop VIDEO-CLIP controls outright: a leading/trailing clip duration
            # ("3:14 ...", "... 06:08") or a Watch/Video marker means this anchor is
            # a video caption, a different register that never appears in the
            # most-read article list -> excluding avoids a register/typography leak.
            if VIDEO_MARK_RE.search(raw):
                continue
            head = strip_prefix(raw)
            # control min-length floor: drops residual furniture/teasers so the
            # control pool is comparable in form to the most-read headlines.
            if len(head) < args.min_ctrl_len or len(head.split()) < 4:
                continue
            k = tkey(head)
            if not k or k in mrk:
                continue
            ctrl_rows.append({
                "headline": head, "tkey": k, "rank": None,
                "href": o.get("href", ""), "timestamp": c["timestamp"],
                "day": day, "channel": c.get("channel"),
                "parser": c.get("parser"), "y": 0,
            })
    emit(f"control (non-most-read) raw rows: {len(ctrl_rows)}")

    df = pd.DataFrame(pos_rows + ctrl_rows)
    if df.empty:
        emit("NO ROWS -- aborting"); return
    df["ts_int"] = df.timestamp.astype(str)

    # ---------- dedup: keep earliest occurrence per normalized headline ----------
    # rule: if a headline ever appears as most-read, keep it as y=1 (earliest
    # most-read capture); else keep its earliest control occurrence as y=0.
    df = df.sort_values("ts_int")
    pos = df[df.y == 1].drop_duplicates("tkey", keep="first")
    pos_keys = set(pos.tkey)
    neg_all = df[df.y == 0]
    neg_not_promoted = neg_all[~neg_all.tkey.isin(pos_keys)]
    neg = neg_not_promoted.drop_duplicates("tkey", keep="first")
    n_dup_pos = int((df.y == 1).sum() - len(pos))
    n_dup_neg = int(len(neg_not_promoted) - len(neg))
    # control rows whose headline is ALSO most-read somewhere -> folded into pos
    n_neg_promoted = int(neg_all.tkey.isin(pos_keys).sum())
    ded = pd.concat([pos, neg], ignore_index=True)
    emit(f"after dedup: pos={len(pos)} neg={len(neg)} "
         f"(dropped {n_dup_pos} dup-pos, {n_dup_neg} dup-neg; "
         f"{n_neg_promoted} control rows folded into pos because that headline "
         f"was most-read on some capture)")
    manifest_dedup = {"dup_pos_dropped": n_dup_pos, "dup_neg_dropped": n_dup_neg,
                      "neg_rows_promoted_to_pos": n_neg_promoted}

    # ---------- spot check ----------
    emit("\n===== SPOT-CHECK 12 random POS (most-read) =====")
    for _, r in ded[ded.y == 1].sample(min(12, len(pos)), random_state=args.seed).iterrows():
        emit(f"  [day={r.day} rank={r.rank} {r.parser}] {r.headline[:100]}")
    emit("===== SPOT-CHECK 12 random NEG (control) =====")
    for _, r in ded[ded.y == 0].sample(min(12, len(neg)), random_state=args.seed).iterrows():
        emit(f"  [day={r.day} {r.parser}] {r.headline[:100]}")

    # ---------- id + split ----------
    ded["headline_id"] = ded.headline.map(hid)
    ded = ded.drop_duplicates("headline_id")
    ded["split"] = ded.headline_id.map(split_of)
    ded["text"] = ded.headline
    ded["section"] = ded.href.map(section_of)

    manifest["dedup"] = manifest_dedup
    manifest["counts"] = {
        "pos": int((ded.y == 1).sum()), "neg": int((ded.y == 0).sum()),
        "total": len(ded),
        "splits": {s: int((ded.split == s).sum()) for s in ("train", "eval", "test")},
        "days": int(ded.day.nunique()),
    }
    emit("\ncounts:", manifest["counts"])

    # ---------- section-prefix confound check ----------
    sec_counts = ded.groupby(["section", "y"]).size().unstack(fill_value=0)
    emit("\nsection x label counts:")
    emit(sec_counts.to_string())
    manifest["section_label_counts"] = sec_counts.to_dict()

    # ---------- TF-IDF/LR floor (grouped by capture-day is implicit via splits) -
    def tfidf_floor(d, col):
        tr = d[d.split == "train"]
        if tr.y.nunique() < 2:
            return {}
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000,
                              sublinear_tf=True, lowercase=True)
        Xtr = vec.fit_transform(tr[col])
        lr = LogisticRegression(max_iter=2000, C=1.0)
        lr.fit(Xtr, tr.y.values)
        out = {}
        for sp in ("eval", "test"):
            dd = d[d.split == sp]
            if len(dd) and dd.y.nunique() == 2:
                out[f"auc_{sp}"] = round(float(roc_auc_score(
                    dd.y.values, lr.predict_proba(vec.transform(dd[col]))[:, 1])), 4)
        names = np.array(vec.get_feature_names_out())
        order = np.argsort(lr.coef_[0])
        out["top20_pos"] = [(names[i], round(float(lr.coef_[0][i]), 3)) for i in order[-20:][::-1]]
        out["top20_neg"] = [(names[i], round(float(lr.coef_[0][i]), 3)) for i in order[:20]]
        return out

    floor = tfidf_floor(ded, "text")
    manifest["tfidf_floor"] = floor
    emit("\n===== TF-IDF/LR floor =====")
    emit(json.dumps({k: v for k, v in floor.items() if k.startswith("auc")}, indent=1))
    emit("top20 POS feats:", floor.get("top20_pos"))
    emit("top20 NEG feats:", floor.get("top20_neg"))

    # ---------- length confound ----------
    L = ded.text.str.len().values
    r_len = float(np.corrcoef(L, ded.y.values)[0, 1])
    manifest["len_label_corr"] = round(r_len, 4)
    manifest["len_means"] = {"pos": round(float(L[ded.y == 1].mean()), 1),
                             "neg": round(float(L[ded.y == 0].mean()), 1)}
    emit(f"\nlength vs label corr: {r_len:.4f}  "
         f"means pos={manifest['len_means']['pos']} neg={manifest['len_means']['neg']}")
    # per-parser-era length (diagnose whether the leak is the 2015-17 source asymmetry)
    era_len = {}
    for pa in sorted(ded.parser.dropna().unique()):
        sub = ded[ded.parser == pa]
        era_len[pa] = {"pos_len": round(float(sub.loc[sub.y == 1, "text"].str.len().mean() or 0), 1),
                       "neg_len": round(float(sub.loc[sub.y == 0, "text"].str.len().mean() or 0), 1),
                       "pos": int((sub.y == 1).sum()), "neg": int((sub.y == 0).sum())}
    manifest["len_by_parser"] = era_len
    emit("length by parser-era:", json.dumps(era_len))

    # ---------- length-DECILE-matched floor (floor net of the length confound) ----
    # Balance pos/neg within text-length deciles, then re-run the TF-IDF floor.
    Ls = ded.text.str.len()
    try:
        ded["_ldec"] = pd.qcut(Ls, 10, labels=False, duplicates="drop")
    except Exception:
        ded["_ldec"] = 0
    keep = []
    rng_local = np.random.RandomState(args.seed)
    for _, g in ded.groupby("_ldec"):
        pi = g.index[g.y == 1].tolist()
        ni = g.index[g.y == 0].tolist()
        m = min(len(pi), len(ni))
        if m == 0:
            continue
        keep += list(rng_local.choice(pi, m, replace=False))
        keep += list(rng_local.choice(ni, m, replace=False))
    lm = ded.loc[keep].copy()
    floor_lm = tfidf_floor(lm, "text") if lm.y.nunique() == 2 else {}
    manifest["tfidf_floor_length_matched"] = {k: v for k, v in floor_lm.items()
                                              if k.startswith("auc")}
    manifest["length_matched_counts"] = {"pos": int((lm.y == 1).sum()),
                                          "neg": int((lm.y == 0).sum())}
    r_len_lm = float(np.corrcoef(lm.text.str.len().values, lm.y.values)[0, 1]) if len(lm) else 0.0
    emit(f"\nlength-matched floor (len corr now {r_len_lm:.3f}, "
         f"n={len(lm)}): {manifest['tfidf_floor_length_matched']}")

    # ---------- rank distribution ----------
    rk = ded.loc[ded.y == 1, "rank"].dropna()
    if len(rk):
        manifest["rank_dist"] = {str(int(k)): int(v) for k, v in rk.value_counts().sort_index().items()}
        emit("rank distribution (pos):", manifest["rank_dist"])

    # ---------- write ----------
    built = out_dir / "built"
    built.mkdir(parents=True, exist_ok=True)
    cols = ["text", "y", "rank", "timestamp", "day", "section", "channel",
            "parser", "href", "headline_id"]
    ded = ded.rename(columns={"y": "judgement"})
    cols[1] = "judgement"
    for sp in ("train", "eval", "test"):
        d = ded[ded.split == sp][cols].sample(frac=1.0, random_state=args.seed)
        p = built / f"{sp}.csv.gz"
        d.to_csv(p, index=False, compression="gzip")
        emit(f"wrote {len(d):6d} -> {p}")

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    with open(out_dir / "build_log.txt", "w") as f:
        f.write("\n".join(log))
    emit("done.")


if __name__ == "__main__":
    main()
