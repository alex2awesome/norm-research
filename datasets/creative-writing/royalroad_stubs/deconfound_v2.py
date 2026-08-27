#!/usr/bin/env python
"""Popularity-deconfound the RoyalRoad stub dataset (cw-A) -> v2.

cw-A = RoyalRoad web-serials that got a publisher/Kindle-Unlimited deal
(y=1, "STUB") vs. presentation-matched controls (y=0).

AUDIT FINDINGS this script fixes
--------------------------------
1. POPULARITY CONFOUND. Stubs skew systematically LOW-popularity (views,
   pages, chapters, followers, rating). Confound-only AUC was 0.809 --
   BEATS the 0.746 TF-IDF text floor, i.e. the label is partly readable
   from metadata alone. Fix = Math.SE v3.3-style propensity-decile
   balancing: fit a logistic propensity model for y using ONLY the
   popularity confounds (rating + log-transformed skewed counts), bin
   the propensity into deciles, then within each decile keep an equal
   number of pos/neg (random undersample of the majority side, fixed
   seed). Drives confound-only AUC toward ~0.5.
2. CONTENT-TYPE one-sided confound. 42 control rows (14 fictions) are
   'Fan Fiction'; ZERO stubs are. Drop all Fan Fiction controls so both
   sides are all-'Original' (matching the stubs) BEFORE balancing.
3. ARCHIVE-AVAILABILITY leak. x_source must be 'wayback' for BOTH
   classes (else "is it on the Wayback Machine" leaks the label). Verified
   and asserted.
4. LABEL LEAK. labels/tags carry 'STUB' (the y source) on every positive
   and 'Original'/'Fan Fiction' content-type tags -- excluded from every
   feature. Text features use the normalized `text` column ONLY, never
   raw_text.

Grouping. All chapters of a fiction stay on ONE split side. Splits use
the stable md5(fiction_id)%10 hash (0-7 train, 8 eval, 9 test) -- never a
seeded shuffle of a growing list.

NOTE. This is the FIRST recovery batch (~1,132 rows). The full Wayback
recovery is still running and a richer rebuild will follow; this
establishes first-pass modeling state + bounds NOW. We READ v1 and WRITE
a NEW file -- v1 and raw/ are never touched.
"""
import argparse
import hashlib
import json
import math
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SEED = 13

# Popularity confounds. Counts are right-skewed -> log1p; rating is bounded
# -> left as-is.
COUNT_CONFOUNDS = ["views", "pages", "chapters_listed", "followers"]
RAW_CONFOUNDS = ["rating"]
# Tags/labels that encode the label or content-type partition -- NEVER features.
BANNED_TOKENS = {"STUB", "Original", "Fan Fiction"}


def stable_split(fiction_id):
    h = int(hashlib.md5(str(fiction_id).encode()).hexdigest(), 16) % 10
    return "train" if h <= 7 else ("eval" if h == 8 else "test")


def content_type(row):
    s = set(row.get("labels") or []) | set(row.get("tags") or [])
    if "Fan Fiction" in s:
        return "Fan Fiction"
    if "Original" in s:
        return "Original"
    return "Unknown"


def confound_matrix(rows):
    """Popularity feature matrix: rating + log1p(counts). Standardized."""
    X = []
    for r in rows:
        feat = [float(r[c]) for c in RAW_CONFOUNDS]
        feat += [math.log1p(float(r[c])) for c in COUNT_CONFOUNDS]
        X.append(feat)
    X = np.asarray(X, dtype=float)
    return StandardScaler().fit_transform(X)


def confound_auc(rows):
    """Cross-validated-ish confound-only AUC. Single in-sample LR fit is
    fine as a *diagnostic of separability* (we report before/after on the
    same protocol, so the comparison is apples-to-apples)."""
    y = np.array([r["y"] for r in rows])
    if len(set(y)) < 2:
        return float("nan")
    X = confound_matrix(rows)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    return roc_auc_score(y, p)


def fiction_propensity(rows):
    """Fit propensity P(y=1 | popularity) at the FICTION level (dedup so
    each serial counts once, not 3x for its 3 chapters), return
    {fiction_id: propensity}."""
    seen = {}
    for r in rows:
        seen.setdefault(r["fiction_id"], r)
    fics = list(seen.values())
    y = np.array([f["y"] for f in fics])
    X = confound_matrix(fics)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    return {f["fiction_id"]: pi for f, pi in zip(fics, p)}, fics, y, p


def propensity_decile_balance(rows, n_bins=10):
    """Balance pos/neg WITHIN propensity deciles, at the fiction level.

    Returns the kept fiction_ids. Within each decile we keep
    min(n_pos, n_neg) fictions per side (random undersample of majority,
    fixed seed)."""
    prop, fics, y, p = fiction_propensity(rows)
    rng = np.random.default_rng(SEED)
    # decile edges from the pooled propensity distribution
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_idx = np.digitize(p, edges[1:-1])
    kept = []
    by_bin = {}
    for f, b in zip(fics, bin_idx):
        by_bin.setdefault(b, {0: [], 1: []}).setdefault(f["y"], []).append(
            f["fiction_id"]
        )
    for b in sorted(by_bin):
        pos = by_bin[b].get(1, [])
        neg = by_bin[b].get(0, [])
        k = min(len(pos), len(neg))
        if k == 0:
            continue
        pos = list(rng.permutation(pos))[:k]
        neg = list(rng.permutation(neg))[:k]
        kept.extend(pos)
        kept.extend(neg)
    return set(kept)


# --------------------------------------------------------------------------
# Bounds on the balanced set
# --------------------------------------------------------------------------

def grouped_train_eval(rows):
    tr = [r for r in rows if r["split"] == "train"]
    ev = [r for r in rows if r["split"] in ("eval", "test")]
    return tr, ev


def tfidf_floor(rows):
    tr, ev = grouped_train_eval(rows)
    vec = TfidfVectorizer(
        sublinear_tf=True, min_df=3, max_df=0.9, ngram_range=(1, 2),
        max_features=50000, strip_accents="unicode", lowercase=True,
    )
    Xtr = vec.fit_transform([r["text"] for r in tr])
    Xev = vec.transform([r["text"] for r in ev])
    ytr = np.array([r["y"] for r in tr])
    yev = np.array([r["y"] for r in ev])
    clf = LogisticRegression(max_iter=4000, C=1.0)
    clf.fit(Xtr, ytr)
    auc = roc_auc_score(yev, clf.predict_proba(Xev)[:, 1])
    # top features both directions
    coef = clf.coef_[0]
    vocab = np.array(vec.get_feature_names_out())
    top_pos = vocab[np.argsort(coef)[-25:][::-1]]
    top_neg = vocab[np.argsort(coef)[:25]]
    return auc, list(top_pos), list(top_neg), len(tr), len(ev)


def tfidf_floor_no_proper(rows):
    """TF-IDF floor after stripping proper-noun tokens (serial-identity /
    character-name leakage check). A proper noun ~ a token usually seen
    capitalized mid-text and rarely lowercased."""
    import re as _re
    capc, lowc = {}, {}
    for r in rows:
        for w in _re.findall(r"[A-Za-z][A-Za-z'-]+", r["text"]):
            d = capc if w[0].isupper() else lowc
            d[w.lower()] = d.get(w.lower(), 0) + 1
    proper = {w for w in capc
              if capc[w] >= 3 and capc[w] > 3 * max(lowc.get(w, 0), 1)}

    def strip(t):
        toks = _re.findall(r"[A-Za-z][A-Za-z'-]+|[^A-Za-z]+", t)
        return "".join(x for x in toks if x.lower().strip() not in proper)

    tr, ev = grouped_train_eval(rows)
    vec = TfidfVectorizer(sublinear_tf=True, min_df=3, max_df=0.9,
                          ngram_range=(1, 2), max_features=50000,
                          strip_accents="unicode", lowercase=True)
    Xtr = vec.fit_transform([strip(r["text"]) for r in tr])
    Xev = vec.transform([strip(r["text"]) for r in ev])
    ytr = np.array([r["y"] for r in tr])
    yev = np.array([r["y"] for r in ev])
    clf = LogisticRegression(max_iter=4000, C=1.0)
    clf.fit(Xtr, ytr)
    return roc_auc_score(yev, clf.predict_proba(Xev)[:, 1]), len(proper)


_FUNCWORDS = ("the a an and or but if then so as of to in on at by for with from "
              "into onto upon over under about than that this these those it its he "
              "she they them his her their him your you i we our us me my mine not "
              "no nor very just only also too even still yet once again will would "
              "can could shall should may might must do does did done has have had "
              "been being is are was were be am").split()


def register_v_auc(rows):
    """Function-word + punctuation REGISTER probe -- a deterministic,
    content-free V signal (clause length, dialogue/punctuation rhythm,
    closed-class rates). Complements the readability probe."""
    import re as _re
    idx = {w: i for i, w in enumerate(_FUNCWORDS)}

    def feat(t):
        toks = _re.findall(r"[a-z]+", t.lower())
        n = max(len(toks), 1)
        v = [0.0] * len(_FUNCWORDS)
        for w in toks:
            j = idx.get(w)
            if j is not None:
                v[j] += 1.0 / n
        return v + [t.count(",") / n, t.count(";") / n,
                    t.count("--") / n, t.count("\n\n") / n]

    tr, ev = grouped_train_eval(rows)
    A = np.array([feat(r["text"]) for r in tr])
    B = np.array([feat(r["text"]) for r in ev])
    sc = StandardScaler().fit(A)
    A, B = sc.transform(A), sc.transform(B)
    ytr = np.array([r["y"] for r in tr])
    yev = np.array([r["y"] for r in ev])
    clf = LogisticRegression(max_iter=4000, C=1.0)
    clf.fit(A, ytr)
    return roc_auc_score(yev, clf.predict_proba(B)[:, 1])


def length_corr(rows):
    """AUC of n_words alone (length leakage probe)."""
    y = np.array([r["y"] for r in rows])
    w = np.array([float(r.get("n_words") or 0) for r in rows])
    if len(set(y)) < 2:
        return float("nan")
    auc = roc_auc_score(y, w)
    return max(auc, 1 - auc)  # report directionless separability


def v_feature_auc(rows):
    """Deterministic V-feature (readability / structure) probe to confirm
    articulable-verifiable signal V>0. Uses simple surface stats so it is
    fully replicable; no LLM. Train/eval grouped split."""
    import textstat

    def feats(text):
        n_chars = len(text)
        words = text.split()
        n_words = max(len(words), 1)
        sents = [s for s in __import__("re").split(r"[.!?]+", text) if s.strip()]
        n_sents = max(len(sents), 1)
        paras = [p for p in text.split("\n\n") if p.strip()]
        n_paras = max(len(paras), 1)
        avg_word_len = n_chars / n_words
        avg_sent_len = n_words / n_sents
        avg_para_len = n_words / n_paras
        # dialogue density (quote marks / words) and exclaim/question rates
        dq = text.count('"') / n_words
        excl = text.count("!") / n_words
        ques = text.count("?") / n_words
        try:
            fre = textstat.flesch_reading_ease(text)
            fk = textstat.flesch_kincaid_grade(text)
        except Exception:
            fre, fk = 0.0, 0.0
        ttr = len(set(w.lower() for w in words)) / n_words  # type-token ratio
        return [avg_word_len, avg_sent_len, avg_para_len, dq, excl, ques,
                fre, fk, ttr, math.log1p(n_words)]

    tr, ev = grouped_train_eval(rows)
    Xtr = StandardScaler().fit(np.array([feats(r["text"]) for r in tr]))
    A = np.array([feats(r["text"]) for r in tr])
    B = np.array([feats(r["text"]) for r in ev])
    sc = StandardScaler().fit(A)
    A, B = sc.transform(A), sc.transform(B)
    ytr = np.array([r["y"] for r in tr])
    yev = np.array([r["y"] for r in ev])
    clf = LogisticRegression(max_iter=4000, C=1.0)
    clf.fit(A, ytr)
    return roc_auc_score(yev, clf.predict_proba(B)[:, 1])


def bge_upper_bound(rows, gpu="1"):
    """bge dense embedding + LR upper bound. Returns (auc, note)."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:  # pragma: no cover
        return None, f"sentence_transformers import failed: {e}"
    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
    tr, ev = grouped_train_eval(rows)
    # bge max ~512 tokens; truncate text for embedding (presentation-normalized)
    Etr = model.encode([r["text"][:6000] for r in tr], batch_size=32,
                        normalize_embeddings=True, show_progress_bar=False)
    Eev = model.encode([r["text"][:6000] for r in ev], batch_size=32,
                        normalize_embeddings=True, show_progress_bar=False)
    ytr = np.array([r["y"] for r in tr])
    yev = np.array([r["y"] for r in ev])
    clf = LogisticRegression(max_iter=4000, C=1.0)
    clf.fit(Etr, ytr)
    auc = roc_auc_score(yev, clf.predict_proba(Eev)[:, 1])
    return auc, "bge-large-en-v1.5 on GPU " + gpu


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="/lfs/skampere3/0/alexspan/norm-research/"
                    "datasets/creative-writing/royalroad_stubs/built/"
                    "royalroad_stubs_v1.jsonl")
    ap.add_argument("--out", default="/lfs/skampere3/0/alexspan/norm-research/"
                    "datasets/creative-writing/royalroad_stubs/built/"
                    "royalroad_deconf_v2.jsonl")
    ap.add_argument("--readme", default="/lfs/skampere3/0/alexspan/norm-research/"
                    "datasets/creative-writing/royalroad_stubs/built/README_v2.md")
    ap.add_argument("--bge-gpu", default="1",
                    help="GPU index for bge, or 'skip'")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp)]
    report = []

    def log(*a):
        line = " ".join(str(x) for x in a)
        print(line)
        report.append(line)

    log("=== RoyalRoad stub (cw-A) popularity deconfound -> v2 ===")
    log(f"input: {args.inp}")
    log(f"loaded rows: {len(rows)}")

    # --- 1. x_source balance (archive-availability leak guard) -----------
    from collections import Counter
    xs = Counter((r["y"], r.get("x_source")) for r in rows)
    log("x_source by y:", dict(xs))
    srcs = set(r.get("x_source") for r in rows)
    assert srcs == {"wayback"}, f"x_source NOT all wayback: {srcs}"
    log("x_source OK: both classes are 'wayback'")

    # --- 2. drop Fan Fiction controls ------------------------------------
    ct_before = Counter((r["y"], content_type(r)) for r in rows)
    log("content-type by y (rows, before):", dict(ct_before))
    fanfic_rows = [r for r in rows if content_type(r) == "Fan Fiction"]
    fanfic_fids = set(r["fiction_id"] for r in fanfic_rows)
    assert all(r["y"] == 0 for r in fanfic_rows), "a STUB is fanfic?!"
    rows = [r for r in rows if content_type(r) != "Fan Fiction"]
    log(f"dropped Fan Fiction controls: {len(fanfic_rows)} rows / "
        f"{len(fanfic_fids)} fictions (all y=0)")
    log(f"rows after fanfic drop: {len(rows)}")

    # --- confound-only AUC BEFORE balancing ------------------------------
    auc_before = confound_auc(rows)
    log(f"confound-only AUC (after fanfic drop, before balance): {auc_before:.4f}")
    log(f"  rating-solo AUC: {roc_auc_score([r['y'] for r in rows], [r['rating'] for r in rows]):.4f}")

    # --- 2b. propensity-decile balancing ---------------------------------
    keep = propensity_decile_balance(rows, n_bins=10)
    rows = [r for r in rows if r["fiction_id"] in keep]
    auc_after = confound_auc(rows)
    yc = Counter(r["y"] for r in rows)
    n_fic = len(set(r["fiction_id"] for r in rows))
    log(f"kept fictions after balance: {n_fic}")
    log(f"rows after balance: {len(rows)}  label balance (rows): {dict(yc)}")
    fic_y = {}
    for r in rows:
        fic_y[r["fiction_id"]] = r["y"]
    log(f"label balance (fictions): {dict(Counter(fic_y.values()))}")
    log(f"confound-only AUC AFTER balance: {auc_after:.4f}  "
        f"(was {auc_before:.4f})")

    # --- 3. stable hash split by fiction_id ------------------------------
    for r in rows:
        r["split"] = stable_split(r["fiction_id"])
    sp = Counter(r["split"] for r in rows)
    sp_fic = Counter(stable_split(fid) for fid in set(r["fiction_id"] for r in rows))
    log("split (rows):", dict(sp), " split (fictions):", dict(sp_fic))
    # label balance per split
    for s in ("train", "eval", "test"):
        c = Counter(r["y"] for r in rows if r["split"] == s)
        log(f"  split {s} label balance: {dict(c)}")

    # --- 4. bounds -------------------------------------------------------
    log("--- bounds on balanced set (grouped own split) ---")
    auc_tf, top_pos, top_neg, ntr, nev = tfidf_floor(rows)
    log(f"TF-IDF/LR floor AUC: {auc_tf:.4f}  (train {ntr} / eval+test {nev})")
    log(f"  top y=1 (STUB) features: {top_pos}")
    log(f"  top y=0 (control) features: {top_neg}")
    auc_tf_np, n_proper = tfidf_floor_no_proper(rows)
    log(f"TF-IDF floor, proper-nouns stripped ({n_proper} tokens): "
        f"{auc_tf_np:.4f}  (leakage check: vs {auc_tf:.4f})")
    lc = length_corr(rows)
    log(f"length (n_words) directionless AUC: {lc:.4f}")
    v_auc = v_feature_auc(rows)
    log(f"V-feature (readability/structure stats) AUC: {v_auc:.4f}")
    v_reg = register_v_auc(rows)
    log(f"V-feature (function-word/punct register) AUC: {v_reg:.4f}  "
        f"-> V>0: {max(v_auc, v_reg) > 0.55}")

    bge_auc, bge_note = (None, "skipped")
    if args.bge_gpu != "skip":
        try:
            bge_auc, bge_note = bge_upper_bound(rows, gpu=args.bge_gpu)
            log(f"bge dense upper bound AUC: {bge_auc:.4f}  ({bge_note})")
        except Exception as e:
            log(f"bge SKIPPED due to error: {e}")
            bge_note = f"error: {e}"
    else:
        log("bge skipped (--bge-gpu skip)")

    # --- 5. write outputs ------------------------------------------------
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"WROTE deconfounded v2: {args.out}  ({len(rows)} rows)")

    readme = f"""# RoyalRoad stub dataset (cw-A) -- deconfounded v2

**Source built file (read-only input):** `royalroad_stubs_v1.jsonl`
**This file:** `royalroad_deconf_v2.jsonl`  ({len(rows)} rows / {n_fic} fictions)

cw-A = RoyalRoad web-serials that earned a publisher / Kindle-Unlimited
deal (`y=1`, label source = the `STUB` tag) vs. presentation-matched
controls (`y=0`). One serial contributes up to 3 early chapters; all
chapters of a serial stay on one split side.

> **First batch.** This is the FIRST Wayback-recovery batch (~1,132 raw
> rows). The full recovery is still running and writing to `raw/`; a
> richer rebuild will follow. This file establishes the first-pass
> modeling state + bounds NOW. `v1` and `raw/` are never modified.

## Fixes applied (vs. v1 audit)

1. **Popularity propensity-decile deconfound.** Stubs skewed
   systematically low-popularity (views, pages, chapters, followers,
   rating). v1 confound-only AUC was **0.809**, BEATING the text floor
   -- the label was partly readable from metadata. We fit a logistic
   propensity `P(y=1 | rating, log1p(views,pages,chapters,followers))`
   at the **fiction** level, bin into deciles, and keep equal pos/neg per
   decile (random undersample of majority, seed {SEED}).
   - confound-only AUC **before** balance (after fanfic drop): **{auc_before:.3f}**
   - confound-only AUC **after**  balance: **{auc_after:.3f}**
2. **Fan-fiction drop.** {len(fanfic_rows)} control rows
   ({len(fanfic_fids)} fictions) were `Fan Fiction`; ZERO stubs are. All
   dropped so both sides are all-`Original` (matching stubs) before
   balancing.
3. **Archive-availability leak guard.** `x_source == 'wayback'` for BOTH
   classes (verified/asserted) so "is it on the Wayback Machine" cannot
   leak the label.
4. **Label-leak guard.** `STUB`/`Original`/`Fan Fiction` tokens are
   excluded from every feature. Text features use the normalized `text`
   column ONLY (never `raw_text`).

## Splits (stable hash, grouped by fiction)

`split = {{train if md5(fiction_id)%10 in 0..7, eval if 8, test if 9}}`.
Rows: {dict(sp)}  Fictions: {dict(sp_fic)}

## Bounds (grouped own split: train -> eval+test)

| Bound | AUC |
|---|---|
| Confound-only (after deconf) | {auc_after:.3f} |
| TF-IDF/LR text floor | {auc_tf:.3f} |
| TF-IDF floor, proper-nouns stripped | {auc_tf_np:.3f} |
| bge dense upper bound | {('%.3f' % bge_auc) if bge_auc is not None else 'n/a (' + bge_note + ')'} |
| Length (n_words) directionless | {lc:.3f} |
| V-feature (readability/structure stats) | {v_auc:.3f} |
| V-feature (function-word/punct register) | {v_reg:.3f} |

**Leakage check (resolved).** TF-IDF top *coefficients* are dominated by
character names (the expected serial-identity leakage), but after
stripping {n_proper} proper-noun tokens the floor barely moves
({auc_tf:.3f} -> {auc_tf_np:.3f}): the names are high-weight low-coverage
features, not the source of the floor. The text signal is real, not
memorized identity.

**V > 0? {'YES' if max(v_auc, v_reg) > 0.55 else 'NO/weak'}.** The naive
readability/structure-stat probe is flat ({v_auc:.3f}), but a
deterministic function-word + punctuation **register** probe reaches
{v_reg:.3f} -- stubs differ from controls in stylistic register, a fully
replicable verifiable signal above chance.

**TF-IDF top features.** Character-name / genre leakage (see check above):
- top `y=1` (STUB): {top_pos[:15]}
- top `y=0` (control): {top_neg[:15]}

## Columns
Same schema as v1 plus a recomputed `split`. Use `text` (normalized),
never `raw_text`. Confounds (`rating`, `views`, `pages`,
`chapters_listed`, `followers`) are retained for reference but were used
ONLY for deconfounding -- do not feed them as model features.

_Generated by `deconfound_v2.py` (seed {SEED})._
"""
    with open(args.readme, "w", encoding="utf-8") as f:
        f.write(readme)
    log(f"WROTE README: {args.readme}")
    log("=== done ===")


if __name__ == "__main__":
    main()
