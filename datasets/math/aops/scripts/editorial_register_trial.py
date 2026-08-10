#!/usr/bin/env python3
"""Editorial-register trial classifier: AoPS Wiki (editorial, y=1) vs forum
(y=0) solutions, restricted to problems present in BOTH corpora, group-split
by problem. Plus: skew analysis of how editorial-looking forum answers are,
and whether editorial-likeness predicts de-confounded thanks.

Outputs: prints cleaned samples for inspection, AUC tables, skew deciles,
thanks follow-on, top word features. Run from datasets/math/aops/.

Usage: python scripts/editorial_register_trial.py [--samples-only]
"""
import argparse
import html
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr, wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = "/Users/spangher/Projects/stanford-research/norm-research/datasets/math/aops"

# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------

def _norm_math(s: str) -> str:
    """Normalize the *content* of a math span to a delimiter-free common form."""
    s = s.replace(r"\minus{}", "-").replace(r"\equal{}", "=").replace(r"\plus{}", "+")
    s = re.sub(r"\\minus(?![a-zA-Z])", "-", s)
    s = re.sub(r"\\equal(?![a-zA-Z])", "=", s)
    s = re.sub(r"\\plus(?![a-zA-Z])", "+", s)
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    s = re.sub(r"\\displaystyle\s*", "", s)
    s = re.sub(r"\\left(?![a-zA-Z])\s*|\\right(?![a-zA-Z])\s*", "", s)
    s = re.sub(r"\\begin\{[a-z*]+\}|\\end\{[a-z*]+\}", " ", s)
    s = re.sub(r"\\[hv]space\*?\{[^{}]*\}", " ", s)            # pure layout
    s = s.replace("&", " ").replace("\\\\", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


_ATTR_WORDS = (r"(?:solutions?|sols?|diagrams?|asymptote|figures?|images?|latex(?:ed|ing)?|"
               r"formatt?(?:ed|ing)?|edit(?:s|ed|ing)?|minor\s+edits?|fix(?:es|ed)?|"
               r"credits?|written|revised|remade|cleaned(?:\s*up)?|"
               r"explanations?|remarks?|comments?|notes?|motivation|video\s*solutions?)")


def _strip_attributions(text: str) -> str:
    """Remove 'Solution by X' / '~username' / '-username' authorship credit."""
    # inline "(Solution by Foo)" / "Solution by Foo." / "Credit to Foo"
    text = re.sub(
        r"(?i)[\(\[]?\s*" + _ATTR_WORDS +
        r"\s*(?:is|was|were)?\s*(?:by|to|from)\s+[A-Za-z0-9_\-. ~&,]{1,60}[\)\]]?",
        " ", text)
    out = []
    for ln in text.split("\n"):
        s = ln.strip()
        # any line carrying an email(-remnant) is an attribution -> drop it
        # (URL stripping may already have eaten the domain: "name@ handle")
        if re.search(r"[A-Za-z0-9._%+-]{2,}@", s):
            continue
        # signature lines: "~username", "- username (minor edit by Y)", ...
        s_np = re.sub(r"\s*\([^)]*\)\s*$", "", s)  # drop trailing parenthetical
        if s_np and len(s_np) <= 70 and re.match(r"^[~\-–—*]+\s*[A-Za-z]", s_np) \
                and not re.search(r"[=+<>\\]|\b\d{3,}\b", s_np) \
                and len(s_np.split()) <= 8:
            continue
        # standalone username-with-digits line ("mathboy282")
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.\-]*\d{2,}[,.]?", s):
            continue
        # mid-line trailing "~username" signature (wiki habit)
        ln = re.sub(r"~\s*[A-Za-z][A-Za-z0-9_\-.]{2,30}\s*$", " ", ln)
        out.append(ln)
    return "\n".join(out)


def clean_wiki(raw: str) -> tuple[str, dict]:
    """Mediawiki solution text -> normalized plain text + structural stats."""
    t = raw
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S)
    t = re.sub(r"<asy>.*?</asy>", " ", t, flags=re.S)          # Asymptote figures
    n_fig = len(re.findall(r"<asy>", raw))
    # math spans -> keep normalized content, drop wiki-specific wrappers
    disp = re.findall(r"<cmath>(.*?)</cmath>", t, flags=re.S)
    t = re.sub(r"<cmath>(.*?)</cmath>",
               lambda m: "\n" + _norm_math(m.group(1)) + "\n", t, flags=re.S)
    t = re.sub(r"<i?math>(.*?)</i?math>",
               lambda m: " " + _norm_math(m.group(1)) + " ", t, flags=re.S)
    n_disp = len(disp)
    math_chars = sum(len(d) for d in disp) + sum(
        len(m) for m in re.findall(r"<i?math>(.*?)</i?math>", raw, flags=re.S))
    # wikitext furniture
    t = re.sub(r"\[\[(?:Category|File|Image|User)[^\]]*\]\]", " ", t, flags=re.I)
    t = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", t)         # [[a|b]] -> b
    t = re.sub(r"\[\[([^\]]*)\]\]", r"\1", t)                  # [[a]] -> a
    for _ in range(2):
        t = re.sub(r"\{\{[^{}]*\}\}", " ", t)                  # templates
    t = re.sub(r"(?m)^=+[^=\n]*=+\s*$", " ", t)                # ==headers==
    t = t.replace("'''", "").replace("''", "")
    # user-page attribution links (with or without label) -> drop entirely
    t = re.sub(r"~?\s*\[https?://\S*?[Uu]ser:[^\]]*\]", " ", t)
    t = re.sub(r"\[https?://\S*\s+([^\]]*)\]", r"\1", t)       # ext link w/ label
    t = re.sub(r"\[?https?://\S+\]?", " ", t)                  # bare urls
    t = re.sub(r"\b(?:www\.)?[a-z0-9\-]+\.(?:com|org|net|edu)\S*", " ", t)  # schemeless urls
    t = re.sub(r"</?(?:br|center|div|span|p|b|i|u|small|big|pre|code|table|tr|td|th|ol|ul|li|font|sup|sub|s)[^>]*/?>", " ", t, flags=re.I)
    t = html.unescape(t)
    t = _strip_attributions(t)
    t = re.sub(r"(?m)^[*#:;]+\s*", "- ", t)                    # wiki list bullets
    return _finish(t, n_disp, math_chars, n_fig)


def clean_forum(raw: str) -> tuple[str, dict]:
    """Forum BBCode (quote-stripped body) -> normalized plain text + stats."""
    t = raw
    # balanced quote blocks (innermost out); then orphan closers/openers:
    # text before an orphan [/quote] (or after an orphan [quote=..]) is
    # quoted material -> drop it.
    for _ in range(4):
        t2 = re.sub(r"\[quote[^\]]*\](?:(?!\[/?quote).)*\[/quote\]", " ", t,
                    flags=re.S | re.I)
        if t2 == t:
            break
        t = t2
    if re.search(r"\[/quote\]", t, flags=re.I):
        t = re.split(r"(?i)\[/quote\]", t)[-1]
    t = re.sub(r"\[quote[^\]]*\].*$", " ", t, flags=re.S | re.I)
    t = re.sub(r"\[asy\].*?\[/asy\]", " ", t, flags=re.S | re.I)
    n_fig = len(re.findall(r"\[asy\]", raw, flags=re.I))
    t = re.sub(r"\[img[^\]]*\].*?\[/img\]", " ", t, flags=re.S | re.I)
    t = re.sub(r"\[(?:attach|youtube|video|rule)[^\]]*\]", " ", t, flags=re.I)
    t = re.sub(r"\[url=[^\]]*\](.*?)\[/url\]", r"\1", t, flags=re.S | re.I)
    t = re.sub(r"\[url\].*?\[/url\]", " ", t, flags=re.S | re.I)
    # tag shells (keep content): hide, b, i, u, s, color, size, center, list, code
    t = re.sub(r"\[\s*/?\s*(?:hide|b|i|u|s|color|size|center|list|code|sup|sub|tip)"
               r"\s*(?:=[^\]]*)?\]", " ", t, flags=re.I)
    t = re.sub(r"(?m)^\s*\[\*\]\s*", "- ", t)                  # bbcode bullets
    t = re.sub(r"\[\*\]", " - ", t)
    t = re.sub(r":[a-z0-9_]{2,}:", " ", t)                     # :emoticon: codes
    t = re.sub(r"(?im)^\s*\[?\s*edit(?:ed)?\s*\d*\s*[:\]].*$", " ", t)  # EDIT: lines
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"\b(?:www\.)?[a-z0-9\-]+\.(?:com|org|net|edu)\S*", " ", t)
    t = html.unescape(t)
    # math spans: $$..$$, \[..\], align-like envs, then inline $..$
    disp_chars = [0]
    n_disp = [0]

    def _disp(m):
        n_disp[0] += 1
        disp_chars[0] += len(m.group(1))
        return "\n" + _norm_math(m.group(1)) + "\n"

    t = re.sub(r"\$\$(.+?)\$\$", _disp, t, flags=re.S)
    t = re.sub(r"\\\[(.+?)\\\]", _disp, t, flags=re.S)
    t = re.sub(r"\\begin\{(align\*?|eqnarray\*?|gather\*?|equation\*?)\}(.*?)"
               r"\\end\{\1\}", lambda m: _disp_env(m, n_disp, disp_chars), t, flags=re.S)
    inline_chars = [0]

    def _inl(m):
        inline_chars[0] += len(m.group(1))
        return " " + _norm_math(m.group(1)) + " "

    t = re.sub(r"\$(.+?)\$", _inl, t, flags=re.S)
    t = _strip_attributions(t)
    return _finish(t, n_disp[0], disp_chars[0] + inline_chars[0], n_fig)


def _disp_env(m, n_disp, disp_chars):
    n_disp[0] += 1
    disp_chars[0] += len(m.group(2))
    return "\n" + _norm_math(m.group(2)) + "\n"


_ENUM_RE = re.compile(
    r"(?im)(^\s*-\s|^\s*\(?\d+[\).]\s|^\s*\([a-z]\)\s|\bcase\s+\d|\bstep\s+\d|"
    r"\blemma\s*\d?|\bclaim\s*\d?\b)")


def _finish(t: str, n_disp: int, math_chars: int, n_fig: int):
    # raw LaTeX environments living outside the recognized math spans
    t = re.sub(r"\\begin\{[a-z*]+\}|\\end\{[a-z*]+\}", " ", t)
    t = re.sub(r"\\[hv]space\*?\{[^{}]*\}", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = t.strip()
    stats = {
        "len_chars": len(t),
        "n_display_math": n_disp,
        "latex_density": math_chars / max(len(t), 1),
        "n_paragraphs": len([p for p in re.split(r"\n\s*\n", t) if p.strip()]),
        "has_enumeration": int(bool(_ENUM_RE.search(t))),
        "n_figures": n_fig,
    }
    return t, stats


STRUCT_COLS = ["len_chars", "n_display_math", "latex_density", "n_paragraphs",
               "has_enumeration"]

# ----------------------------------------------------------------------------
# Data assembly
# ----------------------------------------------------------------------------

def load_corpus():
    wiki = pd.read_parquet(f"{ROOT}/wiki/solutions.parquet")
    wiki = wiki[~wiki.is_redirect].copy()
    forum = pd.read_parquet(f"{ROOT}/editorial_similarity.parquet")
    wiki["pkey"] = wiki.variant + "#" + wiki.problem_n.astype(str)
    forum["pkey"] = forum.variant + "#" + forum.problem_n.astype(str)
    shared = set(wiki.pkey) & set(forum.pkey)
    wiki = wiki[wiki.pkey.isin(shared)].copy()
    forum = forum[forum.pkey.isin(shared)].copy()
    print(f"[data] shared problems: {len(shared)} | wiki sols: {len(wiki)} | "
          f"forum sols: {len(forum)}")

    rows = []
    for _, r in wiki.iterrows():
        txt, st = clean_wiki(r.solution_text or "")
        rows.append(dict(pkey=r.pkey, contest=r.contest, y=1, post_id=-1,
                         sim_word=np.nan, text=txt, **st))
    for _, r in forum.iterrows():
        txt, st = clean_forum(r.body_noquote or "")
        rows.append(dict(pkey=r.pkey, contest=r.contest, y=0, post_id=r.post_id,
                         sim_word=r.sim_word, text=txt, **st))
    df = pd.DataFrame(rows)
    n0 = len(df)
    df = df[df.text.str.len() >= 100].reset_index(drop=True)
    print(f"[data] dropped {n0 - len(df)} docs with <100 cleaned chars -> {len(df)}")
    df["near_dup"] = (df.y == 0) & (df.sim_word > 0.9)
    print(f"[data] near-dup forum posts (sim_word>0.9): {int(df.near_dup.sum())}")
    return df


def print_samples(df, n=5, seed=7):
    rng = np.random.RandomState(seed)
    for y, name in [(1, "WIKI/EDITORIAL"), (0, "FORUM")]:
        sub = df[df.y == y]
        idx = rng.choice(sub.index, n, replace=False)
        for i in idx:
            r = df.loc[i]
            print("=" * 90)
            print(f"[{name}] {r.pkey}  len={r.len_chars} disp={r.n_display_math} "
                  f"paras={r.n_paragraphs} enum={r.has_enumeration}")
            print(r.text[:1200])
        print()

# ----------------------------------------------------------------------------
# Modeling
# ----------------------------------------------------------------------------

def make_vectorizers():
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3,
                           max_features=200_000, sublinear_tf=True,
                           strip_accents="unicode", lowercase=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           max_features=300_000, sublinear_tf=True,
                           lowercase=True)
    return word, char


def run_cv(df, train_mask, label=""):
    """GroupKFold by problem. Train on train_mask rows of each train fold;
    produce OOF probs for ALL rows. Returns dict of OOF prob arrays."""
    y = df.y.values
    groups = df.pkey.values
    oof = {k: np.full(len(df), np.nan) for k in ["word", "char", "struct"]}
    gkf = GroupKFold(n_splits=5)
    for fold, (tr, te) in enumerate(gkf.split(df, y, groups)):
        tr = tr[train_mask[tr]]
        word, char = make_vectorizers()
        Xw_tr = word.fit_transform(df.text.iloc[tr])
        Xw_te = word.transform(df.text.iloc[te])
        Xc_tr = char.fit_transform(df.text.iloc[tr])
        Xc_te = char.transform(df.text.iloc[te])
        S = df[STRUCT_COLS].astype(float).copy()
        S["len_chars"] = np.log1p(S["len_chars"])
        S["n_display_math"] = np.log1p(S["n_display_math"])
        S["n_paragraphs"] = np.log1p(S["n_paragraphs"])
        sc = StandardScaler().fit(S.iloc[tr])
        Xs_tr, Xs_te = sc.transform(S.iloc[tr]), sc.transform(S.iloc[te])
        for key, Xtr, Xte, C in [("word", Xw_tr, Xw_te, 1.0),
                                 ("char", Xc_tr, Xc_te, 1.0),
                                 ("struct", Xs_tr, Xs_te, 1.0)]:
            lr = LogisticRegression(C=C, class_weight="balanced", max_iter=3000,
                                    solver="liblinear" if key == "struct" else "lbfgs")
            lr.fit(Xtr, y[tr])
            oof[key][te] = lr.predict_proba(Xte)[:, 1]
        print(f"  [{label}] fold {fold} done (train n={len(tr)})", flush=True)
    oof["combined"] = (oof["word"] + oof["char"]) / 2
    return oof


def auc_table(df, oof, eval_mask, title):
    print(f"\n### AUC — {title} (n_eval={int(eval_mask.sum())}, "
          f"wiki={int((df.y[eval_mask] == 1).sum())}, forum={int((df.y[eval_mask] == 0).sum())})")
    out = {}
    for k in ["word", "char", "combined", "struct"]:
        out[k] = roc_auc_score(df.y[eval_mask], oof[k][eval_mask])
        print(f"  {k:9s} AUC = {out[k]:.4f}")
    return out


def top_features(df, train_mask, n=25):
    word, _ = make_vectorizers()
    X = word.fit_transform(df.text[train_mask])
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
    lr.fit(X, df.y[train_mask])
    feats = np.array(word.get_feature_names_out())
    order = np.argsort(lr.coef_[0])
    ed = [(feats[i], lr.coef_[0][i]) for i in order[::-1][:n]]
    fo = [(feats[i], lr.coef_[0][i]) for i in order[:n]]
    return ed, fo

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-only", action="store_true")
    args = ap.parse_args()

    df = load_corpus()
    print("\n##### CLEANED SAMPLES (inspect before trusting anything) #####")
    print_samples(df)
    if args.samples_only:
        return

    incl = ~df.near_dup.values  # main-run training mask (near-dups excluded)
    alltrue = np.ones(len(df), bool)

    print("\n[run A] main run: near-dups EXCLUDED from training")
    oofA = run_cv(df, incl, "A")
    aucA = auc_table(df, oofA, incl, "main (near-dups excluded train+eval)")

    print("\n[run B] robustness: near-dups INCLUDED")
    oofB = run_cv(df, alltrue, "B")
    aucB = auc_table(df, oofB, alltrue, "near-dups included (train+eval)")

    # ---- top features (fit on all included data) ----
    ed, fo = top_features(df, incl)
    print("\n### Top 25 word features -> EDITORIAL (wiki)")
    for f, c in ed:
        print(f"  {c:+.2f}  {f}")
    print("\n### Top 25 word features -> FORUM")
    for f, c in fo:
        print(f"  {c:+.2f}  {f}")

    # ---- skew analysis ----
    print("\n### SKEW: OOF P(editorial) for ALL forum posts (run A combined)")
    fmask = df.y.values == 0
    p_forum = oofA["combined"][fmask]
    p_wiki = oofA["combined"][df.y.values == 1]
    wiki_med = np.median(p_wiki)
    dec = np.percentile(p_forum, np.arange(0, 101, 10))
    print("  forum P(editorial) deciles:",
          " ".join(f"{d:.3f}" for d in dec))
    print("  wiki  P(editorial) deciles:",
          " ".join(f"{d:.3f}" for d in np.percentile(p_wiki, np.arange(0, 101, 10))))
    print(f"  wiki median = {wiki_med:.3f}")
    print(f"  frac forum > wiki median : {(p_forum > wiki_med).mean():.4f}")
    print(f"  frac forum > 0.5         : {(p_forum > 0.5).mean():.4f}")
    print(f"  frac forum > 0.9         : {(p_forum > 0.9).mean():.4f}")
    sim = df.sim_word[fmask]
    print("  max-sim (sim_word) deciles:",
          " ".join(f"{d:.3f}" for d in np.percentile(sim.dropna(), np.arange(0, 101, 10))))

    # ---- thanks follow-on ----
    print("\n### FOLLOW-ON: OOF P(editorial) vs thanks_resid")
    th = pd.read_parquet(f"{ROOT}/thanks_deconfounded.parquet",
                         columns=["post_id", "thanks_resid", "is_correct"])
    fdf = df[fmask].copy()
    fdf["p_ed"] = p_forum
    m = fdf.merge(th, on="post_id", how="inner")
    print(f"  merged n={len(m)}")

    def report(sub, name):
        if len(sub) < 30:
            print(f"  {name}: n={len(sub)} too small")
            return
        rho, p = spearmanr(sub.p_ed, sub.thanks_resid)
        print(f"  {name}: pooled rho={rho:+.4f} (p={p:.2g}, n={len(sub)})")
        per = []
        for _, g in sub.groupby("pkey"):
            if len(g) >= 3 and g.p_ed.nunique() > 1 and g.thanks_resid.nunique() > 1:
                per.append(spearmanr(g.p_ed, g.thanks_resid)[0])
        per = np.array([x for x in per if np.isfinite(x)])
        if len(per) > 10:
            try:
                wp = wilcoxon(per).pvalue
            except ValueError:
                wp = np.nan
            print(f"  {name}: within-problem mean rho={per.mean():+.4f} "
                  f"(median {np.median(per):+.4f}, n_problems={len(per)}, "
                  f"wilcoxon p={wp:.2g})")

    dup_ids = set(df.post_id[df.near_dup])
    report(m, "all forum posts")
    report(m[~m.post_id.isin(dup_ids)], "excl near-dups")
    report(m[m.is_correct.apply(lambda v: v is True or v == True)], "verified-correct subset")

    # persist OOF scores for downstream use
    fdf[["pkey", "post_id", "p_ed", "sim_word", "near_dup"]].to_parquet(
        f"{ROOT}/editorial_register_oof.parquet", index=False)
    print(f"\n[saved] {ROOT}/editorial_register_oof.parquet")


if __name__ == "__main__":
    main()
