"""
Editorial-similarity pilot: how close is each forum solution to the wiki
("editorial") canonical solutions for the same problem, and does that
closeness predict community thanks?

For each forum solution: TF-IDF cosine vs each wiki solution of the same
problem, max over wiki solutions. Two featurizations: char_wb 3-5 grams and
word 1-2 grams (both on markup-normalized text, quotes stripped).

Analyses:
  a. similarity distribution overall + by contest
  b. similarity vs thanks: pooled Spearman + within-problem Spearman
     (problems with >= 3 forum solutions)
  c. AIME/AMC answer extraction -> correctness; similarity & thanks by
     correctness; within-correct-only within-problem Spearman ("taste
     within correct")

Outputs:
  editorial_similarity.parquet  (forum_solutions + sim_char, sim_word,
                                 extracted_answer, is_correct)
  stdout tables for the notes file.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from build_forum_wiki_join import norm_text  # noqa: E402

KEY = ["contest", "year", "variant", "problem_n"]
MC_CONTESTS = {"AMC8", "AMC10", "AMC12"}

BOXED_PAT = re.compile(r"\\(?:boxed|framebox|fbox)\s*\{")
LETTER_PAT = re.compile(r"\(?\s*([A-E])\s*\)?\s*$")
ANS_LETTER_ANY = re.compile(r"(?:answer|ans)[^A-Za-z0-9]{0,12}\(?([A-E])\)?", re.I)
INT_PAT = re.compile(r"\b(\d{1,3})\b")


def extract_boxed(text: str) -> str | None:
    """Return content of the LAST \\boxed{...} (brace-balanced)."""
    last = None
    for m in BOXED_PAT.finditer(text):
        i = m.end()
        depth = 1
        j = i
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            last = text[i:j - 1]
    return last


def extract_answer(text: str, contest: str) -> str | None:
    """Final answer from a forum solution: \\boxed first, then fallbacks."""
    t = text or ""
    boxed = extract_boxed(t)
    if contest in MC_CONTESTS:
        for src in ([boxed] if boxed else []) + [t[-400:]]:
            # letter inside boxed / trailing region, e.g. \textbf{(C) }5, (C)
            m = re.findall(r"\(\s*([A-E])\s*\)", src)
            if m:
                return m[-1]
        m = ANS_LETTER_ANY.findall(t)
        if m:
            return m[-1]
        return None
    # AIME: integer 0-999
    if boxed:
        m = INT_PAT.findall(boxed)
        if m:
            return f"{int(m[-1]):03d}"
    tail = re.sub(r"\[asy\].*?\[/asy\]", " ", t[-300:], flags=re.S | re.I)
    m = INT_PAT.findall(tail)
    if m:
        return f"{int(m[-1]):03d}"
    return None


def per_problem_spearman(df: pd.DataFrame, xcol: str, ycol: str,
                         min_n: int = 3) -> pd.Series:
    out = {}
    for k, g in df.groupby(KEY):
        if len(g) < min_n or g[xcol].nunique() < 2 or g[ycol].nunique() < 2:
            continue
        rho = spearmanr(g[xcol], g[ycol]).statistic
        if np.isfinite(rho):
            out[k] = rho
    return pd.Series(out)


def summarize_rhos(rhos: pd.Series, label: str) -> None:
    if not len(rhos):
        print(f"{label}: no eligible problems")
        return
    from scipy.stats import wilcoxon
    w = wilcoxon(rhos, alternative="two-sided")
    print(f"{label}: n_problems={len(rhos)} mean_rho={rhos.mean():.3f} "
          f"median_rho={rhos.median():.3f} frac>0={(rhos > 0).mean():.3f} "
          f"wilcoxon_p={w.pvalue:.2e}")


def main() -> int:
    sol = pd.read_parquet(REPO / "forum_solutions.parquet")
    wiki = pd.read_parquet(REPO / "wiki" / "solutions.parquet")
    keys = pd.read_parquet(REPO / "wiki" / "answer_keys.parquet")

    sol["norm"] = sol.body_noquote.map(norm_text)
    wiki = wiki[~wiki.is_redirect.fillna(False)].copy()
    wiki["norm"] = wiki.solution_text.map(norm_text)
    wiki = wiki[wiki.norm.str.len() >= 50]

    # restrict to problems that have >=1 usable wiki solution
    wkeys = wiki[KEY].drop_duplicates()
    sol = sol.merge(wkeys.assign(_w=1), on=KEY, how="left")
    print(f"forum solutions: {len(sol)}; with wiki solutions: "
          f"{int(sol._w.fillna(0).sum())}")
    sol = sol[sol._w == 1].drop(columns="_w").reset_index(drop=True)

    corpus = pd.concat([sol.norm, wiki.norm]).values
    sims = {}
    for name, kw in [
            ("char", dict(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                          sublinear_tf=True, max_features=400_000)),
            ("word", dict(analyzer="word", ngram_range=(1, 2), min_df=3,
                          sublinear_tf=True))]:
        vec = TfidfVectorizer(**kw)
        vec.fit(corpus)
        Xf = normalize(vec.transform(sol.norm))
        Xw = normalize(vec.transform(wiki.norm))
        # group wiki rows by problem
        wiki_idx = wiki.reset_index(drop=True).groupby(KEY).indices
        out = np.full(len(sol), np.nan)
        for k, fidx in sol.groupby(KEY).indices.items():
            widx = wiki_idx.get(k)
            if widx is None:
                continue
            S = (Xf[fidx] @ Xw[widx].T).toarray()
            out[fidx] = S.max(axis=1)
        sims[name] = out
        print(f"sim_{name} computed; nan={np.isnan(out).sum()}")
    sol["sim_char"], sol["sim_word"] = sims["char"], sims["word"]

    # ---------------- a. distributions -----------------------------------
    print("\n=== (a) similarity distribution ===")
    for c in ["sim_char", "sim_word"]:
        print(f"\n{c} overall: {sol[c].describe().round(3).to_dict()}")
        print(sol.groupby("contest")[c]
              .describe()[["count", "mean", "25%", "50%", "75%"]]
              .round(3).to_string())
    print(f"\ncorr(sim_char, sim_word) = "
          f"{spearmanr(sol.sim_char, sol.sim_word).statistic:.3f}")

    # ---------------- b. similarity vs thanks ----------------------------
    print("\n=== (b) similarity vs thanks ===")
    sol["thanks"] = sol.thanks_received.fillna(0)
    for c in ["sim_char", "sim_word"]:
        r = spearmanr(sol[c], sol.thanks)
        print(f"pooled Spearman {c} vs thanks: rho={r.statistic:.3f} "
              f"p={r.pvalue:.2e} (n={len(sol)})")
        rhos = per_problem_spearman(sol, c, "thanks")
        summarize_rhos(rhos, f"within-problem {c} vs thanks")
        print(rhos.rename("rho").reset_index()
              .groupby("level_0").rho.agg(["count", "mean", "median"])
              .round(3).to_string())

    # control: post order strongly drives thanks (early posts accumulate)
    rhos_pn = per_problem_spearman(sol, "post_number", "thanks")
    summarize_rhos(rhos_pn, "within-problem post_number vs thanks (control)")

    # ---------------- c. correctness-conditioned (AIME/AMC) --------------
    print("\n=== (c) correctness-conditioned (AIME/AMC) ===")
    ans = sol[sol.contest.isin(MC_CONTESTS | {"AIME"})].copy()
    ans["extracted"] = [extract_answer(t, c) for t, c
                        in zip(ans.post_canonical, ans.contest)]
    ans = ans.merge(keys[keys.is_valid][KEY + ["answer"]], on=KEY, how="inner")
    ans["has_answer"] = ans.extracted.notna()
    ans["is_correct"] = ans.extracted == ans.answer
    print(f"forum solutions on answerable problems: {len(ans)}; "
          f"answer extracted: {ans.has_answer.mean():.2%}; "
          f"of those correct: "
          f"{ans[ans.has_answer].is_correct.mean():.2%}")
    print(ans.groupby("contest")[["has_answer", "is_correct"]]
          .mean().round(3).to_string())

    sub = ans[ans.has_answer].copy()
    print("\nsimilarity by correctness:")
    print(sub.groupby("is_correct")[["sim_char", "sim_word", "thanks"]]
          .agg(["mean", "median", "count"]).round(3).to_string())
    for c in ["sim_char", "sim_word"]:
        u = mannwhitneyu(sub.loc[sub.is_correct, c],
                         sub.loc[~sub.is_correct, c])
        print(f"Mann-Whitney {c} correct>incorrect: p={u.pvalue:.2e}")

    print("\nwithin-problem, CORRECT solutions only (taste-within-correct):")
    cor = sub[sub.is_correct]
    for c in ["sim_char", "sim_word"]:
        r = spearmanr(cor[c], cor.thanks)
        print(f"pooled Spearman {c} vs thanks | correct: "
              f"rho={r.statistic:.3f} p={r.pvalue:.2e} (n={len(cor)})")
        summarize_rhos(per_problem_spearman(cor, c, "thanks"),
                       f"within-problem {c} vs thanks | correct")

    print("\nthanks by correctness (within-problem rank of correct vs not):")
    rhos_c = per_problem_spearman(sub.assign(corr=sub.is_correct.astype(int)),
                                  "corr", "thanks")
    summarize_rhos(rhos_c, "within-problem correctness vs thanks")

    out = REPO / "editorial_similarity.parquet"
    full = sol.merge(
        ans[["topic_id", "post_id"] + KEY +
            ["extracted", "answer", "has_answer", "is_correct"]],
        on=["topic_id", "post_id"] + KEY, how="left")
    full.drop(columns=["norm"]).to_parquet(out, index=False)
    print(f"\nwrote {out} ({len(full)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
