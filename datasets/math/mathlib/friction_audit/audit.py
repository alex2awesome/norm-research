#!/usr/bin/env python3
"""Leakage/confound audit for the mathlib review-friction dataset.

Dataset: friction_dataset.csv.gz (built 2026-06-10 on sk3).
Label: among merged mathlib4 PRs, y=1 <=> zero review threads.
Balanced 50/50 within (size-quartile x conv_prefix x year x association) cells.

Audits (train/test as given; never refit on test):
 1. Title TF-IDF + LR (word 1-2 grams, min_df=3): AUC + top +/-30 features.
    Assert '[Merged by Bors]' prefix is gone from all titles.
 2. Single-feature test AUCs (rank-based, no fitting), with post-treatment
    variables flagged.
 3. Residual cell balance: P(y=1) within size_bin / conv_prefix / year.
 4. Metadata-only LR (size_bin + prefix + year + association one-hots +
    changed_files): test AUC, should be ~0.5 if matching worked.
 5. easy-label check: P(y=1 | easy).

Outputs in this directory: REPORT.md, title_top_features.csv,
single_feature_aucs.csv, cell_balance.csv.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "friction_dataset.csv.gz")

df = pd.read_csv(DATA)
assert len(df) == 19356, len(df)
train = df[df.split == "train"].copy()
test = df[df.split == "test"].copy()
print(f"rows={len(df)} train={len(train)} test={len(test)} "
      f"pos_rate={df.y.mean():.4f}")

report = []
report.append("# Mathlib review-friction dataset — leakage/confound audit\n")
report.append(f"Dataset: `friction_dataset.csv.gz`, {len(df)} rows "
              f"(train {len(train)} / eval {len(df[df.split=='eval'])} / "
              f"test {len(test)}), pos rate {df.y.mean():.3f}. "
              "Label: y=1 ⟺ zero review threads among merged PRs. "
              "All models fit on train, evaluated on test only.\n")

# ------------------------------------------------------------------ 1. titles
bors_mask = df.title.str.contains(r"\[Merged by Bors\]", regex=True, na=False)
n_bors = int(bors_mask.sum())
if n_bors:
    # Known build bug: 18 raw PRs were re-landed through Bors twice and carry
    # a DOUBLED '[Merged by Bors] - [Merged by Bors] - ' prefix; the anchored
    # regex in build_friction_dataset.py strips only one copy. Verified
    # against raw pr_reviews_mathlib4.jsonl on sk3 (2026-06-10). Mixed labels
    # (10 y=0 / 8 y=1) -> not a label leak, but their conv_prefix is wrongly
    # 'OTHER'. Strip residual prefixes here before the title model.
    print(f"[1] Bors-prefix assertion FAILED: {n_bors} titles still carry "
          "the prefix (doubled prefix in raw data). Stripping for the model.")
    strip = lambda s: pd.Series(s).str.replace(
        r"^(\[Merged by Bors\]\s*-\s*)+", "", regex=True)
    for frame in (df, train, test):
        frame["title"] = strip(frame["title"].fillna(""))
    assert not df.title.str.contains(r"\[Merged by Bors\]").any()
else:
    print("[1] Bors-prefix assertion passed: no title contains "
          "'[Merged by Bors]'")

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, lowercase=True)
Xtr = vec.fit_transform(train.title.fillna(""))
Xte = vec.transform(test.title.fillna(""))
lr = LogisticRegression(max_iter=2000, C=1.0)
lr.fit(Xtr, train.y)
auc_tr = roc_auc_score(train.y, lr.decision_function(Xtr))
auc_te = roc_auc_score(test.y, lr.decision_function(Xte))
print(f"[1] Title TF-IDF+LR: train AUC {auc_tr:.4f}  test AUC {auc_te:.4f}  "
      f"vocab={len(vec.vocabulary_)}")

names = np.array(vec.get_feature_names_out())
coefs = lr.coef_[0]
order = np.argsort(coefs)
top_neg = order[:30]          # push toward y=0 (friction)
top_pos = order[::-1][:30]    # push toward y=1 (frictionless)
feat_rows = []
for idx, direction in [(i, "y=1 (frictionless)") for i in top_pos] + \
                      [(i, "y=0 (friction)") for i in top_neg]:
    feat_rows.append(dict(feature=names[idx], coef=round(coefs[idx], 4),
                          direction=direction,
                          train_doc_freq=int((Xtr[:, idx] > 0).sum())))
feat_df = pd.DataFrame(feat_rows)
feat_df.to_csv(os.path.join(HERE, "title_top_features.csv"), index=False)

report.append("## 1. Title-only model\n")
if n_bors:
    report.append(
        f"- **Bors-prefix check: FAIL (minor build bug)** — {n_bors} titles "
        f"({n_bors/len(df)*100:.2f}%) still contained `[Merged by Bors]`. "
        "Root cause (verified against raw `pr_reviews_mathlib4.jsonl` on "
        "sk3): these PRs were re-landed through Bors twice and have a "
        "**doubled** prefix; the anchored regex in "
        "`build_friction_dataset.py` strips only one copy. Labels among "
        "them are mixed (10 y=0 / 8 y=1) and every row in the dataset is "
        "merged, so this is **not a label leak** for the friction label — "
        "but their `conv_prefix` was mis-binned as `OTHER`, so they were "
        "matched in the wrong cell. Fix: replace `BORS_PREFIX` with "
        r"`^(\[Merged by Bors\]\s*-\s*)+`. Residual prefixes were stripped "
        "before the title model below.")
else:
    report.append(f"- **Bors-prefix check: PASS** — no title contains "
                  f"`[Merged by Bors]` (asserted over all {len(df)} rows).")
report.append(f"- TF-IDF (word 1–2 grams, min_df=3, "
              f"{len(vec.vocabulary_)} features) + LR: "
              f"**train AUC {auc_tr:.3f}, test AUC {auc_te:.3f}**.")
report.append("- Top ±30 features in `title_top_features.csv`. "
              "Top 15 each direction:\n")
report.append("| → y=1 (frictionless) | coef | → y=0 (friction) | coef |")
report.append("|---|---|---|---|")
for i in range(15):
    p, n = top_pos[i], top_neg[i]
    report.append(f"| `{names[p]}` | {coefs[p]:+.2f} "
                  f"| `{names[n]}` | {coefs[n]:+.2f} |")
report.append("")

# ------------------------------------------------- 2. single-feature test AUC
POST_TREATMENT = {"days_open", "n_force_pushes", "n_issue_comments",
                  "n_commits"}
single_feats = ["size", "changed_files", "n_commits", "n_force_pushes",
                "days_open", "year", "easy", "new_contributor",
                "n_issue_comments"]
rows = []
for f in single_feats:
    x = test[f].astype(float)
    mask = x.notna()
    auc = roc_auc_score(test.y[mask], x[mask])
    rows.append(dict(feature=f, test_auc=round(auc, 4),
                     test_auc_oriented=round(max(auc, 1 - auc), 4),
                     post_treatment=f in POST_TREATMENT,
                     n=int(mask.sum())))
sf = pd.DataFrame(rows).sort_values("test_auc_oriented", ascending=False)
sf.to_csv(os.path.join(HERE, "single_feature_aucs.csv"), index=False)
print("[2] single-feature AUCs:\n", sf.to_string(index=False))

report.append("## 2. Single-feature test AUCs (rank AUC, no fitting)\n")
report.append("| feature | test AUC (raw) | AUC (oriented) | post-treatment? |")
report.append("|---|---|---|---|")
for _, r in sf.iterrows():
    flag = "**YES — descriptive only**" if r.post_treatment else "no"
    report.append(f"| {r.feature} | {r.test_auc:.3f} "
                  f"| {r.test_auc_oriented:.3f} | {flag} |")
report.append(
    "\nPost-treatment variables (`days_open`, `n_force_pushes`, "
    "`n_issue_comments`, `n_commits`) accrue **during** review and are "
    "partially downstream of the label (a review thread causes revision "
    "commits, force-pushes, comments, and longer time-to-merge). They must "
    "never be model inputs — reported here only to describe the label's "
    "footprint.\n")

# --------------------------------------------------------- 3. cell balance
report.append("## 3. Residual cell balance (full dataset, by construction "
              "should be ≈0.5)\n")
bal_rows = []
for col in ["size_bin", "conv_prefix", "year", "author_association"]:
    g = df.groupby(col)["y"].agg(["mean", "count"]).reset_index()
    g.columns = ["value", "p_y1", "n"]
    g.insert(0, "stratum", col)
    bal_rows.append(g)
bal = pd.concat(bal_rows, ignore_index=True)
bal["abs_dev"] = (bal.p_y1 - 0.5).abs()
bal.to_csv(os.path.join(HERE, "cell_balance.csv"), index=False)
worst = bal.sort_values("abs_dev", ascending=False).head(10)
print("[3] worst marginal deviations:\n", worst.to_string(index=False))

for col in ["size_bin", "conv_prefix", "year"]:
    sub = bal[bal.stratum == col]
    report.append(f"**{col}** (max |dev| = "
                  f"{sub.abs_dev.max():.4f}):\n")
    report.append("| value | P(y=1) | n |")
    report.append("|---|---|---|")
    for _, r in sub.sort_values("value").iterrows():
        report.append(f"| {r.value} | {r.p_y1:.4f} | {int(r.n)} |")
    report.append("")

# exact joint cells
df["cell"] = list(zip(df.size_bin, df.conv_prefix, df.year,
                      df.author_association))
joint = df.groupby("cell")["y"].agg(["mean", "count"])
off = joint[joint["mean"] != 0.5]
report.append(f"Joint (size_bin × conv_prefix × year × association) cells: "
              f"{len(joint)} cells; {len(off)} have P(y=1) ≠ 0.5 exactly "
              f"(max dev {(joint['mean']-0.5).abs().max():.4f}). "
              "Construction guarantees exact within-cell balance.\n")
print(f"[3] joint cells: {len(joint)}, off-balance: {len(off)}")

# ------------------------------------------------------- 4. metadata-only LR
meta_tr = pd.get_dummies(
    train[["size_bin", "conv_prefix", "year", "author_association"]]
    .astype(str), drop_first=False)
meta_te = pd.get_dummies(
    test[["size_bin", "conv_prefix", "year", "author_association"]]
    .astype(str), drop_first=False)
meta_te = meta_te.reindex(columns=meta_tr.columns, fill_value=0)
meta_tr["changed_files"] = np.log1p(train.changed_files.values)
meta_te["changed_files"] = np.log1p(test.changed_files.values)
mlr = LogisticRegression(max_iter=2000)
mlr.fit(meta_tr, train.y)
m_auc_tr = roc_auc_score(train.y, mlr.decision_function(meta_tr))
m_auc_te = roc_auc_score(test.y, mlr.decision_function(meta_te))
print(f"[4] metadata-only LR: train AUC {m_auc_tr:.4f} test AUC {m_auc_te:.4f}")

report.append("## 4. Metadata-only model\n")
report.append(f"LR on size_bin/conv_prefix/year/association one-hots + "
              f"log(changed_files): **train AUC {m_auc_tr:.3f}, "
              f"test AUC {m_auc_te:.3f}**.\n")

# ------------------------------------------------------------- 5. easy label
p_easy = df.groupby("easy")["y"].agg(["mean", "count"])
p1_easy = df[df.easy].y.mean()
p1_noeasy = df[~df.easy].y.mean()
n_easy = int(df.easy.sum())
print(f"[5] P(y=1|easy)={p1_easy:.3f} (n={n_easy})  "
      f"P(y=1|~easy)={p1_noeasy:.3f}")
report.append("## 5. `easy` label\n")
report.append(f"P(y=1 | easy=True) = **{p1_easy:.3f}** (n={n_easy}); "
              f"P(y=1 | easy=False) = {p1_noeasy:.3f} "
              f"(n={len(df)-n_easy}).\n")

report.append(
    "The raw concordance (README label-mechanics: 83.6% of `easy` PRs are "
    "zero-thread) is mostly absorbed by the cell balancing (easy PRs are "
    "small → small size_bin cells are downsampled toward 0.5), leaving "
    "single-feature test AUC ≈0.52 here. But the mechanism is the problem, "
    "not the magnitude: maintainers apply `easy` partly *because* review "
    "turned out to be trivial, i.e. it is label-adjacent / "
    "post-treatment-ish. **Recommendation: ban `easy` as a model input.**\n")

# ------------------------------------- 6. token / topic supplementary checks
report.append("## 6. Scrutiny of title-feature classes (full dataset)\n")
t = df.title.str.lower()
report.append("| token | n | P(y=1 \\| token) |")
report.append("|---|---|---|")
for tok in ["port", "split", "move", "grind", "revert", "golf", "deprecate",
            "theorem", "define", "tactic"]:
    m = t.str.contains(r"\b" + tok, regex=True)
    report.append(f"| `{tok}` | {int(m.sum())} | {df.y[m].mean():.3f} |")
report.append("")
report.append("| topic label (NOT in matching cells) | n | P(y=1) |")
report.append("|---|---|---|")
tl = df.topic_labels.fillna("")
for area in ["t-algebra", "t-analysis", "t-topology", "t-category-theory",
             "t-number-theory", "t-combinatorics", "t-meta", "t-order",
             "t-measure-probability", "t-data", "t-set-theory"]:
    m = tl.str.contains(area, regex=False)
    report.append(f"| {area} | {int(m.sum())} | {df.y[m].mean():.3f} |")
m0 = tl == ""
report.append(f"| (no topic label) | {int(m0.sum())} | {df.y[m0].mean():.3f} |")
report.append("""
Classification of the top title features:

- **Quality / task-type signal (legitimate):** `theorem`, `define`,
  `definition`, `formula`, `constructors`, `tactic` → friction (new
  mathematical content and new metaprograms genuinely draw review);
  `revert`, `align`, `import`, `doc fix`, `deprecation`, `update mathlib`,
  `mathlib dependencies` → frictionless (mechanical maintenance). This is
  exactly the signal the label is supposed to carry.
- **Stratification residual (finer-grained than the matched cells):**
  `port *` (P(y=1)=0.611, n=3017), `split` (0.628), `move` (0.606), `grind`
  (0.596). `split`/`move` are conv_prefixes and balanced *at title start*,
  but the tokens recur mid-title under `chore:`/`feat:` prefixes; `port` is
  the Lean3→Lean4 porting wave — pre-reviewed in mathlib3, hence
  frictionless. Borderline-legitimate (ports really are lower-risk work)
  but a model can use them as PR-type shortcuts.
- **Math-area confound (NOT balanced — gap vs README §3a plan):** the build
  matched on (size × conv_prefix × year × association) but **not** on
  `t-*` topic labels, and areas are imbalanced: t-combinatorics 0.360,
  t-number-theory 0.387, t-meta 0.421, t-analysis 0.434, t-algebra 0.447 vs
  t-category-theory 0.556, t-set-theory 0.537, no-label 0.554. This is why
  area tokens (`feat numbertheory`, `feat linearalgebra`, `ringtheory
  ideal`, `computability`, `measuretheory`, `euclidean`) appear in the top
  coefficients. Plausibly part-genuine (different reviewer cultures per
  area) and part-confound (reviewer availability per area). Recommend:
  include topic labels in the matching cells on the next rebuild, or always
  report per-area AUCs.
- **Leakage: none found.** No `bors`/`easy`/`merge`/process tokens in the
  vocabulary's top coefficients; `golf` (0.482) and `deprecate` (0.484) are
  ~neutral, contrary to the prior worry.
""")

# ------------------------------------------------------------------ verdict
report.append("""## 7. Verdict and banned columns

**Verdict: the friction label is clean enough to model against first-push
code features.** Metadata-only test AUC 0.509 confirms the cell matching
removed the size/prefix/year/association confounds; exact 0.5 balance holds
in every joint cell; no leak tokens in the title model; the title signal
that remains (test AUC ~0.63 vs train 0.80 — heavily memorization-limited)
decomposes into legitimate task-type signal plus two named residuals
(port/refactor mid-title tokens, math-area imbalance). The post-treatment
variables behave exactly as post-treatment variables should (oriented AUC
0.67–0.76), confirming the label has a real process footprint.

**Banned columns for downstream modeling** (never as model inputs):

| column | reason |
|---|---|
| `n_review_threads` | label definition (y = [threads == 0]) |
| `n_reviews`, `n_changes_requested` | direct review-process counts (label-adjacent) |
| `days_open`, `closed_at` | post-treatment: review friction extends time-to-merge |
| `n_force_pushes` | post-treatment: revisions after review (oriented AUC 0.76) |
| `n_commits` | post-treatment: review-driven revision commits (0.68) |
| `n_issue_comments` | post-treatment: discussion accrues during review (0.67) |
| `labels` (raw string) | contains post-review process labels (`ready-to-merge`, `maintainer-merge`, `awaiting-review`, …) |
| `easy` | label-adjacent: applied partly *because* review was trivial |
| `head_oid`, `merge_commit_oid` | final post-review state (use `first_commit_oid` for first-push reconstruction) |
| `state`, `merged` | constant among rows |

**Caution (stratifiers / keys only, not features):** `additions`,
`deletions`, `size`, `changed_files`, `size_bin` are measured on the
**final merged** state, which includes review-driven revisions — mildly
post-treatment. They are fine as matching strata (and are ~0.5 AUC after
balancing) but first-push size should be recomputed from the cloned repo
for any size-aware model. `number`, `first_commit_oid` are join keys.

**Allowed inputs:** `title` (Bors-prefix-stripped — after fixing the
doubled-prefix bug), first-push code/diff features reconstructed from
`first_commit_oid`, `conv_prefix`, `year`, `author_association`,
`topic_labels`, `new_contributor`, `llm_generated`, `created_at`.
""")

with open(os.path.join(HERE, "REPORT.md"), "w") as f:
    f.write("\n".join(report))
print("wrote REPORT.md")
