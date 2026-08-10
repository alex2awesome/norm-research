# Manual score audit: why the dense text model shows no taste residual on GitHub-PR-merge

Date: 2026-08-05
Cell audited: code-review / PR-merge, VAT (Verifiability / Articulability / Taste) decomposition
Instrument: Llama-3.1-8B LoRA reward model, `dense_standard_v2`, select-on-eval
Artifacts: `sk3:/lfs/skampere3/0/alexspan/norm-research/datasets/code-review/dense_standard_v2/audit_manual/`
(`audit1.py`–`audit6.py`, `per_repo_eval.csv`, `per_repo_tfidf.csv`, `neardup_pairs.csv`, `examples_20.txt`, `eval_feats.pkl`, `tfidf.log`)

Reference numbers taken as given (not re-derived): clean-eval AUC .5734 (seed 42), test AUC .7035, articulated baseline V+A = .592, eval/test divergence = repo composition, corpus n=63,707 / 1,267 repos / repo-disjoint grouped splits.

---

## Headline

**The dense model is not broken — the input field is.** The `text` column is a **raw unified diff and nothing else**: 100% of eval rows begin with `diff --git`, only 0.5% contain `Title:` and 1.3% contain `Description:`. There is no PR title, no PR body, no review thread, no CI status, no author, no timestamp. On top of that, the merge label for a large and identifiable slice of the corpus (dependency/CI bumps) is decided by **supersession and timing**, which are provably not in the text: on the very subset where the model's most confident false positives live, PR number (pure recency metadata) reaches AUC **.806** while the dense text model reaches **.339**.

Score distribution is healthy, truncation is enormous but demonstrably non-binding, and the honest within-repo dense level is **~.59**, i.e. numerically indistinguishable from V+A = .592.

---

## 1. Score-distribution sanity — NO collapse

Eval n = 6,372; 117 repos; base rate merged = **.8074**.

| stat | seed42 prob |
|---|---|
| mean / sd | .6018 / .2046 |
| min / max | .0002 / 1.0000 |
| q01 / q05 / q25 | .0465 / .1981 / .4774 |
| median | .6388 |
| q75 / q95 / q99 | .7535 / .8758 / .9950 |

| class | n | mean prob | median |
|---|---|---|---|
| rejected (0) | 1,227 | .5650 | .6086 |
| merged (1) | 5,145 | .6106 | .6469 |

The score occupies the full unit interval, is unimodal-ish with a mode near .70, and separates the classes by only **.046** in mean. Three seeds agree (eval AUC .5734 / .5535 / .5713; pairwise Pearson r ≈ .74; 3-seed ensemble .5715 — ensembling buys nothing). **H3 (degenerate/collapsed predictions) is rejected.**

Calibration is the real story — the score is nearly flat against outcome across the bottom 80% of its range:

| decile | n | mean prob | observed merge rate |
|---|---|---|---|
| 0 | 640 | .187 | **.791** |
| 1 | 636 | .377 | .747 |
| 2 | 636 | .475 | .807 |
| 3 | 640 | .551 | .788 |
| 4 | 637 | .613 | .779 |
| 5 | 642 | .662 | .726 |
| 6 | 637 | .708 | .809 |
| 7 | 635 | .754 | .822 |
| 8 | 634 | .806 | .891 |
| 9 | 635 | .891 | .918 |

Deciles 0–7 all sit at or below the .807 base rate. The entire AUC comes from the top two deciles — and see §6: those are dominated by two all-positive repos.

---

## 2. Nuisance correlations — truncation is huge, and irrelevant

Text length in eval: median **24,000 chars** (the build's `char_cap`), mean 16,002, sd 9,388.

- **50.1%** of eval rows sit exactly at the 24,000-char build cap.
- **72.4%** exceed ~7,168 chars ≈ 2,048 Llama tokens, i.e. are clipped by the model window.
- For the median example the model sees roughly **30% of an already-capped diff**.

Yet length carries almost no signal and the model is not using it:

| quantity | value |
|---|---|
| corr(prob, nchar) Pearson / Spearman | **−.034 / −.066** |
| corr(prob, log nchar) | −.015 |
| AUC of char-length alone (pooled eval) | **.5185** |
| AUC of truncated-flag alone | .5118 |

**Dense AUC by truncation status:**

| subset | n | pos rate | dense AUC | length AUC |
|---|---|---|---|---|
| not truncated (<2,048 tok) | 1,758 | .794 | **.5735** | .4479 |
| truncated (>2,048 tok) | 4,614 | .813 | **.5739** | .5210 |

The two are identical to the third decimal. **When the model sees the entire text it does no better.** H2 (truncation) is real in magnitude but is not the binding constraint. (Caveat: the untruncated subset is by construction the short-diff population, so this is a strong hint rather than a randomized test.)

Structural diff features are equally inert (pooled eval AUC):

| feature | AUC | corr with prob |
|---|---|---|
| n_files | .5150 | −.034 |
| n_add | .5279 | +.030 |
| n_del | .4913 | −.114 |
| touches_test | .4643 | +.099 |
| touches_doc | .5092 | −.005 |
| new_file | .5246 | −.015 |
| nchar | .5185 | −.034 |

Repo composition, by contrast, absorbs a great deal: **45.1%** of the model's score variance is between-repo, versus only **30.5%** of label variance. The model spends nearly half its output range on repo-level offsets that carry less than a third of the outcome variance — and since splits are repo-disjoint, none of that transfers.

---

## 3. Per-repo decomposition — dense does beat length, by a little

55 eval repos have n ≥ 30 and both classes (5,070 of 6,372 rows).

| predictor | median AUC | n-wtd mean | IQR | frac > .5 |
|---|---|---|---|---|
| **dense (seed42)** | **.5986** | **.5881** | [.484, .670] | .709 |
| dense (seed1) | .5722 | .5851 | — | .745 |
| dense (seed2) | .6049 | .5972 | — | .782 |
| TF-IDF + LR | .5469 | .5674 | — | .691 |
| char length | .4730 | .4692 | [.418, .529] | .345 |
| n_files | .4873 | .4946 | — | .491 |
| n_add | .5108 | .5025 | — | .545 |
| touches_test | .5470 | .5455 | — | .709 |

Dense beats length in **41/55** repos (Wilcoxon p < 1e-4). Pooled repo-centered AUC: dense **.5693**, length **.4830**. So the dense score is genuinely doing something above surface size — but the honest within-repo level is **~.59**, which is *higher* than the pooled eval .5734 (repo composition is depressing the pooled number, see §6) and is **the same as V+A = .592**.

The per-repo spread is mostly noise, not heterogeneity:

- observed sd of within-repo dense AUC = **.1521**
- mean permutation-null sd = **.1174**
- implied excess "signal" sd = **.0837**

Cross-seed correlation of the per-repo AUC vector is only .54–.78, confirming that repo-level AUCs like pymor .971 or pyteal .238 are largely draws, not findings.

**TF-IDF is the most damning reference point.** A bag-of-words logistic on the first 7,168 chars gets pooled eval AUC **.4881** (below chance) and within-repo median **.5469** — only ~.05 below the 8B LoRA. And its learned vocabulary is almost purely repo identity:

- top merge-predictive tokens: `armeria, linecorp, azkaban, aqua, dingo, jina, beego, owasp, cloudflare, cookiefarm, cylc, keanu`
- top reject-predictive tokens: `bouncycastle, infinispan, hive, artemis, hadoop, coroot, onflow, discord, prometheus, hibernate`

These are project names. A text model trained on this field learns *which repo this is*, and repo identity does not survive a repo-disjoint split. That is exactly the sub-chance pooled AUC.

---

## 4. Label-noise probe

Line-level shingle Jaccard within repo (n ≥ 20 repos), 674 near-duplicate pairs found:

| threshold | n pairs | opposite-label fraction |
|---|---|---|
| jac ≥ .5 | 674 | .129 |
| jac ≥ .6 | 556 | .128 |
| jac ≥ .7 | 488 | .113 |
| jac ≥ .8 | 390 | .118 |
| jac ≥ .9 | 240 | **.096** |
| chance (random within-repo pair) | — | .216 |

So near-duplicate diffs are more label-consistent than random pairs, but **~10% of near-identical diffs carry opposite labels** — a hard irreducible error floor for any text-only model.

Exact-text duplication is small: 6,293 unique texts in 6,372 rows; **17 duplicate groups spanning 35 rows (0.5% of eval) carry both labels**. Concentrated in `xr-ai` (16 rows, 8/8 dup groups both-label, with a systematic pr_number +10 offset — PRs 3/13, 4/14, 6/16, 7/17, 9/19, 11/21, 12/22), `proto-fleet`, `app-bitcoin-legacy`, `android-maps-utils`. Train has the same pattern at similar low rate (35 both-label groups / 101 rows in a 20k sample). Real, worth flagging for corpus hygiene, but **too small to explain the result**.

---

## 5. MANUAL READING — 20 examples

Full text in `audit_manual/examples_20.txt`.

### A. Five highest-prob MERGED (prob .9999–1.0000)

**All five are the same repo, `nomad_m_deep_003`** (2015-era HashiCorp Nomad), and four of five touch `website/source/docs/*.html.md`:

| # | files | what the text is |
|---|---|---|
| 97 | `qemu_test.go`, `qemu.html.md` | reorders a test config key; fills in a `TODO` docs page with prose driver documentation |
| 95 | `java.go`, `java.html.md` | deletes one blank line; fills in a `TODO` docs page |
| 160 | `install.html.md` | one-line URL fix, github.com → raw.githubusercontent.com |
| 110 | 4 × `*.html.md` | trailing-whitespace and typo fixes ("Named makes use" → "Nomad makes use") |
| 101 | `qemu.go`, `qemu_test.go`, `structs.go`, `qemu.html.md` | real feature: port-mapping via task Resources |

The unifying feature is not merge-worthiness — it is **repo/era/style signature**: Go + HashiCorp-flavored Markdown prose. This repo is 100% merged (see §6), so the model can be maximally confident and be right by construction, learning nothing transferable.

### B. Five highest-prob REJECTED (prob .90–.91) — the confident false positives

**Four of five are `android-maps-utils` Dependabot-style CI bumps**, and they are nearly identical to each other:

| # | change |
|---|---|
| 1381 | `actions/setup-java@v4.2.1 → v4.2.2` across docs.yml / release.yml / test.yml |
| 1389 | `actions/setup-java@v4.2.1 → v4.3.0`, same three files |
| 1394 | `actions/setup-java@v4.2.1 → v4.4.0`, same three files |
| 1368 | `gradle/wrapper-validation-action@v3.3.1 → v3.4.2`, same three files |
| mmsegmentation 1388 | docs URL fix + `md2yml.py` path-separator fix + zh/en doc restructure |

These were **closed, not merged, because a later bump superseded them**. The text of a superseded bump is byte-for-byte the same shape as the text of the bump that did merge. Nothing in the diff could ever separate them. The fifth (mmsegmentation) is a benign docs/typo PR that reads exactly like a merge.

### C. Five lowest-prob MERGED (prob .0002–.0016) — the confident false negatives

| # | files | what the text is |
|---|---|---|
| vikunja 3109 | `go.mod`, `go.sum` | `go-mail v0.7.3 → v0.8.0` — a dependency bump. **Merged.** |
| vikunja 3143 | `go.mod`, `go.sum` | `go-mail v0.8.0 → v0.8.1` — a dependency bump. **Merged.** |
| klaytn 182 | `consensus/istanbul/doc.go` | replaces a one-line package comment with a package-doc block |
| klaytn 183 | `consensus/gxhash/doc.go` | same, 5 added lines |
| klaytn 108 | `consensus/doc.go` | same, 11 added lines |

**This is the crux.** Sets B and C are *the same kind of object with opposite labels*. The model's most confident predictions in **both directions** are dependency/CI-manifest edits and boilerplate doc-comment additions. The bump feature is completely non-discriminative; the model has simply attached an arbitrary repo-specific sign to it.

### D. Five nearest prob = .5

Four of five are `rs` (Runestone): substantive feature work — peer-instruction LLM mode routing with theme logging (#1245, 368 added lines), face-chat group broadcasting (#738), a 16-file refactor removing exercise descriptions (#697). The fifth is `hazelcast` #5040, a 10-file 24,000-char checkstyle + docs + blackbox-sensor change. All were real code review objects, and the model is **at exactly chance on them**.

### Synthesis

The model is keying on **repo/language/genre signature** — "this looks like HashiCorp Go docs", "this looks like an Android CI yaml" — plus a weak, repo-specific sign attached to manifest-shaped diffs. It is not keying on anything a reviewer would call quality. TF-IDF's learned vocabulary (project names) confirms this independently, and the near-.5 mass is precisely the substantive feature diffs where merge-worthiness would actually have to be judged.

What is **absent from the text** and plausibly decides the label: (1) whether a bot PR was superseded by a later one; (2) CI pass/fail; (3) the PR title and description stating intent; (4) the review conversation and requested changes; (5) author identity/maintainer status; (6) release-branch timing and freeze windows; (7) whether the change was instead landed via squash/rebase under a different PR. Items 1–2 alone govern the ~12% of the corpus that is bump-shaped, and item 1 is measurable (§6).

---

## 6. Supporting evidence for the verdict

### Supersession is real and beats the text badly

Within `android-maps-utils`, restricting to the 57 `.github/workflows` bump PRs:

| predictor | AUC |
|---|---|
| **pr_number (recency — not in text)** | **.8056** |
| dense text model | **.3389** |

| pr_number quartile | n | merged |
|---|---|---|
| q0 (1356–1412) | 15 | **.400** |
| q1 (1413–1547) | 14 | 1.000 |
| q2 (1549–1648) | 14 | .857 |
| q3 (1649–1709) | 14 | .929 |

Early bumps get closed; later ones get merged. Metadata absent from the input field predicts at .81 where the text model is anti-predictive. Across all repos, `pr_number` reaches |AUC−.5| > .15 within 14 of 55 repos (in *both* directions — captum .137, pootle .955), i.e. queue position and timing matter and are repo-specific.

Bump-shaped PRs are 790/6,372 = 12.4% of eval, and the model is *more* confident there (dense AUC .637 on bumps vs .561 elsewhere) precisely where the text is least informative.

### Five degenerate all-positive repos distort the pooled number

| group | repos | n | pos rate | mean prob |
|---|---|---|---|---|
| `*_deep_NNN` suffix repos | 5 | 462 (7.3%) | **1.000** | .497 |
| ordinary repos | 112 | 5,910 | .792 | .610 |

`helm_m_deep_024`, `klaytn_acc_deep_019`, `nomad_m_deep_003`, `prometheus_m_deep_018`, `xraycore_m_deep_026` are **100% merged** — zero discriminative content. Dropping them raises pooled eval AUC from .5734 to **.5877**.

They also own the confident tail: of the 165 rows with prob < .05 or > .95, **137 (83%) come from just `nomad_m_deep_003` (88) and `klaytn_acc_deep_019` (49)** — both all-positive repos. The model's entire confident region lives in two repos where the label never varies. (Oracle checks for scale: repo-identity oracle = .8021 on eval, matching the known result.)

---

## 7. Verdict — hypothesis ranking

| rank | hypothesis | verdict | key evidence |
|---|---|---|---|
| **1** | **H5 — text field lacks merge-relevant content** | **CONFIRMED, primary** | 100% of rows are bare `diff --git` output; `Title:` 0.5%, `Description:` 1.3%. No title, body, review thread, CI status, author, or timestamp. Manual read: model keys on repo/genre signature; TF-IDF learns project names. |
| **2** | **H1 — label noise / unobservables** | **CONFIRMED, binding on what remains** | Supersession test: pr_number AUC .806 vs dense .339 on the bump subset; q0 merged .40 vs q1–q3 .86–1.00. ~10% of jac ≥ .9 near-dup diffs carry opposite labels. Sets B and C are the same object class with opposite labels. |
| **3** | **H4 — within-repo homogeneity / grouping** | **REAL, secondary** | Only 30.5% of label variance is between-repo; 5 eval repos are 100% positive (7.3% of rows); per-repo AUC spread is mostly sampling noise (null sd .117 of observed .152); cross-seed per-repo AUC r = .54–.78. |
| **4** | **H2 — truncation** | **REAL IN SIZE, NOT BINDING** | 72.4% truncated, 50.1% at the 24k build cap — but dense AUC is .5735 (untruncated) vs .5739 (truncated). Seeing the whole text buys nothing. |
| **5** | **H3 — score collapse / degenerate predictions** | **REJECTED** | prob spans .0002–1.0, sd .205, full-range histogram, 3 seeds agree (r ≈ .74), ensemble .5715 ≈ single seed. |

### Is this (a) true low verifiability, (b) instrument artifact, or (c) mixed?

**(c) mixed, but leaning (a) — with an important scope restriction on what "the text" means.**

- Not (b): the instrument is healthy. Scores are well spread, seeds are stable, ensembling doesn't help, truncation is non-binding, and dense genuinely beats both char-length (41/55 repos, p < 1e-4) and TF-IDF. The 8B LoRA is extracting what there is to extract.
- Mostly (a): **conditional on a bare diff**, merge outcome really is close to unpredictable at ~.59 AUC within repo, and a large identifiable slice of the label (bump supersession, CI, timing) is generated by a process with no textual trace at all. Δ = T − (V+A) ≈ −.02 is an honest reading of that field.
- But the scope caveat is load-bearing: **the field was never the full PR object.** The V+A criteria bank and the dense model are being compared on an input that omits the PR's stated intent and the entire review conversation — the very artifacts where articulated norms would live. The "dense adds nothing over V+A" claim should be stated as *"on diff-only input"*, not as a property of code review.

### Single follow-up that would most change the conclusion

**Rebuild the `text` field to include PR title + body + review-thread comments (+ CI conclusion), keep the 24k cap but truncate the *diff* rather than the discussion, and re-run the same dense recipe on the same repo-disjoint splits.** If dense still lands at ~.59, the low-T result is a genuine domain property and the VAT row is safe as written. If it moves materially, the current cell is measuring the poverty of the input field, not the verifiability of code review.

Cheap prerequisite worth doing alongside: drop the five `*_deep_NNN` all-positive repos (and the 35 both-label exact-duplicate rows) from eval, and report the recency/supersession channel (`pr_number` within repo) as a named non-textual predictor so the paper can say explicitly how much of merge outcome is queue position rather than content.
