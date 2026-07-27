# Org size ↔ accept/reject formalization

**Date:** 2026-07-09
**Question (Alex):** Are larger organizations more likely to have *formalized* accept/reject
procedures? Or are idiosyncratic "one-man bands" more likely to have rigid rules?
**Corpus:** GitHub PR test-execution project. 244 repos with enough within-repo
accept/reject contrast to measure predictability, all org-owned.

---

## TL;DR

**Organization size predicts formalization *on paper* but not *in practice*.** Big projects
write the rules down (CONTRIBUTING, CODEOWNERS, PR templates) — but those documents do **not**
make their per-PR accept/reject decisions any more predictable. If anything, **small maintainer
teams apply a more *consistent* gate** than large ones. The "idiosyncratic one-man band" turns
out to have the *rigid* rule; the large org has the written procedure and the decision
heterogeneity.

| Relationship | ρ (Spearman) | p | Reading |
|---|---|---|---|
| stars → has CONTRIBUTING.md | **+0.36** | <1e-4 | big projects codify |
| stars → # formalization files | **+0.33** | <1e-4 | big projects codify |
| stars → decision *consistency* (predictability) | +0.04 | 0.59 | **null** |
| org members → consistency | −0.07 | 0.29 | **null** |
| having written rules → consistency | +0.00 | 0.96 | **null — docs don't bind** |
| small org (≤5 members) vs large, consistency | 0.611 vs 0.566 | 0.035 | **small MORE consistent** |

Multivariate OLS `consistency ~ log_stars + log_members + n_formal_files`: every term NS, R²=0.007.

---

## How "formalization" was measured

The key move: **operationalize "how institutionalized is a repo's accept/reject rule" as
within-repo predictability.** A repo whose accept/reject decisions follow a learnable, consistent
rule is "formalized/rule-governed"; a repo whose decisions look ad-hoc is "idiosyncratic."
Concretely, for each repo:

- **`within_auc`** = 5-fold stratified-CV AUC of an L2-penalized logistic regression
  (`C=0.1`, StandardScaler) predicting `y = rejected` **within that single repo**
  (train and test folds both drawn from the same repo).
- Restricted to repos with **n ≥ 40 PRs and ≥ 8 in each class** (need real contrast).
- **249 repos** measurable; mean within_auc = **0.579**, but the range is enormous:
  **0.07 → 0.92**. That spread is itself a finding — repos vary wildly in how
  rule-governed their gate is.
  - Most rule-governed: pudl 0.92, go-spacemesh 0.89, hive 0.86, podman 0.84.
  - Most idiosyncratic: datalad 0.07, aws-greengrass 0.09, scikit-bio 0.22.

### Was this predictability on our VAT metrics specifically? — YES, and it's robust to that choice

The `within_auc` model was fit on **exactly our 82 VAT features**:
- **64 A (articulable) metrics** from the bank (≥5% coverage on this corpus) — e.g. a104
  test_presence, a309 test_source_correspondence, a20 dependency_hygiene, a159
  import_organization, a403 idiomatic_patterns.
- **18 V (verifiable) features** — P2F/F2P flags, verdict one-hots, baseline/post test
  counts, smoke_rc, n_fail_genuine.
- **No dense text (T), no raw diff.**

So strictly, `within_auc` measures *"how well our articulable+verifiable instruments capture
this repo's gate"* — a **lower bound** on true predictability. To check the org-size result
isn't an artifact of our particular bank, I recomputed within-repo predictability a completely
independent way: **TF-IDF (1-2 grams, 2000 features) over the raw diff text**, same 5-fold
within-repo CV, on the 105 repos present in both sets.

- The two predictability measures **agree**: VAT-metric vs TF-IDF-text ρ = **+0.335, p=0.0005**.
- The **small-org edge replicates on the independent text measure, more strongly**:
  small-org TF-IDF AUC **0.635** vs large-org **0.569** (MWU **p=0.020**).
  (VAT measure on the same 105-repo subset: 0.576 vs 0.570 — same direction, weaker, because
  this subset trims the extremes.)

The finding is therefore **not** an artifact of the VAT bank; it shows up in model-free text
features too.

## Org covariates

Fetched via GitHub GraphQL (1 call/repo): owner type, stargazers, org `membersWithRole`
total, and existence of CONTRIBUTING / CODEOWNERS / PR-template / CI-workflows.
- 243/244 repos are **Organization**-owned (GitHub's contrast-rich corpus is org-heavy),
  so a clean *user-vs-org* "solo dev" test was impossible → proxied by **small-org (≤5 public
  members) vs large-org**.
- Median 15 org members (p25=6, p75=57, max=4382 for WALinuxAgent).

## Robustness / caveats

- **Sample-size confound rejected.** Small repos could look spuriously predictable if fewer PRs
  → noisier AUC. The confound runs the *wrong* way: within_auc ↔ n_PRs ρ = +0.11 (larger repos
  slightly *more* predictable by noise), so it can't manufacture the small-org edge.
- **Privacy-hidden members.** 36/244 orgs report ≤2 public members (org privacy setting hides
  the roster), so `n_org_members` is noisy at the low end. **Stars is the load-bearing size
  proxy** and gives the same null for consistency (+0.04) alongside the strong positive for
  written files (+0.33).
- **All org-owned.** Can't distinguish true solo-user repos; "small org" is the closest proxy.
- Predictability is a lower bound (see above); the *level* (~0.58 mean) shouldn't be
  over-read, but the *contrasts* across repos are what the analysis rests on and those replicate.

## Interpretation

A single maintainer (or tiny team) applies a personal, often-unwritten gate that is highly
**self-consistent** — one person's taste predicts itself. Large orgs accumulate the *artifacts*
of formalization (written contribution guides, ownership files, templates) but also many
reviewers, so the realized per-PR decision becomes *more heterogeneous*, not less. The written
procedure is largely **decoration that doesn't bind the outcome** (docs→consistency ρ=+0.00).

This dovetails with the main VAT result: average within-repo predictability is only ~0.58
because most of the operative "rule" lives in reviewer taste, not codified procedure — and we
now have direct evidence that codifying procedure does **not** raise predictability.

## Files
- `/tmp/repo_predictability.parquet` — per-repo within_auc (VAT-metric)
- `/tmp/org_meta.jsonl` — org covariates (GraphQL)
- `/tmp/org_formalization.parquet` — merged analysis table
- `/tmp/predictability_robustness.parquet` — VAT vs TF-IDF cross-check
- scripts: `/tmp/fetch_org_meta_gql.py`, `/tmp/org_formalization_analysis.py`

## Follow-ups worth doing
- Re-run once the 50K-signal scale-up lands more repos → tighten the small-org CI and get
  enough true solo-user repos for a real user-vs-org test.
- Add a *reviewer-count* covariate (distinct PR closers per repo) — the mechanism hypothesis is
  that predictability falls with number of decision-makers, not org size per se.
- Test whether repos WITH CODEOWNERS route reviews to consistent people (should raise
  predictability if the mechanism is "single decider"); currently CODEOWNERS presence is null,
  but presence ≠ enforced routing.
