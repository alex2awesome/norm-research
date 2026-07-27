# Sub-community preference heterogeneity: census + screen (2026-07-07)

**Motivation.** All metric-induction runs to date pooled whole tasks, and the MCC acceptance
gate is computed on the pooled label. A community-specific metric worth b bits within a
community holding share s of the corpus contributes ~b·s bits pooled — e.g. 0.10 bits inside a
5%-share genre reads as 0.005 pooled bits, below the detection floor *by construction*. So the
"0 kept / thin articulable tail" results are confounded with sub-community dilution, and the one
partition-conditioned arm (metric_tree) was stalling at 1 node (clustering-depth bug), so
partition-conditioned induction has effectively never run. This screen asks: where do
sub-communities with *specialized* preferences actually exist?

## 1. Community census (metadata-first; topic model where no metadata)

| Task | Axis | Communities (≥200 items) | Source |
|---|---|---|---|
| math-se | `primary_tag` | 53 (real-analysis 10.6K … ) | metadata, v3 position-matched |
| code-review | `repo` (also `language`: Py/Go/Java/TS/JS) | 14,759 repos over 5.34M comments (hundreds ≥1K) | metadata |
| peer-review | `venue` | 21 | metadata |
| press-releases | `press_release_company` | 114 (top values are wires: prnewswire/nasdaq/sec) | metadata |
| legal | statute | 3 (title_vii/flsa/ss_disability) | built-in slices |
| creative-writing | prompt-genre cluster | 24 usable (KMeans k=25 on MiniLM prompt embeddings) | built today: `datasets/creative-writing/writingprompts_prompt_clusters.csv.gz` |
| humor | LDA topic k=50 | ~50, top-20 are 7K–20K items | built today: `datasets/humor/reddit_humor_with_topics.csv.gz` (+`_topics_words.json`) |
| news-homepages | newspaper site | recoverable only from `raw_data/` (v8 dropped site; snapshot_id is a hash) | deferred |

Traps: PR `news_article_domain` exists only for picked-up releases (conditioned on label — cannot
partition). CW story-text LDA (k=100, sk3) is degenerate: 4 mega-topics = 96% of corpus.

**Data-quality finds (as strata to exclude/flag):**
- CW: 1,959 moderator-removal boilerplate stories, base rate 0.001; Theme-Thursday cluster
  (n=3,183, base 0.158) and [IP] image-prompt cluster (n=1,382, base 0.130) are *format/event*
  strata, not genres.
- Humor: topic 36 (n=12.9K; "jokes, reddit, repost, original") = meta/repost stratum.
- PR: `prnewswire` (n=8,319, base 0.57) and `prnewswire.com` (n=1,368, base 0.68) are separate
  company keys — key hygiene issue.

CW genre gradient on the (community-revealed) virality label: alien first-contact 0.674,
dragons 0.653, magic 0.625, superpowers 0.612 vs meta/experimental prompts 0.323, abstract
premises 0.338 — genre alone moves the label a lot (topic confound for pooled CW models).

## 2. Heterogeneity screen (text-probe level, CPU)

Method: per community g — TF-IDF(1-2gram)+logistic, all CV grouped by item id.
`within` = trained on g only (grouped OOF). `transfer` = trained on OTHER communities
downsampled to g's training size, evaluated on g (same-n control). `d_spec = within − transfer`
> 0 ⇒ g's own data teaches what generic data of equal size can't = specialized preference.
`group-identity AUC` = OOF target-encoding of community id alone (confound read).
Script: scratchpad `subgroup_heterogeneity_screen.py`; AUC only (threshold-free).

### math-se (n=100K; group-identity AUC 0.499 — no identity confound; pooled probe 0.668)

| tag | n | within | transfer | d_spec |
|---|---|---|---|---|
| real-analysis | 10,592 | .641 | .603 | +.039 |
| calculus | 9,981 | .676 | .641 | +.036 |
| linear-algebra | 8,208 | .677 | .614 | +.064 |
| abstract-algebra | 5,375 | .622 | .584 | +.038 |
| probability | 4,943 | .638 | .585 | +.053 |
| algebra-precalculus | 3,633 | .648 | .598 | +.050 |
| general-topology | 3,597 | .673 | .585 | +.087 |
| combinatorics | 3,390 | .627 | .563 | +.064 |
| sequences-and-series | 2,886 | .646 | .591 | +.055 |
| complex-analysis | 2,502 | .608 | .556 | +.052 |
| geometry | 2,467 | .659 | .602 | +.057 |
| integration | 2,436 | .672 | .628 | +.044 |

**12/12 tags positive (sign test p≈2e-4), d_spec +0.036…+0.087.** Same site, same label
semantics, no identity confound → cleanest evidence of preference specialization by subfield.
Pooled@g ≥ within everywhere (shared backbone + specialization, not disjoint norms).

### peer-review (n=70K; group-identity AUC **0.827** — venue is a huge confound; pooled 0.724)
Venue base rates 0.39 (ICLR'24) … 0.95 (NeurIPS) — label semantics differ per venue.
F1000Research: transfer 0.528 (≈chance), d_spec **+0.077** — ML-venue data teaches ~nothing
about F1000 review standards (genuinely alien community). ML venues share a backbone
(ICLR/TMLR d_spec +0.01…+0.03). NeurIPS/ICML years show d_spec −0.07…0.00 — power artifact
(95% base ⇒ ~150-250 negatives for the within model).

### press-releases (n=128K; group-identity AUC 0.675 — replicates publisher-id confound; pooled 0.691)
d_spec positive 12/12 companies, +0.008…+0.263: ulta +.263 (base .03; transfer 0.429 below
chance), prnewswire.com +.176, news_delta +.148, spglobal +.091, gartner/sec/uber/goldmansachs
≈+.08. Company-specific pickup dynamics are strong; extreme base rates make some within reads
fragile but the sign is consistent.

### code-review (FULL file = 5.34M comments, 14,759 repos, 5 languages — Python 1.9M / Go 1.4M /
Java 1.2M / TS 0.6M / JS 0.4M; the earlier "84 repos, all-Go" census read only the first 100K
position-clustered rows). Group-identity (repo) AUC 0.677. Top-12 repos, capped 8K each,
grouped by pr_number:

| repo | n | base | within | transfer | d_spec |
|---|---|---|---|---|---|
| hive | 8,000 | .49 | **.827** | .613 | **+.215** |
| ignite | 8,000 | .44 | .726 | .540 | **+.186** |
| ignite-3 | 7,345 | .72 | .639 | .546 | +.093 |
| helix | 7,420 | .90 | .568 | .524 | +.044 |
| operator | 7,613 | .95 | .536 | .507 | +.029 |
| core | 8,000 | .86 | .554 | .545 | +.009 |
| api / gatk / cli | ~8K | .82-.95 | .47-.54 | .50-.55 | −.03..−.01 |
| odl / fdb-record-layer | ~7.5K | .83/.96 | .41/.38 | .50/.49 | −.10 |
| mx-chain-go | 8,000 | .99 | nan | .39 | — |

Same pattern as peer review: **specialization is large exactly where base rates permit
measurement** (hive .49 → +.215; ignite .44 → +.186); extreme-base repos (0.9-0.99) give
weak/negative d_spec = power artifact. Caveats: repo names lack owner (generic "cli"/"core"/
"api"); within-repo predictability may partly ride repo-specific bots/boilerplate comments —
metric-level follow-up needed before interpreting as craft norms.

## 3. Metric-level community comparison (the only post-audit scored matrix with 2 communities)

comp_unified_claude_bank_scores_v3 (988 × 62 usable metrics): cpp (n=573) vs python (n=415).
**Language is 100% confounded with platform** (cpp=luogu, python=codechef) — axis is a
language+platform+editorial bundle.

| community | within | transfer | d_spec |
|---|---|---|---|
| cpp | .624 | .493 | **+.131** |
| python | .699 | .404 | **+.295** |

Coef-vector Spearman rho = **−0.06** (orthogonal weights); sign flips: a500 token-4gram-entropy
+0.58 cpp / −0.47 python; a504 AST-shape flips opposite. Face-valid emphases: cpp →
buffer_string_safety, stl_container_choice, header_hygiene; python → indentation_4_spaces,
import_formatting_aliasing, operator_spacing (PEP8). Within-task analog of the wave-2
cross-task "verdicts invert" finding. Caveats: small n, pooled-mean NaN imputation, a479
applied 0% on cpp.

## 4. Scored-VA inventory per community axis

- comp_unified (2 communities): 988×139 — usable, analyzed above.
- LeetCode C++: 5,000×136 with row_id→question_slug; splittable into problem-topic communities
  (slug→tags join NOT yet done).
- CW arm-run legs on sk3 (wigleaf/virality/royalroad): ~40 clean-craft metrics × 0.4–1.25K
  items; virality leg joinable to genre clusters by text (thin per-genre cells).
- 9-task cells_v1 (~244 aspects × ~300–440 items/task): **QUARANTINED** (LEGACY_NOTE.md —
  pre-2026-06-01 prompt-audit bugs). Do not use for results.
- Math tags / venues / repos / companies / humor topics: **no post-audit scored metrics.**

## 5. Math tag-stratified bank scoring (post-audit, SAME DAY — first community-scored matrix)

**Run:** clean math bank (48 rubrics distilled from the 87,890-rubric pool via anchor-cosine +
junk blocklist + kmedoids; `datasets/math/stackexchange/medoid-bank-clean/bank.json`) scored on
12 tags × 550 items (whole-question stable-hash sample of `math_se_v3_position_matched`),
Llama-3.3-70B-FP8 offline vLLM, executor-closed, 19 min on one B200.
Output: sk3 `outputs/ctree/math_tag_bank/math_tag_bank_scores.parquet` (6,600 × 48; local copy
same path). Validation first (2 tags × 12): no output truncation (all 48 indices parsed),
score distribution healthy, applicability semantics CORRECT — narrow criteria (geometric
diagrams, computer-proof standards) inapplicable off-topic; general craft (clarity, notation)
applied ~1.00. Overall applied fraction 0.37 = bank structure, not a bug.

**Results (n=550/tag):**
- Pooled bank AUC linear **0.597**, GBM **0.621** (+0.024 aggregation headroom — replicates the
  LeetCode pattern at metric level; #61 evidence).
- Per-tag within 0.47–0.61; transfer ≈ within (mean d_spec −0.009); pooled@g ≥ within mostly.
  **No metric-level specialization detectable at n=550.**
- **Split-half control kills the weight-divergence readout**: cross-tag coef Spearman rho 0.22
  (broad metrics) looked like divergence, but same-tag split-half SELF-rho = **0.04** — the
  coef vectors don't correlate with themselves at this n. RETROACTIVE CAVEAT: the cpp/python
  rho −0.06 is likewise noise-dominated; the load-bearing evidence there is the BELOW-CHANCE
  TRANSFER (0.404), which is coef-noise-independent.
- **Matched-n text probe on the SAME 6,600 items: 0.50–0.64** — at equal n the bank captures
  about as much within-tag signal as raw TF-IDF text. The apparent "text ≫ bank" gap and the
  text screen's 12/12 specialization were measured at n=2.4–10K/tag; n=550 is below the
  expression threshold for BOTH representations (power artifact, not bank deficiency).
- Applicability map: mild tag-concentration (formulas/display-equations home=integration;
  hypotheses-before-theorems home=abstract-algebra) — weaker than expected.

**Immediate follow-up LAUNCHED:** re-score at 2,400 items/tag on the 6 strongest text-d_spec
tags (real-analysis, linear-algebra, general-topology, combinatorics, geometry, probability) →
`outputs/ctree/math_tag_bank_2400/` — the powered metric-level specialization test (~11K new
prompts, first 550/tag ride the judge cache).

## 5b. POWERED read (6 tags × 2,400) + metric-generality lattice (same day)

**Powered specialization test (n=2,400/tag, matches text-screen power):** pooled linear 0.601 /
GBM **0.637**; per-tag GBM beats within-linear everywhere (+0.02–0.04). **d_spec is ~0 to
slightly NEGATIVE in 6/6 tags** (−0.002…−0.016) — with the general bank, within-tag reweighting
adds NOTHING over a generic model; the craft backbone is shared. Cross-tag coef rho mean 0.151
vs split-half self-rho 0.172 — weights indistinguishable across tags at the noise ceiling
(weak sibling structure inside the noise: topology~geometry +0.50, real-analysis~geometry −0.12).
**Inference:** the text-level specialization (12/12 tags, +0.04–0.09, §2) is NOT carried by the
general bank ⇒ subfield preference differences, if articulable, require metrics OUTSIDE the
general bank — the reweighting alternative is ruled out; within-tag INDUCTION is the direct
test. (Alternative to kill there: text specialization = topic-vocabulary/base-rate confound,
not craft norms — content-only guard + induction arms adjudicate.)

**Metric-generality lattice (12 tags × 550, per-metric per-tag applicability + univariate AUC):**
- **General backbone (valid |AUC−.5|>.08 in ≥4 tags): the AESTHETIC criteria** — Uniqueness of
  insight (8/12, 0.584), Proof elegance (8/12, 0.587), Good intuition (7/12), Reveals-the-why
  (6/12). Universally-APPLICABLE mechanical craft (clarity, notation, precision) is valid
  NOWHERE — scores near-ceiling (mean 0.89) ⇒ saturated, non-discriminative on math.SE.
  Insight/taste criteria travel; mechanical criteria saturate.
- Tag-local validity candidates (suggestive, n=550): verifying-computation & use-of-examples/
  counterexamples valid only in geometry; example-to-understand-inference only in combinatorics.
- Applicability lattice: 14/48 universal, ~10 tag-conditional, 24/48 dormant in math.
- **Sibling test: metric-profile similarity tracks subfield adjacency** — corr(tag co-occurrence
  sim, metric-profile sim) over 66 pairs rho=0.403, p=0.0008, on top of a ~0.99 shared core
  (calculus~real-analysis closest, abstract-algebra most distant from the analysis cluster).

## 5c. WITHIN-TAG INFILLING — first leg (2026-07-08 00:00): FIRST KEPT METRIC EVER

6-tag arm run live (`scripts/tools/math_tag_infill.sh`, GPU5, n=900/tag, 5 arms × 15 rounds,
clean bank, confirm + content-only + reliability gates, group-split by question). Leg 1
**general-topology (53 min): residual arm KEPT "Tonal Clarity"** — "maintains a consistent and
clear tone throughout, avoiding ambiguity" — guard +0.032 AUC / +0.0164 bits; fresh-seed
confirm +0.034/+0.017 at p_auc=2.6e-05 (< Bonferroni-75 alpha 6.7e-4); retest 0.868;
redundancy R²=0.18 vs bank (bank clarity rubrics ceiling-saturated; this variant has variance
that moves the topology label); applicability ~1.0; gains stable from 50% data. Kept in the
tag where the bank was WEAKEST. Other 4 arms: 0 kept (15 proposals each; drops = gain floors,
confirm, 1 surface). FLAG: recovery-gate agreement ~chance (0.51) — strong on the predictive
gate, weak on the label-free recovery certificate.

**Controls queued:** (1) `math-pooled-12tags` control (same item universe unconditioned, same
recipe; `math_pooled_control.sh` waits for tag legs then runs on GPU 5) — the direct dilution
test; note NO pooled-math arm run existed before, so until it lands the honest claim is
"within-subtask induction produces confirmed keeps where whole-task CW/PR runs plateaued",
not "pooling caused the plateau". (2) Cross-tag validity matrix of all kept metrics after the
6 legs — adjudicates tag-local vs general-bank-gap (Tonal Clarity does not sound
topology-specific on its face).

## 5d. WITHIN-TAG INFILLING — full 6-leg read (2026-07-08 04:14)

**448 proposals, 1 formal keep** — and the keep sits exactly on the bank-weakness gradient:

| tag | bank strength (GBM within) | keeps |
|---|---|---|
| general-topology | 0.572 (weakest) | **1 — Tonal Clarity (residual)** |
| geometry | 0.620 | 0 |
| linear-algebra | 0.630 | 0 |
| combinatorics | 0.647 (strongest) | 0 |
| probability | 0.648 | 0 |
| real-analysis | 0.590 | 0 |

**The confirm tail is non-empty and structured (58 confirm-stage kills, several p<0.05 nominal
that failed only Bonferroni-75):**
- "Tonal Clarity" independently RE-PROPOSED in real-analysis (2x, residual, confirm +0.002/
  +0.005, p=0.008/0.031) — transfers across tags; NOT topology-local.
- Tag-FLAVORED insight family: "Use of Counterintuitive Probability Arguments" (probability,
  +0.0150 bits, p=0.021), "Incorporation of Counterintuitive Insights" (probability, +0.0166,
  p=0.015), "Non-Trivial Inductive Leap" (combinatorics, +0.0137, p=0.0073), "Uses Novel
  Problem Representation" (combinatorics, +0.0130, p=0.0065).
- General clarity family: "Question Clarity" (linear-algebra, +0.0181, p_auc=6.5e-05 — failed
  on the bits-significance side), "Structural Elegance" (topology, +0.0178, p=0.0096),
  "Structural Coherence" (probability, +0.0111, p=0.0093).
- Convergent proposals: "Clear Resolution" proposed 5x in real-analysis by label_contrast.

**Induced-bank-v1 built** (top-10 by confirm bits, with provenance:
`datasets/math/stackexchange/induced-bank-v1/`). Queued on GPU 5 behind the pooled control:
12-tag × 2,400 cross-validity matrix (`math_induced_xtag.sh` → `outputs/ctree/
math_induced_xtag/`). Transfer cells (11 non-source tags) = clean out-of-sample replication;
home-tag cells optimistic (item overlap with the proposing leg). This matrix adjudicates
general-bank-gap vs tag-local for every induced candidate, with proper power.

Reading so far (descriptive): within-subtask induction DOES produce gate-clearing metrics
where whole-task CW/PR runs produced none, the survivors/near-misses split into a general
clarity-gap family + tag-flavored insight family, and the single-keep-out-of-448 rate reflects
a deliberately conservative instrument (per-tag Bonferroni-75) applied to a thin-but-nonzero
articulable residual. Pooled control still pending — dilution claim stays open until it lands.

## 5e. POOLED CONTROL (2026-07-08 05:06): 0 kept / 71 proposals — DILUTION CONFIRMED, both mechanisms

Same 12-tag item universe, same recipe, unconditioned. Two separable mechanisms visible:
1. **Effect dilution**: the pooled residual arm proposed "Tonal Consistency" (the pooled cousin
   of the within-topology keep) — confirm **+0.0059 bits @ p=0.0059** pooled vs **+0.0173 @
   p=2.6e-05** within-topology. Same criterion family, ~3× smaller effect, 3 orders of
   magnitude weaker evidence → killed by the same Bonferroni bar it cleared within-tag.
2. **Proposal dilution**: the tag-flavored insight family (counterintuitive-probability,
   inductive-leap, novel-representation) is ABSENT from pooled proposals entirely — pooled
   WRONG/RIGHT contrasts average over tags, so the proposer never sees tag-specific patterns.
   Pooled proposals are generic ("Employs Relevant Mathematical Concepts", "Methodological
   Transparency"), all confirm-tail p ≥ 0.09 except Tonal Consistency.

**#60 core result**: sub-community conditioning is necessary BOTH to PROPOSE community-local
metrics AND to DETECT their value. Within-tag: 1 keep + structured tail. Pooled, same data:
0 keeps, flat tail, tag-local criteria never proposed.

## 5f. CROSS-TAG VALIDITY + INCREMENTAL REPLICATION of induced-bank-v1 (2026-07-08 05:37)

12 tags × 2,400 univariate matrix + bank-incremental test on home tags (mostly-fresh samples;
~60% items not seen by the proposing leg). Two families, opposite fates:

**Clarity/coherence family — REPLICATES, and is GENERAL (bank-gap, not tag-local):**
| metric | home tag | univariate range (12 tags) | incremental over 48-bank @home |
|---|---|---|---|
| Tonal Clarity (THE KEEP) | topology | 0.56–0.61 everywhere | **+0.028 AUC / +0.011 bits** (arm run: +0.032/+0.016 — replicates) |
| Maintains Consistency in Math Representation | combinatorics | 0.55–0.62 | **+0.026 / +0.018** (confirm-KILLED in arm run p=.018 — conservative-gate miss, now validated) |
| Structural Coherence | probability | 0.56–0.63 | +0.015 / +0.009 |
| Question Clarity | linear-algebra | 0.55–0.62 | +0.011 / +0.010 |

**Insight family (tag-flavored candidates) — does NOT replicate:** counterintuitive-probability
+0.001 bits, inductive-leap/novel-representation/verification-alternative ≈ 0 or negative at
home (combinatorics = strongest-bank tag), Structural Elegance +0.001 (home cell univariately
BELOW chance 0.486 — pure confirm-tail selection artifact). Only Counterintuitive-Insights is
marginal (+0.005).

**Refined #60 conclusion (descriptive):** within-subtask infilling WORKS and the keep is real —
but the validated new metrics are GENERAL bank-gap criteria (a clarity/coherence/consistency
stratum the saturated bank rubrics miss), discovered THROUGH subtask conditioning: per-tag
contrast examples are homogeneous enough for the residual pattern to be visible, while pooled
contrasts blur it (mechanism-2 proposal dilution operates even on general metrics). Genuinely
TAG-LOCAL preferences remain unvalidated — the insight-family candidates were tail artifacts —
consistent with the text-level specialization being either non-articulable at this proposer
strength, or topic-composition confound. Net new certified value: topology bank 0.538→0.567,
combinatorics 0.617→0.644 (+0.011/+0.018 bits) from two induced metrics.

## 5g. PROPOSER DIAGNOSIS (walkthrough) + WAVE-2 GLM PROPOSER (2026-07-08)

**Walkthrough of the residual arm on math-probability** (contrast rebuilt exactly per
contrast.py; scratchpad `residual_walkthrough.py`): 27 viable bank metrics, in-sample bank AUC
0.626, 6+6 WRONG examples, ~6.7K-token prompt. Three findings:
1. **Prompt steering**: `_PROMPT` said "favor aesthetic, tonal, structural, or stylistic
   properties" (CW-era wording) — the residual arm's whole output family (Tonal/Structural
   Clarity) matches the steer. The tag-flavored candidates all came from the arms WITHOUT that
   steer (metric_tree/autometrics). FIXED: prompt now names domain practices/argument patterns/
   community conventions alongside aesthetic/structural (63 tests pass).
2. **★ Proposer strength is the binding constraint (user called it)**: on the IDENTICAL prompt,
   70B mode-collapses (Tonal Clarity ×3, Narrative Flow ×2, all readability variants; zero
   domain content across 15 proposals) while **GLM-5.2 proposes probability-specific practices
   even WITH the old steering** (complement-theorem reframing, assumption auditing, modeling
   abstraction, constructive engagement with the asker's attempt); de-steered it goes fully
   technical (complementary counting, frame transformation, event decomposition). The 70B's
   mode collapse also explains Tonal Clarity's cross-tag re-proposal (it proposes the same
   thing everywhere). **Wave-1's "tag-local is non-articulable" reading is RETRACTED as
   premature — the search was proposer-limited.**
3. Contrast examples themselves are fine (tag-specific patterns visible to a strong reader).

**WAVE-2 launched**: 6 math tags + pooled control, GLM-5.2 proposer (z.ai, ~24 calls/leg,
quota-light) + unchanged 70B offline judge/executor (certificate E unchanged), residual+
metric_tree arms, de-steered prompt (`math_tag_infill_glm.sh`, out dirs `*-glmprop`).
**First leg (topology): proposals TRANSFORMED** — "Deconstructs User's Reasoning", "Explicitly
Evaluates User's Proof", "User-Provided Proof Unpacking", "constructs_minimal_ad_hoc_
counterexample", "provides_explicit_advanced_counterexample" = community-interaction practices
local to math.SE topology Q&A. 0 formal keeps but a hot confirm tail: "Pre-existing
Mathematical Artifacts" p=0.00298 (bar 0.00208 — missed by 1.4×), User-Proof-Unpacking
p=0.0084, contextualization-or-counterexample p=0.015. Same protocol as wave-1: replication
(incremental on fresh 2,400/tag) adjudicates the tail after all legs land.

**Cross-domain expansion QUEUED behind wave-2** (`cw_humor_community_infill.sh`, 10 legs):
CW prompt-genres (abstract-premise/immortality/wakeup-mystery/hell-deal, n=5.5-7.3K, + pooled
control; group-split by prompt; clean craft bank) and humor topics (marriage/bar/family/doctor,
n=11-20K, + pooled control; NEW clean humor bank built — datasets/humor/medoid-bank-clean,
40 rubrics from 53,807-rubric pool, ~36/40 humor-craft).

## 5h. PRE-REGISTRATION for the CW/humor legs (2026-07-08, before any results)

**User's conjecture (verifiability model):** fields with deep/verifiable evaluation (math,
coding, patents) resist new-metric invention because their metric lexicons are already
CONVENTIONALIZED (supporting: lexicon census — math family-invariant vs CW family-variant;
math wave-1/2 keeps are all interactional PRACTICE norms, not content criteria — the content
lexicon is saturated, the community's tacit residual is interactional).
**Competing (gap model):** keep-rate tracks (within-community text ceiling − bank floor) where
that gap is articulable, regardless of domain verifiability.

Measured today (within-community TF-IDF ceilings, cap 6K, grouped where applicable):
| community | n | ceiling | (math tags, for scale: ceilings .61-.68, bank .60-.65 → gap ≈.03) |
|---|---|---|---|
| humor-bar-jokes | 19,114 | **0.658** | |
| humor-marriage | 20,232 | **0.655** | |
| humor-doctor | 11,476 | 0.642 | |
| cw-abstract-premise | 7,256 | 0.625 | base 0.35 |
| humor-family | 13,096 | 0.619 | |
| cw-wakeup-mystery | 5,694 | 0.607 | |
| cw-immortality | 6,481 | 0.606 | |
| cw-hell-deal | 5,551 | 0.580 | |
Bank floors expected weak (CW clean craft bank ranked pooled virality 0.508; humor floor
unmeasured) → gaps ~0.10-0.15, i.e. 3-5× math's.

**Predictions on record:** BOTH models predict CW/humor tails hotter than math's. They part on
(a) mechanism and (b) within-domain ordering — gap model says tail-heat follows the ceiling
order above (bar-jokes/marriage hottest); verifiability model says domain type dominates.
Registered caveat: a wide text-bank gap can be NON-articulable (topic/timing lottery) — the
arms + content guard test the articulability of the gap, so 0-keeps-everywhere with hot
ceilings would mean "wide but tacit" (a §IV result, not a failure).

## 5i. WAVE-2 COMPLETE (2026-07-08 14:12): pooled-vs-within contrast REPLICATES with strong proposer

7 legs (6 tags + pooled), GLM proposer, residual+metric_tree, de-steered prompt. 0 formal keeps
anywhere; the READOUT is the tail contrast: within-tag tails hot in 5/6 tags (p=.003-.028) vs
pooled tail flat (1 candidate p=.098). **Dilution is proposer-robust.** Topology again hottest
(5 tail candidates — the weak-bank gradient repeats). **induced-bank-v2 built** (12 GLM-proposed
candidates, 8 practice-norms + provenance; datasets/math/stackexchange/induced-bank-v2/):
top = broad-contextualization-or-counterexample (+.0093, p=.015), Open-Ended Theoretical
Inquiry (+.0090), User-Proof-Unpacking (+.0073, p=.0084), Pre-existing-Mathematical-Artifacts
(+.0069, p=.003), supplies_fully_explicit_final_result (p=.0145). Replication scoring (12-tag ×
2,400) QUEUED behind the CW/humor batch (math_induced_v2_xtag.sh). Note the family shift vs
wave-1: 70B proposed STYLE metrics (tonal/structural), GLM proposes Q&A PRACTICE norms
(unpacking the asker's proof, explicit final results, counterexample construction) — the
conventionalization story's predicted residual.

## 5j. CW/HUMOR COMMUNITY BATCH COMPLETE (2026-07-08 17:25) + pre-registration scorecard

10 legs (4 CW genres, 4 humor topics, 2 pooled controls), GLM proposer, residual+metric_tree,
n=900. 0 formal stage-1 keeps anywhere; the tails:

| leg | ceiling (§5h) | hot tail (p<.05) | character |
|---|---|---|---|
| humor-marriage | .655 | **4 cands, best p=.00072** (+.0139 power-dynamics-subversion; passed p_auc bar, died on p_bits) | topic-local |
| humor-family | .619 | 2, p=.0025/.0145 (**anti-joke abruptness** family) | topic-local |
| humor-bar-jokes | **.658** | 0 | quiet |
| humor-doctor | .642 | 0 | quiet |
| cw-abstract-premise | .625 | 3, best p=.017 (prompt-integration ×2 convergent) | platform+genre |
| cw-wakeup-mystery | .607 | 4, p=.012-.043 (escalation logic, groundedness-over-melodrama) | genre-local |
| cw-hell-deal | .580 | 5, p=.025-.033 (earnestness-over-irony ×2 convergent; no-melodramatic-monologue +.0113) | genre-local |
| cw-immortality | .606 | 0 | quiet |
| cw-POOLED | — | ~0 (best +.0016 @ p=.089) | **flat** |
| humor-POOLED | — | 0 | **flat** |

**Dilution contrast: 4/4 instances** (math w1, math w2, CW, humor) — within-community tails
hot, pooled flat on the same universes. **Pre-registration scorecard (§5h):** both models'
shared prediction (CW/humor tails ≥ math) = TRUE (humor p=.0007-.0025 strongest of program).
Gap-model within-domain ordering = **FAILED**: bar-jokes has the HIGHEST ceiling and a cold
tail; family the lowest and the hottest. Favors the conventionalization reading at subgenre
grain: bar jokes are the most FORMULAIC micro-genre (conventions fully articulated in the
bank/format), family/dad jokes carry a live unarticulated norm (anti-humor). Caveat: 2/4
quiet humor legs also consistent with "wide but tacit" — stage-2 adjudicates.

Stage-2 replication (queued, end of chain) takes ~18 candidates over 6 hot legs at
Bonferroni/18 ≈ .0028 on fresh n=2400 — marriage/family candidates at stage-1 p=.0007-.005
with 2.5× the sample are strong conversion favorites.

## 5k. WAVE-2 REPLICATION READ (v2 xtag, 2026-07-08 18:01) — largest validated gains of the program

Incremental over the 48-bank in home tags (mostly-fresh 2,400 samples; formal disjoint stage-2
still queued):
| candidate | home | ΔAUC | Δbits | cross-tag univariate |
|---|---|---|---|---|
| provides_broad_contextualization_or_counterexample | topology | **+0.045** | **+0.0132** | 0.54-0.60 EVERYWHERE (general) |
| provides_explicit_advanced_counterexample | topology | **+0.043** | **+0.0117** | 0.55 home max, 0.50-0.54 elsewhere (mildly tag-local) |
| Open-Ended Theoretical Inquiry | probability | +0.009 | +0.0085 | — |
| Pre-existing Mathematical Artifacts | topology | +0.026 | +0.0067 | 0.51-0.56 flat (general-weak) |
| fully_resolves_the_target_problem | geometry | +0.012 | +0.0042 | — |
| NOT replicated: User-Proof-Unpacking, Deconstructs-User's-Reasoning, Proof-Verification-Context, Direct-Answer-Provision | | ~0 | | |

Counterexample-provision — THE canonical currency of topology culture — is the top validated
discovery, though univariately it's general practice whose incremental value CONCENTRATES
where the bank is weakest. Math pattern holds: replicating discoveries = general practice
norms found through the within-tag microscope; strictly tag-local candidates keep failing
replication. The strictly-community-local hopes now ride on humor/CW stage-2 (marriage
power-dynamics p=.00072, family anti-joke p=.0025).

## 5m. STAGE-2 (PANEL protocol) full readout (2026-07-08 19:14) — nulls with ONE partial survivor

CW: 12/12 not-replicated (rep_bits ≈ 0). Humor: 5/6 not-replicated — including the p=.00072
power-dynamics candidate (rep_bits +0.0000, a dramatic stage-1 selection artifact OR a panel-
attenuation casualty) — but **"Punchline Omission" (marriage) partially survives: rep_bits
+0.0039, p_auc=0.0094 on strictly fresh data** (missed Bonferroni/6=.0083 by 13%; p_bits .085
failed). IMPORTANT CAVEAT on all of §5m: this stage-2 ran with the PANEL protocol (45-rubric
prompts) while stage-1 measured candidates SOLO — the foundry round-1 rerun (solo × dense,
queued) is the corrected instrument; treat these as lower bounds. Math stage-2 requeued
(argparse fix). OPS NOTE: sk3 root disk 99% (5.9G free; /tmp only 7.4G — main consumer is
elsewhere; our chain writes to /lfs, but new logins may break if / hits 100%).

## 5n. ★ HONESTY CORRECTION (2026-07-08 19:5x): strictly-fresh stage-2 kills the "mostly-fresh" gains

Math stage-2 on STRICTLY-disjoint samples (panel protocol): 7/8 null — including
broad_contextualization_or_counterexample (**+0.045 AUC on the overlapping xtag read → 0.000
strictly fresh**). RETROACTIVE FLAG on §5f and §5k: the xtag "replication" samples shared
~40-55% of items with the proposing legs (md5-sample ∩ seed-7-sample), and that overlap —
not fresh signal — carried the celebrated gains (Tonal-Clarity +.028, Maintains-Consistency
+.026, counterexample +.045 reads are all SUSPECT; Tonal Clarity's original stage-1 formal keep
stands on its own full-Bonferroni in-run evidence, but its replication read does not).

**Strictly-fresh tally across the program (all panel-protocol so far):**
math 1/8 partial (Open-Ended Theoretical Inquiry +.0045, p=.032/.042), humor 1/6 partial
(Punchline Omission +.0039, p=.0094/.085), CW 0/12. Everything else ≈ 0.

Two live explanations, now cleanly separable by foundry round 1 (solo × dense × strictly
fresh): (a) stage-1 tails are ~all selection noise and the within-community articulable
residual is below detection even at 2,400 — the "wide but tacit" reading; (b) the panel
measurement context destroys candidate signal (stage-1 measured solo). The solo-vs-panel test
answers (b) directly. Lesson locked in: ONLY strictly-disjoint replication counts; overlap
reads are not evidence.

## 6. Next

1. **Math tag-stratified bank scoring on sk3** (offline batch vLLM, 1 GPU): score the math bank
   on a sample stratified over top ~12 tags → per-tag weights/AUC/GBM ceiling, then within-tag
   induction arms through the certificate gate. Prediction: per-tag weight vectors decorrelate
   like cpp/python; pooled gate has been diluting exactly this.
2. LeetCode slug→topic join (arrays/DP/graphs communities on the existing 5,000×136 matrix).
3. CW within-genre induction on top genre clusters (alien/fantasy/superhero…), excluding
   removal/format strata; virality label = community-revealed leg.
4. Certificate architecture implication: per-community certificates + a dilution-aware pooled
   read (Σ community bits · share) instead of one pooled gate.
