# Claim-verification multi-y study — results (2026-07-09)

Thread: does journalism cluster with the substantiation cluster (patents/peer review) or the
effect cluster (CW/humor)? Test: the same claim-based metric families against every available
outcome variable across peer review, press releases, and news. Plus the recovery-MI leg for
claim extraction itself. All grouped CV (year / company / venue×year / outlet×day), no-leak
tiers only (PR T2 coverage tier excluded — availability leakage, see 07-08 notes).

## Recovery-MI: is claim extraction a recoverable/promptable rule?

GLM-5.2 infers the extraction rule from k=6 (head, Gemma-claims) exemplars, applies to 30
held-out docs; agreement = symmetric token-Jaccard claim matching.

| domain | recovery | floor (mismatched pairs) | ceiling (Gemma re-run) |
|---|---|---|---|
| pr | 0.565 | 0.000 | 0.97 |
| newsfull | 0.469 | 0.000 | 0.99 |
| peerintro | 0.388 | 0.000 | 0.96 |

Read: rule gist transmits (>> floor) but ~half the claim-level behavior is execution detail
not carried by the stated rule (<< ceiling). Gradient pr > news > peer matches GEPA headroom
(peer was the flat/0-accepted-mutation domain; news had the most GEPA gain 0.472→0.592).

## The outcome-conditional matrix (grouped-CV AUC)

| metric family | peer accept (inst.) | peer cite-pct (crowd) | PR coverage (inst.) | news placement (inst.) | news twitter (crowd) |
|---|---|---|---|---|---|
| evidence/verification | 0.504 | 0.497 | 0.566 | 0.524 | 0.492 |
| claim quality (A-judge) | 0.557 / 0.507* | 0.522* | 0.562 | — | — |
| sourcing census (CODE) | — | — | — | **0.620** | **0.670** |
| sourcing densities only | — | — | — | 0.537 | **0.689** |
| wordcount alone | — | — | — | 0.527 | 0.597 |

\* peer has two samples: tiered ICLR-2024/25 intros (accept 0.557 on cq) and the
citation-file abstracts 2016-23 (n=853, both y's on the SAME docs: accept 0.507 / cite 0.522).

- Citation-y run (run_citation_y.py): openalex_citations_v3 abstracts, NeurIPS/ICML/ICLR
  2016–2023, venue×year cells, y_accept=judgement, y_cite=pct-rank within cell. Everything
  ≈ chance; best univariates cq_falsifiability .535/.538 and cq_elegance .538/.538 (both y's
  equally — no outcome-conditional split). Accepted-only slice n=428 → folds degenerate (nan).
- Twitter-y sourcing signal SURVIVES the standing length audit: densities-only 0.689 ≥ raw
  0.683; wordcount alone 0.597; within-length-quintile mean 0.64 (strongest in longest
  quintile 0.82). c_attrib_verbs top univariate 0.681. NOT yet topic-stratified (caveat).
- Placement-y (run_placement_y.py): headline-containment join of fetched fulltext →v9 rows
  (12% hit rate on twitter-era sample; 2,591 matched over full fetch → scored 700).
  Sourcing raw counts 0.620 but densities collapse to 0.537 → placement tracks ABSOLUTE
  amounts (c_numbers univariate .577), not rates. Twitter on the same placement-matched docs
  is much weaker (.564 vs .670) — selection: homepage-featured articles are range-restricted
  on sourcing.

## Bugs fixed en route

- Queued chain deadlocked: `while pgrep -f run_claim_quality` matched its own bash -c command
  line — never gate a chain on a pgrep pattern contained in the chain's own cmdline.
- z.ai GLM returns transient 529s under load → glm() now 7 tries, backoff to 60s.
- pandas ≥2.2 groupby.apply drops grouping columns → use groupby.head() after shuffle.
- Citation file (2013–2023, s2 URLs) structurally disjoint from PDF-DB tiered sample
  (ICLR 2024–25) — no id join exists; solved by running metrics on the citation file's own
  abstracts.
- claim-quality random 800-doc sample ∩ tiered random 600-doc sample = 8 docs → added
  --ids-file to score exactly the tiered ids.
- Peer tiered sample spans 2 years only → year-grouped 5-fold CV degenerates to nan; use
  min(5, n_groups) folds.

## Descriptive summary (not conclusions)

Claim-substantiation metrics are chance-to-weak on every y in every domain measured here.
The only claim-family signals above 0.6: sourcing style vs crowd engagement (news twitter,
length-robust) and sourcing raw counts vs placement (count- not rate-driven). The
"journalism = patents" grouping gets no support from outcome prediction; what patents'
prior-art machinery predicts there (+15.1 disclosure gap) has no analog signal on news/PR/peer
outcomes. Outcome-conditional (H_outcome) prediction — quality loads on crowd-y over
institutional-y — shows only a whisper in peer (.522 vs .507, same docs) and a real split in
news (sourcing: crowd .67-.69 vs institutional .62-count/.54-rate), mixed evidence.

Outputs: outputs/recovery_mi_extraction.json, outputs/multi_y.log, outputs/multi_y_news/,
outputs/placement_y/, outputs/citation_y/, outputs/tiered_{pr,peer}/.

## Pipeline validation vs expert-revealed ground truth (reviewer "not supported" complaints)

User q: when reviewers say "this claim isn't supported," does the pipeline find the same?
Instrument: regex-mine 2,982 reviews of the 600 tiered papers -> 110 candidate sentences ->
Gemma confirm-judge (blinded anchors 8/8, flag rate .90) -> 83 papers (13.8%) flagged, 97
challenged-claim paraphrases (outputs/reviewer_flags/).

| test | result |
|---|---|
| paper-level: low t1_support -> flagged | AUC .532 (t1_echo .545) — weak |
| claim population: challenged claims matching our top-4 intro claims | 9/97 (~9%) |
| verifier direct on challenged paraphrases (own paper) | NONE .565 vs same-paper control .421 (+.144, perm p=.011) |
| placebo (challenged vs MISMATCHED paper) | NONE .902 (verifier not degenerate) |
| adequacy mode (ESTABLISHED/ASSERTED_ONLY/ABSENT) | challenged NOT-ESTABLISHED .771 vs control .655 (+.117, p=.041) |

Two failure modes identified (the "measuring the wrong thing" answer, made precise):
1. CLAIM POPULATION: reviewers challenge assumptions, design-choice justifications, and
   scope/generalization statements — ~91% of challenged claims are NOT among our top-4
   intro contribution-claims.
2. ECHO != ADEQUACY: challenged performance claims verify FULL because the paper's own body
   asserts its results ("surpasses SOTA" etc.); reviewers dispute whether EVIDENCE
   establishes the claim. Adequacy-mode reasons qualitatively right (missing ablations,
   paper-admits-gap) but separation modest — control intro claims are also often
   not-established (.655), diluting contrast.
Caveats: n=70-97 challenged claims, single judge, regex recall unknown (explicit phrasings
only). ~20% of judge paraphrases non-propositional (filtered for adequacy arm).
Scripts: run_reviewer_flags.py, run_verify_challenged.py, run_adequacy_mode.py.

## v2 checkers: typed claims + adequacy + prior-art (patents flow ported) — 2026-07-09 pm

Typed extraction v2 (600 papers, 5,929 claims, ~9.9/paper): contribution 1169 / performance
1204 / assumption 1204 / design_justification 1102 / scope 647 / novelty 603. Adequacy
gradient face-valid: EST rate assumptions .150 < design_just .185 < novelty .266 <
performance .309 < scope .386 < contributions .483.

Reviewer novelty complaints mined (2nd expert ground truth): 177 confirmed over 140 papers
(23.3%; anchors 8/8); 46% name the prior work.

| test | result |
|---|---|
| adequacy est_rate vs support-flags | .536 (v1 t1 .532; design_just best .558) — no gain |
| prior-art (BM25 3-venue abstracts) vs novelty-flags | .483 NULL; RELATED saturates 84.6% |
| claim-level disputed→ANTICIPATED/RELATED | 13/21 .619 vs base .954 — uninformative |
| gold-name arm: reviewer-named prior in organic top-5 | 2.4% (patents pathology: organic misses gold) |
| gold-name arm: ANTICIPATED given name-matched gold | 4.8% — name match lands wrong papers; canonical refs (Whisper/T5/FPN) OUT of 3-venue pool |
| RELATED-WORK arm (own citations as evidence base) | disputed .529 vs control .432 ANTICIPATED (+.097, p=.077); real disclosing spans ("multi-step SAM → Foret et al. 2021") |

Diagnosis (descriptive): every prior-art variant is limited by (a) evidence-base coverage
(reviewer refs are any-venue; 3-venue abstracts miss canonical methods), (b) base-rate
saturation — ML text is inherently adjacent, so "substantially the same idea" fires on
controls too (43% of ordinary novelty claims "anticipated" by own related work), and
(c) the SS102/SS103 mismatch replicated from patents: reviewer complaints are mostly
"the delta over prior work is trivial" (obviousness/combination), while the checker asks
single-reference disclosure. Fix candidates: resolve named refs via OpenAlex/S2 (true gold
arm), and a DELTA-graded verdict ("what does the claim add beyond this prior work; is the
delta substantive?") = the SS103 construct. Scripts: run_claims_v2 / run_check_v2 /
run_novelty_flags / run_validate_v2 / run_gold_prior / run_rw_prior.

## Journalism v2 ports — 2026-07-10

**Topic-stratification caveat CLOSED (with nuance):** sourcing->twitter signal within
URL-derived sections: weighted mean AUC .602 over 6 sections (pooled was .689 — part topic
mix, part real). SECTION-CONDITIONAL: politics .748 (n=40), athletic .804, california .820;
internacional .517, sport .537, business .077-INVERTED (n=28, wide CI). Descriptive: heavy
attribution engages in politics/local/sports-features, not business/international.

**Churnalism delta-checker (SS103 -> journalism) VALIDATES:** 200 (PR, coverage) pairs +
100 mismatched placebo. Matched: SUBSTANTIVE .525 / NO_OVERLAP .425 / TRIVIAL .05; placebo
100% NO_OVERLAP (perfect content gate). Convergent CODE check monotone: verbatim token
containment .712 (TRIVIAL) vs .270 (SUBSTANTIVE) vs .139 (NO_OVERLAP); TRIVIAL examples
containment .97/.85 with added="nothing". Churn rate among genuinely-covering articles
10/115 = 8.7% (mapping join is noisy: 42.5% of "matched" pairs judged not-about-this-PR).
Substantive coverage weakly tracks broader pickup (n_outlets AUC .605).
Script: run_churn_check.py; outputs/churn_check/. Queued next: attribution-adequacy port
(SOURCED vs ASSERTED_ONLY per claim) after peer expansion frees Gemma.
