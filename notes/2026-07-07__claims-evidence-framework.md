# Claims / Evidence / Novelty framework — journalism between (peer-review, patents) and (CW, humor)
2026-07-07. Design doc for the cross-field grouping experiments + the claim-verification machinery.

## The framing problem
Journalism sits between two field-clusters:
- **Substantiation cluster** (peer review, patents): text makes CLAIMS that must be LICENSED BY EVIDENCE
  (experiments/baselines; prior-art differentiation; sources/documents).
- **Effect cluster** (creative writing, humor): text is licensed by AUDIENCE RESPONSE, not evidence.

Candidate grouping statements, weakest→strongest:
1. ~~Shared top metrics~~ (arbitrary, name-level).
2. Shared metric-KIND profiles (degree of articulability, retrieval/facteval/novelty style). Better, still descriptive.
3. **THE GOLDEN-STATEMENT CANDIDATE — outcome-conditional evidence structure:** the grouping is not a
   property of the FIELD but of the OUTCOME TYPE. Institutional outcomes (peer-review accept, patent
   grant, editorial placement/pickup) are evidence/substantiation-loaded; audience outcomes (twitter
   engagement, humor scores, RoyalRoad follows) are effect/style-loaded. Fields "cluster" by which
   outcome type they're usually evaluated on. **Journalism is the unique domain where BOTH outcome
   types exist for the SAME items** (editorial placement + coverage-pickup vs twitter engagement) —
   so it's the crossover experiment that can separate the two hypotheses:
   - H_field: journalism items have a fixed V/A profile regardless of outcome (fields are the unit).
   - H_outcome: the SAME items' evidence-metrics predict institutional-y while style-metrics predict
     crowd-y (outcome types are the unit; journalism "sits in the middle" only because it's usually
     measured on a mixture).
   Testable, mechanism-level, and it explains WHY journalism looks intermediate instead of just
   observing that it does.

**Standing guard (user concern): length/word-count short-circuit.** Every V/evidence metric is
reported with (a) length-stratified separation (patents discipline: report SEPARATION not level),
(b) a shuffled-evidence placebo (evidence pool swapped from another doc → metric must collapse),
(c) partial-AUC controlling hl_len/body_len. A metric that survives none of these is a length proxy.

## The isomorphism with patents (already built, 59,937 claims)
patents: claim element → K=8 candidate refs → Gemma localize-then-verify → {discloses, spans, reason}
journalism: atomic claim → evidence passages (body/sources) → localize-then-verify → {supports, spans, type}
peer review: contribution claim → paper internals (experiments/tables) → localize-then-verify → {supported}
Same op-class; the patents corpus already certified the pattern (disclosure gap +15.1pts, V=.591/A=.616).

## Experiment suite
- **E1 CONSISTENCY (claim-support) — press-releases first.** Extract ≤5 atomic claims from headline+lede;
  evidence pool = body remainder; localize-then-verify each claim (FULL/PARTIAL/NONE + span + evidence
  type ∈ {data/statistic, named-source quote, document/official, assertion-only}). Doc metrics:
  claim_support_rate, n_claims, n_supported, evidence_type mix. y = k≥3 coverage (company-grouped).
  Controls per standing guard. Same machinery architected for homepages (blocked on full text; runs on
  headline-vs-context weakly until then).
- **E2 SOURCING CENSUS.** Deterministic + judge source counting: named sources w/ title ("said X, CEO"),
  anonymous ("sources familiar"), documents ("according to the filing"), data ("survey of N").
  n_sources / source_type mix as V-metrics; journalism's classic "how well-sourced" axis.
- **E3 TWITTER-y CROSSOVER (the golden test).** Resume tweetapi drain (quota fresh). Label = within
  outlet×day percentile of engagement (n_tweets / sum_likes). Join to homepage items via anchor_text
  prefix-match. Run the SAME V/A banks against twitter-y vs placement-y on the same items. H_outcome
  predicts: style/effect metrics gain AUC on twitter-y, evidence/hard-news metrics gain on placement-y.
- **E4 EVIDENCE-OP TRANSPORT (deep functional test).** Port patents PriorArtOps→SourceOps for PR:
  op = lookup claim-support verdicts/spans; certified marginal via ops-vs-NullOps twin (metric_seam
  f2p_mock pattern). If the same op-class carries certified marginal in patents AND press-releases
  (and later peer review), the grouping is FUNCTIONAL (same machinery load-bearing), not nominal.
- **E5 NOVELTY-OP.** Journalism novelty = differentiation vs nearest prior coverage (bge/FAISS retrieve
  K nearest earlier docs → judge differentiates or not) — structurally identical to patents novelty
  (done) and peer-review novelty. Predicts coverage/placement?
- **E6 METRIC-KIND PROFILES (formalize user option 2).** Tag every bank metric by KIND
  {verifier-evidence, verifier-form, retrieval-novelty, judgment-articulable, taste}; per field compute
  (kind × predictive-share of each y). Field similarity = similarity of these profiles. Reuses
  isomorphism-census machinery; deeper than name-matching because it locates WHERE signal lives.

## Current factual state (2026-07-07)
- Twitter: 27,812/662,855 URLs scraped (quota drained 06-18; RESETS monthly → resume now). Same URL
  universe as homepages (built from the same hyperlinks.json captures). Fields: n_tweets, sum_* engagement.
- Full text: homepages NO (headline+summary only; fetch needed — natural subset = twitter-covered 27.8k);
  press-releases YES (72,315 full bodies) → E1/E2/E4/E5 start on PR.
- Machinery to reuse: patents localize-then-verify prompts + verdict schema (option3_claims_gemma_scale),
  metric_seam certificates (bootstrap_gate, ops/NullOps), bge/FAISS cascade, isomorphism census.

## Build plan
methods/claim_verification/: claims.py (atomic-claim extraction), evidence.py (pool construction +
retrieval), verify.py (localize-then-verify judge), sourcing.py (source census), novelty.py (FAISS
differentiate), metrics.py (doc-level metric derivation), controls.py (shuffled-evidence placebo,
length stratification). Pilot: 600 balanced k≥3 PRs on sk3 (Gemma or 70B server).

## Fact-pool design (2026-07-07 brainstorm w/ user)

### What FActScore/FactEval-line work uses (for reference)
- **FActScore** (Min et al., EMNLP 2023): atomic-fact decomposition -> dense retrieval (GTR) over
  **Wikipedia** for the target entity -> per-fact support verdict. Alt corpus: Google Search API.
- **SAFE** (Google, long-form factuality): atomic facts -> **Google Search** as the evidence base.
- **RealFactBench / OpenFactCheck / LLM-AGGREFACT**: aggregations; evidence = curated KBs,
  grounding documents, or web search. **ClaimDB**: structured databases as evidence.
- Common thread: the evidence base is chosen to match the CLAIM TYPE (encyclopedic->Wikipedia;
  current events->web search; numeric/structured->databases).

### KEY DESIGN PRINCIPLE (user): different domains need DIFFERENT fact bases
The fact base must hold the documents that could LICENSE each domain's claims:

| domain | claim types | fact base (tiered) |
|---|---|---|
| **press releases** | self-promotional: product/financial/partnership/superlatives ("first", "largest") | T1 own body (internal consistency); T2 **covering news articles** (37.9M news_articles.csv, 794k mappings — did journalists repeat/verify it?); T3 **company's own prior PRs** (superlative/novelty check: "first ever" vs their history); T4 structured: SEC/EDGAR filings for financial claims (future) |
| **news articles** | event/quantity/attribution about the world | T1 own body; T2 **same-story cluster**: other outlets' articles on the same event ±3 days (homepage corpus gives this cross-outlet automatically once fulltext lands — corroboration = how many independent outlets assert the same fact); T3 wire copy (AP/Reuters as ground layer); T4 Wikipedia/web for encyclopedic background |
| **peer review** | contribution claims ("we improve X by Y", "first method to Z") | T1 own paper internals (tables/experiments = the FactEval move); T2 **cited papers + related work** (baseline-comparison claims); T3 retrieval over the field's corpus (novelty: does prior work already do Z? = patents prior-art analog) |

The unifying op stays the same (localize-then-verify); only the POOL changes. This is exactly the
metric_seam evidence-op pattern: the pool IS the op; NullOps twin = foreign pool.
**The tier structure is itself a measurement**: T1-support = internal consistency; T2-support =
external corroboration; T1-but-not-T2 = unechoed self-assertion (PR spin signature);
T2-without-T1 = under-substantiated relay. These RATIOS are the cross-domain comparable quantities
for the journalism<->patents<->peer-review isomorphism.

### GEPA on extraction: objective redesigned (fidelity_scalar doesn't fit extraction)
ext_fidelity = .30 reliability (re-run claim stability, token-Jaccard matching)
             + .30 groundedness (claims locatable in own body via retrieval top1)
             + .25 coverage (head's numbers/entities captured by claims)
             + .15 yield (>=2 parseable claims/doc)
Gemma extracts; GLM-5.2 mutates prompts from failure summaries; 3 rounds x 4 mutations;
probe = 24 docs/domain (PR / news_articles / peer-review modeling texts).
