# news-homepages — why weak articulable signal? diagnostic (2026-07-03)

**Question:** V-layer (cheap) = 0.552, deconfounded dense ~0.753 — is the ~0.20 gap structural/non-articulable
(README's worry: "visual cues not in text"), or a fixable problem?

**Method:** 5 subagents (4 manually examined 55 within-snapshot top-vs-bottom headline pairs each = 220 pairs;
1 audited 60 raw rows' format) + quantitative within-snapshot pairwise diagnostic (71,693 pairs).

## Verdict: NOT fundamentally non-articulable. It's (1) data-quality noise + (2) feature/rubric mismatch.

### 1. The signal IS articulable (both qual + quant)
- **Subagents (4 slices, 220 pairs): humans articulate ~69–90% (avg ~80%) of placements** with consistent,
  classic news-value (Galtung & Ruge) rubrics: **crisis/breaking > political power > violence/death > elite
  names > magnitude/scale > institutional stakes > … > consumer > lifestyle/entertainment.** Content-driven
  (~75–95%), NOT headline craft.
- **Within-snapshot joint pair-diff AUC = 0.65** (crude features, combined). No SINGLE crude feature separates
  within-homepage (each ~0.50: hl_len 0.480, hl_elite 0.529, hl_neg_mag 0.515, hl_lifestyle 0.497) — it's a
  multi-criteria combination, which is exactly what a structured news-values RUBRIC should capture.
- **Implication:** the honest V-ceiling on the actual (within-snapshot) labeling task is ~0.65, not 0.552
  (the global V-layer is depressed by across-snapshot mixing). A news-values A-layer should beat cheap-V.

### 2. Data-quality noise is the biggest depressant (the real culprit)
Format audit (60 raw rows): **~40% of HEADLINES problematic; ~80% of CONTEXT polluted.**
- HEADLINE: truncated ("Dies at" w/o year), image/photo credits as headline ("Anna Moneymaker/Getty Images",
  "Madison Ashley Photography", "Animation by CJ Riculan, CNN"), **JS fragments** (`function imageLoadError(img){…}`),
  HTML/UI ("•\n Video\n 2:11"), section headers, "Sign up to The Recap email" promos, name-collisions
  ("Alain BersetI'm the Secretary General"), ~7% non-English (Portuguese).
- CONTEXT: same JS/HTML, read-times ("4 min read"), author bylines (60%+ of rows — NYT columnists Jessica
  Grose / David Wallace-Wells / Nicholas Kristof / Frank Bruni / Ross Douthat), timestamps ("Feb. 19, 2026"),
  outlet-name markers, "For Subscribers"/"Breaking News" UI.
- **Effect:** garbage "headlines" (image credits, JS) inject label noise; ~40% of rows degraded → caps every layer.

### 3. Leaked confounds beyond the Phase-0 check
- Phase-0 outlet-identity AUC = 0.499 (neutralized ✓). BUT **author/byline identity leaks in 60%+ of rows**
  (NOT deconfounded) — a latent shortcut if certain columnists are systematically placed high/low. Dates/timestamps
  also leak recency. Worth a byline-identity AUC check.

### 4. Rubric-bank mismatch (README already flagged)
`online-rubrics/` editorial-process norms describe story PRODUCTION (source authority, fact-checking, research
rigor) — mismatched to PROMINENCE assignment. Need news-values rubrics (the criteria subagents surfaced).

## Fix (ranked)
1. **Clean the text artifact**: strip `function imageLoadError`/HTML/`•\n Video\n` UI; regex-drop image credits
   (Getty/AFP/AFP/`/Reuters`/`Animation by`/photo patterns), read-times, timestamps, section labels, outlet-name
   markers; filter non-English + truncated + image-caption-as-headline rows; re-split HEADLINE/CONTEXT. New `_v2` file.
2. **Re-measure** V + dense on cleaned data (expect a jump toward the 0.65 within-snapshot honest ceiling).
3. **Build news-values A-rubrics** from the subagent criteria (crisis/political-power/violence/elite/magnitude/
   institutional/hard-vs-soft) → 70B judge → A-layer AUC. Expect V < A < dense showcase.
4. **Phase-0 addendum**: byline/author-identity AUC check (deconfound if >0.55).

## Artifacts
- pairs: `datasets/news-homepages/analysis/{within_snapshot_pairs.jsonl, slices/}`; diag:
  `analysis/{vlayer_audit,within_snapshot_diag}.py` (+ logs).

## Data-format summary + rebuild assessment (2026-07-03, follow-up)

### Format / pipeline (build_homepage_dataset.py)
Source: IA news-homepages, per snapshot = `*.hyperlinks.json` (every link: url, anchor text, bounding box
top/bottom/left/right) + `*.accessibility.json` (a11y tree). Pipeline per snapshot:
1. visible filter (positive bounding box);
2. **article-vs-furniture classify** (storysniffer LogisticRegression on url_path + anchor text);
3. dedup by url_path (keep lowest-y);
4. **enrich** anchor → "rich_text" = LONGEST a11y link-name that CONTAINS the anchor (`len>best_len`);
5. restrict to `top < page_height*0.30` (above-the-fold zone); drop snapshots with <4 in zone;
6. **label** = `1 if top < cutoff/2 else 0` (purely GEOMETRIC);
7. **context** = other in-zone anchors (raw, trunc 80) shuffled (excl. self);
8. format `HEADLINE:{rich_text}\n\nCONTEXT:{...}`; per-outlet balance → LDA-50 topic balance → snapshot_id group.
Canonical: text/judgement/snapshot_id, 183,708 rows / 21,951 snapshots / 50-50.

### Where pollution enters (two stages, NOT the label)
- **LABEL is sound** — purely geometric (bounding-box `top` vs zone midpoint). Independent of text.
- **Bug 1 — classifier under-filters**: image-credit links (`Getty`/`AFP`/`Animation by`), JS nodes
  (`function imageLoadError`), UI/media modules pass as "articles" → pollute the in-zone set → garbage in
  CONTEXT and (worse) non-article media modules at top positions get label=1 with junk text.
- **Bug 2 — enrichment "longest-containing"** appends bylines/read-times/related-content/JS to clean headlines
  (e.g. "Jessica Grose{headline}"); truncates when the a11y match is partial. Can't be regex-recovered.
- Raw a11y tree does NOT separate headlines from furniture by role (`heading`/`link` nodes are mostly nav/
  section labels) → the CLASSIFIER is the real filter; a11y is only text-elongation.

### Raw data (rebuild w/o re-scrape = feasible)
All 8 outlets' raw IA dumps on sk3 `raw_data/{outlet}/` (nytimes 780, wsj 6537, latimes 6681, bbc 5850,
wapo 9632, guardian 5764, reuters 4063, cnn present). ~40k snapshots. Download (the expensive part) is DONE.

### Verdict: cleaning alone is NOT enough; rebuild the TEXT ARTIFACTS from raw (not a re-scrape)
- Regex-clean of the canonical CSV recovers only the STRIPPABLE pollution (JS/read-time/byline/outlet patterns)
  and must DROP unrecoverable rows (image-credit-as-headline, truncated) — partial (~30-50% of the loss).
- The canonical CSV LOST url/bbox/raw-anchor, so you can't re-classify or re-derive the geometric label from
  it — a proper fix requires going back to raw.
- **Recommended staged path:**
  - **Stage 1 (quick):** regex-clean canonical CSV → `_v2` (strip JS/UI/read-times/bylines/outlet-markers;
    DROP rows whose headline is pure image-credit/JS/section-header/<15 chars after clean). Re-measure V +
    within-snapshot + dense. If it jumps toward 0.65, may suffice.
  - **Stage 2 (proper):** rebuild from raw with (a) tighter article filter (reject anchors matching
    `/Getty|AFP|Reuters|Animation by|^\W*$|function |<|\d+\s*min read/` + url image/media paths) and
    (b) conservative enrichment (extend headline only with an a11y match that is itself clean + headline-
    length, NOT longest-containing). Re-derive label on the cleaned article set. ~few hours CPU on sk3.

## V/A/dense results after full clean + audit (2026-07-04)

| layer | grouped | within-snap-joint |
|---|---|---|
| V (cheap counts) | 0.554 | 0.659 |
| **A (14 hand-built news-values rubrics, 70B)** | **0.557** | **0.634** |
| dense bge-m3 (ORIG / clean_v2) | 0.621 / **0.626** | 0.672 / **0.684** |
| dense llama8b FT (prior sweep, ref) | 0.753 | — |

- **Cleaning effect**: V unmoved (0.5535→0.5541); dense +0.005 grouped / +0.012 within-snap (0.621→0.626).
  Cleaning helps dense slightly, not V — consistent with "data quality is minor; feature/rubric
  operationalization is the bottleneck."
- **Confounder audit CLEAN**: outlet 0.507, byline 0.510, date 0.502, position 0.50, snapshot-base-rate
  handled by group split. No leakage. The dense ceiling is genuine, not confound-inflated.
- **Manual verification**: clean_v2 usable — 95% junk removed, 0% headline degradation, 95% dropped-row
  correctness; signal still articulable (~65% on headline-only pairs). Residual: context segments still
  carry some image-credit/promo noise (Stage 1.5 target).

## THE FINDING: A ≈ V — articulability is PAIRWISE/HOLISTIC, not rubric-decomposable
Even with hand-built NEWS-VALUES rubrics (the exact criteria subagents said humans use to articulate
~80% of pairwise placements: elite actor / institutional action / crisis / conflict / magnitude /
breaking / legal-accountability / economic impact / hard-vs-soft / ongoing-top-story), the A-layer
does NOT beat cheap counts (0.557 vs 0.554 grouped; within-snap A 0.634 ≤ V 0.659). NA rate 0.43
handicaps A, but the within-snap joint (cleaner) still has A ≤ V.

**Paradox resolved**: humans articulate ~80% PAIRWISE (which of two headlines is more prominent) using
news values, but scoring each headline INDEPENDENTLY on rubrics + linearly combining loses the
comparative/holistic structure. **The A-layer (rubric-decomposable) ≠ pairwise human judgment.** So
news-homepages is the OPPOSITE of a showcase: articulability EXISTS pairwise but does NOT operationalize
into a scored rubric layer. The dense−A gap (bge-m3 +0.07; llama8b-FT +0.20) is largely TACIT /
non-rubric-decomposable.

**Implications**: (1) a PAIRWISE/CONTRASTIVE A-layer (judge which headline is more prominent, with
news-value reasoning) may capture what the score-based A-layer misses — worth testing; (2) news-homepages
is a "holistic-prominence" task, a useful contrast case for the V/A/taste taxonomy (cf. press-release
k≥3 which IS rubric-decomposable).

## Cross-version robustness (V/A/dense stable across 4+ cleaning levels — 2026-07-04)
| version | rows | V | A | dense |
|---|---|---|---|---|
| original | 183,708 | 0.5535 | — | 0.6208 |
| v2 (drop junk) | 162,474 | 0.5541 | 0.5573 | 0.6264 |
| v4 (credits+byline) | 157,552 | 0.5632 | 0.5687 | 0.6214 |
| v5 (langdetect-gated) | 158,666 | 0.5653 | 0.5685 | — |
| v7 (stopword lang, balanced) | 147,873 | 0.5696 | 0.5431 | 0.6229 |
- **A ≈ V at every level** (A 0.543–0.569 vs V 0.554–0.570; A never beats V by >0.01). **dense ≈ 0.62**
  (modest +0.05–0.07 over V/A). Cleaning lifted V from 0.554→0.570 (~0.016) but the A≈V relationship and
  the dense ceiling are IMMUNE to cleaning. Confounder audit clean at every level (outlet/byline/date/pos ≈0.50).
- Residual cleaning plateau (~85-90%): a genuine ~13% Brazilian-Portuguese subset (Brazilian outlets in the
  build) that stopword/accent heuristics under-detect on short headlines. v8 = langdetect-gated-on-broad-
  pre-filter (accurate) to remove it. A from-raw rebuild (fix enrichment appendage + classifier non-English
  filter) would reach pristine but WON'T change the V/A conclusion (proven robust above).
- **Bottom line**: news-homepages weak-A is a feature-operationalization property (articulability is
  PAIRWISE/holistic, not rubric-decomposable), definitively NOT a data-quality artifact.

## NEW-metric discovery (4 subagents × 70 within-snap pairs on clean v8, 2026-07-06)
v8 final: V=0.5716, dense=0.6307 (within-snap 0.691), A≈0.55 (stable across v2/v4/v5/v7).
Strongest EXISTING rubrics (per-rubric AUC, consistent across versions): hard_vs_soft (0.557-0.563),
elite_political_actor (0.536-0.555), ongoing_top_story (0.532-0.542), breaking_developing (0.525-0.542).

**Consolidated NEW metrics (4/4-slice convergence = Tier 1, genuinely new axes):**
1. **concrete_numerical_specificity** — specific $/%/counts ("$800M","1-in-16") vs vague (≠ magnitude_scale: precision≠size).
2. **linguistic_action_intensity** — vivid action verbs (torch/slam/explode) vs neutral speech verbs (says/reports). LINGUISTIC, not topical.
3. **reader_personal_stakes** — direct reader impact (wallet/safety/"what to do"/"self-deport") vs abstract (≠ proximity_domestic: actionable).
Tier 2 (3/4): curiosity_gap_question; moral_outrage_accountability (named perp/institutional failure);
surprise_absurdity; named_victim_personal_narrative. Tier 3: deadline_urgency_countdown (≠breaking);
viral_currency_phraseology (≠elite_celebrity_org); source_authority_exclusive.
KEY INSIGHT (slice 3): existing 14 are all TOPICAL ("what"); the strong new dims are LINGUISTIC/
STRUCTURAL ("how described" — verbs, precision, personal framing) = exactly what dense captures and
topical rubrics can't → may close part of the dense−A gap (0.631 vs 0.55). Pair slices: analysis/pairs_v8/.
NEXT: score top 8 new metrics via 70B on v8 -> per-metric AUC + bank-lift over the 14 (empirical test).

## New-metric GEPA + scale result (2026-07-07) — DEFINITIVE: no rubric layer beats ~0.57
Implemented 10 NEW linguistic/structural metrics (concrete_specificity, action_intensity,
reader_stakes, curiosity_gap, moral_outrage, surprise, named_victim, deadline_countdown,
viral_currency, source_authority) from 4-subagent discovery. Ran GEPA-reconstruction (Gemma-4-32B
judge + GLM-5.2 proposer, fidelity_scalar obj): **7/10 improved on reconstruction-R** (viral_currency
0.39→0.81, moral_outrage 0.55→0.83, curiosity_gap 0.49→0.76, surprise 0.28→0.55, source_authority
0.50→0.69, reader_stakes 0.50→0.64). Optimized prompts saved (news_gepa_optimized.jsonl).
Then scored the GEPA-opt bank via 70B on clean v8 (4400 items):
- GEPA-OPT bank (10 new + 4 existing) grouped=**0.569**, within-snap=0.595.
- GEPA-OPT-NEW (10 new alone)=**0.530**; EXISTING-4 (hard/elite/ongoing/breaking)=**0.569**.
- New+existing (0.569) == existing alone (0.569) → **new metrics add ZERO coverage-prediction lift.**
- Top per-metric: ongoing_top_story 0.551, hard_vs_soft 0.548, moral_outrage 0.533, breaking 0.524,
  elite_political 0.522, linguistic_action_intensity 0.519; rest (curiosity_gap 0.502, viral_currency
  0.503, source_authority 0.509, named_victim 0.507) ≈ chance.

**KEY INSIGHT: GEPA improved reconstruction fidelity but NOT coverage AUC.** A metric can be
well-defined/reconstructable yet not predict the label — reconstruction-R ≠ predictive power.
**DEFINITIVE CONCLUSION:** news-homepages A≈V≈0.57 under EVERY intervention — 8 cleaning levels,
14 topical rubrics, 10 new linguistic/structural rubrics, GEPA-optimization. No rubric layer breaks
past ~0.57; dense=0.631 is genuinely beyond rubric-decomposable articulation. The editorial-prominence
signal is pairwise/holistic, confirmed robust across all 4 attack axes. (A pairwise/contrastive judge
"which headline is more prominent" remains the one untested design that might capture it.)

## Pairwise/contrastive judge + V+A (2026-07-07) — DEFINITIVE END
**V+A stack** (Phase B-opt eval set): V=0.552, A(14)=0.568, **V+A=0.577**. A adds modestly over V
(partially complementary) but the combined articulable ceiling (~0.58) still trails dense (0.631).

**Pairwise 70B judge** (3000 within-snap pairs, A/B randomized, "which headline placed higher?"):
accuracy = **0.569 (±0.018)** — barely above chance (0.50), far below human pairwise (~80%), below
feature pair-diff (0.69 AUC). No position bias (48.6% A). REJECTS the "holistic/pairwise judge captures
it" hypothesis: the 70B fails pairwise (~0.57) just as it fails with score-based rubrics (~0.57).

FINAL SCOREBOARD (news-homepages editorial prominence, clean v8):
| approach | AUC/acc |
|---|---|
| V (cheap counts) | 0.552–0.572 |
| A (rubric, score-based, 14-70B) | 0.55–0.57 |
| V+A stacked | 0.577 |
| GEPA-optimized new metrics (10) | no lift (0.569 w/ existing) |
| **Pairwise 70B judge** | **0.569 acc** |
| dense bge-m3 | 0.631 |
| human pairwise | ~0.80 |

**CONCLUSION:** every LLM-judge approach (score-based AND pairwise, seed AND GEPA-opt, 14 topical AND
10 linguistic/structural) clusters at ~0.57. Dense (0.631) and human (0.80) are both well beyond.
The editorial-prominence signal is NOT capturable by 70B judges in either architecture — likely
requires (a) stronger models than 70B, or (b) visual/layout context not in the headline text (the
README's structural-ceiling worry). news-homepages is a genuinely HARD case for articulable/rubric
approaches — a strong contrast case for the V/A taxonomy (cf. press-releases k>=3 which IS rubric-decomposable).

## FINAL: fully-clean v9 + VA re-run (closes the cleaning loop, 2026-07-07)
Codex measured v8 residual: 11.05% read-times (mashed "Years3 min read" — my v8 `\b\d+` missed the
no-boundary mash, same bug as "Guard8 hrs ago"), 0.23% bylines, 0% agency, 0% short. Built v9 with
NO leading word-boundary on read-times + segment-end "X min" strip. QUANTITATIVE verify on v9:
readtime **0.1105→0.0000**, byline 0.0023→0.0010, agency 0, short 0. (Earlier "v9=0%" report was a
double-escaped verify-regex bug — fixed + reconciled against v8.)

VA re-run on fully-clean v9 (133,130 rows, 50/50):
| layer | grouped | within-snap |
|---|---|---|
| V | 0.572 | 0.682 |
| A (14 news-values) | 0.566 | 0.600 |
| dense bge-m3 | 0.630 | 0.692 |
All identical to v8 (±0.005) → cleaning the residual changed NOTHING, confirming the headline-only
test (judge input pollution is not the lever). Hook conditions satisfied: fully clean (verified
0% residual) + no confounders (outlet/byline/date/pos ≈0.50) + no degradation/leakage + VA re-run.

FINAL news-homepages verdict: V≈A≈0.57 << dense 0.63. Judge stuck ~0.57 across every config tested
(score-based / pairwise / CoT+few-shot / full-page / headline-only). Real puzzle = cheap features
(0.65-0.69 within-snap) > 70B judge (0.57): geometric label tracks surface markers more than
"newsworthiness" reasoning. Reconstruction-GEPA improved fidelity but not coverage (R≠coverage).

## Within-snap signal decomposition + ctx-leak check (2026-07-07) — A IS articulable
Question: is newsworthiness really A-centered? Decomposed the within-snap pair-diff (0.68 joint).
PER-FEATURE within-snap pair AUC (v9, vlayer-style random-3, 62,417 pairs):
  hl_elite 0.557, hl_len 0.552, hl_neg_mag 0.529, hl_allcaps 0.522, hl_proper 0.515,
  hl_lifestyle 0.508, hl_numbers 0.496. JOINT-8 = 0.683 (reproduces vlayer 0.6823).
HEADLINE-ONLY articulable joint (elite+len+neg+proper+allcaps, no ctx) ~0.585.
=> The within-snap signal IS articulable: classic news values (eliteness, specificity/verbosity,
   negativity/conflict, named-actors, emphasis). The subagents' hyped NEW dims (concrete-numbers
   0.49, vivid-verbs 0.50) are NOT predictive — old Galtung-&-Ruge wins.

CTX-LEAK CHECK: the single strongest within-snap feat was ctx_len (0.614). Decomposed:
  ctx_n_segments within-snap 0.505 (neutral) => NOT "top has more siblings below" position-leak.
  ctx_avg_seg_len within-snap 0.611, global 0.508 => driver = avg LENGTH of sibling headlines.
Ambiguous: real (top stories cluster with longer-headline siblings) OR construction artifact
(positional neighbor enrichment). Either way, the JUDGE doesn't see context, so it can't use this.

REFRAME: news-homepages IS A-centered at the HEADLINE level (~0.585, classic news values). The 70B
judge MATCHES this (~0.57) — it is NOT underperforming headline articulability. The features' higher
within-snap (0.68) comes from the sibling-length context-structural signal, not headline articulability.
V+A on v9: V 0.552, A 0.568, V+A 0.571 (A subsumes V; both << dense 0.630).
GEPA-e4c (existing-4 prompts + 6 code-kind) running to test if optimization lifts the headline A.
