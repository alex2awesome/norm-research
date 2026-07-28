# VAT Paper (Paper #3) — Master Plan

*2026-07-22. Owner: this thread (user put me "squarely in charge"). Living document.*

## 0. What this paper is (and is not)

**The four-paper program** (user, 2026-07-22):
- **Paper #1 — Terms and codability** (= metric-lexicon paper, `project_metric_lexicon_paper_plan`).
- **Paper #2 — Articulation upper bounds** (= prompt-optimization / OSL certification; the
  certified prompt-space bounds. Formerly "Paper #1" in the 2026-07-20 split-decision note).
- **Paper #3 — Articulation and preferences; VAT** ← **this paper (mine).**
- **Paper #4 — Tacit knowledge: how it is measured and learned** (= tacit-installation-channels
  capstone, `project_tacit_installation_channels`).
- **Order:** #1 → #2 → {#3, #4} (the last two may swap; TBD).

**Identity.** This is **Paper #3** — the substantive, broad-audience paper on how the
articulability of a preference behaves: *"preference articulability gaps under construct-faithful
instruments, plus certified bounds over all prompts."* It brings together **every VAT (Verifiable
/ Articulable / Taste) measurement** we've made across preference-based datasets, sets it against
the metric-seam and y-prediction work, and closes the two remaining structural holes:
**(a) heterogeneous preferences** and **(b) different preference variables.** (Under the earlier
two-paper split decision this was "Paper #2 with heterogeneity in it" — same content, renumbered.)

**Boundaries.**
- **NOT Paper #1 (terms & codability / metric-lexicon):** that owns the metric *space* —
  codability, clustering, census. This paper owns the *decomposition of preference outcomes*
  into V/A/T and the *residual that survives every control*.
- **NOT Paper #2 (articulation upper bounds / PO / OSL):** that owns the certified prompt-space
  bounds (DPI fixed-target cap, CR-3 missing-mass, OSL executor scaling). We *import* those
  bounds as a robustness layer; we do not re-derive them.
- **NOT Paper #4 (tacit knowledge capstone):** that owns how tacit knowledge is *gained/learned*
  (channels × types, RL/FT/patching). We measure the tacit *residual* of preference; we do not
  study its acquisition.
- **Standing constraint (reconstruction-only):** "optimized prompts" here means
  **fidelity-optimized (label-blind GEPA)**, NEVER AUC-optimized. Framing must be "gaps under
  construct-faithful instruments," not "gaps under best-possible prompts." No human studies;
  metrics never label-aware.

**Thesis (macro-structure §IV).** After decomposing every preference outcome into what code
recovers (V), what articulated criteria add (A), and what neither reaches (Taste), a real
residual survives — and its *size and character* depend systematically on (i) who/what the
preference is (heterogeneity) and (ii) which preference variable you measure (verdict vs
curation vs revealed attention).

---

## 1. Consolidated VAT master table (the spine of the paper)

All numbers are **outcome-y** decompositions: V = code/regex, A = articulated criteria scored
by an LLM judge, dense/C = dense-model or crowd ceiling. Ordering column is the finding.

| domain / run | y | V | A | A−V | dense / ceiling | regime |
|---|---|---|---|---|---|---|
| peer review (ICLR) | accept/reject | .611 | .676 | **+.071** | ~.77 | V<A<dense |
| legal · title_vii | outcome | .576 | .637 | **+.061** | T .654 / T⁺ .614 (corrected 2026-07-27; "T⁺ .77" was erisa's, misattributed) | V<A<T |
| legal · erisa_ltd | outcome | .548 | **.758** | **+.209** | T .644 / T⁺ .767 | V<T<A |
| patents · prior-art | claim "fell" | .601 | .623 | +.022 | text .654 | V<A thin |
| notice & comment | majority MADE | .595 | .592 | ~0 | .588 (≈char_len .595) | **flat/null** |
| code review (PR) | merge/flip | .576 | .615 | +.039 | dense .584 / repo-id .706 | V<A<repo-id |
| claim-matching | examiner cite | .516 | .573→.594 | +.06–.08 | retrievers ≤ chance | **A on top** |
| press release | pickup | .508 | ~.53–.58 | +.03–.07 | .584 honest | V<A<dense(confound) |
| math · Math.SE | accept∧score≥3 | .55–.58 | **.66** | +.09 | C **.79** | **big tacit band** |
| math · AoPS | same-approach | .544→.727 | ~.73 | ~0 | C .777 | thin/lexical |
| math · mathlib | accept/reject | **.68** | .46–.56 | **−.15** | C .736 | **INVERTED V>A** |

**Four regimes** (this is a result, not a nuisance): canonical V<A<dense (majority);
flat/null (N&C); inverted V>A (mathlib formal proofs); A-on-top with dense-at-chance
(claim-matching). The *outcome-y* A−V lift is small everywhere (+.02..+.09) except erisa
(+.21) — versus the large A-over-V lift when the target is the **judge** (the metric-seam F).

**NEVER re-quote (superseded / confounded):** PR first ladder V .582 / A .734 / T .749
(in-sample + repo confound); press-release dense .71 (publisher-identity confound; honest
.584); patents metadata-no-text .756 (self-match leak); legal GEPA "+.043" (→ +.012 noise);
peer small-n .759 (→ .699 court-grouped).

---

## 2. Where the metric-seam / y-prediction work sits

- **Metric-seam F = A/(V+A)** measures the seam against the **judge** (reconstruction target).
  10-task spectrum (humor .91 ≈ CW .90 > math .83 ≈ patents .83 > N&C .73 > … > peer .61).
  This is the *judge-side* seam and is large.
- **y-prediction** measures the seam against the **real outcome**. Scaled n≈400: legal .699 /
  patents .610 / peer .575. Four findings: (1) lexical ≥ forecast law (4 domains); (2) LLM
  advantage is judge-space only; (3) dense < lexical at our n; (4) GEPA autopsy = search
  failure not representation failure (one evolved prompt < rubric bank < dense).
- **The judge-vs-outcome asymmetry is the bridge into this paper:** A is large for the judge,
  small for the outcome. The VAT master table above is the *outcome* view; §III of the paper
  contrasts it with the *judge* view (metric-seam) to show the residual is about **outcome
  taste**, not instrument weakness.

---

## 3. The A/B/C taste-grid — already 80% of hole (b)

We built and deconfounded a **12-cell grid** (`build_topic_stratified.py`), where columns are
**aggregation states of preference** (= different preference variables on overlapping items):

- **A / expert-verdict (explicit):** accept/reject, merge, MADE — the master table above.
- **B / expert-curation (accountable institutional):** oral/spotlight, caption finalists,
  best-papers, landmark cases. Built: floor/headroom + V>0, **not yet full A-bank VAT.**
- **C / crowd-revealed (uncoordinated attention):** citations, reddit votes, most-read,
  most-emailed. Built + deconfounded. **Key finding: these are IMPACT cells, not
  articulability cells** — high topical floor, ~0 semantic headroom, fractal topic residual.

| cell | col | BoW floor | headroom | V | status |
|---|---|---|---|---|---|
| oral/spotlight (acad) | B | .505 | +.09 | .549 | articulable, needs A-bank |
| caption finalists (humor) | B | .546 | +.083 | .606 | articulable, needs A-bank |
| best_papers (acad) | B | .603 | +.15 | .681 | articulable, needs A-bank |
| citations (acad) | C | .806 | ~0 | .756 | IMPACT (era-fractal) |
| reddit-news | C | .579 | thin | .592 | register/attention |
| reddit-scotus (law) | C | .56–.60 | +.015 | .539 | crowd-favored |
| law_se | C | .49–.62 | ~0 | .532 | register/lexical |
| BBC most-read | C | .604 | thin | (format) | register/format |
| RoyalRoad (cw) | C | .521–.588 | — | .570 | narrative craft, ~0 topic |
| Wigleaf (cw) | B | .537 | ~0 | .570 | **taste-residual null** |
| humor contest (cw) | A | ~.45 | ~0 | ~0 | **taste-residual null** |

**The grid already delivers the headline of hole (b):** the seam is widest for
expert-verdict/curation (A>V, low floor + semantic headroom) and collapses for crowd-revealed
(impact = topic all the way down). But two things are missing (see §4).

---

## 4. Gaps / holes inventory

**H1 — A-bank never run on B/C cells.** B and C cells have only BoW floor/headroom + V>0.
To place them on the *same* V/A/T ladder as the master table, we must score the articulated
A-bank on them. (Core of hole b.)

**H2 — No within-domain preference-variable contrast.** We compare y-variables *across*
domains (confounded by domain). The clean design runs V/A/T on **verdict vs curation vs
revealed on the SAME items**. Academia is the one domain where all three exist on overlapping
papers (accept/reject → oral/spotlight → citations). (Core of hole b.)

**H3 — No noise ceiling / rater model.** Every gap is attenuated by rater disagreement we've
never measured. Sanmi items 5/6 + Dan's IRT point. Without it, (A−V)/(ceiling−V) is
un-normalizable and cross-domain comparison is unfair. (Core of hole a.)

**H4 — Taste never split into community vs personal.** The whole VAT frame promises
Taste = Community Taste + Personal/Faultless. We have never measured the split (high-agreement
residual = community/articulable-in-principle; low-agreement residual = personal/tacit). (Core
of hole a.)

**H5 — Sub-population VAT not integrated.** The subcommunity screen (#60) measured *bank
reweighting* per community (math = shared backbone; cpp/python = orthogonal-but-confounded) but
never tied it to the V/A/T decomposition or the agreement-based Taste split. (Hole a.)

**H6 — Fidelity-optimized A-row only for N&C (+ planned peer).** Paper #2's instrument must be
construct-faithful. The N&C GEPA result (fidelity↑ ⇒ AUC↓, because baseline-A rode
unarticulated judge residue) is *load-bearing* — it must be replicated on ≥2 more banks as the
"baseline-A overstates articulability" robustness column.

**H7 — Cross-task comparability.** Splits/y-defs differ per run. Need one protocol
(grouped-CV unit declared per domain, threshold-free AUC, noise-ceiling normalization).

---

## 5. HOLE (a) — Heterogeneous preferences — design

Three layers, built on datasets that carry **per-rater or distributional** labels.

**Rater-bearing datasets:** peer review (multiple OpenReview reviewers/paper — flagship for
IRT), humor caption contest (crowd rating distribution, ~thousands/caption, ceiling .938),
math.SE (vote distribution), reddit cells (vote distribution). Single-verdict datasets
(code-review, most legal) get community-level heterogeneity only (H5), not item-level.

**Layer 1 — Noise ceiling (H3).** Fit a rater model per dataset:
- IRT / mixed-effects (item difficulty × rater severity) where discrete multi-rater labels
  exist (peer review).
- Beta-binomial / crowd-variance ceiling where distributional (caption, math.SE, reddit).
- Output: per-domain **noise ceiling** C_noise = max achievable AUC given disagreement, and
  per-item agreement score. Report the **disattenuated gap** (A−V)/(C_noise−V) so domains with
  noisy labels aren't unfairly penalized. This is also the honest denominator for the F seam.

**Layer 2 — Community vs personal Taste split (H4).** On the residual (items where the V+A
model is confident-but-wrong, or the dense−(V+A) gap):
- split by inter-rater agreement. **High-agreement residual → Community Taste** (there IS a
  fact of the matter the bank missed — articulable-in-principle, a discovery target).
  **Low-agreement residual → Personal / Faultless** (irreducibly tacit; Kölbel/MacFarlane
  faultless disagreement).
- Deliverable: for each rater-bearing domain, the triple (V, A, Community-Taste,
  Personal-Taste) as a stacked bar — the paper's money figure.

**Layer 3 — Sub-population VAT (H5).** For domains with defined sub-communities (peer venues,
math tags, code repos, CW genres, humor topics):
- run V/A/T *within* each sub-population;
- reuse the subcommunity screen (d_spec = within − transfer) to test whether A needs
  community-specific reweighting (already: math shared backbone; revisit cpp/python
  de-confounded);
- decompose Taste into **between-community** (predictable from community identity) vs
  **within-community** (personal). This is the community-level mirror of Layer 2.

**First experiment (a):** peer review. It has multiple reviewers *and* venues *and* the acad
B/C cells on overlapping papers — the one place all three layers land at once. Fit reviewer
IRT → noise ceiling → community(venue) vs personal split → disattenuated V/A/T. If this closes
cleanly it is the template for the rest.

### 5b. Discovery-as-probe: metric discovery per subfield as the heterogeneity instrument
*(2026-07-23 ideation, user-directed: "I liked our metric-discovery work, discovering new
metrics in different subfields" — marry the mining pipeline to hole (a).)*

Four designs, ordered by novelty; D2/D4 are the ones the mining pipeline uniquely enables.

**D1 — Mined-bank transfer matrix (articulable heterogeneity).** For each subcommunity c
(ICLR subject areas / math.SE tags / subreddits / CW genres), mine a bank B_c label-blind
(mining from text/judge rationales only — reconstruction rule), degeneracy-filter it
(A-bank audit: mined banks 54–68% degenerate), then fill the matrix AUC(B_i → community j).
Readouts: pooled-bank floor (shared backbone), diagonal excess d_spec = within − transfer
(articulable heterogeneity), and per-community dense − (V+A_c) (residual not articulated even
by community-specific mining). Generalizes the #60 screen from *reweighting a fixed bank* to
*discovering different banks*. Controls: matched-n mining per community; topic-stratified eval
(community-specific criteria may be topic proxies).

**D2 — Lexicon-difference readout (bridges Paper #1).** Compare the mined lexicons themselves:
split B_c into universal slice (criteria recovered in most communities) vs community-specific
slice. Test whether the community-specific slice adds AUC over the universal slice
*in-community* (and subtracts nothing out-of-community). Distinguishes "communities talk
differently" (lexical dialect, Paper #1's census) from "communities *want* differently"
(preference-relevant dialect — ours). A criterion can be lexically distinctive but
preference-irrelevant; this is the test that separates the two.

**D3 — Mixture-of-banks at the rater level.** Where per-rater labels exist (peer reviewers,
crowd distributions): fit rater-specific weightings over the shared bank (IRT slope ×
criterion-weight vector per rater). Heterogeneity readout = do rater weight-vectors CLUSTER
(community taste: discrete schools of preference) or SCATTER (personal taste)? Cluster count/
silhouette vs a permutation null. Connects Layer 1–2 to the bank itself.

**D4 — Discovery loop as the operational Community/Personal Taste split (the flagship idea).**
Layer 2 calls the high-agreement residual "a discovery target" — make that literal. Take items
where pooled V+A is confident-but-wrong yet raters AGREE (community-taste residual candidates);
run metric discovery ON that residual slice, per subcommunity; re-score and re-fit.
- Residual that mining FIXES in-community ⇒ was Community Taste (articulable-in-principle,
  community-specific — now articulated).
- Residual that mining CANNOT fix ⇒ operationally personal/tacit.
This turns Taste = Community + Personal from a definition into a *measured* decomposition:
T_discoverable vs T_residual, with the discovery loop itself as the instrument. It is also the
clean bridge to Paper #4 (what remains after best-effort articulation = the tacit payload).
Guard: discovery must stay label-blind (mine from rater rationales/review text on the residual
items, never from the y) — otherwise circular. Guard 2: hold out a residual test-split BEFORE
mining so "fixes the residual" is out-of-sample.

**Where to run:** peer review venues/subject-areas (flagship — raters + venues + 3 y's);
math.SE tags (known shared-backbone ⇒ negative control: D1 diagonal excess should be ~0);
subreddits (register-heterogeneous ⇒ positive control candidate); CW genres. Sequence: Layer 1
noise ceiling FIRST (else heterogeneity is conflated with label noise), then D1/D2 transfer
matrix, then D4 residual-discovery loop on the domain where d_spec > 0.

---

## 6. HOLE (b) — Different preference variables — design

The A/B/C grid is the axis; finish it and make it *within-domain*.

**Step 1 — A-bank VAT on B/C cells (H1).** Score each domain's articulated A-bank (the same
bank used for its verdict-y) on its B and C cells; add V (codegen) and dense. Places all 12
cells on the master ladder. Prediction: B cells show A>V with real headroom; C cells show
A≈V≈topic-floor (the seam is *ill-defined* for revealed attention because the construct is
impact, not quality).

**Step 2 — Within-domain preference-variable contrast (H2).** Flagship = **academia**, the
only domain with all three preference variables on overlapping papers:
- **verdict:** accept/reject (ICLR).
- **curation-B:** oral/spotlight, best-paper.
- **revealed-C:** citation percentile.
Run identical V/A/T on the same paper population under each y. Isolates "how the articulability
seam depends on which preference you measure, holding domain + text constant." Secondary
within-domain contrasts: law (verdict = case outcome; revealed = judicial-citation pct /
r/scotus votes); news (verdict = editorial pickup; revealed = most-read/most-emailed).

**Step 3 — Explicit vs revealed axis.** Cross-cut the grid: verdict/curation are **explicit**
preferences; citations/attention are **revealed**. Report the seam as a function of this axis.
Expected law: articulability high for explicit-quality judgments, ~0 for revealed-attention
(which is topic/impact). This reframes "different preference variables" as a *construct* axis,
not just a label axis.

**First experiment (b):** academia three-y contrast (Step 2). Data already built
(oral_spotlight 33.5K, openalex_citations v3 28K, ICLR verdicts). Only need the A-bank scored
across all three on the shared population + V + dense. No new collection.

### 6b. Scale-out feasibility matrix (2026-07-23 scoping sweep, 4 agents, all repo-verified)

Which domains support the same-item multi-y contrast, and at what cost:

| domain | y-pair on SAME items | status / cost | key caveat |
|---|---|---|---|
| academia | verdict/curation/revealed | **DONE** (Cycle 1) | revealed rides topic floor |
| **N&C** | outcome-majority vs agree-vs-disagree vs responded-or-not (all from same `labels` list per comment) | **FREE** — pre-GEPA 198-rubric scores in `v4/nc_scores_shard{0..4}.npz` locally; just attach y's + strict-common aggregation | all EXPERT axes (verdict-type contrast, no crowd); crowd axis = co-signature/near-dup counts, buildable, not built |
| **humor captions** | finalist (B) vs crowd rating (C), 18,838 captions / 227 contests, sk3 `datasets/caption_contest/built/` | **CHEAP** — sync join + port `score_va_gemma_3y.py`; A-bank never scored on captions (H1) | MUST use hard-negative build (crowd_mean .938 vs random entries but .339 anti-predictive vs crowd-loved); no verdict axis exists; bank choice open (364-rubric standup vs 40 medoid vs mine caption-specific) |
| **math.SE** | accepted (asker verdict) vs score (crowd) — separate fields, same ~100K answers; a01–a14 A-verdicts on 22,521 | **MEDIUM** — checked-in pool pre-filtered to accepted∧score≥3 vs score≤0 corners, so accepted⊥score variance is truncated; clean contrast needs rebuild from un-binarized pool (sk3 raw_dump) | prior A scored by Qwen-122B (anchor-only rule → re-score with Gemma-4 for the new run) |
| **legal** | verdict (title_vii/erisa district) vs district-citation-percentile | **MEDIUM (1–2d eng, no new data)** — citation_percentile recipe is circuit-only by filter, not by data; `clusters_slim.csv` (10M) covers district too; recover cluster_id from upstream `slice_opinions.jsonl.gz` | landmark (curation) cell DOES NOT EXIST — placeholder only; reddit-scotus & law_se are disjoint item-spaces (comment/answer ≠ case) |
| **patents** | allowance (verdict) vs forward citations (revealed) | **LARGER** — forward-citation column not built but edge list on disk (`g_us_patent_citation.tsv.zip`, only ever read backward); NO assembled A-bank or Gemma scorer for patents | nested design only (abandoned apps can't be cited → revealed-y on granted subset); examiner-LOO leniency confound must be controlled (dominates outcome-y, .681) |
| news / CW / reddit | — | **NOT FEASIBLE** without new collection | pickup vs most-read = disjoint corpora (PR bodies vs BBC headlines); RoyalRoad/Wigleaf/WritingPrompts = 3 unlinked platforms; all reddit cells single-label (votes only) |

**Scale-out launch order:** (1) N&C strict-common multi-y NOW (free; sanity gates = known
per-y numbers: outcome V.595/A.592, agree A.612 pre-GEPA, responded A.636); (2) humor caption
B/C after bank decision; (3) math.SE rebuild; (4) legal district-citation build; (5) patents
(biggest lift, weakest instrument). Protocol carried from academia: label-independent scoring
once per item, y attached per rung, strict-common set for apples-to-apples, topic-strat
robustness for every C cell.

---

## 7. Unified protocol + robustness

- **One VAT protocol (H7):** declare the grouping unit per domain (court/app/venue/repo/docket),
  threshold-free AUC, noise-ceiling-normalized gap, fidelity-optimized A as primary + baseline-A
  as the "overstates articulability" reference column.
- **DENSE STANDARD (user decision 2026-07-27): the dense column = ONE recipe everywhere.**
  Llama-3.1-8B + LoRA (r16/α32, lr 5e-5, batch 16, max_len 1024, 2 epochs,
  gradient-checkpointing), single full-data run (frac=1.0, NO scaling ladder), grouped
  80/10/10 split on the task's V/A grouping unit, selection on test / report CLEAN EVAL AUC
  (math-closure convention). **No TF-IDF, no frozen-embedding arms in the dense column** —
  those move to a labeled "lexical floor" appendix row if kept at all. Audit that forced this
  (2026-07-27): peer .690 was TF-IDF `D_lex_big`; N&C .588 was TF-IDF; claim-matching
  "dense-at-chance" was untrained bge-m3 (already retracted as ceiling evidence); only
  math legs + legal T were already Llama-8B LoRA. Driver:
  `methods/dense/run_dense_standard.sh`; bundles built by
  `datasets/notice-and-comment/v4/build_dense_standard_csvs.py` (peer rungs reuse the
  vat_3y stable title-grouped splits; N&C uses a docket-pure greedy 80/10/10 map shared
  across the 3 y's — NOTE agree eval/test pos .64-.66 vs train .50, docket-clustered label).
- **Fidelity-optimized A everywhere (H6):** replicate the N&C GEPA fidelity↑/AUC↓ test on peer
  and ≥1 more bank. Sonnet-via-CLI proposer (unmetered) as in N&C; do not switch to GLM without
  re-asking.
- **Import certified bounds from Paper #2 (articulation upper bounds):** DPI fixed-target cap +
  CR-3 missing-mass as the "the gap isn't just a bad prompt" defense (robustness appendix).

---

## 8. Execution order (proposed)

1. **academia three-y contrast** (b/Step 2) + **A-bank on B cells** (b/Step 1) — no new data,
   fastest, highest-signal; also produces the paper's central within-domain figure.
2. **peer-review heterogeneity** (a/all three layers) — reuses the same academia population;
   reviewer IRT → noise ceiling → community/personal Taste split.
3. **fidelity-optimized A replication** (H6) on peer bank (test the N&C law generalizes).
4. Extend both to caption (crowd heterogeneity), math (tag sub-populations), law (verdict vs
   revealed).
5. Consolidation / synthesis pass over notebook §26–§35 + this table → paper draft.

Steps 1–2 are the load-bearing new results; everything else is breadth/robustness.

---

## Progress log

**2026-07-22 — academia three-y contrast LAUNCHED (Step b/1–2).** User signed off on all
three defaults (b-first; academia flagship; full item+community heterogeneity).

- *Scoping audit (laptop):* three preference-variable labels join cleanly. Verdict↔curation
  join on `id` (100% of 33,890 curation papers). **Citations use a disjoint DBLP/S2 id-space
  (0 id-overlap)** → joined instead on normalized title: **4,296 ICLR papers carry both a
  curation label and a citation percentile** = the apples-to-apples set.
- *Nested ladder (ICLR-only, venue-controlled):* verdict n=28,402 (accept/reject, pos .394);
  curation n=11,021 (oral+spot vs poster, pos .183); revealed n=4,663 (citation pct, pos .493).
  Cells + stable-hash splits (grouped by normalized title, so shared papers get the SAME split
  in curation and revealed — no apples-to-apples leakage): `datasets/peer-review/vat_3y/`.
- *Instrument held constant across rungs:* A/V scores are label-independent, so each abstract
  is scored ONCE and y attached per rung. Union-to-score = 14,307 abstracts (balanced caps per
  rung; shared-4,296 fully covered) = 2.2M prompts. Scorer `score_va_gemma_3y.py` reuses the
  peer A-bank (154 rubrics, `data/peer_review/rubrics.jsonl`) + V-features + SYS verbatim from
  `score_va_gemma.py`; Gemma-4-31B offline batch.
- *sk3 job:* GPU 1 (fleet on 2–7; GPU0 avoided per co-resident flag), pid 102092, gemma4 conda
  env, spawn + flashinfer-sampler-off recipe. Output → `vat_3y/union_scores.npz`. ETA ~2.5–3h.
- *Next (aggregation, laptop/base env):* attach y per rung → V / A / V+A grouped-CV AUC per
  rung (does the seam widen up the selectivity ladder?); then the **apples-to-apples
  curation-vs-revealed** on the identical 4,296. Robustness: topic-strat revealed-y (sk3 v3).

### 2026-07-23 — RESULTS: academia three-y contrast (Cycle 1 complete)

Scoring done (NA .652, in band). **Sanity gate PASSED: verdict reproduces known ICLR
V .613 (known .611) / A .683 (known .676).** Full: `vat_3y/vat_3y_results.json`.

**Nested selectivity ladder** (ICLR, same 154-rubric A-bank + V-features held constant):

| rung | y | n | pos | V | A | V+A | A−V |
|---|---|---|---|---|---|---|---|
| verdict | accept/reject | 6030 | .498 | .613 | .683 | .690 | **+.071** |
| curation | oral+spot vs poster | 7941 | .254 | .550 | .563 | .567 | **+.012** |
| revealed | citation pct | 2387 | .493 | .705 | .751 | .761 | **+.046** |

**Apples-to-apples** — identical 2,202 papers, identical A/V features, only the preference
variable changes:

| preference y | n | pos | V | A | V+A | A−V |
|---|---|---|---|---|---|---|
| curation | 2202 | .198 | .516 | .538 | .548 | **+.022** |
| revealed | 2202 | .491 | .706 | .760 | .769 | **+.054** |

**Reading (descriptive):**
1. **The finest expert gatekeeping distinction is the least predictable.** Among accepted
   papers, which one gets a spotlight/oral (curation) is barely above chance from the abstract
   (A .538–.563), and the seam nearly closes (A−V +.012–.022). This is the taste-residual end
   of the ladder — expert-B curation is the hardest to articulate.
2. **Changing the preference variable swings the whole VAT profile on the SAME papers.**
   curation-y A .538 vs revealed-y A .760 on identical 2,202 items — the construct changes from
   "quality-among-the-good" (tacit) to "impact/attention" (predictable).
3. **CAVEAT — revealed is an impact cell.** Its high V (.706) is the topic/length floor
   (memory: acad-C citations = impact, topic-fractal). The A .760 rides that floor; A−V +.054
   is the increment over topic, NOT clean articulable quality. **Needs the topic-stratified
   revealed-y robustness (Cycle 2) before quoting revealed as an articulability number.**

Consistent with the master table: verdict = canonical V<A; curation = near-null seam (taste
residual); revealed = impact cell. First within-domain evidence that the seam depends on which
preference variable you measure.

### 2026-07-26 — RESULTS: N&C multi-y contrast (second hole-(b) domain) + drill-downs

`datasets/notice-and-comment/v4/nc_multiy_results.json` (aggregate_nc_multiy.py, verbatim port
of the frozen vat_3y design: GroupKFold(5) group=docket, median-impute + degeneracy guard,
StandardScaler+LR). Pre-GEPA 198-rubric Gemma A-bank (headline per GEPA-negative rule).

**Full-pool per-y** (same comments scored once, y attached per rung):

| y | n | pos | V | A | V+A | A−V |
|---|---|---|---|---|---|---|
| outcome-majority (MADE vs NONE) | 7084 | .399 | .591 | .595 | .594 | +.004 |
| agree-vs-disagree | 5046 | .528 | .600 | .524 | .551 | −.077 |
| responded-or-not | 9521 | .781 | .596 | .624 | .635 | +.028 |

**Sanity gates: 5/6 PASS; agree-A FAILED (.524 vs prior .612, Δ−.088) — RESOLVED as
estimator sensitivity, not a data problem.** Diagnostic (scratchpad diag_agree_gap.py,
n=5046, NA-rate .637, 167/198 cols survive guard): full y_audit_nc.py config reproduces
.612 EXACTLY on today's data. Knob decomposition under the frozen matrix: StratifiedGroupKFold
(shuffle) instead of GroupKFold → .601 (dominant, +.077); 0.5-const impute instead of
median+guard → .586 under GroupKFold (+.062, non-additive); class_weight=balanced → nil.
outcome-y is INSENSITIVE to the same knobs (passes gates both ways) → agree-y's A-signal is
docket/agency-structured: it transfers across stratified folds but not across GroupKFold's
harder docket partition. V is robust both ways (.600–.612). Descriptive record: under the
frozen cross-domain design the agree rung reads V>A (inverted); under the y_audit design it
reads V≈A .61. Both numbers are real; which to headline is a DESIGN choice to settle before
the paper table (frozen design = comparable to peer-review 3y ladder).

**Apples-to-apples** — identical 4,764 comments, identical features, only y changes
(frozen design): outcome V .616 / A .597 (A−V −.019) vs agree V .599 / A .589 (A−V −.010).
On the strict-common set both rungs are ~V≈A≈.60 — the N&C seam stays flat/negative for
expert y's, unlike academia's +.07 verdict seam. responded-or-not pairs structurally
degenerate (constant y=1 on matched side; reported as such). Secondary post-GEPA table
(frozen design): outcome A .614 (> pre-GEPA .595) vs agree A .576. NOTE this REVERSES the
2026-07-16 y_audit-design GEPA readout for outcome (.592→.578, "fidelity up prediction down")
— GEPA's direction on outcome-y is itself estimator-design-sensitive; agree drops under both
designs. Treat "GEPA fidelity↑ A↓" as design-conditional pending the H6 replication, not as
a settled sign.

**Drill-down** (`metric_contrast_nc.py` / `.json`, univariate = CV-free, immune to the
estimator issue), mirrors the peer-review drill-down:
- *Which pieces:* outcome×agree on 4,764 shared comments: φ=+.396, P(agree=1|MADE)=.759 vs
  .361, label-label AUC .697 — the two expert y's agree far MORE than peer-review's
  curation×revealed (φ=.16): agencies mostly adopt what they agree with. Still 31% discordant.
- *Which metrics:* per-metric AUC vectors Spearman ρ=+.764 (vs peer's +.47) — heavily shared
  backbone. outcome-favoring = evidence/actionability cluster (adds-new-material .619 vs .562,
  actionable-alternatives .618 vs .542, specificity/evidence .611, evidentiary rigor,
  v_num_density .606, legal record-building .575 vs .501); agree-favoring = burden/flexibility
  cluster (least-burden .586, performance-based flexibility .571 vs .529, burden-minimization
  .569, implementation feasibility .568) + stance kws (v_kw_support .567 / v_kw_oppose .440
  — partly the label's own vocabulary). Reading: outcome rewards WORK (evidence, drafting);
  agreement rewards ALIGNMENT with deregulatory/feasibility framing. Same weights-not-
  vocabulary heterogeneity shape as peer review, but milder (ρ .76 vs .47).

**Status: hole (b) now has TWO within-domain multi-y contrasts** (academia 3-y + N&C 3-y).
N&C adds the contrast case: expert y's that largely AGREE (φ .40) and share metric backbone
(ρ .76) vs academia's committees-vs-crowd divergence (φ .16, ρ .47). New robustness item:
pre-register the CV-design choice (GroupKFold vs StratifiedGroupKFold) for docket-clustered
domains before the consolidated paper table.

### 2026-07-27 — dense-standard chain: first harvest (INTERIM, test-split)

Chain (`methods/dense/run_dense_standard.sh`, sk3 GPU1) completed 3/6 by evening:

| run | n train | best test AUC | vs V+A (frozen) |
|---|---|---|---|
| peer revealed (citation pct, title-grouped) | 1,909 | **.896** | VA .761 → band +.14 |
| N&C agree (docket-grouped) | 4,037 | **.639** | VA .551 → band +.09 |
| N&C outcome (docket-grouped) | 5,667 | **.623** | VA .594 → band +.03 |

CAVEATS: test = selection split (mildly optimistic); the trainer never scores the eval
split — a post-chain eval-scoring pass is REQUIRED before these are quotable as the
standard's clean numbers. Early reads (descriptive): N&C "flat/null" was TF-IDF-conditional
(real Llama band +.03); Llama recovers the agree signal (.639) that frozen-design A lost
(.524) — consistent with docket-structured-but-learnable; revealed-citation is hugely
dense-predictable (.896), topic-floor robustness even more necessary before interpreting.

## Open decisions for the user
- **Priority / parallelism:** do (a) and (b) in parallel or (b) first (it's cheaper and the
  academia population feeds (a))? *Recommend: (b) Step 1–2 first, then (a) on the same data.*
- **Flagship domain confirmation:** academia as the within-domain preference-variable flagship
  (only domain with all 3 y's on overlapping items). Alternative/addition: law.
- **Scope of heterogeneity:** item-level (rater-bearing: peer/caption/math.SE/reddit) +
  community-level (all), or community-level only for a leaner paper?
