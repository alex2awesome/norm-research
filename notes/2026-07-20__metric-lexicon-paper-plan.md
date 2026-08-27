# Metric-lexicon standalone paper — research plan

*2026-07-20. User directive: turn the metric/codability component into its own complete paper.
Eight workstreams (W1–W8) mapping the user's bullets onto concrete experiments. Companion
assets: census (4 domains), CRP cascade campaign (11 fields, 1,526 new criteria), sampling
suite E1–E3/E5 (`notes/2026-07-20__codability-sampling-angle.md`), hierarchy ledger
(`notes/2026-07-06__hierarchy-reconstruction-ledger.md`).*

## Thesis (working)

Evaluation criteria form a **register** — a socially stratified lexicon. We can measure
(1) how codable each field's criteria are (sampling-theoretic codability, PY coincidence),
(2) how large and how closed each field's conceptual inventory is (species richness + closure),
(3) *who* codes them (provenance strata: official guidelines vs community know-how),
(4) *how high* each coding sits (register height / "lexical bar"),
(5) how LLMs speak this register (mode collapse toward one code; whether coding choice
changes measurement), and (6) how to estimate the dominant code without size bias.
LLMs are the instrument throughout; no human subjects; reconstruction-only discipline holds.

## W1 — Codability per task (consolidate + widen)

**User bullet**: "how codable are metrics in each task (we've explored this)."
**Have**: E1 identity (agreement ≡ missing mass at shallow N), E2 three-way novelty
decomposition, E3 PY coincidence parameter (the size-free codability number: humor .128 <
CW .158 < news .185 < math .195), E5 held-out calibration — but census = 4 domains only,
while the CRP campaign spans 11 fields.
**Do**: (a) audit which of the remaining 7 fields have corpora + extraction assets sufficient
to extend the census (contexts_*.jsonl exist for all 11; full census extraction is the
expensive part); (b) extend E2/E3 to every field that clears the audit; (c) one consolidated
codability table (PY d, θ, coincidence + CI) as the paper's Table 1.
**Deliverable**: 6–11-field codability table; per-field naming-process fits.
**Cost**: GLM extraction batches for new fields (medium); analysis local (cheap).

## W2 — Subfield inventory & richness per task

**User bullet**: "how many unique subfields are discussed in each task."
**Have**: frozen R2 theme / R3 category taxonomies for 11 fields; campaign closure law
(median 94% theme absorption; zero new atomic object-domain qualities escaped R2 under
adversarial out-of-register search).
**Do**: (a) per-field inventory counts at each grain (L0 / R1 / R2 / R3) with observed +
Chao1/GT-style richness estimates and accumulation curves; (b) the closure evidence as the
upper-bound argument: theme inventory is effectively CLOSED, construct/phrasing grains are
open (heavy-tailed); (c) optional secondary reading — content subfields of the *items*
(e.g. math subareas) — park unless a reviewer-facing need appears.
**Deliverable**: richness table + accumulation-curve figure; "closed head, open tail" section.
**Cost**: cheap (all local; data exists).

## W3 — Provenance: who codes the criteria

**User bullet**: "what is the source of the metrics for each task: expert-verdict (official
guidelines), community know-how, etc. How are concepts coded by the community we got them from?"
**Do**: (a) FIRST an asset audit: what source metadata (URL/domain/doc type) is attached to
census records and CRP-wave sources per field; (b) define a provenance typology
(institutional codification ladder, e.g.: statutory/official guideline → professional/editorial
standards → academic literature → practitioner blog/talk → community forum/folk) and code
every source with Sonnet+ judges, **blinded anchors in every batch** (hand-verified known-class
sources camouflaged among items); (c) codability conditional on provenance: PY coincidence and
novelty rates per provenance class per field; (d) **within- vs cross-community coincidence**:
P(same name | same construct) for same-class author pairs vs cross-class pairs — does the
originating community code its own concepts more tightly?
**Deliverable**: provenance-stratified codability; the same-community coincidence contrast.
**Prereg**: the within>cross direction is a directional claim — preregister before the
confirmatory run (E4 lesson).
**Cost**: judge batches (medium).

## W4 — Register height: which code is "higher"

**User bullet**: "ranking on which code is more/less classist... germanic vs latin words
('teacher' vs 'professor')."
**Anchor literature** (W7 confirms): Corson's *lexical bar* (Graeco-Latin vocabulary as a
class barrier), etymological stratification of English (Germanic vs Latinate vs Greek),
psycholinguistic norms (age-of-acquisition, frequency), formality/register measures.
**Do**: build a per-name-variant **register-height index** from two independent instruments:
1. *Linguistic*: LLM judges code each head-term variant for etymological stratum,
   nominalization, formality — with deterministic anchor truth (known Germanic/Latinate pairs:
   help/assist/facilitate, teacher/professor/pedagogue) camouflaged in every batch; judged,
   not proxy-coded, per standing rule.
2. *Sociological (our novel move)*: a variant's **institutionality score** = the provenance
   distribution of its actual uses (from W3) — which class of sources uses this code. This is
   prestige as sociolinguistics defines it: ranked by who speaks it, not by etymology.
Then: do the two instruments agree (etymologically "high" variants used by codified sources)?
Per concept with ≥2 variants: what is the height *spread* — do fields differ in how
stratified their evaluative lexicon is?
**Deliverable**: register-height index + stratification-by-field figure; the
linguistic-vs-institutional agreement result.
**Prereg**: "official sources use higher-register codes" is directional — preregister.
**Framing caution**: report stratification descriptively ("register height"), not as a
sociological verdict; cite the lexical-bar literature for the class reading.
**Cost**: judge batches over the head-term inventory (medium; inventory is thousands of terms).

## W5 — LLMs as speakers of the register

**User bullets**: "do LLMs default to restating concepts with one term out of many? How much
codability variance does the LLM capture? Does this affect how it scores anything? How often
do different codable concepts result in different annotations? How often does reconstruction
use another coded concept? Do GT/scaling limits predict saturation quicker within certain
coded concepts? How do we best measure the dominant coded version [→ W6]."
Five sub-experiments:
- **W5a Naming elicitation**: give LLMs concept *definitions with names redacted* (the blind
  protocol from the E4 resurrect recipe), sample namings across ≥2 families × temperatures;
  compare LLM naming distribution vs human head_terms: modal share, entropy ratio, KL.
  Hypothesis to prereg: LLM p_max ≫ human p_max (mode collapse to one code).
- **W5b Variance capture**: species coverage — what fraction of the human name inventory is
  reachable by LLM sampling at matched N? GT missing mass of the LLM's own naming
  distribution (does the LLM's name process saturate where the human one doesn't?).
- **W5c Scoring sensitivity (the applied stake)**: metric prompts differing ONLY in name
  variant (definition held constant; and a name-only arm), scored over fixed item sets on
  sk2/sk3 batch vLLM; readout = threshold-free agreement/AUC deltas between variants.
  Connects to MI-vs-silver within-metric prompt-variant result (CW ρ=+.393). If variants
  score differently, coding choice is not measurement-neutral — register matters to the
  instrument, not just the speaker.
- **W5d Reconstruction cross-coding audit**: in existing recovery/reconstruction outputs,
  when R(Ω) recovers a construct, which code does it emit? Map recovered names to concept
  name-inventories; rate of same-code vs sibling-code vs novel-code recovery; and is the
  emitted code the high-register variant (ties to W4)?
- **W5e Variant reachability**: per-concept discovery curves — at what sample size does each
  variant first appear; are dominant codes reached at small N while tail codes never
  saturate; does the PY fit predict per-variant first-appearance times?
**Deliverable**: LLM-register section: mode collapse (a), coverage (b), measurement
invariance (c), reconstruction coding behavior (d), reachability (e).
**Cost**: (a,b) cheap-medium elicitation batches; (c) the expensive one — GPU scoring
campaign, design to piggyback on existing metric_implementer harnesses; (d) audit of
existing outputs (cheap); (e) local (cheap).

## W6 — Estimating the dominant code

**User bullet**: "how do we best measure the dominant coded version?"
Raw modal share is size-biased (the confound that killed the first codability headline).
**Do**: estimator bake-off — raw mode share, GT-smoothed head probability, PY posterior
predictive head, hierarchical shrinkage — validated two ways: (a) simulation from fitted PY
truth; (b) prequentially with the E5 harness (predict the *next author's* name; the best
dominant-code estimator is the one whose head prediction wins Brier/log-loss held-out).
**Deliverable**: methods subsection + recommended estimator used everywhere else in the paper.
**Cost**: cheap (local; E5 harness extends).

## W7 — Related work & positioning

**User bullet**: "I know some of this has been done already in Reddit guidelines."
**Do**: two Sonnet lit-recon agents (launched 2026-07-20, slow-burn pacing):
1. Community-rules literature: Reddit rules/norms corpus papers (Fiesler et al. "Reddit
   Rules!", Chandrasekharan et al. hidden rules / macro-meso-micro norms, Weld et al.
   community values), Wikipedia policy/guideline research, Discord/Twitch moderation norms —
   what they measured, whether any did naming/codability analysis, the gap we fill.
   → `notes/lit/2026-07-20__community-rules-litrecon.md`
2. Register-hierarchy measurement: Corson lexical bar, etymological stratification,
   formality/prestige measurement, psycholinguistic norm datasets, Brown–Lenneberg codability
   lineage and successors — candidate operationalizations for W4 and precedents for
   "same concept, high vs low code."
   → `notes/lit/2026-07-20__register-hierarchy-litrecon.md`
**Deliverable**: related-work section draft + any method steals for W3/W4.

## W8 — Assembly & prereg registry

Outline the paper (candidate: I. codability as sampling [W1,W6]; II. inventory & closure
[W2]; III. social stratification of the register [W3,W4]; IV. the LLM as speaker [W5];
V. implications for LLM-judged measurement). Maintain a **prereg registry** section in this
file: every directional contrast (W3 within>cross; W4 official=higher; W5a mode-collapse)
gets direction + single test + null written down BEFORE its confirmatory run. E4
(TASTE-vs-CRAFT) stays excluded unless resurrected per its recipe.

## Ordering & dependencies

| phase | work | why first |
|---|---|---|
| 1 (now) | W7 lit agents; W3a + W1a asset audits; W2 richness (all-local) | cheap, unblock design |
| 2 | W6 estimator study; W4 index build on existing 4-domain head_terms | local + small judge batches |
| 3 | W3 provenance coding; W1b census widening | medium judge/extraction batches |
| 4 | W5a/b elicitation; W5d audit; W5e curves | needs W6 estimator + blind protocol |
| 5 | W5c scoring-sensitivity campaign | the GPU-heavy confirmatory piece, design frozen last |

Standing rules that bind every workstream: LLM judges Sonnet-or-better with blinded anchors
in every batch; threshold-free readouts; stable-hash splits; same-family scaling only;
report-results-not-conclusions in notes; no canonical partitions touched.

---

# EXECUTION PLAN v1 (2026-07-20, post-lit-recon + Phase-1 results)

## State snapshot

| WS | status | artifact |
|---|---|---|
| W1 | E1-E3/E5 done (4 fields); **widening RUNNING** (bg GLM, ~34K items, 7 fields, anchors interspersed) | `extract_<task>_glm-4.7.jsonl` (accruing) |
| W2 | **core DONE**: richness all 11 fields; f1/N = 0 at R2/R3 everywhere | `richness_20260720.json`, ledger 20n |
| W3 | audit DONE: `orientation` tag 100% coverage, 16 values → validate+map, not re-code | ledger 20n |
| W4 | input DONE: 2,370 competing-variant rows w/ PY head share | `name_variants_20260720.jsonl` |
| W5 | not started (5c design deliberately last) | — |
| W6 | **DONE**: PY posterior-predictive head wins 4/4; = paper-wide statistic | `dominant_code_estimators_20260720.json` |
| W7 | **DONE**: both recon files complete | `notes/lit/2026-07-20__*.md` |
| W8 | registry v1 below | this file |

## What the lit recon changed

1. **Gap confirmed and citable**: community-rules literature (Fiesler; Chandrasekharan;
   Weld/Zhang/Althoff; Goyal; Beschastnikh; 14 verified sources) taxonomizes rule/value
   CONTENT and studies named-policy invocation, but never measures naming diversity,
   saturation, or register of competing codes. Five targeted searches for rule-WORDING
   similarity work: empty. Document the negative searches in the paper's related-work.
2. **External convergence point for W2**: Weld et al.'s hand-built taxonomy of community
   values has 29 subcategories; our judged theme grain lands at K=38-57 per field —
   independent instruments agreeing on the order of magnitude of the evaluative "head."
3. **Beschastnikh et al. 2008** (how often Wikipedia's NAMED policies get cited) = the
   "codability in the wild" precedent → framing citation for why dominant codes matter.
4. **W4 instruments finalized** from the register shortlist: LLM judges score register
   height; Etymological Wordnet (de Melo 2014) + SimplePPDB/GYAFC pairs serve as ANCHOR
   TRUTH/validation (per LLM-judges-do-measurement rule, resources are calibration not
   measurement); Kuperman AoA + SUBTLEX frequency + concreteness as covariates only.

## Per-workstream execution specs

**W1b (running)**: per-task gate = anchor pass ≥7/8 in `_anchor_report`; on completion
verify extract↔`partition_<task>.json` join coverage ≥95% BEFORE fitting; then extend
`codability_sampling_model.TASKS` to 11 and rerun E2/E3/E5 → consolidated Table 1
(PY d, θ, coincidence+CI per field). If a field's join fails, diagnose before fitting —
never fit on a partial join.

**W2b (new, from user question)**: `subtask_short` is 91-96% singletons → cluster before
counting subfields. Method: BGE-embedding candidate pairs (cos ≥ .80 within task) + Sonnet
same-subfield judge (blinded anchors: known-same pairs from repeated labels e.g. "linux
kernel coding style"/"linux kernel c coding style"; known-different from cross-task donors),
merge-and-stop union-find; then re-run richness on clustered subfields. Deliverable:
subfields-per-field at clustered grain + the recursive result (the annotation layer is
itself an unsaturated naming process).

**W3**: (a) mapping: 16 orientations → 5-rung codification ladder — formal_guideline,
contest_criteria → RUNG 1 (codified/official); professional_standard, stylebook,
course_syllabus → RUNG 2 (professional); research_article, academic_page, textbook_excerpt,
dataset → RUNG 3 (academic); how_to, tutorial, news_article, wiki → RUNG 4 (practitioner/
secondary); blog_post, forum_post → RUNG 5 (folk); other → excluded. (b) validation audit:
stratified sample ~32 sources/field (≈350 total, all 16 orientations covered), Sonnet judges
re-code rung from source text with blinded known-rung anchors; gate = judge-vs-mapped
agreement ≥.8 weighted; disagreement pattern → refine mapping once, re-audit only the moved
orientations. (c) after gate: PY fits per rung×field (min 150 named records per cell, else
pool rungs 1-2 / 4-5); (d) PREREG-1 then the within/cross coincidence test.

**W4**: `register_height.py`. Instrument 1: judge batches over the 2,370-variant inventory —
judge sees the VARIANT STRING ONLY (no task, no concept, no counts), scores etymological
stratum / nominalization / formality 1-7; every batch salted with Etymological-Wordnet-truth
anchors (help/assist/facilitate tier) + SimplePPDB high-low pairs; gate = anchor stratum
accuracy ≥.85. Composite height = mean of z-scored judged formality + Latinate indicator.
Instrument 2 (institutionality): per variant, distribution of its uses across W3 rungs
(from extract records' source_id → orientation → rung); score = mean inverse-rung.
Then: instrument agreement (Spearman, per field + pooled); PREREG-2 then confirmatory.

**W5**: (a/b) elicitation harness `naming_elicitation.py`: per concept with ≥3 human
namings, present DEFINITION with names redacted (E4-resurrect blind protocol) to ≥2
families (Sonnet + GLM-4.7), temp 1.0, k samples matched to human N; readouts: paired
p_max (PY-estimated both sides — same estimator both sides, W6 winner), entropy ratio,
coverage of human inventory, GT missing mass of LLM's own process. PREREG-3 first.
(d) reconstruction cross-coding audit: harvest existing recovery outputs (recon runs for
census tasks), map recovered names into concept inventories (exact + judge-assisted match),
rates of same-code / sibling-code / novel-code; join to W4 heights. (e) variant
reachability: per-variant first-appearance N vs PY prediction, local. (c) LAST: freeze
name-variant-only scoring-sensitivity design after W4 heights exist (variants chosen to
span height, not convenience); offline batch vLLM sk2/sk3; threshold-free AUC deltas.

## Prereg registry v1 (frozen 2026-07-20, BEFORE any confirmatory run)

1. **PREREG-1 (W3 within>cross)** — H: P(same name | same construct) is higher for
   author pairs from the same codification rung than cross-rung. Test: per field, rung-label
   permutation null (1,000 perms) on the same-rung minus cross-rung coincidence gap;
   combine fields by Fisher. One run, mirror-guarded, after the W3 validation gate.
   **AMENDMENT (2026-07-20 late, PRE-DATA — no confirmatory run has occurred)**: the
   orientation→rung mapping FAILED validation (judged-vs-mapped .60 exact/.69 3-class;
   scrape tags encode genre not authority) and GLM full-pool coding agrees with the
   anchor-validated Sonnet reference only at coarse boundaries (3-class .777; binary
   {1,2}v{3,4,5} .869, {1,2,3}v{4,5} .852, {1}vRest .905; GLM self-consistency 94% does NOT
   predict Sonnet agreement — systematic boundary difference, not noise). Therefore
   PREREG-1 and PREREG-2 run on the BINARY codified-vs-uncodified split {1,2} vs {3,4,5}
   (primary), with {1,2,3} vs {4,5} as the sensitivity split. 5-rung codes are stored but
   not used for confirmatory tests. Rung source = GLM-4.7 pool coder (same prompt as the
   Sonnet validators; hybrid-thinking max_tokens=300 fix applied).
2. **PREREG-2 (W4 official=higher)** — H: variant institutionality score correlates
   positively with judged register height. Test: per-field Spearman over variants with ≥3
   uses, null = rung-label permutation within concept; combine via Fisher. One run, after
   both instruments pass their gates. (The killed raw-agreement taste/craft headline does
   NOT license this — different axis, preregistered fresh.)
**EXECUTED 2026-07-21 (single run, `prereg_tests.py` → `prereg_results_20260721.json`):
PREREG-1 NOT SUPPORTED (primary Fisher p=.253; sign-inconsistent per-field). PREREG-2 NOT
SUPPORTED (primary Fisher p=.556; news rho NEGATIVE; sensitivity p=.049 heterogeneous =
not quotable). Ledger 2026-07-21b has caveats. Closed; any follow-up = new prereg.**

3. **PREREG-3 (W5a mode collapse)** — **EXECUTED 2026-07-22, GLM-4.7 lane, single frozen run: SUPPORTED 3/4 tasks (CW p=1e-6, humor .012, math .036; news null .27 = human-ceiling task); results prereg3_results_glm_20260722.json; ledger 2026-07-22f. Second family EXECUTED 2026-07-22: GPT-5.6-sol via Codex REPLICATES (humor 1e-4, CW 6e-6, math .090 marginal, news null again — same ceiling pattern); prereg3_results_gpt56_20260722.json; ledger 2026-07-22i.** Original spec:  H: LLM naming distributions concentrate more than
   human at matched N: paired per-concept PY-estimated head share, LLM > human. Test:
   Wilcoxon signed-rank per task (4 census tasks; extend to 11 if W1b certifies); one run
   per family; families reported separately (same-family rule), never pooled.

E4 (TASTE/CRAFT) remains excluded; resurrect recipe unchanged.

4. **PREREG-4 (FROZEN 2026-07-21, user-approved; runs on the 7 WIDENED fields ONLY, no
   contact with that data yet)** — H: within-class naming coincidence exceeds cross-class at
   **R1 construct grain**, binary primary split {1,2}v{3,4,5}. Test: v2-corrected
   `prereg_tests.py --grain R1` machinery (same-doc pairs excluded, valid-perm denominators,
   eligible-universe permutation), 1,000 doc-level perms per field, Fisher over usable
   widened fields (≥50/≥50 pair gate). Disclosed motivation: census-field R1 exploratory
   trend p≈.07 (Codex read-only audit) — motivation, not contamination; census fields are
   NOT in this test. Rung instrument for widened fields must pass its own gate before the
   run. ONE run.
5. **PREREG-5 (FROZEN 2026-07-21, user-approved) — adoption asymmetry**: for concepts with
   ≥2 uses in EACH class, P(informal record uses the institutional-class dominant code) >
   P(institutional record uses the informal-class dominant code). Dominant code per class =
   PY posterior-predictive head within class (W6 winner). Statistic: pooled asymmetry
   difference; null: same doc-level class permutation; Fisher over usable fields (≥15
   qualifying concepts). Runs on widened fields alongside PREREG-4; census fields may be
   reported as a labeled exploratory replication arm only. ONE run.

**INSTRUMENT PIVOT (2026-07-20 late)**: session subagent limit (200) reached after W2b
wave-4 launch — no further Sonnet judge agents this session. Remaining W4 tail (~695
single-use variants) and W2b waves run on GLM-4.7 via
`methods/codability/lexicon/glm_judge_fallback.py` (verbatim Sonnet prompts) ONLY IF the
cross-instrument gates pass: w4-validate (150 Sonnet-judged variants re-judged; stratum
agreement ≥.85, formality ρ≥.75) and w2b-validate (150 pairs incl. 50 Sonnet-DIFFERENT;
agreement ≥.90, different-recall ≥.50). Instrument recorded per row; Sonnet-vs-GLM
heterogeneity noted in any analysis pooling both.

## Exploratory / low-weight (queued for restart session, local compute only)

User agreed to look but explicitly NOT to weight these heavily — report as secondary
robustness, never as headline.

- **EX-1 Frontier decomposition** (propagates field-level GT missing mass to sources):
  (a) per-stratum missing mass f1/N + Chao1 within each rung/orientation/field, bootstrap
  over docs for CIs; (b) per-document prequential novelty yield (fraction first-occurrence
  concepts at hash-order ingest). Field missing mass = mean of per-doc yields; expose the
  distribution. **Compute at R1 grain, not L0** (L0 singletons conflate frontier w/
  extraction/partition noise). Directional prediction to test: frontier-ness runs INVERSE
  to codification rung (folk = frontier, official = saturated) — would tie to the instrument
  gradient. Expected heavy-tailed, NOT uniform.
- **EX-2 Grain robustness + multilevel codability**: report the PY coincidence LADDER at
  L0 / R1 / R2 as a robustness curve (parallel to the missing-mass waterfall). R2 as the
  ESTIMAND grain is too coarse (measures dimension-label conventionality + within-theme
  heterogeneity, not codability). Correct use of the "R2→L0" idea = R2 as a POOLING PRIOR:
  hierarchical/multilevel PY estimating per-R1 coincidence shrunk toward parent-theme mean
  (right estimand, borrowed strength). Note PY already uses singletons (informs d), so L0 is
  less singleton-starved than a raw-agreement calc would be.

## Schedule (slow-burn: ≤2-3 Sonnet agents concurrent; hourly cron continues)

1. NOW→overnight: W1b bg extraction grinds; W2b candidate pairs (local embed) prepared.
2. Next active block: W3b validation batches (2 Sonnet agents) + W2b judge batch (1 agent)
   — staggered to stay ≤3.
3. Then: W4 instrument-1 judge batches (2 agents/round over ~2.4K variants ≈ 3-4 rounds);
   institutionality computes locally once W3 gate passes.
4. Then: W5a/b elicitation (Sonnet lane + GLM lane), W5d audit, W5e local.
5. LAST: W5c frozen design → GPU campaign (sk2/sk3, offline batch vLLM).
6. W8 assembly ongoing in this file; ledger entry per milestone.

## 2026-07-22 — W9–W12 expansion (user request; PROPOSED, freeze pending sign-off)

**REGISTER BANK (BUILT)**: `outputs/lexicon/register_bank_20260722.jsonl` (+.meta.json) —
one row per (task, con, variant): usage stats, W4 stratum/formality/nominalization (2,195 =
FULL inventory), composite height_z, latinate v2.2, W3 inst_share (407), GLM axes (1,654).
The reusable covariate table for all cross-stream joins below.

**W9 — subfield-conditioned class coincidence (PREREG-6, PROPOSED)**. Concern (user):
PREREG-1/4 pair authors across a whole field; "horror story" vs "adventure story" authors
aren't naming the same practice. Now that W2b clustering is COMPLETE, condition on subfield:
restrict pairs to SAME W2b subfield-cluster (doc→subtask_short→cluster), re-run within-class
vs cross-class gap at R1 grain; permutation shuffles class labels WITHIN subfield strata.
H: PREREG-4's gap survives subfield conditioning (it is class convention, not subfield
composition). Also report the decomposition (how much of the unconditioned gap subfield
composition explains). Fields: 7 widened primary; census secondary. ONE run after freeze.

**W10 — institutional-authorship audit (validity gate, runs BEFORE W9/PREREG-6)**. Sample
~30 institutional-classed + ~15 folk-classed docs per field; Sonnet agents read doc text +
source metadata and code: named human author? institutional affiliation/title (professor,
editor, examiner, official)? org-authored? Blinded known-class anchors per batch. Gate: if
<80% of institutional-classed docs show genuine institutional authorship, the rung
instrument gets ONE correction pass + re-audit (per W3 discipline). Outcome recorded either
way; PREREG-2 "deeper look" inherits this audit before any re-analysis.

**W11 — register × LLM behavior suite (joins via the register bank; exploratory first,
any directional confirmatory = NEW prereg)**:
  (a) rename-fidelity: join height_z to W5e reachability + name→explain→rename mode-
      stabilization (does high register stabilize the round trip?);
  (b) labeling variance / scoring: W5c design (frozen LAST) now explicitly contrasts
      register height within concept — does the high-register variant change judge score
      variance / item ranking?;
  (c) reconstruction: W5d cross-coding audit — are high-register names likelier to be the
      code EMITTED by recovery, and are concepts with high-register heads more recoverable
      (join to recovery C(R(Omega)) stats)?;
  (d) preference: join per-metric predictive stats (cells DB / VAT ladders) to head-variant
      height_z — do high-register-named metrics predict preference better?;
  (e) tacitness: join to mention-AUC / MI-vs-silver per-metric stats — are high-register
      names more articulable/installable (ties to tacit-installation stream)?
**W12 — exploratory prestige correlations** (low-weight): height_z x usage frequency
(Zipf), x concept dispersion, x field position in Table 1, x crosser status
(policy-isomorphism panel), x codification rung at doc level.

Order: W10 audit -> freeze PREREG-6 -> W9 run; W11a/c/d/e joins exploratory (local);
W11b inside the W5c freeze; W12 whenever idle. Nothing confirmatory before sign-off.

## PREREG-6 and PREREG-7 (FROZEN 2026-07-22, user-approved, PRE-DATA)

6. **PREREG-6 — subfield-conditioned within-class coincidence — EXECUTED 2026-07-22, NOT SUPPORTED (Fisher p=.399, signs mixed; ledger 2026-07-22h: composition-vs-power ambiguous; PREREG-4 strong reading retired, unconditioned reading stands). Frozen spec:**
   H: within-class naming coincidence exceeds cross-class among SAME-SUBFIELD author pairs,
   R1 construct grain, binary split {1,2}v{3,4,5}, 7 widened fields. Doc→subfield =
   normalized `subtask_short` → W2b judged union-find cluster (subfield_merges_20260720,
   final 2026-07-22 state); docs with no subfield label or singleton cluster EXCLUDED from
   the pair universe. Statistic: same-class minus cross-class same-name pair rate, pairs
   RESTRICTED to same-subfield-cluster doc pairs (same-doc pairs excluded, mirror-guarded,
   v2 machinery). Null: 1,000 permutations of class labels shuffled WITHIN subfield-cluster
   strata (clusters with <2 docs dropped). Fisher over usable fields (>=50 same-class and
   >=50 cross-class eligible pairs). Secondary readout (descriptive): share of the
   unconditioned PREREG-4 gap explained by subfield composition. ONE run.

7. **PREREG-7 — institutional literalness — EXECUTED 2026-07-22, NOT SUPPORTED (primary Fisher p=.106; 4/5 fields directionally negative; secondary transparency p=.076; ledger 2026-07-22j — census literalness stays descriptive-only). Frozen spec:** Motivating exploratory
   (census fields, DISCLOSED): pooled inst_share x metaphoricity rho=-.177 p=.002,
   sign-consistent 4/4 census tasks; transparency +.128. These census fields are NOT in the
   test. H (primary): on the 7 WIDENED fields, variant-level institutionality (share of a
   variant's uses coming from binary-institutional docs, >=3 unique docs per variant) is
   NEGATIVELY rank-correlated with judged metaphoricity. Test: per-field Spearman, Fisher
   over fields with >=20 eligible variants. Secondary (same direction family, reported
   separately): transparency POSITIVE. Instruments: widened variant inventory built from
   extracts (normalized head_term per R1 concept); metaphoricity/transparency = GLM axis
   judges w/ the 2026-07-21 anchor-gated protocol (18-anchor pilot + 6-anchor camouflage
   per batch, abort <5/6). Inventory building and axis judging happen BEFORE any
   correlation is computed. ONE run.

**PREREG-3 second family (user decision 2026-07-22)**: GPT-5 family via Codex subscription
(replaces the Claude/USC-key option); exact model = gpt-5.6-sol (bare 'gpt-5' rejected by
the ChatGPT-account Codex API — error recorded 2026-07-22). INSTRUMENT DEVIATION disclosed at freeze: Codex
threads process ~100 definitions per fresh thread (15 threads), so samples are independent
ACROSS threads but share context WITHIN a thread — unlike the GLM lane's fully independent
calls. Reported separately, never pooled with GLM, deviation stated wherever quoted.

**EX-3 (exploratory, user musing)**: metaphorization vs neologism as innovation routes —
are singleton/frontier names more metaphorical than established heads? Descriptive only.

**W11b priority raised (user)**: tie register to LLM-judge outcomes — name-variant-induced
scoring variance is now the flagship of the W5c design (still frozen LAST, after PREREG-6/7).

8. **PREREG-8 — EXECUTED 2026-07-22 (ledger 2026-07-22l): gates PASS .941/.968; primary insider test Fisher p=.023 but LOO-FRAGILE (p=.129 w/o grant-funding, signs mixed) — quote as weak; census secondary p=.071 n.s.; SECONDARY LITERALNESS CONFIRMED p=1.3e-4, survives LOO (.022) — P7's hypothesis holds at author grain. (FROZEN 2026-07-22, user-directed fix from the
   2026-07-22k construct-slippage audit; frozen BEFORE any author coding exists)**.
   Instrument: AUTHOR institutionality coded per doc by Codex gpt-5.6-sol (doc text head+tail
   + source_id; taxonomy identical to the W10 audit: institution / credentialed_individual /
   lay_individual / unknown; validated .90 vs 2-tier ground truth). Class: INST =
   {institution, credentialed_individual}; LAY = {lay_individual}; unknown EXCLUDED.
   Instrument gates BEFORE the confirmatory run: (a) agreement with the 330-doc resolved
   W10 truth >= .85 on INST-vs-LAY binary; (b) 100-doc Sonnet cross-instrument recode
   agreement >= .85; failing either -> one correction pass + re-audit, never silent use.
   H (primary): within-class naming coincidence > cross-class at R1 construct grain, class =
   author type, 7 widened fields (same v2 machinery: same-doc pairs excluded, mirror-
   guarded, doc-level label permutation x1000, Fisher over fields passing >=50/>=50 pair
   gates). Census fields = secondary, reported separately. Secondary family (P7 retest,
   reported separately): variant-level author-INST share (>=3 docs) x metaphoricity
   NEGATIVE, same 5-field universe as PREREG-7. ONE run each after gates.

## W13 — lay-author corpus expansion (user-directed 2026-07-22 evening)

Motivation: P8 found lay authors scarce (5-10 docs/field in professionalized fields) —
scarcity could be a SCRAPE/SEARCH CONFOUND rather than a fact about the discourse. Campaign:
(A) targeted web search for lay-authored docs w/ explicit criteria (Sonnet agents w/ web,
1/task, shared instructions laysearch_instructions.txt; scarcity honestly reported per
field); (B) gpt-5.6-sol parses each doc — criteria (verbatim head names, descriptions,
quotes), subfield, AND author+affiliation in one schema; (C) author-coding instrument
(gates already passed) classifies; new criteria matched into existing R1 constructs
(BGE shortlist + judge, blinded anchors); dedupe vs existing source_ids. Target 100-500
new lay metrics/task where the discourse supports it. **Any register/class re-test on the
augmented corpus = PREREG-9, frozen AFTER collection, BEFORE analysis.** If search comes
back scarce despite 8+ angles/field, that upgrades "lay scarcity" from corpus artifact to
finding.

**W13 adaptive-sampling rule (user 2026-07-22 late)**: search effort allocated by
GOOD-TURING MISSING MASS of the growing lay corpus — after each parse round compute per
field: N criterion draws, distinct heads (norm_name grain), f1, GT missing mass f1/N,
Chao1. Fields with HIGH missing mass (>~.4) and non-degenerate yields get further waves;
fields where missing mass plateaus low OR search saturates (repeat URLs, scarcity reports
across angles) STOP. This mirrors the E-suite discipline: the accumulation curve, not the
doc count, is the stopping rule. Dashboard: lay_gt_dashboard (script) →
outputs/lexicon/lay_corpus_gt_20260722.json per round.

10. **PREREG-10 — EXECUTED 2026-07-23 (ledger 23c): primary p=.037 LOO-FRAGILE (news reversed; quote as heterogeneous); secondary community<official p=2e-11 decisive; W5f: BOTH LLM families ON the official class (GLM d=.057; GPT even higher than officials). (FROZEN 2026-07-23, user-approved; frozen BEFORE
    any JUDGED register data exists for these terms)**. Disclosed motivation (lexical
    descriptive, ledger 2026-07-23a): community-rule terms lexical-latinate .584 vs bank
    informal .784 — motivation, not the test instrument.
    H (primary): community-rule criterion terms have LOWER judged register height than
    INDIVIDUAL-LAY head terms from the W13 lay corpus (same lay authority, different
    speaker position) — one-sided Mann-Whitney on composite height (mean of z-scored
    judged formality + latinate-stratum indicator), domain-stratified pooling (domains
    with >=30 terms per side), direction: community-rule < individual-lay.
    Secondary (reported separately): community-rule < official-class names (bank variants
    with author-level INST share >=.8, >=3 docs).
    Instrument: W4 judge protocol (stratum/formality/nominalization) via SONNET agents
    ONLY (GLM failed the etymology gate 2026-07-20 and is barred), 10 etymology anchors
    per chunk, gate >=8/10; stratified samples ~1,000 terms/class; judging happens AFTER
    this freeze, analysis in ONE run. Cross-extractor caveat disclosed (Codex terms vs
    GLM bank names; prior check Δ+.013 n.s.).

**PREREG-10 descriptive companions (2026-07-23, disclosed non-confirmatory; user question
re audience-vs-platform confound)**: (a) platform split — individual-lay side split into
reddit vs non-reddit sources (URL-based); platform-style story predicts no within-reddit
rules-vs-posts gap + big cross-platform gap; speaker-position story predicts the opposite.
(b) audience-breadth gradient — within community rules, judged register vs log subscriber
count (audience-design mechanism predicts negative slope; enforcement-genre story predicts
flat). (c) expert-niche vs general-audience subreddit rules. Framing: Bell audience design
= the MECHANISM reading of the register-tracks-social-configuration claim; the only
deflationary alternative is flat-within-platform "house style", which (a) tests directly.

## W8 REVISED MACRO-STRUCTURE (user-set, 2026-07-23)

**Part 1 (~60%) — The census: what criteria exist, and how they differ by group/register/
audience.** Table 1 codability gradient (11 fields); subfield inventory (W2b clustered);
richness + GT missing-mass laws; provenance ladder + 99.1% authorship verification; the
register bank + 5 triangulated axes; class effects (P4 field-level + P8 author-level
literalness CONFIRMED; honest nulls P1/P2/P5/P6/P7 with construct-slippage diagnosis);
THREE-CLASS speaker-position structure (official / community-rule / individual-lay) +
audience-design companions; lay corpus (W13) + adaptive GT-guided collection.
**Part 2 (~20%) — How LLMs (re)articulate: which groups do LLMs resemble?** PREREG-3
mode collapse (2 families, ceiling pattern); W5e reachability; W5d reconstruction
cross-coding audit (PENDING); NEW W5f "register resemblance": score LLM-elicited names on
the register instruments and measure distributional distance to each of the three classes
— the "LLMs speak like [which group]?" headline. (Prediction on record from mode-collapse
+ latinate-default descriptives: closest to OFFICIAL class; test before claiming.)
**Part 3 (~20%) — Behavioral impact: does register change labeling?** W5c name-variant
scoring sensitivity (design freezes after P10 register data; GPU lane): same items, same
definitions, names swapped across register classes; agreement/AUC deltas + variance;
positioned against Hwang/Huynh paraphrase-instability (we isolate the NAME variable).
**Stage-setting close**: articulation upper bounds (M_omega/DPI), VAT ladders, tacit
knowledge — the follow-on papers this census grounds.

11. **PREREG-11 — name-register scoring variance, W5c realized (FROZEN 2026-07-23,
    user-directed; frozen BEFORE any scoring runs)**. H (primary): substituting a
    criterion's NAME between register classes (high/official vs low/lay-germanic), with
    the prompt otherwise IDENTICAL and no definition given, changes item-level judgments
    BEYOND natural sampling variance. Design:
    - Materials: ~20 same-construct cross-register name PAIRS drawn from (a) register-bank
      concepts holding both a germanic/low and latinate/high judged variant (e.g. "bait
      and switch" vs "incongruity"), and (b) lay-matched pairs (lay head vs professional
      head of the same matched construct). Pair eligibility: same construct evidence +
      judged height gap >= 1.0 SD; fields limited by item availability.
    - Items: 30 fixed items per construct from the task's canonical item pool; identical
      across both names.
    - Subject 1: gpt-5.6-sol via Codex, temp/default sampling. Each (construct, name) is
      scored in k=5 INDEPENDENT fresh threads, each thread scoring all 30 items 1-7 with
      the identical template: only the name token differs between arms.
    - Noise control: within-name variance = across the 5 replicate threads. PRIMARY
      STATISTIC (threshold-free, per house rule): per construct, mean split-half Spearman
      of item ranking WITHIN name (run-halves) minus Spearman BETWEEN names (run-mean item
      rankings, name A vs B). H: within > between (names induce ranking disagreement
      exceeding run noise). One-sided Wilcoxon signed-rank over constructs. Secondary:
      per-item name-gap vs permutation null (shuffle name labels over the 10 runs), Fisher
      over constructs. Tertiary descriptive: does the HIGH-register name yield higher/
      lower/harsher scores (mean shift), and does |shift| correlate with height gap.
    - ONE analysis run after collection. Additional model families (Claude, others) =
      same frozen design, reported separately, never pooled (W5f expansion note: the
      5-class register table also to be extended to more model families as channels allow).
    Novelty: no prior work isolates criterion-NAME substitution from prompt paraphrase
    (Hwang/Huynh = sentence-level rewording; Santurkar = persona) — pluralism lit-recon
    2026-07-22, verified citations.

**Multi-model expansion (user-directed 2026-07-23)**: NAMING (PREREG-3 + W5f register row)
adds CLAUDE family via 15 Sonnet-subagent chunks over the same redacted-definition payload
as the GPT lane (same disclosed within-chunk-dependence deviation; families reported
separately). SCORING (PREREG-11) adds: GLM-4.7 lane via zai API (per-item independent
calls at temp 1.0 — strictly BETTER independence than threads, disclosed) and CLAUDE lane
via 140 Sonnet subagents (fresh context per run = independent replicates, default temp).
Same frozen design/statistics for every family; never pooled. Open-weight families
(Llama/Gemma/Qwen on sk3 vLLM) = optional follow-on, needs GPU session.

12. **PREREG-12 — EXECUTED 2026-07-23 (ledger 23m): SUPPORTED — D_name +.505 vs D_scaffold -.004, p=1.2e-4 (13/14); levels 1.59 vs 0.21 pts p=6e-5; paraphrase inert, name is the causal token. (FROZEN 2026-07-23,
    user-directed "compare the delta for the word changes with deltas for other natural
    prompt changes"; frozen BEFORE any paraphrase-scaffold run)**. H (primary): the rank
    disruption caused by swapping the criterion NAME exceeds the rank disruption caused
    by a full PARAPHRASE of the surrounding prompt scaffold with the name held fixed.
    Null: names are just another surface perturbation (D_name ≈ D_scaffold). Design:
    - Materials: identical 14 construct pairs + 30-item sets as PREREG-11. Scaffold B =
      sentence-by-sentence semantic paraphrase of the PREREG-11 scoring instructions
      (every sentence reworded; same 1-7 scale, same output format, name token untouched)
      — file `p12_score_instructions.txt`, frozen at registration.
    - Subject 1: gpt-5.6-sol via Codex, k=5 independent fresh threads per (construct,
      arm) under scaffold B, BOTH arms (140 runs), mirroring the PREREG-11 lane exactly.
    - PRIMARY per construct: D_name = within − between_names (PREREG-11 gpt56 lane,
      scaffold A) vs D_scaffold = within − between_scaffolds, where between_scaffolds =
      Spearman(run-mean ranking scaffold A, run-mean ranking scaffold B) with the SAME
      name, averaged over the two arms; `within` = the same PREREG-11 within-name
      split-half baseline. One-sided paired Wilcoxon over 14 constructs, D_name >
      D_scaffold.
    - SECONDARY (level shifts, reported separately): per construct |mean shift| under
      name swap (scaffold A, high vs low) vs |mean shift| under scaffold swap (same
      name, A vs B, averaged over arms); one-sided paired Wilcoxon.
    - Descriptive: within-scaffold-B split-half reliability (scaffold B must itself be a
      stable instrument; if its within-reliability is materially below scaffold A's, the
      comparison is flagged, not quietly interpreted).
    - ONE analysis run after collection. Other families = same frozen design, reported
      separately, never pooled.

**PREREG-10R — expanded-corpus recalculation of PREREG-10 (registered 2026-07-23 BEFORE
the expansion judging wave; user-directed "recalculate all of the upper/lower register
preregs now that we have a much more expanded corpus")**. Identical H, statistics
(one-sided MWU community<individual-lay, domain-stratified ≥30/side, Fisher + LOO;
secondary community<official), instrument (SONNET agents only, 10 etymology anchors per
chunk, gate ≥8/10), and one-run discipline as frozen PREREG-10. ONLY change = sample:
full coverage of the corpus as of 2026-07-23 — all 4,663 distinct lay heads (incl. W13
round 2) and all 3,313 distinct community-rule criterion terms (official/LLM class rows
unchanged); the original 1,997 judgments are retained (append + dedup on normalized term
within class), ~5,979 terms newly judged. The original PREREG-10 run STANDS as the frozen
confirmatory result; 10R is the disclosed expanded-sample rerun — both are reported.
Companions (a)-(c) recomputed on the expanded set (descriptive, non-confirmatory).

13. **PREREG-13 — literalness by speaker position: community rules vs lay vs official
    (FROZEN 2026-07-23, user-approved "freeze and launch"; frozen BEFORE any metaphoricity
    judging of community/lay terms exists)**. Motivation (disclosed): PREREG-8 secondary
    established officials-literal/lay-metaphorical INSIDE the professional corpus;
    community rules are the LOWEST-register class, so if literalness merely tracked
    register they should be the MOST metaphorical — but rules are an enforcement genre
    (must be adjudicable), predicting official-level literalness at lay-level register.
    H (primary): community-rule criterion terms are LESS metaphorical than individual-lay
    head terms — one-sided, per-domain MWU (>=30/side), Fisher over domains + LOO; same
    domain machinery as PREREG-10/10R.
    Secondary (descriptive, reported separately): the DISSOCIATION readout — community-rule
    metaphoricity rate vs official-class rate (bank variants inst_share >=.8, >=3 docs,
    existing GLM axis scores) vs lay rate; question: does community-rule literalness sit at
    the official level despite lay register (register and literalness = two independent
    axes of speaker position: institutional voice vs adjudication function).
    Instrument: IDENTICAL to the PREREG-8/bank metaphoricity axis — GLM-4.7 (zai), binary
    metaphorical/literal, the same frozen system prompt + 18-term anchor bank
    (axes_judge_glm.py), 6 camouflaged anchors per 180-term batch, gate >=5/6 (chunk not
    ingested on fail), temp 0. GLM is the axis judge of record (barred for ETYMOLOGY only);
    same-instrument comparability with PREREG-8 outweighs judge diversity here.
    Pool: all 3,313 distinct community-rule terms + all 4,663 lay heads (officials reuse
    existing bank scores — no re-judging, no double instrument). ONE analysis run.

## W14 — instrument calibration (user-directed 2026-07-23)

Goal: report RELIABILITY for every register/metaphoricity instrument in the paper.
Three legs per instrument (etymological stratum, formality, nominalization,
metaphoricity, transparency, thick/thin):
(a) EXTERNAL GOLD — labeled datasets / mechanically-verified methods (candidates:
Etymological WordNet / Wiktionary etymologies [stratum]; Pavlick-Nenkova style lexicon,
Brooke et al. formality lexicon [formality]; NOMLEX + morphological rules
[nominalization]; MOH/TroFi/VUAMC [metaphoricity]; Reddy 2011, LADEC compositionality
[transparency]; thick/thin = philosophy construct, likely NO external gold — report
cross-judge reliability only, disclosed);
(b) EXTERNAL VALIDATION of the deployed judges (Sonnet for register axes, GLM for
binary axes) against that gold;
(c) CROSS-FAMILY AGREEMENT — Codex gpt-5.6-sol (+ the non-deployed judge) re-judges
stratified samples of OUR OWN items; accuracy/kappa vs deployed labels.
Descriptive calibration, not hypothesis tests — no prereg needed; all results to a
reliability table in the paper + ledger. Gold data → datasets/instrument_validation/.

14. **PREREG-14 — name register x silver-label accuracy (FROZEN 2026-07-23, user-directed
    "look at the silver label accuracy"; frozen BEFORE any scoring on silver-labeled
    docs)**. Feasibility scan (silver_join/FEASIBILITY.md): retroactive join to PREREG-11
    runs impossible (opaque a<N> metric keys; 0/30 item overlap both domains); 4 humor
    constructs bridge via exact/substring phrase match of the LOW name to mention-AUC
    metric rubric text (the HIGH/technical names never appear in that vocabulary — itself
    a register datum, disclosed): p1 trim-the-fat/pleonasm->a71; p2 wordplay/
    paronomastic->a119,a144,a145,a177,a224,a46,a78; p4 button/logical-mechanism->a196;
    p5 callback/reincorporation->a138,a197,a205,a266.
    Design: per pair, 25 silver-pos + 25 silver-neg threads (union over matched ids,
    neg=r2+r3 state-negatives minus pos; seed 17; text truncated 4,000 chars), scored
    under BOTH names with the exact PREREG-11 template (name only, no definition),
    gpt-5.6-sol via Codex, k=3 fresh runs per (pair,arm) = 24 jobs.
    PRIMARY (TWO-SIDED, no directional prior): per pair, AUC of run-mean score against
    silver pos/neg under low vs high name; readout = per-pair dAUC with 1,000-resample
    bootstrap 95% CI + pooled stratified-bootstrap dAUC (n=4 pairs — too few for rank
    tests; CIs are the inference). SECONDARY (descriptive): absolute AUC per arm vs
    chance (does either name track the silver construct at all).
    Frozen caveats: silver labels are GLM mention/state-derived (noisy instrument —
    differences read as accuracy-against-silver, not truth); low-name-based metric
    matching could favor the low arm (construal alignment) — reported as a limitation,
    not corrected. ONE analysis run.

15. **PREREG-15 — SCALED register-accuracy on silver labels (FROZEN 2026-07-23,
    user-directed "scale up to many, many more pairs"; frozen BEFORE form-register
    judging or any scoring)**. Discovery enabling scale: mention-AUC metric ids a<N> are
    CONSTRUCT CLUSTERS each carrying multiple real surface FORMS (humor 1,142 forms/189
    ids; cw 284/106) — pairs come from each metric's OWN attested form inventory, not
    synthetic twins (kills the PREREG-14 low-name-matching confound: selection is by
    silver counts + within-metric height spread, blind to which form matched anything).
    Eligibility (frozen rule): humor metrics with >=25 silver-pos and >=25 silver-neg
    docs and >=2 distinct forms -> 88 metrics. ARMS per metric: the LOWEST vs HIGHEST
    judged-height form (heights from the Sonnet register instrument, anchor-gated, run
    AFTER this freeze); metrics whose form height gap < 1.0 SD (of the form-height
    pool) are EXCLUDED from the primary and reported as a no-spread stratum.
    Items: 20 silver-pos + 20 silver-neg threads per metric (seed 19, union grain as in
    PREREG-14, text 4,000 chars). Scaffold (frozen verbatim, addresses "bare name" limit):
    "You are evaluating stand-up comedy material. The criterion named below is one used
    by the comedy community to judge such material. No definition is provided - apply
    the criterion as you understand it from its name." + the PREREG-11 output contract.
    PREREG-12 licenses a single scaffold (paraphrase proven inert). Judge: gpt-5.6-sol
    via Codex, k=2 fresh runs per (metric, arm).
    PRIMARY (two-sided): per metric dAUC (low-form minus high-form) vs silver;
    inference = sign test + Wilcoxon over metrics (n~50-80 now powers rank tests) +
    pooled stratified bootstrap. SECONDARY: dAUC vs form height-gap (dose-response,
    Spearman); absolute AUC per arm. DESCRIPTIVE (W14/Q3 answer): the register census of
    the silver vocabulary itself (all 1,426 humor+cw forms judged). CW excluded from
    primary (no state-negatives exist); disclosed. ONE analysis run after collection.

16. **PREREG-16 — R2-theme absorption of the NEW lay constructs (FROZEN 2026-07-23,
    user-approved "orchestrate this with Codex as the primary driver"; frozen BEFORE any
    seating judgment on lay-new constructs)**. Motivation: the closure law (median 94%
    theme absorption; zero new atomic object-domain qualities under adversarial
    out-of-register search) has never been tested against the W13 lay corpus — a genuine
    out-of-register sample collected AFTER the taxonomies froze. Pool: the 2,754 lay
    criteria with match=NEW in lay_construct_matches_20260723.jsonl (44% matched ones are
    absorbed by construction at R1 and excluded).
    H (primary, one-sided): the lay-new R2 absorption rate is NOT lower than the
    campaign's within-register median of .94 — test: pooled absorption rate with
    stratified (by field) bootstrap 95% CI; SUPPORTED if the CI lower bound >= .94 is not
    excluded... stated precisely: closure is CONFIRMED if the pooled rate's CI contains
    or exceeds .94; a CI entirely below .94 = closure takes a scope caveat (closed for
    professional discourse only), with per-field rates reported either way. Escaping
    constructs are LISTED verbatim (they are candidate genuinely-lay themes — a finding,
    not a failure).
    Instrument: Codex gpt-5.6-sol judging (per the judge-checks-on-Codex rule); each new
    lay construct (head + source quote/context) is judged against its field's FULL frozen
    R2 theme list (theme_id+name+definition, taxonomies located and path-logged by the
    prep task BEFORE judging) -> verdict "seats in <theme_id>" or "NEW-THEME". Anchors:
    6 professional R1 constructs with KNOWN R2 seats camouflaged per chunk, gate >= 5/6
    correct-theme, failed chunks discarded. Canonical ladder files are NOT modified —
    this is a read-only absorption test; any canonical merge stays gated on separate
    sign-off. Secondary leg (descriptive, separate): community-rule criteria absorption,
    same machinery. ONE analysis run after collection.

17. **PREREG-17 — form POLARITY explains the form-accuracy effects register does not
    (FROZEN 2026-07-23, user-authorized; frozen BEFORE any polarity coding exists)**.
    Motivation (disclosed, from PREREG-15 inspection): the largest CI-clean form effects
    look driven by framing — prohibition/anti-pattern forms ("Rule 12: Be brutal",
    "hacky (anti-pattern)", "No toilet humor") fight the "exhibits/satisfies 1-7" scale.
    NEW MEASUREMENT (only new data): each of the 176 PREREG-15 arm-forms coded for
    polarity ∈ {positive_quality (more is better), prohibition_antipattern (names a
    fault or forbids something), neutral_topic (valence-free descriptive label)} —
    Codex gpt-5.6-sol, 6 hand-built unambiguous anchors per chunk, gate >=5/6.
    H (primary, one-sided): among metrics where EXACTLY ONE arm is
    prohibition_antipattern, ΔAUC favors the non-prohibition arm — one-sided sign test +
    Wilcoxon over those metrics (scores = existing PREREG-15 runs, no rescoring).
    SECONDARY: |ΔAUC| is larger for polarity-mismatched metrics (one prohibition arm)
    than polarity-matched ones — one-sided MWU. DESCRIPTIVE: polarity x register
    cross-tab (is prohibition framing itself register-skewed — the compositional link
    back to PREREG-14's out-of-vocabulary effect). ONE analysis run.

19. **PREREG-19 — specificity-selected arms: does phrasing SCOPE predict judge validity
    (FROZEN 2026-07-23, user-directed "rerun the runs for specificity"; frozen BEFORE
    any specificity judging of the humor form inventory)**. Design mirrors PREREG-15
    exactly except arm selection: for each of the same 88 silver-labeled humor metrics,
    arms = the metric's MINIMUM- vs MAXIMUM-specificity attested forms (buckets from the
    validated instrument, ties broken by ordinal then lexicographic; metrics whose forms
    all share one bucket are EXCLUDED and reported as a no-spread stratum). Items: fresh
    20 pos + 20 neg silver draws per metric (seed 73; disclosed: NOT the PREREG-15 item
    sets — those were lost to a scratchpad wipe; reconstruction of the old register arms
    FAILED fidelity 2/10 on known pairs and was abandoned, disclosed). Scoring: the
    PREREG-15 contextualized scaffold verbatim, gpt-5.6-sol, k=2 runs per (metric, arm).
    PRIMARY (two-sided): per-metric dAUC (general-arm minus specific-arm), sign test +
    Wilcoxon + stratified bootstrap. SECONDARY: dose-response vs bucket gap; absolute
    AUC per arm. ONE analysis run. Companion cell (descriptive, this freeze): LLM-emitted
    names' specificity distribution (GPT 26/73/1, GLM 35/64/2, officials 40/60/0 —
    LLMs elevate register but do NOT generalize scope).

## W15 — R1 SATURATION CAMPAIGN, peer-review (user-directed 2026-07-23; design + stop
## rule FROZEN before wave 1)

Phase-0 diagnosis (ledgered): URL frontier NOT saturated (99.8% distinct across 3,121
visits — collection stopped on budget, not saturation); R1 accumulation has NO KNEE
(end/mid slope ratio .84); GT mm .069 vs Chao1 917-vs-317 — low next-draw novelty with
a long estimated tail. The claim to earn: either a real R1 knee at feasible effort, or
a measured open tail — both reportable.

CAMPAIGN DESIGN (frozen):
- WAVES of Codex web-search runners, 6/wave, angle-diversified (assigned per wave from:
  journal reviewer guidelines by discipline; funder/society standards; university/IRB
  committees; preprint-server & overlay policies; practitioner blogs/forums; editorials
  & meta-science papers; non-English sources; historical (pre-1990) guidance). Runners
  report EVERY candidate URL (no pre-exclusion) — the ingest-side collision rate against
  all previously visited URLs IS the frontier-saturation measure.
- PARSE: same criteria-extraction schema as the census (name, description, guidance,
  subfield, audience).
- HARDENED L0->R1 INTEGRATION (the CRP-integration upgrade): (1) normalized-name exact/
  near dedup vs existing 1,858 L0; (2) candidate shortlist per new L0 (lexical overlap +
  top-10 by token containment vs the 317 R1 head forms); (3) Codex same-construct
  judgment per (new L0, candidate) with camouflaged anchors (known-member positives,
  known-distinct negatives; gate >=5/6, chunk discarded on fail); (4) NEW-construct
  verdicts require a SECOND adversarial pass ("find any existing construct that covers
  this; default to seat") before minting an R1; (5) per-3-wave overmerge audit: 50
  random within-construct pairs re-judged, undermerge audit: 50 nearest cross-construct
  pairs.
- PER-WAVE READOUT (the saturation curve): new-URL rate, docs parsed, L0 draws, new-L0
  rate, NEW-R1 RATE, GT mm at L0/R1, Chao1 trajectory, accumulation slope ratio.
- STOP RULE (frozen): stop when new-R1 per 100 L0 draws < 2 for TWO consecutive waves
  (knee reached), or after 8 waves (open-tail verdict), whichever first. Canonical
  ladder untouched until a separate sign-off; campaign R1s live in the campaign dir.
Artifacts: outputs/lexicon/r1_saturation_peer/ (waves, ledger.jsonl, curve.json).

---
20. **PREREG-20 — sampling-theory validation on a known-denominator frame (AIRules)**
REGISTERED 2026-07-24, BEFORE any sampler simulation or estimator run.

FRAME: AIRules rules_subreddit_set (~100K English subreddits, each with subscriber
count + full rule list; Nov 2024 crawl). Unit = one (subreddit, rule) token. Type
systems: (a) L0 surface grain = normalized rule short-name text (lowercase, strip
punct/markdown, collapse whitespace); (b) R1 construct grain on the humor-domain
quality-rule sub-frame, Codex-seated with the W15 hardened protocol (anchor-gated).
The frame gives EXACT truth: K (type count), each type's sampler propensity, true
unseen mass at any n. All estimators are judged against frame truth.

SAMPLERS (simulated within the frame, with replacement):
- UNIFORM: each rule token equiprobable (the design-based reference).
- POPULARITY: draw subreddit ∝ subscribers^α (α=1), then a uniform rule within it
  (the search-reachability proxy: search surfaces big communities).
- POPULARITY-TRUNC: popularity with the bottom-quartile-subscriber strata given π=0
  (the zero-support regime IPW cannot fix; tests honest failure).

PREREGISTERED PREDICTIONS (direction + single test each; null = no difference):
- P20a (bias direction): at matched n, the POPULARITY sampler's true sampler-relative
  missing mass is LOWER than the uniform-draw unseen mass of the frame — i.e. a
  search-like sampler OVERSTATES saturation. Test: sign of the gap
  M_unif(n) − M_pop(n) over 200 replicates at n ∈ {1k, 5k, 20k}; 95% CI excludes 0.
- P20b (GT is sampler-relative, and correct AT that): GT f1/n tracks M_samp(n)
  (its own sampler's next-draw novelty) within its McAllester–Schapire band ≥90% of
  replicates for BOTH samplers — the estimator is right; the estimand is the sampler's.
- P20c (Chao lower-bound validity): Chao1 and the 5-list capture–recapture estimate
  are ≤ true K in ≥95% of replicates under POPULARITY (heterogeneous capture), and
  multi-list beats Chao1 (larger, still valid, tighter coverage gap).
- P20d (IPW frame calibration): Horvitz–Thompson richness/missing-mass correction with
  propensities ESTIMATED from observables (subscriber decile) moves the POPULARITY
  estimate toward frame truth (≥50% of the deception gap closed) — but FAILS on
  POPULARITY-TRUNC (bound stays below truth; the zero-support theorem in practice).
- P20e (partial ID validity): the reachability-floor bound curve
  Unseen(n, π_min) computed from the observed sample contains frame truth for every
  assumed π_min ≤ the true minimum normalized propensity, and first breaks above it.

READOUTS: per-sampler accumulation + missing-mass trajectories vs truth; deception gap
M_unif−M_samp vs n; estimator validity/coverage table; IPW recovery fraction under full
vs truncated support; partial-ID bound fans. R1-grain replication of P20a on the humor
sub-frame (Codex-seated constructs) to show the result is not a surface-grain artifact.
Artifacts: outputs/lexicon/frame_calibration_20260724/. Analysis:
methods/codability/lexicon/frame_calibration_airules.py. No canonical partitions
touched; judging (R1 seating) on Codex gpt-5.6-sol with camouflaged anchors.

---
21. **PREREG-21 — extraction validity: P/R of norm detection & extraction from webpages**
REGISTERED 2026-07-24, BEFORE any extraction run. Motivation: census precision is
validated (84% mechanical validity; 86-92% fidelity) but RECALL and page-level
detection have never been measured, and never against external gold. A
recall-unknown extractor is a biased sampler (PREREG-20's lesson applies to the
extractor itself).

LEGS:
- LEG 1 (structured gold, item-level): 60 AIRules subreddit pages (stratified by
  subscriber size), reconstructed in native layout (header + description + rules
  sidebar). Production-style extraction (Sonnet, census schema) + an "all-norms"
  variant. Codex same-construct matching vs the page's gold rule list (camouflaged
  anchors every batch). Readouts: item recall vs QUALITY-rule gold (primary; the
  census targets quality criteria), recall vs all-rule gold (secondary),
  precision-vs-gold and precision-vs-page (faithfulness).
- LEG 2 (page-level detection): 60 rule-bearing positives + 60 negatives (30 FAR:
  non-community prose from existing local corpora; 30 NEAR: community pages with
  rules stripped, description/FAQ retained). Readout: detection P/R + AUC;
  prediction P21b: near negatives harder than far.
- LEG 3 (prose gold, panel-union recall): ~50 REAL prose pages already in the
  census with stored page text. Panel = GPT-5.6 + GLM (+ existing Sonnet census
  extraction as third arm) each exhaustively enumerates norms; Codex adjudicates
  the union = reference inventory. Readout: existing census extraction's recall
  vs panel union (ceiling-relative, stated as such).
- LEG 3b (synthetic hard pages, known-truth recall): weave AIRules quality rules
  (register-varied: formal + casual phrasings of the same constructs) into
  generated non-rule prose (blog/essay style). ADVERSARIAL REALISM GATE: a fresh
  Codex judge classifies real vs synthetic pages; only synthetics that pass as
  real enter the test set; fooling rate reported. Readouts: recall of planted
  rules; false-extraction rate from distractor prose.
- LEG 4 (concept coverage): Chandrasekharan macro-norms vs our L0/R1 inventories
  (Codex matching). Readout: macro-norm concept coverage.

PREREGISTERED PREDICTIONS (direction + single test; null = no difference):
- P21a: production-extractor item recall on structured quality-rule gold ≥ .75;
  governance rules extracted at LOWER rate than quality rules (by design).
- P21b: detection AUC ≥ .9; NEAR negatives harder than FAR (higher FP rate,
  paired comparison).
- P21c: census recall vs panel union < 1 with missed items skewed LOW-register
  (mean judged formality of missed < extracted; one-sided Mann-Whitney).
- P21d (key register-bias test): on synthetic pages, planted CASUAL phrasings
  are recalled at a LOWER rate than planted FORMAL phrasings of matched
  constructs (paired by construct; one-sided sign test).
- P21e: macro-norm concept coverage ≥ 80% in governance-adjacent domains.
Judging/matching = Codex gpt-5.6-sol w/ anchors; extraction = production Sonnet
path (Max-plan subagents). Artifacts: outputs/lexicon/extraction_validity_20260724/.

---
22. **PREREG-22 — external human-labeled cluster validation (Sanmi's ask)**
REGISTERED 2026-07-24, BEFORE any matching run. Question: do our pipeline's clusters
agree with HUMAN item->group assignments made independently in prior literature?
GOLD SETS (human-labeled item->category partitions):
- Fiesler et al. 2018: Reddit rules hand-coded into 24 categories (IRR-reported) —
  same object type as our community-rule clustering.
- ReviewAdvisor (Yuan et al. 2021): peer-review sentences with 7 human aspect labels.
- DISAPERE (Kennard et al. 2022): review sentences, finer human typology (2nd PR gold).
- Weld et al. 2022: community values taxonomy (29 values/9 groups, human-coded).
- Dagstuhl-15512 ArgQuality (Wachsmuth 2017): 15 dims / 3 human super-groups.
METHOD: seat gold items BLIND through our existing pipeline (extraction where the unit
is prose; L0->R1->R2 seating; GLM judging with camouflaged anchors), then compare our
partition to theirs: Adjusted Rand Index + pairwise same-cluster F1, significance vs
label-permutation null (1,000 perms). Grain statement fixed in advance: their
categories are R2-grain; agreement is scored OUR-R2 vs THEIR-categories; R1 is
expected to REFINE their partition (refinement rate reported, not penalized).
PREREGISTERED PREDICTIONS:
- P22a: ARI(our R2, Fiesler categories) > permutation null at p<.01.
- P22b: ARI(our R2 themes, ReviewAdvisor aspects) > permutation null at p<.01.
- P22c: our R1 refines external categories (majority of external categories split
  into >=2 R1s) while R2 preserves them (majority map 1-to-1 or 1-to-2).
Samples sized to GLM budget (>=300 items/gold). Artifacts:
outputs/lexicon/cluster_gold_validation_20260724/. Data: sk3
datasets/prior_norms/cluster_gold/. NO Codex (credit freeze); judging = GLM.

PREREG-22 AMENDMENT (2026-07-24, BEFORE any matching run): Fiesler et al. 2018 coded
corpus is NOT publicly released (confirmed: no data-availability statement; author repos
searched; only the 24-category codebook schema is public) → P22a infeasible as
registered; recorded, not silently dropped. ReviewAdvisor release mixes ~1K human-coded
with tagger-labeled spans, unflagged → demoted to secondary. PRIMARY item-level gold
becomes DISAPERE (fully human-coded review sentences, 506 reviews): P22b runs on
DISAPERE aspect labels vs our peer-review R2 themes (ARI vs 1,000-permutation null,
p<.01). ArgQuality (320 args × 15 dims × 3 annotators) supports P22c-style
refinement/grouping checks (descriptive, small-n). Weld subreddit-values = value-level
convergence check only. Judging = GLM w/ camouflaged anchors; no Codex.

PREREG-22 LEG 3 ADDED (2026-07-24, registered BEFORE any Fora matching run):
**Fora grain-calibration leg.** Gold: the Fora corpus subset with human
quote->code->theme annotations (NYC DOHMH dialogues; 7 top-level themes, 20 labels
incl. sublabels; per arXiv 2603.08989 usage). This gold is TWO-LEVEL, so beyond
partition agreement it supports a test of the LEVEL DISTINCTION itself:
- P22f (level alignment): running Fora quotes blind through our L0->R1->R2 pipeline,
  the majority of Fora SUBLABELS map 1-to-1 onto single R1 constructs, while the
  majority of Fora THEMES map onto single R2-grain clusters (i.e., their two human
  levels land at our two corresponding grains, not both at one grain). Test: per
  human label, count the minimal our-level cluster count needed to cover >=80% of its
  quotes; sublabels should need ~1 R1; themes should need >1 R1 but ~1 R2. Null:
  levels are indistinguishable (sublabels and themes need the same our-level spread).
- P22g (partition agreement): ARI(our clusters, Fora labels) > permutation null at
  both grains (p<.01, 1,000 perms).
CAVEAT registered up front: Fora is health-dialogue content, not evaluative criteria;
this leg validates the clustering machinery's LEVEL STRUCTURE, not domain coverage.
Judging = GLM (no Codex). If the corpus requires a data-use agreement, record access
status honestly and mark the leg pending rather than substituting.

PREREG-22 FORA LEG STATUS (2026-07-24): PENDING ACCESS. The Fora corpus
(github.com/schropes/fora-corpus; Schroeder/Roy/Kabbara ACL 2024) is gated behind an
access-request form (MIT CCC, RAIL-variant license, no redistribution). The two-level
NYC labels (7 themes / 20 sublabels / 200 quotes) are from arXiv:2507.15821 ("Just
Put a Human in the Loop?"), which publishes the schema but NOT the labeled quotes.
Per the leg's own registered rule: recorded as pending, no substitution. USER ACTION
required to proceed: submit the access form (URL in cluster_gold/MANIFEST.md on sk3)
or contact the authors. P22f/P22g remain frozen and will run unchanged if access is
granted.

PREREG-22 GRAIN-CALIBRATION SUBSTITUTES (2026-07-24, registered BEFORE any matching):
Fora remains PENDING ACCESS. Two public two-level human-coded golds acquired to run
the SAME frozen P22f/P22g tests (predictions unchanged, gold substituted; recorded as
an amendment, not a silent swap):
- SCRUM gold (Alami & Krancher 2022, Zenodo, CC-BY-4.0): 39 interviews + 2 focus
  groups; per-participant "Data chunk -> Code" sheets + Pattern-Codes sheet grouping
  codes into named themes. Primary substitute (real interview segments, explicit
  code->theme layer).
- ATLAS.TI SUSTAINABILITY gold (vendor sample project): 397 quotations, 155 codes
  nested in 20 tag categories, 1,167 quotation-code links. Secondary (smaller, no
  formal license -- report but do not redistribute).
- UCSB Dryad gold (CC0, DOI 10.25349/D9402J): confirmed to exist with codes->
  sub-themes->themes; download blocked by bot-check; USER manual browser download
  flagged as the unblock.
P22f test restated for these golds: seat gold segments blind through L0->R1->R2;
their CODES should map ~1-to-1 at R1 grain, their THEMES ~1-to-1 at R2 grain
(coverage-count test as registered). P22g: ARI vs permutation null at both grains.
Content caveat carried over: software-process / sustainability interviews validate
LEVEL STRUCTURE of the machinery, not norms-domain coverage.

PREREG OUTCOMES RECORDED 2026-07-25 (detail in ledger entry 2026-07-25a):
- P-SPEC-A (KDL dual-prompt, registered as the discriminant-vs-failure test): SUPPORTED —
  detail-prompt rho=+.637 vs gold, scope-prompt rho=-.293, scope~detail rho=-.363. The
  earlier specificity null was DISCRIMINANT VALIDITY. Instrument reinstated, with the
  caveat that no scope gold exists for evaluative criteria.
- P-SPEC-B (prevalence->scope survives length control): SUPPORTED — rho=+.210 short /
  +.202 long, near-identical across bands; saturating shape (single->mid lift, mid->high flat).
- P21c (panel-missed items are lower-register than found): NOT SUPPORTED — A=.494, p=.44,
  anchor gate PASS. Register explanation for Leg-3 misses is ruled out.
- P22b (our R2 themes align with DISAPERE aspects above chance): SUPPORTED but WEAK —
  ARI .097 seated (p=.001 vs 1,000-perm null), AMI .170, K_ours 27 vs K_gold 8,
  homogeneity .265 > completeness .192. Above chance, low in absolute terms; grain
  mismatch and genuine divergence both contribute. NOT to be quoted as a headline
  clustering-quality figure. P22f/P22g on the two-grain Scrum gold remain the designed
  level-alignment test (running 2026-07-25).

PREREG-23 (registered 2026-07-25, BEFORE any run): external + internal validation of the
L0->R1->R2->R3 clustering. Gold inventory and licences: datasets/prior_norms/cluster_gold/MANIFEST.md.

GOLD-MATCHING LEGS (frozen predictions):
- P23a O*NET seating, full 4-level, links present. Seat a stratified sample of O*NET task
  statements blind through L0->R1->R2->R3. PREDICT: unprompted K within +-25% of gold K at
  each matched level (the P22f grain test, now at 4 levels); ARI/V above a 1,000-perm null
  at every level; V highest on the diagonal (our R1 ~ DWA, R2 ~ IWA, R3 ~ GWA).
- P23b Code-review ESEM'23, ON-DOMAIN, 2 levels, 1,829 human-labelled comments.
  PREDICT: fine pass K ~ 19 categories, coarse pass K ~ 5 groups; diagonal V. This is the
  only on-domain linked gold, so it carries the domain claim; O*NET carries the scale claim.
- P23c UCSB codebook reconstruction (TREE ONLY, no links). Strip the backslash paths, seat
  the 93 leaf codes blind, recover the gold parent at depth-2 and depth-1. Registered
  exclusion: drop the Course\Demographics attribute branch; report filtered AND unfiltered.
  This is the closest analogue to what our R1->R2->R3 steps actually do (they group
  construct NAMES, not raw quotes).
- P23d tree-shape test, NO JUDGE CALLS: branching factor and depth distribution of our
  L0->R3 trees vs O*NET's and UCSB's. Runs without any API.

INTERNAL LEGS (no gold needed; these measure things a gold cannot):
- P23e split-half stability. Re-seat two disjoint halves of the same L0 pool; ARI between
  the induced partitions on shared items. PREDICT: >= .60. Below that, no gold number means
  anything, because the instrument is not reliable with itself.
- P23f batch-order invariance. Permute batch composition 5x, same items. Our pipeline is
  batch-based and this has NEVER been quantified. PREDICT: mean pairwise ARI >= .70.
- P23g injection-recovery. Plant known paraphrase sets (should merge) and known near-miss
  distractors (should not) into the L0 pool. Yields clustering P/R against a KNOWN answer
  at OUR grain -- the only design that gives a clean P/R number without an external gold.
- P23h cross-family agreement. Independent seatings by >=2 model families; ARI between them
  separates "our taxonomy" from "one family's prior".

HUMAN LEG (user opened this 2026-07-25, relaxing the standing no-human-studies rule;
metrics remain label-blind -- human labels validate the INSTRUMENT only, never feed a metric).
IRB determination REQUIRED before fielding.
- P23i odd-one-out triads. 300 triads x 5 raters, stratified: (A) 2 same-R1 + 1 other-R1,
  (B) 2 same-R2/different-R1 + 1 other-R2, (C) random control. PREDICT accuracy
  A > B > C, C ~ chance (.33). Directly tests whether OUR boundaries are the boundaries
  humans see. Cheapest decisive instrument; no taxonomy training needed.
- P23j open card sort (run only if P23i passes). 12-15 participants each free-sort ~60
  construct names into self-named groups; consensus co-occurrence matrix vs our R1/R2.
  Elicits a human hierarchy WITHOUT imposing ours -- the strongest form of Sanmi's ask.

PREREG-23 AMENDMENT (2026-07-25, registered BEFORE arm A judging): the design becomes a
TWO-ARM measurement over 6 external human-coded hierarchies. Harness:
methods/codability/lexicon/external_gold_{harness,batches,score}.py; artifacts outputs/lexicon/prereg23/.

DESIGN. A "rung" is a pairwise test of ONE frozen prompt against ONE gold link:
  L0 rung: two surface items, gold same = share an R1 parent  (l0_precision_audit.PROTOCOL)
  R1 rung: two R1 labels,    gold same = share an R2 parent   (build_level.RELATIONS['R1'])
  R2 rung: two R2 labels,    gold same = share an R3 parent   (build_level.RELATIONS['R2'])
Prompt text is IMPORTED from the modules that built our corpus, never retyped, so arm A
measures the instrument we actually used. 15 cells / 3,960 pairs; 140 pos + 140 neg per cell.
Negatives stratified 50/50: HARD = different parent, SAME grandparent; EASY = different
grandparent. HARD-negative AUC is the headline; pooled AUC is composition-dependent and is
never quoted alone. 6 camouflaged anchors per batch, gate >=5/6, discards reported.

ARM A (frozen prompts) PREDICTIONS, registered now:
- A1 every cell AUC_hard > .50 (above chance against the toughest negatives).
- A2 AUC_easy > AUC_hard in every cell (easy negatives are easier). If a cell VIOLATES this,
  the prompt is behaving as a topic detector rather than a level detector -- report as such.
- A3 AUC_hard is HIGHER on the on-domain gold (codereview) than on the off-domain golds
  (onet, pdtb). This is the domain-transfer test; a null here means our prompts encode a
  domain-general notion of "same criterion", which is a POSITIVE finding for the paper,
  not a failure -- registered in advance so it cannot be reinterpreted after the fact.

ARM B (GEPA ceiling) PREDICTIONS:
- B1 GEPA-optimised prompts beat frozen prompts on a HELD-OUT fold of the same gold.
  Train/test split is by gold PARENT (not by pair), so an optimised prompt cannot win by
  memorising specific labels. Ceiling = held-out AUC after N rounds.
- B2 the gap B-A decomposes the arm-A shortfall: gap large => our prompt is suboptimal and
  fixable; gap small => the residual is irreducible disagreement between two coding schemes
  built by different teams for different purposes.

STANDING CAVEAT, registered so it cannot be forgotten later: a GEPA'd prompt is a DIFFERENT
INSTRUMENT from the one that built our corpus. Arm B measures a ceiling; it does NOT license
swapping the optimised prompt into the pipeline. Any such swap is a separate decision requiring
sign-off and a full rebuild, because it would make our hierarchy the target gold's shape.

NOISE-CEILING GAP (open, acknowledged): none of these golds is double-coded, so we do not
know the human-human AUC on these same pairs. Arm A/B numbers are therefore uncalibrated in
absolute terms and must be read as CONTRASTS (cell vs cell, arm vs arm), not as "we are X%
correct". The P23i triad study supplies the missing ceiling via inter-rater agreement.
