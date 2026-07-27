# Seam position, the retrieval thesis, and codability priors

*2026-07-05. Extends the binding/provenance note (2026-07-04) and the transport results
(results note §TRANSPORT). Three questions from AS: (Q1) what does it mean for the
beginning / middle / end SEGMENTS OF A METRIC PROGRAM to be coded vs LLM-prompted, and how
do we test it; (Q2) what would it take to fully prove that LLM-prompting is retrieval-based,
and what literature exists; (Q3) can we predict codability a priori from the metric itself?
Sections marked ◻ are pre-registered designs, written before any of their data was collected.*

---

## 1. Q1 — Seam position: a metric program has three segments, each independently coded or borrowed

### 1.1 The pipeline decomposition

Any implementation of a criterion factorizes as

**m = A ∘ T ∘ R**

- **R (read):** raw document x → intermediate representation u. *What counts as an instance.*
- **T (transform):** u → construct-relevant quantities v. *What the construct means.*
- **A (aggregate):** v → score. *How much each part matters.*

Each stage is either **C** (code: meaning specified in mechanism) or **L** (LLM-prompted:
meaning retrieved from enculturated competence), giving a 2³ architecture lattice. Where we
sit today:

| arch | meaning | status |
|------|---------|--------|
| CCC | pure code | E₁ floor (3 flavors/criterion, all tasks) |
| **LCC** | LLM fields read **raw x**; code transforms + aggregates | our entire E₂ hybrid fleet |
| **CLC** | code builds a view ν(x); LLM prompted **only on ν(x)**; code aggregates | ◻ untested — "middle-section LLM-prompted" |
| **CCL** | code computes features; LLM aggregates the feature vector | ◻ untested |
| LLL | prompt-the-judge | the judge itself (the reconstruction target's own arm) |

### 1.2 What it *means* for the middle to be LLM-prompted

A middle-position LLM never sees the document. Its field output F satisfies
**F ⊥ X | ν(X)** — everything the borrowed judgment learns about the document passes through
the coded interface. Two consequences:

1. **Aperture bound (data processing).** The recoverable signal of a CLC field is capped by
   what the coded view preserves: it can do no better than the best measurable function of
   ν(X). This is the level-matching theorem (C6) turned inward: the evidence-op result said
   *code can't smuggle in world-state the judge never saw*; the aperture bound says *the LLM
   can't retrieve construct-meaning whose textual keys the code threw away.*
2. **Binding is rigid at both boundaries.** In LCC the LLM's input side is open (raw text);
   in CLC both sides are typed, code-conformance-checkable interfaces. Provenance of T stays
   enculturated. CLC is therefore the missing cell of the binding × provenance grid:
   maximally rigid binding wrapped around a still-borrowed transformation. "LLM-prompted
   middle" = **borrowed judgment applied through a coded aperture.**

So the three seam positions localize three distinct loci of borrowed meaning:

- **L at R** — borrowed *perception* (recognizing instances in raw surface),
- **L at T** — borrowed *conceptualization* (the construct itself, evidence base formalized),
- **L at A** — borrowed *valuation* (the community's importance-weighting of evidence).

The document-segment reading of "beginning/middle/end" is subsumed: a code stage that hands
the LLM only the head, middle, or tail of the document is a *positional aperture* — a special
case of CLC. (Flag: our pack-builder already imposes a head+tail aperture at improver time —
`text_excerpt_head`/`_tail` — so improvers never saw document middles; worth one sentence in
limitations.)

### 1.3 ◻ Seam-position experiment (SEAM-POS), drop-in on PR v2

For the 12 gate-certified PR criteria (most instrumented; extend to humor/CW certified sets):

| condition | construction |
|-----------|--------------|
| LCC | existing hybrid, existing fields (certified condition) |
| CLC-digest | ν(x) = JSON of: first 3 + last 3 sentences, doc stats, the baseline program's matched spans/counts, top-5 TF-IDF sentences vs criterion description (all E₁ ops). Same field question, asked of ν(x) only. |
| CLC-pos(head/mid/tail) | ν(x) = one document third; same field question | 
| CCL-llm | LLM prompted with the program's *feature vector* (code signal + field values), returns the score |
| CCL-fit | isotonic/logistic aggregator fit on train over the SAME feature vector (mechanical control) |

Readouts, all on the frozen held-out split: ρ per condition; **fm per position**
(fm_R = ρ_LCC − ρ_blank as now; fm_T = ρ_CLC − ρ_code-only-on-ν; fm_A = ρ_CCL-llm − ρ_CCL-fit);
**aperture loss** = fm_R − fm_T; per-position transport ratios (swap only that stage's
extractor family).

Predictions (retrieval thesis): (i) taste-flavored constructs (tone, craft) have retrieval
keys *distributed in the raw surface* → large aperture loss; extractive constructs (apology
present, spokesperson quoted, date given) survive the aperture. (ii) fm_A > 0 on criteria
whose failed improvers "replaced the construct with a pointer" — if the *weighting* is where
the norm lives, a fitted aggregator on identical features should underperform the LLM
aggregator. (iii) positional apertures locate where in documents the keys live (PR: head;
humor timing: tail).

Cost: ~250 items × 12 criteria × 4 new conditions ≈ 12k short prompts = one GPU-7 Gemma
pass + one CPU fit. Queued behind the legal extraction.

---

## 2. Q2 — The retrieval thesis: what full proof requires

**Thesis (T-RET).** A field prompt functions as a *retrieval key* into competence acquired
from community practice during training. Its operational content mostly **locates** the
construct; it does not **specify** it. (Code is the opposite pole: its meaning is exhausted
by its mechanism.)

### 2.1 Rival hypotheses

- **H_spec** (prompt-as-program): the prompt text specifies the mapping; any competent
  executor merely runs it.
- **H_idio** (checkpoint idiosyncrasy): field signal is model-specific pattern-matching,
  not shared meaning.
- **H_leak** (surface shortcut): fields work through keyword-like correlates; no construct
  at all.
- **Locus question** (within T-RET): is the enculturation acquired in pretraining or
  instruction-tuning?

### 2.2 Evidence in hand

| result | reading |
|--------|---------|
| blank ablation: fm > 0 across 120 criteria | fields carry real signal beyond code (vs nothing) |
| transport: td tracks fm, Spearman .59 | the thing lost under family swap is exactly the borrowed payload — it lives in the interpreter, not the program text (vs H_spec) |
| transport: median ratio .30 | ≥⅔ of payload survives the swap → **shared** training culture (kills strong H_idio); the .30 remainder is family-bound, hence certificates stamped with extractor family |
| judge-level decompression grid (Face-2): name ≈ dense rubric for taste; checklists can hurt | names act as indexes at the judge level (vs H_spec) — but not yet shown at the *field* level |
| 3/12 PR certified gates fail under swap; worst is the most taste-flavored criterion (humble tone, ratio ≈ 1) | boundness is graded and concentrates exactly where enculturation should |

### 2.3 ◻ The proof battery

No single experiment proves T-RET; the battery works because each rival survives some cells
and dies in others.

| # | experiment | design | T-RET predicts | kills |
|---|-----------|--------|----------------|-------|
| E1 | **KEY** (key deprivation) | 3 field-prompt forms per criterion: name-only; nonce-name + full operational definition; full (name + definition) | name-only ≈ full ≫ nonce+definition | H_spec (definition should suffice); partial H_leak |
| E2 | **STIP** (stipulation override) | redefine the term deviantly in the prompt ("for this task, 'humble tone' means mentions a dollar figure"); measure compliance vs snap-back to community meaning | snap-back dominates, weakening with extractor scale (cf. flipped-label ICL) | H_spec directly; quantifies binding ("semantic gravity") |
| E3 | **SCALE** (same-family staircase) | re-extract all fields with Llama 3B / 8B / 70B (sanctioned primary family); fm and transport ratio vs scale | fm grows with enculturation depth; H_spec predicts near-flat (executing a short spec is easy) | separates retrieval from execution |
| E4 | **LOCUS** (base vs instruct) | same fields, Llama-70B base (few-shot format) vs instruct | if base ≈ instruct: payload is pretraining culture; gap = alignment-tuning contribution | resolves locus |
| E5 | **APERTURE** | = SEAM-POS §1.3 | retrieval needs surface keys; aperture starves taste constructs first | H_spec (a spec survives re-representation); partial H_leak |
| E6 | **3FAM** (in flight tonight) | Qwen-122B third-family extraction; per-criterion transport ratios across family pairs | Spearman(ratio_G→L, ratio_G→Q) > 0: boundness is a property of the **criterion** (its cultural specificity), not the pair | strong H_idio, again |
| E7 | **SEL** (selected vs enculturated; flagged, needs sign-off) | LDA/CE channel vs LLM field at matched binding/interface | selected components transport by conformance, enculturated by culture-overlap | isolates the provenance rung |
| E8 | **ARTIC** (soft, optional) | extractors articulate the working definition they used; compare across families | convergent articulation on high-transport criteria | descriptive corroboration only |

Honesty note on H_leak: E1's nonce condition kills "shortcut keyed to the construct name,"
and held-out gating + the anti-overfit contract already price in generic shortcuts, but a
content-preserving paraphrase lesion of the *documents* (does field signal survive
style-normalization?) is the clean kill. Optional E9 if a reviewer pushes.

Status: E6 running now (driver mid-math). E1/E2 are prompt-file builds + one Gemma pass
each (cheap, no new measurement target). E3/E4 need Llama 3B/8B/70B servings on GPU 7
(sequential, one at a time). E7 stays flagged for sign-off.

### 2.4 Literature (three-scout sweep, 2026-07-05, synthesized)

**Line 1 — prompting/ICL as task LOCATION, not learning.** The retrieval thesis has a real
lineage: Xie et al. 2021 (arXiv:2111.02080) model ICL as implicit Bayesian inference of a
latent pretraining concept; Min et al. 2022 (2202.12837) show gold labels in demonstrations
barely matter — demos *locate* the task; Reynolds & McDonell 2021 (2102.07350) said it
first ("prompts locate tasks in the model's existing capability space"). Pan et al. 2023
(2305.09731) give us the vocabulary we should adopt: **task recognition (TR) vs task
learning (TL)** — our fm decomposition + transport is a behavioral, production-context TR/TL
split; descendants localize TR and TL in distinct attention-head populations (Yang et al.,
ICLR 2026, 2509.24164). Mechanistic substrate for "a short prompt reduces to a pointer":
task vectors (Hendel et al. 2023, 2310.15916), function vectors (Todd et al., ICLR 2024,
2310.15213), and label-words-are-anchors (Wang et al. 2023, 2305.14160) — the last is the
circuit-level story for *why a bare construct name can suffice while a verbose checklist
dilutes the anchor*. Two disciplining complications: Wei et al. 2023 (2303.03846) — prior-
override capacity is scale- and tuning-dependent (feeds E2/E3 predictions directly); Kossen
et al., ICLR 2024 (2307.12375) — ICL does use in-context labels, so T-RET stays scoped to
short field prompts, not ICL broadly. Li et al. 2024 (2406.04216) is the only paper that
stages retrieval-vs-learning as competing hypotheses and finds *compositional* retrieval —
our "one construct = one lookup" may really be composition of pretrained sub-skills.

**Line 2 — cross-model convergence + judge sensitivity.** The shared-culture reading of
transport ratio .30 has independent support: Platonic Representation Hypothesis (Huh et al.,
ICML 2024, 2405.07987), relative representations / zero-shot latent communication (Moschella
et al., ICLR 2023, 2209.15430), and model stitching (Bansal et al., NeurIPS 2021) — the
methodological ancestor of swap-and-measure. Models carry a *specific* culture, only partly
prompt-steerable: Santurkar et al. 2023 (OpinionsQA), Durmus et al. 2023 (2306.16388),
AlKhamissi et al. 2024 (2402.13231). Judge-side base rates: GPT-4-judge/human agreement
~80% (Zheng et al. 2023); LLM errors are correlated even across providers ("algorithmic
monoculture", Kim et al., ICML 2025, 2506.07962; a 9-judge/7-family panel ≈ 2 independent
votes, 2605.29800) — which BOUNDS how much independence a family swap can buy and makes our
70%-shared finding the expected sign, quantified per-criterion. Panickssery et al., NeurIPS
2024 (2404.13076): models recognize their own generations — a mechanistic candidate for the
family-bound 30%. Sclar et al. 2023 (2310.11324): formatting alone swings accuracy — the
null our transport must survive (some of the 30% may be format sensitivity, not culture;
worth one ablation sentence). Closest methodological neighbor: **PromptBridge (Wang et al.
2025, 2512.01420)** names "model drifting" of hard prompts across families — but treats
drift as a deficit to FIX with a learned bridge, whereas we measure the un-bridged split
(family-bound vs shared) on frozen programs and make it a certificate coordinate.

**Line 3 — reference vs description, neologisms, tacit knowledge.** The philosophy mapping
is already live in the literature: Mandelkern & Linzen, *Comp. Ling.* 2024 (2308.05576),
argue LM words refer via causal-historical chains in the training data — exactly the
license needed to read field prompts as rigid designators tapping Putnam's division of
linguistic labor; Baggio & Murphy 2024 (2406.00159) is the internalist rejoinder we must
cite as the live objection; Bender & Koller 2020 the skeptical pole; Lederman & Mahowald
2024 ("bibliotechnism", 2401.04854) formalize meaning-inheritance and its novel-reference
stress case — which is precisely our E1 nonce condition. **E1 has a word-level precedent:
WinoDict** (Eisenschlos et al. 2022, 2209.12153): novel word + explicit in-prompt definition
→ performance collapses vs the familiar word. **E2 has two:** MAGNIFICo (Patel et al., EMNLP
2023, 2310.11634) — stipulated novel interpretations are only partially followed; and a
Stroop-paradigm lexical-override study (2606.07555): "doctor means forest"-style glossaries
show lexical-prior strength predicts interference across 11 models — our E2 run at the
construct/field level, with directionality predicted. Wu et al., NAACL 2024 (2307.02477,
"Reasoning or Reciting?") extend override-failure to whole task conventions; Longpre et al.
2021 give the knowledge-conflict methodology. Tacit-knowledge line: Kambhampati 2021
(Polanyi's revenge); CheckEval 2024 (checklists improve judge *reliability* — the other pole
of decompression); Shankar et al., CHI 2024 (evaluation criteria emerge through practice,
not one-shot specification); Feedback-to-Rubrics 2026 (2605.29857) — the most literal
attempt to compile tacit expert judgment into explicit rubrics, i.e., an existence test for
how far the code arm can go.

**Novelty ledger (what remains ours).** (i) The unit of analysis: a *certified field inside
a frozen hybrid metric program*, not a raw ICL benchmark — nobody measures instruction
transport as certificate loss. (ii) The un-bridged cross-family transport ratio as a
*measured split* (shared culture vs family-bound) rather than a deficit to engineer away
(contrast PromptBridge). (iii) Name-vs-checklist decompression at the field level is
untested anywhere (nearest: label-words-are-anchors, CheckEval — neither runs the graded
ablation). (iv) E1/E2 are word-level-precedented, construct-level-open: we inherit
directional priors, and our versions attach to a certificate framework with ceilings and
held-out gates. The TR/TL vocabulary (Pan et al.) should be adopted in the paper's related
work as the nearest conceptual frame.

---

## 3. Q3 — Codability priors: can we predict the seam from the metric text alone?

**Answer: almost certainly partially, and we already have the outcome data to check it
cheaply. Do the light interpretable probe; do NOT build a heavy learned model.** n ≈ 140
criteria clustered in 6 tasks cannot support a serious learned predictor, and the
interesting claim is not "an ML model can predict r̃" but *which readable properties of a
criterion's phrasing predict where its seam falls* — the coefficients ARE the finding.

### 3.1 ◻ CODA probe (pre-registered feature schema)

Unit: one criterion description (+ name). Outcomes (already computed, never label-aware):
**y_code** = ceiling-normalized code floor r̃ (best of 3 flavors); **y_fm** = field marginal
(hybrid tasks); **y_gate** = certified yes/no. Eight typed features, each 0–2, annotated
from the criterion text by a Sonnet pass **with blinded synthetic anchors** in every batch:

| feature | gloss | predicted sign |
|---------|-------|----------------|
| F1 quantifiability | names countable/extractable quantities (dates, counts, lengths, sections, dollar amounts) | +y_code |
| F2 span-locality | truth-makers are locatable spans (a quote, a disclaimer, a citation) vs global gestalt | +y_code |
| F3 norm-deixis density | evaluative terms whose extension is community-fixed ("appropriate", "engaging", "professional") | −y_code, **+y_fm** |
| F4 reader-effect dependence | defined via effect on an audience ("clear to a lay reader", "funny", "persuasive") | −y_code, +y_fm |
| F5 rule-shape | phrased as requirement/prohibition/threshold ("must include", "avoids", "at least") | +y_code |
| F6 aggregation breadth | one check vs weighing many soft parts ("balances", "overall") | −y_gate |
| F7 specialized world-knowledge | statutes, math correctness, domain facts needed | −y_code (E-level confound; record, don't lean) |
| F8 cross-positional structure | requires relating positions/order/dates within the doc | +y_code (ops reachable) |

Analysis: per-feature Spearman with y_code/y_fm; then rank-ridge on F1–F8 evaluated
**leave-one-task-out** (train 5 tasks, predict the held-out 6th) — the only honest CV under
task clustering. Baselines: (a) zero-shot "rate codability 0–10 from the description" — if
zero-shot matches the feature model, the features are explanation, not machinery; (b) rel1
alone (is "predictable" just "reliable"?). Report LOTO Spearman with bootstrap CI, per-task.

Anchors for the annotation batch (blinded, synthetic): "word count exceeds 500" (F1=2,F5=2,
F3=0), "the piece feels alive" (F3=2,F4=2,F1=0), "cites at least two precedent cases by
name" (F1=2,F2=2), "maintains a professional register throughout" (F3=2,F6=2), plus two
mid-band. A batch whose anchors miss expected patterns is discarded and re-run (degenerate-
pass discipline).

### 3.2 Why this is theory, not just engineering

The features are all measures of one latent: **how far the source community has already
compiled the norm into explicit, checkable form.** Law scores high on F1/F5/F8 because
legislatures and courts spent decades compiling; humor craft scores high on F3/F4 because
comedy transmits by apprenticeship and audience feedback, not by rulebook. So a working CODA
probe quantifies the paper's anthropological throughline ex ante: **the seam is predictable
from the criterion's phrasing because the phrasing inherits the institutionalization of its
community.** Corollary prediction worth stating in the paper: the SAME features that predict
low y_code predict high y_fm — enculturation load concentrates exactly where compilation
stopped.

Ex-ante caveat: "a priori" here means *before writing or running any implementation* — the
features still read the criterion's text. That is the right notion: it is what a
practitioner deciding "code it or prompt it?" actually has in hand.

### 3.3 Verdict on "should we build a codability predictive model?"

Build the probe (half a day: one Sonnet annotation batch + 40 lines of LOTO analysis), not a
model. Ship it as one table + one scatter (predicted vs realized r̃, colored by task). If
LOTO Spearman lands ≥ .5 it earns a paper subsection ("the seam is legible in the metric's
phrasing"); if it lands low, that is itself reportable as "phrasing underdetermines the
seam — you must run the pipeline" (a result descriptively, no sweeping verdict).

---

## 4. Run queue implied by this note

1. tonight: E6 Qwen extraction finishes → three-family transport synthesis (ratios + cross-pair criterion correlation).
2. GPU 7 next: legal field extraction → legal gates → legal CAM (5th money-figure task).
3. CODA probe (CPU + Sonnet only) — can run in parallel now; legal fm joins later.
4. E1 KEY + E2 STIP prompt builds (CPU) → one Gemma pass each behind legal.
5. SEAM-POS (§1.3) prompt build → one Gemma pass.
6. E3/E4 Llama staircase/base-instruct — needs 3 servings, schedule after the above.
7. E7 SEL — awaiting sign-off (new channel type).

## 5. CODIF — a segment-level coding scheme for what gets codified (proposed 2026-07-05)

Analogue of the decompression-rung coding in the tacit-scaling stream (name / definition /
explanation / exemplars / dossier = ways tacit knowledge is explicated to a SMALLER MODEL).
CODIF labels the ways judgment is explicated to a MACHINE — code being the zero-parameter
limit of the same staircase. Grounded in a harvest of the 143-program h0 fleet (5 tasks):
def-names are plumbing (_sat 23, _clean 9, _clamp 8...); the semantic vocabulary lives in
ops calls (normalize 137, sent_stats 44, proof_skeleton 20, equation_stats 19,
retrieve_similar 18, delimiter_health 18, extract_math_spans 13, extract_dates 12) and in
inline archetypes, which the program docstrings narrate consistently.

Tags (v2 names; one code segment can carry several). Each is aligned to its decompression
rung: the tacit-stream ladder classifies how competence is explicated to a SMALLER MODEL;
CODIF classifies the same explicitation moves compiled for a machine.

  C1 SCRAPE-REPAIR        unwrap hard-wrapped lines, strip nav chrome, normalize —
                          perceptual cleanup, epistemically neutral. [no rung: legibility]
  C2 SIGNIFIER-MATCH      regexes over the community's own lexicalized markers ("FOR
                          IMMEDIATE RELEASE", newswire tags, "for example") — the
                          concept's NAME frozen on the artifact surface; fleet convention:
                          gates/damps only, never the quality signal.    [rung: NAME]
  C3 FORM-MEASURE         structure made countable: sent_stats, paragraph mass, bullet
                          lines, equation/notation stats, proof_skeleton,
                          delimiter_health.                        [rung: DEFINITION]
  C4 PLACEMENT-RULE       position/deixis made explicit: dateline at head, reveal
                          position, head/tail slicing.             [rung: DEFINITION]
  C5 EXTRACT-COMPUTE      typed spans pulled out and computed over: dates -> arithmetic,
                          math spans, counts.                      [rung: DEFINITION]
  C6 EXEMPLAR-MATCH       retrieve_similar: judge by proximity to reference instances
                          (kNN precedent) — showing cases instead of stating rules.
                                                                   [rung: EXEMPLARS]
  C7 RATIONALE-ARITHMETIC how evidence composes into a verdict: gate x structure,
                          saturations, thresholds, judge-band-tuned weights — the
                          reasoning ABOUT the construct rendered as arithmetic (the
                          A-stage in A.T.R terms).                 [rung: EXPLANATION]
  C8 BORROWED-JUDGMENT    the LLM-field slot + typed guard rails (_is_none_answer,
                          parse/snap): the residual that resists all of the above and is
                          delegated to an enculturated reader. Its PROMPT itself sits on
                          the decompression ladder — E1-KEY manipulates exactly that
                          (name-only vs nonce+definition).   [rung: NONE — stays tacit]

A whole program = stacked rungs (the DOSSIER analogue). LLM fields additionally get a
thick-predicate tag (kind-judgment, groundedness, tone, craft...) — a42_h0's docstring
states the division verbatim ("thick-input distinction ... hence the LLM_FIELDS; the
predicate stays in code").

Cross-taxonomy predictions (tie the two streams):
  TASTE (name suffices, checklist hurts)  -> codifies C2/C6-heavy or C8-only; key-like E1.
  CRAFT (definition/exemplars decompress at cost) -> codifies via C3/C4/C5; spec-like E1
        (nonce+definition survives — observed in math).
  MECH  (codified but never lexicalized)  -> C3/C5/C7 with NO C2 (no community surface
        marker exists to match).

Use: annotate all 143 programs (docstrings make this cheap; blinded anchors per batch),
then cross segment composition with the per-criterion outcome panel we already have
(CAM r-tilde, fm, aperture frac_kept, transport ratio, E1 key-likeness). Readout:
which explicitation moves retain signal when codified — the empirical complement to
CODA's a-priori F1-F8 (predict-from-metric-text) and the code-side mirror of the
decompression isomorphism (large->small ~ LLM->code).
Status: PROPOSED, not run — annotation pass awaits sign-off.

## 6. E7 SEL — spec for sign-off (redesigned to be reconstruction-clean, 2026-07-05)

Original sketch had the selector trained on judge labels — that brushes the
reconstruction-only constraint. Redesign: **distill the field, never touch the judge.**

Design (per E1-selected criterion field):
  1. Train a lightweight selector S on (x_train, F_gemma(x_train)) — the LLM field's OWN
     outputs on train items, exactly the C_dense own-verdict pattern. Two selector
     classes: TF-IDF logistic regression (surface-lexical) and frozen-BGE + linear head
     (surface-semantic). No judge scores anywhere in training.
  2. Test readouts (judge appears ONLY in the final rho, as in every other arm):
     a. agree(S, F) on test items — how distillable is the field into surface features?
     b. fm_S vs fm_F: plug S's predicted values into the frozen program in place of F;
        frac_distilled = fm_S / fm_F.
     c. cross with E6 transport ratio and E1 frac_nonce per criterion.
  3. Interpretation grid (provenance rung):
     high distillability x high transport  -> codified-surface construct (MECH-adjacent;
        the "LLM field" was a surface shortcut — H_leak confirmed for that criterion)
     low distillability x high transport   -> enculturated competence (TASTE-adjacent;
        shared across families but not reducible to surface features — T-RET's home cell)
     low distillability x low transport    -> checkpoint idiosyncrasy residue
     high distillability x low transport   -> overfit surface quirk (rare; flags leakage
        into one family's reading of the corpus)
Cost: TF-IDF arm pure CPU laptop; BGE arm one small sk3 job. No new extraction needed
(reuses existing field results for train targets and the battery's test extractions).
Label-audit: judge scores touched only in rho readouts; selectors never see them.

## 7. E2-KIND — which kinds of stipulation conflict have salience (AS design, 2026-07-06)

AS: thin-redefinition is ONE conflict kind; separate the NAME's gravity from the
DEFINITION's gravity. Full grid, per e2-bearing field (X = true operational def,
X' = deviant def, X'' = arbitrary neutral rule conflicting with nothing):

              | true def X          | deviant def X'        | neutral rule X''
  name        | (1) control=E1 full | (2) = E2 STIP (run)   | (5) name+neutral
  nonce(gorb) | (3) = E1 keynonce   | (4) NEW — key cell    | (6) nonce+neutral

Predictions:
- H(name-gravity): snap-back lives in the lexical key. (4) complies high EVEN fast-mode
  (no community word present -> no gravity well); (2) snaps back (observed ~.40 fast).
- H(concept-gravity): the model infers the underlying community concept from X' itself
  (X' mentions domain vocabulary) and snaps toward X-behavior even under a nonce ->
  (4) shows "phantom snap-back": answers matching X-truth despite gorb+X'. Requires
  scoring (4) against BOTH X'-truth and X-truth (three-way readout).
- Instruction-following-deficit control (AS): (6) nonce+neutral rule = pure rule-following
  capacity per model/scale; conflict effect := compliance(6) - compliance(4) and
  compliance(5) - compliance(2), which subtracts the deficit confound. Run the whole
  grid down the Llama ladder (3B/8B/70B) + Qwen toggle: deficit falls with scale,
  gravity (per T-RET) should NOT.
Lit dedup BEFORE running: E1-nonce precedents already logged (§2.4: WinoDict, MAGNIFICo,
lexical Stroop); AS recalls a Xiang Ren paper w/ invented nouns + instruction following —
scout to confirm we're testing conflict-KIND salience at construct level, not re-running
their word-level result.

### 7.1 E2-KIND lit dedup (scout, 2026-07-06)

The half-remembered Ren-lab paper = **Li, Yan et al., "Instruction-following Evaluation
through Verbalizer Manipulation" (Findings NAACL 2024, 2307.10558)**: natural / neutral
(foo-bar) / unnatural (flipped) VERBALIZERS — our exact true/neutral/deviant axis, but on
OUTPUT LABELS, not concept names; even GPT-4 near chance on unnatural. Closest ancestor
of the grid: **MAGNIFICo (2310.11634)** — plausible/foreign/adversarial word-forms for
the same novel interpretation (SQL only, no neutral arm, no toggle, no scale law).
**WinoDict (2209.12153)** contains a meaning-shift ablation ({real x deviant}) alongside
its nonce arm — cite preemptively; it already reports real-word-shift > nonce difficulty.
Reasoning-toggle x prior-override nearly empty: only 2604.10511 (policy counterfactuals,
CoT benefit VANISHES on counter-intuitive cases — OPPOSITE sign to our toggle result;
cite as domain contrast: belief priors vs lexical priors). Also adjacent: symbol tuning
(2305.08298), Reasoning-or-Reciting (2307.02477), When Models Ignore Definitions
(2602.17520, 30 items), Definition-Specific Familiarity (2606.00467: definitions rescue
only ~35% of prior-driven errors), knowledge-conflict survey (2403.08319).
VERDICT: full factorial {real,nonce} x {true,deviant,neutral} at CONSTRUCT level +
reasoning-mode IV + extraction setting = open. Frame vs MAGNIFICo + 2307.10558.

## 8. AGENTIC-COMPILE — flexible pipeline arm (spec, approved 2026-07-06)

Question: is the code-sufficiency boundary (26% pure-code / 56% code-carried) a fact
about the DOMAIN or about our static compiler (pack -> h0 vs fixed ops -> gate -> one
h1 round)?

Sample: 12 criteria from the tail, stratified — gate-FAIL (rescue test) + certified-but-
field-dominated c8=HIGH (recode test); PR 2 / CW 4 / math 3 / humor 3.
Loop (per criterion, Sonnet agent, <=6 reflective rounds): sees improver pack, ops
library source, current h0/h1 + per-item TRAIN residuals; may restructure freely and
INVENT new pure-python ops (sandboxed per-criterion extension, smoke-tested); field
budget unchanged (<=2 LLM fields) so the comparison isolates pipeline flexibility.
Guardrails: test split never exposed (dpid contract); final candidate frozen, then
certified with the SAME gate machinery on held-out; report train-test gap (winner's-
curse diagnostic) alongside delta gate-pass and delta r-tilde vs h0/h1.
Success metric: how many points of the code-sufficiency % does flexibility buy?
If ~0: field-dominated graduates from pipeline artifact to domain fact.

## 9. GEPA-H2H — certified hybrid vs GEPA-optimized prompt (spec, approved 2026-07-06)

Per criterion (12: 6 PR across sufficiency classes + 6 CW/humor taste pole):
  arm H: frozen certified hybrid (code + <=2 Gemma fields) — existing.
  arm G: GEPA loop (glm proposer, SMALL dev set + few rounds per quota memory;
         GEPA_CORPUS env MUST be set per corpus) optimizing a single scoring prompt
         for the SAME executor (Gemma-31B, one call) against TRAIN judge verdicts
         (own-verdict reconstruction — label-clean).
Readout: held-out r-tilde (ceiling-normalized, same certificates) per arm; headline =
median delta r-tilde and per-class breakdown (does GEPA close the gap only on
field-dominated criteria?). Matched inference: 1 Gemma call (G) vs code + 2 calls (H) —
report cost column alongside.

## 10. Daston framing — what "thick" and "thin" mean here (2026-07-06)

The study is an operationalization of Daston (*Rules*, 2022): thin rules =
context-free, mechanically executable algorithms; thick rules = exemplar-laden,
discretion-requiring, presupposing a shared practice. Historically, thinning
succeeded only where the world was first standardized. Our compiler is a thinning
machine; the battery measures where thinning stops and why.

Component -> Daston concept -> result:

| Component | Daston concept | Result |
|---|---|---|
| code-sufficiency census | thinness needs a standardized world | PR 60% code-sufficient -> humor 8%; gradient tracks genre bureaucratization |
| C1 SCRAPE-REPAIR | "island of stability" precondition | normalization ops are world-stabilization made visible in code |
| C6 EXEMPLAR-MATCH | rule-as-paradigm (pre-modern form) | humor-ONLY (.42): where thinning fails hardest, code regresses to the exemplar form |
| E1 name-vs-nonce | unfactorability of thick concepts (Williams) | frac_name − frac_nonce = name-minus-definition gap; PR name-dependent (nonce ~.58), math definition-dependent (name .74 < nonce .96), CW both ~.98 (fm_full .35–.67, well-identified) |
| E2-STIP + toggle | algorithm-following vs discretion | executor-MODE property: thinking-on complies .97–1.0 w/ deviant rule, off snaps to community meaning .37–.59, same weights; stalls 24% vs 9% under conflict [1 task, 5–6 fields — suggestive] |
| E2-KIND (in flight) | deficit vs semantic gravity | cells 5/6 = pure execution control; gravity_effect = exec6 − comply4 |
| E7 provenance grid | thick/thin is TWO axes, not one | distillability ⊥ transport (ρ=−.011): codifiability and community-anchoring are independent; OVERFIT-SURFACE = thin-but-unshared, ENCULTURATED = shared-but-unthinnable |
| E3/E4 | where enculturation lives | fm monotone in scale; base ≈ 0–half of instruct: thickness capacity substantially from the tuned layer |
| lexicalization census | communities lexicalize what they regulate | coined terms: humor 0/31, PR 1/20 vs math 7/35 |
| AGENTIC-COMPILE (§8) | is the boundary domain-fact or compiler-artifact | pending: can a reflective compiler move the 44% field-dominated share? |

Key refinement to Daston: her thick/thin axis conflates explicitness with
stability. The provenance grid splits them — a judgment can be surface-mimicable
(thin ex post) without carrying the community norm (no transport). Mimicking the
rule-follower's behavior ≠ possessing the rule.

Williams link (thick ethical concepts): E1's nonce+definition condition is an
attempted FACTORING of a thick concept into descriptive content + arbitrary
label; 1 − frac_nonce measures unfactorability. Thick-predicate tags (KIND,
TONE-REGISTER, CRAFT) are the fused descriptive+evaluative predicates.

Caveats logged: math inversion also compatible with name-overload (definition
disambiguates an ambiguous term), not only definitional completeness; toggle
inversion is single-task; thickness readouts are instrument-relative (double
dissociation: key-vs-spec = construct property, stipulation-override = executor
property) — consistent with the anthropological-framing rule that LMs are
instruments.

## 11. E8-ARTIC — pre-registration (2026-07-06 night, BEFORE any data)

Design (approved by user 2026-07-06): every LLM field across the 5 fleet tasks
(PR v2, CW, math, humor, legal) gets one articulation prompt — the field's own
instruction, no document, "articulate the working definition / decision rule you
apply" — run through the three extractor families (Gemma-31B, Llama-70B,
Qwen-122B toff). Convergence per field = cross-family embedding similarity,
background-normalized: conv(f) = mean pairwise cos among the 3 articulations of f
minus the mean cross-family cos of same-task DIFFERENT-field pairs (removes the
"all definitions sound alike" floor). Criterion-level conv = mean over the
criterion's fields.

Pre-registered predictions (T-RET):
- **P1**: Spearman(conv, mean(ratio_llama, ratio_qwen)) < 0 across criteria with
  E6 transport ratios — criteria whose field signal TRANSPORTS (low ratio) are
  the ones the families articulate convergently (shared culture ⇒ shared gloss).
- **P2**: the 11 both-swap degraders from E6 (headed by PR a87 humble-tone)
  have LOWER median conv than the remaining ratio-measured criteria —
  extractor-bound payload should articulate idiosyncratically.
- **P3** (weak/directional): conv correlates positively with fm — fields
  carrying real enculturated payload have something to articulate; fields with
  fm≈0 articulate generically. No sign commitment stronger than "report it".

Status: descriptive corroboration ONLY — no gate claims, no certificates keyed
to conv. Failure modes stated in advance: (i) articulation is cheap
verbal behavior and may not covary with use (cf. E1: names index, definitions
underdetermine); a null here does NOT weaken E1–E6; (ii) embedding similarity
may saturate on register; the background normalization + a rank-only readout is
the guard; (iii) legal has no transport ratios yet (R19 in flight) — legal enters
P1/P2 only after those land, stated now to avoid post-hoc inclusion choices.

### 11.1 E8-ARTIC — RESULTS (2026-07-06 late; battery/artic_eval.json)

Ran gemma-31B / Llama-70B / Qwen-122B-toff over the 259-field articulation prompt
(max_tokens lifted 48→256 so the discriminating clause isn't truncated; medians
58/80/67 words, 259/259 non-empty all families). conv(field) = mean pairwise BGE
cos of the 3 articulations minus same-task cross-field background. Backgrounds
~0.63–0.67 by task ("all field definitions sound alike" floor); conv medians ~0.20
sit cleanly above it, so same-field articulations are ~0.20 cos more similar than
different-field ones. Ratio semantics (verified in transport_eval_3fam.py):
ratio = td/fm = FRACTION OF FIELD SIGNAL LOST under family swap — LOW ratio =
transports/portable, HIGH ratio = extractor-bound.

| prediction | pre-reg | result | verdict |
|---|---|---|---|
| **P1** Spearman(conv, mean_ratio) | < 0 | **−0.284** (n=101, ~2.8 SE from 0) | **CONFIRMED** |
| **P2** median conv: degraders < rest | degraders lower | .206 vs .200 (n_deg=11) | **not confirmed** (≈ equal) |
| **P3** Spearman(conv, fm) | weak/directional, no sign committed | **+0.024** (n=118) | **null** |

Reading (descriptive corroboration only, as pre-registered — no gates):
1. P1 is the powered test and it holds: criteria whose field signal TRANSPORTS
   (low loss under swap = portable enculturated meaning) are articulated
   convergently across families; extractor-BOUND criteria are articulated
   divergently. Shared payload ⇒ shared gloss — the retrieval thesis's
   behavioral prediction, now on the articulation channel and not just the
   extraction channel.
2. ★ DISSOCIATION: convergence tracks PORTABILITY (P1, −.28) but NOT field
   MAGNITUDE (P3, ~0). How much a field CONTRIBUTES is orthogonal to how alike
   the families VERBALIZE it. A strong idiosyncratic field and a weak shared
   field can carry the same fm; only the shared one is glossed convergently.
   This is exactly the split E1/E7 want: culture-overlap ≠ signal strength.
3. P2 fails: the 11 extreme both-swap degraders don't sit below the rest. The
   effect is continuous-graded (P1), not concentrated in the worst-bound tail —
   the significance-thresholded extreme subset is underpowered (n=11) and a
   P_degrade≥.95 cut selects on significance, not effect size. Report the
   continuous correlation, not the tail contrast.
4. Honesty caveats (stated in the pre-reg): articulation is cheap verbal
   behavior and need not covary with USE — a positive P1 corroborates but does
   not prove the mechanism; it does NOT strengthen E1–E6, which carry the
   causal weight. legal is not yet in the ratio join (R19 transport ratios
   pending); its 39 fields entered conv but not P1/P2 — will re-run P1/P2 with
   legal once R19 lands (pre-committed, no post-hoc inclusion choice).

**§11.1 update — legal folded in (R19, 2026-07-07):** after building legal
transport ratios (legal_transport.py) and adding legal to RATIO_FILES, the
pre-committed re-run over all 5 tasks: **P1 = −0.244 (n=118)** — still confirmed,
mildly attenuated from the 4-task −0.284; P2 still null (.206 vs .200, n_deg=13);
P3 still null (+0.030, n=137). Legal's 17 ratio-measured criteria don't disturb the
portability-not-magnitude dissociation. E8 verdict stands on the full fleet.
