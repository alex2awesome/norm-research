# BEST-PRACTICES.md

Consolidated operating rules for every experiment in this repo, distilled from ~2 months of
threads and the memory bank (2026-07-09). Each rule carries a one-line *why* and a provenance
pointer — either a memory slug (files in the Claude memory dir, `feedback_*.md` / `project_*.md`)
or a repo note. **New experiments should be checked against the relevant sections before launch;
when an experiment deliberately deviates (sometimes correct — see the frozen-articulation example
in [prompt optimization]), the deviation gets stated in its note.**

---

## [metric prompting]

- **Constancy is a first-class failure mode: detect it everywhere a model scores texts.** A
  scorer that answers the same thing for every item is evidence-free — worse, under naive
  readouts it can *look* like signal (ρ≈1 leniency fixed point) or like a real null (chance AUC).
  Every scoring run prints, per metric: mean/std, value counts, fraction-at-minimum, yes-rate.
  Warn/reject at std<~0.4 or frac-min>~0.6. *(feedback_check_judge_score_distribution;
  metric_implementer measures.py discrimination gate; zxa_fit degeneracy flags + 2026-07-09
  cell-level collapse-sensitivity.)*
- **Known constancy attractors, by mechanism:**
  - *Structured/guided decoding collapse* — schema-forced JSON can silently emit all-min with a
    100% parse rate; prompt×schema-specific and unpredictable. Fix: free-form + parse+validate+
    retry, not `StructuredOutputsParams`. *(feedback_check_judge_score_distribution.)*
  - *Instruction-load collapse* — small models (≲3-8B) go constant as the instruction text grows;
    dossier-length prompts collapse them MORE than length-matched inert filler, and each model
    has a default polarity (llama1b→all-YES, qwen25-3b→all-NO). Long rubrics on small executors
    need a collapse check before any curve is interpreted. *(z×a degeneracy audit 2026-07-09,
    notes/2026-07-08__zxa spec 21:00 section.)*
  - *Empty-by-default over-learning* — stacking "expect sparse / most items return nothing"
    phrasing makes instruction-tuned models drop true positives. State the null option once,
    neutrally. *(feedback_prompt_empty_by_default_backfires.)*
  - *Strict-bar reading of bare names* — a metric presented as just its NAME gets a much stricter
    threshold than the same metric with a definition (peer-review frontier yes-rates .01-.13 at
    name arm). Arm/prompt depth changes the decision boundary, not just knowledge. *(z×a audit.)*
- **Show the text before the question** (text-first template) and keep the judged criterion
  verbatim when articulation itself is the measured object (`--n-forms 1`). *(osl_sweep
  `_YESNO_TEXTFIRST`; z×a spec.)*
- **Task-correct system prompts** — a judge told it is "scoring scientific papers" while grading
  patents produces near-zero discrimination. Check the system prompt matches the task for every
  new corpus. *(feedback_judge_prompt_cross_task_bug.)*
- **No repetition/frequency penalties to fix bad output** — they distort the output distribution
  (frequency_penalty 0.3 dropped signal density 18→6). Detect bad output, retry with a different
  seed. *(feedback_no_repetition_penalty_retry_instead.)*
- **New task ⇒ task-specific few-shots** demonstrating what a *good evaluative feature* looks like
  for THIS task, else the model emits ~50 generic themes. *(feedback_local_explanations_per_task_fewshots.)*
- **Mechanical/planted rules must appear VERBATIM in any authored articulation of them**, and the
  guidance must match the *coded truth function* exactly — an author "reasonably interpreting" a
  rule (apostrophes as quotation marks, 2026-07-09 news planted-quote) silently corrupts the
  planted gate. Validate authored arms against the truth implementation, not the rule's prose.

## [metric inference / discovery]

The pipeline that INFERS which evaluative metrics a community uses (metric-tree / residual infilling
through a shared MCC gate) and whether new ones must be invented. Distinct from scoring a *fixed*
metric ([metric prompting]) — here the metric itself is the unknown.

- **Two gates, and only the second is evidence.** Stage-1 is an in-run gate (paired-CV bits/AUC on
  the same items the proposer saw) — suggestive, overfits its split, NEVER a finding on its own. A
  metric counts only after strictly-disjoint stage-2 replication: re-score on held-out items never
  touched during proposal or stage-1, Bonferroni over the replicated set. "Proposed, and even
  recurring across slices" ≠ confirmed. *(project_subcommunity_heterogeneity two-stage gate; task #68.)*
- **Discover WITHIN the subtask; pooling dilutes twice.** A slice-local metric worth *b* bits at
  population share *s* reads ~*b·s* pooled — diluting both the proposer's chance of surfacing it AND
  the gate's power to detect it. The general/pooled leg keeps ~0 metrics in every domain tried (4/4);
  the general bank suffices for the general task. Run the arms within each community, never the union.
  *(project_sibling_metric_lattice dilution; cousin of feedback_same_family_scaling.)*
- **Class-balance each slice 50/50 before discovery.** Raw community base-rates are often pathological
  (ICML-accept .979 → ~19 negatives per 900 items = zero discrimination power that masquerades as
  "this slice wants no new metrics"). Balancing isolates the within-slice discrimination question and
  makes slices comparable; auto-drop a slice whose minority class < ~300 rather than run it degenerate.
  Disclose the mixed treatment when earlier slices used natural rates. *(scale-out wave, 2026-07-09.)*
- **NEVER balance a heterogeneous POOLED/control leg overall — balance per-slice then pool, or pool at
  natural rates.** Balancing a union of slices with divergent base-rates 50/50 as one pool MANUFACTURES
  a slice-identity→label correlation (the negatives come from low-accept slices, positives from high-
  accept ones), and the proposer then "discovers" a slice-identity PROXY that reads as a real metric.
  Diagnostic: measure P(label | slice) inside the pooled file — spread >~.2 means the confound is
  planted. Proof a pooled keep is a proxy and not a criterion: it collapses when re-run WITHIN a single
  slice where it cannot proxy identity. Real case: overall-balanced peer×venue pool (neurips .91 /
  iclr .28 accept, spread .63) "kept" a CS-Artifact-Release metric at +.040 bits that is worth ~0
  within every individual venue — a venue detector, not quality. Spurious pooled-invention scales with
  the manufactured spread (.63→keep, .20→hot tails, ~.03→none). *(project_sibling_metric_lattice, 2026-07-10.)*
- **The proposer is label-free; the gate owns every label touch.** Proposals come from TEXT patterns
  under a content-only guard (reject anything keying on length / formatting / metadata — that is the
  cross-source leak, not a metric); labels enter only at the gate. *(feedback_reconstruction_only_no_labels;
  see [labels & leakage].)*
- **A proposal must beat the EXISTING bank, not just chance.** Residual bank-AUC gate: accept only if
  it adds signal *beyond* the current bank (else you rediscover bank metrics reworded). Pair with a
  reliability / test-retest check (an unreliable judge-metric is not a discovery) and a confirm stage
  (fresh-seed CV + Nadeau-Bengio) before anything is even called a stage-1 hot tail. *(content guard /
  reliability / residual bank-AUC, task #57.)*
- **Build the bank as coverage-selected medoids of the rubric pool**, not first-N: gather all
  extracted rubric metrics, dedup by name, embed, cluster (KMeans k≈40), keep each cluster's medoid →
  one fixed, comparable bank per task, used for BOTH the discovery run and the liveness re-score.
  *(task #51; reference_metric_banks; {code-review,notice,press}/medoid-bank-auto 2026-07-09.)*
- **Dedup discoveries by embedding at two grains** — dup (cos>.86, same metric reworded) vs theme
  (cos>.55, same idea family). A single latent metric surfaces reworded across runs; count FAMILIES,
  not strings. Small per-slice n makes "0 shared" partly power-limited — add a permissive name-only
  cross-check before concluding metrics don't transfer across siblings. *(project_sibling_metric_lattice.)*
- **Separate the two senses of "in common."** Shared VOCABULARY = a bank metric carries within-slice
  AUC≥τ in ≥2 slices (bank-liveness re-score). Shared NOVELTY = an *invented* metric recurs across
  slices (embedding dedup). They answer different questions — report both, never conflate.
  *(project_sibling_metric_lattice.)*
- **Hold the confounder fixed when stratifying.** Venue is a base-rate confound → topic-model WITHIN a
  venue to isolate subfield; never stratify along the confounder itself. Whether invented metrics
  transfer is itself a finding, not an assumption: they are slice-LOCAL in math/humor/CW but RECUR
  across peer-review subfields (a shared theoretical-rigor norm). *(project_sibling_metric_lattice.)*
- **Ground truth is the ledger, never the process rc.** Harvest hot tails from the infill ledger
  (`{leg}/{arm}/global_infill_ledger.json`, key `ledgers`); the `echo "[$(date)] rc=$?"` idiom always
  prints 0 because `$(date)` resets `$?` — capture `RC=$?` on its own line. Stage-2 pool-exhaustion is
  status "pool-exhausted" (data limit ≠ method failure); wrap each candidate in try/except so one bad
  one doesn't take the leg down. *(replicate_candidates guards; 2026-07-09.)*
- **Evaluate validity, never gate on it.** A discovered metric is scored for validity, but validity is
  never itself an acceptance gate inside the loop (that would make the metric label-aware). Report the
  recovery metric I(M_ω;·) with its discrimination floor alongside. *(project_metric_implementer
  validity scorecard; feedback_report_recovery_metric_only.)*

## [metric seams]

Determining WHERE a metric's implementation stops being code and starts borrowing judgment
(prompt↔code↔workflow placement, seam depth, certificates). Distilled from the metric_seam
thread (pilot → kill-switch → 8-task frontier → CODA/PREDICTABILITY → expansion, 2026-07-01→08).
Method dir `methods/metric_seam/`; doc-of-record `notes/2026-07-01__metric-seam-pilot-results.md`.

- **Placement is ONE-SIDED decidable — never declare "uncodeable", declare search saturation.**
  A working code witness certifies codability positively; the negative is only ever "N improver
  rounds at budget B didn't certify." Calibrate the saturation claim with plants: the same
  protocol closed 86-90% of ceiling on planted-codable criteria, so real criteria that don't
  move are genuine judgment-layer, not search shortfall. *(proposal §3.4; kill-switch claim vii.)*
- **Run a planted kill-switch BEFORE trusting any seam pipeline.** Plants spanning the op
  taxonomy (codable / evidence-op / null / oracle-at-known-ceiling); demand zero false certs,
  conservative misses, and the oracle landing on the predicted ceiling (ours: 0.999). Keep
  plants blinded from the builder agents — one breach forced a clean-room refleet.
  *(killswitch/DESIGN.md; §E-S1.)*
- **Never trust a single-round codegen verdict — the boundary is partly compiler effort.**
  Agentic round-2 (6 rounds, op invention) reversed round-1's "domain fact" (2 net-new gates;
  legal a0 .69→.75 pure-code). And upgrading the compiler beats rewording the criterion
  (rewrite arm +.058 median vs compiler-version 12/18 sig) — spend budget on rounds/ops, not
  rephrasing. *(§EXPANSION arms 1-2.)*
- **Held-out gates always; train-side reflective refinement is a winner's curse.** 12/12 train
  cells improved while test collapsed .709→.579 (round-1); in the agentic fleet every
  calibration-type train gain evaporated or inverted while objective-bug fixes transferred.
  Field-redesign pattern: winners ADD a discriminating second field, losers REPLACE the
  construct (h1 fleet 0/10 promotions). *(v1 hybrids; §8 AGENTIC-COMPILE; §R4.)*
- **Ceiling-normalize before comparing anything across criteria or tasks.** 2-pass judge
  reliability sets an attenuation ceiling; report r̃ = clip[0,1](ρ/ceiling) and CAM = mean r̃ —
  raw ρ confounds codability with judge noise (ceiling-norming REORDERED the fidelity-cost
  gradient: math's low ceilings had masked its cost). Watch the converse artifact: an absolute
  gate bar (.60) can EXCEED a low-rel criterion's ceiling — flag, don't fail. *(cam_profile.py;
  §CAM; §R20; §R1 artifact.)*
- **Apples-to-apples inputs for every arm** — the v0 pilot bug was code seeing full text while
  the judge saw 8k-truncated. And when the input representation changes, EVERYTHING re-baselines:
  adding diffs rescued judge degeneracy (20/40→8/40) while dropping the code baseline (OOD
  programs) — "evidence-starved" is a (criterion, executor, representation)-triple property,
  never a task property. *(v1 fix; CR re-survey.)*
- **Scope the channel before reading the seam.** Only 105/250 "press releases" were real
  releases; scoping deconfounds (structural criteria rise, soft ones fall). Scoped gates need
  the 4-condition rule: criterion-independent predicate, frozen, symmetric, stamped with
  coverage. *(v1 scoping; Day-3 sign-off.)*
- **Evidence ops ≠ computation ops — cite the right ceiling.** Evidence ops (retrieval, world
  state) genuinely move the channel ceiling I(M;X)→I(M;X,Z); computation ops (Z=f(X)) add zero
  Shannon info by DPI — they widen the executor rung. Decidable per channel via the
  stronger-executor-without-tool test. *(op taxonomy — the recurring-confusion risk.)*
- **Level-match ops to the judge's evidence or the op-marginal is ill-posed.** X-orthogonal
  evidence CANNOT reconstruct a doc-only judge (patents_pa forced null, I(M̄(X);Z|X)=0) — an
  evidence op only helps against an evidence-aware target. Also: over-strict verbatim-grounding
  gates disable borrowed meaning entirely (fields inert, fm≈0 — a design null, not a finding).
  *(§R7.1; binding-provenance note.)*
- **Seam depth is graded — measure it with a budget ladder, not a binary.** Borrowed-field
  budgets 0/1/2/4: the first field dominates, but 2→4 lifts field-dominated criteria; a ≤2-field
  contract silently truncates enculturated criteria. Report min-fields-to-95%-of-max as depth.
  *(§EXPANSION arm 3.)*
- **Certificates carry a 4-tuple stamp: (criterion, judge-family, executor level,
  field-extractor family) + transport ratio.** Gates are judge-dependent (a110: Gemma P=.989
  certified, Llama P=0 on the same frozen hybrid) and extractor-dependent (taste-pole certs
  doubly bound; the two axes are INDEPENDENT at criterion level — legal a44 is field-swap-robust
  but judge-swap-fragile). Single-family certs are quotable only at the doctrine pole.
  *(§TRANSPORT; §CROSSFAM; §JUDGE-REP-FLEET.)*
- **Don't expect to predict the seam from phrasing — run the pipeline.** CODA verdict: within
  community, the seam is NOT phrasing-legible (pooled ~0-.12 after the rank-pooling fix). The
  surviving structure is a weak double dissociation — zero-shot holistic guess → binary floor
  only; decomposed thickness features (F7 world-knowledge, F4 reader-effect) → depth only.
  Phrasing predicts the compiler's STRATEGY, but the world decides whether surface correlates
  exist. *(§CODA; §PREDICTABILITY-2 CORRECTION.)*
- **External anchors: never pool across strata.** The competition-verdict anchor was maximally
  entangled with language (all 62 C++ = AC) — the headline "clean formatting anti-correlates
  with correctness −.437" died within-Python (+.04). Anchor readouts need verdict-balanced
  within-stratum resamples. *(§SKEPTICAL AUDIT — the Simpson trap, again.)*
- **Codegen hygiene is part of the instrument.** Sweep for the substring bug class
  (`punch`/`punchline` — \b-anchor with inflection whitelists; detect_substring_bugs.py), catch
  silent NameErrors that yield constant outputs (Spearman NaN ≠ tacit), and verify "degenerate
  baseline" cases individually (a288: 3/250 docs genuinely have titles). Constancy checks apply
  to every program, not just judges. *(§HYGIENE; CW fleet a324/a288.)*
- **Prompt-rung fidelity is the ceiling, and that's fine — hybrids sell a different axis.** A
  single tuned judge-prompt sits at judge-noise ceiling and beats certified hybrids on fidelity
  ~11/12 (median gap .19ρ = the quantified compression cost); hybrids buy typedness,
  determinism, auditability, near-zero marginal cost. Never headline hybrids on fidelity; never
  run prompt-optimization without gates (one GEPA rewrite collapsed a criterion to constant-0).
  *(§9 GEPA-H2H; §R20 waterfall.)*
- **Sanity anchors for new banks:** bottom-up author units land on the SAME seam as top-down
  rubrics (humor_units CAM .122→.352 ≈ curated .120→.351 — construct validity for the unit
  choice); and naive type predictions invert (codable "TASTE" = content-rating surface markers,
  the craft core stays uncodified) — read type-level orderings only with per-type n in hand.
  *(§EXPANSION arm 5, MECH n=2 caveat.)*

## [judging & eval hygiene]

- **Judges are Sonnet-class or better** (GLM-5.x acceptable, quota permitting) for any generative
  scoring/filtering/ranking. Embedding nets for candidate *generation* are fine; the floor applies
  to judgment. Local 70B screens: retired. *(feedback_judges_sonnet_or_better.)*
- **Blinded known-label anchors ride in EVERY judging batch** — including resumed/continued agent
  passes, which are the documented failure case (a resumed judge scored 88/2/0 on a 44/90-SAME
  anchor set while its original pass was fine). Batches are severity-relative; anchors are the
  cross-batch ruler. *(feedback_anchor_test_annotation_passes.)*
- **Instrument certification carries only if the instrument is identical** — same model, same
  prompt, same validator. An anchors-8/8 run certifies an earlier anchor-less run only under
  identity. *(census memory, CW-certifies-humor precedent.)*
- **Tiebreaking between two judges needs a frontier arbiter told it IS the arbiter** ("your ruling
  sets ground truth"), from an independent family where family-neutrality matters. Never assume
  the expensive judge is right when validating the cheap one — that's circular.
  *(feedback_arbiter_large_model_with_context.)*
- **Two-family verification for anything called a certificate**: builder family ≠ measuring
  family. GLM measured over-lenient on R1 verify (46% SAME → precision .78→.60) — GLM builds are
  fine where *measured against independent truth*, GLM verify of semantic sameness is excluded.
  *(project_metric_lexicon_census sessions 4-5.)*
- **Degenerate-target filter for certificates**: a metric whose executor readout is near-constant
  (H_M < 0.15 bits) is vacuously certifiable — exclude before quoting grids.
  *(feedback_certificate_audit_disciplines D1.)*
- **The reference executor never scores itself** in decompression/agreement grids (self-consistency
  ceiling ≠ data). *(feedback_certificate_audit_disciplines D2.)*

## [prompt optimization / GEPA]

- **Fitness must include a discrimination gate** — constant-scoring prompts are non-promotable
  regardless of apparent fidelity (the leniency fixed point). *(metric_implementer config.py:111,
  measures.py:415/484.)*
- **Acceptance must be coverage-honest**: unparseable outputs count as WRONG and coverage prints
  every eval, else a prompt that answers 8% of items "wins" on the subset it deigns to answer
  (patents GEPA v1 retraction). *(project_patents_prior_art_pipeline; feedback_check_judge_
  score_distribution parse-coverage variant.)*
- **Optimize the scaffold, never the measured object.** When articulation depth is the
  experimental variable (z×a arms), the arm text is FROZEN and GEPA-style optimization of it
  would change the estimand. The YES/NO wrapper, retry policy, and parse layer are fair game;
  the rubric string is not. State this boundary in any prompt-opt proposal. *(z×a spec design.)*
- **GEPA → full-corpus deploy is one pipeline**, not two decisions; deploy at the yield-optimal
  operating point when anchors are the goal. *(feedback_gepa_then_full_deploy.)*
- **Set `GEPA_CORPUS`** or gepa_pr silently runs press_releases. *(reference_gepa_corpus_env_trap.)*

## [probe & dataset design]

- **Dataset-first protocol, in order:** spot-check samples → TF-IDF/LR with top-features read →
  metadata confound gauntlet (era/venue/length/source) → only then judges or dense models. The
  lexical step is a diagnostic, not a ladder rung. *(feedback_dataset_first_protocol.)*
- **Validate any rule/regex on 5-10 eyeballed input→output samples before running at scale.**
  *(feedback_validate_before_scaling.)*
- **Construct-support match (2026-07-09, the probe-slice lesson):** a metric can only be measured
  on texts where it is *applicable*. "Host presence" on 81-char one-liners → frontier all-NO
  unanimity .99 with κ≈0 — a base-rate artifact that mimics "fully tacit". Before classifying a
  metric as tacit/underdetermined, verify it FIRES with non-trivial base rate on the probe slice;
  fix is longer/stratified probes, not a verdict. *(z×a subtask_kappa decomposition.)*
- **Probe slices must span the metric's decision boundary** — homogeneous slices saturate even
  frontier models into constant rows (peer-review 30-50% const at 70B+). More probes / variance-
  targeted windows are the fix; a metric constant-on-slice is unmeasured, not articulable/null.
- **Cross-task comparisons need matched support**: humor probes 81 chars vs CW 3,158 (36%
  truncated at the 4,000-char cap) is not an isomorphism-ready design; match length regimes or
  qualify the claim. Truncation truth-rule: compute mechanical truths on the SHOWN slice
  `text[:max_text_chars]`. *(z×a spec 21:00; zxa_fit planted_truth.)*
- **Junk/language filters are documented, mechanical, and inspected** (news probes: headline-slot
  gates + English stopword gate, 38% pass, samples eyeballed before use). Filters are part of the
  freeze provenance.
- **Stable-hash splits, salted, fine modulus** — `md5("split::"+id) % 1000`; never seeded-shuffle
  a growable list (silent reassignment contaminated patents v6a, 2× inflation); unsalted mod-10
  can be degenerate on structured ids (a 70σ empty bucket). *(feedback_stable_hash_splits.)*
- **Cross-source formatting leak**: positives and negatives from different scrape pipelines encode
  the pipeline (typographic quotes → fake .996 AUC). Normalize presentation on both sides; treat
  any near-perfect probe AUC as a leak alarm. *(feedback_cross_source_formatting_leak.)*
- **Never `gzip.open("at")` in a kill-restartable writer** (deflate window corruption ate 187K
  records); plain JSONL parts + gzip-at-rest. *(feedback_no_long_running_gzip_append.)*
- **Known-bad corpora stay out**: New Yorker caption ratings (crowd-worker, not taste);
  news-homepages label = homepage SPATIAL LAYOUT (not clicks) with ~40% junk headlines — filter
  before probe use. *(feedback_no_newyorker_captions; news_homepages memories.)*

## [labels & leakage]

- **Metrics are reconstruction-aware, never label-aware; no human subjects.** The estimand is
  I(M̄_E;·) recovery — task labels in the loop change the estimand and import confounds.
  *(feedback_reconstruction_only_no_labels — the standing rule of the whole program.)*
- **Never show the judge the label** (or any outcome-adjacent field): `{id, text}` only; labels
  join at analysis time. Any judge AUC >~0.85 is presumed leakage until checked.
  *(feedback_never_leak_label_to_judge.)*
- **Similarity-to-the-reference is a label, not a feature.** If the outcome is defined relative
  to a reference object, features computed against that reference are inadmissible. (Flagged 4×.)
  *(feedback_editorial_sim_is_a_label; feedback_no_similarity_to_reference_as_predictor.)*
- **Apples-to-apples or no claim:** dense-vs-baseline requires the identical eval rows AND the
  identical input text; two stacked split/input mismatches faked a +0.10 "tacit band" that was
  really +0.0005. *(feedback_apples_to_apples_dense_vs_baseline.)*
- **All three label types per task** (expert-verdict / expert-revealed / community-revealed);
  external validity is only expected where the silver label measures judged artifact quality —
  it vanishes on popularity and inverts on gatekeeping outcomes. *(feedback_all_three_label_types;
  bounded-audit label-type taxonomy note.)*

## [silver-labeling]

Extracting community normative signals ("silver norms") from documents and matching each to the
top-k metrics in a task's R2 bank — the bge_pertask pipeline that produces the norm→metric
assignment matrix A[item,metric]. Distinct from [metric inference / discovery] (which INFERS new
metrics from text patterns); here the metric bank is fixed and the question is which apply to
each norm. Doc of record: `methods/metric_implementer/norm_to_metric_matching.md`. Layout:
`data/bge_pertask/<task>/` — `signals_<task>.jsonl`, `matches_ce_<task>.jsonl`,
`matches_joined_<task>.jsonl`, `catalog.txt`.

- **Two layers, two model choices: extraction/labels = GEPA+Gemma; matching = BGE→Llama-8B CE.**
  Stage A extracts silver norms with **Gemma-4** (GLM-5.2 runs GEPA prompt optimization); Stage B
  generates CE training triplets with **Gemma-4**; Stages C–D are base-BGE top-50 → trained
  **Llama-3.1-8B CE** top-10. A Qwen-122B LLM-rerank stage was **dropped** (no lift). Never say
  "the silver labels are from X" without naming the stage. *(reference_bge_pertask_cascade_pipeline.)*
- **Qwen-122B generates, never evaluates.** Qwen is fine for bulk label/triplet *generation* and
  candidate extraction; it is NOT trusted for faithfulness/validity *eval*. Audit extractions with
  GLM-5.2, Claude Opus, or Codex. *(feedback_qwen_not_for_eval.)*
- **Set `GEPA_CORPUS=<corpus>` or every corpus silently runs as press_releases** (0-signal
  cross-contamination); the `--mode`/positional arg does NOT select corpus.
  *(reference_gepa_corpus_env_trap; [prompt optimization / GEPA].)*
- **`signal_id` is a DOCUMENT id, not a per-norm key — the join is POSITIONAL, never by key.**
  One doc yields many norms sharing one signal_id (peer_review doc "2" → 10 norms). A key join is
  ambiguous in 20/26 corpora; it coincides with positional only on the 6 unique-i corpora (5 gold
  tasks + humor_multi). Join `matches_ce` row N ↔ the Nth retained signal. *(2026-07-10 silver-
  provenance audit; reference_bge_pertask_cascade_pipeline.)*
- **Emit a stable per-norm uid AND record the enumeration recipe, or provenance is lost.** The
  `signals_<task>.jsonl` writer was an ad-hoc/notebook step with NO recorded script — so `i`'s
  source corpus became unrecoverable (gold-task signals don't match the current
  `data/<task>/input.jsonl`: ~3.5% verbatim overlap for code_review). Always write down:
  (a) the script that wrote signals, (b) the input file + version/date it enumerated, (c) any
  filter/dedup/shuffle + seed (run_sk3.py uses `df.sample(frac=1, random_state=42)` → filter
  len>50 → reset_index — that's the shuffle that breaks raw-row-order matches), and emit
  `{i, row=<signals-line-index>, ...}` so the join is unambiguous and byte-reproducible.
  *(2026-07-10 provenance-loss finding.)*
- **Mind the `i==0` falsy-drop.** `if sid and stxt:` silently drops every signal whose id is the
  integer 0 (1 norm/corpus in the 5 gold tasks; `mc = sig − 1`). Use `if sid is not None and stxt:`.
  An off-by-one in the joined artifact = the dropped `i==0` row at the *front*, not a missing last
  line. *(match_cascade.py patch 2026-07-10.)*
- **Keep extraction records self-contained: carry `unit_id` AND the source item text (or a join
  key to the input corpus) in each record.** A record with only `unit_id`+`passages` forces a
  second-file join to recover item text; one with only `post_title` is unjoinable. Emit enough to
  rebuild `items_<task>.jsonl` without re-running extraction. *(gepa `deploy` vs Qwen schema gap,
  2026-07-10.)*
- **Snap extractions to source verbatim; audit faithfulness against FULL source, not a truncation.**
  signal_text/passage_text must be exact substrings of the source (`snap_to_source`). A 2k-char
  truncation in the auditor falsely flagged long legal corpora as ungrounded — re-audit at full
  length landed 0 fabricated across 23 corpora. *(GLM-5.2 + Codex dual-judge faithfulness audit,
  2026-06-30; feedback_no_handwaving.)*
- **Codex fabricates when data-unreachable — verify file writes before trusting its output.** Run
  Codex only with the data bundled locally; check record counts and unit_ids against the input
  before believing any verdict. *(Codex rescue audit 2026-06-30.)*
- **CE is the workhorse on objective tasks and plateaus on taste — and the taste plateau is a
  CATALOG artifact, not a matching failure.** Recall@10: math .55, code_review .44, humor .39,
  PR .39 vs **creative_writing .23** (only +21% over base-BGE). CW's ceiling is R2 over-split
  near-duplicate leaves + noisy subjective gold (CE top-10 67% sensible, 7% true misses). Do NOT
  chase CW with more labels (2× data → .230→.232, flat). Fix = re-cluster R2 (looser τ) or score
  at R3. *(project_cw_taste_matching_finding.)*
- **Label noise kills the CE faster than volume helps it.** wp_comments (≤25% sensible) had
  multi-positive triplets — the same anchor stamped positive for unrelated leaves (a spelling
  anchor → both "correctness of spelling" AND "setting integration") + reused negatives; the CE
  trains on the smear. Generate stricter, relevance-verified positives (top-1/2, not top-k) with
  distinct hard negatives per anchor. Weak-corpus cluster: crse 1/8, dol_arb 2/8, litbench 3/8.
  *(wp_comments label diagnosis 2026-07-10.)*
- **Train ONE CE per task, parallelized; score recall vs gold on the SAME split + SAME input as
  the baseline.** Per-task CEs (GPU-stacked, `train_ce_parallel.py`) beat a universal CE because
  metric banks and norm styles differ. A dense>baseline claim needs identical eval rows AND
  identical input text — two stacked mismatches once faked a +0.10 gap that was really +0.0005.
  *(feedback_apples_to_apples_dense_vs_baseline.)*
- **Two pipelines share task names — never conflate.** `data/bge_pertask/` (silver norm→metric
  matching, 26 corpora) vs `methods/metric_implementer/manifest.py` (E7 articulability scaling,
  n=60 seed=7 subsample). Same task strings, different corpora, different i-spaces. *(2026-07-10
  recon — manifest looked like the silver source but isn't.)*
- **The catalog is R2 merged leaves (`a{N}` in `catalog.txt`); roll up via
  `outputs/hierarchy/<task>_general_r3_expanded.json` (`source_r2_cluster_ids` are bare ints →
  prepend `"a"`).** Build the bank as coverage-selected medoids of the rubric pool, not first-N.
  *(reference_metric_banks; [metric inference / discovery] medoid rule.)*

## [scaling ladders & cross-model comparison]

- **Same-family ladders only; panels never pooled** (family offsets confound raw cross-family
  curves; MoE active-params aren't orderable against dense). OSL battery-z latent axis is the
  sanctioned cross-family exception. *(feedback_same_family_scaling.)*
- **Threshold-free readouts for any cross-model comparison** — AUC/rank stats, never bal_acc@0.5
  (a globally under-confident model reads as broken at 0.5; AUC recovered 21/21 measurable cells).
  Balanced agreement (mean of per-class accuracies) when a reference is binary — constant
  predictors score exactly .5. *(feedback_threshold_free_readouts; zxa_fit balanced().)*
- **A ladder must SPAN the criterion** to resolve a crossing (GLM's 1.63-1.99 bunch is left-
  censored for most metrics); report censoring status (`<=`, `=`, `>`), never silent point
  estimates. *(z×a spec.)*
- **Degenerate rungs stay in the ladder as the honest left edge, but every crossing gets a
  collapse-sensitivity check**: re-fit with constant-row cells masked; flag rows whose crossing
  status/z moves >0.15 (17/63 humor rows, 2026-07-09) as articulation-form-limited rather than
  knowledge-limited. *(zxa_fit collapse_sensitive, the GEPA discrimination gate propagated to
  the read side.)*
- **Non-monotone rungs break the params map, not the ordering** — report z*, not N*, above the
  inversion (qwen25-32b z=2.546 > 72b 2.338). Exclude batteries that fail sanity (gemma2 27b<9b
  inversion + noise-profile yes-rates) from ladder claims until investigated.
- **Verify model aliases on API ladders** (glm-4.5-air→4.7, glm-5→5.2) before treating rungs as
  distinct. *(OSL memory.)*
- **No naive scaling extrapolation** — V-info-style quantities are non-Lipschitz in scale;
  forecasts are labeled assumption-arithmetic (T*=20·N* Chinchilla), bounds clamp to censoring
  (ẑ* can't sit below a right-censored name bound). *(project_vinfo_pathologies_koyejo; zxa_fit
  clamp fix.)*
- **Mind the prompt-cache** when sweeping API ladders; and hybrid-thinking models can return
  EMPTY content at low max_tokens (reasoning eats the budget) — `OSL_REASONING_OFF=1`.
  *(reference_openrouter_hybrid_thinking_trap.)*

## [metric upper bounds]

How to measure the *ceiling* of a metric — articulability-curve asymptotes (L), certified
recovery ceilings (OPT_Ω), and frontier placements — without fooling yourself. Distilled from
the OSL multi-task fleet, the inverted-U/threshold thread, and the N&C corpus reframes
(2026-07-08/09).

- **Freeze the WHOLE instrument: the battery AND the crowd.** A consensus reference that drifts
  as new panels land manufactures artifacts — with API frontier panels admitted to the pool,
  kimi-k2.5 briefly measured as the MOST crowd-agreeing frontier model; against the frozen
  11-local-executor crowd it sits below the peak like every other frontier point. Consensus =
  frozen local mid-scale set, family-exclusion hygiene per scored executor, hard-binarized.
  *(qwen3_adjudicate.py LOCAL_MID; bounded-audit note 2026-07-09.)*
- **The bound's TYPE is the curve's shape, and one shape proves nothing:** bend at the planted
  ceiling = REACHES (positive articulability cert); bend below ceiling above floor = BOUNDED
  (tacit-residual candidate); bend at the floor = criterion underdetermination, not executor
  limitation; **no bend = RISING = a lower bound only — a rising curve can never prove
  inarticulability.** *(curve-shape table, osl-executor-scaling spec.)*
- **BOUNDED is x-axis-dependent until frontier points land.** Under declared battery-z, 24 humor
  metrics plateaued; under Beta-IRT's own fitted θ (which stretches the frontier), out-of-sample
  ceiling evidence vanished (2PL beat per-item ceilings LOEO on ALL classes). Either the battery
  compresses the top or IRT launders ceilings into difficulty — the decider is probe points that
  extend both axes. Never publish a BOUNDED verdict from a mid-scale-only ladder.
  *(osl_irt.py cross-check; arXiv:2606.07616.)*
- **Audit every BOUNDED before calling it tacit.** Pooled bends have two impostors: family
  dialects (which family sits where on z) and floor contact (underdetermination). Per metric:
  within-family shape on each ≥3-rung ladder + per-metric frontier floor + classify
  {DIALECT-SUSPECT, AT-FLOOR, CEILING-ADJACENT, TACIT-CANDIDATE}. Only 16/36 humor BOUNDED
  survived; the survivors cohere (voice/persona/embodiment). *(bounded_audit.py.)*
- **Bend claims need a smoothness audit at the right noise level.** Curve SNR against
  *probe-sampling* noise is high (median 20×), but iso-fit residuals run ~5.4× probe noise —
  systematic family dialects, not sampling error. Error bars for shape claims must be
  executor-level residuals; within-family curves are clean but short (4 pts).
  *(curve_smoothness.py.)*
- **On hard-readout API frontier points, compare ORDERINGS and within-executor CONTRASTS, never
  levels.** Absolute agreement carries a dialect discount. The dialect-robust ceiling statistic
  is the within-executor class contrast — e.g. TACIT−PLANTED deficit: −.03..−.05 at the peak
  zone, −.13..−.18 for every frontier point above it, across 4 independent families.
  *(frontier-divergence sections, bounded-audit note.)*
- **Falling agreement above the crowd's level is NOT degradation.** Crowd-agreement is
  non-monotone in capability (inverted-U): rising limb below the crowd's own competence (~.90
  planted truth-acc), then a sharp step down (glm-4.5/4.6 twins at deficit −.04 → glm-4.7/5.2
  at −.18; deltas flat/cliff/flat). Interpreting the falling limb as "the model got worse"
  inverts the finding — it grew past the reference exactly where language underdetermines.
  Anchor every frontier point with planted truth-acc so the two limbs are separable.
  *(4-rung within-GLM ladder, notebook §7/§7b.)*
- **Say WHICH bound.** L = fitted asymptote of the articulability curve (report with CI; at ≤12
  rungs the CIs are wide — the shape VERDICT is the robust readout, L point estimates are not).
  OPT_Ω = certified best-subset recovery ceiling (label-free); g1/OPT_Ω lower-bounds
  prompt-vs-ceiling; T = I(M_ω;X) lower-bounds the ideal metric; B_E upper-bounds. Planted
  ceiling = top-3-executor mean on mechanical controls, per task — REACHES/BOUNDED splits are
  meaningless for a task whose planted curves haven't landed. *(laws fits; gepa_vs_ceiling.py;
  feedback_T_lower_bound_Mstar_be_upper.)*
- **Recovery readout = MCQ by default; free reconstruction is a DIAGNOSTIC only.** The reported R2
  recovery `C(R(Ω)) = I(M_ω;M')` uses `--mode mcq` (reconstructor PICKS the metric from a k-candidate
  pool; X re-executes the pick), with `--distractor hard` (behaviorally-nearest, clone-excluded — the
  honest difficulty; `random` is the easy-ceiling baseline and inflates R, never the headline). MCQ is
  bounded `[0, H]`, has a clean `1/k` chance baseline, and is prior-robust. **Free reconstruction**
  (`--mode free`: reconstructor WRITES a rubric M̂) is kept only for the readable guess — *what did it
  find?* — never for the reported number: it confounds "metric inarticulable" with "the reconstructor
  wrote a bad rubric this run," collapses to a generic-quality prior on sparse labels, and can go
  **negative** (a fluent-but-wrong rubric anti-correlates with M_ω). Both are LOWER bounds on
  articulability; the certified upper bound stays `T_soft` (the DPI cap), never a recovery number.
  *(run_r2_recovery.py `--mode` default = mcq since 2026-07-11; free's quality-prior failure is Codex
  fix #4; reconstructed-prompt gallery in notebooks/2026-07-03__prompt-optimality-gepa-vs-ceiling §9.)*
- **The bound is conditional on the EVIDENCE the instrument sees — declare the evidence rung.**
  Same metrics, same executors: 108-char parsed claims → agency summaries (~240c) → full letters
  (~2K) → full articles (80K chars) give different ceilings, and the deltas are themselves the
  finding (evidence-thinning ladder). Never compare bounds across corpora without matching the
  rung. *(N&C v1→V2→v3.1 reframes; peer-review Leg A abstract→slice→chunks→full.)*
- **Check firing rates before interpreting a low bound.** On thin text most criteria are honestly
  absent — all-NO constancy (qwen25-3b: 122/133 metrics constant-NO on 37-word summaries) floors
  the curve without saying anything about articulability. Distinguish all-NO (absence) / all-YES
  (sycophancy) / flat (collapse); rare-firing metrics get precision/lift readouts, not AUC.
  *(ncv2_checks.py; rare-gate doctrine from the PR V/A/T audit.)*
- **A battery point has a plausibility gate.** API battery bal_acc ~.5-.6 on a frontier model =
  serving failure (rate-limit NaN storm, thinking-mode empty content), not model capability —
  quarantine the artifact (`.bad-429`) and re-run throttled; never let a junk z place a rung.
  Map invalid outputs to NaN (never a default class) and abort on NaN bursts >30%.
  *(reference_openrouter_hybrid_thinking_trap; api_probe hard() guard.)*
- **Every frontier executor rides with battery + planted controls in the same run** — a pair
  panel without truth-acc can't be placed on the capability axis (qwen3-max stalled at 22/36
  metrics = TACIT numbers with no x-position). Order the metric list so planted lands early.
- **The corpus/unit choice moves bounds as much as models do — eyeball N probe texts before any
  scale-up.** The v3 pool carried `<<COMMENT n>>` assembly markers, leaked boolean fields, a
  15MB attachment blob, and 54% single-agency skew — all invisible in summary stats, all caught
  by reading three texts. Dedup mass-mail by cluster representative; cap dominant sources;
  chunk size = the executor window (zero silent truncation). *(v3→v3.1 rebuild; dataset-first
  protocol.)*
- **rc=0 is not "it ran".** Upper-bound chains no-op silently when a registry/metric-set is
  missing ("no R2 groups for notice-and-comment; aborting" — exit 0, same-second timestamps).
  Grep the stage log for its work signature, not just the return code, before marking a bound
  stage complete. *(nc_v2 opt chain 2026-07-09.)*
- **Planted controls must be RE-PARAMETERIZED per corpus.** Planted rubrics bake corpus
  statistics into their text ("longer than 17 words" from a 15-word claims corpus); copying a
  freeze's planted entries onto a longer corpus (v3 letters, k_med 319w) silently breaks the
  rubric↔truth correspondence — length metrics saturate, the positive control loses power, and
  ladder monotonicity wobbles. Regenerate planted entries from the NEW pool's k_med whenever
  the probe corpus changes, even if bank metrics are unchanged. *(nc_v3 checks 2026-07-10.)*

## [isomorphic tacit seams (z×a)]

Locating the TACIT seam — where articulation depth stops compensating for model capability
(crossing z* per arm; β = horizontal shift between arms in battery-z units = the
articulation-capability exchange rate) — and testing whether seams are isomorphic across tasks
and families. Distinct from [metric seams] above (code↔judgment placement); this section is
about articulation↔capability. Doc of record:
`notes/2026-07-08__zxa-articulation-capability-exchange-spec.md`.

- **Freeze the arms; instrument the seam, don't optimize it.** Six arms per metric — name /
  definition / explanation / dossier + dossier_mismatched (derangement) + definition_padded
  (inert filler) — authored once, gated (word bands, section labels, verbatim planted rules),
  then byte-frozen. The two controls are what make a seam interpretable: mismatched separates
  *specificity* from format, padded separates *content* from length. Without padded you cannot
  tell arm-length collapse from content effects. *(z×a spec; [prompt optimization] boundary.)*
- **Crossings via isotonic PAV at a fixed criterion (balanced agreement .75), ALWAYS with
  censoring status** (`<=` left / `=` interior / `>` right). A ladder that doesn't span the
  criterion yields bounds, not estimates; ẑ* forecasts are assumption-arithmetic and clamp to
  censoring. *(zxa_fit.py; [scaling ladders].)*
- **The exchange rate is FAMILY-RELATIVE — never quote a task-level β.** The same metric can be
  β>+.25 in Llama and β=−.74 in Qwen (compressed-quotable, 2026-07-09 headline flip). Per-family
  fits on same-family ladders; the battery-z latent axis places points but never pools curves.
  *(project_osl_executor_scaling 17:50 entry.)*
- **Every crossing gets a collapse-sensitivity re-fit** (constant cells masked; flag >0.15 z
  moves or status changes). 17/63 humor rows moved — those seams are articulation-FORM-limited
  (the model can't process the arm), not knowledge-limited. *(zxa_fit collapse_sensitive.)*
- **κ≈0 across the frontier is NOT yet "fully tacit" — decompose it first.** Two impostors:
  (a) *base-rate artifact*: raw unanimity ≥~.93 with κ≈0 means the construct never fires on the
  probe slice (all-NO on 81-char one-liners) — print unanimity AND firing rate next to every κ;
  (b) *support starvation*: re-test on longer/stratified probes; artifact metrics develop
  variance (brand-identity yes .06→.24 at 12× support) while genuinely contested ones keep κ
  flat (tacit .23→.24). Only flat-κ survivors on adequate support are tacit candidates.
  *(subtask_kappa decomposition; LP first read 2026-07-09.)*
- **Two degeneracy diseases, two different fixes — diagnose before fixing.** Small models
  (≲3-8B) collapse constant in proportion to ARM length (model-specific polarity; fix = shorter
  or same-family-authored arms; check r(arm-words, const-rows) per executor). Frontier models
  go constant on HOMOGENEOUS/dense probe slices (probe-variance saturation; fix = probe
  redesign — no amount of model or prompt fixes a slice the construct doesn't vary on).
  *(zxa_degen_audit; [metric prompting] instruction-load entry.)*
- **Isomorphism claims need matched measurement conditions:** same probe support regime
  (shown-char medians ranged 81→3,158 across tasks — not isomorphism-ready), same slices
  (compute within-topic vs pooled κ to rule out data-slice heterogeneity — the subtask
  hypothesis), same arms, same criterion. A cross-task seam comparison inherits every mismatch
  as fake signal. *(z×a spec 21:00 caveats.)*
- **Planted mechanical rules ride in every slate as the seam's sanity gate — and planted truth
  is a function of the SHOWN text**, so it must be recomputed whenever probes change (on long
  probes PLANTED-length finally fired, yes .06→.27: the old truth vector was silently wrong for
  the new slice). *(LP lesson; [probe & dataset design] truncation truth-rule.)*
- **Same-weights-via-API is admissible for AUTHORING arms, not for scoring rungs.** Generation
  gates (word bands, labels, verbatim rules) validate the artifact regardless of provider;
  scored rungs inherit provider variance (quantization/serving) and stay on the local ladder.
  Record `author_backend`. *(OR scope lanes 2026-07-09.)*
- **Authoring word-count gates fail by undershoot, and blind resampling doesn't fix it** — a
  model that writes 110-125w against a 130w floor does so at every temperature (qwen 8/72 valid
  after 6 resamples each). The fix is a repair pass: expand/trim REWRITE seeded with the
  model's own draft, then re-gate; and state the consequence in the prompt ("under 135 words
  will be REJECTED, aim ~155"). *(author_fam_arms repair pass.)*
- **A seam row is only as good as its rung's battery point** — every new executor lands battery
  + planted in the same run; batteries failing sanity (gemma-2 27b<9b inversion) exclude the
  family from seam claims; above a non-monotone rung report z*, never N*. *([metric upper
  bounds] battery plausibility gate.)*
- **Cross-runner κ≈0.00 against EVERYONE (anchors included) = column misalignment, not
  disagreement — check alignment before interpreting.** `OSL_PROBES_FILE` consumes the file's
  FIRST 60 ROWS as padding: osl_sweep scores file rows 60..59+n, so any external runner must
  slice identically (or_runner `--probe-offset 60`; pre-sliced dumps use 0). Verify with an
  exact-text match of `_load_texts(...)[60:60+n]` against the runner's probes before comparing
  npz across runners. Corollary of the new κ rule: print unanimity + firing rate + an anchor-
  class κ next to every cross-model κ — a zero on metrics models demonstrably agree on is an
  instrument bug. *(hermes-4 κ=.00 artifact, 2026-07-10; spec note 00:30 entry.)*
- **A rung is a (weights, provider) identity — verify the model string before reusing a short
  name.** The `hermes405b` battery was hermes-THREE; tonight's runs were hermes-4 → new rung
  `hermes4-405b` with its own battery, never mixed. A battery json's `model` field is the
  ground truth, not the filename. *(cousin of the glm alias rule in [scaling ladders].)*

## [hierarchy L0→R3 relabeling (codability)]

Rebuilding the metric-statement hierarchy (L0 dedup/repair → R1 same-construct → R2/R3 parents)
with measured recall/precision at every level, per task. The doc of record and runbook is
`notes/2026-07-06__hierarchy-reconstruction-ledger.md` — read it EXACTLY before resuming; the
process was hard-won and every improvised deviation so far has lost to it.

- **Stage order is load-bearing: L0 repair BEFORE any R-level.** P0 frozen eval → P1 arbiter
  truth (Sonnet panel + GLM leg + Opus adjudication; truth locked, extends only additively) →
  P2 L0 repair rounds (loop-until-dry) → L0 freeze + rename → R1 → R2 → R3. Building R1 on
  unrepaired L0 measures the repair debt, not the relation. *(ledger master table.)*
- **Pairwise net → judge fleet → Louvain(res=1.0) is the ONLY validated R-build.** The
  group-proposer method FAILED (2026-07-07 pivot, caught by the gate); union-find/star
  partitioning is forbidden (single-bridge chaining — the reason Louvain was chosen).
- **No confirm stage at R-levels.** The R1 bridge-confirm killed 94% of bridges on 60%-anchor
  (=chance) judgments: recall .673→.537 for flat precision — REVERTED and retired. The L0
  confirm (harvest_screen → Opus confirm → chain-proof apply) IS sanctioned; don't conflate
  the two confirms. *(ledger "CONFIRM STAGE RETIRED".)*
- **Recall gate = bucket ceiling ≥ ~.9** (P(same-bucket | arbiter-SAME): what the batcher/net
  CAPS recall at). The ceiling is usually a CAP artifact before it is a net-type problem: CW R1
  went .675@cap9000 → .946@full-pairs. Raise caps/width first; re-measure the ceiling at EVERY
  level (net diffuseness grows up-tree); add embedding nets only if TF-IDF measurably decays.
- **Precision gate = verify/split pass** (arbiter judges within-group member pairs, splits out
  non-SAME chain-proof) whenever score precision sags. Know the trade: L0 repair is a
  deliberate recall-ward move whose precision cost COMPOUNDS up-tree — both metrics print at
  every level, neither may silently sag. *(v6-vs-repaired measurement.)*
- **Judge placement is fixed by measurement, not cost:** Sonnet-or-better judges the R-levels;
  GLM is L0-only (over-merges R1: 46% SAME → precision .78→.60) and runs with the ≥2-edge
  gate; the arbiter is an independent-family frontier model told its ruling IS truth;
  evaluator independence gets spot-checked (Opus vs Sonnet-5 on humor R1 truth agreed).
  *([judging & eval hygiene].)*
- **Blinded anchors in every batch — they are the tripwire, not a formality.** Every pipeline
  failure caught so far was caught by anchors (resumed-judge drift; the 60%-anchor confirm
  judges). Use the 0/1/2 scale with the calibrated exemplar block, verbatim.
- **Score on the FULL node set, never the edges-only graph.** Dropping eval pairs that touch
  isolated nodes inflated humor R1 to .748/.719; the honest full-set score was .673/.689. Any
  partition score that beats apply_pairwise's own number is a scoring artifact until proven
  otherwise.
- **Throughput settings already validated — use them, don't rediscover:** TERSE output (score
  only, no reasoning sentence) is anchor-validated at 400 pairs/shard (~5× the 150-with-
  reasoning default); ONE Sonnet wave at a time, internally capped ~16 (burst limits); GLM
  HTTP legs may run concurrently with a Sonnet wave; CE-route candidates and screen the top
  band (CE≥.8) first, band-descend the rest; keep a shard manifest (n_remaining + shard paths)
  and re-run missing/short shards on line-count checks. *(ledger wave architecture;
  2026-07-09 orchestrator ran 148-pair reasoning shards = ~5× slower than sanctioned.)*
- **Every fleet prompt carries exact absolute paths** (input shard, output file, one-line
  output schema) so agents never search the repo — the single biggest token lever; reuse the
  stored prompt templates verbatim. *([subagents & fleets]; user directive 2026-07-09.)*
- **Rename before the next level, from node representatives** (GEPA rename prompt) — names
  feed the next level's nets; sanity-read a sample of names+groups before building on them.
- **Collapse sanity at every partition:** too-high collapse ratio = over-merge, too-low =
  batcher/under-merge; check against the level's expected range before scoring, and log the
  per-level recall/precision/collapse triple to the ledger as each cell lands.
- **Widen-to-dryness BEFORE freezing L0 — a single top-band wave is not "repaired":** run the
  widen bands (2500-8000, then deeper if wet) until yield dries, on EVERY task, before the
  L0-vN freeze. *(2026-07-10 audit: humor/CW were frozen after top-band only; CW's ledger even
  listed "widen next" — never executed.)*
- **Miss-by-band triage BEFORE spending screen budget:** locate the current recall misses
  (truth-SAME eval pairs not co-clustered) in the candidate ranking first — it tells you which
  band (if any) can pay. Bands can be provably worthless: humor's parked 10,283 mid-band pairs
  held 0/64 measured misses (all 64 sat outside the old lexical net entirely). *(pair_id rule
  for cross-file matching = sha1(sorted keys, "||")[:16]; arbiter_eval ids use a DIFFERENT
  scheme — match by keys, not ids.)*
- **Net-vintage parity:** when the candidate net is upgraded (e.g. +BGE semantic lane), every
  task frozen on the OLD net must be re-triaged against the NEW net — old-net dryness does not
  transfer. *(humor was the only task still on the pre-BGE net; the new union covers 63/64 of
  its misses vs 0/64 for the old net's unscreened bands.)*
- **Audit legacy blocklists against truth before reusing:** build_candidates' block_score0
  drops cluster-pairs an OLD judge (v6) called DIFF — on humor it permanently blocked 10/64
  true merges. If truth shows false blocks, set block_score0=False and let the two-family
  verifier do the precision work.
- **Late-merge disruption is bounded — measure it BEFORE applying:** star merges keep the tree
  valid with no rebuild (the absorbed cluster's keys inherit the survivor's parent). Quantify
  ripple first: fraction of prospective merges sharing an R1 parent (zero visible change above
  L0) vs cross-parent (mass moves between existing parents; ~1-3% of L0 in practice). Apply to
  a NEW L0-v(N+1) file; never mutate the frozen version downstream levels were built on.

## [stats & readouts]

- **Within-group ranks + permutation-null for pooled statistics** — pooling raw ranks across
  groups of different sizes manufactured a 1.7× effect (PREDICTABILITY-2 retraction).
  *(project_metric_seam_proposal.)*
- **Permutation + leave-few-out for small-n contrasts, not case-bootstrap** — bootstrap CIs alone
  were anti-conservative; the TASTE>CRAFT "effect" was one outlier ("Verbal Wit", leave-1-out
  −71%). Check size confounds (agreement~size Spearman −.85) before any group contrast.
  *(census session 5.)*
- **Text-reuse guard for any lexical-overlap statistic** — URL-independence is NOT independence
  when canonical texts mirror across sites (62-88% same-bucket mirroring inverted the
  "journalism strongest dialects" result on verify, 2026-07-09). Quote-overlap-guard before
  claiming lexical convergence. *(census MIRROR CONFOUND entry.)*
- **Permutation tests need n_perm > m/α under Bonferroni** (≥999 for m=26) or the floor p-value
  can't clear the corrected threshold. *(project_ctree_perm_bonferroni_stump.)*
- **If an analysis is revised after seeing v1's p-value, disclose both versions** (math bucket
  rewrite: defensible via objective coverage defect, still disclosed). *(census math entry.)*
- **Report the recovery metric** C(R(Ω))=I(M_ω;M′) as THE number; discrimination floors are
  sanity preconditions, reported alongside. T=I(M_ω;X) lower-bounds the ideal-metric quantity;
  B_E upper-bounds. *(feedback_report_recovery_metric_only; feedback_T_lower_bound_Mstar_be_upper.)*
- **Results, not conclusions, while exploring** — state numbers/tests/ranges in notes; no verdict
  language ("X is not a useful lever") until the axis is actually spanned.
  *(feedback_report_results_not_conclusions.)*
- **Descriptive analysis names** (≤4 words, what it measures, not "Phase 2"); never rename what
  the user has settled on. Markdown tables for comparative numbers; keep rows short enough to
  render. *(feedback_descriptive_analysis_names; feedback_likes_tables.)*
- **Before calling a design circular, identify producer vs evaluator explicitly** — only flag if
  they're the same entity. *(feedback_reasoning_carefully.)*

## [vLLM & GPU infra (sk3)]

- **Offline batch always** — `LLM.generate/chat` over thousands of prompts; NEVER `vllm serve` +
  HTTP for bulk (22 rev/min vs offline batch; "never NEVER"). OpenRouter/HTTP for <10-example
  spot-checks only. *(feedback_metric_scoring_offline_batch_vllm; feedback_never_openai_server_
  for_bulk; feedback_metric_implementer_sk3_only.)*
- **Fill the GPU you're on before taking another** (batch sizes in the thousands; GPU_MEM_UTIL
  .90-.95; a job at 6.7/183GB is misconfigured); minimize GPU count on the shared cluster; ask
  before >2 GPUs. *(feedback_vllm_batch_size; feedback_gpu_usage; feedback_minimize_gpus.)*
- **Engine-init "free memory < desired" = GPU-state contention, not config** — zombie EngineCore,
  teardown lag, or another workstream grabbing the card between your snapshot and launch
  (2026-07-09: GPU6 went 6GB→169GB in 25 min). Lanes carry retry-on-init with a wait, pin
  CUDA_VISIBLE_DEVICES explicitly, and set VLLM_GPU_MEM_UTIL to fit cards with residents.
  *(reference_flashinfer_workspace_oom_is_contention + this session.)*
- **Kill by explicit PID only** — never pkill/killall on shared boxes; when stopping an offline
  bulk, kill the parent AND the VLLM::EngineCore child (it holds the memory); verify GPU state
  with nvidia-smi before ANY claim about it; verify "launched" with pgrep + engine-init log line.
  *(feedback_targeted_kills; feedback_kill_vllm_engine_core_too; feedback_verify_gpu_state;
  feedback_verify_running_processes.)*
- **Environment quirks**: `if __name__` main-guard for MoE/spawn (Qwen); VLLM_BLOCK_SIZE=32 for
  gemma2 (FlashInfer head-256); HF_HUB_OFFLINE=1 on compute invocations once the cache is
  complete (NAT64 HEAD-revalidation flake); HOME pinned to /lfs for nohup (AFS tokens); one knob
  at a time when tuning a working config. *(feedback_vllm_main_guard_moe_spawn;
  reference_gemma4_env_sk3; feedback_sk3_afs_tokens; feedback_vllm_safe_run_config.)*
- **Downloads**: robust_dl pattern (timeout per attempt + resume + retry loop) for CDN blackholes;
  a downloader with /proc/PID/io read_bytes=0 is wedged — kill by PID, clear .lock files, retry.

## [subagents & fleets]

- **Max-plan subagents by default; never the USC API key unless explicitly told.** GLM
  subscription endpoints are the free bulk alternative; PAAS/pay endpoints stay sparing.
  *(feedback_max_subagents_default; feedback_sparing_with_glm.)*
- **Model tiering: Sonnet for fan-out (any batch), Opus for hard judgment, Fable only
  solo+easy** — the priciest tier as unexamined fleet default burned the quota mid-fleet.
  *(feedback_subagent_model_tiering.)*
- **Lean-fleet envelope: ≤20 concurrent agents, ≤150 pairs/shard, one fleet at a time** — floods
  hit burst limits (which look like weekly walls but aren't — probe with 1 agent before
  concluding); oversized shards blow the 64K output-token cap. *(census session 4.)*
- **Exact data paths in every fleet-agent prompt** (input shard, output file) so agents never
  search the repo — the single biggest token-efficiency lever found; reuse the validated prompt
  templates/protocols rather than improvising. *(user directive 2026-07-09; hierarchy runbook.)*
- **Fleet agents self-validate to the caller's gates before finishing** (word windows, verbatim
  rules, byte-identical names) — and the caller re-validates centrally; agents may synthesize
  in-distribution exemplars when the real pool is degenerate, marked `"synthetic": true`.
  *(z×a authoring protocol; feedback_subagent_synthesize_exemplars.)*
- **Crashed workflows are harvested, not re-run** — the journal + per-agent transcripts persist
  every completed result. *(feedback_recover_crashed_workflow_results.)*
- **Lean monitoring**: fire-on-completion (one notification), no polling loops; batch status
  checks into single ssh calls. *(feedback_conserve_tokens_lean_monitoring;
  feedback_sk3_ssh_multiplex.)*

## [process & reporting]

- **Never delete data**: append + dedup; rename to `.bak` rather than delete; replace only after
  the replacement is verified complete. (Applies to editing authored artifacts too — keep the
  original at `.bak`.) *(feedback_never_delete_data.)*
- **No new measurement target/approach without sign-off**; engineering fixes: fix and move on
  without asking. The line: reversible internal fixes = just do; new estimands/anchors = propose
  first. *(feedback_check_before_new_approach; feedback_fix_and_move_on.)*
- **No handwaving** — "I don't know, investigating" beats a plausible fabricated cause; check the
  data before explaining a surprise. *(feedback_no_handwaving.)*
- **No overengineering** — quantify the gain before replacing a working simple approach; if
  savings < 2-3× build cost, don't. *(feedback_no_overengineering.)*
- **Keep `running-research-notes.md` fresh** after significant moves; every experiment has a
  doc-of-record note with a dated status log; memory hooks stay one line with detail in the file.
  *(feedback_keep_running_notes_fresh.)*
- **Dedicated per-task registries need ALL their pieces** (e.g., a new local-explanations task
  needs its FEW_SHOT_EXAMPLES entry) — grep for the sibling table when adding to any registry.
  *(feedback_local_explanations_per_task_fewshots.)*

## [silver-labeling] addendum — extraction-audit lessons (2026-07-10, Codex gpt-5.6-sol audit)

- **Parse-fail must never be encoded as an empty result.** The v1 review-norm harness wrote
  `passages: []` on parse failure — indistinguishable from "review contains no norms" (~8% of
  reviews, 30%+ on one shard, silently lost; empties in the audit sample were 13/13 failures,
  0/13 genuine). Every extraction row carries an explicit status
  (`ok/salvaged/retried_ok/no_evals/parse_failed`); downstream treats missing ≠ negative.
- **Output-token caps create recall bias against evidence-dense items.** A 1,024-token cap +
  all-or-nothing JSON parsing preferentially destroys evaluation-dense reviews (truncated array
  → whole review dropped). Fix: bigger cap + balanced-array salvage (recover complete objects
  from a truncated array), retry-different-seed for the rest.
- **Never patch the instrument mid-corpus when processing order is blocked.** Venue-blocked
  files (ICLR→NeurIPS→TMLR→eLife) make any mid-run harness change perfectly confounded with
  venue. Pattern: freeze the primary pass; run a uniform REPAIR PASS over the failure-identified
  rows corpus-wide afterward (frozen prompt, fixed mechanics only).
- **Exact-membership snap certifies precision, not selection.** Audit before trusting: here it
  passed 100% verbatim (988/988 grounded, 95.7% norm-bearing) — but it drops near-verbatim
  quotes (~0.9%) and a generic quote can match the wrong occurrence in a long source.
- **Claim-dir resume-safety needs a reaper.** A worker dying between claim and write strands the
  shard forever and deadlocks any "all N done" trigger; run a stale-claim reaper (age > 2×
  expected shard time → reclaim).
- **Signal hygiene before matching:** drop <25-char fragments, within-review containment dedup
  (keep shortest atoms, drop the containing sentence — else one idea double-counts), and tag
  paper-level recommendation sentences (`kind=recommendation`) separately from criterion norms.
