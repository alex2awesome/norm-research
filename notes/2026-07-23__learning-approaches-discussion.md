# Learning-approaches discussion — running record

Date opened: 2026-07-23. Status: living discussion note (user directive: "make sure you take
extensive notes about this learning-approaches discussion and lit review"). Companions:
- mechanism catalog: notes/2026-07-23__tacit-learning-mechanism-catalog.md (M-A/M-B/M-C +
  tacit-specificity filter + route-signature hypothesis)
- operationalization catalog: notes/2026-07-22__tacit-knowledge-operationalization-catalog.md
- capstone note: notes/2026-07-21__adding-tacit-knowledge-installation-channels.md (§7b, §9)
- prereg: notes/2026-07-22__exp-gtk-1-prereg.md (v1 P1-fail; exploratory B1 +.15/.17)

## Discussion timeline (compressed)

1. User hypothesis (07-21): tacit knowledge is environmentally reinforced — install by
   RL/FT, not articulation. → channels program.
2. GTK/STK split → decay-curve + meta-acceleration operationalizations → user pushback:
   "policies can also be explicit" → channel-difference discipline + tier ladder.
3. User: many operationalizations, hope they converge → 42-entry catalog → battery built →
   W0: PC1=.30, tacitness ≥3-dimensional (articulation-resistance ⊥ metacognitive-opacity ⊥
   distributional).
4. User: are we as confident on mechanisms? → 3-harvest mechanism review (74 items → ~25).
   Four cross-literature convergent mechanisms (contingent-fade; on-policy correction;
   structured contrast; explicit-first sequencing).
5. User correction (07-23): those four are GENERAL pedagogy, several with explicit parts —
   what's tacit-SPECIFIC? → specificity filter → 7 survivors, shared skeleton = the signal
   is never a representation of the content (SELECTION NOT INSTRUCTION) → route-signature
   hypothesis.
6. THIS ENTRY — two standing doubts (user, 07-23):

## Doubt 1 — NAMED-METRIC BIAS ("the metrics themselves still have names")

User's statement: we focus on named/explicit metrics. "Newsworthiness" is a loaded implicit
concept; we gathered explicit, NAMED metrics that supposedly compose it; VAT shows that isn't
everything. The program turned to studying implicit parts OF these metrics — but the unit of
analysis is still a named thing. Is namedness itself a biased slice of implicitness?

**Analysis (2026-07-23):** Yes — the bias is real and structural, and the program has been
circling it without naming it:
- Every battery/channel object is a construct WITH A NAME invoked by name. Selection into the
  bank required a name (mined from articulable sources). Tacit structure that never
  crystallized into a name is EXCLUDED BY CONSTRUCTION from the unit of analysis.
- Evidence already in hand that the unnamed layer exists and matters:
  (i) VAT rows: dense ceiling > max(V, A) — the T̂ residual is precisely outcome-relevant
  structure not captured by verifiable or articulable (named) components;
  (ii) A-bank degeneracy: mined (name-derived) banks are 54-68% degenerate — the naming
  pipeline itself loses information;
  (iii) metric-discovery plateau: plateaus are UPSTREAM (the nameable space exhausts);
  (iv) Phase-0 recombination verdict: articulations live inside known vocabulary — names are
  bounded by the same lexicon;
  (v) §5.1 residualization: the residual after named components IS an unnamed component —
  the design already contains the instrument.
- **The fix is to make the UNNAMED RESIDUAL a first-class battery object:**
  1. One cheap target pass per domain: the target's HOLISTIC quality judgment on the same
     items (no construct name — "how good is this X", or better: the domain's natural
     outcome question).
  2. Name-span coverage statistic: R² of holistic on the 90 named-construct policies.
     **1 − R² = the unnamed share** — a direct, per-domain measurement of how biased the
     named slice is. (Prediction from the differentiation result: unnamed share grows with
     target scale — bigger models have MORE structure outside their name-span?)  
  3. The residual vector becomes a TRAINING TARGET (channels can try to install it) and a
     RECONSTRUCTION TARGET (can any channel transfer structure that has no name?) — the
     purest test of the whole program: named metrics are the calibration set; the unnamed
     residual is the real quarry.
  4. Battery probes apply unchanged to the residual (statability of an unnamed component ≈ 0
     by construction on the name axis — but MCQ-recovery, exclusion, pressure-robustness all
     still apply).
- Connection: this is the VAT program (Outcome = V + A + Taste) meeting the channels program:
  **Taste ≡ the unnamed residual, now installable and profilable.**
- Honest caveat: "holistic judgment" still requires SOME prompt words; fully name-free
  invocation is impossible in a language interface. The gradient we can build: bare-outcome
  question ("accept?") < thin holistic ("good?") < named construct. Name-dependence curve
  along that gradient is itself a probe.

## Doubt 2 — WHO IS THE TEACHER? (offline outcome-observation, no instructor)

User's statement: the mechanism harvest is instruction-heavy. Behaviorist interest: what
humans learn OFFLINE by observing WHO GETS AWARDS, WHICH PAPERS GET ACCEPTED, WHICH GET
CITATIONS — end-results only. No teacher anywhere.

**Analysis (2026-07-23):** This names a channel class the harvest under-served:
**ENVIRONMENT-AS-TEACHER / selection-record learning.** Distinct from everything built:
| channel | signal source | teacher? |
|---|---|---|
| distillation (B) | teacher's per-item judgments | model teacher |
| §5.1 distal reward | teacher's HOLISTIC judgment as reward | model teacher (in-model) |
| consequence-exposure (M46) | outcomes of learner's OWN judgments | environment, self-indexed |
| **outcome-corpus (NEW: M63)** | **(artifact, real-world selection result) pairs — awards, accept/reject, citation tiers** | **environment only; no model anywhere in the loop** |
- The teacher-free property matters for two reasons: (i) it is the most tacit-specific
  acquisition route conceivable under the specificity filter (nothing in the loop describes
  anything — pure selection record); (ii) it is how the TARGET MODELS THEMSELVES acquired
  their evaluative dispositions (pretraining = reading the selection record of human
  culture). Teaching the 7B from the record vs from the 72B = replaying vs shortcutting
  enculturation — and comparing the two installed profiles measures what the shortcut loses.
- **Discipline tension, stated honestly:** real outcomes are human-generated labels.
  [[feedback-reconstruction-only-no-labels]] bars them as reconstruction TARGETS/validation.
  Resolution recorded (user's current directive expresses intent to go here):
  - training signal = real selection outcomes: NOW IN SCOPE for the outcome-corpus channel
    (user steer, this discussion);
  - estimand stays dual: (a) reconstruction vs the 72B target (comparability with all other
    channels — does environment-learning arrive at the same structure the model-teacher has?)
    and (b) y-seam AUC vs held-out REAL outcomes (metric_seam battery/y_seam_extend.py
    machinery exists) — no judge-reliability ceiling;
  - confirmatory runs still prereg'd; the real-ICLR §5.1 variant's sign-off is hereby
    superseded by this broader outcome-corpus decision FOR TRAINING SIGNALS ONLY.
- Data candidates already in ecosystem: ICLR accept/reject + sub-scores (peer review, 8,952
  papers exp2 + 272K reviews on sk3); patents grant outcomes; legal outcomes
  (legal-outcome-prediction datasets); PR F2P/P2F merge outcomes; citation counts
  (obtainable). Awards corpora = new collection (parked).
- **Design sketch (EXP-ENV-1, to be prereg'd after battery W1):** same executor; arms:
  (1) outcome-corpus training (real accept/reject, no model teacher), (2) §5.1 in-model
  distal reward, (3) distillation on 72B judgments, (4) shuffled-outcome control; readouts:
  (a) ρ vs 72B named-construct policies + holistic, (b) y-seam AUC on held-out real outcomes,
  (c) FULL BATTERY PROFILE per arm — the route-signature test with the environment as the
  tacit-route pole. Prediction menu: behaviorist (env-training installs Taste/unnamed residual
  better than model-teacher channels); enculturation-shortcut (72B-distillation ≈ env-training
  because the 72B already compressed the record); cynical null (env outcomes too noisy at
  feasible N).

## On the user's residual uncertainty ("not sure the initial filters mattered")

The articulation-failure-set filter was built for CHANNEL-DIFFERENCE tacitness claims (it
defines where the language channel demonstrably fails). For the acquisition question with
environment teachers, that filter is less central — the relevant stratifications become
(i) metric subjectivity/articulability, (ii) NAMEDNESS (doubt 1's gradient), (iii) unnamed-
residual share. Keep the failure-set filter for channel-difference experiments; don't impose
it on outcome-corpus experiments. Recorded as a design rule.

## Standing design consequences (rolling list)
1. Unnamed-residual program: holistic target pass per domain → name-span R² → residual as
   training+reconstruction target. (Battery addition; cheap.)
2. M63 outcome-corpus channel joins the tacit-route arm set; EXP-ENV-1 sketched above.
3. Route-signature hypothesis now has THREE route poles: explicit-route (articulation/
   coaching), model-teacher tacit-route (distillation/§5.1), environment-route (M63/M46).
4. Naming-gradient probe: bare-outcome < thin-holistic < named-construct invocation curve.
5. Failure-set filter scoped to channel-difference experiments only.

## Doubt 3 — "Is the isomorphism dead? It's focused on named concepts" (user, 07-23)

Resolution: **not dead — promoted from phenomenon to instrument.** Five load-bearing roles:
1. The estimand survives the name: isomorphism = reconstruction of a REFERENCE VECTOR (adverse-
   ρ/controls/splits/caps); the name was only the invocation handle. The unnamed-residual
   program uses the identical estimand on the residual vector. Act 3 fades the name-scaffold;
   the measurement underneath is unchanged.
2. Named layer = the calibration basis: unnamed share is DEFINED as 1−R² after the named
   battery — no named vectors, no residualization, no measurable "unnamed."
3. Comparability spine: EXP-ENV-1 compares channels on ONE scale = reconstruction vs the 72B;
   without it the route-signature hypothesis is untestable.
4. The named results are the navigation map (gradient, floors, inversion, failure sets,
   scaling-tacit class) for every new design.
5. The asymptotic-closure question is INHERENTLY about the named/language layer — names are
   the units of language-mediated transfer; named program measures the language channel's
   capacity, unnamed program measures what lies beyond; the program's object was always the
   BOUNDARY between them.
Conceded: named-isomorphism as a STANDALONE headline is complete-able Act 1 (frozen panel +
sealed endgame = its closure), bedrock evidence rather than conclusion. Three-act arc:
Act 1 named isomorphism (nearly complete, sealed) → Act 2 channels × battery (running) →
Act 3 beyond names (unnamed residual + environment record; estimand carried forward
name-free).

## Completeness audit (user question, 07-23: "are we fundamentally missing key components?")

Provenance split of the ~30 probes: ~1/3 ours outright (subspace cap, tier ladder,
scaling-asymptote, differentiation floor, gap-conditioned estimand, exchange rate,
route-signature, unnamed-residual, naming gradient, permuted-control transfer design,
policy-composition ladder); ~1/2 literature-concept/our-instrument (Dienes, Jacoby, Schooler,
ACT-R, Reber, Chase-Simon, Harlow, Imitation Game — none previously instantiated on model
judgment transfer; 3 flagged first-in-literature: §4 assay, rule-vs-practice OOD, SECI
fidelity); ~1/5 straight ports (stratification axes, chicken-sexing control, Sternberg
scoring). NO fundamental gaps; five specifics: (1) variant-row pass planner = the W1
engineering item (~1 day); (2) target-pass scheduling (composed/negated/HOLISTIC — the
holistic pass doubles as the unnamed-residual instrument); (3) M63 ICLR ETL; (4)
pressure-axis operationalization = pending DESIGN DECISION (draft options, don't pick
silently); (5) judge annotations NOT missing (anchored GLM pipeline proven). Unscheduled
channels flagged: M44 immersion dialogue (conversation harness), M53 community corpus (data).
Human studies absent BY DESIGN (standing rule).

## v1c dose-endpoint result and what it does to the GTK discussion (2026-07-23)

EXP-GTK-1 v1c (N=468 rows/construct = the ENTIRE item-half-1; maximal offline dose) failed P1
again: +.108 < +.15, up only +.015 from +.093 at 128 rows — the offline-distillation dose axis
is closed. Three consequences for this discussion:

1. **The M17→M19/M20 argument is now load-bearing, not hypothetical.** More offline soft-labels
   cannot clear installation; the mechanism catalog's P1 tier (on-policy correction M19, KTO
   dispersion M20, active query M21) is the only remaining route to an installed policy in this
   design — and "on-policy correction of own attempts" was one of the four all-three-literature
   convergent mechanisms. The apprenticeship-pipeline v2 (distill → on-policy round →
   contingent fade → hint-free verification) has moved from "synthesis proposal" to "the next
   experiment the data demands."
2. **The GTK reading of v1's B1 signal is REVISED.** At high dose the vs-permuted double-diff
   on held-out constructs collapsed (+.143 → +.025, CI spans zero) while the trained-cell
   vs-permuted effect strengthened (+.202): construct-specific structure installs where
   trained; what generalizes across constructs is construct-GENERAL (the target's shared
   factor — style/calibration), which the wrong-construct control installs equally well at
   dose. This is exactly the differentiation-floor picture: within-domain zero-shot "GTK" from
   plain distillation was partly a low-dose artifact. Real construct-specific generalization,
   if it exists, must come from a richer route — grist for the route-signature hypothesis
   (tacit-route vs explicit-route arms), not from more soft-label SFT.
3. **Far-domain interference appeared** (B3 n&c Δreal −.030, CI-negative): heavy single-domain
   policy installation mildly damages far-domain judgment — the first observed COST of the
   weight channel, worth carrying into the channel-menu design (g-control now has a measured
   counterpart, not just a discipline).

### Change log
- 2026-07-23: note opened; doubts 1-2 analyzed; M63 defined; EXP-ENV-1 sketched; discipline
  carve-out recorded (real outcomes as TRAINING signals in scope; reconstruction-only
  preserved for validation estimands); unnamed-residual program specified.
- 2026-07-23 (later): v1c dose-endpoint section added (P1 fail x2, dose axis closed, B1-perm collapse, B3 interference).
