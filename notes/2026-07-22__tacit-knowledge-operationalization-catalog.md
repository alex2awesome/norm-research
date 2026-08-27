# Tacit-knowledge operationalization catalog — COMPLETE list from the literature

Date: 2026-07-22. Purpose (user directive): list LITERALLY ALL operationalizations of tacit
knowledge the literature offers — this is the exhaustive menu; implementation selection happens
separately (we will not operationalize everyone's definition). Status: ALL THREE harvests
COMPLETE (cognitive psychology §A: 14 entries; philosophy §B: 14 entries; STS/sociology §C: 14
entries) + synthesis section at the end (~40 distinct operationalizations, convergence map,
standing objections, first-in-literature opportunities).

Format per entry: criterion (what operationally counts as tacit) → LLM analog → status in our
program → caveat.

## A. From cognitive psychology / implicit-learning (harvest complete)

1. **Verbal-report criterion (Reber, AGL):** above-chance classification of NOVEL items +
   verbal report that is poor/uncorrelated with performance. → our statability assay. →
   IMPLEMENTED (§4). → weak alone (see #5).
2. **Surface-vocabulary transfer (Reber letter-set):** structure knowledge survives a surface
   alphabet swap → abstraction, not memorization. → reformat items preserving construct
   structure (tweets→captions); rank-agreement survival. → NOT implemented; moderate. →
   critique: shared fragment statistics can fake abstraction (Perruchet); exemplar-similarity
   can explain "rules" (Brooks & Vokey).
3. **Zero-correlation criterion (Dienes & Berry):** above-chance accuracy + ~zero trial-level
   confidence–accuracy correlation = knowledge without metacognitive access. → elicit
   confidence per judgment; corr(confidence, agreement) ≈ 0 while agreement > chance = a
   CONTINUOUS tacitness score. → NOT implemented; TRIVIAL to add; highest-value cheap addition.
   → LLM confidence calibration is prompt-sensitive; fix elicitation method in advance.
4. **Guessing criterion (Dienes):** above-chance accuracy restricted to trials self-labeled
   "guessing." → same, coarser. → NOT implemented; trivial.
5. **Information + sensitivity criteria (Shanks & St John):** a "no explicit knowledge" claim
   requires an EXHAUSTIVE and SENSITIVITY-MATCHED explicit test. → before declaring unstatable:
   ≥2 elicitation methods (free-text, forced-choice among candidate rules, MCQ reconstruction —
   we own recon-MCQ machinery!). → PARTIAL (single elicitation); make multi-method a GATE. →
   unfalsifiable in the limit; treat as discipline.
6. **Behavioral facilitation without recognition (SRT, Nissen & Bullemer):** speedup on
   structured sequences + chance-level explicit recognition (holds in amnesics). → agreement
   gains without recognition of the policy source. → covered by statability variants.
7. **Process dissociation, inclusion/exclusion (Jacoby; Destrebecqz & Cleeremans):** knowledge
   is AUTOMATIC to the extent it leaks through under an instruction to SUPPRESS it. → score
   items instructing the trained model to judge AGAINST its installed policy; residual
   correlation with the target = internalized (not merely instruction-following). → NOT
   implemented; NOVEL probe, moderate effort. → R/A algebra fails quantitatively; use
   directionally.
8. **Compilation signatures (Anderson ACT-R / Fitts-Posner):** (a) power-law practice curve;
   (b) growing immunity to concurrent-verbal interference; (c) DECLINING verbalization accuracy
   as performance rises across training; (d) transfer narrowing. → checkpointed training
   trajectory: agreement↑ while self-articulation quality↓ = compilation. → NOT implemented;
   uses existing pipeline + checkpoints. → explicit-first assumption is contested — our program
   tests it per construct.
9. **Identical-elements transfer narrowing (Singley & Anderson):** compiled skill transfers
   only to tasks sharing exact productions; breadth SHRINKS with practice. → near/far-transfer
   curves across training checkpoints; ACT-R predicts far-transfer plateaus/declines with more
   training — opposite of "more data = more general." → NOT implemented; low-moderate. →
   production-counting needs a neural proxy.
10. **RB vs II category structures (Ashby COVIS):** tacit = optimal decision bound not
    verbalizable; diagnostics: feedback-delay sensitivity (II impaired, RB not), verbal-WM load
    (RB impaired, II spared), observational-vs-feedback training (II needs feedback). →
    partition constructs into nameable-feature (RB-analog) vs holistic-integration (II-analog)
    sets; predict articulation works on RB only, distillation on both, CoT helps RB / hurts II.
    → NOT implemented; RECOMMENDED as the construct-validity backbone (replaces ad-hoc "feels
    tacit"). → two-system architecture contested; use the behavioral dissociations, not the
    mechanism.
11. **Structured-vs-scrambled interaction (Chase & Simon chess):** expertise advantage exists
    ONLY on structurally intact stimuli. → matched structured/scrambled item pairs; transfer
    sensitivity should collapse on scrambled = rules out surface-token keying. → NOT
    implemented; moderate item engineering.
12. **Articulation-success existence proof (Biederman & Shiffrar, chicken sexing):** a
    "maximally tacit" skill reduced to ONE page of instruction (+~40 points, parity with
    18-36yr professionals) — tacitness was under-articulation. → dedicated hand-extraction arm:
    strongest analyst distills ONE minimal instruction from target judgments; if a fresh model
    + that instruction ≈ fine-tuning, the construct was never tacit. → NOT implemented;
    RECOMMENDED positive control — this IS the "not-yet-articulated vs not-articulable"
    mixture-separator at the single-construct level. → best-case single-feature domain; don't
    generalize success.
13. **Verbal overshadowing (Schooler; Melcher & Schooler wine):** verbalizing IMPAIRS
    subsequent holistic judgment — specifically when perceptual expertise EXCEEDS verbal
    expertise (untrained drinkers hurt; novices and true experts not). → articulate-then-judge
    vs judge-directly; degradation localized to constructs where judgment-quality ≫
    rule-statement-quality. → PARTIALLY implemented unknowingly: **our below-floor sign
    inversion + humor articulation-hurts pattern is plausibly a verbal-overshadowing signature
    with the same mismatch boundary condition** — test the interaction, not the main effect. →
    needs a perceptual-vs-verbal fluency proxy per construct.
14. **Learning-set formation (Harlow):** trial-2 accuracy on the Nth NOVEL problem rises with
    N — meta-learning measured with no self-report. → multi-domain training; zero/one-shot
    agreement on domain N+1 as f(N). → our meta-acceleration (B5) is the single-domain version;
    the multi-domain curve is NOT implemented. → a rising curve may reflect a shallow generic
    heuristic; probe what transferred.

## B. From philosophy (harvest complete)

15. **Destructive-analysis / forced-focal-attention (Polanyi; empirical form = Beilock & Carr
    2001 explicit monitoring):** forcing focal attention onto subsidiary particulars disrupts
    integrated performance (pianist/centipede). → "explain-then-score" vs "score-directly"
    arms; degradation = Polanyi signature. Converges with #13 (verbal overshadowing) — TWO
    traditions predict our articulation-hurts findings. → cheap. → Polanyi is NOT a blanket
    inarticulist: some subsidiaries are retrospectively namable; only some permanently
    unspecifiable.
16. **Transmission-channel asymmetry (Polanyi connoisseurship; = Oakeshott cookbook; =
    Stanley's OWN testimony concession):** example transfers what precept cannot. → our core
    channel comparison. → IMPLEMENTED (the program itself).
17. **Deviant continuation / OOD divergence (Wittgenstein §185; Kripkenstein):** in-distribution
    compliance never distinguishes rule-grasp from memorization; the diagnostic is behavior AT/
    BEYOND the training boundary. → stratify held-out items by typicality; channels converge on
    typical items, diverge on edge cases (reslice existing data!). → NOT implemented; cheap
    reslice. → reviewer flags: nobody has run rule-taught-vs-practice-taught OOD comparison —
    literature gap = contribution opportunity.
18. **Community/ensemble criterion (Kripke's skeptical solution):** correctness = agreement
    with a community, not one agent's self-consistency. → ground truth as ENSEMBLE of
    resampled target judgments, not a single run (we partly do this via reps/forms; formalize).
19. **ASYMPTOTIC-CLOSURE TEST (Stanley & Williamson refined by Fodor — THE discriminating
    null-hypothesis test):** KEY SUBTLETY: sophisticated intellectualism does NOT predict
    text=practice (Stanley concedes testimony insufficiency, Know How p.130). What
    distinguishes the camps: does the articulation-channel gap CLOSE ASYMPTOTICALLY as text is
    optimized/enriched (intellectualism vindicated) or PERSIST at any richness
    (Polanyi/Ryle/Dreyfus camp)? → this IS our GEPA-c1 + subspace-cap + Tier ladder — the
    philosophical stakes of Tier 1/2 now named. → IMPLEMENTED in design; the reviewer
    recommends it as the HEADLINE confirmatory test.
20. **Error-correction-in-situ / multi-track adaptability (Ryle):** know-how = mid-course
    self-correction + adaptation across dissimilar exercises, not static reproduction. →
    perturbed/adversarial items; measure self-correction per channel. → NOT implemented;
    moderate.
21. **Two-curve divergence across scale (Dreyfus stages):** agreement-with-actual-judgments
    vs agreement-with-stated-rubric as capability grows — expert-stage divergence (judgments
    outrun any statable rule). → our OSL ladder data can run this: per rung, ρ(executor,
    target) vs ρ(executor, target's-own-articulation-predicted scores). → NOT implemented;
    uses existing data. → perception de-ruleifies at stage 4, action at 5 — two-step curve.
22. **Exemplar-only vs criteria-only arm (Kuhn):** worked exemplars without rules vs rules
    without exemplars. → 4th channel: in-context raw judgment EXAMPLES (no statements). → NOT
    implemented; completes the {in-context, in-weights}×{statements, examples} channel square.
    → CONTESTED empirically: categorization lit (Nosofsky GCM; 2023 JoCN) finds RULE learners
    generalize better on FAR transfer, exemplar learners on NEAR — live hypothesis, not
    established.
23. **Distributional/aggregate recovery (Hayek):** dispersed knowledge shows in population-
    level patterns no compact statement captures. → compare channels on distributional
    statistics (score distribution shape, subgroup calibration), not just item-level ρ. → NOT
    implemented; new DV, cheap.
24. **Self-report instability vs behavioral stability (Searle's Background/Connection
    Principle):** stable judgments + unstable/contradictory self-reports across elicitations =
    non-propositional capacity (confabulation signature). → repeat rule-elicitation under
    paraphrase/resampling; compare report variance vs judgment variance. → NOT implemented;
    cheap; also predicts WHICH constructs articulation-transfer fails on.
25. **Concrete-vs-abstract dissociation (Merleau-Ponty Schneider; Milner & Goodale D.F.):**
    good in-task performance + poor decontextualized rule-statement. → in-task judgment quality
    vs "state your general rule" quality, same model. → overlaps #24/#21; cheap.
26. **Post-hoc-rationalization recoverability (Dreyfus–McDowell, synthesized):** does the
    expert's post-hoc stated rule, handed fresh to a naive agent, reproduce the judgments? →
    literally our §4 articulation-transfer assay. → IMPLEMENTED in design; reviewer flags as
    unclaimed in the literature — contribution opportunity.
27. **Graded weak/medium/strong stratification (Gascoigne & Thornton):** pre-classify domains
    by tacitness grade BEFORE the experiment; test whether channel-gap ordering matches. → the
    prereg'd classification discipline (also answers audit F1). → matches our
    articulability-tier battery design.
28. **Fodor's strong null:** any articulation gap = failure to find the right propositions,
    never a distinct knowledge kind. → falsified only by persistent gaps under best-effort
    optimization + caps — i.e., Tier 2. (The strongest version of the null; subsumes #19.)
## C. From sociology / STS (harvest complete)

29. **Transmission test (Collins, TEA laser; Q-of-sapphire):** full written record fails;
    transfer succeeds only via personal contact (sapphire: ~a week in-person after literature
    alone failed). → the program's core channel comparison IS this test. → IMPLEMENTED.
30. **Imitation Game (Collins & Evans — the scored instrument for interactional expertise):**
    blinded judge tries to distinguish genuine domain member from pretender via Q&A; pass =
    judge at chance. Asymmetries validated (colorblind pass as sighted, not vice versa). → NEW
    instrument: blinded frontier-judge tries to distinguish (target, trained-executor) paired
    judgments/rationales on held-out items; judge-at-chance = strong transfer criterion beyond
    ρ. → NOT implemented; low-moderate.
31. **Mimeomorphic/polimorphic item stratification (Collins & Kusch):** context-invariant
    items can be faked by surface imitation; context-sensitive items are the true tacit
    signature. → per-ITEM (not per-domain) context-sensitivity annotation; predict channel-(a)
    deficit concentrates on polimorphic items. → NOT implemented; needs annotation pass
    (LLM-judged, anchored). → replaces the F1-risky domain-level Collins gloss with item-level
    prereg — answers audit F1 properly.
32. **Uninvention / source-retirement reconstitution (MacKenzie & Spinardi; Fogbank):** with
    the source gone, documentation alone fails to reconstitute capability. → retire the target;
    a fresh executor with ONLY the extracted articulations vs one with prior direct exposure —
    persistent failure from documentation = tacit residue. → variant of asymptotic-closure with
    an ablation framing; cheap on top of existing arms.
33. **SECI externalization-fidelity (Nonaka & Takeuchi; Gourlay critique):** SECI never had a
    quantitative fidelity test for externalization — commercial success was the only evidence.
    → OUR channel-(a) ρ IS the first quantitative externalization-fidelity measurement;
    near-parity of (a) with (b)/(c) = first real SECI validation; (a)-lag = Gourlay vindicated.
    → free (framing).
34. **Habitus generativity (Bourdieu):** competent improvised response to genuinely NOVEL
    field-configurations + poor self-report predictiveness + co-participant-recognized
    correctness. → OOD-stratified agreement + justification-gap; overlaps #17/#24. → the
    Distinction survey = precedent for taste-structure measurement without stated rules.
35. **Turner's deflationary critique (STANDING OBJECTION, not a test):** rank-agreement =
    correlated outputs from correlated training; positing a transferred "policy-object" is
    hypostatization. Legitimate claim: similar training histories → correlated behavior. → what
    answers him: generalization to items/domains sharing NO surface features with training
    (#17/#34); what supports him: decay once items diverge structurally. → cite and pre-empt in
    the paper; phrase all claims as behavior-correlation + generalization-profile.
36. **Profile-similarity scoring (Sternberg tacit-knowledge inventories):** scenario batteries
    scored as similarity of rating VECTOR to expert-consensus vector — never asks for a rule.
    Validity r=.10-.61 vs occupational criteria. → literally our estimand, from psychometrics.
    → IMPLEMENTED. → carries Gottfredson's g-confound critique (see #37).
37. **General-capability confound check (Gottfredson vs Sternberg):** are agreement gains
    domain-specific or just general capability? → add an UNRELATED control judgment task to
    every training arm: gains must be specific. → NOT implemented; LOW cost; should be a
    STANDARD control in EXP-GTK-1b+. 
38. **Justification-gap trajectory (Eraut):** the marker is a persistent gap between judgment
    accuracy and ability to state the criteria, tracked over a learning trajectory. → does the
    model's post-hoc rationale predict its own judgments as well as the judgments predict
    themselves (test-retest)? Track across training checkpoints. → overlaps #8/#24; cheap.
39. **Task-type × channel interaction (Lam):** channel ranking should REVERSE between
    individually-exercised vs collectively-negotiated judgment tasks. → compare channel
    ordering across e.g. elegance (individual) vs newsworthiness (collective). → higher cost;
    informative for the CTK row.
40. **Curriculum / legitimate peripheral participation (Lave & Wenger):** staged, gated,
    rising-stakes exposure vs flat-batch training. → curriculum fine-tuning protocol with
    agreement-gated admission vs flat batch. → higher cost; parked.
41. **Experimenter's regress (Collins):** competence-adjudication is circular — "good judge" is
    defined by getting the right answer. → interpretive caveat for all judge-based instruments
    (incl. #30); not a scored test.
42. **Codifiability gradient (Gorman info→skills→judgment→wisdom):** monotonic channel-(a)
    decay along the gradient. → finer-grained item/construct annotation; overlaps #27/#31.

## Synthesis (all three harvests in)

**~40 distinct operationalizations catalogued.** Convergence structure:
- **Independently derived by 2+ traditions AND already implemented by us:** transmission/channel
  asymmetry (#16=#29=Stanley-concession); statability (#1); profile-similarity estimand (#36);
  post-hoc-rationalization handoff (#26 = our §4 assay — flagged by TWO reviewers as unclaimed
  in the literature); asymptotic-closure (#19/#28/#32 = our Tier ladder — THE camp-separating
  test).
- **Cheap high-value additions (shortlist for the profile):** zero-correlation confidence
  criterion (#3); exclusion probe (#7); explain-then-score degradation (#15, converges with
  #13 verbal overshadowing — two traditions predict our sign-inversion); self-report
  instability (#24); OOD/typicality stratification (#17=#34, answers Turner); unrelated-task
  g-control (#37); Imitation Game (#30); ensemble ground truth (#18).
- **Standing objections the paper must pre-empt:** Turner (#35), Gottfredson g-confound (#37),
  Shanks & St John multi-method gate (#5), experimenter's regress (#41).
- **First-in-literature opportunities:** rule-taught-vs-practice-taught OOD comparison (#17);
  quantitative SECI externalization test (#33); the §4 assay as a Dreyfus–McDowell
  adjudication (#26).

## Selection principle (user, 2026-07-22)
We will NOT implement everyone's definition. The catalog is the menu; the tacitness PROFILE
(§7b.4⅞+++ of the channels note) selects for convergence value and implementability. Current
committed profile rows: channel-difference, statability, token-dependence, subspace-cap,
situational-shift, composability, scaling-asymptote, construct-intrinsic predictability.
Strong candidates from harvest A: #3 zero-correlation (trivial), #5 multi-method gate
(discipline), #10 RB/II backbone, #12 hand-extraction positive control, #7 exclusion probe.


## Master index by measurement cluster (2026-07-22)

The full clustered inventory (13 clusters: estimand/channels; language-ceilings; statability;
token-dependence; interference; internalization; generalization/GTK; composition/usage;
acquisition dynamics; de-relativization; stratification axes; ground-truth/discriminability;
controls/objections) with per-item status flags was delivered in-session 2026-07-22 and is
mirrored in the capstone note's §7b. Statuses: run = {estimand, channels A/B, Phase-0 probe,
subspace cap v0, scaling-asymptote v0, differentiation floor v0, EXP-GTK-1 v1, acceptance test,
control adapters}; designed/prereg'd = {Tier ladder, §4 assay, §5.1 reward, EXP-COMP-1,
EXP-COT-0, CoT-delta, contrast pairs, situational shift, meta-acceleration, ceilings/stop-rule};
shortlisted menu = {zero-correlation confidence, exclusion probe, explain-then-score, OOD
stratification, Imitation Game, self-report instability, g-control(→binding in v1b), RB/II
backbone, chicken-sexing positive control, mimeo/polimorphic item annotation}.
