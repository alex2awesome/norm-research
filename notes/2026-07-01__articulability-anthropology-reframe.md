# Articulability as anthropology — what the human-exceptionalism framing changes

*2026-07-01. Companion to `2026-07-01__prompt-optimality-upperbound-critique.md`. Context (user):
the project is a **linguistic and anthropological study of the articulability of human preference** —
the goal is to show how incommunicable preference is, in support of human exceptionalism. LMs and
prompt upper bounds are the *instrument*, not the subject. This note reviews the theory
(`2026-06-18__prompt-optimality-theory.md`) under that framing and proposes improvements. All target
changes marked ⚠ require sign-off (per `feedback_check_before_new_approach`).*

---

## 0. One framing flag, stated once

"Show X" is a conclusion, not a hypothesis. The apparatus's scrupulous honesty (conservativeness
directions, named assumptions, right-censoring) is the paper's *entire* credibility — because the
first reviewer objection to "human preference is incommunicable" will be **"your articulation
apparatus is just weak."** The theory already knows this (§5: "high `A` is never proof of tacitness —
only 'we couldn't articulate it *yet*'"). So: state the thesis as the falsifiable
**H: a taste residual survives all articulation scaling**, design to *kill* it, and let survival be
the finding. Expect (and want) a **gradient**: some preferences will prove highly codifiable (math
correctness, PR identity-tasks), others not (CW taste). "The articulability map of human preference"
is a stronger and more anthropological paper than a monolithic ineffability claim — variation across
domains is what anthropology is *about*.

## 1. The good news first: the theory's "weaknesses" are exactly the right shape for this claim

Three things that looked like defects for optimality certification become the correct epistemic form
for an ineffability claim:

1. **You can never prove ineffability — only certify exhaustion of a named articulation process.**
   That is precisely §12.5's process-relative scoping: every `B_E`/coverage claim is "relative to
   {these generators, this optimizer, these readers}." For optimality this felt like a concession;
   for anthropology it is the *standard* position (Collins: knowledge is tacit *relative to current
   explicitation practices*). Keep §12.5 and promote it to the estimand definition.
2. **`A = T − R` is an upper bound on tacitness; the certified object is the LOWER bound on the gap.**
   The ε-gap machinery (value missing-mass ★ + adversarial saturation, see the companion critique)
   is what converts "our optimizer plateaued at R" into "no prompt in the reachable class exceeds
   R+ε at confidence 1−δ" — which is what makes the *gap* a lower-bounded finding instead of an
   effort report. The upper-bound half of the bracket is not optional under this framing: it IS the
   evidence.
3. **Right-censored survival curves** ("not yet articulated at budget B") are already the designed
   readout (README scaling section). That is the honest grammar for the claim.

## 2. The structural gap: the human never appears in the current loop

Everything in the theory is LM-internal: `M_ω` = an LM executor's verdict on an LM-compiled rubric;
proposers, reconstructors, executors, probes — all LM. The recovery certificate measures
**instrument-level** articulability (can *this metric's operationalization* be recovered), which is
the right validation layer — but no quantity in the doc is yet *about a human practice*. For an
anthropological claim, the human must enter at two places:

### 2a. ⚠ The human-target bracket (the headline theorem candidate)

Instantiate the same bracket with the target = the **community's own verdict pattern** `M_H`
(outcome labels, multi-annotator panels, pairwise contest results — the practice itself):

```
R(p̂ → M_H)  ≤  OPT_Σ*(→ M_H)  ≤_(★)  OPT_∞(→ M_H)   vs.   T(M_H) = I(M_H; X) ≈ C_dense
─────────────   ──────────────────────────────────         ─────────────────────────────
best articulated  ε-certified ceiling of the LANGUAGE        the practice's predictable
rubric (measured)  CHANNEL (the ★ certificate, human target)  signal (dense/twin ceiling)
```

> **Certified codification gap of the practice:**
> `A_H ≥ lowerCI(C_dense) − [OPT_Ω + ε]  > 0` at confidence 1−δ.

This is the existing `C − B` arithmetic (README, attenuation-corrected, twin/1-NN sandwich) **made
into a certificate** by the prompt-class upper bound. It is the theorem-shaped version of "the
community knows more than it can tell." Notes:
- This does NOT violate the no-`Y` rule: the anchor-free per-metric recovery loop (§1) stays as the
  *instrument-validation* layer (is each rubric a faithful instrument). `Y`/human labels define the
  **practice-level** target of the anthropological claim — a different level, the one the V/A/T
  program always used. Scope both clearly.
- With a human target, `T(M_H) < H(M_H)` (annotator noise), and the ceiling estimate is the dense
  model / twin bound with inter-annotator attenuation — the existing dense sweeps ARE the `T̂`
  estimates. Guard G4 (T̂-estimator ≥ reconstructor strength) is satisfied by construction.
- The dense wrinkle, faced honestly: if `C` is high, a *machine* does capture the preference — from
  demonstrations. So the defensible exceptionalism claim is about the **language bottleneck**
  (learnable by immersion, not compressible into telling), not "machines can't have taste." That is
  Polanyi's claim, and it's the one the data can support.

### 2b. The audience/channel index: `A(m; W, Φ, E)`

"Articulable" currently has only the reader-axis `E` (§5.5 executor ladder, §10 item 5). The claim
"incommunicable" quantifies over **writers** `W` (who articulates) and **channels** `Φ` (what counts
as telling) too. Make both explicit:

| axis | cells | what exists already | what to add |
|---|---|---|---|
| writer `W` | LM reconstructor / **human expert** | GEPA/GLM reconstruction | ⚠ small human-reconstructor arm: experts write the rule from the same (x, verdict) pairs (n≈30, 3–5 experts) |
| reader `E` | LM ladder / **human reader** | §5.5 V→V+A→C ladder | ⚠ human-executor arm: humans apply the best machine rubric; if human+rubric ≈ LM+rubric ≪ C, the bottleneck is the *rubric* (language), not the reader — the cleanest incommunicability evidence |
| channel `Φ` | rules / **exemplars (ostension)** / interaction (apprenticeship) | few-shot axis in scaling design; dense model = learning-from-demonstration limit | elevate rules-vs-exemplars to a headline contrast: "taste transmits by showing, not telling" is the Wittgenstein/Daston prediction, directly testable |

The **native-rubric benchmark** is free and underexploited: the scraped 361K rubric corpus IS the
community's own auto-ethnography of its standards. Measure `R(community's own rubrics → M_H)` next
to `R(machine-optimized → M_H)`. "Even the natives' own best articulation undershoots their
practice — and so does a superhuman optimizer's" is the two-pronged result.

**Emic/etic provenance** (anthropological validity): R2 clusters mined from human rubrics = *emic*
categories (the community's own terms); LM free-gen criteria = *etic*. The census currently pools
`children` (emic) with free-gen (etic). Tag and report per-metric source provenance (the bank
scorecard's source-diversity already counts this) so the objects of study are demonstrably the
community's categories, not LM inventions.

## 3. Reinterpretations of existing quantities under the new framing

- **`α` vs `α_V` = linguistic productivity vs semantic depth.** Count-`α ≈ 1` (inexhaustible
  *phrasings*) with `α_V ≪ 1` (value saturates) = mere paraphrase productivity — a fact about
  language, not about the metric. The exceptionalism signature is **`α_V → 1` under a human target**:
  every new criterion keeps *recovering more of the practice* — unbounded depth, the Wittgensteinian
  regress made measurable. So a NO-GO is not a failed run; it is a *finding of inexhaustibility* —
  provided it's on the value axis and survives the estimator guards. (The current count-axis NO-GOs
  are not yet this finding.)
- **E-axis substitution = enculturation/indexicality.** "Does a stronger `E` reach the same recovery
  with fewer criteria" (§10 item 5) is anthropologically "enculturated readers need less telling"
  (shared context does the work — indexicality). Testable today with existing artifacts: community
  fine-tuned readers (norm_embed CEs / LoRAs) vs base models on the same rubric — a within-project
  measurement of what socialization substitutes for articulation.
- **Sample-efficiency of enculturation.** The dense data-scaling sweeps (`project_dense_model_sweeps`)
  give the machine's demonstrations-needed curve; human apprentices are (by the exceptionalism
  hypothesis) orders of magnitude more efficient. Reporting the machine's curve alongside
  human-learning literature is a cheap, honest exceptionalism datum that doesn't require proving
  ineffability at all.
- **The `1 − C` residual = personal taste.** Currently out of scope; with multi-reviewer structure
  (peer review has 3–4 reviews/paper) the intersubjective ceiling and the personal residual separate:
  communal-articulable (`B`) / communal-tacit (`C − B`) / personal (`1 − C`, split from noise via
  twin structure). That three-layer decomposition IS the anthropology of taste (Bourdieu's habitus =
  the communal-tacit layer).

## 4. The direction-of-error flip (the most important discipline change)

The theory's conservativeness is oriented against over-claiming *articulability* (anti-conservative =
false saturation). Under the tacitness thesis, motivated-reasoning risk points the OTHER way: **every
instrument weakness inflates the gap the thesis wants.** Enumerate and gate:

| gap-inflating failure | existing guard | status |
|---|---|---|
| weak reconstructor (never finds the rule) | reconstructor-scaling curve; W=human arm | partial — human arm missing |
| weak executor (can't apply the rule) | §5.5 E-ladder; C4 compiler-vs-LLM | shipped |
| hard-binary readout artifact | G3 soft P(YES) | fixed |
| undersized probes / over-split species | probe-knee check; semantic-merge partition | in flight (companion note) |
| judge too weak → deflated `B` | judge-scaling asymptote (README) | designed, must run |
| dense `C` inflated by confounds | deconfounding program (publisher/topic/length) | shipped per-task, must be cited per gap |
| label noise deflates `C` | attenuation correction | conservative direction — fine |

Evidentiary standard for every "incommunicable" claim: (i) the C1 positive control (planted
articulable rule recovers `R ≈ 1` through the FULL pipeline) reported next to it; (ii) gap =
lowerCI(ceiling) − upperCI(articulation); (iii) scaling asymptote or right-censoring language;
(iv) the ★ prompt-class certificate. Anything less and the finding is "we didn't try hard enough,"
which is fatal *specifically because* the conclusion is desired.

## 5. Disciplinary anchors to wire in (related work the current doc lacks)

- **Polanyi 1966** (*The Tacit Dimension*) — "we can know more than we can tell": the thesis, verbatim.
- **Collins 2010** (*Tacit and Explicit Knowledge*) — RTK/STK/CTK maps onto the `A`-decomposition:
  (c) executor-limited ≈ relational (contingently tacit, closable); the residual surviving strong-W /
  strong-E / rich-Φ = candidate **collective tacit knowledge** — the layer Collins argues is
  human-distinctive (socialization-bound). This gives the paper its theoretical spine.
- **Nisbett & Wilson 1977** ("Telling more than we can know") — humans confabulate their own
  evaluative criteria; justifies recovering rules from *verdicts*, not introspection (our design).
- **Schooler** (verbal overshadowing; Melcher & Schooler 1996 wine) — articulating degrades expert
  judgment; experimental-psych precedent for the language bottleneck.
- **Bourdieu 1979** (*Distinction*) — taste as embodied class habitus: the communal-tacit layer.
- **Majid** (differential ineffability; Jahai smell lexicon) — linguistic precedent that whole
  domains are lexically under-served; per-domain articulability variation is expected.
- **Ryle** (knowing-how/knowing-that), **Geertz** (thick description), Daston/Wittgenstein/Dreyfus
  (already in `project_thin_thick_rules_philosophy`).
- §8's thin/thick ↔ `(α, γ, A)` is [conjectural] in the doc but becomes the **central hypothesis**
  under this framing — its validation plan (correlate with `v_struct`, expert thin/thick ratings)
  should be executed, not deferred.

## 6. Prioritized additions (cost-ordered)

1. **Free / reuse:** native-rubric benchmark vs machine-optimized rubrics (corpus exists); emic/etic
   tags in the census; multi-reviewer twin split of `1−C`; community-tuned-reader vs base-reader
   E-axis test (CEs exist); dense data-scaling curves re-read as enculturation-efficiency.
2. **Cheap (CPU + small API):** ★ value-census certificate pointed at `M_H` per task; α vs α_V
   productivity/depth read per metric; rules-vs-exemplars channel contrast (few-shot axis exists).
3. **Small human studies (the ones that convert LM-relative → human-relative):** human-executor arm
   (~30 items × 3 annotators × best rubric); human-reconstructor arm (3–5 experts); per-item
   "could you explain why?" self-report correlated with per-item recovery.
4. **Write-up:** rename the headline gap for the human paper (`A_H` = the **codification gap**);
   scope §12.5 process-relativity into the estimand; per-domain articulability map as the main figure.

**Not fixable, and now rhetorically fine:** support-completeness (§12.2.4) — "maybe some un-proposed
articulation exists" — is the epistemically correct residual for an ineffability claim; state it as
the standing falsifiability of the thesis rather than a defect.
