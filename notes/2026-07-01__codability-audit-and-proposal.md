# Codability of preferences — audit of existing approaches + a subfield-controlled, embedding-free proposal

> **IMPLEMENTED 2026-07-01 (same day): `methods/codability/`** — strata/transfer/decompose/levels/
> controls modules + tests (11 pass); the §4.3 planted controls all land on their stated levels
> (`python -m methods.codability.run_codability --controls`); the genre-indexed control lands
> **L2-not-L4**, certifying the indexicality/tacitness separation. Live wiring points at
> `recon_channel` (R per stratum), orbit test–retest (T_g), and per-stratum `value_certificate`
> (ε_frac) — see `methods/codability/README.md`.

*2026-07-01. Context: the anthropological/linguistic study (see `project_anthropological_framing`).
"Codability" in the Brown–Lenneberg sense: how readily a concept is encoded in language such that another
agent can reconstruct it. Concepts = the R1/R2/R3 metric clusters. Two binding constraints from the user:
(1) the domain space is heterogeneous — CW metrics include horror-specific, adventure-specific, and generic
metrics, so **subfield variance must be controlled**, not pooled over; (2) **no embedding spaces in the
measure** — learned Euclidean geometry shifts with model training. Behavioral (verdict-space) quantities,
information-theoretic functionals of verdicts, categorical judge decisions, and corpus metadata are allowed;
embeddings at most as an efficiency pre-filter that never touches the reported number.*

---

## 1. What codability means here (the linguistic anchor)

Brown–Lenneberg (1954) operationalized color codability as: naming agreement across speakers, name brevity,
intra-speaker consistency, naming latency. The modern information-theoretic successors (communication-game
paradigms; efficiency-of-naming) measure it as **communication success**: bits needed for a listener to
reconstruct the referent. Our apparatus is exactly the second tradition, upgraded to be behavioral:

| classic codability component | our operationalization | instrument |
|---|---|---|
| communication success | recovery `R` (articulate → re-execute, held-out) | `recon_channel` |
| naming agreement (inter-speaker) | inter-reconstruction behavioral κ across model families | scorecard #7 |
| name brevity | MDL: shortest rubric reaching agreement α (the L-axis / `C_α`) | budget caps |
| intra-speaker consistency | test-retest ρ + form-orbit stability (§12.6.2 gates) | `measures`, `form_invariance` |
| recognition vs recall | MCQ identification (upper) vs free-gen (lower) | `analyze_bounds` |
| depth of the criterion field | greedy value tail / ε-gap (§12.6.3–4) | value census |

**The retraction lesson (binding).** The earlier subtask-codability attempt was RETRACTED because a
text-level SAME/DIFFERENT judge tracks *meaning*, not codability (`project_subtask_codability_result`).
Rule: codability is never measured by asking a model whether something is describable, or whether two
descriptions match — always by whether an articulation, **re-executed**, reproduces the verdicts.

## 2. Audit of existing approaches

| approach | what it measures | embedding-dep? | subfield control? | verdict |
|---|---|---|---|---|
| `R`, `A = T − R` (recovery loop) | communication success | none (verdict-space) | **NONE — pooled probes** | KEEP as the core primitive; must be stratified (§4) |
| MCQ recognition bracket | recall vs recognition rungs | none | none | KEEP; stratify |
| inter-implementation κ (#7) | naming agreement | none | none | KEEP; stratify |
| reliability ρ / form orbit | consistency | none | n/a | KEEP as gates |
| MDL / L-axis | brevity ("code rate") | none | none | KEEP; per-stratum |
| code↔judge convergence (#8) | compilability (level 0) | none | none | KEEP |
| count census (α, B_E, Chao1, C_lo) | criterion-vocabulary growth | none | none | NOT a codability measure (Lemma 12.6.0); descriptive only |
| value census / ε-gap (§12.6) | depth / tacitness certificate | none | none yet | KEEP; per-stratum it becomes the TACIT rung |
| **R1/R2/R3 clustering itself** | the concept UNITS | **YES — BGE/CE blend, τ=0.92** | mixes strata inside clusters | **the one deep vulnerability** — see below |
| structural Zipf/entropy on clusters | corpus shape | inherits clustering | none | descriptive only |
| noun-verb thickness, `v_struct` | text-structural thickness | none | none | correlational *validation targets*, not measures |
| `semantic_behavioral` judge | partition (merge) tool | none (categorical judge) | n/a | partition only — never the score (retraction lesson) |
| STaR local explanations | per-example articulability | none | none | complementary, per-item signal |

**The embedding vulnerability is in the concept definitions, not the measures.** Every per-cluster
codability number is conditional on cluster membership that was carved by an embedding blend. Mitigations,
in order: (i) run codability at **R3** (workflow-verified, 0 singletons — the human-auditable level);
(ii) the **transfer matrix** (§4.1) re-validates membership *behaviorally* and flags FRAGMENTED clusters —
an embedding-free clustering signal that feeds the deferred re-clustering audit as a by-product;
(iii) the §12.6 quotient partition (judge-merge / behavioral-split) contains no embeddings; where my earlier
handoff suggested embedding pre-filtering of judge pairs, substitute lexical MinHash or a behavioral-MI
floor. State the residual honestly: codability numbers are "codability of the stated concept definition."

## 3. The subfield problem, precisely

Four distinct failure modes when probes/verdicts pool over subfields (CW: horror, adventure, romance, …):

- **(a) Mixture masquerading as tacitness.** A metric perfectly codable *within* each subfield but with
  different realizations per subfield ("good pacing" ≠ same rule in horror vs adventure) has low pooled `R`
  — reads as tacit when it is actually **indexical** (codable given the frame). This is the Simpson pattern
  the project has hit before (F2P/P2F).
- **(b) Probe imbalance.** A horror-specific criterion scored on mostly-adventure probes yields a
  near-constant signature (`frac_near_constant_sig`) → its codability is *undefined*, not low.
- **(c) Ceiling confound.** The pooled dense ceiling `C` partly predicts via subfield identity (genre
  recognition is easy), inflating the apparent gap for genre-blind rubrics. Same class of fix as the
  press-release publisher deconfound.
- **(d) Level-of-hierarchy impurity.** R3 merges R2 variants; a codability number on a heterogeneous
  cluster conflates concept tacitness with cluster mixture. Codability must be reported *at a stated level*
  with a heterogeneity read alongside.

## 4. Proposal: the stratified Codability Profile

### 4.1 Design (per metric `M_i`, all embedding-free)

```
strata g       from corpus METADATA (genre/venue/subtask tags) or a categorical judge tag
               (one-word genre classification) — never topic models / embeddings
probes         stratified quotas: >= ~100 items per stratum (pilot: 4 strata); frozen, held-out
target         m̄_ω,i per stratum (orbit-averaged soft verdict, §12.6.2)

R_global       one rubric induced from pooled pairs, executed on pooled held-out       (status quo)
R_g            rubric r_g induced from stratum-g pairs ONLY, executed on held-out g    (conditional)
T_g            per-stratum transmission ceiling (and per-stratum dense C_g at task level)
TRANSFER       M[g→g'] = behavioral agreement of exec(r_g, ·) with M_i's verdicts on held-out g'
               (diagonal = R_g; the row/column structure is the heterogeneity read)
kappa_g        inter-family reconstruction agreement per stratum      (naming agreement)
MDL_g          shortest rubric reaching alpha-agreement per stratum   (brevity)
eps_g          the §12.6 value-census ε-gap per stratum               (the TACIT certificate)
EXEMPLAR gap   R_g(rules-only) vs R_g(rules+few-shot exemplars)       (ostension channel)
```

**Decomposition (the subfield control):**

```
Δ_context  = mean_g R_g − R_global      ≥ 0 typically   — INDEXICALITY: codable given the frame
A_g        = T_g − R_g                                   — the within-frame articulation gap
mixed model  R_ig = μ + a_i + b_g + (ab)_ig + noise      — a_i = subfield-ADJUSTED codability of metric i;
             Var[(ab)] = indexicality variance; bootstrap over items within strata; attenuation-correct
             each R_ig by the per-stratum reliability ρ_ig.
```

### 4.2 The codability levels (ordinal verdicts, replacing a single scalar)

| level | name | operational criterion |
|---|---|---|
| **L0** | COMPILABLE | code implementation converges with judge (scorecard #8 high); `R → 1` via program |
| **L1** | UNIVERSALLY CODABLE | `R_global ≈ R_g ≈ T_g` ∀g; one short rubric (small MDL); high κ; form-robust |
| **L2** | INDEXICALLY CODABLE | `R_g ≈ T_g` but `R_global ≪ mean R_g` (Δ_context large); transfer matrix diagonal-dominant; the code needs a frame parameter |
| **L3** | OSTENSIVELY TRANSMISSIBLE | rules plateau below `T_g` but exemplars close the gap (EXEMPLAR gap large) — transmissible by showing, not telling |
| **L4** | TACIT-WITHIN-FRAME | `R_g ≪ T_g` with the per-stratum ε-gap certificate at all budgets, exemplars included; `T_g` materially > 0 |

Plus three **exclusion gates** (not levels): UNDERSAMPLED (`f_1/N → 1` in the stratum, Lemma 12.6.0);
FORM-DOMINATED (§12.6.2 gates fail — meaning unstable under rephrasing, itself a linguistic finding);
**FRAGMENTED** (transfer matrix block-structured + per-stratum rubrics judged semantically DIFFERENT + the
cluster's leaf provenance splits by stratum → not one concept; a cluster-audit flag routed to the deferred
re-clustering, not a codability level). **L4 requires `T_g` materially > 0**: "no signal" (T_g ≈ 0) is
excluded — tacitness is a property of a *practice that exists* but resists articulation.

L1→L4 is the anthropological gradient: fully tellable → tellable-in-context → showable → only learnable by
immersion. The per-task **codability map** (fraction of metrics at each level, per community) is the
headline deliverable; Δ_context per task is the "indexicality of this community's evaluative language."

### 4.3 Planted controls (mandatory before any real claim)

1. **Universal code metric** (e.g., dialogue-punctuation rule) → must land **L1**.
2. **Genre-indexed code metric** — `M(x) = f_horror(x) if genre(x)=horror else f_adventure(x)`, both
   branches code → must land **L2**, NOT L4. *This is the positive control for the whole decomposition*:
   it proves the design separates indexicality from tacitness.
3. **Exemplar-only metric** (a rule easy to satisfy from examples, hard to state — e.g., matching a
   specific stylistic template) → should land **L3**.
4. **Noise metric** (shuffled verdicts) → must be EXCLUDED by the `T_g ≈ 0` / reliability gate, never L4.

### 4.4 Cost & pilot

Pilot: CW, 3 R3 metrics spanning the expected range (one craft/verifiable, one generic-taste, one
genre-specific) × 4 genre strata × ~120 probes/stratum × 2 reconstructor families; signatures cacheable
(#6). Then scale to the R3-57 set. Per-stratum probe quotas multiply scoring cost by the stratum count —
the main cost driver; mitigate by sharing strata across metrics of the same task (one stratified probe
pool per task).

### 4.5 Where it plugs into the theory

The per-stratum ε-gap is §12.6.3–4 verbatim, run within strata; the levels extend §12.6.6's verdicts; the
transfer matrix is the behavioral (embedding-free) heterogeneity instrument that both controls confound (c)
and produces the re-clustering evidence for the R2/R3 audit the user has deferred. Task-level: the same
profile with target `M_H` (revealed practice) and per-stratum deconfounded ceilings `C_g` gives the
codability map of the *practice*, which is the anthropological headline.
