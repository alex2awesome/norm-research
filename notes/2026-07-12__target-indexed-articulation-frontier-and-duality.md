# Target-indexed articulation frontiers: joining M_omega, CUF, gestalt, and scale

Date: 2026-07-12  
Status: synthesis/design note; additive to the existing tacitness line. No frozen artifact or legacy
schema is superseded.

## 0. Architectural decision

Preserve the existing decompression, name-sufficiency, stipulation, unit-grid, OSL, CUF, and
prompt-optimality artifacts. Add a target-indexed layer beside them.

The larger-reader/name-only target is a particularly clean **secondary lexical experiment**. It is
not the definition of tacit normative knowledge. The main scientific object is broader: a holistic
normative policy or social practice that may include perceptual gestalt, precedent, framing,
context-sensitive judgment, and enculturated know-how rather than a collection of facts.

## 1. Target views: never collapse these into one random variable

For construct `i` and community/frame `g`, keep four possible targets separate:

1. `M^P_{i,g}` — **practice target**: repeated community/expert judgments, pairwise preferences,
   archival decisions, or an identified latent normative policy.
2. `M^G_{i,g,S}` — **gestalt-informant target**: a large reader's holistic judgment under a minimal
   task/community frame, without decomposing the judgment into criteria.
3. `M^N_{i,g,S}` — **name-invoked target**: the large reader's behavior when given the construct
   name. This is lexical/indexical tacitness.
4. `M^b_{i,g}` — **operational target**: a frozen rubric/checklist realization; `M_omega` is one
   member of this class.

Every result names its target view. The fixed-target theorem applies to all four once a target is
frozen. It does not establish that an operational or model target equals the community's ideal.

## 2. Articulation channels: tacit knowledge made explicit, not generic prompt content

Let `c` index an articulation channel:

- declarative: definitions and constitutive conditions;
- explanatory: mechanisms and rationales;
- procedural: recognition/application rules;
- ostensive: positive/negative examples, precedents, boundary and contrast cases;
- formative: sequenced teaching, critique, dialogue, or apprenticeship-like demonstrations;
- composed/gestalt: narratives, frames, personas, and joint presentations whose effect is not a
  function of separately executed unit verdicts;
- regulative: explicit standards for how judgments should be made (kept distinct in normative
  domains because regulation can partly constitute the target practice).

The treatment is knowledge of the same construct externalized through one of these channels. Role
boilerplate, verbosity, task-independent reasoning advice, and test-optimized prompt tricks are not
doses of articulation.

## 3. The common mathematical object

Fix a target `M = M^{v}_{i,g}`, reader/executor `E_s`, and candidate articulation prompt `p`. Let
`Z_{s,p}(X)` be the reader's soft verdict channel. On one held-out item distribution and one
f-divergence,

```
R_s(p; M) = I_f(M ; Z_{s,p})
T(M)      = I_f(M ; X)
R_s(p;M) <= T(M).
```

This is exactly the surviving fixed-target `M_omega` theorem. Use soft target probabilities when
available. Normalize across targets by `R/T`, while retaining raw bits and degeneracy/reliability
gates. `T-R` is an upper bound on candidate headroom, not proof that a better prompt exists.

Let `P_{i,c}(b,d)` be the frozen class of prompts that externalize construct `i` through channel `c`,
with articulation budget `b` and interaction/combiner degree `d`. Budget should be reported as a
vector—certified-unit count, description length/tokens, and channel type—rather than pretending raw
words are an interval scale. `d=1` is unit-additive; `d=2` permits pair interactions; larger `d` and
the composed channel capture configural/gestalt structure.

With `H = Phi x Lambda x C` the CUF form/position/company measure, define a conservative robust
frontier

```
F_{i,g,v,c}(s,b,d)
  = sup_{p in P_{i,c}(b,d)} lower_H[ R_s(p;M) / T(M) ].
```

`lower_H` means held-out lower confidence bound after form/identity/context charges or the orbit
adverse end. Because `P(b,d)` is nested by free disposal, the true frontier is monotone in budget
even though raw prompt performance is not monotone when text is appended.

The target-indexed articulation debt is

```
D_{i,g,v}(s -> S; delta)
  = inf_{c,b,d} { cost(c,b,d) :
      F_{i,g,v,c}(s,b,d) >= baseline_{i,g,v}(S) - delta }.
```

Never reached is `+infinity`/right-censored, not a large finite cost. For the lexical supplement,
`v=N` and the baseline is the larger reader's name-only channel. For the gestalt experiment, `v=G`
and the baseline is the larger reader's undecomposed holistic judgment. For the anthropological
experiment, `v=P` and all readers attempt to recover the same social-practice target.

### 3.1 Do not collapse capability substitution into policy transplantation

Two experiments share this mathematics but have different success conditions:

1. **Capability substitution:** fix an independent practice, human, or cross-family target `M`.
   Ask whether `E_s + articulation` is noninferior to (or exceeds) `E_S + sparse` in recovery of
   `M`. Overshooting the larger reader is a success. Two-sided equality is a useful calibration
   diagnostic, not a necessary condition for replacing scale as capability.
2. **Policy transplantation:** set the target to the larger reader's sparse verdict channel itself,
   `M = Z_{S,a0}`. Ask whether articulation lets `E_s` reproduce the same item-level signature.
   Here direct fidelity/equivalence is constitutive; equal scalar performance against a third target
   is insufficient because two readers can make different errors at the same aggregate quality.

The phrase **isomorphic performance** must say which of these it means. The first supports the
scale--articulation substitution law; the second supports a claim that tacit policy content has been
successfully externalized and transplanted. Run both, but do not require a capability-improving
prompt to clone an intermediate reader's residual errors, and do not call equal aggregate recovery
policy isomorphism.

### 3.2 Intact articulation is the reachability instrument; units diagnose mechanism

The primary treatment class for finite-debt/reachability claims should include coherent intact
definitions, explanations, rule systems, contrastive examples, rubrics, dossiers, and formative
presentations. An address-segment prefix is a strict sub-class that assumes a fixed ordering and
approximately additive installation. It is therefore a mechanism and cost assay, not the strongest
attempt to externalize tacit knowledge.

Use a nested design on the same frozen target and item split:

`name/frame -> intact channel arms -> composed full specification -> segment/complex ablations`.

Matched inert and wrong-construct controls accompany every semantic arm. If intact text succeeds
while isolated segments fail, the residual is evidence for composition, interaction, or gestalt—not
evidence that explicit articulation as a whole cannot replace scale. Promote fragments to cost units
only after CUF certification; otherwise report address segments and right-censored reachability.

## 4. Gestalt is a measured axis, not leftover error

CUF units are behavioral operators, not factual propositions. This makes them suitable for social
preference, but a bag of units need not contain the gestalt. Use the existing combiner ladder:

```
best single -> additive/linear -> pairwise -> higher degree -> pattern lookup -> composed prompt.
```

- `lookup - linear` is interaction/configural information within the unit span.
- `Delta_comp_beyond` is value carried by joint phrasing/framing beyond separately executed units.
- a held-out span residual asks whether a successful holistic prompt instantiates a missing unit or
  a genuinely different channel.

The parity lemma proves that no finite interaction class gives a class-free ceiling. Therefore every
frontier names `d`; higher-order failure is not folded into an error bar. In taste domains, a large
composition/interaction component is a candidate measurement of gestalt tacitness.

## 5. Conditional bridge theorem: articulation–enculturation duality

The desired universal substitution theorem is available under explicit, falsifiable assumptions.

For each construct, suppose:

1. **Common basis:** the target-relevant distinctions admit a finite basis of CUF-certified units or
   complexes.
2. **Robust identity:** the required units pass U1–U4 over `H`.
3. **Cross-scale semantics:** they are U5 `E-SHARED`, not `E-DRIFT`; the same articulation installs
   the same target-relevant fingerprint along the reader ladder.
4. **Nested enculturation:** the distinctions available from the sparse community/frame prompt form
   nested sets `K_s subset K_S` as capability grows.
5. **Installation:** explicitly presenting a missing unit installs its fingerprint in the smaller
   reader within the certified error band.
6. **Compositional closure:** the smaller reader can combine the installed units at the target's
   required degree `d`.
7. **No residual execution floor:** planted controls show that following/applying the articulation,
   rather than possessing the knowledge, is not the bottleneck.

Then presenting the units/complexes in `K_S \ K_s` to `E_s` realizes the same sufficient target
statistic as `E_S`'s sparse/gestalt readout up to accumulated CUF and estimation error. Hence the
articulation debt is finite. Minimality gives the existing triangle inequality. If unit effects are
additive and the nested differences are disjoint, the triangle is tight and debt is a potential;
redundancy gives sub-additivity, while prerequisites, E-drift, or higher-order interactions produce
localized slack.

This theorem says exactly what would have to be true for scale and articulation to be two sources of
the same effective normative competence. Crucially, CUF and the composition ladder make every premise
empirically attackable.

## 6. The empirical law to seek: curve collapse, not another three-point power law

The strongest elegant law is a single-index or potential structure:

```
F_i(s,a) approximately h_i(kappa_s + a),
```

where `kappa_s` is a reader capability/enculturation potential and `a` is a preregistered articulation
budget over certified knowledge. This predicts:

1. horizontal shifts align the articulation curves across scale;
2. the inferred shift is ordered by a separately measured capability coordinate;
3. direct and composed path costs satisfy a tight or predictably slack triangle;
4. a shift learned on development constructs predicts held-out constructs;
5. the direction replicates in a second genuinely scale-matched model family.

Do not estimate `kappa` and declare success on the same cells. Reuse the OSL planted capability
battery or learn potentials on development constructs and test curve collapse on a lockbox. The
failed three-point Gini power-law preregistration should not be revived; observed staircases,
isotonic frontiers, held-out horizontal-shift prediction, and censoring are better identified.

The residual from curve collapse has scientific content:

- U5 E-DRIFT: the words install different normative functions across readers;
- high required `d`/`Delta_comp`: configural or gestalt tacitness;
- ostensive/formative success after declarative failure: teachable know-how, not propositional
  codification;
- no success within the audited channel horizon: candidate collective/enculturated tacitness;
- planted-control failure: generic execution/capacity, not tacit knowledge.

## 7. What is reusable

| Existing object | Reuse status | Role here |
|---|---|---|
| fixed-target DPI `R <= T(M)` | verbatim | global ceiling for every target view and candidate prompt |
| CR-3 fresh prompt audit | verbatim within its declared process/horizon | expected-best single-prompt upper bound; stress-tests failure to articulate |
| CUF U1–U5 | verbatim, with target/view in the declared tuple | robust knowledge units, form/context charges, cross-scale identity/drift |
| Phi-orbit target and adverse form bands | verbatim | separates stable content from phrasing and target strictness |
| iso-performance debt, censoring, triangle | verbatim after `2*delta` hop correction | horizontal scale/articulation exchange |
| prompt-space combiner ladder and `Delta_comp` | verbatim as estimates | locate gestalt and interaction degree inside the DPI bracket |
| OSL capability battery/gates | reusable coordinate and validity gates | scale axis independent of raw parameter count |
| four functions of language | reusable channel taxonomy | explanation/prediction/formation/regulation are not one treatment |
| old `OPT_Omega + epsilon` ceiling | **do not reuse as a bound** | historical sensitivity only; tail-XOR and audit retracted it |
| count-Heaps/Chao/old alpha scaling | descriptive only | articulation-vocabulary diagnostics, not support or value exhaustion |
| three-point Gini power-law asymptote | rejected | do not use as evidence for the new law |

## 8. What “universal” can honestly mean

1. **Universal measurement law:** the target-indexed frontier/debt applies to every declared target,
   including infinite debt. This is attainable and useful.
2. **Conditional structural theorem:** under the seven premises above, substitution is guaranteed.
   This is mathematically universal and empirically falsifiable premise by premise.
3. **Finite-bank universal reachability:** every preregistered eligible construct/model pair has a
   finite, held-out certified debt within the declared channel/budget. This is testable by an
   intersection-union decision with simultaneous confidence.
4. **Population prevalence:** infer the fraction of social/normative constructs with finite debt.
   With zero failures, roughly 59 independent successes are needed for a one-sided 95% lower bound
   above 95%; roughly 299 for a lower bound above 99%. Domain clustering requires more.
5. **All normative concepts / all language:** not empirically provable from a finite bank. It would
   require the structural theorem plus assumptions strong enough to define the concept universe.

A single constant "units per billion parameters" across constructs is neither implied by the theory
nor especially plausible. A universal frontier form with construct-specific cost and a measured
gestalt/ceiling residual is the stronger scientific claim.

## 9. Experimental sequence

### Experiment N — lexical supplement

Target `M^N_{i,S}` = larger-reader name-only soft behavior. Candidate smaller-reader prompts contain
certified definitions/explanations/rules/examples. This is the clean fixed-target method-validation
experiment and directly tests lexical decompression.

### Experiment G — model-internal gestalt

Target `M^G_{i,S}` = larger reader's holistic policy under a minimal domain/community frame, with no
criterion decomposition. Candidate channels attempt to reconstruct it. Include the full combiner and
composition ladder. This asks whether the larger model's enculturated normative gestalt can be
re-housed in an explicit message for a smaller model.

### Experiment P — social-practice target

Target `M^P_{i,g}` from repeated community/expert decisions or an explicit latent-target confidence
set. All readers recover the same target. Compare native/emic rubrics, machine articulations,
ostension, and formative channels. This is the deeper anthropological claim: what the practice can
transmit by telling, showing, and teaching.

Across all three: same-version within-family reader ladders; second-family replication; source-held-out
probe lockbox; soft target and reliability gates; CUF-certified inventory; inert and wrong-construct
controls; development-only arm ordering; CR-3 audit for the declared prompt process; right censoring;
and planted mechanical/procedural controls.

## 10. Implementation placement

Do not edit or relocate the legacy line. Add:

```
methods/codability/experiments/
    fixed_target_name_substitution.py      # Experiment N
    target_articulation_frontier.py        # target/view-agnostic frontier and debt
    gestalt_substitution.py                # Experiment G + combiner/composition joins
```

The first module can reuse raw grid tensors and `vinfo.fixed_target_channel_certificate`. The generic
frontier should be written only after the target/channel schema is frozen. Existing
`scale_articulation_substitution.py` remains the audited legacy/transition analysis; its schema and
generated artifacts remain unchanged.

## 11. Implementation and first method-validation result (2026-07-12)

The additive layer now exists under `methods/codability/experiments/`:

- `target_articulation_manifest_v1.json` freezes target views, channels, treatment boundaries,
  required provenance, gates, and claim grades;
- `target_articulation_frontier.py` implements target-agnostic adverse-form `R/T`, polarity and
  direct-fidelity gates, paired held-out substitution, dose/frontier records, and matched controls;
- `fixed_target_name_substitution.py` implements Experiment N over the legacy raw grids;
- `common_target_ladder.py` mechanically rejects moving targets and incommensurable triangle costs;
- `gestalt_substitution.py` joins the combiner ladder, composition gap, span residual, and fixed-target
  substitution without upgrading any of those estimates to all-prompt bounds.

The first common-8B-target retrospective pass on humor and math finds frequent directional gains but
no full held-out substitution on 1B -> 3B, 3B -> 8B, or 1B -> 8B. Every eligible debt is right-censored
within the legacy prompt bank; the potential law is not yet evaluable. This is evidence that ordinary
definitions/explanations do not reproduce the larger reader's name-invoked policy, not evidence that
all explicit teaching or gestalt articulation must fail. Full methods, counts, sensitivity, and exact
artifact hashes are recorded in `notes/2026-07-12__fixed-target-name-substitution-first-pass.md`.

## 12. Historical all-task surface atlas update (deprecated)

> This section records a superseded retrospective screen. Mixed-scoring/CW contamination and a
> broken frontier path make its five events non-quotable. It is not evidence for the current
> substitution claim; use the fresh hashed policy pipeline and its explicit errata instead.

The implementation now persists complete per-reader articulation surfaces and aligned bootstrap
draws. Available sk3 generations yield 13 within-family surfaces/14 comparisons and 18 reciprocal
cross-family surfaces/24 comparisons across nine tasks. Cellwise intervals identify five exploratory
3B -> 8B substitutions, including one short humor rubric that repeats under two target families.
None survives per-comparison Bonferroni simultaneous bands, and none has matched-control specificity.
The evidence is therefore a target- and metric-local candidate substitution seam, not a population or
universal law. Full coverage and artifacts are in
`notes/2026-07-12__all-task-fixed-target-name-surface-atlas.md`.
