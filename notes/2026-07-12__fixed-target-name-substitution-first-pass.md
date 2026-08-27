# Fixed-target name substitution: first common-target pass

Date: 2026-07-12  
Status: retrospective held-out method validation; **not preregistered, not paper-grade**

## Question

Can definitions, explanations, rules, examples, or dossiers executed by a smaller reader recover a
larger reader's name-invoked normative policy? This is Experiment N, the lexical supplement to the
broader gestalt/practice program. It is deliberately not treated as all tacit normative knowledge.

## Corrections made before reading the final ladder

1. **One fixed target across scale.** Pairwise targets cannot be composed. The ladder fixes the 8B
   name-only policy while comparing 1B, 3B, and 8B reader baselines.
2. **Target form quotient.** For each metric and item, the fixed target is the mean 8B name-only
   probability across canonical, question, and boilerplate forms. Candidate robustness is the
   adverse candidate form.
3. **Information plus orientation.** `R/T` comes from the surviving fixed-target DPI construction.
   Because MI gives an inverted policy full credit, the primary score multiplies `R/T` by the sign
   of target--candidate covariance and retains direct Spearman fidelity gates.
4. **Held-out selection.** Exemplar items are removed. The remaining probes are split within five
   target-rank strata. The smallest legacy word-count arm reaching the development target is chosen;
   if none reaches it, the highest-recovery arm is retained diagnostically. It is evaluated once on
   the held-out half with 500 paired item bootstraps.
5. **Strong rescue gate.** A result requires a confirmed scale baseline gap, confirmed improvement
   over the smaller name arm, information non-inferiority, positive polarity, an absolute signature
   floor, signature improvement, and signature non-inferiority to the larger baseline.
6. **Honest claim grade.** The old grids have matching dimensions and row metadata but no stored
   cryptographic probe IDs. They also lack matched filler/wrong-construct controls and CUF-certified
   units. Thus even a successful row would remain a retrospective methodological result.

## Frozen inputs and generated artifacts

| Artifact | SHA-256 |
|---|---|
| `fixed_target_name_substitution_1b_3b_v1.json` (moving 3B target; diagnostic only) | `daf5fd980d69ab8a0b3a26494378043e1930745abc19e3fdeb032ed503a2dd1e` |
| `fixed_target_name_substitution_1b_3b_target8b_v1.json` | `2c499192a9753da53fa2e5beb1dde21887d38c687894b40c906b8b00881bed72` |
| `fixed_target_name_substitution_3b_8b_v1.json` | `b650a5eca66fc117fa518f1f1f15a7cb0797e08716f2e1f53292c3edc7a782c1` |
| `fixed_target_name_substitution_1b_8b_v1.json` | `fe5996f497de15b2b4bc314cd9c90f628e8cb504f5492a7b0f169bc9cad4cd2c` |
| `fixed_target_name_common8b_ladder_v1.json` | `fafa5824845ae1d9a5c4745c7225fd7d41069779b4b310506f70a8ef2656ee20` |

The ladder validator confirms one target tag, the same target-grid hash within each domain, the same
per-metric target ID, and the same split seed and held-out size for all 81 common cells.

## Primary result

Counts below are among metrics with a confirmed sparse baseline gap. “Information NI” is held-out
non-inferiority of the articulated smaller reader to the comparison larger reader. “Signature NI”
is the corresponding direct-fidelity gate.

| Fixed-8B-target hop | Domain | Confirmed gaps | Articulation improves information | Information NI | Signature improves | Signature NI | Full methodological substitution |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1B -> 3B | humor | 48 | 44 | 0 | 39 | 0 | 0 |
| 1B -> 3B | math | 0 | -- | -- | -- | -- | -- |
| 3B -> 8B | humor | 54 | 35 | 3 | 3 | 0 | 0 |
| 3B -> 8B | math | 19 | 10 | 0 | 0 | 0 | 0 |
| 1B -> 8B | humor | 57 | 53 | 0 | 46 | 0 | 0 |
| 1B -> 8B | math | 21 | 0 | 0 | 1 | 0 | 0 |

Thus every eligible articulation debt is right-censored within this legacy prompt bank. No common
cell has finite debt on every hop, and the legacy cost is words rather than composable certified
units. The scalar-potential/triangle prediction is therefore **not evaluable**, rather than failed.

The failure is not a knife-edge choice of the primary margins. A post-hoc diagnostic grid varies
information non-inferiority from 0.02 to 0.20, signature non-inferiority from 0.05 to 0.20, and the
absolute signature floor from 0.5 down to 0.3. Only one 3B -> 8B humor metric appears, and only at an
information margin of at least 0.10 and a signature margin of 0.20: “Straight-man/deadpan performance
and contrast,” using its definition. No other hop/domain has a success anywhere in that grid.

## Interpretation

This is a useful negative result for the stronger claim:

- Explicit content frequently changes the smaller model in the target direction. In the common
  1B -> 3B humor comparison, 44/48 improve target information and 39/48 improve itemwise fidelity.
  The treatments are not inert.
- Those changes do not recover the larger reader's name-invoked policy. Especially in math, the
  explicit messages can improve an operational criterion while remaining unlike what the 8B reader
  does from the institutionalized name. This is compatible with the earlier names-beat-definitions
  result.
- The earlier operational-target rescue results are not contradicted. They ask whether articulation
  recovers an operational reference; this experiment asks whether it recovers a larger model's
  culturally invoked name policy. Target choice changes the scientific claim.
- The result does **not** show that the residual is unarticulable. The tested bank contains one-shot
  legacy definitions/explanations/rubrics/examples/dossiers, not residual-targeted teaching,
  CUF-certified units, dialogue, or the full gestalt/composition ladder.

The current evidence therefore supports a sharper statement: ordinary explicit descriptions often
move smaller readers toward the larger reader's lexical policy, but do not replace scale for that
policy under direct behavioral fidelity.

## Next experiment: residual-targeted and gestalt articulation

The next bank should be generated and frozen before its lockbox is scored:

1. Fix either an 8B holistic community-frame target (`M^G`) or a human/community practice target
   (`M^P`); retain the name target as Experiment N.
2. Separate **telling** arms (writers never see target outcomes) from **teaching/formation** arms
   (a teacher sees development disagreements and externalizes recognition lessons) and fitted prompt
   optimization. Never pool these provenance classes.
3. Mine candidate behavioral units from development disagreements, then require CUF U1--U5,
   form/context robustness, cross-scale identity, matched filler, and wrong-construct controls.
4. Execute declarative, procedural, ostensive, formative, additive, pairwise, and composed arms.
   Use `gestalt_substitution.py` to locate interaction within the unit span, composition beyond
   separately executed units, and target-aligned behavior outside the span.
5. Select prompts and articulation budgets on development/certification items; score one frozen arm
   per declared budget/channel on a source-held-out lockbox with cryptographic probe IDs.
6. Only after finite debts exist on common targets should the OSL horizontal-shift and triangle/
   potential predictions be tested.

This is the path by which a future positive result could support actual scale--articulation duality
rather than mere prompt improvement.

