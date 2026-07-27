# All-task fixed-target name surface atlas

> **DEPRECATED / DO NOT QUOTE FOR SUBSTITUTION CLAIMS.** This retrospective atlas is preserved as
> an audit artifact, but it has been superseded by the fresh hashed, form-consistent,
> fixed-target pipeline and its frozen residual-lockbox result. Three of the five events listed
> below involve the known mixed-scoring/CW contamination, and the legacy frontier-report path is
> not executable. The descriptive census may still be used only with these limitations stated.

Date: 2026-07-12  
Status: deprecated retrospective atlas; exploratory only; no quotable substitution events.

## Scope and architecture

The sk3 census recovered and checksum-synchronized the raw grids needed to scale beyond the
humor/math pilot:

- Llama 1B/3B/8B on nine tasks and 267 metrics;
- Llama 70B on eight tasks (all except grant), 251 target candidates;
- Qwen2.5 1.5B/3B/7B on creative writing and math, 67 metrics;
- Gemma 2B/9B/31B on humor and math, 81 metrics;
- legacy cumulative segment grids for humor and math, retained for the next unit-specific pass.

Every legacy grid has canonical/question/boilerplate content forms but no cryptographic probe IDs.
The segment grids have three forms for content rungs and only canonical filler rungs, so they cannot
support a matched-form specificity certificate without regeneration.

`fixed_target_surface.py` evaluates every executor/rung once against a frozen target form quotient,
using one deterministic development/held-out split and shared paired bootstrap indices. It persists
point recovery, polarity, fidelity, and 500 raw bootstrap draws. Pairwise comparisons are derived
from these surfaces rather than recomputed independently. Target-uninformative metrics are marked
ineligible, not errors or tacit failures.

The vectorized TVD bootstrap matches the original pointwise calculation exactly. Point Spearman is
the exact held-out midrank statistic; uncertainty is a paired item bootstrap of those fixed ranks.

## Coverage

| Atlas | Surfaces | Comparisons | Evaluable comparison rows | Confirmed gaps |
|---|---:|---:|---:|---:|
| Within-family/common target | 13/13 | 14/14 | 2,184 | 1,543 |
| Reciprocal cross-family | 18/18 | 24/24 | 1,651 | 861 |

Target-informativity excludes 35/251 metrics under the 70B target and 9/81 under Gemma-31B. News and
specialized institutional-format metrics are overrepresented. A constant name policy has no
definable articulation debt.

## Within-family/common-target results

Counts after “gaps” are conditioned on a confirmed sparse scale gap.

| Comparison | Gaps | Information improves | Information NI | Signature improves | Signature NI | Cellwise substitutions |
|---|---:|---:|---:|---:|---:|---:|
| Llama 1B -> 3B, target 8B | 201 | 104 | 1 | 84 | 0 | 0 |
| Llama 3B -> 8B, target 8B | 216 | 123 | 9 | 11 | 0 | 0 |
| Llama 1B -> 8B, target 8B | 245 | 121 | 0 | 96 | 0 | 0 |
| Llama 1B -> 3B, target 70B | 89 | 34 | 1 | 42 | 0 | 0 |
| **Llama 3B -> 8B, target 70B** | **80** | **49** | **12** | **26** | **8** | **3** |
| Llama 8B -> 70B, target 70B | 150 | 91 | 0 | 42 | 0 | 0 |
| Qwen 3B -> 7B, target Qwen-7B | 65 | 37 | 0 | 3 | 0 | 0 |
| Gemma 9B -> 31B, target Gemma-31B | 56 | 26 | 1 | 22 | 0 | 0 |

Qwen 1.5B -> 3B has zero confirmed sparse gaps against Qwen-7B. Parameter order is therefore not a
valid substitute for empirical capability order even in the cleanest same-version ladder.

## Reciprocal family results

No Qwen or Gemma executor arm substitutes for a Llama-8B or Llama-70B self baseline. Conversely,
Llama 3B -> 8B produces one cellwise substitution under Qwen-7B and one under Gemma-31B. All other
reciprocal comparisons have zero.

This is directional evidence, not a family ranking. Under Llama-8B, Gemma-31B articulation improves
information on 47/57 confirmed gaps and is information-noninferior on 9, but none is signature-
noninferior. Under Gemma-31B, Llama-8B improves 27/56 and is information-noninferior on zero.

## Five exploratory substitution events

All occur at the Llama 3B -> 8B executor seam:

| Target | Task/metric | Selected articulation | Words |
|---|---|---|---:|
| Llama-70B | humor #23, Laugh density and economy | definition | 19 |
| Llama-70B | humor #49, Wordplay quality and clarity | full rubric | 21 |
| Llama-70B | press releases #8, Plain language/readability/minimal jargon | dossier_v2 | 335 |
| Qwen-7B | creative writing #27, Pitch/query and synopsis effectiveness | dossier | 371 |
| Gemma-31B | humor #49, Wordplay quality and clarity | full rubric | 21 |

Each confirms the sparse gap, information and signature improvements, and non-inferiority to the 8B
sparse baseline on both reads. “Wordplay quality and clarity” repeats with the same short rubric
under two targets. It is the highest-priority fresh-probe replication, but not independent yet:
executor outputs and probe items are shared.

## Multiplicity and claim status

The five events use ordinary 95% per-cell intervals. `surface_comparison.py` also computes
Bonferroni simultaneous intervals over every declared metric cell in each comparison. **Zero of
five survives.** Five hundred draws also limit extreme-tail resolution.

Licensed claims:

- five exploratory methodological substitutions;
- a target- and metric-local candidate 3B -> 8B substitution seam;
- broad articulation usefulness below full substitution.

Not licensed: articulation-specific substitution (no matched controls), bank-wide confirmation,
population prevalence, or a universal law.

## Articulation frontiers below substitution

| Target / executor | Metrics | Confirmed information gain | Confirmed signature gain |
|---|---:|---:|---:|
| Llama-8B / Llama-1B | 267 | 129 | 100 |
| Llama-8B / Llama-3B | 267 | 151 | 17 |
| Llama-8B / Llama-8B | 267 | 104 | 1 |
| Llama-70B / Llama-1B | 216 | 65 | 77 |
| Llama-70B / Llama-3B | 216 | 100 | 63 |
| Llama-70B / Llama-8B | 216 | 114 | 58 |
| Llama-70B / Llama-70B | 216 | 51 | 3 |

Direct signature gains decline near the target reader because its sparse name policy already defines
the target. Rich prompts can increase information while redirecting item ordering; the fidelity gate
is therefore indispensable.

## Scientific interpretation

1. There is a candidate substitution seam, not a universal law. The 1B reader often moves correctly
   but does not reach the next baseline; 8B does not replace 70B.
2. Target choice changes reachability: the same 3B -> 8B ladder has zero substitutions under 8B and
   three under 70B. Debt must remain target-indexed.
3. Information gain is much easier than policy recreation. AUC or unsigned MI alone would overstate
   substitution.
4. Reciprocal family targets are required to distinguish capability from E-DRIFT.
5. Legacy words are not a currency. Positive arms span 19--371 words and several channels. The
   potential/triangle law still requires CUF-certified composable units or complexes.

## Next confirmatory work

1. Fresh-probe replication of the four unique positives, led by humor #49.
2. Three-form matched filler and wrong-construct controls at selected length/channel.
3. At least 5,000 draws or analytic/randomization inference for simultaneous bands.
4. Residual-targeted teaching arms, separate from source-only telling and fitted optimization.
5. CUF U1--U5 and cross-scale fingerprint identity for installed units/complexes.
6. Execute the gestalt target, then a human/community practice target.

## Final validated artifact hashes

| Artifact | SHA-256 |
|---|---|
| `name_surface_atlas_manifest_v1.json` | `d14e1fac4b4f59fee4d0695b1e667a0e505573a5431b39a261325dd684f8c3f4` |
| `cross_family_surface_manifest_v1.json` | `a800c7522eaac06ea3491dc33ec6358859f068a4805a863c7cae04dc190093fb` |
| `residual_teaching_manifest_v1.json` | `de033813b7f05cbe4e6f13de998707964f130efdc6feab4095cb02cc31b417f1` |
| `gestalt_execution_manifest_v1.json` | `c942b86755bc8474c7ad34fbf3b5d286462ec925509a1b2917f8e6e688f46f0d` |
| within `atlas_summary.json` | `dfcbc261344994fbbcdb2a19e1756de248ef43c68d98987fd8a3364bda74df9b` |
| `frontier_report.json` | `82a82c2fc6ccf7e53c8aabdf77c7547759ce4d09944b9281d7e0ff3c0004cbb1` |
| cross-family `atlas_summary.json` | `12e3c1f3a3e9a435f5640edd0226f2fe2a4e5953d1273c6fd670e199d08b9836` |
| joint integrity certificate | `fdc4c69e4250c21c937bf7c9744d489d36b963970aad9b59c881de2c5d33aee0` |

The joint certificate re-loads every NPZ, checks its JSON sidecar and bootstrap dimensions, verifies
all referenced hashes and target-valid comparisons, and independently recomputes coverage and
substitution totals. It passed with no errors or warnings.
