# Fresh N/G/P confirmation: freeze, target health, and public queue

Date: 2026-07-12  
Status: fresh target execution and public N-arm development are complete; public arm selection is
frozen; **no smaller-reader lockbox execution has occurred**.

## What changed after the retrospective atlas

The retrospective atlas isolated five cellwise candidates but zero familywise discoveries. It also
lacked probe hashes, matched controls, fresh source groups, and certified units. The fresh phase now
separates three targets rather than treating construct-name behavior as tacit knowledge in general:

- `N`: large-model construct-name policy, for direct replication of the lexical experiment;
- `G`: large-model holistic normative judgment with no construct name, criterion label, rubric, or
  enumerated quality list;
- `P`: archival community preference or external practice outcome, stored separately from text and
  never treated as model truth.

This preserves the old experiment and adds the broader gestalt/social-preference experiment beside
it.

## Fresh item freeze

The sk3 packet contains 8,000 items over four domains and passed an independent read-back
certificate with no errors:

| Domain | Items | Holdout grade |
|---|---:|---|
| humor | 2,200 | exact-item disjoint after corpus deduplication; **not** source-group disjoint |
| creative writing | 2,200 | WritingPrompt-group disjoint |
| press releases | 2,200 | native company-grouped train/validation/test split |
| math | 1,400 | Math.SE-question-group disjoint |

The compiler reconstructs and excludes the legacy seed-7 60+300 probe sample on the exact hashed
dataset revision. No text hash or recoverable source group is reused across partitions. Item text and
practice targets live in separate JSONL files. The integrity validator recomputes every text hash,
dataset hash, source split, partition hash, target alignment, and legacy exclusion.

The distinction between item-held-out and source-held-out is now machine-readable. In particular,
humor results cannot be described as source-group held out.

## Practice targets P

The sealed archival targets are healthy and close to balanced:

| Domain | Target provenance | Overall positive rate | Gestalt-lockbox positive rate |
|---|---|---:|---:|
| humor | stratified Reddit audience-preference proxy | .506 | .488 |
| creative writing | WritingPrompts community-preference proxy | .501 | .500 |
| press releases | news-pickup outcome proxy, **not** a professional rating | .453 | .463 |
| math | Math.SE community answer-quality preference | .488 | .517 |

These are normative/practice proxies, not factual labels. Press-release pickup is an external outcome
and must not be rhetorically upgraded to community judgment.

## Name-target N health already observed

All completed fresh name targets are informative on every partition. Aggregate results from the
first independent launch:

| Target/cell | Items | TVD target information | Mean form flip rate | Max form flip rate |
|---|---:|---:|---:|---:|
| Qwen-7B, CW #27 | 2,200 | .347 | .102 | .118 |
| Gemma-31B, humor #49 | 2,200 | .432 | .055 | .100 |
| Llama-70B, humor #23 | 2,200 | .368 | .056 | .070 |
| Llama-70B, humor #49 | 2,200 | .401 | .057 | .089 |
| Llama-70B, PR #8 | 2,200 | .325 | .050 | .063 |

Qwen's independent-engine repeat is exact to numerical precision (MAE `1.7e-12`, zero binary
flips, Spearman 1.0). Gemma's repeat is also extremely stable (MAE .00122, flip rate .00136,
Spearman .99995). Thus Qwen's roughly 10% variation is prompt-form sensitivity, not engine noise;
the form quotient is load-bearing.

Gemma-4 requires the dedicated `gemma4` environment (`transformers 5.12.1`, `vllm 0.23.0`). The base
environment rejected a checkpoint-only `layer_scalar` before rendering any prompt. No model or
checkpoint substitution was made.

## Clean G instrument

The first mixed target manifest reused the generic N wrapper, which says “one specific criterion.”
That is valid for N but contaminates a strong non-name gestalt claim. It was caught **before any G
execution**. G now has its own holistic-question template and three exact frames per domain:

1. minimal intended-community judgment;
2. veteran-practitioner whole judgment;
3. naturalistic reception/decision frame.

The clean G manifest contains no priority construct names, criterion wrapper, rubric, or enumerated
quality list. Its outputs will be analyzed separately from N and P.

## Source-only arms and specificity controls

Before any fresh smaller-executor outcome, the name experiment froze 64 arms across four unique
priority metrics:

- sparse name;
- source definition, explanation, full rubric, corrected ostensive examples, and corrected dossier;
- for every content arm, a three-form exact-content-word-count inert control;
- for every content arm, a three-form exact-content-word-count wrong-construct control selected from
  the same domain/channel without using target outcomes.

The public selector maximizes the 5,000-draw lower bound of adverse-form oriented recovery among
source-only arms with positive polarity and a held-out signature floor. Within .01 it chooses fewer
content words. This selects a confirmation candidate; it does **not** declare raw words to be CUF
units or a minimum certified debt.

## Lockbox isolation and active queue

The original immutable target matrices are now sharded by declared partition. Selection code accepts
only `residual_prompt_selection` shards. It cannot read the original all-partition matrix or any
lockbox shard. Aggregate target-health checks were permitted, but no item-level lockbox target or
executor result has been inspected.

The guarded sk3 queue performs, in order:

1. finish and shard the second Llama-70B N launch;
2. execute and shard two clean G launches;
3. execute Llama-8B sparse and Llama-3B articulated public development;
4. shard executor outputs;
5. freeze the 5,000-draw public selection artifact;
6. stop.

It contains no lockbox executor command. The live log is
`logs/fresh_public_queue_v1/queue.log` on sk3.

## Frozen hashes

| Artifact | SHA-256 |
|---|---|
| fresh item-partition protocol | `06437cb0b42343c7cee84b6b43ec3a4fef7e62b1f2a79916ca1b41df19ddf6a0` |
| sk3 packet manifest | `9f2b01d85ffaa5f88718ee3733bd2a0049d892c5f8311a7e7984b8324616928a` |
| sk3 packet integrity certificate | `7f56eb90aea6e9009fedef5c3dda5a2ccad0db2427571e55f420bc20d898122f` |
| Qwen/Gemma N target manifest | `e49f153d8dd91de7a4e3125940992e03f294a51749004d9368a33c0916aec4d2` |
| Llama-70B N-only target manifest | `dd15a571aa7c215d93189c1d6262db016f48d6cef143350820d90e1a6287d33f` |
| clean G target manifest | `6a10bf3a638a9c79cc39be1913d3d251e6bc88becf37d9e6de0f5d245434f461` |
| source arm bank | `2358800875b276317e41d64dbd7cf02886d80e41713850fcacba17bc4c29961d` |
| executor/phase manifest | `567cce9cdee6f5f8ee129a661b43ff9409bf9aaccaab68dc24b8a037f19fdc18` |
| practice-target aggregate report | `bd5fffecc2749d7dba11cba95bd68eae83962bd8dbfedc8b32a39edb65c58139` |

The target-health report hash is intentionally not frozen here until the active target repetitions
finish.

## Remaining claim gates

- fresh 3B-versus-8B public selection and untouched lockbox substitution;
- simultaneous inference across all five target-by-metric cells;
- fresh U1/U2 analysis and genuine U3 position/company, U4 minimality/interaction, and U5
  cross-scale identity;
- clean G executor articulation bank and lockbox;
- G-to-P relationship on the separate practice targets;
- a same-version within-family scale ladder and second-family replication;
- CUF curve collapse, finite-hop triangle/potential analysis, and right-censored failures.

Until those pass, the licensed conclusion remains: informative, reliable target instruments and a
fully executable confirmatory design—not a universal scale–articulation law.

## Public N-arm development result and audit (later 2026-07-12)

The expensive public scores completed. The first selection attempt stopped before writing its
artifact because the GPU checkout lacked `target_articulation_frontier.py`; copying the already
tested analysis module repaired that non-GPU step without rerunning a model or opening a lockbox.
A second integrity fault was then found: N and G jobs using the same Llama-70B checkpoint emitted
identical raw basenames, and the flat partition sharder let the later G shard overwrite N. The
sharder is now namespaced by `model_job_id` (`partition_sharded_score/v2_job_scoped`), the selector
recurses through that namespace, and N targets were rebuilt in a separate shard root. A regression
test reproduces the old collision and verifies both target views survive.

Frozen public selection artifact:
`fresh_name_arm_selection_v1.json`, SHA-256
`e0a8c6c3bb6e54eddcce70d35fc9484de470604c69cbfd23165216c431a2b7b1`.
Selected source arms were definition (humor #23), explanation (humor #49/Llama target), full rubric
(humor #49/Gemma target), and dossier-v2 (CW #27 and PR #8). All five selected arms had a positive
public articulation gain. Four cells had a confirmed 8B-name over 3B-name baseline gap; PR #8 did
not and is therefore not substitution-eligible.

The original decision code compared each arm's recovery and target-correlation with the independent
N target, but did not directly compare the articulated 3B item signature with the sparse 8B
signature. That is inadequate for a literal policy-isomorphism claim. The v2 certificate adds an
adverse-form-pair direct Spearman interval and requires its lower bound to clear the fidelity floor
and to improve over 3B-name versus 8B-name similarity.

Exploratory all-source public atlas:
`fresh_name_arm_public_atlas_v2_direct_signature.json`, SHA-256
`eb209759315d6bd364b37f4e68fb85e0cb63305b3e693110fe14bb7c31a1d8af`.
Across 25 source-arm × target cells, 20 were baseline-gap eligible and 16/20 showed a confirmed
target-recovery gain. Four eligible arms were within .02 of the 8B recovery target at the point
estimate and 13 were within .05, but **none** had a two-sided .02 equivalence interval. Six cleared
the direct-signature floor; none showed a confirmed direct-signature gain over the already-high
3B-name/8B-name similarity. Thus intact text frequently changes target recovery and can reach point
parity, but this public panel does not show item-level policy transplantation. The .02 equivalence
test is also severely precision-limited at 200 public items; its margin and power must be calibrated
before freezing a lockbox claim.

This result fixes the experimental hierarchy: intact source text is the primary reachability arm;
address-segment ladders are secondary dose/additivity diagnostics. Capability substitution against
an independent target and transplantation of a larger reader's own policy are reported as distinct
estimands. No public result is confirmation. The separate 400-item `residual_lockbox` for direct
policy isomorphism was subsequently opened and is reported in
`2026-07-12__isomorphism-first-tacit-policy-reconstruction.md`; the fresh N/G/P confirmation
lockboxes, including `gestalt_lockbox`, remain sealed.
