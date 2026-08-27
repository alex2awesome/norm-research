# C_expl pilot: the no-keyword explanation channel (taboo game)

**2026-07-14. Pilot spec. Companion to `2026-07-14__future-work-channel-battery-and-acquisition.md` (idea)
and `2026-07-13__v14-decoder-tuning-roadmap.md` §8 (the ladder this extends). Runs AFTER roadmap Phases A-B
land; shares their frozen designs, panels, H, executor, and controls.**

## 0. Question

How much of a criterion transmits through *paraphrase* — explanation that cannot lean on the criterion's own
vocabulary? This separates two things C2 conflates: carrying the metric's **lexical handle** (taste = index;
the name suffices) from carrying its **content** (craft = decompression). Per-metric,
`C2 − C_expl_banned` is a language-tacitness readout; `C_expl_free − C_expl_banned` is the share of
transmission riding on the keywords alone.

## 1. Design (one trial = one metric x one panel)

Chain (every arm bottoms out in executed verdicts — explanation quality is never judged directly):

    ENCODER A:  sees the metric (frozen description forms + the 8 labeled demos)
                -> writes an explanation E, with the metric's BANNED WORDS excluded (see §2)
    DECODER B:  sees ONLY E (never the description, never the demos)
                -> induces a rule M_hat (same induction template family as C2)
    EXECUTOR:   frozen Llama-3.1-8B executes M_hat on held-out H (same H as the ladder; |H|=240 per §8.5)
    VALUE:      I(M_omega|_H ; y_hat) − max(blind, shuffled), permutation z-score alongside

Arms (all four per metric; encoder/decoder prompts frozen before any scoring):

| arm | encoder input | ban list | isolates |
|---|---|---|---|
| `expl_banned` | description + demos | ENFORCED | content-only paraphrase transmission (headline) |
| `expl_free` | description + demos | none | paraphrase ceiling; `free − banned` = lexical-handle share |
| `expl_demos_only` | demos only (no description) | ENFORCED | can A *articulate from behavior* then transmit? (composes C2's induction with the paraphrase bottleneck) |
| `blind` | task noun only, no metric | — | control (subtract) |

Shuffled control: E explains a DIFFERENT metric from the same task (deterministic derangement of metric
keys); B and the executor run identically. Both controls per panel, deterministic seeds, as everywhere else.

## 2. The ban list (this is where codability plugs in)

Per metric, banned = union of:
1. **Description content words:** every content token (non-stopword, lemmatized) in ALL frozen orbit forms.
2. **Census lexicon terms:** the metric's top-k terms from the metric-lexicon codability census
   (`project_metric_lexicon_census` instruments; the per-metric lexicons behind the exact-Jaccard dialect
   measures). k=15 default, frozen per metric before any generation.
3. Inflections/derivations of 1-2 (stemmer closure).

Enforcement: string/shingle check on E (same mechanism as the no-verbatim arm); on violation regenerate with
the violating terms echoed back, max 3 attempts, then that trial is VOID (fail closed, recorded, never
silently retried into compliance). Report the void rate per metric — a high void rate is itself a codability
signal (the criterion cannot be discussed without its own words).

## 3. Anchors and gates (standing rules apply)

- **Anchor rows in every encoder batch:** planted mechanical criteria (reuse `PLANTED_CRITERIA` from
  `ceiling_ladder.py`) that ARE explainable without their keywords — e.g. "contains the ASCII character ?"
  explained as "the text puts a direct query to its reader." The planted anchors must certify positive value
  through the full banned chain, else the instrument (not the metrics) failed — halt.
- Shuffled control must land ~= 0 across the pilot; if not, the explanation channel is leaking task-generic
  quality priors and the blind control needs strengthening before fan-out.
- Judge-free: no step scores "explanation quality"; only executed verdicts enter value.

## 4. Population, models, cost

- Pilot population: the 6 humor sentinels + 5 peer-review + 5 code-review Tier-B metrics (reuse frozen
  ladder designs; zero new panel work). 16 metrics x 4 arms x (1 encoder gen + 1 induction + |H| executions).
- Encoder A: Llama-3.3-70B local (generation, not judging — no Sonnet requirement). Decoder B: the ladder's
  qualified decoder families; pilot on one (Qwen2.5-14B or 32B). Executor: frozen 8B, unchanged.
- Cost: ~16 x 4 x ~250 executor queries + ~130 generations ~= ONE evening on the 2 free sk3 GPUs, sharing
  the resident models already up for the ladder. CPU aggregation reuses `ceiling_ladder.py` statistics
  (`value_against_controls`, `permutation_pvalue`, z-scores).

## 5. Predictions (declared now, before any data)

1. `C_expl_banned <= C_expl_free <= ~C0` per metric (the executor remains the outer bottleneck).
2. `C_expl_banned` correlates POSITIVELY with the metric's codability level from the L0->R3 hierarchy work
   (chance-corrected kappa ladder) and with census lexicon dialect measures — articulable-by-our-other-
   instruments metrics should paraphrase well. Spearman, threshold-free, reported with n.
3. The lexical-handle share `free − banned` is LARGEST for taste-flavored metrics (name suffices) and
   smallest for mechanical/craft metrics — the what-gets-decompressed prediction, now measured in bits.
4. Void rate correlates negatively with codability.

If prediction 2 holds even moderately, C_expl becomes an independent-instrument validation of the whole
codability program — two entirely different measurement chains agreeing on which criteria are articulable.

## 6. Non-triviality upgrades **[added 2026-07-14 after user review — the game must not be winnable cheaply]**

The naive version is trivially gameable: A writes a thin synonym-swap of the description ("clarity" ->
"lucidity") and B round-trips it. Three upgrades close this:

1. **Ban-list closure through the hierarchy.** Extend the banned set beyond the metric's own words to the
   lexicon of its R2 PARENT cluster (siblings share vocabulary — the sibling-lattice work showed shared
   vocab is domain-dependent but real). Synonym-swapping within the cluster dialect then also violates.
   Declared as a second severity tier: `banned_self` (headline) and `banned_cluster` (hard mode); report
   both.
2. **L0->R3 matching readout (uses existing machinery, no new instrument):** embed/match each explanation E
   back against the full task-local bank with the SAME matching pipeline used for silver->metric matching
   (rerank + abstain). Two numbers per metric: (a) does E match back to its OWN metric (top-1 hit rate) —
   identification-through-paraphrase; (b) does E's induced rule M_hat land in the right R2 cluster when
   scored behaviorally against the bank's orbit signatures. (a) is lexical-ish; (b) is behavioral; their
   divergence is itself the taste/craft split measured a second way.
3. **A stronger null than shuffling:** the "eloquent vacuity" control — A is prompted to write a
   maximally plausible-sounding explanation for the task WITHOUT seeing any metric. If B extracts
   positive value from THAT, the channel is measuring task priors, not transmission (this is the taboo
   analog of the blind-menu prior, and it must be subtracted, not just checked).

## 7. Unit-code merge (exact capacity accounting) **[the version worth certifying]**

Replace free-text E with a SUBSET OF CERTIFIED UNITS from the CUF bank (2,355 units): A selects k units;
z = the unit-ID set; B recomposes a rule from the unit texts; execute as usual. Capacity is then EXACT:
H(z) <= log2(C(|bank|, k)) — the demo channel's clean accounting, for a symbolic compositional code.
Required hygiene BEFORE any capacity claim (each is a declared gate, not an assumption):

- **No concept-word leakage:** units used for metric m must pass m's ban list (both tiers). Units that name
  the metric are the lexical handle smuggled back in — exclude, and report how many survive (a codability
  statistic in itself: "can this metric be assembled from vocabulary-disjoint parts?").
- **Non-overlap / no over-expression:** units must be pairwise behaviorally orthogonal within the selected
  set (reuse `family_coherence` / orthogonalize machinery + the U2 paraphrase-identity census — a unit
  that is a paraphrase of another contributes 0 marginal capacity but the naive log2(C(n,k)) counts it).
  Price effective capacity honestly: report both nominal log2(C(n,k)) and an empirical effective capacity
  from the unit co-occurrence/redundancy structure (entropy of the selected-set distribution under the
  bank's measured unit-unit MI). If effective << nominal, say so in the certificate.
- **Selection is the code:** A's selection must be a deterministic function of (description + demos) at
  temp 0 — otherwise z is not a code, it is a sample.

Three-channel comparison this enables, at matched-or-accounted capacity: demos (k bits, non-symbolic) vs
unit-set (symbolic, compositional) vs free explanation (unbounded, unaccountable). If unit-set ~= free
explanation, articulation is compositional over our units — direct evidence the unit theory carves the
right joints. If unit-set << free explanation, the residual is what free language carries beyond our
units — a measured upper bound on what the unit bank is missing.

## 8. Non-goals / guards

- NOT a certificate channel yet: pilot is diagnostic; no c/r bounds, no prereg claims. Promotion to the
  battery requires the anchor gate + shuffled~=0 + eloquent-vacuity~=0 + a stable readout across the 3
  pilot tasks.
- Never compare C_expl across decoder families (same-family rule); never quote bits without accuracy and z.
- No human studies; all measurement by the frozen executor (LLM-judges-do-all-measurement stands).
