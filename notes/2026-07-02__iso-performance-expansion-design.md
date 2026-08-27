# Iso-performance expansion chains — design, first read, and launch

*2026-07-02 PM. User-approved and LAUNCHED (evening_gpu7_chain.sh, sk3 PID 2574425, 20:24 PT).
Formalism lives in `2026-07-02__two-faces-theory.md` §2.4 (expansion cost x*, triangle inequality,
transitivity = potential structure); this note is the operational record.*

## The question (user, 2026-07-02)

Vertical reader gaps confound knowledge content with generic reader deficits. The horizontal
quantity is the honest one: **how much must the message be expanded for a weaker reader to match
what a stronger reader extracts from a poorer message?** And is that cost ADDITIVE along the
capacity ladder (C→B→A) — i.e., does articulation debt behave like a potential (uniform layers,
fixed exchange rate against capacity), or is background knowledge non-nested (composition
overshoots; slack maps prerequisite structure)?

## Implementation — `methods/codability/run_expansion_chain.py` (7 unit tests green, fake smoke green)

1. **Nested monotone chains** (L0..L7): L0 = metric name; each level APPENDS ~25 words of a fixed
   articulation TYPE — definition → mechanism → procedure → boundary → worked_example →
   counterexample → checklist. Nesting makes the dial scalar and composition well-defined. Writer =
   70B, label-blind (name+rubric only; invents its own examples — no probe excerpts, no leakage).
2. **δ-matching with censoring**: match levels at δ ∈ {0, .02, .05} + paired-bootstrap match
   probabilities (B=500, shared probe resamples); never-matchers right-censored; Kaplan–Meier-style
   median costs + matched fractions.
3. **Planted controls** (programmatic gold: question-mark presence, quoted dialogue, >150 words,
   second-person address): each reader's first level reaching 0.9 = its instruction-following
   floor, measured separately from knowledge content; imbalanced-gold flags.
4. **Type-tagged marginal gains**: gain[type × reader] matrix (planted excluded); model-general
   claim = rank-stability of type orderings across the ladder. Reversed-schedule arm (4 gis/domain)
   controls type-position confounding.
5. **Transitivity**: readers 1B → 3B → 70B (8B excluded: reference executor). Per item:
   h_mid = x*(3B→70B, 0); composed = x*(1B→3B, at level h_mid); direct = x*(1B→70B, 0);
   slack = composed − direct ≥ 0 (nesting ⇒ triangle inequality). Slack location + the increment
   types spanning it = the prerequisite-structure read.

**Selection** (from the v1-grid horizontal read, D1-filtered): CW 18 metrics (8 rescued bars
.55–.65 + 10 censored bars .69–.76), reverse arm {2,6,22,25}; humor 20 (10 rescued .56–.70 + 10
censored .74–.85), reverse arm {7,38,41,50}. Full lists + rationale:
`notebooks/data/two_faces_20260702/expansion_selection.json`.

## First horizontal read (v1 grid as a coarse dial — notebook §4b)

- **Humor: expansion substitutes for capacity.** Cumulative 1B-matches-3B@name: 25% → 38% → 52% →
  55% → 62% along name→definition→explanation→rubric→dossier; 38% censored.
- **CW: capacity floor.** Flat 24–26%; **74% censored at every verbal rung**; 1B best-over-rungs
  median .579 < 3B@name median .634.
- **δ-sensitivity (robust):** censored fraction at δ=0/.02/.05 — CW 74/74/54%, humor 38/35/23%.
  The contrast never inverts.
- **Cross-face coherence (new):** in humor, form-gate PASSING metrics are overwhelmingly the
  MATCHED ones (16 matched vs 3 censored among gate-passers; gate-failers split 17/18) — the
  language-stable concepts are exactly the ones whose content decompresses into words for a small
  reader. In CW censoring dominates regardless (gate-passers: 5 censored vs 2 matched).
- **Failure-mode separation:** 1B `self_agree` ≈ 0.9+ at near-chance accuracy = stable but
  non-discriminative verdicts (signal-extraction failure, NOT format non-compliance) — plus the
  planted controls in the new run measure the compliance floor directly.
- Diagram (real measured cells): `notebooks/figures/iso_performance_expansion.html` — humor
  "Callback craft and deployment": 1B locked out at the name, **exceeds** the 3B's name-read once
  given the 37-word explanation (.715 vs .598); CW "Setting as active force shaping story":
  3B@name .762, 1B censored at .662 best.

## Predictions (registered before the chain run reports)

1. Humor near-additive (slack ≈ 0 where uncensored); CW mostly censored at the 1B end — the
   interesting slack lives in 3B→70B vs 1B→70B on humor + the rescued-CW subset.
2. Marginal-gain type ranking: mechanism/procedure > definition for 1B–3B; worked_example gains
   concentrated in humor (consistent with exemplars transmitting in humor only).
3. Planted floors at L0–L1 for all readers (instruction-following is not the binding constraint).
4. Gate-passing metrics cheaper to expand (lower KM median) than gate-failers, both domains.

## Run plan / status

evening_gpu7_chain.sh (GPU7, VLLM_GPU_MEM_UTIL=0.85): phase 1 = Task-#8 70B reader on BOTH v1
grids + report regen (the 70B−3B clean-gap dynamic range); phase 2 = chains written (70B writer);
phase 3 = 70B chain reader; phase 4 = 1B/3B chain readers; phase 5 = expansion reports.
Est. ~13–16h total (≈ done midday 2026-07-03). Status: `outputs/evening_gpu7.status`; logs
`outputs/r3_{cw,humor}/expansion_v1.log`; local watcher armed on milestones.

## Prior-work scan (quick pass; deep sweep pending)

Adjacent clusters exist: prompt-optimization-for-small-models practice; CoT×scale interactions
(CoT fails to transfer to small models); prompt-specificity dose–response (e.g. "More Than a
Score", arXiv 2508.03678: 14B code-gen saturates by ~100 words of spec); verbosity-compensation
(2411.07858); linguistic-properties-of-prompts (2311.01967). None formalize the horizontal
iso-performance expansion COST per concept, none measure it against a fixed external reference on
a capacity ladder, and none pose composition/transitivity. The formalized quantity appears novel.

## Caveats

Type confounded with position (reversed arm is a partial control); form reformulations of chains
preserve content-nesting, not literal prefix (canonical form does); greedy decoding = deterministic
replicas (no sampling noise, but no draw-variance either); 8B unavailable as a reader (reference);
v1-grid "first read" uses coarse type-rungs as the dial, not the nested chains.

Related: [[two-faces-theory]] §2.4, [[project_humor_vs_cw_crossdomain]],
`2026-07-02__humor-vs-cw-crossdomain.md`, notebook
`notebooks/2026-07-02__two-faces-results-summary.ipynb` §4b.
