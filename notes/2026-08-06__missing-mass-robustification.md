# Robustified missing-mass battery (M1 + M2 + M3) on the peer-review verdict cell

Date: 2026-08-06. Status: **exploratory, CPU + proposer calls only. No Gemma scoring, no
GPU, nothing killed, `latex/` untouched.** Executes the battery specified in
`notes/2026-08-05__taste-decomposition-design.md` §8 against the completed Layer-3
closure pilot (`notes/2026-08-05__layer3_round4_peer_verdict.md`) and its retrospective
analyses (`notes/2026-08-06__closure-swap-and-missing-mass.md`).

Terminology, spelled out on first mention per the standing rule:
**V** = the 17 programmatic surface features; **A** = the articulated-criterion bank;
**VA_nl** = the HistGradientBoosting aggregation of the V+A score matrix, averaged over
seeds {0,1,2}; **T** = the dense readout (Llama-3.1-8B LoRA on raw text);
**Δ_beyond** = T − VA_nl, the unarticulated residual; **ε** = .005, the pilot's
per-round saturation threshold; **AUC** = area under the ROC curve;
**FIT+MINE / MONITOR** = the 80/20 title-grouped closure splits; **honest rows** = the
1,244 dense-held-out population rows where both T and VA are out-of-sample;
**Good-Turing missing mass** = estimated probability that the next independent draw
names a species not yet seen; **Chao1** = a richness estimator built from singleton
(f1) and doubleton (f2) counts; **τ** = the embedding-cosine threshold at which two
criteria are called the same species; **GEPA** = the prompt-iteration pass required
before confirmatory phrasing, deliberately not applied in the pilot or here;
**M1 / M2 / M3** = the three arms of the design-note §8 battery (sealed multi-proposer
fleet / estimator backtest / leave-out recovery); **P** = number of proposers in a
fleet round; **k** = proposals per proposer.

**Reproduction gate passed.** Every refit uses the frozen Layer-1 spec through the
pilot's own `closure_lib` / `stage4_readout.fit_block`. The full-bank round-0 and
round-4 states reproduce `round4_results.json` to machine precision (round-0 MONITOR
.6632685250121815, round-4 MONITOR .6723250724551961 — identical digits). Every
depletion and recovery number below is a difference against those refits.

Artifacts, all under `methods/taste_decomposition/closure/robust_mm/`:

| file | what |
|---|---|
| `harness.py` | M1 sealed-prompt builder + collector (seal contract in the docstring) |
| `run_glm.py`, `run_codex.py` | GLM-5.2 and gpt-5.6-luna proposer legs |
| `m3_concepts.py` → `m3_concepts.json` | 54-concept census, alone-AUCs, stratified holdout design |
| `m3_deplete.py` → `m3_depletion.json`, `slice_*.json` | depleted refits + regenerated slices |
| `m3_detect.py` → `m3_detection.json` | mechanical (τ) detection + pairwise blind adjudication set |
| `m3_recall_build.py` / `m3_recall_analyze.py` → `m3_recall.json` | **primary** rediscovery readout |
| `m3_adjudicate_analyze.py` → `m3_adjudicated.json` | secondary pairwise readout |
| `m3_recover.py` → `m3_recovery.json` | AUC recovery from rediscovered concepts |
| `m2_backtest.py` → `m2_backtest.json` | M2(a) predict-the-next-round backtest |
| `m2_fleet.py` → `m2_fleet_richness.json` | M2(b) fleet capture-recapture |
| `m1_round5_novelty.py` → `m1_novelty_round5.json` | what a sealed 5th round proposes |
| `proposals_*.json` | every proposal, with full proposer provenance |

---

## PART 1 — M1: the sealed multi-proposer fleet

### 1.1 Seal contract

Every proposer receives the same disagreement slice (60 abstracts, each with the dense
model's and the scorecard's percentile ranks) and **nothing else**: no sight of the
criterion bank, no sight of any other proposer, no label. Slice ORDERING is permuted
per proposer by a stable sha256 sort over a proposer-specific salt (never a seeded
shuffle), so two calls to the same model are genuinely independent draws. Prompts are
~85 KB and byte-identical across proposers apart from row order; the six order hashes
are distinct and recorded in each tag's `manifest.json`.

This is the one thing the pilot could not do. The pilot's rounds were sequential and
each proposer was shown the current bank and told not to duplicate it, which drove
observed recapture to ~0 and made Good-Turing return "missing mass = 1.00".

### 1.2 Fleet composition and smoke tests

| slot | family | model | k | status |
|---|---|---|---|---|
| `claude_sonnet` | claude | Claude Sonnet (sealed subagent) | 15 | 4/4 tags, 15/15 distinct each |
| `claude_opus` | claude | Claude Opus (sealed subagent) | 15 | 4/4 tags, 15/15 distinct each |
| `codex_luna_a` | openai | gpt-5.6-luna via `codex exec`, effort high, sandbox read-only | 15 | 4/4 tags, 15/15 distinct |
| `codex_luna_b` | openai | gpt-5.6-luna, independent call, different ordering | 15 | 4/4 tags, 15/15 distinct |
| `glm_a` | glm | glm-5.2 thinking, key `~/.z-ai-api-key.txt` | 15 | **round-5 tag only** |
| `glm_b` | glm | glm-5.2 thinking, key `~/.z-ai-api-key-spangher.txt` | 15 | **round-5 tag only** |

**Smoke tests, in the order they mattered.**

1. *Endpoint.* The first GLM probe went to `/api/coding/paas/v4/chat/completions` and
   returned HTTP 429 `1302` on both keys. Per `reference_glm_subscription_api` the Lite
   subscription covers **`https://api.z.ai/api/anthropic/v1/messages` only**. Re-probed
   there: key A returned `model=glm-5.2`, 401 output tokens, **1,741 characters of
   thinking trace** — thinking mode live and the budget dial working.
2. *Trace length before the think-heavy round* (GLM quota rule). The first real
   proposal call ran at `budget_tokens=6000, max_tokens=16000` and **stopped on
   `max_tokens` with 56,035 characters of trace and only 5 of 15 criteria emitted**.
   The trace ate the budget. Re-tuned to `budget_tokens=2048, max_tokens=32000` — the
   sealed prompt was left byte-identical, only sampling parameters moved — and both
   GLM slots then returned a complete 15/15. Cost per GLM call after prompt caching:
   ~17 K input (cache-read) + 6.5–8.6 K output tokens; the whole GLM leg is well under
   0.1% of the 87 M/week Lite budget.
3. *Codex.* `codex exec --model gpt-5.6-luna -c model_reasoning_effort=high -s
   read-only`, run in a scratch working directory outside the repo (so the proposer
   cannot reach the bank, the labels, or other proposers). Smoke call returned clean
   JSON in 2,502 tokens.

**Did luna suffice? Yes — no escalation to gpt-5.6-sol was needed.** All 8 luna calls
returned exactly 15 parseable, distinctly-named criteria (mean scoring-instruction
length 207–227 characters, comparable to GLM's 207–283 and Sonnet's 321; Opus writes
longer at 651). Luna's proposals are substantive and slice-grounded, e.g. "Nontrivial
component synergy", "Assumption-challenging empirical discovery", "Diagnostic benchmark
validity". The design note's escalation clause was never triggered.

**The gap, reported as instructed.** GLM is **reachable but severely request-rate
limited** on the Lite plan: both keys return `1302 rate_limit_error` within seconds of
a successful call, and a single successful proposal call took 2–4 attempts across
60–200 s of backoff. Both GLM slots landed for the round-5 tag (P = 6, three families).
For the three M3 replicates the GLM leg was still grinding through backoff when the
battery completed, so **M3's replicate fleets are P = 4 across two families** (Claude,
OpenAI) — above the stated minimum of P = 4, below the P ≥ 5 / ≥ 3-family target. The
runner (`run_glm.py`) resumes by output file and rotates keys after five failures, so
the missing calls can be back-filled later without repeating any finished work.

### 1.3 The live probe: what a sealed 5th round proposes

The pilot stopped at round 4 with two consecutive sub-ε rounds. The `round5` tag asks
the question the stopping rule structurally cannot: with the **full round-4 bank**
(152 features: 54 original concepts + 56 mined criteria + 17 V features), six sealed
proposers read the round-4 disagreement slice. 90 proposals → **49 species** at τ = .79.

| τ | proposals recapturing something already in the bank | species novel | recapture of the 56 MINED criteria | recapture of the 54 ORIGINAL bank concepts |
|---|---|---|---|---|
| .77 | 25 / 90 (28%) | 37 / 49 (76%) | 35.7% | **0.0%** |
| .79 | 13 / 90 (14%) | 42 / 49 (86%) | 19.6% | **0.0%** |
| .81 | 6 / 90 (7%) | 43 / 49 (88%) | 10.7% | **0.0%** |

The recaptures are legible and reassuring — the fleet independently re-derives the
pilot's own mined criteria: "States formal guarantee integral to the claim" ↔ "Provable
guarantee TOGETHER WITH demonstrated…" (.855); "Theory-to-experiment coupling" ↔
"Theory and experiment coupled…" (.846); "Unifies disparate tasks under one abstraction"
↔ "Unification — one mechanism handles problems previously…" (.826).

**The 0.0% column is the load-bearing measurement of this whole report**, and §3.2
explains what it does and does not mean.

---

## PART 2 — M3: leave-out recovery (the positive control)

### 2.1 Design

From the bank's **54 distinct effective concepts** (154 delivered criteria → 95 distinct
names → 79 columns surviving the frozen degeneracy screen → 54 distinct concepts among
them; §2.4 of the swap/missing-mass note), three non-overlapping holdout replicates of
K = 8 were drawn, stratified by alone-AUC.

Design decisions, all recorded in `m3_concepts.json` and reproducible:

- **Concept = distinct criterion name** among surviving columns; depletion removes the
  concept's whole column **footprint**, located by VALUE as well as by name so
  bit-identical duplicates cannot survive. (Audited: 0 concepts had a bit-identical
  twin under a *different* name, so name-matching would have sufficed — but the check
  is what licenses that statement.)
- **alone-AUC computed on FIT+MINE only.** MONITOR is never read for a design decision.
- **Strata = terciles of |alone-AUC − .5|** over the 54: high (18, .524–.607),
  mid (18, .482–.524), low (18, .498–.503).
- 3 high + 3 mid + 2 low per replicate, assigned by stable sha256 sort over a fixed
  salt. 24 distinct concepts, no concept in two replicates.

**Scale calibration that matters for reading everything below: this bank's individual
criteria are weak.** The single most informative of all 54 concepts has alone-AUC .607;
the median is ~.51. VA_nl reaches .684 by combining ~71 near-null columns, not by
containing any strong one.

| replicate | held-out concept | stratum | alone-AUC (FIT+MINE) | cols dropped |
|---|---|---|---|---|
| rep1 | Data/materials availability, rationale, and reusability | high | .545 | 1 |
| rep1 | External validity, scope, and generalizability claims | high | .526 | 1 |
| rep1 | Procedural and analytical detail sufficient for replication | high | .526 | 1 |
| rep1 | Discussion and conclusions — interpretation and implications | mid | .519 | 1 |
| rep1 | Open data, code, models, and materials provided for reproduction | mid | .512 | 4 |
| rep1 | TRIPOD adherence for prediction model studies | mid | .511 | 1 |
| rep1 | Ethical/legal compliance and human-data transparency | low | .501 | 2 |
| rep1 | Intervention and comparator description (TIDieR-complete) | low | .500 | 3 |
| rep2 | Claim–evidence alignment and causal caution | high | .559 | 1 |
| rep2 | Novelty and positioning vs prior work | high | .544 | 1 |
| rep2 | Theoretical framing, coherence, and use of theory | high | .537 | 1 |
| rep2 | Title, abstract, and keyword quality | mid | .520 | 1 |
| rep2 | Computational resources, efficiency, and convergence comparisons | mid | .514 | 1 |
| rep2 | Dataset provenance, composition, and representativeness | mid | .505 | 2 |
| rep2 | Citation practice quality, coverage, and ethics | low | .501 | 1 |
| rep2 | Review/synthesis design quality and reporting | low | .500 | 1 |
| rep3 | Abstract accuracy, completeness, and balance | high | .555 | 1 |
| rep3 | Data availability and sharing — actionable access and statements | high | .533 | 1 |
| rep3 | ML/computational experiment setup and model reporting transparency | high | .529 | 1 |
| rep3 | Outcome measures — definition, prioritization, assessment, precision | mid | .521 | 1 |
| rep3 | Study design description (groups/units/allocation/timing) | mid | .482 | 1 |
| rep3 | Research software artifact quality, usability, and impact | mid | .504 | 2 |
| rep3 | Accessibility and inclusive communication | low | .501 | 1 |
| rep3 | Dataset documentation, stewardship, and responsible use | low | .501 | 1 |

### 2.2 Step 1 — the depletion is real and is the right size

VA_nl recomputed under the frozen spec on the depleted matrix, then the disagreement
slice regenerated against the depleted stack (pilot rule: top |dense rank − VA_nl rank|
inside the mining slice, 30 per direction).

| bank state | A cols | features | MONITOR-all AUC | honest (n=1,244) AUC | honest drop vs full | 95% CI | P(drop>0) |
|---|---|---|---|---|---|---|---|
| **full round-0** | 154 | 96 | .6633 | .6844 | — | — | — |
| rep1 depleted | 140 | 82 | .6642 | .6790 | **+.0055** | [−.0026, +.0139] | .907 |
| rep2 depleted | 145 | 87 | .6641 | .6770 | **+.0075** | [−.0001, +.0152] | .974 |
| rep3 depleted | 145 | 87 | .6617 | .6782 | **+.0063** | [−.0032, +.0161] | .903 |
| *(reference)* full round-4 | +56 mined | 152 | .6723 | .6962 | *(−.0118 = the pilot's gain)* | | |

**Mean honest depletion drop = +.0064.** Removing 8 of 54 concepts costs about **54% of
everything four rounds of mining bought** (+.0118). Three independent replicates agree
within .002 and all three have P(drop > 0) ≥ .90. The perturbation is real, consistent,
and of exactly the right magnitude to be detectable by the same readout that measured
the closure curve — which is what a positive control needs.

(On MONITOR-all, n = 1,192 with a much noisier readout, the drops are −.0009/−.0009/
+.0016 — indistinguishable from zero. The honest 1,244-row population governs, per the
pilot's own amendment.)

Regenerated slices overlap the pilot's round-1 slice on 45–50 of 60 rows, so depletion
moves the mining target but does not relocate it to a different corner of the corpus.

### 2.3 Step 2 — the mechanical (τ) detector is OUT OF RANGE and must not be quoted

The planned detector was: held-out concept rediscovered if some fleet proposal sits at
bge-large cosine ≥ τ, with τ ∈ {.77, .79, .81} around the pilot's defensible band
(τ ≥ .78, floored by the planted probes at .739).

**It returns 0.00 at every τ, for held-out and retained concepts alike, in all three
replicates.** The reason is measurable, not conjectural:

- the **maximum cosine between ANY of the 54 bank concepts and ANY fleet proposal** is
  .706 / .717 / .722 across the three replicates (.752 for the round-5 fleet) — the
  entire distribution lies **below** the threshold band;
- within that compressed range the embedder barely ranks: mean cosine falls only from
  **.64 at the nearest candidate to .59 at the tenth**.

The cause is register, and it was already flagged in the pilot's census as a confound:
the 154-bank is written in **general scientific-reporting language** (CONSORT / PRISMA /
STROBE / TIDieR / CHEERS items) while the fleet, reading ML abstracts, writes in **ML
language**. τ was calibrated on *within-register* pairs (mined criterion vs mined
criterion) and does not transfer across registers. Note the internal consistency: the
same fleet recaptures the *mined* criteria at 20–36% and the *original bank* at 0% —
the detector works fine inside a register and has no dynamic range across one.

**So the mechanical rate is an artefact of instrument range, not a finding about the
fleet, and it is not quoted as a rediscovery rate anywhere in this report.**

### 2.4 Step 3 — the adjudicated readout (primary): a FULL-RECALL instrument

Because the embedder cannot shortlist, it was removed from the decision path. For each
target concept a sealed Opus judge sees **every proposal the replicate's fleet
produced** (60 candidates) and answers a recall question: does any of them name this
same concept, and which?

Blinding and controls:

- each replicate's **16 targets** = 8 HELD-OUT concepts + **8 stratum-matched RETAINED
  concepts** (still in the depleted bank), shuffled together by stable hash, unlabeled;
- candidates carry neutral ids in hash order, with no proposer attribution;
- **two independent judges per replicate**; judge agreement **1.00 / .94 / 1.00**;
- the retained concepts are the instrument's **false-positive floor**: the fleet had no
  depletion-driven reason to name them, so *sensitivity − control* is the quantity that
  says whether rediscovery is depletion-directed at all.

| readout | held-out sensitivity | retained control | lift |
|---|---|---|---|
| primary (judges agree; disagreement → strict) | **.333** [.167, .542] | .375 | **−.042** [−.292, +.208], P(>0) = .32 |
| either judge | .375 | .375 | .000 |
| both judges | .333 | .375 | −.042 |

By alone-AUC stratum (primary rule):

| stratum | n | held-out sensitivity | retained control |
|---|---|---|---|
| **high** | 9 | **.556** [.222, .889] | .778 |
| mid | 9 | .333 | .111 |
| low | 6 | **.000** | .167 |

By replicate: rep1 .375, rep2 .500, rep3 .125.

**Secondary instrument agrees.** An independent pairwise pass (top-3 nearest proposals
per concept, 148 provenance-stripped pairs, X/Y order randomised, two sealed Opus
judges, raw agreement .939 / Cohen's κ = .756) gives sensitivity **.292** and control
**.292** — lift exactly **.000**, with the same stratum gradient (high .556, mid .222,
low .000). Two instruments with different coverage, different framings and different
judge sessions land on the same number.

**Anchor battery** (blinded known-label anchors, per the standing rule). Four anchors
were embedded in the pairwise pass: two **strong-label DIFFERENT** (the pilot's
deliberately planted lexical look-alikes) and two **weak-label SAME** (the pilot's two
highest cross-round embedding recaptures, never human-verified). Both judges scored
**2/4**: both passed both planted-DIFFERENT anchors and rejected both weak-SAME
anchors. Read correctly, this says the judges are **strict**, and that the pilot's
embedding-derived "genuine recaptures" do not survive a careful same-concept test. It
makes the reported sensitivity a **lower bound**, and it independently corroborates
§2.3's warning against trusting cosine as a concept-identity test on this corpus.

### 2.5 What the sensitivity number actually means

Three things are true at once and only the combination is informative:

1. **Rediscovery is graded by value.** .556 on high-alone-AUC concepts, .333 on mid
   (.222 in the pairwise instrument), **.000 on low in both**. The fleet finds the
   concepts that carry signal and ignores the ones that do not. That is the right shape
   for a miner, and it is what makes §2.6's AUC recovery work.
2. **Rediscovery is NOT depletion-directed.** The fleet names retained concepts at
   .375 — as often as held-out ones. Lift = −.042, 95% CI [−.292, +.208]. Removing a
   concept from the bank does not measurably increase the chance the fleet names it.
   Mechanistically this is unsurprising once stated: the disagreement slice is a *weak*
   pointer. Depleting 8 near-null columns moves VA_nl by ~.006 and reshuffles only
   10–15 of 60 slice rows, so the proposer is reading almost the same abstracts and
   proposing from its own priors about paper quality, not from the hole in the bank.
3. **The floor in the interpretation contract is not met.** Design note §8 sets it at
   **≥ 70% on high-value holdouts**. Measured: **.556**, 95% CI [.222, .889], with only
   a **16%** bootstrap probability of actually being ≥ .70. And even that .556 cannot be
   attributed to depletion, because the control matches it.

### 2.6 Step 4 — AUC recovery

Re-adding **only the ORIGINAL score columns of rediscovered concepts** (zero new
judging) and refitting VA_nl under the frozen spec. Honest population, n = 1,244:

| replicate | rediscovered | full | depleted | recovered | depletion drop | recovery | **recovered fraction** | recovery 95% CI | P(>0) |
|---|---|---|---|---|---|---|---|---|---|
| rep1 | 3 / 8 | .6844 | .6790 | .6840 | +.0055 | +.0050 | **.916** | [−.0020, +.0127] | .919 |
| rep2 | 4 / 8 | .6844 | .6770 | .6833 | +.0075 | +.0063 | **.848** | [−.0006, +.0135] | .961 |
| rep3 | 1 / 8 | .6844 | .6782 | .6797 | +.0063 | +.0015 | **.245** | [−.0019, +.0051] | .808 |
| **pooled** | **8 / 24** | | | | **+.0192** | **+.0129** | **.671** | | |

**The value-weighted rediscovery rate is double the count-weighted one: 33% of concepts,
67% of the AUC.** In rep1 and rep2 three or four rediscovered concepts return 85–92% of
what removing eight of them cost. This is the one arm of M3 that comes out positive, and
the reason is §2.5's first point — rediscovery is graded by value, so the concepts the
fleet names are disproportionately the ones carrying the depletion cost. rep3, where the
fleet found only one concept, recovers only a quarter and shows what the downside looks
like.

Two limits on how far this can be pushed:

- it is **conditional on the concept already having a scored column**. M3 measures
  "would the fleet have named it", not "would a fresh Gemma scoring of the fleet's
  phrasing have reproduced the column's signal". The latter needs judging and was out
  of scope by design.
- the **lift caveat from §2.5 still applies**. The fleet names high-value general
  concepts whether or not they were removed, so this says the fleet's *output* contains
  enough to restore the AUC — not that depletion is what caused it to be there.

---

## PART 3 — M2: does the estimator earn quotation rights?

### 3.1 (a) Pilot backtest — predict round r+1 from rounds 0..r

Honest-population levels .6844 → .6891 → .6929 → .6958 → .6962; marginal gains
+.00465, +.00380, +.00289, +.00046. Readout-noise band = group-level (`ntitle`) paired
bootstrap of each step's ΔAUC, 2,000 draws, prediction vectors held fixed.

| step | actual | noise band (±1.96 sd) | geometric | error | saturating | persistence baseline | zero baseline |
|---|---|---|---|---|---|---|---|
| g1 → **g2** | +.0038 | ±.0074 | *not identified* (1 point; λ-bracket [.0019, .0042]) | — | *not identified* | +.0047 | .0000 |
| g1,g2 → **g3** | +.0029 | ±.0087 | **+.0031** (λ̂ = .816) | **+.0002** | +.0031 | +.0038 | .0000 |
| g1..g3 → **g4** | +.0005 | ±.0071 | **+.0023** (λ̂ = .793) | **+.0019** | +.0023 | +.0029 | .0000 |

**Verdict: the estimator does NOT earn quotation rights as a point predictor, and DOES
earn them as a conservative upper bound.**

- *Why not a point predictor.* Every predictor "lands" inside the readout-noise band —
  including the zero baseline and the persistence baseline. The band (±.007) is wider
  than every gain in the series (max +.0047), so **at the pilot's precision the
  backtest cannot discriminate any predictor from any other.** Claiming the estimator
  "predicts the next round" on this evidence would be reading noise.
- *Why an upper bound.* In both identified steps the error is **non-negative**
  (+.0002, +.0019); at r = 1 the λ-bracket's upper end (+.0042) also exceeds the actual
  (+.0038). The estimator never under-predicted. It over-predicts the *final* round by
  5× (+.0023 predicted vs +.0005 actual) — exactly the failure mode you want in a
  saturation diagnostic, since it errs toward "keep mining", never toward a premature
  stop.
- *Remaining-mass backtest.* Fitting on g1,g2 predicted +.0056 for rounds 3–4 truncated
  (realised +.0034); fitting on g1..g3 predicted +.0023 for round 4 (realised +.0005).
  Same signature: consistently high, never low.

**New-species backtest: DEGENERATE BY DESIGN, reported to show why M1 exists.** With
sequential, anti-duplication-instructed rounds the observed missing mass pins at
1.00 / 1.00 / .91, so the predictor degenerates to "the next batch is all new" and is
trivially accurate (errors 0.0, +4.0, +0.8 species). It carries no information. The
per-round series confirms the design defect directly: 14 / 14 / 11 / 11 new species
from 14 / 14 / 15 / 13 proposals, with only 0 / 0 / 3 / 2 recaptured from prior rounds.

### 3.2 (b) Fleet-based richness — the quantities the pilot could not compute

Species = single-linkage clusters of the P × k proposals at τ = .79.

| tag | P | families | N | S_obs | f1 | f2 | Good-Turing mass M̂ | Chao1 (bias-corr.) | cross-proposer recapture | species named by ≥ 2 families |
|---|---|---|---|---|---|---|---|---|---|---|
| round5 | 6 | 3 | 90 | 49 | 39 | 7 | **.433** | 142 | **.20** | 6 |
| rep1 | 4 | 2 | 60 | 38 | 28 | 6 | **.467** | 92 | **.26** | 6 |
| rep2 | 4 | 2 | 60 | 37 | 25 | 9 | **.417** | 67 | **.32** | 3 |
| rep3 | 4 | 2 | 60 | 43 | 33 | 7 | **.550** | 109 | **.23** | 3 |

**The design fix works.** Cross-proposer recapture rises from ~0 (pilot) to **20–32%**,
and Good-Turing missing mass falls from the pilot's degenerate **.82–1.00** to a real
**.42–.55**. f2 rises from {0, 2, 3} to {6, 7, 9}. For the first time the estimator has
something to work with.

**Proposer-level uncertainty.** Leave-one-proposer-out jackknife (the primary statement;
resampling proposers *with replacement* duplicates whole proposers and manufactures
doubletons, biasing M̂ down, so the bootstrap is reported only with that caveat):
M̂ = .464 [.347, .520] (round5), .533 [.356, .622] (rep1), .517 [.378, .644] (rep2),
.628 [.556, .711] (rep3). This is the width the pilot structurally could not produce.

**Species accumulation is nearly linear — the concept space is not exhausted.**
Marginal new species per added proposer (mean over random proposer orderings), round5:

| proposer # | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| new species added | 14.0 | 10.2 | 9.1 | 8.5 | 8.3 | **6.5** |
| cumulative | 14.0 | 24.2 | 33.3 | 41.7 | 50.0 | **49** |

The sixth independent proposer still contributes **6.5 new species from 15 proposals —
a 43% novelty rate**. One sealed 6-proposer fleet produced 49 species in a single round,
essentially matching the 50 species the pilot's four sequential rounds produced in
total.

**Fleet vs sequential: do they agree?**

| estimate | value |
|---|---|
| sequential-pilot extrapolation (geometric decay on the gain series) | **+.0030** |
| fleet, odds form R̂ = [M̂/(1−M̂)] × Δ̄ × λ̂ | **+.0014 to +.0024** |
| fleet, species form R̂ = (Chao1_bc − S_obs) × value-per-species × λ̂ | +.0047 to +.0146 |

**The odds form agrees with the sequential extrapolation** — same order of magnitude,
fleet slightly *lower*, and both far below the +.081 residual. The **species form does
not, and should not be quoted**: it inherits Chao1's instability at f2 = 6–9 (estimates
67 to 142 from 37–49 observed species), which is the same non-identification the pilot
already flagged, merely less extreme. The defensible statement is the odds form.

**The mechanism this exposes is the important part.** Put §3.2 next to §1.3: the fleet's
species pool is *rich* (missing mass ~.45, accumulation still linear at P = 6, 86% of
its species novel against the round-4 bank) while the recoverable AUC is *tiny*
(+.0014–.0024). This is a direct, prospective confirmation of the pilot's retrospective
finding that its saturation was **redundancy saturation, not value exhaustion** — the
proposers have not run out of nameable criteria, they have run out of criteria the bank
does not already span. Richness and value have come apart, and only the value side is
saturated.

---

## PART 4 — FREEZE-READY RECOMMENDATION

### 4.1 The sensitivity floor achieved

| quantity | measured |
|---|---|
| rediscovery sensitivity, all 24 held-out concepts (COUNT-weighted) | **.333** [.167, .542] |
| rediscovery sensitivity, high-alone-AUC holdouts (n = 9) | **.556** [.222, .889] |
| P(high-stratum sensitivity ≥ .70) | **.16** |
| retained-concept control (false-positive floor) | .375 |
| depletion lift (sensitivity − control) | **−.042** [−.292, +.208], P(>0) = .32 |
| AUC recovery from rediscovered concepts (VALUE-weighted) | **.671** pooled (.92 / .85 / .25) |
| depletion magnitude the fleet had to detect | +.0064 honest AUC (54% of the pilot's 4-round gain) |
| fleet size in M3 | P = 4, 2 families (GLM rate-limited out) |

### 4.2 Is "no more articulable signal" quotable?

**No — not at this sensitivity, and the correct wording is already specified in the
design note.** The interpretation contract in §8 makes the strong claim conditional on
≥ 70% rediscovery of high-value holdouts. Measured: 55.6% with a CI that spans .22–.89,
a 16% chance of clearing the floor, **and no measurable lift over concepts that were
never removed**. The second point is the harder one: even if sensitivity had cleared
70%, a lift of zero would mean the instrument is measuring what the fleet names anyway,
not what depletion made it name.

So the peer-verdict plateau should be quoted as:

> **"Δ_plateau = +.081 is not discoverable by this miner"** — a bound on what a sealed
> multi-model criterion-proposal fleet reading a dense-disagreement slice recovers, at a
> measured rediscovery sensitivity of 33% overall / 56% on high-value concepts and a
> depletion lift indistinguishable from zero.

Never as "no further nameable criteria exist", and never as "the residual is tacit".

The AUC-recovery arm (§2.6) is the one arm that comes out positive and it slightly
softens — but does not overturn — this verdict. **A fleet that names only a third of the
held-out concepts restores two thirds of the AUC they carried**, because the ones it
names are the ones that matter. So the honest two-line summary is: *this miner would
have recovered most of the recoverable signal, and we cannot show that removing a
concept is what makes it name that concept.* The first clause is the positive control
passing on value; the second is why the strong closure claim stays unquotable.

**What is quotable, and is now better supported than before:**

- The remaining-AUC bound. Two independent routes — the sequential decay extrapolation
  (+.0030) and the fleet's Good-Turing odds form (+.0014 to +.0024) — agree that
  continued proposal buys **≈ +.002 to +.003 AUC**, ~2–4% of the +.081 residual. The
  fleet route is the stronger of the two because it rests on a *measured* missing mass
  (.42–.55) rather than a 4-point decay fit, and it carries proposer-level width for
  the first time.
- The mechanism: **redundancy saturation, not value exhaustion**, now demonstrated
  prospectively (rich species pool, near-linear accumulation, negligible recoverable
  AUC) rather than inferred from alone-AUC series.
- The estimator's role: a **conservative upper bound** on the next round's gain, which
  never under-predicted in 3/3 backtestable steps.

### 4.3 Changes to carry into the confirmatory freeze

1. **Run M3 as a gate, not as a report.** Every confirmatory cell must publish its
   rediscovery sensitivity *and its retained-concept control* before its plateau is
   quoted. Sensitivity without the control is uninterpretable — this battery would have
   read "56% rediscovery on high-value concepts, respectable" without it.
2. **Never use an embedding threshold to decide concept identity across registers.**
   Calibrate τ on pairs drawn from *both* corpora being compared, and refuse to report a
   τ-based rate whenever the maximum cross-corpus cosine sits below the band (as it did
   here, .72 < .78). The full-recall adjudication instrument in `m3_recall_build.py` is
   the replacement and costs ~2 judge calls per replicate.
3. **Deduplicate and REGISTER-MATCH the incoming bank at round 0.** The pilot's existing
   dedup recommendation stands (154 delivered → 54 effective concepts). Add: report the
   register mismatch between bank and corpus, because it silently governs every
   similarity-based measurement downstream.
4. **P ≥ 5 with ≥ 3 families is affordable but not on GLM Lite alone.** Claude subagents
   and `codex exec` each delivered 8/8 complete responses with no retries; GLM delivered
   2/8 under a hard request-rate limit. Either budget ~10 minutes of wall-clock backoff
   per GLM call, or treat GLM as a bonus family and build the guaranteed P = 4 from
   Claude + Codex.
5. **Keep the anchor battery, and fix the anchors.** The weak-label SAME anchors were
   derived from embedding cosine and both judges rejected them. Confirmatory rounds need
   SAME anchors constructed the way the DIFFERENT ones were: deliberately authored
   paraphrase pairs, not high-cosine pairs.
6. **STOP-M stays a second gate, with the fleet's M̂ in it.** M̂ = .42–.55 is far above
   the suggested M_crit = .25, so on the pilot's own data STOP-M would **not** have
   fired even though the ΔAUC rule did. That is the intended behaviour: the run still
   stops, and the report must carry "saturation declared with estimated remaining mass
   R̂ ≈ +.002 [fleet odds form], missing mass M̂ ≈ .45".

---

## Caveats that travel with every number here

1. **Exploratory.** No prereg covers this battery; it is a validation pass on a
   completed exploratory pilot. Nothing here changes Δ_plateau = +.081.
2. **Pre-GEPA**, like the pilot. A GEPA-iterated proposer could shift both the
   rediscovery rate and the species pool.
3. **M3's replicate fleets are P = 4 across 2 families.** GLM's 1302 rate limit kept the
   third family out of the replicates (it is present in the round-5 tag). A third family
   could only raise sensitivity, so the reported figure is a lower bound on a 3-family
   fleet — but the *lift* (which is what the gate turns on) has no reason to move,
   since a third family adds candidates for held-out and retained targets alike.
   One GLM slot did land for rep1 after its P = 4 pool had been frozen and judged; it is
   banked at `proposals_rep1_glm_LATE_unused.json` and deliberately **excluded** so all
   three replicates stay matched. Resume the rest with
   `python3 run_glm.py --tags rep1,rep2,rep3` (resumes by output file, repeats nothing).
4. **n = 24 held-out concepts.** All rediscovery CIs are wide; the stratum cells hold
   9 / 9 / 6.
5. **Judges are strict** (2/4 anchors, both failures on the weak-label SAME side), so
   sensitivity is a lower bound and the lift is the more robust statistic.
6. **The depletion drop is measured on the honest 1,244 rows.** On MONITOR-all it is
   indistinguishable from zero; the honest population governs per the pilot's amendment.
7. **Chao1 remains non-identified** at f2 = 6–9. The species-form remaining-AUC bound is
   reported to exercise the machinery and must not be quoted.
8. **No criterion proposed in this battery was scored.** New species were counted and
   banked (`proposals_*.json`), never judged, per the design note's cost rule.
