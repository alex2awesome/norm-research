# BOUNDED audit · curve explanation · label-type taxonomy — 2026-07-08

Doc-of-record for the three analysis legs the user prioritized ("1 is extremely important...
especially interested in 2... [3] press forward with speed"). Data: v2 multi-task panel
(1,347 bank metrics + 5 planted/task, 8 tasks, 9–12 executors; `outputs/osl_multi/`).
Scripts: `bounded_audit.py`, `explain_curves.py`, `panel_analysis.py` (same dir).

## 1. BOUNDED audit — which plateaus survive scrutiny?

Every BOUNDED verdict is a tacitness claim with two impostors: **family dialects** (pooled bend
= artifact of which families sit where on z) and **floor contact** (plateau = criterion
underdetermination, not executor limitation). Audit per metric: (a) within-family shape from
Llama-only and Qwen-only rungs; (b) per-metric frontier floor = mean cross-family agreement of
top-z executors; (c) classification.

| task | BOUNDED | → TACIT-CANDIDATE | → DIALECT-SUSPECT | → AT-FLOOR / other |
|---|---|---|---|---|
| humor (full panel) | 36 | **16** | 10 | 10 ceiling-adjacent |
| creative-writing | 5 | 1 (epistolary form authenticity) | 3 | 1 unaudit. |
| math | 2 | 0 | 2 | — |
| news-homepages | 3 | 0 | 2 | 1 at-floor |
| peer-review | 3 | 1 (audience-tailored communication) | 1 | 1 unaudit. |
| notice-and-comment | 1 | 1 (concision) | — | — |

REACHES converse check: ~90% dialect-clean (e.g. humor 95/105 REACHES-OK).

**The 16 humor TACIT-CANDIDATES are qualitatively coherent** — essentially all are
voice/persona/embodiment/cultural-convention criteria: *distinctive personal comic voice,
character & impersonation craft, Australian humor conventions, balancing humor with pathos,
roast tone (affection vs contempt), one-liner timing/rhythm, compressed quotable phrasing,
originality vs derivativeness, brand identity coherence, host presence, satire target
sharpness, reference accessibility calibration*. This is the taste/embodied cluster surfacing
again from an independent instrument — the content that craft lore says can't be transmitted
by description. `outputs/osl_multi/bounded_audit.json`. Pending upgrade: 405B point (5-rung
within-Llama ladder) re-audits every one of these.

## 2. What explains curve shape? (behavioral features beat the a-priori-null precedent)

Precedent honored: text-only concept tags FAILED LODO on name-deficit (2026-07-05, Simpson
reversal). Here features are *behavioral* and all correlations are WITHIN-task partials
(| desc-length), Stouffer-combined over 8 tasks (n=1,137 classified metrics):

| feature | slope | frontier level (top_y) | asymptote L |
|---|---|---|---|
| **frontier floor** (underdetermination) | +5.2z** | **+11.7z**** | **+10.9z**** |
| **H_M** (behavioral entropy) | +5.9z** | +10.2z** | +5.5z** |
| **OPT_Ω bits** | +5.6z** | +9.5z** | +4.3z** |
| mean/max **δ^M** (unit semantic effect) | ~0 | **+6.1z**** | ns |
| frac detect_M | ns | +4.0z** | ~ |
| **mean ε_ctx** (unit context-fragility) | −2.4z* | **−5.1z**** | −2.1z* |
| n_units / atom_frac / dead_frac | ns | ns | ns |

Reading: (i) **underdetermination is the dominant constraint on asymptotes** — where frontier
models disagree about meaning, no curve can climb (content → floor → L mediation candidate);
(ii) **unit QUALITY, not quantity**: metrics whose units carry strong, context-robust semantic
effects (high δ^M, low ε_ctx) reach higher frontiers; unit COUNT is null (consistent with the
CUF corr(n_units, OPT_Ω)≈0 deconfound); (iii) δ^M predicts *where you end up*, not *how fast
you climb* — level and slope have different physics.

Classifier (LOO rank-ridge, honest AUC): BOUNDED-vs-REACHES **0.69 pooled / 0.72 humor**;
**audited labels (TACIT-CANDIDATE vs REACHES-OK): 0.78** — the audit-cleaned labels are MORE
predictable, itself evidence the audit removed label noise. `outputs/osl_multi/explain_curves.json`.

## 3. Label-type taxonomy — why external validity varies by task

The MI-arm sign pattern is systematic once tasks are grouped by what their silver label
*measures* (a-priori classification):

| label type | tasks | MI arm (frontier, length-controlled) |
|---|---|---|
| **direct crowd quality judgment** | humor (funny votes) | **+0.53** (p<1e-4) |
| **uptake/popularity** (exposure-confounded) | CW (upvotes), press-releases (pickup), news-homepages (placement) | +0.03 … +0.10 (ns) |
| **gatekeeping/acceptance outcome** | math (accepted answer), peer-review (accept), patents (grant) | −0.18* (math) … ~0 |
| degenerate-text | notice-and-comment (17-word comments) | +0.35 raw → +0.15 after length control (was length) |

Reading: reconstruction-based metrics track silver *where silver measures judged quality of
the artifact itself*; the correlation vanishes for popularity signals and **inverts for
acceptance outcomes** (math — consistent with the known math verdict-inversion in the
cross-task census). This is a claim about what community signals measure, not a validity
failure: the instrument's external validity should only appear where the external signal is
itself a quality judgment. (Sibling doctrine: all-three-label-types rule; length controls now
standard in `panel_analysis.py` after n&c's artifact.)

## 405B-scale API probe (Hermes-3-Llama-3.1-405B via OpenRouter, 2026-07-08)

`meta-llama/llama-3.1-405b-instruct` is no longer served on OpenRouter; nearest = Hermes-3-405B
(same base, Nous instruct-tune) → labeled family `hermes`, its own dialect point (like GLM), NOT
the 5th local-Llama rung. Minimal probe: 21 freeze-285 metrics (5 tacit + 7 dialect-suspect +
4 REACHES anchors + 5 planted), battery + 200 pairs × both orders = 9,470 calls, **$1.40**.
Scripts: `outputs/osl/or405_probe.py`, `hermes_adjudicate.py` → `hermes_adjudication.json`.

- battery bal_acc .863 (z 1.84 hard-scale); **planted pairwise truth-acc .894** — instrument
  executes mechanics near-frontier (llama70b .927, glm-52 .924). Positive control passes.
- consensus-agreement class means at hermes405b: PLANTED .598 ≈ DIALECT .593 > ANCHOR .557 >
  **TACIT .501 (lowest)**. The same within-executor ordering (TACIT lowest) holds at llama70b,
  qwen25-72b, and glm-52. **Three independent frontier dialects + two local frontiers all rank
  the tacit-candidate class worst while executing planted mechanics at ~.9** — the tacit residual
  is not a single-model artifact. Dialect-suspects score near planted (consistent with the audit:
  their pooled bends were artifacts, not tacitness).
- Caveats: 5/16 tacit candidates probed (rest are supplement-bank, outside freeze-285 rubrics);
  hard readout; absolute levels carry a dialect discount (compare orderings, not levels). The
  local FP8 405B full-width P(YES) run (queued) remains the clean adjudicator.

## Frontier divergence — the within-GLM ladder result (2026-07-08 eve)

Expanded adjudication (`hermes_adjudication.json`) over 7 frontier executors × 76 metrics
(16 tacit + 10 dialect + 10 ceiling-adjacent + 14 anchors + 21 rising + 5 planted), all
hard-binarized. Two findings:

**(1) TACIT ranks below REACHES-anchors in ALL SEVEN frontier executors** (2 local + 5 API
dialects: glm-4.5/4.7/5.2, hermes-3/4-405B). The tacit residual is now multiply replicated.

**(2) Within-GLM frontier divergence (dialect-free, same readout/serving):** as GLM scales
4.5 → 4.7 → 5.2, planted truth-accuracy RISES (.896 → .921 → .924) while agreement with the
mid-scale consensus FALLS in every class — and the fall is ORDERED BY UNDERDETERMINATION:

| class | glm-4.5 | glm-4.7 | glm-5.2 | Δ(4.5→5.2) |
|---|---|---|---|---|
| PLANTED (consensus-agr) | .627 | .621 | .576 | −.051 |
| RISING | .535 | .501 | .464 | −.071 |
| REACHES-ANCHOR | .628 | .579 | .498 | −.130 |
| CEIL-ADJ | .569 | .559 | .430 | −.139 |
| **TACIT** | .579 | .469 | .382 | **−.197** |

Capability up + crowd-agreement down = **frontier models diverge from the mid-scale consensus
as they scale, and they diverge fastest precisely on the tacit/underdetermined criteria.**
This resolves the user's "GLM-5.2 looks low" question: 5.2 is not weaker (truth-acc 2nd of all
executors) — it is *elsewhere*, and increasingly so with scale, exactly where language
underdetermines the criterion. Pending: glm-4.5-air + glm-4.6 rungs complete the 5-point
trajectory; monotonicity across 5 rungs would firm this from trend to law.

## The inverted-U — second family resolves the divergence (2026-07-09)

Qwen3 API ladder (30B-A3B-2507 → 235B-A22B-2507, OpenRouter, hard A/B readout, ~$0.51; scripts
`outputs/osl/qwen3_ladder.sh`, `qwen3_adjudicate.py` → `qwen3_adjudication.json`) on the same
36-metric adjudication core. Result: **within-Qwen3 everything RISES** (TACIT .399→.579, planted
.491→.597, truth-acc .744→.830) — the *opposite* sign of within-GLM. The local Qwen2.5 ladder
(3B→72B) rises identically (TACIT .226→.582, truth .772→.885).

Put all 9 trusted executors on one axis (agreement vs planted truth-accuracy) and the two signs
are one shape — an **inverted-U**:

| executor | truth-acc | TACIT agr | limb |
|---|---|---|---|
| qwen3-30a3 | .744 | .399 | rising |
| qwen25-3b → 72b | .772→.885 | .226→.582 | rising |
| qwen3-235b | .830 | .579 | rising |
| **glm-4.5** | **.896** | **.616** | **peak** |
| glm-4.7 | .921 | .489 | falling |
| glm-5.2 | .924 | .402 | falling |

Reading: the mid-scale crowd is a **capability-anchored reference, not a truth**. Executors
below its level (~.89 truth-acc) gain agreement as they improve; past it, further capability
buys *divergence* — class-ordered (PLANTED −.05, REACHES −.13, TACIT −.20 over the falling
limb). Frontier models grow past the crowd first where language underdetermines the criterion.
This unifies "GLM-5.2 looks low" and "Qwen ladders look normal" without any dialect story:
both Qwen ladders sit entirely on the rising limb; GLM spans the peak. Notebook §7b figure.
qwen35-122b excluded (planted agr .203, truth .684 — degenerate pair panel, standing
not-for-eval rule); consensus is hard-binarized LOFO family-balanced throughout.

### Falling limb CLOSED — four families + the frozen-crowd correction (2026-07-09 late)

Closure points (kimi-k2.5, deepseek-v3-0324, qwen3-max via OpenRouter; two serving traps cost
three aborted runs, see `reference_openrouter_hybrid_thinking_trap.md`: hybrid thinkers return
EMPTY content at max_tokens=8 → `reasoning:{enabled:false}` env-gated into backends.py;
qwen3-max Alibaba-only limit_rpm → crawl mode. NaN-guard caught every one — no garbage saved).

**Methodological catch first:** with the consensus pool drifting (API panels joining as they
land), kimi-k2.5 initially looked like the MOST crowd-agreeing frontier model (TACIT .665) —
because kimi/hermes/dsv3 had entered each other's reference crowds. **Fix: consensus frozen to
the 11 local mid-scale executors** (family-exclusion hygiene kept). All trajectories recomputed
against the frozen crowd. Same-instrument doctrine, new corollary: *freeze the crowd, not just
the battery.*

**Result under the frozen crowd — inverted-U CONFIRMED, falling limb now 4 families:**

| executor | truth | TACIT agr | TACIT−PLANTED deficit |
|---|---|---|---|
| qwen25-3b → 72b | .772→.885 | .202→**.539 (peak)** | −.22 → −.05 |
| qwen3-235b | .830 | .468 | **−.03** |
| glm-4.5 | .896 | .501 | **−.04** |
| hermes-3/4-405B | .886/.894 | .430/.423 | −.07/−.10 |
| glm-4.7 | .921 | .373 | −.18 |
| **kimi-k2.5** | **.924** | **.468** | **−.13** |
| glm-5.2 | .924 | .317 | −.18 |

Every executor above the crowd's own level (~.885) sits below the peak, across GLM, Hermes,
Kimi, DeepSeek. The **slope of the fall is a family trait** (GLM steepest −.22, Kimi mildest
−.07 from peak) but the **sign is universal**. Dialect-robust core: within-executor TACIT
deficit is −.03..−.05 in the peak zone and −.13..−.18 for every frontier point above it —
survives both dialect offsets and crowd choice. (Weak models also show large deficits (−.22):
they can't execute tacit criteria at all — the deficit itself is U-shaped; the frontier deficit
is divergence, the weak-model deficit is incapacity, separated by planted truth-acc.) Scripts:
`qwen3_adjudicate.py` (LOCAL_MID frozen crowd), `frontier_close.sh`.

### The 4-rung within-GLM ladder: divergence is a THRESHOLD, not a slide (2026-07-09)

glm-4.6 completed the interior rung (183-metric go-BIG panel). Full within-GLM ladder, frozen
crowd:

| rung | truth-acc | TACIT agr | deficit |
|---|---|---|---|
| glm-4.5 | .896 | .501 | −.040 |
| glm-4.6 | .894 | .505 | −.041 |
| glm-4.7 | .921 | .373 | −.178 |
| glm-5.2 | .924 | .317 | −.180 |

TACIT deltas rung-to-rung: **[+.005, −.133, −.056]** — flat, cliff, flat. glm-4.5 and glm-4.6
are **capability-twins** (truth .896/.894, agreement identical) sitting AT the inverted-U peak;
glm-4.7 and glm-5.2 both jump past ~.90 truth-acc and both collapse to −.18 deficit. So the
divergence is NOT a gradual monotone slide with scale — it is a **STEP at the crowd-capability
threshold (~.90 truth-acc)**: below it, frontier models track the crowd on tacit criteria; above
it, they systematically pull away, and the transition is sharp (one rung, 4.6→4.7). This
REFINES "trend → law": the law is not "agreement falls monotonically with capability" but
"agreement is bistable — high below the crowd's own competence level, low above it, with a
threshold transition." The inverted-U peak IS the threshold. (Caveat: the cliff sits between two
adjacent released rungs; a denser ladder near truth-acc .90-.92 would resolve whether the
transition is a true discontinuity or just steep. qwen3-max resume in flight = 5th family point
near the threshold.)

## In flight (step 4 + fleet)

405B (weights ready, dynamic 4-GPU waiter), qwen25-32B + gemma2-9B + mistral-24B
(battery + humor285 + 9-task panels chained on GPU2), remaining llama70b/qwen72b panels,
code-review curves-only panels (133 metrics; retry chain armed). Every landing re-runs
through `panel_analysis.py` → `bounded_audit.py` → `explain_curves.py` unchanged.
