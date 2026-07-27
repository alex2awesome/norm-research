# z×a joint scaling: the articulation–capability exchange rate — SPEC (2026-07-08)

**User objective (declared 2026-07-08, "next main objective for this thread"):**
- Metric A: if "one phrase" works on a large model and "one complex set of phrases and
  definitions" works on a small model, that measures the tacit-knowledge gain necessary and
  how it scales with model size.
- Metric B: if "one phrase" DOESN'T work even on the larger model but the complex set DOES,
  the previously fit scaling laws should say how much MORE capability this class needs before
  the knowledge is internalized.
- Plus (same-day directive): substantially better controls that "bigger/more semantically
  complex prompts are truly reflective, for smaller models, of tacit knowledge and processes"
  — i.e. kill the length/format confound and the reading-vs-executing confound.

## What already exists (the two marginals of the surface)

- **y(z) at fixed full articulation** — the v2 multi-task OSL fleet: 1,347 bank metrics × 300
  probes × 12 executors, frozen battery x-axis (sha ac4c4ed2), curve classes
  REACHES/RISING/BOUNDED/NOISY, bounded-audit → 16 humor TACIT-CANDIDATES (+3 in CW/peer/n&c),
  asymptote fits L per metric. The panel's rubric = "Name: description" ≈ the *definition* rung.
- **∂y/∂a at small z** — name-sufficiency ladders (Llama 1/3/8B, gemma-2 2/9B, Qwen 3/7B):
  name-vs-definition deficit d(z) closes 1B→3B for TASTE, flat for CRAFT, never-deficient for
  MECH. This is the finite-difference of the surface along a, measured only at small scale and
  only for 2 rungs — the metric-A phenomenon, already replicated, never joined to the big ladder.
- **y(a) at fixed small reader** — decompression grids (5 rungs × 9 domains): definition-peaked;
  dossier ≤ definition for small readers (dossier_v2 verdict: rich articulation *genuinely*
  underperforms crisp definition at 8B tier). ⇒ articulation value is NOT monotone in length —
  exactly why the placebo arms below are load-bearing.

Missing cell: **name rung at big z** (metric-B cells; nobody has run "one phrase" on 70B/72B/405B
for these metrics), and any single instrument holding metrics, probes, y-readout, and x-axis
fixed across BOTH axes. That is this experiment.

## Formal object

For metric j, executor i with battery capability z_i, articulation arm a:

    y_j(z, a) = L_j(a) · σ( k_j · (z − z0_j + β_j(a)) ),   β_j(name) ≡ 0

- **β_j(a)** = horizontal shift = capability-equivalent value of articulation a, in battery-z
  units → translated to ladder rungs ("a dossier is worth the 8B→70B jump"). THE EXCHANGE RATE.
- **Metric A** measures β directly: criterion c reached at (z_big, name) and (z_small, dossier)
  ⇒ β̂(dossier) ≈ z_big − z_small.
- **Metric B** forecast: name fails at all observed z; predict z*_name = z_crit(dossier) +
  β̂(dossier) = the capability at which the phrase alone would suffice (knowledge internalized).
  Report ordinal (in rung units) with bootstrap CI, family-conditional — Koyejo guardrail: no
  raw-MI extrapolation, the fitted family is declared, forecasts are conditional on it.
- **Decomposition — the two physics:** (i) pure horizontal shift with shared L(a)=L ⇒
  articulation is a capability substitute; the deficit is *transmissible* explicit knowledge the
  small model lacks. (ii) ΔL — articulation raises/limits the asymptote itself ⇒
  transmission-resistant residual: no tested articulation closes it at any tested scale =
  the true tacit candidate. Model comparison (shift-only vs L-free, LOEO by executor) decides
  per metric class. The 16 tacit candidates are candidates precisely because pooled L sits below
  ceiling at full articulation — this experiment asks whether MORE articulation moves L or only β.
- a*(z) = minimal rung reaching criterion at capability z = "articulation budget"; its decay
  with z is the internalization (enculturation) rate — the user's "tacit knowledge gain
  necessary and how it scales."

## Design

**Arms (6 per metric; arm text = freeze entry's `rubric`, so `osl_sweep --mbar-only --n-forms 1`
runs it unchanged — no runner code changes):**

| arm | rubric text | role |
|---|---|---|
| name | bare name phrase | "one phrase" |
| definition | bank rubric verbatim ("Name: desc") | continuity with v2 panels |
| explanation | name + authored ~150w (recognition cues, failure modes) | mid rung |
| dossier | name + authored ~400w (DEFINITION / WHAT COUNTS / CONTRAST EXEMPLARS (synthetic) / BOUNDARY CASES) | "complex set of phrases and definitions" |
| dossier_mismatched | target metric's NAME + a different same-task metric's dossier body (derangement) | specificity control: matched length/register/domain, wrong content |
| definition_padded | definition + inert generic-evaluator filler to dossier length | pure length/format control |

**Controls & gates (the "truly reflective of tacit knowledge" battery):**
1. β(dossier) must exceed BOTH β(dossier_mismatched) and β(definition_padded); otherwise the
   articulation gain is length/format artifact. β_content ≡ β(dossier) − max(placebo βs).
2. Planted metrics under dossier_mismatched: executor truth-AUC must COLLAPSE toward .5
   (positive control that executors read the body, not just the name header). Planted under
   real arms: truth-AUC stays high (authoring preserved the rule; validates the authored layer).
3. Rungs are TYPES not lengths (standing doctrine) — but placebo arms are length-matched to
   the dossier, so type and length are separable in analysis.
4. Comprehension side-arm (cheap, later this week): small executors paraphrase each dossier;
   Sonnet grades adequacy with blinded anchors. Dissociates can't-read from can't-execute:
   read-OK + execute-fail = process-tacit, the strong claim.
5. n_forms=1 everywhere (orbit reformulation would rewrite the controlled articulation).
6. Threshold-free y (below); same-family fits only; cross-family = replication of orderings.

**Slate v1 (72 metrics, `outputs/osl_multi_local/zxa_slate_v1.json`):** humor 16 TACIT-CANDIDATE
+ 10 DIALECT-SUSPECT + 10 REACHES-anchor + 5 planted; CW 1 TC + 2 DS + 3 RA + 5 planted; peer 1 TC
+ 1 DS + 3 RA + 5 planted; math 2 DS + 3 RA + 5 planted. (n&c excluded: 17-word texts = NOISY.)

**y-readout:** P(YES) m̄ per probe (primary, per standing user decision) → (a) agreement with
frontier leave-one-out consensus computed at the SAME arm (consensus is arm-specific — frontier
disagreement under the name arm is itself the underdetermination floor of the phrase), and
(b) truth-AUC on planted (absolute validity scale). Probes = the exact freeze-v2 manifests
texts[60:360] per task, so v2 panels join as the definition-arm cross-check.

**Executors:** Llama 1/3/8/70B; Qwen2.5 3/7/14/32/72B; gemma-2 9/27B (fits within family;
405B local = adjudicator when the 4-GPU waiter fires; hermes-405B/hermes-4/GLM via API = frontier
dialect checks on orderings). Small ladder runs first (GPU7 chain), 70B/72B arms chain after the
current v2 panels finish on GPUs 1/4.

**Freeze files:** per task, `freeze_zxa_<task>_v1.json`, metrics list = slate × 6 arms, entry
name = `<metric>||<arm>`, kind = `<class>|<arm>`. Assembly script validates: word-count windows
(explanation 130-180w, dossier 360-450w), planted-rule-verbatim in explanation+dossier, no
cross-metric name leakage in dossiers, derangement has no fixed points. sha256 of each freeze
logged here before launch.

**Sizing:** (41×6 + 11×6 + 10×6 + 10×6) = 432 entries × 300 probes × 1 form ≈ 130k calls per
executor ≈ one v2 task-panel equivalent; small ladder ≈ hours on one GPU; 70B/72B add ~2×3h.

## Collision guard

The 70B name-sufficiency PREREG (frozen sha 62e4b3f0, eval due post-Jul-10) is a DIFFERENT
instrument (hierarchy metrics, verdict-recovery AUC y-axis). This sweep uses bank metrics and
the consensus/planted y. Do not report joint conclusions across the two until the prereg eval
has run as frozen; check construct-name overlap (expect ≈none: hierarchy node names vs bank
metric names) at analysis time and flag any shared construct.

## Status log

- 2026-07-08 15:0x — slate v1 built (72); 4 Sonnet authoring agents launched (explanation +
  dossier per metric; planted rules verbatim-preserved by instruction). Fleet: cr-retry (9 old
  executors × code_review) running GPU7 → then run_newmodels.sh (gemma2-9b, mistral-24b,
  qwen25-32b: battery→humor285→9 tasks); llama70b panels GPU1 (2/8), qwen25-72b GPU4 (4/8);
  405B dynamic waiter alive; hermes-3-405B 38-metric expand + hermes-4-405B core probes running
  via OpenRouter. code_review rc=1 root cause was GPU contention (engine-init), not the freeze.
- 2026-07-08 15:2x — INSTRUMENT FACT verified in code: `orbit_metric_verdict(n_forms=1)` scores
  the canonical rubric string verbatim (forms list = [("canonical", rubric)]) — arms are
  unmodified. Sweep armed on sk3: `outputs/osl_multi/run_zxa_small.sh` (waits ≤3h for freezes,
  then task-outer loop humor→cw→peer→math × llama1/3/8B + qwen 3/7/14B + gemma2-27B (+9B when
  weights land), `--n-forms 1`, resumable, outputs `mbar_zxa_<task>_<exec>.npz`), inserted into
  the GPU7 chain between cr-retry and run_newmodels. cr-retry confirming contention diagnosis:
  code_review panels now landing rc=0 (llama1b/3b/qwen3b/mistral7b done by 15:14).
- Authoring: agents 1, 2, 4 landed + validated (word windows pass, planted rules verbatim —
  an apparent agent-4 planted failure was a smoke-test join bug: planted NAMES repeat across
  tasks, key by (task,name); build script already does). Agent 2 deliberately differentiated 5
  near-duplicate incongruity constructs by mechanism; agent 4 same for 3 Lakatosian math
  constructs — good for the specificity placebo (mismatched bodies are plausible-but-wrong).
- 2026-07-08 15:34 — ALL 4 authoring agents landed; `build_zxa_freeze.py` PASS (72/72; one
  WARN: adjacent-metric name "Delivery, timing, and commitment" appears in the boundary section
  of the genre/delivery dossier — legitimate boundary-case writing, kept). FREEZES BUILT +
  UPLOADED to sk3 osl_multi: humor 246 entries sha 048e91f72bb9 · creative_writing 66 sha
  e5496095a34d · peer_review 60 sha f6853c771643 · math 60 sha 5bed95c8d7ed. Padding matched to
  own dossier within ≤7 words. Spot-check (Distinctive personal comic voice): ladder correct;
  mismatched arm = comic-voice name atop quotable-phrasing body (derangement partner) —
  plausible register, wrong content, as designed.
- Fit-time caveats: (1) peer entry "Actionable clarity/style improvement suggestions" judges
  the REVIEW document while peer probes are paper texts (agent-3 flag) — scope-mismatch, keep
  but flag; within-metric arm contrasts remain valid (same probes across arms). (2) Explanation
  word window slightly exceeded post-assembly (name prefix added) — analysis uses recorded
  per-arm n_words covariate, not nominal windows.
- NEXT (armed, no action needed): cr-retry (7/9 done, all rc=0) → run_newmodels.sh line 13
  fires run_zxa_small.sh (freezes present, wait-loop skips) → ~32 runs ≈ 4-5h → then newmodels
  panels resume. Tomorrow: write zxa_fit.py (per-metric 6-arm × ladder fits; interim y =
  planted truth-AUC + provisional cross-executor consensus; frontier consensus y once 70B/72B
  z×a arms run after their v2 panel chains) → first β/z*/ΔL table.

## GLM ladder leg (user directive 2026-07-08 ~16:00)

User: "fully use the GLM family... two keys under /lfs/.z-ai*... free if we use the anthropic
endpoint https://api.z.ai/api/anthropic. use these endpoints to test tacitness across all GLM
endpoints." ⇒ GLM = a WITHIN-FAMILY FRONTIER LADDER for the z×a surface — puts the metric-B
cells (bare phrase at frontier) on the x-axis immediately, no GPUs. Rungs (battery z, hard
bal_acc scale): glm-4.5 1.64 · glm-4.5-air 1.735 · glm-4.7 1.735 · glm-5.2 1.985 · glm-4.6
auto-battery queued · glm-5 alias-check running (if its battery == 5.2's, it's an alias — skip).

- Runner `outputs/osl/zxa_glm.py`: freeze-driven analog of glm_mbar_probe.py — every
  freeze_zxa entry (arm shown verbatim) × 300 probes, HARD YES/NO readout (Anthropic-style
  endpoint has no logprobs; binarize local m̄ for apples-to-apples), resume-safe by entry name,
  auto-battery for new rungs → `mbar_zxaglm_<task>_<short>.npz`, hard_readout=1. BUGFIX at
  launch: battery items use criterion/truth keys, not rubric/label (KeyError caught on the
  alias-check run, patched, relaunched; the two humor runs were unaffected).
- Two-key priority queues `outputs/osl/glm_zxa.sh` (K1: 5.2 humor→cw→peer→math, 4.5 humor;
  K2: glm-5 alias battery, 4.7 humor, 4.6 humor (auto-battery), 4.5-air humor, 4.7 minis).
  Paused the in-flight expand/go-BIG queues by PID (all resume-safe, name-keyed appends);
  each queue tail AUTO-RESUMES the paused work (K1: 45air+4.6 expand → K1-DONE → glm_big.sh 1;
  K2: glm_big.sh 2). First entry sane: "Australian humor conventions||name" yes=.15 nan=0.
- Fit note: GLM y joins as hard-class agreement (binarized), orderings-not-levels across
  dialects; within-GLM β fits use the GLM battery z (hard scale) — flag that hard-z and
  soft-z batteries are the same items but different readouts; report per-family β and only
  compare RATIOS across families.
## FIRST FIT (2026-07-08 ~19:30) — instrument lessons + first result

Fitter `outputs/osl_multi/zxa_fit.py` (local copy in osl_multi_local), three versions in one
evening, each catching a real artifact:
- **v1 lesson (same-arm-consensus trap):** y = agreement with frontier consensus AT THE SAME ARM
  measures prompt DETERMINACY, not content — the mismatched placebo scored as high as the true
  dossier (everyone following the same wrong body agrees). Kept as the form-determinacy readout;
  β must NOT use it.
- **v2 fix:** y_ref = agreement with frontier consensus FIXED AT THE DOSSIER ARM of the same
  base (content-anchored referent). Frontier = explicit list (glm-47, glm-52, +llama70b/
  qwen25-72b/hermes when landed) — never a z-threshold (hard/soft z scales don't mix).
- **v3 fix (base-rate trap):** raw agreement inflates for skewed executors (qwen25-3b answers
  ~always-NO, yes_rate .007 → raw y_ref .95 vs mostly-NO consensus). All readouts now
  chance-robust: balanced per-ref-class agreement (constant predictor = .5 exactly) + kappa for
  frontier pairs + degeneracy flags (yes-rate extreme, nan>10%, planted-chance).

**Degenerate executors on this template (n_forms=1 textfirst):** llama1b (planted ~.50 flat =
reading collapse; its earlier "name .60" was base rate), qwen25-3b (constant-NO). gemma2-27b
borderline (planted def .638, rest ~.5) — kept with caution.

**First humor readout (balanced y_ref; GLM partial ~200/246 entries, single-ref frontier):**
- PLANTED gates: mismatched COLLAPSES (14b .871 def → .544 mism; 7b .791→.497; 8b .761→.509);
  planted NAME arm ≈ .50 everywhere (cryptic slugs carry nothing — an anchor case of
  "phrase fails, definition works at every scale ≥3B" for mechanical content).
- **Content-specificity gate PASSES everywhere**: dossier_mismatched ≈ .49-.55 at ALL scales
  and classes (the v1 "wrong dossier helps too" was entirely artifact).
- **DIALECT-SUSPECT = transmissible**: locals climb with articulation (llama8b name .737 →
  dossier .869, placebo .496; qwen7b .653→.821) — content reaches 8B.
- **TACIT-CANDIDATE = frontier-only**: locals flat .52-.60 on EVERY arm (articulation buys ~0
  below ~14B); GLM rungs climb name .69-.71 → dossier .66-.78; dossier−mismatched rises with z
  (8b +.022, 14b +.065, glm-47 +.110, glm-52 +.295). β(a)≈0 at small z for tacit content —
  the capability floor, not the prompt, is binding. (glm-47's dossier dip = single-ref noise.)
- Missing tonight: REACHES/PLANTED y_ref rows (GLM hadn't reached late entries), gemma2-9b
  (weights landed after humor pass — morning re-pass), 70B/72B arms (after their v2 chains).

## FORMAL FORECAST OBJECT (user directive 2026-07-08 night — "metric-B scaling law")

User's target statement, made rigorous. What we can and cannot identify:

**Definitions (per metric j, capability z, criterion c):**
- y_j(z, a) = balanced agreement with the frontier dossier-consensus referent (content-anchored).
- Articulation-buyable deficit  **D_j(z) = y_j(z, dossier) − y_j(z, name)** — the explicit
  supplement the model still needs at capability z ("tacit knowledge Y at size X").
- Articulation-resistant residual  **R_j(z) = c − y_j(z, dossier)** — what NO tested
  articulation buys at z.
- Internalization threshold  **z*_j = min{z : y_j(z, name) ≥ c}** — the capability at which
  the community's word alone suffices (enculturation point). Metric-B question ≡ estimate z*
  for metrics where name fails at every observed z.

**Estimator (shift model, per family):** y_j = L_j(a)·σ(k_j(z − z0_j + β_j(a))), β_j(name)=0.
If the shift model holds (placebo-gated: β uses only the content-specific component,
β_content = β(dossier) − max(β(mismatched), β(padded))):
  **ẑ*_j = ẑ_crit,j(dossier) + β̂_j(dossier)** — measure where the dossier curve crosses c
  (interpolation on the observed ladder), then add the fitted horizontal offset. Bootstrap CI
  over metrics×probes. If instead the ΔL model wins (dossier asymptote < c), then z* = ∞ under
  this family: "no finite scale of this family internalizes it" — the strong tacitness verdict.
  Model comparison (shift vs ΔL, LOEO by executor) decides per metric/class.
- Criterion policy: primary c = .75 balanced; sensitivity at c = frontier dossier-consensus
  split-half ceiling. Report both.

**Translation to size/data (the user's X→XXX, T→TTT):**
1. z→params: within family, battery-z vs log N is monotone (Llama 1/3/8/70/405; Qwen 3-72;
   GLM 4.5→5.2). Fit z = g(log N) on observed rungs → **N* = g⁻¹(z*)**, reported as
   "Llama-family-equivalent parameters" with CI. HONESTY: z* beyond the top rung is a
   family-conditional forecast under the declared parametric family (Koyejo guard: ordinal
   language, falsifiable by the next rung — the 405B point tests exactly this).
2. z→data: NOT identified from our data. Only via a stated coupling assumption
   (compute-optimal N∝T, Chinchilla ~20 tok/param) may one append "≈20·N* training tokens" —
   label as assumption-laden appendix arithmetic, never a finding. The defensible framing is
   enculturation: what grows along z within a family is exposure to community practice; z* is
   an exposure threshold in capability units.
3. "Tacit knowledge Y" decomposes as (D_j(z), R_j(z)) — buyable vs resistant. Report both at
   each rung; their z-profiles are the scaling law of tacitness.

**Template sentence the fits will fill:** "Metric j at 8B carries deficit D=…; fitted
β(dossier)=… z-units ≈ the 8B→70B jump; the phrase-only curve reaches c at ẑ*=… [CI],
≈ …B Llama-equivalent params; under the shift model this is internalizable, under ΔL it is not."
First fills at the morning pass (needs ≥4 z-points per family incl. frontier: Llama +70B(+405B),
Qwen +32B/72B, GLM 4 rungs).

## MECHBAT: mechanical-capability battery (user 2026-07-08 night)

"Nail down what the LLM can/cannot do that code CAN (scanning/regexing)" — a capabilities
test in its own right, and the calibration backbone for planted gates. Built:
`build_mechbat.py` → `freeze_zxa_mechbat_{humor,peer_review}_v1.json` — ~19 code-verifiable
rules/task across families {presence_char, presence_word(mid-freq, corpus-adaptive), casing,
pattern, length, count, position(prefix/suffix), negation, parity(hard), order(2-token)};
truth computed by code ON THE SHOWN SLICE and stored INSIDE the freeze (mech.truth);
base-rate filter [.15,.85] per corpus. The DROPPED lists are themselves corpus facts (humor:
no digits/URLs/parens; peer: hyphens 97%, question marks 2% — negations/parities of rare
events are untestable there). Runner-compatible with osl_sweep + zxa_glm unchanged.
- ARMED: `run_mechbat.sh` (locals incl. new executors + gemma2-9b block-size fix; ~1-3 min
  per exec-task — run on GPU7 when free, morning).
- GLM rungs: insert `zxa_glm.py <model> <short> mechbat_humor <key>` (+ peer) into the key
  queues at the next safe queue-edit window (morning; do NOT run concurrent with an active
  key process — throttling, see NaN lesson).
- Readout: per family×locality×length accuracy map vs code truth, by executor scale — where
  does "LLM as scanner" break: aggregate/counting first? parity ever? position vs global?
- v2 expansion candidate: corpus-adaptive thresholds so every family survives both tasks
  ("at least K digits", K tuned to median) — after v1 shows the interesting boundaries.

## EVENING HARVEST (2026-07-08 ~22:00)

**State:** local small-ladder COMPLETE (28/28 = 4 tasks × 7 execs; gemma2-9b failed all inits —
FlashInfer head-256, 27b escapes via head_dim=128; VLLM_BLOCK_SIZE=32 fix now patched into
run_zxa_small.sh + run_newmodels.sh, morning re-pass picks it up). GLM: glm-52 humor 166/246,
glm-47 85/246 (~15-29 entries/h at llm_concurrency=8), then key1→5.2 minis, key2→4.6 humor.
qwen25-72b v2 panels 6/8 (peer running); llama70b on press_releases+. 405B waiter alive.

**CONCURRENCY LESSON (recorded the hard way):** bumping llm_concurrency 8→24 produced 21-27%
NaN on the first resumed entries (endpoint throttling → validate-retry exhaustion) vs 0.00-0.01
at 8. REVERTED to 8 — speed < data quality. The poisoned rows were almost certainly never
persisted (saves every 6 entries; npz mtimes 21:43/21:49 predate the 24-conc window; runs
resumed from 162/84 = pre-poison counts) — morning pass should still sanity-scan
mbar_zxaglm_humor_glm-{52,47}.npz for any row with >10% NaN and, if found, delete those rows
and re-run (resume-safe redo; verify replacement before overwrite).

**4-task fit (zxa_fit v3 + metric-B table + truncation-aligned planted truth):**
- Truncation alignment CHANGED NOTHING (peer 14b def .598→.598) ⇒ weak mini-task planted gates
  are REAL DOMAIN DIFFICULTY: mechanical rules on dense academic prose execute far worse than
  on short humor texts at ≤14B (humor def .87 vs peer .60 / math .63 / CW .73 at qwen25-14b).
- DEGENERACY IS TASK-DEPENDENT: humor {llama1b, qwen25-3b}; CW adds {llama3b, gemma2-27b}
  (llama1b flips to YES-spam .877); peer/math add qwen25-3b/llama3b (llama1b .84/.77 yes).
  Small locals are only usable executors on SHORT-text tasks; the z×a mini-task small-ladder
  rows are mostly noise below 8B.
- Humor contrasts (fuller GLM): dossier−mismatched gradient holds — 8b +.028, 7b +.028,
  14b +.089, glm-47 +.110, glm-52 +.286. glm-52: def−name +.089, dossier−name +.100,
  explanation adds ≈0 (+.006), padded ≈ def. (glm-47 absolute dossier cell noisy — single-ref.)
- **METRIC-B TABLE v0 (9/16 tacit candidates covered; single-ref frontier, orderings only):**
  clearest "dossier transmits at frontier / local can't use it" = the COMPRESSION cluster:
  one-liners f_name .691→f_doss .829 (l_doss .666 ≤ l_name .722); quotable phrasing .674→.811
  (local flat); concision .667→.782 (local .512 flat). PHRASE-SUFFICIENT-at-frontier so far:
  originality (.750), humor-pathos (.762), character/impersonation (.783 — but its dossier
  cell .542 < mismatched .747 = partial-coverage artifact, DO NOT interpret until full data).
  7 candidates (incl. comic voice, Australian conventions, roast tone) await GLM late entries.
- Fit artifact: outputs/osl_multi/zxa_fit_v1.json regenerated (all 4 tasks).

- 2026-07-08 ~16:40 ALIAS + PAAS AUDIT: served-model probe shows glm-4.5-air→glm-4.7 and
  glm-5→glm-5.2 (battery double-confirm: 5alias .881/2.000 ≈ 5.2 .879/1.985) ⇒ TRUE GLM ladder
  = 4 rungs (4.5 / 4.6 / 4.7 / 5.2). Queued 45air humor leg neutralized via alias-skip stub
  (mbar_zxaglm_humor_glm-45air.npz, alias_skip=1 — fit must EXCLUDE files with alias_skip; the
  resumed key1 "45air expand" run is 4.7-expand data misnamed, treat executor glm-45air ≡
  glm-47). PAAS endpoint: both keys code-1113 insufficient-balance (subscription = anthropic
  route only) ⇒ no logprobs available on any free GLM route; soft-upgrade option if contrasts
  are noise-limited = k-sample voting (temp>0, ×k calls, free) on the tacit subset. Pace:
  glm-52 humor 49/246 in ~50min (≈4h/rung-task), glm-47 23/246.

## MORNING HARVEST (2026-07-09 ~07:40-08:20) — triage, repairs, FIRST CROSSINGS

**Chain triage (targeted-PID kills only, all resume-safe):**
- GLM queues healthy: K1 finished glm-52 humor 246/246 (rc=0 02:52) + creative_writing 66/66
  (06:04), on peer_review; K2 finished alias battery, glm-47 humor 198/246 in flight. glm-46 /
  glm-45 rungs queue behind (each auto-runs its battery → 3rd+4th GLM z-points for the ladder fit).
- newmodels chain was WEDGED 12h: mistral-24b snapshot_download at "0/22 files" with nothing
  written to cache in 12h (stale-lock/NAT64 wedge). Killed download 2293413 + parent 1704715,
  cleared the two repos' .lock files, relaunched both downloads (mistral resumed from 23G,
  qwen25-32b fresh) — mistral immediately progressing.
- llama70b v2-panel driver (resume_fleet_after_405) died silently 23:11 Jul 8 mid-math
  (2/8 tasks done vs qwen25-72b 8/8). Its remaining 7 panels are chained AFTER the frontier z×a.
- gemma2-9b post-mortem in two acts: (1) Jul 8 failures = the FlashInfer block-16 assert
  (shell-script env fix landed AFTER those attempts; backend passthrough had existed since
  Jul 6); (2) TODAY the engine got block_size:32 (visible in args) but transformers'
  HEAD-revalidation of config.json hit the NAT64 read-timeout flake and errored DESPITE a
  complete cache. Fix: HF_HUB_OFFLINE=1 on all compute invocations (downloads stay online) via
  mv-replace into run_zxa_small.sh / run_mechbat.sh / run_newmodels.sh. gemma2-9b
  creative_writing z×a then completed rc=0 (07:59) — executor fully unblocked; humor picked up
  by the inner re-pass.
- 405B: battery DONE via OpenRouter — hermes-3-llama-3.1-405b z=1.836 (auc .8625),
  hermes-4-405b z=1.285 (Hermes-4 scores BELOW llama70b 2.128 on hard readout; Nous fine-tunes
  are NOT Meta lineage ⇒ excluded from the Llama z→params map). 405B z×a arms = OpenRouter
  spend decision (74k calls full slate / ~29k tacit-subset) — flagged, not launched.

**NaN correction (evening assumption was WRONG):** the conc-24 poison DID persist — humor
glm-52 90/246 rows >10% NaN (up to 20%, overall 9.5%), cw glm-52 6 rows, glm-47 humor 25+ rows;
peer_review glm-52 (fully post-revert at conc 8) is 0.000 NaN. Repair lane running: backup →
drop rows >10% NaN → zxa_glm resume redoes exactly those entries at ZXA_CONC=4 (new env knob)
→ verify count + new-row NaN ≤5% else restore backup. glm-47 humor repair waits for its writer
to exit. Tail of the lane = mechbat GLM legs (glm-52, glm-47 × humor/peer) since running queue
scripts can't be edited.
- **Fitter lesson (backup leak):** the first v4 fit ingested mbar_zxaglm_humor_glm-52_prenanfix
  .bak.npz as an "executor" (glob matched); its TACIT dossier cell read 1.000 = self-agreement
  with glm-52 in the reference. Fitter now skips prenanfix/.bak files. Backups stay on disk
  (never-delete) — they are just not executors.

**zxa_fit v4/v4.1 (shipped + live in the auto-refit hooks):** section 6 = the formal forecast
object implemented: per-family (llama/qwen25/glm/gemma2) isotonic (PAV) y_ref-vs-z per
metric × arm; crossings at c=.75 with explicit censoring; β̂ = zc(name)−zc(dossier) interior-only
else lower bound; pooled-β placebo correction; ẑ* observed (metric-A) or forecast
zc(dossier)+β_pool (metric-B, pooling level labeled); N* through the SAME family's battery
z~a+g·log10(params) map (llama/qwen only — GLM z is hard-scale + params unknown); T*=20·N*
printed as Chinchilla assumption-arithmetic. Shift-vs-ΔL first pass = top-rung gap + PLATEAU
flag (flat last step below c). Degenerate rungs STAY in scaling ladders (balanced y_ref ≈ .5 is
the honest left edge — a constant predictor scores exactly .5) but stay OUT of frontier/metric-B.

**FIRST CROSSINGS (humor, c=.75; ladders llama 1b/3b/8b, qwen25 3b/7b/14b — pre-70b/72b):**
- qwen25 × "Accumulation and format-subversion" (DIALECT): zc_name=1.92, zc_doss=1.62 ⇒
  **β=+0.30 z-units** (≈ the 7B→14B hop); z*=1.92 observed ⇒ N*≈1.3e10 (~13B qwen-equiv),
  T*~2.6e11 tok (assumption). First complete metric-A row of the program.
- qwen25 × "Self-deprecation" (DIALECT): dossier crosses 1.92, name censored >2.00 ⇒ FIRST
  FORECAST ROW: ẑ*=2.22 [class-pooled β] ⇒ N*≈3.4e10 (~34B qwen-equiv, EXTRAP) — falsifiable
  TODAY by qwen25-32b (downloading; z expected ~2.1-2.2) and qwen25-72b (z=2.338, z×a running).
- llama × "Self-deprecation": name crosses 1.34 (N*≈5.4e9) while dossier does NOT cross by
  8B — an inversion (n=3 rungs; treat as noise candidate until 70b lands).
- ALL TACIT-CANDIDATE rows right-censored on both local ladders (every arm >z_max) with
  PLATEAU flags on concision (both families), originality (llama), specificity (qwen) —
  the articulation-resistant signature, awaiting 70b/72b/GLM rungs.
- Placebo pool ≈ 0 where estimable (llama REACHES-ANCHOR −0.04) — placebo arms buy nothing.
- METRIC-B table refresh: "Shared knowledge and reference accessibility" is the cleanest
  PHRASE-FAILS/DOSSIER-WORKS row (f_name .562 → f_doss .796, mism .501). 6 tacit rows nan =
  glm-47 partial + repair holes; fills as those land.

**In flight (all self-recording):** GPU4 zxa_frontier chain (llama70b humor→cw→peer→math →
mechbat ×2 → AUTO-REFIT → qwen25-72b same → AUTO-REFIT → llama70b 7 remaining v2 panels;
fit logs = zxa_fit_after_llama70b.log / _after_qwen25-72b.log). GPU7 morning chain
(zxa_small re-pass → mechbat small ladder → newmodels: gemma2-9b full redo + mistral-24b +
qwen25-32b battery/z×a/mechbat/panels — z×a arms inserted into run_newmodels.sh post-battery).
GLM repair lane (PID 3337543). Both key queues. Downloads 3329901/3329902.

### 10:15 checkpoint — LLAMA 4-RUNG LADDER LANDED (zxa_fit_after_llama70b.log + clamped rerun)
Recovery sprint first: downloads punched through on watchdog attempt 5 (~09:10; the CDN
blackhole was confirmed by a 0-byte curl probe); mistral-24b battery z=2.101 + all 6 z×a/
mechbat legs done by 09:47; gemma2-9b z=2.043 (INVERTS gemma2 family — 27b z=1.10; 9b also
DEGENERATE on peer/math planted = small-local long-text rule again); qwen25-32b priority
chain launched on GPU2 (battery→z×a→mechbat→own refit — the forecast falsification rung,
was stuck behind mistral panels); queue supervisor resurrected mbar_fleet.sh which lost the
GPU4 race (12s rc=1s) and settled on GPU6 doing the dead llama70b panels — coexistence fine.
- **FITTER CLAMP FIX (logical-consistency bug):** forecast rows could print ẑ* BELOW the
  name-arm's right-censor bound when pooled β ≤ 0 (impossible). Now: ẑ* = max(zc_doss+β_net,
  z_max); when β can't push past the bound, row reports the BOUND (z*>z_max, N*>N(z_max)),
  flagged z_star_is_lower_bound. Shipped before the qwen32/72 refits.
- **HUMOR, LLAMA LADDER (1b/3b/8b/70b):** the TACIT compression cluster gains interior
  DOSSIER crossings: one-liners zc_doss=1.88, compressed-quotable 2.01, concision 2.04
  (≈27-37B Llama-equiv via the params map) while NAME stays censored at 70B ⇒ z*>2.13 =
  **phrase-only competence needs >70B Llama-equiv; dossier transmits from ~30B-equiv** —
  the metric-B sentence with real numbers, β lower-bounds >0.09-0.25 and growing. Qwen
  ≤14B: same cluster fully censored BOTH arms — family asymmetry (llama70b can use the
  dossier where 14B-qwen cannot; frontier metric-B row: one-liners f_name .720→f_doss .889).
- **ARTICULATION CAN SUBTRACT (new sub-story):** llama Concrete-imagery name crosses BEFORE
  dossier (1.55 vs 1.81, β=−0.26); REACHES-ANCHOR pooled β=−0.52; peer_review anchors show
  the DEFINITION arm collapsing (~.52) while name/explanation/dossier track consensus
  (.72-.99). For already-indexed metrics the added text DELAYS/derails — articulation helps
  where knowledge is missing, hurts where the name already suffices (ties to name=index /
  what-gets-decompressed). Placebos: mismatched never crosses anywhere ✓.
- **peer TACIT frontier kappa = 0.000 on dossier** (llama70b vs glm-52) → no consensus
  cells for locals = the tacit candidate is underdetermined AT the frontier on peer.
- Transients: repair lane mid-rebuild (glm-52 humor ~180/246 during this fit) blanked some
  anchors (Self-deprecation qwen row temporarily absent; 8 metric-B rows nan) — refills by
  ~11:30. Character/impersonation dossier still artifact-flagged (mism .797 ≈ name .839 ≫
  doss .556) — spot-read that authored dossier before interpreting.

### 17:50 checkpoint — THREE FULL LADDERS → THE HEADLINE FLIPS: EXCHANGE RATE IS FAMILY-RELATIVE
Data now: Llama 4-rung (1b/3b/8b/70b), Qwen 5-rung (3b/7b/14b/72b/32b), GLM 4-rung humor
(45/46/47/52, all 246 rows clean). qwen25-32b landed at battery z=2.546 — ABOVE qwen25-72b
(2.338) = within-Qwen NON-MONOTONICITY (32B-Instruct genuinely strong; the z→params map is
now non-monotone at the top so Qwen N* estimates above 14B are UNRELIABLE — report z*, not N*,
for Qwen top rungs). GLM ladder is bunched (1.63-1.985) and STARTS above criterion for most
metrics → mostly left-censored (`<=1.63`), limited crossing-resolution: a within-family ladder
must SPAN c, GLM's rungs are all at the capable end.

**THE STORY CHANGED — "metric-B / tacit" is turning out FAMILY-RELATIVE, not metric-intrinsic.**
The compression cluster (one-liners / compressed-quotable / concision) was my clean metric-B
example all morning. With all three ladders it splits by family, SAME metric, opposite sign,
content gate holding throughout (mismatched never crosses: >1.98 / >2.13 / >2.55 everywhere):
  - Compressed-quotable phrasing:
      Llama  zc_name >2.13 (censored past 70B) / zc_doss 1.88 / β>+0.25  = METRIC-B (phrase
             fails through 70B; dossier transmits from ~30B-equiv) — the morning story.
      Qwen   zc_name 1.63 (~7B) / zc_doss 2.36 / β=−0.74            = METRIC-A w/ HARMFUL
             dossier (phrase suffices from ~7B; the 400-word dossier DELAYS competence).
      GLM    both solved at the 1.63 floor (can't resolve; already-internalized by glm-4.5).
  ⇒ what needs a dossier in one family is already indexed from the bare name in another. The
    exchange rate β is a property of the (metric × family) PAIR. Mechanistic sibling of the
    OSL result already in memory (size-slope family agreement ρ≈.1 = size-scaling is dialect-
    specific); now the ARTICULATION-uptake is dialect-specific too, and can flip sign.
  ⇒ metric-B verdict-column FLIPPED to PHRASE-SUFFICIENT for the whole compression cluster
    once qwen25-32b (z=2.55) became best_local (one-liners l_name .858/l_doss .946) — the
    "MID/phrase-fails" morning reading was an artifact of the local ladder topping out at 14B.

**"ARTICULATION SUBTRACTS" (β<0) is now a ROBUST second regime**, multiple families × classes:
Llama REACHES β=−0.61, Llama concrete-imagery −0.27; Qwen TACIT-pool −0.27, Qwen compressed-
quotable −0.74. Dossier helps where knowledge is MISSING, hurts where the name already
suffices — signed, and the sign is family-conditional. PLANTED sanity check fires correctly:
Llama PLANTED β=+1.11 (slug carries nothing, definition executes the rule = huge positive).

**THE GENUINELY-TACIT CANDIDATES STILL DON'T YIELD — and that's the strong-tacit signature.**
The 12 voice/persona/enculturation candidates (comic voice, impersonation, Australian
conventions, roast warmth, satire, host presence…) = nan in the metric-B table because
frontier dossier-consensus has <30 strict-agreement probes (kappa≈0): even the frontier
models DON'T CONVERGE given the full 400-word dossier. So they can't be forecast — not because
z* is large but because z* is UNDEFINED (no convergent target to forecast toward). Cleanest
operationalization yet of the fully-tacit layer (links tacitness-two-layers C<1, what-gets-
decompressed): the forecastable metrics (compression/concision) were always the tractable
ones; the originally-flagged tacit candidates remain frontier-non-convergent. The program now
bifurcates cleanly: (A) tractable metrics → forecastable but family-relative A/B; (B) genuine
tacit → no frontier consensus on the dossier = un-forecastable by construction.
Artifacts: zxa_fit_1750_humor.log / _all.log; zxa_fit_v1.json (4-task) regenerated.

### 2026-07-09 ~21:00 — CAVEAT AUDIT (user-directed) + FOUR FIX LANES LAUNCHED

**DEGENERACY AUDIT (zxa_degen_audit.py, sk3; per task × exec × arm nan/const/yes + length
correlations). Two mechanisms, neither is parse failure (NaN≈0 on all local execs):**
1. *Small-model constant-collapse scales with ARM LENGTH*: r(arm-words, const-row) positive for
   every small/mid model (llama1b +.55/.66, llama3b +.43, mistral-24b +.28, qwen25-3b +.29);
   direction is model-specific (llama1b→all-YES y=.98, qwen25-3b→all-NO y=.00). Dossier collapses
   MORE than the length-matched padded control (llama1b cw: doss .91 vs padded .45 const) ⇒
   instruction load, not raw tokens. Balanced-y scores collapsed cells .5 exactly (honest), but
   small-z dossier points are partly form-collapse, not knowledge absence — part of what "capability
   buys" is surviving long instructions at all.
2. *FRONTIER const on dense tasks = probe-variance saturation*: peer_review worst — llama70b .40,
   qwen25-72b .30, qwen25-32b .40-.50, glm-52 .30 const rows; metric never varies on the
   homogeneous 300-probe slice. Also peer name-arm yes-rates at frontier are extreme (.01-.13):
   bare phrase read as a strict bar. cw probes 36% TRUNCATED at 4000-char cap. ⇒ dense-task
   "few BOUNDED metrics" is partly a probe-slice power problem, NOT only construct absence.
   Audit-pool check: slate leftovers cw 3 / peer 1 / math 0 ⇒ pool exhausted; the lever is probe
   redesign (peer p900 pilot designed, queued), not slate mining.

**PROBE-SLICE COMPARABILITY (user's isomorphism concern — CONFIRMED ISSUE):** shown-char medians
humor 81 / peer 1,298 / math 1,276 / cw 3,158(36% truncated). Cross-task "isomorphism" is
currently measured across wildly different support regimes. Documented; matched-length probe
sets = design requirement for any cross-task tacitness comparison.

**TACIT κ≈0 DECOMPOSES (subtask_kappa.py, frontier dump × 50-topic model):** the "fully-tacit"
signature has two faces —
  (A) *construct-not-applicable-on-slice*: host presence (raw unanimity .99, κ=.10), brand identity
      (.99, κ=.00), delivery/timing, personal authenticity, roast craft, character/impersonation:
      frontiers agree all-NO on 81-char one-liners; κ≈0 is a base-rate artifact. NOT yet evidence
      of tacitness — needs long-form support first.
  (B) *genuinely contested*: reference accessibility (unanimity .22!), concision/word-economy (.28),
      compressed-quotable (.52): frontiers actively split = real content-level underdetermination.
Voice κ drops .93→.63 unanimity on longer probes = the construct starts to LIVE there.
Within-topic κ unresolvable at 300 probes / 50 topics (~6/topic) → folded into LP lane (below).

**LANES LAUNCHED (sk3):**
- *FAM (same-family decompression, user request)*: author_fam_arms.py — llama70b + qwen25-72b each
  author explanation(130-180w)+dossier(360-450w, 4 labels, planted-verbatim) for ALL 72 slate
  metrics; gates byte-identical to v1 Sonnet arms. llama70b: 66/72 valid first pass (~11 min);
  mop-up pass reruns invalids. build_zxa_freeze_fam.py → freeze_zxa_<task>_fam_v1.json, 4 arms
  (expl/doss × fam_llama/fam_qwen); 2×2 author-family × executor-family = the dialect-matched
  decompression test. Lane fam_zxa.sh GPU2 (PID 4184141): authoring → freezes → 9-exec sweeps ×4
  tasks. y_ref stays v1 dossier consensus (fixed target; fam arms are just different articulations).
- *LP (long-form humor probes, cluster-A fix)*: humor_long_probes.jsonl = 360 topic-stratified
  reddit long-tail texts (800-4000 chars, med 1018 = 12× v1 support, 22 topics, 12 at 30/topic,
  stable-hash selection). Same freeze_zxa_humor_v1 arms via OSL_PROBES_FILE. OUT=mbar_zxaLP_humor_*
  (glob-safe vs fitter). lp2_zxa.sh GPU0 util .60 smalls+32b (PID 227893); GPU1 chain (PID 377215)
  llama70b+qwen25-72b then news ladder. First LP failure = engine-init 8.9GiB free (GPU6 claimed by
  another workstream between snapshot and launch) — split-GPU relaunch, retry-on-init-fail wired in.
  Planted-truth NOTE: frozen planted rules' truth must be recomputed on LP probes by parsing the 5
  rule sentences (planted_metrics(LP) would generate different rules); readout-side task.
- *NEWS (journalism z×a, user request)*: probes = homepage_newsworthiness corpus, junk-filtered
  (headline-slot gates + English stopword gate, 38% pass, med 2,030 chars; news_probes.jsonl).
  Slate news_slate_v1.json (21): 10 TACIT-RECRUIT (voice/newsworthiness/proximity/tabloid-boundary/
  curation — recruited BY TYPE mirroring humor's audited cluster, class name keeps that honest),
  2 audited DIALECT-SUSPECT, 4 audited REACHES-ANCHOR, 5 PLANTED regenerated on curated probes
  (k_med=296w). 4 Sonnet agents authoring 6-arm texts (v1 gates + self-validation);
  build_zxa_freeze_news.py validates+assembles (derangement seed 20260709, FILLER padding);
  news_zxa.sh waits for freeze then runs the full 9-exec ladder (GPU1 chain, after LP bigs).
  Census anchor: journalism = strongest subtask dialects (Δ+.0547) ⇒ prime DIALECT hunting ground.
- *Hierarchy R1 expansion (separate program)*: background orchestrator continuing L0→rename→R1 for
  news-homepages + math-se per ledger runbook, Sonnet fleets ≤20 agents / ≤150-pair shards,
  exact-paths-in-prompts discipline (user directive relayed ×2).

**MODEL-LADDER NOTES:** gemma2 EXCLUDED from ladder claims (27b z=1.102 < 9b z=2.043 inversion +
humor yes≈.5 noise profile — battery-level anomaly, investigate before any gemma rung is used);
qwen25-32b>72b non-monotonicity stands (report z*, not N*, above 14B); mistral-24b = single rung
(no ladder), notably const-prone for its size (.24-.60 dense).

### 2026-07-09 ~21:40 — COLLAPSE-SENSITIVITY IN THE FITTER (GEPA discrimination-gate practice
propagated to the read side, user-prompted). zxa_fit.py now flags cell-level constancy
(exec×base×arm rows with finite≥30 and std=0) and re-fits name/dossier crossings with collapsed
cells MASKED; rows whose crossing status/z moves >0.15 print !COLLAPSE-SENS(nR:nm…/ds…) and carry
collapse_sensitive/n_const_rungs/cross_*_masked in zxa_fit_v1.json. FIRST RUN (zxa_fit_2130_
collsens.log): **17/63 humor metric×family crossing rows are collapse-sensitive** (+3/12 on cw) —
incl. qwen×compressed-quotable (name =1.63 → ≤1.66 left-censored masked: phrase-sufficient
verdict survives but rests partly on a collapsed rung) and llama concision-cluster dossier
crossings that DISAPPEAR masked (<3 live rungs). Read: those crossings are articulation-FORM-
limited (small models can't maintain non-constant judgment under the arm at all), not knowledge-
limited. Main numbers unchanged (additive sensitivity). Also: repo-root BEST-PRACTICES.md created
(user request) consolidating all standing rules; z×a scaffold-vs-frozen-object boundary stated
under [prompt optimization]. NB census MIRROR-CONFOUND revision (other session, 2026-07-09):
journalism/math author-dialects were shared-canonical-text artifacts — "journalism strongest
dialects" INVERTED; news z×a slate unaffected (stands on audited classes + by-type recruits, not
on the dialect census).

## 2026-07-09 ~23:30 — LP FIRST READ; FAM AUTHORING QC FAILURE + REPAIR PATCH

**LP lane (probe-support fix for frontier/dense constancy + cluster-A artifact): 8/9 execs
landed** (qwen25-72b at 225/246 on GPU1; news ladder chained after). First read, v1 (81ch) vs
LP (med 1018ch), panel = llama70b/qwen25-32b/qwen25-14b:
- **Cluster-A support-starvation CONFIRMED causal for the flagship case:** Brand identity
  coherence (TACIT-CANDIDATE) name-arm yes-rate .06 -> .24 (develops variance on long support).
  PLANTED-length-long .06 -> .27 (the length rule finally has room to fire) — confirms the
  standing caveat: **recompute planted truth on LP probes before reading planted cells.**
  PLANTED-digit stays floored (.03 -> .07), as it should (digits rare either way).
- **Tacit kappa FLAT: .23 -> .24** (name arm, llama70b x qwen25-32b, n=16 tacit bases).
  DIALECT .30 -> .35, ANCHOR .46 -> .39, PLANTED .01 -> .06. Long support dissolves the
  all-NO artifact but does NOT manufacture cross-family agreement: contested cluster-B stays
  contested. The tacitness signature survives its strongest artifact challenge.
- **Const rows: 0% on humor, all 8 local execs, BOTH probe sets.** Humor's degeneracy was
  near-unanimity (not literal constancy); literal const lives on long-input dense tasks
  (cw/peer) and GLM/frontier arms. Note: 12x longer INPUTS did not induce small-model
  collapse — more evidence the small-model disease is arm-length/task-side, which the fam
  sweeps (cw/peer/math included) will test directly.

**FAM lane (same-family decompression, fix for small-model arm-length collapse): first
authoring run FAILED QC.** qwen25-72b 8/72 valid — it writes 110-125w explanations vs the
130w gate floor (dossiers fine: 4/4 labels, mostly in-band); llama70b 66/72. Freeze build
hard-failed on the missing rows. Engine-core-died lines at the end of author logs are
teardown, not the cause. Patched (backups .bak2):
- author_fam_arms.py: prompt now warns "under 135 words REJECTED, aim ~155"; added a 2-round
  **repair pass** — expand/trim rewrite seeded with the model's OWN draft (well-conditioned,
  unlike blind resampling which failed 6x), gated by the same validators.
- build_zxa_freeze_fam.py: drops bases lacking BOTH-author validity (2x2 stays balanced)
  instead of sys.exit; floor 12 bases/task.
- All 8 GPUs busy at relaunch (GPU2 grabbed mid-window; GPU0 65GB = another user's 49-day
  server). fam_waiter.sh (pid 2226344) polls 120s and launches fam_zxa.sh on the first GPU
  free for 3 consecutive polls. Ops lesson re-learned twice: pgrep guards self-match when the
  pattern's path is in the same ssh cmdline — use `ssh sk3 'bash -s' < launcher.sh`.

### 2026-07-09 ~23:55 — OPENROUTER SCOPE LANES (user-authorized, CPU/API only, no GPU)
Kit: sk3 outputs/osl_multi/orkit/ (or_runner.py = generic freeze x probes -> npz, zxa_glm.py
clone w/ --arms filter + explicit --out; or_author_fam.py = OR authoring, fills invalid rows
only). Launched (setsid, logs in osl_multi/logs/, status lines in FLEET_STATUS):
- or_scope.sh pid 2969316: OR-author fam arms (llama70b via meta-llama/llama-3.3-70b-instruct
  = SAME weights as local; qwen via qwen/qwen-2.5-72b-instruct — NOTE dashed slug works,
  qwen2.5 no-dash slug returns EMPTY) -> build fam freezes -> dump humor_v1_probes.jsonl
  (column-aligned via _load_texts) -> llama-3.2-1b (OR) on humor fam arms ->
  mbar_zxafamOR_humor_llama1b-or.npz = first small-model 2x2 signal without GPUs.
- hermes_lp.sh pid 2969317: hermes-4-405b LP humor NAME arm (41x300) ->
  mbar_zxaLP_humor_hermes405b.npz (battery json already existed, z consistent).
- glm_lp.sh pid 2969318: glm-5.2 then glm-4.7 LP name+dossier (82x300 each) on key B
  (.z-ai-api-key.txt, conc 6) so the running glm-4.7 peer_review chain (key A) is undisturbed
  -> mbar_zxaLP_humor_glm-5x.npz. Completes the frontier-panel kappa re-test on long probes.

### 2026-07-10 ~00:30 — PROBE-OFFSET ARTIFACT CAUGHT (hermes kappa=0.00) + HERMES IDENTITY FIX
Extended kappa run: hermes-4-405B (OR) read kappa .00 vs EVERY local on EVERY kind incl.
REACHES-ANCHOR (locals .4-.6) — too-clean zero = alignment alarm, not disagreement.
ROOT CAUSE: `OSL_PROBES_FILE` convention CONSUMES THE FILE'S FIRST 60 ROWS AS PADDING —
osl_sweep scores file rows 60..59+n; or_runner scored rows 0..n-1. All local<->local LP
results stand (internally consistent); only OR-vs-local was garbage. FIX: or_runner
--probe-offset (60 for raw probe files, 0 for pre-sliced dumps); misaligned outputs kept as
*.rows0-299.misaligned.bak.npz; glm-5.2 restarted @offset60 (was 4/82); stray old-wrapper
glm-4.7 killed (relative-path pgrep miss — absolute patterns only).
ALSO: existing hermes405b battery json = hermes-THREE (nousresearch/hermes-3-llama-3.1-405b,
z=1.836); tonight's runs are hermes-4-405b -> re-keyed as NEW rung `hermes4-405b` (own
battery, family hermes4), LP name @offset60 + v1 name on humor_v1_probes.jsonl dump
(texts[60:360], offset 0) via hermes4_chain.sh. Never mix the two hermes rungs in a ladder.
Healthy hermes-4 profile: name-arm yes median .41 (IQR .15-.73), nan 0.
LP kappa WITH qwen72 (aligned, local): tacit llama70b x qwen72 .24 v1 -> .24 LP (flat,
replicates); within-family qwen72 x qwen32 .47 v1 -> .42 LP (same-family agreement ~2x
cross-family = dialect structure visible at name arm).

### 2026-07-10 ~01:00 — HERMES-4-405B v1<->LP CONTRAST (aligned) + FAM AUTHORING COMPLETE
Hermes-4 rung (OR, OSL_REASONING_OFF=1, offset-aligned both probe sets), name arm:
- **Cluster-A causality replicates in a 3rd family at 405B-class:** Brand identity yes
  .01 (v1 81ch, all-NO) -> .15 (LP). Strict-name-bar relaxes with support: median name-arm
  yes .16 -> .41. nan 0.
- **Anchors validate the instrument:** REACHES-ANCHOR kappa stable v1->LP (x llama70b
  .26->.27, x qwen72 .45->.47, x qwen32 .49->.46) — no alignment artifact this time.
- **Tacit kappa does NOT rise on long support — it mildly FALLS cross-family:** x qwen72
  .42->.31, x qwen32 .40->.28, x llama70b .20->.17 (dialect kind falls too: .44->.26 x q32).
  With anchors flat, the drop is construct-specific: longer support gives contested
  constructs MORE surface to diverge on while verifiable ones hold. Tacitness signature
  survives a 405B-class rung; support-fix rescues applicability, not agreement.
- Oddity: hermes-4 agrees with the QWEN family (~.3-.45) far more than with llama70b
  (~.2) despite Llama-3.1 lineage — instruction-tuning dialect trumps base-family?
- **CAVEAT: hermes-4 battery bal_acc .783 / z=1.285 — BELOW llama8b (1.562)** and below
  hermes-3's .8625. No NaN storm (0.0), deterministic even/odd .783/.783 — not the serving
  artifact signature; probing whether OSL_REASONING_OFF is the cause via battery-only run
  under short `hermes4-405b-rsn` (reasoning on). Until resolved, hermes-4 kappa rows are
  fine (kappa is z-free) but do NOT place hermes-4 on the capability axis.
FAM lane: qwen OR-authoring reached 72/72 (repair pass: 8->67->72); all 4 fam freezes
built FULL-slate (41/11/10/10 bases; shas fc05dd54e85c/4ef259673bb8/682f79825c8e/
a84d8fed27a8); floor bug fixed (12 -> 8; slates only hold 10-41 bases/task).
llama1b-or fam lane live; early rows: doss_fam_llama yes .99-1.00 (all-YES collapse on
llama-AUTHORED dossiers at 1B) vs expl_fam_llama .32 — first hint that same-family
authorship does NOT rescue dossier-length arms at 1B; verdict = full 164-row const grid.

### 2026-07-10 ~01:20 — HERMES-4 BATTERY CAVEAT RESOLVED + FAM 2x2 EARLY GRID
- Hermes-4 battery WITH reasoning: bal_acc .796 / z=1.360 (vs .783/1.285 reasoning-off) —
  OSL_REASONING_OFF is NOT the cause; hermes-4 genuinely sits ~.78-.80 on the code-truth
  battery, below hermes-3's .8625. Real low rung, not artifact; keep hermes-4 kappa rows,
  place it at z~1.29-1.36 with the "newer != higher-battery" note.
- llama1b-or fam grid @60/164 rows (15/arm, OR provider, humor):
  doss_fam_llama 67% const-YES | doss_fam_qwen 80% | expl_fam_llama 0% | expl_fam_qwen 7%.
  EARLY READ: **length dominates author-family** — dossier arms collapse 1B regardless of
  author; explanations stay healthy regardless of author. Same-family authorship gives at
  most marginal relief (both deltas in the llama-author direction but n=15/arm). If it
  holds at 164 + on the local ladder (incl. 3B/8B partial-collapse regime), the answer to
  "can same-family decompression fix small-model arms" is NO for 1B dossiers — the fix is
  arm LENGTH (explanation-depth), not dialect matching.

### 2026-07-10 ~01:50 — JOURNALISM (news_homepages) FIRST FULL z×a READ (9 local execs, 21-slate)
zxa_fit news_homepages @ OSL_PROBES_FILE=news_probes.jsonl (saved zxa_fit_v1.json). Headlines:
- **Instrument REPLICATES on a new domain**: dossier−mismatched positive on all capable execs
  (ANCHOR +.28..+.43, TACIT-RECRUIT +.17..+.31, PLANTED up to +.40); mismatched sits BELOW name
  (anchors −.13..−.36) = wrong specifics actively mislead. Planted sanity passes (definition−name
  +.07..+.24; dossier≈definition for 1-sentence rules, as designed).
- **Exchange rate FAMILY-RELATIVE again, now on journalism**: pooled TACIT-RECRUIT β qwen=+0.548
  (placebo −.109) vs llama=+0.127 (placebo +.018); ANCHOR β≈0/neg (no room above name, correct).
  First clean positive articulation-gain reading on the recruited-tacit class: a dossier buys a
  qwen executor ~half a battery-z unit.
- **Small models FLOOR on news** (llama1b/3b, qwen3b ≈ .500 everywhere; qwen25-3b const):
  long-input task (med 2030ch) reconfirms input-side collapse; ladder effectively 8B+.
  18/32 metric×family rows collapse-sensitive (vs 27% humor) — most crossings right-censored.
- **TASK CONTRAST vs humor (the interesting one): articulation CREATES cross-model agreement on
  news** — frontier κ DIALECT .053(name) → .583(dossier) (.689 on mismatched = unanimous
  rejection of wrong specifics); TACIT-RECRUIT .253 → .356. On humor, tacit κ stayed FLAT with
  support/arms. News constructs behave definitionally (given the dossier, models converge);
  humor's contested constructs don't. κ-vs-arm-depth is a new task-level signature to add to the
  isomorphism comparison — same instrument, opposite convergence behavior.
