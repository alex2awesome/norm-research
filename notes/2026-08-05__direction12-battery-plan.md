# Direction 1+2 battery — PLAN OF RECORD + live tracker
(created 2026-08-05 late; user: "write a detailed plan for Direction #1 and #2, keep track of
exactly where we are." Framing per the standing decision: everything below probes the
BLOCKERS/REMEDIES to explicit knowledge — Direction 1 = internal structure of the better-string
remedy, Direction 2 = characterization of the better-listener remedy. H-labels appear only as
local failure explanations. Task IDs = harness task list #16–#24; update BOTH on every change.)

## Status board (update in place)

| id | experiment | status | box/GPU | next action |
|---|---|---|---|---|
| #16 | 1a unit-vs-exemplar prefix | **RUNNING** since 08-05 ~22:30Z | sk3 GPU6 | harvest on "MODES1A COMPLETE" (monitor armed) |
| #17 | 2c remedy-verdict table | **v1 BUILT** (1,270 rows; β columns pending 1c) | CPU local | join per-metric β/κ when 1c lands |
| #18 | 1d content-count law | labels IN HAND (494 steps, local); needs per-step score join | local | trace extractor provenance → join round scores |
| #19 | 1c z×a exemplar arms | **RUNNING** (freeze built: 124 entries; ladder on GPU5+7 since 08-06 00:06Z) | sk3 GPU5+7 | harvest on 2× "LANE-DONE" (monitor armed) |
| #20 | 1b feedback-blind GEPA | **RUNNING** (hotpot @16,700, since 08-05 ~23:55Z) | sk3 GPU3 | v1 = feedback-blind (traces still visible); trace-blind = v2 decision after |
| #21 | 2a type-conditioned unit values | **BLOCKED**: labels file lost on all boxes | — | regenerate via HB172 recipe (Codex wave) |
| #22 | 2b de-censor RISING tail | queued (big GPU); TARGET LIST READY (260 oids) | sk3 ×4 GPUs | after 1a/1c settle; needs 405B-FP8 TP=4 |
| #23 | 1e calibration frontier | queued (after 1c) | sk3 | build constructs after exemplar formatting exists |
| #24 | 2d LoRA in-weights probe | DEFERRED | — | needs user re-confirm before start |

**1c first structural finding (before any GPU result)**: crowd-labelable exemplars exist for
9/10 REACHES-anchors and 5/5 planted but 0/16 TACIT-CANDIDATE and 0/10 DIALECT-SUSPECT — for
contested criteria no consensus labeler can certify a demonstration. Covered instead by the
`exemplars_authored` arm (dossier CONTRAST-EXEMPLARS section, all 41 bases) with its own
mismatched placebo; corpus-exemplar arms run on the 14 decisive bases. Fit-time rule: mask
zxa.exemplar_idx items (verbatim in rubric).

Also in flight (campaign, not battery): hvcert10110 hover rescore (sk1 GPU0 srv / sk2 client);
hotpot GEPA seeds s1/s2 (sk1 GPU1/2); ifbench→aime chain (sk1 GPU3); hover 5-pass prefix
(sk2, queued behind l1ly). All monitored.

## Detailed designs

### 1a — unit-prefix vs exemplar-prefix (RUNNING)
One session, sk3 GPU6, Qwen3-8B, hotpot: seed(5 passes) + unit-prefix k∈{1..20,24,28,32,36,40}
(frozen pool delta_8b desc, md5 27966bce identical on all boxes) + exemplar-prefix
k∈{1..12,14,16,20} (train-set Q→A pairs, fixed train order, appended to final_answer.predict).
5 passes/point; added_chars stored per point. ~61.5k scored items ≈ 1 day.
**Declared readouts (stated before results):** R1 exemplar rollover — is the exemplar curve's
slope ≤0 by k≈10 while units still rise (ICL prediction)? R2 token-matched comparison — at
equal added chars, which channel is higher? R3 asymptote gap at each channel's own best k.
Analysis: mean±SE curves, one figure, outputs/analyses/modes1a_20260806/. Never mix with the
07-28 sweep in one panel (different session). Extension decision after v1: hover exemplars
(claim→label) and a mixed units+exemplars arm.

### 1b — example-blind GEPA
Mechanism: dspy.GEPA's reflection consumes the metric's feedback strings (failing traces).
Patch: `--blind-feedback` → gepa_metric returns Prediction(score, feedback="") — scores-only
reflection; 5-line change at paperexact_arms.py:270-274. Run: hotpot official @16,700,
gepa-seed 0, tag truematch16700_blind, arm_lane_sk1.sh on first free sk1 GPU.
**Readout:** accept-count + val trajectory + 5-pass final test vs the three sighted seeds
(seed0 .580 raw; s1/s2 running now = the natural comparison set, same box for s1/s2).
Interpretation: fraction of GEPA's improvement that is example-mediated.

### 1c — mode × metric grid (z×a + exemplar arms)
z×a recap: same metric × same ladder × articulation arms; β = horizontal shift in z-units.
NEW arms: `exemplars` (~4 high-agreement YES + 4 NO probe items for that metric),
`def_exemplars`, `exemplars_mismatched` (another metric's exemplars, content placebo).
Exemplar labels WITHOUT humans: frozen LOCAL_MID frontier-consensus verdicts from the mbar
panels (reconstruction-only rule preserved). Steps: (i) per-item scores + probe texts from
mbar2 npz on sk3; (ii) selection script (agreement threshold, length cap, balance);
(iii) build_zxa_freeze_exemplars.py — arms are freeze entries with different rubric strings
(zero runner changes, the standing zxa trick); (iv) ladder run on sk3 allowed GPUs.
**Gates (frozen before results):** exemplar arm counts only where it beats
exemplars_mismatched (content gate); definition_padded remains the length control; planted
metrics must transmit under exemplars (sanity).
**Readout:** β per arm per metric, crossed with regime labels. Key cells: do BOUNDED metrics
respond to exemplars where definitions fail; do REACHES respond to nothing.

### 1d — archival content-count law
Join 487 step labels (evolution_change_type_labels_20260728.json) to per-step val gains from
the source runs' proposals.jsonl (consolidated local copies). Model: step gain ~ content
family + step index, clustered by run. Secondary: final-prompt score vs content counts across
runs. Pure description of the optimizer's own trajectories — closes the decompression-confound
loop from the optimization side.

### 1e — synthetic-construct calibration frontier (side)
Constructs = compositions of measurable text properties (k clauses, thresholds, exceptions),
truth by code; arms nonce-name / definition / def+exemplars / exemplars-only × ladder.
Readout: AUC vs programmatic truth = mode channel capacity with no blockers (nothing
internalized, message complete, search eliminated). Used ONLY as a ruler for 1c metrics at
matched complexity. Build after 1c; drop if flat.

### 2a — type-conditioned unit-value curves
Per-unit marginals by executor exist (hover 4B/8B/32B stair runs; aime ladder); join with the
HB172 unit-type labels (runs/omega_unit_labels_checkability.json — git-ignored, locate or
regenerate per note recipe). Output: marginal-value-vs-scale per content type → which content
kinds have windows (pay only in a capability band) vs flat.

### 2b — de-censor the RISING tail
List = deep-censored RISING (z90 > z_frontier + 1 under the logistic refit; ~235). One
stronger rung (local Llama-405B-FP8, TP=4, 4 simultaneously free allowed sk3 GPUs) through
the mbar panel for JUST those metrics. Readout: saturate-below-1 (bounded-late) vs
still-rising. Schedule after 1a/1c GPU pressure clears.

### 2c — per-metric remedy-failure verdict table (paper object)
One row per metric (1,270): regime verdict; refit L/k/z0/R²; censoring depth z90−z_frontier;
9-type label + externality axis; bounded-audit class (humor); tacit-candidate flag; zxa β per
arm + frontier κ where the 72-slate overlaps; missing-mass note where available. Columns read
as remedies: better-string (β, mode responses) / better-listener (regime, censoring). v1 =
CPU build from local artifacts tonight; grows columns as 1a/1c land.

### 2d — LoRA in-weights probe (LAST; user re-confirm before start)

## Decision points to bring to the user (not before data)
- After 1a: hover extension + mixed-arm? After 1c gates: which metrics go into 1e's matched-
  complexity comparison set. After 2b: regime census refresh wording.

## 1c FIRST READ (2026-08-06 00:55Z — ladder complete in 35 min, logprob readouts, 2,466 rows)

Balanced agreement with the frozen frontier-dossier reference, big-tier locals
(qwen25-14b/32b, llama70b, qwen25-72b), all 41 bases, reference recomputed per item from 4
frontier executors' dossier verdicts (unanimity + tie rates stored in refstats):

| class | name | defn | dossier* | corpus-ex | def+ex | corpus-ex MM | authored-ex | authored MM |
|---|---|---|---|---|---|---|---|---|
| PLANTED | .582 | **.889** | .885 | .606 | .885 | .548 | .653 | .527 |
| REACHES-ANCHOR | .722 | .797 | .835 | .772 | .803 | **.784** | .774 | .762 |
| DIALECT-SUSPECT | .782 | .818 | .911 | — | — | — | .813 | .771 |
| TACIT-CANDIDATE | .700 | .738 | .791 | — | — | — | .737 | .690 |

(*dossier is favored by construction — the reference IS dossier-anchored; compare among
non-dossier arms and vs placebos.)

**Findings (first-read, big tier):**
1. **The demonstration channel is weak everywhere as a standalone.** Exemplars never beat a
   definition in any class. Sharpest on PLANTED (mechanical rules with code truth): 8 examples
   .606 vs the stated rule .889 — even 70B-class locals barely induce "wordcount>median" from
   examples. Direct evidence for Noah's "models are not good enough at learning from tacit
   examples" — and it extends to fully-mechanical content.
2. **Corpus exemplars FAIL their content gate on REACHES** (.772 true vs .784 mismatched —
   the placebo does as well): the small lift over name is generic anchoring, not content.
3. **Authored contrast exemplars carry real content exactly where criteria are contested**:
   TACIT-CANDIDATE +.047 over placebo (.737/.690), DIALECT +.042 — the largest content-specific
   exemplar effects — and reach parity with definitions there. But:
4. **The dossier's edge is NOT reducible to its exemplar section** (authored-ex ≪ dossier in
   every class): whatever the 400-word dossier transmits, most of it is not the examples.
5. Placebo ordering sane everywhere (MM ≤ true arm; PLANTED placebos at floor) — instrument
   behaves.

Caveats attached: dossier-anchored reference (see *), 8 exemplars fixed (k-scaling is 1a's
job), big-tier aggregate only (per-executor β fits + small tier pending), exemplar-item
masking applied uniformly per base. Artifacts: sk3 outputs/osl_multi/zxaex_firstread.json
(+ local snapshot outputs/osl_multi_local/zxaex_firstread_20260806.json), panels
mbar_zxaex_humor_<exec>.npz ×9, freeze_zxa_ex_humor_v1.json.

## 2a FIRST PASS + 1c-v2 replication arms + unblocks (2026-08-06 ~01:30Z)

**2a (labels found LOCALLY — datasets/prompt-optimality-test/runs/, task unblocked).** Joins
perfect (0 unmatched). Hover Qwen3 ladder: window replicates and extends — 1.7B mild+ / 4B
PEAK (craft +.042 89%pos > mech +.029 81%) / 8B ≈0 / 14B +.035 / 32B negative. AIME: 1.7B
floor (units hurt) / 8B strongly NEGATIVE (−.059) / 32B positive (+.053, 88%pos) — window
sits high on hard tasks. TWO CAVEATS OWED: (i) the 8B trough is partly SELECTION-REGRESSION,
not window — pools were screened on 8B behavior (delta_8b/won_8b), so 8B re-measures inherit
winner's-curse regression while other rungs measure fresh; HB172's "8B≈0" carries the same
caveat. (ii) aime-8B sign here (strongly negative) conflicts with HB172's "8B 75% pos" —
different measurement pass; reconcile against HB172's source before quoting either.
Artifact: outputs/analyses/unit_value_by_type_20260805/.

**1c-v2 ICL-replication arms RUNNING** (sk3 GPU5+7, freeze_zxa_ex2: 14 bases × 2 arms):
`exemplars_fmt` (canonical interleaved Text→Yes/No demos, same items) tests the format
hypothesis; `exemplars_shuf` (same items, ~half labels wrong — the Min et al. 2022 control)
tests content-vs-anchoring. Emergence check ALREADY DONE from existing rows: content-specific
exemplar effect (true−mismatched) grows with scale — PLANTED ~0 below 14B → +.07/+.11 @70B/72B
corpus, +.17/+.15 authored; TACIT ~0 → +.05-.08; REACHES flat everywhere = Wei-et-al-style
emergent in-context label learning, in-house.

**2b scoped for OpenRouter**: 260 metrics × 300 probes ≈ 78k judgments, hard YES/NO readout
(no logprobs on OR), est. $25–80 at big-open-model pricing — no 4-GPU local 405B needed;
dialect-discount rules apply (compare orderings not levels). Awaiting user OK on ~$50.
**1e, 2d, 1d: user green-lit ("do it").** 1d next CPU block; 1e after ex2 harvest; 2d needs
training-infra plan.

## 1c-v2 REPLICATION HARVEST + per-base Noah-hypothesis test (2026-08-06 ~02:10Z; 8/9 execs,
## final re-harvest after qwen25-72b lands)

**Format leg (exemplars_fmt, canonical interleaved demos, same items):** PLANTED .671 vs
list-format .606 (+.065 — format real but small) — still ≈.22 below the stated rule (.889).
REACHES .781 ≈ list .772 ≈ placebo .784.
**Label leg (exemplars_shuf, ~half labels wrong — Min et al. control):** PLANTED .589 vs fmt
.671 (−.082: labels DO carry content at big tier, consistent with the emergence curve);
REACHES .759 vs .781 (−.022: labels nearly irrelevant — anchoring-dominated).
**Verdict (firm):** we replicate BOTH halves of the ICL literature in one instrument —
Min-style content-insensitivity below the emergence point and on already-internalized
criteria; Wei-style emergent label-learning at 70B-class; and the stated>shown gap for
rule-induction survives the format and label controls. No contradiction with prior work.

**Per-base channel comparison (big tier, 29 bases with all arms):**
- corr(definition gain, exemplar content-specific gain) = +.45 overall but −.03 EXCLUDING
  PLANTED → for real bank metrics the two channels are UNCORRELATED (neither redundant nor
  systematically complementary).
- Exemplars>definitions bases are gestalt/pattern constructs: "Framing via titles/headlines"
  (+.100), "High-concept hook uniqueness" (+.092), "Accumulation/format-subversion" (+.052),
  "Self-deprecation" (+.036, def_gain only +.008), and "Shared knowledge and reference
  accessibility" (+.034 with def_gain NEGATIVE −.012 — the cleanest show-beats-tell case).
- Definitions>>exemplars: all 5 planted (statable rules) + "Genre/style classification"
  (taxonomic content).
**Noah's hypothesis ("ICL helps where words fail"): WEAK FORM SUPPORTED** — at the item level
a recognizable subset of contested, gestalt-type criteria transmit better by showing than
telling, including one where telling outright fails; **strong form NOT supported** — channels
are uncorrelated rather than anti-correlated, and many tacit bases respond to neither.

## ALL-DOMAINS EXPANSION LAUNCHED (2026-08-06 ~02:59Z, user directive: past humor into all domains)

freeze_zxa_ex_{creative_writing,peer_review,math,news_homepages}_v1.json built (46/29/47/66
entries; long-text tasks: 3+3 exemplars truncated 500ch). Corpus-base decisiveness by task:
CW 8/11, math 9/10, news 8/21, peer 3/10 — the contested-crowd pattern replicates across
domains (peer/news weakest). Ladders on sk3 GPU5 (small execs) + GPU7 (big), ~36 panels.
Launch hiccups fixed en route: meta.task must be HYPHENATED preset name; meta needs
n_probes (merged runner fields from v1 freezes); self-match kill trap avoided via bash -s.
PLANTED replication comes free: 5 planted bases × each new task = cross-domain planted table.
Queued next: k-scaling exemplar arms (k∈{2,8,16,32}, humor) after this ladder; CoT/generative
arm = design note pending; user ruled NO OpenRouter for 2b (wait for local GPUs).

## Direction 3 added (user, 2026-08-06): family-robustness + reasoning-as-articulation

**3a (#25) family-robust plateau census.** Finding that motivates it: 63/185 plateau-adjacent
metrics get DIFFERENT family verdicts, and the split is structured — Qwen saturates early on
PROSE-ECONOMY constructs (economy/concision/endings cluster, 7/8 top examples) while Llama
saturates early on PERFORMANCE/PERSONA constructs (delivery, comic voice, conjecture-
refutation, actionable suggestions). Same fault line the z×a exchange-rate work found
independently (compressed-quotable: Llama metric-B / Qwen metric-A) — two instruments, one
family-culture split. Plan: Qwen3 ladder (1.7/4/8/14/32B, registry extended, weights
downloading to sk3) as the third staircase, no_think mode → "plateaus-in-ALL-families" set =
the executor-robust plateau candidates. Launches on GPUs 5/7 when EXALL finishes.
**Hive-mind caveat (queued appendix discussion, user raised):** model families share
pretraining corpora and practice, so cross-family agreement is NOT independence — an
everywhere-plateau bounds EXECUTOR-RELATIVE tacitness (the paper's construct) and must never
be quoted as human-tacitness. Mitigations: family-conditional verdicts always reported; a
non-US-lab family point (GLM hard rungs) in the triangulation; planted anchors certify the
instrument independently of shared culture.

**3b (#26) self-articulation arm.** User: reasoning models "are essentially doing extra
articulation." Qwen3 hybrid thinking = same weights ± reasoning → the cleanest possible test:
think−no_think gain per metric class = the value of SELF-articulation, comparable to the
external-articulation arms in the same grid. Requires the generative readout path (also
serves the CoT-amplified exemplar rebuttal leg). Sharpest prediction to test: reasoning gains
should concentrate where external articulation fails (contested classes) IF self-articulation
accesses content that external strings cannot; if instead gains mirror the definition arm,
reasoning is just internal restatement.

### 3b expanded to MULTI-FAMILY reasoning slate (user, 2026-08-06) + build scope

Reasoning gain measured three ways, family-crossed (motivated by the dialect findings):
1. **Same-weights toggle**: Qwen3 1.7-32B think vs no_think.
2. **Matched-base pairs** (reasoning-trained vs instruct of the SAME base already in panels):
   R1-Distill-Qwen-14B/32B ↔ Qwen2.5-14B/32B-Instruct; R1-Distill-Llama-8B ↔
   Llama-3.1-8B-Instruct; Phi-4-reasoning ↔ phi-4.
3. **Non-US-lab point**: GLM-Z1-32B-0414 (weights TBD).
Weights downloading to sk3 (cache on the 84T array, 10T free — the 438G scare was the root
mount, not the cache). Build: vllm_backend.score_binary_gen (chat template with
enable_thinking=True / R1-native, generate, parse post-</think> YES/NO, hard 0/1 +
parse-fail rate — backend already templates with enable_thinking=False, so this is a
surgical addition) + osl_sweep --readout flag. COST NOTE: thinking generation is 10-100×
logprob reads → think panels run the FOCUSED slate (bases × definition + planted), 20-probe
smoke test gates any ladder. Prereg prediction recorded in task #26.

### GLM REVIVED (2026-08-06): Lite plan active — 3b frontier leg UNBLOCKED

User bought z.ai Lite (10k credits/wk ≈ 87M glm-5.2 tok/wk). Verified: `.z-ai-api-key.txt`
LIVE (laptop+sk3, identical) + `.z-ai-api-key-spangher.txt` LIVE (distinct string);
alexander-spangher still dead. **Thinking mode works on the subscription endpoint**
(`thinking:{type:enabled,budget_tokens:N}`, returns thinking blocks; smoke 248 tok @ budget
256) and glm-4.7 alias resolves → 3b gains the frontier leg: GLM-4.7/5.2 think toggle AND a
thinking-BUDGET dial (third dose-response instrument, zero marginal cost). Plan: 100-call
trace-length smoke on real metric prompts first, then the humor focused slate think/no-think
(~21M tok ≈ 25% of one week). Weekly use-it-or-lose-it batching.

## ALL-DOMAINS HARVEST (2026-08-06 ~06:20Z; 36/36 panels; big-tier balanced agreement)

**PLANTED replication (user-required): the stated≫shown result holds in ALL FIVE domains.**
exemplars-alone vs definition: humor .606/.889 · CW .550/.818 · peer .619/.774 · math
.577/.707 · news .500/.761. Examples of a code-checkable rule never transmit better than ~name
level; the stated rule adds +.15–.28 everywhere. Placebos ≤ true arms throughout.
Notable secondary: def+exemplars ≥ definition in 4/5 domains (CW .911 vs .818 — examples ADD
on top of statements there); authored-exemplar content-specificity (auth−auth_mm) replicates:
CW +.057 reaches / +.072 planted, math +.130 planted, peer +.136 planted.
**NEWS = instrument-suspect, not a finding**: all exemplar arms ≈ .50 (chance) while name .71
and dossier .89 work — the 500-char truncation of ~2,000-char homepage probes likely destroys
exemplar recognizability, and the news reference has only 2 frontier voters. Flag, don't quote.
Peer REACHES definition .515 < name .719 is n=1-base × 4 execs — noisy, park.
Artifacts: sk3 outputs/osl_multi/zxaex_read_{task}.json.

## k-SCALING HARVEST (2026-08-06 ~07:00Z): the exemplar channel is FLAT in k

Dose-response k=2/8/16/32 (humor, 14 decisive bases, same reference/masking; big tier):
- PLANTED: .576 → .593 → .602 → .600 — a +.03 crawl that stops by k16; definition = .877.
  QUADRUPLING the examples (8→32) adds nothing; the stated rule's lead is untouchable at any
  dose we can fit in context.
- REACHES: .748 → .760 → .764 → .764 — flat, and ≤ the mismatched placebo (.771) at EVERY k:
  the anchoring interpretation is dose-independent.
- Small tier: flat at ~chance on planted; mildly DECLINING in k on reaches (.590→.561) —
  below the capability floor, more examples add confusion, not signal.
**Verdict: the exemplar channel saturates by k≈2–8 in the metric frame — "more examples" is
not a transmission path for rules or criteria.** ICL rollover prediction confirmed here;
the benchmark-frame version (1a, k to 20 with token matching) still running. Artifacts:
sk3 outputs/osl_multi/zxaex3_kcurve.json + mbar_zxaex3_humor_*.npz.

## Qwen3 staircase COMPLETE + stale-battery catch (2026-08-06 ~10:35Z)

All 5 rungs × battery + humor285 + 9 task panels done (45 panels, rc=0 throughout).
Fresh battery ladder: 1.7B z=0.98 → 4B 1.19 → 8B 1.98 → 14B 2.32 — clean dense monotone
staircase; 8B matches Qwen2.5-14B (generational gain ~one size class).
**CATCH: the apparent 32B inversion (z=1.55) was a STALE FILE** — outputs/osl/qwen3-32b.json
was the July OpenRouter HARD-readout probe (api_serving/hard_readout keys, different z scale,
never poolable); the lane's skip-if-exists guard suppressed tonight's battery. Archived to
qwen3-32b_APIHARD_stale_20260706.json.bak; fresh battery re-running on GPU5 (panels are fine —
they ran locally tonight). LESSON for the registry: skip-if-exists guards must check the file's
READOUT ERA, not just existence; same trap class as prenanfix/.bak leakage.
2b hinges on the fresh 32B z clearing qwen25-32b's 2.546; if not, the de-censoring listener
falls to gpt-oss-120b (weights down). 3a family-verdict join runs after the battery lands.

### Fresh qwen3-32b battery + reasoning-slate triage (2026-08-06 ~10:50Z)

qwen3-32b TRUE z = **2.424** (auc .919) — ladder complete & monotone (0.98/1.19/1.98/2.32/2.42)
but BELOW qwen25-32b (2.546) → not a frontier point; 2b listener falls to the triage winners.
**Battery triage chain launched (GPU5, 11 models)**: gpt-oss-120b first (2b candidate), then
r1-distills / phi4-reasoning / magistral / seed-oss / glm-z1 / gpt-oss-20b. Dual purpose:
(i) pick the 2b listener (need z>2.546); (ii) map instrument compatibility — reasoning models
with forced-<think> templates may break the no-think logprob readout (degenerate battery = a
FINDING that routes them to 3b's generative readout, not a bug). Era-guard added to
skip-if-exists (requires per_family present). Registry +12 entries (gptoss/r1qwen/r1llama/
phi/mistral/seedoss/glmz families; MoE = z-points only per axis discipline).
NEXT CPU: 3a family-verdict join (qwen3 5-rung y vs frozen crowd × llama/qwen25 verdicts →
plateaus-everywhere set).

### Triage post-mortem (2026-08-06 ~11:00Z): premature — downloads still in flight

10/11 legs crashed instantly with NoneType/EngineCore errors = WEIGHTS NOT YET DOWNLOADED
(9 downloads active; R1-32B at 9.6/65GB — NAT64 is slow). Auto-relaunch armed: triage refires
when download count hits 0 (lane resume-safe; the degenerate gpt-oss-20b file will retry by
the era-guard). REAL finding from the run: **gpt-oss is incompatible with the logprob battery
in this env** — 120b fails snapshot-path resolution; 20b loads but nan_rate=1.00 (harmony
reasoning-channel format defeats first-token P(YES)) → gpt-oss routes to 3b's generative
readout for ANY use, and likely drops as 2b listener. 2b listener fallback order:
(1) any triage survivor with z>2.546; (2) GLM-5.2 API hard-readout panels next weekly window
(~65M tokens); (3) disclose 2b as bounded-by-current-frontier.

## 1c-v3 SPEC (user design, 2026-08-06): FUNCTIONAL exemplar validity via flip-influence

User's construction, adopted: an example-label assignment is CORRECT iff it improves
transmission — flip m(x_i), re-run decoder, accept iff HELD-OUT reconstruction MI rises.
Replaces the consensus labeler (which structurally fails on contested metrics — 0/16 TACIT
crowd-certifiable, the biased-slice problem the user identified). Design: candidates =
crowd-AMBIVALENT items only (~20-40/metric — the ones consensus discards); batched flips with
greedy refinement (~15-30 decoder+rescore cycles/metric); train/holdout split for
selection-vs-report (Q1 machinery); confirmation passes on accepted flips (winner's-curse);
scope disclosure: correctness is encoder-relative (transmission of m_ω's concept, T-lower-bound
framing). Pilot: the 16 humor TACIT-CANDIDATEs, next free GPU. Readout: do functionally-
selected exemplars beat definitions where definitions fail — the un-biased version of the
show-vs-tell hunt.
ALSO (user base-rate demand): labeled-domain replications = (a) 1a gold Q→A exemplars
(running); (b) planted arms re-selected by CODE TRUTH (launching now — kills the crowd-label
confound on the headline stated≫shown claim); (c) silver-labeled humor/CW exemplars queued.
