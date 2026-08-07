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

## CODE-TRUTH PLANTED HARVEST (2026-08-06 ~18:55Z): the crowd-label confound is DEAD for the
## planted claim

Provably-correct exemplars (selected by computed rule truth) transmit NO better than
crowd-labeled ones — big tier, vs code truth: name .573 / definition .842 / crowd-ex .601 /
**code-truth-ex .595** / ct-placebo .558. Same picture vs the frontier reference
(.607 vs .598). Identical result pattern in the small tier (~.51 everything except
definition .68-.71). ⇒ The stated≫shown gap on rules is a property of the CHANNEL, not of
label quality — even perfect examples of "contains a digit" barely beat the bare name at
70B-class, while the stated rule scores .84-.89 against actual truth. The unlabeled-examples
concern does NOT touch the planted headline. (It remains live for CONTESTED metrics — exactly
what the flip-functional lanes are measuring now.)

## 1c-v3 PILOT RESULTS (2026-08-06): functional selection WORKS at 70B — and the crowd was
## suppressing it

Family-crossed pilot (5 contested humor bases scoreable; 2 broke on sparse refs — v2 fixes):

**qwen25-14b selection: functional never beats definition** (sets stay near-seed; train gains
don't transfer). Consistent with the emergence curve — 14B can't read example content.

**llama70b selection: functional exemplars WIN on the predicted cells:**
- "Shared knowledge & reference accessibility" (frontier obj): **functional .908 > name .864 >
  definition .773** — a +.135 holdout show-beats-tell win, and **2/3 selected labels CONTRADICT
  the crowd majority** — the flips are doing the work. This is the same base that was
  definition-NEGATIVE in the earlier scan; the user's functional design recovered the signal
  the consensus labeler was destroying.
- "Balancing humor with pathos" (encoder obj): functional .907 > definition .878.
- Definition-friendly metrics stay definition-friendly (Concision: def .818 ≫ functional .665)
  — functional selection does not fake wins where the statement channel is genuinely better.

**Answer to "how much is crowd labeling affecting things": on contested metrics, materially —
it was suppressing real demonstration-channel wins that only appear when (a) labels are chosen
by transmission (user's flip design) and (b) the reader is at content-emergence scale (70B).**
Caveats: n=5 bases, single split, no confirmation passes — deltas ~2-3σ, promising not
certified. V2 (queued): full-bank ambivalence panel → 16/16 coverage, confirmation passes on
accepted flips, multi-split holdout, then the ladder rerun of winning functional sets.
Artifacts: outputs/osl_multi_local/flip_functional_{qwen25-14b,llama70b}.json.

## 1c-v3 SCALE-UP LAUNCHED (2026-08-06 ~21:00Z, user directive: more metrics, more flips,
## airtight train/test)

flip_functional_v2: ~73 non-planted slate bases × 5 domains; candidates 24 items × both
labels (48 proposals), 6×8 greedy rounds, set cap 12. **Leakage protocol upgraded to
three-way stable-hash split**: train-A (selection, ΔA≥.01) / train-B (confirmation gate —
accepted flip must not degrade B) / holdout H (touched ONCE, at the end). Plus a
**selection-null leak detector** every 3rd base (search vs permuted labels; holdout scored vs
TRUE ref must sit ≈ name — if null shows gains, the harness leaks). Long-text exemplars
truncated 500ch. Selection executors BOTH at content-emergence scale: llama70b (GPU7, live)
+ qwen25-72b (GPU5, waits behind triage). Outputs flip_functional_v2_<exec>.json.

## DISCUSSION SEED (2026-08-06, from user question "is the example itself explicit knowledge?")

Queued prose direction (freeze respected): an example is an explicit ARTIFACT whose
knowledge-content is conditionally tacit — transmission-by-example = string transfer +
receiver-side induction; the tacit component is RELOCATED from sender to receiver, not
eliminated. Channels differ in division of epistemic labor: definitions carry intension
(sender did the abstraction), examples carry extension (receiver must induce). Empirical
mapping: emergence curve = receiver induction capacity switching on (dead string → live
evidence); k-flatness = bottleneck is induction capacity, never evidence quantity; code-truth
control = limit is the induction step, not label quality; functional selection = even CHOOSING
good teaching examples is tacit (discoverable only by transmission — Polanyi at the meta-level).
Inversion hypothesis: LLMs are the limiting case of Collins's interactional expert (encultured
at training time through text) → intensional channel lands on a prepared prior, in-context
induction emerges late; predicts the human/LLM show-vs-tell inversion weakens with scale
(matches Table-2 trend). Construct consequence: better-string remedy splits into intension-
carrying vs extension-carrying strings; the example-channel bound is irreducibly JOINT
(string × listener) — the case where executor-indexing of articulation bounds is essential.

### Discussion-seed REFINEMENT (user objection 2026-08-06: "a bare name also imposes tacit
### comprehension")

Correct — replace "demonstration relocates the tacit component" with the DIVISION-OF-LABOR
formulation: every channel is a contract over receiver contribution, differing in KIND:
name = pointer (~0 content; receiver supplies the whole concept from prior POSSESSION);
examples = extension samples (receiver supplies the INDUCTIVE leap); definition = compressed
intension (receiver supplies GROUNDING + application); dossier = scaffolded grounding.
"Relocation" is only correct RELATIVE TO the definition channel (abstraction step moves
across the wire); relative to the name, examples move work toward the sender.
Empirical teeth: the three receiver contributions DISSOCIATE — possession is family-relative
(dialects) and window-shaped; grounding gains exist at every scale; induction gains are
scale-emergent (~0 below 14B) and uncorrelated with grounding gains across metrics. One lump
of "tacit comprehension" would not produce three independent curves. Operational "tacit" is
reserved for: NO channel's delta closes the gap (possession absent + grounding insufficient +
induction insufficient) = the H-message cell. Name-only success = internalization, never
articulation evidence.

## 1a + 1b HARVESTS (2026-08-06 ~22:00Z) + three-capabilities figure

**1a COMPLETE (gold-labeled benchmark frame, same-session, 5 passes):** hotpot seed .409;
UNIT curve climbs to .627 @k=40 (still rising); EXEMPLAR curve (gold Q→A demos) climbs to
.564 @k=16 then ROLLS OVER (.541 @k=20) — the ICL-literature rollover, observed with
ground-truth examples. Token-matched at ~2,700-2,850 added chars: units .607 vs examples .541
(+.066). Examples DO help over seed (+.13-.15 — real few-shot gains on a real task) but the
articulated-unit channel dominates at every matched budget and keeps climbing. **The
stated>shown ordering survives the labeled domain — user's base-rate demand fully answered
(gold labels, no crowd anywhere).** Artifact: runs/prefix_modes1a_hotpot.json (sk3) + local.

**1b COMPLETE (feedback-blind GEPA @16,700): seed .411 → best .412 (+.001).** Sighted GEPA at
the same budget: .408 → .580. **Removing the metric's textual feedback kills GEPA's entire
improvement** — prompt optimization is feedback-compilation; scores alone teach nothing.
(v1 scope: feedback strings blanked; GEPA's reflective traces still nominally available —
yet zero progress, so the feedback text was the load-bearing example-mediated signal.)

**Three-capabilities figure BUILT** (user request): possession (name level) / grounding
(def−name) / induction (exemplar−placebo) vs battery-z, per family (solid Llama / dashed
Qwen2.5), bank-metrics + planted panels. Story visible at a glance: possession rises early
and family-locally; grounding positive at every scale (planted: +.20-.39); induction pinned
at ~0 until z≈1.5 then climbs in BOTH families. outputs/analyses/figs_20260806/
three_capabilities_emergence.png (+ .py).

## PAPER-2 STRUCTURE DECISIONS (user, 2026-08-06 evening)

1. §4 order: **Tacit-knowledge isomorphism FIRST, then "Statements versus demonstrations" as
   its own subsection** with the rung-ladder tie-in (name→definition→explanation→exemplars→
   dossier: 4.2 shows rungs rescue smaller models; 4.2b measures what each rung KIND carries).
2. **Receiver-capabilities + reasoning material goes INTO paper #2** (not held for #4).
3. User critique adopted: §4 lacks a unifying claim. Proposed thesis (recorded for the
   Discussion/section-lead): transmission = string content + receiver complement; the four
   subsections measure four complements (capability / internalization / induction /
   zero-complement mechanization); articulable = some complement available, tacit = none.
   Supporting devices: (a) three-capabilities dissociation figure (built); (b) acquisition
   ordering as §4's arc (grounding early → possession mid/dialect → induction late →
   self-articulation frontier → mechanization as receiver-elimination limit, explaining the
   seam cliff's location); (c) NEW ANALYSIS (first Phase-2 item): per-metric cross-instrument
   boundary agreement — do OSL-bounded/beyond-text, mode-grid all-channels-fail, and
   seam-noncompiling pick out the SAME items ("four instruments, one boundary"), with
   disagreement cells shown honestly. Join-feasibility check first (seam labels are
   sub-rule-level).

## §4.2b "STATEMENTS VERSUS DEMONSTRATIONS" — CONTENT PLAN (the examples program in the paper)

Tie-in sentence from §4.2: the decompression rungs (name→definition→explanation→exemplars→
dossier) rescue smaller executors; this subsection asks what each rung KIND actually carries.

Paragraph-level plan (result → artifact):
1. **Setup + the two frames** (metric grid template with rubric slot; benchmark prefix frame).
   One design figure or a compact prose description; the arms table (name/definition/dossier/
   corpus-ex/authored-ex + placebos) → appendix for full grids.
2. **Stated ≫ shown on statable content, 5/5 domains** — the planted table (exemplars ~.55-.62
   vs stated rule .71-.89 in every domain), WITH the code-truth control (label-quality confound
   dead) and the labeled-domain replication (1a gold demos). [all-domains harvest + exct +
   prefix_modes1a artifacts]
3. **The channel is dose-flat**: k-scaling (metric frame 2→32 flat; benchmark frame rolls over
   at k≈16 while units climb to k=40; token-matched +.066 units). ICL-literature positioning
   paragraph HERE (Noah's citation: scaling-in-articulations vs scaling-in-examples; Min/Wei
   replication-in-one-instrument). [ex3 kcurve + modes1a]
4. **Content-from-examples is emergent and placebo-gated**: mismatched/shuffled/format controls;
   emergence table (content-specific gain ~0 below 14B → +.07-.17 at 70B-class). [read2 +
   controls]
5. **Where demonstrations win**: the gestalt subset (per-base table: headline framing +.100,
   hooks +.092, shared-knowledge with NEGATIVE def gain); authored-contrast arms cover
   contested metrics (crowd-labelable-exemplars coverage table 14/15 uncontested vs 0/26
   contested = the structural finding). [per-base scan + coverage counts]
6. **Functional selection (the flip design)**: labels validated by transmission itself;
   A-select/B-confirm/H-report protocol + null detector; pilot: 70B finds wins the crowd
   suppressed (+.135 Shared-knowledge, 2/3 labels contradict crowd), 14B finds nothing
   (emergence-consistent); v2 scale-up results slot here when harvested. [flip v1/v2 jsons]
7. **The receiver-capabilities synthesis**: possession/grounding/induction dissociation —
   THE FIGURE (three_capabilities_emergence.png, re-rendered with flip-v2) — feeding the §4
   unifying thesis; forward-pointer to 3b (self-articulation) closing the capability set.
Appendix companions: full mode-grid tables per domain + placebo columns; flip protocol +
per-base results; exemplar-coverage-by-class table; news instrument-suspect flag.
Discussion deposits: division-of-labor, interactional-expert inversion + scale prediction,
1b feedback-compilation cross-reference.

## §4 ORDER FINAL (user, 2026-08-06) + EXAMPLES-AT-SCALE CAMPAIGN

**Order: 4.1 code seam → 4.2 tacit isomorphism → 4.2b statements-vs-demonstrations → 4.3 OSL.**
Arc = ascending receiver-dependence (receiver eliminated → grounding → induction → capability);
each section hands its residual to the next; survives-all-four = the tacit remainder. Mirrors
the intro's Collins criteria order (coded string → better string → better listener) — the
unification made structural.

**Examples campaign NOT finished — full-bank scale-up (extends Phase 1 by ~2-3 days):**
Leg 1: corpus-exemplar + mm arms over all 1,270 bank metrics against a DEFINITION-anchored
frontier reference (bank-wide panels exist; anchor change flagged — conservative for
show-vs-tell). ~1 day on 2 B200s when flip-v2 frees GPUs 5/7.
Leg 2: GLM-4.7 authored (+)/(−) contrast exemplars for ~300 metrics (all 65 BOUNDED +
contested classes + stratified RISING/REACHES sample); ~0.5-1M tokens; gated authoring;
starts next work block (API-only).
Leg 3: flip-functional at scale over the contested tail with the same A/B/H + null protocol,
after Leg 2 so all arms ride one ladder.
§4.2b tables fill from full-bank versions; slate results become the pilot/replication tier.

## THE TARGET EXHIBIT (user scheme, 2026-08-06): THE ARTICULABILITY LADDER

Per-metric assignment to highest achievable rung: COMPILES → STATEMENT-ARTICULABLE →
DEMONSTRATION-ARTICULABLE → CAPABILITY-BOUNDED (rising) → PLATEAUS-EVERYWHERE (remainder).
First population computed (big tier, cross-domain): compilables = attribution/lexical/numeric/
counts/format/normative (seam); statement top = genre classification +.158, incongruity
mechanics +.133/.100/.095, line-by-line justifiability +.087 (rule/taxonomy-shaped);
demonstration top = headline framing +.178 (+.100 vs own definition), format-subversion +.153,
satire tone +.115, hooks +.099 (gestalt-shaped); plateau = persona/enculturation/reception.
**AMENDMENT (data-forced): the middle is a FORK not a step** — def-gain ⊥ ex-gain (r≈−.03);
statement vs demonstration prongs differ in KIND (intension-friendly vs configural), rejoining
at the capability rung. Bonus cell to watch at scale: STATEMENT-NEGATIVE metrics (definition
HURTS: peer "reproducibility evidence" −.204, news framing criteria — weak refs, hedged).
Ladder-assignment table = the §4 money object; feeds: #27 (rung-1 seam join), #28 full-bank
campaign (middle fork), 2c verdicts (rungs 4-5), flip-v2 (demonstration prong), 3b
(self-articulation column). Every lane now serves ONE exhibit.

## PREREG — ABSTRACTNESS VALIDATION OF THE LADDER (frozen 2026-08-06, BEFORE the join is run)

Danger named (user): circularity — "abstract" defined by rung makes the ladder a tautology.
Design: the ladder's axis stays OPERATIONAL (receiver complement, measured); abstractness is
a validation correlate ONLY, tested with two instruments independent of all transmission
results: (i) published lexical concreteness norms (Brysbaert et al. 40k) over metric
name+definition; (ii) our blinded 9-type/externality labels (assigned 07-28 from names,
blind to regime, before the mode grid existed; κ=.75).
FROZEN PREDICTIONS (computed only after final rung assignments from the full-bank campaign):
  P1 mean concreteness decreases monotonically down the rungs;
  P2 beyond-text share increases monotonically down the rungs;
  P3 (the discriminating one): the two middle prongs (statement- vs demonstration-articulable)
     are INDISTINGUISHABLE in concreteness but DIFFER in type profile (rule/taxonomy vs
     gestalt/configural) — a kind-difference at matched abstractness that a single
     abstractness scalar cannot produce.
Landing zones: all hold → ladder tracks abstractness except where it forks by kind; a rung
breaks monotonicity → reportable structure (watch: compilable-but-abstract normative
standards); weak overall → "operational ladder does not reduce to lexical abstractness"
(publishable as such). No abstractness language in the core results either way.

## §4.3 OSL — IDEAL-INSIGHTS SCHEME (user request 2026-08-06; the target claims for the section)

Roles terminology settled (user): encoder = proposer of the articulation; receiver = the
executor applying the rubric to make measurements; decoder = the model reconstructing the
rubric from measurements. Encoder-ladder arm DECLINED (user): the bound requires the BEST
encoder; a ladder of weaker encoders tightens nothing. Frontier-authored strings stay.

The six target insights (each = one exhibit; status in parens):

I1 CAPABILITY-BOUNDED vs ARTICULATION-BOUNDED census — per metric: curve still rising at
   frontier (remedy = better receiver) vs plateau below ceiling (residual survives every
   receiver we can buy = the articulation bound binds). Headline split X%/Y%.
   (census done; 27%-of-rising deep-censored tail needs 2b listener or a disclosure caveat.)
I2 PLATEAUS ARE FAMILY-ROBUST — plateau-everywhere set replicated across Llama/Qwen2.5/Qwen3
   staircases; dialect set quantified & structured (Qwen prose-economy / Llama persona).
   Answers hive-mind + dialect critiques. (3a staircases done; family-verdict join pending.)
I3 ONE CURVE = A STACK OF PROCESS BOUNDS — decompose z-curve into internalization (name),
   grounding (definition), induction (examples), self-articulation (reasoning): different
   emergence points (possession early, grounding everywhere, induction top-of-ladder,
   reasoning TBD via 3b think-toggle/budget-dial). Which process binds depends on receiver
   scale. (three-capabilities figure done; 3b = the missing panel.)
I4 WHAT KIND OF THING RESISTS — the plateau set is not random: enriched beyond-text/gestalt
   (within-domain p=.049 humor/.004 peer), abstractness prereg P1-P3 as validation. Noah's
   "how much is there vs what kind of thing is it". (join pending final rungs.)
I5 FALLING LIMBS = DIVERGENCE-TOWARD-TRUTH — bigger receivers disagree with the crowd
   because they are righter (verified on planted/code-truth); OSL curves are not monotone-
   good and must not be read as degradation. (established on planted; general form = 2c join.)
I6 LAW-NESS / PREDICTIVITY — LOO backtest .036-.063: low-z behavior PREDICTS frontier
   transmission → cheap small-model probes forecast whether articulation will work for a new
   metric. This is what makes it a scaling LAW and not a census. (done; needs write-up.)
(+I7 = the 2c verdict table as the section's closer: every failure gets its remedy —
   better receiver / better statement / demonstrations / nothing-works — Noah's literal
   closing line "what can we do about the gap".)

2d LoRA in-weights probe — DESIGN ANSWER (user asked what it looks like): ground truth is
NOT flips (≤12 items, too few) and NOT task labels (reconstruction-only rule) — it is the
same frontier-dossier reference verdicts every exemplar arm uses (~150 train-split items per
metric). LoRA rank-8 on a small executor per metric = the WEIGHT channel toward the same
reference the PROMPT channel targets; compare asymptote profile vs the ICL k-curve (which
rolls over at k≈16). Controls: permuted-label LoRA + mismatched-metric adapter. Cost ~1-2
GPU-days for a 20-metric × 2-base pilot + new peft training loop. RECOMMENDATION: park as
future-work — the ICL rollover already carries the in-context half of Noah's point; revisit
only if reviewers ask for the in-weights contrast. (Awaiting user ruling.)

## FLIP-V2 HARVEST — llama70b selector COMPLETE 5/5 domains (2026-08-06 ~20:15 PT)

Paired holdout functional−definition, 20k bootstrap (outputs/osl_multi_local/
flip_functional_v2_llama70b.json; analysis inline, this session):
| task | frontier obj | encoder obj | NULL control |
|---|---|---|---|
| humor | +.036 [−.020,+.093] n=22 | +.026 [−.014,+.064] n=28 | **+.059 win 4/5 n=5** |
| creative_writing | −.002 n=4 | **+.071 [+.035,+.136] 5/5** | −.031 n=1 |
| math | −.033 n=2 | −.002 n=5 | — |
| news_homepages | −.014 n=11 | −.010 n=11 | −.023 n=3 |
| peer_review | n=0 (holdout scoring returned null — 2 bases, min-count fails) | | |
READ: the humor "gain" does NOT separate from the label-PERMUTED null selector — same-size
boost with no label information = content-free exemplar-format effect, not transmission.
Verdict shaping: optimized selection does NOT recover definition-level transmission beyond
format; the ONE surviving cell is CW/encoder-objective (n=5, all-wins) — flag for the
at-scale leg, don't headline. qwen25-72b selector (4/5 tasks, news mid-run): functional ≤
definition throughout (humor encoder 10/28 wins) — selector-dependence replicates v1 pilot.
Peer instrument gap: holdout min-counts fail at 2 bases — needs bigger peer slate in Leg 1.

## OVERNIGHT sk1 LANDINGS (2026-08-06): PUPA merge + MIPROv2-hover raw numbers
- PUPA official_merge @600 calls: seed_test .60486 → best_test .78494 (n_test 221) — raw,
  needs the 4-candidate one-judge re-mint rescore before any Table-1 use.
- hover MIPROv2 @2400: seed .356 → best .48466 — lands between GEPA@2400 (.4740) and
  GEPA@10110 (.5020), far below M_ω .5767 (all HB200 certified scale); re-mint rescore
  (5-candidate incl. MIPROv2-shipped) required before quoting side-by-side.
- ifbench seed-1 replicate @2400: .415 → .404 (ships-the-seed 2/2, already HB202).
- Triage chain RELAUNCHED on sk3 GPU7 after cache-layout fix (hf CLI wrote models--* WITHOUT
  hub/ prefix; 9 symlinks added shared_hf_cache/hub/ → ../models--*). gpt-oss-120b still
  rc=1 (incompatible, expected); magistral-24b running.

## FLIP-V2 qwen25-72b selector COMPLETE (2026-08-06 20:49) — CONTRADICTS the llama70b cells

Paired functional−definition, 20k bootstrap (flip_functional_v2_qwen25-72b.json, local):
null-to-NEGATIVE in EVERY domain×objective — humor frontier −.013 [−.048,+.020], encoder
−.007; CW frontier −.036* / encoder −.051* (the llama70b +.071 5/5 cell is **−.051 under
qwen**: selector-specific, as advisor suspected); math/news negative. With the STRONGER
selector, optimized exemplar selection never beats the stated definition. Remaining open
cell: llama70b null-on-ALL-bases (v2b, running GPU3) → then flip verdict final.

## OPS INCIDENT (2026-08-06 ~21:00): CUDA enumeration mismatch + zombie stubs
battery_triage_lane's CUDA_VISIBLE_DEVICES=7 landed on PHYSICAL GPU6 (bus E3) — lane lacked
CUDA_DEVICE_ORDER=PCI_BUS_ID (the standing rule; flip lanes got lucky). Chain left ALONE
(working, just mislabeled: its remaining legs occupy physical 6). Killed by explicit PID:
mag-waiter wrapper 1406990→child 1406992, orphan EngineCore stubs 559429 (GPU5, flip-qwen
teardown residue) + 1313292 (GPU7, magistral timeout residue). gen_smoke_lane.sh +
mag_retry.sh now export CUDA_DEVICE_ORDER=PCI_BUS_ID; smoke relaunched on physical GPU5
(setsid, immune to ssh-channel death — 1st relaunch died silently with its ssh); magistral
solo retry (timeout 5400) queued behind the chain for physical 7. Backend fix en route to
smoke: sk3 OfflineVLLM has no _maybe_lora (local-only refactor) → score_binary_gen now
calls eng.generate(texts, sp) plainly; magistral 40-min hang at weight-load under
load_format=auto = separate issue, retry will tell.

## 3b SMOKE PASSED + FULL-LADDER THINK PANEL LAUNCHED (2026-08-06 ~21:00)

**Smoke (gen_smoke_{qwen3-8b,r1-qwen-14b}.json):** qwen3-8b — nan 0.0 all readouts,
gen-nothink vs logprob agreement 1.00 (instrument-valid), think flips 15% of verdicts,
think traces mean 1,583 / max 2,819 chars, truncation 0.0. r1-qwen-14b — logprob nan 1.0
but GEN nan 0.0 (readout RESCUES the R1 family), thinks natively in both modes
(nothink_has_think_tag 1.0) → matched-base pairs measurable, no within-model toggle.
**Triage chain complete:** glm-z1-32b WORKS (z=1.31, auc .788, nan 0) — non-US-lab point
banked; r1-llama-8b exactly .500 (degeneracy signature 3rd family member); r1-qwen-7b nan
.82 / 1.5b auc .53; seed-oss-36b all-nan (incompatible, gen-readout rescue candidate);
magistral solo retry running (physical 7, timeout 5400).
**LAUNCHED: gen_think_panel across FULL qwen3 ladder** (definition arm, all humor slate
bases incl. planted, 300 probes, both modes, MAX_GEN 1536): GPU5 chain 8b→32b, GPU6 chain
1.7b→4b→14b. Outputs mbar_zxagen_{think,nothink}_humor_{exec}.npz. Harvest = think−nothink
gain per metric class vs z: prereg prediction in task #26 (gains concentrate on contested
classes iff self-articulation > restatement). Ops hardening: lanes now abort-if-busy after
waitfree (no proceed-anyway), setsid launches, CUDA_DEVICE_ORDER=PCI_BUS_ID pinned.

## 3b HARVEST — THINK-GAIN LADDER COMPLETE (2026-08-06 21:10 PT): PREREG LANDING ZONE 2

Full qwen3 ladder, definition arm, balanced agreement vs frontier-dossier ref, think−nothink:
| z | PLANTED | DIALECT | REACHES | TACIT-CAND |
|---|---|---|---|---|
| 1.7B z=.98 | **+.335** (.605→.940) | +.196 | +.144 | +.113 |
| 4B z=1.19 | **+.207** | +.097 | −.029 | +.059 |
| 8B z=1.98 | +.030 | −.008 | −.029 | −.022 |
| 14B z=2.32 | +.095 | −.001 | −.006 | +.013 |
| 32B z=2.42 | +.015 | +.062 | −.032 | −.018 |
READ: think-gains are (a) INVERSELY capability-graded — large below z≈1.2, ≈0 from 8B up;
(b) ordered PLANTED ≫ DIALECT > REACHES > TACIT-CAND at the small end — largest exactly
where content is code-checkable, smallest on the contested/tacit classes. A thinking 1.7B
hits .940 on planted = frontier no-think level. This is the prereg's SECOND landing zone:
reasoning = internal restatement/derivation of already-articulable content (compute
amplification for EXECUTING stated rules), NOT access to inarticulable content. The tacit
residual survives inference-time reasoning at every rung → the articulation bound is not
breached by self-articulation. Slots into §4.3 as the I3 panel closer + receiver-complement
framing (reasoning complements CAPABILITY, not articulation).
Caveats: humor only, definition arm only, one family, n=5-9/class, ref=no-think frontier.
Artifacts: sk3 outputs/osl_multi/mbar_zxagen_{think,nothink}_humor_qwen3-*.npz (10 files);
harvest inline this session. Extension candidates (tomorrow, user call): 2nd domain (CW),
name-arm (does thinking rescue possession?), R1 matched-base pairs via same readout.

## PREREG — 3b-EX: THINKING x SCALE x EXAMPLES (frozen 2026-08-06 ~21:20 PT, BEFORE harvest)

User directive: "thinking + scale + examples, we still need to more fully explore."
Design: exemplar arms (exemplars / exemplars_mm / def_exemplars / exemplars_authored /
exemplars_authored_mm, from freeze_zxa_ex_humor_v1) x qwen3 ladder x think/no-think,
same generative readout + frontier-dossier reference + exemplar_idx masking as all ex arms.
Launched GPU5 behind CW ladder; outputs mbar_zxagenex_{mode}_humor_{exec}.npz.
FROZEN READOUT: content-specific exemplar effect = (true − mismatched), separately for
corpus and authored pairs, per rung, per mode. The think-gain OF that difference is the
quantity of record (not raw exemplar levels — format effects cancel in the mm-subtraction).
FROZEN PREDICTIONS:
  P-RESCUE: thinking raises (true−mm) below the no-think emergence point (z<2) — reasoning
    substitutes for induction capability; induction emergence shifts LEFT. Would partly
    reopen the demonstration channel and qualify stated≫shown at small scale.
  P-RESTATE (favored by tonight's definition panel): thinking leaves (true−mm) ≈ 0 at all
    rungs while it lifts definition/planted transmission — reasoning amplifies only STATED
    content; the induction boundary is a capability wall that compute cannot cross.
  Discriminator: think-gain(true−mm) vs think-gain(definition) per rung; report both with
    paired bootstrap over bases. Landing either way is §4.2b/§4.3 bridge material.

## 3b-EX HARVEST (2026-08-06 22:40 PT) — PREREG DISCRIMINATOR: P-RESTATE

Content-specific exemplar effect (true−mismatched, exemplar_idx masked, vs frontier ref):
| rung | corpus nt→th (gain) | authored nt→th (gain) | definition think-gain (ref: 3b panel) |
|---|---|---|---|
| 1.7B | .000→.003 (+.003) | −.006→.023 (+.029) | planted +.335 / dialect +.196 |
| 4B | .016→.042 (+.026) | .022→.057 (+.035) | planted +.207 |
| 8B | .022→.039 (+.017) | .037→.040 (+.003) | ≈0 |
| 14B | .005→.054 (+.049) | .018→.043 (+.025) | planted +.095 |
| 32B | .047→.051 (+.004) | .046→.050 (+.004) | ≈0 |
READ: think-gains on the CONTENT-FROM-EXAMPLES channel are +.00 to +.05 at every rung —
an order of magnitude below the stated-channel gains at small z (+.20-.34). No leftward
shift of the induction emergence point: a thinking 1.7B executes stated rules at frontier
level yet still extracts ≈nothing from demonstrations. **P-RESTATE: reasoning amplifies
stated content; the induction boundary is a capability wall that inference-time compute
does not cross.** (Small positive gains at 4B/14B = "at most marginal" pending paired
bootstrap — do not claim exactly zero.) Bridge sentence for §4.2b→§4.3: the two remedies
that DON'T work at small scale (examples, k-dose) and the one that DOES (thinking over
statements) triangulate the same conclusion — transmission is bounded by what can be
STATED, and compute only helps execute statements, not induce or receive what statements
don't carry. n=14 corpus / 29 authored pairs; humor; qwen3 family.
Artifacts: mbar_zxagenex_{mode}_humor_qwen3-*.npz (10 files) + inline harvest.

## THINK-LADDER CROSS-DOMAIN REPLICATION (2026-08-06 22:50 PT): CW + PEER CONFIRM

PLANTED think-gain by rung — humor / creative_writing / peer_review:
1.7B +.335/+.209/+.135 · 4B +.207/+.201/+.093 · 8B +.030/+.017/+.043 ·
14B +.095/−.033/+.003 · 32B +.015/−.076/−.078
**Inverse capability-grading of the think-gain on stated rules replicates 3/3 domains**
(peer is perfectly monotone). NEW sub-finding: think-gain goes NEGATIVE at top rungs in
CW and peer (−.03 to −.12 at 14B/32B incl. CW dialect −.121 n=1) — overthinking cost on
already-mastered content, consistent with phi4-reasoning < base phi4 (−.26z). Flag for
paired bootstrap; do not headline before CIs. n small (CW 3-4/class, peer 1-5).
Artifacts: mbar_zxagen_{mode}_{creative_writing,peer_review}_qwen3-*.npz (20 files).
3b domain matrix now: humor FULL (def+ex panels), CW def, peer def; remaining: math/news
def panels + R1 matched-base pairs (tomorrow's block, GPUs free once flip-null ends).

## FLIP VERDICT CLOSED (2026-08-06 23:00 PT) — null-all control in; ADVISOR RULE → FOOTNOTE

Full-n within-base paired contrast (v2 true objectives vs v2b null-all, llama70b selector):
humor n=22: d_true +.039 vs d_null +.004 → paired +.035 CI[+.017,+.054], true>null 82%;
CW n=4: paired +.062 CI[+.015,+.145] 4/4; news null (instrument-suspect); peer/math n<3.
The earlier "null matches functional" was the cross-composition artifact (advisor called
it). Applying the PRE-AGREED decision rule (functional>null ⇒ scoped shown-channel
footnote, NOT an overturn): **flip-selected exemplars carry a real label-dependent
component (~+.04 over definition) — but ONLY for the llama70b selector/receiver; qwen25-72b
shows functional ≤ definition everywhere.** Receiver-relative demonstration channel:
demonstrations add where definitions are weakly absorbed (llama def .746 vs qwen .876) —
consistent with family-dialect ownership. stated≫shown stands; footnote text should say
"selection can recover a small, receiver-specific exemplar channel that placebo selection
cannot" and cite both selectors. User's flip-functional design: VALIDATED as an instrument
(the null control + three-way split did exactly their job).

## QUEUED (user 2026-08-07 am): CHANNEL x SCALE OSL EXHIBIT + edge-case emphasis

User: edge cases where demonstrations succeed are of primary interest; want ALL variants
crossed with model size, OSL-style. Two parts:
(A) ANALYSIS JOIN (no GPU): assemble one exhibit from EXISTING panels — arms name/def/
    def+ex/corpus-ex/authored-ex(+placebos) x 9-executor ladder x 4-5 domains (36 panels)
    + think/nothink ladders (humor/CW/peer/math) → per-class curves over z per channel;
    treat channel curves like OSL regimes (emergence point, plateau, crossings). Edge-case
    callouts: def+ex>def cells, 70B-class content emergence, receiver-relative flip channel.
(B) NEW RUNS: flip-selected (functional) rubrics scored across the FULL local ladder
    (logprob readout, both selectors' selected sets, exemplar_idx masked) — tests whether
    the llama-receiver-specific +.035 channel is a curve (grows/shrinks with z) or a point.
    Plus math think-panel (launched GPU3) + R1 matched-base gen pairs (launched GPU5).
Morning allocations: GPU3 math think-ladder; GPU5 r1-qwen-14b/32b + phi4-reasoning +
qwen25-14b/32b gen pairs; GPU6 agent A magistral-load fix; GPU7 OCCUPIED by other lineage
(PID 2122639 — hands off) → agent B (seed-oss/gpt-oss rescue) does CPU diagnosis first.

## MATH THINK-LADDER (2026-08-07 12:40 PT): 4/4 DOMAINS DIRECTIONALLY CONSISTENT
math PLANTED think-gain: 1.7B +.070 / 4B +.017 / 8B −.035 / 14B +.027 / 32B +.103;
REACHES: +.131/+.011/−.048/+.050/−.090. Small-z gain positive in 4/4 domains
(1.7B planted: humor +.335, CW +.209, peer +.135, math +.070 — magnitude tracks how much
the domain's texts are readable by a 1.7B), mid-ladder ≈0-to-negative 4/4. Top-rung: math
32B planted +.103 is the one outlier vs the CW/peer negative-at-top pattern (32B REACHES
−.090 fits) — top-rung behavior stays "mixed, needs CIs", do not fold into the
overthinking claim yet. Artifacts mbar_zxagen_{mode}_math_qwen3-*.npz.

## FABLE AUDIT RULING (2026-08-07): §4.2b STAYS A SUBSECTION — do not fold

Verdict: the examples program is six orthogonal instruments converging on ONE
characterization (capability-gated / dose-saturated / receiver-relative / additive-only),
and demonstrating that the bound survives every adversarial manipulation IS the thesis of
an "upper bounds" paper. Folding would overload §4.2, bury the code-truth + null-selector
controls reviewers of negative results demand, and forfeit the standalone ICL contribution.
Skeleton P1-P7 adopted (P6 = flip as methodological positive with scoped footnote; P7 =
think-wall bridge to §4.3). CONTINGENCY RULE FROZEN: structure fixed, only prose promotes —
flip becomes a named result with figure iff ex−def>0, 95% CI excluding 0, ≥2 domains,
null-controlled, in ≥2 receiver families OR one frontier receiver; else stays footnote.
Framing language captured in the audit transcript. Note hygiene: duplicate flip-verdict
section removed (audit caught it).

## SCALE-UP DIRECTIVE (user 2026-08-07 pm): "instruments need more scale" + frontier receivers

Mapping instrument → scale-up in flight/queued:
1. stated-vs-shown grid → full-bank flip-v3 (284 bases, GPU3 queue, running tonight) + Leg 1
   full-bank arms when GPUs allow.
2. code-truth → more planted rules only if user asks (parked).
3. gold-label benchmark → done at benchmark scale (hotpot).
4. k-dose → full-bank replication rides Leg 1.
5. flip → v3 full-bank + functional-rubric LADDER (12 execs, queued ahead of v3).
6. think×examples → 4-domain replication done; CIs pending.
LIMIT LEG (user: "expand this scale question with bigger models incl GLM 5.2"):
- GLM-5.2 ex-arm panel LAUNCHED (API agent): humor+math, authored/bare pairs + def_ex,
  150-probe deterministic subset, 35M-token hard cap inside weekly window, planted-anchor
  gate ≥.85, output mbar_zxaglmex_{task}_glm-52.npz + per-class limit table vs qwen25-32b.
- gpt-oss-120b as 2nd frontier receiver: gated on GPU5 rescue validation (65GB MXFP4 fits
  one B200; harmony parse rule from rescue agent) — then ex-arm panel via score_binary_gen.
- Llama-405B-FP8 TP=4: still blocked on 4 simultaneously-free GPUs (other lineage holds 6/7).
Exhibit v1 landed: outputs/analyses/channel_scale_20260807/ — 9/30 top-rung limit cells
(authored 7/9, mostly REACHES); NOTE per-family view weakens the additive claim (def+ex >
def only top-qwen REACHES +.023/.034 and llama endpoints) — P5 of the 4.2b skeleton must
use the per-family numbers, not the pooled 4/5-domains line.

## 3b MATCHED-BASE HARVEST (2026-08-07 13:30 PT): POST-TRAINING ≠ INFERENCE-TIME THINKING

R1-distill vs same-pretraining instruct base, generative readout both sides (R1 native-think
vs base no-think — the mode difference IS the treatment), humor, vs frontier-dossier ref:
| pair | DIALECT | PLANTED | REACHES | TACIT-C |
|---|---|---|---|---|
| r1-14b vs qwen25-14b | **+.089** | +.003 | −.056 | +.004 |
| r1-32b vs qwen25-32b | **+.090** | +.069 | −.083 | −.036 |
READ: reasoning POST-TRAINING gains concentrate on DIALECT-SUSPECT (both rungs, +.09) and
persist at 32B — a different signature from the within-model think toggle (planted-heavy,
vanishing by 8B). REACHES degrade under R1 (−.06/−.08; deliberation hurts already-anchored
constructs — consistent with the overthinking cells). TACIT-CANDIDATE ≈ 0 both rungs: the
tacit wall holds under reasoning post-training too. n=5-9/class, needs paired bootstrap;
dialect n=6. Mechanism question for §4.3 prose: does distilled deliberation re-align
family dialect (R1 trained on DeepSeek traces), or does deliberation help contested-but-
statable constructs generally? Cross-family test would need r1-llama-8b vs llama8b (gen
readout run pending — r1-llama-8b logprob was degenerate).
INSTRUMENT FLAGS: phi4-reasoning under GEN readout = chance on all classes (.47-.50) while
its LOGPROB battery is healthy (auc .842) — its template defeats the </think> parse; gen
numbers INVALID for phi4-reasoning until a template-specific parse lands (queue with the
rescue-agent parser work). R1 enable_thinking no-op CONFIRMED exactly (think−nothink =
+.000/.001 all classes — same native mode both passes).
