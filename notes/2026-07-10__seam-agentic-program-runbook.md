# Seam agentic program — runbook & pre-registrations (2026-07-10)

User approved ALL proposed workstreams ("all of these are good, cue up everything"). This note
is the doc-of-record: designs are PRE-REGISTERED here before any data; deviations get logged
here. Parent thread: notes/2026-07-01__metric-seam-pilot-results.md; rules: BEST-PRACTICES.md
[metric seams]. Positioning guard (applies to every WS): these are INSTRUMENTS with
certificates — each WS is pitched by what it measures, never by "our agents compile better."

Resource gating: Claude weekly quota resets 2026-07-11 — all Sonnet/Opus fleets launch AFTER
reset (probe with 1 agent first; burst limits mimic the weekly wall). Fleet envelope: ≤20
concurrent, ≤150 items/shard, Sonnet fan-out / Opus judgment / exact data paths in every
prompt. GPU work (re-extraction, new judging) rides sk3 via the queue/gpu_waiter pattern —
nvidia-smi before any claim; OSL thread has lanes in flight, never contend.

---

## WS1 — Compiler-effort ladder + construct-contract crews  (PRIMARY)

**Measures:** the effort-dependence of the seam — is the uncertified residual genuine
judgment-layer or search shortfall? Converts "saturation" from one budget point into a
per-criterion certification-effort curve, read with the OSL curve-shape verdicts.

**Panel (FROZEN 2026-07-10):** battery/effort_ladder/panel_freeze.json — 32 criteria over the
5 fleet tasks (2 certified controls + 4 mid-headroom + ≤2 floor per task; rule + cam sha in the
file) + plants p901/p903/p905/p907 riding every rung on PR items.

**Effort rungs (x-axis):**
- E0 = h0 (1-round, archived — free)
- E1 = agentic-r2 protocol (6-round reflective + op invention — exists for overlap criteria,
  re-run only where missing)
- E2 = contract-guarded crew: planner + implementer + adversary, 6 rounds
- E3 = E2 + execution feedback (crew sees TRAIN-side score distributions + planted CF traps;
  test stays sealed)
- E4 = Opus-planned crew (Opus plans/arbitrates, Sonnet implements)

**Construct contracts (the discipline layer, WS1a — authored FIRST):** per criterion, one
frozen contract = verbatim construct definition + 4-6 counterfactual probes (minimal pairs the
construct must separate) + 2 discrimination checks (score std / frac-min bounds on train).
Authored by a Sonnet fleet from the aspect bank entries, validated centrally (probes must be
label-free, executable, and pass on h0 where h0 is certified). The adversary runs the contract
EVERY round; a candidate that fails contract never reaches a gate. Targets the two observed
failure modes: construct replacement (h1 fleet, 0/10) and constant-collapse (GEPA a135).

**Readouts & pre-registered classification:** per criterion, r̃(effort) = ρ_test/ceiling at
E0..E4 under the SAME frozen items/splits/judge (Gemma) + G1 bootstrap gates. Curve verdicts:
REACHES (certifies at some rung), RISING (still climbing at E4 — effort-censored `>E4`, never
"uncodeable"), FLAT (Δr̃ ≤ .03 across E1→E4 = saturation evidence, the honest tacit candidate),
DEGRADING (contract/gate catching overfit). Depth = min rung to certify. Promotion rule
(pre-registered): candidate replaces head only at P(cand>head)≥.8 held-out; controls must not
degrade (any control gate-flip = instrument alarm, halt and audit). Plants calibrate per rung:
plant-certification must be monotone non-decreasing in effort; a plant that DE-certifies at a
higher rung = harness bug.

**Cost sketch:** 36 units × 3 new rungs × ~3 agents ≈ 330 Sonnet calls (+ ~35 Opus at E4) +
one field re-extraction GPU pass per rung per task (queue2 pattern). Order: E2 on
press_releases + humor first (poles, ~1 fleet-day), read, then math/CW/legal, then E3, E4.

## E2L — LIBRARY-AUGMENTED E2 (user-directed design amendment, 2026-07-11, PRE-REGISTERED
## before any E2L data)

**User's observation:** the 9 function-wall kills were all REGEX-genre code — "code can't
hold a function boundary" may be "string-matching can't." Real compiled instruments execute
and verify (statcheck recomputes p-values; AST linters; the PR corpus runs tests).

**Design:** E2L = identical to E2 (same frozen contracts, same 3-role crews, protocol v2,
same adversary battery, TRAIN only, held-out promotion rule unchanged) with ONE change: a
sanctioned capability library (methods/metric_seam/hybrids/ops_capability.py) importable by
candidates — spaCy dependency-parse quote/attribution ops, SymPy parse/entailment-check ops
(via pylatexenc), statcheck-style consistency recomputation, date/deadline arithmetic,
citation parsing, NER-based fact-density. All ops label-free; ops keying on metadata/length
rejected (WS2 content guard applies). Library FROZEN per E2L wave (versioned; crews may not
edit it mid-cell — toolsmith evolution stays WS2's question).

**Primary targets (pre-registered):** the 9 function-wall kill cells, hardest-first on the
killed AXIS: a31 (attribution via dep-parse), a150+a30 (vacuity via SymPy entailment),
a41/a111/a25 (asset/CTA/substantiation via NER+parse), a342/a117/a189/a315 (humor structure
— weakest library coverage, honest test of limits). Prediction registered BOTH ways: kills
persist with real capabilities → the wall is genuinely judgment-layer (strong form); accepts
appear → the E2 wall was toolbox-bounded and the E-axis needs a capability dimension.
Environment: sympy 1.14 / spacy 3.7.5+en_core_web_sm / nltk / dateutil / pylatexenc all
verified installed locally.

## WS2 — Toolsmith: op-library evolution

**Measures:** returns-to-institutional-accumulation — does a shared typed op library make late
criteria cheaper to certify (the mechanical re-enactment of why law sits at the compiled pole)?

**Design (pre-registered):** split each task's WS1 non-control panel into batch-A / batch-B by
stable hash of aspect id. Sequence: compile batch-A at E2 (toolsmith harvests ops each round →
generalizes → unit-tests on plants → publishes ops_<task>_lib vN). Then compile batch-B TWICE,
same crews/budgets: arm L = library-from-A importable; arm ∅ = frozen empty library. Paired
readout: Δr̃ and Δ(rounds-to-certify) B(L) vs B(∅); library growth curve + op-reuse rate.
Causal claim rests ONLY on the B(L)-vs-B(∅) contrast (same criteria, same budget). Op tests are
label-free; ops keying on metadata/length rejected (content guard).

## WS3 — Evidence-aware judge target M̄(x,Z)  (unblocks patents; prerequisite for WS4)

**Measures:** well-posed op-marginals — an evidence op can only be credited against a judge
that SEES the evidence (I(M̄(X);Z|X)=0 forced the patents null). Design: patents_pa 250 apps ×
4 evidence-dominant aspects; judge prompt = doc + PriorArtOps payload Z (label-free, strips
gold/rejection fields — existing leakage guard); NullOps twin = ablation. Gemma one-GPU-pass
(~6k prompts, sk3 one-off). Pre-registration: op-marginal vs M̄(x,Z) expected POSITIVE for
evidence-dominant criteria; vs doc-only M̄(x) stays ≈0 (the replication of the null is part of
the design). 2-pass reliability on the new target before any seam read (new target = new
ceiling).

**WS3 RESULT (2026-07-10; REVISED post external Codex review — eval v2: intersection-only
2-pass judges, NaN-skipping bootstrap; report tasks/patents_pa/ws3_eval_report.json).**
Op-marginal (PriorArtOps full − NullOps twin, held-out R7.1 split) vs M̄(x,Z) is POSITIVE:
a26 +.211 (P>null .976), a34 +.661 (P 1.00 — v1's P .62 was a bootstrap-NaN artifact:
723/2000 resamples had undefined null-rho counted as losses; defined-pairs P = 1.0),
a35 +.609 (P 1.00). Doc-only M̄(x) marginals: a26 −.218, a34 +.008, a60 −.019 (null
replicates), but a35 +.202 (P .95) does NOT replicate the null — its NullOps twin
anti-correlates with the doc judge (−.32); prediction (b) holds 3-of-4, stated as such.
Filler ≈ doc-only descriptively on all 4 (no equivalence test run; filler controls
length/header only, not payload syntax). Evidence raises judge 2-pass reliability on all 4
(.873/.640/.197/.916 vs .774/.444/.133/.913). a60 EXCLUDED as an EXPLORATORY quality
exclusion (evidence-target rel1 .197; no reliability cutoff was pre-registered — cutoff
rel1≥.30 now FROZEN for future evidence-judge runs). SCOPE (review-added): judge and
hybrids consume the SAME disclosure-summary representation — the result establishes
well-posed op-marginals (visibility/alignment), the WS3 question; external validity of the
representation itself is separate. SCOPE NARROWED FURTHER (Codex audit via patents thread,
2026-07-10, recorded in datasets/patents/2026-07-10__evidence_aware_judge_ws3.md): the
option3 candidate sets under pa_features.json were built with examiner gold FORCE-INCLUDED
for targeted claims → positive marginals may partly reflect examiner-citation leakage;
"the judge needs the evidence" stands, "the retriever DISCOVERS the evidence" is NOT
established (clean test = retriever-only candidates, examiner doc held out). → patents'
criterion-level nulls were JUDGE-VISIBILITY artifacts; WS4 patents cells use M̄(x,Z) with
this scope caveat attached.

## WS4 — Workflow rung: typed DAGs  (the proposal's unbuilt third rung)

**Measures:** where judgment lives INSIDE a program — seam position as program structure, not a
scalar. Pilot 8-10 criteria: PR retrieval-friendly (a119 + 2), patents_pa on the WS3 target
(3), legal procedural chains (a23 exhaustion + 2). Nodes typed R/T/A, each C or L, per-node op
declaration (evidence vs computation — cite the right ceiling per node), per-edge lattice-level
match check. CLC aperture cells reuse seam-position machinery. THEORY TASK rides along: extend
U₂ matroid over the DAG partition (lemma-gap list). Gate machinery unchanged (G1 held-out).
Starts AFTER WS3 lands (patents cells need the evidence-aware target).

**PILOT REGISTRATION (2026-07-10, frozen before any DAG authoring):** 9 cells —
PR a119/a28/a76 (top-3 by ops-call count in programs_v2, measured 2/4/2), patents a26/a34/a35
on M̄(x,Z) (a60 excluded by the frozen rel1≥.30 cutoff), legal a23 (exhaustion chain) +
a21 (neutral-policy→disparity chain) + a13 (constructive-discharge conditions chain).
Machinery: battery/dag_schema.py (schema, executor+trace, per-edge level-match validator,
seam_frontier readout, score() adapter — smoke-tested incl. level-inversion catch).
Readouts per cell: (i) held-out r̃ vs the scalar-hybrid incumbent under the SAME G1 gates
(op_class-matched ceilings: patents nodes citing evidence ops use the M̄(x,Z) rel1);
(ii) seam_frontier (where judgment enters + its abstraction level); (iii) per-node marginals
via train-median node ablation. DAG authoring crews follow E2_CREW_PROTOCOL v2 (one crew per
wake, adversary battery + laundering probe unchanged; level-inversion check runs every round).

## WS5 — Shared-schema compilation  (legal pilot)

**Measures:** whether evidence-starvation is partly an amortization problem. One typed
task-record schema for title_vii (elements record: protected class, adverse action, exhaustion
dates, employer size, ...), authored once by a small crew; the 20 criteria recompiled as
predicates over the record; compare per-criterion r̃ + total compile cost vs current per-metric
hybrids. Success = floor lift on the survey-zero-diagnosed criteria (a36 same-actor, a28
15-employee pattern) at lower marginal cost. Extension on success: ss_disability (drop-in).

## WS6 — Cleanup queue (small, each pre-registered inline)

1. **a42 fresh items** — frozen h1 vs h0 on a NEW stable-hash item sample (never rule-bend the
   .011 miss). GPU judging + eval only.
2. **a110 Llama-side h1** — one improver round with Llama as target judge; tests whether
   judge-dependent gates are recoverable judge-side (paper limitation → measurement).
3. **Selected-vs-enculturated at matched binding** — same field slot, payload = CE/LDA-selected
   features vs LLM-enculturated text, binding rigidity matched; readout = transport ratio +
   fidelity. The one untested theory cell (provenance ladder).
4. **Legal breadth** — flsa + 2 more registry domains, title_vii survey recipe (survey-grade,
   lean).

## Launch order

| when | action |
|---|---|
| tonight (no quota) | this runbook; panel freeze ✅; WS1a contract TEMPLATE + pack builder; WS3 prompt build staged (no GPU grab while OSL lanes run) |
| Jul 11 (reset) | probe 1 agent → WS1a contract-authoring fleet (36 units) → validate centrally |
| Jul 11-12 | WS1 E2 on PR+humor → read → remaining tasks; WS3 GPU pass when a card frees (gpu_waiter) |
| next | WS2 batch design (after first E2 read fixes budgets); WS6.1/6.2 alongside (small); E3/E4 rungs |
| gated on WS3 | WS4 pilot; WS5 after WS1 E2 read (crews validated) |

**Standing guards for every WS:** held-out gates always; plants every rung; 4-tuple certificate
stamps; apples-to-apples inputs; no label-aware metrics; constancy checks on every program;
ranks normalized within-group before pooling + perm-null on pooled stats; report effort/curve
censoring, never bare "uncodeable".

---

## OVERNIGHT ORDERS (2026-07-10 night; user asleep, hourly cron wakes, standing go-ahead)

Wake protocol: read this section + TaskList (#8-14) → ONE consolidated status check → advance
the next unblocked step → append ONE line to the OVERNIGHT LOG below → stop. Lean monitoring:
no polling between wakes. If genuinely blocked on a user decision, log it and advance other
lanes; never rule-bend a pre-registration.

**A. WS3 GPU pass (no Claude quota needed — first priority).** Prompts are ON sk3 already
(4,000 rows verified: outputs/metric_seam_pilot/tasks/patents_pa/ws3_evidence_prompts.jsonl).
Launch the Gemma judge pass → ws3_evidence_results.jsonl next to it. Scorer =
gemma_score_v1.py (methods/metric_seam/ tree; READ ITS HEADER for CLI before running). MUST
replicate the queue2 runner env (bare-SSH AFS home breaks flashinfer JIT: PATH+=cuda-12.8/bin,
CUDA_HOME, LD_LIBRARY_PATH, TMPDIR + CUDA_CACHE_PATH on /lfs, HOME=/lfs pin) and use the
gpu_waiter pattern (free GPU <2GB, double-check, retry on contention — OSL lanes are in
flight, NEVER contend, 1 GPU). Verify launched = pgrep + engine-init log line. Per-wake check
= `wc -l` on the results file. On completion: rsync results back, run 2-pass reliability on
the evidence arm (new target = new ceiling), then op-marginal eval per WS3 prereg; update #11.

**B. Claude-quota probe + WS1a fleet.** Once per wake, probe with ONE tiny Sonnet agent
(burst limits mimic the weekly wall — 1-agent probe is the documented test). When quota is
back: launch the WS1a contract-authoring fleet — one Sonnet agent per pack in
outputs/metric_seam_pilot/battery/effort_ladder/contract_packs/ (32 packs; give each agent
the EXACT pack path + output path; ≤20 concurrent). Output contract JSON per the pack's
author_instructions → battery/effort_ladder/contracts/<task>__<aid>.json. Central validation
before freeze: definition verbatim-matches criterion_description; 4-6 probes, pos/neg
nonempty + distinct; no probe references labels/metadata/length. Then update #8.

**C. WS1 E2 crews (#8 DONE — contracts frozen 32/32).** ONE crew per wake (= 3 Sonnet
agents, the full ration): follow
outputs/metric_seam_pilot/battery/effort_ladder/E2_CREW_PROTOCOL.md — roles, tools
(agentic_run.py + contract_check.py, both local/cost-free), output layout, and the
13-criterion queue are all pre-registered there. Take the next queue entry without an
e2/<task>__<aid>/ dir; TRAIN only.

**D. Do NOT touch:** OSL lanes/crons/GPU-waiters (other thread); census GLM quota; anything
requiring a new measurement target.

**E. PACING (user order 2026-07-10 ~02:00, overrides fleet envelope):** cue agents SLOWLY —
the 5-hour rolling budget must last all night. Per hourly wake: at most ~4-6 Sonnet agents
(one E2 crew = 3 agents ≈ one wake's ration; contract-validation fixups count too). Never
launch a burst >6 between wakes; if a step wants more, spread it across wakes. GPU work is
free (doesn't count); prefer advancing GPU lanes when the agent ration is spent.

### OVERNIGHT LOG
- 2026-07-10 ~23:50 staged: runbook + panel freeze (32+4) + contract packs 32/32 + WS3 prompts
  built and rsynced to sk3 (4,000 rows verified); hourly cron armed; fleets await quota reset.
- 2026-07-10 ~02:10 WS3 gpu_waiter live on sk3 (pid 2633845, waits for a <2GB card, resume-safe
  scorer). Quota probe PASSED → WS1a fleet ran (32 Sonnet agents, ~35K tok each): 32/32 contracts
  authored; central validation 23 clean + 9 keyword-flags all adjudicated benign on manual read →
  **32/32 FROZEN** (sha stamps in contracts_validation.json). #8 DONE, #9 unblocked. NOTE: user
  pacing order landed mid-fleet — section E added; E2 crews now ONE per wake (3 agents/ration).
- 2026-07-10 ~02:35 E2 harness built cost-free: contract_check.py (probe SEP/INV + train
  discrimination; h0 on PR a2 = 6/6 SEP, PASS) + E2_CREW_PROTOCOL.md (roles, tools, output
  layout, 13-criterion queue). Next wake: crew 1 = press_releases a41. WS3 waiter still
  waiting (all 8 cards busy with OSL).
- 2026-07-10 ~03:25 wake: WS3 waiter alive, still 0 free cards (OSL holds all 8). CREW 1
  (PR a41, 3 agents = ration): planner diagnosed gate-override bug; implementer .4502→.4730
  contract 6/6 (4 rounds, 1 self-revert); adversary **REJECT** — round-2 detectors fire on
  format/DPI/hex MENTIONS not asset OFFERS (wrong-way perturbation .3834>.2425), = the
  pre-registered construct-replacement failure mode, caught by design. Cell logged in
  e2/press_releases__a41/meta.json; protocol amended (no probe-targeted detectors); h0 itself
  fails its contract 2/6 (probes authored blind — a real E2 finding). Next wake: crew 2 =
  humor a342.
- 2026-07-10 ~06:40 wake (4 queued wakes = ONE ration): **WS3 GPU PASS DONE** (waiter caught a
  card, 4,000/4,000 rows, rc=0, scores healthy: full 0-10 range, 0 unparsed, 1.2% NA) →
  rsynced, eval run, WS3 RESULT block added above (3/4 aspects confirm op-marginal>0 on
  M̄(x,Z), null replicates on M̄(x), filler clean, a60 excluded target-unreliable). #11 DONE,
  #12 (WS4) unblocked. Crew 2 (humor a342) planner launched — chain continues on
  notifications.
- 2026-07-10 ~07:20 crew 2 (humor a342) complete: implementer .3532→.3960 contract 6/6
  (5 rounds, disciplined); adversary **REJECT** — mention-vs-presence AGAIN: 5/6 genre
  regexes fire full-strength on mention-only prose (identical 0.16 = 0.16 to genuine
  execution), single-marker rescue was the enabler; frozen probes structurally miss it
  (text_negs avoid trigger phrases). E2 pattern emerging after 2 cells: contract+adversary
  = honest nulls, gains are mention-inflation; protocol amended (co-occurrence rule;
  contract-pass = necessary-not-sufficient). h0 fails own contract in BOTH cells (2/6, 0/6).
  Next wake: crew 3 = press_releases a31.
- 2026-07-10 ~08:40 crew 3 (PR a31) complete: implementer honest (rho wash .6625 vs .6629,
  robustness rework, own laundering self-tests, early stop rd 2); adversary **REJECT** —
  3rd distinct sub-mode: NO cheating, but attribution-regex precision holes leak third-party
  discounts onto self-voiced overclaims (train d00964 0.00-judge: h0 .000 → cand .303) and
  full-train scan = error-type SWAP on mention-vs-presence (20 new wrong-way vs 23 fixed).
  E2 pattern after 3 cells: 0 accepts; binding constraint = voice/attribution/presence axis,
  which code-level machinery keeps failing three different ways. Next wake: crew 4 =
  humor a189.
- 2026-07-10 ~10:05 crew 4 (humor a189) complete: strongest implementer yet (.2597→.3921,
  +51% rel, dominance audit 86/62, 3 self-reverts, paid rho for FP fixes) — adversary
  **REJECT** #4, sharpest yet: (i) ~7%-recall form detector gates 70% of weight → 64% of
  top-judge non-template items regress (thin-spread class regression under real net gains);
  (ii) rounds.log round-6 claim 'd01296 fixed' is FALSE on re-run of the crew's own audit
  tool — first implementer-integrity finding; (iii) brevity leak (craft-free 10-worder inside
  genuine-epigram band). Protocol amended: adversary re-verifies per-item log claims, audits
  per-CLASS deltas, kills low-recall-gates-high-weight. 4/4 rejects; 4 distinct kill modes.
  Next wake: crew 5 = press_releases a111.
- 2026-07-10 ~11:30 crew 5 (PR a111) complete: **first ACCEPT** (cell 5/13, now 1A/4R).
  Implementer .5332→.5838 (+9.5%), contract 6/6 vs h0's 2/6, recalls measured & proportionate
  (signup detector tightened 85→7/150), tercile audit FP-concentrated; adversary reproduced
  EVERY re-executable claim, probes clean or strictly-better (fixes an h0 inversion on the
  diffusion boundary), sole blemish an oversold sentence (.005, disclosed outlier). NOTE:
  accepted-at-E2 ≠ certified — held-out G1 + promotion rule deferred to batch eval per prereg.
  Read: the accept came from the crew carrying all 4 accumulated kill-lessons — protocol-level
  accumulation datum (WS2-relevant). h0 fails own contract AGAIN (2/6; that's 4 of 5 cells
  where h0 fails or was never probed on this axis). Next wake: crew 6 = humor a117.
- 2026-07-10 ~12:40 wake: HOLD (user awake + steering). E2 ladder PAUSED pending external
  Codex review (gpt-5.6-sol, running) of harness + accepted cell + WS3 stats. This hour
  (user-directed): gate audit DONE — contract_check verified on known-answers (constant/
  inverted/h0), PROBE-MODE-DETECTION hole found + patched (sentinel-dict guard; exploit demo
  caught; a31's `if extracted:` branch trips it retroactively — no verdict changes; a111
  still PASS under hardened gate); WS3 split identity verified byte-identical vs R7.1; a34
  P-anomaly explained (null twin 99% constant on test → bootstrap noise). Patents hand-off
  note written: datasets/patents/2026-07-10__evidence_aware_judge_ws3.md (+README pointer,
  memory scoped-corrected). Crew 6 resumes only after Codex report is reconciled.
- 2026-07-10 ~13:40 CODEX REVIEW RECONCILED (report battery/effort_ladder/codex_review_
  2026-07-10.md; verdicts a111 BROKEN / harness BROKEN / WS3 BROKEN / rejections SOUND /
  design-holes BROKEN — most findings CONFIRMED on verification). Actions: (1) harness v2
  FROZEN: agentic_run + contract_check now train-only execution (a111 numbers unchanged
  under v2: .5838/.5332, 6/6), completeness/range gates added; (2) a111 DOWNGRADED to
  accept-with-defects, BLOCKED from promotion (laundering inversion at candidate.py:342
  confirmed by executed probe — diffuse+incidental-ask .3025 > clean single-contact .2700;
  root cause = my adversary prompt's 'no worse than h0' excuse rule — v2 protocol makes
  construct validity absolute); (3) WS3 eval v2: intersection-only judges + NaN-skipping
  bootstrap → a34 P .62→1.00, a26 .85→.98, doc-only a34 marginal +.008; a35 doc-only
  non-null stated honestly (3-of-4); a60 exclusion labeled exploratory, rel1≥.30 cutoff
  frozen; coupling caveat added to note+runbook; (4) validate_contracts v2 (flags=warnings,
  per-flag adjudications file, exit code) re-run → 32/32 with 14 specific adjudications;
  (5) E2_CREW_PROTOCOL v2 frozen (absolute-validity kill rule, laundering probe mandatory,
  v1-era cells 1-5 labeled learning-curve-confounded). bootstrap_gate gains opt-in
  skip_undefined (default OFF — frozen results untouched).
- 2026-07-10 ~14:30 user go on all lanes. Crew 6 (humor a117, first v2 cell) launched → hit
  the 5-hr session limit (resets 11:30am PT); cron retries after reset. Quota-free lanes
  advanced instead: **WS6.1 a42 fresh-items STAGED+ARMED** — resolution rule FROZEN in
  build_a42_expansion.py (400 stable-hash fresh items of 4,750 unused; promote iff
  P(h1>h0)≥.80 AND G1-floor P(rho≥.60)≥.95 on combined 500 held-out, intersection judges;
  margin-arm G1 disclosed unavailable), 2,400 prompts (2 judge passes + 4 fields ×400)
  rsynced to sk3, a42_gpu_waiter armed (never contends); eval_a42_expansion.py ready.
  #14 in progress.
- 2026-07-10 ~11:35 wake: **WS6.1 RESOLVED — h0 STAYS HEAD.** a42 GPU pass done (2,400/2,400
  rc=0, fresh rel1 .790); eval bug caught+fixed (h1 needs its OWN field extraction
  field_results_h1.jsonl 'a42.h1__' keys — plumbing validated by exact repro of fleet
  numbers on old-100: h0 .5500, h1 .6345). Frozen rule applied: combined n=497
  P(h1>h0)=.595 <.80, floor P=.005 → NO promotion. Key read: h1's +.085 old-test edge
  VANISHES on 397 fresh items (−.009) — the fleet's no-transfer finding confirmed on 5×
  fresh sample for its best h1; never-rule-bend vindicated. (Caveat: old/fresh pools use
  different extraction batches, symmetric within pool.) Quota probe PASSED post-reset →
  crew 6 (humor a117, v2) planner resumed. WS6 remaining: 6.2 a110 Llama-h1, 6.3
  selected-vs-enculturated, 6.4 legal breadth.
- 2026-07-10 ~12:50 crew 6 (humor a117) complete — FIRST FULL-v2 CELL: implementer flawless
  on discipline (.4393→.5340 +21.6%, recalls measured 9.3%/2.0%, laundering self-test
  exact-equality, integrity CLEAN — all claims reproduced) yet adversary **REJECT** #5:
  repetition-FUNCTION is not Jaccard-separable — near-dup detector penalizes rule-of-three
  escalation (frame-dominated overlap; constructed pair + live d01575 colon-script refrain
  evading the quote exemption) + tail-excision blind-deletes a punchline clustering two
  attribution markers. Read: the E2 judgment-layer axis now spans mention-vs-presence,
  voice-attribution, and repetition-function — surface recurrence ≠ role. 6 cells: 5R + 1
  blocked accept. (2 transient server rate-limit interruptions mid-adversary, resumed OK.)
  Next: crew 7 = press_releases a25.
- 2026-07-10 ~13:55 crew 7 (PR a25) complete: **REJECT #6, scoped** — Strategies 1+2 sound
  (ablation-verified gate removal + corroboration cap, +.054 rho, laundering-clean) but R4's
  zero-recall disclosure detectors are syntax-locked probe overfits (0/150 train, 0/4 OOD
  astroturf texts, killed by though→although swap) and they alone separated 6/6 from h0's
  own 4/6 fail → no valid passing candidate. SALVAGE noted in meta.json. Protocol addendum:
  self-tests must be saved; gap-closing detectors must fire on OOD realistic instances.
  Also this hour: WS4 pilot REGISTERED (9 cells frozen, dag_schema.py built+smoke-tested,
  #12 in progress). Tally 7/13 cells: 6R + 1 blocked accept. Next: crew 8 = humor a315
  (last mid), then floors/controls.
- 2026-07-10 ~15:10 wake (2 queued = one): crew 8 (humor a315) planner done (h0 .3638; strategy
  = collapse near-noise 30%-weight code_structure + 3-tier rebuild, sim ~.60); implementer hit
  session limit MID-ROUND (resets 4:10pm PT, transcript preserved — resume next wake, it was at
  rho .6042 with a frac_at_mode fix in hand). Quota-free this wake: MATH PULL-FORWARD registered
  in protocol (user-approved): from cell 9, interleave math m1-m7 (a150/a30/a60/a204 mids,
  a36/a222 floors, a198 control) with remaining humor/PR — tests whether the function-vs-surface
  kill pattern is taste-domain-specific (compiled-pole prediction: math reject rate drops).
  Coding-domain panel extension offered to user (needs items+judge+h0 build; datasets/code-review
  or MathlibPR as source) — awaiting explicit go.
- 2026-07-10 ~18:15 wake: quota back (4:10pm reset long past; a couple of cron fires coalesced).
  a315 implementer RESUMED from mid-round state (had rho .6042 + frac_at_mode fix in hand).
  Chain: adversary on its completion, then math m1 (a150) next wake per the pull-forward order.
  sk3 quiet; no other lanes need attention.
- 2026-07-10 ~19:20 crew 8 (humor a315) complete: **REJECT #7 — ALL 8 MIDS DONE, 0 survivors**
  (7R + 1 blocked accept). Ladder-best gain (.3638→.6042, integrity fully clean, genuine
  tie-skew diagnosis 74.7%@1.0) killed on the crew's OWN pre-flagged #1 risk: the load-bearing
  AND-NOT gate launders paraphrased deflation + one routine neutral word from VIOLATION .20 to
  max CLEAN .90 (L5-L8, matched controls) — fixed lexicons cannot hold a function boundary.
  Secondary: discretization confirmed boundary-gamed vs frac_at_mode bound (+.0014 only,
  disclosed, not kill-worthy). READ: crews now PREDICT their own kills without being able to
  prevent them = strongest E2 saturation evidence yet. Next: math m1 (a150) — does the
  function-wall drop in a quantitative domain?
- 2026-07-10 ~21:10 crew 9 (math a150, FIRST MATH CELL) complete: **REJECT #8 — math does
  NOT bend the wall (this cell)**. Structural crew (.2676→.3385, contract 0/6→5/6, honest
  probe-2 TIE, integrity clean) killed on: gamed non-answer OUTSCORING genuine derivation
  (+.2175 laundering gap — repeated 'note that' clears the >=2-marker gate AND inflates the
  density it corroborates = gate/signal double-counting), vacuous-licensing fires, quantifier
  detector non-restrictive on 50% of real fires. DOMAIN READ: in math the wall shifts from
  lexicon-fragility to VACUITY (syntax fires without semantic work) — form parseable, form-
  PERFORMING still judgment. Caveat: a150 is mushy math; a36 proof-step labeling (near-pure
  form) is the sharper test. Tally 9 cells: 8R + 1 blocked. Next per interleave: humor a216
  (floor), then math a30.
- 2026-07-10 ~22:20 cell 10 (humor a216, FIRST FLOOR — planner-diagnostic + null-verifier
  path): **E2 NULL CONFIRMED with CONTRACT-BLINDNESS rider** — the run's biggest instrument
  finding since the Codex review. Code-channel null proven (algebraic: no reweighting fixes
  probe-0 INV; ~22 families swept null across both agents) BUT real ρ=.26 signal exists in
  min/avg(throughline,resolution) — L-CHANNEL ONLY, categorically invisible to contract_check
  (probes score extracted={}). RULING: frozen 0.00 = contract limitation, NOT tacitness;
  floors must be labeled CONTRACT-BLIND vs genuinely-null; contract-v2 needs probe-time field
  extraction before floors count as tacitness evidence. Bonus: 2 h0 bugs found (self-match
  evidence channel, rambling-rewarding flow term) + 1 probe-authoring artifact. Protocol
  rider added. Tally 10 cells: 8R + 1 blocked + 1 null-confirmed(contract-blind). Next:
  math a30 (m2).
- 2026-07-11 ~00:20 crew 11 (math a30, 2nd math cell) complete: **REJECT #9 — math kill
  species REPLICATES 2/2** (vacuity + double-counting, independent implementations). Crew
  fixed a real denominator bug + h0's 2-INV contract fail (.3137→.3714, integrity clean,
  honest TIE kept, 5-item residual self-disclosed) but: theorem-name credit stacks unbounded
  per citation-name (.522→.658, zero new content) and the '25-char vacuity guard' is a
  LENGTH gate not a CONTENT gate → computation dump (the boundary_notes-named failure)
  launders to +.44 over h0. LESSONS: per-fact caps; content-tests not thresholds. READ:
  threshold guards ≈ lexicon gates — neither holds a semantic boundary; compiled-pole
  hypothesis keeps losing at criterion level. Tally 11: 9R + 1 blocked + 1 contract-blind
  null. Next per interleave: humor a90 (floor 2), then math a60.
- 2026-07-11 ~01:30 cell 12 (humor a90, floor 2) complete: **CONTRACT-BLIND CONFIRMED, 2/2
  floors** — story_frame L-channel rho .417 (≈2× certified blend) invisible to the gate;
  code channel honest ceiling ~.30 and its best family construct-invalid (max-scores stock
  jokes). Verifier's gem: an honest narrow bug fix LITERALLY PASSES the harness contract by
  coincidental probe-space alignment while dropping train rho and crushing top-judge items
  (rank .978→.000) — constructive proof probe-pass ≠ validity even without gaming; floor
  claims now require train-rho-delta + per-class checks alongside probes. Contract-v2 fix
  (probe-time field extraction) empirically sanity-checked (flips to 5/5 SEP). Bug classes
  recur: rewards-rambling ×3, self-match retrieval ×3. READ: the floors' tacit-candidate set
  is EMPTY so far — at E2 the floors are instrument-limited, not tacitness evidence.
  Tally 12: 9R + 1 blocked + 2 contract-blind nulls. Next: math a60 (m3).
- 2026-07-11 ~02:40 cell 13 (math a60, m3 — verifier path) complete: **CONTRACT-BLIND-MID
  confirmed, a NEW SPECIES** — blindness isn't floor-only; the WS1a probe design under-covers
  L-heavy constructs panel-wide (h0 contract 1/6, probes TIE because code path silent; 7
  refutation families fail the triple test; unmodified-h0 + plausible probe-time fields flips
  probes = mechanism is harness-artifact). Salvage: pure 2-trigger removal .4935→.5712
  verified, downside confined to one bucket. Verifier also tightened the L-claim (length
  confound: .553→.480 partial, undisclosed by planner) and re-killed a planner-rejected idea
  more decisively (right probes, WRONG population sign — a90 lesson repeating). Tally 13:
  9R + 1 blocked + 2 CB-floors + 1 CB-mid. Next per interleave: PR a2 (control 1 — crews
  must NOT degrade; instrument check).
- 2026-07-11 ~03:40 cell 14 (PR a2, CONTROL 1) complete: **AT-CEILING, control intact** —
  h0 at 76.6% of judge-reliability ceiling; the one coherent residual class (unconditional
  L-boost crediting off-genre sources) ablates to +.003-.005 vs bootstrap SD .045 = noise;
  planner named and declined its own manufactured-strategy trap. Verifier agent SKIPPED
  (deviation, logged: conservative claim, no tacitness load; headline spot-checked locally —
  rho .6720 + 6/6 PASS reproduce). CALIBRATION READ: E2 machinery says honest nothing-to-do
  on the known-good case → the 9 kills are signal, not reflex. Tally 14: 9R + 1 blocked +
  3 CB + 1 at-ceiling control. Next: math a204 (m4).
- 2026-07-11 ~04:45 cell 15 (math a204, m4 — verifier path) complete: **PROBE-AUTHORING
  FLAW confirmed — 4th instrument-side species.** 5/6 probes differ ONLY by newline
  placement; broadened 4-way corpus scan = ZERO genuine wraps in 150 docs (stronger than
  planner's claim). Salvage best-of-run: 3 strategies incl. exp(-penalty) soft-combine w/
  formal pointwise non-regression → .5973→.6081, 0/149 regressions. Contract-v2 axes
  proposed from corpus-verified signals (\$-boundary punctuation 53%, LaTeX \\\\ breaks 25%,
  ellipsis pairs). Instrument-side taxonomy now: L-channel blindness (×3), probe-flaw (×1),
  uncoverable-clause (a25), vs the 9 function-wall kills. Tally 15: 9R + 1 blocked + 3 CB +
  1 probe-flaw + 1 at-ceiling. Next: PR a119 (control 2).
- 2026-07-11 ~05:45 cell 16 (PR a119, CONTROL 2) complete: **CONTROL INTACT + the run's
  cleanest salvage — self-match retrieval bug (4th occurrence) live on a CERTIFIED control**:
  sv<0.985 threshold misses self for 44% of items → docs scored as own near-duplicates.
  Fix (drop rank-0) planner-verified CI [+.0021,+.0190] excl. zero + main-loop replicated
  exactly (.8002→.8100, contract 6/6 unchanged; candidate built locally, no 2nd agent).
  Two tempting alternatives correctly rejected full-train (a2-precedent discipline).
  ESCALATION: fleet-wide h0 bug sweep added to WS6 (#14) — self-match ×4, rambling-reward
  ×3, denominators, inverted lexicons across all 32 h0s; fixes via held-out promotion batch.
  Tally 16: 9R + 1 blocked + 3 CB + 1 probe-flaw + 2 controls-intact (1 at-ceiling, 1
  bug-fix-queued). Next: math a36 (floor — the purest-form test).
- 2026-07-11 ~07:00 cell 17 (math a36, marquee floor) complete: **FIRST CLEAN ACCEPT OF THE
  LADDER — and on exactly the construct the compiled-pole hypothesis predicted.** Frozen 0.00
  was BUG+instrument, not tacitness: h0's rambling-lexicon inverted 2 probes and h0 itself
  scores content-free step-skeletons above genuine content. Fix = removal/reweight only
  (.2589→.3215 +24%, contract 3/5+2INV → 5/5 0INV); implementer caught its own new laundering
  path mid-flight (split credit- from exemption-evidence); adversary attacked full-force,
  measured the inherited Case-1 hole UNAMPLIFIED at mechanism level (byte-identical), added
  one non-decisive finding (question-numbering backref, ~.01, → E3 queue). LOCALIZATION
  RESULT: E2 compiles FORM-like constructs; cannot compile FUNCTION-boundary constructs
  (9 kills) — the wall is semantic-function-specific, not universal. Candidate → held-out
  promotion batch. Tally 17: 1 ACCEPT + 9R + 1 blocked + 3 CB + 1 probe-flaw + 2 controls.
  Next: humor a351 (control 3), then math a222, a198.
- 2026-07-11 ~08:00 cell 18 (humor a351, CONTROL 3) complete: **CONTROL INTACT, AT-CEILING
  (contract-blind — 4th L-channel case: all 6 probes TIE, L-alone .7189 > blend .6971 i.e.
  code channel net-negative)**. All 4 recurring bug classes ABSENT; a 5th pattern FOUND but
  declined per evidence bar (substring collisions spic→spice etc., CI touches zero — added
  to WS6.5 sweep checklist); a +.0247 gate-drop REJECTED on construct grounds (punching-up
  satire boundary). Controls 3/3 clean. Headline spot-checked locally (no 2nd agent).
  Tally 18: 1 ACCEPT + 9R + 1 blocked + 4 CB + 1 probe-flaw + 3 controls-intact.
  Next: math a222 (floor), then a198 (control) — final two.
- 2026-07-11 ~09:10 cell 19 (math a222, floor 2 of math) complete: **PROBE-FLAW confirmed,
  non-discriminating-axis subspecies, HARDENED** — 7/7 independent codings of the probe-2/5
  axis null/wrong-signed vs judge; NEW structural close: text extraction destroys layout
  (0/408 display spans retain newlines) → the probes' axis is UNREPRESENTABLE in this corpus.
  Salvage beats plan (.0059→.1386 via 3 wrong-signed-penalty removals + mention-vs-use
  html_penalty bug fix, 0/149 regressions, 18/18 self-tests; still sub-significance p=.092)
  BUT code channel fails TWO gates (dropping the dominant wrong-signed penalty newly fails
  discrimination frac_at_mode .88) — 'no viable code path' was an UNDERstatement. One
  non-load-bearing table slip found in plan (train vs full-pool judge mean). NOT contract-
  blind (exhaustive L-sweep), not genuinely-null (fixable bugs + weak residual). Tally 19:
  1 ACCEPT + 9R + 1 blocked + 4 CB + 2 probe-flaw + 3 controls. FINAL CELL next: math a198
  (control 4) → then the full E2 read package (curve verdicts, E3 go/no-go, contract-v2).
- 2026-07-11 ~10:10 cell 20 (math a198, CONTROL 4) complete: **CONTROL INTACT, AT-CEILING
  (contract-blind, a351 pattern — 5th L-channel case: cited_source ext-alone .622 > blend
  .589 > code-only .141)**. All 5 bug classes absent; 6th-pattern candidate ablated 4 ways,
  every CI spans zero → REJECTED at the a119 bar; residuals = correct exclusions + judge
  noise (rel .564, panel lowest). Headline spot-checked (.5887, FAIL 2/6 reproduce).
  **E2 SWEEP COMPLETE: 20/20 cells — 1 ACCEPT (math a36) + 9 function-wall rejects +
  1 blocked accept + 5 contract-blind + 2 probe-flaw + 4 controls-intact (2 with queued
  bug-fix salvage). Zero genuine tacit candidates found at E2. E2 STAGE READ due next.**
- 2026-07-11 ~10:25 **HELD-OUT PROMOTION BATCH 1 (pre-registered): 0 PROMOTIONS**
  (e2_promotion_batch1.json). math a36: BOTH programs ~zero on test (h0 −.0343, cand +.0010,
  P .667<.80) — the train-side accept was real and attack-proof but does NOT transfer; the
  frozen test-side 0.00 was accurate; a36 curve verdict = FLAT-at-zero held-out, train accept
  = instructive train/test-gap artifact at a floor criterion. PR a119: wash (.8405 vs .8400,
  P .451) — bug real, test effect nil; control non-degradation confirmed held-out. READ:
  E2's honest bottom line = train-side improvements are achievable and even construct-valid,
  but NOTHING transferred at promotion strength — replicates the fleet's 0/10 at the ladder's
  strongest candidate. The held-out discipline is the load-bearing instrument. E2 ends:
  0 promotions, 0 tacit candidates, 2 instrument-defect classes with validated fixes, 1
  train-only accept, 4 clean controls.

---

## E2 STAGE READ (2026-07-11 — sweep complete, 20/20 cells + held-out batch 1)

| criterion | h0 train ρ | cand ρ | cell outcome | held-out |
|---|---|---|---|---|
| PR a41 | .4502 | .4730 | REJECT (probe-gaming mention detectors) | — |
| humor a342 | .3532 | .3960 | REJECT (single-marker mention firing) | — |
| PR a31 | .6629 | .6625 | REJECT (attribution error-swap) | — |
| humor a189 | .2597 | .3921 | REJECT (low-recall/high-weight + false log claim) | — |
| PR a111 | .5332 | .5838 | ACCEPT→BLOCKED (Codex: laundering inversion) | — |
| humor a117 | .4393 | .5340 | REJECT (repetition-function not Jaccard-able) | — |
| PR a25 | .3921 | .4462 | REJECT (zero-recall probe-flippers; 2 sound salvages) | — |
| humor a315 | .3638 | .6042 | REJECT (lexicon gate paraphrase-laundered) | — |
| math a150 | .2676 | .3385 | REJECT (vacuity + gate/signal double-count) | — |
| humor a216 | .1952 | — | CONTRACT-BLIND floor (L-signal ρ.26 invisible) | — |
| math a30 | .3137 | .3714 | REJECT (vacuity: length-gate + per-name stacking) | — |
| humor a90 | .2521 | — | CONTRACT-BLIND floor (story_frame ρ.417 L-only) | — |
| math a60 | .4935 | — | CONTRACT-BLIND MID (lead_pattern L-only; salvage +.077) | — |
| PR a2 | .6720 | — | CONTROL: AT-CEILING (honest nothing-to-do) | — |
| math a204 | .5973 | — | PROBE-FLAW (corpus-absent \n axis; salvage 0/149 regr) | — |
| PR a119 | .8002 | .8100 | CONTROL: intact + self-match bug fix | h0 STAYS (wash) |
| math a36 | .2589 | .3215 | ACCEPT (train-side, full adversary survive) | **h0 STAYS — DENIED (both ~0 test)** |
| humor a351 | .6971 | — | CONTROL: AT-CEILING (contract-blind; code net-negative) | — |
| math a222 | .0059 | — | PROBE-FLAW (non-discriminating axis; unrepresentable) | — |
| math a198 | .5887 | — | CONTROL: AT-CEILING (contract-blind, a351 pattern) | — |

**Headline numbers:** 0 promotions / 0 genuine tacit candidates / 9 function-wall kills (one
species) / 5 contract-blind + 2 probe-flaw (instrument defects, fixes validated) / 4 controls
intact / crew integrity spotless post-a189 / every train gain +5..+66% rel, zero transferred.

**Decisions queued for user:** (1) contract-v2 rebuild BEFORE E3 — probe-time field extraction
(sanity-checked ×2), corpus-present re-authored axes (a204/a222), L-aware probes, probe-artifact
audit; (2) WS6.5 fleet h0-bug sweep (5 classes); (3) E3 scope: the 9 function-wall cells only;
(4) WS2/WS4/WS5 sequencing with known budgets (~3 agents/cell, ~1 cell/hr).

- 2026-07-11 ~11:15 wake (post-sweep hold): E2 STAGE READ table appended above; WS6.5 grep
  stage done locally (0 agents): 22 h0 programs call retrieve_similar; threshold-style
  self-exclusion anti-pattern CONFIRMED in 3 files incl. NEW hit cw/a297 (sv<0.98);
  inventory + audit queue → ws65_bug_sweep_inventory.json. Ladder holds at the decision
  point; no crews launched.
- 2026-07-11 ~12:30 **WS4 CELL 1 COMPLETE (PR a119 DAG)** — the workflow rung is live and
  its first readouts are clean: 28-node typed DAG, equivalence BIT-EXACT first attempt
  (rho .8100 identical, max Δscore 0.000000, contract 6/6 matches; spot-checked by main
  loop), seam_frontier = {doc_kind L2, new_fact L1} exactly as planned → the certified
  criterion is NEARLY-COMPILED: judgment enters at 2 thin, LOW-abstraction points and the
  assess layer above is pure code. TWO structural findings: (i) the d00602 precedence
  inversion is now an explicit node trace (2-signal boost max()-overriding an 8-family
  assess — invisible to level checks); (ii) INSTRUMENT: median-ablation marginals confound
  importance with sink-proximity → cells 2+ use own-transformation ablation. Next WS4 cells
  (a28, a76, then patents on M̄(x,Z)) can run 1/wake alongside the ladder decision hold.
- 2026-07-11 ~13:30 **WS4 CELL 2 COMPLETE (PR a28)**: bit-exact equivalence round 1 (rho
  .7002 identical, 0.000000 max Δ incl. empty-text edge); frontier = {doc_type L2,
  primary_source L1} — **SAME nearly-compiled seam signature as a119 (2/2), different
  architecture**. Marginals (own-transformation, cell-1 lesson applied) surface a real
  discovery: release_gate NET-HARMFUL (+.0695 if removed; evidence channel alone recovers
  .6867/.7002) → improvement shelf. Substring bug CONFIRMED via synthetics (train-null,
  preserved). Registration corrected (3 ops calls, no retrieve_similar). WS4 running total:
  2 cells, 4 latent defects surfaced, 0 behavior changes. Next: PR a76, then patents cells.
- 2026-07-11 ~[user decisions landed]: **(1) contract-v2 rebuild: YES (before E3); (2) E3:
  FULL PANEL; (3) WS6.5 fleet bug sweep: YES; (4) WS2/WS5: YES, budgets may be DOUBLED.
  Plus: E2 PANEL EXPANSION ordered — more math + coding + peer review ("20 metrics is not
  enough"); pacing relaxed ("spin up a few more subagents"); GPUs may be freer.**
  Extension FROZEN (panel_extension_v2.json, same rule, cam-sha stamped): peer_review 8
  cells (a100/a114 ctl, a29/a172/a14/a128 mid, a163/a214 floor — task is fully fleet-
  instrumented, 20 h0s + judge + fields on disk) + math 8 (mids a234/a180/a24/a174/a228/a72,
  floors a0/a84). CODING = new-build lane from datasets/code-review/crse_balanced_v2 (items
  → judge GPU pass → h0 fleet → contracts; longest pole). sk3 currently in a NAT64 reset
  spell — GPU staging queues locally, pushes when it recovers.
- 2026-07-11 ~[afternoon] **WS4 CELL 3 COMPLETE (PR a76) — CERTIFIED-PR TRIPLE CLOSED, all
  3 bit-exact.** Topology fingerprint holds 3/3 (2 thin L-nodes, pure-code assess above; the
  finer L1+L2 split is 2/3 — a76's code does its own gating, both L-nodes are L1 spans). BUT
  the pilot's first theory-refining surprise: a76's WEIGHT is L-side — lede_news ablation
  −.2264 (17× any code node; code-only pipeline .5781 vs intact .8045), contradicting the
  plan's code-dominant hypothesis. READ: frontier topology and channel weight are TWO
  independent seam coordinates — same shape, different depth (a119/a28 code-heavy, a76
  L-heavy). Also: another NEW substring bug ('overstated'→'grounded', synthetic-confirmed,
  → WS6.5 census) + the soft-override quantified (max +.221 vs a119's hard .594). WS4 next:
  patents cells a26/a34/a35 on M̄(x,Z). CONCURRENT LANES NOW: bug-sweep census, coding
  stage-1, 8 peer-review v2 contract authors (wave 1; math wave 2 queued).
- 2026-07-11 ~[expansion sprint] **EXTENSION CONTRACTS 16/16 FROZEN (v2 standard)** —
  peer_review wave (8) + math wave (8) authored, validated (16/16 structural incl. v2 fields:
  corpus_phenomenon + ≥1 mention_only + channel tags on every probe), 11 keyword flags all
  adjudicated benign (ML content vocabulary), sha-stamped in contracts_v2_validation.json.
  Quality notes: authors regex-scanned the corpus before authoring (a228), floors declared
  inherently-L (a0 5/6, a84 4/5, a163 5/6 L-channel) — the blindness lesson operating at
  authoring time. CODING STAGE-1 DONE: 250 stable-hash PR-diff items (86/14 merge base rate),
  bank audit independently reproduced (58 alive/127), 18 candidates, 9,250 judge prompts;
  repo confound (rdk+spire 56%) flagged, repo kept for grouped CV (BUILD_PLAN.md). sk3 still
  in reset spell (3 attempts) → local recovery watcher armed (checks /3min; on recovery:
  rsync coding prompts + arm codejudge gpu_waiter automatically). E2 extension cells can
  start next wake: peer_review + math interleave, one crew per wake, protocol v2, using
  contracts_v2. Bug census still running.
- 2026-07-11 ~[census lands] **WS6.5 BUG CENSUS COMPLETE** (ws65_bug_census.json): 163 h0
  programs audited, **85 confirmed bugs** — substring collisions DOMINANT (38; bare `word in
  text`, e.g. 'mom' in 'moment', 'son' in 'prison'), self-match 14, rambling-reward 14
  (largest single swing: math a30 +.566 — the class its E2 crew independently found),
  denominator 11, inverted-lexicon 8. **28/48 panel+extension criteria carry ≥1 confirmed
  bug; only 12/48 fully clean; several controls affected** (PR a119 known; census labels
  humor a90 a 'control r_hyb .755' — DISCREPANCY vs panel_freeze where a90 is a floor at
  0.00; role label needs re-check, findings stand). Worst: humor a90 (3 stacked: dead
  self-match channel + rambling +.258 + family-term substrings). SEQUENCING DECISION: h0s
  stay FROZEN mid-ladder (comparability); all fixes = candidates for the post-sweep held-out
  promotion batch (a119 precedent: real bug ≠ held-out gain). Protocol updated: planners
  consult the census before diagnosing. E2-ext cell 1 (peer a29) planner in flight.
  graph in hand), resumed once (~09:40) but still silent + cell dir empty after ~70 min —
  second nudge sent ('write plan.md with what you have'). The message was QUEUED (agent
  possibly still in a long tool round). If no completion by next wake: write off and
  relaunch fresh (partial design is recoverable from its transcript if needed). No other
  lanes need attention; ration held.
- 2026-07-11 [peer genre fix + E2L registered] **PEER-REVIEW GENRE ERROR (mine) FOUND+FIXED.**
  a29 verifier's corpus spot-check caught all 8 peer v2 contracts wrote probes as
  reviewer-commentary (0/250 corpus hits); peer items are the REVIEWED PAPERS (abstract text,
  accept/reject), criteria = paper quality properties. My authoring prompts said 'peer reviews
  of ML papers'. Wrong 8 → contracts_v2_peer_GENREBUG/; all 8 RE-AUTHORED correct-genre +
  CONTRACT-v2.1 (corpus_phenomenon grep-VERIFIED vs real ctext, counts+doc IDs). 8/8 PASS,
  re-frozen. Corrected authors caught what the wrong version invented: 'model card' 0/250,
  own-license 0/250, 'limitation' 28× but 0 self-directed, dropout/population ML-sense
  collisions. a29 cell PROVISIONAL (re-run on corrected contract; genre-independent
  Strategy-1 +.10 salvage survives). Math 8 unaffected.
- 2026-07-11 **E2L ARM REGISTERED (user-directed):** the 9 function-wall kills were all
  REGEX-genre code → re-run those 9 cells with a frozen capability library
  (ops_capability.py: spaCy dep-parse attribution, SymPy entailment, statcheck recompute,
  date arithmetic, sentence-graph structure — all libs verified installed). Prediction both
  ways: kills persist = wall genuinely judgment-layer; accepts = E2 wall was toolbox-bounded,
  E-axis needs capability dimension. Library being built + smoke-tested now.
- 2026-07-11 [wake] E2L cell 1 (a31 attribution) IN FLIGHT — candidate.py built, spaCy
  dep-parse grinding 150 train items × rounds + self-adversary (compute-heavy, slow). Held:
  no 2nd E2L cell launched — a31's verdict shapes the a150 brief (is the crux the dominance
  swap-test or op adequacy?). sk3 still NAT64-down (watcher pid 54785 gently retrying /3min,
  per reference_sk3_nat64 guidance); coding GPU pass + contract-v2 probe-extraction auto-queue
  on recovery. No other lane advanceable without a31 verdict or sk3. Ration held.
- 2026-07-11 **E2L CELL 1 (a31 attribution) = THE REFRAME.** Capability lib (spaCy dep-parse)
  vs the regex kill: attack surface 43/142→9/142 (5x smaller), swap 23f/20b→5f/4b (net-positive,
  not clean dominance), decisive d00964 self-quote HELD .000 (regex leaked .303). Verdict
  PARTIAL/REJECT-informative: 4 items still leak, residual MOVED from string-fragility to
  COREFERENCE (bare-title/bare-surname self-quotes attributions() can't resolve). **READ: the
  function wall is CAPABILITY-GRADED, not binary — regex→dep-parse receded it 5x; the residual
  NAMES the next capability tier. Wall height = maturity of the needed capability.** This is the
  strong reframe the user predicted. Next: E2L a150 (math vacuity via SymPy — a DIFFERENT,
  more MATURE capability; tests whether recede-pattern replicates & whether a mature capability
  fully closes on its checkable subset). Coref tier (E2L-v2) invested only if pattern holds.
- 2026-07-11 **E2L CELL 2 (a150 vacuity) = THE REFRAME DOES NOT REPLICATE — SCOPE-MISMATCH,
  NOT MATURITY-GAP.** SymPy licensing_does_work()/parse_math() replaces the killed length/
  tech-token-presence licensing gate (4 rounds: K1 symbolic-verify + circular/bare-atom/short-
  English-word noise guards, K2 per-fact-capped density2 + disjoint hyp_fix gate, K3 \in
  bugfix). Train rho .2676→.4136 (+.146, LARGEST math-cell gain in the whole ladder). Laundering
  gap CLOSED on both verbatim and $-delimited reproductions (+0.2175/fails-bar → +0.0786/clear).
  Vacuity sweep clean at the MECHANISM level (0/5 vacuous get credit incl. a circular-restatement
  guard catch, genuine credited, wrong penalized) but END-TO-END confounded by h0's untouched
  base-density connective mechanism (disclosed, not closed). Dominance 28 fixed/44 broken (swap,
  not dominance — same pattern as every other cell). Contract REGRESSED 5/6→1/6. **THE KEY
  DATUM: checkable-subset coverage on real train = 0/20 (0%)** — audited: 4/20 pre-guard
  "checkable" hits were ALL noise (bare-symbol mismatches + a SymPy edge case misparsing 2-letter
  "is"→i*s). Root cause: math/a150's real licensing clauses are "condition licenses an
  OPERATION/theorem-application" (nonzero licenses division, continuity licenses IVT); SymPy's 4
  checkable cases verify "equation A rescales to equation B" — a DIFFERENT relation that merely
  shares the word "licensing". **READ: a31's recede-pattern (capability-graded wall, residual
  names the next tier) does NOT replicate. a150's SymPy component is clean where it fires but
  almost NEVER fires (0% vs a31's meaningful-if-incomplete coverage) — not because it's immature
  (it's arguably the MOST mature capability in the library) but because it targets the wrong
  mathematical relation for this construct. The train-rho gain is almost entirely non-SymPy
  (per-fact caps + disjoint gating, generic anti-gaming discipline any implementer could apply).
  REVISED THEORY: capability-gradedness requires the tool's checkable domain and the construct's
  content to be the SAME relation at different maturity levels (a31: both are "who said this",
  dep-parse vs coreference); when they're different relations entirely (a150), maturity isn't
  the axis — scope-match is. Full detail: outputs/metric_seam_pilot/battery/effort_ladder/e2l/
  math__a150/{meta.json,adversary.md}.
- 2026-07-11 **E2L CELL 2 (a150 vacuity/SymPy) SHARPENS THE REFRAME.** rho .2676→.4136
  (+.146, largest math gain) but contract 5/6→1/6 (honest abstain refuses NL-prose probes),
  dominance 28f/44b SWAP, laundering gap CLOSED (.2175→.0786) BUT checkable coverage 0/20
  (0%). Verdict RESISTS-MORE-than-a31 via SCOPE-MISMATCH not immaturity: SymPy verifies
  'equation A rescales to B'; a150 licensing is 'condition licenses operation' (nonzero→÷,
  continuity→IVT) — DIFFERENT RELATION, so the capability never engages (gain is non-SymPy
  bookkeeping). **REFINED THEORY (2 cells): the wall recedes ONLY when a capability's RELATION
  matches the construct's relation. Matched+immature (a31: attribution, coref tier) → recedes
  to maturity ceiling. Relation-MISMATCH (a150) → capability doesn't engage, wall stands
  (though a surface sub-attack can still close). Two axes: relation-match THEN maturity.**
  Next: E2L a111 (substantiation via NER/entities_with_evidence — a 3rd relation type, existing
  frozen lib) to add a relation-match datapoint. Coref tier (E2L-v2, a31 re-run) = registered
  next investment but needs an install decision (fastcoref/coreferee) → defer to user.
- 2026-07-11 **E2L CELL 3 (a111 CTA-diffusion) = 3rd framework point: PROXY-MATCH.**
  sentence_graph() content-word-Jaccard clustering is a PROXY (not exact) for the construct's
  destination-clustering relation. rho .5332→.5931 (+.060, exceeds even the blocked E2
  candidate .5838), contract 6/6 (probe-2 diffusion-boundary SEP throughout), dominance 56f/21b
  swap; **RECEDES AND CLOSES the exact Codex laundering defect that blocked the E2 accept** —
  diffusion multiplier fires 15/150 (locate-coverage 149/150; a150 LOCATE-vs-JUDGE distinction
  applied). → candidate for held-out promotion batch (fixes the block, still train-side).
  adversary.md not written (session-limit) but data in meta.json + self_adversary_results.json.
  **THREE-CELL E2L TAXONOMY (relation-match axis × maturity axis):
   • EXACT match + immature capability (a31 attribution/dep-parse) → recedes 5x, residual NAMES
     next tier (coreference = harder instance of SAME relation).
   • RELATION MISMATCH (a150 vacuity: SymPy checks equation-rewrite, construct needs
     condition→operation) → capability never engages; wall stands; only surface sub-attack closes.
   • PROXY match (a111 CTA: Jaccard-cluster ≈ destination-cluster) → recedes + closes the
     specific defect, still a swap not clean dominance.
  HEADLINE: the function wall is not binary — it recedes iff a capability's RELATION matches the
  construct's, to that capability's MATURITY ceiling. "Regex can't" was "no matched capability
  in the toolbox." This is the user-predicted reframe, now with 3 relation-types.**
  Next: E2L a117 (repetition-FUNCTION via sentence_graph/is_refrain) = 4th relation type +
  the pre-registered WEAKEST-COVERAGE limit test (aesthetic judgment — does ANY capability match?).
- 2026-07-11 **E2L CELL 4 (a117 repetition-function) = THE WEAKEST-COVERAGE LIMIT TEST
  RECEDES, NOT THE FLOOR.** is_refrain() was purpose-built by reading this cell's own
  decisive_reason (module docstring names humor/a117 verbatim) — the tightest a priori
  relation-match of any E2L cell. Both named kill mechanisms verified FIXED on live train +
  reconstructed minimal pairs: frame-dominated Jaccard misfire (rule-of-three probe gap-to-h0
  0.086->0.024) and colon/script-format blindness (live d01575: killed-e2 wrong-direction
  0.514 vs h0 0.583 -> candidate 0.584); tail-excision punchline-deletion also fixed via
  discourse_position()+punchline-field cross-check (killed-e2 0.868 above h0's 0.735 ->
  candidate 0.749), laundering invariant re-verified exact. rho .4393->.5069 (+15.4%),
  contract 5/5 PASS. Self-adversary found 2 disclosed residuals, both traced to ONE design
  property (varied_final is a strict word-set-inequality, not a meaningful-escalation check):
  a non-live exact-verbatim-callback misread (0/6 padding groups on train start with a quote
  mark — confirmed absent, not asserted) and a live-but-SHARED title-echo false-positive
  (2/8, also missed by killed-e2 for its own reason). Dominance swap (6 fixed/43 broken) but
  41/43 broken INHERITED from the pre-existing e2-round-3 base (isolated via a 3-way
  killed-e2/h0/candidate comparison) — this cell's own marginal contribution is 2 small new
  regressions. Relation-coverage healthy (20% locate/4% judge-fires, 13/13 hand-spot-checked
  correct). **FRAMEWORK PLACEMENT: same pattern class as a31 (EXACT-leaning match + immature
  -> recedes, residual NAMES next tier within the SAME relation), the 2nd clean instance of
  'purpose-built capability -> exact-leaning+immature.' META-QUESTION ANSWER: NOT the floor —
  even this pre-registered hardest case (aesthetic judgment) found a workable near-exact match
  once a capability was purpose-built to its decisive failure; the honest floor candidate is
  further out, at FULLY HOLISTIC funniness/timing judgment, which every a117 program (h0
  included) has ALWAYS delegated to LLM_FIELDS, never code, at any maturity.** Full detail:
  outputs/metric_seam_pilot/battery/effort_ladder/e2l/humor__a117/{meta.json,adversary.md}.
- 2026-07-11 **E2L CELL 4 (a117 repetition-function) = 4th point + FLOOR IDENTIFIED.**
  Pre-registered HARDEST case (aesthetic judgment) but is_refrain() was purpose-built for it →
  EXACT-LEANING match. rho .4393→.5069 (+15.4%), contract 5/5; escalation-vs-padding kill FIXED
  (d01575 colon/script refrain, laundering invariant exact); dominance raw swap 6f/43b but 41/43
  broken PRE-EXISTING (inherited), marginal = 2 new small regressions. Pattern = a31's (exact +
  immature → recede, residual names next tier). **NOT the floor — even the hardest pre-reg case
  found a workable match once the op was purpose-built. The TRUE floor = fully holistic
  funniness/comic-timing, which EVERY a117 program incl. h0 delegates to LLM fields, never code,
  at any maturity.**
  ---
  **E2L 4-CELL TAXONOMY COMPLETE (all 4 relation-types the frozen lib covers):**
   a31 attribution (dep-parse) = EXACT+immature → recede, residual=coreference
   a150 vacuity (SymPy)        = MISMATCH → stands (surface sub-attack only)
   a111 CTA (sentence-graph)   = PROXY → recede+close Codex defect (promotion candidate)
   a117 repetition (is_refrain)= EXACT-leaning+immature → recede, residual=holistic-funniness
  UNIFIED THEORY: wall recedes IFF capability RELATION matches, to MATURITY ceiling; the
  irreducible FLOOR = holistic aesthetic/semantic judgment, which is L-ONLY — exactly what the
  base sweep's 5 CONTRACT-BLIND cells showed lives in the LLM channel. E2 sweep + E2L + floor +
  contract-blind = ONE coherent finding.
  HOLD: remaining killed cells (a30/a41/a25/a189/a315/a342) fall into the 4 relation-families
  already mapped → would REPLICATE not extend; autonomous thread at natural consolidation point.
  USER DECISIONS pending: (1) E2L-v2 coref tier (a31 mature-capability test, needs install);
  (2) E3 full-panel launch; (3) WS2/WS5 sequencing. sk3 still down (watcher live); GPU lanes
  (coding judge, contract-v2 extraction) auto-resume on recovery.

---
## PROGRAM REFACTOR v2 (2026-07-12, user-directed: "merge the sprawl; 20-30 metrics PER task;
## go with what is working")

**TIER RENAME (docs + paper; old E-codes in parentheses):** draft(E0) / solo(E1) / team(E2) /
team+feedback(E3) / team+architect(E4) / **team+tools(E2L)**. Team+tools is now the PRIMARY
protocol — it produced the program's FIRST HELD-OUT PROMOTION.

**FIRST PROMOTION:** press_releases/a31 team+tools candidate (dep-parse attribution) —
held-out P(cand>h0)=.910, d=+.0145 (e2l_promotion_batch.json). PROMOTION-QUALIFIED per the
pre-registered rule; physical head-swap deferred to post-sweep batch (frozen-h0 comparability
rule). a111 train gain did NOT transfer (−.034, P=.17); a117 wash — consistent with the
train/test-gap lesson; 1-of-3 promotes and it is the exact-relation-match cell.

**LANE STRUCTURE (absorbs the old WS-sprawl):**
- **LANE A — Compilability census (primary; absorbs WS1+E2L+expansion):** panel v3 = CENSUS,
  159 criteria / 7 tasks (panel_v3_census.json; coding joins on judge pass). Protocol = solo
  team+tools cells (1 agent: plan+implement+self-adversary, the E2L pattern — 3x cheaper than
  3-agent crews and it produced the promotion). Per-cell readout: relation-match verdict +
  train delta + kill-battery + held-out batch per task-wave. team-only(E2) contrast on a 10%
  subsample; team+feedback/architect (E3/E4) DEMOTED to a small subset, run only if the tools
  axis leaves unexplained residual (user-approved full-panel E3 superseded by this refactor —
  the tools axis is the interesting one).
- **LANE B — Instruments:** contracts v2.1 authoring for the 115 new criteria (corpus-grep
  verification mandatory) + probe-time field extraction (sk3-gated) + bug-census fixes via
  held-out batches + a204/a222 probe re-authoring. Absorbs #15 + WS6.5.
- **LANE C — Seam structure (WS4):** 6 remaining DAG cells (patents on M̄(x,Z) = the
  different-seam prediction; then legal chains). Unchanged.
- **PARKED:** WS2 (tool-accumulation — E2L's frozen-library design partially answers it;
  revisit after census), WS5 (shared schema), WS6.2-6.4. Rationale: replication or
  lower-information relative to the census.

**CENSUS EXECUTION PLAN:** wave 1 = pack-building for all 115 (mechanical, 1 agent);
wave 2+ = contract authoring in batches of 8-10 (v2.1: corpus-grep verified) + central
validation per batch; wave 3+ = team+tools cells, 6-10/day at relaxed pacing, task-by-task
(finish a task → run its held-out batch → next task). Order: creative_writing (new domain,
26) → humor (23) → math (17) → PR (14) → legal (14) → ss_disability (13) → peer (8) →
coding (on GPU-pass). Held-out discipline unchanged: NOTHING is a result until its batch runs.
- 2026-07-12 [wake] Census wave 1 (115 packs) in flight — 0 packs on disk yet (agent in
  reconciliation/genre-note phase; not stalled-evidence yet, next wake is the deadline).
  sk3 still NAT64-down, watcher alive. All lanes correctly gated (contracts←packs,
  cells←contracts, GPU←sk3). Ration held.
- 2026-07-12 **CW CONTRACT DOMAIN FROZEN: 26/26 v2.1** (3 batches, all corpus-grep-verified,
  3 benign flags adjudicated). Authoring quality at scale: sibling-lattice coordination
  emerged (authors read existing contracts and steer to uncovered axes, bidirectional
  boundary agreements a45↔a207); representability limits flagged not papered-over (a333
  covers/layout in a text corpus); one self-caught miscount (a9). Census cells begin:
  ascending r_hyb order, solo team+tools protocol, held-out batch at domain completion.
- 2026-07-12 [wake] **sk3 RECOVERED → automated cascade fired flawlessly**: watcher handoff
  17:49 → codejudge waiter caught a free card → CODING JUDGE PASS DONE (rc=0). 3 GPUs free.
  Results pulled + deduped: file pre-existed from July-2-era build → 29,500 unique rows = old
  run (different dpids/aspects; carries the NA mass + mode-collapsed aspects) + our new 9,250;
  clean separation by crb-prefixed dpids. Launched: coding STAGE-2 reliability agent (scoped
  to new run, rel1≥.30 qualification, NA-adjusted) + humor contract batch 1 (8 authors:
  a0/a126/a135/a144/a153/a162/a18/a180). CW a54 census cell still grinding. Pipeline now
  runs 3 lanes concurrently: CW cells + humor contracts + coding qualification.
- 2026-07-12 **CODING JUDGE-QUALIFIED: 18/18 aspects clear rel1≥.30 (.496-.880)** — diff-based
  redesign fixed the comments-only-era degeneracy completely. Flags: a400 near-degenerate
  (99.2% mode; technically-qualified ≠ graded target); judge-NA ≠ checker-NA (2% vs 93% on
  a20 — different mechanisms, don't conflate); repo confound (rdk+spire 56%) → repo-grouped
  CV at eval. Scope: 249/250 items valid. NEXT coding gate: stage-3 h0 baselines (biggest
  remaining build; after humor contracts). reliability_report.json + results_newrun.jsonl.
- 2026-07-12 [incidental instrument finding] humor a198's author grep-verified that v1
  contract humor__a342's SIX probe genres (recipe/ToS/trailer/dating-bio/placard/warning-
  label) are ALL corpus-absent — the a204 probe-flaw species in a frozen v1 contract.
  The a342 E2 kill STANDS (adversary used own constructions, not the probes) but the
  contract → Lane-B retrofit queue for corpus-present re-authoring. v2.1 sibling-reading
  is now incidentally auditing the v1 fleet.
- 2026-07-12 **USER: "no new agents for now" — LAUNCH FREEZE.** In flight (completing
  naturally): humor batch-3 authors a36/a45/a54/a63/a72 + the CW a54 census cell. PAUSED:
  humor a81/a9, all remaining contract batches (math/PR/legal/ssdis/peer), census cells
  beyond CW a54, coding stage-3. State at freeze: contracts frozen = CW 26/26 + humor 16/23
  (+5 in flight) + extension 16 + v1 32; coding judge-qualified 18/18; 1 held-out promotion
  (a31). Hourly cron: status-and-log only, NO launches, until user lifts the freeze.
- 2026-07-12 **PROVENANCE INCIDENT (documented, PROVENANCE_INCIDENT_2026-07-12.json):** the
  CW a54 census cell self-authored a contract and OVERWROTE the frozen v1 file (content lost);
  root causes: my cell prompt assumed v3 existence (a54 was v1-covered — census queue didn't
  cross-check), my v2 validator rewrite had dropped the sha ledger, agent improvised instead
  of stopping. Containment: incident file + rebuilt sha ledger + STOP-AND-FLAG rule for
  missing contracts + a54 cell marked PROVISIONAL. Cell's science (first construct found to
  DECOMPOSE: mismatched core relation + rare exact sub-relation via NER; 3 removal-genre
  fixes +6.9% train; self-adversary caught a doc-level NER laundering hole and fixed it
  span-local) stands as train-side observation pending contract re-validation. LAUNCH FREEZE
  remains in force; user decisions pending: a54 contract accept-vs-reauthor, cell re-check.
- 2026-07-12 [freeze-state ledger] All pre-freeze in-flight agents COMPLETE. Contracts:
  CW 26/26 frozen + humor 21/23 written (16 frozen, 5 batch-3 validated locally; a81+a9
  paused) + extension 16 frozen + v1 32 (one file replaced per the incident). Coding:
  judge-qualified 18/18, stage-3 paused. Census cells: 1 run (CW a54, PROVISIONAL per
  incident). Promotions: 1 (a31). ZERO agents running. Awaiting user: freeze lift, a54
  contract decision, remaining queues.
- 2026-07-12 [wake, FREEZE] Status-and-log only per user's launch freeze. Zero agents
  running; all pre-freeze work landed and validated; sk3 up, idle for our lanes. State
  unchanged from the freeze-state ledger. Awaiting user: freeze lift + a54 incident
  decisions.
- 2026-07-12 [wake, FREEZE] Quiet: 0 agents, sk3 up/idle, state unchanged. Holding.
- 2026-07-12 [2 wakes, FREEZE] Quiet: 0 agents, sk3 up/idle, state unchanged. Holding.
- 2026-07-12 **USER: "continue" — FREEZE LIFTED.** Resume plan: (1) a54 incident → conservative
  default: INDEPENDENT validation of the self-authored contract (v2.1 gates + corpus re-grep +
  self-dealing check: do probes conveniently match the candidate's own new guards?); accept
  only if it survives, else re-author + cell re-check; (2) humor a81+a9 (completes humor 23/23);
  (3) census cell 2 = CW a333 (contract exists in v3 ✓, stop-and-flag rule active); (4) then
  math contract batches. Moderate concurrency (4 agents).
- 2026-07-12 [wake] 4 agents in flight (a54 blind re-author, humor a81+a9, CW a333 census
  cell); a54 self-authored contract REJECTED by independent validation (self-dealing gate:
  probes minted from the fix's own bug-hunt; mechanics all clean) → re-author running.
  Pipeline saturated at resumed pace; nothing to advance this wake.
- 2026-07-12 **HUMOR CONTRACT DOMAIN FROZEN: 23/23 v2.1** (3 batches; anchor-deconfliction
  across a 28-contract lattice held; corpus-thin clauses honestly disclosed rather than
  forced, e.g. a9's TW/CW clause 0/250). Two census domains now fully contracted (CW 26,
  humor 23). In flight: a54 blind re-author, CW a333 census cell.
- 2026-07-12 **CENSUS CELL 2 (CW a333, the CODE-tagged calibration cell): largest
  non-confounded gain of the census** — .2964→.4584 (+54.7% rel), contract 6/6 throughout,
  dominance 17f/2b (8.5:1). THEORY MATURED: one construct = a GRADED LADDER of 3
  sub-relations — device PRESENCE (CODE-native), device POSITION (EXACT match to
  discourse_position(), load-bearing on 6/6 live links), device FUNCTION ("serves the story"
  — MISMATCH, irreducibly L). 3rd construct family confirming function-stays-L. Fixes: 2
  census bugs + a corpus-verified genre-mislabel gate (26/150 items) + 2 new corroborated
  detectors; self-adversary caught its own bold-header detector's repetition-gaming hole
  (8× contentless heading) and shipped the dedup fix. Prediction check: CODE-tagged probes
  DID predict strong surface compilability, but marker compilability ≠ construct
  compilability. Next: CW a279 (cell 3).
- 2026-07-12 **a54 INCIDENT RESOLVED — and the resolution is itself a finding.** Blind
  re-author delivered (≥3-doc mention-only anchors, tainted docs excluded, rejected version
  preserved unread); structural gates PASS; installed sha 2a6ba433. RE-CHECK: the cell's
  candidate FAILS the independent contract (4/6 + 1 INVERSION) after passing its
  self-authored one 6/6 — SELF-DEALING WAS SUBSTANTIVE. Cell verdict: REJECT; science
  (sub-relation decomposition, 3 bug fixes, +6.9% train) stands as observation. Process
  datum for the paper: contract-author independence is load-bearing, measured (6/6 vs 4/6+INV
  on the same candidate). Census continues: cell 3 = CW a279.
- 2026-07-12 launched: census cell 3 (CW a279, r_hyb .202) + math contract batch 1 (9 of 17:
  a102 a108 a114 a12 a120 a126 a132 a144 a156; batch 2 = remaining 8 after validation).
  Soft standard tightening logged: mention_only trigger terms should recur in >=3 docs
  (1-doc anchors need written justification) — adopted from the a54 blind re-author;
  CW/humor frozen domains predate it and are NOT retro-edited.
- 2026-07-12 **MATH CONTRACT BATCH 1: 9/9 authored + centrally validated** (a102 a108 a114
  a12 a120 a126 a132 a144 a156). 12/12 independent grep spot-checks exact; 6 'label' flags
  adjudicated benign (LaTeX/discourse labels). Honest disclosures: a132 whole-construct
  corpus-thin (3 docs); a108 surprise-clause corpus-absent. Channel census: 33 L / 5 CODE of
  38 probes — mid-band math constructs contract as L-heavy (contrast CW where structure
  probes skew CODE). Batch 2 (8) launched; domain freeze after it validates.
- 2026-07-12 [wake ~01:00] 3 agents in flight: CW a279 census cell (self-adversary phase per
  cell-dir mtimes), math contract batch 2 (8, just launched), WS4 CELL 4 LAUNCHED (patents_pa
  a26 DAG on M̄(x,Z) — first structural test of "evidence nodes carry the marginals"; refactor-
  only, bit-exact, train-only readouts, rng(7)/40% split reused). Math batch 1 validated
  earlier this hour (9/9). Quiet otherwise; no GPU use, OSL lanes untouched.
- 2026-07-12 **CENSUS CELL 3 (CW a279 ending-effectiveness): CONTRACT-FAIL census datum,
  theory-consistent.** Train .4334→.4783 (+10.4%) but contract never passes (h0 itself 2/5
  w/ 3 INV; candidate plateaus 4/5 + 1 INV on a mention_only L-channel probe = genuine
  semantic mismatch). NOT a promotion candidate. Theory check: the ONE sub-relation with a
  matched library capability (bookend-recurrence ↔ sentence_graph) is the one place code
  flipped INV→SEP; core resonance judgment = MISMATCH. Dominance 41f/27b/80w — honest
  global-constant tradeoff (bimodal CLIFFHANGER label collapsed to one constant). 5 h0/census
  bugs found incl. CLIFFHANGER≡ABRUPT ontology contradiction + serial-regex substring
  collision (0/2 precision) + d03999 corpus-confirms a SUSPECTED census entry (details:
  census/creative_writing__a279/meta.json). Self-adversary rejected 2 tempting fixes that
  hurt train rho and reported 2 unresolved gaps honestly. Cells 1-3: REJECT(process) /
  strong-gain / contract-fail — census cell 4 = CW a171 (.276) launched.
- 2026-07-12 **MATH CONTRACT DOMAIN FROZEN: 17/17 v2.1** (b1 9 + b2 8; all centrally
  validated, non-author). B2 spot-checks: 4/6 exact; 2 count drifts corrected centrally
  PRE-freeze (a6 'in other words' 3→9 case-insensitive — informativeness claim re-verified
  on all 9 by main loop; a42 example 64→65). 6 'label' flags adjudicated benign. a48
  (venue-fit) carries a WEAK-INSTRUMENT flag: construct vocabulary 0/250 on MSE corpus —
  a48 census contract verdicts are not construct evidence. Notable: batch-2 channel mix has
  real CODE mass (a168/a216/a54 all-CODE — notation precision, eqn referencing, navigation
  structure) vs batch-1's 33L/5CODE — the math panel itself spans the seam. Contract
  production now 66/115 v3 (CW 26, humor 23, math 17). Next: PR batch (14).
- 2026-07-12 **WS4 CELL 4 COMPLETE (patents_pa a26 DAG, 4/9) — first node-level confirmation
  of the WS3 evidence prediction.** Bit-exact round 1 (250 items + 18 edge cases). 13 nodes
  (2 L, 8 evidence-class). Own-transformation ablation vs M̄(x,Z): prior_art_lookup Δρ=−.718
  (nulling retrieval REVERSES the sign, −.11; cross-checked 150/150 identical to NullOps
  run); evidence nodes dominate computation ~150× (Σ|Δρ| .740 vs .005) BUT concentrated in
  the R node itself, not its 6 evidence descendants (−.022 combined). NEW BUG CLASS logged
  (ws65 census, first programs_pa entry): dead-gate substring — both L frontier nodes ablate
  to 0.0 EXACTLY because an ≤80-char case-sensitive grounding gate fires 0/150 despite real
  field content (false-NEGATIVE shape; PR triple's class was false-positive containment).
  Harness first: per-item corpus key passed via reserved extracted["__dpid__"] (no schema
  edit). Frontier = 2 computation-class L1 nodes; evidence enters via a disjoint R root.
  Next: WS4 cell 5 = patents a34 (the +.661 op-marginal aspect).
- 2026-07-12 **WS4 CELL 5 COMPLETE (patents_pa a34 DAG, 5/9) — retrieval-node concentration
  REPLICATES.** Bit-exact round 1. rho_intact .7355 train (judge cov 99.3%). prior_art_lookup
  Δρ=−.841 (sign reversal, cross-checked vs NullOps 150/150) = 88.0% of summed |Δρ| — vs
  a26's 96.4%: slightly LESS concentrated (against the cell's a-priori guess for the
  strongest-marginal aspect); a34's tool-layer sub-terms frac_any/frac_max carry ~.05 each.
  Evidence:computation Σ|Δρ| ≈ 79×. Frontier fully DISJOINT from evidence subgraph (a26 had
  one coupling point). NEW census class 7 "starved frontier": a34's L-fields populated only
  2/250 + 1/250 corpus-wide (extraction scarcity) — distinct from a26's dead-gate (match
  strictness rejecting real content); natural fix vehicle = Lane B probe-time re-extraction.
  Next: WS4 cell 6 = patents a35 (triad, +.609) closes the M̄(x,Z) triple.
- 2026-07-12 **CENSUS CELL 4 (CW a171 diction/word-choice): FIRST FULL CONTRACT-PASS CENSUS
  BUILD → promotion queue.** h0 FAILS its contract (1/5, 1 INV — 40-char floor bug inverts
  short-clean text) → candidate PASS 4/5 (probe4 TIE is L-channel-only per the contract's own
  boundary_notes; 80%>75% gate) with train .3605→.4024 (+11.6%). Largest single gain (+.028)
  = fixing h0 DOUBLE-COUNTING LLM-verified flaws already caught by the regex bank (9/128
  docs). Self-adversary caught 2 real OOD holes pre-battery (its/it's trigger list broke 7/7
  legitimate-grammar probes; kind-of classifier-vs-hedge sense split 35/51) and fixed both at
  −.0008 net. Dominance 13f/4b/133w (3.25:1). RELATION-MATCH: 4/5 sub-relations CODE-native
  (cliché, flab, mechanical, intensifier-stack) vs 1 MISMATCH (mention-vs-use = L) — a
  LEXICAL construct sits near the compiled pole with no library capability needed (checked:
  none applies; correct absence, not a gap). Census pattern firming: lexical/structural
  sub-relations compile, semantic-use judgments don't. Queued for held-out batch
  (census/PROMOTION_QUEUE.json). Cell 5 = CW a198 (.286) launched.
- 2026-07-12 **PR CONTRACT BATCH 1: 7/7 validated** (a103 a104 a112 a115 a28 a42 a64).
  Spot-checks 10/11 exact + 1 pattern-precision fix (a64 \b-boundary stated). 4 'label'
  flags benign. Author self-caught a repr()-quoting corruption and rebuilt byte-exact
  quoting — the hardened brief (from math b2's quote-drift lesson) caught it pre-delivery.
  Deconfliction vs frozen v1 six explicitly resolved (sharpest: a112 vs a2/a111; a42 vs
  a111 'next steps' overlap). a64/a42 declared majority-L; a103/a104/a112 CODE-heavy —
  PR panel also spans the seam. 73/115 v3. PR batch 2 (7) launched.
- 2026-07-12 **WS4 CELL 6 COMPLETE (patents_pa a35 DAG, 6/9) — M̄(x,Z) TRIPLE CLOSED.**
  Bit-exact round 1 (250×2 configs + 22 edge checks). rho_intact .6907. prior_art_lookup
  Δρ=−.667 = 84.0% concentration — series 96.4→88.0→84.0, and the cell FALSIFIED the
  tool-layer-count explanation (a35 has most tool nodes yet mid-pack tool signal); the real
  driver is a35's computation-layer strength (~4-10× predecessors; industrial_kw_delta +.030
  = largest computation marginal, ANTI-correlated). a35 does NOT sign-reverse when nulled
  (+.024; a26/a34 both flipped). WS3's doc-only anti-correlation does NOT reproduce against
  the evidence judge — judge-arm-specific. DEAD-GATE CLASS 6 CONFIRMED CLEANEST: 78% field
  population, 0.85% gate fire → population is not the limiter (resolves a34 ambiguity).
  New structure: evidence-gated fallback branch (if pa: 5 terms / else: 1) needed a
  node-ordering guard; frontier is a SINGLETON computation L1 node. Agent self-corrected a
  wrong plan prediction against measurement (ablation ≡ NullOps 150/150). PATENTS TRIPLE
  READ: retrieval-R-node concentration is the family signature; derived evidence transforms
  are thin everywhere; L frontiers are non-functional in all 3 (dead-gated ×2, starved ×1) —
  the a26-era "hybrid" programs are effectively CODE+RETRIEVAL, their LLM seam is vestigial.
  Next: WS4 cell 7 = legal a23 (exhaustion chain).
- 2026-07-12 [wake 01:39] 3 agents in flight, all healthy per artifact mtimes: census cell 5
  (CW a198) in final write-up (adversary.md 01:37); PR batch 2 all 7 files written, in
  verification pass; WS4 cell 7 (legal a23) in read/plan phase (cell dir not yet populated).
  Nothing stuck, no GPU use, OSL untouched. Next steps all gated on these landing (PR b2
  central validation → PR freeze; a198 log; census cell 6; WS4 cells 8-9). Holding.
- 2026-07-12 **CENSUS CELL 5 (CW a198 magic-system design): honest-flat cell, inversion
  fixed.** h0 .4359 CONTRACT-FAIL (2/5, 1 INV — bare-label literal-phrase collision);
  candidate .4352 (−0.16%, flat by design: 2 of 4 strategies built→ablated→REVERTED for
  hurting) but the INVERSION is fixed (3/5 SEP, inverted=False). Self-adversary caught a
  real number-word-fallback hole (rule-COUNT mistaken for rule-CONTENT) and fixed at −.003.
  RELATION-MATCH: marked-circumvention = MISMATCH (L); length-density = CODE-native closed;
  rule-content-vs-label = mixed/recall-capped; NOTABLE: (a) contract tagged deepen-vs-breadth
  CODE but zero corpus support — first TAG-VS-REALITY mismatch datum (contract channel tags
  are hypotheses, census cells test them); (b) ops_capability.attributions() has a structural
  conjunct-verb subject-sharing gap (fails marked-circumvention probe) — logged for the
  E2L-v2 library wishlist. NOT a promotion candidate (contract still fails; gain ~0).
- 2026-07-12 **PR BATCH 2 HELD AT VALIDATION — text-vs-ctext grounding defect.** N-gram
  sweep (new central tool: longest verbatim word-run per corpus_phenomenon vs cited docs)
  found 5 probes across 5 contracts grounded on quotes present ONLY in the raw `text` field
  (stripped from ctext = unrepresentable to every scorer) + 1 count on the wrong field
  (a87 'excited to' 17→15). Author's 46/46 byte-exact claim was true against the WRONG FIELD.
  Sweep run on ALL other v3 domains: PR b1, math 17, CW 26, humor 23 = 0 defects — isolated
  to this batch's author process. Fix pass dispatched to the same agent (re-ground on ctext
  only, state basis explicitly). PR freeze blocked until it lands + re-validates. LESSON
  INSTITUTIONALIZED: future briefs must pin the grep basis to ctext by name; n-gram sweep is
  now a standing validation step.
- 2026-07-12 **WS4 CELL 7 COMPLETE (legal a23 exhaustion chain, 7/9) — first zero-evidence
  cell + depth-vs-signal REVERSAL.** Bit-exact round 1 (250 items, 16+6 edge checks).
  rho_intact .8838 (family high; judge cov exactly 90.0%). 14 nodes, 0 R / 0 evidence ops
  (grep-confirmed). Chain shape: NOT a linear path — DAG with reconvergence (7/14 nodes
  in-degree ≥2, two diamonds). Judgment enters EARLY (2 L1 leaf fields) with a SHORT path to
  verdict (2-3 edges) vs the 5-edge pure-code date-arithmetic engine — which carries almost
  NO signal (Σ|Δρ|=.006 across its 5 nodes) while the shallow L field exhaustion_step is the
  top marginal (−.0856, forced-identical with its classifier node; the false-positive
  discount gate it drives is worth only −.0008 — signal is the direct step term). h0's
  headline invention (date engine) is structurally deep but informationally empty. Class 5
  measured live (6.0%/0.7% fire); class 6 N/A (no gate construct); class 7 not starved
  (42%/89% populated). Next: WS4 cell 8 = legal a21 (neutral-policy→disparity chain).
- 2026-07-12 **PR CONTRACT DOMAIN FROZEN: 14/14 v2.1** (+6 v1 = all 20 census criteria
  covered). Batch 2 passed re-validation after the ctext fix pass: 0 text-only residuals
  (n-gram sweep), 6/6 corrected-count spot-checks exact, author self-caught 2 further bugs
  mid-fix (transcription typo; wrong-field aggregate that coincidentally matched). STANDING
  RULE ADDED to validation ledger: ctext is the corpus basis BY DEFINITION; the text-vs-ctext
  n-gram sweep is now a required central step for every batch. Contract production 80/115 v3
  (CW 26, humor 23, math 17, PR 14). Next domain: legal_title_vii (14).
- 2026-07-12 **WS4 CELL 8 COMPLETE (legal a21 disparate-impact chain, 8/9).** Bit-exact
  round 1 (250 items, 17+8 checks + 1 documented non-gated synthetic-only divergence).
  rho_intact .6075 (cov 98.7%). SHAPE REPLICATES a23: zero-evidence, reconvergent DAG (5/14
  in-degree≥2), judgment enters at depth-0 L leaves with 2-edge path to verdict vs 5-edge
  deepest code chain. NEW: the two-step doctrine materializes as a real INTERACTION —
  final_combine multiplies policy_component×stat_component; the AND fires on only 2/150
  items yet disabling it costs Δρ −.149 (rare-but-consequential structure). Top marginals:
  policy_component −.314, neutral_practice_field −.233, stat_component +.136
  (ANTI-correlated, reported unsmoothed). Forced-identical pairs: NONE — agent corrected its
  own initial false positive by per-item check (2 near-pairs differ on 1/150 items =
  coincidence, not identity). CLASS 7 CONTRAST: a21's L frontier is STARVED (6.0%/2.7%
  populated) where a23's was rich (42%/89%) — yet the starved field still carries the #2
  marginal. Next: WS4 cell 9 = legal a13 (constructive-discharge conditions chain) closes
  the pilot.
- 2026-07-12 **CENSUS CELL 6 (CW a315 transformation-over-imitation): +17.2% BY SUBTRACTION;
  bug-census CLEAN verdict OVERTURNED.** h0 .3316 contract-FAIL (2/6, 4 INV — h0 had ZERO
  code for the contract's two CODE-tagged axes; its only code-mode differentiator was a
  TF-IDF proximity nudge = pure noise, rho~.015 vs judge). Candidate .3885 (+17.2%): the
  ENTIRE dominance gain (136f/12b, 11.3:1) attributes to REMOVING the noise nudge — reported
  precisely rather than crediting the new detectors; removal also honestly demoted 2 fake
  noise-SEP probes to real TIE (5/6 → 2/6 SEP, 0 INV). A 144-point constant grid-sweep was
  tried and REJECTED with a proof it only gamed Spearman tie-ordering. Contract still FAIL →
  NOT queued (a279 precedent). Self-adversary caught its own ungated expect/instead detector
  firing on zero-genre text; fixed with genre gate. 4 bugs: census 'comic'/'Comics' substring
  CLEAN verdict OVERTURNED (ws65 updated); TF-IDF nudge construct-invalid; double-counting;
  own-B2. RELATION-MATCH (7 sub-relations): 2 CODE-native (closed from zero), 1 mixed,
  1 inherently-L, 3 MISMATCH single-instance (correctly un-chased); library functions
  tested directly = no match exists for this family (correct absence). Contract tags all
  verified honest. Cell 7 = CW a135 (.311) launched.
- 2026-07-12 **LEGAL CONTRACT BATCH 1: 7/7 validated** (a0 a10 a15 a18 a26 a28 a3). N-gram
  sweep clean; outcome-vocab absence verified 0/250. TWO INSTRUMENT FINDINGS: (1) 10/20
  legal aspect descriptions truncated at exactly 600 chars MID-SENTENCE at source
  aspects.json — traced: build_task.py judge prompts used the SAME truncated text, so the
  truncation IS the operative construct (contracts verbatim-copy correctly; do NOT fix;
  frozen-history judge-noise caveat on the whole legal ladder, logged in validation ledger);
  (2) corpus contamination: 2/250 items are non-Title-VII cases (d00667 ECOA, d00223 VA
  loan) — flagged for census reads. Author self-caught 'race'-in-'racial' substring traps
  pre-delivery. Legal batch 2 (7) launched. 87/115 v3.
- 2026-07-12 **WS4 CELL 9 COMPLETE (legal a13 constructive discharge) — PILOT CLOSED 9/9,
  ALL BIT-EXACT ROUND 1.** Legal family signature 3/3 CONFIRMED: zero-evidence + reconvergent
  DAG + depth-0 L-leaf judgment entry + short verdict path (2-3 edges) vs deep code chain
  (exactly 5 edges in ALL THREE legal cells) that is signal-poor. a13 interaction nuance:
  two within-field AND-gates (1.33%/2.0% fire) but load-bearing on only 1/150 each — more
  attenuated than a21's cross-field product. Class 5 confirmed-but-inert ('cut' in
  'executive'/'prosecute'); class 7 starved-side (8%/4%). UNPLANNED HARD-RULE TRIGGER: a13
  TRAIN judge coverage 58.0% — first sub-90% cell in the ladder; traced to judge-side
  parsing/refusal (43.6% of pass2 unparsed; rel1 .935 when parsed) NOT program/split →
  structure-only fallback engaged, no rho readouts computed (score-delta substitutes,
  labeled). DATA-QUALITY FLAG: any consumer of legal a13 judgments must know about the 42%
  missingness.
- 2026-07-12 **WS4 PILOT SYNTHESIS (9 cells: PR a119/a28/a76, patents a26/a34/a35, legal
  a23/a21/a13).** (1) Bit-exact refactor is CHEAP — 9/9 round-1, the typed-DAG rung costs
  ~1 agent/cell. (2) Seam position is FAMILY-STRUCTURED: PR = thin 2-node L frontier under a
  pure-code assess layer; patents = CODE+RETRIEVAL with vestigial L frontier (dead-gated ×2,
  starved ×1) and retrieval-R concentration 84-96%; legal = zero-evidence reconvergent DAGs,
  judgment at depth-0 leaves, 5-edge code chains signal-poor. (3) Depth ≠ signal (a23 date
  engine; a13 code layer top-marginal instead). (4) Doctrinal structure can materialize as
  rare-but-consequential interaction nodes (a21 −.149 at 1.33% fire; a13 attenuated version).
  (5) Instrument yields: 2 new bug classes (6 dead-gate, 7 starved-frontier) + 1 judge-side
  coverage hole (legal a13) found only because the DAG forced per-node visibility. Remaining
  registered follow-ons: held-out r̃ vs scalar incumbents under G1 (one batch, later);
  U₂-matroid-over-DAG-partition theory task (parked).
- 2026-07-12 [wake 02:39] 3 agents in flight, all healthy per mtimes: census cell 7 (CW
  a135, candidate.py active 02:35) + cell 8 (CW a324, plan+candidate 02:38) + legal contract
  batch 2 (grep/authoring phase, no files written yet — normal for first ~25 min). WS4 pilot
  closed 9/9 earlier this hour (task #12 completed); contracts 87/115. No GPU use, OSL
  untouched. Next gates: legal b2 validation → legal freeze → ssdis batch; census cells 7-8
  land → cells 9+ (CW tail) + CW held-out promotion batch planning. Holding.
- 2026-07-12 **LEGAL CONTRACT DOMAIN FROZEN: 14/14 v2.1** (+6 v1 = all 20 legal census
  criteria). B2 7/7 clean (zero flags, 0 n-gram residuals, spot-checks exact incl. the
  'opposed' 1-of-8-genuine lexical trap). Author self-caught an ADA-vs-Title-VII positive
  and reclassified it as a mention-only trap (a0-precedent-consistent). 94/115 v3 (CW 26,
  humor 23, math 17, PR 14, legal 14). Remaining: ssdis 13 + peer 8. ssdis batch 1 launched.
- 2026-07-12 **CENSUS CELL 8 (CW a324 character dimensionality): SECOND FULL CONTRACT-PASS →
  queue.** h0 .4398 FAIL (1 INV — keyword scanner had no change-vs-stasis polarity) →
  candidate .4421 (+0.52%, wash-level) CONTRACT PASS 5/5. Subtraction-first: 3 bug-census
  CONFIRMED fixes (rambling-reward, denominator, inverted-lexicon). NEW CAPABILITY flagged:
  spaCy interiority-vs-partitive 'part of {pronoun}' disambiguator (candidate-local, E2L-v2
  wishlist). 3 strategies for the dominant residual cluster tested and REJECTED as L-bound
  (matches contract boundary_notes). Self-adversary caught 3 real holes incl. its own
  NameError silently collapsing scores (ablation catch) + a cross-sentence recall bug +
  type-coercion credit. Dominance 44f/48b/58w = flat, honest. Queued on the contract-PASS
  criterion; held-out batch will decide (gain likely wash). Sha verified vs _cw_domain_freeze
  (26 entries; cell agent's 'not covered' note was a lookup in the wrong block — provenance
  intact). Cell 9 = CW a9 (.358) launched.
- 2026-07-12 **SSDIS CONTRACT BATCH 1: 7/7 validated** (a0 a1 a10 a11 a13 a15 a16). 6/6
  spot-checks exact; 0 n-gram residuals; author self-caught 7 transcription errors + 1
  count in its own final pass. SPECIAL ADJUDICATION a13: genuine axis CORPUS-ABSENT (0/250,
  matches r_hyb=1.0 constant signature) → authored-positives-from-real-doctrine + 3 real
  constitutional-adjacent mention-only negatives ACCEPTED with untestable-genuine-axis rider.
  a15 paraphrase note flagged for census (literal DLI string under-counts 3/7 true docs).
  Corpus fact: 'remand' = claimant boilerplate in 235/250, correctly excluded as signal.
  101/115 v3. ssdis batch 2 (6, final) launched.
- 2026-07-12 **CENSUS CELL 7 (CW a135 foreshadowing): THIRD QUEUE ENTRY, first PASS→PASS
  cell.** h0 .3358 and ALREADY CONTRACT-PASSING (census first — cells 3-6,8 h0s all failed)
  → improvement under no-regression constraint: candidate .3750 (+11.7%) PASS 5/5 every
  round. Diagnosis: h0 length-confounded (rho-vs-length .672 vs judge's .240) via
  genre-topic-word 'echo' detectors + redundant length gate. Round-1 bigram fix caused a
  real probe regression (74-word probe too short for bigram recurrence) → REVERTED and
  repaired with fallback-hybrid: the discipline held. Uses is_refrain() from the frozen
  library (2nd census cell to find a library match). Self-adversary caught mention-only
  padding out-scoring h0 on identical text; fixed. Dominance 80f/52b/17w; broken traced to
  the contract's own disclosed L-boundary. TAG-VERIFICATION NOTE: probe-0 corpus count
  claim (49/250) not replicable (6-8/250 best-effort; noun list unspecified) — frozen
  contract NOT edited; logged for a contract-quality audit pass. Cell 10 = CW a207 (.378).
- 2026-07-12 **SSDIS CONTRACT DOMAIN FROZEN: 13/13 v2.1** (b1 7 + b2 6). B2 6/6 clean; the
  2 apparent count deltas were MY spot-check patterns vs the contracts' STATED patterns —
  with stated patterns both reproduce EXACTLY (27/23, 43) — the pattern-statement rule is
  paying off. a19 joins a13 with the authored-positives/near-untestable rider. Rich
  mention-only trap set (step-3-vs-step-2 'combination' 33:1, unchallenged step-5
  boilerplate 78/250). 107/115 v3. PEER BATCH (8, FINAL) launched.
- 2026-07-12 **CENSUS CELL 9 (CW a9 plot coherence): inversion-fix cell, honest flat.** h0
  .6387 = strongest CW h0 yet, but CONTRACT FAIL (1/5, 1 INV) — root-caused to
  position-blind chrome-stripping in _story_paragraphs exposing a setup sentence as 'the
  ending' (NEW unrecorded defect in the class-5 mechanism family). Candidate .6394 (+0.11%)
  3/5 SEP 0 INV; probes 3/4 confirmed genuine L. SIX candidate signals corpus-tested and
  REJECTED (incl. 2 with real counter-example docs — 7/9 scene-break-chrome instances are
  HIGH-judged). Self-adversary caught _yn() granting malformed types unearned credit (same
  class as a324's coercion bug — 2nd instance, now a named pathology: TYPE-COERCION CREDIT).
  Dominance 0f/1b/149w. NOT queued (contract FAIL). Cell 11 = CW a18 (.391) launched.
- 2026-07-12 [wake 03:39] 3 agents in flight, healthy: census cell 10 (CW a207, candidate.py
  active 03:38), cell 11 (CW a18, read/plan phase), peer contract batch (final batch,
  authoring phase, 0/8 files yet — normal). This hour's completions already logged: SSDIS
  FROZEN 13/13, cells 7+9 landed (a135 → queue #3; a9 inversion-fix), cell 8 a324 → queue #2.
  107/115 contracts; queue 3; WS4 done. No GPU, OSL untouched. Holding.
- 2026-07-12 **CONTRACT PROGRAM COMPLETE: 115/115 v3 AUTHORED + VALIDATED + FROZEN (7/7
  domains).** Peer batch 8/8 clean with GENRE CHECK (all 33 probes abstract-register — the
  GENREBUG mode did not recur; author read GENREBUG quarantine as its cautionary example).
  Peer author also FLAGGED the a29 slot: contracts/peer_review__a29.json was byte-identical
  to the QUARANTINED GENREBUG file (3f2f4c70), not the corrected re-author (c984311f) — the
  2026-07-11 genre-fix wave placed the wrong file in the harness slot. CORRECTED (v2
  re-author now installed); impact = none (a29 e2 cell was already PROVISIONAL pending
  corrected-contract re-run; no other consumer read the slot). Ledger: _a29_slot_correction
  + _peer_domain_freeze + _PROGRAM_COMPLETE blocks. Census coverage now: every one of the
  159 census criteria has a frozen contract (115 v3 + 32 v1 + 8 peer v2 + 4 CW v1-era).
- 2026-07-12 **CENSUS CELL 10 (CW a207 conflict design): NEW CENSUS DATUM TYPE — gain by
  SEAM PLACEMENT, not compilation.** h0 .5769 PASS → candidate .6028 (+4.5%) PASS; queue
  entry #4 (self-written by the cell, well-formed, honest disclosures verified). ~100% of the
  gain came from ADDING 2 new structured LLM fields (conflict_source ORGANIC/IMPOSED,
  stakes_proof DEMONSTRATED/ASSERTED, extracted via glm-4.7; shared field cache appended
  with .bak backup) — the first cell whose win is BETTER L-CHANNEL STRUCTURE rather than
  code: for L-heavy constructs the compilable move is structuring the judgment interface,
  not replacing judgment (WS4's seam-position lens, now visible in the census). Class-5 bug
  found WORSE than cited (2 new collision families: 'Killian', 'fleet'); fixed for validity
  at wash-level rho. Self-adversary: laundering hole in its OWN new field mitigated to
  +.045 residual and DISCLOSED as not closed; a fuller fix tested and REJECTED (regresses
  43% of train). attributions() gave a narrow partial signal (beyond clean-negative — 3rd
  library-relevance datum). All 14 contract grep claims re-verified, zero discrepancies.
  Cell 12 = CW a189 (.453) launched.
- 2026-07-12 **CENSUS CELL 11 (CW a18 suspense/pressure): biggest single L-field
  interaction bug of the census.** h0 .4988 FAIL (2/5, 1 INV) → candidate .5410 (+8.46%)
  3/5, 0 INV (still FAIL → not queued). MARQUEE: the LLM tone field's flat comedic penalty
  was FLATTENING 40% OF ALL TRAIN ITEMS to an identical 0.0 floor — including 13 items where
  the LLM's own peril field independently confirmed real danger (judge .254, h0 crushed to
  .036); fixed by gating tone on peril — an L×L interaction bug invisible to code-only
  audits (new census pathology: FLAT-PENALTY FLATTENING). Probe-1 inversion root-caused by
  hand-decomposition (30-word floor × flat penalty = shorter-scores-higher). TYPE-COERCION
  CREDIT 3rd instance. Self-adversary caught a SELF-INTRODUCED length-reward bug (its own
  denominator-cap shortcut hit rho-vs-len .404 vs judge .255) via the length-keys battery
  and replaced it with the census-specified rolling-window-max. 2 laundering vectors fixed;
  2 contract-motivated lexicon additions REJECTED on corpus evidence. Cell 13 = CW a270
  (.457) launched.
- 2026-07-12 **CENSUS CELL 12 (CW a189 form-content integration): CONTRACT-RESCUE → queue
  #5.** h0 .4305 FAIL (3/5, 1 INV, 1 TIE) → candidate .4451 (+3.4%) PASS 4/5, 0 INV.
  NEW BUG CLASS 8 (register collision): bureaucratic-caps vs shouting-caps — an all-caps
  detector for one register fires on another (memo headers), causing the inversion (ws65
  updated). BOTH levers used in one cell: COMPILATION (closing-signature detector gated on
  ops_capability.discourse_position — 4th library-relevance datum, incl. a worked-around
  library edge case) + SEAM PLACEMENT (new compound form_status field, glm-4.7, 150 items
  0 errors, cache backed up + dedup-verified). R4 fixed 2 holes the new field itself
  introduced (LLM over-labeling dialogue as 'transcript'; a BACKWARDS net-positive credit
  for BROKEN forms). Laundering residual disclosed. 3 inherited PROXY mechanisms honestly
  flagged as not-audited-this-cell. All 7 contract grep claims re-verified exact. Queue: 5.
  Cell 14 = CW a216 (.469) launched.
- 2026-07-12 [wake 04:39] 2 agents in flight, healthy: census cell 13 (CW a270,
  self-adversary phase, files active 04:39) + cell 14 (CW a216 contract-blind floor cell,
  read/plan). This hour's completions already logged: 115/115 CONTRACTS COMPLETE (7/7
  domains frozen) + a29 slot correction; cells 10-12 landed (a207 seam-placement datum,
  a18 flat-penalty-flattening, a189 rescue → queue 5). Lane B residue = probe-time field
  extraction GPU job (sk3, not yet built) + bug-fix promotion batches (flowing through
  census cells). No GPU, OSL untouched. Holding.
- 2026-07-12 **CENSUS CELL 14 (CW a216): STOP-AND-FLAG FIRED — the a54 incident rule worked
  end-to-end.** My queue brief missed that a216 is one of the 4 v1-era CW stragglers
  (a54/a90/a144/a216, never in the v3 domain freeze); the cell verified (v3 missing, v1
  present + sha-intact vs incident ledger, freeze block cross-checked), STOPPED with zero
  build (57k tokens vs ~300k typical), wrote a meta.json documenting the check, and
  recommended exactly the right fix. Contrast with a54's improvise-in-place two days ago:
  the containment rule turned the same gap from an incident into a cheap clean stop.
  RESOLUTION IN MOTION: built v3 packs for a90/a144/a216 (improver-pack recipe, source_note
  stamped); straggler contract batch launching (v2.1 authoring; v1 contracts stay untouched
  in slots until validated replacements land with .v1_backup copies). a216 cell re-queued
  after validation.
- 2026-07-12 **CENSUS CELL 13 (CW a270 compression/flash-forms): PASS → queue #6, with the
  census's most honest disclosure set.** h0 .4498 FAIL with 3/5 INVERTED — root cause a NEW
  pathology shape: DEAD-ZONE CURVE (length curve active 0-120 words has ZERO train support,
  min train doc = 122 words — it exists only where synthetic probes live, dominating the
  contract while invisible to train rho; h0 rho-vs-len −.80 vs judge −.34). Candidate .4509
  (+0.24%, wash) PASS 4/5 0 INV via SUBTRACTION (flatten the untested curve, 0 rho cost,
  fixed all 3 inversions). DISCLOSED-NOT-SPUN: dominance UNFAVORABLE (22f/32b/96w, .688)
  and post-fix length-correlation MORE extreme (−.88), traced to noise removal. is_refrain()
  MISMATCH documented with mechanism (≥3-word clustering floor vs corpus's 1-2-word
  refrains) — 2nd concrete library-maturity item for the E2L-v2 wishlist (with attributions'
  conjunct gap). Refrain probe left as deliberate honest TIE (corpus's own cited refrain
  example scores LOW — gaming it would be probe-targeting). Queue: 6.
- 2026-07-12 **CW STRAGGLERS CLOSED: 30/30 CW criteria now v2.1** (a90/a144/a216 authored
  fresh, 3/3 centrally validated, spot-checks exact, v1 slot files backed up as .v1_backup
  before v3 installation; with a54's blind re-author the v1-era subset is fully retired).
  a216 carries the authored-positives rider (positive pole corpus-absent — 3rd such
  criterion after ssdis a13/a19; note the pattern: corpus-absent POSITIVE poles cluster in
  control/floor criteria where the population rarely exhibits the virtue). a90's
  overload-pole reoperationalization disclosed. a216 census cell RE-QUEUED (unblocked).
  118/118 v3 contracts total.
- 2026-07-12 **CENSUS CELL 15 (CW a126 scene-vs-summary): FIRST CONTRACT-PASS AT TRAIN
  COST — construct-vs-judge divergence datum.** h0 .6297 FAIL (1/5, 2 INV: half the
  criterion definition uncompiled + a backwards dialogue-density reward = TYPE-COERCION
  CREDIT family) → candidate .6174 (−1.95%, DISCLOSED) PASS 5/5 0 INV. The two
  contract-required detectors are corpus-present but NULL-TO-NEGATIVE vs judge on train
  (checked BEFORE implementation; the 2 biggest losses are genuine construct instances
  embedded in low-judged pieces) — first clean case of the criterion DEFINITION demanding
  axes the JUDGE doesn't reward: contract fidelity and judge fit pull OPPOSITE ways.
  QUEUED with the negative delta carried in the note (held-out arbitrates; the queue now
  spans +11.7% to −1.95%, which is exactly the spread the G1 batch should adjudicate).
  Class-4 SUSPECTED→CONFIRMED (bare 'suddenly'/'eventually'); a rejected ping-pong-dialogue
  penalty documented as construct-invalid (correlates POSITIVELY with judge — legitimate
  banter). Queue: 7. Cell 16 = CW a180 (.497) launched.
- 2026-07-12 [wake 05:39] 2 agents in flight, both mid-build and healthy: cell 14R (CW a216
  contract-blind measurement, rounds.log active 05:39, dominance already computing) + cell
  16 (CW a180, plan+rounds active 05:38). This hour already logged: CW stragglers closed
  (30/30 CW v2.1, slots installed w/ v1 backups), cell 13 queue #6 (dead-zone curve
  pathology), cell 15 queue #7 (FIRST pass-at-train-cost — construct-vs-judge divergence).
  Queue 7; 118/118 contracts; CW census 15/30 resolved. No GPU, OSL untouched. Holding.
- 2026-07-12 **CENSUS CELL 16 (CW a180 endings/payoff): PASS → queue #8; DEAD-ZONE CURVE
  2nd confirmation, most extreme form.** h0 .6059 but 0/5 SEP — ALL probes tied at one
  constant: h0's only code fallback has a <80-token floor calibrated for ~600-word stories
  while probes run 22-34 tokens (the harness saw literally nothing). Candidate .6204 (+2.4%)
  PASS 4/5 round 1 via 1 class-5 fix ('thank you' vs 'Thanksgiving', 2 corpus examples) + 4
  corpus-verified detectors (rare-noun bookend echo +.166 standalone). Probe 2 = safe TIE
  (every lever corpus-absent; contract's own majority-L framing). Self-adversary: 2 holes
  fixed (1 capped-not-closed, disclosed), 3 more surfaced-and-disclosed incl. unfavorable
  absolute-calibration dominance (5f/48b) root-caused to h0's pre-existing +.32 scale bias,
  not this cell. All 7 contract grep claims re-verified exact. Library: is_refrain clean-
  negative, discourse_position considered-and-rejected. Queue: 8. Cell 17 = CW a252 (.513).
- 2026-07-12 **CENSUS CELL 14R (CW a216 cultural authenticity): PASS → queue #9 — and the
  CONTRACT-BLINDNESS phenomenon is now MEASURED.** rho .4355→.4528 (+3.97%), FAIL→PASS
  (4/5, 0 INV). Blindness fractions (3 cuts): only 10% of train items ever get an L-field
  value; 14% have scores moved by fields; CODE-ONLY EXPLAINS 83.7% of the candidate's rho
  (vs h0's 79%) — part of what the E2-era floor-cell diagnosis called 'inherently L-only'
  was fixable in code once diagnosed. Classes 4+5 fixed via ONE shared mechanism
  (≥2-distinct-stem requirement), verified on ws65's own cited docs then OOD-demonstrated.
  The mandated DIALECT-ERASURE self-test caught a real hole: glm-4.7's group_mockery field
  flagged bare AAVE dialect-NAMING as mockery — fixed with a mockery-indicator gate at zero
  train recall loss (checked all 10 real hits). Probe 4 honest TIE (its corpus citation is
  sealed in TEST — noted, not chased). Straggler-freeze sha + slot byte-identity verified.
  Queue: 9. Cell 18 = CW a63 (.514) launched.
- 2026-07-12 **CENSUS CELL 17 (CW a252 withholding/payoff): PASS → queue #10 (pass-at-cost
  −0.59%); DEAD-ZONE 3rd confirmation; CONTRACT_ERRATA ledger opened.** h0 .4424 but 0/5 —
  entire construct predicate lives in 2 LLM fields (contract_check scores extracted={}) +
  a length floor identical on both probe sides: the purest both-pathologies case yet.
  Candidate: pure ADDITION cell (ws65 classes 1-5 re-verified CLEAN, no subtraction target)
  — 4 corpus-grounded detectors → PASS 5/5 at −0.59% disclosed (a +2.49% intermediate was
  DELIBERATELY REJECTED after self-adversary found a real precision hole in it — validity
  over rho, the discipline holding at its most tempting point). 3 holes fixed incl. a
  present-vs-mentioned payoff guard; 4 residuals disclosed as genuine L-limits. Dominance
  5f/8b/137w unfavorable, root-caused (not inherited). ERRATUM #2 in a frozen contract found
  (a252 p4 anchor d01192: all 'secret' hits are 'secretary' substrings) → central
  CONTRACT_ERRATA.json opened (never-edit-frozen rule maintained; errata feed the eventual
  audit). Queue: 10. Cell 19 = CW a81 (.524) launched.
- 2026-07-12 **CENSUS CELL 18 (CW a63 plot-progression): PASS → queue #11 — first
  FAVORABLE-dominance flat-rho cell.** h0 .6154 FAIL (both CODE probes INVERTED; root cause
  = double-counted generic-connective lexicon carrying 0.45 effective weight in exactly the
  code-only branch contract_check exercises). Candidate .6150 (flat −0.06%) PASS 5/5, 0 INV
  — round 2 had +2.39% but round 3's robustness fixes (mention/padding false-positive +
  synonym-swap generalization) cost it back; disclosed as the tradeoff it is. DOMINANCE
  FAVORABLE: 84f/37b/28w, MAE −4.9% — rank-flat but absolute-error-better (a readout
  Spearman can't see; adds weight to reporting both). Contract-blindness heterogeneity:
  code-only share 49.0% here vs 83.7% (a216) — blindness is construct-specific, not a CW
  constant. discourse_position() PERFECTLY resolves probe-0's axis but was not adopted
  (needs a 3rd LLM field > contract's 2-field budget) — first case of a library capability
  blocked by BUDGET rather than relation-mismatch (new wall category for the taxonomy:
  matched-but-unaffordable). Queue: 11. Cell 20 = CW a162 (.537) launched.
- 2026-07-12 [wake 06:39] 2 agents in flight, healthy: cell 19 (CW a81, self-adversary
  phase, 06:38) + cell 20 (CW a162, implement rounds, 06:37). This hour already logged:
  cells 17 (queue #10, dead-zone 3rd confirm, +2.49%-intermediate REJECTED for validity,
  CONTRACT_ERRATA ledger opened) and 18 (queue #11, favorable-MAE flat-rho,
  matched-but-unaffordable wall category). Queue 11; CW census 19/30. No GPU, OSL
  untouched. Holding.
- 2026-07-12 **CENSUS CELL 19 (CW a81 multithread/ensemble): PASS → queue #12 (+7.32%).**
  h0 .4165 FAIL (1 INV) → candidate .4470 PASS 4/5 0 INV. NEW h0 defect shape: EXACTLY-ONE-
  ACTOR DEAD ZONE (0 actors penalized, 1 actor escapes with neither penalty nor credit) —
  a categorical dead-zone variant, corpus-prevalent 26.7% of train. Self-adversary found a
  45.2%-prevalence apostrophe bug (contractions treated as dialogue-open markers) — fixed
  at −.027 disclosed rho cost, then re-ablation dropped a now-net-negative bonus (+.013
  back). External validity: candidate SEP on a held-out different-surface pair where h0
  still inverts. 2 new candidate capabilities (_voiced_names, _sentence_lead_actors) —
  attributions() tested and STRUCTURALLY BLIND to action-beat-then-quote attribution (3rd
  concrete library-maturity item: no reporting verb needed in fiction register). Dominance
  unfavorable but root-caused to pre-existing h0 calibration. Errata: none (probe claims
  re-verified exact). Queue: 12. Cell 21 = CW a45 (.559) launched.
- 2026-07-12 **CENSUS CELL 21 (CW a45 stakes/vulnerability): PASS 6/6 → queue #13 — seam
  MOVED, measured.** h0 .6510 FAIL (probes 0-4 all tie: predominantly-L construct, code
  channel = only 32% of h0's rho) → candidate .6610 (+1.54%) PASS with ALL margins healthy.
  The build kept h0's L-field core UNCHANGED and added a corpus-checked code-only layer —
  CODE-ONLY SHARE 32.0%→55.5%: the first cell to show compilation shifting the seam INTO
  code on an L-heavy construct (vs a216 where blindness was overestimated, and a207 where
  the win was L-structuring). ws65 class-4 SUSPECTED→CONFIRMED + fixed; dead-zone shape (ii)
  confirmed as the contract blocker; a quote-mark clause-splitting bug found+fixed twice
  over. Self-adversary: 3 holes fixed at +0.0001 net; 4 lexicon gaps disclosed-not-fixed
  (corpus-absent vocab = overfit risk). Dominance MIXED reported honestly (rank favors
  candidate; MAE unfavorable in one tone bucket, root-caused to h0's overshooting core).
  Queue: 13. Cell 22 = CW a297 (.561) launched.
- 2026-07-12 **CENSUS CELL 20 (CW a162 theme clarity): PASS → queue #14 at rho parity;
  first is_refrain() ADOPTION.** h0 .6382 FAIL 0/5 (code path literally constant 0.34 —
  L fields carry 0.74 of weight; all-L constant-tie pathology re-verified live on a 20-text
  battery, zero exceptions) → candidate .6383 PASS 4/5 0 INV. Levers: SUBTRACTION (2 dead
  signals) + SEAM PLACEMENT via the LIBRARY (is_refrain()'s varied_final flag adopted for
  probe 2, corpus rho +.137, NO new LLM field — 5th library-relevance datum and the first
  is_refrain success after cell 13's mismatch: same function, different sub-flag, different
  construct) + code-only intent/theme proxies. 4-step round-1 self-correction chain (1a
  regression traced → 1b reflexive-pronoun gate → 1c noise drop → 1d regex trim to
  validated core) — the ablate-every-round rule drove all 4. Probe-4 axis investigated
  twice, 0/150 hits, correctly NOT built. Honest gameability ceiling: probe-2 fix gameable
  by filler-append but structurally inseparable from the contract's own positive; exploit
  ceiling measured (.0078) + disclosed. Self-caught substring bug ('moved on' in 'moved
  one') + a parents[5]-vs-[6] scaffolding bug. Queue: 14. Cell 23 = CW a234 (.565) launched.
- 2026-07-12 [wake 07:39] 2 agents in flight, healthy: cell 22 (CW a297, adversary.md
  writing 07:39 — near done) + cell 23 (CW a234, implement rounds 07:39). This hour already
  logged: cells 19-21 landed (queue #12 a81 +7.3% w/ categorical-dead-zone find; #13 a45
  seam-shift 32→55.5% code share; #14 a162 parity w/ first is_refrain sub-flag adoption).
  Queue 14; CW census 22/30; after a297/a234: a153, a117, a342, a99 + 3 controls close CW →
  CW held-out promotion batch becomes next major step. No GPU, OSL untouched. Holding.
- 2026-07-12 **CENSUS CELL 22 (CW a297 voice authenticity): PASS → queue #15 (+5.85%) —
  first ANTI-SIGNAL-to-positive seam shift.** h0 .4107 FAIL (probe-3 INVERSION: a
  n_sent≤2 degenerate floor fires 0/250 corpus but 4/5 probes — fresh dead-zone-(i)
  instance, $0-cost fix) → candidate .4347 PASS 5/5. SEAM-SHIFT: code-only channel from
  −8.8% of rho (h0's code path was ANTI-signal) to +13.4% — modest but sign-flipping,
  consistent with majority-L framing (every added signal 0-5/250 hits). ws65 class-1
  self-match CONFIRMED bug fixed (d00779 self-sim .9554 < .98 threshold; rank-0-drop).
  External validity: fresh different-surface pair SEPARATES where h0 ties. Dominance wash
  (rank +5.8% favorable, MAE neutral, root-caused). 5/6 OOD synonym-swaps disclosed as
  misses-never-inversions. Queue: 15. Cell 24 = CW a153 (.571) launched.
- 2026-07-12 **CENSUS CELL 23 (CW a234 rasa/aesthetic-emotion): PASS → queue #16 (+2.76%);
  weakest code channel yet (2.8%) shifted ~8× to 22.2%.** h0 .5502 FAIL (shape-(ii)
  dead-zone, 4/5 tied) → candidate .5654 PASS 5/5 with 4 new length-independent corpus-
  grounded signals (one per dead probe). PRE-BUILD REJECTIONS did the safety work: a naive
  motif family measured at rho-vs-wordcount .91 and killed before implementation (the
  length-echo trap detector working as designed). Padding attack on the visceral bonus fixed
  with a words-per-hit DENSITY GATE calibrated against real corpus stats (308 w/hit train
  min vs 9.1 attack); a second +0.05-ceiling hole disclosed-not-fixed exactly as the
  contract itself predicted for that proxy. Dominant unresolved limiter named: the
  (strong-rasa, mid-craft) half-of-train bucket has ZERO within-bucket resolution — natural
  fix is a 3rd continuous LLM field = matched-but-unaffordable again (2nd instance).
  Dominance mixed (rank favorable, MAE unfavorable, root-caused to additive widening on
  overshooting items). Queue: 16. Cell 25 = CW a117 (.595) launched.
- 2026-07-12 **CENSUS CELL 24 (CW a153 humor craft): PASS → queue #17 (+0.56%) —
  calibration-aware tuning debut.** h0 .7280 (strongest CW h0, code share +33%) FAIL (1 INV:
  _HUMOR_KW double-counting + negation-blindness — 'Nobody laughed' earned full credit) →
  candidate .7321 PASS 4/5. Self-adversary deliberately swept its own deadpan bonus DOWN
  (0.12→0.05) after root-causing 12/12 dominance-broken items to it — trading rho for a
  CLEAN dominance win (5f/1b/144w, MAE −0.18%): first cell to tune on the calibration
  readout. Probe 3 honest TIE (two corpus-tested proxies both wrong-signed −.145/−.154 —
  correctly not built). New substring collision found beyond ws65's list (pun/punish).
  Erratum #3 recorded (probe-3 count off 1-3; anchors valid). Bounded gaming residual
  (+.007 max) disclosed. Queue: 17. Cell 26 = CW a342 (.612) launched.
- 2026-07-12 **CENSUS CELL 25 (CW a117 causal coherence): PASS → queue #18 — best
  both-ways dominance of the census.** h0 .5730 FAIL 0/5 (severest shape-(ii) case: leap
  penalty needs ≥120 words, all probes shorter; h0 is 5× MORE length-driven than the judge,
  rho-vs-len .277 vs .054 — class 2 CONFIRMED) → candidate .5744 (+0.24%) PASS 5/5. Dominance
  FAVORABLE BOTH WAYS: MAE −16.1% relative, 115 fixed/29 broken — the calibration payoff of
  removing a length confound dwarfs the rank movement (flat rho hid a 16% calibration win;
  the both-readouts rule keeps proving out). 3 bugs caught in-build (a clamp silently
  flooring differentiated negatives — erased SEP on 3 probes; negation-blindness on the
  probe's own text; a wrong-signed 'suddenly' marker +.082 caught by residual inspection).
  Padding attack closed with the cell-23 density-gate recipe (125 w/hit corpus min). 1
  corpus-thin load-bearing device disclosed. Queue: 18. Cell 27 = CW a99 (.696) launched.
- 2026-07-12 [wake 08:39] 2 agents in flight, both near completion: cell 26 (CW a342,
  self-adversary results written 08:35) + cell 27 (CW a99, final tool runs 08:38). This
  hour already logged: cells 23-25 (queue #16 a234 8× seam-shift; #17 a153
  calibration-aware tuning debut + erratum #3; #18 a117 best both-ways dominance, MAE
  −16.1%). Queue 18; CW census 26/30; only 3 controls left after in-flight pair → CW
  held-out promotion batch is next major step. No GPU, OSL untouched. Holding.
- 2026-07-12 **CENSUS CELL 26 (CW a342 mechanical correctness): PASS → queue #19 (+2.69%,
  MAE −16.6%, 0 wash).** h0 .5644 FAIL — min(code,0.4) clamp crushed 111/112 ERROR-bucket
  items (75% of train!) onto ONE flat point while pre-clamp scores meant something (classic
  coarse-bucket zero-resolution, now with the largest measured footprint). REPAIR = replace
  clamp/floor with 3 train-calibrated flat anchors; SEAM PLACEMENT = 2 new contract-required
  detectors; ablation shows levers SUPERADDITIVE (each alone .571-.573, combined .5796).
  Self-adversary caught the mixed-quote detector false-firing on inch-marks (6'2") — fixed
  via digit-adjacency at $0. Rejected+disclosed: precap-blend (never beat w=0), doubled-word
  detector (scanner-polarity trap). Residual ERROR-bucket ceiling disclosed (deliberate
  chat-log at judge 1.0 scores .68). Queue: 19. Cell 28 = CW a72 (.705, FIRST CONTROL-BAND
  cell) launched — controls test whether improvement machinery REGRESSES already-good code.
- 2026-07-12 **CENSUS CELL 27 (CW a99 sentence musicality): PASS → queue #20 (+5.58%) —
  MID-BAND CW CENSUS COMPLETE (27/27).** h0 .6579 FAIL (2 undocumented contract-blocking
  bugs: n<4→0.4 dead-zone floor + a wrong-signed monotony penalty punishing deliberate
  anaphora — the construct's own device; ws65 SUSPECTED class_3 confirmed) → candidate
  .6946 PASS 5/5, code share 34.6%→49.6%. Counter-datum: h0 was 0.66× the judge's
  length-sensitivity (cell 25's was 5.1×) — length confounds run BOTH directions. Self-
  adversary fixed and_chain noun-listing full-credit (clause-vs-list gate) + a too-blunt
  monotony guard; MAE +1.86% unfavorable disclosed + root-caused to compounding h0's
  pre-existing over-scoring on 44% of docs. a99↔a144 deconfliction adversarially verified.
  is_refrain (maturity near-miss) + attributions (mismatch) tested+rejected+disclosed.
  Queue: 20. Cell 29 = CW a90 (.755, control #2, straggler contract) launched. CW remaining:
  a72 + a90 in flight, a144 last.
- 2026-07-12 **CENSUS CELL 28 (CW a72, control #1): CONTROL-CONFIRMED with one real
  repair — the compiled pole characterized.** h0 .6872, code share +54.3% (despite
  majority-L framing) but contract FAIL w/ probe-4 INVERSION — fully decomposed: 14% from
  per-1000-word density normalization (ws65 class-3 SUSPECTED→CONFIRMED) + 86% from a fixed
  last-500-char 'ending' window that covers 100% of any short text (NEW bug: fixed-window
  length interaction). Repair-only fix (raw-count caps at train p90 + proportional window
  that is a no-op on every real doc): INV removed, +0.92%, length-sensitivity HALVED, code
  share →56.7%. The contract's one CODE probe was re-verified corpus-true but measured
  judge-NULL (rho .02; group means indistinguishable) — corpus-present ≠ judge-
  discriminative, correctly declined (divergence datum from the control side). NOT queued:
  4/5 probes independently confirmed L-unresolvable — contract stays FAIL, exactly the
  honest control outcome. COMPILED-POLE ANALYSIS: strong code here = a REGISTER CLASSIFIER
  (crude/shout tells vs abstract/flowing tells); the construct's defining semantic work
  (earned-vs-mechanical, mention-vs-use) stays in the L fields as the contract predicts.
  Dominance 57f/57b/36w exactly balanced, MAE +.006 disclosed. Cell 30 = CW a144 (.864,
  LAST CW CELL) launched.
- 2026-07-12 **CENSUS CELL 30 (CW a144, control #3, FINAL CW CELL): PASS → queue #21 —
  repair-only, and the compiled pole demystified.** h0 .7895 (strongest CW) FAIL (1 INV:
  NEW bug f_ellip — the probe's OWN device, bare-'...' dialogue-silence, penalized as
  melodrama) → candidate .7922 PASS 4/5 via 3 REPAIRS only (ellipsis exemption = sole PASS
  lever at $0; class-3 denominator floor on h0's strongest term = sole rho lever; class-4
  inverted lexicon = $0 validity keep). Every candidate ADDITION was corpus-tested and
  declined — incl. probe0's own headline CODE axis (dialogue-exchange-length: TWO detectors
  both separate the probe cleanly but rho ≤.055 vs judge = corpus-present-judge-null AGAIN).
  COMPILED-POLE PUNCHLINE: even the domain's best code channel (44.5%) is GENERAL
  mechanics/punctuation-density work, not construct-specific detection — 2 probes' SEP is
  coincidental. Self-adversary caught its own as-gate suppressing the penalty on the 2 real
  items it existed for (dominance's #1 broken item) → ratio gate, code-only rho +.01.
  Dominance blast radius near-zero (2f/3b/145w). Queue: 21. CW: 29/30 done, a90 (cell 29)
  in flight = last.
- 2026-07-12 [wake 09:39] 1 agent in flight: cell 29 (CW a90, last CW cell, self-adversary
  phase 09:38 — landing soon). This hour already logged: cells 26-28 + 30 (queue #19 a342
  clamp-crushing-75% find; #20 a99 closes mid-band; a72 CONTROL-CONFIRMED w/ compiled-pole
  = register-classifier; #21 a144 repair-only + compiled-pole = general-mechanics
  punchline). Queue 21; CW 29/30. NEXT MAJOR STEP staged: on a90 landing → CW domain close
  synthesis + held-out promotion batch (test extraction + G1, ONE batch, main-loop-audited
  script not a free-form agent). No GPU, OSL untouched. Holding.
- 2026-07-12 **CENSUS CELL 29 (CW a90, control #2): PASS → queue #22 — and CW DOMAIN
  COMPLETE (30/30 cells).** h0 .8309, code share 93.9% (most code-dominant CW criterion)
  but 0/5 + 1 INV from an n<30 dead-zone cliff. Candidate .8229 (−0.96% DISCLOSED regression
  on a strong control; MAE IMPROVES −1.07% — rank and error disagree in sign, root-caused:
  4/5 broken items are paratext-tail docs). Key discipline: the contract's tail-breaks-
  atmosphere theory was TESTED (rho(has_tail,judge)=+.047 null) → shipped the narrower
  construct-honest design with NO artificial penalty; a class-3 windowed-max fix was built,
  quantified, and REJECTED for costing real rho. 2 gaming holes fixed $0 (density ceiling +
  prose-plausibility ramp: attack 0.68→0.15); ws65 class-4 largely REBUTTED on this file.
  COMPILED-POLE: a90 compiles because its core relation is LEXICAL (multisensory vocab
  density = exact lexicon+density match); 'judicious' + embodied-movement stay MISMATCH.
- 2026-07-12 **CW CENSUS SYNTHESIS (30 cells).** Outcomes: 22 QUEUED (contract-PASS incl. 3
  pass-at-cost + 1 control-with-regression), 6 contract-fail theory-consistent cells, 1
  process-REJECT (a54), 1 stop-and-flag (a216 v1-gap → re-run PASS). h0 contract verdicts:
  only 2/30 h0s passed their own contracts. Instrument harvest: bug classes 6-8 + 6 named
  pathology families (type-coercion credit, flat-penalty flattening, dead-zone curves ×3
  shapes + fixed windows + categorical middles, register collision, negation blindness,
  wrong-signed markers); 3 frozen-contract errata; 4 library maturity items + 2 adoptions
  + 1 partial; wall taxonomy: relation-mismatch / maturity-limited / matched-but-
  unaffordable; corpus-present-vs-judge-null divergence ×3. Seam-shift series (code-only
  share, signed): −8.8→+13.4 (a297), 2.8→22.2 (a234), 32→55.5 (a45), 34.6→49.6 (a99),
  42.9→44.5 (a144), 93.9 (a90). NEXT: CW held-out promotion batch (22 candidates; test
  extraction + G1; main-loop-audited script).
- 2026-07-12 **CW HELD-OUT PROMOTION BATCH LAUNCHED** (22 candidates; cert_census_cw.py
  adapted from the audited cert_agentic.py; new-field test extraction via GLM where needed;
  report → census/cw_heldout_report.json). **HUMOR DOMAIN OPENED**: 7 v1-era stragglers
  identified up front this time (a117 a189 a216 a315 a342 a351 a90 — incl. the queue's
  first two floors); straggler packs built + contract batch launched BEFORE any cell hits
  the gap (a216 lesson institutionalized). Humor census cell H1 = a279 (.071 floor, v3
  contract) launched.
- 2026-07-12 **CW HELD-OUT PROMOTION BATCH COMPLETE (22 candidates, cert_census_cw.py,
  n_test=99-100): 4 PROMOTED / 6 WASH / 10 REGRESSED / 2 AMBIGUOUS-low-judge-coverage.**
  PROMOTED: a135 (P=.949, test Δ+.083 — the length-confound fix TRANSFERS), a207 (P=.909,
  Δ+.069 — the new-LLM-fields seam-placement cell TRANSFERS), a144 (P=.954, Δ+.003), a90
  (P=.967, Δ+.009 — its train regression INVERTED on test: the construct-honest design won).
  HEADLINE HONEST RESULT: train gain does NOT predict transfer — a171 (+11.6% train, queue
  #1) tests at −.009/P=.30; most REGRESSED deltas are wash-magnitude (−.008 to −.024) but
  with low P. Pattern in the transfers: cells that fixed a DIAGNOSED CONFOUND (a135 length
  echo) or added NEW INFORMATION (a207 structured fields) promote; lexical/detector tweak
  cells wash or slightly regress — consistent with the compiled-pole finding (general-
  quality code was already saturated; only new signal or removed bias transfers). a216/a81
  AMBIGUOUS on low test judge coverage (flagged, not read). Winner's-curse gaps largest
  where train gains were engineered latest. Report: census/cw_heldout_report.json.
- 2026-07-12 [held-out batch addendum] Two-bar nuance: G1-vs-frozen-codegen-baseline passes
  a342/a99/a144/a90 (mostly DISJOINT from the P>=.90 set); only a144+a90 clear BOTH bars.
  a135/a207 beat h0 convincingly but not the absolute codegen gate. Largest winner's-curse
  gap: a207 +.170 train→test (still promoted, flagged). a216/a81 low test judge coverage =
  PRE-EXISTING judge-store gaps (44/100, 58/100), not backfilled without sign-off.
  Test-field extraction: 300 GLM calls 0 errors, cache 18200→18500 rows 0 dupes, artifacts
  quarantined in census/_heldout_batch/ (not cell dirs). Sealed-split discipline held.
- 2026-07-12 **USER DIRECTIVE: shift census outside CW/humor.** Launched 3 first-of-domain
  census cells in parallel: MATH M1 (a54, lowest v3-contracted math criterion; SymPy-family
  library tests mandated), PEER P1 (a0 methodological rigor; genre warning + dead-L-field
  audit mandated), LEGAL L1 (a26 90-day timeliness — deliberately chosen as the census's
  cleanest EXACT-relation-match test: date_chain()/deadline_satisfied() were built for this
  construct family; WS4 found legal date code deep-but-signal-poor, this tests whether it
  compiles under the census protocol). All 3 briefs carry the CW transfer law (chase
  confounds + new information, not rho). Humor continues in background (straggler contracts
  + H1 a279 in flight). Remaining unopened: ssdis, PR census cells, coding stage-3.
- 2026-07-12 **HUMOR STRAGGLERS FROZEN 7/7 → humor 30/30 v2.1; slots installed w/ v1
  backups.** Gates clean, 0 text-only residuals; a342/a90 carry authored-positive corpus-
  thin riders; a216/a90 declared mostly-L (floor-cell context). Notable authoring finds: a
  real natural minimal pair in-corpus (d00809 padded vs d00258 same joke tight) + fresh
  a351 harm anchors. Flags adjudicated benign (stereotype-label senses). Humor census can
  now run its full 30-cell queue without straggler stops.
- 2026-07-12 **PEER P1 (peer_review a0, FIRST peer_review-domain census cell) COMPLETE —
  LARGEST TRAIN-RHO GAIN IN THE PROGRAM TO DATE, CONTRACT-FAIL, theory-consistent.** Genre
  verified FIRST (3 items read: abstracts, judgement=accept/reject held out, not the
  GENREBUG confusion). Freeze sha re-verified exact (9e6e4c071aab match); contracts/
  peer_review__a0.json absent → copied from v3, byte-identical. Class-7 dead-L-field audit:
  NOT starved (a0__justification_quote/unaddressed_flaw both 250/250 populated, NONE-rate
  21-28%, healthy) — ruling out the a29/a34 precedent for this cell. h0 train rho=0.1113
  (n=85), contract 1/5 SEP; component decomposition found h0's OWN structural() term is
  NEGATIVELY correlated with judge (-0.083) standalone, and the flaw_penalty LLM field's
  grounded-only gate (rho+0.007, ~zero) was discarding 86% of a much stronger raw signal
  (rho(flaw_present_ANY,judge)=-0.549, not a length confound). FIX 3 (dominant lever, ~89%
  of gain): loosened the gate from word-overlap grounding to plain presence — grounding is
  construct-appropriate for the literal-quote field (kept unchanged, 0/67 flips) but a
  relation MISMATCH for the absence-type flaw field (grounded flaws are MILDER, judge mean
  .55, than ungrounded ones, .415 — the opposite of what a trust filter should select). +2
  ws65-genre bug fixes (FIX1 substring-collision word-boundary, FIX2 plural-form coverage
  gap, new finding: 14/250 'baselines'-only, 2/250 'ablations'-only). Capability-library
  scan: stat_consistency/number_consistency (task's own flagged "statcheck-style" op)
  CORPUS-ABSENT 0/250 on this ABSTRACT corpus (genre-representability limit, not a relation
  mismatch); fact_density/entities_with_evidence noise-level (|rho|<=0.05) — 5th capability-
  library MISMATCH instance in this program, joining math/a150. Final: train rho 0.1113→
  0.3986 (+258.1% relative, beats CW a333's prior record +54.7%), dominance 30f/13b/42w
  (2.31:1, MAE -12.7%), self-adversary 0 new holes (FIX3 mechanism directly confirmed to
  the exact -0.35 delta; all NONE-variants/malformed-types/quote-gate non-regression
  verified). CONTRACT FAIL 2/5 SEP (up from h0's 1/5) — mathematically capped at 40% since
  3/5 probes are genuinely L-channel per the contract's own boundary_notes (verified: no
  regex variant, including a broader confound-vocabulary scan, moves probes 1/3/4 off TIE
  without probe-targeting/false-positive risk). NOT queued (contract-FAIL, theory-consistent,
  matches CW a279's precedent). New pathology named: GATE-DESIGN MISMATCH on an absence-type
  LLM field, distinct from ws65's 5 tracked classes and the class-7 population bug — a
  candidate 6th bug class. Full detail: census/peer_review__a0/{meta.json,adversary.md,
  rounds.log}.
- 2026-07-12 [main-loop addendum, PEER P1] NEW PATHOLOGY NAMED: ABSENCE-FIELD GROUNDING GATE
  — requiring text-overlap "grounding" on an LLM field that describes what's MISSING is
  structurally self-defeating (absence vocabulary can't appear in the text); ungating it =
  89% of the +258% gain (largest in program). Not queued (2/5, 3 probes genuinely L). Peer
  floor was an ARTIFACT of this gate, not tacitness — mirrors the a216 blindness-
  overestimated finding at far larger magnitude.
- 2026-07-12 **HUMOR H1 (a279 misdirection): floor −.033 → +.458, PASS 6/6 → queue #23.**
  Floor = ARTIFACT-dominant: binary L field collapsed 94% of train into one judge-null
  bucket (h0's own L fields made it WORSE than code alone); wrong-signed 'unjustified
  reveal' bucket; length-echo opposite-signed vs judge (+.60 vs −.09). Fix: graded 0/1/2
  misdirection field (judge-discrimination-tested +.452 BEFORE wiring; one GLM pass) + 
  discourse_position reuse + 2 evidence-driven subtractions. Genuine residual L-ceiling
  disclosed (pun-driven 26% of grade-1). 2nd domain-floor overturned in one day (peer a0
  absence-gate, humor a279 binary-collapse) — floors are mostly INSTRUMENT artifacts.
- 2026-07-12 **LEGAL L1 (a26 90-day timeliness): PASS → queue #24 — the exact-relation-match
  test answered.** h0 .6523 FAIL (2 CODE-probe INVs: real date arithmetic only fires off L
  fields; probe mode saw only a keyword fallback) → candidate: train byte-identical wash
  (0f/0b/150w, root-caused: the new code date-anchoring's success set is a strict SUBSET of
  the L-field path) but CONTRACT PASS + code-only floor +11.15% (69.3→77.0%). Date parsing
  = CODE-native exact; anchor selection = NEW CODE-native capability; which-notice-controls
  = irreducibly L. LIBRARY VERDICT: date_chain/deadline_satisfied are exact-relation matches
  but PARTIALLY REJECTED for production with 3 concrete defects (silent April-31 drop —
  load-bearing on real d00905; missing-year falls to TODAY not epoch; no negative-gap
  guard) — library-maturity items #5-7. Queue: 24 (22 CW + 1 humor + 1 legal).
- 2026-07-12 [wake 10:39] 1 agent in flight: math M1 (a54) writing final meta.json (10:38,
  landing imminently). Shift-day results already logged: peer a0 +258% (absence-gate
  pathology), humor a279 floor→.458 PASS (queue #23), legal a26 PASS (queue #24, library
  exact-match-but-3-defects verdict), humor stragglers frozen (30/30). Queue 24 across 3
  domains. On M1 landing: log + open PR/ssdis census cells per the shift directive. No GPU,
  OSL untouched. Holding.
- 2026-07-12 **MATH M1 (a54 organization/navigation): PASS 3/4 → queue #25 (first math
  entry; +0.69%).** h0 .1917 FAIL (2 probes passed only coincidentally; L fields
  mode-collapsed 94%; code share 85.6%). SymPy family = clean RELATION-MISMATCH for a
  prose-discourse construct (tested on all anchor spans). NEW census pattern:
  EXTREME-CORPUS-RARITY — every headline axis 0-2/150 real hits; the '(a)' regex vs f(x)
  function-notation collision found and avoided. 3 self-adversary holes fixed $0. Opening
  PR + ssdis census cells next per shift directive.
- 2026-07-12 **EXTERNAL REVIEW TRIAGE (5 findings, dispositions logged).** (1) Contract
  harness certifies CODE PATH ONLY — TRUE by design (documented as contract-blindness) but
  the reviewer is right that a207's 'promoted' label implied more: new LLM fields are
  UNCERTIFIED by cf_probes; Lane B probe-time field extraction is the standing fix, now
  priority-bumped. (2) Threshold: the '0.80 pre-registered' detail is inaccurate — worse,
  the promotion rule was DEFERRED to batch eval per prereg and set at ANALYSIS TIME (.90).
  Addendum written into cw_heldout_report.json: G1 is the only pre-registered gate (a144,
  a90 clear both bars); a135/a207 relabeled pairwise-only/exploratory. (3) Multiplicity:
  BH-FDR(.10) applied across the 20 unambiguous tests — addendum records survivors; rule
  FROZEN pre-registered for all future domain batches (P>=.90 AND delta>=0 AND BH<=.10;
  G1 separate). (4) DAG schema checks-but-doesn't-enforce: TRUE — ctx exposes everything;
  WS4 claims narrowed to 'declared + review-verified', executor enforcement queued as
  schema v2. (5) ops retrieval indexes RAW text + FULL corpus incl. test split: flagged
  for audit — no judge-score leakage, but text-basis inconsistency (ctext is the declared
  scoring basis) and sealed-split TEXT exposure via retrieval ops; affects class-1-family
  h0s; audit queued before any retrieval-using candidate is certified.
- 2026-07-12 **SSDIS S1 (a10 harmless-error, floor): −.176 → +.657, PASS 3/4 → queue #26.
  FLOORS NOW 3/3 ARTIFACTS.** Third distinct artifact mechanism found: starved+wrong-signed
  alt_grounds gate (fires 5/116, judge-mean HIGHER when it fires) on top of code-constant
  blindness (a279-pattern) + 81.6% coarse-bucket collapse. Lever: graded dispositive_weight
  field (pilot rho .76 pre-tested → full-train .665), essentially the whole gain; dominance
  93f/21b, MAE −69%. Discipline: a synthetic 4/4-SEP detector was built, measured 0/250
  corpus, and REJECTED as probe-overfit — under the new frozen promotion rule this cell's
  numbers face BH-FDR at the ssdis domain batch. h0 length-sign INVERTED vs judge (−.11 vs
  +.35). Headline claim maturing: floor-band r_hyb ≈ 0 measures INSTRUMENT BUGS, not
  tacitness — 3 domains, 3 distinct bug mechanisms, 3 large reversals.
- 2026-07-12 **CENSUS CELL PR1 (press_releases a64, "why it matters"): FIRST PR-DOMAIN
  CELL — PASS, → queue #27.** .4900→.5117 (+4.43%), contract FAIL(2/5,1 INV)→PASS(4/5,0
  INV; probe2 'important' mention_only stays a disclosed L-channel TIE, 80%>75% gate). PR
  uses the OLDER v2 harness (items_v1.json/harness.split_ids/analyze_v2, PROGDIR=
  programs_v2) — traced via battery_common.load_ctx before touching any tool, per brief.
  contracts/press_releases__a64.json was absent (first PR contract installed into that
  dir this program); sha 52d2603ef68c matched _pr_domain_freeze exactly. 3 fixes, 2
  removals + 1 narrow new CODE detector, 0 new LLM fields: FIX_FACILIT (dominant, ~87% of
  the rho gain, ablation-confirmed) is a NEW ws65 bug — h0's Spanish-verb stem
  'facilit\w+' (for facilita/facilitar) is a cross-lingual false-friend collision with
  English 'facility/facilities', confirmed 25/250 corpus docs, upgrades the census's
  same-language-only class-5 CLEAN verdict to a documented exception; narrowed to
  'facilita\w*' (still credits genuine 'facilitate(s/d/ing)', 8/250 preserved). FIX_FAMILY
  (fixes the probe-4 inversion) is a plain _PUBLIC recall bug — only had plural
  'families', missing singular 'family' (32/250 vs 14/250 corpus docs); DISCLOSED
  residual — collides with proper-noun/product-line names ("X Family Foundation",
  "modem family"), corroborated on 3-4 real train items via dominance, a capitalization
  guard was considered and rejected (only closes 2/4 residuals). FIX_WHYMATTERS (fixes
  the probe-0 TIE) is a new, corpus-verified (1/250, exact match to the contract's own
  cited d04828) literal-header detector for "why it/this matters"; self-adversary
  confirmed it is presence-only (a hollow-content laundering probe scores the same 0.55
  bump), matching creative_writing__a333's PRESENCE-vs-FUNCTION distinction — disclosed,
  not chased. TASK'S NAMED AUDIT TARGET TESTED AND REJECTED: the "position-blind
  chrome-stripping" hypothesis (5-6 worst-residual items with a large leading nav-menu
  block getting crushed) was investigated with real numbers, not eyeballing — chrome_frac
  is aggregately judge-supported (rho=−0.20, real bucket-mean trend), a damp-coefficient
  sweep found only a noise-level peak (0.5117→0.5127 at best, WORSE at the extreme) —
  disclosed as a genuine tested non-finding. Relation-match: 5 sub-relations, 1 CODE-
  native (header presence, narrow), 3 L-tagged-but-empirically-CODE-resolvable (causal
  chain, stakeholder type, concrete-vs-platitude — this cell's contribution), 1 genuine
  MISMATCH (mention-vs-use significance framing, re-confirmed via self-adversary at 0.443
  vs 0.466 — a 4th independent construct family confirming CW's core-relation-stays-L
  finding). Dominance 4f/4b/142w (raw wash) but MAE favorable (.2172 vs .2203); code-only
  share ~86% unchanged (all 3 fixes are code-path). ops_capability.py checked in full,
  correct absence (no library op matches this construct's core relation). Queue: 27,
  first PR entry. Next: PR cell 2.
- 2026-07-12 **MILESTONE: ALL 7 CENSUS DOMAINS OPEN + first cell complete in each** (CW
  30/30; humor H1; math M1; peer P1; legal L1; ssdis S1; PR PR1 — queue 27). First-cell
  domain sweep verdicts: 3/3 floors = artifacts (3 distinct mechanisms); legal = exact-
  match-library-with-defects + probe-floor-only lever; math = extreme-corpus-rarity
  pattern; PR = plain recall/collision bugs (facilit\w+ Spanish-stem matching English
  'facility' = 87% of gain). Remaining to full census: ~100 cells across 6 non-CW domains
  + coding stage-3 fleet; all future domain batches under the frozen multiplicity-
  controlled promotion rule.
- 2026-07-12 [wake 11:39] 0 agents were in flight after PR1 landed (all-domains-open
  milestone logged) → advanced: launched humor H2 (a306 floor — 4th floor-artifact test)
  + math M2 (a48 — the WEAK-INSTRUMENT-flagged venue-fit criterion, briefed to lead with
  the rider and measure what the judge actually rewards). Queue 27; external-review fixes
  in force (BH-FDR rule frozen, code-path-only certification caveat). No GPU, OSL
  untouched. Holding.
- 2026-07-12 **HUMOR H2 (a306 GTVH script-opposition): floor +0.043 → +0.479, CONTRACT
  FAIL (4/6, not queued) — FLOORS NOW 4/4 ARTIFACTS, first cell where the artifact IS
  fixed but the contract still legitimately fails.** h0 CODE-ONLY channel (+0.187) beats
  h0 FULL blended channel (+0.043) — 4th confirmed instance of the L-channel-makes-it-
  worse pattern (a279, legal_ss_disability/a10, now a306). THREE additive mechanisms:
  (1) code-only construct blindness under extracted={} (a279-pattern); (2) coarse-bucket
  collapse, a NEW compounding form — mechanism-bucket × resolution-bucket (96.7% one
  value) × h0's own brevity_score hard-clip all saturate together, freezing 52.0%
  (78/150) of TRAIN to one IDENTICAL score spanning judge 0.35-1.00; (3) NEW wrong-signed
  bug — h0 scores pun/wordplay HIGHEST (1.0) but pun-type jokes are the WORST-judged
  mechanism category on this corpus (0.656 vs misdirection/ironic's 0.832), replicated
  within the new field's own grade=2 bucket too; (4) NEW negation-blindness — bare
  'punchline' in the positive-word list false-matches inside 'no punchline', overturning
  ws65_bug_census.json's own CLEAN verdict for this file's classes 4/5 (that sweep was a
  code-read, not an empirical check). Lever: new graded LLM field
  script_opposition_grade (0/1/2, ONE GLM pass after a pilot cleared go/no-go, rho=0.453
  alone, 94% of the total gain). Self-adversary CAUGHT AND CLOSED a real 3-round
  laundering hole in its own continuous dialogue-density corroboration term (decorative
  quote-padding on a construct-empty text inflated score +0.08 → +0.06 → +0.04 → 0.0 via
  span-length, then word-count, then a nearby-attribution-verb gate) — the final
  attribution-gated fix ALSO improved train rho (0.4525→0.4790), not merely defended
  against gaming. Tested-and-REJECTED temptation: uncapping h0's brevity clip to chase
  the remaining 2 TIE probes off-tie (rho=-0.001 on real data, pure probe-fishing,
  declined). Contract stays 4/6 SEP (66.7%<75%) because the contract's own author named
  only 1/6 probes code-tractable — theory-consistent with CW a279/peer_review a0's
  precedent that a floor can be BOTH an instrument artifact (fixed) AND a genuine
  contract-fail (irreducibly L-channel) in the same cell. NOT queued. Full detail:
  census/humor__a306/{meta.json,adversary.md,rounds.log,plan.md}.
- 2026-07-12 [main-loop addendum, H2] **FLOORS NOW 4/4 ARTIFACTS** — and a 4th "L-channel
  makes it worse" instance (code-only .187 beats blended .043). New mechanisms: compounding
  triple bucket-collapse freezing 52% of train to one score; wrong-signed pun preference
  (h0 rates pun highest, judge rates it worst mechanism category); "no punchline"
  negation-collision overturning ws65 CLEAN verdicts (classes 4/5). Fix .043→.479 (94% from
  judge-pretested script_opposition_grade field); honest contract FAIL 4/6 (author declared
  only 1/6 code-tractable) → NOT queued, matching the artifact-fixed-but-L-construct
  precedent. Laundering hole closed over 3 gate rounds ($0, rho actually improved).
- 2026-07-12 **CENSUS CELL M2 (math a48, venue-fit, WEAK-INSTRUMENT rider) complete —
  contract INVERSION is mechanism-level, not corpus-thinness; REFRAME delivered per
  brief.** h0 CONTRACT FAIL 0/4 SEP with 3 INV (worst inversion count of the math census
  to date) — root-caused directly: h0's unconditional math-richness signal runs the
  EXACT OPPOSITE direction from the register-matching construct (more jargon = higher
  h0 score; contract wants plain-language-matches-request = higher). Corpus-check found
  the rider's own claim UNDERSTATED for 2/4 probes: their sole real anchor (d04962) sits
  entirely in the TEST split (train-ABSENT, not merely thin), and probe3's one train
  anchor (d01742, "Advanced"-titled/routine-content) is judge=1.0 — an ACTIVE real-data
  counter-example to the needed polarity, not just thinness. Library check (SymPy +
  NER ops, tested directly): clean RELATION-MISMATCH, 2/2 math cells to date. Shipped 1
  real zero-real-footprint lever (register_match, closes probe0 INV→SEP) + 1 marginal
  real-train floor-soften (+1.66% rho, MAE −16.55%, honest 1-item cost disclosed) →
  candidate reaches 1/4 SEP, still CONTRACT FAIL, NOT queued. Self-adversary (12 cases)
  caught 2 real holes and fixed both at $0 cost: a quoted/reported-speech laundering
  false-fire, and a raw-LaTeX-token-count complexity proxy that inverted on trivial
  arithmetic (a simple fraction identity tokenized HIGHER than the construct's own dense
  jargon anchor) — both required real fixes (quote-masking; switched to a non-basic-
  LaTeX-macro-command count), not just disclosure. Separately: h0's own answers_question
  off-topic gate is a NEW real-train finding — 19% precision (4/21 true) yet net
  rho-positive (removing it: −46% rel) given 96% judge mode-collapse; FOUR independent
  attempts to sharpen it (Jaccard overlap, richness AND-gate, a 30-item GLM-4.7 graded
  topic-relatedness pilot via api_field_runner.py, tag-hedge extraction) all failed or
  were declined — the GLM pilot mode-collapsed to SCORE:10 on 28/30, independently
  confirming the corpus's generosity is construct-level, not one-judge-specific. REFRAME
  delivered: what the judge actually rewards on this corpus ≈ coarse substantiveness +
  same-subfield gate, with ~zero sensitivity to register/genre — falsified head-on by
  d01742. Full detail: census/math__a48/{meta.json,plan.md,adversary.md,rounds.log,
  dominance_supporting.json,self_adversary_results.json,candidate.py,
  topicrel_pilot_prompts.jsonl,topicrel_pilot_results.jsonl}. Next: math M3 per Lane A
  interleave.
- 2026-07-12 [main-loop addendum, M2] a48 weak-instrument rider UNDERSTATED: sole probe
  anchor lives in the TEST split (erratum #4 recorded, with the SYSTEMIC note: contract
  authoring grepped the full corpus incl. sealed-split text — echoes external-review point
  5; future batches grep train-only or disclose split membership). Judge measured as
  near-unconditionally generous (96% of train at 1.0; independent GLM pilot mode-collapses
  to 10/10 → construct-corpus mismatch, not judge quirk). 2/2 math cells = SymPy
  relation-mismatch. Not queued (1/4).
- 2026-07-12 [wake 12:39] 0 in flight after H2+M2 landed (both logged w/ addenda: floors
  4/4 artifacts; a48 weak-instrument characterization + erratum #4 anchor-in-test-split
  systemic note) → advanced: launched ssdis S2 (a3 RFC-evidentiary, .053 floor — 5th floor
  test, brief now includes the ANCHOR-SPLIT CHECK born from erratum #4). Queue 27, errata
  4. Next after S2: legal/peer second-wave + coding stage-3 fleet (needs a dedicated
  build-plan turn). No GPU, OSL untouched. Holding.
- 2026-07-12 **SSDIS S2 (a3 RFC-evidentiary support): −.265 → +.609, PASS 3/4 → queue #28
  — FLOORS NOW 5/5 ARTIFACTS.** h0 CODE-ONLY channel (+0.148) again beats h0 FULL blended
  channel (−0.265) — 5th confirmed "L-channel makes it worse" instance (a279, ssdis/a10,
  humor/a306, now ssdis/a3). FOUR additive mechanisms: (1) code-only construct blindness
  under extracted={} (a279/a10 pattern, 70% of h0's weight — BOTH LLM-field gates, not
  just one — frozen constant under probes); (2) a DOUBLED, more severe version of a10's
  starved-wrong-signed-gate finding: BOTH of h0's LLM fields (its entire 70%-weight
  L-channel) are starved AND backward-signed on real train (rfc_opinion_basis: 15.6%
  populated, judge-mean HIGHER when present [.321 vs .147]; opinion_conflict: 2.8%
  populated — thinner than a10's alt_grounds — also backward-signed, 2/3 real fires
  confirmed misextractions); (3) NEW — the SURVIVING 30%-weight code channel is itself
  near-binary (92.8% of judge-scored train in 2 values) because its citation-marker vocab
  is a stylistic mismatch for this narrative corpus; (4) NEW, not in the 5-class ws65
  taxonomy — a genuine regex RECALL bug, h0's RFC-term detector requires bare word forms
  only, missing gerund forms ("lifting"/"standing"/"walking"/"sitting") this corpus
  actually uses (confirmed 17/33 = 51.5% of h0's own false-zero items). Lever: new graded
  rfc_evidentiary_fit_grade field (pilot rho .70 pre-tested → full-train .486) PLUS a
  re-signed reuse of h0's own rfc_opinion_basis field as an independent corroboration
  nudge (+.124 isolated, large, not redundant) — unlike a10, BOTH shipped levers are
  L-channel by construction here (L2 re-purposes an existing field rather than adding
  code), so 100% of the measured gain is L-channel by design, not merely by outcome.
  Disclosed: grade=0 never fired on the full 150-item pass (plausible structural cause —
  one-sided claimant narrative text + a3 only judge-scored on 56% of corpus); base for
  grade=0 set by design intent, not measured. opinion_conflict (n=3) DROPPED entirely
  rather than narrow-gated (too thin to validate, more conservative than a10's precedent).
  Dominance 69f/10b/4w, 6.90:1, MAE −70.91%. Self-adversary caught+fixed 1 real L3-fallback
  laundering hole (window-scoped bare-phrase guard, $0 train-rho cost) and disclosed 4
  further L3-only limits (0% real-train impact each). ANCHOR-SPLIT CHECK (erratum #4):
  5/7 unique contract anchors (all of probes 0-2's) sit in TEST, only 2/7 in TRAIN —
  matches math/a48's systemic pattern, reported not chased, no leakage risk (TRAIN-only
  execution). Probe 3 (contract's own CODE-tagged "well-supported" mention_only trap)
  stays a disclosed TIE, matching a10's precedent for its analogous probe. Full detail:
  census/legal_ss_disability__a3/{meta.json,plan.md,rounds.log,adversary.md,
  dominance_supporting.json,self_adversary_results.json,candidate.py,
  rfc_fit_prompts_pilot.jsonl,rfc_fit_results_pilot.jsonl,rfc_fit_prompts_train.jsonl,
  rfc_fit_results_train.jsonl}. Next: legal/peer second-wave + coding stage-3 fleet.
- 2026-07-12 [main-loop addendum, S2] **FLOORS 5/5 ARTIFACTS; biggest reversal yet
  (−.265 → +.609, PASS → queue #28); 5th 'L-channel makes it worse' instance.** DOUBLED
  starved+wrong-signed gates (both h0 L fields backward on real data); new taxonomy-
  external recall bug (gerund forms missing = 51.5% of h0's false zeros); code-only share
  of the shipped gain = 0% (purest seam-placement result). ANCHOR-SPLIT CHECK 2nd
  confirmation: 5/7 contract anchors in TEST (systemic, per erratum #4). Cell self-caught
  a false rho reading (field-append race) and logged it rather than smoothing. MAE −70.9%.
- 2026-07-12 **ADDITIVE RECONSTRUCTION-V2 TECHNICAL LANE COMPLETE (initial pass).** Canonical
  terms now separate prompt ARTICULABILITY, code VERIFIABILITY, and frozen-LLM-reference
  ISOMORPHISM; pipeline selection is orthogonal to historical manual/mock/oracle/replay
  provenance; no negative result permits a tacitness claim. New blind compiler + sealed
  evaluator ran Math a144 with no labels/residuals/held-out IDs: 100/100 execution coverage,
  reference available 52/100 (rel1 .859), rho=.066 vs historical hybrid .483. Candidate
  locally separated 3/4 frozen all-L probes but an independent pre-frozen 26-pair adversary
  REJECTED it (14/26 orderings, 33/52 ranges): canonical outcome PROXY_MISMATCH, ineligible
  for certification, held-out may not be reused for confirmation. Technical selected-seed
  replay now covers Math, legacy Code prototype, Patents, Science: current classification of
  frozen telemetry from 800 earlier executions yields 65 pinned/partial/vacuous certificates
  (571 indeterminate; one duplicate ID disclosed). It is not the active coding census;
  full-paper science uses 2,400 records / 1,957 bodies and a document-local BM25 + exact
  bipartite claim/evidence graph to issue 171 STRONG numeric/comparative certificates across
  158 papers plus 431 explicitly WEAK evidence links; y ignored throughout. Patent a34
  oracle-conditioned evidence marginal remains +.661 (not autonomous retrieval). Additive
  instruments: channel-faithful contract checker, enforced typed DAG, corrected v2 capability
  wrapper (7/7 audited counterexamples), immutable paired permutation/bootstrap/BH batch
  certification. Full brief:
  notes/2026-07-12__metric-seam-reconstruction-v2-progress.md. Targeted verification: 66
  unittest + 14 science pytest = 80/80 PASS; blind prepare hashes reverified.
- 2026-07-12 [wake 13:39] 1 agent in flight: the reconstruction-v2 FULL AUDIT (Opus,
  adversarial, 8 targets incl. additivity sha-sweep, blind-a144 re-run, replay/science
  certificate re-execution, external-review closure checks). DELIBERATE HOLD on new census
  cells while the audit reads the tree (avoid mutating shared dirs mid-audit). S2 landed
  earlier this hour (floors 5/5 artifacts, queue 28, addendum logged). No GPU, OSL
  untouched. Holding.
- 2026-07-12 **RECONSTRUCTION-V2 AUDIT COMPLETE** (notes/2026-07-12__reconstruction-v2-
  audit.md). VERIFIED: additivity (hash/timestamp/import forensics); blind a144 END-TO-END
  (every number re-derived independently incl. adversary recount; sealing real; proxy_
  mismatch correct — strongest artifact in the lane); patents reuse faithful w/ caveat;
  FDR machinery; channel-faithful checker; DAG enforcement core. THREE REAL PROBLEMS:
  (1) 'corrected library' fixes only ~2 of 4 named defects — attributions conjunct/action-
  beat + is_refrain floor silently delegate to unchanged v1; 7/7 counterexample flip real
  but coverage overstated; (2) science 'strong' bucket contaminated by a _QUANTITY_RE
  truncation bug ('100k'→'10'; 5/10 sampled strong certs spurious); (3) technical-replay
  prose implies live execution, actually classification of frozen artifacts (metadata
  discloses, prose doesn't). Design seams: DAG provenance defeatable by adversarial node;
  no git baseline for frozen files (audit recommends committing frozen baselines).
- 2026-07-12 [audit final correction] 6c revised DOWN: only 1.5/4 named library defects
  actually fixed — April-31 silent drop ALSO delegates to unchanged v1 (joins attributions
  conjunct/action-beat + is_refrain floor as unfixed-despite-v2-named-functions); none of
  the 3 appears in the 7 counterexamples. 6a nuance: the channel-faithful checker has NEVER
  been exercised on real L-channel data (repo-wide grep: zero files produce the extraction
  format its HYBRID/L gate consumes — correctly built, correctly abstains, untested).
  CONSEQUENCE for our adoption plan: checker adoption for external-review finding 1 is
  GATED on a real-data exercise — which is exactly the Lane B probe-time field-extraction
  job; the two components fit (our extraction produces what their gate consumes).
- 2026-07-12 **RECONSTRUCTION-V2 AUDIT PART 2 COMPLETE** (audit note now 913 lines).
  Conceptual-reframing verdict: NO distortion/downgrade of any runbook-certified result
  anywhere; pattern is OMISSION/undersell, not overclaim. Two deltas beyond my earlier
  summary: (a) peer a214 'manual' label = never a certified result anyway; (b) EXTERNAL-
  REVIEW FINDING 5 (retrieval indexing raw text + full corpus) is actually FIXED in v2's
  split_ops_v2.py WITHOUT being credited — our queued ops audit is partially closed by v2
  code (verify before closing #5 formally). Standing cautions for anyone quoting v2 docs:
  'Code' = old f2p_mock prototype not our pending coding lane; v2 axes are whole-criterion,
  census taxonomy is per-sub-relation — complementary, not substitutable.
- 2026-07-12 **CONCURRENT-THREADS PROTOCOL (user directive).** Other thread implements audit
  fixes; this thread runs trials NON-BLOCKING + EVIDENCE-PRESERVING: (1) BASELINE_MANIFEST_
  2026-07-12.json written (sha256 of all instruments/contracts/h0s/ledgers) — census cells
  must hash-verify their instruments at START and END and STOP on mid-flight drift; (2) no
  git commit performed — the index holds the user's staged work; the frozen-baseline commit
  needs a clean index (user to sanction); (3) census cells write only to own cell dir +
  queue + backed-up field caches; never touch v2 lane files. Resuming census second wave:
  ssdis S3 (a0) + humor H3 (a90 floor, straggler contract).
- 2026-07-12 **RECONSTRUCTION-V2 AUDIT REMEDIATION + TECHNICAL FOLLOW-UP COMPLETE.** Preserved
  blind Math a144 core. Science v2.2 full-corpus correction: historical contaminated 171/158
  preserved but superseded by 136 strong certificates / 126 papers (99 numeric, 37
  comparative); weak tier 435/382; exact identity 129 retained, 42 removed, 7 newly exposed;
  corrected deterministic sample 10/10 plausible relation recurrence (not scientific truth).
  Capability v2.1 completes all named audit defects: v1 0/17 -> v2.1 17/17, including invalid
  date abstention, conjunct attribution, bounded action-beat association, short-refrain
  progression. DAG hardened v3 rejects auditor's ambient-global reproducer before side effects
  (callable provenance guard, not OS sandbox). Finding 5 fresh planted-marker verification
  PASS and formally CLOSED for blind-v2 only: ctext-only, TRAIN-only, self-excluding retrieval;
  historical path unchanged. Real L-channel checker exercise complete with pair-blind frozen
  GLM-4.7 fields: Math a66 HYBRID FAIL (1/4 + mode fingerprint), Math a78 CODE PASS 1/1 but
  HYBRID FAIL 1/4, peer-review a25 CODE PASS 2/2 + HYBRID PASS 4/4 (100% L coverage; one true
  L-dependent pair +.15). Corpus discrimination NOT_RUN, so probe-local only. Per-sub-relation
  records now emit independent articulable/verifiable outcomes and never infer a parent without
  a frozen aggregation. Legacy Code replay prose corrected: old prototype/frozen prior-execution
  telemetry, not active coding census or 800 live runs.
- 2026-07-12 [wake 14:39] 2 agents in flight under the concurrent-threads protocol, both
  healthy: ssdis S3 (a0 — piloting a graded bridge field, v2 pilot iteration at 14:35) +
  humor H3 (a90 — candidate build 14:36). Baseline manifest in force (603 files); other
  thread implementing audit fixes in v2 lane (disjoint write-sets). Queue 28. No GPU, OSL
  untouched. Holding.
- 2026-07-12 **SSDIS S3 (a0 logical-bridge, mid): PASS 4/4 → queue #29 (+33.1%, dominance
  108f/3b/0w = 36:1, MAE −48.8%).** Census firsts: (1) GENUINE NEGATIVE on the graded-field
  lever — two GLM pilots run and REJECTED (v1 mode-collapsed, v2 non-monotonic), root-caused,
  field cache never touched: the pilot gate has teeth in both directions; (2) class-7
  sub-species named: STARVED-BUT-CORRECTLY-SIGNED L fields (why a0 is mid not floor);
  (3) sharpest code-blindness instance: _CITE_MARKER fires 0/250 (dead constant); (4)
  anchor-split does NOT replicate the all-in-TEST pattern (5/8 train — honest
  non-replication of erratum #4); (5) self-caught a wrong-signed marker in its OWN candidate
  pre-ship (rho −.30). Instrument hashes verified START+END, zero drift; detected the other
  thread editing the runbook mid-session and correctly stayed out of it (main loop logging
  this line instead). Shipped lever = reweight correctly-signed fields + 2 corpus-validated
  corroboration terms.
- 2026-07-12 **HUMOR H3 (a90 storytelling, floor): PASS 4/4 → queue #30 (+37.8%, MAE
  −29.2%). FLOORS 6/6 ARTIFACTS — 6th mechanism named: STRUCTURAL-PROXY INVERSION**
  (elaborateness heuristics reward multi-beat stock-joke structure over tight personal
  narrative; independently re-confirms ws65 rambling_reward via a different method).
  COUNTER-DATUM: L-channel HELPS here (code-only .055 vs full .252 — opposite sign from
  4/5 prior floors; story_frame alone .390 beats h0's own blend). New field
  narrative_shaping_grade honestly disclosed as UNDERPERFORMING the existing field alone
  but genuinely complementary (blend .464). INSTRUMENT-TENSION datum: contract_check's
  frac_at_mode gate forced a code-only cap that costs rho (~.46→.35) — the discrimination
  gate itself shapes shipped candidates on L-heavy constructs. ws65 PANEL-CONTROL mislabel
  for this file flagged (pre-existing). Instrument hashes clean START+END. Queue 30.
- 2026-07-12 [wake 15:39] 0 were in flight after S3+H3 landed (both logged: queue 29+30,
  floors 6/6 artifacts w/ 6th mechanism, first genuine pilot-negative) → advanced: launched
  peer P2 (a130 title/abstract structure — CODE-heavy contract) + legal L2 (a44 protected
  activity — the mention-vs-use doctrinal test on the 1-of-8-genuine 'opposed' trap). Both
  under concurrent-threads protocol (hash checks, disjoint write-sets, no runbook writes by
  cells). Queue 30. No GPU, OSL untouched. Holding.
- 2026-07-12 **USER: PAUSE — no new agent launches after current cells finish.** In flight:
  peer P2 (a130) + legal L2 (a44) — these run to completion and get logged; then HOLD (log
  results + wake status lines only, no new launches until the user says resume). Other
  thread continues its v2-lane implementation independently.
- 2026-07-12 **LEGAL L2 (a44 protected activity): PASS 4/4 → queue #31 (+16.13%, MAE
  −17.4%, dominance 11f/2b/137w).** The doctrinal mention-vs-use test delivered: found a
  corpus-scale instance of the trap — bare \bfiled\b scores any mention of the INSTANT
  lawsuit's own filing as prior protected activity (10/11 judge-zero items near ceiling);
  fixed with an uncorroborated-suit gate. Code-only floor byte-identical (opposite shape
  from a26/L1 — the lever here is real-data, not probe-floor). LIBRARY TAXONOMY SHARPENED:
  attributions() mismatch precisely characterized as WHO-SAID-A-PROPOSITION vs
  WHO-DID-AN-ACT (ccomp/xcomp-required vs transitive act-attribution; zero discriminative
  signal on the contract's own minimal pair). ⚠ NEEDS USER SIGN-OFF: 15/150 ADA/ADEA-only
  items have judge mean .857, CONTRADICTING the contract's a0-precedent classification
  (d00291 judge=1.0 while classed as mention-only trap) — the judge appears to treat
  non-Title-VII protected activity as qualifying; candidate takes no position; affects a0
  + a44 contract semantics. 'Opposed' trap itself remains open (upstream extraction).
  PAUSE IN FORCE: no new launches; P2 (a130) last in flight.
- 2026-07-12 **PEER P2 (a130 title/abstract structure): PASS → queue #32 (sign flip
  −.068 → +.192, MAE −13.2%, dominance favorable both views).** Most extreme starved-field
  case of the census: BOTH LLM fields 100% NONE 250/250 — a corpus-REPRESENTABILITY limit
  (abstracts never contain printed title lines), not a pipeline bug; 35% of h0's budget was
  dead constants (and h0 still 'passed' its contract — gate-vs-reality datum). Lever:
  drop dead fields + self-naming detector (+.266 standalone, 71/71 spot-checked) +
  structured-header regex. Round-1 INVERSION self-caught+fixed. Graded-field lever
  evaluated and correctly NOT pursued (code-only sufficed). Anchor-split 9/13 TEST —
  erratum #4 replicates (3rd instance). Queue dedup-verified against the concurrent a44
  append. **PAUSE NOW FULLY IN FORCE: 0 agents in flight; wakes = status/log only until
  user resumes.**
- 2026-07-12 **RECONSTRUCTION-V2 CONTINUATION — BLIND MATH a216 PREEMPTED.** Fresh
  ctext-only compiler proposal passed 30/34 adversarial pairs and 7/10 range anchors but
  failed the frozen per-category floor (semantic target 0/1; subequation grouping 1/3).
  Canonical outcome `proxy_mismatch`; construct fidelity preempted evaluation before the
  held-out LLM reference was opened. Independent CPU replay and finalizer are byte-exact.
  Positive claim is automatic decomposition/program proposal only—not code verifiability,
  isomorphism, or tacitness.
- 2026-07-12 **ACTIVE CODE a104 V3 VERIFIED.** On common heldout n=97, TRAIN-selected
  prompt-compiled baseline rho=.5089; pre-existing deep static/AST checker rho=.6498
  (delta +.1409, Pgate=.5615, Pbeats=.9455; current gate PASS); new relation h0 rho=.6064
  (delta +.0975, Pgate=.3235; sub-gate, no tuning). This is the active coding lane, not
  legacy f2p. H0 provenance corrected to manual/mock retrospective seed with
  label-unreferenced—not label-inaccessible—execution. Independent sanitized rerun: zero
  mismatches; repo-grouped companion supports ordering but is exploratory. Code-over-prompt
  reconstruction does not by itself adjudicate their disagreements. CPU only; no model/GPU,
  repository checkout, or test execution in this run.
- 2026-07-12 **SCIENCE SAME-INPUT PROMPT SMOKE — LITERAL-GUARD V4.** Five serial calls
  only; no full 2,400-paper batch and no GPU. Exact `paper_id+abstract+body` inputs, no `y`.
  Verbatim whitespace-canonical guard accepts 2/5 and rejects 3/5 (BODY evidence absent,
  ABSTRACT claim absent, non-verbatim weak-link evidence). Accepted-only status and
  certificate-presence agreement are each 1/2; strong witness overlap 0/2 prompt/0 code.
  This is an instrument/transport result, not criterion-level articulability/isomorphism.
  Next transport should return source addresses hydrated by code so copy serialization does
  not confound prompt-side semantic selection. Reasoning was requested off, but one response
  reported 12,426 reasoning tokens; hidden reasoning text was not retained.
- 2026-07-13 **CODE-REVIEW RELATION-LOCAL RECONSTRUCTION — HARNESS BUG, RUN
  RECOVERED, CEILING ARM ADDED.** The reported "0 valid / 4,500 contract errors"
  was a **deserialization defect in the runner, not a GLM-5.2 contract failure**:
  `run_hierarchy_prompt_jobs.py` parsed `raw_response` with a bare `json.loads`,
  unwrapping no Markdown fence and rejecting literal tabs inside evidence strings
  (a *non-random* drop, selecting against tab-indented languages). GLM-5.2 honored
  the response schema on ~99% of rows. Replaying the fixed parse over the retained
  raw text recovers **4,442/4,500 valid (98.71%); true contract errors 58 (1.29%)**,
  with **no new model calls**. The test suite had *enshrined* the bug (a case
  asserting that a fenced valid response is a `contract_error`); it is flipped.
  The 12:52 transport smoke had already returned 2/2 contract errors and was
  "excluded from analysis" — a 100% smoke failure is a stop signal, not a row to
  exclude.
  **Corrected result (not a null):** rho is **defined and low**. Median raw
  Spearman **0.146**, 95% clustered-bootstrap CI **[−0.280, +0.623]**; 3/18
  mappings confirmatory, 5 exploratory, 8 without support. Median two-pass
  reliability **0.897** — so the prompt side is self-consistent and the low rho is
  genuine divergence, not noise. Three cells are **negative** (a43 −.300,
  a401 −.201, a15 −.200), and a43 "intention-revealing naming" (R3, −.300) opposes
  a70 "intention-revealing naming" (R1, +.327) — the *program*, not the construct,
  drives the sign, as withheld polarity predicts. The wrong-relation control
  contrast is **undefined (no support)** — a designed control that currently yields
  nothing; diagnose before any specificity claim.
  **Ceiling result → instrument limit; lower arms stopped.** v3 shipped with
  `full_executable_contract_ceiling: omitted`, i.e. no upper anchor. The additive
  `full_executable_contract` arm disclosed the complete program source,
  digest-bound to the executed artifact. Its frozen 4,500-request run completed
  with **4,426 valid responses (98.36%) and 74 contract errors (1.64%)**, all in
  one attempt, with no missing, duplicate, unexpected, or hash-mismatched request.
  All 18 mappings now have confirmatory support (common n=34–102), but median raw
  Spearman is only **.149**, 95% vector-cluster/shared-item bootstrap CI
  **[−.076, +.478]**. Median reliability ceiling is **.905** and median
  ceiling-normalized rho is **.137**. The pre-declared **rho < .40** branch
  therefore applies: GLM-5.2 cannot reliably simulate the program even given
  full source, so this is an **executor/item-panel instrument limit**, not a
  tacitness result. The recovered implementation-summary rho=.146 cannot be
  interpreted as disclosure loss. No lower disclosure arm was launched.
  Execution was local CPU plus the hosted z.ai API; no `sk3` GPU workload was
  launched, and `sk3` GPUs 1–4 remain excluded from continuation work.

  **2026-07-13 correction — the instrument-limit inference above is retracted.**
  The frozen executions and response-accounting numbers remain valid, but the
  `rho < .40` branch did not support its interpretation. The full-source arm
  asks the model to perform exact program execution—including enumeration and
  nonlinear arithmetic—over long diffs, so it measures arithmetic simulation
  rather than a clean articulation-transport ceiling. More importantly, 10/18
  code vectors are dominated by ties: their top and bottom terciles coincide,
  and `a1_simplicity_yagni` returns exactly 1.0 on 102/125 held-out items.
  Median rho=.149 is therefore not a codability estimate, and the recovered
  implementation-summary rho=.146 is not interpretable as disclosure loss.
  The corrected CPU readout reports target resolution and top-versus-bottom
  tercile AUC only for cells with actual spread. Forward work replaces scalar
  scorers with independently authored boolean, witness-bearing verifiers and a
  TRAIN discrimination gate. Failure to certify is bounded non-verification
  for the frozen corpus, verifier class, and budget—not evidence of tacitness.
  The completed CPU replay finds separated target terciles in **8/18** mappings.
  Their median top-versus-bottom AUC is **.573**, descriptive item-bootstrap
  95% CI **[.502, .678]**. The three named cells reproduce (a0=.720,
  a37=.711, R3 a92=.710), but the earlier median .547 and zero-inversions
  shorthand do not: 2/8 AUCs are below .5. Artifact:
  `results/code_review_target_resolution_v1/readout.json`.
  The same CPU-only TRAIN diagnostic over already-existing technical targets
  finds adequate resolution in **14/16 Math vectors**, **6/15 Patent relation
  vectors**, and **1/2 full-article Science views**. Failures are retained as
  corpus/target-resolution results; they are not tacitness claims. Artifact:
  `results/technical_target_discrimination_v1/readout.json`. No prompt batch
  was launched for these domains.

  **Free applicability diagnostic.** Mapping-weighted code outcomes are 51.9%
  scored, 20.0% `not_applicable`, and 28.1% applicable-but-abstained (48.1%
  total unscored). On strict-valid prompt rows, full contract is 57.2% scored and
  42.8% `not_applicable`, versus implementation summary 16.9% scored, 82.9%
  `not_applicable`, and 0.2% `applicable_abstain`. The ceiling's total unscored
  rate is much closer to code, so the summary channel clearly withholds support-
  relevant information; however, the ceiling never reproduces the code's
  abstain state, and its score-order correlation remains below the instrument
  threshold. The previously cited 84.1% shorthand is not reproduced by the
  frozen primary denominators: implementation-summary `not_applicable` is 82.9%
  over valid rows (81.8% over expected rows), while total unscored is 83.1%
  (82.0% over expected). This support diagnostic does not rescue a tacitness
  claim.

  **Specificity control.** Under full contract, the median correct-minus-wrong
  rho is **.148**, CI **[−.180, +.546]**, with 13/18 defined contrasts; the
  interval crosses zero, so specificity is not established. Under the
  implementation summary it remains undefined: median correct/wrong prompt
  overlap is zero, 13/18 mappings have no five-way common support, 16/18 have
  n<10, and the two with n≥10 contain a constant prompt vector. This is a
  support-and-variance failure, not specificity evidence.

  Exact executor command:

  ```bash
  python -m methods.metric_seam.run_hierarchy_prompt_jobs \
    --jobs outputs/metric_seam_pilot/hierarchy_r123/code_review_reconstruction_ceiling_jobs_v1.jsonl.gz \
    --channel full_executable_contract \
    --backend zai_anthropic \
    --model glm-5.2 \
    --temperature 0.2 \
    --max-tokens 1024 \
    --concurrency 3 \
    --expected-jobs 4500 \
    --output outputs/metric_seam_pilot/hierarchy_r123/results/code_review_glm52_ceiling_v1/responses.jsonl
  ```

  Exact analyzer command:

  ```bash
  python -m methods.metric_seam.analyze_code_review_reconstruction \
    --prompt-manifest outputs/metric_seam_pilot/hierarchy_r123/code_review_reconstruction_prompt_manifest_v3.json \
    --prompt-jobs outputs/metric_seam_pilot/hierarchy_r123/code_review_reconstruction_ceiling_jobs_v1.jsonl.gz \
    --responses outputs/metric_seam_pilot/hierarchy_r123/results/code_review_glm52_ceiling_v1/responses.jsonl \
    --code-execution outputs/metric_seam_pilot/hierarchy_r123/code_review_heldout_execution_v1.json \
    --bootstrap-draws 10000 \
    --bootstrap-seed 20260713 \
    --output outputs/metric_seam_pilot/hierarchy_r123/results/code_review_glm52_ceiling_v1/readout.json
  ```

  Claim limits: conditional relation-local reconstruction only; no whole-metric
  codability, tacitness, specificity, external-correctness, or R-level-trend
  claim. Full-source prompting is model simulation, not literal code execution.
  Code/prompt disagreement does not establish code underperformance.
  Artifacts: `results/code_review_glm52_impl_summary_v2_recovered/` (recovered
  readout), `results/code_review_glm52_ceiling_v1/` (ceiling arm),
  `code_review_reconstruction_ceiling_jobs_v1.jsonl.gz`.

- 2026-07-13 **VERIFIER-NATIVE SHORT PATH — CODE REVIEW STOPS; MATH a12
  ADVANCES.** The corrected natural-only TRAIN gate was implemented with the
  shared three-state contract `(not_applicable, satisfied, violated)` and
  mandatory file-qualified witnesses. Synthetic plants cannot rescue natural
  prevalence, completeness, or modal concentration.

  **Code review:** four frozen CUF relation candidates were executed over the
  merged-PR TRAIN corpus using independently authored structured/tree-sitter
  verifiers. **0/4 pass natural corpus measurability.** a0 has only 2/25
  violated applicable occasions; a18 and a38 have zero natural violations;
  a92 applies on only 11/122 completed items and has zero violations. Yet the
  same implementations detect **152/160 planted violations with zero
  inversions**, and both co-run positive controls achieve 40/40 planted
  separation. This is evidence that the verifiers are operative and that the
  post-review corpus lacks the target violations. It is a bounded
  `corpus_unmeasurable` result—not evidence of tacitness or unqualified
  non-codability. Per the frozen STOP rule, no code-review V_llm or held-out
  certificate run was launched. Canonical corrected artifact:
  `outputs/metric_seam_pilot/hierarchy_r123/results/code_review_ast_train_v2/readout.json`.

  **Math a12:** the existing manually constructed SymPy equality-step pipeline
  was preserved and adapted, rather than rediscovered, as a mock pipeline seed.
  The measurement unit is one structurally extracted adjacent equality pair,
  not the whole rigor metric. The operational relation is exact
  rational-expression identity/nonidentity on the algebraic domain inferred by
  the bounded parser; the code does **not** recover a domain declared by the
  document. On 150 compiler-TRAIN documents it yields **443
  natural pairs: 328 not-applicable, 24 exact identities (satisfied), and 91
  exact nonidentities (violated)**. Thus P(applies)=115/443 and
  P(violated|applies)=91/115; all **24/24** available rhs+1 probes flip from
  satisfied to violated with zero inversions. The frozen TRAIN discrimination
  gate passes. Exact nonidentity remains a pair-relation result, not a document
  error without separately established universal claim scope. Artifact:
  `outputs/metric_seam_pilot/hierarchy_r123/results/math_a12_symbolic_train_v1/readout.json`.

  A separately authored/executed Sonnet contract receives all 443 structural
  pairs and never receives SymPy verdicts. Both arms deliberately share the
  code-proposed equality-pair extractor, so the estimand is conditional
  applicability/polarity agreement on proposed pairs—not independent relation
  discovery. Its first 10-call smoke initially parsed
  6/10 because four responses prefixed explanatory prose before one valid JSON
  fence. Production stopped exactly as required. A parser-only v2 replay
  recovers the same retained raw responses to 10/10 with no prompt change and
  no additional call. This is explicitly recorded as **post-smoke parser
  calibration**: the original smoke was 6 valid + 4 contract errors, while the
  v2 CPU replay is 10 valid; parser version was historically outside each
  request digest, so bundle-file hashes and a separate calibration receipt bind
  the revision. The initially compiled two-pass sensitivity bundle was
  then stopped after 69 envelopes: a second stochastic pass is not part of the
  two-implementation certificate and would double runtime. A deterministic
  all-pairs/pass-1 bundle reuses 34 byte-identical completed requests and runs
  the remaining calls at concurrency four, one response ledger row committed
  per completion. The change selects neither pairs nor outcomes—all 443 pairs
  remain in the estimand. No local or server GPU is used. This is still TRAIN
  work and uses Sonnet 4.5 specifically. Witness coincidence is bound by the
  supplied span, and no real witness-ablation rerun has occurred; neither is
  treated as empirical certification. No Math held-out certificate has been
  claimed.

- 2026-07-14 **PROPOSAL-FIRST CONSTRUCT-VALIDITY REPAIR — BOUNDED FIRST
  RELEASE COMPLETE.** The forward workflow is now `PROPOSE → BASE-RATE PROBE
  → AUTHOR/IMPORT → CONSTRUCT CHALLENGE → PER-NODE GATE → SELECT → FREEZE →
  TRANSCRIBE → EVALUATE`. Agreement cannot rescue a unit that fails construct
  validity.

  **Math a12 is reclassified as the falsifying case.** Twelve contextual
  controls were executed against the old symbolic adapter. It scored **0/12**
  correctly: all eight rigorous definitions, hypotheses, constraints, initial
  conditions, and equations-to-solve were called violations because their
  sides were nonidentical; all four genuine rigor defects without an equality
  were not-applicable. The prior 91/91 conditional polarity agreement is
  retained as agreement between two context-free identity implementations
  under a shared construct misconstrual—not validity, certification, or
  whole-metric isomorphism. The narrower rational-expression identity
  capability remains valid and is not destroyed.

  **Fresh prospective Patent antecedent unit.** A proposal-only, detector-blind
  Sonnet 4.5 probe used a fixed hash sample of 32 compiler-TRAIN patent
  documents with full context. Its first transport smoke was **2/5 valid**:
  two responses cited nonexistent line coordinates and one applicable response
  omitted witnesses. Production stopped. A disclosed v2 transport showed
  exact one-based line numbers without changing the proposal, sample, items,
  model, or construct; its smoke passed 5/5 and production completed **32/32
  valid**, yielding **0 not-applicable / 14 satisfied / 18 violated**. This
  licenses code authorship/import only; it is not a verifiability result.

  The preexisting manual depth-3 claim graph was then imported as the draft.
  Its binary relation over all 150 natural TRAIN documents is **1
  not-applicable / 1 satisfied / 148 violated**: P(violated|applies)=148/149,
  above the frozen 0.90 maximum. It also classifies only 6/8 fixed construct
  controls as expected before blind control adjudication, missing explicit CPU
  alias and plural-reference cases. The natural code gate already preempts
  selection, so the planned 150-item prompt transcription and held-out pass
  were not launched. This is a prospective bounded code-degeneracy result.

  **Code review remains corpus-unmeasurable:** 0/4 natural gates pass while
  152/160 plants are detected. The audit-reported a34 dead subtree has no exact
  local node artifact in this workspace and is therefore recorded as a
  provenance gap rather than independently reproduced. Canonical consolidated
  artifact: `outputs/metric_seam_pilot/verifier_pipeline_v2/construct_validity_repair_summary_v1/`.

- 2026-07-14 **VALIDITY-BOUNDS CORRECTION — A12 HEADLINE RETRACTED; A34
  EXACT ATTRIBUTION COMPLETE.** The old Math a12 κ values are retained only as
  a context-free control. Two independent defects preclude the former reading:
  (1) Sonnet received `(lhs,rhs,span)` but never the document, so κ=.445 on
  applicability is a context-withholding artifact; and (2) the shared unit
  misconstrued definitions, hypotheses, constraints, and equations-to-solve as
  rigor occasions, so κ=1.0 is two symbolic-identity implementations agreeing
  on symbolic identity—not adjudication transport, validity, or isomorphism.
  Canonical G2 controls now include both true violations and construct-satisfied
  proxy traps. The frozen symbolic verifier detects 2/2 true errors but fires
  on **4/4 proxy traps**, therefore G2 FAIL. Its bounded algebra capability is
  retained for contextually asserted identity steps.

  Patent a34 now has an exact, TRAIN-only structural readout over all **2,048
  coalitions** of its 11 ablatable nodes. Shapley efficiency holds with residual
  `-3.33e-16` (required ≤1e-9). Absolute Shapley mass is **0.801 evidence**
  versus **0.118 computation** and **0 individuation**; `prior_art_lookup`
  alone has φ=.572. The preregistered practical per-node rule
  `|φ|<.01 AND applies<.10` retrospectively marks five nodes DELETE. The
  declared-DAG certificate enumerates 648 separating cuts and 8
  inclusion-minimal cuts. Every minimal cut passes the empirical DPI check;
  the minimum-information cut is `{norm, prior_art_lookup,
  closest_art_field, distinguishing_field}` at 2.338 plug-in bits versus 1.507
  bits at the program output. This bounds the frozen artifact on its empirical
  TRAIN distribution, never codability in general.

  The old code-review `full_executable_contract` arm is reclassified as an
  **execution** result, not articulation transport. With complete source
  disclosed, GLM-5.2 reconstructs frozen program rankings at median raw
  ρ=.149, 95% clustered bootstrap CI [−.076,.478], and median
  ceiling-normalized ρ=.137. The CI crosses zero and no positive threshold
  claim is licensed. The bounded conclusion is that source disclosure did not
  make this model execute these programs rank-faithfully; this is not evidence
  of tacitness or external correctness.

  Canonical artifacts:
  `outputs/metric_seam_pilot/hierarchy_r123/results/math_a12_g2_validity_v1/`,
  `outputs/metric_seam_pilot/battery/effort_ladder/ws4/patents_pa__a34/readouts.json`,
  and
  `outputs/metric_seam_pilot/hierarchy_r123/results/code_review_glm52_ceiling_v1/readout.json`.

  **A12 full-context arm (preregistered prediction confirmed).** All 443 frozen
  pair/document requests were executed once with Sonnet 4.5; 430 satisfy the
  strict role/witness contract and 13 failures are retained without retry. The
  prompt classifies 203 asserted identity steps, 122 definitions, 44
  hypotheses, 20 equations-to-solve, and 41 other occasions. Of 111 pairs the
  symbolic arm calls applicable on common support, **64 (57.7%)** are
  reclassified as non-asserted occasions (21 definition, 15 equation-to-solve,
  11 hypothesis, 17 other). Applicability κ is **−.052**, versus .445 in the
  now-retracted context-free control. This fall was the frozen prediction and
  measures occasion individuation, not model failure. On the residual 47
  jointly asserted steps, polarity κ is only **.082**: 23 both-satisfied, 2
  both-violated, and 22 symbolic-violation/context-satisfied. Thus even after
  role filtering, context-dependent substitution defeats bare pair identity.
  Role labels remain unsupervised prompt reconstruction, not external truth.
  Canonical artifact:
  `outputs/metric_seam_pilot/hierarchy_r123/results/math_a12_context_train_v1/`.

  **Pipeline inversion replay and node-width estimate.** The frozen [.10,.90]
  pre-authoring occurrence rule would have killed all six known dead units
  before authorship: a34 closest-art 2/250, a34 distinguishing 1/250, and the
  four code-review target violations at 2/122, 0/122, 0/122, 0/122. This is a
  retrospective counterfactual; the rule is prospective for new work. Three
  blind a34 authoring fleets (K=3 frozen before outputs) each proposed nine
  nodes. One unsupervised semantic reconciliation yields 13 observed node
  types; bias-corrected incidence Chao2 estimates 13.8 total and **94.2%**
  coverage. This is a small-K authoring-width estimate only; it neither
  validates constructs nor bounds general codability. Artifacts:
  `outputs/metric_seam_pilot/hierarchy_r123/results/pipeline_inversion_replay_v1/`
  and
  `outputs/metric_seam_pilot/hierarchy_r123/results/patents_a34_capture_recapture_k3_v1/`.

  **A34 matched-information result.** Arm B gave Sonnet the application and the
  exact `PriorArtOps` JSON received by frozen program C; both reconstruct the
  same frozen two-pass Gemma evidence-aware target. Execution completed 100
  requests with 98 contract-valid scores and two retained empty-response
  failures, no retries. On common support, B reaches **ρ=.661** and C reaches
  **ρ=.740**; `C−B=.079`, paired bootstrap 95% CI **[.013,.151]** (5,000
  draws). With target attenuation ceiling .883 and deterministic code
  reliability pinned to 1.0, normalized ρ is .748 for B and .838 for C. This
  licenses the preregistered **algorithmic execution advantage** reading for
  this frozen, mocked/precomputed patent pipeline. It does not establish
  external correctness. Document-only A is descriptive and
  information-confounded (ρ=.278 on its 75-item support). Artifact:
  `outputs/metric_seam_pilot/hierarchy_r123/results/patents_a34_matched_info_v1/`.
