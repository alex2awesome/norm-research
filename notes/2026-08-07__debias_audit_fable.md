# Adversarial-debiasing V2 failure: audit, root cause, bottleneck fix — N&C responded

Date: 2026-08-07 (battery landed 2026-08-08T01:22Z). Agent:
claude-debias-audit-fable. Status: **CLOSED — DEFINITIVE NEGATIVE VERDICT (§5)**.
Prior state: `notes/2026-08-06__debias_pilot_nc.md` (pilot + V2 FAIL at every λ).
Spec: `notes/2026-08-05__taste-decomposition-design.md` §9 — BINDING; no
real-cell number is quotable until V1–V4 all pass. None is quoted: the
instrument itself is retired (§5).

Terms unpacked: **GRL** = gradient-reversal layer (adversarial head predicting the
named nuisance channels from the model's internal representation; its gradient is
multiplied by −λ on the way into the encoder). **V1–V4** = the four planted-check
gates (EXPLOIT / REMOVAL / SPECIFICITY / CONSISTENCY). **probe** = fresh 2-layer
MLP trained on frozen representations, scored on held-out eval rows (3 seeds,
mean). **h** = pooled last-token hidden state (what the LoRA model's scalar head
reads). **z** = bottleneck projection (new architecture, §3).

## 1. Audit of the failed instrument (all five checks run, in order)

### 1.1 Mechanics — CORRECT (verified numerically, not by code reading)
One-batch gradient probe (`audit_grl.py`, results in
`debias_pilot/results_audit_grl.json` on sk3):
- cosine(reversed adv gradient, forward adv gradient) over all shared LoRA
  params = **−1.0000000**; norm ratio = λ **exactly** (1.0 at λ=1, .49998 at λ=.5).
- Adversarial gradient reaches all 448 LoRA tensors (224 zero-grad tensors are
  the lora_A matrices, zero only because lora_B=0 at *init* in this fresh-weights
  probe — an initialization artifact, not a routing failure).
- Score-head equivalence: the pooled `hidden_states[-1][:, -1]` used by the GRL
  head and the probe is numerically the vector the model's scalar head consumes
  (max |Δlogit| < 1e-2 in bf16).
The reversal is live end-to-end. Corroborated during real training: the
co-trained adversary's held-out R² on the plant DEGRADES as λ rises
(.853 → .795 → .11 across λ .1/.5/1.0) — the encoder is genuinely fighting it.

### 1.2 λ dynamics — NOT the classic adversary collapse
Adversary loss never collapses (stays .43–1.2 across training in every arm;
curves in each `runs/R0*/result.json` history). The failure is the *other*
known one: the encoder defeats the particular co-trained adversary while the
information stays perfectly present for any fresh reader (Elazar & Goldberg
2018 / INLP line), reproduced here with known ground truth.

### 1.3 Probe validity — VALID
Positive control: probe on planted-vanilla (R01) rep = 1.000. Negative
controls: unplanted arms read the plant at .525–.531 (chance). Probe fits on
train rows, early-stops on a train holdout, is scored on eval rows only; fresh
head per run × 3 seeds. A linear probe replicates the MLP probe
(`results_linear_probe.json`: .998–.9998 on all GRL arms) — and still reads the
plant at ~.99–1.0 *within each y stratum*, so it is reading the plant itself,
not y leakage.

### 1.4 Plant visibility — 100%
`⟦QX7⟧` tokenizes to a stable 7-token prefix and survives right-truncation on
**5,597/5,597 (100%)** of planted rows (`results_audit_tok.json`). V1's exploit
signal was real.

### 1.5 The LoRA blind spot — CONFIRMED (root cause)
The dense standard trains rank-16 LoRA adapters over a FROZEN Llama-3.1-8B;
h = frozen-base activations + low-rank deltas, and the scalar head reads h
directly (score-head equivalence verified exactly: max |Δlogit| = 0.0). Probe on
the FROZEN BASE model's h (no adapters, no training, same layer, same rows,
`runs/BASE_frozen_planted/`):

| channel | frozen-base probe AUC (eval, 3-seed mean) |
|---|---|
| plant `⟦QX7⟧` | **.955** |
| char_len | .964 |
| docket_year | .783 |
| realtok `⟦RS4⟧` | .620 |

The plant (and the length channel) is carried near-perfectly by the frozen
substrate BEFORE any training. "Stop encoding it" is therefore not an available
move — GRL-on-LoRA would have to actively CANCEL frozen-substrate information
with rank-16 deltas at every planted input, and it never does (probe .997–.999
at every λ; training actually sharpens plant readability, .955 → 1.000).

### 1.6 The remedial escalation kills the alternative explanation
R05B (previous agent's coordinator-bounded attempt, landed today): plant as the
ONLY adversary target (27× gradient concentration) at λ=5 (5× the specced max):
- fresh probe on plant: **1.000** (unmoved);
- task eval AUC **.679** vs vanilla .793 (instrument destroyed);
- plant-ablation Δ(eval) **+.102** (vanilla +.028) — under maximal GRL pressure
  the model relies on the plant **3.7× more**, not less.
So the failure is not dilution and not insufficient λ: on this architecture there
is NO operating point where GRL removes the channel — pressure only destroys task
signal while *increasing* plant reliance. Monotone across λ .1→.5→1→5:
ablation Δ .0275 → .0325 → .0363 → .0500 → .1023.

**Root-cause verdict: architectural. GRL is mechanically correct but has no
parameterization available that removes frozen-substrate information; the
min-max game settles on "defeat the co-trained adversary, keep the channel".**

## 2. Why the old V2 could never pass (summary table, old arch = pooled h)

| arm | λ | eval AUC | fresh plant probe | plant ablation Δ(eval) | co-trained adv R² (plant) |
|---|---|---|---|---|---|
| R01 vanilla planted | — | .8088 | 1.000 | +.0275 | — |
| R02 | 0.1 | .8024 | .999 | +.0325 | .853 |
| R03 | 0.5 | .8122 | .997 | +.0363 | .795 |
| R04 | 1.0 | .7966 | .999 | +.0500 | .110 |
| R05B plant-only | 5.0 | .6785 | 1.000 | +.1023 | .125 |
| frozen BASE (no adapters, no training) | — | — | .955 | — | — |

## 3. The fix: bottleneck projection (arch="bottleneck")

`train_grl.py` now supports `"arch": "bottleneck"`: a trainable projection
z = proj(h) (4096→256) is inserted between the pooled representation and BOTH
heads — the task head is a fresh linear on z, the adversary reads z, GRL shapes
proj (and, through it, the adapters). Design points:
- **Removal is now in the trainable span**: proj is a full-rank map whose input
  includes the frozen-substrate plant direction; zeroing it is a single linear
  operation, reachable by gradient descent — unlike cancelling it inside h.
- **The certificate is scoped to the decision path**: the score is a function of
  z alone, so "plant unreadable from z" ⇒ "the score cannot use the plant".
  Information may (and will) persist in h; that is reported as a scope note, not
  hidden: the probe target is z (`reps.npz[rep]`), with a secondary probe on h
  (`rep_h`) recorded for the GRL arms.
- Task capacity is unchanged in kind: the original head is linear in h; head∘proj
  is also linear in h (rank ≤ 256 ≫ 1). Vanilla-arm parity is checked (B00 vs R00).
- Everything else frozen: same recipe, same splits, same corpora, same probe
  protocol, same paired docket-level bootstrap. λ sweep {.1, .5, 1.0} per spec.
- Full fine-tune noted as the theoretically clean but expensive fallback
  (≈16× adapter cost, new recipe departure) — not exercised.

Battery rerun (B-arms, tags mirror the R-arms): stage B1 = B00/B01 (V1) +
B02–B04 (V2 sweep) → auto-gate (`gate_bn.py`: probe ≤ .55 with B01 control
≥ .75 [primary] + |ablation Δ| ≤ .01 [causal] + spec-literal AUC-within-.005
[reported; noise-limited, CI ±.03 at n=953]) → stage B2 = V3a/V3b/V4/final at
λ*. Orchestration `run_battery_bn.sh`; one ledger-claimed GPU (GPU 7, claim
2026-08-07T19:59Z after R05B release; battery start 20:02:49Z).
Smoke test (B99, 360 rows): architecture runs end-to-end — z=(·,256) probed,
h=(·,4096) kept for scope, best_bn.pt checkpoint/reload, ablation + adversary
diagnostics intact.

## 4. Bottleneck battery results (landed 2026-08-08T01:22Z, gate_rc=3)

All arms: identical rows/splits, paired docket-level bootstrap (2,000 resamples),
select-on-eval. Vanilla parity check: B00 eval **.8283** / test .711 vs pooled
R00 .7926/.704 — the bottleneck costs nothing (it trains slightly better).
Probes are on **z**, the 256-d vector the score head reads; h-probes reported
alongside for scope.

| arm | λ | eval AUC | Δ vs B00 [95% CI] | z-probe (plant) | h-probe (plant) | plant ablation Δ(eval/test) | co-trained adv R² (plant) |
|---|---|---|---|---|---|---|---|
| B00 vanilla real | — | .8283 | — | .587 (ctrl) | — | — | — |
| B01 vanilla planted | — | .8466 | +.0183 [+.003, +.034] | .990 | — | +.0184 / +.0555 | — |
| B02 | 0.1 | .8543 | +.0260 [+.007, +.046] | .980 | .996 | +.0204 / +.0513 | .678 |
| B03 | 0.5 | .8510 | +.0228 [+.000, +.047] | .970 | .993 | +.0205 / +.0648 | .201 |
| B04 | 1.0 | .6898 | −.1384 [−.197, −.083] | **1.000** | 1.000 | **+.1228** / +.0848 | −.130 |

- **V2 REMOVAL under the fixed architecture: FAIL, 0/3 λ, on every gate.** The
  planted-vs-unplanted jump never vanishes (it *grows* at λ=.1/.5); the fresh
  probe reads the plant from z at .97–1.00; causal reliance (ablation Δ) is
  unchanged at low λ and **6.7× amplified** at λ=1.0 while the task loses .14
  AUC. The co-trained adversary is progressively defeated (R² .678 → .201 →
  −.13) — the exact pooled-arch signature, reproduced where removal is
  architecturally trivial (nulling the plant-predictive directions of a 256-d
  trainable linear projection).
- λ=1.0 is also unstable under the bottleneck (adv-loss spike to 361, eval AUC
  dip to .345 mid-training; best checkpoint = end of epoch 1).
- V1 note: the literal eval statistic lands at +.0183 [+.003,+.034] — below the
  .02 bar, as the pooled battery's +.0162 was. Exploit substance is
  unambiguous (B01 z-probe .990, test ablation +.056), but on this cell the
  independent-run eval difference is under-powered for a .02 bar at n=953;
  any future battery should gate V1 on the within-model ablation readout.

| gate | old arch (pooled h) | new arch (bottleneck z) |
|---|---|---|
| V1 EXPLOIT | PASS substance; literal marginal (+.0162) | PASS substance (probe .990, test abl +.056); literal marginal (+.0183) |
| V2 REMOVAL | **FAIL** every λ (probe .997–1.000) | **FAIL** every λ (probe .970–1.000; reliance amplified at λ=1) |
| V3a SPECIFICITY | not run (moot after V2) | not run (auto-gate stopped the chain, correctly) |
| V3b SPECIFICITY | not run | not run |
| V4 CONSISTENCY | not run | not run |

## 5. Verdict — definitive negative (per the bounded-budget brief)

**GRL cannot certify removal on this program's dense standard, at any operating
point, on either architecture.** Two-layer root cause, each layer independently
evidenced with ground-truth plants:

1. **The original instrument was unfixable by construction** (LoRA blind spot):
   the score head reads pooled h = frozen-base activations + rank-16 deltas, and
   the frozen substrate alone exposes the plant at probe .955 before any
   training. "Stop encoding it" was never an available move.
2. **Repairing the architecture does not repair the procedure.** With a fully
   trainable 256-d projection feeding both heads — where a linear map nulling
   the plant direction demonstrably exists — the GRL min-max game still settles
   on *defeating its own adversary while keeping the channel* (fresh probe
   .97–1.00), and under pressure *increases* behavioral reliance on the channel
   it is supposed to remove (ablation Δ +.018 → +.123) while destroying task
   signal. This reproduces Elazar & Goldberg 2018 / the INLP line's negative
   result inside our own instrument, with known ground truth, on both
   architectures, across λ .1–5.0 and a 27×-concentration remedial arm.

Consequences:
- Adversarial representation debiasing is **retired as a removal-certification
  instrument** for the taste-decomposition program. The §9 conditional approval
  is resolved NEGATIVE: the planted battery did exactly its job — every AUC-only
  reading would have been a false PASS (λ=1.0 pooled: +.0039 "removal" with the
  plant at probe .999).
- **Instruments of record for spurious-influence control remain the two ADOPTED
  ones: stacked-increment readout + matched sampling** (design §9), already
  frozen and running; nuisance stratification stays as the Layer-2 appendix
  readout.
- If a removal-style certificate is ever genuinely needed, the literature-
  standard candidate is post-hoc linear projection (INLP/LEACE-style) on the
  frozen representation — a closed-form operation with no min-max game — but it
  would need its own planted battery and is NOT proposed as work now.
- Do not quote any debiased real-cell number from either battery; none exists
  (V3/V4 never ran; the gates stopped the chain by design).

## 6. Artifacts
- Audit: `methods/taste_decomposition/debias/audit_grl.py`; sk3
  `debias_pilot/results_audit_grl.json`, `results_audit_tok.json`,
  `runs/BASE_frozen_planted/` (frozen-base reps + probe).
- Fix: `methods/taste_decomposition/debias/train_grl.py` (arch=bottleneck),
  `gate_bn.py`, `run_battery_bn.sh`, `make_configs.py --arch bn`,
  `probe_reps.py --rep_key`, `analyze_battery.py --arch bn`
  (→ `results/battery_bn.json`).
- Runs: sk3 `datasets/notice-and-comment/debias_pilot/runs/B*` (weights/reps);
  local mirrors of result/probe/preds under
  `methods/taste_decomposition/debias/runs/B*` and `runs/R05B_grl_plantonly_l5.0/`.
- Gate + battery readouts: `debias/results/v2_gate_bn.json`,
  `debias/results/battery_bn.json` (paired docket-bootstrap CIs).
- GPU ledger: GPU 7 claim 2026-08-07T19:59Z → RELEASE 2026-08-08T01:22:54Z
  (gate_rc=3); no GPU held at close. GPU 3 claim 19:51:58Z retracted 19:57Z
  (lost race to claude-closure-patents).
