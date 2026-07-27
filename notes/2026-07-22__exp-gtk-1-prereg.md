# EXP-GTK-1 pre-registration — tacit transfer via the channel-difference discipline

Date frozen: 2026-07-22 (before any adapter is trained). Status: FROZEN — edits after this
point go in a dated addendum section, never in the body. Companion:
`notes/2026-07-21__adding-tacit-knowledge-installation-channels.md` §7b (design rationale).

## Question

Does distilling a target model's tacit judgment policies (channel B, weights) transfer
structure the language channel cannot carry — and does that structure generalize beyond the
trained constructs (TK-general), or die at the construct boundary (TK-local)?

## Fixed design

- **Executor:** Qwen2.5-7B-Instruct (below the 12–14B differentiation/rescue floor).
  Replication rung (exploratory): Qwen2.5-14B.
- **Target:** Qwen2.5-72B name-invoked policies (existing frozen npz,
  `family_scores_qwen25/qwen25_72b_name_target`).
- **Readout:** teacher-forced P(YES), adverse-ρ (min over 3 forms), identical template
  (`tacit_breadth_readout_manifest_v1.json::readout_template`); scoring via
  `channels/eval/score_with_adapter.py` (LoRA fork), gated on the zero-adapter acceptance test
  (per-row ρ > .999 vs frozen scorer npz).
- **A (training):** 16 humor constructs from the 7B articulation-FAILURE set
  (gap > .10 ∧ best articulation < .70 at 7B), deterministic selection: order by
  sha256(cell_id), stratified R1/R2/R3 proportional to the failure set; **items = stable-hash
  half-1** (salt `exp_gtk1`).
- **Adapter arms (N=128 rows/construct, same trainer/hyperparams/seed):**
  `real` (target's true policies) / `shuffled` (item-shuffled labels) / `construct-permuted`
  (true target policies, derangement of construct assignment).
- **B batteries (evaluation, item-half-2 for humor):**
  B1 = 20 held-out humor failure-set constructs; B2 = the 7B humor articulation-success set
  (best ≥ .70; small n, reported descriptively); B3 = notice-and-comment failure-set (far,
  procedural); B4 = math-stackexchange (capability-floor control); B5 = exchange-rate
  re-training (N ∈ {8, 32}) on 4 fresh humor constructs, real-adapter vs fresh executor.
  (CW unavailable in the Qwen 3-domain lanes; the 11-domain Llama replication is future work.)
- **Reported quantity:** double difference GTK(B) = Δρ(B|real) − Δρ(B|control), separately per
  control. Cluster bootstrap over constructs. Threshold-free readouts.

## Frozen predictions (directional; confirmatory = P1–P4, P6; exploratory = P5, P7, P8)

- **P1 (installation gate):** `real` raises trained-construct ρ on item-half-2 by ≥ +.15 over
  fresh executor. If P1 fails, HALT — nothing was installed; no transfer claims.
- **P2 (within-domain zero-shot, steep-decay prior):** GTK(B1) > 0 but < half the trained-
  construct gain. (Human far-transfer literature + sibling-locality both predict steep decay.)
- **P3 (dissociation):** GTK(B1) > GTK(B2) — weight-transfer buys most where language fails.
- **P4 (distance decay):** GTK(B3 n&c) ≈ 0 (CI includes 0) and GTK(B4 math) ≈ 0. Ordering:
  B1 > B3 ≥ B4.
- **P5 (meta-acceleration, exploratory):** N-to-reach ρ=.60 on B5 constructs is lower under
  `real` than fresh; small if present.
- **P6 (quadrant):** transfer gains are present in the token-free teacher-forced readout;
  CoT-delta on B1 ≈ 0 for the real-adapter gain (the installed structure is not
  inference-time deliberation).
- **P7 (statability, exploratory):** the trained 7B's self-articulation does NOT reproduce
  its B1 gain in a fresh 7B (transfer gap > half the gain) — tacit-proper.
- **P8 (EXP-COT-0 companion, exploratory):** on below-floor articulation-failure cells,
  articulation+CoT > articulation alone (the capacity floor is partly a COMPILATION floor —
  tokens substitute for missing differentiated policy-slots); partial, not full, rescue.

## Analysis discipline

Rank statistics only; no per-item thresholds. Double differences vs BOTH controls must agree in
sign for any transfer claim. Item-disjointness binding (train half-1 / eval half-2). Tacitness
claims tiered per §7b.4½ (Tier 0/1/2); Tier 2 requires the prompt-subspace cap (v0) or DPI-cap
adaptation. Failure of any prediction is reported as such (feedback: report results not
conclusions; no retro-fitting).

## Stop rules

Acceptance test fails → fix scoring path first, no results quoted. P1 fails → halt transfer
analysis. GPU: 1 GPU total, stacked jobs, targeted kills only.

---

## Dated addenda (body above is frozen)

### 2026-07-22 — v1 P1 OUTCOME: FAIL (reported, halt honored) + v1b dose escalation

- Acceptance gate PASSED (30 rows, min ρ .99998, median 1.0000). vLLM-LoRA × prompt_logprobs
  works on the installed stack; no merge fallback needed.
- **P1 FAILED as frozen:** A-cells (n=16), item-half-2, adverse-ρ: mean gain **+.093** <
  +.15; median +.085; 13/16 positive; 4/16 ≥ +.15. Interpretation recorded at face value:
  N=128/construct × 4 epochs installs a real but PARTIAL policy. **Confirmatory transfer
  analysis (P2–P5) is halted for v1; B-battery grids will be analyzed EXPLORATORY-ONLY.**
- **Descriptive installation-level result (not a transfer claim; cap comparison is per-cell
  and independent of the P1 mean gate):** real-adapter vs prompt-subspace caps, ALL 16 A-cells
  (no selection): **11/16 beat cap_oos; 3/5 saturated-cap cells beaten** (.612>.567, .708>.652,
  .793>.643). These are the program's first per-cell Tier-1.9 tacit residuals: weight-installed
  structure above the (saturated) observed articulation channel's out-of-sample ceiling, on
  articulation-failure cells. Symmetric split discipline on both sides (caps and adapters both
  fit half-1, evaluated half-2). Caveats: cap estimation noise; 11 unsaturated caps are weaker
  evidence; no significance claims.
- **v1b (pre-declared dose escalation, not a new design):** N=512/construct — within the
  §8/prereg N-sweep grid {8,32,128,512} — same trainer, same arms, same seeds policy, same
  batteries. P1 re-tested at the same +.15 bar. v1's P1 failure remains permanently reported.
  Launch after the v1 scoring chain completes (GPU serialization).

### 2026-07-23 — v1b INVALID (training divergence, not a result) → v1c relaunch

v1b's three N=512 adapters all trained through a SILENT NaN divergence (loss → nan at epoch 1
step ~3,500 of 8,192; constant lr, no grad clipping, bf16) and the scoring chain "succeeded"
while writing all-NaN grids. **v1b is void — an infrastructure failure; its P1 was never
measured.** Fixes (permanent): trainer gains warmup + linear-decay schedule + grad-norm clip
1.0 + fail-fast on non-finite loss (never continues past divergence, never saves NaN
adapters); scorer gains a fail-closed guard (refuses to persist non-finite grids). **v1c =
the SAME prereg'd N=512 dose with fixed optimization — an infrastructure repair, not a design
change; the +.15 P1 bar is unchanged.** v1b's pre-divergence checkpoints (step ≤3072) retained
for exploratory dose points only.

### 2026-07-23 — v1 B-battery EXPLORATORY readout (P1 failed → NOTHING confirmatory)

All 9 grids scored. Mean Δρ vs fresh executor (mean-over-forms statistic; point estimates,
no CIs): A-cells real +.059 / shuffled **−.462** / permuted −.107; **B1 held-out humor: real
+.070, both controls negative → double-diff +.147 (vs shuf) / +.169 (vs perm) — the first
GTK-positive exploratory signal, surviving the judging-style control**; B2 +.057 (real
doesn't hurt where articulation works); B4 math cross-domain +.067 real, +.061 vs permuted
(weak far-transfer pulse where steep-decay predicted ≈0). Transfer ≈ trained-gain (+.070 vs
+.059) = consistent with general-factor/shared-structure installation. Shuffled's −.462 on
trained constructs = construct-specific-training positive control. B3 n&c missing (rescue
extraction predates 7B n&c grid — re-extract before v1b analysis). All numbers EXPLORATORY:
quote only with that label; confirmatory versions await v1b P1-pass + cluster-bootstrap CIs.

### 2026-07-23 — M17 fidelity diagnostic (Stanton) on the v1 real adapter: DATA-SIDE verdict

Run on existing grids (free). 16 A-cells, teacher-probability fidelity: **train-items Pearson
.759 / MAE .067 (fresh baseline .127) vs held-out .500 / .103 (fresh .125)**; student
UNDER-DISPERSED (teacher p_yes std .237, student .129 — never learned the confident extremes).
Diagnosis per the Stanton matrix: NOT a representational/capacity wall (on-support fit is
strong), NOT idiosyncrasy-cloning (off-support fidelity is low, not high) — **partial
installation with support-limited generalization → the fix family is data-acquisition**:
on-policy relabeling (DAgger/GKD), active querying, dose (v1b N=512), dispersion-correcting
loss. Representation-level channels (P3) deprioritized on evidence. v1b stays the prereg'd
arm; M19 (one on-policy round) and M20 (KTO loss on identical data) ride alongside as
LABELED EXPLORATORY arms — same batteries, never quoted as confirmatory.

### 2026-07-23 — v1c P1 OUTCOME: FAIL (+.108) — the offline-distillation dose axis is CLOSED

Training (v1c3): all three arms trained cleanly under the hardened trainer (warmup+decay,
grad-clip, batch quarantine): **n_train = 7,488/arm = 468 rows/construct — the ENTIRE
item-half-1; the prereg'd 512/construct exceeds available items, so this is the maximal
offline dose the design permits.** 5 quarantined batches per arm (0.2% of steps, identical
across arms; forensics: same deterministic batches produce non-finite forwards — data-side,
arm-independent). Scoring via the fail-closed LoRA path; grids finite by construction.

Tally instrument: `channels/gtk/tally_gtk1.py` (new; replaces the ad-hoc v1 readout).
Validation: reproduces v1's recorded P1 EXACTLY (+.093 mean / +.085 median / 13/16 positive /
4/16 ≥ bar) from the v1 grids. Artifacts: `outputs/tacit_channels/exp_gtk1/tally_v1c.json`
(+ `tally_v1_repro.json`).

- **P1 FAILED as frozen:** A-cells (n=16), item-half-2, adverse-ρ: mean gain **+.108** < +.15
  (median +.122; 14/16 positive; 6/16 ≥ +.15). **Confirmatory transfer analysis remains
  halted; every battery number below is EXPLORATORY.**
- **Dose-response (the decisive fact):** +.093 @ 128 rows/construct → +.108 @ 468 (3.7× dose
  → +.015). Strongly sublinear, and half-1 is exhausted — **P1 cannot pass by offline dose
  within this design.** This confirms the M17 data-side verdict: the fix family is data
  ACQUISITION (on-policy relabeling M19 / dispersion-correcting loss M20 / active querying),
  not more of the same offline labels.
- **Exploratory battery table (adverse-ρ, cluster-bootstrap 95% CIs; double diffs vs BOTH
  controls):**

  | battery | n | Δreal [CI] | GTK vs shuf [CI] | GTK vs perm [CI] |
  |---|---|---|---|---|
  | A trained | 16 | +.108 [+.04,+.17] | +.641 [+.50,+.78] | **+.202 [+.11,+.30]** |
  | B1 held-out humor | 20 | +.071 [+.02,+.12] | +.504 [+.43,+.58] | **+.025 [−.03,+.08]** |
  | B2 humor success | 6 | +.071 [+.04,+.10] | +.396 [+.32,+.49] | +.156 [+.08,+.24] |
  | B3 n&c failure | 77 | −.030 [−.05,−.01] | +.192 [+.15,+.24] | −.034 [−.06,−.01] |
  | B4 math (all 90) | 90 | +.082 [+.07,+.10] | +.198 [+.18,+.22] | **+.126 [+.11,+.14]** |

- **B4 (math) is the surprise of the table (exploratory):** the humor-trained real adapter
  IMPROVES far-domain math matching (+.082) while permuted HURTS it (delta −.044) and
  shuffled hurts more (−.116) — a monotone TRAINING-COHERENCE gradient (coherent real >
  incoherent permuted > noise shuffled) whose vs-perm double-diff STRENGTHENED with dose
  (+.057 → +.126). Combined with B1-vs-perm ≈ 0 and B3 < 0, the observed vs-permuted
  ordering is **B4 > B1 > B3 — essentially the INVERSE of P4's frozen prediction**
  (B1 > B3 ≥ B4 ≈ 0). Descriptive reading: what varies between real and permuted is not
  construct-specific content (both carry the same 16 policy vectors) but LABEL-COHERENCE of
  the training pairs; coherent pairing appears to transfer as a general judgment benefit
  that is visible where the executor has headroom (math) and invisible-to-harmful where
  domain interference dominates (n&c). No confirmatory standing (P1 halt); flagged as a
  named target for the route-signature arms.

- **v1's headline exploratory signal is REVISED at high dose:** B1's vs-permuted double-diff
  collapsed (+.143 CI-positive at N=128 → **+.025 CI-spanning-zero** at N=468) while the
  TRAINED-cell vs-permuted diff strengthened (+.156 → +.202). Reading (descriptive):
  construct-SPECIFIC structure does install on trained constructs, but what generalizes to
  held-out constructs is the construct-GENERAL component of the target's judgment — and at
  high dose the permuted control (real policies, wrong constructs) installs that shared
  component equally well. The v1 B1 double-diff was partly a low-dose artifact.
- **B3 (far-domain n&c) turns NEGATIVE at high dose** (Δreal −.030, vs-perm −.034, both
  CI-negative): mild interference, consistent with P4's steep-decay prediction plus an
  interference cost invisible at N=128. GTK-vs-shuffled stays large everywhere because the
  shuffled arm degrades the executor more at higher dose (destructive-noise positive control).
- **Line verdict:** EXP-GTK-1's prereg'd arm is complete: P1 failed at both doses and the
  dose axis is exhausted. The line continues ONLY via the pre-declared labeled-exploratory
  arms (M19 on-policy round, M20 KTO loss) and the battery's channel probes.
