# Form-effects control plan — decomposing FORM-DOMINATED into instrument vs linguistics

*Written 2026-07-01 after the CW Day-0 certificate read (36/42 FORM-DOMINATED, median paraphrase
drift 11%, flip rate 12%, vs same-form τ₀ ≈ 0.2%). Status 2026-07-02: §3 patches + τ₀
carry-through + `--form-invariance-n` + `form_decompose.py` IMPLEMENTED, fake-smoke green;
aligned_8b forminv copied (§2). Remaining: 3B forminv pass (dir turned out NOT sig-identical),
70B/Qwen relaunch with the flag, M_i_forms backfill. The gate redesign (§6) and any change to a
certified quantity still need explicit sign-off first.*

## 0. Correction that motivates this note

The form gate in `decide()` is `form_invariant = median(binary_flip_rate) ≤ 0.10` — a **fixed 10%
median-probe-flip bar**, NOT `drift > 2·τ₀` as earlier stated (`frac_over_2tau0` is a separate
reported diagnostic, not the gate). CW median flip = 0.122 > 0.10 ⇒ **the 36/42 count is genuine**,
but the margin is thin (12.2% vs a 10% bar), so the binary verdict manufactures a cliff where the
data shows a distribution hugging the bar. Quote magnitudes (11% drift, 12% flips, ~55× τ₀), not
counts.

## 1. Form dependence has TWO seats — the in-flight fix covers only one

| seat | what is form-fragile | what it corrupts | fix status |
|---|---|---|---|
| **Target** | the metric rubric → M_i readout | M itself; every T/R | **IN FLIGHT** — `--orbit-target 4` m̄_ω (§12.6.2), Φ-invariant by construction |
| **Instruments** | the Ω criteria prompts → signatures σ_c | species partition, B_E, OPT_Ω head | **NOT covered** — `form_invariance` samples criteria prompts; `--orbit-target` never touches them |

The Day-0 FORM-DOMINATED verdicts came from the **instrument seat** (run_alpha_probe's
`form_invariance` samples 12 Ω prompts × ~3.5 reformulations = 42 pairs). Orbit-averaging the
target does not change that number at all.

## 2. Two traps in the aligned read (flag before Day-3, or the result self-flatters)

1. **The gate silently vanishes.** `value_certificate` looks for `*_forminv.json` next to each
   ckpt; the `aligned_*_orbit` dirs (rescore_executor output) contain **none**, so
   `form_invariance=None` → the gate is skipped, and FORM-DOMINATED metrics will re-emerge as
   CODIFIABLE **because the gate lost its data, not because form was controlled**. Per the
   anthropological framing, instrument weakness inflates the desired gap — this is exactly the
   self-flattering failure the positive controls exist to prevent.
   *Fix:* for same-executor dirs (aligned_8b_orbit) the source forminv applies verbatim — copy it
   over; for cross-executor dirs (70B/Qwen) mark "criteria fragility unmeasured at this scale" in
   the report, or add an optional `--form-invariance-n` pass to rescore_executor (cheap: 12
   criteria × 3 forms × 300 probes).
   **DONE 2026-07-02:** 41 forminv copied into `aligned_8b_orbit` (sigs verified identical, 3/3
   spot-checks) + `FORMINV_PROVENANCE.txt`; `--form-invariance-n` implemented (smoke green).
   **Surprise:** `aligned_3b_orbit` sigs are NOT identical to src (0/3) — despite the supervisor
   log calling it a retarget, the 3B dir re-scored criteria with the 3B executor, so the 8B gate
   data does NOT apply; it needs its own forminv pass (~500K 3B scores, ~1–2 GPU-h) before its
   certificate read is gate-valid.
2. **Hardcoded τ₀.** rescore_executor saves `tau0=0.05, tau=0.05` literals — not the measured
   floor (llama8b measured τ₀≈0.002). Any downstream consumer of aligned τ₀ inherits a 25×-inflated
   noise floor. Harmless for the flip-rate gate (which doesn't use τ₀), wrong for drift-vs-τ₀
   diagnostics. **Fixed in code 2026-07-02** (τ₀/τ carried from the src ckpt; smoke shows 0.00207
   carried). Already-written 3b/8b ckpts keep the stale literal until the §3 backfill rewrites them.

Ops flag (same sweep): TWO identical `run_alpha_probe` 16-fill processes were live on sk3 at
review time (PIDs 613059 & 621422, same gi-list, same out-dir, both `--skip-existing`) — they race
per-metric and double GLM spend. Left untouched (other agent's launch); user should adjudicate.

## 3. Data-preservation patches (IMPLEMENTED on sk3, 2026-07-01)

Both are additive save-format changes; no measurement target changed. Running procs (loaded
modules) are unaffected; future launches pick them up.

1. **`rescore_executor.py`**: the orbit branch now saves `M_i_forms` (the raw n_forms × n_probes
   per-form readout matrix `orbit_metric_verdict` was computing and discarding) + `M_i_form_names`.
   Without the matrix, the main-effect/interaction split (§4) is unrecoverable from the scalar
   `M_i_var_phi`/`M_i_flip_rate`.
2. **`alpha_probe.py::form_invariance`**: now returns `pairs` — per (criterion, form) records
   `{criterion, form, drift, flip, bias, yes_shift}` where `bias` = signed mean shift (the form
   MAIN effect per pair) — so we can attribute **which linguistic transformation** (question /
   boilerplate / reorder / suffix) breaks readouts, which the medians aggregated away.

3. **`form_decompose.py`** (NEW, CPU): the §4 decomposition as a runnable script — target seat
   raw→calibrated flips (per-form quantile YES-rate matching) + boundary concentration; instrument
   seat per-transformation drift / flip / |bias| / main-effect-share table. Validated on FakeVLLM
   smoke data (both seats) and graceful on pre-patch dirs.
   `python -m methods.metric_implementer.experiments.form_decompose --dir <ckpt_dir>`.

Backfill: 3B/8B orbit dirs finished under the old format. A `--retarget-mi-only --orbit-target 4`
re-run over a finished dir reuses its sigs and recomputes only M (≈49K short scores for 41
metrics) — queue AFTER the 70B/Qwen rescores free their GPUs (1-GPU rule).

## 4. The decomposition — the sharpest cheap analysis (CPU, post-hoc)

Total form effect = **main effect** (a phrasing is uniformly stricter ⇒ a threshold shift,
calibratable away, instrument artifact) + **item×form interaction** (WHICH items pass depends on
phrasing ⇒ genuine entanglement — the linguistic quantity: the words do not pin the concept).

Procedure, per metric, on `M_i_forms` (target seat) and on `pairs` (instrument seat):
1. Per form φ: recenter (subtract per-form mean, or equalize binarized YES rate), re-binarize,
   recompute flip rate vs m̄.
2. `flip_calibrated ≪ flip_raw` ⇒ mostly strictness artifact → **report calibrated numbers and
   the FORM-DOMINATED story shrinks**. `flip_calibrated ≈ flip_raw` ⇒ genuine entanglement →
   **the fragility IS the finding** (CW rubrics are paraphrase-unpinnable).
3. Flip concentration vs |P(YES)−0.5|: boundary-concentrated flips = benign threshold noise;
   diffuse flips = semantic instability. (Do NOT filter to confident probes for certification —
   boundary items are where taste lives; concentration is a diagnostic only.)
4. Per-form attribution from `pairs.bias`/`pairs.form`: rank transformations by damage; test the
   multi-clause hypothesis (reorder exists only for multi-clause criteria — do they flip more?).
5. Cross-domain: same analysis on the peer-review forminv files → is CW *distributionally* more
   form-fragile than PR, or does PR just sit under the bar?

**FIRST RESULTS (2026-07-02 00:40, corrected v2 dirs, `_log/form_decompose_{8b,3b}_v2.json`):**
- **TARGET seat (46 metrics, 8B):** raw single-form-vs-m̄ flip median **13.0% → 6.2% after
  per-form quantile calibration — calibration removes 52%**. 3B: 13.0% → 6.5% (50%). So roughly
  HALF the form effect is a calibratable strictness shift (instrument artifact); the surviving
  ~6% is genuine item×form entanglement, still ~30× τ₀ — real, but BELOW the 10% gate bar.
  Most FORM-DOMINATED metrics would pass a calibrated gate → the Day-0 wall was ~half artifact.
- **INSTRUMENT seat (pairs, first 3 new metrics):** main-effect share |bias|/drift ≈ **0.93–1.00
  for every transformation** — reformulations push essentially ALL probes in one direction
  (uniformly stricter/looser), i.e. sign-uniform shifts, exactly the calibratable kind. Damage
  ranking: suffix (drift .185) > question (.134) > reorder (.129) > boilerplate (.115); reorder has
  the LOWEST main-effect share (0.93) = the most genuine entanglement — the one transformation
  that changes content-order changes WHICH items pass.
- **Scale hint:** 3B ≈ 8B on both raw and calibrated flips — fragility flat across the first two
  rungs (the on-thesis branch); 70B (in flight) is the decisive third point.
- **STAIRCASE COMPLETE (2026-07-02 AM, 70B partial 30/46):** raw 13.0 / 13.0 / 11.6% and
  calibrated 6.5 / 6.2 / 5.3% at 3B / 8B / 70B — **~flat across 23× parameters**, calibration
  removes ~52% at every scale. Paraphrase fragility is substantially a property of the rubric
  LANGUAGE, not the reader. `_log/form_decompose_{3b,8b}_v2.json`, `_log/form_decompose_70b_partial.json`.

## 5. Fragility-vs-scale — form dependence as a measurand (free from the aligned fleet)

`M_i_flip_rate` and `M_i_var_phi` per metric at 3B / 8B / 70B within Llama (Qwen as replication
panel, never pooled — same-family rule). Falls with scale ⇒ instrument weakness (form noise is an
executor deficiency and the big-model numbers are the honest ones); flat ⇒ property of the rubric
LANGUAGE itself (on-thesis, Polanyi-flavored: the telling won't hold still). Either branch is a
talk slide; jointly with flat Δ(E) it also defends the residual against the "your judge is just
noisy" attack.

## 6. Gate redesign (NEEDS SIGN-OFF — changes a certified verdict)

Replace the preempting binary FORM-DOMINATED with form-uncertainty folded into the band:
- Target seat: m̄_ω is a bounded K-average → leave-one-form-out / McDiarmid term ε_form added to
  the anytime band. FORM-DOMINATED then = "band too wide to decide", a width statement, not a dead
  end.
- Instrument seat: criteria fragility → widen the OPT_Ω head's error bar (it is uncertainty in
  what the checklist head is worth), rather than vetoing the whole certificate.
- Always report the graded axis (median flip, calibrated flip, per-form worst) alongside
  %H/tailF — form-fragility becomes a per-metric linguistic measurement, not a gate.

## 7. What does NOT help (so we don't spend there)

- **Same-form repeats:** τ₀ ≈ 0.002 — the readout is near-deterministic; nothing to average away.
- **More probes N:** flips are systematic per-item effects; N tightens the CI on the flip rate
  but not the rate. (N helps the 7 UNDERSAMPLED CW metrics — a different problem.)
- **Canonicalization** (one fixed template): certifies an arbitrary point of the orbit; rejected.
- **More forms K beyond ~4–5:** returns scale ~σ_φ/√K AND the deterministic orbit is exhausted at
  4 templates; richer orbits need LLM paraphrase (a second instrument — sign-off). Escalate K only
  if the aligned read still shows median flip(m̄) > 10% after calibration.

## 8. Order of operations

1. 70B/Qwen rescores finish (in flight) → copy/annotate forminv per §2 → Day-0-style certificate
   on aligned checkpoints (CPU).
2. Backfill `M_i_forms` via retarget re-runs (§3) → run the §4 decomposition (CPU notebook).
3. §5 scale read from the saved per-ckpt flip rates (CPU).
4. Decide §6 gate redesign + K escalation with the user, informed by the calibrated numbers.

Related: [[project_cw_day0_certificate_read]] (corrected), [[project_anthropological_framing]],
theory §12.6.2/§12.8; roadmap §2b talk cut.
