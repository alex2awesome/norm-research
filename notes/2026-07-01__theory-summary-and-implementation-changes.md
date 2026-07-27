# Theory summary + implementation changes (handoff)

*2026-07-01. **This note is a snapshot for handoff and will go stale quickly as the changes land** — the
living sources of truth are `2026-06-18__prompt-optimality-theory.md` (esp. the new **§12.6**) and the two
2026-07-01 critique notes. When in doubt, trust §12.6 over this.*

> **STATUS 2026-07-01 (later the same day): IMPLEMENTED.** All nine changes below landed
> (`value_certificate.py` new; `alpha_probe.py` quotient/orbit/decide; `crc_analyze` / `be_report` /
> `run_alpha_probe` / `run_value_census` updated; planted-control tests). Full suite: **103 pass**
> (`methods/metric_implementer/tests` + `methods/codability/tests`). Two deviations from this note as
> first written, both now reflected in theory §12.6.6/D3:
> 1. **`decide()` is asymmetric in f₁/N** (per the §12.6.6 table, which this note oversimplified):
>    CODIFIABLE is issued at ANY f₁/N (over-split singletons carry `v|S_g ≈ 0` and can only inflate ε
>    — conservative), DEEP requires `f₁/N < 0.8` (undersampled pools fake spread gains), and
>    ε-unresolved routes to UNDERSAMPLED (more draws shrink `Ĝ`+slack). Lemma 12.6.0 kills the
>    exponent readout, never the certificate.
> 2. **Good–Toulmin uses Efron–Thisted truncation `k₀=4`** — the raw alternating series manufactures
>    horizon mass out of saturated spectra (`w={5,6,7}` → `w₅−w₆+w₇`); terms `j>4` are variance, not
>    signal, for `c ≤ 1`.
> The codability work now lives in **`methods/codability/`** (stratified Codability Profile, L0–L4 +
> gates, planted controls: `python -m methods.codability.run_codability --controls`).

## Theory in six lines

1. Bracket (orientation is load-bearing): `R(p̂) ≤ OPT_Ω ≤ OPT_process-horizon ≤ I(M*;X)-under-named-assumptions`;
   recovery `R` = LOWER bound; `T(m_ω)` = ceiling on the operationalized metric = FLOOR on the ideal.
2. The study is anthropological: target = **revealed preference** (decades of `Y`; NO realtime humans).
   Task level: certified codification gap `A_H ≥ lowerCI(C_dense) − [OPT_Ω + ε]`. Metric level: anchor-free
   instrument layer (no labels needed).
3. **Lemma 12.6.0 (singleton degeneracy):** `f₁/N → 1 ⇒ α ≡ α_V ≡ 1` mechanically, value profile irrelevant.
   The observed high α/α_V are this artifact — neither evidence of depth nor against it.
4. **ε-gap certificate (§12.6.3–4):** value-weighted capture-recapture on the QUOTIENT partition —
   flux `V̇ = w₁/N` (Robbins), slack `B√(2log(1/δ)/N)+B/N` (McDiarmid), horizon `Ĝ(c) = −Σ(−c)ʲw_j`
   (Good–Toulmin); `ε = (1/γ̂)[Ĝ(c)+slack]`, backstopped by `adversarial_saturation` and the DPI ceiling.
   Value is **conditional on the frozen post-run greedy set `S_g`** (keeps the stream stationary, GV7).
5. **Partition-robustness is asymmetric (§12.6.2):** the head (pool-level greedy gains `g_k`, `OPT_Ω`) is
   partition-FREE; the flux `w₁` is over-split-SAFE (fake singletons carry `v|S_g ≈ 0`) but
   over-merge-UNSAFE (anti-conservative) ⇒ merge-precision is the one binding gate.
6. Scaling leg: per-tier gap `Δ(E_t)` with the flatness verdict; fitted asymptotes forbidden; count-census
   (α, Chao1, B_E, C_lo) demoted to descriptive vocabulary statistics with error bars.

## Implementation changes (priority order)

Files: `methods/metric_implementer/experiments/{alpha_probe, crc_analyze, be_report, semantic_behavioral,
run_value_census, run_alpha_probe}.py`. Changes 1–4 + 8 are **CPU post-hoc on existing `_sigs.npz`
checkpoints** (they store `sigs`, `tags`, `prompts`, `M_i`); 5 needs small new scoring passes.

1. **Quotient partition — semantic-merge / behavioral-split.** New `quotient_species(sigs, prompts, ...)`:
   candidate pairs = behavioral-MI above a floor OR embedding-near (prune hard — GLM quota is binding);
   GLM SAME/DIFFERENT judge (reuse `semantic_behavioral` prompt) may only **merge**; behavioral difference
   only **splits**; judge never splits. Output labels replace `conditional_species` for all population
   statistics. *Accept:* merge-precision on a held-out judged sample ≥ ~0.8 (vs 0.28 now); recapture appears
   (`f₁/N` drops materially).
2. **`f₁/N` everywhere + UNDERSAMPLED verdict.** Add `f1_over_N` to `alpha_probe.alpha_probe` report,
   `crc_analyze`, `be_report` rows/printouts. If `f₁/N → 1`: suppress α/α_V interpretation (Lemma 12.6.0).
3. **Value stream (the certificate).** In `run_value_census.py` (or new `value_certificate.py`):
   (a) pool-level greedy on raw criteria vs the target → gains `g₁ ≥ g₂ ≥ …`, `OPT_Ω`, tail sum, Hill index,
   top-k value share (partition-free — no species needed);
   (b) freeze `S_g`; per-species conditional value `v(s|S_g)` (closed-form binary MI / `shannon_cmi_surrogate`);
   (c) value spectrum `w_j` on the QUOTIENT labels; flux `V̇ = w₁/N`; McDiarmid slack; `Ĝ(c) = −Σ(−c)ʲ w_j`
   for `c ≤ 1` (defer OSW smoothing for c>1);
   (d) `ε = (1/γ̂)(Ĝ + slack)` with `γ̂` = lower-tail submodularity estimate on the discovered tail (§6.2
   machinery in `large_omega.py`); run `orthogonalize.adversarial_saturation` as the backstop.
   *Accept:* GV7 respected — marginal values NEVER mixed into the certificate stream; `w₁` reported at two
   probe sizes (over-merge sensitivity).
4. **`alpha_probe.decide()` rewrite** → four verdicts (§12.6.6):
   `UNDERSAMPLED` (f₁/N→1) / `FORM-DOMINATED` (form gates fail post-quotient) /
   `CODIFIABLE` (ε(c) ≤ ε₀ ∧ adv-sat ≈ 0) / `DEEP` (heavy greedy tail ∧ gates passed ∧ f₁/N ≪ 1).
   Count-α/C_lo/Chao1 gates removed from the decision (kept in the report as descriptive).
5. **Orbit-averaged target `m̄_ω`.** Average soft P(YES) over 3–5 form variants of the metric rubric
   (reuse `_reformulations` + the `template` kwarg; probe set only; #6 caching). Use `m̄_ω` as the value/
   recovery target; report `Var_φ` as the instrument error bar. DPI/T unchanged (still deterministic given X).
6. **Adverse-orbit reporting.** Partition statistics quoted as `[min, max]` over {template form ×
   Ω-order perms (`order_stability`) × probe subsample (`_probe_chao_std`)}; certificate quantities at the
   adverse end. Mostly wiring — the three perturbation axes already exist.
7. **Scaling table.** Per executor tier (llama-3B/8B/70B-FP8/Qwen-122B): `OPT_Ω(E)`, `ε(E)`,
   `Δ(E) = lowerCI(C) − [OPT_Ω+ε]`, slope CI over tiers → flat/shrinking verdict. NO fitted asymptotes.
   Keep the `discL1`-plateau capacity-artifact check from `pipeline_audit`.
8. **Bug fix `be_report._recovery`:** top-30 |r| feature selection currently happens on the full data
   before `cross_val_score` → selection leakage → optimistic `rec_r2`. Select within folds.
9. **Positive control (mandatory companion of any DEEP claim):** plant an articulable code metric, run the
   FULL quotient pipeline; must land CODIFIABLE with `R → 1` (C1/§5.5). A DEEP verdict without a passing
   planted control is not reportable.

## Do NOT change

- The count census code itself (rarefaction, Chao1, B–K `C_lo`, Fienberg IRLS, two-list Chapman) — correct
  as coded; it is *re-scoped* to descriptive vocabulary statistics, not deleted.
- GV7 discipline (marginal for any Heaps read, conditional-on-frozen-`S_g` for the certificate).
- The frozen-iid breadth stream / no-GEPA-in-the-accounting rule (§12.1a D/E guards).
- The DPI core (`R ≤ T`, soft readout G3, T̂-strength G4).

## Run order

(1) quotient on existing R3 CW ckpts → (2) `f₁/N` + spectra → (3) value stream + ε + verdicts →
(4) planted positive control → (5) `m̄_ω` re-target (small GPU) → (6) per-tier Δ table from the live
llama/qwen scaling run → (7) task-level `A_H` assembly against the deconfounded dense ceilings.
