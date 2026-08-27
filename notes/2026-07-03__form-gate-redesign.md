# Form-gate redesign: FORM-DOMINATED verdict → ε_form band (2026-07-03)

**Decision (user-delegated).** Retire FORM-DOMINATED as a categorical verdict; charge form
fragility as a quantified bits band against the ceiling. Adopted in theory doc §1.

## What shipped (CPU, no GPU, tests green 30/30)

`methods/metric_implementer/experiments/alpha_probe.py`:
- `decide(..., form_mode=)` — new switch. `'verdict'` (DEFAULT, byte-unchanged: cert-builder path
  and split-half job keep old behavior) still returns "FORM-DOMINATED". `'band'` does NOT preempt:
  the metric keeps its census verdict (CODIFIABLE/DEEP/UNDERSAMPLED) and `form_invariant` survives
  as a reported diagnostic only.
- `eps_form_bits(form_invariance, certificate)` → `{bits, source, q}`. **Exact** when the cert
  carries `opt_omega_by_form` (per-form re-census): `ε_form = max_form OPT_Ω(form) − OPT_Ω(canon)`.
  **Proxy** (`q·OPT_Ω`, linear in median flip rate) when only the drift summary exists — a
  proportionate, self-labeled placeholder. `source ∈ {exact, proxy, none}` so a proxy is never
  read as the certified number. Kept OUT of the CODIFIABLE gate on purpose: census ε and form
  stability are orthogonal axes; ε_form only widens the ceiling in the residual
  `Δ = lowerCI(C_dense) − [OPT_Ω + ε + ε_form]`.

## The honest re-read finding

Re-ran the Day-0 certs under `form_mode='band'` (`notebooks/data/two_faces_20260702/band_mode_reread.json`):

| domain | verdict mode (old) | band mode (new) | the ejected FORM-DOMINATED become |
|---|---|---|---|
| creative-writing (43) | 36 FD / 7 US | 2 COD / 41 US | 34 UNDERSAMPLED, 2 CODIFIABLE |
| humor (60) | 36 FD / 19 US / 5 COD | 8 COD / 52 US | 33 UNDERSAMPLED, 3 CODIFIABLE |

**The form cliff was mostly masking undersampling at n=300, not certifying a distinct
"form-dominated" population.** When not ejected, ~92% of ex-FORM-DOMINATED metrics are simply
ε-unresolved (UNDERSAMPLED — the cure is more probes), and only 2–3 per domain are genuinely
CODIFIABLE-but-form-fragile. So the redesign's real payoff is diagnostic honesty: it reveals the
binding constraint is probe count, and it stops conflating "phrasing-sensitive readout" with
"can't be certified." (ε_form source is 'none' here: Day-0 certs saved only the boolean, not the
flip rate — so these band verdicts use ε_form=0, the most generous-to-CODIFIABLE case; the real
ε_form only makes the residual more conservative.)

## Still owed (GPU, rides with the forminv passes)

1. Per-form re-census so `eps_form_bits` returns `source='exact'` — needs reformulated signatures
   scored on the executor (the drift test currently discards `sig2`). Then the residual Δ uses the
   certified ε_form.
2. Persist `binary_flip_rate` (and eventually `opt_omega_by_form`) into the cert record so the
   proxy/exact ε_form is available without recompute.
3. Sync `alpha_probe.py` to sk3 — HELD until the split-half job finishes (it imported the old file;
   the finalizer doesn't need the new code). Sync debt flagged.

Related: theory `notes/2026-07-02__two-faces-theory.md` §1; `[[project_metric_count_certificates]]`;
`[[certificate-audit-disciplines]]`.
