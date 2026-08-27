# Is reconstruction the optimal unsupervised metric? — what can and cannot be proved

**Question (user, 2026-07-19):** "is there any way to prove that reconstruction is the optimal
unsupervised metric? Any mathematical proof? Maybe borrowing from latent variable recovery
(e.g. EM has a p(x|z) measurement, which is exactly what reconstruction is measuring)."

**Short answer.** There are two real theorems that together give reconstruction a principled
status — it is the *tightest certifiable label-free lower bound* on a quantity that *provably
caps all downstream usefulness*. There is **no** proof that reconstruction is optimal for any
particular downstream task, and §4 gives the counterexample class showing why no such proof can
exist. The EM/ELBO intuition is exactly right in form (§5), and pushing it through gives us the
sharpest available justification for a design choice we already made for other reasons: that the
recovery readout must be **behavioral** (§6).

Notation: texts `X ~ p(X)`; a criterion `ω` (natural language) compiles to a scorer `M_ω = C(ω)`
applied to `X`; a reconstructor sees `(X, M_ω(X))` pairs and emits `ω̂ = R(X, M_ω(X))`, which
re-compiles to `M̂ = C(ω̂)`. The project's canonical readout is `C(R(Ω)) = I(M_ω; M̂)`.

---

## 1. Theorem A (Barber–Agakov): reconstruction is the tight variational bound

For any variational decoder `q`,

    I(M; X) = H(X) + E_{p(x,m)}[log p(x|m)] ≥ H(X) + E_{p(x,m)}[log q(x|m)]

with **equality iff `q(·|m) = p(·|m)` almost everywhere**. Since `H(X)` is a property of the
corpus and does not depend on the metric, maximizing the reconstruction log-likelihood
`E[log q(x|m)]` maximizes a lower bound on `I(M;X)`, and the bound is *tight exactly when the
reconstructor is the true posterior*.

**Consequence.** Among label-free scores of the variational-lower-bound family, reconstruction
with an optimal reconstructor is not merely *a* bound but *the* attaining one; any other
label-free member of the family is weakly dominated. This is the precise sense in which "use a
strong reconstructor" is not a convenience but a condition for the estimand to mean what we say
it means — and it retroactively justifies the standing rule that judging/reconstruction runs on
Sonnet-or-better models. (Caveat: attainment assumes the decoder family is rich enough to
contain the true posterior `p(x|m)`; with any restricted reconstructor the bound is strict, so
in practice "tight" means "as tight as the strongest available reconstructor makes it.")

## 2. Theorem B (Data Processing): `I(M;X)` caps every downstream use

If the metric is computed from the text alone, then for **any** target `Y` the chain
`Y — X — M` is Markov, so

    I(M; Y) ≤ min( I(X; Y), I(M; X) )    for every Y.

**Consequence.** A metric with small `I(M;X)` is provably useless for *every* downstream target;
no clever supervised head can rescue it. So `I(M;X)` is a *universal capacity ceiling*, and a
certified lower bound on it (Theorem A) is a certificate of potential — obtained with no labels.

Note this is also the sharp form: for a deterministic metric `M = f(X)`, `I(M;X) = H(M)` and
`sup_Y I(M;Y) = I(M;M) = H(M)`, so the ceiling is *attained* by the best-case target. For the
stochastic (LLM-executed) metrics we actually use, `I(M;X) = H(M) − H(M|X)`: **reliable**
discriminative capacity, entropy minus judge noise. That decomposition is worth keeping in view
— it is why execution reliability and discrimination are not separate desiderata but two terms
of one quantity.

## 3. What the two theorems jointly license

    reconstruction score  ≤  I(M;X)  ≥  I(M;Y)   for every downstream Y
    └── certified from below, no labels ──┘   └── universal cap ──┘

Reconstruction is therefore **the optimal unsupervised criterion for certifying capacity**
(within the variational-lower-bound family — no formal optimization over all conceivable
label-free certificates exists). That is a defensible, publishable claim. It is strictly weaker
than "the optimal unsupervised metric," and the next section says why the stronger claim is false.

**Terminology guard (two senses of "reconstruction" — do not conflate in the paper).** Theorem A's
"reconstruction" is a decoder recovering **X from M** (`E[log q(x|m)]`), which certifies `I(M;X)`.
The project's canonical readout is **criterion recovery**: `R` sees `(X, M_ω(X))`, emits `ω̂`,
and we score `I(M_ω; M̂)` — a different object. Theorem A does *not* directly certify the
`I(M_ω;M̂)` pipeline (we never compute the Barber–Agakov text-decoder); the two are separate rows
in §7's table, and a paper section must state the distinction explicitly or readers will read the
certified capacity bound as if it were the measured articulability number.

## 4. Why no stronger optimality theorem can exist (the degeneracy counterexample)

`I(M;X)` is **necessary but not sufficient** for usefulness. Take `M = ` "1 if the SHA-256 of the
text is odd." It is a short natural-language criterion, it is deterministic, it maximizes `H(M)`
and hence `I(M;X)` (within the class of binary metrics; higher-arity metrics can carry more bits)
— and it carries negligible information about any semantic `Y` of interest (cryptographic
pseudo-independence; not literally zero over a fixed finite corpus, but effectively so). It sits
at the top of the capacity ranking and the bottom of the utility ranking.

Two consequences we should hold onto:

1. **Any program of the form "maximize `I(M;X)`" is entropy maximization** (exactly so in the
   deterministic case). This is the formal statement of the audited empirical finding that
   *variance-revival ≠ information-revival* — the A-bank degeneracy result was not an artifact,
   it is what the mathematics predicts. It also means the discrimination-maximizing objective in
   the M_ω optimizer (`std − ½|mean − ½|`) is a **capacity** objective, and cannot by itself
   distinguish a meaningful criterion from a reliable hash. Worth revisiting on those grounds.
2. Therefore the articulability construct cannot be `I(M;X)`. It must be the recovery quantity
   `I(M_ω; M̂)` — which is what the project already reports. §6 shows this is not a workaround
   but the mathematically correct object.

**Do not attempt to close this gap with Fano.** Fano-style identification bounds
(`OPT_Ω+ε` / `CR-ε`) were derived and then **retracted** in this project (see the M_ω audit); the
only certified all-prompt bound that survived is the DPI fixed-target cap, i.e. Theorem B above.
Re-deriving a Fano bound here would resurrect a known-bad result.

## 5. The EM / latent-variable correspondence (the user's intuition, made precise)

The ELBO for a latent-variable model is

    log p(x) ≥ E_{q(z|x)}[ log p(x|z) ] − KL( q(z|x) ‖ p(z) )
               └── reconstruction term ──┘   └── complexity term ──┘

The user's mapping is exact: **our reconstruction is the `p(x|z)` term with the criterion as the
latent.** Concretely `z ↔ ω`, observations `↔ (text, label) pairs`, and the reconstructor `R` is
the variational posterior `q(ω | evidence)`. Re-executing `C(ω̂)` and scoring agreement against
`M_ω` is a behavioral surrogate for the conditional likelihood of the evidence under `ω̂`.

What this buys us, and what it exposes:

- **Buys:** it explains why reconstruction is the natural label-free objective — it is the term
  EM maximizes. Our procedure is a hard-assignment, behaviorally-scored analogue of the E-step.
- **Exposes:** we have no **complexity term**. The ELBO's `KL(q‖p)` penalizes latents that are
  improbable under the prior; we approximate that with a hard constraint (bounded
  natural-language criteria) rather than a penalty. Making it explicit would turn the recovery
  readout toward a proper MDL objective — but **the penalty must target the right complexity
  notion, and English description length is the wrong one**: the §4 hash was chosen precisely
  because its English description is *short* ("1 if the SHA-256 is odd"), so a
  `agreement − λ·description-length` score is *minimized*, not maximized, by the hash and cannot
  exclude it. What excludes the hash is a complexity measure on the **executed computation**
  (Kolmogorov/runtime/incompressibility of `M_ω` as a function of the text — the hash's short
  description names an incompressible computation), or equivalently a prior `p(ω)` that weights
  criteria by semantic plausibility rather than string length. The already-logged description
  lengths therefore do NOT suffice for a pilot; a pilot needs a computational-complexity or
  semantic-prior surrogate (e.g., can a judge predict the label from the criterion *without
  executing it*? — executability-by-reading as a cheap anti-hash probe). Still the most
  promising open theoretical move, but not the cheap one the first draft of this note claimed.

## 6. Identifiability — and why measuring *behavior* is the mathematically right choice

The load-bearing argument here is elementary and does not need imported theorems: behavioral
evidence reaches `R` only through `M_ω`, so any two criteria with `M_{ω₁} = M_{ω₂}` produce
identical evidence and **no reconstructor, however strong, can separate them** — pure
non-injectivity of `C` on its fibers, and no amount of data fixes it. The latent-variable
literature says the same thing in its own setting and is cited as motivation, not proof:
nonlinear latent-variable models are **not identifiable** in general (Hyvärinen & Pajunen 1999);
identifiability results such as iVAE (Khemakhem et al. 2020) recover latents only up to an
equivalence class, and only under auxiliary-variable conditions.

So the identifiable object is not `ω` but the equivalence class `[ω]_behavior`. Measuring
`I(M_ω; M̂)` — agreement of *induced behavior*, never text similarity to a reference — is exactly
measurement on that quotient. The project's standing rules ("report recovery metric only",
"never use similarity-to-reference as a predictor") were adopted for empirical reasons; the
identifiability argument says they are also the only well-posed choice. A text-similarity
readout would be attempting to measure an unidentifiable quantity.

## 7. Summary — what to claim in the paper

| Claim | Status |
|---|---|
| Reconstruction is the tight variational label-free bound on `I(M;X)` | **Provable** (Thm A; tightness assumes rich decoder family) |
| `I(M;X)` upper-bounds `I(M;Y)` for every downstream `Y` | **Provable** (Thm B, DPI) |
| Reconstruction is the optimal unsupervised *capacity certificate* (within the variational-bound family) | **Provable** (A+B) |
| Reconstruction is optimal for a *specific* downstream task | **False** (§4 counterexample) |
| Recovery `I(M_ω;M̂)` measures the identifiable quotient `[ω]_behavior` | **Provable** (§6, elementary fiber argument) |
| Adding a *computational*-complexity term would exclude the degenerate maximizers | **Open, promising** (§5 — description-length alone provably does NOT) |

The honest headline: *reconstruction is provably the best label-free certificate of a metric's
capacity, and provably not a guarantee of its usefulness.* For a paper about articulability that
is the right claim anyway — we are measuring whether a criterion can be transmitted through
language, not whether it predicts a particular outcome.

### Open items this note generates
1. Pilot the complexity-penalized recovery readout (§5). NOT via the logged description lengths
   (a description-length penalty is minimized by the §4 hash and cannot exclude it — see §5);
   needs a computational-complexity or semantic-prior surrogate first.
2. Re-examine the discrimination-maximizing M_ω objective in light of §4 (it is a capacity
   objective; consider replacing/augmenting with the recovery objective itself).
3. Neither item changes any current estimand; both require sign-off before becoming headline
   measurements.

*(2026-07-20 audit pass: §1 decoder-family caveat, §3 within-family qualifier + two-senses-of-
"reconstruction" guard, §4 binary-class/negligible-information tightenings, §5 MDL rewrite —
description length provably cannot exclude the hash — and §6 fiber-argument attribution added
after independent verification; the two load-bearing theorems and the headline survived audit
unchanged.)*
