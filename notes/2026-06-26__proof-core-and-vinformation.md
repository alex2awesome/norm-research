# The proof in one page — what's certified, and why it's Shannon-DPI, not V-information

*Concise companion to `2026-06-18__prompt-optimality-theory.md`. Per metric `M_i` (§1 of that doc): one
latent evaluation metric, recovered from its OWN (item, verdict) pattern, anchor-free — no label `Y`.*

## What we're proving

For a found prompt `p̂` (a rubric for metric `M_i`), bracket the optimum `OPT_i = sup_p R_i(p)`:

```
        R̂_i(p̂)   ≤   OPT_i   ≤   U_i
        (lower)                  (upper)
```

- **The lower bound `R̂_i` is free:** evaluate the recovery of `p̂`. Recovery `R_i = I(M_i; M̂_i)` —
  articulate-then-re-execute, label-free (§1's loop).
- **The upper bound `U_i` is the whole job.** The big doc is the *ladder of available `U_i`'s*: the
  assumption-free information cap (loose), then tighter `U`'s that buy tightness by **shrinking scope**
  (within-class → finite-set → covered-`B_E`). The certified gap `U_i − R̂_i` upper-bounds the true gap.

## The two facts the upper bound rests on (both Shannon/f-MI)

1. **Transmission `T_i` is convex in behavior.** `T_i(s) = I_f(I; V)` under behavior `s`, convex over the
   behavior cube ⇒ the cap, and the *shape* of the optimum.
2. **`R_i ≤ T_i` by the data-processing inequality.** Reconstruct-then-re-execute is *processing*, and
   processing can't increase MI ⇒ `T_i = I(M_i; X)` is the ceiling — and it is **estimable directly, with
   no enumeration of criteria.**

That's the core: **convexity + DPI**, both properties of Shannon/TVD-MI.

## Why it is NOT V-information — and the one monotonicity that does the work

V-information (Xu et al. predictive-V-info) is tempting because the §6 rubric-composition staircase
`ΔR_x = R(S∪x) − R(S)` reads naturally as the V-usable-information increment. But:

- **V-information violates the DPI** — *fatal* for the ceiling, since `R ≤ T` is exactly a DPI statement and
  DPI is the property V-info lacks. The doc uses Shannon/TVD for §2–§3 (ceiling + cap) **on purpose**, and
  explicitly refuses "V-information everywhere."
- **V-information appears only in §6** (attribution / the `(α, γ)` axes), where DPI is irrelevant
  (attribution needs no DPI). It is a value-function *language* there, nothing more.
- **Even in §6 its monotonicity fails.** §6.1: adding a criterion can *lower* recovery ⇒ the rung-2
  certificate is non-monotone USM, not the monotone-submodular bound. §6.2/§6.6: submodularity isn't
  automatic either — it's *measured*, not assumed.

**So the single monotonicity the proof rests on is Shannon's DPI (`R_i ≤ T_i`) — precisely the property
V-information does not have.** V-information is a secondary attribution device whose own monotonicity the
doc actively distrusts.

## Recovery and α are the two SIDES of the bracket — not substitutes

A recurring slip (mine): "the α-census is the wrong tool, use recovery." Wrong — they target opposite ends:

```
recovery  R_i      →  the LOWER bound (what p̂ achieves; free)
α / B_E census     →  the UPPER-bound machinery (characterize/cover B_E to bound OPT_i from ABOVE)
```

`α_i` (Heaps exponent of `M_i`'s reachable criterion space) decides whether the upper bound can be
*tightened by enumeration*: `α_i ≪ 1` ⇒ `B_E_i` is low-dim, cover it, get a tight `U_i`; `α_i ≈ 1` ⇒
inexhaustible, enumeration can't tighten it, so the upper bound **falls back to the direct DPI ceiling
`T_i = I(M_i;X)`** (still valid — just not tightened by coverage). Recovery is untouched by any of this; it
is the independent floor. Net: `R_i ≤ OPT_i ≤ T_i`, with the α-census deciding how tight the top can get.

## Regime split (which path for which goal)

- **Recovery (known target `M_i`):** estimate the soft posterior `η(X)=E[M_i|X]`, certify via `T_i`; no
  census — `B_E`-approximation degenerates to a single peak (§12.0, R10).
- **Unknown target:** the frozen·multi-list·value-annotated atlas (§12.1); `α` is the run-first go/no-go.

*Full development, proofs, and corrections: `2026-06-18__prompt-optimality-theory.md` — §1 (setup/units, the
per-metric scope), §2 (the two facts), §3 (the `U`-ladder), §5–§6 (articulation gap; the V-info attribution
layer and where its monotonicity fails), §11–§12 (recovery vs within-class; the α/value census).*
