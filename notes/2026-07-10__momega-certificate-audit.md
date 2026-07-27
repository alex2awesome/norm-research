# M_omega certificate audit

Date: 2026-07-10

## Bottom line

The project currently contains one sound all-prompt theorem and two different checklist analyses that
were later described as if they tightened that theorem.

1. **Sound:** for a fixed target channel `M_omega` and a candidate prompt evaluated on the same held-out
   item distribution, conditional independence gives
   `I_f(M_omega; M_p) <= I_f(M_omega; X)`. This is an upper bound over every candidate prompt.
2. **Conditional, not established:** equality with the prompt optimum requires the prompt/readout class
   to realize a sufficient statistic such as the target posterior. DPI alone does not prove this.
3. **Not an upper bound:** the greedy `OPT_omega` field in `value_certificate.py` is an adaptively selected
   achieved checklist value. Exhaustive `OmegaCertificate` is exact only for its finite declared subset
   class and fixed empirical probes.
4. **Not currently certified:** the capture-recapture `epsilon` is a process-horizon heuristic. It has no
   implemented one-sided horizon error bound, no certified unseen-tail gamma lower bound, and reuses the
   value probes for head selection. The planted tail-XOR test demonstrates under-coverage.

Therefore the July prompt-optimality and Law-B/Fano ceiling claims do not validate. The fixed-target DPI
cap survives, but it is often the deterministic-target entropy cap and is not shown to be tight for the
prompt interface.

## The theorem that survives

Let `Q(x)=P(M_omega=1|X=x)` and `P_p(x)=P(M_p=1|X=x)`. Assume the target and candidate executions use
independent randomness conditional on `X`, and the candidate prompt was frozen without seeing evaluation
items. Then `M_omega <- X -> M_p`, so for any f-divergence,

`R_f(p) = I_f(M_omega;M_p) <= T_f(M_omega) = I_f(M_omega;X)`.

For an item-uniform frozen probe set:

- Shannon: `T = H(mean Q) - mean H(Q)`.
- TVD: `T = mean |Q - mean Q|`.
- Binary-channel TVD recovery: `R = 2 |Cov(Q,P_p)|`.

These quantities are now implemented together in `vinfo.fixed_target_channel_certificate`. The empirical
DPI check is algebraic. `target_channel_ceiling(..., delta=...)` adds conservative one-sided population
upper bounds for iid items when per-item channel probabilities are exact. A population optimality gap is
issued only for a candidate frozen before evaluation; finite-pass probability uncertainty remains blocked.

### Tightness condition

For Shannon information, the unrestricted measurable posterior `eta(X)=P(M_omega=1|X)` is sufficient and
attains `T`. This does not imply that a prompt can realize `eta`. The current free-generation interface,
sampled binary re-execution, and finite MCQ pool do not establish posterior realizability. Consequently:

- `T-R` is a valid upper bound on the fixed-target prompt gap;
- `T-R` is not necessarily the true gap;
- a large `T-R` does not prove that an improving prompt exists.

When the target is thresholded deterministically, `T_Shannon=H(M_omega)` and `T_TVD=2*pi*(1-pi)`.
That is a real but often loose target-marginal cap. The current experiments mostly use this case.

## Objective mismatch

Sections 1-3 of the theory optimize the target prompt itself: each `p` generates the labels that are then
reconstructed, so the objective is `sup_p I(M_p; Mhat_p)`. Section 11 instead fixes `M_omega` and optimizes
only the reconstructor/candidate prompt. A fixed `T(M_omega)` bounds the second objective. It does not bound
the first except through `sup_p T(M_p)` or the universal channel cap. The two problems need different names
in every result artifact.

## Implementation findings

### Canonical R2 recovery

The old `run_r2_recovery.py` certificate compared:

- `R`: Shannon `iv_transmission` in bits on held-out items;
- `T`: TVD `tvd_transmission` on all items.

This mixed both the divergence and evaluation distribution. In the two current 10-row CW result sets, the
stored old `T` averaged 0.418/0.420 while the matching deterministic-target Shannon cap reconstructed from
`R/transmission_norm` averaged 0.845/0.881. Mean headroom was understated as 0.265/0.262 instead of
0.691/0.724 bits. The code now emits same-heldout Shannon and TVD bundles separately.

### Value certificate

`value_certificate.py` cannot currently support `OPT_omega + epsilon` as a high-confidence upper bound:

- candidate pre-ranking and greedy selection happen on the same probes used for value estimation;
- five-fold OOF scoring is inside each candidate evaluation, not around the candidate search;
- `H - achieved_head` was a data-dependent bounded-differences cap;
- truncated Good-Toulmin/OSW is a horizon point predictor, while the added radius bounds next-draw flux;
- `gamma_tail` is a percentile of ten sampled discovered-tail blocks, not a lower confidence bound for
  all discovered or unseen blocks;
- `adv_saturated=None` was treated as a pass;
- the repository's own tail-XOR control reports epsilon 0.031 against hidden residual 0.096 bits.

The module now marks `upper_bound_valid=False`, uses `H(M)` as the fixed value cap, and cannot return
`CODIFIABLE` without both a validated bound payload and `adv_saturated is True`.

### Finite Omega certificates

For `K <= 15`, exhaustive enumeration really does give the maximum observed recovery over all declared
subsets on the fixed probe set. This is a transductive finite-class statement, not a population bound and
not a prompt-space bound. Selecting on train probes and reporting one test value evaluates the chosen subset;
it does not upper-bound the best population subset without simultaneous confidence bounds.

For large Omega, sampled `gamma_measured` cannot validate U2. A sampled minimum need not be below the true
global minimum. U2 is certified only when monotonicity and a genuine global gamma lower bound are supplied.

### Decoder-capacity and atlas proposals

The exact decoder-capacity object `OPT_prime = max_m R(m)` over all `2^N` labelings is mathematically an upper
bound on prompt-induced labelings. The proposed PCA/spectral search evaluates only a subset of labelings, so
its maximum is a lower bound on exact `OPT_prime`, not a computable ceiling. It must not be used for certified
headroom without a relaxation that supplies a proved optimizer upper bound.

Likewise, the atlas stopping conjunction (coverage estimate, low singleton-value flux, dry adversarial list)
does not certify support exhaustion. The current concentration argument also reuses the realized pool to select
the head and estimate values; changing one capture draw can change many downstream values, invalidating the
claimed two-species bounded-difference argument. The time-uniform delta schedule is useful bookkeeping for a
future split-sample version, not a repair of the current horizon bridge.

## Result reclassification

The 46-row CW snapshot contains 39 `FORM-DOMINATED`, 5 `UNDERSAMPLED`, and 2 `CODIFIABLE` labels, but all
46 have `adv_saturated=null`; none qualifies as certified. Five rows have `OPT+epsilon > H` and are capped at
the entropy line. Three observed single prompts exceed the achieved checklist head; one exceeds the claimed
`min(OPT+epsilon,H)` ceiling. Across the 46 matched rows:

- achieved checklist head: mean 0.583 bits;
- heuristic epsilon: mean 0.146 bits;
- observed single prompt: mean 0.384 bits;
- achieved head minus observed single: mean 0.199 bits;
- historical claimed ceiling minus observed single: mean 0.328 bits;
- valid empirical DPI cap minus observed single: mean 0.436 bits.

The extra 0.129 bits is chiefly the heuristic epsilon being interpreted as attainable content. It is not a
measured decomposition gain. The larger 0.436-bit DPI headroom is a valid upper bound on the true fixed-target
optimization gap, but it is not an estimate of attainable content. On two nested splits of the showcased metric #24, head selection scored
0.561/0.625 bits on the selection half but only 0.444/0.431 after freezing on the other half, direct evidence
that ordinary OOF candidate scores do not remove search optimism.

The older Aigner notebook remains useful as a finite-class walkthrough: its 0.133 TVD value is the exact
empirical maximum across 63 non-empty subsets of six mined criteria. It also documents why the example is
not construct-valid as a general ceiling: the harvester missed half the schema, the elegance facet vanished,
and the GEPA revise prompt drifted into competitive-programming language.

## Fano and OSL

The Law-B Fano inversion needs an information upper bound. Plugging in achieved `OPT_omega` reverses the
needed role. More importantly, Law B has no reconstruction bottleneck: both executor and crowd read the text
and rubric, so their agreement is not mediated only by the prompt channel. Shared priors can also create
agreement with zero reconstructed information. The Fano line must be removed from Law B.

The planted-control reference is the appropriate operational comparator, but it is an empirical reference,
not a proof-level ceiling for every bank metric. Its implementation also selected each control's top three
`y` values instead of the outcomes at the top three capability `z` values. The corrected helper selects by
`z`; downstream BOUNDED labels should be called candidates and audited within family.

The eight task panels were regenerated on `sk3` after this repair. On the same fresh curves, replaying the old
top-`y` rule would produce 91 `BOUNDED` calls; the top-`z` rule produces 65. All 26 changes are
`BOUNDED -> REACHES`. The corrected counts are: creative writing 2, humor 41, math 0, news homepages 13,
notice-and-comment 0, patents 0, peer review 9, and press releases 0. These counts differ from earlier saved
snapshots because additional executor panels had landed; the paired 91-to-65 comparison isolates the code fix.

## What Dossier adds

Dossier does not add an upper bound. Its value is external and cross-speaker validity:

- fixed-target M_omega asks whether one executor's behavioral policy can be reconstructed;
- Dossier asks whether richer messages cause different executors/families to converge on the same policy.

That second question detects executor dialect and self-reconstruction triviality that DPI cannot. It should be
kept as a validity/transport arm, not combined numerically with the M_omega ceiling. A strong story uses the
DPI ceiling for the fixed-target upper bound and Dossier to test whether the target and its reconstruction are
socially portable.

## Requirements for a tighter certified ceiling

1. Keep `M_omega` fixed and preserve soft target probabilities rather than thresholding when possible.
2. Freeze the candidate prompt before evaluation; use the same held-out items and same f on every DPI leg.
3. Report empirical-probe and iid-population scopes separately.
4. For a finite declared grammar, enumerate candidates and use independent evaluation with simultaneous
   upper confidence bounds if a population class optimum is required.
5. Do not use capture-recapture as a prompt ceiling unless head/partition/value estimation are split,
   a one-sided weighted horizon theorem is implemented, and a global structural gamma lower bound is proved.
6. Treat planted, Dossier, and cross-family scaling as validity evidence, not replacements for the DPI bound.
