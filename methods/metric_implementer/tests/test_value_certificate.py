"""Planted-ground-truth tests for the §12.6 checklist diagnostics
(experiments/value_certificate.py). All CPU, no model, no judge API (judges are injected fakes).

Includes the offline half of the §12.6.7 positive-control discipline: a planted articulable rule
through the full quotient pipeline must recover nearly all of H(M), while the unvalidated horizon
bridge remains explicitly non-certifying; a
planted value-spread metric must land DEEP; the all-singleton regime must trip UNDERSAMPLED and
exhibit the Lemma 12.6.0 degeneracy (α_V ≡ 1 regardless of the value profile). The live planted-C1
control (real executor) remains a GPU-side run.
"""
from __future__ import annotations

import numpy as np

from methods.metric_implementer.experiments import alpha_probe as ap
from methods.metric_implementer.experiments.value_census import value_rarefaction
from methods.metric_implementer.experiments.value_certificate import (
    _cmi_block, anytime_delta, certificate, epsilon_gap, flux_certificate,
    good_toulmin_value, greedy_head, osw_horizon_value, tail_gamma, value_spectrum,
)

NPROBE = 240


def test_value_spectrum_and_good_toulmin_identities():
    n_s = np.array([1, 1, 2, 3])
    v_s = np.array([0.4, 0.1, 0.2, 0.3])
    w = value_spectrum(n_s, v_s)
    assert np.isclose(w[1], 0.5) and np.isclose(w[2], 0.2) and np.isclose(w[3], 0.3)
    # Ĝ(c) = −Σ (−c)^j w_j ; at c=1: w1 − w2 + w3
    assert np.isclose(good_toulmin_value(w, c=1.0), 0.5 - 0.2 + 0.3)
    # c clamped to ≤ 1 (variance explodes beyond — OSW smoothing not implemented)
    assert np.isclose(good_toulmin_value(w, c=5.0), good_toulmin_value(w, c=1.0))
    # no singletons + heavy multiplicities ⇒ small horizon gain
    w_sat = value_spectrum(np.array([5, 6, 7]), np.array([0.3, 0.3, 0.3]))
    assert good_toulmin_value(w_sat, 1.0) <= 0.3 * 3 * 0.05 + 1e-6


def test_flux_certificate_direction_and_scaling():
    w = {1: 0.5, 2: 0.2}
    fc_small = flux_certificate(w, N=50, B=0.5, delta=0.05)
    fc_big = flux_certificate(w, N=5000, B=0.5, delta=0.05)
    assert fc_small["flux_hi"] >= fc_small["flux"] >= 0            # one-sided upper bound
    assert fc_big["slack"] < fc_small["slack"]                     # slack shrinks with N
    assert np.isclose(fc_big["flux"] / fc_small["flux"], 50 / 5000)  # flux = w1/N
    # ε assembly: bigger γ̂ ⇒ tighter; never negative
    assert epsilon_gap(0.2, 0.1, 1.0) < epsilon_gap(0.2, 0.1, 0.5)
    assert epsilon_gap(-0.3, 0.0, 1.0) == 0.0                      # Ĝ clipped at 0


def test_lemma_1260_singleton_degeneracy_of_alpha_V():
    # all-singleton spectrum ⇒ E[V(m)] is exactly linear ⇒ α_V ≡ 1 REGARDLESS of the value profile.
    N = 60
    n_s = np.ones(N, int)
    for values in (np.ones(N), np.linspace(0.01, 1.0, N), np.random.default_rng(0).uniform(0, 1, N)):
        ms, V = value_rarefaction(n_s, values, N)
        alpha = ap.heaps_alpha(ms, np.maximum(V, 1e-12))
        assert np.all(np.abs(alpha[1:-1] - 1.0) < 0.02)            # pinned at 1, value-blind


def _planted_pool(rng, *, n_probes=NPROBE, n_para=6, n_distract=40, flip=0.02):
    """A codifiable planted metric: M = one latent binary rule; pool = the CLEAN rule criterion (i=0)
    + flip-jittered paraphrase duplicates (i ≥ 1) + independent distractors, each drawn TWICE
    (a recaptured stream). Returns (sigs, prompts, tags, M).

    Two design constraints imposed by the SOUND band (§12.8.0 T1): (a) the i=0 copy carries no verdict
    flips — verdict noise on the best criterion puts h₂(flip) bits of irreducible residual into
    H(M|S_g), which becomes the a-priori McDiarmid cap and correctly blocks 0.05·H certification at
    this N (judge-noise robustness is the §12.8 L6 attenuation lemma's territory, not this control's);
    (b) distractors are RECAPTURED (two jittered copies each) — a positive-control world is a
    saturated process, and singleton distractors would put spurious OOF-clamped value mass into w₁."""
    M = (rng.uniform(0, 1, n_probes) > 0.5).astype(float)
    rule_sig = M * 0.8 + 0.1
    sigs, prompts = [], []
    base_txt = "the story maintains a consistent narrative tone throughout every scene"
    for i in range(n_para):
        noise = rng.uniform(-0.04, 0.04, n_probes)
        flips = (rng.uniform(0, 1, n_probes) < flip) if i > 0 else np.zeros(n_probes, bool)
        s = np.where(flips, 1 - rule_sig, rule_sig) + noise
        sigs.append(np.clip(s, 0, 1))
        prompts.append(base_txt if i == 0 else
                       f"the narrative maintains a consistent tone in scene after scene variant {i}")
    for j in range(n_distract):
        d = (rng.uniform(0, 1, n_probes) > 0.5).astype(float) * 0.8 + 0.1
        for k in range(2):                                          # recaptured: two copies per species
            sigs.append(np.clip(d + rng.uniform(-0.04, 0.04, n_probes), 0, 1))
            prompts.append(f"distractor criterion {j} take {k} about {rng.integers(0, 9)} unrelated aspects")
    tags = ["fam%d" % (i % 3) for i in range(len(sigs))]
    return np.vstack(sigs), prompts, tags, M


def test_greedy_head_recovers_planted_rule_partition_free():
    rng = np.random.default_rng(11)
    sigs, prompts, tags, M = _planted_pool(rng)
    B = (sigs > 0.5).astype(int)
    head = greedy_head(B, M.astype(int), top_k=10, seed=0)
    assert head["selected"][0] < 6                                  # a rule copy picked first
    assert head["frac_H"] > 0.7                                     # head recovers most of H(M)
    assert head["tail_frac"] < 0.25                                 # value concentrated in the head
    # paraphrase duplicates land in the tail with ~0 gain — over-splitting cannot corrupt the head
    assert all(g < 0.1 for g in head["gains"][1:])


def test_certificate_planted_codifiable_recovers_head_but_does_not_certify():
    rng = np.random.default_rng(3)
    sigs, prompts, tags, M = _planted_pool(rng)
    cert = certificate(sigs, M, tags, prompts=prompts, quotient="behavioral", top_k=10, seed=0)
    verdict = ap.decide({"f1_over_N": cert["f1_over_N"]}, certificate=cert)
    assert cert["frac_H"] > 0.7
    assert np.isfinite(cert["eps_bits"])
    # The order-adverse sensitivity is still reported, but it is not a confidence bound.
    assert cert["eps_bits_adv"] >= cert["eps_bits"] - 1e-12
    assert cert["order_band"]["n_orders"] == 8
    # Optional-stopping bookkeeping does not repair the horizon/synergy bridge.
    assert cert["stopping"] == "anytime" and cert["n_union"] == 9
    assert cert["delta_effective"] < cert["delta"] / 9
    assert cert["B_flux_cap"] >= cert["resid_cap"] > 0.0
    assert cert["upper_bound_valid"] is False
    assert verdict != "CODIFIABLE"


def test_xor_blindspot_documented_and_pairs_combiner_catches_it():
    # §12.6.4 combiner-class escape: M = PARITY of two pool criteria. Under F₁ (additive logistic)
    # every unit — and every raw pair — has v ≈ 0, so the head finds nothing and the certificate
    # reads "tiny OPT, saturated": the checklist bound is COMBINER-INDEXED, and this test documents
    # the limitation. F₂ ('pairs': candidates [x_i, x_j, x_i·x_j] — the product column makes parity
    # linearly representable) recovers it.
    rng = np.random.default_rng(6)
    n = NPROBE
    x1 = (rng.uniform(0, 1, n) > 0.5).astype(int)
    x2 = (rng.uniform(0, 1, n) > 0.5).astype(int)
    M = ((x1 + x2) % 2).astype(int)
    rows = [x1, x2] + [(rng.uniform(0, 1, n) > 0.5).astype(int) for _ in range(10)]
    B = np.vstack(rows)
    lin = greedy_head(B, M, top_k=8, seed=0)                       # F₁: documented blind spot
    assert lin["opt_omega_bits"] < 0.1
    pr = greedy_head(B, M, top_k=8, seed=0, combiner="pairs", pair_prerank=len(B))
    assert pr["opt_omega_bits"] > 0.5                              # F₂ finds the parity structure
    pair_picks = [e for e in pr["selected"] if isinstance(e, tuple) and e[0] == "pair"]
    assert pair_picks and set(pair_picks[0][1:]) == {0, 1}         # and it is THE planted pair
    assert {0, 1} <= set(pr["selected_units"])
    assert pr["S_cols"].shape[1] >= 3                              # incl. the product column


def test_certificate_planted_deep_lands_deep():
    # value SPREAD: M = majority vote of 12 independent weak bits; the pool = those 12 bits (each seen
    # once — a genuinely wide value profile) + a few distractors. Greedy gains spread ⇒ heavy tail.
    rng = np.random.default_rng(4)
    bits = (rng.uniform(0, 1, (12, NPROBE)) > 0.5).astype(int)
    M = (bits.sum(axis=0) >= 6).astype(float)
    sigs = [b * 0.8 + 0.1 for b in bits]
    prompts = [f"weak aspect {i} of the piece is present" for i in range(12)]
    for j in range(8):
        sigs.append((rng.uniform(0, 1, NPROBE) > 0.5).astype(float) * 0.8 + 0.1)
        prompts.append(f"distractor {j}")
    tags = ["fam%d" % (i % 3) for i in range(len(sigs))]
    cert = certificate(np.vstack(sigs), M, tags, prompts=prompts, quotient="behavioral",
                       top_k=15, seed=0)
    assert cert["tail_frac"] >= 0.25                                # value NOT head-concentrated
    # this planted pool is every-species-once (f1/N = 1): the SINGLETON regime blocks DEEP (theory
    # table: ∧ f₁/N ≪ 1 — spread gains could mean the head just hasn't arrived) ⇒ UNDERSAMPLED raw…
    assert ap.decide({"f1_over_N": cert["f1_over_N"]}, certificate=cert) == "UNDERSAMPLED"
    # …and DEEP once the stream recaptures (simulated: same certificate, recapturing-regime f1/N)
    assert ap.decide({"f1_over_N": 0.3}, certificate=cert) == "DEEP"


def test_certificate_through_judge_quotient_positive_control():
    # §12.6.7 offline positive control: the FULL pipeline (quotient with an injected judge → flux →
    # ε → verdict) on the planted codifiable metric. The fake judge merges lexical paraphrases —
    # f1/N must DROP vs the strict partition; the bridge still must not issue CODIFIABLE.
    rng = np.random.default_rng(9)
    sigs, prompts, tags, M = _planted_pool(rng, flip=0.15)          # strong form fragility → strict split
    from methods.metric_implementer.experiments.alpha_probe import _shingle_jaccard
    judge = lambda pairs: [_shingle_jaccard(x, y, k=1) > 0.5 for x, y in pairs]
    cert_strict = certificate(sigs, M, tags, prompts=prompts, quotient="behavioral", top_k=10, seed=0)
    cert_judge = certificate(sigs, M, tags, prompts=prompts, quotient="judge", judge_fn=judge,
                             top_k=10, seed=0)
    assert cert_judge["quotient_audit"]["n_merged"] >= 1            # paraphrases collapsed
    assert cert_judge["D_iid"] <= cert_strict["D_iid"]
    v = ap.decide({"f1_over_N": cert_judge["f1_over_N"]}, certificate=cert_judge)
    assert v != "CODIFIABLE"


def test_osw_horizon_beats_raw_gt_beyond_c1():
    # §12.8.7 I1: uniform-world simulation with ANALYTIC conditional truth. At t = 4 the raw
    # alternating series swings by t^j per unit fluctuation of f_j (a single realized f_5 adds
    # ±1024), while the OSW binomial taper P(L ≥ j) kills the high-j coefficients — the import
    # buys the horizon with a published guarantee (vanishing NMSE for t ≲ log N).
    S, N, t = 300, 150, 4.0
    p = 1.0 / S
    per_unseen = 1.0 - (1.0 - p) ** (t * N)      # P[an unseen species appears within t·N more draws]
    rng_v = np.random.default_rng(99)
    v_all = rng_v.uniform(0.1, 1.0, S)           # fixed per-species values (value-weighted variant)
    err_raw, err_osw, err_osw_v = [], [], []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, S, N)
        counts = np.bincount(idx, minlength=S)
        seen = counts > 0
        n_s = counts[seen]
        # count world (v ≡ 1): truth = E[new species in next t·N | sample]
        w = value_spectrum(n_s, np.ones(seen.sum()))
        truth = float((~seen).sum() * per_unseen)
        raw = float(-sum(((-t) ** j) * wj for j, wj in w.items()))     # raw GT, no smoothing
        osw = osw_horizon_value(w, t, N)["estimate"]
        err_raw.append(abs(raw - truth))
        err_osw.append(abs(osw - truth))
        # value-weighted world: same linearity, spectrum weighted by v_s
        w_v = value_spectrum(n_s, v_all[seen])
        truth_v = float(v_all[~seen].sum() * per_unseen)
        err_osw_v.append(abs(osw_horizon_value(w_v, t, N)["estimate"] - truth_v) / max(truth_v, 1e-9))
    assert np.median(err_osw) < 0.5 * np.median(err_raw)     # the taper beats the raw series
    assert np.median(err_osw) / (300 * 0.6 * per_unseen) < 0.5   # and lands near truth (loose)
    assert np.median(err_osw_v) < 0.5                        # value-weighting carries through
    # delegation + clamp behavior
    w0 = {1: 0.5, 2: 0.2}
    d = osw_horizon_value(w0, 0.7, N)
    assert d["family"] == "et" and np.isclose(d["estimate"], good_toulmin_value(w0, 0.7))
    cl = osw_horizon_value(w0, 50.0, 150)
    assert cl["clamped"] and np.isclose(cl["t"], np.log(150))    # provable-horizon boundary = ln N


def test_certificate_osw_horizon_wiring():
    rng = np.random.default_rng(5)
    sigs, prompts, tags, M = _planted_pool(rng, n_distract=15)
    cert = certificate(sigs, M, tags, prompts=prompts, quotient="behavioral", top_k=8,
                       n_orders=2, c=2.5, seed=0)
    assert cert["horizon_estimator"] == "osw" and np.isclose(cert["c"], 2.5)
    assert cert["c_requested"] == 2.5 and not cert["osw"]["clamped"]     # ln(36) ≈ 3.58 > 2.5
    assert cert["osw"]["k"] >= 1 and np.isfinite(cert["eps_bits_adv"])


def _emp_h2(x):
    x = np.asarray(x, float)
    p = float(np.clip(x.mean(), 1e-12, 1 - 1e-12))
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def test_planted_tail_xor_breaker_A3_is_anticonservative():
    # §12.8.8 C2's designated breaker — the ONE anti-conservative escape, planted and documented
    # (a DOCUMENTED-FAILURE test, like the F₁ XOR-blindspot test).
    #
    # World: M = u_r ⊕ z where z = u_a ⊕ u_b ⊕ u_c flips exactly 3/240 probes — the "judge noise"
    # on an otherwise codifiable rule is secretly the parity of THREE pool criteria. The triple is
    # pairwise-independent with uniform marginals, so every census instrument reads zero BY
    # CONSTRUCTION, not estimator weakness: the species partition finds no merges (pairwise MI = 0),
    # the head rejects each unit (I(u_i; M | u_r) = 0 exactly), γ̂'s sampled blocks measure joint
    # ≈ 0, and all units are RECAPTURED — no unseen mass, no singleton flux. First-order truncation
    # of the chain rule (assumption A3) is exactly what fails: ε certifies while H_emp(M|u_r) bits
    # sit one conjunction away. Permutations, more draws, and richer flux accounting cannot detect
    # it — the mitigations are the composition-covering adversarial probe (asserted via the
    # adv_saturated gate: a composed prompt STATING the full rule executes it) and this breaker
    # traveling with every CODIFIABLE verdict (§12.6.7).
    rng = np.random.default_rng(12)
    n = NPROBE
    u_r = (rng.uniform(0, 1, n) > 0.5).astype(int)
    u_a = (rng.uniform(0, 1, n) > 0.5).astype(int)
    u_b = (rng.uniform(0, 1, n) > 0.5).astype(int)
    z = np.zeros(n, int)
    z[rng.choice(n, 3, replace=False)] = 1                      # exact flip count → ρ̂ = 3/240
    u_c = u_a ^ u_b ^ z                                          # parity triple: z = u_a ⊕ u_b ⊕ u_c
    M = (u_r ^ z).astype(float)
    rows, prompts = [], []
    for name, col in (("rule", u_r), ("alpha", u_a), ("beta", u_b), ("gamma", u_c)):
        for k in range(2):                                       # recaptured — no unseen mass at all
            rows.append(np.clip(col * 0.8 + 0.1 + rng.uniform(-0.04, 0.04, n), 0, 1))
            prompts.append(f"criterion {name} copy {k} of the hidden-parity world")
    for j in range(96):
        d = (rng.uniform(0, 1, n) > 0.5).astype(float) * 0.8 + 0.1
        for k in range(2):
            rows.append(np.clip(d + rng.uniform(-0.04, 0.04, n), 0, 1))
            prompts.append(f"distractor criterion {j} take {k}")
    sigs = np.vstack(rows)
    tags = ["fam%d" % (i % 3) for i in range(len(rows))]

    # each parity unit is individually worthless given the head — exactly, not approximately
    for u in (u_a, u_b, u_c):
        assert _cmi_block(M.astype(int), u_r[:, None].astype(float), u) < 0.02

    cert = certificate(sigs, M, tags, prompts=prompts, quotient="behavioral", top_k=8,
                       n_orders=0, seed=0)
    verdict = ap.decide({"f1_over_N": cert["f1_over_N"]}, certificate=cert)
    assert verdict != "CODIFIABLE"                               # the breaker cannot certify
    assert cert["upper_bound_valid"] is False

    # ...while the parity triple jointly closes the ENTIRE residual: exact plug-in, no estimator —
    # (u_r, u_a, u_b, u_c) determine M, so I_emp(u_a,u_b,u_c; M | u_r) = H_emp(M | u_r).
    hidden = sum(_emp_h2(M[u_r == v] ) * float((u_r == v).mean()) for v in (0, 1))
    assert hidden > 0.05                                          # genuine higher-order residual exists

    # the implemented combiner ladder does NOT see it either (F₂ pair blocks lack the u_r product:
    # 4-parity) — a genuine ESCAPE, not a fixable blind spot
    B_small = np.vstack([u_r, u_a, u_b, u_c] +
                        [(np.random.default_rng(40 + i).uniform(0, 1, n) > 0.5).astype(int)
                         for i in range(6)])
    lin = greedy_head(B_small, M.astype(int), top_k=6, seed=0)
    pr = greedy_head(B_small, M.astype(int), top_k=6, seed=0, combiner="pairs",
                     pair_prerank=len(B_small))
    assert abs(pr["opt_omega_bits"] - lin["opt_omega_bits"]) < 0.05

    # the in-pipeline mitigation: a FAILED composition-covering adversarial probe blocks CODIFIABLE
    assert ap.decide({"f1_over_N": cert["f1_over_N"]},
                     certificate={**cert, "adv_saturated": False}) != "CODIFIABLE"


def test_anytime_delta_allocation_and_checkpoints():
    # §12.8.0 T1: δ_j = δ/(j(j+1)) telescopes — total spend over ALL checkpoints ≤ δ
    assert sum(1.0 / (j * (j + 1)) for j in range(1, 20001)) <= 1.0 + 1e-9
    a16, a32, a1k = (anytime_delta(n, delta=0.05) for n in (16, 32, 1024))
    assert (a16["checkpoint"], a32["checkpoint"], a1k["checkpoint"]) == (1, 2, 7)
    assert a16["delta_eff"] > a32["delta_eff"] > a1k["delta_eff"] > 0
    assert anytime_delta(3, delta=0.05)["checkpoint"] == 1           # sub-n0 clamps to checkpoint 1
    # the union over the n_stats simultaneous statistics divides the level further
    assert np.isclose(anytime_delta(64, n_stats=9)["delta_eff"],
                      anytime_delta(64)["delta_eff"] / 9)
    # checkpoint index is a nondecreasing function of N ALONE (deterministic grid) — this is what
    # makes the union cover ANY adaptive continuation rule
    cps = [anytime_delta(n)["checkpoint"] for n in range(4, 600)]
    assert all(a <= b for a, b in zip(cps, cps[1:]))


def test_certificate_stopping_regimes_order_and_soundness_fields():
    # §12.8.0 T1: the anytime band is never tighter than the declared-budget band (it covers the
    # §12.6.6 continuation loop), and both regimes carry the a-priori B-cap and the order union.
    rng = np.random.default_rng(7)
    sigs, prompts, tags, M = _planted_pool(rng)
    kw = dict(prompts=prompts, quotient="behavioral", top_k=10, n_orders=2, seed=0)
    c_any = certificate(sigs, M, tags, stopping="anytime", **kw)
    c_fix = certificate(sigs, M, tags, stopping="fixed", **kw)
    assert c_any["stopping"] == "anytime" and c_fix["stopping"] == "fixed"
    assert c_any["n_union"] == c_fix["n_union"] == 3                 # canonical + 2 orders
    assert c_any["delta_effective"] < c_fix["delta_effective"] <= 0.05 / 3 + 1e-12
    assert c_fix["checkpoint"] is None and c_any["checkpoint"] >= 1
    assert c_any["slack"] >= c_fix["slack"] - 1e-12                  # sound ≥ declared-budget
    assert c_any["eps_bits_adv"] >= c_fix["eps_bits_adv"] - 1e-12
    # Optional-stopping allocation can be sound while the horizon/synergy bridge remains invalid.
    assert c_any["upper_bound_valid"] is False
    assert ap.decide({"f1_over_N": c_any["f1_over_N"]}, certificate=c_any) != "CODIFIABLE"


def test_tail_gamma_bounds_and_empty_tail():
    rng = np.random.default_rng(2)
    B = (rng.uniform(0, 1, (10, NPROBE)) > 0.5).astype(int)
    M = B[0]
    g = tail_gamma(B, M, selected=[0], seed=0)
    assert 0.0 < g <= 1.0
    assert tail_gamma(B, M, selected=list(range(10)), seed=0) == 1.0   # empty tail ⇒ γ̂ = 1
