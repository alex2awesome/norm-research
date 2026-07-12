"""Planted-ground-truth tests for the ALPHA-PROBE species-accounting layer
(experiments/alpha_probe.py, theory §12.1a). All CPU, no model.

Gates the §12.1a construction the way test_orthogonalize gates §6.5: on planted data with a KNOWN
answer, the estimators must satisfy their advertised invariants (rarefaction monotone & ≤ Chao1,
C_lo ≤ C_hat, heaps_α recovers a planted power-law), collide must merge behavioral duplicates and
split orthogonal signatures, and decide() must route the three regimes correctly.
"""
from __future__ import annotations

import numpy as np

from methods.metric_implementer.experiments.alpha_probe import (
    _rarefy, alpha_probe, chao1, cmi_agreement, collide, coverage_fienberg,
    coverage_interval, coverage_pairwise_petersen, decide, decide_census, diversity_gap,
    good_turing_coverage, heaps_alpha, missing_mass_bound, quotient_species, rarefaction,
    spectrum,
)

N = 300


def _spectrum_from_labels(labels, family_tags=None):
    """Drive spectrum() off integer labels for estimator tests (decouples from collide/model)."""
    if family_tags is None:
        family_tags = ["fam0"] * len(labels)
    return spectrum(labels, family_tags)


def test_good_turing_in_unit_interval_and_mass_bound():
    # f_1=10 singletons among N=100 → Ĉ=0.9; CI eps positive; m0_ci_hi ≤ 1
    f = {1: 10, 2: 5, 3: 3}
    N = sum(j * c for j, c in f.items())
    c_hat = good_turing_coverage(f, N)
    assert 0.0 <= c_hat <= 1.0
    assert np.isclose(c_hat, 1 - 10 / N)
    mm = missing_mass_bound(f, N, delta=0.05)
    assert mm["ci_eps"] > 0
    assert mm["m0_ci_hi"] <= 1.0
    assert mm["m0_ci_hi"] >= mm["m0_hat"]          # upper bound is above the point estimate


def test_chao1_bounds_and_singleton_branch():
    # f_2>0 branch: Chao1 = D + f1^2/(2 f2)
    f = {1: 10, 2: 5}
    D = 15
    assert np.isclose(chao1(f, D), 15 + 100 / 10.0)
    assert chao1(f, D) >= D                          # richness ≥ observed distinct
    # singleton-only branch: f_2=0 → D + f1(f1-1)/2
    f2 = {1: 6}
    assert np.isclose(chao1(f2, 6), 6 + 6 * 5 / 2.0)


def test_rarefaction_monotone_and_bounded_by_chao():
    f = {1: 12, 2: 8, 3: 4, 5: 2}
    N = sum(j * c for j, c in f.items())
    D = sum(f.values())
    ms, S = rarefaction(f, N, n_points=30)
    assert ms[0] == 1 and ms[-1] == N
    assert np.all(np.diff(S) >= -1e-9)              # E[S(m)] non-decreasing in m
    assert S[-1] == D                               # full sample recovers all distinct
    assert _rarefy(f, N, 1) >= 0
    # rarefied richness never exceeds the Chao1 richness lower bound
    assert np.all(S <= chao1(f, D) + 1e-6)


def test_heaps_alpha_recovers_powerlaw():
    # E[S(m)] = m^0.5 ⇒ α(m) ≈ 0.5 on interior points
    ms = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=float)
    S = ms ** 0.5
    a = heaps_alpha(ms, S)
    interior = a[2:-2]
    assert np.all(np.abs(interior - 0.5) < 0.05)
    # S = m^1 (linear) ⇒ α ≈ 1
    a1 = heaps_alpha(ms, ms)
    assert np.all(np.abs(a1[2:-2] - 1.0) < 0.05)


def test_collide_merges_duplicates_splits_orthogonal():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.2, 0.8, size=N)
    dup = base + rng.uniform(-0.01, 0.01, size=N)          # within τ
    ortho = (base > 0.5).astype(float) * 0.4 + 0.05         # very different signature
    sigs = np.vstack([base, dup, ortho])
    lab = collide(sigs, tau=0.05)
    assert lab[0] == lab[1]                                 # base & dup → same species
    assert lab[2] != lab[0]                                 # ortho → its own species
    # all-identical → one species
    five = np.vstack([base] * 5)
    assert len(set(collide(five, tau=1e-6))) == 1
    # mutually orthogonal → 5 species at small τ
    ortho5 = np.vstack([(rng.uniform(0, 1, N) > 0.5).astype(float) * rng.uniform(0.6, 0.9)
                        for _ in range(5)])
    assert len(set(collide(ortho5, tau=0.05))) == 5


def test_cmi_agreement_coherent_cut():
    # two tight clusters of duplicated signatures → within-species CMI ≪ between-species
    rng = np.random.default_rng(1)
    a = (rng.uniform(0, 1, N) > 0.5).astype(float) * 0.8 + 0.1
    b = (rng.uniform(0, 1, N) > 0.5).astype(float) * 0.8 + 0.1
    sigs = np.vstack([a, a + 1e-3, b, b + 1e-3])
    lab = collide(sigs, tau=0.05)
    chk = cmi_agreement(sigs, lab, seed=0)
    assert chk["coherent"]                                 # the cut is behaviorally meaningful


def test_coverage_interval_lo_le_hat():
    f = {1: 15, 2: 10, 3: 6, 4: 3}
    N = sum(j * c for j, c in f.items())
    ci = coverage_interval(f, N, delta=0.05)
    assert ci["C_lo"] <= ci["C_hat"] + 1e-9                # conservative bound ≤ optimistic point
    assert 0.0 <= ci["C_lo"] <= 1.0
    assert ci["C_hat"] <= 1.0


def test_coverage_pairwise_petersen_sanity():
    # one species shared by both families, one unique each → raw Petersen coverage well-defined, ≤ 1.
    # positive_only=False: the sanity check wants the raw value (the default skips non-positive pairs).
    spec = spectrum([0, 0, 1, 2], ["A", "B", "A", "B"])
    c = coverage_pairwise_petersen(spec, positive_only=False)
    assert np.isfinite(c) and 0.0 < c <= 1.0 + 1e-6


def test_coverage_fienberg_k2_recovers_petersen():
    # 4 species in BOTH lists, 2 in A-only, 3 in B-only → D=9.
    # Lincoln–Petersen empty cell n_10·n_01/n_11 = 2·3/4 = 1.5 ⇒ C = 9/10.5 = 0.857.
    labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8]
    tags = ["A", "B", "A", "B", "A", "B", "A", "B", "A", "A", "B", "B", "B"]
    spec = spectrum(labels, tags)
    assert spec["D"] == 9
    cf = coverage_fienberg(spec)
    assert cf["K"] == 2
    assert abs(cf["n_empty_independence"] - 1.5) < 0.05          # Petersen empty-cell estimate
    assert abs(cf["C_independence"] - 9 / 10.5) < 0.02
    # K=2 ⇒ pairwise model saturated ⇒ empty cell NOT identifiable
    assert cf["pairwise_identified"] is False
    assert np.isnan(cf["C_pairwise"])


def test_coverage_fienberg_k4_identifies_pairwise_and_bounded():
    # 4 lists, enough species that 2^4-1=15 cells > 11 params ⇒ pairwise identified.
    rng = np.random.default_rng(3)
    fams = ["A", "B", "C", "D"]
    labels, tags = [], []
    sp = 0
    for _ in range(40):
        k = 1 + int(rng.integers(0, 4))                       # 1..4 families capture this species
        cap = rng.choice(4, size=k, replace=False)
        for c in cap:
            labels.append(sp); tags.append(fams[int(c)])
        sp += 1
    spec = spectrum(labels, tags)
    cf = coverage_fienberg(spec)
    assert cf["K"] == 4
    assert cf["pairwise_identified"] is True
    for key in ("C_independence", "C_pairwise"):
        v = cf[key]
        assert np.isfinite(v) and 0.0 <= v <= 1.0 + 1e-6
    # both estimates must lie above the assumption-free certificate ceiling 1.0 trivially; the
    # dependence-corrected pairwise is a finite point, not a bound — just sanity it's not wild
    assert cf["n_empty_pairwise"] >= 0.0


def test_diversity_gap_pooled_exceeds_richest_family():
    # two families with DISJOINT species and mixed multiplicities → pooled rarefied to M is strictly
    # richer than the richest single family at M. (Singleton-only families are degenerate: their
    # rarefaction is flat, so Δ_div collapses to 0 by construction — use f_2>0.)
    per_f = {"A": {1: 4, 2: 2}, "B": {1: 4, 2: 2}}
    per_N = {"A": 8, "B": 8}
    # pooled spectrum from GLOBAL labels (disjoint families): 8 species seen once, 4 seen twice
    pooled_f, pooled_N = {1: 8, 2: 4}, 16
    dg = diversity_gap(pooled_f, pooled_N, per_f, per_N)
    assert np.isfinite(dg["delta_div"])
    assert dg["delta_div"] > 0                      # pooled finds more than the richest family


def test_decide_census_routes_three_regimes():
    # the ORIGINAL count-axis rule, retained as the descriptive vocabulary read (decide_census)
    base = {"chao1": 30.0, "D": 30, "diversity_gap": {"delta_div": 0.0},
            "alpha_terminal": 0.2, "coverage": {"C_lo": 0.97}}
    assert decide_census(dict(base)) == "GO"                # low α, high C_lo, saturated
    nogo_alpha = dict(base); nogo_alpha["alpha_terminal"] = 0.95
    assert decide_census(nogo_alpha) == "NO-GO"             # α ~ 1
    ambig = dict(base); ambig["coverage"] = {"C_lo": 0.50}  # low α but coverage NOT established
    assert decide_census(ambig) == "AMBIGUOUS"
    nogo_div = dict(base); nogo_div["diversity_gap"] = {"delta_div": 10.0}  # Δ_div large
    assert decide_census(nogo_div) == "NO-GO"


def test_decide_v2_routes_verdicts():
    # §12.6.6: UNDERSAMPLED / FORM-DOMINATED / NEEDS-CERTIFICATE / CODIFIABLE / DEEP / INDETERMINATE
    assert decide({"f1_over_N": 0.95}) == "UNDERSAMPLED"    # Lemma 12.6.0 singleton regime
    ok = {"f1_over_N": 0.2}
    assert decide(dict(ok), form_invariance={"form_invariant": False}) == "FORM-DOMINATED"
    assert decide(dict(ok)) == "NEEDS-CERTIFICATE"          # census alone can't issue a depth verdict
    cod = {"eps_bits": 0.01, "H_M": 1.0, "tail_frac": 0.05, "hill": 0.2,
           "adv_saturated": True, "upper_bound_valid": True}
    assert decide(dict(ok), certificate=cod) == "CODIFIABLE"
    # adv-saturation failure blocks CODIFIABLE (ε already small + light tail ⇒ synergy escape, not an
    # undersampling problem — INDETERMINATE, more draws won't fix it)
    cod_adv = dict(cod); cod_adv["adv_saturated"] = False
    assert decide(dict(ok), certificate=cod_adv) == "INDETERMINATE"
    cod_missing_adv = dict(cod); cod_missing_adv["adv_saturated"] = None
    assert decide(dict(ok), certificate=cod_missing_adv) == "INDETERMINATE"
    cod_unvalidated = dict(cod); cod_unvalidated["upper_bound_valid"] = False
    assert decide(dict(ok), certificate=cod_unvalidated) == "INDETERMINATE"
    deep = {"eps_bits": 0.4, "H_M": 1.0, "tail_frac": 0.6, "hill": 0.3, "adv_saturated": None}
    assert decide(dict(ok), certificate=deep) == "DEEP"     # heavy greedy tail + recapturing stream
    # ε unresolved (Ĝ+slack still shrink with N) with a light tail ⇒ the cure is more draws
    ind = {"eps_bits": 0.3, "H_M": 1.0, "tail_frac": 0.1, "hill": 0.2, "adv_saturated": None}
    assert decide(dict(ok), certificate=ind) == "UNDERSAMPLED"
    # the f₁/N gate is ASYMMETRIC (§12.6.2/§12.6.6): over-split singletons carry v|S_g≈0 and can only
    # INFLATE ε, so an ε-small certificate is conservative at ANY f₁/N — CODIFIABLE survives the
    # singleton regime…
    assert decide({"f1_over_N": 0.95}, certificate=cod) == "CODIFIABLE"
    # …but DEEP does not (theory table: ∧ f₁/N ≪ 1): spread greedy gains in an undersampled pool may
    # just mean the head criteria haven't arrived yet
    assert decide({"f1_over_N": 0.95}, certificate=deep) == "UNDERSAMPLED"


def test_form_band_mode_keeps_metric_and_quantifies_fragility():
    from methods.metric_implementer.experiments.alpha_probe import eps_form_bits
    ok = {"f1_over_N": 0.2}
    cod = {"eps_bits": 0.01, "H_M": 1.0, "tail_frac": 0.05, "hill": 0.2,
           "adv_saturated": True, "upper_bound_valid": True, "opt_omega_bits": 0.6}
    frag = {"form_invariant": False, "binary_flip_rate": 0.30}   # form-fragile metric
    # verdict mode (default, unchanged): fragility EJECTS the metric
    assert decide(dict(ok), form_invariance=frag, certificate=cod) == "FORM-DOMINATED"
    # band mode: metric KEEPS its census verdict; fragility is an orthogonal reported axis, not a
    # gate input — even a heavily form-fragile metric stays CODIFIABLE by the census
    assert decide(dict(ok), form_invariance=frag, certificate=cod, form_mode="band") == "CODIFIABLE"
    heavy = {"form_invariant": False, "binary_flip_rate": 0.49}
    assert decide(dict(ok), form_invariance=heavy, certificate=cod, form_mode="band") == "CODIFIABLE"
    # a DEEP metric likewise keeps its verdict in band mode (nothing is ejected)
    deep = {"eps_bits": 0.4, "H_M": 1.0, "tail_frac": 0.6, "hill": 0.3, "opt_omega_bits": 0.4}
    assert decide(dict(ok), form_invariance=frag, certificate=deep, form_mode="band") == "DEEP"
    # ε_form (ceiling-widening for Δ): non-negative, 0 without form data, monotone in flip rate,
    # and self-labeling (source ∈ exact/proxy/none) so a proxy is never mistaken for the real number
    assert eps_form_bits(None, cod) == {"bits": 0.0, "source": "none", "q": None}
    assert eps_form_bits({"binary_flip_rate": 0.0}, cod)["bits"] == 0.0
    lo, hi = eps_form_bits({"binary_flip_rate": 0.10}, cod), eps_form_bits({"binary_flip_rate": 0.40}, cod)
    assert hi["bits"] > lo["bits"] > 0 and lo["source"] == "proxy"
    assert abs(lo["bits"] - 0.10 * 0.6) < 1e-9                # linear proxy q·OPT_Ω
    # exact per-form ceilings, when present, take precedence and are labeled 'exact'
    exact = dict(cod); exact["opt_omega_by_form"] = {"canonical": 0.6, "question": 0.72, "reorder": 0.55}
    ef = eps_form_bits(frag, exact)
    assert ef["source"] == "exact" and abs(ef["bits"] - 0.12) < 1e-9   # max(0.72,0.55) − 0.60
    # invariant, mode-agnostic verdict for a robust metric
    robust = {"form_invariant": True, "binary_flip_rate": 0.02}
    assert decide(dict(ok), form_invariance=robust, certificate=cod, form_mode="band") == "CODIFIABLE"
    assert decide(dict(ok), form_invariance=robust, certificate=cod) == "CODIFIABLE"
    # guardrail: unknown form_mode is a hard error, not silent
    import pytest
    with pytest.raises(ValueError):
        decide(dict(ok), certificate=cod, form_mode="bogus")


def test_quotient_species_merges_only_with_judge_and_never_splits():
    # three strict species: A (balanced signal), A' (a paraphrase of A that behaves DIFFERENTLY —
    # the form-fragile beh-DIFF & sem-SAME cell: low MI with A), and B (a distinct criterion).
    rng = np.random.default_rng(5)
    a = (rng.uniform(0, 1, N) > 0.5).astype(float) * 0.8 + 0.1
    a_frag = (rng.uniform(0, 1, N) > 0.5).astype(float) * 0.8 + 0.1     # independent pattern
    b = (rng.uniform(0, 1, N) > 0.5).astype(float) * 0.8 + 0.1
    sigs = np.vstack([a, a + 0.01, a_frag, b])
    prompts = ["the story maintains a consistent tone throughout the narrative",
               "the story maintains a consistent tone throughout the narrative",   # dup of A
               "the narrative maintains a consistent tone throughout the story",   # paraphrase, frag. behavior
               "the plot resolves every open thread by the ending"]
    # no judge: strict behavioral partition only — paraphrase stays split, audit reports unresolved
    lab0, aud0 = quotient_species(sigs, prompts, judge_fn=None)
    assert lab0[0] == lab0[1]                              # behavioral near-duplicates merged
    assert lab0[2] != lab0[0]                              # form-fragile paraphrase NOT merged (no judge)
    assert aud0["n_merged"] == 0 and aud0["n_unresolved"] >= 1
    # judge that answers by lexical rule (SAME iff shingle overlap high) merges the paraphrase pair
    from methods.metric_implementer.experiments.alpha_probe import _shingle_jaccard
    judge = lambda pairs: [_shingle_jaccard(x, y, k=1) > 0.8 for x, y in pairs]
    lab1, aud1 = quotient_species(sigs, prompts, judge_fn=judge)
    assert lab1[2] == lab1[0]                              # semantic-judge MERGE across strict species
    assert lab1[3] != lab1[0]                              # distinct criterion stays split
    assert aud1["n_merged"] >= 1
    assert aud1["n_quotient"] < aud1["n_strict"]
    # judge NEVER splits: behavioral collisions (rows 0,1) share a species under every judge
    always_diff = lambda pairs: [False for _ in pairs]
    lab2, _ = quotient_species(sigs, prompts, judge_fn=always_diff)
    assert lab2[0] == lab2[1]


def test_alpha_probe_end_to_end_lowdim_goes_go_highdim_goes_nogo():
    rng = np.random.default_rng(7)
    families = ["A", "B", "C", "D"]
    # LOW-DIM: 4 behavioral archetypes, each drawn many times → saturates → small α. Families are
    # DECOUPLED from archetypes (each family spans all 4) so Δ_div≈0 — coupling them 1:1 would make
    # each family see one species and trip NO-GO on Δ_div, which is correct for THAT pathology.
    archetypes = [(rng.uniform(0, 1, N) > 0.5).astype(float) * 0.7 + 0.15 for _ in range(4)]
    sigs_lo, tags_lo = [], []
    for i in range(40):
        k = rng.integers(0, 4)
        sigs_lo.append(archetypes[k] + rng.uniform(-0.01, 0.01, N))
        tags_lo.append(families[i % 4])                     # family = draw-index, not archetype
    rep_lo = alpha_probe(np.vstack(sigs_lo), tags_lo, tau=0.05)
    assert rep_lo["decision_census"] in {"GO", "AMBIGUOUS"}  # AMBIGUOUS honest at N=40 (C_lo loose)
    assert rep_lo["D"] <= 8                                 # collapsed to ~the archetypes (+jitter splits)
    assert rep_lo["alpha_terminal"] < 0.9
    assert rep_lo["f1_over_N"] < 0.5                        # recapture present ⇒ exponents readable
    assert rep_lo["decision"] == "NEEDS-CERTIFICATE"        # §12.6.6: census alone ⇒ no depth verdict
    # HIGH-DIM: every draw a fresh random signature → inexhaustible → α ~ 1
    sigs_hi = np.vstack([((rng.uniform(0, 1, N) > 0.5).astype(float) * rng.uniform(0.6, 0.9) + 0.05)
                         for _ in range(120)])
    tags_hi = [families[i % 4] for i in range(120)]
    rep_hi = alpha_probe(sigs_hi, tags_hi, tau=0.05)
    assert rep_hi["decision_census"] == "NO-GO"
    assert rep_hi["D"] > 80
    # Lemma 12.6.0: singleton-dominated spectrum ⇒ α pinned ≈ 1 AND the §12.6.6 gate fires
    assert rep_hi["f1_over_N"] > 0.8
    assert rep_hi["decision"] == "UNDERSAMPLED"
    assert rep_hi["alpha_terminal"] > 0.9
