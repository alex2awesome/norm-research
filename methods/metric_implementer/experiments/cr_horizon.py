"""[STATUS 2026-07-11: DESCRIPTIVE DIAGNOSTIC — superseded for certified claims by cr_audit.py]
External review (accepted): the estimand here (one-doubling Good-Toulmin flux, kappa/lambda
rescaled to a battery's pool truth) does not match a pool endpoint; the freezing premise of the
slack term is unmet (species/head/values share the capture stream; only PROBES are split); the
stream is family-stratified, not iid; battery test coverage 30/36=.83 (95% CI ~.67-.94) does not
support bound semantics; additive per-species marks can exceed H(M) (humor: G1=2.68 vs H=.27).
USE: saturation detection + component decomposition ONLY. Certified missing-mass / horizon /
value bounds live in cr_audit.cr3_certificate (discovery/audit split, exact CP + emp-Bernstein).

CR-2 — rehabilitated capture-recapture horizon estimation (user decision 2026-07-10).

Scope decision (supersedes the fixed-target-only reading of the 2026-07-10 audit): the project's
target claim is CONCEPT-LEVEL — "how much better could this metric possibly be?" — not recovery of
a rubric-as-stated. Two instruments, cleanly separated:

  1. ``cr2_certificate`` — POOL-HORIZON ESTIMATE for a fixed target M_i: what the mining process
     would deliver at its horizon (unseen criteria + unseen synergy priced in). Capped by H(M);
     the DPI cap T(M_i) still binds any fixed instantiation. This prices POOL incompleteness —
     the thing the audit showed the old ``epsilon`` mis-priced.
  2. ``concept_horizon`` — INSTANTIATION-HORIZON ESTIMATE: across many re-instantiations of the
     metric (paraphrases, re-rubrications, GEPA-of-target rounds), how high could T itself go?
     "That's how much more self-consistent it could be."

Design (v3 — after the 2026-07-10 adversarial verification pass; v1/v2 flaws each map to a fix):
  PROBE-SPLIT      head selected on half A; every value/flux/synergy statistic computed on half B
                   with the head frozen; both directions, adverse end. The A/B split is a
                   FROZEN-SEED random permutation (not index parity — systematic probe ordering
                   would bias parity splits). Species partition + representatives + candidate
                   strata are ALL fixed from half A; B is spent only on evaluation. Residual
                   within-B selection in the pair chain is deliberate and is part of what the
                   calibration battery prices (worlds share the exact procedure).
  GATES            every single-species value and every pair excess must clear its own
                   permutation-null quantile before carrying mass; below-gate => exactly zero.
                   Values that pass are used RAW (never null-mean-subtracted: Good–Toulmin's
                   alternating series does not commute with constant shifts). The bulk stratum,
                   whose statistic is multiplied by ~K^2/2 pairs, uses a multiplicity-adjusted
                   gate z = Phi^-1(1 - 0.5/n_rest) so expected false-positive mass < one pair.
  PAIR NULL        conditional permutation: one member's rows are permuted WITHIN y-strata,
                   preserving both units' marginal relation to M while breaking their pairing —
                   the null is "both related to M, no interaction", not "unrelated feature".
  SYNERGY          exhaustive pairs among A-chosen top-k + A-chosen stable-null suspects (the
                   parity fingerprint), aggregated by a greedy CHAIN (each pair priced as its
                   excess-over-individuals conditional on everything already priced — overlapping
                   pairs cannot double count; the chain's repeated testing gets its own
                   multiplicity-adjusted gate), plus a gated random bulk stratum. Synergy with a
                   NEVER-CAPTURED partner is a declared blind spot, so CR-2 REFUSES to
                   extrapolate unseen-pair growth (v3 of this file priced it via a quadratic
                   Chao factor and manufactured ~9 phantom bits on pure noise — the term is
                   structurally explosive and extrapolates into the blind spot by definition).
                   Chao1 unseen-species counts are reported as descriptive context only.
  CALIBRATION      ONE multiplier ``lam`` on the gap term, fit on the TRAIN split of a declared
                   planted-world battery (analytic truth I(M;R) = H(M) - H_b(flip)) and then
                   REPORTED on the disjoint TEST split — the coverage number quoted is always
                   out-of-sample. fit_lambda also reports tightness (mean estimate - truth) and
                   vacuity (how often the estimate just hits the H cap): a "bound" that always
                   answers H(M) is coverage-perfect and worthless, so vacuity must stay low.
  BLIND SPOTS      declared, measured, never averaged into coverage: (i) unseen-by-UNREACHABILITY
                   (true criteria at ~zero capture probability — the user-accepted limitation:
                   no frequency method can see them); (ii) parity3+ (synergy deeper than pairs);
                   (iii) xor2-with-hidden-partner (pair synergy whose partner was never captured
                   leaves no scent in a gated stream).

Everything here is ESTIMATE-grade: ``certified=False`` on every output; the number's trust
credential is its out-of-sample planted-world coverage, shipped alongside. No edits to
``value_certificate.py`` — its historical fields stay frozen; CR-2 is the successor instrument.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from methods.metric_implementer import vinfo
from methods.metric_implementer.experiments.value_certificate import (
    _cmi_block, good_toulmin_value, greedy_head)

_EPS = 1e-12
STREAM_EXCLUDE = ("children", "gepa")          # guard G1: only frozen-iid proposal draws recapture


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------

def _halves(n: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Frozen-seed random permutation split (reproducible; immune to systematic probe order)."""
    perm = np.random.default_rng(seed).permutation(n)
    a = np.zeros(n, bool)
    a[perm[: n // 2]] = True
    return a, ~a


def _h_bits(p: float) -> float:
    return float(vinfo._h_bits(p))


def leader_species(binmat_half: np.ndarray, idx: Sequence[int], *, agree_tau: float = 0.9) -> Dict[int, List[int]]:
    """Greedy leader clustering of stream units into behavioral species on ONE probe half (the
    selection half — the partition must not see the evaluation half). Uses NO target information.
    Returns {leader_idx: [member idx...]}."""
    leaders: Dict[int, List[int]] = {}
    for k in idx:
        placed = False
        for l in leaders:
            if float(np.mean(binmat_half[k] == binmat_half[l])) >= agree_tau:
                leaders[l].append(k)
                placed = True
                break
        if not placed:
            leaders[k] = [k]
    return leaders


def _s_cols(binmat_half: np.ndarray, selected: Sequence) -> np.ndarray:
    """Rebuild the head's conditioning matrix on a probe half from the FROZEN selection (ints and
    ('pair', i, j) product triples), mirroring greedy_head's construction."""
    cols = []
    for e in selected:
        if isinstance(e, (tuple, list)) and len(e) == 3 and e[0] == "pair":
            i, j = int(e[1]), int(e[2])
            cols += [binmat_half[i], binmat_half[j], binmat_half[i] * binmat_half[j]]
        else:
            cols.append(binmat_half[int(e)])
    if not cols:
        return np.empty((binmat_half.shape[1], 0), float)
    return np.stack(cols, axis=1).astype(float)


def _pblock(Bh: np.ndarray, ci: np.ndarray, cj: np.ndarray) -> np.ndarray:
    return np.stack([ci, cj, ci * cj], axis=1).astype(float)


def _strata_perm(col: np.ndarray, y: np.ndarray, rng) -> np.ndarray:
    """Permute a column WITHIN y-strata: preserves the unit's marginal relation to the target
    while destroying its pairing with any other unit — the conditional-independence null."""
    out = col.copy()
    for v in (0, 1):
        m = np.flatnonzero(y == v)
        out[m] = col[m[rng.permutation(len(m))]]
    return out


def pool_truth_bits(rule: str, flip: float, unit_noise: float, z_bias: float = 0.5) -> float:
    """EXACT I(M; U_true) by joint enumeration: the information the FULL pool of noisy true units
    carries about M. This — not the concept ceiling I(M;R) — is what a pool-horizon estimator must
    be calibrated against: noisy units cannot reach I(M;R) even fully mined (unit-noise
    attenuation), and calibrating against the unreachable target either fails or forces vacuity.
    I(M;R) stays the CONCEPT ceiling, the concept_horizon instrument's business."""
    kmap = {"linear3": 3, "linear5": 5, "and3": 3, "xor2": 2, "parity3": 3}
    K = kmap[rule]
    n_u = 2 ** K
    joint = np.zeros((2, n_u))
    for zi in range(2 ** K):
        z = [(zi >> k) & 1 for k in range(K)]
        if rule.startswith("linear"):
            r = int(sum(z) >= int(np.ceil(K / 2)))
        elif rule == "and3":
            r = int(sum(z) == K)
        else:
            r = int(sum(z) % 2 == 1)
        pz = 1.0
        for k in range(K):
            pz *= z_bias if ((zi >> k) & 1) else (1.0 - z_bias)
        for ui in range(n_u):
            pu = 1.0
            for k in range(K):
                u_k = (ui >> k) & 1
                pu *= (1 - unit_noise) if u_k == z[k] else unit_noise
            joint[1, ui] += pz * pu * ((1 - flip) if r == 1 else flip)
            joint[0, ui] += pz * pu * (flip if r == 1 else (1 - flip))
    pm = joint.sum(axis=1)
    pu_ = joint.sum(axis=0)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log2(joint[nz] / np.outer(pm, pu_)[nz])))


def _chao1_unseen(f1: int, f2: int) -> float:
    """Bias-corrected Chao1 unseen-species estimate (f2=0 form: f1(f1-1)/2)."""
    return float(f1 * f1 / (2.0 * f2)) if f2 > 0 else float(f1 * max(f1 - 1, 0) / 2.0)


# --------------------------------------------------------------------------------------------
# the CR-2 pool-horizon estimate (fixed target)
# --------------------------------------------------------------------------------------------

def cr2_certificate(sigs: np.ndarray, M: np.ndarray, tags: Sequence, *, top_k: int = 8,
                    combiner: str = "linear", agree_tau: float = 0.9, topk_pairs: int = 10,
                    suspect_cap: int = 40, n_rand_pairs: int = 100, n_null: int = 40,
                    gate_z: float = 2.5, chain_cap: int = 8, chain_pool: int = 30,
                    unseen_ratio_cap: float = 10.0, lam: Optional[float] = None, kappa: float = 1.0,
                    split_seed: int = 0, delta: float = 0.05, seed: int = 0) -> dict:
    """Split-sample, gated, synergy-priced pool-horizon ESTIMATE for one (metric, executor)
    checkpoint. Every component is reported so the estimate can be audited term by term.
    ``lam`` should come from ``fit_lambda`` on the planted battery's TRAIN split (None -> 1.0,
    flagged uncalibrated). Output is an estimate with declared out-of-sample calibration — it is
    never a certified bound."""
    S = np.asarray(sigs, float)
    m = np.asarray(M, float)
    B_all = (S > 0.5).astype(int)
    y_all = (m > 0.5).astype(int)
    tags = [str(t) for t in tags]
    stream_idx = [k for k in range(len(tags)) if not any(tags[k].startswith(x) for x in STREAM_EXCLUDE)]
    if not stream_idx:
        return {"valid": False, "error": "no_stream_units", "certified": False}
    ev, od = _halves(m.size, seed=split_seed)

    directions = []
    for di, (a_mask, b_mask, dtag) in enumerate([(ev, od, "A->B"), (od, ev, "B->A")]):
        rng = np.random.default_rng(seed * 7919 + di)          # per-direction stream (C1)
        Ba, Bb = B_all[:, a_mask], B_all[:, b_mask]
        ya, yb = y_all[a_mask], y_all[b_mask]
        if len(np.unique(yb)) < 2 or len(np.unique(ya)) < 2:
            directions.append({"direction": dtag, "degenerate": True, "H_M": 0.0,
                               "value_frozen": 0.0, "raw_gap": 0.0, "slack": 0.0})
            continue
        h_b = _h_bits(float(yb.mean()))
        empty_a = np.empty((len(ya), 0), float)
        empty_b = np.empty((len(yb), 0), float)

        # -- head on A, frozen; value on B ----------------------------------------------------
        head = greedy_head(Ba, ya, top_k=top_k, combiner=combiner, seed=seed)
        sel = head["selected"]
        Scols_b = _s_cols(Bb, sel)
        value_b = float(max(0.0, _cmi_block(yb, empty_b, Scols_b, seed=seed))) if sel else 0.0

        # -- species partition, representatives, candidate strata: ALL from half A (S2) -------
        species = leader_species(Ba, stream_idx, agree_tau=agree_tau)
        add_a = {k: float(max(0.0, _cmi_block(ya, empty_a, Ba[k][:, None], seed=seed)))
                 for k in stream_idx}
        reps = [max(mem, key=lambda k: add_a[k]) for mem in species.values()]
        n_s = np.array([len(mem) for mem in species.values()], int)
        N_stream = int(n_s.sum())
        K_sp = len(reps)

        def v_of(col: np.ndarray) -> float:
            return float(_cmi_block(yb, Scols_b, col[:, None], seed=seed))

        # -- gated single-species values on B: raw-if-significant, zero otherwise (S1) --------
        null_s = [v_of(Bb[int(rng.choice(reps))][rng.permutation(len(yb))]) for _ in range(n_null)]
        mu_s, sd_s = float(np.mean(null_s)), float(np.std(null_s) + 1e-9)
        v_raw = np.array([v_of(Bb[r]) for r in reps])
        v_gate = np.where(v_raw > mu_s + gate_z * sd_s, np.maximum(v_raw, 0.0), 0.0)

        # -- capture spectrum of gated value; GT one-doubling gain; Chao1 (S8) ----------------
        w: Dict[int, float] = {}
        for j in np.unique(n_s):
            w[int(j)] = float(v_gate[n_s == j].sum())
        f1, f2 = int((n_s == 1).sum()), int((n_s == 2).sum())
        G1 = float(max(0.0, good_toulmin_value(w, 1.0)))
        chao_unseen = _chao1_unseen(f1, f2)
        unseen_ratio = float(min(chao_unseen / max(K_sp, 1), unseen_ratio_cap))

        # -- pair synergy ----------------------------------------------------------------------
        v_by_rep = {r: float(v_raw[k]) for k, r in enumerate(reps)}
        pair_seen = pair_unseen = chain_total = bulk = 0.0
        picks, n_sig = 0, 0
        mu_p = sd_p = float("nan")
        if K_sp >= 2:
            def syn0(i: int, j: int) -> float:
                return (float(_cmi_block(yb, Scols_b, _pblock(Bb, Bb[i], Bb[j]), seed=seed))
                        - v_by_rep[i] - v_by_rep[j])

            # conditional-permutation null (S5): permute one member WITHIN y-strata
            null_p = []
            for _ in range(n_null):
                i, j = (int(x) for x in rng.choice(reps, 2, replace=False))
                ci = _strata_perm(Bb[i], yb, rng)
                null_p.append(float(_cmi_block(yb, Scols_b, _pblock(Bb, ci, Bb[j]), seed=seed))
                              - v_of(ci) - v_by_rep[j])
            mu_p, sd_p = float(np.mean(null_p)), float(np.std(null_p) + 1e-9)
            p_gate = mu_p + gate_z * sd_p

            # candidate strata chosen on A (add_a), evaluated on B
            rep_by_add = sorted(reps, key=lambda r: -add_a[r])
            top = rep_by_add[: min(topk_pairs, K_sp)]
            add_med = float(np.median([add_a[r] for r in reps]))
            stds_a = Ba.std(axis=1)
            suspects = [r for r in reps if add_a[r] <= add_med and stds_a[r] > 0.1][:suspect_cap]
            ex1 = [(top[a], t) for a in range(len(top)) for t in top[a + 1:]]
            ex2 = [(suspects[a], suspects[b]) for a in range(len(suspects))
                   for b in range(a + 1, len(suspects))]
            if len(ex2) > 800:                      # score ALL suspect pairs when feasible —
                ex2 = [ex2[i] for i in rng.choice(len(ex2), 800, replace=False)]
                # (subsampling here once lost the true XOR pair 74% of the time; it then leaked
                # through the bulk stratum with n_rest amplification — v3 smoke finding)
            cand = list(dict.fromkeys(ex1 + ex2))
            scored = [(i, j, syn0(i, j)) for i, j in cand]
            sig_pairs = [(i, j, s) for i, j, s in scored if s > p_gate]
            n_sig = len(sig_pairs)

            # greedy chain: excess beyond individuals, conditional on everything priced so far.
            # The chain re-tests <= chain_pool x chain_cap statistics, so its acceptance gate is
            # multiplicity-adjusted (winner's curse over ~240 draws walks straight past 2.5 sd).
            from scipy.stats import norm as _norm
            Sc = Scols_b.copy()
            pool = sorted(sig_pairs, key=lambda t: -t[2])[:chain_pool]
            n_chain_tests = max(len(pool) * chain_cap, 1)
            chain_gate = mu_p + max(gate_z, float(_norm.ppf(1.0 - 0.05 / n_chain_tests))) * sd_p
            while pool and picks < chain_cap:
                best = None
                for i, j, _ in pool:
                    gp = float(_cmi_block(yb, Sc, _pblock(Bb, Bb[i], Bb[j]), seed=seed))
                    gi = float(_cmi_block(yb, Sc, Bb[i][:, None], seed=seed))
                    gj = float(_cmi_block(yb, Sc, Bb[j][:, None], seed=seed))
                    ex = gp - gi - gj
                    if ex > chain_gate and (best is None or ex > best[3]):
                        best = (i, j, gp, ex)
                if best is None:
                    break
                i, j, gp, ex = best
                chain_total += ex
                Sc = np.hstack([Sc, _pblock(Bb, Bb[i], Bb[j])])
                pool = [t for t in pool if not (t[0] == i and t[1] == j)]
                picks += 1

            # random bulk stratum with multiplicity-adjusted gate (expected FP mass < one pair)
            from scipy.stats import norm
            total_pairs = K_sp * (K_sp - 1) // 2
            n_rest = max(0, total_pairs - len(cand))
            if n_rest:
                z_bulk = max(gate_z, float(norm.ppf(1.0 - 0.5 / n_rest)))
                bulk_gate = mu_p + z_bulk * sd_p
                n_draw = min(n_rand_pairs, n_rest)
                rnd_sig = []
                for _ in range(n_draw):
                    i, j = (int(x) for x in rng.choice(reps, 2, replace=False))
                    s = syn0(i, j)
                    if s > bulk_gate:
                        rnd_sig.append(max(s, 0.0))                 # raw, gated (S1)
                # a single lucky draw must not extrapolate x n_rest: demand replication (>=2)
                bulk = (float(len(rnd_sig) / n_draw * n_rest * np.mean(rnd_sig))
                        if len(rnd_sig) >= 2 else 0.0)
            pair_seen = float(chain_total + bulk)
            # pair_unseen stays 0 BY DESIGN: synergy with a never-captured partner is the
            # declared blind spot; extrapolating into it is refused (see module docstring)

        b_cap = float(max(h_b - value_b, v_gate.max() if K_sp else 0.0, 1e-6))
        slack = float(b_cap * np.sqrt(2.0 * np.log(1.0 / delta) / max(N_stream, 2)) + b_cap / max(N_stream, 2))
        raw_gap = float(G1 + pair_seen + pair_unseen)
        directions.append({
            "direction": dtag, "H_M": h_b, "value_frozen": value_b,
            "n_head": len(sel), "K_species": K_sp, "N_stream": N_stream,
            "f1": f1, "f2": f2, "chao_unseen": chao_unseen, "unseen_ratio": unseen_ratio,
            "G1": G1, "chain_pairs": picks, "chain_total": float(chain_total), "bulk": float(bulk),
            "pair_seen": pair_seen, "pair_unseen": pair_unseen,
            "gate_single": {"mu": mu_s, "sd": sd_s}, "gate_pair": {"mu": mu_p, "sd": sd_p},
            "n_sig_pairs": n_sig, "slack": slack, "raw_gap": raw_gap,
        })

    lam_eff = 1.0 if lam is None else float(lam)
    for d in directions:
        # kappa corrects the finite-sample ATTENUATION of the achieved reading (the CMI-chain on a
        # probe half systematically under-reads I(M;U_selected)); lam scales the horizon gap. Both
        # are fit on the planted battery's train split and frozen (see fit_calibration).
        d["estimate"] = float(min(kappa * d["value_frozen"] + lam_eff * d["raw_gap"] + d["slack"],
                                  d["H_M"]))
    live = [d for d in directions if not d.get("degenerate")]
    if not live:
        return {"valid": False, "error": "degenerate_target_on_both_halves", "certified": False}
    achieved = float(np.mean([d["value_frozen"] for d in live]))
    estimate = float(max(d["estimate"] for d in live))               # adverse-up end
    return {
        "valid": True, "directions": directions, "achieved_frozen": achieved,
        "horizon_estimate": estimate, "at_H_cap": bool(any(
            abs(d["estimate"] - d["H_M"]) < 1e-9 for d in live)),
        "lam": lam_eff, "kappa": float(kappa), "calibrated": lam is not None, "certified": False,
        "scope": "ESTIMATE of the declared mining process's pool-horizon for the FIXED target, "
                 "capped by H(M); trust credential = out-of-sample planted-battery coverage at "
                 "the frozen lam; declared measured blind spots: unseen-by-unreachability, "
                 "parity depth>=3, pair synergy with a never-captured partner",
    }


# --------------------------------------------------------------------------------------------
# planted worlds — the calibration battery (truth is ANALYTIC)
# --------------------------------------------------------------------------------------------

def planted_world(rng, *, rule: str = "linear3", n_probes: int = 300, flip: float = 0.05,
                  unit_noise: float = 0.05, para_noise: float = 0.02, n_partial: int = 8,
                  n_noise: int = 22, capture: str = "zipf", hidden_mode: str = "natural",
                  n_stream_draws: int = 120, web_distract: bool = False,
                  guarantee_pair: bool = False, z_bias: float = 0.5) -> dict:
    """One synthetic world. TWO truths are returned:

      truth        = POOL truth I(M; U_true_noisy), EXACT by enumeration — what the full mining
                     pool can deliver; THE calibration target for the pool-horizon estimator.
      truth_concept = I(M;R) = H(M) - H_b(flip) — the concept ceiling (unreachable by noisy
                     units; concept_horizon's business, reported for context only).

    Capture design v2 (the v1 boost destroyed the Good-Turing signal): ALL species draw capture
    probability from the same natural law (zipf/geom/uniform, random rank assignment) — so
    valuable species appear across the whole frequency spectrum, including exactly-once
    (singletons ARE the GT signal) and not-yet (the extrapolation target, counted inside the pool
    truth). ``hidden_mode='unreachable'`` pins one true species at ~zero probability (declared
    blind spot). ``guarantee_pair`` lifts both xor components to the mean (the pair machinery
    needs both present; a naturally-hidden partner is the declared xor blind spot, tested via its
    own stratum). Value tail realism: ``n_partial`` degraded copies of true latents (noise
    0.15-0.35 — spanned by the best copies, so they do NOT move the pool truth) + ``n_noise``
    pure-noise distractors (``web_distract`` correlates them in 4 webs to stress the quotient)."""
    kmap = {"linear3": 3, "linear5": 5, "and3": 3, "xor2": 2, "parity3": 3}
    K = kmap[rule]
    Z = (rng.random((K, n_probes)) < z_bias).astype(int)   # z_bias<0.5 -> skewed base rate, low H(M)
    if rule.startswith("linear"):
        R = (Z.sum(0) >= int(np.ceil(K / 2))).astype(int)
    elif rule == "and3":
        R = (Z.sum(0) == K).astype(int)
    else:
        R = Z[0]
        for k in range(1, K):
            R = R ^ Z[k]
    fl = rng.random(n_probes) < flip
    M = np.where(fl, 1 - R, R).astype(float)
    b = float(R.mean())
    pm1 = b * (1 - flip) + (1 - b) * flip
    truth_concept = float(max(0.0, _h_bits(pm1) - _h_bits(flip)))
    truth = pool_truth_bits(rule, flip, unit_noise, z_bias=z_bias)

    sig_species, is_true = [], []
    for k in range(K):
        nz = rng.random(n_probes) < unit_noise
        sig_species.append(np.where(nz, 1 - Z[k], Z[k]))
        is_true.append(True)
    for _ in range(n_partial):                    # degraded copies: value tail, span-neutral
        k = int(rng.integers(K))
        dn = float(rng.uniform(0.15, 0.35))
        nz = rng.random(n_probes) < dn
        sig_species.append(np.where(nz, 1 - Z[k], Z[k]))
        is_true.append(False)
    web = (rng.random((4, n_probes)) < 0.5).astype(int) if web_distract else None
    for di in range(n_noise):
        if web_distract:
            nz = rng.random(n_probes) < 0.1
            sig_species.append(np.where(nz, 1 - web[di % 4], web[di % 4]))
        else:
            sig_species.append((rng.random(n_probes) < 0.5).astype(int))
        is_true.append(False)
    n_sp = len(sig_species)

    if capture == "zipf":
        p = 1.0 / np.arange(1, n_sp + 1) ** 1.3
    elif capture == "geom":
        p = 0.7 ** np.arange(n_sp)
    else:
        p = np.ones(n_sp, float)
    order = list(rng.permutation(n_sp))
    p_assigned = np.empty(n_sp)
    for rank, sp in enumerate(order):
        p_assigned[sp] = p[rank]
    true_ids = [i for i in range(n_sp) if is_true[i]]
    if guarantee_pair:                            # xor in-scope worlds need both partners present:
        for i in true_ids:                        # deterministic top-rank guarantee (a mean-level
            p_assigned[i] = float(p.max())        # boost still lost a partner ~4% of the time)
    if hidden_mode == "unreachable":
        h = int(rng.choice(true_ids))
        p_assigned[h] = float(p.min()) / 20.0
    p_assigned = p_assigned / p_assigned.sum()

    draws = rng.choice(n_sp, size=n_stream_draws, replace=True, p=p_assigned)
    sigs, tags = [], []
    seen = sorted(set(int(d) for d in draws))
    for s_id in seen:
        for _ in range(int((draws == s_id).sum())):
            pz = rng.random(n_probes) < para_noise
            sigs.append(np.where(pz, 1 - sig_species[s_id], sig_species[s_id]).astype(float))
            tags.append("iid")
    hidden_true = [i for i in true_ids if i not in seen]
    # scope ground truth (generator-side, NEVER estimator-side): the pool-horizon's declared
    # scope is criteria reachable within ~10x the stream; a true species whose capture
    # probability is below 1/(10*n_draws) is beyond-horizon BY CONSTRUCTION and belongs to the
    # declared unreachability blind spot regardless of the stratum label it was generated under.
    min_true_p = float(min(p_assigned[i] for i in true_ids))
    reachable = bool(min_true_p >= 1.0 / (10.0 * n_stream_draws))
    return {"sigs": np.array(sigs), "M": M, "tags": tags, "truth": truth,
            "truth_concept": truth_concept, "rule": rule, "n_species_seen": len(seen),
            "hidden_true": len(hidden_true), "H_M": _h_bits(float(M.mean())),
            "min_true_capture_p": min_true_p, "reachable_at_10x": reachable}


_STRESS = {   # adversarial-world-designer additions (implementable set), linear3/zipf base
    "para015": dict(rule="linear3", capture="zipf", para_noise=0.15),
    "stream40": dict(rule="linear3", capture="zipf", n_stream_draws=40),
    "web": dict(rule="linear3", capture="zipf", web_distract=True),
    "and3_lowbase": dict(rule="and3", capture="zipf"),
}


def world_grid(seed: int = 0, n_train_seeds: int = 3, n_test_seeds: int = 2) -> List[dict]:
    """The declared battery, SPLIT into train (kappa/lam fit here) and test (coverage is REPORTED
    here — always out-of-sample). In-scope: linear/and3/xor2 under the natural capture continuum
    (valuable species may be seen often, once, or not yet — the GT regime). Blind spots
    (measured, never averaged into coverage): unreachable-hidden, parity3, xor2 natural
    (hidden-partner risk)."""
    grid = []

    def add(cfg, in_scope, blind=None):
        for s in range(n_train_seeds + n_test_seeds):
            grid.append(dict(cfg, in_scope=in_scope, blind_spot=blind,
                             split="train" if s < n_train_seeds else "test",
                             seed=hash((tuple(sorted(cfg.items())), s, seed)) % 2 ** 31))

    for rule in ("linear3", "linear5"):
        for capture in ("zipf", "geom", "uniform"):
            add(dict(rule=rule, capture=capture), in_scope=True)
    for capture in ("zipf", "geom", "uniform"):
        add(dict(rule="xor2", capture=capture, guarantee_pair=True), in_scope=True)
    for name, cfg in _STRESS.items():
        add(dict(cfg), in_scope=True)
    # skewed-base-rate strata (the real-data regime: H(M) well below 1 bit) — pilot v1 found all
    # four real estimates saturating at the H cap because the battery was near-balanced-only
    for zb in (0.2, 0.3):
        for capture in ("zipf", "uniform"):
            add(dict(rule="linear3", capture=capture, z_bias=zb), in_scope=True)
        add(dict(rule="linear5", capture="zipf", z_bias=zb), in_scope=True)
    for rule, mode in (("linear3", "unreachable"), ("linear5", "unreachable"),
                       ("parity3", "natural"), ("xor2", "natural")):
        for capture in ("zipf", "uniform"):
            add(dict(rule=rule, capture=capture, hidden_mode=mode),
                in_scope=False, blind=f"{rule}:{mode}")
    return grid


_WORLD_KEYS = ("rule", "capture", "hidden_mode", "para_noise", "n_stream_draws",
               "web_distract", "guarantee_pair", "z_bias")


def _battery_one(g: dict, top_k: int, cr_kw: dict) -> dict:
    rng = np.random.default_rng(g["seed"])
    w = planted_world(rng, **{k: g[k] for k in _WORLD_KEYS if k in g})
    try:
        cert = cr2_certificate(w["sigs"], w["M"], w["tags"], top_k=top_k,
                               seed=g["seed"] % 997, **cr_kw)
        if not cert.get("valid"):
            return {**g, "error": cert.get("error", "invalid")}
    except Exception as e:
        return {**g, "error": f"{type(e).__name__}: {e}"}
    live = [d for d in cert["directions"] if not d.get("degenerate")]
    d_adv = max(live, key=lambda d: d["value_frozen"] + d["raw_gap"] + d["slack"])
    out = {**g, "truth": w["truth"], "H_M": w["H_M"], "hidden_true": w["hidden_true"],
           "n_species_seen": w["n_species_seen"], "value_frozen": d_adv["value_frozen"],
           "raw_gap": d_adv["raw_gap"], "slack": d_adv["slack"],
           "achieved_mean": cert["achieved_frozen"],
           "min_true_capture_p": w["min_true_capture_p"],
           "reachable_at_10x": w["reachable_at_10x"]}
    if out["in_scope"] and not w["reachable_at_10x"]:
        # generator-side reclassification: the world violated its stratum's declaration
        out["in_scope"] = False
        out["blind_spot"] = "beyond-horizon"
    return out


def run_battery(grid: List[dict], *, top_k: int = 6, n_procs: int = 1, verbose: bool = True,
                **cr_kw) -> List[dict]:
    """Run cr2 (raw components stored) on every world; lam is fit POST-HOC from train-split
    components and evaluated on the test split (the estimate is piecewise-linear in lam, so one
    battery pass supports the whole lam grid)."""
    if n_procs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(n_procs) as pool:
            rows = pool.starmap(_battery_one, [(g, top_k, cr_kw) for g in grid])
    else:
        rows = []
        for i, g in enumerate(grid):
            rows.append(_battery_one(g, top_k, cr_kw))
            if verbose and (i + 1) % 20 == 0:
                print(f"  battery {i + 1}/{len(grid)}", flush=True)
    return rows


def _est(r: dict, lam: float, kappa: float = 1.0) -> float:
    return float(min(kappa * r["value_frozen"] + lam * r["raw_gap"] + r["slack"], r["H_M"]))


def coverage_at(rows: List[dict], lam: float, kappa: float = 1.0, *,
                split: Optional[str] = None) -> dict:
    """Coverage, tightness and vacuity of estimate(kappa, lam) vs the POOL truth, split by scope
    stratum (and optionally by train/test split)."""
    good = [r for r in rows if "error" not in r and (split is None or r.get("split") == split)]

    def block(rs):
        if not rs:
            return None
        est = np.array([_est(r, lam, kappa) for r in rs])
        tru = np.array([r["truth"] for r in rs])
        hm = np.array([r["H_M"] for r in rs])
        return {"coverage": float(np.mean(est >= tru - 1e-9)),
                "tightness_mean": float(np.mean(est - tru)),
                "vacuity_rate": float(np.mean(np.abs(est - hm) < 1e-9)), "n": len(rs)}

    out = {"in_scope": block([r for r in good if r["in_scope"]])}
    for bs in sorted({r.get("blind_spot") for r in good if not r["in_scope"]} - {None}):
        out[f"blind:{bs}"] = block([r for r in good if r.get("blind_spot") == bs])
    return out


def fit_calibration(rows: List[dict], *, target: float = 0.95,
                    kappa_grid: Sequence[float] = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0),
                    lam_grid: Sequence[float] = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)) -> dict:
    """Two-parameter calibration, both frozen before real data: ``kappa`` corrects the achieved
    reading's finite-sample attenuation, ``lam`` scales the horizon gap. Among all (kappa, lam)
    pairs reaching >= target coverage on the TRAIN split, the pair with the SMALLEST mean train
    tightness wins (coverage alone rewards vacuity). The quoted credential is the TEST-split
    coverage at the frozen pair — always out-of-sample — plus tightness and vacuity."""
    qual = []
    for k in kappa_grid:
        for l in lam_grid:
            blk = coverage_at(rows, float(l), float(k), split="train").get("in_scope")
            if blk and blk["coverage"] >= target:
                qual.append((blk["tightness_mean"], float(k), float(l)))
    if not qual:
        best = max(((coverage_at(rows, float(l), float(k), split="train")["in_scope"]["coverage"],
                     -float(k) - float(l), float(k), float(l))
                    for k in kappa_grid for l in lam_grid), default=None)
        return {"kappa": None, "lam": None, "target": target,
                "best_train_coverage": best[0] if best else None,
                "at_best": {"kappa": best[2], "lam": best[3]} if best else None,
                "train": coverage_at(rows, best[3], best[2], split="train") if best else None,
                "test": coverage_at(rows, best[3], best[2], split="test") if best else None,
                "n_worlds": len([r for r in rows if "error" not in r]),
                "note": "NO (kappa,lam) reached target train coverage — estimator design must "
                        "iterate; numbers above are at the best-coverage pair for diagnosis"}
    _, kappa, lam = min(qual)
    return {"kappa": kappa, "lam": lam, "target": target,
            "train": coverage_at(rows, lam, kappa, split="train"),
            "test": coverage_at(rows, lam, kappa, split="test"),
            "n_qualifying_pairs": len(qual),
            "n_worlds": len([r for r in rows if "error" not in r]),
            "note": "(kappa,lam) fit on train worlds (tightness-optimal among covering pairs); "
                    "credential = TEST coverage (out-of-sample); blind-spot strata reported, "
                    "never averaged into coverage"}


def fit_calibration_binned(rows: List[dict], *, h_split: float = 0.85, target: float = 0.95,
                           **kw) -> dict:
    """Per-H-regime calibration: a single global (kappa, lambda) tuned on near-balanced worlds
    (H ~ 1) over-corrects skewed-base-rate metrics (small H) into the vacuous H cap — pilot v1
    finding on real data (all four estimates saturated; real H .27-.81). Fit each regime on its
    own train worlds; apply to a real metric by its H bin. Both fits stay train/test split."""
    lo = [r for r in rows if "error" not in r and r["H_M"] < h_split]
    hi = [r for r in rows if "error" not in r and r["H_M"] >= h_split]
    return {"h_split": h_split,
            "lowH": fit_calibration(lo, target=target, **kw) if lo else None,
            "highH": fit_calibration(hi, target=target, **kw) if hi else None,
            "n_lowH": len(lo), "n_highH": len(hi)}


# backwards-compat shim for tests written against the single-parameter fit
def fit_lambda(rows: List[dict], *, target: float = 0.95,
               lam_grid: Sequence[float] = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0)) -> dict:
    out = fit_calibration(rows, target=target, kappa_grid=(1.0,), lam_grid=lam_grid)
    return out


# --------------------------------------------------------------------------------------------
# no-violation audit + concept horizon
# --------------------------------------------------------------------------------------------

def audit_no_violation(cert: dict, observed: Dict[str, float], *, tol: float = 1e-6) -> dict:
    """Field test of the estimate: every OBSERVED achieved value (single prompts, GEPA rounds,
    prose attacks, other-half heads) must sit at or below the horizon estimate. A violation is a
    falsification event — reported loudly, never clipped away."""
    est = cert["horizon_estimate"]
    viol = {k: float(v) for k, v in observed.items() if np.isfinite(v) and v > est + tol}
    return {"estimate": est, "n_observed": len(observed), "violations": viol,
            "ok": not viol,
            "max_observed": float(max((v for v in observed.values() if np.isfinite(v)),
                                      default=float("nan")))}


def concept_horizon(t_values: Sequence[float], *, tail_frac: float = 0.4, seed: int = 0) -> dict:
    """INSTANTIATION-horizon ESTIMATE (never a bound): given T (self-consistency information cap)
    across many re-instantiations of the same concept, estimate how much more self-consistent it
    could be — the answer to "how much better could the metric itself be?". EVT endpoint over the
    T tail; estimate-grade with the same EVT assumptions as prompt_space_bracket.evt_endpoint;
    needs >= ~40 behaviorally distinct instantiations to say anything."""
    from methods.metric_implementer.experiments.prompt_space_bracket import evt_endpoint
    t = np.asarray([x for x in t_values if np.isfinite(x)], float)
    out = {"n_instantiations": int(t.size),
           "t_max_observed": float(t.max()) if t.size else None,
           "t_median": float(np.median(t)) if t.size else None,
           "certified": False}
    if t.size >= 40:
        ev = evt_endpoint(t, tail_frac=tail_frac, seed=seed)
        out["evt"] = ev
        if ev.get("valid") and ev.get("endpoint") is not None:
            out["headroom_estimate"] = float(max(0.0, ev["endpoint"] - t.max()))
    else:
        out["error"] = "too_few_instantiations"
    return out
