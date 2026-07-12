"""OmegaCertificate — the unified entry point for the within-class optimality certificate (§6.7a).

Auto-detects the criterion-class size K and dispatches to the right technique:
  * K ≤ --large-k (default 15) → EXACT brute force: score all 2^K−1 subsets, emit the exact OPT_Ω
    certificate, and run U₂/double-greedy ALONGSIDE as a cross-check (how well the large-Ω
    approximation matches exact on THIS real, non-monotone R — the open question the theory leaves).
  * K > --large-k → LARGE-Ω FALLBACK: score only the subsets greedy/double-greedy touch (lazy
    memoizer), run U₂ on a MEASURED γ, no full-lattice enumeration (2^K infeasible). Graceful
    degradation, reported as a measured fact (note §6.1), not a verified ratio.

This is the operational realization of the theory note's "within-class certificate" (Rung 2 / §6.7a):
certify `OPT_Ω = max_{S⊆Ω} R(C(S))` — how well a criterion subset re-expresses the full standard.

The target M is BEHAVIORAL: the full criterion set's verdict C(Ω) applied to each item (NOT a
strong-LLM "holistic quality" anchor — the no-anchor pivot). So the certificate is reconstruction of
the full standard, not self-recovery of one model's taste.

Engines: small_omega_brute_force.exact_certificate (exact) and large_omega.U2_bound/double_greedy (fallback).

GPU run (sk3):  pick a free GPU; HOME=/lfs; VLLM_GPU_MEM_UTIL modest.
  $PY -m methods.metric_implementer.experiments.omega_certificate \
    --rubric-file <Ω> --pool <pool> --text-col code --task code-review \
    --model meta-llama/Llama-3.1-8B-Instruct --n-items 70 --max-chars 3000 --out <npz>
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .. import config as cfgmod
from .. import vinfo
from .small_omega_brute_force import _criteria_from_rubric, _compile, exact_certificate
from .real_gamma import _YESNO, _signal, _median_split


# --------------------------------------------------------------------------------------------
# the lazy-memoizing scorers that warm only touched subsets (large-Ω fallback plumbing)
# --------------------------------------------------------------------------------------------

def _greedy_touch(f, K, budget, score_one):
    """Forward greedy that warms `score_one(S)` for every subset it evaluates (large-Ω fallback:
    only touched subsets hit the GPU). Returns (S_g, f(S_g), per-step gains)."""
    S, gains, fS = [], [], 0.0
    for _ in range(budget):
        best_e, best_g = None, 0.0
        for e in range(K):
            if e in S:
                continue
            cand = tuple(sorted(S + [e]))
            score_one(cand)
            g = f(cand) - fS
            if g > best_g:
                best_e, best_g = e, g
        if best_e is None or best_g <= 1e-12:
            break
        S.append(best_e); gains.append(best_g)
        cand = tuple(sorted(S)); score_one(cand); fS = f(cand)
    return tuple(sorted(S)), fS, gains


def _dg_touch(f, K, score_one, seed=0):
    """double-greedy that warms `score_one` for every subset it evaluates (large-Ω fallback)."""
    rng = np.random.default_rng(seed)
    A, B = [], set(range(K))
    for i in range(K):
        if i not in B:
            continue
        for cand in (tuple(sorted(A)), tuple(sorted(A + [i])),
                     tuple(sorted(B)), tuple(sorted(x for x in B if x != i))):
            score_one(cand)
        fA = f(tuple(sorted(A)))
        dA = f(tuple(sorted(A + [i]))) - fA
        fB = f(tuple(sorted(B)))
        dB = fB - f(tuple(sorted(x for x in B if x != i)))
        dA_c, dB_c = max(dA, 0.0), max(dB, 0.0)
        if dA_c + dB_c == 0:
            keep = rng.random() < 0.5
        else:
            keep = rng.random() < (dA_c / (dA_c + dB_c))
        if keep:
            A.append(i)
        B.discard(i)
    return tuple(sorted(A)), f(tuple(sorted(A)))


# --------------------------------------------------------------------------------------------
# the certificate dispatcher
# --------------------------------------------------------------------------------------------

@dataclass
class OmegaCertificate:
    """Certify OPT_Ω = max_{S⊆Ω} R(C(S)), auto-dispatching exact vs. large-Ω on K.

    Construct directly (OmegaCertificate(rubric_file=..., pool=..., ...)) or via
    `from_argv`. `.run()` scores the (subsets × items) recovery matrix against the behavioral full-
    rubric target M, dispatches on K vs --large-k, prints the certificate, saves the npz, and returns
    a summary dict. The class is the single entry point; small_omega_brute_force and large_omega are its engines.
    """
    rubric_file: str
    pool: str
    text_col: str = "text"
    task: str = "code-review"
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    n_items: int = 70
    max_chars: int = 3000
    budget: int = 5
    chunk: int = 120
    out: str = "/tmp/omega_certificate.npz"
    fake: bool = False
    large_k: int = 15            # K threshold: ≤ this = exact brute force; > this = large-Ω fallback
    prose_prompt: Optional[str] = None   # the ORIGINAL prose prompt p̂ (if set, also measures I(M, M_ω))
    compiler: str = "conjunction"   # the C(Ω) framing (small_omega_brute_force._compile): conjunction
    # (default, the historical baseline) | weighted_sum (p̂'s additive aggregation) | prose_join
    # (holistic). The decomposition gap compares p̂ to C(Ω); an unfaithful compiler depresses I(M,M_ω)
    # for purely formal reasons, so this is a real axis (theory §5/§7). Sweep it via compiler_sweep.py.
    holdout_frac: float = 0.5       # Q1: fraction of items HELD OUT for subset-selection generalization.
    # The in-sample certificate (greedy/OPT/γ) is a combinatorial statement on all N items; this re-splits
    # the already-scored MHAT/M (no new GPU), picks S* on train, and reports held-out R plus the
    # fixed-target DPI legs. This is an out-of-sample point evaluation of the chosen subset; it is not
    # a population upper confidence bound on the best subset.
    # Exact mode only (fallback path doesn't materialize MHAT); set 0.0 to disable.
    holdout_seed: int = 0

    # populated by run()
    result: dict = field(default_factory=dict, repr=False)

    @staticmethod
    def from_argv(argv, *, force_large_k=None):
        """argparse adapter → a plain dict suitable for the OmegaCertificate constructor. The
        `force_large_k` hook lets the small_omega_brute_force deprecation shim force exact mode."""
        ap = argparse.ArgumentParser(prog="omega_certificate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument("--rubric-file", required=True); ap.add_argument("--pool", required=True)
        ap.add_argument("--text-col", default="text"); ap.add_argument("--task", default="code-review")
        ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
        ap.add_argument("--n-items", type=int, default=70); ap.add_argument("--max-chars", type=int, default=3000)
        ap.add_argument("--budget", type=int, default=5); ap.add_argument("--chunk", type=int, default=120)
        ap.add_argument("--out", default="/tmp/omega_certificate.npz"); ap.add_argument("--fake", action="store_true")
        ap.add_argument("--large-k", type=int, default=15,
                        help="K threshold: ≤ this = exact brute force + approximation cross-check; "
                             "> this = large-Ω fallback (U2 + double-greedy on a measured γ, no enumeration)")
        ap.add_argument("--prose-prompt", default=None,
                        help="the ORIGINAL prose prompt p̂. If set, ALSO measures I(M, M_ω): does the "
                             "prose prompt's verdict match the all-criteria verdict? (decomposition gap)")
        ap.add_argument("--compiler", default="conjunction",
                        choices=["conjunction", "weighted_sum", "prose_join", "echo_prose"],
                        help="C(Ω) framing (form axis); the decomposition gap compares p̂ to C(Ω).")
        ap.add_argument("--holdout-frac", type=float, default=0.5,
                        help="Q1: fraction of items held out for subset-selection generalization "
                             "(exact mode only; 0.0 disables).")
        ap.add_argument("--holdout-seed", type=int, default=0)
        a = ap.parse_args(argv)
        d = {k: getattr(a, k) for k in ("rubric_file", "pool", "text_col", "task", "model",
            "n_items", "max_chars", "budget", "chunk", "out", "fake")}
        d["large_k"] = force_large_k if force_large_k is not None else a.large_k
        d["prose_prompt"] = a.prose_prompt
        d["compiler"] = a.compiler
        d["holdout_frac"] = a.holdout_frac
        d["holdout_seed"] = a.holdout_seed
        return d

    def run(self) -> dict:
        from ..vllm_backend import make_judge_backend
        cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), self.task)
        cfg.vllm_fake = self.fake
        backend = make_judge_backend(self.model, cfg)

        # ---- load corpus + Ω (criterion set from the rubric file) ----
        if self.pool.endswith(".parquet"):
            df = pd.read_parquet(self.pool)
        elif self.pool.endswith(".csv") or self.pool.endswith(".csv.gz"):
            df = pd.read_csv(self.pool)
        else:
            df = pd.read_json(self.pool, lines=True,
                              compression="gzip" if self.pool.endswith(".gz") else None)
        df = df[df[self.text_col].astype(str).str.len() > 50]
        texts = df[self.text_col].astype(str).tolist()[: self.n_items]
        crits = _criteria_from_rubric(self.rubric_file)
        K = len(crits)
        exact = K <= self.large_k
        mode = "EXACT (brute-force all subsets)" if exact else \
               f"FALLBACK (large Ω: K={K} > --large-k={self.large_k})"
        print(f"loaded {len(texts)} items; Ω = {K} criteria; mode = {mode}"
              + (f"; lattice = {2**K - 1} non-empty subsets" if K < 24
                 else "; lattice too large to enumerate"))

        # ---- behavioral target M = the FULL criterion set's verdict C(Ω) (no holistic anchor) ----
        full_rubric = _compile(crits, tuple(range(K)), compiler=self.compiler)
        M_cont = _signal(backend, full_rubric, texts, self.max_chars)
        M = _median_split(M_cont)
        N = len(M)

        # ---- I(M, M_ω): does the ORIGINAL prose prompt p̂'s verdict match the all-criteria verdict?
        # The decomposition gap (prose → atomic criteria). Expected LARGE if decomposition is lossless.
        # I(M_ω, M_s) is the selection gap below (the subset certificate). Each carries its own T.
        #
        # ESTIMATOR + DEGENERACY NOTES (2026-06-23, post-diagnosis on the CW run; see
        #   notes/2026-06-23__omega-certificate-decomp-gap-diagnosis.md):
        #  (0) SCALE (fixed earlier): I(M, M_ω) uses `tvd_recovery` (per-item-MAD TVD, raw scale [0,½]),
        #      NOT `tvd_mi` (a 2×2-contingency TVD cap-normalized to [0,1] that was incomparable to T and
        #      spuriously 2× larger — vinfo.py:30 forbids mixing them across R≤T).
        #  (1) WHICH DPI LEG IS VALID. tvd_recovery(M_cont, P) keeps M_cont SOFT and uses P only as a
        #      BINARY grouping label. The termwise-Jensen-guaranteed DPI leg is therefore R ≤ T(recovered
        #      side) = T_omega = tvd_transmission(M_cont) — exactly the design tvd_guardrail uses. The
        #      label-side leg R ≤ T_prose is NOT a valid check here: R is built on the BINARIZED P while
        #      T_prose is the SOFT P_cont transmission — different representations of the prose channel, so
        #      the soft T_prose does NOT bound a binarized-label R. (The earlier min(T_prose,T_omega) check
        #      fired a SPURIOUS "violation" on exactly this mismatch.) T_prose is kept as a DIAGNOSTIC only.
        #  (2) DEGENERACY. If a channel barely varies across items (soft transmission below the noise
        #      floor), its median-split is an ~arbitrary balanced label and the decomposition number is
        #      untrustworthy — a near-constant channel MANUFACTURES a spurious 50/50 split. We FLAG this
        #      (prose_collapsed / omega_collapsed) rather than report a clean R. The CW run hit exactly
        #      this: the GEPA prose prompt scored ~0.97 for every item (T_prose=0.013, at the noise floor).
        decomp = {}
        if self.prose_prompt:
            P_cont = _signal(backend, self.prose_prompt, texts, self.max_chars)
            P = _median_split(P_cont)
            I_MMw = vinfo.tvd_recovery(M_cont, P, n_boot=200, n_perm=32, seed=0)["tvd_recovery"]
            T_prose = vinfo.tvd_transmission(P_cont, debias=False)["tvd_t"]  # prose soft transmission (DIAGNOSTIC)
            T_omega = vinfo.tvd_transmission(M_cont, debias=False)["tvd_t"]  # recovered-side ceiling (DPI-valid leg)
            dpi_ok = (I_MMw <= T_omega + 1e-6)                              # the termwise-guaranteed leg
            T_FLOOR = 0.02                                                  # ~per-item-MAD noise floor
            prose_collapsed = bool(T_prose < T_FLOOR)
            omega_collapsed = bool(T_omega < T_FLOOR)
            trustworthy = bool(dpi_ok and not prose_collapsed and not omega_collapsed)
            decomp = {"I_M_Momega": float(I_MMw), "T_prose": float(T_prose), "T_omega": float(T_omega),
                      "dpi_ok": bool(dpi_ok), "prose_collapsed": prose_collapsed,
                      "omega_collapsed": omega_collapsed, "trustworthy": trustworthy,
                      "P_base_rate": float(np.mean(P)), "M_omega_base_rate": float(np.mean(M))}
            reasons = (([] if dpi_ok else ["R>T_ω (true DPI breach)"])
                       + (["prose near-constant (T_prose<%.2f → median-split ~arbitrary)" % T_FLOOR]
                          if prose_collapsed else [])
                       + (["ω near-constant"] if omega_collapsed else []))
            tag = "trustworthy ✓" if trustworthy else "⚠ UNTRUSTWORTHY: " + "; ".join(reasons)
            print(f"[I(M, M_ω)] decomposition gap: I(M, M_ω)={I_MMw:.3f}  "
                  f"(T_ω(ceiling)={T_omega:.3f} [R≤T_ω: {dpi_ok}]; T_prose(diag)={T_prose:.3f}; "
                  f"P̄={np.mean(P):.2f}, M_ω̄={np.mean(M):.2f})  → {tag}")

        def score_subsets(subset_list):
            mh = np.full((len(subset_list), N), np.nan)
            rd = {}
            for c0 in range(0, len(subset_list), self.chunk):
                chunk = subset_list[c0:c0 + self.chunk]
                prompts = [_YESNO.format(rubric=_compile(crits, S, compiler=self.compiler), text=t[: self.max_chars])
                           for S in chunk for t in texts]
                pyes = np.array(backend.score_binary(prompts, pos="YES", neg="NO"), dtype=float)
                for j, S in enumerate(chunk):
                    mhat = pyes[j * N:(j + 1) * N]
                    mh[c0 + j] = mhat
                    rd[tuple(sorted(S))] = vinfo.tvd_recovery(mhat, M)["tvd_recovery"]
                print(f"  scored {min(c0 + self.chunk, len(subset_list))}/{len(subset_list)} subsets")
            return mh, rd

        if exact:
            subsets = [S for r in range(1, K + 1) for S in itertools.combinations(range(K), r)]
            MHAT, R = score_subsets(subsets)
        else:
            scored: dict = {}

            def _score_one(S):
                S = tuple(sorted(S))
                if S in scored:
                    return
                _, _rd = score_subsets([S])
                scored.update(_rd)
            for i in range(K):           # seed singletons (solo R + γ-measurement) + the full set
                _score_one((i,))
            _score_one(tuple(range(K)))
            from .large_omega import f_from_dict
            f_probe = f_from_dict(scored)
            _greedy_touch(f_probe, K, self.budget, _score_one)   # warm greedy's touched subsets
            _dg_touch(f_probe, K, _score_one, seed=0)            # warm double-greedy's touched subsets
            subsets = sorted(set(scored.keys()))
            MHAT = np.full((len(subsets), N), np.nan)
            R = {S: scored[S] for S in subsets}

        save_kw = dict(M=M, M_cont=M_cont, mhat=MHAT, crits=np.array(crits, dtype=object),
                 subset_order=np.array([",".join(map(str, sorted(S))) for S in subsets], dtype=object),
                 subset_keys=np.array([",".join(map(str, k)) for k in R], dtype=object),
                 subset_R=np.array(list(R.values())), model=self.model, n_items=N, mode=mode, K=K)
        # persist the prose-prompt verdict vectors so the run is recalibrable without re-scoring
        # (2026-06-23): previously only M_cont was saved, so I(M,M_ω) couldn't be re-derived.
        if self.prose_prompt and "P_cont" in dir():
            save_kw.update(P_cont=P_cont, P_bin=P, prose_prompt=np.array(self.prose_prompt))
        np.savez(self.out, **save_kw)

        # =================== the certificate (dispatched on `exact`) ===================
        if exact:
            cert = exact_certificate(R, subsets, K, self.budget, M, self.task)
        else:
            from .large_omega import U2_bound, double_greedy, gamma_measured
            from .small_omega_brute_force import greedy_on_R

            # lazily-scoring f over the `scored` memoizer (in run()'s closure so it can call _score_one)
            def fR(S):
                S = tuple(sorted(S))
                if S not in scored:
                    _score_one(S)
                return float(scored[S])
            Sg, fg = greedy_on_R(scored, K, self.budget)
            gmin, gp10 = gamma_measured(fR, K, n_samples=400, seed=0)
            prune_detected = (gmin == 0.0)
            dg = double_greedy(fR, K, randomized=True, seed=0)
            print(f"\n===== {self.task}: LARGE-Ω FALLBACK certificate "
                  f"(K={K} > {self.large_k}, no enumeration) =====")
            print(f"subsets scored (touched only): {len(scored)}  (vs 2^{K}−1 = {2**K-1:,} if enumerated)")
            print(f"γ measured on raw R (min over sampled (S,Ω') pairs) = {gmin:.3f}  (p10={gp10:.3f})")
            print(f"greedy(k)  picks {Sg}  R={fg:.3f}")
            print(f"double-greedy (½-approx, randomized): picks {dg['S']} R={dg['f_S']:.3f}")
            if not prune_detected:                     # γ>0 ⇒ U2 well-defined (if measured-γ)
                gamma_use = gmin if np.isfinite(gmin) else 0.5
                u2 = U2_bound(fR, K, gamma=gamma_use, budget=self.budget)
                print(f"U2 diagnostic (sampled γ={gamma_use:.3f}; NOT certified): {u2['U2']:.3f}  "
                      f"= greedy {fg:.3f} + (1/γ)·Σ_top-k δ")
                print(f"  → estimated U2 − greedy = {u2['U2']-fg:+.3f}; greedy ≥ "
                      f"{fg/u2['U2']:.3f}·OPT_Ω under the measured γ (an ESTIMATE, not a certificate)")
            else:
                # PRUNE ⇒ raw R non-monotone ⇒ γ clamped to 0 ⇒ U2 (needs monotonicity / γ>0) INVALID
                # on raw R; the honest R↑ can't be built exactly at large K → certificate degrades.
                print(f"U2: WITHHELD — γ clamped to 0 (PRUNE: raw R non-monotone). U2 is only valid on "
                      f"the monotone envelope R↑, which at K={K} (> {self.large_k}) cannot be built "
                      f"exactly. The per-instance bound is unrecoverable here; only double-greedy remains.")
            print(f"*** LARGE-Ω certificate: greedy/double-greedy on the TOUCHED subsets "
                  f"{'with a measured-γ U2' if not prune_detected else '(U2 withheld: non-monotone R)'}. "
                  f"Graceful degradation — NOT exact; reported as a measured fact (note §6.1), not "
                  f"a verified ratio, because real R is non-monotone (PRUNE) + weakly-submodular. ***")
            cert = {"greedy_S": Sg, "greedy_R": fg, "gamma_measured": gmin, "gamma_p10": gp10,
                    "prune_detected": bool(prune_detected), "dg_S": dg["S"], "dg_R": dg["f_S"]}
        # ---- Q1: held-out SUBSET-SELECTION generalization (free re-split of scored arrays) ----
        # The certificate above is in-sample (a combinatorial statement on all N items). This certifies
        # how the CHOSEN subset generalizes: pick S* on a train split, re-score on held-out, report
        # R/T_test/A via the DPI legs (tvd_recovery/tvd_transmission). Pure numpy over the already-scored
        # MHAT/M — NO new GPU. Exact mode only (the fallback path never materializes MHAT); set
        # --holdout-frac 0 to disable. Prototype (Aigner-Ziegler): gen_gap +0.013, DPI 8/8.
        heldout = {}
        if (exact and self.holdout_frac and 0.0 < self.holdout_frac < 1.0 and N >= 8
                and np.isfinite(MHAT).all()):
            rng = np.random.default_rng(self.holdout_seed)
            perm = rng.permutation(N)
            n_tr = max(4, int(round(N * (1.0 - self.holdout_frac))))
            tr, te = perm[:n_tr], perm[n_tr:]
            if len(te) >= 4:
                Rtr = np.array([vinfo.tvd_recovery(MHAT[j, tr], M[tr])["tvd_recovery"]
                                for j in range(len(subsets))])
                Rte = np.array([vinfo.tvd_recovery(MHAT[j, te], M[te])["tvd_recovery"]
                                for j in range(len(subsets))])
                j_star = int(np.argmax(Rtr))
                r_tr, r_te = float(Rtr[j_star]), float(Rte[j_star])
                dpi = vinfo.fixed_target_channel_certificate(M[te].astype(float), MHAT[j_star, te])
                T_te = float(dpi["tvd"]["T_target"])
                T_candidate_te = float(dpi["tvd"]["T_candidate"])
                in_opt = float(max(R.values()))
                heldout = {"S_star_train": list(subsets[j_star]), "R_train_star": r_tr,
                           "R_test_star": r_te, "T_test": T_te,
                           "T_target_test": T_te, "T_candidate_test": T_candidate_te,
                           "A_test": T_te - r_te,
                           "dpi_ok": bool(dpi["tvd"]["dpi_ok"]), "in_sample_OPT": in_opt,
                           "gen_gap": in_opt - r_te, "n_train": int(len(tr)), "n_test": int(len(te))}
                print(f"[Q1 held-out] S*(train)={list(subsets[j_star])} | R_train={r_tr:.3f} "
                      f"R_test={r_te:.3f} T_test={T_te:.3f} A={T_te - r_te:.3f} "
                      f"DPI_ok={heldout['dpi_ok']} | in-sample OPT={in_opt:.3f} "
                      f"gen_gap={in_opt - r_te:+.3f}")

        self.result = {"mode": mode, "K": K, "n_items": N, "M_base_rate": float(np.mean(M)),
                       "subsets_scored": len(R), "certificate": cert, "decomposition": decomp,
                       "heldout": heldout}
        return self.result


def main(argv=None):
    import sys
    args = OmegaCertificate.from_argv(argv if argv is not None else sys.argv[1:])
    OmegaCertificate(**args).run()


if __name__ == "__main__":
    main()
