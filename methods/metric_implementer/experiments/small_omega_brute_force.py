"""§6.7a — the EXACT within-class certificate ENGINE (brute-force the criterion-subset lattice).

This module is the exact-mode engine of `omega_certificate.OmegaCertificate`, which auto-detects K
and dispatches: K ≤ --large-k → here (exact brute force + the U₂/double-greedy cross-check); K >
--large-k → large_omega (the large-Ω fallback). Do NOT call this directly for new work — invoke
`OmegaCertificate`, which picks the technique. (The `main` here is a deprecation shim that forwards.)

For every non-empty subset S ⊆ Ω, compile the rubric C(S) (only the criteria in S) and score the REAL
recovery  R(C(S)) = I_TVD(M; M̂_S)  where M̂_S = C(S)'s verdict and M = the FULL criterion set's verdict
(the canonical rubric C(Ω) applied to each item — a BEHAVIORAL target, NOT a strong-LLM "holistic
quality" anchor; no-anchor pivot, project_recovery_reconstruction_no_anchor). This makes the §6.7a
certificate a genuine RECONSTRUCTION certificate (how well does subset S re-express the full standard),
not self-recovery of one model's taste. With all
2^K−1 subset values in hand we report, with NO approximation:
  * exact OPT_Ω at each budget k, and greedy/OPT_Ω (the realized gap) — the certificate;
  * a TRUSTWORTHY exact γ on the real R (each R(C(S)) is a clean 2×2 MI, well-sampled — unlike the
    cheap Ǐ(S)=I(M;X_S) whose joint MI was under-sampled);
  * the non-monotonicity directly: if argmax R is a STRICT subset, PRUNE is real (adding criteria hurts).

This is §6.7a: Rung 2 collapses into an exact best-in-class statement, and it CAPTURES the executor
bottleneck (M̂_S is the compiled rubric's single compressed verdict, not the ideal signal vector).

GPU run (sk3) — via the entry point:  pick a free GPU; HOME=/lfs; VLLM_GPU_MEM_UTIL modest.
  $PY -m methods.metric_implementer.experiments.omega_certificate \
    --rubric-file /tmp/code_quality_rubric.txt --pool <pool> --text-col code --task code-review \
    --model meta-llama/Llama-3.1-8B-Instruct --n-items 70 --max-chars 3000 --out <npz>
"""
from __future__ import annotations

import itertools

import numpy as np

from .. import vinfo


def _criteria_from_rubric(path):
    """Ω = the BULLET lines of a pre-bulleted rubric file (drops the header / non-bullet intro)."""
    crits = []
    for ln in open(path).read().splitlines():
        s = ln.strip()
        bulleted = s[:1] in "-*•" or (len(s) > 1 and s[0].isdigit() and s[1] in ").")
        if bulleted:
            c = s.lstrip("-*•0123456789.) ").strip()
            if len(c) > 12:
                crits.append(c)
    return crits


def monotone_envelope(R, K):
    """R↑(S) = max_{T⊆S} R(T) — the free-disposal monotonization (DP). Diagnostic only: optimizing
    R↑ ≠ optimizing R when the judge can't costlessly ignore a listed criterion (the PRUNE effect)."""
    Rup = {(): 0.0}
    for r in range(1, K + 1):
        for S in itertools.combinations(range(K), r):
            best = R[S]
            for e in S:
                best = max(best, Rup[tuple(x for x in S if x != e)])
            Rup[S] = best
    return Rup


def _compile(crits, S, *, compiler="conjunction", prose=None):
    """Canonical compiler C(S): the criteria in S as ONE prompt the executor scores. ``compiler``
    picks the FRAMING (theory §5/§7 form axis — a fixed template so f(S)=R(C(S)) is a genuine set
    function, killing path-dependence). The decomposition gap I(M, M_ω) compares the prose prompt
    p̂'s verdict to C(Ω)'s; an UNFAITHFUL compiler (e.g. a flat conjunction against p̂'s weighted
    sum) can depress I for purely formal reasons, so the framing is a real axis to sweep:

      * ``conjunction`` (default) — flat bullet list, "meets this standard". Equal-weight
        conjunctive read; the historical baseline and the loosest match to a weighted-sum p̂.
      * ``weighted_sum`` — p̂'s OWN additive aggregation: "score = fraction of checks satisfied".
        Mimics a mechanized point-sum rubric (the GEPA leniency-attractor form) at equal weights.
      * ``prose_join`` — holistic prose: "a strong item is one that …; weigh together". Closest to
        natural-language holistic judgment; isolates the prose-vs-checklist form effect.
      * ``echo_prose`` (needs ``prose``) — returns p̂ verbatim; the faithful-decomposition CEILING
        (M_ω := the prose verdict itself), so I(M,M_ω) → T_prose. A reference, not a real compiler.
    """
    subs = [crits[i] for i in S]
    if compiler == "weighted_sum":
        n = max(len(subs), 1)
        return ("Score the item in [0,1] as the fraction of these checks it satisfies "
                f"(each worth 1/{n}; sum clamped to 1.0):\n"
                + "\n".join(f"- {c}" for c in subs))
    if compiler == "prose_join":
        bullets = "\n".join(f"- {c.rstrip('?.')}" for c in subs)
        return ("Judge this item's overall quality holistically. Consider:\n"
                + bullets
                + "\nWeigh all of these together as a whole; do not reduce the judgment to a "
                "mechanical checklist or point sum. Output a single overall score in [0,1].")
    if compiler == "echo_prose" and prose:
        return prose
    return ("Judge whether the item meets this standard:\n"
            + "\n".join(f"- {c}" for c in subs))


def gamma_from_f(R, K):
    """Exact submodularity ratio from the precomputed R-dict (R keyed by sorted tuple; f(∅)=0)."""
    def f(S):
        return 0.0 if not S else R[tuple(sorted(S))]
    g = np.inf
    items = list(range(K))
    for r in range(K):
        for S in itertools.combinations(items, r):
            Sset, rest = set(S), [e for e in items if e not in S]
            for ro in range(1, len(rest) + 1):
                for Om in itertools.combinations(rest, ro):
                    denom = f(Sset | set(Om)) - f(S)
                    if denom > 1e-6:
                        num = sum(f(Sset | {e}) - f(S) for e in Om)
                        g = min(g, num / denom)
    return float(min(1.0, g)) if np.isfinite(g) else float("nan")


def greedy_on_R(R, K, budget):
    S = []
    for _ in range(budget):
        cands = [(R[tuple(sorted(S + [e]))] - (R[tuple(sorted(S))] if S else 0.0), e)
                 for e in range(K) if e not in S]
        if not cands:                  # all K criteria already in S (budget >= K) -> stop
            break
        g, e = max(cands)
        if g <= 1e-9:
            break
        S.append(e)
    return sorted(S), (R[tuple(sorted(S))] if S else 0.0)


def exact_certificate(R, subsets, K, budget, M, task, fR=None):
    """The EXACT within-class certificate (§6.7a) + the large_omega approximation CROSS-CHECK.

    Pure function on a precomputed R-dict (all 2^K−1 subsets scored): reports the exact greedy/
    OPT_Ω/global optimum, the PRUNE diagnostic, γ on the monotone envelope R↑, and — using the SAME
    R — how well U₂ (on R↑) and double-greedy (on raw R) match the exact optimum. That cross-check
    is the scientific payload: it measures, on real non-monotone R, how badly PRUNE breaks the
    approximation that becomes the ONLY certificate at large Ω. Called by OmegaCertificate (exact
    mode). `fR` is the f(S)->float over R (built by the caller); if None it is rebuilt here."""
    from .large_omega import U2_bound, double_greedy, f_from_dict
    fR = fR or f_from_dict(R)
    Sg, fg = greedy_on_R(R, K, budget)
    opt_k = max((s for s in subsets if len(s) <= budget), key=lambda s: R[tuple(sorted(s))])
    fo = R[tuple(sorted(opt_k))]
    glob = max(subsets, key=lambda s: R[tuple(sorted(s))]); fglob = R[tuple(sorted(glob))]
    full = R[tuple(range(K))]
    print(f"\n===== {task}: EXACT within-class optimum (K={K}, budget k={budget}, M base-rate={np.mean(M):.2f}) =====")
    print(f"greedy(k)  picks {Sg}  R={fg:.3f}")
    print(f"OPT_Ω(k≤{budget}) picks {list(opt_k)}  R={fo:.3f}")
    print(f"*** CERTIFICATE: greedy rubric = {fg/fo if fo>1e-9 else float('nan'):.3f} of the EXACT OPT_Ω "
          f"(budget {budget}). No approximation — OPT_Ω is brute-forced, so this is the global optimum "
          f"of the criterion class, not a heuristic local optimum. ***")
    print(f"global best (any size) {list(glob)} (|S|={len(glob)}) R={fglob:.3f}; full-set R={full:.3f}")
    pruned = len(glob) < K and fglob > full + 0.01
    if pruned:
        print(f"  -> NON-MONOTONE (PRUNE confirmed): best rubric DROPS {K-len(glob)} criteria; "
              f"adding all {K} LOWERS R {fglob:.3f}->{full:.3f}. So submodularity-ratio γ on raw R is "
              f"ill-defined; γ is reported below only on the monotone envelope R↑ (diagnostic).")
    else:
        print(f"  -> monotone-ish: full set within {full-fglob:+.3f} of best")
    g_env = gamma_from_f(monotone_envelope(R, K), K)
    print(f"γ on monotone envelope R↑ = {g_env:.3f}  (diagnostic: what submodularity would certify if "
          f"you could NOT brute-force; γ≈1 ⇒ envelope submodular). The exact certificate above does not "
          f"need it.")

    # the §6.1/§3.2 approximation, CROSS-CHECKED against exact (does U2/double-greedy match exact
    # OPT on THIS real, non-monotone R? — the open question the theory leaves hanging). U2 needs γ>0;
    # a degenerate R (γ=0, e.g. tiny fake-back-end samples) ⇒ withhold U2, keep double-greedy.
    dg = double_greedy(fR, K, randomized=True, seed=0)
    print(f"\n--- APPROXIMATION cross-check (large_omega, on the same R) ---")
    if g_env > 1e-6:
        u2_env = U2_bound(f_from_dict(monotone_envelope(R, K)), K, gamma=g_env, budget=budget)
        print(f"U2 (R↑, γ={g_env:.3f}): U2={u2_env['U2']:.3f}  [valid upper bound iff R↑ monotone]")
        print(f"  U2 − OPT = {u2_env['U2']-fo:+.3f}  (≥0 is the bound holding; small ⇒ tight per-instance)")
        u2_val = u2_env["U2"]
    else:
        print(f"U2: WITHHELD — γ=0 on R↑ (degenerate R; U2 needs γ>0).")
        u2_val = float("nan")
    print(f"double-greedy (raw R, ½-approx): picks {dg['S']} R={dg['f_S']:.3f}  vs exact global {fglob:.3f}")
    dg_ratio = dg["f_S"] / fglob if fglob > 1e-9 else float("nan")
    print(f"  double-greedy/global = {dg_ratio:.3f}  (½ is the GUARANTEED floor only if R submodular; "
          f"PRUNE breaks that — this number measures the break, not a guaranteed ratio)")
    return {"greedy_S": Sg, "greedy_R": fg, "OPT_S": list(opt_k), "OPT_R": fo,
            "global_S": list(glob), "global_R": fglob, "full_R": full, "pruned": bool(pruned),
            "gamma_env": g_env, "U2_env": u2_val, "dg_S": dg["S"], "dg_R": dg["f_S"]}


def main():
    """Deprecated shim — the entry point is now `omega_certificate.py` (OmegaCertificate), which
    auto-detects K and dispatches to this module (exact) or large_omega (fallback). Kept so old
    invocations still run, in EXACT mode only (it ignores --large-k and forces brute force).
    (This file was renamed from brute_force_class.py → small_omega_brute_force.py.)"""
    import sys
    print("NOTE: small_omega_brute_force (formerly brute_force_class) is the EXACT engine; the entry "
          "point is omega_certificate.OmegaCertificate (auto-dispatches on K). Forwarding this "
          "invocation to OmegaCertificate in exact mode (--large-k forced to a large value).\n",
          file=sys.stderr)
    from .omega_certificate import OmegaCertificate
    args = OmegaCertificate.from_argv(sys.argv[1:], force_large_k=10 ** 9)   # returns a plain dict
    OmegaCertificate(**args).run()


if __name__ == "__main__":
    main()
