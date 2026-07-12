"""§6.6 on REAL rubric criteria — Discovery-to-Selection γ (Phase 1, ≤1 GPU offline vLLM).

Pipeline:  pick a metric rubric  ->  M = full-rubric YES/NO verdict per item (n_passes -> majority)
           ->  DECOMPOSE the rubric into Ω atomic criteria (LLM)  ->  X_i = per-criterion verdict
           ->  ideal recovery Ǐ(S)=I(M;X_S); exact γ; per-pair co-information; greedy vs OPT_Ω
               (all via experiments.submod_conditional — the SAME analysis validated on planted data).

Tests on REAL criteria whether §6.6's conditional theorem holds: CI-given-M ⇒ γ≈1 (greedy near-OPT),
synergy ⇒ γ<1 (greedy fails; co-information flags the pair). Saves the (M, X) signal matrix to .npz so
the analysis re-runs with NO GPU. Phase 2 (executor bottleneck R vs Ǐ via re-scored subsets) is separate.

GPU run (sk3):  HOME=/lfs ... $PY -m methods.metric_implementer.experiments.real_gamma \
                  --pool <pool.jsonl.gz> --text-col facts --task law --rubric-text "<rubric>" \
                  --model meta-llama/Llama-3.1-8B-Instruct --n-items 60 --passes 3 --max-k 9 --out <npz>
Plumbing check (no GPU):  ... real_gamma --inject synergy --out /tmp/g.npz
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from .. import config as cfgmod
from . import submod_conditional as sc

_YESNO = ("{rubric}\n\nDoes the following item satisfy the criterion above? "
          "Answer with exactly one word: YES or NO.\n\nITEM:\n{text}")
_DECOMP_SYS = "You split an evaluation rubric into atomic, independently-checkable criteria."
_DECOMP = ("Rubric:\n{rubric}\n\nIdentify the DISTINCT quality DIMENSIONS this rubric evaluates "
           "(e.g. if it scores descriptive depth, stylistic engagement, grammatical fluency, and "
           "show-vs-tell, those are four separate dimensions). Output ONE atomic criterion PER "
           "DIMENSION: a single positive, independently-checkable YES/NO question asking whether the "
           "{noun} does well on THAT one dimension (e.g. \"Does the {noun} use vivid sensory detail "
           "and imagery?\"). Do NOT fragment one dimension into threshold buckets (no \"exactly N\" / "
           "point-value clauses); do NOT include scoring weights or aggregation rules; each criterion "
           "must stand alone. At most {k} lines, one criterion per line, no numbering, no preamble.")


def _signal(backend, rubric, texts, max_chars):
    """Per-item CONTINUOUS P(YES) via the logprob readout (recon_channel._pyes style): always
    defined, deterministic. NaN only if neither YES/NO token is in the top logprobs (rare).
    Binarization is done downstream by MEDIAN split (balanced) so M/criteria can't collapse to a
    constant — the historical 0.5-threshold collapse was an artifact of an earlier (now-removed)
    holistic-quality target; with a behavioral full-rubric M it does not arise."""
    prompts = [_YESNO.format(rubric=rubric, text=t[:max_chars]) for t in texts]
    return np.array(backend.score_binary(prompts, pos="YES", neg="NO"), dtype=float)


def _median_split(v):
    """Continuous -> balanced binary at the median (ties to the high side)."""
    return (np.asarray(v, float) >= float(np.median(v))).astype(float)


def _majority(passes, base_rate=None):
    """(N, R) verdict matrix -> (N,) binary by majority; NaN if no applicable pass."""
    P = np.asarray(passes, float)
    out = np.full(P.shape[0], np.nan)
    for i in range(P.shape[0]):
        row = P[i][np.isfinite(P[i])]
        if row.size:
            out[i] = 1.0 if row.mean() >= 0.5 else 0.0
    return out


def _decompose(backend, rubric, noun, k):
    raw = backend.generate(_DECOMP.format(rubric=rubric, k=k, noun=noun),
                           system=_DECOMP_SYS, max_tokens=500, temperature=0.3)
    lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip() for ln in (raw or "").splitlines()]
    crits = [ln for ln in lines if len(ln) > 10]
    return crits[:k]


def _analyze(M, X, crits, label):
    M = M.astype(int)
    X = X.astype(int)
    K = X.shape[1]
    budget = max(2, K // 2)              # BINDING (< K) so greedy can fail on synergy
    g = sc.gamma_exact(M, X, K) if K <= 12 else float("nan")
    # spectral γ (Das-Kempe λ_min): DOWNGRADED 2026-06-22 to a HEURISTIC REDUNDANCY DIAGNOSTIC, NOT a
    # certificate. It lower-bounds γ only if the executor combines criteria LINEARLY (regression value
    # function); an LLM judge mixes them through attention -- exactly the non-linear interaction γ<1
    # captures -- so λ_min can sit ABOVE the true γ and is unsafe as a guarantee (theory §6.2). The
    # trustworthy route is the Shannon-CMI orthogonalization filter (experiments/orthogonalize.py) then
    # brute-force the within-class optimum for |Ω|≤15 (bypasses γ). Kept here only as a cheap pairwise-
    # redundancy readout for diagnostics.
    cm = np.nan_to_num(np.corrcoef(X.astype(float).T), nan=0.0) if K > 1 else np.array([[1.0]])
    gamma_spec = float(max(0.0, min(1.0, np.linalg.eigvalsh(cm)[0])))
    Sg, fg = sc.greedy(M, X, K, budget)
    So, fo = sc.opt_brute(M, X, K, budget)
    print(f"\n===== {label}  (N={len(M)}, K={K}, M base-rate={M.mean():.2f}) =====")
    ratio = fg / fo if fo > 1e-9 else float("nan")
    # curvature α (Ǐ(S)=I(M;X_S) is monotone, so the greedy certificates apply):
    #   α = 1 − min_e δ(e | U\{e}) / δ(e | ∅)   (∈[0,1]; α→0 modular ⇒ greedy exact)
    sing = {e: sc._mi(M, X[:, e]) for e in range(K)}
    fullv = sc.f_recovery(M, X, range(K))
    aterms = [(fullv - sc.f_recovery(M, X, [x for x in range(K) if x != e])) / sing[e]
              for e in range(K) if sing[e] > 1e-9]
    alpha = min(1.0, max(0.0, 1.0 - min(aterms))) if aterms else 1.0
    das_kempe = 1.0 - np.exp(-g)                      # greedy ≥ (1−e^{−γ})·OPT  (monotone)
    bian = (1.0 / alpha) * (1.0 - np.exp(-alpha * g)) if alpha > 1e-9 else g
    cert = max(das_kempe, bian)                       # both valid; report the tighter
    print(f"spectral γ = {gamma_spec:.3f} (heuristic redundancy DIAGNOSTIC, NOT a certificate — §6.2)   "
          f"exact γ = {g:.3f} (joint-MI; trust only if 2^K≪N={len(M)})   curvature α = {alpha:.3f}   "
          f"(k={budget} of K={K})")
    print(f"greedy picks {Sg} f={fg:.3f}   OPT_Ω picks {So} f={fo:.3f}")
    print(f"*** OPTIMALITY GUARANTEE: greedy rubric ≥ {cert:.3f}·OPT_Ω (certified, Bian/Das-Kempe); "
          f"realized greedy/OPT_Ω = {ratio:.3f} ***")
    # complexity of the metric (NOT raw |S|): info-weighted effective #units + greedy accumulation curve
    optset = list(So)
    deltas = [max(0.0, sc.f_recovery(M, X, optset) - sc.f_recovery(M, X, [x for x in optset if x != e]))
              for e in optset]
    sden = sum(d * d for d in deltas)
    neff = (sum(deltas) ** 2 / sden) if sden > 1e-12 else float("nan")
    curve, Sacc = [], []
    for _ in range(K):
        gains = [(sc.f_recovery(M, X, Sacc + [e]) - sc.f_recovery(M, X, Sacc), e)
                 for e in range(K) if e not in Sacc]
        if not gains:
            break
        gv, e = max(gains)
        Sacc.append(e)
        curve.append(round(sc.f_recovery(M, X, Sacc), 3))
    print(f"metric complexity: N_eff = {neff:.2f} effective units of |OPT|={len(optset)} raw "
          f"(info-weighted; collapses redundant/useless units)")
    print(f"greedy accumulation R(S_k) = {curve}  (saturation shape = how distributed the signal is)")
    print("per-criterion I(M;X_i) and fire-rate:")
    for i in range(K):
        print(f"  X_{i}: I={sc._mi(M, X[:, i]):.3f}  fire={X[:, i].mean():.2f}  | {crits[i][:66]}")
    from itertools import combinations
    pairs = sorted(((sc._cmi(X[:, i], X[:, j], M) - sc._mi(X[:, i], X[:, j]), i, j)
                    for i, j in combinations(range(K), 2)), reverse=True)
    syn = [(round(c, 3), i, j) for c, i, j in pairs if c > 0.02]
    print(f"synergy pairs (co-info>0): {syn if syn else 'none — redundant/submodular'}")
    return {"label": label, "gamma": g, "gamma_spectral": gamma_spec, "alpha": alpha, "cert_floor": cert,
            "greedy_over_opt": ratio, "K": K, "synergy_pairs": syn, "M_rate": float(M.mean()),
            "R_opt": float(fo), "N_eff": float(neff), "accum_curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", choices=["ci", "synergy"], help="bypass backend; plant matrix (plumbing test)")
    ap.add_argument("--pool"); ap.add_argument("--text-col", default="text")
    ap.add_argument("--task", default="law"); ap.add_argument("--rubric-text"); ap.add_argument("--rubric-file")
    ap.add_argument("--bank", help="online-rubric json with extracted.rubrics_metrics (pre-decomposed Ω)")
    ap.add_argument("--crit-file", help="pre-mined Ω: one criterion per bullet/line, loaded DIRECTLY "
                    "(NO LLM decomposition — avoids the _DECOMP independence-prompt confound)")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--n-items", type=int, default=60); ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--max-k", type=int, default=9); ap.add_argument("--out", default="/tmp/real_gamma.npz")
    ap.add_argument("--fire-lo", type=float, default=0.12); ap.add_argument("--fire-hi", type=float, default=0.88)
    ap.add_argument("--min-std", type=float, default=0.04, help="drop criteria whose P(YES) std is below this")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()

    if a.inject:                                   # ---- no-GPU plumbing validation ----
        rng = np.random.default_rng(0)
        if a.inject == "ci":
            M, X = sc.plant_ci(400, 6, rng)
        else:
            M, X = sc.plant_synergy(400, 6, rng)
        crits = [f"planted-criterion-{i}" for i in range(X.shape[1])]
        np.savez(a.out, M=M, X=X, crits=crits)
        _analyze(M, X, crits, f"INJECT[{a.inject}]")
        return

    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    cfg.vllm_fake = a.fake
    from ..vllm_backend import make_judge_backend
    backend = make_judge_backend(a.model, cfg)

    if a.pool.endswith(".parquet"):
        df = pd.read_parquet(a.pool)
    elif a.pool.endswith(".csv") or a.pool.endswith(".csv.gz"):
        df = pd.read_csv(a.pool)                       # pandas auto-decompresses .gz
    else:
        df = pd.read_json(a.pool, lines=True, compression="gzip" if a.pool.endswith(".gz") else None)
    df = df[df[a.text_col].astype(str).str.len() > 50]  # drop empty/stub rows
    texts = df[a.text_col].astype(str).tolist()[: a.n_items]
    print(f"loaded {len(texts)} items from {a.pool}")

    mc, temp = cfg.max_text_chars, cfg.other_temperature
    if a.bank:
        # online-rubric bank: extracted.rubrics_metrics is ALREADY a decomposed criterion set ->
        # no LLM decomposition needed (removes that confound). M = the full (all-criteria) standard.
        rm = json.load(open(a.bank))["extracted"]["rubrics_metrics"][: a.max_k]
        crits = [f"{m.get('name','')}: {m.get('description','')}".strip(": ") for m in rm]
        rubric = "Judge whether the item meets this overall standard:\n" + "\n".join(f"- {c}" for c in crits)
        print(f"bank gave {len(crits)} pre-decomposed criteria")
    elif a.crit_file:
        # pre-mined Ω (e.g. GEPA-evolved criteria): bullets ARE the criteria, loaded verbatim. No LLM
        # decomposition -> no independence-prompt confound; γ is measured on Ω as discovered.
        crits = []
        for ln in open(a.crit_file).read().splitlines():
            s = ln.strip()
            if s[:1] in "-*•" or (len(s) > 1 and s[0].isdigit() and s[1] in ").:"):
                c = s.lstrip("-*•0123456789.):; ").strip()
                if len(c) > 10:
                    crits.append(c)
        crits = crits[: a.max_k]
        rubric = "Judge whether the item meets this overall standard:\n" + "\n".join(f"- {c}" for c in crits)
        print(f"loaded {len(crits)} pre-mined criteria DIRECTLY from {a.crit_file} (no decomposition)")
    else:
        rubric = a.rubric_text or (open(a.rubric_file).read() if a.rubric_file else None)
        if a.rubric_file and a.rubric_file.endswith(".json"):
            obj = json.loads(rubric)
            rubric = obj.get("body") or obj.get("rubric") or obj.get("prompt") or obj.get("description") or rubric
        assert rubric, "need --bank, --rubric-text, or --rubric-file"
        crits = _decompose(backend, rubric, cfg.item_noun, a.max_k)
        print(f"decomposed into {len(crits)} criteria")

    # M target = the FULL criterion set's verdict (the canonical rubric C(Ω)), a BEHAVIORAL target —
    # NOT a "holistic quality" strong-LLM anchor (no-anchor pivot; see small_omega_brute_force). The full
    # rubric is already built above as `rubric` from the bank / crit-file / decomposition.
    M = _signal(backend, rubric, texts, mc)          # logprob readout: deterministic, no parse loss
    Xcols = [_signal(backend, c, texts, mc) for c in crits]
    X = np.column_stack(Xcols) if Xcols else np.zeros((len(texts), 0))

    keep = np.isfinite(M) & np.all(np.isfinite(X), axis=1)
    M, X = M[keep], X[keep]
    # DROP DEAD criteria by VARIATION of continuous P(YES) (std), not fire-rate: a criterion whose
    # P(YES) barely moves across items carries ~no discriminating signal regardless of its level.
    sd = X.std(axis=0) if X.size else np.array([])
    live = sd > a.min_std
    dropped = [(round(float(X[:, i].mean()), 2), round(float(sd[i]), 3), crits[i][:46])
               for i in range(len(crits)) if not live[i]]
    X, crits = X[:, live], [c for c, v in zip(crits, live) if v]
    print(f"kept {keep.sum()} items; {X.shape[1]} LIVE criteria (std>{a.min_std}); "
          f"dropped {len(dropped)} flat: {dropped[:6]}")
    # BEHAVIORAL dedup (the theory's individuation): two criteria are the SAME element iff they induce
    # the same per-item signal. Surface paraphrases ("Determine/Check/Verify X") collapse here.
    if X.shape[1] >= 2:
        keepb = [0]
        for i in range(1, X.shape[1]):
            if all(abs(np.corrcoef(X[:, i], X[:, j])[0, 1]) <= 0.97 for j in keepb):
                keepb.append(i)
        merged = [crits[i][:46] for i in range(X.shape[1]) if i not in keepb]
        X, crits = X[:, keepb], [crits[i] for i in keepb]
        print(f"behavioral dedup: merged {len(merged)} near-duplicate-signal criteria "
              f"-> {X.shape[1]} behaviorally-distinct; merged: {merged[:5]}")
    # MEDIAN-split to balanced binary so M/criteria can't be constant (the degenerate-M fix)
    Mb = _median_split(M)
    Xb = np.column_stack([_median_split(X[:, i]) for i in range(X.shape[1])]) if X.shape[1] else X
    np.savez(a.out, M=Mb, X=Xb, M_cont=M, X_cont=X, crits=np.array(crits, dtype=object))
    if Xb.shape[1] >= 2 and len(Mb) >= 8 and len(np.unique(Mb)) >= 2:
        _analyze(Mb, Xb, crits, f"{a.task}:{(a.rubric_file or 'rubric')}")
    else:
        print("degenerate (M has no variation, or <2 live criteria) — saved matrix; analysis skipped")


if __name__ == "__main__":
    main()
