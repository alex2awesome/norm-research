"""CERTIFIED class-relative ceiling #2: the RANK / EXCHANGEABILITY certificate (2026-07-24).

WHY THIS REPLACES EVT. The EVT endpoint was audited and retracted (HB86): with 11-75 adaptive
candidates on a 1/n score grid the fitted GPD shape sits at xi < -1, where the MLE is pinned to
the largest order statistic, so the "endpoint" was exactly the sample max (margin 0.00e+00 on
every bench). Three things were broken at once: (a) n far too small, (b) heavy discreteness/ties,
(c) i.i.d. violated because GEPA's proposals condition on earlier scores. Re-fitting cannot fix
any of them.

THE FIX IS TO CHANGE THE SAMPLING, NOT THE ESTIMATOR. If prompts are drawn I.I.D. from a
DECLARED, FROZEN, NON-ADAPTIVE generator G, then exchangeability alone gives a finite-sample,
distribution-free certificate — no tail model, no asymptotics, no shape parameter, and ties only
make it more conservative:

    for a fresh draw X_{N+1} ~ G,      P( X_{N+1} > max(X_1..X_N) )  <=  1/(N+1)
    more generally                     P( X_{N+1} > X_(N-k+1) )      <=  k/(N+1)

That is exactly the paper's claim ("nothing in this class beats what we shipped") stated as a
certificate instead of an extrapolation. It bounds the GENERATOR's tail, not all of prompt space
— which is honest, and is the object Paper #2 actually argues about.

G HERE = uniform random recombination: each unit of the frozen pool is included independently
with probability p (declared; default 0.5), appended to the GEPA-shipped init in pool order.
This is the same class the M_omega search explores, so the certificate speaks directly to it.

NOISE NOTE — CORRECTED 2026-07-25 (the earlier version of this note was WRONG and the error
reached a draft; do not restore it). It previously claimed observed = true + symmetric noise, so
the observed max is biased upward and the bound is "conservative as a bound on true skill". That
is backwards on these benchmarks: livebench-style noise is predominantly ONE-SIDED DOWNWARD
(item errors, truncation, timeouts, missing-dependency exceptions all force a hard 0), so
observed <= true per item and the observed max can sit BELOW the best true skill in the class.
The certificate therefore does NOT upper-bound achievable true skill.

WHAT IT DOES BOUND, exactly: under the declared metric AND the declared measurement protocol,
a fresh draw's MEASURED score exceeds the k-th largest measured score with probability <= k/(N+1).
This is protocol-relative in the same way the metric-reachability cap is, and it is defensible as
such. (Winner's curse — the argmax draw's observed score overstating ITS OWN skill — is a
different statement and remains true.) For a tighter true-skill reading, evaluate draws at k>=3
passes so the downward-noise mass shrinks, and state the residual gap explicitly.

Usage:
  python3 bound_rank_certificate.py livebench --lm-tag Qwen3-8B \
      --task-lm openai/Qwen3-8B --api-base http://127.0.0.1:8078/v1 \
      --n-draws 150 --max-tokens 24000 --include-p 0.5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import dspy
import numpy as np

import paperexact_arms as px

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", choices=["aime", "hover", "hotpot", "ifbench", "livebench", "pupa"])
    ap.add_argument("--lm-tag", default="Qwen3-8B")
    ap.add_argument("--task-lm", required=True)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--pool-file", default=None)
    ap.add_argument("--n-draws", type=int, default=150)
    ap.add_argument("--include-p", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--eval-threads", type=int, default=32)
    ap.add_argument("--n-items", type=int, default=0, help="0 = full test split")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    lm = dspy.LM(a.task_lm, api_base=a.api_base, api_key="EMPTY", temperature=a.temperature,
                 top_p=a.top_p, max_tokens=a.max_tokens, num_retries=10, timeout=300,
                 extra_body={"top_k": a.top_k})
    dspy.configure(lm=lm)
    px.EVAL_THREADS = a.eval_threads

    bench, program, metric, _ = px.load_bench(a.bench)
    bench_dir = HERE / "runs_paperexact" / a.bench / a.lm_tag
    seed_cand = px.get_instructions(program)
    init_cand = dict(seed_cand)
    off = bench_dir / "official" / "result.json"
    if off.exists():
        init_cand = {**seed_cand, **json.loads(off.read_text())["best_candidate"]}

    pool_file = Path(a.pool_file) if a.pool_file else HERE / "pools" / f"{a.bench}_{a.lm_tag}_frozen.json"
    units = [(d["module"], d["unit"]) for d in json.loads(pool_file.read_text())["units"]
             if d["module"] in seed_cand]
    test = list(bench.test_set)
    if a.n_items:
        test = test[:a.n_items]
    rng = np.random.default_rng(a.seed)

    rundir = bench_dir / "rank_certificate"
    rundir.mkdir(parents=True, exist_ok=True)
    log_path = rundir / "draws.jsonl"
    print(f"[{a.bench}] pool={len(units)} n_test={len(test)} N={a.n_draws} p={a.include_p} "
          f"max_tokens={a.max_tokens}", flush=True)

    # the shipped candidate, measured on the SAME panel so the comparison is apples-to-apples
    shipped = px.evaluate_cand(program, init_cand, test, metric, log_path, "cert_init")
    print(f"  init (GEPA shipped) = {shipped:.4f}", flush=True)

    scores = []
    for i in range(a.n_draws):
        mask = rng.random(len(units)) < a.include_p
        chosen = [u for u, m in zip(units, mask) if m]
        cand = dict(init_cand)
        for mod, clause in chosen:
            cand[mod] = cand[mod] + "\n- " + clause
        s = px.evaluate_cand(program, cand, test, metric, log_path, f"cert_draw_{i}")
        scores.append(s)
        if (i + 1) % 10 == 0:
            arr = np.array([x for x in scores if x is not None])
            print(f"  draw {i+1}/{a.n_draws}  max={arr.max():.4f} mean={arr.mean():.4f}",
                  flush=True)
        with open(rundir / "running.json", "w") as fh:
            json.dump({"n_done": i + 1, "scores": scores, "shipped": shipped,
                       "ts": time.time()}, fh)

    arr = np.array([x for x in scores if x is not None])
    N = len(arr)
    order = np.sort(arr)[::-1]
    # exchangeability certificate: P(fresh draw > k-th largest) <= k/(N+1)
    ladder = [{"k": k, "threshold": float(order[k - 1]),
               "p_fresh_draw_exceeds_at_most": k / (N + 1)}
              for k in (1, 2, 3, 5, 10) if k <= N]
    out = {
        "bench": a.bench, "lm_tag": a.lm_tag, "n_draws_valid": int(N),
        "generator": {"kind": "uniform random recombination over frozen pool",
                      "pool_file": str(pool_file), "pool_size": len(units),
                      "include_p": a.include_p, "seed": a.seed,
                      "init": "GEPA official shipped candidate"},
        "eval": {"n_items": len(test), "max_tokens": a.max_tokens,
                 "temperature": a.temperature},
        "shipped_init_score": shipped,
        "draw_max": float(arr.max()), "draw_mean": float(arr.mean()),
        "draw_sd": float(arr.std()),
        "frac_draws_beating_init": float((arr > (shipped or 0)).mean()),
        "certificate_ladder": ladder,
        "headline": (f"P(a fresh draw from the declared generator exceeds {float(order[0]):.4f}) "
                     f"<= {1/(N+1):.4f}"),
        "scope": ("distribution-free, finite-sample, exchangeability-only; certifies the TAIL OF "
                  "THIS GENERATOR under THIS MEASUREMENT PROTOCOL, not all of prompt space and "
                  "not true skill. Ties make it conservative. Noise on these benchmarks is "
                  "one-sided downward (errors/truncation force hard 0), so the observed max may "
                  "sit BELOW the class's best true skill — do not read this as a true-skill cap. "
                  "Per-pool: a newly mined pool needs a new certificate."),
    }
    p = HERE / "runs" / f"bound_rank_certificate_{a.bench}_{a.lm_tag}.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in
                      ("shipped_init_score", "draw_max", "draw_mean",
                       "frac_draws_beating_init", "headline")}, indent=1), flush=True)
    print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
