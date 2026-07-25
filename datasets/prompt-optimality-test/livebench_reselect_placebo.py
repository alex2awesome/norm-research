"""P0 controls for the livebench random-recombination result (2026-07-25, advisor-mandated).

Two defects in the first pass, both fixed here, in ONE session so nothing is cross-session:

DEFECT 1 - TEST-PANEL SELECTION. The rank-certificate draws were scored on TEST, and we then
promoted the test-argmax (draw #86) and reported it on the SAME 126 test items. The k=3 fresh
rescore cures winner's-curse-on-noise but NOT prompt-to-item overfit: the argmax of 120 draws is
partly the draw that best fits those particular items. PHASE 1 repairs this the standard way -
rescore the top-5 test draws on the SELECT panel (train[:select_n], never touched by test),
promote the SELECT-argmax, and report that candidate on test.

DEFECT 2 - NO PLACEBO. "120/120 random recombinations beat the GEPA prompt" has an alternative
explanation: appending ~24 clauses of ANYTHING changes length/structure and suppresses the
abstain->zero failure mode that livebench's Levenshtein partial-credit metric punishes with an
exact 0. PHASE 2 runs a placebo generator - identical inclusion process and identical per-draw
clause COUNT, but clauses drawn from ANOTHER benchmark's frozen pool (hover), i.e. length-matched
and structure-matched but content-irrelevant. If placebo also beats the init, the finding is
"livebench punishes short prompts" and the content claim dies.

PHASE 2 also interleaves GEPA-init replicates among the draws in a RANDOMIZED order, which fixes
a third weakness: the 120/120 count compared 120 draw readings against essentially ONE init
reading (.6438), on a benchmark where that same prompt has measured .479-.6956 across sessions.
Randomized interleaving makes session drift orthogonal to arm and gives the init its own
within-session replicate distribution. Separation is only real if the init replicate MAX sits
below the draw MIN.

Reproducibility note: draw i of the certificate is recoverable exactly because the mask sequence
came from np.random.default_rng(0) advanced i+1 times over the same 48-unit pool.

Usage:
  python3 livebench_reselect_placebo.py --task-lm openai/Qwen3-8B \
      --api-base http://127.0.0.1:8078/v1 --max-tokens 24000
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


def replay_masks(n_units: int, n_draws: int, include_p: float, seed: int):
    """Reproduce the exact mask sequence the certificate used."""
    rng = np.random.default_rng(seed)
    return [rng.random(n_units) < include_p for _ in range(n_draws)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="livebench")
    ap.add_argument("--lm-tag", default="Qwen3-8B")
    ap.add_argument("--task-lm", required=True)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--placebo-bench", default="hover",
                    help="frozen pool to draw content-irrelevant placebo clauses from")
    ap.add_argument("--top-k-reselect", type=int, default=5)
    ap.add_argument("--select-n", type=int, default=81)
    ap.add_argument("--n-control", type=int, default=40,
                    help="real draws AND placebo draws, each")
    ap.add_argument("--n-init-replicates", type=int, default=10)
    ap.add_argument("--include-p", type=float, default=0.5)
    ap.add_argument("--cert-seed", type=int, default=0)
    ap.add_argument("--control-seed", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--eval-threads", type=int, default=32)
    ap.add_argument("--final-passes", type=int, default=3)
    a = ap.parse_args()

    lm = dspy.LM(a.task_lm, api_base=a.api_base, api_key="EMPTY", temperature=a.temperature,
                 top_p=a.top_p, max_tokens=a.max_tokens, num_retries=10, timeout=300,
                 extra_body={"top_k": a.top_k})
    dspy.configure(lm=lm)
    px.EVAL_THREADS = a.eval_threads

    bench, program, metric, _ = px.load_bench(a.bench)
    bench_dir = HERE / "runs_paperexact" / a.bench / a.lm_tag
    seed_cand = px.get_instructions(program)
    init_cand = {**seed_cand,
                 **json.loads((bench_dir / "official" / "result.json").read_text())["best_candidate"]}

    pool = json.loads((HERE / "pools" / f"{a.bench}_{a.lm_tag}_frozen.json").read_text())
    units = [(d["module"], d["unit"]) for d in pool["units"] if d["module"] in seed_cand]
    placebo_pool = json.loads(
        (HERE / "pools" / f"{a.placebo_bench}_{a.lm_tag}_frozen.json").read_text())
    # placebo clauses get retargeted onto THIS bench's modules; content is foreign by design
    placebo_units = [d["unit"] for d in placebo_pool["units"]]

    test = list(bench.test_set)
    select_panel = list(bench.train_set)[:a.select_n]
    rundir = bench_dir / "reselect_placebo"
    rundir.mkdir(parents=True, exist_ok=True)
    log_path = rundir / "evals.jsonl"
    out_path = HERE / "runs" / f"reselect_placebo_{a.bench}_{a.lm_tag}.json"
    out_path.parent.mkdir(exist_ok=True)

    def build(clauses):
        cand = dict(init_cand)
        mod = units[0][0]
        for c in clauses:
            cand[mod] = cand[mod] + "\n- " + c
        return cand

    print(f"[{a.bench}] pool={len(units)} placebo_pool={len(placebo_units)} "
          f"select={len(select_panel)} test={len(test)}", flush=True)
    result = {"bench": a.bench, "lm_tag": a.lm_tag,
              "max_tokens": a.max_tokens, "include_p": a.include_p}

    # ---------------- PHASE 1: re-select on the SELECT panel ----------------
    cert = json.loads((bench_dir / "rank_certificate" / "running.json").read_text())
    cert_scores = [s if s is not None else -1 for s in cert["scores"]]
    masks = replay_masks(len(units), len(cert_scores), a.include_p, a.cert_seed)
    top_idx = list(np.argsort(cert_scores)[::-1][:a.top_k_reselect])
    print(f"PHASE 1 re-selection: top-{a.top_k_reselect} test draws = "
          f"{[(int(i), round(cert_scores[i], 4)) for i in top_idx]}", flush=True)

    reselect = []
    for i in top_idx:
        clauses = [u[1] for u, m in zip(units, masks[i]) if m]
        s = px.evaluate_cand(program, build(clauses), select_panel, metric, log_path,
                             f"reselect_draw_{i}")
        reselect.append({"draw": int(i), "n_units": len(clauses),
                         "test_score_at_selection": cert_scores[i], "select_score": s})
        print(f"  draw {i}: select={s:.4f} (test-at-selection {cert_scores[i]:.4f})", flush=True)
    best = max(reselect, key=lambda r: r["select_score"] if r["select_score"] is not None else -1)
    result["phase1_reselect"] = {"candidates": reselect, "promoted_draw": best["draw"],
                                 "note": "promoted by SELECT panel; test never used for selection"}
    print(f"  -> promoted draw #{best['draw']} on select ({best['select_score']:.4f})", flush=True)
    out_path.write_text(json.dumps(result, indent=1))

    # ---------------- PHASE 2: interleaved real / placebo / init ----------------
    rng = np.random.default_rng(a.control_seed)
    jobs = []
    for j in range(a.n_control):
        mask = rng.random(len(units)) < a.include_p
        jobs.append(("real", j, [u[1] for u, m in zip(units, mask) if m]))
    # placebo: match each real draw's clause COUNT exactly, content from another benchmark
    for j in range(a.n_control):
        k = len(jobs[j][2])
        pick = rng.choice(len(placebo_units), size=min(k, len(placebo_units)), replace=False)
        jobs.append(("placebo", j, [placebo_units[int(x)] for x in pick]))
    for j in range(a.n_init_replicates):
        jobs.append(("init", j, []))
    order = rng.permutation(len(jobs))          # session drift now orthogonal to arm
    print(f"PHASE 2: {len(jobs)} interleaved evals in randomized order", flush=True)

    recs = []
    for pos, ji in enumerate(order):
        arm, j, clauses = jobs[int(ji)]
        s = px.evaluate_cand(program, build(clauses), test, metric, log_path,
                             f"ctrl_{arm}_{j}")
        recs.append({"order_pos": pos, "arm": arm, "idx": j,
                     "n_clauses": len(clauses), "score": s})
        if (pos + 1) % 10 == 0:
            print(f"  {pos+1}/{len(jobs)} done", flush=True)
        result["phase2_controls"] = recs
        out_path.write_text(json.dumps(result, indent=1))

    def arr(name):
        return np.array([r["score"] for r in recs if r["arm"] == name and r["score"] is not None])
    real, plac, init = arr("real"), arr("placebo"), arr("init")

    def stats(x):
        return {} if not len(x) else {"n": int(len(x)), "mean": float(x.mean()),
                                      "sd": float(x.std()), "min": float(x.min()),
                                      "max": float(x.max())}
    summ = {"real": stats(real), "placebo": stats(plac), "init": stats(init)}
    if len(real) and len(init):
        summ["separation_real_vs_init"] = bool(real.min() > init.max())
    if len(plac) and len(init):
        summ["separation_placebo_vs_init"] = bool(plac.min() > init.max())
    if len(real) and len(plac):
        # Mann-Whitney U via rank sum, normal approx (n>=40 each)
        allx = np.concatenate([real, plac])
        ranks = allx.argsort().argsort().astype(float) + 1
        # average ranks for ties
        for v in np.unique(allx):
            idx = np.where(allx == v)[0]
            if len(idx) > 1:
                ranks[idx] = ranks[idx].mean()
        n1, n2 = len(real), len(plac)
        u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2
        mu, sd = n1 * n2 / 2, np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z = (u1 - mu) / sd if sd else 0.0
        summ["real_vs_placebo"] = {"U": float(u1), "z": float(z),
                                   "interpretation": "z>0 => real content beats placebo"}
    result["phase2_summary"] = summ
    print(json.dumps(summ, indent=1), flush=True)
    out_path.write_text(json.dumps(result, indent=1))

    # ---------------- PHASE 3: k-pass test of the SELECT-promoted candidate ----------------
    clauses = [u[1] for u, m in zip(units, masks[best["draw"]]) if m]
    promoted = build(clauses)
    s_init = px.evaluate_cand(program, init_cand, test, metric, log_path, "final_init",
                              passes=a.final_passes)
    s_prom = px.evaluate_cand(program, promoted, test, metric, log_path, "final_promoted",
                              passes=a.final_passes)
    result["phase3_final"] = {"passes": a.final_passes, "gepa_init_test": s_init,
                              "select_promoted_test": s_prom,
                              "delta": (s_prom - s_init) if (s_prom and s_init) else None,
                              "promoted_draw": best["draw"], "n_units": len(clauses)}
    (rundir / "promoted_candidate.json").write_text(json.dumps(
        {"best_candidate": promoted, "promoted_draw": best["draw"],
         "selection": "select-panel argmax of top-5 test draws"}, indent=1))
    print(json.dumps(result["phase3_final"], indent=1), flush=True)
    result["ts"] = time.time()
    out_path.write_text(json.dumps(result, indent=1))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
