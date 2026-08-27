"""Generic placebo grid (HB103 contingency, fired 2026-07-26 by the livebench falsification).

DESIGN (identical to livebench P0 Phase 2): N real draws (each unit of the bench's OWN frozen
pool included independently with p) vs N placebo draws (clauses from a FOREIGN bench's pool,
count-matched per draw, appended to the same modules-by-position) vs R init replicates, all
interleaved in one randomized order in one session. If the bench's gain is reproduced by
content-free foreign clauses, the gain is structural (prompt bulk), not mined content.

WHY HOTPOT FIRST: livebench's placebo landed in the pre-registered FALSIFIES cell (placebo
.7342 >= real .7055, both >> init .6213), so per HB103#1 the paper's content claim now rests on
whether hotpot's +.220 survives its own placebo.

Notes: fd-hardened (RLIMIT bump), checkpoint/resume per eval, cache=False on the task LM is NOT
needed (single pass per draw, distinct prompts -> no cache collisions), server fingerprint
recorded. Placebo clauses are appended to the REAL pool's module slots by position so module
targeting is structure-matched, not content-matched.

Usage: python3 placebo_grid.py hotpot --placebo-bench livebench --task-lm openai/Qwen3-8B \
         --api-base http://127.0.0.1:8078/v1 --max-tokens 8000
"""
from __future__ import annotations

import argparse
import datetime
import json
import resource
import socket
from pathlib import Path

import dspy
import numpy as np

import paperexact_arms as px

HERE = Path(__file__).parent

try:
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _hard), _hard))
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--placebo-bench", required=True)
    ap.add_argument("--lm-tag", default="Qwen3-8B")
    ap.add_argument("--task-lm", required=True)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--n-control", type=int, default=40)
    ap.add_argument("--n-init-replicates", type=int, default=10)
    ap.add_argument("--include-p", type=float, default=0.5)
    ap.add_argument("--control-seed", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--eval-threads", type=int, default=32)
    a = ap.parse_args()

    lm = dspy.LM(a.task_lm, api_base=a.api_base, api_key="EMPTY", temperature=a.temperature,
                 top_p=a.top_p, max_tokens=a.max_tokens, num_retries=10, timeout=300,
                 extra_body={"top_k": a.top_k})
    dspy.configure(lm=lm)
    px.EVAL_THREADS = a.eval_threads

    bench, program, metric, _ = px.load_bench(a.bench)
    seed_cand = px.get_instructions(program)
    init_cand = {**seed_cand, **json.loads((HERE / "runs_paperexact" / a.bench / a.lm_tag /
                                            "official" / "result.json").read_text())["best_candidate"]}
    pool = [(d["module"], d["unit"]) for d in
            json.loads((HERE / "pools" / f"{a.bench}_{a.lm_tag}_frozen.json").read_text())["units"]
            if d["module"] in seed_cand]
    placebo_texts = [d["unit"] for d in
                     json.loads((HERE / "pools" / f"{a.placebo_bench}_{a.lm_tag}_frozen.json").read_text())["units"]]

    test = list(bench.test_set)
    out_path = HERE / "runs" / f"placebo_grid_{a.bench}_{a.lm_tag}.json"
    out_path.parent.mkdir(exist_ok=True)
    prev = json.loads(out_path.read_text()) if out_path.exists() else {}
    log_path = HERE / "runs_paperexact" / a.bench / a.lm_tag / "placebo_grid" / "evals.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fp = {"host": socket.gethostname(), "api_base": a.api_base,
          "utc": datetime.datetime.utcnow().isoformat() + "Z",
          "max_tokens": a.max_tokens, "placebo_bench": a.placebo_bench}
    print("FINGERPRINT:", json.dumps(fp), flush=True)

    def build(mask=None, placebo_count=None, rng=None):
        cand = dict(init_cand)
        if mask is not None:                      # real draw: own units on their own modules
            for (mod, clause), m in zip(pool, mask):
                if m:
                    cand[mod] = cand[mod] + "\n- " + clause
        elif placebo_count is not None:           # placebo: foreign clauses on matched modules
            pick = rng.choice(len(placebo_texts), size=min(placebo_count, len(placebo_texts)),
                              replace=False)
            for j, x in enumerate(pick):          # same module slots by position as the pool
                mod = pool[j % len(pool)][0]
                cand[mod] = cand[mod] + "\n- " + placebo_texts[int(x)]
        return cand

    rng = np.random.default_rng(a.control_seed)
    jobs = []
    for j in range(a.n_control):
        mask = rng.random(len(pool)) < a.include_p
        jobs.append(("real", j, {"mask": mask.tolist()}))
    for j in range(a.n_control):
        k = sum(jobs[j][2]["mask"])
        jobs.append(("placebo", j, {"count": int(k)}))
    for j in range(a.n_init_replicates):
        jobs.append(("init", j, {}))
    order = rng.permutation(len(jobs))
    done = {r["order_pos"]: r for r in prev.get("controls", [])}
    recs = [done[p_] for p_ in sorted(done)]
    print(f"[{a.bench}] placebo grid: {len(jobs)} evals, {len(done)} resumed", flush=True)

    result = {"bench": a.bench, "placebo_bench": a.placebo_bench, "fingerprint": fp,
              "include_p": a.include_p, "max_tokens": a.max_tokens}
    for pos, ji in enumerate(order):
        if pos in done:
            continue
        arm, j, spec = jobs[int(ji)]
        rng2 = np.random.default_rng(1000 + j)
        if arm == "real":
            cand = build(mask=np.array(spec["mask"]))
        elif arm == "placebo":
            cand = build(placebo_count=spec["count"], rng=rng2)
        else:
            cand = dict(init_cand)
        s = px.evaluate_cand(program, cand, test, metric, log_path, f"grid_{arm}_{j}")
        recs.append({"order_pos": pos, "arm": arm, "idx": j, "score": s})
        result["controls"] = recs
        out_path.write_text(json.dumps(result, indent=1))
        if (pos + 1) % 10 == 0:
            print(f"  {pos+1}/{len(jobs)}", flush=True)

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
    if len(real) and len(plac):
        allx = np.concatenate([real, plac])
        ranks = allx.argsort().argsort().astype(float) + 1
        for v in np.unique(allx):
            idx = np.where(allx == v)[0]
            if len(idx) > 1:
                ranks[idx] = ranks[idx].mean()
        n1, n2 = len(real), len(plac)
        u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2
        sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        summ["real_vs_placebo"] = {"U": float(u1),
                                   "z": float((u1 - n1 * n2 / 2) / sd) if sd else 0.0}
    result["summary"] = summ
    out_path.write_text(json.dumps(result, indent=1))
    print(json.dumps(summ, indent=1), flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
