"""LIVEBENCH CONTENT RESCUE (user directive 2026-07-27: 'get them both to a win').

MOTIVATION (HB117): the P0 draw-regression showed the livebench pool is a MIXTURE — ~20 units
with positive draw-level marginals (top ones are genuine math content, +.03..+.07) diluted by
~6 toxic units (worst −.06). Random p=.5 draws average the mixture down to placebo level; the
content claim died on the MIXTURE, not necessarily on the content. Hypothesis: SELECTED content
beats bulk.

DESIGN — two stages, one process, selection never touches test (the draw regression was
test-derived, so it serves as hypothesis only; selection is re-derived cleanly here):
  STAGE 1 (select panel, train[:81], k=2 cache-off): score init and init+unit_i for all 48
  units -> per-unit select-panel marginals. Deterministic rule, pre-registered: keep the top-12
  units by marginal (ties by index).
  STAGE 2 (test, one session): init replicates x8; SELECTED candidate x8 replicate evals;
  foreign-content control x15 (count-matched: 12 hover clauses per draw). Cache off everywhere.

PRE-REGISTERED VERDICT (HB118, committed before launch):
  CONTENT RESTORED  iff selected-mean > placebo-mean with rank-sum p < .05.
  FAIL              otherwise -> livebench's content claim stays dead; NO further attempts.
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
    _s, _h = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _h), _h))
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-lm", required=True)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--top-k-units", type=int, default=12)
    ap.add_argument("--select-n", type=int, default=81)
    ap.add_argument("--stage1-passes", type=int, default=2)
    ap.add_argument("--n-init", type=int, default=8)
    ap.add_argument("--n-selected", type=int, default=8)
    ap.add_argument("--n-placebo", type=int, default=15)
    ap.add_argument("--placebo-bench", default="hover")
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--eval-threads", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    lm = dspy.LM(a.task_lm, api_base=a.api_base, api_key="EMPTY", cache=False,
                 temperature=0.6, top_p=0.95, max_tokens=a.max_tokens,
                 num_retries=10, timeout=300, extra_body={"top_k": 20})
    dspy.configure(lm=lm)
    px.EVAL_THREADS = a.eval_threads

    bench, program, metric, _ = px.load_bench("livebench")
    seed_cand = px.get_instructions(program)
    init_cand = {**seed_cand, **json.loads(
        (HERE / "runs_paperexact/livebench/Qwen3-8B/official/result.json").read_text())["best_candidate"]}
    pool = [(d["module"], d["unit"]) for d in
            json.loads((HERE / "pools/livebench_Qwen3-8B_frozen.json").read_text())["units"]
            if d["module"] in seed_cand]
    placebo_texts = [d["unit"] for d in
                     json.loads((HERE / f"pools/{a.placebo_bench}_Qwen3-8B_frozen.json").read_text())["units"]]

    select_panel = list(bench.train_set)[:a.select_n]
    test = list(bench.test_set)
    log = HERE / "runs_paperexact/livebench/Qwen3-8B/content_rescue/evals.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    out_path = HERE / "runs/content_rescue_livebench.json"
    prev = json.loads(out_path.read_text()) if out_path.exists() else {}
    fp = {"host": socket.gethostname(), "api_base": a.api_base, "cache": False,
          "max_tokens": a.max_tokens, "utc": datetime.datetime.utcnow().isoformat() + "Z"}
    print("FINGERPRINT:", json.dumps(fp), flush=True)
    result = {"fingerprint": fp, "rule": f"top-{a.top_k_units} units by stage-1 select marginal"}

    # ---------------- STAGE 1: select-panel unit marginals ----------------
    if prev.get("stage1"):
        result["stage1"] = prev["stage1"]
        print("stage 1 resumed from checkpoint", flush=True)
    else:
        base = px.evaluate_cand(program, init_cand, select_panel, metric, log,
                                "s1_init", passes=a.stage1_passes)
        print(f"stage1 init = {base}", flush=True)
        margs = []
        for i, (mod, clause) in enumerate(pool):
            c = dict(init_cand)
            c[mod] = c[mod] + "\n- " + clause
            s = px.evaluate_cand(program, c, select_panel, metric, log,
                                 f"s1_unit_{i}", passes=a.stage1_passes)
            margs.append({"i": i, "unit": clause, "module": mod,
                          "marginal": (s - base) if s is not None else None})
            result["stage1"] = {"init": base, "marginals": margs}
            out_path.write_text(json.dumps(result, indent=1))
        print("stage 1 complete", flush=True)

    margs = [m for m in result["stage1"]["marginals"] if m["marginal"] is not None]
    margs.sort(key=lambda m: (-m["marginal"], m["i"]))
    top = margs[:a.top_k_units]
    result["selected_units"] = top
    sel_cand = dict(init_cand)
    for m in top:
        sel_cand[m["module"]] = sel_cand[m["module"]] + "\n- " + m["unit"]
    print("selected units:", [round(m["marginal"], 4) for m in top], flush=True)
    out_path.write_text(json.dumps(result, indent=1))

    # ---------------- STAGE 2: one-shot test session ----------------
    rng = np.random.default_rng(a.seed)
    jobs = ([("init", j, None) for j in range(a.n_init)] +
            [("selected", j, None) for j in range(a.n_selected)] +
            [("placebo", j, rng.choice(len(placebo_texts), size=a.top_k_units, replace=False).tolist())
             for j in range(a.n_placebo)])
    order = rng.permutation(len(jobs))
    done = {r["order_pos"]: r for r in prev.get("stage2", [])}
    recs = [done[k] for k in sorted(done)]
    for pos, ji in enumerate(order):
        if pos in done:
            continue
        arm, j, picks = jobs[int(ji)]
        if arm == "init":
            c = dict(init_cand)
        elif arm == "selected":
            c = sel_cand
        else:
            c = dict(init_cand)
            for k, x in enumerate(picks):
                mod = pool[k % len(pool)][0]
                c[mod] = c[mod] + "\n- " + placebo_texts[int(x)]
        s = px.evaluate_cand(program, c, test, metric, log, f"s2_{arm}_{j}")
        recs.append({"order_pos": pos, "arm": arm, "idx": j, "score": s})
        result["stage2"] = recs
        out_path.write_text(json.dumps(result, indent=1))
        if (pos + 1) % 6 == 0:
            print(f"  stage2 {pos+1}/{len(jobs)}", flush=True)

    def arr(name):
        return np.array([r["score"] for r in recs if r["arm"] == name and r["score"] is not None])
    ini, sel, plc = arr("init"), arr("selected"), arr("placebo")
    def st(x):
        return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std()),
                "min": float(x.min()), "max": float(x.max())} if len(x) else {}
    summ = {"init": st(ini), "selected": st(sel), "placebo": st(plc)}
    if len(sel) and len(plc):
        allx = np.concatenate([sel, plc])
        ranks = allx.argsort().argsort().astype(float) + 1
        for v in np.unique(allx):
            idx = np.where(allx == v)[0]
            if len(idx) > 1:
                ranks[idx] = ranks[idx].mean()
        n1, n2 = len(sel), len(plc)
        u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2
        sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        summ["selected_vs_placebo_z"] = float((u1 - n1 * n2 / 2) / sd) if sd else 0.0
    result["summary"] = summ
    out_path.write_text(json.dumps(result, indent=1))
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
