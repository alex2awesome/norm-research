"""Arm B — the IN-HOUSE GEPA loop (faithful mirror of the norm-research predeclared recipe used in
v14 decoder tuning / silver norms: single-lineage greedy with batched GLM mutation rounds), on the
SAME datasets, seeds, evaluators, LMs, and budget accounting as Arm A.

Structure mirrored from methods/metric_implementer/experiments/v14_decoder_tuning.py
`tune_shared_template`: each round proposes (K-1) mutations of the incumbent from dev-set failure
feedback (GLM, temperature 0.7), dedupes, evaluates every candidate on a FROZEN dev panel, accepts
the best iff it beats the incumbent by `minimum_gain`, capped rounds. The deliberate algorithmic
difference vs official GEPA is candidate selection: in-house = beam-1 greedy single lineage;
official = Pareto frontier over instances. That difference is exactly what Phase 3 measures.

Budget accounting matches Arm A: every item evaluation (dev panel + final val) counts against
--max-metric-calls. Every candidate evaluation is appended to runs/<ds>/inhouse/proposals.jsonl.

  source .venv/bin/activate
  python run_inhouse_gepa.py hover --max-metric-calls 600
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from run_official_gepa import EVALS, GLM, SEEDS, _load

HERE = Path(__file__).parent
CANDIDATES_PER_ROUND = 8          # incumbent + 7 mutations (v14 frozen value)
MIN_GAIN = 0.01                   # v14 minimum_gain


def _norm(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()[:16]


def _mutation_prompt(incumbent: str, feedback_rows: list[dict], idx: int) -> str:
    fb = json.dumps(feedback_rows, ensure_ascii=False)[:12000]
    return f"""Improve a task instruction (system prompt) for a language model.
Mutation index: {idx}

INCUMBENT INSTRUCTION:
{incumbent}

DEVELOPMENT FEEDBACK (inputs, the model's outputs under the incumbent, and evaluator feedback):
{fb}

Use the failures to improve the instruction. Keep it general (no copying specific inputs or
answers). Return ONLY the rewritten instruction text."""


def evaluate_candidate(candidate: str, panel, task_lm, evaluator, log_path, budget_state,
                       tag: str):
    """Evaluate one candidate on a fixed panel; charge the budget; log the draw."""
    n = min(len(panel), budget_state["remaining"])
    if n <= 0:
        return None, []
    rows, scores = [], []
    for data in panel[:n]:
        resp = task_lm([{"role": "system", "content": candidate},
                        {"role": "user", "content": data["input"]}])
        res = evaluator(data, resp)
        scores.append(res.score)
        rows.append({"input": data["input"][:400], "output": resp[:400], "feedback": res.feedback})
    budget_state["remaining"] -= n
    mean = sum(scores) / len(scores)
    with open(log_path, "a") as fh:
        fh.write(json.dumps({"ts": time.time(), "candidate": {"system_prompt": candidate},
                             "n_batch": n, "mean_score": mean, "scores": scores,
                             "phase": tag}) + "\n")
    return mean, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(SEEDS))
    ap.add_argument("--max-metric-calls", type=int, default=600)
    ap.add_argument("--panel-n", type=int, default=25)
    ap.add_argument("--val-n", type=int, default=100)
    ap.add_argument("--task-model", default="glm-4.7")
    ap.add_argument("--reflection-model", default="glm-5")
    a = ap.parse_args()

    rundir = HERE / "runs" / a.dataset / "inhouse"
    rundir.mkdir(parents=True, exist_ok=True)
    (rundir / "seed.txt").write_text(SEEDS[a.dataset])
    log_path = rundir / "proposals.jsonl"

    task_lm = GLM(a.task_model, max_tokens=2048, temperature=0.2)
    refl_lm = GLM(a.reflection_model, max_tokens=1200, temperature=0.7)
    evaluator = EVALS[a.dataset]
    train = _load(a.dataset, "train", 150)
    panel = train[: a.panel_n]                                # frozen dev panel (v14 style)
    val = _load(a.dataset, "val", a.val_n)
    budget = {"remaining": a.max_metric_calls - len(val)}     # reserve the final val pass

    print(f"[{a.dataset}|inhouse] panel={len(panel)} val={len(val)} "
          f"budget={a.max_metric_calls}", flush=True)

    incumbent = SEEDS[a.dataset]
    inc_score, feedback = evaluate_candidate(incumbent, panel, task_lm, evaluator, log_path,
                                             budget, "round0_incumbent")
    seen = {_norm(incumbent)}
    trace = [{"round": 0, "score": inc_score, "accepted": True}]
    rnd = 0
    while budget["remaining"] >= len(panel):
        rnd += 1
        fails = [r for r in (feedback or []) if "Incorrect" in r["feedback"]][:6] or (feedback or [])[:4]
        prompts = [_mutation_prompt(incumbent, fails, i) for i in range(CANDIDATES_PER_ROUND - 1)]
        muts = []
        for p in prompts:
            try:
                muts.append(refl_lm(p).strip())
            except Exception as e:                            # noqa: BLE001
                print(f"  mutation call failed: {e}", flush=True)
        best_c, best_s, best_fb = None, -1.0, None
        for m in muts:
            if not m or _norm(m) in seen:
                continue
            seen.add(_norm(m))
            s, fb = evaluate_candidate(m, panel, task_lm, evaluator, log_path, budget,
                                       f"round{rnd}_mutation")
            if s is None:
                break
            if s > best_s:
                best_c, best_s, best_fb = m, s, fb
        accepted = best_c is not None and best_s >= (inc_score or 0) + MIN_GAIN
        trace.append({"round": rnd, "score": best_s, "accepted": bool(accepted)})
        print(f"  round {rnd}: best mutation {best_s:.3f} vs incumbent {inc_score:.3f} "
              f"-> {'ACCEPT' if accepted else 'reject'} (budget left {budget['remaining']})",
              flush=True)
        if accepted:
            incumbent, inc_score, feedback = best_c, best_s, best_fb

    budget["remaining"] += len(val)                           # spend the reserve
    val_score, _ = evaluate_candidate(incumbent, val, task_lm, evaluator, log_path, budget,
                                      "final_val")
    summary = {"dataset": a.dataset, "arm": "inhouse",
               "best_candidate": {"system_prompt": incumbent},
               "panel_score": inc_score, "val_score": val_score, "rounds": trace,
               "task_lm_calls": task_lm.n_calls, "reflection_lm_calls": refl_lm.n_calls}
    (rundir / "result.json").write_text(json.dumps(summary, indent=2))
    print(f"[{a.dataset}|inhouse] DONE val={val_score:.3f}; wrote {rundir/'result.json'}",
          flush=True)


if __name__ == "__main__":
    main()
