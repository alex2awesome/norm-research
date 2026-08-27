"""Arm C — UNIT-LEVEL RECOMBINATION (the M_ω-style construct) vs the monolithic GEPA prompt.

The M_ω analog for a task-prompt setting:
  1. UNIT MINING: every distinct prompt proposed by BOTH optimizer arms is split into
     instruction-clause UNITS (sentence-level), normalized and deduped — the unit pool is the
     trajectory's articulable content, not any single winner.
  2. BEHAVIORAL SCREEN: each unit is appended alone to the frozen seed prompt and evaluated on a
     small screen panel — the per-unit standalone-value read.
  3. GREEDY RECOMBINATION: from the top screened units, greedily add units to the compiled prompt
     while a disjoint dev panel improves (the conditional-value / orthogonalization analog);
     redundant or harmful units are never compiled in.
  4. EVAL: the compiled prompt on the same val split with the same evaluator as Arms A/B.

Budget accounting matches the other arms (--max-metric-calls, val reserve included). Every
evaluation is logged to runs/<ds>/unitrecomb/proposals.jsonl (same schema; phase field tags
screen/greedy/final).

  source .venv/bin/activate && python run_unit_recombination.py hover --max-metric-calls 600
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_official_gepa import EVALS, GLM, SEEDS, _load

HERE = Path(__file__).parent
MIN_GAIN = 0.01
MAX_UNITS = 40


def _norm(t: str) -> str:
    return hashlib.sha256(" ".join(t.lower().split()).encode()).hexdigest()[:16]


def mine_units(ds: str) -> list[str]:
    """Clause-level units from every distinct prompt in both arms' proposal logs."""
    texts, seen = [], set()
    for arm in ("official", "inhouse"):
        p = HERE / "runs" / ds / arm / "proposals.jsonl"
        if not p.exists():
            continue
        for line in open(p):
            t = next(iter(json.loads(line)["candidate"].values()))
            if _norm(t) not in seen:
                seen.add(_norm(t))
                texts.append(t)
    units, useen = [], set()
    for t in texts:
        for clause in re.split(r"(?<=[.!?])\s+|\n+|^[-*•]\s*", t):
            c = clause.strip().strip("-*• ").rstrip()
            if len(c) < 25 or len(c) > 400:
                continue
            h = _norm(c)
            if h not in useen:
                useen.add(h)
                units.append(c)
    if len(units) > MAX_UNITS:
        print(f"  unit pool {len(units)} capped at {MAX_UNITS} (draw order kept, rest DROPPED)")
        units = units[:MAX_UNITS]
    return units


def evaluate_prompt(prompt, panel, task_lm, evaluator, log_path, budget, tag):
    n = min(len(panel), budget["remaining"])
    if n <= 0:
        return None
    def _one(data):
        resp = task_lm([{"role": "system", "content": prompt},
                        {"role": "user", "content": data["input"]}])
        return evaluator(data, resp).score

    with ThreadPoolExecutor(max_workers=6) as ex:
        scores = list(ex.map(_one, panel[:n]))
    budget["remaining"] -= n
    mean = sum(scores) / len(scores)
    with open(log_path, "a") as fh:
        fh.write(json.dumps({"ts": time.time(), "candidate": {"system_prompt": prompt},
                             "n_batch": n, "mean_score": mean, "scores": scores,
                             "phase": tag}) + "\n")
    return mean


def compile_prompt(seed: str, units: list[str]) -> str:
    if not units:
        return seed
    return seed + "\n\nAdditional requirements:\n" + "\n".join(f"- {u}" for u in units)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(SEEDS))
    ap.add_argument("--max-metric-calls", type=int, default=600)
    ap.add_argument("--screen-n", type=int, default=8)
    ap.add_argument("--panel-n", type=int, default=15)
    ap.add_argument("--val-n", type=int, default=100)
    ap.add_argument("--task-model", default="glm-4.7")
    a = ap.parse_args()

    rundir = HERE / "runs" / a.dataset / "unitrecomb"
    rundir.mkdir(parents=True, exist_ok=True)
    log_path = rundir / "proposals.jsonl"
    task_lm = GLM(a.task_model, max_tokens=2048, temperature=0.2)
    evaluator = EVALS[a.dataset]
    seed = SEEDS[a.dataset]
    train = _load(a.dataset, "train", 150)
    screen_panel = train[: a.screen_n]
    dev_panel = train[a.screen_n: a.screen_n + a.panel_n] or train[: a.panel_n]
    val = _load(a.dataset, "val", a.val_n)
    budget = {"remaining": a.max_metric_calls - len(val)}

    units = mine_units(a.dataset)
    # BUDGET SPLIT: screening must leave the greedy stage a base eval + one full 12-unit round
    # + a partial second round, or recombination never runs and Arm C degenerates to the seed.
    greedy_reserve = len(dev_panel) * (1 + 12 + 4)
    n_screenable = max(0, (budget["remaining"] - greedy_reserve) // len(screen_panel))
    if len(units) > n_screenable:
        print(f"  unit pool {len(units)} -> screening first {n_screenable} "
              f"(greedy reserve {greedy_reserve} calls)", flush=True)
        units = units[:n_screenable]
    print(f"[{a.dataset}|unitrecomb] {len(units)} units; screen={len(screen_panel)} "
          f"panel={len(dev_panel)} val={len(val)} budget={a.max_metric_calls}", flush=True)

    # 2. behavioral screen (standalone value per unit)
    screened = []
    for u in units:
        s = evaluate_prompt(compile_prompt(seed, [u]), screen_panel, task_lm, evaluator,
                            log_path, budget, "screen")
        if s is None:
            print("  budget exhausted during screen", flush=True)
            break
        screened.append((s, u))
    screened.sort(key=lambda x: -x[0])
    top = [u for _, u in screened[:12]]

    # 3. greedy recombination on the disjoint dev panel (conditional value)
    base = evaluate_prompt(seed, dev_panel, task_lm, evaluator, log_path, budget, "greedy_base")
    chosen: list[str] = []
    cur = base or 0.0
    improved = True
    while improved and budget["remaining"] >= len(dev_panel) and top:
        improved = False
        best_u, best_s = None, cur
        for u in list(top):
            if budget["remaining"] < len(dev_panel):
                break
            s = evaluate_prompt(compile_prompt(seed, chosen + [u]), dev_panel, task_lm,
                                evaluator, log_path, budget, f"greedy_k{len(chosen)+1}")
            if s is not None and s >= best_s + MIN_GAIN:
                best_u, best_s = u, s
        if best_u is not None:
            chosen.append(best_u)
            top.remove(best_u)
            cur = best_s
            improved = True
            print(f"  +unit ({len(chosen)}): panel {cur:.3f}", flush=True)

    final_prompt = compile_prompt(seed, chosen)
    budget["remaining"] += len(val)
    val_score = evaluate_prompt(final_prompt, val, task_lm, evaluator, log_path, budget, "final_val")
    summary = {"dataset": a.dataset, "arm": "unitrecomb",
               "n_units_mined": len(units), "n_units_screened": len(screened),
               "n_units_compiled": len(chosen), "panel_base": base, "panel_final": cur,
               "best_candidate": {"system_prompt": final_prompt},
               "val_score": val_score, "task_lm_calls": task_lm.n_calls,
               "unit_screen_top12": [{"screen_score": s, "unit": u} for s, u in screened[:12]],
               "units_compiled": chosen}
    (rundir / "result.json").write_text(json.dumps(summary, indent=2))
    print(f"[{a.dataset}|unitrecomb] DONE val={val_score:.3f} ({len(chosen)} units compiled, "
          f"panel {base if base is not None else float('nan'):.3f}->{cur:.3f})", flush=True)


if __name__ == "__main__":
    main()
