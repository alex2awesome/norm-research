"""M_ω v2 — unit recombination redesigned around the UNIT_BOTTLENECK_DIAGNOSIS.md evidence.

v1 compiled 1 unit on every dataset and went BACKWARD on hover (val .73 < seed .76). The
diagnosis showed why, and what the fixes are:

  CAUSE (dominant)   greedy_reserve = panel*(1+12+4) deterministically capped compiles at ~1;
                     screening burned ~half the budget to produce a near-random shortlist.
  CAUSE (strong)     15-item selection panel: sd of a panel mean ≈ 1.8 items vs a 0.15-item
                     acceptance bar ⇒ the accepted unit was a coin flip (hover's was a false
                     positive: +3 items on panel, −0.03 on val).
  RULED OUT          unit redundancy (top-12 token-Jaccard .04-.12, 10-12 distinct clusters);
                     "GLM saturation" (16-19 of ~30 units beat the seed on val);
                     a paired sign-test acceptance rule (0 units reach p<.1 — the 15-item
                     panel is structurally underpowered; PANEL SIZE is the lever, not the test).

v2 therefore changes GEOMETRY, not the acceptance statistic:
  1. INIT FROM GEPA-BEST (D2). The compile starts at the candidate official GEPA actually
     shipped, so M_ω is a superset of GEPA's answer and only has to find marginal additions.
     (On hover/hotpotqa GEPA shipped candidate idx 0 = the seed — it found no improvement.)
  2. NO cheap screen. The 60-item PAIRED marginal evaluation IS the selection step; every unit
     is measured against the same init on the same items (paired ⇒ lower variance).
  3. Budget goes to selection, not to a pre-filter. Declared, not implicit.
  4. Cumulative-prefix sweep + drop-one pass instead of many-round greedy (handles redundancy
     at a fraction of the cost).
  5. NO-REGRET GUARD on a disjoint confirmation slice: if the compiled prompt does not beat
     init there, we ship init. So M_ω ≥ GEPA-shipped by construction, up to confirmation noise.
  6. Units = official-GEPA trajectory + in-house trajectory + LLM-SUGGESTED units (D2).
     (The third D2 source — explicit children metrics, e.g. R1 children when judging R2 — has
     no analog in this benchmark setting; it applies to the reconstruction side only.)

SPLIT HYGIENE (the reason v1's numbers were not trustworthy): every arm in phases 1-4 selected
on val[:100], so val[:100] is contaminated for reporting. v2 selects on TRAIN only and reports
on val[100:300], which no arm has ever touched, re-scoring GEPA-shipped and the raw seed on
those same items so the comparison is apples-to-apples.

  python run_momega_v2.py hover --select-n 60 --confirm-n 40 --test-lo 100 --test-hi 300
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
N_LLM_UNITS = 12
TOP_K = 8


def _norm(t: str) -> str:
    return hashlib.sha256(" ".join(t.lower().split()).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ unit pool
def mine_trajectory_units(ds: str) -> list[str]:
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
            if _norm(c) not in useen:
                useen.add(_norm(c))
                units.append(c)
    return units


def suggest_llm_units(ds: str, init_prompt: str, failures: list[dict], proposer: GLM,
                      n: int = N_LLM_UNITS) -> list[str]:
    """D2 third source: novel units proposed by a strong LLM from observed failures."""
    fail_txt = "\n\n".join(
        f"INPUT:\n{f['input'][:700]}\nMODEL SAID: {f['resp'][:200]}\nCORRECT: {f['gold']}"
        for f in failures[:4]) or "(no failures captured)"
    prompt = (
        f"A system prompt for the task '{ds}' is being improved by adding short, standalone "
        f"instruction clauses.\n\nCURRENT PROMPT:\n{init_prompt}\n\nCASES IT GETS WRONG:\n"
        f"{fail_txt}\n\nPropose {n} DISTINCT candidate instruction clauses that might each "
        "independently improve accuracy. Each must be one or two sentences, self-contained, "
        "generally applicable (never referencing a specific example above), and actionable. "
        "Reply with ONLY a JSON array of strings.")
    raw = proposer([{"role": "user", "content": prompt}])
    try:
        arr = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
        return [str(x).strip() for x in arr if 25 <= len(str(x).strip()) <= 400][:n]
    except Exception as exc:
        print(f"  LLM unit suggestion failed ({exc}); continuing with trajectory units only")
        return []


def compile_prompt(init: str, units: list[str]) -> str:
    if not units:
        return init
    return init + "\n\nAdditional requirements:\n" + "\n".join(f"- {u}" for u in units)


# ------------------------------------------------------------------ evaluation
def evaluate(prompt: str, panel: list[dict], task_lm: GLM, evaluator, log_path: Path,
             tag: str, budget: dict, capture_failures: bool = False):
    if budget["remaining"] < len(panel):
        return None, None, []
    budget["remaining"] -= len(panel)

    def _one(d):
        resp = task_lm([{"role": "system", "content": prompt},
                        {"role": "user", "content": d["input"]}])
        return evaluator(d, resp).score, resp

    with ThreadPoolExecutor(max_workers=8) as ex:
        out = list(ex.map(_one, panel))
    scores = [float(s) for s, _ in out]
    fails = ([{"input": d["input"], "resp": r, "gold": str(d.get("gold", d.get("answer", "")))}
              for d, (s, r) in zip(panel, out) if s < 1.0] if capture_failures else [])
    mean = sum(scores) / len(scores)
    with open(log_path, "a") as fh:
        fh.write(json.dumps({"ts": time.time(), "candidate": {"system_prompt": prompt},
                             "hash": _norm(prompt), "n_batch": len(panel),
                             "mean_score": mean, "item_scores": scores, "phase": tag}) + "\n")
    return mean, scores, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(SEEDS))
    ap.add_argument("--select-n", type=int, default=60)
    ap.add_argument("--confirm-n", type=int, default=40)
    ap.add_argument("--test-lo", type=int, default=100)
    ap.add_argument("--test-hi", type=int, default=300)
    ap.add_argument("--max-units", type=int, default=40)
    ap.add_argument("--task-model", default="glm-4.7")
    ap.add_argument("--proposer-model", default="glm-5.2")
    ap.add_argument("--budget-calls", type=int, default=6000)
    a = ap.parse_args()

    rundir = HERE / "runs" / a.dataset / "momega_v2"
    rundir.mkdir(parents=True, exist_ok=True)
    log_path = rundir / "proposals.jsonl"
    task_lm = GLM(a.task_model, max_tokens=2048, temperature=0.2)
    proposer = GLM(a.proposer_model, max_tokens=4096, temperature=1.0)
    evaluator = EVALS[a.dataset]
    budget = {"remaining": a.budget_calls}

    # ---- splits: select/confirm from TRAIN; test = val[lo:hi], untouched by every prior arm
    train = _load(a.dataset, "train", 150)
    select_panel = train[: a.select_n]
    confirm_panel = train[a.select_n: a.select_n + a.confirm_n]
    val_all = _load(a.dataset, "val", a.test_hi)
    test = val_all[a.test_lo: a.test_hi]
    if len(test) < 50:
        raise SystemExit(f"test slice too small ({len(test)}) — need an untouched val tail")

    seed_prompt = SEEDS[a.dataset]
    init = next(iter(json.loads(
        (HERE / "runs" / a.dataset / "official" / "result.json").read_text()
    )["best_candidate"].values()))
    print(f"[{a.dataset}|momega_v2] select={len(select_panel)} confirm={len(confirm_panel)} "
          f"test={len(test)} (val[{a.test_lo}:{a.test_hi}], untouched) budget={a.budget_calls}",
          flush=True)
    print(f"  init = official-GEPA shipped candidate ({len(init)} chars); "
          f"init==seed: {_norm(init) == _norm(seed_prompt)}", flush=True)

    # ---- 1. init baseline on the selection panel (paired reference), capture failures
    base_mean, base_vec, failures = evaluate(init, select_panel, task_lm, evaluator, log_path,
                                             "select_init", budget, capture_failures=True)
    print(f"  init select score {base_mean:.3f} ({len(failures)} failures captured)", flush=True)

    # ---- 2. unit pool: trajectories + LLM suggestions
    traj_units = mine_trajectory_units(a.dataset)
    llm_units = suggest_llm_units(a.dataset, init, failures, proposer)
    units, useen = [], set()
    for u in llm_units + traj_units:            # LLM units first (novel content priority)
        if _norm(u) not in useen:
            useen.add(_norm(u))
            units.append(u)
    units = units[: a.max_units]
    print(f"  units: {len(llm_units)} LLM-suggested + {len(traj_units)} mined "
          f"-> {len(units)} used (cap {a.max_units})", flush=True)

    # ---- 3. paired marginal value of every unit on the SAME selection panel
    marginals = []
    for u in units:
        m, vec, _ = evaluate(compile_prompt(init, [u]), select_panel, task_lm, evaluator,
                             log_path, "select_marginal", budget)
        if m is None:
            print("  budget exhausted during marginal pass", flush=True)
            break
        marginals.append({"unit": u, "mean": m, "delta": m - base_mean,
                          "wins": sum(1 for x, y in zip(vec, base_vec) if x > y),
                          "losses": sum(1 for x, y in zip(vec, base_vec) if x < y)})
    marginals.sort(key=lambda r: -r["delta"])
    positives = [r for r in marginals if r["delta"] > 0]
    print(f"  {len(positives)}/{len(marginals)} units have positive paired marginal; "
          f"top delta {marginals[0]['delta']:+.3f}" if marginals else "  no marginals",
          flush=True)

    # ---- 4. cumulative-prefix sweep over the top-K, then a drop-one pass
    ordered = [r["unit"] for r in positives[:TOP_K]]
    best_set, best_score = [], base_mean
    cum = []
    for u in ordered:
        cum = cum + [u]
        m, _, _ = evaluate(compile_prompt(init, cum), select_panel, task_lm, evaluator,
                           log_path, f"prefix_k{len(cum)}", budget)
        if m is None:
            break
        print(f"    prefix k={len(cum)}: {m:.3f}", flush=True)
        if m > best_score:
            best_set, best_score = list(cum), m
    for u in list(best_set):                     # drop-one: remove units that don't earn a slot
        trial = [x for x in best_set if x != u]
        m, _, _ = evaluate(compile_prompt(init, trial), select_panel, task_lm, evaluator,
                           log_path, "drop_one", budget)
        if m is not None and m >= best_score:
            best_set, best_score = trial, m
            print(f"    dropped a unit -> {m:.3f}", flush=True)

    compiled = compile_prompt(init, best_set)
    print(f"  compiled {len(best_set)} units; select {base_mean:.3f} -> {best_score:.3f}",
          flush=True)

    # ---- 5. NO-REGRET GUARD on the disjoint confirmation slice
    c_init, _, _ = evaluate(init, confirm_panel, task_lm, evaluator, log_path,
                            "confirm_init", budget)
    c_comp, _, _ = evaluate(compiled, confirm_panel, task_lm, evaluator, log_path,
                            "confirm_compiled", budget)
    shipped, fell_back = compiled, False
    if c_comp is None or c_init is None or c_comp < c_init:
        shipped, fell_back = init, True
        print(f"  NO-REGRET GUARD: confirm {c_comp} < init {c_init} -> shipping GEPA init",
              flush=True)
    else:
        print(f"  confirm: init {c_init:.3f} -> compiled {c_comp:.3f} (kept)", flush=True)

    # ---- 6. report on the untouched test slice; rescore GEPA-shipped and raw seed there too
    t_momega, v_momega, _ = evaluate(shipped, test, task_lm, evaluator, log_path,
                                     "test_momega", budget)
    t_gepa, v_gepa, _ = evaluate(init, test, task_lm, evaluator, log_path, "test_gepa", budget)
    t_seed = t_gepa if _norm(init) == _norm(seed_prompt) else evaluate(
        seed_prompt, test, task_lm, evaluator, log_path, "test_seed", budget)[0]

    result = {
        "dataset": a.dataset, "arm": "momega_v2",
        "design": {"select_n": len(select_panel), "confirm_n": len(confirm_panel),
                   "test_slice": [a.test_lo, a.test_hi], "test_n": len(test),
                   "init": "official-GEPA shipped best_candidate",
                   "init_is_seed": _norm(init) == _norm(seed_prompt),
                   "no_regret_guard": True, "fell_back_to_init": fell_back},
        "units": {"n_llm_suggested": len(llm_units), "n_mined": len(traj_units),
                  "n_used": len(units), "n_compiled": len(best_set),
                  "compiled_units": best_set,
                  "marginals_top10": marginals[:10]},
        "select": {"init": base_mean, "compiled": best_score},
        "confirm": {"init": c_init, "compiled": c_comp},
        "test_untouched": {"momega_v2": t_momega, "gepa_shipped": t_gepa, "seed": t_seed},
        "test_item_scores": {"momega_v2": v_momega, "gepa_shipped": v_gepa},
        "task_lm_calls": task_lm.n_calls, "budget_used": a.budget_calls - budget["remaining"],
    }
    (rundir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[{a.dataset}|momega_v2] DONE test(untouched n={len(test)}): "
          f"M_ω {t_momega:.3f} vs GEPA-shipped {t_gepa:.3f} vs seed {t_seed:.3f}", flush=True)


if __name__ == "__main__":
    main()
