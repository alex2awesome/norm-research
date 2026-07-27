"""E5: OSL articulation battery — one model rung (green-lit 2026-07-27; user: "run some of
Tatsu's 2024 OSL experiments" = observational scaling, existing models as scale points).

Per (model, bench), all in ONE session at 24k:
  1. init score           — the GEPA-official 8B candidate, held FIXED across scales;
  2. N random draws       — each frozen-pool unit included w.p. p, appended to init
                            -> pool-value distribution at this scale + per-unit masks
                            (draw-level unit-value regression comes free downstream);
  3. transfer arm         — the 8B winner p*_8B verbatim -> cross-scale prompt isomorphism.

Downstream readouts (analysis, not here): absorption vs capability; relational-tacit onset
(units valuable only above a capability threshold); thin-vs-thick scaling; ceiling-predicts-
scaling (B-hat fitted at small scales vs realized best above).

Usage: osl_battery.py hotpot --lm-out Qwen3-1.7B --task-lm openai/Qwen3-1.7B \
         --api-base http://127.0.0.1:8181/v1
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
    ap.add_argument("bench")
    ap.add_argument("--lm-out", required=True, help="tag for output naming, e.g. Qwen3-1.7B")
    ap.add_argument("--task-lm", required=True)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--pool-tag", default="Qwen3-8B", help="frozen pool + init/transfer source tag")
    ap.add_argument("--transfer-run", default=None,
                    help="run dir whose best_candidate is the transfer arm (default: bench-specific)")
    ap.add_argument("--n-draws", type=int, default=40)
    ap.add_argument("--include-p", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--eval-threads", type=int, default=32)
    a = ap.parse_args()

    transfer_defaults = {"hotpot": "unitrecomb_v5sk2", "hover": "unitrecomb_stair"}
    transfer_run = a.transfer_run or transfer_defaults.get(a.bench, "unitrecomb")

    lm = dspy.LM(a.task_lm, api_base=a.api_base, api_key="EMPTY", cache=False,
                 temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens,
                 num_retries=10, timeout=300, extra_body={"top_k": a.top_k})
    dspy.configure(lm=lm)
    px.EVAL_THREADS = a.eval_threads

    bench, program, metric, _ = px.load_bench(a.bench)
    test = list(bench.test_set)
    seed_cand = px.get_instructions(program)
    bdir = HERE / "runs_paperexact" / a.bench / a.pool_tag
    init_cand = {**seed_cand, **json.loads((bdir / "official" / "result.json").read_text())["best_candidate"]}
    transfer_cand = {**seed_cand,
                     **json.loads((bdir / transfer_run / "result.json").read_text())["best_candidate"]}
    pool = [(d["module"], d["unit"]) for d in
            json.loads((HERE / "pools" / f"{a.bench}_{a.pool_tag}_frozen.json").read_text())["units"]
            if d["module"] in seed_cand]

    out_path = HERE / "runs" / f"osl_{a.bench}_{a.lm_out}.json"
    out_path.parent.mkdir(exist_ok=True)
    prev = json.loads(out_path.read_text()) if out_path.exists() else {}
    log = HERE / "runs_paperexact" / a.bench / a.pool_tag / f"osl_{a.lm_out}" / "evals.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)

    fp = {"host": socket.gethostname(), "api_base": a.api_base, "task_lm": a.task_lm,
          "max_tokens": a.max_tokens, "cache": False,
          "utc": datetime.datetime.utcnow().isoformat() + "Z"}
    print(f"[{a.bench}|{a.lm_out}] pool={len(pool)} test={len(test)}", flush=True)
    print("FINGERPRINT:", json.dumps(fp), flush=True)
    result = {"bench": a.bench, "model": a.lm_out, "fingerprint": fp,
              "include_p": a.include_p, "seed": a.seed, "transfer_run": transfer_run}
    result.update({k: prev[k] for k in ("init", "transfer", "draws") if k in prev})

    if "init" not in result:
        result["init"] = px.evaluate_cand(program, init_cand, test, metric, log, "osl_init")
        print("  init =", result["init"], flush=True)
        out_path.write_text(json.dumps(result, indent=1))
    if "transfer" not in result:
        result["transfer"] = px.evaluate_cand(program, transfer_cand, test, metric, log, "osl_transfer")
        print("  transfer(p*_8B) =", result["transfer"], flush=True)
        out_path.write_text(json.dumps(result, indent=1))

    rng = np.random.default_rng(a.seed)
    draws = result.get("draws", [])
    for i in range(a.n_draws):
        mask = (rng.random(len(pool)) < a.include_p)   # regenerate the SAME sequence on resume
        if i < len(draws):
            continue
        cand = dict(init_cand)
        for (mod, clause), m in zip(pool, mask):
            if m:
                cand[mod] = cand[mod] + "\n- " + clause
        s = px.evaluate_cand(program, cand, test, metric, log, f"osl_draw_{i}")
        draws.append({"i": i, "mask": mask.tolist(), "score": s})
        result["draws"] = draws
        out_path.write_text(json.dumps(result, indent=1))
        if (i + 1) % 10 == 0:
            ok = [d["score"] for d in draws if d["score"] is not None]
            print(f"  draw {i+1}/{a.n_draws} mean={sum(ok)/len(ok):.4f}", flush=True)
    ok = [d["score"] for d in draws if d["score"] is not None]
    print(f"DONE [{a.bench}|{a.lm_out}] init={result['init']} transfer={result['transfer']} "
          f"draws mean={sum(ok)/len(ok):.4f} max={max(ok):.4f} n={len(ok)}", flush=True)


if __name__ == "__main__":
    main()
