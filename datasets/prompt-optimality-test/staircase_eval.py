"""OSL staircase — cheap arms: evaluate FIXED candidates at each model scale.

Arms (design v2, 2026-07-22): seed / GEPA-8B-winner transplant / M_omega-8B-winner transplant.
Each is a fixed instruction set evaluated on the paper test split with the scale-s task LM —
2-3 evals per (bench, scale), no search. The expensive per-scale arms (M_omega frozen-pool
selection, GEPA re-optimization) run via paperexact_arms.py --pool-file / --arm official with
--task-lm pointed at the scale server and --run-tag stair_<model>.

Usage:
  python3 staircase_eval.py <bench> --model Qwen3-1.7B --api-base http://127.0.0.1:8171/v1 \
      [--top-k 20] [--eval-threads 32]
Writes runs_staircase/<bench>/<model>/fixed_arms.json (+ item_scores per arm for sign tests).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent


def main():
    import paperexact_arms as px
    import dspy

    ap = argparse.ArgumentParser()
    ap.add_argument("bench", choices=["aime", "hover", "hotpot", "ifbench", "livebench",
                                      "pupa"])
    ap.add_argument("--model", required=True, help="served model name, e.g. Qwen3-1.7B")
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--eval-threads", type=int, default=32)
    a = ap.parse_args()

    px.EVAL_THREADS = a.eval_threads
    # Pre-flight + mid-run outage guard (2026-07-22 sk1 lesson: 6 eval runs launched against
    # argparse-dead servers; without this, dead-endpoint evals record all-zero artifacts).
    import urllib.request

    def _probe():
        try:
            urllib.request.urlopen(a.api_base.rstrip("/") + "/models", timeout=15)
            return True
        except Exception:
            return False
    if not _probe():
        raise SystemExit(f"endpoint {a.api_base} DEAD — refusing to run (would zero-score)")
    px.HEALTH_PROBE = _probe
    lm = dspy.LM(f"openai/{a.model}", api_base=a.api_base, api_key="EMPTY",
                 temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens,
                 num_retries=10, timeout=300, extra_body={"top_k": a.top_k})
    dspy.configure(lm=lm)

    bench, program, metric, _ = px.load_bench(a.bench)
    test = list(bench.test_set)

    lm8b = HERE / "runs_paperexact" / a.bench / "Qwen3-8B"
    arms = {"seed": px.get_instructions(program)}
    off = lm8b / "official" / "result.json"
    if off.exists():
        arms["gepa8b_transplant"] = {**arms["seed"],
                                     **json.loads(off.read_text())["best_candidate"]}
    # best M_omega variant by best_test (skip INVALID_*)
    best_v, best_score = None, -1
    for rdir in lm8b.glob("unitrecomb*"):
        res = rdir / "result.json"
        if res.exists():
            r = json.loads(res.read_text())
            if (r.get("best_test") or -1) > best_score:
                best_v, best_score = r, r["best_test"]
    if best_v:
        arms["momega8b_transplant"] = {**arms["seed"], **best_v["best_candidate"]}

    outdir = HERE / "runs_staircase" / a.bench / a.model
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "evals.jsonl"
    out = {"bench": a.bench, "model": a.model, "n_test": len(test),
           "top_k": a.top_k, "max_tokens": a.max_tokens, "arms": {}}
    for name, cand in arms.items():
        s = px.evaluate_cand(program, cand, test, metric, log_path, f"stair_{name}")
        out["arms"][name] = s
        print(f"[{a.bench}|{a.model}] {name}: {s}", flush=True)
    (outdir / "fixed_arms.json").write_text(json.dumps(out, indent=1))
    print(f"[{a.bench}|{a.model}] DONE -> {outdir}/fixed_arms.json", flush=True)


if __name__ == "__main__":
    main()
