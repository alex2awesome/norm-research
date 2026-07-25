"""k-pass TEST re-measurement of specific shipped candidates (audit instrument, 2026-07-24).

The audit found the pupa/livebench GEPA-official headline numbers are single-pass draws under
heavy eval noise (livebench same-candidate replicate spreads up to .37 from AMPS-timeout/parse
zeros; pupa test-level same-prompt swing ~.07 from the GLM-judged metric). This re-measures the
shipped best_candidate of named run dirs with k independent test passes (v8 evaluate_cand
multi-pass; per-item elementwise means preserved for paired tests).

Usage: rescore_k3.py <bench> --lm-tag Qwen3-8B --task-lm openai/Qwen3-8B \
         --api-base http://127.0.0.1:PORT/v1 --runs official unitrecomb_v4local --passes 3
Appends runs_paperexact/<bench>/<lm-tag>/<run>/rescore_k3.jsonl (phase=rescore_k3).
"""
import argparse
import json

import dspy

import paperexact_arms as px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", choices=["aime", "hover", "hotpot", "ifbench", "livebench", "pupa"])
    ap.add_argument("--lm-tag", required=True)
    ap.add_argument("--task-lm", required=True)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--eval-threads", type=int, default=32)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()

    # cache=False: dspy's response cache made repeated passes return identical completions
    # (pass_means bit-identical, found 2026-07-25) -> k-pass averaging was silently k=1.
    lm = dspy.LM(a.task_lm, api_base=a.api_base, api_key="EMPTY", cache=False,
                 temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens,
                 num_retries=10, timeout=300, extra_body={"top_k": a.top_k})
    dspy.configure(lm=lm)
    px.EVAL_THREADS = a.eval_threads

    bench, program, metric, _ = px.load_bench(a.bench)
    test = list(bench.test_set)
    print(f"[{a.bench}] test={len(test)} passes={a.passes}", flush=True)
    for run in a.runs:
        rd = px.HERE / "runs_paperexact" / a.bench / a.lm_tag / run
        res = json.loads((rd / "result.json").read_text())
        cand = res["best_candidate"]
        s = px.evaluate_cand(program, cand, test, metric, rd / "rescore_k3.jsonl",
                             "rescore_k3", passes=a.passes)
        print(f"{run}: original best_test={res.get('best_test')}  "
              f"k{a.passes}_mean={s}", flush=True)


if __name__ == "__main__":
    main()
