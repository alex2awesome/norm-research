"""E2+E3: per-unit CAUSAL ablation battery + minimality probe (green-lit 2026-07-27).

E2: for the top-N chosen units of a winning candidate, measure candidate-minus-unit in ONE
session (paired, k passes, cache off) -> per-unit causal deltas replacing the correlational
marginals. E3: for the top few units, also test each natural sub-clause alone in the unit's
place -> units are 'minimal' iff sub-clauses lose the effect.

Usage: ablation_battery.py hotpot --run unitrecomb_v5sk2 --task-lm openai/Qwen3-8B \
         --api-base http://127.0.0.1:8180/v1 --passes 3
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import resource
import socket
from pathlib import Path

import dspy

import paperexact_arms as px

HERE = Path(__file__).parent
try:
    _s, _h = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _h), _h))
except Exception:
    pass


def norm(t):
    return " ".join(str(t).lower().split())


def remove_unit(cand, unit_text):
    """Drop every line of every module whose normalized text contains the unit."""
    key = norm(unit_text)[:100]
    out, hits = {}, 0
    for mod, text in cand.items():
        lines = text.split("\n")
        keep = [l for l in lines if key not in norm(l)]
        hits += len(lines) - len(keep)
        out[mod] = "\n".join(keep)
    return out, hits


def find_module(cand, unit_text):
    key = norm(unit_text)[:100]
    for mod, text in cand.items():
        if key in norm(text):
            return mod
    return list(cand)[0]


def subclauses(unit_text):
    parts = [p.strip() for p in re.split(r"(?<=[.;])\s+", unit_text) if len(p.strip()) >= 20]
    return parts if len(parts) >= 2 else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--run", required=True)
    ap.add_argument("--lm-tag", default="Qwen3-8B")
    ap.add_argument("--task-lm", required=True)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--sub-top", type=int, default=3)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--eval-threads", type=int, default=32)
    a = ap.parse_args()

    # cache=False: passes must be independent samples (F3 rule)
    lm = dspy.LM(a.task_lm, api_base=a.api_base, api_key="EMPTY", cache=False,
                 temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens,
                 num_retries=10, timeout=300, extra_body={"top_k": a.top_k})
    dspy.configure(lm=lm)
    px.EVAL_THREADS = a.eval_threads

    bench, program, metric, _ = px.load_bench(a.bench)
    test = list(bench.test_set)
    rd = HERE / "runs_paperexact" / a.bench / a.lm_tag / a.run
    res = json.loads((rd / "result.json").read_text())
    cand = res["best_candidate"]
    blob = norm(" ".join(str(v) for v in cand.values()))
    marg = (res.get("units") or {}).get("marginals") or []
    chosen = [m for m in marg if norm(m["unit"])[:100] in blob]
    chosen.sort(key=lambda m: -(m.get("delta") or 0))
    top = chosen[:a.top_n]

    outdir = HERE / "runs_paperexact" / a.bench / a.lm_tag / "ablation_battery"
    outdir.mkdir(parents=True, exist_ok=True)
    log = outdir / "evals.jsonl"
    out_path = HERE / "runs" / f"ablation_battery_{a.bench}.json"
    out_path.parent.mkdir(exist_ok=True)
    prev = json.loads(out_path.read_text()) if out_path.exists() else {}
    done = {r["arm"]: r for r in prev.get("arms", [])}

    fp = {"host": socket.gethostname(), "api_base": a.api_base, "passes": a.passes,
          "cache": False, "max_tokens": a.max_tokens,
          "utc": datetime.datetime.utcnow().isoformat() + "Z"}
    print("FINGERPRINT:", json.dumps(fp), flush=True)

    arms = [("full", cand, None)]
    for i, m in enumerate(top):
        ab, hits = remove_unit(cand, m["unit"])
        arms.append((f"minus_{i}", ab, {"unit": m["unit"], "greedy_delta": m.get("delta"),
                                        "lines_removed": hits}))
    for i, m in enumerate(top[:a.sub_top]):                      # E3: minimality
        parts = subclauses(m["unit"])
        base, _ = remove_unit(cand, m["unit"])
        mod = find_module(cand, m["unit"])
        for j, part in enumerate(parts):
            sub = dict(base)
            sub[mod] = sub[mod] + "\n- " + part
            arms.append((f"sub_{i}_{j}", sub, {"unit_idx": i, "part": part}))

    recs = [done[k] for k in done]
    for name, c, meta in arms:
        if name in done:
            continue
        s = px.evaluate_cand(program, c, test, metric, log, f"abl_{name}", passes=a.passes)
        recs.append({"arm": name, "mean": s, "meta": meta})
        print(f"  {name}: {s}", flush=True)
        out_path.write_text(json.dumps({"bench": a.bench, "run": a.run, "fingerprint": fp,
                                        "top_units": [m["unit"] for m in top],
                                        "arms": recs}, indent=1))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
