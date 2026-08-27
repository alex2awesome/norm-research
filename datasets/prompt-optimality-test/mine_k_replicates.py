"""K independent re-mining replicates per benchmark (HB105; user directive 2026-07-26).

WHY. Capture-recapture needs independent captures. Our historical "pools" mostly are not:
staircase and follow-on runs consumed a FROZEN pool file (deliberate at the time — it made the
SEARCH reproducible and removed the flaky z.ai dependency from GPU runs), so the same units
appear in every run by construction, f1=0, and Chao collapses to the observed count. That
freezing was right for search comparability and wrong for certifiability. This script does the
missing thing: freeze the MINER (not the pool) and run it K times independently.

DECLARED SAMPLER. One replicate = the paper's LLM-suggestion channel exactly as the harness
runs it (_suggest_units_paper: 5 framings x n clauses, GLM reflection LM at its default
temperature, train-example grounding from the same 3 items). Independence across replicates
comes from API sampling stochasticity; nothing is cached (fresh process per replicate would be
ideal, but fresh GLM calls suffice — every call is a fresh sample). The trajectory-harvest
channel is NOT re-run (it reads fixed trajectories); the certifiability claim is therefore
indexed to the declared LLM-suggestion sampler. Report it that way.

Usage:  python3 mine_k_replicates.py --benches ifbench aime hover hotpot --k 3
Writes: pools/remine/{bench}_replicate_{i}.json  +  a summary with capture-recapture stats.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import paperexact_arms as px

HERE = Path(__file__).parent


def capture_stats(replicates):
    caps = Counter()
    n_draws = 0
    for units in replicates:
        seen = {(m, " ".join(c.lower().split())[:120]) for m, c in units}
        n_draws += len(seen)
        for k in seen:
            caps[k] += 1
    f = Counter(caps.values())
    S = len(caps)
    f1, f2 = f.get(1, 0), f.get(2, 0)
    out = {"S": S, "f1": f1, "f2": f2, "n_draws": n_draws,
           "freq_spectrum": dict(sorted(f.items()))}
    if f1 and f2:
        N = S + f1 * f1 / (2 * f2)
        var = f2 * ((f1 / f2) ** 2 / 2 + (f1 / f2) ** 3 + (f1 / f2) ** 4 / 4)
        D = N - S
        if D > 0 and var > 0:
            C = math.exp(1.96 * math.sqrt(math.log(1 + var / D ** 2)))
            out["chao1"] = {"N_hat": N, "ci95": [S + D / C, S + D * C],
                            "unseen_ucb95": S + D * C - S}
        else:
            out["chao1"] = {"N_hat": N}
    if n_draws:
        gt = f1 / n_draws
        out["good_turing"] = {"p_next_novel": gt,
                              "ucb95_one_sided": gt + math.sqrt(2 * math.log(20) / n_draws)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", nargs="+", required=True)
    ap.add_argument("--lm-tag", default="Qwen3-8B")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n", type=int, default=12, help="clauses requested per framing")
    ap.add_argument("--reflection-model", default="glm-5.2")
    a = ap.parse_args()

    outdir = HERE / "pools" / "remine"
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    for bench_name in a.benches:
        bench, program, _, _ = px.load_bench(bench_name)
        seed_cand = px.get_instructions(program)
        init_cand = dict(seed_cand)
        off = HERE / "runs_paperexact" / bench_name / a.lm_tag / "official" / "result.json"
        if off.exists():
            init_cand = {**seed_cand, **json.loads(off.read_text())["best_candidate"]}
        # train-example grounding, byte-identical to the harness's construction
        ex_strs = []
        for ex in list(bench.train_set)[:3]:
            try:
                d = ex.toDict() if hasattr(ex, "toDict") else dict(ex)
                ex_strs.append(json.dumps({k: str(v)[:300] for k, v in d.items()})[:1000])
            except Exception:
                pass
        train_examples = "\n".join(ex_strs) or None

        replicates = []
        for i in range(a.k):
            rep_path = outdir / f"{bench_name}_replicate_{i}.json"
            if rep_path.exists():                       # resume: never redo a finished replicate
                units = [(d["module"], d["unit"]) for d in json.loads(rep_path.read_text())["units"]]
                print(f"[{bench_name}] replicate {i}: loaded {len(units)} units (cached)", flush=True)
            else:
                # cache=False is MANDATORY here: with dspy's response cache on, all K
                # "replicates" returned byte-identical unit lists (verified: identical md5 per
                # bench, f1=0, 45s total) — the same F3 pathology that made k-passes fictitious.
                # A replicate that can be served from cache is not a capture.
                import dspy
                if a.reflection_model.startswith("local:"):
                    # local:<served-name>@<api-base> — the GLM route died 2026-07-25, so the
                    # declared sampler is now a locally served model. cache=False still
                    # mandatory (see above); temperature=1.0 preserves the sampling channel
                    # that makes replicates independent captures.
                    _spec = a.reflection_model[len("local:"):]
                    _name, _base = _spec.split("@", 1)
                    refl = dspy.LM(f"openai/{_name}", api_base=_base, api_key="EMPTY",
                                   temperature=1.0, max_tokens=16000,
                                   num_retries=20, timeout=300, cache=False)
                else:
                    refl = dspy.LM(f"anthropic/{a.reflection_model}",
                                   api_base="https://api.z.ai/api/anthropic",
                                   api_key=px._zai_key(), temperature=1.0, max_tokens=16000,
                                   num_retries=40, timeout=300, cache=False)
                units = px._suggest_units_paper(bench.__class__.__name__, init_cand, refl,
                                                n=a.n, train_examples=train_examples)
                rep_path.write_text(json.dumps({
                    "bench": bench_name, "replicate": i,
                    "sampler": "llm_suggestion_5framings",
                    "reflection_model": a.reflection_model, "n_per_framing": a.n,
                    "units": [{"module": m, "unit": c, "source": "llm"} for m, c in units],
                }, indent=1))
                print(f"[{bench_name}] replicate {i}: mined {len(units)} units", flush=True)
            replicates.append(units)

        stats = capture_stats(replicates)
        summary[bench_name] = {"k": len(replicates),
                               "per_replicate_sizes": [len(r) for r in replicates],
                               **stats}
        summary_path.write_text(json.dumps(summary, indent=1))
        print(f"[{bench_name}] capture-recapture over {len(replicates)} replicates: "
              f"S={stats['S']} f1={stats['f1']} f2={stats['f2']} "
              f"chao1={stats.get('chao1', {}).get('N_hat')}", flush=True)

    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
