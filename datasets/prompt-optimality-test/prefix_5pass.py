"""5-pass prefix value curves (user directive 2026-07-28: 'cue up the 5-fold runs to get
averages, maybe this will show a smoother increase'). Same declared-order construction as
prefix_extend.py, but each prefix-k candidate is evaluated with FIVE independent passes
(cache=False server-side sampling at temp .6) and all five scores are stored; the figure will
plot the mean. Ranges chosen to the figure windows: hotpot k=1..40, ifbench k=1..20.
Resumable per (bench, k): reruns skip completed ks (all 5 passes done).
Prefix order: frozen-pool units ranked by delta_8b desc (hotpot, same as prefix_extend) /
v6ctx32k run marginals delta desc (ifbench, same ranking statistic its own prefix sweep used).
"""
import json, datetime, socket, sys
from pathlib import Path
import paperexact_arms as px

HERE = Path(__file__).parent
api = sys.argv[1]
PASSES = 5
JOBS = [("hotpot", 40), ("ifbench", 20)]

for bench_name, KMAX in JOBS:
    bench, program, metric, _ = px.load_bench(bench_name)
    test = list(bench.test_set)
    seed_cand = px.get_instructions(program)
    if bench_name == "hotpot":
        units = json.loads((HERE / "pools/hotpot_Qwen3-8B_frozen.json").read_text())["units"]
        units = [u for u in units if u["module"] in seed_cand]
        units.sort(key=lambda u: -(u.get("delta_8b") or -9))
        order_tag = "pools/hotpot frozen delta_8b desc"
    else:
        r = json.loads((HERE / "runs_paperexact/ifbench/Qwen3-8B/unitrecomb_v6ctx32k/result.json")
                       .read_text())
        ms = (r.get("units") or {}).get("marginals") or []
        ms.sort(key=lambda m: -(m.get("delta") if m.get("delta") is not None else -9))
        units = [m for m in ms if m.get("module") in seed_cand]
        order_tag = "v6ctx32k marginals delta desc"
    out = HERE / f"runs/prefix_5pass_{bench_name}.json"
    res = json.loads(out.read_text()) if out.exists() else {
        "fingerprint": {"host": socket.gethostname(), "api_base": api, "order": order_tag,
                        "passes": PASSES, "utc": datetime.datetime.utcnow().isoformat() + "Z"},
        "curve": {}}
    log = HERE / f"runs_paperexact/{bench_name}/Qwen3-8B/prefix_5pass/evals.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    cand = dict(seed_cand)
    for k, u in enumerate(units[:KMAX], 1):
        cand[u["module"]] = cand[u["module"]] + "\n- " + u["unit"]
        got = res["curve"].get(str(k), [])
        if len(got) >= PASSES:
            continue
        for p in range(len(got), PASSES):
            s = px.evaluate_cand(program, dict(cand), test, metric, log,
                                 f"p5_{bench_name}_k{k}_pass{p}")
            got = res["curve"].setdefault(str(k), got if isinstance(got, list) else [])
            if not isinstance(res["curve"][str(k)], list):
                res["curve"][str(k)] = []
            res["curve"][str(k)].append(s)
            out.write_text(json.dumps(res, indent=1))
        m = sum(res["curve"][str(k)]) / len(res["curve"][str(k)])
        print(f"  [{bench_name}] k={k}/{KMAX}: mean={m:.4f} {res['curve'][str(k)]}", flush=True)
    print(f"DONE {bench_name}: {len(res['curve'])} ks", flush=True)
