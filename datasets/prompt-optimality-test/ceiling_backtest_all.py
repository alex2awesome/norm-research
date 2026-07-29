"""E1-full extended (2026-07-28, user directive): prefix-fit ceiling-prediction backtest over
ALL SIX benches' greedy prefix trajectories (the original E1-full covered hotpot/aime/livebench
only; ifbench had no prefix rows at the time -- v6ctx32k now has k=1..48).

For each bench's newest prefix_k curve y(k): fit on the first j points (all j >= 5), predict all
later k, record |err| bucketed by horizon (k - j): 1-3 / 4-8 / 9+. Two forms:
  satexp:   y = a - b * r^k        (grid over r)
  powerlaw: y = a - b * k^{-c}     (grid over c)
Writes runs/ceiling_backtest_all.json (new file; runs/ceiling_backtest.json from the original
E1 run is left untouched per the never-delete-data rule).
"""
import json, re, glob, math, statistics as st
from pathlib import Path

HERE = Path(__file__).parent
SOURCES = {  # newest / canonical trajectory per bench
    "hotpot":    "runs/prefix_extend_hotpot.json",              # same-session k=1..68 (lane2)
    "aime":      "runs_paperexact/aime/Qwen3-8B/unitrecomb_v5sk2/proposals.jsonl",
    "hover":     "runs_paperexact/hover/Qwen3-8B/unitrecomb/proposals.jsonl",
    "ifbench":   "runs_paperexact/ifbench/Qwen3-8B/unitrecomb_v6ctx32k/proposals.jsonl",
    "livebench": "runs_paperexact/livebench/Qwen3-8B/unitrecomb/proposals.jsonl",
    "pupa":      "runs_paperexact/pupa/Qwen3-8B/unitrecomb/proposals.jsonl",
}

def load_curve(path):
    p = HERE / path
    if p.suffix == ".json":
        d = json.loads(p.read_text())
        if "curve" in d:
            d = d["curve"]
        return {int(k): float(v) for k, v in d.items()}
    curve = {}
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        m = re.fullmatch(r"prefix_k(\d+)", str(r.get("phase", "")))
        if not m:
            continue
        y = r.get("mean_score")
        if y is None and r.get("scores"):
            y = sum(r["scores"]) / len(r["scores"])
        if y is None:
            continue
        y = float(y)
        nb = r.get("n_batch") or (len(r.get("scores") or []) or None)
        if y > 1.5 and nb:      # count-style score (aime logs correct-counts)
            y = y / nb
        curve[int(m.group(1))] = y   # later rows overwrite -> newest session wins
    return curve

def fit_satexp(ks, ys):
    best = None
    for ri in range(30, 100):
        r = ri / 100.0
        R = [r ** k for k in ks]; n = len(ks)
        sR, sRR = sum(R), sum(t * t for t in R)
        sy, sRy = sum(ys), sum(t * u for t, u in zip(R, ys))
        det = n * sRR - sR * sR
        if abs(det) < 1e-12:
            continue
        b = -(n * sRy - sR * sy) / det; a = (sy + b * sR) / n
        sse = sum((a - b * (r ** k) - u) ** 2 for k, u in zip(ks, ys))
        if best is None or sse < best[0]:
            best = (sse, lambda k, a=a, b=b, r=r: a - b * r ** k)
    return best[1] if best else None

def fit_powerlaw(ks, ys):
    best = None
    for ci in range(20, 300, 5):
        c = ci / 100.0
        R = [k ** (-c) for k in ks]; n = len(ks)
        sR, sRR = sum(R), sum(t * t for t in R)
        sy, sRy = sum(ys), sum(t * u for t, u in zip(R, ys))
        det = n * sRR - sR * sR
        if abs(det) < 1e-12:
            continue
        b = -(n * sRy - sR * sy) / det; a = (sy + b * sR) / n
        sse = sum((a - b * (k ** (-c)) - u) ** 2 for k, u in zip(ks, ys))
        if best is None or sse < best[0]:
            best = (sse, lambda k, a=a, b=b, c=c: a - b * k ** (-c))
    return best[1] if best else None

BUCKETS = (("1-3", 1, 3), ("4-8", 4, 8), ("9+", 9, 10**9))
out = {"per_bench": {}, "pooled": {}, "sources": SOURCES}
pooled = {form: {b[0]: [] for b in BUCKETS} for form in ("satexp", "powerlaw")}
for bench, src in SOURCES.items():
    curve = load_curve(src)
    ks = sorted(curve)
    if len(ks) < 8:
        out["per_bench"][bench] = {"n_points": len(ks), "skipped": True}
        print(f"{bench:10} SKIP (only {len(ks)} prefix points)")
        continue
    errs = {form: {b[0]: [] for b in BUCKETS} for form in ("satexp", "powerlaw")}
    for j in range(5, len(ks) - 1):
        fk, fy = ks[:j], [curve[k] for k in ks[:j]]
        fits = {"satexp": fit_satexp(fk, fy), "powerlaw": fit_powerlaw(fk, fy)}
        for form, f in fits.items():
            if f is None:
                continue
            for k in ks[j:]:
                h = k - ks[j - 1]
                e = abs(f(k) - curve[k])
                for name, lo, hi in BUCKETS:
                    if lo <= h <= hi:
                        errs[form][name].append(e)
                        pooled[form][name].append(e)
    out["per_bench"][bench] = {
        "n_points": len(ks), "k_range": [ks[0], ks[-1]],
        **{f"{form}_{name}": {"median": round(st.median(v), 4), "max": round(max(v), 4),
                              "n": len(v)}
           for form in errs for name, _, _ in BUCKETS if (v := errs[form][name])}}
    print(bench, out["per_bench"][bench])
out["pooled"] = {f"{form}_{name}": {"median": round(st.median(v), 4), "max": round(max(v), 4),
                                     "n": len(v)}
                 for form in pooled for name, _, _ in BUCKETS if (v := pooled[form][name])}
print("POOLED:", json.dumps(out["pooled"], indent=1))
(HERE / "runs" / "ceiling_backtest_all.json").write_text(json.dumps(out, indent=1))
print("wrote runs/ceiling_backtest_all.json")
