"""Cross-model judge analysis — Claude vs Llama as judges.

Reads:
  judge/score_responses/<key>.json         (Claude judge)
  judge/score_responses_llama/<key>.json   (Llama judge)

Computes per-metric-per-level:
  - within-Claude paraphrase σ (already in main analyze)
  - within-Llama paraphrase σ
  - **cross-judge Pearson** per datapoint (Claude mean across paraphrases vs Llama mean)
  - **cross-judge mean diff** (systematic bias)
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def parse(p):
    raw = p.read_text().strip()
    m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.S)
    if m: raw = m.group(1).strip()
    try: return json.loads(raw)
    except json.JSONDecodeError:
        s, e = raw.find("{"), raw.rfind("}")
        return json.loads(raw[s:e + 1])


def mean(xs):
    return statistics.mean(xs) if xs else 0.0


def std(xs):
    return statistics.pstdev(xs) if len(xs) >= 2 else 0.0


def pearson(xs, ys):
    if len(xs) < 2: return 0.0
    mx, my = mean(xs), mean(ys)
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    dx = sum((x-mx)**2 for x in xs) ** 0.5
    dy = sum((y-my)**2 for y in ys) ** 0.5
    return num / max(dx*dy, 1e-9)


def collect(base, subdir):
    """{metric_id}{level} -> {datapoint_id} -> [scores_across_paraphrases]"""
    sm = json.loads((base / "judge" / "score_manifest.json").read_text())
    out = defaultdict(lambda: defaultdict(list))
    for entry in sm:
        rp = base / "judge" / subdir / f"{entry['key']}.json"
        if not rp.exists(): continue
        try:
            obj = parse(rp)
            for sc in obj.get("scores", []):
                dp_id = sc.get("id"); s = sc.get("score")
                if dp_id is None or s is None: continue
                key = f"{entry['metric_id']}__{entry['level']}"
                out[key][dp_id].append(s / 10.0)
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="smoke")
    args = ap.parse_args()
    base = Path(f"runs/validity_pilot/{args.run_name}")
    metrics = json.loads((base / "metrics.json").read_text())
    datapoints = json.loads((base / "datapoints.json").read_text())
    dp_ids = [d["datapoint_id"] for d in datapoints]

    claude_j = collect(base, "score_responses")
    llama_j = collect(base, "score_responses_llama")

    if not llama_j:
        print(f"no Llama judge results yet — skipping")
        return

    rows = []
    for m in metrics:
        for level in ("r1", "r2"):
            key = f"{m['metric_id']}__{level}"
            c_per_dp = [mean(claude_j[key].get(d, [])) if claude_j[key].get(d) else None
                         for d in dp_ids]
            l_per_dp = [mean(llama_j[key].get(d, [])) if llama_j[key].get(d) else None
                         for d in dp_ids]
            paired = [(c, l) for c, l in zip(c_per_dp, l_per_dp)
                      if c is not None and l is not None]
            if not paired:
                continue
            cs = [x[0] for x in paired]
            ls = [x[1] for x in paired]
            # within-judge paraphrase σ (averaged across datapoints)
            c_sigma_p = mean([std(claude_j[key].get(d, [])) for d in dp_ids
                              if len(claude_j[key].get(d, [])) >= 2])
            l_sigma_p = mean([std(llama_j[key].get(d, [])) for d in dp_ids
                              if len(llama_j[key].get(d, [])) >= 2])
            rows.append({
                "metric_id": m["metric_id"],
                "level": level,
                "metric_name": (m["r1_focal_name"] if level == "r1"
                                else m["r2_aspect_name"]),
                "claude_judge_mean": mean(cs),
                "llama_judge_mean": mean(ls),
                "abs_diff": abs(mean(cs) - mean(ls)),
                "claude_sigma_para": c_sigma_p,
                "llama_sigma_para": l_sigma_p,
                "cross_judge_pearson": pearson(cs, ls),
            })

    report = ["# Cross-judge analysis — Claude vs Llama as judge", "",
              f"## {len(metrics)} metrics × {len(datapoints)} datapoints", "",
              "Cross-judge Pearson = how well Claude-judge and Llama-judge agree per datapoint",
              "(both averaged across 5 paraphrases). High Pearson = robust convergent validity",
              "across judge model families.", "",
              "| Metric | Level | Claude μ | Llama μ | |diff| | Claude σ_para | "
              "Llama σ_para | **Cross-judge ρ** |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        report.append(
            f"| {r['metric_id']}: {r['metric_name'][:40]} | {r['level'].upper()} | "
            f"{r['claude_judge_mean']:.3f} | {r['llama_judge_mean']:.3f} | "
            f"{r['abs_diff']:.3f} | {r['claude_sigma_para']:.3f} | "
            f"{r['llama_sigma_para']:.3f} | **{r['cross_judge_pearson']:+.3f}** |")

    by_level = defaultdict(list)
    for r in rows: by_level[r["level"]].append(r)
    report.append("")
    report.append("## Aggregate by level")
    report.append("")
    report.append("| Level | n | Claude μ | Llama μ | mean|diff| | Claude σ_para | "
                  "Llama σ_para | **mean cross-judge ρ** |")
    report.append("|---|---|---|---|---|---|---|---|")
    for lvl in ("r1", "r2"):
        rs = by_level[lvl]
        if not rs: continue
        report.append(
            f"| {lvl.upper()} | {len(rs)} | "
            f"{mean([r['claude_judge_mean'] for r in rs]):.3f} | "
            f"{mean([r['llama_judge_mean'] for r in rs]):.3f} | "
            f"{mean([r['abs_diff'] for r in rs]):.3f} | "
            f"{mean([r['claude_sigma_para'] for r in rs]):.3f} | "
            f"{mean([r['llama_sigma_para'] for r in rs]):.3f} | "
            f"**{mean([r['cross_judge_pearson'] for r in rs]):+.3f}** |")

    out_path = base / "analysis_cross_judge.md"
    out_path.write_text("\n".join(report))
    print(f"wrote {out_path}")
    print("\n" + "\n".join(report[-15:]))


if __name__ == "__main__":
    main()
