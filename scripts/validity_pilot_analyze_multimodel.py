"""Cross-model analysis — Claude vs Llama code-gen agreement.

Reads:
  codegen/exec_results.jsonl        (Claude trials)
  codegen/exec_results_llama.jsonl  (Llama trials)

Computes per-metric-per-level:
  - within-Claude inter-trial std (already in main analyze)
  - within-Llama inter-trial std (here, Llama only ran once so this is 0)
  - **cross-model Pearson per datapoint** (Claude mean vs Llama mean)
  - **cross-model mean diff** (systematic bias between models)
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def mean(xs):
    return statistics.mean(xs) if xs else 0.0


def pearson(xs, ys):
    if len(xs) < 2: return 0.0
    mx, my = mean(xs), mean(ys)
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    dx = sum((x-mx)**2 for x in xs) ** 0.5
    dy = sum((y-my)**2 for y in ys) ** 0.5
    return num / max(dx*dy, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="smoke")
    args = ap.parse_args()
    base = Path(f"runs/validity_pilot/{args.run_name}")
    metrics = json.loads((base / "metrics.json").read_text())
    datapoints = json.loads((base / "datapoints.json").read_text())
    dp_ids = [d["datapoint_id"] for d in datapoints]

    # Claude scores
    claude = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for line in (base / "codegen" / "exec_results.jsonl").open():
        r = json.loads(line)
        if r["score"] is None: continue
        claude[r["metric_id"]][r["level"]][r["datapoint_id"]].append(r["score"])

    # Llama scores
    llama_path = base / "codegen" / "exec_results_llama.jsonl"
    if not llama_path.exists():
        print(f"no Llama exec results at {llama_path}")
        return
    llama = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for line in llama_path.open():
        r = json.loads(line)
        if r["score"] is None: continue
        llama[r["metric_id"]][r["level"]][r["datapoint_id"]].append(r["score"])

    rows = []
    for m in metrics:
        mid = m["metric_id"]
        for level in ("r1", "r2"):
            claude_per_dp = [mean(claude[mid][level].get(dp, [])) if claude[mid][level].get(dp) else None
                              for dp in dp_ids]
            llama_per_dp = [mean(llama[mid][level].get(dp, [])) if llama[mid][level].get(dp) else None
                             for dp in dp_ids]
            paired = [(c, l) for c, l in zip(claude_per_dp, llama_per_dp)
                      if c is not None and l is not None]
            if not paired:
                continue
            cs = [x[0] for x in paired]
            ls = [x[1] for x in paired]
            r_cross = pearson(cs, ls)
            rows.append({
                "metric_id": mid,
                "level": level,
                "metric_name": (m["r1_focal_name"] if level == "r1"
                                else m["r2_aspect_name"]),
                "n_paired": len(paired),
                "claude_mean": mean(cs),
                "llama_mean": mean(ls),
                "abs_diff": abs(mean(cs) - mean(ls)),
                "cross_model_pearson": r_cross,
            })

    report = ["# Cross-model analysis — Claude vs Llama code-gen", "",
              f"## {len(metrics)} metrics × {len(datapoints)} datapoints", "",
              "Cross-model Pearson measures how well Claude-generated and Llama-generated",
              "score functions AGREE per-datapoint (averaged across trials). High Pearson means",
              "the two models converge on the same metric semantics; low means they extracted",
              "different concrete behaviors from the same rubric.", "",
              "| Metric | Level | Claude μ | Llama μ | |diff| | Cross-model Pearson |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        report.append(
            f"| {r['metric_id']}: {r['metric_name'][:45]} | {r['level'].upper()} | "
            f"{r['claude_mean']:.3f} | {r['llama_mean']:.3f} | {r['abs_diff']:.3f} | "
            f"**{r['cross_model_pearson']:+.3f}** |")

    by_level = defaultdict(list)
    for r in rows: by_level[r["level"]].append(r)
    report.append("")
    report.append("## Aggregate by level")
    report.append("")
    report.append("| Level | n metrics | mean Claude μ | mean Llama μ | mean |diff| | "
                  "mean cross-model Pearson |")
    report.append("|---|---|---|---|---|---|")
    for lvl in ("r1", "r2"):
        if not by_level[lvl]: continue
        rs = by_level[lvl]
        report.append(
            f"| {lvl.upper()} | {len(rs)} | "
            f"{mean([r['claude_mean'] for r in rs]):.3f} | "
            f"{mean([r['llama_mean'] for r in rs]):.3f} | "
            f"{mean([r['abs_diff'] for r in rs]):.3f} | "
            f"{mean([r['cross_model_pearson'] for r in rs]):+.3f} |")

    out_path = base / "analysis_cross_model.md"
    out_path.write_text("\n".join(report))
    print(f"wrote {out_path}")
    print("\n" + "\n".join(report[-10:]))


if __name__ == "__main__":
    main()
