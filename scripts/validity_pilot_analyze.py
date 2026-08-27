"""Analyze validity pilot results — compute consistency stats and R1 vs R2 verdict.

Loads:
  runs/validity_pilot/<run>/codegen/exec_results.jsonl   (code score per metric × trial × datapoint)
  runs/validity_pilot/<run>/judge/score_responses/*.json (judge score per metric × paraphrase × datapoint)

Computes per-metric:
  CODE-GEN
    - inter-trial std    σ_trial   = mean over datapoints of std across trials
    - across-datapoint σ σ_data     = mean over trials of std across datapoints
    - signal/noise       σ_data / σ_trial   (>1 means real discrimination > trial noise)
  JUDGE
    - paraphrase-consistency ICC-like: σ_para = mean over datapoints of std across paraphrases
    - across-datapoint σ                  = mean over paraphrases of std across datapoints
    - signal/noise                         = σ_data / σ_para
  CROSS-PIPELINE
    - Pearson between mean code score and mean judge score per datapoint

Output: runs/validity_pilot/<run>/analysis_report.md
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def parse_score_response(p):
    raw = p.read_text().strip()
    m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.S)
    if m: raw = m.group(1).strip()
    try: return json.loads(raw)
    except json.JSONDecodeError:
        s, e = raw.find("{"), raw.rfind("}")
        return json.loads(raw[s:e + 1])


def std(xs):
    return statistics.pstdev(xs) if len(xs) >= 2 else 0.0


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

    # ---- code-gen scores ----
    # code[metric_id][level][datapoint_id] = [score_trial0, score_trial1, ...]
    code = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for line in (base / "codegen" / "exec_results.jsonl").open():
        r = json.loads(line)
        if r["score"] is None: continue
        code[r["metric_id"]][r["level"]][r["datapoint_id"]].append(r["score"])

    # ---- judge scores ----
    # judge[metric_id][level][datapoint_id] = [paraphrase0, paraphrase1, ...]
    judge = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    sm_path = base / "judge" / "score_manifest.json"
    if sm_path.exists():
        score_manifest = json.loads(sm_path.read_text())
        for entry in score_manifest:
            rp = base / "judge" / "score_responses" / f"{entry['key']}.json"
            if not rp.exists(): continue
            try:
                obj = parse_score_response(rp)
                for sc in obj.get("scores", []):
                    dp_id = sc.get("id")
                    s = sc.get("score")
                    if dp_id is None or s is None: continue
                    judge[entry["metric_id"]][entry["level"]][dp_id].append(s/10.0)
            except Exception as e:
                print(f"  parse fail {entry['key']}: {e}")

    # ---- per-metric, per-level stats ----
    rows = []
    for m in metrics:
        mid = m["metric_id"]
        for level in ("r1", "r2"):
            # code stats
            per_dp_trials = [code[mid][level].get(dp, []) for dp in dp_ids]
            n_trials_per_dp = [len(x) for x in per_dp_trials if x]
            if not per_dp_trials or not n_trials_per_dp:
                continue
            # inter-trial std per datapoint
            sigma_trial_code = mean([std(x) for x in per_dp_trials if len(x) >= 2])
            # across-datapoint std per trial
            n_trials = max(n_trials_per_dp)
            sigma_data_code = mean([
                std([per_dp_trials[di][ti] for di in range(len(per_dp_trials))
                      if ti < len(per_dp_trials[di])])
                for ti in range(n_trials)])
            mean_code_per_dp = [mean(x) if x else None for x in per_dp_trials]

            # judge stats
            per_dp_paras = [judge[mid][level].get(dp, []) for dp in dp_ids]
            n_paras_per_dp = [len(x) for x in per_dp_paras if x]
            if not n_paras_per_dp:
                sigma_para_judge = sigma_data_judge = 0
                mean_judge_per_dp = [None] * len(dp_ids)
            else:
                sigma_para_judge = mean([std(x) for x in per_dp_paras if len(x) >= 2])
                n_paras = max(n_paras_per_dp)
                sigma_data_judge = mean([
                    std([per_dp_paras[di][pi] for di in range(len(per_dp_paras))
                          if pi < len(per_dp_paras[di])])
                    for pi in range(n_paras)])
                mean_judge_per_dp = [mean(x) if x else None for x in per_dp_paras]

            # cross-pipeline Pearson
            paired = [(c, j) for c, j in zip(mean_code_per_dp, mean_judge_per_dp)
                      if c is not None and j is not None]
            r_pearson = pearson([x[0] for x in paired], [x[1] for x in paired])

            rows.append({
                "metric_id": mid,
                "level": level,
                "metric_name": (m["r1_focal_name"] if level == "r1"
                                else m["r2_aspect_name"]),
                "code_sigma_trial": sigma_trial_code,
                "code_sigma_data": sigma_data_code,
                "code_snr": sigma_data_code / max(sigma_trial_code, 1e-3),
                "judge_sigma_para": sigma_para_judge,
                "judge_sigma_data": sigma_data_judge,
                "judge_snr": sigma_data_judge / max(sigma_para_judge, 1e-3),
                "code_vs_judge_pearson": r_pearson,
                "mean_code_score": mean([m for m in mean_code_per_dp if m is not None]),
                "mean_judge_score": mean([m for m in mean_judge_per_dp if m is not None]),
            })

    # ---- report ----
    report = ["# Validity pilot analysis — " + args.run_name,
              "",
              f"## {len(metrics)} metrics × {len(datapoints)} datapoints",
              "",
              "**SNR = across-datapoint σ / inter-trial-or-paraphrase σ.** SNR > 1 means "
              "the metric discriminates between datapoints more than it varies across "
              "trials/paraphrases — i.e., it carries real signal.",
              "",
              "## Per-metric, per-level stats", "",
              "| Metric | Level | Code σ_trial | Code σ_data | **Code SNR** | "
              "Judge σ_para | Judge σ_data | **Judge SNR** | Code↔Judge ρ |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        report.append(
            f"| {r['metric_id']}: {r['metric_name'][:50]} | {r['level'].upper()} | "
            f"{r['code_sigma_trial']:.3f} | {r['code_sigma_data']:.3f} | "
            f"**{r['code_snr']:.2f}** | "
            f"{r['judge_sigma_para']:.3f} | {r['judge_sigma_data']:.3f} | "
            f"**{r['judge_snr']:.2f}** | {r['code_vs_judge_pearson']:+.3f} |")

    # ---- aggregate by level ----
    by_level = defaultdict(list)
    for r in rows:
        by_level[r["level"]].append(r)
    report.append("")
    report.append("## Aggregate by level")
    report.append("")
    report.append("| Level | n metrics | mean Code SNR | mean Judge SNR | "
                  "mean Code↔Judge ρ |")
    report.append("|---|---|---|---|---|")
    for lvl in ("r1", "r2"):
        if not by_level[lvl]: continue
        rs = by_level[lvl]
        report.append(
            f"| {lvl.upper()} | {len(rs)} | "
            f"{mean([r['code_snr'] for r in rs]):.2f} | "
            f"{mean([r['judge_snr'] for r in rs]):.2f} | "
            f"{mean([r['code_vs_judge_pearson'] for r in rs]):+.3f} |")

    # ---- R1 vs R2 determination ----
    report.append("")
    report.append("## R1 vs R2 determination")
    report.append("")
    r1_code_snr = mean([r['code_snr'] for r in by_level['r1']])
    r2_code_snr = mean([r['code_snr'] for r in by_level['r2']])
    r1_judge_snr = mean([r['judge_snr'] for r in by_level['r1']])
    r2_judge_snr = mean([r['judge_snr'] for r in by_level['r2']])
    r1_pearson = mean([r['code_vs_judge_pearson'] for r in by_level['r1']])
    r2_pearson = mean([r['code_vs_judge_pearson'] for r in by_level['r2']])

    def cmp(a, b, label):
        winner = "R1" if a > b else "R2" if b > a else "tie"
        delta = abs(a - b)
        return f"- **{label}**: R1={a:.3f}, R2={b:.3f} — winner: **{winner}** (Δ={delta:.3f})"

    report.append(cmp(r1_code_snr, r2_code_snr, "Code SNR"))
    report.append(cmp(r1_judge_snr, r2_judge_snr, "Judge SNR"))
    report.append(cmp(r1_pearson, r2_pearson, "Code↔Judge Pearson"))

    out_path = base / "analysis_report.md"
    out_path.write_text("\n".join(report))
    print(f"wrote {out_path}")
    print("\n" + "\n".join(report[-15:]))

    # Save raw rows too
    (base / "analysis_rows.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
