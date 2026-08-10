"""Aggregate R1-within-R2 code scores per aspect; compare to direct R2 methods.

For each R2 aspect A with R1 members {f1...fk}:
  - Execute each f_i's code on each datapoint, average across (trial=1 here).
  - Aggregate per aspect using {mean, max} → aspect_code_aggregated.
  - Compare against:
      (a) direct R2 Claude code score (mean across trials)
      (b) direct R2 Llama code score (single trial)
      (c) Claude R2 judge mean across paraphrases
      (d) Llama R2 judge mean across paraphrases

Outputs:
  runs/validity_pilot/<run>/codegen/r1_within_r2_exec.jsonl
  runs/validity_pilot/<run>/analysis_r1_aggregated_vs_r2.md
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
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


def run_one(code, text, timeout=5.0):
    runner = (
        f"{code}\n\n"
        "import json, sys\n"
        "text = json.loads(sys.stdin.read())\n"
        "try:\n"
        "    s = float(score(text))\n"
        "    if not (0.0 <= s <= 1.0): s = max(0.0, min(1.0, s))\n"
        "    print(json.dumps({'score': s}))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'error': str(e)[:120]}))\n"
    )
    try:
        proc = subprocess.run([sys.executable, "-c", runner],
                              input=json.dumps(text), capture_output=True,
                              text=True, timeout=timeout)
        if proc.returncode != 0: return None
        try:
            out = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            return None
        if "error" in out: return None
        return out.get("score")
    except subprocess.TimeoutExpired:
        return None


def parse_judge(p):
    raw = p.read_text().strip()
    m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.S)
    if m: raw = m.group(1).strip()
    try: return json.loads(raw)
    except json.JSONDecodeError:
        s, e = raw.find("{"), raw.rfind("}")
        return json.loads(raw[s:e + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="smoke")
    args = ap.parse_args()
    base = Path(f"runs/validity_pilot/{args.run_name}")
    metrics = json.loads((base / "metrics.json").read_text())
    datapoints = json.loads((base / "datapoints.json").read_text())
    dp_ids = [d["datapoint_id"] for d in datapoints]
    manifest = json.loads((base / "codegen" / "r1_within_r2_manifest.json").read_text())

    # Execute each R1-within-R2 Llama code on each datapoint
    exec_out = base / "codegen" / "r1_within_r2_exec.jsonl"
    code_dir = base / "codegen" / "responses_r1_within_r2_llama"
    # aspect_id -> r1_fi -> dp_id -> score
    per_r1 = defaultdict(lambda: defaultdict(dict))
    print(f"Executing {len(manifest)} R1 codes × {len(dp_ids)} datapoints...")
    with exec_out.open("w") as f:
        for entry in manifest:
            cp = code_dir / f"{entry['key']}.py"
            if not cp.exists():
                continue
            code = cp.read_text()
            try: compile(code, cp.name, "exec")
            except SyntaxError: continue
            for dp in datapoints:
                s = run_one(code, dp["text"])
                if s is None: continue
                per_r1[entry["metric_id"]][entry["r1_family_id"]][dp["datapoint_id"]] = s
                f.write(json.dumps({**entry, "datapoint_id": dp["datapoint_id"],
                                     "score": s}) + "\n")
    print(f"executed all. wrote {exec_out}")

    # Load existing Claude/Llama R2 codes (focal R1 + R2 framing)
    claude_r2_code = defaultdict(lambda: defaultdict(list))
    llama_r2_code = defaultdict(dict)
    for line in (base / "codegen" / "exec_results.jsonl").open():
        r = json.loads(line)
        if r["score"] is None: continue
        if r["level"] == "r2":
            claude_r2_code[r["metric_id"]][r["datapoint_id"]].append(r["score"])
    if (base / "codegen" / "exec_results_llama.jsonl").exists():
        for line in (base / "codegen" / "exec_results_llama.jsonl").open():
            r = json.loads(line)
            if r["score"] is None: continue
            if r["level"] == "r2":
                llama_r2_code[r["metric_id"]][r["datapoint_id"]] = r["score"]

    # Load judge scores at R2
    claude_r2_judge = defaultdict(list)  # mid -> [dp -> [scores_across_paraphrases]]
    llama_r2_judge = defaultdict(list)
    score_manifest = json.loads((base / "judge" / "score_manifest.json").read_text())
    j_claude = defaultdict(lambda: defaultdict(list))
    j_llama = defaultdict(lambda: defaultdict(list))
    for entry in score_manifest:
        if entry["level"] != "r2": continue
        rp_c = base / "judge" / "score_responses" / f"{entry['key']}.json"
        rp_l = base / "judge" / "score_responses_llama" / f"{entry['key']}.json"
        if rp_c.exists():
            try:
                obj = parse_judge(rp_c)
                for sc in obj.get("scores", []):
                    j_claude[entry["metric_id"]][sc["id"]].append(sc["score"] / 10.0)
            except Exception: pass
        if rp_l.exists():
            try:
                obj = parse_judge(rp_l)
                for sc in obj.get("scores", []):
                    j_llama[entry["metric_id"]][sc["id"]].append(sc["score"] / 10.0)
            except Exception: pass

    # Build comparison per aspect
    rows = []
    for m in metrics:
        mid = m["metric_id"]
        # Per-datapoint aggregations of R1-within-R2 codes
        r1_codes = per_r1[mid]
        r1_fis = list(r1_codes.keys())
        if not r1_fis: continue
        agg_mean = {dp: mean([r1_codes[fi].get(dp) for fi in r1_fis
                              if dp in r1_codes[fi]])
                    for dp in dp_ids}
        agg_max = {dp: max([r1_codes[fi].get(dp, 0) for fi in r1_fis])
                   for dp in dp_ids}

        # Existing R2 methods (per dp means)
        claude_r2_code_per_dp = {dp: mean(claude_r2_code[mid].get(dp, [])) if claude_r2_code[mid].get(dp) else None
                                  for dp in dp_ids}
        llama_r2_code_per_dp = {dp: llama_r2_code[mid].get(dp) for dp in dp_ids}
        claude_r2_judge_per_dp = {dp: mean(j_claude[mid].get(dp, [])) if j_claude[mid].get(dp) else None
                                   for dp in dp_ids}
        llama_r2_judge_per_dp = {dp: mean(j_llama[mid].get(dp, [])) if j_llama[mid].get(dp) else None
                                  for dp in dp_ids}

        def corr(a, b):
            paired = [(a[dp], b[dp]) for dp in dp_ids
                      if a.get(dp) is not None and b.get(dp) is not None]
            if len(paired) < 2: return 0.0
            return pearson([x[0] for x in paired], [x[1] for x in paired])

        rows.append({
            "metric_id": mid,
            "r2_aspect_name": m["r2_aspect_name"],
            "n_r1_used": len(r1_fis),
            # convergent validity: aggregated R1 codes vs each R2 method
            "rho_aggMean_vs_claudeJudge": corr(agg_mean, claude_r2_judge_per_dp),
            "rho_aggMax_vs_claudeJudge": corr(agg_max, claude_r2_judge_per_dp),
            "rho_aggMean_vs_llamaJudge": corr(agg_mean, llama_r2_judge_per_dp),
            "rho_aggMean_vs_directClaudeR2Code": corr(agg_mean, claude_r2_code_per_dp),
            "rho_aggMean_vs_directLlamaR2Code": corr(agg_mean, llama_r2_code_per_dp),
            # baseline: direct R2 code vs R2 judge
            "rho_directClaudeR2Code_vs_claudeJudge": corr(claude_r2_code_per_dp, claude_r2_judge_per_dp),
            "rho_directLlamaR2Code_vs_llamaJudge": corr(llama_r2_code_per_dp, llama_r2_judge_per_dp),
        })

    report = ["# R1-aggregated vs direct R2 — convergent validity", "",
              f"## {len(rows)} R2 aspects × {len(dp_ids)} datapoints", "",
              "**Question**: does aggregating R1-level code scores give better convergence",
              "with R2-level LLM-judge than coding directly at R2?", "",
              "Aggregators: `mean` = average across R1 sub-rules; `max` = at least one sub-rule satisfied.", "",
              "| Aspect | n R1 | agg-mean ↔ Claude-judge | agg-max ↔ Claude-judge | "
              "agg-mean ↔ Llama-judge | agg-mean ↔ direct-R2-Claude-code | "
              "**baseline**: direct-R2-Claude-code ↔ Claude-judge |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        report.append(
            f"| {r['metric_id']}: {r['r2_aspect_name'][:35]} | {r['n_r1_used']} | "
            f"**{r['rho_aggMean_vs_claudeJudge']:+.3f}** | "
            f"{r['rho_aggMax_vs_claudeJudge']:+.3f} | "
            f"{r['rho_aggMean_vs_llamaJudge']:+.3f} | "
            f"{r['rho_aggMean_vs_directClaudeR2Code']:+.3f} | "
            f"{r['rho_directClaudeR2Code_vs_claudeJudge']:+.3f} |")

    report.append("")
    report.append("## Aggregate")
    report.append("")
    cols = [
        ("agg-mean ↔ Claude-judge", "rho_aggMean_vs_claudeJudge"),
        ("agg-max ↔ Claude-judge", "rho_aggMax_vs_claudeJudge"),
        ("agg-mean ↔ Llama-judge", "rho_aggMean_vs_llamaJudge"),
        ("agg-mean ↔ direct-Claude-R2-code", "rho_aggMean_vs_directClaudeR2Code"),
        ("agg-mean ↔ direct-Llama-R2-code", "rho_aggMean_vs_directLlamaR2Code"),
        ("baseline direct-Claude-R2-code ↔ Claude-judge", "rho_directClaudeR2Code_vs_claudeJudge"),
        ("baseline direct-Llama-R2-code ↔ Llama-judge", "rho_directLlamaR2Code_vs_llamaJudge"),
    ]
    report.append("| Comparison | mean ρ |")
    report.append("|---|---|")
    for label, key in cols:
        report.append(f"| {label} | **{mean([r[key] for r in rows]):+.3f}** |")

    out_path = base / "analysis_r1_aggregated_vs_r2.md"
    out_path.write_text("\n".join(report))
    print(f"\nwrote {out_path}")
    print("\n" + "\n".join(report[-15:]))


if __name__ == "__main__":
    main()
