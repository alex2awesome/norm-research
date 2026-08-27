"""Full validity analysis — paraphrase convergence + correlation with labels.

Outputs per-R1 and per-R2 scores aggregated across paraphrases, then computes:

  Convergence (within-method, per metric):
    - code paraphrase agreement σ      (across 5 paraphrases of the same R1)
    - judge paraphrase agreement σ     (across 5 paraphrases of the same R2)
    - mean intra-method Pearson per metric (paraphrases as "raters")

  Convergent validity (cross-method, per aspect):
    - aggregated R1 code (mean across R1 sub-families × paraphrases) vs R2 judge

  Predictive validity (per aspect, vs label):
    - Pearson(aggregated_R1_code, judgement_label)
    - Pearson(R2_judge_score, judgement_label)
    - AUC for each (binary label)

Reports:
  runs/validity_full/<run>/analysis_full.md
  runs/validity_full/<run>/per_aspect_scores.parquet   (or .csv if no pyarrow)
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def mean(xs):
    return statistics.mean(xs) if xs else 0.0


def std(xs):
    return statistics.pstdev(xs) if len(xs) >= 2 else 0.0


def pearson(xs, ys):
    if len(xs) < 2: return 0.0
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / max(dx * dy, 1e-9)


def auc(scores, labels):
    """Simple AUC for binary labels."""
    paired = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    rank_sum_pos = 0
    for i, (s, l) in enumerate(paired):
        if l == 1: rank_sum_pos += i + 1
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def parse_score(p):
    raw = p.read_text().strip()
    m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.S)
    if m: raw = m.group(1).strip()
    try: return json.loads(raw)
    except json.JSONDecodeError:
        s, e = raw.find("{"), raw.rfind("}")
        return json.loads(raw[s:e + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="full_v1")
    args = ap.parse_args()
    base = Path(f"runs/validity_full/{args.run_name}")

    r1_metrics = json.loads((base / "r1_metrics.json").read_text())
    r2_aspects = json.loads((base / "r2_aspects.json").read_text())
    datapoints = json.loads((base / "datapoints.json").read_text())
    labels = {d["datapoint_id"]: d["judgement"] for d in datapoints}
    dp_ids = [d["datapoint_id"] for d in datapoints]

    # --- Load code exec results ---
    # code_scores[metric_id][paraphrase_idx][dp_id] = score (or None)
    code = defaultdict(lambda: defaultdict(dict))
    cep = base / "codegen_exec_results.jsonl"
    if cep.exists():
        for line in cep.open():
            r = json.loads(line)
            if r.get("score") is None: continue
            code[r["metric_id"]][r["paraphrase_idx"]][r["datapoint_id"]] = r["score"]
        print(f"loaded code exec: {sum(len(v) for v in code.values())} metric-paraphrase pairs")

    # --- Load judge scores ---
    # judge[aspect_id][paraphrase_idx][dp_id] = score (or None)
    judge = defaultdict(lambda: defaultdict(dict))
    jm_path = base / "judge_manifest.json"
    if jm_path.exists():
        jm = json.loads(jm_path.read_text())
        n_parse_fail = 0
        for entry in jm:
            rp = base / "judge_responses_llama" / f"{entry['key']}.json"
            if not rp.exists(): continue
            try:
                obj = parse_score(rp)
                for sc in obj.get("scores", []):
                    dp_id = sc.get("id"); s = sc.get("score")
                    if dp_id is None or s is None: continue
                    judge[entry["aspect_id"]][entry["paraphrase_idx"]][dp_id] = s / 10.0
            except Exception:
                n_parse_fail += 1
        print(f"loaded judge: {n_parse_fail} parse failures")

    # --- Per-aspect aggregation + analysis ---
    rows = []
    for asp in r2_aspects:
        aid = asp["aspect_id"]

        # Code: per-paraphrase mean across R1 children
        # per_para_code[paraphrase_idx][dp_id] = mean(over R1 children's code scores)
        per_para_code = defaultdict(dict)
        for r1_id in asp["r1_metric_ids"]:
            for pi in range(5):
                if pi not in code[r1_id]:
                    continue
                for dp in dp_ids:
                    s = code[r1_id][pi].get(dp)
                    if s is None: continue
                    per_para_code[pi].setdefault(dp, []).append(s)
        for pi in per_para_code:
            for dp in per_para_code[pi]:
                per_para_code[pi][dp] = mean(per_para_code[pi][dp])

        # Aggregated code per datapoint = mean across paraphrases (of mean across R1 children)
        agg_code = {}
        for dp in dp_ids:
            ps = [per_para_code[pi][dp] for pi in per_para_code if dp in per_para_code[pi]]
            if ps: agg_code[dp] = mean(ps)

        # Code paraphrase consistency: at each datapoint, std across paraphrase means
        code_para_sigma = mean([
            std([per_para_code[pi][dp] for pi in per_para_code if dp in per_para_code[pi]])
            for dp in dp_ids
            if sum(1 for pi in per_para_code if dp in per_para_code[pi]) >= 2
        ])

        # Judge: per paraphrase per dp
        per_para_judge = judge.get(aid, {})
        agg_judge = {}
        for dp in dp_ids:
            ps = [per_para_judge[pi][dp] for pi in per_para_judge if dp in per_para_judge[pi]]
            if ps: agg_judge[dp] = mean(ps)
        judge_para_sigma = mean([
            std([per_para_judge[pi][dp] for pi in per_para_judge if dp in per_para_judge[pi]])
            for dp in dp_ids
            if sum(1 for pi in per_para_judge if dp in per_para_judge[pi]) >= 2
        ])

        # Convergent validity
        paired_cv = [(agg_code[dp], agg_judge[dp]) for dp in dp_ids
                     if dp in agg_code and dp in agg_judge]
        rho_code_judge = pearson([x[0] for x in paired_cv],
                                  [x[1] for x in paired_cv]) if paired_cv else 0.0

        # Predictive validity
        code_scores = [agg_code[dp] for dp in dp_ids if dp in agg_code]
        code_labels = [labels[dp] for dp in dp_ids if dp in agg_code]
        judge_scores = [agg_judge[dp] for dp in dp_ids if dp in agg_judge]
        judge_labels = [labels[dp] for dp in dp_ids if dp in agg_judge]

        rows.append({
            "aspect_id": aid,
            "name": asp["name"],
            "n_r1_used": asp["n_r1_used"],
            "n_dp_code": len(code_scores),
            "n_dp_judge": len(judge_scores),
            "code_para_sigma": code_para_sigma,
            "judge_para_sigma": judge_para_sigma,
            "rho_code_judge": rho_code_judge,  # convergent validity
            "rho_code_label": pearson(code_scores, code_labels) if code_scores else 0.0,
            "rho_judge_label": pearson(judge_scores, judge_labels) if judge_scores else 0.0,
            "auc_code": auc(code_scores, code_labels) if code_scores else 0.5,
            "auc_judge": auc(judge_scores, judge_labels) if judge_scores else 0.5,
            "mean_code": mean(code_scores) if code_scores else 0.0,
            "mean_judge": mean(judge_scores) if judge_scores else 0.0,
        })

    # Save
    (base / "per_aspect_scores.json").write_text(json.dumps(rows, indent=1))

    # ---- Report ----
    n = len(rows)
    n_code_ok = sum(1 for r in rows if r["n_dp_code"] >= 10)
    n_judge_ok = sum(1 for r in rows if r["n_dp_judge"] >= 10)

    report = [
        "# Full validity pipeline analysis — peer-review",
        "",
        f"## {len(r2_aspects)} R2 aspects × {len(r1_metrics)} R1 metrics × "
        f"{len(datapoints)} datapoints", "",
        f"- aspects with code data on ≥10 dp: {n_code_ok}",
        f"- aspects with judge data on ≥10 dp: {n_judge_ok}",
        "",
        "## Aggregate metrics across all aspects",
        "",
        "| Quantity | Mean | Median |",
        "|---|---|---|",
    ]
    valid = [r for r in rows if r["n_dp_code"] >= 10 and r["n_dp_judge"] >= 10]
    for k in ("code_para_sigma", "judge_para_sigma", "rho_code_judge",
              "rho_code_label", "rho_judge_label", "auc_code", "auc_judge"):
        vs = [r[k] for r in valid]
        report.append(f"| {k} | {mean(vs):.3f} | "
                      f"{statistics.median(vs) if vs else 0:.3f} |")

    # Top 15 by convergent validity
    report.append("")
    report.append("## Top 15 aspects by convergent validity (code↔judge Pearson)")
    report.append("")
    report.append("| Aspect | n R1 | code↔judge ρ | code-label ρ | judge-label ρ | "
                  "AUC code | AUC judge | code σ_para | judge σ_para |")
    report.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(valid, key=lambda r: -r["rho_code_judge"])[:15]:
        report.append(f"| {r['name'][:45]} | {r['n_r1_used']} | "
                      f"{r['rho_code_judge']:+.3f} | "
                      f"{r['rho_code_label']:+.3f} | {r['rho_judge_label']:+.3f} | "
                      f"{r['auc_code']:.3f} | {r['auc_judge']:.3f} | "
                      f"{r['code_para_sigma']:.3f} | {r['judge_para_sigma']:.3f} |")

    # Top 15 by predictive validity (judge-label correlation, absolute)
    report.append("")
    report.append("## Top 15 aspects by |judge↔label| correlation (predictive)")
    report.append("")
    report.append("| Aspect | judge↔label ρ | code↔label ρ | code↔judge ρ | judge σ_para |")
    report.append("|---|---|---|---|---|")
    for r in sorted(valid, key=lambda r: -abs(r["rho_judge_label"]))[:15]:
        report.append(f"| {r['name'][:50]} | {r['rho_judge_label']:+.3f} | "
                      f"{r['rho_code_label']:+.3f} | {r['rho_code_judge']:+.3f} | "
                      f"{r['judge_para_sigma']:.3f} |")

    out_path = base / "analysis_full.md"
    out_path.write_text("\n".join(report))
    print(f"\nwrote {out_path}")
    print("\n--- aggregate ---")
    for line in report[6:14]: print(line)


if __name__ == "__main__":
    main()
