"""Rank metrics by within-method criteria:
  (a) Paraphrase σ (stability across rubric paraphrases)
  (b) Label Pearson (predictive validity)

Per technique (code or judge), produce ranked tables.

This is the "best metrics for code" and "best metrics for judge" view —
some metrics may rank well for code (lexically detectable, stable across
paraphrasing) but poorly for judge, and vice versa.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
import re


def mean(xs): return statistics.mean(xs) if xs else 0.0
def std(xs): return statistics.pstdev(xs) if len(xs) >= 2 else 0.0


def pearson(xs, ys):
    if len(xs) < 2: return 0.0
    mx, my = mean(xs), mean(ys)
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    dx = sum((x-mx)**2 for x in xs)**.5
    dy = sum((y-my)**2 for y in ys)**.5
    return num / max(dx*dy, 1e-9)


def parse_score(p):
    raw = p.read_text().strip()
    m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.S)
    if m: raw = m.group(1).strip()
    try: return json.loads(raw)
    except json.JSONDecodeError:
        s, e = raw.find("{"), raw.rfind("}")
        return json.loads(raw[s:e + 1])


def load_codes_jsonl(path, metric_field="metric_id"):
    d = defaultdict(lambda: defaultdict(dict))
    if not Path(path).exists(): return d
    for line in open(path):
        r = json.loads(line)
        if r.get("score") is None: continue
        d[r[metric_field]][r["paraphrase_idx"]][r["datapoint_id"]] = r["score"]
    return d


def load_judge_dir(base, sub_dir, manifest_file, metric_field):
    d = defaultdict(lambda: defaultdict(dict))
    mp = base / manifest_file
    if not mp.exists(): return d
    manifest = json.loads(mp.read_text())
    rdir = base / sub_dir
    for entry in manifest:
        rp = rdir / f"{entry['key']}.json"
        if not rp.exists(): continue
        try:
            obj = parse_score(rp)
            for sc in obj.get("scores", []):
                dp_id = sc.get("id"); s = sc.get("score")
                if dp_id is None or s is None: continue
                d[entry[metric_field]][entry["paraphrase_idx"]][dp_id] = s / 10.0
        except Exception:
            continue
    return d


def stats_for_method(scores, dp_ids, labels):
    """Per (metric, method): paraphrase-σ + mean score + label-ρ."""
    # Per dp, list of scores across paraphrases
    per_dp = {dp: [scores[pi].get(dp) for pi in scores if dp in scores[pi]]
              for dp in dp_ids
              if any(dp in scores[pi] for pi in scores)}
    # Paraphrase σ = mean (across dp) of std (across paraphrases at each dp)
    sigmas = [std(vs) for vs in per_dp.values() if len(vs) >= 2]
    sigma_para = mean(sigmas)
    # Per-dp mean across paraphrases = the aggregated score for label correlation
    agg = {dp: mean(vs) for dp, vs in per_dp.items()}
    xs = [agg[dp] for dp in dp_ids if dp in agg]
    ys = [labels[dp] for dp in dp_ids if dp in agg]
    label_rho = pearson(xs, ys) if xs else 0.0
    return {
        "n_dp": len(agg),
        "sigma_para": sigma_para,
        "mean_score": mean(xs) if xs else 0.0,
        "label_rho": label_rho,
        "abs_label_rho": abs(label_rho),
    }


def composite_rank(s):
    """A simple within-method quality score: high |label_rho|, low sigma_para.
    Returns a single number where bigger = better metric for this technique."""
    if s is None: return -1e9
    # Penalize σ_para; reward |label_rho|
    return s["abs_label_rho"] - 0.5 * s["sigma_para"]


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

    # Scores
    qwen_r1_codes = load_codes_jsonl(base / "codegen_exec_results_qwen_all.jsonl")
    llama_r1_codes = load_codes_jsonl(base / "codegen_exec_results.jsonl")
    qwen_r2_codes = load_codes_jsonl(base / "codegen_exec_results_qwen_r2.jsonl",
                                      metric_field="aspect_id")
    r1_judge = load_judge_dir(base, "judge_r1_responses",
                               "judge_r1_manifest.json", "metric_id")
    r2_judge = load_judge_dir(base, "judge_responses_llama",
                               "judge_manifest.json", "aspect_id")

    # === Per-R1 metric stats ===
    r1_rows = []
    for m in r1_metrics:
        mid = m["metric_id"]
        r1_rows.append({
            "metric_id": mid,
            "name": m["name"],
            "parent_aspect_id": m["parent_aspect_id"],
            "qwen_code":  stats_for_method(qwen_r1_codes[mid],  dp_ids, labels),
            "llama_code": stats_for_method(llama_r1_codes[mid], dp_ids, labels),
            "judge":      stats_for_method(r1_judge[mid],       dp_ids, labels),
        })

    # === Per-R2 aspect stats ===
    r2_rows = []
    for a in r2_aspects:
        aid = a["aspect_id"]
        r2_rows.append({
            "aspect_id": aid,
            "name": a["name"],
            "n_r1_used": a["n_r1_used"],
            "qwen_code":  stats_for_method(qwen_r2_codes[aid], dp_ids, labels),
            "judge":      stats_for_method(r2_judge[aid],      dp_ids, labels),
        })

    # === Report ===
    out = [
        "# Within-method ranking — paraphrase stability + label correlation",
        "",
        "**Quality score = |label_ρ| − 0.5 × σ_para**",
        "(rewards predictive metrics, penalizes paraphrase-sensitive ones)",
        "",
    ]

    # --- R1 metrics: best for CODE (Qwen) ---
    out += ["## Top 20 R1 metrics — best for CODE (Qwen)", "",
            "Ranked by: |code label-ρ| − 0.5 × code σ_para", "",
            "| Metric | code σ_para | code label-ρ | code mean | parent |",
            "|---|---|---|---|---|"]
    ranked = sorted(r1_rows, key=lambda r: -composite_rank(r["qwen_code"]))
    for r in ranked[:20]:
        s = r["qwen_code"]
        out.append(f"| {r['name'][:55]} | {s['sigma_para']:.3f} | "
                   f"**{s['label_rho']:+.3f}** | {s['mean_score']:.2f} | "
                   f"{r['parent_aspect_id']} |")

    # --- R1 metrics: best for JUDGE ---
    out += ["", "## Top 20 R1 metrics — best for JUDGE (Llama)", "",
            "Ranked by: |judge label-ρ| − 0.5 × judge σ_para", "",
            "| Metric | judge σ_para | judge label-ρ | judge mean | parent |",
            "|---|---|---|---|---|"]
    ranked = sorted(r1_rows, key=lambda r: -composite_rank(r["judge"]))
    for r in ranked[:20]:
        s = r["judge"]
        out.append(f"| {r['name'][:55]} | {s['sigma_para']:.3f} | "
                   f"**{s['label_rho']:+.3f}** | {s['mean_score']:.2f} | "
                   f"{r['parent_aspect_id']} |")

    # --- R2 aspects: best for CODE (Qwen R2 direct) ---
    out += ["", "## Top 20 R2 aspects — best for CODE (Qwen R2 direct)", "",
            "| Aspect | code σ_para | code label-ρ | code mean | n R1 |",
            "|---|---|---|---|---|"]
    ranked = sorted(r2_rows, key=lambda r: -composite_rank(r["qwen_code"]))
    for r in ranked[:20]:
        s = r["qwen_code"]
        out.append(f"| {r['name'][:55]} | {s['sigma_para']:.3f} | "
                   f"**{s['label_rho']:+.3f}** | {s['mean_score']:.2f} | "
                   f"{r['n_r1_used']} |")

    # --- R2 aspects: best for JUDGE ---
    out += ["", "## Top 20 R2 aspects — best for JUDGE (Llama R2)", "",
            "| Aspect | judge σ_para | judge label-ρ | judge mean | n R1 |",
            "|---|---|---|---|---|"]
    ranked = sorted(r2_rows, key=lambda r: -composite_rank(r["judge"]))
    for r in ranked[:20]:
        s = r["judge"]
        out.append(f"| {r['name'][:55]} | {s['sigma_para']:.3f} | "
                   f"**{s['label_rho']:+.3f}** | {s['mean_score']:.2f} | "
                   f"{r['n_r1_used']} |")

    # --- Aggregate stats by method ---
    out += ["", "## Aggregate paraphrase σ + label ρ by method", "",
            "| Method | mean σ_para | median σ_para | mean |label ρ| | "
            "median |label ρ| | n_metrics with σ<0.05 AND |ρ|>0.1 |",
            "|---|---|---|---|---|---|"]
    for label, rows, k in [
        ("R1 Qwen code",  r1_rows, "qwen_code"),
        ("R1 Llama code", r1_rows, "llama_code"),
        ("R1 judge",      r1_rows, "judge"),
        ("R2 Qwen code",  r2_rows, "qwen_code"),
        ("R2 judge",      r2_rows, "judge"),
    ]:
        sigmas = [r[k]["sigma_para"] for r in rows if r[k]]
        abs_rhos = [r[k]["abs_label_rho"] for r in rows if r[k]]
        gold = sum(1 for r in rows
                   if r[k] and r[k]["sigma_para"] < 0.05 and r[k]["abs_label_rho"] > 0.10)
        out.append(
            f"| {label} | {mean(sigmas):.3f} | {statistics.median(sigmas):.3f} | "
            f"{mean(abs_rhos):.3f} | {statistics.median(abs_rhos):.3f} | "
            f"{gold}/{len(rows)} |"
        )

    # --- "Code-better" vs "judge-better" metrics ---
    out += ["", "## R1 metrics where CODE beats JUDGE (and vice versa)", "",
            "Difference = |code label ρ| − |judge label ρ| (positive = code is more predictive)", "",
            "### Top 10 where CODE beats JUDGE most", "",
            "| Metric | code ρ | judge ρ | Δ |",
            "|---|---|---|---|"]
    diffs = [(r, r["qwen_code"]["abs_label_rho"] - r["judge"]["abs_label_rho"])
             for r in r1_rows if r["qwen_code"] and r["judge"]]
    for r, d in sorted(diffs, key=lambda x: -x[1])[:10]:
        out.append(f"| {r['name'][:55]} | {r['qwen_code']['label_rho']:+.3f} | "
                   f"{r['judge']['label_rho']:+.3f} | {d:+.3f} |")

    out += ["", "### Top 10 where JUDGE beats CODE most", "",
            "| Metric | code ρ | judge ρ | Δ |",
            "|---|---|---|---|"]
    for r, d in sorted(diffs, key=lambda x: x[1])[:10]:
        out.append(f"| {r['name'][:55]} | {r['qwen_code']['label_rho']:+.3f} | "
                   f"{r['judge']['label_rho']:+.3f} | {d:+.3f} |")

    out_path = base / "analysis_within_method.md"
    out_path.write_text("\n".join(out))
    print(f"wrote {out_path}")

    (base / "per_r1_within_method.json").write_text(json.dumps(r1_rows, indent=1))
    (base / "per_r2_within_method.json").write_text(json.dumps(r2_rows, indent=1))


if __name__ == "__main__":
    main()
