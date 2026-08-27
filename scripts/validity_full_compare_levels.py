"""Multi-axis comparison: R1↔R1 vs R2↔R2 vs R2-aggregated for each model+method combination.

Comparisons computed:
  Per R1 metric:
    - Qwen R1 code ↔ Llama R1 judge    (direct R1, no aggregation)
    - Llama R1 code ↔ Llama R1 judge   (Llama coder for context)

  Per R2 aspect:
    - Qwen R2 code (direct) ↔ Llama R2 judge       (direct R2, no aggregation)
    - Qwen R1 aggregated  ↔ Llama R2 judge        (R1→R2 aggregation)
    - Llama R1 aggregated ↔ Llama R2 judge        (original baseline)

  Sign alignment check:
    - For each metric, mean label correlation. Flag any negative ones.

Outputs:
  runs/validity_full/<run>/analysis_comparison.md
  runs/validity_full/<run>/per_metric_full.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


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
    """{metric_id}{paraphrase_idx}{dp_id} -> score"""
    d = defaultdict(lambda: defaultdict(dict))
    if not Path(path).exists(): return d
    for line in open(path):
        r = json.loads(line)
        if r.get("score") is None: continue
        d[r[metric_field]][r["paraphrase_idx"]][r["datapoint_id"]] = r["score"]
    return d


def load_judge_dir(base, sub_dir, manifest_file, metric_field="metric_id"):
    """{metric_id_or_aspect}{paraphrase_idx}{dp_id} -> score/10"""
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
            mid = entry[metric_field]
            for sc in obj.get("scores", []):
                dp_id = sc.get("id"); s = sc.get("score")
                if dp_id is None or s is None: continue
                d[mid][entry["paraphrase_idx"]][dp_id] = s / 10.0
        except Exception:
            continue
    return d


def cross_method(method_a, method_b, dp_ids):
    """Pearson between method_a and method_b's mean-across-paraphrases scores."""
    a_per_dp = {dp: mean([method_a[pi].get(dp) for pi in method_a if dp in method_a[pi]])
                for dp in dp_ids
                if any(dp in method_a[pi] for pi in method_a)}
    b_per_dp = {dp: mean([method_b[pi].get(dp) for pi in method_b if dp in method_b[pi]])
                for dp in dp_ids
                if any(dp in method_b[pi] for pi in method_b)}
    common = sorted(set(a_per_dp) & set(b_per_dp))
    if len(common) < 10: return None, len(common)
    return pearson([a_per_dp[d] for d in common], [b_per_dp[d] for d in common]), len(common)


def code_aggregate_r2(r1_codes, r1_ids_for_aspect, dp_ids):
    """Aggregate R1 codes across children + paraphrases into a single per-dp score."""
    # For each paraphrase, mean across R1 children at each dp
    per_para = defaultdict(dict)
    for r1_id in r1_ids_for_aspect:
        for pi in range(5):
            if pi not in r1_codes[r1_id]: continue
            for dp in dp_ids:
                s = r1_codes[r1_id][pi].get(dp)
                if s is None: continue
                per_para[pi].setdefault(dp, []).append(s)
    for pi in per_para:
        for dp in per_para[pi]:
            per_para[pi][dp] = mean(per_para[pi][dp])
    return per_para


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

    # === Code scores (Llama and Qwen variants) ===
    llama_r1_codes = load_codes_jsonl(base / "codegen_exec_results.jsonl")
    qwen_r1_codes = defaultdict(lambda: defaultdict(dict))
    if (base / "codegen_exec_results_qwen_all.jsonl").exists():
        qwen_r1_codes = load_codes_jsonl(base / "codegen_exec_results_qwen_all.jsonl")
    qwen_r2_codes = defaultdict(lambda: defaultdict(dict))
    if (base / "codegen_exec_results_qwen_r2.jsonl").exists():
        qwen_r2_codes = load_codes_jsonl(base / "codegen_exec_results_qwen_r2.jsonl",
                                          metric_field="aspect_id")

    # === Judge scores ===
    r2_judge = load_judge_dir(base, "judge_responses_llama",
                               "judge_manifest.json", metric_field="aspect_id")
    r1_judge = load_judge_dir(base, "judge_r1_responses",
                               "judge_r1_manifest.json", metric_field="metric_id")

    print(f"Llama R1 codes: {len(llama_r1_codes)} R1 metrics")
    print(f"Qwen  R1 codes: {len(qwen_r1_codes)} R1 metrics")
    print(f"Qwen  R2 codes: {len(qwen_r2_codes)} R2 aspects")
    print(f"R2 judge:       {len(r2_judge)} R2 aspects")
    print(f"R1 judge:       {len(r1_judge)} R1 metrics")

    # === PER-R1-METRIC ANALYSIS (R1↔R1 DIRECT) ===
    per_r1 = []
    for m in r1_metrics:
        mid = m["metric_id"]
        row = {"metric_id": mid, "name": m["name"], "parent_aspect_id": m["parent_aspect_id"]}
        # Label corr per method (averaged across paraphrases)
        for label, codes in [("llama_code", llama_r1_codes),
                             ("qwen_code", qwen_r1_codes),
                             ("r1_judge", r1_judge)]:
            mp = codes[mid]
            scores = {dp: mean([mp[pi].get(dp) for pi in mp if dp in mp[pi]])
                      for dp in dp_ids if any(dp in mp[pi] for pi in mp)}
            if scores:
                xs = list(scores.values())
                ys = [labels[d] for d in scores]
                row[f"{label}_n"] = len(xs)
                row[f"{label}_mean"] = mean(xs)
                row[f"{label}_label_rho"] = pearson(xs, ys)
                # Paraphrase σ
                if mp:
                    per_dp_paras = [
                        [mp[pi][dp] for pi in mp if dp in mp[pi]]
                        for dp in dp_ids if any(dp in mp[pi] for pi in mp)
                    ]
                    row[f"{label}_para_sigma"] = mean([std(x) for x in per_dp_paras if len(x) >= 2])
            else:
                row[f"{label}_n"] = 0
        # Cross-method (R1 direct)
        for a_name, a in [("llama_code", llama_r1_codes), ("qwen_code", qwen_r1_codes)]:
            rho, n = cross_method(a[mid], r1_judge[mid], dp_ids)
            row[f"{a_name}_vs_r1judge_rho"] = rho
            row[f"{a_name}_vs_r1judge_n"] = n
        # Llama code vs Qwen code (within-coder agreement)
        rho, n = cross_method(llama_r1_codes[mid], qwen_r1_codes[mid], dp_ids)
        row["llama_vs_qwen_code_rho"] = rho
        per_r1.append(row)

    # === PER-R2-ASPECT ANALYSIS (R2↔R2 DIRECT + R1→R2 AGGREGATED) ===
    per_r2 = []
    for asp in r2_aspects:
        aid = asp["aspect_id"]
        row = {"aspect_id": aid, "name": asp["name"], "n_r1_used": asp["n_r1_used"]}

        # R2-direct comparisons
        for a_name, a in [("qwen_r2_code", qwen_r2_codes)]:
            rho, n = cross_method(a[aid], r2_judge[aid], dp_ids)
            row[f"{a_name}_vs_r2judge_rho"] = rho
            row[f"{a_name}_vs_r2judge_n"] = n

        # R1-aggregated comparisons (Llama + Qwen as coders)
        for coder_name, coder in [("llama_r1_agg", llama_r1_codes),
                                   ("qwen_r1_agg", qwen_r1_codes)]:
            agg = code_aggregate_r2(coder, asp["r1_metric_ids"], dp_ids)
            rho, n = cross_method(agg, r2_judge[aid], dp_ids)
            row[f"{coder_name}_vs_r2judge_rho"] = rho
            row[f"{coder_name}_vs_r2judge_n"] = n

        # Predictive: each method vs label
        for label, scores_dict in [
            ("r2_judge", r2_judge[aid]),
            ("qwen_r2_code", qwen_r2_codes[aid]),
        ]:
            sd = scores_dict
            scs = {dp: mean([sd[pi].get(dp) for pi in sd if dp in sd[pi]])
                   for dp in dp_ids if any(dp in sd[pi] for pi in sd)}
            if scs:
                xs = list(scs.values())
                ys = [labels[d] for d in scs]
                row[f"{label}_label_rho"] = pearson(xs, ys)
        per_r2.append(row)

    # === REPORT ===
    out = ["# Multi-axis validity comparison — peer-review", "",
           f"## {len(r1_metrics)} R1 metrics × {len(r2_aspects)} R2 aspects × {len(datapoints)} datapoints", ""]

    # R1↔R1 direct
    valid_r1 = [r for r in per_r1 if r.get("qwen_code_vs_r1judge_rho") is not None]
    out += ["## Direct R1↔R1 comparison (per R1 metric)", "",
            f"Aspects with code+judge data: {len(valid_r1)}", ""]
    if valid_r1:
        out.append("| Comparison | Mean ρ | Median ρ | % > 0.10 |")
        out.append("|---|---|---|---|")
        for label, key in [
            ("Qwen R1 code  ↔ Llama R1 judge", "qwen_code_vs_r1judge_rho"),
            ("Llama R1 code ↔ Llama R1 judge", "llama_code_vs_r1judge_rho"),
            ("Llama vs Qwen codes (agreement)", "llama_vs_qwen_code_rho"),
        ]:
            rs = [r[key] for r in valid_r1 if r.get(key) is not None]
            n_strong = sum(1 for r in rs if r > 0.10)
            med = statistics.median(rs) if rs else 0.0
            out.append(f"| {label} | {mean(rs):+.3f} | {med:+.3f} | "
                       f"{n_strong}/{len(rs)} ({100*n_strong/max(len(rs),1):.0f}%) |")

    # R2↔R2 direct
    valid_r2 = [r for r in per_r2 if r.get("qwen_r2_code_vs_r2judge_rho") is not None]
    out += ["", "## Direct R2↔R2 + aggregated R1→R2 (per R2 aspect)", "",
            f"Aspects with both code+judge: {len(valid_r2)}", ""]
    if valid_r2:
        out.append("| Comparison | Mean ρ | Median ρ | % > 0.10 |")
        out.append("|---|---|---|---|")
        for label, key in [
            ("Qwen R2 direct code ↔ R2 judge", "qwen_r2_code_vs_r2judge_rho"),
            ("Qwen R1 agg ↔ R2 judge",          "qwen_r1_agg_vs_r2judge_rho"),
            ("Llama R1 agg ↔ R2 judge",         "llama_r1_agg_vs_r2judge_rho"),
        ]:
            rs = [r[key] for r in valid_r2 if r.get(key) is not None]
            n_strong = sum(1 for r in rs if r > 0.10)
            out.append(f"| {label} | {mean(rs):+.3f} | {statistics.median(rs):+.3f} | "
                       f"{n_strong}/{len(rs)} ({100*n_strong/max(len(rs),1):.0f}%) |")

    # Predictive
    out += ["", "## Predictive (vs accept/reject label)", ""]
    out.append("| Method | Mean |ρ| | Median |ρ| | % > 0.10 |")
    out.append("|---|---|---|---|")
    for label, key, source in [
        ("Llama R1 code (R1 metric → label)", "llama_code_label_rho", per_r1),
        ("Qwen  R1 code (R1 metric → label)", "qwen_code_label_rho",  per_r1),
        ("Llama R1 judge (R1 metric → label)", "r1_judge_label_rho",   per_r1),
        ("Qwen  R2 code (R2 aspect → label)", "qwen_r2_code_label_rho", per_r2),
        ("Llama R2 judge (R2 aspect → label)", "r2_judge_label_rho",   per_r2),
    ]:
        rs = [abs(r[key]) for r in source if r.get(key) is not None]
        n_strong = sum(1 for r in rs if r > 0.10)
        med = statistics.median(rs) if rs else 0.0
        out.append(f"| {label} | {mean(rs):.3f} | {med:.3f} | "
                   f"{n_strong}/{len(rs)} ({100*n_strong/max(len(rs),1):.0f}%) |")

    # Top 15 per axis
    if valid_r1:
        out += ["", "## Top 15 R1 metrics by R1↔R1 convergent validity", "",
                "| Metric | Qwen↔judge ρ | Llama↔judge ρ | parent aspect |",
                "|---|---|---|---|"]
        for r in sorted(valid_r1, key=lambda r: -r["qwen_code_vs_r1judge_rho"])[:15]:
            out.append(f"| {r['name'][:50]} | {r['qwen_code_vs_r1judge_rho']:+.3f} | "
                       f"{r.get('llama_code_vs_r1judge_rho', 0):+.3f} | "
                       f"{r['parent_aspect_id']} |")

    if valid_r2:
        out += ["", "## Top 15 R2 aspects by Qwen R2 direct ↔ R2 judge", "",
                "| Aspect | Qwen R2 direct | Qwen R1 agg | Llama R1 agg | n R1 |",
                "|---|---|---|---|---|"]
        for r in sorted(valid_r2, key=lambda r: -r["qwen_r2_code_vs_r2judge_rho"])[:15]:
            out.append(f"| {r['name'][:50]} | {r['qwen_r2_code_vs_r2judge_rho']:+.3f} | "
                       f"{r.get('qwen_r1_agg_vs_r2judge_rho', 0):+.3f} | "
                       f"{r.get('llama_r1_agg_vs_r2judge_rho', 0):+.3f} | "
                       f"{r['n_r1_used']} |")

    out_path = base / "analysis_comparison.md"
    out_path.write_text("\n".join(out))
    (base / "per_r1_full.json").write_text(json.dumps(per_r1, indent=1))
    (base / "per_r2_full.json").write_text(json.dumps(per_r2, indent=1))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
