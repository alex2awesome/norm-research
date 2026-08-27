"""Code-only intermediate analysis (judge still running).

Shows:
  - paraphrase convergence (how stable are scores across 5 wordings of same rubric?)
  - correlation with judgement label per aspect
  - AUC for predicting accept/reject
  - top/bottom aspects by code-label correlation
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path


def mean(xs):
    return statistics.mean(xs) if xs else 0.0


def main():
    base = Path("runs/validity_full/full_v1")
    data = json.loads((base / "per_aspect_scores.json").read_text())
    valid = [r for r in data if r["n_dp_code"] >= 10]

    # Aggregate stats
    out = ["# Code-only intermediate report (judge pending)", "",
           f"## {len(valid)} aspects analyzed (peer-review, 500 datapoints)", ""]

    para_sigmas = [r["code_para_sigma"] for r in valid]
    rho_labels = [r["rho_code_label"] for r in valid]
    abs_rho_labels = [abs(r["rho_code_label"]) for r in valid]
    aucs = [r["auc_code"] for r in valid]
    aucs_dev = [abs(r["auc_code"] - 0.5) for r in valid]

    out.append("## Paraphrase convergence (code-side)")
    out.append("")
    out.append("Across 5 paraphrases of the same rubric (aggregated across R1 children "
               "per aspect), compute std of mean scores per datapoint. Low σ → high stability.")
    out.append("")
    out.append("| σ_para | count |")
    out.append("|---|---|")
    bins = [(0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 1.0)]
    for lo, hi in bins:
        n = sum(1 for s in para_sigmas if lo <= s < hi)
        out.append(f"| [{lo:.2f}, {hi:.2f}) | {n} |")
    out.append(f"\nMean σ_para = {mean(para_sigmas):.3f}, median = {statistics.median(para_sigmas):.3f}")

    out.append("")
    out.append("## Predictive validity vs accept/reject label")
    out.append("")
    out.append(f"Mean |Pearson(code, label)| = {mean(abs_rho_labels):.3f}")
    out.append(f"Mean AUC = {mean(aucs):.3f} (deviation from 0.5: {mean(aucs_dev):.3f})")
    out.append(f"Aspects with |ρ| > 0.1 vs label: {sum(1 for r in abs_rho_labels if r > 0.1)}")
    out.append(f"Aspects with AUC > 0.55 or < 0.45: {sum(1 for r in aucs_dev if r > 0.05)}")

    # Top 20 aspects by absolute label correlation
    out.append("")
    out.append("## Top 20 aspects: |Pearson(code, accept/reject)|")
    out.append("")
    out.append("| Aspect | n R1 | code↔label ρ | AUC | code σ_para | mean code |")
    out.append("|---|---|---|---|---|---|")
    for r in sorted(valid, key=lambda r: -abs(r["rho_code_label"]))[:20]:
        out.append(f"| {r['name'][:48]} | {r['n_r1_used']} | "
                   f"**{r['rho_code_label']:+.3f}** | {r['auc_code']:.3f} | "
                   f"{r['code_para_sigma']:.3f} | {r['mean_code']:.2f} |")

    # Top 10 most-stable paraphrases
    out.append("")
    out.append("## Top 10 most paraphrase-stable aspects (low σ_para)")
    out.append("")
    out.append("| Aspect | n R1 | σ_para | code↔label ρ |")
    out.append("|---|---|---|---|")
    for r in sorted(valid, key=lambda r: r["code_para_sigma"])[:10]:
        out.append(f"| {r['name'][:48]} | {r['n_r1_used']} | "
                   f"{r['code_para_sigma']:.3f} | {r['rho_code_label']:+.3f} |")

    out_path = base / "code_only_report.md"
    out_path.write_text("\n".join(out))
    print(f"wrote {out_path}")
    print("\n".join(out[:25]))


if __name__ == "__main__":
    main()
