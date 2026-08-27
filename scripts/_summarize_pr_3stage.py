import json, numpy as np
s = json.load(open("/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/pr_3stage_summary.json"))
print("=== COMPARISON TABLE ===")
ct = s["comparison_table"]
for k, v in ct.items():
    pool = v["pool_size"]; lb = v["ladder_b_repos"]
    ab = v["ladder_a_best"]; bm = v["ladder_b_median"]
    print(f"  {k}: pool={pool}, ladder_b_repos={lb}, A_best={ab}, B_median={bm}")
print()
print("=== STAGE 0 DIAG ===")
d = s["stage0_diag"]
print("  title bot flips:", d["bot_merge_title_flips"])
print("  total title-flipped:", d["bot_merge_total_flipped"])
print("  bot authors dropped:", d["bot_authors_dropped_count"])
print("  author coverage before drop:", round(d["author_coverage_before_drop"], 3))
print("  temporal-gap repos dropped:", d["temporal_gap_n_repos"])
sample_gaps = list(d["temporal_gap_repos_dropped"].items())[:8]
print("  sample temporal gaps (months):", sample_gaps)
print()
print("=== STAGE A TOP 5 FEATURES ===")
for f in s["stageA_top5_features"]:
    name = f["feature"]; coef = f["coef"]
    print(f"  {name}: {coef:.3f}")
print()
print("=== STAGE B MATCH STATS ===")
mb = s["stageB_match_stats"]
print("  pairs:", mb["total_pairs"], "repos:", mb["n_repos_with_matches"])
print("  pairs/repo mean:", round(mb["match_rate_pairs_per_repo_mean"], 2),
      "median:", mb["match_rate_pairs_per_repo_median"])
print("  top10 repos by pairs:")
for r in mb["per_repo_stats_top10"]:
    print("   ", r)
print()
print("=== PER-REPO TRACKING ===")
prc = s["per_repo_changes"]
for label in ["prev_top5", "prev_bot5"]:
    print(f"-- {label} --")
    base = prc[f"{label}_baseline_aucs"]
    s0 = {r["repo"]: r for r in prc[f"stage0_{label}"]}
    sa = {r["repo"]: r for r in prc[f"stageA_{label}"]}
    print(f"  {'repo':40} {'base':>6} {'S0_n':>6} {'S0_auc':>8} {'SA_auc':>8}")
    for repo, base_auc in base.items():
        s0r = s0.get(repo, {"n_balanced": None, "mean_auc": None})
        sar = sa.get(repo, {"mean_auc": None})
        s0n = s0r["n_balanced"]; s0a = s0r["mean_auc"]; saa = sar["mean_auc"]
        s0a_s = f"{s0a:.3f}" if isinstance(s0a, (float, int)) and s0a is not None else "n/a"
        saa_s = f"{saa:.3f}" if isinstance(saa, (float, int)) and saa is not None else "n/a"
        s0n_s = f"{s0n}" if s0n is not None else "n/a"
        print(f"  {repo:40} {base_auc:>6.3f} {s0n_s:>6} {s0a_s:>8} {saa_s:>8}")
print()
print("=== STAGE0/STAGEA LADDER B TOP/BOT 5 ===")
for sect in ["stage0_ladder_b_top5", "stage0_ladder_b_bot5",
             "stageA_ladder_b_top5", "stageA_ladder_b_bot5"]:
    print(f"-- {sect} --")
    for r in s[sect]:
        auc = r.get("mean_auc")
        auc_s = f"{auc:.3f}" if isinstance(auc, float) else "n/a"
        print(f"  {r['repo']:45} n={r['n_balanced']} auc={auc_s}")
print()
print("=== ALL LADDER A CELLS ===")
print("Stage 0:")
for c in s["stage0_ladder_a_cells"]:
    print(f"  {c['cell']:30} mean={c['mean_auc']:.4f} sd={c['std_auc']:.4f}")
print("Stage A:")
for c in s["stageA_ladder_a_cells"]:
    print(f"  {c['cell']:30} mean={c['mean_auc']:.4f} sd={c['std_auc']:.4f}")
print("Stage B:")
for c in s["stageB_ladder_a_cells"]:
    print(f"  {c['cell']:30} mean={c['mean_auc']:.4f} sd={c['std_auc']:.4f}")
