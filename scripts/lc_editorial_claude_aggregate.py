"""Parse Claude shard responses into a single labeled parquet + run analysis."""
import json, re, sys, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
SAMPLE_PARQUET = ROOT / "outputs/v2_analysis/lc_editorial_judging_sample_1500.parquet"
WORK_DIR = ROOT / "outputs/v2_analysis/claude_labeling_work"
OUT_LABELS = ROOT / "outputs/v2_analysis/lc_editorial_judging_claude_1500.parquet"
OUT_COMPARE = ROOT / "outputs/v2_analysis/lc_editorial_judging_comparison.json"


def strip_fences(s):
    s = s.strip()
    # remove leading/trailing ```json ... ``` fences
    m = re.match(r"^```(?:json)?\s*\n(.*)\n```\s*$", s, re.DOTALL)
    if m:
        return m.group(1)
    return s


def extract_json_array(s):
    """Extract the first top-level JSON array from s."""
    s = strip_fences(s)
    # find first '['
    start = s.find("[")
    if start == -1:
        return None
    # find matching ']' by depth count, respecting strings
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
    return None


def parse_shard(path):
    raw = Path(path).read_text()
    if not raw.strip():
        return [], "empty_output"
    arr_text = extract_json_array(raw)
    if arr_text is None:
        # try parsing line-by-line as JSONL
        items = []
        for line in raw.splitlines():
            line = line.strip().rstrip(",")
            if not line or line in ("[", "]"):
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
        if items:
            return items, "jsonl_fallback"
        return [], "no_json"
    try:
        arr = json.loads(arr_text)
        return arr, "ok"
    except Exception as e:
        # try greedy repair: split by '},'
        return [], f"json_error:{e}"


def main():
    df = pd.read_parquet(SAMPLE_PARQUET)
    print(f"sample rows: {len(df)}")

    shard_files = sorted(WORK_DIR.glob("shard_*_response.txt"))
    print(f"shard files: {len(shard_files)}")

    all_items = []
    parse_status = {}
    for sf in shard_files:
        items, status = parse_shard(sf)
        parse_status[sf.name] = {"n": len(items), "status": status}
        all_items.extend(items)
        print(f"  {sf.name}: {len(items)} items ({status})")

    # Dedup by pair_id (last wins, but they should be unique)
    by_pid = {}
    bad = 0
    for it in all_items:
        if not isinstance(it, dict):
            bad += 1
            continue
        pid = it.get("pair_id")
        lab = it.get("label")
        if pid is None or lab is None:
            bad += 1
            continue
        try:
            lab = int(lab)
        except Exception:
            bad += 1
            continue
        if lab not in (0, 1, 2, 3):
            bad += 1
            continue
        by_pid[pid] = {"pair_id": pid, "claude_label": lab, "claude_reason": it.get("reason", "")}
    print(f"valid labels: {len(by_pid)}, malformed: {bad}")

    labels_df = pd.DataFrame(list(by_pid.values()))
    labels_df.to_parquet(OUT_LABELS, index=False)
    print(f"wrote {OUT_LABELS}")

    # Merge for analysis
    merged = df.merge(labels_df, on="pair_id", how="left")
    n_total = len(merged)
    n_claude = merged["claude_label"].notna().sum()
    n_llama = merged["llama_label"].notna().sum()
    print(f"\nmerged: {n_total} rows, claude_labeled={n_claude}, llama_labeled={n_llama}")
    missing_claude = merged[merged["claude_label"].isna()]["pair_id"].tolist()
    print(f"missing claude labels: {len(missing_claude)}")
    if missing_claude[:20]:
        print(f"  first missing: {missing_claude[:20]}")

    # ============= ANALYSIS =============
    out = {
        "n_pairs_total": int(n_total),
        "n_claude_labeled": int(n_claude),
        "n_llama_labeled": int(n_llama),
        "parse_status_by_shard": parse_status,
        "missing_claude_pair_ids": missing_claude,
    }

    # Label distributions
    out["claude_label_dist"] = merged["claude_label"].dropna().astype(int).value_counts().sort_index().to_dict()
    out["llama_label_dist"] = merged["llama_label"].dropna().astype(int).value_counts().sort_index().to_dict()

    # === A. Claude vs Llama on overlap ===
    overlap = merged[merged["llama_label"].notna() & merged["claude_label"].notna()].copy()
    overlap["llama_label"] = overlap["llama_label"].astype(int)
    overlap["claude_label"] = overlap["claude_label"].astype(int)
    n_ov = len(overlap)
    out["overlap_n"] = int(n_ov)

    # Confusion matrix
    cm = {}
    for l in (0, 1, 2, 3):
        cm[str(l)] = {}
        for c in (0, 1, 2, 3):
            cm[str(l)][str(c)] = int(((overlap["llama_label"] == l) & (overlap["claude_label"] == c)).sum())
    out["confusion_llama_rows_claude_cols"] = cm

    # Cohen's kappa
    def cohen_kappa(y1, y2, K=4, weighted=False):
        n = len(y1)
        if n == 0:
            return None
        cm = np.zeros((K, K))
        for a, b in zip(y1, y2):
            cm[int(a), int(b)] += 1
        Po = 0.0
        Pe = 0.0
        if weighted:
            # linear weights
            w = np.zeros((K, K))
            for i in range(K):
                for j in range(K):
                    w[i, j] = 1 - abs(i - j) / (K - 1)
            r_marg = cm.sum(axis=1) / n
            c_marg = cm.sum(axis=0) / n
            Po = (w * cm / n).sum()
            Pe = (w * np.outer(r_marg, c_marg)).sum()
        else:
            Po = np.trace(cm) / n
            r_marg = cm.sum(axis=1) / n
            c_marg = cm.sum(axis=0) / n
            Pe = (r_marg * c_marg).sum()
        if Pe == 1:
            return 1.0
        return float((Po - Pe) / (1 - Pe))

    out["exact_agreement_rate"] = float((overlap["llama_label"] == overlap["claude_label"]).mean()) if n_ov else None
    out["cohen_kappa_unweighted"] = cohen_kappa(overlap["llama_label"].values, overlap["claude_label"].values, K=4, weighted=False)
    out["cohen_kappa_linear_weighted"] = cohen_kappa(overlap["llama_label"].values, overlap["claude_label"].values, K=4, weighted=True)

    # Systematic shift: where do they disagree?
    disagree = overlap[overlap["llama_label"] != overlap["claude_label"]]
    out["disagreement_n"] = int(len(disagree))
    shift_keys = []
    for k, v in cm.items():
        for c, n in v.items():
            if k != c and n > 0:
                shift_keys.append({"llama": int(k), "claude": int(c), "n": n})
    shift_keys.sort(key=lambda r: -r["n"])
    out["top_disagreement_cells"] = shift_keys[:10]

    # === B. Claude vs cosine on full 1500 ===
    cl = merged[merged["claude_label"].notna()].copy()
    cl["claude_label"] = cl["claude_label"].astype(int)

    from scipy.stats import spearmanr
    rho, p = spearmanr(cl["max_sim"], cl["claude_label"])
    out["spearman_max_sim_vs_claude_label"] = {"rho": float(rho), "p": float(p), "n": int(len(cl))}

    out["mean_sim_per_claude_label"] = {str(l): float(cl[cl["claude_label"] == l]["max_sim"].mean()) for l in sorted(cl["claude_label"].unique())}
    out["claude_label_counts"] = cl["claude_label"].value_counts().sort_index().to_dict()

    # AUC: claude=0 vs claude=2
    from sklearn.metrics import roc_auc_score
    sub = cl[cl["claude_label"].isin([0, 2])]
    if len(sub) > 0 and sub["claude_label"].nunique() == 2:
        y = (sub["claude_label"] == 0).astype(int)
        out["auc_claude0_vs_claude2"] = {
            "n": int(len(sub)),
            "auc": float(roc_auc_score(y, sub["max_sim"])),
        }
    sub2 = cl[cl["claude_label"].isin([0, 1, 2])]
    if len(sub2) > 0:
        y = (sub2["claude_label"] == 0).astype(int)
        if y.nunique() == 2:
            out["auc_claude0_vs_claude_ge1"] = {
                "n": int(len(sub2)),
                "auc": float(roc_auc_score(y, sub2["max_sim"])),
            }

    # Threshold sweep
    thr_table = []
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.94]:
        sub = cl[cl["max_sim"] >= thr]
        n = len(sub)
        if n == 0:
            thr_table.append({"thr": thr, "n": 0})
            continue
        row = {
            "thr": thr,
            "n": int(n),
            "frac_same": float((sub["claude_label"] == 0).mean()),
            "frac_partial": float((sub["claude_label"] == 1).mean()),
            "frac_diff": float((sub["claude_label"] == 2).mean()),
            "frac_junk": float((sub["claude_label"] == 3).mean()),
        }
        thr_table.append(row)
    out["threshold_band_table_claude"] = thr_table

    # === C. Side-by-side overlap comparison (Llama vs Claude same 300) ===
    # Recompute Llama metrics on the same overlap set using sample's max_sim
    ll = overlap.copy()
    rho_l, p_l = spearmanr(ll["max_sim"], ll["llama_label"])
    rho_c, p_c = spearmanr(ll["max_sim"], ll["claude_label"])
    sxs = {
        "n_overlap": int(len(ll)),
        "spearman_llama": {"rho": float(rho_l), "p": float(p_l)},
        "spearman_claude": {"rho": float(rho_c), "p": float(p_c)},
    }
    for judge_col, judge_name in [("llama_label", "llama"), ("claude_label", "claude")]:
        sub = ll[ll[judge_col].isin([0, 2])]
        if len(sub) > 0 and sub[judge_col].nunique() == 2:
            y = (sub[judge_col] == 0).astype(int)
            sxs[f"auc_0v2_{judge_name}"] = float(roc_auc_score(y, sub["max_sim"]))
        sub2 = ll[ll[judge_col].isin([0, 1, 2])]
        if len(sub2) > 0:
            y = (sub2[judge_col] == 0).astype(int)
            if y.nunique() == 2:
                sxs[f"auc_0v_ge1_{judge_name}"] = float(roc_auc_score(y, sub2["max_sim"]))
        sxs[f"frac_label0_{judge_name}"] = float((ll[judge_col] == 0).mean())
        sxs[f"frac_label3_{judge_name}"] = float((ll[judge_col] == 3).mean())
    out["overlap_side_by_side"] = sxs

    # === D. Junk detection ===
    junk = {
        "claude_junk_total": int((cl["claude_label"] == 3).sum()),
        "claude_junk_mean_sim": float(cl[cl["claude_label"] == 3]["max_sim"].mean()) if (cl["claude_label"] == 3).any() else None,
        "claude_junk_pct_below_0p5": float((cl[cl["claude_label"] == 3]["max_sim"] < 0.5).mean()) if (cl["claude_label"] == 3).any() else None,
    }
    # Overlap: same pairs flagged junk?
    ll_junk = set(overlap[overlap["llama_label"] == 3]["pair_id"].tolist())
    cl_junk = set(overlap[overlap["claude_label"] == 3]["pair_id"].tolist())
    junk["overlap_llama_junk_n"] = len(ll_junk)
    junk["overlap_claude_junk_n"] = len(cl_junk)
    junk["overlap_both_junk_n"] = len(ll_junk & cl_junk)
    junk["overlap_llama_only_junk"] = sorted(ll_junk - cl_junk)
    junk["overlap_claude_only_junk"] = sorted(cl_junk - ll_junk)
    out["junk_detection"] = junk

    OUT_COMPARE.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote comparison: {OUT_COMPARE}")
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, (dict, list)) or k in ("overlap_side_by_side", "spearman_max_sim_vs_claude_label", "junk_detection", "auc_claude0_vs_claude2", "auc_claude0_vs_claude_ge1", "exact_agreement_rate", "cohen_kappa_unweighted", "cohen_kappa_linear_weighted")}, indent=2, default=str))


if __name__ == "__main__":
    main()
