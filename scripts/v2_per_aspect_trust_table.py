"""Per (task, aspect): compute ρ(llama_bf16, claude) and ρ(code, claude).

For each aspect, code has 3 variants (v0_keyword, v1_structure, v2_holistic).
We aggregate code by taking the **max ρ across variants** — most generous proxy
("if any variant agrees with Claude, we trust code").

Output: outputs/v2_analysis/per_aspect_trust.csv with columns:
  task, aspect_id, n_paired_claude, rho_llama_vs_claude, rho_code_vs_claude
"""
import json, sys
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr

DB = Path("outputs/v2_db/cells_v1")
MIN_PAIRED = 10  # need at least 10 paired observations to compute ρ

def load(task, judge):
    for ext in ["data.parquet", "data.csv.gz"]:
        p = DB / f"task={task}/judge={judge}" / ext
        if p.exists():
            return pd.read_parquet(p) if ext.endswith("parquet") else pd.read_csv(p, compression="gzip")
    return pd.DataFrame()


def to_score_idx(df):
    """(dp, aspect) → numeric score (applicable=True only). Take first dup."""
    if "paraphrase_idx" in df.columns:
        df = df[df["paraphrase_idx"] == 0]
    df = df[df["score"].notna()]
    return df.drop_duplicates(["datapoint_id", "aspect_id"]).set_index(["datapoint_id", "aspect_id"])["score"]


def code_scores_for_task(task: str):
    """Load codegen_exec_results.jsonl, aggregate by max across variants per aspect.

    Returns: {aspect_id: {dp_id: score}} dict.
    Score is averaged across variants (each variant has score in [0,1]).
    """
    f = Path(f"runs/validity_full/v2/{task}/codegen_exec_results.jsonl")
    if not f.exists(): return {}
    # JSONL: {"aspect_id":"a0","variant":"v0_keyword","datapoint_id":"d00042","score":0.43,"error":null}
    per_aspect_dp_vals = {}  # aspect → dp → list of variant scores
    with open(f) as fp:
        for line in fp:
            try: r = json.loads(line)
            except: continue
            if r.get("error"): continue
            a = r["aspect_id"]; dp = r["datapoint_id"]; sc = r.get("score")
            if sc is None: continue
            per_aspect_dp_vals.setdefault(a, {}).setdefault(dp, []).append(sc)
    # Aggregate by mean
    return {a: {dp: sum(vs)/len(vs) for dp, vs in dp_vals.items()}
            for a, dp_vals in per_aspect_dp_vals.items()}


def analyze_task(task: str):
    print(f"--- {task} ---")
    cl = to_score_idx(load(task, "claude"))
    bf = to_score_idx(load(task, "llama_bf16"))
    code = code_scores_for_task(task)

    rows = []
    aspects = set(a for _, a in cl.index)
    for aspect in aspects:
        # llama vs claude
        cl_a = cl[cl.index.get_level_values(1) == aspect]
        bf_a = bf[bf.index.get_level_values(1) == aspect] if len(bf) > 0 else pd.Series(dtype=float)
        common_llama = cl_a.index.intersection(bf_a.index)
        rho_llama = None
        if len(common_llama) >= MIN_PAIRED:
            x, y = cl_a.loc[common_llama].values, bf_a.loc[common_llama].values
            if len(set(x)) > 1 and len(set(y)) > 1:
                rho_llama = float(spearmanr(x, y).statistic)
        # code vs claude
        code_aspect = code.get(aspect, {})
        rho_code = None
        if code_aspect:
            paired = [(cl_a.loc[(dp, aspect)], code_aspect[dp])
                      for dp in code_aspect.keys()
                      if (dp, aspect) in cl_a.index]
            if len(paired) >= MIN_PAIRED:
                x, y = zip(*paired)
                if len(set(x)) > 1 and len(set(y)) > 1:
                    rho_code = float(spearmanr(x, y).statistic)
        rows.append({
            "task": task, "aspect_id": aspect,
            "n_paired_llama": int(len(common_llama)),
            "rho_llama_vs_claude": rho_llama,
            "n_paired_code": len(code_aspect) if code_aspect else 0,
            "rho_code_vs_claude": rho_code,
        })
    df = pd.DataFrame(rows)
    print(f"  {len(df)} aspects testable; "
          f"ρ(llama,cl)≥0.5: {(df['rho_llama_vs_claude']>=0.5).sum()}, "
          f"ρ(code,cl)≥0.5: {(df['rho_code_vs_claude']>=0.5).sum()}")
    return df


def main():
    TASKS = ["peer_review", "math", "notice_and_comment", "press_releases",
             "humor", "news_homepages", "patents", "code_review", "creative_writing"]
    all_dfs = []
    for t in TASKS:
        try:
            all_dfs.append(analyze_task(t))
        except Exception as e:
            print(f"  {t}: ERR {e}")
    big = pd.concat(all_dfs, ignore_index=True)
    out = Path("outputs/v2_analysis/per_aspect_trust.csv")
    big.to_csv(out, index=False)
    print(f"\nWrote {len(big)} rows to {out}")
    print(f"\nGlobal summary:")
    print(f"  Aspects with ρ(llama, claude) ≥ 0.5: {(big['rho_llama_vs_claude']>=0.5).sum()}/{len(big)} ({(big['rho_llama_vs_claude']>=0.5).mean()*100:.1f}%)")
    print(f"  Aspects with ρ(code, claude)  ≥ 0.5: {(big['rho_code_vs_claude']>=0.5).sum()}/{len(big)} ({(big['rho_code_vs_claude']>=0.5).mean()*100:.1f}%)")
    print(f"  Aspects with EITHER ρ ≥ 0.5: {((big['rho_llama_vs_claude']>=0.5) | (big['rho_code_vs_claude']>=0.5)).sum()}/{len(big)}")


if __name__ == "__main__":
    main()
