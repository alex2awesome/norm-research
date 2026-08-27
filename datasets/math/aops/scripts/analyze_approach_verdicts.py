"""Join LLM approach-judge verdicts to de-confounded thanks and test
novel_approach / elegance / approach-match against thanks_resid,
within-problem, vs the established nulls (similarity AUC 0.500,
editorial-register p=0.48, length AUC 0.594)."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

BASE = Path(__file__).resolve().parents[1]

rows = [json.loads(l) for l in open(BASE / "approach_verdicts.jsonl")]
v = pd.DataFrame([r for r in rows if "judge_failed" not in r])
print(f"verdict rows: {len(v)} (of {len(rows)} total)")

td = pd.read_parquet(BASE / "thanks_deconfounded.parquet")
es = pd.read_parquet(BASE / "editorial_similarity.parquet")[
    ["post_id", "body_noquote"]
]
es["log_len"] = np.log1p(es.body_noquote.str.len())

df = v.merge(
    td[["post_id", "topic_id", "problem_n", "variant", "thanks_resid",
        "thanks", "is_correct", "has_answer"]],
    on="post_id", suffixes=("", "_td"),
).merge(es, on="post_id", how="left")
df["problem"] = df.variant.astype(str) + "#" + df.problem_n_td.astype(str)
print(f"joined: {len(df)}")

# clean fields
df["elegance"] = pd.to_numeric(df.elegance, errors="coerce")
for c in ["same_approach", "novel_approach"]:
    df[c] = df[c].map({True: 1, False: 0, "true": 1, "false": 0})
df["matched"] = df.matched_solution.notna().astype(int)

print("\n=== DISTRIBUTIONS ===")
print("elegance:", df.elegance.value_counts().sort_index().to_dict())
print(f"same_approach rate: {df.same_approach.mean():.3f}  "
      f"novel_approach rate: {df.novel_approach.mean():.3f}  "
      f"matched-to-wiki rate: {df.matched.mean():.3f}")
print(f"salvaged: {df.get('salvaged', pd.Series(dtype=float)).fillna(False).sum():.0f}")
print("top approach_labels:",
      df.approach_label.str.lower().value_counts().head(10).to_dict())

def within_problem_rho(d, xcol, ycol="thanks_resid"):
    rhos = []
    for _, g in d.groupby("problem"):
        g = g.dropna(subset=[xcol, ycol])
        if len(g) >= 3 and g[xcol].nunique() > 1:
            r = stats.spearmanr(g[xcol], g[ycol]).statistic
            if np.isfinite(r):
                rhos.append(r)
    rhos = np.array(rhos)
    if len(rhos) < 10:
        return None
    w = stats.wilcoxon(rhos)
    return dict(n_problems=len(rhos), mean=rhos.mean(),
                median=np.median(rhos), p=w.pvalue)

def pooled(d, xcol, ycol="thanks_resid"):
    g = d.dropna(subset=[xcol, ycol])
    r = stats.spearmanr(g[xcol], g[ycol])
    return f"pooled rho={r.statistic:+.4f} (p={r.pvalue:.2g}, n={len(g)})"

def auc_binary(d, bcol, ycol="thanks_resid"):
    g = d.dropna(subset=[bcol, ycol])
    a, b = g.loc[g[bcol] == 1, ycol], g.loc[g[bcol] == 0, ycol]
    if len(a) < 30 or len(b) < 30:
        return None
    u = stats.mannwhitneyu(a, b)
    return dict(auc=u.statistic / (len(a) * len(b)), p=u.pvalue,
                n1=len(a), n0=len(b))

def report(name, d):
    print(f"\n--- {name} (n={len(d)}) ---")
    for col in ["elegance", "novel_approach", "same_approach", "matched"]:
        if d[col].nunique(dropna=True) < 2:
            continue
        print(f"{col:15s} {pooled(d, col)}")
        wp = within_problem_rho(d, col)
        if wp:
            print(f"{'':15s} within-problem mean rho={wp['mean']:+.4f} "
                  f"(median {wp['median']:+.4f}, n_prob={wp['n_problems']}, "
                  f"wilcoxon p={wp['p']:.2g})")
        if col != "elegance":
            ab = auc_binary(d, col)
            if ab:
                print(f"{'':15s} AUC({col}=1 vs 0) = {ab['auc']:.3f} "
                      f"(p={ab['p']:.2g}, n1={ab['n1']}, n0={ab['n0']})")

print("\n=== MAIN TESTS: verdict fields vs thanks_resid ===")
report("all judged", df)
vc = df[df.is_correct.fillna(False).astype(bool)]
report("verified-correct subset", vc)

print("\n=== ELEGANCE vs LENGTH (the one live signal) ===")
g = df.dropna(subset=["elegance", "log_len"])
r = stats.spearmanr(g.elegance, g.log_len)
print(f"elegance ~ log_len: rho={r.statistic:+.4f} (p={r.pvalue:.2g})")
# does elegance add anything beyond length? partial spearman via ranks
for sub, name in [(df, "all"), (vc, "verified-correct")]:
    g = sub.dropna(subset=["elegance", "log_len", "thanks_resid"])
    if len(g) < 100:
        continue
    rk = g[["elegance", "log_len", "thanks_resid"]].rank()
    res_x = rk.elegance - np.polyval(np.polyfit(rk.log_len, rk.elegance, 1), rk.log_len)
    res_y = rk.thanks_resid - np.polyval(np.polyfit(rk.log_len, rk.thanks_resid, 1), rk.log_len)
    r = stats.spearmanr(res_x, res_y)
    print(f"[{name}] elegance ~ thanks_resid | log_len: "
          f"partial rho={r.statistic:+.4f} (p={r.pvalue:.2g}, n={len(g)})")

print("\n=== ELEGANCE top-vs-bottom AUC (parallel to length AUC 0.594) ===")
for sub, name in [(df, "all"), (vc, "verified-correct")]:
    g = sub.dropna(subset=["elegance", "thanks_resid"])
    hi, lo = g[g.elegance >= 4], g[g.elegance <= 2]
    if len(hi) > 30 and len(lo) > 30:
        u = stats.mannwhitneyu(hi.thanks_resid, lo.thanks_resid)
        print(f"[{name}] eleg>=4 (n={len(hi)}) vs <=2 (n={len(lo)}): "
              f"AUC={u.statistic/(len(hi)*len(lo)):.3f} (p={u.pvalue:.2g})")

out = BASE / "approach_verdicts_joined.parquet"
df.to_parquet(out)
print(f"\n[saved] {out}")
