"""Rebuild the Phase-1 stratified subsample (parity with stratified_auc.py).

Output columns: pair_id, platform, decile, qwen_label, editorial_text, candidate_text,
                editorial_lang, candidate_lang

LEAKAGE GUARD: cosine is used ONLY for decile bucketing (stratification).
It is NEVER written as a feature; consumers must not read it as a feature.
"""
import json
import re
import numpy as np
import pandas as pd

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
OUT = f"{ROOT}/outputs/v2_analysis"

POOL = f"{OUT}/comp_qwen_phase1_pool.parquet"
LABELS = f"{OUT}/comp_qwen_phase1_labels.parquet"
COSINE = f"{OUT}/comp_qwen_phase1_cosine.parquet"

OUT_PARQUET = f"{OUT}/comp_qwen_phase1_stratified_subsample.parquet"

RNG = 42
DECILES = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.000001)]
DECILE_LABELS = ["[0.0-0.2)", "[0.2-0.4)", "[0.4-0.6)", "[0.6-0.8)", "[0.8-1.0]"]


def assign_decile(c):
    for i, (lo, hi) in enumerate(DECILES):
        if c >= lo and c < hi:
            return i
    return 4


# very light language heuristics for code; informational only
_LANG_PATTERNS = [
    ("python", re.compile(r"\b(def |import |from .+ import |print\(|self\.)")),
    ("cpp", re.compile(r"#include\s*<|std::|using\s+namespace\s+std|cin\s*>>|cout\s*<<")),
    ("java", re.compile(r"public\s+class\s+\w+|System\.out\.println|public\s+static\s+void\s+main")),
    ("c", re.compile(r"#include\s*<stdio\.h>|printf\(|scanf\(")),
    ("csharp", re.compile(r"using\s+System;|namespace\s+\w+|Console\.WriteLine")),
    ("javascript", re.compile(r"\bconst\s+\w+\s*=|\blet\s+\w+\s*=|console\.log\(|=>")),
    ("go", re.compile(r"\bfunc\s+\w+\s*\(|package\s+main|fmt\.Println")),
    ("rust", re.compile(r"\bfn\s+\w+\s*\(|let\s+mut\s+|::<|println!\(")),
    ("kotlin", re.compile(r"\bfun\s+\w+\s*\(|val\s+\w+\s*=|var\s+\w+\s*=")),
    ("ruby", re.compile(r"\bdef\s+\w+\b|puts\s+|end\b")),
    ("php", re.compile(r"<\?php|\$\w+\s*=")),
]


def detect_lang(code: str) -> str:
    if not isinstance(code, str) or not code.strip():
        return "unknown"
    for name, pat in _LANG_PATTERNS:
        if pat.search(code):
            return name
    # fallback by braces
    if "{" in code and "}" in code:
        return "cstyle"
    return "unknown"


def main():
    print("loading parquets...", flush=True)
    pool = pd.read_parquet(POOL)
    labels = pd.read_parquet(LABELS)[["pair_id", "qwen_label", "status"]]
    cos = pd.read_parquet(COSINE)

    df = pool.merge(labels, on="pair_id", how="left")
    df = df.merge(cos, on="pair_id", how="left")
    df = df[df["status"].isin(["ok", "regex_verdict"])].copy()
    df = df[df["cosine"].notna()].copy()
    df = df.reset_index(drop=True)
    df["qwen_label"] = df["qwen_label"].astype(int)
    df["decile"] = df["cosine"].apply(assign_decile).astype(int)

    print(f"usable rows: {len(df)}", flush=True)
    print(df.groupby("platform")["qwen_label"].value_counts().to_dict(), flush=True)

    rng = np.random.default_rng(RNG)
    chosen_idx = []
    summary = {}
    for plat in ["lc", "luogu"]:
        per_decile_n = {}
        for d_idx, d_label in enumerate(DECILE_LABELS):
            sel = df.index[(df["platform"] == plat) & (df["decile"] == d_idx)].to_numpy()
            n = min(500, len(sel))
            if n > 0:
                picked = rng.choice(sel, size=n, replace=False)
                chosen_idx.extend(picked.tolist())
            per_decile_n[d_label] = int(n)
        summary[plat] = per_decile_n
    chosen_idx = np.array(chosen_idx, dtype=int)
    sub = df.loc[chosen_idx].reset_index(drop=True)

    print("language detection (heuristic)...", flush=True)
    sub["editorial_lang"] = sub["editorial_text"].astype(str).map(detect_lang)
    sub["candidate_lang"] = sub["candidate_text"].astype(str).map(detect_lang)

    keep_cols = [
        "pair_id", "platform", "decile", "qwen_label",
        "editorial_text", "candidate_text",
        "editorial_lang", "candidate_lang",
    ]
    sub = sub[keep_cols].copy()

    # LEAKAGE GUARD assertions
    bad = [c for c in sub.columns if any(t in c.lower() for t in ["cos", "sim", "embed"])]
    assert not bad, f"LEAKAGE GUARD violated: {bad}"

    sub.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET} rows={len(sub)}", flush=True)
    print("per-platform per-decile sizes:", json.dumps(summary, indent=2), flush=True)
    print("label dist:", sub["qwen_label"].value_counts().to_dict(), flush=True)
    print("editorial_lang:", sub["editorial_lang"].value_counts().to_dict(), flush=True)
    print("candidate_lang:", sub["candidate_lang"].value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
