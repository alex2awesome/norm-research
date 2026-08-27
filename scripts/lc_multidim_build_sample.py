"""Build the 2000-pair multi-dimensional sample for the bank-vs-Claude correlation study.

Runs on sk3. Reads:
  - lc_editorial_similarity.parquet (max_sim per candidate)
  - lc_candidate_corpus.parquet     (candidate code)
  - lc_editorial_corpus.parquet     (editorial code)
  - leetcode_cpp_metric_scores_fixed.parquet (bank scores, 5000 rows)

Strategy: restrict to the 4966 bank rows that have an editorial pair, stratify by
max_sim deciles (5 buckets, 400 each), and within each bucket spread across detected
language families. Writes the 2K-pair table both as parquet (for joining later) and
as JSONL (only the labeling-relevant fields for the laptop Claude pipeline).
"""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis")
SIM = ROOT / "lc_editorial_similarity.parquet"
CAND = ROOT / "lc_candidate_corpus.parquet"
ED = ROOT / "lc_editorial_corpus.parquet"
BANK = ROOT / "leetcode_cpp_metric_scores_fixed.parquet"

OUT_SAMPLE_PARQUET = ROOT / "lc_multidim_sample_2000.parquet"
OUT_SAMPLE_JSONL = ROOT / "lc_multidim_sample_2000.jsonl"

N_TARGET = 2000
N_BUCKETS = 5  # quintiles of max_sim -> 400 per bucket
SEED = 12345

LANG_REGEX = {
    "cpp": re.compile(r"\b(vector\s*<|std::|->|public:|::)|#include\b"),
    "java": re.compile(r"\b(public class|public static|System\.out|ArrayList<)"),
    "python": re.compile(r"^\s*(def |class .*:|import |from .* import )", re.MULTILINE),
    "javascript": re.compile(r"\b(function |const |let |var |=>)"),
}


def detect_lang(code: str, declared: str) -> str:
    if isinstance(declared, str) and declared in ("cpp", "java", "python", "javascript", "go", "rust", "csharp", "swift", "ruby", "typescript", "kotlin"):
        return declared
    if not isinstance(code, str):
        return "unknown"
    head = code[:1500]
    for lang, rx in LANG_REGEX.items():
        if rx.search(head):
            return lang
    return "unknown"


def main():
    sim = pd.read_parquet(SIM)
    cand = pd.read_parquet(CAND)
    ed = pd.read_parquet(ED)
    bank = pd.read_parquet(BANK)

    bank_ids = set(bank["row_id"].tolist())
    print(f"bank rows: {len(bank_ids)}")

    # Restrict to bank's candidates that have an editorial similarity row
    cand_b = cand[cand["candidate_id"].isin(bank_ids)].copy()
    cand_b = cand_b.merge(
        sim[["candidate_id", "argmax_editorial_id", "max_sim", "mean_sim", "n_editorials"]],
        on="candidate_id",
        how="inner",
    )
    print(f"cand_b after sim merge: {len(cand_b)}")

    # Join in editorial code/approach
    cand_b = cand_b.merge(
        ed[["editorial_id", "canonical_code", "canonical_code_stripped", "approach", "code_lang"]].rename(
            columns={"editorial_id": "argmax_editorial_id"}
        ),
        on="argmax_editorial_id",
        how="left",
    )
    cand_b = cand_b[cand_b["canonical_code"].notna()].copy()
    print(f"with editorial code: {len(cand_b)}")

    # Detect language for candidate (since most are 'unknown')
    cand_b["lang_detected"] = [detect_lang(c, d) for c, d in zip(cand_b["code"], cand_b["language_norm"])]
    print("detected language distribution:")
    print(cand_b["lang_detected"].value_counts().head(10))

    # Drop very-very tiny code (less than ~80 chars) — labeling junk
    cand_b = cand_b[cand_b["code"].str.len() > 80].copy()
    cand_b = cand_b[cand_b["canonical_code"].str.len() > 80].copy()
    print(f"after min-length filter: {len(cand_b)}")

    # Cap each code at 4000 chars to keep prompts manageable
    cand_b["candidate_code_for_prompt"] = cand_b["code"].str.slice(0, 4000)
    cand_b["editorial_code_for_prompt"] = cand_b["canonical_code"].str.slice(0, 4000)

    # Stratify: 5 quintile buckets x lang stratification
    cand_b["sim_bucket"] = pd.qcut(cand_b["max_sim"], q=N_BUCKETS, labels=False, duplicates="drop")
    print("sim bucket counts:")
    print(cand_b["sim_bucket"].value_counts().sort_index())

    rng = np.random.RandomState(SEED)
    per_bucket = N_TARGET // N_BUCKETS

    sampled = []
    for b in sorted(cand_b["sim_bucket"].dropna().unique()):
        sub = cand_b[cand_b["sim_bucket"] == b].copy()
        # Lang-stratified within bucket: cap each lang at 30% of bucket
        bucket_target = per_bucket
        # Heuristic: sample from each non-unknown lang proportional to its count, plus the rest from unknown
        lang_groups = sub.groupby("lang_detected")
        n_langs = sub["lang_detected"].nunique()
        # Cheap path: just stratified sample preserving lang ratios but never letting unknown exceed 60%
        if n_langs == 1:
            picked = sub.sample(min(len(sub), bucket_target), random_state=int(rng.randint(0, 1 << 31)))
        else:
            # Soft target per lang
            counts = sub["lang_detected"].value_counts()
            picked_frames = []
            remaining = bucket_target
            # First, give each non-unknown lang at least proportional share, capped so no lang >50%
            non_unknown_total = counts[counts.index != "unknown"].sum()
            for lg, ct in counts.items():
                if lg == "unknown":
                    continue
                want = int(round(min(bucket_target * 0.5, bucket_target * ct / max(1, len(sub)))))
                want = min(want, ct, remaining)
                if want > 0:
                    picked_frames.append(sub[sub["lang_detected"] == lg].sample(want, random_state=int(rng.randint(0, 1 << 31))))
                    remaining -= want
            # Fill the rest from unknown (which is realistic since most code lacks a declared language)
            unk = sub[sub["lang_detected"] == "unknown"]
            if remaining > 0 and len(unk) > 0:
                take = min(remaining, len(unk))
                picked_frames.append(unk.sample(take, random_state=int(rng.randint(0, 1 << 31))))
                remaining -= take
            # If still short (rare), take whatever's left
            if remaining > 0:
                rest = sub.drop(pd.concat(picked_frames).index)
                take = min(remaining, len(rest))
                if take > 0:
                    picked_frames.append(rest.sample(take, random_state=int(rng.randint(0, 1 << 31))))
            picked = pd.concat(picked_frames) if picked_frames else sub.sample(min(bucket_target, len(sub)), random_state=int(rng.randint(0, 1 << 31)))
        sampled.append(picked)

    out = pd.concat(sampled).reset_index(drop=True)
    out["pair_id"] = np.arange(len(out))
    print(f"\nfinal sample: {len(out)}")
    print("lang_detected x sim_bucket:")
    print(pd.crosstab(out["lang_detected"], out["sim_bucket"]))

    keep = [
        "pair_id", "candidate_id", "argmax_editorial_id", "question_slug",
        "language_norm", "lang_detected", "max_sim", "mean_sim", "sim_bucket",
        "code", "canonical_code", "approach", "code_lang",
        "candidate_code_for_prompt", "editorial_code_for_prompt",
    ]
    out_keep = out[keep].copy()
    out_keep.to_parquet(OUT_SAMPLE_PARQUET, index=False)
    print(f"wrote {OUT_SAMPLE_PARQUET}")

    with open(OUT_SAMPLE_JSONL, "w") as f:
        for _, r in out_keep.iterrows():
            rec = {
                "pair_id": int(r["pair_id"]),
                "candidate_id": int(r["candidate_id"]),
                "editorial_id": int(r["argmax_editorial_id"]),
                "question_slug": r["question_slug"],
                "language": r["lang_detected"],
                "max_sim": float(r["max_sim"]),
                "sim_bucket": int(r["sim_bucket"]),
                "candidate_code": r["candidate_code_for_prompt"],
                "editorial_code": r["editorial_code_for_prompt"],
                "editorial_approach": r["approach"] if isinstance(r["approach"], str) else None,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_SAMPLE_JSONL}")


if __name__ == "__main__":
    main()
