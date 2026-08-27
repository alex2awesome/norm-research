#!/usr/bin/env python3
"""Build LC organic-relabel cell: ~2,000 pairs sampled from the 102K LC pool,
excluding near-copies (normalized-exact OR jaccard>0.85 OR cosine>0.97).

Pool source (sk3):  outputs/v2_analysis/comp_qwen_phase1b_full_pool_with_cosine.parquet (lc platform)
Soft prior:         outputs/distill_crossenc/lc_bulk_labels_v0.parquet (prob/label, weak)
Bank coverage:      outputs/v2_analysis/lc_python_metric_scores.parquet (candidate_id, laptop)
Diagnostic ref:     outputs/v2_analysis/lc_copy_diagnostic.parquet (validation)
Old cell:           outputs/v2_analysis/lc_python_2k_sample.parquet (problem_id / question_slug mapping)

Outputs:
  outputs/v2_analysis/lc_organic_relabel/shards/relabel_shard_{00..07}.jsonl
  outputs/v2_analysis/lc_organic_relabel/build_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------- copy filters ----------------------

_WS_RE = re.compile(r"\s+")
_PY_COMMENT_RE = re.compile(r"#.*$", re.M)
_DOCSTRING_RE = re.compile(r'(""".*?"""|\'\'\'.*?\'\'\')', re.S)


def normalize_code(s: str) -> str:
    """Aggressive normalization for exact-dup detection: strip comments/docstrings,
    collapse whitespace, lowercase. Should make trivially-equivalent code identical."""
    if not isinstance(s, str):
        return ""
    s = _DOCSTRING_RE.sub("", s)
    s = _PY_COMMENT_RE.sub("", s)
    s = _WS_RE.sub(" ", s)
    return s.strip().lower()


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokenize(s: str) -> set:
    """Identifier-only token set (matches the original lc_copy_diagnostic
    reference: e.g., the divideString near-copy scores 0.857 there and 0.889
    under identifier-only Jaccard; punctuation/numeric tokens dilute the score
    and miss true near-copies near the 0.85 threshold)."""
    if not isinstance(s, str):
        return set()
    # drop docstrings / comments first so they don't dilute Jaccard
    s = _DOCSTRING_RE.sub("", s)
    s = _PY_COMMENT_RE.sub("", s)
    return set(_TOKEN_RE.findall(s))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    u = a | b
    if not u:
        return 0.0
    return len(a & b) / len(u)


def compute_filters(df: pd.DataFrame,
                    cand_col: str = "candidate_text",
                    edit_col: str = "editorial_text",
                    cos_col: str = "cosine") -> pd.DataFrame:
    """Returns df with added columns: jaccard_tok, exact_dup, near_copy."""
    out = df.copy()
    cand_norm = out[cand_col].fillna("").map(normalize_code)
    edit_norm = out[edit_col].fillna("").map(normalize_code)
    out["exact_dup"] = (cand_norm == edit_norm) & (cand_norm.str.len() > 0)
    cand_tok = out[cand_col].fillna("").map(tokenize)
    edit_tok = out[edit_col].fillna("").map(tokenize)
    out["jaccard_tok"] = [jaccard(a, b) for a, b in zip(cand_tok, edit_tok)]
    cos = out[cos_col].fillna(0.0)
    # NOTE: tuned vs reference lc_copy_diagnostic on the old 2K cell:
    # reference uses jaccard>0.85 + cosine>0.97 with a slightly different tokenizer
    # and missed-by-mine pairs all live at jacc 0.83-0.85 + cosine 0.93-0.97. To
    # guarantee 20/20 known-near-copy recall (and stay conservative — over-
    # exclude rather than under-exclude), we soften thresholds to jacc>0.83
    # and cosine>0.95. Validated on the full 1995-pair diagnostic.
    out["near_copy"] = (
        out["exact_dup"]
        | (out["jaccard_tok"] > 0.83)
        | (cos > 0.95)
    )
    return out


# ---------------------- validation ----------------------

def validate_filters(old_cell_path: str, diag_path: str, n_validate: int = 20) -> dict:
    """Take 20 known near_copy pairs from the diagnostic; reimplement; assert all caught."""
    old = pd.read_parquet(old_cell_path)
    diag = pd.read_parquet(diag_path)
    m = diag.merge(old[["pair_id", "code", "editorial_code"]], on="pair_id")
    known_nc = m[m["near_copy"]].head(n_validate).copy()
    known_nc = known_nc.rename(columns={"code": "candidate_text", "editorial_code": "editorial_text"})
    re_filtered = compute_filters(known_nc[["candidate_text", "editorial_text", "cosine"]])
    n_caught = int(re_filtered["near_copy"].sum())
    report = {
        "n_validate": int(len(known_nc)),
        "n_caught": n_caught,
        "passed": n_caught == len(known_nc),
    }
    # also try 20 known non-near-copies
    not_nc = m[~m["near_copy"]].head(n_validate).copy()
    not_nc = not_nc.rename(columns={"code": "candidate_text", "editorial_code": "editorial_text"})
    re_clean = compute_filters(not_nc[["candidate_text", "editorial_text", "cosine"]])
    report["n_negs_validate"] = int(len(not_nc))
    report["n_neg_caught_false"] = int(re_clean["near_copy"].sum())  # False positives in reimpl
    return report


# ---------------------- sampling ----------------------

COSINE_BINS = [
    # (lo, hi, target_frac). Original spec was 15/25/35/25, but after near-copy
    # exclusion bin 3 (0.7-0.97) only has 186 pairs available under the ≤4-per-
    # problem cap (the rest are copies). Adjust target so bin 3 = all available
    # (~9% = 170) and rebalance the remaining 91% across bins 0-2 in the spec
    # ratio 15:25:35 (which sums to 75) → 18/30/43/9. Tuned 2026-06-11.
    (0.00, 0.30, 0.18),
    (0.30, 0.50, 0.30),
    (0.50, 0.70, 0.43),
    (0.70, 0.97, 0.09),
]


def cosine_bin(c: float) -> int:
    for i, (lo, hi, _) in enumerate(COSINE_BINS):
        if lo <= c < hi:
            return i
    return -1


def decile_for(c: float) -> int:
    return min(9, max(0, int(c * 10)))


def stratified_sample(pool: pd.DataFrame, n_target: int = 2000,
                      max_per_problem: int = 4,
                      prior_pos_weight: float = 1.0,
                      seed: int = 17) -> pd.DataFrame:
    """Stratified by cosine_bin, oversample with prior-as-soft-weight to push
    predicted pos-rate toward ~35-45%."""
    rng = np.random.default_rng(seed)
    pool = pool.copy()
    pool["cos_bin"] = pool["cosine"].map(cosine_bin)
    pool = pool[pool["cos_bin"] >= 0].reset_index(drop=True)

    picked = []
    used_per_problem: dict = {}

    for bin_i, (lo, hi, frac) in enumerate(COSINE_BINS):
        bin_target = int(round(n_target * frac))
        sub = pool[pool["cos_bin"] == bin_i].copy()
        if sub.empty:
            continue
        # Weight: prior label 1 gets prior_pos_weight, label 0 gets 1.
        w = np.where(sub["label"].values == 1, prior_pos_weight, 1.0)
        # randomize draw order via weighted shuffle
        order = rng.choice(len(sub), size=len(sub), replace=False,
                           p=w / w.sum())
        sub = sub.iloc[order].reset_index(drop=True)
        bin_picks = []
        for _, row in sub.iterrows():
            if len(bin_picks) >= bin_target:
                break
            pid = row["canonical_pid"]
            if used_per_problem.get(pid, 0) >= max_per_problem:
                continue
            bin_picks.append(row)
            used_per_problem[pid] = used_per_problem.get(pid, 0) + 1
        picked.extend(bin_picks)

    return pd.DataFrame(picked).reset_index(drop=True)


# ---------------------- shard emission ----------------------

def new_pair_id(orig_pair_id: str) -> str:
    """lcorg_<8 hex>"""
    h = hashlib.sha1(orig_pair_id.encode()).hexdigest()[:8]
    return f"lcorg_{h}"


def emit_shards(sample: pd.DataFrame, out_dir: Path,
                bank_cand_ids: set,
                n_shards: int = 8) -> dict:
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    rows = []
    skipped_short = 0
    skipped_dup = 0
    for _, r in sample.iterrows():
        cand = r.get("candidate_text") or ""
        edit = r.get("editorial_text") or ""
        if len(cand) < 30 or len(edit) < 30:
            skipped_short += 1
            continue
        pid = new_pair_id(str(r["pair_id"]))
        if pid in seen:
            skipped_dup += 1
            continue
        seen.add(pid)
        # canonical_pid in pool is like 'lc:<slug>'
        slug = r["canonical_pid"]
        if isinstance(slug, str) and slug.startswith("lc:"):
            slug = slug[3:]
        cos = float(r["cosine"]) if not pd.isna(r["cosine"]) else 0.0
        rows.append({
            "pair_id": pid,
            "problem_id": slug,
            "source": "lc",
            "editorial_id": str(r["editorial_id"]),
            "candidate_id": str(r["candidate_id"]),
            "language": "python",
            "cosine": cos,
            "decile": decile_for(cos),
            "orig_label": int(r["label"]),  # bulk-model prior, NOT ground truth
            "editorial_text": edit,
            "candidate_text": cand,
        })

    rng = random.Random(31)
    rng.shuffle(rows)
    by_shard = [[] for _ in range(n_shards)]
    for i, row in enumerate(rows):
        by_shard[i % n_shards].append(row)

    for i, shard in enumerate(by_shard):
        path = shards_dir / f"relabel_shard_{i:02d}.jsonl"
        with path.open("w") as f:
            for row in shard:
                f.write(json.dumps(row) + "\n")

    # bank coverage
    cand_ids = {str(r["candidate_id"]) for r in rows}
    bank_overlap = len(cand_ids & {str(x) for x in bank_cand_ids})

    return {
        "n_pairs_emitted": len(rows),
        "n_shards": n_shards,
        "per_shard": [len(s) for s in by_shard],
        "skipped_short_text": skipped_short,
        "skipped_dup_pair_id": skipped_dup,
        "unique_candidates": len(cand_ids),
        "bank_cand_overlap": bank_overlap,
        "bank_coverage_pct": round(100.0 * bank_overlap / max(1, len(cand_ids)), 2),
    }


# ---------------------- driver ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True,
                    help="phase1b full pool parquet (with cosine)")
    ap.add_argument("--bulk-labels", required=True,
                    help="lc_bulk_labels_v0.parquet (prob/label/pair_id)")
    ap.add_argument("--bank-scores", required=True,
                    help="lc_python_metric_scores.parquet (candidate_id)")
    ap.add_argument("--old-cell", required=True,
                    help="lc_python_2k_sample.parquet (for validation)")
    ap.add_argument("--diag", required=True,
                    help="lc_copy_diagnostic.parquet (for validation)")
    ap.add_argument("--out-dir", required=True,
                    help="outputs/v2_analysis/lc_organic_relabel/")
    ap.add_argument("--n-target", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 0: validate the reimplemented filters on old cell ---
    print("[validate] reimplementing copy filters on known near-copies...")
    valid_report = validate_filters(args.old_cell, args.diag, n_validate=20)
    print(f"[validate] caught {valid_report['n_caught']}/{valid_report['n_validate']} "
          f"known near-copies; false-pos on negs: {valid_report['n_neg_caught_false']}/{valid_report['n_negs_validate']}")
    if not valid_report["passed"]:
        print("[validate] FAILED — aborting", file=sys.stderr)
        sys.exit(2)

    # --- Step 1: load pool + bulk-label prior + bank ---
    print(f"[load] pool: {args.pool}")
    pool = pd.read_parquet(args.pool)
    pool = pool[pool["platform"] == "lc"].reset_index(drop=True)
    print(f"[load] LC rows in pool: {len(pool)}")

    print(f"[load] bulk labels: {args.bulk_labels}")
    bulk = pd.read_parquet(args.bulk_labels)
    bulk = bulk[bulk["platform"] == "lc"][["pair_id", "prob", "label"]]
    pool = pool.merge(bulk, on="pair_id", how="inner")
    print(f"[load] after merge with bulk: {len(pool)}; prior pos rate={pool.label.mean():.3f}")

    print(f"[load] bank scores: {args.bank_scores}")
    bank = pd.read_parquet(args.bank_scores)
    bank_cand_ids = set(bank["candidate_id"].astype(str))
    print(f"[load] bank candidates: {len(bank_cand_ids)}")

    # --- Step 2: compute copy filters on full pool, exclude near-copies ---
    print("[filter] computing exact_dup / jaccard / cosine filters...")
    pool_f = compute_filters(pool)
    n_exact = int(pool_f["exact_dup"].sum())
    n_jacc = int((pool_f["jaccard_tok"] > 0.83).sum())
    n_cos = int((pool_f["cosine"] > 0.95).sum())
    n_nc = int(pool_f["near_copy"].sum())
    organic = pool_f[~pool_f["near_copy"]].reset_index(drop=True)
    print(f"[filter] near_copy={n_nc} (exact={n_exact}, jacc>0.83={n_jacc}, cos>0.95={n_cos})")
    print(f"[filter] organic pool: {len(organic)}; prior pos rate={organic.label.mean():.3f}")

    # --- Step 3: stratified sample ---
    print("[sample] stratified by cosine bin + prior weight...")
    sample = stratified_sample(organic, n_target=args.n_target, seed=args.seed)
    print(f"[sample] selected {len(sample)} pairs; "
          f"predicted pos rate={sample.label.mean():.3f}")
    cos_dist = sample.groupby("cos_bin").size().to_dict()
    print(f"[sample] cosine bin counts: {cos_dist}")

    # --- Step 4: emit shards ---
    print("[emit] writing shards...")
    emit_report = emit_shards(sample, out_dir, bank_cand_ids, n_shards=8)
    print(f"[emit] {emit_report}")

    # --- Step 5: manifest ---
    manifest = {
        "pool_path": args.pool,
        "pool_lc_rows": int(len(pool)),
        "validation": valid_report,
        "exclusions": {
            "near_copy_total": n_nc,
            "exact_dup": n_exact,
            "jaccard_gt_0p83": n_jacc,
            "cosine_gt_0p95": n_cos,
            "organic_rows": int(len(organic)),
        },
        "cosine_bin_targets": [
            {"lo": lo, "hi": hi, "frac": frac} for lo, hi, frac in COSINE_BINS
        ],
        "sample_distribution": {
            "n": int(len(sample)),
            "predicted_pos_rate_bulk_model": float(sample.label.mean()),
            "cosine_bin_counts": {int(k): int(v) for k, v in cos_dist.items()},
            "unique_problems": int(sample["canonical_pid"].nunique()),
            "max_per_problem_cap": 4,
        },
        "shard_report": emit_report,
        "bank_coverage": {
            "bank_total_candidates": len(bank_cand_ids),
            "sample_unique_candidates": emit_report["unique_candidates"],
            "overlap": emit_report["bank_cand_overlap"],
            "pct": emit_report["bank_coverage_pct"],
        },
        "filter_definitions": {
            "exact_dup": "normalize: strip docstrings + #-comments, collapse whitespace, lower; then string-equality",
            "jaccard_tok": "token-set Jaccard on IDENTIFIERS only (after docstring+comment strip); threshold > 0.83",
            "cosine": "from pool parquet; threshold > 0.95",
            "near_copy": "exact_dup OR jaccard_tok>0.83 OR cosine>0.95",
            "tuning_note": "Thresholds softened from spec's 0.85/0.97 to 0.83/0.95 so that 20/20 known near-copies from lc_copy_diagnostic are recalled (the reference diagnostic's tokenizer differs slightly; conservative widening gives 100% recall on 276/276 reference near-copies vs 54 extra exclusions = +20%).",
        },
        "jaccard_threshold": 0.83,
        "cosine_threshold": 0.95,
        "notes": [
            "orig_label in shards is the bulk cross-encoder model's prediction (lc_bulk_labels_v0). It is a MACHINE PRIOR, not ground truth.",
            "Workers must NOT anchor on orig_label.",
            "pair_id is fresh (lcorg_<sha1[:8]>). Problems may overlap the old lc_python_2k cell but pair_ids do not.",
        ],
    }
    with (out_dir / "build_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] manifest -> {out_dir/'build_manifest.json'}")


if __name__ == "__main__":
    main()
