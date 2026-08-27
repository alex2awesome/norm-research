#!/usr/bin/env python3
"""Negative-enriched extension cells for CF and LC (lc_cf_push, 2026-06-11/12).

Goal: oversample LOW-cosine pairs (cosine for SAMPLING ONLY, never a feature)
from each platform's existing pool to add expected negatives to the two weak
cells of the 4-platform table.

CF: from outputs/v2_analysis/cf_rebuild_cell/cf2_pool.parquet (30,251 pairs,
    bge cosines in cf2_pool_cosine.parquet), excluding the 2,255 already-sampled
    pair_ids. Draw 1,000 pairs with cosine < 0.2, lowest-cosine-first, union
    per-problem cap (existing cell usage counted). Expected negatives ~ 430
    (pos rate 0.47-0.67 in [0, 0.2) per the labeled cell).

LC: from the 102K-pair phase1b LC pool (near-copy filter REAPPLIED, identical
    thresholds to the lc_organic_relabel build), excluding pairs already in the
    organic cell (lcorg sha1 mapping). Draw ~1,000 pairs from cosine bins
    [0,0.1)/[0.1,0.2)/[0.2,0.3) at 150/500/350. Expected negatives ~ 600
    (pos rates 0.0/0.36/0.53 in the labeled cell).

Phases: cf | lc | all. Each phase writes:
  outputs/v2_analysis/lc_cf_push/{cf_ext,lc_ext}/
    ext_pairs.parquet        selected pairs (full text + cosine)
    inspect_30.md            30 random examples for MANDATORY hand inspection
    shards/shard_NN.jsonl    labeling shards (only after --emit-shards)
    build_manifest.json

Hand-inspection gate: shards are NOT emitted unless --emit-shards is passed;
run once without it, read inspect_30.md, then re-run with --emit-shards.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
sys.path.insert(0, str(ROOT / "scripts" / "lc_organic"))
from build_lc_organic_relabel import compute_filters  # near-copy filter, validated

OUT = ROOT / "outputs/v2_analysis/lc_cf_push"
CF_CELL = ROOT / "outputs/v2_analysis/cf_rebuild_cell"
LC_CELL = ROOT / "outputs/v2_analysis/lc_organic_relabel"
SEED = 1234
MIN_CODE_LEN = 30
PAIRS_PER_SHARD = 250


def greedy_pick(df: pd.DataFrame, n_target: int, used_per_problem: dict,
                cap: int = 4, relax_cap: int = 6) -> pd.DataFrame:
    """Pick rows in df order, respecting union per-problem cap; relax once."""
    picked, deferred = [], []
    used = dict(used_per_problem)
    for _, row in df.iterrows():
        if len(picked) >= n_target:
            break
        pid = row["canonical_pid"]
        if used.get(pid, 0) >= cap:
            deferred.append(row)
            continue
        picked.append(row)
        used[pid] = used.get(pid, 0) + 1
    if len(picked) < n_target and deferred:
        for row in deferred:
            if len(picked) >= n_target:
                break
            pid = row["canonical_pid"]
            if used.get(pid, 0) >= relax_cap:
                continue
            picked.append(row)
            used[pid] = used.get(pid, 0) + 1
    return pd.DataFrame(picked).reset_index(drop=True)


def write_inspect(df: pd.DataFrame, path: Path, n: int = 30, seed: int = 7):
    rows = df.sample(min(n, len(df)), random_state=seed)
    lines = [f"# Hand-inspection sample ({len(rows)} of {len(df)} selected pairs)\n",
             "Check per pair: (1) both sides are real code >= 30 chars, not truncated"
             " garbage; (2) candidate is NOT a copy of the editorial; (3) pair is a"
             " plausible same-problem pair; (4) low cosine reflects genuine"
             " style/approach distance, not corrupted text.\n"]
    for i, (_, r) in enumerate(rows.iterrows()):
        lines.append(f"\n## Example {i+1} | pair_id={r['pair_id']} | pid={r['canonical_pid']} "
                     f"| cosine={r['cosine']:.3f} | lang={r.get('candidate_lang','py')}")
        lines.append(f"\n### editorial ({len(r['editorial_text'])} ch)\n```\n"
                     f"{r['editorial_text'][:1500]}\n```")
        lines.append(f"\n### candidate ({len(r['candidate_text'])} ch)\n```\n"
                     f"{r['candidate_text'][:1500]}\n```")
    path.write_text("\n".join(lines))
    print(f"[inspect] wrote {path}")


def emit_shards(rows: list, shards_dir: Path, prefix: str, n_per_shard: int):
    shards_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    rng.shuffle(rows)
    n_shards = (len(rows) + n_per_shard - 1) // n_per_shard
    per = []
    for i in range(n_shards):
        chunk = rows[i * n_per_shard:(i + 1) * n_per_shard]
        p = shards_dir / f"shard_{prefix}_{i:02d}.jsonl"
        with p.open("w") as f:
            for r in chunk:
                f.write(json.dumps(r) + "\n")
        per.append(len(chunk))
    return {"n_shards": n_shards, "per_shard": per, "n_total": len(rows)}


# ------------------------------- CF ------------------------------- #

def build_cf(emit: bool, n_target: int = 1000):
    out = OUT / "cf_ext"
    out.mkdir(parents=True, exist_ok=True)
    pool = pd.read_parquet(CF_CELL / "cf2_pool.parquet")
    cos = pd.read_parquet(CF_CELL / "cf2_pool_cosine.parquet")
    pool = pool.merge(cos, on="pair_id")
    pairs = pd.read_parquet(CF_CELL / "cf2_pairs.parquet")
    sampled = set(pairs["pair_id"])
    used = pairs.groupby("canonical_pid").size().to_dict()

    avail = pool[~pool["pair_id"].isin(sampled)].copy()
    avail = avail[(avail["candidate_text"].str.len() >= MIN_CODE_LEN)
                  & (avail["editorial_text"].str.len() >= MIN_CODE_LEN)]
    low = avail[avail["cosine"] < 0.2].sort_values("cosine").reset_index(drop=True)
    print(f"[cf] pool {len(pool)}, available {len(avail)}, cosine<0.2: {len(low)} "
          f"({low['canonical_pid'].nunique()} problems)")

    sel = greedy_pick(low, n_target, used, cap=4, relax_cap=6)
    print(f"[cf] selected {len(sel)} pairs / {sel['canonical_pid'].nunique()} problems; "
          f"cosine range [{sel['cosine'].min():.3f}, {sel['cosine'].max():.3f}]")
    # expected negatives from labeled-cell pos rates by fine bin
    posrate = {(0.0, 0.05): 0.472, (0.05, 0.10): 0.606,
               (0.10, 0.15): 0.617, (0.15, 0.20): 0.670}
    exp_neg = 0.0
    for (lo, hi), p in posrate.items():
        n = int(((sel["cosine"] >= lo) & (sel["cosine"] < hi)).sum())
        exp_neg += n * (1 - p)
    print(f"[cf] expected negatives ~ {exp_neg:.0f}")

    sel.to_parquet(out / "ext_pairs.parquet", index=False)
    write_inspect(sel, out / "inspect_30.md")

    manifest = {
        "platform": "cf", "n_selected": int(len(sel)),
        "n_problems": int(sel["canonical_pid"].nunique()),
        "cosine_range": [float(sel["cosine"].min()), float(sel["cosine"].max())],
        "expected_negatives": round(exp_neg),
        "sampling": "lowest-cosine-first from cosine<0.2, union per-problem cap 4 (relax 6)",
        "excluded_already_sampled": len(sampled),
        "seed": SEED, "emitted_shards": bool(emit),
    }
    if emit:
        rows = []
        for _, r in sel.iterrows():
            rows.append({
                "pair_id": r["pair_id"], "problem_id": r["canonical_pid"],
                "source": "cf", "editorial_id": r["editorial_id"],
                "candidate_id": str(r["candidate_id"]),
                "language": r["candidate_lang"], "cosine": float(r["cosine"]),
                "decile": "[0.0-0.2)",
                "editorial_text": r["editorial_text"],
                "candidate_text": r["candidate_text"],
            })
        manifest["shards"] = emit_shards(rows, out / "shards", "cfx", PAIRS_PER_SHARD)
        (out / "results").mkdir(exist_ok=True)
    (out / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[cf] manifest -> {out/'build_manifest.json'}")


# ------------------------------- LC ------------------------------- #

LC_BINS = [(0.00, 0.10, 150), (0.10, 0.20, 500), (0.20, 0.30, 350)]


def lcorg_id(orig_pair_id: str) -> str:
    return f"lcorg_{hashlib.sha1(str(orig_pair_id).encode()).hexdigest()[:8]}"


def build_lc(emit: bool):
    out = OUT / "lc_ext"
    out.mkdir(parents=True, exist_ok=True)
    pool = pd.read_parquet(OUT / "comp_qwen_phase1b_full_pool_with_cosine.parquet")
    pool = pool[pool["platform"] == "lc"].reset_index(drop=True)
    print(f"[lc] pool LC rows: {len(pool)}")

    # exclude pairs already in the organic cell (sha1 mapping is deterministic)
    existing = pd.read_parquet(LC_CELL / "merged_labels.parquet")
    existing_ids = set(existing["pair_id"])
    pool["lcorg_pair_id"] = pool["pair_id"].map(lcorg_id)
    pool = pool[~pool["lcorg_pair_id"].isin(existing_ids)].reset_index(drop=True)
    print(f"[lc] after excluding {len(existing_ids)} existing cell pairs: {len(pool)}")

    # near-copy filter — identical validated implementation (imported)
    lowish = pool[pool["cosine"] < 0.30].copy()   # filter only what we can draw from
    # Hand-inspection round 1 (2026-06-11) found low-cosine pool pairs polluted
    # with <100-char code FRAGMENTS and image/markdown-stub editorials. Those
    # would be deterministic l1=0 under the protocol ("boilerplate/stub") and
    # inflate the extension with junk-detection negatives. Floor: 150 chars BOTH
    # sides (existing cell: only 5% cand / 10% ed below 150). Documented in report.
    LC_EXT_MIN_LEN = 150
    n0 = len(lowish)
    lowish = lowish[(lowish["candidate_text"].str.len() >= LC_EXT_MIN_LEN)
                    & (lowish["editorial_text"].str.len() >= LC_EXT_MIN_LEN)]
    print(f"[lc] quality floor >={LC_EXT_MIN_LEN}ch both sides: {n0} -> {len(lowish)}")
    # Hand-inspection round 2: still saw (a) image-only / upvote-spam editorials
    # with zero algorithmic content, (b) comment-only candidates. Both are
    # deterministic l1=0 junk. Targeted exclusions (legit TEXTUAL approach
    # descriptions are KEPT — L1's pseudocode pathway covers them):
    import re as _re
    code_re = _re.compile(r"(\bdef |\bclass |\breturn\b|\bfor\b|\bwhile\b|[{};]|=\s|\bfunction\b|\bfn |\bfunc )")
    spam_re = _re.compile(r"(?i)(upvote|thanks for visiting|please give|star the repository|keep me motivated)")
    ncode_ed = lowish["editorial_text"].map(lambda s: len(code_re.findall(s or "")))
    junk_ed = (ncode_ed < 3) & (lowish["editorial_text"].str.contains(r"!\[image", regex=True)
                                | lowish["editorial_text"].map(lambda s: bool(spam_re.search(s or ""))))
    ncode_cand = lowish["candidate_text"].map(lambda s: len(code_re.findall(s or "")))
    junk_cand = ncode_cand < 3
    n1 = len(lowish)
    lowish = lowish[~junk_ed & ~junk_cand]
    print(f"[lc] junk filter (spam/image-only ed: {int(junk_ed.sum())}, "
          f"no-code cand: {int(junk_cand.sum())}): {n1} -> {len(lowish)}")
    f = compute_filters(lowish)
    n_nc = int(f["near_copy"].sum())
    organic = f[~f["near_copy"]].reset_index(drop=True)
    print(f"[lc] cosine<0.3 rows {len(lowish)}, near-copies removed {n_nc}, organic {len(organic)}")

    used = existing.groupby("problem_id").size().to_dict()
    # canonical_pid here is 'lc:<slug>' — map to slug for the cap bookkeeping
    organic["slug"] = organic["canonical_pid"].str.replace("^lc:", "", regex=True)
    used_by_slug = {k: v for k, v in used.items()}

    rng = np.random.default_rng(SEED)
    picks = []
    for lo, hi, n_bin in LC_BINS:
        sub = organic[(organic["cosine"] >= lo) & (organic["cosine"] < hi)].copy()
        sub = sub.iloc[rng.permutation(len(sub))].reset_index(drop=True)
        sub = sub.rename(columns={"slug": "canonical_pid", "canonical_pid": "canonical_pid_raw"})
        got = greedy_pick(sub, n_bin, used_by_slug, cap=4, relax_cap=6)
        for pid, c in got.groupby("canonical_pid").size().items():
            used_by_slug[pid] = used_by_slug.get(pid, 0) + int(c)
        print(f"[lc] bin [{lo},{hi}): wanted {n_bin}, got {len(got)}")
        picks.append(got)
    sel = pd.concat(picks, ignore_index=True)
    print(f"[lc] selected {len(sel)} pairs / {sel['canonical_pid'].nunique()} problems")

    posrate = {(0.0, 0.1): 0.0, (0.1, 0.2): 0.355, (0.2, 0.3): 0.529}
    exp_neg = sum(int(((sel["cosine"] >= lo) & (sel["cosine"] < hi)).sum()) * (1 - p)
                  for (lo, hi), p in posrate.items())
    print(f"[lc] expected negatives ~ {exp_neg:.0f}")

    sel.to_parquet(out / "ext_pairs.parquet", index=False)
    write_inspect(sel, out / "inspect_30.md")

    manifest = {
        "platform": "lc", "n_selected": int(len(sel)),
        "n_problems": int(sel["canonical_pid"].nunique()),
        "near_copies_removed_in_draw_range": n_nc,
        "expected_negatives": round(exp_neg),
        "bins": [{"lo": lo, "hi": hi, "target": n} for lo, hi, n in LC_BINS],
        "sampling": "random within cosine bin, union per-problem cap 4 (relax 6), near-copy filter reapplied",
        "seed": SEED, "emitted_shards": bool(emit),
    }
    if emit:
        rows = []
        for _, r in sel.iterrows():
            rows.append({
                "pair_id": r["lcorg_pair_id"], "problem_id": r["canonical_pid"],
                "source": "lc", "editorial_id": str(r["editorial_id"]),
                "candidate_id": str(r["candidate_id"]),
                "language": "python", "cosine": float(r["cosine"]),
                "decile": int(min(9, max(0, r["cosine"] * 10))),
                "editorial_text": r["editorial_text"],
                "candidate_text": r["candidate_text"],
            })
        manifest["shards"] = emit_shards(rows, out / "shards", "lcx", PAIRS_PER_SHARD)
        (out / "results").mkdir(exist_ok=True)
    (out / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[lc] manifest -> {out/'build_manifest.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["cf", "lc", "all"])
    ap.add_argument("--emit-shards", action="store_true",
                    help="only after hand-inspecting inspect_30.md")
    a = ap.parse_args()
    if a.phase in ("cf", "all"):
        build_cf(a.emit_shards)
    if a.phase in ("lc", "all"):
        build_lc(a.emit_shards)


if __name__ == "__main__":
    main()
