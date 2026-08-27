"""Build the four-platform comparable editorial-similarity evaluation cell.

Frozen recipe applied identically to CC, LC, CF, AC:

  1. Pair pool: (candidate, editorial) per canonical_pid; both codes >= 30 chars.
     - CC, LC, Luogu sources of editorials live in the legacy pools
       (comp_qwen_phase1b_full_pool_with_cosine.parquet for CC/LC/Luogu and the
       lc_python_2k_sample.parquet for the LC-2K cell). We DO NOT re-pool these.
     - CF and AC editorials come from
       datasets/competition_unified/editorials_code_extracted.parquet filtered to
       extraction_confidence in {confident, confident_multi}; candidates come from
       datasets/competition_unified/candidates.parquet keyed on canonical_pid.
  2. Embed both sides with bge-code-v1 (reused for legacy where available; fresh
     embeddings for CF / AC). Cosine is used ONLY for stratification.
  3. Cosine-decile stratification matched to the 5K-source distribution:
       <0.2: 13.7%, 0.2-0.4: 14.6%, 0.4-0.6: 17.6%, 0.6-0.8: 31.2%, 0.8-1.0: 22.8%.
     Cap pairs/problem at 4. Target eval-cell size: min(1000, available).
  4. Labels: reuse existing Claude R4 labels where pairs overlap (CC + Luogu
     "comp_unified_editorial_labels"-Claude side; LC-2K via lc_python_2k_claude_labels).
     Unlabeled pairs (all CF / AC + gaps) are STAGED as Claude shards in the
     exact format of the 5K shard runner.
  5. Bank scoring: same metric bank used by comp_unified_claude_bank_score.py;
     re-run on the new sampled pool.
  6. Eval: StratifiedGroupKFold(5) by canonical_pid, LR + RF, on labelled rows.

Outputs (paths are relative to the repo root):
  outputs/v2_analysis/comp_fourplatform_cells/
      {platform}_pairs.parquet                  (sampled pair list w/ texts)
      {platform}_bank_scores.parquet            (bank-score columns + label-status)
  outputs/v2_analysis/comp_fourplatform_label_shards/
      shards/shard_{platform}_{NN}.jsonl
      r4_prompt.txt                             (verbatim copy)
      README.md                                 (how many shards per platform)
  outputs/v2_analysis/comp_fourplatform_status.md

Run modes (sequential, idempotent, checkpointed):
  --phase pool       build per-platform pair pool + reuse / compute cosine.
  --phase sample     stratified-sample per platform + label join + status col.
  --phase shard      emit Claude-labeling shards for unlabelled rows.
  --phase score      run metric-bank scoring on sampled pairs (SK3 only).
  --phase eval       grouped-CV AUC table on the labelled subset.
  --phase all        run all phases in order.

Seed = 42 everywhere.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DECILE_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]
DECILE_LABELS = ["[0.0-0.2)", "[0.2-0.4)", "[0.4-0.6)", "[0.6-0.8)", "[0.8-1.0]"]
# Source distribution from the 5K Claude-labeled set (frozen)
TARGET_DECILE_FRACTIONS = {
    0: 0.137,
    1: 0.146,
    2: 0.176,
    3: 0.312,
    4: 0.228,
}
TARGET_CELL_SIZE = 1000
MAX_PAIRS_PER_PROBLEM = 4
MIN_CODE_LEN = 30

# -------- environment-aware roots --------
LAPTOP_ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
SK3_ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
ROOT = SK3_ROOT if SK3_ROOT.exists() else LAPTOP_ROOT

DATA_ED = ROOT / "datasets/competition_unified/editorials_code_extracted.parquet"
DATA_CN = ROOT / "datasets/competition_unified/candidates.parquet"

OUT_BASE = ROOT / "outputs/v2_analysis"
OUT_CELLS = OUT_BASE / "comp_fourplatform_cells"
OUT_SHARDS = OUT_BASE / "comp_fourplatform_label_shards"
OUT_STATUS = OUT_BASE / "comp_fourplatform_status.md"

# legacy pools we may reuse for cosines + texts on CC / LC (phase1b)
LEGACY_POOL_PHASE1B = OUT_BASE / "comp_qwen_phase1b_full_pool_with_cosine.parquet"
# LC-2K pool already has `max_sim` cosines on the lcp_NNNNN ids
LC_2K_SAMPLE = OUT_BASE / "lc_python_2k_sample.parquet"
LC_2K_LABELS = OUT_BASE / "lc_python_2k_claude_labels.parquet"

# Existing CC/Luogu Claude labels (sources: 5K shards + 3900 shards)
SHARDS_5K_RESULTS = sorted(
    glob.glob(str(OUT_BASE / "comp_unified_claude5k_shards/results/shard_*.jsonl"))
)
SHARDS_5K_INPUT = sorted(
    glob.glob(str(OUT_BASE / "comp_unified_claude5k_shards/shards/shard_*.jsonl"))
)
SHARDS_3900_INPUT = OUT_BASE / "comp_unified_claude3900_shards/_input_3900.parquet"
SHARDS_3900_RESULT_PARQUETS = sorted(
    glob.glob(str(OUT_BASE / "comp_unified_claude3900_shards/comp_unified_claude3900_shard*.parquet"))
)

R4_PROMPT = OUT_BASE / "r4_error_targeted.txt"


# ============================================================
# PHASE 1 -- POOL CONSTRUCTION
# ============================================================

def _read_candidates(platform: str) -> pd.DataFrame:
    """Pull candidates for one platform, code >= 30 chars, dedup."""
    cols = ["platform", "canonical_pid", "candidate_id", "code", "code_lang",
            "language_norm", "source"]
    df = pd.read_parquet(DATA_CN, columns=cols)
    df = df[df["platform"] == platform].copy()
    df["code"] = df["code"].fillna("")
    df = df[df["code"].str.len() >= MIN_CODE_LEN]
    # dedup by candidate_id (some sources duplicate)
    df = df.drop_duplicates("candidate_id")
    return df


def _read_editorials(platform: str) -> pd.DataFrame:
    """Pull confident editorials for one platform."""
    df = pd.read_parquet(DATA_ED)
    df = df[df["platform"] == platform]
    df = df[df["extraction_confidence"].isin(["confident", "confident_multi"])]
    df = df.copy()
    df["extracted_code"] = df["extracted_code"].fillna("")
    df = df[df["extracted_code"].str.len() >= MIN_CODE_LEN]
    return df.reset_index(drop=True)


def build_cf_ac_pool(platform: str) -> pd.DataFrame:
    """Cross-join candidate x editorial per canonical_pid for CF / AC."""
    assert platform in ("cf", "ac")
    ed = _read_editorials(platform)
    cn = _read_candidates(platform)
    pids = set(ed["canonical_pid"].unique()) & set(cn["canonical_pid"].unique())
    ed = ed[ed["canonical_pid"].isin(pids)]
    cn = cn[cn["canonical_pid"].isin(pids)]
    pairs = cn.merge(
        ed[["editorial_id", "canonical_pid", "extracted_code", "code_lang"]],
        on="canonical_pid", how="inner", suffixes=("", "_ed"),
    )
    pairs = pairs.rename(columns={
        "code": "candidate_text",
        "extracted_code": "editorial_text",
        "language_norm": "candidate_lang",
        "code_lang_ed": "editorial_lang",
        "source": "candidate_source",
    })
    # synthetic pair_id stable across runs
    h = (pairs["candidate_id"].astype(str) + "|" + pairs["editorial_id"].astype(str)).apply(
        lambda s: hashlib.md5(s.encode()).hexdigest()[:14]
    )
    pairs["pair_id"] = f"4p_{platform}_" + h
    pairs["platform"] = platform
    keep = ["pair_id", "platform", "canonical_pid", "candidate_id", "editorial_id",
            "candidate_text", "editorial_text", "candidate_lang", "editorial_lang",
            "candidate_source"]
    return pairs[keep].drop_duplicates("pair_id").reset_index(drop=True)


def build_cc_lc_pool_from_legacy() -> dict:
    """Materialize CC + LC pools from the existing labelled sets only.

    Frozen-recipe stance: for CC + LC, the 4-platform evaluation cell IS the
    pre-existing labelled set (CC 449 / LC-2K 1995). They were sampled with a
    different decile stratification, but the texts + cosines + bank scoring
    pipeline are already aligned to the recipe. We do NOT re-sample them. The
    sampling step (Phase 2) then re-runs the decile-match check and reports
    whether the legacy decile profile differs from the 5K target.

    Returns dict[platform -> pool DataFrame].
    """
    out = {}
    # ----- CC: pull from 5K and 3900 shards -----
    cc_rows = []
    # 5K JSONL inputs carry both texts + cosine
    for p in SHARDS_5K_INPUT:
        for ln in open(p):
            r = json.loads(ln)
            if r.get("source") != "cc":
                continue
            cc_rows.append({
                "pair_id": r["pair_id"],
                "platform": "cc",
                "candidate_id": r["candidate_id"],
                "editorial_id": r["editorial_id"],
                "candidate_text": r.get("candidate_text", ""),
                "editorial_text": r.get("editorial_text", ""),
                "candidate_lang": r.get("language"),
                "editorial_lang": None,
                "candidate_source": None,
                "cosine": float(r.get("cosine", float("nan"))),
                "problem_id": r.get("problem_id"),  # codechef_<slug>; canonical_pid proxy
            })
    # 3900 inputs (have max_sim as cosine and candidate_code/editorial_code)
    inp3900 = pd.read_parquet(SHARDS_3900_INPUT)
    cc3 = inp3900[inp3900["platform"] == "cc"].copy()
    for _, r in cc3.iterrows():
        cc_rows.append({
            "pair_id": r["pair_id"],
            "platform": "cc",
            "candidate_id": r["candidate_id"],
            "editorial_id": r["editorial_id"],
            "candidate_text": r["candidate_code"] or "",
            "editorial_text": r["editorial_code"] or "",
            "candidate_lang": r.get("language"),
            "editorial_lang": None,
            "candidate_source": None,
            "cosine": float(r.get("max_sim", float("nan"))),
            "problem_id": None,
        })
    cc_pool = pd.DataFrame(cc_rows).drop_duplicates("pair_id").reset_index(drop=True)
    # add canonical_pid via problem_id where possible; else hash-fallback
    cc_pool["canonical_pid"] = cc_pool["problem_id"].fillna("cc::" + cc_pool["editorial_id"].astype(str))
    out["cc"] = cc_pool

    # ----- LC: lc_python_2k_sample is the full labelled pool -----
    lcs = pd.read_parquet(LC_2K_SAMPLE)
    lc_pool = pd.DataFrame({
        "pair_id": lcs["pair_id"].astype(str),
        "platform": "lc",
        "candidate_id": lcs["candidate_id"].astype(str),
        "editorial_id": lcs["argmax_editorial_id"].astype(str),
        "candidate_text": lcs["code"].astype(str),
        "editorial_text": lcs["editorial_code"].astype(str),
        "candidate_lang": "python",
        "editorial_lang": "python",
        "candidate_source": "lc_discuss",
        "cosine": lcs["max_sim"].astype(float),
        "canonical_pid": lcs["question_slug"].astype(str),
    })
    # filter both >=30 chars
    lc_pool = lc_pool[
        (lc_pool["candidate_text"].str.len() >= MIN_CODE_LEN)
        & (lc_pool["editorial_text"].str.len() >= MIN_CODE_LEN)
    ].reset_index(drop=True)
    out["lc"] = lc_pool
    return out


def phase_pool() -> None:
    OUT_CELLS.mkdir(parents=True, exist_ok=True)
    # CC + LC: legacy-derived pools (no fresh embedding)
    legacy = build_cc_lc_pool_from_legacy()
    for plat, df in legacy.items():
        path = OUT_CELLS / f"{plat}_pool.parquet"
        df.to_parquet(path, index=False)
        print(f"[pool] {plat}: {len(df)} pairs, {df['canonical_pid'].nunique()} problems -> {path}")
    # CF + AC: fresh pools (texts only; cosine added below)
    for plat in ("cf", "ac"):
        df = build_cf_ac_pool(plat)
        path = OUT_CELLS / f"{plat}_pool.parquet"
        df.to_parquet(path, index=False)
        print(f"[pool] {plat}: {len(df)} pairs, {df['canonical_pid'].nunique()} problems -> {path}")


# ============================================================
# PHASE 1b -- EMBED CF + AC (sk3 only)
# ============================================================

def phase_embed_cfac() -> None:
    """Run bge-code-v1 over CF + AC pools and write per-platform cosines."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        print("[embed] no GPU; skipping (run this phase on sk3 with a free GPU)")
        return

    BATCH = 64
    MAX_LEN = 1024

    def last_token_pool(h, m):
        if (m[:, -1].sum() == m.shape[0]).item():
            return h[:, -1]
        seq_lens = m.sum(dim=1) - 1
        return h[torch.arange(h.size(0), device=h.device), seq_lens]

    print("[embed] loading bge-code-v1...")
    tok = AutoTokenizer.from_pretrained("BAAI/bge-code-v1", trust_remote_code=True)
    mdl = AutoModel.from_pretrained(
        "BAAI/bge-code-v1", trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()

    def embed_unique(ids, texts):
        out = {}
        order = sorted(range(len(ids)), key=lambda i: len(texts[i]))
        ids_s = [ids[i] for i in order]
        txt_s = [texts[i] if texts[i] else "EMPTY" for i in order]
        t0 = time.time()
        with torch.inference_mode():
            for b in range(0, len(ids_s), BATCH):
                bi = ids_s[b:b + BATCH]
                bt = txt_s[b:b + BATCH]
                enc = tok(bt, max_length=MAX_LEN, padding=True, truncation=True,
                          return_tensors="pt", pad_to_multiple_of=8).to("cuda")
                try:
                    o = mdl(**enc)
                    e = F.normalize(last_token_pool(o.last_hidden_state, enc["attention_mask"]).float(),
                                    p=2, dim=1)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    half = len(bt) // 2
                    e1 = tok(bt[:half], max_length=MAX_LEN, padding=True, truncation=True,
                             return_tensors="pt", pad_to_multiple_of=8).to("cuda")
                    e2 = tok(bt[half:], max_length=MAX_LEN, padding=True, truncation=True,
                             return_tensors="pt", pad_to_multiple_of=8).to("cuda")
                    o1 = mdl(**e1); o2 = mdl(**e2)
                    p1 = F.normalize(last_token_pool(o1.last_hidden_state, e1["attention_mask"]).float(),
                                     p=2, dim=1)
                    p2 = F.normalize(last_token_pool(o2.last_hidden_state, e2["attention_mask"]).float(),
                                     p=2, dim=1)
                    e = torch.cat([p1, p2], dim=0)
                arr = e.cpu().numpy().astype(np.float32)
                for k, i in enumerate(bi):
                    out[str(i)] = arr[k]
                if b % (BATCH * 50) == 0:
                    dt = time.time() - t0
                    rate = (b + len(bi)) / max(dt, 1e-3)
                    eta = (len(ids_s) - b - len(bi)) / max(rate, 1e-3)
                    print(f"  {b + len(bi)}/{len(ids_s)} rate={rate:.1f}/s eta={eta/60:.1f}min",
                          flush=True)
        return out

    for plat in ("cf", "ac"):
        pool_path = OUT_CELLS / f"{plat}_pool.parquet"
        cos_path = OUT_CELLS / f"{plat}_pool_cosine.parquet"
        if cos_path.exists():
            print(f"[embed] {plat}: cosines already exist -> skipping")
            continue
        df = pd.read_parquet(pool_path)
        ed_u = df.drop_duplicates("editorial_id")[["editorial_id", "editorial_text"]]
        cn_u = df.drop_duplicates("candidate_id")[["candidate_id", "candidate_text"]]
        print(f"[embed] {plat}: {len(ed_u)} unique editorials, {len(cn_u)} unique candidates")
        ed_emb = embed_unique(ed_u["editorial_id"].tolist(), ed_u["editorial_text"].tolist())
        cn_emb = embed_unique(cn_u["candidate_id"].tolist(), cn_u["candidate_text"].tolist())
        cos = np.full(len(df), np.nan, dtype=np.float32)
        eids = df["editorial_id"].astype(str).to_numpy()
        cids = df["candidate_id"].astype(str).to_numpy()
        for k in range(len(df)):
            e = ed_emb.get(eids[k]); c = cn_emb.get(cids[k])
            if e is not None and c is not None:
                cos[k] = float(np.dot(e, c))
        out = pd.DataFrame({"pair_id": df["pair_id"].values, "cosine": cos})
        out.to_parquet(cos_path, index=False)
        print(f"[embed] wrote {cos_path} (missing={int(np.isnan(cos).sum())})")


# ============================================================
# PHASE 2 -- STRATIFIED SAMPLE + LABEL JOIN
# ============================================================

def assign_decile(c: float) -> int:
    if math.isnan(c):
        return -1
    for i, (lo, hi) in enumerate(DECILE_BINS):
        if c >= lo and c < hi:
            return i
    return 4


def _labels_existing() -> pd.DataFrame:
    """Stack all existing Claude labels with their candidate_id+editorial_id."""
    rows = []
    # 5K results merged with 5K inputs
    in5k = {}
    for p in SHARDS_5K_INPUT:
        for ln in open(p):
            r = json.loads(ln)
            in5k[r["pair_id"]] = r
    for p in SHARDS_5K_RESULTS:
        for ln in open(p):
            r = json.loads(ln)
            if r.get("claude_label") not in (0, 1):
                continue
            inp = in5k.get(r["pair_id"])
            if not inp:
                continue
            rows.append({
                "claude_label": int(r["claude_label"]),
                "candidate_id": inp["candidate_id"],
                "editorial_id": inp["editorial_id"],
                "source": inp.get("source"),
                "src_pair_id": r["pair_id"],
            })
    # 3900 result parquets joined to 3900 input
    inp39 = pd.read_parquet(SHARDS_3900_INPUT)[["pair_id", "candidate_id", "editorial_id", "platform"]]
    inp39 = inp39.set_index("pair_id")
    for p in SHARDS_3900_RESULT_PARQUETS:
        d = pd.read_parquet(p)
        d = d[d["claude_label"].isin([0, 1])]
        for _, r in d.iterrows():
            ix = inp39.loc[r["pair_id"]] if r["pair_id"] in inp39.index else None
            if ix is None:
                continue
            rows.append({
                "claude_label": int(r["claude_label"]),
                "candidate_id": ix["candidate_id"],
                "editorial_id": ix["editorial_id"],
                "source": ix["platform"],
                "src_pair_id": r["pair_id"],
            })
    # LC 2K labels
    lc = pd.read_parquet(LC_2K_LABELS)
    lcs = pd.read_parquet(LC_2K_SAMPLE)[["pair_id", "candidate_id", "argmax_editorial_id"]]
    lc = lc.merge(lcs, on="pair_id", how="left").rename(columns={
        "label": "claude_label",
        "pair_id": "src_pair_id",
        "argmax_editorial_id": "editorial_id",
    })
    lc["source"] = "lc"
    rows_lc = lc[["claude_label", "candidate_id", "editorial_id", "source", "src_pair_id"]].to_dict("records")
    rows.extend(rows_lc)
    out = pd.DataFrame(rows)
    out["candidate_id"] = out["candidate_id"].astype(str)
    out["editorial_id"] = out["editorial_id"].astype(str)
    return out


def stratified_sample(pool: pd.DataFrame, target: int, rng: np.random.Generator) -> pd.DataFrame:
    """Decile-stratified, problem-capped sampler. Returns indices into pool.

    If pool has a `is_labelled` boolean column (when label info is pre-joined),
    we prefer labelled rows within each decile before backfilling from
    unlabelled rows. This maximises reuse of existing Claude labels.
    """
    pool = pool.copy()
    pool["decile"] = pool["cosine"].apply(assign_decile)
    pool = pool[pool["decile"] >= 0]
    # planned per-decile sizes
    plan = {d: int(round(target * f)) for d, f in TARGET_DECILE_FRACTIONS.items()}
    diff = target - sum(plan.values())
    if diff != 0:
        big = max(plan, key=lambda d: plan[d])
        plan[big] += diff
    picked = []
    has_labels = "is_labelled" in pool.columns

    def _pick_from(sub: pd.DataFrame, want: int, per_problem: dict, chosen: list):
        for _, r in sub.iterrows():
            if len(chosen) >= want:
                return
            pid = r["canonical_pid"]
            if per_problem.get(pid, 0) >= MAX_PAIRS_PER_PROBLEM:
                continue
            chosen.append(r)
            per_problem[pid] = per_problem.get(pid, 0) + 1

    for d, want in plan.items():
        sub = pool[pool["decile"] == d].sample(frac=1.0, random_state=int(rng.integers(0, 2**31)))
        per_problem: dict = {}
        chosen: list = []
        if has_labels:
            # 1) Labelled rows first (helps reuse existing CC + LC labels).
            _pick_from(sub[sub["is_labelled"]], want, per_problem, chosen)
            # 2) Unlabelled rows to backfill.
            _pick_from(sub[~sub["is_labelled"]], want, per_problem, chosen)
        else:
            _pick_from(sub, want, per_problem, chosen)
        # 3) Last resort: drop the problem-cap.
        if len(chosen) < want:
            picked_ids = {r["pair_id"] for r in chosen}
            for _, r in sub.iterrows():
                if r["pair_id"] in picked_ids:
                    continue
                chosen.append(r)
                if len(chosen) >= want:
                    break
        picked.extend(chosen)
    out = pd.DataFrame(picked).reset_index(drop=True)
    return out


def phase_sample() -> None:
    OUT_CELLS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    labels = _labels_existing()
    # Build a (candidate_id, editorial_id) -> claude_label dict
    lbl_key = dict(zip(
        list(zip(labels["candidate_id"], labels["editorial_id"])),
        labels["claude_label"].astype(int),
    ))
    print(f"[sample] existing label index: {len(lbl_key)} (candidate_id, editorial_id) keys")
    summary = {}
    for plat in ("cc", "lc", "cf", "ac"):
        pool_path = OUT_CELLS / f"{plat}_pool.parquet"
        cos_path = OUT_CELLS / f"{plat}_pool_cosine.parquet"
        if not pool_path.exists():
            print(f"[sample] {plat}: pool missing, skipping")
            continue
        df = pd.read_parquet(pool_path)
        if "cosine" not in df.columns:
            if not cos_path.exists():
                print(f"[sample] {plat}: cosine missing; run --phase embed first")
                continue
            cdf = pd.read_parquet(cos_path)
            df = df.merge(cdf, on="pair_id", how="left")
        # filter both lengths
        df = df[(df["candidate_text"].astype(str).str.len() >= MIN_CODE_LEN)
                & (df["editorial_text"].astype(str).str.len() >= MIN_CODE_LEN)]
        df = df[df["cosine"].notna()]
        df["decile"] = df["cosine"].apply(assign_decile)
        # pre-mark labelled rows so the sampler prefers them
        df["candidate_id"] = df["candidate_id"].astype(str)
        df["editorial_id"] = df["editorial_id"].astype(str)
        df["is_labelled"] = [
            (c, e) in lbl_key for c, e in zip(df["candidate_id"], df["editorial_id"])
        ]
        target = min(TARGET_CELL_SIZE, len(df))
        samp = stratified_sample(df, target, rng)
        # attach labels
        samp["candidate_id"] = samp["candidate_id"].astype(str)
        samp["editorial_id"] = samp["editorial_id"].astype(str)
        samp["claude_label"] = [
            lbl_key.get((c, e), -1) for c, e in zip(samp["candidate_id"], samp["editorial_id"])
        ]
        samp["label_status"] = np.where(samp["claude_label"] >= 0, "labelled", "needs_labeling")
        out_path = OUT_CELLS / f"{plat}_pairs.parquet"
        samp.to_parquet(out_path, index=False)
        decile_dist = samp["decile"].value_counts().sort_index().to_dict()
        decile_dist = {DECILE_LABELS[d]: int(n) for d, n in decile_dist.items() if d >= 0}
        summary[plat] = {
            "pool_available": int(len(df)),
            "pool_problems": int(df["canonical_pid"].nunique()),
            "sampled_n": int(len(samp)),
            "sampled_problems": int(samp["canonical_pid"].nunique()),
            "labelled": int((samp["claude_label"] >= 0).sum()),
            "to_label": int((samp["claude_label"] == -1).sum()),
            "decile_dist": decile_dist,
            "pos_rate_labelled_subset": (
                float(samp.loc[samp["claude_label"] >= 0, "claude_label"].mean())
                if (samp["claude_label"] >= 0).any() else None
            ),
        }
        print(f"[sample] {plat}: n={len(samp)} problems={samp['canonical_pid'].nunique()} "
              f"labelled={summary[plat]['labelled']} to_label={summary[plat]['to_label']}")
    (OUT_CELLS / "sample_summary.json").write_text(json.dumps(summary, indent=2))


# ============================================================
# PHASE 3 -- SHARD UNLABELED PAIRS
# ============================================================

PAIRS_PER_SHARD = 100


def phase_shard() -> None:
    OUT_SHARDS.mkdir(parents=True, exist_ok=True)
    (OUT_SHARDS / "shards").mkdir(exist_ok=True)
    # carry over R4 prompt verbatim
    if R4_PROMPT.exists():
        shutil.copy(R4_PROMPT, OUT_SHARDS / "r4_prompt.txt")
    counts = {}
    for plat in ("cc", "lc", "cf", "ac"):
        pair_path = OUT_CELLS / f"{plat}_pairs.parquet"
        if not pair_path.exists():
            continue
        df = pd.read_parquet(pair_path)
        todo = df[df["claude_label"] == -1].copy()
        if len(todo) == 0:
            counts[plat] = {"to_label": 0, "shards": 0}
            continue
        # synthesize the 5K-shard record format
        recs = []
        for _, r in todo.iterrows():
            recs.append({
                "pair_id": str(r["pair_id"]),
                "problem_id": str(r["canonical_pid"]),
                "source": plat,
                "editorial_id": str(r["editorial_id"]),
                "candidate_id": str(r["candidate_id"]),
                "language": str(r.get("candidate_lang") or "unknown"),
                "cosine": float(r["cosine"]) if pd.notna(r["cosine"]) else 0.0,
                "decile": DECILE_LABELS[int(r["decile"])],
                "editorial_text": str(r["editorial_text"]),
                "candidate_text": str(r["candidate_text"]),
            })
        n_shards = math.ceil(len(recs) / PAIRS_PER_SHARD)
        for s in range(n_shards):
            chunk = recs[s * PAIRS_PER_SHARD:(s + 1) * PAIRS_PER_SHARD]
            with open(OUT_SHARDS / "shards" / f"shard_{plat}_{s:02d}.jsonl", "w") as f:
                for rec in chunk:
                    f.write(json.dumps(rec) + "\n")
        counts[plat] = {"to_label": len(recs), "shards": n_shards}
        print(f"[shard] {plat}: {len(recs)} unlabelled -> {n_shards} shards")
    readme = ["# comp_fourplatform Claude-labeling shards", "",
              "Format: identical to comp_unified_claude5k_shards/shards/shard_NN.jsonl.",
              "Each line is a JSON record with keys "
              "{pair_id, problem_id, source, editorial_id, candidate_id, language, "
              "cosine, decile, editorial_text, candidate_text}.",
              "", "Run the existing worker.py from comp_unified_claude5k_shards/ on each shard, "
              "passing r4_prompt.txt as the prompt file.", "",
              "## Counts per platform", ""]
    readme.append("| platform | pairs to label | shards |")
    readme.append("|---|---:|---:|")
    for plat in ("cc", "lc", "cf", "ac"):
        c = counts.get(plat, {"to_label": 0, "shards": 0})
        readme.append(f"| {plat} | {c['to_label']} | {c['shards']} |")
    (OUT_SHARDS / "README.md").write_text("\n".join(readme))
    (OUT_SHARDS / "shard_counts.json").write_text(json.dumps(counts, indent=2))


# ============================================================
# PHASE 4 -- BANK SCORING ENTRY (SK3-only; defers to existing scorer)
# ============================================================

def phase_score() -> None:
    """Re-run the metric bank on the sampled pairs.

    The existing scorer (scripts/comp_unified_claude_bank_score.py and
    scripts/comp_unified_claude_bank_score_new_metrics.py) ingests text via the
    5K-style JSONL shard layout. For minimal divergence, we DO NOT re-implement
    bank scoring here. Instead we emit an "_input" parquet per platform with
    the schema [pair_id, candidate_text, candidate_lang, editorial_text], and
    invoke the bank as a subprocess if the scorer is import-able.

    For now this phase emits the bank-input parquets; the actual scoring is run
    by the operator with the existing scorer on sk3 (it streams ProcessPool
    across the metric-bank predict_fns, which are sk3-only).
    """
    for plat in ("cc", "lc", "cf", "ac"):
        pair_path = OUT_CELLS / f"{plat}_pairs.parquet"
        if not pair_path.exists():
            continue
        df = pd.read_parquet(pair_path)
        bi = pd.DataFrame({
            "pair_id": df["pair_id"].astype(str),
            "platform": plat,
            "candidate_text": df["candidate_text"].astype(str),
            "candidate_lang": df.get("candidate_lang", "unknown").fillna("unknown").astype(str),
            "editorial_text": df["editorial_text"].astype(str),
            "claude_label": df["claude_label"].astype(int),
            "label_status": df["label_status"].astype(str),
        })
        out_path = OUT_CELLS / f"{plat}_bank_input.parquet"
        bi.to_parquet(out_path, index=False)
        print(f"[score] {plat}: wrote bank-input -> {out_path} (n={len(bi)})")
    print("[score] Now run: python scripts/comp_unified_claude_bank_score.py "
          "with INPUT_PATHS pointing at comp_fourplatform_cells/{plat}_bank_input.parquet "
          "for each platform. Output goes to comp_fourplatform_cells/{plat}_bank_scores.parquet.")


# ============================================================
# PHASE 5 -- EVAL ON LABELLED SUBSETS
# ============================================================

def phase_eval() -> None:
    """Comparable LR + RF grouped-CV AUC table.

    Only uses CC + LC labelled subsets since CF + AC labels are not yet
    available (Phase 3 staged them). Re-uses comp_unified_claude_bank_scores_v2
    for the 1004 5K-shard pairs (CC+Luogu), and lc_python_metric_scores for
    the LC 2K pairs.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def safe_auc(y, p):
        y = np.asarray(y); p = np.asarray(p)
        if len(np.unique(y)) < 2:
            return float("nan")
        try:
            return float(roc_auc_score(y, p))
        except Exception:
            return float("nan")

    def grouped_cv(X, y, groups, model_kind):
        try:
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
            splits = list(sgkf.split(X, y, groups))
            ok = all(len(np.unique(y[te])) >= 2 and len(np.unique(y[tr])) >= 2
                     for tr, te in splits)
            if not ok:
                raise ValueError
        except Exception:
            gkf = GroupKFold(n_splits=5)
            splits = list(gkf.split(X, y, groups))
        oof = np.full(len(y), np.nan)
        for tr, te in splits:
            if model_kind == "lr":
                pipe = Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")),
                ])
            else:
                pipe = Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("rf", RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                                  n_jobs=-1, class_weight="balanced",
                                                  random_state=SEED)),
                ])
            pipe.fit(X[tr], y[tr])
            oof[te] = pipe.predict_proba(X[te])[:, 1]
        m = ~np.isnan(oof)
        return safe_auc(y[m], oof[m])

    rows = []

    # ----- CC + Luogu via the 1004-pair bank scores parquet -----
    bank = pd.read_parquet(OUT_BASE / "comp_unified_claude_bank_scores_v2.parquet")
    # Need canonical_pid for grouping: pull from 5K shards
    in5k = {}
    for p in SHARDS_5K_INPUT:
        for ln in open(p):
            r = json.loads(ln)
            in5k[r["pair_id"]] = r.get("problem_id")
    bank["canonical_pid"] = bank["pair_id"].map(in5k).fillna(bank["pair_id"])
    score_cols = sorted(c for c in bank.columns if c.endswith("_score"))
    for plat, sub in bank.groupby("source"):
        X = sub[score_cols].values.astype(float)
        y = sub["claude_label"].values.astype(int)
        g = sub["canonical_pid"].astype(str).values
        lr = grouped_cv(X, y, g, "lr")
        rf = grouped_cv(X, y, g, "rf")
        rows.append({
            "platform": plat, "n": int(len(sub)),
            "n_problems": int(pd.unique(g).size),
            "pos_rate": float(y.mean()),
            "lr_auc": lr, "rf_auc": rf,
        })

    # ----- LC-2K via lc_python_metric_scores -----
    lcm = pd.read_parquet(OUT_BASE / "lc_python_metric_scores.parquet")
    lcs = pd.read_parquet(LC_2K_SAMPLE)[["candidate_id", "pair_id", "question_slug"]]
    lcl = pd.read_parquet(LC_2K_LABELS)
    lcm = lcm.merge(lcs, on="candidate_id", how="inner")
    lcm = lcm.merge(lcl, on="pair_id", how="inner")
    lc_score_cols = sorted(c for c in lcm.columns if c.endswith("_score"))
    X = lcm[lc_score_cols].values.astype(float)
    y = lcm["label"].values.astype(int)
    g = lcm["question_slug"].astype(str).values
    lr = grouped_cv(X, y, g, "lr")
    rf = grouped_cv(X, y, g, "rf")
    rows.append({
        "platform": "lc", "n": int(len(lcm)),
        "n_problems": int(pd.unique(g).size),
        "pos_rate": float(y.mean()),
        "lr_auc": lr, "rf_auc": rf,
    })

    eval_df = pd.DataFrame(rows)
    eval_df.to_parquet(OUT_BASE / "comp_fourplatform_eval.parquet", index=False)
    print(eval_df.to_string(index=False))

    # ----- write status markdown -----
    write_status_md(eval_df)


def write_status_md(eval_df: pd.DataFrame) -> None:
    summary_path = OUT_CELLS / "sample_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    shard_counts_path = OUT_SHARDS / "shard_counts.json"
    shard_counts = json.loads(shard_counts_path.read_text()) if shard_counts_path.exists() else {}

    lines = []
    lines.append("# Four-platform comparable editorial-similarity status")
    lines.append("")
    lines.append("Frozen recipe: pair pool per canonical_pid (both codes >= 30 chars), "
                 "bge-code-v1 cosines, decile-stratified sample matched to the 5K source "
                 "distribution (<0.2: 13.7%, 0.2-0.4: 14.6%, 0.4-0.6: 17.6%, 0.6-0.8: 31.2%, "
                 "0.8-1.0: 22.8%), cap 4 pairs/problem, target min(1000, available). "
                 "Labels reused where pairs overlap with prior Claude R4 runs; otherwise "
                 "staged for fresh labeling. Eval: StratifiedGroupKFold(5) by canonical_pid, "
                 "LR + RF, identical preprocessing.")
    lines.append("")
    lines.append("## Pool + sample stats")
    lines.append("")
    lines.append("| platform | pool pairs | pool problems | sampled | sampled problems | labelled | to_label | pos rate (labelled) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for plat in ("cc", "lc", "cf", "ac"):
        s = summary.get(plat, {})
        if not s:
            lines.append(f"| {plat} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        pr = s.get("pos_rate_labelled_subset")
        pr_s = f"{pr:.3f}" if pr is not None else "n/a"
        lines.append(
            f"| {plat} | {s['pool_available']} | {s['pool_problems']} | {s['sampled_n']} | "
            f"{s['sampled_problems']} | {s['labelled']} | {s['to_label']} | {pr_s} |"
        )
    lines.append("")
    lines.append("## Comparable AUC (labelled subsets, StratifiedGroupKFold(5))")
    lines.append("")
    lines.append("| platform | n | n_problems | pos_rate | LR AUC | RF AUC |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in eval_df.iterrows():
        lines.append(f"| {r['platform']} | {r['n']} | {r['n_problems']} | "
                     f"{r['pos_rate']:.3f} | {r['lr_auc']:.4f} | {r['rf_auc']:.4f} |")
    lines.append("")
    lines.append("## Shards staged for Claude labeling")
    lines.append("")
    lines.append("| platform | pairs to label | shards |")
    lines.append("|---|---:|---:|")
    for plat in ("cc", "lc", "cf", "ac"):
        c = shard_counts.get(plat, {"to_label": 0, "shards": 0})
        lines.append(f"| {plat} | {c['to_label']} | {c['shards']} |")
    lines.append("")
    lines.append("## Blockers / next steps")
    lines.append("")
    lines.append("- CF + AC bank scoring requires running comp_unified_claude_bank_score on "
                 "comp_fourplatform_cells/{cf,ac}_bank_input.parquet on sk3.")
    lines.append("- CF + AC AUC numbers populate only after Claude shards above are labelled.")
    lines.append("- CC + LC AUC rows in the eval table use the existing labelled cells. They "
                 "are the directly comparable numbers under the frozen recipe.")
    OUT_STATUS.write_text("\n".join(lines))
    print(f"[status] wrote {OUT_STATUS}")


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["pool", "embed", "sample", "shard", "score", "eval", "all"])
    args = ap.parse_args()
    if args.phase in ("pool", "all"):
        phase_pool()
    if args.phase in ("embed", "all"):
        phase_embed_cfac()
    if args.phase in ("sample", "all"):
        phase_sample()
    if args.phase in ("shard", "all"):
        phase_shard()
    if args.phase in ("score", "all"):
        phase_score()
    if args.phase in ("eval", "all"):
        phase_eval()


if __name__ == "__main__":
    main()
