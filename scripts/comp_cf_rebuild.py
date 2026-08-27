"""CF cell rebuild from exec-gated rendered-HTML editorials (2026-06-11).

Mirrors the frozen 4-platform recipe (scripts/comp_fourplatform_build.py):
  pool   : (organic candidate x gated editorial) per canonical_pid, both >= 30
           chars; candidates from competition_unified (matrixstudio /
           code_contests / taco — NO editorial-derived candidates; we also
           hard-drop any candidate whose normalized code equals an editorial's).
  embed  : bge-code-v1, cosine for STRATIFICATION ONLY (never a feature).
  sample : decile-stratified to the frozen 5K source distribution
           (13.7/14.6/17.6/31.2/22.8), cap 4 pairs/problem, target 2500.
  shard  : 250-pair JSONL shards in the 5K record format for L1 labeling.

Editorial side: datasets/codeforces_delta/editorials_rendered_gated.parquet
filtered to gate_pass == True (compiled + 100% of its problem's TACO tests).

Outputs: outputs/v2_analysis/cf_rebuild_cell/
  cf2_pool.parquet  cf2_pool_cosine.parquet  cf2_pairs.parquet
  cf2_bank_input.parquet  shards/shard_cf2_NN.jsonl  build_summary.json

Phases: pool | embed | sample | shard | all      Seed = 42.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
LAPTOP_ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
SK3_ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
ROOT = SK3_ROOT if SK3_ROOT.exists() else LAPTOP_ROOT

GATED = ROOT / "datasets/codeforces_delta/editorials_rendered_gated.parquet"
CANDS = ROOT / "datasets/competition_unified/candidates.parquet"
OUT = ROOT / "outputs/v2_analysis/cf_rebuild_cell"

DECILE_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]
DECILE_LABELS = ["[0.0-0.2)", "[0.2-0.4)", "[0.4-0.6)", "[0.6-0.8)", "[0.8-1.0]"]
TARGET_DECILE_FRACTIONS = {0: 0.137, 1: 0.146, 2: 0.176, 3: 0.312, 4: 0.228}
TARGET_CELL_SIZE = 2500
MAX_PAIRS_PER_PROBLEM = 4
MIN_CODE_LEN = 30
PAIRS_PER_SHARD = 250
MAX_EDITORIALS_PER_PID = 2


def norm_code(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def phase_pool():
    OUT.mkdir(parents=True, exist_ok=True)
    ed = pd.read_parquet(GATED)
    ed = ed[ed["gate_pass"] == True].copy()  # noqa: E712
    ed = ed[ed["extracted_code"].str.len() >= MIN_CODE_LEN]
    # dedup identical (pid, code); prefer anchor sections, then more tests passed
    ed["method_rank"] = ed["section_method"].map(
        lambda m: 0 if m == "anchor" else (1 if str(m).startswith("text") else 2))
    ed["code_norm"] = ed["extracted_code"].map(norm_code)
    ed = ed.sort_values(["method_rank", "n_pass"], ascending=[True, False])
    ed = ed.drop_duplicates(["canonical_pid", "code_norm"])
    # cap editorial variants per pid
    ed = ed.groupby("canonical_pid", group_keys=False).head(MAX_EDITORIALS_PER_PID)
    ed["editorial_id"] = [
        "edr_" + hashlib.md5(f"{b}|{p}|{c[:2000]}".encode()).hexdigest()[:14]
        for b, p, c in zip(ed["blog_id"], ed["canonical_pid"], ed["extracted_code"])
    ]
    print(f"[pool] gated editorials: {len(ed)} rows / "
          f"{ed['canonical_pid'].nunique()} problems")

    cn = pd.read_parquet(CANDS, columns=[
        "platform", "canonical_pid", "candidate_id", "code", "code_lang",
        "language_norm", "source", "verdict"])
    cn = cn[cn["platform"] == "cf"].copy()
    cn["code"] = cn["code"].fillna("")
    cn = cn[cn["code"].str.len() >= MIN_CODE_LEN].drop_duplicates("candidate_id")
    pids = set(ed["canonical_pid"]) & set(cn["canonical_pid"])
    ed = ed[ed["canonical_pid"].isin(pids)]
    cn = cn[cn["canonical_pid"].isin(pids)].copy()
    print(f"[pool] joinable problems: {len(pids)}; candidates in them: {len(cn)}")
    # hard-exclude editorial-derived candidates (normalized exact match)
    ed_norms = set(ed["code_norm"])
    cn["code_norm"] = cn["code"].map(norm_code)
    n_before = len(cn)
    cn = cn[~cn["code_norm"].isin(ed_norms)]
    print(f"[pool] dropped {n_before - len(cn)} candidates identical to an editorial")
    # cap candidates per problem to keep the pool tractable (sampled later anyway)
    rng = np.random.default_rng(SEED)
    cn = cn.sample(frac=1.0, random_state=SEED)
    cn = cn.groupby("canonical_pid", group_keys=False).head(40)

    pairs = cn.merge(
        ed[["editorial_id", "canonical_pid", "extracted_code", "code_lang"]],
        on="canonical_pid", how="inner", suffixes=("", "_ed"))
    pairs = pairs.rename(columns={
        "code": "candidate_text", "extracted_code": "editorial_text",
        "language_norm": "candidate_lang", "code_lang_ed": "editorial_lang",
        "source": "candidate_source"})
    h = (pairs["candidate_id"].astype(str) + "|" + pairs["editorial_id"]).map(
        lambda s: hashlib.md5(s.encode()).hexdigest()[:14])
    pairs["pair_id"] = "cf2_" + h
    pairs["platform"] = "cf"
    keep = ["pair_id", "platform", "canonical_pid", "candidate_id", "editorial_id",
            "candidate_text", "editorial_text", "candidate_lang", "editorial_lang",
            "candidate_source", "verdict"]
    pairs = pairs[keep].drop_duplicates("pair_id").reset_index(drop=True)
    pairs.to_parquet(OUT / "cf2_pool.parquet", index=False)
    print(f"[pool] {len(pairs)} pairs, {pairs['canonical_pid'].nunique()} problems "
          f"-> {OUT/'cf2_pool.parquet'}")


def phase_embed():
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    cos_path = OUT / "cf2_pool_cosine.parquet"
    if cos_path.exists():
        print("[embed] cosines exist, skipping")
        return
    assert torch.cuda.is_available(), "need a GPU for bge-code-v1"
    BATCH, MAX_LEN = 64, 1024

    def last_token_pool(h, m):
        if (m[:, -1].sum() == m.shape[0]).item():
            return h[:, -1]
        seq_lens = m.sum(dim=1) - 1
        return h[torch.arange(h.size(0), device=h.device), seq_lens]

    tok = AutoTokenizer.from_pretrained("BAAI/bge-code-v1", trust_remote_code=True)
    mdl = AutoModel.from_pretrained("BAAI/bge-code-v1", trust_remote_code=True,
                                    torch_dtype=torch.bfloat16).to("cuda").eval()

    def embed_unique(ids, texts):
        out = {}
        order = sorted(range(len(ids)), key=lambda i: len(texts[i]))
        ids_s = [ids[i] for i in order]
        txt_s = [texts[i] if texts[i] else "EMPTY" for i in order]
        t0 = time.time()
        with torch.inference_mode():
            for b in range(0, len(ids_s), BATCH):
                bi, bt = ids_s[b:b + BATCH], txt_s[b:b + BATCH]
                enc = tok(bt, max_length=MAX_LEN, padding=True, truncation=True,
                          return_tensors="pt", pad_to_multiple_of=8).to("cuda")
                o = mdl(**enc)
                e = F.normalize(
                    last_token_pool(o.last_hidden_state, enc["attention_mask"]).float(),
                    p=2, dim=1)
                arr = e.cpu().numpy().astype(np.float32)
                for k, i in enumerate(bi):
                    out[str(i)] = arr[k]
                if b % (BATCH * 50) == 0:
                    rate = (b + len(bi)) / max(time.time() - t0, 1e-3)
                    print(f"  {b+len(bi)}/{len(ids_s)} rate={rate:.1f}/s", flush=True)
        return out

    df = pd.read_parquet(OUT / "cf2_pool.parquet")
    ed_u = df.drop_duplicates("editorial_id")[["editorial_id", "editorial_text"]]
    cn_u = df.drop_duplicates("candidate_id")[["candidate_id", "candidate_text"]]
    print(f"[embed] {len(ed_u)} unique editorials, {len(cn_u)} unique candidates")
    ed_emb = embed_unique(ed_u["editorial_id"].tolist(), ed_u["editorial_text"].tolist())
    cn_emb = embed_unique(cn_u["candidate_id"].tolist(), cn_u["candidate_text"].tolist())
    cos = np.full(len(df), np.nan, dtype=np.float32)
    eids = df["editorial_id"].astype(str).to_numpy()
    cids = df["candidate_id"].astype(str).to_numpy()
    for k in range(len(df)):
        e, c = ed_emb.get(eids[k]), cn_emb.get(cids[k])
        if e is not None and c is not None:
            cos[k] = float(np.dot(e, c))
    pd.DataFrame({"pair_id": df["pair_id"].values, "cosine": cos}).to_parquet(
        cos_path, index=False)
    print(f"[embed] wrote {cos_path} (missing={int(np.isnan(cos).sum())})")


def assign_decile(c: float) -> int:
    if math.isnan(c):
        return -1
    for i, (lo, hi) in enumerate(DECILE_BINS):
        if lo <= c < hi:
            return i
    return 4 if c >= 1.0 else 0


def phase_sample():
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(OUT / "cf2_pool.parquet")
    cdf = pd.read_parquet(OUT / "cf2_pool_cosine.parquet")
    df = df.merge(cdf, on="pair_id", how="left")
    df = df[df["cosine"].notna()]
    df["decile"] = df["cosine"].map(assign_decile)
    df = df[df["decile"] >= 0]
    target = min(TARGET_CELL_SIZE, len(df))
    plan = {d: int(round(target * f)) for d, f in TARGET_DECILE_FRACTIONS.items()}
    plan[max(plan, key=plan.get)] += target - sum(plan.values())
    picked = []
    for d, want in plan.items():
        sub = df[df["decile"] == d].sample(frac=1.0,
                                           random_state=int(rng.integers(0, 2**31)))
        per_problem, chosen = {}, []
        for _, r in sub.iterrows():
            if len(chosen) >= want:
                break
            pid = r["canonical_pid"]
            if per_problem.get(pid, 0) >= MAX_PAIRS_PER_PROBLEM:
                continue
            chosen.append(r)
            per_problem[pid] = per_problem.get(pid, 0) + 1
        if len(chosen) < want:  # relax problem cap as last resort
            got = {r["pair_id"] for r in chosen}
            for _, r in sub.iterrows():
                if len(chosen) >= want:
                    break
                if r["pair_id"] not in got:
                    chosen.append(r)
        picked.extend(chosen)
    samp = pd.DataFrame(picked).reset_index(drop=True)
    samp["claude_label"] = -1
    samp["label_status"] = "needs_labeling"
    samp.to_parquet(OUT / "cf2_pairs.parquet", index=False)
    summary = {
        "pool_available": int(len(df)),
        "pool_problems": int(df["canonical_pid"].nunique()),
        "sampled_n": int(len(samp)),
        "sampled_problems": int(samp["canonical_pid"].nunique()),
        "decile_dist": {DECILE_LABELS[d]: int(n) for d, n in
                        samp["decile"].value_counts().sort_index().items()},
        "lang_dist": samp["candidate_lang"].value_counts().to_dict(),
        "source_dist": samp["candidate_source"].value_counts().to_dict(),
    }
    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    # bank input parquet (same schema as comp_fourplatform phase_score)
    bi = pd.DataFrame({
        "pair_id": samp["pair_id"].astype(str),
        "platform": "cf",
        "canonical_pid": samp["canonical_pid"].astype(str),
        "candidate_text": samp["candidate_text"].astype(str),
        "candidate_lang": samp["candidate_lang"].fillna("unknown").astype(str),
        "editorial_text": samp["editorial_text"].astype(str),
        "claude_label": samp["claude_label"].astype(int),
        "label_status": samp["label_status"].astype(str),
    })
    bi.to_parquet(OUT / "cf2_bank_input.parquet", index=False)
    print(f"[sample] wrote bank input n={len(bi)}")


def phase_shard():
    (OUT / "shards").mkdir(exist_ok=True)
    samp = pd.read_parquet(OUT / "cf2_pairs.parquet")
    recs = []
    for _, r in samp.iterrows():
        recs.append({
            "pair_id": str(r["pair_id"]),
            "problem_id": str(r["canonical_pid"]),
            "source": "cf",
            "editorial_id": str(r["editorial_id"]),
            "candidate_id": str(r["candidate_id"]),
            "language": str(r.get("candidate_lang") or "unknown"),
            "cosine": float(r["cosine"]),
            "decile": DECILE_LABELS[int(r["decile"])],
            "editorial_text": str(r["editorial_text"]),
            "candidate_text": str(r["candidate_text"]),
        })
    n_shards = math.ceil(len(recs) / PAIRS_PER_SHARD)
    for s in range(n_shards):
        chunk = recs[s * PAIRS_PER_SHARD:(s + 1) * PAIRS_PER_SHARD]
        with open(OUT / "shards" / f"shard_cf2_{s:02d}.jsonl", "w") as f:
            for rec in chunk:
                f.write(json.dumps(rec) + "\n")
    print(f"[shard] {len(recs)} pairs -> {n_shards} shards of {PAIRS_PER_SHARD}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["pool", "embed", "sample", "shard", "all"])
    a = ap.parse_args()
    if a.phase in ("pool", "all"):
        phase_pool()
    if a.phase in ("embed", "all"):
        phase_embed()
    if a.phase in ("sample", "all"):
        phase_sample()
    if a.phase in ("shard", "all"):
        phase_shard()
