"""Phase 2 step 3b: per-app sim-stratified random candidates.

Why: top-200 mining (step 3) only yields HIGH-sim fillers (floor ~0.55-0.6), but
real examiner cites live lower (p50 ~0.545, p10 ~0.37). Distribution matching
clamped at the candidate floor -> rejected apps betray themselves via one
low-sim attachment (sim_std AUC leak ~0.61). Fix: give stage C candidates that
span the FULL sim range.

Method: one shared random subset R of 100K pool docs (year>0 only). For each
app, sims vs R on GPU, keep 256 evenly-spaced order statistics -> parquet
shards mirroring top200_shard schema (app_id, pool_row, sim).
"""
import os, time
import numpy as np
import pandas as pd
import torch

BASE = "/lfs/skampere3/0/alexspan/norm-research"
IDX_DIR = f"{BASE}/indexes/patent_claims_v3"
CAND_DIR = f"{BASE}/datasets/patents/processed/phase2_filler_candidates"
V0_DIR = f"{BASE}/datasets/patents/processed/phase2_dataset_v0"
N_RAND = 100_000
N_KEEP = 256
SEED = 0

pool = np.load(f"{IDX_DIR}/pool_fp16.npy", mmap_mode="r")
years = np.load(f"{V0_DIR}/pool_years.npy")
rng = np.random.default_rng(SEED)
valid = np.where(years > 0)[0]
R_rows = np.sort(rng.choice(valid, N_RAND, replace=False))
print(f"shared random subset: {N_RAND:,} of {len(valid):,} year-known pool docs", flush=True)

dev = "cuda"
Rg = torch.from_numpy(pool[R_rows].astype(np.float16)).to(dev)  # (100K, 1024)
R_rows_t = torch.from_numpy(R_rows).to(dev)

Q = np.load(f"{CAND_DIR}/app_claim_emb_fp16.npy", mmap_mode="r")
qids = [l.split("\t")[0] for l in open(f"{CAND_DIR}/app_ids.txt")]
nq = Q.shape[0]
keep_idx = torch.linspace(0, N_RAND - 1, N_KEEP).long().to(dev)

Q_CH = 8192
t0 = time.time()
done = {int(f.split("_")[-1].split(".")[0]) for f in os.listdir(CAND_DIR)
        if f.startswith("rand_shard_")}
for qs in range(0, nq, Q_CH * 2):  # shard = 16,384 apps (matches top200 shards)
    si = qs // (Q_CH * 2)
    if si in done:
        continue
    qe = min(qs + Q_CH * 2, nq)
    rows_out, sims_out = [], []
    for iqs in range(qs, qe, Q_CH):
        iqe = min(iqs + Q_CH, qe)
        Qg = torch.from_numpy(np.ascontiguousarray(Q[iqs:iqe])).to(dev)
        sims = (Qg @ Rg.T).float()                # (q, 100K)
        srt = torch.sort(sims, dim=1)
        sims_out.append(srt.values[:, keep_idx].cpu().numpy())
        rows_out.append(R_rows_t[srt.indices[:, keep_idx]].cpu().numpy())
        del Qg, sims, srt
    sims_np = np.concatenate(sims_out)
    rows_np = np.concatenate(rows_out)
    df = pd.DataFrame({
        "app_id": np.repeat(qids[qs:qe], N_KEEP),
        "pool_row": rows_np.ravel(),
        "sim": sims_np.ravel().astype(np.float32),
    })
    df.to_parquet(f"{CAND_DIR}/rand_shard_{si:04d}.parquet", index=False)
    print(f"  rand shard {si}: apps {qe:,}/{nq:,}  ({(time.time()-t0)/60:.1f} min)", flush=True)

print("Step 3b done.", flush=True)
