#!/usr/bin/env python3
"""Post-hoc probe head (V2 gate + V3a survival readout).

A FRESH 2-layer MLP is trained on the FROZEN representations of a finished run
to recover a named target (the plant indicator, the real-signal token, or a
continuous nuisance channel).  Encoder weights are never touched -- the probe
only asks "is this channel still linearly/nonlinearly READABLE from the
representation the scalar head consumes?"

Honest split: probe fits on the run's own dense-TRAIN rows (10% of them held
back for early stopping) and is scored on the dense-EVAL rows.  3 seeds, mean
reported.  CPU is fine (4096 -> 256 -> 1 on <8K rows).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

HIDDEN = 256
SEEDS = (0, 1, 2)
EPOCHS = 60
LR = 1e-3
BATCH = 256


def fit_probe(Xtr, ytr, Xva, yva, Xte, yte, seed, device):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    lf = nn.BCEWithLogitsLoss()
    Xtr_t = torch.tensor(Xtr, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, device=device)
    Xte_t = torch.tensor(Xte, device=device)
    best_va, best_te, patience = -1.0, 0.5, 0
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            lf(net(Xtr_t[b])[:, 0], ytr_t[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            va = roc_auc_score(yva, net(Xva_t)[:, 0].cpu().numpy())
            te = roc_auc_score(yte, net(Xte_t)[:, 0].cpu().numpy())
        if va > best_va:
            best_va, best_te, patience = va, te, 0
        else:
            patience += 1
            if patience >= 10:
                break
    return float(best_te), float(best_va)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--nuisance", required=True)
    ap.add_argument("--targets", default="plant,realtok")
    ap.add_argument("--rep_key", default="rep",
                    help="'rep' = what the task head consumes (z under arch=bottleneck); "
                         "'rep_h' = pooled h alongside it (bottleneck scope check)")
    args = ap.parse_args()

    run = Path(args.run_dir)
    z = np.load(run / "reps.npz", allow_pickle=True)
    ids = np.array([str(s) for s in z["doc_id"]])
    rep = z[args.rep_key].astype(np.float32)
    if rep.size == 0:
        raise SystemExit(f"rep_key={args.rep_key} is empty in {run}/reps.npz")
    split = np.array([str(s) for s in z["split"]])
    nz = np.load(args.nuisance, allow_pickle=True)
    nz_ids = np.array([str(s) for s in nz["doc_id"]])
    order = {d: i for i, d in enumerate(nz_ids)}
    pos = np.array([order[d] for d in ids])
    names = [str(s) for s in nz["names"]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = rep[split == "train"].mean(0), rep[split == "train"].std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    X = (rep - mu) / sd

    out = {"run_dir": str(run), "n": int(len(ids)), "device": device, "probes": {}}
    rng = np.random.default_rng(0)
    tr_all = np.flatnonzero(split == "train")
    hold = rng.permutation(len(tr_all))
    va_idx, tr_idx = tr_all[hold[: max(1, len(tr_all) // 10)]], tr_all[hold[len(tr_all) // 10:]]
    te_idx = np.flatnonzero(split == "eval")

    for tname in args.targets.split(","):
        tname = tname.strip()
        if not tname:
            continue
        if tname in ("plant", "realtok"):
            t = nz[tname].astype(int)[pos]
        elif tname in names:                     # continuous channel -> median split
            v = nz["raw"][pos, names.index(tname)]
            t = (v >= np.median(v[split == "train"])).astype(int)
        else:
            raise ValueError(tname)
        if len(set(t[te_idx])) < 2 or len(set(t[tr_idx])) < 2:
            out["probes"][tname] = {"auc_eval": None, "note": "degenerate target"}
            continue
        aucs = [fit_probe(X[tr_idx], t[tr_idx], X[va_idx], t[va_idx], X[te_idx], t[te_idx], s, device)
                for s in SEEDS]
        out["probes"][tname] = {
            "auc_eval_mean": float(np.mean([a for a, _ in aucs])),
            "auc_eval_seeds": [a for a, _ in aucs],
            "auc_innerval_mean": float(np.mean([b for _, b in aucs])),
            "target_rate": float(t.mean()),
        }
        print(f"  probe[{tname}] eval AUC = {out['probes'][tname]['auc_eval_mean']:.4f}", flush=True)

    out["rep_key"] = args.rep_key
    fname = "probe.json" if args.rep_key == "rep" else f"probe_{args.rep_key}.json"
    (run / fname).write_text(json.dumps(out, indent=2))
    print("PROBE " + json.dumps(out["probes"]), flush=True)


if __name__ == "__main__":
    main()
