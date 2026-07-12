"""Score the single-prompt rungs (metric NAME, merged_description) for EVERY CW R3 metric with a
certificate row — the per-metric 'best single prompt' needed to compute Δ_upper = (OPT_Ω+ε) − R_single
across the whole bank (prompt-optimality notebook §6).

One executor load; for each metric: R_name = I(M_i; binarize(name verdict)), R_desc likewise, on the
same frozen 300 probes the certificate used. ~5 min on one GPU, no GLM.

  CUDA_VISIBLE_DEVICES=1 HOME=/lfs/skampere3/0/alexspan VLLM_GPU_MEM_UTIL=0.9 \\
    python -m methods.metric_implementer.experiments.score_name_rungs
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

from ..config import ImplementerConfig, apply_task_preset
from ..vllm_backend import make_judge_backend
from . import alpha_probe as aprobe
from .mine_clusters import r3_groups
from .run_real_test import _load_texts
from .value_census import i_binary


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="creative-writing")
    p.add_argument("--bucket", default="general")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--gepa-reserve", type=int, default=60)
    p.add_argument("--n-probes", type=int, default=300)
    p.add_argument("--npz-dir", default="/lfs/skampere3/0/alexspan/outputs/r3_cw/aligned_8b_orbit_v2")
    p.add_argument("--out", default="/lfs/skampere3/0/alexspan/tmp_vinfo/gepa_for_plot/name_rungs_cw.json")
    args = p.parse_args(argv)

    groups = r3_groups(args.task, args.bucket)
    cfg = ImplementerConfig()
    apply_task_preset(cfg, args.task)
    cfg.n_oracle_items = 0
    all_texts, _ = _load_texts(args.task, args.gepa_reserve + args.n_probes, cfg)
    probes = all_texts[args.gepa_reserve: args.gepa_reserve + args.n_probes]
    executor = make_judge_backend(args.model, cfg, 0.0)
    max_chars = getattr(cfg, "max_text_chars", 4000)

    def score_R(prompt_text, M):
        pyes = aprobe.signature(executor, prompt_text, probes, max_chars, template=None)
        v = (np.nan_to_num(np.asarray(pyes, float), nan=0.5) > 0.5).astype(int)
        return float(i_binary(M, v)), int(v.sum())

    rows = []
    for f in sorted(glob.glob(f"{args.npz_dir}/*_sigs.npz")):
        m = re.search(r"metric(\d+)_sigs", f)
        if not m:
            continue
        gi = int(m.group(1))
        z = np.load(f, allow_pickle=True)
        if "M_i" not in z.files:
            continue
        M = (np.asarray(z["M_i"], float) > 0.5).astype(int)
        if len(M) != len(probes):
            print(f"[rungs] gi={gi}: M_i len {len(M)} != {len(probes)}, skip", flush=True)
            continue
        g = groups[gi]
        name, desc = g["merged_name"], g["merged_description"]
        Rn, yn = score_R(name, M)
        Rd, yd = score_R(desc, M)
        rows.append({"gi": gi, "name": name, "H_M_base_rate": float(M.mean()),
                     "R_name": Rn, "nyes_name": yn, "R_desc": Rd, "nyes_desc": yd})
        print(f"[rungs] gi={gi:2d} R_name={Rn:.3f} R_desc={Rd:.3f}  {name[:60]}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"task": args.task, "bucket": args.bucket, "model": args.model,
               "n_probes": len(probes), "rows": rows}, open(args.out, "w"), indent=1)
    print(f"[rungs] wrote {args.out} ({len(rows)} metrics)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
