"""Lean GEPA sweep across MANY CW R3 metrics — the per-iteration data behind the prompt-optimality
notebook's §7 aggregate ("how much does GEPA miss the certified ceiling, per round, across metrics").

Same loop as run_gepa_for_plot (generic seed, GLM reviser sees NAME + failures + calibration,
accept-if-better), minus the atomize/union step. One executor load for all metrics; trajectories with
full prompt text per accepted round go to one JSON.

GLM budget: len(gi_list) × rounds × candidates calls (12×5×2 = 120 short calls).

  CUDA_VISIBLE_DEVICES=1 HOME=/lfs/skampere3/0/alexspan VLLM_GPU_MEM_UTIL=0.9 \\
    python -m methods.metric_implementer.experiments.run_gepa_sweep --gi-list 24,7,38,44,...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..config import ImplementerConfig, apply_task_preset
from ..backends import LLMBackend
from ..vllm_backend import make_judge_backend
from . import alpha_probe as aprobe
from .mine_clusters import r3_groups
from .run_real_test import _load_texts
from .value_census import i_binary
from .run_gepa_for_plot import _SYS, _USER, _HINTS, _GENERIC_SEED, _strip_fences, _excerpt


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="creative-writing")
    p.add_argument("--bucket", default="general")
    p.add_argument("--gi-list", default="24,7,38,44,14,23,11,18,10,41,47,40")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--gepa-reserve", type=int, default=60)
    p.add_argument("--n-probes", type=int, default=300)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--candidates", type=int, default=2)
    p.add_argument("--glm-model", default="glm-5")
    p.add_argument("--seed-mode", default="description", choices=["description", "name", "generic"])
    p.add_argument("--npz-dir", default="/lfs/skampere3/0/alexspan/outputs/r3_cw/aligned_8b_orbit_v2")
    p.add_argument("--out", default="/lfs/skampere3/0/alexspan/tmp_vinfo/gepa_for_plot/gepa_sweep_cw.json")
    args = p.parse_args(argv)

    gis = [int(x) for x in args.gi_list.split(",")]
    groups = r3_groups(args.task, args.bucket)
    cfg = ImplementerConfig()
    apply_task_preset(cfg, args.task)
    cfg.n_oracle_items = 0
    all_texts, _ = _load_texts(args.task, args.gepa_reserve + args.n_probes, cfg)
    probes = all_texts[args.gepa_reserve: args.gepa_reserve + args.n_probes]
    executor = make_judge_backend(args.model, cfg, 0.0)
    max_chars = getattr(cfg, "max_text_chars", 4000)
    gcfg = ImplementerConfig()
    gcfg.backend = "zai_anthropic"
    gcfg.other_temperature = 0.8
    gcfg.request_timeout_s = 120
    glm = LLMBackend(model=args.glm_model, role="reviser", cfg=gcfg)

    t0 = time.time()
    metrics = []
    for gi in gis:
        npz = f"{args.npz_dir}/creative-writing_R3_metric{gi}_sigs.npz"
        z = np.load(npz, allow_pickle=True)
        M = (np.asarray(z["M_i"], float) > 0.5).astype(int)
        if len(M) != len(probes):
            print(f"[sweep] gi={gi}: len mismatch, skip", flush=True)
            continue
        name = groups[gi]["merged_name"]
        desc = groups[gi]["merged_description"]
        target_rate = float(M.mean())

        def score_R(prompt_text):
            pyes = aprobe.signature(executor, prompt_text, probes, max_chars, template=None)
            v = (np.nan_to_num(np.asarray(pyes, float), nan=0.5) > 0.5).astype(int)
            return float(i_binary(M, v)), v

        seed_text = {"description": desc, "name": name, "generic": _GENERIC_SEED}[args.seed_mode]
        traj = []
        R0, v0 = score_R(seed_text)
        traj.append({"round": 0, "prompt": seed_text, "R_bits": R0,
                     "n_yes": int(v0.sum()), "accepted": True})
        cur, vcur, Rcur = seed_text, v0, R0
        rng = np.random.default_rng(gi)
        for rnd in range(1, args.rounds + 1):
            disagree = np.where(vcur != M)[0]
            pos = list(rng.permutation([int(i) for i in disagree if M[i] == 1])[:4])
            neg = list(rng.permutation([int(i) for i in disagree if M[i] == 0])[:4])
            pos_blk = "\n".join(f"- {_excerpt(i, probes)}" for i in pos) or "(none this round)"
            neg_blk = "\n".join(f"- {_excerpt(i, probes)}" for i in neg) or "(none this round)"
            cur_rate = float(vcur.mean())
            direction = ("permissive (says YES too often)" if cur_rate > target_rate
                         else "strict (says NO too often)")
            best = None
            for ci in range(args.candidates):
                user = _USER.format(name=name, cur=cur, target_rate=target_rate,
                                    cur_rate=cur_rate, direction=direction,
                                    pos=pos_blk, neg=neg_blk, hint=_HINTS[ci % len(_HINTS)])
                new = None
                for _ in range(2):
                    try:
                        new = glm.generate(user, system=_SYS, max_tokens=700, temperature=0.8)
                        break
                    except Exception as e:
                        print(f"[sweep] gi={gi} r{rnd}c{ci} GLM fail: {str(e)[:80]}", flush=True)
                if not new:
                    continue
                Rn, vn = score_R(_strip_fences(new))
                if best is None or Rn > best[0]:
                    best = (Rn, _strip_fences(new), vn)
            accepted = bool(best and best[0] > Rcur)
            if accepted:
                Rcur, cur, vcur = best
            traj.append({"round": rnd, "prompt": cur, "R_bits": Rcur,
                         "n_yes": int(vcur.sum()), "accepted": accepted,
                         "best_cand_R": best[0] if best else None})
        R_name, _ = score_R(name)
        R_desc, _ = score_R(groups[gi]["merged_description"])
        metrics.append({"gi": gi, "name": name, "M_i_mean": target_rate,
                        "R_name": R_name, "R_desc": R_desc, "lineage": traj})
        print(f"[sweep] gi={gi:2d} done: R {R0:.3f}→{Rcur:.3f} (name {R_name:.3f}) "
              f"glm_calls={glm.stats.n_calls} {time.time()-t0:.0f}s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"task": args.task, "bucket": args.bucket, "model": args.model,
               "rounds": args.rounds, "candidates": args.candidates, "seed_mode": args.seed_mode,
               "n_probes": len(probes), "glm_calls": int(glm.stats.n_calls),
               "metrics": metrics, "elapsed_s": round(time.time() - t0, 1)},
              open(out, "w"), indent=1)
    print(f"[sweep] wrote {out} ({len(metrics)} metrics, {glm.stats.n_calls} GLM calls)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
