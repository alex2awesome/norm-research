"""Rescore a SOURCE checkpoint's frozen criteria (Ω) with a DIFFERENT executor — apples-to-apples
cross-executor capture-recapture B_E, GPU-only (NO proposer / openrouter needed).

The source checkpoint stores `prompts` (the criteria) + `tags` (their proposer families). We reload
the same probe texts + the metric's merged_description, score every criterion with the TARGET executor
to get new soft signatures, and save a rescored checkpoint. Because Ω is IDENTICAL across executors,
the resulting B_E/D_obs/coverage are directly comparable across the ladder (no proposer-draw confound) —
this is the cleanest scaling-law input, and it sidesteps a dead/pricey proposer API.

Example:
    python -m methods.metric_implementer.experiments.rescore_executor \
        --src-dir /lfs/.../crc_scaling/llama8b --target-model meta-llama/Llama-3.3-70B-Instruct \
        --out-dir /lfs/.../crc_rescore/llama8b_to_llama70b --task creative-writing
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os

import numpy as np

from .. import config as cfgmod
from ..vllm_backend import make_judge_backend
from . import alpha_probe as ap
from .mine_clusters import r2_groups
from .run_real_test import _load_texts


def main(argv=None):
    p = argparse.ArgumentParser(prog="rescore_executor", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-dir", required=True, help="source executor dir with valid *_sigs.npz (the Ω source)")
    p.add_argument("--target-model", required=True, help="executor to RESCORE with")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--task", default="creative-writing")
    p.add_argument("--r2-bucket", default="general")
    p.add_argument("--n-metrics", type=int, default=0, help="0 = all source ckpts")
    p.add_argument("--n-probes", type=int, default=300)
    p.add_argument("--gepa-reserve", type=int, default=60)
    p.add_argument("--cmi-thresh", type=float, default=0.15)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--orbit-target", type=int, default=0,
                   help=">1 = recompute M_i as the §12.6.2 orbit-averaged m_bar_omega over n_forms rubric "
                        "reformulations (saves M_i_var_phi/M_i_flip_rate/orbit_forms); 0 = single-form metric_verdict")
    p.add_argument("--retarget-mi-only", action="store_true",
                   help="reuse the src ckpt's sigs/tags/prompts and recompute ONLY M_i (orbit if "
                        "--orbit-target>1). Same-executor orbit upgrade -- no criteria re-scoring, ~200x "
                        "cheaper than full rescore; --target-model MUST match the src ckpt's executor, and "
                        "--n-probes MUST match the src run so Mi aligns with the reused sigs")
    p.add_argument("--fake", action="store_true",
                   help="FakeVLLM dry-run (no GPU) -- validates orbit/retarget wiring + savez fields")
    a = p.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)

    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    ecfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    if a.fake:
        ecfg.vllm_fake = True
    executor = make_judge_backend(a.target_model, ecfg, temperature=None)
    probe_texts, _ = _load_texts(a.task, a.gepa_reserve + a.n_probes, cfg)
    probe_texts = probe_texts[a.gepa_reserve: a.gepa_reserve + a.n_probes]
    groups = r2_groups(a.task, a.r2_bucket)
    src = sorted(glob.glob(os.path.join(a.src_dir, "*_sigs.npz")))
    if a.n_metrics > 0:
        src = src[: a.n_metrics]
    print(f"rescore {len(src)} metrics' Ω (from {a.src_dir}) with {a.target_model} -> {a.out_dir}")

    for n_done, f in enumerate(src):
        z = np.load(f, allow_pickle=True)
        prompts = list(z["prompts"])
        tags = list(z["tags"])
        gi = int(z["r2_idx"])
        name = str(z["name"]) if "name" in z.files else os.path.basename(f)
        out_path = os.path.join(a.out_dir, os.path.basename(f))
        if a.skip_existing and os.path.exists(out_path):
            print(f"  [{n_done+1}/{len(src)}] {name[:40]} CACHED")
            continue
        desc = groups[gi].get("merged_description", "") if gi < len(groups) else ""
        if a.retarget_mi_only:
            sigs = np.asarray(z["sigs"], float)        # reuse src executor's criteria signatures
        else:
            sigs = np.vstack([ap.signature(executor, p, probe_texts, cfg.max_text_chars) for p in prompts])
        extra = {}
        if a.orbit_target and a.orbit_target > 1:      # §12.6.2: m_bar_omega orbit-averaged target (Phi-invariant)
            orb = ap.orbit_metric_verdict(executor, desc, probe_texts, cfg.max_text_chars,
                                          n_forms=a.orbit_target)
            Mi = orb["m_bar"]
            extra = {"M_i_var_phi": float(orb["var_phi_mean"]), "M_i_flip_rate": float(orb["flip_rate"]),
                     "orbit_forms": int(orb["n_forms"])}
        else:
            Mi = ap.metric_verdict(executor, desc, probe_texts, cfg.max_text_chars)
        if sigs.shape[1] != len(Mi):
            raise ValueError(f"probe mismatch for gi={gi}: signatures={sigs.shape[1]}, M_i={len(Mi)}")
        probe_sha256 = np.asarray([hashlib.sha256(str(t).encode()).hexdigest()
                                   for t in probe_texts[:len(Mi)]])
        probe_set_sha256 = hashlib.sha256("\n".join(probe_sha256).encode()).hexdigest()
        np.savez(out_path, sigs=sigs, tags=np.array(tags, dtype=object),
                 prompts=np.array(prompts, dtype=object), M_i=Mi, name=name, r2_idx=gi,
                 tau0=0.05, tau=0.05, source_ckpt=os.path.basename(f),
                 target_model=a.target_model, rescoring=True,
                 probe_sha256=probe_sha256, probe_set_sha256=probe_set_sha256, **extra)
        cr = ap.conditional_crc_report(sigs, tags, cmi_thresh=a.cmi_thresh)
        print(f"  [{n_done+1}/{len(src)}] {name[:36]:36s} B_E={cr['B_E_upper']:.1f} "
              f"D_obs={cr['D_obs_lower']:.0f} cov={cr['coverage']:.2f}")


if __name__ == "__main__":
    main()
