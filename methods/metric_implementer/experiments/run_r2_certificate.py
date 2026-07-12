#!/usr/bin/env python3
"""run_r2_certificate — the R2-METRIC-LEVEL certificate driver.

Each R2 cluster (a family of related criteria) is treated as ONE metric. For each R2 metric we
(1) GEPA-optimize a prompt seeded by its `merged_description`, (2) build a scoped Ω = the family's
children + GEPA-mined + GLM free-generation, (3) run the within-class reconstruction certificate
(M = C(Ω), scoped to the family — no M-injection), and (4) measure child-coherence (separate
diagnostic). Plan: ~/.claude/plans/vivid-giggling-sutton.md; notes: R2_CERTIFICATE_NOTES.md.

Why R2: more distinct than R1, fewer of them → lower bar to test. v1 runs a small sample (GLM quota).

  laptop dry-run (no GPU/API):
    python -m methods.metric_implementer.experiments.run_r2_certificate --dry-run \
        --task peer-review --bucket specific --groups 3,36 --n-items 30
  sk3 live v1 (1 GPU):
    HOME=/lfs python -m methods.metric_implementer.experiments.run_r2_certificate \
        --task peer-review --bucket specific --n-metrics 12 \
        --reconstructor-backend zai_anthropic --reconstructor-model glm-4.7 \
        --target-model meta-llama/Llama-3.1-8B-Instruct --n-items 60 --rounds 3
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

from .. import config as cfgmod
from .mine_clusters import r2_groups, r2_children
from .orthogonalize import family_coherence
from .run_real_test import phase_a_gepa, phase_b_certificate, _load_texts


# ---- match the family's children to the scored crits (8-token-prefix, same as mine_clusters._dedup) ----
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def _child_cols(crits, children):
    prefs = {" ".join(_norm(c).split()[:8]) for c in children}
    return [i for i, c in enumerate(crits) if " ".join(_norm(c).split()[:8]) in prefs]


def _san(o):
    """JSON-sanitize numpy scalars/arrays recursively (signals matrix is dropped before this)."""
    if isinstance(o, dict):
        return {k: _san(v) for k, v in o.items() if k != "signals"}
    if isinstance(o, (list, tuple)):
        return [_san(x) for x in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def run_one(task, bucket, grp, cfg, texts, ids, a, sb, large_k):
    """Certificate ONE R2 metric: GEPA(seed=merged_description) → scoped-Ω certificate → coherence."""
    gidx = grp["group_idx"]
    children = r2_children(task, bucket, gidx)
    seed = f"{grp['merged_name']}: {grp['merged_description']}"
    reg_dir = os.path.join(a.out_dir, f"{task}_{bucket}_g{gidx}.registry")
    omega_out = os.path.join(a.out_dir, f"{task}_{bucket}_g{gidx}.npz")
    print(f"\n--- R2 metric g{gidx}: {grp['merged_name']!r}  ({len(children)} children) ---")
    registry, cfg2 = phase_a_gepa(task, texts, ids, a.target_model, a.reconstructor_model, sb,
                                  rounds=a.rounds, fake=a.dry_run, seed_body=seed, registry_dir=reg_dir)
    result = phase_b_certificate(task, registry, cfg2, a.target_model, a.reconstructor_model, sb,
                                 n_items=a.n_items, budget=a.budget, large_k=large_k, fake=a.dry_run,
                                 compiler=a.compiler, cmi_thresh=a.cmi_thresh, deep_k=a.deep_k,
                                 pool_max=a.pool_max, scoped_criteria=children,
                                 free_gen_k=a.free_gen_k, omega_out=omega_out)
    # coherence on the family's children columns of the scored signal matrix
    coh = None
    crits, signals = result.get("crits"), result.get("signals")
    if crits is not None and signals is not None:
        cols = _child_cols(crits, children)
        if len(cols) >= 2:
            coh = family_coherence(np.asarray(signals)[:, cols])
        else:
            coh = {"mean_pairwise_corr": None, "participation_ratio": None,
                   "n_criteria": len(cols), "n_usable": len(cols), "coherent": None,
                   "note": "<2 children scored"}
    rec = {"task": task, "bucket": bucket, "r2_group_idx": gidx, "merged_name": grp["merged_name"],
           "merged_description": grp["merged_description"], "n_children": len(children),
           "n_children_scored": (coh or {}).get("n_criteria"), "coherence": coh}
    for k in ("mode", "K", "subsets_scored", "M_base_rate", "decomposition",
              "missing_impact", "n_candidates", "n_collapsed", "error"):
        if k in result:
            rec[k] = result[k]
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(prog="run_r2_certificate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="peer-review")
    ap.add_argument("--bucket", default="specific", choices=["general", "specific", "hyper_specific"])
    ap.add_argument("--n-metrics", type=int, default=12, help="sample size (top-N by #leaves)")
    ap.add_argument("--groups", default=None, help="comma list of specific group_idx (overrides sampling)")
    ap.add_argument("--target-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--reconstructor-backend", default="zai_anthropic",
                    choices=["zai_anthropic", "zai", "openrouter", "mock"])
    ap.add_argument("--reconstructor-model", default="glm-4.7",
                    help="strong model (reviser/reconstructor/free-gen); glm-4.7 has quota now, 5.2 out till ~06-30")
    ap.add_argument("--n-items", type=int, default=60)
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3, help="GEPA rounds (Phase A)")
    ap.add_argument("--large-k", type=int, default=15)
    ap.add_argument("--compiler", default="conjunction", choices=["conjunction", "weighted_sum", "prose_join"])
    ap.add_argument("--cmi-thresh", type=float, default=0.02)
    ap.add_argument("--deep-k", type=int, default=25)
    ap.add_argument("--pool-max", type=int, default=60)
    ap.add_argument("--free-gen-k", type=int, default=8, help="novel-criteria free-gen per metric (0=off)")
    ap.add_argument("--out-dir", default="/tmp/r2_cert")
    ap.add_argument("--dry-run", action="store_true", help="FakeVLLM X + mock GLM (no GPU, no API)")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    sb = "mock" if a.dry_run else a.reconstructor_backend

    groups = r2_groups(a.task, a.bucket)
    if not groups:
        print(f"no R2 groups for {a.task}/{a.bucket} (expected outputs/hierarchy/{a.task}_{a.bucket}_r2_expanded.json)")
        return
    if a.groups:
        want = {x.strip() for x in a.groups.split(",")}
        sel = [g for g in groups if str(g["group_idx"]) in want]
    else:
        sel = sorted(groups, key=lambda g: g["total_leaf_rubrics"], reverse=True)[:a.n_metrics]
    print(f"{a.task}/{a.bucket}: {len(groups)} R2 groups → running {len(sel)} "
          f"({'specified' if a.groups else 'top-'+str(a.n_metrics)+' by leaves'})")

    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    texts, ids = _load_texts(a.task, a.n_items, cfg)
    if len(texts) < 20:
        print(f"only {len(texts)} items loadable locally (manifest corpus may live on sk3); aborting")
        return

    out_jsonl = os.path.join(a.out_dir, f"{a.task}_{a.bucket}_results.jsonl")
    n_ok = 0
    with open(out_jsonl, "w") as fout:
        for grp in sel:
            try:
                rec = run_one(a.task, a.bucket, grp, cfg, texts, ids, a, sb, a.large_k)
                n_ok += 1
            except Exception as e:
                import traceback
                traceback.print_exc()
                rec = {"task": a.task, "bucket": a.bucket, "r2_group_idx": grp["group_idx"],
                       "merged_name": grp["merged_name"], "error": repr(e)}
            fout.write(json.dumps(_san(rec)) + "\n")
            fout.flush()
            coh = rec.get("coherence")
            coh_tag = (coh.get("coherent") if isinstance(coh, dict) else None)
            print(f"  -> g{grp['group_idx']}: mode={rec.get('mode')} K={rec.get('K')} "
                  f"coh={coh_tag} err={rec.get('error')}")
    print(f"\ndone: {n_ok}/{len(sel)} metrics certified → {out_jsonl}")


if __name__ == "__main__":
    main()
