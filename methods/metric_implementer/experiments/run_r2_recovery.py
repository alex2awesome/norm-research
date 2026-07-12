#!/usr/bin/env python3
"""run_r2_recovery — anchor-free Reconstruction-MCQ for R2 metrics.

Given only executor X's target annotations and a frozen option codebook, can a strong LLM identify the
generating metric? MCQ target-option probability/accuracy is primary; across randomized targets the driver
reports I(J;Jhat). Re-executing the selected canonical body on untouched items supplies secondary behavioral
MI. No anchor or silver labels enter either reconstruction readout.

Two modes (the measurement; M_ω quality is the gate — a near-constant M_ω caps both at ~0):
  --mode mcq  : [DEFAULT — canonical readout] GLM PICKS the metric from k candidate descriptions
                after exact design-split contrastive example selection. Position counterbalancing and
                no-demo/shuffled-label controls identify option priors. X may then re-execute the pick
                for the separate behavioral replay. The default
                distractor is `contrastive`: behaviorally related options are screened and the shown
                examples are selected to distinguish every option on a calibration-only split.
                --distractor contrastive|hard|random|matched_base_rate|graded ; --n-options ; --n-pool
  --mode free : [DIAGNOSTIC only] GLM ARTICULATES M̂ from (x, M_ω(x)) examples; X re-executes M̂ →
                I(M_ω; M′). Keep for the READABLE guessed rubric (what did the reconstructor find?),
                not as the reported recovery number — it confounds "metric inarticulable" with "this
                reconstructor wrote a bad rubric this time" and collapses to a generic-quality prior on
                sparse labels. --induce free|gepa|free_dd|reasoning ; --target merged_desc|conjunction

M_ω scoring is BACKEND-AWARE: vLLM executor → continuous logprob P(YES) (`_pyes`); API executor (GLM
via zai_anthropic) → sampled YES/NO (`_sampled_binary`) — so either mode runs laptop-local on GLM-API.

  laptop (GLM-API executor + reconstructor, no GPU):
    python -m methods.metric_implementer.experiments.run_r2_recovery --mode mcq \
        --task peer-review --bucket specific --groups 36 --n-items 40 --n-pool 8 --R 5 \
        --target-model glm-4.7 --reconstructor-backend zai_anthropic --reconstructor-model glm-5.2
  sk3 (vLLM executor; mcq is the default, contrastive teaching design):
    HOME=/lfs python -m methods.metric_implementer.experiments.run_r2_recovery \
        --task peer-review --bucket specific --n-metrics 12 \
        --target-model meta-llama/Llama-3.1-8B-Instruct \
        --reconstructor-backend zai_anthropic --reconstructor-model glm-4.7 --n-items 60 --R 5
    # add `--mode free` only to inspect the READABLE guessed rubric — not for the reported R.
"""
from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np

from .. import config as cfgmod
from ..backends import LLMBackend
from ..recon_channel import mcq_identity_channel, run_metric, _pyes, _sampled_binary
from ..vllm_backend import make_judge_backend
from .mine_clusters import r2_groups, r2_children
from .m_omega_gepa import gepa_discriminative_m_omega
from .orthogonalize import submodular_tail_bound
from .run_real_test import _load_texts

# vLLM model id prefixes (or a snapshot path) ⇒ use logprob P(YES); anything else ⇒ API + sampled YES/NO
_VLLM_PREFIXES = ("meta-llama/", "google/", "Qwen/", "microsoft/", "nvidia/", "mistralai/")


def _san(o):
    """JSON-sanitize numpy scalars/arrays recursively."""
    if isinstance(o, dict):
        return {k: _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_san(x) for x in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return [_san(x) for x in o.tolist()]
    return o


def _is_api_executor(target_model: str) -> bool:
    """API executor (GLM via zai_anthropic) vs vLLM (resident). Decides sampled-vs-logprob M_ω scoring."""
    if target_model.startswith("/") or "/snapshots/" in target_model:
        return False
    return not target_model.startswith(_VLLM_PREFIXES)


def _metric_body(grp, children, target):
    """The rubric whose executor-verdict = M_ω (free mode). `merged_desc` (default) = holistic cluster
    construct; `conjunction` = AND of children (the within-class target — near-constant, for comparison)."""
    if target == "conjunction":
        from .small_omega_brute_force import _compile
        return _compile(children, tuple(range(len(children))), compiler="conjunction")
    return grp["merged_description"] or grp["merged_name"]


def _score_m_omega(executor, body, texts, max_chars, temperature, use_sampled) -> np.ndarray:
    """Backend-aware M_ω scoring: sampled YES/NO (API) or continuous logprob P(YES) (vLLM)."""
    if use_sampled:
        return _sampled_binary(executor, body, texts, max_chars, temperature=temperature, seed=0)
    return _pyes(executor, body, texts, max_chars)


def _build_pool_metrics(groups, n_pool=None):
    """Ω candidate pool for MCQ = the R2 metrics as MetricArtifacts (body = merged_description)."""
    pool = []
    for g in (groups[:n_pool] if n_pool else groups):
        desc = g.get("merged_description") or ""
        name = g.get("merged_name") or f"r2_{g['group_idx']}"
        pool.append(SimpleNamespace(metric_id=f"r2_{g['group_idx']}", kind="metric", name=name,
                                    description=desc, body=desc or name, meta={}))
    return pool


def _score_pool_pyes(executor, metrics, texts, max_chars, temperature, use_sampled):
    """Each pool metric's verdict on the items (for MCQ distractor mining)."""
    out = []
    for met in metrics:
        v = _sampled_binary(executor, met.body, texts, max_chars, temperature=temperature, seed=0) \
            if use_sampled else _pyes(executor, met.body, texts, max_chars)
        out.append((met, v))
    return out


def run_one(task, bucket, grp, cfg, texts, item_ids, executor, recon, a, use_sampled,
            pool_metrics=None, pool_pyes=None):
    """Recover ONE R2 metric. `mode=free` articulates M̂; `mode=mcq` picks from the Ω pool. Both report
    recovery = I(M_ω; M′); mcq additionally reports identification accuracy."""
    gidx = grp["group_idx"]
    children = r2_children(task, bucket, gidx)
    body = (_metric_body(grp, children, a.target) if a.mode == "free"
            else (grp["merged_description"] or grp["merged_name"]))
    raw_pyes = _score_m_omega(executor, body, texts, cfg.max_text_chars, a.temperature, use_sampled)
    raw_std = float(np.nanstd(raw_pyes))
    gepa_info = None
    if getattr(a, "gepa_m_omega", False) and recon is not None:
        # Optimize only on the same declared design split used by run_metric. The resulting prompt is
        # frozen before it is scored on held-out items; otherwise GEPA would consume the replay lockbox.
        split = np.random.default_rng(0).permutation(len(texts))
        design_idx = split[:a.n_train]
        design_texts = [texts[i] for i in design_idx]
        g = gepa_discriminative_m_omega(executor, recon, body, design_texts, cfg.item_noun,
                                       rounds=a.gepa_rounds, n_mutations=a.gepa_mutations,
                                       max_chars=cfg.max_text_chars)
        body = g["optimized_prompt"]            # the metric is now the optimized prompt
        pyes = _score_m_omega(
            executor, body, texts, cfg.max_text_chars, a.temperature, use_sampled)
        gepa_info = {"raw_std": raw_std, "gepa_std": float(g["std"]),
                     "gepa_base_rate": float(g["base_rate"]),
                     "optimization_scope": "design split only; prompt frozen before held-out scoring",
                     "design_item_ids": [str(item_ids[i]) for i in design_idx],
                     "optimized_prompt": str(g["optimized_prompt"])[:200],
                     # OPT trajectory: GEPA's search toward sup_p R(p) — (round, std, base_rate, discrimination)
                     "opt_trajectory": [(t[0], float(t[2]), float(t[3]), float(t[4]))
                                        for t in (g.get("trajectory") or [])]}
    else:
        pyes = raw_pyes
    m_std, m_mean = float(np.nanstd(pyes)), float(np.nanmean(pyes))
    # description = body so the MCQ option GLM SEES matches the behavior that's actually scored
    # (with --gepa-m-omega, body is the optimized prompt whose verdict = M_ω; showing the raw
    # merged_description instead was the mismatch that gave correct-id-but-zero-recovery).
    metric = SimpleNamespace(body=body, name=grp["merged_name"], metric_id=f"r2_{gidx}",
                             description=body, meta={})
    nz = lambda x: float("nan") if (x is None or (isinstance(x, float) and not np.isfinite(x))) else x

    rec = {"task": task, "bucket": bucket, "r2_group_idx": gidx,
           "metric_id": f"r2_{gidx}", "mode": a.mode,
           "merged_name": grp["merged_name"], "merged_description": grp["merged_description"],
           "n_children": len(children),
           # M_ω discrimination gate — if std < floor, M_ω is near-constant and recovery is capped at ~0
           # (NOT "tacit"); flag so low recovery is interpreted correctly.
           "m_omega_pyes_mean": m_mean, "m_omega_pyes_std": m_std,
           "discriminating": bool(m_std >= a.std_floor),
           "m_omega_gepa": gepa_info,          # raw_std vs gepa_std + optimized prompt (if --gepa-m-omega)
           # === C(R(Ω)) = articulability = I(M_ω; M′) (headline, both modes) ===
           "R": a.R, "n_train": a.n_train}

    if a.mode == "free":
        rows = run_metric(executor, cfg.item_noun, metric, [], texts, pyes,
                          R=a.R, n_train=a.n_train, mode="free", max_chars=cfg.max_text_chars,
                          temperature=a.temperature, l_cap=a.l_cap, induce=a.induce,
                          n_examples=a.n_examples, recon_backend=recon, item_ids=item_ids)
        r = rows[0] if rows else {}
        induced = r.get("induced") or []
        rec.update({"target": a.target,
                    "recovery": nz(r.get("iv_transmission")),
                    "recovery_ci": [nz(c) for c in (r.get("iv_transmission_ci") or [None, None])],
                    "transmission_norm": nz(r.get("transmission_norm")),
                    "iv_recon": nz(r.get("iv_recon")),
                    "fidelity_pearson": nz(r.get("fidelity_to_m_pearson")),
                    "iv_consistency_held": nz(r.get("iv_consistency_held")),
                    "l_cap": a.l_cap, "induce": a.induce, "n_held": r.get("n_held"),
                    "induced_rules": [str(x.get("rule", "")) for x in induced[:3]],
                    "n_induced": len(induced)})
    else:  # mcq
        rows = run_metric(executor, cfg.item_noun, metric, pool_metrics, texts, pyes,
                          R=a.R, n_train=a.n_train, mode="mcq", max_chars=cfg.max_text_chars,
                          temperature=a.temperature, l_cap=a.l_cap, distractor=a.distractor,
                          pool_pyes=pool_pyes, n_options=a.n_options, recon_backend=recon,
                          mcq_n_examples=a.mcq_examples,
                          mcq_min_design_disagreements=a.mcq_min_disagreements,
                          mcq_min_demo_disagreements=a.mcq_min_disagreements,
                          item_ids=item_ids, mcq_choice_readout=a.mcq_choice_readout)
        r = rows[0] if rows else {}
        induced = r.get("induced") or []
        rec.update({"identification_acc": nz(r.get("identification_acc")),
                    "identification_score": nz(r.get("identification_score")),
                    "identification_cc": nz(r.get("identification_cc")),
                    "identification": r.get("identification"),
                    "primary_reconstruction_score": nz(r.get("primary_reconstruction_score")),
                    "primary_reconstruction_readout": r.get("primary_reconstruction_readout"),
                    "annotation_attributable_lift": nz(r.get("annotation_attributable_lift")),
                    "n_options": r.get("n_options"), "option_set_S": nz(r.get("option_set_S")),
                    "distractor_mode": a.distractor,
                    "mcq_design": r.get("mcq_design"),
                    "mcq_raw": r.get("mcq_raw"),
                    "mcq_measurement_grade": r.get("mcq_measurement_grade"),
                    "mcq_validity_warnings": r.get("mcq_validity_warnings"),
                    # Compatibility aliases retained for historical analyses. They are the optional
                    # behavioral replay, not the primary MCQ reconstruction score.
                    "recovery": nz(r.get("iv_transmission")),
                    "recovery_ci": [nz(c) for c in (r.get("iv_transmission_ci") or [None, None])],
                    "behavioral_replay_mi_bits": nz(r.get("behavioral_replay_mi_bits")),
                    "behavioral_replay_mi_ci": [
                        nz(c) for c in (r.get("behavioral_replay_mi_ci") or [None, None])],
                    "transmission_norm": nz(r.get("transmission_norm")),
                    "iv_recon": nz(r.get("iv_recon")),
                    "fidelity_pearson": nz(r.get("fidelity_to_m_pearson")),
                    "iv_consistency_held": nz(r.get("iv_consistency_held")),
                    "n_held": r.get("n_held"),
                    "induced_picks": [ind.get("choice", -1) for ind in induced],
                    "n_induced": len(induced)})

    # === fixed-target DPI certificate ===
    # The bundle is computed inside recon_channel._ivs from the SAME held-out target/candidate vectors,
    # separately in Shannon and TVD. The previous implementation compared Shannon R to TVD T and also
    # computed T on all items; both made R_le_T/headroom invalid.
    # (Missing-impact — "could an unseen criterion change the verdict" — is the within-class Ω-completeness
    #  concept; for recovery the analog is "is the true metric in the MCQ pool", tracked via id_acc.)
    # Legacy missing-impact diagnostic: could an unseen criterion improve the mined pool? It is not part
    # of the all-prompt DPI certificate and is not treated as a high-confidence upper bound.
    missing_impact = None
    if pool_pyes:
        try:
            cols = [np.asarray(p, float) for _, p in pool_pyes if np.asarray(p).size == len(pyes)]
            if cols:
                X_pool = np.column_stack(cols)
                M_int = (np.asarray(pyes, float) > 0.5).astype(int)
                mi = submodular_tail_bound(M_int, X_pool)
                missing_impact = {"certified_bound": float(mi.get("certified_bound", float("nan"))),
                                  "tail_bound": float(mi.get("tail_bound", float("nan")))}
        except Exception:
            missing_impact = None
    channel_cert = r.get("fixed_target_certificate") or {"valid": False,
                                                          "error": "missing_fixed_target_payload"}
    channel_cert["missing_impact_diagnostic"] = missing_impact
    channel_cert["search_trajectory_diagnostic"] = (gepa_info or {}).get("opt_trajectory")
    channel_cert["global_tightness_note"] = (
        "DPI bounds every candidate prompt for the fixed target. Equality is not established unless "
        "a sufficient target-posterior readout is realizable by the declared prompt interface."
    )
    rec["certificate"] = channel_cert
    rec["recovery_tvd"] = r.get("tvd_recovery")
    rec["target_ceiling_tvd"] = r.get("tvd_target_ceiling")
    return rec


def _build_executor(a, cfg0):
    """vLLM (resident, default) or API executor. Decided by --executor-backend (explicit), NOT model-name
    prefix — OpenRouter slugs share HF prefixes (meta-llama/...), so a prefix check can't tell API from vLLM."""
    if getattr(a, "executor_backend", "vllm") == "vllm":
        return make_judge_backend(a.target_model, cfg0, temperature=None), False
    rcfg = cfgmod.ImplementerConfig(); rcfg.backend = a.executor_backend
    return LLMBackend(a.target_model, "executor", rcfg), True


def main(argv=None):
    ap = argparse.ArgumentParser(prog="run_r2_recovery", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="mcq", choices=["mcq", "free"],
                    help="mcq = canonical recovery readout (bounded, prior-robust, 1/k chance baseline); "
                         "free = readable-guess DIAGNOSTIC only (can go negative / collapse to a quality prior)")
    ap.add_argument("--task", default="peer-review")
    ap.add_argument("--bucket", default="specific", choices=["general", "specific", "hyper_specific"])
    ap.add_argument("--n-metrics", type=int, default=12, help="sample size (top-N by #leaves)")
    ap.add_argument("--groups", default=None, help="comma list of group_idx (overrides sampling)")
    # executor / reconstructor
    ap.add_argument("--target-model", default="meta-llama/Llama-3.1-8B-Instruct",
                    help="executor X — vLLM model/snapshot OR an API slug (OpenRouter/GLM)")
    ap.add_argument("--executor-backend", default="vllm",
                    choices=["vllm", "openrouter", "zai_anthropic", "zai"],
                    help="executor X backend: vllm (resident model/snapshot) | openrouter/zai_anthropic/zai "
                         "(API). For OpenRouter slugs (meta-llama/llama-3.1-8b-instruct) set this to the API "
                         "backend — the model-name prefix alone can't distinguish API from vLLM.")
    ap.add_argument("--reconstructor-backend", default="zai_anthropic",
                    choices=["zai_anthropic", "zai", "openrouter"])
    ap.add_argument("--reconstructor-model", default="glm-4.7",
                    help="strong model that articulates M̂ / picks the option")
    # free-mode knobs
    ap.add_argument("--target", default="merged_desc", choices=["merged_desc", "conjunction"],
                    help="[free] what M_ω scores: merged_desc (default) or conjunction (AND of children)")
    ap.add_argument("--induce", default="free", choices=["free", "gepa", "free_dd", "reasoning"],
                    help="[free] free=articulate from labels; free_dd=data-driven (Codex#4); "
                         "gepa=propose/select/refine; reasoning=multi-turn critique")
    ap.add_argument("--n-examples", type=int, default=4,
                    help="[free] hi/lo examples shown (4 default; 30 for balanced/reasoning)")
    ap.add_argument("--gepa-m-omega", action="store_true",
                    help="compute M_ω via GEPA (optimize the metric prompt for discrimination/balance "
                        "before scoring; this is a discrimination search, not a recovery optimum). "
                        "Lifts moderate-variance metrics "
                         "(CW Show-Tell 0.40→0.49 std); can't rescue std≈0 (no gradient). [free; MCQ experimental]")
    ap.add_argument("--gepa-rounds", type=int, default=3, help="[--gepa-m-omega] GEPA rounds")
    ap.add_argument("--gepa-mutations", type=int, default=4, help="[--gepa-m-omega] mutations/round")
    # mcq-mode knobs
    ap.add_argument("--distractor", default="contrastive",
                    choices=["contrastive", "random", "hard", "matched_base_rate", "graded"],
                    help="[mcq] contrastive (DEFAULT) = related, non-clone options with an exactly "
                         "optimized separating teaching set on the design split; other modes are "
                         "diagnostic option-selection rules and never inspect held-out behavior")
    ap.add_argument("--n-options", type=int, default=4, help="[mcq] options (1 target + n-1 distractors)")
    ap.add_argument("--n-pool", type=int, default=30, help="[mcq] candidate pool size (R2 metrics)")
    ap.add_argument("--mcq-examples", type=int, default=8,
                    help="[mcq] number of calibration examples in the contrastive teaching set")
    ap.add_argument("--mcq-min-disagreements", type=int, default=2,
                    help="[mcq] minimum target/distractor disagreements available and demonstrated")
    ap.add_argument("--mcq-choice-readout", default="auto",
                    choices=["auto", "logits", "sampled"],
                    help="[mcq] auto uses normalized choice logits when available; APIs fall back "
                         "to counterbalanced stochastic choices")
    # shared
    ap.add_argument("--n-items", type=int, default=60)
    ap.add_argument("--n-train", type=int, default=30, help="items GLM sees as (x,label) examples")
    ap.add_argument("--R", type=int, default=20,
                    help="independent reconstructions per metric; MCQ should use a multiple of n-options")
    ap.add_argument("--l-cap", type=int, default=450, help="articulation budget L (tokens) for M̂")
    ap.add_argument("--temperature", type=float, default=0.7, help="X re-execution sampling temp")
    ap.add_argument("--std-floor", type=float, default=0.12,
                    help="min M_ω P(YES) std to count as discriminating (below ⇒ recovery capped)")
    ap.add_argument("--out-dir", default="/tmp/r2_recovery")
    ap.add_argument("--dry-run", action="store_true", help="FakeVLLM X + no GLM (plumbing check only)")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)

    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    cfg0 = cfgmod.ImplementerConfig()
    if a.dry_run:
        cfg0.vllm_fake = True
    executor, use_sampled = (None, False) if a.dry_run else _build_executor(a, cfg0)
    recon = None
    if a.dry_run:
        print("DRY-RUN: FakeVLLM X + no GLM reconstructor (single fake backend) — plumbing check only")
        executor = make_judge_backend(a.target_model, cfg0, temperature=None)
    else:
        rcfg = cfgmod.ImplementerConfig(); rcfg.backend = a.reconstructor_backend
        recon = LLMBackend(a.reconstructor_model, "reconstructor", rcfg)
        xkind = "API" if use_sampled else "vLLM"
        print(f"executor X = {a.target_model} ({xkind}) | reconstructor = {a.reconstructor_model} "
              f"via {a.reconstructor_backend} | mode={a.mode}")

    texts, ids = _load_texts(a.task, a.n_items, cfg)
    if len(texts) < 20:
        print(f"only {len(texts)} items loadable (corpus may live on sk3); aborting")
        return

    groups = r2_groups(a.task, a.bucket)
    if not groups:
        print(f"no R2 groups for {a.task}/{a.bucket}")
        return
    if a.groups:
        want = {x.strip() for x in a.groups.split(",")}
        sel = [g for g in groups if str(g["group_idx"]) in want]
    else:
        sel = sorted(groups, key=lambda g: g["total_leaf_rubrics"], reverse=True)[:a.n_metrics]

    pool_metrics = pool_pyes = None
    if a.mode == "mcq":
        pool_metrics = _build_pool_metrics(groups, a.n_pool)
        print(f"Scoring pool_pyes for {len(pool_metrics)} candidates via "
              f"{'SAMPLED YES/NO' if use_sampled else 'logprob P(YES)'}…")
        pool_pyes = _score_pool_pyes(executor, pool_metrics, texts, cfg.max_text_chars,
                                     a.temperature, use_sampled)
    tag = (f"induce={a.induce}" if a.mode == "free" else f"distractor={a.distractor},n_opt={a.n_options}")
    print(f"{a.task}/{a.bucket}: {len(groups)} R2 groups → recovering {len(sel)} ({tag}, R={a.R})")

    out_jsonl = os.path.join(a.out_dir, f"{a.task}_{a.bucket}_recovery.jsonl")
    n_ok = 0
    with open(out_jsonl, "w") as fout:
        for grp in sel:
            try:
                rec = run_one(a.task, a.bucket, grp, cfg, texts, ids, executor, recon, a,
                              use_sampled, pool_metrics, pool_pyes)
                n_ok += 1
            except Exception as e:
                import traceback
                traceback.print_exc()
                rec = {"task": a.task, "bucket": a.bucket, "r2_group_idx": grp["group_idx"],
                       "merged_name": grp["merged_name"], "error": repr(e)}
            fout.write(json.dumps(_san(rec)) + "\n")
            fout.flush()
            if "error" in rec:
                print(f"  -> g{grp['group_idx']}: ERR {rec['error'][:80]}")
            elif a.mode == "free":
                print(f"  -> g{grp['group_idx']}: R_shannon={rec.get('recovery')} "
                      f"R_tvd={rec.get('recovery_tvd')} "
                      f"M_ωstd={rec.get('m_omega_pyes_std'):.3f} disc={rec.get('discriminating')} "
                      f"| {grp['merged_name'][:40]}")
            else:
                print(f"  -> g{grp['group_idx']}: id_acc={rec.get('identification_acc'):.3f} "
                      f"R_shannon={rec.get('recovery'):.3f} R_tvd={rec.get('recovery_tvd')} "
                      f"M_ωstd={rec.get('m_omega_pyes_std'):.3f} "
                      f"| {grp['merged_name'][:40]}")

    # ---- summary ----
    rows = [json.loads(l) for l in open(out_jsonl) if "recovery" in l]
    if rows:
        recs = [r["recovery"] for r in rows if isinstance(r.get("recovery"), float)]
        recs_tvd = [r["recovery_tvd"] for r in rows if isinstance(r.get("recovery_tvd"), float)]
        stds = [r["m_omega_pyes_std"] for r in rows if isinstance(r.get("m_omega_pyes_std"), float)]
        n_disc = sum(1 for r in rows if r.get("discriminating"))
        print(f"\n=== SUMMARY {a.task}/{a.bucket} (mode={a.mode}, {tag}) ===")
        print(f"  {len(rows)} metrics ({n_ok}/{len(sel)} ok); {n_disc}/{len(rows)} M_ω discriminating")
        if a.mode == "mcq":
            ida = [r["identification_acc"] for r in rows if isinstance(r.get("identification_acc"), float)]
            if ida:
                print(f"  identification_acc: mean={np.mean(ida):.3f} (chance={1.0/a.n_options:.2f})")
            identity_channels = {
                condition: mcq_identity_channel(rows, condition=condition)
                for condition in ("annotations", "no_demonstrations", "shuffled_labels")
            }
            identity_path = os.path.join(
                a.out_dir, f"{a.task}_{a.bucket}_mcq_identity_channel.json")
            with open(identity_path, "w") as identity_file:
                json.dump(_san({
                    "schema": "reconstruction-mcq-identity-channel-v1",
                    "task": a.task,
                    "bucket": a.bucket,
                    "target_prior": "equal weight over successfully measured target metrics",
                    "channels": identity_channels,
                }), identity_file, indent=2)
            main_channel = identity_channels["annotations"]
            if main_channel.get("valid"):
                print(f"  identity I(J;Jhat): {main_channel['mutual_information_bits']:.3f} bits "
                      f"(normalized={main_channel['normalized_mutual_information']:.3f})")
            print(f"  identity channel artifact: {identity_path}")
        if recs:
            print(f"  Shannon recovery: mean={np.mean(recs):.3f} median={np.median(recs):.3f} "
                  f"max={np.max(recs):.3f}")
        if recs_tvd:
            print(f"  TVD recovery:     mean={np.mean(recs_tvd):.3f} median={np.median(recs_tvd):.3f} "
                  f"max={np.max(recs_tvd):.3f}")
        if stds:
            print(f"  M_ω P(YES) std:     mean={np.mean(stds):.3f} (floor={a.std_floor})")
        print(f"  → {out_jsonl}")


if __name__ == "__main__":
    main()
