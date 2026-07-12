"""Trial driver: scorecards, improvement runs, and the scaling grid on competitive code.

Subcommands
-----------
    smoke      tiny live sanity pass (one metric, small samples) — confirms backends + cost
    scorecard  full scorecard for every trial metric x kind; persists to registry
    improve    GEPA loop for chosen metrics/kinds under a single budget cap
    scaling    the budget grid: frontier fidelity per (metric x kind x cap); writes CSV

Examples
--------
    PYTHONPATH=methods python -m metric_implementer.run_trial smoke
    PYTHONPATH=methods python -m metric_implementer.run_trial scorecard --n-pool 80
    PYTHONPATH=methods python -m metric_implementer.run_trial improve \
        --metric identifier_quality --kind prompt --rounds 2
    PYTHONPATH=methods python -m metric_implementer.run_trial scaling \
        --token-caps 120,300,800 --rounds-caps 1,2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .artifact import MetricArtifact
from .backends import make_roles
from .config import BudgetCaps, ImplementerConfig, apply_task_preset
from .judges import CodeJudge, PromptJudge
from .measures import compute_scorecard, convergence
from .optimizer import improve
from .registry import Registry
from .trial.trial_metrics import trial_metrics


def load_pool(cfg, n: int) -> pd.DataFrame:
    df = pd.read_json(cfg.pool_path, lines=True)
    rng = np.random.default_rng(cfg.random_seed)
    if n and len(df) > n:
        df = df.iloc[rng.choice(len(df), size=n, replace=False)].reset_index(drop=True)
    return df


def _setup(args):
    cfg = ImplementerConfig()
    apply_task_preset(cfg, getattr(args, "task", None) or "code-review")
    if getattr(args, "judge_model", None):
        cfg.judge_model = args.judge_model
    if getattr(args, "oracle_items", None) is not None:
        cfg.n_oracle_items = args.oracle_items
    registry = Registry(cfg.registry_dir())
    df = load_pool(cfg, getattr(args, "n_pool", 60))
    texts = df[cfg.text_column].astype(str).str.slice(0, cfg.max_text_chars).tolist()
    ids = df[cfg.id_column].astype(str).tolist()
    roles = make_roles(cfg)
    return cfg, registry, texts, ids, roles


def cmd_smoke(args) -> int:
    args.n_pool = 16
    cfg, registry, texts, ids, roles = _setup(args)
    cfg.n_reliability_items = 8
    cfg.reliability_passes = 2
    cfg.n_reconstruct_label_items = 12
    cfg.n_reconstruct_shown = 8
    cfg.n_reconstruct_behavioral = 8
    cfg.n_cf_base_texts = 3
    cfg.n_consistency_items = 6
    cfg.n_convergence_items = 16

    prompt_art, code_art = _metrics_for(cfg)[0]    # first metric of the task's ladder
    rng = np.random.default_rng(cfg.random_seed)
    registry.register_metric(prompt_art.metric_id, prompt_art.name, prompt_art.description)
    t0 = time.time()
    card_p = compute_scorecard(prompt_art, texts[: cfg.n_reliability_items * 2],
                               roles, cfg, rng)
    card_c = compute_scorecard(code_art, texts[: cfg.n_reliability_items * 2],
                               roles, cfg, rng)
    conv = convergence(prompt_art, code_art, texts[: cfg.n_convergence_items], roles, cfg)
    dt = time.time() - t0
    print(json.dumps({
        "prompt_fidelity": card_p["fidelity_scalar"],
        "prompt": {k: card_p[k].get("rho") or card_p[k].get("cf_validity")
                   or card_p[k].get("behavioral") or card_p[k].get("mean")
                   for k in ("reliability", "counterfactual", "reconstruction",
                             "consistency")},
        "code_fidelity": card_c["fidelity_scalar"],
        "code_judge_spearman": conv["spearman"],
        "induced_rule_prompt": card_p["reconstruction"].get("induced_rule", ""),
        "wall_s": round(dt, 1),
        "cost_usd": round(roles.total_cost(), 4),
        "role_stats": roles.stats(),
    }, indent=2))
    registry.log("RUN_FINISHED", run_id="smoke", cost_usd=roles.total_cost(),
                 role_stats=roles.stats())
    return 0


def cmd_scorecard(args) -> int:
    cfg, registry, texts, ids, roles = _setup(args)
    rng = np.random.default_rng(cfg.random_seed)
    rows = []
    for prompt_art, code_art in _metrics_for(cfg):
        registry.register_metric(prompt_art.metric_id, prompt_art.name,
                                 prompt_art.description)
        for art in (prompt_art, code_art):
            vid = registry.head(art.metric_id, art.kind)
            if vid is None:
                vid = registry.create_version(
                    art, operator="INIT", parent_version=None, run_id="scorecard",
                    optimizer_round=None, data_budget_ids=ids, caps_in_force=None,
                    models={"judge": cfg.judge_model})
            card = compute_scorecard(art, texts, roles, cfg, rng)
            registry.save_scorecard(art.metric_id, vid, art.kind, card, "full")
            rows.append({"metric": art.metric_id, "kind": art.kind,
                         "fidelity": card["fidelity_scalar"],
                         "rho": card["reliability"].get("rho"),
                         "cf": card["counterfactual"].get("cf_validity"),
                         "recon_b": card["reconstruction"].get("behavioral"),
                         "recon_s": card["reconstruction"].get("semantic"),
                         "consist": card["consistency"].get("mean")})
            print(rows[-1])
        conv = convergence(prompt_art, code_art, texts[: cfg.n_convergence_items],
                           roles, cfg)
        registry.save_disagreements(prompt_art.metric_id, "current",
                                    conv["disagreement_queue"], "code_vs_judge")
        print({"metric": prompt_art.metric_id, "code_judge_spearman": conv["spearman"]})
    print(f"total cost: ${roles.total_cost():.3f}")
    registry.log("RUN_FINISHED", run_id="scorecard_all", cost_usd=roles.total_cost(),
                 role_stats=roles.stats())
    return 0


def _metrics_for(cfg, only=None):
    if cfg.task == "creative-writing":
        from .trial.trial_metrics_cw import trial_metrics_cw
        pairs = trial_metrics_cw()
    elif cfg.task == "law":
        from .trial.trial_metrics_law import trial_metrics_law
        pairs = trial_metrics_law()
    else:
        pairs = trial_metrics()
    if only:
        keep = {m.strip() for m in only.split(",") if m.strip()}
        pairs = [pc for pc in pairs if pc[0].metric_id in keep]
        if not pairs:
            raise SystemExit(f"no metrics matched filter {only!r}")
    return pairs


def _find_seed(metric_id: str, kind: str, cfg=None) -> MetricArtifact:
    pool = _metrics_for(cfg) if cfg is not None else trial_metrics()
    for p, c in pool:
        if p.metric_id == metric_id:
            return p if kind == "prompt" else c
    raise SystemExit(f"unknown metric {metric_id}")


def cmd_improve(args) -> int:
    cfg, registry, texts, ids, roles = _setup(args)
    caps = BudgetCaps(instruction_tokens=args.token_cap, n_fewshots=args.fewshot_cap,
                      optimizer_rounds=args.rounds, code_ast_nodes=args.ast_cap,
                      data_budget=args.data_budget)
    seed = _find_seed(args.metric, args.kind, cfg)
    # convergence disagreements feed the reviser when both kinds exist
    other = _find_seed(args.metric, "code" if args.kind == "prompt" else "prompt", cfg)
    conv = convergence(seed if args.kind == "prompt" else other,
                       other if args.kind == "prompt" else seed,
                       texts[: cfg.n_convergence_items], roles, cfg)
    improve(seed, texts, roles, cfg, registry, caps=caps, rounds=args.rounds,
            data_ids=ids, convergence_queue=conv["disagreement_queue"])
    return 0


def cmd_triad(args) -> int:
    """CrowdTruth x G-theory panel run: one metric, n items, several judge models x
    passes (+ optional strong anchor). Persists the long score table + analysis to the
    registry's triad/ dir and prints the summary."""
    from .backends import LLMBackend
    from . import triad as T

    cfg, registry, texts, ids, _ = _setup(args)
    seed = _find_seed(args.metric, "prompt", cfg)
    registry.register_metric(seed.metric_id, seed.name, seed.description)
    mid = seed.metric_id
    seed_vid = None
    for v in registry.versions(mid, "prompt"):           # canonical = first INIT version
        if v["lineage"]["operator"] == "INIT":
            seed_vid = v["version_id"]
            break
    if seed_vid is None:
        seed_vid = registry.create_version(
            seed, operator="INIT", parent_version=None, run_id="triad",
            optimizer_round=None, data_budget_ids=ids, caps_in_force=None,
            models={"panel": args.judge_models})

    # Per-judge prompt resolution: prompts are optimized FOR a judge, so each panel
    # judge scores with ITS OWN iteration — explicit --versions map, else its per-tier
    # HEAD, else --version (shared), else the canonical seed. The anchor ALWAYS reads
    # the canonical seed rubric: it is the fixed construct reference.
    explicit = dict(kv.split("=", 1) for kv in args.versions.split(",")) \
        if args.versions else {}
    models = [m.strip() for m in args.judge_models.split(",") if m.strip()]
    panel_arts, versions_by_judge = {}, {}
    for m in models:
        short = m.split("/")[-1]
        vid_m = explicit.get(short) or registry.head(mid, "prompt", judge_tier=m) \
            or args.version or seed_vid
        vrec = registry.get_version(mid, vid_m, "prompt")
        panel_arts[short] = (LLMBackend(m, "judge", cfg, cfg.judge_temperature),
                             registry.artifact_from_version(vrec), vid_m,
                             (vrec.get("budget") or {}).get("data_budget_n"))
        versions_by_judge[short] = vid_m
    anchor = (args.anchor, LLMBackend(args.anchor, "judge", cfg, 0.3)) \
        if args.anchor else None

    registry.log("RUN_STARTED", run_id=f"triad_{mid}", versions=versions_by_judge,
                 anchor_version=seed_vid, panel=models, anchor=args.anchor,
                 passes=args.passes, n_items=len(texts))
    t0 = time.time()
    run_ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    tables = [T.build_long_table(art, texts, ids, {short: be}, cfg,
                                 passes=args.passes, version_id=vid_m,
                                 n_data=nd or len(texts), run_ts=run_ts)
              for short, (be, art, vid_m, nd) in panel_arts.items()]
    if anchor is not None:
        tables.append(T.build_long_table(seed, texts, ids, {}, cfg, passes=0,
                                         anchor=anchor, version_id=seed_vid,
                                         n_data=len(texts), run_ts=run_ts))
    table = pd.concat(tables, ignore_index=True)
    uniq = sorted(set(versions_by_judge.values()))
    vid = uniq[0] if len(uniq) == 1 else "multi"
    ct = T.crowdtruth(table)                       # anchor included: it is a measured worker
    vc = T.variance_components(table)              # anchor excluded: 1 pass, no replicates
    item_strata = T.strata(ct["uqs"])
    anch = T.anchor_agreement(table, item_strata) if anchor else {}
    cost = sum(be.stats.cost_usd for be, _, _, _ in panel_arts.values()) + \
        (anchor[1].stats.cost_usd if anchor else 0.0)

    from .judges import judge_prompts
    _sys, _tmpl = judge_prompts(cfg)
    payload = {
        "metric_id": mid, "version_id": vid, "kind": "prompt",
        "versions_by_judge": versions_by_judge,
        "rubrics_by_judge": {s: a.body for s, (_, a, _, _) in panel_arts.items()},
        "anchor_version": seed_vid, "anchor_rubric": seed.body,
        "judge_system_prompt": _sys, "judge_template": _tmpl,
        "facet_note": ("panel judges score with their OWN prompt iteration; the judge "
                       "facet in the variance decomposition is therefore the whole "
                       "instrument (judge + its prompt), and anchor agreement reads as "
                       "construct recovery vs the canonical seed rubric"),
        "panel_models": models, "anchor_model": args.anchor, "passes": args.passes,
        "n_items": len(texts),
        "crowdtruth": ct, "variance_components": vc,
        "d_study": T.d_study(vc), "item_strata": item_strata,
        "anchor_agreement": anch,
        "wall_s": round(time.time() - t0, 1), "cost_usd": round(cost, 4),
        "parse_fail_rate": float(table["score"].isna().mean()),
    }
    path = registry.save_triad(seed.metric_id, vid, payload, table)
    registry.log("RUN_FINISHED", run_id=f"triad_{seed.metric_id}", cost_usd=cost)

    hard = [i for i, s in item_strata.items() if s == "hard"]
    print(json.dumps({
        "versions_by_judge": versions_by_judge,
        "aqs": round(ct["aqs"], 3),
        "wqs_by_judge": {k: round(v, 3) for k, v in ct["wqs_by_judge"].items()},
        "variance_components": {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in vc.items()},
        "anchor_agreement_overall": {j: e.get("overall") for j, e in anch.items()},
        "anchor_agreement_hard": {j: e.get("hard") for j, e in anch.items()},
        "n_hard_items": len(hard),
        "parse_fail_rate": round(payload["parse_fail_rate"], 3),
        "wall_s": payload["wall_s"], "cost_usd": payload["cost_usd"],
        "saved": str(path),
    }, indent=2))
    return 0


def cmd_triad_history(args) -> int:
    """Print the prompt-evolution x data-scaling panel for one metric: every triad run
    over time, joined to version lineage (operator, tokens) and data budget."""
    from . import triad as T
    cfg = ImplementerConfig()
    apply_task_preset(cfg, getattr(args, "task", None) or "code-review")
    registry = Registry(cfg.registry_dir())
    df = T.triad_history(registry, args.metric)
    if df.empty:
        print(f"no triad runs recorded for {args.metric}")
        return 1
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    return 0


def cmd_scaling(args) -> int:
    from .backends import make_roles as _mk
    cfg, registry, texts, ids, _ = _setup(args)
    Path(cfg.runs_dir()).mkdir(parents=True, exist_ok=True)   # per-task, not default-task
    token_caps = [int(x) for x in args.token_caps.split(",")]
    rounds_caps = [int(x) for x in args.rounds_caps.split(",")]
    judge_models = [m.strip() for m in args.judge_models.split(",") if m.strip()]
    out_rows = []
    run_tag = f"scaling_{int(time.time())}"
    total_cost = 0.0
    # The judge-capability axis: each judge tier gets its OWN optimizer runs (the frontier
    # is per-tier — a prompt optimized for 70B is not the 8B frontier). Reviser/oracle stay
    # strong and fixed across tiers.
    for jm in judge_models:
        roles = _mk(cfg, judge_model=jm)
        jtag = jm.split("/")[-1][:24]
        for prompt_art, code_art in _metrics_for(cfg, only=getattr(args, "metrics", None)):
            arts = ([prompt_art] if args.kinds in ("prompt", "both") else []) + \
                   ([code_art] if args.kinds in ("code", "both") else [])
            for art in arts:
                for tc in token_caps:
                    for rc in rounds_caps:
                        caps = BudgetCaps(
                            instruction_tokens=tc if art.kind == "prompt" else None,
                            code_ast_nodes=tc * 2 if art.kind == "code" else None,
                            n_fewshots=max(0, tc // 150), optimizer_rounds=rc)
                        rid = f"{run_tag}_{art.metric_id}_{art.kind}_t{tc}_r{rc}_{jtag}"
                        s = improve(art, texts, roles, cfg, registry, caps=caps,
                                    rounds=rc, run_id=rid, data_ids=ids)
                        out_rows.append({
                            "metric": art.metric_id, "kind": art.kind,
                            "judge_model": jm, "token_cap": tc, "rounds_cap": rc,
                            "frontier_fidelity": s["best_fidelity_acceptance"],
                            "seed_fidelity": s["seed_fidelity_acceptance"],
                            "accepted": s["accepted"], "run_id": rid,
                        })
                        pd.DataFrame(out_rows).to_csv(
                            Path(cfg.runs_dir()) / f"{run_tag}.csv", index=False)
        total_cost += roles.total_cost()
    print(pd.DataFrame(out_rows).to_string(index=False))
    print(f"total cost: ${total_cost:.3f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("smoke", "scorecard", "improve", "scaling", "triad", "triad-history"):
        p = sub.add_parser(name)
        p.add_argument("--n-pool", type=int, default=60)
        p.add_argument("--judge-model", default=None)
        p.add_argument("--oracle-items", type=int, default=None,
                       help="override cfg.n_oracle_items (cost control)")
        p.add_argument("--task", default="code-review",
                       choices=["code-review", "creative-writing", "law"])
        if name == "triad":
            p.add_argument("--metric", required=True)
            p.add_argument("--judge-models", required=True,
                           help="comma list of panel judge models")
            p.add_argument("--passes", type=int, default=3)
            p.add_argument("--anchor", default=None,
                           help="strong reference model, scored once at temp 0.3")
            p.add_argument("--version", default=None,
                           help="shared fallback iteration when a judge has no "
                                "per-tier HEAD and no --versions entry")
            p.add_argument("--versions", default=None,
                           help="per-judge map, e.g. "
                                "'llama-3.1-8b-instruct=v003,qwen-2.5-72b-instruct=v005'"
                                " (keys are model short names)")
        if name == "triad-history":
            p.add_argument("--metric", required=True)
        if name == "improve":
            p.add_argument("--metric", required=True)
            p.add_argument("--kind", choices=["prompt", "code"], default="prompt")
            p.add_argument("--rounds", type=int, default=2)
            p.add_argument("--token-cap", type=int, default=None)
            p.add_argument("--fewshot-cap", type=int, default=None)
            p.add_argument("--ast-cap", type=int, default=None)
            p.add_argument("--data-budget", type=int, default=None)
        if name == "scaling":
            p.add_argument("--token-caps", default="120,300,800")
            p.add_argument("--rounds-caps", default="1,2")
            p.add_argument("--metrics", default=None,
                           help="comma filter of metric_ids (default: all for the task)")
            p.add_argument("--kinds", choices=["prompt", "code", "both"], default="both")
            p.add_argument("--judge-models",
                           default="meta-llama/llama-3.1-8b-instruct",
                           help="comma list: the judge-capability scaling axis, e.g. "
                                "meta-llama/llama-3.2-1b-instruct,"
                                "meta-llama/llama-3.1-8b-instruct,"
                                "meta-llama/llama-3.3-70b-instruct")
    args = ap.parse_args(argv)
    Path(ImplementerConfig().runs_dir()).mkdir(parents=True, exist_ok=True)
    return {"smoke": cmd_smoke, "scorecard": cmd_scorecard,
            "improve": cmd_improve, "scaling": cmd_scaling,
            "triad": cmd_triad, "triad-history": cmd_triad_history}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
