"""Run prompt optimization on the planted `is_scary` metric and emit the optimized prompts.

By default this runs entirely OFFLINE against the deterministic planted judge
(``scary_judge.scary_roles``) — zero GPU, zero network, zero API key — so it doubles as the
end-to-end smoke for the whole optimize -> score -> recover stack across ALL optimizers.

Optimizers exercised: GEPA (reflective operator loop) + EvoPrompt (population GA) + ProTeGi
(textual-gradient beam) + APE/OPRO (instruction-induction resampling) — all over the same
unsupervised ``fidelity_scalar`` objective, all tagged in the registry by ``optimizer``.

    # offline planted judge (default): all 4 optimizers
    python -m methods.metric_implementer.synthetic_examples.test_metric_scary.run_optimizers

    # real judge (sk3 offline-vLLM) — needs a live backend; not run by tests
    python -m ...run_optimizers --real --judge-model meta-llama/llama-3.1-8b-instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from ...config import BudgetCaps, ImplementerConfig, apply_task_preset
from ...measures import compute_scorecard
from ...optimizer import improve
from ...optimizers import ape, evoprompt, protegi
from ...registry import Registry
from . import cues
from . import scary_metric as SM
from .scary_judge import JUDGE_TAG, scary_roles

HERE = Path(__file__).resolve().parent
POOL_PATH = HERE / "data" / "scary_pool.jsonl"


def _make_cfg(out_dir: Path, judge_model: str) -> ImplementerConfig:
    cfg = ImplementerConfig()
    apply_task_preset(cfg, "creative-writing")          # correct story framing
    cfg.task = "test_metric_scary"
    cfg.output_dir = str(out_dir)
    cfg.pool_path = str(POOL_PATH)
    cfg.text_column = "text"
    cfg.id_column = "id"
    cfg.judge_model = judge_model
    cfg.n_reliability_items = 24
    cfg.reliability_passes = 2
    cfg.n_reconstruct_label_items = 40
    cfg.n_reconstruct_shown = 12
    cfg.n_reconstruct_behavioral = 24
    cfg.n_cf_base_texts = 8
    cfg.n_consistency_items = 12
    cfg.n_oracle_items = 24
    cfg.n_mutations = 1
    return cfg


def _load_pool(n: int, seed: int = 0):
    recs = [json.loads(l) for l in POOL_PATH.open()]
    rng = np.random.default_rng(seed)
    if n and len(recs) > n:
        recs = [recs[i] for i in rng.choice(len(recs), size=n, replace=False)]
    return [r["text"] for r in recs], [r["id"] for r in recs]


# ---- optimizer adapters (uniform signature) ---------------------------------------------

def _run_gepa(seed, texts, ids, roles, cfg, reg, caps, rounds):
    return improve(seed, texts, roles, cfg, reg, caps=caps, rounds=rounds,
                   run_id="gepa_is_scary", data_ids=ids, log=print)


def _adapt(fn, tag):
    def g(seed, texts, ids, roles, cfg, reg, caps, rounds):
        return fn(seed, texts, roles, cfg, reg, caps=caps, rounds=rounds,
                  run_id=f"{tag}_is_scary", data_ids=ids, log=print)
    return g


OPTIMIZERS = {"gepa": _run_gepa, "evoprompt": _adapt(evoprompt, "evoprompt"),
              "protegi": _adapt(protegi, "protegi"), "ape": _adapt(ape, "ape")}


def run(n_pool: int = 200, rounds: int = 4, data_budget=None, n_fewshots: int = 2,
        real: bool = False, judge_model: str = JUDGE_TAG,
        out_dir: Path = HERE / "runs_out") -> dict:
    cfg = _make_cfg(Path(out_dir), judge_model if real else JUDGE_TAG)
    if real:
        from ...backends import make_roles
        roles = make_roles(cfg)
        print(f"[real] live judge = {cfg.judge_model} (backend={cfg.backend})")
    else:
        roles = scary_roles()
    texts, ids = _load_pool(n_pool)
    registry = Registry(cfg.registry_dir())
    seed_art = SM.seed_artifact()
    caps = BudgetCaps(instruction_tokens=400, n_fewshots=n_fewshots,
                      optimizer_rounds=rounds, data_budget=data_budget)

    summaries: Dict[str, dict] = {}
    for name, fn in OPTIMIZERS.items():
        print(f"\n### optimizer: {name} ###")
        summaries[name] = fn(seed_art, texts, ids, roles, cfg, registry, caps, rounds)

    # floor / ceiling reference scorecards (for contrast; outside the optimizer lineages)
    rng = np.random.default_rng(cfg.random_seed)
    floor_card = compute_scorecard(seed_art, texts, roles, cfg, rng)
    ref_art = SM.reference_artifact()
    ref_vid = registry.create_version(
        ref_art, operator="REFERENCE", optimizer="reference", parent_version=None,
        run_id="reference", optimizer_round=None, data_budget_ids=ids,
        caps_in_force=None, models={"judge": cfg.judge_model})
    ceil_card = compute_scorecard(ref_art, texts, roles, cfg, rng)
    registry.save_scorecard(ref_art.metric_id, ref_vid, "prompt", ceil_card, "reference")

    per_opt = {}
    for name, s in summaries.items():
        body = registry.get_version(SM.METRIC_ID, s["best_version"], "prompt")["body"]
        per_opt[name] = {
            "accepted": s["accepted"],
            "seed_fidelity": round(float(s["seed_fidelity_acceptance"]), 3),
            "best_fidelity": round(float(s["best_fidelity_acceptance"]), 3),
            "cues_named": sorted(cues.coverage(body)),
            "best_version": s["best_version"], "best_prompt": body}
    report = {
        "n_pool": len(texts), "rounds": rounds, "data_budget": data_budget,
        "n_fewshots": n_fewshots, "judge_model": cfg.judge_model,
        "floor_seed_fidelity": round(float(floor_card["fidelity_scalar"]), 3),
        "ceiling_reference_fidelity": round(float(ceil_card["fidelity_scalar"]), 3),
        "ceiling_cues": sorted(cues.coverage(SM.REFERENCE_RUBRIC)),
        "optimizers": per_opt, "cost_usd": round(roles.total_cost(), 4)}
    out_json = Path(out_dir) / "is_scary_result.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 72)
    print(f"is_scary — {len(OPTIMIZERS)} optimizers  (floor={report['floor_seed_fidelity']:.3f}"
          f"  ceiling={report['ceiling_reference_fidelity']:.3f}, 5 cues)")
    print("=" * 72)
    for name, o in per_opt.items():
        print(f"  {name:10} seed={o['seed_fidelity']:.3f} -> best={o['best_fidelity']:.3f}  "
              f"accepted={str(o['accepted']):5}  cues={len(o['cues_named'])}/5 {o['cues_named']}")
    print(f"\n  cost=${report['cost_usd']:.4f}  -> wrote {out_json}")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-pool", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--data-budget", type=int, default=None, help="N axis cap")
    ap.add_argument("--n-fewshots", type=int, default=2, help="K axis cap")
    ap.add_argument("--real", action="store_true",
                    help="use a live judge backend (sk3 offline-vLLM) instead of the planted judge")
    ap.add_argument("--judge-model", default=JUDGE_TAG)
    args = ap.parse_args(argv)
    if not POOL_PATH.exists():
        from . import build_dataset
        build_dataset.write(build_dataset.build())
    run(n_pool=args.n_pool, rounds=args.rounds, data_budget=args.data_budget,
        n_fewshots=args.n_fewshots, real=args.real, judge_model=args.judge_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
