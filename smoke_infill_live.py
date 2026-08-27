"""Trimmed LIVE smoke for metrics_tree_infilling (the "ctree" gap-detecting tree).

Builds a small synthetic creature-dossier corpus and runs the FULL live infilling
loop — real LLM proposer + judge — over the z.ai/anthropic proxy. This proves the
end-to-end path executes and discovers a feature, far cheaper than the 2,400-item
``_run_live()``. The judge materialization caches to judge.jsonl, so the run is
resumable: if it crashes mid-way (e.g. proxy rate-limit), just re-run.

Run from the norm-research repo root:
    python smoke_infill_live.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

# io_metrics.make_vllm_judge_scorer / feature_gen.make_proposer lazily import a bare
# `verification_library` (it lives under methods/). Put methods/ on sys.path so that
# resolves when this file is run as a standalone script from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent / "methods"))

from methods.metrics_tree_infilling.config import InfillConfig
from methods.metrics_tree_infilling.io_metrics import (
    discover_test_split,
    make_vllm_judge_scorer,
    materialize,
)
from methods.metrics_tree_infilling.feature_gen import make_proposer
from methods.metrics_tree_infilling.loop import run_infill
from methods.metrics_tree_infilling.tests.test_scenario.generate import build_corpus
from methods.metrics_tree_infilling.tests.test_scenario.metrics import companion_code

# Both proposer and judge route through z.ai via the anthropic SDK (ANTHROPIC_BASE_URL
# + ANTHROPIC_AUTH_TOKEN are already in env). glm-5.2 is confirmed served.
MODEL = "glm-5.2"


def _harden_against_rate_limits() -> None:
    """Wrap LLMClient.generate with outer retries so a 429 storm doesn't kill the run.

    The proxy rate-limits aggressively (shared with the Claude Code session). The
    inner generate() retries 3x then raises; we add up to 6 outer attempts with
    long backoffs. Contained to this smoke — does not modify library source.
    """
    from methods.verification_library import client as _vc

    orig = _vc.LLMClient.generate

    async def robust(self, prompt, max_tokens=2048, temperature=0.7, system=""):
        last = None
        for outer in range(6):
            try:
                return await orig(self, prompt, max_tokens, temperature, system)
            except Exception as e:  # noqa: BLE001 - retry on any transient API failure
                last = e
                wait = min(60.0, 8.0 * (2 ** outer))
                print(f"[smoke] generate failed ({type(e).__name__}: {str(e)[:80]}); "
                      f"outer retry {outer + 1}/6 in {wait:.0f}s")
                await asyncio.sleep(wait)
        raise last

    _vc.LLMClient.generate = robust


def main() -> int:
    t0 = time.time()
    _harden_against_rate_limits()

    n = 500
    print(f"[smoke] building trimmed corpus (n={n}) ...")
    df, _ = build_corpus(n=n, seed=7)
    print(f"[smoke] corpus: {len(df)} items, label rate={df['judgement'].mean():.2f}")

    cfg = InfillConfig(
        # tree engine (mirrors test_scenario._config, scaled to n=500)
        n_permutations=199,
        min_node_size=30,
        max_depth=4,
        random_seed=0,
        include_text_length_in_z=False,
        gap_deviance_per_item=1.20,
        gap_auc_threshold=0.55,
        contrastive_pairs_k=6,
        # trimmed loop
        max_outer_rounds=2,
        max_features_per_round=4,
        reliability_sample_size=40,
        # live backends via z.ai anthropic proxy
        proposer_backend="anthropic",
        proposer_model=MODEL,
        materialize_backend="anthropic",
        materialize_model=MODEL,
        llm_concurrency=5,
        output_dir="/Users/spangher/Projects/stanford-research/norm-research/outputs/metrics_tree_infilling/smoke_live",
        verbose=True,
    )

    df_d, df_t = discover_test_split(df, cfg)
    print(f"[smoke] split: discover={len(df_d)} test={len(df_t)}")
    metrics = companion_code()
    print(f"[smoke] explicit metrics: {len(metrics)} ({sum(m.kind=='judge' for m in metrics)} judge)")

    judge = make_vllm_judge_scorer(cfg)
    proposer = make_proposer(cfg)

    print("[smoke] materializing base metrics over discover + test ...")
    sm_d = materialize(metrics, df_d, cfg, judge)
    sm_t = materialize(metrics, df_t, cfg, judge)

    result = run_infill(df_d, df_t, metrics, sm_d, sm_t, cfg, proposer, judge)

    kept = [r for r in result.records if r.status == "kept"]
    print(f"\n[smoke] DONE in {time.time() - t0:.0f}s | rounds={result.rounds} "
          f"final_gaps={result.final_gap_count} kept={len(kept)}")
    for r in result.records:
        print(f"     {r.status:7} {r.name!r:28} coverage={getattr(r, 'coverage', float('nan'))}")
    print(f"[smoke] outputs under {cfg.output_dir}/smoke_live/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
