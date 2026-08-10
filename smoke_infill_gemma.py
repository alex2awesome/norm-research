"""Capstone validation of the proposer fix: FULL live loop with Gemma-4 proposer + judge
(via OpenRouter), n=500, asserting song/glow are rediscovered AND kept end-to-end.

This is smoke_infill_live.py with the backend swapped off the quota-exhausted z.ai proxy to
Gemma-4/OpenRouter, and with the result records persisted. Judge calls cache to judge.jsonl.

Run from the norm-research repo root:
    python smoke_infill_gemma.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "methods"))

from methods.metrics_tree_infilling.config import InfillConfig
from methods.metrics_tree_infilling.io_metrics import (
    discover_test_split, make_vllm_judge_scorer, materialize,
)
from methods.metrics_tree_infilling.feature_gen import make_proposer
from methods.metrics_tree_infilling.loop import run_infill
from methods.metrics_tree_infilling.tests.test_scenario.generate import build_corpus
from methods.metrics_tree_infilling.tests.test_scenario.metrics import companion_code

_KEY = Path.home() / ".openrouter-api-key.txt"
import os
os.environ.setdefault("OPENAI_API_KEY", _KEY.read_text().strip())
os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

MODEL = "google/gemma-4-31B-it"
OUT = "/Users/spangher/Projects/stanford-research/norm-research/outputs/metrics_tree_infilling/smoke_gemma"


def _harden_against_rate_limits() -> None:
    """Wrap LLMClient.generate with outer retries (OpenRouter can 429 under concurrency)."""
    from methods.verification_library import client as _vc
    orig = _vc.LLMClient.generate

    async def robust(self, prompt, max_tokens=2048, temperature=0.7, system=""):
        last = None
        for outer in range(6):
            try:
                return await orig(self, prompt, max_tokens, temperature, system)
            except Exception as e:  # noqa: BLE001
                last = e
                wait = min(60.0, 8.0 * (2 ** outer))
                print(f"[gemma-smoke] generate failed ({type(e).__name__}: {str(e)[:80]}); "
                      f"retry {outer + 1}/6 in {wait:.0f}s")
                await asyncio.sleep(wait)
        raise last

    _vc.LLMClient.generate = robust


def main() -> int:
    t0 = time.time()
    _harden_against_rate_limits()
    n = 500
    print(f"[gemma-smoke] corpus n={n} ...")
    df, _ = build_corpus(n=n, seed=7)
    print(f"[gemma-smoke] label rate={df['judgement'].mean():.2f}")

    cfg = InfillConfig(
        n_permutations=199, min_node_size=30, max_depth=4, random_seed=0,
        include_text_length_in_z=False, gap_deviance_per_item=1.20, gap_auc_threshold=0.55,
        contrastive_pairs_k=6, max_outer_rounds=3, max_features_per_round=4,
        reliability_sample_size=40,
        proposer_backend="openai_compatible", proposer_model=MODEL,
        materialize_backend="openai_compatible", materialize_model=MODEL,
        openai_base_url="https://openrouter.ai/api/v1", llm_concurrency=6,
        output_dir=OUT, verbose=True,
    )
    df_d, df_t = discover_test_split(df, cfg)
    metrics = companion_code()
    judge = make_vllm_judge_scorer(cfg)
    proposer = make_proposer(cfg)
    # base metrics are all code (free) -> no judge calls for materialization
    sm_d = materialize(metrics, df_d, cfg, judge)
    sm_t = materialize(metrics, df_t, cfg, judge)

    result = run_infill(df_d, df_t, metrics, sm_d, sm_t, cfg, proposer, judge)

    kept = [r for r in result.records if r.status == "kept"]
    names = " ".join(r.name.lower() for r in kept)
    song = any(s in names for s in ("song", "melod", "luminesc"))  # noqa
    glow = any(s in names for s in ("glow", "lumin", "biolum", "light-emitt", "luminesc"))
    print(f"\n[gemma-smoke] DONE in {time.time() - t0:.0f}s | rounds={result.rounds} "
          f"final_gaps={result.final_gap_count} kept={len(kept)}")
    print(f"[gemma-smoke] song_rediscovered={song}  glow_rediscovered={glow}")
    for r in result.records:
        cov = getattr(r, "coverage", float("nan"))
        print(f"     {r.status:18} {r.name!r:30} coverage={cov:.3f}"
              if isinstance(cov, float) else f"     {r.status:18} {r.name!r}")

    Path(OUT).mkdir(parents=True, exist_ok=True)
    with open(Path(OUT) / "records.json", "w") as f:
        json.dump([{k: getattr(r, k) for k in ("name", "status", "reliability",
                    "redundancy_r2", "gap_drop_fraction", "coverage", "origin")}
                   for r in result.records], f, indent=2, default=str)
    return 0 if (song and glow) else 1


if __name__ == "__main__":
    raise SystemExit(main())
