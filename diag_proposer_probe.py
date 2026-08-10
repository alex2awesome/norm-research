"""Surgical proposer probe (task #32, second half): is the broken seam the PROPOSER?

diag_bisect exonerated engine + cfg + corpus size -> the live smoke failed on the LLM seam.
This isolates the PROPOSER: fit the gap tree on the n=500 creature corpus with the FREE oracle
judge, flag gap nodes, build the marsh/cavern contrasts, and call a LIVE proposer (Gemma-4 via
OpenRouter) on each — next to the oracle proposer's answer on the *same* prompt (the known-correct
articulation).

Read-out:
  * live proposer articulates song/glow  -> proposer seam is fine for a competent model; the
    smoke's glm-5.2 was simply too weak (or the judge was the issue). GEPA-judge optimization
    (run_distillation) may still be the wrong lever.
  * live proposer also fails             -> deeper problem (contrast design / prompt); GEPA-judge
    optimization is definitely NOT the fix.

Run from repo root (reads ~/.openrouter-api-key.txt):
    python diag_proposer_probe.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "methods"))

from methods.metrics_tree_infilling.config import InfillConfig
from methods.metrics_tree_infilling.io_metrics import discover_test_split, materialize, make_design
from methods.metrics_tree_infilling.mob.glmtree import GapTree
from methods.metrics_tree_infilling.gaps import flag_gap_nodes
from methods.metrics_tree_infilling.contrast import build_contrast
from methods.metrics_tree_infilling.feature_gen import make_proposer, propose_feature

from methods.metrics_tree_infilling.tests.test_scenario.generate import build_corpus
from methods.metrics_tree_infilling.tests.test_scenario.metrics import companion_code
from methods.metrics_tree_infilling.tests.test_scenario.oracle import (
    oracle_judge_scorer, oracle_proposer,
)

_KEY = Path.home() / ".openrouter-api-key.txt"
os.environ.setdefault("OPENAI_API_KEY", _KEY.read_text().strip())
os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")


def _cfg() -> InfillConfig:
    return InfillConfig(
        n_permutations=199, min_node_size=30, max_depth=4, random_seed=0,
        max_outer_rounds=2, gap_deviance_per_item=1.20, gap_auc_threshold=0.55,
        contrastive_pairs_k=6, include_text_length_in_z=False,
        proposer_backend="openai_compatible", proposer_model="google/gemma-4-31b-it",
        openai_base_url="https://openrouter.ai/api/v1", llm_concurrency=4,
    )


def _short(s: str, n: int = 150) -> str:
    return (s or "").replace("\n", " ")[:n]


def main() -> int:
    c = _cfg()
    df, _ = build_corpus(n=500, seed=7)
    df_d, df_t = discover_test_split(df, c)
    metrics = companion_code()
    sm_d = materialize(metrics, df_d, c, oracle_judge_scorer)
    Xd, fnd, Zd, _ = make_design(sm_d, df_d, c)
    yd = df_d["judgement"].to_numpy(float)
    tree = GapTree(c).fit(Xd, yd, Zd, fnd)
    gaps = flag_gap_nodes(tree, Xd, yd, Zd, c)
    print(f"gaps flagged: {len(gaps)}  (terminal nodes: {len(tree.terminal_nodes())})")

    known = [(m.description or m.name) for m in metrics][:40]
    proposer = make_proposer(c)
    rng = np.random.default_rng(0)

    for gi, g in enumerate(gaps):
        con = build_contrast(tree, g, df_d, Xd, yd, c, rng)
        if con is None or not con.wrong_pos or not con.wrong_neg:
            print(f"\n[gap {gi}] node={g.node.node_id} — no usable contrast")
            continue
        nn = g.node.n_pos + g.node.n_neg
        print(f"\n[gap {gi}] node={g.node.node_id} depth={g.node.depth} "
              f"n={nn} base_rate={g.node.base_rate:.2f}")
        print("  WRONG-POS (label 1, metrics missed):")
        for t in con.wrong_pos[:3]:
            print("    + " + _short(t))
        print("  WRONG-NEG (label 0, metrics missed):")
        for t in con.wrong_neg[:3]:
            print("    - " + _short(t))

        # oracle reference: same prompt, deterministic correct answer
        pf_or = propose_feature(con, known, c, oracle_proposer)
        print(f"  ORACLE : {_short(pf_or.name + ' | ' + (pf_or.description or ''), 90)}"
              if pf_or else "  ORACLE : (none)")

        # live proposer
        t0 = time.time()
        try:
            pf = propose_feature(con, known, c, proposer)
        except Exception as e:  # noqa: BLE001
            pf = None
            print(f"  LIVE   : ERROR {type(e).__name__}: {_short(str(e), 120)}")
            continue
        dt = time.time() - t0
        if pf:
            print(f"  LIVE   ({dt:.0f}s): {_short(pf.name + ' | ' + (pf.description or ''), 90)}")
            print(f"          rubric: {_short(pf.rubric, 120)}")
            print(f"          raw   : {_short(pf.raw, 160)}")
        else:
            print(f"  LIVE   ({dt:.0f}s): None / unparseable")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
