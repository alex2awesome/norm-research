"""CLI entry point for metrics-tree infilling.

Loads a labeled corpus + an explicit metric set, runs the gap-detecting infilling loop, and
writes the discovered features (with measured depth / reliability / gap-closure) and the final
tree summary to ``output_dir/<task>/``.

Example
-------
    PYTHONPATH=methods python -m metrics_tree_infilling.run \
        --task peer-review --metrics rubric --max-metrics 40 \
        --proposer-backend anthropic --materialize-backend openai_compatible \
        --openai-base-url http://localhost:8000/v1

Metric sources:  ``rubric`` (datasets/<task>/online-rubrics/*-parsed),  ``code`` (a directory
of ``score(text)`` modules, e.g. methods/existing_metrics_runner/coded/metrics),  or ``both``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from .config import InfillConfig
from .io_metrics import (
    REPO_ROOT,
    load_code_metrics,
    load_items,
    load_rubric_metrics,
    load_rubric_metrics_from_dir,
    make_design,
    make_vllm_judge_scorer,
    materialize,
    discover_test_split,
)
from .feature_gen import make_proposer
from .loop import run_infill

# Dataset configs (mirrors scripts/run_metric_tree.py)
DATASET_CONFIGS = {
    "press-release": {"split": "datasets/press-releases/press_release_modeling_dataset.csv",
                      "id": "id", "text": "text", "label": "judgement"},
    # creative-writing carries all THREE label types (§IV). "creative-writing" = the
    # community-REVEALED leg (WritingPrompts upvotes); the two expert legs below are built.
    "creative-writing": {"split": "datasets/creative-writing/litbench-to-train.csv.gz",
                         "id": "Unnamed: 0", "text": "text", "label": "judgement",
                         "label_type": "community-revealed"},
    "creative-writing-wigleaf": {  # expert-REVEALED (curatorial): Wigleaf Top-50 editor cut
        "split": "datasets/creative-writing/wigleaf/built/train.csv.gz",
        "id": "text", "text": "text", "label": "judgement", "label_type": "expert-revealed"},
    "creative-writing-royalroad": {  # expert/market-REVEALED: RoyalRoad -> KU/Amazon deal
        "split": "datasets/creative-writing/royalroad_stubs/built/royalroad_v2_fiction_topicstrat.csv.gz",
        "id": "fiction_id", "text": "text", "label": "judgement", "label_type": "expert-revealed"},
    "peer-review": {"split": "datasets/peer-review/peer_review_modeling_dataset.csv.gz",
                    "id": "paper_id", "text": "text", "label": "judgement"},
    "code-review": {"split": "datasets/code-review/code_review_dense_4096tok",
                    "id": "paper_id", "text": "text", "label": "judgement"},
    "notice-and-comment": {"split": "datasets/notice-and-comment/notice_and_comment_len_balanced",
                           "id": "id", "text": "text", "label": "judgement"},
}

# Math sub-community (primary_tag) legs — the within-SUBTASK infilling experiment (task #60).
# The 2026-07-07 powered read showed subfield preference differences are NOT expressible by
# reweighting the general bank (d_spec~0, coef rho at the split-half noise ceiling), so the
# arms run WITHIN each tag: proposals must carry tag-local content to clear the gate.
# id=question_id + group_split keeps same-question answers in one split (position-matched
# pairs never straddle discover/guard/test).
_MATH_TAGS = ["real-analysis", "calculus", "linear-algebra", "abstract-algebra", "probability",
              "algebra-precalculus", "general-topology", "combinatorics", "sequences-and-series",
              "complex-analysis", "geometry", "integration"]
DATASET_CONFIGS.update({
    f"math-{t}": {"split": f"datasets/math/stackexchange/by_tag/{t}.csv.gz",
                  "id": "question_id", "text": "text", "label": "judgement",
                  "label_type": "community-revealed", "group_split": True}
    for t in _MATH_TAGS
})
# Pooled control for the within-tag legs: SAME 12-tag item universe, unconditioned. Tests the
# dilution mechanism directly — a tag-local metric worth b bits at share s reads ~b*s pooled.
DATASET_CONFIGS["math-pooled-12tags"] = {
    "split": "datasets/math/stackexchange/by_tag/pooled-12tags.csv.gz",
    "id": "question_id", "text": "text", "label": "judgement",
    "label_type": "community-revealed", "group_split": True}

# CW prompt-genre communities (KMeans k=25 on MiniLM prompt embeddings; removal-boilerplate
# excluded) + humor LDA-topic communities — the cross-domain replication of the math
# within-subtask infilling result (#60). CW groups by prompt (stories sharing a prompt never
# straddle splits); humor jokes have no group structure.
_CW_GENRES = ["abstract-premise", "immortality", "wakeup-mystery", "hell-deal", "pooled-4genres",
              "aliens", "villain", "soulmate", "ai", "time-travel", "meta-experimental"]
DATASET_CONFIGS.update({
    f"cw-genre-{g}": {"split": f"datasets/creative-writing/by_genre/{g}.csv.gz",
                      "id": "prompt", "text": "text", "label": "judgement",
                      "label_type": "community-revealed", "group_split": True}
    for g in _CW_GENRES
})
_HUMOR_TOPICS = ["marriage", "bar-jokes", "family", "doctor", "pooled-4topics",
                 "political-classroom", "police", "chicken-crossing", "everyday-observational",
                 "absurd-wordplay", "topical-corona"]
DATASET_CONFIGS.update({
    f"humor-topic-{t}": {"split": f"datasets/humor/by_topic/{t}.csv.gz",
                         "id": "text", "text": "text", "label": "judgement",
                         "label_type": "community-revealed"}
    for t in _HUMOR_TOPICS
})

# Peer-review SUBFIELDS (task #66): venue held fixed to ICLR (isolates subfield from the venue
# base-rate confound) and abstracts topic-modeled into subfields (build_iclr_subfields.py).
# Registered programmatically from the by_subfield dir so TF-IDF-derived slugs are picked up;
# a no-op where the dir is absent (e.g. laptop). paper_id is unique per abstract (no groups).
import glob as _glob
import os as _os
_PR_SUBFIELD_DIR = "datasets/peer-review/by_subfield"
for _p in sorted(_glob.glob(f"{_PR_SUBFIELD_DIR}/*.csv.gz")):
    _slug = _os.path.basename(_p)[: -len(".csv.gz")]
    _name = "peer-iclr-general" if _slug == "_general-iclr" else f"peer-iclr-{_slug}"
    DATASET_CONFIGS[_name] = {"split": _p, "id": "paper_id", "text": "text",
                              "label": "judgement", "label_type": "community-revealed",
                              "group_split": False}

# Scale-out wave (task #66): more sibling axes registered from balanced strata dirs
# (build_strata.py). Siblings = files NOT starting with "_"; the control is the exact
# "_general-<axis>.csv.gz" -> "<prefix>-general". Other "_"-prefixed files (orphan unbalanced
# generals) are ignored. Each axis is class-balanced 50/50 within-sibling.
_STRATA_AXES = [
    ("datasets/peer-review/by_venue",         "peer-venue",   "paper_id"),  # metadata
    ("datasets/code-review/by_language",      "code-lang",    "paper_id"),  # metadata
    ("datasets/notice-and-comment/by_topic",  "notice-topic", "id"),        # topic-model
    ("datasets/press-releases/by_topic",      "press-topic",  "id"),        # topic-model
]
for _dir, _prefix, _idcol in _STRATA_AXES:
    for _p in sorted(_glob.glob(f"{_dir}/*.csv.gz")):
        _b = _os.path.basename(_p)[: -len(".csv.gz")]
        if _b.startswith("_general"):
            _name = f"{_prefix}-general"
        elif _b.startswith("_"):
            continue
        else:
            _name = f"{_prefix}-{_b}"
        DATASET_CONFIGS[_name] = {"split": _p, "id": _idcol, "text": "text",
                                  "label": "judgement", "label_type": "community-revealed",
                                  "group_split": False}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(DATASET_CONFIGS))
    p.add_argument("--metrics", choices=["rubric", "code", "both"], default="rubric")
    p.add_argument("--code-metrics-dir", default="methods/existing_metrics_runner/coded/metrics")
    p.add_argument("--rubrics-dir", default=None,
                   help="dir of distilled rubrics JSON; overrides datasets/<task>/online-rubrics")
    p.add_argument("--max-metrics", type=int, default=40,
                   help="Cap the explicit metric set (rubric sources can yield thousands).")
    p.add_argument("--max-outer-rounds", type=int, default=None)
    p.add_argument("--n-permutations", type=int, default=None)
    p.add_argument("--proposer-backend", default=None, choices=["anthropic", "openai_compatible"])
    p.add_argument("--proposer-model", default=None)
    p.add_argument("--materialize-backend", default=None,
                   choices=["vllm", "openai_compatible", "anthropic"])
    p.add_argument("--materialize-model", default=None)
    p.add_argument("--openai-base-url", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=None)
    return p


def build_config(args) -> InfillConfig:
    dcfg = DATASET_CONFIGS[args.task]
    overrides = dict(
        id_column=dcfg["id"], text_column=dcfg["text"], label_column=dcfg["label"],
    )
    for src, dst in [
        ("max_outer_rounds", "max_outer_rounds"), ("n_permutations", "n_permutations"),
        ("proposer_backend", "proposer_backend"), ("proposer_model", "proposer_model"),
        ("materialize_backend", "materialize_backend"), ("materialize_model", "materialize_model"),
        ("openai_base_url", "openai_base_url"), ("output_dir", "output_dir"), ("seed", "random_seed"),
    ]:
        v = getattr(args, src)
        if v is not None:
            overrides[dst] = v
    return InfillConfig(**overrides)


def load_metric_set(args, cfg):
    metrics = []
    if args.metrics in ("rubric", "both"):
        if getattr(args, "rubrics_dir", None):
            metrics += load_rubric_metrics_from_dir(args.rubrics_dir)
        else:
            metrics += load_rubric_metrics(args.task, limit=None)
    if args.metrics in ("code", "both"):
        metrics += load_code_metrics(args.code_metrics_dir)
    # Cap to a tractable explicit set; rubric sources can return thousands.
    if args.max_metrics and len(metrics) > args.max_metrics:
        rng = np.random.default_rng(cfg.random_seed)
        sel = rng.choice(len(metrics), size=args.max_metrics, replace=False)
        metrics = [metrics[i] for i in sorted(sel)]
    return metrics


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = build_config(args)
    dcfg = DATASET_CONFIGS[args.task]

    df = load_items(REPO_ROOT / dcfg["split"], cfg)
    df_d, df_t = discover_test_split(df, cfg)
    print(f"Loaded {len(df)} items -> discover={len(df_d)} test={len(df_t)}")

    metrics = load_metric_set(args, cfg)
    print(f"Explicit metric set: {len(metrics)} "
          f"({sum(m.kind=='judge' for m in metrics)} judge, {sum(m.kind=='code' for m in metrics)} code)")

    judge_scorer = make_vllm_judge_scorer(cfg)
    proposer = make_proposer(cfg)

    print("Materializing base metric set over discover + test ...")
    sm_d = materialize(metrics, df_d, cfg, judge_scorer)
    sm_t = materialize(metrics, df_t, cfg, judge_scorer)

    result = run_infill(df_d, df_t, metrics, sm_d, sm_t, cfg, proposer, judge_scorer)

    out_dir = Path(cfg.output_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_outputs(out_dir, result, cfg)
    kept = [r for r in result.records if r.status == "kept"]
    print(f"\nDone: {len(kept)} features kept across {result.rounds} rounds; "
          f"final gap nodes={result.final_gap_count}. Wrote {out_dir}")
    return 0


def _write_outputs(out_dir: Path, result, cfg) -> None:
    (out_dir / "features.json").write_text(json.dumps(
        [dataclasses.asdict(r) for r in result.records], indent=2))
    (out_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2, default=str))
    summary = {
        "rounds": result.rounds,
        "final_gap_count": result.final_gap_count,
        "n_terminal_nodes": len(result.tree.terminal_nodes()),
        "kept": [r.name for r in result.records if r.status == "kept"],
        "root_std_coef": result.tree.root_std_coef,
        "tree": _tree_dict(result.tree.root),
    }
    (out_dir / "tree_summary.json").write_text(json.dumps(summary, indent=2, default=str))


def _tree_dict(node) -> dict:
    if node is None:
        return {}
    d = {"id": node.node_id, "depth": node.depth, "n": int(len(node.indices)),
         "base_rate": round(node.base_rate, 3), "terminal": node.is_terminal}
    if node.split is not None:
        d["split"] = node.split.describe()
        d["left"] = _tree_dict(node.left)
        d["right"] = _tree_dict(node.right)
    return d


if __name__ == "__main__":
    raise SystemExit(main())
