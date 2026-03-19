#!/usr/bin/env python3
"""Test script for Metric Tree with VLLM batch backend (Llama-3.3-70B).

Runs a small end-to-end test: builds a depth-1 tree on 50 train + 20 eval
examples, then predicts on 20 test examples. Verifies every stage works
with the real VLLM backend (no mocks).

Usage:
    # Default: uses VLLM_MODEL env var or Llama-3.3-70B-Instruct
    python scripts/test_metric_tree.py

    # Specify model explicitly
    python scripts/test_metric_tree.py --model meta-llama/Llama-3.3-70B-Instruct

    # Smaller model for faster testing
    python scripts/test_metric_tree.py --model meta-llama/Llama-3.1-8B-Instruct

    # Skip GPU test, verify only CPU logic (no VLLM)
    python scripts/test_metric_tree.py --cpu-only
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AUTOMETRICS_PKG = REPO_ROOT / "methods" / "autometrics"
METRIC_TREE_PKG = REPO_ROOT / "methods"

sys.path.insert(0, str(AUTOMETRICS_PKG))
sys.path.insert(0, str(METRIC_TREE_PKG))

for k in list(sys.modules):
    if k == "autometrics" or k.startswith("autometrics."):
        del sys.modules[k]
    if k == "metric_tree" or k.startswith("metric_tree."):
        del sys.modules[k]

import logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test_metric_tree")

import numpy as np
import pandas as pd

# ── Parse args ──
parser = argparse.ArgumentParser(description="Test Metric Tree end-to-end")
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--cpu-only", action="store_true", help="Run CPU-only tests (no VLLM)")
parser.add_argument("--n-train", type=int, default=50, help="Number of train examples")
parser.add_argument("--n-eval", type=int, default=20, help="Number of eval examples")
parser.add_argument("--n-test", type=int, default=20, help="Number of test examples")
parser.add_argument("--tensor-parallel-size", type=int, default=1)
parser.add_argument("--max-model-len", type=int, default=16384)
args = parser.parse_args()

MODEL = args.model or os.environ.get("VLLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

# ── Helpers ──
passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}{f': {detail}' if detail else ''}")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Load a small sample of real data ──
section("1. Data Loading")
SPLIT_DIR = REPO_ROOT / "datasets" / "press-releases" / "press_release_modeling_dataset.csv"

train_df = pd.read_csv(SPLIT_DIR / "train.csv.gz", low_memory=False).head(args.n_train)
eval_df = pd.read_csv(SPLIT_DIR / "eval.csv.gz", low_memory=False).head(args.n_eval)
test_df = pd.read_csv(SPLIT_DIR / "test.csv.gz", low_memory=False).head(args.n_test)

ID_COL = "id"
TEXT_COL = "output"
LABEL_COL = "newsworthiness_score"

rename_map = {"press_release_id": ID_COL, "text": TEXT_COL, "judgement": LABEL_COL}
train_df = train_df.rename(columns=rename_map).dropna(subset=[ID_COL, TEXT_COL, LABEL_COL])
eval_df = eval_df.rename(columns=rename_map).dropna(subset=[ID_COL, TEXT_COL, LABEL_COL])
test_df = test_df.rename(columns=rename_map).dropna(subset=[ID_COL, TEXT_COL, LABEL_COL])

check("Train loaded", len(train_df) > 0, f"{len(train_df)} rows")
check("Eval loaded", len(eval_df) > 0, f"{len(eval_df)} rows")
check("Test loaded", len(test_df) > 0, f"{len(test_df)} rows")
check("Labels are binary", set(train_df[LABEL_COL].unique()) <= {True, False, 0, 1})

print(f"  Train: {len(train_df)} rows, label dist: {dict(train_df[LABEL_COL].value_counts())}")
print(f"  Eval:  {len(eval_df)} rows, label dist: {dict(eval_df[LABEL_COL].value_counts())}")
print(f"  Test:  {len(test_df)} rows, label dist: {dict(test_df[LABEL_COL].value_counts())}")


# ── Test CPU-only components ──
section("2. CPU Components (no GPU)")

from metric_tree.config import TreeConfig
from metric_tree.data_structures import TreeMetric, MetricTreeNode, MetricTree
from metric_tree.routing import tune_threshold, build_learned_router
from metric_tree.scoring import add_interaction_features, build_feature_matrix
from metric_tree.analysis import analyze_tree_complexity, measure_depth_distribution, export_tree_summary
from metric_tree.inference import predict_root_only

config = TreeConfig(max_depth=1, min_subset_size=5, n_metrics_to_propose=3, n_rubrics_to_propose=3)
check("TreeConfig created", config.max_depth == 1)

# Test interaction features
X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
names = ["a", "b", "c"]
X_aug, aug_names, pairs = add_interaction_features(X, names)
check("Interaction features shape", X_aug.shape == (3, 6), f"got {X_aug.shape}")
check("Interaction names", aug_names == ["a", "b", "c", "a__x__b", "a__x__c", "b__x__c"])
check("Interaction values", X_aug[0, 3] == 2.0 and X_aug[0, 5] == 6.0)  # 1*2, 2*3

# Test build_feature_matrix
scored = pd.DataFrame({
    "id": ["a", "b", "c"],
    "label": [0, 1, 0],
    "m1": [1.0, 2.0, 3.0],
    "m2": [4.0, 5.0, 6.0],
})
X, y = build_feature_matrix(scored, ["m1", "m2"], "label")
check("Feature matrix shape", X.shape == (3, 2))
check("Labels extracted", list(y) == [0.0, 1.0, 0.0])

# Test NaN handling
scored_nan = scored.copy()
scored_nan.loc[0, "m1"] = np.nan
X_nan, _ = build_feature_matrix(scored_nan, ["m1", "m2"], "label")
check("NaN replaced with 0", X_nan[0, 0] == 0.0)

# Test tune_threshold
probs = np.array([0.95, 0.85, 0.6, 0.3, 0.1, 0.2, 0.7, 0.9])
labels = np.array([1, 1, 1, 0, 0, 0, 0, 1])
thresh, acc, rate = tune_threshold(probs, labels)
check("Threshold tuning", 0.5 <= thresh <= 0.95, f"threshold={thresh:.3f}")
check("Tuned accuracy > random", acc > 0.5, f"acc={acc:.3f}")

# Test tree analysis with mock tree
mock_tree = MetricTree()
root = MetricTreeNode(
    node_id="root", depth=0,
    point_indices=np.arange(10),
    correct_mask=np.array([True]*8 + [False]*2),
    predictions=np.array([1,1,1,1,0,0,0,0,1,0]),
    train_accuracy=0.8, eval_accuracy=0.75,
)
mock_tree.root = root
mock_tree.all_nodes["root"] = root
complexity = analyze_tree_complexity(mock_tree)
check("Tree complexity analysis", len(complexity) == 1)
check("Complexity fields", set(complexity.columns) >= {"node_id", "depth", "train_accuracy"})

# Test export
with tempfile.TemporaryDirectory() as tmpdir:
    export_tree_summary(mock_tree, f"{tmpdir}/summary.txt")
    summary_text = Path(f"{tmpdir}/summary.txt").read_text()
    check("Tree summary exported", "METRIC TREE SUMMARY" in summary_text)

# Test depth distribution
mock_preds = pd.DataFrame({
    "resolving_node": ["root"] * 8 + ["root_false_positive"] * 2,
    "prediction": [1,1,1,1,0,0,0,0,1,0],
})
child = MetricTreeNode(node_id="root_false_positive", depth=1)
mock_tree.all_nodes["root_false_positive"] = child
depth_dist = measure_depth_distribution(mock_tree, mock_preds)
check("Depth distribution", len(depth_dist) == 2, f"depths: {depth_dist['depth'].tolist()}")

# Test confidence-weighted ensemble voting
from metric_tree.ensemble import ensemble_predict
# (can't fully test without trees, but verify import works)
check("Ensemble import", True)


# ── Stop here if CPU-only ──
if args.cpu_only:
    section("RESULTS (CPU-only)")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    sys.exit(0 if failed == 0 else 1)


# ── Test with VLLM backend ──
section("3. VLLM Backend Initialization")

import dspy
from autometrics.backends import create_backend
from autometrics.generator.ContrastiveRubricProposer import ContrastiveRubricProposer
from autometrics.iterative_refinement.label_cache import LabelCache
from autometrics.task_descriptions import get_task_description

TASK_DESCRIPTION = get_task_description("PressReleaseModeling")

print(f"  Loading VLLM with model: {MODEL}")
print(f"  tensor_parallel_size={args.tensor_parallel_size}, max_model_len={args.max_model_len}")
t0 = time.time()

backend = create_backend(
    "vllm",
    model_name_or_path=MODEL,
    tensor_parallel_size=args.tensor_parallel_size,
    max_model_len=args.max_model_len,
    gpu_memory_utilization=0.95,
)
load_time = time.time() - t0
check("VLLM backend created", backend is not None, f"loaded in {load_time:.1f}s")

# Load tokenizer for token-precise truncation (budgets computed dynamically)
from metric_tree.token_utils import get_tokenizer
tokenizer = get_tokenizer(MODEL)
token_budgets = {"max_model_len": args.max_model_len}
print(f"  Token truncation: max_model_len={args.max_model_len}, budgets computed dynamically")

generator_llm = dspy.LM(model=f"openai/{MODEL}", api_key="unused", api_base="http://localhost:1")

proposer = ContrastiveRubricProposer(
    generator_llm=generator_llm,
    seed=42,
    scoring_backend=backend,
)
check("ContrastiveRubricProposer created", proposer is not None)


# ── Test 4: Metric generation via VLLM batch ──
section("4. Metric Generation (VLLM batch)")

from autometrics.iterative_refinement.runner import (
    _format_examples, _coerce_binary_labels, _truncate_text,
)

train_bin = _coerce_binary_labels(train_df.copy(), LABEL_COL)
pos_sample = train_bin[train_bin[LABEL_COL] == 1].head(3)
neg_sample = train_bin[train_bin[LABEL_COL] == 0].head(3)

pos_text = _format_examples(pos_sample, ID_COL, TEXT_COL, LABEL_COL)
neg_text = _format_examples(neg_sample, ID_COL, TEXT_COL, LABEL_COL)

print(f"  Generating metrics from {len(pos_sample)} pos + {len(neg_sample)} neg examples...")
t0 = time.time()
raw_metrics = proposer.propose(
    task_description=TASK_DESCRIPTION,
    positive_examples=pos_text,
    negative_examples=neg_text,
    current_metrics="",
    contrastive_pairs="",
    num_metrics=3,
    num_rubrics=3,
)
gen_time = time.time() - t0
check("Metrics generated", len(raw_metrics) > 0, f"{len(raw_metrics)} metrics in {gen_time:.1f}s")

if raw_metrics:
    for m in raw_metrics:
        print(f"    - {m.get('name', '?')} (scale={m.get('scale', '?')})")
        rubric = m.get("rubric", {})
        check(f"  Metric '{m.get('name')}' has rubric", isinstance(rubric, dict) and len(rubric) > 0)


# ── Test 5: Metric scoring via VLLM batch ──
section("5. Metric Scoring (VLLM batch)")

from metric_tree.scoring import score_subset
from metric_tree.tree_builder import _create_tree_metrics
from autometrics.iterative_refinement.runner import _normalize_rubric, _rubric_to_text

with tempfile.TemporaryDirectory() as cache_dir:
    label_cache = LabelCache(cache_dir)

    existing_ids = set()
    existing_names = set()
    tree_metrics = _create_tree_metrics(raw_metrics, "root", existing_ids, existing_names)
    check("TreeMetrics created", len(tree_metrics) > 0, f"{len(tree_metrics)} metrics")

    if tree_metrics:
        # Score a small batch
        small_df = train_bin.head(10)
        print(f"  Scoring {len(small_df)} examples on {len(tree_metrics)} metrics...")
        t0 = time.time()
        scored_df = score_subset(
            small_df, np.arange(len(small_df)), tree_metrics, label_cache,
            id_column=ID_COL, text_column=TEXT_COL, label_column=LABEL_COL,
            judge_llm=generator_llm, task_description=TASK_DESCRIPTION,
            batch_size=200, verbose=True,
            stage="test_score", scoring_backend=backend,
        )
        score_time = time.time() - t0

        metric_cols = [m.name for m in tree_metrics if m.name in scored_df.columns]
        check("Scored DataFrame has metric columns", len(metric_cols) > 0, f"cols: {metric_cols}")
        check("Scores are numeric", scored_df[metric_cols].dtypes.apply(lambda d: np.issubdtype(d, np.number)).all())
        check("Score range reasonable", scored_df[metric_cols].max().max() <= 6 and scored_df[metric_cols].min().min() >= 0,
              f"range [{scored_df[metric_cols].min().min():.1f}, {scored_df[metric_cols].max().max():.1f}]")
        print(f"  Scoring took {score_time:.1f}s for {len(small_df)} examples × {len(tree_metrics)} metrics")
        print(f"  Score stats:\n{scored_df[metric_cols].describe().to_string()}")

        # Test caching: re-score same examples, should be instant
        t0 = time.time()
        scored_df2 = score_subset(
            small_df, np.arange(len(small_df)), tree_metrics, label_cache,
            id_column=ID_COL, text_column=TEXT_COL, label_column=LABEL_COL,
            judge_llm=generator_llm, task_description=TASK_DESCRIPTION,
            batch_size=200, verbose=True,
            stage="test_cache_hit", scoring_backend=backend,
        )
        cache_time = time.time() - t0
        check("Cache hit is fast", cache_time < score_time * 0.5,
              f"cache={cache_time:.2f}s vs fresh={score_time:.2f}s")

        # Verify cached scores match
        for col in metric_cols:
            match = np.allclose(scored_df[col].values, scored_df2[col].values, equal_nan=True)
            check(f"Cache consistency ({col})", match)


# ── Test 6: Full tree build (depth=1, small data) ──
section("6. Full Metric Tree Build (depth=1)")

from metric_tree.tree_builder import build_metric_tree

with tempfile.TemporaryDirectory() as tmpdir:
    test_config = TreeConfig(
        max_depth=1,
        min_subset_size=5,
        n_metrics_to_propose=3,
        n_rubrics_to_propose=3,
        use_interactions=False,  # keep simple for test
        random_seed=42,
        output_dir=tmpdir,
        verbose=True,
    )

    print(f"  Building tree: {len(train_df)} train, {len(eval_df)} eval, depth=1...")
    t0 = time.time()
    tree = build_metric_tree(
        train_df=train_bin,
        eval_df=_coerce_binary_labels(eval_df.copy(), LABEL_COL),
        config=test_config,
        proposer=proposer,
        task_description=TASK_DESCRIPTION,
        id_column=ID_COL,
        text_column=TEXT_COL,
        label_column=LABEL_COL,
        judge_llm=generator_llm,
        cache_dir=f"{tmpdir}/cache",
        scoring_backend=backend,
        tokenizer=tokenizer,
        token_budgets=token_budgets,
    )
    build_time = time.time() - t0
    print(f"  Tree built in {build_time:.1f}s")

    check("Tree has root", tree.root is not None)
    check("Root has classifier", tree.root.classifier is not None)
    check("Root has metrics", len(tree.root.all_metrics) > 0, f"{len(tree.root.all_metrics)} metrics")
    check("Root has feature names", len(tree.root.feature_names) > 0)
    check("Root train_accuracy > 0", tree.root.train_accuracy > 0)
    check("All nodes registered", len(tree.all_nodes) >= 1, f"{len(tree.all_nodes)} nodes")
    check("All metrics registered", len(tree.all_metrics) >= 1, f"{len(tree.all_metrics)} metrics")

    n_children = len(tree.root.children)
    print(f"  Root: {len(tree.root.all_metrics)} metrics, acc={tree.root.train_accuracy:.3f}, {n_children} children")
    for child_type, child in tree.root.children.items():
        n_local = len(child.local_metrics)
        n_all = len(child.all_metrics)
        print(f"    {child_type}: {n_local} local + {n_all - n_local} inherited = {n_all} total metrics, "
              f"acc={child.train_accuracy:.3f}")
        check(f"Child '{child_type}' inherits parent metrics", n_all > n_local)

    # Tree complexity
    complexity = analyze_tree_complexity(tree)
    print(f"\n  Tree complexity:\n{complexity.to_string()}")

    # Export summary
    export_tree_summary(tree, f"{tmpdir}/summary.txt")
    summary = Path(f"{tmpdir}/summary.txt").read_text()
    check("Summary has root node", "[root]" in summary)
    check("Summary has metrics", "Rubric:" in summary)


    # ── Test 7: Inference ──
    section("7. Inference (predict on test set)")

    from metric_tree.inference import predict_batch, predict_root_only

    test_bin = _coerce_binary_labels(test_df.copy(), LABEL_COL)
    label_cache = LabelCache(f"{tmpdir}/cache")

    print(f"  Predicting on {len(test_bin)} test examples...")
    t0 = time.time()
    preds_df = predict_batch(
        tree=tree,
        df=test_bin,
        label_cache=label_cache,
        id_column=ID_COL,
        text_column=TEXT_COL,
        label_column=LABEL_COL,
        judge_llm=generator_llm,
        task_description=TASK_DESCRIPTION,
        batch_size=200,
        scoring_backend=backend,
        verbose=True,
        max_model_len=args.max_model_len,
        tokenizer=tokenizer,
    )
    pred_time = time.time() - t0
    print(f"  Prediction took {pred_time:.1f}s")

    check("Predictions DataFrame", len(preds_df) == len(test_bin))
    check("Has prediction column", "prediction" in preds_df.columns)
    check("Has probability column", "probability" in preds_df.columns)
    check("Has resolving_node column", "resolving_node" in preds_df.columns)
    check("Predictions are 0/1", set(preds_df["prediction"].unique()) <= {0, 1})
    check("Probabilities in [0,1]", preds_df["probability"].between(0, 1).all())

    test_labels = test_bin[LABEL_COL].values
    test_acc = (preds_df["prediction"].values == test_labels).mean()
    print(f"  Test accuracy: {test_acc:.3f}")
    check("Test accuracy > random", test_acc > 0.4, f"acc={test_acc:.3f}")

    # Resolution depth distribution
    depth_dist = measure_depth_distribution(tree, preds_df)
    print(f"  Depth distribution:\n{depth_dist.to_string()}")

    # Root-only predictions for articulability gap
    print(f"  Computing root-only predictions...")
    root_preds = predict_root_only(
        tree=tree,
        df=test_bin,
        label_cache=label_cache,
        id_column=ID_COL,
        text_column=TEXT_COL,
        label_column=LABEL_COL,
        judge_llm=generator_llm,
        task_description=TASK_DESCRIPTION,
        batch_size=200,
        scoring_backend=backend,
        max_model_len=args.max_model_len,
        tokenizer=tokenizer,
    )
    root_acc = (root_preds == test_labels).mean()
    print(f"  Root-only accuracy: {root_acc:.3f}")

    # Articulability gap
    from metric_tree.analysis import compute_articulability_gap
    gap = compute_articulability_gap(tree, preds_df, test_labels, root_preds)
    print(f"\n  Articulability Gap:")
    print(f"    Root accuracy:  {gap['root_accuracy']:.3f}")
    print(f"    Tree accuracy:  {gap['tree_accuracy']:.3f}")
    print(f"    Gap:            {gap['articulability_gap']:.3f}")
    print(f"    Per-depth:      {gap['per_depth_accuracy']}")
    check("Gap computed", "articulability_gap" in gap)


# ── Summary ──
section("RESULTS")
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
if failed > 0:
    print(f"\n  {failed} test(s) FAILED!")
    sys.exit(1)
else:
    print(f"\n  All tests passed!")
    sys.exit(0)
