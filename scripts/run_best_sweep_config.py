#!/usr/bin/env python
"""Validate a specific Optuna sweep config on FULL train/eval/test.

Loads a trial's params from `trial_XXXX/params.json` and runs the full
pipeline (HDBSCAN → dedup → rubric → FULL scoring → L1) using Llama-70B
on the full splits. Writes final test AUC.
"""

import argparse
import asyncio
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "methods" / "autometrics"))

from methods.autometrics.autometrics.task_descriptions import get_task_description
from methods.local_explanations.cluster_features import (
    embed_with_model, label_clusters_with_llm, run_umap_hdbscan,
)
from methods.local_explanations.config import (
    TASK_METADATA_REGISTRY, LocalExplanationConfig,
)
from methods.local_explanations.explainer import LocalExplainer
from methods.local_explanations.feature_weights import compute_star_weights
from methods.local_explanations.rubric_generation import (
    CanonicalFeature, generate_rubrics,
)
from methods.local_explanations.scorer import score_examples_on_vocabulary
from scripts.run_local_explanations import (
    DATASET_CONFIGS, create_generate_fn, load_splits,
)
from scripts.select_canonical_via_dedup import select_top_families
from scripts.sweep_cluster_selection_epsilon import run_dedup_judge
from scripts.sweep_umap_target_weight import compute_feature_label_props

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--sweep-output-dir", type=str, required=True,
                        help="Dir with caches/, clustering/, explanation_cache/")
    parser.add_argument("--params-json", type=str, required=True,
                        help="Path to a trial's params.json")
    parser.add_argument("--final-output-dir", type=str, required=True)
    parser.add_argument("--model", type=str,
                        default="/lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/fde04ee76a27704c88f569542ef023b57d4d0362")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--proposer-model", type=str, default="openai/gpt-5-mini")
    parser.add_argument("--agency", type=str, default=None)
    args = parser.parse_args()

    with open(args.params_json) as f:
        params = json.load(f)
    logger.info("Loaded params from %s: %s", args.params_json, params)

    dcfg = dict(DATASET_CONFIGS[args.dataset])
    if args.agency:
        dcfg["split_dir"] = (REPO_ROOT / "datasets" / "notice-and-comment" / "agencies" / args.agency)
    task_description = get_task_description(dcfg["dataset_name"])
    task_metadata = TASK_METADATA_REGISTRY[dcfg["dataset_name"]]

    train_df, eval_df, test_df = load_splits(dcfg["split_dir"])
    id_col, label_col, text_col = dcfg["id_column"], dcfg["label_column"], dcfg["text_column"]
    for df, tag in [(train_df, "train"), (eval_df, "eval"), (test_df, "test")]:
        if id_col not in df.columns:
            df[id_col] = [f"{tag}_{i}" for i in range(len(df))]

    logger.info("FULL splits: train=%d, eval=%d, test=%d",
                len(train_df), len(eval_df), len(test_df))

    out = Path(args.final_output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load cached extractions from sweep dir
    sweep = Path(args.sweep_output_dir)
    cache_dir = sweep / "explanation_cache"
    cluster_dir = sweep / "clustering"

    cfg = LocalExplanationConfig(
        approach="star_local",
        n_canonical_features=params["top_k"],
        output_dir=str(out),
        clustering_method="hdbscan",
        hdbscan_min_cluster_size=params["min_cluster_size"],
        umap_target_weight=0.0,
        hdbscan_cluster_selection_epsilon=0.0,
        weight_correct_winning=params["wcw"],
        weight_correct_losing=params["wcl"],
        weight_incorrect_winning=params["wiw"],
        weight_incorrect_losing=params["wil"],
        shrinkage_alpha=(None if params["shrinkage_alpha"] == "None"
                          else float(params["shrinkage_alpha"])),
    )

    # --- Load VLLM backend FIRST ---
    from methods.autometrics.autometrics.backends import create_backend
    logger.info("Loading VLLM (Llama-70B)")
    backend = create_backend(
        "vllm", model_name_or_path=args.model,
        tensor_parallel_size=1, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # --- Load cached extractions (full train for clustering weights) ---
    explainer = LocalExplainer(
        cfg, scoring_backend=None,
        task_description=task_description, task_metadata=task_metadata,
        cache_dir=str(cache_dir),
    )
    doc_ids = train_df[id_col].astype(str).tolist()
    texts = train_df[text_col].tolist()
    star_results = explainer.explain_star_local_batch(texts=texts, doc_ids=doc_ids)
    true_labels = dict(zip(
        train_df[id_col].astype(str), train_df[label_col].astype(int),
    ))

    # Weighted features with this trial's config
    wfs = compute_star_weights(
        star_results, true_labels, task_metadata.positive_label, cfg,
    )
    fw = defaultdict(float)
    for feat, w in wfs:
        feat = feat.strip()
        if feat:
            fw[feat] += w
    features = list(fw.keys())
    weights = np.array([fw[f] for f in features])
    logger.info("Unique features: %d", len(features))

    # Load cached embeddings
    emb = np.load(cluster_dir / "feature_embeddings.npy")
    assert emb.shape[0] == len(features), (emb.shape, len(features))

    # Load cached assignments + names for this mcs
    mcs = params["min_cluster_size"]
    assign_path = sweep / "caches" / f"assignments_mcs{mcs}.npz"
    names_path = sweep / "caches" / f"names_mcs{mcs}.json"
    data = np.load(assign_path)
    assignments, centroids = data["assignments"], data["centroids"]
    with open(names_path) as f:
        names_by_cid = {int(k): v for k, v in json.load(f).items()}
    n_clusters = centroids.shape[0]
    logger.info("Using cached HDBSCAN for mcs=%d: K=%d, %d named",
                mcs, n_clusters, len(names_by_cid))

    # Rank clusters with this trial's weights
    stats = []
    for cid in range(n_clusters):
        mask = assignments == cid
        mi = np.where(mask)[0]
        if len(mi) == 0:
            continue
        mw = weights[mi]
        agg = float(mw.sum())
        count = len(mi)
        alpha = cfg.shrinkage_alpha
        rs = agg if alpha is None else agg / (count + alpha)
        stats.append({
            "cluster_id": cid, "members_idx": mi, "member_weights": mw.tolist(),
            "aggregate_weight": agg, "ranking_score": rs, "count": count,
        })
    stats.sort(key=lambda s: s["ranking_score"], reverse=True)
    top = stats[: params["top_n"]]

    # Build named candidates for dedup judge
    generate_fn = create_generate_fn(args.proposer_model)
    feature_label_props_dict = compute_feature_label_props(star_results, true_labels)
    feature_label_props = np.array([feature_label_props_dict.get(f, 0.5) for f in features])

    named_for_judge = []
    for s in top:
        cid = s["cluster_id"]
        info = names_by_cid.get(cid, {"name": f"cluster_{cid}", "description": ""})
        named_for_judge.append({
            "cluster_id": cid,
            "name": info["name"],
            "description": info["description"],
            "ranking_score": s["ranking_score"],
            "aggregate_weight": s["aggregate_weight"],
            "count": s["count"],
            "member_features": [features[i] for i in s["members_idx"][:20]],
            "centroid": centroids[cid].tolist(),
            "mean_p_positive": float(feature_label_props[s["members_idx"]].mean()),
        })

    logger.info("Running dedup judge on top %d clusters", len(named_for_judge))
    groups = run_dedup_judge(named_for_judge, task_metadata.text_type, generate_fn)
    top_families = select_top_families(named_for_judge, groups, params["top_k"])
    logger.info("Dedup → %d families; picked top %d",
                len(groups), len(top_families))

    # Build canonical stubs → rubrics
    from types import SimpleNamespace
    stubs = []
    for fam in top_families:
        rep = fam["rep"]
        merged_members, merged_weight = [], 0.0
        for m in fam["members"]:
            merged_members.extend(m["member_features"])
            merged_weight += m["aggregate_weight"]
        stubs.append(SimpleNamespace(
            cluster_id=rep["cluster_id"], name=rep["name"],
            description=rep["description"],
            member_features=merged_members[:50], member_weights=[],
            aggregate_weight=merged_weight,
            ranking_score=fam["family_score"],
            centroid=np.array(rep["centroid"]),
        ))
    canonical = generate_rubrics(
        stubs, text_type=task_metadata.text_type,
        generate_fn=generate_fn, top_k=len(stubs),
    )

    # Save canonical_features.json
    (out / "clustering").mkdir(parents=True, exist_ok=True)
    with open(out / "clustering" / "canonical_features.json", "w") as f:
        json.dump([
            {"feature_id": cf.feature_id, "name": cf.name,
             "description": cf.description, "rubric": cf.rubric,
             "aggregate_weight": cf.aggregate_weight,
             "cluster_size": cf.cluster_size,
             "member_features": cf.member_features[:20]}
            for cf in canonical
        ], f, indent=2)
    logger.info("Saved %d canonical features", len(canonical))

    # --- Score FULL train/eval/test ---
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Truncate text column
    from methods.local_explanations.runner import _truncate_texts
    train_df = _truncate_texts(train_df, text_col, cfg.max_text_tokens)
    eval_df  = _truncate_texts(eval_df,  text_col, cfg.max_text_tokens)
    test_df  = _truncate_texts(test_df,  text_col, cfg.max_text_tokens)

    label_cache = str(out / "score_cache")
    scored_train = score_examples_on_vocabulary(
        train_df, canonical, backend, task_description,
        id_column=id_col, text_column=text_col, label_column=label_col,
        label_cache_dir=label_cache,
        tokenizer=tokenizer, max_model_len=args.max_model_len,
        stage="train_final",
    )
    scored_eval = score_examples_on_vocabulary(
        eval_df, canonical, backend, task_description,
        id_column=id_col, text_column=text_col, label_column=label_col,
        label_cache_dir=label_cache,
        tokenizer=tokenizer, max_model_len=args.max_model_len,
        stage="eval_final",
    )
    scored_test = score_examples_on_vocabulary(
        test_df, canonical, backend, task_description,
        id_column=id_col, text_column=text_col, label_column=label_col,
        label_cache_dir=label_cache,
        tokenizer=tokenizer, max_model_len=args.max_model_len,
        stage="test_final",
    )

    # --- L1 fit with trial's C + class_weight ---
    feat_cols = [c for c in scored_train.columns
                 if c not in (id_col, label_col, text_col)]
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
        average_precision_score,
    )
    class_weight = None if params["class_weight"] == "none" else "balanced"
    clf = LogisticRegression(
        penalty="l1", solver="liblinear",
        C=params["l1_C"], class_weight=class_weight, max_iter=500,
    )
    clf.fit(
        scored_train[feat_cols].values,
        scored_train[label_col].values,
    )

    def _metrics(df):
        y = df[label_col].values
        X = df[feat_cols].values
        p = clf.predict_proba(X)[:, 1]
        yhat = clf.predict(X)
        return {
            "accuracy": float(accuracy_score(y, yhat)),
            "precision": float(precision_score(y, yhat, zero_division=0)),
            "recall": float(recall_score(y, yhat, zero_division=0)),
            "f1": float(f1_score(y, yhat, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
        }

    train_m = _metrics(scored_train)
    eval_m  = _metrics(scored_eval)
    test_m  = _metrics(scored_test)

    # Active features
    coefs = clf.coef_[0]
    active = sorted(
        [(feat_cols[i], float(coefs[i])) for i in range(len(coefs)) if abs(coefs[i]) > 1e-10],
        key=lambda x: abs(x[1]), reverse=True,
    )

    results = {
        "approach": "star_local",
        "source_trial_params": params,
        "train": train_m, "eval": eval_m, "test": test_m,
        "n_features": len(feat_cols),
        "n_active_features": len(active),
        "active_features": [{"name": n, "coefficient": c} for n, c in active],
        "canonical_features": [
            {"name": cf.name, "description": cf.description,
             "weight": cf.aggregate_weight}
            for cf in canonical
        ],
    }
    with open(out / "results_star_local.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("=" * 60)
    logger.info("FINAL: train=%.4f eval=%.4f TEST=%.4f active=%d/%d",
                train_m["roc_auc"], eval_m["roc_auc"], test_m["roc_auc"],
                len(active), len(feat_cols))
    logger.info("Results → %s", out / "results_star_local.json")


if __name__ == "__main__":
    main()
