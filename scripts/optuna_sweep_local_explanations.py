#!/usr/bin/env python
"""Optuna sweep over local-explanations pipeline hyperparameters.

Subsamples train/eval/test deterministically, reuses cached extractions +
embeddings + LoRA-merged BGE. Fits UMAP once at startup (deterministic within
this run) so trials differ only on clustering, selection, weighting, and
predictor params.

Caches rubrics + scores by content hash across trials for incremental speedup.

Per-trial search:
  weight_correct_winning, weight_correct_losing,
  weight_incorrect_winning, weight_incorrect_losing      (continuous)
  hdbscan_min_cluster_size, top_n_candidates,
  top_k_families, shrinkage_alpha                        (categorical)
  l1_C (log-uniform), class_weight                       (predictor)
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "methods" / "autometrics"))

from methods.autometrics.autometrics.task_descriptions import get_task_description
from methods.local_explanations.cluster_features import (
    embed_with_model,
    label_clusters_with_llm,
)
from methods.local_explanations.config import (
    TASK_METADATA_REGISTRY,
    LocalExplanationConfig,
)
from methods.local_explanations.explainer import LocalExplainer
from methods.local_explanations.feature_weights import compute_star_weights
from methods.local_explanations.rubric_generation import generate_rubrics
from methods.local_explanations.scorer import score_examples_on_vocabulary
from scripts.run_local_explanations import (
    DATASET_CONFIGS,
    create_generate_fn,
    load_splits,
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


# -----------------------------------------------------------------------------
# Global state (shared across trials)
# -----------------------------------------------------------------------------

class SweepState:
    def __init__(self):
        self.features: List[str] = []
        self.embeddings: np.ndarray = None
        self.umap_emb: np.ndarray = None
        self.feature_label_props: np.ndarray = None
        self.star_results = None
        self.true_labels = None
        self.task_metadata = None
        self.task_description = None
        # Pre-computed per min_cluster_size:
        #   assignments_by_mcs[mcs] = np.array of cluster ids (or -1)
        #   centroids_by_mcs[mcs] = np.array of centroid embeddings
        #   names_by_mcs[mcs][cid] = {"name": ..., "description": ...}
        self.assignments_by_mcs: Dict[int, np.ndarray] = {}
        self.centroids_by_mcs: Dict[int, np.ndarray] = {}
        self.names_by_mcs: Dict[int, Dict[int, Dict[str, str]]] = {}
        self.openai_async_client = None
        self.naming_model = "gpt-5-mini"
        self.naming_concurrency = 30
        self.dataset_cfg = None
        self.train_sample = None
        self.eval_sample = None
        self.test_sample = None
        self.naming_cache: Dict[str, Dict[str, str]] = {}
        self.rubric_cache: Dict[str, Dict[str, Any]] = {}
        self.scoring_backend = None
        self.tokenizer = None
        self.max_model_len = 0
        self.generate_fn = None
        self.output_dir: Path = None


STATE = SweepState()


# -----------------------------------------------------------------------------
# Cache helpers
# -----------------------------------------------------------------------------

def _hash(obj) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _save_jsonl_append(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _load_jsonl_cache(path: Path) -> Dict[str, Any]:
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[rec["key"]] = rec["value"]
            except (json.JSONDecodeError, KeyError):
                continue
    return out


# -----------------------------------------------------------------------------
# Trial pipeline
# -----------------------------------------------------------------------------

_REASONING_MODEL_TAGS = ("gpt-5", "gpt5", "o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    return any(tag in model.lower() for tag in _REASONING_MODEL_TAGS)


async def _name_one_async(client, prompt: str, model: str,
                          sem: asyncio.Semaphore) -> str:
    async with sem:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if _is_reasoning_model(model):
            kwargs["temperature"] = 1.0
            kwargs["max_completion_tokens"] = 6000
        else:
            kwargs["temperature"] = 0.3
            kwargs["max_tokens"] = 500
        try:
            resp = await client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("Naming failed: %s", exc)
            return ""


async def _name_batch_async(prompts: List[str], model: str,
                            concurrency: int) -> List[str]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)
    tasks = [_name_one_async(client, p, model, sem) for p in prompts]
    results: List[str] = [""] * len(prompts)
    done = 0
    # Use as_completed with index tracking
    task_to_idx = {}
    for i, t in enumerate(tasks):
        task_to_idx[t] = i
    for coro in asyncio.as_completed(tasks):
        # NOTE: asyncio.as_completed loses the idx mapping because tasks are
        # re-awaitable by order of completion. Use a different approach: gather.
        raise RuntimeError("use gather below")
    return results


async def _name_batch_ordered(prompts: List[str], model: str,
                              concurrency: int) -> List[str]:
    """Name all prompts concurrently, preserving order."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)
    n = len(prompts)
    done_count = [0]

    async def _wrap(idx: int, prompt: str):
        r = await _name_one_async(client, prompt, model, sem)
        done_count[0] += 1
        if done_count[0] % 100 == 0 or done_count[0] == n:
            logger.info("Naming progress: %d/%d", done_count[0], n)
        return idx, r

    tasks = [asyncio.create_task(_wrap(i, p)) for i, p in enumerate(prompts)]
    results_dict: Dict[int, str] = {}
    for fut in asyncio.as_completed(tasks):
        idx, r = await fut
        results_dict[idx] = r
    return [results_dict[i] for i in range(n)]


def precompute_cluster_names_for_mcs(min_cluster_sizes: List[int]):
    """For each min_cluster_size, run HDBSCAN on the fixed UMAP embedding and
    name every cluster via async gpt-5-mini. Stores in STATE.names_by_mcs."""
    from methods.local_explanations.prompts import render_cluster_naming_prompt
    from methods.local_explanations.parsing import parse_json_dict

    for mcs in min_cluster_sizes:
        cache_path = STATE.output_dir / "caches" / f"names_mcs{mcs}.json"
        assign_path = STATE.output_dir / "caches" / f"assignments_mcs{mcs}.npz"

        # Compute or load assignments
        if assign_path.exists():
            data = np.load(assign_path)
            assignments = data["assignments"]
            centroids = data["centroids"]
            logger.info("Loaded cached assignments for mcs=%d (K=%d)",
                        mcs, centroids.shape[0])
        else:
            assignments, centroids = run_hdbscan_on_fixed_umap(mcs)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(assign_path,
                                assignments=assignments, centroids=centroids)
            logger.info("Computed assignments for mcs=%d (K=%d, %d noise)",
                        mcs, centroids.shape[0], int((assignments == -1).sum()))
        STATE.assignments_by_mcs[mcs] = assignments
        STATE.centroids_by_mcs[mcs] = centroids

        # Load cached names if present
        cached_names: Dict[int, Dict[str, str]] = {}
        if cache_path.exists():
            with open(cache_path) as f:
                cached_names = {int(k): v for k, v in json.load(f).items()}
            logger.info("Loaded %d cached names for mcs=%d", len(cached_names), mcs)

        n_clusters = centroids.shape[0]
        cids_to_name = [c for c in range(n_clusters) if c not in cached_names]
        if not cids_to_name:
            STATE.names_by_mcs[mcs] = cached_names
            continue

        # Build deterministic naming prompts (unweighted; first 10 members by
        # sorted text order — stable across trials so cache hits are robust).
        prompts = []
        for cid in cids_to_name:
            member_idx = np.where(assignments == cid)[0]
            if len(member_idx) == 0:
                prompts.append(None)
                continue
            member_texts = sorted([STATE.features[i] for i in member_idx])
            sampled = member_texts[:10]
            prompts.append(render_cluster_naming_prompt(sampled))

        logger.info("Async-naming %d clusters for mcs=%d (concurrency=%d)",
                    len([p for p in prompts if p is not None]), mcs,
                    STATE.naming_concurrency)
        # Filter out None prompts
        valid_pairs = [(cid, p) for cid, p in zip(cids_to_name, prompts)
                       if p is not None]
        raw_responses = asyncio.run(_name_batch_ordered(
            [p for _, p in valid_pairs], STATE.naming_model, STATE.naming_concurrency,
        ))
        for (cid, _p), raw in zip(valid_pairs, raw_responses):
            parsed = parse_json_dict(raw)
            cached_names[cid] = {
                "name": parsed.get("name", f"cluster_{cid}"),
                "description": parsed.get("description", ""),
            }
        with open(cache_path, "w") as f:
            json.dump(cached_names, f)
        STATE.names_by_mcs[mcs] = cached_names
        logger.info("Cached %d names for mcs=%d → %s",
                    len(cached_names), mcs, cache_path)


def run_hdbscan_on_fixed_umap(min_cluster_size: int, seed: int = 42):
    """HDBSCAN on the cached UMAP projection. Returns (assignments, centroids)."""
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.0,
        core_dist_n_jobs=-1,
    )
    assignments = clusterer.fit_predict(STATE.umap_emb)
    unique = set(assignments.tolist())
    sorted_ids = sorted(c for c in unique if c != -1)
    if not sorted_ids:
        return np.full_like(assignments, -1), np.zeros((0, STATE.embeddings.shape[1]))
    remap = {old: new for new, old in enumerate(sorted_ids)}
    new_assignments = np.full_like(assignments, -1)
    centroids = np.zeros((len(sorted_ids), STATE.embeddings.shape[1]),
                         dtype=STATE.embeddings.dtype)
    for old, new in remap.items():
        mask = assignments == old
        new_assignments[mask] = new
        centroids[new] = STATE.embeddings[mask].mean(axis=0)
    return new_assignments, centroids


def build_weighted_features(cfg_weights: Dict[str, float]) -> Tuple[List[str], List[float]]:
    """Rebuild feature→weight mapping from cached extractions with trial weight matrix."""
    tmp_cfg = LocalExplanationConfig(
        approach="star_local",
        weight_correct_winning=cfg_weights["wcw"],
        weight_correct_losing=cfg_weights["wcl"],
        weight_incorrect_winning=cfg_weights["wiw"],
        weight_incorrect_losing=cfg_weights["wil"],
    )
    wfs = compute_star_weights(
        STATE.star_results, STATE.true_labels,
        STATE.task_metadata.positive_label, tmp_cfg,
    )
    # Dedup + aggregate
    weights_by_feat: Dict[str, float] = defaultdict(float)
    for feat, w in wfs:
        feat = feat.strip()
        if feat:
            weights_by_feat[feat] += w
    # Align with cached feature order (embeddings are indexed by STATE.features)
    weights_aligned = np.array([weights_by_feat.get(f, 0.0) for f in STATE.features])
    return weights_aligned


def name_clusters_cached(assignments: np.ndarray, cluster_ids: List[int],
                         weights: np.ndarray, samples_per_cluster: int) -> Dict[int, Dict[str, str]]:
    """Name clusters with content-hash caching."""
    # For each cluster, sample members deterministically (weighted) → compute hash
    import random
    rng = random.Random(42)
    out: Dict[int, Dict[str, str]] = {}
    to_fetch = {}  # cid -> content hash -> sampled features
    for cid in cluster_ids:
        member_idx = np.where(assignments == cid)[0]
        if len(member_idx) == 0:
            continue
        member_w = np.abs(weights[member_idx])
        total_w = member_w.sum() + 1e-8
        probs = member_w / total_w
        n = min(samples_per_cluster, len(member_idx))
        picks = rng.choices(member_idx.tolist(), weights=probs.tolist(), k=n)
        sampled_texts = sorted([STATE.features[i] for i in picks])
        h = _hash(sampled_texts)
        if h in STATE.naming_cache:
            out[cid] = STATE.naming_cache[h]
        else:
            to_fetch[cid] = (h, sampled_texts)

    if to_fetch:
        from methods.local_explanations.prompts import render_cluster_naming_prompt
        from methods.local_explanations.parsing import parse_json_dict
        for cid, (h, sampled_texts) in to_fetch.items():
            prompt = render_cluster_naming_prompt(sampled_texts)
            response = STATE.generate_fn(prompt)
            parsed = parse_json_dict(response)
            name_info = {
                "name": parsed.get("name", f"cluster_{cid}"),
                "description": parsed.get("description", ""),
            }
            STATE.naming_cache[h] = name_info
            _save_jsonl_append(
                STATE.output_dir / "caches" / "naming_cache.jsonl",
                {"key": h, "value": name_info},
            )
            out[cid] = name_info
    return out


def objective(trial) -> float:
    """Single Optuna trial. Returns eval AUC."""
    t0 = datetime.now()

    # --- Sample hyperparameters ---
    wcw = trial.suggest_float("wcw", 0.25, 2.0)
    wcl = trial.suggest_float("wcl", -0.5, 0.5)
    wiw = trial.suggest_float("wiw", -2.0, 0.0)
    wil = trial.suggest_float("wil", 0.0, 1.0)
    mcs = trial.suggest_categorical("min_cluster_size", [30, 50, 100, 200])
    top_n = trial.suggest_categorical("top_n_candidates", [50, 100, 200, 400])
    top_k = trial.suggest_categorical("top_k_families", [10, 14, 20])
    shrink_choice = trial.suggest_categorical("shrinkage_alpha", ["0", "1", "5", "10", "None"])
    shrinkage_alpha = None if shrink_choice == "None" else float(shrink_choice)
    l1_C = trial.suggest_float("l1_C", 1e-3, 10.0, log=True)
    class_weight_choice = trial.suggest_categorical("class_weight", ["balanced", "none"])
    class_weight = None if class_weight_choice == "none" else "balanced"

    params = {
        "wcw": wcw, "wcl": wcl, "wiw": wiw, "wil": wil,
        "min_cluster_size": mcs, "top_n": top_n, "top_k": top_k,
        "shrinkage_alpha": shrink_choice, "l1_C": l1_C,
        "class_weight": class_weight_choice,
    }
    trial_dir = STATE.output_dir / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    with open(trial_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    # --- Weight feature pool ---
    weights_aligned = build_weighted_features({"wcw": wcw, "wcl": wcl, "wiw": wiw, "wil": wil})

    # --- Use pre-cached HDBSCAN assignments + pre-named clusters ---
    assignments = STATE.assignments_by_mcs[mcs]
    centroids = STATE.centroids_by_mcs[mcs]
    names_for_mcs = STATE.names_by_mcs[mcs]
    n_clusters = centroids.shape[0]
    if n_clusters < top_k:
        logger.info("Trial %d: only %d clusters, < top_k=%d — pruning",
                    trial.number, n_clusters, top_k)
        raise optuna.TrialPruned()

    # --- Rank clusters by shrunk mean ---
    cluster_stats = []
    for cid in range(n_clusters):
        mask = assignments == cid
        member_idx = np.where(mask)[0]
        if len(member_idx) == 0:
            continue
        member_w = weights_aligned[member_idx]
        agg = float(member_w.sum())
        count = len(member_idx)
        rs = agg if shrinkage_alpha is None else agg / (count + shrinkage_alpha)
        cluster_stats.append({
            "cluster_id": cid, "members_idx": member_idx,
            "member_weights": member_w.tolist(),
            "aggregate_weight": agg, "ranking_score": rs, "count": count,
        })
    cluster_stats.sort(key=lambda s: s["ranking_score"], reverse=True)
    top_candidates = cluster_stats[:top_n]
    top_ids = [s["cluster_id"] for s in top_candidates]

    # --- Look up pre-computed names (NO LLM call in the hot path) ---
    named_for_judge = []
    for s in top_candidates:
        cid = s["cluster_id"]
        info = names_for_mcs.get(cid, {"name": f"cluster_{cid}", "description": ""})
        named_for_judge.append({
            "cluster_id": cid,
            "name": info["name"],
            "description": info["description"],
            "ranking_score": s["ranking_score"],
            "aggregate_weight": s["aggregate_weight"],
            "count": s["count"],
            "member_features": [STATE.features[i] for i in s["members_idx"][:20]],
            "centroid": centroids[cid].tolist(),
            "mean_p_positive": float(STATE.feature_label_props[s["members_idx"]].mean()),
        })

    # --- Dedup judge + top-K family selection ---
    try:
        groups = run_dedup_judge(named_for_judge, STATE.task_metadata.text_type, STATE.generate_fn)
    except Exception as exc:
        logger.warning("Trial %d: dedup judge failed (%s) — pruning", trial.number, exc)
        raise optuna.TrialPruned()
    top_families = select_top_families(named_for_judge, groups, top_k)

    if len(top_families) < 2:
        logger.info("Trial %d: dedup produced %d families < 2 — pruning",
                    trial.number, len(top_families))
        raise optuna.TrialPruned()

    # --- Generate rubrics (with caching by family content) ---
    from types import SimpleNamespace
    canonical_stubs = []
    for fam in top_families:
        rep = fam["rep"]
        merged_members = []
        merged_weight = 0.0
        for m in fam["members"]:
            merged_members.extend(m["member_features"])
            merged_weight += m["aggregate_weight"]
        fam_hash = _hash({"name": rep["name"], "members": sorted(merged_members[:20])})
        canonical_stubs.append(SimpleNamespace(
            cluster_id=rep["cluster_id"],
            name=rep["name"],
            description=rep["description"],
            member_features=merged_members[:50],
            member_weights=[],
            aggregate_weight=merged_weight,
            ranking_score=fam["family_score"],
            centroid=np.array(rep["centroid"]),
            _fam_hash=fam_hash,
        ))

    # Short-circuit rubric gen via cache where possible
    uncached_stubs = []
    cached_canonical = []
    for stub in canonical_stubs:
        if stub._fam_hash in STATE.rubric_cache:
            cached = STATE.rubric_cache[stub._fam_hash]
            from methods.local_explanations.rubric_generation import CanonicalFeature
            cached_canonical.append(CanonicalFeature(
                feature_id=cached["feature_id"],
                name=stub.name,
                description=stub.description,
                rubric=cached["rubric"],
                aggregate_weight=stub.aggregate_weight,
                cluster_size=len(stub.member_features),
                member_features=stub.member_features,
            ))
        else:
            uncached_stubs.append(stub)

    if uncached_stubs:
        try:
            new_canonical = generate_rubrics(
                uncached_stubs,
                text_type=STATE.task_metadata.text_type,
                generate_fn=STATE.generate_fn,
                top_k=len(uncached_stubs),
            )
        except Exception as exc:
            logger.warning("Trial %d: rubric gen failed (%s) — pruning", trial.number, exc)
            raise optuna.TrialPruned()
        for stub, cf in zip(uncached_stubs, new_canonical):
            STATE.rubric_cache[stub._fam_hash] = {
                "feature_id": cf.feature_id, "rubric": cf.rubric,
            }
            _save_jsonl_append(
                STATE.output_dir / "caches" / "rubric_cache.jsonl",
                {"key": stub._fam_hash, "value": {"feature_id": cf.feature_id, "rubric": cf.rubric}},
            )
        canonical = cached_canonical + list(new_canonical)
    else:
        canonical = cached_canonical

    if len(canonical) < 2:
        raise optuna.TrialPruned()

    # --- Score train/eval/test subsamples via Llama-70B (cached) ---
    scored_train = score_examples_on_vocabulary(
        STATE.train_sample, canonical, STATE.scoring_backend, STATE.task_description,
        id_column=STATE.dataset_cfg["id_column"],
        text_column=STATE.dataset_cfg["text_column"],
        label_column=STATE.dataset_cfg["label_column"],
        label_cache_dir=str(STATE.output_dir / "score_cache"),
        tokenizer=STATE.tokenizer, max_model_len=STATE.max_model_len,
        stage=f"train_trial{trial.number}",
    )
    scored_eval = score_examples_on_vocabulary(
        STATE.eval_sample, canonical, STATE.scoring_backend, STATE.task_description,
        id_column=STATE.dataset_cfg["id_column"],
        text_column=STATE.dataset_cfg["text_column"],
        label_column=STATE.dataset_cfg["label_column"],
        label_cache_dir=str(STATE.output_dir / "score_cache"),
        tokenizer=STATE.tokenizer, max_model_len=STATE.max_model_len,
        stage=f"eval_trial{trial.number}",
    )

    feat_cols = [c for c in scored_train.columns if c not in (
        STATE.dataset_cfg["id_column"], STATE.dataset_cfg["label_column"],
        STATE.dataset_cfg["text_column"],
    )]
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    clf = LogisticRegression(
        penalty="l1", solver="liblinear", C=l1_C,
        class_weight=class_weight, max_iter=500,
    )
    clf.fit(
        scored_train[feat_cols].values,
        scored_train[STATE.dataset_cfg["label_column"]].values,
    )
    eval_probs = clf.predict_proba(scored_eval[feat_cols].values)[:, 1]
    eval_auc = float(roc_auc_score(
        scored_eval[STATE.dataset_cfg["label_column"]].values, eval_probs,
    ))

    # Also compute train AUC as a diagnostic
    train_probs = clf.predict_proba(scored_train[feat_cols].values)[:, 1]
    train_auc = float(roc_auc_score(
        scored_train[STATE.dataset_cfg["label_column"]].values, train_probs,
    ))

    elapsed = (datetime.now() - t0).total_seconds()
    logger.info(
        "Trial %d: eval_auc=%.4f train_auc=%.4f | K=%d top_n=%d top_k=%d "
        "mcs=%d wcw=%.2f wcl=%.2f wiw=%.2f wil=%.2f C=%.4g cw=%s | %.1fs",
        trial.number, eval_auc, train_auc, n_clusters, top_n, top_k,
        mcs, wcw, wcl, wiw, wil, l1_C, class_weight_choice, elapsed,
    )
    # Persist per-trial metrics
    with open(trial_dir / "result.json", "w") as f:
        json.dump({
            "params": params, "eval_auc": eval_auc, "train_auc": train_auc,
            "n_clusters": n_clusters, "n_canonical": len(canonical),
            "elapsed_sec": elapsed,
            "canonical_names": [cf.name for cf in canonical],
        }, f, indent=2)
    return eval_auc


# -----------------------------------------------------------------------------
# Setup + main loop
# -----------------------------------------------------------------------------

def setup_state(args):
    dcfg = dict(DATASET_CONFIGS[args.dataset])
    STATE.dataset_cfg = dcfg
    STATE.task_metadata = TASK_METADATA_REGISTRY[dcfg["dataset_name"]]
    STATE.task_description = get_task_description(dcfg["dataset_name"])

    train_df, eval_df, test_df = load_splits(dcfg["split_dir"])
    id_col = dcfg["id_column"]
    if id_col not in train_df.columns:
        for df, tag in [(train_df, "train"), (eval_df, "eval"), (test_df, "test")]:
            df[id_col] = [f"{tag}_{i}" for i in range(len(df))]

    # Deterministic subsamples
    STATE.train_sample = train_df.sample(
        n=min(args.n_train, len(train_df)), random_state=42,
    ).reset_index(drop=True)
    STATE.eval_sample = eval_df.sample(
        n=min(args.n_eval, len(eval_df)), random_state=42,
    ).reset_index(drop=True)
    STATE.test_sample = test_df.sample(
        n=min(args.n_test, len(test_df)), random_state=42,
    ).reset_index(drop=True)
    logger.info("Subsamples: train=%d, eval=%d, test=%d",
                len(STATE.train_sample), len(STATE.eval_sample), len(STATE.test_sample))

    # Load cached extractions
    explainer = LocalExplainer(
        LocalExplanationConfig(output_dir=args.output_dir),
        scoring_backend=None,
        task_description=STATE.task_description, task_metadata=STATE.task_metadata,
        cache_dir=str(Path(args.output_dir) / "explanation_cache"),
    )
    doc_ids = train_df[id_col].astype(str).tolist()
    texts = train_df[dcfg["text_column"]].tolist()
    STATE.star_results = explainer.explain_star_local_batch(texts=texts, doc_ids=doc_ids)
    STATE.true_labels = dict(zip(
        train_df[id_col].astype(str),
        train_df[dcfg["label_column"]].astype(int),
    ))

    # Build feature list THE SAME WAY compute_star_weights does (skipping docs
    # where prediction is unparseable) so cached embeddings align. Use a dummy
    # weight config — we only care about which features get in, not weights.
    dummy_cfg = LocalExplanationConfig(approach="star_local")
    wfs_dummy = compute_star_weights(
        STATE.star_results, STATE.true_labels,
        STATE.task_metadata.positive_label, dummy_cfg,
    )
    fw_tmp = defaultdict(float)
    for feat, w in wfs_dummy:
        feat = feat.strip()
        if feat:
            fw_tmp[feat] += w
    features = list(fw_tmp.keys())
    STATE.features = features
    logger.info("Feature pool: %d unique", len(features))

    # Per-feature P(y=1)
    props = compute_feature_label_props(STATE.star_results, STATE.true_labels)
    STATE.feature_label_props = np.array([props.get(f, 0.5) for f in features])

    # Load cached embeddings; re-embed if cache mismatches
    cluster_dir = Path(args.output_dir) / "clustering"
    emb_path = cluster_dir / "feature_embeddings.npy"
    emb = None
    if emb_path.exists():
        emb = np.load(emb_path)
        if emb.shape[0] != len(features):
            logger.warning("Cached embeddings %s != %d features — recomputing",
                           emb.shape, len(features))
            emb = None
    if emb is None:
        model_dir = cluster_dir / "similarity_model"
        emb = embed_with_model(features, str(model_dir))
        np.save(emb_path, emb)
    STATE.embeddings = emb

    # VLLM backend FIRST — load Llama-70B on GPU before anything touches CUDA.
    # UMAP (below) doesn't use CUDA but pynndescent can init it and deadlock
    # VLLM's subprocess-based init. Order matters.
    from methods.autometrics.autometrics.backends import create_backend
    logger.info("Loading VLLM backend (Llama-70B) — must happen before any CUDA use")
    STATE.scoring_backend = create_backend(
        "vllm", model_name_or_path=args.model,
        tensor_parallel_size=1, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    STATE.max_model_len = args.max_model_len
    logger.info("VLLM loaded")

    # Fit UMAP ONCE (CPU)
    import umap
    logger.info("Fitting UMAP once (n_neighbors=30, n_components=20, metric=cosine)")
    reducer = umap.UMAP(
        n_neighbors=30, n_components=20, min_dist=0.0, metric="cosine",
        random_state=None, low_memory=True, verbose=True,
    )
    STATE.umap_emb = reducer.fit_transform(emb)
    logger.info("UMAP fit complete")

    # gpt-5-mini generate_fn (sync) — used per-trial for dedup judge + rubric gen
    STATE.generate_fn = create_generate_fn(args.proposer_model)

    # Load per-trial caches from disk
    rubric_cache_path = STATE.output_dir / "caches" / "rubric_cache.jsonl"
    STATE.rubric_cache = _load_jsonl_cache(rubric_cache_path)
    logger.info("Loaded rubric cache: %d", len(STATE.rubric_cache))

    # Pre-compute HDBSCAN + cluster names for all candidate min_cluster_size values.
    # One-time cost (~10-15 min) massively reduces per-trial cost later.
    min_cluster_size_options = [30, 50, 100, 200]
    logger.info("Pre-computing HDBSCAN + names for mcs ∈ %s", min_cluster_size_options)
    precompute_cluster_names_for_mcs(min_cluster_size_options)
    logger.info("Pre-computation done; per-trial hot path has zero naming LLM calls")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--input-output-dir", type=str, required=True,
                        help="Existing output dir with explanation_cache + clustering artifacts")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="New directory for sweep artifacts")
    parser.add_argument("--model", type=str,
                        default="/lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/fde04ee76a27704c88f569542ef023b57d4d0362")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--proposer-model", type=str, default="openai/gpt-5-mini")
    parser.add_argument("--n-train", type=int, default=1500)
    parser.add_argument("--n-eval", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--study-name", type=str, default="local_explanations_peer_review")
    args = parser.parse_args()

    # Alias input dir to output dir's explanation_cache etc (reuse in place)
    # Simpler: copy/symlink into output_dir if not present
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("explanation_cache", "clustering"):
        src = Path(args.input_output_dir) / sub
        dst = out / sub
        if src.exists() and not dst.exists():
            os.symlink(src.resolve(), dst)
            logger.info("Symlinked %s → %s", dst, src.resolve())
    STATE.output_dir = out

    setup_state(args)

    import optuna
    globals()["optuna"] = optuna
    storage = f"sqlite:///{out / 'study.db'}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        load_if_exists=True,
    )
    logger.info("Study: %s | trials so far: %d", args.study_name, len(study.trials))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    # Save best
    best = study.best_trial
    logger.info("BEST trial #%d: eval_auc=%.4f", best.number, best.value)
    logger.info("Best params: %s", best.params)
    with open(out / "best.json", "w") as f:
        json.dump({
            "trial_number": best.number,
            "eval_auc": best.value,
            "params": best.params,
        }, f, indent=2)


if __name__ == "__main__":
    main()
