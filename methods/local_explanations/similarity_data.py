"""Clustering step 1: Create supervised similarity data from raw features.

Following the silver-label approach from github.com/alex2awesome/schema-generation (step 2).
"""

import asyncio
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import LocalExplanationConfig
from .parsing import parse_similarity_labels
from .prompts import render_similarity_prompt

logger = logging.getLogger(__name__)


def _embed_features(features: List[str], model_name: str) -> np.ndarray:
    """Embed feature strings using sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(features, show_progress_bar=True, batch_size=256)
    return np.array(embeddings)


def find_candidate_pairs(
    features: List[str],
    embeddings: np.ndarray,
    sim_low: float = 0.3,
    sim_high: float = 0.9,
    max_pairs: int = 50000,
    seed: int = 42,
    polarities: Optional[List[str]] = None,
) -> List[Tuple[int, int]]:
    """Find feature pairs with cosine similarity in [sim_low, sim_high].

    These are candidate pairs: similar enough to potentially be the same concept,
    but not so similar that it's obvious.

    If `polarities` is provided (one of "for", "against", "both" per feature),
    cross-polarity pairs (one strictly "for", one strictly "against") are skipped:
    a feature arguing FOR positive and a feature arguing AGAINST positive cannot
    be the same concept by construction, so we never even ask the LLM about them.
    "both" features (text seen on both sides in different docs) pair freely.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normed = embeddings / norms

    n = len(features)
    pairs = []
    block_size = 1000
    n_skipped_polarity = 0

    def _is_cross_polarity(p_i: str, p_j: str) -> bool:
        return (p_i == "for" and p_j == "against") or (p_i == "against" and p_j == "for")

    for i_start in range(0, n, block_size):
        i_end = min(i_start + block_size, n)
        block = normed[i_start:i_end]
        sims = block @ normed.T  # (block_size, n)

        for local_i in range(sims.shape[0]):
            global_i = i_start + local_i
            for j in range(global_i + 1, n):
                s = sims[local_i, j]
                if sim_low <= s <= sim_high:
                    if polarities is not None and _is_cross_polarity(polarities[global_i], polarities[j]):
                        n_skipped_polarity += 1
                        continue
                    pairs.append((global_i, j))

        if len(pairs) > max_pairs * 3:
            break

    if len(pairs) > max_pairs:
        rng = random.Random(seed)
        pairs = rng.sample(pairs, max_pairs)

    logger.info(
        "Found %d candidate pairs (sim in [%.2f, %.2f]) from %d features"
        + (f"; skipped {n_skipped_polarity} cross-polarity pairs" if polarities is not None else ""),
        len(pairs), sim_low, sim_high, n,
    )
    return pairs


_REASONING_MODEL_TAGS = ("gpt-5", "gpt5", "o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    return any(tag in model.lower() for tag in _REASONING_MODEL_TAGS)


async def _label_one_pair_async(
    client,
    pair_idx: int,
    feat_a_idx: int,
    feat_b_idx: int,
    feat_a_text: str,
    feat_b_text: str,
    polarity: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> Tuple[int, Tuple[int, int, bool]]:
    """Label a single pair via OpenAI AsyncClient. One pair per prompt to avoid
    JSON-batch confusion that destroys label quality at scale.

    feat_a_text / feat_b_text: the text to send to the LLM (typically the
    em-dash TAIL of the feature, not the full feature, so the labeler judges
    the abstract evaluative concept rather than per-paper specifics).
    polarity: "for" or "against" — selects the polarity-conditioned prompt.
    """
    from .prompts import render_similarity_prompt_single

    async with semaphore:
        prompt = render_similarity_prompt_single(feat_a_text, feat_b_text, polarity=polarity)
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if _is_reasoning_model(model):
            # GPT-5 family rejects temperature != 1.0 and requires generous max_tokens.
            # reasoning_effort left at API default ("medium" for gpt-5-mini chat completions).
            kwargs["temperature"] = 1.0
            kwargs["max_completion_tokens"] = 16000
        else:
            kwargs["temperature"] = 0
            kwargs["max_tokens"] = 32
        try:
            resp = await client.chat.completions.create(**kwargs)
            response_text = (resp.choices[0].message.content or "").strip().lower()
        except Exception as exc:
            logger.warning("Pair %d failed: %s — treating as dissimilar", pair_idx, exc)
            response_text = ""
        # First word must be yes/no; default to no (conservative)
        is_similar = response_text.startswith("yes")
        return pair_idx, (feat_a_idx, feat_b_idx, is_similar)


async def _label_all_async(
    features_for_labeling: List[str],
    pairs: List[Tuple[int, int]],
    polarities_per_pair: List[str],
    cache_path: Path,
    model: str,
    concurrency: int,
) -> List[Tuple[int, int, bool]]:
    """Async driver: one pair per prompt, dispatch all unlabeled pairs with
    concurrency cap, append each completed result to the JSONL cache as it lands.

    features_for_labeling: text shown to the LLM per feature index (typically
    the em-dash tail, not the full feature).
    polarities_per_pair: "for" or "against" per pair, picks the prompt variant.
    """
    from openai import AsyncOpenAI

    done_pairs: Dict[int, Tuple[int, int, bool]] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done_pairs[rec["pair_idx"]] = (rec["a"], rec["b"], rec["is_similar"])
                except (json.JSONDecodeError, KeyError):
                    continue
        if done_pairs:
            logger.info("Resuming similarity labeling from cache: %d pairs already done", len(done_pairs))

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    for pair_idx, (a, b) in enumerate(pairs):
        if pair_idx in done_pairs:
            continue
        tasks.append(_label_one_pair_async(
            client, pair_idx, a, b,
            features_for_labeling[a], features_for_labeling[b],
            polarities_per_pair[pair_idx],
            model, semaphore,
        ))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    logged_since = 0
    done_count = len(done_pairs)
    n_total = len(pairs)
    with open(cache_path, "a") as cache_f:
        for coro in asyncio.as_completed(tasks):
            pair_idx, (a, b, is_sim) = await coro
            done_pairs[pair_idx] = (a, b, is_sim)
            cache_f.write(json.dumps({
                "pair_idx": pair_idx, "a": a, "b": b, "is_similar": is_sim,
            }) + "\n")
            cache_f.flush()
            done_count += 1
            logged_since += 1
            if logged_since >= 200 or done_count == n_total:
                n_yes = sum(1 for _, _, s in done_pairs.values() if s)
                logger.info(
                    "Similarity labeling: %d/%d pairs, %d positive (%.1f%%)",
                    done_count, n_total, n_yes, 100 * n_yes / max(done_count, 1),
                )
                logged_since = 0

    labeled: List[Tuple[int, int, bool]] = [done_pairs[i] for i in sorted(done_pairs)]
    n_yes = sum(1 for _, _, s in labeled if s)
    logger.info(
        "Similarity labeling complete: %d pairs, %d similar (%.1f%%)",
        len(labeled), n_yes, 100 * n_yes / max(len(labeled), 1),
    )
    return labeled


def label_pairs_with_llm(
    features_for_labeling: List[str],
    pairs: List[Tuple[int, int]],
    polarities_per_pair: List[str],
    cache_path: Optional[Path] = None,
    model: str = "gpt-5-mini",
    concurrency: int = 30,
) -> List[Tuple[int, int, bool]]:
    """Ask LLM whether each pair describes the same concept.

    features_for_labeling: text shown to the LLM (em-dash tail recommended).
    polarities_per_pair: per-pair "for" or "against" polarity, picks prompt.
    """
    if cache_path is None:
        raise ValueError("cache_path is required for resumable labeling")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY env var must be set")
    return asyncio.run(_label_all_async(
        features_for_labeling, pairs, polarities_per_pair,
        Path(cache_path), model, concurrency,
    ))


def build_triplets(
    features: List[str],
    labeled_pairs: List[Tuple[int, int, bool]],
    max_positives: int = 10,
    max_negatives_per_positive: int = 10,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Convert labeled pairs into anchor-positive-negative triplets.

    Returns: list of {"anchor": str, "positive": str, "negative": str} dicts.
    """
    rng = random.Random(seed)

    # Build adjacency: for each feature, track positives and negatives
    positives = defaultdict(set)  # feature_idx -> set of similar feature indices
    negatives = defaultdict(set)  # feature_idx -> set of dissimilar feature indices

    for a, b, is_similar in labeled_pairs:
        if is_similar:
            positives[a].add(b)
            positives[b].add(a)
        else:
            negatives[a].add(b)
            negatives[b].add(a)

    triplets = []
    for anchor_idx in positives:
        pos_list = list(positives[anchor_idx])
        neg_list = list(negatives.get(anchor_idx, set()))

        if not neg_list:
            continue

        # Subsample positives
        if len(pos_list) > max_positives:
            pos_list = rng.sample(pos_list, max_positives)

        for pos_idx in pos_list:
            # Subsample negatives per positive
            neg_sample = neg_list
            if len(neg_sample) > max_negatives_per_positive:
                neg_sample = rng.sample(neg_sample, max_negatives_per_positive)

            for neg_idx in neg_sample:
                triplets.append({
                    "anchor": features[anchor_idx],
                    "positive": features[pos_idx],
                    "negative": features[neg_idx],
                })

    rng.shuffle(triplets)
    logger.info("Built %d triplets from %d labeled pairs", len(triplets), len(labeled_pairs))
    return triplets


def _extract_tail(feature: str) -> str:
    """Split a feature on ' — ' (em-dash with spaces) and return the evaluative tail.

    Falls back to the full feature if no em-dash present.
    """
    if " — " in feature:
        return feature.split(" — ", 1)[1].strip()
    return feature.strip()


def create_similarity_data(
    features: List[str],
    config: LocalExplanationConfig,
    generate_fn=None,  # retained for signature compat; ignored when cache_path is provided
    cache_path: Optional[Path] = None,
    polarities: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Full step 1: embed → find pairs → LLM label → build triplets.

    Embedding + clustering use the EVALUATIVE TAIL of each feature (the part
    after the ' — ' em-dash separator), not the full feature. This concentrates
    the embedding space on abstract evaluative concepts rather than on per-paper
    specifics that vary arbitrarily and only confuse clustering.

    If `polarities` is provided, cross-polarity (for+against) candidate pairs
    are skipped before LLM labeling — they are wrong by construction.

    Returns: list of triplet dicts ready for fine-tuning.
    """
    logger.info("Step 1: Creating supervised similarity data from %d features", len(features))

    # Use TAILS for embedding/labeling — abstracts away per-paper specifics
    features_for_embedding = [_extract_tail(f) for f in features]
    n_with_dash = sum(1 for f in features if " — " in f)
    logger.info(
        "Using em-dash tails for embedding/labeling: %d/%d (%.1f%%) had em-dash separator",
        n_with_dash, len(features), 100 * n_with_dash / max(len(features), 1),
    )

    embeddings = _embed_features(features_for_embedding, config.embedding_model)

    pairs = find_candidate_pairs(
        features, embeddings,
        sim_low=config.similarity_pair_low,
        sim_high=config.similarity_pair_high,
        max_pairs=config.max_similarity_pairs,
        seed=config.random_seed,
        polarities=polarities,
    )

    # Per-pair polarity for prompt selection (post-filter, all pairs are intra-polarity)
    if polarities is not None:
        polarities_per_pair = [polarities[a] if polarities[a] in ("for", "against") else polarities[b] for (a, b) in pairs]
        # If both are "both", default to "for"
        polarities_per_pair = [p if p in ("for", "against") else "for" for p in polarities_per_pair]
    else:
        polarities_per_pair = ["for"] * len(pairs)

    if not pairs:
        logger.warning("No candidate pairs found! Check similarity thresholds.")
        return []

    # LLM label pairs — async, concurrent, resumable, ONE pair per prompt,
    # showing only the evaluative TAIL to the LLM.
    labeled = label_pairs_with_llm(
        features_for_embedding,  # tails, not full features
        pairs,
        polarities_per_pair,
        cache_path=cache_path,
        model=config.similarity_labeling_model,
        concurrency=config.similarity_labeling_concurrency,
    )

    # Build triplets
    triplets = build_triplets(
        features, labeled,
        max_positives=config.triplet_max_positives,
        max_negatives_per_positive=config.triplet_max_negatives_per_positive,
        seed=config.random_seed,
    )

    return triplets
