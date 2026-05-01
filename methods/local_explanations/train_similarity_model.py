"""Clustering step 2: Fine-tune sentence-transformer on similarity triplets.

Following github.com/alex2awesome/schema-generation (step 3).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .config import LocalExplanationConfig

logger = logging.getLogger(__name__)


def train_similarity_model(
    triplets: List[Dict[str, str]],
    config: LocalExplanationConfig,
    output_dir: str,
) -> str:
    """Fine-tune a sentence-transformer on anchor/positive/negative triplets.

    Uses MultipleNegativesRankingLoss from sentence-transformers.

    Args:
        triplets: list of {"anchor": str, "positive": str, "negative": str}
        config: configuration with model_base, epochs, batch_size
        output_dir: where to save the fine-tuned model

    Returns: path to the saved model directory.
    """
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    model_path = Path(output_dir) / "similarity_model"
    model_path.mkdir(parents=True, exist_ok=True)

    if not triplets:
        logger.warning("No triplets provided — returning base model path")
        return config.similarity_model_base

    logger.info(
        "Training similarity model: %d triplets, base=%s, epochs=%d",
        len(triplets), config.similarity_model_base, config.similarity_model_epochs,
    )

    # Load base model on GPU — the runner releases the VLLM extractor/scorer
    # before calling us so we have the full device to ourselves. sentence-
    # transformers' Trainer auto-detects CUDA and will use it here.
    model = SentenceTransformer(config.similarity_model_base)

    # Create training examples
    train_examples = [
        InputExample(texts=[t["anchor"], t["positive"], t["negative"]])
        for t in triplets
    ]

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=config.similarity_model_batch_size,
    )

    # MultipleNegativesRankingLoss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config.similarity_model_epochs,
        warmup_steps=min(100, len(train_dataloader)),
        output_path=str(model_path),
        show_progress_bar=True,
    )

    logger.info("Similarity model saved to %s", model_path)
    return str(model_path)
