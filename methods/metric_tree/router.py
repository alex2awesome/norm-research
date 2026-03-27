"""Per-node text classifier router for selective tree deepening.

Trains a small MLP classification head on frozen sentence embeddings to predict
minority-class membership within a partition. At inference, examples where
p(minority | text) > threshold continue deeper; others stop with base-rate prediction.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .example_selection import embed_texts

logger = logging.getLogger("metric_tree.router")


class NodeRouter(nn.Module):
    """Lightweight binary classifier on frozen sentence embeddings.

    Architecture: Linear(input_dim, hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim, 1)
    Trained with class-weighted BCE loss for balanced learning.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        minority_is_positive: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.minority_is_positive = minority_is_positive

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits (pre-sigmoid) of shape (n, 1)."""
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict p(minority_class | embedding) for each row.

        Args:
            embeddings: (n, input_dim) array of sentence embeddings.

        Returns:
            (n,) array of probabilities in [0, 1].
        """
        self.eval()
        device = next(self.parameters()).device
        x = torch.from_numpy(embeddings.astype(np.float32)).to(device)
        logits = self.forward(x).squeeze(-1)
        return torch.sigmoid(logits).cpu().numpy()


def train_node_router(
    texts: List[str],
    labels: np.ndarray,
    base_rate: float,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    *,
    n_epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    seed: int = 42,
    min_examples: int = 40,
) -> Optional[NodeRouter]:
    """Train a per-node router on frozen sentence embeddings.

    The router learns to predict minority-class membership: examples the
    partition's base rate would get wrong.

    Returns:
        Trained NodeRouter, or None if training is not feasible.
    """
    n = len(texts)
    if n < min_examples:
        logger.info("Too few examples (%d < %d) to train router", n, min_examples)
        return None

    n_pos = int((labels == 1).sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        logger.info("Single-class partition, skipping router")
        return None

    # Determine minority class
    minority_is_positive = base_rate < 0.5

    # Router target: 1 = minority class, 0 = majority class
    if minority_is_positive:
        targets = labels.astype(np.float32)
        n_minority, n_majority = n_pos, n_neg
    else:
        targets = 1.0 - labels.astype(np.float32)
        n_minority, n_majority = n_neg, n_pos

    # Compute frozen embeddings
    embeddings = embed_texts(texts, model_name=embedding_model_name)
    if embeddings is None:
        logger.warning("Failed to compute embeddings, skipping router")
        return None

    input_dim = embeddings.shape[1]

    # Set up torch training
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    router = NodeRouter(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        minority_is_positive=minority_is_positive,
    ).to(device)

    X = torch.from_numpy(embeddings.astype(np.float32)).to(device)
    y = torch.from_numpy(targets).to(device)

    # Class-weighted BCE: weight minority class higher
    pos_weight = torch.tensor([n_majority / max(n_minority, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(router.parameters(), lr=learning_rate)

    # Training loop
    router.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            logits = router(X[idx]).squeeze(-1)
            loss = criterion(logits, y[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

    avg_loss = epoch_loss / n
    logger.info("Router trained: %d examples, %d epochs, final_loss=%.4f, "
                "minority=%s (n_min=%d, n_maj=%d, pos_weight=%.2f)",
                n, n_epochs, avg_loss,
                "positive" if minority_is_positive else "negative",
                n_minority, n_majority, pos_weight.item())

    # Quick training accuracy check
    router.eval()
    train_probs = router.predict_proba(embeddings)
    train_preds = (train_probs > 0.5).astype(int)
    acc = float((train_preds == targets).mean())
    minority_recall = float(train_preds[targets == 1].mean()) if (targets == 1).any() else 0.0
    logger.info("Router train accuracy=%.3f, minority_recall=%.3f", acc, minority_recall)

    # Move to CPU for storage (save GPU memory)
    router.cpu()
    return router


def predict_router(
    router: NodeRouter,
    texts: List[str],
    embedding_model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Predict p(minority_class | text) for a batch of texts.

    Returns:
        (n,) array of probabilities. Higher = more likely to be minority class
        (i.e., the base-rate prediction would be wrong).
    """
    embeddings = embed_texts(texts, model_name=embedding_model_name)
    if embeddings is None:
        logger.warning("Embedding failed, returning 0.5 for all examples")
        return np.full(len(texts), 0.5)

    return router.predict_proba(embeddings.astype(np.float32))


def save_router(router: NodeRouter, path: str) -> None:
    """Save router weights and metadata."""
    torch.save({
        "state_dict": router.state_dict(),
        "input_dim": router.input_dim,
        "hidden_dim": router.hidden_dim,
        "dropout": router.dropout_rate,
        "minority_is_positive": router.minority_is_positive,
    }, path)
    logger.info("Router saved to %s", path)


def load_router(path: str) -> NodeRouter:
    """Load router from saved checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    router = NodeRouter(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint["dropout"],
        minority_is_positive=checkpoint["minority_is_positive"],
    )
    router.load_state_dict(checkpoint["state_dict"])
    router.eval()
    return router
