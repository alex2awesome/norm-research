"""Gated Interaction MLP for sparse feature and interaction selection.

Uses group lasso (L2,1 norm) on the input projection layer to zero out entire
features and interactions. The first layer projects features and their pairwise
interactions through separate weight matrices; group lasso on each input
column drives unused features/interactions to exactly zero. An MLP head
provides nonlinear classification capacity.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from autometrics.aggregator.regression import Regression

log = logging.getLogger(__name__)


class _GroupLassoInteractionNet(nn.Module):
    """Features + pairwise interactions with group lasso sparsity."""

    def __init__(self, n_features: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        n_interactions = n_features * (n_features - 1) // 2
        self.n_interactions = n_interactions

        # Separate projections for features and interactions
        # Group lasso on columns of these weight matrices drives
        # unused features/interactions to zero
        self.feature_proj = nn.Linear(n_features, hidden_dim, bias=False)
        self.interaction_proj = nn.Linear(n_interactions, hidden_dim, bias=False)

        # MLP head
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Precompute interaction pair indices
        pairs_i, pairs_j = [], []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                pairs_i.append(i)
                pairs_j.append(j)
        self.register_buffer("pairs_i", torch.tensor(pairs_i, dtype=torch.long))
        self.register_buffer("pairs_j", torch.tensor(pairs_j, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_features) -> logits: (batch, 1)"""
        # Pairwise interactions: x_i * x_j (AND for binary features)
        interactions = x[:, self.pairs_i] * x[:, self.pairs_j]

        h_feat = self.feature_proj(x)
        h_inter = self.interaction_proj(interactions)

        combined = torch.cat([h_feat, h_inter], dim=1)
        return self.head(combined)

    def feature_group_norms(self) -> torch.Tensor:
        """L2 norm of each feature's weight column in the projection layer."""
        return self.feature_proj.weight.norm(dim=0)  # (n_features,)

    def interaction_group_norms(self) -> torch.Tensor:
        """L2 norm of each interaction's weight column."""
        return self.interaction_proj.weight.norm(dim=0)  # (n_interactions,)

    def group_lasso_loss(self, lambda_feature: float, lambda_interaction: float) -> torch.Tensor:
        """Group lasso (L2,1 norm): sum of L2 norms per input group."""
        return (
            lambda_feature * self.feature_group_norms().sum()
            + lambda_interaction * self.interaction_group_norms().sum()
        )


class GatedInteractionMLP(Regression):
    """Group-Lasso Interaction MLP for the autometrics Regression interface.

    Uses group lasso on the input projection to drive unused features and
    interactions to exactly zero, providing interpretable sparse selection
    with an MLP's nonlinear capacity.
    """

    def __init__(
        self,
        base_feature_names: Optional[List[str]] = None,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        n_epochs: int = 200,
        learning_rate: float = 1e-3,
        lambda_feature: float = 0.1,
        lambda_interaction: float = 0.05,
        gate_threshold: float = 0.01,
        batch_size: int = 256,
        patience: int = 20,
        name: Optional[str] = None,
        description: Optional[str] = None,
        dataset=None,
        **kwargs,
    ):
        self._sklearn_stub = _SklearnStub()

        if not name:
            name = "GatedInteractionMLP"
        if not description:
            description = "Group-lasso MLP with sparse feature and interaction selection"

        super().__init__(name, description, model=self._sklearn_stub, dataset=dataset, **kwargs)

        self._base_feature_names: List[str] = base_feature_names or []
        self._hidden_dim = hidden_dim
        self._dropout = dropout
        self._n_epochs = n_epochs
        self._lr = learning_rate
        self._lambda_feature = lambda_feature
        self._lambda_interaction = lambda_interaction
        self._gate_threshold = gate_threshold
        self._batch_size = batch_size
        self._patience = patience

        self._net: Optional[_GroupLassoInteractionNet] = None
        self._device = torch.device("cpu")
        self._interaction_pairs: List[Tuple[str, str, str]] = []

    @property
    def interaction_columns(self) -> List[str]:
        return [p[0] for p in self._interaction_pairs]

    @property
    def interaction_pairs(self) -> List[Tuple[str, str, str]]:
        return list(self._interaction_pairs)

    def learn(self, dataset, target_column=None):
        """Train the group-lasso interaction MLP."""
        self.ensure_dependencies(dataset)
        df = dataset.get_dataframe()

        input_columns = self.get_input_columns()
        if not target_column:
            target_column = dataset.get_target_columns()[0]

        X = df[input_columns].copy()
        y = df[target_column].copy()

        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = y.replace([np.inf, -np.inf], np.nan).fillna(0)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Build interaction pair names
        self._interaction_pairs = []
        names = list(input_columns)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                self._interaction_pairs.append((f"{names[i]} × {names[j]}", names[i], names[j]))

        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=self._device)
        y_t = torch.tensor(y.values, dtype=torch.float32, device=self._device).unsqueeze(1)

        n_features = X_t.shape[1]
        self._net = _GroupLassoInteractionNet(
            n_features=n_features,
            hidden_dim=self._hidden_dim,
            dropout=self._dropout,
        ).to(self._device)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self._lr, weight_decay=1e-5)
        bce = nn.BCEWithLogitsLoss()

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        self._net.train()
        n = X_t.shape[0]
        for epoch in range(self._n_epochs):
            perm = torch.randperm(n, device=self._device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, self._batch_size):
                idx = perm[start : start + self._batch_size]
                xb, yb = X_t[idx], y_t[idx]

                logits = self._net(xb)
                loss = bce(logits, yb) + self._net.group_lasso_loss(
                    self._lambda_feature, self._lambda_interaction
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if avg_loss < best_loss - 1e-5:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self._patience:
                log.info(f"GatedMLP early stopping at epoch {epoch + 1} (best loss: {best_loss:.4f})")
                break

        if best_state is not None:
            self._net.load_state_dict(best_state)

        self._net.eval()
        self._selected_columns = list(input_columns)
        self._populate_sklearn_stub()

        # Log summary
        fg = self._net.feature_group_norms().detach().cpu().numpy()
        ig = self._net.interaction_group_norms().detach().cpu().numpy()
        n_active_f = int((fg > self._gate_threshold).sum())
        n_active_i = int((ig > self._gate_threshold).sum())
        log.info(
            f"GatedMLP trained: {n_active_f}/{n_features} active features, "
            f"{n_active_i}/{self._net.n_interactions} active interactions"
        )

    def _populate_sklearn_stub(self):
        """Fill sklearn stub with group norms as pseudo-coefficients."""
        fg = self._net.feature_group_norms().detach().cpu().numpy()
        ig = self._net.interaction_group_norms().detach().cpu().numpy()

        feature_coefs = np.where(fg > self._gate_threshold, fg, 0.0)
        interaction_coefs = np.where(ig > self._gate_threshold, ig, 0.0)

        self._sklearn_stub.coef_ = np.concatenate([feature_coefs, interaction_coefs]).reshape(1, -1)
        self._sklearn_stub.intercept_ = np.array([0.0])

    def predict_proba(self, dataset) -> np.ndarray:
        """Return P(y=1) for each row."""
        df = dataset.get_dataframe().copy()
        input_columns = self.get_input_columns()

        missing = [c for c in input_columns if c not in df.columns]
        if missing:
            self.ensure_dependencies(dataset)
            df = dataset.get_dataframe().copy()

        X = df[input_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = self.scaler.transform(X) if self.scaler is not None else X.values

        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            logits = self._net(X_t).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def get_interaction_coef_map(
        self,
        metric_specs,
    ) -> Dict[str, Tuple[str, str, float]]:
        """Extract interaction group norms as pseudo-coefficients."""
        if self._net is None:
            return {}
        ig = self._net.interaction_group_norms().detach().cpu().numpy()
        result: Dict[str, Tuple[str, str, float]] = {}
        for idx, (col_name, col_a, col_b) in enumerate(self._interaction_pairs):
            norm_val = float(ig[idx])
            if norm_val > self._gate_threshold:
                result[col_name] = (col_a, col_b, norm_val)
        return result

    def get_feature_importance(self) -> Dict[str, float]:
        """Return group norm per feature (higher = more important)."""
        if self._net is None:
            return {}
        fg = self._net.feature_group_norms().detach().cpu().numpy()
        input_columns = self.get_input_columns()
        return {col: float(fg[i]) for i, col in enumerate(input_columns)}

    def get_active_features(self) -> List[str]:
        """Return feature names with group norm > threshold."""
        importance = self.get_feature_importance()
        return [name for name, val in importance.items() if val > self._gate_threshold]

    def get_active_interactions(self) -> List[Tuple[str, str, float]]:
        """Return active interaction pairs with group norms."""
        if self._net is None:
            return []
        ig = self._net.interaction_group_norms().detach().cpu().numpy()
        result = []
        for idx, (col_name, col_a, col_b) in enumerate(self._interaction_pairs):
            norm_val = float(ig[idx])
            if norm_val > self._gate_threshold:
                result.append((col_a, col_b, norm_val))
        return result

    def get_input_columns(self):
        """Return base feature columns (interactions are handled internally)."""
        cols = super().get_input_columns()
        if not cols and self._base_feature_names:
            return list(self._base_feature_names)
        return cols

    def identify_important_metrics(self):
        """Return (importance, name) pairs sorted by group norm."""
        importance = self.get_feature_importance()
        return sorted(
            [(val, name) for name, val in importance.items()],
            key=lambda x: abs(x[0]),
            reverse=True,
        )


class _SklearnStub:
    """Minimal stub mimicking sklearn model attributes for interface compat."""

    def __init__(self):
        self.coef_ = None
        self.intercept_ = np.array([0.0])

    def predict(self, X):
        raise NotImplementedError("Use GatedInteractionMLP.predict_proba() instead")

    def predict_proba(self, X):
        raise NotImplementedError("Use GatedInteractionMLP.predict_proba() instead")
