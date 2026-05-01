"""Combine multiple program predictions into a single score."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ProgramCombiner:
    def __init__(self, method: str = "logistic"):
        self.method = method
        self._model = None
        self._fitted = False

    def fit(self, program_outputs: np.ndarray, labels: np.ndarray) -> None:
        """Fit combination weights on training data.

        Args:
            program_outputs: (n_samples, n_programs) matrix of predictions
            labels: (n_samples,) true binary labels
        """
        if self.method == "logistic":
            from sklearn.linear_model import LogisticRegression
            mask = ~np.isnan(program_outputs).any(axis=1)
            X = program_outputs[mask]
            y = labels[mask]
            if len(np.unique(y)) < 2 or X.shape[0] < 10:
                logger.warning("Insufficient data for logistic, falling back to mean")
                self.method = "mean"
                return
            self._model = LogisticRegression(
                penalty="l1", solver="saga", C=1.0, max_iter=1000
            )
            self._model.fit(X, y)
            self._fitted = True

    def predict(self, program_outputs: np.ndarray) -> np.ndarray:
        """Predict combined scores.

        Args:
            program_outputs: (n_samples, n_programs)

        Returns:
            (n_samples,) combined scores in [0, 1]
        """
        # Replace NaN with 0.5 (neutral)
        filled = np.nan_to_num(program_outputs, nan=0.5)

        if self.method == "vote":
            votes = (filled > 0.5).astype(float)
            return votes.mean(axis=1)
        elif self.method == "mean":
            return filled.mean(axis=1)
        elif self.method == "logistic" and self._fitted:
            return self._model.predict_proba(filled)[:, 1]
        else:
            return filled.mean(axis=1)

    @property
    def feature_importances(self) -> Optional[np.ndarray]:
        if self._fitted and hasattr(self._model, "coef_"):
            return self._model.coef_[0]
        return None
