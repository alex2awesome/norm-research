from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from autometrics.aggregator.regression import Regression


class LogisticL1(Regression):
    """Sparse logistic regression with L1 regularization for metric selection."""

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        dataset=None,
        C: float = 1.0,
        solver: str = "liblinear",
        max_iter: int = 1000,
        class_weight: Optional[str | dict] = None,
        **kwargs,
    ):
        model = LogisticRegression(
            penalty="l1",
            solver=solver,
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
        )

        if not name:
            name = "LogisticL1"
        if not description:
            description = f"Logistic regression with L1 regularization (C={C}, solver={solver})"

        super().__init__(name, description, model=model, dataset=dataset, **kwargs)

    def predict_proba(self, dataset):
        """Return P(y=1) for each row, using the same scaling as training."""
        df = dataset.get_dataframe().copy()
        input_columns = self.get_input_columns()

        missing_inputs = [c for c in input_columns if c not in df.columns]
        if missing_inputs:
            self.ensure_dependencies(dataset)
            df = dataset.get_dataframe().copy()
            missing_inputs = [c for c in input_columns if c not in df.columns]
        if missing_inputs:
            raise KeyError(f"LogisticL1 input columns missing from dataset: {missing_inputs}")

        X = df[input_columns]
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        mins = X_clean.min()
        maxs = X_clean.max()
        X = X.clip(lower=mins, upper=maxs, axis=1).fillna(0)

        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)[:, 1]
        else:
            scores = self.model.decision_function(X_scaled)
            probs = 1.0 / (1.0 + np.exp(-scores))
        return probs
