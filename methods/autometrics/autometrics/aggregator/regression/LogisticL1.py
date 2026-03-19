from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegressionCV

from autometrics.aggregator.regression import Regression


class LogisticL1(Regression):
    """Sparse logistic regression with L1 regularization for metric selection."""

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        dataset=None,
        solver: str = "liblinear",
        max_iter: int = 1000,
        class_weight: Optional[str | dict] = None,
        cv: int = 5,
        **kwargs,
    ):
        model = LogisticRegressionCV(
            penalty="l1",
            solver=solver,
            Cs=10,
            cv=cv,
            max_iter=max_iter,
            class_weight=class_weight,
            scoring="roc_auc",
            refit=True,
        )

        if not name:
            name = "LogisticL1"
        if not description:
            description = f"Logistic regression with L1 regularization ({cv}-fold CV, solver={solver})"

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


class LogisticL1WithInteractions(LogisticL1):
    """LogisticL1 with pairwise interaction features.

    Automatically computes pairwise products of base metric columns
    and includes them as additional features in the regression.
    """

    def __init__(
        self,
        base_feature_names: Optional[List[str]] = None,
        interaction_feature_names: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._base_feature_names: List[str] = base_feature_names or []
        # If interaction_feature_names is provided, only generate interactions
        # among those names (a subset of base_feature_names). This prevents
        # quadratic feature explosion when new candidate metrics are added
        # alongside established active metrics.
        interact_names = (
            interaction_feature_names
            if interaction_feature_names is not None
            else self._base_feature_names
        )
        # (interaction_col_name, base_col_a, base_col_b)
        self._interaction_pairs: List[Tuple[str, str, str]] = []
        for i, col_a in enumerate(interact_names):
            for j in range(i + 1, len(interact_names)):
                col_b = interact_names[j]
                interaction_col = f"{col_a} × {col_b}"
                self._interaction_pairs.append((interaction_col, col_a, col_b))

    @property
    def interaction_columns(self) -> List[str]:
        return [p[0] for p in self._interaction_pairs]

    @property
    def interaction_pairs(self) -> List[Tuple[str, str, str]]:
        return list(self._interaction_pairs)

    def get_input_columns(self):
        base = super().get_input_columns()
        return base + self.interaction_columns

    def _ensure_interaction_columns(self, df):
        """Add interaction product columns to df in-place."""
        for col_name, col_a, col_b in self._interaction_pairs:
            if col_name not in df.columns and col_a in df.columns and col_b in df.columns:
                df[col_name] = df[col_a] * df[col_b]

    def learn(self, dataset, target_column=None):
        # ensure_dependencies may replace the dataframe, so call it first
        # then add interaction columns to the (possibly new) dataframe.
        self.ensure_dependencies(dataset)
        df = dataset.get_dataframe()
        self._ensure_interaction_columns(df)
        dataset.set_dataframe(df)
        super().learn(dataset, target_column)

    def predict_proba(self, dataset):
        df = dataset.get_dataframe()
        self._ensure_interaction_columns(df)
        dataset.set_dataframe(df)
        return super().predict_proba(dataset)

    def get_interaction_coef_map(
        self,
        metric_specs,
    ) -> Dict[str, Tuple[str, str, float]]:
        """Extract interaction coefficients after fitting.

        Returns dict mapping interaction column name to
        (metric_a_name, metric_b_name, coefficient).
        """
        coef = getattr(self.model, "coef_", None)
        if coef is None:
            return {}
        coef_vec = np.array(coef).reshape(-1)
        n_base = len(self._base_feature_names)
        result: Dict[str, Tuple[str, str, float]] = {}
        for idx, (col_name, col_a, col_b) in enumerate(self._interaction_pairs):
            coef_idx = n_base + idx
            if coef_idx < len(coef_vec):
                c = float(coef_vec[coef_idx])
                if abs(c) > 1e-6:
                    result[col_name] = (col_a, col_b, c)
        return result
