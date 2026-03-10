from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class MetricLifecycle:
    metric_id: str
    name: str
    rubric_text: str
    born_iteration: int
    active_iterations: List[int] = field(default_factory=list)
    retired_iteration: Optional[int] = None
    coefficients: Dict[int, float] = field(default_factory=dict)
    source_pairs: List[str] = field(default_factory=list)
    marginal_contributions: Dict[int, float] = field(default_factory=dict)


class MetricLifecycleTracker:
    def __init__(self):
        self._metrics: Dict[str, MetricLifecycle] = {}

    def register_metric(
        self,
        metric_id: str,
        name: str,
        rubric_text: str,
        iteration: int,
        source_pairs: Optional[List[str]] = None,
    ) -> None:
        if metric_id not in self._metrics:
            self._metrics[metric_id] = MetricLifecycle(
                metric_id=metric_id,
                name=name,
                rubric_text=rubric_text,
                born_iteration=iteration,
            )
        if source_pairs:
            self._metrics[metric_id].source_pairs.extend(source_pairs)

    def record_coefficients(self, iteration: int, coef_map: Dict[str, float]) -> None:
        for metric_id, coef in coef_map.items():
            if metric_id not in self._metrics:
                self._metrics[metric_id] = MetricLifecycle(
                    metric_id=metric_id,
                    name=metric_id,
                    rubric_text="",
                    born_iteration=iteration,
                )
            self._metrics[metric_id].coefficients[iteration] = float(coef)

    def mark_active(self, iteration: int, active_metric_ids: List[str]) -> None:
        for metric_id in active_metric_ids:
            if metric_id in self._metrics:
                if iteration not in self._metrics[metric_id].active_iterations:
                    self._metrics[metric_id].active_iterations.append(iteration)
        for metric_id, lifecycle in self._metrics.items():
            if metric_id not in active_metric_ids and lifecycle.retired_iteration is None:
                if lifecycle.active_iterations and lifecycle.active_iterations[-1] == iteration - 1:
                    lifecycle.retired_iteration = iteration

    def record_marginal_contributions(self, iteration: int, contribs: Dict[str, float]) -> None:
        for metric_id, contrib in contribs.items():
            if metric_id in self._metrics:
                self._metrics[metric_id].marginal_contributions[iteration] = float(contrib)

    def get(self, metric_id: str) -> Optional[MetricLifecycle]:
        return self._metrics.get(metric_id)

    def to_metrics_dataframe(self) -> pd.DataFrame:
        rows = []
        for lifecycle in self._metrics.values():
            rows.append(
                {
                    "metric_id": lifecycle.metric_id,
                    "name": lifecycle.name,
                    "rubric_text": lifecycle.rubric_text,
                    "born_iteration": lifecycle.born_iteration,
                    "active_iterations": json.dumps(lifecycle.active_iterations),
                    "retired_iteration": lifecycle.retired_iteration,
                    "coefficients": json.dumps(lifecycle.coefficients),
                    "source_pairs": json.dumps(lifecycle.source_pairs),
                    "marginal_contributions": json.dumps(lifecycle.marginal_contributions),
                }
            )
        return pd.DataFrame(rows)
