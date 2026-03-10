from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import time

from autometrics.metrics.Metric import Metric, MetricResult


class LabelCache:
    """Disk-backed cache for metric scores keyed by metric_id and example id."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._scores: Dict[str, Dict[str, float]] = {}

    def _metric_path(self, metric_id: str) -> Path:
        return self.cache_dir / f"{metric_id}.csv"

    def _load_metric(self, metric_id: str) -> Dict[str, float]:
        if metric_id in self._scores:
            return self._scores[metric_id]
        path = self._metric_path(metric_id)
        if not path.exists():
            self._scores[metric_id] = {}
            return self._scores[metric_id]
        df = pd.read_csv(path, dtype={"id": str})
        scores = {str(row["id"]): float(row["score"]) for _, row in df.iterrows()}
        self._scores[metric_id] = scores
        return scores

    def available_ids(self, metric_id: str) -> set[str]:
        scores = self._load_metric(metric_id)
        return set(scores.keys())

    def get_scores(
        self,
        metric_id: str,
        metric: Metric,
        df: pd.DataFrame,
        id_column: str,
        text_column: str,
        batch_size: int = 64,
        stats: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        log_prefix: str = "",
    ) -> List[float]:
        scores = self._load_metric(metric_id)
        ids = df[id_column].astype(str).tolist()
        missing_indices: List[int] = [i for i, _id in enumerate(ids) if _id not in scores]
        total = len(ids)
        cached = total - len(missing_indices)
        if verbose:
            prefix = f"[LabelCache]{log_prefix} " if log_prefix else "[LabelCache] "
            print(f"{prefix}total={total} cached={cached} missing={len(missing_indices)} batch_size={batch_size}")

        if missing_indices:
            missing_ids = [ids[i] for i in missing_indices]
            texts = df.iloc[missing_indices][text_column].astype(str).tolist()
            new_scores: List[float] = []
            start_time = time.time()
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                if verbose:
                    prefix = f"[LabelCache]{log_prefix} " if log_prefix else "[LabelCache] "
                    print(f"{prefix}scoring batch {start // batch_size + 1} ({start}:{start+len(chunk)})")
                chunk_scores = metric.calculate_batched(chunk, chunk)
                for val in chunk_scores:
                    if isinstance(val, MetricResult):
                        new_scores.append(float(val.score))
                    else:
                        new_scores.append(float(val))
            if verbose:
                prefix = f"[LabelCache]{log_prefix} " if log_prefix else "[LabelCache] "
                print(f"{prefix}scored {len(new_scores)} in {time.time() - start_time:.1f}s")

            for _id, score in zip(missing_ids, new_scores):
                scores[_id] = float(score)

            # Persist new rows
            path = self._metric_path(metric_id)
            new_df = pd.DataFrame({"id": missing_ids, "score": new_scores})
            if path.exists():
                new_df.to_csv(path, mode="a", header=False, index=False)
            else:
                new_df.to_csv(path, index=False)

        if stats is not None:
            stats.update({
                "total": total,
                "cached": cached,
                "missing": len(missing_indices),
                "new_scored": len(missing_indices),
            })
        return [scores[_id] for _id in ids]

    def set_scores(self, metric_id: str, ids: List[str], scores_list: List[float]) -> None:
        """Persist a batch of scores for a metric, skipping already-cached ids."""
        if not ids:
            return
        scores = self._load_metric(metric_id)
        new_ids: List[str] = []
        new_scores: List[float] = []
        for _id, score in zip(ids, scores_list):
            if _id in scores:
                continue
            scores[_id] = float(score)
            new_ids.append(_id)
            new_scores.append(float(score))
        if not new_ids:
            return
        path = self._metric_path(metric_id)
        new_df = pd.DataFrame({"id": new_ids, "score": new_scores})
        if path.exists():
            new_df.to_csv(path, mode="a", header=False, index=False)
        else:
            new_df.to_csv(path, index=False)
