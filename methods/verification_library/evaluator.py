"""Evaluate programs on held-out test set."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .combiner import ProgramCombiner
from .program_store import GeneratedProgram
from .sandbox import execute_programs_batch

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    round_idx: int
    n_programs_total: int
    n_programs_successful: int
    individual_accuracies: List[Tuple[str, float]] = field(default_factory=list)
    individual_correlations: List[Tuple[str, float]] = field(default_factory=list)
    ensemble_accuracy: float = 0.0
    ensemble_auc: float = 0.0
    ensemble_f1: float = 0.0
    best_single_accuracy: float = 0.0
    best_single_program_id: str = ""


class ProgramEvaluator:
    def __init__(
        self,
        test_texts: List[str],
        test_labels: np.ndarray,
        sandbox_timeout: float = 10.0,
    ):
        self.test_texts = test_texts
        self.test_labels = test_labels
        self.sandbox_timeout = sandbox_timeout

    def evaluate_program(
        self,
        program: GeneratedProgram,
        library_code: Optional[str] = None,
    ) -> Dict[str, float]:
        """Run one program on the test set, return metrics."""
        if not program.execution_success or not program.source_code:
            return {"accuracy": 0.0, "correlation": 0.0, "success": False}

        results = execute_programs_batch(
            program.source_code,
            self.test_texts,
            timeout=self.sandbox_timeout,
            library_code=library_code,
        )

        scores = []
        for score, err in results:
            scores.append(score if score is not None else np.nan)
        scores = np.array(scores)

        valid_mask = ~np.isnan(scores)
        if valid_mask.sum() < 10:
            return {"accuracy": 0.0, "correlation": 0.0, "success": False}

        preds = (scores[valid_mask] > 0.5).astype(int)
        labels = self.test_labels[valid_mask]
        accuracy = (preds == labels).mean()

        # Pearson correlation
        if np.std(scores[valid_mask]) > 0:
            correlation = float(np.corrcoef(scores[valid_mask], labels.astype(float))[0, 1])
        else:
            correlation = 0.0

        return {
            "accuracy": float(accuracy),
            "correlation": correlation,
            "valid_fraction": float(valid_mask.mean()),
            "success": True,
        }

    def evaluate_all(
        self,
        programs: List[GeneratedProgram],
        library_code: Optional[str] = None,
        round_idx: int = 0,
        train_texts: Optional[List[str]] = None,
        train_labels: Optional[np.ndarray] = None,
        combination_method: str = "logistic",
    ) -> EvaluationResult:
        """Evaluate all programs and compute ensemble prediction."""
        successful = [p for p in programs if p.execution_success and p.source_code]
        if not successful:
            return EvaluationResult(
                round_idx=round_idx,
                n_programs_total=len(programs),
                n_programs_successful=0,
            )

        logger.info("Evaluating %d programs on %d test examples", len(successful), len(self.test_texts))

        # Collect all program outputs on test set
        test_scores = np.full((len(self.test_texts), len(successful)), np.nan)
        individual_accs = []
        individual_corrs = []

        for j, prog in enumerate(successful):
            results = execute_programs_batch(
                prog.source_code,
                self.test_texts,
                timeout=self.sandbox_timeout,
                library_code=library_code,
            )
            for i, (score, err) in enumerate(results):
                if score is not None:
                    test_scores[i, j] = score

            valid = ~np.isnan(test_scores[:, j])
            if valid.sum() >= 10:
                preds = (test_scores[valid, j] > 0.5).astype(int)
                acc = float((preds == self.test_labels[valid]).mean())
                s = test_scores[valid, j]
                corr = float(np.corrcoef(s, self.test_labels[valid].astype(float))[0, 1]) if np.std(s) > 0 else 0.0
            else:
                acc, corr = 0.0, 0.0
            individual_accs.append((prog.program_id, acc))
            individual_corrs.append((prog.program_id, corr))

        # Ensemble combination
        combiner = ProgramCombiner(method=combination_method)

        if combination_method == "logistic" and train_texts is not None and train_labels is not None:
            # Fit combiner on training data
            train_scores = np.full((len(train_texts), len(successful)), np.nan)
            for j, prog in enumerate(successful):
                results = execute_programs_batch(
                    prog.source_code,
                    train_texts,
                    timeout=self.sandbox_timeout,
                    library_code=library_code,
                )
                for i, (score, err) in enumerate(results):
                    if score is not None:
                        train_scores[i, j] = score
            combiner.fit(train_scores, train_labels)

        ensemble_scores = combiner.predict(test_scores)
        ensemble_preds = (ensemble_scores > 0.5).astype(int)
        ensemble_acc = float((ensemble_preds == self.test_labels).mean())

        # AUC
        try:
            from sklearn.metrics import f1_score, roc_auc_score
            ensemble_auc = float(roc_auc_score(self.test_labels, ensemble_scores))
            ensemble_f1 = float(f1_score(self.test_labels, ensemble_preds))
        except Exception:
            ensemble_auc, ensemble_f1 = 0.0, 0.0

        best_acc = max(individual_accs, key=lambda x: x[1])

        return EvaluationResult(
            round_idx=round_idx,
            n_programs_total=len(programs),
            n_programs_successful=len(successful),
            individual_accuracies=individual_accs,
            individual_correlations=individual_corrs,
            ensemble_accuracy=ensemble_acc,
            ensemble_auc=ensemble_auc,
            ensemble_f1=ensemble_f1,
            best_single_accuracy=best_acc[1],
            best_single_program_id=best_acc[0],
        )
