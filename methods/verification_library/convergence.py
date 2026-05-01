"""Library convergence and stability metrics."""

import ast
import difflib
import logging
from dataclasses import dataclass
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceMetrics:
    round_idx: int
    library_size_lines: int
    library_size_functions: int
    edit_distance_from_previous: float
    function_names_added: List[str]
    function_names_removed: List[str]
    function_overlap_jaccard: float
    ensemble_accuracy: float
    ensemble_auc: float
    n_programs_total: int
    n_programs_successful: int
    marginal_accuracy_gain: float

    def to_dict(self) -> dict:
        return {
            "library_size_lines": self.library_size_lines,
            "library_size_functions": self.library_size_functions,
            "edit_distance_from_previous": self.edit_distance_from_previous,
            "function_names_added": self.function_names_added,
            "function_names_removed": self.function_names_removed,
            "function_overlap_jaccard": self.function_overlap_jaccard,
            "ensemble_accuracy": self.ensemble_accuracy,
            "ensemble_auc": self.ensemble_auc,
            "n_programs_total": self.n_programs_total,
            "n_programs_successful": self.n_programs_successful,
            "marginal_accuracy_gain": self.marginal_accuracy_gain,
        }


def compute_edit_distance(lib_a: str, lib_b: str) -> float:
    """Normalized line-level edit distance between two library versions."""
    if not lib_a and not lib_b:
        return 0.0
    if not lib_a or not lib_b:
        return 1.0
    lines_a = lib_a.splitlines()
    lines_b = lib_b.splitlines()
    ratio = difflib.SequenceMatcher(None, lines_a, lines_b).ratio()
    return 1.0 - ratio


def extract_function_names(library_code: str) -> Set[str]:
    """Parse function names from library source."""
    try:
        tree = ast.parse(library_code)
        return {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
    except SyntaxError:
        return set()


def compute_convergence_metrics(
    round_idx: int,
    current_library: str,
    previous_library: Optional[str],
    ensemble_accuracy: float,
    ensemble_auc: float,
    previous_accuracy: float,
    n_programs_total: int,
    n_programs_successful: int,
) -> ConvergenceMetrics:
    """Compute all convergence metrics for one refactoring round."""
    current_funcs = extract_function_names(current_library)
    previous_funcs = extract_function_names(previous_library or "")

    added = sorted(current_funcs - previous_funcs)
    removed = sorted(previous_funcs - current_funcs)

    union = current_funcs | previous_funcs
    intersection = current_funcs & previous_funcs
    jaccard = len(intersection) / len(union) if union else 1.0

    edit_dist = compute_edit_distance(previous_library or "", current_library)

    return ConvergenceMetrics(
        round_idx=round_idx,
        library_size_lines=current_library.count("\n") + 1,
        library_size_functions=len(current_funcs),
        edit_distance_from_previous=edit_dist,
        function_names_added=added,
        function_names_removed=removed,
        function_overlap_jaccard=jaccard,
        ensemble_accuracy=ensemble_accuracy,
        ensemble_auc=ensemble_auc,
        n_programs_total=n_programs_total,
        n_programs_successful=n_programs_successful,
        marginal_accuracy_gain=ensemble_accuracy - previous_accuracy,
    )


def check_convergence(
    metrics_history: List[ConvergenceMetrics],
    window: int = 3,
    edit_threshold: float = 0.05,
    accuracy_threshold: float = 0.005,
) -> bool:
    """Has the library converged?

    True if the last `window` rounds all have small edit distances
    AND accuracy has not improved meaningfully.
    Requires at least one round with a non-empty library.
    """
    if len(metrics_history) < window:
        return False
    # Don't declare convergence if library is empty
    if all(m.library_size_functions == 0 for m in metrics_history):
        return False
    recent = metrics_history[-window:]
    # Need at least some library content in the window
    if any(m.library_size_functions == 0 for m in recent):
        return False
    edits_stable = all(m.edit_distance_from_previous < edit_threshold for m in recent)
    accuracy_stable = all(abs(m.marginal_accuracy_gain) < accuracy_threshold for m in recent)
    return edits_stable and accuracy_stable
