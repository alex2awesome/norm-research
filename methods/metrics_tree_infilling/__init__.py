"""Metrics-Tree Infilling.

LLM feature discovery over a labeled corpus via a gap-detecting classification tree.

Given a corpus of binary-labeled items (``y in {0,1}``) and a set of explicit metrics
(code-based ``.py`` scorers and frozen LLM-judge rubrics), this method:

1. Fits a *gap-detecting* classification tree (model-based recursive partitioning, MOB)
   whose nodes hold a logistic model of the item label as a function of metric *levels*,
   split on partitioning covariates ``z``. Splits happen where the metric->label
   relationship is *unstable* across subpopulations.
2. Flags nodes where the articulated metrics fail to explain the labels and where no
   available split repairs it (a localized articulability gap).
3. Uses an LLM to invent the missing textual feature that closes each gap, distills it to a
   cheap reproducible scorer, materializes it over the whole corpus, and reinserts it.
4. Reads off the feature's *measured* generality (minimal depth) on held-out data.

See ``README.md`` for the mapping of modules to the spec sections, and the hard limitation
(missing interactions of absent features are out of scope).
"""

from .config import InfillConfig

__all__ = ["InfillConfig"]
