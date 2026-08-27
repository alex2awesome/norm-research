"""Model-based recursive partitioning (MOB) engine.

A pure-Python port of the gap-detecting tree from R ``partykit::glmtree``:

- ``mfluctuation`` — the generalized M-fluctuation parameter-instability test
  (Zeileis & Hornik 2007). Statistics (sup-LM for numeric ``z``, chi-squared for
  categorical ``z``) are kept faithful to ``strucchange``; only the p-value is computed
  from a permutation null instead of the asymptotic Brownian-bridge approximation.
- ``glmtree`` — the recursive tree: a per-node logistic GLM of the label on metric levels,
  split on the most unstable ``z`` via exhaustive cutpoint search.
"""

from .glmtree import GapTree, TreeNode
from .mfluctuation import FluctResult, fit_node_glm, score_contributions, test_node

__all__ = [
    "GapTree",
    "TreeNode",
    "FluctResult",
    "fit_node_glm",
    "score_contributions",
    "test_node",
]
