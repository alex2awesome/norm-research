"""STaR loop for per-datapoint normative articulation.

Goal: train a generator to articulate community-agreeable reasons for a
binary label, without letting it learn the label-prediction shortcut.

Three anti-leakage mechanisms:
  (1) blind extraction — generator never sees the label,
  (2) rationale-only SFT loss — no verdict in the training target,
  (3) information-bottleneck judge — a frozen separate model filters
      rationales by reading rationale-only (not the artifact).

See loop.py for orchestration.
"""
