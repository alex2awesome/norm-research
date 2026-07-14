"""Frozen similarity-distillation data, training, and evaluation utilities."""

# Keep package import side-effect free.  In particular this avoids importing
# the large dataset builder twice when invoked with ``python -m``.
__all__: list[str] = []
