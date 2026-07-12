"""Code-native, within-paper scientific claim verification.

This package is additive.  The frozen ``cv1``/``cv2``/``cv3`` programs remain the
historical decomposition; v2 treats that decomposition as a retrospectively seeded,
selected pipeline decision and makes its executable witnesses auditable.
"""

from .core import verify_document

__all__ = ["verify_document"]
