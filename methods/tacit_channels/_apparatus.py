"""Single indirection onto the frozen parts-1-2 apparatus (policy-isomorphism scorer).

tacit_channels code NEVER imports methods.codability directly - always through this module.
When the gated stage-2 move relocates methods/codability/experiments/ to
methods/tacit_channels/isomorphism/, flipping APPARATUS_ROOT below is the ONLY edit the
channels/ code needs.

All exports are lazy (PEP 562) so importing tacit_channels never pulls vLLM/torch onto a
CPU-only host.
"""
from __future__ import annotations

import importlib

# Stage-2 flip: "methods.codability.experiments" -> "methods.tacit_channels.isomorphism"
APPARATUS_ROOT = "methods.codability.experiments"

_SOURCES = {
    # exact scoring-prompt renderer + teacher-forced YES/NO readout (frozen semantics)
    "score_prompt": ("score_adaptive_ostensive_orbits", "score_prompt"),
    "score_declared_binary": ("score_adaptive_ostensive_orbits", "score_declared_binary"),
    # item loading with content-hash verification
    "load_domain_items": ("score_fresh_target_views", "load_domain_items"),
    # hashing utilities used across the apparatus
    "sha256_file": ("build_fresh_item_partitions", "sha256_file"),
    "text_sha256": ("build_fresh_item_partitions", "text_sha256"),
}

__all__ = sorted(_SOURCES) + ["APPARATUS_ROOT", "apparatus_module"]


def apparatus_module(name: str):
    """Import a module of the frozen apparatus by its unqualified name."""
    return importlib.import_module(f"{APPARATUS_ROOT}.{name}")


def __getattr__(name: str):
    if name in _SOURCES:
        module_name, symbol = _SOURCES[name]
        return getattr(apparatus_module(module_name), symbol)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
