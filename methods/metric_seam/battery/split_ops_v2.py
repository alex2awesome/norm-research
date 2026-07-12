"""Split-scoped operation view for blind metric reconstruction.

This module is deliberately independent of ``battery_common.load_ctx``.  The latter
loads the judge and both data partitions, which is appropriate for historical
evaluation but not for a blind reconstruction interface.

``SplitScopedOps`` is constructed from *only* the opaque TRAIN items in a compiler
bundle.  ``for_item`` returns a narrow view whose retrieval operation always excludes
the current item.  A caller cannot undo that exclusion by supplying another id.

Terminology used by the v2 lane:
  * articulability: prompt/LLM implementation;
  * verifiability: executable/code implementation.
Neither channel is treated as external ground truth; the objective is reconstruction
of the same articulated construct under two implementation media.
"""

from __future__ import annotations

from collections import Counter
import math
import pathlib
import re
import sys
from typing import Dict, Iterable, Mapping, Sequence, Tuple


ROOT = pathlib.Path(__file__).resolve().parents[3]
HYBRIDS = ROOT / "methods" / "metric_seam" / "hybrids"
if str(HYBRIDS) not in sys.path:
    sys.path.insert(0, str(HYBRIDS))

from ops import Ops  # noqa: E402


SUPPORTED_CAPABILITIES = frozenset({"base", "retrieval", "math", "capability"})
BASE_OPS = frozenset({"normalize", "extract_dates", "sent_stats"})
MATH_OPS = frozenset(
    {
        "extract_math_spans",
        "latex_tokens",
        "notation_census",
        "equation_stats",
        "proof_skeleton",
        "delimiter_health",
    }
)

_TOKEN_RE = re.compile(r"(?u)\b[a-zA-Z0-9_]{2,}\b")


def _tokens(text: str) -> Sequence[str]:
    return _TOKEN_RE.findall(text[:8000].lower())


def _unit_tfidf(counts: Counter, idf: Mapping[str, float]) -> Dict[str, float]:
    values = {term: (1.0 + math.log(freq)) * idf[term] for term, freq in counts.items()
              if term in idf and freq > 0}
    norm = math.sqrt(sum(value * value for value in values.values()))
    if not norm:
        return {}
    return {term: value / norm for term, value in values.items()}


class _TrainRetriever:
    """Small deterministic TF-IDF index over the supplied TRAIN ctext only."""

    def __init__(self, corpus: Mapping[str, str]):
        if not corpus:
            raise ValueError("retrieval requires at least one TRAIN item")
        self._keys = tuple(sorted(corpus))
        counts = {key: Counter(_tokens(corpus[key])) for key in self._keys}
        df = Counter(term for row in counts.values() for term in row)
        n_docs = len(counts)
        self._idf = {term: math.log((1.0 + n_docs) / (1.0 + freq)) + 1.0
                     for term, freq in df.items()}
        self._vectors = {key: _unit_tfidf(row, self._idf)
                         for key, row in counts.items()}

    def query(self, text: str, *, current_key: str, k: int = 5,
              exclude_id: str | None = None) -> list[Tuple[float, str]]:
        if current_key not in self._vectors:
            raise KeyError("current item is outside the TRAIN retrieval scope")
        if not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non-negative integer")
        q = _unit_tfidf(Counter(_tokens(text)), self._idf)
        excluded = {current_key}
        if exclude_id is not None:
            excluded.add(exclude_id)
        scored = []
        for key in self._keys:
            if key in excluded:
                continue
            row = self._vectors[key]
            # Iterate over the shorter sparse vector.
            left, right = (q, row) if len(q) <= len(row) else (row, q)
            sim = sum(value * right.get(term, 0.0) for term, value in left.items())
            scored.append((float(sim), key))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return scored[:k]


class BoundSplitOps:
    """Capability-limited operation view bound to one opaque TRAIN item."""

    __slots__ = ("_owner", "_current_key")

    def __init__(self, owner: "SplitScopedOps", current_key: str):
        self._owner = owner
        self._current_key = current_key

    def retrieve_similar(self, text: str, k: int = 5,
                         exclude_id: str | None = None) -> list[Tuple[float, str]]:
        """Retrieve TRAIN neighbors while unconditionally excluding this item."""
        if self._owner._retriever is None:
            raise AttributeError("retrieval capability was not allowed for this run")
        return self._owner._retriever.query(
            text, current_key=self._current_key, k=k, exclude_id=exclude_id
        )

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in BASE_OPS and "base" in self._owner.allowed_capabilities:
            return getattr(Ops, name)
        if name in MATH_OPS and self._owner._math is not None:
            return getattr(self._owner._math, name)
        if self._owner._capability is not None and name in self._owner._capability_index:
            return getattr(self._owner._capability, name)
        raise AttributeError(
            f"operation {name!r} is not in the run's explicit capability allowlist"
        )


class SplitScopedOps:
    """Factory for per-item operation views over a TRAIN-only ctext corpus."""

    __slots__ = ("allowed_capabilities", "_keys", "_retriever", "_math", "_capability",
                 "_capability_index")

    def __init__(self, train_ctext: Mapping[str, str],
                 allowed_capabilities: Iterable[str] = ("base",)):
        corpus = dict(train_ctext)
        if not corpus or any(not isinstance(k, str) or not isinstance(v, str)
                             for k, v in corpus.items()):
            raise ValueError("train_ctext must be a non-empty string-to-string mapping")
        allowed = frozenset(allowed_capabilities)
        unknown = allowed - SUPPORTED_CAPABILITIES
        if unknown:
            raise ValueError(f"unknown capabilities: {sorted(unknown)}")
        self.allowed_capabilities = allowed
        self._keys = frozenset(corpus)
        self._retriever = _TrainRetriever(corpus) if "retrieval" in allowed else None
        self._math = None
        if "math" in allowed:
            from ops_math import MathOps
            self._math = MathOps(corpus_path=None)
        self._capability = None
        self._capability_index = frozenset()
        if "capability" in allowed:
            # v1 remains frozen for historical replay; new blind runs use the
            # conservative, certificate-oriented fixes in reconstruction v2.
            from ops_capability_v2 import CAPABILITIES, CapabilityOps
            self._capability = CapabilityOps()
            self._capability_index = frozenset(CAPABILITIES)

    def for_item(self, item_key: str) -> BoundSplitOps:
        if item_key not in self._keys:
            raise KeyError("item is outside the TRAIN operation scope")
        return BoundSplitOps(self, item_key)
