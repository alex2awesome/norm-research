"""Private held-out worker for :mod:`evaluate_blind_v2`.

The request contains the frozen TRAIN corpus (only to support explicitly allowed
retrieval operations), opaque held-out texts, cached extraction fields, and no LLM
reference values.  Candidate code therefore cannot observe the reference while it
scores held-out text.  This is a reproducibility boundary, not an adversarial OS
sandbox; the compiler lane's AST policy remains the accidental-leak guard.
"""

from __future__ import annotations

from collections import Counter
import importlib.util
import json
import math
import pathlib
import re
import signal
import sys


HERE = pathlib.Path(__file__).resolve().parent
HYBRIDS = HERE.parent / "hybrids"
if str(HYBRIDS) not in sys.path:
    sys.path.insert(0, str(HYBRIDS))

from ops import Ops  # noqa: E402


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
SUPPORTED_CAPABILITIES = frozenset({"base", "retrieval", "math", "capability"})
_TOKEN_RE = re.compile(r"(?u)\b[a-zA-Z0-9_]{2,}\b")


class _ItemTimeout(Exception):
    pass


def _alarm(_signum, _frame):
    raise _ItemTimeout("candidate score timed out")


def _tokens(text: str):
    return _TOKEN_RE.findall(text[:8000].lower())


def _unit_tfidf(counts: Counter, idf: dict[str, float]) -> dict[str, float]:
    values = {
        term: (1.0 + math.log(freq)) * idf[term]
        for term, freq in counts.items()
        if term in idf and freq > 0
    }
    norm = math.sqrt(sum(value * value for value in values.values()))
    return ({term: value / norm for term, value in values.items()} if norm else {})


class _TrainOnlyRetriever:
    """Deterministic TF-IDF index containing compiler TRAIN text and nothing else."""

    def __init__(self, corpus: dict[str, str]):
        if not corpus:
            raise ValueError("retrieval requires a non-empty frozen TRAIN corpus")
        self._keys = tuple(sorted(corpus))
        counts = {key: Counter(_tokens(corpus[key])) for key in self._keys}
        df = Counter(term for row in counts.values() for term in row)
        n_docs = len(counts)
        self._idf = {
            term: math.log((1.0 + n_docs) / (1.0 + freq)) + 1.0
            for term, freq in df.items()
        }
        self._vectors = {
            key: _unit_tfidf(row, self._idf) for key, row in counts.items()
        }

    def query(self, text: str, *, k: int = 5, exclude_id: str | None = None):
        if not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non-negative integer")
        query = _unit_tfidf(Counter(_tokens(text)), self._idf)
        scored = []
        for key in self._keys:
            if key == exclude_id:
                continue
            row = self._vectors[key]
            left, right = (query, row) if len(query) <= len(row) else (row, query)
            similarity = sum(value * right.get(term, 0.0) for term, value in left.items())
            scored.append((float(similarity), key))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return scored[:k]


class _BoundEvaluationOps:
    __slots__ = ("_owner",)

    def __init__(self, owner: "_SealedEvaluationOps"):
        self._owner = owner

    def retrieve_similar(self, text: str, k: int = 5,
                         exclude_id: str | None = None):
        if self._owner._retriever is None:
            raise AttributeError("retrieval capability was not allowed for this run")
        # The current held-out item is not present in the TRAIN index.  A caller may
        # additionally exclude a TRAIN alias, matching the compiler interface.
        return self._owner._retriever.query(text, k=k, exclude_id=exclude_id)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in BASE_OPS and "base" in self._owner.allowed:
            return getattr(Ops, name)
        if name in MATH_OPS and self._owner._math is not None:
            return getattr(self._owner._math, name)
        if self._owner._capability is not None and name in self._owner._capability_index:
            return getattr(self._owner._capability, name)
        raise AttributeError(f"operation {name!r} is outside the frozen capability set")


class _SealedEvaluationOps:
    __slots__ = ("allowed", "_retriever", "_math", "_capability",
                 "_capability_index", "_bound")

    def __init__(self, train_corpus: dict[str, str], capabilities):
        self.allowed = frozenset(capabilities)
        unknown = self.allowed - SUPPORTED_CAPABILITIES
        if unknown:
            raise ValueError(f"unknown capabilities: {sorted(unknown)}")
        self._retriever = (
            _TrainOnlyRetriever(train_corpus) if "retrieval" in self.allowed else None
        )
        self._math = None
        if "math" in self.allowed:
            from ops_math import MathOps
            self._math = MathOps(corpus_path=None)
        self._capability = None
        self._capability_index = frozenset()
        if "capability" in self.allowed:
            from ops_capability_v2 import CAPABILITIES, CapabilityOps
            self._capability = CapabilityOps()
            self._capability_index = frozenset(CAPABILITIES)
        self._bound = _BoundEvaluationOps(self)

    def for_item(self):
        return self._bound


def _load_candidate(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("sealed_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError("could not create candidate module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "score", None)):
        raise TypeError("candidate must define callable score(text, extracted, ops)")
    return module


def execute(request: dict, candidate_path: pathlib.Path) -> dict:
    train_corpus = {
        row["item_key"]: row["ctext"] for row in request["train_items"]
    }
    ops = _SealedEvaluationOps(train_corpus, request["capabilities"])
    module = _load_candidate(candidate_path)

    declared = getattr(module, "LLM_FIELDS", {}) or {}
    if (not isinstance(declared, dict)
            or any(not isinstance(key, str) or not isinstance(value, str)
                   for key, value in declared.items())):
        raise TypeError("LLM_FIELDS must map field names to prompts")
    expected_fields = request["expected_fields"]
    if declared != expected_fields:
        raise ValueError(
            "candidate LLM_FIELDS do not exactly match frozen field/prompt provenance"
        )

    timeout_s = float(request["timeout_per_item"])
    if timeout_s <= 0:
        raise ValueError("timeout_per_item must be positive")
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm)

    outputs, errors = [], []
    for item in request["eval_items"]:
        key = item["item_key"]
        try:
            if hasattr(signal, "setitimer"):
                signal.setitimer(signal.ITIMER_REAL, timeout_s)
            score = float(module.score(
                item["ctext"], dict(item.get("fields", {})), ops.for_item()
            ))
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("score must be finite and in [0,1]")
            outputs.append({"item_key": key, "score": score})
        except Exception as exc:
            outputs.append({"item_key": key, "score": None})
            errors.append({
                "item_key": key,
                "type": type(exc).__name__,
                "message": str(exc)[:300],
            })
        finally:
            if hasattr(signal, "setitimer"):
                signal.setitimer(signal.ITIMER_REAL, 0)

    values = [row["score"] for row in outputs if row["score"] is not None]
    return {
        "schema": "metric-seam.blind-reconstruction.sealed-worker-result.v2",
        "candidate_declared_fields": sorted(declared),
        "n_items": len(outputs),
        "n_scoreable": len(values),
        "coverage": len(values) / len(outputs) if outputs else 0.0,
        "output_min": min(values) if values else None,
        "output_max": max(values) if values else None,
        "outputs": outputs,
        "errors": errors,
    }


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: _sealed_worker_v2.py REQUEST CANDIDATE OUTPUT")
    request_path, candidate_path, output_path = map(pathlib.Path, sys.argv[1:])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    result = execute(request, candidate_path)
    output_path.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
