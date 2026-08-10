from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import SCHEMA_VERSION


WS_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^a-z0-9]+")


def normalize_space(value: Any) -> str:
    # Some scraped Reddit exports contain lone UTF-16 surrogates.  Replace
    # them deterministically so every manifest is valid UTF-8/JSONL.
    text = str(value or "").encode("utf-8", "replace").decode("utf-8")
    return WS_RE.sub(" ", text).strip()


def normalize_name(value: Any) -> str:
    # Preserve token boundaries for punctuation that ASCII transliteration
    # would otherwise silently delete (notably en/em dashes).
    raw = re.sub(r"[\u2010-\u2015\u2212]", "-", normalize_space(value))
    text = unicodedata.normalize("NFKD", raw).encode(
        "ascii", "ignore"
    ).decode("ascii")
    return PUNCT_RE.sub(" ", text.lower()).strip()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_uid(*parts: Any) -> str:
    payload = "\x1f".join(normalize_space(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8", "ignore")).hexdigest()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            yield value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write a complete versioned artifact atomically.

    This is used only for generated outputs, never to mutate source corpora.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    tmp.replace(path)
    return count


def extract_norm(record: dict[str, Any]) -> str:
    for key in ("signal_text", "s", "norm", "anchor", "text", "signal"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_space(value)
    return ""


def extract_source_id(record: dict[str, Any], row: int) -> str:
    for key in ("source_id", "i", "signal_id", "id", "unit_id"):
        value = record.get(key)
        if value is not None:
            return normalize_space(value)
    return str(row)


def metric_card(metric: dict[str, Any], max_examples: int = 4) -> str:
    examples = [normalize_space(v) for v in metric.get("examples", []) if normalize_space(v)]
    text = f"{metric['name']}. Definition: {metric.get('description', '')}"
    if examples:
        text += " Examples: " + "; ".join(examples[:max_examples])
    return normalize_space(text)


def norm_statement_query(norm: dict[str, Any]) -> str:
    """Render the extracted statement without its surrounding passage."""
    statement = normalize_space(norm.get("norm"))
    pieces = [f"Task: {norm['task']}", f"Human evaluative statement: {statement}"]
    if norm.get("aspect"):
        pieces.append(f"Weak extraction aspect hint: {norm['aspect']}")
    return normalize_space(". ".join(pieces))


def norm_query(norm: dict[str, Any], context_chars: int = 1600) -> str:
    """Render the evidence available to a matcher/retriever.

    The extracted phrase can be deictic or intentionally terse. Retrieval
    must see the same source passage used by human and LLM adjudicators;
    otherwise recovered faithful rows are systematically misrouted. Keep the
    weak extractor aspect last so it cannot dominate the actual evidence.
    """
    statement = normalize_space(norm.get("norm"))
    pieces = [f"Task: {norm['task']}", f"Human evaluative statement: {statement}"]
    context = normalize_space(norm.get("context"))
    if context and context != statement:
        if context_chars < 1:
            raise ValueError("context_chars must be positive")
        context = (
            context
            if len(context) <= context_chars
            else context[:context_chars].rstrip() + "…"
        )
        pieces.append(f"Evidence passage: {context}")
    if norm.get("aspect"):
        pieces.append(f"Weak extraction aspect hint: {norm['aspect']}")
    return normalize_space(". ".join(pieces))


def norm_query_views(norm: dict[str, Any], context_chars: int = 1600) -> tuple[str, ...]:
    """Return evidence and statement views, deduplicated in priority order."""
    values = (norm_query(norm, context_chars=context_chars), norm_statement_query(norm))
    return tuple(dict.fromkeys(values))


def schema_record(**kwargs: Any) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, **kwargs}
