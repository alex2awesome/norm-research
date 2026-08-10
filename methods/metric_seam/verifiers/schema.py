"""Strict JSON schemas for source-local verifier outcomes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import PurePosixPath
from typing import Mapping


class SchemaError(ValueError):
    """Raised when a verifier output violates its frozen JSON contract."""


def _reject_float(_: str) -> None:
    raise SchemaError("floating-point values are forbidden in verifier JSON")


def validate_json_no_floats(value: object, *, path: str = "$") -> None:
    """Validate the JSON value domain and reject floats at every depth."""

    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        raise SchemaError(f"{path}: floating-point values are forbidden")
    if isinstance(value, list):
        for index, child in enumerate(value):
            validate_json_no_floats(child, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise SchemaError(f"{path}: JSON object keys must be strings")
            validate_json_no_floats(child, path=f"{path}.{key}")
        return
    raise SchemaError(f"{path}: value is not JSON-compatible: {type(value).__name__}")


def load_json_no_floats(raw: str | bytes | bytearray) -> object:
    """Deserialize one JSON value while rejecting all floating-point tokens."""

    try:
        value = json.loads(
            raw,
            parse_float=_reject_float,
            parse_constant=_reject_float,
        )
    except SchemaError:
        raise
    except (TypeError, json.JSONDecodeError) as exc:
        raise SchemaError("invalid verifier JSON") from exc
    validate_json_no_floats(value)
    return value


def _require_exact_keys(value: Mapping[str, object], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SchemaError(f"{path}: key mismatch; missing={missing}, extra={extra}")


@dataclass(frozen=True, order=True)
class Span:
    """Inclusive one-based source-line span used as a replayable witness.

    ``path`` is part of witness identity: line 10 in two files denotes two
    different source locations.  ``node_id`` may bind the span to an optional
    parser/graph node without changing line-set overlap semantics.
    """

    path: str
    start_line: int
    end_line: int
    node_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise SchemaError("span path must be a nonempty string")
        if self.path != self.path.strip():
            raise SchemaError("span path must not have surrounding whitespace")
        if (
            "\\" in self.path
            or ":" in self.path
            or self.path.startswith("~")
            or any(ord(char) < 32 for char in self.path)
        ):
            raise SchemaError("span path contains unsafe characters")
        parsed_path = PurePosixPath(self.path)
        if (
            parsed_path.is_absolute()
            or any(part in ("", ".", "..") for part in self.path.split("/"))
            or str(parsed_path) != self.path
        ):
            raise SchemaError("span path must be a safe relative POSIX path")
        if isinstance(self.start_line, bool) or not isinstance(self.start_line, int):
            raise SchemaError("span start_line must be an integer")
        if isinstance(self.end_line, bool) or not isinstance(self.end_line, int):
            raise SchemaError("span end_line must be an integer")
        if self.start_line < 1:
            raise SchemaError("span start_line must be one-based")
        if self.end_line < self.start_line:
            raise SchemaError("span end_line must not precede start_line")
        if self.node_id is not None:
            if not isinstance(self.node_id, str) or not self.node_id:
                raise SchemaError("span node_id must be a nonempty string when present")
            if any(ord(char) < 32 for char in self.node_id):
                raise SchemaError("span node_id contains unsafe characters")

    @classmethod
    def from_json_value(cls, value: object, *, path: str = "$") -> "Span":
        validate_json_no_floats(value, path=path)
        if not isinstance(value, dict):
            raise SchemaError(f"{path}: span must be an object")
        required = {"path", "start_line", "end_line"}
        actual = set(value)
        if not required <= actual or actual - required not in (set(), {"node_id"}):
            missing = sorted(required - actual)
            extra = sorted(actual - required - {"node_id"})
            raise SchemaError(f"{path}: key mismatch; missing={missing}, extra={extra}")
        return cls(
            path=value["path"],
            start_line=value["start_line"],
            end_line=value["end_line"],
            node_id=value.get("node_id"),
        )

    def to_json_value(self) -> dict[str, object]:
        value: dict[str, object] = {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }
        if self.node_id is not None:
            value["node_id"] = self.node_id
        return value

    def lines(self) -> frozenset[tuple[str, int]]:
        """Return file-qualified inclusive line identities."""

        return frozenset(
            (self.path, line) for line in range(self.start_line, self.end_line + 1)
        )


@dataclass(frozen=True)
class Verdict:
    """A verifier's applicability, polarity, and replayable source witnesses.

    The boolean wire format is equivalent to a three-state outcome:
    ``applies=false`` is ``None``/not-applicable; ``applies=true,
    violated=false`` is satisfied; and ``applies=true, violated=true`` is
    violated.  Every applicable verdict carries at least one source witness:
    on a satisfied outcome it grounds the occasion judged, and on a violated
    outcome it grounds the violation.  This also prevents an ungrounded
    ``applies=true`` pair from receiving perfect empty-set witness agreement.
    Non-applicable verdicts carry none.
    """

    applies: bool
    violated: bool
    witnesses: tuple[Span, ...] = ()

    def __post_init__(self) -> None:
        if type(self.applies) is not bool:  # bool is intentionally exact here
            raise SchemaError("verdict applies must be boolean")
        if type(self.violated) is not bool:
            raise SchemaError("verdict violated must be boolean")
        if not isinstance(self.witnesses, tuple) or not all(
            isinstance(span, Span) for span in self.witnesses
        ):
            raise SchemaError("verdict witnesses must be a tuple of Span values")
        if self.violated and not self.applies:
            raise SchemaError("a non-applicable verdict cannot be violated")
        if not self.applies and self.witnesses:
            raise SchemaError("a non-applicable verdict cannot carry witnesses")
        if self.applies and not self.witnesses:
            raise SchemaError("an applicable verdict requires at least one witness")

    @classmethod
    def from_json(cls, value: str | bytes | bytearray | object) -> "Verdict":
        decoded = (
            load_json_no_floats(value)
            if isinstance(value, (str, bytes, bytearray))
            else value
        )
        validate_json_no_floats(decoded)
        if not isinstance(decoded, dict):
            raise SchemaError("$: verdict must be an object")
        _require_exact_keys(decoded, {"applies", "violated", "witnesses"}, "$")
        witnesses = decoded["witnesses"]
        if not isinstance(witnesses, list):
            raise SchemaError("$.witnesses: must be an array")
        return cls(
            applies=decoded["applies"],
            violated=decoded["violated"],
            witnesses=tuple(
                Span.from_json_value(span, path=f"$.witnesses[{index}]")
                for index, span in enumerate(witnesses)
            ),
        )

    def to_json_value(self) -> dict[str, object]:
        return {
            "applies": self.applies,
            "violated": self.violated,
            "witnesses": [span.to_json_value() for span in self.witnesses],
        }

    @property
    def state(self) -> str:
        """Return the canonical three-state label used by discrimination gates."""

        if not self.applies:
            return "not_applicable"
        return "violated" if self.violated else "satisfied"
