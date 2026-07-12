"""Hardened callable provenance for enforced metric DAGs (v3, additive).

``dag_schema_enforced`` v2 closes the historical accidental leak in which every node was
given the full execution context.  Its restricted input mapping does not, by itself, stop
an adversarial Python function from reading a module global or closure.  This module adds a
stricter *callable provenance policy* on top of the unchanged v2 schema and trace engine:

* node implementations must be plain Python functions with no closure, defaults, mutable
  function attributes, imports, ambient global reads, or private/frame introspection;
* the only global names available to accepted bytecode are a small set of pure builtins;
* accepted functions are reconstructed with a fresh minimal globals dictionary before
  execution, so a module-level shadow of an allowed builtin is not inherited;
* every declared data input is normalized to immutable builtins per node invocation (and
  every ``ops`` object is cloned), preventing nodes from communicating through shared input
  mutation.

This deliberately stricter version may reject otherwise-benign metric code.  Such code
should move reusable algorithms behind the explicitly declared ``ops`` capability source,
or express them as self-contained functions.  ``ops`` implementations remain a trusted,
separately audited capability boundary: this policy audits node implementations, not the
transitive internals of arbitrary capability objects.

Security boundary and limitations
---------------------------------
This is a fail-closed provenance guard for trusted/reviewable metric programs, **not an OS
sandbox**.  It does not provide process isolation, resource limits, syscall filtering, or a
proof against every Python object-graph exploit.  Candidate code from an untrusted compiler
must still run in a separately isolated process.  The policy prevents the concrete ambient
global/closure/module-state channels named above and makes unsupported callables refuse to
run instead of silently receiving ``C`` provenance.
"""

from __future__ import annotations

import ast
import builtins
import copy
import dis
import hashlib
import inspect
import marshal
import textwrap
import types
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterator, Optional

try:  # Package import in the fleet; standalone import in the local battery scripts/tests.
    from . import dag_schema_enforced as _v2
except ImportError:  # pragma: no cover - exercised by direct script-style imports
    import dag_schema_enforced as _v2


HARDENED_POLICY_VERSION = "metric-seam-dag-provenance-v3"

# Deliberately small.  In particular this excludes open/eval/exec/compile/__import__,
# globals/locals/vars/dir, getattr/setattr/delattr, object/type, and exception classes.
_SAFE_BUILTIN_NAMES = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }
)
_SAFE_BUILTINS = MappingProxyType(
    {name: getattr(builtins, name) for name in sorted(_SAFE_BUILTIN_NAMES)}
)

_FORBIDDEN_BYTECODE = frozenset(
    {
        "IMPORT_NAME",
        "IMPORT_FROM",
        "IMPORT_STAR",
        "LOAD_BUILD_CLASS",
        "STORE_GLOBAL",
        "DELETE_GLOBAL",
        "STORE_DEREF",
        "DELETE_DEREF",
    }
)
_FORBIDDEN_ATTRIBUTES = frozenset(
    {
        "f_back",
        "f_builtins",
        "f_code",
        "f_globals",
        "f_locals",
        "gi_frame",
        "cr_frame",
        "ag_frame",
        "tb_frame",
    }
)


class CallableProvenanceError(_v2.DagValidationError):
    """A node callable cannot be certified as using only its declared inputs."""


class InputIsolationError(_v2.DagTypeError):
    """A declared value could not be copied into a node-local input boundary."""


@dataclass(frozen=True)
class CallableAudit:
    """Deterministic audit record for one node implementation."""

    policy_version: str
    code_sha256: Optional[str]
    source_sha256: Optional[str]
    source_audited: bool
    errors: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return not self.errors


def _private_or_frame_attribute(name: str) -> bool:
    return name.startswith("_") or name in _FORBIDDEN_ATTRIBUTES or "frame" in name.lower()


def _walk_code(code: types.CodeType) -> Iterator[types.CodeType]:
    yield code
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            yield from _walk_code(constant)


def _source_audit(fn: types.FunctionType) -> tuple[bool, Optional[str], list[str]]:
    """Best-effort AST defense; bytecode audit remains mandatory when source is absent."""

    try:
        source = textwrap.dedent(inspect.getsource(fn))
        parsed = ast.parse(source)
    except (OSError, IOError, TypeError, IndentationError, SyntaxError):
        return False, None, []

    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    # inspect.getsource(lambda) commonly returns its containing call.  Avoid attributing a
    # sibling expression to the lambda; recursive bytecode inspection still covers it.
    definitions = [
        item
        for item in parsed.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == fn.__name__
    ]
    if len(definitions) != 1:
        return False, source_hash, []

    errors: list[str] = []
    for item in ast.walk(definitions[0]):
        if isinstance(item, (ast.Global, ast.Nonlocal)):
            errors.append(f"source contains {type(item).__name__.lower()} declaration")
        elif isinstance(item, (ast.Import, ast.ImportFrom)):
            errors.append("source contains an import")
        elif isinstance(item, ast.Attribute) and _private_or_frame_attribute(item.attr):
            errors.append(f"source accesses forbidden attribute {item.attr!r}")
    return True, source_hash, errors


def audit_callable(fn: Any) -> CallableAudit:
    """Audit whether ``fn`` can be rebound to a declared-input-only namespace."""

    errors: list[str] = []
    if not inspect.isfunction(fn):
        errors.append("callable must be a plain Python function (no object, method, or partial)")
        return CallableAudit(HARDENED_POLICY_VERSION, None, None, False, tuple(errors))

    assert isinstance(fn, types.FunctionType)
    code_hash = hashlib.sha256(marshal.dumps(fn.__code__)).hexdigest()
    source_audited, source_hash, source_errors = _source_audit(fn)
    errors.extend(source_errors)

    if fn.__closure__ or fn.__code__.co_freevars:
        errors.append("callable closes over nonlocal state")
    if fn.__defaults__ or fn.__kwdefaults__:
        errors.append("callable defaults can carry undeclared state")
    if fn.__dict__:
        errors.append("callable attributes can carry mutable module state")

    for code in _walk_code(fn.__code__):
        for instruction in dis.get_instructions(code):
            opname = instruction.opname
            name = instruction.argval
            if opname in _FORBIDDEN_BYTECODE:
                errors.append(f"forbidden bytecode {opname}")
            elif opname in {"LOAD_GLOBAL", "LOAD_NAME"}:
                if not isinstance(name, str) or name not in _SAFE_BUILTIN_NAMES:
                    errors.append(f"ambient global read {name!r}")
            elif opname in {"LOAD_ATTR", "LOAD_METHOD", "STORE_ATTR", "DELETE_ATTR"}:
                if isinstance(name, str) and _private_or_frame_attribute(name):
                    errors.append(f"forbidden attribute access {name!r}")

    return CallableAudit(
        HARDENED_POLICY_VERSION,
        code_hash,
        source_hash,
        source_audited,
        tuple(sorted(set(errors))),
    )


def validate(prog: Mapping[str, Any]) -> list[str]:
    """Return v2 schema errors plus v3 callable-provenance errors."""

    errors = list(_v2.validate(prog))
    if not isinstance(prog, Mapping):
        return errors
    declared_policy = prog.get("provenance_policy")
    if declared_policy is not None and declared_policy != HARDENED_POLICY_VERSION:
        errors.append(
            f"provenance_policy must be {HARDENED_POLICY_VERSION!r}, got {declared_policy!r}"
        )
    nodes = prog.get("nodes")
    if not isinstance(nodes, list):
        return errors
    for node in nodes:
        if not isinstance(node, Mapping) or "fn" not in node:
            continue
        audit = audit_callable(node["fn"])
        for error in audit.errors:
            errors.append(f"{node.get('id', '?')}: callable provenance: {error}")
    return errors


def assert_valid(prog: Mapping[str, Any]) -> None:
    errors = validate(prog)
    if errors:
        raise CallableProvenanceError("; ".join(errors))


def _isolate_declared_value(value: Any, type_name: str) -> Any:
    """Normalize non-ops values to builtin immutable containers; clone explicit ops."""

    base_type = type_name[9:-1] if type_name.startswith("optional[") else type_name
    if base_type == "ops":
        # A capability object is an explicit trusted boundary, but isolate its instance
        # state between nodes where deepcopy supports it.
        cloned = copy.deepcopy(value)
        if cloned is value:
            raise InputIsolationError(
                "ops capability did not produce an independent deepcopy; "
                "hardened execution cannot isolate its instance state"
            )
        return cloned
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise InputIsolationError(
                    f"hardened mapping input requires string keys, got {type(key).__name__}"
                )
            normalized[key] = _isolate_declared_value(child, "json")
        return MappingProxyType(normalized)
    if isinstance(value, (list, tuple)):
        return tuple(_isolate_declared_value(child, "json") for child in value)
    raise InputIsolationError(
        f"non-ops input contains unsupported object {type(value).__name__}; "
        "use JSON-like data or an explicitly declared ops capability"
    )


class _IsolatedInputs(Mapping[str, Any]):
    """Copy-on-first-read view over v2's access-tracking restricted mapping."""

    __slots__ = ("__base", "__cache", "__types")

    def __init__(self, base: Mapping[str, Any], input_types: Mapping[str, str]) -> None:
        self.__base = base
        self.__cache: dict[str, Any] = {}
        self.__types = dict(input_types)

    def __getitem__(self, key: str) -> Any:
        if key not in self.__cache:
            try:
                value = self.__base[key]
                self.__cache[key] = _isolate_declared_value(value, self.__types[key])
            except Exception as exc:
                if isinstance(exc, InputIsolationError):
                    raise
                raise InputIsolationError(
                    f"declared input {key!r} cannot cross the node-local copy boundary: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        return self.__cache[key]

    def get(self, key: str, default: Any = None) -> Any:
        # Delegate even missing keys so v2 retains fail-closed undeclared-read behavior.
        return self[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__base)

    def __len__(self) -> int:
        return len(self.__base)

    def __contains__(self, key: object) -> bool:
        return key in self.__base


def _rebind(fn: types.FunctionType) -> types.FunctionType:
    # A fresh globals dictionary per callable prevents cross-node module-state channels.
    safe_globals = {"__builtins__": _SAFE_BUILTINS}
    return types.FunctionType(fn.__code__, safe_globals, fn.__name__, None, None)


def _v2_adapter(
    fn: Callable[[Mapping[str, Any]], Any], input_types: Mapping[str, str]
):
    """Build the trusted one-required-argument adapter expected by the v2 engine."""

    def invoke(inputs: Mapping[str, Any]):
        return fn(_IsolatedInputs(inputs, input_types))

    return invoke


def _isolated_program(prog: Mapping[str, Any]) -> dict[str, Any]:
    isolated = dict(prog)
    isolated.pop("provenance_policy", None)
    isolated_nodes: list[dict[str, Any]] = []
    for original in prog["nodes"]:
        node = dict(original)
        rebound = _rebind(node["fn"])
        input_types = {
            alias: binding["type"] for alias, binding in node["inputs"].items()
        }
        # Only user node functions are audited.  This module-owned closure is the trusted
        # adapter that installs the copy-on-read boundary inside the unchanged v2 engine.
        node["fn"] = _v2_adapter(rebound, input_types)
        isolated_nodes.append(node)
    isolated["nodes"] = isolated_nodes
    return isolated


def execute(
    prog: Mapping[str, Any],
    *,
    text: str,
    ops: Any = None,
    llm_fields: Optional[Mapping[str, Any]] = None,
    evidence: Optional[Mapping[str, Any]] = None,
) -> _v2.ExecutionResult:
    """Validate, isolate, execute, and derive the v2 dynamic seam trace."""

    assert_valid(prog)
    return _v2.execute(
        _isolated_program(prog),
        text=text,
        ops=ops,
        llm_fields=llm_fields,
        evidence=evidence,
    )


def score_fn(prog: Mapping[str, Any], *, evidence: Optional[Mapping[str, Any]] = None):
    """Adapter to the fleet's ``score(text, extracted, ops)`` signature."""

    assert_valid(prog)

    def score(text: str, extracted: Mapping[str, Any], ops: Any) -> float:
        return float(
            execute(
                prog,
                text=text,
                ops=ops,
                llm_fields=extracted,
                evidence=evidence,
            ).output
        )

    return score


# Public v2 result/error types remain the wire format for additive compatibility.
DagExecutionError = _v2.DagExecutionError
DagTypeError = _v2.DagTypeError
ExecutionResult = _v2.ExecutionResult
NodeTrace = _v2.NodeTrace
NodeExecutionError = _v2.NodeExecutionError
SeamSummary = _v2.SeamSummary
UndeclaredInputError = _v2.UndeclaredInputError
