"""Enforced, provenance-derived metric DAGs (additive successor to ``dag_schema``).

The historical WS4 DAG records author-declared ``impl``/``needs`` metadata while giving
every function the full runtime context.  This module reverses that trust boundary:

* each node receives a restricted mapping containing only explicitly bound, typed inputs;
* document text, operations, prompt/LLM fields, and external evidence are explicit sources;
* dependencies are the executable bindings, not a parallel annotation;
* C/LLM/evidence provenance is derived from values actually accessed at runtime;
* every declared node must reach the output, and dynamic seam summaries include only nodes
  that actually contribute to that run's output.

Terminology follows reconstruction v2: prompt/LLM inputs are *articulability*; executable
transforms are *verifiability*.  A code verifier may disagree with the frozen LLM reference,
so evidence provenance is recorded independently instead of being forced into an L label.

Node schema::

    {
      "id": "score",
      "ntype": "A",                 # R | T | A
      "level": 3,                   # 0 raw -> 3 verdict
      "inputs": {
        "feature": {"node": "feature", "type": "number"},
        "clarity": {"source": "llm", "key": "clarity", "type": "score"},
        "prior_art": {"source": "evidence", "key": "prior_art", "type": "sequence"}
      },
      "output_type": "score",
      "fn": lambda inputs: ...
    }

Supported sources are ``text``, ``ops``, ``llm``, and ``evidence``.  LLM/evidence sources
must name one exact key, preventing a node from browsing the full field/evidence mapping.
There is deliberately no author-declared ``impl`` or ``op_class``.

This is an execution-isomorphism guard for trusted metric code, not an OS security sandbox:
a hostile Python closure can still read globals or the filesystem and must be isolated by a
separate process sandbox in a fully adversarial compiler.
"""

from __future__ import annotations

import inspect
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterator, Optional


LEVELS = {0: "raw-text", 1: "span", 2: "feature", 3: "verdict"}
NODE_TYPES = frozenset({"R", "T", "A"})
SOURCES = frozenset({"text", "ops", "llm", "evidence"})
TYPE_NAMES = frozenset(
    {
        "text",
        "string",
        "mapping",
        "sequence",
        "evidence",
        "ops",
        "number",
        "score",
        "bool",
        "json",
    }
)
REQUIRED_NODE_FIELDS = ("id", "ntype", "level", "inputs", "output_type", "fn")
FORBIDDEN_DECLARATIONS = ("impl", "op_class", "needs", "sources")


class DagValidationError(ValueError):
    pass


class DagExecutionError(RuntimeError):
    pass


class UndeclaredInputError(DagExecutionError, KeyError):
    pass


class SourceUnavailableError(DagExecutionError):
    pass


class DagTypeError(DagExecutionError, TypeError):
    pass


class NodeExecutionError(DagExecutionError):
    pass


@dataclass(frozen=True)
class NodeTrace:
    node_id: str
    value: Any
    output_type: str
    accessed_inputs: tuple[str, ...]
    accessed_node_ids: tuple[str, ...]
    taints: frozenset[str]
    implementation: str
    channels: tuple[str, ...]
    evidence_tainted: bool

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["taints"] = sorted(self.taints)
        return result


@dataclass(frozen=True)
class SeamSummary:
    articulability_frontier: tuple[str, ...]
    evidence_frontier: tuple[str, ...]
    n_contributing_nodes: int
    n_articulability_tainted: int
    n_evidence_tainted: int
    frontier_levels: tuple[int, ...]


@dataclass(frozen=True)
class ExecutionResult:
    output: Any
    trace: Mapping[str, NodeTrace]
    contributing_nodes: tuple[str, ...]
    seam: SeamSummary


def _split_optional(type_name: str) -> tuple[bool, str]:
    if isinstance(type_name, str) and type_name.startswith("optional[") and type_name.endswith("]"):
        return True, type_name[9:-1]
    return False, type_name


def _valid_type_name(type_name: Any) -> bool:
    if not isinstance(type_name, str):
        return False
    _, base = _split_optional(type_name)
    return base in TYPE_NAMES


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def value_matches_type(value: Any, type_name: str) -> bool:
    optional, base = _split_optional(type_name)
    if value is None:
        return optional
    if base in ("text", "string"):
        return isinstance(value, str)
    if base == "mapping":
        return isinstance(value, Mapping)
    if base == "sequence":
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    if base == "evidence":
        return isinstance(value, (str, Mapping, Sequence)) and not isinstance(
            value, (bytes, bytearray)
        )
    if base == "ops":
        return value is not None
    if base == "number":
        return _is_number(value)
    if base == "score":
        return _is_number(value) and 0.0 <= float(value) <= 1.0
    if base == "bool":
        return isinstance(value, bool)
    if base == "json":
        if isinstance(value, (str, bool, int)) or value is None:
            return True
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, Mapping):
            return all(isinstance(k, str) and value_matches_type(v, "json") for k, v in value.items())
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return all(value_matches_type(v, "json") for v in value)
        return False
    raise ValueError(f"unknown type {type_name!r}")


def _assignable(actual: str, expected: str) -> bool:
    a_optional, a_base = _split_optional(actual)
    e_optional, e_base = _split_optional(expected)
    if a_optional and not e_optional:
        return False
    if a_base == e_base:
        return True
    # A bounded score is a number; text and string are aliases.
    if a_base == "score" and e_base == "number":
        return True
    if {a_base, e_base} == {"text", "string"}:
        return True
    return False


def _binding_edge(binding: Mapping[str, Any]) -> Optional[str]:
    return binding.get("node") if isinstance(binding, Mapping) else None


def _function_has_one_clean_input(fn: Any) -> bool:
    if not callable(fn):
        return False
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    return bool(
        len(params) == 1
        and params[0].kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and params[0].default is inspect.Parameter.empty
    )


def validate(prog: Mapping[str, Any]) -> list[str]:
    """Return all static schema, type, DAG, and reachability errors."""

    errors: list[str] = []
    if not isinstance(prog, Mapping):
        return ["program must be a mapping"]
    raw_nodes = prog.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        return ["program.nodes must be a non-empty list"]

    ids = [
        node.get("id")
        for node in raw_nodes
        if isinstance(node, Mapping) and isinstance(node.get("id"), str)
    ]
    if len(ids) != len(raw_nodes):
        errors.append("every node must be a mapping with an id")
    if len(ids) != len(set(ids)):
        errors.append("duplicate node ids")
    nodes = {
        node.get("id"): node
        for node in raw_nodes
        if isinstance(node, Mapping) and isinstance(node.get("id"), str)
    }
    out = prog.get("out")
    if not isinstance(out, str) or out not in nodes:
        errors.append(f"out node {out!r} missing")

    for raw in raw_nodes:
        if not isinstance(raw, Mapping):
            continue
        node_id = raw.get("id", "?")
        for field in REQUIRED_NODE_FIELDS:
            if field not in raw:
                errors.append(f"{node_id}: missing {field}")
        for field in FORBIDDEN_DECLARATIONS:
            if field in raw:
                errors.append(
                    f"{node_id}: {field} is author-declared metadata; enforced DAGs derive it"
                )
        if raw.get("ntype") not in NODE_TYPES:
            errors.append(f"{node_id}: ntype must be R, T, or A")
        if raw.get("level") not in LEVELS:
            errors.append(f"{node_id}: level must be one of {sorted(LEVELS)}")
        if not _valid_type_name(raw.get("output_type")):
            errors.append(f"{node_id}: invalid output_type {raw.get('output_type')!r}")
        if not _function_has_one_clean_input(raw.get("fn")):
            errors.append(f"{node_id}: fn must take exactly one required positional input mapping")
        inputs = raw.get("inputs")
        if not isinstance(inputs, Mapping):
            errors.append(f"{node_id}: inputs must be a mapping")
            continue
        for alias, binding in inputs.items():
            if not isinstance(alias, str) or not alias:
                errors.append(f"{node_id}: input aliases must be non-empty strings")
                continue
            if not isinstance(binding, Mapping):
                errors.append(f"{node_id}.{alias}: binding must be a mapping")
                continue
            has_node = "node" in binding
            has_source = "source" in binding
            if has_node == has_source:
                errors.append(
                    f"{node_id}.{alias}: binding needs exactly one of node or source"
                )
                continue
            expected_type = binding.get("type")
            if not _valid_type_name(expected_type):
                errors.append(
                    f"{node_id}.{alias}: invalid input type {expected_type!r}"
                )
            if has_node:
                dep = binding.get("node")
                if not isinstance(dep, str) or dep not in nodes:
                    errors.append(f"{node_id}.{alias}: unknown node {dep!r}")
                elif dep == node_id:
                    errors.append(f"{node_id}.{alias}: self-dependency")
                else:
                    dep_type = nodes[dep].get("output_type")
                    if _valid_type_name(dep_type) and _valid_type_name(expected_type):
                        if not _assignable(dep_type, expected_type):
                            errors.append(
                                f"{node_id}.{alias}: {dep} outputs {dep_type}, "
                                f"binding expects {expected_type}"
                            )
                    dep_level = nodes[dep].get("level")
                    node_level = raw.get("level")
                    if dep_level in LEVELS and node_level in LEVELS and dep_level > node_level:
                        errors.append(
                            f"LEVEL INVERSION {dep}(L{dep_level}) -> {node_id}(L{node_level})"
                        )
                if "key" in binding:
                    errors.append(f"{node_id}.{alias}: node binding must not declare key")
            else:
                source = binding.get("source")
                if not isinstance(source, str) or source not in SOURCES:
                    errors.append(f"{node_id}.{alias}: unknown source {source!r}")
                key = binding.get("key")
                if source in ("llm", "evidence"):
                    if not isinstance(key, str) or not key:
                        errors.append(
                            f"{node_id}.{alias}: {source} source requires one exact key"
                        )
                elif "key" in binding:
                    errors.append(f"{node_id}.{alias}: {source} source must not declare key")
                if source == "text" and _valid_type_name(expected_type):
                    if not _assignable("text", expected_type):
                        errors.append(f"{node_id}.{alias}: text source must be typed text")
                if source == "ops" and expected_type != "ops":
                    errors.append(f"{node_id}.{alias}: ops source must be typed ops")

    if out in nodes:
        if nodes[out].get("ntype") != "A":
            errors.append("out node must have ntype A")
        if nodes[out].get("output_type") != "score":
            errors.append("out node must have output_type score")

    # Kahn cycle check over executable node bindings.
    indegree = {node_id: 0 for node_id in nodes}
    kids = {node_id: [] for node_id in nodes}
    for node_id, node in nodes.items():
        inputs = node.get("inputs", {})
        if not isinstance(inputs, Mapping):
            continue
        for binding in inputs.values():
            dep = _binding_edge(binding)
            if isinstance(dep, str) and dep in nodes:
                indegree[node_id] += 1
                kids[dep].append(node_id)
    queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    seen: list[str] = []
    while queue:
        node_id = queue.popleft()
        seen.append(node_id)
        for kid in kids[node_id]:
            indegree[kid] -= 1
            if indegree[kid] == 0:
                queue.append(kid)
    if len(seen) != len(nodes):
        errors.append("cycle detected")

    # Every node must be on a declared path to the scalar output.
    if out in nodes:
        ancestors: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in ancestors:
                return
            ancestors.add(node_id)
            inputs = nodes[node_id].get("inputs", {})
            if isinstance(inputs, Mapping):
                for binding in inputs.values():
                    dep = _binding_edge(binding)
                    if isinstance(dep, str) and dep in nodes:
                        visit(dep)

        visit(out)
        unreachable = sorted(set(nodes) - ancestors)
        if unreachable:
            errors.append(f"nodes do not reach out: {unreachable}")
    return errors


def assert_valid(prog: Mapping[str, Any]) -> None:
    errors = validate(prog)
    if errors:
        raise DagValidationError("; ".join(errors))


def topo_order(prog: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    assert_valid(prog)
    nodes = {node["id"]: node for node in prog["nodes"]}
    order: list[str] = []
    done: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in done:
            return
        for binding in nodes[node_id]["inputs"].values():
            dep = binding.get("node")
            if dep is not None:
                visit(dep)
        done.add(node_id)
        order.append(node_id)

    visit(prog["out"])
    return [nodes[node_id] for node_id in order]


class _RestrictedInputs(Mapping[str, Any]):
    """Lazy mapping that rejects undeclared reads and records actual value access."""

    __slots__ = ("__resolvers", "__accessed", "__cache")

    def __init__(self, resolvers: Mapping[str, Callable[[], Any]]) -> None:
        self.__resolvers = dict(resolvers)
        self.__accessed: list[str] = []
        self.__cache: dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        if key not in self.__resolvers:
            raise UndeclaredInputError(f"undeclared input access: {key!r}")
        if key not in self.__accessed:
            self.__accessed.append(key)
        if key not in self.__cache:
            self.__cache[key] = self.__resolvers[key]()
        return self.__cache[key]

    def get(self, key: str, default: Any = None) -> Any:
        # Silent fallback would recreate the old full-context bypass.  Every attempted
        # input read must be declared, even when candidate code supplies a default.
        if key not in self.__resolvers:
            raise UndeclaredInputError(f"undeclared input access: {key!r}")
        return self[key]

    def __iter__(self) -> Iterator[str]:
        # Enumerating the declared interface does not consume source values and therefore
        # does not create LLM/evidence taint.  ``dict(inputs)`` subsequently calls getitem.
        return iter(self.__resolvers)

    def __len__(self) -> int:
        return len(self.__resolvers)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str) or key not in self.__resolvers:
            raise UndeclaredInputError(f"undeclared input access: {key!r}")
        # Presence of a declared binding is schema metadata, not consumption of its value.
        return True

    @property
    def accessed(self) -> tuple[str, ...]:
        return tuple(self.__accessed)


def _mapping_key(source_name: str, payload: Optional[Mapping[str, Any]], key: str) -> Any:
    if payload is None:
        raise SourceUnavailableError(f"{source_name} source was not supplied")
    if not isinstance(payload, Mapping):
        raise DagTypeError(f"{source_name} source must be a mapping")
    if key not in payload:
        raise SourceUnavailableError(f"{source_name} key {key!r} is unavailable")
    return payload[key]


def _source_taints(source: str) -> frozenset[str]:
    return {
        "text": frozenset({"document"}),
        "ops": frozenset({"ops"}),
        "llm": frozenset({"llm"}),
        "evidence": frozenset({"evidence"}),
    }[source]


def _implementation(taints: frozenset[str]) -> str:
    return "C+LLM" if "llm" in taints else "C"


def execute(
    prog: Mapping[str, Any],
    *,
    text: str,
    ops: Any = None,
    llm_fields: Optional[Mapping[str, Any]] = None,
    evidence: Optional[Mapping[str, Any]] = None,
) -> ExecutionResult:
    """Execute one item and derive the dynamic articulability/evidence seam."""

    assert_valid(prog)
    traces: dict[str, NodeTrace] = {}

    for node in topo_order(prog):
        node_id = node["id"]
        resolvers: dict[str, Callable[[], Any]] = {}
        input_taints: dict[str, frozenset[str]] = {}
        input_nodes: dict[str, Optional[str]] = {}

        for alias, binding in node["inputs"].items():
            expected_type = binding["type"]
            if "node" in binding:
                dep = binding["node"]

                def resolve_dep(
                    dep: str = dep,
                    expected_type: str = expected_type,
                    alias: str = alias,
                ) -> Any:
                    value = traces[dep].value
                    if not value_matches_type(value, expected_type):
                        raise DagTypeError(
                            f"{node_id}.{alias}: runtime value from {dep} is not {expected_type}"
                        )
                    return value

                resolvers[alias] = resolve_dep
                input_taints[alias] = traces[dep].taints
                input_nodes[alias] = dep
            else:
                source = binding["source"]
                key = binding.get("key")

                def resolve_source(
                    source: str = source,
                    key: Optional[str] = key,
                    expected_type: str = expected_type,
                    alias: str = alias,
                ) -> Any:
                    if source == "text":
                        value = text
                    elif source == "ops":
                        if ops is None:
                            raise SourceUnavailableError("ops source was not supplied")
                        value = ops
                    elif source == "llm":
                        assert key is not None
                        value = _mapping_key("llm", llm_fields, key)
                    else:
                        assert source == "evidence" and key is not None
                        value = _mapping_key("evidence", evidence, key)
                    if not value_matches_type(value, expected_type):
                        raise DagTypeError(
                            f"{node_id}.{alias}: {source} value is not {expected_type}"
                        )
                    return value

                resolvers[alias] = resolve_source
                input_taints[alias] = _source_taints(source)
                input_nodes[alias] = None

        restricted = _RestrictedInputs(resolvers)
        try:
            value = node["fn"](restricted)
        except (UndeclaredInputError, SourceUnavailableError, DagTypeError):
            raise
        except Exception as exc:
            raise NodeExecutionError(
                f"node {node_id!r} raised {type(exc).__name__}: {exc}"
            ) from exc
        if not value_matches_type(value, node["output_type"]):
            raise DagTypeError(
                f"{node_id}: output {value!r} does not satisfy {node['output_type']}"
            )

        accessed = restricted.accessed
        taints: set[str] = {"code"}
        for alias in accessed:
            taints.update(input_taints[alias])
        frozen_taints = frozenset(taints)
        channels = ["verifiability"]
        if "llm" in frozen_taints:
            channels.append("articulability")
        traces[node_id] = NodeTrace(
            node_id=node_id,
            value=value,
            output_type=node["output_type"],
            accessed_inputs=accessed,
            accessed_node_ids=tuple(
                input_nodes[alias] for alias in accessed if input_nodes[alias] is not None
            ),
            taints=frozen_taints,
            implementation=_implementation(frozen_taints),
            channels=tuple(channels),
            evidence_tainted="evidence" in frozen_taints,
        )

    # Dynamic backward slice: a computed but unaccessed branch does not define the seam.
    contributing: set[str] = set()

    def add_dynamic_ancestors(node_id: str) -> None:
        if node_id in contributing:
            return
        contributing.add(node_id)
        for dep in traces[node_id].accessed_node_ids:
            add_dynamic_ancestors(dep)

    add_dynamic_ancestors(prog["out"])
    node_map = {node["id"]: node for node in prog["nodes"]}

    def frontier_for(taint: str) -> tuple[str, ...]:
        frontier = []
        for node_id in sorted(contributing):
            trace = traces[node_id]
            if taint not in trace.taints:
                continue
            if not any(
                dep in contributing and taint in traces[dep].taints
                for dep in trace.accessed_node_ids
            ):
                frontier.append(node_id)
        return tuple(frontier)

    llm_frontier = frontier_for("llm")
    evidence_frontier = frontier_for("evidence")
    seam = SeamSummary(
        articulability_frontier=llm_frontier,
        evidence_frontier=evidence_frontier,
        n_contributing_nodes=len(contributing),
        n_articulability_tainted=sum(
            "llm" in traces[node_id].taints for node_id in contributing
        ),
        n_evidence_tainted=sum(
            "evidence" in traces[node_id].taints for node_id in contributing
        ),
        frontier_levels=tuple(sorted(node_map[node_id]["level"] for node_id in llm_frontier)),
    )
    return ExecutionResult(
        output=traces[prog["out"]].value,
        trace=traces,
        contributing_nodes=tuple(sorted(contributing)),
        seam=seam,
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
