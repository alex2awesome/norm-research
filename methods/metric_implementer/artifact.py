"""MetricArtifact — one *implementation* of a metric (prompt OR code) with its budget
measured automatically.

Every scaling axis that describes the artifact itself is computed here, never hand-entered:
instruction tokens, few-shot count, clause count (prompt kind); LOC, AST nodes, cyclomatic
complexity, function count (code kind). The registry stores these with each version, so the
scaling analysis can always recover "how big/complex was this implementation" without
re-parsing old artifacts.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


# ---- complexity measurement -------------------------------------------------------------

_FEWSHOT_PAT = re.compile(r"(?:^|\n)\s*(?:EXAMPLE|Example)\s*\d*\s*[:\n]")


def measure_prompt(prompt_text: str) -> Dict[str, int]:
    words = prompt_text.split()
    # ~0.75 words/token for English prose; good enough for budget caps, and consistent.
    tokens = max(1, round(len(words) / 0.75))
    sentences = len(re.findall(r"[.!?](?:\s|$)", prompt_text))
    bullets = len(re.findall(r"(?:^|\n)\s*(?:[-*•]|\d+[.)])\s", prompt_text))
    fewshots = len(_FEWSHOT_PAT.findall(prompt_text))
    return {"instruction_tokens": tokens, "clause_count": sentences + bullets,
            "n_fewshots": fewshots}


def measure_code(code_text: str) -> Dict[str, int]:
    loc = sum(1 for ln in code_text.splitlines() if ln.strip() and not ln.strip().startswith("#"))
    try:
        tree = ast.parse(code_text)
    except SyntaxError:
        return {"code_loc": loc, "code_ast_nodes": -1, "code_cyclomatic": -1,
                "code_functions": -1}
    n_nodes = sum(1 for _ in ast.walk(tree))
    branch_types = (ast.If, ast.For, ast.While, ast.Try, ast.BoolOp, ast.IfExp,
                    ast.comprehension, ast.ExceptHandler, ast.Assert, ast.With)
    cyclomatic = 1 + sum(isinstance(n, branch_types) for n in ast.walk(tree))
    n_funcs = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree))
    return {"code_loc": loc, "code_ast_nodes": n_nodes, "code_cyclomatic": cyclomatic,
            "code_functions": n_funcs}


# ---- the artifact -----------------------------------------------------------------------

@dataclass
class MetricArtifact:
    """One implementation of a metric.

    kind="prompt": ``body`` is the judge rubric applied per item by an LLM.
    kind="code":   ``body`` is a Python module exposing ``score(text) -> float | None``
                   (None = not applicable), executed in a subprocess sandbox.
    """

    metric_id: str
    kind: str                       # "prompt" | "code"
    body: str
    name: str = ""
    description: str = ""
    invariances: List[str] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)   # free-form metadata, e.g. {"subtask_breadth": "narrow"}
    # transformations the metric SHOULD be invariant to (declared per metric, because a
    # probe that is neutral for one metric is the target of another — e.g. identifier
    # renaming must move identifier_quality but not comment metrics)

    @property
    def body_hash(self) -> str:
        return _hash(self.body)

    def complexity(self) -> Dict[str, int]:
        if self.kind == "prompt":
            return measure_prompt(self.body)
        return measure_code(self.body)

    def violates(self, caps) -> List[str]:
        """Names of budget caps this artifact exceeds (empty = within budget)."""
        c = self.complexity()
        out = []
        if caps.instruction_tokens is not None and \
                c.get("instruction_tokens", 0) > caps.instruction_tokens:
            out.append("instruction_tokens")
        if caps.n_fewshots is not None and c.get("n_fewshots", 0) > caps.n_fewshots:
            out.append("n_fewshots")
        if caps.code_ast_nodes is not None and \
                c.get("code_ast_nodes", 0) > caps.code_ast_nodes:
            out.append("code_ast_nodes")
        return out
