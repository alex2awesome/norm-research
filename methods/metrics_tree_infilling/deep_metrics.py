"""Deep metric programs: multi-step (judge/code) metrics executed step-synchronously.

A deep metric is a small typed program (notes/2026-07-08__deep-metrics-design.md):

    {"name": ..., "description": ...,
     "steps": [
        {"id": "s1", "kind": "judge_score",   "prompt": "... {text} ... -> JSON {\"score\": x}"},
        {"id": "s2", "kind": "judge_extract", "prompt": "... {text} ... -> JSON {\"span\": str}"},
        {"id": "s3", "kind": "code",          "code": "out = s1 * (1 if len(s2) > 0 else 0)"}],
     "aggregate": "out = s3"}

Judge steps are BATCHED across all items per step (offline vLLM discipline: one llm.chat call
per step, never per item). Code steps run under an AST whitelist (no imports, no attributes
except ``math.*``/``re.search``/``re.findall``, no dunder names). Failures yield NaN for that
item; the caller's viability gate handles degenerate programs.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------------
# Safe code-step execution
# ---------------------------------------------------------------------------------

_ALLOWED_CALLS = {"len", "min", "max", "abs", "sum", "float", "int", "round", "str", "sorted"}
_ALLOWED_ATTR_BASES = {"math", "re"}
_ALLOWED_ATTRS = {"sqrt", "log", "log2", "exp", "floor", "ceil", "isnan", "search", "findall",
                  "pi", "e", "lower", "upper", "strip", "count", "split", "startswith",
                  "endswith"}

_ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call, ast.Attribute,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    ast.And, ast.Or, ast.Not, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Subscript, ast.Index, ast.Slice, ast.Tuple, ast.List,
    ast.ListComp, ast.comprehension, ast.GeneratorExp,
)


def _check_ast(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder name")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr not in _ALLOWED_ATTRS:
                raise ValueError(f"disallowed attribute: {node.attr}")
            base = node.value
            # allow attribute on math/re modules and on step-output strings (s1.lower() etc.)
            if isinstance(base, ast.Name) and base.id in _ALLOWED_ATTR_BASES:
                continue
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id not in _ALLOWED_CALLS:
                raise ValueError(f"disallowed call: {f.id}")


def safe_exec(code: str, ns: Dict) -> Optional[float]:
    """Execute a code step; the step must assign ``out``. Returns None on any failure."""
    try:
        _check_ast(code)
        local = dict(ns)
        local.update({"math": math, "re": re,
                      "len": len, "min": min, "max": max, "abs": abs, "sum": sum,
                      "float": float, "int": int, "round": round, "str": str,
                      "sorted": sorted})
        exec(compile(code, "<deep-metric>", "exec"), {"__builtins__": {}}, local)  # noqa: S102
        return local.get("out")
    except Exception:
        return None


# ---------------------------------------------------------------------------------
# Program schema + step-synchronous executor
# ---------------------------------------------------------------------------------

@dataclass
class DeepMetricProgram:
    name: str
    description: str
    steps: List[dict]
    aggregate: str
    raw: str = ""

    @classmethod
    def from_json(cls, obj: dict, raw: str = "") -> "DeepMetricProgram":
        steps = obj.get("steps") or []
        if not steps or len(steps) > 5:
            raise ValueError("programs need 1-5 steps")
        for s in steps:
            if s.get("kind") not in ("judge_score", "judge_extract", "code"):
                raise ValueError(f"bad step kind {s.get('kind')}")
            if not re.fullmatch(r"s\d+", str(s.get("id", ""))):
                raise ValueError("step ids must be s1..sN")
        return cls(name=str(obj["name"]).strip(), description=str(obj.get("description", "")),
                   steps=steps, aggregate=str(obj.get("aggregate", "out = s1")), raw=raw)

    @property
    def n_judge_steps(self) -> int:
        return sum(1 for s in self.steps if s["kind"].startswith("judge"))


_JUDGE_SCORE_SUFFIX = ('\nReturn ONLY JSON: {"score": <number in [0,1]>}.')
_JUDGE_EXTRACT_SUFFIX = ('\nReturn ONLY JSON: {"span": "<the extracted text, at most 300 '
                         'characters, or empty string if absent>"}.')


def _render(template: str, text: str, outs: Dict[str, list], i: int, max_chars: int) -> str:
    vals = {"text": text[:max_chars]}
    for sid, arr in outs.items():
        v = arr[i]
        vals[sid] = "" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)[:300])
    try:
        return template.format(**vals)
    except (KeyError, IndexError):
        return template.replace("{text}", vals["text"])


def _parse_json_field(resp: str, fld: str):
    try:
        s = resp[resp.find("{"): resp.rfind("}") + 1]
        return json.loads(s).get(fld)
    except Exception:
        m = re.search(r'"%s"\s*:\s*"?([^",}]+)' % fld, resp or "")
        return m.group(1) if m else None


def execute_program(
    prog: DeepMetricProgram,
    texts: List[str],
    judge_batch: Callable[[List[str]], List[str]],
    max_text_chars: int = 2800,
) -> np.ndarray:
    """Run one program over all texts, one batched judge call per judge step."""
    n = len(texts)
    outs: Dict[str, list] = {}
    for step in prog.steps:
        sid = step["id"]
        if step["kind"] == "code":
            outs[sid] = [safe_exec(step["code"],
                                   {**{k: v[i] for k, v in outs.items()},
                                    "text": texts[i][:max_text_chars]})
                         for i in range(n)]
            continue
        suffix = _JUDGE_SCORE_SUFFIX if step["kind"] == "judge_score" else _JUDGE_EXTRACT_SUFFIX
        prompts = [_render(step["prompt"], texts[i], outs, i, max_text_chars) + suffix
                   for i in range(n)]
        resps = judge_batch(prompts)
        if step["kind"] == "judge_score":
            vals = []
            for r in resps:
                v = _parse_json_field(r, "score")
                try:
                    vals.append(float(np.clip(float(v), 0.0, 1.0)))
                except (TypeError, ValueError):
                    vals.append(None)
            outs[sid] = vals
        else:
            outs[sid] = [str(_parse_json_field(r, "span") or "")[:300] for r in resps]
    final = np.full(n, np.nan)
    for i in range(n):
        ns = {k: v[i] for k, v in outs.items()}
        if any(v is None for v in ns.values()):
            continue
        v = safe_exec(prog.aggregate, {**ns, "text": texts[i][:max_text_chars]})
        if v is not None:
            try:
                final[i] = float(np.clip(float(v), 0.0, 1.0))
            except (TypeError, ValueError):
                pass
    return final


def flatten_program(prog: DeepMetricProgram) -> str:
    """One-shot rubric with the same semantics (the depth-premium comparator)."""
    parts = [prog.description]
    for s in prog.steps:
        if s["kind"].startswith("judge"):
            parts.append(s["prompt"].replace("{text}", "the text"))
    parts.append("Considering all of the above in ONE reading, score the text.")
    return " ".join(p.strip() for p in parts if p.strip())


PROGRAM_PROPOSER_PROMPT = """These items were labeled by a community. POSITIVES were labeled 1, \
NEGATIVES 0, but the known one-shot criteria below FAILED on these items — whatever separates \
them is not visible to a single-pass rubric.

KNOWN CRITERIA (already measured; do not restate):
{known}

POSITIVES (label 1):
{pos}

NEGATIVES (label 0):
{neg}

Propose {k} evaluation PROCEDURES (not one-shot rubrics) that could separate positives from \
negatives. Each procedure has 2-4 typed steps: "judge_score" (an LLM sub-judgment returning a \
number in [0,1]; prompt must contain {{text}}), "judge_extract" (extract a short span, e.g. the \
key claim or computation; prompt must contain {{text}}), or "code" (simple arithmetic over \
earlier step outputs s1..sN; assign to `out`). Good procedures VERIFY something (extract a \
claim then check it), COMPARE parts of the text, or measure a property that requires an \
intermediate extraction — things a single reading cannot score. Domain-specific procedures \
(checking the mathematics, the argument structure, the community conventions) are encouraged.

Return ONLY JSON:
{{"programs": [{{"name": str, "description": str, "steps": [{{"id": "s1", "kind": "judge_score"|"judge_extract"|"code", "prompt": str (judge kinds) , "code": str (code kind)}}], "aggregate": "out = <expression over s1..sN>"}}]}}"""
