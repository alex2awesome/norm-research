"""Obfuscate Python codes for recoverability test.

Strips: docstrings, comments, replaces variable/function/parameter names with
generic identifiers (f1, x1, x2, ...). Keeps imports, string literals
(KEYWORDS!), control flow, and numeric constants — those are the LOGICAL
content of the metric.

The key question: can an LLM recover the rubric from the LOGIC alone, without
hint from semantic identifier names?

Input:  runs/validity_full/<run>/codegen_responses_qwen_r2/<key>.py
Output: runs/validity_full/<run>/codegen_responses_qwen_r2_obfuscated/<key>.py
"""
from __future__ import annotations

import argparse
import ast
import json
import string
import sys
from pathlib import Path


PROTECTED_NAMES = {
    # Keep these intact — they're not "metric semantics" but Python machinery
    "score", "text", "self", "True", "False", "None",
    "print", "len", "str", "int", "float", "bool", "list", "dict", "set", "tuple",
    "range", "enumerate", "zip", "map", "filter", "sorted", "reversed", "sum",
    "min", "max", "abs", "round", "any", "all", "isinstance", "type",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "ZeroDivisionError",
    # stdlib modules
    "re", "math", "json", "os", "sys", "string", "collections",
    "itertools", "functools", "operator", "statistics", "Counter",
    "defaultdict", "OrderedDict", "namedtuple",
    # re methods/objects
    "search", "findall", "match", "compile", "split", "sub", "MULTILINE",
    "IGNORECASE", "DOTALL", "VERBOSE", "I", "M", "S", "X",
    # common str methods
    "lower", "upper", "strip", "lstrip", "rstrip", "split", "join", "replace",
    "startswith", "endswith", "count", "find", "index", "title", "capitalize",
    # math/statistics functions
    "sqrt", "log", "exp", "floor", "ceil", "pow", "mean", "median", "stdev",
    "variance",
    # itertools
    "chain", "product", "combinations", "permutations",
}


def obfuscate_code(code: str) -> str:
    """Strip docstrings and comments, rename identifiers."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code  # leave bad codes alone

    # Collect identifiers to rename
    rename_map = {}
    counter = [1]
    def get_new_name(old):
        if old in PROTECTED_NAMES: return old
        if old in rename_map: return rename_map[old]
        new = f"v{counter[0]}"
        counter[0] += 1
        rename_map[old] = new
        return new

    # First pass: collect function/argument/local names to rename
    class Collector(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            # Rename function name (except 'score')
            if node.name not in PROTECTED_NAMES:
                get_new_name(node.name)
            for arg in node.args.args:
                if arg.arg not in PROTECTED_NAMES:
                    get_new_name(arg.arg)
            self.generic_visit(node)
        def visit_Assign(self, node):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id not in PROTECTED_NAMES:
                    get_new_name(tgt.id)
            self.generic_visit(node)
        def visit_For(self, node):
            if isinstance(node.target, ast.Name) and node.target.id not in PROTECTED_NAMES:
                get_new_name(node.target.id)
            self.generic_visit(node)
    Collector().visit(tree)

    # Second pass: rewrite identifiers, strip docstrings
    class Rewriter(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            # Rename function (unless protected)
            if node.name in rename_map:
                node.name = rename_map[node.name]
            # Rename arg names
            for arg in node.args.args:
                if arg.arg in rename_map:
                    arg.arg = rename_map[arg.arg]
            # Strip docstring
            if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:]
            self.generic_visit(node)
            return node
        def visit_Name(self, node):
            if node.id in rename_map:
                node.id = rename_map[node.id]
            return node
        def visit_Attribute(self, node):
            # Don't rename attributes (those are str method names etc.)
            self.generic_visit(node)
            return node

    new_tree = Rewriter().visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="full_v1")
    ap.add_argument("--src-subdir", default="codegen_responses_qwen_r2")
    ap.add_argument("--out-subdir", default="codegen_responses_qwen_r2_obfuscated")
    args = ap.parse_args()

    base = Path(f"runs/validity_full/{args.run_name}")
    src = base / args.src_subdir
    out = base / args.out_subdir
    out.mkdir(exist_ok=True, parents=True)

    n_ok = n_fail = 0
    for sp in sorted(src.glob("*.py")):
        try:
            obf = obfuscate_code(sp.read_text())
            (out / sp.name).write_text(obf)
            n_ok += 1
        except Exception as e:
            print(f"FAIL {sp.name}: {e}")
            n_fail += 1
    print(f"obfuscated {n_ok}/{n_ok+n_fail} -> {out}")


if __name__ == "__main__":
    main()
