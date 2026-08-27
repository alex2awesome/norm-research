"""a405: Pure-function-ness (Python).

For each function in the added code we test:
  - It declares no `global` or `nonlocal` statement.
  - Its body contains no augmented assignment (`+=`, `-=`, ...) to a name
    that is NOT also a parameter or a local assignment within the same
    function — i.e. no mutation of names defined outside.
  - Its body contains no calls to a small denylist of impure builtins/
    methods: print, input, open (top-level), os.system, subprocess.*,
    requests.*, .write(, .send(, .recv(, .commit(, time.sleep(.
  - It declares no `yield` (generators stream side-effecting iteration).

Score per file = fraction of functions that are pure. Score per diff = mean
across files. If no functions present, abstain.

Examples:
  + def add(x, y): return x + y                  -> pure
  + def f(x):
  +     print(x)
  +     return x                                  -> impure (print)
  + g = 0
  + def h(): global g; g += 1                     -> impure (global)
  + def i(): return [x*x for x in range(5)]       -> pure
  + def j(): time.sleep(1)                        -> impure

CLASSIFICATION: PARTIALLY_THIN — true purity requires whole-program
analysis (e.g. does this call eventually hit IO?). We catch direct,
syntactically-visible impurity only.
"""
from __future__ import annotations

from typing import List, Optional, Set

from ..sandbox import added_files_by_ext

ASPECT_ID = "a405"
ASPECT_NAME = "Pure-function fraction"
TIER = 2
TOOLS = ["tree-sitter-python"]
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "PARTIALLY_THIN"

PY_EXTS = [".py", ".pyi"]

IMPURE_IDENT_CALLS = {"print", "input", "open", "exec", "eval"}
IMPURE_ATTR_SUFFIXES = {
    "system", "popen", "Popen", "run",  # subprocess / os
    "write", "writelines", "send", "recv", "sendall",
    "commit", "rollback", "execute",  # db
    "remove", "unlink", "rename", "mkdir", "rmdir",  # filesystem
    "sleep",  # time.sleep
    "get", "post", "put", "delete", "patch",  # http when full path is requests.*
}
HTTP_MODULES = {"requests", "httpx", "urllib"}

_PARSER = None


def _get_parser():
    global _PARSER
    if _PARSER is None:
        try:
            import tree_sitter_python
            from tree_sitter import Language, Parser
            _PARSER = Parser(Language(tree_sitter_python.language()))
        except ImportError:
            return None
    return _PARSER


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", errors="replace")


def _params_of(fn_node, src: bytes) -> Set[str]:
    out: Set[str] = set()
    for c in fn_node.children:
        if c.type == "parameters":
            for p in c.children:
                if p.type == "identifier":
                    out.add(_text(p, src))
                elif p.type in ("typed_parameter", "default_parameter",
                                "typed_default_parameter",
                                "list_splat_pattern",
                                "dictionary_splat_pattern"):
                    for cc in p.children:
                        if cc.type == "identifier":
                            out.add(_text(cc, src))
                            break
    return out


def _local_assignments(node, src: bytes, locals_so_far: Set[str]):
    """Walk node, append identifiers ever appearing as plain assignment LHS."""
    if node.type == "assignment":
        lhs = node.children[0] if node.children else None
        if lhs is not None and lhs.type == "identifier":
            locals_so_far.add(_text(lhs, src))
        elif lhs is not None and lhs.type == "pattern_list":
            for c in lhs.children:
                if c.type == "identifier":
                    locals_so_far.add(_text(c, src))
    for c in node.children:
        _local_assignments(c, src, locals_so_far)


def _is_impure(fn_node, src: bytes) -> bool:
    body = None
    for c in fn_node.children:
        if c.type == "block":
            body = c
            break
    if body is None:
        return False  # no body -> can't decide; treat as pure

    params = _params_of(fn_node, src)
    locals_set: Set[str] = set(params)
    _local_assignments(body, src, locals_set)

    impure = False

    def walk(n):
        nonlocal impure
        if impure:
            return
        t = n.type
        if t in ("global_statement", "nonlocal_statement"):
            impure = True
            return
        if t == "yield":
            impure = True
            return
        if t == "augmented_assignment":
            lhs = n.children[0] if n.children else None
            if lhs is not None and lhs.type == "identifier":
                nm = _text(lhs, src)
                if nm not in locals_set:
                    impure = True
                    return
        if t == "call":
            first = n.children[0] if n.children else None
            if first is not None:
                if first.type == "identifier":
                    nm = _text(first, src)
                    if nm in IMPURE_IDENT_CALLS:
                        impure = True
                        return
                elif first.type == "attribute":
                    txt = _text(first, src)
                    base = txt.split(".", 1)[0] if "." in txt else ""
                    suffix = txt.rsplit(".", 1)[-1]
                    if base in HTTP_MODULES and suffix in IMPURE_ATTR_SUFFIXES:
                        impure = True
                        return
                    # generic attribute call: only flag known
                    # filesystem/db/time/network suffixes (not .append/.pop)
                    if suffix in ("write", "writelines", "send", "recv",
                                  "sendall", "commit", "rollback",
                                  "execute", "system", "popen", "Popen",
                                  "sleep", "unlink", "mkdir", "rmdir",
                                  "remove", "rename"):
                        impure = True
                        return
        for c in n.children:
            walk(c)

    walk(body)
    return impure


def _file_score(code: bytes) -> Optional[float]:
    parser = _get_parser()
    if parser is None:
        return None
    try:
        tree = parser.parse(code)
    except Exception:
        return None
    pure_count = 0
    total = 0

    def walk(n):
        nonlocal pure_count, total
        if n.type == "function_definition":
            total += 1
            if not _is_impure(n, code):
                pure_count += 1
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    if total == 0:
        return None
    return pure_count / total


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, PY_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, PY_EXTS)
    if not by_path:
        return None
    if _get_parser() is None:
        return None
    scs: List[float] = []
    for content in by_path.values():
        s = _file_score(content.encode("utf8", errors="replace"))
        if s is not None:
            scs.append(s)
    if not scs:
        return None
    return float(sum(scs) / len(scs))
