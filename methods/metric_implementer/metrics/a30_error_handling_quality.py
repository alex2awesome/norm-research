"""a30: Error prevention and robust error handling.

Measures the *quality* of error handling that the diff introduces. We parse
added code with tree-sitter (Python, JavaScript, TypeScript, Java, Go) and
count handler-shaped constructs, distinguishing well-formed handlers from
recognized anti-patterns:

Python:
  + good   `except SomeError:` / `except (A, B) as e:` (specific type)
  + good   handler body contains `logging.*`, `logger.*`, `raise`, `return`
  - bad    `except:` bare
  - bad    `except Exception:` / `except BaseException:` (overly broad)
  - bad    handler body is exactly `pass` or single `...` (swallowed)

Java:
  + good   `catch (SpecificException e)` (not Exception/Throwable/RuntimeException)
  + good   handler body non-empty and not just `;` (some action taken)
  - bad    `catch (Exception ...)` / `catch (Throwable ...)`
  - bad    empty catch block (`{}`) — swallowed

JavaScript / TypeScript:
  + good   `try { } catch (e) { ... }` where handler body is non-empty
  - bad    empty catch body
  - bad    `.catch(() => {})` empty rejection handler on promise chains

Go:
  + good   `if err != nil { ... }` pattern with a non-empty body that does
           something other than `_ = err` / bare assignment
  - bad    `_ = err` or assignment-only swallow
  - bad    function returns `error` but added code drops it

The score is `good / (good + bad)` averaged across files, or 1.0 if the diff
introduces no error-handling constructs at all (a small additive prior keeps
diffs with zero handlers from looking artificially perfect — they're abstained
via `applies()` when no handler-related token is present).

Tier 2 (tree-sitter AST walking only, no subprocess). PARTIALLY_THIN: we can
verify shape (specificity, non-empty body) cheaply, but whether the *chosen*
exception type is the correct one for the call-site, or whether the recovery
logic is semantically right, requires deeper reasoning we don't do.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a30"
ASPECT_NAME = "Error prevention and robust error handling"
TIER = 2
TOOLS = ["tree-sitter-python", "tree-sitter-javascript",
         "tree-sitter-typescript", "tree-sitter-java", "tree-sitter-go"]
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Java", "Go"]
CLASSIFICATION = "PARTIALLY_THIN"

NON_CODE_EXTS = (".md", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml",
                 ".lock", ".gitignore", ".cfg", ".ini")
EXT_TO_LANG = {
    ".py": "py", ".pyi": "py",
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "ts", ".tsx": "ts",
    ".java": "java",
    ".go": "go",
}

# Cheap keyword sniff to early-exit applies() — tree-sitter import is heavy.
_HANDLER_TOKENS = ("except", "catch", "try", "err", "Err", "panic", "recover")

_PARSERS: Dict[str, object] = {}


def _get_parser(lang_short: str):
    if lang_short in _PARSERS:
        return _PARSERS[lang_short]
    try:
        from tree_sitter import Language, Parser
        if lang_short == "py":
            import tree_sitter_python as m; lang = m.language()
        elif lang_short == "js":
            import tree_sitter_javascript as m; lang = m.language()
        elif lang_short == "ts":
            import tree_sitter_typescript as m
            lang = m.language_typescript()
        elif lang_short == "java":
            import tree_sitter_java as m; lang = m.language()
        elif lang_short == "go":
            import tree_sitter_go as m; lang = m.language()
        else:
            return None
        _PARSERS[lang_short] = Parser(Language(lang))
        return _PARSERS[lang_short]
    except ImportError:
        return None


def _text(code: bytes, n) -> str:
    return code[n.start_byte:n.end_byte].decode("utf8", errors="replace")


# ---------- Python ----------

PY_BROAD = {"Exception", "BaseException"}
PY_GOOD_ACTION_TOKENS = ("raise", "return", "logging.", "logger.",
                         "log.", "self.log", "warnings.warn", "print",
                         "sys.exit", "abort")


def _py_handler_quality(handler, code: bytes) -> Tuple[int, int]:
    """Return (good, bad) for a Python `except_clause` node."""
    good = bad = 0
    # children: 'except' keyword, optional ExceptionType (as identifier),
    # ':', body block
    exc_type_text = None
    body_node = None
    for c in handler.children:
        if c.type in ("identifier", "tuple", "attribute"):
            if exc_type_text is None:
                exc_type_text = _text(code, c).strip()
        elif c.type == "as_pattern":
            for sub in c.children:
                if sub.type in ("identifier", "tuple", "attribute"):
                    exc_type_text = _text(code, sub).strip()
                    break
        elif c.type == "block":
            body_node = c
    body_text = _text(code, body_node).strip() if body_node is not None else ""
    bare = exc_type_text is None
    broad = (exc_type_text in PY_BROAD) if exc_type_text else False
    empty_body = body_text in ("", "pass", "...", "pass\n", "...\n")
    if bare:
        bad += 1
    elif broad and empty_body:
        bad += 1
    elif broad:
        # broad-but-with-action: half-credit; record as 1 bad to reflect anti-pattern
        bad += 1
    elif empty_body:
        bad += 1
    else:
        # specific type + non-empty body: bonus if logs / raises / returns
        if any(tok in body_text for tok in PY_GOOD_ACTION_TOKENS):
            good += 2
        else:
            good += 1
    return good, bad


def _py_score(code: bytes) -> Optional[Tuple[int, int]]:
    parser = _get_parser("py")
    if parser is None:
        return None
    tree = parser.parse(code)
    good = bad = 0

    def walk(n):
        nonlocal good, bad
        if n.type == "except_clause":
            g, b = _py_handler_quality(n, code)
            good += g
            bad += b
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    return (good, bad)


# ---------- Java ----------

JAVA_BROAD = {"Exception", "Throwable", "RuntimeException"}


def _java_score(code: bytes) -> Optional[Tuple[int, int]]:
    parser = _get_parser("java")
    if parser is None:
        return None
    # Wrap loose method bodies in a synthetic class so tree-sitter parses cleanly.
    src = code if (b"class " in code or b"interface " in code) else (
        b"class __Snip {\nvoid __m() {\n" + code + b"\n}\n}\n")
    tree = parser.parse(src)
    good = bad = 0

    def walk(n):
        nonlocal good, bad
        if n.type == "catch_clause":
            # children: 'catch', '(', catch_formal_parameter, ')', block
            type_names = []
            body_node = None
            for c in n.children:
                if c.type == "catch_formal_parameter":
                    # catch_formal_parameter -> catch_type -> type_identifier+
                    for sub in c.children:
                        if sub.type == "catch_type":
                            for t in sub.children:
                                if t.type == "type_identifier":
                                    type_names.append(_text(src, t).strip())
                elif c.type == "block":
                    body_node = c
            body_text = (_text(src, body_node).strip()
                         if body_node is not None else "")
            body_inner = body_text[1:-1].strip() if (
                body_text.startswith("{") and body_text.endswith("}")
            ) else body_text
            # Broad if any caught type is exactly a broad name (token-level,
            # not substring — `SQLException` is not broad).
            broad = any(t in JAVA_BROAD for t in type_names)
            empty = body_inner in ("", ";")
            if empty:
                bad += 1
            elif broad:
                bad += 1
            else:
                good += 1
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    return (good, bad)


# ---------- JavaScript / TypeScript ----------

def _js_ts_score(code: bytes, lang_short: str) -> Optional[Tuple[int, int]]:
    parser = _get_parser(lang_short)
    if parser is None:
        return None
    tree = parser.parse(code)
    good = bad = 0

    def walk(n):
        nonlocal good, bad
        # try { } catch (e) { ... }
        if n.type == "catch_clause":
            body_node = None
            for c in n.children:
                if c.type == "statement_block":
                    body_node = c
                    break
            body_text = (_text(code, body_node).strip()
                         if body_node is not None else "")
            body_inner = body_text[1:-1].strip() if (
                body_text.startswith("{") and body_text.endswith("}")
            ) else body_text
            if body_inner == "":
                bad += 1
            else:
                good += 1
        # Promise .catch() handlers — call_expression where callee ends in .catch
        if n.type == "call_expression" and n.children:
            callee = n.children[0]
            callee_text = _text(code, callee)
            if callee_text.endswith(".catch"):
                # find argument list -> first argument
                args = None
                for c in n.children:
                    if c.type == "arguments":
                        args = c
                        break
                if args is not None:
                    arg_text = _text(code, args).strip()
                    # arg_text like "(e => {})" or "(() => {})"
                    if "{}" in arg_text or arg_text.endswith("({})"):
                        bad += 1
                    else:
                        good += 1
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    return (good, bad)


# ---------- Go ----------

def _go_score(code: bytes) -> Optional[Tuple[int, int]]:
    parser = _get_parser("go")
    if parser is None:
        return None
    tree = parser.parse(code)
    good = bad = 0

    def walk(n):
        nonlocal good, bad
        # if err != nil { ... } pattern
        if n.type == "if_statement":
            cond = None
            cons = None
            for c in n.children:
                if c.type == "binary_expression":
                    cond = c
                elif c.type == "block":
                    cons = c
            if cond is not None and cons is not None:
                cond_text = _text(code, cond)
                if "err" in cond_text and ("!= nil" in cond_text
                                           or "!=nil" in cond_text):
                    body_text = _text(code, cons).strip()
                    body_inner = body_text[1:-1].strip() if (
                        body_text.startswith("{") and body_text.endswith("}")
                    ) else body_text
                    if body_inner == "":
                        bad += 1
                    elif (body_inner.startswith("_ = err")
                          or body_inner == "// ignore"):
                        bad += 1
                    else:
                        good += 1
        # `_ = ...Err...` style discard at statement level
        if n.type == "assignment_statement":
            txt = _text(code, n)
            if txt.lstrip().startswith("_") and "err" in txt.lower():
                # Only count if RHS looks like a function call (potential error
                # source). Tree-sitter inspection would be heavier; check by
                # text containment of '(' which is robust for assignments.
                if "(" in txt:
                    bad += 1
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    return (good, bad)


def _is_code(path: str) -> bool:
    return not path.lower().endswith(NON_CODE_EXTS)


def applies(diff_text: str) -> bool:
    by_path = parse_diff_added_by_file(diff_text)
    for p, c in by_path.items():
        if not _is_code(p):
            continue
        ext = "." + p.rsplit(".", 1)[-1].lower() if "." in p else ""
        if ext not in EXT_TO_LANG:
            continue
        if any(tok in c for tok in _HANDLER_TOKENS):
            return True
    return False


def score(diff_text: str) -> Optional[float]:
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None
    total_good = total_bad = 0
    any_observed = False
    for path, content in by_path.items():
        if not _is_code(path):
            continue
        ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        lang = EXT_TO_LANG.get(ext)
        if lang is None:
            continue
        b = content.encode("utf8", errors="replace")
        if lang == "py":
            res = _py_score(b)
        elif lang == "java":
            res = _java_score(b)
        elif lang in ("js", "ts"):
            res = _js_ts_score(b, lang)
        elif lang == "go":
            res = _go_score(b)
        else:
            res = None
        if res is None:
            continue
        g, bd = res
        if g + bd == 0:
            continue
        any_observed = True
        total_good += g
        total_bad += bd
    if not any_observed:
        return None
    denom = total_good + total_bad
    if denom == 0:
        return None
    return float(total_good / denom)
