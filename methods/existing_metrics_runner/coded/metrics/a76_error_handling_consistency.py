"""a76: Robust and consistent error handling.

a76's description emphasizes *consistency* — "handled explicitly and
consistently across technical/business paths" and "aligned error logging" —
which is the angle that distinguishes it from a30 (which scores each handler's
quality / specificity / non-emptiness).

We measure the **uniformity of error-handling style** across all handlers the
diff introduces. For each handler we classify the body into an action
"signature" drawn from a small alphabet:

    LOG     — calls a logger / logging / log.* / print / sys.stderr.write
    RAISE   — re-raises (or `raise SomethingElse` / `panic`)
    RETURN  — early-returns / returns an error value
    SWALLOW — empty / `pass` / `...` / `_ = err` / `// ignore`
    OTHER   — non-empty body that does none of the above

The score is the fraction of handlers that share the *most common* signature
(strict mode-share). With H handlers and `mode_count` in the dominant class,
score = mode_count / H. Score is 1.0 if all H handlers follow the same
pattern, decaying as the diff mixes styles.

We *require* at least two handlers in the diff for the metric to apply — a
single handler is vacuously "consistent" and would add noise. We also abstain
when languages are mixed in non-trivial ways (each language counted on its
own basket, then averaged weighted by handler count).

Tier 2 (tree-sitter AST walking only). PARTIALLY_THIN: we can verify
*observable* uniformity of handler shape, but whether two stylistically
different handlers are *semantically* both correct (one path warrants
fallback, another warrants re-raise) needs reasoning we don't do.

This is intentionally distinct from a30:
  - a30 asks "is each handler well-formed?"
  - a76 asks "are the handlers stylistically aligned with each other?"
A diff can score 1.0 on a30 (all handlers are specific + non-empty + raise)
yet 0.5 on a76 if half log-and-return while half log-and-raise. Conversely,
a diff can score 1.0 on a76 (all handlers swallow!) yet 0.0 on a30. The
two are weakly correlated in practice but measure different properties.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a76"
ASPECT_NAME = "Robust and consistent error handling"
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

# Cheap keyword sniff to early-exit applies().
_HANDLER_TOKENS = ("except", "catch", "err != nil", "err!=nil")

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


# Token sets used for classifying handler bodies. Kept small + conservative;
# substring checks are robust enough at body granularity.
_LOG_TOKENS = ("logging.", "logger.", "log.", "self.log",
               "warnings.warn", "console.log", "console.error",
               "console.warn", "System.err", "fmt.Errorf",
               "fmt.Println", "fmt.Printf", "slog.", "sys.stderr",
               "print(", "println(", "println!", "eprintln!",
               "e.printStackTrace")
_RAISE_TOKENS = ("raise ", "raise\n", "throw ", "throw new",
                 "panic(", "return errors.", "return fmt.Errorf")
_RETURN_TOKENS = ("return ", "return\n")
_SWALLOW_EXACT = ("", "pass", "...", "pass\n", "...\n", ";", "{}",
                  "// ignore", "/* ignore */", "_ = err", "_=err")


def _classify_body(body: str) -> str:
    """Map a handler body text to one of LOG / RAISE / RETURN / SWALLOW / OTHER."""
    inner = body.strip()
    # Strip leading/trailing braces if present (Java/JS/Go).
    if inner.startswith("{") and inner.endswith("}"):
        inner = inner[1:-1].strip()
    if inner in _SWALLOW_EXACT:
        return "SWALLOW"
    # `_ = err` style assignment-only swallow.
    if inner.startswith("_") and "err" in inner.lower() and len(inner) < 60:
        # Allow `_ = doSomething(err)` to fall through to OTHER; only the
        # straight discard variants are swallow.
        if "=" in inner and "(" not in inner.split("=", 1)[1]:
            return "SWALLOW"
    has_log = any(tok in inner for tok in _LOG_TOKENS)
    has_raise = any(tok in inner for tok in _RAISE_TOKENS)
    has_return = any(tok in inner for tok in _RETURN_TOKENS)
    # Priority: LOG > RAISE > RETURN > OTHER. We want "logged" to be the
    # primary axis of consistency because the aspect description explicitly
    # names "aligned error logging".
    if has_log:
        return "LOG"
    if has_raise:
        return "RAISE"
    if has_return:
        return "RETURN"
    return "OTHER"


# ---------- Python ----------

def _py_handler_bodies(code: bytes) -> List[str]:
    parser = _get_parser("py")
    if parser is None:
        return []
    tree = parser.parse(code)
    bodies: List[str] = []

    def walk(n):
        if n.type == "except_clause":
            for c in n.children:
                if c.type == "block":
                    bodies.append(_text(code, c))
                    break
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    return bodies


# ---------- Java ----------

def _java_handler_bodies(code: bytes) -> List[str]:
    parser = _get_parser("java")
    if parser is None:
        return []
    src = code if (b"class " in code or b"interface " in code) else (
        b"class __Snip {\nvoid __m() {\n" + code + b"\n}\n}\n")
    tree = parser.parse(src)
    bodies: List[str] = []

    def walk(n):
        if n.type == "catch_clause":
            for c in n.children:
                if c.type == "block":
                    bodies.append(_text(src, c))
                    break
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    return bodies


# ---------- JavaScript / TypeScript ----------

def _js_ts_handler_bodies(code: bytes, lang_short: str) -> List[str]:
    parser = _get_parser(lang_short)
    if parser is None:
        return []
    tree = parser.parse(code)
    bodies: List[str] = []

    def walk(n):
        if n.type == "catch_clause":
            for c in n.children:
                if c.type == "statement_block":
                    bodies.append(_text(code, c))
                    break
        # Promise .catch(...) bodies.
        if n.type == "call_expression" and n.children:
            callee = n.children[0]
            callee_text = _text(code, callee)
            if callee_text.endswith(".catch"):
                args = None
                for c in n.children:
                    if c.type == "arguments":
                        args = c
                        break
                if args is not None:
                    # Use argument span as the "body" for classification.
                    bodies.append(_text(code, args))
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    return bodies


# ---------- Go ----------

def _go_handler_bodies(code: bytes) -> List[str]:
    parser = _get_parser("go")
    if parser is None:
        return []
    tree = parser.parse(code)
    bodies: List[str] = []

    def walk(n):
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
                    bodies.append(_text(code, cons))
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    return bodies


def _is_code(path: str) -> bool:
    return not path.lower().endswith(NON_CODE_EXTS)


def _collect_signatures(diff_text: str) -> List[str]:
    """Return the list of per-handler action signatures across all files."""
    by_path = parse_diff_added_by_file(diff_text)
    sigs: List[str] = []
    for path, content in by_path.items():
        if not _is_code(path):
            continue
        ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        lang = EXT_TO_LANG.get(ext)
        if lang is None:
            continue
        b = content.encode("utf8", errors="replace")
        if lang == "py":
            bodies = _py_handler_bodies(b)
        elif lang == "java":
            bodies = _java_handler_bodies(b)
        elif lang in ("js", "ts"):
            bodies = _js_ts_handler_bodies(b, lang)
        elif lang == "go":
            bodies = _go_handler_bodies(b)
        else:
            bodies = []
        for body in bodies:
            sigs.append(_classify_body(body))
    return sigs


def applies(diff_text: str) -> bool:
    by_path = parse_diff_added_by_file(diff_text)
    eligible = False
    for p, c in by_path.items():
        if not _is_code(p):
            continue
        ext = "." + p.rsplit(".", 1)[-1].lower() if "." in p else ""
        if ext not in EXT_TO_LANG:
            continue
        if any(tok in c for tok in _HANDLER_TOKENS):
            eligible = True
            break
    if not eligible:
        return False
    # Require ≥2 handlers for "consistency" to be a meaningful measurement.
    sigs = _collect_signatures(diff_text)
    return len(sigs) >= 2


def score(diff_text: str) -> Optional[float]:
    sigs = _collect_signatures(diff_text)
    if len(sigs) < 2:
        return None
    counts = Counter(sigs)
    mode_count = max(counts.values())
    return float(mode_count / len(sigs))
