"""a112: Production observability and diagnostics.

The norm: code added to the diff uses proper telemetry infrastructure
(structured loggers, metrics, tracing) rather than ad-hoc `print` /
`console.log` debugging output. We can measure a structural proxy of this
across five languages by walking the AST with tree-sitter and counting:

  PROPER observability call sites
    Python  : `logging.<level>(...)`, `logger.<level>(...)` where <level> in
              debug/info/warning/error/critical/exception/log; also
              `prometheus_client.*` / `opentelemetry.*` calls
    Java    : `log.<level>(...)` / `logger.<level>(...)` /
              `LOGGER.<level>(...)` (slf4j/log4j idiom)
    JS/TS   : member calls on `winston`/`pino`/`bunyan`/`logger`/`log`
              objects (`.info`, `.debug`, `.warn`, `.error`, `.trace`,
              `.fatal`)
    Go      : `log.Print*` / `log.Fatal*` / `log.Panic*` from the stdlib
              "log" package; `zap` / `zerolog` / `logrus` calls

  ANTI-PATTERN bare-debug call sites (production anti-patterns)
    Python  : `print(...)`           (not inside a `__main__` guard — but
              we don't distinguish; bare `print` in a diff is the signal)
    JS/TS   : `console.log/.debug/.info/.warn/.error`
    Java    : `System.out.println` / `System.out.print` /
              `System.err.println` / `System.err.print`
    Go      : `fmt.Println` / `fmt.Printf` / `fmt.Print`

  IMPORT signal (additive bump): the diff imports one of the recognised
  observability libraries.

Score = (proper + 0.5 * import_signal) /
        (proper + bare + 0.5 * import_signal)
with a small Laplace term so an "all-proper" 1/0 file scores well but
doesn't pin to exactly 1.0. Abstain when neither proper nor bare nor
import signal is present (the diff isn't doing observability-relevant
work — score would be meaningless).

We *don't* exclude test files: test code that uses `print` for debugging is
still an anti-pattern signal at the prod-readiness level the norm cares about,
and most reviewers would flag it. We also don't try to verify that loggers
get *structured* metadata (key=value vs string concatenation) — that needs
semantic call-site reasoning across positional arguments, which is past
what a deterministic walk can claim. Hence PARTIALLY_THIN, not THIN.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a112"
ASPECT_NAME = "Production observability and diagnostics"
TIER = 2
TOOLS = ["tree-sitter-python", "tree-sitter-javascript",
         "tree-sitter-typescript", "tree-sitter-java", "tree-sitter-go"]
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Java", "Go"]
CLASSIFICATION = "PARTIALLY_THIN"

EXT_TO_LANG = {
    ".py": "py", ".pyi": "py",
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "ts", ".tsx": "ts",
    ".java": "java",
    ".go": "go",
}

# Standard log-level method names, applied as a member-call filter.
LOG_LEVELS = frozenset({
    "debug", "info", "warning", "warn", "error", "critical",
    "exception", "fatal", "trace", "log", "notice",
})

# Per-language observability libraries that, if imported, indicate the file
# is "doing observability". A direct import of any of these is a soft bonus.
OBSERVABILITY_IMPORTS = {
    "py": {
        "logging", "structlog", "loguru", "prometheus_client",
        "opentelemetry", "sentry_sdk", "ddtrace", "newrelic", "statsd",
    },
    "js": {
        "winston", "pino", "bunyan", "log4js", "@opentelemetry/api",
        "prom-client", "@sentry/node", "@sentry/browser", "datadog",
    },
    "ts": {
        "winston", "pino", "bunyan", "log4js", "@opentelemetry/api",
        "prom-client", "@sentry/node", "@sentry/browser", "datadog",
    },
    "java": {
        "org.slf4j", "ch.qos.logback", "org.apache.logging.log4j",
        "java.util.logging", "io.opentelemetry", "io.micrometer",
        "io.prometheus", "io.sentry",
    },
    "go": {
        "log", "log/slog",
        "go.uber.org/zap", "github.com/rs/zerolog",
        "github.com/sirupsen/logrus",
        "go.opentelemetry.io/otel",
        "github.com/prometheus/client_golang",
    },
}

# Names that, when used as the receiver of a method call (e.g.
# `logger.info(...)`), are recognised as logger handles.
LOGGER_NAMES = frozenset({
    "log", "logger", "LOG", "LOGGER", "logging",
    "slog", "winston", "pino", "bunyan", "zap", "zerolog", "logrus",
})

_PARSERS: Dict[str, object] = {}


def _get_parser(lang: str):
    if lang in _PARSERS:
        return _PARSERS[lang]
    try:
        from tree_sitter import Language, Parser
        if lang == "py":
            import tree_sitter_python as m; L = m.language()
        elif lang == "js":
            import tree_sitter_javascript as m; L = m.language()
        elif lang == "ts":
            import tree_sitter_typescript as m; L = m.language_typescript()
        elif lang == "java":
            import tree_sitter_java as m; L = m.language()
        elif lang == "go":
            import tree_sitter_go as m; L = m.language()
        else:
            return None
        _PARSERS[lang] = Parser(Language(L))
        return _PARSERS[lang]
    except ImportError:
        return None


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", errors="replace")


# ----- Python ----------------------------------------------------------------

def _py_collect(root, src: bytes) -> Tuple[int, int, bool]:
    """Return (proper, bare, has_obs_import) for a Python file AST."""
    proper = bare = 0
    has_import = False

    def walk(node):
        nonlocal proper, bare, has_import
        t = node.type

        if t in ("import_statement", "import_from_statement"):
            txt = _text(node, src)
            for lib in OBSERVABILITY_IMPORTS["py"]:
                if lib in txt:
                    has_import = True
                    break

        if t == "call":
            # function child is the callable expression
            fn = node.child_by_field_name("function")
            if fn is not None:
                fn_text = _text(fn, src)
                # Bare print(): identifier "print"
                if fn.type == "identifier" and fn_text == "print":
                    bare += 1
                # Member call: x.y(...)
                elif fn.type == "attribute":
                    # last identifier is the attribute name
                    method = None
                    base = None
                    for c in fn.children:
                        if c.type == "identifier":
                            if base is None:
                                base = _text(c, src)
                            method = _text(c, src)
                    if method is not None and method in LOG_LEVELS:
                        # logger.info / logging.warning / log.debug
                        if base in LOGGER_NAMES or base == "logging":
                            proper += 1
                    # Treat metric/tracer member calls as proper too
                    if base in ("metrics", "tracer", "meter", "counter",
                                "gauge", "histogram", "span"):
                        proper += 1
        for c in node.children:
            walk(c)

    walk(root)
    return proper, bare, has_import


# ----- Java ------------------------------------------------------------------

def _java_collect(root, src: bytes) -> Tuple[int, int, bool]:
    proper = bare = 0
    has_import = False

    def walk(node):
        nonlocal proper, bare, has_import
        t = node.type
        if t == "import_declaration":
            txt = _text(node, src)
            for lib in OBSERVABILITY_IMPORTS["java"]:
                if lib in txt:
                    has_import = True
                    break
        if t == "method_invocation":
            # children: [object?, ".", name(identifier), arguments]
            obj = node.child_by_field_name("object")
            nm = node.child_by_field_name("name")
            method = _text(nm, src) if nm is not None else ""

            if obj is not None:
                obj_text = _text(obj, src)
                # bare System.out.println / System.err.print
                if obj_text in ("System.out", "System.err") and \
                        method in ("println", "print", "printf"):
                    bare += 1
                # logger.info etc.
                else:
                    # extract trailing receiver identifier
                    last = obj_text.rsplit(".", 1)[-1]
                    if last in LOGGER_NAMES and method in LOG_LEVELS:
                        proper += 1
                    elif last in ("counter", "gauge", "histogram", "timer",
                                  "meter", "span", "tracer", "metrics"):
                        proper += 1
        for c in node.children:
            walk(c)

    walk(root)
    return proper, bare, has_import


# ----- JS / TS ---------------------------------------------------------------

def _js_collect(root, src: bytes) -> Tuple[int, int, bool]:
    proper = bare = 0
    has_import = False
    obs_set = OBSERVABILITY_IMPORTS["js"]  # same as ts

    def walk(node):
        nonlocal proper, bare, has_import
        t = node.type
        # ES import: import x from "winston"
        if t == "import_statement":
            txt = _text(node, src)
            for lib in obs_set:
                if f'"{lib}"' in txt or f"'{lib}'" in txt or \
                        f'"{lib}/' in txt or f"'{lib}/" in txt:
                    has_import = True
                    break
        # CommonJS: require("winston")
        if t == "call_expression":
            fn = node.child_by_field_name("function")
            args = node.child_by_field_name("arguments")
            if fn is not None and fn.type == "identifier" and \
                    _text(fn, src) == "require" and args is not None:
                a_txt = _text(args, src)
                for lib in obs_set:
                    if f'"{lib}"' in a_txt or f"'{lib}'" in a_txt:
                        has_import = True
                        break

            # Bare console.X(...) calls
            if fn is not None and fn.type == "member_expression":
                obj = fn.child_by_field_name("object")
                prop = fn.child_by_field_name("property")
                obj_txt = _text(obj, src) if obj is not None else ""
                prop_txt = _text(prop, src) if prop is not None else ""
                if obj_txt == "console" and prop_txt in (
                        "log", "debug", "info", "warn", "error", "trace"):
                    bare += 1
                # logger-like member calls
                elif obj_txt in LOGGER_NAMES and prop_txt in LOG_LEVELS:
                    proper += 1
                elif obj_txt in ("metrics", "tracer", "span", "counter",
                                 "gauge", "histogram", "Sentry"):
                    proper += 1
        for c in node.children:
            walk(c)

    walk(root)
    return proper, bare, has_import


# ----- Go --------------------------------------------------------------------

def _go_collect(root, src: bytes) -> Tuple[int, int, bool]:
    proper = bare = 0
    has_import = False

    def walk(node):
        nonlocal proper, bare, has_import
        t = node.type
        if t in ("import_spec", "import_declaration"):
            txt = _text(node, src)
            for lib in OBSERVABILITY_IMPORTS["go"]:
                q = f'"{lib}"'
                if q in txt or txt.strip().strip('"') == lib:
                    has_import = True
                    break
        if t == "call_expression":
            fn = node.child_by_field_name("function")
            if fn is not None and fn.type == "selector_expression":
                # x.y(...): operand is x, field is y
                operand = fn.child_by_field_name("operand")
                field = fn.child_by_field_name("field")
                base = _text(operand, src) if operand is not None else ""
                method = _text(field, src) if field is not None else ""
                # bare fmt.Println / Printf / Print
                if base == "fmt" and method in (
                        "Println", "Printf", "Print", "Fprintln",
                        "Fprintf", "Fprint"):
                    bare += 1
                # log.Print* / Fatal* / Panic*  (stdlib log)
                elif base == "log" and (
                        method.startswith("Print") or
                        method.startswith("Fatal") or
                        method.startswith("Panic") or
                        method in ("Debug", "Info", "Warn", "Error")):
                    proper += 1
                elif base in ("slog", "logger", "log", "zap", "zerolog",
                              "logrus") and method.lower() in (
                        "debug", "info", "warn", "warning", "error",
                        "fatal", "panic", "trace", "with", "named", "info",
                        "infof", "errorf", "warnf", "debugf"):
                    # Already covered for stdlib `log`; here covers zap/etc.
                    if base != "log":
                        proper += 1
        for c in node.children:
            walk(c)

    walk(root)
    return proper, bare, has_import


# ----- Dispatch --------------------------------------------------------------

def _path_lang(path: str) -> Optional[str]:
    p = path.lower()
    for ext, lang in EXT_TO_LANG.items():
        if p.endswith(ext):
            return lang
    return None


def _collect(code: bytes, lang: str) -> Optional[Tuple[int, int, bool]]:
    parser = _get_parser(lang)
    if parser is None:
        return None
    root = parser.parse(code).root_node
    if lang == "py":
        return _py_collect(root, code)
    if lang == "java":
        return _java_collect(root, code)
    if lang in ("js", "ts"):
        return _js_collect(root, code)
    if lang == "go":
        return _go_collect(root, code)
    return None


def applies(diff_text: str) -> bool:
    by_path = parse_diff_added_by_file(diff_text)
    return any(_path_lang(p) is not None for p in by_path)


def score(diff_text: str) -> Optional[float]:
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None

    tot_proper = tot_bare = 0
    n_import_files = 0
    saw_any = False

    for path, content in by_path.items():
        lang = _path_lang(path)
        if lang is None:
            continue
        res = _collect(content.encode("utf8", errors="replace"), lang)
        if res is None:
            continue
        proper, bare, has_import = res
        tot_proper += proper
        tot_bare += bare
        if has_import:
            n_import_files += 1
        saw_any = True

    if not saw_any:
        return None

    # No observability-relevant activity in the added code — abstain.
    if tot_proper == 0 and tot_bare == 0 and n_import_files == 0:
        return None

    import_bonus = 0.5 * min(n_import_files, 4)
    numer = tot_proper + import_bonus
    denom = tot_proper + tot_bare + import_bonus
    if denom == 0:
        return None
    return float(numer / denom)
