"""a155: Resource lifetime management and deterministic cleanup.

Norm: acquired resources (files, sockets, locks, DB cursors, etc.) must be
released deterministically on every path. The canonical idioms per language:

  Python:      `with open(...) as f:` / contextlib  (NOT bare `open(...)`)
  Java:        `try (Resource r = ...)` try-with-resources
               or `try { ... } finally { r.close(); }`
  Go:          `defer X.Close()` immediately after acquisition
  JavaScript/  `try { ... } finally { ... }` or `using` (TC39 stage 3)
  TypeScript:  Promise.finally chain

We parse added file contents with tree-sitter per language and count
resource-opening call sites, then classify each site as MANAGED vs.
UNMANAGED based on its enclosing structural context. Score is the fraction
MANAGED. This is a structural surrogate for the norm: a managed call is
*necessary* for deterministic cleanup but not strictly *sufficient* (the
managed scope could still leak via early return after acquisition, etc.).
That gap is why this is PARTIALLY_THIN, not THIN.

The signal is conservative: we score only call sites whose function
identifier matches a known resource-acquiring API. We never penalize a
diff that opens no resources — we abstain.

Reference idioms covered:

  Python   `open`, `socket.socket`, `tempfile.NamedTemporaryFile`,
           `threading.Lock().acquire`-ish... (we focus on explicit acquisition
           functions: open, NamedTemporaryFile, TemporaryFile, mkstemp,
           socket, connect — and any `.acquire()` call on a Lock-ish thing).
  Java     `new FileInputStream`, `new FileOutputStream`,
           `new BufferedReader`, `new FileReader`, `new Scanner(...)`,
           `Files.newBufferedReader`, `socket = new Socket(...)`,
           `DriverManager.getConnection`, `connection.prepareStatement`,
           `connection.createStatement`.
  Go       `os.Open`, `os.Create`, `os.OpenFile`, `net.Dial`,
           `net.Listen`, `sql.Open`, `bufio.NewReader(f)` (we treat the
           first Open as the resource — bufio wrappers around an already-
           open file don't add a leak path).
  JS/TS    `fs.open`, `fs.createReadStream`, `fs.createWriteStream`,
           `net.createConnection`, `new WebSocket`, `await db.connect()`
           (heuristic: any await/then on these returns a closeable).

Caveats and known false-positive sources:
  - `open()` used for read-only one-liner where the file handle is
    immediately consumed (e.g. `json.load(open(...))`) is technically
    unmanaged; it'll still flag as such. That's the correct behavior under
    this norm — the resource isn't deterministically closed.
  - In Go, `defer rows.Close()` placed after a non-trivial intervening
    block still counts as managed; we just require defer-Close in the
    same function body. Good enough for the norm.
  - In Java, returning a Closeable from a method (factory pattern) bypasses
    this metric. We don't try to model ownership transfer.

Tree-sitter is required; if any per-language parser is missing, the metric
abstains rather than guessing.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a155"
ASPECT_NAME = "Resource lifetime management and deterministic cleanup"
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

# Python resource-acquiring callees (last identifier of dotted call)
PY_RESOURCE_FUNCS = {
    "open", "socket", "create_connection", "NamedTemporaryFile",
    "TemporaryFile", "mkstemp", "mkdtemp", "TemporaryDirectory",
    "Popen",  # subprocess.Popen owns pipes
}
# Constructors that return Closeable in Java
JAVA_RESOURCE_CTORS = {
    "FileInputStream", "FileOutputStream", "FileReader", "FileWriter",
    "BufferedReader", "BufferedWriter", "BufferedInputStream",
    "BufferedOutputStream", "PrintWriter", "PrintStream", "Scanner",
    "DataInputStream", "DataOutputStream", "ObjectInputStream",
    "ObjectOutputStream", "InputStreamReader", "OutputStreamWriter",
    "Socket", "ServerSocket", "RandomAccessFile",
}
# Java static factory methods that return Closeable
JAVA_RESOURCE_FACTORIES = {
    "newBufferedReader", "newBufferedWriter", "newInputStream",
    "newOutputStream",  # java.nio.file.Files
    "getConnection",  # DriverManager
    "prepareStatement", "createStatement", "executeQuery",  # Connection
}
# Go stdlib resource openers (pkg.Func form)
GO_RESOURCE_FUNCS = {
    ("os", "Open"), ("os", "Create"), ("os", "OpenFile"),
    ("net", "Dial"), ("net", "DialTCP"), ("net", "DialUDP"),
    ("net", "Listen"), ("net", "ListenTCP"), ("net", "ListenUDP"),
    ("sql", "Open"),
    ("ioutil", "TempFile"), ("ioutil", "TempDir"),
    ("os", "CreateTemp"), ("os", "MkdirTemp"),
    ("http", "Get"), ("http", "Post"),  # responses need Body.Close
}
# JS/TS resource creators
JS_RESOURCE_CTORS = {"WebSocket"}
JS_RESOURCE_FUNCS = {
    "createReadStream", "createWriteStream",  # fs
    "createConnection", "connect",            # net / db
    "openSync", "open",                       # fs
}

_PARSERS: Dict[str, object] = {}


def _get_parser(lang: str):
    if lang in _PARSERS:
        return _PARSERS[lang]
    try:
        from tree_sitter import Language, Parser
        if lang == "py":
            import tree_sitter_python as m; ts_lang = m.language()
        elif lang == "js":
            import tree_sitter_javascript as m; ts_lang = m.language()
        elif lang == "ts":
            import tree_sitter_typescript as m; ts_lang = m.language_typescript()
        elif lang == "java":
            import tree_sitter_java as m; ts_lang = m.language()
        elif lang == "go":
            import tree_sitter_go as m; ts_lang = m.language()
        else:
            return None
        _PARSERS[lang] = Parser(Language(ts_lang))
        return _PARSERS[lang]
    except ImportError:
        return None


def _text(code: bytes, n) -> str:
    return code[n.start_byte:n.end_byte].decode("utf8", errors="replace")


# ---------- Python ----------

def _py_count(code: bytes) -> Tuple[int, int]:
    """Return (managed, unmanaged) resource-acquisition sites."""
    parser = _get_parser("py")
    if parser is None:
        return 0, 0
    tree = parser.parse(code)
    managed = unmanaged = 0

    def callee_name(call_node) -> Optional[str]:
        # `call` -> first child is function (identifier or attribute)
        if not call_node.children:
            return None
        fn = call_node.children[0]
        if fn.type == "identifier":
            return _text(code, fn)
        if fn.type == "attribute":
            # last child after a `.` is the attribute identifier
            for c in reversed(fn.children):
                if c.type == "identifier":
                    return _text(code, c)
        return None

    def in_with_block(node) -> bool:
        """True iff this call is the resource expression in a `with` clause."""
        cur = node.parent
        # Resource expr lives inside with_item -> with_clause -> with_statement
        while cur is not None:
            if cur.type == "with_clause" or cur.type == "with_statement":
                return True
            # Inside a function body / other block: stop traversing up.
            if cur.type in ("function_definition", "module", "class_definition"):
                return False
            cur = cur.parent
        return False

    def walk(node):
        nonlocal managed, unmanaged
        if node.type == "call":
            name = callee_name(node)
            if name in PY_RESOURCE_FUNCS:
                if in_with_block(node):
                    managed += 1
                else:
                    unmanaged += 1
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return managed, unmanaged


# ---------- Java ----------

def _java_count(code: bytes) -> Tuple[int, int]:
    parser = _get_parser("java")
    if parser is None:
        return 0, 0
    src = code if (b"class " in code or b"interface " in code
                   or b"record " in code) else (
        b"class __Snip {\n" + code + b"\n}\n")
    tree = parser.parse(src)
    managed = unmanaged = 0

    def in_try_with_resources(node) -> bool:
        cur = node.parent
        while cur is not None:
            if cur.type == "resource_specification":
                return True
            if cur.type in ("method_declaration", "constructor_declaration",
                            "class_body"):
                return False
            cur = cur.parent
        return False

    def has_finally_close(node, var_name: Optional[str]) -> bool:
        """Heuristic: is this acquisition inside a try whose finally calls
        close on the same variable? We do a simple containing-try lookup
        and grep the finally clause text for `.close(`. If we have a var
        name, require it to appear in the finally."""
        cur = node.parent
        while cur is not None:
            if cur.type == "try_statement":
                for c in cur.children:
                    if c.type == "finally_clause":
                        ftext = _text(src, c)
                        if ".close(" in ftext:
                            if var_name is None or var_name in ftext:
                                return True
                        return False
                return False
            if cur.type in ("method_declaration", "constructor_declaration"):
                return False
            cur = cur.parent
        return False

    def enclosing_var_name(node) -> Optional[str]:
        """If this acquisition is the RHS of a local variable, return name."""
        cur = node.parent
        # variable_declarator contains the identifier and the initializer
        while cur is not None and cur.type not in (
                "method_declaration", "constructor_declaration", "class_body"):
            if cur.type == "variable_declarator":
                for c in cur.children:
                    if c.type == "identifier":
                        return _text(src, c)
                return None
            cur = cur.parent
        return None

    def walk(node):
        nonlocal managed, unmanaged
        # `new Foo(...)` -> object_creation_expression
        if node.type == "object_creation_expression":
            type_name = None
            for c in node.children:
                if c.type in ("type_identifier", "scoped_type_identifier"):
                    type_name = _text(src, c).split(".")[-1]
                    break
            if type_name in JAVA_RESOURCE_CTORS:
                vn = enclosing_var_name(node)
                if in_try_with_resources(node) or has_finally_close(node, vn):
                    managed += 1
                else:
                    unmanaged += 1
        elif node.type == "method_invocation":
            # Files.newBufferedReader(...), conn.prepareStatement(...)
            method_name = None
            for c in node.children:
                if c.type == "identifier":
                    method_name = _text(src, c)
            if method_name in JAVA_RESOURCE_FACTORIES:
                vn = enclosing_var_name(node)
                if in_try_with_resources(node) or has_finally_close(node, vn):
                    managed += 1
                else:
                    unmanaged += 1
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return managed, unmanaged


# ---------- Go ----------

def _go_count(code: bytes) -> Tuple[int, int]:
    parser = _get_parser("go")
    if parser is None:
        return 0, 0
    tree = parser.parse(code)
    managed = unmanaged = 0

    def pkg_func(call_node) -> Optional[Tuple[str, str]]:
        if not call_node.children:
            return None
        fn = call_node.children[0]
        if fn.type == "selector_expression":
            parts = []
            for c in fn.children:
                if c.type in ("identifier", "field_identifier"):
                    parts.append(_text(code, c))
            if len(parts) >= 2:
                return parts[0], parts[1]
        return None

    def enclosing_func_body(node):
        cur = node.parent
        while cur is not None:
            if cur.type == "function_declaration" or \
               cur.type == "method_declaration":
                # find the block child
                for c in cur.children:
                    if c.type == "block":
                        return c
                return cur
            cur = cur.parent
        return None

    def assigned_var(node) -> Optional[str]:
        """The lhs name if this call is the RHS of a short var decl
        (`f, err := os.Open(...)`)."""
        cur = node.parent
        while cur is not None:
            if cur.type in ("short_var_declaration", "var_spec",
                            "assignment_statement"):
                # first identifier in expression_list on the lhs
                for c in cur.children:
                    if c.type == "expression_list":
                        for sc in c.children:
                            if sc.type == "identifier":
                                return _text(code, sc)
                        break
                return None
            cur = cur.parent
        return None

    def find_defer_close(func_body, var_name: Optional[str]) -> bool:
        """Walk the function body, return True if there's a
        `defer X.Close()` (where X matches the variable name if known)."""
        if func_body is None:
            return False
        found = [False]

        def w(n):
            if found[0]:
                return
            if n.type == "defer_statement":
                t = _text(code, n)
                if ".Close(" in t:
                    if var_name is None or var_name in t:
                        found[0] = True
                        return
            for c in n.children:
                w(c)
        w(func_body)
        return found[0]

    def walk(node):
        nonlocal managed, unmanaged
        if node.type == "call_expression":
            pf = pkg_func(node)
            if pf in GO_RESOURCE_FUNCS:
                fb = enclosing_func_body(node)
                vn = assigned_var(node)
                if find_defer_close(fb, vn):
                    managed += 1
                else:
                    unmanaged += 1
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return managed, unmanaged


# ---------- JS / TS ----------

def _js_count(code: bytes, lang: str) -> Tuple[int, int]:
    parser = _get_parser(lang)
    if parser is None:
        return 0, 0
    tree = parser.parse(code)
    managed = unmanaged = 0

    def callee_name(call_node) -> Optional[str]:
        if not call_node.children:
            return None
        fn = call_node.children[0]
        if fn.type == "identifier":
            return _text(code, fn)
        if fn.type == "member_expression":
            for c in reversed(fn.children):
                if c.type in ("property_identifier", "identifier"):
                    return _text(code, c)
        return None

    def in_try_block(node) -> bool:
        cur = node.parent
        while cur is not None:
            if cur.type == "try_statement":
                # require a finally clause
                for c in cur.children:
                    if c.type == "finally_clause":
                        return True
                return False
            if cur.type in ("function_declaration", "method_definition",
                            "arrow_function", "function"):
                return False
            cur = cur.parent
        return False

    def in_finally_chain(node) -> bool:
        """call.then(...).finally(...) pattern: walk up the chain and look
        for a `.finally(...)` call wrapping this one."""
        cur = node.parent
        while cur is not None:
            if cur.type == "call_expression":
                cn = callee_name(cur)
                if cn == "finally":
                    return True
            if cur.type in ("function_declaration", "method_definition",
                            "arrow_function", "function", "statement_block"):
                # leave block — stop
                break
            cur = cur.parent
        return False

    def walk(node):
        nonlocal managed, unmanaged
        if node.type == "new_expression":
            # new WebSocket(...)
            for c in node.children:
                if c.type == "identifier" and _text(code, c) in JS_RESOURCE_CTORS:
                    if in_try_block(node) or in_finally_chain(node):
                        managed += 1
                    else:
                        unmanaged += 1
                    break
        elif node.type == "call_expression":
            cn = callee_name(node)
            if cn in JS_RESOURCE_FUNCS:
                if in_try_block(node) or in_finally_chain(node):
                    managed += 1
                else:
                    unmanaged += 1
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return managed, unmanaged


# ---------- public API ----------

def _path_lang(path: str) -> Optional[str]:
    ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return EXT_TO_LANG.get(ext)


def applies(diff_text: str) -> bool:
    """True iff the diff adds code in a language we can parse."""
    by_path = parse_diff_added_by_file(diff_text)
    return any(_path_lang(p) is not None for p in by_path)


def score(diff_text: str) -> Optional[float]:
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None
    total_managed = total_unmanaged = 0
    for path, content in by_path.items():
        lang = _path_lang(path)
        if lang is None:
            continue
        b = content.encode("utf8", errors="replace")
        if lang == "py":
            m, u = _py_count(b)
        elif lang == "java":
            m, u = _java_count(b)
        elif lang == "go":
            m, u = _go_count(b)
        elif lang in ("js", "ts"):
            m, u = _js_count(b, lang)
        else:
            continue
        total_managed += m
        total_unmanaged += u
    total = total_managed + total_unmanaged
    if total == 0:
        # Norm doesn't fire on this diff — no resource acquisitions detected.
        return None
    return float(total_managed / total)
