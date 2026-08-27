"""Shared C++ structural analyzer used by a600-a614 metrics.

Uses tree-sitter for C++ AST analysis. Falls back to validated regex when
tree-sitter is unavailable or parsing fails. All public functions return a
single feature dict for a single C++ source string.

Design:
  - One pass per source, results cached by source-id (hash) so 15 metrics
    that share one candidate don't reparse 15 times.
  - All numeric outputs are floats; boolean signals returned as 0.0/1.0.

Coverage scope: This is intended only for C++ (.cpp/.cc/.cxx/.h/...);
callers gate applies() via file extension.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, Optional

try:
    import tree_sitter
    import tree_sitter_cpp
    _LANG = tree_sitter.Language(tree_sitter_cpp.language())
    _TS_AVAILABLE = True
except Exception:
    _LANG = None
    _TS_AVAILABLE = False


CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h"]


# --- preprocessing helpers ------------------------------------------------- #

_STR_LIT = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')


def _strip_strings_comments(src: str) -> str:
    """Remove block comments, line comments, string literals. Keeps newlines."""
    # Block comments
    s = re.sub(r"/\*.*?\*/", " ", src, flags=re.DOTALL)
    # String / char literals
    s = _STR_LIT.sub(' "" ', s)
    out = []
    for ln in s.splitlines():
        i = ln.find("//")
        if i >= 0:
            ln = ln[:i]
        out.append(ln)
    return "\n".join(out)


# --- tree-sitter helpers -------------------------------------------------- #


def _parse_ts(src_bytes: bytes):
    if not _TS_AVAILABLE:
        return None
    try:
        p = tree_sitter.Parser(_LANG)
        return p.parse(src_bytes)
    except Exception:
        return None


def _walk(node):
    yield node
    for ch in node.children:
        yield from _walk(ch)


def _max_loop_depth(root) -> int:
    """Max nesting depth of for/while/do-while statements."""
    LOOP_TYPES = {"for_statement", "while_statement", "do_statement",
                  "for_range_loop"}
    best = [0]

    def rec(node, depth):
        cur = depth + 1 if node.type in LOOP_TYPES else depth
        if cur > best[0]:
            best[0] = cur
        for ch in node.children:
            rec(ch, cur)

    rec(root, 0)
    return best[0]


def _count_node_types(root, types):
    counts = {t: 0 for t in types}
    for n in _walk(root):
        if n.type in counts:
            counts[n.type] += 1
    return counts


# --- analyzer cache ------------------------------------------------------- #

_CACHE: Dict[str, Dict[str, float]] = {}
_CACHE_MAX = 1024


def _cache_key(src: str) -> str:
    return hashlib.md5(src.encode("utf-8", errors="ignore")).hexdigest()


def analyze(src: str) -> Optional[Dict[str, float]]:
    """Return a feature dict, or None if `src` is empty / non-text.

    Always returns the same key set so callers can pick a single key.
    """
    if not src or not src.strip():
        return None
    k = _cache_key(src)
    if k in _CACHE:
        return _CACHE[k]

    # nloc
    nloc = sum(1 for ln in src.splitlines() if ln.strip())
    # Stripped variant used for regex
    stripped = _strip_strings_comments(src)
    nloc_strip = max(1, sum(1 for ln in stripped.splitlines() if ln.strip()))

    feats: Dict[str, float] = {
        "nloc": float(nloc),
        "nloc_strip": float(nloc_strip),
    }

    # --- regex-based features (work even without tree-sitter) --- #

    # Boilerplate / macro densities
    macro_define_int = len(
        re.findall(r"#\s*define\s+int\s+long\s+long\b", stripped))
    typedef_ll = len(
        re.findall(r"\btypedef\s+long\s+long\s+\w+\s*;", stripped))
    define_rep = len(re.findall(r"#\s*define\s+(?:rep|REP|FOR|fr)\b", stripped))
    feats["boilerplate_count"] = float(
        macro_define_int + typedef_ll + define_rep)
    feats["boilerplate_density_100"] = (
        100.0 * feats["boilerplate_count"] / nloc_strip)

    # All #define count
    feats["macro_define_count"] = float(
        len(re.findall(r"^\s*#\s*define\b", stripped, re.MULTILINE)))
    feats["macro_density_100"] = (
        100.0 * feats["macro_define_count"] / nloc_strip)

    # Fast I/O
    feats["sync_with_stdio"] = float(bool(
        re.search(r"ios_base\s*::\s*sync_with_stdio|ios\s*::\s*sync_with_stdio",
                  stripped)))
    feats["cin_tie_zero"] = float(bool(
        re.search(r"cin\s*\.\s*tie\s*\(\s*0?\s*\)|cin\s*\.\s*tie\s*\(\s*NULL\s*\)|cin\s*\.\s*tie\s*\(\s*nullptr\s*\)",
                  stripped)))
    # scanf/printf are competitive C-style
    feats["scanf_printf_count"] = float(
        len(re.findall(r"\b(?:scanf|printf|getchar|putchar)\s*\(", stripped)))
    feats["cin_cout_count"] = float(
        len(re.findall(r"\b(?:cin|cout)\b", stripped)))

    # STL container sophistication
    feats["use_unordered"] = float(bool(
        re.search(r"\bunordered_(?:map|set|multimap|multiset)\b", stripped)))
    feats["use_ordered_map_set"] = float(bool(
        re.search(r"\b(?:std::)?(?:map|set|multimap|multiset)\s*<", stripped)))
    feats["use_priority_queue"] = float(bool(
        re.search(r"\bpriority_queue\b", stripped)))
    feats["use_deque"] = float(bool(re.search(r"\bdeque\b", stripped)))
    feats["use_bitset"] = float(bool(re.search(r"\bbitset\s*<", stripped)))

    # std:: qualification vs namespace std
    feats["using_namespace_std"] = float(bool(
        re.search(r"\busing\s+namespace\s+std\s*;", stripped)))
    feats["std_qual_count"] = float(
        len(re.findall(r"\bstd\s*::\s*", stripped)))
    feats["std_qual_density_100"] = (
        100.0 * feats["std_qual_count"] / nloc_strip)

    # Templates
    feats["template_def_count"] = float(
        len(re.findall(r"\btemplate\s*<", stripped)))

    # Lambda (regex pattern; tree-sitter will refine if available)
    feats["lambda_count_regex"] = float(
        len(re.findall(r"\[[^\]]*\]\s*\([^;{)]*\)\s*(?:->\s*[^{]+)?\s*\{",
                       stripped)))

    # 'auto' keyword usage
    feats["auto_count"] = float(
        len(re.findall(r"\bauto\b", stripped)))
    feats["auto_density_100"] = 100.0 * feats["auto_count"] / nloc_strip

    # Range-based for vs C-style for
    feats["range_for_count"] = float(
        len(re.findall(r"\bfor\s*\([^;()]*:\s*[^;()]+\)", stripped)))
    # Classic for: has two semicolons in the head
    classic_for = re.findall(r"\bfor\s*\(([^()]*;[^()]*;[^()]*)\)", stripped)
    feats["classic_for_count"] = float(len(classic_for))

    # bits/stdc++ vs granular includes
    feats["includes_bits"] = float(bool(
        re.search(r"#\s*include\s*[<\"]bits/stdc\+\+\.h[>\"]", stripped)))
    inc_lines = re.findall(r"#\s*include\s*[<\"][^>\"]+[>\"]", stripped)
    feats["include_count"] = float(len(inc_lines))
    feats["granular_include_count"] = float(
        len(inc_lines) - int(feats["includes_bits"]))

    # struct/class definitions
    feats["struct_class_def_count"] = float(len(re.findall(
        r"^\s*(?:struct|class)\s+[A-Za-z_]\w*[\s\{:]", stripped, re.MULTILINE)))

    # Global variables (regex fallback estimate via top-level
    # uninitialized/initialized variable declarations outside any { })
    # This will be refined by tree-sitter when available.

    # --- tree-sitter features ---------------------------------------- #
    feats["ts_used"] = 0.0
    feats["max_loop_depth"] = 0.0
    feats["self_call_recursion"] = 0.0
    feats["lambda_count_ts"] = 0.0
    feats["function_def_count_ts"] = 0.0
    feats["non_main_function_count"] = 0.0
    feats["global_var_count_ts"] = 0.0

    if _TS_AVAILABLE:
        tree = _parse_ts(src.encode("utf-8", errors="replace"))
        if tree is not None:
            feats["ts_used"] = 1.0
            root = tree.root_node
            feats["max_loop_depth"] = float(_max_loop_depth(root))

            # Functions defined at top-level: collect names + check self-call
            func_names = []
            func_bodies = []  # (name, body_node)
            for n in _walk(root):
                if n.type == "function_definition":
                    # Find the declarator's identifier
                    name = None
                    for ch in _walk(n):
                        if ch.type == "function_declarator":
                            for c2 in ch.children:
                                if c2.type == "identifier":
                                    name = c2.text.decode(
                                        "utf-8", errors="replace")
                                    break
                            if name:
                                break
                    if name is None:
                        # try a field_identifier (for method defs)
                        for ch in _walk(n):
                            if ch.type in ("identifier", "field_identifier"):
                                name = ch.text.decode(
                                    "utf-8", errors="replace")
                                break
                    func_names.append(name or "")
                    # Find compound_statement body
                    body = None
                    for ch in n.children:
                        if ch.type == "compound_statement":
                            body = ch
                            break
                    func_bodies.append((name, body))

            feats["function_def_count_ts"] = float(len(func_names))
            feats["non_main_function_count"] = float(sum(
                1 for n in func_names if n and n != "main"))

            # Self-call recursion: any function whose body invokes itself
            recurses = 0
            for nm, body in func_bodies:
                if not nm or body is None or nm == "main":
                    continue
                body_text = body.text.decode("utf-8", errors="replace")
                # Check name as a call site
                if re.search(rf"\b{re.escape(nm)}\s*\(", body_text):
                    recurses += 1
            feats["self_call_recursion"] = float(1.0 if recurses else 0.0)
            feats["recursive_func_count"] = float(recurses)

            # Lambda count (lambda_expression)
            feats["lambda_count_ts"] = float(sum(
                1 for n in _walk(root) if n.type == "lambda_expression"))

            # Global variable count: declaration nodes that are children of
            # translation_unit (not inside any function body)
            gv = 0
            for ch in root.children:
                if ch.type == "declaration":
                    gv += 1
            feats["global_var_count_ts"] = float(gv)

    else:
        # Regex fallback for max loop depth
        # Walk through lines tracking brace depth and { for/while occurrence
        depth = 0
        max_d = 0
        cur_d = 0
        brace_stack = []  # list of bool: was loop?
        i = 0
        s = stripped
        while i < len(s):
            ch = s[i]
            if ch == "{":
                # check whether the most recent token before { was a loop head
                head = s[max(0, i-80):i]
                is_loop = bool(re.search(
                    r"\b(?:for|while|do)\s*(?:\([^()]*\))?\s*$", head))
                brace_stack.append(is_loop)
                if is_loop:
                    cur_d += 1
                    if cur_d > max_d:
                        max_d = cur_d
            elif ch == "}":
                if brace_stack:
                    was_loop = brace_stack.pop()
                    if was_loop:
                        cur_d -= 1
            i += 1
        feats["max_loop_depth"] = float(max_d)
        # Lambda fallback uses regex count
        feats["lambda_count_ts"] = feats["lambda_count_regex"]
        # Functions / non-main count: use simple regex
        cpp_def = re.findall(
            r"^[ \t]*(?:(?:static|inline|virtual|constexpr|explicit|extern|"
            r"friend)\s+)*(?:[A-Za-z_][\w:<>,\s\*&]*\s+)"
            r"([A-Za-z_]\w*)\s*\([^;{)]*\)\s*(?:const\s*)?\{",
            stripped, re.MULTILINE)
        feats["function_def_count_ts"] = float(len(cpp_def))
        feats["non_main_function_count"] = float(
            sum(1 for n in cpp_def if n != "main"))
        recurses = 0
        for fn in cpp_def:
            if fn == "main":
                continue
            if len(re.findall(rf"\b{re.escape(fn)}\s*\(", stripped)) >= 2:
                recurses += 1
        feats["self_call_recursion"] = float(1.0 if recurses else 0.0)
        feats["recursive_func_count"] = float(recurses)
        # Global var fallback: count top-level lines that look like decls
        # (not includes, not pragmas, not blank); very rough.
        feats["global_var_count_ts"] = float(0.0)  # leave 0 in fallback

    # cache (with simple eviction)
    if len(_CACHE) >= _CACHE_MAX:
        # drop one arbitrary entry
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[k] = feats
    return feats


def looks_like_cpp(src: str) -> bool:
    """Cheap syntactic check on top of file-extension.

    Permissive: rejects only obviously-Python sources. The caller has already
    enforced the .cpp/.cc/.h extension via `added_files_by_ext`, so this is
    just a guard against misclassified candidates (e.g., a Python file
    mistakenly labeled as cpp). When in doubt, accept.
    """
    if not src or not src.strip():
        return False

    # Strong Python signals (rare in C++)
    has_python_hash_only = bool(re.search(
        r"^[ \t]*from\s+\w+\s+import\s+|^[ \t]*import\s+\w+\s*$",
        src, re.MULTILINE))
    has_python_def = bool(re.search(
        r"^[ \t]*def\s+[A-Za-z_]\w*\s*\([^)]*\)\s*:", src, re.MULTILINE))
    has_python_class = bool(re.search(
        r"^[ \t]*class\s+[A-Za-z_]\w*(?:\s*\([^)]*\))?\s*:", src, re.MULTILINE))
    py_signal = has_python_hash_only or has_python_def or has_python_class

    # Strong C++ signals (any one is enough)
    cpp_signals = (
        re.search(r"#\s*include\b", src) is not None,
        re.search(r"\busing\s+namespace\b", src) is not None,
        re.search(r"\bstd\s*::", src) is not None,
        re.search(r"\btemplate\s*<", src) is not None,
        re.search(r"\b(?:int|void)\s+main\s*\(", src) is not None,
        re.search(r"\bcout\s*<<|cin\s*>>", src) is not None,
        re.search(r"#\s*pragma\b", src) is not None,
        re.search(r"#\s*define\b", src) is not None,
        re.search(r"#\s*ifdef\b|#\s*ifndef\b", src) is not None,
    )
    cpp_signal = any(cpp_signals)

    if cpp_signal and not py_signal:
        return True
    if cpp_signal and py_signal:
        # ambiguous; favor accept if cpp signals dominate
        return True
    if not cpp_signal and py_signal:
        return False
    # Neither — could be a tiny snippet; accept by default since extension
    # already screened.
    return True
