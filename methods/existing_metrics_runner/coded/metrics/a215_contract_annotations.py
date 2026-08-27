"""a215: Contract annotations (pre/postconditions, invariants).

Design-by-Contract (Meyer 1988) asks code to declare *what must hold* before
and after a routine runs, plus invariants kept across method calls. In modern
mainstream code this rarely takes the formal `@require`/`@ensure` form Eiffel
gave it — it shows up as decorators (icontract), schema validators (pydantic,
joi, zod), guard-style early returns (Go), or argument-precondition helpers
(Guava Preconditions, java.util.Objects.requireNonNull, assert with message).

We can detect these patterns structurally via tree-sitter and ratio them
against the number of added functions/methods to produce a *density* signal.
We mark PARTIALLY_THIN because:

  - The pattern set is recognizable but not exhaustive — bespoke project
    helpers (e.g. `validate_request(x)`) won't be caught.
  - Density correlates with contract discipline but doesn't measure *frame
    specifications* or postcondition coverage strictly — those would require
    JML-aware static analysis, which mainstream code rarely has.
  - An `assert(x)` with no message is ambiguous (sanity check vs. contract);
    we require a message or comparison-vs-None/null to count it.

Reference patterns (per language):

  Python (icontract / pydantic / hypothesis / argument asserts):
    - decorators: @require, @ensure, @invariant, @icontract.require, …
    - decorators: @validator, @field_validator, @root_validator, @model_validator
    - pydantic Field(...) calls inside class bodies (precondition schema)
    - assert with message: `assert x > 0, "x must be positive"`
    - early raise: `if not ...: raise ValueError(...)` at function head

  Java (JML / Guava / java.util.Objects):
    - annotations: @Requires, @Ensures, @Invariant, @Pure, @NonNull, @Nullable
    - method_invocation: Objects.requireNonNull, Preconditions.checkArgument,
      Preconditions.checkState, Preconditions.checkNotNull
    - throw IllegalArgumentException / NullPointerException at method head

  JS/TS (joi / zod / yup / superstruct / runtime-asserts):
    - import of joi, zod, yup, ajv, superstruct, runtypes, io-ts
    - call: invariant(...), assert(...), tiny-invariant(...)
    - schema.parse(x) / schema.validate(x) call expressions
    - throw new Error / TypeError at function head

  Go (idiomatic guard returns):
    - if-statement at start of a function body whose then-branch is a single
      return statement returning a non-nil error (errors.New, fmt.Errorf,
      identifier ending in 'Err'/'Error')

Score = clipped(sum_of_contract_sites / max(1, added_functions)) ∈ [0, 1],
with tanh squashing so density 0 → 0, 1 → ~0.76, 2+ → ~0.96.

Abstain (return None) when no source file with parseable functions is added
in any covered language.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a215"
ASPECT_NAME = "Contract annotations (pre/postconditions, invariants)"
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

# ----------------------------------------------------------------------------
# Pattern vocabularies
# ----------------------------------------------------------------------------

PY_DECORATOR_NAMES = {
    "require", "ensure", "invariant", "snapshot",
    "icontract.require", "icontract.ensure", "icontract.invariant",
    "validator", "field_validator", "root_validator", "model_validator",
    "validates", "validates_schema",
}

JAVA_ANNOTATION_NAMES = {
    "Requires", "Ensures", "Invariant", "Pure",
    "NonNull", "Nonnull", "NotNull", "Nullable",
    "Min", "Max", "NotEmpty", "NotBlank", "Size", "Valid",
    "Positive", "PositiveOrZero", "Negative", "NegativeOrZero",
    "Past", "Future", "Pattern", "AssertTrue", "AssertFalse",
}

JAVA_PRECONDITION_CALLS = {
    "Objects.requireNonNull",
    "Preconditions.checkArgument", "Preconditions.checkState",
    "Preconditions.checkNotNull", "Preconditions.checkElementIndex",
    "Preconditions.checkPositionIndex",
    "Assert.notNull", "Assert.isTrue", "Assert.hasText",
    "Validate.notNull", "Validate.notEmpty", "Validate.isTrue",
}

# Identifier-only call names (for unqualified invocations like requireNonNull(x))
JAVA_PRECONDITION_SIMPLE = {
    "requireNonNull", "checkArgument", "checkState", "checkNotNull",
}

JS_TS_CONTRACT_IMPORTS = {
    "joi", "@hapi/joi",
    "zod",
    "yup",
    "ajv", "ajv/dist/ajv",
    "superstruct",
    "runtypes",
    "io-ts",
    "class-validator",
    "tiny-invariant", "invariant",
    "assert", "node:assert",
}

JS_TS_CONTRACT_CALLS = {
    "invariant", "assert", "ok",
    "tinyInvariant",
}

# Method names that strongly suggest schema-driven validation
JS_TS_VALIDATE_METHODS = {"parse", "safeParse", "validate", "validateSync",
                          "validateAsync", "check", "assert"}

# ----------------------------------------------------------------------------
# Parser cache
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Python
# ----------------------------------------------------------------------------

def _py_count(code: bytes) -> Tuple[int, int]:
    """Return (n_functions, n_contract_sites) for Python source."""
    parser = _get_parser("py")
    if parser is None:
        return 0, 0
    tree = parser.parse(code)
    n_fns = 0
    n_sites = 0

    def walk(node, in_fn_head: bool = False):
        nonlocal n_fns, n_sites
        t = node.type
        if t == "function_definition":
            n_fns += 1
            # Look at decorators on the surrounding decorated_definition is
            # handled elsewhere; but tree-sitter-python attaches decorators
            # via parent decorated_definition. We'll match by walking parent.
            # Inspect first statements in body for guard-style raises.
            for c in node.children:
                if c.type == "block":
                    n_sites += _py_guard_raises(c, code)
                    n_sites += _py_assert_with_message(c, code)
                    break
        elif t == "decorator":
            name = _text(code, node).lstrip("@").split("(", 1)[0].strip()
            # canonical: drop module prefix
            tail = name.rsplit(".", 1)[-1]
            if name in PY_DECORATOR_NAMES or tail in PY_DECORATOR_NAMES:
                n_sites += 1
        elif t == "call":
            # pydantic Field(...) is treated as a precondition schema marker
            if node.children:
                head = _text(code, node.children[0])
                tail = head.rsplit(".", 1)[-1]
                if tail == "Field":
                    n_sites += 1
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return n_fns, n_sites


def _py_guard_raises(block_node, code: bytes) -> int:
    """Count `if cond: raise ExcType(...)` patterns inside a function body
    *before* the first non-guard statement. We approximate by counting all
    such patterns within the block (over-counts in long bodies, but density
    is still informative).
    """
    count = 0
    for c in block_node.children:
        if c.type == "if_statement":
            # find body
            for cc in c.children:
                if cc.type == "block":
                    for ccc in cc.children:
                        if ccc.type == "raise_statement":
                            count += 1
                            break
                    break
    return count


def _py_assert_with_message(block_node, code: bytes) -> int:
    """Count `assert <cond>, <message>` statements (the message turns a
    sanity-check assert into a contract-style assertion)."""
    count = 0
    for c in block_node.children:
        if c.type == "assert_statement":
            # tree-sitter-python emits children: 'assert', expr, ',', expr
            commas = sum(1 for ch in c.children if ch.type == ",")
            if commas >= 1:
                count += 1
    return count


# ----------------------------------------------------------------------------
# Java
# ----------------------------------------------------------------------------

def _java_count(code: bytes) -> Tuple[int, int]:
    parser = _get_parser("java")
    if parser is None:
        return 0, 0
    tree = parser.parse(code)
    n_fns = 0
    n_sites = 0

    def walk(node):
        nonlocal n_fns, n_sites
        t = node.type
        if t in ("method_declaration", "constructor_declaration"):
            n_fns += 1
            # scan annotations on this declaration
            for c in node.children:
                if c.type == "modifiers":
                    for sub in c.children:
                        if sub.type in ("marker_annotation", "annotation"):
                            for s2 in sub.children:
                                if s2.type == "identifier":
                                    if _text(code, s2) in JAVA_ANNOTATION_NAMES:
                                        n_sites += 1
                                    break
            # body: look for precondition calls + IllegalArgumentException throws
            for c in node.children:
                if c.type in ("block", "constructor_body"):
                    n_sites += _java_body_contract_sites(c, code)
                    break
        elif t == "field_declaration":
            # JSR-303 / bean validation annotations on fields count
            for c in node.children:
                if c.type == "modifiers":
                    for sub in c.children:
                        if sub.type in ("marker_annotation", "annotation"):
                            for s2 in sub.children:
                                if s2.type == "identifier":
                                    if _text(code, s2) in JAVA_ANNOTATION_NAMES:
                                        n_sites += 1
                                    break
        elif t == "formal_parameter":
            # @NonNull / @Valid on parameters
            for sub in node.children:
                if sub.type in ("marker_annotation", "annotation"):
                    for s2 in sub.children:
                        if s2.type == "identifier":
                            if _text(code, s2) in JAVA_ANNOTATION_NAMES:
                                n_sites += 1
                            break
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return n_fns, n_sites


def _java_body_contract_sites(block_node, code: bytes) -> int:
    count = 0

    def walk(n):
        nonlocal count
        if n.type == "method_invocation":
            txt = _text(code, n).split("(", 1)[0]
            # last two dotted segments form the canonical name
            parts = txt.split(".")
            short = parts[-1]
            qual2 = ".".join(parts[-2:]) if len(parts) >= 2 else short
            if qual2 in JAVA_PRECONDITION_CALLS or short in JAVA_PRECONDITION_SIMPLE:
                count += 1
        elif n.type == "throw_statement":
            txt = _text(code, n)
            if ("IllegalArgumentException" in txt
                    or "IllegalStateException" in txt
                    or "NullPointerException" in txt):
                count += 1
        elif n.type == "assert_statement":
            count += 1
        for c in n.children:
            walk(c)

    walk(block_node)
    return count


# ----------------------------------------------------------------------------
# JavaScript / TypeScript
# ----------------------------------------------------------------------------

def _js_ts_count(code: bytes, lang: str) -> Tuple[int, int]:
    parser = _get_parser(lang)
    if parser is None:
        return 0, 0
    tree = parser.parse(code)
    n_fns = 0
    n_sites = 0

    # Track contract-library imports per-file: if joi/zod/yup is imported,
    # every schema.parse / .validate / .check call counts as a contract site.
    has_contract_import = False

    def walk_pass1(node):
        nonlocal has_contract_import
        if node.type in ("import_statement", "import_declaration"):
            for c in node.children:
                if c.type == "string":
                    src = _text(code, c).strip("'\"`")
                    if src in JS_TS_CONTRACT_IMPORTS:
                        has_contract_import = True
                        return
                    # tolerate '@hapi/joi'-style scoped paths
                    head = src.split("/", 1)[0]
                    if head in JS_TS_CONTRACT_IMPORTS:
                        has_contract_import = True
                        return
        for c in node.children:
            walk_pass1(c)

    walk_pass1(tree.root_node)

    fn_node_types = {
        "function_declaration", "function_expression",
        "arrow_function", "method_definition",
        "function_signature", "generator_function_declaration",
    }

    def walk_pass2(node):
        nonlocal n_fns, n_sites
        t = node.type
        if t in fn_node_types:
            n_fns += 1
        elif t == "call_expression":
            head = node.children[0] if node.children else None
            if head is not None:
                head_text = _text(code, head)
                # invariant(...) / assert(...) / tinyInvariant(...)
                short = head_text.rsplit(".", 1)[-1]
                if short in JS_TS_CONTRACT_CALLS:
                    n_sites += 1
                # schema.parse(x) / schema.validate(x) — count only if we
                # have evidence of a contract library import in the file
                elif has_contract_import and "." in head_text:
                    method = head_text.rsplit(".", 1)[-1]
                    if method in JS_TS_VALIDATE_METHODS:
                        n_sites += 1
        elif t == "throw_statement":
            txt = _text(code, node)
            if ("new Error" in txt or "new TypeError" in txt
                    or "new RangeError" in txt):
                # only count if at function head — approximate by checking
                # ancestor depth: a throw inside an if at depth ~2 from the
                # function block is usually a guard. We over-count slightly.
                n_sites += 1
        # TS-only: decorators (class-validator @IsString, @Min, …)
        elif t == "decorator":
            txt = _text(code, node)
            short = txt.lstrip("@").split("(", 1)[0].rsplit(".", 1)[-1]
            if short in JAVA_ANNOTATION_NAMES:  # class-validator shares names
                n_sites += 1
        for c in node.children:
            walk_pass2(c)

    walk_pass2(tree.root_node)
    # Penalty: throw-counting over-fires on error-construction code. If the
    # file has zero contract imports and no library calls, but lots of
    # `throw new Error`, we may be miscounting normal error paths. Halve the
    # contribution by capping sites at 3 * n_fns + 1.
    if not has_contract_import:
        n_sites = min(n_sites, 3 * n_fns + 1)
    return n_fns, n_sites


# ----------------------------------------------------------------------------
# Go
# ----------------------------------------------------------------------------

def _go_count(code: bytes) -> Tuple[int, int]:
    parser = _get_parser("go")
    if parser is None:
        return 0, 0
    tree = parser.parse(code)
    n_fns = 0
    n_sites = 0

    def is_error_return(stmt_node) -> bool:
        """True iff the statement is `return ..., fmt.Errorf(...)` or
        `return ..., errors.New(...)` or returns a *Err identifier."""
        if stmt_node.type != "return_statement":
            return False
        txt = _text(code, stmt_node)
        return ("fmt.Errorf" in txt
                or "errors.New" in txt
                or "errors.Wrap" in txt
                or ", err" in txt and "return" in txt
                or "Err" in txt and ("return" in txt))

    def count_guards(block_node) -> int:
        count = 0
        for c in block_node.children:
            if c.type == "if_statement":
                # find then-block
                for cc in c.children:
                    if cc.type == "block":
                        # search for return inside (any depth=1)
                        for ccc in cc.children:
                            if ccc.type == "return_statement" and is_error_return(ccc):
                                count += 1
                                break
                        break
        return count

    def walk(node):
        nonlocal n_fns, n_sites
        if node.type in ("function_declaration", "method_declaration"):
            n_fns += 1
            for c in node.children:
                if c.type == "block":
                    n_sites += count_guards(c)
                    break
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return n_fns, n_sites


# ----------------------------------------------------------------------------
# Main dispatch
# ----------------------------------------------------------------------------

def _count_for(path: str, code: str) -> Optional[Tuple[int, int]]:
    ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
    lang = EXT_TO_LANG.get(ext)
    if not lang:
        return None
    code_bytes = code.encode("utf8", errors="replace")
    if lang == "py":
        return _py_count(code_bytes)
    if lang in ("js", "ts"):
        return _js_ts_count(code_bytes, lang)
    if lang == "java":
        return _java_count(code_bytes)
    if lang == "go":
        return _go_count(code_bytes)
    return None


def applies(diff_text: str) -> bool:
    by_path = parse_diff_added_by_file(diff_text)
    for path in by_path:
        ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext in EXT_TO_LANG:
            return True
    return False


def score(diff_text: str) -> Optional[float]:
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None
    total_fns = 0
    total_sites = 0
    any_parsed = False
    for path, code in by_path.items():
        res = _count_for(path, code)
        if res is None:
            continue
        any_parsed = True
        fns, sites = res
        total_fns += fns
        total_sites += sites
    if not any_parsed or total_fns == 0:
        # No parseable functions in covered langs → abstain rather than
        # claim 1.0 (no functions cannot violate contracts) or 0.0
        # (no functions cannot satisfy them either).
        return None
    density = total_sites / total_fns
    # tanh squashes: density 0 -> 0, 1 -> 0.76, 2 -> 0.96, plateau at 1.0.
    return float(math.tanh(density))


# Optional: expose internals for debug fixtures
__all__ = ["applies", "score", "ASPECT_ID", "ASPECT_NAME", "TIER", "TOOLS",
           "APPLIES_TO_LANGS", "CLASSIFICATION"]
