"""a312: Robust server-side input validation and allowlisting.

The norm: "Validate all inbound data on the server with strict allowlists,
enforcing discrete-option fields/domain formats and duplicating client-side
checks as needed (avoid DNS-based validation)."

How this is distinct from sibling metrics:

  - a267 (Injection prevention) is a *negative* signal: it runs `bandit` and
    fires when the diff contains the patterns of *failed* validation
    (eval/exec, shell=True, raw SQL string concat, mark_safe, etc.). High
    a267 = "no vulnerable shapes added"; it says nothing about whether
    inbound data is positively validated.

  - a47 (Security hardening) is a coarse THICK rollup of security posture.

  - a215 (Contract annotations) is the closest sibling but measures
    pre/postcondition density anywhere in the codebase (Objects.requireNonNull,
    @require, assert with message, guard returns). It does NOT specifically
    detect *server-side input-validation frameworks* nor does it distinguish
    handler/endpoint code from internal helpers.

  - a312 is a *positive* signal that targets the specific norm by detecting
    that a recognized validation framework is (a) imported and (b) actually
    USED in the file — schema declarations, validator decorators, .parse()/
    .validate() calls. Frameworks recognized:

      Python: pydantic, marshmallow, cerberus, voluptuous, schema, jsonschema,
              djangorestframework serializers, django.forms, wtforms, attrs
              validators
      JS/TS:  joi/@hapi/joi, zod, yup, ajv, superstruct, runtypes, io-ts,
              class-validator, express-validator
      Java:   javax.validation / jakarta.validation (@Valid, @NotNull, @Size,
              ...), org.springframework.validation, hibernate-validator
      Go:     github.com/go-playground/validator, ozzo-validation,
              asaskevich/govalidator

We additionally up-weight files that look like server endpoints (handlers,
routes, controllers, views, serializers) so the signal aligns with
"server-side" rather than e.g. a build-script schema check.

Classification: PARTIALLY_THIN. We can reliably detect framework import + use
via tree-sitter, but cannot verify that validation is *strict* (allowlist vs.
denylist), that it covers *all* inbound fields, or that DNS-based validation
is avoided — those would need semantic understanding of every schema.

Score = tanh( sum_of_validation_sites / max(1, added_endpoint_files) ),
with endpoint files counted as 2x weight on sites (server-side focus).
0 sites → 0.0, 1 site/file → 0.76, 2+ → ~0.96.

Abstain (None) when no source file in a covered language is added.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a312"
ASPECT_NAME = "Robust server-side input validation and allowlisting"
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
# Framework registries
# ----------------------------------------------------------------------------

# Python: module -> set of marker symbols (decorators, class bases, callables)
PY_FRAMEWORK_IMPORTS = {
    "pydantic", "pydantic.v1",
    "marshmallow", "marshmallow.fields",
    "cerberus",
    "voluptuous",
    "schema",  # the `schema` PyPI package
    "jsonschema",
    "rest_framework", "rest_framework.serializers",
    "rest_framework.validators",
    "django.forms", "django.core.validators",
    "wtforms",
    "attrs", "attr",  # attrs validators
    "validators",  # `validators` pypi
    "pyrsistent",
}

# Python decorator / call markers that signal active validation USE
PY_VALIDATION_DECORATORS = {
    "validator", "field_validator", "root_validator", "model_validator",
    "validates", "validates_schema", "pre_load", "post_load",
    "validate_arguments", "validate_call",
}
PY_VALIDATION_CALLS = {
    "parse", "parse_obj", "model_validate", "model_validate_json",
    "validate", "load", "loads", "deserialize",
    "is_valid", "full_clean",
}
# Class-base markers — inheriting these means the class IS a schema
PY_SCHEMA_BASES = {
    "BaseModel", "Schema", "ModelSchema", "Serializer", "ModelSerializer",
    "Form", "ModelForm", "Validator",
}

# JS/TS frameworks
JS_TS_FRAMEWORK_IMPORTS = {
    "joi", "@hapi/joi",
    "zod",
    "yup",
    "ajv", "ajv-formats",
    "superstruct",
    "runtypes",
    "io-ts",
    "class-validator", "class-transformer",
    "express-validator",
    "validator",  # validator.js
    "fastest-validator",
    "@sinclair/typebox",
    "valibot",
}
JS_TS_VALIDATION_CALLS = {
    # zod / yup / joi / superstruct / runtypes shared method names
    "parse", "safeParse", "parseAsync", "safeParseAsync",
    "validate", "validateSync", "validateAsync",
    "check", "assert", "is", "cast",
    # express-validator
    "body", "query", "param", "header", "cookie", "checkSchema",
    "matchedData", "validationResult",
}
# class-validator decorators
JS_TS_VALIDATION_DECORATORS = {
    "IsString", "IsNumber", "IsInt", "IsBoolean", "IsArray", "IsObject",
    "IsNotEmpty", "IsOptional", "IsEmail", "IsUrl", "IsUUID", "IsEnum",
    "IsIn", "IsDate", "IsPositive", "IsNegative", "Min", "Max",
    "MinLength", "MaxLength", "Length", "Matches", "ValidateNested",
    "ArrayMinSize", "ArrayMaxSize", "IsDefined",
}

# Java
JAVA_FRAMEWORK_IMPORT_PREFIXES = (
    "javax.validation", "jakarta.validation",
    "org.springframework.validation",
    "org.hibernate.validator",
    "org.springframework.web.bind.annotation",  # @Valid commonly co-imported
    "org.apache.commons.validator",
)
JAVA_VALIDATION_ANNOTATIONS = {
    "Valid", "Validated",
    "NotNull", "NotEmpty", "NotBlank", "Null",
    "Size", "Min", "Max", "Range", "Digits",
    "Pattern", "Email", "URL",
    "Positive", "PositiveOrZero", "Negative", "NegativeOrZero",
    "AssertTrue", "AssertFalse",
    "Past", "PastOrPresent", "Future", "FutureOrPresent",
    "DecimalMin", "DecimalMax",
    # Spring controller method param annotations indicating server intake
    "RequestBody", "RequestParam", "PathVariable", "ModelAttribute",
}

# Go
GO_FRAMEWORK_IMPORT_SUBSTRINGS = (
    "go-playground/validator",
    "ozzo-validation",
    "asaskevich/govalidator",
    "go-ozzo/ozzo-validation",
    "thedevsaddam/govalidator",
)
GO_VALIDATION_CALL_TAILS = {
    "Struct", "StructCtx", "Var", "VarCtx", "Validate", "ValidateStruct",
    "RegisterValidation",
}

# ----------------------------------------------------------------------------
# Endpoint / server-side heuristics
# ----------------------------------------------------------------------------

ENDPOINT_PATH_HINTS = (
    "handler", "handlers", "route", "routes", "router", "routers",
    "controller", "controllers", "view", "views", "endpoint", "endpoints",
    "serializer", "serializers", "schema", "schemas", "api", "rest",
    "rpc", "graphql", "resolver", "resolvers",
)


def _is_endpoint_path(path: str) -> bool:
    p = path.lower()
    return any(h in p for h in ENDPOINT_PATH_HINTS)


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
            import tree_sitter_python as m
            lang = m.language()
        elif lang_short == "js":
            import tree_sitter_javascript as m
            lang = m.language()
        elif lang_short == "ts":
            import tree_sitter_typescript as m
            lang = m.language_typescript()
        elif lang_short == "java":
            import tree_sitter_java as m
            lang = m.language()
        elif lang_short == "go":
            import tree_sitter_go as m
            lang = m.language()
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

def _py_count(code: bytes) -> Tuple[bool, int]:
    """Return (has_framework_import, validation_sites)."""
    parser = _get_parser("py")
    if parser is None:
        return False, 0
    tree = parser.parse(code)
    has_import = False
    sites = 0

    def walk(node):
        nonlocal has_import, sites
        t = node.type
        if t in ("import_statement", "import_from_statement"):
            txt = _text(code, node)
            # crude module extraction: works for `import X`, `from X import Y`
            for tok in txt.replace(",", " ").split():
                # strip leading "from"/"import" and trailing punctuation
                tok = tok.strip().strip(":")
                if tok in ("import", "from", "as"):
                    continue
                # collapse to dotted module
                if tok in PY_FRAMEWORK_IMPORTS:
                    has_import = True
                # also match leading dotted prefix (e.g. pydantic.v1)
                head = tok.split(".", 1)[0]
                if head in {h.split(".", 1)[0] for h in PY_FRAMEWORK_IMPORTS}:
                    # but only set if it's a plausible whole token from a
                    # known root
                    if any(tok == fi or tok.startswith(fi + ".")
                           for fi in PY_FRAMEWORK_IMPORTS):
                        has_import = True
        elif t == "decorator":
            name = _text(code, node).lstrip("@").split("(", 1)[0].strip()
            tail = name.rsplit(".", 1)[-1]
            if tail in PY_VALIDATION_DECORATORS:
                sites += 1
        elif t == "class_definition":
            # check superclasses
            for c in node.children:
                if c.type == "argument_list":
                    txt = _text(code, c)
                    for base in PY_SCHEMA_BASES:
                        if base in txt:
                            sites += 1
                            break
        elif t == "call":
            if node.children:
                head = _text(code, node.children[0])
                tail = head.rsplit(".", 1)[-1]
                if tail in PY_VALIDATION_CALLS:
                    # only count if we have a framework import (avoid
                    # generic `.validate()` on app objects)
                    if has_import:
                        sites += 1
        for c in node.children:
            walk(c)

    # Two passes so imports are seen before calls
    def walk_imports(node):
        nonlocal has_import
        if node.type in ("import_statement", "import_from_statement"):
            txt = _text(code, node)
            for tok in txt.replace(",", " ").split():
                tok = tok.strip().strip(":")
                if tok in ("import", "from", "as"):
                    continue
                if tok in PY_FRAMEWORK_IMPORTS:
                    has_import = True
                if any(tok == fi or tok.startswith(fi + ".")
                       for fi in PY_FRAMEWORK_IMPORTS):
                    has_import = True
        for c in node.children:
            walk_imports(c)

    walk_imports(tree.root_node)
    walk(tree.root_node)
    return has_import, sites


# ----------------------------------------------------------------------------
# JS/TS
# ----------------------------------------------------------------------------

def _js_ts_count(code: bytes, lang: str) -> Tuple[bool, int]:
    parser = _get_parser(lang)
    if parser is None:
        return False, 0
    tree = parser.parse(code)
    has_import = False
    sites = 0

    def walk_imports(node):
        nonlocal has_import
        if node.type in ("import_statement", "import_declaration"):
            for c in node.children:
                if c.type == "string":
                    src = _text(code, c).strip("'\"`")
                    if src in JS_TS_FRAMEWORK_IMPORTS:
                        has_import = True
                        return
                    # scoped path: @hapi/joi -> head '@hapi/joi' itself; or
                    # subpath like 'zod/lib/x' -> head 'zod'
                    parts = src.split("/")
                    if src.startswith("@") and len(parts) >= 2:
                        head = "/".join(parts[:2])
                    else:
                        head = parts[0]
                    if head in JS_TS_FRAMEWORK_IMPORTS:
                        has_import = True
                        return
        elif node.type in ("call_expression",):
            # detect: const x = require('joi')
            ch = node.children
            if ch and _text(code, ch[0]) == "require":
                # arg list child
                for c in ch:
                    if c.type == "arguments":
                        for arg in c.children:
                            if arg.type == "string":
                                src = _text(code, arg).strip("'\"`")
                                if src in JS_TS_FRAMEWORK_IMPORTS:
                                    has_import = True
                                    return
                                head = src.split("/", 1)[0]
                                if (src.startswith("@") and "/" in src):
                                    head = "/".join(src.split("/")[:2])
                                if head in JS_TS_FRAMEWORK_IMPORTS:
                                    has_import = True
                                    return
        for c in node.children:
            walk_imports(c)

    walk_imports(tree.root_node)

    def walk(node):
        nonlocal sites
        t = node.type
        if t == "call_expression":
            head = node.children[0] if node.children else None
            if head is not None:
                head_text = _text(code, head)
                short = head_text.rsplit(".", 1)[-1]
                if short in JS_TS_VALIDATION_CALLS and has_import:
                    sites += 1
        elif t == "decorator":
            txt = _text(code, node)
            short = txt.lstrip("@").split("(", 1)[0].rsplit(".", 1)[-1]
            if short in JS_TS_VALIDATION_DECORATORS:
                sites += 1
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return has_import, sites


# ----------------------------------------------------------------------------
# Java
# ----------------------------------------------------------------------------

def _java_count(code: bytes) -> Tuple[bool, int]:
    parser = _get_parser("java")
    if parser is None:
        return False, 0
    tree = parser.parse(code)
    has_import = False
    sites = 0

    def walk_imports(node):
        nonlocal has_import
        if node.type == "import_declaration":
            txt = _text(code, node)
            for pref in JAVA_FRAMEWORK_IMPORT_PREFIXES:
                if pref in txt:
                    has_import = True
                    return
        for c in node.children:
            walk_imports(c)

    walk_imports(tree.root_node)

    def walk(node):
        nonlocal sites
        t = node.type
        if t in ("marker_annotation", "annotation"):
            # find the annotation identifier
            for c in node.children:
                if c.type == "identifier":
                    name = _text(code, c)
                    if name in JAVA_VALIDATION_ANNOTATIONS:
                        sites += 1
                    break
                if c.type == "scoped_identifier":
                    name = _text(code, c).rsplit(".", 1)[-1]
                    if name in JAVA_VALIDATION_ANNOTATIONS:
                        sites += 1
                    break
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return has_import, sites


# ----------------------------------------------------------------------------
# Go
# ----------------------------------------------------------------------------

def _go_count(code: bytes) -> Tuple[bool, int]:
    parser = _get_parser("go")
    if parser is None:
        return False, 0
    tree = parser.parse(code)
    has_import = False
    sites = 0

    def walk_imports(node):
        nonlocal has_import
        if node.type in ("import_declaration", "import_spec"):
            txt = _text(code, node)
            for hint in GO_FRAMEWORK_IMPORT_SUBSTRINGS:
                if hint in txt:
                    has_import = True
                    return
        for c in node.children:
            walk_imports(c)

    walk_imports(tree.root_node)

    def walk(node):
        nonlocal sites
        t = node.type
        if t == "call_expression":
            head = node.children[0] if node.children else None
            if head is not None:
                head_text = _text(code, head)
                tail = head_text.rsplit(".", 1)[-1]
                if tail in GO_VALIDATION_CALL_TAILS and has_import:
                    sites += 1
        # struct field tags carrying `validate:"..."`
        elif t == "field_declaration":
            txt = _text(code, node)
            if 'validate:"' in txt or "`validate:" in txt:
                # one site per tagged field (often multiple per struct)
                sites += 1
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return has_import, sites


# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------

def _count_for(path: str, code: str) -> Optional[Tuple[bool, int]]:
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
    total_sites = 0.0
    n_files_covered = 0
    n_endpoint_files = 0
    any_parsed = False
    for path, code in by_path.items():
        res = _count_for(path, code)
        if res is None:
            continue
        any_parsed = True
        n_files_covered += 1
        has_imp, sites = res
        # endpoint files contribute their sites at 1.5x — keeps the "server-
        # side" framing of the norm while not zeroing internal validation.
        weight = 1.5 if _is_endpoint_path(path) else 1.0
        if _is_endpoint_path(path):
            n_endpoint_files += 1
        # a framework import alone (no use) earns a half-site for the file
        # so files declaring schemas at module scope (no detected calls)
        # still get partial credit.
        per_file = sites + (0.5 if has_imp and sites == 0 else 0.0)
        total_sites += per_file * weight
    if not any_parsed:
        return None
    # normalize per covered file
    density = total_sites / max(n_files_covered, 1)
    return float(math.tanh(density))


__all__ = ["applies", "score", "ASPECT_ID", "ASPECT_NAME", "TIER", "TOOLS",
           "APPLIES_TO_LANGS", "CLASSIFICATION"]
