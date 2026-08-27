"""Deterministic, structured code-review verifiers for the TRAIN pilot.

This module is the isolated ``V_ast`` implementation.  It deliberately does
not import scalar metric programs or any model-facing code.  Each real unit
implements a named, narrower relation inside its source CUF span; a passing
certificate therefore concerns that relation, not the whole metric.

Regular expressions are not used to decide source-code relations.  Paths are
classified by their POSIX segments and source is parsed with tree-sitter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable, Iterable, Iterator

from tree_sitter import Language, Node, Parser

from .diff_lines import DiffLine, parse_new_side_lines, validate_verdict_addresses
from .schema import Span, Verdict


SUPPORTED_SUFFIXES = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
}


@dataclass(frozen=True)
class UnitSpec:
    unit_id: str
    aspect_id: str | None
    source_cuf_node_id: int | None
    source_cuf_span: str
    implemented_relation: str
    relation_scope: str
    verifier: Callable[[str], Verdict]


@dataclass(frozen=True)
class ParsedAddedFile:
    path: str
    language: str
    source: bytes
    row_to_new_line: tuple[int, ...]
    root: Node

    def text(self, node: Node) -> str:
        return self.source[node.start_byte : node.end_byte].decode(
            "utf-8", errors="replace"
        )

    def span(self, node: Node, *, node_id: str | None = None) -> Span:
        """Return the first contiguous visible span for a parsed node.

        Prefer :meth:`spans` for verdict evidence: a reconstructed node may
        cross multiple diff hunks whose unshown lines cannot form one valid
        witness span.
        """

        return self.spans(node, node_id=node_id)[0]

    def spans(self, node: Node, *, node_id: str | None = None) -> tuple[Span, ...]:
        """Return all contiguous new-side added-line runs covered by ``node``."""

        start_row = node.start_point.row
        end_row = node.end_point.row
        # tree-sitter end positions are exclusive.  A node ending at column 0
        # belongs to the preceding row.
        if node.end_point.column == 0 and end_row > start_row:
            end_row -= 1
        if not (0 <= start_row < len(self.row_to_new_line)):
            raise ValueError("node starts outside reconstructed added source")
        end_row = min(max(end_row, start_row), len(self.row_to_new_line) - 1)
        addressed = self.row_to_new_line[start_row : end_row + 1]
        runs: list[tuple[int, int]] = []
        run_start = addressed[0]
        previous = addressed[0]
        for line in addressed[1:]:
            if line != previous + 1:
                runs.append((run_start, previous))
                run_start = line
            previous = line
        runs.append((run_start, previous))
        return tuple(
            Span(self.path, start, end, node_id=node_id) for start, end in runs
        )


_PARSERS: dict[str, Parser] = {}


def _parser(language: str) -> Parser:
    if language in _PARSERS:
        return _PARSERS[language]
    if language == "python":
        import tree_sitter_python as grammar

        capsule = grammar.language()
    elif language == "javascript":
        import tree_sitter_javascript as grammar

        capsule = grammar.language()
    elif language == "typescript":
        import tree_sitter_typescript as grammar

        capsule = grammar.language_typescript()
    elif language == "java":
        import tree_sitter_java as grammar

        capsule = grammar.language()
    elif language == "go":
        import tree_sitter_go as grammar

        capsule = grammar.language()
    else:  # pragma: no cover - guarded by SUPPORTED_SUFFIXES
        raise ValueError(f"unsupported language: {language}")
    parser = Parser(Language(capsule))
    _PARSERS[language] = parser
    return parser


def _group_added_lines(diff_text: str) -> dict[str, list[DiffLine]]:
    grouped: dict[str, list[DiffLine]] = {}
    for line in parse_new_side_lines(diff_text):
        if line.added:
            grouped.setdefault(line.path, []).append(line)
    return grouped


def parse_added_files(diff_text: str) -> tuple[ParsedAddedFile, ...]:
    """Parse supported added source while preserving new-side line addresses.

    Added lines from separate hunks are concatenated in source order, exactly
    as the historical metric runner does.  The explicit row map prevents that
    reconstruction choice from corrupting source witnesses.  Files whose
    reconstructed source has a parse error are excluded: this verifier does
    not infer structure from malformed fragments.
    """

    files: list[ParsedAddedFile] = []
    for path, lines in _group_added_lines(diff_text).items():
        language = SUPPORTED_SUFFIXES.get(PurePosixPath(path).suffix.lower())
        if language is None:
            continue
        source = "\n".join(line.text for line in lines).encode("utf-8")
        root = _parser(language).parse(source).root_node
        if root.has_error:
            continue
        files.append(
            ParsedAddedFile(
                path=path,
                language=language,
                source=source,
                row_to_new_line=tuple(line.line for line in lines),
                root=root,
            )
        )
    return tuple(files)


def _walk(node: Node) -> Iterator[Node]:
    yield node
    for child in node.named_children:
        yield from _walk(child)


FUNCTION_TYPES = {
    "python": {"function_definition"},
    "javascript": {
        "function_declaration",
        "generator_function_declaration",
        "method_definition",
        "arrow_function",
    },
    "typescript": {
        "function_declaration",
        "generator_function_declaration",
        "method_definition",
        "arrow_function",
    },
    "java": {"method_declaration", "constructor_declaration"},
    "go": {"function_declaration", "method_declaration", "func_literal"},
}

CONTROL_TYPES = {
    "python": {
        "if_statement",
        "for_statement",
        "while_statement",
        "match_statement",
        "try_statement",
    },
    "javascript": {
        "if_statement",
        "for_statement",
        "for_in_statement",
        "while_statement",
        "do_statement",
        "switch_statement",
        "try_statement",
    },
    "typescript": {
        "if_statement",
        "for_statement",
        "for_in_statement",
        "while_statement",
        "do_statement",
        "switch_statement",
        "try_statement",
    },
    "java": {
        "if_statement",
        "for_statement",
        "enhanced_for_statement",
        "while_statement",
        "do_statement",
        "switch_expression",
        "try_statement",
    },
    "go": {
        "if_statement",
        "for_statement",
        "expression_switch_statement",
        "type_switch_statement",
        "select_statement",
    },
}


def _functions(file: ParsedAddedFile) -> Iterator[Node]:
    function_types = FUNCTION_TYPES[file.language]
    yield from (node for node in _walk(file.root) if node.type in function_types)


def _deep_control_node(file: ParsedAddedFile, minimum_depth: int = 3) -> Node | None:
    controls = CONTROL_TYPES[file.language]

    def descend(node: Node, depth: int, inside_function: bool) -> Node | None:
        if node.type in FUNCTION_TYPES[file.language]:
            depth = 0
            inside_function = True
        elif inside_function and node.type in controls:
            depth += 1
            if depth >= minimum_depth:
                return node
        for child in node.named_children:
            found = descend(child, depth, inside_function)
            if found is not None:
                return found
        return None

    return descend(file.root, 0, False)


def verify_control_nesting(diff_text: str) -> Verdict:
    """Violate when a function has three nested structured control nodes."""

    applicability_witness: tuple[Span, ...] = ()
    for file in parse_added_files(diff_text):
        controls = CONTROL_TYPES[file.language]
        if not applicability_witness:
            first_control = next(
                (
                    descendant
                    for function in _functions(file)
                    for descendant in _walk(function)
                    if descendant.type in controls
                ),
                None,
            )
            if first_control is not None:
                applicability_witness = file.spans(
                    first_control, node_id=f"{file.language}:control"
                )
        violating = _deep_control_node(file)
        if violating is not None:
            verdict = Verdict(
                True,
                True,
                file.spans(violating, node_id=f"{file.language}:nested-control"),
            )
            validate_verdict_addresses(diff_text, verdict, require_added=True)
            return verdict
    verdict = Verdict(bool(applicability_witness), False, applicability_witness)
    validate_verdict_addresses(diff_text, verdict, require_added=True)
    return verdict


def verify_conditional_nesting(diff_text: str) -> Verdict:
    """Positive control: three nested conditionals, excluding other controls."""

    applicability_witness: tuple[Span, ...] = ()
    for file in parse_added_files(diff_text):
        conditionals = {"if_statement"}

        def descend(node: Node, depth: int, inside_function: bool) -> Node | None:
            if node.type in FUNCTION_TYPES[file.language]:
                depth = 0
                inside_function = True
            elif inside_function and node.type in conditionals:
                depth += 1
                if depth >= 3:
                    return node
            for child in node.named_children:
                found = descend(child, depth, inside_function)
                if found is not None:
                    return found
            return None

        first_if = next(
            (
                descendant
                for function in _functions(file)
                for descendant in _walk(function)
                if descendant.type == "if_statement"
            ),
            None,
        )
        if first_if is not None and not applicability_witness:
            applicability_witness = file.spans(
                first_if, node_id=f"{file.language}:conditional"
            )
        violating = descend(file.root, 0, False)
        if violating is not None:
            verdict = Verdict(
                True,
                True,
                file.spans(
                    violating, node_id=f"{file.language}:nested-conditional"
                ),
            )
            validate_verdict_addresses(diff_text, verdict, require_added=True)
            return verdict
    verdict = Verdict(bool(applicability_witness), False, applicability_witness)
    validate_verdict_addresses(diff_text, verdict, require_added=True)
    return verdict


def _parameter_count(function: Node) -> int:
    parameters = function.child_by_field_name("parameters")
    if parameters is None:
        # Go exposes the parameter list as a named direct child rather than a
        # stable field across grammar versions.
        parameters = next(
            (
                child
                for child in function.named_children
                if child.type in {"parameter_list", "formal_parameters"}
            ),
            None,
        )
    if parameters is None:
        return 0
    return len(parameters.named_children)


def verify_maintainability_smells(diff_text: str) -> Verdict:
    """Implement the explicit long-function/long-parameter-list sub-relation.

    Applicability excludes trivial functions: at least one added function must
    contain ten source lines or one declared parameter.  A function violates
    at eight parameter declarations or eighty source lines.  These thresholds
    are declared anchors, not a weighted or continuous score.
    """

    applicability_witness: tuple[Span, ...] = ()
    for file in parse_added_files(diff_text):
        for function in _functions(file):
            line_count = function.end_point.row - function.start_point.row + 1
            parameters = _parameter_count(function)
            if (line_count >= 10 or parameters >= 1) and not applicability_witness:
                applicability_witness = file.spans(
                    function, node_id=f"{file.language}:assessed-function"
                )
            if parameters >= 8 or line_count >= 80:
                verdict = Verdict(
                    True,
                    True,
                    file.spans(
                        function,
                        node_id=(
                            f"{file.language}:function:"
                            f"params={parameters}:lines={line_count}"
                        ),
                    ),
                )
                validate_verdict_addresses(diff_text, verdict, require_added=True)
                return verdict
    verdict = Verdict(bool(applicability_witness), False, applicability_witness)
    validate_verdict_addresses(diff_text, verdict, require_added=True)
    return verdict


def _python_top_declarations(file: ParsedAddedFile) -> tuple[Node, ...]:
    declarations: list[Node] = []
    for child in file.root.named_children:
        node = child
        if child.type == "decorated_definition":
            node = next(
                (
                    nested
                    for nested in child.named_children
                    if nested.type in {"function_definition", "class_definition"}
                ),
                child,
            )
        if node.type in {"function_definition", "class_definition"}:
            declarations.append(node)
    return tuple(declarations)


def _declared_name(file: ParsedAddedFile, declaration: Node) -> str | None:
    name = declaration.child_by_field_name("name")
    if name is None:
        name = next(
            (child for child in declaration.named_children if child.type == "identifier"),
            None,
        )
    return file.text(name) if name is not None else None


def _python_declares_all(file: ParsedAddedFile) -> bool:
    for child in file.root.named_children:
        if child.type != "expression_statement":
            continue
        for node in _walk(child):
            if node.type == "identifier" and file.text(node) == "__all__":
                return True
    return False


def verify_python_visibility_boundary(diff_text: str) -> Verdict:
    """Violate a broad Python public surface lacking an explicit boundary."""

    applicability_witness: tuple[Span, ...] = ()
    for file in parse_added_files(diff_text):
        if file.language != "python":
            continue
        declarations = _python_top_declarations(file)
        if not declarations:
            continue
        if not applicability_witness:
            applicability_witness = file.spans(
                declarations[0], node_id="python:module-declaration"
            )
        names = [(_declared_name(file, node), node) for node in declarations]
        public = [(name, node) for name, node in names if name and not name.startswith("_")]
        private = [(name, node) for name, node in names if name and name.startswith("_")]
        if len(public) >= 4 and not private and not _python_declares_all(file):
            witnesses = tuple(
                span
                for name, node in public
                for span in file.spans(
                    node, node_id=f"python:public-declaration:{name}"
                )
            )
            verdict = Verdict(True, True, witnesses)
            validate_verdict_addresses(diff_text, verdict, require_added=True)
            return verdict
    verdict = Verdict(bool(applicability_witness), False, applicability_witness)
    validate_verdict_addresses(diff_text, verdict, require_added=True)
    return verdict


_E2E_PATH_SEGMENTS = {
    "e2e",
    "ui",
    "system",
    "browser",
    "smoke",
    "end-to-end",
    "end_to_end",
    "acceptance",
}
_INTEGRATION_PATH_SEGMENTS = {
    "integration",
    "functional",
    "contract",
    "api",
    "component",
}
_UNIT_PATH_SEGMENTS = {"unit", "__tests__", "spec", "specs", "test", "tests"}
_E2E_IMPORT_PREFIXES = {
    "selenium",
    "playwright",
    "pytest_playwright",
    "@playwright/test",
    "cypress",
    "puppeteer",
    "webdriverio",
    "selenium-webdriver",
    "com.microsoft.playwright",
    "org.openqa.selenium",
    "github.com/chromedp/chromedp",
}
_INTEGRATION_IMPORT_PREFIXES = {
    "requests",
    "httpx",
    "fastapi.testclient",
    "starlette.testclient",
    "django.test",
    "rest_framework.test",
    "supertest",
    "testcontainers",
    "org.testcontainers",
    "io.restassured",
    "net/http/httptest",
}
_UNIT_IMPORT_PREFIXES = {
    "unittest.mock",
    "pytest_mock",
    "sinon",
    "jest-mock",
    "org.mockito",
    "github.com/stretchr/testify/mock",
}


def _literal_module(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] in "'\"`" and text[-1] == text[0]:
        return text[1:-1]
    return text


def _structured_imports(file: ParsedAddedFile) -> tuple[tuple[str, Node], ...]:
    imports: list[tuple[str, Node]] = []
    for node in _walk(file.root):
        if file.language == "python" and node.type in {
            "import_statement",
            "import_from_statement",
        }:
            module = node.child_by_field_name("module_name")
            candidates = [module] if module is not None else list(node.named_children)
            for candidate in candidates:
                if candidate is None:
                    continue
                if candidate.type in {"dotted_name", "identifier", "aliased_import"}:
                    value = file.text(candidate).split(" as ", 1)[0].strip()
                    if value not in {"from", "import"}:
                        imports.append((value, candidate))
                        break
        elif file.language in {"javascript", "typescript"} and node.type == "import_statement":
            string_node = next(
                (child for child in node.named_children if child.type == "string"), None
            )
            if string_node is not None:
                imports.append((_literal_module(file.text(string_node)), string_node))
        elif file.language == "java" and node.type == "import_declaration":
            identifier = next(
                (
                    child
                    for child in node.named_children
                    if child.type in {"scoped_identifier", "identifier"}
                ),
                None,
            )
            if identifier is not None:
                imports.append((file.text(identifier), identifier))
        elif file.language == "go" and node.type == "import_spec":
            string_node = next(
                (
                    child
                    for child in node.named_children
                    if child.type == "interpreted_string_literal"
                ),
                None,
            )
            if string_node is not None:
                imports.append((_literal_module(file.text(string_node)), string_node))
    return tuple(imports)


def _matches_prefix(module: str, prefixes: Iterable[str]) -> bool:
    return any(
        module == prefix
        or module.startswith(prefix + ".")
        or module.startswith(prefix + "/")
        for prefix in prefixes
    )


def _looks_like_test_path(path: str) -> bool:
    pure = PurePosixPath(path)
    name = pure.name.casefold()
    parts = {part.casefold() for part in pure.parts[:-1]}
    stem = PurePosixPath(name).stem
    return bool(
        parts & (_E2E_PATH_SEGMENTS | _INTEGRATION_PATH_SEGMENTS | _UNIT_PATH_SEGMENTS)
        or name.startswith("test_")
        or "_test." in name
        or ".test." in name
        or "_spec." in name
        or ".spec." in name
        or stem.endswith(("_test", "_spec"))
    )


def _test_layer(file: ParsedAddedFile) -> tuple[str, Node] | None:
    if not _looks_like_test_path(file.path):
        return None
    path_parts = {part.casefold() for part in PurePosixPath(file.path).parts[:-1]}
    imports = _structured_imports(file)
    evidence = imports[0][1] if imports else next(iter(file.root.named_children), file.root)
    for module, node in imports:
        if _matches_prefix(module, _E2E_IMPORT_PREFIXES):
            return "e2e", node
    if path_parts & _E2E_PATH_SEGMENTS:
        return "e2e", evidence
    for module, node in imports:
        if _matches_prefix(module, _INTEGRATION_IMPORT_PREFIXES):
            return "integration", node
    if path_parts & _INTEGRATION_PATH_SEGMENTS:
        return "integration", evidence
    for module, node in imports:
        if _matches_prefix(module, _UNIT_IMPORT_PREFIXES):
            return "unit", node
    return "unit", evidence


def verify_test_layer_balance(diff_text: str) -> Verdict:
    """Violate a broad E2E-only test change with no lower-layer test."""

    layers: list[tuple[str, ParsedAddedFile, Node]] = []
    for file in parse_added_files(diff_text):
        layer = _test_layer(file)
        if layer is not None:
            layers.append((layer[0], file, layer[1]))
    if not layers:
        return Verdict(False, False)
    broad = [(file, node) for layer, file, node in layers if layer == "e2e"]
    lower = [layer for layer, _, _ in layers if layer in {"unit", "integration"}]
    if broad and not lower:
        witnesses = tuple(
            span
            for file, node in broad
            for span in file.spans(node, node_id="test-layer:e2e-evidence")
        )
        verdict = Verdict(True, True, witnesses)
        validate_verdict_addresses(diff_text, verdict, require_added=True)
        return verdict
    witnesses = tuple(
        span
        for layer, file, node in layers
        for span in file.spans(node, node_id=f"test-layer:{layer}-evidence")
    )
    verdict = Verdict(True, False, witnesses)
    validate_verdict_addresses(diff_text, verdict, require_added=True)
    return verdict


def _identifier_texts(file: ParsedAddedFile, node: Node) -> set[str]:
    return {
        file.text(child)
        for child in _walk(node)
        if child.type == "identifier"
    }


def _is_nil_return(file: ParsedAddedFile, node: Node) -> bool:
    return node.type == "return_statement" and any(
        child.type == "nil" or file.text(child) == "nil" for child in _walk(node)
    )


def _discards_identifier(file: ParsedAddedFile, node: Node, identifier: str) -> bool:
    if node.type != "assignment_statement":
        return False
    named = node.named_children
    if len(named) < 2:
        return False
    left = _identifier_texts(file, named[0])
    right = _identifier_texts(file, named[-1])
    return "_" in left and identifier in right


def verify_swallowed_go_error(diff_text: str) -> Verdict:
    """Detect a Go non-nil error branch that discards it and returns success."""

    applicability_witness: tuple[Span, ...] = ()
    for file in parse_added_files(diff_text):
        if file.language != "go":
            continue
        for node in _walk(file.root):
            if node.type != "if_statement":
                continue
            condition = node.child_by_field_name("condition")
            consequence = node.child_by_field_name("consequence")
            if condition is None:
                condition = next(
                    (child for child in node.named_children if child.type != "block"),
                    None,
                )
            if consequence is None:
                consequence = next(
                    (child for child in node.named_children if child.type == "block"),
                    None,
                )
            if condition is None or consequence is None:
                continue
            condition_text = file.text(condition)
            identifiers = {
                name
                for name in _identifier_texts(file, condition)
                if name == "err" or name.endswith("Err") or name.endswith("Error")
            }
            if "!=" not in condition_text or "nil" not in condition_text or not identifiers:
                continue
            if not applicability_witness:
                applicability_witness = file.spans(
                    node, node_id="go:non-nil-error-branch"
                )
            body_nodes = tuple(_walk(consequence))
            for identifier in identifiers:
                if any(
                    _discards_identifier(file, child, identifier) for child in body_nodes
                ) and any(_is_nil_return(file, child) for child in body_nodes):
                    verdict = Verdict(
                        True,
                        True,
                        file.spans(
                            node, node_id=f"go:swallowed-error:{identifier}"
                        ),
                    )
                    validate_verdict_addresses(diff_text, verdict, require_added=True)
                    return verdict
    verdict = Verdict(bool(applicability_witness), False, applicability_witness)
    validate_verdict_addresses(diff_text, verdict, require_added=True)
    return verdict


REAL_UNIT_SPECS = (
    UnitSpec(
        unit_id="code-review:llama8b:k0:n0",
        aspect_id="a0",
        source_cuf_node_id=0,
        source_cuf_span=(
            "Keep control flow simple and easy to follow: minimize decision points "
            "and nesting, use guard clauses/early exits appropriately, avoid long "
            "switch/if-else chains, choose clear idiomatic constructs and loop forms, "
            "and favor analyzable shapes (e.g., SESE) to reduce cognitive complexity."
        ),
        implemented_relation="An added function contains three nested structured control nodes.",
        relation_scope="narrow_subrelation",
        verifier=verify_control_nesting,
    ),
    UnitSpec(
        unit_id="code-review:llama8b:k18:n0",
        aspect_id="a18",
        source_cuf_node_id=0,
        source_cuf_span=(
            "Identify and reduce change‑impeding smells (e.g., long methods/parameter "
            "lists, shotgun surgery, primitive obsession, inappropriate intimacy, long "
            "message chains) that harm readability and cohesion."
        ),
        implemented_relation=(
            "An added function has at least eight parameter declarations or spans at "
            "least eighty source lines."
        ),
        relation_scope="named_example_subrelation",
        verifier=verify_maintainability_smells,
    ),
    UnitSpec(
        unit_id="code-review:llama8b:k38:n0",
        aspect_id="a38",
        source_cuf_node_id=0,
        source_cuf_span=(
            "Adopt a balanced test mix emphasizing fast unit tests, few broad E2E/UI "
            "tests, pragmatic coverage, realistic data, limited brittle doubles, and "
            "awareness of automation limits."
        ),
        implemented_relation=(
            "An added test change contains an E2E/UI-layer test and no added unit- or "
            "integration-layer test."
        ),
        relation_scope="narrow_subrelation",
        verifier=verify_test_layer_balance,
    ),
    UnitSpec(
        unit_id="code-review:llama8b:k92:n0",
        aspect_id="a92",
        source_cuf_node_id=0,
        source_cuf_span=(
            "Define clear module/class boundaries, organize package/crate trees logically, "
            "and set appropriate visibility to keep APIs maintainable and evolvable."
        ),
        implemented_relation=(
            "An added Python module exposes at least four public top-level declarations "
            "without __all__ or any underscore-marked internal declaration."
        ),
        relation_scope="language_specific_subrelation",
        verifier=verify_python_visibility_boundary,
    ),
)


CONTROL_UNIT_SPECS = (
    UnitSpec(
        unit_id="pcr901",
        aspect_id=None,
        source_cuf_node_id=None,
        source_cuf_span="An added function contains a conditional nested inside two other conditionals.",
        implemented_relation="An added function contains a conditional nested inside two other conditionals.",
        relation_scope="positive_control",
        verifier=verify_conditional_nesting,
    ),
    UnitSpec(
        unit_id="pcr902",
        aspect_id=None,
        source_cuf_node_id=None,
        source_cuf_span="An added error branch discards a non-nil error and returns success.",
        implemented_relation=(
            "An added Go non-nil error branch discards that error and returns nil."
        ),
        relation_scope="positive_control",
        verifier=verify_swallowed_go_error,
    ),
)


ALL_AST_UNIT_SPECS = REAL_UNIT_SPECS + CONTROL_UNIT_SPECS


def specs_by_id() -> dict[str, UnitSpec]:
    return {spec.unit_id: spec for spec in ALL_AST_UNIT_SPECS}
