"""CPU-only structural operations for the active ``code_review`` census.

This module is deliberately separate from ``f2p_mock`` and
``pr_test_execution`` replay code.  Its input is the active census item
representation: a (possibly head/tail-truncated) unified diff.  It does not
load labels, judge results, repositories, or corpus neighbours.

The important unit is a relation, not a keyword.  A diff is parsed into
files/hunks, added code is parsed with tree-sitter, and tests are connected
to changed source symbols using identifier references and file/symbol naming
relations.  The resulting profile can support several h0 programs without
making each program reinvent a fragile diff parser.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
import re
import shlex
from typing import Iterable


_HUNK = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?:\s*(.*))?$"
)
_TEST_DIRS = {"test", "tests", "spec", "specs", "__tests__", "e2e", "integration"}
_SOURCE_EXTS = {
    ".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
    ".java", ".go",
}
_LANGUAGE = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".cjs": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".go": "go",
}


@dataclass(frozen=True)
class DiffLine:
    kind: str
    text: str
    old_lineno: int | None
    new_lineno: int | None


@dataclass
class DiffHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    section: str = ""
    lines: list[DiffLine] = field(default_factory=list)


@dataclass
class ChangedFile:
    old_path: str
    new_path: str
    hunks: list[DiffHunk] = field(default_factory=list)
    binary: bool = False

    @property
    def path(self) -> str:
        return self.new_path if self.new_path != "/dev/null" else self.old_path

    @property
    def added_lines(self) -> list[str]:
        return [line.text for hunk in self.hunks for line in hunk.lines
                if line.kind == "add"]

    @property
    def removed_lines(self) -> list[str]:
        return [line.text for hunk in self.hunks for line in hunk.lines
                if line.kind == "remove"]


@dataclass
class DiffDocument:
    files: list[ChangedFile]
    truncated: bool = False
    orphan_fragments: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CodeSymbol:
    path: str
    language: str
    kind: str
    name: str
    is_test: bool

    @property
    def qualified_name(self) -> str:
        return f"{self.path}::{self.name}"


def _clean_path(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith(('"', "'")):
        try:
            raw = shlex.split(raw)[0]
        except (ValueError, IndexError):
            raw = raw.strip('"\'')
    raw = raw.split("\t", 1)[0]
    if raw.startswith(("a/", "b/")):
        raw = raw[2:]
    return raw


def _diff_paths(line: str) -> tuple[str, str]:
    try:
        parts = shlex.split(line)
    except ValueError:
        parts = line.split()
    if len(parts) >= 4:
        return _clean_path(parts[2]), _clean_path(parts[3])
    return "", ""


def parse_unified_diff(text: str) -> DiffDocument:
    """Parse git unified diff structure, preserving hunks and line numbers.

    Active census ``ctext`` can contain a literal ``[...]`` head/tail splice.
    Each side is parsed independently so the tail cannot corrupt the valid
    head.  Prefix-bearing lines before a tail-side file header are retained as
    explicit orphan evidence but are never assigned to a guessed file.
    """
    marker = "\n[...]\n"
    segments = (text or "").split(marker)
    document = DiffDocument(files=[], truncated=len(segments) > 1)

    for segment_index, segment in enumerate(segments):
        current_file: ChangedFile | None = None
        current_hunk: DiffHunk | None = None
        old_lineno = new_lineno = 0
        orphan: list[str] = []

        for raw in segment.splitlines():
            if raw.startswith("diff --git "):
                if orphan:
                    document.orphan_fragments.append("\n".join(orphan))
                    orphan = []
                old_path, new_path = _diff_paths(raw)
                current_file = ChangedFile(old_path=old_path, new_path=new_path)
                document.files.append(current_file)
                current_hunk = None
                continue
            if current_file is None:
                if segment_index and raw[:1] in {"+", "-", " "}:
                    orphan.append(raw)
                continue
            if raw.startswith("--- "):
                current_file.old_path = _clean_path(raw[4:])
                continue
            if raw.startswith("+++ "):
                current_file.new_path = _clean_path(raw[4:])
                continue
            if raw.startswith(("Binary files ", "GIT binary patch")):
                current_file.binary = True
                current_hunk = None
                continue
            match = _HUNK.match(raw)
            if match:
                current_hunk = DiffHunk(
                    old_start=int(match.group(1)), old_count=int(match.group(2) or 1),
                    new_start=int(match.group(3)), new_count=int(match.group(4) or 1),
                    section=(match.group(5) or "").strip(),
                )
                current_file.hunks.append(current_hunk)
                old_lineno, new_lineno = current_hunk.old_start, current_hunk.new_start
                continue
            if current_hunk is None or raw.startswith("\\ No newline at end of file"):
                continue
            prefix, body = (raw[:1], raw[1:]) if raw else (" ", "")
            if prefix == "+":
                current_hunk.lines.append(DiffLine("add", body, None, new_lineno))
                new_lineno += 1
            elif prefix == "-":
                current_hunk.lines.append(DiffLine("remove", body, old_lineno, None))
                old_lineno += 1
            elif prefix == " ":
                current_hunk.lines.append(DiffLine("context", body, old_lineno, new_lineno))
                old_lineno += 1
                new_lineno += 1
        if orphan:
            document.orphan_fragments.append("\n".join(orphan))
    return document


def is_test_path(path: str) -> bool:
    original = PurePosixPath(path)
    p = PurePosixPath(path.lower())
    parts = set(p.parts)
    stem = p.stem
    return bool(parts & _TEST_DIRS) or stem.startswith("test_") or any(
        stem.endswith(suffix) for suffix in ("_test", ".test", "_spec", ".spec")
    ) or bool(re.search(r"(?:Test|Tests|Spec)$", original.stem))


def _language(path: str) -> str | None:
    return _LANGUAGE.get(PurePosixPath(path.lower()).suffix)


_PARSERS: dict[str, object] = {}


def _parser(language: str):
    if language in _PARSERS:
        return _PARSERS[language]
    try:
        from tree_sitter import Language, Parser
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
        else:
            return None
        parser = Parser(Language(capsule))
    except (ImportError, TypeError):
        return None
    _PARSERS[language] = parser
    return parser


def _walk(node) -> Iterable:
    yield node
    for child in node.children:
        yield from _walk(child)


def _node_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


_SYMBOL_NODES = {
    "python": {"function_definition": "function", "class_definition": "class"},
    "javascript": {
        "function_declaration": "function", "method_definition": "method",
        "class_declaration": "class",
    },
    "typescript": {
        "function_declaration": "function", "method_definition": "method",
        "class_declaration": "class", "interface_declaration": "interface",
    },
    "java": {"method_declaration": "method", "class_declaration": "class"},
    "go": {
        "function_declaration": "function", "method_declaration": "method",
        "type_declaration": "type",
    },
}


def _declared_name(node, source: bytes) -> str:
    named = node.child_by_field_name("name")
    if named is not None:
        return _node_text(named, source)
    for child in node.children:
        if child.type in {"identifier", "type_identifier"}:
            return _node_text(child, source)
    return ""


def _parse_added_file(changed_file: ChangedFile):
    language = _language(changed_file.path)
    parser = _parser(language) if language else None
    if parser is None:
        return None, b"", language
    source = "\n".join(changed_file.added_lines).encode("utf-8", errors="replace")
    return parser.parse(source), source, language


def extract_symbols(document: DiffDocument) -> list[CodeSymbol]:
    symbols: list[CodeSymbol] = []
    for changed_file in document.files:
        tree, source, language = _parse_added_file(changed_file)
        if tree is None or language is None:
            continue
        for node in _walk(tree.root_node):
            kind = _SYMBOL_NODES.get(language, {}).get(node.type)
            if not kind:
                continue
            name = _declared_name(node, source)
            if name:
                symbols.append(CodeSymbol(
                    path=changed_file.path, language=language, kind=kind, name=name,
                    is_test=is_test_path(changed_file.path),
                ))
    return symbols


def _identifiers(changed_file: ChangedFile) -> set[str]:
    tree, source, _ = _parse_added_file(changed_file)
    if tree is None:
        return set()
    return {
        _node_text(node, source)
        for node in _walk(tree.root_node)
        if node.type in {"identifier", "property_identifier", "type_identifier"}
    }


def _normal_name(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def _source_stem(path: str) -> str:
    stem = _normal_name(PurePosixPath(path).stem)
    for prefix in ("test", "spec"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    for suffix in ("tests", "test", "specs", "spec"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    return stem


def _assertion_count(changed_file: ChangedFile) -> int:
    tree, source, _ = _parse_added_file(changed_file)
    if tree is None:
        return 0
    count = 0
    for node in _walk(tree.root_node):
        if node.type == "assert_statement":
            count += 1
        if node.type in {"call", "call_expression"}:
            function = node.child_by_field_name("function")
            if function is None and node.children:
                function = node.children[0]
            name = _node_text(function, source).lower() if function is not None else ""
            leaf = re.split(r"[.:]", name)[-1]
            if leaf.startswith(("assert", "expect", "require", "verify")):
                count += 1
    return count


def test_design_profile(text: str) -> dict:
    """Return code-only sub-relation evidence for automated-test quality."""
    document = parse_unified_diff(text)
    code_files = [f for f in document.files if _language(f.path) and f.added_lines]
    source_files = [f for f in code_files if not is_test_path(f.path)]
    test_files = [f for f in code_files if is_test_path(f.path)]
    symbols = extract_symbols(document)
    source_symbols = [s for s in symbols if not s.is_test]
    test_symbols = [s for s in symbols if s.is_test and (
        s.name.lower().startswith("test") or s.kind in {"function", "method"}
    )]
    test_by_path = {f.path: f for f in test_files}
    refs = {f.path: _identifiers(f) for f in test_files}

    edges: list[dict] = []
    matched_sources: set[str] = set()
    # Symbol-level reference relation.  This is the main function-wall step:
    # a test must refer to the changed symbol, not merely coexist in the PR.
    for source in source_symbols:
        normal_source = _normal_name(source.name)
        for test in test_symbols:
            evidence: list[str] = []
            if source.name in refs.get(test.path, set()):
                evidence.append("ast_identifier_reference")
            normal_test = _normal_name(test.name)
            if len(normal_source) >= 3 and normal_source in normal_test:
                evidence.append("symbol_name_relation")
            # A same-stem test file is useful supporting provenance, but it
            # cannot by itself establish that the changed *function* is
            # exercised.  Require a symbol/reference relation first.
            if evidence and _source_stem(source.path) == _source_stem(test.path):
                evidence.append("file_stem_relation")
            if evidence:
                matched_sources.add(source.qualified_name)
                edges.append({
                    "source": source.qualified_name,
                    "test": test.qualified_name,
                    "evidence": sorted(set(evidence)),
                })

    # Body-only edits often omit the source declaration from added lines.
    # Preserve a weaker, explicitly file-level relation instead of pretending
    # that no source unit exists.
    file_matches = 0
    for source_file in source_files:
        if any(_source_stem(source_file.path) == _source_stem(test_file.path)
               for test_file in test_files):
            file_matches += 1

    assertions = sum(_assertion_count(f) for f in test_files)
    added_source_lines = sum(len(f.added_lines) for f in source_files)
    added_test_lines = sum(len(f.added_lines) for f in test_files)
    symbol_correspondence = (
        len(matched_sources) / len(source_symbols) if source_symbols else None
    )
    file_correspondence = file_matches / len(source_files) if source_files else None
    correspondence = (symbol_correspondence if symbol_correspondence is not None
                      else file_correspondence if file_correspondence is not None else 1.0)
    test_function_count = len(test_symbols)
    assertion_density = min(1.0, assertions / max(1, 2 * test_function_count))
    line_balance = min(1.0, added_test_lines / max(1, added_source_lines))

    return {
        "schema_version": "metric-seam-code-test-profile-v1",
        "truncated_input": document.truncated,
        "n_orphan_fragments": len(document.orphan_fragments),
        "source_files": [f.path for f in source_files],
        "test_files": [f.path for f in test_files],
        "source_symbols": [s.qualified_name for s in source_symbols],
        "test_symbols": [s.qualified_name for s in test_symbols],
        "test_to_source_edges": edges,
        "added_source_lines": added_source_lines,
        "added_test_lines": added_test_lines,
        "assertions": assertions,
        "presence": 1.0 if test_files else 0.0,
        "line_balance": line_balance,
        "correspondence": correspondence,
        "assertion_density": assertion_density,
    }


class CodeOps:
    """Outcome-blind op surface for active coding-census h0 programs."""

    parse_unified_diff = staticmethod(parse_unified_diff)
    extract_symbols = staticmethod(lambda text: extract_symbols(parse_unified_diff(text)))
    test_design_profile = staticmethod(test_design_profile)
