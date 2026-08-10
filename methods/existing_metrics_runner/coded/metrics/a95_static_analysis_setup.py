"""a95: Static analysis with actionable feedback.

Aspect (from aspects.json):
  "Use static analysis/linting (including advanced analyses) to catch defects
   and design issues beyond compiler checks; surface prioritized, developer-
   actionable findings."

This is a META-norm about CI / tooling configuration, not about code content.
A PR conforms to this norm when it adds (or strengthens) the project's static-
analysis surface: new linter config files, new CI workflow steps that run
linters, or new `[tool.*]` sections in `pyproject.toml`.

Detection (applies):
  The diff adds at least one of:
    1. A recognized static-analysis CONFIG FILE  (e.g. `.pylintrc`,
       `.eslintrc`, `eslint.config.js`, `mypy.ini`, `pyrightconfig.json`,
       `.flake8`, `tox.ini`, `.golangci.yml`, `checkstyle.xml`, `pmd.xml`,
       `.rubocop.yml`, `.standardrb.yml`, `phpstan.neon`, `psalm.xml`,
       `clippy.toml`, `.semgrep.yml`, `bandit.yaml`, `.pre-commit-config.yaml`).
    2. A `pyproject.toml` change containing a `[tool.<linter>]` section.
    3. A `.github/workflows/*.yml` (or `.yaml`) file whose run-steps invoke
       at least one known static-analysis CLI.

  All three checks are diff-level — no need to inspect repo state.

Scoring:
  We compute three boolean signals from the diff and average them:

    config_added : a linter config file is added (or a pyproject [tool.X]
                   section appears in added lines)
    workflow_added : a CI workflow file invocation of a static-analysis
                   tool is added
    multi_tool : more than one distinct static-analysis tool surfaces
                   across the diff (indicates "advanced" coverage, matching
                   the "including advanced analyses" phrasing)

  Score = 0.4*config_added + 0.4*workflow_added + 0.2*multi_tool.
  Score is in [0,1]. A diff that introduces both config and CI step with
  multiple tools scores 1.0; a single workflow tweak that touches lint
  scores ~0.4.

Workflow parsing uses tree-sitter-yaml: we walk run/uses steps as YAML
scalar values and check whether any contains a known linter token. This is
strictly more robust than substring search on the raw text, because the
YAML parse rejects malformed/commented-out steps and respects nesting.

Tier 2 (parser-based, no subprocess). PARTIALLY_THIN: file/path/config-
section presence is a thin proxy for "use static analysis"; whether the
analysis genuinely produces *actionable* feedback is THICK and not measured.

Applicability is narrow by design: most PRs do not touch CI or linter
configs and will abstain via applies()=False. The norm is precisely about
those that do.
"""
from __future__ import annotations

import string
from typing import Dict, List, Optional, Set, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a95"
ASPECT_NAME = "Static analysis with actionable feedback"
TIER = 2
TOOLS = ["tree-sitter-yaml"]
APPLIES_TO_LANGS = ["YAML", "TOML", "INI", "JSON", "Any"]
CLASSIFICATION = "PARTIALLY_THIN"


# ---------------------------------------------------------------------------
# Linter / static-analysis tool registry.
#
# Tokens are matched case-insensitively against YAML scalar values in workflow
# run/uses lines and against pyproject `[tool.X]` table names. Each token
# names a distinct static-analysis surface so a "multi-tool" count is
# meaningful.
# ---------------------------------------------------------------------------

LINTER_TOKENS: Set[str] = {
    # Python
    "ruff", "pylint", "flake8", "mypy", "pyright", "pyre", "pyflakes",
    "pycodestyle", "bandit", "semgrep", "vulture", "pylama", "prospector",
    "isort", "black",  # formatter-as-linter is contested, but routinely
                       # configured alongside pylint/ruff and is named in
                       # most "code quality" CI sets
    # JS / TS
    "eslint", "tslint", "biome", "standardjs", "xo", "rome",
    # Java
    "checkstyle", "pmd", "spotbugs", "errorprone", "infer", "sonarqube",
    "sonar-scanner",
    # Go
    "golangci-lint", "staticcheck", "go vet", "govet", "revive", "gosec",
    "errcheck", "ineffassign",
    # Rust
    "cargo clippy", "clippy", "cargo-clippy",
    # Ruby
    "rubocop", "standardrb", "reek", "brakeman",
    # PHP
    "phpstan", "psalm", "phpcs", "phan",
    # C / C++ / Swift
    "cppcheck", "clang-tidy", "scan-build", "infer",
    # Scala / Kotlin
    "scalafix", "scalastyle", "detekt", "ktlint",
    # SQL
    "sqlfluff",
    # YAML / Shell / Docker meta-linters (relevant to "actionable feedback")
    "yamllint", "shellcheck", "hadolint", "actionlint",
    # Cross-language scanners
    "codeql", "trunk check", "trunk-check", "megalinter", "super-linter",
}

# pyproject [tool.X] table names we recognize as static-analysis configuration.
PYPROJECT_TOOL_SECTIONS: Set[str] = {
    "ruff", "pylint", "mypy", "pyright", "flake8", "isort", "black",
    "bandit", "semgrep", "pyflakes", "pycodestyle", "vulture", "pylama",
}

# Recognized standalone linter config files. Matched on the filename only
# (last path segment), case-insensitive. Some are exact matches; some are
# prefix/suffix patterns where a prefix tuple is given.
EXACT_CONFIG_FILENAMES: Set[str] = {
    ".pylintrc", "pylintrc",
    ".flake8",
    "mypy.ini", ".mypy.ini",
    "pyrightconfig.json",
    "tox.ini",
    "setup.cfg",        # often hosts [flake8]/[mypy] tables
    ".eslintrc", ".eslintrc.json", ".eslintrc.js", ".eslintrc.yml",
    ".eslintrc.yaml", ".eslintrc.cjs",
    "eslint.config.js", "eslint.config.cjs", "eslint.config.mjs",
    "eslint.config.ts",
    ".jshintrc",
    "biome.json", "biome.jsonc",
    ".golangci.yml", ".golangci.yaml", ".golangci.toml", ".golangci.json",
    "checkstyle.xml",
    "pmd.xml", "pmd-ruleset.xml",
    ".rubocop.yml", ".rubocop.yaml",
    ".standardrb.yml",
    "phpstan.neon", "phpstan.neon.dist", "phpstan.dist.neon",
    "psalm.xml", "psalm.xml.dist",
    "phpcs.xml", "phpcs.xml.dist", ".phpcs.xml",
    "clippy.toml", ".clippy.toml",
    ".semgrep.yml", ".semgrep.yaml", "semgrep.yml", "semgrep.yaml",
    "bandit.yaml", ".bandit",
    ".pre-commit-config.yaml", ".pre-commit-config.yml",
    ".pmdrc", ".checkstyle",
    "sonar-project.properties",
    "detekt.yml", "detekt.yaml",
    ".scalafix.conf",
    ".ktlint",
    "yamllint.yml", ".yamllint", ".yamllint.yml", ".yamllint.yaml",
    ".shellcheckrc",
    ".hadolint.yaml", ".hadolint.yml",
    ".trunk.yaml",
    ".sqlfluff",
}


# ---------------------------------------------------------------------------
# YAML parser for CI workflow detection.
# ---------------------------------------------------------------------------

_PARSER = None


def _parser():
    global _PARSER
    if _PARSER is not None:
        return _PARSER
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_yaml as mod
        lang = Language(mod.language())
        _PARSER = Parser(lang)
    except Exception:
        _PARSER = False
    return _PARSER


def _scalar_text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", errors="replace")


def _collect_yaml_scalars(code: bytes) -> List[str]:
    """Return all scalar leaf strings in the YAML tree.

    For workflow detection we look at scalar values (uses:, run:, etc.) — we
    don't need to track which key they hang off; any scalar that contains a
    known linter token is a static-analysis invocation.
    """
    parser = _parser()
    if not parser:
        return []
    try:
        tree = parser.parse(code)
    except Exception:
        return []
    out: List[str] = []

    def walk(n):
        # tree-sitter-yaml emits both `plain_scalar` (with inner string_scalar)
        # and quoted scalars (`single_quote_scalar`, `double_quote_scalar`,
        # `block_scalar`). We collect all leaf scalar text.
        if n.type in ("string_scalar", "single_quote_scalar",
                      "double_quote_scalar", "block_scalar"):
            t = _scalar_text(n, code).strip()
            if t:
                out.append(t)
            return  # leaf, no children worth recursing into
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    return out


# ---------------------------------------------------------------------------
# Per-file analyzers.
# ---------------------------------------------------------------------------

def _is_workflow_path(path: str) -> bool:
    p = path.lower().replace("\\", "/")
    return (
        "/.github/workflows/" in "/" + p
        and (p.endswith(".yml") or p.endswith(".yaml"))
    )


def _is_pyproject(path: str) -> bool:
    return path.lower().endswith("pyproject.toml")


def _filename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _config_tools_from_filename(path: str) -> Set[str]:
    """If the file is a known linter config, return the implied tool(s).

    Mapping uses substring matching of the filename against linter tokens —
    paths are not a language to parse, so simple string ops are correct.
    """
    name = _filename(path).lower()
    if name not in EXACT_CONFIG_FILENAMES:
        return set()
    tools: Set[str] = set()
    for tok in LINTER_TOKENS:
        if tok.replace(" ", "").replace("-", "") in name.replace(
                "-", "").replace(".", ""):
            tools.add(tok)
    # Generic catch-alls for configs whose filename doesn't carry the tool
    # name (e.g. .pre-commit-config.yaml, setup.cfg).
    if not tools:
        # We still know SOMETHING static-analysis is being configured.
        tools.add(name)
    return tools


def _pyproject_tools_from_added(added: str) -> Set[str]:
    """Find `[tool.X]` section headers in the added pyproject lines.

    pyproject.toml is TOML; section headers are line-anchored `[tool.X]`
    with optional further dots. We scan added text only and read the token
    right after `tool.`. This is text-format scanning of a well-defined
    format (REGEX_OK).
    """
    out: Set[str] = set()
    for line in added.splitlines():
        s = line.strip()
        if not s.startswith("[tool."):
            continue
        body = s[len("[tool."):]
        # cut at "]" or "." (for nested [tool.ruff.lint])
        end_dot = body.find(".")
        end_brk = body.find("]")
        cuts = [c for c in (end_dot, end_brk) if c >= 0]
        if not cuts:
            continue
        token = body[:min(cuts)].strip().lower()
        if token in PYPROJECT_TOOL_SECTIONS:
            out.add(token)
    return out


def _workflow_tools_from_added(added: str) -> Set[str]:
    """Run added workflow lines through tree-sitter-yaml.

    We only see the ADDED lines (not the full file) but YAML's block
    structure is line-anchored, so a fragment usually parses well enough to
    expose scalar values. If the parse yields no scalars (e.g. heavily
    truncated), we still return what we have — false negatives here are
    fine because applies() over-abstains.
    """
    if not added.strip():
        return set()
    scalars = _collect_yaml_scalars(added.encode("utf8", errors="replace"))
    out: Set[str] = set()
    for sc in scalars:
        sc_low = sc.lower()
        for tok in LINTER_TOKENS:
            # Require the token to appear as a "word" — bounded by start/end
            # or by a non-identifier char — so "ruff" doesn't fire on "truffle".
            if _token_in(sc_low, tok):
                out.add(tok)
    return out


_WORD_CHARS = set(string.ascii_lowercase + string.digits + "_-")


def _token_in(haystack: str, token: str) -> bool:
    """Word-boundary substring match. Treats `-` as part of identifiers."""
    if not token or not haystack:
        return False
    if " " in token:
        # multi-word tokens like "go vet" or "cargo clippy" — match raw.
        return token in haystack
    n = len(token)
    i = 0
    while True:
        j = haystack.find(token, i)
        if j < 0:
            return False
        left_ok = (j == 0) or (haystack[j - 1] not in _WORD_CHARS)
        right_end = j + n
        right_ok = (right_end == len(haystack)) or (
            haystack[right_end] not in _WORD_CHARS)
        if left_ok and right_ok:
            return True
        i = j + 1


# ---------------------------------------------------------------------------
# Top-level: aggregate across files in a diff.
# ---------------------------------------------------------------------------

def _analyze_diff(diff_text: str) -> Tuple[bool, bool, Set[str]]:
    """Return (config_added, workflow_added, tools_seen)."""
    by_path = parse_diff_added_by_file(diff_text)
    config_added = False
    workflow_added = False
    tools_seen: Set[str] = set()

    for path, added in by_path.items():
        # Standalone linter config file.
        cfg_tools = _config_tools_from_filename(path)
        if cfg_tools:
            config_added = True
            tools_seen |= cfg_tools
            continue

        # pyproject [tool.X] sections.
        if _is_pyproject(path):
            tools = _pyproject_tools_from_added(added)
            if tools:
                config_added = True
                tools_seen |= tools
            continue

        # CI workflow.
        if _is_workflow_path(path):
            tools = _workflow_tools_from_added(added)
            if tools:
                workflow_added = True
                tools_seen |= tools
            continue

    return config_added, workflow_added, tools_seen


def applies(diff_text: str) -> bool:
    """True iff the diff touches a CI workflow, a recognized linter config
    file, or a pyproject.toml with a [tool.X] section.

    We use diff parsing only (no subprocess) so applies() is cheap.
    """
    by_path = parse_diff_added_by_file(diff_text)
    for path, added in by_path.items():
        if _is_workflow_path(path):
            return True
        if _filename(path).lower() in EXACT_CONFIG_FILENAMES:
            return True
        if _is_pyproject(path) and "[tool." in added:
            return True
    return False


def score(diff_text: str) -> Optional[float]:
    if not applies(diff_text):
        return None
    config_added, workflow_added, tools_seen = _analyze_diff(diff_text)

    # If neither signal fired despite applies()=True (e.g. workflow file
    # touched but no static-analysis scalar found — maybe it edits a deploy
    # step), we abstain: we cannot say the norm was satisfied OR violated.
    if not (config_added or workflow_added):
        return None

    multi_tool = len(tools_seen) >= 2
    return float(
        0.4 * (1.0 if config_added else 0.0)
        + 0.4 * (1.0 if workflow_added else 0.0)
        + 0.2 * (1.0 if multi_tool else 0.0)
    )
