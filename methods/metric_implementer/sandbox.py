"""Sandbox for metric tool invocation.

Wraps subprocess with timeout and a tool whitelist. Metrics use the helpers
here instead of calling subprocess directly so behavior stays uniform.

Diff parsing uses `unidiff` (proper unified-diff parser), not regex. Code
parsing is delegated to tree-sitter (per-language) in metric modules.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import whatthepatch

ALLOWED_TOOLS = {"lizard", "ruff", "radon", "eslint", "mypy", "pylint",
                 "python3", "py_compile",
                 # Formatters used by a72_formatting_layout
                 "gofmt", "prettier", "google-java-format",
                 # SQL linter/formatter used by a226_sql_formatting_conventions
                 "sqlfluff",
                 # Security scanners used by a267_injection_prevention
                 "bandit", "semgrep",
                 # Dead-code detector used by a206_dead_unreachable_code
                 "vulture",
                 # Copy-paste detector used by a5_avoid_duplication
                 "jscpd",
                 # C/C++ static analyzer used by a145_buffer_string_safety
                 "cppcheck",
                 # C++ tooling used by a410-a417 C++ metrics
                 "clang-tidy", "cpplint", "flawfinder",
                 # Used by a420-a445 new C++ metrics
                 "clang-format",
                 # YAML linter + secret scanner used by a25_config_design
                 "yamllint", "detect-secrets",
                 # Dockerfile linter used by a222_dockerfile_best_practices
                 "hadolint"}

_TOOL_AVAILABLE: Dict[str, bool] = {}


def have_tool(name: str) -> bool:
    if name not in ALLOWED_TOOLS:
        raise ValueError(f"tool {name!r} not in ALLOWED_TOOLS")
    if name not in _TOOL_AVAILABLE:
        _TOOL_AVAILABLE[name] = shutil.which(name) is not None
    return _TOOL_AVAILABLE[name]


def run(cmd: List[str], stdin: Optional[str] = None,
        timeout: float = 10.0, cwd: Optional[str] = None) -> Tuple[int, str, str]:
    tool = cmd[0]
    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"tool {tool!r} not in ALLOWED_TOOLS")
    try:
        p = subprocess.run(cmd, input=stdin, capture_output=True,
                           text=True, timeout=timeout, cwd=cwd)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except FileNotFoundError:
        return -2, "", f"TOOL_MISSING:{tool}"


EXT_BY_LANG = {
    ".py": "py", ".pyi": "py",
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "ts", ".tsx": "ts",
    ".java": "java",
    ".go": "go",
    ".rs": "rs",
    ".rb": "rb",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "h",
    ".cs": "cs",
    ".kt": "kt", ".kts": "kt",
    ".scala": "scala",
    ".php": "php",
    ".swift": "swift",
}


def parse_diff_added_by_file(diff_text: str) -> Dict[str, str]:
    """Walk a unified diff; return {filepath: added_lines_concatenated}.

    Uses whatthepatch (line-oriented unified-diff parser, tolerant of
    truncation — important because the dense PR text format token-truncates
    the diff). Skips the preamble (## PR Title / ## Description) by jumping
    to the first `diff --git`.
    """
    idx = diff_text.find("diff --git")
    if idx == -1:
        return {}
    out: Dict[str, List[str]] = {}
    try:
        diffs = whatthepatch.parse_patch(diff_text[idx:])
    except Exception:
        return {}
    for d in diffs:
        if d is None:
            continue
        path = (d.header.new_path or d.header.old_path or "")
        if path.startswith("b/"):
            path = path[2:]
        if not path or path == "/dev/null":
            continue
        added = [ch.line for ch in (d.changes or [])
                 if ch.old is None and ch.new is not None and ch.line is not None]
        if added:
            out.setdefault(path, []).extend(added)
    return {f: "\n".join(ls) for f, ls in out.items() if ls}


def added_files_by_ext(diff_text: str, exts: List[str]) -> Dict[str, str]:
    """Filter parse_diff_added_by_file to files matching the given extensions."""
    all_files = parse_diff_added_by_file(diff_text)
    return {f: t for f, t in all_files.items()
            if any(f.lower().endswith(e) for e in exts)}


def write_temp_files(by_path: Dict[str, str]) -> Path:
    """Write each {path: content} pair to a temp directory.

    Preserves extension; collapses path to filename so subprocess tools see
    one file per source. Returns the temp directory path. Caller must
    clean up via shutil.rmtree.
    """
    td = Path(tempfile.mkdtemp(prefix="metric_imp_"))
    for i, (path, content) in enumerate(by_path.items()):
        ext = "." + path.rsplit(".", 1)[-1] if "." in path else ".txt"
        fname = f"f{i:03d}{ext}"
        (td / fname).write_text(content, errors="ignore")
    return td
