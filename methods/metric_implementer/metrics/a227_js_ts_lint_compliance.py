"""a227: JavaScript/TypeScript lint and style compliance.

Measures whether code ADDED in a PR diff conforms to a baseline ESLint
ruleset (a subset of `eslint:recommended` + common style rules) on JS/JSX/
MJS/CJS/TS/TSX files. Mirrors the `a181` ruff approach: score by violation
DENSITY on added lines, decaying exponentially to 0.

Per-language tooling (library-first; NO regex on code):
    JS / JSX / MJS / CJS    eslint --config <flat-config> --format=json
    TS / TSX                eslint with @typescript-eslint/parser

Score = exp(-violations_per_line * 20). Same scale as a181:
    0/line   = 1.0
    0.05/line = 0.37
    0.1/line  = 0.14

Note: We grade ONLY the added JS/TS lines reconstructed from the diff hunks.
This text is often syntactically incomplete (mid-function snippets), so
ESLint may emit parse errors. We do NOT count parse errors as lint
violations — when a file is fully unparseable we exclude it from both
numerator and denominator. If EVERY file is unparseable we return None
(abstain) rather than 0 to avoid penalising well-formatted code that just
happened to be chopped mid-statement.

Implementation notes / surprises:
- ESLint v9 dropped the `--no-eslintrc` flag and switched to a flat config
  (`eslint.config.js`). We write a minimal flat config to the temp dir per
  call. No project-level config is consulted because `--config <path>`
  + cwd of the temp dir means ESLint only sees our config.
- ESM `eslint.config.js` cannot import packages by bare name from a global
  npm install (Node ESM resolution doesn't honour NODE_PATH). We sidestep
  this by:
    (a) writing the config as a `.cjs` CommonJS module (eslint loads .cjs
        configs via require()), and
    (b) importing `@typescript-eslint/parser` via an ABSOLUTE path resolved
        once at module import time from `npm root -g`. JS/JSX needs no
        external parser (espree is bundled with eslint).
- If `npm root -g` fails or the TS parser cannot be resolved, TS/TSX files
  are simply skipped (treated as unmeasurable for THIS metric).
"""
from __future__ import annotations

import json
import os
import shutil
import math
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..sandbox import added_files_by_ext, have_tool, run, write_temp_files

ASPECT_ID = "a227"
ASPECT_NAME = "JavaScript/TypeScript lint and style compliance"
TIER = 3
TOOLS = ["eslint"]
APPLIES_TO_LANGS = ["JavaScript", "TypeScript"]
CLASSIFICATION = "THIN"

JS_EXTS = [".js", ".jsx", ".mjs", ".cjs"]
TS_EXTS = [".ts", ".tsx"]
ALL_EXTS = JS_EXTS + TS_EXTS


# ---------------------------------------------------------------------------
# Locate @typescript-eslint/parser (resolved once at import).
# We look in `npm root -g`/typescript-eslint/node_modules/@typescript-eslint/
# parser/dist/index.js (the install path from `npm install -g
# typescript-eslint`). If not found, TS/TSX files just won't be linted by
# this metric; JS/JSX still work.
# ---------------------------------------------------------------------------

def _find_ts_parser() -> Optional[str]:
    try:
        p = subprocess.run(
            ["npm", "root", "-g"], capture_output=True, text=True, timeout=5.0
        )
        if p.returncode != 0:
            return None
        npm_root = p.stdout.strip()
    except Exception:
        return None
    candidates = [
        Path(npm_root) / "typescript-eslint" / "node_modules"
        / "@typescript-eslint" / "parser" / "dist" / "index.js",
        Path(npm_root) / "@typescript-eslint" / "parser" / "dist" / "index.js",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


_TS_PARSER_PATH = _find_ts_parser()


# ---------------------------------------------------------------------------
# ESLint flat config (CommonJS module). Embedded as a Python string so the
# metric is self-contained.
# ---------------------------------------------------------------------------

_BASE_RULES = {
    "no-unused-vars": "error",
    "no-undef": "error",
    "no-var": "warn",
    "eqeqeq": "warn",
    "semi": ["warn", "always"],
    "no-empty": "warn",
    "no-extra-semi": "warn",
    "no-redeclare": "error",
    "no-unreachable": "error",
    "no-dupe-keys": "error",
    "no-dupe-args": "error",
    "no-cond-assign": "error",
    "no-constant-condition": "warn",
    "no-debugger": "error",
    "no-empty-pattern": "error",
    "no-irregular-whitespace": "error",
    "no-self-assign": "error",
    "no-sparse-arrays": "error",
    "use-isnan": "error",
    "valid-typeof": "error",
    "no-func-assign": "error",
    "no-import-assign": "error",
    "no-obj-calls": "error",
    "no-unsafe-finally": "error",
    "no-unsafe-negation": "error",
}

_GLOBALS = {
    g: "readonly" for g in [
        "console", "window", "document", "process", "require", "module",
        "exports", "__dirname", "__filename", "Buffer", "global",
        "setTimeout", "setInterval", "clearTimeout", "clearInterval",
        "Promise", "Map", "Set", "Symbol", "fetch", "URL", "URLSearchParams",
        "Error", "TypeError", "RangeError", "Date", "JSON", "Math",
        "Object", "Array", "String", "Number", "Boolean", "RegExp",
        "Infinity", "NaN", "undefined", "WeakMap", "WeakSet", "Proxy",
        "Reflect", "globalThis", "queueMicrotask",
        # Browser
        "navigator", "location", "history", "localStorage", "sessionStorage",
        "alert", "confirm", "HTMLElement", "Event", "CustomEvent",
        # Node
        "Buffer", "process", "__dirname", "__filename", "Promise",
        # Test
        "describe", "it", "test", "expect", "beforeEach", "afterEach",
        "beforeAll", "afterAll", "jest",
    ]
}


def _flat_config_cjs(ts_parser_path: Optional[str]) -> str:
    """Return the CJS source of the flat config to write to the temp dir."""
    base_rules_json = json.dumps(_BASE_RULES)
    globals_json = json.dumps(_GLOBALS)
    ts_block = ""
    if ts_parser_path:
        # Escape backslashes (Windows paths) in case.
        safe_path = ts_parser_path.replace("\\", "\\\\")
        ts_block = f"""
let tsParser = null;
try {{ tsParser = require({json.dumps(safe_path)}); }} catch (e) {{ tsParser = null; }}
if (tsParser) {{
  configs.push({{
    files: ["**/*.ts", "**/*.tsx"],
    rules: Object.assign({{}}, baseRules, {{
      // typescript-eslint own rules would handle these correctly; the core
      // no-unused-vars / no-undef trip on TS syntax (interfaces, types).
      "no-unused-vars": "off",
      "no-undef": "off"
    }}),
    languageOptions: {{
      parser: tsParser,
      ecmaVersion: "latest",
      sourceType: "module",
      globals: baseGlobals,
      parserOptions: {{ ecmaFeatures: {{ jsx: true }} }}
    }}
  }});
}}
"""
    return f"""
const baseRules = {base_rules_json};
const baseGlobals = {globals_json};
const configs = [
  {{
    files: ["**/*.js", "**/*.jsx", "**/*.mjs", "**/*.cjs"],
    rules: baseRules,
    languageOptions: {{
      ecmaVersion: "latest",
      sourceType: "module",
      globals: baseGlobals
    }}
  }}
];
{ts_block}
module.exports = configs;
"""


# ---------------------------------------------------------------------------

def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, ALL_EXTS))


def _line_count(s: str) -> int:
    if not s:
        return 0
    return s.count("\n") + (0 if s.endswith("\n") else 1)


def _run_eslint(td: Path, rel_files: List[str]) -> Optional[List[dict]]:
    """Run eslint --format=json over filenames RELATIVE to ``td``.

    We run with ``cwd=td`` and relative filenames because ESLint v9 treats
    absolute paths that lie outside any project root (no parent eslint
    config or package.json) as "ignored", emitting an empty result set.
    Running from inside the temp dir with the `eslint.config.cjs` we wrote
    there makes the temp dir the effective project root.

    Returns the parsed JSON list (one entry per file) or None on tool/config
    failure. ESLint exit codes: 0 = no problems, 1 = lint problems found,
    2 = config/system error. We accept rc<=1.
    """
    cmd = [
        "eslint",
        "--config", "eslint.config.cjs",
        "--no-config-lookup",
        "--format=json",
        *rel_files,
    ]
    rc, out, err = run(cmd, timeout=60.0, cwd=str(td))
    if rc < 0:
        return None
    if rc >= 2:
        # config/system error
        return None
    if not out.strip():
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def score(diff_text: str) -> Optional[float]:
    if not have_tool("eslint"):
        return None
    by_path = added_files_by_ext(diff_text, ALL_EXTS)
    if not by_path:
        return None

    td = write_temp_files(by_path)
    try:
        # Write the flat config alongside the source files.
        config_src = _flat_config_cjs(_TS_PARSER_PATH)
        (td / "eslint.config.cjs").write_text(config_src)

        # Build the file list: every file we wrote (write_temp_files
        # collapses paths into f000.ext, f001.ext, ...). Pass RELATIVE
        # filenames; eslint runs with cwd=td (see _run_eslint).
        rel_files = sorted(
            p.name for p in td.iterdir()
            if p.name != "eslint.config.cjs"
            and any(p.name.endswith(e) for e in ALL_EXTS)
        )
        if not rel_files:
            return None

        results = _run_eslint(td, rel_files)
        if results is None:
            return None

        # Tally measurable lines and lint violations.
        # A file is "unmeasurable" iff eslint reported a fatal parse error
        # (errorCount=1 + fatalErrorCount=1 + ruleId=null) or no messages at
        # all due to ignored file. Fatal parse errors are common because we
        # feed eslint partial code snippets reconstructed from the diff.
        total_measurable_lines = 0
        total_violations = 0
        n_measurable = 0
        # Map from on-disk filename -> reconstructed added text. Iterate in
        # the same order write_temp_files used (sorted by f### prefix above).
        contents_in_order = list(by_path.values())
        for res, content in zip(results, contents_in_order):
            msgs = res.get("messages", [])
            fatal = any(m.get("fatal") for m in msgs)
            ignored = any(
                m.get("ruleId") is None
                and "ignored" in (m.get("message", "") or "").lower()
                for m in msgs
            )
            if fatal or ignored:
                # parse error or unmatched config -> unmeasurable
                continue
            nlines = _line_count(content)
            if nlines == 0:
                continue
            # Count real lint violations (any message with a ruleId).
            n_viol = sum(1 for m in msgs if m.get("ruleId"))
            total_measurable_lines += nlines
            total_violations += n_viol
            n_measurable += 1

        if n_measurable == 0 or total_measurable_lines == 0:
            return None

        density = total_violations / total_measurable_lines
        return float(math.exp(-density * 20.0))
    finally:
        shutil.rmtree(td, ignore_errors=True)
