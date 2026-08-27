"""a25: Configuration design — minimal, explicit, and managed.

The norm: "Expose configuration only for concrete needs, prefer explicit
arguments over hidden defaults, and treat configuration as code with sound
design and management practices." We focus on the *managed* and *minimal*
sub-axes because they are the ones a determinist tool can measure on a diff:

  managed   = no secrets committed to config files; YAML/JSON well-formed
              under a real linter (i.e. config is treated as code, not
              dumped text)
  minimal   = config addition is not a huge "kitchen-sink" payload; bursty
              additions of hundreds of keys at once score lower than small,
              targeted changes
  explicit  = NOT directly measurable from the diff alone — that would
              require call-site analysis tying default-argument removal to
              an explicit-parameter rewrite. We leave that sub-axis for a
              future thicker metric and acknowledge it in CLASSIFICATION
              = PARTIALLY_THIN.

Tools used (all called via sandbox.run, no regex on code):
  - yamllint (Python, pip)      → YAML structural lint
  - detect-secrets (Python, pip) → hardcoded secret detection
  - json.loads / tomllib (stdlib) → JSON / TOML parse validity

Tool registration: yamllint and detect-secrets added to sandbox.ALLOWED_TOOLS.

Applicability gate: True iff the diff adds lines to any file matching a
config extension/name (.yaml/.yml/.toml/.json/.ini/.cfg/.conf/.properties
or env-file names like .env, .env.local, dotenv). Hard skip on lock files
(package-lock.json, yarn.lock, Pipfile.lock, poetry.lock) — those are
machine-generated and outside the "config design" norm.

Scoring. Three sub-scores in [0, 1] are combined multiplicatively so any
strongly-violated sub-axis can pull the overall score down:

  s_format  = exp(-yamllint_errors_per_added_line * 4.0)
              (parse/structural quality; only on YAML files)
  s_secret  = exp(-n_secret_findings * 2.0)
              (anti-managed; any committed secret is a strong negative)
  s_minimal = exp(-max(0, added_lines - 50) / 200.0)
              (anti-bursty; <=50 added config lines is "minimal", larger
              additions decay smoothly. 50 lines→1.0, 250→0.37, 450→0.13)

Sub-scores that don't apply (e.g. no YAML present → no s_format) default
to 1.0 so they don't penalize the overall score.

Final score = s_format * s_secret * s_minimal, then clamped to [0, 1].
1.0 means: small, well-formed config addition with no secrets. Lower
means at least one of (large config dump / lint errors / committed
secrets) is present.

What we deliberately DO NOT do:
  - Look inside source code for hardcoded config-ish values that should
    be in config files. That's a separate, much harder norm and would
    require call-site analysis. a47 already detects some of this
    (hardcoded passwords, hardcoded /tmp, hardcoded bind-all).
  - Score TOML / .ini lint quality. We only parse-check them; their
    "lint" community is fragmented and not worth a third tool for a
    single sub-axis. JSON likewise: parse-success only.

This is a PARTIALLY_THIN metric: well-grounded for the managed/minimal
sub-axes, blind to the explicit sub-axis.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

from ..sandbox import added_files_by_ext, have_tool, parse_diff_added_by_file, run

ASPECT_ID = "a25"
ASPECT_NAME = "Configuration design: minimal, explicit, managed"
TIER = 3
TOOLS = ["yamllint", "detect-secrets"]
APPLIES_TO_LANGS = ["YAML", "JSON", "TOML", "INI", "ENV"]
CLASSIFICATION = "PARTIALLY_THIN"

YAML_EXTS = (".yaml", ".yml")
JSON_EXTS = (".json",)
TOML_EXTS = (".toml",)
INI_EXTS = (".ini", ".cfg", ".conf", ".properties")

# Lock files are machine-generated; they violate "minimal" by design and
# linting them is not what the norm is about.
LOCK_FILE_NAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "composer.lock",
    "poetry.lock", "pipfile.lock", "gemfile.lock", "cargo.lock",
    "go.sum",
}


def _is_env_filename(path: str) -> bool:
    """Recognize .env-style dotenv files by *filename*, not extension.

    Examples: .env, .env.local, .env.production, dotenv, env.example
    We check the basename so paths like "config/.env.local" work.
    """
    base = os.path.basename(path).lower()
    if base == "dotenv":
        return True
    # .env, .env.<anything>
    if base == ".env" or base.startswith(".env."):
        return True
    return False


def _is_lock_file(path: str) -> bool:
    return os.path.basename(path).lower() in LOCK_FILE_NAMES


def _classify_config_files(added: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """Partition added files into config-format buckets, dropping lock files.

    Returns {"yaml": {path: content}, "json": {...}, "toml": {...},
             "ini": {...}, "env": {...}}.
    """
    out: Dict[str, Dict[str, str]] = {
        "yaml": {}, "json": {}, "toml": {}, "ini": {}, "env": {},
    }
    for path, content in added.items():
        if _is_lock_file(path):
            continue
        low = path.lower()
        if _is_env_filename(path):
            out["env"][path] = content
        elif low.endswith(YAML_EXTS):
            out["yaml"][path] = content
        elif low.endswith(JSON_EXTS):
            out["json"][path] = content
        elif low.endswith(TOML_EXTS):
            out["toml"][path] = content
        elif low.endswith(INI_EXTS):
            out["ini"][path] = content
    return out


def _added_config_files(diff_text: str) -> Dict[str, Dict[str, str]]:
    """Return classified config files added in the diff (no lock files)."""
    all_added = parse_diff_added_by_file(diff_text)
    return _classify_config_files(all_added)


def _has_any_config(diff_text: str) -> bool:
    bins = _added_config_files(diff_text)
    return any(bins[k] for k in bins)


def applies(diff_text: str) -> bool:
    """True iff the diff adds lines to any (non-lock) config-format file."""
    return _has_any_config(diff_text)


def _line_count(s: str) -> int:
    if not s:
        return 0
    return s.count("\n") + (0 if s.endswith("\n") else 1)


def _yamllint_errors(yaml_files: Dict[str, str]) -> Optional[int]:
    """Total yamllint error+warning count over all YAML snippets.

    Returns None iff yamllint is unavailable. Errors and warnings both count
    — the norm is about "treat config as code", and either signals
    insufficient hygiene. We pass each file via stdin.
    """
    if not yaml_files:
        return 0
    if not have_tool("yamllint"):
        return None
    total = 0
    for _path, content in yaml_files.items():
        if not content.strip():
            continue
        # The diff parser strips trailing newlines from extracted file
        # content; yamllint's "new-line-at-end-of-file" rule then fires
        # spuriously on every file. Re-add a trailing newline so we only
        # measure structural issues actually present in the contributed
        # text.
        feed = content if content.endswith("\n") else content + "\n"
        # -f parsable → "file:line:col: [level] message (rule)"
        # -d relaxed → uses the relaxed built-in config; we want structural
        # signal, not e.g. line-length nits.
        rc, out, _err = run(
            ["yamllint", "-f", "parsable", "-d", "relaxed", "-"],
            stdin=feed, timeout=15.0,
        )
        if rc < 0:
            # Tool error/timeout — abstain conservatively for this file.
            continue
        for ln in out.splitlines():
            if ln.strip():
                total += 1
    return total


def _detect_secrets_count(all_config: Dict[str, str]) -> Optional[int]:
    """Count distinct secret findings across all added config files.

    detect-secrets requires file *paths*, so we materialize the content
    into a temp dir and cd into it (the JSON output keys are relative).
    Returns None iff the tool is unavailable.
    """
    if not all_config:
        return 0
    if not have_tool("detect-secrets"):
        return None
    td = Path(tempfile.mkdtemp(prefix="metric_a25_"))
    try:
        for i, (path, content) in enumerate(all_config.items()):
            # Preserve extension so detector heuristics can use it.
            ext = "." + path.rsplit(".", 1)[-1] if "." in os.path.basename(path) else ""
            (td / f"f{i:03d}{ext}").write_text(content, errors="ignore")
        # Build relative file args so detect-secrets emits relative keys.
        rels = [p.name for p in td.iterdir()]
        rc, out, _err = run(
            ["detect-secrets", "scan"] + rels,
            cwd=str(td), timeout=25.0,
        )
        if rc < 0 or not out.strip():
            return None
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            return None
        results = data.get("results", {}) or {}
        # Count unique (file, line_number) pairs to avoid double-counting
        # when multiple detectors (e.g. AWS Access Key + Secret Keyword)
        # fire on the same physical secret.
        seen: set[Tuple[str, int]] = set()
        for fname, findings in results.items():
            for f in (findings or []):
                seen.add((fname, int(f.get("line_number", -1))))
        return len(seen)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _parse_ok_json(json_files: Dict[str, str]) -> Tuple[int, int]:
    """(n_parsed_ok, n_total). JSON-format files that fail to parse contribute
    to s_format as if they were lint errors."""
    ok = 0
    n = 0
    for _path, content in json_files.items():
        if not content.strip():
            continue
        n += 1
        try:
            json.loads(content)
            ok += 1
        except (json.JSONDecodeError, ValueError):
            pass
    return ok, n


def _parse_ok_toml(toml_files: Dict[str, str]) -> Tuple[int, int]:
    ok = 0
    n = 0
    try:
        import tomllib  # py3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            return 0, 0
    for _path, content in toml_files.items():
        if not content.strip():
            continue
        n += 1
        try:
            tomllib.loads(content)
            ok += 1
        except Exception:
            pass
    return ok, n


def score(diff_text: str) -> Optional[float]:
    bins = _added_config_files(diff_text)
    if not any(bins[k] for k in bins):
        return None

    # All added config content for secret-scanning + total-line counting.
    all_cfg: Dict[str, str] = {}
    for bucket in bins.values():
        all_cfg.update(bucket)

    total_added_lines = sum(_line_count(c) for c in all_cfg.values())
    if total_added_lines == 0:
        return None

    # ---- Format sub-score (managed-as-code) ----
    # YAML: yamllint findings (errors + warnings) per added YAML line.
    # JSON: parse fails count as findings (1 per failed file).
    # TOML: parse fails count as findings (1 per failed file).
    # INI / env: no lint tool — neutral (treat as 0 findings, 1.0 contribution).
    yaml_lines = sum(_line_count(c) for c in bins["yaml"].values())
    yaml_errs = _yamllint_errors(bins["yaml"])

    json_ok, json_n = _parse_ok_json(bins["json"])
    toml_ok, toml_n = _parse_ok_toml(bins["toml"])
    fmt_findings: float = 0.0
    fmt_basis_lines = 0

    if yaml_errs is not None and yaml_lines > 0:
        fmt_findings += yaml_errs
        fmt_basis_lines += yaml_lines
    if json_n > 0:
        # Each failed-parse JSON file = 1 finding; treat each parsed-OK file
        # as contributing its lines to the denominator with 0 findings.
        fmt_findings += (json_n - json_ok)
        fmt_basis_lines += sum(_line_count(c) for c in bins["json"].values())
    if toml_n > 0:
        fmt_findings += (toml_n - toml_ok)
        fmt_basis_lines += sum(_line_count(c) for c in bins["toml"].values())

    if fmt_basis_lines > 0:
        s_format = math.exp(-(fmt_findings / fmt_basis_lines) * 4.0)
    else:
        # Nothing measurable (e.g. ini-only diff, no yamllint, no JSON/TOML).
        s_format = 1.0

    # ---- Secret sub-score (managed: no committed credentials) ----
    n_secrets = _detect_secrets_count(all_cfg)
    if n_secrets is None:
        s_secret = 1.0  # tool missing → don't penalize
    else:
        s_secret = math.exp(-float(n_secrets) * 2.0)

    # ---- Minimal sub-score ----
    excess = max(0, total_added_lines - 50)
    s_minimal = math.exp(-excess / 200.0)

    final = s_format * s_secret * s_minimal
    # Numerical clamp.
    if final < 0.0:
        final = 0.0
    if final > 1.0:
        final = 1.0
    return float(final)
