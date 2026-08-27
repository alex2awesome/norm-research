"""a20: Dependency hygiene and minimization.

Aspect (from aspects.json):
  "Minimize and manage dependencies; avoid unnecessary or heavy libraries and
   transitive bloat. Keep project structure understandable and dependency
   graph clean to limit coupling and maintenance costs."

What we measure deterministically from the diff:

  minimization  = small number of net-added direct dependencies.
                  A PR that adds 1 dep scores higher than one that adds 12.
                  Smooth exponential decay so the metric stays continuous.

  pinning       = added requirements specify a version constraint, not a
                  floating "*" / "latest" / "^" range that lets transitive
                  drift creep in over time. Ecosystem-aware: `==1.2.3` /
                  `~=1.2.3` for pip; exact version strings (no `^`/`~`) in
                  npm; pinned commit / tag for go.mod; `=`/`~=` for Cargo.

  lock_sync     = if the PR modifies a manifest with an associated lockfile
                  (package.json + package-lock.json / yarn.lock; Pipfile +
                  Pipfile.lock; Cargo.toml + Cargo.lock; pyproject.toml +
                  poetry.lock; go.mod + go.sum), the lockfile is also
                  updated. Forgetting the lock is a hygiene red flag.

What we deliberately DO NOT do:
  - Resolve transitive bloat / size on disk (would need network + ecosystem-
    specific resolution). The norm mentions "transitive bloat" but only the
    *direct* additions are visible in a diff.
  - Judge whether a specific dependency is "heavy" or "unnecessary" — that's
    a THICK taste judgment about ecosystem choice. We measure only that the
    SET of added direct deps is small and pinned.
  - Use `pip-audit` / `npm-check-updates` / `cargo audit` — those require
    a populated environment and network access, neither of which the
    sandbox provides. The TOML/JSON parsers we use already give us pinning
    and count, which are the audit signals most diff-visible anyway.

Detection (applies):
  The diff parses (via whatthepatch) and at least one of the recognized
  manifest files has added lines that introduce dependency entries:
    - requirements.txt / requirements-*.txt / requirements.in
    - pyproject.toml with [project.dependencies] or [tool.poetry.dependencies]
      or [project.optional-dependencies.*] or [tool.poetry.dev-dependencies]
    - package.json with "dependencies" / "devDependencies" / "peerDependencies"
    - Cargo.toml with [dependencies] / [dev-dependencies] / [build-dependencies]
    - go.mod with `require (` blocks or `require <module> <version>` lines

  Pure lockfile-only diffs do not pass applies(): those are mechanical and
  the human-authored "what dependency did you add" decision is not present.

Scoring (in [0,1]):
  s_minimal = exp(- max(0, n_added_deps - 1) / 5.0)
              1 → 1.00, 2 → 0.82, 5 → 0.45, 10 → 0.17
              (one added dep is the friction-free baseline; we don't reward
              zero — that's typically applies()=False territory anyway.)

  s_pinned  = pinned_fraction  (in [0,1]) over added deps we could classify
              Fallback to 1.0 (no penalty) if pinning is ill-defined for
              the manifest (go.mod entries are always versioned).

  s_locksync = 1.0 if lock co-modified OR no associated lockfile in repo's
               likely paths; 0.5 if a lockfile co-existed in the diff's
               file-set with the manifest but was not touched. We cannot
               check repo state for absent lockfiles, so missing co-touch
               is treated as a soft penalty rather than a binary fail.

  final = s_minimal * (0.5 + 0.5 * s_pinned) * s_locksync
          (pinning is weighted half so an unpinned single-dep add still
          scores ~0.5 rather than 0, reflecting that minimization itself is
          the dominant axis.)

Classification: PARTIALLY_THIN. We measure count + pinning + lock-sync
deterministically, but "is THIS dependency heavy/unnecessary" remains
THICK. Tier 2 (TOML/JSON parsers, no subprocess).

This metric is narrowly applicable by design: most PRs do not touch
manifests. The norm is precisely about those that do.
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Dict, List, Optional, Set, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a20"
ASPECT_NAME = "Dependency hygiene and minimization"
TIER = 2
TOOLS = []  # pure stdlib parsers (json, tomllib, line-based for reqs/go.mod)
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Rust", "Go"]
CLASSIFICATION = "PARTIALLY_THIN"


# ---------------------------------------------------------------------------
# Manifest classification by filename.
# ---------------------------------------------------------------------------

def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _is_requirements_txt(path: str) -> bool:
    """Match requirements.txt and variants like requirements-dev.txt, dev-reqs.txt."""
    b = _basename(path).lower()
    if not (b.endswith(".txt") or b.endswith(".in")):
        return False
    # REGEX_OK: file_path — matching filename conventions, not parsing code.
    return bool(re.match(r"^(.+[-_])?requirements([-_].+)?\.(txt|in)$", b)) or \
           b in ("requirements.txt", "requirements.in")


def _is_pyproject(path: str) -> bool:
    return _basename(path).lower() == "pyproject.toml"


def _is_package_json(path: str) -> bool:
    return _basename(path).lower() == "package.json"


def _is_cargo_toml(path: str) -> bool:
    return _basename(path).lower() == "cargo.toml"


def _is_go_mod(path: str) -> bool:
    return _basename(path).lower() == "go.mod"


def _is_pipfile(path: str) -> bool:
    return _basename(path).lower() == "pipfile"


def _lockfile_for(path: str) -> Optional[str]:
    """Return the basename of the lockfile associated with this manifest."""
    b = _basename(path).lower()
    if b == "package.json":
        return "package-lock.json"  # also yarn.lock / pnpm-lock.yaml; handled in caller
    if b == "pipfile":
        return "pipfile.lock"
    if b == "cargo.toml":
        return "cargo.lock"
    if b == "pyproject.toml":
        return "poetry.lock"  # uv.lock also possible; handled in caller
    if b == "go.mod":
        return "go.sum"
    return None


# Lockfiles considered "associated" with each manifest type, for lock-sync.
LOCKFILES_BY_MANIFEST = {
    "package.json": {"package-lock.json", "yarn.lock", "pnpm-lock.yaml",
                     "npm-shrinkwrap.json"},
    "pipfile": {"pipfile.lock"},
    "cargo.toml": {"cargo.lock"},
    "pyproject.toml": {"poetry.lock", "uv.lock", "pdm.lock"},
    "go.mod": {"go.sum"},
}


# ---------------------------------------------------------------------------
# Per-manifest parsers: return (n_added_deps, n_pinned_deps_classified).
# n_pinned_deps_classified is the count over which pinning is *defined* (so
# the caller can compute s_pinned = pinned / classified safely).
# ---------------------------------------------------------------------------

# A pip requirement line has a name plus optional extras plus version specifier.
# We accept anything with `==` or `===` or `~=` as "pinned". `>=`, `<`, `*`,
# or bare-name (no specifier) are NOT pinned. Markers (`; python_version<'3.10'`)
# are ignored for pinning purposes.
# REGEX_OK: binary_format — requirements.txt is a well-defined line format with
# no parser library that's worth pulling in (pip's internal parser isn't a
# public API). We only need name + specifier.
_PIP_LINE = re.compile(
    r"^\s*([A-Za-z0-9_.\-]+)\s*"           # package name
    r"(\[[A-Za-z0-9_,\-\s]+\])?\s*"        # optional extras
    r"(?P<spec>[<>=!~]+[^;#\s]+)?"         # optional version spec
)


def _parse_requirements_added3(added: str) -> Tuple[int, int, int]:
    """Parse added lines from a requirements*.txt and return
    (n_deps, n_pinned, n_classified).

    Skips comments, blank lines, -r/-c includes, --hash lines, and constraint-
    only lines. `-e <thing>` counts as a dep but pinning is undefined.
    """
    n_deps = n_pinned = n_classified = 0
    for raw in added.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-") or line.startswith("--"):
            if line.startswith("-e ") or line.startswith("--editable"):
                n_deps += 1
            continue
        m = _PIP_LINE.match(line)
        if not m:
            continue
        n_deps += 1
        spec = m.group("spec") or ""
        if "==" in spec or spec.startswith("~="):
            n_pinned += 1
            n_classified += 1
        elif spec:
            n_classified += 1
    return n_deps, n_pinned, n_classified


def _parse_pyproject_added(added: str, full_diff_for_file: str = "") -> Tuple[int, int, int]:
    """Parse [project.dependencies] / [tool.poetry.dependencies] additions.

    Returns (n_deps, n_pinned, n_classified). We walk the full per-file diff
    (context + added lines), tracking which TOML section we're currently
    inside from ANY line, but only COUNT additions (lines that start with
    '+'). This is the standard pattern: section headers and array-opening
    lines may be context, so we cannot restrict ourselves to added-only.
    """
    n_deps = n_pinned = n_classified = 0
    current_section: Optional[str] = None
    DEP_SECTIONS = {
        "project.dependencies",
        "project.optional-dependencies",
        "tool.poetry.dependencies",
        "tool.poetry.dev-dependencies",
        "tool.poetry.group",
        "dependency-groups",
    }
    in_array_dep_list = False
    if full_diff_for_file:
        # Walk diff with marker awareness.
        lines_with_marker = []
        for raw in full_diff_for_file.splitlines():
            if not raw:
                continue
            if raw.startswith("@@"):
                # hunk header. Reset section to None — TOML sections don't
                # carry across hunks unless we explicitly see them again.
                current_section = None
                in_array_dep_list = False
                continue
            if raw[0] not in (" ", "+", "-"):
                continue
            lines_with_marker.append((raw[0], raw[1:]))
    else:
        # Fallback: only the added-line concatenation. Markers are unknown,
        # treat each line as "added" but still track section headers.
        lines_with_marker = [("+", ln) for ln in added.splitlines()]

    for marker, body in lines_with_marker:
        s = body.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("[") and s.endswith("]"):
            hdr = s.strip("[]")
            current_section = hdr
            in_array_dep_list = False
            continue
        if current_section is None and not in_array_dep_list:
            continue
        # Special case: PEP 621 puts the `dependencies = [...]` array directly
        # under `[project]` (not under a `[project.dependencies]` table). We
        # accept this at the [project] level as well so a `dependencies = [` opener
        # is recognized regardless of section depth.
        is_project_level_deps_open = (
            current_section == "project"
            and "=" in s
            and s.split("=", 1)[0].strip() in ("dependencies", "requires")
        )
        # Are we in a dep section?
        active = False
        cs = current_section or ""
        if cs == "project.dependencies":
            active = True
        elif cs.startswith("project.optional-dependencies"):
            active = True
        elif cs == "tool.poetry.dependencies" or cs == "tool.poetry.dev-dependencies":
            active = True
        elif cs.startswith("tool.poetry.group.") and cs.endswith(".dependencies"):
            active = True
        elif cs == "dependency-groups":
            active = True
        elif is_project_level_deps_open or in_array_dep_list:
            # PEP 621 inline-array form lives under [project]; once we see the
            # opener we enter array mode and keep counting until ]/section end.
            active = True
        if not active:
            continue

        # Two TOML shapes:
        #  (a) array-of-strings: `dependencies = ["foo==1.0", "bar>=2"]`
        #      or multi-line `dependencies = [\n  "foo==1.0",\n]`
        #  (b) inline-table: `foo = "==1.0"` or `foo = "1.0"` or
        #      `foo = {version = "1.0", extras = [...]}`
        # We update array-state from ALL lines (context + added) but only
        # COUNT additions (marker == "+").
        if "=" in s and not (s.startswith('"') or s.startswith("'")):
            name, _, rhs = s.partition("=")
            name = name.strip()
            rhs = rhs.strip().rstrip(",")
            if name in ("dependencies", "requires") and rhs.startswith("["):
                in_array_dep_list = True
                # inline contents (if `dependencies = ["a", "b"]` on one line):
                inline = rhs.strip("[]")
                if marker == "+":
                    d2, p2, c2 = _count_array_strings(inline)
                    n_deps += d2
                    n_pinned += p2
                    n_classified += c2
                if rhs.endswith("]"):
                    in_array_dep_list = False
                continue
            if not name or name.startswith("["):
                continue
            if current_section.startswith("tool.poetry") and name == "python":
                continue
            if marker != "+":
                continue
            n_deps += 1
            ver = _extract_toml_version(rhs)
            d, p = _classify_poetry_or_pep508(ver, current_section)
            if d:
                n_classified += 1
                if p:
                    n_pinned += 1
        elif in_array_dep_list:
            if marker == "+":
                d, p, c = _count_array_strings(s)
                n_deps += d
                n_pinned += p
                n_classified += c
            if s.endswith("]"):
                in_array_dep_list = False
    return n_deps, n_pinned, n_classified


def _count_array_strings(body: str) -> Tuple[int, int, int]:
    """Count `"name SPEC"` entries inside a TOML array-of-strings fragment.

    PEP 508-ish: `foo==1.0`, `foo>=1.0`, `foo[extra]>=1.0; python_version<'3.10'`.
    """
    n_deps = n_pinned = n_classified = 0
    # REGEX_OK: binary_format — TOML array of double-quoted PEP 508 strings,
    # well-defined and not worth a full TOML parse since we only see added
    # fragments (which may not be syntactically complete).
    for m in re.finditer(r'"([^"\\]*(?:\\.[^"\\]*)*)"', body):
        spec = m.group(1)
        if not spec.strip():
            continue
        n_deps += 1
        d, p = _classify_pep508(spec)
        if d:
            n_classified += 1
            if p:
                n_pinned += 1
    return n_deps, n_pinned, n_classified


def _classify_pep508(spec: str) -> Tuple[bool, bool]:
    """Return (classified, pinned) for a PEP-508-style requirement string.

    classified = could decide whether pinning is satisfied or not
    pinned = has `==` or `~=` constraint
    """
    body = spec.split(";", 1)[0]  # drop environment marker
    body = body.strip()
    if "==" in body or "~=" in body:
        return True, True
    # presence of ANY operator means "classified, not pinned"
    if any(op in body for op in (">=", "<=", ">", "<", "!=", "===")):
        return True, False
    return False, False  # bare name → undefined pinning


def _classify_poetry_or_pep508(version_str: str,
                                section: str) -> Tuple[bool, bool]:
    """Classify a poetry-style version constraint or a bare PEP-508 version.

    In poetry, `foo = "^1.2"` and `foo = "*"` are unpinned; `foo = "1.2.3"`
    (no operator) is interpreted as "^1.2.3" → also unpinned. Exact pinning
    in poetry is `foo = "==1.2.3"`.
    For non-poetry sections we fall back to _classify_pep508.
    """
    if not version_str:
        return False, False
    v = version_str.strip()
    if section.startswith("tool.poetry") or section == "dependency-groups":
        if v == "*":
            return True, False
        if v.startswith("^") or v.startswith("~"):
            return True, False
        if v.startswith("=="):
            return True, True
        # bare version "1.2.3" in poetry → caret-equivalent → not pinned
        # REGEX_OK: binary_format — checking semver-shaped bare string.
        if re.match(r"^\d", v):
            return True, False
        # >=/<= etc.
        if any(op in v for op in (">=", "<=", ">", "<")):
            return True, False
        return False, False
    return _classify_pep508(v)


def _extract_toml_version(rhs: str) -> str:
    """Pull version string out of TOML rhs (either `"1.0"` or `{version="1.0",...}`)."""
    rhs = rhs.strip().rstrip(",")
    if rhs.startswith('"') or rhs.startswith("'"):
        q = rhs[0]
        end = rhs.find(q, 1)
        if end > 0:
            return rhs[1:end]
    if rhs.startswith("{"):
        # REGEX_OK: binary_format — TOML inline-table version key.
        m = re.search(r'version\s*=\s*"([^"]+)"', rhs)
        if m:
            return m.group(1)
    return ""


# package.json --------------------------------------------------------------

def _parse_package_json_added(added: str, full_diff_for_file: str) -> Tuple[int, int, int]:
    """Detect newly-added dependency entries in a package.json diff hunk.

    package.json is JSON; trying to json.loads(added) usually fails because
    we only see fragments. We instead look at added LINES that match
    `"name": "version",` inside contexts that look like a dep block. Whether
    we're inside `"dependencies"` etc. is approximated by scanning context
    lines for a recent `"dependencies"` or `"devDependencies"` /
    `"peerDependencies"` / `"optionalDependencies"` key in the SURROUNDING
    diff (added or unchanged), since the section header itself is rarely
    re-added when only entries change.
    """
    # Build a section map from the surrounding diff text (which includes
    # context lines starting with " " in unified diffs).
    n_deps = n_pinned = n_classified = 0
    DEP_KEYS = {"dependencies", "devDependencies", "peerDependencies",
                "optionalDependencies", "bundledDependencies", "bundleDependencies"}
    in_dep_section: Optional[str] = None
    # We re-scan the per-file diff text rather than just `added` so we have
    # both context and additions.
    for raw in full_diff_for_file.splitlines():
        if not raw:
            continue
        marker = raw[0]
        if marker not in (" ", "+", "-"):
            continue
        body = raw[1:]
        s = body.strip().rstrip(",")
        # Update section state from any line (additions or context).
        # REGEX_OK: binary_format — matching JSON key at line start.
        m_sec = re.match(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*\{?\s*$', s)
        if m_sec and m_sec.group(1) in DEP_KEYS:
            in_dep_section = m_sec.group(1)
            continue
        if s.startswith("}") or s.startswith("],"):
            in_dep_section = None
            continue
        if marker != "+":
            continue
        if not in_dep_section:
            continue
        # REGEX_OK: binary_format — matching JSON string-pair inside object.
        m = re.match(r'"([^"]+)"\s*:\s*"([^"]*)"', s)
        if not m:
            continue
        name = m.group(1)
        ver = m.group(2)
        if not name:
            continue
        n_deps += 1
        d, p = _classify_npm_version(ver)
        if d:
            n_classified += 1
            if p:
                n_pinned += 1
    return n_deps, n_pinned, n_classified


def _classify_npm_version(ver: str) -> Tuple[bool, bool]:
    """npm semver-range pinning.

    pinned   = bare semver `1.2.3` or `=1.2.3` (no operator that lets minor
               or patch float)
    classified+unpinned = `^`, `~`, `>=`, `<`, `||`, `*`, `x`, `latest`, URL,
               git+, file:
    """
    v = ver.strip()
    if not v:
        return False, False
    if v in ("*", "latest", ""):
        return True, False
    if v.startswith(("^", "~", ">", "<", "=")):
        if v.startswith("=") and not v.startswith("=="):
            # `=1.2.3` is exact
            # REGEX_OK: binary_format — semver after `=`.
            if re.match(r"^=\d+(\.\d+){0,2}", v) and "x" not in v.lower():
                return True, True
        return True, False
    if v.startswith(("git+", "github:", "file:", "http://", "https://",
                     "npm:", "workspace:", "link:")):
        # Pinning is *kind-of* satisfied (git sha refs are pinned by hash)
        # but we cannot tell from the string alone. Treat as not classified.
        return False, False
    # bare semver-ish
    # REGEX_OK: binary_format — semver-ish exact version.
    if re.match(r"^\d+(\.\d+){0,2}([.\-][A-Za-z0-9._\-]+)?$", v):
        if "x" in v.lower() or "*" in v:
            return True, False
        return True, True
    return False, False


# Cargo.toml ----------------------------------------------------------------

def _parse_cargo_added(added: str, full_diff_for_file: str = "") -> Tuple[int, int, int]:
    """Parse [dependencies]/[dev-dependencies]/[build-dependencies] in Cargo.toml.

    Cargo pinning convention: a bare `"1.2"` means `^1.2` (caret) — NOT
    pinned in the strict sense. `"=1.2.3"` is exact-pinned. Updates section
    state from ALL lines (context+added) but counts only additions.
    """
    n_deps = n_pinned = n_classified = 0
    current_section: Optional[str] = None
    if full_diff_for_file:
        lines_with_marker = []
        for raw in full_diff_for_file.splitlines():
            if not raw:
                continue
            if raw.startswith("@@"):
                current_section = None
                continue
            if raw[0] not in (" ", "+", "-"):
                continue
            lines_with_marker.append((raw[0], raw[1:]))
    else:
        lines_with_marker = [("+", ln) for ln in added.splitlines()]
    for marker, body in lines_with_marker:
        s = body.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("[") and s.endswith("]"):
            hdr = s.strip("[]")
            current_section = hdr
            continue
        if current_section is None:
            continue
        # Activate on any dep-bearing section. Be permissive.
        active = (
            current_section in ("dependencies", "dev-dependencies",
                                "build-dependencies")
            or current_section.startswith("workspace.dependencies")
            or current_section.startswith("target.")  # target.cfg(...).dependencies
            and "dependencies" in current_section
        )
        if not active:
            continue
        # Only count additions. State above (section) updates from all lines.
        if marker != "+":
            continue
        # `foo = "1.2"` or `foo = {version = "1.2", features = [...]}`
        # or `foo.workspace = true`
        if "=" not in s:
            continue
        name, _, rhs = s.partition("=")
        name = name.strip()
        rhs = rhs.strip().rstrip(",")
        if not name or name.startswith("["):
            continue
        # Skip dotted keys like `foo.version = "1.0"` — these mutate, not add.
        if "." in name:
            continue
        n_deps += 1
        ver = _extract_toml_version(rhs)
        if not ver:
            # No version (e.g. workspace = true, or git/path only) — undefined.
            continue
        n_classified += 1
        # Cargo: exact pin requires `=X.Y.Z`.
        if ver.startswith("="):
            n_pinned += 1
        # bare `1.2.3` is implicit caret in Cargo → unpinned.
    return n_deps, n_pinned, n_classified


# go.mod --------------------------------------------------------------------

def _parse_go_mod_added(added: str, full_diff_for_file: str) -> Tuple[int, int, int]:
    """Parse `require` blocks in go.mod.

    go.mod entries are always semver-pinned (or commit-pinned via pseudo-
    version like `v0.0.0-2019xxx-abc123`), so pinning is always satisfied
    when a parseable entry is found. We just count entries.
    """
    n_deps = n_pinned = n_classified = 0
    in_require = False
    # Track require-block state across the whole hunk (context + added).
    # Unified-diff hunk headers like `@@ -10,6 +10,7 @@ require (` carry the
    # enclosing-construct hint; recognize that and enter require-block state.
    for raw in full_diff_for_file.splitlines():
        if not raw:
            continue
        if raw.startswith("@@"):
            # hunk header — may carry the function/section context after `@@`.
            tail = raw.split("@@", 2)[-1] if raw.count("@@") >= 2 else ""
            if "require (" in tail:
                in_require = True
            # we intentionally do NOT set in_require=False here just because
            # tail doesn't mention require — the previous hunk's state may
            # still apply within the same block. But blocks are scoped to one
            # file, so it's a safe local approximation.
            continue
        marker = raw[0]
        if marker not in (" ", "+", "-"):
            continue
        body = raw[1:].strip()
        if body.startswith("require ("):
            in_require = True
            continue
        if in_require and body == ")":
            in_require = False
            continue
        if marker != "+":
            continue
        # Single-line: `require <mod> <ver>` outside of a block.
        if body.startswith("require ") and not body.endswith("("):
            # `require example.com/foo v1.2.3` or w/ // indirect
            parts = body.split()
            if len(parts) >= 3:
                n_deps += 1
                n_classified += 1
                n_pinned += 1  # go.mod versions are always concrete
            continue
        if in_require:
            # `<mod> <ver>` or `<mod> <ver> // indirect`
            # Skip blank lines / comments / `// indirect`-only lines.
            if not body or body.startswith("//"):
                continue
            parts = body.split()
            if len(parts) >= 2:
                # Skip `indirect` markers we counted via // indirect logic; we
                # still count indirect adds — but the norm cares about direct
                # too. We treat indirect as not-directly-authored.
                # `// indirect` suffix
                is_indirect = "// indirect" in body
                if is_indirect:
                    # Don't count purely indirect adds — those are go mod tidy
                    # bookkeeping.
                    continue
                n_deps += 1
                n_classified += 1
                n_pinned += 1
    return n_deps, n_pinned, n_classified


# Pipfile -------------------------------------------------------------------

def _parse_pipfile_added(added: str) -> Tuple[int, int, int]:
    """Pipfile is TOML with [packages] and [dev-packages]."""
    n_deps = n_pinned = n_classified = 0
    current_section: Optional[str] = None
    for raw in added.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("[") and s.endswith("]"):
            current_section = s.strip("[]")
            continue
        if current_section is None:
            continue
        if current_section not in ("packages", "dev-packages"):
            continue
        if "=" not in s:
            continue
        name, _, rhs = s.partition("=")
        name = name.strip()
        rhs = rhs.strip().rstrip(",")
        if not name or name.startswith("["):
            continue
        n_deps += 1
        ver = _extract_toml_version(rhs) or rhs.strip('"\'')
        if not ver:
            continue
        n_classified += 1
        # Pipfile uses pip-style: `==1.0` pinned, `"*"` unpinned, `">=1"` unpinned.
        if ver.startswith("==") or ver.startswith("==="):
            n_pinned += 1


    return n_deps, n_pinned, n_classified


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _all_changed_paths(diff_text: str) -> Set[str]:
    """Every file path that appears in a `diff --git` header (added or modified).

    Used for the lock-sync sub-score: did the diff also touch the lockfile
    that pairs with a modified manifest?
    """
    out: Set[str] = set()
    idx = diff_text.find("diff --git")
    if idx == -1:
        return out
    # REGEX_OK: format_header — `diff --git a/X b/Y` is a fixed git header.
    for m in re.finditer(r"^diff --git a/(\S+) b/(\S+)", diff_text[idx:],
                         flags=re.MULTILINE):
        out.add(m.group(2))
    return out


def _per_file_diff_block(diff_text: str, target_path: str) -> str:
    """Return the lines of the diff that pertain to a single file.

    We split on `diff --git` headers and return the chunk whose header
    contains the target path. Used by JSON / go.mod parsers that need
    surrounding context lines (not just additions).
    """
    idx = diff_text.find("diff --git")
    if idx == -1:
        return ""
    blocks = diff_text[idx:].split("\ndiff --git ")
    # First element already starts with `diff --git`; rest were split, so
    # re-prepend for uniformity.
    norm_blocks: List[str] = []
    for i, b in enumerate(blocks):
        if i == 0:
            norm_blocks.append(b)
        else:
            norm_blocks.append("diff --git " + b)
    for b in norm_blocks:
        # check header line for path
        first_line = b.split("\n", 1)[0]
        # REGEX_OK: format_header — `diff --git a/X b/Y` header.
        m = re.match(r"diff --git a/(\S+) b/(\S+)", first_line)
        if not m:
            continue
        if m.group(2) == target_path or m.group(1) == target_path:
            return b
    return ""


def _classify_manifests(added: Dict[str, str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {
        "req": [], "pyproject": [], "package": [], "cargo": [],
        "gomod": [], "pipfile": [],
    }
    for path in added.keys():
        if _is_requirements_txt(path):
            out["req"].append(path)
        elif _is_pyproject(path):
            out["pyproject"].append(path)
        elif _is_package_json(path):
            out["package"].append(path)
        elif _is_cargo_toml(path):
            out["cargo"].append(path)
        elif _is_go_mod(path):
            out["gomod"].append(path)
        elif _is_pipfile(path):
            out["pipfile"].append(path)
    return out


def _has_any_dep_addition(added: Dict[str, str], diff_text: str) -> bool:
    """Cheap check used by applies(): does any manifest's added text contain
    at least one dep line?"""
    bins = _classify_manifests(added)
    if bins["req"]:
        for p in bins["req"]:
            d, _, _ = _parse_requirements_added3(added[p])
            if d > 0:
                return True
    if bins["pyproject"]:
        for p in bins["pyproject"]:
            d, _, _ = _parse_pyproject_added(
                added[p], _per_file_diff_block(diff_text, p))
            if d > 0:
                return True
    if bins["package"]:
        for p in bins["package"]:
            d, _, _ = _parse_package_json_added(
                added[p], _per_file_diff_block(diff_text, p))
            if d > 0:
                return True
    if bins["cargo"]:
        for p in bins["cargo"]:
            d, _, _ = _parse_cargo_added(
                added[p], _per_file_diff_block(diff_text, p))
            if d > 0:
                return True
    if bins["gomod"]:
        for p in bins["gomod"]:
            d, _, _ = _parse_go_mod_added(
                added[p], _per_file_diff_block(diff_text, p))
            if d > 0:
                return True
    if bins["pipfile"]:
        for p in bins["pipfile"]:
            d, _, _ = _parse_pipfile_added(added[p])
            if d > 0:
                return True
    return False


def applies(diff_text: str) -> bool:
    added = parse_diff_added_by_file(diff_text)
    if not added:
        return False
    return _has_any_dep_addition(added, diff_text)


def score(diff_text: str) -> Optional[float]:
    added = parse_diff_added_by_file(diff_text)
    if not added:
        return None
    bins = _classify_manifests(added)
    total_deps = total_pinned = total_classified = 0
    manifest_basenames_seen: Set[str] = set()

    for p in bins["req"]:
        d, pi, c = _parse_requirements_added3(added[p])
        total_deps += d; total_pinned += pi; total_classified += c
        if d > 0:
            manifest_basenames_seen.add("requirements.txt")
    for p in bins["pyproject"]:
        d, pi, c = _parse_pyproject_added(
            added[p], _per_file_diff_block(diff_text, p))
        total_deps += d; total_pinned += pi; total_classified += c
        if d > 0:
            manifest_basenames_seen.add("pyproject.toml")
    for p in bins["package"]:
        d, pi, c = _parse_package_json_added(
            added[p], _per_file_diff_block(diff_text, p))
        total_deps += d; total_pinned += pi; total_classified += c
        if d > 0:
            manifest_basenames_seen.add("package.json")
    for p in bins["cargo"]:
        d, pi, c = _parse_cargo_added(
            added[p], _per_file_diff_block(diff_text, p))
        total_deps += d; total_pinned += pi; total_classified += c
        if d > 0:
            manifest_basenames_seen.add("cargo.toml")
    for p in bins["gomod"]:
        d, pi, c = _parse_go_mod_added(
            added[p], _per_file_diff_block(diff_text, p))
        total_deps += d; total_pinned += pi; total_classified += c
        if d > 0:
            manifest_basenames_seen.add("go.mod")
    for p in bins["pipfile"]:
        d, pi, c = _parse_pipfile_added(added[p])
        total_deps += d; total_pinned += pi; total_classified += c
        if d > 0:
            manifest_basenames_seen.add("pipfile")

    if total_deps == 0:
        return None

    # ---- s_minimal ----
    excess = max(0, total_deps - 1)
    s_minimal = math.exp(-excess / 5.0)

    # ---- s_pinned ----
    if total_classified > 0:
        s_pinned = total_pinned / total_classified
    else:
        # No specs we could classify (all bare names / git URLs). Treat as
        # neutral rather than as a violation — punishing here would amplify
        # the minimization penalty for genuinely package-manager-shaped diffs.
        s_pinned = 1.0

    # ---- s_locksync ----
    all_changed = {_basename(p).lower() for p in _all_changed_paths(diff_text)}
    s_locksync = 1.0
    for mb in manifest_basenames_seen:
        expected = LOCKFILES_BY_MANIFEST.get(mb, set())
        if not expected:
            continue
        if any(lf in all_changed for lf in expected):
            continue
        # Manifest changed, no corresponding lockfile in this diff at all.
        # Soft penalty: 0.7 multiplier per missing-lock manifest, capped at
        # 0.5 floor for many missing.
        s_locksync *= 0.7
    if s_locksync < 0.5:
        s_locksync = 0.5

    final = s_minimal * (0.5 + 0.5 * s_pinned) * s_locksync
    if final < 0.0:
        final = 0.0
    if final > 1.0:
        final = 1.0
    return float(final)
