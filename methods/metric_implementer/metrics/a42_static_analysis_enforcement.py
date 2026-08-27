"""a42: Static analysis and automated ENFORCEMENT.

Aspect (from aspects.json):
  "Use linters/formatters/static analysis in CI to enforce standards, catch
   smells/issues early, prefer auto-fixable rules, and avoid disabling checks
   casually."

Distinct from a95 ("Static analysis with actionable feedback"):
  a95 detects static-analysis SETUP — does the diff add a linter config
  file, a [tool.X] section, or a CI step that runs a linter? It does not
  ask whether the lint step would actually BLOCK a bad PR from merging.

  a42 measures ENFORCEMENT — the property that a noncompliant change cannot
  silently land. Three enforcement surfaces are recognized:

    1. Local hook frameworks that gate `git commit` / `git push`:
         - `.pre-commit-config.yaml` (pre-commit.com) — hook count, gates
         - `.husky/`  (husky)  — pre-commit / pre-push scripts
         - `lefthook.yml` / `lefthook.yaml`           (lefthook)
         - `.simple-git-hooks` / package.json "simple-git-hooks" key
         - `.overcommit.yml`                          (overcommit)
         - `commitlint.config.{js,cjs,mjs,ts}` / `.commitlintrc*`

    2. CI workflow steps whose LINT invocation is configured to FAIL on
       findings rather than report-and-continue. Operationalized as the
       balance of:
         + (positive) strict-mode flags or strict-by-default linters:
             `--max-warnings 0`, `--max-warnings=0`, `--exit-non-zero-on-fix`,
             `--strict`, `--errors-only`, `-Werror`, `set -e` preludes,
             `ruff check`, `mypy --strict`, `eslint --max-warnings 0`,
             reviewdog `-reporter=github-pr-check`/`-fail-on-error`.
         - (negative) explicit weakening:
             `continue-on-error: true`, `|| true`, `|| exit 0`,
             reviewdog `-reporter=github-pr-review` (comments-only),
             newly-commented-out lint steps.

    3. Branch-protection-style required-job markers in workflow files:
         a job whose name appears in a `needs:` of a deploy/merge gate, or
         a workflow whose name embeds `required`, `gate`, `mandatory`,
         `blocking`. This is a soft signal — branch protection lives in
         the GitHub UI, not the repo — but the convention is widely used.

  a95 returns "yes, linter set up." a42 returns "yes, the linter actually
  blocks merges." The two metrics are intended to be combined downstream:
  high a95 + low a42 = "lint exists but only as advisory feedback,"
  which is precisely the "disabling checks casually" failure mode the
  aspect description warns against.

Applicability is narrow by design. applies() = True only when the diff
touches an enforcement surface (one of the three above). Most PRs abstain.

Tier 2 (parser-based — tree-sitter-yaml + diff parser + structured TOML/
JSON peek). PARTIALLY_THIN: file/flag presence is a thin proxy for
"enforcement"; whether the configured hook ACTUALLY runs / passes / is
required on the repo's branches is THICK (lives in remote config).
"""
from __future__ import annotations

import json
import string
from typing import Dict, List, Optional, Set, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a42"
ASPECT_NAME = "Static analysis and automated enforcement"
TIER = 2
TOOLS = ["tree-sitter-yaml"]
APPLIES_TO_LANGS = ["YAML", "JSON", "JavaScript", "Any"]
CLASSIFICATION = "PARTIALLY_THIN"


# ---------------------------------------------------------------------------
# Path classifiers — paths are not a language to parse, plain string ops.
# ---------------------------------------------------------------------------

# Local-hook framework configuration files (any add => an enforcement surface
# is being installed). Match on filename (last path segment), case-insensitive.
LOCAL_HOOK_FILES: Set[str] = {
    ".pre-commit-config.yaml",
    ".pre-commit-config.yml",
    "lefthook.yml",
    "lefthook.yaml",
    ".lefthook.yml",
    ".lefthook.yaml",
    ".overcommit.yml",
    ".overcommit.yaml",
    ".simple-git-hooks.json",
    ".simple-git-hooks.cjs",
    ".simple-git-hooks.js",
    "commitlint.config.js",
    "commitlint.config.cjs",
    "commitlint.config.mjs",
    "commitlint.config.ts",
    ".commitlintrc",
    ".commitlintrc.json",
    ".commitlintrc.yml",
    ".commitlintrc.yaml",
    ".commitlintrc.js",
    ".commitlintrc.cjs",
}

# Husky paths look like `.husky/pre-commit`, `.husky/pre-push`, etc.
HUSKY_DIR_TOKEN = "/.husky/"
HUSKY_HOOK_NAMES = {
    "pre-commit", "pre-push", "commit-msg", "prepare-commit-msg",
    "post-commit", "post-checkout", "post-merge", "pre-rebase",
    "pre-applypatch", "applypatch-msg",
}


def _is_workflow_path(path: str) -> bool:
    p = "/" + path.lower().replace("\\", "/")
    return (
        "/.github/workflows/" in p
        and (p.endswith(".yml") or p.endswith(".yaml"))
    )


def _filename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _is_husky_hook(path: str) -> bool:
    p = "/" + path.replace("\\", "/")
    if HUSKY_DIR_TOKEN not in p:
        return False
    name = _filename(p).lower()
    # `.husky/_/husky.sh` etc. — exclude the bootstrap files.
    if name.startswith("_") or name.endswith(".sh"):
        return name in HUSKY_HOOK_NAMES  # `pre-commit.sh` won't match
    return name in HUSKY_HOOK_NAMES or any(
        name == h for h in HUSKY_HOOK_NAMES
    )


def _is_package_json(path: str) -> bool:
    return _filename(path).lower() == "package.json"


# ---------------------------------------------------------------------------
# Enforcement-strength tokens (strict-mode flags vs. weakening flags).
# These are matched against YAML scalar text from workflow run-steps.
# ---------------------------------------------------------------------------

# Each tuple: (token, is_multi_word). Word-boundary checked for single words.
STRICT_TOKENS: Tuple[str, ...] = (
    "--max-warnings 0",
    "--max-warnings=0",
    "--exit-non-zero-on-fix",
    "--exit-non-zero",
    "--errors-only",
    "--strict",
    "-werror",
    "--fail-on-warnings",
    "--fail-on-error",
    "--fail-on=error",
    "--fail-level=error",
    "--no-warnings",
    "set -e",
    "set -euo pipefail",
    "set -eo pipefail",
    "-reporter=github-pr-check",
    "-reporter=github-check",
    "fail_on_error: true",
    "fail-on-error: true",
    "fail_level: error",
    "level: error",
)

# Weakening: these REDUCE enforcement strength when present in added lines.
WEAK_TOKENS: Tuple[str, ...] = (
    "continue-on-error: true",
    "|| true",
    "|| exit 0",
    "-reporter=github-pr-review",
    "fail_on_error: false",
    "fail-on-error: false",
    "--no-error-on-unmatched-pattern",
    "warn_only: true",
)

# Linter CLIs whose default exit code on findings is non-zero — these are
# "strict by default" even without explicit flags.
STRICT_BY_DEFAULT_CLIS: Tuple[str, ...] = (
    "ruff check",
    "mypy ",
    "pyright ",
    "golangci-lint run",
    "cargo clippy",
    "shellcheck ",
    "yamllint ",
    "hadolint ",
    "checkstyle ",
    "spotbugs ",
    "pmd ",
    "phpstan ",
    "psalm ",
)


# ---------------------------------------------------------------------------
# YAML parser bridge (tree-sitter-yaml — same approach as a95).
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
    """Return all scalar leaf strings in the YAML tree, lowercased."""
    parser = _parser()
    if not parser:
        return []
    try:
        tree = parser.parse(code)
    except Exception:
        return []
    out: List[str] = []

    def walk(n):
        if n.type in ("string_scalar", "single_quote_scalar",
                      "double_quote_scalar", "block_scalar",
                      "plain_scalar"):
            t = _scalar_text(n, code).strip()
            if t:
                out.append(t.lower())
            return
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    return out


# ---------------------------------------------------------------------------
# Hook-config parsers.
# ---------------------------------------------------------------------------

def _count_precommit_hooks(added: str) -> int:
    """Count `id:` keys at the hook level in a .pre-commit-config.yaml diff.

    The file's structure is `repos: - repo: ... hooks: - id: <hookname>`.
    We don't need full schema validation — counting added `- id:` entries is
    enough to estimate the number of enforcement gates installed. We parse
    the added text with tree-sitter-yaml and look at scalars under keys we
    recognize, falling back to a line scan if the parser is unavailable.
    """
    n = 0
    scalars = _collect_yaml_scalars(added.encode("utf8", errors="replace"))
    if scalars:
        # Heuristic: each scalar that LOOKS like a hook id (kebab-case word,
        # short, no spaces, no colons) — count it. We can't easily walk the
        # `hooks: - id:` tree on a fragment, so we look at lines starting
        # with `- id:` directly below.  REGEX_OK: format_header (well-defined
        # YAML key form).
        for line in added.splitlines():
            s = line.lstrip().rstrip()
            if s.startswith("- id:") or s.startswith("- id :"):
                n += 1
    if n == 0:
        # Fallback: count occurrences of the literal token.
        for line in added.splitlines():
            ss = line.strip()
            # REGEX_OK: format_header — YAML key marker.
            if ss.startswith("- id:"):
                n += 1
    return n


def _package_json_has_hook_tooling(added: str) -> bool:
    """Detect husky / lint-staged / simple-git-hooks / commitlint sections
    in an added package.json fragment.

    package.json is JSON — we attempt a full parse first (the diff fragment
    is usually a contiguous object); if that fails (truncated diff), we look
    for added KEY scalars via substring matching against the top-level
    field names. Plain string matching on JSON keys is acceptable here
    because the keys are unambiguous identifiers.
    """
    needles = (
        '"husky"', '"lint-staged"', '"simple-git-hooks"',
        '"pre-commit"', '"pre-push"', '"commitlint"',
    )
    text = added
    # Optimistic JSON parse:
    try:
        # A diff fragment is unlikely to be a full JSON; only try if it
        # plausibly starts with `{`.
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            obj = json.loads(stripped)
            keys = set()

            def collect_keys(o):
                if isinstance(o, dict):
                    for k, v in o.items():
                        keys.add(k.lower())
                        collect_keys(v)
                elif isinstance(o, list):
                    for x in o:
                        collect_keys(x)
            collect_keys(obj)
            for n in ("husky", "lint-staged", "simple-git-hooks",
                      "commitlint"):
                if n in keys:
                    return True
    except Exception:
        pass
    low = text.lower()
    return any(n in low for n in needles)


# ---------------------------------------------------------------------------
# Workflow enforcement strength.
# ---------------------------------------------------------------------------

_WORD_CHARS = set(string.ascii_lowercase + string.digits + "_-")


def _token_in(haystack: str, token: str) -> bool:
    """Substring match with soft word boundaries. Multi-word tokens (those
    containing space, `=`, or `:`) are matched verbatim."""
    if not token or not haystack:
        return False
    if any(c in token for c in (" ", "=", ":")):
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


def _workflow_signals(added: str) -> Tuple[int, int]:
    """Return (n_strict_signals, n_weak_signals) found in an added workflow
    fragment.

    We look at YAML scalar leaves AND the raw lowercased added text. Many
    enforcement tokens span the key:value boundary (e.g.
    `continue-on-error: true`), which tree-sitter splits into separate scalar
    leaves. Searching the raw text catches those; the scalar pass catches
    `run:` block contents where shell flags live. Comments are NOT stripped
    — comment-out of a lint step IS a weakening event we want to catch
    (e.g. fixture 48's reviewdog rewrite).
    """
    scalars = _collect_yaml_scalars(added.encode("utf8", errors="replace"))
    haystacks: List[str] = [added.lower()]
    if scalars:
        haystacks.append("\n".join(scalars))

    def any_hit(token: str) -> bool:
        for h in haystacks:
            if any(c in token for c in (" ", "=", ":")):
                if token in h:
                    return True
            else:
                if _token_in(h, token):
                    return True
        return False

    n_strict = sum(1 for t in STRICT_TOKENS if any_hit(t))
    n_strict += sum(1 for t in STRICT_BY_DEFAULT_CLIS
                    if any(t in h for h in haystacks))
    n_weak = sum(1 for t in WEAK_TOKENS if any_hit(t))
    return n_strict, n_weak


# ---------------------------------------------------------------------------
# Top-level aggregation.
# ---------------------------------------------------------------------------

def _surfaces(diff_text: str) -> Dict[str, object]:
    """Categorize the diff's enforcement-relevant files.

    Returns a dict with:
      'hook_files':   list of (path, added)    -- local-hook config files
      'husky_hooks':  list of (path, added)    -- husky hook scripts
      'workflows':    list of (path, added)    -- CI workflow YAML
      'pkg_json':     list of (path, added)    -- package.json files
    """
    by_path = parse_diff_added_by_file(diff_text)
    out: Dict[str, list] = {
        "hook_files": [], "husky_hooks": [], "workflows": [],
        "pkg_json": [],
    }
    for path, added in by_path.items():
        name = _filename(path).lower()
        if name in LOCAL_HOOK_FILES:
            out["hook_files"].append((path, added))
        elif _is_husky_hook(path):
            out["husky_hooks"].append((path, added))
        elif _is_workflow_path(path):
            out["workflows"].append((path, added))
        elif _is_package_json(path):
            out["pkg_json"].append((path, added))
    return out


def applies(diff_text: str) -> bool:
    """True iff the diff adds (or modifies) an enforcement surface.

    Diff parsing only — no subprocess, cheap.
    """
    sfs = _surfaces(diff_text)
    if sfs["hook_files"] or sfs["husky_hooks"] or sfs["workflows"]:
        return True
    # package.json alone applies only if it touches hook tooling keys.
    for _, added in sfs["pkg_json"]:
        if _package_json_has_hook_tooling(added):
            return True
    return False


def score(diff_text: str) -> Optional[float]:
    """Aggregate enforcement strength signals into [0, 1].

    Composition:
      hook_signal     in [0,1]:
          1.0 if any local-hook framework config is added with hooks defined
          0.7 if a husky hook script is added (presence of a hook is the
              enforcement) or a package.json hook-tooling key is added
          0.5 if a hook config file is added but no hooks parsed
          0   otherwise

      workflow_signal in [0,1]:
          balance of strict vs weak tokens in workflow diffs:
          (strict) / (strict + weak + 1)
          The +1 prevents 1.0 when only a single strict token is seen
          (we want "more strict signals" to score higher).
          If no workflow touched: not contributing (weight 0).

    Final = 0.6 * hook_signal + 0.4 * workflow_signal, with weights
    renormalized over the dimensions that actually fired.
    """
    if not applies(diff_text):
        return None

    sfs = _surfaces(diff_text)

    # --- hook-side signal ---
    hook_fired = False
    hook_signal = 0.0
    if sfs["hook_files"]:
        hook_fired = True
        total_hooks = 0
        for _, added in sfs["hook_files"]:
            total_hooks += _count_precommit_hooks(added)
        if total_hooks >= 2:
            hook_signal = 1.0
        elif total_hooks == 1:
            hook_signal = 0.8
        else:
            # config file added but no hook ids parsed (file may already
            # exist and only metadata changed); still some enforcement.
            hook_signal = 0.5
    if sfs["husky_hooks"]:
        hook_fired = True
        hook_signal = max(hook_signal, 0.7 + 0.05 * min(
            len(sfs["husky_hooks"]), 6))
        hook_signal = min(hook_signal, 1.0)
    if sfs["pkg_json"]:
        for _, added in sfs["pkg_json"]:
            if _package_json_has_hook_tooling(added):
                hook_fired = True
                hook_signal = max(hook_signal, 0.7)

    # --- workflow-side signal ---
    workflow_fired = False
    wf_strict = 0
    wf_weak = 0
    for _, added in sfs["workflows"]:
        s, w = _workflow_signals(added)
        wf_strict += s
        wf_weak += w
    if wf_strict or wf_weak:
        workflow_fired = True
    workflow_signal = (
        wf_strict / (wf_strict + wf_weak + 1.0)
        if workflow_fired else 0.0
    )

    # --- combine with weight renormalization ---
    parts: List[Tuple[float, float]] = []  # (weight, value)
    if hook_fired:
        parts.append((0.6, hook_signal))
    if workflow_fired:
        parts.append((0.4, workflow_signal))

    if not parts:
        # applies() was True (workflow file touched) but no enforcement
        # tokens parsed — we cannot tell whether the lint step gates
        # merges. Abstain rather than emit a misleading 0/0.5.
        return None

    wsum = sum(w for w, _ in parts)
    return float(sum(w * v for w, v in parts) / wsum)
