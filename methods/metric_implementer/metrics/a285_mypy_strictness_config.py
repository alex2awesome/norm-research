"""a285: Mypy strictness configuration.

Aspect (from aspects.json[285]):
  "Configure Python static typing rigor in mypy: explicit package/module
   bases, disallow bare generics and unsafe Optional/None usage, and respect
   known unsupported inline annotation forms/untyped decorators/missing
   imports policies."

This is a META-norm about how mypy is *configured*, not about whether the
project's source actually passes mypy. A PR conforms when it adds (or
strengthens) the strictness knobs in the project's mypy configuration —
i.e. enables flags that bring the project closer to `strict = true`. A PR
violates when it weakens those knobs (e.g. flips `strict = true` off, or
adds `ignore_errors = true` / global `disable_error_code = [...]`).

Detection (applies):
  The diff adds or modifies one of:
    1. A standalone mypy config file: `mypy.ini`, `.mypy.ini` (INI format).
    2. A `pyproject.toml` whose added lines include a `[tool.mypy]` (or
       nested `[tool.mypy.overrides]`) section.
    3. A `setup.cfg` whose added lines include a `[mypy]` section.

  All checks are diff-level — no need to inspect repo state.

Scoring:
  We extract the set of mypy options *added* to the config and compute a
  strictness score from a fixed registry of known mypy flags. Each flag
  carries a sign (+1 if turning it on tightens checking; -1 if it loosens)
  and a weight (1 for ordinary flags; 2 for `strict = true` since it
  encompasses many flags at once; 2 for negative blanket flags like
  `ignore_errors = true` that wholesale disable checking).

  raw = sum_over_added_flags(sign * weight * (1 if "on" else -1))
  cap = sum_over_added_flags(weight)
  score = (raw / cap + 1) / 2  ∈ [0, 1]

  So an all-tightening diff → 1.0; an all-loosening diff → 0.0; a mix
  trends to 0.5. If no recognized strictness flags appear despite a mypy
  config being touched (e.g. only `python_version` or `files` set), score
  abstains (returns None).

Tooling:
  - stdlib `tomllib` (py 3.11+) for pyproject.toml parsing, with `tomli`
    fallback for 3.10. We parse the FULL added pyproject text into a
    TOML document and walk `tool.mypy` / `tool.mypy.overrides[*]`.
  - stdlib `configparser` for mypy.ini / setup.cfg parsing.
  - We do NOT shell out to `mypy` itself — it would require a working
    project and is overkill for "did the config get stricter."

Tier 2 (parser-based, no subprocess). PARTIALLY_THIN: known flag presence
is a thin proxy; whether the strictness produces *useful* findings on the
codebase is THICK and outside scope.

Applicability is narrow by design: only PRs that touch a mypy config
section register. Most PRs abstain via applies()=False.
"""
from __future__ import annotations

import configparser
import io
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a285"
ASPECT_NAME = "Mypy strictness configuration"
TIER = 2
TOOLS = ["tomllib", "configparser"]  # stdlib only
APPLIES_TO_LANGS = ["TOML", "INI", "Python"]
CLASSIFICATION = "PARTIALLY_THIN"


# ---------------------------------------------------------------------------
# Mypy strictness flag registry.
#
# `sign`:
#   +1: turning the flag ON tightens checking (e.g. disallow_untyped_defs)
#   -1: turning the flag ON loosens checking (e.g. ignore_errors)
#
# `weight`:
#   1  for normal flags
#   2  for the umbrella `strict` flag (it implies many others)
#   2  for blanket loosen-flags (`ignore_errors`, `follow_imports = skip`)
#
# Keys are normalized to lowercase. Values from TOML are typed (bool/str);
# values from INI/CFG are raw strings — we normalize both via `_truthy`.
# ---------------------------------------------------------------------------

# (sign, weight)
TIGHTEN_FLAGS: Dict[str, Tuple[int, int]] = {
    # Umbrella
    "strict": (+1, 2),
    # Untyped-def family
    "disallow_untyped_defs": (+1, 1),
    "disallow_incomplete_defs": (+1, 1),
    "disallow_untyped_calls": (+1, 1),
    "disallow_untyped_decorators": (+1, 1),
    "check_untyped_defs": (+1, 1),
    # Generic / Any restrictions
    "disallow_any_generics": (+1, 1),
    "disallow_any_explicit": (+1, 1),
    "disallow_any_expr": (+1, 1),
    "disallow_any_decorated": (+1, 1),
    "disallow_any_unimported": (+1, 1),
    "disallow_subclassing_any": (+1, 1),
    # Optional / None
    "strict_optional": (+1, 1),
    "no_implicit_optional": (+1, 1),
    "strict_equality": (+1, 1),
    "strict_concatenate": (+1, 1),
    # Warnings escalated
    "warn_return_any": (+1, 1),
    "warn_unused_ignores": (+1, 1),
    "warn_unused_configs": (+1, 1),
    "warn_redundant_casts": (+1, 1),
    "warn_unreachable": (+1, 1),
    "warn_no_return": (+1, 1),
    # Package / module bases (explicit per aspect description)
    "namespace_packages": (+1, 1),
    "explicit_package_bases": (+1, 1),
    # Implicit reexport — turning OFF tightens; treated below specially
    # ("implicit_reexport = false" is a tighten). We handle by recording
    # the flag with sign=-1 so ON=loosen, OFF=tighten.
    "implicit_reexport": (-1, 1),
    # extra_checks (recent mypy strictness umbrella add-on)
    "extra_checks": (+1, 1),
    # Pretty / report-only flags are NOT scored — they're cosmetic.
}

LOOSEN_FLAGS: Dict[str, Tuple[int, int]] = {
    # Blanket loosen
    "ignore_errors": (-1, 2),
    "ignore_missing_imports": (-1, 1),
    # follow_imports = "skip" is a wholesale loosen; we score the *value*
    # via _follow_imports_sign rather than ON/OFF.
}

# Bidirectional: depend on value, not on boolean.
VALUE_FLAGS: Set[str] = {"follow_imports", "disable_error_code",
                         "enable_error_code"}


# ---------------------------------------------------------------------------
# Value normalization
# ---------------------------------------------------------------------------

_TRUE_TOKENS = {"1", "true", "yes", "on", "t", "y"}
_FALSE_TOKENS = {"0", "false", "no", "off", "f", "n"}


def _truthy(v) -> Optional[bool]:
    """Coerce a TOML/INI value to bool. Returns None if not boolean-like."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return v != 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _TRUE_TOKENS:
            return True
        if s in _FALSE_TOKENS:
            return False
    return None


# ---------------------------------------------------------------------------
# TOML parsing for pyproject.toml
# ---------------------------------------------------------------------------

def _tomllib():
    try:
        import tomllib  # py3.11+
        return tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
            return tomllib
        except ImportError:
            return None


def _pyproject_mypy_options(added: str) -> Dict[str, object]:
    """Parse added pyproject.toml text; return flat dict of mypy options.

    Returns the union of:
      - tool.mypy.<key>          → key
      - tool.mypy.overrides[i].<key> → key  (any override entry contributes
        its overrides to the same flat namespace; we don't track which
        module they apply to — the norm is "is mypy being made stricter"
        in aggregate)
    """
    tomllib = _tomllib()
    if tomllib is None:
        return {}
    try:
        data = tomllib.loads(added)
    except Exception:
        return {}
    out: Dict[str, object] = {}
    tool = data.get("tool", {}) if isinstance(data, dict) else {}
    if not isinstance(tool, dict):
        return {}
    mypy = tool.get("mypy", {})
    if not isinstance(mypy, dict):
        return {}
    for k, v in mypy.items():
        if k == "overrides":
            continue
        out[k.lower()] = v
    overrides = mypy.get("overrides", [])
    if isinstance(overrides, list):
        for ov in overrides:
            if not isinstance(ov, dict):
                continue
            for k, v in ov.items():
                if k == "module":
                    continue
                # Overrides may conflict; we keep the *last* one. This is
                # OK because we score the aggregate strictness signal, not
                # any per-module decision.
                out[k.lower()] = v
    return out


# ---------------------------------------------------------------------------
# INI / CFG parsing for mypy.ini and setup.cfg
# ---------------------------------------------------------------------------

def _ini_mypy_options(added: str, section_filter: str) -> Dict[str, object]:
    """Parse INI/CFG; return flat dict from any section whose name starts
    with `section_filter`.

    For mypy.ini, section_filter="mypy"  → catches "[mypy]" and the
    per-module "[mypy-foo.bar]" sections.
    For setup.cfg the same filter is used; setup.cfg's [mypy] section
    behaves identically.
    """
    parser = configparser.ConfigParser(
        interpolation=None, strict=False, allow_no_value=True,
    )
    # configparser fails on duplicate sections in some configs; suppress
    # via strict=False above. Wrap in StringIO.
    try:
        parser.read_file(io.StringIO(added))
    except configparser.Error:
        return {}
    out: Dict[str, object] = {}
    for sect in parser.sections():
        # Normalize: strict_filter matches "mypy" and "mypy-foo.bar".
        sl = sect.strip().lower()
        if sl != section_filter and not sl.startswith(section_filter + "-"):
            continue
        for k, v in parser.items(sect):
            out[k.strip().lower()] = v if v is not None else ""
    return out


# ---------------------------------------------------------------------------
# Per-flag scoring
# ---------------------------------------------------------------------------

def _follow_imports_sign(value) -> Optional[Tuple[int, int]]:
    """follow_imports has 4 values:
      normal | silent | skip | error
    `skip` is a wholesale loosen (treat all unfound imports as Any without
    even type-checking them); `silent` is mild loosen; `error` is the
    strict choice; `normal` is the default (no signal).
    """
    if not isinstance(value, str):
        return None
    v = value.strip().strip('"').strip("'").lower()
    if v == "skip":
        return (-1, 2)
    if v == "silent":
        return (-1, 1)
    if v == "error":
        return (+1, 1)
    return None


def _flag_contribution(key: str, value) -> Optional[Tuple[int, int]]:
    """Map (key, value) → (signed_unit, weight), or None if not strictness-
    relevant.

    `signed_unit` is in {-w, +w}: positive = strictness-increasing, negative
    = strictness-decreasing. The runner then sums and normalizes.
    """
    k = key.lower()

    if k in TIGHTEN_FLAGS:
        sign, weight = TIGHTEN_FLAGS[k]
        t = _truthy(value)
        if t is None:
            return None
        # turning ON (t=True): contribute +sign*weight
        contrib = (sign if t else -sign) * weight
        return contrib, weight

    if k in LOOSEN_FLAGS:
        sign, weight = LOOSEN_FLAGS[k]  # sign is -1
        t = _truthy(value)
        if t is None:
            return None
        # turning ON a loosen flag: -1 * weight (loosen)
        # turning OFF a loosen flag: +1 * weight (tighten)
        contrib = (sign if t else -sign) * weight
        return contrib, weight

    if k == "follow_imports":
        res = _follow_imports_sign(value)
        if res is None:
            return None
        signed, w = res  # already signed
        return signed * w, w

    if k == "disable_error_code":
        # Turning off error codes loosens. Score by count of codes.
        codes = _split_codes(value)
        if not codes:
            return None
        # cap weight at 2 so single big lists don't dominate.
        w = min(len(codes), 2)
        return -1 * w, w

    if k == "enable_error_code":
        codes = _split_codes(value)
        if not codes:
            return None
        w = min(len(codes), 2)
        return +1 * w, w

    return None


def _split_codes(value) -> List[str]:
    """disable_error_code may be a string (INI) or a list (TOML)."""
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        # comma-separated in INI
        return [c.strip() for c in value.split(",") if c.strip()]
    return []


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def _collect_mypy_options(diff_text: str) -> Dict[str, object]:
    """Walk the diff; merge all mypy options surfaced in added lines.

    Sources:
      - pyproject.toml  → tool.mypy + overrides
      - mypy.ini / .mypy.ini → [mypy] + [mypy-*]
      - setup.cfg → [mypy] + [mypy-*]
    """
    by_path = parse_diff_added_by_file(diff_text)
    merged: Dict[str, object] = {}
    for path, added in by_path.items():
        low = path.lower()
        base = low.rsplit("/", 1)[-1]
        if base.endswith("pyproject.toml"):
            opts = _pyproject_mypy_options(added)
            # Only count if the added text actually contained a tool.mypy
            # section — otherwise tomllib's empty parse gives {}.
            if opts:
                merged.update(opts)
        elif base in ("mypy.ini", ".mypy.ini"):
            opts = _ini_mypy_options(added, "mypy")
            if opts:
                merged.update(opts)
        elif base == "setup.cfg":
            opts = _ini_mypy_options(added, "mypy")
            if opts:
                merged.update(opts)
    return merged


def _applies_by_path(diff_text: str) -> bool:
    """Diff-level applicability gate: did the diff add to a recognized
    mypy config surface?

    We use cheap text checks (the file is a config and the added text
    contains a `[tool.mypy]` / `[mypy]` header). Full parsing happens
    in score().
    """
    by_path = parse_diff_added_by_file(diff_text)
    for path, added in by_path.items():
        low = path.lower()
        base = low.rsplit("/", 1)[-1]
        if base.endswith("pyproject.toml") and "[tool.mypy" in added:
            return True
        if base in ("mypy.ini", ".mypy.ini") and "[mypy" in added:
            return True
        if base == "setup.cfg" and "[mypy" in added:
            return True
    return False


def applies(diff_text: str) -> bool:
    return _applies_by_path(diff_text)


def score(diff_text: str) -> Optional[float]:
    if not applies(diff_text):
        return None
    opts = _collect_mypy_options(diff_text)
    if not opts:
        return None

    raw = 0
    cap = 0
    n_scored = 0
    for k, v in opts.items():
        contrib = _flag_contribution(k, v)
        if contrib is None:
            continue
        signed, w = contrib
        raw += signed
        cap += w
        n_scored += 1

    if n_scored == 0 or cap == 0:
        # mypy config was touched but no recognized strictness flag was
        # added/changed (e.g. only `python_version` or `files = ...`).
        # We cannot say strictness moved either way.
        return None

    # Normalize [-cap, +cap] → [0, 1]
    norm = (raw / cap + 1.0) / 2.0
    if norm < 0.0:
        norm = 0.0
    if norm > 1.0:
        norm = 1.0
    return float(norm)
