"""a188: ACSL specification constructs and usage.

Aspect (from aspects.json[188]):
    "Use ACSL correctly and precisely: author contracts
     (requires/ensures/assigns/behaviors), memory allocation/leak predicates,
     and built-ins (\\old, \\result, \\at); employ ghost code and
     memory/pointer modeling (including abrupt clauses) to specify and verify
     behavior."

Scope. ACSL = ANSI/ISO C Specification Language, the formal annotation
language consumed by Frama-C. ACSL specifications live inside C-style
comments of the form ``/*@ ... */`` (block) or ``//@ ...`` (line) attached
to C/C++ functions, loops, or types. The norm is therefore **only**
meaningful on diffs that add C/C++ source containing such annotations.

Empirically, in mainstream open-source PR corpora (and the 50 fixtures in
this repo's sandbox), the fraction of PRs that ship ACSL annotations is
~0/50 — Frama-C is used almost exclusively in safety-critical embedded /
verified-software projects (e.g. EBPF verifier, seL4, contiki-ng's TCP/IP
stack, OpenSSL fork ``s2n-tls``'s C interop layer). We therefore expect
``applies()`` to fire on a vanishingly small slice. That narrowness is the
*point* of the metric — when it does fire, the signal it produces is much
sharper than any broad-coverage measurement could be.

Detection (Tier 2 — structural, no subprocess).

We do NOT regex-parse C code. We:

  1. Filter the diff to added C/C++ files (extensions in C_EXTS).
  2. For each such file, walk its added text with a small **C-comment
     scanner** that yields (kind, body, offset) for each ``/*@ ... */`` or
     ``//@ ...`` annotation. The scanner is *not* parsing the C source —
     it's parsing C's comment grammar, which is a well-defined non-code
     text format (``# REGEX_OK: format_header`` semantics, though we
     actually use a stateful scanner not a regex).
  3. For each annotation body, count occurrences of ACSL keywords from a
     curated vocabulary, partitioned into three buckets:

        CORE     — requires, ensures, assigns, behavior, assumes
        ADVANCED — \\old, \\result, \\at, \\valid, \\separated, \\fresh,
                   allocates, frees, ghost, loop invariant, loop variant,
                   loop assigns, decreases, terminates, complete behaviors,
                   disjoint behaviors, abrupt, breaks, returns, continues,
                   global invariant, type invariant, axiomatic, predicate,
                   logic, lemma, inductive
        BUILTINS — \\true, \\false, \\null, \\forall, \\exists, \\nothing,
                   \\everything, \\let, \\block_length, \\offset,
                   \\base_addr, \\initialized, \\allocable, \\freeable

ACSL applicability gate (the second, finer applicability check inside
``score()``):

  - At least 1 annotation block must exist *and* contain at least 1 CORE
    keyword. A comment ``/*@ foo */`` with no ACSL keyword is just a
    Doxygen-flavored marker, not ACSL, and we abstain.

Score. Conformance to "use ACSL correctly and precisely":

  Let n_blocks    = number of annotation blocks
      n_core      = total core keywords across blocks
      n_advanced  = total advanced keywords
      n_builtins  = total builtin keywords
      n_functions = number of C/C++ functions defined in the added lines
                    (approximated via tree-sitter-c when available; falls
                    back to counting braces-with-paren heuristics otherwise)

  We score on two dimensions, each in [0, 1], then take their mean:

    coverage = clamp01(n_core_blocks / max(n_functions, n_core_blocks))
        where n_core_blocks is the number of annotation blocks that
        contain ≥1 CORE keyword. Intuition: every function defined alongside
        ACSL should have a contract — coverage measures how many do.
        When n_functions is 0 but n_core_blocks > 0, coverage = 1.0
        (the diff is pure-spec; we don't penalize that).

    richness = clamp01((n_core + n_advanced + 2 * n_builtins)
                       / (4 * n_blocks))
        Rewards specifications that use more than just bare requires/ensures
        (e.g. that reach for \\old, \\result, behaviors). Built-ins get
        2x weight because they are the strongest signal of *precise* ACSL
        (the description explicitly calls out built-ins).

  score = 0.5 * coverage + 0.5 * richness

Return states:
  - applies() = True iff at least one C/C++ file is added with at least
    one ``/*@`` or ``//@`` annotation marker present in its added text.
    (The marker alone — cheap to check, no AST.)
  - score() = float in [0, 1] when applies() is True and at least one
    annotation block contains ≥1 CORE keyword.
  - score() = None when applies() is True but no annotation block contains
    a CORE keyword (markers but no real ACSL — abstain rather than score
    0 on what may be GCC pragma comments).

Classification. ``PARTIALLY_THIN``. Detecting whether ACSL keywords are
*present* is fully deterministic. Detecting whether they are used
*correctly* in the formal sense (does ``\\old`` actually refer to a
pre-state quantity? do behaviors cover their disjoint preconditions?) would
require running Frama-C itself, which is out of sandbox scope. We measure
the thin slice — presence, density, vocabulary breadth — and leave Frama-C
WP discharge as the thick residual.

Notes on overlap. There is no overlap with ``a145`` (C buffer safety,
cppcheck) — a145 measures runtime-style safety against a defect taxonomy;
``a188`` measures formal annotation density against the ACSL keyword set.
A C file can score 1.0 on a145 (no buffer bugs) while ``a188`` abstains
(no ACSL annotations). They are orthogonal.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..sandbox import added_files_by_ext

ASPECT_ID = "a188"
ASPECT_NAME = "ACSL specification constructs and usage"
TIER = 2
TOOLS: List[str] = []  # pure-stdlib + optional tree-sitter-c (no subprocess)
APPLIES_TO_LANGS = ["C", "C++"]
CLASSIFICATION = "PARTIALLY_THIN"

C_EXTS = [".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx", ".c++"]

# ----------------------------------------------------------------------------
# ACSL keyword vocabulary
# ----------------------------------------------------------------------------

# Contract-clause kinds — the indispensable backbone of a function contract.
ACSL_CORE = {
    "requires", "ensures", "assigns", "behavior", "behaviors",
    "assumes", "exits", "allocates",
}

# Richer specification machinery: ghost code, abrupt clauses, axiomatic
# definitions, loop annotations, allocation/leak predicates. These are the
# constructs the aspect description explicitly enumerates ("ghost code",
# "memory/pointer modeling", "abrupt clauses").
ACSL_ADVANCED = {
    # built-in identifiers (with leading backslash in source — we strip it
    # before lookup)
    "old", "result", "at", "valid", "valid_read", "valid_index",
    "valid_range", "separated", "fresh",
    # clause kinds
    "frees", "ghost", "decreases", "terminates",
    "complete", "disjoint",
    # loop annotations
    "loop",  # appears in "loop invariant", "loop variant", "loop assigns"
    "invariant", "variant",
    # abrupt termination
    "breaks", "returns", "continues", "abrupt",
    # global/type invariants
    "global", "type",
    # logic-spec section keywords
    "axiomatic", "predicate", "logic", "lemma", "inductive", "axiom",
    "reads", "writes",
}

# Built-in symbols / quantifiers — strongest evidence of precise ACSL use.
ACSL_BUILTINS = {
    "true", "false", "null", "forall", "exists", "nothing", "everything",
    "let", "block_length", "offset", "base_addr", "initialized",
    "allocable", "freeable", "max", "min", "sum", "numof",
}

ALL_BACKSLASH_KEYWORDS = ACSL_BUILTINS | {
    # \old, \result, \at etc. live in ADVANCED but also start with backslash
    "old", "result", "at", "valid", "valid_read", "valid_index",
    "valid_range", "separated", "fresh",
}


# ----------------------------------------------------------------------------
# Stateful C-comment scanner (NOT a regex) — extracts /*@...*/ and //@... blocks
# ----------------------------------------------------------------------------

def _scan_annotation_blocks(code: str) -> List[str]:
    """Walk `code` as C source; yield the **bodies** of every ACSL-marked
    comment (``/*@ body */`` and ``//@ body``). This is a small state
    machine that knows about C string literals, char literals, and the two
    comment kinds; we do NOT use regex to find comments because regex over
    code conflates comment-like substrings inside strings.
    """
    out: List[str] = []
    i = 0
    n = len(code)
    while i < n:
        c = code[i]
        # String literal: skip until matching unescaped quote
        if c == '"':
            j = i + 1
            while j < n:
                if code[j] == "\\":
                    j += 2
                    continue
                if code[j] == '"':
                    j += 1
                    break
                j += 1
            i = j
            continue
        # Char literal: same idea
        if c == "'":
            j = i + 1
            while j < n:
                if code[j] == "\\":
                    j += 2
                    continue
                if code[j] == "'":
                    j += 1
                    break
                j += 1
            i = j
            continue
        # Block comment
        if c == "/" and i + 1 < n and code[i + 1] == "*":
            start = i + 2
            end = code.find("*/", start)
            if end == -1:
                # unterminated — diff truncation. Take everything we have.
                body = code[start:]
                if body.startswith("@"):
                    out.append(body[1:])
                break
            body = code[start:end]
            # ACSL marker: comment body STARTS with '@'.
            if body.startswith("@"):
                out.append(body[1:])
            i = end + 2
            continue
        # Line comment
        if c == "/" and i + 1 < n and code[i + 1] == "/":
            start = i + 2
            end = code.find("\n", start)
            if end == -1:
                end = n
            body = code[start:end]
            if body.startswith("@"):
                out.append(body[1:])
            i = end
            continue
        i += 1
    return out


# ----------------------------------------------------------------------------
# Per-block keyword counting
# ----------------------------------------------------------------------------

def _identifiers_in(body: str) -> List[str]:
    """Yield identifier-shaped tokens from an ACSL annotation body, plus
    backslash-prefixed identifiers transcribed as their bare form (we
    preserve a leading '\\' as a marker by emitting both '\\name' and
    'name' so the caller can recognize them as built-ins).

    This is a tiny scanner — ACSL annotation bodies are *not* C, so we
    don't need a full C parser. We just want identifier tokens.
    """
    out: List[str] = []
    i = 0
    n = len(body)
    while i < n:
        c = body[i]
        if c == "\\" and i + 1 < n and (body[i + 1].isalpha() or body[i + 1] == "_"):
            j = i + 1
            while j < n and (body[j].isalnum() or body[j] == "_"):
                j += 1
            out.append("\\" + body[i + 1:j])
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (body[j].isalnum() or body[j] == "_"):
                j += 1
            out.append(body[i:j])
            i = j
            continue
        i += 1
    return out


def _classify_block(body: str) -> Tuple[int, int, int]:
    """Return (n_core_hits, n_advanced_hits, n_builtins_hits) for one
    annotation body."""
    n_core = n_adv = n_built = 0
    for tok in _identifiers_in(body):
        if tok.startswith("\\"):
            bare = tok[1:]
            if bare in ACSL_BUILTINS:
                n_built += 1
            elif bare in ACSL_ADVANCED:
                # \old, \result, \at, \valid, etc. — count as advanced
                n_adv += 1
        else:
            if tok in ACSL_CORE:
                n_core += 1
            elif tok in ACSL_ADVANCED:
                n_adv += 1
            elif tok in ACSL_BUILTINS:
                # rare: bare 'forall' without backslash — still count
                n_built += 1
    return n_core, n_adv, n_built


# ----------------------------------------------------------------------------
# Function counting (tree-sitter-c if available, else cheap brace heuristic)
# ----------------------------------------------------------------------------

_C_PARSER = None
_C_PARSER_TRIED = False


def _c_parser():
    global _C_PARSER, _C_PARSER_TRIED
    if _C_PARSER_TRIED:
        return _C_PARSER
    _C_PARSER_TRIED = True
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_c as mod
        _C_PARSER = Parser(Language(mod.language()))
    except Exception:
        _C_PARSER = None
    return _C_PARSER


def _count_functions(code: str) -> int:
    """Best-effort count of C/C++ function definitions in `code`.

    Prefers tree-sitter-c. Falls back to a tiny structural counter that
    looks for tokens shaped like ``ident ( … ) {`` at top level. Not a
    regex on code — we walk character-by-character with awareness of
    strings, char literals, and comments so we don't false-positive on
    ``"foo(x){"`` inside a string.
    """
    p = _c_parser()
    if p is not None:
        try:
            tree = p.parse(code.encode("utf8", errors="replace"))
            n = 0

            def walk(node):
                nonlocal n
                if node.type == "function_definition":
                    n += 1
                for c in node.children:
                    walk(c)

            walk(tree.root_node)
            return n
        except Exception:
            pass

    # Fallback: count occurrences of ')' immediately followed (modulo
    # whitespace / annotation comments) by '{' at brace-depth 0. This is
    # an over-estimate when struct initializers like `foo = (T){...}` are
    # at top level, but C top-level usually means a function.
    i = 0
    n = len(code)
    depth = 0
    fns = 0
    while i < n:
        c = code[i]
        if c == '"' or c == "'":
            quote = c
            j = i + 1
            while j < n:
                if code[j] == "\\":
                    j += 2
                    continue
                if code[j] == quote:
                    j += 1
                    break
                j += 1
            i = j
            continue
        if c == "/" and i + 1 < n and code[i + 1] == "*":
            end = code.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if c == "/" and i + 1 < n and code[i + 1] == "/":
            end = code.find("\n", i + 2)
            i = n if end == -1 else end
            continue
        if c == "{":
            if depth == 0:
                # look backwards skipping whitespace + ACSL comments for ')'
                k = i - 1
                while k >= 0 and code[k] in " \t\r\n":
                    k -= 1
                if k >= 0 and code[k] == ")":
                    fns += 1
            depth += 1
            i += 1
            continue
        if c == "}":
            depth = max(0, depth - 1)
            i += 1
            continue
        i += 1
    return fns


# ----------------------------------------------------------------------------
# Cheap pre-filter for applies()
# ----------------------------------------------------------------------------

def _added_c_files(diff_text: str) -> Dict[str, str]:
    return added_files_by_ext(diff_text, C_EXTS)


def _has_acsl_marker(code: str) -> bool:
    """True iff the added text contains either ``/*@`` or ``//@``. This is
    a substring check, not a regex, and intentionally over-includes so
    that ``score()`` (which uses the proper scanner) makes the final call.
    """
    return ("/*@" in code) or ("//@" in code)


# ----------------------------------------------------------------------------
# Public contract
# ----------------------------------------------------------------------------

def applies(diff_text: str) -> bool:
    """True iff the diff adds C/C++ source that contains at least one ACSL
    annotation marker (``/*@`` or ``//@``)."""
    by_path = _added_c_files(diff_text)
    if not by_path:
        return False
    return any(_has_acsl_marker(body) for body in by_path.values())


def score(diff_text: str) -> Optional[float]:
    by_path = _added_c_files(diff_text)
    if not by_path:
        return None

    # Per-file scan and aggregate
    n_blocks = 0
    n_core_blocks = 0
    n_core = n_adv = n_built = 0
    n_functions = 0
    saw_any_marker = False

    for body in by_path.values():
        blocks = _scan_annotation_blocks(body)
        if blocks:
            saw_any_marker = True
        for b in blocks:
            n_blocks += 1
            c, a, bt = _classify_block(b)
            n_core += c
            n_adv += a
            n_built += bt
            if c >= 1:
                n_core_blocks += 1
        n_functions += _count_functions(body)

    if not saw_any_marker:
        # applies() over-included via cheap marker — final scan found
        # nothing. Abstain (consistent with applies()=True noise).
        return None

    if n_core_blocks == 0:
        # Markers exist but no CORE clause keyword in any of them — not
        # really ACSL (could be GCC pragma comments / Doxygen). Abstain.
        return None

    # Coverage component
    denom = max(n_functions, n_core_blocks)
    coverage = min(1.0, n_core_blocks / denom) if denom > 0 else 1.0

    # Richness component
    richness_num = n_core + n_adv + 2 * n_built
    richness = min(1.0, richness_num / (4 * max(n_blocks, 1)))

    return float(0.5 * coverage + 0.5 * richness)


__all__ = [
    "applies", "score",
    "ASPECT_ID", "ASPECT_NAME", "TIER", "TOOLS",
    "APPLIES_TO_LANGS", "CLASSIFICATION",
]
