"""a96: Formal specification and verification — narrow PARTIALLY_THIN.

The norm asks code to use *formal* specifications/contracts and *verification
or proof tools* (model checkers, SMT solvers, proof assistants, deductive
verifiers) to state and check properties precisely, complementing testing.

This is sharply narrower than a215 ("contract annotations"). a215 catches
mainstream design-by-contract patterns (asserts with messages, pydantic,
Objects.requireNonNull, joi schemas). a96 specifically asks for *formal*
machinery: Coq, Lean, Dafny, TLA+, Z3, pysmt, Alloy, F*, CBMC, Frama-C,
KeY, OpenJML — tools that produce machine-checkable proofs or bounded /
exhaustive verification results, not just runtime guards.

What we can deterministically detect in a PR diff:

  1. File extensions that belong exclusively to formal-method ecosystems:
       .dfy             Dafny
       .v               Coq/Rocq (also: Verilog — we de-dupe by inspecting
                                  contents for `Theorem`/`Lemma`/`Proof.`)
       .lean            Lean 4 / Lean
       .tla, .cfg       TLA+ (cfg only if a sibling .tla)
       .smt2, .smt      SMT-LIB
       .als             Alloy
       .fst             F*
       .why, .mlw       Why3
       .ivy             Ivy
       .uclid           UCLID5
       .key             KeY proof bundles
       .p, .tptp        TPTP
       .v.thy, .thy     Isabelle/HOL theory files

  2. Imports of formal-spec libraries in mainstream-language source:
       Python:  z3, z3-solver, pysmt, cvc5, dreal, mythril, manticore,
                hypothesis (property-based — we count, but mark separately),
                deal (contracts that can be statically verified),
                crosshair (concolic verification of Python contracts)
       Java:    org.sosy_lab.java_smt, com.microsoft.z3, openjml,
                org.cprover.cbmc, daikon, randoop
       JS/TS:   z3-solver (npm), tla-plus tools — very rare; we still try
       C/C++:   ACSL annotations (`/*@ ... */`), CBMC pragmas
                (`__CPROVER_assert`, `__CPROVER_assume`)
       Java:    JML annotations (`//@`, `/*@ requires/ensures/invariant @*/`)

  3. CI / config evidence:
       dafny, coqc, lean, leanpkg, tlc, tlapm, why3, frama-c, cbmc commands
       referenced in .yml/.yaml/.sh/Makefile diffs.

Output is the *count* of distinct formal-verification signals tanh-squashed
into [0, 1]. A PR that adds a `.dfy` file, imports z3, AND mentions `cbmc`
in CI scores higher than one that just imports z3.

This metric **abstains** (returns None) on PRs with no covered files and no
matching imports. On a typical web-app PR fixture it will abstain — that is
the correct behavior. THICK would deny that the norm is observable; the
norm IS observable, just rarely instantiated, so PARTIALLY_THIN with a
narrow `applies()` gate is the honest classification.

Tier 2 (AST / tree-sitter + path inspection + small content scans).
"""
from __future__ import annotations

import math
import re  # REGEX_OK: format_header — ACSL/JML annotation markers + SMT lib filename
from typing import Dict, List, Optional, Set, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a96"
ASPECT_NAME = "Formal specification and verification"
TIER = 2
TOOLS = ["tree-sitter-python", "tree-sitter-java"]
APPLIES_TO_LANGS = [
    "Dafny", "Coq", "Lean", "TLA+", "SMT-LIB", "Alloy", "F*", "Why3",
    "Isabelle", "Python", "Java", "C", "C++",
]
CLASSIFICATION = "PARTIALLY_THIN"

# ---------------------------------------------------------------------------
# Formal-method file extensions
# ---------------------------------------------------------------------------

# Extensions whose presence alone strongly implies a formal tool is in use.
FORMAL_EXTS: Dict[str, str] = {
    ".dfy": "dafny",
    ".lean": "lean",
    ".tla": "tla+",
    ".smt2": "smt-lib",
    ".smt": "smt-lib",
    ".als": "alloy",
    ".fst": "fstar",
    ".why": "why3",
    ".mlw": "why3",
    ".ivy": "ivy",
    ".uclid": "uclid",
    ".key": "key",
    ".tptp": "tptp",
    ".thy": "isabelle",
}

# Extensions that overlap with non-formal ecosystems; inspect content to
# disambiguate.
AMBIGUOUS_EXTS: Dict[str, str] = {
    ".v": "coq_or_verilog",
    ".p": "tptp_or_pascal_or_prolog",
}

# Python modules whose import implies SMT / formal-verification use.
PY_FORMAL_IMPORTS: Set[str] = {
    "z3", "z3_solver",
    "pysmt", "pysmt.shortcuts", "pysmt.typing",
    "cvc5", "cvc5_pythonic",
    "dreal",
    "deal",          # design-by-contract that supports static verification
    "crosshair",     # symbolic execution of Python contracts
    "mythril",       # symbolic exec for EVM
    "manticore",     # symbolic exec
    "claripy",       # angr's SMT layer
    "angr",          # binary symbolic execution
    "smtlibutils",
}

# Java/JVM packages that signal SMT / model-checking / verification use.
JAVA_FORMAL_IMPORT_PREFIXES: Set[str] = {
    "org.sosy_lab.java_smt",
    "com.microsoft.z3",
    "edu.cmu.cs.openjml",
    "org.cprover",
    "daikon",
    "randoop",
    "de.uka.ilkd.key",       # KeY prover
    "org.key_project",
    "tlc2",                  # TLA+ model checker Java API
    "io.github.lambdaprime.smtlib",
}

# Shell / CI tokens that reference formal-method runners.
CI_FORMAL_TOKENS: Set[str] = {
    "dafny", "coqc", "coqtop", "coq_makefile",
    "lean", "leanpkg", "lake",
    "tlc", "tla2tools", "tlapm", "apalache",
    "why3", "frama-c", "framac",
    "cbmc", "esbmc", "kani",
    "openjml", "keymaerax", "keymaera",
    "alloy", "z3", "cvc5", "cvc4",
    "isabelle", "fstar",
}

# Files where the CI tokens above are meaningful.
CI_PATH_HINTS = (".yml", ".yaml", ".sh", "Makefile", "makefile", ".mk",
                 ".toml", ".cfg")

# ACSL: `/*@ ... @*/` or `//@ ...` in C/C++.
# JML: same syntax in Java. We require a verification keyword inside the
# annotation block to count it.
ACSL_JML_KEYWORDS = (
    "requires", "ensures", "assigns", "behavior", "complete behaviors",
    "disjoint behaviors", "loop invariant", "loop variant", "decreases",
    "invariant", "assignable", "modifies", "pure", "ghost", "model",
    "axiom", "predicate", "logic", "lemma",
)

# REGEX_OK: format_header — annotation block markers in C/Java source.
_RE_ACSL_BLOCK = re.compile(r"/\*@(.*?)@\*/", re.DOTALL)
# REGEX_OK: format_header — line-style ACSL/JML annotation prefix.
_RE_ACSL_LINE = re.compile(r"//\s*@\s*(.+)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ext(path: str) -> str:
    p = path.lower()
    if "." not in p:
        return ""
    return "." + p.rsplit(".", 1)[-1]


def _classify_v(content: str) -> Optional[str]:
    """`.v` is Coq AND Verilog. Decide based on keywords.

    Returns 'coq' if Coq markers appear, 'verilog' if Verilog markers
    appear, else None (ambiguous → don't count).
    """
    coq_markers = ("Theorem ", "Lemma ", "Proof.", "Qed.", "Inductive ",
                   "Require Import", "Definition ")
    ver_markers = ("module ", "endmodule", "wire ", "reg ", "assign ",
                   "always @", "input ", "output ")
    has_coq = any(m in content for m in coq_markers)
    has_ver = any(m in content for m in ver_markers)
    if has_coq and not has_ver:
        return "coq"
    if has_ver and not has_coq:
        return "verilog"
    return None


def _classify_p(content: str) -> Optional[str]:
    """`.p` could be TPTP (formal logic), Pascal, or Prolog. TPTP files
    typically have `fof(...)` / `cnf(...)` / `tff(...)` declarations."""
    if any(tok in content for tok in ("fof(", "cnf(", "tff(", "thf(")):
        return "tptp"
    return None


def _py_imports(code: str) -> Set[str]:
    """Extract import names from Python source via the stdlib `ast` module."""
    try:
        import ast
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return set()
    out: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name)
                out.add(a.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module)
                out.add(node.module.split(".", 1)[0])
    return out


def _java_imports(code: str) -> Set[str]:
    """Extract `import x.y.z;` statements from Java without tree-sitter to
    keep this metric tier-2 and dependency-light."""
    out: Set[str] = set()
    for line in code.splitlines():
        s = line.strip()
        if not s.startswith("import "):
            continue
        s = s[len("import "):].rstrip(";").strip()
        if s.startswith("static "):
            s = s[len("static "):].strip()
        if s.endswith(".*"):
            s = s[:-2]
        out.add(s)
    return out


def _count_acsl_jml(code: str) -> int:
    """Return the number of ACSL/JML annotation blocks that contain at least
    one verification keyword. Counts blocks, not lines."""
    n = 0
    for m in _RE_ACSL_BLOCK.finditer(code):
        body = m.group(1).lower()
        if any(k in body for k in ACSL_JML_KEYWORDS):
            n += 1
    for m in _RE_ACSL_LINE.finditer(code):
        body = m.group(1).lower()
        if any(k in body for k in ACSL_JML_KEYWORDS):
            n += 1
    # CBMC-flavored intrinsics in C: __CPROVER_assert / __CPROVER_assume
    if "__CPROVER_assert" in code or "__CPROVER_assume" in code:
        n += 1
    return n


def _count_ci_tokens(path: str, code: str) -> int:
    """Count CI/config references to formal-method tools."""
    name = path.rsplit("/", 1)[-1]
    if not (any(path.endswith(suf) for suf in CI_PATH_HINTS)
            or name in ("Makefile", "makefile")):
        return 0
    low = code.lower()
    return sum(1 for tok in CI_FORMAL_TOKENS if tok in low)


# ---------------------------------------------------------------------------
# Signal collection
# ---------------------------------------------------------------------------

def _collect_signals(diff_text: str) -> Tuple[List[str], int]:
    """Walk added files, return (list_of_signal_descriptions, total_count).

    Each signal counts once per occurrence type per file so we don't get
    swamped by a single annotation-heavy file.
    """
    by_path = parse_diff_added_by_file(diff_text)
    sigs: List[str] = []

    for path, code in by_path.items():
        ext = _ext(path)

        # 1) Unambiguous formal-method extensions
        if ext in FORMAL_EXTS:
            sigs.append(f"ext:{FORMAL_EXTS[ext]}:{path}")
            continue

        # 2) Ambiguous extensions — disambiguate by content
        if ext in AMBIGUOUS_EXTS:
            if ext == ".v":
                kind = _classify_v(code)
                if kind == "coq":
                    sigs.append(f"ext:coq:{path}")
                continue
            if ext == ".p":
                kind = _classify_p(code)
                if kind == "tptp":
                    sigs.append(f"ext:tptp:{path}")
                continue

        # 3) Python imports of formal libraries
        if ext in (".py", ".pyi"):
            imports = _py_imports(code)
            hit = imports & PY_FORMAL_IMPORTS
            for h in sorted(hit):
                sigs.append(f"py_import:{h}:{path}")

        # 4) Java imports of formal-verification packages
        elif ext == ".java":
            imports = _java_imports(code)
            for imp in imports:
                for pref in JAVA_FORMAL_IMPORT_PREFIXES:
                    if imp == pref or imp.startswith(pref + "."):
                        sigs.append(f"java_import:{pref}:{path}")
                        break
            # JML annotations in body
            n_jml = _count_acsl_jml(code)
            for _ in range(min(n_jml, 5)):  # cap per-file
                sigs.append(f"jml:{path}")

        # 5) ACSL annotations / CBMC intrinsics in C / C++ source / headers
        elif ext in (".c", ".h", ".cpp", ".cc", ".cxx", ".hpp"):
            n_acsl = _count_acsl_jml(code)
            for _ in range(min(n_acsl, 5)):
                sigs.append(f"acsl:{path}")

        # 6) CI / config / shell mentions of formal-method runners
        n_ci = _count_ci_tokens(path, code)
        for _ in range(min(n_ci, 3)):
            sigs.append(f"ci:{path}")

    return sigs, len(sigs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def applies(diff_text: str) -> bool:
    """Cheap pre-filter: only apply when the diff plausibly touches formal
    methods. We check (a) any formal-method file extension, (b) any Python
    `.py` file containing a candidate import token in its added text,
    (c) any Java/C file containing an ACSL/JML marker, (d) any CI file
    mentioning a formal-tool token.

    We avoid full AST parsing here — that happens in `score()`.
    """
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return False
    py_tokens = tuple(PY_FORMAL_IMPORTS)
    java_tokens = tuple(JAVA_FORMAL_IMPORT_PREFIXES)
    for path, code in by_path.items():
        ext = _ext(path)
        if ext in FORMAL_EXTS or ext in AMBIGUOUS_EXTS:
            return True
        if ext in (".py", ".pyi") and any(t in code for t in py_tokens):
            return True
        if ext == ".java":
            if any(t in code for t in java_tokens):
                return True
            if "/*@" in code or "//@" in code:
                return True
        if ext in (".c", ".h", ".cpp", ".cc", ".cxx", ".hpp"):
            if "/*@" in code or "//@" in code or "__CPROVER_" in code:
                return True
        # CI files: scan only paths that look like CI/config/build
        name = path.rsplit("/", 1)[-1]
        if (any(path.endswith(suf) for suf in CI_PATH_HINTS)
                or name in ("Makefile", "makefile")):
            low = code.lower()
            if any(tok in low for tok in CI_FORMAL_TOKENS):
                return True
    return False


def score(diff_text: str) -> Optional[float]:
    """Score = tanh(n_signals / 2) ∈ [0, 1].

    1 signal  → 0.46
    2 signals → 0.76
    4 signals → 0.96
    """
    sigs, n = _collect_signals(diff_text)
    if n == 0:
        # applies() said yes but every candidate failed verification
        # (e.g. ambiguous .v file turned out to be Verilog, or a Python
        # file that mentions "z3" inside a string but doesn't import it).
        return None
    return float(math.tanh(n / 2.0))


__all__ = [
    "applies", "score",
    "ASPECT_ID", "ASPECT_NAME", "TIER", "TOOLS",
    "APPLIES_TO_LANGS", "CLASSIFICATION",
]
