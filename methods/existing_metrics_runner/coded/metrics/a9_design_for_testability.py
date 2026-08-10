"""a9: Design for testability — THICK.

The norm asks whether the diff structures code so that future tests will be
easy and reliable to write: small decoupled components, clear seams, explicit
state, dependency injection, and refactoring that preserves these properties.
That is an *architectural* property of the relationship between the added code
and the rest of the system, not a local lexical property of the diff hunk.

Concretely, every defensible signal turns out to require information we do
not have:

1. "Dependency injection" — recognizing DI requires knowing whether a
   constructor argument *replaces* an otherwise-hardcoded collaborator. A
   function `def __init__(self, db)` is DI only if it used to be (or would
   otherwise be) `self.db = SomeGlobalDb()`. Without the project's other
   files, we cannot tell DI from any other parameter.

2. "Avoiding singletons" — singleton detection requires recognizing the
   pattern across a module, often with module-level state and a classmethod
   gate. Even with tree-sitter we can spot module-level `_instance = None`
   plus a guarded `cls._instance = cls(...)`, but real codebases hide the
   pattern behind decorators, metaclasses, or DI containers we don't see.

3. "Pure functions" — purity is an interprocedural property. A function is
   impure if *anything it calls* performs IO or mutates state. Determining
   that from a diff requires the call graph of the entire repository. Local
   absence of `open(`/`requests.`/`time.` is necessary but very far from
   sufficient.

4. "Clear seams" — a "seam" (Feathers 2004) is a place where behaviour can
   be replaced without editing in place. Identifying a seam requires
   understanding the alternative compositions the architecture admits.

5. "No time/network dependencies in core logic" — distinguishing "core logic"
   from "boundary code" is itself the architectural judgment under test.
   A `time.time()` call inside a logging wrapper is fine; the same call
   inside business logic isn't. We cannot tell them apart from the diff.

Surrogates we considered and rejected as too weak:

  - Count of `global` statements / module-level mutable state in added code.
    Correlates loosely with "hard-to-test" code, but real projects rarely
    add literal `global` declarations; the architectural sins live in
    classmethod registries, app-context singletons, and import-time side
    effects we cannot detect locally.

  - Ratio of pure-looking functions (no `open`/IO/`time`/`random` calls) to
    total added functions. This is what a131-style structural analysis
    *could* compute, but the ratio is dominated by trivial helper functions
    and would invert the signal for any diff that legitimately needs IO.

  - Count of added test files relative to added source files. Already
    measured by a104 (test_presence); duplicating it here would conflate
    "tests exist" with "designed for testability", which the literature
    explicitly distinguishes (you can have tests without DI, and DI without
    tests).

Per the GUIDE, "a regex that returns 0.55 for everyone is worse than a
True/None pair that says 'I cannot measure this.'" Marking THICK is the
informative answer here.

External information that would unblock this norm:
  - Full repository (call graph, module structure) — would let us run
    impact analysis and detect whether added code introduces or removes
    hidden coupling.
  - Before/after commit pair — would let us measure whether the diff
    *increases or decreases* parameter count, module-level state, or static
    method count relative to baseline.
  - Project test suite + coverage delta — the most honest measure of
    "designed for testability" is whether the new code is, in fact, easy
    to test, observable through coverage and mock/stub density in the
    accompanying tests.
"""
from __future__ import annotations
from typing import Optional

ASPECT_ID = "a9"
ASPECT_NAME = "Design for testability"
TIER = 0
TOOLS = []
APPLIES_TO_LANGS = []
CLASSIFICATION = "THICK"

THICK_REASON = (
    "Design for testability is an architectural property of the relationship "
    "between added code and the rest of the system (seams, DI, purity, "
    "absence of hidden state). All of those require interprocedural / "
    "whole-repo information we do not have in a single diff. Local lexical "
    "proxies (count of `global`, presence of `time.`/`requests.` calls, "
    "ratio of test files) either don't fire at all on real diffs or invert "
    "the signal for legitimate boundary code. THICK is the honest answer."
)


def applies(diff_text: str) -> bool:
    return False


def score(diff_text: str) -> Optional[float]:
    return None
