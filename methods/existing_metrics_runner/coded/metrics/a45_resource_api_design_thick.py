"""a45: Resource-oriented API design and simplicity — THICK.

The norm asks whether the diff designs "predictable, resource-focused APIs
(e.g., REST) with shallow, sensible resource models and minimal sub-resource
depth." Satisfying it requires judgments that are simultaneously (a) about a
property of the *API surface*, not the diff hunk, and (b) about taste-laden
conventions (resource-orientation, "sensible" depth) that have no formal
definition.

Why this norm is not deterministically verifiable from a diff:

1. MOST DIFFS HAVE NO ROUTES. Empirically on the 50-PR fixture set, only
   1/50 PRs add any HTTP route / handler definition at all (the rest are
   plumbing, refactors, infra, business logic, tests, docs). An applicability
   gate that fires this rarely cannot contribute a useful column to the
   downstream feature matrix even when its score is right.

2. ROUTES ARE FRAMEWORK-IDIOSYNCRATIC. A "route" can be a Flask
   `@app.route("/x")`, a FastAPI `@router.get("/x")`, a Django `path(...)`
   tuple, an Express `app.get(...)`, a Spring `@GetMapping`, a JAX-RS
   `@GET @Path(...)`, a Go `mux.HandleFunc`, a gRPC `service Foo { rpc ... }`,
   or a Kong/Envoy/Istio config. The norm even calls out REST as merely an
   *example*; an internal RPC API can also be resource-oriented. We would
   need a tree-sitter pass for each of ~10 ecosystems, and many "APIs"
   wouldn't be visible at all (config-driven gateways, OpenAPI yamls,
   generated code).

3. THE NORM'S CRITERIA REQUIRE SEMANTIC JUDGMENTS. Even with a perfect route
   extractor, "resource-focused" demands judging whether path segments name
   nouns vs. verbs (`/users/{id}` vs. `/getUser`), whether nesting reflects
   genuine sub-resource relationships vs. accidental coupling, whether HTTP
   methods correspond to CRUD semantics, and whether the model is
   "predictable" relative to the rest of the API (which is not in the diff).
   Plural-vs-singular and verb-vs-noun on a single path segment is a
   contestable lexical heuristic that the codegen_claude diagnostic showed
   collapses into text-length signal.

4. "MINIMAL SUB-RESOURCE DEPTH" IS UNGROUNDED WITHOUT CONTEXT. A path of
   depth 4 (`/orgs/{id}/teams/{id}/members/{id}/perms`) might be the only
   sensible representation of a real domain hierarchy, while a depth-2 path
   (`/data/{blob}`) might be hiding a domain that needs structure. The norm
   asks for "sensible" depth — a meta-property of the domain model that is
   not derivable from the route shape alone.

Surrogates considered and rejected as too weak:

  - Average slash count of added route paths. Cheap, but conflates "deep
    nesting" with "expressive nesting." With only ~1/50 diffs containing any
    route at all, the column would be NaN almost everywhere and noise
    everywhere it fires.

  - Plural-vs-verb ratio on first path segment (e.g. `/users` vs. `/getUser`).
    Lexical, contestable (`/auth/login` is a verb-segment by convention and
    is not bad design), framework-dependent (Django regex routes don't
    cleanly tokenize), and on this fixture set, only one fixture would
    contribute a sample.

  - HTTP-method correspondence: GET = read, POST = create, PUT = full
    update, etc. Detectable in principle via decorator inspection, but the
    norm-satisfying answer depends on the *intent* of the endpoint, not just
    its method choice. POST-for-search and POST-for-create both validly use
    POST.

  - OpenAPI spec deltas. The fixture format only contains code diffs; spec
    files when present are not parsed as APIs by us.

A regex hunt for words like "REST", "resource", "endpoint" in PR text would,
per the codegen_claude diagnostic, collapse into the text-length proxy and
produce no signal beyond it.

Per the GUIDE: "A regex that returns 0.55 for everyone is worse than a
True/None pair that says 'I cannot measure this.'" The fixture-level
applicability rate (~1/50 by direct measurement) puts this metric in the
regime where honest THICK is strictly more informative than a Tier-1 proxy.

UNBLOCKERS — what would convert this to a deterministic measurement:
  - A repo-level pass that snapshots the whole API surface (all current
    routes), not just diff-touched files, so depth/breadth/naming can be
    judged in context.
  - A canonical resource-model spec (OpenAPI, gRPC `.proto`, GraphQL SDL)
    that the diff modifies, scored against linting rules
    (e.g. `spectral`, `protolint`, `graphql-eslint`).
  - A per-framework tree-sitter route extractor across the ~10 ecosystems
    above, plus a per-framework "idiomatic" lint ruleset.
  - A domain-model oracle that knows when nesting is "sensible."
"""
from __future__ import annotations
from typing import Optional

ASPECT_ID = "a45"
ASPECT_NAME = "Resource-oriented API design and simplicity"
TIER = 0
TOOLS = []
APPLIES_TO_LANGS = []
CLASSIFICATION = "THICK"

THICK_REASON = (
    "Resource-oriented API design requires (a) detecting route definitions "
    "across ~10 framework ecosystems, (b) judging predictability and "
    "naming conventions against the rest of the API surface (not in the "
    "diff), and (c) deciding whether sub-resource nesting reflects a "
    "'sensible' domain model — a taste-laden meta-property of the design. "
    "Empirically only 1/50 fixture PRs add any route at all, so a Tier-1 "
    "depth/naming proxy would be NaN almost everywhere and weak where it "
    "fires. Honest THICK is more informative than a near-constant surrogate."
)


def applies(diff_text: str) -> bool:
    return False


def score(diff_text: str) -> Optional[float]:
    return None
