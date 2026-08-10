"""Emit the blind static construct-fidelity audit for code-review R3 seeds.

This is a hand-adjudicated, source-only artifact builder.  It reads the
hierarchy construct, the selected candidate identity, and nothing from an
execution or evaluation channel.  The row payloads below record the actual
decision-contributing operations found by static inspection of each selected
source module.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_MAP = (
    ROOT
    / "outputs/metric_seam_pilot/hierarchy_r123/code_review_seed_map_v2.json"
)
OUTPUT = (
    ROOT
    / "outputs/metric_seam_pilot/hierarchy_r123/"
    "code_review_construct_fidelity_R3_v1.json"
)

INTERPRETATION = (
    "A partial verdict authorizes, at most, relation-local execution; it is "
    "not whole-construct verification. A mismatch or absent candidate is "
    "bounded non-discovery within this frozen source-only candidate library, "
    "not evidence that the construct is tacit."
)


def _a(
    *,
    expected_aspect: str | None,
    requested_relation: str,
    implemented_relations: list[str],
    verdict: str,
    scope: str,
    depth: int | None,
    caveats: list[str],
    rationale: str,
) -> dict[str, Any]:
    return {
        "expected_aspect": expected_aspect,
        "requested_relation": requested_relation,
        "implemented_relations": implemented_relations,
        "verdict": verdict,
        "scope": scope,
        "eligible_for_relation_local_execution": verdict in {"exact", "partial"},
        "audited_depth": depth,
        "dependency_applicability_caveats": caveats,
        "rationale": rationale,
    }


AUDITS: dict[str, dict[str, Any]] = {
    "TB::code-review::general::R3::grandparent::11::2c562440277fde02d164": _a(
        expected_aspect=None,
        requested_relation=(
            "added implementation plus current requirements -> whether the change is "
            "the simplest explicit solution, with minimal configuration and no "
            "unnecessary magic, metaprogramming, or speculative abstraction"
        ),
        implemented_relations=[],
        verdict="no_candidate_bounded_non_discovery",
        scope="none",
        depth=None,
        caveats=[
            "Source-only retrieval abstained; no program was selected, so there is no implementation path to audit."
        ],
        rationale=(
            "No candidate crossed the frozen source-only seed gate. This records absence "
            "from the searched library, not failure of all possible code programs."
        ),
    ),
    "TB::code-review::general::R3::grandparent::14::38c3dcb57d12d9cbd880": _a(
        expected_aspect="a112",
        requested_relation=(
            "added code and deployment context -> whether purposeful telemetry and "
            "diagnostics can detect issues, explain impact across environments, and "
            "inform technical or business decisions"
        ),
        implemented_relations=[
            "parse added Python, JavaScript/TypeScript, Java, and Go code with tree-sitter",
            "count recognized logger, metric, and tracer call shapes as proper telemetry",
            "count print, console, System.out/err, and fmt print call shapes as debug anti-patterns",
            "add a bounded signal for recognized observability-library imports and ratio proper to bare calls",
        ],
        verdict="partial",
        scope="subrelation_only",
        depth=1,
        caveats=[
            "Only added fragments in five languages are parsed.",
            "Receiver-name and import catalogs miss aliases and custom telemetry wrappers.",
            "The program abstains when it finds no recognized observability activity.",
            "It does not inspect deployment configuration, structured metadata, alerting, environment coverage, impact analysis, or decision use.",
        ],
        rationale=(
            "The program directly measures one operative sub-relation: use of recognized "
            "telemetry calls instead of ad-hoc printing. It does not establish whether "
            "the telemetry is purposeful, sufficient, cross-environment, or useful for diagnosis."
        ),
    ),
    "TB::code-review::general::R3::grandparent::19::65db7483b9b1675f88f0": _a(
        expected_aspect=None,
        requested_relation=(
            "change plus existing architecture, local idioms, and system requirements -> "
            "whether the change fits the established architecture and its operating context"
        ),
        implemented_relations=[],
        verdict="no_candidate_bounded_non_discovery",
        scope="none",
        depth=None,
        caveats=[
            "Source-only retrieval abstained; no program was selected, so there is no implementation path to audit."
        ],
        rationale=(
            "No candidate crossed the frozen source-only seed gate. The missing repository "
            "architecture relation remains an unimplemented search target."
        ),
    ),
    "TB::code-review::general::R3::merged_group::1::d6f5bd2d4ef0a792711a": _a(
        expected_aspect="a15",
        requested_relation=(
            "added comments plus adjacent code and relevant context -> whether comments are "
            "minimal, clear, accurate, maintained, and explain intent, rationale, trade-offs, "
            "or assumptions instead of restating code"
        ),
        implemented_relations=[
            "parse inline comments in Python, JavaScript/TypeScript, Java, and Go while excluding doc comments",
            "detect a fixed vocabulary of rationale and intent markers in comment text",
            "relate each short comment to the next nearby non-comment code line and flag identifier-token restatement",
            "score rationale-marker rate minus adjacent-code restatement rate",
        ],
        verdict="partial",
        scope="subrelation_only",
        depth=2,
        caveats=[
            "Rationale markers are lexical and do not establish truth, clarity, or sufficiency.",
            "The restatement check only looks up to three following lines and uses identifier overlap.",
            "Only added inline comments in five languages contribute; zero-comment diffs abstain.",
            "Historical maintenance and accuracy against behavior are unavailable from the added fragment.",
        ],
        rationale=(
            "The comment-to-nearby-code comparison implements a real why-versus-what "
            "sub-relation. Accuracy, prose quality, minimality in context, and continued "
            "maintenance remain outside the program."
        ),
    ),
    "TB::code-review::general::R3::merged_group::20::55d11b5d73aa3ced65cd": _a(
        expected_aspect="a148",
        requested_relation=(
            "dependency and implementation choices plus project ecosystem -> whether the "
            "change reuses established standards, libraries, and internal components and "
            "integrates new technology safely with justified deviations"
        ),
        implemented_relations=[
            "parse added Python imports and penalize a curated set of third-party packages with standard-library analogues",
            "relate range(len(sequence)) loops to indexing of that sequence in the loop body while excluding index arithmetic",
            "detect an empty-dict initialization followed by two nearby update calls on the same binding",
            "combine import-reuse and two hand-rolled-idiom penalties across Python files",
        ],
        verdict="partial",
        scope="subrelation_only",
        depth=2,
        caveats=[
            "Python only.",
            "The replacement catalog is curated and version/context dependent.",
            "No internal-component inventory, dependency quality, integration boundary, or deviation rationale is inspected.",
            "Some explicit loops or third-party packages are intentional and can be falsely penalized.",
        ],
        rationale=(
            "The program implements the explicit prefer-standard-reuse sub-relation and two "
            "cross-node reimplementation checks. It does not measure safe integration or the "
            "fitness of project-internal reuse."
        ),
    ),
    "TB::code-review::general::R3::merged_group::17::5b66a1627e1e35eb9b9b": _a(
        expected_aspect="a191",
        requested_relation=(
            "changed public API plus prior versions, stability contracts, and client usage -> "
            "whether API style is appropriate and evolution is additive, backward compatible, "
            "versioned, and governed by safe deprecation"
        ),
        implemented_relations=[
            "parse Java, JavaScript/TypeScript, and Python class and function declarations",
            "count factory-like names, constructors with defaults, wide raw constructors, and telescoping constructor-width patterns",
            "ratio construction conveniences to wide or telescoping initialization surfaces",
        ],
        verdict="mismatch",
        scope="none",
        depth=2,
        caveats=[
            "Only added construction APIs in four languages are inspected.",
            "Factory-name recognition is heuristic and constructor appropriateness is not checked.",
            "No old API, clients, wire contract, version marker, deprecation policy, REST/RPC boundary, or compatibility evidence is read.",
        ],
        rationale=(
            "Construction ergonomics is not the requested API-evolution relation. The program "
            "never compares versions or contracts and therefore cannot contribute a decision "
            "about backward compatibility, versioning, or deprecation."
        ),
    ),
    "TB::code-review::general::R3::grandparent::22::5e63e1115f2d0f8b88f2": _a(
        expected_aspect="a410",
        requested_relation=(
            "code pattern plus its usage context -> whether established design patterns and "
            "language idioms are applied judiciously and only where they fit"
        ),
        implemented_relations=[
            "parse added C++ with tree-sitter",
            "count auto type specifiers, range-for, lambdas, structured bindings, and using aliases as modern idiom signals",
            "count typedef and C-style for statements as legacy signals and compute a modern-to-total ratio",
        ],
        verdict="partial",
        scope="subrelation_only",
        depth=1,
        caveats=[
            "C++ only and limited to a fixed C++11/14/17 surface catalog.",
            "The program abstains if none of its positive or negative nodes occur.",
            "Frequency does not establish fit, readability, semantic correctness, or design-pattern literacy.",
        ],
        rationale=(
            "Modern-versus-legacy C++ syntax is an operative language-idiom sub-relation. "
            "The program does not decide whether any idiom or higher-level design pattern is "
            "appropriate in context."
        ),
    ),
    "TB::code-review::general::R3::grandparent::6::451817f945294fa4abd8": _a(
        expected_aspect="a76",
        requested_relation=(
            "introduced error paths plus API and failure context -> whether errors are prevented, "
            "detected, reported actionably, propagated with context, and handled consistently "
            "without crashes, silence, or corruption"
        ),
        implemented_relations=[
            "parse exception/catch handlers and Go err != nil branches in five languages",
            "classify each handler body as LOG, RAISE, RETURN, SWALLOW, or OTHER using body-level tokens",
            "compare signatures across at least two handlers and return the dominant-style fraction",
        ],
        verdict="partial",
        scope="subrelation_only",
        depth=2,
        caveats=[
            "At least two recognized handlers are required.",
            "Handler actions are coarse substring classes; mixed semantically-correct styles can be penalized.",
            "Uniformly swallowed failures receive a high consistency score.",
            "Message actionability, error typing, propagation context, prevention, recovery, and corruption are not checked.",
        ],
        rationale=(
            "Cross-handler style uniformity is a genuine consistency sub-relation, but it is "
            "not robustness or quality: the program deliberately can reward consistently bad handling."
        ),
    ),
    "TB::code-review::general::R3::grandparent::3::681c2abce3bef33e3781": _a(
        expected_aspect="a131",
        requested_relation=(
            "changed tests plus intended requirements and public interfaces -> whether tests "
            "specify externally observable behavior, mirror real interactions, and support "
            "behavior-driven design and safe refactoring"
        ),
        implemented_relations=[
            "identify test paths and Gherkin feature files",
            "parse imports and calls in Python, JavaScript/TypeScript, Java, and Go tests",
            "count boundary-driver imports and BDD-shaped calls as pro signals",
            "count mocking imports and mock calls as implementation-detail anti-signals and compute a smoothed ratio",
        ],
        verdict="partial",
        scope="subrelation_only",
        depth=1,
        caveats=[
            "Test-path and import catalogs determine applicability and style.",
            "Boundary tools can still assert internals, and unit tests can exercise public behavior.",
            "No requirement-to-test, action-to-assertion, or behavior-to-interface correspondence is checked.",
            "Tests are not executed.",
        ],
        rationale=(
            "The program implements a structural boundary-testing-style sub-relation. It does "
            "not establish that tests express the requested behavior or mirror real interactions."
        ),
    ),
    "TB::code-review::general::R3::merged_group::9::c500175a8ff2b991a5d4": _a(
        expected_aspect="a410",
        requested_relation=(
            "added code plus project and community conventions -> whether local and language "
            "idioms are followed consistently and any deviations are rare, reasoned, and clearer"
        ),
        implemented_relations=[
            "parse added C++ with tree-sitter",
            "count a fixed set of modern C++ constructs and a fixed set of legacy constructs",
            "return the modern-idiom share of recognized constructs",
        ],
        verdict="partial",
        scope="subrelation_only",
        depth=1,
        caveats=[
            "C++ only.",
            "No project-local baseline or style configuration is read.",
            "The fixed modern/legacy polarity may not match the project's language standard or context.",
            "Consistency and reasons for deviation are not evaluated.",
        ],
        rationale=(
            "Use of common modern C++ idioms is a narrow community-idiom sub-relation. It "
            "does not measure adherence to local conventions or justified exceptions."
        ),
    ),
    "TB::code-review::general::R3::merged_group::25::af86a763745e1d7c2f83": _a(
        expected_aspect="a47",
        requested_relation=(
            "code, configuration, and architecture -> whether authentication/authorization, "
            "secret and key handling, encryption, DoS resilience, sensitive-data protection, "
            "security logging, defense in depth, and safe defaults are robust and standards-aligned"
        ),
        implemented_relations=[
            "run Bandit static analysis over added Python source",
            "retain a curated set of hardening findings for secrets, weak crypto, insecure defaults, risky deserialization, TLS, permissions, and dynamic execution",
            "weight retained findings by Bandit severity and normalize by added Python lines",
        ],
        verdict="partial",
        scope="subrelation_only",
        depth=1,
        caveats=[
            "Python only and requires the Bandit executable.",
            "Only the curated Bandit finding IDs contribute; injection findings are deliberately excluded.",
            "No architecture, identity policy, authorization model, DoS behavior, security-log coverage, data flow, or standards profile is inspected.",
            "Absence of findings is not proof of a secure system.",
        ],
        rationale=(
            "Bandit's selected unsafe-construct and insecure-default rules implement a real "
            "hardening-hygiene sub-relation. They cover only a small code-local slice of the "
            "multi-control construct."
        ),
    ),
    "TB::code-review::general::R3::merged_group::31::4b956308225f91ac3a64": _a(
        expected_aspect="a76",
        requested_relation=(
            "introduced error paths plus surrounding API context -> whether the happy path is "
            "clear, failures are explicitly and consistently propagated or mapped, messages are "
            "actionable, and silent failures or inappropriate panics are avoided"
        ),
        implemented_relations=[
            "parse handlers in Python, JavaScript/TypeScript, Java, and Go",
            "map each handler body to LOG, RAISE, RETURN, SWALLOW, or OTHER",
            "compare at least two handler signatures and score modal uniformity",
        ],
        verdict="partial",
        scope="subrelation_only",
        depth=2,
        caveats=[
            "At least two recognized handlers are required.",
            "Body classification is token-based and loses control-flow and error-type semantics.",
            "A set of uniformly swallowing handlers scores as consistent.",
            "Happy-path visibility, message actionability, propagation mapping, and panic appropriateness are not checked.",
        ],
        rationale=(
            "The program measures only consistency of coarse handler shape. That relation is "
            "operative but cannot stand in for robust handling quality or appropriate propagation."
        ),
    ),
    "TB::code-review::general::R3::grandparent::12::090af4e25717489168bb": _a(
        expected_aspect=None,
        requested_relation=(
            "changed implementation plus workload evidence, profiles, and resource constraints -> "
            "whether algorithm and implementation choices are efficient and time/space/latency "
            "trade-offs are empirically justified without premature optimization"
        ),
        implemented_relations=[],
        verdict="no_candidate_bounded_non_discovery",
        scope="none",
        depth=None,
        caveats=[
            "Source-only retrieval abstained; no program was selected, so there is no implementation path to audit."
        ],
        rationale=(
            "No candidate crossed the frozen source-only seed gate. In particular, no profiler, "
            "complexity-to-workload, or benchmark-evidence relation was selected."
        ),
    ),
    "TB::code-review::general::R3::grandparent::20::b295cf2b241026f56f13": _a(
        expected_aspect="a20",
        requested_relation=(
            "dependency manifests, code choices, repository state, and toolchain constraints -> "
            "whether dependencies are minimized and managed, proven components are reused, "
            "integration is safe, and designs remain portable"
        ),
        implemented_relations=[
            "parse added dependency entries in Python, JavaScript, Rust, and Go manifest formats",
            "count direct additions and classify version pinning with ecosystem-specific rules",
            "relate changed manifests to corresponding lockfile paths in the same diff",
            "combine direct-dependency minimization, pinning fraction, and lockfile co-change",
        ],
        verdict="partial",
        scope="subrelation_only",
        depth=2,
        caveats=[
            "Only recognized manifest additions apply.",
            "Repository state outside the diff is unavailable, so lockfile absence is only softly penalized.",
            "No dependency resolution, transitive graph, vulnerability, size, provenance, necessity, or internal-component reuse is evaluated.",
            "Portability and integration boundaries are not inspected.",
        ],
        rationale=(
            "Direct-addition count, pinning, and manifest-to-lockfile synchronization are real "
            "dependency-management sub-relations. Component fitness, safe integration, and "
            "portability remain unimplemented."
        ),
    ),
    "TB::code-review::general::R3::grandparent::15::7868422a985d046dcdc2": _a(
        expected_aspect="a47",
        requested_relation=(
            "code and design artifacts plus threats and trust boundaries -> whether security is "
            "built into the design through threat modeling, safer languages or constructs, and "
            "integration with review and technical decisions"
        ),
        implemented_relations=[
            "run Bandit on added Python source",
            "filter findings to a curated unsafe-construct and insecure-default hardening set",
            "compute a severity-weighted finding density",
        ],
        verdict="partial",
        scope="subrelation_only",
        depth=1,
        caveats=[
            "Python only and requires Bandit.",
            "No threat model, trust boundary, architectural decision, language-choice comparison, or review record is read.",
            "The retained rules identify code-local unsafe constructs, not security-by-design process or completeness.",
        ],
        rationale=(
            "Avoidance of known unsafe constructs is one explicitly requested safer-construct "
            "sub-relation. The candidate provides no evidence about threat modeling or security "
            "being integrated into architecture and review."
        ),
    ),
    "TB::code-review::general::R3::merged_group::0::7e7105b23885efd5e5ff": _a(
        expected_aspect="a43",
        requested_relation=(
            "declared identifiers plus lexical scope, API use, and project/language conventions -> "
            "whether names reveal intent, balance brevity and clarity, follow conventions, and "
            "avoid ambiguity or obscure abbreviations"
        ),
        implemented_relations=[
            "parse declarations and loop binders in Python, JavaScript/TypeScript, Java, and Go",
            "exempt conventional loop binders and language-standard special names",
            "flag non-loop one- or two-character names, a curated placeholder set, and bare numeric suffixes",
            "score the fraction of at least three declarations not matching those obvious-failure shapes",
        ],
        verdict="partial",
        scope="subrelation_only",
        depth=1,
        caveats=[
            "Five languages and added fragments only.",
            "At least three eligible declarations are required.",
            "The placeholder and abbreviation catalogs are fixed and context insensitive.",
            "Case/style convention, call-site readability, semantic intent, ambiguity, and project-local usage are not evaluated.",
        ],
        rationale=(
            "The program directly catches a narrow class of structurally obvious low-information "
            "names. It cannot establish intention revelation or convention fit for names that "
            "survive the catalog."
        ),
    ),
    "TB::code-review::general::R3::merged_group::14::f57a574fa387748b99c2": _a(
        expected_aspect="a130",
        requested_relation=(
            "changed ADR/RFC documents -> whether context, goals and non-goals, alternatives and "
            "rejections, trade-offs, consequences, and rationale are substantively captured and "
            "organized with lightweight discoverability and governance"
        ),
        implemented_relations=[
            "identify ADR/RFC files from path conventions",
            "parse Markdown heading structure with tree-sitter",
            "check the document-wide heading set for context, decision, consequences, status, alternatives, trade-offs, follow-up, and stakeholder groups",
            "average required- and bonus-section presence fractions",
        ],
        verdict="partial",
        scope="subrelation_only",
        depth=2,
        caveats=[
            "Only recognized ADR/RFC paths and mostly Markdown-like files apply.",
            "Heading keyword presence does not establish that a section is non-empty, accurate, reasoned, or complete.",
            "Goals/non-goals, rejected-alternative substance, discoverability across the repository, and governance behavior are not verified.",
        ],
        rationale=(
            "Cross-section template coverage is an operative completeness sub-relation. The "
            "program checks headings, not the substantive quality or governance requested by "
            "the whole construct."
        ),
    ),
    "TB::code-review::general::R3::merged_group::21::bdce02e573b7f97faf5d": _a(
        expected_aspect="a92",
        requested_relation=(
            "source tree and declarations plus client boundaries -> whether code is organized "
            "into cohesive, discoverable packages/modules with appropriate visibility and "
            "maintainable, evolvable APIs"
        ),
        implemented_relations=[
            "parse top-level declarations, exports, access modifiers, and naming-based visibility in five languages",
            "penalize high per-file type or export counts using fixed thresholds",
            "score visibility-discipline proxies such as Python __all__, Java modifiers, Go capitalization mix, and TypeScript access modifiers",
            "average per-file structural organization scores",
        ],
        verdict="partial",
        scope="subrelation_only",
        depth=1,
        caveats=[
            "Only added fragments in Python, JavaScript/TypeScript, Java, and Go are parsed.",
            "Thresholds for classes and exports per file are calibrated preferences.",
            "The repository package tree, imports between modules, client use, cohesion, discoverability, history, and API evolution are not read.",
        ],
        rationale=(
            "Per-file declaration breadth and explicit visibility implement a structural "
            "organization sub-relation. They do not establish cohesive package boundaries or "
            "evolvability across the repository."
        ),
    ),
    "TB::code-review::general::R3::grandparent::8::ca1ea174680697aba810": _a(
        expected_aspect=None,
        requested_relation=(
            "public API plus documentation and representative client use -> whether the API is "
            "ergonomic, minimal, safe, and documented well enough for correct use without reading implementation"
        ),
        implemented_relations=[],
        verdict="no_candidate_bounded_non_discovery",
        scope="none",
        depth=None,
        caveats=[
            "Source-only retrieval abstained; no program was selected, so there is no implementation path to audit."
        ],
        rationale=(
            "No candidate crossed the frozen source-only seed gate. No client-usage-to-API or "
            "documentation-to-correct-use relation was selected."
        ),
    ),
    "TB::code-review::general::R3::grandparent::16::9a8efffaad0ee10facbb": _a(
        expected_aspect="a280",
        requested_relation=(
            "code, configuration, and sensitive-data flows -> whether defense-in-depth controls "
            "and safe defaults protect data through minimization, least privilege, and careful handling and logging"
        ),
        implemented_relations=[
            "gate on session, cookie, or JWT tokens in added code across five languages",
            "parse cookie setter options and cookie object fields for HttpOnly, Secure, and SameSite signals",
            "parse selected JWT decode, verify, and encode argument shapes for disabled verification, none algorithms, and expiration signals",
            "ratio recognized secure to insecure session call sites",
        ],
        verdict="partial",
        scope="subrelation_only",
        depth=1,
        caveats=[
            "Only recognized session/cookie/JWT call shapes in five languages apply.",
            "Aliases, wrappers, dynamic options, and configuration outside added code can evade the detector.",
            "No data inventory, minimization, least-privilege policy, authorization, logging content, runtime binding, or full data flow is inspected.",
        ],
        rationale=(
            "Secure session-cookie and JWT options are genuine safe-default and data-protection "
            "sub-relations. They cover neither the breadth of controls nor sensitive-data "
            "governance in the requested construct."
        ),
    ),
    "TB::code-review::general::R3::grandparent::7::5d7903455895f9dde690": _a(
        expected_aspect=None,
        requested_relation=(
            "external HTTP API schemas, routes, domain model, and internal representations -> "
            "whether resources model the domain cleanly without leaking internal schemas or boundaries"
        ),
        implemented_relations=[],
        verdict="no_candidate_bounded_non_discovery",
        scope="none",
        depth=None,
        caveats=[
            "Source-only retrieval abstained; no program was selected, so there is no implementation path to audit."
        ],
        rationale=(
            "No candidate crossed the frozen source-only seed gate. No external-to-internal "
            "schema or resource-boundary relation was selected."
        ),
    ),
    "TB::code-review::general::R3::merged_group::30::c5836627124ef9cf4b51": _a(
        expected_aspect="a1",
        requested_relation=(
            "current proven requirements plus added implementation -> whether the change is the "
            "simplest adequate design, without incidental complexity, speculative generality, "
            "premature optimization, or hard-to-explain structure"
        ),
        implemented_relations=[
            "parse added Python ASTs and count ABC machinery, placeholders, selected single-method abstraction names, and multiple inheritance",
            "scan extracted added-line comment substrings across supported source files for TODO, FIXME, XXX, and HACK markers",
            "convert structural-flag and comment-marker densities into a weighted simplicity proxy",
        ],
        verdict="partial",
        scope="subrelation_only",
        depth=1,
        caveats=[
            "Only Python receives AST-based abstraction checks; other languages contribute comment markers only.",
            "Partial diff fragments can fail Python parsing and then lose structural flags.",
            "Abstract classes, named patterns, placeholders, and TODOs can be justified by real requirements.",
            "No requirement, usage, alternative implementation, performance evidence, or explanation is compared to the code.",
        ],
        rationale=(
            "The program detects several explicit speculative-generality shapes, which is a "
            "real but noisy sub-relation. It cannot decide whether those shapes are necessary "
            "or whether the implementation is simplest for the proven requirements."
        ),
    ),
    "TB::code-review::general::R3::merged_group::28::66cb8ac9cbc2fd756a34": _a(
        expected_aspect="a0",
        requested_relation=(
            "changed function control flow -> whether the flow is easy to analyze, with low "
            "nesting and decision burden, useful guard clauses or early exits, and no long branch chains"
        ),
        implemented_relations=[
            "extract added supported-language source fragments and analyze them with lizard",
            "read per-function cyclomatic complexity from lizard output",
            "select the maximum added-function CCN and apply an exponential low-complexity score",
        ],
        verdict="partial",
        scope="subrelation_only",
        depth=2,
        caveats=[
            "Requires the lizard executable and at least one parseable function in a supported language.",
            "Only added fragments are analyzed, which can distort enclosing-function structure.",
            "Cyclomatic complexity counts decisions but does not distinguish nesting, cognitive penalties, guard clauses, early exits, or idiomatic clarity.",
        ],
        rationale=(
            "Per-function control-flow complexity is a direct analyzability sub-relation. It "
            "does not cover the named control-flow shapes or cognitive-complexity semantics of "
            "the whole construct."
        ),
    ),
    "TB::code-review::general::R3::merged_group::19::40c516351670d980a895": _a(
        expected_aspect="a3",
        requested_relation=(
            "classes and interfaces plus client usage -> whether interfaces are small and "
            "client-focused, encapsulate implementation details, decouple abstraction from "
            "implementation, and expose clear stable contracts"
        ),
        implemented_relations=[
            "parse class and interface members in Python, Java, and JavaScript/TypeScript",
            "count public and non-public methods and maximum public-method arity per class",
            "penalize public-method count and arity and reward a smoothed private-member fraction",
            "average structural surface scores across classes and files",
        ],
        verdict="partial",
        scope="subrelation_only",
        depth=1,
        caveats=[
            "Only classes/interfaces in four language families contribute.",
            "Visibility conventions and fixed method/arity thresholds are structural proxies.",
            "Client focus, method cohesion, state leakage, contract clarity, decoupling, and historical stability are not inspected.",
        ],
        rationale=(
            "Public-surface size, arity, and hidden-helper ratio implement a minimal-interface "
            "sub-relation. They cannot establish cohesion, client focus, information hiding in "
            "behavior, or contract stability."
        ),
    ),
    "TB::code-review::general::R3::grandparent::0::9c815cf7273820f4f7f1": _a(
        expected_aspect="a37",
        requested_relation=(
            "before/after code, smell evidence, and behavioral checks -> whether appropriate "
            "cataloged refactorings safely remediate smells while preserving behavior and avoiding cosmetic churn"
        ),
        implemented_relations=[
            "parse source-file diff hunks into added and removed lines",
            "compare whole-diff add/remove balance and the fraction of files with removals",
            "relate whitespace-normalized added and removed line multisets within a file to identify cosmetic-only churn",
            "combine balance, existing-file editing, and anti-cosmetic components",
        ],
        verdict="partial",
        scope="subrelation_only",
        depth=2,
        caveats=[
            "Requires recognizable source diffs with both additions and removals.",
            "Balanced rewrites can change behavior, and legitimate refactorings need not balance line counts.",
            "Only whitespace-equivalent churn is recognized as cosmetic.",
            "No tests are executed; no code smell, named refactoring, semantic equivalence, intent, or risk isolation is checked.",
        ],
        rationale=(
            "Added-to-removed structure and exact whitespace-churn detection implement a narrow "
            "refactoring-shape sub-relation. They do not verify smell remediation, technique "
            "choice, safety, or behavior preservation."
        ),
    ),
    "TB::code-review::general::R3::grandparent::18::baf83bcbdb28b38c6317": _a(
        expected_aspect=None,
        requested_relation=(
            "changed UI plus design-system tokens, component usage, rendered output, and brand "
            "guidance -> whether interfaces are visually coherent, usable, and brand aligned"
        ),
        implemented_relations=[],
        verdict="no_candidate_bounded_non_discovery",
        scope="none",
        depth=None,
        caveats=[
            "Source-only retrieval abstained; no program was selected, so there is no implementation path to audit."
        ],
        rationale=(
            "No candidate crossed the frozen source-only seed gate. No design-token, component, "
            "rendered-layout, or visual-coherence relation was selected."
        ),
    ),
    "TB::code-review::general::R3::grandparent::21::3420365609caef197413": _a(
        expected_aspect="a184",
        requested_relation=(
            "changed code plus specifications, contracts, tests, and proof or verification "
            "artifacts -> whether functional correctness is assured across edge cases"
        ),
        implemented_relations=[
            "extract static regex literals from five languages with tree-sitter",
            "compile extracted patterns with Python's regex engine as a syntax proxy",
            "walk the regex parse tree to flag nested unbounded quantifiers and average literal-level scores",
        ],
        verdict="mismatch",
        scope="none",
        depth=1,
        caveats=[
            "Only statically extractable regex literals in five languages apply.",
            "Python regex compatibility is only a proxy for JavaScript, Java, and Go flavors.",
            "Dynamic patterns and semantic behavior on inputs are not checked.",
            "No specifications, contracts, tests, proof artifacts, edge-case coverage, or whole-program behavior are inspected.",
        ],
        rationale=(
            "Regex literal well-formedness is a component-specific code check, not evidence of "
            "the requested correctness-assurance strategy. The candidate never relates behavior "
            "to specifications, tests, contracts, or edge cases."
        ),
    ),
    "TB::code-review::general::R3::merged_group::3::2e8ab4cdcc604febf558": _a(
        expected_aspect="a88",
        requested_relation=(
            "changed test suite plus runtime characteristics -> whether the suite has many fast "
            "unit tests, fewer integration/service tests, minimal end-to-end tests, and remains "
            "focused, quick, and reliable"
        ),
        implemented_relations=[
            "identify supported-language test files from path conventions",
            "parse imports and use path/import precedence to classify each file as unit, integration, end-to-end, or unknown",
            "compare classified files across the diff and return the unit-file fraction",
        ],
        verdict="partial",
        scope="subrelation_only",
        depth=2,
        caveats=[
            "Only changed test files and a fixed path/import catalog are observed.",
            "Unknown files are excluded from the denominator.",
            "Path and imports can misclassify actual test isolation level.",
            "Tests are not run, so speed, focus, flakiness, reliability, and the repository-wide mix are unmeasured.",
        ],
        rationale=(
            "The cross-file unit/integration/end-to-end mix is an operative pyramid sub-relation. "
            "It does not verify the target proportions, runtime speed, reliability, or full-suite composition."
        ),
    ),
    "TB::code-review::general::R3::merged_group::4::fd97c9079d16266eb063": _a(
        expected_aspect="a38",
        requested_relation=(
            "requirements, changed tests, coverage, infrastructure, and execution evidence -> "
            "whether testing is deliberate and rigorous, uses an appropriate level mix and "
            "methodology, is readable and reliable, and actually runs"
        ),
        implemented_relations=[
            "identify tests from paths and feature files and parse imports in five languages",
            "classify each changed test file as unit, integration, or end-to-end using path and import signals",
            "compare the cross-file distribution with a fixed 0.7/0.2/0.1 target",
            "blend distribution distance with a changed-layer diversity score",
        ],
        verdict="partial",
        scope="subrelation_only",
        depth=2,
        caveats=[
            "Only changed test files contribute, not the existing suite.",
            "The 0.7/0.2/0.1 target and diversity weights are author-chosen.",
            "Paths and imports are imperfect proxies for actual layer and isolation.",
            "No requirements trace, assertion quality, readability, coverage, infrastructure health, flakiness, or execution is checked.",
        ],
        rationale=(
            "Changed-test layer distribution implements the testing-mix sub-relation. The "
            "candidate does not implement the broader rigor, adequacy, coverage, reliability, "
            "methodology fitness, or actually-running relations."
        ),
    ),
    "TB::code-review::general::R3::merged_group::13::b71a08674e6794d89a82": _a(
        expected_aspect="a52",
        requested_relation=(
            "added documentation or UI prose plus audience and locale context -> whether voice "
            "and tone are clear, plain, globally appropriate, active or imperative where needed, "
            "and free of ambiguity or confusing phrasing"
        ),
        implemented_relations=[
            "identify project-documentation paths while excluding ADRs and test fixtures",
            "parse Markdown or inspect reStructuredText structure for headings, lists, links, and code examples",
            "count meaningful added lines and relate documentation changes to source-code co-change",
            "average size, structure, concreteness, and co-change signals",
        ],
        verdict="mismatch",
        scope="none",
        depth=2,
        caveats=[
            "Only recognized project-documentation paths apply.",
            "The structure parser does not analyze sentence syntax, vocabulary, voice, tone, audience, or locale.",
            "Documentation/source co-change does not prove freshness or accuracy.",
            "UI strings and unchanged surrounding prose are not evaluated.",
        ],
        rationale=(
            "Documentation structure and co-change are not the requested prose-style relation. "
            "The program contains no decision-contributing operation for plain language, voice, "
            "tone, ambiguity, active voice, imperative mood, or global appropriateness."
        ),
    ),
}


def build() -> dict[str, Any]:
    source = json.loads(SOURCE_MAP.read_text())
    source_rows = [row for row in source["rows"] if row["level"] == "R3"]
    source_ids = [row["cell_id"] for row in source_rows]
    if len(source_ids) != 30 or len(set(source_ids)) != 30:
        raise ValueError("source map must contain exactly 30 unique R3 rows")
    if set(source_ids) != set(AUDITS):
        missing = sorted(set(source_ids) - set(AUDITS))
        extra = sorted(set(AUDITS) - set(source_ids))
        raise ValueError(f"audit/source row mismatch; missing={missing}, extra={extra}")

    rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        audit = dict(AUDITS[source_row["cell_id"]])
        expected = audit.pop("expected_aspect")
        selected = source_row.get("selected_seed")
        actual = selected.get("aspect_id") if selected else None
        if actual != expected:
            raise ValueError(
                f"candidate changed for {source_row['cell_id']}: expected {expected}, got {actual}"
            )
        candidate = None
        if selected is not None:
            candidate = {
                "aspect_id": selected["aspect_id"],
                "source_path": selected["source_path"],
            }
        row = {
            "cell_id": source_row["cell_id"],
            "level": source_row["level"],
            "metric_name": source_row["metric_name"],
            "candidate": candidate,
            **audit,
            "interpretation": INTERPRETATION,
        }
        rows.append(row)

    allowed_verdicts = {
        "exact",
        "partial",
        "mismatch",
        "no_candidate_bounded_non_discovery",
    }
    allowed_scopes = {"whole_construct", "subrelation_only", "none"}
    for row in rows:
        if row["verdict"] not in allowed_verdicts:
            raise ValueError(f"invalid verdict: {row['verdict']}")
        if row["scope"] not in allowed_scopes:
            raise ValueError(f"invalid scope: {row['scope']}")
        if row["eligible_for_relation_local_execution"] != (
            row["verdict"] in {"exact", "partial"}
        ):
            raise ValueError(f"eligibility/verdict mismatch: {row['cell_id']}")
        if row["candidate"] is None:
            if row["audited_depth"] is not None or row["implemented_relations"]:
                raise ValueError(f"candidate-free row has implementation claims: {row['cell_id']}")
        elif row["audited_depth"] not in {0, 1, 2, 3, 4}:
            raise ValueError(f"candidate row lacks valid audited depth: {row['cell_id']}")
        if row["verdict"] == "partial" and row["scope"] != "subrelation_only":
            raise ValueError(f"partial row overstates scope: {row['cell_id']}")
        if row["verdict"] in {"mismatch", "no_candidate_bounded_non_discovery"} and row[
            "scope"
        ] != "none":
            raise ValueError(f"failed row overstates scope: {row['cell_id']}")

    verdict_counts = Counter(row["verdict"] for row in rows)
    scope_counts = Counter(row["scope"] for row in rows)
    depth_counts = Counter(
        "null" if row["audited_depth"] is None else str(row["audited_depth"])
        for row in rows
    )
    counts = {
        "n_rows": len(rows),
        "n_candidates": sum(row["candidate"] is not None for row in rows),
        "verdicts": {key: verdict_counts.get(key, 0) for key in sorted(allowed_verdicts)},
        "scopes": {key: scope_counts.get(key, 0) for key in sorted(allowed_scopes)},
        "eligible_for_relation_local_execution": sum(
            row["eligible_for_relation_local_execution"] for row in rows
        ),
        "audited_depth": {
            key: depth_counts.get(key, 0) for key in ("0", "1", "2", "3", "4", "null")
        },
    }

    return {
        "schema": "metric-seam.code-review-construct-fidelity-r3.v1",
        "design_scope": "blind_static_construct_fidelity",
        "audit_date": "2026-07-13",
        "task": "code-review",
        "level": "R3",
        "source_map": str(SOURCE_MAP.relative_to(ROOT)),
        "forbidden_inputs": [
            "program execution",
            "reference judgments",
            "outcome labels",
            "heldout identifiers or outputs",
            "candidate score vectors",
            "correlations",
            "reconstruction or isomorphism results",
            "external models or APIs",
        ],
        "depth_vocabulary": {
            "0": "surface lexical matching",
            "1": "parsed document or code structure",
            "2": "cross-span, cross-file, or cross-section relation",
            "3": "formal solver or evidence-graph execution",
            "4": "environment or test execution",
        },
        "depth_policy": (
            "audited_depth is the deepest decision-contributing operation found in the "
            "candidate source path; it is independent of the library's declared TIER"
        ),
        "verdict_policy": {
            "exact": "the operative requested relation is implemented for the whole construct",
            "partial": "at least one requested sub-relation has a real decision-contributing implementation, while material requested relations remain absent",
            "mismatch": "the candidate computes a different relation despite topical or naming overlap",
            "no_candidate_bounded_non_discovery": "the frozen source-only retrieval selected no candidate",
        },
        "global_interpretation": INTERPRETATION,
        "counts": counts,
        "rows": rows,
    }


def main() -> None:
    payload = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
