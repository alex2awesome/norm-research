"""Blind, static construct-fidelity adjudication for math hierarchy R1/R2.

This is deliberately an authored audit, not an automatic semantic classifier.
It binds the frozen source-only candidate map to an independent reading of the
selected historical program revision and ``ops_math.py``.  Only code relations
are credited below: relations delegated through ``LLM_FIELDS`` are recorded as
excluded and cannot make a row partial or exact.

The script reads no items, references, outcomes, program outputs, or split
identifiers, and it never imports or executes a candidate program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED_MAP = Path(
    "outputs/metric_seam_pilot/hierarchy_r123/math_stackexchange_seed_map_v1.json"
)
DEFAULT_OUT = Path(
    "outputs/metric_seam_pilot/hierarchy_r123/"
    "math_stackexchange_construct_fidelity_R1_R2_v1.json"
)


def _spec(
    verdict: str,
    depth: int,
    implemented_relations: list[str],
    residual_construct: str,
    justification: str,
    caveats: list[str],
) -> dict[str, Any]:
    return {
        "verdict": verdict,
        "audited_depth": depth,
        "implemented_relations": implemented_relations,
        "residual_construct": residual_construct,
        "justification": justification,
        "caveats": caveats,
    }


# Keyed by (round, frozen panel metric name).  Every retrieved R1/R2 seed must
# have exactly one authored entry; abstentions are generated without inference.
AUDIT: dict[tuple[str, str], dict[str, Any]] = {
    (
        "R1",
        "Citation precision and reliability",
    ): _spec(
        "partial",
        2,
        [
            "Isolate the answer span and detect explicit citation commands, locator phrases such as theorem/section/page numbers, and weak 'see/cf.' placement cues.",
            "Parse numbered-equation markers and relate repeated label tokens across the answer to distinguish a defined-and-reused internal cross-reference from an unreused number.",
            "Combine locator/citation-markup evidence with internal numbered-reference breadth and reuse.",
        ],
        "Whether an external source is reliable, actually supports the claim, is attributed accurately, and is appropriate for novice readers; bibliography completeness and vague informal attribution also remain unimplemented in code.",
        "The code implements locator precision and internal cross-reference reuse, both requested precision/checkability subrelations, but it cannot establish evidentiary support or source reliability.",
        [
            "The named-source signal is an LLM field and is excluded from the credited relations.",
            "A theorem/equation number can earn locator credit even when no external source is present.",
            "Bare parenthesized integers and repeated labels remain surface proxies for genuine cross-reference function.",
        ],
    ),
    (
        "R1",
        "Design and consistency of notation",
    ): _spec(
        "partial",
        2,
        [
            "Parse LaTeX spans and compute delimiter, brace, environment, left/right, naked-command, and adjacent-inline hygiene diagnostics.",
            "Build a document-wide notation census and compare bare mathematical function names with their LaTeX-command forms to detect mixed surface conventions.",
            "Apply a bounded penalty to exceptionally long average formula token length.",
        ],
        "Semantic one-symbol/one-object consistency, conventionality for the field, initial definition of symbols, clarity, and reader friendliness remain unimplemented in code.",
        "Cross-span comparison of incompatible surface forms is a real notation-consistency subrelation, while the broader semantic design relation is not compiled.",
        [
            "The census cannot tell whether two identical symbols denote different objects or two different symbols denote the same object.",
            "Delimiter health is typesetting hygiene, not semantic notation consistency.",
            "Definition and semantic-clash judgments are LLM fields and are excluded.",
        ],
    ),
    (
        "R1",
        "Audience targeting, prerequisites, and accessibility",
    ): _spec(
        "partial",
        2,
        [
            "Isolate the answer and relate proof-skeleton definition markers to nearby math markers as a lower-bound signal of locally grounded notation.",
            "Measure case/connective scaffolding relative to sentence count and penalize math-span load unsupported by such scaffolding.",
            "Penalize malformed markup and unusually long-sentence burden.",
        ],
        "The intended audience, assumed prerequisites/background, jargon familiarity, accessibility beyond specialists, and whether explanations are pedagogically adequate remain unimplemented in code.",
        "Locally grounded definitions and explicit connective scaffolding are requested accessibility supports, but they do not identify or validate fit to an intended readership.",
        [
            "A definition word within 90 characters of any math marker need not define that symbol.",
            "The scorer cannot infer what the audience already knows.",
            "Undefined-jargon and external-pointer judgments are LLM fields and are excluded.",
        ],
    ),
    (
        "R1",
        "Concise expression without loss of clarity",
    ): _spec(
        "partial",
        2,
        [
            "Detect revision accretion, apologies, postscripts, and similar answer-side digression markers.",
            "Compare question and answer four-grams to penalize redundant restatement of supplied wording/notation.",
            "Measure parsed-math token density relative to answer prose and apply a run-on-sentence penalty.",
        ],
        "Whether retained content is necessary, whether brevity loses clarity or completeness, and whether the answer actually focuses on and resolves the question remain unimplemented in code.",
        "Accretion and redundant Q-to-A restatement are genuine economy subrelations, but clarity preservation and semantic focus are not code-checked.",
        [
            "Legitimate restatement for clarity can be penalized.",
            "High math density is not inherently concise or clear.",
            "Off-topic/focus evidence is an LLM field and is excluded.",
        ],
    ),
    (
        "R1",
        "Practical verifiability of large/computational proofs",
    ): _spec(
        "mismatch",
        2,
        [
            "Parse math spans, delimiter health, proof markers, connective positions, case markers, and surface handwave/ellipsis phrases.",
            "Aggregate those presentation signals into a generic proof-rigor score.",
        ],
        "Manageable proof size, availability and documentation of code/data, reproducibility of computational components, error sources, and independent rerunnability are all absent from the code path.",
        "Generic proof-presentation structure is not the requested reproducibility relation for large or computational proofs; the retrieval is proof/checkability topic overlap.",
        [
            "No computation is executed and no artifact, dependency, environment, or data availability is inspected.",
            "Equation counts do not operationalize manageable verification effort.",
            "Unjustified-step and derived-result judgments are LLM fields and are excluded.",
        ],
    ),
    (
        "R1",
        "Definition of a mathematical proof (deductive)",
    ): _spec(
        "mismatch",
        2,
        [
            "Detect proof/definition/connective/QED/case markers, their rough closure position, malformed LaTeX, and handwave phrases.",
            "Aggregate surface proof-presentation features without checking any inference.",
        ],
        "Whether a conclusion follows deductively from accepted assumptions, whether each inference is valid, and the criterion's definitional claim about proof definitiveness remain unimplemented in code.",
        "Proof-shaped markers do not implement the operative deductive-entailment relation; this is mention/presentation overlap rather than a deductive verifier.",
        [
            "The capability library has no CAS, proof assistant, formalization, or entailment checker.",
            "Assume/therefore/QED tokens can occur in invalid or non-deductive arguments.",
            "The substantive-result and gap judgments are LLM fields and are excluded.",
        ],
    ),
    (
        "R1",
        "Originality/novelty (results must be new)",
    ): _spec(
        "mismatch",
        1,
        [
            "Aggregate display-math volume, case/connective counts, notation diversity, and average formula-token complexity as a structural-depth proxy.",
        ],
        "Comparison with prior literature, duplicate-result detection, newness of results or definitions, and originality of the contribution remain wholly unimplemented in code.",
        "Derivation complexity and notation diversity are not novelty relations; the actual standard-versus-unusual decision is entirely prompt-side.",
        [
            "A complex answer can be routine and a short answer can be original.",
            "No corpus retrieval, citation graph, prior-art comparison, or theorem equivalence is performed.",
            "Creative-trick and technique-novelty judgments are LLM fields and are excluded.",
        ],
    ),
    (
        "R1",
        "Multimedia and visual communication effectiveness",
    ): _spec(
        "mismatch",
        1,
        [
            "Count generic example markers, numbered cases, and numeric math-span density, detect high-precision numeric tables, and penalize malformed LaTeX.",
        ],
        "Media choice, actual images/diagrams/3D/interactive content, color use, visual-narration alignment, animation, and information-load management are not parsed or judged in code.",
        "Generic example elaboration and LaTeX hygiene do not implement multimedia effectiveness; the visual-content gate is prompt-side.",
        [
            "No image, figure, caption, layout, color, or interaction representation is available to the program.",
            "Numeric density is neither visual reasoning nor media quality.",
            "Example and visual-reasoning detections are LLM fields and are excluded.",
        ],
    ),
    (
        "R1",
        "Consistency of symbol meaning and notation",
    ): _spec(
        "partial",
        2,
        [
            "Parse LaTeX spans and compute delimiter/environment hygiene.",
            "Compare bare versus command spellings of known mathematical functions across the document and penalize mixed forms.",
            "Apply a mild long-formula complexity penalty.",
        ],
        "One meaning per symbol, one symbol per object, scope-sensitive reuse, and consistency with the intended mathematical signification remain unimplemented in code.",
        "The program checks a real cross-span surface-form consistency relation, but not symbol identity or semantic meaning.",
        [
            "The notation census is occurrence-based and has no binding, scope, or object model.",
            "Raw command variants can be stylistic synonyms rather than meaning conflicts.",
            "Semantic definition/clash judgments are LLM fields and are excluded.",
        ],
    ),
    (
        "R1",
        "Conciseness and surprising compression (without loss of clarity)",
    ): _spec(
        "partial",
        2,
        [
            "Detect answer-side accretion/digression markers and run-on sentences.",
            "Compare question and answer four-grams for redundant restatement.",
            "Estimate math-token-to-prose density as a bounded presentation-economy signal.",
        ],
        "Whether the presentation achieves surprising conceptual compression, preserves clarity, selects minimal machinery, and communicates substantial content remains unimplemented in code.",
        "The code implements ordinary accretion/restatement economy, a proper subset of the construct, but cannot recognize surprising compression or clarity.",
        [
            "Question overlap may reflect helpful setup rather than redundancy.",
            "Dense notation can conceal rather than compress an argument.",
            "Semantic focus is an LLM field and is excluded.",
        ],
    ),
    (
        "R1",
        "Formal correctness and checkability",
    ): _spec(
        "partial",
        2,
        [
            "Parse delimiter/environment well-formedness and proof-skeleton categories, including whether connective/QED markers occur near the end.",
            "Penalize explicit handwave phrases and excessive ellipsis as surface evidence of omitted justification.",
            "Cap very short, math-free presentations as thin answers.",
        ],
        "Logical validity, airtight inference, truth of claims, adequacy of each justification, and mechanical proof checking remain unimplemented in code.",
        "Surface well-formedness and explicit elision burden are checkability/presentation subrelations, but they cannot establish formal correctness.",
        [
            "'Clearly' can precede a valid detailed step, while an invalid proof can avoid all handwave markers.",
            "No expression semantics, solver, theorem prover, or proof-state execution is present.",
            "Gap and substantive-result decisions are LLM fields and are excluded.",
        ],
    ),
    (
        "R1",
        "Plausible and analogical reasoning quality",
    ): _spec(
        "partial",
        2,
        [
            "Detect explicit analogy phrases, numeric-experiment tables, literal sequences, named-heuristic phrases, and case-based exploration in the answer.",
            "Require co-occurrence of guess/conjecture and check/verify language for the strongest guess-then-test signal.",
            "Relate example triggers to nearby negation to detect counterexample-style testing.",
        ],
        "Whether an analogy or induction is warranted, credible, truth-preserving, sufficiently constrained, and correctly applied remains unimplemented in code.",
        "The code reconstructs several concrete non-deductive workflow subrelations, but it measures their expressed use rather than their epistemic quality.",
        [
            "Lexical co-occurrence does not establish that verification was performed successfully.",
            "Numeric tables can report final results rather than a discovery process.",
            "The paraphrase-robust discovery judgment is an LLM field and is excluded.",
        ],
    ),
    (
        "R1",
        "Appropriate balance of words and symbols",
    ): _spec(
        "partial",
        1,
        [
            "Parse inline math and count logical/quantifier commands that occupy an entire span as symbol-for-word usage.",
            "Reward selected prose quantifier idioms and relation-chain forms, while penalizing delimiter breakage.",
            "Apply a narrow code-side penalty when congruence notation co-occurs with modular-language evidence.",
        ],
        "Whether prose or notation is clearer in context, whether dense symbol strings are understandable, and the accuracy/ambiguity of general symbol use remain unimplemented in code.",
        "Bare logical symbols used as words are a direct requested subrelation, while the broader contextual balance and clarity judgment remains unresolved.",
        [
            "The fixed bare-symbol list covers only a small subset of symbolic prose.",
            "Whole-document phrase/co-occurrence rules ignore local mathematical scope.",
            "Notation-flaw and substantive-claim judgments are LLM fields and are excluded.",
        ],
    ),
    (
        "R1",
        "Provide hierarchical, skimmable scaffolding",
    ): _spec(
        "partial",
        1,
        [
            "Count numbered display equations and formal theorem/definition/proof/QED markers as addressable structural units.",
            "Penalize run-on sentence burden and malformed delimiters as mechanical obstacles to skimming.",
        ],
        "An overall plan, semantic signposting, meaningful labeled parts, global-to-local navigation, and independently readable sections remain unimplemented in code.",
        "Numbered/formal units are a narrow structural-scaffolding witness, but the primary partitioning and upfront-plan relations are prompt-side.",
        [
            "Counts do not test whether labels are informative, ordered, or referenced.",
            "A well-organized short answer may have no formal markers; a disorganized answer may have many.",
            "Explicit partition and answer-upfront decisions are LLM fields and are excluded.",
        ],
    ),
    (
        "R1",
        "Effective, clearly labeled figures with informative captions",
    ): _spec(
        "mismatch",
        1,
        [
            "Count generic example/case markers and numeric math-span density, detect high-precision verification tables, and penalize malformed LaTeX.",
        ],
        "Figure detection, label-caption association, caption content, visual construction quality, explanation, and what the reader should notice are all unimplemented in code.",
        "The program does not read figures or captions; generic concrete-example structure is a different relation.",
        [
            "No image or document-layout channel is available.",
            "The code cannot distinguish a figure from prose or symbolic math.",
            "Visual/diagram presence is an LLM field and is excluded.",
        ],
    ),
    (
        "R1",
        "Mathematical economy and elegance",
    ): _spec(
        "partial",
        1,
        [
            "Penalize more than four parsed case markers as a bounded casework-burden signal.",
            "Penalize answers with more than ten math spans and high average formula-token length as a bounded heavy-computation signal.",
            "Apply a strong floor correction to near-empty math-free non-answers.",
        ],
        "Suitability and simplicity of the method, minimal hypotheses/apparatus, strength or generality of the result, conceptual economy, and elegance remain unimplemented in code.",
        "Casework and symbolic-load burden are requested economy subrelations, albeit crude; method appropriateness and elegance are prompt-side.",
        [
            "Long or case-rich proofs can be the most economical valid method for a problem.",
            "The code nudges are bounded and secondary to excluded LLM technique/completeness fields.",
            "No alternative-method comparison or result-strength analysis is performed.",
        ],
    ),
    (
        "R1",
        "Heuristic value and method transfer",
    ): _spec(
        "partial",
        2,
        [
            "Detect named general tricks, analogy phrases, numeric experiments, sequences, counterexample tests, and guided-discovery questions.",
            "Require guess/check language together for a stronger explicit discovery-cycle signal and parse repeated cases as exploratory structure.",
        ],
        "Whether the exposed method is plausible, explanatory, reusable beyond the instance, pedagogically transferable, or merely a symptom of a good explanation remains unimplemented in code.",
        "Explicitly naming/generalizing a trick and exposing a guess-test process are direct method-exposure subrelations, not a full transfer-quality judgment.",
        [
            "Surface phrases can announce a 'general trick' without explaining it.",
            "The code cannot test transfer to another problem or reader uptake.",
            "The semantic discovery-move extraction is an LLM field and is excluded.",
        ],
    ),
    (
        "R1",
        "Balance of intuition and rigor",
    ): _spec(
        "partial",
        1,
        [
            "Parse definition/theorem/proof/QED/case/connective markers and formal quantifier/definition language as a structural-axiomatic pole.",
            "Normalize those markers by answer length and add a small numbered-equation and delimiter-hygiene component.",
        ],
        "Intuitive or visual appeal, explanatory intuition, whether axiomatic form is appropriate, and the balance between the intuitive and rigorous poles remain unimplemented in code.",
        "The code measures one explicitly requested pole—formal/axiomatic presentation—but neither the intuition pole nor their balance.",
        [
            "Formal-language density is presentation form, not rigor or correctness.",
            "A balanced answer can be structurally sparse; a formal-looking answer can lack intuition and validity.",
            "Abstract-versus-concrete classification is an LLM field and is excluded.",
        ],
    ),
    (
        "R2",
        "Scholarship, citation, and literature grounding",
    ): _spec(
        "partial",
        2,
        [
            "Detect citation commands, theorem/section/page locators, and see/cf. placement cues in the answer.",
            "Relate numbered-equation definitions and later repeated references as an internal cross-reference discipline signal.",
        ],
        "Ethical source use, source reliability, claim-to-source support, attribution accuracy, literature coverage/currency, and bibliography completeness remain unimplemented in code.",
        "Locator precision and reference reuse are real scholarship mechanics, but they do not establish evidentiary or literature grounding.",
        [
            "Named external-source grounding is an LLM field and is excluded.",
            "Locator language can refer to the answer's own equations rather than scholarship.",
            "No external corpus or citation metadata is accessed.",
        ],
    ),
    (
        "R2",
        "Logical correctness, rigor, and completeness",
    ): _spec(
        "partial",
        2,
        [
            "Parse mathematical delimiter well-formedness and proof-skeleton presence, cases, and end-position closure markers.",
            "Penalize explicit handwave phrases and excessive ellipsis as surface evidence of omitted steps.",
            "Cap short math-free answer shapes as incomplete presentation evidence.",
        ],
        "Truth, inference validity, exhaustive/disjoint case coverage, hidden assumptions, adequacy of all justifications, and complete necessary proof details remain unimplemented in code.",
        "The code supplies a presentation/checkability lower bound for explicit gaps and malformed notation, not logical correctness or completeness itself.",
        [
            "Case markers are counted but their coverage and disjointness are never tested.",
            "Surface closure does not imply a closed proof.",
            "Unjustified-step and concrete-result judgments are LLM fields and are excluded.",
        ],
    ),
    (
        "R2",
        "Consistency of symbol meaning and notation",
    ): _spec(
        "partial",
        2,
        [
            "Parse document-wide math spans and notation occurrences.",
            "Compare bare and command spellings for known functions across the answer and penalize mixed forms.",
            "Compute delimiter/environment hygiene and a mild long-formula penalty.",
        ],
        "One symbol per object, one object per symbol, definition of nonstandard symbols, scope/status of symbols, and semantic meaning consistency remain unimplemented in code.",
        "Cross-span surface convention consistency is implemented, but the requested symbol-object mapping is not.",
        [
            "The census has no variable binding, scope, type, or referent representation.",
            "Delimiter breakage and formula length are adjacent hygiene signals rather than meaning consistency.",
            "Definition and semantic-clash judgments are LLM fields and are excluded.",
        ],
    ),
    (
        "R2",
        "Problem‑solving heuristics and exploration",
    ): _spec(
        "partial",
        2,
        [
            "Detect numeric-experiment tables, pattern sequences, analogy phrases, named general tricks, counterexample tests, and guided-discovery questions.",
            "Relate guess/conjecture language to check/verify language and parse repeated cases as exploratory evidence.",
        ],
        "Whether a plan makes genuine headway, whether an analogy or auxiliary is appropriate, whether experiments are interpreted correctly, and whether explanatory patterns are valid remain unimplemented in code.",
        "Several explicit Pólya-style exploration operations are compiled as document relations, but their success and appropriateness are not.",
        [
            "Co-occurrence and surface tables cannot distinguish exploration from retrospective exposition.",
            "The code does not evaluate the problem state before and after a move.",
            "The paraphrase-robust discovery move is an LLM field and is excluded.",
        ],
    ),
    (
        "R2",
        "Graphical communication and integrity",
    ): _spec(
        "mismatch",
        1,
        [
            "Count generic examples/cases and numeric math-span density, flag runs of high-precision decimals, and penalize malformed LaTeX.",
        ],
        "Figures, tables, plots, encodings, axes/scales, labels/captions, chartjunk, layering, separation, truthfulness, and preservation of context are unimplemented in code.",
        "The candidate never parses a graphic or its encoding, so its example-elaboration logic does not implement graphical integrity.",
        [
            "A high-precision table is merely down-weighted as non-pedagogical; its accuracy or visual integrity is not checked.",
            "No image/layout/table representation is available.",
            "Visual-content detection is an LLM field and is excluded.",
        ],
    ),
    (
        "R2",
        "Mathematical typesetting and layout quality",
    ): _spec(
        "partial",
        2,
        [
            "Parse inline and display math, LaTeX token length, line breaks, alignment environments, ampersand alignment, tags, and labels.",
            "Relate long or multiply-equal inline formulas to display choice; distinguish aligned/tagged displays from unaligned stacks and overlong flat displays.",
            "Relate repeated case markers to absence of display structure, relate numbered equations to prose references, and penalize delimiter/environment breakage.",
        ],
        "Stacked-script restraint, page width after rendering, headings, whitespace, overall page design, true legibility, and whether layout choices match semantic derivation complexity remain partly or wholly unimplemented in code.",
        "Display/inline selection, alignment structure, overlong formula shape, and referenceable layout are direct requested relations; the broader rendered-page quality remains residual.",
        [
            "Token/character thresholds approximate rendered width without fonts or a renderer.",
            "The parser can misclassify nested or malformed MathJax markup.",
            "The true multi-step derivation classification is an LLM field and is excluded.",
        ],
    ),
    (
        "R2",
        "Data and visual communication",
    ): _spec(
        "mismatch",
        1,
        [
            "Count example/case markers and numeric math-span density, detect high-precision numeric runs, and penalize malformed LaTeX.",
        ],
        "Data accuracy, ethical quantitative presentation, chart/table encoding, labels/scales, integration with prose, and visual clarity are all unimplemented in code.",
        "Generic concrete-example elaboration is not a data- or visual-communication evaluator.",
        [
            "No data values are checked against a source or computation.",
            "No table, figure, axis, scale, or caption structure is parsed.",
            "Visual/example presence is prompt-side and excluded.",
        ],
    ),
    (
        "R2",
        "Visual and diagrammatic reasoning in proofs",
    ): _spec(
        "mismatch",
        1,
        [
            "Measure generic example markers, case markers, and numeric density inside parsed math spans, plus malformed-LaTeX burden.",
        ],
        "The presence and content of diagrams/pictures, spatial mechanism, inferential role of a visual, validity of diagrammatic steps, computational analogs, and their limitations remain unimplemented in code.",
        "The code does not identify visual reasoning; that entire gate is delegated to an excluded LLM field.",
        [
            "Symbolic formulas with digits can look 'elaborated' without any visual content.",
            "No diagram or image channel is parsed.",
            "Visual-or-diagram reasoning is an LLM field and is excluded.",
        ],
    ),
    (
        "R2",
        "Abstract and introduction clarity",
    ): _spec(
        "mismatch",
        2,
        [
            "Split the StackExchange question from its answer, detect answer accretion markers, compare Q/A four-grams for restatement, measure math density, and penalize run-on sentences.",
        ],
        "Identification of an abstract or introduction, statement of main results and context, prerequisite signaling, explanation of preliminaries, and conventional welcoming front-matter phrasing are unimplemented in code.",
        "A whole-answer concision instrument is not an abstract/introduction clarity relation, especially on a corpus with no identified front-matter section.",
        [
            "The input representation is a Q/A pair rather than a manuscript with front matter.",
            "Question-answer overlap is not prerequisite or context signaling.",
            "Semantic focus is an LLM field and is excluded.",
        ],
    ),
    (
        "R2",
        "Axiomatic presentation and commitment",
    ): _spec(
        "partial",
        1,
        [
            "Parse definition, theorem, proof, QED, case, and connective markers and normalize their density by answer length.",
            "Detect formal quantifier, implication, definition, and 'let ... be' language and include numbered-equation structure.",
            "Apply a bounded malformed-LaTeX penalty.",
        ],
        "Whether stated axioms/definitions are semantically clear, whether results actually follow deductively, whether the style is abstract rather than concrete computation, and whether an axiomatic presentation is appropriate remain unimplemented in code.",
        "Formal definitional scaffolding is a direct axiomatic-presentation subrelation, while deductive validity and abstract commitment are unresolved.",
        [
            "Marker density can be high in concrete or invalid work and low in a concise axiomatic answer.",
            "No axiom dependency graph or inference checker is constructed.",
            "Abstract-versus-concrete classification is an LLM field and is excluded.",
        ],
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(seed: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(seed["source_path"])
    absolute = REPO_ROOT / source_path
    if not absolute.is_file():
        raise ValueError(f"missing selected source: {source_path}")
    return {
        "aspect_id": seed["aspect_id"],
        "source_heading": seed["source_heading"],
        "selected_revision": seed["selected_revision"],
        "source_path": str(source_path),
        "program_sha256": _sha256(absolute),
        "historical_hybrid_provenance": seed["hybrid_provenance"][
            "historical_construction"
        ],
        "llm_fields_excluded_from_implemented_relations": seed[
            "hybrid_provenance"
        ]["llm_field_names"],
    }


def build(seed_map_path: Path) -> dict[str, Any]:
    seed_map_file = (
        seed_map_path if seed_map_path.is_absolute() else REPO_ROOT / seed_map_path
    )
    seed_map = json.loads(seed_map_file.read_text())
    if seed_map.get("task") != "math-stackexchange":
        raise ValueError("seed map task is not math-stackexchange")

    source_rows = [r for r in seed_map["rows"] if r["level"] in {"R1", "R2"}]
    if len(source_rows) != 60 or len({r["cell_id"] for r in source_rows}) != 60:
        raise ValueError("expected 60 unique R1/R2 seed-map cells")

    selected_keys = {
        (r["level"], r["metric_name"])
        for r in source_rows
        if r["selected_seed"] is not None
    }
    if selected_keys != set(AUDIT):
        missing = sorted(selected_keys - set(AUDIT))
        extra = sorted(set(AUDIT) - selected_keys)
        raise ValueError(f"authored audit/candidate mismatch: missing={missing}, extra={extra}")

    rows: list[dict[str, Any]] = []
    for source in source_rows:
        key = (source["level"], source["metric_name"])
        seed = source["selected_seed"]
        requested = (
            "Given one Math StackExchange question/answer document, decide whether "
            + source["metric_description"][0].lower()
            + source["metric_description"][1:]
        )
        if seed is None:
            row = {
                "cell_id": source["cell_id"],
                "task": source["task"],
                "level": source["level"],
                "metric_name": source["metric_name"],
                "metric_description": source["metric_description"],
                "candidate": None,
                "requested_relation": requested,
                "implemented_relations": [],
                "residual_construct": source["metric_description"],
                "verdict": "no_candidate_bounded_non_discovery",
                "scope": "none",
                "eligible_for_relation_local_execution": False,
                "audited_depth": None,
                "polarity_aggregation_applicability_caveats": [
                    "No candidate survived the frozen source-only retrieval gate, so no code relation was available to inspect.",
                    "This abstention is bounded to the retrospective historical math-hybrid bank and search budget.",
                ],
                "justification": "No selected candidate exists; the full construct remains an open bounded non-discovery rather than evidence of tacitness or non-verifiability.",
                "interpretation": "No whole-construct or relation-local code-fidelity claim. Negative evidence is bounded non-discovery only.",
            }
        else:
            spec = AUDIT[key]
            partial = spec["verdict"] == "partial"
            row = {
                "cell_id": source["cell_id"],
                "task": source["task"],
                "level": source["level"],
                "metric_name": source["metric_name"],
                "metric_description": source["metric_description"],
                "candidate": _candidate(seed),
                "requested_relation": requested,
                "implemented_relations": spec["implemented_relations"],
                "residual_construct": spec["residual_construct"],
                "verdict": spec["verdict"],
                "scope": "subrelation_only" if partial else "none",
                "eligible_for_relation_local_execution": partial,
                "audited_depth": spec["audited_depth"],
                "polarity_aggregation_applicability_caveats": spec["caveats"],
                "justification": spec["justification"],
                "interpretation": (
                    "Eligible only for the named code-only relation-local subrelation; "
                    "this does not establish whole-construct verifiability, reconstruction, or isomorphism."
                    if partial
                    else "Retrieved topic overlap is rejected for this construct. The rejection is bounded non-discovery, not evidence of tacitness or non-verifiability."
                ),
            }
        rows.append(row)

    counts: dict[str, Any] = {}
    for level in ("R1", "R2", "overall"):
        subset = rows if level == "overall" else [r for r in rows if r["level"] == level]
        verdicts = Counter(r["verdict"] for r in subset)
        depths = Counter(
            "null" if r["audited_depth"] is None else str(r["audited_depth"])
            for r in subset
        )
        eligible = [r for r in subset if r["eligible_for_relation_local_execution"]]
        eligible_depths = Counter(str(r["audited_depth"]) for r in eligible)
        n_retrieved = sum(r["candidate"] is not None for r in subset)
        counts[level] = {
            "n_cells": len(subset),
            "n_retrieved_candidates": n_retrieved,
            "verdicts": dict(sorted(verdicts.items())),
            "eligible_for_relation_local_execution": len(eligible),
            "eligible_fraction_of_cells": round(len(eligible) / len(subset), 6),
            "eligible_fraction_of_retrieved_candidates": round(
                len(eligible) / n_retrieved, 6
            ),
            "audited_depths": dict(sorted(depths.items())),
            "eligible_audited_depths": dict(sorted(eligible_depths.items())),
        }

    artifact = {
        "schema": "metric-seam.math-static-construct-fidelity.v1",
        "status": "complete_static_code_only_adjudication_pre_execution",
        "design_scope": "outcome_blind_static_construct_fidelity",
        "task": "math-stackexchange",
        "levels": ["R1", "R2"],
        "source_candidate_map": str(seed_map_file.relative_to(REPO_ROOT)),
        "panel_content_sha256": seed_map["panel_content_sha256"],
        "n_rows": len(rows),
        "audit_inputs": [
            "hierarchy construct name and description",
            "selected historical program revision source",
            "methods/metric_seam/hybrids/ops_math.py implementation",
        ],
        "forbidden_inputs": [
            "candidate program execution",
            "items or item identifiers",
            "reference judgments or outcome labels",
            "heldout identifiers or outputs",
            "program outputs or correlations",
            "reconstruction or isomorphism results",
            "model/API calls",
        ],
        "execution_performed": False,
        "ops_math_source": "methods/metric_seam/hybrids/ops_math.py",
        "ops_math_sha256": _sha256(
            REPO_ROOT / "methods/metric_seam/hybrids/ops_math.py"
        ),
        "provenance": {
            "retrieval": "frozen source-only retrospective retrieval from the historical math hybrid bank",
            "program_authorship": "manual historical hybrids, commonly train-residual-informed; no automatic-discovery claim",
            "audit": "independent static reading of selected program and capability implementation",
            "llm_field_policy": "LLM_FIELDS are disclosed but excluded from every implemented_relations list and cannot establish code fidelity",
        },
        "adjudication_policy": {
            "exact": "Code implements the operative requested input-to-property relation for the whole construct.",
            "partial": "Code implements at least one decision-contributing requested subrelation, but the residual construct remains unimplemented.",
            "mismatch": "The candidate computes features, but none implement a requested operative subrelation; topical or naming overlap is insufficient.",
            "no_candidate_bounded_non_discovery": "No candidate survived the frozen source-only retrieval in this bank and budget.",
            "negative_interpretation": "Mismatch, abstention, and later execution failure are bounded non-discovery; none establish tacitness, inarticulability, or non-verifiability.",
        },
        "audited_depth_vocabulary": {
            "0": "surface lexical operation",
            "1": "parsed math/document structure or within-unit aggregation",
            "2": "cross-span, Q/A-section, reuse, or positional relation",
            "3": "formal solver or evidence-graph execution",
            "4": "environment, code, or test execution",
            "rule": "Deepest decision-contributing code-only operation actually present; the seed mapper's declared depth is not inherited.",
        },
        "capability_limit": "MathOps performs document-local LaTeX parsing/tokenization and structural aggregation only; it has no CAS, proof assistant, formal proof checking, answer execution, retrieval corpus, visual parser, or artifact-reproducibility runner.",
        "counts": counts,
        "interpretation": "This artifact measures static code-only relation fidelity, not codability prevalence. Partial rows may seed relation-local execution only. It makes no articulability, verifiability outcome, reconstruction, isomorphism, or automatic-discovery claim.",
        "rows": rows,
    }
    validate(artifact, seed_map)
    return artifact


def validate(artifact: dict[str, Any], seed_map: dict[str, Any]) -> None:
    rows = artifact["rows"]
    if len(rows) != 60 or len({r["cell_id"] for r in rows}) != 60:
        raise ValueError("artifact must contain 60 unique cells")
    if Counter(r["level"] for r in rows) != {"R1": 30, "R2": 30}:
        raise ValueError("artifact must contain 30 cells per level")
    source = {
        r["cell_id"]: r
        for r in seed_map["rows"]
        if r["level"] in {"R1", "R2"}
    }
    if set(source) != {r["cell_id"] for r in rows}:
        raise ValueError("artifact cells do not exactly match seed-map R1/R2 cells")
    allowed = {"exact", "partial", "mismatch", "no_candidate_bounded_non_discovery"}
    for row in rows:
        src = source[row["cell_id"]]
        if row["verdict"] not in allowed:
            raise ValueError(f"invalid verdict: {row['cell_id']}")
        if not row["residual_construct"] or not row["justification"]:
            raise ValueError(f"missing adjudication text: {row['cell_id']}")
        if src["selected_seed"] is None:
            if row["candidate"] is not None or row["verdict"] != "no_candidate_bounded_non_discovery":
                raise ValueError(f"abstention mismatch: {row['cell_id']}")
            if row["audited_depth"] is not None or row["implemented_relations"]:
                raise ValueError(f"abstention cannot claim code: {row['cell_id']}")
            continue
        seed = src["selected_seed"]
        candidate = row["candidate"]
        if candidate is None:
            raise ValueError(f"missing candidate identity: {row['cell_id']}")
        for field in ("aspect_id", "source_heading", "selected_revision", "source_path"):
            if candidate[field] != seed[field]:
                raise ValueError(f"candidate identity mismatch {field}: {row['cell_id']}")
        if row["verdict"] == "no_candidate_bounded_non_discovery":
            raise ValueError(f"selected candidate marked no-candidate: {row['cell_id']}")
        if row["audited_depth"] not in {0, 1, 2, 3, 4}:
            raise ValueError(f"candidate missing valid depth: {row['cell_id']}")
        if not row["implemented_relations"]:
            raise ValueError(f"candidate missing code relation description: {row['cell_id']}")
        if row["eligible_for_relation_local_execution"] != (row["verdict"] in {"partial", "exact"}):
            raise ValueError(f"eligibility/verdict mismatch: {row['cell_id']}")
        if row["verdict"] == "partial" and row["scope"] != "subrelation_only":
            raise ValueError(f"partial scope mismatch: {row['cell_id']}")
        if row["verdict"] == "mismatch" and row["scope"] != "none":
            raise ValueError(f"mismatch scope mismatch: {row['cell_id']}")

    expected = {
        "R1": {"partial": 13, "mismatch": 5, "no_candidate_bounded_non_discovery": 12},
        "R2": {"partial": 6, "mismatch": 4, "no_candidate_bounded_non_discovery": 20},
    }
    for level, exp in expected.items():
        got = Counter(r["verdict"] for r in rows if r["level"] == level)
        if got != exp:
            raise ValueError(f"unexpected {level} verdict counts: {dict(got)}")
    if any(r["verdict"] == "exact" for r in rows):
        raise ValueError("no R1/R2 row passed whole-construct exactness")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-map", type=Path, default=DEFAULT_SEED_MAP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    seed_path = args.seed_map if args.seed_map.is_absolute() else REPO_ROOT / args.seed_map
    out_path = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    artifact = build(seed_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(artifact["counts"], indent=2))


if __name__ == "__main__":
    main()
