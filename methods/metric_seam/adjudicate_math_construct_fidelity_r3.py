"""Emit the independent static construct-fidelity audit for math R3 seeds.

This builder reads the frozen source-only seed map and the selected Python
sources as syntax/text.  It never imports or executes a historical hybrid and
never reads items, extracted LLM values, references, labels, scores, outputs,
or reconstruction results.  Every ``implemented_relations`` entry describes
behavior that remains decision-contributing when ``LLM_FIELDS`` are absent.
"""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SOURCE_MAP = ROOT / "outputs/metric_seam_pilot/hierarchy_r123/math_stackexchange_seed_map_v1.json"
OPS_MATH = ROOT / "methods/metric_seam/hybrids/ops_math.py"
OUTPUT = ROOT / (
    "outputs/metric_seam_pilot/hierarchy_r123/"
    "math_stackexchange_construct_fidelity_R3_v1.json"
)
SCHEMA = "metric-seam.math-static-construct-fidelity.v1"

DEPTH_VOCABULARY = {
    "0": "surface lexical operation",
    "1": "parsed math/document structure or within-unit aggregation",
    "2": "cross-span, Q/A-section, reuse, or positional relation",
    "3": "formal solver or evidence-graph execution",
    "4": "environment, code, or test execution",
    "rule": "Deepest decision-contributing code-only operation actually present; the seed mapper's declared depth is not inherited.",
}

INTERPRETATION = (
    "Partial authorizes at most the named code-only subrelation; it is not whole-construct "
    "verification. Mismatch and no-candidate verdicts are bounded failures within the frozen "
    "historical program class and static audit budget, never evidence of tacitness. Historical "
    "hybrids remain manually constructed and commonly train-residual-informed; this audit does "
    "not relabel them as automatically discovered programs."
)


def _a(
    *,
    expected_aspect: str | None,
    requested_relation: str,
    implemented_relations: list[str],
    residual_construct: str,
    verdict: str,
    depth: int | None,
    applicability: str,
    polarity: str,
    aggregation: str,
    caveats: list[str],
    justification: str,
) -> dict[str, Any]:
    return {
        "expected_aspect": expected_aspect,
        "requested_relation": requested_relation,
        "implemented_relations": implemented_relations,
        "residual_construct": residual_construct,
        "verdict": verdict,
        "scope": (
            "whole_construct" if verdict == "exact"
            else "subrelation_only" if verdict == "partial"
            else "none"
        ),
        "eligible_for_relation_local_execution": verdict in {"exact", "partial"},
        "audited_depth": depth,
        "polarity_aggregation_applicability_caveats": {
            "applicability": applicability,
            "polarity": polarity,
            "aggregation": aggregation,
        },
        "caveats": caveats,
        "justification": justification,
    }


def _none(requested_relation: str, residual_construct: str) -> dict[str, Any]:
    return _a(
        expected_aspect=None,
        requested_relation=requested_relation,
        implemented_relations=[],
        residual_construct=residual_construct,
        verdict="no_candidate_bounded_non_discovery",
        depth=None,
        applicability="No candidate crossed the frozen source-only retrieval gate.",
        polarity="No executable polarity exists to audit.",
        aggregation="No executable aggregation exists to audit.",
        caveats=[
            "The static historical library search returned no selected program for this cell.",
            "Absence from the frozen program class is not evidence that the construct is tacit or uncodable.",
        ],
        justification=(
            "There is no selected implementation path. The whole requested relation remains a "
            "bounded non-discovery target."
        ),
    )


AUDITS: dict[str, dict[str, Any]] = {
    "TB::math-stackexchange::general::R3::grandparent::12::5312904bea74c529ade2": _none(
        "title/abstract/introduction/front matter -> accurate early orientation to the problem, result, and overview",
        "The complete entry-framing and front-matter construct remains unimplemented.",
    ),
    "TB::math-stackexchange::general::R3::grandparent::5::be12d3c9ffec8d99a472": _a(
        expected_aspect="a42",
        requested_relation=(
            "figures/tables plus their encodings, labels, captions, and surrounding claims -> "
            "truthful, clear, publication-ready, self-sufficient visualization"
        ),
        implemented_relations=[
            "assign a fixed low base to normalized nonempty text",
            "penalize four-or-more high-precision decimal tokens as a numeric-verification-table genre",
            "subtract a small penalty when the LaTeX delimiter scanner reports multiple issue categories",
        ],
        residual_construct=(
            "Figure/table presence, graphical encoding, scale and baseline integrity, labels, captions, "
            "chart design, truthfulness, and self-sufficiency are all unmeasured."
        ),
        verdict="mismatch",
        depth=1,
        applicability=(
            "Every normalized text of at least 30 characters receives a value; no figure, table, caption, "
            "or visualization occasion is required."
        ),
        polarity=(
            "High-precision decimal runs and malformed LaTeX only lower the fixed base; neither polarity "
            "establishes good or bad visualization design."
        ),
        aggregation=(
            "The code-only projection is nearly constant (0.05 before penalties), because elaboration is "
            "computed only after an excluded LLM gate fires."
        ),
        caveats=[
            "No image, plot, table structure, scale, caption, or claim-to-graphic relation is parsed.",
            "A precise numeric table can be truthful and useful, while a misleading plot can receive the base value.",
            "The selected h1 is a manual historical hybrid designed from training residuals.",
        ],
        justification=(
            "Once LLM_FIELDS are excluded, the program does not detect the object whose integrity is being "
            "judged. LaTeX hygiene and decimal density are not a functional subrelation of graphical design."
        ),
    ),
    "TB::math-stackexchange::general::R3::grandparent::6::054fe2aa8cf142b7fd45": _a(
        expected_aspect="a42",
        requested_relation=(
            "diagrams/visual representations plus argument context -> coherent visual reasoning whose "
            "epistemic role is justified within a rigorous framework"
        ),
        implemented_relations=[
            "assign a fixed low base to normalized nonempty text",
            "penalize high-precision decimal runs and multiple LaTeX delimiter-issue categories",
        ],
        residual_construct=(
            "Visual or diagram detection, integration with reasoning, claim support, epistemic warrant, "
            "rigorous use, and coherent example integration are unmeasured."
        ),
        verdict="mismatch",
        depth=1,
        applicability="No visual occasion is required; ordinary nonempty prose receives the same base.",
        polarity=(
            "The only varying code-only features are negative decimal-table and LaTeX-hygiene penalties, "
            "which do not order epistemically sound versus unsound visual reasoning."
        ),
        aggregation="The excluded LLM gates suppress the only positive example/visual branch, leaving a near-constant score.",
        caveats=[
            "Math-span extraction is used only inside an elaboration function that is gated off without LLM fields.",
            "No graph, diagram, spatial configuration, visual claim, or warrant is recognized in code.",
        ],
        justification="The surviving code does not functionally measure any requested visual-reasoning relation.",
    ),
    "TB::math-stackexchange::general::R3::merged_group::0::96891d12e95b8771ac29": _a(
        expected_aspect="a30",
        requested_relation=(
            "answer plus intended audience/background -> explicit assumptions and enough definitions, "
            "preliminaries, and scaffolding for self-contained comprehension"
        ),
        implemented_relations=[
            "isolate the Answer segment before computing structural signals",
            "relate proof-skeleton definition markers to nearby math-marker offsets as a grounded-definition count",
            "count case and logical-connective markers and normalize total scaffolding by sentence count",
            "add a small bonus when theorem, proof, or QED markers are present",
            "penalize math-span load unsupported by structural scaffolding, LaTeX delimiter problems, and long-sentence fraction",
        ],
        residual_construct=(
            "Audience identity, background fit, actual prerequisite coverage, semantic definition quality, "
            "undefined jargon, unexplained external pointers, tone, difficulty, and peripheral-reader on-ramps remain unmeasured."
        ),
        verdict="partial",
        depth=2,
        applicability="All nonempty answers are scored; the program does not establish that a target audience or new concept is present.",
        polarity=(
            "Grounded definitions, cases, and connectives raise the score; unsupported math load, markup problems, "
            "and long sentences lower it. This is aligned with local scaffolding but not audience fit."
        ),
        aggregation=(
            "A fixed weighted scalar combines scaffold density, formal markers, dump ratio, markup, and sentence shape; "
            "the excluded jargon/pointer penalties contribute nothing."
        ),
        caveats=[
            "Definition-marker proximity to a math delimiter does not show that the concept is defined correctly or when needed.",
            "Proof-skeleton regexes can count ordinary connective language as pedagogical scaffolding.",
            "Dense symbolic work can be appropriate for the intended audience.",
        ],
        justification=(
            "The marker-to-math proximity relation and scaffold-to-math-load aggregation implement a real local "
            "self-containment subrelation, but most audience and semantic coverage judgments remain outside code."
        ),
    ),
    "TB::math-stackexchange::general::R3::merged_group::20::60970a454f0b487605de": _none(
        "result/method plus related problem family -> clean generalization, transfer, reusable tools, and generative value",
        "Generalization, transfer, reuse, first applications, and generative value remain unimplemented.",
    ),
    "TB::math-stackexchange::general::R3::merged_group::19::4ee74b54866416859cfb": _none(
        "existence/realizability claim -> explicit construction or stepwise algorithm when feasible",
        "Claim typing, feasibility, construction correctness, and algorithmic transparency remain unimplemented.",
    ),
    "TB::math-stackexchange::general::R3::grandparent::8::2cb22f4124bacb534dec": _none(
        "proof steps plus hypotheses and edge cases -> logically valid, justified, complete reasoning without gaps",
        "Logical validity, warrants, assumption tracking, case completeness, and edge-condition coverage remain unimplemented.",
    ),
    "TB::math-stackexchange::general::R3::grandparent::17::6c6a55696fc92dcd5b7e": _a(
        expected_aspect="a126",
        requested_relation=(
            "problem-solving exposition -> effective discovery moves using patterns, analogy, experiment, "
            "planning, checking, and adaptation"
        ),
        implemented_relations=[
            "isolate the Answer segment before detecting discovery-process cues",
            "detect numeric experiment tables and repeated numeric sequences",
            "detect analogy phrases, guess/conjecture plus verify/check co-occurrence, named-heuristic phrases, and counterexample-shaped local windows",
            "count graphical/numerical observation phrases, rhetorical-question density, and multiple proof-skeleton cases",
            "combine the code-only cues with fixed weights and saturating transforms",
        ],
        residual_construct=(
            "Whether a detected move is mathematically meaningful, effective, genuinely analogical, appropriately planned, "
            "or adapted in response to evidence remains unmeasured."
        ),
        verdict="partial",
        depth=2,
        applicability="Every nonempty answer is scored; absence of detected cues receives a low floor rather than abstention.",
        polarity=(
            "More experiment, sequence, analogy, guess-check, heuristic, counterexample, graphical-observation, question, "
            "or multi-case cues increase the score. Presence can still be perfunctory or unsuccessful."
        ),
        aggregation=(
            "Nine cue families are weighted and averaged after saturation; the excluded discovery_move field supplies no boost."
        ),
        caveats=[
            "Most cues are regex presence or document-level co-occurrence rather than validated mathematical process.",
            "Direct rigorous derivations can appropriately omit discovery narration.",
            "Rhetorical questions and numerical tables can occur without productive heuristic reasoning.",
        ],
        justification=(
            "Several code relations capture articulated discovery-process forms, especially guess-to-check co-occurrence "
            "and numeric experimentation, but they cannot judge effectiveness or adaptation."
        ),
    ),
    "TB::math-stackexchange::general::R3::grandparent::14::dcbeab76010542fcf042": _none(
        "counterexample/pathology plus claim or concept -> constructive delimitation, refinement, and learning",
        "Counterexample correctness, target-claim relation, boundary refinement, and pedagogical use remain unimplemented.",
    ),
    "TB::math-stackexchange::general::R3::merged_group::10::a7ca222d4664b6e8021d": _none(
        "proof plus assumptions/cases -> field-standard logical correctness and completeness with only reconstructable omissions",
        "Inference validity, hidden assumptions, case coverage, gap severity, and reconstructability remain unimplemented.",
    ),
    "TB::math-stackexchange::general::R3::merged_group::5::4a713fb748549d3d0245": _none(
        "plot/figure plus data and claim -> truthful encoding, scale/baseline integrity, efficient design, labels, context, and pattern revelation",
        "All graphical objects, encodings, scale/baseline checks, labels, context, and chart design remain unimplemented.",
    ),
    "TB::math-stackexchange::general::R3::merged_group::11::d5ce3c2432ca508dba8a": _a(
        expected_aspect="a72",
        requested_relation=(
            "claim and answer method -> a natural, simple, commensurate proof technique without overkill, "
            "gratuitous casework, or empty symbol-pushing"
        ),
        implemented_relations=[
            "isolate the Answer segment and initialize the code-only score at a neutral base",
            "strongly penalize a near-empty math-free answer",
            "slightly penalize more than four proof-skeleton case markers",
            "slightly penalize answers with more than ten math spans and high average token length",
        ],
        residual_construct=(
            "Actual method identity, fit to the claim, simplicity relative to alternatives, mathematical completeness, "
            "whether casework is gratuitous, and whether symbol manipulation is purposeful remain unmeasured."
        ),
        verdict="partial",
        depth=1,
        applicability="All answers longer than ten characters are scored; no proof or method occasion is required.",
        polarity=(
            "Near-empty responses and extreme case/math-density shapes lower the neutral base, matching named anti-patterns "
            "only as noisy form proxies."
        ),
        aggregation=(
            "Without LLM labels the output is a narrow set of downward nudges from 0.5; code cannot award an appropriate technique."
        ),
        caveats=[
            "Many cases or dense formulas can be the natural method, and short answers can be mismatched or wrong.",
            "The code-only range is deliberately narrow and mostly constant.",
            "The core method-fit and completeness decisions live entirely in excluded LLM_FIELDS.",
        ],
        justification=(
            "The surviving code implements only the explicitly named negative form subrelations of excessive casework, "
            "dense symbol manipulation, and vacuous response shape; it does not judge technique appropriateness."
        ),
    ),
    "TB::math-stackexchange::general::R3::grandparent::4::573bc6a1b44aee4f6236": _a(
        expected_aspect="a198",
        requested_relation=(
            "answer citations and attributions -> accurate, complete, reliable, precise, fairly credited, ethical sourcing"
        ),
        implemented_relations=[
            "isolate the Answer segment before citation-pattern analysis",
            "detect theorem/lemma/section/chapter/equation and page locators, generic source nouns, see/cf phrases, and LaTeX citation commands",
            "detect numbered/tagged/reference tokens and reward repeated reuse of the same token as internal cross-reference discipline",
            "combine external-locator and internal-numbering signals by maximum plus a small two-channel synergy",
        ],
        residual_construct=(
            "Actual source identity, relevance, reliability, claim support, attribution accuracy, completeness, fair credit, "
            "retrievability, and citation ethics remain unmeasured."
        ),
        verdict="partial",
        depth=2,
        applicability="Every answer is scored; answers with no citation occasion receive zero rather than an applicability abstention.",
        polarity=(
            "Pinpoint locators, citation commands, and reused internal reference tokens increase the score. A locator without "
            "a named source can still receive credit."
        ),
        aggregation=(
            "External and internal subscores are combined by max plus 0.15 times their minimum; the excluded cited_source field "
            "removes named-source grounding. The equation_stats tuple is ignored because the program checks for a dict."
        ),
        caveats=[
            "Bare parenthesized integers can still be mistaken for cross-references.",
            "A citation command or locator does not show that the source is correct, reliable, or fairly credited.",
            "The selected h1 is a manual train-residual-informed revision.",
        ],
        justification=(
            "Pinpoint-locator and repeated-reference relations implement citation precision and internal cross-reference form, "
            "but the substantive sourcing and ethical construct remains outside code."
        ),
    ),
    "TB::math-stackexchange::general::R3::grandparent::10::2cdcd5d981d447ebf94c": _a(
        expected_aspect="a66",
        requested_relation=(
            "mathematical exposition -> economical, idea-forward presentation that foregrounds the crux with aesthetic clarity"
        ),
        implemented_relations=[
            "isolate the Answer segment and compute word count, parsed math density, sentence count, and proof-skeleton connective/QED counts",
            "in the code-only fallback, reward shorter answers and connective markers",
            "penalize long answers, dense equation walls with little prose, and long QED-marked derivations",
            "cap short math-free or leading counter-question/pointer deflection shapes at a low score",
        ],
        residual_construct=(
            "Identification of the key idea, its mathematical power, whether it is genuinely foregrounded, proof correctness, "
            "and holistic elegance or aesthetic clarity remain unmeasured."
        ),
        verdict="partial",
        depth=1,
        applicability="Every nonempty answer is scored; no distinct key idea must be identified.",
        polarity=(
            "Concision and connective-marked exposition raise the fallback, while length, equation walls, and deflection lower it. "
            "This tracks economy but can reward shallow brevity."
        ),
        aggregation="A length-decay fallback plus small bonuses/penalties produces the entire score when insight/foregrounded fields are absent.",
        caveats=[
            "Shortness is not elegance and long derivations can foreground a deep idea clearly.",
            "The deflection regex and word/sentence statistics are surface form proxies.",
            "No code relation extracts or evaluates the crux itself.",
        ],
        justification=(
            "The program implements economy-versus-bookkeeping and deflection subrelations, but the central key-idea and "
            "aesthetic judgments remain prompt-side."
        ),
    ),
    "TB::math-stackexchange::general::R3::grandparent::18::7b74dfeee0d2aff90266": _a(
        expected_aspect="a96",
        requested_relation=(
            "claims, proof status, gaps, and revision context -> clear distinction among proved results, conjectures, "
            "heuristics, corrections, and unresolved limitations"
        ),
        implemented_relations=[
            "isolate the Answer segment and count explicit Proof/QED markers as a weak proved-status signal",
            "slightly reward proof-skeleton connective density",
            "penalize a thin assertion shape with at most one very short math span and at most three sentences",
        ],
        residual_construct=(
            "Conjecture/heuristic labeling, gap detection, honesty of certainty, correction/revision/errata communication, "
            "and consistency between claimed status and actual proof remain unmeasured."
        ),
        verdict="partial",
        depth=1,
        applicability="Every nonempty answer is scored; no proof-status or correction occasion is required.",
        polarity=(
            "Formal Proof/QED and connective structure increase the score and thin assertions decrease it. A dishonest QED can "
            "therefore receive the wrong polarity."
        ),
        aggregation=(
            "With rigor_label and overclaim absent, code applies only a small marker bonus, thinness penalty, and connective bonus around 0.5."
        ),
        caveats=[
            "The program never detects conjecture, heuristic, gap, revision, correction, or erratum language in code.",
            "Proof/QED presence is status form, not proof validity or honest calibration.",
            "The code-only dynamic range is narrow because the core status judgment is in LLM_FIELDS.",
        ],
        justification=(
            "Explicit Proof/QED marking is a narrow proved-status articulation and thin-assertion structure is a support cue, "
            "so a relation-local partial survives, but almost all transparency and correction content is residual."
        ),
    ),
    "TB::math-stackexchange::general::R3::merged_group::16::bedf9a334646bda9ae1e": _a(
        expected_aspect="a168",
        requested_relation=(
            "formulas and surrounding prose -> grammatical integration with complete sentences, connectives, punctuation, "
            "and no symbol-only chains or article misuse"
        ),
        implemented_relations=[
            "isolate the Answer segment and compute LaTeX delimiter health",
            "parse math spans and penalize isolated logical/quantifier command spans used in prose while exempting standard direction labels",
            "reward a fixed set of prose quantifier/connective idioms and short relational-chain shapes",
            "penalize document-level co-occurrence of congruence notation with mod language as a possible relation mismatch",
        ],
        residual_construct=(
            "Full grammatical roles, sentence completeness, punctuation around formulas, article use, technical-term correctness, "
            "semantic symbol choice, prose coherence, and ambiguity remain unmeasured."
        ),
        verdict="partial",
        depth=2,
        applicability="Every nonempty answer is scored, including all-prose answers with no formula-integration occasion.",
        polarity=(
            "Delimiter health, prose idioms, and relational chains raise the score; isolated bare logical symbols and cong/mod "
            "co-occurrence lower it. Several defaults are constants when LLM fields are absent."
        ),
        aggregation=(
            "Six fixed weighted components are summed; code-only variation comes from hygiene, bare-span, idiom, chain, and cong/mod terms."
        ),
        caveats=[
            "Fixed phrase and regex catalogs do not parse English grammar or formula punctuation.",
            "A relational chain can be hard to read, and an isolated symbol can be conventional in context.",
            "The cong/mod co-occurrence penalty does not locate or type-check the relation use.",
        ],
        justification=(
            "The bare-symbol-to-prose and connective/formula-form checks directly implement a narrow integration subrelation, "
            "while grammatical and semantic readability remain outside code."
        ),
    ),
    "TB::math-stackexchange::general::R3::merged_group::4::7a187ce24529c9c48ce1": _a(
        expected_aspect="a198",
        requested_relation=(
            "citations and attributions -> relevant, reliable, accurate, specific, retrievable, consistent, complete sourcing without padding"
        ),
        implemented_relations=[
            "isolate the Answer segment and detect source-locator phrases, page numbers, see/cf markers, generic source nouns, and LaTeX citation commands",
            "detect numbered/tagged/reference tokens and repeated token reuse as internal cross-reference discipline",
            "combine external and internal structural signals by maximum plus a small synergy term",
        ],
        residual_construct=(
            "Named-source identity, relevance, reliability, attribution accuracy, retrieval success, claim-level support, "
            "citation completeness, consistency, and padding remain unmeasured."
        ),
        verdict="partial",
        depth=2,
        applicability="All answers receive a value; absence of a citation is treated as zero without checking whether sourcing is needed.",
        polarity="Locators, citation commands, and repeated internal references raise the score, but can be irrelevant or inaccurate.",
        aggregation=(
            "External and internal form scores use max-plus-synergy; named-source grounding is absent with cited_source excluded, "
            "and MathOps equation_stats numbering is silently ignored because it returns a tuple rather than the expected dict."
        ),
        caveats=[
            "Locator syntax does not establish a retrievable, relevant, or reliable source.",
            "Parenthesized integers and equation identifiers can be false cross-reference positives.",
            "Citation necessity and padded reference lists are not modeled.",
        ],
        justification=(
            "Precise-locator and cross-reference-reuse form are real subrelations of citation precision, but substantive source quality is residual."
        ),
    ),
    "TB::math-stackexchange::general::R3::merged_group::2::e7c15a601217e58c6d50": _a(
        expected_aspect="a60",
        requested_relation=(
            "introduction/opening -> accessible motivation, early main result, context/literature/applications, and roadmap/strategy"
        ),
        implemented_relations=[
            "isolate the Answer segment and inspect its first 220 characters",
            "classify direct-result, question-back, hedge, and numbered/structured opening patterns with fixed regexes",
            "relate the earliest proof-skeleton connective/QED offset to total answer length",
            "blend opening and early-connective scores with deterministic question/direct overrides",
        ],
        residual_construct=(
            "Motivation, accessibility, literature/application context, roadmap quality, actual result correctness, and whether "
            "the opening accurately summarizes the contribution remain unmeasured."
        ),
        verdict="partial",
        depth=2,
        applicability="All nonempty answers are scored, including answers whose genre does not require a formal introduction.",
        polarity=(
            "Direct phrases and early connectives raise the score; questions and hedge-shaped openings lower it. Words such as "
            "clearly or note that can trigger false direct-result credit."
        ),
        aggregation="The code-only path is 80% opening classification and 20% earliest-connective position, followed by hard caps/floors.",
        caveats=[
            "A direct lexical cue can introduce setup rather than the main result.",
            "A motivated introduction can appropriately begin with a question or assumption.",
            "Only the Answer segment is available; titles and abstracts are not separately represented.",
        ],
        justification=(
            "Opening-window and positional-connective relations directly implement the early-result subrelation, while the rest "
            "of introduction quality remains outside code."
        ),
    ),
    "TB::math-stackexchange::general::R3::grandparent::21::0be03122563d028dc2e2": _none(
        "sentences plus embedded formulas -> grammatical, unambiguous mechanics, coherent tense/connectives, and seamless symbol integration",
        "Grammar parsing, tense/cohesion, ambiguity, and formula-to-sentence integration remain unimplemented for this retrieved cell.",
    ),
    "TB::math-stackexchange::general::R3::grandparent::13::65bdccb4bd57bd095f25": _a(
        expected_aspect="a42",
        requested_relation=(
            "examples/simple cases/visuals/multiple representations plus concepts -> useful grounding, motivation, and clarification"
        ),
        implemented_relations=[
            "assign a fixed low base to normalized nonempty text",
            "penalize high-precision decimal-table genre and multiple LaTeX delimiter-issue categories",
        ],
        residual_construct=(
            "Example or visual presence, whether it is additional grounding, representational multiplicity, concept-to-example relation, "
            "motivation, clarification, correctness, and integration are all unmeasured."
        ),
        verdict="mismatch",
        depth=1,
        applicability="No example/visual occasion is required and code does not positively detect one without the excluded LLM gates.",
        polarity=(
            "Only numeric-verification and malformed-LaTeX penalties vary; a strong worked example and no example can receive the same base."
        ),
        aggregation="The positive elaboration computation is unreachable in the code-only projection, so aggregation is near-constant.",
        caveats=[
            "Regex example markers, case markers, and numeric span density are only evaluated after an LLM-confirmed signal.",
            "The code-only behavior cannot distinguish illustrative grounding from absence of grounding.",
        ],
        justification=(
            "After excluding LLM_FIELDS, no functional example/representation detector contributes to the scalar; the remaining penalties do not match the construct."
        ),
    ),
    "TB::math-stackexchange::general::R3::merged_group::1::1e5c7fb7705d8ee96a18": _a(
        expected_aspect="a54",
        requested_relation=(
            "sections/proofs/paragraphs -> logical macro-organization, informative navigation, previews/transitions/summaries, and traceable hypothesis use"
        ),
        implemented_relations=[
            "isolate the Answer segment and count sentences, mean words per sentence, numbered equations, and formal theorem/definition/proof/QED markers",
            "reward any numbered equation, up to two formal markers, and at least two sentences",
            "penalize high mean words per sentence and aggregate LaTeX delimiter issues",
        ],
        residual_construct=(
            "Section/paragraph order, headings, previews, transitions, summaries, explicit partition labels, upfront claim, skimmability, "
            "argument-flow tracking, and hypothesis-use traceability remain unmeasured."
        ),
        verdict="partial",
        depth=1,
        applicability="Every nonempty answer is scored; no long-argument or navigation need is required.",
        polarity=(
            "Formal/numbered structure and multiple sentences raise the score; long sentences and delimiter problems lower it. "
            "Mere numbering or formal markers need not improve navigation."
        ),
        aggregation=(
            "A fixed 0.30 base receives small additive form bonuses and sentence/delimiter penalties; both semantic organization LLM fields are zero."
        ),
        caveats=[
            "Numbered equations are not checked for later reference or navigational use.",
            "Formal-marker counts do not establish motivated decomposition or coherent order.",
            "Heading, paragraph, transition, and summary structure is not parsed.",
        ],
        justification=(
            "Numbered/formal structure and mechanical sentence readability are narrow organization-form subrelations, not macro-navigation as a whole."
        ),
    ),
    "TB::math-stackexchange::general::R3::merged_group::9::6d40ac86f7e9adf63d2b": _none(
        "each proof inference plus its hypotheses/definitions/prior lines -> correct implication with explicit warrant and valid goal reduction",
        "Inference checking, warrant linkage, hypothesis tracking, goal-reduction validity, and omission severity remain unimplemented.",
    ),
    "TB::math-stackexchange::general::R3::merged_group::14::da47b04ffaa9bf294ae9": _a(
        expected_aspect="a36",
        requested_relation=(
            "long proof -> motivated modular lemmas/claims with explicit dependencies and short transparent lowest-level steps"
        ),
        implemented_relations=[
            "count numbered equations and relate repeated tag/parenthesized-number tokens as potential backreferences",
            "count proof-skeleton cases, formal theorem/definition/proof/QED markers, and logical-connective density",
            "score sentence and average equation-token transparency and LaTeX delimiter health",
            "penalize very long answers with zero counted structural scaffolding",
        ],
        residual_construct=(
            "Whether modules are motivated or each does one job, actual logical dependency edges, lemma correctness, claim-to-proof linkage, "
            "stage labels outside the skeleton catalog, and semantic transparency remain unmeasured."
        ),
        verdict="partial",
        depth=2,
        applicability="Every nonempty answer is scored; no long or decomposable proof is required.",
        polarity=(
            "Numbering, repeated reference tokens, cases, formal markers, and connectives raise the score; shorter sentences/equations and healthy delimiters also raise it."
        ),
        aggregation=(
            "Eight saturated components are weighted and averaged, then a long-zero-scaffolding penalty is applied; excluded STAGE_LABELS add no cases."
        ),
        caveats=[
            "Bare parenthesized numbers can create false backreferences, including function values.",
            "Counts do not reconstruct a dependency graph or verify that a lemma is motivated and cohesive.",
            "Short equations and many connectives are not necessarily transparent reasoning.",
        ],
        justification=(
            "Repeated-reference and scaffolding relations are genuine structural surveyability subrelations with cross-span depth, "
            "but modular purpose and logical dependencies remain semantic residuals."
        ),
    ),
    "TB::math-stackexchange::general::R3::grandparent::1::0314c80be0eeaaa8afb8": _a(
        expected_aspect="a54",
        requested_relation=(
            "answer-level exposition -> previews, transitions, method declarations, explicit assumptions/goals, reminders, and closures that orient readers"
        ),
        implemented_relations=[
            "count numbered equations and formal theorem/definition/proof/QED markers in the Answer segment",
            "reward at least two sentences and penalize high mean sentence length and LaTeX delimiter issues",
        ],
        residual_construct=(
            "Previews, transitions, method declarations, assumptions, goals, reminders, closures, upfront result, and explicit "
            "addressable partitioning remain unmeasured."
        ),
        verdict="partial",
        depth=1,
        applicability="All nonempty answers are scored even when global orientation devices are unnecessary.",
        polarity=(
            "Formal/numbered structure and multiple sentences receive small positive weights; long sentences and broken delimiters are negative. "
            "The polarity is at most a mechanical orientation proxy."
        ),
        aggregation="Small additive code bonuses/penalties modify a fixed base; answer_upfront and explicit_partition are excluded.",
        caveats=[
            "Formal markers and numbered equations can occur without any guidance to the reader.",
            "No positional, transition, assumption, goal, reminder, or closure relation is evaluated.",
        ],
        justification=(
            "The candidate supplies only a narrow explicit-structure/readability subrelation; the requested global guidance remains almost entirely residual."
        ),
    ),
    "TB::math-stackexchange::general::R3::grandparent::20::4731069f85003f46da5f": _a(
        expected_aspect="a108",
        requested_relation=(
            "result/method plus field and prior art -> novelty/originality, significance, sharpness, applications, generativity, and impact"
        ),
        implemented_relations=[
            "isolate the Answer segment and count display spans, average equation-token length, cases, connectives, and notation-census diversity",
            "combine those counts as a low-weight structural depth/complexity score",
        ],
        residual_construct=(
            "Prior-art comparison, novelty, originality, technique unusualness, correctness, result strength/sharpness, significance, applications, "
            "transfer, generativity, adoption, and broader impact are all unmeasured."
        ),
        verdict="mismatch",
        depth=1,
        applicability="Every nonempty answer is scored; no contribution or field-comparison occasion is required.",
        polarity=(
            "More display math, cases, connectives, notation diversity, and longer equations increase the structural score, but complexity has no reliable novelty or impact polarity."
        ),
        aggregation=(
            "With both LLM novelty fields absent, output is 0.05 plus at most 0.15 times structural complexity; the code-only range is compressed and proxy-based."
        ),
        caveats=[
            "Routine textbook derivations can be structurally deep and genuinely novel arguments can be short.",
            "No corpus, prior art, citation graph, field baseline, application evidence, or uptake evidence is consulted.",
            "The selected h1 is a manual train-residual-informed hybrid revision.",
        ],
        justification=(
            "Structural complexity is not a functionally matched subrelation of novelty, significance, or impact; the actual novelty gate is entirely in excluded LLM_FIELDS."
        ),
    ),
    "TB::math-stackexchange::general::R3::merged_group::3::e7911c7b707a53bacba4": _a(
        expected_aspect="a180",
        requested_relation=(
            "notation/terminology over the answer -> conventional, simple, mnemonic, timely introduced, one-to-one, consistent, economical use"
        ),
        implemented_relations=[
            "parse LaTeX delimiter health and normalize issue count by number of math spans",
            "build a notation census and compare bare versus command spellings for a fixed catalog of mathematical functions across the answer",
            "apply a mild penalty when average math-span token length exceeds sixty",
        ],
        residual_construct=(
            "Symbol meaning, one-symbol-per-object and one-meaning-per-symbol semantics, timely introduction, definition quality, mnemonic naming, "
            "terminology, conventionality beyond the function catalog, ambiguity, and economy remain unmeasured."
        ),
        verdict="partial",
        depth=2,
        applicability="All nonempty answers are scored; with no recognized functions or math, consistency defaults high.",
        polarity=(
            "Delimiter problems, mixed bare/command spellings, and very long spans lower the score; absence of observed clashes receives full consistency credit."
        ),
        aggregation=(
            "Code score is 55% delimiter hygiene and 45% fixed-function spelling consistency, optionally multiplied by 0.85 for long spans; LLM nudges are absent."
        ),
        caveats=[
            "The census detects surface spelling clashes, not two meanings for one symbol or two symbols for one object.",
            "No evidence that a notation was introduced before use is computed.",
            "A clean answer using undefined or idiosyncratic symbols can score highly.",
        ],
        justification=(
            "Document-wide representation-consistency and delimiter hygiene are direct notation-consistency subrelations, "
            "but naming, meaning, introduction, and economy remain residual."
        ),
    ),
    "TB::math-stackexchange::general::R3::merged_group::18::1314ecb88addef16bee4": _none(
        "primary and alternative proofs/reformulations/applications -> illuminating comparison of differences, strengths, and transfer",
        "Alternative detection, correctness, comparative strengths, perspective, application, and transfer remain unimplemented.",
    ),
    "TB::math-stackexchange::general::R3::grandparent::0::e7acd1ba1eac09a8a06a": _a(
        expected_aspect="a30",
        requested_relation=(
            "answer plus readership/background -> suitable level, detail, explicitness, accessibility, self-containment, and breadth toward non-specialists"
        ),
        implemented_relations=[
            "isolate the Answer segment and relate definition markers to nearby math delimiters",
            "count case/connective scaffolding relative to sentence count and compare math-span load to scaffolding density",
            "add a small bonus when theorem, proof, or QED markers are present",
            "penalize LaTeX delimiter issues and long-sentence fraction",
        ],
        residual_construct=(
            "Identified readership, actual background assumptions, level/difficulty fit, explanatory adequacy, semantic definitions, undefined jargon, "
            "unexplained pointers, accessibility accommodations, and breadth remain unmeasured."
        ),
        verdict="partial",
        depth=2,
        applicability="Every nonempty answer receives a score without identifying an audience or accessibility need.",
        polarity=(
            "Grounded structural scaffolding raises the score and unsupported symbolic load, markup problems, and long sentences lower it; "
            "the same form can be appropriate or inappropriate for different audiences."
        ),
        aggregation=(
            "Fixed scaffold-density, formal-marker, dump, markup, and sentence components are combined; excluded jargon/pointer fields remove the only audience-relative checks."
        ),
        caveats=[
            "No question-derived audience model or prerequisite inventory is constructed.",
            "Definition-marker proximity does not verify a usable explanation.",
            "Sentence and math densities are not calibrated to any readership.",
        ],
        justification=(
            "The candidate implements local scaffolding/self-containment form, a real subrelation, but cannot evaluate audience targeting or level fit."
        ),
    ),
    "TB::math-stackexchange::general::R3::grandparent::16::570eed33fe5f1ce2a120": _a(
        expected_aspect="a210",
        requested_relation=(
            "mathematical markup -> consistent readable typesetting, correct labels/numbering, and conservative editorial conventions"
        ),
        implemented_relations=[
            "parse math spans and delimiter health, including unclosed delimiters, brace/environment and left/right mismatch, naked commands, and adjacent inline formulas",
            "penalize eqnarray and deprecated font-switch commands",
            "penalize bare multi-letter function names, letter-digit forms without subscripts, repeated manual spacing, mixed Unicode/command symbols, and scripts trapped inside font-command braces",
            "reward a code-clean document and return neutral when no parseable math span exists",
        ],
        residual_construct=(
            "Label and numbering correctness, reference integrity, layout/readability beyond fixed defects, semantic symbol choice, full style-guide compliance, "
            "editorial consistency outside math spans, and publication context remain unmeasured."
        ),
        verdict="partial",
        depth=2,
        applicability="No parseable math yields a neutral 0.5; any answer with at least one parsed math span is scored without an explicit typesetting-opportunity gate.",
        polarity=(
            "Each cataloged defect lowers the score and a defect-free catalog scan raises it. Several regexes can flag legitimate notation or miss equivalent defects."
        ),
        aggregation=(
            "Capped additive defect penalties are subtracted from 0.55, with a 0.19 clean-code bonus; the notation_issue LLM penalty is absent."
        ),
        caveats=[
            "Letter-digit detection can flag conventional names, and manual spacing can be intentional.",
            "The parser is a MathJax-oriented scanner, not a full TeX engine.",
            "No label/reference pair or numbering sequence is actually validated despite the parent construct naming them.",
        ],
        justification=(
            "The delimiter, command, spacing, font, and representation checks directly implement a substantial typesetting-convention subrelation, "
            "but labels/numbering and broader editorial mechanics remain residual."
        ),
    ),
    "TB::math-stackexchange::general::R3::merged_group::7::85e1c7911132b8d8f013": _none(
        "code/data/materials/method description plus access conditions -> documented availability, transparent sources/limitations/corrections, and independent replicability",
        "Artifact availability, access, documentation, source reporting, limitations, corrections, method sufficiency, and replication remain unimplemented.",
    ),
}


def _literal_llm_fields(source_path: Path) -> list[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        target = value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and target.id == "LLM_FIELDS" and value is not None:
            literal = ast.literal_eval(value)
            if not isinstance(literal, dict):
                raise ValueError(f"{source_path}: LLM_FIELDS is not a literal dict")
            return sorted(str(key) for key in literal)
    return []


def _source_candidate(seed: Mapping[str, Any], expected_aspect: str) -> dict[str, Any]:
    if seed.get("aspect_id") != expected_aspect:
        raise ValueError(
            f"candidate identity drift: expected {expected_aspect}, found {seed.get('aspect_id')}"
        )
    source_path = ROOT / seed["source_path"]
    if not source_path.is_file():
        raise ValueError(f"selected program missing: {source_path}")
    observed_fields = _literal_llm_fields(source_path)
    declared_fields = sorted(seed["hybrid_provenance"]["llm_field_names"])
    if observed_fields != declared_fields:
        raise ValueError(f"{expected_aspect}: LLM_FIELDS identity drift")
    return {
        "aspect_id": seed["aspect_id"],
        "source_heading": seed["source_heading"],
        "selected_revision": seed["selected_revision"],
        "source_path": seed["source_path"],
        "program_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "historical_hybrid_provenance": seed["hybrid_provenance"]["historical_construction"],
        "llm_fields_excluded_from_implemented_relations": observed_fields,
    }


def build_artifact(seed_map: Mapping[str, Any]) -> dict[str, Any]:
    if seed_map.get("schema") != "metric-seam.hierarchy-math-seed-map.v1":
        raise ValueError("unexpected math seed-map schema")
    if seed_map.get("task") != "math-stackexchange" or seed_map.get("n_cells") != 90:
        raise ValueError("unexpected math seed-map scope")
    rows = [row for row in seed_map.get("rows", []) if row.get("level") == "R3"]
    if len(rows) != 30 or len({row.get("cell_id") for row in rows}) != 30:
        raise ValueError("R3 must contain exactly 30 unique cells")
    if set(AUDITS) != {row["cell_id"] for row in rows}:
        missing = {row["cell_id"] for row in rows} - set(AUDITS)
        extra = set(AUDITS) - {row["cell_id"] for row in rows}
        raise ValueError(f"audit closure mismatch; missing={sorted(missing)}, extra={sorted(extra)}")

    output_rows = []
    for source_row in rows:
        cell_id = source_row["cell_id"]
        audit = AUDITS[cell_id]
        seed = source_row.get("selected_seed")
        if audit["expected_aspect"] is None:
            if seed is not None or source_row.get("decision") != "abstain":
                raise ValueError(f"{cell_id}: expected a source-map abstention")
            candidate = None
        else:
            if seed is None or source_row.get("decision") != (
                "candidate_seed_pending_independent_construct_fidelity_audit"
            ):
                raise ValueError(f"{cell_id}: expected a selected candidate")
            candidate = _source_candidate(seed, audit["expected_aspect"])

        if audit["verdict"] == "no_candidate_bounded_non_discovery":
            if candidate is not None or audit["implemented_relations"] or audit["audited_depth"] is not None:
                raise ValueError(f"{cell_id}: invalid no-candidate audit")
        elif candidate is None:
            raise ValueError(f"{cell_id}: candidate verdict without candidate")
        if audit["verdict"] == "partial" and not audit["implemented_relations"]:
            raise ValueError(f"{cell_id}: partial verdict has no implemented relation")
        if audit["verdict"] in {"mismatch", "no_candidate_bounded_non_discovery"} and audit[
            "eligible_for_relation_local_execution"
        ]:
            raise ValueError(f"{cell_id}: ineligible verdict marked eligible")
        if audit["audited_depth"] is not None and audit["audited_depth"] not in range(5):
            raise ValueError(f"{cell_id}: invalid depth")
        forbidden_strings = set(
            candidate["llm_fields_excluded_from_implemented_relations"]
            if candidate else []
        )
        relation_text = " ".join(audit["implemented_relations"])
        if any(field in relation_text for field in forbidden_strings):
            raise ValueError(f"{cell_id}: implemented relation names an excluded LLM field")

        output_rows.append({
            "cell_id": cell_id,
            "task": "math-stackexchange",
            "level": "R3",
            "metric_name": source_row["metric_name"],
            "metric_description": source_row["metric_description"],
            "candidate": candidate,
            "requested_relation": audit["requested_relation"],
            "implemented_relations": audit["implemented_relations"],
            "residual_construct": audit["residual_construct"],
            "verdict": audit["verdict"],
            "scope": audit["scope"],
            "eligible_for_relation_local_execution": audit[
                "eligible_for_relation_local_execution"
            ],
            "audited_depth": audit["audited_depth"],
            "polarity_aggregation_applicability_caveats": [
                "Applicability: " + audit[
                    "polarity_aggregation_applicability_caveats"
                ]["applicability"],
                "Polarity: " + audit[
                    "polarity_aggregation_applicability_caveats"
                ]["polarity"],
                "Aggregation: " + audit[
                    "polarity_aggregation_applicability_caveats"
                ]["aggregation"],
                *audit["caveats"],
            ],
            "justification": audit["justification"],
            "interpretation": INTERPRETATION,
        })

    verdict_counts = Counter(row["verdict"] for row in output_rows)
    depth_counts = Counter(
        str(row["audited_depth"])
        for row in output_rows
        if row["audited_depth"] is not None
    )
    eligible_depth_counts = Counter(
        str(row["audited_depth"])
        for row in output_rows
        if row["eligible_for_relation_local_execution"]
    )
    audited_depths = dict(sorted(depth_counts.items()))
    audited_depths["null"] = verdict_counts["no_candidate_bounded_non_discovery"]
    n_eligible = verdict_counts["partial"] + verdict_counts["exact"]
    level_counts = {
        "n_cells": len(output_rows),
        "n_retrieved_candidates": sum(row["candidate"] is not None for row in output_rows),
        "verdicts": dict(sorted(verdict_counts.items())),
        "eligible_for_relation_local_execution": n_eligible,
        "eligible_fraction_of_cells": round(n_eligible / len(output_rows), 6),
        "eligible_fraction_of_retrieved_candidates": round(
            n_eligible / sum(row["candidate"] is not None for row in output_rows), 6
        ),
        "audited_depths": audited_depths,
        "eligible_audited_depths": dict(sorted(eligible_depth_counts.items())),
    }
    return {
        "schema": SCHEMA,
        "status": "complete_static_code_only_adjudication_pre_execution",
        "design_scope": "outcome_blind_static_construct_fidelity",
        "task": "math-stackexchange",
        "levels": ["R3"],
        "source_candidate_map": str(SOURCE_MAP.relative_to(ROOT)),
        "panel_content_sha256": seed_map["panel_content_sha256"],
        "n_rows": len(output_rows),
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
        "ops_math_source": str(OPS_MATH.relative_to(ROOT)),
        "ops_math_sha256": hashlib.sha256(OPS_MATH.read_bytes()).hexdigest(),
        "audited_depth_vocabulary": DEPTH_VOCABULARY,
        "capability_limit": (
            "MathOps performs document-local LaTeX parsing/tokenization and structural aggregation only; "
            "it has no CAS, proof assistant, formal proof checking, answer execution, retrieval corpus, "
            "visual parser, or artifact-reproducibility runner."
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
        "counts": {"R3": level_counts, "overall": level_counts},
        "interpretation": (
            "This artifact measures static code-only relation fidelity, not codability prevalence. "
            "Partial rows may seed relation-local execution only. It makes no articulability, "
            "verifiability outcome, reconstruction, isomorphism, or automatic-discovery claim."
        ),
        "rows": output_rows,
    }


def main() -> int:
    seed_map = json.loads(SOURCE_MAP.read_text(encoding="utf-8"))
    artifact = build_artifact(seed_map)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(artifact["counts"]["R3"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
