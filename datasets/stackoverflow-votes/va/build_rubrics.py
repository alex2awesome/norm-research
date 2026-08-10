#!/usr/bin/env python3
"""V6 SO-votes A bank: merge + audit the mined proposals into the frozen bank.

PROVENANCE. Four label-blind proposer agents each read a disjoint batch of real
TRAIN-SPLIT exemplars (whole questions with all their answers, no y attached,
sampled by sha256("so-mine|"+question_id)) and proposed 16-18 candidate criteria
in the math.SE house style, each self-labelled Track A (real quality property)
or Track B (surface correlate). 70 candidates came back: 53 A, 17 B.
Raw proposals: so_proposals_b{0,1,2,3}.jsonl (scratchpad; contents reproduced in
the `origin` field of every entry below).

MERGE + AUDIT (this file). 22 duplicate clusters were collapsed -- the four
proposers converged hard, which is itself the redundancy signal the program
looks for. The surviving bank is 40 criteria:

  * 36 Track A "real" criteria.
  * 4 Track B "surface" probe criteria, DECLARED SPURIOUS UP FRONT. These are
    the corpus-matched probe pairs the charge specifies: the real pole is
    "diagnoses the actual cause of the asked problem" (here a01-a05) and the
    surface pole is "contains a code block" (here s01). They are scored in the
    SAME matrix so the surface channel is measurable inside the instrument
    rather than only assumed, and they carry track="B" so every downstream
    readout can split A_real from A_surface without a re-score.

DROPPED, with reasons (recorded rather than silently discarded):
  * "Answer exceeds typical length" (b2, Track B) -- a pure length proxy. The V
    channel already carries v_log_len / v_word_count / v_prose_char_count, so
    scoring it here would re-inject the length channel into A and contaminate
    the V-vs-A contrast, which is one of the ledger's load-bearing comparisons.
  * "Code snippet is self-contained" (b0) -- subsumed by "Code runs as posted".
  * "States method's range of applicability" (b0) -- subsumed by "States the
    boundary of applicability".
  * "Uses asker's own data as given" (b0) + "Reuses the asker's own names" (b3)
    -- subsumed by "Engages the asker's actual code".
  * "Nontrivial API call is attributed" (b0) -- subsumed by "Documentation link
    is usable in context".
  * "Reader can act without further help" (b0) -- duplicate of "Handover is
    actionable".
  * "Efficiency claim is substantiated" (b0) -- duplicate of "Performance claim
    is substantiated".
  * "Shown output matches question's example" (b0) -- duplicate of "Shown output
    matches the stated ask".
  * "Flags version-dependent syntax" (b0) + "Flags deprecated or version-bound
    usage" (b3) -- subsumed by "Version or platform dependency disclosed".
  * "Names the proper tool, not workaround" (b3) -- subsumed by "Idiomatic
    facility matches the structure".
  * "Trade-off against an alternative named" (b3) -- duplicate of "Alternatives
    compared with a stated reason".
  * "Asker's visible edge cases covered" (b1) -- duplicate of "Edge cases in
    asker's data addressed".
  * "Declares reading of ambiguous question" (b2) -- duplicate of "Resolves
    ambiguous ask before answering".
  * "Implicit misconception corrected" (b1) -- duplicate of "Corrects a
    misconception in the question".
  * "Pitched to the asker's level" (b1) -- duplicate of "Pitched to the asker's
    evident level".
  * duplicate Track B link/list/output/backtick criteria across batches --
    collapsed to one each.

GEPA-STYLE PHRASING RULES applied to every surviving entry:
  1. judgeable from the question title + this one answer, nothing else;
  2. never references votes, acceptance, other answers, order, or dates;
  3. two-sided where the property admits it (a match test, not a "more is
     better" test), so it cannot be read off length or formatting;
  4. an NA branch that is a real "attempts nothing bearing on this" case rather
     than a synonym for 0.0;
  5. one property per criterion.

  python3 datasets/stackoverflow-votes/va/build_rubrics.py
"""
from __future__ import annotations

import json
from pathlib import Path

# (name, description, track, origin)
BANK = [
    # ================= cause / diagnosis (the REAL probe pole) ==============
    ("Says why the original approach failed",
     "Whether the answer states the mechanism that made the asker's own code, or the naive "
     "approach implied by the question, produce the wrong result, rather than only supplying a "
     "working substitute. Score 1.0 when the answer says explicitly what causes the original to "
     "fail or misbehave; 0.5 when it hints at the cause without naming the mechanism; 0.0 when it "
     "supplies a working alternative and never addresses why the original went wrong; NA when the "
     "question presents no prior approach to diagnose, for instance a bare 'how do I do X' request.",
     "A", ["b0 'States why the original approach failed'"]),
    ("Mechanism behind the fix explained",
     "Whether the answer says why its fix works, naming the rule of the language or library that "
     "actually produces the behaviour, rather than only supplying a working replacement. Score 1.0 "
     "when the underlying mechanism is stated explicitly; 0.5 when a mechanism is named but not "
     "spelled out, so the reader cannot see how it produces the effect; 0.0 when a fix is supplied "
     "with no attempt to explain why it works though the question's own phrasing asks why; NA when "
     "the question asks only for a working snippet and raises no why-question.",
     "A", ["b1 'Mechanism behind the fix explained'"]),
    ("Symptom fix distinguished from cause fix",
     "Whether the answer is explicit about whether it is patching the visible symptom the asker hit "
     "or addressing the underlying condition that produced it. Score 1.0 when the answer states "
     "which of the two it is doing, or does both and marks the boundary; 0.5 when the distinction "
     "is implicit in how the answer is organised but never stated; 0.0 when it silently fixes only "
     "the symptom although the question's own description points at a deeper cause; NA when the "
     "question describes no failure whose symptom and cause could diverge.",
     "A", ["b1 'Symptom fix distinguished from cause fix'"]),
    ("Fix is tied to its diagnosis",
     "Where the answer identifies what is wrong, whether it connects the change it proposes back to "
     "that cause. Score 1.0 when the proposed change is explicitly linked to the stated cause; 0.5 "
     "when both a cause and a fix are given but the link is left for the reader to infer; 0.0 when "
     "a cause is named and an unmotivated fix follows beside it with no stated connection; NA when "
     "the answer offers no diagnosis for a fix to be linked to.",
     "A", ["b3 'Fix is tied to its diagnosis'"]),
    ("Proposes a way to narrow the cause",
     "Whether, instead of only guessing at a fix, the answer proposes a concrete test or observation "
     "the asker can make to narrow down what is actually wrong. Score 1.0 when a specific diagnostic "
     "step is proposed, in addition to or instead of a guessed fix; 0.5 when it suggests checking "
     "something but not how to check it; 0.0 when the cause is genuinely unclear from what the "
     "question gives and the answer jumps to a fix with no way to confirm it applies; NA when the "
     "question already makes the cause unambiguous.",
     "A", ["b3 'Proposes a way to narrow cause'"]),

    # ============================ correctness ==============================
    ("Code runs as posted",
     "Whether the code supplied is complete and internally consistent enough to execute without the "
     "reader guessing at missing pieces: no undefined names left from a partial edit, no broken "
     "indentation carried over from example code, no half-finished statement. Score 1.0 when the "
     "block could be run as shown, modulo the reader's own data setup; 0.5 when it runs after one "
     "small obvious repair such as a missing colon or import; 0.0 when it contains undefined "
     "references or structurally broken syntax the reader could not resolve without guessing the "
     "author's intent; NA when the answer contains no code.",
     "A", ["b1 'Code runs as posted'", "b0 'Code snippet is self-contained' (merged)"]),
    ("Solves the precise operation asked",
     "Whether the technique implemented matches the specific behaviour the question describes -- for "
     "instance a substring occurring immediately adjacent rather than anywhere nearby -- instead of "
     "a looser or stricter neighbour of that condition. Score 1.0 when the implemented check matches "
     "the stated condition exactly; 0.5 when it matches on the examples given but would diverge from "
     "the stated condition on other inputs; 0.0 when it visibly solves a different condition than the "
     "one described; NA when the question specifies no precise condition to match against.",
     "A", ["b0 'Solves the precise operation asked'"]),
    ("Fix targets the named failure",
     "Whether the proposed code, if applied, would actually eliminate the specific exception or "
     "symptom the question describes, rather than changing unrelated code that leaves the same "
     "problem reachable. Score 1.0 when the fix addresses the statement or condition responsible for "
     "the described problem; 0.5 when it is adjacent and would plausibly help but does not clearly "
     "resolve the exact point of failure; 0.0 when it leaves the erroring behaviour unchanged or "
     "targets a different symptom; NA when the question describes no error or failure.",
     "A", ["b1 'Fix targets the named exception'"]),
    ("Matches the asker's data shape",
     "Whether the operations are built for the structure the question actually shows -- the nesting "
     "depth of a dictionary, the column layout of a dataframe -- rather than a superficially similar "
     "structure that only shares a name. Score 1.0 when the code is written against the structure "
     "exactly as shown; 0.5 when it works for a simplified version and would need adaptation to reach "
     "the full case shown; 0.0 when it operates on a structure that does not match what is shown, so "
     "applying it as written would fail or silently do the wrong thing; NA when the question does not "
     "show enough of the structure to judge a match.",
     "A", ["b1 'Matches the asker's data shape'"]),
    ("Shown output matches the stated ask",
     "Whether, when the question supplies a concrete expected value or behaviour, the answer's own "
     "demonstrated output actually corresponds to it rather than merely looking plausible. Score 1.0 "
     "when the shown output demonstrably matches what the question specifies; 0.5 when output is shown "
     "but does not correspond directly, leaving the reader to translate it; 0.0 when a checkable claim "
     "is made about what the code produces and the shown output contradicts it or nothing is shown; "
     "NA when the question gives no concrete expected value to check against.",
     "A", ["b0 'Shown output matches question's example'", "b3 'Shown output matches what was asked'"]),
    ("Performance claim is substantiated",
     "Where the answer asserts its approach is faster, more efficient, or scales better, whether that "
     "claim is backed by a measurement or an argument tied to the operation actually performed, rather "
     "than asserted as a bare adjective. Score 1.0 when every such claim is backed by a reported "
     "timing or an explicit reason; 0.5 when the headline claim is backed but a secondary one is left "
     "bare; 0.0 when a speed or efficiency claim appears with no support at all; NA when the answer "
     "makes no comparative performance claim.",
     "A", ["b1 'Performance claims are substantiated'", "b0 'Efficiency claim is substantiated'"]),
    ("Complexity claim matches the code",
     "When the answer states or clearly implies how its approach scales with input size, whether that "
     "claim is consistent with what the posted code actually does to the input. Score 1.0 when every "
     "scaling claim matches the algorithm as written; 0.5 when the qualitative direction is right but "
     "the specific claim overstates or understates it; 0.0 when a stated scaling claim contradicts "
     "what the code actually does; NA when the answer makes no claim about scaling.",
     "A", ["b1 'Complexity claim matches the code'"]),

    # ========================= scope and conditions =========================
    ("Version or platform dependency disclosed",
     "When the proposed code depends on a particular library version, Python version, or platform "
     "behaviour -- a method only available in a recent release, semantics that changed between "
     "releases, functionality since deprecated -- whether the answer says so. Score 1.0 when the "
     "sensitive construct is flagged as such, with the version or current alternative named; 0.5 when "
     "a dependency is implied by an incidental remark, such as a version number in a benchmark, "
     "without being stated as a caveat; 0.0 when the code relies on version- or platform-specific "
     "behaviour with no acknowledgement; NA when nothing in the solution is version- or "
     "platform-sensitive.",
     "A", ["b1 'Version or platform dependency disclosed'", "b0 'Flags version-dependent syntax'",
           "b3 'Flags deprecated or version-bound usage'"]),
    ("Respects the implied version constraint",
     "Where the question's tags or body indicate a specific Python or library version, whether the "
     "answer's code and idioms are compatible with that version rather than silently assuming a "
     "different one. Score 1.0 when the solution is compatible with the indicated version throughout; "
     "0.5 when it is compatible but uses a construct whose availability under that version is not "
     "obvious and goes unremarked; 0.0 when it uses syntax or an API not available under the version "
     "the question indicates; NA when the question indicates no version and nothing in the solution "
     "is version-sensitive.",
     "A", ["b2 'Respects the implied version constraint'"]),
    ("States the boundary of applicability",
     "Whether the answer names a concrete condition under which its approach stops working or needs "
     "adapting, rather than presenting the fix as good in every situation. Score 1.0 when a specific "
     "limit or breaking condition is stated; 0.5 when the answer hints that a limit exists without "
     "saying what it is; 0.0 when a plausible limit is never mentioned though the situation calls for "
     "one; NA when the approach has no meaningful limits given what is asked.",
     "A", ["b3 'States the boundary of applicability'", "b0 'States method's range of applicability'"]),
    ("States assumptions the fix depends on",
     "Where a proposed fix only works under a condition the question does not guarantee -- a "
     "particular operating system, a resource not already held open elsewhere, a particular starting "
     "state of the data -- whether the answer states that condition. Score 1.0 when every such "
     "dependency is stated; 0.5 when the main dependency is stated but a secondary one is left "
     "implicit; 0.0 when the fix is asserted with no mention of a condition it silently requires; NA "
     "when the fix holds unconditionally given only what the question establishes.",
     "A", ["b2 'States assumptions the fix depends on'"]),

    # ===================== risk, pitfalls, edge cases =======================
    ("Names a pitfall of its own approach",
     "Whether the answer identifies a specific way its own proposal can go wrong, degrade, or "
     "surprise the reader -- an edge case, a performance cost, a fragile assumption -- rather than "
     "presenting the fix as unconditionally safe. Score 1.0 when a concrete pitfall specific to the "
     "approach is named; 0.5 when a vague disclaimer is given without saying what could actually go "
     "wrong; 0.0 when the approach has a visible limitation in the code itself and none is mentioned; "
     "NA when the approach has no notable pitfall to flag.",
     "A", ["b0 'Names a pitfall of its approach'", "b2 'Fix's failure conditions are flagged'"]),
    ("Data-loss or side-effect risk flagged",
     "Whether the answer names a concrete risk its approach could introduce -- losing data, a race "
     "condition, an unintended mutation, a security exposure -- rather than leaving it implicit. Score "
     "1.0 when a specific risk of this kind is named; 0.5 when a risk is gestured at without saying "
     "what could actually go wrong; 0.0 when the approach carries an evident risk of this kind and it "
     "is never mentioned; NA when the approach as posed carries no such risk.",
     "A", ["b3 'Data-loss or side-effect risk flagged'"]),
    ("Names recourse when the approach fails",
     "Where the proposed approach might not succeed on the first try -- a search, a heuristic, an "
     "environment-dependent step -- whether the answer says what to check or change if that happens. "
     "Score 1.0 when the answer states what to do if it does not immediately work; 0.5 when it "
     "acknowledges the approach could fail without saying what to do about it; 0.0 when an approach "
     "that could plausibly fail is given with no such guidance; NA when the approach is deterministic "
     "and cannot fail once correctly applied.",
     "A", ["b3 'Names recourse when approach fails'"]),
    ("Edge cases in the asker's data addressed",
     "Whether values or shapes the question's own data plainly admits -- a duplicate entry, a tie, a "
     "missing value, an empty group, a non-numeric input -- either do not break the proposed code or "
     "are named as unhandled. Score 1.0 when such a case is handled, or is named as not handled; 0.5 "
     "when it is handled inconsistently and the gap goes unmentioned; 0.0 when a case plainly present "
     "in the question's own data would break the code and the answer neither handles it nor says so; "
     "NA when the question's data admits no such case.",
     "A", ["b3 'Edge cases in asker's data addressed'", "b1 'Asker's visible edge cases covered'"]),

    # ==================== engagement with the question ======================
    ("Engages the asker's actual code",
     "Whether the answer refers to what the asker specifically wrote -- a named variable, a specific "
     "line, the particular library call they chose -- rather than substituting a generic rewrite that "
     "could answer any similarly-titled question. Score 1.0 when the answer is anchored throughout to "
     "the asker's own names, structure, or specific line; 0.5 when it engages the posted code for part "
     "of the answer but drifts into an unrelated generic example; 0.0 when it ignores the posted code "
     "entirely and supplies an unrelated example; NA when the question poses no code of its own.",
     "A", ["b2 'Engages the asker's actual code'", "b0 'Uses asker's own data as given'",
           "b3 'Reuses the asker's own names'"]),
    ("Corrects a misconception in the question",
     "Where the question's code or description shows the asker believes something false about how the "
     "language or a library behaves, whether the answer names and corrects that belief rather than "
     "only working around it. Score 1.0 when the visible false belief is identified and corrected; 0.5 "
     "when the fix silently routes around the misconception without naming it; 0.0 when the answer's "
     "own content endorses or repeats the same false belief; NA when the question shows no false "
     "belief, only a missing technique.",
     "A", ["b2 'Corrects a misconception in the question'", "b1 'Implicit misconception corrected'"]),
    ("Reframes a misdirected approach",
     "Where the question's literal ask is a workaround for a different underlying goal a more direct "
     "route would serve better, whether the answer says so and points at the underlying goal. Score "
     "1.0 when the answer names the underlying goal and offers the more direct route; 0.5 when it "
     "supplies the more direct route but never says why the literal request was the wrong one to "
     "chase; 0.0 when it solves only the literal, misdirected request with no acknowledgement that a "
     "more direct route exists; NA when the literal request already is the direct route.",
     "A", ["b2 'Reframes a misdirected approach'"]),
    ("Resolves the ambiguous ask before answering",
     "Where the question admits more than one reasonable reading, whether the answer names which "
     "reading it is answering or asks a concrete disambiguating question, rather than silently "
     "guessing. Score 1.0 when the ambiguity is named and a reading is chosen or asked for explicitly; "
     "0.5 when a reading is picked silently, for instance by building around one specific data shape, "
     "without flagging that another exists; 0.0 when the answer addresses a reading the question does "
     "not clearly support and never flags the mismatch; NA when the question has only one plausible "
     "reading.",
     "A", ["b3 'Resolves ambiguous ask before answering'", "b2 'Declares reading of ambiguous question'"]),
    ("Pitched to the asker's evident level",
     "Whether explanation is tuned to what the question shows the asker already knows: library calls, "
     "error messages, and terms are glossed when the question's own code shows the asker would not "
     "recognise them, and left unglossed when the question already uses them correctly. Score 1.0 when "
     "the pitch matches the asker's evident level throughout; 0.5 when there is one local mismatch, "
     "such as a single unglossed term the question gives no evidence the asker knows; 0.0 when the "
     "pitch is badly mismatched, for instance drowning a clearly novice question in unexplained "
     "jargon, or re-teaching a library the question shows fluent use of; NA when the question gives no "
     "cue about the asker's level.",
     "A", ["b2 'Pitched to the asker's evident level'", "b1 'Pitched to the asker's level'"]),
    ("Critique of the original is substantive",
     "Whether, when the answer characterises the asker's original approach negatively, it gives a "
     "concrete technical reason rather than a bare aesthetic judgment. Score 1.0 when every such "
     "critique is paired with a specific technical reason; 0.5 when a reason is gestured at but not "
     "made concrete; 0.0 when the critique is purely aesthetic -- calling the code messy or overly "
     "clever -- with no technical grounding; NA when the answer offers no critique of the original.",
     "A", ["b0 'Critique of original code is substantive'"]),

    # ==================== craft, alternatives, handover =====================
    ("Idiomatic facility matches the structure",
     "Where the library in use provides a native, vectorised, or purpose-built mechanism for the "
     "operation needed, whether the answer reaches for it rather than reimplementing the same result "
     "by hand. Score 1.0 when the answer uses the library's native facility, or explains why none "
     "fits; 0.5 when it uses a workable but heavier-than-necessary construction, such as a manual loop "
     "where a comparably simple vectorised call exists; 0.0 when it hand-rolls something the library "
     "directly supports with no indication a native option exists; NA when the task has no natural "
     "library-native form.",
     "A", ["b1 'Idiomatic facility matches the structure'", "b3 'Names the proper tool, not workaround'"]),
    ("Alternatives compared with a stated reason",
     "Where the answer presents more than one way to solve the problem, whether it says why one is "
     "preferable in this situation rather than only listing options side by side. Score 1.0 when a "
     "preference is stated and tied to a reason relevant to the asker's situation; 0.5 when a "
     "preference is implied by ordering or emphasis but never justified; 0.0 when multiple options are "
     "listed with no indication of which to choose or why; NA when the answer offers only one "
     "approach.",
     "A", ["b2 'Alternatives compared with a stated reason'", "b3 'Trade-off against an alternative named'"]),
    ("Marks fix versus workaround",
     "Whether the answer states, accurately, that what it offers is a durable fix or a stopgap, rather "
     "than leaving the reader to guess which they are getting. Score 1.0 when the status is stated and "
     "matches what the code actually does; 0.5 when the status is implied by tone but never stated "
     "outright; 0.0 when a stopgap is presented with the certainty of a settled fix, or the reverse; "
     "NA when the question asks for a single concrete action where the distinction does not apply.",
     "A", ["b3 'Marks fix vs. workaround'"]),
    ("Fix is scoped to the fault",
     "Whether the change proposed alters only the part of the code that is actually broken, leaving "
     "the surrounding logic the asker already had working untouched. Score 1.0 when the change is "
     "scoped to the faulty part and preserves the rest of the asker's working logic; 0.5 when the fix "
     "is correct but folded into a broader rewrite that also changes correct behaviour without "
     "flagging it; 0.0 when applying the fix would silently drop or alter behaviour the asker's code "
     "already had working, with no note; NA when the question shows no prior working code.",
     "A", ["b1 'Fix scoped to the bug only'"]),
    ("Generalises past the literal example values",
     "Whether the offered solution is parameterised to the asker's actual data rather than hardcoded "
     "to a value that only happens to fit the posted example. Score 1.0 when the solution scales with "
     "the real data, such as its size or column set, instead of a number lifted from the example; 0.5 "
     "when the core logic generalises but one supporting detail stays hardcoded; 0.0 when the solution "
     "silently depends on a value from the example that would not hold on the asker's actual data; NA "
     "when the question concerns a single fixed value with nothing to generalise over.",
     "A", ["b3 'Generalizes past literal example values'"]),
    ("Generalises the fix beyond this instance",
     "Whether the answer names the general principle or pattern behind the specific fix -- why this "
     "class of problem arises -- so the technique transfers to similar cases the asker has not yet "
     "hit. Score 1.0 when a transferable principle or pattern is stated explicitly; 0.5 when a general "
     "note is gestured at, such as 'this happens a lot', but never named; 0.0 when the fix is "
     "presented as tied only to this exact code with nothing the reader could carry elsewhere; NA when "
     "the problem is a one-off, such as a typo or a missing import.",
     "A", ["b2 'Generalizes the fix beyond this instance'"]),
    ("Names the technique for further reference",
     "Whether the answer gives the reader a name or term for the technique it uses -- a pattern, a "
     "module, a concept -- that the reader could search on, rather than only demonstrating it silently "
     "in code. Score 1.0 when the technique is named explicitly in prose, separate from the code; 0.5 "
     "when a name appears only inside a code comment or identifier and never in the surrounding prose; "
     "0.0 when a specific, nameable technique is used with no name given anywhere; NA when the content "
     "has no discrete technique to name, for instance a one-character syntax correction.",
     "A", ["b2 'Names the technique for further reference'"]),
    ("Documentation link is usable in context",
     "Where the answer points at external documentation or a source, whether it says what the reader "
     "will find there and why it bears on this problem, rather than dropping a bare link. Score 1.0 "
     "when the reference is previewed and its relevance to the current problem stated; 0.5 when it is "
     "present with only a generic label, such as the bare function name, and no note on relevance; 0.0 "
     "when a load-bearing claim rests on a reference with no indication of what it supports; NA when "
     "the answer points at no external source.",
     "A", ["b2 'Documentation link is usable in context'", "b0 'Nontrivial API call is attributed'"]),
    ("Handover is actionable",
     "When the answer leaves work for the reader rather than supplying a complete drop-in fix, whether "
     "it hands over something the reader can act on directly -- a named next step, a specific line to "
     "change, a concrete alternative to try. Score 1.0 when the remaining step is concrete and the "
     "reader knows exactly what to do next; 0.5 when a direction is given but underdetermined, such as "
     "naming a technique without saying where to apply it; 0.0 when the answer trails off or gestures "
     "at unspecified other ways with nothing actionable; NA when the answer is already a complete, "
     "self-contained fix with nothing left over.",
     "A", ["b2 'Handover is actionable'", "b0 'Reader can act without further help'"]),
    ("Confidence matches the solution's strength",
     "Whether the certainty with which the answer states its solution matches how solid the reasoning "
     "behind it actually is: a guess is flagged as a guess, an untested suggestion is marked untested, "
     "and a verified fix is stated plainly. Score 1.0 when stated confidence tracks the strength of "
     "the underlying reasoning throughout; 0.5 when one claim is under- or over-stated relative to its "
     "actual strength; 0.0 when a guess or partial fix is presented with the same certainty as a "
     "verified one; NA when the answer makes no claim of confidence to check.",
     "A", ["b3 'Confidence matches solution's strength'"]),

    # ============ Track B: DECLARED SURFACE PROBES (spurious pole) ==========
    ("Contains a code block",
     "A surface check of whether the answer contains a fenced or indented block of code at all, "
     "independent of whether that code is correct, relevant, or runnable. Score 1.0 when at least one "
     "code block is present; 0.5 when code appears only as short inline spans within prose; 0.0 when "
     "the answer contains no code in any form; NA never applies -- every answer either shows code or "
     "does not.",
     "B", ["charge-specified surface calibration probe"]),
    ("Contains an external hyperlink",
     "A surface check of whether the answer text includes a URL or Markdown link to any external page, "
     "independent of whether the destination is relevant, authoritative, or ever needed to follow the "
     "answer's logic. Score 1.0 when a properly formatted Markdown link is present; 0.5 when only a "
     "bare unformatted URL appears; 0.0 when no link of any kind appears; NA when the answer is a "
     "single inline expression with no surrounding prose in which a link could sit.",
     "B", ["b0/b1/b2/b3 external-link candidates (collapsed)"]),
    ("Shows an output block",
     "A surface check of whether the answer includes a block of output text -- a printed result, a "
     "REPL transcript, an Out cell, a timing line -- set apart from the code, independent of whether "
     "that output is correct or was ever produced by running the shown code. Score 1.0 when a distinct "
     "output block is present; 0.5 when result values appear only inline in prose rather than set off "
     "as a block; 0.0 when no output of any kind is shown; NA when the code shown produces no "
     "observable output, for instance a bare function definition with no call.",
     "B", ["b0 'Shows a labelled output block'", "b1 'Pastes raw console output'",
           "b2 'Includes a rendered output block'"]),
    ("Uses a numbered or bulleted list",
     "A surface check of whether the answer's prose, outside any code block, is organised with "
     "explicit list markers such as numbers or dashes, independent of whether the listed points are "
     "well organised or correct. Score 1.0 when a list structure is used; 0.5 when list markers appear "
     "inconsistently, such as only the first point being numbered; 0.0 when the prose runs as "
     "continuous paragraphs with no list markers; NA when the answer has no prose outside of code.",
     "B", ["b0/b3 'Uses a numbered or bulleted list' (collapsed)"]),
]


def main():
    out = Path(__file__).resolve().parent / "rubrics.jsonl"
    seen = set()
    lines = []
    n_a = n_b = 0
    for i, (name, desc, track, origin) in enumerate(BANK, start=1):
        assert name not in seen, f"duplicate criterion name: {name}"
        seen.add(name)
        prefix = "a" if track == "A" else "s"
        idx = (n_a + 1) if track == "A" else (n_b + 1)
        if track == "A":
            n_a += 1
        else:
            n_b += 1
        lines.append(json.dumps({
            "rubric_id": f"{prefix}{idx:02d}",
            "name": name,
            "description": desc,
            "track": track,
            "origin": origin,
            "gepa_revision": ("merged from the mined proposals listed in `origin`, "
                              "rephrased to a two-sided match test with a distinct "
                              "NA branch so it cannot be read off answer length or "
                              "formatting"),
        }, ensure_ascii=False))
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}  n={len(lines)}  trackA={n_a}  trackB={n_b}")
    lens = [len(json.loads(l)["description"]) for l in lines]
    print(f"description chars: min {min(lens)} median "
          f"{sorted(lens)[len(lens)//2]} max {max(lens)}")


if __name__ == "__main__":
    main()
