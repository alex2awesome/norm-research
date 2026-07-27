# Noun/verb thickness — a richer thickness model than scalar scores

*Started 2026-05-14. Migrated from Appendix A of the analyses plan (`~/.claude/plans/mellow-knitting-quilt.md`).*

## Conceptual move

The original "input thickness" idea conceptualized rubrics like inputs to a function — with the noun (what's being judged) tagged as thick or thin. The boundary of "what counts as a function" was hazy. The refinement: think of rubric application as a chain of operations:

```
input_noun → verb → intermediate_noun → verb → … → final_noun
```

**Both nouns and verbs can be thickly or thinly described.**

- **Nouns** = things being operated on (inputs, intermediates, outputs)
- **Verbs** = operations transforming nouns into other nouns

This is **not about syntax** — there are latent / fundamental states that the chain reveals. A rubric like "the piece has voice" syntactically looks like one verb on one noun, but conceptually the verb itself ("judge for voice") is irreducible thickness.

## Examples (where thickness lives in the chain)

| Rubric | Chain | Thickness location |
|---|---|---|
| "Use Oxford commas" | text → regex match | thin everywhere |
| "Methods section reports a power analysis" | full doc (thick noun) → locate methods (thin verb) → methods text → check for power analysis (thin verb) → bool | thickness in the input noun only |
| "Argument addresses strongest counter-claim" | full doc → identify argument → identify counter-claims → check engagement (medium verb) → bool | thickness in middle verb (reasoning required) |
| "The piece has voice" | full doc → judge for voice (thick verb) → judgment | thickness in the verb (irreducible) |

## Why a single "thickness" score is insufficient

"Voice" rubrics and "argument addresses methods over the whole document" rubrics could both score "thick" overall, but they fail differently:
- **Voice**: no procedure exists; the verb itself resists decomposition
- **Whole-document arguments**: procedure exists, but it operates on a thick input

These have different implications for what kind of model could capture them. A code-generation pipeline could produce useful tools for the second (find-and-extract) but cannot help with the first.

## Thickness anchors

**Noun thickness** (1–4):
- 1 = bounded / scalar / well-typed (e.g., word count, citation list)
- 2 = a small bounded chunk (e.g., a sentence, a citation, a header)
- 3 = a structured artifact (e.g., a section, an argument, a methods description)
- 4 = whole document or context-dependent latent state (e.g., voice, narrative arc, reader-experience)

**Verb thickness** (1–4):
- 1 = mechanical / procedural (e.g., count, lookup, regex match)
- 2 = lexical / shallow-semantic (e.g., classify by type, locate a section)
- 3 = reasoning / inference (e.g., judge if argument addresses counter-claim)
- 4 = irreducible holistic judgment (e.g., judge if piece has voice)

## Connection to original variance plan

This extends §1d of `notes/2026-05-11__rubric-variance-analysis-plan.md`, which already proposed thin/thick tagging for input features:

> Per rubric, classifier also extracts:
> - List of input feature(s) the rubric depends on (free-text)
> - Per-input binary thin/thick (`word count` → thin; `narrative arc` → thick)

The new move adds **verb thickness** alongside noun thickness, and treats them as a chain rather than independent tags.

## Connection to V/A nesting

This complements the realization (from `project_verifiability_explainability_gaps`) that Articulability and Verifiability are nested points on one spectrum, not parallel axes:

- **Articulability score** answers: how operationalizable is the rule overall?
- **Noun/verb chain** answers: where in the operation does the thickness live?

They are different views of the same conceptual structure. Together they're more expressive than either alone.

## Operational implication

The noun/verb chain is more expressive than 1–4 scalar scores, but harder to aggregate. The plan:
1. Elicit chains free-form (variable-length JSON) per merged_group via gpt-5-mini.
2. Cluster chain shapes into archetypes for cross-rubric comparison.
3. Report per-task archetype prevalence as the cross-cutting summary.

## Open conceptual questions

- Is the noun/verb distinction itself thin or thick? (Probably thick — the chain decomposition is itself a judgment call.)
- Does chain length correlate with task properties? (E.g., taste tasks may have 1-step chains with thick verbs; procedural tasks may have many steps with thin verbs.)
- Could the chain structure be used as input to a code-generation pipeline? (Each verb becomes a candidate function to write; thin verbs write cleanly, thick verbs need LLM-judge or are taste residual.)
- Is "noun" really one thing, or is there a distinction between *input nouns* (what's read), *intermediate nouns* (what's computed), and *output nouns* (the final judgment)? They might have different thickness profiles.

## Where this fits in the broader research

The articulability-gap paper measures (C - B): the residual that even an LLM-judge with full world knowledge cannot reach. Noun/verb decomposition is a hypothesis-generator for *why* a rubric falls in that residual:
- Thick verb → no procedure, irreducible (true tacit)
- Thick noun + thin verb → procedure exists but applies to a hard-to-bound input (long-context limit)
- Both thin → in (B - A) zone (LLM-judge can do it, code can't yet)

Different failure modes, potentially different fixes.

## Procedural vs. predicate rubrics — a limit of the noun/verb frame (added 2026-05-14)

The noun/verb chain assumes evaluation is a **computation**: a function transforming an input through operations into a verdict. That model fits **procedural** rubrics — "the conclusion follows from the methods" genuinely decomposes into retrieve-methods → retrieve-conclusion → check-entailment. The chain there is *discovered* structure.

But many rubrics — probably most — are **predicates**, not procedures: "avoid stereotypes", "ensure accuracy", "use plain language", "the thesis must be falsifiable". A predicate is an *attribute* asserted to hold (or not hold) of the work. Predicates do not decompose into steps. Forcing one into a chain produces one of two failure modes:

1. **Collapse** — input → [detect the attribute] → verdict. One thick verb. Tells you nothing the articulability scalar didn't.
2. **Fabrication** — the LLM invents a plausible procedure ("locate human-referential language → classify for harm → weigh context"). That procedure is not discovered; it is *one possible operationalization*. The chain becomes an artifact of the procedural frame, not a fact about the rubric.

So the chain adds signal for procedural rubrics and adds **noise** for predicate rubrics.

**A third thickness type the noun/verb frame cannot represent: concept-definitional thickness.** The difficulty of "avoid stereotypes" is not in a noun or a verb — it is that the predicate *term* ("stereotype") refers to a contested, underdefined concept. Same for "voice", "elegance", "newsworthy", "substantive". This may be the most important thickness type for the articulability gap, and the chain has no slot for it.

**Implication.** The noun/verb chain is a valid model for the procedural subset only — not a universal framework. The cheaper, more robust replacement is a categorical classification:
- **procedural vs. predicate** (does the rubric admit a genuine step-decomposition?)
- for predicates, *where the thickness sits*: input-holism / operation-irreducibility / concept-contestedness

Three categorical tags, not a variable-length chain-extraction pipeline. Answers the same "what could be coded" question with far less machinery. The articulability scalar already handles predicates natively because it asks "how operationalizable" without assuming a procedure exists.

## Related work — procedural vs. predicate (lit search 2026-05-14)

The procedural-vs-predicate distinction is not one named thing, but it re-discovers several well-established distinctions across fields. None of them is in NLP-evaluation specifically — the rubric / LLM-judge literature uses "analytic vs. holistic" (a decomposition-granularity axis, not a procedural-vs-predicate axis) and "verifiable criteria" checklists (which implicitly assume predicates can be made checkable). So there is a genuine positioning gap: no one in NLP eval explicitly theorizes which rubrics admit a procedural decomposition vs. which are irreducible predicates.

Closest established distinctions, by relevance:

1. **Procedural vs. declarative knowledge** — cognitive science / AI knowledge representation. Declarative = "knowing that" (facts, propositions), consciously verbalizable; procedural = "knowing how" (skills), often implicit and *resists explanation*. The procedural/declarative split carries "control information" — declarative knowledge states what is true but not how to act on it. Maps directly: "avoid stereotypes" is declarative (a proposition about a desired state); "count words, check < 80" is procedural. Crucially, the cog-sci framing already notes procedural knowledge "may resist explanation" — i.e. articulability is itself one of the dividing lines.
   - Procedural knowledge (overview): https://en.wikipedia.org/wiki/Procedural_knowledge

2. **Construct validity vs. operational definition** — psychometrics. Cronbach & Meehl (1955), "Construct Validity in Psychological Tests," *Psychological Bulletin* 52:281–302. "Construct validation is involved whenever a test is to be interpreted as a measure of some attribute which is not operationally defined." This is exactly the "avoid stereotypes" problem — the rubric names a *construct*; the procedure to detect it is underdetermined, and the gap between construct and operationalization is the validity question. Most relevant single framing for this project: the articulability gap is partly a construct-validity gap.
   - Cronbach & Meehl 1955 (PDF): https://meehl.umn.edu/sites/meehl.umn.edu/files/files/036constructvalidityidx.pdf
   - Classics in History of Psychology mirror: https://psychclassics.yorku.ca/Cronbach/construct.htm

3. **Rules vs. standards** — legal theory / law-and-economics. Kaplow (1992), "Rules Versus Standards: An Economic Analysis," *Duke Law Journal* 42(3):557–629. Distinguishes by whether legal content is given *before* or *after* individuals act: rules are determinate ex ante, standards acquire content ex post via judgment. Related but not identical — it is about determinacy/timing, not procedural decomposability.
   - Kaplow 1992 (PDF): https://scholarship.law.duke.edu/cgi/viewcontent.cgi?article=3207&context=dlj

4. **Property-based vs. example-based testing** — software engineering (QuickCheck lineage). A property is a predicate that must hold for all generated inputs; an example test is a concrete procedure with assertions. Note the SE framing literally calls the test function a "predicate." Direct analog in the testing domain — and our rubrics-as-tests framing makes this the closest operational cousin.
   - In praise of property-based testing: https://increment.com/testing/in-praise-of-property-based-testing/

5. **Analytic vs. holistic rubrics** — education / NLP evaluation. Analytic rubrics score criterion-by-criterion; holistic rubrics give one overall judgment. This is a *granularity* axis, not a procedural-vs-predicate axis, but it is the closest thing the rubric literature has. Recent LLM-as-judge rubric work:
   - LLM-Rubric (multidimensional calibrated evaluation): https://arxiv.org/html/2501.00274v1
   - Autorubric (unified rubric-based LLM evaluation): https://arxiv.org/html/2603.00077
   - Learning to Judge (LLMs designing + applying rubrics): https://arxiv.org/html/2602.08672v1

6. **Categorical vs. hypothetical imperatives** — Kant. Already invoked in conversation; a hypothetical imperative ("if you want X, do Y") is closer to predicate-style normativity than to a procedure.

**Positioning takeaway:** the procedural-vs-predicate cut is well-grounded in cog-sci, psychometrics, SE, and legal theory, but absent from NLP evaluation. Framing the articulability gap partly as a *construct-validity* gap (Cronbach & Meehl) is the strongest available hook, and "no one in NLP eval theorizes which rubrics admit procedural decomposition" is a real gap the paper could claim.


## References (auto-verified BibTeX, 2026-06-15)

> Extracted from this document and web-verified + independently audited by an automated fact-check pass (search → fetch → resolvable id; attributed claim checked against the located paper). 5 entries. Real located works; not hand-checked. See "needs manual review" for 0 contradicted-claim and 0 unlocatable/rejected items.

```bibtex
@article{cronbach1955construct,
  title={Construct Validity in Psychological Tests},
  author={Lee J. Cronbach and Paul E. Meehl},
  journal={Psychological Bulletin},
  volume={52},
  number={4},
  pages={281--302},
  year={1955},
  doi={10.1037/h0040957}
}

@article{hashemi2024llmrubric,
  author  = {Helia Hashemi and Jason Eisner and Corby Rosset and Benjamin Van Durme and Chris Kedzie},
  title   = {LLM-Rubric: A Multidimensional, Calibrated Approach to Automated Evaluation of Natural Language Texts},
  journal = {arXiv preprint arXiv:2501.00274},
  year    = {2024},
  eprint  = {2501.00274},
  archivePrefix = {arXiv}
}

@article{kaplow1992rules,
  author  = {Louis Kaplow},
  title   = {Rules Versus Standards: An Economic Analysis},
  journal = {Duke Law Journal},
  volume  = {42},
  number  = {3},
  pages   = {557--629},
  year    = {1992},
  doi     = {10.2307/1372840}
}

@article{rao2026autorubric,
  author  = {Delip Rao and Chris Callison-Burch},
  title   = {Autorubric: Unifying Rubric-based LLM Evaluation},
  journal = {arXiv preprint arXiv:2603.00077},
  year    = {2026},
  eprint  = {2603.00077},
  archivePrefix = {arXiv}
}

@article{siro2026learning,
  author  = {Clemencia Siro and Pourya Aliannejadi and Mohammad Aliannejadi},
  title   = {Learning to Judge: LLMs Designing and Applying Evaluation Rubrics},
  journal = {arXiv preprint arXiv:2602.08672},
  year    = {2026},
  eprint  = {2602.08672},
  archivePrefix = {arXiv}
}

```

