---
source_url: https://link.springer.com/article/10.1007/s11023-010-9209-8
title: Pablo Gervás - Computational Metrics for Narrative Quality
source_type: computational_creativity
fetched: 2026-05-09
---

# Pablo Gervás — Computational Metrics for Story Evaluation

Gervás (Universidad Complutense de Madrid) develops formal metrics for evaluating computer-generated narratives. The same metrics apply to human-written stories.

## Foundational Frame

Narrative generation involves a four-task feedback cycle:
1. **Inventing** content
2. **Organizing** content
3. **Interpreting** content
4. **Validating** content

Each task is a place where evaluation happens.

## Novelty Assessment

Novelty requires comparison against prior works. The Gervás team operationalizes novelty in the folktale domain along four basic narrative elements:
- **Events** — actions that happen
- **Characters** — agents who act
- **Props** — objects manipulated
- **Scenarios** — settings and situations

For each element, novelty is measured by similarity (or dissimilarity) to elements in the reference corpus. Lower similarity = higher novelty.

## Quality Dimensions

Drawing on tested fitness functions, Gervás identifies multiple quality axes:

### Narrative Coherence
- Causal continuity between events
- Character consistency across actions
- Settings stable unless explicitly changed

### Tension
- Buildup of conflict
- Strategic withholding
- Stakes calibrated to reader engagement

### Empathy
- Reader connection to character experience
- Perspective access (whose interiority is rendered)
- Emotional plausibility

### Discourse Organization
- Plot ordering (chronological vs. artistic)
- Multi-plot interleaving
- Subplot integration

### Information Distribution
- When and how information is revealed
- Mystery vs. dramatic irony management

## Multi-Plot Story Assessment

Gervás's later work (Assessing MultiPlot Stories, 2021) provides metrics specifically for multi-plot narratives:
- **Plot count** and word distribution
- **Plot intersection density** (how often plots converge)
- **Character overlap** between plots
- **Causal connections** between plots
- **Thematic resonance** across plots

A multi-plot story is well-constructed when plots interweave causally and thematically without diluting any single plot's drive.

## Combining Novelty and Appropriateness

Following Margaret Boden's classic definition, creativity = novelty + appropriateness. Gervás emphasizes that **either alone is insufficient**:
- Pure novelty without appropriateness = nonsense
- Pure appropriateness without novelty = cliché

The challenge is finding outputs at the productive intersection.

## Evaluation-Driven Rejection

A key methodological move: generate broadly, evaluate strictly, reject heavily. The conceptual space of stories is explored productively only when most generations are rejected. This mirrors what writers do — most ideas are rejected; few become drafts.

## Implications for Story Rubrics

A short story (computer-generated or human) can be assessed on:
- **Novelty** of events / characters / props / scenarios vs. prior works in the genre
- **Coherence** (causal, character, spatiotemporal)
- **Tension** (escalation, withholding, stakes)
- **Empathy capacity** (does the reader feel through the characters?)
- **Discourse organization** (plot order, multi-plot management)
- **Information distribution** (timing of reveals)
- **Novelty + appropriateness intersection** (creative, not just new or just well-made)

## Significance

Gervás's program demonstrates that fiction quality is **measurable** along multiple dimensions, and that these dimensions can be operationalized with concrete metrics. This makes computational creativity research a useful source of evaluative criteria for human-authored work.
