# Rubric taxonomy + two-axis classification — current setup

*Written 2026-05-14. Reference doc for the analysis setup as it stands.*

## 1. The corpus and the rubric tree

We extracted evaluation rubrics from expert sources across **11 tasks**: code-review, creative-writing, grant-funding, humor, legal-outcome-prediction, math-stackexchange, news-homepages, notice-and-comment, patents, peer-review, press-releases.

- **~38,300 source pages** (expert guides, textbooks, standards, blogs, papers — HTML + PDF), roughly balanced across the 11 tasks.
- gpt-5-mini extracted structured rubrics from each page → **~72,000 unique rubrics** (a rubric = `{name, description, guidance}` + provenance `key = task::source_dir::source_file::rubric_idx`).

The rubrics are then organized into a hierarchy, per (task, specificity-bucket) cell. There are 3 specificity buckets — **general / specific / hyper_specific** — giving 31 cells (creative-writing has only `general`).

### Levels of the tree (bottom to top)

| Level | What it is | How produced | Count (approx) |
|---|---|---|---|
| **Leaf** | one raw extracted rubric instance | gpt-5-mini extraction | ~72K unique |
| **Cluster** (= R1 child) | one *deduped* rubric concept, with a medoid name + description | complete-linkage dedup clustering on embedding similarity | ~53.7K across all cells; ~24.6K in the general bucket |
| **R1 parent** | a group of related clusters under one concept | `build_hierarchy.py` (gpt-5, anchor-batched) | — |
| **R1-refined** | R1 parents cleaned: misfit children dropped, within-parent duplicate children merged | `refine_parent.py` | — |
| **R2 merged_group** | redundant R1 parents merged into one canonical concept | `meta_merge.py` (round 2) | ~30–500 per task (general bucket) |
| **R2 grandparent** | a parent-of-parents grouping | `meta_merge.py` (round 2) | — |
| **R3** | further cross-batch merges R2 missed | `meta_merge.py` on R2 output | — |

Key facts about the levels:
- A **cluster** is the dedup unit. Each carries `medoid_name` + `medoid_description` and a list of near-duplicate member rubrics. Average cluster size ≈ 1.36 (dedup is modest — most clusters are singletons or pairs).
- A **merged_group** is a mid-level concept that bundles many clusters (10s–100s of leaves). It carries `merged_name`, `merged_description`, `all_leaves`.
- Provenance is preserved end to end: every leaf's `key` decomposes to a source page, joinable to `pages_df` (source type, wave, and — once Step 0 lands — publication year).

### Which level analyses run at

- **Taxonomy-shape analyses** (concentration / Zipf / entropy, cross-task concept overlap) run at the **merged_group** level — they are about the shape of the taxonomy.
- **Per-criterion classification** (the two-axis classification below) runs at the **cluster** level — clusters are the deduped units and carry enough description to classify; merged_groups are too coarse.

## 2. The classification: four ordinal axes

Each rubric cluster is classified by an LLM-as-judge (gpt-5-mini) on four ordinal 1–4 axes, in one call. **Articulability is the primary axis.** `reasoning_depth` and `indeterminacy` are *diagnostic components* — assessed first so the articulability call is decompose-then-aggregate rather than a single holistic guess. `surface_vs_substance` is an independent parallel axis. Across all axes, **higher = harder / more resistant to capture**.

The classification runs at the **cluster** level (the deduped rubric concepts), not the merged_group or raw-leaf level.

### 2.1 reasoning_depth (1–4) — diagnostic component

*Assuming the criterion is perfectly clear, how much inference does applying it take?*

| Value | Name | Meaning | Example |
|---|---|---|---|
| 1 | Mechanical | Count, match, look up, compare values. No interpretation. | "Abstract under 250 words." |
| 2 | Shallow-semantic | One interpretive step: classify by type, locate a section, judge a surface property. | "The register is formal." |
| 3 | Inference | Relate multiple parts, evaluate an argument, weigh evidence, synthesize across a document. | "The argument addresses the strongest counter-claim." |
| 4 | Deep / holistic | Sustained domain reasoning, or apprehending the whole work at once as a gestalt. | "The proof is elegant." |

### 2.2 indeterminacy (1–4) — diagnostic component

*Independent of how hard it is to apply — how fully does the criterion specify what it is asking?* **Mere threshold/degree vagueness ("how much is *enough*") is universal — every gradable criterion has it — and does NOT count as indeterminacy.** Levels 3–4 are reserved for *structural* under-specification.

| Value | Name | Meaning | Example |
|---|---|---|---|
| 1 | Fully specified | Agreed terms, no free parameters. A competent person knows exactly what is asked. | "Use Oxford commas." |
| 2 | Threshold-vague only | The core concept is clear; only "how much is enough" is open. The normal case for gradable criteria. | "Provide concrete detail." |
| 3 | Structural free parameter | The criterion is genuinely *different* depending on something it leaves open. | "The summary is audience-oriented." (which audience?) |
| 4 | Contested core concept | The central term has no agreed definition. | "Avoid stereotypes." "The piece has voice." |

### 2.3 surface_vs_substance (1–4) — independent parallel axis

*What does the criterion govern — form or content?*

| Value | Name | Meaning | Example |
|---|---|---|---|
| 1 | Pure surface / form | Syntax, formatting, mechanics, layout. How something is presented. | "Use Oxford commas." |
| 2 | Mostly surface, some substance | A form rule whose point is to constrain meaning. | "Headlines title-case AND accurately reflect the article." |
| 3 | Mostly substance, some form | A content rule with formal scaffolding. | "Methods section reports a power analysis." |
| 4 | Pure substance | Meaning, validity, truth, soundness. What is actually claimed or meant. | "The thesis is falsifiable." |

### 2.4 articulability (1–4) — THE PRIMARY AXIS

*Where does the criterion sit on the capture spectrum?* A nested spectrum — the score is the highest level at which it still resists capture. Articulability is largely a **function of the two diagnostic components**: deep reasoning (high `reasoning_depth`) OR structural under-specification (high `indeterminacy`) pushes it up.

| Value | Name | Meaning | Typical components |
|---|---|---|---|
| 1 | Code-checkable | A program / regex / lint / test resolves it deterministically. | reasoning_depth 1, indeterminacy 1 |
| 2 | Language-tacit | Code cannot, but an LLM-as-judge given the rubric *can* apply it reliably. LLM-judge handles clarity, tone, register, structure, completeness. | reasoning_depth 2–3, indeterminacy 1–2 |
| 3 | Defensible judgment | Neither code nor an LLM-judge fully captures it; briefed human raters agree above chance. Needs domain expertise the LLM lacks, OR has a structural free parameter. | reasoning_depth 4, OR indeterminacy 3 |
| 4 | Fully tacit / taste | Even briefed expert raters disagree; little fact of the matter. | reasoning_depth 4 and/or indeterminacy 4 |

## 3. What we are going for

The articulability axis maps directly onto the project's gap framework, `Outcome = f(Verifiable) + g(Articulable) + h(Taste)`:

- **Level 1** ≈ what a formal program can capture — the verifiable component `f`. AUC of a code implementation = **A**.
- **Levels 2–3** ≈ what an LLM-as-judge adds on top — the articulable component `g`. AUC of an LLM-judge = **B**.
- **Level 4** ≈ the residual even an LLM-judge cannot reach — the taste component `h`.

So the per-task **articulability distribution** is a descriptive backbone for the paper: it says, for each task, what fraction of the evaluation criteria are mechanically checkable vs. LLM-judge-able vs. genuinely tacit. The "articulability gap" the paper measures — (C − B), what even a full-knowledge LLM-judge misses — is the level-4 mass plus whatever level-3 the LLM-judge falls short on.

The surface-vs-substance axis is the orthogonal cut. Crossing it with articulability gives a 2D map per task. The hypothesis it lets us test: **the articulability gap concentrates in substance rubrics** — surface/form rules are mostly thin and checkable, and the tacit residual lives in rules about meaning and validity. If true, "the gap is a substance gap, not a form gap" is a clean finding.

The two diagnostic components — `reasoning_depth` and `indeterminacy` — give the *why* behind a task's articulability profile. Two tasks can have the same articulability distribution for different reasons: one task's tacitness may be driven by deep reasoning (high `reasoning_depth` — the criteria need sustained inference), another's by under-specification (high `indeterminacy` — the criteria invoke contested concepts). These point to different interventions, so the components are worth keeping even though articulability is the headline.

Concretely, the deliverable from this classification is, per task:
- the distribution over articulability 1–4 (how codeable / LLM-able / tacit the task's evaluation is)
- the distributions over reasoning_depth and indeterminacy 1–4 (the *why* behind the articulability profile)
- the distribution over surface-vs-substance 1–4
- the joint articulability × surface 2D distribution
- per-merged_group, the *spread* of its clusters across articulability — capturing within-concept heterogeneity

These per-task profiles are what the taxonomy work hands off to the downstream gap measurement (training code verifiers, LLM-judges, and dense models, and comparing AUCs A ≤ B ≤ C).
