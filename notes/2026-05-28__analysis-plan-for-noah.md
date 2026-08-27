# Analysis plan for Noah meeting (May 28 2026)

## Conceptual diagram to draw

The expanded VAT hierarchy. Each measurable layer further bisects on
**group consensus** vs **personal/individual** dimension:

```
        verifiable ──► articulable ──► inarticulable ──► noise
                            │                 │
                            ├── group         ├── group
                            │   articulable   │   inarticulable
                            │   (community-   │   (experts
                            │    consensus    │    converge but
                            │    rubric)      │    can't state)
                            │                 │
                            └── personalized  └── personal
                                articulable       inarticulable
                                (individual       (pure taste,
                                 taste, can       no convergence)
                                 state)
```

- **Verifiable** = code-checkable (typography, citation presence, length)
- **Group articulable** = R2 rubrics, LLM-judgeable, inter-rater reliability high
- **Personalized articulable** = "I have a rubric but it's mine, e.g., 'no semicolons'"
- **Group inarticulable** = expert taste convergence ("I know it when I see it" + others agree)
- **Personal inarticulable** = pure individual taste residue
- **Noise** = label measurement error / random variation

Operationalize each via:
- Verifiable = code-AUC
- Group articulable = Claude-rubric AUC; high inter-judge agreement (judge × dp σ² low)
- Personalized articulable = aspects where individual judges differ but each is self-consistent
- Group inarticulable = dense ceiling AUC − judge AUC (the "tacit" layer); replicable across human raters
- Personal inarticulable = unreproducible across raters
- Noise = irreducible label error

This sits alongside the 4b/4c lenses (articulability × type × thinking) and provides the
master interpretive frame for Noah.



By morning we'll have ~1000 dps × full aspects × Qwen-thinking scored across 9 tasks,
plus existing Claude (~40-770 dps × all aspects), Llama-BF16 (5000 dps for most tasks),
and code (5000 dps × ~250 aspects for 8 tasks; patents fresh).

## 1. Articulable AUC ceiling vs dense upper bound vs code-only

**The headline:**
- **Dense AUC** = Llama-8B reward model trained on full text → outcome (the [[project_dense_model_sweeps]] result). The "everything goes" ceiling — what's possible if you don't try to articulate features.
- **Judge AUC** = sum/logistic of aspect-judge scores → outcome. The "articulable + LLM-judged" layer.
- **Code AUC** = sum/logistic of code-only scores → outcome. The "fully programmatic" layer.

**Plot**: per task, bar chart of three AUCs. The **dense − judge** gap is the "language-tacit" component; **judge − code** gap is the "needs LLM" articulable layer.

This is the [[project_tacitness_two_layers]] decomposition — A (code) < B (LLM judge) < C (dense ceiling) < 1 (taste residue).

**Inputs we have**:
- Dense AUC: from existing per-task sweeps in `runs/{task}_sweep_llama8b/` subset_1p0/validation_metrics.csv
- Judge AUC: regress Qwen scores onto labels, LOO CV across 1000 dps
- Code AUC: same, but using `codegen_exec_results.jsonl`

**Modeling**: LASSO with cross-validated alpha on the 1000-dp panel per task. Outcome = `judgement` from datapoints.json.

## 2. Aspect correlation with labels × L1-L4 articulability levels

The 2-axis classification (Articulability 1-4 × Surface-vs-Substance 1-4) was done per
merged_group (R2). For each R2 (or R2_post canonical) aspect we have:
- `articulability` ∈ {1,2,3,4} from `outputs/analyses/merged_group_2axis.parquet`
- `outcome_correlation` = per-aspect ρ(score, label) — already computed in `claude_aspect_label_correlation.csv`

**Plots**:
- Scatter: x = articulability (1-4), y = |ρ(aspect, label)| per aspect. Color by task.
- Hypothesis: **higher articulability → lower ρ** (the most-code-like aspects matter least to outcome; the most-tacit are where the judgment counts).
- Histogram of articulability levels per task — which tasks live where on the spectrum.

This tests whether tacit aspects are actually carrying more signal — central to the VAT story.

## 3. LASSO + LLM-as-judge baselines

**Three predictors per task:**
- **LASSO on judge cells** (current main analysis): drop aspects with near-zero correlation; CV alpha
- **Logistic regression on code cells**: bigger n (5000 dps), but cruder features
- **"Synthetic judge" LLM**: prompt Claude/Qwen with the dp text + the 250 aspect scores, ask for binary judgment. Tests whether scores + LLM > scores + lasso.

**Variant on #3**: provide LLM with only the **aspect names** (no scores), let it score and predict. Tests whether the rubric structure carries any signal beyond the LLM's own reading.

## 4. G-theory decomposition

Requires multi-judge × multi-paraphrase × multi-dp cells.

**What we have:**
- Multi-judge: Claude, Qwen, Llama on overlapping cells (40-1000 dps depending on task)
- Multi-paraphrase: weak. p0 dominates; p1/p2 only on the early random samples (sparse)

**Realistic G-theory**:
- σ²(judge): Claude vs Qwen vs Llama variance — strong signal at n=40 panel
- σ²(dp): cross-dp variance — strong
- σ²(aspect): cross-aspect variance — strong
- σ²(paraphrase): blocked by sparse p1/p2 data — partial only

So a **3-facet G-study (judge × dp × aspect)** is doable. A 4th paraphrase facet is partial.

For a clean σ²(paraphrase) component, would need to run another wave on Claude's panel at p1/p2 — possibly tomorrow if budget allows.

## 4b. Thinking-token analysis (NEW — complements articulability)

For each Qwen response (1 dp × ~15-20 aspects), measure:
- **Thinking tokens** = `len(tokenizer.encode(text_between_<think>_and_</think>))`
- **Average per aspect** = thinking / N_aspects_in_prompt

Aggregates and hypotheses:
1. **Per task — mean thinking/cell**: hardest tasks need most reasoning. Predict patents/PR (long inputs) > NC/humor (short).
2. **Per bundle — mean thinking/cell**: which aspect-clusters demand most reasoning? Cross-ref with articulability (1-4) — predict articulability=4 (taste) bundles take more thinking than articulability=1 (code-like).
3. **Thinking × parse rate**: ρ(avg_thinking, parse_success) per prompt. Predict: very-long-thinking prompts hit budget cap → parse failures cluster there.
4. **Thinking × score** (when parsed): higher thinking → more 0.5 scores (uncertainty)? Or higher → more confident 0/1?

This adds a fourth layer to the VAT story:
- code (no thinking, articulability=1)
- LLM-judge with low thinking (articulability=2)
- LLM-judge with high thinking (articulability=3)
- Tacit residue (articulability=4) — beyond what even high-reasoning LLM captures

## 4c. Per-task aspect typology (NEW — complements articulability + thinking)

For each task, classify aspects into 5-8 natural conceptual types (e.g.,
syntactic / procedural / craft / audience-engagement / evidentiary / structural /
ethical / domain-specific). Types are discovered per-task because they vary
(creative_writing's "craft" ≠ peer_review's "methodological rigor").

### Method (cheap version)
For each task:
1. Filter to informative aspects (|ρ(aspect, label)| ≥ 0.05)
2. Send the filtered aspects' (name + description) list to Claude/GPT-5 in one prompt:
   - "Here are N evaluation rubrics for task <T>. Discover 5-8 natural categories that group them. For each aspect, assign a category."
3. Get back: `{category_name: [aspect_ids]}` + 1-line definitions per category
4. ~$0.05/task → ~$0.50 total

### Method (more rigorous, optional)
For each task:
1. Embed aspect_name + description with `text-embedding-3-small`
2. Hierarchical clustering at threshold for ~5-8 clusters
3. Use Claude on each cluster to generate a category label
4. Inspect membership manually

### Cross-references
- **Type × articulability (1-4)**: predict types like "syntactic" cluster at articulability=1, "craft" at articulability=3-4. Validates the L1-L4 scale.
- **Type × outcome correlation**: which type carries most predictive signal per task? Predict different per task — code_review may be evidentiary-heavy, creative_writing craft-heavy.
- **Type × thinking tokens**: which types demand most reasoning? Triangulates with articulability and outcome corr.
- **Cross-task universals**: do similar type labels appear across tasks (e.g., "clarity" everywhere)? Hints at a domain-general aspect taxonomy.

### Deliverable
A table per task: `category | n_aspects | mean_|ρ_label| | mean_articulability | mean_thinking`.
Plus one cross-task heatmap of category prevalence.

## 5. Additional candidates

- **Aspect redundancy map** on 1000-dp panel: re-do the per-task aspect-aspect clustering (was on 40 dps earlier, got cluster of 12 for peer_review Open Science). At 1000 dps the clusters should be cleaner.
- **R2 vs R2_post AUC**: does the merge actually help predictive performance? Quick compare.
- **Cross-task universal aspects**: at 1000 dps × 9 tasks, can identify aspects that are outcome-predictive across multiple tasks (clarity, evidence-support, etc.)
- **Failure-mode inspection**: pull 20 most-disagreed-on (dp, aspect) cells across Claude/Qwen; look at the texts and Q reasoning to understand what they disagree about. Qualitative anchor.
- **Code-trust map** (already partly done): which aspects can be code-judged reliably? At 1000 dps the ρ(code, claude) estimates are tight.

## Priority order for tomorrow

Highest-value first:

| # | Analysis | Time | Output |
|---|---|---|---|
| 1 | Dense vs Judge vs Code per-task AUC | 1h | Bar chart, 9 tasks |
| 2 | Articulability × ρ(aspect, label) | 30m | Scatter + hypothesis test |
| 3a | LASSO judge → label per task | 1h | Per-task table + selected aspects |
| 4 | G-theory σ² decomposition (3-facet) | 1h | Per-task variance pie chart |
| 5a | Aspect redundancy at n=1000 | 30m | Cluster sizes; compare to n=40 |
| 3b | LLM-as-synthetic-judge baseline | 1-2h | One more AUC bar per task |
| 5d | Qualitative disagreement inspection | 30m | 20-case appendix |

Plus 1-2h slack for surprises. Total ~6-8h of analysis.

## Notebook deliverable

`notebooks/2026-05-28__noah-meeting-prep.ipynb` with:
- Section 1: VAT three-layer AUC plot
- Section 2: Articulability × signal scatter
- Section 3: LASSO + LLM baselines
- Section 4: G-theory σ²
- Section 5: Redundancy + qualitative