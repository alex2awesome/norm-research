# LLMs for Annotation: A Thematic Literature Review

*(2026-06-13. Broad sweep — 171 papers across 9 sub-areas + 6 gap-fills, via multi-agent
workflow. Feeds the metric-articulability project: prompt-length/kind scaling of LLM-judge
fidelity on tasks without clean ground truth. Companion to `2026-06-12__formalization.md`
and `project_observational_scaling_irsl_2026_06_13` memory.)*

## 1. Landscape

LLM annotation research has moved through three phases in roughly four years. An optimistic
2021–2023 wave (Wang 2021; Gilardi 2023; Törnberg 2023) showed that zero-shot LLMs can match or
beat crowd workers — and sometimes experts — at a fraction of the cost, with higher inter-coder
reliability. A correction wave (Pangakis 2023; Reiss 2023; Bavaresco 2025; Baumann 2025)
established that LLM annotation quality is acutely task-, prompt-, and model-contingent, that
single-prompt results are non-reproducible, and that downstream conclusions can be silently
flipped by defensible-but-arbitrary choices ("LLM hacking"). Running underneath is an older and
deeper current — data perspectivism and learning-from-disagreement (Aroyo 2015; Pavlick 2019;
Plank 2022) — which denies that a single gold label exists for subjective tasks at all, and a
parallel statistical machinery (Dawid-Skene 1979; Ratner 2016/2017; Angelopoulos 2023; Egami
2023) for estimating reliability and drawing valid inferences without gold. The frontier today
fuses these: validating LLM judges under genuine label indeterminacy (Guerdan 2025),
prompt-as-program optimization (APE/OPRO/DSPy/GEPA), and capability/oversight scaling laws
(Engels 2025; Kenton 2024). The recurring, load-bearing distinction across all of it is
**reliability ≠ validity**: LLM judges can be highly self-consistent yet badly miscalibrated
against expert taste, especially where humans themselves disagree.

---

## 2. Themes

### (a) Do LLMs work as annotators, and when?

"Yes for objective/classification tasks, contingently for subjective ones, and never without
validation." Zero-shot LLMs beat crowd workers (Gilardi 2023) and even experts on relatively
objective labeling (Törnberg 2023), and pseudo-labels cut cost 50–96% (Wang 2021). But
performance does not transfer across tasks (Pangakis 2023), collapses on subjective
expert-taxonomy tasks like hate/empathy (Ziems 2024), and is high on free-form generation where
LLM output can exceed crowd "gold." Mature consensus: augment-don't-replace + mandatory per-task
human validation.

| Paper | Finding |
|---|---|
| Wang 2021 | GPT-3 pseudo-labels cut labeling cost 50–96% for equal downstream accuracy; hybrid best under budget |
| Gilardi 2023 | Zero-shot ChatGPT beats MTurk ~25 pts, higher inter-coder agreement than crowd AND trained annotators, ~20–30× cheaper |
| Törnberg 2023 | Zero-shot GPT-4 beats experts AND crowd on political-tweet labeling, equal-or-lower bias |
| Ziems 2024 | LLMs reach fair agreement on classification, trail fine-tuned models; on free-form coding can exceed crowd gold; use as augmenters |
| Pangakis 2023 | Quality highly task-contingent across 27 tasks; per-task human validation mandatory |
| Tan 2024 (survey) | Taxonomy: annotation generation / assessment / learning-with; catalogs prompting + reliability limits |

### (b) Prompt length & instruction-budget effects

No monotone "more instruction = better" law; dominant finding is a task-dependent sweet spot and
a sharp dissociation between token count and task complexity. Brief codebook descriptions beat
verbose all-in-one (Halterman 2025), optimal verbosity is domain-specific (Majer 2024), detailed
analytic rubrics can *hurt* via contextual interference while helping holistic scoring (Kucia
2026). Cleanest causal control (Pipal 2026): accuracy loss under long stacked prompts comes from
juggling multiple coding *schemes*, not extra tokens — a confound any articulability-vs-length
curve must rule out. The one explicit budget-scaling law: "curse of instructions" (Jaroslawicz
2025), prompt-acc ≈ (instr-acc)^n with a model-specific budget (~150–200) before collapse, plus
position dropout (lost-in-the-middle, Liu 2023).

| Paper | Finding |
|---|---|
| Jaroslawicz 2025 (IFScale) | "Curse of instructions": prompt-acc ≈ (instr-acc)^n; per-model budget ~150–200 before collapse; later instructions dropped more |
| Pipal 2026 | Token-count vs complexity dissociated: length-matched control shows extra tokens don't hurt; multi-scheme load does |
| Liu 2023 (Lost in Middle) | U-shaped position curve; mid-prompt content underused — long rubrics risk losing middle criteria |
| Halterman 2025 | BRIEF codebook task descriptions improve fidelity; verbose all-in-one hurts |
| Majer 2024 | No globally optimal guideline length; optimal verbosity domain-dependent |
| Kucia 2026 | Longer rubric text helps HOLISTIC scoring but HURTS multi-trait ANALYTIC scoring |
| Yamauchi 2025 | Anchoring only extreme score points (1 & 5) is best; intermediate descriptions add little |
| Song 2024 | More in-context shots raises judge consistency for *capable* judges |

### (c) Prompt kind / format / few-shot / CoT effects

Prompt KIND is a high-variance, often dominant lever, but its sign is task- and model-dependent.
Meaning-preserving format/separator changes swing accuracy up to 76 pts (Sclar 2024) and reorder
leaderboards (Mizrahi 2024); few-shot ordering alone can move SOTA→random (Lu 2022); example
selection matters (Liu 2022); yet demo *label correctness* mostly doesn't (Min 2022). CoT helps
mainly math/symbolic, unreliable for soft judgments (Sprague 2025; Bavaresco 2025), can degrade
disagreement modeling (Ni/Lin 2026). Most reliably beneficial: rationale-augmented few-shot
("explain-then-annotate," AnnoLLM 2024), criteria + reference anchoring (Yamauchi 2025; Krumdick
2025), decomposition/checklists. Forcing rigid output schemas can depress reasoning (Tam 2024).

| Paper | Finding |
|---|---|
| Sclar 2024 (FormatSpread) | Trivial format/separator changes swing few-shot accuracy up to 76 pts; persists across size/shots/instruction-tuning |
| Mizrahi 2024 | Meaning-preserving paraphrases reorder model rankings; report distribution over paraphrases |
| Lu 2022 | Few-shot ORDER alone: SOTA→random; good orders don't transfer; label-free entropy selection |
| Liu 2022 (KATE) | Nearest-neighbor demo selection beats random |
| Min 2022 | Random demo labels barely hurt; demos work via format/label-space/distribution, not correctness |
| AnnoLLM (He 2024) | "Explain-then-annotate" self-generated CoT exemplars close the human-LLM gap |
| Sprague 2025 | CoT helps mainly math/symbolic; little benefit on soft/subjective judgments |
| Tam 2024 | Forcing JSON/strict schema degrades reasoning; free-form-then-parse better |
| Atreja 2025 | Prompt design strong but unpredictable; numeric scores hurt compliance; explanations shift label distribution |

### (d) Labeling WITHOUT ground truth (subjective tasks, disagreement, perspectivism)

The conceptual core for subjective tasks. Human label variation is signal, not noise (Plank 2022;
Aroyo 2015); disagreement is inherent and persists with more ratings/context (Pavlick 2019; Nie
2020/ChaosNLI); dataset creators must *choose* a descriptive vs prescriptive paradigm — which
changes the label distribution (Röttger 2022). Evaluate against the human distribution (KL/JS)
and stratify by agreement level, where models go near-random (Nie 2020). "Whose ground truth"
becomes a tunable design parameter (Gordon 2022 jury learning; Santurkar 2023 OpinionQA). Crucially
disagreement decomposes into resolvable (ambiguity/missing-info/task-design) vs irreducible
(genuine subjectivity) (Sandri 2023) — mirroring an articulable-vs-taste split. Most on-target
methods paper: validate judges *under rating indeterminacy*; standard gold-validation mis-ranks
judges up to 34% (Guerdan 2025).

| Paper | Finding |
|---|---|
| Plank 2022 | Human label variation is signal; drop single-gold assumption; keep disaggregated labels |
| Pavlick 2019 | NLI disagreement inherent; persists with more ratings/context; recover full distribution |
| Nie 2020 (ChaosNLI) | Models near-perfect on high-agreement, near-random on low-agreement items; score by KL/JS |
| Aroyo 2015 (CrowdTruth) | Quality = structured disagreement over worker/item/label triangle, no gold |
| Röttger 2022 | Prescriptive vs descriptive guidelines yield different label distributions — instruction framing *constructs* the target |
| Sandri 2023 | Taxonomy of disagreement: resolvable (ambiguity/missing-info) vs irreducible (subjectivity) |
| Guerdan 2025 | Gold-based validation mis-ranks judges up to 34% under indeterminacy; decompose disagreement/error/forced-choice |
| Gordon 2022 (Jury Learning) | "Whose ground truth" made an explicit, configurable jury |

### (e) LLM-as-judge bias & reliability

Strong judges reach ~80% human agreement (Zheng 2023) but carry a canonical bias trio — position,
verbosity/length, self-preference — plus more (12 in CALM/Ye 2024; ~40% of pairwise comparisons
biased, CoBBLEr/Koo 2024). Position bias can flip rankings (Wang 2023); verbosity bias exceeds
human length preference (Saito 2023), correctable by regressing length out (Dubois 2024).
Self-preference is causally tied to self-recognition (Panickssery 2024) — a direct leakage risk
when judge and dense target share a base family. Judges are internally inconsistent (Stureborg
2024; Reiss 2023; JudgeSense 2026 — detailed rubrics grade harsher). Mitigations: multi-evidence +
balanced-position calibration (Wang 2023), reference-guided grading (Zheng 2023), many-shot with
reference (Song 2024), design-over-scale tuning (Beyer 2025).

| Paper | Finding |
|---|---|
| Zheng 2023 (MT-Bench) | GPT-4 ~80% human agreement; position/verbosity/self-enhancement biases; reference-guided CoT cuts errors |
| Wang 2023 | Position bias flips rankings; multi-evidence + balanced-position calibration +14.3% |
| Saito 2023 | LLMs over-prefer longer answers at equal quality, more than humans |
| Dubois 2024 (LC-AlpacaEval) | Regress length out via GLM; Spearman 0.94→0.98 |
| Panickssery 2024 | Self-recognition causally drives self-preference (near-linear) |
| Ye 2024 (CALM) | 12 biases; perturb-and-measure-invariance quantifies each without gold |
| JudgeSense 2026 | Detailed/structured rubrics grade systematically harsher; rubric ID/order shifts scores |
| Beyer 2025 | Prompt design (few-shot/CoT/format) > raw judge size for agreement; ~1/1000 cost |

### (f) Quality/accuracy estimation & aggregation without gold

Mature toolkit estimates per-annotator reliability and consensus truth from agreement structure
alone. Dawid-Skene (1979) is the latent-true-label EM base; spectral+EM is minimax-optimal (Zhang
2014); accuracies of conditionally-independent classifiers are identifiable from
covariance/tensor structure (Jaffe 2015; Platanios 2017). Weak supervision (Ratner 2016/2017)
treats each noisy source — naturally, each prompt/rubric variant — as a labeling function with
learnable accuracy. Neural variants learn per-annotator confusion jointly (Rodrigues 2018). For
LLM judges: diverse-model panels beat a single large judge and reduce self-preference (Verga 2024
PoLL); reliability-weighted Bradley-Terry handles unequal judges (BT-σ 2026); but multi-agent
debate can *amplify* bias (Ma 2025). CrowdTruth (Dumitrache 2018) + Bayesian Truth Serum (Prelec
2004) handle the genuinely no-gold case. Critically: reference-free agreement overstates
articulability — judges agree only where they could answer themselves (Krumdick 2025); high F1
hides downstream prevalence bias (Stolwijk 2025).

| Paper | Finding |
|---|---|
| Dawid-Skene 1979 | Latent true labels + per-annotator confusion matrices via EM, no gold |
| Ratner 2016/2017 (Data Programming/Snorkel) | Learn labeling-function accuracies from agreement; denoise to soft labels |
| Jaffe 2015 / Platanios 2017 | Estimate per-classifier accuracy from agreement geometry/logic, fully unsupervised |
| Verga 2024 (PoLL) | Diverse-model jury beats single GPT-4 judge, less self-preference, ~7× cheaper |
| BT-σ 2026 | Reliability-weighted Bradley-Terry for unequal-quality judge panels |
| Ma 2025 | Multi-agent debate AMPLIFIES position/verbosity/CoT bias; meta-judge more resistant |
| Krumdick 2025 | Reference-free judge agreement bounded by judge competence; expert references close gap |
| Stolwijk 2025 | High F1 hides systematic prevalence bias; judges overlap each other more than humans |

### (g) Uncertainty & calibration

Verbalized confidence is elicitable and often better-calibrated than logits (Lin 2022; Tian
2023), best from multi-guess/top-k prompting; but all methods are overconfident and collapse on
expert/professional tasks (Xiong 2024) — exactly the subjective regime. Sampling-based
meaning-space signals (semantic entropy, Kuhn 2023; Farquhar 2024) flag unreliable outputs
without gold. Sharpest warning: zero-shot LLMs match the *modal* human label but badly mis-estimate
the *spread* of disagreement (Inoshita 2026 — emotion-entropy ρ≈0.20 vs RoBERTa 0.47), and naive
best-of-N collapses to consensus, destroying the disagreement distribution one wants (Ruiz 2025).
Calibration/temperature/sampling choices matter for distribution recovery (Pavlovic 2024).

| Paper | Finding |
|---|---|
| Lin 2022 / Tian 2023 | Verbalized confidence elicitable & better-calibrated than logits; multi-guess prompting best |
| Xiong 2024 | All confidence methods overconfident; none dominant; collapse on expert/subjective tasks |
| Farquhar 2024 / Kuhn 2023 | Semantic entropy (meaning-cluster sampling) flags unreliable outputs without gold |
| Inoshita 2026 | LLMs capture majority label, NOT uncertainty distribution; calibration only partially fixes |
| Ruiz 2025 (BoN Appetit) | Best-of-N collapses to consensus, loses human disagreement; need distribution-preserving aggregation |
| Pavlovic 2024 | Sampling/log-prob extraction beats direct-distribution prompting for human-opinion alignment |

### (h) Downstream statistical correction (CSS validation, prediction-powered inference)

Raw LLM labels bias downstream estimates and invalidate CIs even at 80–90% accuracy; fix is
design-based correction with a small probability-sampled gold set. PPI (Angelopoulos 2023) and DSL
(Egami 2023) give provably valid CIs from imperfect labels; choice is context-dependent (Audinet
2025). Perspectivist extension targets group-conditional distributions with adaptive human
sampling (Mehrotra 2026 — finds few-shot > persona prompting). Most sobering: across 13M labels,
regression correction was largely ineffective and ~31% of conclusions wrong (~50% for small
models), 100 human annotations beat 100K LLM annotations, and prompt KIND explained <1% of
conclusion-correctness variance (Baumann 2025). Meta-stance: keep the goal *inference*, not
classification accuracy (Barrie 2025).

| Paper | Finding |
|---|---|
| Angelopoulos 2023 (PPI) | Valid CIs/p-values from many LLM + few gold labels; better predictor → tighter intervals |
| Egami 2023 (DSL) | Doubly-robust correction; valid even for arbitrarily biased surrogates, needs known sampling probs |
| Audinet 2025 | PPI vs DSL: best method context-dependent; PPI wins at tiny budgets |
| Mehrotra 2026 (PDI) | Target group-conditional distribution; PPI++ + adaptive sampling; few-shot > persona prompting |
| Baumann 2025 (LLM hacking) | ~31% conclusions wrong (50% small models); correction largely ineffective; 100 human > 100K LLM; prompt KIND <1% of variance |
| Barrie 2025 | Keep goal = valid inference not accuracy; LLMs are "fickle" instruments; bespoke validation |

### (i) Reproducibility pitfalls

LLM annotation is uniquely hard to reproduce: outputs are non-deterministic across reruns and
shift with minor wording (Reiss 2023; Haldar 2025 "Rating Roulette"; Stureborg 2024); closed-API
models drift over months (Chen 2023 — GPT-4 prime-ID 84%→51%) and get deprecated (Barrie/Palmer/
Spirling 2025); "garden of forking paths" config choices swing human-LLM correlation r=.23→.84
(Cummins 2025). Benchmark contamination inflates apparent competence (Xu 2024). Controls:
prompt-stability scoring (Barrie 2024 PSS), versioned codebooks (SILICON, Cheng 2024), multi-prompt
distributions (Mizrahi 2024; PromptEval/Polo 2024), pinned open-weight judges. Statistical power
is fragile: metric-vs-human correlation CIs are wide enough that many "improvements" are within
resampling noise (Deutsch 2021).

| Paper | Finding |
|---|---|
| Chen 2023 | Same API model drifts months apart; CoT responsiveness dropped between versions |
| Barrie/Palmer/Spirling 2025 | LMs fragile, can't replicate exactly, closed APIs change/vanish; need new replication standards |
| Cummins 2025 | Config choices swing human-LLM correlation r=.23→.84; pre-register/sweep |
| Barrie 2024 (PSS) | Prompt Stability Score: intra-/inter-prompt reliability without gold; Python package |
| Cheng 2024 (SILICON) | Treat LLM annotation like human coding; versioned validated guidelines + reliability reporting |
| Xu 2024 | Benchmark contamination inflates scores, grows with model size; favor fresh/refactored data |
| Deutsch 2021 | Metric-vs-human correlation CIs very wide; most metrics indistinguishable from ROUGE |

---

## 3. DIRECT HITS for our project

| Paper | Why it is a direct hit |
|---|---|
| Guerdan 2025 (Validating LLM-Judges under Rating Indeterminacy) | Most on-target: validates judges *without gold* under principled disagreement; standard validation mis-ranks judges up to 34%; decomposes disagreement/error/forced-choice — exactly our scoring setup |
| Choi 2026 (Diagnosing Judge Reliability via IRT) | No-gold latent-quality θ from cross-prompt invariance; treats prompt variants as IRT items; splits intrinsic consistency vs human alignment — a candidate instrument for the articulability measurement itself (CONVERGES with IRSL 2606.07616) |
| Yamauchi 2025 (Judge Design Choices) | Quantifies which prompt elements carry signal: criteria > references > anchors; extreme-anchoring beats verbose; CoT redundant once criteria exist — empirical playbook for our prompt-KIND/length ablations |
| Krumdick 2025 (No Free Labels) | Reference-free judge agreement bounded by judge competence; a written reference substitutes for capability — i.e. "articulability" partly = judge competence ceiling; reference-free agreement *overstates* articulability |
| GEPA (Agrawal 2025) + MIPRO (Opsahl-Ong 2024) | Named source of our operators; reflective evolution makes instructions longer/more explicit but reports NO controlled length-scaling — the gap we fill; instruction-vs-demo edits are separable jointly-optimizable levers |
| Jaroslawicz 2025 (IFScale) + Pipal 2026 | Closest to instruction-budget scaling: acc≈(instr-acc)^n, per-model budget ~150–200; Pipal isolates token-count from complexity confound — essential controls for our length axis |
| Engels 2025 (Scaling Laws for Scalable Oversight) + Kenton 2024 | Importable Double-ReLU capability-gap scaling law + weak-judges-judging-strong setup — maps to our judge-capability axis, predicts where weak judges fail on tacit tasks |
| Chakrabarty 2024 (TTCW) + Chhun 2024 (HANNA) | LLM judges fail to reproduce *expert* aesthetic judgment even with CoT + decomposition + anchored rubrics; longer/clarified prompts don't help, sometimes hurt; high self-consistency + low human agreement = clean articulability-ceiling argument |
| Pan 2026 (Bi-Level Judge Prompt Optimization) | Optimizes the JUDGE's rubric to match human verdicts — operationally identical to optimizing a rubric so a judge reproduces expert judgment (our articulability measurement) |
| OpenRubrics 2025 / RRD 2026 | Hard-rules-vs-principles split maps onto verifiable/articulable/taste; RRD characterizes failure modes of decomposing a metric (coverage vs redundancy vs preference-misalignment) |

**Concrete connections to our framework:**

1. **Articulability gap = judge-competence-bounded reproduction gap.** Krumdick 2025 + Engels 2025
   jointly imply our measured "articulability" is confounded with the judge's own competence and the
   judge-vs-target capability gap (Double-ReLU). Report articulability *as a function of judge
   capability*; a written rubric/reference is a capability substitute (Beyer 2025: design > scale;
   JudgeLM: reference helps small judges most, shrinks at 33B).

2. **V/A/Taste decomposition** lines up with: OpenRubrics' hard-rules vs principles; Sandri 2023's
   resolvable vs irreducible disagreement; Sprague 2025's symbolic-vs-soft CoT divide. The "taste
   residual" is the irreducible-disagreement component Pavlick 2019 / Nie 2020 show models go
   near-random on — predicting an articulability ceiling below 1.0 that is *not* methodological failure.

3. **Chinchilla-style scaling for prompt length** has a functional-form precedent in Jaroslawicz
   2025's acc≈(instr-acc)^n and per-model "instruction budget," plus non-monotone sweet spots
   (Halterman, Majer, Kucia, Yamauchi). Pipal 2026 = mandatory token-vs-scheme confound control; Liu
   2023 = position-within-rubric confound. Our instruction-budget axis is the controlled scaling study
   none of these ran.

4. **Judge panels / CrowdTruth-style reliability without gold** is supplied by PoLL, BT-σ,
   Dawid-Skene/Snorkel (each prompt-variant = a labeling function with estimable accuracy), and
   CrowdTruth. Caution (Ma 2025): debate-style aggregation *amplifies* bias — prefer panel/meta-judge
   topologies. Schroeder 2024 (McDonald's ω) + Barrie 2024 (PSS) give the test-retest statistics.

5. **words_share / disagreement as signal.** Plank 2022, Aroyo 2015, Röttger 2022, Inoshita 2026
   license scoring against a *disagreement distribution* rather than a point label — and warn
   (Inoshita; Ruiz 2025) that sampled judges recover the mode but mis-estimate the spread, and BoN
   collapses it. Use distribution-preserving aggregation + soft metrics; Rizzi 2024 warns CE can
   mislead (prefer distance-based soft metrics).

6. **Prompt-kind ≠ neutral knob; report distributions.** Sclar 2024, Mizrahi 2024, Polo 2024
   (PromptEval), Baumann 2025 jointly mandate that any articulability number from one phrasing is
   exploitable ("LLM hacking"). The GEPA-operator sweep should report articulability as a
   *distribution/interval over phrasings*, with downstream estimates corrected via PPI/DSL.

---

## 4. Open questions / what nobody seems to have done (our contribution slots)

1. **A controlled instruction-token-budget scaling law for articulability under no gold.** Every
   optimizer (GEPA, MIPRO, OPRO, EvoPrompt) lets prompts grow but reports no controlled
   length-vs-fidelity curve; IFScale/Pipal isolate length only on mechanically-verifiable tasks.
   Nobody has measured a Chinchilla-style "articulability vs instruction-token budget" curve on
   subjective tasks targeting a disagreement distribution — decomposed from scheme-complexity (Pipal)
   and position (Liu 2023) confounds.

2. **Crossing prompt-KIND × prompt-LENGTH × judge-CAPABILITY as one design.** Pieces exist
   separately (operators GEPA; length sweet spots Kucia/Halterman/Yamauchi; capability scaling
   Engels/JudgeLM). No paper runs the full 3-way interaction to test whether better rubric KIND
   substitutes for length, and whether either substitutes for judge capability.

3. **Per-metric articulability as an estimand with valid no-gold inference.** PPI/DSL/PDI correct
   downstream parameters; Guerdan/Choi validate judge choice. Nobody defines a per-metric
   articulability coefficient with a PPI-style valid CI across prompt conditions — letting one claim
   "metric A more articulable than B" with calibrated uncertainty and minimal expert anchoring.

4. **Disentangling articulability from self-preference leakage when the target is a dense model.**
   Panickssery 2024 + Meta-Evaluation Collapse (Mukherjee 2025) imply reproducing a dense-model target
   may reflect shared bias/self-recognition, not recovered expertise. No protocol controls this
   (cross-family judge≠target) while measuring against an *expert* (vs crowd vs dense) standard.

5. **An operator-level "articulability lift" attribution.** Decomposition lit (BSM, TICK, CheckEval,
   FLASK, DnA-Eval, HD-Eval) shows decomposition helps on average; Furuhashi 2025 shows it helps only
   selectively. Nobody attributes which operator (clarify vs mechanize vs anchor vs decompose) closes
   the gap *for which kind of metric* (verifiable vs articulable vs taste) — our operator × metric-type map.

6. **Articulability ceiling vs taste residual, measured cross-task with a constant instrument.**
   Chhun 2024 / Chakrabarty 2024 show ceilings below 1.0 on creativity but never separate "judge can't
   articulate it" from "humans irreducibly disagree." A cross-task constant-instrument measure (IRT θ
   à la Choi 2026, anchored to the human disagreement distribution à la Nie 2020) reporting 1−ceiling
   as a task-level taste constant appears un-done.


## References (auto-verified BibTeX, 2026-06-15)

> Citations below were extracted from this document and web-verified by an automated fact-check pass (search → fetch → retrieve resolvable id), with the attributed claim checked against the located paper. 92 entries; 3 also passed an independent second-pass audit (the rest were verified once — the audit pass was cut off by a quota limit, not by a failure). Entries are real located works; do not treat as hand-checked. See "needs manual review" below for 2 citations whose attributed claim the source paper appears to **contradict** and 0 unlocatable shorthands.

```bibtex
@misc{agrawal2025gepa,
  title        = {GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning},
  author       = {Lakshya A Agrawal and Shangyin Tan and Dilara Soylu and Noah Ziems and Rishi Khare and Krista Opsahl-Ong and Arnav Singhvi and Herumb Shandilya and Michael J Ryan and Meng Jiang and Christopher Potts and Koushik Sen and Alexandros G. Dimakis and Ion Stoica and Dan Klein and Matei Zaharia and Omar Khattab},
  year         = {2025},
  eprint       = {2507.19457},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  note         = {Accepted at ICLR 2026 (Oral)},
  url          = {https://arxiv.org/abs/2507.19457}
}

@article{angelopoulos2023prediction,
  title={Prediction-powered inference},
  author={Angelopoulos, Anastasios N. and Bates, Stephen and Fannjiang, Clara and Jordan, Michael I. and Zrnic, Tijana},
  journal={Science},
  volume={382},
  number={6671},
  pages={669--674},
  year={2023},
  publisher={American Association for the Advancement of Science},
  doi={10.1126/science.adi6000}
}

@article{argyle2025arti,
  title   = {Arti-'fickle' intelligence: using LLMs as a tool for inference in the political and social sciences},
  author  = {Argyle, Lisa P. and Busby, Ethan C. and Gubler, Joshua R. and Hepner, Bryce and Lyman, Alex and Wingate, David},
  journal = {Nature Computational Science},
  volume  = {5},
  pages   = {737--744},
  year    = {2025},
  doi     = {10.1038/s43588-025-00843-4}
}

@article{aroyo2015truth,
  title={Truth Is a Lie: Crowd Truth and the Seven Myths of Human Annotation},
  author={Aroyo, Lora and Welty, Chris},
  journal={AI Magazine},
  volume={36},
  number={1},
  pages={15--24},
  year={2015},
  doi={10.1609/aimag.v36i1.2564}
}

@inproceedings{atreja2025whats,
  title     = {What's in a Prompt?: A Large-Scale Experiment to Assess the Impact of Prompt Design on the Compliance and Accuracy of LLM-Generated Text Annotations},
  author    = {Atreja, Shubham and Ashkinaze, Joshua and Li, Lingyao and Mendelsohn, Julia and Hemphill, Libby},
  booktitle = {Proceedings of the International AAAI Conference on Web and Social Media},
  volume    = {19},
  number    = {1},
  pages     = {122--145},
  year      = {2025},
  doi       = {10.1609/icwsm.v19i1.35807}
}

@inproceedings{audinet2025benchmarking,
  title     = {Benchmarking Debiasing Methods for {LLM}-based Parameter Estimates},
  author    = {Audinet de Pieuchon, Nicolas and Daoud, Adel and Jerzak, Connor Thomas and Johansson, Moa and Johansson, Richard},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year      = {2025},
  pages     = {19757--19772},
  publisher = {Association for Computational Linguistics},
  address   = {Suzhou, China},
  doi       = {10.18653/v1/2025.emnlp-main.1000}
}

@misc{barrie2024prompt,
  title        = {Prompt Stability Scoring for Text Annotation with Large Language Models},
  author       = {Barrie, Christopher and Palaiologou, Elli and T{\"o}rnberg, Petter},
  year         = {2024},
  eprint       = {2407.02039},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  doi          = {10.48550/arXiv.2407.02039},
  url          = {https://arxiv.org/abs/2407.02039}
}

@misc{barrie2025replication,
  author       = {Barrie, Christopher and Palmer, Alexis and Spirling, Arthur},
  title        = {Replication for Language Models: Problems, Principles, and Best Practice for Political Science},
  year         = {2025},
  note         = {Working paper, conditionally accepted at American Journal of Political Science},
  howpublished = {\url{https://arthurspirling.org/documents/BarriePalmerSpirling_TrustMeBro.pdf}}
}

@misc{baumann2025large,
  title={Large Language Model Hacking: Quantifying the Hidden Risks of Using LLMs for Text Annotation},
  author={Baumann, Joachim and R{\"o}ttger, Paul and Urman, Aleksandra and Wendsj{\"o}, Albert and Plaza-del-Arco, Flor Miriam and Gruber, Johannes B. and Hovy, Dirk},
  year={2025},
  eprint={2509.08825},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2509.08825}
}

@inproceedings{bavaresco2025llms,
  title     = {{LLMs} instead of Human Judges? A Large Scale Empirical Study across 20 {NLP} Evaluation Tasks},
  author    = {Bavaresco, Anna and Bernardi, Raffaella and Bertolazzi, Leonardo and Elliott, Desmond and Fern{\'a}ndez, Raquel and Gatt, Albert and Ghaleb, Esam and Giulianelli, Mario and Hanna, Michael and Koller, Alexander and Martins, Andr{\'e} F. T. and Mondorf, Philipp and Neplenbroek, Vera and Pezzelle, Sandro and Plank, Barbara and Schlangen, David and Suglia, Alessandro and Surikuchi, Aditya K and Takmaz, Ece and Testoni, Alberto},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
  year      = {2025},
  pages     = {238--255},
  address   = {Vienna, Austria},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2025.acl-short.20},
  url       = {https://aclanthology.org/2025.acl-short.20/}
}

@misc{bellibatlu2026judgesense,
  title={JudgeSense: A Benchmark for Prompt Sensitivity in LLM-as-a-Judge Systems},
  author={Bellibatlu, Rohith Reddy and Raff, Edward and Zhang, Wenbin},
  year={2026},
  eprint={2604.23478},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2604.23478}
}

@inproceedings{chakrabarty2024art,
  title={Art or Artifice? Large Language Models and the False Promise of Creativity},
  author={Chakrabarty, Tuhin and Laban, Philippe and Agarwal, Divyansh and Muresan, Smaranda and Wu, Chien-Sheng},
  booktitle={Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems (CHI '24)},
  year={2024},
  publisher={Association for Computing Machinery},
  doi={10.1145/3613904.3642731}
}

@misc{chen2023how,
  title        = {How is ChatGPT's behavior changing over time?},
  author       = {Chen, Lingjiao and Zaharia, Matei and Zou, James},
  year         = {2023},
  eprint       = {2307.09009},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2307.09009}
}

@misc{cheng2024err,
  title={To Err Is Human; To Annotate, SILICON? Toward Robust Reproducibility in LLM Annotation},
  author={Cheng, Xiang and Mayya, Raveesh and Sedoc, Jo\~{a}o},
  year={2024},
  eprint={2412.14461},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  doi={10.48550/arXiv.2412.14461},
  url={https://arxiv.org/abs/2412.14461}
}

@article{chhun2024do,
  title     = {Do Language Models Enjoy Their Own Stories? Prompting Large Language Models for Automatic Story Evaluation},
  author    = {Chhun, Cyril and Suchanek, Fabian M. and Clavel, Chlo\'{e}},
  journal   = {Transactions of the Association for Computational Linguistics},
  volume    = {12},
  pages     = {1122--1142},
  year      = {2024},
  publisher = {MIT Press},
  doi       = {10.1162/tacl_a_00689},
  url       = {https://aclanthology.org/2024.tacl-1.62/}
}

@misc{choi2026diagnosing,
  title        = {Diagnosing the Reliability of LLM-as-a-Judge via Item Response Theory},
  author       = {Choi, Junhyuk and Park, Sohhyung and Cho, Chanhee and Park, Hyeonchu and Kim, Bugeun},
  year         = {2026},
  eprint       = {2602.00521},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL}
}

@misc{cummins2025threat,
  title        = {The threat of analytic flexibility in using large language models to simulate human data},
  author       = {Cummins, Jamie},
  year         = {2025},
  eprint       = {2509.13397},
  archivePrefix = {arXiv},
  primaryClass = {cs.CY},
  howpublished = {arXiv preprint arXiv:2509.13397},
  url          = {https://arxiv.org/abs/2509.13397}
}

@article{dawid1979maximum,
  title={Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm},
  author={Dawid, A. P. and Skene, A. M.},
  journal={Journal of the Royal Statistical Society: Series C (Applied Statistics)},
  volume={28},
  number={1},
  pages={20--28},
  year={1979},
  publisher={Wiley},
  doi={10.2307/2346806}
}

@article{deutsch2021statistical,
    title = "A Statistical Analysis of Summarization Evaluation Metrics Using Resampling Methods",
    author = "Deutsch, Daniel and Dror, Rotem and Roth, Dan",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "9",
    year = "2021",
    publisher = "MIT Press",
    pages = "1132--1146",
    doi = "10.1162/tacl_a_00417"
}

@inproceedings{dubois2024length,
  title={Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators},
  author={Dubois, Yann and Galambosi, Bal{\'a}zs and Liang, Percy and Hashimoto, Tatsunori B.},
  booktitle={Conference on Language Modeling (COLM)},
  year={2024},
  eprint={2404.04475},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2404.04475}
}

@inproceedings{dumitrache2018crowdtruth,
  title={CrowdTruth 2.0: Quality Metrics for Crowdsourcing with Disagreement},
  author={Dumitrache, Anca and Inel, Oana and Aroyo, Lora and Timmermans, Benjamin and Welty, Chris},
  booktitle={Proceedings of the 1st Workshop on Subjectivity, Ambiguity and Disagreement in Crowdsourcing (SAD) and CrowdBias 2018, co-located with HCOMP 2018},
  series={CEUR Workshop Proceedings},
  volume={2276},
  year={2018},
  note={arXiv:1808.06080},
  eprint={1808.06080},
  archivePrefix={arXiv},
  primaryClass={cs.HC}
}

@inproceedings{egami2023using,
  title={Using Imperfect Surrogates for Downstream Inference: Design-based Supervised Learning for Social Science Applications of Large Language Models},
  author={Egami, Naoki and Hinck, Musashi and Stewart, Brandon M. and Wei, Hanying},
  booktitle={Advances in Neural Information Processing Systems 36 (NeurIPS 2023)},
  year={2023},
  eprint={2306.04746},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@misc{engels2025scaling,
  title={Scaling Laws For Scalable Oversight},
  author={Engels, Joshua and Baek, David D. and Kantamneni, Subhash and Tegmark, Max},
  year={2025},
  eprint={2504.18530},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2504.18530}
}

@article{farquhar2024detecting,
  title={Detecting hallucinations in large language models using semantic entropy},
  author={Farquhar, Sebastian and Kossen, Jannik and Kuhn, Lorenz and Gal, Yarin},
  journal={Nature},
  volume={630},
  number={8017},
  pages={625--630},
  year={2024},
  publisher={Nature Publishing Group},
  doi={10.1038/s41586-024-07421-0}
}

@inproceedings{furuhashi2025checklists,
  title        = {Are Checklists Really Useful for Automatic Evaluation of Generative Tasks?},
  author       = {Furuhashi, Momoka and Nakayama, Kouta and Kodama, Takashi and Sugawara, Saku},
  booktitle    = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year         = {2025},
  eprint       = {2508.15218},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2508.15218}
}

@article{gilardi2023chatgpt,
  title={ChatGPT outperforms crowd workers for text-annotation tasks},
  author={Gilardi, Fabrizio and Alizadeh, Meysam and Kubli, Ma{\"e}l},
  journal={Proceedings of the National Academy of Sciences},
  volume={120},
  number={30},
  pages={e2305016120},
  year={2023},
  doi={10.1073/pnas.2305016120}
}

@inproceedings{gordon2022jury,
  author    = {Gordon, Mitchell L. and Lam, Michelle S. and Park, Joon Sung and Patel, Kayur and Hancock, Jeffrey T. and Hashimoto, Tatsunori and Bernstein, Michael S.},
  title     = {Jury Learning: Integrating Dissenting Voices into Machine Learning Models},
  booktitle = {Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems},
  series    = {CHI '22},
  year      = {2022},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3491102.3502004}
}

@inproceedings{guerdan2025validating,
  title     = {Validating {LLM}-as-a-Judge Systems under Rating Indeterminacy},
  author    = {Guerdan, Luke and Barocas, Solon and Holstein, Kenneth and Wallach, Hanna and Wu, Zhiwei Steven and Chouldechova, Alexandra},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  eprint    = {2503.05965},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2503.05965}
}

@inproceedings{haldar2025rating,
  title = {Rating Roulette: Self-Inconsistency in {LLM}-As-A-Judge Frameworks},
  author = {Haldar, Rajarshi and Hockenmaier, Julia},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2025},
  year = {2025},
  month = {November},
  pages = {24986--25004},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/2025.findings-emnlp.1361},
  url = {https://aclanthology.org/2025.findings-emnlp.1361/}
}

@article{halterman2025codebook,
  title   = {Codebook LLMs: Evaluating LLMs as Measurement Tools for Political Science Concepts},
  author  = {Halterman, Andrew and Keith, Katherine A.},
  journal = {Political Analysis},
  volume  = {34},
  number  = {2},
  pages   = {188--204},
  year    = {2025},
  publisher = {Cambridge University Press},
  doi     = {10.1017/pan.2025.10017}
}

@inproceedings{he2024annollm,
    title = "{A}nno{LLM}: Making Large Language Models to Be Better Crowdsourced Annotators",
    author = "He, Xingwei and Lin, Zhenghao and Gong, Yeyun and Jin, A-Long and Zhang, Hang and Lin, Chen and Jiao, Jian and Yiu, Siu Ming and Duan, Nan and Chen, Weizhu",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 6: Industry Track)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    pages = "165--190",
    doi = "10.18653/v1/2024.naacl-industry.15"
}

@misc{inoshita2026llms,
  title={LLMs Capture Emotion Labels, Not Emotion Uncertainty: Distributional Analysis and Calibration of Human-LLM Judgment Gaps},
  author={Inoshita, Keito and Zhou, Xiaokang and Kawai, Akira and Yada, Katsutoshi},
  year={2026},
  eprint={2604.27345},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@inproceedings{jaffe2015estimating,
  title = {Estimating the accuracies of multiple classifiers without labeled data},
  author = {Jaffe, Ariel and Nadler, Boaz and Kluger, Yuval},
  booktitle = {Proceedings of the Eighteenth International Conference on Artificial Intelligence and Statistics},
  pages = {407--415},
  year = {2015},
  editor = {Lebanon, Guy and Vishwanathan, S. V. N.},
  volume = {38},
  series = {Proceedings of Machine Learning Research},
  address = {San Diego, California, USA},
  month = {09--12 May},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v38/jaffe15.html}
}

@misc{jaroslawicz2025how,
  title        = {How Many Instructions Can LLMs Follow at Once?},
  author       = {Jaroslawicz, Daniel and Whiting, Brendan and Shah, Parth and Maamari, Karime},
  year         = {2025},
  eprint       = {2507.11538},
  archivePrefix= {arXiv},
  primaryClass = {cs.AI},
  url          = {https://arxiv.org/abs/2507.11538}
}

@inproceedings{kenton2024scalable,
  title={On scalable oversight with weak {LLMs} judging strong {LLMs}},
  author={Kenton, Zachary and Siegel, Noah Y. and Kram{\'a}r, J{\'a}nos and Brown-Cohen, Jonah and Albanie, Samuel and Bulian, Jannis and Agarwal, Rishabh and Lindner, David and Tang, Yunhao and Goodman, Noah D. and Shah, Rohin},
  booktitle={Advances in Neural Information Processing Systems},
  volume={37},
  pages={75229--75276},
  year={2024},
  eprint={2407.04622},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@misc{krumdick2025no,
  title        = {No Free Labels: Limitations of LLM-as-a-Judge Without Human Grounding},
  author       = {Krumdick, Michael and Lovering, Charles and Reddy, Varshini and Ebner, Seth and Tanner, Chris},
  year         = {2025},
  eprint       = {2503.05061},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2503.05061}
}

@misc{kucia2026llm,
  title={LLM Essay Scoring Under Holistic and Analytic Rubrics: Prompt Effects and Bias},
  author={Kucia, Filip J. and Chakraborty, Anirban and Wr{\'o}blewska, Anna},
  year={2026},
  eprint={2604.00259},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2604.00259}
}

@inproceedings{kuhn2023semantic,
  title={Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation},
  author={Kuhn, Lorenz and Gal, Yarin and Farquhar, Sebastian},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023},
  eprint={2302.09664},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@article{lin2022teaching,
  title={Teaching Models to Express Their Uncertainty in Words},
  author={Lin, Stephanie and Hilton, Jacob and Evans, Owain},
  journal={Transactions on Machine Learning Research},
  year={2022},
  url={https://arxiv.org/abs/2205.14334},
  note={arXiv:2205.14334}
}

@inproceedings{liu2022makes,
  title     = {What Makes Good In-Context Examples for {GPT}-3?},
  author    = {Liu, Jiachang and Shen, Dinghan and Zhang, Yizhe and Dolan, Bill and Carin, Lawrence and Chen, Weizhu},
  booktitle = {Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures},
  month     = may,
  year      = {2022},
  publisher = {Association for Computational Linguistics},
  pages     = {100--114},
  doi       = {10.18653/v1/2022.deelio-1.10},
  url       = {https://aclanthology.org/2022.deelio-1.10/}
}

@article{liu2023lost,
  author    = {Nelson F. Liu and Kevin Lin and John Hewitt and Ashwin Paranjape and Michele Bevilacqua and Fabio Petroni and Percy Liang},
  title     = {Lost in the Middle: How Language Models Use Long Contexts},
  journal   = {Transactions of the Association for Computational Linguistics},
  volume    = {12},
  pages     = {157--173},
  year      = {2024},
  doi       = {10.1162/tacl_a_00638},
  note      = {arXiv:2307.03172}
}

@misc{liu2025openrubrics,
  title={OpenRubrics: Towards Scalable Synthetic Rubric Generation for Reward Modeling and LLM Alignment},
  author={Tianci Liu and Ran Xu and Tony Yu and Ilgee Hong and Carl Yang and Tuo Zhao and Haoyu Wang},
  year={2025},
  eprint={2510.07743},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2510.07743}
}

@inproceedings{lu2022fantastically,
    title = "Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity",
    author = "Lu, Yao and Bartolo, Max and Moore, Alastair and Riedel, Sebastian and Stenetorp, Pontus",
    editor = "Muresan, Smaranda and Nakov, Preslav and Villavicencio, Aline",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.556/",
    doi = "10.18653/v1/2022.acl-long.556",
    pages = "8086--8098"
}

@misc{ma2025judging,
  title={Judging with Many Minds: Do More Perspectives Mean Less Prejudice? On Bias Amplification and Resistance in Multi-Agent Based LLM-as-Judge},
  author={Ma, Chiyu and Zhang, Enpei and Zhao, Yilun and Liu, Wenjun and Jia, Yaning and Qing, Peijun and Shi, Lin and Cohan, Arman and Yan, Yujun and Vosoughi, Soroush},
  year={2025},
  eprint={2505.19477},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2505.19477}
}

@inproceedings{majer2024claim,
    title = "Claim Check-Worthiness Detection: How Well do {LLM}s Grasp Annotation Guidelines?",
    author = "Majer, Laura and {\v{S}}najder, Jan",
    booktitle = "Proceedings of the Seventh Fact Extraction and VERification Workshop (FEVER)",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    pages = "245--263",
    doi = "10.18653/v1/2024.fever-1.27",
    url = "https://aclanthology.org/2024.fever-1.27/"
}

@misc{mehrotra2026multi,
  title={Multi-Perspective LLM Annotations for Valid Analyses in Subjective Tasks},
  author={Mehrotra, Navya and Visokay, Adam and Gligori\'{c}, Kristina},
  year={2026},
  eprint={2603.21404},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@inproceedings{min2022rethinking,
    title = {Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?},
    author = {Min, Sewon and Lyu, Xinxi and Holtzman, Ari and Artetxe, Mikel and Lewis, Mike and Hajishirzi, Hannaneh and Zettlemoyer, Luke},
    booktitle = {Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
    year = {2022},
    publisher = {Association for Computational Linguistics},
    pages = {11048--11064},
    doi = {10.18653/v1/2022.emnlp-main.759}
}

@article{mizrahi2024state,
  title   = {State of What Art? A Call for Multi-Prompt LLM Evaluation},
  author  = {Mizrahi, Moran and Kaplan, Guy and Malkin, Dan and Dror, Rotem and Shahaf, Dafna and Stanovsky, Gabriel},
  journal = {Transactions of the Association for Computational Linguistics},
  volume  = {12},
  pages   = {933--949},
  year    = {2024},
  doi     = {10.1162/tacl_a_00681},
  url     = {https://aclanthology.org/2024.tacl-1.52/}
}

@misc{mukherjee2025meta,
  title        = {Meta-Evaluation Collapse: Who Judges the Judges of Judges?},
  author       = {Mukherjee, Sourabrata},
  year         = {2025},
  note         = {OpenReview preprint; ICLR 2026 Conference Withdrawn Submission, submitted 20 Sept 2025},
  howpublished = {\url{https://openreview.net/forum?id=IF0L7HSs3K}}
}

@inproceedings{nie2020what,
    title = "What Can We Learn from Collective Human Opinions on Natural Language Inference Data?",
    author = "Nie, Yixin and Zhou, Xiang and Bansal, Mohit",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    pages = "9131--9143",
    doi = "10.18653/v1/2020.emnlp-main.734"
}

@inproceedings{opsahl-ong2024optimizing,
    title = {Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs},
    author = {Opsahl-Ong, Krista and Ryan, Michael J. and Purtell, Josh and Broman, David and Potts, Christopher and Zaharia, Matei and Khattab, Omar},
    booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
    month = nov,
    year = {2024},
    address = {Miami, Florida, USA},
    publisher = {Association for Computational Linguistics},
    pages = {9340--9366},
    doi = {10.18653/v1/2024.emnlp-main.525},
    url = {https://aclanthology.org/2024.emnlp-main.525/}
}

@misc{pan2026bilevel,
  title={Bi-Level Prompt Optimization for Multimodal LLM-as-a-Judge},
  author={Pan, Bo and Kan, Xuan and Zhang, Kaitai and Yan, Yan and Tan, Shunwen and He, Zihao and Ding, Zixin and Wu, Junjie and Zhao, Liang},
  year={2026},
  eprint={2602.11340},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  doi={10.48550/arXiv.2602.11340},
  url={https://arxiv.org/abs/2602.11340}
}

@misc{pangakis2023automated,
  title={Automated Annotation with Generative AI Requires Validation},
  author={Pangakis, Nicholas and Wolken, Samuel and Fasching, Neil},
  year={2023},
  eprint={2306.00176},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  doi={10.48550/arXiv.2306.00176},
  url={https://arxiv.org/abs/2306.00176}
}

@inproceedings{panickssery2024llm,
  title={LLM Evaluators Recognize and Favor Their Own Generations},
  author={Panickssery, Arjun and Bowman, Samuel R. and Feng, Shi},
  booktitle={Advances in Neural Information Processing Systems 37 (NeurIPS 2024)},
  year={2024},
  eprint={2404.13076},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2404.13076}
}

@article{pavlick2019inherent,
    title = {Inherent Disagreements in Human Textual Inferences},
    author = {Pavlick, Ellie and Kwiatkowski, Tom},
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {7},
    pages = {677--694},
    year = {2019},
    publisher = {MIT Press},
    doi = {10.1162/tacl_a_00293}
}

@misc{pavlovic2024understanding,
  title={Understanding The Effect Of Temperature On Alignment With Human Opinions},
  author={Pavlovic, Maja and Poesio, Massimo},
  year={2024},
  eprint={2411.10080},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2411.10080}
}

@misc{pipal2026researchers,
  title={Researchers waste 80\% of LLM annotation costs by classifying one text at a time},
  author={Pipal, Christian and Vogel, Eva-Maria and Wack, Morgan and Esser, Frank},
  year={2026},
  eprint={2604.03684},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  doi={10.48550/arXiv.2604.03684},
  url={https://arxiv.org/abs/2604.03684}
}

@inproceedings{plank2022problem,
    title = "The ``Problem'' of Human Label Variation: On Ground Truth in Data, Modeling and Evaluation",
    author = "Plank, Barbara",
    editor = "Goldberg, Yoav and Kozareva, Zornitsa and Zhang, Yue",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.731/",
    doi = "10.18653/v1/2022.emnlp-main.731",
    pages = "10671--10682"
}

@inproceedings{platanios2017estimating,
  title={Estimating Accuracy from Unlabeled Data: A Probabilistic Logic Approach},
  author={Platanios, Emmanouil A. and Poon, Hoifung and Mitchell, Tom M. and Horvitz, Eric},
  booktitle={Advances in Neural Information Processing Systems 30 (NIPS 2017)},
  pages={4361--4370},
  year={2017}
}

@inproceedings{polo2024tinybenchmarks,
  title     = {tinyBenchmarks: evaluating LLMs with fewer examples},
  author    = {Maia Polo, Felipe and Weber, Lucas and Choshen, Leshem and Sun, Yuekai and Xu, Gongjun and Yurochkin, Mikhail},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {235},
  pages     = {34303--34326},
  year      = {2024},
  publisher = {PMLR},
  eprint    = {2402.14992},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

@article{prelec2004bayesian,
  title={A Bayesian Truth Serum for Subjective Data},
  author={Prelec, Dra{\v{z}}en},
  journal={Science},
  volume={306},
  number={5695},
  pages={462--466},
  year={2004},
  publisher={American Association for the Advancement of Science},
  doi={10.1126/science.1102081}
}

@misc{qian2026who,
  title        = {Who can we trust? LLM-as-a-jury for Comparative Assessment},
  author       = {Qian, Mengjie and Sun, Guangzhi and Gales, Mark J.F. and Knill, Kate M.},
  year         = {2026},
  eprint       = {2602.16610},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  note         = {Accepted to ICML 2026},
  url          = {https://arxiv.org/abs/2602.16610}
}

@inproceedings{ratner2016data,
  author    = {Alexander J. Ratner and Christopher De Sa and Sen Wu and Daniel Selsam and Christopher R{\'e}},
  title     = {Data Programming: Creating Large Training Sets, Quickly},
  booktitle = {Advances in Neural Information Processing Systems (NIPS)},
  volume    = {29},
  pages     = {3567--3575},
  year      = {2016},
  url       = {https://proceedings.neurips.cc/paper/2016/hash/6709e8d64a5f47269ed5cea9f625f7ab-Abstract.html}
}

@article{ratner2017snorkel,
  author  = {Alexander Ratner and Stephen H. Bach and Henry R. Ehrenberg and Jason Alan Fries and Sen Wu and Christopher R{\'e}},
  title   = {Snorkel: Rapid Training Data Creation with Weak Supervision},
  journal = {Proceedings of the VLDB Endowment},
  volume  = {11},
  number  = {3},
  pages   = {269--282},
  year    = {2017},
  doi     = {10.14778/3157794.3157797}
}

@misc{reiss2023testing,
  title={Testing the Reliability of ChatGPT for Text Annotation and Classification: A Cautionary Remark},
  author={Reiss, Michael V.},
  year={2023},
  eprint={2304.11085},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  doi={10.48550/arXiv.2304.11085},
  url={https://arxiv.org/abs/2304.11085}
}

@inproceedings{rizzi2024soft,
    title = "Soft metrics for evaluation with disagreements: an assessment",
    author = "Rizzi, Giulia and
      Leonardelli, Elisa and
      Poesio, Massimo and
      Uma, Alexandra and
      Pavlovic, Maja and
      Paun, Silviu and
      Rosso, Paolo and
      Fersini, Elisabetta",
    editor = "Abercrombie, Gavin and
      Basile, Valerio and
      Bernadi, Davide and
      Dudy, Shiran and
      Frenda, Simona and
      Havens, Lucy and
      Tonelli, Sara",
    booktitle = "Proceedings of the 3rd Workshop on Perspectivist Approaches to NLP (NLPerspectives) @ LREC-COLING 2024",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.nlperspectives-1.9/",
    pages = "84--94"
}

@inproceedings{rodrigues2018deep,
  title     = {Deep Learning from Crowds},
  author    = {Rodrigues, Filipe and Pereira, Francisco C.},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-18)},
  volume    = {32},
  number    = {1},
  pages     = {1611--1618},
  year      = {2018},
  doi       = {10.1609/aaai.v32i1.11506}
}

@inproceedings{rottger2022two,
    title = "Two Contrasting Data Annotation Paradigms for Subjective {NLP} Tasks",
    author = {R{\"o}ttger, Paul and Vidgen, Bertie and Hovy, Dirk and Pierrehumbert, Janet},
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.13/",
    doi = "10.18653/v1/2022.naacl-main.13",
    pages = "175--190"
}

@inproceedings{ruiz2025bon,
    title = "{B}o{N} Appetit Team at {L}e{W}i{D}i-2025: Best-of-N Test-time Scaling Can Not Stomach Annotation Disagreements (Yet)",
    author = "Ruiz, Tomas and Peng, Siyao and Plank, Barbara and Schwemmer, Carsten",
    editor = "Abercrombie, Gavin and Basile, Valerio and Frenda, Simona and Tonelli, Sara and Dudy, Shiran",
    booktitle = "Proceedings of the The 4th Workshop on Perspectivist Approaches to NLP",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.nlperspectives-1.14/",
    doi = "10.18653/v1/2025.nlperspectives-1.14",
    pages = "153--170",
    ISBN = "979-8-89176-350-0"
}

@misc{saito2023verbosity,
  title        = {Verbosity Bias in Preference Labeling by Large Language Models},
  author       = {Saito, Keita and Wachi, Akifumi and Wataoka, Koki and Akimoto, Youhei},
  year         = {2023},
  eprint       = {2310.10076},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  note         = {Presented at the NeurIPS 2023 Workshop on Instruction Tuning and Instruction Following},
  url          = {https://arxiv.org/abs/2310.10076}
}

@misc{salinas2025tuning,
  title        = {Tuning LLM Judge Design Decisions for 1/1000 of the Cost},
  author       = {Salinas, David and Swelam, Omar and Hutter, Frank},
  year         = {2025},
  eprint       = {2501.17178},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  note         = {Accepted as a poster at ICML 2025},
  url          = {https://arxiv.org/abs/2501.17178}
}

@inproceedings{sandri2023why,
    title = "Why Don't You Do It Right? Analysing Annotators' Disagreement in Subjective Tasks",
    author = "Sandri, Marta and Leonardelli, Elisa and Tonelli, Sara and Jezek, Elisabetta",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    pages = "2428--2441",
    doi = "10.18653/v1/2023.eacl-main.178",
    url = "https://aclanthology.org/2023.eacl-main.178/"
}

@misc{schroeder2024reliability,
  title        = {Reliability of Topic Modeling},
  author       = {Schroeder, Kayla and Wood-Doughty, Zach},
  year         = {2024},
  eprint       = {2410.23186},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2410.23186}
}

@inproceedings{sclar2024quantifying,
  title={Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting},
  author={Sclar, Melanie and Choi, Yejin and Tsvetkov, Yulia and Suhr, Alane},
  booktitle={The Twelfth International Conference on Learning Representations (ICLR)},
  year={2024},
  eprint={2310.11324},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2310.11324}
}

@misc{shen2026rethinking,
  title={Rethinking Rubric Generation for Improving LLM Judge and Reward Modeling for Open-ended Tasks},
  author={Shen, William F. and Qiu, Xinchi and Whitehouse, Chenxi and Alazraki, Lisa and Goel, Shashwat and Barbieri, Francesco and Willi, Timon and Mathur, Akhil and Leontiadis, Ilias},
  year={2026},
  eprint={2602.05125},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  doi={10.48550/arXiv.2602.05125}
}

@misc{song2024can,
  title={Can Many-Shot In-Context Learning Help LLMs as Evaluators? A Preliminary Empirical Study},
  author={Song, Mingyang and Zheng, Mao and Luo, Xuan},
  year={2024},
  eprint={2406.11629},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  note={Published in Proceedings of COLING 2025 (Main Conference), ACL Anthology 2025.coling-main.548},
  url={https://arxiv.org/abs/2406.11629}
}

@inproceedings{sprague2025cot,
  title     = {To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning},
  author    = {Sprague, Zayne and Yin, Fangcong and Rodriguez, Juan Diego and Jiang, Dongwei and Wadhwa, Manya and Singhal, Prasann and Zhao, Xinyu and Ye, Xi and Mahowald, Kyle and Durrett, Greg},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  eprint    = {2409.12183},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url       = {https://arxiv.org/abs/2409.12183}
}

@misc{stolwijk2025generative,
  title        = {Are generative AI text annotations systematically biased?},
  author       = {Stolwijk, Sjoerd B. and Boukes, Mark and Trilling, Damian},
  year         = {2025},
  eprint       = {2512.08404},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2512.08404}
}

@misc{stureborg2024large,
  title        = {Large Language Models are Inconsistent and Biased Evaluators},
  author       = {Stureborg, Rickard and Alikaniotis, Dimitris and Suhara, Yoshi},
  year         = {2024},
  eprint       = {2405.01724},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2405.01724}
}

@inproceedings{tam2024let,
    title = "Let Me Speak Freely? A Study On The Impact Of Format Restrictions On Large Language Model Performance",
    author = "Tam, Zhi Rui and Wu, Cheng-Kuang and Tsai, Yi-Lin and Lin, Chieh-Yen and Lee, Hung-yi and Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track",
    month = nov,
    year = "2024",
    address = "Miami, Florida, US",
    publisher = "Association for Computational Linguistics",
    pages = "1218--1236",
    doi = "10.18653/v1/2024.emnlp-industry.91"
}

@inproceedings{tan2024large,
    title = "Large Language Models for Data Annotation and Synthesis: A Survey",
    author = "Tan, Zhen and Li, Dawei and Wang, Song and Beigi, Alimohammad and Jiang, Bohan and Bhattacharjee, Amrita and Karami, Mansooreh and Li, Jundong and Cheng, Lu and Liu, Huan",
    editor = "Al-Onaizan, Yaser and Bansal, Mohit and Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.54/",
    doi = "10.18653/v1/2024.emnlp-main.54",
    pages = "930--957"
}

@inproceedings{tian2023just,
  title = {Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback},
  author = {Tian, Katherine and Mitchell, Eric and Zhou, Allan and Sharma, Archit and Rafailov, Rafael and Yao, Huaxiu and Finn, Chelsea and Manning, Christopher D.},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages = {5433--5442},
  year = {2023},
  publisher = {Association for Computational Linguistics},
  address = {Singapore},
  doi = {10.18653/v1/2023.emnlp-main.330},
  url = {https://aclanthology.org/2023.emnlp-main.330/}
}

@misc{tornberg2023chatgpt,
  title={ChatGPT-4 Outperforms Experts and Crowd Workers in Annotating Political Twitter Messages with Zero-Shot Learning},
  author={T{\"o}rnberg, Petter},
  year={2023},
  eprint={2304.06588},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  howpublished={arXiv:2304.06588}
}

@misc{verga2024replacing,
  title={Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models},
  author={Verga, Pat and Hofst{\"a}tter, Sebastian and Althammer, Sophia and Su, Yixuan and Piktus, Aleksandra and Arkhangorodsky, Arkady and Xu, Minjie and White, Naomi and Lewis, Patrick},
  year={2024},
  eprint={2404.18796},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2404.18796}
}

@inproceedings{wang2021want,
    title = "Want To Reduce Labeling Cost? {GPT}-3 Can Help",
    author = "Wang, Shuohang and Liu, Yang and Xu, Yichong and Zhu, Chenguang and Zeng, Michael",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    year = "2021",
    pages = "4195--4205",
    publisher = "Association for Computational Linguistics",
    doi = "10.18653/v1/2021.findings-emnlp.354"
}

@misc{wang2023large,
  title={Large Language Models are not Fair Evaluators},
  author={Peiyi Wang and Lei Li and Liang Chen and Zefan Cai and Dawei Zhu and Binghuai Lin and Yunbo Cao and Lingpeng Kong and Qi Liu and Tianyu Liu and Zhifang Sui},
  year={2023},
  eprint={2305.17926},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  note={Published at ACL 2024, pp. 9440--9450, DOI: 10.18653/v1/2024.acl-long.511}
}

@inproceedings{xiong2024can,
  title={Can {LLM}s Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in {LLM}s},
  author={Xiong, Miao and Hu, Zhiyuan and Lu, Xinyang and Li, Yifei and Fu, Jie and He, Junxian and Hooi, Bryan},
  booktitle={The Twelfth International Conference on Learning Representations (ICLR)},
  year={2024},
  eprint={2306.13063},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2306.13063}
}

@misc{xu2024benchmark,
  title        = {Benchmark Data Contamination of Large Language Models: A Survey},
  author       = {Xu, Cheng and Guan, Shuhao and Greene, Derek and Kechadi, M-Tahar},
  year         = {2024},
  eprint       = {2406.04244},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2406.04244}
}

@misc{yamauchi2025empirical,
  title={An Empirical Study of LLM-as-a-Judge: How Design Choices Impact Evaluation Reliability},
  author={Yamauchi, Yusuke and Yano, Taro and Oyamada, Masafumi},
  year={2025},
  eprint={2506.13639},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  doi={10.48550/arXiv.2506.13639},
  url={https://arxiv.org/abs/2506.13639}
}

@article{ye2024justice,
  title={Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge},
  author={Ye, Jiayi and Wang, Yanbo and Huang, Yue and Chen, Dongping and Zhang, Qihui and Moniz, Nuno and Gao, Tian and Geyer, Werner and Huang, Chao and Chen, Pin-Yu and Chawla, Nitesh V. and Zhang, Xiangliang},
  journal={arXiv preprint arXiv:2410.02736},
  year={2024},
  note={Published at ICLR 2025},
  url={https://arxiv.org/abs/2410.02736}
}

@inproceedings{zhang2014spectral,
  title     = {Spectral Methods Meet {EM}: A Provably Optimal Algorithm for Crowdsourcing},
  author    = {Zhang, Yuchen and Chen, Xi and Zhou, Dengyong and Jordan, Michael I.},
  booktitle = {Advances in Neural Information Processing Systems (NIPS) 27},
  year      = {2014},
  eprint    = {1406.3824},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML}
}

@inproceedings{zheng2023judging,
  title={Judging {LLM}-as-a-Judge with {MT}-Bench and Chatbot Arena},
  author={Zheng, Lianmin and Chiang, Wei-Lin and Sheng, Ying and Zhuang, Siyuan and Wu, Zhanghao and Zhuang, Yonghao and Lin, Zi and Li, Zhuohan and Li, Dacheng and Xing, Eric P. and Zhang, Hao and Gonzalez, Joseph E. and Stoica, Ion},
  booktitle={Advances in Neural Information Processing Systems 36 (NeurIPS 2023) Datasets and Benchmarks Track},
  year={2023},
  eprint={2306.05685},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@article{ziems2024can,
    title = {Can Large Language Models Transform Computational Social Science?},
    author = {Ziems, Caleb and Held, William and Shaikh, Omar and Chen, Jiaao and Zhang, Zhehao and Yang, Diyi},
    journal = {Computational Linguistics},
    volume = {50},
    number = {1},
    pages = {237--291},
    month = {March},
    year = {2024},
    publisher = {MIT Press},
    doi = {10.1162/coli_a_00502}
}

```

### Citations needing manual review

**Claim possibly contradicted by the source (2)** — paper is real, but our text's attribution may be wrong:

- `bellibatlu2026judgesense` — JudgeSense 2026
- `polo2024tinybenchmarks` — Polo et al. 2024

**Partial claim-match (20)** — paper located, attributed claim only partly supported; spot-check before relying on the exact number/wording:

- `agrawal2025gepa`; `angelopoulos2023prediction`; `audinet2025benchmarking`; `bavaresco2025llms`; `chakrabarty2024art`; `chhun2024do`; `choi2026diagnosing`; `guerdan2025validating`; `haldar2025rating`; `halterman2025codebook`; `jaroslawicz2025how`; `kenton2024scalable`; `lin2022teaching`; `liu2023lost`; `liu2025openrubrics`; `mukherjee2025meta`; `platanios2017estimating`; `ruiz2025bon`; `sandri2023why`; `stureborg2024large`
