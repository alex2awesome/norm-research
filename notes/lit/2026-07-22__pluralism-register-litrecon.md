# Lit recon: pluralism, register, and LLM-judge wording sensitivity

**Date:** 2026-07-22
**Scope:** literature reconnaissance for the "register of evaluation criteria" program — does the same evaluative concept, expressed under different names/phrasings, change an LLM judge's scores or variance? Sweep covers value pluralism, perspectivism/annotator disagreement, LLM-as-judge prompt sensitivity, steerable pluralism ("whose values"), and register/formality effects.
**Method:** 5 parallel search agents (one per angle) via WebSearch/WebFetch, verified against arXiv abstract pages before inclusion. All citations below were independently re-checked (abstract/HTML page resolves, author list and headline finding match).

---

## 1. Directly on point: criterion-wording / framing → LLM-judge score or verdict variance

These are the papers that actually manipulate the *surface form* of an evaluation criterion or judge prompt (holding the underlying construct/content fixed) and measure a change in judge output. This is the closest existing evidence to the register hypothesis.

| Paper | Authors | Year/Venue | Finding |
|---|---|---|---|
| **Quantifying the Statistical Effect of Rubric Modifications on Human-Autorater Agreement** (arXiv:2605.06283) | Huynh, Gomez, Deviyani, Shelby, Bigham, Diaz | 2026, preprint | Measures how rubric rewording/restructuring shifts human vs. LLM-autorater score agreement across holistic and analytic (decomposed-criteria) judgments. Rubric edits that add representative examples/context and reduce positional bias raise agreement; higher rubric complexity and conservative aggregation lower it. LLM autoraters track surface rubric changes more than human raters do — the strongest direct empirical anchor found for "wording of the criterion, not just its content, moves the judge." |
| **When Wording Steers the Evaluation: Framing Bias in LLM Judges** (arXiv:2601.13537) | Yerin Hwang, Dongryeol Lee, Taegwan Kang, Minwoo Lee, Kyomin Jung | 2026, preprint | Tests predicate-positive vs. predicate-negative phrasings of *identical* evaluative content across 14 LLM judges on truthfulness, jailbreak, toxicity, grammaticality. Universal framing-bias vulnerability (best model still ~5.7% inconsistency); model families show distinct acquiescence/rejection tendencies. Same construct → systematically different verdict purely from linguistic framing. |
| **JudgeSense: A Benchmark for Prompt Sensitivity in LLM-as-a-Judge Systems** (arXiv:2604.23478) | Rohith Reddy Bellibatlu, Edward Raff, Wenbin Zhang | 2026, preprint | Hand-validated benchmark of semantically-equivalent prompt-paraphrase pairs across factuality, coherence, relevance, preference. Coherence (a holistic-quality criterion, closest analog to "register"-sensitive constructs) is most paraphrase-sensitive; factuality most stable. Larger/newer models are *not* more consistent. |
| **How Sensitive Are Safety Benchmarks to Judge Configuration Choices?** (arXiv:2604.24074) | — | 2026, preprint | Quantifies wording effects directly: judge-prompt wording alone shifts measured harmful-response rates by up to 24.2 pp on one model; surface rewording within the same condition swings results up to 20.1 pp; full configuration space reaches 40-45% swings. Best quantitative magnitude estimate for "wording alone moves judge output." |
| **Flaw or Artifact? Rethinking Prompt Sensitivity in Evaluating LLMs** (arXiv:2509.01790) | — | 2025, preprint | Changing a semantically-equivalent prompt template flips the majority-preferred response in 25% of cases on a 10-question subset (avg. 13.4pp flip-rate change) — shows wording sensitivity isn't cosmetic, it changes the modal verdict. |
| **Framing Matters: Addressing Framing Sensitivity in Decision-Making through Behaviorally-Grounded Value Alignment** (arXiv:2605.28188) | Seojin Hwang, Minju Kim, Junhyuk Choi, JeongHyun Park, Hwanhee Lee | 2026, preprint | "Fragile" benchmark: fact-identical inputs reframed via value-tinted narration/temporal slice/narrative vividness cause a 28.6% avg decision-flip rate. Operates on framing of the *input/decision* rather than renaming of a *judge-facing criterion*, but is the closest hit inside the value-pluralism literature proper to a naming/register effect. |
| **Diagnosing the Reliability of LLM-as-a-Judge via Item Response Theory** (arXiv:2602.00521) | — | 2026, preprint | Different criteria (e.g., Coherence vs. Helpfulness in HelpSteer-2) show very different reliability depending on how the prompt specifies them — relevant to "same underlying construct, different name → different reliability," though it compares different named constructs rather than paraphrases of one construct. |

**Read-across:** the LLM-as-judge prompt-sensitivity literature has, in the last ~12 months, converged on the claim that judge verdicts are unstable under semantically-equivalent rewording of the *evaluation prompt itself* — and Huynh et al. is close to the register program's exact design (rubric wording as the manipulated variable, LLM vs. human autorater as the outcome-comparison axis). None of these, however, isolate *naming* (giving the same rubric item a different label, e.g. "helpfulness" vs. "usefulness" vs. community-specific jargon) as a clean single-factor manipulation — they vary phrasing/framing/paraphrase more broadly (sentence-level rewording, predicate polarity, template structure).

---

## 2. Adjacent: value pluralism in AI alignment

Establishes that "the same evaluative concept" is inherently contested/plural, but does not test wording→judge-variance directly.

| Paper | Authors | Year/Venue | Relevance |
|---|---|---|---|
| **Value Kaleidoscope: Engaging AI with Pluralistic Human Values, Rights, and Duties** (arXiv:2309.00779) | Sorensen, Jiang, Hwang, Levine, Pyatkin, West, Dziri, Lu, Rao, Bhagavatula, Sap, Tasioulas, Choi | AAAI 2024, 38(18):19937-19947 | Introduces ValuePrism (218K values/rights/duties tied to 31K situations) and Kaleido, a model assessing relevance/valence of values in context. Shows values are plural and context-dependent, but doesn't manipulate naming/wording as the IV. |
| **Position: A Roadmap to Pluralistic Alignment** (arXiv:2402.05070) | Sorensen, Moore, Fisher, Gordon, Mireshghallah, Rytting, Ye, Jiang, Lu, Dziri, Althoff, Choi | ICML 2024, PMLR 235:46280-46302 | Canonical taxonomy: Overton pluralism (spectrum of reasonable responses), steerable pluralism (steer to a perspective), distributional pluralism (calibrated to a population). Provides the field's standard vocabulary for "pluralism," itself an instance of naming proliferation for related constructs. Definitional backbone, not an empirical wording-effect result. |
| **Benchmarking Overton Pluralism in LLMs** (arXiv:2512.01351) | — | 2025/2026, preprint | Extends a six-type pluralism taxonomy (pluralistic models × pluralistic benchmarks). Shows how the field keeps subdividing "pluralism" into finer-grained named sub-constructs — a meta-illustration of the register phenomenon at the level of research vocabulary, not judge behavior. |
| **Isolating LLM Lexical Bias: A Curation-Free Triangulated Metric for Preference-Stage Learning** (arXiv:2606.00334) | Xiaoyang Ming, Jose Hernandez, Thomas Stephan Juzek | 2026, preprint | Triangulated Preference Shift score isolating lexical/stylistic drift ("language of prestige") introduced by RLHF. Studies word-choice drift from *training*, not whether relabeling a judge's rubric changes its score of a fixed input — methodological cousin (triangulation design), not a direct hit. |

---

## 3. Adjacent: perspectivism and disagreement-as-signal in NLP annotation

Reframes annotator disagreement from noise to signal; the newest strand (Huynh et al., listed in §1) bridges this literature into judge-wording sensitivity directly.

| Paper | Authors | Year/Venue | Relevance |
|---|---|---|---|
| **We Need to Consider Disagreement in Evaluation** / Perspectivist Data Manifesto (PDAI) | Basile, Fell, Fornaciari, Hovy, Paun, Plank, Poesio, Uma | 2021, ACL workshop; manifesto at pdai.info | Foundational "strong perspectivism" position: disagreement is information, not error; evaluation should preserve annotator-level (non-aggregated) judgments. Frame-setting for the whole angle. |
| **Learning from Disagreement: A Survey** | Uma, Fornaciari, Hovy, Paun, Plank, Poesio | JAIR 72 (2021), 1385-1470 | Canonical survey cataloguing NLP/CV tasks with documented human disagreement and methods (soft labels, multi-annotator models) for learning from unaggregated judgments. |
| **The "Problem" of Human Label Variation: On Ground Truth in Data, Modeling and Evaluation** | Barbara Plank | EMNLP 2022 | Reframes disagreement as "Human Label Variation" (HLV); catalogues datasets with unaggregated labels. (Note: this is the paper that best matches the "Plank neglected-disagreement" reference from the task brief — a differently-titled paper by that exact name was not located.) |
| **Agreeing to Disagree: Annotating Offensive Language Datasets with Annotators' Disagreement** + follow-up **Why Don't You Do It Right? Analysing Annotators' Disagreement in Subjective Tasks** | Leonardelli, Menini, Aprosio, Guerini, Tonelli; Sandri, Leonardelli, Tonelli, Jezek | EMNLP 2021; EACL 2023 | Empirical demonstration that disagreement in subjective tasks (offensive language) reflects genuine interpretive variation, not annotator error. Co-organized SemEval-2023 Task 11 "Learning With Disagreements" (LeWiDi; arXiv:2304.14803), with a third edition LeWiDi-2025 at NLPerspectives (arXiv:2510.08460). |
| **The Impact of Annotator Personas on LLM Behavior Across the Perspectivism Spectrum** (arXiv:2508.17164) | — | 2025, preprint | Studies how persona/framing instructions given to an LLM annotator shift its outputs along weak-vs-strong perspectivist axes — a naming/framing effect, though operationalized as persona rather than pure criterion rewording. |
| **From Self to Other: Evaluating Demographic Perspective-Taking in LLM Hate Speech Annotation** (arXiv:2606.06266); **Can LLMs Evaluate What They Cannot Annotate? Revisiting LLM Reliability in Hate Speech Detection** (arXiv:2512.09662) | — | 2025/2026, preprints | Both find LLMs capture the modal/majority label but fail to reproduce the shape of human disagreement/uncertainty distributions under persona/instruction prompting. |

---

## 4. Adjacent: steerable pluralism / "whose values"

| Paper | Authors | Year/Venue | Relevance |
|---|---|---|---|
| **Whose Opinions Do Language Models Reflect?** | Santurkar, Durmus, Ladhak, Chiang, Liang, Hashimoto | ICML 2023, PMLR 202:29971-30004 | OpinionQA benchmark vs. 60 US demographic groups; substantial misalignment persists even after explicit steering toward a group — naming/persona framing alone doesn't guarantee value transfer. Foundational for "whose values does the judge actually reflect." |
| **Steerable Pluralism: Pluralistic Alignment via Few-Shot Comparative Regression** (arXiv:2508.08509) | — | 2025, preprint | Operationalizes steerable pluralism as a reward-model design problem; LLM scores candidates against varying attribute profiles — mechanism showing differently-specified value attributes change downstream scoring, analogous to differently-named rubric criteria. |
| **Can Persona-Prompted LLMs Emulate Subgroup Values? An Empirical Analysis of Generalisability and Fairness in Cultural Alignment** (arXiv:2604.12851) | — | 2026, preprint | Tests whether persona prompting reproduces subgroup value distributions — mixed/negative results, relevant to whether "asking as X" reliably changes outputs the way community-specific rubric language might. |
| **The Ghost in the Machine has an American accent: value conflict in GPT-3** (arXiv:2203.07785) | — | 2022, preprint | Early evidence that default (unsteered) LLM value expression carries a specific cultural "accent" — motivates why cross-community register differences matter for judge-neutrality claims. |

---

## 5. Adjacent: register/formality/politeness of prompts (not judge-specific)

| Paper | Authors | Year/Venue | Relevance |
|---|---|---|---|
| **Should We Respect LLMs? A Cross-Lingual Study on the Influence of Prompt Politeness on LLM Performance** (arXiv:2402.14531) | Ziqi Yin, Hao Wang, Kaito Horio, Daisuke Kawahara, Satoshi Sekine | ACL SICon 2024 workshop | Politeness levels tested across English/Chinese/Japanese; impolite prompts often underperform but overly polite prompts don't reliably win — optimal register is language-specific, cautioning against a monotonic formality→quality story. |
| **Mind Your Tone: Investigating How Prompt Politeness Affects LLM Accuracy** (arXiv:2510.04950) | — | 2025, preprint | 250 prompts (50 questions × 5 tone variants) on MCQs; rudest prompts scored *highest* accuracy (84.8% vs. 80.8% very polite) — contradicts the paper above, useful as a "findings don't replicate across studies" caveat. |
| **Does Tone Change the Answer? Evaluating Prompt Politeness Effects on Modern LLMs: GPT, Gemini, and LLaMA** (arXiv:2512.12812) | — | 2025/2026, preprint | Replicates tone-variation-on-MMLU design across three model families; effects are model- and language-dependent in direction/magnitude. |
| **Same Question, Different Words: A Latent Adversarial Framework for Prompt Robustness** (arXiv:2503.01345) | — | 2025, preprint | 10 GPT-4-generated, human-reviewed paraphrases per AlpacaEval query; up to 45% performance swings across semantically-equivalent surface forms (e.g., Llama-2-70B-chat range 0.094-0.549). General (non-judge) evidence that surface wording alone drives large output-quality variance. |
| **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** | Zheng, Chiang, Sheng, et al. | NeurIPS D&B 2023, arXiv:2306.05685 | Foundational LLM-judge paper naming position bias, verbosity bias, self-enhancement bias as canonical failure modes. Essential baseline even though wording-sensitivity is secondary to these bias findings. |
| **The Coin Flip Judge? Reliability and Bias in LLM-as-a-Judge Evaluation** (arXiv:2606.13685) | Yagubyan | 2026, preprint | Variance-decomposition study: surface rewording of criteria/prompts sometimes drives reliability down to near-chance ("coin flip"); situates wording sensitivity alongside positional/majority biases as a top reliability threat. |

---

## Gap statement

Across all five search angles, no paper isolates **pure criterion renaming** — giving an LLM judge the *same* rubric item under two or more different names/labels for the same underlying construct (e.g., calling a criterion "helpfulness" vs. "usefulness" vs. a community-specific term for the same quality standard) while holding everything else (definition, examples, scale, task) fixed — as its manipulated variable. The nearest work clusters into two groups: (1) rubric/prompt **paraphrase and framing** studies (Huynh et al. 2605.06283; Hwang et al. "When Wording Steers the Evaluation" 2601.13537; JudgeSense 2604.23478; "Flaw or Artifact?" 2509.01790), which reword sentences or flip predicate polarity around a fixed construct and do find substantial judge-verdict instability, and (2) **persona/steering** studies (Santurkar et al.; persona-prompted subgroup-emulation work), which reframe *who* the judge is speaking as/for rather than *what the criterion is called*. Neither line has tested cross-community lexical variants of one evaluative concept (the "register" hypothesis proper) as a controlled single-factor manipulation on judge score/variance — this is a genuine, currently open gap that the planned register program would be first to fill directly.
