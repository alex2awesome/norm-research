# Prompt-optimization (GEPA vein): what happens to the prompt itself

*(2026-06-13 deep dive — 5 threads, 55 papers. Question: does anyone study the optimized PROMPT —
its length, bloat, structure, meaning, transfer — and is prompt length a studied variable? Built on
the annotation lit review + scaling notes (not re-derived). Connection to our plan in §5.)*

## 1. GEPA & the optimizer landscape

**What GEPA does** (Agrawal et al. 2025, arXiv:2507.19457, ICLR 2026 Oral): (1) *reflective NL
mutation* — diagnose failures in natural language, edit one module's instruction; (2) *Pareto
retention* — keep the frontier of candidates each winning ≥1 instance (+6.4–8.2 pts over greedy);
(3) *system-aware merge*. Instructions only, no demos. Beats GRPO avg +6% at up to 35× fewer
rollouts; beats MIPROv2 >10%.

**"Simple yet an Oral" — fair?** Mechanically lean: GEPA ≈ TextGrad reflective edits + a Pareto
frontier (its Table 3 ablation = +8.17 pts from Pareto retention alone). Lineage is explicit:
APE → OPRO → ProTeGi → Promptbreeder/EvoPrompt → TextGrad; **no part is new.** The real
contributions are *empirical*: (a) reflective NL beats RL at vastly lower rollout cost (against the
"RL is needed" default); (b) **GEPA actually inspects its output artifact** — rare in this field;
(c) it reverses Wan et al. 2024 on generalization. So "just reflection + a frontier" is fair on
mechanism-novelty, unfair on what was demonstrated.

| Optimizer | How it changes the prompt | Inspects output prompt? |
|---|---|---|
| GEPA (Agrawal 2025) | Reflective NL edit + Pareto frontier + merge | **Yes** — length (9.2× < MIPROv2), edge-case content, gen-gap |
| OPRO (Yang 2023) | LLM emits instruction from score-trajectory | Partly — fluent-but-unintuitive winners; no length |
| TextGrad/ProTeGi (Yuksekgonul 2024; Pryzant 2023) | Backprop NL "gradient" / minibatch critique | No |
| MIPROv2 (Opsahl-Ong 2024) | Bayesian-opt joint instruction + demos | No — its demo-bulk is GEPA's baseline |
| Promptbreeder/EvoPrompt (Fernando 2023; Guo 2023) | Evolutionary NL mutation/crossover | No |
| AutoPrompt/RLPrompt (Shin 2020; Deng 2022) | Gradient/RL over discrete tokens | Yes (negative) — gibberish, fixed length |

## 2. What happens to the prompt (the core answer)

**Known:** reflective/NL optimizers reliably **grow** the prompt — "from telling the model what to do
to coaching it how" (Decagon 2025) — accumulating format specs + domain best-practices + edge-case
patches absorbed from training trajectories (GEPA App. L: evolved prompts add Eisenstein's Criterion,
Lean-format warnings, a "Handling False Statements" protocol). So GEPA **does** produce instruction
bloat / edge-case stuffing — just *less total* than MIPROv2 (whose bulk is demos; GEPA ~9.2× shorter
while scoring higher). The named failure mode: **"prompt distributional overfitting"** (TextReg, Fu et
al. 2026 — "optimized prompts become longer, accumulate narrow sample-specific rules, generalize
poorly"); GRACE (Shi et al. 2025) calls it "monotonic length growth" → "bloated prompts that provide
no benefit—or actively harm." Mechanistic driver: the minibatch-patching edit rule (ProTeGi) *is*
structurally a bloat operator. Long prompts are sentence-fragile (Hsieh et al. 2023).

**Optimistic counter-evidence:** GEPA reports a **lower** gen-gap (test−val) for evolved *instructions*
than for *few-shot demos* (Fig 16), reversing Wan et al. 2024 — so on its benchmarks the instruction
artifact transfers rather than memorizing.

**The gap (stated plainly):** no optimizer paper — GEPA included — reports a **controlled per-iteration
length-vs-fidelity curve**; length appears only as an aggregate cost proxy. And **nobody asks whether
the added text recovers a meaningful construct vs exploits judge/benchmark quirks on subjective/no-gold
tasks** — the bloat literature measures verbosity and OOD accuracy, never construct fidelity of the
added clauses. The richest GEPA-specific bloat report is a **company blog** (Decagon 2025), not
peer-reviewed. **Verdict: the user's suspicion is confirmed — prompt-length-during-optimization as a
fidelity object is genuinely under-studied.**

## 3. Is prompt length a studied variable?

Yes — but almost entirely as **cost/efficiency**, and mostly 2025–26. (a) *Length-as-penalty in the
optimizer:* CAPO (Zehle 2025, token-cost as a Pareto term), Cost-aware APO (2507.15884), Nano-Capsulator
(Chuang 2024, length-in-reward, −81.4% length, still transferable), TextReg (Fu 2026, `|p|×(1−coverage)`
penalty, +11.8% OOD), Prompt-MII (2510.16932, 3–13× fewer tokens). (b) *Curve shape — non-monotone,
sweet-spot, flat-then-cliff:* **Decagon 2025 (GEPA in production): 50→500 samples = +75% length while
accuracy DROPS; 1,500-char cap = 4× compression at −0.8%; 500-char = −3% ⇒ ~80% of bloat is
non-load-bearing filler, "length constraints act as regularization."** Compression lit (other
direction): LLMLingua plateau-then-cliff ~20× (Jiang 2023); LLMLingua-2 safe 2–5× (Pan 2024); gist
tokens 26× (Mu 2023). **Dissent:** Madras 2025 — the generalization gap is controlled by **perplexity
(fluency), not length** — so "shorter = better-generalizing" isn't the only model.

## 4. Do optimized prompts mean anything / transfer?

**Incantation evidence is strong (method-dependent):**
- **Waywardness (Khashabi 2022):** an effective continuous prompt projects onto the definition of an
  *unrelated/contradictory* task within ~2% of optimal; worsens with size + length. Surface text
  decoupled from task solved.
- **Webson & Pavlick 2022 (strongest single source for the confound):** models learn just as fast with
  *irrelevant or misleading* templates, even at 175B; performance tracks target/answer words, not
  instruction meaning ⇒ **a high score under a rubric is not evidence the construct was articulated.**
- **Gibberish-that-works (AutoPrompt 2020; RLPrompt 2022; evil twins, Melamed 2023):** ungrammatical
  prompts work and **transfer across families** (>50%) ⇒ **transfer does NOT certify meaning.**

**Nuance:** transfer is method-dependent (soft prompts often *don't* port — SPoT, Su 2021; PromptBridge
2025 shows source→target drift). Optimistic pole: APE (Zhou 2022) and **Instruction Induction (Honovich
2022)** show NL optimization *can* recover human-meaningful, transferable instructions **on tasks with a
clean latent target** — and Honovich scores by *execution/behavioral equivalence*, not surface match
(the recover-vs-restate boundary). Open whether this holds for **subjective/taste tasks**. Theory says
the ambiguity is intrinsic: an instruction *locates* a latent concept in the prior (Xie 2021), and gains
can come from format/label-space cues (Min 2022) — a higher-scoring rubric need not match the construct.

## 5. Implications for us

**Does GEPA-optimizing a rubric measure construct articulability or prompt-hack the judge? Both are
live, and a raw score gain cannot distinguish them.** Exploit pole well-attested (Webson-Pavlick
misleading templates; evil-twin gibberish; Decagon edge-case memorization that *drops* test accuracy;
FormatSpread/Sclar 2023 format-only swings up to 76 pts). Recovery pole exists (APE/Honovich on
clean-latent; GEPA's lower gen-gap; Prompt-MII compact+transferable). For our **"best articulation"
frontier** this is a direct confound: a GEPA-optimized rubric that agrees better with the expert anchor
may be (a) articulating the construct, or (b) bloating with judge-pleasing edge-case/format clauses
tuned to *our* judge. If (b), the frontier is inflated by optimizer slack and is not a property of the
metric.

**The control that separates recovery from exploit** (causal/counterfactual feature intervention —
CPO/SCIE/DiCap, 2412.15314, 2507.19882): intervene separately on **construct-relevant** vs
**construct-irrelevant (format/length/anchor-phrasing)** features and test whether the agreement gain
**survives** (recovery) or **collapses** (exploit). Three converging checks:
1. **Cross-family / cross-corpus transfer** — re-score with a different judge family + fresh corpus.
   Survival = recovery; brittleness = exploit. (Necessary, NOT sufficient — incantations can transfer;
   pair with 2–3.)
2. **Counterfactual edits** — perturb only construct-irrelevant features (reformat, paraphrase, strip
   length without changing claims); gain should be invariant if construct-borne.
3. **Length-penalty arm** — length-capped (CAPO/Decagon-style) vs uncapped GEPA. If capped recovers
   comparable agreement (cf. Decagon 4×-at-−0.8%), the bloat was filler and the **capped frontier is
   the honest articulability estimate; the uncapped gap is optimizer slack.**

**Maps onto our existing plan:** length-penalty arm = the **E7 instruction-token-cap grid** + **T1**
optimizer-slack framing (repurpose CAPO's cost penalty as a no-gold *fidelity* outcome). Counterfactual
/cross-family battery = **E4 discriminant**. The "rubric as pointer into the judge's prior" diagnostic
(Xie 2021) = **E2** (`corr(full,stub)~1` ⇒ locating a concept, not transmitting content). Phrasing-
distribution reporting + execution-not-surface validation (Honovich) = guardrails on the **E0/E3**
baseline-vs-optimized comparison. **Net: GEPA on a no-gold subjective rubric is confounded by default;
the length-cap arm + counterfactual/transfer battery is what converts a score gain into a defensible
articulability claim. That exact study — length-vs-fidelity curve + recover-vs-exploit discriminant for
an LLM-judge rubric optimized toward an expert construct — does not yet exist in this literature.**


## References (auto-verified BibTeX, 2026-06-15)

> Extracted from this document and web-verified + independently audited by an automated fact-check pass (search → fetch → resolvable id; attributed claim checked against the located paper). 29 entries. Real located works; not hand-checked. See "needs manual review" for 0 contradicted-claim and 1 unlocatable/rejected items.

```bibtex
@article{agrawal2025gepa,
  title={GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning},
  author={Agrawal, Lakshya A. and Tan, Shangyin and Soylu, Dilara and Ziems, Noah and Khare, Rishi and Opsahl-Ong, Krista and Singhvi, Arnav and Shandilya, Herumb and Ryan, Michael J. and Jiang, Meng and Potts, Christopher and Sen, Koushik and Dimakis, Alexandros G. and Stoica, Ion and Klein, Dan and Zaharia, Matei and Khattab, Omar},
  journal={arXiv preprint arXiv:2507.19457},
  year={2025}
}

@article{chuang2024learning,
  title={Learning to Compress Prompt in Natural Language Formats},
  author={Chuang, Yu-Neng and Xing, Tianwei and Chang, Chia-Yuan and Liu, Zirui and Chen, Xun and Hu, Xia},
  journal={arXiv preprint arXiv:2402.18700},
  year={2024}
}

@inproceedings{deng2022rlprompt,
  title={RLPrompt: Optimizing Discrete Text Prompts with Reinforcement Learning},
  author={Deng, Mingkai and Wang, Jianyu and Hsieh, Cheng-Ping and Wang, Yihan and Guo, Han and Shu, Tianmin and Song, Meng and Xing, Eric P. and Hu, Zhiting},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2022},
  eprint={2205.12548},
  archivePrefix={arXiv}
}

@article{fernando2023promptbreeder,
  title={Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution},
  author={Fernando, Chrisantha and Banarse, Dylan and Michalewski, Henryk and Osindero, Simon and Rockt{\"a}schel, Tim},
  journal={arXiv preprint arXiv:2309.16797},
  year={2023}
}

@misc{fu2026textreg,
  title={TextReg: Mitigating Prompt Distributional Overfitting via Regularized Text-Space Optimization},
  author={Fu, Lucheng and Yu, Ye and Wang, Yiyang and Jin, Yiqiao and Jin, Haibo and Prakash, B. Aditya and Wang, Haohan},
  year={2026},
  eprint={2605.21318},
  archivePrefix={arXiv}
}

@article{guo2023evoprompt,
  title={EvoPrompt: Connecting LLMs with Evolutionary Algorithms Yields Powerful Prompt Optimizers},
  author={Guo, Qingyan and Wang, Rui and Guo, Junliang and Li, Bei and Song, Kaitao and Tan, Xu and Liu, Guoqing and Bian, Jiang and Yang, Yujiu},
  journal={arXiv preprint arXiv:2309.08532},
  year={2023}
}

@article{honovich2022instruction,
  title={Instruction Induction: From Few Examples to Natural Language Task Descriptions},
  author={Or Honovich and Uri Shaham and Samuel R. Bowman and Omer Levy},
  year={2022},
  eprint={2205.10782},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@article{hsieh2023longprompts,
  title={Automatic Engineering of Long Prompts},
  author={Hsieh, Cho-Jui and Si, Si and Yu, Felix X. and Dhillon, Inderjit S.},
  year={2023},
  eprint={2311.10117},
  archivePrefix={arXiv},
  note={Later published in Findings of ACL 2024}
}

@article{jiang2023llmlingua,
  title={LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models},
  author={Jiang, Huiqiang and Wu, Qianhui and Lin, Chin-Yew and Yang, Yuqing and Qiu, Lili},
  journal={arXiv preprint arXiv:2310.05736},
  year={2023}
}

@inproceedings{khashabi2022prompt,
  title={Prompt Waywardness: The Curious Case of Discretized Interpretation of Continuous Prompts},
  author={Daniel Khashabi and Xinxi Lyu and Sewon Min and Lianhui Qin and Kyle Richardson and Sean Welleck and Hannaneh Hajishirzi and Tushar Khot and Ashish Sabharwal and Sameer Singh and Yejin Choi},
  year={2022},
  booktitle={Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL)},
  eprint={2112.08348},
  archivePrefix={arXiv}
}

@misc{madras2025prompts,
  title={Prompts Generalize with Low Data: Non-vacuous Generalization Bounds for Optimizing Prompts with More Informative Priors},
  author={David Madras and Joshua Safyan and Qiuyi (Richard) Zhang},
  year={2025},
  eprint={2510.08413},
  archivePrefix={arXiv}
}

@misc{melamed2023prompts,
  title={Prompts have evil twins},
  author={Rimon Melamed and Lucas H. McCabe and Tanay Wakhare and Yejin Kim and H. Howie Huang and Enric Boix-Adsera},
  year={2023},
  eprint={2311.07064},
  archivePrefix={arXiv}
}

@article{min2022rethinking,
  title={Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?},
  author={Sewon Min and Xinxi Lyu and Ari Holtzman and Mikel Artetxe and Mike Lewis and Hannaneh Hajishirzi and Luke Zettlemoyer},
  year={2022},
  eprint={2202.12837},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@article{mu2023learning,
  title={Learning to Compress Prompts with Gist Tokens},
  author={Mu, Jesse and Li, Xiang Lisa and Goodman, Noah},
  journal={arXiv preprint arXiv:2304.08467},
  year={2023}
}

@article{opsahlong2024mipro,
  title={Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs},
  author={Opsahl-Ong, Krista and Ryan, Michael J. and Purtell, Josh and Broman, David and Potts, Christopher and Zaharia, Matei and Khattab, Omar},
  journal={arXiv preprint arXiv:2406.11695},
  year={2024}
}

@article{pan2024llmlingua2,
  title={LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression},
  author={Pan, Zhuoshi and Wu, Qianhui and Jiang, Huiqiang and Xia, Menglin and Luo, Xufang and Zhang, Jue and Lin, Qingwei and R{\"u}hle, Victor and Yang, Yuqing and Lin, Chin-Yew and Zhao, H. Vicky and Qiu, Lili and Zhang, Dongmei},
  journal={arXiv preprint arXiv:2403.12968},
  year={2024}
}

@article{pryzant2023apo,
  title={Automatic Prompt Optimization with "Gradient Descent" and Beam Search},
  author={Pryzant, Reid and Iter, Dan and Li, Jerry and Lee, Yin Tat and Zhu, Chenguang and Zeng, Michael},
  journal={arXiv preprint arXiv:2305.03495},
  year={2023}
}

@article{sclar2023quantifying,
  title={Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting},
  author={Melanie Sclar and Yejin Choi and Yulia Tsvetkov and Alane Suhr},
  year={2023},
  eprint={2310.11324},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@inproceedings{shi2025grace,
  title={No Loss, No Gain: Gated Refinement and Adaptive Compression for Prompt Optimization},
  author={Shi, Wenhang and Chen, Yiren and Bian, Shuqing and Zhang, Xinyi and Tang, Kai and Hu, Pengfei and Zhao, Zhe and Lu, Wei and Du, Xiaoyong},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025},
  eprint={2509.23387},
  archivePrefix={arXiv}
}

@inproceedings{shin2020autoprompt,
  title={AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts},
  author={Shin, Taylor and Razeghi, Yasaman and Logan IV, Robert L. and Wallace, Eric and Singh, Sameer},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2020},
  eprint={2010.15980},
  archivePrefix={arXiv}
}

@misc{su2021transferability,
  title={On Transferability of Prompt Tuning for Natural Language Processing},
  author={Yusheng Su and Xiaozhi Wang and Yujia Qin and Chi-Min Chan and Yankai Lin and Huadong Wang and Kaiyue Wen and Zhiyuan Liu and Peng Li and Juanzi Li and Lei Hou and Maosong Sun and Jie Zhou},
  year={2021},
  eprint={2111.06719},
  archivePrefix={arXiv}
}

@article{wan2024teach,
  title={Teach Better or Show Smarter? On Instructions and Exemplars in Automatic Prompt Optimization},
  author={Wan, Xingchen and Sun, Ruoxi and Nakhost, Hootan and Ar{\i}k, Sercan {\"O}.},
  journal={arXiv preprint arXiv:2406.15708},
  year={2024}
}

@article{wang2025promptbridge,
  title={PromptBridge: Cross-Model Prompt Transfer for Large Language Models},
  author={Yaxuan Wang and Quan Liu and Zhenting Wang and Zichao Li and Wei Wei and Yang Liu and Yujia Bao},
  year={2025},
  eprint={2512.01420},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@inproceedings{webson2022prompts,
  title={Do Prompt-Based Models Really Understand the Meaning of Their Prompts?},
  author={Albert Webson and Ellie Pavlick},
  year={2022},
  booktitle={Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL)},
  eprint={2109.01247},
  archivePrefix={arXiv}
}

@article{xie2021explanation,
  title={An Explanation of In-context Learning as Implicit Bayesian Inference},
  author={Sang Michael Xie and Aditi Raghunathan and Percy Liang and Tengyu Ma},
  year={2021},
  eprint={2111.02080},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@article{yang2023llmoptimizers,
  title={Large Language Models as Optimizers},
  author={Yang, Chengrun and Wang, Xuezhi and Lu, Yifeng and Liu, Hanxiao and Le, Quoc V. and Zhou, Denny and Chen, Xinyun},
  journal={arXiv preprint arXiv:2309.03409},
  year={2023}
}

@article{yuksekgonul2024textgrad,
  title={TextGrad: Automatic "Differentiation" via Text},
  author={Yuksekgonul, Mert and Bianchi, Federico and Boen, Joseph and Liu, Sheng and Huang, Zhi and Guestrin, Carlos and Zou, James},
  journal={arXiv preprint arXiv:2406.07496},
  year={2024}
}

@article{zehle2025capo,
  title={CAPO: Cost-Aware Prompt Optimization},
  author={Zehle, Tom and Schlager, Moritz and Hei{\ss}, Timo and Feurer, Matthias},
  journal={arXiv preprint arXiv:2504.16005},
  year={2025}
}

@misc{zhou2022large,
  title={Large Language Models Are Human-Level Prompt Engineers},
  author={Yongchao Zhou and Andrei Ioan Muresanu and Ziwen Han and Keiran Paster and Silviu Pitis and Harris Chan and Jimmy Ba},
  year={2022},
  eprint={2211.01910},
  archivePrefix={arXiv}
}

```

### Citations needing manual review

**Could not be located / rejected by audit (1)**:

- Decagon 2025 — audit reject_mismatch: The blog post exists at the stated URL with exact title 'Optimizing GEPA for production: A test-driven approach to promp

**Partial claim-match (9)** — spot-check exact numbers/wording:

- `deng2022rlprompt`; `jiang2023llmlingua`; `madras2025prompts`; `melamed2023prompts`; `pan2024llmlingua2`; `shi2025grace`; `shin2020autoprompt`; `wan2024teach`; `wang2025promptbridge`

