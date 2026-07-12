# The Irreducible Term *E*: how scaling-law literatures quantify a theoretical ceiling on latent ability

*(2026-06-13 deep dive — 5 parallel literature threads, 56 papers. Answers: how is the Chinchilla
`E` term — the Bayes/entropy floor a perfect model with unlimited data/params can't beat —
actually estimated, and is it identified or extrapolated. Connection to our `τ(E)`/dense-ceiling
in the final section.)*

## 1. What *E* is, and its analogs

In `L̂(N,D) ≜ E + A/Nᵅ + B/Dᵝ`, the constant *E* is the loss a perfect model with unlimited
parameters and data still cannot beat. Hoffmann 2022 (Sec. 3.3): *E* "captures the loss for an
ideal generative process on the data distribution, and should correspond to the entropy of natural
text"; App. D.2 identifies it as "the Bayes risk, i.e. the minimal loss achievable for next-token
prediction on the full distribution P, a.k.a the 'entropy of natural text'." Fitted E ≈ 1.69 nats.
Why it must be positive: Jeon & Van Roy 2024 — even an omniscient predictor knowing the true
process F incurs conditional entropy H(Y_{t+1}|F,Hₜ) ≥ 0; everything above is reducible.

| Literature | E-analog | Symbol / form |
|---|---|---|
| Chinchilla parametric (Hoffmann 2022) | Irreducible loss = Bayes risk = entropy of natural text | E in L=E+A/Nᵅ+B/Dᵝ; ≈1.69 nats |
| Asymptote extrapolation (Henighan 2020) | Entropy of true data distribution | L∞ ≈ S(True); reducible ≈ D_KL(True‖Model) |
| Info-theoretic floor (Jeon & Van Roy 2024) | Omniscient-predictor conditional entropy | (1/T)Σ H(Y_{t+1}|F,Hₜ) |
| Observational scaling (Ruan 2024) | Sigmoidal floor/ceiling scale | h in Eₘ≈h·σ(βᵀSₘ+α), h∈[0.8,1.0] |
| Bayesian-ICL (Arora 2024) | Per-example expected likelihood of dominant task | P_{λ,m} (diagonal of task likelihood matrix), <1 |
| Bayes error (Ishida 2023; Cover-Hart 1967) | Best achievable classification error | β = E_x[min{p(+1|x),p(−1|x)}]; R* |
| Rate-distortion (Shannon 1959) | Min achievable distortion at a rate | R(D)=min I(X;X̂) s.t. E[d]≤D |
| Aleatoric uncertainty (Hüllermeier 2021) | Irreducible predictive uncertainty | H[p(y|x)] |
| Latent-trait ceiling (4PL IRT; Barton & Lord 1981; PSN-IRT 2026) | Upper asymptote of item curve | d<1 = max success prob of a maximally-able model |
| Human/annotator ceiling (Nie 2020) | Inter-annotator disagreement entropy | entropy of human label distribution |

Important contrast: the data-model *theory* papers (Hutter 2021, Bahri 2021, Sharma & Kaplan 2020,
Michaud 2023) mostly **assume the noiseless floor is zero** and derive only the decay exponent
(α∼1/d, 4/d, or α/(1+α) from Zipf). E appears only when label noise / intrinsic entropy is added by
hand. So "E exists" is **not universal** — it's a modeling choice about whether aleatoric noise is
part of the data process.

## 2. The estimation taxonomy (the heart of it)

Pivotal distinction: **identified** (data pin the value) vs **extrapolated** (intercept of a fit
dominated by points far from the limit) vs **assumed/fixed** (property of the model, not learned).

| Method | What it estimates | Point or bound | Identified vs extrapolated |
|---|---|---|---|
| (i) Joint parametric fit (Hoffmann 2022; Muennighoff 2023; Gadre 2024) | E as a free additive constant | Point | **Extrapolated** (intercept of a finite-N,D fit) |
| (ii) Asymptote extrapolation (Henighan 2020; Kaplan 2020) | L∞ via power-law-plus-constant | Point (read as entropy) | **Extrapolated** |
| (iii) Nonparametric Bayes-error (Cover-Hart 1967; Berisha 2016; Sekeh 2020; Noshad 2019; Theisen 2021; Ishida 2023) | R*/β | Bound (kNN/divergence) OR point | **Identified-from-data** (under assumptions) |
| (iv) Info-theoretic lower bounds (Fano 1961; Shannon 1959; Jeon & Van Roy 2024) | Achievable floor | **Bound only** | Derived, not fit |
| (v) Latent-trait ceiling (4PL IRT; Barton & Lord 1981; PSN-IRT 2026; metabench 2025) | Upper asymptote d<1 | Point per item | Identified (4PL) or **assumed** (1PL/2PL/GRM fix=1) |
| (vi) Human/inter-annotator ceiling (Nie 2020) | Aleatoric floor proxy | Point proxy | Identified-from-data (proxy only) |

**(i) Joint fit.** E is one of five free params (A,B,E,α,β), Huber loss + L-BFGS. Not measured — a
constant confounded with the decay terms and the optimizer settings. Muennighoff 2023 doesn't even
re-fit it (fixes E≈1.87). Fails if too few near-converged large runs.

**(ii) Asymptote extrapolation (Henighan 2020 — the central paper).** Fits `L(x)=L∞+(x₀/x)ᵅ`, reads
L∞ ≈ S(True), extrapolates the reducible part down to a few nats. Requires that an infinite
transformer could model the distribution exactly — and the authors **concede** they "cannot yet
obtain a meaningful estimate for the entropy of natural language." Kaplan 2020 omits E entirely;
its crossover L*≈1.7 is "uncertain by an order of magnitude."

**(iii) Nonparametric Bayes-error.** *Bounds:* Cover-Hart 1-NN bracket (upper bound up to 2×R*);
Berisha/Sekeh divergence brackets via MST edge counts. *Point estimates (under assumptions):*
Noshad 2019 (density smoothness), Theisen 2021 (exact for a flow *surrogate*), Ishida 2023/Ushio
2026 (`β̂=(1/n)Σ min{cᵢ,1−cᵢ}` from soft labels, unbiased). Verdict paper **FeeBee (Renggli 2021):
on real data no estimator is reliably accurate; all acutely hyperparameter-sensitive.**

**(iv) Info-theoretic bounds.** Fano, rate-distortion R(D), Jeon & Van Roy conditional-entropy
floor — genuinely derived, hence the only truly identified version, but they **bound, not pin**, and
explain *why* E>0 without giving its number. Rate-distortion is metric-dependent.

**(v) IRT latent-trait ceiling.** In 1PL/2PL/GRM/**Beta-IRT (IRSL 2026, Choi 2026)** the upper
asymptote is **structurally fixed at 1.0** — only difficulty is identified. A data-identified sub-1
ceiling exists **only in 4PL** (Barton & Lord 1981 upper asymptote; PSN-IRT 2026 "feasibility"
d = max prob even highly-proficient models answer correctly; metabench 2025). **Weakly identified
without enough high-ability respondents** (Barton & Lord: helped fit in only 2/4 datasets).

**(vi) Human ceiling.** Nie 2020 (ChaosNLI): entropy of ~100 labels/example as the aleatoric floor;
shows the single-label "human ceiling" *overstates* achievable accuracy. Mixes aleatoric ambiguity
with annotator noise; protocol-specific; a proxy, not a measurement.

## 3. Reliability verdict

An extrapolated E is an artifact unless specifically defended. Documented failure modes:
- **Functional-form indistinguishability.** Clauset 2009: over a finite range, power law is
  "empirically indistinguishable from log-normal and stretched exponential." Alabdulmohsin 2022:
  exponents from −0.24 to −0.40 fit the same curve, imply different asymptotes; small interpolation
  error ≠ small extrapolation error.
- **Refit variance.** Besiroglu 2024: Chinchilla E shifts 1.69 → ~1.82 (SE 0.03), original CIs
  "implausibly narrow … would require over 600,000 experiments" vs ~400 run; traced to L-BFGS
  stopping early.
- **Regime breaks.** Caballero 2022 (BNSL): sharp breaks beyond the fitted range are
  provably un-extrapolable.
- **Metric dependence.** Schaeffer 2023: a sharp ceiling can be manufactured/erased purely by
  metric choice; Schaeffer 2024 (2406.04391): downstream metric transforms degrade the scale
  relationship.
- **Most downstream tasks don't scale smoothly.** Lourie 2025: only 39% of 46 task-setups show
  smooth predictable scaling; trends can flip with the validation corpus. Gadre 2024: dropping one
  1.4B model blew relative error 0.05% → 10.64%.

**Required to defend an asymptote as a bound** (convergent recommendation): (1) held-out
backtesting; (2) multiple functional forms compared; (3) honest bootstrap CIs; (4)
known-ground-truth calibration; (5) a stated, ideally continuous, metric.

## 4. The crisp answer

*Knowable:* the ceiling is real and positive in any task with aleatoric noise — an omniscient
predictor still incurs the conditional entropy (Jeon & Van Roy 2024); Fano / rate-distortion give
achievable floors. These are **derived bounds**, not the value for a specific real distribution.

*Not knowable:* the **numerical value** of E for natural language is not identified by any trained
-scaling method. Henighan (who pioneered the entropy-via-asymptote reading) says so explicitly.
Chinchilla's E≈1.69 and the refit ~1.82–1.89 are jointly-fit, decay-confounded, dataset-specific,
replication-fragile constants.

*Which methods give a defensible number vs only a bound:*
- **Defensible point number — but only for a measurable proxy task:** nonparametric Bayes-error
  point estimators (Ishida/Ushio from soft labels; Noshad; Theisen for a surrogate) and the 4PL IRT
  upper asymptote (PSN-IRT; metabench). Caveats: need clean soft labels / smoothness / surrogate
  fidelity (FeeBee: fail on real high-dim data), or are per-item, benchmark-specific, weakly
  identified without high-ability models.
- **Only a bound:** Fano, rate-distortion, Jeon & Van Roy; divergence/kNN brackets.
- **Point estimates that should be reported as bounds / model-dependent facts:** Chinchilla joint-fit
  E and Henighan/Kaplan L∞ — the most-cited numbers, the least defensible as facts.

**Honest summary:** a perfect model converges to the data entropy / Bayes risk, which is provably
positive — but for natural language that number is *extrapolated, not measured*, reliable only when
defended by held-out backtesting + multiple forms + honest CIs + known-ground-truth calibration. The
only methods that *identify* a number do so for a constrained proxy (binary Bayes error from soft
labels, or a per-item 4PL ceiling), not for the open-ended next-token-entropy floor.

*Threads disagree on:* (a) whether E exists at all (theory papers set it to zero, inject noise as an
add-on); (b) no paper estimates a nonparametric Bayes error of a real natural-language task — the
bridge to a measured language-entropy floor is the open gap; (c) Ruan's floor h is nominally fit but
pinned to ~1 in practice, carrying little ceiling information.

---

## 5. What this means for our framework (`τ(E)` / the dense ceiling)

Mostly **vindication** of stances already in §6, plus one concrete new constraint on the IRT plan.

1. **"No executor-free E → a function, not a number" is the same epistemic situation one level up,
   not a dodge.** The field's most famous irreducible term — the *model-free* next-token entropy —
   is itself *not numerically identified* (Henighan concedes it; Chinchilla's E is a fragile
   joint-fit constant). So our refusal to claim a single articulability number, and our `τ(E)`-as-a-
   function-of-executor framing, sits in exactly the same boat as the entropy-of-language question.
   We're not being evasive; we're being as honest as the scaling literature is forced to be.

2. **The dense-model ceiling = method (vi), the Nie-2020 human/annotator ceiling — an empirical,
   label-free, protocol-specific UPPER bound that overstates achievable performance** (mixes
   aleatoric ambiguity with model noise). This is exactly our `R_dense ≥ R*` "doubly conservative"
   stance (§1, T5). Action: cite the dense ceiling as a Nie-style empirical proxy, never as a
   measured Bayes risk.

3. **Our sandwich-not-asymptote discipline is precisely what the literature says is required.** The
   "defend a bound" checklist (held-out backtest, multiple forms, honest bootstrap CIs, known-
   ground-truth calibration, continuous metric) maps 1:1 onto E7 (fresh-holdout F_lo, multi-form
   fits, paired-bootstrap CIs, E0 planted-thickness, continuous fidelity per U3). The 06-11
   amendment ("τ̂ descriptive-with-CI, never a defended bound") is independently corroborated by
   Besiroglu + Clauset + Caballero.

4. **NEW constraint on U4 (the κ_r / IRT estimator): an articulability *ceiling* is NOT free from
   the Beta-IRT/2PL that IRSL and Choi use.** Those models **structurally pin the upper asymptote at
   1.0** — so "max fidelity as judge capability θ→∞" cannot be read off them. To get a sub-1
   articulability ceiling (which *is* the `τ(E)` object) we'd need a **4PL upper-asymptote
   parameter** — and 4PL ceilings are *weakly identified without high-ability respondents* (Barton &
   Lord 1981). The payoff: **the dense model (our strongest judge) is exactly the high-ability anchor
   that would identify that 4PL ceiling** — so the dense anchor isn't just the Bayes-floor proxy
   (point 2), it's also what makes the IRT ceiling estimable. If we only ever use 2PL/Beta-IRT, the
   ceiling is assumed, not measured.

5. **T2 (power-law frontier) is Henighan's "extrapolate the reducible part" move**, and the Clauset
   form-indistinguishability + Besiroglu refit-variance warnings are why T2's exponent must be
   defended by the *shared-Zipf-exponent* prediction (an independent anchor), not by curve-fit alone
   — already the §4/T2 plan, now externally justified.


## References (auto-verified BibTeX, 2026-06-15)

> Citations below were extracted from this document and web-verified by an automated fact-check pass (search → fetch → retrieve resolvable id), with the attributed claim checked against the located paper. 36 entries; 4 also passed an independent second-pass audit (the rest were verified once — the audit pass was cut off by a quota limit, not by a failure). Entries are real located works; do not treat as hand-checked. See "needs manual review" below for 2 citations whose attributed claim the source paper appears to **contradict** and 0 unlocatable shorthands.

```bibtex
@inproceedings{alabdulmohsin2022revisiting,
  title={Revisiting Neural Scaling Laws in Language and Vision},
  author={Alabdulmohsin, Ibrahim and Neyshabur, Behnam and Zhai, Xiaohua},
  booktitle={Advances in Neural Information Processing Systems 35 (NeurIPS 2022)},
  year={2022},
  eprint={2209.06640},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@misc{arora2024bayesian,
  title        = {Bayesian scaling laws for in-context learning},
  author       = {Arora, Aryaman and Jurafsky, Dan and Potts, Christopher and Goodman, Noah D.},
  year         = {2024},
  eprint       = {2410.16531},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  note         = {Later published at COLM 2025},
  url          = {https://arxiv.org/abs/2410.16531}
}

@misc{bahri2021explaining,
  title        = {Explaining Neural Scaling Laws},
  author       = {Bahri, Yasaman and Dyer, Ethan and Kaplan, Jared and Lee, Jaehoon and Sharma, Utkarsh},
  year         = {2021},
  eprint       = {2102.06701},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2102.06701},
  note         = {Later published in Proceedings of the National Academy of Sciences 121(27):e2311878121 (2024), doi:10.1073/pnas.2311878121}
}

@techreport{barton1981upper,
  author      = {Barton, Mark A. and Lord, Frederic M.},
  title       = {An Upper Asymptote for the Three-Parameter Logistic Item-Response Model},
  institution = {Educational Testing Service},
  year        = {1981},
  month       = {July},
  number      = {RR-81-20},
  type        = {ETS Research Report},
  series      = {ETS Research Report Series},
  doi         = {10.1002/j.2333-8504.1981.tb01255.x}
}

@article{berisha2016empirically,
  title={Empirically Estimable Classification Bounds Based on a Nonparametric Divergence Measure},
  author={Berisha, Visar and Wisler, Alan and Hero, Alfred O. and Spanias, Andreas},
  journal={IEEE Transactions on Signal Processing},
  volume={64},
  number={3},
  pages={580--591},
  year={2016},
  publisher={IEEE},
  doi={10.1109/TSP.2015.2477805}
}

@misc{besiroglu2024chinchilla,
  title={Chinchilla Scaling: A replication attempt},
  author={Besiroglu, Tamay and Erdil, Ege and Barnett, Matthew and You, Josh},
  year={2024},
  eprint={2404.10102},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  doi={10.48550/arXiv.2404.10102},
  url={https://arxiv.org/abs/2404.10102}
}

@misc{caballero2022broken,
  title={Broken Neural Scaling Laws},
  author={Caballero, Ethan and Gupta, Kshitij and Rish, Irina and Krueger, David},
  year={2022},
  eprint={2210.14891},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  note={Published at ICLR 2023},
  url={https://arxiv.org/abs/2210.14891}
}

@misc{choi2026diagnosing,
  title        = {Diagnosing the Reliability of LLM-as-a-Judge via Item Response Theory},
  author       = {Choi, Junhyuk and Park, Sohhyung and Cho, Chanhee and Park, Hyeonchu and Kim, Bugeun},
  year         = {2026},
  eprint       = {2602.00521},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL}
}

@article{clauset2009power,
  title={Power-law distributions in empirical data},
  author={Clauset, Aaron and Shalizi, Cosma Rohilla and Newman, M. E. J.},
  journal={SIAM Review},
  volume={51},
  number={4},
  pages={661--703},
  year={2009},
  publisher={Society for Industrial and Applied Mathematics},
  doi={10.1137/070710111}
}

@article{cover1967nearest,
  author  = {Cover, Thomas M. and Hart, Peter E.},
  title   = {Nearest neighbor pattern classification},
  journal = {IEEE Transactions on Information Theory},
  volume  = {13},
  number  = {1},
  pages   = {21--27},
  year    = {1967},
  doi     = {10.1109/TIT.1967.1053964}
}

@book{fano1961transmission,
  title     = {Transmission of Information: A Statistical Theory of Communications},
  author    = {Fano, Robert M.},
  year      = {1961},
  publisher = {The MIT Press},
  address   = {Cambridge, MA},
  isbn      = {9780262060011}
}

@article{gadre2024language,
  title={Language models scale reliably with over-training and on downstream tasks},
  author={Gadre, Samir Yitzhak and Smyrnis, Georgios and Shankar, Vaishaal and Gururangan, Suchin and Wortsman, Mitchell and Shao, Rulin and Mercat, Jean and Fang, Alex and Li, Jeffrey and Keh, Sedrick and Xin, Rui and Nezhurina, Marianna and Vasiljevic, Igor and Jitsev, Jenia and Soldaini, Luca and Dimakis, Alexandros G. and Ilharco, Gabriel and Koh, Pang Wei and Song, Shuran and Kollar, Thomas and Carmon, Yair and Dave, Achal and Heckel, Reinhard and Muennighoff, Niklas and Schmidt, Ludwig},
  journal={arXiv preprint arXiv:2403.08540},
  year={2024}
}

@misc{henighan2020scaling,
  title={Scaling Laws for Autoregressive Generative Modeling},
  author={Henighan, Tom and Kaplan, Jared and Katz, Mor and Chen, Mark and Hesse, Christopher and Jackson, Jacob and Jun, Heewoo and Brown, Tom B. and Dhariwal, Prafulla and Gray, Scott and Hallacy, Chris and Mann, Benjamin and Radford, Alec and Ramesh, Aditya and Ryder, Nick and Ziegler, Daniel M. and Schulman, John and Amodei, Dario and McCandlish, Sam},
  year={2020},
  eprint={2010.14701},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@article{hoffmann2022training,
  title={Training Compute-Optimal Large Language Models},
  author={Hoffmann, Jordan and Borgeaud, Sebastian and Mensch, Arthur and Buchatskaya, Elena and Cai, Trevor and Rutherford, Eliza and de Las Casas, Diego and Hendricks, Lisa Anne and Welbl, Johannes and Clark, Aidan and Hennigan, Tom and Noland, Eric and Millican, Katie and van den Driessche, George and Damoc, Bogdan and Guy, Aurelia and Osindero, Simon and Simonyan, Karen and Elsen, Erich and Rae, Jack W. and Vinyals, Oriol and Sifre, Laurent},
  journal={arXiv preprint arXiv:2203.15556},
  year={2022},
  eprint={2203.15556},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@article{hullermeier2021aleatoric,
  title   = {Aleatoric and Epistemic Uncertainty in Machine Learning: An Introduction to Concepts and Methods},
  author  = {H{\"u}llermeier, Eyke and Waegeman, Willem},
  journal = {Machine Learning},
  volume  = {110},
  number  = {3},
  pages   = {457--506},
  year    = {2021},
  publisher = {Springer},
  doi     = {10.1007/s10994-021-05946-3}
}

@misc{hutter2021learning,
  title         = {Learning Curve Theory},
  author        = {Hutter, Marcus},
  year          = {2021},
  eprint        = {2102.04074},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  howpublished  = {arXiv preprint arXiv:2102.04074}
}

@inproceedings{ishida2023,
  title={Is the Performance of My Deep Network Too Good to Be True? A Direct Approach to Estimating the Bayes Error in Binary Classification},
  author={Ishida, Takashi and Yamane, Ikko and Charoenphakdee, Nontawat and Niu, Gang and Sugiyama, Masashi},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023},
  note={Notable-top-5\%},
  eprint={2202.00395},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2202.00395}
}

@misc{jeon2024information,
  title        = {Information-Theoretic Foundations for Machine Learning},
  author       = {Jeon, Hong Jun and Van Roy, Benjamin},
  year         = {2024},
  eprint       = {2407.12288},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  note         = {arXiv:2407.12288}
}

@article{kaplan2020scaling,
  title={Scaling Laws for Neural Language Models},
  author={Kaplan, Jared and McCandlish, Sam and Henighan, Tom and Brown, Tom B. and Chess, Benjamin and Child, Rewon and Gray, Scott and Radford, Alec and Wu, Jeffrey and Amodei, Dario},
  journal={arXiv preprint arXiv:2001.08361},
  year={2020}
}

@inproceedings{kipnis2025metabench,
  title={metabench -- A Sparse Benchmark of Reasoning and Knowledge in Large Language Models},
  author={Kipnis, Alex and Voudouris, Konstantinos and Schulze Buschoff, Luca M. and Schulz, Eric},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025},
  eprint={2407.12844},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2407.12844}
}

@misc{lourie2025scaling,
  title={Scaling Laws Are Unreliable for Downstream Tasks: A Reality Check},
  author={Lourie, Nicholas and Hu, Michael Y. and Cho, Kyunghyun},
  year={2025},
  eprint={2507.00885},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  note={Findings of the Association for Computational Linguistics: EMNLP 2025},
  url={https://arxiv.org/abs/2507.00885}
}

@inproceedings{michaud2023quantization,
  title={The Quantization Model of Neural Scaling},
  author={Michaud, Eric J. and Liu, Ziming and Girit, Uzay and Tegmark, Max},
  booktitle={Advances in Neural Information Processing Systems 36 (NeurIPS 2023)},
  year={2023},
  eprint={2303.13506},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  doi={10.48550/arXiv.2303.13506}
}

@inproceedings{muennighoff2023scaling,
  title={Scaling Data-Constrained Language Models},
  author={Muennighoff, Niklas and Rush, Alexander M. and Barak, Boaz and Le Scao, Teven and Piktus, Aleksandra and Tazi, Nouamane and Pyysalo, Sampo and Wolf, Thomas and Raffel, Colin},
  booktitle={Advances in Neural Information Processing Systems 36 (NeurIPS 2023)},
  year={2023},
  eprint={2305.16264},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2305.16264}
}

@inproceedings{nie2020what,
    title = "What Can We Learn from Collective Human Opinions on Natural Language Inference Data?",
    author = "Nie, Yixin and Zhou, Xiang and Bansal, Mohit",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    pages = "9131--9143",
    publisher = "Association for Computational Linguistics",
    doi = "10.18653/v1/2020.emnlp-main.734",
    url = "https://aclanthology.org/2020.emnlp-main.734/"
}

@misc{noshad2019learning,
  title        = {Learning to Benchmark: Determining Best Achievable Misclassification Error from Training Data},
  author       = {Noshad, Morteza and Xu, Li and Hero, Alfred},
  year         = {2019},
  eprint       = {1909.07192},
  archivePrefix = {arXiv},
  primaryClass = {stat.ML}
}

@inproceedings{renggli2021evaluating,
  title     = {Evaluating Bayes Error Estimators on Real-World Datasets with FeeBee},
  author    = {Renggli, C{\'e}dric and Rimanic, Luka and Hollenstein, Nora and Zhang, Ce},
  booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS Datasets and Benchmarks)},
  year      = {2021},
  eprint    = {2108.13034},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2108.13034}
}

@inproceedings{ruan2024observational,
  title={Observational Scaling Laws and the Predictability of Language Model Performance},
  author={Ruan, Yangjun and Maddison, Chris J. and Hashimoto, Tatsunori},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024},
  eprint={2405.10938},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2405.10938}
}

@inproceedings{schaeffer2023emergent,
  title     = {Are Emergent Abilities of Large Language Models a Mirage?},
  author    = {Schaeffer, Rylan and Miranda, Brando and Koyejo, Sanmi},
  booktitle = {Advances in Neural Information Processing Systems 36 (NeurIPS 2023)},
  year      = {2023},
  eprint    = {2304.15004},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI}
}

@misc{schaeffer2024why,
  title={Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive?},
  author={Schaeffer, Rylan and Schoelkopf, Hailey and Miranda, Brando and Mukobi, Gabriel and Madan, Varun and Ibrahim, Adam and Bradley, Herbie and Biderman, Stella and Koyejo, Sanmi},
  year={2024},
  eprint={2406.04391},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  howpublished={arXiv preprint arXiv:2406.04391}
}

@article{sekeh2020learning,
  title={Learning to Bound the Multi-Class Bayes Error},
  author={Sekeh, Salimeh Yasaei and Oselio, Brandon and Hero, Alfred O.},
  journal={IEEE Transactions on Signal Processing},
  volume={68},
  pages={3793--3807},
  year={2020},
  publisher={IEEE},
  doi={10.1109/TSP.2020.2994807}
}

@inproceedings{shannon1959coding,
  author    = {Claude E. Shannon},
  title     = {Coding Theorems for a Discrete Source With a Fidelity Criterion},
  booktitle = {Institute of Radio Engineers (IRE) National Convention Record},
  volume    = {7},
  number    = {4},
  pages     = {142--163},
  year      = {1959}
}

@misc{sharma2020neural,
  title        = {A Neural Scaling Law from the Dimension of the Data Manifold},
  author       = {Sharma, Utkarsh and Kaplan, Jared},
  year         = {2020},
  eprint       = {2004.10802},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  note         = {Published as ``Scaling Laws from the Data Manifold Dimension'', Journal of Machine Learning Research 23(9):1--34, 2022},
  url          = {https://arxiv.org/abs/2004.10802}
}

@inproceedings{theisen2021evaluating,
  title     = {Evaluating State-of-the-Art Classification Models Against Bayes Optimality},
  author    = {Theisen, Ryan and Wang, Huan and Varshney, Lav R. and Xiong, Caiming and Socher, Richard},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {34},
  year      = {2021},
  eprint    = {2106.03357},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML}
}

@misc{truong2026item,
  title={Item Response Scaling Laws: A Measurement Theory Approach for Efficient and Generalizable Neural Scaling Estimation},
  author={Truong, Sang T. and Tu, Yuheng and Schaeffer, Rylan and Koyejo, Sanmi},
  year={2026},
  eprint={2606.07616},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@inproceedings{ushio2026practical,
  title={Practical estimation of the optimal classification error with soft labels and calibration},
  author={Ushio, Ryota and Ishida, Takashi and Sugiyama, Masashi},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026},
  eprint={2505.20761},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2505.20761}
}

@inproceedings{zhou2026lost,
  title={Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory},
  author={Zhou, Hongli and Huang, Hui and Zhao, Ziqing and Han, Lvyuan and Wang, Huicheng and Chen, Kehai and Yang, Muyun and Bao, Wei and Dong, Jian and Xu, Bing and Zhu, Conghui and Cao, Hailong and Zhao, Tiejun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026},
  note={Oral. arXiv preprint arXiv:2505.15055}
}

```

### Citations needing manual review

**Claim possibly contradicted by the source (2)** — paper is real, but our text's attribution may be wrong:

- `caballero2022broken` — Caballero 2022 (BNSL)
- `muennighoff2023scaling` — Muennighoff 2023

**Partial claim-match (6)** — paper located, attributed claim only partly supported; spot-check before relying on the exact number/wording:

- `barton1981upper`; `choi2026diagnosing`; `hullermeier2021aleatoric`; `kaplan2020scaling`; `renggli2021evaluating`; `sharma2020neural`
