# Literature review: debiasing models against spurious features / shortcut learning

Date: 2026-08-10. Agent: claude-litreview-spurious. Status: DRAFT (assembling).

Purpose: position this program's instrument choices comparatively, and find
anything better we have missed. Written against our concrete situation, not as a
generic survey. Our four instrument decisions, for reference:

- **(a) ADOPTED — readout-time spurious-influence measurement.** Stacked
  increment of the model of interest over a *nuisance model* (a model built only
  from the named nuisance channels), plus matched sampling on the nuisance score.
  Frozen and running. Spec: `notes/2026-08-05__taste-decomposition-design.md` §9.
- **(b) RETIRED — adversarial representation debiasing (gradient-reversal, GRL).**
  Failed a planted-token battery under two architectures. Audit:
  `notes/2026-08-07__debias_audit_fable.md`.
- **(c) PILOTING — LEACE-style closed-form linear concept erasure**, same planted
  battery. Code: `methods/taste_decomposition/debias/leace/`.
- **(d) PILOTING — decorrelated training.** Importance-reweight the dense training
  distribution so y ⊥ joint-nuisance-score (weights only; no text edits, no row
  deletion), retrain → T_decor. Removes the INCENTIVE, not the decodability;
  gated on RELIANCE (ablation), not on probe accuracy. Spec: design note §12.
- **REJECTED (standing, user directive) — counterfactual text editing /
  paraphrase canonicalisation.** Rewrites cannot be trusted to perturb only the
  intended channel.

Terms unpacked on first use throughout. **GRL** = gradient-reversal layer.
**DRO** = distributionally robust optimisation. **INLP** = iterative nullspace
projection. **LEACE** = LEAst-squares Concept Erasure. **DFR** = deep feature
reweighting. **PoE** = product of experts. **CAD** = counterfactually augmented
data. **ESS** = effective sample size. **JTT** = Just Train Twice. **LfF** =
Learning from Failure.

---

## Class 1. Train-time distribution interventions (this is where decorrelated training lives)

### 1.1 The two papers that should govern the decorrelated-training design

**Byrd & Lipton, ICML 2019, "What is the Effect of Importance Weighting in Deep
Learning?"** (VERIFIED). The central negative: *"While importance weighting impacts
deep nets early in training, so long as the nets are able to separate the training
data, its effect diminishes over successive epochs."* Models trained with wildly
different weights converge to similar solutions. Reported mitigations: **early
stopping** (the effect "may only occur in conjunction with early stopping,
disappearing asymptotically"; in their runs it took >100 epochs for extreme-weight
models' classification ratios to stabilise), **L2** and **batchnorm** (which the
authors call "the wrong abstraction" — L2 works by preventing SGD from reaching
the large-norm max-margin solution, not by any principled link to the weights).
Their own practical recommendation is that **sub-sampling may be preferable to
importance weighting** for deep nets on large training sets.

**Sagawa, Raghunathan, Koh & Liang, ICML 2020, "An Investigation of Why
Overparameterization Exacerbates Spurious Correlations"** (VERIFIED). Same
mechanism, measured on spurious correlations specifically: **subsampling the
majority group works in the overparameterised regime; upweighting/reweighting
fails there.** CelebA reweighted: worst-group error **>60%** overparameterised vs
25.6% underparameterised — adding capacity makes reweighting *worse*. Waterbirds
reweighted: 42.4% vs 26.6%. Subsampling under the same capacity reaches 15.1%
worst-group error on CelebA. Theory: once the data are separable, unregularised
logistic regression converges to the max-margin classifier and the reweighted
solution **equals** the unweighted one (Rosset et al. 2004; Soudry et al. 2018
implicit bias).

**Consequence for us — this is the single most important finding in the review.**
Our planted `⟦QX7⟧` token is a *near-perfect* predictor, which is exactly the
condition that makes the training set separable. The literature predicts that
importance weights become **provably inert on the converged model** in that
regime. Our V2' gate could therefore fail for a reason that has nothing to do with
whether decorrelation is the right idea. Three design implications, all cheap:
1. **Do not evaluate only the converged checkpoint.** Score the ablation-reliance
   readout across checkpoints; the effect, if any, is early-training.
2. **Weight decay is a first-class knob**, not a default. Byrd & Lipton and
   Sagawa et al. both say the weighting effect only survives under explicit
   capacity control.
3. **Our "weights only, no row deletion" constraint is exactly the arm the
   literature says is weaker.** Idrissi et al. (below) and both papers above
   favour subsampling. We should either run a subsample arm as a comparator or
   state in the paper why row deletion is disallowed (it changes n per cell and
   breaks the paired docket-level bootstrap) and accept the caveat.

### 1.2 The rest of the class

| method | canonical cite | guarantees | needs | failure mode | fit |
|---|---|---|---|---|---|
| **Group DRO** | Sagawa et al., ICLR 2020 (1911.08731) | convergence to minimax solution **in the convex case only**, O(1/√T); no NN generalisation bound | group label g on **every** training row | degenerates on overparameterised nets unless strongly regularised (Waterbirds worst-group 60.0% → 86.0% with strong L2/early stop) | **doesn't fit**: needs discrete groups; our nuisances are continuous scores; and our LoRA-on-8B is squarely in the regime it warns about |
| **JTT** | Liu et al., ICML 2021 (2107.09044) | none formal | no train-time groups, **but a group-labelled validation set** to tune λ_up and T | closes only ~75% of the ERM→GroupDRO gap; authors flag they don't know why early-stopped ERM latches on | **partial**: identify-then-upweight is adaptable to a continuous score, but the group-labelled val set is unavailable to us. Substitute: partial correlation of held-out residuals with the nuisance score |
| **LfF** | Nam et al., NeurIPS 2020 (2007.02561) | none formal | **no group labels** — only class labels | assumes the spurious feature is learned *faster*; untested when the bias model saturates instantly (our plant would) | **closest method cousin**: per-example continuous weights W(x)=CE(f_B)/(CE(f_B)+CE(f_D)), no groups. At 99.5% bias-aligned, CMNIST 35.3%→63.4% — real but far from ceiling, the best available data point on strong shortcuts |
| **Spectral Decoupling** | Pezeshki et al., NeurIPS 2021 (2011.09468) | NTK-regime analysis of gradient starvation (Thm 2: dz₂*/d(s₁²)<0) | nothing beyond labels | guarantees are NTK/binary/two-feature | **different class** (output-logit L2: L = Σlog[1+exp(−Yŷ)] + (λ/2)‖ŷ‖²), but the **mechanism citation we want**: it explains *why* a near-perfect nuisance starves gradient to the real quality signal. Cheap complementary arm |
| **Simple balancing** | Idrissi et al., CLeaR 2022 (2110.14503) | none formal | discrete/discretisable groups | — | **counter-evidence to our design**: subsample/reweight balancing matches fancier methods; and "group information is most critical for **model selection**, not training" — worth internalising |
| **CNC** | Zhang et al., ICML 2022 (2203.01517) | none | no train-time attribute labels | contrastive, representation-level not distribution-level | doesn't fit (different intervention point) |
| **EIIL** | Creager et al., ICML 2021 (2010.07249) | none | no env labels (infers them) | inherits IRM's problems | **possible bridge**: infers pseudo-environments; how we'd go from continuous nuisance to groups if we ever needed to |
| **GEORGE** | Sohoni et al., NeurIPS 2020 (2011.12945) | none | no subclass labels (clusters features) | — | same role as EIIL (LEAD) |
| **IRM** | Arjovsky et al. 2019 (1907.02893) | invariance across environments | **multiple labelled environments** | Rosenfeld et al. ICLR 2021: non-linear IRM "can fail catastrophically unless the test data are sufficiently similar to the training distribution"; Kamath et al. AISTATS 2021: IRMv1 can generalise *worse* than ERM | **doesn't fit** (no environments) — but the IRM cautionary arc is the right rhetorical precedent for "theoretically appealing invariance objective, underperforms outside narrow settings" |
| **LISA** | Yao et al., ICML 2022 (2201.00299) | linear-setting worst-group bound | domain + class labels | needs groups | doesn't fit |

### 1.3 The variance/effective-sample-size cost — the part we have not budgeted for

- **Cortes, Mansour & Mohri, NeurIPS 2010, "Learning Bounds for Importance
  Weighting"** (VERIFIED): generalisation bounds under importance weighting depend
  on the **second moment of the weights** (not the max), tied to the Rényi
  divergence between the source and target distributions; they give guarantees for
  unbounded weights **only** under a bounded-second-moment assumption, and
  identify explicit cases where importance weighting fails.
- **Shimodaira 2000** — the origin of weighted log-likelihood under covariate
  shift (SNIPPET).
- **Fang, Lu, Niu & Sugiyama, NeurIPS 2020, "Rethinking Importance Weighting for
  Deep Learning under Distribution Shift"** (SNIPPET): a **circularity** specific
  to deep learning — you need a good representation to estimate the density ratio,
  but you get a good representation by training on the weighted objective. Their
  fix is to alternate the two steps online.

**Consequence for us.** Decorrelating y from a *near-perfect* nuisance predictor
requires near-degenerate weights: the rows where y and the nuisance agree get
crushed, the rare disagreeing rows get blown up. That is an ESS collapse
**independent of** the Byrd–Lipton "weights vanish under interpolation" pathway.
We should compute and report ESS = (Σw)²/Σw² before and after decorrelation on
every cell, and budget weight clipping/trimming as a declared, tunable safeguard.
Also check that our joint-nuisance-score density-ratio estimate isn't circularly
dependent on a representation trained on un-decorrelated data (Fang et al.).

### 1.4 LLM-era work in this class

- **Du et al., CACM 2024 / 2208.11857, "Shortcut Learning of LLMs in NLU"** —
  the survey to cite for the problem class in our modality (SNIPPET).
- **Shuieh et al., 2025 (2505.05704), "Assessing Robustness to Spurious
  Correlations in Post-Training Language Models"** (VERIFIED) — closest *genre*
  match: SFT/DPO/KTO compared under **controlled spuriousness strength 10-90%**.
  Findings: degradation is task-dependent, not universal; no method dominates.
  It stops at 90% and tests no mitigation — so the near-100% regime our plant
  occupies is untested there too.
- **Yuan et al., 2024 (2410.13343), "Do LLMs Overcome Shortcut Learning?"** —
  *larger* LLMs exploit shortcuts *more* under in-context prompting (SNIPPET).
  In-context, not training-time; adjacent context only.
- **Le et al., 2024 (2407.14974), "Out of Spuriousity"** — extract a robust
  subnetwork via supervised contrastive loss, no group annotations (VERIFIED).
- **Yang, Prenkaj & Kasneci, AAAI 2025 (2412.07675), "RAZOR"** — unsupervised
  **LLM text rewriting** to break token/positional shortcuts (SNIPPET). This is
  the field's other 2024-era answer to our exact problem, and it is *edit-based*.
  Useful as the named contrast to our standing rejection of text rewriting.
- **Dagaev et al., Pattern Recognition Letters 2022 (2102.06406), "A
  Too-Good-to-be-True Prior"** — low-capacity net identifies easy/shortcut
  examples; downweight them when training the high-capacity net (VERIFIED).
  LfF-family, no groups needed.

### 1.5 Does anyone stress-test reweighting under a *near-perfect* shortcut?

**No.** Nothing found sweeps a mitigation all the way to a deterministic spurious
predictor with rigorous before/after measurement. The closest four, in order:
Sagawa ICML 2020 (mechanism argument about the separable limit, no strength
sweep), Byrd & Lipton (separability argument, not spurious-specific), LfF (one
empirical point at 99.5% bias-aligned, where the method recovers only 35→63%),
and Shuieh et al. 2025 (10-90% sweep, LLM post-training, but diagnostic only).
**This is a genuine gap our battery would fill** — and it is also a warning that
our battery's near-perfect plant is the *hardest* possible case, so a V2' failure
would not by itself condemn decorrelated training on real nuisance channels.

**Closest-precedent ranking for decorrelated training:** (1) Byrd & Lipton 2019
— same intervention, and the pitfall we must design around; (2) Sagawa ICML 2020
— the theory that predicts our regime; (3) LfF — closest method cousin
(group-free, per-example continuous weights); (4) Idrissi et al. 2022 — standing
counter-evidence for the reweighting-vs-subsampling choice we foreclosed;
(5) Cortes et al. 2010 + Fang et al. 2020 — the variance/circularity toolkit.

---

## Class 2. Representation-level concept erasure (our LEACE pilot)

**Headline: the pilot is well-founded, but two published results narrow its
certificate more than our current code comments admit, and one of them is a live
correctness issue in `leace/leace.py`'s docstring.**

### 2.1 What LEACE actually guarantees, and the continuous-Z caveat

**Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff & Biderman, NeurIPS 2023,
"LEACE: Perfect Linear Concept Erasure in Closed Form"** (VERIFIED from full text).
Definition 2.3 / Theorem 3.1: for **categorical, one-hot Z**, the following are
equivalent — (i) X linearly guards Z; (ii) no linear classifier beats the constant
predictor **under any convex loss**; (iii) all class-conditional means equal the
global mean; (iv) **Cov(X,Z)=0**. LEACE sets Cov(r(X),Z)=0 by construction and is
the **unique** minimiser of E‖r(X)−X‖²_M over *every* inner-product-induced norm,
among all linear guarding functions — a global closed-form optimum (contrast INLP,
whose minimality is per-step and local).

> **LIVE ISSUE for our pilot.** §4.3 of LEACE ("Extension to Continuous Z") says
> the method applies for Z ∈ ℝ^k *"as long as we restrict ourselves to the
> ordinary least squares regression loss."* **The strong "no linear predictor under
> any convex loss" certificate does NOT provably extend to continuous Z.** Our
> `leace.py` docstring currently asserts the strong form ("no linear predictor
> trained with a convex loss beats the constant predictor"). That claim is correct
> for the **plant arm** (binary Z, lines 226/255 of `run_battery_leace.py`) and for
> the median-binarised readouts (`cl_bin`, `yr_bin`), but the real-nuisance arms
> fit the eraser on **standardized continuous** matrices (`Zs6`, `Zd8`, `Zl0`,
> `Zk` at lines 292/344/368/435) and then evaluate with a *binarised* probe — i.e.
> the guarantee we hold and the readout we report are not the same object.
> **Cheap fix: bin each continuous nuisance (terciles or median) into one-hot
> columns and erase that**, which recovers the full theorem and makes the
> certificate match the probe. Alternatively erase both and report the pair.

Explicit non-guarantee, verbatim: *"it is intractable to nondestructively edit X
to prevent a general nonlinear adversary from recovering Z."* They also flag that
kernel extensions do not transfer — *"erasure functions fit using one kernel do not
generalize to other kernels."*

Concept-scrubbing (LEACE applied at every layer in the forward pass, erasing POS),
bits per UTF-8 byte: Pythia-160M 0.90 → **2.79**; Pythia-12B 0.62 → **3.20**;
LLaMA-7B 0.69 → **1.73**. (SAL, the spectral alternative: 3.53 / 4.69 / 3.24 — so
LEACE does *less* collateral LM damage than SAL at matched task.)

### 2.2 The direct threat: linear guardedness does not survive a nonlinear reader

Three independent demonstrations, all from the erasure literature itself:

- **R-LACE** (Ravfogel, Twiton, Goldberg & Cotterell, ICML 2022, "Linear
  Adversarial Concept Erasure") — after their theoretically near-optimal **rank-1**
  projection drives a linear SVM to chance on gender, *"as expected given that we
  removed a linear subspace, non-linear classifiers are still able to recover
  gender: Both RBF-SVM and a ReLU MLP with 1 hidden layer of size 128 predict
  gender with above 90% accuracy."* This is the cleanest quotable precedent for
  what to expect from a nonlinear probe on a LEACE-erased h.
- **SAL** (Shao, Ziser & Cohen, EACL 2023) — same pattern with a different method:
  linear SVM at 50.2% (chance) by K=2 removed directions, while RBF/poly-2
  classifiers stay high through K=30.
- **Ravfogel, Goldberg & Cotterell, ACL 2023, "Log-linear Guardedness and its
  Implications"** — the formal version. **Theorem 3.2** (binary Z, binary Y,
  δ-discretised log-linear predictors): guardedness *does* survive composition, so
  the downstream prediction cannot leak Z. **Theorem 3.4** (multiclass): for
  K-Voronoi-structured data, **even under ε-guardedness against binary Z there
  exists a K-class softmax model recovering Z almost perfectly** — the softmax
  nonlinearity lets the model identify which Voronoi region a point is in, and
  region identity encodes Z though no single linear direction does.

**Consequence for our certificate language.** Our scalar head is linear in z in
the bottleneck arch and linear in h in the pooled arch — which is the *good* case,
and we should say so explicitly and keep it that way. But (i) any hidden
nonlinearity in the head, and (ii) any multi-threshold / multi-bin readout of a
continuous score, moves us into the Theorem-3.4 regime where the certificate does
not compose. **The honest claim after a LEACE pass is: "the score, being a linear
functional of the erased representation, cannot use the channel" — not "the
channel is gone."** That is already how the bottleneck note scoped its
certificate, and it should be carried over verbatim.

R-LACE also contains an **independent published replication of our GRL negative**:
their Table 2 shows adversarially-finetuned BERT (both linear and MLP adversary —
the gradient-reversal family) *"show somewhat decreased bias… but do not hinder
the ability to predict gender at all"* (gender accuracy **98-99.4%** post-training).

### 2.3 Amnesic probing — and the critique that also lands on our gate

**Elazar, Ravfogel, Jacovi & Goldberg, TACL 2021, "Amnesic Probing"** (VERIFIED):
*"a high probing accuracy does not necessarily imply that the model is using the
probed information to make predictions"*; instead *"we measure the reliance of a
model on a property by measuring the change in the model's predictions after
removing that property… we show that high probing accuracy does not imply high
reliance."* **This is the published precedent for our V2 gate's ablation-reliance
readout** — cite it exactly there.

But **Kumar, Tan & Sharma, NeurIPS 2022, "Probing Classifiers are Unreliable for
Concept Removal and Detection"** (VERIFIED) names amnesic probing as vulnerable:
Lemma 3.1 shows a probing classifier picks up **non-concept** (main-task) features
whenever concept and task labels are even weakly correlated; Theorem 3.2 shows this
"mixing" corrupts INLP's first projection direction and that the representation
norm strictly decreases each iteration, so *"after sufficiently many steps, INLP
can destroy all information in the representation."* Their explicit warning: a drop
in main-task accuracy after ablation can be an artifact of representation
corruption rather than evidence the model relied on the concept.

**Actionable for the LEACE battery:** LEACE's projection is *oblique*, not
orthogonal, so it too moves task-relevant directions when Z correlates with the
task signal. We should **report the correlation between the erased channel and the
real task signal in h**, and run a **specificity control** where LEACE is fit on a
channel the cell verifiably does not use — i.e. exactly the V3a/V3b gates that the
GRL chain never reached. Do not skip them this time; they are the check on this
confound.

### 2.4 INLP, and the collateral-damage comparison

**Ravfogel, Elazar, Gonen, Twiton & Goldberg, ACL 2020 (INLP)** — iterative
nullspace projection. No closed-form guarantee; linear guardedness reached
empirically. Their own numbers: after 35 iterations on GloVe gender, a **1-layer
ReLU MLP still recovers gender at 85.0%**. Downstream profession classification:
+1.9% (BOW), −5.1% (BWV), **−5.51% (BERT)**. Their §7: *"it is designed to be fed
to a linear decoder, and the attributes are not protected under non-linear
classifiers… there are no magic guarantees, and the burden of verification remains
on the user."*

Collateral damage, cross-paper and directionally consistent (no single controlled
head-to-head exists): on frozen-BERT profession classification, INLP(rank 50)
79.91% → **71.27%** (−8.6) vs R-LACE(rank 50) → **76.73%** (−3.2). The closed-form
/ globally-minimal methods (LEACE, R-LACE) consistently report **less** collateral
damage than the greedy iterative one. **This is a reason to prefer LEACE over INLP
in our pilot, and we already do.**

### 2.5 Erasure-evaluation critiques

- **Gonen & Goldberg, NAACL 2019, "Lipstick on a Pig"** — the founding "hides, not
  removes" result: *"current debiasing methods are actually hiding the bias rather
  than removing it from the embeddings, creating a false sense of fairness."*
  Gendered words still cluster; classifiers still recover gender; bias-by-neighbors
  persists (0.852 → 0.734 after INLP, i.e. most of it survives).
- **Goldfarb-Tarrant et al., ACL 2021, "Intrinsic Bias Metrics Do Not Correlate
  with Application Bias"** — representation-level probes do not predict downstream
  behaviour; measure the task. Supports gating on behaviour, as we do.
- **Orgad & Belinkov, GeBNLP 2022 "Choose Your Lenses"** and **ACL 2023 "BLIND"**
  (SNIPPET) — intrinsic/extrinsic metrics are dataset-coupled; passing a probe is
  insufficient evidence.

### 2.6 Nonlinear and continuous-concept erasure — the successors to watch

- **KRaM** (Basu Roy Chowdhury, Monath, Dubey, Ahmed & Chaturvedi, NeurIPS 2023,
  2312.00194), "Robust Concept Erasure via Kernelized Rate-Distortion
  Maximization" — the one method that explicitly claims to erase **categorical,
  continuous, and vector-valued** concepts. Rate-distortion objective (push apart
  representations with similar concept labels) rather than zeroing cross-covariance.
  No worst-case adversary-class guarantee comparable to LEACE Thm 3.1. **Directly
  on-point for our continuous nuisance scores; worth reading in full** (SNIPPET —
  guarantee statement and head-to-head numbers not verified).
- **Kernelized concept erasure** (Ravfogel, Vargas, Goldberg & Cotterell, EMNLP
  2022) — kernelised R-LACE; certificate holds **only for the kernel family used**.
- **"Fundamental Limits of Perfect Concept Erasure"** (Basu Roy Chowdhury et al.,
  AISTATS 2025, 2503.20098) — information-theoretic erasure/utility frontier;
  *"there seems to be an inherent tradeoff between erasure and retaining utility."*
- **Obliviator** (Akbari, Afshari & Boddeti, NeurIPS 2025) — HSIC penalties +
  constrained RKHS optimisation, applied **iteratively**, explicitly to price
  **nonlinear** guardedness. Closest published successor to the exact question our
  pilot faces. **LEAD — arXiv id/date inconsistent in the fetch; verify before
  citing.**
- **JSE** (Holstege, Wouters, van Giersbergen & Diks, ICML 2024, 2310.11991),
  "Removing Spurious Concepts… via Joint Subspace Estimation" — jointly estimates
  two orthogonal low-dimensional subspaces (spurious vs main-task), motivated by
  the claim that existing concept-removal methods are *"overzealous by
  inadvertently eliminating features associated with the main task."* Evaluated on
  Waterbirds, CelebA and **MultiNLI**. This is the closest published method to what
  our V3 specificity gate is trying to certify.
- **Free-Form LEACE** (EleutherAI blog, Belrose team) — removes the need for Z
  labels at inference by learning r(x) directly; relevant only if we ever need to
  erase at deployment.

### 2.7 Does erasure survive further fine-tuning? (nobody has tested our exact case)

- **"Unlearning Isn't Deletion"** (Xu et al., 2505.16831, VERIFIED) — near-zero
  forget accuracy recovers to **80%+ after brief fine-tuning**, retain accuracy
  rebounds to ~65%, and *"relearning on the forget set can often yield higher
  accuracy than that of the original model."* Mechanism: unlearning updates
  concentrate in shallow/output-adjacent parameters *"leaving deeper
  representations intact."*
- **Pham et al., ICLR 2024, "Circumventing Concept Erasure Methods For
  Text-to-Image Generative Models"** — *"Post hoc concept erasure in generative
  models provides a false sense of security."* Five erasure methods, all defeated
  by Concept Inversion **with no weight changes**.
- Parameter-level erasers built for robustness (**ELM**, Gandikota et al., NeurIPS
  2025; **PISCES**, Gur-Arieh et al., EMNLP 2025) report partial robustness gains,
  not guarantees.

**Gap we can claim.** No published result applies a representation-erasure method
and then **further fine-tunes an adapter over the same frozen backbone** to see
whether the erased signal reappears in h. Our architecture makes this an obvious
next test, and the prior points one way: the backbone's *capacity to compute* the
channel is untouched by a post-hoc projection of h, so a subsequent adapter update
can plausibly re-expose it. If we claim a LEACE certificate for a *trained*
pipeline, we should run erase → retrain-head → re-probe.

**Also a gap:** no paper found erases a feature that a fresh probe reads at
**AUC .97-1.00** and reports post-erasure recovery. Our planted battery would be
the first data point at that difficulty.

---

## Class 3. Post-hoc classifier surgery and bias-expert ensembles

**This class contains the biggest thing we have missed.** Our architecture is a
*frozen backbone + a small head*, which is literally DFR's precondition — more
literally than most published DFR applications.

### 3.1 DFR and its group-label-free descendants — the high-fit branch

**Kirichenko, Izmailov & Wilson, ICLR 2023, "Last Layer Re-Training is Sufficient
for Robustness to Spurious Correlations" (DFR).** Core claim: ERM-trained networks
*do* learn adequate core features even while relying on spurious ones; the bias
lives in how the last linear layer weighs them. Retraining only the last layer on
a small **group-balanced** reweighting set matches or beats group DRO (Waterbirds
worst-group ≈ 92-93 vs group DRO 91.4; CelebA ≈ 86-88 vs 88.9 — SNIPPET, numbers
triangulated from AFR's and ExMap's comparison tables, not from the primary PDF).

Two critiques that matter to us:
- **Le, Schlötterer & Seifert, 2023 (2308.00473), "Is Last Layer Re-Training Truly
  Sufficient…?"** — on realistic medical imaging DFR "remains susceptible to
  spurious correlations": it improves worst-group accuracy but does not solve the
  problem outside curated benchmarks.
- **"On the Unreasonable Effectiveness of Last-layer Retraining" (2512.01766,
  2025)** — the neural-collapse explanation was **not** supported; the evidence
  says DFR's success is **almost entirely explained by better group balance in the
  held-out reweighting set**, not by the layer choice. *So the balanced set is the
  active ingredient.* Whatever construction we use, the thing to report is how
  balanced the reweighting set is across (y × nuisance) cells.

**Group-free descendants — the ones that fit us, because we have no group labels:**

| method | mechanism | needs | reported |
|---|---|---|---|
| **AFR** (Qiu, Potapczynski, Izmailov & Wilson, ICML 2023, 2306.11074) | last-layer retraining with **continuous** per-example weights from the first-stage model's own predicted probability of the correct class: μᵢ ∝ β_{yᵢ}·exp(−γ·p̂ᵢ) — downweight what the base model already gets right | **no group labels**; held-out split with class labels; small group-labelled slice ideally for tuning | Waterbirds WG 90.4±1.1 (DFR 92.9), CelebA 82.0 (DFR 88.3), **MultiNLI 73.4 (DFR 74.7)**, **CivilComments 68.7 (DFR 70.1, group DRO 69.9)** — on text tasks it is essentially tied with group-labelled DFR |
| **SELF** (2309.08534) | build the reweighting set from **misclassifications or model disagreement** instead of true groups; proves disagreement upsamples worst-group data | no group annotations except for model selection; <3% of DFR's class annotations | "nearly matches DFR" on four vision+language benchmarks |
| **LFR** (2312.04893) | select high-loss misclassified + low-loss correct examples in balanced numbers = pseudo-groups by loss percentile | none | claims to beat DFR-with-group-labels in high-spurious regimes (UNVERIFIED at table level) |

AFR's own §5.2 has a limitation we should carry: they **prove** that on CelebA no
function of only the first-stage model's predictions can produce fully
group-balanced weights — self-referential reweighting has a ceiling set by how
cleanly the base model's errors track the true groups.

**Can we run DFR? Yes, and cheaply.** Three concrete constructions, in order of
preference:
1. **AFR's continuous weighting, no binning.** Compute per-example weights from
   our existing head's confidence, or — better and closer to our program — from a
   **nuisance-only model's** prediction of y (which we already build for the
   stacked increment), upweighting rows where the nuisance-only model is confident
   and wrong. No discretisation of continuous nuisance scores required.
2. **Explicit pseudo-group DFR**: bin each nuisance score into terciles, cross
   with y, stratify-sample so every (y × bin) cell is equally represented, refit
   the head on that subset only.
3. **EIIL** (Creager et al., ICML 2021) if hand-binning interacts badly with our
   *correlated* channels (length, hedging, formatting are not independent).

The missing ingredient in all published versions is the group-labelled validation
slice used for hyperparameter selection. **Our planted battery is a substitute for
exactly that** — tune γ (AFR) or the bin scheme on the planted cell where ground
truth is known, then freeze and apply.

### 3.2 Logit correction / logit adjustment

- **Menon et al., ICLR 2021 (2007.07314), logit adjustment**: subtract the log
  class prior from the logits, post-hoc or in-loss; Fisher-consistent for balanced
  error under label-frequency shift. **Wrong problem for us** — it targets P(y)
  imbalance, not feature-conditional shortcut reliance. Borrow only the
  post-hoc-on-a-frozen-model pattern.
- **Liu et al., ICLR 2023 (2212.01433), "Avoiding spurious correlations via logit
  correction"**: adds a per-example correction ln P̂(y, a_x) estimated by a
  deliberately-biased GCE-trained model as a group proxy. Proposition 1: minimising
  the LC loss ≡ maximising group-balanced accuracy. Reported: Waterbirds 90.5,
  CelebA 88.1, CivilComments 70.3. **Medium fit**: the mechanism ports if we swap
  their learned bias proxy for our judge-scored nuisance models, but the guarantee
  assumes a **single one-hot** spurious attribute — our channels are continuous
  and co-occurring, outside the proven regime.

### 3.3 Product-of-experts / bias-expert ensembles (the NLI lineage)

Shared shape: train or specify a **bias-only model**, combine with the main model
(usually additively in log space), so the main model is discouraged from
re-deriving what the bias model already gets free.

| paper | bias model | known vs automatic | ID cost |
|---|---|---|---|
| He, Zha & Wang, DeepLo@EMNLP 2019 (DRiFt) | fixed hypothesis-only model; main model fits the **residual** | known | not extracted |
| **Clark, Yatskar & Zettlemoyer, EMNLP 2019** (PoE, Learned-Mixin+H) | hand-specified per task | known | **MNLI-m 78.73 → 74.50 (−4.2 pts)** while HANS 50.58 → 53.35; VQA-CP 39.18 → 52.05 |
| Karimi Mahabadi, Belinkov & Henderson, ACL 2020 | hand-specified bias-only model(s); PoE + debiased focal loss | known | not extracted |
| **Utama, Moosavi & Gurevych, EMNLP 2020**, "Towards Debiasing NLU Models from Unknown Biases" | **shallow copy of the same architecture trained on a tiny random subset** (2,000 rows / 3 epochs for MNLI) — a model that memorised surface patterns but not the task | **automatic** | MNLI ID 81.4-84.3 with HANS +7.1-8.2; QQP → PAWS +33.6-52.3; an annealing schedule α: 1→0.8 recovers most of the ID cost |
| **Utama, Moosavi & Gurevych, ACL 2020**, "Mind the Trade-off" | confidence regularisation instead of discarding biased examples | automatic | designed to preserve ID; +7 pts HANS with ID essentially retained (SNIPPET) |
| **Sanh, Wolf, Belinkov & Rush, ICLR 2021**, "Learning from Others' Mistakes" | **TinyBERT, 4M params (2 layers, hidden 128)** — capacity limitation alone is the inductive bias | **automatic**, zero domain knowledge | MNLI 84.52 → 83.32 (−1.2), HANS non-entailed 26.74 → 41.35 |
| Clark et al., Findings EMNLP 2020, "Mixed Capacity Ensembles" | low-capacity model in a joint ensemble | automatic | not extracted |

> **NOMENCLATURE CORRECTION for our bib**: both 2020 papers are **Utama, Moosavi,
> Gurevych** — *Nafise Sadat Moosavi*, not Kassner. Verified against ACL Anthology
> `.bib` records.

**The single most transferable warning in this class** — Sanh et al.'s capacity
sweep (4.4M → 41.4M params for the weak learner): *"weaker learners assure good
balance between OOD and ID… stronger learners encourage OOD generalization at the
expense of ID performance."* At their largest weak model, OOD peaked near 97%
while ID **collapsed to 28%** on MNLI. **Read against our GRL result:** if a
nuisance expert is *too informative* about y, using it to shape the main model
destroys the main model. That is the same shape as our λ=1.0 arm (eval AUC .8283 →
.6898) and it suggests the failure was not only "min-max game" but partly
"the nuisance channel we pushed against carried too much of y."

Our mined nuisance channels sit in the **known/hand-specified** column, not the
automatic one. That buys interpretability and control at the cost of only catching
nuisances we thought to name — which is exactly what §5 (Feng et al.) warns about.

### 3.4 The diagnostic ancestry of our stacked-increment readout

- **Gururangan et al., NAACL 2018** — a hypothesis-only classifier recovers the
  label on ~67% of SNLI purely from annotator artifacts.
- **Poliak et al., *SEM 2018** — hypothesis-only baselines beat majority class on
  **10** NLI datasets; partial-input baselines become a standard artifact audit.
- **McCoy, Pavlick & Linzen, ACL 2019 (HANS)** — MNLI-trained BERT scores near 0%
  on non-entailment cases that break its three syntactic heuristics.

**Our nuisance model is a partial-input baseline in exactly this sense** — the
nuisance channels standing in for "hypothesis text." We should say so; it gives
the readout a seven-year-old, well-understood pedigree.

- **Feng, Wallace & Boyd-Graber, ACL 2019, "Misleading Failures of Partial-input
  Baselines"** — the standing caveat. A *successful* partial-input baseline proves
  the data are cheatable; a *failing* one does **not** prove they are clean. They
  show that augmenting a hypothesis-only model with trivial patterns solves **15%**
  of SNLI examples previously classed as "hard."
  **Direct implication:** a clean stacked increment rules out only the nuisance
  channels we scored — not interactions among them, and not channels we never
  named. This belongs verbatim in our limitations.

### 3.5 2023-2026 reward-model applications of this class

- **CARD** (Ng, Blöbaum, Bhandari, Zhang, Kasiviswanathan, 2510.23751, 2025),
  "Debiasing Reward Models by Representation Learning with Guarantees" — VAE-style
  factorisation into causal vs spurious latents *before* the reward head, with
  identifiability theorems under two regimes: a **surrogate proxy for the spurious
  feature** (our exact setting) or **multiple raters with diverse preferences**.
  Experiments use discrete/binary surrogates; not demonstrated for continuous
  confounders (SNIPPET).
- **FiMi-RM** (2505.12843, 2025) — three stages: train RM, fit a lightweight
  **nonlinear** length↔reward relation, subtract it out. A near-exact analogue of
  what we would do for the length channel alone; does not generalise to several
  channels at once.
- **DynaCF** (2606.09043, 2026) — per-sample "shortcut sensitivity" from
  semantics-preserving counterfactual perturbations, downweighted online in the
  Bradley-Terry loss (LEAD on numbers).
- **PRISM** (2510.19050, 2025) — group-invariant kernel learning, closed-form
  objective (LEAD on numbers).
- **"Factored Causal Representation Learning for Robust Reward Modeling in RLHF"**
  (2601.21350, ICML 2026 poster) — decomposes the embedding into causal vs
  non-causal factors with **an adversarial head and gradient reversal** on the
  non-causal factors. Worth flagging as a base rate: a contemporaneous ICML 2026
  paper is still shipping the exact mechanism our battery just retired.

---

## Class 4. Adversarial removal — does our negative match consensus?

**Answer: it MATCHES the founding critique's mechanism, is EXPLAINED by a 2022
impossibility-flavoured theorem, EXCEEDS every prior empirical demonstration in
effect size and architectural coverage, and one component of it (reliance
*increasing* under pressure) has NO located precedent.** No paper found reports a
clean success case that would contradict us.

### 4.1 The founding critique — same mechanism, weaker evidence

**Elazar & Goldberg, EMNLP 2018 (D18-1002), "Adversarial Removal of Demographic
Attributes from Text Data"** (VERIFIED). Verbatim:

> "When attempting to remove such demographic information using adversarial
> training, we find that **while the adversarial component achieves chance-level
> development-set accuracy during training, a post-hoc classifier, trained on the
> encoded sentences from the first part, still manages to reach substantially
> higher classification accuracies on the same data.**"

and the closing advice, "do not rely on the adversarial training to achieve
invariant representation to sensitive features."

Two facts we should use:
- **Their gap is modest**: adversary 49.0% (chance) vs post-hoc attacker 56.0% on
  mention/age; ~50% vs 58.5% on mention/gender. Ours is **probe AUC .97-1.00
  against a chance floor of .50** — categorically stronger.
- **They already tried the fixes we might worry we missed**, in the 2018 paper:
  larger adversary capacity (500-8000 hidden units), λ sweeps, **adversarial
  ensembles (multiple discriminators)**, periodic reinitialisation, staged
  training, extra hidden layers, dropout. None closed the gap. So "more/harder
  adversaries" was tested and rejected seven years before our experiment.

**Barrett, Kementchedjhieva, Elazar, Elliott & Søgaard, EMNLP-IJCNLP 2019
(D19-1662), "Adversarial Removal of Demographic Attributes Revisited"** — this is
*not* what the brief assumed. They argue Elazar & Goldberg's **diagnostic
classifier generalises poorly** to new in-domain samples and new domains, i.e.
in-sample probe accuracy may reflect sample-specific correlations. Their own
conclusion, verbatim:

> "Our contribution is mainly methodological… **Our results are also orthogonal to
> the main contribution of Elazar and Goldberg (2018), which is to show that
> adversarial debiasing is not always able to remove bias.**"

**This is good news for us and worth an explicit sentence in the paper.** Barrett
et al. raise the bar: a probe result must be shown to generalise, not just fit
in-sample. Our design clears that bar **by construction** — the target is a
planted ground-truth channel, not a noisy demographic label, and the probe is
fitted on train rows, early-stopped on a train holdout, and scored on eval rows
only, across 3 seeds, with a linear-probe replication.

### 4.2 The theory that explains our result

**Kumar, Tan & Sharma, NeurIPS 2022, "Probing Classifiers are Unreliable for
Concept Removal and Detection"** (VERIFIED abstract):

> "…we show that **these methods can be counter-productive: they are unable to
> remove the concepts entirely, and in the worst case may end up destroying all
> task-relevant features. The reason is the methods' reliance on a probing
> classifier as a proxy for the concept. Even under the most favorable conditions
> for learning a probing classifier when a concept's relevant features in
> representation space alone can provide 100% accuracy, we prove that a probing
> classifier is likely to use non-concept features and thus post-hoc or
> adversarial methods will fail to remove the concept correctly.**"

This is our mechanism, proved: the adversary is a *proxy*, so defeating it does
not certify the concept direction is gone. Note the scope — **"post-hoc or
adversarial"**: this paper is a warning for the LEACE pilot too, wherever the
erasure target is defined by a *learned* probe rather than by a closed-form
covariance condition.

Supporting theory:
- **McAllester & Stratos, AISTATS 2020, "Formal Limitations on the Measurement of
  Mutual Information"** — no distribution-free high-confidence MI *lower* bound
  from N samples can exceed O(ln N). Formal reason why a healthy adversary loss is
  not evidence of removed information (SNIPPET).
- **Cheng et al., ICML 2020 (CLUB)** — built because MI *lower*-bound estimators
  (MINE, InfoNCE) are structurally inapplicable to *minimising* MI. Same shape as
  our finding (SNIPPET).
- **Zhao et al., ICML 2019, "On Learning Invariant Representations for Domain
  Adaptation"** — ε_S + ε_T ≥ ½(d_JS(label marginals) − d_JS(rep marginals))²:
  under label shift, driving representation invariance to zero forces joint error
  up. **Different mechanism** (an accuracy/invariance trade-off), but a useful
  companion for our "task AUC collapses under pressure" observation (SNIPPET).
- **Lechner et al. 2021 (2107.03483), "Impossibility results for fair
  representations"** — no representation guarantees fairness for arbitrary
  downstream tasks (SNIPPET).

### 4.3 A failure mode we can explicitly rule out

**Acuna, Zhang, Law & Fidler, ICLR 2022, "Domain Adversarial Training: A Game
Perspective"** (SNIPPET): the GRL "transforms gradient descent into a competitive
gradient-based algorithm which may converge to periodic orbits and other
non-trivial limiting behavior that arise… in chaotic systems"; they propose
Runge-Kutta solvers. **This is a different pathology from ours** — non-convergence
/ oscillation — and our audit shows our adversary's loss stayed in a normal range
(.43-1.2) throughout. Citing this *strengthens* our claim: the negative is not
explainable by "the optimiser never converged." (One exception to note honestly:
our bottleneck λ=1.0 arm *did* spike, adv-loss to 361 with a mid-training eval dip
to .345, so for that arm the Acuna pathology cannot be fully excluded — the λ=.1
and λ=.5 arms are the clean evidence.)

Note also the original **Ganin et al., JMLR 2016** admission: SGD on the GRL
objective is only *stated* to converge to a saddle point of the minimax problem,
with no proof; and DANN itself evaluates removal with a separate **post-hoc linear
SVM** (Proxy A-distance) rather than trusting the co-trained adversary. Even the
founding paper implicitly distrusts the co-trained adversary as arbiter — which is
the distrust our fresh-probe design formalises.

### 4.4 Claimed fixes since 2018, and whether any is a threat to our verdict

| fix | cite | status |
|---|---|---|
| multiple **orthogonal** adversaries | Han, Baldwin & Cohn, EACL 2021 (2101.10001), "Diverse Adversaries for Mitigating Bias in Training" | claims improvement, not elimination; its own motivation concedes "current adversarial techniques only partially mitigate model bias"; and E&G already ablated adversarial ensembles. **Not a threat**, but it is the one fix we have not run ourselves — worth one sentence acknowledging it |
| abandon the min-max game: iterative projection | Ravfogel et al., ACL 2020 (INLP) | the field's actual successor; see Class 2 |
| closed-form / convex erasure | Ravfogel et al., ICML 2022 (R-LACE); Belrose et al., NeurIPS 2023 (LEACE) | explicitly built to avoid SGD-adversarial instability; **linear-only** guarantee. Our pilot |
| spectral / kernel removal | Shao, Ziser & Cohen, EACL 2023 (SAL) | non-adversarial, extends to nonlinear via kernels |
| MMD distribution matching instead of an adversary | **Prost et al., 2019 (1910.11779), MinDiff** (VERIFIED) | Google's production fairness team replacing adversarial debiasing explicitly because adversarial techniques "might generate instability in the training process"; **no head-to-head numeric comparison** is reported — the critique is asserted, not measured |
| HSIC / distance-correlation independence penalties | — | **LEAD**; nothing found that re-runs the E&G setup with HSIC and reports a clean win |
| MI minimisation | CLUB (above) | see 4.2 |

**A live base rate worth stating honestly:** the "adversarial removal is dead"
consensus is **not universal**. A 2025 taxonomy of bias-mitigation methods
(WOAH 2025, `2025.woah-1.1`) describes adversarial debiasing neutrally as
"flexible… an effective tool for bias mitigation" and **does not mention INLP,
R-LACE or LEACE at all**; and an ICML 2026 reward-modelling paper (§3.5) ships
gradient reversal. Within the concept-erasure subfield the consensus *is* that
projection-based closed-form methods have superseded adversarial training, but
that is a within-subfield view, not a field-wide verdict. **So our negative is
worth publishing** — it is not a restatement of something every reader already
believes.

### 4.5 The part with no precedent

We searched specifically for "adversarial debiasing increases reliance," "bias
amplification adversarial training," "debiasing backfires," and found **nothing**
reporting a monotonic increase in behavioural/ablation-measured reliance on the
adversarially-suppressed feature as λ rises. Adjacent but different: work on
fairness-constrained models being *more* vulnerable to data poisoning; and
shortcut-competition results where reliance on one shortcut falls when an easier
one appears.

**So the ablation-Δ escalation (pooled: .0275 → .0325 → .0363 → .0500 → .1023
across λ .1→5; bottleneck: +.018 → +.123) is, as far as this review can tell, a
new empirical finding.** It is *consistent* with Kumar et al.'s "may end up
destroying all task-relevant features," and the natural causal story is that
adversarial pressure destroys legitimate task signal faster than it destroys the
plant, leaving the plant relatively *more* necessary. No paper makes that
connection explicitly. This should be framed in the paper as a contribution, not
as a replication.

Also note what our result adds beyond E&G: **behavioural** evidence (ablation of
the channel changes the score) rather than probe-only evidence. Amnesic probing
(Elazar et al., TACL 2021) is the methodological precedent for insisting on that
distinction; our battery's V2 gate implements it.

---

## Class 5. Data-side interventions (counterfactual augmentation, filtering, balancing)

**Bottom line: the literature backs our rejection of counterfactual text editing,
and does so more strongly than the rejection memo claims — including one formal
impossibility-flavoured result.**

### 5.1 CAD fidelity problems — the citations that support our standing rejection

| cite | finding |
|---|---|
| Kaushik, Hovy & Lipton, ICLR 2020 | the original positive: human minimal-edit counterfactual revisions of IMDb/SNLI; combined original+CAD training reduces sensitivity to spurious features |
| **Huang, Liu & Bowman, EMNLP 2020 Insights** | **failed replication**: CAD-augmented SNLI does *not* generalise better than unaugmented data of matched size, and can be *less* robust on challenge sets |
| **Joshi & He, ACL 2022** | the mechanistic critique closest to our wording: the features CAD perturbs *are* robust, but forcing training onto them **crowds out** other robust features; CAD can **exacerbate** spurious correlations elsewhere. Root cause: **lack of perturbation diversity** — editors make the same small set of edits |
| Kaushik, Setlur, Hovy & Lipton, ICLR 2021 | the *original authors'* own retreat: linear-Gaussian analysis showing noise on **causal** features degrades OOD while noise on non-causal features helps — CAD only helps if editors happen to touch the right features. This is precisely our "form ≈ content" worry, formalised |
| **Chandra Mouli, Zhou & Ribeiro, UAI 2022, "Bias Challenges in Counterfactual Data Augmentation"** | **formal result**: if augmentation is done by a "context-guessing machine" (what any human or LLM editor is), the resulting representation is **not** counterfactual-invariant; they construct an NLP task where CAD provably fails |
| **Sen et al., NAACL 2022** | editing **introduces new artifacts**: construct-driven CAD makes hate/sexism models ignore context, inflating false positives on benign identity-term uses |
| Khashabi et al., EMNLP 2020 | the cheaper alternative (natural perturbation of seed examples, label may change); still rewrite-based |

### 5.2 LLM-generated counterfactuals: the fidelity metric is itself unreliable

- Polyjuice (Wu et al., ACL 2021) and Tailor (Ross et al., ACL 2022) control the
  *intent* of the edit (control codes / semantic roles) but publish **no
  independent isolation guarantee** that nothing else changed. AutoCAD (Wen et
  al., EMNLP 2022 Findings) automates rationale-span intervention, evaluated on
  downstream metrics only.
- **Sen et al., EMNLP 2023, "People Make Better Edits"**: human CAD > ChatGPT >
  Polyjuice/Flan-T5; the automated failure mode is **under-editing** (edits
  "often insufficient to flip the original label").
- **Wang et al., INLG 2025, "Truth or Twist?"**: studies the standard fidelity
  metric — **Label Flip Rate** — across 2 methods × 3 datasets × 4 generators ×
  **15 judge models** + a 90-person user study. Finding: LFR is measured
  **inconsistently depending on which LLM judges it**, with a "considerably large
  gap" to human judgment even at the best configuration; conclusion is that a
  **fully automated counterfactual-augmentation pipeline is inadequate**.

**Verdict: ALREADY REJECTED, correctly, and now well-cited.** Note honestly in the
paper that the *published* failure mode is mostly under-editing, whereas our
stated worry is over-editing (removing taste-bearing content). Joshi & He 2022,
Kaushik et al. ICLR 2021, and Chandra Mouli et al. UAI 2022 are the three that
support our direction specifically. Do not re-propose text editing.

### 5.3 Filtering and balancing — the non-editing data-side options

- **Dataset Cartography** (Swayamdipta et al., EMNLP 2020): training on the
  **ambiguous** third alone beats training on 100% of the data for OOD; "hard"
  examples often correlate with label errors. The paper is itself partly an
  argument *against* naive filtering.
- **AFLite** (Le Bras et al., ICML 2020): iteratively removes examples a linear
  ensemble can predict from surface features; SNLI model accuracy on the filtered
  set drops ~92% → ~62% while human accuracy holds. Guarantees only statistical
  bias reduction relative to the chosen feature representation.
- **Fit verdict:** neither edits content, so both dodge our fidelity objection —
  but they trade it for the assumption that *shortcut-solvable ⇒ spurious*, which
  Cartography's own data undercut. Worth naming as the honest answer to a
  reviewer's "why not just filter the training data?": filtering has a documented
  signal-loss failure mode of its own.

---

## Class 6. LLM-era specifics — reward models and LLM judges

This class is the most directly relevant to our program and it contains **the
closest published precedent for both of our adopted/piloted instruments**.

### 6.1 Precedent for our stacked-increment readout: regression control at readout time

**Dubois, Galambosi, Liang & Hashimoto, COLM 2024, "Length-Controlled
AlpacaEval"** (VERIFIED mechanism). Fits a **GLM** over pairwise preferences with
covariates for model identity, instruction difficulty, and a nonlinear function of
the normalised output-length difference; the **length-controlled win rate** is the
GLM's prediction **conditioned on zero length difference**. The authors frame it
explicitly as the counterfactual "what would the win rate be if outputs had
matched length?" Reported: Spearman correlation with (then-unstyled) Chatbot Arena
0.94 → 0.98, and much harder to game by verbosity.

**Chatbot Arena "Style Control"** (LMSYS blog, 2024-08-28; VERIFIED, **blog not
peer-reviewed**). Joint logistic Bradley-Terry with model-quality coefficients β
and style coefficients γ (length difference + markdown header/bold/list count
differences) fit **simultaneously in one optimisation** — not a two-stage
residualise-then-fit. Their own caveat, worth quoting in our limitations because
it is our caveat verbatim: *"our analysis is still observational... there are
possible unobserved confounders such as positive correlation between length and
substantive quality that are not accounted for."*

**Verdict: cite both as precedent for the FORM of instrument (a).** They establish
that regression-adjustment-at-readout is the field-standard move for exactly our
problem, and their published caveats (observational, not causal) plus Westfall &
Yarkoni's reliability critique (§X.1) give us the honest limitations paragraph.

### 6.2 Precedent for decorrelated training: ODIN — and why it is NOT gradient reversal

**Chen et al., ICML 2024, "ODIN: Disentangled Reward Mitigates Hacking in RLHF"**
(VERIFIED against full text). Two linear heads on a **shared backbone**: quality
head r^Q and length head r^L. The objective is the ranking loss **plus additive
penalties**:
- correlation term `L^L = |ρ(r^Q, L(y))| − ρ(r^L, L(y))` — drive the quality
  head's Pearson correlation with length to **zero** while driving the length
  head's to **one** (the nuisance head's job is to *absorb* the channel);
- weight-orthogonality term `L^O = |W_Q · W_L^T|`;
- full: `L^R + λ_L[L^L(y_w) + L^L(y_l)] + λ_O L^O`. At RL time the length head is
  **discarded**.

Reported: reward-length Pearson 0.451 → −0.03. **No gradient reversal anywhere in
the graph.** Acknowledged limits: cites Locatello et al. 2019 that unsupervised
disentanglement is impossible without inductive bias; minibatch training limits
OOD generalisation of the disentanglement; only length-hacking is evaluated, and
generalisation to other hack types is called future work.

**This is the most actionable finding in the review after Byrd & Lipton.** ODIN is
the same architectural family we retired (an auxiliary nuisance head on a shared
trunk) with a **different coupling**: an additive statistical-decorrelation
penalty on head *outputs* plus weight orthogonality, instead of a min-max game on
reversed gradients. Our audit showed the min-max game is what fails (the encoder
defeats its own adversary); a penalty with no adversary has no game to defeat.
See adoption rec #1.

**RRM (Liu, Xiong, Ren et al., ICLR 2025)** (VERIFIED): a causal DAG separating a
contextual signal S from a context-free artifact A, with counterfactual
augmentation by **permuting responses across different prompts** (plus "neutral"
tie pairs), merged into the standard Bradley-Terry loss. Proposition 3.2 claims
the augmentation removes the A→C edge. Gains: RewardBench 80.61 → 84.15;
AlpacaEval-2 LC win rate 33.46 → 52.49. Limits: math/coding *drops* ~4%; validated
mainly against **artificially injected** artifacts (inserted prefixes) — i.e. a
planted battery, which corroborates our methodology. **Design idea worth noting:**
RRM gets counterfactuals **without rewriting any text** — it permutes *pairings*.
That is a form our standing rejection does not forbid, and it is a near-relative
of our matched sampling.

**ArmoRM (Wang et al., EMNLP 2024 Findings)**: multi-objective absolute-rating
reward model with a MoE gate; verbosity is a named objective that can be
down-weighted at inference. Decomposition/exposure rather than decorrelation —
closer in spirit to our A-bank + stack than to a debiasing method.

Leads not verified in mechanism: Wang et al. 2025 "Beyond Reward Hacking: Causal
Rewards" (2501.09620, counterfactual-invariance regulariser); Xu et al. 2025
"Reward Models Identify Consistency, Not Causality" (2502.14619); Ye, Zheng &
Zhang 2025 "Rectifying Shortcut Behaviors in Preference-based Reward Learning"
(2510.19050).

### 6.3 Judge bias — magnitudes we should know, and one 2026 signal that the target moved

- **Singhal et al., COLM 2024, "A Long Way to Go"**: across WebGPT/Stack/RLCD, a
  **purely length-based reward reproduces most of RLHF's gain** over the SFT
  baseline. The scale-of-confound citation.
- **Zheng et al., NeurIPS 2023 D&B (MT-Bench)**: position-swap consistency —
  Claude-v1 23.8%, GPT-3.5 46.2%, GPT-4 65.0%; under a repetitive-list verbosity
  attack the judge preferred the padded answer 91.3% (Claude-v1, GPT-3.5) vs 8.7%
  (GPT-4); self-enhancement ~+10 pts (GPT-4) / ~+25 pts (Claude-v1), with the
  authors' own caveat that they cannot separate it from genuine quality.
- **Wang et al., ACL 2024, "LLMs are not Fair Evaluators"**: reordering alone
  makes Vicuna-13B "beat" ChatGPT on 66/80 queries. Fixes: multiple-evidence
  calibration, balanced position calibration, human-in-the-loop routing.
- **CALM (Ye et al., ICLR 2025)**: 12 judge bias types via principle-guided
  perturbation injection — a planted-perturbation battery for judges.
- **RM-Bench (Liu et al., ICLR 2025 Oral)**: 3 chosen/3 rejected per prompt
  varying **only in style** (concise / detailed-plain / detailed-markdown). ~40
  reward models average **46.6% — below chance** under style interference. This
  is a ready-made off-the-shelf style-robustness test for our dense scorers.
- **Sharma et al., ICLR 2024 (sycophancy)**: both humans and preference models
  prefer convincingly-written sycophantic responses over correct ones a
  non-negligible fraction of the time — sycophancy is partly a *reward-model*
  artifact.
- SNIPPET/low-confidence but directionally important for us in 2026: a
  single-author 2026 preprint (2604.23178) reports that for current-generation
  judges **style bias (0.76-0.92) now dominates position bias (≤0.04)**, and that
  judges show a *conciseness* preference rather than the classic verbosity
  preference. Not peer-reviewed; do not quote as fact. But it argues our declared
  nuisance set should include **markdown/format** channels, not just length.

---

## Class 7. Evaluation methodology — planted synthetic shortcuts as an instrument check

This is the class where our practice has the *strongest* published backing, and it
is worth saying so in the paper, because the planted battery is the thing that
saved us from publishing a false PASS.

### 7.1 The closest precedent: Bastings et al., EMNLP 2022 — a protocol built on planted lexical shortcuts

`"Will You Find These Shortcuts?" A Protocol for Evaluating the Faithfulness of
Input Salience Methods for Text Classification` (Bastings, Ebert, Zablotskaia,
Sandholm, Filippova). VERIFIED by reading the PDF (pp. 977-979).

Their six-step protocol is, step for step, the shape of our battery:

| Bastings et al. step | our battery |
|---|---|
| 1. Define a shortcut *type* | `⟦QX7⟧` planted token, mechanical append |
| 2. Build a **partially synthetic** dataset (shortcut tokens inserted, label set by the shortcut) + a **fully synthetic test set** | planted arm vs unplanted arm, same rows/splits |
| 3. Train two models of the same architecture, on original and on mixed data; both must perform comparably on the *unmodified* test set | B00 vanilla-real vs B01 vanilla-planted parity check |
| 4. **"Verify that the shortcut tokens can indeed be assumed to be the ground truth of token importance for the model trained on the mixed data (by measuring accuracy)"** | **V1 EXPLOIT gate** |
| 5-6. Run the method under test; score against ground truth | V2 REMOVAL gate (probe + ablation) |

Details worth copying:
- They plant **three** shortcut types of increasing subtlety: **single token (st)**
  (presence of `#0`/`#1` determines the label), **token-in-context (tic)** (the
  indicator only fires if a second special token is present in the same input),
  and **ordered pair (op)** (the *order* of `#0`,`#1` determines the label;
  neither token is predictive alone; tokens placed ≤50 tokens apart).
- Shortcut tokens are **added to the vocabulary explicitly** so "the shortcut is
  unambiguous with regard to the label and its significance to the model
  increases" — the same reason we verified `⟦QX7⟧` tokenises to a stable 7-token
  prefix on 5,597/5,597 rows.
- **20%** of the modified dataset is synthetic, "not tuned but picked so that the
  shortcut data is sufficiently large to be picked up by the model but not too
  large to deteriorate the performance on the unmodified data."
- Their Fig. 3 verification is exactly our planted-vs-unplanted contrast: the
  mixed-data model scores **100%** on the synthetic test set vs **50%** (chance)
  for the control model trained on original data only, while both score 92-93% on
  the original test set.
- They also decontaminate a subtle confound we should copy: for multi-token
  shortcuts they inject one of the two tokens at random into *unmodified* rows
  without changing the label, "to mitigate the potential problem of making
  synthetic examples go off-manifold and thus being treated differently by the
  model."

**Verdict: this is our methodology, published, in our modality, four years
earlier.** We should cite it as the precedent for the battery and adopt the
`tic`/`op` shortcut types as a difficulty ladder (see adoption rec #3). Their own
headline is a negative for a *different* instrument class (salience methods fail
to surface even simple planted shortcuts), which is itself useful to us: it is a
reason not to substitute attribution methods for ablation.

### 7.2 The ancestor arguments: instruments must be validated against known ground truth

- **Adebayo et al., NeurIPS 2018, "Sanity Checks for Saliency Maps"** — model- and
  data-randomisation tests; methods that are invariant to model parameters or to
  the labels "will not be helpful to debug a model." The generic form of our
  argument that an instrument which cannot fail is not an instrument.
- **Adebayo et al., NeurIPS 2020, "Debugging Tests for Model Explanations"** —
  builds models with a *known planted* spurious background and asks whether
  explanation methods detect it. Result: they diagnose the spurious-background
  bug but not mislabeled examples, and a human study found subjects fell back on
  model predictions rather than attributions. Follow-up, **Adebayo et al., ICLR
  2022, "Post hoc Explanations may be Ineffective for Detecting Unknown Spurious
  Correlation."**
- **Hewitt & Liang, EMNLP 2019, "Designing and Interpreting Probes with Control
  Tasks"** — *control tasks* (random word-type→output maps) and the *selectivity*
  metric. The direct precedent for our probe-validity controls (positive control
  on planted-vanilla = 1.000; negative controls at .525-.531 chance). Their point,
  that complex probes memorise, is why we also ran a **linear** probe replication
  (.998-.9998) — a selectivity argument in all but name.
- **Carlini et al., USENIX Security 2019, "The Secret Sharer"** — *canaries*
  inserted into training data at controlled rates, with an *exposure* metric.
  The cleanest statement in ML of the general move we are making: plant a known
  artifact so the measurement has ground truth. Worth citing as the general
  pattern even though the target quantity (memorisation) differs.
- **Geirhos et al., Nature Machine Intelligence 2020, "Shortcut learning in deep
  neural networks"** — the framing citation for the problem class.
- **Gardner et al., EMNLP 2021 (Findings), "Competency Problems"** — "for complex
  language understanding tasks, all simple feature correlations are spurious."
  A useful caution on the *well-posedness* of nuisance declaration: in a
  competency problem there is no clean line between "nuisance channel" and
  "real feature," which is one reason our program declares nuisances by a blind
  routing audit rather than by fiat, and reports the mixed-channel decomposition.

### 7.3 Independent corroboration from an adjacent field: LLM unlearning audits

The LLM-unlearning literature has converged on our exact epistemics from a
different direction — behavioural suppression ≠ removal, and the audit must use
a *fresh* reader/attack rather than the training-time objective:
- Position papers and audits report that "task-level metrics (e.g., forget
  accuracy) are insufficient to distinguish reversible forgetting from
  catastrophic failure, as surface-level performance collapse may occur while
  internal representations remain intact" (SNIPPET; see e.g. *Unlearning Isn't
  Deletion*, arXiv 2505.16831; *Position: LLM Unlearning Benchmarks are Weak
  Measures of Progress*, arXiv 2410.02879; *Does Machine Unlearning Truly Remove
  Knowledge?*, arXiv 2505.23270).
- Benign-relearning and prompt attacks recover "unlearned" content (arXiv
  2406.13356; arXiv 2506.10236).
This is the same failure signature we measured for GRL: the optimisation defeats
its own objective-time reader while the information and the behaviour survive.

### 7.4 Benchmarks with controllable planted shortcuts (if we want a difficulty ladder)

- **SpuCo** (Joshi et al., 2023; `BigML-CS-UCLA/SpuCo`) — SpuCoMNIST lets you dial
  spurious-feature *magnitude and variance*, plus label and core-feature noise;
  SpuCoAnimals is the realistic counterpart. Vision-only, so not directly usable,
  but the *design* (a difficulty dial on the planted feature) is what our battery
  lacks: our plant is near-perfectly predictive, i.e. the easiest possible case,
  which makes our GRL negative strong but tells us nothing about the regime where
  a method might work.

---

## Cross-cutting: two published critiques that hit our ADOPTED instruments

These are not about debiasing methods; they are about the two instruments we kept.
They belong in the paper's limitations, and one of them is an adoption
recommendation.

### X.1 The stacked-increment readout is an incremental-validity claim, and incremental validity is fragile under measurement error

Our readout — fit a nuisance model N on the named nuisance channels, then report
the increment of the full model over N — is exactly *incremental validity* tested
by hierarchical regression (ΔR², or in our case ΔAUC), the standard
psychometric design.

**Westfall & Yarkoni, PLoS ONE 2016, "Statistically Controlling for Confounding
Constructs Is Harder than You Think"** (VERIFIED, abstract fetched):

> "We use intuitive examples, Monte Carlo simulations, and a novel analytical
> framework to demonstrate that common strategies for establishing incremental
> construct validity using multiple regression analysis exhibit extremely high
> Type I error rates under parameter regimes common in many psychological
> domains. Counterintuitively, we find that error rates are highest—in some cases
> approaching 100%—when sample sizes are large and reliability is moderate."

Reported regime: peak error at **moderate reliability (~.3-.7)** of the control
measure, error rates **increasing monotonically with n**, exceeding 90% at n=100
with reliability .4 and large indirect effects, and "near certainty" at n≥1,000.
Recommended remedies: SEM / latent-variable specification of the control
construct, or sensitivity analysis across a range of assumed reliabilities.

**Why this is live for us, specifically.** Our nuisance model is built from
*LLM-judged nuisance scores* — a **single noisy indicator** per nuisance
construct. That is precisely the "controlling for the measured variable and
wrongly interpreting this as controlling for the underlying construct" case. Our
n is in the thousands and rising, which is the *bad* direction here. So the
stacked increment is biased **towards** declaring real signal beyond the nuisance
— i.e. towards our preferred conclusion. This deserves an explicit caveat and a
cheap fix (adoption rec #2).

Note the asymmetry that partially rescues us: unreliability inflates Type I error
for *positive* increment claims. Where we report a **null or negative** increment
(fused ≤ bank; a channel that adds nothing), the critique does not bite the same
way.

### X.2 Matched sampling on a scalar score: the propensity-score-matching paradox

**King & Nielsen, Political Analysis 2019, "Why Propensity Scores Should Not Be
Used for Matching"** (SNIPPET): matching on a scalar propensity score approximates
a *completely randomised* experiment rather than a *fully blocked* one, and
pruning by propensity distance can *increase* imbalance, model dependence and
bias once the sample is already roughly balanced.

Applicability to us is partial and should be stated precisely: our matched
sampling matches on the **nuisance score itself**, which is the covariate of
interest — that is coarsened-exact/blocking-style matching, the thing King &
Nielsen *recommend*, not propensity matching. The critique bites only if we ever
(i) collapse several nuisance channels into a single learned propensity-like
score and match on that, or (ii) prune aggressively after balance is already
achieved. Our joint-nuisance-score design in decorrelated training (§12 of the
design note) is exactly case (i) — so the caveat transfers there.

---
## Mechanism-class summary table

Fit verdicts are for **our** setting: LoRA adapters over a frozen Llama-3.1-8B, a
scalar head on a pooled 4096-d state, binary outcome y, **continuous**
LLM-judged nuisance scores (no discrete groups, no group-labelled validation set,
no human annotation), ~1k-50k rows per cell, AUC readouts, planted-token battery
available as the tuning/validation oracle.

| # | method | guarantee | main failure mode | fit verdict |
|---|---|---|---|---|
| 1 | **Group DRO** (Sagawa ICLR'20) | minimax convergence, **convex case only** | degenerates on overparameterised nets without strong regularisation | **doesn't fit** — needs a group label on every row |
| 1 | **Importance reweighting** (Shimodaira'00; Byrd & Lipton'19) | none for deep nets | **effect vanishes once the data are separable**; ESS collapse under extreme weights | **our (d), and the pitfall is exact** — near-perfect plant ⇒ separable ⇒ weights inert at convergence |
| 1 | **Subsampling / balancing** (Sagawa ICML'20; Idrissi CLeaR'22) | none | discards data; needs groups | **works where reweighting fails**, but our design forbids row deletion — flag as a caveat or run as comparator |
| 1 | **JTT** (Liu ICML'21) | none | needs a **group-labelled val set** to tune; closes only ~75% of the gap | **partial** — mechanism ports; substitute our planted cell for the val set |
| 1 | **LfF** (Nam NeurIPS'20) | none | assumes the shortcut is learned *faster*; degenerate if the bias model saturates instantly | **closest method cousin** — group-free, per-example continuous weights |
| 1 | **Spectral Decoupling** (Pezeshki NeurIPS'21) | NTK-regime analysis | guarantees only NTK/binary/two-feature | **mechanism citation** for why a strong nuisance starves the real signal; cheap complementary arm |
| 1 | **IRM** (Arjovsky'19) | invariance across environments | Rosenfeld ICLR'21: nonlinear IRM "can fail catastrophically"; Kamath: worse than ERM | **doesn't fit** — no environments |
| 2 | **INLP** (Ravfogel ACL'20) | none formal; linear guardedness reached empirically | MLP recovers the concept at 85% post-erasure; **−5.5 pts** BERT task accuracy; Kumar'22: iterating can destroy all information | **superseded by LEACE** — more damage, weaker guarantee |
| 2 | **LEACE** (Belrose NeurIPS'23) | **Cov(r(X),Z)=0 exactly**; no linear predictor under **any convex loss** beats constant — **for categorical one-hot Z**; minimal-distortion optimal | **OLS-loss-only for continuous Z**; nothing against nonlinear readers; oblique projection can move task directions when Z ⟂̸ signal | **our (c), keep** — but bin Z to one-hot, and scope the claim to "the score, a linear functional of the erased rep, cannot use the channel" |
| 2 | **R-LACE** (Ravfogel ICML'22) | rank-optimal linear erasure via convex relaxation | its own paper: RBF-SVM/1-layer MLP recover the concept at **>90%** after optimal rank-1 erasure | fits (drop-in alternative), same ceiling as LEACE |
| 2 | **Log-linear guardedness** (Ravfogel ACL'23) | Thm 3.2: composition-safe for **binary Z, binary Y, δ-discretised** predictors | **Thm 3.4: multiclass softmax recovers Z near-perfectly despite guardedness** | **the limit on any certificate we claim** |
| 2 | **SAL/kSAL** (Shao EACL'23) | heuristic covariance reduction, no optimality | own paper: nonlinear probes recover through K=30; more LM damage than LEACE | doesn't beat LEACE for us |
| 2 | **KRaM** (NeurIPS'23) | rate-distortion objective; no adversary-class guarantee | unverified vs LEACE | **worth reading** — the one method claiming **continuous** and vector-valued concepts |
| 2 | **JSE** (Holstege ICML'24) | none formal | assumes a two-orthogonal-subspace structure | **worth reading** — directly targets erasure being "overzealous"; tested on MultiNLI |
| 3 | **DFR** (Kirichenko ICLR'23) | none formal | needs a **group-balanced** reweighting set; fails on realistic medical data; 2025 re-analysis: the balance *is* the mechanism | **premise matches our architecture exactly**, but we lack the balanced set |
| 3 | **AFR** (Qiu ICML'23) | none formal; proves a ceiling for prediction-only reweighting | a few points below DFR on vision; ties it on text (CivilComments 68.7 vs 70.1) | **BEST NEW FIT** — continuous weights, no group labels, post-hoc on an existing checkpoint |
| 3 | **SELF / LFR** (2309.08534 / 2312.04893) | none | model-selection still wants some labels | fit; pseudo-group construction from disagreement or loss percentiles |
| 3 | **Logit correction** (Liu ICLR'23) | Prop 1: LC loss ≡ max group-balanced accuracy | assumes a **single one-hot** attribute | medium — our channels are continuous and co-occurring, outside the proof |
| 3 | **PoE / bias-expert** (Clark'19; Utama'20; Sanh'21) | none formal | **ID cost** (MNLI −4.2 pts, Clark'19); acutely sensitive to bias-model capacity (Sanh: OOD 97% / ID 28% at the largest weak model) | medium-high — our judge-scored channels are ready-made bias experts, but capacity/informativeness must be capped |
| 4 | **Adversarial removal / GRL** (Ganin'16; Xie'17; Zhang'18) | none; DANN admits no convergence proof | **encoder defeats its own adversary, information persists** (Elazar & Goldberg'18); Kumar'22 proves the proxy failure; Acuna'22: chaotic dynamics | **ALREADY TRIED, RETIRED** — our result matches consensus and exceeds it |
| 4 | **Diverse/orthogonal adversaries** (Han EACL'21) | none | E&G already ablated adversary ensembles without success | the one variant we did not run; not expected to change the verdict |
| 4 | **MinDiff / MMD, HSIC, distance correlation** (Prost'19) | none formal | instability critique asserted, never measured head-to-head | plausible non-adversarial substitute; **untested on this problem** |
| 5 | **Counterfactual text editing (CAD)** | none | Huang'20 failed replication; Joshi & He'22 crowds out other robust features and can worsen spurious structure; Chandra Mouli UAI'22 proves editor-generated CAD is not counterfactual-invariant; Sen'22 introduces new artifacts; Wang INLG'25 the fidelity metric itself is judge-dependent | **REJECTED, correctly** — do not re-propose |
| 5 | **Filtering (AFLite, Cartography)** | statistical bias reduction relative to a chosen feature set | assumes shortcut-solvable ⇒ spurious; Cartography shows "easy" ≠ useless | possible, honest answer to "why not filter" |
| 6 | **Length-controlled AlpacaEval / Arena style control** | none (explicitly **observational**) | unobserved confounders; authors' own caveat | **precedent for our instrument (a)** — cite both |
| 6 | **ODIN** (Chen ICML'24) | none formal | only length is demonstrated; disentanglement needs inductive bias (Locatello'19); minibatch-limited | **BEST REPLACEMENT for the retired GRL slot** — additive Pearson + orthogonality penalty, no min-max game |
| 6 | **RRM** (Liu ICLR'25) | Prop 3.2: augmentation removes the artifact→preference edge | math/coding drop ~4%; validated on injected artifacts | interesting: counterfactuals **without rewriting text**, by permuting pairings |
| 6 | **CARD / FiMi-RM / DynaCF / PRISM** | CARD: identifiability theorems | discrete surrogates; single named nuisance | directional evidence the field is converging on our problem |
| 7 | **Planted-shortcut protocol** (Bastings EMNLP'22) | — | — | **our battery, published four years earlier, same modality** |
| X | **Stacked increment (incremental validity)** | none | **Westfall & Yarkoni'16: Type I error up to ~100% when the control measure is unreliable and n is large** | **our (a) — needs a reliability caveat and a sensitivity analysis** |
| X | **Matched sampling on a scalar score** | none | King & Nielsen'19 PSM paradox (applies to *propensity*-style scalars, not to blocking on the covariate itself) | **our (a)** — bites only for the joint-nuisance-score variant |

---

## What we'd adopt tomorrow (3 recommendations)

### R1. Fix the two instruments we already ship — half a day, protects every number in the ledger

Two cheap corrections, both of which currently make our claims *stronger than the
evidence supports*:

**(a) LEACE: bin continuous Z to one-hot before erasing.** LEACE's "no linear
predictor under any convex loss" theorem is proven for categorical Z; §4.3 restricts
continuous Z to the **OLS loss only**. Our battery fits the eraser on standardized
continuous nuisance matrices (`Zs6`, `Zd8`, `Zl0`, `Zk` in `run_battery_leace.py`)
and then evaluates with median-binarised probes — the certificate and the readout
are different objects. Binning each nuisance into terciles/median one-hot columns
and erasing that recovers the full theorem at essentially zero cost. Also correct
the `leace/leace.py` docstring, which currently asserts the strong form. *Effort:
~2 hours + one battery rerun.*

**(b) Stacked increment: report a reliability sensitivity.** Westfall & Yarkoni
(PLoS ONE 2016) show that incremental-validity claims tested by hierarchical
regression have Type I error rates **approaching 100%** when the control measure
has moderate reliability and n is large — and error rates **rise with n**. Our
nuisance model is a single noisy LLM-judged indicator per construct, our n is in
the thousands, and the bias runs *towards* our preferred conclusion (real signal
beyond the nuisance). Minimum viable fix: **estimate nuisance-score reliability by
re-scoring a sample with a second judge seed/model**, then report the increment
under a range of assumed reliabilities (their recommended sensitivity analysis).
Where we report null or negative increments the critique does not bite the same
way — say so. *Effort: one re-scoring batch + a small analysis; ~half a day.*
Also add ESS = (Σw)²/Σw² reporting to any decorrelated-training arm (Cortes et
al. 2010) and a declared weight-clipping rule.

### R2. Run an AFR-style head reweighting arm — 1-2 days, no training loop, brand new capability

**Deep Feature Reweighting's premise is literally our architecture** (frozen
backbone + small head), and **AFR** (Qiu et al., ICML 2023) removes DFR's one
blocking requirement — the group-balanced set — by weighting each example
continuously from the first-stage model's own predicted probability,
μᵢ ∝ β_{yᵢ}·exp(−γ·p̂ᵢ). On text benchmarks it essentially ties group-labelled DFR
(CivilComments 68.7 vs 70.1 vs group DRO 69.9). For us:

- weights can come from our **existing nuisance-only model** (upweight rows where
  the nuisance model is confident and wrong) — which reuses machinery already
  built for the stacked increment;
- it is **post-hoc on checkpoints we already have** — no retraining of adapters,
  no GPU-days, and it can be a closed-form re-solve of the head on frozen features;
- the missing group-labelled validation slice that AFR/SELF/JTT all quietly need
  is exactly what our **planted cell** supplies: tune γ where ground truth is
  known, freeze, apply;
- and the 2025 re-analysis (2512.01766) tells us what to report: **how balanced
  the reweighting set ends up across (y × nuisance) cells** is the active
  ingredient, not the retraining procedure.

This is a strictly cheaper experiment than either LEACE or decorrelated training,
and it addresses the *usage* of the channel rather than its decodability — the
same target as our decorrelated-training gate. *Effort: 1-2 days, CPU-only.*

### R3. If a removal-style intervention is still wanted, replace GRL with ODIN's additive decorrelation penalty — not another adversary

**ODIN** (Chen et al., ICML 2024) occupies the same architectural slot we retired —
an auxiliary nuisance head on a shared trunk — with a fundamentally different
coupling: instead of a reversed gradient and a min-max game, it adds
`λ_L·(|ρ(r^Q, nuisance)| − ρ(r^L, nuisance)) + λ_O·|W_Q·W_L^T|` to the loss, so the
nuisance head is *rewarded* for absorbing the channel while the quality head is
*penalised* for correlating with it, and the nuisance head is discarded at
inference. Reported: reward-length Pearson 0.451 → −0.03. Our audit's root cause
was that the min-max game settles on "defeat the adversary, keep the channel"; a
penalty with no adversary has no game to defeat.

Two honest caveats to carry: ODIN demonstrates only the **length** nuisance and
explicitly declines to claim generality; and Sanh et al.'s capacity sweep warns
that if the nuisance channel carries too much of y, pushing against it destroys
the main model (their largest weak learner: OOD 97% / **ID 28%**) — which is
plausibly part of what our λ=1.0 arm was doing (eval AUC .8283 → .6898).
Implementation is a small edit to `train_grl.py` (swap the reversal for the
penalty terms) and the **entire existing battery, gates and orchestration apply
unchanged**. *Effort: ~1 day to implement + one battery run.* Priority: below R1
and R2 — the program's instruments of record do not depend on it.

**Explicitly NOT recommended:** more/stronger/ensembled adversaries (Elazar &
Goldberg ablated that in 2018); counterfactual text editing (Class 5 — the
rejection is now well-cited); group DRO or JTT as specified (no groups, no
group-labelled validation set); INLP (strictly dominated by LEACE on both
guarantee and collateral damage).

---

## Consensus check: where we sit relative to the literature

| our position | literature | verdict |
|---|---|---|
| GRL fails: adversary defeated, information persists for a fresh reader | Elazar & Goldberg 2018 (same mechanism, weaker effect); R-LACE Table 2 (adversarially-finetuned BERT still predicts gender at 98-99.4%); Kumar/Tan/Sharma NeurIPS 2022 (proves the proxy failure) | **MATCHES consensus**, and is **explained by theory** we should cite |
| our evidence is stronger than the precedent | E&G's gap is 49→56%; ours is chance→AUC .97-1.00, on **two** architectures, with **ground-truth** plants, and **behavioural** (ablation) as well as probe evidence | **EXCEEDS** the published demonstrations |
| E&G's probe evidence was criticised | Barrett et al. EMNLP 2019 — but they state their point is *"orthogonal to the main contribution of Elazar and Goldberg (2018)"*; their bar is out-of-sample probe generalisation | our design **clears their bar by construction** — say so |
| reliance *increases* under adversarial pressure | nothing found | **NO PRECEDENT** — frame as a contribution, not a replication |
| "everyone already knows GRL is dead" | a WOAH 2025 taxonomy still calls adversarial debiasing "an effective tool" and never mentions INLP/LEACE; an ICML 2026 reward-model paper ships gradient reversal | **the consensus is NOT universal** — the negative is worth publishing |
| planted-shortcut battery as instrument validation | Bastings et al. EMNLP 2022 — six-step protocol with partially synthetic data, a fully synthetic test set, an architecture-parity check, and an explicit **step 4 verify-the-model-learned-the-shortcut** gate; plus Hewitt & Liang control tasks, Adebayo sanity/debugging tests, Carlini canaries, and (independently) the LLM-unlearning audit literature | **MATCHES and is well-precedented** — we should cite Bastings et al. as the protocol precedent rather than presenting the battery as novel |
| our plant is near-perfectly predictive | nothing in the literature stress-tests reweighting or erasure at that difficulty (closest: LfF at 99.5% bias-aligned; Shuieh et al. 2025 sweeps only to 90%) | **GAP** — our battery would be the first data point, but also the hardest possible case, so a failure there does not condemn a method on real channels |

Two further gaps we can claim: (i) no published result erases a channel a fresh
probe reads at AUC .97-1.00 and reports post-erasure recovery; (ii) no published
result applies representation erasure and then **fine-tunes an adapter over the
same frozen backbone** to test whether the channel returns.

---

## Candidate bibliography (BibTeX, refs-shared.bib convention — DO NOT merge without review)

Tags per the shared convention: **VERIFIED** = claim confirmed against fetched
primary text; **SNIPPET** = from a search snippet or a secondary source that
quotes the primary; **LEAD** = real and on-topic, no claim confirmed. `keywords`
uses `class=` (mechanism class 1-7 above) and `role=` (precedent / critique /
theory / method / benchmark).

```bibtex
% ---------------------------------------------------------------------
% CLASS 1 — train-time distribution interventions
% ---------------------------------------------------------------------
@inproceedings{sagawa2020groupdro,
  author = {Sagawa, Shiori and Koh, Pang Wei and Hashimoto, Tatsunori B. and Liang, Percy},
  title = {Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization},
  booktitle = {ICLR}, year = {2020}, eprint = {1911.08731}, archivePrefix = {arXiv},
  keywords = {class=1; role=method},
  annote = {VERIFIED. Minimax convergence O(1/sqrt(T)) in the CONVEX case only; no NN generalisation bound. Needs a group label on every training row. Overparameterised nets need strong regularisation or the objective is inert: Waterbirds worst-group 60.0 -> 84.6 (strong L2) -> 86.0 (early stopping); CelebA 41.1 -> 88.3.}
}
@inproceedings{sagawa2020overparam,
  author = {Sagawa, Shiori and Raghunathan, Aditi and Koh, Pang Wei and Liang, Percy},
  title = {An Investigation of Why Overparameterization Exacerbates Spurious Correlations},
  booktitle = {ICML}, year = {2020}, eprint = {2005.04345}, archivePrefix = {arXiv},
  keywords = {class=1; role=theory},
  annote = {VERIFIED. THE load-bearing paper for decorrelated training: subsampling works in the overparameterised regime, upweighting/reweighting FAILS. CelebA reweighted worst-group error >60 pct overparameterised vs 25.6 underparameterised. Mechanism: on separable data the reweighted solution equals the unweighted max-margin solution.}
}
@inproceedings{byrd2019importance,
  author = {Byrd, Jonathon and Lipton, Zachary C.},
  title = {What is the Effect of Importance Weighting in Deep Learning?},
  booktitle = {ICML}, year = {2019}, eprint = {1812.03372}, archivePrefix = {arXiv},
  keywords = {class=1; role=critique},
  annote = {VERIFIED. "While importance weighting impacts deep nets early in training, so long as the nets are able to separate the training data, its effect diminishes over successive epochs." Mitigations: early stopping (their runs needed >100 epochs to stabilise), L2, batchnorm -- which they call "the wrong abstraction". They recommend sub-sampling over importance weighting for deep nets.}
}
@inproceedings{liu2021jtt,
  author = {Liu, Evan Z. and Haghgoo, Behzad and Chen, Annie S. and Raghunathan, Aditi and Koh, Pang Wei and Sagawa, Shiori and Liang, Percy and Finn, Chelsea},
  title = {Just Train Twice: Improving Group Robustness without Training Group Information},
  booktitle = {ICML}, year = {2021}, eprint = {2107.09044}, archivePrefix = {arXiv},
  keywords = {class=1; role=method},
  annote = {VERIFIED. Upweight the ERM error set. No train-time group labels but REQUIRES a group-labelled validation set to tune lambda_up and T. Worst-group: Waterbirds 72.6 -> 86.7 (group DRO 91.4); CelebA 47.2 -> 81.1 (88.9); MultiNLI 67.9 -> 72.6 (77.7); CivilComments 57.4 -> 69.3 (69.9). Closes ~75 pct of the ERM-to-groupDRO gap.}
}
@inproceedings{nam2020lff,
  author = {Nam, Junhyun and Cha, Hyuntak and Ahn, Sungsoo and Lee, Jaeho and Shin, Jinwoo},
  title = {Learning from Failure: Training Debiased Classifier from Biased Classifier},
  booktitle = {NeurIPS}, year = {2020}, eprint = {2007.02561}, archivePrefix = {arXiv},
  keywords = {class=1; role=method},
  annote = {VERIFIED. Closest method cousin to decorrelated training: no group labels, per-example CONTINUOUS weights W(x)=CE(f_B)/(CE(f_B)+CE(f_D)), with a GCE-amplified biased model. At 99.5 pct bias-aligned (the strongest published shortcut tested): CMNIST 35.34 -> 63.39; CelebA 70.25 -> 84.24.}
}
@inproceedings{pezeshki2021gradientstarvation,
  author = {Pezeshki, Mohammad and Kaba, Oumar and Bengio, Yoshua and Courville, Aaron and Precup, Doina and Lajoie, Guillaume},
  title = {Gradient Starvation: A Learning Proclivity in Neural Networks},
  booktitle = {NeurIPS}, year = {2021}, eprint = {2011.09468}, archivePrefix = {arXiv},
  keywords = {class=1; role=theory},
  annote = {VERIFIED. The MECHANISM citation for why a near-perfect nuisance starves gradient to the real signal (Thm 2: dz2*/d(s1^2) < 0). Spectral Decoupling regulariser is an L2 penalty on OUTPUT LOGITS. CMNIST ERM 23.7 vs SD 68.4; CelebA worst-group 40.35 vs 83.24.}
}
@inproceedings{idrissi2022simplebalancing,
  author = {Idrissi, Badr Youbi and Arjovsky, Martin and Pezeshki, Mohammad and Lopez-Paz, David},
  title = {Simple Data Balancing Achieves Competitive Worst-Group-Accuracy},
  booktitle = {CLeaR}, year = {2022}, eprint = {2110.14503}, archivePrefix = {arXiv},
  keywords = {class=1; role=critique},
  annote = {VERIFIED. Subsampling/reweighting balancing matches fancier robustness methods. Key line for us: "access to group information is most critical for model selection purposes, and not so much during training." Counter-evidence to a weights-only, no-row-deletion design.}
}
@inproceedings{cortes2010learningbounds,
  author = {Cortes, Corinna and Mansour, Yishay and Mohri, Mehryar},
  title = {Learning Bounds for Importance Weighting},
  booktitle = {NeurIPS}, year = {2010},
  keywords = {class=1; role=theory},
  annote = {VERIFIED. Generalisation bounds under importance weighting depend on the SECOND MOMENT of the weights (Renyi divergence between source and target), with guarantees for unbounded weights only under a bounded-second-moment assumption; identifies explicit failure cases. The formal basis for reporting ESS on any decorrelated arm.}
}
@article{shimodaira2000covariateshift,
  author = {Shimodaira, Hidetoshi}, title = {Improving Predictive Inference Under Covariate Shift by Weighting the Log-Likelihood Function},
  journal = {Journal of Statistical Planning and Inference}, volume = {90}, number = {2}, pages = {227--244}, year = {2000},
  keywords = {class=1; role=precedent},
  annote = {SNIPPET. Origin of the weighted log-likelihood estimator under covariate shift; the ancestor of "reweight to match a target distribution".}
}
@inproceedings{fang2020rethinkingiw,
  author = {Fang, Tongtong and Lu, Nan and Niu, Gang and Sugiyama, Masashi},
  title = {Rethinking Importance Weighting for Deep Learning under Distribution Shift},
  booktitle = {NeurIPS}, year = {2020},
  keywords = {class=1; role=critique},
  annote = {SNIPPET. Circularity specific to deep learning: estimating the density ratio needs a good representation, which needs training on the weighted objective. Proposes alternating ("dynamic") IW. Check our joint-nuisance density-ratio estimator against this.}
}
@article{arjovsky2019irm,
  author = {Arjovsky, Martin and Bottou, L\'eon and Gulrajani, Ishaan and Lopez-Paz, David},
  title = {Invariant Risk Minimization}, journal = {arXiv preprint}, year = {2019}, eprint = {1907.02893}, archivePrefix = {arXiv},
  keywords = {class=1; role=method},
  annote = {VERIFIED. Needs multiple labelled environments; no worst-group risk bound. Included mainly for its critique arc (see rosenfeld2021risks, kamath2021doesirm).}
}
@inproceedings{rosenfeld2021risks,
  author = {Rosenfeld, Elan and Ravikumar, Pradeep and Risteski, Andrej},
  title = {The Risks of Invariant Risk Minimization}, booktitle = {ICLR}, year = {2021}, eprint = {2010.05761}, archivePrefix = {arXiv},
  keywords = {class=1; role=critique},
  annote = {VERIFIED. In the non-linear case IRM "can fail catastrophically unless the test data are sufficiently similar to the training distribution"; concludes IRM and alternatives do not improve over ERM under realistic non-linear shift. The cautionary arc for any elegant invariance objective.}
}
@inproceedings{kamath2021doesirm,
  author = {Kamath, Pritish and Tangella, Akilesh and Sutherland, Danica J. and Srebro, Nathan},
  title = {Does Invariant Risk Minimization Capture Invariance?}, booktitle = {AISTATS}, year = {2021}, eprint = {2101.01134}, archivePrefix = {arXiv},
  keywords = {class=1; role=critique},
  annote = {SNIPPET. IRMv1 can fail on the very problems motivating IRM and sometimes generalises worse than unconstrained ERM.}
}
@inproceedings{creager2021eiil,
  author = {Creager, Elliot and Jacobsen, J\"orn-Henrik and Zemel, Richard},
  title = {Environment Inference for Invariant Learning}, booktitle = {ICML}, year = {2021}, eprint = {2010.07249}, archivePrefix = {arXiv},
  keywords = {class=1; role=method},
  annote = {VERIFIED. Infers environments/groups from a reference model's per-example loss, no environment labels. The principled route from a continuous nuisance score to pseudo-groups if we ever need one.}
}
@article{shuieh2025posttraining,
  author = {Shuieh, Julia and Singhal, Prasann and Shanker, Apaar and Heyer, John and Pu, George and Denton, Samuel},
  title = {Assessing Robustness to Spurious Correlations in Post-Training Language Models},
  journal = {arXiv preprint (ICLR 2025 Workshop on Spurious Correlation and Shortcut Learning)}, year = {2025}, eprint = {2505.05704}, archivePrefix = {arXiv},
  keywords = {class=1; role=benchmark},
  annote = {VERIFIED. Closest GENRE match to our design: SFT/DPO/KTO under controlled spuriousness strength 10-90 pct on three task families. Degradation is task-dependent, not universal; no method dominates. Stops at 90 pct and tests no mitigation -- the near-100 pct regime our plant occupies is untested.}
}
@article{du2022shortcutsurvey,
  author = {Du, Mengnan and He, Fengxiang and Zou, Na and Tao, Dacheng and Hu, Xia},
  title = {Shortcut Learning of Large Language Models in Natural Language Understanding},
  journal = {Communications of the ACM}, year = {2024}, eprint = {2208.11857}, archivePrefix = {arXiv},
  keywords = {class=1; role=survey},
  annote = {SNIPPET. Survey of shortcut types in NLU; the framing citation for the problem class in our modality.}
}
@inproceedings{yang2025razor,
  author = {Yang, Shuo and Prenkaj, Bardh and Kasneci, Gjergji},
  title = {{RAZOR}: Sharpening Knowledge by Cutting Bias with Unsupervised Text Rewriting},
  booktitle = {AAAI}, year = {2025}, eprint = {2412.07675}, archivePrefix = {arXiv},
  keywords = {class=1; role=method},
  annote = {SNIPPET. The field's other 2024-era answer to our exact problem, and it is EDIT-BASED. Cite as the named contrast to our standing rejection of text rewriting.}
}

% ---------------------------------------------------------------------
% CLASS 2 — representation-level concept erasure
% ---------------------------------------------------------------------
@inproceedings{belrose2023leace,
  author = {Belrose, Nora and Schneider-Joseph, David and Ravfogel, Shauli and Cotterell, Ryan and Raff, Edward and Biderman, Stella},
  title = {{LEACE}: Perfect Linear Concept Erasure in Closed Form},
  booktitle = {NeurIPS}, year = {2023}, eprint = {2306.03819}, archivePrefix = {arXiv},
  keywords = {class=2; role=method},
  annote = {VERIFIED. Def 2.3/Thm 3.1: for CATEGORICAL one-hot Z, Cov(X,Z)=0 iff no linear classifier beats the constant predictor under ANY convex loss; LEACE is the unique minimal-distortion linear guarding function under every inner-product norm. CRITICAL: sec 4.3 restricts CONTINUOUS Z to the ordinary-least-squares loss only -- the strong certificate does not extend. Explicit non-guarantee: "it is intractable to nondestructively edit X to prevent a general nonlinear adversary from recovering Z." Concept scrubbing bits/byte: Pythia-160M 0.90->2.79, Pythia-12B 0.62->3.20, LLaMA-7B 0.69->1.73.}
}
@inproceedings{ravfogel2020null,
  author = {Ravfogel, Shauli and Elazar, Yanai and Gonen, Hila and Twiton, Michael and Goldberg, Yoav},
  title = {Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection},
  booktitle = {ACL}, pages = {7237--7256}, year = {2020}, eprint = {2004.07667}, archivePrefix = {arXiv},
  keywords = {class=2; role=method},
  annote = {VERIFIED. No closed-form guarantee; linear guardedness reached iteratively. After 35 iterations a 1-layer ReLU MLP still recovers gender at 85.0 pct. Downstream: +1.9 (BOW), -5.1 (BWV), -5.51 (BERT). Sec 7: "there are no magic guarantees, and the burden of verification remains on the user."}
}
@inproceedings{ravfogel2022lace,
  author = {Ravfogel, Shauli and Twiton, Michael and Goldberg, Yoav and Cotterell, Ryan},
  title = {Linear Adversarial Concept Erasure}, booktitle = {ICML}, year = {2022}, eprint = {2201.12091}, archivePrefix = {arXiv},
  keywords = {class=2; role=method},
  annote = {VERIFIED. Rank-optimal linear erasure (chance at K=1 where INLP needs >20 dims). Their own negative: after optimal erasure "Both RBF-SVM and a ReLU MLP with 1 hidden layer of size 128 predict gender with above 90% accuracy." Table 2 is an INDEPENDENT REPLICATION of our GRL negative: adversarially-finetuned BERT still predicts gender at 98-99.4 pct. Collateral damage on frozen-BERT profession: INLP(50) 79.91->71.27 vs R-LACE(50) ->76.73.}
}
@inproceedings{ravfogel2023loglinear,
  author = {Ravfogel, Shauli and Goldberg, Yoav and Cotterell, Ryan},
  title = {Log-linear Guardedness and its Implications}, booktitle = {ACL}, pages = {9413--9431}, year = {2023}, eprint = {2210.10012}, archivePrefix = {arXiv},
  keywords = {class=2; role=theory},
  annote = {VERIFIED. Thm 3.2: for binary Z, binary Y and delta-discretised log-linear predictors, guardedness survives composition. Thm 3.4 (the threat): for K-Voronoi data, even under epsilon-guardedness against binary Z there EXISTS a K-class softmax model recovering Z almost perfectly. The formal limit on any erasure certificate we claim for a nonlinear or multi-bin readout.}
}
@inproceedings{kumar2022probing,
  author = {Kumar, Abhinav and Tan, Chenhao and Sharma, Amit},
  title = {Probing Classifiers are Unreliable for Concept Removal and Detection},
  booktitle = {NeurIPS}, year = {2022}, eprint = {2207.04153}, archivePrefix = {arXiv},
  keywords = {class=2; role=theory},
  annote = {VERIFIED. "these methods can be counter-productive: they are unable to remove the concepts entirely, and in the worst case may end up destroying all task-relevant features. The reason is the methods' reliance on a probing classifier as a proxy for the concept." Lemma 3.1: probes use non-concept features even under maximally favourable conditions. Thm 3.2: INLP mixing corruption; iterated, the representation norm goes to zero. Explicitly warns that amnesic-probing-style accuracy drops can be an erasure artifact. Applies to POST-HOC as well as adversarial methods -- read it against our LEACE gate too.}
}
@article{elazar2021amnesic,
  author = {Elazar, Yanai and Ravfogel, Shauli and Jacovi, Alon and Goldberg, Yoav},
  title = {Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals},
  journal = {TACL}, volume = {9}, pages = {160--175}, year = {2021}, eprint = {2006.00995}, archivePrefix = {arXiv},
  keywords = {class=2; role=precedent},
  annote = {VERIFIED. "a high probing accuracy does not necessarily imply that the model is using the probed information to make predictions"; "we measure the reliance of a model on a property by measuring the change in the model's predictions after removing that property". The published precedent for our V2 ablation-reliance gate.}
}
@inproceedings{shao2023gold,
  author = {Shao, Shun and Ziser, Yftah and Cohen, Shay B.},
  title = {Gold Doesn't Always Glitter: Spectral Removal of Linear and Nonlinear Guarded Attribute Information},
  booktitle = {EACL}, pages = {1611--1622}, year = {2023}, eprint = {2203.07893}, archivePrefix = {arXiv},
  keywords = {class=2; role=method},
  annote = {VERIFIED. SVD on cross-covariance, keep the LOWEST-covariance directions; kernel extension kSAL. Own admission: linear SVM at chance by K=2, but RBF/poly-2 classifiers stay high through K=30. More LM-loss collateral damage than LEACE in the every-layer scrubbing regime.}
}
@inproceedings{ravfogel2022kernelized,
  author = {Ravfogel, Shauli and Vargas, Francisco and Goldberg, Yoav and Cotterell, Ryan},
  title = {Adversarial Concept Erasure in Kernel Space}, booktitle = {EMNLP}, pages = {6034--6055}, year = {2022}, eprint = {2201.12191}, archivePrefix = {arXiv},
  keywords = {class=2; role=method},
  annote = {SNIPPET. Kernelised R-LACE; the certificate holds only for the kernel family used to construct it.}
}
@inproceedings{roychowdhury2023kram,
  author = {Basu Roy Chowdhury, Somnath and Monath, Nicholas and Dubey, Avinava and Ahmed, Amr and Chaturvedi, Snigdha},
  title = {Robust Concept Erasure via Kernelized Rate-Distortion Maximization},
  booktitle = {NeurIPS}, year = {2023}, eprint = {2312.00194}, archivePrefix = {arXiv},
  keywords = {class=2; role=method},
  annote = {SNIPPET. The one erasure method explicitly claiming CATEGORICAL, CONTINUOUS and VECTOR-VALUED concepts -- directly on point for our continuous nuisance scores. Rate-distortion objective, not zero-cross-covariance; no worst-case adversary-class guarantee comparable to LEACE Thm 3.1. Read in full before relying on it.}
}
@inproceedings{holstege2024jse,
  author = {Holstege, Floris and Wouters, Bram and van Giersbergen, Noud and Diks, Cees},
  title = {Removing Spurious Concepts from Neural Network Representations via Joint Subspace Estimation},
  booktitle = {ICML}, year = {2024}, eprint = {2310.11991}, archivePrefix = {arXiv},
  keywords = {class=2; role=method},
  annote = {LEAD. Motivated by existing concept-removal methods being "overzealous by inadvertently eliminating features associated with the main task of the model". Jointly identifies two orthogonal low-dimensional subspaces (spurious vs main-task). Evaluated on Waterbirds, CelebA and MultiNLI. Closest published method to what our V3 specificity gate certifies. Whether they test near-deterministic spurious signals is UNVERIFIED.}
}
@inproceedings{gonen2019lipstick,
  author = {Gonen, Hila and Goldberg, Yoav},
  title = {Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases in Word Embeddings But do not Remove Them},
  booktitle = {NAACL-HLT}, pages = {609--614}, year = {2019},
  keywords = {class=2; role=critique},
  annote = {VERIFIED. "current debiasing methods are actually hiding the bias rather than removing it from the embeddings, creating a false sense of fairness." Bias-by-neighbors 0.852 -> 0.734 after projection, i.e. most of it survives. The founding hides-not-removes result.}
}
@inproceedings{goldfarbtarrant2021intrinsic,
  author = {Goldfarb-Tarrant, Seraphina and Marchant, Rebecca and Mu\~noz S\'anchez, Ricardo and Pandya, Mugdha and Lopez, Adam},
  title = {Intrinsic Bias Metrics Do Not Correlate with Application Bias},
  booktitle = {ACL-IJCNLP}, pages = {1926--1940}, year = {2021},
  keywords = {class=2; role=critique},
  annote = {SNIPPET. Representation-level bias probes do not predict downstream application bias; measure the task. Supports gating on behaviour rather than on probe accuracy.}
}
@misc{xu2025unlearning,
  author = {Xu, Xiaoyu and Yue, Xiang and Liu, Yang and Ye, Qingqing and Hu, Haibo and Du, Minxin},
  title = {Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in {LLM}s},
  year = {2025}, eprint = {2505.16831}, archivePrefix = {arXiv},
  keywords = {class=2; role=critique},
  annote = {VERIFIED. Near-zero forget accuracy recovers to 80 pct+ after brief fine-tuning; retain accuracy rebounds to ~65 pct; "relearning on the forget set can often yield higher accuracy than that of the original model." Updates concentrate in shallow output-adjacent parameters "leaving deeper representations intact." The closest analogue to "does erasure survive further fine-tuning" for LLMs.}
}
@inproceedings{pham2024circumventing,
  author = {Pham, Minh and Marshall, Kelly and Cohen, Niv and Mittal, Govind and Hegde, Chinmay},
  title = {Circumventing Concept Erasure Methods For Text-to-Image Generative Models},
  booktitle = {ICLR}, year = {2024}, eprint = {2308.01508}, archivePrefix = {arXiv},
  keywords = {class=2; role=critique},
  annote = {VERIFIED. "Post hoc concept erasure in generative models provides a false sense of security." Five erasure methods defeated by Concept Inversion with NO weight changes. Different modality; the methodological lesson transfers: erasure that blocks the tested interface is not removal.}
}
@misc{roychowdhury2025limits,
  author = {Basu Roy Chowdhury, Somnath and Dubey, Avinava and Beirami, Ahmad and Kidambi, Rahul and Monath, Nicholas and Ahmed, Amr and Chaturvedi, Snigdha},
  title = {Fundamental Limits of Perfect Concept Erasure}, year = {2025}, note = {AISTATS 2025}, eprint = {2503.20098}, archivePrefix = {arXiv},
  keywords = {class=2; role=theory},
  annote = {SNIPPET. Information-theoretic erasure/utility frontier; "there seems to be an inherent tradeoff between erasure and retaining utility". Theorem statements not verified.}
}
@misc{akbari2025obliviator,
  author = {Akbari, Ramin and Afshari, Milad and Boddeti, Vishnu Naresh},
  title = {Obliviator Reveals the Cost of Nonlinear Guardedness in Concept Erasure},
  year = {2025}, note = {NeurIPS 2025 poster; arXiv identifier and date UNVERIFIED},
  keywords = {class=2; role=method},
  annote = {LEAD. HSIC penalties plus constrained RKHS optimisation applied iteratively, explicitly to price NONLINEAR guardedness -- the closest published successor to the exact question our LEACE pilot faces. Venue/authors confirmed via the NeurIPS poster page; the arXiv id/date returned by fetch was internally inconsistent. Verify before citing.}
}

% ---------------------------------------------------------------------
% CLASS 3 — post-hoc surgery, logit correction, bias-expert ensembles
% ---------------------------------------------------------------------
@inproceedings{kirichenko2023dfr,
  author = {Kirichenko, Polina and Izmailov, Pavel and Wilson, Andrew Gordon},
  title = {Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations},
  booktitle = {ICLR}, year = {2023}, eprint = {2204.02937}, archivePrefix = {arXiv},
  keywords = {class=3; role=method},
  annote = {SNIPPET (claim VERIFIED, numbers triangulated from AFR and ExMap comparison tables, not the primary PDF). ERM nets learn adequate core features; the bias lives in the last layer. Retraining only the last layer on a small GROUP-BALANCED set matches group DRO. Premise matches our frozen-backbone-plus-head architecture exactly.}
}
@inproceedings{qiu2023afr,
  author = {Qiu, Shikai and Potapczynski, Andres and Izmailov, Pavel and Wilson, Andrew Gordon},
  title = {Simple and Fast Group Robustness by Automatic Feature Reweighting},
  booktitle = {ICML}, year = {2023}, eprint = {2306.11074}, archivePrefix = {arXiv},
  keywords = {class=3; role=method},
  annote = {VERIFIED (Table 1 and the weighting formula fetched). Last-layer retraining with CONTINUOUS per-example weights mu_i proportional to beta_{y_i} exp(-gamma p_hat_i) from the first-stage model's own predicted probability. NO group labels. Worst-group: Waterbirds 90.4 (DFR 92.9), CelebA 82.0 (88.3), MultiNLI 73.4 (74.7), CivilComments 68.7 (DFR 70.1, group DRO 69.9). Sec 5.2 proves a ceiling: on CelebA no function of first-stage predictions alone yields group-balanced weights. OUR TOP ADOPTION CANDIDATE.}
}
@misc{lastlayer2025unreasonable,
  title = {On the Unreasonable Effectiveness of Last-layer Retraining}, year = {2025}, eprint = {2512.01766}, archivePrefix = {arXiv},
  keywords = {class=3; role=critique},
  annote = {LEAD on authorship, VERIFIED on the claim (abstract fetched). The neural-collapse explanation of DFR was NOT supported; the evidence says DFR's success is almost entirely explained by better GROUP BALANCE in the held-out reweighting set. So report the balance of the reweighting set, not the retraining procedure.}
}
@article{le2023islastlayer,
  author = {Le, Phuong Quynh and Schl\"otterer, J\"org and Seifert, Christin},
  title = {Is Last Layer Re-Training Truly Sufficient for Robustness to Spurious Correlations?},
  journal = {arXiv preprint}, year = {2023}, eprint = {2308.00473}, archivePrefix = {arXiv},
  keywords = {class=3; role=critique},
  annote = {SNIPPET. On realistic medical imaging DFR "remains susceptible to spurious correlations" despite improving worst-group accuracy -- it helps but does not solve the problem outside curated benchmarks.}
}
@misc{self2023lastlayer,
  title = {Towards Last-layer Retraining for Group Robustness with Fewer Annotations}, year = {2023}, eprint = {2309.08534}, archivePrefix = {arXiv},
  keywords = {class=3; role=method},
  annote = {LEAD on authorship, SNIPPET on the claim. SELF builds the reweighting set from misclassifications or model DISAGREEMENT; proves disagreement upsamples worst-group data. Nearly matches DFR with zero group annotations and <3 pct of the class annotations.}
}
@inproceedings{liu2023logitcorrection,
  author = {Liu, Sheng and Zhang, Xu and Sekhar, Nitesh and Wu, Yue and Singhal, Prateek and Fernandez-Granda, Carlos},
  title = {Avoiding Spurious Correlations via Logit Correction}, booktitle = {ICLR}, year = {2023}, eprint = {2212.01433}, archivePrefix = {arXiv},
  keywords = {class=3; role=method},
  annote = {VERIFIED on mechanism and Proposition 1 (LC loss equivalent to maximising group-balanced accuracy); SNIPPET on numbers (Waterbirds 90.5, CelebA 88.1, CivilComments 70.3). Correction term ln P_hat(y, a_x) from a deliberately-biased GCE-trained proxy. Guarantee assumes a SINGLE ONE-HOT attribute -- our continuous co-occurring channels are outside the proof.}
}
@inproceedings{menon2021logitadjustment,
  author = {Menon, Aditya Krishna and Jayasumana, Sadeep and Rawat, Ankit Singh and Jain, Himanshu and Veit, Andreas and Kumar, Sanjiv},
  title = {Long-Tail Learning via Logit Adjustment}, booktitle = {ICLR}, year = {2021}, eprint = {2007.07314}, archivePrefix = {arXiv},
  keywords = {class=3; role=method},
  annote = {VERIFIED abstract. Fisher-consistent for balanced error under label-frequency shift; post-hoc or in-loss. WRONG PROBLEM for us (P(y) imbalance, not feature-conditional shortcut reliance) -- borrow only the post-hoc-on-frozen-model pattern.}
}
@inproceedings{clark2019dont,
  author = {Clark, Christopher and Yatskar, Mark and Zettlemoyer, Luke},
  title = {Don't Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Bias},
  booktitle = {EMNLP-IJCNLP}, pages = {4069--4082}, year = {2019}, eprint = {1909.03683}, archivePrefix = {arXiv},
  keywords = {class=3; role=method},
  annote = {VERIFIED (formulas and all numbers from full text). Product of experts p_hat = softmax(log p_i + log b_i); Learned-Mixin+H adds a learned gate plus an entropy penalty, without which the model zeroes the bias term. Requires a HAND-SPECIFIED bias model. ID cost is real: MNLI-m 78.73 -> 74.50 while HANS 50.58 -> 53.35; VQA-CP 39.18 -> 52.05.}
}
@inproceedings{utama2020towards,
  author = {Utama, Prasetya Ajie and Moosavi, Nafise Sadat and Gurevych, Iryna},
  title = {Towards Debiasing {NLU} Models from Unknown Biases},
  booktitle = {EMNLP}, pages = {7597--7610}, year = {2020}, eprint = {2009.12303}, archivePrefix = {arXiv},
  keywords = {class=3; role=method},
  annote = {VERIFIED author list via ACL Anthology bib. NOTE: often mis-cited as "Utama, Kassner, Gurevych" -- there is no Kassner co-author. AUTOMATIC bias proxy: a shallow copy of the same architecture trained on a tiny random subset (2,000 rows / 3 epochs for MNLI). MNLI ID 81.4-84.3 with HANS +7.1-8.2; QQP to PAWS +33.6-52.3. Annealing alpha 1 -> 0.8 recovers most of the ID cost.}
}
@inproceedings{utama2020mind,
  author = {Utama, Prasetya Ajie and Moosavi, Nafise Sadat and Gurevych, Iryna},
  title = {Mind the Trade-off: Debiasing {NLU} Models without Degrading the In-distribution Performance},
  booktitle = {ACL}, pages = {8717--8729}, year = {2020},
  keywords = {class=3; role=method},
  annote = {VERIFIED author list. Confidence regularisation instead of discarding biased examples; the lineage's own answer to the ID-cost critique. +7 pts HANS with in-distribution accuracy essentially retained (SNIPPET on the numbers).}
}
@inproceedings{sanh2021learning,
  author = {Sanh, Victor and Wolf, Thomas and Belinkov, Yonatan and Rush, Alexander M.},
  title = {Learning from Others' Mistakes: Avoiding Dataset Biases without Modeling Them},
  booktitle = {ICLR}, year = {2021}, eprint = {2012.01300}, archivePrefix = {arXiv},
  keywords = {class=3; role=method},
  annote = {VERIFIED from full text. Weak learner is TinyBERT (4M params, 2 layers, hidden 128); capacity limitation alone is the inductive bias, no domain knowledge. MNLI 84.52 -> 83.32; HANS non-entailed 26.74 -> 41.35. THE TRANSFERABLE WARNING: sweeping weak-model capacity 4.4M to 41.4M, "weaker learners assure good balance between OOD and ID... stronger learners encourage OOD generalization at the expense of ID performance" -- at the largest, OOD ~97 pct while ID collapsed to 28 pct on MNLI.}
}
@inproceedings{he2019drift,
  author = {He, He and Zha, Sheng and Wang, Haohan},
  title = {Unlearn Dataset Bias in Natural Language Inference by Fitting the Residual},
  booktitle = {Proceedings of the 2nd Workshop on Deep Learning Approaches for Low-Resource NLP (DeepLo)}, pages = {132--142}, year = {2019}, eprint = {1908.10763}, archivePrefix = {arXiv},
  keywords = {class=3; role=method},
  annote = {SNIPPET. DRiFt: the debiased model fits the RESIDUAL of a fixed biased (hypothesis-only) model. Numbers not extracted.}
}
@inproceedings{karimimahabadi2020endtoend,
  author = {Karimi Mahabadi, Rabeeh and Belinkov, Yonatan and Henderson, James},
  title = {End-to-End Bias Mitigation by Modelling Biases in Corpora},
  booktitle = {ACL}, pages = {8706--8716}, year = {2020}, eprint = {1909.06321}, archivePrefix = {arXiv},
  keywords = {class=3; role=method},
  annote = {SNIPPET. PoE plus debiased focal loss with hand-specified bias-only models. Numbers not extracted.}
}
@inproceedings{gururangan2018annotation,
  author = {Gururangan, Suchin and Swayamdipta, Swabha and Levy, Omer and Schwartz, Roy and Bowman, Samuel R. and Smith, Noah A.},
  title = {Annotation Artifacts in Natural Language Inference Data},
  booktitle = {NAACL-HLT}, pages = {107--112}, year = {2018},
  keywords = {class=3; role=precedent},
  annote = {VERIFIED. A hypothesis-only classifier recovers the label on ~67 pct of SNLI. The ancestor of our nuisance-only model: a partial-input baseline as an artifact audit.}
}
@inproceedings{poliak2018hypothesis,
  author = {Poliak, Adam and Naradowsky, Jason and Haldar, Aparajita and Rudinger, Rachel and Van Durme, Benjamin},
  title = {Hypothesis Only Baselines in Natural Language Inference},
  booktitle = {*SEM}, pages = {180--191}, year = {2018},
  keywords = {class=3; role=precedent},
  annote = {VERIFIED. Hypothesis-only baselines beat majority class on 10 NLI datasets; establishes partial-input baselines as standard practice.}
}
@inproceedings{mccoy2019right,
  author = {McCoy, R. Thomas and Pavlick, Ellie and Linzen, Tal},
  title = {Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference},
  booktitle = {ACL}, pages = {3428--3448}, year = {2019},
  keywords = {class=3; role=benchmark},
  annote = {VERIFIED. HANS: MNLI-trained models including BERT score near 0 pct on non-entailment cases that break three syntactic heuristics despite ~85 pct in-distribution accuracy.}
}
@inproceedings{feng2019misleading,
  author = {Feng, Shi and Wallace, Eric and Boyd-Graber, Jordan},
  title = {Misleading Failures of Partial-input Baselines}, booktitle = {ACL}, pages = {5533--5538}, year = {2019}, eprint = {1905.05778}, archivePrefix = {arXiv},
  keywords = {class=3; role=critique},
  annote = {VERIFIED. A SUCCESSFUL partial-input baseline proves a dataset is cheatable; a FAILING one does not prove it is clean. Augmenting a hypothesis-only model with trivial patterns solves 15 pct of SNLI examples previously classed as hard. The standing caveat on any clean stacked-increment result: it rules out only the channels we scored.}
}
@article{ng2025card,
  author = {Ng, Ignavier and Bl\"obaum, Patrick and Bhandari, Siddharth and Zhang, Kun and Kasiviswanathan, Shiva},
  title = {Debiasing Reward Models by Representation Learning with Guarantees},
  journal = {arXiv preprint}, year = {2025}, eprint = {2510.23751}, archivePrefix = {arXiv},
  keywords = {class=3; role=method},
  annote = {SNIPPET. VAE-style factorisation into causal vs spurious latents before the reward head, with identifiability theorems under two regimes: a SURROGATE PROXY for the spurious feature (our setting) or multiple raters with diverse preferences. Experiments use discrete/binary surrogates; continuous confounders not demonstrated.}
}

% ---------------------------------------------------------------------
% CLASS 4 — adversarial removal
% ---------------------------------------------------------------------
@inproceedings{elazar2018adversarial,
  author = {Elazar, Yanai and Goldberg, Yoav},
  title = {Adversarial Removal of Demographic Attributes from Text Data},
  booktitle = {EMNLP}, pages = {11--21}, year = {2018}, eprint = {1808.06640}, archivePrefix = {arXiv},
  keywords = {class=4; role=critique},
  annote = {VERIFIED. "while the adversarial component achieves chance-level development-set accuracy during training, a post-hoc classifier, trained on the encoded sentences from the first part, still manages to reach substantially higher classification accuracies on the same data." Advice: "do not rely on the adversarial training to achieve invariant representation to sensitive features." Their gap is modest (49.0 vs 56.0; ~50 vs 58.5) -- ours is chance vs AUC .97-1.00. IMPORTANT: they already ablated larger adversary capacity (500-8000 units), lambda sweeps, ADVERSARIAL ENSEMBLES, periodic reinitialisation, staged training and dropout; none closed the gap.}
}
@inproceedings{barrett2019adversarial,
  author = {Barrett, Maria and Kementchedjhieva, Yova and Elazar, Yanai and Elliott, Desmond and S{\o}gaard, Anders},
  title = {Adversarial Removal of Demographic Attributes Revisited},
  booktitle = {EMNLP-IJCNLP}, pages = {6330--6335}, year = {2019},
  keywords = {class=4; role=critique},
  annote = {VERIFIED. NOT a rebuttal of the failure finding: "Our results are also orthogonal to the main contribution of Elazar and Goldberg (2018), which is to show that adversarial debiasing is not always able to remove bias." Their point is that an in-sample diagnostic classifier may fit sample-specific correlations, so out-of-sample probe generalisation is the correct bar. Our planted ground-truth design with held-out probe evaluation clears that bar by construction.}
}
@article{ganin2016domain,
  author = {Ganin, Yaroslav and Ustinova, Evgeniya and Ajakan, Hana and Germain, Pascal and Larochelle, Hugo and Laviolette, Fran\c{c}ois and Marchand, Mario and Lempitsky, Victor},
  title = {Domain-Adversarial Training of Neural Networks},
  journal = {JMLR}, volume = {17}, number = {59}, pages = {1--35}, year = {2016},
  keywords = {class=4; role=method},
  annote = {SNIPPET. The origin of the gradient-reversal layer. Convergence to the saddle point is STATED, not proven; and removal is evaluated with a separate post-hoc linear SVM (Proxy A-distance), not with the co-trained adversary -- the founding paper already implicitly distrusts its own adversary as arbiter.}
}
@inproceedings{xie2017controllable,
  author = {Xie, Qizhe and Dai, Zihang and Du, Yulun and Hovy, Eduard and Neubig, Graham},
  title = {Controllable Invariance through Adversarial Feature Learning}, booktitle = {NeurIPS}, year = {2017}, eprint = {1705.11122}, archivePrefix = {arXiv},
  keywords = {class=4; role=method},
  annote = {SNIPPET. Positive-claim original; probe-based evaluation with no fresh post-hoc discriminator. Even their own numbers show incomplete removal (attribute accuracy 0.96 -> 0.57, well above chance). Analysis assumes a non-parametric infinite-capacity limit.}
}
@inproceedings{zhang2018mitigating,
  author = {Zhang, Brian Hu and Lemoine, Blake and Mitchell, Margaret},
  title = {Mitigating Unwanted Biases with Adversarial Learning}, booktitle = {AIES}, year = {2018},
  keywords = {class=4; role=method},
  annote = {SNIPPET. The canonical predictor-plus-adversary fairness formulation. No fresh-adversary re-attack test reported.}
}
@article{feder2021causalm,
  author = {Feder, Amir and Oved, Nadav and Shalit, Uri and Reichart, Roi},
  title = {{CausaLM}: Causal Model Explanation Through Counterfactual Language Models},
  journal = {Computational Linguistics}, volume = {47}, number = {2}, pages = {333--386}, year = {2021}, eprint = {2005.13407}, archivePrefix = {arXiv},
  keywords = {class=4; role=method},
  annote = {SNIPPET. Estimates the causal effect of a concept by adversarially pre-training a representation to "forget" it while keeping control concepts. Our GRL negative applies directly to this estimator family -- worth naming as an affected method.}
}
@inproceedings{acuna2022domain,
  author = {Acuna, David and Zhang, Guojun and Law, Marc T. and Fidler, Sanja},
  title = {Domain Adversarial Training: A Game Perspective}, booktitle = {ICLR}, year = {2022}, eprint = {2202.05352}, archivePrefix = {arXiv},
  keywords = {class=4; role=theory},
  annote = {SNIPPET. The GRL "transforms gradient descent into a competitive gradient-based algorithm which may converge to periodic orbits and other non-trivial limiting behavior that arise... in chaotic systems." A DIFFERENT failure mode from ours -- our adversary loss stayed in a normal range -- so citing it lets us rule out non-convergence as the explanation (except possibly for the unstable lambda=1.0 bottleneck arm).}
}
@article{mcallester2020formal,
  author = {McAllester, David and Stratos, Karl},
  title = {Formal Limitations on the Measurement of Mutual Information}, journal = {AISTATS}, year = {2020}, eprint = {1811.04251}, archivePrefix = {arXiv},
  keywords = {class=4; role=theory},
  annote = {SNIPPET. No distribution-free high-confidence MI LOWER bound from N samples can exceed O(ln N). The formal reason a healthy adversary loss cannot certify that information was removed.}
}
@inproceedings{zhao2019learning,
  author = {Zhao, Han and Tachet des Combes, R\'emi and Zhang, Kun and Gordon, Geoffrey J.},
  title = {On Learning Invariant Representations for Domain Adaptation}, booktitle = {ICML}, year = {2019}, eprint = {1901.09453}, archivePrefix = {arXiv},
  keywords = {class=4; role=theory},
  annote = {SNIPPET. Under label shift, driving representation invariance to zero forces joint error up (their Thm 4.3). A different mechanism from ours but a useful companion for the task-AUC collapse we observed under pressure.}
}
@inproceedings{prost2019mindiff,
  author = {Prost, Flavien and Qian, Hai and Chen, Qiuwen and Chi, Ed H. and Chen, Jilin and Beutel, Alex},
  title = {Toward a Better Trade-off Between Performance and Fairness with Kernel-based Distribution Matching},
  booktitle = {arXiv preprint / NeurIPS Workshop on Machine Learning with Guarantees}, year = {2019}, eprint = {1910.11779}, archivePrefix = {arXiv},
  keywords = {class=4; role=method},
  annote = {VERIFIED. MinDiff: Google's production fairness team replacing adversarial debiasing with an MMD regulariser because adversarial techniques "might generate instability in the training process". No head-to-head numeric comparison is reported -- the instability critique is asserted, not measured.}
}
@inproceedings{han2021diverse,
  author = {Han, Xudong and Baldwin, Timothy and Cohn, Trevor},
  title = {Diverse Adversaries for Mitigating Bias in Training}, booktitle = {EACL}, year = {2021}, eprint = {2101.10001}, archivePrefix = {arXiv},
  keywords = {class=4; role=method},
  annote = {SNIPPET. Multiple discriminators constrained to orthogonal hidden representations. Its own motivation concedes that current adversarial techniques only partially mitigate bias. The one adversarial variant we have not run; E and G already ablated adversary ensembles without success.}
}

% ---------------------------------------------------------------------
% CLASS 5 — data-side interventions
% ---------------------------------------------------------------------
@inproceedings{kaushik2020learning,
  author = {Kaushik, Divyansh and Hovy, Eduard and Lipton, Zachary C.},
  title = {Learning the Difference that Makes a Difference with Counterfactually-Augmented Data},
  booktitle = {ICLR}, year = {2020},
  keywords = {class=5; role=method},
  annote = {VERIFIED. The original positive CAD result: human minimal-edit revisions of IMDb and SNLI; combined original-plus-CAD training reduces sensitivity to spurious features.}
}
@inproceedings{huang2020counterfactually,
  author = {Huang, William and Liu, Haokun and Bowman, Samuel R.},
  title = {Counterfactually-Augmented {SNLI} Training Data Does Not Yield Better Generalization Than Unaugmented Data},
  booktitle = {Proceedings of the First Workshop on Insights from Negative Results in NLP}, year = {2020},
  keywords = {class=5; role=critique},
  annote = {VERIFIED. Failed replication under matched-size controls; CAD can be LESS robust on challenge sets. The cleanest failed-replication citation for our rejection of text editing.}
}
@inproceedings{joshi2022investigation,
  author = {Joshi, Nitish and He, He},
  title = {An Investigation of the (In)effectiveness of Counterfactually Augmented Data},
  booktitle = {ACL}, pages = {3668--3681}, year = {2022},
  keywords = {class=5; role=critique},
  annote = {VERIFIED. The mechanistic critique closest to our own wording: the perturbed features are robust, but forcing training onto them CROWDS OUT other robust features, and CAD can EXACERBATE spurious correlations elsewhere. Root cause: lack of perturbation diversity -- editors make the same small set of edits.}
}
@inproceedings{kaushik2021explaining,
  author = {Kaushik, Divyansh and Setlur, Amrith and Hovy, Eduard and Lipton, Zachary C.},
  title = {Explaining the Efficacy of Counterfactually Augmented Data}, booktitle = {ICLR}, year = {2021},
  keywords = {class=5; role=critique},
  annote = {VERIFIED. The original authors' own retreat: noise on CAUSAL features degrades OOD while noise on non-causal features helps -- CAD only helps if editors happen to touch the right features. This is our form-equals-content worry, formalised.}
}
@inproceedings{chandramouli2022bias,
  author = {Chandra Mouli, S. and Zhou, Yangze and Ribeiro, Bruno},
  title = {Bias Challenges in Counterfactual Data Augmentation}, booktitle = {UAI}, year = {2022},
  keywords = {class=5; role=theory},
  annote = {VERIFIED. FORMAL result: if augmentation is performed by a "context-guessing machine" (any human or LLM editor), the resulting representation is NOT counterfactual-invariant; they construct an NLP task where CAD provably fails. The strongest single citation for our standing rejection.}
}
@inproceedings{sen2022counterfactually,
  author = {Sen, Indira and Samory, Mattia and Wagner, Claudia and Augenstein, Isabelle},
  title = {Counterfactually Augmented Data and Unintended Bias: The Case of Sexism and Hate Speech Detection},
  booktitle = {NAACL}, year = {2022},
  keywords = {class=5; role=critique},
  annote = {VERIFIED. Editing INTRODUCES new artifacts: construct-driven CAD makes models ignore context, inflating false positives on benign identity-term uses.}
}
@inproceedings{sen2023people,
  author = {Sen, Indira and Assenmacher, Dennis and Samory, Mattia and Augenstein, Isabelle and van der Aalst, Wil and Wagner, Claudia},
  title = {People Make Better Edits: Measuring the Efficacy of {LLM}-Generated Counterfactually Augmented Data for Harmful Language Detection},
  booktitle = {EMNLP}, year = {2023},
  keywords = {class=5; role=critique},
  annote = {VERIFIED. Human CAD > ChatGPT > Polyjuice/Flan-T5. The automated failure mode is UNDER-editing: generated edits are "often insufficient to flip the original label".}
}
@inproceedings{wang2025truth,
  author = {Wang, Qianli and Nguyen, Van Bach and Feldhus, Nils and Villa-Arenas, Luis Felipe and Seifert, Christin and M\"oller, Sebastian and Schmitt, Vera},
  title = {Truth or Twist? Optimal Model Selection for Reliable Label Flipping Evaluation in {LLM}-based Counterfactuals},
  booktitle = {INLG}, year = {2025},
  keywords = {class=5; role=critique},
  annote = {VERIFIED. The standard CAD fidelity metric (Label Flip Rate) is measured INCONSISTENTLY depending on which LLM judges it, with a large gap to human judgment even at the best of 4 generators x 15 judges; concludes a fully automated counterfactual-augmentation pipeline is inadequate.}
}
@inproceedings{swayamdipta2020cartography,
  author = {Swayamdipta, Swabha and Schwartz, Roy and Lourie, Nicholas and Wang, Yizhong and Hajishirzi, Hannaneh and Smith, Noah A. and Choi, Yejin},
  title = {Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics},
  booktitle = {EMNLP}, pages = {9275--9293}, year = {2020},
  keywords = {class=5; role=method},
  annote = {VERIFIED. Training on the AMBIGUOUS third alone beats training on 100 pct of the data for OOD; hard-to-learn examples often correlate with label errors. Partly an argument against naive filtering.}
}
@inproceedings{lebras2020aflite,
  author = {Le Bras, Ronan and Swayamdipta, Swabha and Bhagavatula, Chandra and Zellers, Rowan and Peters, Matthew and Sabharwal, Ashish and Choi, Yejin},
  title = {Adversarial Filters of Dataset Biases}, booktitle = {ICML}, year = {2020},
  keywords = {class=5; role=method},
  annote = {VERIFIED. AFLite removes examples a linear ensemble predicts from surface features; SNLI model accuracy on the filtered set drops ~92 -> ~62 pct while human accuracy holds. Guarantees only statistical bias reduction relative to the chosen feature representation.}
}

% ---------------------------------------------------------------------
% CLASS 6 — reward models and LLM judges
% ---------------------------------------------------------------------
@article{dubois2024lengthcontrolled,
  author = {Dubois, Yann and Galambosi, Bal\'azs and Liang, Percy and Hashimoto, Tatsunori B.},
  title = {Length-Controlled {AlpacaEval}: A Simple Way to Debias Automatic Evaluators},
  journal = {COLM}, year = {2024},
  keywords = {class=6; role=precedent},
  annote = {VERIFIED mechanism. Fits a GLM over pairwise preferences with covariates for model identity, instruction difficulty and a nonlinear function of normalised length difference; the length-controlled win rate is the prediction CONDITIONED ON ZERO length difference, framed by the authors as a counterfactual. Spearman vs Chatbot Arena 0.94 -> 0.98. THE closest published precedent for our stacked-increment readout.}
}
@misc{lmsys2024style,
  author = {{LMSYS Org}}, title = {Does Style Matter? Disentangling Style and Substance in Chatbot Arena},
  howpublished = {LMSYS Blog}, year = {2024},
  keywords = {class=6; role=precedent},
  annote = {VERIFIED, but a BLOG POST, not peer-reviewed. Joint logistic Bradley-Terry with quality coefficients and four style covariates (length difference plus markdown header/bold/list counts) fit SIMULTANEOUSLY. Their own caveat is ours: "our analysis is still observational... there are possible unobserved confounders such as positive correlation between length and substantive quality that are not accounted for."}
}
@inproceedings{chen2024odin,
  author = {Chen, Lichang and Zhu, Chen and Soselia, Davit and Chen, Jiuhai and Zhou, Tianyi and Goldstein, Tom and Huang, Heng and Shoeybi, Mohammad and Catanzaro, Bryan},
  title = {{ODIN}: Disentangled Reward Mitigates Hacking in {RLHF}}, booktitle = {ICML}, year = {2024},
  keywords = {class=6; role=method},
  annote = {VERIFIED from full text. Two heads on a SHARED backbone plus ADDITIVE penalties -- NOT gradient reversal: L^L = |rho(r^Q, length)| - rho(r^L, length) drives the quality head's correlation with length to zero while the length head absorbs it, plus a weight-orthogonality term |W_Q W_L^T|; the length head is discarded at RL time. Reward-length Pearson 0.451 -> -0.03. Limits: cites Locatello 2019 that unsupervised disentanglement needs inductive bias; minibatch-limited; only length is evaluated. OUR RECOMMENDED REPLACEMENT for the retired GRL slot.}
}
@inproceedings{liu2025rrm,
  author = {Liu, Tianqi and Xiong, Wei and Ren, Jie and others},
  title = {{RRM}: Robust Reward Model Training Mitigates Reward Hacking}, booktitle = {ICLR}, year = {2025},
  keywords = {class=6; role=method},
  annote = {VERIFIED from full text. Causal DAG separating a contextual signal S from a context-free artifact A; counterfactual augmentation by PERMUTING RESPONSES ACROSS PROMPTS plus neutral tie pairs, merged into the Bradley-Terry loss. Prop 3.2 claims the A -> C edge is removed. RewardBench 80.61 -> 84.15; AlpacaEval-2 LC win rate 33.46 -> 52.49; math/coding DROPS ~4 pct. Validated against injected artifacts -- i.e. a planted battery. Note: counterfactuals WITHOUT rewriting text.}
}
@article{singhal2024longway,
  author = {Singhal, Prasann and Goyal, Tanya and Xu, Jiacheng and Durrett, Greg},
  title = {A Long Way to Go: Investigating Length Correlations in {RLHF}}, journal = {COLM}, year = {2024},
  keywords = {class=6; role=critique},
  annote = {VERIFIED. Across WebGPT, Stack and RLCD, a purely LENGTH-BASED reward reproduces most of RLHF's gain over the SFT baseline; the bias is localised to the reward model, described as non-robust and easily influenced by length in the preference data. The scale-of-confound citation.}
}
@inproceedings{shen2023looselips,
  author = {Shen, Wei and Zheng, Rui and Zhan, Wenyu and Zhao, Jun and Dou, Shihan and Gui, Tao and Zhang, Qi and Huang, Xuanjing},
  title = {Loose Lips Sink Ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback},
  booktitle = {Findings of EMNLP}, year = {2023},
  keywords = {class=6; role=method},
  annote = {VERIFIED. Product-of-experts reward model: a main expert trained jointly with a deliberately length-biased expert; only the main expert is used at RL time. Distinct from ODIN's correlation-penalty mechanism.}
}
@inproceedings{zheng2023judging,
  author = {Zheng, Lianmin and Chiang, Wei-Lin and Sheng, Ying and Zhuang, Siyuan and Wu, Zhanghao and Zhuang, Yonghao and Lin, Zi and Li, Zhuohan and Li, Dacheng and Xing, Eric P. and Zhang, Hao and Gonzalez, Joseph E. and Stoica, Ion},
  title = {Judging {LLM}-as-a-Judge with {MT}-Bench and Chatbot Arena},
  booktitle = {NeurIPS Datasets and Benchmarks}, year = {2023},
  keywords = {class=6; role=benchmark},
  annote = {VERIFIED magnitudes. Position-swap consistency: Claude-v1 23.8, GPT-3.5 46.2, GPT-4 65.0 pct. Repetitive-list verbosity attack success: Claude-v1 and GPT-3.5 91.3 pct, GPT-4 8.7 pct. Self-enhancement ~+10 pts (GPT-4) and ~+25 pts (Claude-v1), with the authors' own caveat that they cannot separate it from genuine quality.}
}
@inproceedings{wang2024notfair,
  author = {Wang, Peiyi and Li, Lei and Chen, Liang and Cai, Zefan and Zhu, Dawei and Lin, Binghuai and Cao, Yunbo and Liu, Qi and Liu, Tianyu and Sui, Zhifang},
  title = {Large Language Models are not Fair Evaluators}, booktitle = {ACL}, year = {2024}, eprint = {2305.17926}, archivePrefix = {arXiv},
  keywords = {class=6; role=critique},
  annote = {VERIFIED. Reordering alone makes Vicuna-13B "beat" ChatGPT on 66/80 queries. Fixes: multiple-evidence calibration, balanced position calibration, human-in-the-loop routing.}
}
@inproceedings{ye2025calm,
  author = {Ye, Jiayi and Wang, Yanbo and Huang, Yue and Chen, Dongping and Zhang, Qihui and Moniz, Nuno and Gao, Tian and Geyer, Werner and Huang, Chao and Chen, Pin-Yu and Chawla, Nitesh V. and Zhang, Xiangliang},
  title = {Justice or Prejudice? Quantifying Biases in {LLM}-as-a-Judge}, booktitle = {ICLR}, year = {2025},
  keywords = {class=6; role=benchmark},
  annote = {VERIFIED. The CALM benchmark: 12 judge bias types measured by automated principle-guided PERTURBATION INJECTION -- a planted-perturbation battery for judges, structurally the same move as our token battery.}
}
@inproceedings{liu2025rmbench,
  author = {Liu, Yantao and Yao, Zijun and Min, Rui and Cao, Yixin and Hou, Lei and Li, Juanzi},
  title = {{RM-Bench}: Benchmarking Reward Models of Language Models with Subtlety and Style},
  booktitle = {ICLR (Oral)}, year = {2025},
  keywords = {class=6; role=benchmark},
  annote = {VERIFIED. Three chosen and three rejected responses per prompt varying ONLY in style (concise / detailed-plain / detailed-markdown). ~40 reward models average 46.6 pct under style interference -- BELOW CHANCE. An off-the-shelf style-robustness test for our dense scorers.}
}
@inproceedings{sharma2024sycophancy,
  author = {Sharma, Mrinank and Tong, Meg and Korbak, Tomasz and Duvenaud, David and others},
  title = {Towards Understanding Sycophancy in Language Models}, booktitle = {ICLR}, year = {2024},
  keywords = {class=6; role=critique},
  annote = {VERIFIED. Both human annotators and preference models prefer convincingly-written sycophantic responses over correct ones a non-negligible fraction of the time -- evidence that sycophancy is partly a REWARD-MODEL artifact.}
}
@inproceedings{wang2024armorm,
  author = {Wang, Haoxiang and Xiong, Wei and Xie, Tengyang and Zhao, Han and Zhang, Tong},
  title = {Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts},
  booktitle = {Findings of EMNLP}, pages = {10582--10592}, year = {2024},
  keywords = {class=6; role=method},
  annote = {VERIFIED. Multi-objective absolute-rating reward model with a MoE gate; verbosity is a named objective that can be down-weighted at inference. Decomposition and exposure rather than decorrelation -- closer to our A-bank plus stack than to a debiasing method.}
}

% ---------------------------------------------------------------------
% CLASS 7 — evaluation methodology (planted shortcuts, control tasks)
% ---------------------------------------------------------------------
@inproceedings{bastings2022shortcuts,
  author = {Bastings, Jasmijn and Ebert, Sebastian and Zablotskaia, Polina and Sandholm, Anders and Filippova, Katja},
  title = {"Will You Find These Shortcuts?" A Protocol for Evaluating the Faithfulness of Input Salience Methods for Text Classification},
  booktitle = {EMNLP}, pages = {976--991}, year = {2022}, eprint = {2111.07367}, archivePrefix = {arXiv},
  keywords = {class=7; role=precedent},
  annote = {VERIFIED from the PDF (pp. 977-979). THE precedent for our planted battery, same modality, four years earlier. Six-step protocol: define a shortcut type; build a PARTIALLY SYNTHETIC dataset plus a fully synthetic test set; train matched models on original and mixed data that must perform comparably on the unmodified test set; "Verify that the shortcut tokens can indeed be assumed to be the ground truth of token importance for the model trained on the mixed data (by measuring accuracy)"; run the method; score. Three shortcut types: single token, token-in-context, ordered pair. Shortcut tokens are added to the vocabulary explicitly; 20 pct of the mixed dataset is synthetic; the mixed model scores 100 pct on the synthetic test set vs 50 pct chance for the control model. They also inject one of a multi-token shortcut's tokens into unmodified rows without changing the label, to avoid off-manifold artifacts.}
}
@inproceedings{hewitt2019control,
  author = {Hewitt, John and Liang, Percy},
  title = {Designing and Interpreting Probes with Control Tasks}, booktitle = {EMNLP-IJCNLP}, year = {2019}, eprint = {1909.03368}, archivePrefix = {arXiv},
  keywords = {class=7; role=precedent},
  annote = {VERIFIED. Control tasks (random word-type to output maps) and the SELECTIVITY metric; complex probes memorise a large number of labelling decisions independently of the representation. The precedent for our probe positive/negative controls and for replicating the MLP probe with a linear one.}
}
@inproceedings{adebayo2018sanity,
  author = {Adebayo, Julius and Gilmer, Justin and Muelly, Michael and Goodfellow, Ian and Hardt, Moritz and Kim, Been},
  title = {Sanity Checks for Saliency Maps}, booktitle = {NeurIPS}, year = {2018}, eprint = {1810.03292}, archivePrefix = {arXiv},
  keywords = {class=7; role=precedent},
  annote = {VERIFIED. Model- and data-randomisation tests; a method invariant to model parameters or to labels "will not be helpful to debug a model". The general form of our argument that an instrument which cannot fail is not an instrument.}
}
@inproceedings{adebayo2020debugging,
  author = {Adebayo, Julius and Muelly, Michael and Liccardi, Ilaria and Kim, Been},
  title = {Debugging Tests for Model Explanations}, booktitle = {NeurIPS}, year = {2020}, eprint = {2011.05429}, archivePrefix = {arXiv},
  keywords = {class=7; role=precedent},
  annote = {VERIFIED. Builds models with a KNOWN PLANTED spurious background and asks whether explanation methods detect it; they diagnose the spurious-background bug but not mislabeled examples, and a human study found subjects relied on predictions rather than attributions. Follow-up: Post hoc Explanations may be Ineffective for Detecting Unknown Spurious Correlation, ICLR 2022.}
}
@inproceedings{carlini2019secretsharer,
  author = {Carlini, Nicholas and Liu, Chang and Erlingsson, \'Ulfar and Kos, Jernej and Song, Dawn},
  title = {The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks},
  booktitle = {USENIX Security}, year = {2019}, eprint = {1802.08232}, archivePrefix = {arXiv},
  keywords = {class=7; role=precedent},
  annote = {SNIPPET. CANARIES inserted at controlled rates plus an EXPOSURE metric. The cleanest general statement of the move we are making: plant a known artifact so the measurement has ground truth. Different target quantity (memorisation).}
}
@article{geirhos2020shortcut,
  author = {Geirhos, Robert and Jacobsen, J\"orn-Henrik and Michaelis, Claudio and Zemel, Richard and Brendel, Wieland and Bethge, Matthias and Wichmann, Felix A.},
  title = {Shortcut Learning in Deep Neural Networks}, journal = {Nature Machine Intelligence}, volume = {2}, pages = {665--673}, year = {2020},
  keywords = {class=7; role=survey},
  annote = {VERIFIED. The framing citation: shortcuts are decision rules that perform well on standard benchmarks but fail to transfer to more challenging testing conditions.}
}
@inproceedings{gardner2021competency,
  author = {Gardner, Matt and Merrill, William and Dodge, Jesse and Peters, Matthew E. and Ross, Alexis and Singh, Sameer and Smith, Noah A.},
  title = {Competency Problems: On Finding and Removing Artifacts in Language Data},
  booktitle = {EMNLP}, year = {2021}, eprint = {2104.08646}, archivePrefix = {arXiv},
  keywords = {class=7; role=theory},
  annote = {VERIFIED. "For complex language understanding tasks, all simple feature correlations are spurious." A caution on the well-posedness of nuisance declaration: in a competency problem there is no clean line between nuisance channel and real feature.}
}
@article{steinmann2024navigating,
  author = {Steinmann, David and Divo, Felix and Kraus, Maurice and W\"ust, Antonia and Struppek, Lukas and Friedrich, Felix and Kersting, Kristian},
  title = {Navigating Shortcuts, Spurious Correlations, and Confounders: From Origins via Detection to Mitigation},
  journal = {arXiv preprint}, year = {2024}, eprint = {2412.05152}, archivePrefix = {arXiv},
  keywords = {class=7; role=survey},
  annote = {VERIFIED abstract. Formal definition of shortcuts bridging Clever Hans, spurious correlation and confounder terminology; organises detection and mitigation approaches and classifies datasets built to study shortcut learning.}
}
@article{ye2024cleverhans,
  author = {Ye, Wenqian and others},
  title = {The Clever Hans Mirage: A Comprehensive Survey on Spurious Correlations in Machine Learning},
  journal = {arXiv preprint}, year = {2024}, eprint = {2402.12715}, archivePrefix = {arXiv},
  keywords = {class=7; role=survey},
  annote = {VERIFIED. Four mitigation families: data-centric, representation learning, post-hoc, specialised. Consensus: no universal solution; group balancing can fail depending on parameterisation; scaling parameters "will not solve this problem and could even exacerbate spurious biases"; and GROUP-LABEL-FREE MITIGATION remains the key open problem. Notably it gives adversarial training only a brief mention.}
}

% ---------------------------------------------------------------------
% CROSS-CUTTING — critiques of our two ADOPTED instruments
% ---------------------------------------------------------------------
@article{westfall2016controlling,
  author = {Westfall, Jacob and Yarkoni, Tal},
  title = {Statistically Controlling for Confounding Constructs Is Harder than You Think},
  journal = {PLoS ONE}, volume = {11}, number = {3}, pages = {e0152719}, year = {2016},
  keywords = {class=X; role=critique},
  annote = {VERIFIED. "common strategies for establishing incremental construct validity using multiple regression analysis exhibit extremely high Type I error rates under parameter regimes common in many psychological domains. Counterintuitively, we find that error rates are highest -- in some cases approaching 100% -- when sample sizes are large and reliability is moderate." Peak error at control-measure reliability ~.3-.7; error rises MONOTONICALLY with n; >90 pct at n=100 with reliability .4 and large indirect effects. Remedies: SEM/latent specification of the control construct, or sensitivity analysis across assumed reliabilities. DIRECTLY THREATENS our stacked-increment readout, which controls for a single noisy LLM-judged indicator at n in the thousands.}
}
@article{king2019propensity,
  author = {King, Gary and Nielsen, Richard},
  title = {Why Propensity Scores Should Not Be Used for Matching}, journal = {Political Analysis}, volume = {27}, number = {4}, pages = {435--454}, year = {2019},
  keywords = {class=X; role=critique},
  annote = {SNIPPET. The PSM paradox: matching on a scalar propensity score approximates a completely randomised rather than a fully blocked experiment, and pruning by propensity distance can INCREASE imbalance, model dependence and bias once the sample is already balanced. Applies to us only if we match on a collapsed joint-nuisance propensity-like score; blocking on the nuisance covariate itself is what they recommend.}
}
@inproceedings{xu2020vinformation,
  author = {Xu, Yilun and Zhao, Shengjia and Song, Jiaming and Stewart, Russell and Ermon, Stefano},
  title = {A Theory of Usable Information Under Computational Constraints}, booktitle = {ICLR}, year = {2020}, eprint = {2002.10689}, archivePrefix = {arXiv},
  keywords = {class=X; role=theory},
  annote = {VERIFIED. Predictive V-information: informativeness relative to a restricted family of observers, estimable with PAC-style guarantees and, unlike Shannon MI, creatable by computation. The right formal frame for our probe-decodability vs score-usage distinction, and for LEACE's "linear reader" scoping. See also Ethayarajh et al., ICML 2022, pointwise V-usable information.}
}
```

---

## Provenance

Six parallel sweeps (five delegated by mechanism class, one run here for class 7
and the cross-cutting critiques), 2026-08-10, WebSearch + WebFetch. The session's
WebSearch budget (200 calls) was exhausted; a handful of items are therefore
tagged LEAD/SNIPPET where a confirming fetch was not possible. PDFs fetched
during the sweep are cached under this session's tool-results directory. Every
`annote` above carries the tag that the shared-bibliography convention requires;
**nothing here has been merged into `latex/refs-shared.bib`.**
