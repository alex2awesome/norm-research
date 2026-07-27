# From unsupervised recovery `R` to supervised accuracy `Y` — a map, the novelty, and the unification

*2026-06-19. We can certify a prompt/rubric optimal for the **label-free** recovery objective `R = I_TVD(M; M̂)`
(§6.7, brute-force / submodular). The question: what can we say about its accuracy against a **supervised**
target `Y`, ideally *without* spending many labels — and can the `Y`-side fit inside the same submodular/
bounded framework rather than spinning off a new one? This note maps the literature, positions our novelty
honestly, and states the unification. Lit verified 2026-06-19; "(unverified)" tags where I couldn't confirm
a specific claim.*

---

## 0. The setup (so the map lands on our objects)

Each candidate rubric is a binary predictor `M̂_S(x)` of items; the unsupervised pass gives us **every
candidate's full behavior on the whole unlabeled pool, for free**. We have a label-free target `M` (one
model's verdict, or a consensus). We want, for a rubric, its accuracy against the true label `Y` — and how
many `Y` (if any) that costs.

**The one fact everything turns on:** all the strong results require **conditional independence of the
predictors given the latent target** — and that is *exactly* our §6.6 submodularity condition. So the
`Y`-bridge and the optimization guarantee share a single structural hinge, which our **co-information /
`γ`** machinery already measures.

---

## 1. The unsupervised accuracy-estimation literature — three families

**Family A — latent-variable / conditional-independence (strongest claims).** Predictors CI given `Y` ⇒
accuracies recoverable from agreement alone.
- **Parisi–Nadler–Kluger** (PNAS 2014, arXiv 1303.3257): under CI-given-`Y` the **off-diagonal predictor
  covariance is rank-one**; the leading eigenvector's entries `∝` balanced accuracies → **rank** predictors
  and build the **Spectral Meta-Learner (SML)**, label-free.
- **Jaffe–Nadler** (AISTATS 2015, arXiv 1407.7644): + class imbalance, asymmetric errors via a 3-way tensor.
- **Steinhardt–Liang** (NIPS 2016, arXiv 1606.05313): most general — estimate the **risk** of a predictor
  (a whole family of losses, not just 0-1) from unlabeled data via method-of-moments on
  `E[f_i f_j | y] = E[f_i|y]E[f_j|y]`; even gives gradients of the estimated risk. This is the cleanest
  "**bound a supervised objective from unlabeled data**."
- Foundations: Dawid–Skene (1979) latent-class EM; Zhang–Chen–Zhou–Jordan (2014) spectral DS w/ error bounds;
  Platanios et al. (UAI 2014) agreement equations.
- *What you get:* **consistent estimates + rankings** of accuracy/risk, label-free, as unlabeled `n→∞`.

**Family B — weak supervision / data programming (CI + *modeled* dependencies; engineering-grade).**
Ratner et al. (Data Programming, NIPS 2016; Multi-Task Weak Supervision, arXiv 1810.02840; Snorkel): labeling
functions = *our criteria*. A generative model recovers LF **accuracies AND their correlation structure**
from agreement (matrix-completion / inverse-covariance), no labels, with sample complexity that improves in
unlabeled `n` and degrades with the dependency structure. **This is the version that handles correlated
errors** — our `γ<1` regime — rather than assuming them away (structure learning: Varma et al.).

**Family C — agreement/disagreement empirics (weakest assumptions, predictions/bounds not identities).**
ATC (Garg et al., ICLR 2022, arXiv 2201.04234); disagreement≈error (Jiang et al., ICLR 2022);
Agreement-on-the-Line (Baek et al., NeurIPS 2022); actual bounds in Disagreement-Discrepancy (NeurIPS 2023,
arXiv 2306.00312). Predict/bound accuracy from unlabeled agreement with little structure but need a
calibration anchor.

---

## 2. The prompt / LLM-judge landscape — the novelty check (am I missing something?)

The adjacent prior work is real but **none combines our three ingredients** (certified optimality +
recovery-grounding + the CI-hinge to `Y`):

- **Prompts as labeling functions** — *Language Models in the Loop* (Smith et al., 2022, arXiv 2205.02318):
  prompts → labeling functions → Snorkel denoising → train a downstream classifier. **Closest to "prompts as
  weak supervisors," but:** (i) the goal is to *train a downstream model*, not certify the prompt/rubric
  itself; (ii) **no optimality guarantee** on the prompt; (iii) **not grounded in consistency/reconstruction**
  (no `R`, no TVD-MI); (iv) no submodular/brute-force certification.
- **Unsupervised LLM-judge evaluation via consistency** — *Sage* (self- + logical consistency, no labels),
  *No-Knowledge Alarms for Misaligned LLMs-as-Judges* (arXiv 2509.08593: "logical consistency, not
  correctness, is the only tool available in no-knowledge situations"), *LLM-as-a-Jury* (reliability-aware
  aggregation, but explicitly: "without ground truth it is not possible to distinguish a consistent biased
  majority from a correct minority"). **These corroborate our framing and the `M`-vs-`Y` wall, but have NO
  optimality guarantees and NO link to accuracy-estimation theory.**
- **Peer prediction for label-free eval** — Miller–Resnick–Zeckhauser (2005); multi-task peer prediction
  (Shnayder et al., 2016, arXiv 1603.03151); **Robertson–Koyejo TVD-MI** (our metric foundation). Gaming-robust
  evaluation, but not prompt-*optimality*, not submodular, not the `Y`-bridge.
- **Accuracy-estimation theory** (Family A/B) is *not* applied to prompts/LLM-judges at all.

*"Baumann":* I could **not** verify a specific Baumann paper on this (searched prompt-eval / unsupervised-judge
/ consistency). If you have the exact title I'll slot it in; provisionally the closest named works are the
four bullets above. **(Action: get the Baumann cite before any novelty claim in writing.)**

**Net (honest) novelty.** Each *piece* exists; the **unification does not**: a framework that (a) **certifies a
prompt optimal** (brute-force/submodular on recovery, §6.7), (b) is **grounded in consistency + reconstruction**
with a gaming-robust f-divergence (TVD-MI), and (c) **bridges to label-free `Y`-accuracy via the same
conditional-independence structure**, with (d) **co-information/`γ` as the CI-validity diagnostic** that tells
you when both the optimality cert and the `Y`-estimate are trustworthy. The prompt-based instantiation *plus*
the guarantees *plus* the recovery grounding is, as far as 2026-06-19 search shows, new.

---

## 3. The fundamental wall — what no unlabeled data escapes (and where labels are unavoidable)

All of Family A/B recover accuracy **against the latent variable the predictors are CI *given*** — not
necessarily your `Y`. If the rubrics are independent noisy views of the *true* label → you get `Y`-accuracy
free. If they're independent views of a **shared bias** (the latent = "what they commonly track" = *our `M`*)
→ you recover accuracy w.r.t. that bias. **Unsupervised methods cannot tell these apart** (plus a sign/symmetry
ambiguity: a rubric vs. its negation). This is *exactly* §2.5/§4.3's "articulable wrong attribute," sharpened:

> The unsupervised accuracy estimate is against the **consensus latent `M`**, and equals `Y`-accuracy **iff
> `M = Y`** — the one thing unidentifiable without labels. The irreducible bridge is `I(M;Y)` (cheap: `M` is a
> single fixed function, `O(1/ε²)` labels), which **caps** label-recovery: `I(Y;M̂) ≤ min(R, I(M;Y))`.

So: a few dozen labels test `M=Y` and pin the ceiling; everything else (search, ranking, accuracy estimation)
is label-free under CI.

---

## 4. The unification — `Y`-guarantees live *inside* the bounded framework (one hinge)

**Conditional independence given the target is the single structural condition** that, when it holds (and
`γ`/co-information *checks* it), simultaneously delivers all three layers:

| layer | object | what CI-given-`T*` buys | breaks when (diagnostic) |
|---|---|---|---|
| **optimize** (§6.6/§6.7) | `R(S)=I(T*;X_S)` | monotone submodular ⇒ greedy/brute-force **certified-optimal rubric** | `γ<1` (synergy) — co-information `I(X_i;X_j∣T*)−I(X_i;X_j)>0` |
| **estimate** (Family A) | per-rubric accuracy vs `T*` | rank-one covariance ⇒ **label-free accuracy ranking / risk** (Parisi, Steinhardt–Liang) | `γ<1` (correlated errors fool agreement) — *same diagnostic* |
| **identify** (the wall) | `T* = Y ?` | — | needs `O(1/ε²)` labels to test `M=Y` |

The punchline: **the same `γ` that breaks submodularity breaks the unsupervised accuracy estimate** — correlated
criteria both (i) make recovery super-modular and (ii) make agreement a liar. So `γ` is a **unified validity
gauge**: `γ≈1` ⇒ *both* the optimality certificate *and* the `Y`-accuracy estimate hold; `γ<1` ⇒ both degrade
and you move to Family B (model the dependencies — for *both* purposes). Nothing new spun off; the `Y`-side is
the same condition, the same diagnostic, one extra `I(M;Y)` label-cost.

**A cleaner `M` for free.** Today `M` = one model's holistic verdict (single-LLM, §-limitation-5). The SML
(Parisi) recovers the **CI-optimal consensus latent** from the rubrics' agreement — provably more accurate than
most ensemble members — i.e. a *better, model-independent* `M`, *and* the per-rubric accuracy estimates, in one
spectral step. That directly upgrades the multi-executor consensus-`M` we just built (aggregate_executors.py)
from "mean-then-median-split" to "spectral meta-learner."

**The bounded story, end to end:** rung-1 cap (`R ≤ cap_f`, all strings) → rung-2/§6.7 within-class certified
optimum (brute-force `R(C(S))`) → **under CI (γ-checked): label-free accuracy ranking against `M` (Parisi) +
risk bound (Steinhardt–Liang)** → `+O(1/ε²)` labels to test `M=Y` and cap `I(Y;M̂) ≤ min(R, I(M;Y))`. One ladder,
the `Y`-rung gated by a single checkable assumption and a tiny label budget.

---

## 5. Next experiments (all reuse what we built; none is a new direction)

1. **SML as `M`.** Run Parisi's spectral meta-learner over the rubric behaviors (already saved per executor in
   `bfm_*.npz`) → consensus `M` + label-free per-rubric accuracy *ranking*. Compare to our mean-then-median `M`.
2. **`γ` as the CI gate.** We already compute co-information; report it as the **trust flag** on the SML
   accuracy estimates (high `γ` ⇒ estimates reliable). One number, two uses.
3. **The `M=Y` test (minimal labels).** On a task with real `Y` (e.g. competitive-code accept/reject, or any
   v2 task with `judgement`), spend ~30–100 labels to estimate `I(M;Y)` and the ceiling `min(R, I(M;Y))`; check
   whether the label-free accuracy *ranking* of rubrics matches the `Y`-ranking (does unsupervised pick the
   `Y`-best?).
4. **Steinhardt–Liang risk on a rubric** as the formal "supervised-bound-from-unlabeled" instance.

---

## References (verified 2026-06-19)

- Robertson & Koyejo (2025), TVD-MI (arXiv 2508.05469) — our metric foundation; peer-prediction eval.
- Parisi, Strino, Nadler, Kluger (PNAS 2014, arXiv 1303.3257) — rank-one covariance, Spectral Meta-Learner.
- Jaffe, Nadler, Kluger (AISTATS 2015, arXiv 1407.7644) — accuracies + class balance, tensor.
- Steinhardt & Liang (NIPS 2016, arXiv 1606.05313) — unsupervised **risk** estimation via 3-view moments.
- Platanios et al. (UAI 2014) — estimating accuracy from unlabeled data (agreement equations).
- Dawid & Skene (1979); Zhang, Chen, Zhou, Jordan (2014) — latent-class crowdsourcing, spectral w/ bounds.
- Ratner et al. (NIPS 2016 Data Programming; 2019 Multi-Task Weak Supervision, arXiv 1810.02840); Snorkel.
- Smith et al. (2022, arXiv 2205.02318) — *Language Models in the Loop* (prompts as labeling functions).
- Garg et al. (ICLR 2022, arXiv 2201.04234) ATC; Baek et al. (NeurIPS 2022) Agreement-on-the-Line;
  Jiang et al. (ICLR 2022) disagreement; Disagreement-Discrepancy bounds (NeurIPS 2023, arXiv 2306.00312).
- Unsupervised LLM-judge: *Sage* (OpenReview JFTSZa2stt); *No-Knowledge Alarms* (arXiv 2509.08593);
  *LLM-as-a-Jury* (arXiv 2602.16610).
- Peer prediction: Miller–Resnick–Zeckhauser (2005); multi-task peer prediction (Shnayder et al. 2016,
  arXiv 1603.03151).
- *(unverified)* "Baumann" prompt-evaluation cite — get exact title before use.
