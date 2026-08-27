# Tacit knowledge in LLMs — related-work map (for the talk)

*Deep-research sweep 2026-07-01 (5 angles, 22 sources fetched, 89 claims → 25 adversarially
verified, only 4 survived a 2/3-refute vote). Headline: **no identified work does what we do** —
certified per-metric ceilings on what a checklist/rubric can transmit. The closest neighbors measure
LLM capability gaps, not human-articulability ceilings, and none carry a certificate. The high kill
rate (21/25) is itself a finding: the field's specific quantitative articulability claims are mostly
not robust on inspection.*

## The five nearest neighbors (place, don't bury)

| work | what it does | why it's NOT us |
|---|---|---|
| **Yu et al., "Grading the Unspoken" (arXiv 2604.14188, Apr 2026)** — CLOSEST | Operationalizes tacit knowledge as LLM ability to reconstruct reasoning steps experts omit; 4-level rubric L0 correctness→L3 tacit reconstruction; sharp drop 0.92–1.00 → 0.17–0.50 | Measures **what the LLM can't do**, not **what humans can't tell**; no bound, no coverage estimate, no certificate; 12 questions, theoretical physics. We certify a ceiling; they report a score. |
| **"Limits of Prompt-Conditioned LMs as General-Purpose Learners" (arXiv 2606.23668)** — closest FRAMING | Prompts as a capacity-limited communication channel; irreducible error floors when task complexity exceeds channel capacity | Conceptually adjacent to our executor-indexed channel — but every specific claim was refuted 0-3/1-2 in verification (overstated vs what's proven). A framing to cite and distinguish, not a result to lean on. |
| **Shen et al. (Meta), "Rethinking Rubric Generation" (arXiv 2602.05125, Feb 2026)** | Exponential bound on rubric-judge misclassification `(wᵀμ)²/(wᵀΣw)`; naive rubrics DROP GPT-4o 55.6%→42.9% on JudgeBench (13 pts below no-rubric) | Bounds how to **weight/aggregate given rubric items**, not what can be **articulated** in them. The "naive rubrics hurt" result supports our checklist-vs-holistic gap empirically. |
| **Prompt-optimization PAC-Bayes (arXiv 2510.08413, Oct 2025)** | Perplexity-based generalization bound for prompt SEARCH, scales with √Σlog P(q\|p) not vocab size | Bounds generalization over a prompt-search space; not a ceiling on expressiveness. Authors concede bounds ~0.46 "not necessarily useful." |
| **Outcome-vs-process faithfulness (arXiv 2603.16600 + 2604.22074)** | Outcome-supervised RL judges reach right verdicts via flawed reasoning — trained evaluators ≠ faithful articulated evaluators | Training-method reliability, not an articulability bound. Corroborates that trained ≠ stated, which motivates our C−B (dense − checklist) gap. |

## Adjacent clusters (context slides, not competitors)

- **ELK / latent-knowledge probes** (CCS-style, arXiv 2312.01037): "the model knows more than it
  says," extracted via activation probes. Every specific number refuted 0-3 here. Ours is
  behavioral/information-theoretic, not activation-probing — and about *humans*, with the LM as
  instrument.
- **"Evaluation dialects" (arXiv 2602.08672):** rubrics don't transfer across model architectures
  even when explicit. Refuted 0-3, but the *idea* is our executor-indexing (a rubric is
  executor-relative) — worth one sentence as independent corroboration of the framing.
- **Implicit-memory ceiling ~66% across 17 models (arXiv 2604.08064):** behaviorally-demonstrated
  but not-articulable knowledge. Refuted 1-2 (overstated). Same shape as our decompression gap, no
  bound.
- **Prompt-compression rate–distortion (arXiv 2407.15504):** fundamental limit of prompt compression
  as an LP. Refuted 0-3. Closest to our decompression curves in *spirit* (how much can a short
  message carry) — but it's token-compression of a fixed prompt, not name-only-vs-exemplar
  articulation of a concept.
- **Procedural/video-tacit knowledge** (arXiv 2605.07639, 2606.25984): demonstrations carry
  procedural knowledge text can't (video 0.94 tool-recall vs text 0.57–0.81). The
  demonstration-beats-telling intuition, in a different modality.
- **KM/philosophy** (Polanyi, Collins RTK/STK/CTK; IJIKM, AI&Society pieces): the tacit-knowledge
  taxonomy exists and is occasionally name-checked for AI, but **not operationalized with bounds** —
  the gap we fill between the philosophy and a measurement.

## What remains uniquely ours (the "so what" slide)

The workflow's own open questions read like a description of our method — no identified work:
1. bounds the **Shannon capacity of the instruction/checklist channel** per executor (our
   executor-indexed transmission `T(m_ω)`);
2. uses **capture-recapture / missing-mass (Good–Toulmin) over an LM-proposed criterion pool** to
   estimate coverage and certify an ε-gap ceiling `OPT_Ω + ε`;
3. reads **same-family scaling flatness** of the residual `Δ(E)` as tacitness evidence;
4. compares **decompression curves** (name vs definition vs explanation vs exemplar) with a strong-vs-weak
   reader gap;
5. formalizes **Polanyi with a certificate** rather than an observed capability gap — and targets
   **human** articulability, with the LM as the instrument.

One-liner for the talk: *the field measures how well models do on hard-to-verbalize tasks; we
certify how much of a human preference no checklist can carry, and separate "can't tell" (our
certified residual) from "can't sample yet" (undersampled) from "won't hold still in words"
(form-dominated) — a diagnostic nobody else produces.*

Related: [[project_cw_day0_certificate_read]], theory §12.8; the CW Day-0 read is the first live
instance of that three-way diagnostic.
