# Lit review: law × program synthesis, and programs-with-LLM-calls (2026-08-14)

Deep-research run (5 angles, 22 sources fetched, 109 claims extracted, top 25 adversarially
verified 3-vote: 25 confirmed / 0 refuted; one 2-1). Purpose: positioning for the metric-seam
paper (paper #5) — does anyone study WHERE the code/LLM seam sits per criterion, gate
prompt→code migration, or certify faithfulness?

**Headline: across all surveyed lineages the code/LLM boundary is FIXED BY DESIGN — no prior
work does per-criterion codability measurement, gated migration, or faithfulness certificates.**
(Absence claim; see coverage gap at bottom.)

## Generation 1 — classic computational law (seam fixed, faithfulness procedural)

- **British Nationality Act in Prolog (Sergot/Kowalski et al., CACM 1986).** Statute compiled
  to executable Prolog; open-textured concepts ("good character", "reasonable excuse")
  deliberately NOT compiled — assumed true, emitting *qualified answers* with judgment left
  outside the program. Faithfulness = structural isomorphism to statute wording ("increases
  confidence"), no certification. The original code-vs-judgment seam, drawn a priori.
  [3-0 ×3] https://www.doc.ic.ac.uk/~rak/papers/British%20Nationality%20Act.pdf
- **Catala + Lawsky (ICFP 2021; Lawsky SMU L.Rev. 2022).** Key structural insight: statutory
  rules have canonical WHEN/THEN/UNLESS shape → prioritized default logic is the right
  substrate; this rule+exception structure is WHY the elements layer is codable (independent
  corroboration of our legal-pole finding). Fidelity = lawyer+programmer pair programming +
  literate programming — human-process assurance. NB: Catala's F* verification certifies
  code→code compilation, NOT statute fidelity. [3-0 ×3] SSRN 4291177; arXiv:2103.03198.
- **PROLEG (Satoh et al., JURISIN 2010).** Japanese civil-code doctrine (presupposed-ultimate-
  facts theory) as pure Prolog; what compiles is exactly the elements + burden-of-proof layer;
  fact-establishment stays outside via allege/provide_evidence inputs. Interesting precedent:
  the code representation was itself redesigned on expert-legibility grounds (lawyers couldn't
  read the negation-as-failure version) — an early gate on the formalization, but on
  legibility, not fidelity. [3-0 ×3]

## Generation 2 — LLM-era statute→code synthesis (execution-tested, never certified)

- **LLM→Catala / Dagger (ProLaLa @ SPLASH 2024).** LLMs generate executable Catala for tax
  questions; fail where tax text is "highly contextual". Their response: redesign the target
  DSL (Dagger, implicit context) — i.e., attack the boundary problem by changing the code
  substrate, not by studying seam placement. [3-0 ×3]
- **LLM+Prolog on SARA (Jurayj, Holzenberger, Van Durme 2025, arXiv:2508.21051).** Closest
  architecture to ours. Explicit rationale: offload compositional reasoning to SWI-Prolog
  because LLMs can't do it faithfully. Best-trust config = seam at FACT-PARSING ONLY (GPT-5
  parses facts against gold human-coded statutes): 84/100 with 10 abstentions (GPT-4.1
  few-shot hits 87 raw in same seam class). Boundary fixed across 3 predefined architectures,
  not per-rule. Trust = execution-failure abstention + 2-sample self-consistency; authors flag
  manual statute translation as "a constraining assumption". [3-0 ×4]
- **SOLAR (arXiv:2509.00710).** Multi-agent neurosymbolic: LLM agents extract concepts +
  formalize statutes into DL ontology (TBox) + FOL rules → compiled to Python/SMT inference.
  Faithfulness via iterative judge-agent loop classifying failures as representation-vs-
  implementation defects (a verification LOOP, no certificates). SARA-numeric: 18.8%→76.4%
  zero-shot foundational; o1-mini 87.5% — but o1-mini's own chain-of-code baseline (93.8%)
  BEATS its SOLAR score; self-reported preprint; accuracy claim was the sole 2-1 vote.
  Defeasible/gestalt reasoning deliberately left unformalized. [3-0 ×3, 2-1 ×1]
- **Legal text→DMN decision models (Graus, ICAIL 2026, arXiv:2604.17153).** GPT-5.1 generates
  executable DMN from Dutch Environment & Planning Act vs 95 production models. Closest thing
  to execution-based fidelity verification: 13,080 gold test inputs; I/O specs dominate
  structural-fidelity gains (+37%/+54% graph-kernel sim, p<.001); functional faithfulness
  partial (51-53% macro outcome agreement; 33% fully outcome-equivalent). ★ Structural
  similarity and outcome equivalence are complementary, NON-redundant checks (high structural
  sim ≠ outcome equivalence) — useful precedent for why our gates need both κ-type and
  ΔR-type conditions. Human review still mandated; no deployment certificates. [3-0 ×3]

## Generation 3 — hybrid neurosymbolic reasoning (element decomposition, LLM-side)

- **Chain of Logic (Servantez et al., ACL Findings 2024).** Decomposes compositional legal
  rules into per-element independent reasoning threads, recomposes via the rule's logical
  expression — direct analog of per-criterion channels, confirming the elements layer has
  explicit decomposable logical structure. But seam entirely LLM-side: every element AND the
  boolean recomposition resolved by in-context prompting; no code, no study of which elements
  could be code. Evaluated on LegalBench rule-application tasks. [3-0 ×2]
- Also surfaced (fetched, claims extracted, not in verified top-25): ContractNLI FOL hybrid
  ("Know Your Limits" arXiv:2606.16118 — accuracy gain from formal structure ≠ faithful formal
  reasoning); IRC §121 Prolog inconsistency detection (arXiv:2511.11954); LLM-predicate +
  Logic-Tensor-Network procurement pipeline (arXiv:2604.05539 — LLM scores 8 predefined
  predicates, hand-coded fuzzy rules aggregate).

## Angle 4 — programs-with-LLM-calls paradigm (⚠ COVERAGE GAP: extracted but NOT verified)

The verifier budget dropped this angle's claims, so these are *unverified* extractions —
but they are the nearest general-paradigm competitors and worth a follow-up pass:

- **LOTUS / semantic operators (arXiv:2407.11418).** Declarative typed operators (sem_filter/
  map/join/topk) — code provides pipeline structure, LLM implements operator semantics; boundary
  fixed by design; optimization via proxy/gold cascades with statistical accuracy guarantees
  (precision/recall targets on operator substitution — the closest thing to a "certificate" in
  this literature, but it certifies a cheap-model substitution against a gold LLM, not
  prompt→code migration).
- **★ SemBaker (arXiv:2608.06677).** One-time LLM call synthesizes a deterministic Python
  function that REPLACES per-row LLM interpretation of semantic operators — this IS
  prompt→code migration, apparently accuracy-checked. Most important un-verified neighbor;
  read before writing the paper's related-work section. Question to answer: is the migration
  gated per-operator, and against what fidelity criterion?
- **Abacus (PVLDB 2026, arXiv:2505.14661).** Cost-based optimizer choosing physical
  implementations per semantic operator under quality/cost/latency constraints — treats
  where/how each LLM sub-step is implemented as an optimization variable (cost-driven, not
  codability-driven; no faithfulness-to-construct notion).
- **DocETL (arXiv:2410.12189).** Operators explicitly split LLM-powered (map/reduce/resolve/
  filter) vs deterministic (split/gather/unnest) — fixed-by-design boundary at operator level.
- **Arize LLM-as-judge guide (blog).** Industry heuristic: deterministic checks → code evals,
  subjective quality → LLM judge. Boundary by rule-of-thumb, never measured.

## Positioning takeaways for paper #5

1. The recurring empirical pattern across 40 years — elements/burden layers compile,
   open-textured/contextual/gestalt content stays outside code — independently corroborates
   our legal-elements-most-codable finding (BNA 1986 → PROLEG → Catala → Dagger → SARA-Prolog).
2. Our three claimed novelties survive the survey: (a) per-criterion codability as a MEASURED
   variable, (b) gated prompt→code migration, (c) faithfulness certificates on the seam.
   Every surveyed system fixes the boundary a priori (logic choice, DSL redesign, architecture
   variant, pipeline stage, or all-LLM).
3. Fidelity-mechanism taxonomy to cite in related work: structural isomorphism (BNA),
   human-process assurance (Catala), expert legibility (PROLEG), execution-failure abstention +
   self-consistency (SARA-Prolog), iterative judge-agent loop (SOLAR), gold test-suite
   execution (DMN) — none is a certificate gating deployment/migration.
4. SARA-Prolog's result is a nice frame: the trustworthy seam today sits at fact-parsing with
   human-certified statute code — our gated-migration machinery is exactly what would let the
   seam move statute-side safely.
5. ⚠ Before claiming novelty in print: verify SemBaker + LOTUS-cascade guarantees (angle-4
   gap above); also note SOLAR/DMN are 2025-26 preprints and may move.

## Addendum 2026-08-14 (full per-source extractions from journal — includes papers not in verified top-25)

Papers fetched + claim-extracted but not in the verified findings (all UNVERIFIED extractions):

- **"Know Your Limits" (arXiv:2606.16118) — LLM faithfulness as solvers/autoformalizers, ContractNLI NDAs.**
  LLM-interpreting-FOL wins on accuracy (83.0% Claude) but exhibits "SCOPE LAUNDERING" in
  15.3–52.5% of cases — reports solver-consistent answers without executing the formal reasoning.
  Z3 codegen error rates 25.5–63.2%; solver-side is logically valid but conservative because legal
  text is underspecified. Verification = 3-iteration auto-fix loop + manual correction, no certs.
  ★ Best citation for WHY prompt→code migration needs a faithfulness gate (LLM claims of
  code-following can't be trusted without execution).
- **IRC §121 Prolog inconsistency detection (arXiv:2511.11954).** GPT-4o/GPT-5 formalize §121 to
  Prolog; Prolog deterministically detects a known statutory inconsistency GPT-4o alone finds only
  1/3 prompting strategies. Prolog-augmented PROMPTING degraded rule coverage 100%→66% (formal
  structure in-prompt hurts completeness). Fidelity via validation runners + replication against
  Lawsky's independent Z3 implementation; GPT-5-as-collaborator found 4 defects in the GPT-4o code.
- **Insurance contracts → Prolog (arXiv:2502.17638).** Unguided LLM→Prolog = low-fidelity/often
  non-executing; expert-provided schemas + helper-rule docs → 1.00 accuracy on simplified Chubb
  policy (3/4 LLMs), o1 95% on Stanford Cardinal Care ART (2,020 test cases) vs 37-72% others —
  hybrid fidelity is capability-graded by the LLM (echoes our function-wall capability-grading).
  Manual eval + execution tests only.
- **Amortized Intelligence / DACL contract adjudication (arXiv:2605.02472).** Claude 4.5 compiles
  contracts to deterministic DACL once; gpt-5-mini routes at runtime; symbolic engine executes.
  99.5% on 400 adjudication events vs 71.5-82.8% end-to-end LLMs. ★ Error taxonomy: 71% of pure-LLM
  errors are variable-dependency tracking, <1 arithmetic — code-side payoff is STATEFUL DEPENDENCY
  TRACKING, not math. Both residual DACL errors were LLM-side. Layered verification (type checks,
  synthetic edge cases, human review); concedes errors that pass review become deterministic+repeat.
- **LLM predicates + Logic Tensor Network, German procurement (arXiv:2604.05539).** LLM scores 8
  predefined predicates; hand-coded fuzzy-logic LTN aggregates. ★ ONE OF THE ONLY WORKS THAT
  EMPIRICALLY STUDIES SEAM PLACEMENT: RQ1 quantifies replacing learned predicates with
  LLM-derived ones + variants moving the seam (pure LTN, pure LLM, IE+rules, BERT). Moving the
  seam toward LLM extraction COSTS accuracy (classic LTN F1 .899 > best hybrid .874); hybrid
  advantage claimed = interpretability/auditability (regulated-domain traceability), n=200 docs.
- **LegalBench (arXiv:2308.11462).** 162 expert-hand-crafted tasks / 6 legal reasoning types —
  practitioner-doctrine decomposition our channels can map onto; evaluates pure prompting, no
  code/LLM boundary study.
- **LOTUS semantic operators (arXiv:2407.11418) — sharper than the first-pass gloss:** each
  operator has a GOLD ALGORITHM (oracle LLM) and cheaper implementations (proxy models,
  embeddings) substitute ONLY under a statistical accuracy guarantee: user-set target γ met with
  prob 1−δ vs gold. ★ That is a PROBABILISTIC FIDELITY CERTIFICATE gating migration — but for
  LLM→cheaper-surrogate substitution, not prompt→code, and vs an oracle-LLM reference, not a
  construct. Up to 1000x speedups; FEVER program 0.998 accuracy retention at (γ=.9, δ=.2).
- **SemBaker (arXiv:2608.06677) — sharper:** boundary is ADAPTIVE per operator, not fixed — a
  cost-based optimizer decides per-operator between native LLM execution and compiled-Python
  execution by workload cardinality/threshold model. 4.8-6.3x speedups, 5.4-10.7x cost cuts.
  Faithfulness check is weak: ≤3 candidates validated on ≤12 LLM-pseudo-labeled rows, helpers
  unstable in ablations. So: adaptive seam ✓, but placement decided by COST, and migration gated
  by a near-vacuous fidelity check.
- **Abacus / Palimpzest (PVLDB 2026, arXiv:2505.14661).** Per-operator physical-implementation
  search under quality/cost/latency constraints — but the search space is LLM-strategy variants
  (model choice, ensembles, MoA), NOT a code-vs-LLM boundary; quality estimated via 5-10 labeled
  samples + bandits, no certificates. Evaluated on legal CUAD contracts among others.
- **DocETL (arXiv:2410.12189).** Operators split LLM-semantic vs deterministic-code by design,
  BUT rewrite directives decompose one LLM op into finer LLM+code op sequences with agents
  instantiating rewrites — seam placement is SEARCHED by an optimizer (accuracy-driven,
  LLM-judged validation agents, no certificates). 25-80% gains incl. legal clause extraction.
- **Arize LLM-as-judge guide (industry).** Boundary heuristic: deterministic checks→code,
  subjective→LLM judge; 75-90% human-agreement validation as informal gate. = the industry
  rule-of-thumb our per-criterion measurement replaces.

**Revised positioning nuance (vs the headline "nobody studies the seam"):** three near-neighbors
DO make the boundary a variable — SemBaker (cost-driven adaptive placement, weak fidelity check),
DocETL (optimizer-searched rewrites, LLM-judged), LTN-procurement (empirical seam ablation, finds
code-side wins). And LOTUS has genuine probabilistic fidelity certificates, but for model-
substitution not codification. None combines (a) construct-fidelity objective, (b) per-criterion
codability as the measured quantity, (c) certificate-gated prompt→code migration. State the claim
that precisely in the paper — the coarse version is falsifiable by these four.

Open questions logged by the run: (i) does DSPy/LOTUS/DocETL/palimpzest literature contain any
empirical code-vs-LLM sub-step study? (ii) can DMN-style gold-suite execution testing be turned
into a formal migration certificate (what coverage suffices)? (iii) is Dagger's contextual-
failure a hard codability ceiling or a Catala-design artifact? (iv) does the optimal seam move
statute-side as codegen improves, or is certified statute encoding persistent?

Provenance: workflow wf_8a9c4117-d1a (104 agents); full JSON at task output w20wzwveb;
per-agent journal at subagents/workflows/wf_8a9c4117-d1a/journal.jsonl (session dir).
