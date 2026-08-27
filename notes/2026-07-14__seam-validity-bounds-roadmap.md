# Roadmap: from agreement to validity, and from programs to bounded units

**2026-07-14. For Codex to implement.** Extends `notes/2026-07-13__codability-verifier-roadmap.md`.
The verifier contract, the discrimination gate, and the plant arm all stand. What changes: a
**validity gate** is added (agreement between two implementations is not validity), the Math a12
headline is **retracted as confounded**, and the reconstruction objective gets the **theoretical
bounds** it was missing.

---

## 0. Vocabulary, made concrete

Everything below is grounded in one real item, `train_0001`, a linear-algebra answer whose line 5
contains a chain like *"Let B = SXS⁻¹. Then B² = SXS⁻¹SXS⁻¹ = SX²S⁻¹…"*

| Term | What it actually is |
|---|---|
| **Unit** | One `(lhs, rhs, line)` triple pulled out of the answer by a code extractor. For train_0001: `("B", "SXS^{-1}", answer.md:5)`. That triple, with its span, **is** the unit. There are 443 of them across 150 TRAIN documents. |
| **Shared unit** | Both arms — SymPy and Sonnet — were handed the **same 443 pre-extracted triples**. Neither arm had to *find* them. So the experiment measures what each does *with* a unit, never whether they'd independently *discover* the same units. |
| **Occasion** (= `applies`) | A place in the text where the norm has something to say. Not every `=` is one. `2r+4 = r−7` is an *equation to solve*. `B = SXS⁻¹` is a *definition*. `a²+b² = 1` is a *constraint*. None of those is an occasion for a rigor norm about equality-preservation — only an **asserted identity step in an argument** is. |
| **The two bits** | You are right that it's binary — and there are **two** binary decisions, not one. `applies ∈ {0,1}` (is this an occasion?) and, defined only when `applies=1`, `violated ∈ {0,1}` (does the asserted equality hold?). The reported κ=0.445 is on bit one. The reported κ=1.0 is on bit two. |

---

## 1. RETRACTED: the Math a12 headline. Two independent defects.

### 1a. The LLM arm never sees the document

`methods/metric_seam/compile_math_a12_llm_requests.py:62-67` constructs the LLM request from
`pair_id`, `lhs_display`, `rhs_display`, and spans — **and nothing else**. `ctext` is read only to
extract pairs; it is never sent to the model.

Sonnet's own words on `train_0001.pair-004` are the proof:

> *"The rhs simplifies to `SX^2S^{-1}`… but this is not identically equal to `B^2` **without additional
> context establishing that `B = SXS^{-1}`**."*

The document establishes `B = SXS⁻¹` **on the same line** — it is `pair-003` of the same item. Sonnet
could not see it. It then complied with the contract and returned `applies: true, violated: true`.

Consequences:

- **κ = 1.0 on adjudication is vacuous.** Both arms were reduced to context-free symbolic-identity
  checking, which is a task both perform correctly. They agree because they are running the same
  degenerate test, not because adjudication "transports."
- **κ = 0.445 on applicability is confounded, not a finding.** Sonnet was asked whether a bare
  expression pair is an occasion for the norm, with the document withheld. That question is not
  answerable. The low κ measures the harness, not the seam.
- The standing apples-to-apples rule was met *between the arms* (both got the pair) and violated
  *against the construct* (the construct needs the document). Both arms were information-starved in
  the same way, which is why they agreed.

### 1b. The construct is broken — the "violations" are not violations

Printing the 91 pairs SymPy calls `violated`:

```
B          =?= 0            C          =?= 0           B        =?= SXS^{-1}
f'         =?= 0            f          =?= g + h       a^2+b^2  =?= 1
N_b + N_s  =?= N            \lambda    =?= 0           x        =?= 0
2r+4       =?= r-7
```

Every one of these is a **definition, a hypothesis, a constraint, or an equation being solved**. Not
one is a rigor violation. `f' = 0` is the condition defining a critical point. `2r+4 = r−7` is an
equation whose *whole purpose* is that the two sides are not identically equal. Meanwhile the 24
`satisfied` pairs are arithmetic tautologies: `2^2 = 4`, `4+3 = 7`, `2^5 − 6 = 32 − 6`.

So the verifier's actual behavior is: **"satisfied" iff the pair is a closed-form arithmetic identity;
"violated" otherwise.** `P(violated | applies) = 91/115 = 0.79` is an artifact of that misconstrual,
not a property of the corpus. Codex's own claim limit says it without drawing the conclusion:
*"exact nonidentity is not a document error without separately established claim scope."*

### 1c. The lesson that matters most

**My verifiability certificate would have PASSED this unit.** κ ≥ 0.80 ✓ (κ=1.0). Witness overlap ✓
(identical spans). Ablation flip ✓. Planted-probe capability ✓ (24/24). Every gate green — on a unit
that measures the wrong thing.

> **Agreement is not validity. Two independent implementations that share a misconstrual will certify
> perfectly, and the certificate will not notice.**

That is the hole in the 2026-07-13 roadmap, and §2 closes it.

---

## 2. Three gates, not one

| Gate | Question | Fails when |
|---|---|---|
| **G1 — Agreement** (existing) | Do two independently authored implementations return the same verdict, from the same witness? | κ < 0.80, or witness Jaccard < 0.50 |
| **G2 — Validity** (NEW) | Does the unit measure the *construct*, or a proxy that correlates with it? | The verifier fires on a **negative control** — an item that trips the proxy but does not violate the construct |
| **G3 — Capability** (existing) | Is the instrument operative at all? | Planted true violations are not detected |

### G2 is a kill-switch, and it is cheap

For every unit, author **two** planted sets, not one:

- **P+ (true violations)** — the construct is genuinely violated. A verifier that misses these fails G3.
- **P− (proxy traps)** — the construct is **satisfied**, but the naive proxy fires. *A verifier that
  fires on these fails G2.*

For a12, P− writes itself from the data above: a document containing `Let B = SXS^{-1}.` followed by
`B = SXS^{-1}` is a **perfectly rigorous** text that the current verifier calls a violation. Ship that
as a negative control and the current unit **fails immediately** — which is the point. G2 would have
caught this before 443 Sonnet calls.

**Implementation tip.** `methods/metric_seam/killswitch/plants.py` already registers plants with known
truth types. Add a `polarity` field (`"positive" | "negative_proxy_trap"`) and require **both** kinds
per unit. `battery/contract_check.py` already asserts "no INVERTED probe" and ≥75% separation — extend
its probe list to carry P− and assert **zero fires on P−**. This is a ~30-line change to machinery that
already exists.

---

## 3. The real a12 experiment: the context arm

The interesting question was never "do SymPy and Sonnet agree on symbolic identity." It is:

> **How much of the norm lives in the *discursive role* of an equation rather than in its symbols?**

Deciding that `B = SXS⁻¹` is a *definition* and not a *claimed step* requires reading what the author
is **doing** with the equation. That is not in the symbols. It is exactly Collins's collective tacit
knowledge — knowing what counts as the same kind of move in a practice.

### Design — a 2×2, and the asymmetry is the result

| | pair only | pair + document |
|---|---|---|
| **SymPy** | current arm (443 pairs) | **structurally identical** — SymPy cannot read prose. This cell *is* the top-left cell. |
| **Sonnet** | current arm (confounded) | **the new arm** |

The empty cell is not a gap; it is the finding. **Code cannot use the context even when handed it.**
The prompt can. So:

- **Individuation gap** = Sonnet(pair+doc) applies-rate vs SymPy applies-rate, per role class.
- Ask the LLM arm for the **role**, not just the bit: `role ∈ {asserted_identity_step, definition,
  hypothesis, equation_to_solve, other}`. `applies = (role == asserted_identity_step)`. This makes the
  individuation decision *inspectable* instead of a bare bit.
- **Prediction, prereg before running:** agreement will **fall**, because Sonnet-with-context will
  reclassify the definitional/solving pairs as `not_applicable` while SymPy keeps calling them violations.
  **The size of that fall is the measurement.** It is the fraction of the norm that symbol manipulation
  cannot reach.
- On the residual — pairs both call `asserted_identity_step` — re-measure adjudication κ. *That* number
  is the honest "adjudication transports" claim, and it will be on a much smaller, much more meaningful n.

**Implementation tips.**
- `compile_math_a12_llm_requests.py:50` currently enforces `set(row) == {"ctext", "item_key"}`. `ctext`
  is already in the allowlist — plumbing it through is a **one-field change**, not a new data dependency.
- `methods/metric_seam/verifiers/math_a12_llm_contract.py` — add `document: str` and `role: Literal[...]`
  to `RationalExpressionPair` / the response schema.
- Keep the old arm. Do **not** overwrite `math_a12_llm_train_v3_single_pass/`. The pair-only arm is now
  the *control* for the context arm, and the contrast is the whole experiment.
- Cost: 443 calls × 1 pass. Sonnet. Trivial.

---

## 4. The bounds you asked for — and yes, there is low-hanging fruit

Your instinct is right on both counts: "the program is 20% code" is not a meaningful decomposition,
and a reconstruction objective without a certificate loses the capture/recapture magic. Here is what a
**DAG** buys you that a monolithic prompt never could. All four are CPU-only and cost zero model calls.

### 4a. Exact Shapley over nodes — makes "how much work is the code doing" an *answerable* question

One-at-a-time ablation marginals (what `build_readouts.py` computes today) are **not an attribution**:
they ignore interactions and they do not sum to the total. Shapley values do — that is the efficiency
axiom — so they give an **additive** decomposition of the program's performance across its nodes.

a34 has 11 ablatable nodes ⇒ **2¹¹ = 2048 coalitions**. That is nothing. Enumerate them exhaustively;
no sampling, no approximation.

Then "is the code doing 20% of the work?" becomes exactly computable — and note that on a34 the answer
by node count (11/13 = 85% code) and the answer by Shapley mass (88% in **one** node) are wildly
different. **Node count is meaningless. Report Shapley mass.**

And the axis to report it on is not code-vs-LLM. It is **op class**: `computation` / `evidence` /
`individuation`. That axis maps onto Collins directly, and it is the one that generalizes.

**Implementation tip.** `outputs/.../ws4/patents_pa__a34/build_readouts.py` already has the
ablate-to-own-additive-identity convention and the node registry. Add a coalition loop over
`itertools.chain.from_iterable(combinations(nodes, k) for k in range(len(nodes)+1))` and reuse
`dag_schema.run` with a node mask. Assert the efficiency axiom as a test: `sum(phi) == rho_intact −
rho_all_ablated`, to 1e-9.

### 4b. A DAG cut is an information certificate — the prompt has no analogue

The program is a Markov chain `x → (node outputs) → score`. By the data-processing inequality, for
**any cut** C of the DAG that separates the inputs from the sink:

```
I(score ; M_ω)  ≤  I(C ; M_ω)
```

Every cut is an upper bound on what the program can achieve; the **min-cut is the tightest one**. This
is the "firm objective" the code-side reconstruction was missing.

And here is the asymmetry that makes it worth doing: **the code's intermediate representations are
observable. The prompt's are not.** You cannot cut a prompt. So code admits a certified upper bound
that prompt-based reconstruction structurally cannot — which is not a weakness of the code direction,
it is its *distinctive theoretical advantage*.

**Caveat, state it in every report:** this bounds *this program*, not codability in general. A better
program may exist. **The plants (G3) are what bound the instrument; the cut is what bounds the artifact.**
Two different certificates; never conflate them.

### 4c. Capture–recapture, restored — the unit is the NODE, not the program

You are right that a program is not a unit — a single program is many units, so the estimator dies. But
**the DAG hands you the decomposition for free**: its nodes *are* units, typed and enumerable.

So run K **independent authoring fleets** (different agents, different seeds, blind to each other) on the
same metric text, type every node by `(op_class, witness kind, relation)`, and run capture–recapture over
**node types**. That estimates how many distinct verifiable units the metric admits — the same estimator
as the prompt-optimality work, now with an *observable* unit.

**We do not lose the magic here. We gain it.** In the prompt world the unit had to be inferred; in the
code world it is written down, typed, and executable.

Standing caution from that lane applies unchanged: **mining moves the bound's level, auditing only its
width.** Independent fleets = mining. Do not read a re-audit of one fleet as a level shift.

### 4d. Code has zero executor variance — so all attenuation is prompt-side

A deterministic program has **two-pass reliability = 1.0 by construction**. In any code↔prompt
comparison, every bit of attenuation lives on the prompt side, and it is measurable. Report
ceiling-normalized ρ with the code side's ceiling **pinned at 1.0** and say so. (a34's judge rel₁ =
0.640 — that attenuation belongs to the *judge*, not the program, and must not be charged to the code.)

---

## 5. Beating the strong model with code — the matched-information design

The claim is currently incoherent, and it's fixable. a34 shows *code + retrieval* beating *a model
without corpus access*. That is an **information-access** result, not a capability result. To get the
capability claim you must **match the information** and let only execution differ.

Three arms, same items, same target, prereg the reading:

| Arm | Input | What it isolates |
|---|---|---|
| **A** `prompt(x)` | document only | today's baseline |
| **B** `prompt(x, Z)` | document **+ the code's evidence-op output, verbatim** | the model, fully informed |
| **C** `program(x, Z)` | the code | the program |

- **A vs C** — the confounded comparison. Mixes information access with execution. Stop quoting it alone.
- **B vs C** — **the capability claim.** Identical information; the only difference is that one side
  *executes* and the other *simulates*.

**Pre-declared reading (write this down before the run):**
- **C > B** ⇒ the advantage is **algorithmic**. Name the operation: exactness, enumeration, arithmetic,
  symbolic normalization. This is a real "code beats the strong model" result.
- **B ≈ C** ⇒ it was **information access** all along. Say so plainly; it is still a true and useful result
  (it says the seam is retrieval, not computation).
- **B > C** ⇒ the program is leaving signal on the table; the code is a lossy encoding of what the model
  could do with the same evidence.

### Reclaim the failed ceiling arm — it is a valid *execution* result

The `full_executable_contract` arm was a bad measure of articulation transport, and I retracted it for
that. But it is a **fine** measure of execution: GLM-5.2 could not reproduce a program's item ranking
**with the complete source code disclosed**, because it cannot evaluate `exp(−mean_density × 25.0)` and
enumerate-count over a 200-line diff accurately enough to preserve rank across 125 items.

That is a clean, defensible, publishable finding: **models cannot execute; code can.** It is the
strongest existing evidence for the "beat the strong model" thesis, and it has been sitting inside an
experiment I threw away. Reframe it — do not re-run it.

---

## 6. Why nothing evolved. Five causes, one root.

1. **Measurement is downstream of freeze.** The per-node ablation table is a **WS4 output** — and WS4's
   own contract is *"refactor-only, no new detectors, bit-exact."* The single stage that finally measures
   what each node is worth is **structurally forbidden from acting on it.**
2. **There was never a per-node objective.** E2's gate is whole-program ρ. A node worth Δρ = 0.0001 is
   *invisible* to a whole-program gate: nothing pushes to delete it, and nothing pushes to improve it. It
   simply persists.
3. **Patents could not be diagnosed at all until WS3 existed.** The runbook says WS3 *"unblocks patents;
   prerequisite for WS4"* — the evidence-aware judge M̄(x,Z) did not exist when `a34_h0.py` was written.
   By the time it did, the launch order had moved on to WS4, which is transcription.
4. **The launch order was breadth-first.** Nine WS4 cells across three tasks, rather than one task driven
   to convergence. **Coverage as the organizing principle** — precisely the disease that produced the
   50 → 27 → 18 funnel and its constant targets.
5. **No base-rate check at authoring time.** Nothing ever required the authoring agent to ask *"how often
   does this feature even occur in the corpus?"* before writing a detector for it. `a34_h0.py`'s docstring
   asserts the LLM layer earns "a (small) weight." The corpus population is **2 in 250**. Nobody looked.

> **Root cause: the pipeline measures after it freezes, and it selects on coverage rather than on
> information.**

Every "disease" in this stream — a34's dead LLM subtree, the code-review constants, the a12 misconstrual —
is a specialization of that one sentence.

---

## 7. The pipeline inversion

Current order: `PROPOSE → AUTHOR → (maybe repair) → FREEZE → TRANSCRIBE → MEASURE`.
The measurement is last, so it can only be *reported*, never *acted on*.

**New order — measure first, freeze last:**

```
PROPOSE  →  BASE-RATE PROBE  →  AUTHOR  →  PER-NODE GATE  →  SELECT  →  FREEZE  →  TRANSCRIBE
             (CPU, corpus-only,           (Shapley + applies-rate,
              no detector written yet)     BEFORE the equivalence freeze)
```

- **Base-rate probe.** A `probe_base_rate(feature_spec, corpus) -> {occurrence_rate, n}` step that runs
  **before a detector is authored**. Kill any proposed feature whose occurrence is < 0.10 or > 0.90. This
  one step kills a34's two LLM nodes (2/250) and all four code-review units, at zero cost, before any code
  is written.
- **Per-node gate.** Every node must carry its own Shapley φ and its own applies-rate, computed on TRAIN,
  **before** the equivalence freeze. A node with |φ| < ε **and** applies-rate < 0.10 is **deleted**, not
  reported. (This is the gate that does not currently exist at any stage.)
- **Depth-first.** Drive **one** task to convergence before opening cells in another.

---

## 8. Order of work

| # | Step | Cost | Notes |
|---|---|---|---|
| 1 | **Retract** the a12 headline (§1) in the runbook + notebook. State both defects. | free | The κ=1.0 must not be quoted as an adjudication result. |
| 2 | Add **G2 (validity) negative controls** to `plants.py` + `contract_check.py`. Ship a12's definitional P− set. | CPU | ~30 lines. Confirm the current a12 unit **fails**. |
| 3 | **Exact Shapley** on a34 (2048 coalitions) + the op-class decomposition. Assert the efficiency axiom in a test. | CPU, free | Answers "how much work is the code doing", exactly. |
| 4 | **DAG cut certificate** — enumerate cuts of a34, report min-cut bound. | CPU, free | The certificate the code side was missing. |
| 5 | **a12 context arm** — plumb `ctext` through, add the `role` field, 443 Sonnet calls. | ~443 calls | The real experiment. Prereg the reading first. |
| 6 | **Matched-information arms B vs C** on a34. Prereg the reading. | moderate | The "beat the strong model" claim. |
| 7 | Reframe the ceiling arm as an **execution** result. No re-run. | free | Reclaims a discarded experiment. |
| 8 | **Pipeline inversion** (§7): base-rate probe + per-node gate. | CPU | The structural fix. |
| 9 | Node-level **capture–recapture** across K independent authoring fleets. | K fleets | Restores the estimator. Needs sign-off on K. |

Steps 1–4 and 7–8 are CPU-only and cost **zero model calls**. Do them first.

---

## 9. Ban list — additions

Each one is now backed by a real failure in this stream.

1. **Never hand a verifier a unit stripped of its context and report the agreement as a finding.** Two
   information-starved arms agree because they are starved, not because the seam is closed.
2. **Agreement is not validity.** Two independent implementations sharing a misconstrual certify perfectly.
   Every unit needs a negative control that trips the proxy but satisfies the construct.
3. **One-at-a-time ablation marginals are not an attribution.** They do not sum and they ignore
   interactions. Use Shapley; it is 2ⁿ and n is small.
4. **Never report κ without stating what information each arm had.** The a12 κ's are uninterpretable
   without "the LLM never saw the document," and that sentence appears nowhere in the readout.
5. **Never count nodes as a measure of contribution.** a34 is 85% code by node and 88% one-node by mass.
6. **Never let the measuring stage be downstream of the freezing stage.**

---

## 10. Implementation status — completed 2026-07-14

All nine steps above are implemented and executed. K was frozen at 3 before fleet outputs (the
smallest multi-list Chao2 design); this is a bounded small-K pilot, not a permanent default.

| step | result | canonical artifact |
|---|---|---|
| 1 | a12 κ headline retracted in runbook and executable notebook; both defects stated | `notes/2026-07-10__seam-agentic-program-runbook.md`; notebook §24 |
| 2 | G2 FAIL: 4/4 proxy traps fire; 2/2 true errors detected | `outputs/metric_seam_pilot/hierarchy_r123/results/math_a12_g2_validity_v1/` |
| 3 | 2,048 exact coalitions; efficiency residual −3.33e−16; evidence abs φ-mass .801 vs computation .118 | `outputs/metric_seam_pilot/battery/effort_ladder/ws4/patents_pa__a34/readouts.json` |
| 4 | 648 cuts / 8 minimal; minimum empirical cut bound 2.338 bits; DPI verified on all minimal cuts | same a34 readout |
| 5 | 430/443 valid, no retries; 57.7% symbolic-applicable pairs reclassified by role; residual polarity κ=.082 | `outputs/metric_seam_pilot/hierarchy_r123/results/math_a12_context_train_v1/` |
| 6 | matched x+Z: B ρ=.661, C ρ=.740; C−B=.079, 95% CI [.013,.151] | `outputs/metric_seam_pilot/hierarchy_r123/results/patents_a34_matched_info_v1/` |
| 7 | full-source arm reframed, no rerun: median ρ=.149, CI [−.076,.478], no positive gate | `outputs/metric_seam_pilot/hierarchy_r123/results/code_review_glm52_ceiling_v1/execution_reframe.md` |
| 8 | prospective API integrated; retrospective replay kills 6/6 known dead units before authorship | `outputs/metric_seam_pilot/hierarchy_r123/results/pipeline_inversion_replay_v1/` |
| 9 | K=3 fleets: 13 observed node types, Chao2 total 13.8, estimated coverage 94.2% | `outputs/metric_seam_pilot/hierarchy_r123/results/patents_a34_capture_recapture_k3_v1/` |

Consolidated machine-readable and prose reports are in
`outputs/metric_seam_pilot/validity_bounds_v1/`. The notebook has been executed end-to-end and saved.
