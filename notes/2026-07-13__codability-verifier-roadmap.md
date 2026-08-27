# Codability roadmap: from *scorers* to *verifiers*

**2026-07-13. Written for Codex to implement.** Supersedes the operationalization instructions in
`notes/2026-07-13__seam-ceiling-arm-handoff-to-codex.md` §"Continuation priority order". The ceiling-arm
handoff's *corrections to the record* still stand; its *forward plan* does not.

---

## 0. What this replaces, and why

The code-review seam lane ran 4,500 GLM-5.2 calls, then a 4,500-call full-source ceiling arm, and reported
median Spearman ρ = 0.149, which fired a preregistered "instrument-limit" branch. **That branch is retracted.**
Three defects, in increasing order of seriousness:

| # | Defect | Evidence | Status |
|---|---|---|---|
| 1 | Harness: `json.loads(raw)` with no fence unwrap and `strict=True` | 4,498/4,500 responses were Markdown-fenced; strict mode also rejected literal tabs in evidence spans, dropping rows *non-randomly* (selects against tab-indented languages) | **Fixed** (`run_hierarchy_prompt_jobs.deserialize_response`); 4,442/4,500 recovered on CPU |
| 2 | Ceiling arm asks GLM to *be a Python interpreter* | The programs compute `math.exp(-mean_density*25.0)`, `1.0 - l1/2.0`; the arm demands exact enumerate-count-arithmetic over 200-line diffs, precise enough that *rank across 125 items* survives | **My design error.** The ceiling ρ measures arithmetic execution, not articulation transport. Do not read it as an upper anchor. |
| 3 | **The targets are constants.** | 10 of 18 cells have code targets whose top tercile *equals* their bottom tercile (a131 mode 0.97, a401 0.95, a1 0.94). `a1_simplicity_yagni` scores exactly 1.0 on 102/125 held-out items. Spearman on a 94%-tied vector is a tie-density statistic, not a reconstruction statistic. | **The real finding.** |

Defect 3 is the one that matters. The coded "metric parts" are **violation detectors** — absence of `foo`/`bar`/
`data` names, absence of TODOs, absence of YAGNI flags — and they were run on a corpus of **merged, reviewed
PRs**, where review has already removed the violations. A violation detector on post-review code is a constant
function. *You cannot measure the codability of a constant.* The only program with real held-out variance
(`a8_small_focused_changes`, mode fraction 0.02, 116 distinct values) measures diff **size** — and it was not
among the 10 mapped programs.

Arithmetic-free tercile AUC on the 8 cells that *do* have spread reproduces the three named values:
**a37 = 0.711, a0 = 0.720, a92 = 0.710**. A fresh same-estimator replay corrects the aggregate,
however: the median is **0.573**, not 0.547, with descriptive item-bootstrap CI **[0.502, 0.678]**;
2/8 cells fall below 0.5. The signal is heterogeneous rather than inversion-free. Artifact:
`outputs/metric_seam_pilot/hierarchy_r123/results/code_review_target_resolution_v1/readout.json`.

The 50/90 → 27/90 → 18/90 funnel was gated on **coverage** at every step and on **discrimination** at none.
That is the process bug this roadmap fixes.

---

## 1. The reframe

The codability work exists to identify **verifiable (RLVR-sense) components of complex metrics** — not to
guess at feature extraction and then regress. Two consequences, both non-negotiable:

> **A component is *verifiable* iff two independent implementations, given the same item, return the same
> verdict *for the same reason*.**

Agreement alone is not verification (two implementations can agree by coincidence, or by both keying on
length). "For the same reason" is what makes it RLVR-shaped: the component must emit a **witness** — the
concrete thing in the item that determined the verdict — and the witnesses must coincide.

> **A component that does not certify is a RESULT, not an embarrassment to optimize away.** Record the
> bounded non-verification—corpus, verifier class, and budget—and move on. It is not evidence of
> unqualified tacitness.

This inverts the incentive that produced the current mess. Under the old framing, a component that resisted
coding was an embarrassment to be papered over with a permissive gate (`min_unique_scores: 2`). Under this
framing, it is the measurement.

---

## 2. What already exists and must be reused

Do **not** build new machinery where the repo has it. Three assets, none of which the hierarchy lane used:

| Asset | Path | What it gives you |
|---|---|---|
| **Certified Unit Framework** | **sk3: `/lfs/skampere3/0/alexspan/outputs/unit_cert/bank/<task>/<executor>/bank_units.jsonl`** — 9 tasks × {llama3b, llama8b, llama70b}. Coverage summary also local at `notebooks/data/polarity_20260706/cuf_coverage_summary.json`. | The decomposition layer. **Stage 0 is a join, not a new decomposition** — see the path note below. |
| **Contract check** | `methods/metric_seam/battery/contract_check.py` | A *pre-registered* candidate gate that already does: planted pos/neg probe separation (≥75%), no-inversion, `min_std ≥ 0.05`, `max_frac_at_mode ≤ 0.85`, TRAIN-only execution, ≥90% completeness, and a sentinel guard against probe-mode gaming. **This is the gate the hierarchy lane needed and did not run.** |
| **Planted kill-switch** | `methods/metric_seam/killswitch/{plants.py,DESIGN.md}` | Seven criteria of *known* placement (code / code+op / mixed / a-layer / null / known-noise), laundered as ordinary aspect ids, with pass bars written before any verdict. Proves the pipeline can tell codable from tacit. |

The weak gate that let the constants through is `methods/metric_seam/hierarchy_train_gate.py:27`
(`min_coverage: 0.05, min_unique_scores: 2`). **It must be replaced, not tuned.**

### CUF path note (added 2026-07-13 — this cost Codex a blocking question)

The unit bank is **not** under `norm-research/outputs/`. It is at
**`/lfs/skampere3/0/alexspan/outputs/unit_cert/`** on sk3 — a sibling of the repo, not inside it. A repo-side
search finds nothing and the inventory looks absent. It is not absent. Verified 2026-07-13:

- `bank/code-review/llama8b/bank_units.jsonl` — **133 metrics, 378 units, 342 `CERTIFIED-UNIT`.**
- Each row: `{metric, k, rows: [{node_id, level, span, delta_free, p_free, delta_M, p_M, sign_stability,
  kappa, eps_ctx, verdict, atom, detect_free, detect_M, certified_lo}]}`.
- A unit's `span` is a **sub-relation of the metric's own definition text** — exactly the component granularity
  the verifier contract wants.

**The join is direct: this is the same metric bank the seam lane is coding.** CUF
"Simplicity (KISS/YAGNI) and complexity management" *is* a1; "Small, focused, reviewable changes" *is* a8.

Two cautions on the join:

- **Join by metric name, adjudicated by an LLM (Sonnet+), not by fuzzy string match.** Names drift across
  banks ("Avoid duplication (DRY)" vs "Avoid duplication (DRY) with pragmatism"). Name matching is a semantic
  comparison, and semantic comparisons are the judge's job, not a string metric's (standing rule).
- **Units are E-indexed** — certified against a named executor, per the CUF spec ("no executor-free unit").
  Use the **llama8b** bank as the reference and say so; do not silently pool 3b/8b/70b units.

---

## 3. The verifier contract (Stage 1)

Every component is a **verifier**, not a scorer. Signature:

```python
def verify(item: Item) -> Verdict: ...

@dataclass(frozen=True)
class Verdict:
    applies: bool                    # is there an occasion to judge at all?
    verdict: Literal["satisfied", "violated"] | None   # None iff applies is False
    witness: tuple[Span, ...]        # the spans/AST nodes that DETERMINED the verdict
    # NO score. NO float. NO confidence.
```

`Span` is `(path, line_start, line_end)` or an AST node id — an **identity**, not a similarity. This matters
in §5: witness overlap must be computable without a semantic proxy measure.

Rules:

- **No floats anywhere in a verifier's output.** The moment a component returns 0.75, someone has chosen a
  weight, and the codability question has been replaced with a regression question.
- **`applies` is first-class.** A verifier that cannot find an occasion must say so. The current lane's
  48.1%-unscored rate was reported as a diagnostic afterthought; it is a primary quantity.
- **The witness is mandatory and must be *load-bearing*.** If deleting the witness spans from the item does
  not flip the verdict, the witness is decorative and the verifier fails Stage 5's witness-ablation check.
- **Verifiers are pure and TRAIN-only during development.** Held-out is touched exactly once, at certification.

### Aggregation

Unit verdicts → metric score is **either learned on TRAIN and disclosed as a frozen fit, or absent.**
Hand-set weights are banned (see §9). The default is *absent*: report unit-level verdicts and stop.

When aggregation is fit, **the gap between the best learned aggregation over verified units and the
prompt-metric's own item-level score is the tacit residue.** That gap is the deliverable. Do not author it
away with a magic constant; measure it.

---

## 4. Discrimination is a GATE, not a diagnostic (Stage 2)

Run **before** any model call. A component that cannot vary cannot be reconstructed, and spending 4,500 GLM
calls to discover that is what we just did.

Pre-registered thresholds, on **TRAIN** items:

| Gate | Threshold | Rationale |
|---|---|---|
| Applies-rate | `0.20 ≤ P(applies) ≤ 0.95` | Below 0.20 there is no occasion to study; above 0.95 the `applies` channel is itself constant |
| Violation base-rate | `0.10 ≤ P(violated \| applies) ≤ 0.90` | The binding fix. `a1_simplicity_yagni` has P(violated) ≈ 0.02 on merged PRs and dies here, **before** the run |
| Rank-readout headroom | at least 3 distinct verdict-pattern classes across items, and `max_frac_at_mode ≤ 0.85` | Reuse `contract_check.py`'s existing constants; do not invent new ones |
| Probe separation | ≥75% strict separation on planted pos/neg probes, **zero inversions** | Already implemented in `contract_check.py`. Wire the hierarchy lane into it. |

**A component that fails the base-rate gate on this corpus is not "uncodable" — it is *unmeasurable on this
corpus*.** Those are different claims and the report must not conflate them. Route it to §6.

---

## 5. The verifiability certificate (Stage 3) — the new gate

This is the RLVR operationalization and the core of the roadmap.

For each unit, produce **two independent implementations**:

- **V_ast** — deterministic program over the diff's AST / structured form. No regex-as-substance (§9).
- **V_llm** — a schema-constrained LLM extractor (Sonnet-or-better, per standing rule; GLM acceptable) that
  returns the *same* `Verdict` type: `applies`, `verdict`, `witness` spans. Not a score. Not a rationale-only
  free-text blob.

They must be authored **independently** — different agents, neither shown the other's implementation. An
agreement measured between two implementations one author wrote back-to-back measures the author, not the unit.

**Certificate = both conditions, on HELD-OUT items:**

1. **Verdict agreement:** chance-corrected **κ ≥ 0.80** between V_ast and V_llm on items where both say
   `applies`. (Chance-corrected, per the standing rule — raw agreement on a 90%-satisfied unit is 0.9 for
   free.) Report κ *and* the applies-rate agreement separately; a unit can be verifiable in verdict and
   incoherent in applicability.
2. **Witness coincidence:** on agreeing items, Jaccard over witness line-sets **≥ 0.50**, or — when spans are
   semantic rather than positional — an LLM adjudicator (Sonnet+) rules *same-referent*. **Agreement with
   disjoint witnesses is coincidence, not verification, and must be reported as a certificate FAILURE even
   though the verdicts match.** This is the check that catches two implementations both secretly keying on
   diff length.

### What a failed certificate does and does NOT establish

**Corrected 2026-07-13 after Codex pushback — the earlier wording here overreached and is withdrawn.**

A failed certificate establishes a **bounded failure**: *these two implementations, from this verifier class,
at this budget, on this frozen corpus, did not converge.* The unit is **uncertified**. It does **not** follow
that no implementation could converge, and the results table must not say "tacit."

This is precisely the unfalsifiability that `killswitch/DESIGN.md` already names: *"a105 is A-layer" could mean
"our codegen is weak."* Do not reintroduce it.

**The only thing that licenses a stronger reading is a co-run positive control.** If the *same* verifier class,
at the *same* budget, certifies the planted known-codable units (p901 `code`, p902 `code+computation op`) and
correctly declines to certify p905 (`a_layer`) and p906 (`null`), then the instrument has demonstrated
capability, and a failure on a real unit carries information. Even then the claim is bounded:

> *tacit relative to a verifier class demonstrated adequate on known-codable plants* — never unqualified tacitness.

The plant arm must be **co-run with** the real units, never retrofitted afterward. A capability control run
later, at a different budget, against a different verifier class, controls nothing.

Report every uncertified unit with its κ, its witness Jaccard, and the plant-arm verdict alongside it.

Third condition, cheap and mandatory:

3. **Witness ablation:** delete the witness spans from the item and re-run. Verdict must flip (or `applies`
   must go False) on ≥90% of `violated` items. A verifier whose verdict survives deletion of its own stated
   evidence is keying on something it did not disclose.

---

## 6. Corpus adequacy (Stage 4) — the phenomenon must be present

**Feasibility fact, already checked:** `outputs/metric_seam_pilot/hierarchy_r123/items_v2/code-review/sealed_heldout.json`
items carry **only `ctext` and `item_key`** — no PR metadata, no pre-review snapshot, no commit history. So
"score the pre-review commit instead of the merged state" **cannot be done by re-slicing the existing corpus.**
It requires re-mining GitHub.

Therefore, two legs, in this order:

### Leg A (do now) — planted mutation / violation injection

Reuse `killswitch/plants.py` machinery. For each unit, generate mutated items that inject the anti-pattern the
unit detects (rename a variable to `data`, add a TODO, introduce the YAGNI abstraction). This does three jobs
at once:

- **A verifier that does not fire on a planted violation is broken.** Unit test, for free.
- **A verifier that fires *only* on planted violations proves the corpus contains none** — which converts
  defect 3 from an interpretation into a *measurement*. Report `P(violated | natural)` vs `P(violated | planted)`.
- It supplies the pos/neg probes `contract_check.py` already expects.

Mutants are TRAIN-side and diagnostic; **they never enter the held-out reconstruction estimate.** Planting
into held-out would manufacture the very variance whose absence is the finding.

### Leg B (needs sign-off — do NOT start unilaterally)

Re-mine PR head-commits *before* review to get naturally-occurring violation variance. This is a **new
measurement target** and a new corpus, and the standing rule is no new measurement target without user
sign-off. Write the design, cost it, and ask. Do not spend GPU/API on it in the meantime.

---

## 7. The iteration loop (Stage 5) — "iterating until they actually have meaning"

Iteration is driven by **disagreement classes**, on TRAIN only.

```
repeat:
  1. run V_ast and V_llm on TRAIN
  2. bucket the disagreements into CLASSES by cause, adjudicated by an LLM (Sonnet+):
       - scope    (one saw an occasion the other didn't)     -> fix `applies`
       - polarity (agree on the fact, disagree on satisfied/violated) -> fix the predicate
       - witness  (same verdict, disjoint evidence)          -> at least one is keying on a proxy
       - lexical  (V_ast matched a token in a comment/string) -> the regex smell; kill it
  3. the LOSING implementation is revised — never both, and never toward each other's output
  4. re-run gates (§4) and the TRAIN-side certificate proxy (§5 on TRAIN)
until: no NEW disagreement class appears for K=2 consecutive rounds  (loop-until-dry)
       AND §4 gates pass
```

Then, and only then, touch held-out **once** for the certificate.

**Overfitting tripwire, mandatory:** if TRAIN κ rises across rounds while held-out κ stays flat, the verifiers
have been fit to each other rather than to the unit. Record it and stop; the unit is uncertified at this
verifier class and budget. Held-out κ is
checked exactly once precisely so this tripwire has teeth.

Revision is done by an LLM authoring agent given the disagreement class and the failing items — **not** by a
human hand-tuning constants, and **not** by a search over thresholds.

---

## 8. Readouts (Stage 6)

- **Never Spearman on a tied target.** Report **tercile AUC** (threshold-free, rank-based, per standing rule)
  and always alongside the target's tie structure: mode fraction, distinct-value count, n at each tercile.
  A ρ with no tie-structure disclosure is uninterpretable and is how 0.149 got shipped.
- **Report κ and lift together** (chance-corrected + against a base-rate baseline), per the hierarchy rule.
- **Applicability is a primary result**, not a footnote: `P(applies)` per unit, per corpus, natural vs planted.
- **The aggregation gap** — best learned aggregation over certified units vs the prompt-metric's item score —
  is the headline tacit-residue number.
- Every figure names its unit-level artifact path (standing rule).

---

## 9. Ban list (each entry killed a real result in the last 72 hours)

1. **Hand-set weights and magic constants.** `0.75 * py + 0.25 * comment`, `math.exp(-x * 25.0)`. If a number
   was chosen rather than fit or derived, it is banned.
2. **Regex-as-substance.** Regex may *locate* a candidate span; it may not *decide* a verdict. (The E2L
   function-wall kills were REGEX, not code — see `project_metric_seam_proposal`.)
3. **Scalar verifier outputs.** No floats. See §3.
4. **Permissive gates.** `min_unique_scores: 2` admits a program that returns 1.0 on 124/125 items.
5. **Spearman on tied targets.** See §8.
6. **Coverage-as-selection.** The funnel must gate on discrimination, or it selects for constants.
7. **Fixing the prompt when the parser is broken.** Fixing `SYSTEM_PROMPT` would have changed `request_sha256`
   and forced a needless 4,500-call re-run. Audit the harness before the model.
8. **Blaming the instrument before auditing the statistic.** Both confident verdicts in the last 24 hours were
   wrong in the same direction. When a result says "the model can't do this," the next move is to check
   whether *the target has variance*, not to write the negative result up.
9. **Excluding a failed smoke test.** The 12:52 transport smoke returned 2/2 contract errors and was "excluded
   from analysis"; 4,500 production calls launched 73 minutes later. **A 100% smoke failure is a STOP.**

---

## 10. Order of work

| # | Step | Cost | Blocking? |
|---|---|---|---|
| 1 | **Retract** the ρ<0.40 instrument-limit branch in the runbook + notebook. State the three defects (§0) and that the ceiling arm measures arithmetic, not articulation. | free | yes — the false record is live |
| 2 | Re-score the existing 18 cells with **tercile AUC + tie structure** (data already on disk, CPU only). Publish a37/a0/a92 ≈ 0.71 alongside the tie disclosure. | free, CPU | no |
| 3 | Replace `hierarchy_train_gate.py` with the §4 gate; wire the lane into `battery/contract_check.py`. Re-run the funnel. **Expect ~4 of 16 programs to die before any model call.** | CPU | yes for step 5 |
| 4 | Stage 0 join: map the 18 cells' metrics onto Certified Unit Framework units. Pick the ~10 units that pass §4. | CPU | yes for step 5 |
| 5 | Author V_ast + V_llm independently per surviving unit (§5). Run Leg-A plants (§6). Iterate (§7) on TRAIN. | Sonnet fan-out; modest | — |
| 6 | Certificate pass on held-out, **once**. Report κ, witness Jaccard, ablation. | one held-out pass | — |
| 7 | Aggregation gap (§3), only over certified units. | CPU + one fit | — |
| 8 | Write up Leg B (§6) and **ask** before mining. | free | — |

Steps 1–4 are CPU-only and cost nothing. **Do not launch a model run until step 3 has killed the constants.**

### Science / Patents / Math (added 2026-07-13)

**Do not port the verifier interface to other domains until it is proven on code-review.** It is unproven;
porting an unproven interface to three more lanes multiplies the blast radius of any design error in it (and
this roadmap has already had one — see §0 defect 2).

**But run the *diagnostic* on those lanes now.** The tie-structure / discrimination audit (§4 gates + mode
fraction + distinct-value count on each lane's existing coded targets) is CPU-only, costs zero model calls,
and answers a question that gates everything downstream: **are those lanes' targets degenerate too?** If they
are, any prompt batch already prepared or already run against them is invalid for the same reason the
code-review ρ is invalid, and we want to know that *before* more calls are spent — not after.

That is an audit, not a lane. Do not run the Science or Patent prompt batches in the meantime.

---

## 11. Landmines

- **Never `--resume` a Codex rescue review** (hangs). Fresh run + inline context.
- `SYSTEM_PROMPT` in `hierarchy_prompt_batch.py` is **frozen** — changing it invalidates the 4,442 recovered
  responses via `request_sha256`.
- v3's prompt-jobs bundle is byte-frozen; `compile_ceiling_channel.py` exists precisely so the ceiling arm was
  emitted *additively* rather than by recompiling (v3 was itself a filter of v2, 21→18 cells).
- GLM monthly quota is binding. Steps 1–4 need zero GLM.
- sk3 GPUs 1–4 remain excluded.
- Standing rules that bind this lane: LLM judges do all measurement (code orchestrates); Sonnet-or-better for
  any judging/adjudication; threshold-free readouts for cross-family comparison; stable hash splits; never
  delete data (append + dedup); no new measurement target without sign-off.

---

## 12. Implementation status (2026-07-13, additive)

The verifier interface has now been tested far enough to make the domain-routing
decision the roadmap was intended to support.

- Shared schema, path-aware witnesses, natural-only TRAIN gate, certificate
  statistics, exact smoke STOP, split binding, and one-shot held-out receipt
  enforcement are implemented. The complete `methods/metric_seam` suite passes
  (1,046 passed, 1 skipped, 1 healthy XPASS at the integration audit).
- Code review does **not** advance to a certificate: 0/4 real CUF units pass the
  natural merged-PR gate, although 152/160 plants are detected. This establishes
  corpus inadequacy for these relations and satisfies the roadmap's early-stop
  purpose. Keeping the interface confined to code review would now prevent the
  study from reaching a certificate for a known corpus reason.
- The technical diagnostic therefore routes the next bounded run to Math. The
  existing a12 SymPy pipeline supplies a deep (parser + exact symbolic solver)
  retrospective seed. Its pair-level natural TRAIN distribution is 328
  not-applicable / 24 satisfied / 91 violated over 443 candidates, and it passes
  every frozen prevalence/discrimination/probe check.
- The Math prompt-side contract was authored separately from the symbolic
  implementation and never receives SymPy outcomes. It does share the
  code-proposed adjacent-pair extractor, however. The licensed TRAIN estimand is
  therefore Sonnet/SymPy applicability and polarity agreement conditional on a
  shared proposed pair—not independent end-to-end decomposition. Witness
  overlap is fixed by that interface, and empirical ablation remains undone, so
  the TRAIN run is not a certificate. This still preserves the distinction:
  articulability is prompt-based, verifiability is code-based, and their
  conditional agreement is a separate empirical readout.

This port does not reopen Grants, Legal, Creative Writing, or Humor. Patents and
full-article Science remain diagnostic-only until the Math TRAIN comparison
either passes or gives a bounded failure. No GPU is used by this path.


## Proposed extension for discussion: reconstruct code units, not only outputs

**Status:** proposal only. Do not implement or launch until the research team gives explicit
go-ahead.

The current reconstruction objective is primarily behavioral: freeze a code verifier, run it
on items, and ask whether a prompt-based judge reconstructs the verifier's output ordering or
verdicts. This is useful, but it assumes the code-side decomposition instead of testing whether
the articulated metric and executable program decompose into corresponding relation-level
units.

A stronger design would make the **code unit** the reconstruction target while keeping the
study unsupervised. Neither side is treated as supervised external ground truth. Freeze the
existing relation-level code units and, independently, have a prompt-only process reconstruct
relation units from the metric text without seeing the code, code outputs, or held-out items.
Then align the two sets as a matching problem and measure four distinct objects:

1. **Structural reconstruction:** which relation-level units appear on both sides, which prompt
   units lack executable counterparts, and which executable units add relations not recovered
   from the prompt.
2. **Applicability reconstruction:** for an aligned unit, whether prompt and code agree about
   the items or spans to which the relation applies.
3. **Behavioral reconstruction:** conditional on joint applicability, whether the prompt
   reproduces the executable verdict or ordering.
4. **Residual seam:** the unmatched prompt-side units, unmatched code-side units, and aligned
   units whose applicability or polarity diverges.

This preserves the project vocabulary: **articulability is prompt-based, verifiability is
code-based, and isomorphism/agreement is a separate measured relation between them.** Code may
also outperform the prompt; an unmatched or higher-resolution executable unit is therefore a
result, not automatically a reconstruction failure.

### Why Math a12 motivates this extension

The current Math a12 TRAIN experiment supplies both SymPy and Sonnet with equality pairs
proposed by the same structural code extractor. It therefore measures applicability and
polarity agreement conditional on a code-proposed unit, not independent reconstruction of the
unit itself. On the 437 contract-valid responses observed so far, applicability agreement is
326/437 (74.6%, kappa=.445). On the 91 pairs both systems call applicable, equality-versus-
nonidentity agreement is 91/91 (100%, kappa=1.0; 23 satisfied and 68 violated). This suggests a
capability frontier at representation/applicability for this bounded relation, but it does not
show that prompt and code independently discovered the same decomposition.

### Minimal decision experiment for tomorrow

Do not redesign the full census first. Select a small technical slice—Math a12 plus one
claim-based verifier from Patents or full-article Science—and:

1. Freeze the existing code-unit inventory and its provenance.
2. Generate prompt-only relation units from metric text under a sealed instruction that
   excludes code, code outputs, reference judgments, and held-out items.
3. Blindly align prompt units to code units using an explicit relation-matching rubric. Retain
   unmatched units on both sides rather than forcing matches.
4. Report set-level unit alignment, item-level applicability agreement, and conditional
   verdict agreement as separate readouts.
5. Use implementation-disclosed prompting only as a manipulation check, not the primary
   structural-reconstruction arm.

Before implementation, decide the unit representation and matching statistic. Candidate
readouts are precision/recall or maximum-weight bipartite matching over relation units, with
agreement uncertainty reported by unit- and item-clustered resampling. Avoid converting this
into another per-cell promotion grid.

### Guardrails

- No supervised external anchor or imported ground truth.
- No use of held-out outcomes to author, select, align, or revise units.
- No claim that unmatched prompt units are tacit; they are only unverified within the frozen
  code class and budget.
- No claim that unmatched code units are unarticulable; they may reflect executable
  overperformance or a prompt-reconstruction miss.
- Keep structural, applicability, and behavioral agreement separate; do not compress them into
  one codability percentage.
- Reuse the existing manual/mock technical decompositions honestly as frozen pipeline
  artifacts rather than pretending to rediscover them automatically.

## Implementation result: proposal-first repair (2026-07-14)

The bounded first release of the repaired ordering is complete. The implemented
workflow is `PROPOSE → BASE-RATE PROBE → AUTHOR/IMPORT → CONSTRUCT CHALLENGE →
PER-NODE GATE → SELECT → FREEZE → TRANSCRIBE → EVALUATE`.

- **Math a12 stops at construct validity.** The old context-free identity
  verifier scores 0/12 contextual controls correctly: it calls eight legitimate
  definitions, hypotheses, constraints, and equations-to-solve violations, and
  abstains on four rigor defects without equations. Its 91/91 conditional
  agreement is now a counterexample to agreement-as-validity. The narrow
  symbolic-identity capability remains retained.
- **Code review stops at corpus support.** Zero of four natural TRAIN gates pass
  while 152/160 plants are detected; no prompt reconstruction is launched.
- **The fresh Patent antecedent proposal passes pre-authoring prompt support**
  with 32/32 valid judgments (14 satisfied, 18 violated). The first smoke was
  2/5 and stopped; a disclosed line-numbered transport correction passed 5/5
  before production. Only then was the preexisting manual claim graph imported.
  Its binary natural TRAIN output is 1 not-applicable / 1 satisfied / 148
  violated, so P(violated|applies)=148/149 and the unit stops before selection.
  It also misses two of eight fixed construct controls before blind control
  adjudication. No 150-item transcription or held-out run was launched.
- **a34 remains audit-reported only.** The two dead subtree nodes were not found
  as exact locally bound artifacts, so this release records the provenance gap
  rather than claiming independent reproduction.

Canonical summary:
`outputs/metric_seam_pilot/verifier_pipeline_v2/construct_validity_repair_summary_v1/`.
The proposed code-unit structural reconstruction experiment above remains
unlaunched pending separate approval.

## Approved family-scale technical pilot (2026-07-14)

The separate approval has now been given. Scaling proceeds by reusing relation
families, not by authoring one scorer per metric. The frozen first batch is 60
outcome-blind metrics: five stable-hash selections from each R1/R2/R3 round in
Math, Code Review, Patents, and full-article Peer Review/Science. Equal metric
budgets do not imply equal expected yields; base-rate kills are primary data.

Canonical manifest:
`outputs/metric_seam_pilot/family_scale_v1/study_manifest.json`.
Its decomposition view exposes only `construct` and `description`. Corpus
items, programs, children, scores, prior judgments, and held-out records are
excluded by construction. Three independent decomposition fleets are compared
on all 60 metrics; stability is reported rather than assumed.

The implementation adds four non-negotiable design constraints learned from
the audited pilots:

1. The occasion proposer and occasion IDs are shared byte-for-byte across
   prompt and code channels. Whole-document discovery is out of scope except
   for a separately labeled blind-discovery subsample.
2. Prompt batching is capped at two or three relations, relation co-occurrence
   is deterministically randomized, and 10% of occasions receive an unbatched
   calibration call. Prompt-call ID is retained as an analysis cluster.
3. G1 dual implementations and G2 proxy traps certify the reusable family x
   domain-corpus combination, not every metric independently. A failed family
   certificate uncertifies every dependent cell together.
4. Primary behavioral fidelity is tie-robust pairwise concordance conditional
   on joint applicability, normalized by two-pass prompt reliability. Spearman
   is secondary. Every report carries proposed, base-rate-killed, authored,
   gate-killed, and operational denominators plus target tie/mode structure.

Thirty-item base-rate probes are provisional authorship kill screens, not rate
certificates. Discrimination is recomputed on full TRAIN execution before any
prompt transcription. Code Review is restricted to AST/dataflow/static
relations supported by diffs; mocked test execution is excluded until a
pre-review corpus is explicitly approved and re-mined. Science uses full
articles, not abstract-only artifacts. Mocked/precomputed Patent retrieval is
allowed only with explicit provenance.

The six predeclared distributional estimands are decomposition stability,
pre-authoring corpus-support incidence, family-certification incidence,
full-TRAIN operational witness incidence, prompt/code reconstruction fidelity,
and program depth/typed seam position. No per-cell promotion grid is primary.

### First scaled result

The structural stage is complete for all 60 metrics. Three blind fleets yielded
180 decompositions; one blinded semantic-alignment call per metric yielded 60
complete structural readouts. Median metric-level mean pairwise semantic
Jaccard is `.603` with metric-bootstrap 95% CI `[.524,.667]`. Domain medians
are Code `.587`, Math `.576`, Patents `.667`, and full-article Science `.524`.
Exact normalized-string overlap is zero for all 60 metrics, so it is retained
only as an observed-zero lower bound.

This is an articulability/decomposition-stability result, not verifiability or
behavioral isomorphism. The semantic aligner has one pass and no external
anchor. Also, 148/180 fleet outputs hit the frozen five-relation maximum, so
decomposition width is censored and capture–recapture is not interpreted.
Canonical artifacts are
`outputs/metric_seam_pilot/family_scale_v1/structural_readout.json` and
`structural_report.md`. The next active stage is cross-metric family induction
followed by thirty-item provisional corpus base-rate probes.
