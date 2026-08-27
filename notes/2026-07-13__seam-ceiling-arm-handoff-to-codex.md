# Metric-seam handoff — ceiling arm + the 2026-07-13 contract-error correction

**Author:** Claude (audit + fixes), 2026-07-13. **Owner from here:** Codex.
**Status:** parser fixed, run recovered, ceiling arm compiled and RUNNING.

---

## 1. Corrections to the record — please do not re-litigate these

The 2026-07-13 `code_review_glm52_impl_summary_v1` run was **not** a GLM-5.2
output-contract failure. That diagnosis is wrong and the runbook entry currently
in the repo (`notes/2026-07-10__seam-agentic-program-runbook.md`, the
`2026-07-13 CODE-REVIEW RELATION-LOCAL RECONSTRUCTION` bullet) records it as fact.
**You own that file — please correct it.** Replacement text is in §6.

| claim in the record | truth |
|---|---|
| "0 valid, 4,500 contract errors" | **4,442 valid / 58 true contract errors (1.29%)** |
| "a bounded GLM-5.2 output-contract failure" | a harness deserialization bug; GLM honored the schema on ~99% of rows |
| "rho is undefined, not low or zero" | **rho is defined and it is low**: median raw 0.146, 95% CI [−0.280, +0.623] |
| "the run failed before measurement" | the run reached measurement; 3/18 mappings have confirmatory support |

The evidence was already inside the entry: it reports "4,498 responses were
Markdown-fenced" and then blames the model. The fence *was* the bug.

**Root cause:** `run_hierarchy_prompt_jobs.py:334`, `parsed = json.loads(raw_response)`.
Two independent defects: (a) no Markdown-fence unwrap; (b) `strict=True`, which
rejects literal tabs inside JSON strings — and evidence spans quote tab-indented
source, so it dropped rows *non-randomly*, selecting against tab-indented languages.

**The test suite had enshrined the bug.** `test_run_hierarchy_prompt_jobs.py`
contained a case feeding a fenced *valid* response and asserting `contract_error`.
It is now flipped.

**Process lesson:** the 12:52 transport smoke returned **2/2 contract errors** and
was "excluded from analysis"; the 4,500-call run launched anyway at 14:05.
A smoke test that fails 100% is a stop signal, not a row to exclude.

---

## 2. What changed in the code (all tests pass: 14/14)

| file | change |
|---|---|
| `run_hierarchy_prompt_jobs.py` | new `deserialize_response()`: unwraps a fence, parses `strict=False`. **Deserialization only** — no payload repair, no prose-scraping, no retry. `validate_prompt_response` still governs the payload unchanged. |
| `recover_prompt_responses.py` | **new.** CPU replay of the fixed parse path over an existing responses file. No model calls. Never writes in place. |
| `hierarchy_prompt_batch.py` | new channel `full_executable_contract` (the ceiling arm) + `_full_contract_prompt`; `omitted_channels` now `{}`, replaced by a `ceiling_arm` block. |
| `compile_ceiling_channel.py` | **new.** Emits the ceiling arm additively against v3's exact 18 cells, so v3 stays byte-identical and the filter is not re-derived. |
| `test_run_hierarchy_prompt_jobs.py` | bug-enshrining test flipped; added tab-in-evidence and schema-violation cases. |
| `test_hierarchy_prompt_batch.py` | ceiling-arm assertions; `n_channels` 3→4, `n_jobs` 15750→21000. |

**Do not "fix" the formatting by editing `SYSTEM_PROMPT`.** That changes
`request_sha256` and would invalidate the 4,442 recovered responses, forcing a
needless re-run. The fix belongs at the parser and it is done.

---

## 3. Artifacts

Recovered implementation-disclosed arm (originals untouched):
```
outputs/metric_seam_pilot/hierarchy_r123/results/code_review_glm52_impl_summary_v2_recovered/
    responses.jsonl   readout.json   report.md
```

Ceiling arm:
```
outputs/metric_seam_pilot/hierarchy_r123/code_review_reconstruction_ceiling_jobs_v1.jsonl.gz   (4,500 jobs)
outputs/metric_seam_pilot/hierarchy_r123/results/code_review_glm52_ceiling_v1/
    smoke.jsonl (6/6 valid)   responses.jsonl (RUNNING)   run.log
```

---

## 4. What is running right now

The full ceiling arm: 18 cells × 125 items × 2 passes = 4,500 calls, glm-5.2,
concurrency 3, ~45 rows/min, ETA ~100 min from 2026-07-13 ~15:0x. **Resume-safe** —
if it dies, re-run the identical command; `load_completed_request_ids` skips what
finished. Smoke was 6/6 valid (the old smoke was 2/2 contract errors).

```bash
python -m methods.metric_seam.run_hierarchy_prompt_jobs \
  --jobs   outputs/metric_seam_pilot/hierarchy_r123/code_review_reconstruction_ceiling_jobs_v1.jsonl.gz \
  --channel full_executable_contract \
  --backend zai_anthropic --model glm-5.2 \
  --temperature 0.2 --max-tokens 1024 --concurrency 3 --expected-jobs 4500 \
  --output outputs/metric_seam_pilot/hierarchy_r123/results/code_review_glm52_ceiling_v1/responses.jsonl
```

**Cost note:** ~16M GLM input tokens (prompts carry the full program source,
1.7K–5.2K tokens each). GLM quota is monthly and binding — be sparing elsewhere.

### Why this arm exists

v3 had **no upper anchor**. Its own manifest admitted it:
`"full_executable_contract_ceiling": "omitted"`. All three channels were
*impoverished* articulations, so a low rho anywhere was uninterpretable — it could
not separate "the executor cannot reconstruct this program" from "no summary at
this disclosure level could." The ceiling arm discloses the literal program
(digest-bound to the executed artifact) and asks the model to simulate it.

The smoke confirms it does exactly that — rationales cite `applies()`, the LIZARD
extension filter, tree-sitter — rather than exercising the model's own taste.

---

## 5. What to do when it lands

### 5a. Run the analyzer

```bash
python -m methods.metric_seam.analyze_code_review_reconstruction \
  --prompt-manifest outputs/metric_seam_pilot/hierarchy_r123/code_review_reconstruction_prompt_manifest_v3.json \
  --prompt-jobs     outputs/metric_seam_pilot/hierarchy_r123/code_review_reconstruction_ceiling_jobs_v1.jsonl.gz \
  --responses       outputs/metric_seam_pilot/hierarchy_r123/results/code_review_glm52_ceiling_v1/responses.jsonl \
  --code-execution  outputs/metric_seam_pilot/hierarchy_r123/code_review_heldout_execution_v1.json \
  --bootstrap-draws 10000 --bootstrap-seed 20260713 \
  --output outputs/metric_seam_pilot/hierarchy_r123/results/code_review_glm52_ceiling_v1/readout.json
```
The analyzer may need its manifest/channel binding relaxed to accept the ceiling
jobs file. If so, extend it — do not mutate the frozen v3 manifest.

### 5b. The reading, declared now, before you see the number

| ceiling rho vs code | what it licenses |
|---|---|
| **high (≥ .70)** | The executor *can* simulate the program. The ladder becomes readable, and `implementation_disclosed`'s rho = .146 is **disclosure loss** — a real, quantified tacitness result localized to applicability/polarity/aggregation. This is the paper. |
| **low (< .40)** | The executor cannot simulate the program even given the full source. Then **nothing below it is interpretable**: rho = .146 indicts the executor or the item panel, not articulation. Report as an instrument limit. **Do not claim tacitness.** |
| intermediate | Report the ladder descriptively with CIs. No verdict. |

Report ceiling-normalized rho (rho / two-pass reliability), as the analyzer
already does — the reliability is high (median .897), so the low rho at
`implementation_disclosed` is genuine divergence, not noise.

### 5c. The free diagnostic — do this one first, it is decisive and costs nothing

Compare three abstention rates on the same 18 cells × 125 items:

1. the **code's** own abstain rate (from `code_review_heldout_execution_v1.json`),
2. the **ceiling arm's** `not_applicable` rate,
3. the **implementation_disclosed** arm's `not_applicable` rate (**84.1%**).

If ceiling ≈ code, the program's `applies()` gate genuinely rarely fires: sparse
firing is **real**, and the design is item-starved (125 items is small) — fix by
adding items, not channels. If ceiling fires far more than
`implementation_disclosed`, then the 84.1% was the model **guessing applicability
that the channel withheld** — which localizes the tacit residue precisely, and is
itself a result.

---

## 6. Replacement text for the runbook entry (you own that file)

> - 2026-07-13 **CODE-REVIEW RELATION-LOCAL RECONSTRUCTION — HARNESS BUG, RUN
>   RECOVERED, CEILING ARM ADDED.** The reported "0 valid / 4,500 contract errors"
>   was a **deserialization defect in the runner, not a GLM-5.2 contract failure**:
>   `run_hierarchy_prompt_jobs.py` parsed `raw_response` with a bare `json.loads`,
>   unwrapping no Markdown fence and rejecting literal tabs inside evidence strings
>   (a *non-random* drop, selecting against tab-indented languages). GLM-5.2 honored
>   the response schema on ~99% of rows. Replaying the fixed parse over the retained
>   raw text recovers **4,442/4,500 valid (98.71%); true contract errors 58 (1.29%)**,
>   with **no new model calls**. The test suite had *enshrined* the bug (a case
>   asserting that a fenced valid response is a `contract_error`); it is flipped.
>   The 12:52 transport smoke had already returned 2/2 contract errors and was
>   "excluded from analysis" — a 100% smoke failure is a stop signal, not a row to
>   exclude.
>   **Corrected result (not a null):** rho is **defined and low**. Median raw
>   Spearman **0.146**, 95% clustered-bootstrap CI **[−0.280, +0.623]**; 3/18
>   mappings confirmatory, 5 exploratory, 8 without support. Median two-pass
>   reliability **0.897** — so the prompt side is self-consistent and the low rho is
>   genuine divergence, not noise. Three cells are **negative** (a43 −.300,
>   a401 −.201, a15 −.200), and a43 "intention-revealing naming" (R3, −.300) opposes
>   a70 "intention-revealing naming" (R1, +.327) — the *program*, not the construct,
>   drives the sign, as withheld polarity predicts. The wrong-relation control
>   contrast is **undefined (no support)** — a designed control that currently yields
>   nothing; diagnose before any specificity claim.
>   **Interpretive limit → new arm.** v3 shipped with `full_executable_contract_ceiling:
>   omitted`, i.e. **no upper anchor**: every channel was an impoverished articulation,
>   so a low rho could not be separated from disclosure loss. A ceiling arm
>   (`full_executable_contract`, 4,500 jobs, complete program source digest-bound to
>   the executed artifact) is compiled and running. Smoke 6/6 valid; rationales cite
>   `applies()`/LIZARD/tree-sitter, i.e. the model simulates the program rather than
>   exercising its own taste. Until it lands, **the .146 must not be reported as
>   tacitness or as code/prompt disagreement** — only as reconstruction under an
>   incomplete relation summary.
>   Artifacts: `results/code_review_glm52_impl_summary_v2_recovered/` (recovered
>   readout), `results/code_review_glm52_ceiling_v1/` (ceiling arm),
>   `code_review_reconstruction_ceiling_jobs_v1.jsonl.gz`.

---

## 7. Continuation, in priority order

1. **Ceiling arm lands → run §5c diagnostic, then §5a analyzer, then apply §5b.**
   Everything else waits on this. It decides whether the ladder is readable at all.
2. **Then, and only if the ceiling is high**, run the two remaining v3 channels —
   `source_only_subrelation` and `source_only_whole_construct`, 4,500 calls each
   from `code_review_reconstruction_prompt_jobs_v3.jsonl.gz` — to complete a
   4-point disclosure ladder: whole construct → subrelation → relation summary →
   full contract. If the ceiling is *low*, these are wasted calls; do not run them.
3. **Fix the wrong-relation control.** Its contrast is undefined for want of
   support. It is the specificity arm — without it, a positive rho cannot be
   distinguished from a generic code-quality prior. Diagnose why it has no support.
4. **Power.** Only 8/18 cells are gradeable and 3 are confirmatory; median cell
   n ≈ 12. The §5c diagnostic tells you whether this is fixable by adding items
   (sparse firing real) or by disclosing applicability (sparse firing artifactual).
   Do not scale channels before scaling support.
5. **Double-selection hazard — name it in any writeup.** Common support is the
   triple intersection (code ∧ pass-1 ∧ pass-2). Both the program and the prompt
   choose which items to speak on, and that choice is not independent of the
   construct. Every rho here is computed on a doubly-selected subsample.

## 8. Landmines

- **Never** repair formatting via `SYSTEM_PROMPT` — it breaks `request_sha256`
  comparability with the 4,442 recovered responses.
- **Never** overwrite `..._impl_summary_v1/` or the v3 jobs/manifest. Everything
  here is additive; the originals are evidence.
- **Do not** re-run the 4,500 implementation-disclosed calls. They are recovered.
- A 100% smoke failure is a **stop** signal.
- The claim limits in the existing `report.md` are correct and hard-won — carry
  them forward verbatim. This is *conditional relation-local* reconstruction after
  an executable witness was already found. It is **not** whole-metric codability,
  literal source reconstruction, tacitness, or external correctness.
