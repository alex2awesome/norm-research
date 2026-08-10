# Code-review (GitHub PR merge) A-bank: provenance of V .576 / V+A .592, and rescore on the enriched v3 text

2026-08-06. Two-stage task. Stage 1 = verify what instrument produced the published
articulated-baseline numbers for the software-code verdict cell. Stage 2 = rescore that
bank on the NEW enriched v3 text (Title + Description + inline review comments + diff) so
that a same-input dense-vs-(V+A) comparison exists.

Trigger: `notes/2026-07-27__vat-run-registry.md` 2026-08-06 entry — code dense on the v3
enriched corpus is eval .6488 / test .7373 vs a matched diff-only baseline .5851 / .6618.
Face-value that inverts the old ordering (T .649 > V+A .592), but the same-input rule
forbids any residual claim until the articulated bank is rescored on the same text.

---

## STAGE 1 — PROVENANCE VERDICT

### Headline: the published code "A" was NOT an LLM judge

**`instrument_change = true`.** The A layer behind the published code-cell numbers was the
**deterministic CODED metric backend** — `methods/existing_metrics_runner/coded/`, one
Python module per aspect calling real verification tools (tree-sitter parsers, lizard,
ruff, eslint, radon, semgrep, …) with the contract `applies(diff_text) -> bool`,
`score(diff_text) -> float | None`. **No language model was involved in producing A.**

This is the math.SE landmine again, in a different shape: math.SE's problem was a
non-standard judge; code's problem is *no judge at all*. Under the standing rule
"measurements = LLM judges, NEVER coded proxies"
(`feedback_llm_judges_do_all_measurement`), the published code A is a coded-proxy
measurement and is **not** an A-bank measurement in the sense of the Gemma-4-31B banks
used for humor / creative-writing / peer review.

### The exact chain

| element | what it actually was |
|---|---|
| **Judge model** | **none** — deterministic Python (`methods/existing_metrics_runner/coded/metrics/a*.py`), run CPU-side, 32-way parallel, ~43 diffs/s |
| **A criteria** | **127 coded aspect modules**, columns of `pr_a_metrics_full.parquet`; the ladder further filtered to **>5% coverage → 72 aspects** |
| **Criteria text source** | `runs/validity_full/v2/code_review/aspects.json` (394 mined code-review aspects: `aspect_id` / `name` / `description`). Only **74 of the 127** coded aspects appear in that catalog; the other 53 (a4xx/a5xx) were added after the catalog was frozen |
| **A input** | the **bare diff** — `batch_runs/<repo>/diffs/pr_<N>.diff`. No title, no description, no review comments |
| **V features** | **execution-derived, not text-derived**: `v_p2f`, `v_f2p`, `v_has_signal`, `v_smoke_rc`, `v_baseline_failed/passed`, `v_post_failed/passed` — i.e. outputs of the docker test-execution fleet, read from `outputs/consolidated_verdicts_ALL_final.csv` |
| **Corpus** | 44,751 "sound" PRs / 594 repos = join of consolidated verdicts × `pr_a_metrics_full.parquet` (68,083 diffs), flaky-batch filter applied |
| **Aggregation** | `scripts/pr_vat/run_vat_ladder.py`: StandardScaler + `LogisticRegression(C=0.1, max_iter=500)`, NaN→0, `cross_val_score` → **mean of fold AUCs** (not pooled OOF) |
| **y** | `judgement == 'rejected'` (merge status) |

Scripts: `scripts/pr_vat/compute_a_metrics.py` (A), `scripts/pr_vat/build_vat_table.py`
(join), `scripts/pr_vat/run_vat_ladder.py` (ladder).
Artifacts that survive on sk3:
`datasets/code-review/pr_test_execution/outputs/pr_a_metrics_full.parquet`
(68,083 × 257 = 127 `_score` + 127 `_applied` + keys) and
`datasets/code-review/pr_test_execution/batch_runs/*/verdicts.jsonl` (4,344 files).
**Gone:** `outputs/consolidated_verdicts_ALL_final.csv` (the V source) and
`/tmp/final_ladder_table.parquet`.

### Second finding: the published row splices two protocols

The registry row is `V .576 / VA .592`. Those two numbers come from **different
cross-validation designs** in the same run (`project_pr_vat_audit`, 2026-07-09 ladder):

| protocol | V | A | V+A |
|---|---|---|---|
| pooled (StratifiedKFold, repos mixed across folds) | **.576** | .615 | **.632** |
| grouped (GroupKFold by repo, repo-disjoint) | **.549** | .596 | **.592** |

So `V .576` is the **pooled** V and `VA .592` is the **grouped** V+A. Read within one
protocol the cell is either pooled **.576 → .632 (+.056)** or repo-disjoint
**.549 → .592 (+.043)**. The spliced pair understates A's lift by ~25% and mismatches the
grouping of the dense number it is compared against. The repo-disjoint pair
(**V .549 / V+A .592**) is the one that is protocol-matched to a repo-disjoint dense
number, and is what should be quoted alongside .592.

### Third finding: ~16 of the 72 scored aspects are not code-review norms

Of the 127 coded aspects, 53 post-date the mined catalog. Inspecting them:

* **a400–a409** are genuine review norms (complexity class, function decomposition,
  recursion quality, idiom density, data-structure choice, purity, magic numbers, name
  expressiveness, error handling, explanatory comments) — judge-portable.
* **a478–a482** are LeetCode-corpus **execution outcomes** (does the candidate pass its
  tests; runtime/memory percentile) — no text-judge form, and imported from a different
  study.
* **a500–a522** are **authorship/structure fingerprints** (anonymized 4-gram entropy, AST
  node-type sequence entropy, branching density, max loop nesting, Chinese-character
  comment ratio, competitive-programming tag counts, function-def density) — again
  imported, and not norms anyone asserts in a review.

16 of these 25 non-norm aspects clear the >5% coverage filter, i.e. **they were inside the
published A**. This is consistent with the 2026-07-08 audit finding that part of A's edge
was repo fingerprinting.

### Reproducibility from on-disk artifacts

**NO — the published ladder does not reproduce from what is on disk.**
`pr_a_metrics_full.parquet` survives, so A is reconstructible. V is **not**: the
consolidated CSV is gone, and today's `verdicts.jsonl` no longer carries `smoke_rc` /
`baseline_*` / `post_*` — only `verdict`, `rc`, `judgement`, `base_sha`, `era_id`,
`image_tag`. A good-faith reconstruction (`abank_rescore/repro_published_ladder.py`, V
rebuilt as verdict-category one-hots + `rc` + P2F/F2P gates, published CV protocol
verbatim) gives:

| protocol | V | A | V+A | published |
|---|---|---|---|---|
| pooled StratifiedKFold | .6213 ±.005 | .6559 ±.010 | .6858 ±.008 | .576 / .615 / .632 |
| grouped GroupKFold(repo) | .4971 ±.041 | .5493 ±.041 | **.5260** ±.040 | .549 / .596 / **.592** |

Corpus recovered: **28,556 rows / 352 repos** against the published **44,751 / 594** — the
verdicts×A-parquet intersection has lost ~36% of the rows since July. V came out with 20
features instead of 8 (verdict one-hots + `rc` carry repo/era information the original 8
did not), which is why pooled V lands *above* the published .576 while the grouped numbers
land *below*. Grouped fold spread is ±.04, and in this reconstruction V+A (.526) sits
*below* A alone (.549).

**Consequence: `V+A = .592` is no longer verifiable.** It should be treated as a
historical figure with a lost provenance chain, not as a live baseline. Combined with the
instrument finding (it was never an LLM judge) and the protocol splice (pooled V spliced
onto grouped V+A), the honest position is that the code cell needs a fresh articulated
baseline — which is what stage 2 produces.

### Decision-gate call

Gate said: proceed to Gemma-4-31B if the A instrument is Gemma-compatible *or the criteria
are judge-portable text rubrics*; if the original judge was something else, flag
`instrument_change=true` and proceed anyway because the criteria are what matter.

The original "judge" was a deterministic program, so **both the input and the instrument
change** versus .592. The criteria *are* judge-portable for 83 of the 127 aspects (74 with
catalog `description` text + a400–a409 with norm statements written from their module
`ASPECT_NAME` + docstring). Proceeding with those **83 portable criteria**; the 16
non-portable fingerprint/LeetCode aspects and 28 aspects with no catalog text are
documented and dropped rather than silently included
(`abank_rescore/criteria_code_abank.jsonl` carries `portable` + `drop_reason` per aspect).

**Every comparison below therefore changes TWO things at once versus V+A .592: the input
(bare diff → enriched PR object) and the instrument (coded programs → Gemma-4-31B judge).
The rescored numbers are new-instrument numbers and must never be differenced against .592
as if only the text had changed.**

---

## STAGE 2 — ENRICHED RESCORE (in flight)

### Design

* **Rows**: v3 eval (5,822) + test (5,630) = 11,452. Train rows (47,659) are **not**
  scored — the stack is fit grouped-OOF *within* eval and, separately, *within* test,
  mirroring the published aggregation, which also fit inside the same row set with
  GroupKFold(repo).
* **Judge**: Gemma-4-31B-it, offline batch vLLM, sk3 GPU 6, `gpu_memory_utilization` .90
  (GPU was idle: 0 MiB), spawn + `CUDA_DEVICE_ORDER=PCI_BUS_ID` per the fork-wedge rule,
  temperature 0, `max_tokens` 6, prefix caching on, one token per (row, criterion),
  label-blind (the merge outcome never enters a prompt).
* **Criteria**: 83 portable code-review criteria (74 with catalog `description` text + 9
  hand-written a400–a409 norm statements). **13 of the 83 are enrichment-sensitive** —
  they ask about evidence that exists only in the v3 text and was simply absent from the
  v2 bare diff: a105 change-description clarity/rationale, a78 change/commit/PR
  communication quality, a258 commit message structure, a123 contribution readiness and
  submission norms, a130 decision records (RFC/ADR), a8 small/focused/reviewable changes,
  a52 documentation quality, a175 documentation formatting, a50 + a15 + a409 commenting
  strategy/intent, a309 tests-included-with-changes, a12 design-pattern use. This is the
  concrete mechanism by which enrichment could move A: under v2 these 13 criteria were
  unanswerable in principle, so the coded bank scored them off the diff or abstained.
* **Truncation**: `max_model_len` 12,288, text budget 11,600 tokens. Median v3 text is
  **6,801** Gemma tokens; **427 / 11,452 = 3.73%** of rows truncate.
* **Fairness note that cuts the right way**: the dense v3 model is Llama-3.1-8B LoRA with
  `max_length` **2048 tokens**, i.e. it saw only the first ~2K tokens of the same text.
  The judge sees up to 11.6K. So A is *advantaged* on input; a dense-beats-(V+A) result is
  therefore conservative. If A were to win, a matched-2048 rerun would be required before
  quoting it.
* **V on v3**, two blocks, both on the exact v3 rows:
  * `V_exec` — the published instrument family (verdict category, rc, P2F/F2P gates),
    **joined**, not recomputed: its extractor is a docker test run, not a text extractor,
    so it is invariant to the text enrichment by construction. Joins **11,452 / 11,452
    (100%)**.
  * `V_text` — deterministic text/structure features **recomputed on the v3 text**
    (title/description/comment/diff geometry: section lengths, comment-bullet count,
    `\`\`\`suggestion` count, files, test files, test-file fraction, hunks, lines
    added/removed, issue refs, …).
  * `V_all = V_exec + V_text` is the headline V — the most generous V, so a residual
    measured against it is conservative.
* **A matrix** mirrors the published coded construction: 83 score columns (NA
  median-imputed) **plus** 83 `applied` indicators (1 = judge scored it, 0 = judge answered
  NA), exactly as the coded parquet carried `(score, applied)` pairs.
* **Stacks**: linear = StandardScaler + LogisticRegression, grouped OOF by repo, reported
  under both the Layer-1 protocol (C=1.0, pooled OOF AUC — comparable to the other cells)
  and the published protocol (C=0.1, mean of folds). Nonlinear `VA_nl` = frozen Layer-1
  HistGB grid (`max_leaf_nodes` {15,31}, lr .06, 400 iters, early stopping), nested inner
  GroupKFold(3), seeds 0/1/2.
* **Anchors**: blinded 3-tier battery, **K=50 per tier**, drawn from the **train** split
  only (never eval/test): merged / closed-unmerged / word-scrambled. Note on
  interpretation — pos-vs-neg here *is* the outcome contrast, which is exactly the weak
  quantity under test, so the binding gate is **scrambled << both real tiers** (the judge
  is reading content, not length); pos>neg is reported but not retried on.

### Smoke test (40 rows × 83 criteria = 3,320 prompts)

53.5 prompt/s → ~4.9 h for the full 11,452 × 83 = 950,516 prompts.
**No collapse**: answered values spread 0.0 / 0.5 / 1.0 = 285 / 427 / 618.
NA rate 0.599 — high, but expected for a multi-language 83-criterion bank on a single PR
(the coded bank's `applies()` gate behaved the same way), and it is captured as signal by
the `applied` indicator columns rather than thrown away.

### Anchor battery — PASS (K=50 per tier)

| tier | mean over answered cells | NA (abstention) rate | mean with NA scored 0 |
|---|---|---|---|
| merged (pos) | .7288 | .579 | **.3066** |
| closed-unmerged (neg) | .6060 | .619 | **.2307** |
| word-scrambled | .0242 | **.945** | **.0013** |

All three gates hold: `scram < pos`, `scram < neg`, and `pos > neg`. The scrambled tier is
the informative one — the judge abstains on 94.5% of word-salad cells and scores ~0 on the
rest, so it is reading content and not length. `pos > neg` holding at the anchor level is a
bonus: the bank orders merged above closed-unmerged even before any stack is fit.

### Status / TODO

* Full scoring launched on sk3 GPU 6 (PID 594504, `setsid nohup`, detached), writing
  `abank_rescore/scores_shard{00..15}.npz` + `anchors.npz` + `score_meta.json`.
  Throughput ~16.5 min/shard (~62K prompts each), NA rate stable at .571–.587 per shard.
* Readout pipeline validated end-to-end on the first 6 shards (4,331 / 11,452 rows) — a
  **dry run, numbers not for quoting**. It confirmed: A matrix assembles (83 score + 83
  applied → 151/166 columns survive `clean_cols` on eval), collapse check runs
  (**1 / 83 criteria** collapsed at >98% modal), V builds and joins, dense preds align
  (dense on the 2,200-row eval subset = .6521 vs .6488 on full eval — consistent), and the
  grouped GBM seed spread is produced. Two things visible already and worth watching in
  the final run: `V_exec` is *below* chance out-of-repo (.45–.47), i.e. execution verdict
  category does not transfer across repositories; and the GBM train AUC is .93–.97, so the
  frozen Layer-1 grid is fitting hard — the overfit check in the JSON matters here.
* Scoring finished: 16/16 shards, 11,452 rows × 83 criteria = **950,516 judge calls**,
  ~16.5 min/shard, pooled NA rate **.5813**.
* Note: a NAT64 outage cut the laptop→sk3 path mid-run; the remote job was detached and
  survived untouched.

### Collapse check — 1 / 83 criteria collapsed

Only **a188 "ACSL specification constructs and usage"** collapsed (100% NA — ACSL is an
ANSI-C formal-annotation language that appears nowhere in this corpus; a true corpus fact,
same class as the peer-review "seeds/splits-variability" collapse). Median per-criterion NA
rate .566; **19 / 83 criteria have NA > .90** (mostly language-specific: Rust crate
organisation, C++ idioms, Java import grouping — the judge correctly abstains when the PR is
not in that language). The `applied` indicator columns keep that abstention as signal rather
than discarding it; 155 of 166 A columns survive `clean_cols` on both splits.

### RESULT — V / A / VA vs dense on the same rows

Grouped OOF (GroupKFold by repo) fit **within** each split, Layer-1 protocol
(StandardScaler + LogisticRegression C=1.0, pooled OOF AUC; nonlinear = frozen HistGB grid,
seeds 0/1/2). Dense = v3 seed-42 per-row preds on the identical rows.

| readout | eval (n=5,822 / 112 repos) | test (n=5,630 / 143 repos) |
|---|---|---|
| V_exec (execution, joined) | .5281 | **.4578** |
| V_text (recomputed on v3 text) | .5448 | .5273 |
| **V_all** | .5503 | .4822 |
| **A** (Gemma-4-31B, 83 criteria) | .5896 | .5466 |
| **VA_lin** | .6058 | .5403 |
| **VA_nl** (HistGB, mean of seeds 0/1/2) | **.7422** | **.5806** |
| **dense v3, same rows** | **.6488** | **.7373** |
| Δ_total (dense − V_all) | +.0985 | +.2551 |
| Δ_beyond (dense − best VA) | **−.0934** | **+.1567** |

VA_nl seed spread is tight (eval .7420/.7431/.7415; test .5744/.5778/.5896). GBM train AUC
.94–.95 versus OOF .58–.74 — the frozen grid is fitting hard, as flagged in the dry run.

**The two splits contradict each other at the pooled level**: eval says the articulated
stack *beats* dense (−.093), test says dense wins by a wide margin (+.157). This is the same
eval-vs-test instability that made the v2 dense number untrustworthy pooled (.573 vs .704 =
repo composition). So the pooled row cannot settle the residual question in either
direction, and neither the eval −.093 nor the test +.157 should be quoted alone.

### The adjudicating readout — within-repo

Per the standing lesson on this cell, the trustworthy level is within-repo (it removes the
repo-composition term that swings the pooled numbers). Repos with n ≥ 20 and both classes
present; VA_nl = mean of the three seed OOF vectors; `within_repo_dense_vs_vanl.json`.

| | eval | test |
|---|---|---|
| repos scored (rows) | 71 (5,384) | 73 (4,900) |
| dense, n-weighted | **.7093** | **.6540** |
| VA_nl, n-weighted | .6517 | .6150 |
| dense median / VA_nl median | .7273 / .6857 | .6772 / .6302 |
| **Δ (dense − VA_nl)** | **+.0576** | **+.0390** |
| repos where dense wins | 44 / 71 | 43 / 73 |
| Wilcoxon p | **.006** | **.040** |

**Within repo, dense beats the articulated stack on BOTH splits, significantly, in the same
direction and with a similar magnitude (+.04 to +.06).** The pooled eval anomaly (VA_nl
.742 > dense .649) is composition: the GBM exploits between-repo structure that inflates
pooled AUC but does not survive repo-centering. Both splits agreeing at the honest level,
where they flatly disagreed at the pooled level, is the reason to trust +.04/+.06.

### What carries the signal — the mechanism

Top criteria by univariate |AUC − .5|, and the same criterion tops **both** splits:

| rank | eval | test |
|---|---|---|
| 1 | **a78 change/commit/PR communication quality** .550 | **a78 change/commit/PR communication quality** .552 |
| 2 | **a105 change-description clarity/rationale** .549 | **a123 contribution readiness & submission norms** .547 |
| 3 | a76 robust and consistent error handling .539 | a1 simplicity (KISS/YAGNI) .541 |
| 4 | a30 error prevention / robust error handling .533 | a37 refactoring quality and practice .541 |

a78, a105 and a123 all have **NA rate 0.00** — the judge always found evidence for them,
because that evidence is the title/description/review-comment prose that v3 added and v2
did not have. The single most predictive articulated criterion in the code cell is
*"is this change communicated well?"*, and it was **unmeasurable in principle** under the
v2 bare diff. That is the concrete mechanism behind the enrichment gain, and it is a
finding in its own right: on GitHub, the articulable part of merge preference is
substantially about PR *communication*, not only about code.

### Reading

**The code cell now shows a positive taste residual, and the old "no residual" conclusion
does not survive.** But note carefully what did and did not change. The v2 conclusion
("dense .573 ≈ V+A .592, no residual") was built on an input-starved dense model *and* a
non-judge A instrument whose headline number no longer reproduces. On the enriched input,
with a real Gemma-4-31B judge over 83 portable criteria, and measured at the only level
where the two splits agree, dense sits **+.058 (eval) / +.039 (test)** above the best
articulated stack, both significant. The residual is *small* — of the order of the +.03
bands seen in N&C outcome and peer curation, not the +.11–.17 of peer revealed or N&C
responded — so code belongs in the thin-residual group, not with the big-band cells.

Three things keep this honest and all of them cut toward *understating* the residual:
(1) the judge saw up to 11.6K tokens while dense saw only 2,048, so A was advantaged on
input; (2) A got the enrichment-sensitive criteria that v2 could not even ask about, and
they are exactly the ones that carry signal; (3) V_all is the most generous V available
(execution features plus recomputed text geometry).

Two caveats that cut the other way and bound the claim: dense v3 is **single-seed (42)**,
worth ±.02 on this corpus, so +.039 on test is roughly two seed-sigmas and not comfortable;
and `V_exec` is **below chance out-of-repo on test (.4578)** — execution verdict category
anti-transfers across repositories, so the execution V layer is contributing noise, not
floor, in the grouped design. Neither changes the sign, but the test-side margin is thin.

**Never write this as "+.06 versus V+A .592."** Both the input and the instrument changed;
.592 is itself unreproducible. The defensible sentence is: *on the enriched PR object, with
an LLM-judge articulated bank scored on the same rows, dense exceeds the best articulated
stack by +.058 (eval) / +.039 (test) within repo, p = .006 / .040.*

### Next (not run)

* Dense v3 seeds 1–2 — the single-seed dense is the weakest link in the test-side margin.
* A matched-context arm (judge truncated to the same 2,048 tokens dense saw) would convert
  the "A was advantaged" caveat into a measured quantity.
* The 44 dropped aspects: 16 are non-norm fingerprints that should leave the code bank
  permanently; the other 28 lack catalog text and could be written up as norms if the bank
  is ever refreshed.

### Artifact locations

* sk3 `datasets/code-review/dense_standard_v3/abank_rescore/` —
  `score_code_abank_v3.py` (judge), `readout_code_v3.py` (V build + stacks),
  `build_criteria.py` + `criteria_code_abank.jsonl` (the bank),
  `repro_published_ladder.py` + `.json` (stage-1 level check),
  `scores_shard*.npz`, `anchors.npz`, `score_meta.json`, `score_full.log`.
* Local machine-readable copy: `methods/taste_decomposition/results/code_v3_enriched_layer1.json`.
* Dense v3 per-row preds (comparison target): sk3
  `datasets/code-review/dense_standard_v3/rm_out_seed42/preds_eval.csv`, `preds_test.csv`.
