# Reconstruction-v2 lane — adversarial audit (2026-07-12)

**Auditor stance:** independent, adversarial, verify-by-re-execution. Default to recomputing
numbers from raw artifacts rather than trusting REPORT.md prose. All checks below either
re-ran code, independently re-derived a statistic from raw score/JSON files with a
from-scratch implementation, or did a mechanical file/hash/timestamp audit. Read-only on
everything except this note.

**Verdict key:** VERIFIED (recomputed/re-executed, matches) / DISCREPANT (numbers don't
reproduce) / UNVERIFIABLE (artifact/inputs missing) / DESIGN CONCERN (works as coded but
overstates the claim). Multiple verdicts may apply to one claim.

---

## Claim 1 — Additivity ("implemented without altering historical programs")

**Verdict: VERIFIED.**

The repo has no prior git history for `methods/metric_seam/` at all (this whole tree is
either `??` untracked or `A`/`AM` newly-staged in `git status`; `git log --follow` on
`contract_check.py`, `agentic_run.py`, `battery_common.py`, `ops_capability.py` returns
nothing beyond the current staged blob for the two that are staged). So "git diff against a
frozen baseline" is not directly available — I substituted a timestamp + hash audit:

- `find methods/metric_seam -type f -newermt "2026-07-12 12:45:00"` (the reconstruction-v2
  lane's first file, `blind_reconstruction_v2.py`, appears at 12:50) returns **only** files
  under new v2-suffixed names/dirs: `reconstruction_v2.py`, `battery/*_v2.py`,
  `battery/_blind_worker_v2.py`, `battery/_sealed_worker_v2.py`, `battery/split_ops_v2.py`,
  `hybrids/ops_capability_v2.py`, `science_claims_v2/`, `technical_replay/`,
  `environment_v2.py`, `RECONSTRUCTION_V2.md`, plus their test files. Zero pre-existing
  filenames appear in that list.
- mtimes of the candidate "frozen" files: `contract_check.py` 07-10 10:49, `agentic_run.py`
  07-10 10:49, `battery_common.py` 07-08 12:58, `ops_capability.py` 07-11 12:22,
  `certificates.py` 07-10 10:51 — all strictly before the 12:45 boundary.
- Every `hybrids/programs_{cw,math,humor,peer,ssdis,v2,legal}/*_h0.py` file's mtime is
  2026-07-02 through 2026-07-08 — weeks/days before the v2 lane, none touched by it.
- `git diff` (working tree vs. index) on the two files that ARE staged
  (`agentic_run.py`, `battery_common.py`) and on `certificates.py` shows only the **prior**
  "harness v2" changes dated 2026-07-10 in their own docstrings/comments (train-only
  execution enforcement, `skip_undefined` bootstrap opt-in) — pre-existing, uncommitted work
  from *before* the reconstruction-v2 lane, not new v2-lane edits.
- **Contract freeze integrity:** recomputed sha1[:12] for all 125 hashes recorded across
  every `_*_domain_freeze` / `_*_straggler_freeze` block in `contracts_v3_validation.json`
  against the actual files in `contracts_v3/` — **0 mismatches, 0 missing**.
- **Import graph:** none of the seven new top-level v2 modules
  (`reconstruction_v2.py`, `blind_reconstruction_v2.py`, `evaluate_blind_v2.py`,
  `contract_check_isomorphic.py`, `dag_schema_enforced.py`, `certify_batch_v2.py`,
  `ops_capability_v2.py`) import `contract_check`, `agentic_run`, `battery_common`, or
  `dag_schema`. `ops_capability_v2.py` does `import ops_capability as v1` — read-only reuse.
  `split_ops_v2.py` explicitly comments "v1 remains frozen for historical replay; new blind
  runs use ... reconstruction v2" and only historical `programs_*/*_h0.py` files import
  `ops_capability` (grepped — none import `ops_capability_v2`).
- `contract_check_isomorphic.py`'s own docstring states it "intentionally does *not*
  replace `contract_check.py`."

One pre-existing, **unrelated** finding surfaced during this check: a `PROVENANCE_INCIDENT_
2026-07-12.json` (self-authored contract overwrite of `contracts/creative_writing__a54.json`,
with a rebuilt sha1 ledger) already existed before the v2 lane started. Re-hashing the 12
files in that ledger's `current_sha1` block against the live `contracts/` directory today
shows **all 12 no longer match** the incident's ledger — but every changed file's mtime
(07-11 23:53 through 07-12 10:10) is *before* the v2 lane boundary (12:45), so this is
further legitimate historical-program churn (the incident's own "pending user decision" to
re-author a54, apparently resolved), not a v2-lane mutation. Flagging it here only because it
technically fails a naive sha-diff against that ledger; it is **not** attributable to the v2
lane.

Also observed: one **new, concurrent, non-v2** census cell (`legal_ss_disability__a3`)
appears in the historical `census/` tree with files modified after 12:45 — this is the
original program's own agents continuing their separate work in parallel, correctly
additive within the existing structure (new cell directory, `PROMOTION_QUEUE.json` updated,
a `.bak_a3census` backup made before overwriting `field_results.jsonl`). Not a v2 lane
action; noted only to explain why some non-v2 files also show post-boundary mtimes.

**Conclusion:** no evidence of any frozen historical file being mutated by the
reconstruction-v2 lane. All new code lives in new files/dirs, imports old modules read-only,
and the contract freeze ledger verifies bit-exact.

---

## Claim 2 — Blind Math a144 (the central result)

**Verdict: VERIFIED**, and this is the most thoroughly load-bearing result in the lane — every
headline number reproduces from an independent, from-scratch recomputation.

**Blindness protocol.** `compiler_bundle.json` (what the clean-room compiler actually saw)
was read directly: `item_keys` are opaque aliases (`"train_0001"`, ...), not real datapoint
ids; `interface.judge_values_available: false`, `heldout_identifiers_available: false`,
`residuals_available: false`; `representation: "ctext only"`; `allowed.fields: {}` (no LLM
field access this run). `prepare_manifest.json` self-declares
`"loads_heldout_items_into_compiler_bundle": false, "loads_judge_data": false"`. Grepped all
150 `train_items` for any real `dNNNNN`-pattern id leaking through a field value — none
found. Partition: `sorted datapoint_id; random.Random(seed=7).shuffle; first 150 of 250 train`
— same rng(7) convention used elsewhere in this program (e.g. patents WS3 split).

**Sealed evaluator (also answers Claim 7).** `sealed_manifest.json`'s
`evaluation_order` block timestamps `candidate_execution_completed_at` =
`20:10:45.100823Z`, then `llm_reference_load_started_at` = `20:10:45.410508Z` — i.e. the
frozen candidate ran to completion ~0.3s *before* the LLM reference file was even opened;
this is a machine-recorded fact, not a narrated claim. `blind_reconstruction_v2.py`'s
`_audit_candidate_text()` statically AST-walks the candidate for imports/calls: allowlist is
`{__future__, re, math, statistics, collections, itertools, functools, decimal, fractions,
typing, datetime, calendar}`; forbidden calls `{open, exec, eval, compile, __import__, input,
breakpoint}`. Execution runs in a fresh `subprocess` with `-I` (isolated) flag, temp `HOME`,
minimal env. The manifest **honestly discloses** `"os_security_boundary": false` — this is a
static/allowlist guard, not a sandbox, and is documented as such in the code comment ("a
reproducibility and accidental-leak guard, not an OS security boundary"). For this specific
run it's moot: the actual frozen candidate (read in full) is a pure `re`-only regex/heuristic
scorer with signature `score(text, extracted, ops)` and no `os`/`sys`/file-adjacent imports at
all — structurally incapable of reading files regardless of the sandbox gap. **Retrieval
scope** (relevant to external-review finding 5, see Claim 6e below): `split_ops_v2.py`'s
`SplitScopedOps`/`_TrainRetriever` builds any TF-IDF index from `train_ctext` only (the
bundle's 150 train items), and `for_item()` raises `KeyError` for any id outside that set —
a held-out item cannot even be queried, unlike the flagged v1 ops path.
Ran `python3 methods/metric_seam/battery/blind_reconstruction_v2.py verify --bundle
.../compiler_bundle.json` myself → `{"verified": true, ...}` (all pinned sha256 hashes over
implementation/environment/inputs still match live files).

**Recomputed numbers** (independent from-scratch Python, not the repo's own spearman
function), reading `candidate_scores.json` / `llm_reference_scores.json` /
`historical_comparisons.json` directly:

| Quantity | Claimed | Recomputed (mine) | Match |
|---|---:|---:|:---:|
| Candidate rho vs frozen reference (52 common items) | 0.06599 | 0.06598665482101687 | exact |
| Candidate MAE | 0.36615 | 0.36615384615384616 | exact |
| Historical h0 rho on same 52 items | 0.4832 | 0.48321550560293935 | exact |
| Common-support id list (52) | — | intersection of the 100-item candidate map and the 52-item reference map | identical set, byte-for-byte |
| h0 program sha256 | recorded in artifact | recomputed from live `programs_math/a144_h0.py` | matches — confirms genuine re-execution of the *actual* frozen h0 file, not a copied number |
| h0 cached-field-source sha256 | recorded | recomputed from live `field_results.jsonl` | matches |
| Pure-code baseline columns (v0/v1/v2) | .0106 / .0305 / .0945 | .01061237175992545 / .03047338608440523 / .09448708066223244 | exact |

**Adversary recount.** Manually re-tabulated the 26-row pair table in `adversary_001/
REPORT.md` by hand, counting the "Ordering" ✓ column and both P/N "expected range" ✓ marks
independently: **14/26 ordering** and **33/52 range** — both reproduce the claimed counts
exactly. Also independently re-hashed all 7 files pinned across `FROZEN_SHA256.txt` and
`POSTRUN_SHA256.txt` (contract, candidate, `RECONSTRUCTION_V2.md`, `FROZEN_SUITE.json`,
`run_once.py`, `RESULTS.json`, `REPORT.md`) with my own `hashlib.sha256` calls — **all 7
match exactly**, confirming the "5/5 pins match" (pre-run) claim and extending it to the
post-run artifact hashes too.

**Off-label channel challenge.** `channel_challenge.json` gives exactly +0.230, +0.510,
−0.050, +0.240 for the four probes (3/4 clear the 0.05 margin), `is_contract_pass: false`,
`frozen_channel_labels_changed: false` — matches the progress note's characterization and is
explicitly, correctly labeled as not a contract pass or relabeling.

**Outcome logic.** `reconstruction_v2.py`'s `classify()` was imported and **re-run live**
against a `ReconstructionEvidence` object built directly from `reconstruction_record.json`'s
own field values: reproduces `outcome == "proxy_mismatch"` and an identical
`claim_permissions` dict, byte-for-byte, including `may_claim_tacitness: False`. The rule that
fires is unconditional: `if fidelity is Status.FAIL: return Outcome.PROXY_MISMATCH` — checked
*before* the `verified_reference_disagreement` / `reference_divergence` branches, and the
dataclass's `__post_init__` independently forbids constructing a
`verified_reference_disagreement=True` record when `construct_fidelity` isn't PASS. Notably,
`Outcome` as an enum has **no tacitness member at all** — the classifier is structurally
incapable of emitting a tacitness-flavored outcome, not merely configured not to.

**Is `proxy_mismatch` the right call?** Yes, on the evidence: the code failed the isomorphism
metric (.066 vs h0's .483, both far below any plausible bar) **and** failed the independent
construct-adversary on substantively diagnostic grounds — a false witness (`x=1` for `x²=2`)
scored identically to the true witness (0.695 both), a non-surjective map scored within 0.08
of a valid one, reversed-quantifier and shadowed-name cases were mis-ordered, and pure
keyword-salad scored 0.635. That is a textbook description of a program tracking correlated
surface proxies (existence/causal-language density) rather than the actual mathematical
relation, which is precisely what `proxy_mismatch` is defined to mean — not tacitness (search
was bounded, not exhaustive) and not reference-divergence/constructive-extension (would
require the code to be *right* where the LLM disagrees, which was never shown). The design
choice to let `construct_fidelity` FAIL pre-empt any correlation-based outcome — "a
construct-invalid proxy cannot be rescued by reference agreement" — is a real guard against
exactly the failure mode (spurious rho) this kind of experiment is most at risk of, and it is
enforced in code, not just asserted in prose.

**Test suite re-run** (from repo root):
```
python3 -m unittest methods.metric_seam.test_reconstruction_v2 \
  methods.metric_seam.battery.test_blind_reconstruction_v2 \
  methods.metric_seam.battery.test_evaluate_blind_v2 \
  methods.metric_seam.battery.test_contract_check_isomorphic \
  methods.metric_seam.battery.test_dag_schema_enforced \
  methods.metric_seam.battery.test_certify_batch_v2 \
  methods.metric_seam.hybrids.test_ops_capability_v2 \
  methods.metric_seam.technical_replay.test_replay
```
→ **`Ran 66 tests in 23.580s — OK`** (I ran this myself; all green).

---

## Claim 3 — Technical replay (65 certificates / 800 runs)

**Verdict: VERIFIED** (arithmetic, duplicate-row disclosure, classification logic) +
**DESIGN CONCERN** (framing of "real code execution").

Delegated to a subagent that traced the data lineage and re-executed the classifier. Summary
of its independently-checked findings:

- The 800 rows trace to `datasets/code-review/pr_test_execution/outputs/
  transplant_consolidated_2026_07_12_canonical.csv` (sha256-verified against
  `code_execution_probe.json`), itself built from many `batch_runs/<repo>/
  transplant_*_verdicts.jsonl` files containing genuine execution telemetry: real pytest
  stdout/stderr tails ("2 failed, 8 passed, 3 warnings"), named Go test IDs, docker return
  codes, a `rc=124` wall-clock timeout at 1200.006s, and a real git error
  ("fatal: reference is not a tree"). This is authentic prior execution output, not
  fabricated for the 2026-07-12 report.
- Re-ran `code_execution_probe.py --check` against the live CSV: output byte-identical to
  the persisted JSON — 29 pinned / 10 partial-pinned / 26 vacuous / 571 indeterminate / 145
  none / 19 other errors, `n_rows=800`, `n_unique_row_ids=799`. Arithmetic checks:
  29+10+26=65=8.125% of 800 ✓; 65+571+145+19=800 ✓.
- `chia-blockchain:496` duplicate confirmed as two distinct raw JSONL records (one
  `error_no_output`, one `indeterminate`) — neither is a certificate row, so the duplicate
  does not inflate the 65 count; disclosure language in REPORT.md is accurate.
- Hand-checked 15 rows (12 certificate + 3 non-certificate) against the raw CSV fields and
  the classification rule in `datasets/code-review/pr_test_execution/runners/
  run_transplant.py:190` (`aggregate_pr_label()`) — all matched.
- **Design concern:** `code_execution_probe.py` itself performs no subprocess/docker/pytest
  invocation — it only reads the CSV's pre-existing `transplant_pr_label` field and runs
  consistency checks. The actual execution happened earlier (frozen artifact,
  `discovery_mode: "replay"` — correctly disclosed in the JSON metadata). REPORT.md's prose
  ("actual repository checkout, test transplant, and execution in era-matched containers")
  is accurate about what *produced* the data but risks being read as live execution
  performed *as part of* the 2026-07-12 replay run rather than classification of a frozen
  prior artifact. The structured metadata is honest; the prose is a plausible
  misreading risk, not a fabrication.

---

## Claim 4 — Science claims (171 strong / 431 weak / 2,400 papers)

**Verdict: VERIFIED (all headline counts) + DISCREPANT/DESIGN CONCERN (numeric-match
integrity of the "strong" bucket).**

Delegated to a subagent that re-ran the pipeline fresh and manually spot-checked 10 "strong"
certificates against source text. Findings:

- **Counts reproduce bit-exact** from a fresh, independent re-run of
  `methods.metric_seam.science_claims_v2.evaluate` and
  `methods.metric_seam.technical_replay.fullpaper_probe` against the live source JSONLs
  (sha256-matched to what's frozen in REPORT.md): 2,400 papers / 1,957 nonempty bodies / 660
  eligible / 429 positive recurrence / 171 strong across 158 papers / 431 evidence_link /
  1,370 insufficient / 519 abstain — all identical to the committed `results.json`.
  `pytest test_science_claims_v2.py` → 14/14 pass. "Never reads `y`" independently confirmed
  by source read: `load_unlabelled()` allowlists only `paper_id/abstract/body`, `y` is never
  referenced in `core.py`/`evaluate.py`.
- **Spot-check of 10 "strong" certificates: 5 genuine, 5 spurious.** Genuine examples
  (e.g. "91.58% AUC", "25 billion sentence pairs", "reproduce 20 algorithms") are precise,
  well-matched claims. The 5 spurious cases share a root cause: a **reproducible extraction
  bug** in `_QUANTITY_RE` (`science_claims_v2/core.py` lines 99-105) that silently truncates
  any multi-digit-or-decimal number followed by an unrecognized unit letter —
  independently confirmed live: `"100k"→"10"`, `"33B"→"3"`, `"6.7B"→"6"`, `"1.5B"→"1"`,
  `"30nm"→"3"`. Combined with an index-word filter that excludes `table/figure/equation`
  prefixes but not `stage/step/phase`, this produces matches like "P³EFT"'s superscript "3"
  matching an unrelated "3 following measures," or "Stage 1" matching an unrelated "Table
  1...Stage 1" caption. Neither the strong nor weak bucket has an entity-binding check for
  purely numeric claims (unlike comparative claims, which do check entity terms) — e.g. a
  claim of "28 adapters" was certified against evidence text about "28 tasks."
- Sweep: of the 134 "strong numeric" certificates, 64 (48%) rest solely on small bare
  unitless integers; the subagent's spot-check rate (5/10 spurious) extrapolates to roughly
  a quarter to a half of the 134 numeric certificates being contaminated by this bug or
  related index-marker leakage.
- The "all 140 strong quantity-bearing certificates reproduce every normalized obligation"
  framing is **tautological**, not an independent check: `_evaluate_edge` requires
  `matches == len(quantities)` as the precondition for the "supported"/strong label in the
  first place, so 140/140 restates the decision rule rather than validating it.
- No LLM in this pipeline (mode-collapse concern inapplicable, confirmed by grep).
  Absence-gating is correctly implemented: missing body/abstract/claim/evidence all route to
  explicit `abstain`, never silently become a certificate. Score distributions are
  non-degenerate.

**Net assessment:** the pipeline's architecture (no-`y`, explicit abstention, separate strong
vs. weak buckets) is sound and the reported counts are real and reproducible. But the
"strong" label is weaker evidence of genuine claim/evidence correspondence than the report
implies — a meaningful fraction of the 171 strong certificates are numeric-coincidence
artifacts of a specific, fixable regex bug rather than genuine abstract-to-body claim
verification. This is the most significant numerical/methodological problem found in this
audit.

---

## Claim 5 — Patents rho=.745 vs .084 (WS3 reuse)

**Verdict: VERIFIED — faithful, direct reuse, caveat carried forward correctly.**

`methods/metric_seam/technical_replay/initial_manifest.json`'s `patents_a34_oracle_
prior_art_replay` case reads its numbers directly from
`outputs/metric_seam_pilot/tasks/patents_pa/ws3_eval_report.json` (path is in the manifest,
not a hardcoded literal). Opened that file directly: `a34.evidence = {rho_full: 0.745,
rho_null: 0.084, op_marginal: 0.661, n_test: 100, ...}` — this **is** the canonical WS3
artifact (`datasets/patents/2026-07-10__evidence_aware_judge_ws3.md`'s own results table
shows the identical numbers: a34 novelty bars, evidence arm, rho full .745 / rho null .084 /
op-marginal +.661, n=100 held-out). Since the technical-replay case reads this exact file,
"same split rng(7)/40%" and "same judge intersections" are automatically satisfied — it's
not a re-derivation from raw judgments that could silently drift, it's the same artifact.

Caveat carry-forward: the WS3 note's "Stronger caveat — oracle gold injection" section
states plainly that "the judge needs the evidence" stands but "the retrieval machinery
discovers the evidence" is **NOT established**, because `pa_features.json` force-includes
the examiner-cited gold document. The progress note's §3.4 states: "This is strong
selected-pipeline utility and isomorphic reconstruction conditional on privileged evidence;
it is not autonomous prior-art retrieval or pure-code verification" — and the
`technical_replay_v2/REPORT.md` case entry explicitly lists as a corpus limit: "Candidate
prior-art evidence is examiner/privileged injected and must not be interpreted as autonomous
prior-art discovery." Both carry the caveat faithfully; neither drops it.

---

## Claim 6 — New infrastructure

### 6a — channel-faithful contract checker

**Verdict: VERIFIED** (confirmed independently by me, in addition to a subagent covering the
same ground in more depth — see its findings folded in below once returned).

Read `contract_check_isomorphic.py` directly. `load_frozen_extractions()` (lines 282-348)
answers the "how does an LLM field get populated" question precisely: it consumes a
**pre-computed extraction artifact**, not a live LLM call and not a simulated value — every
row is checked against `text_sha256(probes[index]["text_pos"/"text_neg"])` (binds the
extraction to the *exact* probe text) and the payload's `contract_sha256` must match the
contract under test. This directly matches the module's own docstring claim ("L probes test
prompt-based articulability only when a frozen extraction artifact is supplied and
cryptographically bound to this exact contract and probe text"). `_build_gate()`
(lines 440-506) confirms the abstention claim in code: when declared L-coverage falls below
the required threshold, `status = "ABSTAIN"` (not `"FAIL"`) — read directly, not inferred.
CODE and HYBRID gates are separate `GateResult` objects. The module's own docstring states
explicitly it "intentionally does *not* replace `contract_check.py`" (consistent with the
additivity finding in Claim 1). Checked the CLI entry point (`main()`, lines 716-772) for
accidental-mutation risk: the only file-write path is `args.json_out.write_text(...)`, gated
on an optional `--json-out` argument with **no default** — running the checker against a real
frozen contract writes nothing anywhere unless a caller explicitly names an output path, so
it cannot silently overwrite `contracts_v3_validation.json` even by accident.

**Subagent corroboration adds two findings:** ran the checker live against 3 real frozen
contracts (`a117`, `a126`, `a135` h0 programs) with no extraction artifact supplied:
L-channel probes correctly report `ABSTAIN, "no frozen extraction row"`,
`hybrid_gate.status="ABSTAIN"` while `code_gate` is evaluated independently;
`contracts_v3_validation.json`'s mtime/content were byte-identical before and after. It also
clarifies that `contracts_v3_validation.json` is a contract-*lint* ledger (schema/leakage
`errors/warnings/pass`), not a per-candidate CODE/HYBRID verdict store — not even the same
estimand as this checker's output, so there is no side-by-side "verdict diff" to compute.
**Gap**: a repo-wide grep found **zero** files anywhere that produce a
`metric-seam-probe-extractions-v1` artifact. The HYBRID/L gate is fully implemented and
correctly abstains when data is absent, but as of this audit it has **never been exercised
on real L-field data** — only the CODE gate has actually run against real contracts. The
instrument is real and correctly designed, but the L-channel half of finding 1 remains
unexercised, not merely "fixed."

### 6c — corrected capability library

**Verdict: VERIFIED (the 7/7 counterexample replay itself) + DISCREPANT (defect-coverage
mapping against the four named historical defects, and against the progress note's own
"seven counterexamples" description).**

I directly imported both `ops_capability` (v1, frozen) and `ops_capability_v2` and called two
of the four named historical defects live, myself:

- **`date_chain()` missing-year defect**: `v1.date_chain("...April 2, 2000... a follow-up on
  April 5.")` returns `April 5 → 2026-04-05` (defaults the missing year to *today's* year,
  producing a 9,499-day gap from a 3-day-apart pair of dates in the text).
  `v2.date_chain(...)` on the identical input returns `April 5 → 2000-04-05` (infers the year
  from surrounding context), a 3-day gap. Clean, reproduced-live confirmation of the
  "missing-year→today default" defect and its fix.
- **`deadline_satisfied()` negative-gap defect**: calling both versions with the same
  `(event_date=2020-01-10, filing_date=2020-01-01, days=30)` triple: `v1.deadline_satisfied
  (...) = True`, `v2.deadline_satisfied(...) = False` — a clean behavioral flip on identical
  input, matching the stored counterexample-replay artifact's `date.negative_gap_rejected`
  case (`frozen_v1.actual=True/pass=False`, `corrected_v2.actual=False/pass=True`) exactly.
- Verified `outputs/metric_seam_pilot/reconstruction_v2/capability_counterexamples.json`'s
  own `frozen_v1_sha256` and `corrected_v2_sha256` fields against the live files' sha256
  (`hashlib.sha256` on both) — **both match exactly**, confirming the stored 0/7→7/7 replay
  reflects the actual current code, not a stale snapshot.

**The subagent's independent re-execution of all 7 counterexamples reproduced 0/7 v1 → 7/7
v2 exactly** (full input/output table cross-checked against the stored replay, matching
row-for-row), and 8/8 unit tests pass. But it then cross-mapped the 7 counterexamples against
the four *named* historical defects from `notes/2026-07-10__seam-agentic-program-runbook.md`
and found real gaps that change the overall verdict:

| Named historical defect | Fixed? | Covered by one of the 7 counterexamples? |
|---|:---:|:---:|
| `deadline_satisfied()` negative-gap acceptance | **yes** | yes (`date.negative_gap_rejected`) |
| `date_chain()` missing-year→today default | **yes** | yes (`date.missing_year_frozen_epoch`) |
| `date_chain()` silent April-31 drop | **no** | no — direct test: v1 *and* v2* both silently swallow an invalid "April 31" via an identical bare `except: continue`; the runbook itself already logs this as a still-open item |
| `attributions()` conjunct-verb/action-beat blindness | **no** | no — `v2.attributions()` calls `v1.attributions(text)` unchanged and only post-processes ownership; the runbook logs both sub-defects as still-open "E2L-v2 wishlist" items |
| `is_refrain()` ≥3-word floor | **no** | no — `v2.is_refrain()` calls `v1.is_refrain(text)` unchanged; a direct test with a genuine 1-word refrain returns `[]` from both versions. (v2's `is_refrain`-adjacent change fixes a *different* bug — adjacent-repetition vs. craft — not the word-count floor.) |

Net: **only 1.5 of the 4 originally-named defects are both fixed and represented among the
"7 audited counterexamples"**; the other 2.5 remain open despite `ops_capability_v2.py`
containing functions with the matching names (which silently delegate to the frozen v1
behavior for those specific paths). The progress note's own prose list of what the 7
counterexamples cover ("date epoch and signed gaps, percentage direction, p-value bounds,
**refrain adjacency**, ambiguous offsets, and attribution ambiguity") also does not cleanly
match the actual 7 case IDs in `capability_counterexamples.json` — there is no counterexample
that actually exercises `is_refrain()`; the closest-sounding entry
(`attribution.repeated_span_abstains`) tests attribution-over-a-repeated-span, not refrain
detection. So "7/7" is a real, reproducible number for the specific 7 cases chosen, but the
narrative framing of *what* those 7 cases demonstrate is broader than what they actually
cover, and two of the four defects that motivated this whole instrument are unaddressed.

Provenance was independently reconfirmed by both me and the subagent: `ops_capability.py`
sha256 matches `frozen_v1_sha256` in the counterexample file (untouched); `ops_capability_v2`
imports v1 read-only; a repo-wide grep of `hybrids/programs_*/*.py` found **zero** files
importing either `ops_capability` module directly (historical h0/h1 programs don't reference
the library by that path at all) — only the new `split_ops_v2.py`/`_sealed_worker_v2.py`
wire `ops_capability_v2` into the new blind-run "capability" op-view, with an explicit
comment that v1 stays frozen for historical replay. So additivity for this file is intact;
it's the *scope* of the correction (6c's headline claim), not the isolation of it, that is
overstated.

### 6b — enforced typed DAG

**Verdict: VERIFIED (core mechanism) + DESIGN CONCERN (provenance can be defeated by an
adversarially-written node).**

Delegated to a subagent that constructed and ran real adversarial 3-node test DAGs against
`dag_schema_enforced.py` (comparing to unmodified `dag_schema.py` v1). Findings:

- Each node's `fn` receives a `_RestrictedInputs` wrapper built only from its *declared*
  input mapping; any undeclared key access raises `UndeclaredInputError`, and `execute()`
  unconditionally calls `assert_valid()` first (not a skippable separate step). Reproduced
  the exact v1 bug pattern (`ctx['other_node']` ambient read) and confirmed it is rejected
  at runtime, both via the existing test suite and the subagent's own constructed case.
- **A genuine gap the subagent found by writing its own adversarial node**: a node whose
  `fn` never touches its restricted `inputs` argument at all, and instead reads an
  LLM-derived value stashed in a plain Python **module-level global dict** by an upstream
  node, **executes successfully** — and worse, the executor's own seam/taint bookkeeping
  reports it as pure `code` provenance (`taints == ['code']`) even though the value it
  returned actually originated from the LLM channel. This is disclosed in the module's own
  docstring ("not an OS security sandbox... a hostile Python closure can still read
  globals"), so it is not a hidden defect, but it means the enforcement protects against
  *careless* undeclared reads (the historical bug pattern), not against an *adversarially
  written* node — and when defeated, it silently mislabels provenance rather than failing
  loudly.
- Disconnected-node test: a node present in code but not wired to any edge causes the whole
  program to refuse to run at all (`DagValidationError: nodes do not reach out: [...]`) —
  stricter than "excluded," confirmed by direct execution.

### 6d — immutable permutation/bootstrap/FDR certification

**Verdict: VERIFIED.**

- **BH correctness**: the subagent wrote an independent reference BH implementation plus a
  separate literal largest-k rejection-set implementation, and ran both against
  `certify_batch_v2.benjamini_hochberg` on 5 synthetic p-value sets (mixed, all-p=1,
  all-tiny, singleton, ties) — all three agreed exactly, including the reject-set at
  α=0.10.
- **Family definition** (gap, not fatal): `certify_batch()` treats "family" as
  *everything in one manifest file* — there's no code-level enforcement that a manifest
  corresponds to exactly one domain/batch. This matches the frozen rule only by convention,
  not by structural guarantee.
- **Immutability is real, not aspirational**: SHA-256-pinned artifact refs, a TOCTOU
  re-hash immediately before writing, exclusive-create output with `chmod 0o444`. The
  subagent ran `certify_batch` twice on identical inputs → byte-identical report hashes;
  then mutated one candidate score by `1e-9` without updating the manifest hash →
  `IntegrityError: SHA-256 mismatch`, no output file written. This is enforced, tested
  behavior, not a naming convention.
- Permutation test is one-sided by design (matches the directional `delta>=0` gate) —
  correctly documented, not a bug.
- Separate denominators (reference availability over full held-out vs. candidate coverage
  conditional on reference) were confirmed with a constructed case with mismatched
  reference/candidate/held-out sizes (12/6/24): three distinct numbers reported
  (`reference_availability_over_heldout=0.5`, `candidate_coverage_over_heldout=0.25`,
  `paired_coverage=0.5`), not conflated into one.

### 6e — provenance-aware claim permissions / negative-result guard

**Verdict: VERIFIED behavior, but DESIGN CONCERN on mechanism.**

I independently confirmed the outcome-classification half of this myself (see Claim 2
above): `Outcome` enum has no tacitness member at all; `classify()` gates on
`construct_fidelity` FAIL before any correlation-based branch (enforced twice: dataclass
invariant + branch order); re-ran `classify()`/`claim_permissions()` live against the real
a144 record and reproduced the recorded output exactly, byte for byte.

A subagent independently ran its own adversarial case: an evidence record with
articulability/verifiability/isomorphism all FAIL, construct_fidelity PASS, plus a
`provenance_note` explicitly containing "requires human judgment / tacit expertise" text
designed to see if the rationale leaks into a claim-shaped output. Result: `outcome =
"unresolved"`, every `may_claim_*` False, and the smuggled rationale text does not appear
anywhere in the returned permission dict (checked programmatically). The retrieval-scope
half of the same external-review finding (finding 5, ops indexing raw/full-corpus text
including test split) is fixed for the v2 blind path by `split_ops_v2.py`'s TRAIN-only
`_TrainRetriever`, independently confirmed by me by direct source read (see Claim 2).

**Design concern, confirmed independently by both me and the subagent**: `may_claim_
tacitness` is a **hardcoded `False` literal** in `claim_permissions()` — not computed from
any branch of the evidence or outcome, unlike every other field in that function, which is a
live boolean expression. This is honestly disclosed (the progress note itself says "`may_
claim_tacitness` is always false"), and a repo-wide grep confirms no code path anywhere ever
sets it `True`. As a *permission system* it therefore does zero discriminating work on this
one field — it is a standing declaration ("we never claim this"), not an enforced decision
procedure with a live dangerous branch that could in principle fire and is correctly gated
off. That is a legitimate design choice (never allow the claim, full stop) but it should not
be described as the system "refusing" tacitness claims case-by-case; it never considers
granting one.

---

## Claim 7 — Sealed evaluator

**Verdict: VERIFIED**, folded into Claim 2 above (evaluation-order timestamps, AST
import/call allowlist, subprocess isolation, honest `os_security_boundary: false`
disclosure, and `split_ops_v2.py`'s TRAIN-only retrieval scoping). No file-read log exists
to trace beyond the sha256-pinned manifest, but the manifest's own pinning (re-verified live
via the `verify` CLI command) plus the candidate's actual (file-I/O-incapable) source code
make the blindness claim solid for this specific run.

---

## Claim 8 — Handoff note diff

**Verdict: UNVERIFIABLE (no true diff possible) but passes internal-consistency check.**

`notes/2026-07-12__metric-seam-verification-handoff.md` is untracked in git (`??`, no
history), and no `.bak`/prior-session copy exists anywhere in the repo or `.claude/` — so a
literal diff against the "original intent" is not recoverable. Internal-consistency check
performed instead:

- §7 ("Additive reconstruction-v2 lane (post-review)") is clearly the addition — it cites
  the progress-note file by exact name and numbers (a144 rho=.066, sealed adversary result)
  that only exist as of the v2 lane's later timestamps.
- All 6 caveats in §4 (600-char legal truncation, frozen-h0 bugs, v1-era contracts, legal
  a13 58% coverage, legal corpus contamination, patents oracle-gold injection) remain intact
  and are corroborated by independent artifacts I read directly (the runbook's own EXTERNAL
  REVIEW TRIAGE entry, the WS3 note's caveat section) — none read as watered down relative to
  their independent corroborating sources.
- §5's "suggested verification targets" list is unchanged in scope and was in fact the basis
  for several of the checks in this audit (freeze integrity, train-only discipline).
- §7's closing directive ("Do not promote the blind a144 candidate... Do not use its opened
  held-out split for a second confirmatory a144 build") matches the progress note's own
  "Next confirmatory moves" §5 verbatim in intent — no inconsistency between the two docs.

No internal contradiction found; the addition reads as a faithful, appropriately hedged
appendage rather than a retroactive softening of the original caveats.

---

## Top-4 most consequential findings

1. **Claim 2 (blind math a144) is the strongest-verified result in the lane.** Every
   headline number — candidate rho .066, h0 rho .483, three pure-code baselines, 14/26 and
   33/52 adversary counts, the 3/4 off-label diagnostic deltas, and the `proxy_mismatch`
   outcome itself — reproduces exactly under independent from-scratch recomputation, and the
   sealing/blindness machinery is real (timestamped execution-before-reference-load,
   AST-restricted candidate, TRAIN-only retrieval scope, live `verify` command passing).
   This is not a narrated negative result; it is a mechanically checkable one.
2. **The "corrected capability library" fixes 1.5 of the 4 defects it was built to fix.**
   Independent re-execution of all 7 stored counterexamples reproduces the claimed 0/7→7/7
   flip exactly, and two of the four *named* historical defects (`deadline_satisfied`
   negative-gap, `date_chain` missing-year) are genuinely fixed and covered. But the other two
   named defects (`attributions()` conjunct-verb/action-beat blindness, `is_refrain()`'s
   ≥3-word floor) are **not** fixed — `ops_capability_v2.py` has functions with those exact
   names that silently delegate to the unchanged v1 implementation for those specific bugs —
   and neither is represented among the 7 counterexamples despite the progress note's prose
   listing "refrain adjacency" as one of the seven. The 7/7 number is real; the implied
   coverage of the original defect list is not.
3. **The science-claims "strong" bucket is meaningfully contaminated by a numeric-extraction
   bug**, found by direct execution (`"100k"→"10"`, `"33B"→"3"`, etc. in `_QUANTITY_RE`) and
   confirmed by manual spot-check (5/10 sampled "strong" certificates were spurious digit
   collisions, not genuine claim/evidence matches). The 2,400/1,957/660/429/171/431 headline
   counts are all bit-exact reproducible, but "171 strong certificates" overstates how much of
   that bucket is genuine abstract-to-body verification versus coincidental short-integer
   matches.
4. **The three "enforcement"/"correction" instruments this audit could stress-test by
   construction (DAG typing, tacitness permission, capability library) all protect against
   carelessness/omission rather than adversarial or complete correction, and mostly disclose
   this honestly rather than hide it.** A constructed adversarial DAG node can smuggle an
   LLM-derived value through a module-level global and have the executor *silently mislabel*
   its provenance as pure code (`dag_schema_enforced.py` — confirmed by live execution).
   `may_claim_tacitness` is a hardcoded `False` literal with no live branch that could ever
   evaluate `True` — a standing declaration, not a decision procedure. The capability-library
   gap above is the one place where the *scope* of what was fixed is actually overstated in
   prose rather than merely narrower than the word "enforced"/"corrected" implies — everywhere
   else, "enforced" turned out to mean "enforced against the originally-diagnosed failure
   mode," which is a fair reading of the word; here it means "corrected against roughly half
   of the originally-diagnosed defects."

## Does v2 close external-review findings 1/3/4?

- **Finding 1** (contract harness certifies code path only; new LLM fields uncertified):
  **substantially addressed.** `contract_check_isomorphic.py` reads a cryptographically
  bound, pre-computed L-field extraction artifact (sha256-keyed to the exact probe text and
  contract) rather than a live LLM call or a simulated value — confirmed by direct code read
  (`load_frozen_extractions`). CODE and HYBRID gates are reported separately; missing L
  coverage below the required threshold produces `status="ABSTAIN"` (verified in
  `_build_gate`), not a failed CODE probe, matching the claimed fix exactly. It does not
  retroactively touch the historical `contract_check.py` ledger — addressed as a genuinely
  new, additive instrument. It does not retroactively re-certify the a207-style historical
  "promoted" labels that motivated the finding, nor does it claim to; producing the actual
  extraction artifacts (Lane B's probe-time field extraction) is still a separate, pending
  step this checker depends on but does not itself perform — confirmed by a repo-wide grep
  finding zero files anywhere that produce the extraction-artifact schema this checker
  consumes, so the HYBRID/L gate, while correctly built, has not yet been exercised on real
  data (see Claim 6a).
- **Finding 3** (multiplicity / BH-FDR): **closed for correctness, partially closed for
  scope.** `certify_batch_v2.py`'s BH step-up implementation was independently cross-checked
  against two from-scratch reference implementations on 5 synthetic p-value sets (including
  edge cases) and agreed exactly. Immutability is real, not aspirational: SHA-256-pinned
  inputs, a TOCTOU re-hash before writing, exclusive-create + read-only output — a 1e-9 score
  mutation without updating the manifest hash was tested and correctly rejected
  (`IntegrityError`, no file written). Gap: "family" = whatever one manifest file contains;
  there is no code-level guarantee that a manifest maps 1:1 to one domain/batch, so the
  frozen "BH-FDR across one domain batch" rule is upheld by convention, not by a structural
  constraint.
- **Finding 4** (DAG schema checks-but-doesn't-enforce): **closed for the historical bug
  pattern, not for adversarial construction.** Runtime enforcement (not just `validate()`) is
  real: a `_RestrictedInputs` wrapper raises on any undeclared key access, reproducing and
  blocking the exact v1 `ctx['other_node']` bug at execution time. Disconnected nodes cause
  the whole program to refuse to run (fail-closed, stricter than "excluded"). But a
  constructed adversarial node that bypasses the restricted-input wrapper entirely via a
  module-level global still executes successfully and has its provenance *silently
  mislabeled* as pure code — disclosed in the module's own docstring ("not an OS security
  sandbox"), but a real residual gap for finding 4's underlying concern (can seam
  measurements be trusted). The closely-related external-review finding 5 (ops retrieval
  indexing the full corpus including test split) **is** cleanly closed for the v2 blind path:
  `split_ops_v2.py`'s `_TrainRetriever` is built from TRAIN items only and raises `KeyError`
  on any held-out query, confirmed by direct source read.

## New validity seams v2 introduces

- The sealed-evaluator "seal" is a static AST allowlist plus subprocess isolation, explicitly
  **not** an OS security boundary — fine for today's regex-only candidate, but the guard
  would need hardening (or an actual sandbox) before trusting it against a more adversarial
  or larger candidate program.
- The typed-DAG enforcement has the same shape of gap: it stops the historical accidental-leak
  bug pattern but not a deliberately-written node that reads ambient globals instead of its
  declared inputs — and when defeated, it fails *silently* (wrong provenance label) rather
  than loudly (an exception). A future WS4-style cell built on `dag_schema_enforced.py` should
  not be treated as immune to the class of bug finding 4 originally flagged just because it
  uses the new schema.
- `may_claim_tacitness` being a hardcoded constant means the "provenance-aware claim
  permission" system has, for this one (most consequential) flag, no live logic to audit at
  all — it's correct by fiat rather than verified by a decision procedure. That's a reasonable
  engineering choice (never allow the claim) but future readers should not describe the system
  as "refusing" tacitness claims case-by-case.
- Reliance on `mtime` + hash/import-graph evidence for the additivity claim, in the absence
  of git history, is inherently weaker than a real `git diff` against a committed baseline —
  it would not have caught a same-timestamp or backdated edit. (No such edit was found, but
  the *method* of checking is itself a seam worth closing by actually committing the frozen
  baseline to git.)
- `technical_replay_v2`'s "real code execution" framing (65/800 certificates) risks being
  read as live execution performed today; it is in fact classification of a frozen prior
  execution artifact. The structured metadata discloses this correctly; the prose does not
  make it as clear as it could.
- `science_claims_v2`'s numeric-quantity extractor (`_QUANTITY_RE`) has a reproducible
  truncation bug that silently contaminates a meaningful fraction of the "strong" certificate
  bucket with spurious digit-collision matches (see Claim 4) — this is a fixable bug, not a
  design concern, but it currently sits uncaught by the existing test suite.
- `ops_capability_v2.py`'s function names create a false impression of coverage: `attributions()`
  and `is_refrain()` exist in the v2 module with the same names as the historically-flagged
  defective functions, but for the specific named defects (conjunct-verb gap, ≥3-word floor)
  they silently delegate to the unchanged v1 body. A future reader who sees "corrected
  capability library, `import ops_capability_v2`" and assumes the four originally-catalogued
  defects are behind them would be wrong for two of the four (see Claim 6c). The regression
  test suite (`test_ops_capability_v2.py`, 8/8 passing) does not include a case for either
  unfixed defect, so nothing currently guards against this gap being mistaken for closure.

## Bottom line

The v2 lane's central result — articulability (prompt) and verifiability (code) kept as
separate typed axes from isomorphism, with a144 landing on `proxy_mismatch` rather than a
tacitness or constructive-extension claim — is well supported by its own artifacts, and is
the strongest thing in the lane. I independently re-derived every number in the a144 chain
from raw score files using my own spearman implementation (not the repo's), independently
recounted the adversary's 26-row pass/fail table by hand, independently re-ran the outcome
classifier live against the recorded evidence and reproduced its output byte-for-byte, and
independently confirmed the blindness/sealing claims by reading the actual compiler bundle,
the actual (file-I/O-incapable) candidate source, and the actual timestamp ordering in the
sealed manifest. Nothing in that chain failed to reproduce, and the design that forces
`construct_fidelity` failure to pre-empt any correlation-based outcome is a genuine,
code-enforced guard against the most tempting overclaim this kind of experiment could make.
The rest of the lane is more mixed than the central result: additivity holds (verified by
timestamp/hash/import-graph forensics rather than `git diff`, since none of this code has
prior git history to diff against — a self-inflicted verification cost worth fixing by
committing frozen baselines going forward); the new BH-FDR/immutability instrument is
correctly implemented and genuinely hash-enforced; the contract checker's design and
abstention semantics are correct as built; and the retrieval-scope half of external review's
finding 5 is cleanly fixed. But three of the "new infrastructure" claims oversell their own
robustness or completeness in ways this audit could demonstrate by construction/execution
rather than by inference: the typed-DAG executor stops the historical careless-leak bug but
silently mislabels provenance when a node is adversarially written to bypass it; the
science-claims "171 strong certificates" figure, while bit-exact reproducible as a count, is
measurably inflated in evidentiary value by a numeric-extraction bug that a 10-item spot-check
caught at a 50% spurious rate; and the "corrected capability library" fixes and covers only
two of the four originally-named defects (the other two — `attributions()`'s conjunct-verb
gap and `is_refrain()`'s word-count floor — still silently delegate to the unchanged, frozen
v1 code despite living under matching function names in the v2 module, and are absent from
both the 7 stored counterexamples and the regression test suite). None of these three
undermines the a144 result or the additivity claim, but each means "v2 fixes this" should be
read narrowly and checked against what was actually tested rather than taken at the level of
the summary prose — this is, in fact, exactly the discipline the v2 lane itself tries to
enforce for its own central result (typed axes, explicit abstention, no claim beyond the
evidence), and it is worth applying that same discipline back onto the lane's own
secondary claims.

---

# Part 2 — conceptual reframing audit (follow-up, 2026-07-12)

Scope: does the v2 lane's *framing* of the pre-existing metric-seam program (not just its own
new artifacts) hold up — are its provenance labels for retained Math/Code/Patent/Science
machinery accurate, is its reframed central result consistent with the census/WS4 empirical
findings, and does any v2 document quietly re-scope a runbook-certified conclusion?

## 1. Provenance labels (manual/mock/oracle/replay) for retained Math/Code/Patent/Science machinery

**Verdict: VERIFIED (labels are individually accurate) + DESIGN CONCERN (the umbrella phrase
"Existing Math/Code/Patent/Science machinery" folds together artifacts of very different
vintage and relationship to the current program, without saying so at the level a reader
would see first).**

Traced every case in `technical_replay/initial_manifest.json` back to (a) the actual
underlying code/data and (b) whatever the runbook says about that same aspect ID, to check
for two failure modes: a label that's factually wrong, and a label that *downgrades*
something the runbook holds as certified without new evidence.

- **Math (`math_a150_sympy_scope_replay`, discovery_mode `replay`).** Runbook: math a150 was
  the *first* math census cell (crew 9), logged as **REJECT #8** (line 378) with the verdict
  table entry `math a150 | .2676 | .3385 | REJECT (vacuity + gate/signal double-count) | —`
  (line 523), and a follow-up E2L cell explicitly concludes "**THE REFRAME DOES NOT
  REPLICATE — SCOPE-MISMATCH**" (line 651) because "a150's real licensing clauses are
  'condition licenses an operation'... SymPy checks equation-rewrite" (line 663) — i.e. a
  clean *relation mismatch*, never promoted, never queued. The technical-replay case's own
  language ("a precise relation/corpus mismatch, not evidence that the criterion is tacit")
  is a faithful, almost verbatim carry-forward of the runbook's own conclusion. `replay` is
  the correct `DiscoveryMode` (the SymPy program was built by the *original* agentic census
  process, not by v2's blind compiler in this run, so `agentic` would be wrong; `replay`
  correctly means "a frozen prior artifact evaluated as if proposed to the blind selector").
  **No downgrade** — there was nothing certified to downgrade; a150 was already rejected.
- **Patents (`patents_a34_oracle_prior_art_replay`, discovery_mode `oracle`).** Already
  covered in Part 1, Claim 5. `oracle` matches the hand-off note's own caveat 6 and the WS3
  note's "oracle gold injection" section verbatim. Does not conflict with WS4 cell 5's
  separate, bit-exact-certified ablation (`prior_art_lookup Δρ=−.841` on TRAIN, runbook line
  938-946) — confirmed by grep that no v2 document anywhere cites or conflates the `−.841`
  WS4 train-ablation number with the WS3 `+.661` held-out evidence-op-marginal number; they
  are correctly kept as two separate experiments on the same aspect ID.
- **"Code" (`code_a104_mock_execution_replay`, discovery_mode `mock`; and
  `code_transplant_execution_2026_07_12_selected`, discovery_mode `replay`).** These are
  **not** the runbook's "coding fleet-build" lane. The runbook's own coding plan is explicit:
  "CODING = new-build lane from `datasets/code-review/crse_balanced_v2` (items → judge GPU
  pass → h0 fleet → contracts; longest pole)" (line 575), and the file-map table logs it as
  still at "stage-3 h0 baselines pending" — i.e. the current program's coding census has not
  produced a single cell result yet. The technical-replay "Code" cases instead trace to
  `methods/metric_seam/f2p_mock/` (an *already self-labeled* mock directory — `ops_exec.py`
  is confirmed by direct read to be a lookup over precomputed transitions, not live
  execution, matching its own `discovery_mode: "mock"` tag exactly) and to
  `datasets/code-review/pr_test_execution/`, a sibling directory to `crse_balanced_v2` that
  the runbook **never once mentions** (grepped "crse_balanced" and "pr_test_execution" across
  the full 1821-line runbook — only `crse_balanced_v2` appears, in that single line). So
  "Code" in the v2 progress note's "prior expensive Math/Code/Patent/Science machinery" does
  not refer to the program's active/planned coding census at all — it refers to an entirely
  separate, older prototype pipeline that predates or sits outside the current census
  program and is not tracked in its doc of record. This is disclosed at the per-case level
  (`code_a104`'s own `corpus.limitations` says "The ordinary code-review corpus is diff-only.
  Repository state exists only in this mocked replay lane") but **not** at the level of the
  progress note's summary sentence, which a reader skimming just the headline additions could
  easily mistake for a retrospective evaluation of the program's actual, still-pending coding
  lane. No factual error; a real clarity gap.
- **Science/peer-review (`peer_a214_release_grounding_replay`, discovery_mode `manual`; plus
  the new 2,400-paper full-text sets, also `manual`).** Grepped the runbook for `a214` (only
  appears once, in the 2026-07-11 "peer_review 8 cells" *panel extension registration* list —
  i.e. a214 was added to the queue, never run) and for "release_grounding"/`hybrid_gate`
  (zero matches anywhere in the runbook). Confirmed only `peer_review__a0` has actually been
  run as a census cell to date ("PEER P1 (peer_review a0...) FIRST peer_review-domain census
  cell", line 1534) — a214 has no runbook-certified result of any kind. The artifact v2 reads
  (`outputs/metric_seam_pilot/tasks/peer_review/hybrid_gate_report.json`) is a pre-census,
  never-cited fleet-build-stage evaluation. **No downgrade** — again nothing was certified to
  downgrade; v2's "unsupported" verdict on a214 is the first evaluation this aspect has ever
  received in either program.

**Net for item 1**: no case found where v2 mischaracterizes or downgrades a runbook-certified
census/WS3/WS4/held-out result. Every provenance label checked is individually accurate. The
one real issue is presentational, not factual: the phrase "existing ... Code ... machinery"
in the progress note's §3.4 heading invites conflation with the program's own still-pending
coding census lane, when the actual artifacts are from an unrelated, older, already
self-labeled "mock" pipeline the runbook never references.

## 2. Is the reframed central result consistent with census/WS4 empirical findings?

**Verdict: CONSISTENT where checked, with one genuine SCOPE GAP (not a contradiction).**

Checked the "articulability is prompt-based; verifiability is code-based; isomorphism
evaluated separately" framing against three specific census/WS4 findings named in the
follow-up request:

- **Floors-as-artifacts / bounded non-discovery.** The census's own accumulated finding —
  "floor-band r_hyb ≈ 0 measures INSTRUMENT BUGS, not tacitness" (runbook, 3 independently
  found floor mechanisms across CW/legal_ss_disability/humor) — is not just consistent with
  v2's "failure is only bounded non-discovery... never tacit" principle, it is close to a
  direct generalization of it: the census discovered this pattern empirically, criterion by
  criterion; v2 turns it into a structural, code-enforced rule (`Outcome` has no tacitness
  member at all, see Part 1 Claim 2). No tension.
- **Compiled-pole characterization.** The runbook's running "compiled pole" theory (some
  criteria — e.g. CW a144's "repair-only + compiled-pole = general-mechanics", line 1465 —
  sit near-fully code-explainable; others stay irreducibly L) is a claim about *where a
  criterion sits on a spectrum*; v2's typed axes are a claim about *how you'd verify either
  end of that spectrum once you're looking at one criterion*. These operate at different
  levels and don't collide — a compiled-pole criterion is exactly the shape of case v2 would
  classify `verifiable_only` or `dual_reconstruction`.
- **Seam placement / structured L fields as a winning lever — the one place with a genuine
  gap.** Several of the census's largest wins were driven specifically by *adding or grading
  a new LLM field* consumed by an otherwise-code hybrid program: ssdis a10's "graded
  `dispositive_weight` field... essentially the whole gain"; humor a306's "new graded LLM
  field `script_opposition_grade`... 94% of the total gain." Others (PR a64: "0 new LLM
  fields", CW a315: "all from removing an h0 noise component") were pure-code fixes. So the
  census's own empirical record is genuinely mixed, not uniformly "L-field placement wins" —
  this needed checking directly rather than assumed, and it turned out to be a real mix, so
  there is no single census "finding" that a clean articulability/verifiability split could
  contradict. Does v2 have a category for the L-field-driven wins? Yes: `RECONSTRUCTION_V2.md`
  explicitly names "prompt, code, and hybrid" as three evaluable channels, and the progress
  note draws on exactly this pattern in its own central result ("The historical hybrid's much
  higher agreement also shows why prompt/code complementarity should remain an explicit
  outcome" — directly citing a144's h0 rho .483 beating the code-only candidate's rho .066).
  So `hybrid` is real, used, and not silently erased. **The actual gap**: v2's typed
  vocabulary (`reconstruction_v2.py`) operates at *whole-criterion* granularity (one
  `ReconstructionEvidence` record per criterion), while the census's own "primary readout"
  (per the hand-off note's own §2 table: "primary readout = relation-match verdict per
  sub-relation") operates at *sub-relation* granularity — e.g. a333's 6-cell census pattern
  ("device presence = CODE-native, device position = exact library match, device function =
  irreducibly L" — all within one criterion). Nothing in `reconstruction_v2.py` or
  `dag_schema_enforced.py`'s public API currently emits a per-sub-relation classification;
  grepped both files for anything resembling the census's relation-match taxonomy (CODE-native
  / L-tagged-but-empirically-resolvable / genuine MISMATCH) — not present. So v2's reframing
  is not wrong about anything it covers, and it is not a superset of the census's own primary
  measurement object either — it is a differently-grained, whole-criterion typed vocabulary
  that currently coexists with, rather than subsumes, the census's finer sub-relation
  taxonomy. A reader treating v2's three axes as "the new version of" the census's
  relation-match findings would be over-reading it.

## 3. Retroactive scope — do v2 docs re-state runbook results in a way that changes their meaning?

Delegated to a subagent for a systematic grep-and-diff sweep across all v2 documents against
six specific runbook items. Two of those items I pre-verified directly myself (ground truth,
before checking what any v2 doc says):

- **WS4 ablation numbers** — directly re-confirmed in the runbook: patents a26
  `prior_art_lookup Δρ=−.718` (line 929, "WS4 CELL 4"), patents a35 `prior_art_lookup
  Δρ=−.667` (line 969, "WS4 CELL 6") — both match the hand-off note's own citations exactly
  ("a26 −.718 / a34 −.841 / a35 −.667"), so the hand-off note's own numbers are themselves
  accurate ground truth for the subagent to check v2 docs against.
- **BH-FDR survivor count** — pulled the actual addendum the runbook's prose points to
  (`outputs/metric_seam_pilot/battery/effort_ladder/census/cw_heldout_report.json
  ["_multiplicity_and_threshold_addendum"]`): `"multiplicity_BH_FDR_0.10": {"n_tests": 20,
  "pass": []}` — **zero** survivors at BH-FDR ≤ .10 across the 20 unambiguous CW held-out
  tests; only a144+a90 (creative_writing domain) clear the separate, actually-pre-registered
  G1 gate, and a135/a207 are explicitly demoted to "pairwise-only (exploratory, analysis-time
  rule)." Grepped all top-level v2 docs (`RECONSTRUCTION_V2.md`, the progress note,
  `technical_replay_v2/REPORT.md`, `science_claims_v2/REPORT.md`) for
  "BH-FDR"/"BH_FDR"/"Benjamini" — the only hit is the progress note's generic capability
  description of `certify_batch_v2.py` ("adds ... separate Benjamini-Hochberg families"),
  which does not cite the CW batch's specific 0/20 survivor count or the a144/a90 G1 status
  at all. **MATCHES (by omission)** — nothing is misstated because nothing specific is
  restated; the new instrument's capability is described in the abstract, not tied to a
  reprised historical number.
- Also independently grepped the same four top-level v2 docs for `a171`, `a333`, `a315`
  (the hand-off note's census-tally examples), `a135`, `a207` (the G1-adjacent exploratory
  candidates), and `a54` (the PROVENANCE_INCIDENT aspect) — **zero matches for every one of
  these six aspect IDs, in any v2 document.** The v2 lane's documents do not reach into or
  re-narrate the CW/legal/ssdis/PR census results at all; they stay confined to their own new
  artifacts (math a144, the technical-replay cases, the new science corpus) plus the two
  pre-existing results they explicitly retrospect on (math a150, patents a34/WS3). This
  itself lowers the retroactive-scope risk for this category of finding — there is very
  little surface area where a v2 doc could misstate a census/WS4/held-out number, because
  it essentially never quotes one outside of the a34/WS3 pair already checked in item 1.

**Subagent's full sweep (returned) — Verdict: no distortion found; the consistent pattern is
narrow scope / omission, not mischaracterization.** Six-item results:

| # | Item | Runbook ground truth | v2 doc coverage | Verdict |
|---|---|---|---|---|
| 1 | EXTERNAL REVIEW findings 2 & 5 | (2) "0.80 pre-registered" was inaccurate — G1 is the only pre-registered gate; (5) ops retrieval indexes full corpus incl. test split, audit queued | Neither finding is itemized or named anywhere in any v2 doc — unlike 1/3/4, which map to visibly-named modules (`contract_check_isomorphic.py`, `certify_batch_v2.py`, `dag_schema_enforced.py`) | **OMITTED** |
| 2 | G1/CW promotion status (a144, a90, a135, a207) | CW a144 P=.954/Δ+.003, a90 P=.967/Δ+.009, "only a144+a90 clear BOTH bars"; a135/a207 relabeled exploratory | Not mentioned; the only "a144" anywhere in v2 material is consistently `"criterion_id": "math__a144"` — **no conflation with CW a144 found** | **OMITTED, correctly disambiguated where touched** |
| 3 | WS4 a26/a35 ablations + legal a23/a21/a13 | a26 Δρ=−.718, a35 Δρ=−.667/84.0%, legal rho_intact figures (a23 .8838, a21 .6075) | Zero hits anywhere in v2 material; only the *unrelated, already-checked* WS3 patents-a34 numbers appear, matching exactly | **OMITTED entirely** |
| 4 | BH-FDR survivors | Runbook itself defers the count to an addendum file (n=20, 0 survivors — I pulled this directly, see above) | `certify_batch_v2.py` implements BH-FDR generically, cites no historical count | **MATCHES (trivially)** — nothing to conflict with |
| 5 | Census tally (a171, a333, a315) | a333 +54.7%; a171 +11.6% train but regressed on test (P=.30); a315 +17.2% but contract-FAIL/not queued | None of the three IDs or percentages appear anywhere in v2 material | **OMITTED** |
| 6 | a54 PROVENANCE_INCIDENT | CW a54 self-authored-contract incident, marked PROVISIONAL, later independently re-checked to **REJECT** | No genuine mention (one apparent grep hit was a SHA-hash substring coincidence, confirmed false positive by direct inspection); a *separate*, unrelated math-domain a54 census cell (PASS, queue #25) is also untouched by any v2 doc | **OMITTED** |

The subagent's own bottom line, which matches mine: *"I found no case where a v2 document
actually restates a runbook number or verdict and gets it wrong. Where v2 does cite
runbook-traceable numbers (math a144 blind-lane figures, the WS3 patents a34 rho
.745/.084/+.661, the 7/7 counterexample count), they match exactly."*

One connective observation worth adding: EXTERNAL REVIEW finding 5 (ops retrieval indexing
the full corpus including test split) is never named or credited in any v2 document's prose —
but Part 1 of this audit (Claim 6e) independently confirmed that `split_ops_v2.py`'s
`_TrainRetriever` *does* structurally fix exactly the mechanism finding 5 flagged (TRAIN-only
index, `KeyError` on held-out queries) for the new blind-run path. So finding 5 is fixed on
the merits without being claimed — the opposite failure mode from overclaiming, and arguably
the safer direction to err in, but it does mean a reader relying on the progress note's own
"headline additions" list to learn which of the 5 original findings were addressed would
undercount (missing 5) rather than overcount.

## Part 2 bottom line

No factual mischaracterization or retroactive re-scoping was found anywhere the follow-up
request pointed to. The provenance labels (manual/mock/oracle/replay) on retained Math/
Code/Patent/Science machinery are individually accurate, and in every case checked they
either faithfully carry forward a runbook verdict that was already negative/rejected (math
a150) or evaluate an aspect the runbook itself never certified in the first place (code a104,
peer a214) — so there is no instance of v2 *downgrading* something the runbook holds as
certified. The reframed central result (articulability/verifiability/isomorphism as separate
typed axes) is consistent with the census's own accumulated theory (floors-as-artifacts,
compiled-pole) and explicitly retains a `hybrid` channel that is actually exercised in the
lane's own central worked example — it is not a superset of the census's finer sub-relation
relation-match taxonomy, but it does not contradict it either; the two frameworks currently
operate at different granularities and neither document claims otherwise. On retroactive
scope, the dominant pattern — confirmed independently by both me and a dedicated subagent
sweep across six specific runbook items — is that v2 documents almost never reach back into
the CW/legal/ssdis/PR census or WS4 results at all (extensive grepping for a144-CW, a90,
a135, a207, a171, a333, a315, a54, and the WS4 a26/a35/legal ablation numbers returned zero
hits across every top-level v2 document). Where v2 *does* cite a runbook-traceable number —
math a144's blind-lane figures, patents a34's WS3 evidence numbers, the capability-library
counterexample count — it reproduces them exactly (already independently re-verified by
recomputation in Part 1). The one real gap is an omission, not a distortion: EXTERNAL REVIEW
findings 2 and 5 are never itemized or credited in v2's own summary prose, even though
finding 5's underlying mechanism is in fact fixed by `split_ops_v2.py` (see Part 1, Claim 6e)
— the v2 lane undersells its own coverage on that specific point rather than oversells it.
