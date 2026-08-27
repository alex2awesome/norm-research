# Tacit knowledge — roadmap to completion

Date: 2026-07-14
Status: living document. Consolidates two related but separately-scoped research lines that have
previously only had status scattered across memory, a notebook cell, notes, and an ephemeral local
plan file. Update this file, not a new one, as either line progresses.

## How to use this document

- **Sub-line A (policy-isomorphism)** is the active line. It has a running experiment, a
  concluding-experiments program now partially implemented by Codex, and a real path to a defensible
  close.
- **Sub-line B (name-sufficiency scaling law)** is bookmarked per the user's own call
  (2026-07-14: "I never understood our taste/craft/mech dissociation and what it meant so let's just
  bookmark that for way later"). It is included here only as a status snapshot so it isn't lost, not
  as active work.
- Do not conflate the two. They ask different questions (model-to-model policy transplant vs.
  whether tacit knowledge becomes lexicalized as models scale) and use different instruments.

---

## Sub-line A: Policy-isomorphism (active)

### The question

Does explicit articulation (a definition, a rubric) let a smaller same-family LLM reconstruct a
larger LLM's own name-invoked policy, item by item, on unseen data — not external ground truth, not
task performance. Full formal estimand, claim ladder (`H_J`, `H_J^eq`, `H_NI`, `H_fiber`, `H_prev`),
and sampling/power design are frozen in
`notes/2026-07-12__strong-scale-articulation-substitution-protocol.md` — that note is the protocol
reference; this document is the status/next-steps tracker.

### Status ledger

| Result | Verdict | Notes |
|---|---|---|
| **gi35** (Llama-3.1 8B→70B, "Specific, quantified, checkable claims", press-releases, 400 sealed items) | **CERTIFIED existence**, sealed 2026-07-15 | Second sealed certified construct, new domain + construct family vs H49. Definition arm: adverse/quotient rho `.782/.796`, six-member simultaneous CIs `[.717,.829]`/`[.737,.843]`, native 70B−8B gap `.573 [.486,.663]`, all joint gates pass at `functional_ordinal` grade. Calibration→sealed shrinkage tiny (`.801→.782`). Batch-mates on sealed data: H23 `.648 [.556,.728]` fails floor (articulation gain certified), gi11 `.676 [.584,.743]` fails floor. Full numbers: `notebooks/data/two_faces_20260702/concluding_policy_confirmation_v2/lockbox_report.json` + sealed-results section of `notes/2026-07-14__concluding-policy-isomorphism-confirmation.md`. |
| **H49** (Llama-3.1 8B→70B, "Wordplay quality and clarity", 1,500 sealed items) | **CERTIFIED existence**, sealed 2026-07-13 | Adverse/quotient rho `.740/.754` vs `.562/.656` name-only; six-member simultaneous CI `[.701,.773]`/`[.717,.786]`; beats matched inert + wrong-construct controls. Full rubric arm observed only (`.722/.735`, CI crosses `.70`). Full numbers: `notebooks/data/two_faces_20260702/same_version_upper_confirmation_v1/H49_RESULT.md` + `lockbox_report.json`. |
| G6 residual lockbox (3B→8B, same construct) | Documented negative | Articulation transports level/bias/MAE but **rank transfer fails** (rho .790→.710). The reversal vs. H49 (fails at 3B→8B, succeeds at 8B→70B) is the single most interesting open pattern in the line — replicated 3× on independent draws, not yet a controlled ladder. |
| Old fixed-target ladder (1B→3B, 3B→8B, 1B→8B; G0–G13 chain, `fixed70_*`/`fixed_target_name_*` families) | **0 certified anywhere — confirmed still true** | Re-checked this session across every `_corrected`/`policy_v4` revision (v1→v7): every `n_certified_*` field is 0 at every revision. The corrections changed methodology/schema, not the null conclusion. Right-censored, not falsified — see protocol note's claim-ladder tier 4. |
| `target_surface_atlas_v1` (+ `_integrity_v1`, `_cross_family_v1`) | **RETRACTED — never quote** | 3 disqualifying defects, never fixed (byte-identical to a 2026-07-12 backup snapshot). Superseded, not repaired. |
| Everything else in `notebooks/data/two_faces_20260702/` dated 2026-07-04–13 (~150 files) | Load-bearing precursors or documented nulls | Full triage this session (fork) confirmed: G0–G13 search-program artifacts, all hash-pinned in the protocol note's artifact map. Two undocumented clusters (`adjacent_scale_*`, `residual_policy_isomorphism_arm_bank_v1`) are both confirmed-null/pre-scoring scaffolding — safe to ignore, not hidden findings. |

### Currently running (as of 2026-07-14 ~00:30 PDT)

**1. `tacit_breadth_confirmation_v3`** — sk3, physical GPU 0 only (enforced by the launcher's own
busy-check), started 2026-07-13 20:05. Open search/calibration pass: 990 cells (90/domain) × 11
domains × 3 prompt forms, name-only target (Llama-3.1-70B-Instruct BF16) then 3 domain-disjoint 8B
executor shards, then shard+analyze → `calibration_report.json`. Verified this session: correct
model paths/roles in the frozen execution manifest (`status: frozen-before-tacit-breadth-search-
model-outcomes`), zero failure markers, launcher script is byte-identical to the archived
`frozen_implementation_v2` copy actually running (not the stale live v1 copy still sitting in
`methods/codability/experiments/` on sk3 — that stale copy defaults to 2 GPUs and a weaker free-GPU
check; **do not launch from it directly**, always go through a frozen copy). Rough progress: ~171 of
an estimated ~1,485 total scoring batches for the target phase alone (~40 batches/hour) → target
phase ETA **~35-37h from launch (~2026-07-15 morning)**; full search (3 executor shards + shard +
analyze) likely **3-5 days total**. This is the machinery that will answer `H_prev`-adjacent breadth
questions once it lands — nothing to build, just track
`logs/target_70b_rep0.log` and artifact counts under `search_scores/llama31_70b_name_target/`
(expect 11 `.npz` when the target phase finishes).

**2. `concluding_policy_confirmation_v1`** — sk3, physical GPU 5/6/7 only. **Update 2026-07-14
~11:35 PDT: launched, calibration phase complete, lockbox blocked on a verified-benign gate bug —
see below.** This is the multi-construct sealed confirmation discussed this session. Confirmed
frozen scope from `methods/codability/experiments/concluding_policy_selection_v1.json` /
`concluding_policy_construct_panel_v1.json`:
- **3 prior-selected constructs**, same 8B→70B hop as H49 (`meta-llama/Llama-3.1-8B-Instruct` →
  `meta-llama/Llama-3.1-70B-Instruct`), same arm/control structure as H49 (source_definition +
  source_full_rubric, each with inert + wrong-construct controls):
  - `N_humor_23` **H23** "Laugh density and economy" — the lead the archival triage surfaced this
    session (observed rho .70-.74 at 3B in last week's exploration, never sealed).
  - `N_humor_11` **gi11** "Parody/pastiche craft" — the only legacy seed with both 3B→8B and
    fixed-70B rank already above .70; source-group-disjoint holdout.
  - `N_press-releases_35` **gi35** "Specific, quantified, checkable claims" — institutional-domain
    breadth, company-group-disjoint holdout.
- Selection is explicitly labeled `"prior-selected existence batch; not a prevalence sample"` in the
  frozen artifact — good, matches the discipline this line has held throughout.
- **Correction to this session's earlier plan:** the plan proposed a bonus 3B→70B rung for H23.
  Codex's frozen panel explicitly excludes this: **Llama-3.1 has no 3B checkpoint** (3B only exists
  in the Llama-3.2 family) — a 3B→70B test would cross model versions, breaking the same-family
  discipline this whole line depends on. Any future 3B involvement here must be a separately frozen,
  explicitly-labeled cross-version exploratory result, not folded into the primary confirmatory
  family. This was a real gap in the plan handed to Codex; Codex caught it correctly.
- **Open gap, not yet actioned:** this batch does **not** yet include an unselected/prevalence
  sample. This was identified as a fix mid-session (the prevalence tier doesn't actually need to wait
  for the breadth-search job — it could be drawn now from the same `tacit_breadth_arm_bank_v3.json`
  990-cell bank and folded into this same sealed batch) but has not been communicated to Codex as of
  this writing. **Next action: decide whether to fold ~15-20 unselected cells into this batch before
  the lockbox opens, or open the 3-construct lockbox as-is and add prevalence as a fast-follow batch.**

#### Calibration-stage result (2026-07-14, dev-fold only — NOT sealed, verified against raw floats)

On the open 400-item `tacit_breadth_search` calibration fold, **only `gi35` (press-releases)
cleared the functional gate — not H23**, the lead this session's archival triage was most bullish
on:

| Construct | Arm | Point rho (adverse/quotient) | Six-member simultaneous CI (adverse/quotient) | Gate |
|---|---|---|---|---|
| `N_press-releases_35` (gi35) | source_definition | `.801` / `.817` | `[.745,.844]` / `[.765,.858]` | **passes** — both lower bounds clear .70 |
| `N_humor_23` (H23) | source_definition | `.672` / — | `[.581,.748]` / `[.603,.764]` | fails — lower bound below .70, upper bound above (close) |
| `N_humor_11` (gi11) | source_definition | `.693` / — | `[.610,.760]` / `[.640,.783]` | fails — same pattern, also close |

None of the `source_full_rubric` arms passed for any construct. `n_certified_functional_equal_but_
different_pairs = 0` — no fiber, consistent with only one passing arm existing at all. This is
**calibration-fold evidence only**; H23 and gi11 are close enough that more items (the eventual
sealed panel, or a top-up) could still tip them either way — "didn't pass at calibration" is not the
same as "ruled out." Recomputed directly from
`concluding_policy_confirmation_v1/calibration_report.json` (pulled from sk3, not read from any
generated note).

#### Lockbox release-gate bug (verified benign, not a scoring/validity issue)

`write_calibration_release_artifact` → `validate_lockbox_release` (`policy_data.py:1029`) raised
`ValueError: calibration report analysis implementation differs from manifest`: the frozen execution
manifest declares 15 files as the "analysis implementation closure," but the calibration report's
own runtime self-report only lists 12. Traced the exact 3-file diff — all inert:
`methods/metric_implementer/__init__.py` (0 bytes), `methods/codability/experiments/__init__.py`
(docstring only, no code), and `methods/codability/__init__.py` (deliberately lazy-loaded via
`__getattr__` — its own docstring says this refactor was specifically done "to avoid... defeat[ing]
implementation-closure audits" by not eagerly importing unrelated legacy modules). Zero hash
mismatches on any of the 12 overlapping files. **Verdict: genuine closure-bookkeeping mismatch
between the manifest-freezing tool and the runtime self-report, not a code-drift or scoring issue.**
The fix (patch the closure declaration, reanalyze the already-saved calibration shards, no new model
calls) is scientifically sound — nothing about the scoring code changed between manifest-freeze and
runtime.

**RESOLVED 2026-07-15 (Fable took over; commit `451d841`).** The reanalyze-saved-shards route turned
out to be impossible by design: the manifest SHA is bound into every score artifact, and the drifted
file lists both include the very files a patch touches, so any repair self-invalidates part of the
frozen chain. But the calibration phase costs only ~20 minutes of GPU (not hours — 3 constructs ×
400 items is tiny for vLLM), so the clean fix was chosen instead: **one canonical
`ANALYSIS_IMPLEMENTATION_PATHS` tuple in `policy_data.py`** now feeds both the manifest compiler and
the report generator's runtime self-record (verified byte-equal end-to-end), the concluding
execution manifest was recompiled as **v2** (sha `a879ccd8…`; structural diff vs v1 = exactly the
four edited-file hashes + the v2 output directory, nothing else), `selection_v1` reused
byte-identical (it binds only bank+packet, not the manifest), 461/461 tests green (H49's
archive-fallback test now archives the drifted frozen bytes), and the full calibration+lockbox run
relaunched from a fresh hash-verified snapshot (`concluding_policy_a879ccd8`, GPU 5, PID 3567311,
launched 2026-07-15 12:06 PDT). v1's sealed validation partition was never read, so the redo is
uncontaminated; v1 calibration outcomes were observed, but this batch has no adaptive step between
phases (all arms pre-frozen in the selection artifact), so seeing them changes nothing about the
sealed test. **The same 15-vs-12 drift exists in the breadth job's frozen manifest** — its analysis
step will record 12 files against a 15-file declaration. That one cannot be re-scored cheaply
(days of GPU); reconciliation decision needed before the breadth selection step runs (see below).

### Concluding-experiments program

1. **Piece 1 (running, no action):** let `tacit_breadth_confirmation_v3` finish. Track only.
2. **Piece 2 (frozen, not yet launched):** the 3-construct `concluding_policy_confirmation_v1` batch
   above. **Decision needed from user before/at launch: add the unselected prevalence draw now, or
   launch the 3-construct batch as-is and add prevalence as a second batch later.**
3. **Piece 3 (downstream, gated on piece 1's report):** once `calibration_report.json` lands from the
   breadth search, select via two separate pre-declared rules in one more batched sealed pass, reusing
   the same generalized sealing path as piece 2:
   - **Confirmatory tier:** top-K observed winners per domain (e.g. top-2/domain ≈ 22 candidates),
     strict per-construct correction, existence-style claims only.
   - **Prevalence tier:** see "110-construct scale-up proposal" below — supersedes the original
     ~18-24 unselected-construct sketch with a much more developed design, now vetted.
   - Flat Bonferroni will be punishing at this member count — use Holm, or a pre-registered two-family
     split (confirmatory family strictly corrected; prevalence family reported with wider simultaneous
     CIs, not corrected away).
   - Add 1B/3B executor rungs (same-family: Llama-3.2 for 1B/3B, which means any ladder spanning
     3.2→3.1 crosses versions — see the correction above; a clean same-version capacity ladder needs
     either a same-version small checkpoint or an explicitly-labeled cross-version secondary result)
     for whichever constructs from piece 2 certify, to localize the capacity threshold behind the
     3B→8B-fails / 8B→70B-succeeds reversal.
4. **Power caveat that applies to pieces 2 and 3:** the reused `tacit_breadth_validation` held-out
   partitions are 400 items/domain vs. H49's 1,500-item sealed panel. CI half-widths scale roughly as
   1/√n, so expect wider CIs than H49's — a construct landing at "observed, not certified" may reflect
   lower power, not a weaker effect. Decide per-construct whether to top up with more sealed items
   before treating a near-miss as final.

#### 110-construct scale-up proposal (Codex, 2026-07-14) — vetted, two corrections applied

Codex proposed replacing further hand-picked confirmations with a proper population-level design:
11 domains × 10 constructs, stable-hash selected without model outcomes, excluding H49/H23/gi11/gi35
from the denominator, a fixed name/definition/rubric(+matched controls) instrument for every
construct (avoids per-construct prompt search contaminating cross-construct comparison), a
hierarchical/block-bootstrap population estimator (mean Δρ, functional-reconstruction prevalence,
content-specific prevalence), and two never-combined panels (a representative panel vs. a
discovery-replication panel drawn from piece 3's confirmatory tier).

**I initially over-claimed that the already-running 990-cell `tacit_breadth_arm_bank_v3.json`
panel could substitute for this — that was wrong, corrected in place:**

- **The 990-cell panel's `nominal_poststratification_weight` is NOT a Horvitz-Thompson/design-based
  weight.** Verified verbatim in the code's own claim-boundary text
  (`compile_fresh_name_arm_bank.py:461-464`, `run_policy_isomorphism.py:5466-5474`): the selector
  deliberately maximizes dependency/provenance-component diversity within each stratum rather than
  sampling with known inclusion probability, so "those factors are not inclusion probabilities and
  the result is not a Horvitz-Thompson or randomization-based survey estimate." The defensible
  estimand from that panel is **prevalence over the frozen balanced 990-cell panel** plus a nominal
  poststratified *descriptive sensitivity* — not population-representative prevalence over the
  general construct universe. Still meaningfully stronger than 3-4 hand-picked constructs, just not
  what "representative sample" implies. A genuine population-prevalence claim (Codex's stated aim)
  needs real randomization in the selector, not reuse of this panel's diversity-maximized one.
- **The native-gap conditioning is genuinely missing from the pooled prevalence summary — confirmed
  by reading the code, not just plausible.** `_BREADTH_BINARY_OUTCOMES`
  (`run_policy_isomorphism.py:3789-3805`, 15 outcomes) and the `outcomes` dict that populates them
  (`:4738-4767`) compute every substitution/fiber/control-improvement flag unconditionally per cell —
  there is no `has_native_gap` outcome and no gap-conditional rescue rate. A cell with no real 8B/70B
  gap gets pooled as a flat failure into the same denominator as a genuine-gap cell that failed to
  rescue, deflating any prevalence estimate drawn from this summarizer today. **This is a fix to the
  analysis layer, not a new experiment**: the raw ingredient already exists per cell
  (`scale_step_certificate.differences.native_rho_larger_minus_small` with a bootstrap CI, already
  computed) — it just isn't surfaced as a separate outcome. Needs three separated quantities per the
  protocol note's `H_prev` design: native-gap prevalence, gap-conditional rescue prevalence,
  unconditional substitution mass. Fix and re-run on already-scored data before trusting any pooled
  prevalence number from either the 990-cell panel or a new 110-construct panel.

**Action items, in order:** (1) fix the native-gap decomposition in the summarizer — cheap, no new
compute, blocks every downstream prevalence claim; (2) decide whether the 990-cell panel's
"prevalence over this frozen panel" claim is sufficient on its own, or whether a true-randomization
110-construct panel is still wanted for a population claim — these are different estimands and both
may be worth having, just never combined or mislabeled as each other; (3) if building the 110-construct
panel, also pre-register the "prevalent" threshold (the protocol note requires this before scoring,
and Codex's proposal doesn't yet name one) and verify the sampler actually uses known-probability
selection this time, not another diversity-maximizing picker; (4) note the existing 990-cell bank
already has 6 content routes per cell (not the proposed 2), all already being scored by the running
job — reuse that richer fiber structure rather than narrowing to 2 if/when a fiber analysis is run.

**Resolution (2026-07-15, commit `fb510ae` on `codex/tacit-breadth-v1`, independently verified):**
Codex implemented (1) and decided (2)–(4): the 990-cell finite panel is frozen as the primary
population (no second 110-cell panel; poststratified values demoted to descriptive sensitivities),
no post-calibration binary "prevalent" cutoff is added (point + interval only), and the 6-route
bank is retained. Verified against the diff, not the narrative:

- **Estimand fix is real and stricter than asked.** `tacit_breadth_decomposition_report/v4` now
  reports native-gap prevalence at three evidence grades, gap-conditional rescue prevalence
  (`n_defined_cells` denominators), unconditional substitution mass, and fidelity-qualified
  variants — plus `_native_gap_eligibility()` (`run_policy_isomorphism.py:3833`) fails closed if
  gap evidence varies across arms within a cell, so the conditional denominator can never become
  arm-dependent.
- **No frozen-hash collision with the running job.** The gap evidence (`native_scale_gap`) is
  emitted by `policy_isomorphism.py`, which fb510ae does NOT touch and which is byte-identical
  (sha `fe2e33d8…`) in the running job's `frozen_implementation_v2` snapshot and the v2 manifest.
  The frozen analyzer will therefore produce a report containing the needed evidence; the v4
  decomposition and fidelity gate run downstream as separately-recorded readout steps with their
  own implementation records. The concluding-policy lockbox failure class does not recur here.
- **New fidelity gate** (`compile_fresh_name_arm_bank.py:1633`): outcome-blind review packet of the
  exact frozen selected arms (construct + articulation text only; no scores/roles/items), hash-bound
  to the bank + selection artifact; only `faithful` arms enter the primary claim,
  `faithful_but_incomplete` is a declared sensitivity, failed arms are never replaced. The note
  correctly states the narrower defensible claim (review follows calibration selection but precedes
  sealed validation outcomes).
- **Test caveat:** in the clean worktree, `test_compile_fresh_name_arm_bank.py` cannot even collect —
  committed `compile_fresh_name_arm_bank.py:27` imports `hierarchy_groups` from
  `mine_clusters.py`, which exists ONLY in main's uncommitted 753-line working-tree modification.
  With that file supplied, all logic tests incl. the new fidelity test pass; the 11 remaining
  failures are missing untracked data (`outputs/hierarchy/`, frozen banks), environmental only.
  → the "commit `methods/codability/experiments/`" infra debt now also includes committing
  `mine_clusters.py` (or breaking the dependency); until then no commit of this pipeline is
  self-contained.
- Breadth job status at verification: 11/11 70B target domains done (last `press-releases`,
  07-15 09:37); 8B executor shard-a live on grant-funding/humor/peer-review/press-releases from the
  frozen snapshot. Branch is unmerged; running job is unaffected by it.

### Infra hardening (cheap, do alongside the above)

- `git add`/commit the entire `methods/codability/experiments/` tree — still 0 files tracked as of
  this writing; a laptop wipe would lose the pipeline that produced H49 and everything above it. The
  only current defense is one archived-file hash (`vllm_backend.py`), not the code.
- ~~Re-run `methods/codability/tests/` and resync/re-archive `vllm_backend.py`~~ **DONE, verified
  2026-07-14: 461 passed, 0 failed** (up from 401/403 on 2026-07-13; the benign `vllm_backend.py`
  drift is resolved and the suite grew, presumably from new concluding-policy-pipeline tests).
- Split `policy_oracle_recalibration_diagnostic_v1.md` by target model + precision — it currently
  mixes Llama-3.3-70B-FP8 dev rows and Llama-3.1-70B-BF16 sealed rows in one unlabeled table.
- Sync the stale live `run_tacit_breadth_search_sk3.sh` copy on sk3 (`methods/codability/
  experiments/`) to match the laptop/frozen v2 copy, or establish a convention of only ever launching
  from an archived/frozen copy — the live copy's weaker GPU-request defaults are a standing footgun.

### Where results live

- `notebooks/2026-07-02__two-faces-results-summary.ipynb` — the growing results notebook. §6 = H49.
  Next dated section (§8, after the roadmap currently in §7) should hold piece 2/3 results using the
  same pattern (recompute headline numbers by hand from raw `lockbox_report.json` floats, don't trust
  a generated note).
- `notes/2026-07-12__strong-scale-articulation-substitution-protocol.md` — estimand/protocol,
  authoritative for definitions and the frozen sampling/power design.
- `notebooks/data/two_faces_20260702/` — all raw artifacts; see the fork triage above for what's
  load-bearing vs. safely ignorable.

---

## Sub-line B: Name-sufficiency scaling law (bookmarked)

### The question

A different question from Sub-line A: does tacit knowledge become explicit/lexicalized as models
scale — measured as executor-verdict recovery AUC gain from adding a definition, relative to
name-only, not a model-to-model policy transplant. Ladder: Llama 1B/3B/8B (8B = self-recovery
column).

### Status snapshot (last substantively touched 2026-07-12; frozen here, not re-verified this session)

- **Taste/craft/mech three-way dissociation:** TASTE names come online 1B→3B (survival .63→.33
  deficient), CRAFT stays flat (~.61 to 8B), MECHANICAL never lexicalizes (.72) — i.e. codified ≠
  lexicalized. This is the result the user flagged as never fully understood/interpreted.
- **70B prereg:** frozen sha `62e4b3f0`, resolved 2026-07-12 — persistence 34/51 **falsified**
  (real reversals, not just noise), literal-prefix **falsified** (288 violations), but the frozen
  deficit-ranking has real ordinal signal (rank-AUC .689, perm p=.008); parametric law **rejected**
  by its own frozen rule (6/8 cells outside CI). Headline: "failed as frozen, ranking predictive."
- **Cross-family DiD:** math positive and robust across tiers (probe-clustered, p<.05); CW
  **sign-flips with scale** (+.036→−.015→−.017), replicates math-inversion, ruled out as an
  instruction-following artifact (would need to be monotone in capability gap; it isn't).
- **A-priori LODO:** null on the AUC scale (tags anti-predict, −.19) — an earlier bal_acc-scale
  positive (.35) was a calibration artifact, not a real signal.
- **`name_sufficiency_scaling_70b.json`** (new 2026-07-12 addition, confirmed this session): usable
  70B rung for code-review/humor/grant domains; self-flagged caveat says its CW cell mixes M_i
  scorings — never quote that cell.
- Snap-back/stipulation-suppression result (E2): promoted to load-bearing 2026-07-06 after nonce
  control + salience covariate + probe-clustered bootstrap + 2 families — humor-specific, not
  general; safety-artifact hypothesis doubly disconfirmed.
- Open technical debts if picked back up: gemma-3 serving still blocked (vLLM engine-init freeze
  across 6 strategies); gemma-4-31b weight-mapping bug; E1 stipulation probe contaminated (19/21 math
  defs leak name words, needs a name-scrubbed rerun) before it can be trusted.

### Why bookmarked

User's call, 2026-07-14: the taste/craft/mech dissociation was established empirically across many
sessions but was never brought to a clean mechanistic/interpretive account of *why* the three
categories behave differently — that synthesis work is the actual blocker to closing this line, not
more data collection.

### What closing this would take, whenever it's picked back up

1. A synthesis pass whose job is interpretation, not new measurement: given the existing dissociation
   (taste online early/craft flat/mech never), what's the actual mechanistic story? This is writing
   and re-reading existing results, not new experiments.
2. Resolve E1 contamination with a name-scrubbed rerun (cheap, CPU-side).
3. Decide whether a third model family (beyond Llama + Gemma-2) is worth unblocking gemma-3/gemma-4,
   or whether Llama+Gemma-2 is sufficient evidence for the family-generality claims already made.
4. Turn the 70B prereg's "failed as frozen, ranking predictive" into one clean, quotable headline
   claim rather than two separate verdicts.

---

## Open process debts (apply to both lines, concentrated in A)

- `methods/codability/experiments/` — the entire policy-isomorphism pipeline — untracked in git.
- ~~Test suite 401/403~~ resolved 2026-07-14, now 461/461.
- `policy_oracle_recalibration_diagnostic_v1.md` mixes FP8/BF16 targets in one unlabeled table.
- Stale live launcher copy on sk3 (`run_tacit_breadth_search_sk3.sh`) out of sync with the
  frozen/laptop copy — footgun for future launches if used directly.

## Change log

- 2026-07-19 (~23:55 PDT): **cross-FAMILY replication launched — Qwen2.5.** User asked to expand
  models/families; estimand clarified for the record (large model gets NAME ONLY — the claim is
  "the name suffices to invoke the policy in the big model, and the written-out knowledge fails to
  install it in the small one"; the old Face-2 grid measured articulation-vs-external-labels, a
  different estimand). Design: Qwen2.5-72B name-only target + executor ladder (3B/7B/14B/32B) on
  the three-class contrast panel (notice-and-comment / humor / math, 270 unselected cells), SAME
  frozen arms/items (articulations are mining-pipeline artifacts, not Llama-authored — cross-family
  reuse legitimate). Derived per-host family manifests (honest env + model jobs; interim-only,
  never canonical). LIVE: 72B target on sk3 GPU 5 (reclaimed by killing my redundant OOB lane —
  canonical had overtaken it), 14B executor on sk2 GPU 0; 3B/7B weights relaying to sk2 via laptop
  (sk3↔sk2 ssh dead both directions: sk2's expired AFS token breaks key auth); 32B queues for a
  free sk3 B200. GPU contention on sk3 is fierce (three launch attempts sniped mid-race by other
  users). Tally plan: same validated interim script with --target-job/--executor-job qwen ids.
  **UPDATE (2026-07-20 ~00:10): both core lanes LIVE and scoring; first target domain (humor)
  landed** after clearing two family gates: (1) sk3's cached Qwen2.5-72B was missing shard 1/37
  (layer-0+embeddings) → repaired via hf_hub_download, landed in `hub/` cache layout not the flat
  one → symlinked into the snapshot dir; (2) frozen manifest pinned LLAMA label-token ids
  (YES=14331/NO=9173) but Qwen tokenizes YES=14004/NO=8996 → patched
  `teacher_forced_label_validation` in all family manifests (the readout TEMPLATE is
  family-agnostic; only the id map is family-specific, and the runtime's own audit surfaced the
  correct Qwen ids). 72B target sk3 GPU4, 14B executor sk2 GPU0, contrast panel (n&c/humor/math).
  This 2-lane pair = minimum viable cross-family replication; 3B/7B/32B rungs deferred (3B weights
  still relaying). Landmine logged: broken shared-cache checkpoints need per-model
  index-vs-shard verification before a family run.
  **UPDATE (2026-07-20 ~01:05): deeper ladder launched (user asked).** Killed the laptop weight
  relay — it was the wrong fix. sk3↔sk2 direct ssh is dead (sk2 AFS-token expiry breaks key auth
  reading ~/.ssh on AFS), which is why I'd bridged through the laptop, BUT sk2 has working internet
  and pulls from HF itself (that's how it got its own 14B). So the 3B/7B/32B rungs now DOWNLOAD
  DIRECT FROM HF ON sk2 (self-contained `qwen_rung_sk2.sh`: download → index-vs-shard verify →
  derive manifest from the 14B template → score), one tmux + free H200 each (3B GPU1, 7B GPU3,
  32B GPU4), alongside the running 14B (GPU0) and 72B target (sk3 GPU4). Full Qwen ladder =
  3B/7B/14B/32B → 72B name target, contrast panel. Lesson: when a node-to-node hop breaks, check
  whether the destination can fetch the artifact directly before building a relay.
- 2026-07-23: **FROZEN CALIBRATION REPORT LANDED — the canonical 990-cell campaign's
  calibrated form** (tacit_breadth_confirmation_v3/calibration_report.json, schema
  policy_isomorphism_experiment/v5, search partition, arm bank sha e61999c6…; 990 cells /
  27,324 arms; identity validation clean). Certified tiers (Bonferroni over arms):
  **policy-isomorphic (strict target-self-band): 10/990 (1.0%); certified functional-ordinal
  (absolute-rank tier): 183/990 (18.5%; observed 636); certified functional substitutions:
  32 (observed 358); certified equal-but-different pairs: 58 (observed 640); certified local
  primary scale substitutions: 4 (observed 44; simultaneous 0); rescues: 10.** Reading: the
  STRICT tier is rare (~1%) — consistent with the sealed existence-result story (gi35/H49),
  NOT with the interim 9.7% which used a weaker conditioned-rescue statistic; the certified
  ordinal-approximation tier (~18.5%) is the prevalence-flavored number. NEVER map interim
  crosser-% onto these tiers 1:1 — different estimands (claim_boundary in the report is
  explicit: search/validation only; fiber members are behavioral certificates pending
  provenance/semantic review; sealed endgame remains the separate confirmatory step, still
  gated on the closure-reconciliation sign-off). Per-domain tier tables = follow-up parse of
  the per-cell shards. sk3 GPU0 frees after the residual scoring tail → W1 target passes
  (composed/negated/HOLISTIC) unblock.
- 2026-07-22 (~16:30 PDT): **Qwen-7B n&c: conditioned 10/90 (raw 13), median gain −.010, 49/90
  neg → the 7B FULL ROW completes (n&c 10 / humor 4 / math 2) and reveals STAGGERED UNLOCK
  RUNGS: n&c rescue turns on between 3B→7B while humor waits for 14B — the domain gradient is
  capacity-staggered, procedural first. The 3B sign-inversion is gone by 7B (−.010 vs −.151).**
  Qwen ladder now: 3B 0/0/0 (inversion) → 7B 10/4/2 → 14B —/19/— → 32B —/9(gap-closed)/—.
  7B lane GPU freed → EXP-GTK-1 launch sequence started.
- 2026-07-22 (~10:30 PDT): **Four new rungs tallied — three headline replications.** (1)
  **Gemma-4B n&c 0/90, median gain −.089, 72/90 hurt → the below-floor SIGN INVERSION replicates
  cross-family** (Qwen-3B was −.151, 68/90; humor +.008 / math +.023 merely inert — same shape).
  (2) **Qwen-32B humor: raw 36/90 but conditioned 9/90** — the collapse is the NATIVE GAP
  vanishing (32B-name ≈ 72B-name, gap>.10 fails), i.e. the rescuable window closes from the
  right; the conditioned rescue curve is HUMPED: Qwen 3B 0 → 7B 4 → 14B 19 → 32B 9. Do not read
  the 32B decline as articulation failing — it's the estimand's precondition (a gap to close)
  disappearing. (3) **Qwen-7B math: 2/90 conditioned — the first nonzero math cells in any
  family/rung** (Llama-8B math 0/90; Qwen is math-strong; n=2, flag not claim). Full
  conditioned table now: Llama-8B n&c 28 / humor 6 / math 0; Qwen 3B 0/0/0, 7B —/4/2, 14B —/19/—,
  32B —/9/—; Gemma 4B 0/1/0, 12B —/20/—. Remaining in flight: Qwen 7B/14B/32B n&c + 14B/32B math,
  Gemma-12B n&c/math. Artifacts: notebooks/data/two_faces_20260702/family_scores_{qwen25,gemma3}/;
  tally = gap-conditioned (best≥.70 ∧ gain>0 ∧ beats-controls ∧ gap>.10), interim/point-estimate.
- 2026-07-21 (~15:00 PDT): **Qwen-3B full gradient — SIGN INVERSION below the floor.** n&c 0/90
  with median gain −.151 and 68/90 hurt; humor 0/90 (−.005); math 0/90 (+.007). Below the
  capacity floor the domain gradient INVERTS: the richest/most procedural articulations (n&c
  statutory text) are the most HARMFUL to a sub-capacity executor, while humor/math articulations
  are merely inert. Articulation = a program; below the capacity gate its cost scales with its
  richness. The communication≫judgment≫verification gradient is an ABOVE-floor phenomenon. Key
  remaining cells: 7B/14B n&c (where does each domain's rescue turn on?).
- 2026-07-21 (~13:30 PDT): **Gemma-12B humor: conditioned 20/89 (raw 39), gain −.007, 50/90 neg —
  the 14B-Qwen inflection + bimodality REPLICATES at 12B in the second family.** Curve now
  0→1→4→6→20→19 (3B/4B/7B/8B/12B-Gemma/14B-Qwen): the articulable-craft unlock lands at the
  12-14B executor class in both families, targets 27B and 72B alike. Also Qwen-3B math: 0/90
  (floor replicates). Earlier: hardened GLM retry stack per user rule
  (api_field_runner.py::call = reference; empty-200 retryable; memory rule saved).
- 2026-07-20 (~21:00 PDT): **Qwen-14B humor: raw 33/90 was gap-confounded; GAP-CONDITIONED truth
  = 19/90 (gain>0 + controls beaten + native gap >.10).** Added gap-conditioning to the interim
  tally — the same denominator bug fixed in the frozen pipeline, caught in my own instrument
  (Llama panel rows barely move under conditioning ⇒ never confounded there). Two findings:
  (1) humor's upper limb exists — 0→1→4→6→19 across 3B/4B/7B/8B/14B, three families, monotone:
  judgment-domain tacitness is capacity-RELATIVE; (2) bimodality at 14B — median gain ≈0 with
  45/90 negative alongside 19 strong rescues: at high capacity the domain splits into
  articulable-craft vs anti-articulable-core constructs (fractal pattern, now within one rung).
- 2026-07-20 (~19:30 PDT): **THREE-FAMILY convergence on the humor capacity curve.** Gemma
  4B→27B: 1/90 rescued (+.008, 41 neg) — sits exactly between Qwen-3B (0/90) and Qwen-7B (6/90).
  Full curve: 3B:0 → 4B:1 → 7B:6 → 8B:6 across Qwen/Gemma/Llama — monotone in EXECUTOR size
  across three unrelated families and targets from 27B to 72B. Small target doesn't soften the
  floor ⇒ the bottleneck is executor capacity to EXECUTE the articulation, not teacher size.
  Sub-floor negative-gain fractions replicate too (41-48/90). Master table:
  `family_ladder_table.py` (jobs tmp) over `family_scores_{llama31,qwen25,gemma3}/`.
- 2026-07-20 (~14:00 PDT): **humor REPLICATES cross-family.** Qwen 7B→72B: 6/90 rescued (4 beat
  controls), gain +.003, 43 neg — same rescue count as Llama 8B→70B (6/90), same capacity floor
  (Qwen 3B: 0/90), same survivor type (show-don't-tell .743 recurs verbatim from the Llama panel;
  story beats .730; farce mechanics .727 — structure/craft, never core funniness). Gemma-27B
  target banked humor; Gemma-4B executor scoring. Tallies:
  `notebooks/data/two_faces_20260702/family_scores_qwen25/tally_qwen{3b,7b}_humor.json`.
- 2026-07-20 (~12:40 PDT): **THIRD family launched — Gemma-3 (user: "more model families").**
  Target = gemma-3-27b-it name-only, executors 4b (now, GPU 6) and 12b (queued for next free GPU);
  target on GPU 5. Same contrast panel + frozen arms. Generalized rung script downloads gated
  weights direct from HF on sk2 (token at $HF_HOME/token), verifies checkpoint, computes Gemma
  label-token ids via AutoTokenizer at derive time (fails loudly if YES/NO not single-token), and
  scores. Adds a smaller-target contrast (27B vs 70/72B): if the gradient replicates at a 27B
  target, it's not an artifact of giant targets either. Counterpart job entries in per-rung
  manifests are marked descriptive_only (never scored under that manifest).
- 2026-07-20 (~10:30 PDT): **first cross-family row — Qwen 3B→72B, humor: 0/90 rescued, median
  gain −.005, 48/90 NEGATIVE, native gap .545.** Consistent with the Llama capacity floor: a
  3B-class executor cannot use humor articulations in EITHER family (Llama-3.2-3B exploratory
  showed the same collapse). Qwen 72B target phase complete (3/3 domains, clean exit, GPU
  released); 7B/14B/32B rungs still scoring. The decisive gradient test (does n&c ≫ humor ≫ math
  replicate?) lands with each rung's remaining domains. Local-tally landmine fixed: NpzFile
  re-decompresses the full scores array on EVERY indexed access — 7.5K accesses OOM-killed the
  laptop tally 3× (sk3's RAM masked it); materialize `np.asarray(d["scores"])` once.
- 2026-07-19 (~19:30 PDT): **PANEL COMPLETE — all 11 domains, 990 unselected constructs. And
  advance prediction #2 FAILED, which kills the strong codification hypothesis.** Patents: 4/90
  rescued (predicted ≳15% from MPEP codification), gain +.085, 14 neg. Its few crossers are
  form/measurement constructs (biotech clarity/industrial-applicability .765, CII eligibility
  .739, parameter definitions .714). Revision forced by the miss: codified RULES are not enough —
  what matters is whether *executing* the policy is shallow (checking form/communication
  properties of the text: notice-and-comment 34%, grant 24%, PR 21%) vs. requiring deep content
  evaluation (patents 4%, legal 2%, math 0%, regardless of how codified the rulebook is).
  Panel totals: 96/990 (9.7%) point-wise crossers overall. Interim gradient final (interim
  caveats stand): n&c 34% > grant 24% > PR 21% ≫ humor 7% > CW 6% > peer 4% ≈ patents 4% >
  code-review 2% ≈ legal 2% > news 1% > math 0%. Prediction scorecard: #1 (creative-writing
  Class B) HIT; #2 (patents Class A) MISS — recorded as-is. sk2 lanes stopped and GPUs freed
  (outputs redundant once the canonical lane overtook them); accumulator exited cleanly
  ("ALL 11 DONE"); canonical GPU-0 pipeline continues re-scoring OOB-covered domains for the
  frozen analysis (~2 days), then: selection → blind fidelity review → sealed validation —
  still gated on the closure-reconciliation sign-off.
- 2026-07-19: **notice-and-comment (31/90 — new panel MAXIMUM) + news-homepages (1/90) — 10 of 11
  domains done, and the gradient's organizing variable sharpens.** Notice-and-comment rescues 34%
  with the SMALLEST native gaps (.304) and its survivors are codified-procedure constructs (ESA
  permitting .833, CAA §111(d) .816, petition procedures .780) — statutory knowledge that is
  already externalized in written rules. News-homepages (editorial newsworthiness judgment) lands
  at 1/90, near the floor. Revised reading: rescue rate tracks **how much of the domain's policy
  is already written down somewhere** — codified procedure (34%) > formulaic communication
  (21-24%) ≫ subjective judgment (1-7%) ≥ verifiable reasoning (0-2%). ADVANCE PREDICTION #2
  (registered before patents lands): patent examination is heavily codified (MPEP), so patents
  should land Class-A-side (≳15%) despite being "technical." Sealed-harvest candidate pool now
  much larger (n&c alone adds 31 crossers).
- 2026-07-18 (~01:30 PDT): **legal-outcome-prediction interim tally (8th domain, via sk2 lane +
  laptop relay; sk2→sk3 ssh broken by expired AFS token, persistent laptop relay armed).**
  Rescued 2/90, gain +.103, 13 neg — near-math despite having the LARGEST native gaps in the
  panel (median .557; most room to rescue, least rescued). The kicker: both survivors are
  RHETORIC constructs (pathos/affect management .802, facts-as-vivid-narrative .717) — the
  communication-craft periphery inside the legal domain, not legal-doctrine judgment. The
  verifiable-core-resists pattern holds even within-domain: what crosses is always the
  describable communication layer.
- 2026-07-18 (~00:40 PDT): **math-stackexchange interim tally (7th domain) — the panel floor:
  0/90 rescued.** Median gain +.048 (R1 .028 < R2 .046 < R3 .057), 13/90 negative, median gap
  .381. The verifiable-reasoning domain is not an intermediate case — it's the hardest domain in
  the panel for articulation-rescue at 8B→70B, consistent with the old cross-task math inversions
  and the MECH-never-names finding from the tacit-scaling thread. Also diagnosed sk2 lanes: alive
  and healthy but ~6× slower than sk3 effective rates (CPU contention with the v14 campaign
  between batches; in-batch it/s looks normal) — lane2 still inside legal after ~25h. Canonical
  sk3 lane delivered math first.
- 2026-07-17 (~02:30 PDT): **creative-writing interim tally (6th domain) — Class B as predicted
  in advance.** Rescued 5/90 (6%), gain +.057, 13/90 negative. Survivors are craft-technique
  constructs (diction .746, dialogue .735, characterization mechanics .709-.723) — describable
  craft, not holistic taste. Two-class structure now 6-for-6, with the class membership of this
  domain called before its data landed.
- 2026-07-17 (~01:15 PDT): **press-releases interim tally (5th domain; gi35's home) — the domains
  now split into two clean classes.** Rescued 19/90 (21%), median gain +.169, only 1/90 negative —
  near-identical profile to grant-funding (24%, +.163, 6 neg). Class A "institutional/formulaic
  communication" (grant-funding, press-releases): ~21-24% rescued, gain ≈ +.165, negatives rare.
  Class B "core-judgment" (humor 7%, peer-review 4%, code-review 2%): gain +.03-.09, negatives
  common. gi35 certifying out of a Class-A domain is consistent. Wrinkle: press-releases R3
  rescues 7/30 (best level, alongside R2), unlike other domains where R3 is worst. Top unselected
  constructs are strong (.834 public-safety/risk comm, .826 value proposition, .793 impact
  storytelling) — plausible future sealed candidates. Shard-a complete; canonical lane advanced to
  executor-b (math/news/patents).
- 2026-07-17 (~00:30 PDT): **code-review interim tally (4th domain, via the sk3 OOB lane).**
  Rescued 2/90 (2%), median gain +.066, 13/90 negative-gain. Both survivors are COMMUNICATION
  constructs (change-description clarity .751, PR communication clarity .703); the technical-
  judgment constructs just miss (tests well-designed .684 despite a .63 native gap, refactoring
  discipline .697). Four-domain picture: grant-funding 24% ≫ humor 7% > peer-review 4% ≈
  code-review 2% — grant-funding increasingly looks like the outlier; the typical domain rescues
  2-7% point-wise with modest positive gain, and the survivors are consistently the describable/
  structural constructs, never the domain's core judgment.
- 2026-07-16 (~23:45 PDT): **scaled to three hosts — every one of the 11 panel domains now scoring
  somewhere.** User authorized sk1/sk2/sk3. Cross-host lanes CANNOT feed the canonical pipeline
  (frozen manifest pins hostname/executable/packages + sk3 model paths), so remote lanes run under
  per-host DERIVED manifests (each host records its true environment; only env block + executor
  model path differ from the frozen manifest; hash preflights 0 mismatches on both hosts) and are
  interim-descriptive only. Live lanes: sk3-GPU0 canonical (press-releases → b → c), sk3-GPU5 OOB
  (code-review → creative-writing → …), **sk2-GPU0 H200** (math → news → patents), **sk2-GPU5
  H200** (legal → notice-and-comment). Each remote lane pushes finished npz to sk3's
  `search_scores_oob/` where the tmux accumulator tallies them and sk3's OOB scorer then SKIPS
  those domains (verified `already_complete` skip logic). sk1 ABANDONED after 4 env layers
  (vllm-0.25/py3.12 env needs a missing conda toolchain: `x86_64-conda-linux-gnu-cc`; plus GPUs
  kept being claimed) — effort-boxed, not worth overnight risk. Landmines recorded:
  `vllm_backend.py:231` defaults worker-home to `/lfs/skampere3/0/alexspan` (must export
  `METRIC_IMPLEMENTER_LFS_HOME` off-host); sk2 tmux needs `LD_LIBRARY_PATH=` (conda ncurses);
  transitive packet deps = prior packet manifests + 1.6GB raw datasets
  (`breadth_interim_deps.tgz`). Cross-host scores are same snapshot/prompts/readout but different
  vllm versions (0.16 sk2 vs 0.17 sk3) — tally rows traceable via npz `reader` path; never mix
  into canonical analysis. ETA: most domains tallied by 2026-07-17 evening.
- 2026-07-16 (~00:50 PDT): **old-artifact audit (user question: are the pre-refactor "many
  isomorphic pairs" usable?).** Swept all 68 old reports with nonzero isomorphism counts. Decoded:
  the counts are OBSERVED-grade (uncorrected dev-fold points, open partitions) on a HANDFUL of
  hand-picked constructs — the biggest (19 fiber pairs, 8 substitutions) are all `N_humor_49` =
  H49's own dev artifacts; the 68 files are v1→v7 re-analyses of the same few cells, not 68
  results. Crossfold stable-certified floors explain the sequel: llama8 .67 (→ later certified as
  H49 with a bigger sealed panel), llama3 .63 (never certified; consistent with today's 3B
  exploratory rung), llama1 .16 (dead). `adaptive_ostensive_isomorphism_v1` self-reports 0
  policy-isomorphic (n=64 fold candidates, 32 stable recipes, 0 both-margin). Verdict: old scores
  are valid as SEARCH/lead-generation artifacts (same scorer lineage) but must never be quoted as
  findings; the sealed H23/gi11 failures measured the dev-optimism gap directly. ALSO: the new
  humor negative-gain result has a pre-refactor antecedent — the old two-faces "humor
  rubric-collapse" (3B scored worse from the full rubric than from a 35-word explanation,
  notebook §4 cell) — same phenomenon, different criterion (bal-acc vs labels then, rho vs 70B
  policy now); independent-method convergence.
- 2026-07-16 (~00:20 PDT): **peer-review interim tally — a THIRD distinct domain signature.**
  Rescued 4/90 (4%), but median gain +.090 with only 8/90 negative-gain cells. So three domains,
  three shapes: grant-funding = articulation *reliably strong* (24% rescued, gain +.163);
  humor = articulation *often harmful* (7%, gain +.035, 21/90 negative); peer-review = articulation
  *reliably helpful but rarely sufficient* (4%, gain +.090, only 8 negative). Rescue rate is NOT
  monotone in mean gain — peer-review helps more consistently than humor yet certifies less, because
  its executor ceiling/native gaps sit lower. Peer-review's few crossers are all meta-scientific
  (novelty/significance .756, positioning vs prior work .743/.735, methods justification .733) —
  structural, not holistic quality. Notebook §6b.3 now renders 3 rows (data synced). Same interim
  caveats.
- 2026-07-15 (~20:45 PDT): **humor interim tally (90 unselected cells) — first cross-domain
  contrast, and it's large.** Rescue rate 6/90 (6.7%) vs grant-funding's 22/90 (24.4%); median
  articulation gain +.035 vs +.163; 21/90 humor cells have *negative* gain (articulation hurts).
  Native gaps are smaller too (median .352 vs .481). No level is good (R1 1/30, R2 3/30, R3 2/30).
  The few humor constructs that do cross ≥.70 are meta/structural (trauma-topic ethics & timing
  .755, specificity/show-don't-tell .754, sociality of laughter .740) rather than core funniness
  judgment. Retroactive context for the whole line: humor — where H49 was found and H23/gi11
  failed sealed — is apparently the HARD domain; the sealed humor failures look like domain
  difficulty, not bad luck. Same interim caveats as grant-funding. Artifacts:
  `sk3:/lfs/skampere3/0/alexspan/interim_tallies/` (per-domain jsons + `combined_summary.tsv`,
  autonomous tmux accumulator `tally_acc` appends each domain as either executor lane finishes it).
- 2026-07-15 (~19:50 PDT): **scaled the breadth executor to a second GPU lane to cover more tasks
  tonight.** The supervisor scores 11 domains sequentially on GPU 0 in three shards (a: grant/humor/
  peer-review/press-releases; b: math/news/patents; c: code-review/creative-writing/legal/
  notice-and-comment — c lands *last*, days out). Added an out-of-band 8B executor on the one free
  GPU (5) scoring c-then-b into a SEPARATE `search_scores_oob/` dir, purely to feed interim tallies;
  it never touches the supervisor's GPU-0 source-of-truth pipeline. **Key infra landmine found and
  documented:** sk3's live `norm-research` is NOT a git repo and its `methods/codability/__init__.py`
  (hash 78cb83, July 1) does NOT match the frozen breadth manifest (b3e769); the breadth job only
  validates because the supervisor sets `ROOT`/cwd to `frozen_implementation_v2/` (which has a
  manifest-matching `methods/` tree + symlinks to live `datasets/`+`notebooks/`). Any parallel/OOB
  run MUST cd into that snapshot, not the live tree, and mirror the launcher's env (notably
  `unset FLASHINFER_CUDA_ARCHS`). Corollary confirmed: executor-b/c will validate fine when the
  supervisor reaches them (same mechanism as the healthy executor-a) — my earlier commit `451d841`
  is laptop-only and never touched sk3's live tree.
- 2026-07-15 (~15:40 PDT): **first unselected-panel numbers — grant-funding interim tally, 90
  cells.** Computed with the validated interim tally (reproduces the frozen v2 report's gi35
  numbers to 3 decimals; point estimates only, open partition, best-arm-per-cell is adaptive, no
  CIs/multiplicity/fidelity — NOT H_prev, superseded by the frozen 990-cell analysis).
  Results: 89/90 cells have a native 70B−8B gap > .10 (median .481); median best-arm articulation
  gain over name-only +.163 (p10 .021 / p90 .289 — positive almost everywhere); 22/90 (24%) reach
  best-arm adverse ≥ .70 point-wise and all 22 beat their matched controls. Winning routes are
  diverse (leaf_inventory 5, units_full 5, dossier 4, units_4 4, rules 2, definition_rules 2 —
  richer routes beat bare definition in this domain). By level: R2 11/30 ≥ .70, R1 7/30, R3 4/30
  (R3 has the largest gaps, .549 median, but is hardest to rescue). Artifact:
  `sk3:/lfs/skampere3/0/alexspan/interim_tally_grant-funding.json`; script
  `interim_breadth_tally.py` (scratchpad + sk3 home). Remaining shard-a domains land overnight
  (grant took ~6h wall; 400-item domains are ~4× the prompts); persistent watcher armed.
- 2026-07-15 (~15:00 PDT): **capacity ladder complete (exploratory, cross-version).** Built and ran
  Llama-3.2-1B and -3B executor rungs against the frozen 70B name target on the open partition
  (fake-backend rehearsed first; both runs clean on GPU 5, ~35 min each). Definition-arm adverse
  rho — gi35: .268 (1B) → .721 (3B) → .801 (8B); H23: −.04 → .53 → .67; gi11: .07 → .55 → .69.
  Hard executability floor between 1B and 3B (all arms inert at 1B); threshold is
  construct-dependent (gi35 nearly recovers at 3B, humor doesn't). Cross-version baselines
  non-monotone → exploratory only, never a same-family scaling claim. Full table + manifests in
  the concluding-confirmation note's capacity-ladder section.
- 2026-07-15 (~12:45 PDT): **SEALED RESULT — gi35 certified.** The v2 re-execution completed
  cleanly end-to-end in 39 minutes (calibration 12:06–12:25, release gate passed, lockbox
  12:25–12:45). On the never-before-read 400-item validation partition: gi35's definition arm
  certified at the full six-member simultaneous grade (adverse `.782 [.717,.829]`, quotient
  `.796 [.737,.843]`, native gap `.573`), replicating calibration with tiny shrinkage; H23 and
  gi11 fail the .70 floor on sealed data (H23 retains a certified articulation *gain*). Verified
  against raw report floats, release-chain hashes, and gate flags — not the run's own summary.
  Status ledger updated; sealed-results section added to the concluding-confirmation note.
- 2026-07-15 (later, ~12:10 PDT): took over execution at the user's request. Root-caused the lockbox
  release-gate bug to two independently-maintained closure lists (manifest compiler's 15-path tuple
  vs report generator's hardcoded 12-path list); unified them into one canonical
  `ANALYSIS_IMPLEMENTATION_PATHS` in `policy_data.py` (commit `451d841`, on branch
  `codex/metric-seam-family-scale-v1` — the checkout's current branch, tidy at merge). Recompiled
  concluding manifest v2 (`a879ccd8…`), audited its structural diff vs v1 (only expected changes),
  verified 461/461 tests, built a fresh hash-verified sk3 snapshot, and relaunched the full
  calibration+lockbox run on GPU 5 (PID 3567311). This is the path to the first sealed result since
  H49: gi35's sealed validation. Flagged: the breadth manifest carries the same closure drift and
  needs a reconciliation decision before its selection step (re-scoring is not an option there).
- 2026-07-15: verified Codex's implementation of the estimand + fidelity corrections (commit
  `fb510ae`, branch `codex/tacit-breadth-v1`, worktree `/tmp/norm-research-tacit`). All headline
  claims check out against the diff and sk3 artifacts: v4 decomposition with three-grade native-gap
  conditioning (plus a fail-closed arm-invariance guard), outcome-blind hash-bound fidelity review
  gate, 990-cell panel frozen as primary population, no new panel. Confirmed no frozen-hash
  collision with the running breadth job (`policy_isomorphism.py`, the file that emits gap
  evidence, is untouched and hash-matches the frozen snapshot). Breadth job: 11/11 70B target
  domains complete; 8B executor shard-a running. One new landmine found: the committed pipeline
  imports `hierarchy_groups` from `mine_clusters.py`, which only exists as an uncommitted
  working-tree edit on main — no commit of this pipeline is currently self-contained. Details in
  the "Resolution (2026-07-15…)" block under the 110-construct subsection.
- 2026-07-14 (still later): Codex proposed a 110-construct population-level scale-up. Reviewed and
  mostly endorsed the design (fixed instrument, two never-combined panels, freeze-before-more-results
  sequencing), but initially over-claimed the existing 990-cell breadth panel could substitute for a
  new one — corrected after verifying the code's own claim-boundary text: its poststratification
  weights are explicitly not Horvitz-Thompson/design-based, so it supports "prevalence over this
  frozen panel," not general population prevalence. Also confirmed, by reading the code, a real gap
  Codex flagged: the pooled prevalence summarizer (`_BREADTH_BINARY_OUTCOMES`) has no native-gap
  conditioning and silently pools no-gap cells as failures — fixable from already-scored data, no new
  compute, and blocks every downstream prevalence number until fixed. See "110-construct scale-up
  proposal" above for the full vetted version and action items.
- 2026-07-14 (later, ~11:40 PDT): calibration phase of `concluding_policy_confirmation_v1`
  completed and independently verified against raw `calibration_report.json` floats. Correction to
  the earlier entry: **gi35 (press-releases), not H23**, is the construct that cleared the
  calibration-stage functional gate (rho .801/.817, simultaneous CI lower bounds .745/.765); H23 and
  gi11 came close (CIs straddling .70) but did not pass. Lockbox blocked by a release-gate exception
  (`policy_data.py:1029`); traced and verified benign — a closure-bookkeeping mismatch over 3 inert
  `__init__.py` files, zero hash mismatches on any real code. Breadth job independently reconfirmed
  healthy (4/11 domains done, 0 failure markers). Test suite independently rerun: 461/461 passing
  (supersedes the 401/403 status below).

- 2026-07-13: H49 sealed positive result certified and independently re-verified; §6/§7 added to
  `notebooks/2026-07-02__two-faces-results-summary.ipynb`.
- 2026-07-14: full archival triage of last week's ~150-file artifact dump (fork) — all load-bearing
  precursors or confirmed nulls, no revision ever overturned the old ladder's 0-certified result, one
  new lead (H23) surfaced; concluding-experiments program discussed and drafted with user; handed to
  Codex, which froze `concluding_policy_*` manifests for the 3-construct existence batch (not yet
  launched) and correctly caught/excluded the plan's invalid 3B-rung-for-H23 idea (no 3B checkpoint
  in the Llama-3.1 family); this roadmap document created to consolidate status across both sub-lines
  in one durable, git-tracked place.
