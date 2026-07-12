# Metric-seam program — verification hand-off (2026-07-12)

**Audience:** an independent agent verifying this work. No prior context assumed. Everything
below is checkable against files in this repo; the doc of record is
`notes/2026-07-10__seam-agentic-program-runbook.md` (pre-registrations at top, OVERNIGHT LOG
= one line per event, all results logged there first).

## 1. What the program is

Research question: when a human evaluative judgment (a rubric criterion like "diction
clarity", "novelty over prior art", "administrative exhaustion") is implemented as a metric
program, **how much of the judgment compiles into executable code, and what irreducibly
requires an LLM call?** The "seam" is the code↔LLM boundary inside a hybrid metric program.
We measure where the seam sits per criterion, whether tooling (capability libraries) moves
it, and how seam position varies across domains. Metrics are NEVER label-aware; all
improvement work is TRAIN-side only (test ids sealed; held-out evaluation happens in one
batch under pre-registered G1 bootstrap gates).

Working theory (accumulated, each with cited cells in the runbook):
- Code detects SURFACE; the "function wall" (mention-vs-executed, template-vs-craft,
  vacuity) blocks naive code. The wall is capability-graded: it recedes iff a library
  capability's RELATION matches the construct's sub-relation, up to that capability's
  maturity ceiling. Holistic aesthetic judgment is the floor.
- Constructs DECOMPOSE into sub-relations with different compilability (e.g. CW a333:
  device presence = CODE-native, device position = exact library match, device function =
  irreducibly L). Census pattern through 6 cells: lexical/structural sub-relations compile;
  semantic-use judgments stay L.
- WS4 (typed DAGs): seam position is FAMILY-structured — PR programs have a thin 2-node L
  frontier under pure-code assessment; patents programs are code+retrieval with a vestigial
  L frontier (dead-gated/starved); legal programs are zero-evidence reconvergent DAGs with
  judgment at depth-0 leaves and signal-poor deep code chains.
- WS3 (patents): a judge that can't see retrieval evidence produces structural false nulls;
  evidence-aware judging M̄(x,Z) flips op-marginals to +.21/+.61/+.66 (scope caveats in
  `datasets/patents/2026-07-10__evidence_aware_judge_ws3.md`).

## 2. Lane structure and status

| Lane | What | Status |
|---|---|---|
| A — Compilability census | 159 criteria / 7 tasks, solo "team+tools" cells in ascending r_hyb order; primary readout = relation-match verdict per sub-relation | cells 1-6 done (CW a54, a333, a279, a171, a198, a315), cells 7-8 in flight (a135, a324) |
| B — Instruments | v2.1 construct contracts ×115, harness v2, bug census | 87/115 contracts authored+validated; CW 26 / humor 23 / math 17 / PR 14 frozen; legal b1 7 validated, b2 in flight; ssdis 13 + peer 8 remain |
| C — WS4 typed-DAG pilot | 9 pre-registered cells (PR a119/a28/a76, patents a26/a34/a35, legal a23/a21/a13) | COMPLETE 9/9, all bit-exact round 1 |
| Coding fleet-build | 250 PR-diff items, 18 judge-qualified aspects | stage-3 h0 baselines pending |
| Parked | WS2, WS5, WS6.2-4, E2L-v2 coreference tier (needs user install decision) | — |

Census tally: 1 promotion-queued candidate (CW a171, first full contract-PASS build,
+11.6% train), 1 strong gain (a333 +54.7%), 1 gain-by-subtraction (a315 +17.2%, all from
removing an h0 noise component), 2 theory-consistent contract-fails (a279, a198),
1 process-REJECT (a54, see incident below). Earlier banked: press_releases a31 held-out
PROMOTION (P=.910) from the E2L arm; E2 base sweep = 0/20 promotions (the result that
motivated the capability-library reframe).

## 3. File map

**Doc of record / notes**
- `notes/2026-07-10__seam-agentic-program-runbook.md` — pre-registrations, protocols, all
  results, OVERNIGHT LOG. Claims in this hand-off should each have a log line there.
- `datasets/patents/2026-07-10__evidence_aware_judge_ws3.md` — self-contained WS3 note.

**Artifact tree** (root: `outputs/metric_seam_pilot/battery/effort_ladder/`)
- `panel_v3_census.json` — the 159-criterion census (per task: aspect, r_hyb, band).
- `contract_packs_v3/` — 115 authoring packs (criterion_description, genre_note,
  author_instructions incl. grep-verification spec).
- `contracts_v3/` — authored v2.1 contracts (95 files incl. one preserved
  `creative_writing__a54.REJECTED_SELFDEALING.json`).
- `contracts_v3_validation.json` — per-contract pass + sha1[:12] + flag adjudications +
  per-batch verdict blocks + domain freeze blocks (`_cw_domain_freeze`,
  `_humor_domain_freeze`, `_math_domain_freeze`, `_pr_domain_freeze`, `_legal_b1_verdict`).
- `contracts/` — v1 contracts (32) + harness-slot copies of v3 contracts (the harness reads
  this dir; census cells file-copy v3→here, never overwrite).
- `census/` — one dir per census cell (`creative_writing__a54` … `__a324`): plan.md,
  candidate.py, rounds.log, self_adversary.py + results, adversary.md,
  dominance_supporting.json, meta.json. Plus `census/PROMOTION_QUEUE.json`.
- `e2/` (20 base-sweep cells), `e2l/` (4 capability-arm cells), `e2_promotion_batch1.json`,
  `e2l_promotion_batch.json` (held-out results incl. the a31 promotion).
- `ws4/` — 9 DAG cells (`press_releases__a119/a28/a76`, `patents_pa__a26/a34/a35`,
  `legal_title_vii__a23/a21/a13`): plan.md, dag_program.py, equiv_check.py + result json,
  build_readouts.py + readouts.json, meta.json.
- `ws65_bug_census.json` — per-h0-program bug audit (classes 1-7; class 6 = dead-gate
  substring, class 7 = starved frontier, both added by WS4 patents cells).
- `PROVENANCE_INCIDENT_2026-07-12.json` — the a54 incident record + rebuilt sha ledger +
  containment rules + resolution.
- `E2_CREW_PROTOCOL.md` — frozen crew protocol v2.

**Code** (root: `methods/metric_seam/`)
- `battery/contract_check.py` — contract gate (harness v2: TRAIN-only execution, SEP/INV/TIE
  probe logic, SEP_FRAC=0.75, probe-mode-detection sentinel, completeness/range gates).
- `battery/agentic_run.py` — train-signal runner (same train-only pattern).
- `battery/battery_common.py` — task loading (PROGDIR map, load_ctx), split conventions.
- `battery/validate_contracts.py` — v1 central validation (v3 validation is done centrally
  by the main loop; verdicts live in contracts_v3_validation.json).
- `battery/dag_schema.py` — WS4 typed-DAG schema/executor/level-match validator.
- `battery/build_contract_packs_v3.py` — pack builder (descriptions from
  `runs/validity_full/v2/<task>/aspects.json`).
- `hybrids/ops_capability.py` (+ `test_ops_capability.py`) — FROZEN e2l-v1 capability
  library (spaCy attribution, SymPy, sentence_graph/is_refrain/discourse_position, etc.).
  Known defect (cell 5): attributions() conjunct-verb subject-sharing gap.
- `hybrids/programs_cw|programs_math|programs_humor|programs_legal|programs_v2(PR)|…` —
  frozen h0 hybrid programs per task.
- `f2p_mock/` — patents: `ops_pa.py` (PriorArtOps/NullPriorArtOps), `programs_pa/`,
  `ws3_eval_evidence.py`, `build_ws3_evidence_judge.py`, `eval_patents_pa.py` (split:
  rng(7) over sorted ids, 40% test).
- `pilot/build_task.py` — judge prompt builder (descriptions from aspects.json — same text
  the contracts copy), `pilot/gemma_score_v1.py` — offline batch vLLM judge scorer.
- `certificates.py` — spearman, attenuation_ceiling, bootstrap_gate (opt-in skip_undefined).

**Data**
- `outputs/metric_seam_pilot/tasks/<task>/items.json` for math, humor, creative_writing,
  legal_title_vii, legal_ss_disability, peer_review, patents_pa, code_review — list of
  {datapoint_id, text, judgement, ctext}. **ctext is the scoring basis everywhere**; the raw
  `text` field is NOT what programs/judges see (a PR batch was sent back over this).
- Press releases items: `outputs/metric_seam_pilot/v1/items_v1.json` (same schema).
- Judge scores: `tasks/<task>/results.jsonl` (pass1/pass2); LLM fields:
  `tasks/<task>/field_results.jsonl`. Patents evidence-arm judgments:
  `tasks/patents_pa/ws3_evidence_results.jsonl` + `ws3_eval_report.json`.
- Aspect registries: `runs/validity_full/v2/<task>/aspects.json`.

## 4. Known caveats a verifier must not "fix"
1. **Legal 600-char truncation:** 10/20 legal_title_vii aspect descriptions are truncated
   mid-sentence at the source aspects.json. The judge prompts used the SAME truncated text
   (pilot/build_task.py), so the truncated definition IS the operative construct. Contracts
   copy it verbatim BY DESIGN. Repairing it would break instrument consistency.
2. **h0 programs are FROZEN** including confirmed bugs (ws65 census); DAG refactors
   reproduce bugs bit-exact by design. Bug fixes are candidate material only.
3. **v1-era contracts** (contracts/, 32 files) predate the v2.1 standard — do not hold them
   to v3 gates. Same for the two frozen-domain standards deltas logged in the runbook
   (≥3-doc mention-only anchors adopted mid-program; earlier domains not retro-edited).
4. **Legal a13 judge coverage is 58%** (judge-side parse failures, rel1 .935 when parsed) —
   the WS4 cell correctly used structure-only readouts. Any rho computed on a13 judgments
   is underpowered; treat as flagged.
5. **Legal corpus contamination:** d00667 (ECOA), d00223 (VA loan) are not Title VII cases.
6. **patents pa_features.json has oracle-gold injection** (examiner-cited doc
   force-included). "Judge needs evidence" stands; "retriever discovers evidence" is NOT
   established. See the patents note §caveats.

## 5. Suggested verification targets (highest value first)
1. **Freeze integrity:** recompute sha1[:12] for every file in contracts_v3/ and diff
   against contracts_v3_validation.json entries + domain freeze blocks; confirm
   `creative_writing__a54.REJECTED_SELFDEALING.json` matches the sha cited in
   PROVENANCE_INCIDENT_2026-07-12.json; confirm v1 `contracts/` files match the incident
   file's rebuilt ledger.
2. **Train-only discipline:** audit contract_check.py + agentic_run.py for any test-id
   leakage; grep census cell code (candidate.py, self_adversary.py, build_readouts.py in
   ws4/) for reads of test-split judge scores.
3. **Census cell claims:** for each completed cell, re-run
   `agentic_run.py <task> <aid> census/<cell>/candidate.py` and `contract_check.py …` and
   compare against the rho/verdict lines in rounds.log + meta.json + the runbook log lines.
   Recompute dominance from dominance_supporting.json.
4. **WS4 equivalence:** re-run each ws4/<cell>/equiv_check.py (expect max|Δ|=0.0) and
   build_readouts.py; verify dag_program.validate() returns zero errors; spot-check the
   headline ablation numbers (a26 −.718 / a34 −.841 / a35 −.667 retrieval-node marginals;
   legal depth/frontier claims).
5. **Contract quality:** independently re-run the v2.1 gate script + the text-vs-ctext
   n-gram sweep (both described in the runbook 2026-07-12 entries) over all of
   contracts_v3/; re-verify a sample of grep counts per domain.
6. **Promotion queue:** confirm census/PROMOTION_QUEUE.json contains only contract-PASS
   candidates and that no held-out numbers exist anywhere for them yet.
7. **Bug census:** spot-check 5-10 ws65 entries against the actual h0 source (incl. the
   overturned a315 'comic'/'Comics' CLEAN→CONFIRMED flip).
8. **Runbook-vs-artifact consistency:** sample 10 OVERNIGHT LOG claims and locate the
   supporting artifact for each.

## 6. Where the program goes next
- Finish contract production (legal b2 in flight; then ssdis 13, peer 8) → all 7 domains
  frozen.
- Continue census cells: CW tail (a135, a324 in flight; then remaining CW), then humor →
  math → PR → legal → ssdis → peer → coding, ascending r_hyb within domain.
- Per-domain held-out promotion batches (one batch per domain when its cells complete;
  G1 bootstrap gates; only contract-PASS candidates).
- Coding stage-3: h0 baseline fleet → contracts → census cells (repo-grouped CV at eval).
- Lane B: probe-time LLM field extraction GPU pass (unblocks contract-blind floor cells +
  the patents starved-frontier fix vehicle).
- WS4 follow-ons (registered, parked): held-out r̃ vs scalar incumbents; U₂ matroid over
  DAG partitions.
- Deferred user decisions: E2L-v2 coreference install (tests whether a31's residual closes
  when its matched capability matures); whether E2L candidates share the effort-ladder
  promotion column.
- End state: a per-criterion compilability census + relation-match taxonomy with certified
  (held-out, gated) promotions as the paper's central empirical object.

## 7. Additive reconstruction-v2 lane (post-review)

The independent review led to a new additive lane; it does not rewrite the frozen artifacts
described above. The self-contained result and verification brief is
`notes/2026-07-12__metric-seam-reconstruction-v2-progress.md`.

Headline additions:

- prompt articulability, code verifiability, and frozen-reference isomorphism are now separate
  typed axes; negative results can license only bounded non-discovery, never tacitness;
- a label-free compiler and sealed evaluator produced one genuinely blind Math a144 candidate;
  independent adversarial construct testing rejected it (canonical outcome `proxy_mismatch`),
  and sealed reconstruction was rho=.066 versus .483 for the historical hybrid;
- prior expensive Math/legacy-Code-prototype/Patent/Science machinery is retained as selected
  retrospective pipeline seeds, with original manual/mock/oracle/replay provenance intact;
- a current replay classifies frozen telemetry from 800 earlier legacy-prototype code
  executions and finds 65 behavioral certificates; this is not the active coding census;
- the audit-corrected 2,400-paper full-text science lane yields 136 strong
  numeric/comparative relation certificates across 126 papers plus 435 separately labeled
  weak evidence links across 382 papers, without reading `y`; the historical 171/431 artifact
  is preserved but superseded after the quantity/identifier-collision audit;
- new contract, DAG, capability, and immutable multiplicity-aware certification instruments
  live under `methods/metric_seam/` with focused regression tests.

Do not promote the blind a144 candidate: it failed the frozen independent adversary. Do not use
its opened held-out split for a second confirmatory a144 build. Use a new criterion/split for the
next blind technical run.
