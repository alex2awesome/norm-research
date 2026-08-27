# Silver-match v3 task completeness ledger

This ledger prevents teacher filtering, retrieval diagnostics, or one corpus
from being mistaken for completed all-norm production matching. A task is not
release-ready until its extraction, selected retriever, every corpus retrieval,
canonical typed final output, and final-production blind audit all pass the
fail-closed all-task auditor. The component table below tracks additional GEPA
and verifier work but does not supersede those release gates. `REJECTED` is
evidence about an attempted component, not a completed cell; the selected
fallback must still be frozen explicitly.

## Authoritative all-task release gate — 2026-07-13 06:40 PDT

The canonical manifest is SHA256 `b614e345…`: 1,732,515 norms, 23 corpora,
and 8 tasks. The append-only coverage report is
`production_v1/audits/alltask_coverage.snapshot_20260713_v6.json` (SHA256
`cdcd8c97…`). Its current gates are:

| Gate | Verified coverage | Status |
|---|---:|---|
| Frozen faithful extraction | 1,732,515/1,732,515; 23/23 corpora | COMPLETE |
| External-dev-only selected retriever/fusion | 8/8 tasks | COMPLETE |
| Exact-audited production K50 retrieval | 1,732,515/1,732,515; 23/23 corpora | COMPLETE |
| Canonical typed final outputs supplied to the auditor | 0/1,732,515; 0/23 corpora | MISSING |
| Passed final-production blind audits supplied to the auditor | 0/8 tasks | MISSING |

Retrieval is complete for Code 611,785/611,785, Creative 334,443/334,443,
Humor 77,378/77,378, Legal 326,522/326,522, Math 56,925/56,925, N&C
21,046/21,046, Peer 277,420/277,420, and PR 26,996/26,996. Every corpus is
materialized at K50 and bound to a passing exact row/ID/bank-hash audit in the
v6 coverage report.

The manifest-pinned current-bank leaf counts are Code 133, Creative 371,
Humor 285, Legal 104, Math 141, N&C 88, Peer 88, and PR 221.  Counts from role
overlap audits or historical banks must not be substituted for these values.

| Task | Frozen retriever/fusion | K50 dev-only GEPA adjudicator | Supported strict two-order verifier | Exact-audited all-corpus K50 retrieval |
|---|---|---|---|---|
| notice-and-comment | COMPLETE — promoted task Nemotron LoRA + dev fusion | COMPLETE — selected prompt `c839a2…` | COMPLETE — selection `f1567f…`, policy `11da2e…` | COMPLETE — 21,046/21,046 across both corpora |
| press-releases | COMPLETE — promoted task Nemotron LoRA + unretested dev fusion | IN PROGRESS — fresh source-disjoint 300-row, two-pass full-bank audit is frozen and queued locally in timeout-safe 5-row chunks; the prior 25-row execution timed out and was preserved failed-closed | MISSING — current same-code verifier is teacher filtering only | COMPLETE — 26,996/26,996 |
| code-review | COMPLETE — promoted r4 task LoRA; dense dev R@50 .955, frozen dev fusion 1.000 | FAILED CLOSED / REMEDIATING — explicit-role variants missed the precision gate; clean select400 and distill2000 full-bank packs are frozen and independent labeling is queued | MISSING | COMPLETE — exact-audited 611,785/611,785 across both corpora; frozen retrieval plan `dee85e…` |
| math-stackexchange | COMPLETE FALLBACK — r4 rejected; pretrained Nemotron + dev fusion retained | FAILED CLOSED / REMEDIATING — every explicit-role round missed the precision gate; fresh source-disjoint select400/distill2000 panels now cover all 141 leaves with zero prior/cross-panel overlap, and two-pass 5-row local plans are attested and queued | MISSING | COMPLETE — exact-audited 56,925/56,925 across four corpora; frozen retrieval plan `3efe34…` |
| creative-writing | COMPLETE FALLBACK — clean human-only LoRA rejected with zero dev gain; pretrained Nemotron + dev fusion selected at exact R@50 .900, macro .899; selection `6efe3d4f…`; test sealed | IN PROGRESS — fresh source-disjoint optimize120/select240 full-bank packs and timeout-safe independent plans are frozen/queued | MISSING | COMPLETE — exact-audited 334,443/334,443 across both corpora |
| humor | COMPLETE — clean human-only task LoRA + dev fusion; dense R@50 .717, macro fusion .750, test sealed; a new retriever LoRA remains unselected until its sealed external gate | FAILED CLOSED / REMEDIATING — clean 298/300 adjudicator and 292/300 verifier truth proved the current automatic stack imprecise; fresh disjoint optimize300/select300 full-bank plans are frozen, optimize truth is queued, and select remains sealed until all variants are frozen | FAILED CLOSED / REMEDIATING — three-order high confirmation retained only .511 exact precision, so no production path was opened | COMPLETE — exact-audited 77,378/77,378 K50 plus three 77,378-row full-bank K285 systems |
| peer-review | COMPLETE FALLBACK — audited-teacher task LoRA rejected; all 26 external-dev fused ranks tied the pretrained retriever, so saturated-R50 policy `994246fd…` retained frozen pretrained Nemotron fusion; selection `0695fedb…` | REBUILDING — fresh strict-v2 optimize160/select300 full-bank packs and independent plans are frozen/queued | REJECTED/MISSING — unsupported legacy attempts may not run production | COMPLETE — exact-audited 277,420/277,420; candidate `be0ea4f7…`, audit `ce464ee9…` |
| legal-outcome-prediction | COMPLETE FALLBACK — exact retry externally rejected: base and adapter R@50 .9667, gain 0 < .03, R@80 1.0; test remains sealed; base-retention selection `78c332ce…` binds decision `a1209d40…` and report `23be286f…` | REBUILDING — fresh strict-v2 optimize200/select360 full-bank packs and independent plans are frozen/queued with zero overlap against 2,403 prior exposures | REJECTED/MISSING — unsupported legacy attempts may not run production | COMPLETE — exact-audited 326,522/326,522 across all ten corpora; bound in coverage snapshot v6 |

N&C retrieval covers all 21,046 rows. Its currently promotable adjudication
components resolve 13,158 rows: 10,242 MATCH and 2,916 typed abstentions.
The two-order Gemma full-bank pass over the remaining 7,888 rows is complete.
Those rows are now in transcript-audited, truth-hidden multi-vote consensus.
After two independent Codex passes, strict unique two-vote consensus has
accepted 5,455 rows (including 3,757 exact MATCHes) and retained 2,433 as
unresolved.  A third isolated resolver pass over exactly those 2,433 rows is
in flight in 98 chunks. These components are not yet a canonical merged final.
Merge/final audit and blind MATCH/false-abstention packets remain required.

Exact current-bank IDs are primary throughout.  Concurrent L0→R3
re-clustering can populate only a separately versioned family-sensitivity
column; it cannot change any cell above or relabel a failed exact leaf.

## Sealed fresh-teacher gates

- Math StackExchange: the 172-row task-specific prompt-train panel adds no new
  retriever supervision relative to the rejected r4 LoRA teacher: exact UID
  overlap 172/172 and source-group overlap 172/172.  The hash-pinned overlap
  audit is `39519a06…` with status `LEFT_FULLY_RECYCLED_FROM_RIGHT`.  Therefore
  the frozen pretrained Nemotron fallback remains selected; no recycled-label
  LoRA retry is eligible.  A later retry requires genuinely new clean exact
  labels and the same external-dev promotion gate, with test still sealed.

- Peer review: three-pass all-high consensus `b28a1707…`; blind-audit labels
  `663ad907…`; 60/60 exact, raw Wilson 95% `[.93983, 1]`, design-weighted
  approximate Wilson 95% `[.92825, 1]`; all 60 audit rows permanently excluded;
  109 unique train rows, SHA `600e27ae…`. Combined with the prior audit-disjoint
  manual set, 184 exact teachers trained the task LoRA (`2f81d813…`). Internal
  depth-lexicographic selection retained epoch 0; the external-dev fused
  comparison then tied all 26 item ranks (R@50/R@80 1.0, R@30 .8462,
  R@16 .6923, MRR .3049). The predeclared paired policy therefore rejected the
  adapter for zero supported MRR/depth gain and retained pretrained Nemotron
  (`0695fedb…`). The external retriever test remains untouched.

- Legal outcome prediction: three-pass all-high consensus retained 69 exact
  MATCHes. A balanced 60-row/12-leaf hidden-ID fourth audit found 57/60 exact
  (raw Wilson 95% `[.86299, .98285]`; design-weighted precision .94499,
  approximate lower .85564, effective n 59.44). All audit rows are permanently
  excluded; the 9 remaining source-disjoint teachers were promoted as
  `f8029e8a…`. The exact task-local retry was subsequently evaluated on external
  dev and rejected: adapter and frozen base tied at exact R@50 `.9667`, with
  zero gain against the predeclared `.03` gate and no R@80 loss. The portable
  production selection `78c332ce…` therefore retains the base. The external
  retriever test remains `SEALED_UNCONSUMED` and must not be opened for the
  rejected adapter.

## Contamination / unavailable-leg ledger

- `press-releases` adjudicator test:
  `UNAVAILABLE_PRESELECTION_MATERIALIZATION`. The 81-row test label/candidate
  subset was materialized before dev prompt selection, but no adjudication was
  run, inspected, or scored. It must never be scored, quoted, used for prompt
  or threshold selection, or replaced with a favorable new test. Quarantine
  record:
  `/lfs/skampere3/0/alexspan/models/silver_match_v3_nemotron_lora_20260712_r3_context/adjudicator_k50/press-releases/PR_ADJUDICATOR_TEST.UNAVAILABLE_PRESELECTION_MATERIALIZATION.json`
  (SHA256 `595cd0d8925bd201433bfd5e6c386d7c034f1c2db998f1758beafe2c20e2e7c2`).
  PR claims therefore require frozen train/dev GEPA plus the independent
  uniform blind final-production MATCH audit.
