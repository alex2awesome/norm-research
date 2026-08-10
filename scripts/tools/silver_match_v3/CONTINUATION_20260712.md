# Silver-match v3 continuation checkpoint (2026-07-12)

Active goal: produce append-only, independently validated norm-to-metric matches or typed abstentions for every faithful extracted norm in every admissible corpus/task. Do not mark complete before final exact coverage of all 1,732,515 rows.

## Current continuation state

- The four-hour timer elapsed successfully; the goal remains active and unblocked.
- Three live branches are training/labeling: `train_code_math`, `train_content_tasks`, and `train_peer_legal_nc`.
- Do not change models, frozen manifests/banks, task routing, external splits, or sk3 settings. Exact current-bank metric IDs remain primary; concurrent L0→R3 reclustering is a separately versioned future sensitivity analysis.

## Canonical input

- Remote root: `/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful`
- 1,732,515 rows, 23 corpora, 8 tasks
- Manifest SHA-256: `b614e345a07123f9fe79d9521351886107476d34cf2b09daa50efce71dc1356f`
- Inclusion doctrine: `faithful=1`; retain old extraction validity only as provenance.

## Completed implementation and validation

- Query construction now preserves up to 1,600 characters of evidence and exposes evidence/statement views.
- Retrieval records dense/word/character component ranks and supports dev-selected weighted RRF.
- Added task-specific fusion optimization, GEPA prompt add-ons, independent contrastive verification, order checks, and exact-coverage finalization.
- Added multi-encoder capture audits, complementary 50-metric rescue blocks that exhaust a bank, rescue aggregation/finalist adjudication, independent typed-abstention verification, fail-closed rescue merging, exact false-abstention confidence bounds, and streaming all-corpus decision-rate audits.
- Added an exact shard merger for all Gemma inference stages. It rejects missing,
  duplicate, misrouted, hash-drifted, or runtime-inconsistent shards before
  producing the single metadata-bound artifact required by production audits;
  this makes the 277K–612K-row tasks safely parallelizable across GPUs.
- Added leakage-safe staggered task releases plus a v3-native MI↔silver
  validation harness. It consumes only final audited exact MATCHes, retains
  typed abstentions in the denominator, uses source-group presence as the
  primary salience estimand, reports raw/equal-corpus/polarity sensitivities,
  and carries blind precision/false-abstention bounds into task and
  random-effects summaries.
- Final audit reports micro, equal-task macro, equal-corpus macro, task/corpus ranges, polarity, norm kind, and old-valid/recovered provenance. Scientific reporting contract: `CLAIM_MATRIX.md`.
- Full local suite: **219 passed** at the latest production-hardening checkpoint.
- Full manifest validation passed and the canonical artifact is locked.

## Human calibration panels

All eight old-valid panels are complete (240/task; 1,920 total). Press releases: 181 MATCH, 53 NO_EXPLICIT, 2 GENERIC, 4 NOISE; validated remotely as `silver_match_v3_20260712/teachers/manual_press_releases.validated.jsonl`.

Recovered-faithful panels complete and remotely validated for:

- math: 160; 143 MATCH
- humor: 160; 113 MATCH, 12 FAMILY
- code review: 160; 145 MATCH, 1 FAMILY
- notice-and-comment: 160; 140 MATCH
- press releases: 160; 86 MATCH, 2 FAMILY
- creative writing: 160; 158 MATCH, 1 FAMILY, 1 CONTEXT
- legal: 160; 67 MATCH, 79 NO_EXPLICIT, 9 GAP, 4 GENERIC, 1 CONTEXT

All seven applicable recovered-faithful panels are complete (160/task; 1,120 total). There is no recovered peer-review stratum. The final creative-writing and legal panels were revalidated against faithful manifest SHA `b614e...` and written as `added_calibration/labels/manual_creative_writing.validated.jsonl` and `manual_legal.validated.jsonl`.

## Retriever/adapter evidence

- Base fusion reports are under `faithful/fusion/` for code, humor, math, and notice-and-comment.
- Frozen recovered-faithful test macro recall@50: code .763, humor .492, math .708, notice-and-comment .785. This is insufficient for narrow production slates.
- Context-inference evaluation of the old-query code r2 adapter completed at `added_calibration/candidates_code_nemotron_r2_full.jsonl`; optimized frozen test macro recall@50 was .679, below the .763 base. Reject it for production.
- Humor r1 LoRA completed with `REJECTED_NO_DEV_GAIN`; report at `/lfs/skampere3/0/alexspan/models/silver_match_v3_nemotron_lora_20260712_r1/humor/training_report.json`.
- Math r2 was rejected; code r2's internal old-query gain does not transfer to the corrected query. All old-query adapters are diagnostic only.
- Naive tiny-label peer cross-encoder training was also rejected. Do not promote it.

### Corrected-query r3 context round

- Leakage-safe teachers block every external dev/test source group. Forced top-3 events use rank 1 as a weak positive and all three IDs as nonnegative alternatives.
- Base BGE seven-lane top-50 union on 852 recovered gold MATCH rows: .8885; one-sided 95% miss upper bound .1308.
- Base Nemotron seven-lane union: .9085; BGE + base-Nemotron cross-system union: .9460, miss upper bound .0685. Legal and N&C already clear the <.05 candidate-miss bound; the overall claim does not yet.
- **Promote Notice-and-Comment context LoRA.** External dev selected fusion: @16/.30/.50/.80 = .701/.925/.985/1.000 vs base Nemotron .433/.612/.910/1.000. Frozen test once: .671/.886/.937/1.000. Adapter: `...r3_context/manual_only/notice-and-comment/adapter`; selection: `...r3_context/external_eval/notice-and-comment.dev-selection.json`.
- **Promote Press Releases context LoRA.** External dev pure dense @50 .574→.815 and @80 .685→.963; frozen test once @50 .714→.857 and @80 .857→.946. Dev-only final fusion reaches .889 @50/.944 @80 and is explicitly unretested. Adapter: `...r3_context/adapters_lr2e5_e2/press-releases/adapter`; sealed records under `...r3_context/promotions/`.
- **Reject Creative Writing r3.** External dev @50 .814→.229 due Sonnet corpus imbalance/noisy exact leaves.
- **Reject weak-forced Code and Math r3.** External dev dense @50: code .806→.701; math .843→.786. Dev fusion: code base .910 vs adapter .881; math base .857 vs adapter .786. No external test touched.
- Forced-top3 human-train audit: code rank1 5/8, gold-in-top3 6/8; math rank1 16/46, gold-in-top3 25/46. New code/math teachers must be independently exact-labeled, not Gemma/forced-only.
- Gemma full-bank exact-ID distillation was rejected for peer/legal/N&C gradients (dev exact precision only .83/.67/.62). The 434K v6 pairs were also rejected as gradients: rubric↔rubric, stale/unmapped, missing norm/context/bank/judge provenance. They remain diagnostics only.

## GEPA state

- Round 0/1/2 completed for peer, legal, and notice-and-comment. Dev-selected prompts: peer r1, legal r2, notice-and-comment r1.
- Their frozen tests were run once. Keep them frozen; do not tune on test.
- Legacy K16 scores are not the final production calibration. Re-run final adjudicator GEPA on each dev-selected retriever's compact **top-50 dev slate**, then freeze and touch adjudicator test once.
- Content branch is calibrating order-stable Gemma verification of Sonnet proposals; only exact stable confirmations may become gradient positives. OpenRouter remains allowed only for tiny prompt iteration; never log `~/.openrouter-api-key.txt`.

## Resume sequence

1. Finish independent train-only full-bank labeling for Code, Math, Peer, Legal, and Humor; train high-precision second-round task LoRAs and gate on external dev. Never use weak/Gemma-only exact labels as gradients.
2. Freeze one retriever/fusion per task. Retain base Nemotron/BGE where a task adapter does not clear the dev gate; use multiple encoder systems as diverse rescue lanes.
3. Select final top-50 GEPA prompts on dev, then run frozen adjudicator tests once.
4. Production-shard primary retrieval/adjudication across all 23 corpora. For provisional abstentions, run at least two complementary full-bank captures with shifted system/lane partitions and re-include the primary K; assert every frozen metric was re-exposed exactly twice.
5. Contrastively adjudicate rescue proposals; independently verify exhaustive no-match types. Any `POSSIBLE_EXACT_BANK_MATCH`, invalid, or low-confidence result remains unresolved and must be subagent-labeled—not silently abstained.
6. Re-run all MATCH decisions in swapped order and contrastively verify. Finalize exact matches/typed abstentions and validate all 1,732,515 rows.
7. Claim <5% false-abstention risk only if the blind audit's one-sided 95% exact upper bound is <.05. Then run the all-three-label-type downstream external-validity/meta-analysis specified in `CLAIM_MATRIX.md`.

## Production progress

- Production root is append-only under `faithful/production_v1/`.
- N&C primary K50 retrieval is complete and exact-audited for both corpora:
  `notice_and_comment` 5,176/5,176 and `nc_public_comments` 15,870/15,870.
  Both use the promoted task LoRA, dev-selected fusion SHA `d53df0...`, exact
  bank SHA `67205f...`, and canonical manifest SHA `b614e3...`.
- Press-releases primary K50 retrieval is complete and exact-audited at
  26,996/26,996 using the promoted task LoRA and dev-fusion SHA `2af2de...`.
  The first launch failed before writing rows because AFS was selected for the
  transformers dynamic-module cache; its failure log is preserved.  Production
  retrieval now pins writable HF/module/XDG caches, and the clean retry sealed.
- Candidate-stage audits now fail closed on exact UID/row coverage, duplicates,
  bank membership/hash, contiguous ranks, adapter hashes, dev-fusion provenance,
  and candidate/meta file hashes (`audit_candidate_outputs.py`).
- Gemma adjudication now streams prompt batches instead of materializing a
  corpus-sized prompt list and deduplicates only byte-identical prompt/card
  requests, recording the representative UID and item-prompt hash per row.
- N&C final K50 GEPA selection is frozen at prompt SHA `c839a2...`: dev
  original/hashed exact-ID `.642/.642`, MATCH P/R `.969/.925`, exact order
  agreement `.857`.  Its one-shot two-order frozen adjudicator test marker was
  atomically reserved before labels were opened and later sealed as reported below.
- The frozen N&C K50 adjudicator test subsequently sealed at n=85. Retriever
  capture was 74/79=.937, but strict two-order consensus exact-ID precision was
  only 29/53=.547. This **fails** the production exact-leaf gate. The test is
  closed permanently; the retriever remains promoted, while exact `MATCH`
  requires a new precision-first contrastive verifier selected only on external
  dev. Unresolved/rejected rows flow to exhaustive rescue/family-only output.
- The optional clean task-specific cross-encoder now requires minimum retained
  support and Wilson lower-bound precision on dev and frozen test in addition
  to point precision and dev gain; tiny lucky retains cannot promote it.
- Strict production finalization now requires task-matched dev selection
  records for both adjudicator and verifier, complete original+hashed coverage,
  identical model/prompt/candidate sets across orders, exact selected prompt
  hashes and candidate depth, verifier primary-prompt/bank provenance, and
  complete verification coverage of every primary MATCH.
- Strict exhaustive-rescue merge applies the same standard to rescued exact
  leaves: selected original+hashed adjudicator prompts, identical finalist
  slates/model/bank, and selected contrastive confirmation are mandatory.
  A finalist abstention cannot be accepted from one pass; it is unresolved
  until independently typed/audited.
- Sharded JSONL combination is now append-only and records every input count
  and SHA-256 while rejecting missing/duplicate keys. Production retriever
  metadata also pins the canonical manifest hash directly (older completed
  N&C/PR outputs are equivalently pinned by their immutable audit reports).
- The downstream three-label-type design, canonical outcome paths, known
  missingness restrictions, and unresolved task inventories are recorded in
  `DOWNSTREAM_OUTCOME_CONTRACT.md`; outcome labels remain unavailable to the
  matcher until its assignments freeze.
- N&C's second, dev-only shepherded contrastive verifier now clears the
  precision-first gate without reopening test: strict original+hashed/high
  consensus retained 27 proposals, 26 exact (`.963`; Wilson lower `.817`), with
  `.667` recall of correct proposals. Policy SHA `11da2e40...`, selection SHA
  `f1567f0e...`; production still requires a blind final-MATCH precision audit
  because this verifier could not receive another frozen-test evaluation.
- Deterministic uniform blind audit builders now sample final MATCHes and
  abstentions both globally and independently per task. System outcomes are
  hidden in separate keys. Exact one-sided bounds cover both the `<5%`
  false-abstention claim and any claimed final exact-MATCH precision floor.
- First-pass full-bank strong-model labels for Peer and Legal failed the exact
  teacher gate (anchor exact leaf `.500` and `.567` respectively). They remain
  excluded from gradients; hidden-ID/reshuffled-bank second passes and strict
  consensus calibration are in progress.
- N&C's rendering-bound production plan is frozen at SHA `bfc26a...`; it binds
  both corpora (21,046 rows), candidate/retriever selections, exact dev-tested
  prompt rendering, Gemma snapshot, prompt components, verifier policy, and
  inference implementation hashes.  Two-order primary adjudication is live on
  GPU0, followed by an automated exact audit, two-order strict verifier, and
  per-corpus pre-rescue finalization.
- Production verification now combines original+hashed rows only when both are
  high-confidence confirmations of the same exact ID.  Strict finalization
  requires the combined row to carry the frozen selection, policy, and complete
  production-plan hashes; a single verifier pass cannot emit `MATCH`.
- N&C full-bank adapter and base-Nemotron rankings are being materialized at
  K=88 for every row.  Rescue supports repeated full-bank captures that
  re-include the primary K, shifted retrieval-system/lane partitions, exact
  exposure-count assertions, and capture–recapture proposal-discovery
  diagnostics.  The `<5%` false-abstention claim still requires the blind exact
  binomial audit.
- N&C now has exact-audited full-bank K=88 rankings from three complementary
  systems: promoted task adapter, pretrained Nemotron, and BGE.  Both corpora
  pass exact canonical coverage, complete-bank membership/ranks, model/fusion,
  manifest, and artifact-hash audits for all three systems.
- Final typed abstentions are also fail-closed across original+hashed trial-
  summary orders.  Only two high-confidence identical typed decisions after
  repeated full-bank rescue can merge automatically; any possible exact match,
  disagreement, low confidence, or parse failure remains unresolved for an
  independent blind label.
- The automated N&C rescue runner is frozen at SHA `85590bcb...`; it pins
  adjudicator implementation `66e5bd7f...` and verifier implementation
  `797e6ade...` before every inference stage. If strict merge finds any
  unresolved rows, it writes a complete reason ledger and creates shuffled
  full-bank blind labeling chunks rather than emitting a final file.
- Primary parser failures remain explicit `INVALID_OUTPUT` rows during the
  exact two-order audit and are automatically eligible for repeated full-bank
  rescue. They are not force-parsed from truncated text or allowed to block
  canonical coverage; final unresolved invalids still require a blind label.
- Code r4 is promoted on external dev: dense R@50 `.806→.955`, R@80
  `.940→.985`; frozen dev fusion reaches `1.000/1.000` at K50/K80 with test
  untouched. Math r4 fails its K50 gain gate (`.843→.857`, +1.43pp), is
  diagnostic-only, and the frozen unadapted base+dev fusion is retained.
- Peer and Legal both rejected their first independent-label promotions and
  final GEPA verifier attempts on precision/support uncertainty. Promotion
  outputs are empty and no rows entered gradients. Fresh non-overlapping,
  bank-boundary-enriched 600-row train-only packs are undergoing independent
  hidden-order exact labeling; thresholds remain unchanged.
- A fresh human-only Humor Nemotron LoRA now clears external dev: dense R@50
  `.604→.717`; its dev-selected adapter+word fusion has macro R@50 `.750`
  versus prior base fusion `.614`, with no R80 loss. Selection SHA
  `cac2b796...`, fusion SHA `ec0c1435...`, adapter weights SHA `36017fdf...`;
  Humor test remains untouched. A separate clean CW LoRA is now training.
- Fresh Peer three-pass labeling froze 169 strict all-high exact proposals.
  Sixty balanced rows/25 metrics/60 source groups are permanently held out for
  the fourth blind audit. The audit sealed at 60/60 exact: raw precision 1.000,
  Wilson lower `.93983`; design-weighted precision 1.000, approximate lower
  `.92825`. All 60 audit rows remain excluded. The 109 non-audit teachers
  (SHA `600e27ae...`) combined with 75 disjoint manual teachers into 184 unique
  train groups/30 metrics (SHA `ba3c22c1...`). The task LoRA is queued, not
  running, under frozen saturated-R50 policy `994246fd...`; external dev/test
  remain untouched by training and the pretrained fusion is the fail-closed
  fallback.
- Peer retriever training is now resolved rather than queued. The 184-row
  audited-teacher LoRA selected epoch 0 internally and tied all 26 external-dev
  fused ranks; saturated-R50 policy therefore rejected it and retained the
  pretrained Nemotron fusion (`0695fedb...`). External test is untouched.
  Production retrieval was priority-paused at 8,448/277,420 valid unique K50
  rows, partial SHA `6fc26683...`; exact resume pins/command are recorded in
  adjacent pause record `b0ee3e2b...`.
- Legal fresh three-pass labeling completed 600/600. Strict all-three-high
  consensus retained 69 MATCHes; the independent balanced fourth audit found
  57/60 exact (raw Wilson lower `.86299`, design-weighted lower `.85564`). All
  60 audit rows are permanently excluded and 9 disjoint teachers were promoted
  (`f8029e8a...`). LoRA/retrieval work is deferred behind the 12-hour N&C
  analysis-release priority.
- N&C original-order production adjudication sealed 21,046/21,046 at output
  SHA `7f30b4eb...`, selected prompt `c839a28c...`, with two explicit fail-closed
  parser-invalid rows. Hashed-order inference remains live. The waiting
  continuation now runs the two frozen verifier orders concurrently on the two
  available coordinated GPUs, then the unchanged strict combine/finalize
  (`245f681f...`). GPU0 was claimed by an unrelated job after primary release;
  hashed verifier runs on GPU5 and the unchanged original verifier was moved
  to explicitly released GPU7. A post-meta watcher prevents any combine before
  both order artifacts seal.
  Rescue similarly parallelizes only disjoint frozen trial/order artifacts
  while preserving every aggregate/strict merge/audit gate (`5ba1304f...`).
  Rescue uses verifier GPUs 5/7 only after each reports at least 170,000 MiB
  free, preventing both unrelated-job preemption and EngineCore teardown races.
- Strict rescue unresolved rows now have a complete fail-closed closure path.
  The generated standalone label workspace excludes all hidden system keys and
  reasons (`e5f0e22c...`). A rerun may consume independent labels only when they
  exactly cover the recomputed frozen UID/reason ledger, link to that immutable
  pack, use the current bank, validate complete, and are high-confidence
  (`8181d2a6...`). Automatic rescue acceptance logic is unchanged.
- Final N&C MATCH/abstention task samples will be independently labeled as soon
  as rescue seals. `prepare_final_decision_label_pack.py` (`288d4cdd...`) reads
  only declared blind samples, frozen banks, and canonical norms; it verifies
  hashes/no leaked decisions, never opens system keys, and emits immutable
  25-row label chunks. Join both validated audit samples against the exact final
  files with `audit_false_abstentions`, then freeze the task analysis release.
  The pinned N&C MI certificate is `4dacd71d...` (18/88 exact unique-name
  joins). MI validator `7fbf38eb...` enforces certificate task/schema identity,
  uses source-presence as the predeclared primary estimand, and will run 2,000
  permutations, 1,000 source-group bootstraps, and 200 stratified split-halves.
- At 2026-07-12 15:48 PDT, explicit user instruction released N&C GPUs 5/7 and
  disabled its GPU launchers.  Both devices were verified at 4 MiB used,
  182,628 MiB free, and 0% utilization before handoff.  Resume-safe verifier
  partials are 13,440/17,473 original-order rows, SHA
  `12677a76c2f7d703238c169be38db91bae645ae25a57044b3a505728332cdd39`, and
  13,952/17,473 hashed-order rows, SHA
  `f9f3dfc19679020ad12447f112946559c20f50cb760acd966961f3d1a32daa10`.
  Neither completion meta exists.  The post-meta strict-combine launcher,
  rescue launcher, and local watcher capable of rescue relaunch were stopped;
  non-GPU pack/release watchers remain inert while waiting for a final audit.
  Do not resume any N&C GPU stage without a new explicit instruction.
- At 2026-07-12 16:13 PDT, the user explicitly authorized continued polling
  and claiming a slot only when the server-wide occupied-GPU count is below
  four. `run_notice_quota_slot.sh` requires six consecutive safe polls and
  attaches `guard_gpu_quota.sh`; `run_notice_quota_chain.sh` serializes hashed
  then original and retries only quota-preempted runs. A GPU0 hashed resume was
  safely launched after a 3/4 window, then stopped by the guard when another
  user's multi-GPU reservation raised the server total to eight. It emitted no
  new rows, so both partial counts and hashes above remain unchanged. The
  chain is still polling with zero owned GPUs and will resume only under the
  same cap; after both metadata files seal it invokes only the unchanged
  pinned combine/finalize path.

## Completed-agent reports

- Press-releases old-valid raw SHA: `265c0a2243a9ca11b5759e2633ef9b44f34008ec4e2d888b3427c5669ab7b6e1`.
- Creative-writing recovered raw SHA: `e4d7e9f9dfee45a3920800f47cf367857f8089440c75ead5147aab5f1c6b0e95`.
- Legal recovered raw SHA: `1df3cfb74a1826200c1a620e06220a90cf351556a3cb792c9fd450c6cd1a2190`.
- All three panels passed exact coverage, schema, bank-membership, and provenance validation; no agent reported a blocker.
