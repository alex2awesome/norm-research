# L0→R3 concept-hierarchy reconstruction — pipeline of record

Consolidated description of the *current* process that reconstructs the metric-statement hierarchy
(L0 same-criterion → R1 same-construct → R2 same-theme → R3 same-category) with measured
recall/precision at every level, per task. This is the method description; the two living records are:

- **Runbook + chronological results:** `notes/2026-07-06__hierarchy-reconstruction-ledger.md` (read
  it before resuming — every improvised deviation so far has lost to it).
- **Discipline/rules (do & don't):** `BEST-PRACTICES.md` → *"hierarchy L0→R3 relabeling (codability)"*.

Engine: `methods/codability/lexicon/build_level.py` (R-levels), `repair.py` (L0), `level_naming.py`
(group naming between levels). All artifacts live in `outputs/lexicon/`. Reconstruction-only: judges
PARTITION, never see labels; no human subjects.

---

## 0. Data model

- **Node at level L** = a *named group* from level L−1. L0 nodes are the deduped rubric criteria;
  R1 nodes are L0 clusters; R2 nodes are R1 groups; etc. `nodes_from_level(task, level)` returns
  `[{node_id, name, gloss}]` and composes membership down to the original L0 keys for later census.
- **Relation per level** (`RELATIONS` in `build_level.py`), each strictly broader than the last:
  - **L0** = *same criterion* — would give the same verdict on real work; differ only in wording.
  - **R1** = *same construct* — facets of ONE underlying quality; a reader treats them as one dimension.
  - **R2** = *same theme* — belong to one broad evaluative area/family.
  - **R3** = *same category* — top-level subsumption.
- **Truth** at each level = a frozen eval set of node-pairs, labeled 0/1/2 by an **arbiter** (single
  independent-family frontier pass; the arbiter *defines* truth at R-levels). `pair_id = sha1(sorted
  node keys joined by "||")[:16]`.

## 1. Stage order (load-bearing)

```
P0 frozen eval  →  P1 arbiter truth  →  P2 L0 repair (loop-until-dry)  →  L0 freeze + rename
   →  R1 (build → apply → score → name)  →  R2  →  R3
```
Building an R-level on *unrepaired* L0 measures the repair debt, not the relation. Each level prints
BOTH recall and precision; neither may silently sag (L0 repair is a deliberate recall-ward move whose
precision cost compounds up-tree).

## 2. L0 build + repair (`repair.py`)

L0 starts from the deduped rubric partition and is *repaired* (under-merge recovery) to dryness:

1. **Candidate net** — union of three signals so paraphrases aren't missed:
   `TF-IDF(min_df=2) ∪ shared-name ∪ v6-BGE semantic`. Measured ceiling 1.0 on adjudicated eval.
2. **Screen** (permissive: keep at ≥1) → **Confirm** (independent LLM, strict `==2`, frozen
   `CONFIRM_PROTOCOL_L0_V2.txt` = the L0 "same criterion" relation) → **≥2-edge STAR merge**.
   Historical repairs used `CONFIRM_PROTOCOL_R1.txt`, whose wording was the broader R1
   "same construct" relation despite earlier notes calling it L0. Preserve those artifacts but do
   not reuse that prompt; the cross-task L0 coherence audit measures the resulting precision debt.
3. **STAR merge invariant** — a head must never also become a tail (else A→B, B→C chains leave A→B
   un-realized). `apply_merges` locks BOTH tails and heads (`heads` set); **union-find is forbidden**
   (single-bridge chaining). Vote parsing is strict-int (`type(s) is int and s in (0,1,2)` — bool is
   an int subclass, so `True==1` would otherwise slip through).
4. **Widen-to-dryness** — sweep candidate bands (rank ~2500→8000, deeper if still wet) until yield
   dries; miss-by-band triage first; net-vintage parity. Only then freeze.
5. Write `partition_<task>_L0vN.json` (NEW file each version; never overwrite), `score_vs_truth`,
   ledger row. Applying a new L0vN does NOT rebuild R1/R2/R3 — `apply_merges` reuses existing cluster
   ids, so merged nodes inherit their parents by composition; upper levels just GROW → a cheap rescore.

**Exception — precision regroup/splits:** `l0_regroup.py` can split an overbroad L0v3 cluster back
into strict same-criterion groups of its v6 source clusters. A split changes the parent node inventory;
R1 and all descendants MUST be rebuilt against the new manifest-frozen L0v4. Never use the cheap
composition rule for splits.

## 3. R-level build (`build_level.py`) — the ONLY validated R-build

For each level R1→R3, in order:

1. **`eval-emit`** → `emit_level_eval` writes ~900 frozen node-pair truth candidates
   (`level_eval_<task>_<level>.jsonl`).
2. **`arb-emit`** → `emit_arbiter_payloads` chunks the eval pairs into arbiter shards
   (`level_arbiter/<task>_<level>_NNN.jsonl`). The **arbiter fleet** (Sonnet-or-better, `ARBITER_
   PROTOCOL_<level>` = that level's relation) writes `level_votes/arb_<task>_<level>_NNN.jsonl`.
   No anchors here — the arbiter *is* the truth definer.
3. **`emit_verify_net`** — build (dedup) candidate net:
   - Global **TF-IDF kNN** (k=20, cosine) over node `rep_text = "name. gloss"`. *(Not an embedding
     model — see §7; for these short jargon-dense strings TF-IDF beats BGE.)*
   - **EXCLUDE the eval pairs** from the build net (verify & arbiter are the same Sonnet-5 family
     since 2026-07-07; overlapping them is correlated-error optimism, not measurement).
   - Rank by cosine, take top `cap` (band). Report `topband_ceiling_vs_eval_same` (recall ceiling).
   - Ride-along **blinded QC anchors** (a few arbiter score==2 and score==0 eval pairs), persisted to
     `level_anchor_ids_<task>_<level>.json`. **⚠ DESTRUCTIVE**: it deletes+rewrites the verify payload
     shards. Never call it for diagnostics — use a read-only ceiling calc.
4. **Verify fleet** (Sonnet, TERSE score-only, ~300–400 pairs/shard) → `level_votes/vrf_<task>_
   <level>_NNN.jsonl`.
5. **`apply_pairwise`** — fail-closed, in order:
   - **Completeness check**: every non-anchor net pair must have ≥1 well-formed vote (by pair_id, not
     shard count) → else `IncompleteVotesError`.
   - **Anchor gate**: positive-anchor accuracy ≥ 0.8, negative ≥ 0.9 → else `AnchorGateFailure`.
     Anchors never contribute a build edge regardless of outcome. `exclude_from_gate={pids}` drops a
     *confirmed-bad-gold* anchor from the gate math ONLY (it stays edge-blocked) — document every use.
   - **Louvain** community detection (`res=1.0`) on the verified-SAME graph. **No fallback** (a silent
     connected-components fallback once rotted R1 precision .80→.43 undetected). The default
     `related_weight=0` exactly preserves this hard-edge graph. A preregistered positive value may
     use LLM score-1 RELATED pairs as weaker weighted structure evidence, never as SAME labels; that
     candidate still requires the independent uniform LLM precision audit and every >30-member
     whole-group certification. Write `partition_<task>_<level>.json`.
6. **`score`** → recall/precision vs arbiter truth on the FULL node set (never edges-only — that
   inflates the number).
7. **`level_naming`**: `emit_group_names` (payloads of each multi-member group's members) → Sonnet
   name fleet → `ingest_group_names` writes `node_names_<task>_<level>.json` (singletons inherit their
   member's name). Required before the next level can build.

**No confirm stage at R-levels** — the R1 bridge-confirm judged bridges at ≈chance (60% anchor) and
killed 94% of them (recall .673→.537 flat precision); REVERTED. (The L0 screen→confirm is a different,
sanctioned confirm — don't conflate.)

## 4. The two gates (measured at EVERY level)

- **Recall gate = net/bucket ceiling ≥ ~0.9** = `P(same-band | arbiter-SAME)`: what the candidate net
  CAPS recall at. Usually a *cap* artifact before a *net-type* problem — raise the cap/width first and
  re-measure (net diffuseness grows up-tree). E.g. CW R1 `.675@cap9000 → .946@full`.
- **Precision gate = verify/split pass** — arbiter re-judges within-group member pairs and splits out
  non-SAME. Applied whenever score precision sags.

## 5. Judge placement & throughput (fixed by measurement, not cost)

- Sonnet-or-better judges R-levels; **GLM is L0-only** (over-merges R1: 46% SAME → precision .78→.60),
  run with the ≥2-edge gate. Arbiter = an independent-family frontier model told its ruling IS truth.
- **Blinded anchors in every judging batch** are the tripwire — every failure caught so far (resumed-
  judge drift; 60%-anchor confirm) was caught by anchors. Use the 0/1/2 scale + calibrated exemplars.
- TERSE output (score only) validated at ~400 pairs/shard; consolidated multi-file agents hit a 64k
  output ceiling → 1 file/agent, no reasoning field.
- **JUDGE-AGENT EFFICIENCY (2026-07-11) — go straight to scoring, no overhead.** Observed waste: agents
  spending 20-40 tool calls on self-validation, re-reading, cross-checking pair_ids, and even spawning
  their own subagents (one caused a shard content-mixup). Standard judge prompt now mandates: (a) do the
  work YOURSELF — never spawn/delegate to subagents; (b) exactly ONE Read per input file and ONE Write
  per output file; (c) process files ONE AT A TIME (read file → judge all lines → write file → next),
  never hold multiple files' content together (juggling caused the mixup); (d) do NOT re-read, validate,
  verify, count, or cross-check your output — the orchestrator runs the completeness + per-shard 0-fraction
  distribution checks downstream, so agent self-QC is pure wasted latency; (e) no repo search/exploration;
  (f) snap-judge each pair, terse {pair_id,score} only. This cut ~4-5× the tool calls per shard.
  BIGGER lever (if 2-3× isn't enough): the ledger's sanctioned speedup = local Gemma/Llama offline-batch
  vLLM on sk3 SCREENS the net (drops the obvious 0s, ~50-70% of pairs), Claude judges only screen-positives
  → ~2-3× Claude cut; keep Sonnet as the actual R-level judge (local models over-merge at R1).

## 6. Reproduce (news R1 example)

```bash
PYTHONPATH=. python3 -m methods.codability.lexicon.build_level eval-emit --task news-homepages --level R1
PYTHONPATH=. python3 -m methods.codability.lexicon.build_level arb-emit  --task news-homepages --level R1
#   → arbiter fleet judges level_arbiter/news-homepages_R1_NNN.jsonl → level_votes/arb_...
PYTHONPATH=. python3 -c "from methods.codability.lexicon import build_level as b; b.emit_verify_net('news-homepages','R1')"
#   → verify fleet judges level_arbiter/news-homepages_R1_verify_NNN.jsonl → level_votes/vrf_...
PYTHONPATH=. python3 -c "from methods.codability.lexicon import build_level as b, json; print(b.apply_pairwise('news-homepages','R1', exclude_from_gate={'ec28b299f4a77434'}))"  # NOTE: anchor exclusions are REQUIRED to reproduce stored partitions where a generous-arbiter gold anchor tripped the gate (documented per use in the ledger): news R1 {ec28b299f4a77434}; math R1 {7cf850530f358a75, efff06cae9aeff08}; math R3 {52b039191ba50b7d}. Without them apply_pairwise raises AnchorGateFailure (working as designed).
PYTHONPATH=. python3 -m methods.codability.lexicon.build_level score --task news-homepages --level R1
#   then level_naming.emit_group_names / ingest_group_names before R2.
```

## 7. Empirical findings that shape interpretation (2026-07-10)

- **R-level arbiter truth is systematically GENEROUS (over-calls SAME).** Cross-family blind
  validation (Codex gpt-5.6-sol, a different family, 120 held-out pairs/task) agrees with the single-
  pass Sonnet arbiter's "same construct = 2" only **28% (news) / 33% (math)** of the time; the verify
  fleet shows the same strictness. ⇒ **low R-level recall is deflated by truth generosity, not (only)
  build failure**; precision stays high because what the strict build merges is clean. A de-noised
  truth (multi-pass / cross-family arbiter) would raise apparent recall — a methodology change that
  needs sign-off. Read every R-recall with this caveat.
- **Recall is net-band-capped as node count grows.** The recall ceiling ≈ `f(#nodes, cap)`. math R1
  (2893 nodes, full net 27,826) at cap=9000 ceils recall at **0.623**; widening cap→21,000 lifts it to
  **0.891** (~40 more verify shards). news R1 (1225 nodes) at cap=9000 already ceils at 0.926. Log the
  cap (no silent caps).
- **TF-IDF ≥ embeddings for these node reps.** BGE-large kNN on math R1 nodes reached only 0.52 full-
  net ceiling vs TF-IDF's 0.913 (and adds 1/138 truth-SAME pairs TF-IDF misses): short jargon-dense
  criteria collapse into a generic neighborhood under a general-English embedder, losing the
  discriminating rare terms TF-IDF keys on. Only add embedding nets if TF-IDF *measurably* decays.

## 8. Results ledger (recall / precision, band cap=9000 unless noted)

| task | L0v3 | R1 | R2 | R3 |
|---|---|---|---|---|
| news-homepages | .913 / .587 | .477 / .583 | .176 / .417 | .753 / .732 |
| math-stackexchange | .881 / .742 | .438 / .750 | .365 / .547 | .746 / .815 |
| humor | ~.86 | .673 / .689 | .425 / .703 | .649 / .689 (anchored) |
| creative-writing | ~.90 | .620 / .678 | .384 / .682 | .729 / .670 (anchored) |
| peer-review | .850 / .748 | .587 / .677 | .259 / .658 | .756 / .749 (classify-derive) |

**Chance-corrected recall** (historical key `recall_kappa`, retained only as a deprecated alias) —
peer-review: L0v3 .850 → R1 **.576** → R2 **.241** → R3 **.691**. This is
`(recall-p0)/(1-p0)`, NOT Cohen's κ; `score()` now reports actual binary Cohen κ separately plus
Wilson uncertainty and high-sim/random eval strata.

**R2 remains a measured trough, but its cause is not identified by the old ceiling diagnostic.** The
eval set is deliberately TF-IDF enriched and the candidate net uses the same representation. For
peer-review R2, the `.829` aggregate ceiling consists of 157/157 high-similarity positives but only
3/36 random positives. Therefore it does not establish `.829` population findability. The candidate
graph has 1,103 edges (avg degree 5.7, 43 isolates); the LLM verifier retains 291 (avg degree 1.5,
141 isolates). Loss is a mixture of candidate retrieval, conservative LLM verification, forced
disjoint partitioning, and Louvain fragmentation. `recall/ceiling` is not a clean stage-conditional
probability because held-out eval pairs never contribute build edges.

**Fresh blind R2 cross-family audit (2026-07-12):** 120 pairs/task, balanced 40/40/40 over Sonnet
scores 0/1/2; independent Codex-family LLM judges saw no Sonnet labels. `P(Codex SAME | Sonnet SAME)`:
math `.525`, creative-writing `.800`, humor `.850`, news `.775`, peer-review `.850`. Binary SAME/not
agreement: `.758/.892/.858/.758/.850`; binary Cohen κ on this deliberately balanced diagnostic sample:
`.424/.752/.691/.491/.675`. Thus R2 positives are reproducible in CW/humor/peer, moderately so in news,
and materially unstable in math. These replace the older nonuniform `.925/.75` shorthand; frozen
payload manifests and complete reports live under `outputs/lexicon/codex_val/*_R2_*reaudit*`.

**Interpretation discipline:** the R1→R2 decline uses the same code family but not a measurement-
invariant instrument: graph exposure, semantic representations, relation transitivity, and verified
edge density all change. R3 classify-derive also avoids fragmentation and its anchored arbiter shares
the proposed taxonomy. Report the observed R2 trough, but do not call it an intrinsic property of
themes until an independent LLM-proposed candidate comparison and an evaluation sample not selected by
the build representation reproduce it. Themes may also overlap; Louvain's disjoint equivalence
partition is itself a hypothesis to test.
Full, current numbers live in the ledger. Update BOTH this table and the ledger when a cell lands.
