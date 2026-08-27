# Sibling-subtask metric lattice: common vs invented (task #66) — 2026-07-09

User ask: for a few similar sibling subtasks, count (a) metrics in common and (b) new metrics
that need inventing; highlight PEER REVIEW subfields; and check the GENERAL tasks with the same
(metric-tree) machinery — do they want new metrics too?

## Framework (two layers)
- **IN COMMON = shared vocabulary**: bank metrics that carry within-sibling univariate signal
  (AUC≥.55) in ≥2 siblings. GPU re-score (`bank_liveness.py`). The bank is the SAME per domain.
- **IN COMMON = shared novelty**: invented metrics that RECUR across siblings (embedding dedup).
- **NEW TO INVENT**: invented metrics (GLM-proposed, gate-surviving) UNIQUE to one sibling;
  of those, how many survive disjoint replication. Harvest from existing ledgers
  (`sibling_lattice.py`), both arms, GLM legs only.

## Result 1 — GENERAL tasks do NOT want new metrics (dilution, cleanest form)
Pooled/general leg per domain, SAME gate as subtasks:
| domain | proposals | kept |
|---|---|---|
| math-pooled-12tags (GLM) | 23 | **0** |
| humor-topic-pooled-4topics | 21 | **0** |
| cw-genre-pooled-4genres | 22 | **0** |
The general task proposes ~2 dozen candidate metrics and the gate kills ALL. The general bank
suffices; novelty is entirely a SUBTASK phenomenon. Direct answer to "run metric-tree on the
general task — does it want new metrics?" → no.

## Result 2 — invented metrics are sibling-LOCAL (even among close siblings)
`sibling_lattice.py`, GLM legs, hot tails = kept OR confirm p<.05 & bits>0. dup=cos>.86,
theme=cos>.55; name-only permissive pass at 0.45 as a cross-check.
| set | sibs | invented(raw) | theme-shared | theme-unique | unique-replicated |
|---|---|---|---|---|---|
| math-analysis | 5 | 3 | 1(loose) | 1 | 1 |
| math-algebra | 3 | 2 | 0 | 2 | 0 |
| math-discrete-geom | 4 | 7 | 0 | 5 | 0 |
| humor-scenario | 5 | 14 | 0 | 13 | (several) |
| humor-wordplay | 3 | 1 | 0 | 1 | 0 |
| cw-scifi | 3 | 2 | 0 | 2 | 0 |
| cw-fate-deal | 3 | 5 | 0 | 4 | 0 |
Only humor shows loose cross-sibling families (name-only cos): a "punchline/resolution" family
(police↔marriage↔observational) and a "setup-logic" family (police↔marriage↔family), each
INSTANTIATED differently per topic. CW & math siblings share ~nothing. So invented metrics do
NOT transfer across siblings — each subtask needs its own (~1 replicated keep/sibling that
survives). CAVEAT: per-sibling invented n is small (0-5), so "0 shared" is partly power-limited
(a shared latent family may only clear the gate in one sibling per run) — the name-only pass
mitigates but does not eliminate this.

## HEADLINE (all legs in, 2026-07-09 17:00)
1. **General/pooled tasks want 0 new metrics — now 4/4 domains incl. peer.** math / humor / cw /
   peer-iclr-general each propose ~22 candidates; the gate kills ALL. Answer to "run metric-tree
   on the general task, does it want new metrics?" → **no, in every domain.**
2. **In math/humor/CW invented metrics are sibling-LOCAL** (invRecur=0 at every level, dup=0).
   Each subtask invents its own; they do not transfer to siblings.
3. **Peer review is the ONLY set where invented metrics RECUR across subfields** — and all three
   recurring families are the SAME idea: theoretical/mathematical rigor (see Result 3).
4. **Peer bank is the deadest** (5/40 live in ≥2 subfields, 0 common to all 6) yet also the one
   that needs a NEW metric invented → ICLR subfields are more heterogeneous than its bank assumes.

Consolidated (`consolidate_lattice.py` → outputs/ctree/lattice/consolidated.json):
| sibling set | sibs | bankLive≥2 | commonAll | sibSpecBank | invUnique | invRecur | invRepl |
|---|---|---|---|---|---|---|---|
| math-analysis | 5 | 13 | 3 | 4 | 1 | **1** | 1 |
| math-algebra | 3 | 10 | 0 | 11 | 2 | 0 | 0 |
| math-discrete-geom | 4 | 12 | 2 | 13 | 5 | 0 | 2 |
| humor-scenario | 5 | 28 | 9 | 4 | 13 | 0 | 6 |
| humor-wordplay | 3 | 19 | 9 | 5 | 1 | 0 | 1 |
| cw-scifi | 3 | 6 | 1 | 6 | 2 | 0 | 1 |
| cw-fate-deal | 3 | 5 | 2 | 14 | 4 | 0 | 1 |
| **peer-iclr** | 6 | 5 | 0 | 2 | 7 | **3** | 0 |
bankLive≥2 = bank metrics carrying within-sibling AUC≥.55 in ≥2 siblings (metrics IN COMMON);
invUnique = invented metrics unique to one sibling (NEW to invent); invRecur = invented families
shared across ≥2 siblings; invRepl = invented names that reached stage-2 KEPT/p<.05.

## Result 3 — PEER REVIEW subfields (DONE, 7 legs)
Topic-modeled 28,402 ICLR abstracts (venue fixed) into 6 subfields + general. Stage-1 hot-tail
harvest (`peek_peer.py`):
| subfield | proposals | hot tails | what got invented |
|---|---|---|---|
| neural | 22 | **7** | theoretical proof / unification / failure-mechanism rectification |
| graph | 23 | **3** | theoretical-analytical insight / non-incremental methodology |
| image | 23 | **2** | mathematical formalism & proofs / counterintuitive insight |
| policy | 24 | 1 | algorithmic naming convention (weak, p=.039) |
| adversarial | 22 | 0 | — |
| language | 22 | 0 | — |
| **general-iclr** | 22 | **0** | — (pooled control: bank suffices) |
The theory-heavy subfields (neural/graph/image) independently surface a **"theoretical depth /
mathematical rigor / non-incremental insight"** norm that the 40-metric general bank does NOT
capture; the empirical subfields (adversarial/language) do not. This norm RECURS: sibling_lattice
found 3 thematic families shared across ≥2 of {neural,graph,image} (theme cos>.55; dup=0, so
thematic not literal). This is the OPPOSITE of math/humor/CW where invented metrics are local.
CAVEAT: peer recurrence is stage-1 in-run gate, not disjoint replication (invRepl=0 so far);
bits gains small (+0.002–0.010b). Worth a stage-2 pass on the neural theory-rigor metric.

## Result 3-OLD — PEER REVIEW subfields (superseded; kept for the subfield table)
Peer-review data has no subfield column (only venue×domain). Held venue fixed to ICLR (venue is
a base-rate confound per project_peer_review_va) and topic-modeled 28,402 ICLR abstracts into 6
subfields (`build_iclr_subfields.py`: MiniLM→KMeans k=6→TF-IDF names). Subfields (venue fixed →
isolates subfield; note the real base-rate spread):
| subfield | n | accept | terms |
|---|---|---|---|
| neural | 7406 | .385 | neural, networks, training, architecture |
| language | 5005 | .427 | language, llms, models, tasks |
| policy | 4099 | .394 | policy, rl, agent, reinforcement learning |
| adversarial | 4448 | .346 | adversarial, attacks, robustness |
| image | 5267 | .436 | image, visual, video |
| graph | 2177 | .344 | graph, node, gnns |
Running wave-3 recipe on each subfield + general ICLR (peer-iclr-general) → same
common/invented decomposition.
Registered in run.py as `peer-iclr-<slug>` / `peer-iclr-general` (programmatic glob of
by_subfield/). Bank = peer-review medoid-bank (40 rubrics, ICLR V/A baseline ~0.61).

## Result 4 — bank-commonality (DONE)
`bank_liveness.py` scored each domain's bank on 600 stable-hash items/sibling → per-metric
univariate AUC → common-all / common≥2 / sibling-specific (see HEADLINE table, bankLive≥2 col).
Humor banks most alive (28/19 metrics live in ≥2 siblings, 9 common to all); peer bank deadest
(5/40 in ≥2, 0 common to all 6). So "metrics IN COMMON" is large for humor, thin for peer —
and peer is precisely the one that also wants a new metric invented (double heterogeneity signal).

## SCALE-OUT WAVE (queued 2026-07-09 23:41, task #66) — 4 new sibling axes
User ask: scale to more metadata stratifications across more tasks + start topic-diverse (not just
metadata) subsections. Two axes × two types, identical wave-3 recipe + bank-liveness + lattice:
| task | axis | type | siblings | bank |
|---|---|---|---|---|
| peer-review | venue | metadata | iclr / neurips / tmlr (icml auto-dropped) | peer medoid-bank (existing) |
| code-review | language | metadata | python/go/java/typescript/javascript | NEW medoid-bank-auto (40 medoids of 67k pool) |
| notice-and-comment | topic | topic-model | cms(health)/faa(transport)/species(wildlife)/epa(env)/rule | NEW medoid-bank-auto (40 of 90k) |
| press-release | topic | topic-model | earnings/global-biz/credit/health-covid/tech | NEW medoid-bank-auto (40 of 48k) |
- **Class-balanced 50/50 within each sibling** (build_strata.py, cap 3000/class, drop if minority<300).
  WHY: raw venue/language base-rates are pathological (icml .979, neurips .951, code ~.87–.90) —
  natural-rate sampling would see ~19 negatives/900 and read "wants no new metrics" when it's really
  zero discrimination power. Balancing isolates the within-sibling discrimination question and makes
  siblings comparable. NOTE: earlier subfield/domain runs used natural rates (~.34–.44, already near-
  balanced) — disclose the mixed treatment; each set is internally consistent so per-set counts hold.
- notice-and-comment has NO metadata (text+judgement only) → topic-model is the ONLY stratification
  (the purest test of the topic-diverse idea). notice topics are clean policy areas; press "amp" slug
  is &amp; HTML noise but topics cohere (earnings/global/credit/health/tech).
- Supervisor `scaleout_wave.sh` (sk3 PID 2086924): waits for a TRULY-FREE GPU (mem<15GB, never shares
  another job's card — all 8 loaded at launch, so it polls @180s), runs axes in priority order
  peer-venue→notice→press→code (code 4096-tok slow → n=600, rest n=900), per-axis liveness, then
  Lane-1 stage-2, then sibling_lattice (now globs the 4 new axes) + consolidate. ~22 legs, ~9 GPU-hr.
- **Lane 1 folded in**: replicate_candidates.py disjoint stage-2 (salt rep2, n_rep 1200) on the
  peer-iclr theory-rigor candidates (neural/graph/image) → turns the "proposed & recurring" peer
  headline into confirmed-or-killed. Output outputs/ctree/stage2/peer_theory_rigor.

## SCALE-OUT WAVE RESULTS (all 4 axes in, 2026-07-10 08:32; wave rc=0)
22 legs + 4 liveness passes, all rc=0. sibling_lattice.py + consolidate rc=0. (Lane-1 stage-2 crashed
twice — template `{}`→`{community}` fix, then GPU teardown-lag OOM; re-run in flight, see below.)

### Invented-novelty layer — apparent RECURRENCE in 3 of 12 sets, but STAGE-1 ONLY
`thmShr` = invented themes (cos>.55) shared across ≥2 siblings = the "common invented" metrics.
⚠ This is the STAGE-1 (in-run gate) view. Stage-2 disjoint replication (below) DISSOLVES every
cross-sibling recurrence: peer theory-rigor 0/13 replicate; math "alternative method" replicates in
ONE sibling only. Read this table as "what thematically echoed across siblings before the disjoint
check," not "confirmed shared metrics." Final verdict = CROSS-THREAD STAGE-2 RECONCILIATION section.
| set | sib | raw | thmShr | thmUnq | what recurs |
|---|---|---|---|---|---|
| **math-analysis** | 5 | 3 | **1** | 1 | "answer via alternative method / resolve the specific error" (integration+series) — **formal KEEP + replicated** |
| **peer-iclr** (subfield) | 6 | 13 | **3** | 7 | theory-rigor family (proof / theoretical advance / non-incremental) across graph·image·neural — stage-1 |
| **code-lang** | 5 | 10 | **2** | 5 | "completion of stated scope" (go+typescript) + "deletion magnitude / net-lines-removed" (java+python) — stage-1 |
| peer-venue | 3 | 2 | 0 | 2 | — (venues invent locally, no recurrence) |
| notice-topic | 5 | 5 | 0 | 5 | — (topic-model axis: purely local) |
| press-topic | 5 | 3 | 0 | 3 | — (topic-model axis: purely local) |
| math-algebra / math-discrete-geom / humor-scenario / humor-wordplay / cw-scifi / cw-fate-deal | | | 0 | all | sibling-local |
CORRECTED HEADLINE: recurrence of an invented metric across siblings is NOT peer-review-unique. It
shows up wherever a **shared craft transcends the sub-community boundary** — proof-rigor across ML
subfields, code-review norms (finish-what-you-claim, deletion-magnitude) across languages, and
solution-validity across analysis topics (the math-analysis one is the strongest: formal keep +
replicated). It is ABSENT under topic-model partitions (notice, press) and style/scenario partitions
(humor, cw), where invention is purely local. So: metadata/craft axes → some shared invention;
topic axes → none. Sibling-locality of the MAJORITY of invention holds everywhere.

### Bank-commonality (shared vocabulary) — a sharp live/dead gradient
| axis | union live /40 | common-all | common-≥2 | read |
|---|---|---|---|---|
| code-lang | 24 | 1 ("Static code analysis") | 20 | LIVEST — code-review criteria are language-general |
| press-topic | 20 | 0 | 12 | live, widely shared, none universal |
| peer-venue | 5 | 1 | 1 | DEAD (peer bank always ~.50 base) |
| notice-topic | 3 | 0 | 1 | DEAD (bank poor fit; base ~.50) |
The two LIVE-bank axes (code, press) are exactly where you'd expect eval criteria to be domain-general;
the two DEAD-bank axes (peer, notice) are where the medoid bank barely discriminates at all.

### Pooled/general leg — the apparent "break" is a BALANCING ARTIFACT (dilution stands)
Old natural-rate axes (math/humor/cw/peer-subfield): pooled invents **0** (dilution, 4/4).
New balanced axes: pooled invents peer-venue 4 (incl 1 formal KEEP), code-lang 4, notice 3, press **0**.
Cause: `_write(balance=True)` balances the POOLED file 50/50 OVERALL. When siblings have divergent
natural rates, overall-balancing MANUFACTURES a sibling-identity→label correlation. Measured spread of
P(accept | sibling) INSIDE the balanced general file:
- peer-venue: neurips .91 / tmlr .57 / iclr .28 → **spread .63** (huge) → 4 invented + KEEP
- code-lang: go .58 … java .38 → spread .20 (moderate) → 4 invented
- notice / press topics: ~.03–.04 (flat; topics from one corpus share a base rate) → 3 weak / **0**
The count of spurious pooled-invention tracks the manufactured spread. PROOF the peer-venue KEEP
("Computer Science Artifact Release", pooled bits +.040, p_auc .0007) is a venue-identity proxy, not a
quality criterion: the SAME idea, proposed WITHIN each single venue where it cannot proxy venue,
collapses — iclr "Explicit Research Artifact Release" +.009 p=.16 (dropped), iclr "release_of_concrete_
reproducible_artifacts" +.018 fails Bonferroni, neurips nothing, tmlr ~0. Worth 4× more pooled than in
the best single venue, non-significant within every venue = textbook cross-group base-rate artifact.
→ Dilution finding intact; the machinery finding a shortcut exactly when one is planted (and its being
diagnosable by within-venue collapse) is itself a validation of the gate's behavior.
LESSON (→ BEST-PRACTICES): for a pooled/general CONTROL leg, balance PER-SIBLING then pool, or pool at
natural rates — NEVER balance a heterogeneous pool overall; it plants a sibling-identity confound.

### Lane-1 stage-2 VERDICT (2026-07-10 10:37, rc=0) — peer theory-rigor is a STAGE-1 MIRAGE
13 candidates, disjoint n_rep=1200, salt rep2, Bonferroni α2 bar .003846. **0 KEPT.** 3 hit nominal
p_auc<.05 (image "Unconventional Insight" .0064, graph "Explicit theoretical formulation" .028, graph
"Theoretical/Analytical Insight" .029) but ALL fail Bonferroni AND fail on bits (p_bits .20–.32); the
other 10 have rep_bits ≈0 or negative. **The peer-subfield theory-rigor recurrence does NOT survive
disjoint replication.** (Crashes en route: template `{}`→`{community}`; then GPU teardown-lag OOM —
fixed with <5GB free-check + PYTORCH_ALLOC_CONF=expandable_segments. Monitor grepped wrong file, job
had already finished; ground truth = ledger, not monitor.)

### CROSS-THREAD STAGE-2 RECONCILIATION — invention is sibling-LOCAL after all
Audited ALL outputs/ctree/stage2/*/stage2_ledger.json. Every FORMAL (disjoint-replicated) keep is
tied to ONE sibling; NONE is a metric shared across siblings:
| stage-2 KEPT | sibling | p_auc | rep_bits |
|---|---|---|---|
| provides_broad_contextualization_or_counterexample | math general-topology | 1e-6 | .020 |
| Question Answered by Alternative Method | math sequences-and-series | .026 | .014 |
| meta_prompt_subverts_expected_tone_or_medium | cw meta-experimental | .0007 | .013 |
| Multilayered_Resolution_in_the_Pivot | humor everyday-observational | 1e-5 | .012 |
**CORRECTION to the invented-recurrence table above**: the math-analysis "alternative method" cluster
recurred across integration+series at STAGE-1, but replicated in sequences-and-series ONLY (the
integration twin "Resolves the specific mathematical error" did NOT replicate — w3-math-integration
ledger has 0 keeps/repls). So it is a SINGLE-SIBLING keep, not a confirmed shared metric. Likewise
peer theory-rigor recurred at stage-1 (graph/image/neural) → 0 replicate. **Net: the shared-NOVELTY
layer is entirely a stage-1 phenomenon; under disjoint replication, every surviving invented metric is
sibling-local.** What genuinely varies across siblings by domain is shared VOCABULARY (existing bank
metrics reused — code/press live, peer/notice dead), NOT shared new invention. This is the two-stage
gate doing its job: it killed the recurrence mirage that stage-1 thematic clustering suggested.

## Ops notes (2026-07-09)
- GPU5 taken by ANOTHER workstream (src.batch_detect, 70GB, PID 1388460) — not touched; moved
  my work to free GPU 2 (1 GPU, respecting the limit).
- LESSON: never `scp`-overwrite a script a live bash is executing — it reads by byte offset;
  the overwrite shifted offsets and killed the wave-3 tail supervisor with a syntax error at
  its wait loop. Kill first, then replace, then relaunch.
- rc=$? echo bug recurred (`echo "[$(date)] rc=$?"` always prints 0 — $(date) resets $?);
  all supervisors now capture RC=$? on its own line. Ground truth = ledgers, not rc lines.
- ICLR subfield build is CPU-slow (~15+ min: 28k CPU-embed + KMeans n_init=10 + bigram TF-IDF);
  future rebuilds → MiniBatchKMeans / batched GPU embed.
- LESSON: any nohup job that imports torch/sentence-transformers on sk3 SEGFAULTS at import unless
  OpenMP threads are pinned. Always launch embedding/ST jobs with
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false`.
  Foreground import works without it; the crash only appears under nohup backgrounding.
- LESSON: a dropped/renamed stratum leaves an ORPHAN split from a prior build that the run.py glob
  re-registers (stale unbalanced icml.csv.gz got picked up). After rebuilding strata, sideline
  orphans (`mv x.csv.gz x.csv.gz.stale`) — the glob is `dir/*.csv.gz` non-recursive.
