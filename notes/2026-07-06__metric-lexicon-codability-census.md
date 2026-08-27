# Metric-lexicon codability census — text-only, original-source, subfield-controlled

*2026-07-06. User-approved framing + two additions (key-term extraction pass; hierarchy-completion
gate). Task #26. Companion to the behavioral profile (`2026-07-01__codability-audit-and-proposal.md`,
`methods/codability/`) — this is the OBSERVATIONAL leg: no behavior, no re-execution, just how the
original human authors wrote the metrics. Complements E2 frame-commitment (enregisterment predicts
corpus-level conventionalization) and feeds #23 (a-priori text features).*

## 0. The question and the estimands

How often do different low-level metrics express the SAME evaluative concept in DIFFERENT words —
on the original source text, controlling for subfield? Conventionalization of a community's
evaluative vocabulary, in the Brown–Lenneberg naming-agreement sense, measured from the corpus.

Per concept (an audited hierarchy node), from the lexeme sets of its member expressions:

- **Lexicalization count**: number of distinct head-term lexicalizations among independent sources
  (the user's readout: "the different ways subtrees are coded underneath an R3 node").
- **Naming agreement / name entropy**: distribution over head terms across independent sources;
  H=0 ⇒ fully conventionalized name ("punching up"); high H or dominant `named_in_source=false`
  ⇒ the community describes without naming.
- **Synonymy rate** (per task): fraction of concepts with ≥2 lexically disjoint realizations from
  independent sources.
- **Unnamed-concept rate**: share of expressions with `named_in_source=false` — describing without
  a name is itself the anti-conventionalization signal.
- **Subfield decomposition**: within-subfield vs cross-subfield lexical divergence (reuses
  `strata.py` metadata strata + `decompose.py` mixed model) so genre-driven wording variation never
  masquerades as low conventionalization.

Registered directional expectation (before running, consistent with the E2 prereg): humor's
vernacular register is MORE conventionalized (higher naming agreement on its enregistered terms)
than institutional domains' document-local phrasing. Either outcome informative.

## 1. User addition A — the key-term extraction pass (the lexeme substrate)

One LLM pass over ALL metrics, reading the ORIGINAL source document (not the gpt-parsed or
canonicalized layers), returning per metric:

```
found            does the source actually express this criterion (false ⇒ bank parse artifact)
quote            verbatim passage (≤60 words) copied exactly from the source
key_terms        ≤8 words/short phrases THE AUTHOR uses for the evaluative concepts (never synonyms)
head_term        the author's NAME for the concept (term of art), or null if described-not-named
named_in_source  bool — conventionalized name present
```

**Mechanical validation (no trust in the model):** `quote` must be a whitespace-normalized substring
of the source text; every `key_term` must appear in the source; violations are rejected and retried
(no repetition_penalty — retry with different seed per house rule). `found=false` rate per task is
reported (bank provenance-quality datum).

**Model choice = an equivalence check, not an assumption** (user: GLM-4.7 preferred if equivalent
to Sonnet). Protocol: same 150 items (75 humor + 75 CW, stable-hash sampled) + 8 blinded planted
anchors run through BOTH; agreement = per-item Jaccard on normalized key-term sets + head-term match
+ named/found agreement; both must pass anchors. If GLM ≈ Sonnet ⇒ GLM-4.7 bulk (zai_anthropic
backend, HTTP, 0-GPU; quota watched — fall back to Sonnet subagents if quota binds mid-bulk).
Anchors ride blinded in EVERY bulk batch too (resumed-agent degeneracy guard).

**Provenance chain** (verified 2026-07-06): `canon_all_real_forms.jsonl` (53,413 forms, 11 tasks)
keys = `task::layer::source_doc::item_idx`; `datasets/<task>/online-rubrics/raw/` retains 2.6–4.6K
original HTML docs per task for ALL 10 scraped tasks; gpt-parsed items (name/description) act as the
pointer, never as the text. Long docs: pass the best lexical-overlap window (~8K chars) + doc head.
Math is out of scope for the raw leg (bank not scraped from rubric pages).

## 2. User addition B — the hierarchy-completion gate ("beyond v6 llama")

Codability-by-subtree is only meaningful if L0/R1/R2/R3 are RIGHT. Current state (from
2026-06-12 audits, `project_rubric_clustering_pipeline`): leaf partition (tau 0.825 blend) has
realized under-merge FN 45%/68%/33% (peer/humor/code) against held-out judge labels; single greedy
v6 score=2 labels overturn ~55% on re-adjudication; "same rule" is NOT transitive (union-find
snowballs); R2/R3 = LLM merge rounds on top. The audit must certify each level, not just pairs.

Plan (chain-proof, judge-grounded, embedding-free in every reported number):

1. **Trusted-edge graph**: v6 score=2 edges kept only with ≥2 independent edges or fresh-majority
   re-adjudication (the 6/12 RULE); readjudicate_verdicts.jsonl merged in.
2. **Under-merge candidate net, two prongs**: (a) *lexeme-overlap net* from the extraction pass —
   shared rare key_terms across clusters (embedding-free, and made possible by addition A);
   (b) frozen exogenous encoder neighbors (`emb_rubric_cluster_<task>.npy`, clean-S list) as an
   *efficiency pre-filter only* — candidates go to a fresh judge; embeddings never in numbers.
3. **Fresh judge = the equivalence-check winner** (this is what "beyond v6 llama" buys: a second,
   stronger family adjudicating, with anchors).
4. **Level certificates**: per task × level, sampled FP (within-group pairs judged different) and
   FN (cross-group judged same) with CIs; R2/R3 merge rounds re-audited by sampling
   `merged_groups` members against their R3 anchor. Ship `hierarchy_cert_<task>.json`;
   the census runs only over tasks/levels whose certificate clears (FP and FN both bounded ≤ ~15%
   with the trusted-edge repairs applied; threshold reported, not hidden).
5. **Repairs are append-only** (never-delete rule): original partitions untouched; repaired
   partition = base + adoption/merge lists, versioned.

Then the census readout is exactly the user's framing: for each certified R3 node, the subtree's
lexeme sets = the different ways the community codes that concept.

## 3. Discipline inherited from the retraction + S⊥R lessons

- Judge decisions may PARTITION (merge/split); they are never the codability score. The score is
  descriptive corpus statistics over author lexemes.
- Surface axis (lexemes, from raw spans) ⊥ concept axis (judge partition): no verdict-trained
  encoder (LoRA-bge/CE) and no blend-cluster co-membership anywhere in a reported number — the
  blend partition is biased toward lexical similarity, which would UNDERSTATE synonymy.
- Independence unit for naming agreement = source DOCUMENT (dedup by source_doc; same-doc repeats
  never count as independent conventionalization evidence).
- Dataset-first: extraction pilot (humor+CW) manually spot-checked before any bulk run; anchors in
  every batch; stable-hash sampling everywhere.

## 4. Consolidation map (user: "consolidate all the code")

New package `methods/codability/lexicon/` (this experiment's single home):

| module | role |
|---|---|
| `sources.py` | key → original doc text (raw HTML→text w/ fallback parser, claude-parsed md); window selection; extraction-context builder |
| `anchors.py` | planted anchor docs + expected answers (blinded IDs) |
| `extract.py` | prompt, GLM/zai batch runner, mechanical quote/term validation, retry-on-violation, JSONL sink |
| `compare.py` | GLM-vs-Sonnet equivalence report (Jaccard, head-term, anchors) |
| `audit.py` | trusted-edge graph, chain-proof repairs, level certificates (consumes `.cache/norm_embed/` pulls) |
| `census.py` | lexeme normalization, naming agreement/name entropy, synonymy rate, per-R3 subtree census, strata join |
| `run_lexicon.py` | driver: `--phase build|equivalence|bulk|audit|census` |

Legacy scripts (kept, now documented as superseded-for-this-purpose): `scripts/cluster_canon.py`,
`leaf_name_clusters.py`, `cluster_pairs.py`, `adopt_tail_clusters.py` (its ≥2-edge rule is imported
into `audit.py`), sk3 `sk3_match_pipeline.py` family (locked artifacts consumed read-only).

Outputs: `outputs/lexicon/` (extraction JSONLs, equivalence report, hierarchy certs, census JSONs)
→ pulled to `notebooks/data/` for the results notebook.

## 5. Order of operations

1. rsync judge artifacts from sk3 (running, background) — unblocks `audit.py`.
2. Build extraction contexts (humor+CW pilot) + anchors; stable-hash 150-item equivalence sample.
3. Equivalence: GLM-4.7 batch (HTTP) ∥ Sonnet subagent batch (Max plan) → `compare.py` report.
4. Manual spot-check gate on pilot extractions → bulk extraction (winner model), task by task.
5. Hierarchy audit with fresh judge on candidate nets → level certificates.
6. Census on certified levels → results notebook + notes harvest.

GPU-free throughout; runs beside day_runner/battery without touching the queue.

---

## 6. Taxonomy re-evaluation — decode + baseline (appended 2026-07-06 evening, user push)

**The chain, decoded and verified end-to-end (humor general):** L0 rubrics (10,253; canon = the
filtered 3-bucket subset, 5,885 task-wide) → complete-linkage tree-leaves (9,708, **96%
singletons** — this level barely merges; the real first grouping is the LLM's) → R1 = 1,148
parented + 681 merged nodes (meta_merge order: parents then merges) → R2 = 285 merged_groups
(+164 grandparents; `all_leaves` carry canon keys) → R3 = 60 merged + 38 grandparents
(input = R2 merged_groups in enumeration order). Decoder: `lexicon/hierarchy.py`; artifacts
`outputs/lexicon/upmap{,_r2}_<task>.json`, `hier_nodes_<task>.json`.

**Two leaf-partition systems coexist** (a fact the re-evaluation must own): the tree's
complete-linkage leaves (no real merging) vs the tau-0.825 judge-distilled partition (May 18,
separate lineage). All leaf FP/FN statements are about the latter; the tree effectively treats
raw rubrics as leaves and relies on R1 to group.

**Findings that qualify "do we have the right levels":**
1. **R2/R3 are NOT partitions.** merged_groups × grandparents overlap heavily (humor general:
   3,723 R2 / 2,552 R3 key collisions — an R1 parent is often both merged into a group AND child
   of a grandparent). Census assignment = tightest-first (merged before grandparent), stated.
2. **R3 reaches only 55–62% of canon keys** (humor 3,219/5,885; CW 3,049/4,950); R2 85–88%.
   Subtree census results must always quote coverage.
3. **Leaf level (tau partition), all 11 tasks, after trusted repairs:** FP ≤ 0.1% everywhere;
   FN 5.7% (code-review) – 15.2% (humor). Work-list: 2,341 merge candidates + 16,278 untrusted
   T3 pairs. Pilot (humor+CW) fresh 3-vote adjudication RUNNING (v6 prompt verbatim, GLM-4.7,
   blinded pos/neg anchor pairs; smoke: pos anchors 9/10, 0 unparsed).
4. **R1/R2/R3 coherence audit staged** (`containment_payload_<task>.jsonl`, ~1.9K checks/task,
   4 leaves/node + planted anchors): umbrella levels get a belongs-under certificate, not a
   sameness one.

**Equivalence gate result:** GLM-4.7 anchors 8/8 (incl. the substitution trap), validity 92%
vs Sonnet 7/8 / 75%; categorical agreement high (found .99, head .78, named .84) but term-set
Jaccard .49 < .55 ⇒ formally NOT equivalent — GLM chosen for ALL bulk passes (stronger on every
gated criterion; census stays single-model). Sonnet arm retained for a term-coverage-union
robustness check later.

**Sequenced GLM queue (serial to respect quota/rate):** fresh-judge pilot (running) →
containment audit → bulk extraction humor+CW (~33K calls) → apply fresh labels → final leaf
certificates + census v1 (leaf + R3-subtree readouts).

## 7. CORRECTION + recall audit (user question caught a metric error, 2026-07-06 late)

**The §6 leaf table's "FN 5.7–15.2%" was mis-defined**: it computed P(same | split) over all
labeled split pairs — diluted by the large score-0 pool. The pipeline's convention (realized
recall, 2026-06-12 correction) is **recall = P(co-clustered | labeled same)**. `leaf_certificate`
fixed to report recall/precision; corrected table (trusted same-pairs; tau-repaired partition,
base in parens):

| task | recall | (base) | precision |
|---|---|---|---|
| humor | .596 | .474 | .9998 |
| creative-writing | .698 | .558 | 1.0000 |
| code-review | .887 | .833 | .9999 |
| peer-review | .725 | .622 | .9995 |
| press-releases | .689 | .581 | .9998 |
| news-homepages | .622 | .507 | .9998 |
| grant-funding | .698 | .593 | .9990 |
| legal-outcome | .678 | .518 | .9996 |
| notice-and-comment | .607 | .454 | .9991 |
| patents | .704 | .617 | .9990 |
| math-se | .653 | .535 | .9997 |

**Tree-level realized recall** (co-noded iff node SETS intersect — fair to overlaps; humor / CW):
R1 .051/.075, R2 .207/.256, R3 .465/.608; score-0 co-noding ≤.037 at every level. The LLM R1
round (batch-local visibility, K=100 slices) is the fragmentation bottleneck; R2/R3 rounds only
partially recover cross-batch merges. **Every layer of every lineage errs on under-merge;
precision is never the problem.** Census implication: un-repaired trees UNDERCOUNT synonymy and
deflate per-concept source counts — the leaf partition carries the census; R3 readouts must
quote coverage + recall.

**Recall denominators, stated:** (a) trusted (readjudicated/triangle) same-pairs only — T3
single-vote pairs excluded until the RUNNING fresh 3-vote pass resolves them (humor done, CW in
flight); recompute over trusted+fresh after. (b) All labels live on the kNN candidate net ⇒
these are IN-NET recalls; net coverage was measured 99.5% on eval pairs (locked-pipeline memory);
the lexeme-overlap net (extraction pass) is the independent beyond-net probe → two-net
capture-recapture recall bound.
