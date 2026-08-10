# dense_standard_v3: enriched-text rebuild of the GitHub-PR-merge dense corpus

Date: 2026-08-05
Follow-up to: `notes/2026-08-05__code-dense-score-audit.md` (§7 "single follow-up that would most change the conclusion")
Machine: sk3. Out dir: `sk3:/lfs/skampere3/0/alexspan/norm-research/datasets/code-review/dense_standard_v3/`
Build script: `sk3:/lfs/.../datasets/code-review/build_pr_dense_v3_enriched.py`
Build log: `dense_standard_v3/build_v3_dense.log` · Manifest: `dense_standard_v3/build_manifest.json`
Work dir (intermediates): `dense_standard_v3_work/` (`repo_dir_map.json`, `comments_filtered.parquet`, `join_resolution_report.json`, `hygiene_report.json`)
`dense_standard_v2/` was **not** modified.

---

## HEADLINE — build succeeded, training GATED and NOT launched

The corpus was built, validated, and is training-ready. **Training was not launched**: the
pre-registered hard gate was "if PR-body join coverage < 60% of rows, STOP and report."
Observed body coverage is **47.4%** (28,008 / 59,111). The gate fires.

The shortfall is **not a join bug** — it is a genuine coverage hole in
`pr_descriptions.csv.gz`, proven three ways (§2.3). No amount of remapping fixes it: the
absolute ceiling on body coverage, ignoring owners entirely and matching on bare repo name,
is **48.4%**.

Launcher `sk3:/lfs/.../methods/dense/run_pr_dense_v3.sh` is staged (seed 42 only, GPU 1 default)
and can be fired the moment you say go.

---

## 1. Join resolution: batch_runs dir → (owner, repo)

`dense_standard_v2.repo` is a bare `pr_test_execution/batch_runs/` directory name
(`android-maps-utils`, `0chain`, `nomad_m_deep_003`). The enrichment tables key on
`owner` + `repo`. No explicit map file exists, but an **authoritative** one is recoverable
from the scraper's own logs.

### 1.1 Method (priority cascade)

| src | mechanism | repo_dirs resolved |
|---|---|---|
| **A** | `batch_runs/<dir>/logs/factory_process.log` line 1: `[factory] === <owner>/<repo> -> <path> ===` | **1,155** |
| **B** | `batch_runs/<dir>/logs/fetch.log` JSON header `{"owner": ..., "repo": ...}` | 0 (all already covered by A) |
| **C** | bare-name fallback: `dir == enrichment.repo`, applied **only** when that bare name has exactly one owner in the enrichment table | 24 |
| — | unresolved | 88 (49 surviving hygiene) |

This is an explicit, logged owner/repo string written by the scraper at fetch time — not a
guess. Source C was applied to only 24 dirs and required owner uniqueness, so it cannot
introduce a wrong-owner join.

The 88 unresolved dirs are all re-scrape variants (`*_pilot_[mr]`, `*_m23_NNN`,
`*_rej23`, `*_rejtop23_NNN`, `*_deep_NNN`, plus `swebench-django`, `swebench-flask`,
`mctx`, `pypeit`). Per instruction these were **not** force-joined to a base repo. 49 of
them survive hygiene, carrying 5,071 rows (8.6% of v3), which get diff-only text.

### 1.2 Collision checks (both clean)

- **No two batch_run dirs map to the same `owner/repo`** — 0 collisions among the 1,155
  log-resolved dirs. So no re-scrape is silently double-counted against one enrichment repo.
- **Bare-name ambiguity**: 307 bare repo names in the enrichment table carry >1 owner
  globally (`agent`, `api`, `cli`, `common`, `contrib`, `cortex`, `dns`, …). 69 of our
  resolved dirs have such an ambiguous bare name — **and all 69 came from source A**, i.e.
  their owner is taken from the log, not guessed. The 24 source-C dirs were all
  owner-unique by construction. Zero of our joins rest on an ambiguous bare name.

### 1.3 Sanity validation (2 independent checks, both pass)

**(a) Title + creation-date cross-check — 20/20 agreement.** For 20 randomly sampled joined
`(repo_dir, pr_number)`, the locally stored `batch_runs/<dir>/pr_meta_full.jsonl` title and
`created_at` were compared against `pr_merge_status.csv.gz` (`pr_title`, `pr_created_at`),
which shares the `owner`+`repo`+`pr_number` key space with `pr_descriptions.csv.gz` and
`parsed_comments.csv.gz`. **20/20 exact title match and 20/20 exact date match.** Examples:

```
commons-numbers | apache/commons-numbers#117  'Numbers 188.refactor complex log function'  2022-07-15 == 2022-07-15
Slimefun4       | Slimefun/Slimefun4#1101     'Code Cleanup, Addition of Electric Compressor' 2019-09-08 == 2019-09-08
OpenPype        | ynput/OpenPype#2495         'Flame: create publishable clips'            2022-01-06 == 2022-01-06
armeria         | line/armeria#672            'Allow more customization of LoggingService' 2017-07-10 == 2017-07-10
```

This is decisive: the mapping recovers the same PR object on both sides of the join.

**(b) Comment-path ↔ diff-file check — 20/30 raw, effectively 20/20.** For 30 randomly
sampled joined PRs, we asked whether any inline-comment `path` appears among the
`diff --git a/<file>` entries of the stored diff. 20/30 hit. **All 10 misses are rows whose
v2 diff sits at exactly the 24,000-char cap** (verified individually — every one is
`difflen=24000`), i.e. the commented file was cut off by the v2 truncation, not
mis-joined. Among rows where the whole diff is visible, the rate is 100%. The pattern is
diagnostic of truncation, not of a bad key: e.g. `iotaledger/goshimmer#2051` has 29 comment
paths but only 8 diff files survive the cap.

---

## 2. Coverage

### 2.1 Post-hygiene coverage (n = 59,111)

| field | rows | frac | train | eval | test |
|---|---:|---:|---:|---:|---:|
| **Title** | 48,274 | **.8167** | — | — | — |
| **PR body (non-empty)** | 28,008 | **.4738** | .4636 | .5046 | .5284 |
| **≥1 inline review comment** | 31,261 | **.5289** | .5180 | .5495 | .5996 |
| body **or** comments | 31,261 | .5289 | | | |
| any of the three | 49,013 | .8292 | | | |
| **nothing but the diff** | 10,098 | **.1708** | | | |

Title sources: 47,895 from `batch_runs/<dir>/pr_meta_full.jsonl` (+`corpus_prs.jsonl`),
plus 379 from a `pr_merge_status.csv.gz` fallback.

Note `body ⊂ comments` exactly (28,008 ⊂ 31,261): every PR with a body also has comments.
Both enrichment tables descend from the same fetch universe, so they fail together.

When comments are present there are a lot of them — mean **19.54**, median **20** — i.e. the
20-comment cap binds for ~75% of enriched rows. The comment table only contains PRs with
substantial review threads.

### 2.2 Pre-hygiene coverage (for reference)

Over all 63,707 v2 rows: description **row** present 47.8%, **non-empty body** 44.0%.
By resolution source: A_factory 56.8% body, C_bare 13.8%, unresolved 0%.

### 2.3 Why coverage is low — it is the table, not the join

Three independent proofs:

1. **Case-insensitive rematch gains exactly 0 rows** (.4738 → .4738). Not a casing problem.
2. **Owner-agnostic upper bound is .4844.** If we throw away the owner entirely and match on
   `(bare repo name, pr_number)` — a deliberately over-permissive join that would sweep in
   wrong-owner matches — coverage rises only from 47.4% to 48.4%. Even matching on the raw
   batch-dir name gives the same 48.4%. **There is no remapping that reaches 60%.**
3. **`pr_descriptions.csv.gz` is broad and shallow**: 143,158 rows over **15,484**
   owner/repo pairs — median **2** PRs per repo, max 154. It was built from a different
   (much wider, much thinner) PR universe than the 63,707-row execution corpus. Per-repo
   coverage is bimodal: 347 repos at 100%, 376 at 80–99%, and **200 repos at exactly 0%**
   (131 of them post-hygiene, 17,681 rows = 30% of v3 in repos the fetch simply never
   touched). 107 of our 1,179 owner/repo pairs are absent from the table entirely.

Even if all 5,071 unresolved-dir rows were perfectly resolved and 100% covered — the
optimistic ceiling — total body coverage would be (28,008 + 5,071)/59,111 = **56.0%**, still
under the gate. Reaching 60% requires re-fetching bodies from the GitHub API, not re-joining.

---

## 3. Hygiene

### 3.1 R1 — single-class `_deep` repos

Rule: drop every repo_dir matching `_deep` whose labels are 100% single-class.
Result: **37 of 37** `_deep` repos are single-class — **zero mixed-label `_deep` repos exist**.
All 37 dropped; **3,666 rows**.

| split | repos dropped | rows dropped |
|---|---:|---:|
| train | 27 | 2,688 |
| eval | **5** | **462** |
| test | 5 | 516 |

The eval five are exactly the audit's `helm_m_deep_024`, `klaytn_acc_deep_019`,
`nomad_m_deep_003`, `prometheus_m_deep_018`, `xraycore_m_deep_026` (462 rows) — **exact
match with §6 of the audit note.** 25 of the 37 are 100%-merged; 12 (`*_rej*_deep_*`,
`mx-chain_rej_deep`, `prebid_rej_deep`) are 100%-rejected — the corpus-wide sweep caught a
whole all-negative family the eval-only view never saw.

### 3.2 R2 — exact-duplicate diffs (on v2 `text`), corpus-wide

| | groups | rows dropped |
|---|---:|---:|
| **both-label groups → whole group dropped** | **122** | **303** |
| same-label groups → keep first | 516 | 627 |
| total | | **930** |

Both-label rows per split: train 259 (103 groups), **eval 35 (17 groups)**, test 9 (3 groups).
The eval figure is **exactly the 35 rows / 17 groups** the audit predicted.

### 3.3 Net

| | v2 | v3 | Δ |
|---|---:|---:|---:|
| rows | 63,707 | **59,111** | −4,596 (−7.2%) |
| repos | 1,267 | **1,228** | −39 |
| train | 51,037 | 47,659 | −3,378 |
| eval | 6,372 | **5,822** | −550 |
| test | 6,298 | 5,630 | −668 |
| eval pos-rate | .8074 | .7994 | −.008 |

---

## 4. Text composition (frozen spec, implemented verbatim)

```
Title: {title}

Description: {pr_body}

Review comments:
{comments}

Diff:
{diff}
```
- comments ordered by `comment_id` ascending, max 20, each truncated to 400 chars, rendered
  `- [{author_association}] {body}`, section capped at 16,000 chars; `(none)` if absent.
- missing body → `(none)`; missing title → `(unknown)`.
- TOTAL cap 24,000, achieved by truncating **the diff only**. Diff truncated in
  **32,226 / 59,111** rows.

Length: mean 16,888, median 24,000, min 151, max 236,203.

**One documented spec edge case.** The frozen spec caps the comment section (16k) and each
comment (400 ch) but places **no cap on the PR body**, while also mandating a 24,000-char
total achieved by truncating the diff only. For **22 rows (0.037%)** the header+body+comments
prefix alone exceeds 24,000, so the diff budget went to 0 and the row exceeds the total cap
(the max, 236,203 chars, is a single enormous PR body). I kept the spec's stated priority
("discussion is never truncated") rather than silently capping the body. These rows are
truncated by the 2,048-token model window anyway. Flagging in case you want a body cap in v3.1.

### Leak diagnostic (reported, not filtered)

Regex `\b(merg(e|ed|ing)|closing|superseded|abandon)` (case-insensitive), over comments
actually included in the text: **4,881 / 610,768 = 0.799%**. Low. Note this is per-comment;
per-row exposure is higher since enriched rows carry ~19.5 comments each. Worth a
sensitivity check post-training rather than a pre-emptive filter.

---

## 5. Recency channel (named non-textual predictor)

Predictor: within-repo `pr_number` percentile rank (rank as a fraction, computed inside each
repo), evaluated against `judgement`. Pooled AUC:

| split | AUC | n |
|---|---:|---:|
| **eval, PRE-hygiene** | **.4994** | 6,372 |
| **eval, POST-hygiene** | **.4961** | 5,822 |
| train, post-hygiene | .5260 | 47,659 |
| test, post-hygiene | .4893 | 5,630 |

Within-repo (55 eval repos with n ≥ 30 and both classes): median AUC .487, n-weighted mean
.487, and **25.5% of repos have |AUC − .5| > .15**.

**Reading — this is not a contradiction of the audit, it is the correct scope statement.**
The audit's headline `pr_number` AUC of **.806** was measured on *one repo's bump subset*
(`android-maps-utils`, 57 `.github/workflows` PRs). Pooled across all repos the recency
channel is **worth nothing (.4961)**, because the effect runs in *opposite directions in
different repos* — the audit already noted captum .137 vs pootle .955. Recency is a strong,
real, **repo-local** channel that cancels on aggregation. The paper should say "queue
position predicts merge outcome within particular repos, at |AUC−.5| > .15 in a quarter of
them, but carries no pooled signal", not "recency predicts merge at .81".

---

## 6. Split integrity — all assertions pass

Splits reused **verbatim** from v2's `split/{train,eval,test}.csv` repo membership. Asserted
in-build (build would have aborted otherwise):

- ✅ every v3 `(repo, pr_number)` exists in v2 — `chk.index.isin(v2idx.index).all()`
- ✅ every v3 row's `judgement` equals its v2 `judgement`
- ✅ every v3 row's `split` equals its v2 `split`
- ✅ repo-disjointness: `train&eval=0, train&test=0, eval&test=0`

| split | n | frac | repos | pos | pos-rate |
|---|---:|---:|---:|---:|---:|
| train | 47,659 | .8063 | 973 | 38,946 | .8172 |
| eval | 5,822 | .0985 | 112 | 4,654 | .7994 |
| test | 5,630 | .0952 | 143 | 4,551 | .8083 |

Files: `data.csv` (1.01 GB), `split/{train,eval,test}.csv`, `split/split_metadata.json`,
`build_manifest.json`, `build_v3_dense.log`, `samples_5.txt`.
Extra columns carried for downstream slicing: `n_review_comments`, `has_body`, `has_title`.

---

## 7. Manual validation — 5 composed texts read

`dense_standard_v3/samples_5.txt`. Content belongs to the right PR in every case; formatting
is clean; no stray encoding or delimiter damage.

**Sample 1 — `numba-dpex` → `IntelPython/numba-dpex#598`** (merged, train, 20 comments, body present, 24,000 ch).
Title "Implement compute-follows-data programming model [kernel API]"; body links the
DPPY-Spec compute-follows-data doc; comments discuss `get_execution_queue`,
`IndeterminateExecutionQueueError`, renaming `current_queue`. Comment paths
`numba_dppy/compiler.py`, `.../test_compute_follows_data.py`; the diff adds
`selecting_device.rst` documenting exactly that model. **Title, body, comments and diff are
one coherent object** — this is the strongest single spot-check in the build.

```
Title: Implement compute-follows-data programming model [kernel API]

Description: This PR introduces compute follows data for kernel.
The spec can be found here: https://github.com/IntelPython/DPPY-Spec/blob/compute-follows-data/compute_follows_data.md.

Review comments:
- [COLLABORATOR] The error message can be improved. What does "to be uniform" imply?
- [COLLABORATOR] A comment about what the `get_execution_queue` does will be helpful.
- [COLLABORATOR] I will prefer us raising a specialized Error (say `IndeterminateExecutionQueueError`) instead of a `ValueError`.
- [CONTRIBUTOR] Can we test that computations was performed on specific queue?
  ... (20 total)

Diff:
diff --git a/CHANGELOG.md b/CHANGELOG.md
+* Implement compute-follows-data programming model [kernel API] (#598)
diff --git a/docs/user_guides/kernel_programming_guide/selecting_device.rst b/... (new file)
```

**Sample 2 — `java-bigtable-hbase` → `googleapis/java-bigtable-hbase#2755`** (merged, train,
20 comments, body present). Title "feat: Ability to import HBase Snapshot data into Cloud
Bigtable using Dataflow"; comments reference `SerializableConfiguration`, `gs://$HBASE_ROOT_PATH`;
comment paths are the four `hbasesnapshots/*.java` files. Consistent.

**Sample 3 — `server-tools` → `OCA/server-tools#134`** (**rejected**, train, 19 comments, **no body**).
Title "add auth_password_settings module", `Description: (none)` renders correctly. Comments
are a textbook OCA review thread ("Use relative import", "Please respect PEP8 80 cols max
length", "s/degits/digits/"); comment paths are all `auth_password_settings/*`. This is
precisely the articulated-norm content that v2 was missing — and note it is a **rejected** PR
whose rejection reason is legible only in the comments.

**Sample 4 — `mx-chain-go` → `multiversx/mx-chain-go#4073`** (merged, train, 0 comments, no body).
Renders `Title: (unknown)`, `Description: (none)`, `Review comments:\n(none)` then the diff —
i.e. the fully-bare 17.1% case degrades exactly to v2 behaviour plus a ~60-char header.

**Sample 5** — second no-comment/no-body case, same clean degradation.

**Length distribution sane**: min 151, median 24,000, mean 16,888. No empty texts, no rows
consisting only of the header scaffold.

---

## 8. Training — GATED, NOT LAUNCHED

Per the pre-registered rule ("if body coverage < 60%, STOP and report instead of training"),
**no training job was started.** Observed body coverage **47.4% < 60%**.

No process was launched; nothing is resident on any GPU on my account. GPU state at
decision time: GPUs 0, 1, 3, 6, 7 idle (0 MiB); GPUs 2, 4, 5 in use by other jobs. The
scorer previously on GPU 0 (PID 1675629) had finished on its own — it was not touched.

The launcher is staged and ready:
`sk3:/lfs/skampere3/0/alexspan/norm-research/methods/dense/run_pr_dense_v3.sh`
Seed 42 only (no chain), `CUDA_VISIBLE_DEVICES=${GPU:-1}`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
`HOME` pinned to `/lfs/skampere3/0/alexspan`. Recipe is byte-identical to v2's:
Llama-3.1-8B, LoRA r16/α32, lr 5e-5, batch 16, eval batch 32, grad-accum 1, max_len 2048,
2 epochs, `--gradient-checkpointing`, `--class_weight_auto`, `--selection_split eval`.
Launch with (after re-checking `nvidia-smi`):

```bash
ssh sk3 'export HOME=/lfs/skampere3/0/alexspan; cd /lfs/skampere3/0/alexspan/norm-research/methods/dense; \
  GPU=1 nohup bash run_pr_dense_v3.sh > /lfs/skampere3/0/alexspan/norm-research/datasets/code-review/dense_standard_v3/runner_v3.log 2>&1 &'
```
Expected wall-clock ≈ 11 h (v2 recipe on 47.7k train rows, ~7% fewer than v2).

---

## 9. Decision the gate is asking you to make

The corpus is built and validated; the only question is whether 47.4% body / 52.9% comment
coverage is interpretable. Three options:

**(a) Train anyway, and read it as a lower bound.** The comparison stays clean — same splits,
same recipe, same repos — and a positive Δ over v2's .5734 would be a *floor* on the true
enrichment effect, since 17.1% of rows got no enrichment at all. A null result, however,
would be uninterpretable: you could not tell "discussion doesn't help" from "we only enriched
half the corpus". This asymmetry is the honest reason to be willing to run it: **it can
confirm the enrichment hypothesis but cannot refute it.**

**(b) Train on the enriched subset only** (the 31,261 rows with comments, or the 28,008 with
bodies), against a v2-text model trained on that same subset. This makes the contrast fully
interpretable in both directions, at the cost of a smaller, non-representative corpus and a
new split derivation. Cleanest scientifically.

**(c) Backfill bodies from the GitHub API** for the 131 zero-coverage repos before training.
The fetcher already exists (`fetch_pr_descriptions.py`, GraphQL, 25 PRs/request, resumable).
~31k missing PRs ≈ 1,250 requests, hours not days at the built-in 800 req/h limit. Highest
cost, highest interpretability.

My recommendation is **(b) or (c)**, and if you want a fast read, (a) *plus* the
enriched-subset contrast from (b) run as the confirmatory arm — (a) alone risks producing a
null you can't publish either way. Tell me which and I'll launch immediately.

---

## Artifact index

| artifact | path (sk3) |
|---|---|
| v3 corpus | `datasets/code-review/dense_standard_v3/data.csv` |
| v3 splits | `datasets/code-review/dense_standard_v3/split/{train,eval,test}.csv` |
| build manifest (all stats above, machine-readable) | `dense_standard_v3/build_manifest.json` |
| build log | `dense_standard_v3/build_v3_dense.log` |
| 5 composed samples | `dense_standard_v3/samples_5.txt` |
| build script | `datasets/code-review/build_pr_dense_v3_enriched.py` |
| repo_dir → owner/repo map | `dense_standard_v3_work/repo_dir_map.json` |
| join-resolution detail (collisions, ambiguity) | `dense_standard_v3_work/join_resolution_report.json` |
| filtered comment table (1.22M rows) | `dense_standard_v3_work/comments_filtered.parquet` |
| hygiene report | `dense_standard_v3_work/hygiene_report.json` |
| helper scripts | `datasets/code-review/{resolve_join,scan_comments,hygiene,diag,diag2,validate}.py` |
| staged (unlaunched) trainer | `methods/dense/run_pr_dense_v3.sh` |

---
---

# BACKFILL ROUND (appended 2026-08-05, after coordinator approval)

Everything above describes **round 1** and is left unedited. This section records the
GitHub-API body backfill, the redefined coverage gate, the v3 rebuild, and the training
launch. Round-1 artifacts were preserved before the rebuild as
`dense_standard_v3/build_manifest_round1.json` and `build_v3_dense_round1.log`.

## B1. Gate redefinition (now authoritative)

> **A row's body is COVERED iff a description record exists for it with an empty `error`
> field — i.e. the fetch was ATTEMPTED AND SUCCEEDED. A genuinely empty PR body counts as
> COVERED**, because an empty description is real signal: `Description: (none)` in the
> composed text is then a *true* statement about the PR rather than a missing-data
> placeholder. Fetch errors and never-fetched rows count as UNCOVERED.

Recorded verbatim in `build_manifest.json` → `coverage.gate_definition`. Under this
definition the round-1 (pre-backfill) figure was **51.4%** of resolved targets covered; the
round-1 headline of 47.4% was the *non-empty-body* fraction, which is now reported
separately as `coverage.body_frac`.

## B2. Backfill execution

Script: `datasets/code-review/fetch_pr_descriptions_backfill.py` (new). It **imports and
reuses** `fetch_pr_descriptions.py`'s `RateLimiter`, `query_batch`, `build_graphql_query`
and `parse_graphql_response` rather than duplicating them, but differs in three ways that
mattered:

1. **Target universe.** The original fetcher's PR list is `pr_merge_status.csv.gz`, *not*
   our corpus — running it unmodified would have fetched the wrong set. The backfill builds
   its target list from `dense_standard_v3/data.csv` keys mapped through
   `repo_dir_map.json`.
2. **Never rewrites existing data.** The original does a read-modify-write of
   `pr_descriptions.csv.gz` (loads all rows, appends, writes the whole file back). The
   backfill writes a **separate supplementary file**, `pr_descriptions_backfill.csv.gz`, via
   an atomic `os.replace` from a `.tmp` so a crash can never leave a truncated file.
   **`pr_descriptions.csv.gz` was opened read-only and its mtime is still `Mar 26 22:02`** —
   verified post-run.
3. **Resumable on its own file**, skipping keys already attempted (success *or* error) so a
   restart never re-burns quota on known-dead PRs.

The build script now reads **both** description files.

**Credentials smoke test.** `gh` is not on `PATH` on sk3, but `GITHUB_TOKEN` is exported in
the login environment (`get_gh_token()` checks env first). Verified against the GraphQL API
before spending anything: `viewer.login = alex2awesome`, `rateLimit.limit = 5000`. Then a
**50-PR smoke batch**: 50/50 succeeded, 0 errors, 40 non-empty / 10 empty, and the returned
bodies were manually confirmed to be genuine PR descriptions matching their repos.

**Targets.** Of 59,111 v3 rows: 54,040 resolved to an `owner/repo` (5,071 rows in the 88
unresolved re-scrape dirs were **skipped per instruction**); 0 had a non-numeric
`pr_number`. 30,439 already covered → **23,601 to fetch** in 945 batches of 25.

**Full run** (`backfill_v3.log`, 4 workers, 1,600 req/hr):

| metric | value |
|---|---:|
| attempted | **23,601** |
| succeeded (error-free) | **23,261 (98.56%)** |
| errors | **340 (1.44%)** — *all* `pr not found` |
| of succeeded, **non-empty** body | **20,865 (89.70%)** |
| of succeeded, genuinely **empty** body (counts as covered) | **2,396 (10.30%)** |
| wall clock | 35 min (0.59 h) @ ~40,000 PRs/hr |

The error mode is homogeneous and benign: `pr not found` = the PR was deleted or the repo
renamed//made private since the corpus was scraped. There were **zero** rate-limit pauses,
HTTP failures, or transport errors. Supplementary file: 23,651 rows total (23,601 + the 50
smoke PRs), 23,311 error-free.

## B3. Coverage after backfill

| metric | round 1 | **after backfill** |
|---|---:|---:|
| **body GATE coverage** (attempted & succeeded) | .4738* | **.9085** (53,700 / 59,111) |
| — eval | — | **.9761** |
| — test | — | .9426 |
| — train | — | .8962 |
| body **non-empty** fraction | .4738 | **.8275** (48,913) |
| covered but genuinely empty | — | 4,787 |
| title | .8167 | .8167 (unchanged) |
| ≥1 review comment | .5289 | .5289 (unchanged — comments were not backfilled) |
| body **or** comments | .5289 | **.8686** (51,346) |
| **nothing but the diff** | .1708 | **.0946** (5,589) |

\* round-1 .4738 was the non-empty fraction; the gate-definition equivalent was .514.

Eval coverage of **97.6%** is the number that matters most for the headline comparison — the
eval split is now essentially fully enriched, so a v3-vs-v2 eval-AUC contrast is no longer
confounded by missing bodies. The residual 9.5% fully-bare rows are dominated by the 5,071
skipped re-scrape-dir rows, which are concentrated in train.

## B4. Rebuild

Same frozen spec, same hygiene, same reused v2 split assignment. Hygiene and recency numbers
are **identical to round 1** (same 59,111 rows survive — the backfill changed text content,
not row membership): R1 dropped 37/37 `_deep` repos / 3,666 rows; R2 dropped 122 both-label
groups / 303 rows plus 627 same-label extras; eval recency AUC .4994 pre / .4961 post.

Composition shifts from the richer bodies:

| | round 1 | after backfill |
|---|---:|---:|
| text length mean | 16,888 | **17,444** |
| text length median | 24,000 | 24,000 |
| rows with diff truncated | 32,226 | 32,612 |
| rows where discussion prefix alone exceeds 24k | 22 (.04%) | **216 (.37%)** |

Per instruction I kept the round-1 handling of the over-cap rows (discussion never
truncated; diff budget goes to 0). It grew 10× with more bodies present but is still 0.37%
of rows, and all such rows are cut by the 2,048-token model window regardless. Flagging
again in case a body cap is wanted for a future v3.1.

**Integrity assertions re-run and all passed** (the build aborts otherwise):
every v3 `(repo, pr_number)` exists in v2 · judgement matches v2 · split matches v2 ·
repo overlaps `train&eval=0, train&test=0, eval&test=0`.
Splits unchanged: train 47,659 / eval 5,822 / test 5,630; pos-rates .8172 / .7994 / .8083.

## B5. GATE DECISION

**.9085 ≥ .60 → GATE PASSES → training launched.**

## B6. Training launch evidence

Launched `methods/dense/run_pr_dense_v3.sh` with `GPU=1`, seed 42 only, at
**Wed Aug 5 16:00:50 PDT 2026**. GPU chosen after re-checking `nvidia-smi` at launch time
(GPU 1: 0 MiB, 0%, no compute procs; GPUs 2/4/5 busy with other jobs and untouched).

Verified per server-diligence — all three conditions:

**1. Process.** `ps` shows PID **1809024** (runner shell PID 1809021):
```
/lfs/.../envs/ai_usage/bin/python .../methods/dense/train_reward_model.py
  --data_path .../dense_standard_v3/data.csv --split_dir .../dense_standard_v3/split
  --model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 --learning_rate 5e-5
  --batch_size 16 --eval_batch_size 32 --gradient_accumulation_steps 1 --max_length 2048
  --epochs 2 --gradient-checkpointing --class_weight_auto --selection_split eval
  --seed 42 --output_dir .../dense_standard_v3/rm_out_seed42
```
Recipe is byte-identical to v2's apart from the v3 paths and seed-42-only.

**2. GPU residency.** `nvidia-smi -i 1`: **41,272 MiB used, 100% utilization**, and
`--query-compute-apps` confirms **PID 1809024 resident with 41,258 MiB** on GPU 1.

**3. Real training progress** (`rm_out_seed42.train.log`):
```
Loading checkpoint shards: 100%|██████████| 4/4
Applying LoRA (r=16, alpha=32, dropout=0.050) to q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj
Enabling gradient checkpointing
Train loader batches: 2979 | Eval loader batches: 182 | Train batch size: 16 | Eval batch size: 32
Total optimizer steps: 5958 | Warmup steps: 595 | Grad accum: 1
Class weight auto: 38946 pos, 8713 neg -> pos_weight=0.224
Starting epoch 1/2
Validation cadence this epoch: 5 checkpoints at optimizer steps [596, 1192, 1788, 2384, 2979]
Epoch 1 Step 1 | Batch 1/2979 (0.0%) | Recent avg loss (1 steps): 0.3544
```
Model loaded, LoRA applied, optimizer built, and the loop is emitting real losses.

ETA ≈ **11 h** (2 epochs × 2,979 batches), i.e. finishing ~03:00 PDT 2026-08-06.
Select-on-eval; compare the resulting clean-eval AUC against v2 seed-42 **.5734**.

**Caveat for the comparison.** v3's eval set is v2's eval **minus hygiene drops** (5,822 vs
6,372 rows). The v2 .5734 baseline was computed on the *un-cleaned* eval. Before quoting a
Δ, rescore the existing v2 seed-42 model on the v3 eval row set — otherwise the enrichment
effect is entangled with the removal of the 5 all-merged `_deep` repos (which the audit
already showed moves pooled eval AUC .5734 → .5877 on its own). **The honest reference point
for "did enrichment help" is ~.5877, not .5734.**

## B7. Round-2 artifacts

| artifact | path (sk3) |
|---|---|
| backfill fetcher | `datasets/code-review/fetch_pr_descriptions_backfill.py` |
| backfill data (supplementary, never merged into the original) | `datasets/code-review/pr_descriptions_backfill.csv.gz` |
| backfill log | `datasets/code-review/backfill_v3.log` |
| rebuilt corpus + manifest | `dense_standard_v3/{data.csv,split/,build_manifest.json}` |
| round-1 manifest/log (preserved) | `dense_standard_v3/{build_manifest_round1.json,build_v3_dense_round1.log}` |
| training log | `dense_standard_v3/rm_out_seed42.train.log` |
| runner log | `dense_standard_v3/runner_v3.log` |
| training output dir | `dense_standard_v3/rm_out_seed42/` |

---

# MATCHED BASELINE (appended 2026-08-05, after coordinator approval)

**Purpose.** v3's eval is v2's eval *minus hygiene drops* (5,822 vs 6,372 rows), so the
published v2 numbers are not a legal reference for "did enrichment help." This section
establishes the honest reference pair: **the v2 models, on their native diff-only v2 TEXT,
scored on exactly the v3 row sets.**

## M1. Method — and why no re-inference was needed for the headline

Each `dense_standard_v2/rm_out_seed*/` already contains `preds_eval.csv` / `preds_test.csv`:
canonical per-row `(repo, pr_number, judgement, prob)` from the selected `best_model`. Since
the build asserted that **every v3 row is a v2 row with identical text, judgement and
split**, restricting those stored prediction vectors to the v3 keys *is* the matched
baseline — and it is strictly better than a fresh rescore, because it is the very
prediction vector that produced the published .5734.

Two integrity checks gate this:

1. **Key coverage** — all three seeds, both splits: 5,822/5,822 eval and 5,630/5,630 test v3
   keys present in the stored preds, **0 missing, 0 duplicate keys**.
2. **Provenance** — recomputing AUC over the *full* stored preds reproduces the published
   values **exactly** for all six seed×split cells: eval .5734/.5535/.5713 and
   test .7035/.6706/.6868. The preds are the canonical best_model outputs, not stale.

## M2. Independent GPU confirmation (seed 42, both splits)

To rule out the shortcut being subtly wrong, seed 42's `best_model` was **reloaded and
re-scored from scratch** on v2 text for the v3 rows (`confirm_matched_gpu.py`, same pattern
as `methods/dense/score_eval_pr_v2.py`, bf16, max_len 2048, batch 16).

| | subset-derived | **independent GPU rescore** |
|---|---:|---:|
| matched **eval** AUC (n=5,822) | .5851 | **.5851** |
| matched **test** AUC (n=5,630) | .6618 | **.6618** |

**Exact agreement to 4 decimals on both splits.** The subsetting method is validated; the
seed-1 and seed-2 matched figures below come from that validated method (each also passing
the same provenance check), not from a separate GPU pass.

Server-diligence for the confirmation run: launched only after `nvidia-smi` showed **GPU 0
free (0 MiB, no compute apps)**; PID **1815573** verified resident on GPU 0
(uuid `GPU-d2b20a67…`, 28,616 MiB) while the v3 training stayed on GPU 1
(uuid `GPU-603974b5…`, PID 1809024) untouched; log showed real batch progress; process
exited cleanly writing `matched_baseline_gpu_confirm_seed42.json`; GPU 0 back to 0 MiB after.

## M3. Results — the honest reference pair

v2 model · v2 diff-only text · v3 row sets. 95% CIs are 2,000-sample bootstrap.

| seed | v2 eval (published, n=6,372) | **MATCHED eval (n=5,822)** | Δ | v2 test (published, n=6,298) | **MATCHED test (n=5,630)** | Δ |
|---|---:|---:|---:|---:|---:|---:|
| **42** | .5734 | **.5851** [.5680, .6025] | **+.0117** | .7035 | **.6618** [.6429, .6815] | **−.0416** |
| 1 | .5535 | **.5632** [.5442, .5820] | +.0097 | .6706 | **.6270** [.6078, .6471] | −.0435 |
| 2 | .5713 | **.5841** [.5666, .6018] | +.0128 | .6868 | **.6478** [.6286, .6675] | −.0390 |
| **mean** | .5661 | **.5775** | +.0114 | .6870 | **.6455** | −.0414 |

### The numbers to quote

- **Matched eval baseline = .5851 (seed 42); seed band .5632 – .5851, mean .5775.**
  **v3 must beat .5851 to demonstrate that enrichment helps.** Quoting .5734 would credit
  enrichment with ~+.012 of pure hygiene effect.
- **Matched test baseline = .6618 (seed 42); band .6270 – .6618, mean .6455.**

### Why hygiene moves the two splits in opposite directions

The sign flip is consistent across all three seeds (eval +.010 to +.013; test −.039 to
−.044), so it is compositional, not noise. Both splits lost all-single-class `_deep` repos,
but those repos sat in opposite places in the score distribution:

- In **eval** the five all-merged `_deep` repos were *misranked* — the audit measured their
  mean prob at **.497** against .610 for ordinary repos, despite being 100% merged. They
  were positives the model scored low, so deleting them **raises** pooled AUC (.5734 → .5851,
  in line with the audit's .5877 estimate for deep-repo removal alone; the extra movement is
  the duplicate-group drops).
- In **test** the dropped repos were evidently scored high while being all-merged — free
  correctly-ranked positives — so deleting them **removes** easy wins and pooled AUC falls.

This is the same lesson as §2/§6 of the audit: **pooled AUC on this corpus is dominated by
degenerate-repo composition**, which is precisely why a matched baseline was required and
why cross-corpus AUC comparisons here must always be row-set-matched.

## M4. Artifacts

| artifact | path (sk3) |
|---|---|
| matched baseline, all 3 seeds, both splits (+CIs, provenance checks) | `dense_standard_v3/matched_baseline_v2seed42.json` |
| independent GPU confirmation (seed 42) | `dense_standard_v3/matched_baseline_gpu_confirm_seed42.json` |
| confirmation per-row preds | `dense_standard_v3/matched_baseline_preds_seed42_{eval,test}.csv` |
| confirmation log | `dense_standard_v3/matched_baseline_confirm.log` |
| scripts | `datasets/code-review/{matched_baseline.py,confirm_matched_gpu.py}` |
