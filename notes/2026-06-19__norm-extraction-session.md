# Norm-extraction pipeline session — 2026-06-16 → 19

## What this session covered

A multi-day iteration on the 23-corpus Qwen-122B norm-extraction pipeline. Three phases:
1. **Recovery**: fixed long-running gzip-append corruption + recovered 303K records
2. **Quality**: fixed validator bug, parallelized across 6 GPUs
3. **Framing audit**: spot-checked all 23 corpora through a sharper `(doc X, comment-on-X)` lens; iterated v2 prompts for 5 corpora; identified 2 to drop

The framing pivot (mid-session) is the most important conceptual outcome — see §4.

---

## 1. Recovery (Phase A, fixed before session)

### Problem
The runner wrote all batches to a single gzip file via `gzip.open(path, "at")` which corrupts long-running deflate streams past the last sync point. Six corpora's extracted files were partially unreadable.

### Fix
- Patched `run_sk3_batch.py` to write each batch to its own chunk file: `chunk_<run_ts>_<counter:06d>.jsonl.gz`, opened with `"wt"` and closed via context manager.
- Added `migrate_legacy_to_chunks()` to peel readable lines out of corrupted files into one preserved "legacy" chunk per task.
- **Recovery**: 303,038 records preserved from 6 corrupted legacy files. Originals renamed `.legacy_corrupted` per never-delete-data policy.

---

## 2. Validator bug + throughput crisis (Phase B)

### Symptom
Within 6 batches of restarting, throughput degraded from 22.4/min → 14.6/min, and final-fail rate climbed from 5.9% → 41.6%. Projected ETA for 11-task queue: 45 days.

### Diagnosis
`is_loop_string(s)` in the validator had:
```python
if len(s) > 800: return True  # treats any string >800 chars as "bad"
```
This was applied to `passage_text`. Real review passages routinely exceed 800 chars — long reviewer paragraphs were getting flagged as corruption, retried 5 times (all producing the same long output), then marked `bad_after_retries`.

Length-vs-fail-rate confirmed it:
| Input length | fail rate |
|---|---:|
| 0-500 chars | 0% |
| 2000-4000 | 49% |
| 4000-8000 | 71% |
| 8000+ | 82% |

### Fix
Removed the `len > 800` line; kept only the regex loop check `(.{4,40})\1{8,}`.

### Result
33 post-fix batches: rate stable at 23.5/min, cumulative fail rate steady ~3%. Throughput recovered fully.

---

## 3. Parallelization (Phase C)

User authorized using all available GPUs. Launched 6 concurrent vLLM runners on B200s. Math.SE and Law.SE (short-text comments) ran at **267/min and 227/min** — ~11× faster than peer_review (23.5/min for multi-page reviews). Wall-clock for the current 11-task queue collapsed from ~24d to ~1.5d.

Notable findings:
- `code_review`, `humor_multi`, `press_releases_full` were already 100% covered by legacy chunks — runners auto-skipped them.
- math_se DONE: 97,203 ok / 10 final-fail / 100K total (374 min)
- law_se DONE: 141,597 ok / 1 final-fail / 146K total (568 min)

GPU 2 was repurposed mid-session to start Wave 4 tasks (litbench, nc_public_comments, courtlistener). GPU 3 was repurposed to aops/bva/cavc. GPU 7 was repurposed to nlrb/ttab/ptab/dol.

---

## 4. The framing pivot (most important conceptual shift)

### User clarified
The pipeline is collecting **noisy distant labels for norms** — phrases that reviewers/commenters used to express normative judgments about a document, which can then be matched to candidate metrics m in our existing bank.

Frame: `(doc X, comment-on-X) → signals about X's craft`. A signal_text is one noisy distant label.

### What this changed
Many corpora that were extracting *something* were extracting the **wrong something**:
- law_se: Posts.xml edit summaries ("added 26 characters in body") were being extracted as norm signals
- reddit_supremecourt: pure political opinions were extracted as legal-reasoning critiques
- legaladvice_uk: fresh advice given to OP was extracted as feedback on prior advice
- press_releases_full: extracting facts from news articles as if they were craft critiques
- competition_editorials: editorials are the doc X itself (author self-narration), not feedback on it
- nc_public_comments: pure policy assertions, not comments-on-X
- humor_multi: AST forum posts are show flyers, not feedback on jokes
- aops_forum: `post_type=solution` records are doc X, only `post_type=reply` are feedback

### Why this matters for downstream
Signal clusters → norm vocabulary → match against R2/R3 metric leaves. If clusters are polluted with non-norm content (substantive facts, advice, etc.), the validity matrix will be noisy. Distant-supervision benefits from noise, but only if the noise is in the labels of TRUE norm cases — not entirely off-task records being labeled as "fired".

---

## 5. Spot-check verdicts on all 23 corpora

### ✅ Clean (keep v1) — 6 corpora
- **peer_review** — gold-standard ICLR/NeurIPS reviews
- **math_se** — clean critique under answers (97% ok rate, ~0% real fail)
- **crse** — CR.SE answers as code review
- **wp_comments** — reader feedback on stories
- **code_review** — GitHub PR comments (caveat: 28% input truncation at 16K context)
- **litbench_rationales** — LLM-written craft rationales fit `comment-on-stories` frame

### 🟡 Adjudication track (keep v1, but DON'T mix with craft signals downstream) — 7 corpora
- nlrb_decisions, ttab_inter_partes, bva_opinions, cavc_decisions, ptab_fwd, dol_arb, courtlistener_opinions
- Configs were already adjudication-aware. NLRB confirms model produces 84% observations / 5% complaints — inverse of craft corpora, correct for tribunals applying standards.
- These should be analyzed in a separate research track (legal-standards-applied norms).
- Caveat: `dol_arb` inputs carry OALJ web-template HTML chrome that should be stripped pre-extraction.

### 🟠 v2 prompts deployed — 5 corpora
- **law_se** v2 + input filter (drop 57K `edit_` records, keep 89K `comment_*`)
- **reddit_supremecourt** v2
- **legaladvice_uk** v2
- **press_releases_full** v3 (softened — see §6)
- **notice_and_comment** v2 (agency-critiques-commenter direction; SEC dockets explicitly excluded)

### 🟡 Input filter only — 2 corpora
- **humor_multi** — drop AST forum records (49,762 → 48,505)
- **aops_forum** — keep only `post_type=reply` (30,000 → 15,000)

### 🔴 Drop from norm-extraction — 2 corpora
- **competition_editorials** — editorials ARE the doc X (author self-narration). Would need to scrape LeetCode/Codeforces *community discussion threads* as a separate effort to get real comment-on-X data.
- **nc_public_comments** — policy assertions, not comment-on-X. No comment-on-X structure exists; should be reframed as a completely different task (policy-position classification) or dropped.

---

## 6. The press_releases_full iteration story

A multi-step iteration that illustrates the prompt-engineering vs data-engineering interaction.

### Step 1: original v2
Drafted with strong "expect SPARSE / many will return empty" language. Result on 6 random articles: ALL EMPTY. Cause: my prompt was over-pushing the empty-by-default behavior.

### Step 2: clean-article sweep
Sampled 50 articles, scored by prose-density heuristic. Top 5 articles had clean body text. Tested v2: ALL EMPTY again. Reason: even the cleanest articles were mostly straight reporting with no editorial judgment of the PR.

### Step 3: meta-marker sweep
Sampled 200 articles, scored by regex matches for meta-commentary markers ("declined to comment", "critics said", "former employees said", "notably absent", etc.). Top 6 included **NYT's Theranos coverage** (pair_159975_449577). Tested v2: still ALL EMPTY.

### Step 4: diagnostic — relaxed prompt on Theranos
Stripped the v2 prompt down to a plain instruction: "extract any phrases where the journalist notes what the company didn't say, reports contestation by third parties, identifies inconsistencies, etc." Result on Theranos:
- "Mr. Balwani declined to comment further" — non-disclosure norm ✅
- "Officials from CMS also declined to comment" — third-party non-disclosure ✅
- "The company insisted that the most recent changes were not a result of the regulatory pressure" — defensive framing norm ✅
- "subject of numerous critical articles in The Wall Street Journal questioning its proprietary technology" — prior-contestation norm ✅

**Conclusion**: v2 was over-rejecting. The "expect SPARSE" / "many will return empty" / "DO NOT force-extract" instructions were biasing Qwen toward empty even when real signals existed.

### Step 5: v3 (softened)
Removed the conservative bias. Kept the marker-based extraction guidance ("DO extract when you see X, Y, Z"). User confirmed: "what matters is true positives" — false negatives on empty articles are fine; missing true positives is not.

### Lesson
Negative-example prompt engineering can backfire. The model can over-learn "expect empty" and stop extracting even when extraction is warranted. Better to give explicit positive markers + brief negative criteria than to load the prompt with empty-by-default rhetoric.

---

## 7. notice_and_comment — what actually gets extracted

Spot-check of existing chunks showed v1 IS already extracting real agency-critiques-commenter signals, just diluted by rule-announcement boilerplate. Example from EPA Montana SIP (`EPA-R08-OAR-2018-0136-0006`):

> "After reviewing the comments, the EPA has determined that the comments are outside the scope of our proposed action or fail to identify any material issue necessitating a response."

Signals extracted:
- `[complaint/negative]` "comments are outside the scope of our proposed action" → **norm: comments must be within scope of proposed action**
- `[complaint/negative]` "fail to identify any material issue necessitating a response" → **norm: comments must identify a material issue**

Other recurring patterns:
- Scope-of-action norm
- Materiality norm
- Anti-backsliding-in-rulemaking
- Safe-harbor-on-no-gain

v2 prompt sharpens toward these by:
1. Targeting agency = speaker, public comment = target (one direction only — earlier config straddled both)
2. Adding markers: "we agree/disagree with the commenter", "the commenter did not provide", "outside the scope", "addressed in section X", etc.
3. Explicitly excluding SEC self-regulatory dockets (Commission adjudicating exchange rule filings — wrong speech act)

Open issue: rtc_text averages 50K chars; most is preamble. Real per-comment slicing on "Response to Comment X" markers would give sharper signal attribution, but for now v2 with the 16K context window will at least extract from whichever sections fit.

---

## 8. Distant-label → metric-bank validity matrix

Conceptual outcome from the framing discussion. Rather than "cluster signals → match to existing metric bank" as a vocabulary-mapping exercise, the correct framing is:

```
For each (doc X, metric m_j, signal cluster c_k):
  Is m_j(X) correlated with [c_k fires on X across many docs]?

Validity matrix:
  rows = existing metrics m (R2/R3 leaves)
  cols = signal clusters c
  cells = correlation across docs
  
Read:
  Diagonal strong → metric validly captures its norm
  Off-diagonal strong → confounded metric or shared semantic
  Column with no diagonal → gap (need new metric for this norm)
  Row with no column firing → dead metric (no reviewers care)
```

This is **distant supervision** (Mintz/Snow/Bach line; analog to Baumann label-noise work). Clustering is an efficiency move for cheaper m-matching at scale, not a correctness requirement. Per-signal (X, Y, m) classification preserves more noise structure and is more faithful to distant-supervision practice.

The unit of distant supervision is **(X, Y) → m\***, where Y is the individual signal text. m* is the metric this Y points at. Then validity check: does \hat{m}(X) correlate with m*(Y)_for_X across docs.

---

## 9. What's deployed on sk3 now

### Configs (`scripts/llama_norm_extraction/configs/`)
- `law_se_v2.json` — drops edit_extraction guidance, adds explicit "RETURN EMPTY WHEN" criteria
- `reddit_supremecourt_v2.json` — explicit reasoning-move requirement, drops political opinion / ad-hominem / speculation
- `legaladvice_uk_v2.json` — focuses on commenter-correcting-prior-advice; drops fresh advice / sympathy / referrals
- `press_releases_full_v2.json` — softened v3, marker-based extraction (sourcing, defensive framing, contestation, omission, timing, claim-discrepancy, novelty)
- `notice_and_comment_v2.json` — agency-critiques-commenter direction; excludes SEC self-regulatory dockets

### Filtered inputs
- `data/law_se/input_v2_comments_only.jsonl` (89,194 records, dropped 57,087 edit_ summaries — 39%)
- `data/humor/standup_multi/input_v2_no_ast.jsonl` (48,505 records, dropped 1,257 ast_forum show flyers)
- `data/aops_forum/input_v2_reply_only.jsonl` (15,000 records, dropped 15,000 solutions)

### Patches
- `run_sk3_batch.py` line 284 — chunked write (Phase A)
- `run_sk3_batch.py` `is_loop_string` — dropped 800-char cap (Phase B)

### Still TODO
- Add new `TASK_SOURCES` entries in `run_sk3_batch.py` for the 7 v2 task names (e.g., `law_se_v2`, `humor_multi_v2`, etc.) pointing at the filtered inputs + v2 configs + new output dirs (`extracted_qwen_v2.chunks/`). v1 chunks preserved.

---

## 10. What to expect when current runs finish

### Throughput projections
- Current parallel queue finishes in ~1.5 days (longest pole: peer_review at ~35h)
- Wave 4 (10 tasks) splits across GPUs as they free
- GPU 7 frees in ~5h (smallest queue: nlrb+ttab+ptab+dol)

### Data on disk after current runs
**Craft-feedback corpora** (use for primary norm-clustering):
- peer_review: ~258K records when done
- code_review: 93K (legacy, already complete)
- math_se: 97K (DONE)
- crse: 14K (DONE)
- wp_comments: 16,729 (DONE)
- humor_multi (post-filter): 48K
- aops_forum (post-filter): 15K
- litbench_rationales: 43K (running)
- press_releases_full: ~85K (v1 data — likely noisy; v3 re-extraction will sharpen)

**Conversational-thread corpora**:
- law_se v2 (post-filter): 89K
- reddit_supremecourt v2: 58K
- legaladvice_uk v2: 50K
- notice_and_comment + v2backfill: ~11K (with v2 re-extraction sharpening agency-critiques-commenter signals)

**Legal adjudication corpora** (separate track):
- nlrb: 5,604 + ttab: 2,054 + bva: 15K + cavc: 15K + ptab: 4,863 + dol: 1,742 + courtlistener: 30K
- Configs already adjudication-aware

**Dropped**:
- competition_editorials (no comment-on-X structure)
- nc_public_comments (wrong frame; no config; never started)

### Downstream pipeline (post-extraction)
1. For each corpus, gather all `signal_texts` from `ok` records (plus `ok=False but parsed` records — these have `passage no_match` flag but valid data)
2. Build (unit_id, signal_text, signal_type, polarity, source_corpus) table
3. Per-corpus clustering (or skip and do per-signal classification): UMAP+HDBSCAN with operating point `tw=0/eps=0 + LLM dedup` per `project_local_explanations_clustering_findings`
4. Cluster naming via gpt-5-mini
5. Match clusters to existing R2/R3 metric bank by cosine + judge confirm
6. Build the validity matrix described in §8

### Quality expectations after re-extraction
Based on the law_se/reddit/legal_advice OpenRouter tests (~60→90% precision after v2):
- Effective extraction success: ~95-99% per task (after filtering on parsed-data-present, not strict ok=True)
- True-positive recall: high on real feedback comments; the 3 already-tested v2s caught all real-critique samples
- False-positive rate: dropped substantially (no longer extracting from substantive asides, OP follow-ups, edit summaries, AST flyers, etc.)

### Open follow-up items (not blocking the current run)
- Add `TASK_SOURCES` entries for v2 variants and launch when GPUs free
- For `press_releases_full`: data engineering effort to strip web chrome via readability/newspaper3k would meaningfully improve signal yield
- For `notice_and_comment`: per-comment slicing on "Response to Comment X" markers would give sharper signal attribution
- For `dol_arb`: strip OALJ HTML template before processing
- For legal adjudication corpora: cluster separately from craft corpora; analyze as their own research track ("legal-standards-applied norms")
- For `nc_public_comments`: consider as a policy-position-classification dataset, not a norm-extraction one
- Consider scraping LeetCode discussion threads + Codeforces problem comments to get real (doc X, comment-on-X) data for the competitive-programming domain
