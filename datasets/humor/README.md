# Humor

Task in the norm-research portfolio: model **what counts as funny** — i.e. recover
the implicit norms that distinguish jokes/sketches that audiences (or editorial
gatekeepers) reward from those they don't.

This task is one of the two tasks (along with creative-writing) that carry a
substantial **irreducibly-tacit (L4) tail** in the cross-task articulability
analysis — see `notes/2026-05-15__overnight-exploration-log.md` (L4 ≈ 20% for
humor, vs. ≤ 5% for most other tasks). Sampled L4 clusters such as "Commit to
the bit", "Authenticity versus shtick", and "Emotional resonance" are face-valid
taste/craft criteria.

---

## 1. Task

Per-row binary label: `judgement = 1` if a piece of humorous text was rewarded
by its native taste community, else `0`. The unit and reward signal vary by
source (see below); the cross-source idea is that "high judgement" examples
embody more of the latent humor norms than "low judgement" examples drawn from
the same context.

---

## 2. Key decision — New Yorker Caption Contest is RULED OUT

Per memory `feedback_no_newyorker_captions.md`:

> Do not use New Yorker caption ratings as a humor evaluation dataset. The
> ratings are crowd-worker annotated, not expert or genuine humor judgments.
> Crowd ratings on humor have well-known issues: annotator inconsistency,
> surface-level judgments, paid workers don't necessarily have calibrated taste
> for what's actually funny.

The original task spec in `running-research-notes.md` (lines 11–17, currently
**stale**) describes the NY Caption Contest setup
(`newyorker_caption_ratings.csv.gz`, 2.2 M Turker ratings, `judgement = 1` if
`mean >= 1.3`). That plan is no longer the active humor task.

Files that remain on disk for reference only — **do not use for modeling or
rubric extraction**:

- `newyorker_caption_ratings.csv.gz` (77 MB) — 2.2 M Turker ratings
- `newyorker_cartoon_descriptions.csv.gz` (32 KB) — cartoon metadata

Replacement portfolio: Reddit r/Jokes (audience signal), r/StandUpWorkshop
(critique-bearing community), McSweeney's editor rejections (editorial expert
gating), and the A Special Thing forum archive (comedy-community taste). See §3
for current status.

---

## 3. Sources

### 3a. r/Jokes — Reddit upvotes (primary modeling corpus today)

- **Raw**: `reddit_jokes_1m.csv.gz` (299 MB)
- **Build script**: `build_reddit_humor_dataset.py`
- **Canonical labeled dataset**: `reddit_humor_modeling_dedup.csv.gz` (22 MB)
- **Labeling**: within each (length_bin × format × topic) cell, top 25% by
  score → `judgement = 1`, bottom 25% → `0`, middle 50% dropped. Length binned
  in 100-char buckets; format ∈ {narrative, one_liner, riddle}; topics from
  LDA with 30 components on word counts (custom stoplist).
- **Confound controls baked in**:
  - Edit-marker stripping (`EDIT:`, `Update:`, `TL;DR`, "thanks for the gold")
    because those only appear on already-popular posts — pure leakage.
  - NSFW filter (lexical + subreddit flag).
  - Length cap [30, 5000] chars.
  - MinHash LSH (5-shingles, Jaccard ≥ 0.8, 128 perm) over word-level shingles
    + union-find clusters → one row per cluster, canonical text = longest
    variant, score = mean across cluster. Addresses leakage audit item #2
    (repeat jokes; `notes/dataset_leakage_audit.md` line 151).
- **Comment-explanation check**: `check_joke_comments.py` confirmed empirically
  (via arctic-shift Pushshift mirror) that r/Jokes top comments are mostly
  riffs, not explanations of why a joke is funny — so no
  `_with_reasoning` variant is needed (audit item #1).

### 3b. r/StandUpWorkshop — community critique (rubric source)

- **Location**: `standup_reddit/`
- **Pipeline**: `sampled_threads.jsonl` (raw posts + comments) →
  `filter_comments.py` → `filtered_threads.jsonl` + `filtered_comments.jsonl`.
- **Filter rationale** (per `standup_reddit/filter_comments.py`): comments
  ≥ 100 chars, exclude AutoModerator / bot patterns / `[deleted]` /
  `[removed]`, require ≥ 2 substantive comments per thread. Per-thread output
  concatenates OP + threaded comments in score order.
- **Use**: critique-bearing source for rubric / norm extraction, not (yet)
  a binary-labeled modeling target. `build_rubrics.py` ingests the merged
  groups from the two hierarchies at
  `outputs/hierarchy/humor_general_r2_expanded.json` and
  `outputs/hierarchy/humor_specific_r2_expanded.json` and emits
  `standup_reddit/rubrics.jsonl`.

### 3c. McSweeney's Internet Tendency — editor rejections

- **Location**: `mcsweeneys_rejections/`
- **Output**: `pairs.jsonl` — **30 verbatim editor rejection responses**
  harvested 2026-06-01 from 8 source pages (Sean Hewlett WordPress, Victor
  Beigelman Substack, Rejection Wiki Internet Tendency + Quarterly, etc.).
- **Substantiveness** (per `harvest_log.md`): ≈ 9/30 carry diagnostic craft
  critique (premise vs. execution, hook timing, target too easy, category
  saturation), ≈ 10/30 are brief polite passes with light positive notes,
  ≈ 10/30 are pure form letters.
- **Constraint**: submitted-piece text is missing for **all 30** pairs —
  authors quote rejections but withhold submissions. This caps the corpus at a
  *supplementary / anchor* role for the norm-extraction track, not a training
  corpus.

### 3d. A Special Thing forum (Wayback)

- **Script**: `scrape_aspecialthing.py` (three stages: `--stage cdx`,
  `--stage fetch`, `--stage parse`).
- **Storage** (on sk3): `/lfs/skampere3/0/alexspan/data/humor/aspecialthing/`
  with `raw/`, `parsed/threads_normalized.jsonl`, and `logs/`.
- **Targets**: vBulletin `showthread.php?t=NNN`, `NNNN-postN.html` permalinks,
  `forumdisplay.php?f=N` section listings; one earliest-200 capture per URL.
- **Use**: comedy-community taste discussion as additional rubric source.

### 3e. #HashtagWars — SemEval-2017 Task 6 (editorial verdicts, NEW 2026-07-28)

- **Location**: `hashtagwars/` (own README with provenance/license/counts).
- 12,734 labeled tweets across 112 hashtag files (train 101 / trial 5 /
  gold-eval 6). Verdict = @midnight show staff picks: label 0 not-top-10,
  1 top-10, 2 episode winner. Loader `hashtagwars/load.py` yields
  (hashtag, tweet_text, label).

### 3f. WaPo Style Invitational — nrars.org Book of Weeks (editorial verdicts, NEW 2026-07-28)

- **Location**: `style_invitational/` (own README with provenance/counts +
  resume instructions; raw text files preserved under `raw/01_text/`).
- Weekly contest results 1993–2023; per-entry tier verdicts winner /
  runnerup / honorable_mention in `style_invitational.jsonl`
  (week_id, contest_prompt, entry_text, tier). First ~400 weeks collected;
  full archive is ~1,530 weeks (see its README for how to resume).

### 3g. online-rubrics (expert / blog craft corpus, already harvested)

- `online-rubrics/raw/` — 4,607 HTML pages of standup / sketch / humor-writing
  craft material (e.g. `act_out_stand_up_comedy.md`,
  `apte_anthropological_humor_universals.md`).
- `online-rubrics/claude-parsed/` — 230 parsed markdown extracts.
- Feeds the `humor_general` / `humor_specific` / `humor_hyper_specific` rubric
  hierarchies at `outputs/hierarchy/humor_*.json` (r1 → r2_expanded → r3).

---

## 4. File layout

```
datasets/humor/
├── README.md                                ← this file
├── build_reddit_humor_dataset.py            stratified labeler for r/Jokes
├── check_joke_comments.py                   audit: jokes have no reasoning replies
├── reddit_jokes_1m.csv.gz                   raw r/Jokes dump (299 MB)
├── reddit_humor_modeling_dedup.csv.gz       canonical labeled dataset (22 MB)
├── newyorker_caption_ratings.csv.gz         ARCHIVED — do not use (NY captions)
├── newyorker_cartoon_descriptions.csv.gz    ARCHIVED — do not use (NY captions)
├── scrape_aspecialthing.py                  Wayback CDX scraper for ASpecialThing
├── mcsweeneys_rejections/
│   ├── pairs.jsonl                          30 verbatim editor rejections
│   └── harvest_log.md                       per-source provenance log
├── standup_reddit/
│   ├── sampled_threads.jsonl                raw r/StandUpWorkshop threads
│   ├── filter_comments.py                   substantive-comment filter
│   ├── filtered_threads.jsonl               per-thread text for rubric mining
│   ├── filtered_comments.jsonl              per-comment rows
│   ├── build_rubrics.py                     emits rubrics from hierarchy JSONs
│   └── rubrics.jsonl                        canonical rubric vocabulary
└── online-rubrics/
    ├── raw/                                 4,607 HTML pages
    ├── claude-parsed/                       230 markdown extracts
    ├── gpt-parsed/
    └── urls-visited.csv, waveh{3..6}_*      crawl logs
```

---

## 5. Canonical dataset file

**`reddit_humor_modeling_dedup.csv.gz`** — `text` + `judgement` columns,
deduped via MinHash, balanced 1 / 0 by the stratified labeler. This is the
file modeling jobs should load for the humor task today.

Group-split / leakage audit status: the audit at
`notes/dataset_leakage_audit.md` (lines 144–156) flags items #2 (repeat jokes)
as **mitigated by MinHash dedup in the current build script**; #3 (length /
punchline confounder) and the dark-joke subreddit-norm point are not yet
explicitly audited on the deduped file.

---

## 6. Modeling state

No published cross-task modeling result for humor on the new (r/Jokes-based)
canonical dataset is recorded in memory yet. The previous task spec used the
ruled-out NY captions, so prior numbers do not transfer. Humor is included in
the dense-sweep / norm-extraction infrastructure that runs on sk3 — see
`reference_v2_task_datasets.md` and the per-task pipelines under
`outputs/v2_db/` — but a clean point-estimate AUC for r/Jokes-deduped is still
pending.

What we do know about humor as a task (cross-task analyses, not a model
result):

- Articulability bucket distribution (after the L3-merge correction,
  `notes/2026-05-15__overnight-exploration-log.md` line 102):
  L1 ≈ 3%, L2 ≈ 62%, L3 ≈ 16%, **L4 ≈ 20%**, surface-vs-substance mean ≈ 2.52
  (high-substance like creative-writing). The L4 elevation is robust to source
  orientation.
- This places humor with creative-writing at the high-tacit-tail end of the
  portfolio.

---

## 7. Open questions / next steps

1. **Run the cross-task modeling pipeline on the r/Jokes-deduped file** and
   record the result alongside other v2 tasks. Without this number we cannot
   place humor on the L1–L4 verifiability ladder empirically.
2. **Complete the leakage audit on the deduped file** for items #3 (length /
   punchline confound after the topic+length stratification) and the dark-joke
   subreddit-norm effect.
3. **Decide the role of McSweeney's pairs**. 30 pairs is too small to train on
   and all lack submission text. Useful as a held-out qualitative probe of how
   well humor-norm models surface the same diagnostic critiques that real
   editors give (e.g. "premise vs. execution", "target too easy").
4. **Finish the A Special Thing parse** (`scrape_aspecialthing.py --stage
   parse`) and decide whether community taste discussion belongs in the rubric
   corpus alongside online-rubrics + StandUpWorkshop.
5. **Update `running-research-notes.md` § Humor** — it currently still
   describes the NY captions setup. Replace with the r/Jokes-based portfolio
   above.
6. **Cross-source norm comparison**: do the norms that r/Jokes upvotes select
   for (audience signal) overlap with the norms that McSweeney's editors apply
   (editorial gatekeeping) and that r/StandUpWorkshop critics articulate
   (peer-craft critique)? This is the central interesting question for the
   humor task within the broader articulability-ceiling framing.
