# math-aops

Norm-research task built on **Art of Problem Solving** (AoPS) community forum
threads. The unit of analysis is a topic (a posted problem + its replies); the
label of interest is the perceived **quality** of problem statements and
candidate solutions, as expressed through community-level engagement signals
(per-post `thanks`, edit counts, etc.).

This task sits alongside `math-stackexchange/` in the broader "math aesthetics"
arm of the project. Status: **data collection in pilot phase, no modeling yet.**

---

## 1. Task

We want to predict / explain **proof and solution quality** on AoPS using
articulable rubric dimensions, building on the academic literature on
mathematical aesthetics (see
`~/.claude/projects/.../memory/project_math_elegance_research.md`).

Working list of measurable dimensions (from Inglis & Aberdein 2015 and
Johnson & Steinerberger 2019):

| Dimension   | Working gloss                                                   |
|-------------|-----------------------------------------------------------------|
| Elegance    | Concise, surprising, hits the key idea cleanly                  |
| Profundity  | Connects to deeper structure, generalises, reveals "why"        |
| Clarity     | Easy to follow; well-structured exposition                      |
| Precision   | Rigour, no hand-waving, fully justified steps                   |

Engagement signal we have natively from AoPS:

- `thanks_received` / `nothanks_received` per post (relative preference within
  a topic)
- `num_edits`, `last_edit_reason` (effort / community correction)
- `n_solutions` (how many alternative writeups exist for the same problem)

AoPS has no "accepted answer" concept — `is_accepted` is always `null` in our
parsed schema (see header of
`scripts/aops_parse_shards.py:30`). So preference must be inferred from
thanks-deltas within a thread, not from an accept marker as on Stack Exchange.

---

## 2. Sources

- **Primary:** Art of Problem Solving community forum, ajax endpoint
  `https://artofproblemsolving.com/m/community/ajax.php`, fetched by topic id.
- Topic ids appear to densely cover the forum; pilot range was
  `[495500, 495600)`.
- MathOverflow is **not** crawled in this directory; it is collected separately
  under `datasets/math/stackexchange/` (where Stack Exchange's upvote +
  accepted-answer structure provides cleaner preference labels).

The forum sits behind Cloudflare, so all fetching is done through a real
Chromium context (Playwright + `playwright_stealth`), warming session cookies
on `/community` before POSTing to the ajax endpoint.

---

## 3. Collection scripts

All under
`/Users/spangher/Projects/stanford-research/norm-research/datasets/math/aops/scripts/`.

| Script | Purpose |
|---|---|
| `aops_fetch_playwright.py` | Single-topic fetcher; takes one or more topic ids on the CLI, writes `raw/topic_<id>.json` each. Used for prototyping / one-off inspection. |
| `aops_bulk_crawl.py` | Scaled crawler. One process = one Chromium context = one worker. Iterates a topic-id range, writes `raw/shards/<shard>__w<worker>.jsonl.gz` (gzipped jsonl, ~5–10× smaller than flat json) + a `.done` ledger for resumability. Supports `--worker i --num-workers N` so multiple workers can share a range modulo N. |
| `aops_parse_shards.py` | Reads all `raw/shards/*.jsonl.gz` and emits the normalised per-topic schema (see §5) to `parsed/topics.jsonl.gz`. |

Reference / third-party code (not part of our pipeline, kept for borrowed
parsing logic):

- `_ref_aops_instruct/` — full clone of the AoPS-Instruct repo
  (Mahdavi et al. 2025, arXiv:2501.14275). Their `aops_crawler/`, `parse_aops.py`,
  `classify_aops.py` were the starting point for our own scripts. See
  `_ref_aops_instruct/README.md`.
- `_ref_shoryamalani/` — a small login + progress-tracking utility
  (`aops_login.py`, `scrape_progress.py`). Used only for reference; we do not
  log in (anonymous session cookies suffice for public topics).

---

## 4. File layout

```
datasets/math/aops/
├── README.md                        # this file
├── scripts/
│   ├── aops_fetch_playwright.py     # single-topic Playwright fetcher
│   ├── aops_bulk_crawl.py           # scaled, resumable, sharded crawler
│   └── aops_parse_shards.py         # jsonl.gz shards → normalised topics
├── raw/
│   ├── topic_495607.json            # example single-topic fetch
│   └── shards/
│       ├── pilot100__w0.jsonl.gz    # gzipped one-topic-per-line raw responses
│       └── pilot100__w0.done        # text file of completed topic ids
├── parsed/                          # currently empty; populated by aops_parse_shards.py
├── logs/
│   ├── pilot100__w0.log             # per-worker append log
│   └── pilot100.stdout
├── _ref_aops_instruct/              # third-party reference repo (AoPS-Instruct)
└── _ref_shoryamalani/               # third-party reference (login helper)
```

Current data on disk: one pilot shard, `pilot100__w0`, covering topic-id range
`[495500, 495600)`. From `logs/pilot100__w0.log`:

```
[Fri May 29 23:27:06 2026] start range=[495500,495600) shard=pilot100 worker=0/1 delay=2.0s
[Fri May 29 23:30:27 2026] done shard=pilot100 worker=0 seen=100 ok=100 err=0
```

100/100 topic ids fetched, no HTTP errors. Note many topics legitimately
return `E_NO_PERMISSION` or `E_NO_SUCH_TOPIC` from the AoPS endpoint — those
get logged but the response body is still saved as a row so we don't re-fetch.

---

## 5. Canonical dataset file

`parsed/topics.jsonl.gz` (will be) produced by

```bash
python scripts/aops_parse_shards.py \
    --shard-glob '*.jsonl.gz' \
    --out parsed/topics.jsonl.gz \
    --min-solutions 0
```

Per-topic schema (from the docstring at
`scripts/aops_parse_shards.py:7`):

```jsonc
{
  "topic_id":      int,
  "topic_title":   "string | null",
  "category_path": ["..."] ,
  "n_posts":       int,
  "first_post": {
    "post_id": int, "user": "...", "user_id": int, "ts": int,
    "canonical": "LaTeX/Markdown source",
    "rendered_html": "...",
    "thanks": int, "nothanks": int,
    "num_edits": int, "last_edit_reason": "string | null"
  },
  "solutions": [
    {
      "post_number": int, "post_id": int,
      "user": "...", "user_id": int, "ts": int,
      "canonical": "...", "rendered_html": "...",
      "thanks": int, "nothanks": int,
      "num_edits": int, "char_len": int,
      "is_accepted": null              // always null on AoPS
    }
  ],
  "n_solutions": int,
  "max_thanks":  int,
  "sum_thanks":  int,
  "earliest_ts": int,
  "latest_ts":   int
}
```

The split between `first_post` (the problem statement) and `solutions`
(everything posted after, in order) is the central structural choice: it lets
us model "is this a good problem statement?" and "is this a good solution to
*this* problem?" as separate tasks while keeping them paired.

`parsed/` is empty right now because we have not yet decided on a final
`min_solutions` filter and the pilot shard is too small to be the canonical
file.

---

## 6. Modeling state

**No modeling has been done on math-aops yet.** This dataset is not currently
in the `runs/` tree, not in the v2 cells DB (`outputs/v2_db/cells_v1/`), and
not in the dense-reward-model sweep grid.

What exists is exclusively the collection pipeline above and one 100-topic
pilot shard. Before modeling can start we need:

1. A non-pilot crawl (probably ≥ 100K topics) so per-topic stats stabilise.
2. A decision on the label: per-post thanks-rank within topic, vs. binary
   above-median, vs. pairwise.
3. Topic filtering — AoPS covers everything from "high school competition" to
   "research-level olympiad", and quality norms differ sharply by subforum.

---

## 7. Key decisions

- **Four-dimension rubric (elegance / profundity / clarity / precision).**
  Adopted from Inglis & Aberdein 2015 and Johnson & Steinerberger 2019, since
  both factor-analytic studies converge on essentially these axes and find
  that beauty and simplicity are largely uncorrelated. See
  `~/.claude/projects/.../memory/project_math_elegance_research.md` for the
  literature review.
- **AoPS instead of (or in addition to) Math.SE.** AoPS topics tend to be
  problem-centric and attract multiple distinct solution attempts in one
  thread, which is the structure we want for solution-vs-solution preference
  modelling. Math.SE's upvote signal is suspected to track "clarity of
  exposition" more than "elegance of mathematics".
- **Engagement-as-label is provisional.** `thanks` on AoPS is a noisy proxy;
  we expect to eventually need expert relabelling on a subsample (cf.
  ProofBench / IMO-GradingBench / Open Proof Corpus, listed in the memory
  file), but the engagement signal is what makes the corpus scale.
- **Cloudflare-aware fetching.** We POST through a live Chromium context
  (`page.evaluate(fetch(...))`) so the request inherits `cf_clearance` and the
  browser's TLS fingerprint. A pure `requests`/`httpx` client is consistently
  challenged.
- **Gzipped jsonl shards over flat per-topic json.** Saves ~5–10× on disk and
  makes shards trivially parallel-writable (one file per worker). The flat
  `raw/topic_<id>.json` form is kept only as the output of the single-topic
  prototyping script.

---

## 8. Open questions / next steps

- **Run a real crawl.** Pilot covered 100 ids; a full sweep is needed to get
  enough topics with ≥3 distinct solutions for within-topic preference labels.
- **Editorial-similarity parallel (added 2026-06-10, mirrors the LeetCode pivot).**
  The forum crawl has no reference solutions, but the **AoPS Wiki**
  (`artofproblemsolving.com/wiki`) hosts community-canonical solutions for
  AMC / AIME / USAMO / IMO / Putnam problems, plus **official answer keys**.
  Two label/V routes this unlocks:
  1. **V via answer keys**: AMC (multiple choice) and AIME (integer 0–999)
     have determinate answers — extract the claimed final answer from a forum
     solution, check against the key. A genuinely *code-checkable* correctness
     bit for competition math, no autoformalization needed.
  2. **y via editorial similarity**: per-solution max embedding similarity to
     the problem's wiki solutions, exactly the LC editorial-similarity recipe
     (`project_lc_editorial_similarity_pivot_2026_06_02.md`).
  Contest threads are identifiable in the crawl (pilot shard contains
  olympiad/AMC/AIME markers); the wiki is a separate, easy MediaWiki scrape.
  Thanks-deltas then become the *taste* layer on top of a verified-correct
  subset: among solutions with the right answer, which one does the community
  thank? That is the V-controlled preference signal we want.
- **Label definition.** Settle on the head label: pairwise (better-of-two
  posts in same thread), regression on log-thanks, or rubric-dimension scores
  from an LLM judge. The rubric-judge route is most consistent with the rest
  of the project but unvalidated on math.
- **Subforum / topic filtering.** Parse `category_path` and decide which
  subforums are in-scope. Olympiad-level threads probably behave very
  differently from "Middle School Math" threads.
- **Online-rubrics integration.** Other tasks have a sister
  `<task>/online-rubrics/` directory of crowd-sourced rubric statements
  (`peer_review`, `humor`, `press_releases`, etc.). math-aops does not yet —
  scrape candidates would be AoPS sticky posts on "what makes a good
  solution", competition-grading rubrics (USAMO / IMO), and the dimension list
  from Inglis & Aberdein.
- **Cross-task tacit-knowledge measure.** Math may be a good "high-V,
  low-Taste" anchor in the verifiability / articulability / taste
  decomposition (`project_verifiability_explainability_gaps.md`): elegance and
  precision are both highly articulable, so the (1 − C) residual should be
  smaller than e.g. creative writing.
- **De-duplicate against contamination benchmarks.** The `_ref_aops_instruct/`
  paper builds an explicitly contamination-resistant benchmark
  (`LiveAoPSBench-2024`). Any quality model we train should be evaluated for
  overlap with that split if we want comparability.
