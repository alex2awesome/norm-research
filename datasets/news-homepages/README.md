# news-homepages

Editorial homepage-prominence prediction from Internet Archive news homepage snapshots.

## Task

Given an article (headline + summary) plus surrounding context from the same homepage, predict whether the article was placed in the **top half** of the visible above-the-fold zone vs. the **bottom half**.

**CRITICAL — the label is homepage SPATIAL LAYOUT, not engagement.**

- Label = where the outlet's editors placed the article on their homepage (spatial position in the rendered page).
- Label is **NOT** clicks, CTR, dwell-time, share-counts, or any reader-behavior signal.
- This reflects an **editorial-prominence decision** made by the outlet itself (which story leads, which story gets banished below the fold).
- Earlier framing described the label as "engagement-driven" — that framing is **wrong**; see `~/.claude/.../memory/project_news_homepages_label_correction.md` (corrected 2026-06-01).

Concretely: label `1` = top half of the top-30%-of-page zone, label `0` = bottom half of that zone.

## Sources

- **Internet Archive `news-homepages` collection** (Ben Welsh / news-homepages project) — periodic snapshots of major outlet homepages, with two structured artifacts per snapshot:
  - `*.hyperlinks.json` — every link on the page with bounding-box coordinates (`top`, `bottom`, `left`, `right`), URL, and anchor text.
  - `*.accessibility.json` — full accessibility tree, which often contains richer headline+summary text than the raw anchor text.
- **Outlets covered (8)**: `nytimes`, `wsj`, `latimes`, `bbc`, `washingtonpost`, `cnn`, `guardian`, `reuters`. List defined in `build_homepage_dataset.py:DEFAULT_OUTLETS`.
- **storysniffer labeled data** (`storysniffer_labeled.csv`) — Ben Welsh's labeled URL/anchor-text dataset of `article` vs `furniture` (nav links, ads, login buttons, etc.). Used to train the article-vs-furniture filter.

## Collection / preprocessing scripts

All paths absolute under `/Users/spangher/Projects/stanford-research/norm-research/datasets/news-homepages/`.

| Script | Purpose |
|---|---|
| `build_homepage_dataset.py` | End-to-end pipeline: `download` (pull `hyperlinks.json` + `accessibility.json` per outlet from archive.org), `build` (classify links, filter to top-30% zone, label top-half vs bottom-half, balance per-outlet, write CSV), or `all` (both). |
| `train_article_classifier.py` | Trains the article-vs-furniture LogisticRegression on `storysniffer_labeled.csv`; emits `article_classifier.joblib`. URL-path + anchor-text features. |
| `topic_balance.py` | Reads the per-outlet-balanced output, runs TF-IDF + 50-topic LDA on headlines, hard-assigns each row to its top topic, and balances 1s/0s **within each topic**. Writes `homepage_newsworthiness_topic_balanced.csv.gz`. |
| `/Users/spangher/Projects/stanford-research/norm-research/scripts/add_snapshot_id_homepages.py` | Adds `snapshot_id` (md5 of the sorted set of headline+context strings) so that all rows from the same homepage snapshot share a group key. Also re-balances per-class within each snapshot group. Writes `homepage_newsworthiness_topic_balanced_groupsplit.csv.gz`. |

### Pipeline summary

1. Download `*.hyperlinks.json` + `*.accessibility.json` for each outlet from the `news-homepages` IA collection (`build_homepage_dataset.py:download_outlet`).
2. For each snapshot: filter to visible links (positive bounding box), classify article vs. furniture via the joblib classifier, dedup by URL path keeping highest occurrence.
3. Enrich each article's anchor text with the longest matching string from the accessibility tree (gives headline + summary instead of bare headline).
4. Restrict to articles whose `top` coordinate is in the **top 30%** of `page_height` (the "above-the-fold zone"). Drop snapshots with fewer than 4 articles in that zone.
5. Split that zone at its midpoint: label `1` if `top < midpoint`, else `0`.
6. Build context: every *other* cleaned headline from the same snapshot (excluding the target headline), **shuffled** to remove positional signal.
7. Format: `HEADLINE: {headline+summary}\n\nCONTEXT: {h1; h2; …}`.
8. Balance 50/50 within each outlet (drop-to-min).
9. (`topic_balance.py`) Balance 50/50 within each LDA topic.
10. (`add_snapshot_id_homepages.py`) Add `snapshot_id` group key and rebalance per class within each snapshot group.

## File layout

```
datasets/news-homepages/
├── README.md                                                    (this file)
├── build_homepage_dataset.py                                    download + build script
├── train_article_classifier.py                                  classifier training
├── topic_balance.py                                             LDA topic balancing
├── article_classifier.joblib                                    trained article/furniture classifier
├── storysniffer_labeled.csv                                     Ben Welsh's URL-vs-furniture labels
├── homepage_newsworthiness_topic_balanced_groupsplit.csv.gz     CANONICAL clean dataset (~173 MB)
├── test_output.csv.gz                                           tiny smoke-test output
├── raw_data/
│   └── nytimes/                                                 per-outlet IA dumps (hyperlinks/accessibility)
└── online-rubrics/                                              norm-source corpus + parsed rubrics
    ├── raw/                                                     downloaded HTML/PDF norm sources (editorial guidelines, news-values literature, etc.)
    ├── claude-parsed/                                           Claude-distilled rubric markdown
    ├── gpt-parsed/gpt-5-mini/                                   GPT-distilled rubric markdown
    ├── urls-visited.csv                                         crawler bookkeeping
    └── waveh{3,4,5,6}_{log.csv,seen.txt}                        crawl-wave manifests
```

Raw IA dumps for the other 7 outlets live at `/lfs/skampere3/0/alexspan/norm-research/datasets/news-homepages/raw_data/{outlet}/` on sk3 (not mirrored locally).

## Canonical dataset file

**Use:** `homepage_newsworthiness_topic_balanced_groupsplit.csv.gz` (~173 MB, ~180 MB on disk).

| Field | Description |
|---|---|
| `text` | `HEADLINE: {h}\n\nCONTEXT: {shuffled other headlines from same snapshot, joined by "; "}` |
| `judgement` | `1` = top half of the top-30% zone, `0` = bottom half |
| `snapshot_id` | First 16 hex chars of md5 over the sorted set of headline-prefix(50 chars) strings in this row |

- **~21,951 unique snapshots**, 50/50 label distribution.
- **Group-split status: DONE.** Always use `snapshot_id` as the group key for train/val/test splits — never random row splits. Rows from the same snapshot otherwise leak (they share all the same context headlines).
- This is the file listed as canonical in `reference_clean_datasets_per_task.md`.
- sk3 mirror: `/lfs/skampere3/0/alexspan/norm-research/datasets/news-homepages/homepage_newsworthiness_topic_balanced_groupsplit.csv.gz`.

## Modeling state

Three confounds were identified and removed during a leakage investigation 2026-04-20 → 2026-04-25 (see `project_homepage_newsworthiness.md`). Clean deconfounded AUC stabilizes around **~0.75**.

| Version | AUC | Remaining confound |
|---|---|---|
| Original build | 0.989 | Outlet + position + topic |
| No SOURCE/DATE in text + outlet-balanced | 0.963 | Position + topic |
| + shuffled context, target headline removed | 0.984 | Topic |
| + topic-balanced (LDA 50) — **canonical** | **~0.753** | (clean) |

Sweep runs: `runs/homepage_newsworthiness_sweep_llama8b/`, `runs/homepage_newsworthiness_groupsplit_sweep_llama8b/`.

## Key decisions

### Label is spatial layout, not clicks (2026-06-01 correction)

The label is which articles end up in prominent vs. less-prominent slots on the rendered homepage — an editorial decision by the outlet, not a reader-behavior signal. This reframes which norms are relevant:

- **Relevant norms**: news values (Galtung & Ruge; importance/timeliness/magnitude), editorial priorities of the outlet, audience-of-record expectations, time-sensitivity/news-cycle position.
- **Less relevant than initially assumed**: engagement-mechanics norms (virality factors, click-driving headline craft).
- **Mismatched** (same way as code_review): the standard "editorial-process" norm library (source authority, fact-checking, research rigor) describes how a story is *produced*, not how prominence is *assigned*.

### Three deconfounding choices

1. **Drop SOURCE/DATE from text.** Original text included outlet and date markers; the model could learn outlet identity (e.g., BBC 34% top vs. WashPost 73% top under raw build) as a base-rate shortcut. Fixed by removing these markers and per-outlet balancing.
2. **Remove the target headline from context and shuffle.** In an ordered context string, the target headline's character position directly leaked the label (top articles at ~0.25 of the string, bottom at ~0.73). Fixed by excluding the target from context and shuffling the rest.
3. **Topic-balance via 50-topic LDA.** Certain topics (breaking news, disasters) sit at the top systematically; lifestyle/evergreen at the bottom. The model exploited topic identity. Fixed by hard-assigning each row to its top LDA topic and balancing 1s/0s within each topic — this is what dropped AUC from 0.96 → 0.75.

### Group split

The per-snapshot group key (`snapshot_id` = md5 over the sorted headline set) was added in a separate pass (`scripts/add_snapshot_id_homepages.py`) because all rows from the same snapshot share their context headlines. Within-snapshot leakage was inflating eval AUC; the `_groupsplit` file fixes that.

## Open questions / next steps

- **Can the artifact text even support the label?** The text artifact is headline + summary + a shuffled bag of sibling headlines. Spatial-layout decisions may depend on visual/typographic cues, sectioning, and structural prominence that aren't recoverable from text alone. There may be a structural ceiling here, similar to the situation flagged for code_review.
- **Norms library mismatch.** The current editorial-process norm library (research rigor, source authority, fact-checking) describes story production, not prominence assignment. A prominence-relevant library (news values, editorial-priority norms, audience-of-record concerns) needs to be assembled or surfaced from the corpus already crawled in `online-rubrics/raw/`.
- **Press releases parallel pending group-split.** Per `reference_clean_datasets_per_task.md`, press_releases is the next group-split rebuild owed.
- **Online-rubrics ingestion for V/A/T.** `online-rubrics/{raw,claude-parsed,gpt-parsed}` holds editorial guidelines and news-values literature; not yet wired into the verifiability + articulability + taste decomposition pipeline for this task.
