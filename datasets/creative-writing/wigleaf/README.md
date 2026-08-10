# Wigleaf Top 50 — cw-B cell (expert flash-fiction curation)

Within-selection expert quality labels for very short fiction (flash, <1000 words)
from the [Wigleaf Top 50](https://wigleaf.com), an annual editor-curated selection
of the best (very) short stories published online, 2008-2025. This is the
creative-writing **B-cell** of the taste taxonomy: an *expert, revealed* quality
distinction made by a single curator (series editor Scott Garson).

## Label

**Binary, within-selection.** Every story in the file was already chosen by
Wigleaf onto its annual longlist; the Top 50 is the *finer cut* a guest editor
makes from that longlist. So both classes are expert-selected — the label is the
**within-selection expert distinction**, not selected-vs-rejected.

- `judgement = 1` — **Top50**: one of the ~50 stories chosen for the year's Top 50.
- `judgement = 0` — **longlist**: on the year's longlist / "...and the rest of the
  Top 200" but not promoted to the Top 50.

There is **no "unselected" tier** in this corpus (the label file is binary; the
earlier three-tier framing was aspirational and is not built here).

## Recovery (the engineering problem)

The label file (`wigleaf_labels.csv`, 3,909 rows = 905 Top50 + 3,004 longlist) had
story URLs **only for Top50** rows. Longlist entries are plain text on the Wigleaf
list pages (`Lastname, Firstname, "Title," Magazine (date)`) with **no hyperlinks**.

**Top50 recovery** (`scripts/fetch_top50_par.py`): fetch each of the 905 stored
`story_url`s live; on a dead/parked/homepage-redirect, fall back to the Wayback
Machine (availability API then CDX index, timestamped to the list year). The
2008-2013 micro-zines (elimae, Pindeldyboz, juked `.asp`, ghoti, snowvigate, …)
recover almost entirely via Wayback.

**Longlist recovery** (`scripts/recover_longlist*.py`): no URL, so per
`(title, author, magazine, year)` we (a) resolve the magazine's domain — first
from the Top50 known-URL map (covers ~74% of longlist rows directly), then a
curated alias dictionary (→ ~84% with a resolvable domain); (b) pull the magazine
domain's **Wayback CDX index** and **slug-match** the story title against archived
URL paths (lit-mag CMSes put the title slug in the path); (c) fetch the matched
snapshot, run the *same* text pipeline, and accept only if the recovered text
contains the title tokens or the author surname (guards against wrong-page
matches). This is inherently **partial**; the recovery rate is reported below and
in the final report. Longlist rows whose magazine has no resolvable domain (~16%,
a long tail of 260+ tiny/defunct zines) and rows whose story was never archived
are not recovered.

See the FINAL REPORT (run output) for exact per-class recovery counts/rates.

## De-confounding (audit-critical)

Top50 (live + Wayback) and longlist (Wayback CDX) come from **different fetch
paths and different magazines**, so presentation could leak the label. Mitigations,
applied **identically to both classes** (`scripts/wig_textproc.py`):

- **Identical text pipeline** for every row regardless of class/source:
  largest-paragraph-block extraction → strip leading CMS/masthead lines → strip
  trailing CMS/promo/credit junk (copyright stamps, "More fiction at…", image
  credits, cookie banners, nav) → **strip the author-bio tail** (third-person
  bio cues; audit found bio tails in ~37% of pages) → NFKC, curly→straight quotes,
  dashes→hyphen, ellipsis, collapse whitespace, strip wrapper quotes.
- **Junk-page rejection**: pages where extraction grabbed a cookie banner / paywall
  / sidebar schedule instead of the story are dropped (same rule both classes).
- **Magazine-casing canonicalization**: SmokeLong/Smokelong, elimae/Elimae, etc.
  mapped to one spelling (the raw parse had 652 spellings → 634), so casing can't
  leak the parse pipeline.
- **`fetch_source` recorded per row** (`live` / `wayback`) and tested: does it
  predict y? (AUC reported in the final report; ~0.5 = no leak.)

## De-confliction of negatives (partial — documented gap)

A longlist story that *also* won Best Small Fictions / Best Microfiction / Best of
the Net is not a clean negative. **There is no winner file**, so full de-confliction
is impossible here. What we do:

- **Cross-tier dedup**: any longlist `(title, author-surname)` that also appears in
  the recovered Top50 set is dropped from the negatives (fixes the audit's
  cross-tier dupes; count in the final report).

**Residual gap**: longlist stories that won an *external* prize remain in y=0
uncorrected. This is a known limitation — the negatives are "not promoted by
Wigleaf's guest editor," not "won nothing anywhere."

## Label parse fixes

- **First-author header bug** (`scripts/fix_labels.py`): for 2008-2017 the longlist
  parser merged the section-header line ("Others from the List of 200" / "Longlist:
  The Wigleaf Top 50 (YYYY)") into the first entry's author field. 10 rows fixed
  (e.g. "Daniel Others from the List of 200 Alarcón" → "Daniel Alarcón"). Original
  `wigleaf_labels.csv` is never overwritten; the fix writes
  `wigleaf_labels_fixed.csv`.

## Files

| path | role |
|---|---|
| `wigleaf_labels.csv` | original labels (3,909 rows; **never modified**) |
| `wigleaf_labels_fixed.csv` | + first-author fix + magazine canonicalization |
| `scripts/wig_textproc.py` | shared extraction + presentation normalization (both classes) |
| `scripts/fetch_engine.py` | concurrent, per-domain-throttled fetch engine |
| `scripts/fetch_top50_par.py` | parallel Top50 fetch (live → Wayback) |
| `scripts/retry_top50_dead.py` | 2nd pass: re-fetch false-dead Top50 (archive.org overload window) |
| `scripts/recover_longlist*.py` | longlist recovery via CDX slug-match |
| `scripts/fix_labels.py` | label fixes |
| `scripts/build_dataset.py` | build splits + run all audits |
| `scripts/v_probe.py` | deterministic V-feature probe (reuses creative_writing codegen) |
| `top50_texts.jsonl`, `longlist_texts.jsonl` | recovered texts per class |
| `built/{train,eval,test}.csv.gz` | final dataset |

## Recovery rates (final)

| class | attempted | recovered (text ≥300) | rate | method |
|---|---|---|---|---|
| Top50 (y=1) | 905 | ~404 | ~45% | Wayback-only snapshot of each stored `story_url` |
| longlist (y=0) | 3,004 | ~1,168 | ~39% | Wayback CDX slug-match per magazine domain |

Top50 recovery is *lower than the live-fetch rate* (~66%) on purpose: routing every
Top50 story through Wayback to match the longlist's fetch path (de-confound) drops
the ~45% of Top50 stories that have a live page but **no usable archive snapshot**.
Longlist no-domain rows (~16%) and un-archived stories are the main misses.

## Bounds & splits (final)

Stable hash split by `md5(title|author|year) % 10` → 0-7 train / 8 eval / 9 test
(deterministic, grows safely). Columns: `text, raw_text, judgement, year,
magazine, tier, fetch_source`.

- **Size**: ~1,568 rows (~404 Top50 / ~1,164 longlist); train ~1,246 / eval ~170 /
  test ~152; base rate ~26% positive (use class weights).
- **fetch_source → y leak**: AUC **0.500** — both classes 100% `wayback`. The
  live-vs-wayback leak the first build showed at AUC 0.90 is removed.
- **TF-IDF/LR lexical floor**: AUC **≈0.54** (near chance); top features are content
  words, not magazine/CMS boilerplate — the bio/nav/credit stripping holds.
- **Length**: AUC ≈0.57 (mild; Top50 slightly longer on average).
- **V-feature probe**: codegen-V AUC **0.570 on the clean build (V > 0, +0.07)**.
  An earlier build still carrying CMS/bio leakage gave an inflated 0.72; the 0.57
  is the honest post-de-leak number.

## Verifiable-layer signal (V>0)

The deterministic creative_writing codegen V-features
(`runs/validity_full/v2/creative_writing/codegen_claude/`, 1,104 `score(text)`
programs, balanced LR) reach **codegen-V AUC = 0.570** on the clean cw-B build
(+0.07 over chance) — so a small but real part of the Top50-vs-longlist
distinction is captured by the verifiable layer. **V > 0 confirmed.** (A pre-de-leak
build inflated this to 0.72; the 0.57 is the honest number after bio/CMS stripping.)

## Known biases

- **Survivorship / online-only**: only stories still reachable (live or archived)
  enter the corpus. Defunct zines with no Wayback snapshot drop out — and these
  skew toward older years and smaller venues, so recovery is **not uniform across
  years/magazines**. Both tiers are affected, but not necessarily equally.
- **Recovery asymmetry**: Top50 had stored URLs; longlist was recovered by
  title-slug matching, which favors CMSes that slugify titles. If this correlates
  with magazine prestige it could correlate with the label — `fetch_source` and
  magazine top-features are audited to bound this.
- **Single-curator label**: this is one editor's revealed taste, not a consensus.
