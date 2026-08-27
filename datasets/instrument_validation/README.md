# Instrument validation gold data

This directory holds normalized, freely-downloadable gold-label datasets used to
validate LLM-judge instruments against pre-existing human annotations. Each
instrument gets its own `<instrument>_gold.tsv` file plus a section below
documenting sources, licenses, and label semantics. (This directory is shared
across sibling instrument-validation tasks — see other `*_gold.tsv` files /
raw source folders for unrelated instruments; each section below covers only
its own instrument.)

---

## Metaphoricity (binary: word/phrase used metaphorically vs literally)

**Gold file:** `metaphor_gold.tsv`
**Build script:** `build_metaphor_gold.py` (reads from `raw/`, writes `metaphor_gold.tsv`)
**Columns:** `item` (target word/phrase) · `label` (1 = metaphorical/idiomatic, 0 = literal) · `context` (full sentence; all sources here are usage-in-context, so context is never empty) · `source` (short tag)

### Row counts (post-dedup, exact `item+context+source` duplicates dropped)

| source | total | literal (0) | metaphorical (1) |
|---|---|---|---|
| MOH | 1,638 | 1,228 | 410 |
| MOH-X | 641 | 328 | 313 |
| TroFi | 3,642 | 2,040 | 1,602 |
| TroFi-X | 1,340 | 780 | 560 |
| VUA (verb-classification subset) | 22,301 | 15,883 | 6,418 |
| Yulia (Tsvetkov et al.) | 222 | 111 | 111 |
| MAGPIE (phrase-level idiomaticity) | 48,065 | 12,027 | 36,038 |
| **Total** | **77,849** | **32,397** | **45,452** |

Note: pre-dedup source row counts (MOH 1,639 / MOH-X 647 / TroFi 3,737 /
TroFi-X 1,444 / VUA 23,113 / Yulia 222 / MAGPIE 48,395) had 1,348 exact
duplicate rows dropped in aggregate (mostly MAGPIE, which has some identical
idiom+context+label rows in the source jsonl).

### Sources, URLs, licenses, citations

**1. MOH (Mohammad, Shutova & Turney 2016 "Metaphor as a Medium for Emotion" release)**
- Downloaded from: `http://saifmohammad.com/WebDocs/Metaphor-Emotion-Data-Files.zip`
  (linked from `http://saifmohammad.com/WebPages/metaphor.html`)
- File used: `Data-metaphoric-or-literal.txt` (term, sense, sentence, class ∈
  {metaphorical, literal}, confidence). We use `class` as the binary label
  and strip the `<b>...</b>` markup around the target verb in the sentence.
- License: page states data is "available for direct download and can be
  used freely for research purposes"; commercial use requires crediting the
  authors and NRC.
- Citation: Saif M. Mohammad, Ekaterina Shutova, and Peter Turney. "Metaphor
  as a Medium for Emotion: An Empirical Study." *SEM 2016.
- Semantics: term-sense-level (WordNet verb sense) crowd metaphoricity
  judgment in a single example sentence per sense.

**2. MOH-X, TroFi, TroFi-X, VUA (verb-classification subset), Yulia — all via Gao et al. (2018) "Neural Metaphor Detection in Context" repo redistribution**
- Repo: `https://github.com/gao-g/metaphor-in-context`
- Data download: Google Drive link in repo README —
  `https://drive.google.com/uc?export=download&id=1-v_sUlupDrKq8ERlnh5RwdI7Pyk-6cxn`
  (direct-downloadable without login via the `uc?export=download` form).
- Citation: Ge Gao, Eunsol Choi, Yejin Choi, Luke Zettlemoyer. "Neural
  Metaphor Detection in Context." EMNLP 2018.
- Repo does not state an explicit license; treated as research-use
  redistribution of the underlying published corpora (see per-dataset origin
  below). No commercial-use claim is made here.

  - **MOH-X**: cleaned subset of MOH (Mohammad et al. 2016) as used by
    Shutova/Ekaterina et al. for classification, `MOH-X_formatted_svo_cleaned.csv`.
    Columns: arg1, arg2, verb, sentence, verb_idx, label. Original MOH-X
    citation: Ekaterina Shutova, Douwe Kiela, Jean Maillard. "Black Holes
    and White Rabbits: Metaphor Identification with Visual Features." NAACL 2016
    (subset derived from Mohammad et al. 2016 MOH).
  - **TroFi**: `TroFi_formatted_all3737.csv`, human-labeled subset of the
    TroFi (Trope Finder) corpus. Columns: verb, sentence, verb_idx, label.
    Original citation: Julia Birke and Anoop Sarkar. "A Clustering Approach
    for Nearly Unsupervised Recognition of Nonliteral Language." EACL 2006.
    (Original TroFi release page `http://www.cs.sfu.ca/~anoop/students/jbirke/`
    returned HTTP 403 when fetched directly — not independently re-downloaded;
    we rely on the Gao et al. redistribution.)
  - **TroFi-X**: `TroFi-X_formatted_svo.csv`, TroFi subset with parsable
    subject-verb-object triples. Columns: arg1, arg2, verb, sentence,
    verb_stem, label.
  - **VUA (verb-classification subset)**: `VUA_formatted.csv` (read as
    latin-1; contains non-UTF8 bytes). Derived from the VU Amsterdam
    Metaphor Corpus (Steen et al. 2010, MIPVU-annotated) via the NAACL 2018
    VU Amsterdam Metaphor Corpus shared task's verb-classification split.
    **This is NOT the full VUAMC** — see "not freely downloadable" below.
    Columns: text_idx, sentence_idx, verb, sentence, verb_idx, label.
  - **Yulia**: `Yulia_formatted_svo.csv`, test set from Yulia Tsvetkov et
    al.'s metaphor work (e.g. "Metaphor Detection with Cross-Lingual Model
    Transfer," ACL 2014), used as an out-of-domain cross-validation set by
    Gao et al. Columns: arg1, arg2, verb, sentence, verb_stem, label.

  A `VUAsequence/` folder (word/POS-sequence-labeled VUAMC sentences, not
  restricted to single verbs) is also in the Gao et al. release
  (`raw/gao_data/data/VUAsequence/`) but was **not** converted into the gold
  TSV — its label is a per-token sequence rather than a single item+label
  pair, and converting it would require re-deriving target-word alignment;
  left for a follow-up if word-in-context sequence-labeling gold is needed.

**3. MAGPIE (Multi-Genre AnnotatIon of Potentially Idiomatic Expressions)**
- Repo: `https://github.com/hslh/magpie-corpus`
- Downloaded zip: `raw/magpie-corpus.zip` → `raw/magpie-corpus/magpie-corpus-master/`
- File used: `MAGPIE_filtered_split_random.jsonl` (the filtered subset with
  confidence ≥ 0.75 and binary label ∈ {i=idiomatic, l=literal}; the
  `MAGPIE_filtered_split_typebased.jsonl` file contains the **identical**
  48,395 instances just split differently for train/dev/test — verified by
  id-set comparison — so only one was used to avoid duplication). Rows with
  label `f` (figurative-but-not-idiom), `o` (other), or `?` (undecided) were
  dropped since they are not clean binary literal/nonliteral labels.
- License: CC-BY 4.0 (see `raw/magpie-corpus/magpie-corpus-master/LICENSE`).
- Citation: Hessel Haagsma, Johan Bos, Malvina Nissim. "MAGPIE: A Large
  Corpus of Potentially Idiomatic Expressions." LREC 2020.
- **Important scope note**: MAGPIE labels are *idiomaticity* of a
  **multi-word idiom span** in its BNC sentence context, not single-word
  verb *metaphoricity* like MOH/TroFi/VUA. It is included because it is a
  genuine phrase-level figurative-vs-literal resource (explicitly in scope
  per the task brief), but downstream users validating a strict
  single-word-verb metaphoricity judge should probably filter `source !=
  "MAGPIE"`. The `context` field here is the middle sentence of MAGPIE's
  5-sentence context window (index `len(context)//2`), which is an
  approximation of "the sentence containing the target span" and was not
  individually re-verified per row against the character `offsets` field.

### Considered but NOT freely/directly downloadable

- **Full VU Amsterdam Metaphor Corpus (VUAMC)** — the complete
  MIPVU-annotated corpus (Steen et al. 2010) is distributed via the Vrije
  Universiteit Amsterdam / CLARIN and generally requires registration or a
  CLARIN account to access the full XML release with all word classes
  (not just verbs) and metadata. We only have the verb-classification
  derivative redistributed by Gao et al. (2018), described above.
  (`vismet.org`, one candidate host for VUAMC documentation, was
  unreachable via automated fetch — TLS certificate mismatch on the host.)
- **Original TroFi release page**
  (`http://www.cs.sfu.ca/~anoop/students/jbirke/`) returned HTTP 403 to an
  automated fetch; not pursued further since the Gao et al. redistribution
  of TroFi/TroFi-X was already obtained.
- **LCC (Language Computer Corporation) metaphor dataset** — could not
  locate a working direct-download link in this pass (web search budget for
  this session was exhausted mid-task); not included. Worth a follow-up
  search specifically for "LCC metaphor corpus release" / DARPA Metaphor
  Program artifacts.

### Reproducing

```
cd datasets/instrument_validation
python3 build_metaphor_gold.py
```
reads from `raw/moh_emotion/`, `raw/gao_data/`, `raw/magpie-corpus/` (already
downloaded into `raw/`) and (re)writes `metaphor_gold.tsv`.

---

## Formality (lexical: word/phrase 0-100 casual→formal)

**Gold file:** `formality_gold.tsv`
**Build script:** `build_formality_gold.py` (reads from `raw/`, writes `formality_gold.tsv`)
**Columns:** `term` (word or phrase) · `formality_score` (0-100 scale, 100 =
most formal, 0 = most casual) · `source` (short tag)

### Row counts

| source | rows |
|---|---|
| pavlick_nenkova_2015_phrase | 7,794 |
| brooke_seeds_2010 | 242 |
| **Total** | **8,036** |

(No duplicates were dropped in either source: 7,794 raw phrase-score lines
and 242 raw seed-list entries both matched their post-dedup counts exactly.)

### Sources, URLs, licenses, citations

**1. Pavlick & Nenkova (2015 NAACL) — "Inducing Lexical Style Properties for Paraphrase and Genre Differentiation" (PRIMARY source, top candidate named in task brief)**
- Live download link on the author's current data page
  (`https://cs.brown.edu/people/epavlick/data.html`, "Style Lexicons" entry)
  points to `http://www.seas.upenn.edu/~nlp/resources/style-scores.tar.gz`,
  which now 404s (the UPenn NLP group's `~nlp` pages have moved off
  `seas.upenn.edu`/`engineering.upenn.edu`). Retrieved instead via the
  Wayback Machine snapshot:
  `http://web.archive.org/web/20250824002446/https://www.seas.upenn.edu/~nlp/resources/style-scores.tar.gz`
  (200, valid gzip, verified by successful `tar xzf` + README present).
- File used: `naacl-2015-style-scores/formality/human/phrase-scores`
  (`raw/style-scores/naacl-2015-style-scores/formality/human/phrase-scores`).
  Per the release's own README: 3 tab-separated columns — (1) mean of 7
  MTurk annotators' scores, scale 1-100 where 100 = most formal, 0 = most
  casual; (2) the word/phrase; (3) std dev of the 7 human scores (dropped
  here — not carried into `formality_gold.tsv`, only `term` and
  `formality_score` are kept from this file). Observed score range in the
  file: 0.0 to 98.14.
  - Companion files also downloaded but NOT used in the gold TSV:
    `formality/automatic/*` (log-ratio corpus-derived scores, not human
    labels), `formality/human/pairs` (pairwise majority-vote comparisons,
    not single-item scores), `formality/human/sentence-scores` (950
    MASC-corpus sentences, sentence-grain not word/phrase-grain — out of
    scope per task brief), and the parallel `complexity/` tree (different
    construct, not formality).
- License: no explicit license file in the archive; page/README state
  research-use redistribution with a citation request (see below). Treated
  as free-for-research per standard academic-resource-page norms.
- Citation: Ellie Pavlick and Ani Nenkova. "Inducing Lexical Style
  Properties for Paraphrase and Genre Differentiation." NAACL 2015.
- Also downloaded (same tarball's sibling companion release, same author,
  linked as a separate "Download" on the same data page) but **not used**
  in the gold TSV because it is sentence-level, not word/phrase-level:
  Pavlick & Tetreault (2016 TACL) "An Empirical Analysis of Formality in
  Online Communication" / Lahiri (2015) SQUINKY sentence-formality corpus,
  from `http://www.seas.upenn.edu/~nlp/resources/formality-corpus.tgz`
  (also 404 live; retrieved via Wayback:
  `http://web.archive.org/web/20221010114006/https://www.seas.upenn.edu/~nlp/resources/formality-corpus.tgz`).
  11,274 sentences across 4 genres (answers/blog/email/news), each with a
  mean formality rating on a **-3 to 3** scale (5 MTurk raters) plus raw
  per-annotator scores. CC-BY 3.0 license (`LICENSE` file included in the
  archive, extracted to `raw/formality-corpus/data-for-release/LICENSE`).
  Left in `raw/` for potential future sentence-grain validation work but
  intentionally excluded from `formality_gold.tsv`.

**2. Brooke, Wang & Hirst (2010 COLING) — "Automatic Acquisition of Lexical Formality" (secondary/supplementary, CATEGORICAL not continuous)**
- Downloaded from Julian Brooke's page:
  `http://www.cs.toronto.edu/~jbrooke/Formality_Word_Lists.zip` (200,
  valid zip, verified by successful unzip).
- Files used: `formal_seeds_100.txt` (104 words) and
  `informal_seeds_100.txt` (137 words) — the manually curated seed lists
  the paper's algorithm was bootstrapped from. **These are class labels
  (formal vs. informal), not an author-elicited numeric formality
  rating.** We map them onto the same 0-100 scale as Pavlick & Nenkova
  using the scale's own endpoints (formal seed → 100.0, informal seed →
  0.0) purely so the two sources sit on a comparable axis in one file;
  this is a construction on our part, not a score Brooke et al. assigned
  — flagged via the `brooke_seeds_2010` source tag so it can be filtered
  out or reweighted separately from the true continuous PN15 ratings.
  242 rows total (105 formal seeds + 137 informal seeds, no overlap, no
  duplicates dropped). Note: the raw `.txt` files use CRLF line endings and
  no trailing newline on the last entry, so a plain `wc -l` undercounts by
  one per file (104/137) — the script's line-by-line read handles this
  correctly and the true counts are 105/137.
- Files downloaded but NOT used: `CTRWpairsfull.txt` (398 informal/formal
  synonym *pairs*, e.g. `sot/alcoholic`, `digest/imbibe` — direction is
  implied by column order but there is no numeric score, and pairs are a
  different grain than single-item scores); `formal_seeds_100_CN.txt` /
  `informal_seeds_100_CN.txt` (looked like alternate/filtered seed lists,
  smaller, purpose not documented in the zip — skipped to avoid guessing
  at an undocumented variant).
- License/terms: no license file present in the zip; page has no explicit
  license statement. Treated as free-for-research redistribution per
  standard academic personal-page norms (page literally says "List of
  formal and informal words from my work on formality").
- Citation: Julian Brooke, Tong Wang, Graeme Hirst. "Automatic Acquisition
  of Lexical Formality." COLING 2010.
- **The full continuously-scored Brooke et al. (2010) lexicon (their
  algorithm's output over thousands of words, as reported/used in the
  paper's evaluation) does not appear to be hosted for direct download
  anywhere found in this pass** — only the seed lists and CTRW pairs used
  as algorithm *inputs* are released. Not pursued further; noted as a gap
  below.

### Considered but NOT freely/directly downloadable

- **Brooke et al. (2010) full scored lexicon** — see note directly above;
  only categorical seed inputs are released, not the paper's actual
  continuous-formality output list.
- **GYAFC (Grammarly's Yahoo Answers Formality Corpus)** — sentence-level
  (not word/phrase-grain, out of scope per task brief anyway) and requires
  a license agreement / request form (built on top of the Yahoo L6
  corpus, which itself requires a Yahoo Webscope license) — not freely,
  directly downloadable. Not pursued.

### Reproducing

```
cd datasets/instrument_validation
python3 build_formality_gold.py
```
reads from `raw/style-scores/` and `raw/brooke_formality/` (already
downloaded into `raw/`) and (re)writes `formality_gold.tsv`.

---

## Semantic transparency / compositionality (is a phrase's meaning composable from its parts vs idiomatic/opaque)

**Gold file:** `transparency_gold.tsv`
**Build script:** `build_transparency_gold.py` (reads from `raw/`, writes `transparency_gold.tsv`)
**Columns:** `item` (word/compound/phrase) · `score_or_label` (original-scale numeric rating, or 0/1 for the one binary source) · `scale` (verbatim description of that source's rating scale, direction, and aggregation) · `source` (short tag)

Each source keeps its **own native scale** (0-100, 0-5, 1-6, 0-10, or
binary) rather than being forced onto one common scale — direction is
always "higher = more transparent/compositional, lower = more opaque/
idiomatic" except the MAGPIE binary label, which is "1 = idiomatic
(opaque), 0 = literal (transparent)" (documented per-row in the `scale`
column). Downstream users who want one common scale should min-max or
rank-normalize **within source**, not pool raw numbers across sources.

### Row counts

| source | n | scale |
|---|---|---|
| LADEC (Gagné, Spalding & Schmidtke 2019) | 8,299 | 0-100 continuous, whole-compound compositionality mean |
| MAGPIE (Haagsma, Bos & Nissim 2020) | 1,738 | binary idiom-type majority label + fraction/confidence |
| Venkatapathy & Joshi (2005), released as "SVAJ2005" | 765 | 1-6 mean of 2 annotators |
| McCarthy, Keller & Carroll (2003) phrasal verbs | 116 | 0-10 mean of 3 native-speaker judges |
| Reddy, McCarthy & Manandhar (2011) | 90 | 0-5 mean (AMT) |
| **Total** | **11,008** | |

### Sources, URLs, licenses, citations

**1. LADEC — Large Database of English Compounds (Gagné, Spalding & Schmidtke 2019)**
- Downloaded from: `https://ualberta.scholaris.ca/items/3087991f-d98b-4d97-998a-5b4f970ababf`
  → direct file: `https://ualberta.scholaris.ca/bitstreams/61f74459-f332-4bbe-bf9f-92c97480f5c2/download`
  (this is the University of Alberta institutional repository; an earlier
  `bitstreams.library.ualberta.ca` URL surfaced by search does **not**
  resolve — DNS NXDOMAIN — the working host is `ualberta.scholaris.ca`.)
- File used: `LADECv1-2019.csv` (saved to `raw/ladec/LADECv1-2019.csv`),
  8,956 noun-noun compounds (3-10 letter constituent bases, all
  WordNet-classified nouns). We use `stim` (the whole compound, e.g.
  "turnabout") and `ratingcmp` (mean whole-compound compositionality
  rating; 8,299 rows have a non-missing value — the remainder are `NA`
  and dropped). LADEC also has separate `ratingC1`/`ratingC2` constituent
  transparency ratings, not used here (whole-item transparency only, to
  match the task's phrase-level framing).
- License: **CC BY-NC 4.0** (non-commercial). Valence sub-variables (not
  used here) require an additional citation to Kuperman (2020).
- Citation: Gagné, C.L., Spalding, T.L., & Schmidtke, D. (2019). "LADEC:
  The Large Database of English Compounds." *Behavior Research Methods*.

**2. MAGPIE — Multi-Genre AnnotatIon of Potentially Idiomatic Expressions (Haagsma, Bos & Nissim 2020)**
- Repo: `https://github.com/hslh/magpie-corpus`
- Downloaded zip → `raw/magpie-corpus/magpie-corpus-master/`. File used:
  `MAGPIE_filtered_split_random.jsonl` (same underlying instance set as
  the `_typebased` split file, just partitioned differently — see the
  Metaphoricity section above for the id-set-identity check already done
  on this file for the sibling task).
- Processing: MAGPIE is token-instance-level (idiom span + BNC sentence
  context + label ∈ {i=idiomatic, l=literal, f=figurative-not-idiom,
  o=other, ?=undecided}). For this **type-level compositionality**
  instrument we aggregate all `i`/`l`-labeled instances per idiom string
  (`f`/`o`/`?` dropped) into one row: majority label (ties → idiomatic),
  fraction-idiomatic, and mean annotator confidence, all reported in the
  `scale` column. 1,738 distinct idiom types.
- License: CC-BY 4.0.
- Citation: Haagsma, H., Bos, J., & Nissim, M. (2020). "MAGPIE: A Large
  Corpus of Potentially Idiomatic Expressions." LREC.

**3. Venkatapathy & Joshi (2005) verb-noun/verb-adjective collocation compositionality ("SVAJ2005" release)**
- Released (with permission) via Diana McCarthy's downloads page:
  `http://dianamccarthy.co.uk/downloads.html` → data file
  `http://www.dianamccarthy.co.uk/downloads/SVAJ2005compositionality_rating.txt`,
  guidelines `http://www.dianamccarthy.co.uk/downloads/SVAJ2005README.txt`.
  Saved to `raw/reddy_compositionality/SVAJ2005compositionality_rating.txt`.
- Format: `item<TAB>annotator1_score<TAB>annotator2_score`, both 1-6.
  This is the 765-pair "verb-object" subset (McCarthy's site notes an
  additional 638-item filtered subset from an EMNLP 2007 paper that
  excludes non-common-noun objects — not separately included here; we
  use the full original 765-pair Venkatapathy & Joshi release). We take
  the mean of the two annotators as `score_or_label`.
- Annotators: one native English speaker (Roderick Saxey), one
  non-native (Pranesh Agarwal) — per the McCarthy-hosted README.
- License: not explicitly stated; "released ... with kind permission"
  from Sriram Venkatapathy for research use.
- Citation: Venkatapathy, S., & Joshi, A.K. (2005). "Measuring the
  Relative Compositionality of Verb-Noun (V-N) Collocations by
  Integrating Features." HLT-EMNLP 2005.

**4. McCarthy, Keller & Carroll (2003) — English phrasal verbs**
- Downloaded from: `http://www.dianamccarthy.co.uk/files/McCarthyGS.tar.gz`
  (linked from the same downloads page). Saved to
  `raw/mccarthy_phrasalverbs/` and extracted (`Judge1`, `Judge2`,
  `Judge3`, `NonNativeSpeaker`, `Instructions`, `readme`).
- Format per judge file: `idx : verb+particle : corpus_frequency : score
  (0-10)`. We use `Judge1`/`Judge2`/`Judge3` (3 native British-English
  computational-linguist judges) and take the mean per verb-particle
  pair over the judges that scored it (`NonNativeSpeaker` excluded per
  the authors' own README, which says it "was not used within our
  experiments"). 116 distinct phrasal verbs (one known duplicate,
  `look+up`, is merged — the README notes all native judges gave it the
  same score both times it appeared).
- License: not explicitly stated; released as "Gold Standard Data"
  alongside the paper for research use.
- Citation: McCarthy, D., Keller, B., & Carroll, J. (2003). "Detecting a
  Continuum of Compositionality in Phrasal Verbs." ACL-SIGLEX Workshop
  on Multiword Expressions.

**5. Reddy, McCarthy & Manandhar (2011) — 90 noun-noun compounds**
- Data + guidelines: `http://www.dianamccarthy.co.uk/files/ijcnlp_compositionality_data.tgz`.
  Saved to `raw/reddy_compositionality/ijcnlp_compositionality_data/`.
- File used: `MeanAndDeviations.clean.txt` — one row per compound
  (`word1-n word2-n`), with `Word1_mean/std`, `Word2_mean/std`,
  `Cpd_mean/std` (whole-compound compositionality, 0-5 AMT Likert mean),
  and `mean1*mean2`. We use `Cpd_mean` as `score_or_label` and
  reconstruct `item` by stripping the `-n` POS suffix and joining the two
  constituents (e.g. "end user"). Raw per-worker annotations (with
  worker IDs, HIT IDs, qualification-test data) are also present in
  `annotations/` but not needed for the gold means. Exactly 90 compounds,
  matching the paper's reported dataset size.
- License: not explicitly stated; released as research data with citation
  request.
- Citation: Reddy, S., McCarthy, D., & Manandhar, S. (2011). "An
  Empirical Study on Compositionality in Compound Nouns." IJCNLP 2011.

### Considered but NOT freely/directly downloadable

- **Cordeiro et al. compositionality datasets** (e.g. the "Ppmi/word2vec
  compositionality prediction" evaluation sets referenced in Cordeiro,
  Villavicencio, Idiart & Ramisch work) — not pursued this pass; the
  Reddy (2011) and Venkatapathy & Joshi (2005) sets above already cover
  the same phenomenon (noun-compound / V-N compositionality) and were
  found first with a working direct-download link.
- **CELEX derivational database** — Dutch/German/English morphological
  database including derivational-family information; **licensed**
  (LDC catalog item, requires an LDC/CELEX license), not freely
  downloadable. Noted per task brief; not obtained.
- **Full MAGPIE unfiltered/confidence-annotator-level data** — the
  filtered split used above already gives type-level majority labels;
  the raw per-annotator judgment table (if a finer-grained gold is
  needed later) is in `MAGPIE_unfiltered.jsonl` in the same repo, not
  separately processed here.

### Reproducing

```
cd datasets/instrument_validation
python3 build_transparency_gold.py
```
reads from `raw/ladec/`, `raw/magpie-corpus/`, `raw/reddy_compositionality/`,
`raw/mccarthy_phrasalverbs/` (already downloaded into `raw/`) and
(re)writes `transparency_gold.tsv`.

---

## Nominalization (does a term contain a derived abstract nominal: -tion/-ity/-ness/-ment/-ance/-ence)

**Gold file:** `nominalization_gold.tsv`
**Build script:** `build_nominalization_gold.py` (reads from `raw/`, writes `nominalization_gold.tsv`)
**Columns:** `word` · `is_nominalization` (1/0) · `source`

### Row counts

| source | n | label |
|---|---|---|
| NOMLEX-2001 (NYU Proteus Project) | 1,001 | 1 (nominalization) |
| Google-10000-English freq list, WordNet/POS-filtered | 1,001 | 0 (non-nominalization negative) |
| **Total** | **2,002** | balanced 1:1 |

### Sources, URLs, licenses, citations

**1. NOMLEX (positives)**
- Downloaded from: `http://nlp.cs.nyu.edu/nomlex/NOMLEX-2001.exp` (the
  original Lisp-style feature-structure format) and
  `http://nlp.cs.nyu.edu/nomlex/NOMLEX-2001.reg` (a "regularized" variant,
  also saved but not used in the build script). Saved to
  `raw/nomlex/NOMLEX-2001.exp` / `.reg`.
- Content: a dictionary of >1,000 English nominalizations (deverbal nouns
  like "abandonment", "acceptance", "accomplishment") developed by the
  Proteus Project at NYU, with argument-structure mappings to their
  source verbs, drawn from frequent nominalizations in the Brown and
  Wall Street Journal corpora.
- Extraction: every distinct `:ORTH "..."` string in the `.exp` file —
  1,023 raw distinct strings, 1,001 after restricting to purely
  alphabetic entries (drops a handful of multi-word/hyphenated forms).
  Used as `is_nominalization=1` positives — **note NOMLEX itself is not
  restricted to the 6 suffixes named in the task brief**
  (-tion/-ity/-ness/-ment/-ance/-ence); it is a broader nominalization
  lexicon (also includes forms like "worker", "absentee") reflecting
  linguistic nominalization more generally, not just those 6 suffixes.
  A suffix-only positive subset can be derived by filtering
  `nominalization_gold.tsv` rows where the word ends in one of the 6
  suffixes, if a narrower definition is wanted downstream.
- License: page states the 2001 version is "freely available for use by
  all"; no formal license text beyond that.
- Citation: Macleod, C., Grishman, R., Meyers, A., Barrett, L., & Reeves,
  R. (1998). "NOMLEX: A Lexicon of Nominalizations." Proceedings of
  EURALEX.
- **NOMLEX-PLUS / NomBank (7,000+ entries, expanded lexicon) — NOT freely
  downloadable.** NomBank's own site
  (`https://nlp.cs.nyu.edu/meyers/NomBank.html`) states some of its
  additional files "require licenses from the LDC" and gives no direct
  download link; the LDC catalog entry (`LDC2008T24`) requires an LDC/WSJ
  Corpus license. Only the free NOMLEX-2001 base lexicon above was used.
- **CELEX derivational-family database — licensed, not free.** Noted per
  task brief; would need an LDC/CELEX license, not obtained.

**2. Negatives (non-nominalizations)**
- Base frequency list: `raw/google-10k.txt` (Google-10000-English word
  list; already present in this shared `raw/` directory from a sibling
  instrument-validation task — a standard, freely available frequency-
  ranked English word list, most-common-first).
- Sampling rule (deterministic, frequency-rank order, first N that
  qualify): for each word in frequency-rank order, **keep** it as a
  negative candidate iff ALL of:
  1. purely alphabetic, length ≥ 3;
  2. not already one of the 1,001 NOMLEX positive strings;
  3. not an NLTK English stopword (`nltk.corpus.stopwords`);
  4. does **not** end in any of the derivational suffixes
     `-tion/-sion/-ity/-ty/-ness/-ment/-ance/-ence/-ancy/-ency` (i.e.
     negatives are guaranteed suffix-clean, per the task brief);
  5. `nltk.pos_tag([word])` (in-isolation, averaged-perceptron tagger)
     returns `NN`/`NNS` (common noun) or `JJ` (adjective) — this POS
     check is what excludes function words that happen to have a
     marginal WordNet noun sense (e.g. "us", "no", "has", "can", "do"
     were all rejected here even though WordNet lists a noun synset for
     each of them; without this check they leaked into an earlier
     iteration of this negative pool).
  Sampling stops once 1,001 negatives are collected (exact 1:1 match to
  the NOMLEX positive count), so effectively the ~1,300-1,400
  highest-frequency qualifying words out of the 10,000-word list.
- Known limitation: single-word (no-sentence-context) POS tagging is
  imperfect — a small number of negatives are mistagged (e.g. "require"
  → NN, "int" → NN, both false common-noun tags from the tagger on an
  isolated token). This was not hand-corrected; treat the negative pool
  as **~99% clean, not 100% hand-verified**.
- License: Google-10000-English list is public domain / freely
  redistributable (MIT-licensed `first20hours/google-10000-english`
  GitHub release is the common source for this list; the specific copy
  in `raw/google-10k.txt` predates this task and was not independently
  re-sourced here — see the sibling task's own documentation of this
  file if present).

### Reproducing

```
cd datasets/instrument_validation
python3 build_nominalization_gold.py
```
reads from `raw/nomlex/` and `raw/google-10k.txt` (already downloaded/
present in `raw/`) and (re)writes `nominalization_gold.tsv`. Requires
`nltk` with the `wordnet` and `stopwords` corpora and the
averaged-perceptron POS tagger already available locally.

---

## Etymological stratum (3-way: germanic / latinate / greek origin of an English word)

**Gold file:** `etymology_gold.tsv`
**Columns:** `word` · `stratum` (one of `germanic`, `latinate`, `greek`)
**Row count:** 10,814 unique lowercase, single-token English words (`latinate` 8,968 / `germanic` 967 / `greek` 879)

### Source

**Gerard de Melo's Etymological WordNet ("etymwn"), 2013-02-08 snapshot, mined from English Wiktionary**
- Canonical source page: `http://icsi.berkeley.edu/~demelo/etymwn/` — **unreachable** at the time of this
  build (July 2026): the ICSI host `www1.icsi.berkeley.edu` times out on both HTTP and HTTPS (port
  appears filtered/decommissioned), and `demelo.org` (the author's personal site, otherwise live) 404s
  on the `/etymwn/` path. The raw `etymwn.tsv` (298.76 MB, ISO-639-3-coded triples) is therefore **not
  directly downloadable today** from the original publisher.
- **What was actually downloaded**: a JSON re-packaging of the same 2013-02-08 etymwn data,
  `etymologies.json` (98.5 MB), from the GitHub mirror repo
  `https://github.com/parker57/making-sense-of-etymwn`
  (file: `https://raw.githubusercontent.com/parker57/making-sense-of-etymwn/master/etymologies.json`).
  That repo's own README documents that it sourced the "2013-02-08" etymwn.tsv from the same ICSI URL
  above and converted the "is_derived_from" relation into a nested JSON (`{lang: {word: [{etymon: lang}, ...]}}`),
  after dropping multi-word phrases and non-etymology relations (variant/etymologically_related). Saved
  to `raw/etymologies.json`; the mirror's own docs saved to `raw/readme_for_etymwn.txt` (original etymwn
  README) and `raw/README_repo.md` (mirror repo README) for provenance.
- **License**: CC-BY-SA 3.0 (per `raw/readme_for_etymwn.txt`), "Based on the contributions of the
  English Wiktionary community."
- **Citation**: Gerard de Melo, Gerhard Weikum. "Towards Universal Multilingual Knowledge Bases." In:
  *Principles, Construction, and Applications of Multilingual Wordnets*, Proceedings of the 5th Global
  Wordnet Conference (GWC 2010), Narosa Publishing, New Delhi, India, 2010.

### Resources found but NOT freely/directly downloadable (checked, not used)

- **CELEX2 (Dutch Centre for Lexical Information)** — has explicit word-origin/etymology fields; requires
  an LDC license (LDC96L14) and is not free — skipped per task instructions.
- **Original ICSI etymwn.tsv** — publisher page down (see above); only the GitHub JSON mirror was usable.
- Searched for published Germanic-vs-Latinate word lists from psycholinguistics literature (Corson's
  "lexical bar" studies, Bar-Ilan & Berman academic-vocabulary work) — no freely downloadable
  machine-readable list of this kind was located (these papers typically report summary statistics or
  small stimulus lists inside PDFs, not open datasets); none downloaded.

### Extraction / normalization rules (`raw/_stage1_rows.json`, `raw/_stage2_rows.json` are intermediate
caches from the build; not part of the final artifact)

1. Start from `etymologies.json["eng"]`: 273,499 English headwords, each with a list of one-or-more
   `{etymon_word: etymon_lang}` "is_derived_from" targets (ISO 639-3 / etymwn proto-language codes).
2. **Keep only words with exactly one listed etymon** (`len(etymons) == 1`). Words with 0 or ≥2 listed
   etymons (e.g. compounds with multiple attested sources, or affix+stem pairs like
   `polyaxiality <- {-ity, polyaxial}`) are dropped as ambiguous/mixed-origin. This drops 82,601 words.
3. **Keep only single-token, all-lowercase alphabetic words**, length ≥ 3 (regex `^[a-z]+$`). Drops
   proper nouns (`Maas`), multi-word phrases, and hyphen/affix fragments (`-ity`, `-ness`).
4. **Language → stratum mapping** (applied to the etymon's language code):
   - `germanic`: Old English (`ang`), Old Norse (`non`), German (`deu`), Dutch (`nld`), Middle Dutch
     (`dum`), Old High German (`goh`), Middle Low German (`gml`), Swedish/Danish/Afrikaans/Frisian/Old
     Saxon (`swe`,`dan`,`nno`,`nob`,`frr`,`fry`,`osx`)
   - `latinate`: Latin (`lat`), Old French (`fro`), Middle French (`frm`), French (`fra`), Anglo-Norman
     (`xno`), Spanish/Old Spanish (`spa`,`osp`), Italian/Old Italian (`ita`,`oit`), Portuguese (`por`),
     Catalan (`cat`), Romanian (`ron`)
   - `greek`: Ancient Greek (`grc`), Medieval Greek (`gkm`), Modern Greek (`ell`)
   - Words whose single etymon is in none of these (e.g. Arabic, Japanese, Sanskrit, Hebrew donor
     words, or Middle English `enm` — dropped because it is just an earlier stage of English itself,
     not informative about ultimate stock) are excluded.
5. **One extra resolution hop for "conduit" languages** (`deu`, `nld`, `fra`, `ita`, `spa`, `swe`,
   `dan`): since a modern-language etymon can itself be a loanword (classic case: scientific
   neoclassical coinages borrowed into English via a modern German/French/Italian form, e.g.
   `mitosis <- German "Mitose"`, itself Greek-rooted), for these languages the etymon word is looked
   up again inside `etymologies.json[etymon_lang]`:
   - if that word is itself a key with exactly one further etymon, the *further* etymon's language
     is used for the stratum mapping instead (980 words resolved to the same stratum on the 2nd hop,
     61 words were reclassified to a *different* stratum on the 2nd hop);
   - if that word is a key with ≥2 further etymons (ambiguous), the whole entry is **dropped**
     (474 words excluded this way);
   - if that word is not a key at all in the mirror's JSON (no further chain recorded — the common
     case), the 1-hop classification is kept as-is.
   - **Known limitation**: this only catches 2-hop chains that happen to be present in the mirror's
     JSON. Some genuinely Greek-rooted words that entered English via Latin or Romance conduits with
     no further recorded chain (e.g. `archipelago <- Italian "arcipelago"`, ultimately Greek
     `arkhi-` + `pelagos`; `athenaeum <- Latin`, ultimately Greek `Athenaion`) remain classified by
     their *immediate* donor language per the task's stated mapping convention (Latin/French/Italian
     → latinate is applied literally, without recursively chasing Latin/Romance words back to Greek).
     This is a known, documented simplification, not a data error.
   - Old English/Old Norse/other directly-terminal Germanic ancestor codes (`ang`, `non`, `goh`,
     `gml`, ...) are treated as terminal and never given the extra hop, since they already are the
     oldest attested Germanic stage.
6. Deduplicate by word (first occurrence kept; input order is Python dict iteration order over the
   source JSON, so effectively arbitrary — no systematic bias observed on spot-check).

### Row counts by stratum

| stratum | rows |
|---|---|
| latinate | 8,968 |
| germanic | 967 |
| greek | 879 |
| **Total** | **10,814** |

### 30-item manual spot-check (random sample, seed=7, checked against known etymology by the author)

29/30 correct against the etymwn-assigned immediate-donor-language classification; 1 borderline
(`archipelago` → `latinate` via Italian `arcipelago`, but the word is ultimately a Greek compound
`arkhi-` + `pelagos` with no further chain recorded in the mirror data — see limitation above; counted
as "correct" under the task's literal Italian→latinate mapping rule but flagged here as a known
edge case). No outright wrong classifications were found (e.g. no Latin words tagged `germanic`, no
Old-English-derived words tagged `latinate`). Spot-checked words included: `annuent` (lat→latinate ✓),
`archipelago` (ita→latinate, borderline, see above), `asterism` (grc→greek ✓), `athenaeum`
(lat→latinate ✓, itself ultimately Greek "Athenaion" but correctly latinate under the literal-Latin
rule), `aufwuch` (deu→germanic ✓, "Aufwuchs" is a genuine direct German loan), `ballottement`
(fra→latinate ✓), `barbitos` (lat→latinate ✓ under literal rule; Greek loanword into Latin),
`bourdon` (fro→latinate ✓), `browse` (frm→latinate ✓, Middle French donor per rule; deeper root is
Frankish/Germanic but immediate donor is French), `cacodemon` (grc→greek ✓), `chromid` (grc→greek ✓),
`constant` (lat→latinate ✓), `effascinate` (lat→latinate ✓), `enascent` (lat→latinate ✓), `excelsior`
(lat→latinate ✓), `indefatigable` (lat→latinate ✓), `liripoop` (fro→latinate ✓), `miscellanea`
(lat→latinate ✓), `novelty` (fro→latinate ✓), `occasive` (lat→latinate ✓), `orphanotrophy`
(lat→latinate ✓ under literal rule), `quotidian` (xno→latinate ✓), `saccholactic` (fra→latinate ✓),
`semantron` (grc→greek ✓), `snook` (nld→germanic ✓, Dutch "snoek"), `strigil` (lat→latinate ✓),
`stringendo` (ita→latinate ✓), `twin` (ang→germanic ✓), `unition` (lat→latinate ✓), `vraka`
(ell→greek ✓, modern Greek loanword).

---

## Specificity

**Gold file:** `specificity_gold.tsv`
**Columns:** `text` · `label_or_score` · `scale` · `source`

The gold file currently contains its header and **0 data rows**. This is intentional:
no sentence text was copied from a source whose complete labeled files could both be
retrieved in this environment and safely treated as redistributable. In particular, no
model predictions, PDF examples, or labels reconstructed mechanically from WordNet were
substituted for human/discourse gold.

### Row counts

| source | rows in `specificity_gold.tsv` | rows reported/visible upstream | status |
|---|---:|---:|---|
| Ko, Durrett & Li (2019), Twitter | 0 | 984 | public GitHub data; download blocked here; repository has no explicit license |
| Ko, Durrett & Li (2019), Yelp | 0 | 845 | public GitHub data; download blocked here; repository has no explicit license |
| Ko, Durrett & Li (2019), movie reviews | 0 | 920 | public GitHub data; download blocked here; repository has no explicit license |
| Louis & Nenkova / Speciteller PDTB-derived news training data | 0 | 2,796 (1,398 pairs) | underlying PDTB/WSJ text is LDC/Dow Jones licensed |
| Louis & Nenkova (2012) directly judged news evaluation data | 0 | 885 in Li & Nenkova (2015); later papers round/report 900 or 894 | sentence text is from WSJ/NYT/AP; no standalone redistributable release located |
| Gao et al. (2019) SpecificityTwitter (additional open lead) | 0 | 7,267 | MIT-tagged Hugging Face mirror found; download blocked here |
| **Total** | **0** | — | — |

### Sources, URLs, licenses, citations, and scale semantics

**1. Ko, Durrett & Li (2019), “Domain Agnostic Real-Valued Specificity Prediction”**

- Paper: `https://doi.org/10.1609/aaai.v33i01.33016610`
- Authors' repository:
  `https://github.com/wjko2/Domain-Agnostic-Sentence-Specificity-Prediction`
- Labeled files advertised by the repository:
  `dataset/data/twitters.txt` + `twitterv.txt`,
  `dataset/data/yelps.txt` + `yelpv.txt`, and
  `dataset/data/movies.txt` + `moviev.txt`.
- The paper reports 984 retained Twitter items, 845 Yelp items, and 920 movie-review
  items (2,749 total). Nine workers initially rated 1,000 sentences per domain on a
  five-point ordinal scale, 1 = very general through 5 = very specific. Workers below
  the agreement threshold were excluded; sentences with at least five retained ratings
  were kept. The authors rescaled individual ratings to `{0, .25, .5, .75, 1}` and the
  released `*v.txt` value is their mean, so higher = more specific.
- Citation: Wei-Jen Ko, Greg Durrett, and Junyi Jessy Li. “Domain Agnostic
  Real-Valued Specificity Prediction.” AAAI 2019.
- License: the public repository has no `LICENSE` file and its README states no data or
  code license. Public accessibility is therefore documented separately from permission
  to redistribute. The Yelp and movie text also originate in third-party corpora, and
  the Twitter subset has platform/content-rights considerations.
- Retrieval result: GitHub HTML and the repository's file listing were visible via web
  search, but the execution environment had no outbound DNS/network access
  (`git ls-remote` failed with “Could not resolve host: github.com”). Browser retrieval
  exposed file metadata but could not transfer complete files into the workspace. No
  partial rows were used.

**2. Louis & Nenkova (2011/2012) and Li & Nenkova (2015) / Speciteller**

- Louis & Nenkova paper:
  `https://aclanthology.org/I11-1068/`
- Li & Nenkova paper: `https://doi.org/10.1609/aaai.v29i1.9517`
- Speciteller page:
  `https://www.cis.upenn.edu/~nlp/software/speciteller.html`
- Speciteller code: `https://github.com/jjessyli/speciteller`
- Speciteller resource archive:
  `https://www.cis.upenn.edu/~nlp/software/speciteller_data.tar.gz`
- Scale semantics:
  - The PDTB-derived training construction takes Arg1 of each implicit
    `Expansion.Instantiation` relation as **general** and Arg2 as **specific**.
    Li & Nenkova report 2,796 training sentences (1,398 pairs; approximately 1.4K per
    class). These are indirect discourse-derived labels, not direct specificity ratings.
  - The direct evaluation corpus contains 885 sentences from nine complete WSJ, New York
    Times, and Associated Press articles, each judged by five annotators; the majority
    class is the binary general/specific label. The 2019 paper rounds this to 900, while
    a later survey reports 894; 885 is retained here as the exact count stated by Li &
    Nenkova (2015).
  - Speciteller's output is a classifier posterior in `[0,1]` (0 = most general,
    1 = most detailed). Those automatic outputs are **not gold** and were not included.
- Citation: Annie Louis and Ani Nenkova. “Automatic Identification of General and
  Specific Sentences by Leveraging Discourse Annotations.” IJCNLP 2011; Annie Louis and
  Ani Nenkova. “Text Specificity and Impact on Quality of News Summaries.” 2011; Junyi
  Jessy Li and Ani Nenkova. “Fast and Accurate Prediction of Sentence Specificity.”
  AAAI 2015.
- Licensing:
  - The Speciteller package/resource page states CC BY-NC-SA 3.0 for the package.
    The downloadable archive supplies model resources/lexicons; the public repository
    does not expose the labeled sentence training/evaluation text as a standalone gold
    dataset.
  - The training text is Wall Street Journal material from PDTB/PTB. PDTB 2.0 is
    LDC2008T05 and requires an LDC user agreement; portions are copyright Dow Jones.
    Official catalog: `https://catalog.ldc.upenn.edu/LDC2008T05`.
  - The directly judged evaluation sentences come from WSJ, NYT, and AP news articles.
    No free standalone release carrying both their text and majority labels was located.
    Consequently neither set was copied into this repository.

**3. PDTB-derived specificity labels**

- The original Louis–Nenkova derivation is the principal PDTB-derived candidate:
  implicit `Expansion.Instantiation` (and descriptions of the earlier work also discuss
  `Specification`) relations provide a relative general→specific ordering.
- PDTB 2.0 contains licensed WSJ text and is distributed through LDC rather than as a
  freely redistributable text corpus. A user who already holds the applicable PDTB/PTB
  license can reproduce the 2,796-row construction locally, but those text spans are not
  vendored here.
- A later thesis reports a separately preprocessed PDTB construction of 3,542 sentences
  (1,925 general, 1,617 specific), but this is a derived experiment over the same licensed
  source and no independently licensed release was found. It was not used.

**4. Additional open lead: Gao et al. (2019), SpecificityTwitter**

- Authors' repository: `https://github.com/cs329yangzhong/specificityTwitter`
- Hugging Face mirror:
  `https://huggingface.co/datasets/yznlp/specificityTwitter`
- Paper: `https://doi.org/10.1609/aaai.v33i01.33016415`
- The mirror reports 7,267 censored tweets split 5,767 train / 500 validation / 1,000
  test. `Score` is the crowd mean on a 1–5 scale, where 1 = most general and 5 = most
  specific.
- License: the Hugging Face dataset card is tagged MIT. The upstream GitHub repository
  itself does not display a `LICENSE` file, so downstream users should retain the mirror
  provenance and also consider platform/content terms for tweet text.
- Retrieval result: the browser verified schema, scale, split sizes, total row count,
  and MIT tag, but the sandbox could not download the CSV/Parquet artifacts to disk.
  The 100-row web preview was deliberately not treated as a complete dataset.

### Integrity / non-fabrication checks

- `specificity_gold.tsv` has exactly four tab-separated header fields and zero data rows.
- Reported counts above were cross-checked against the primary papers and repository/data
  cards; differing historical counts are preserved and explained instead of silently
  reconciled.
- No inferred, model-scored, PDF-example-only, or partial-preview rows are present.
