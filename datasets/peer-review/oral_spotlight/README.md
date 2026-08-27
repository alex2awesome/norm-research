# oral_spotlight — acad-B (expert curation) task

Oral/spotlight vs poster, **conditional on acceptance**, ICLR 2018–25 +
NeurIPS 2021–25 (excl. 2022, no tiers) + ICML 2023–25. Built from
`../unified_papers.csv.gz` `decision_raw` (tier alias table in
`build_oral_spotlight.py`); the main peer-review task's binary `judgement`
is untouched — this is a separate derived task.

- 33,890 papers; 4,768 positive (oral+spotlight), 14.1%.
- `train/eval/test.csv.gz` (80/10/10 by stable md5 of paper_id).
  Columns: `id, text (TITLE+ABSTRACT), judgement, tier, venue_key, year`.
- Label is **budget-bound (B1)**: committees promote a roughly fixed
  fraction → label is RELATIVE within venue×year. Always stratify/evaluate
  within (venue_key, year). Per-venue-year counts cross-validated against
  the OpenReview API (exact match on ICLR 2024).

## Dataset-first protocol results (2026-06-12)

- Spot-check: clean; no leakage tokens in text.
- **TF-IDF/LR floor: AUC 0.530 overall, 0.528 mean within venue×year** —
  essentially chance. No length confound (r=0.01); top features noise-like.
- Interpretation: expert curation among accepted papers is nearly invisible
  to bag-of-words — the articulable layer, if any, must be semantic
  (judge metrics). Contrast: accept/reject tasks have much higher lexical
  floors. Reference point: Gong et al. 2026 report 59.2% for fine-tuned
  LLMs on publication tiers.
- **bge+LR dense-lite probe: AUC 0.597 pooled, 0.591 mean within
  venue×year** (eval split, bge-large-en-v1.5 on sk3). Semantic headroom
  over BoW is real (+0.06) and survives the venue×year control — but much
  smaller than caption_contest's (+0.16 on the same probe pair). With
  title+abstract X, this cell sits near the bottom of the gradient.
- Caveat for V/A/T: oral selection partly reflects review scores +
  presentation considerations not visible in the abstract; full text via
  arXiv linkage is the planned X upgrade — the dense-lite number above is
  the floor that upgrade must beat.
