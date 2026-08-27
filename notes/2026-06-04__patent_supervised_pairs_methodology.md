# Methodology Notes: Compiling Clean §102 / §103 Supervised Pairs

Built up incrementally while debugging the v2 retriever (June 4, 2026).
Capture both the data structures we found and the pitfalls so we don't
relearn them.

## Data sources on disk

| File | What | Rows | Granularity |
|---|---|---:|---|
| `patents_dataset.jsonl.gz` | Pre-grant publications + grant text | 4.69M | per pgpub |
| `processed/granted_patents_claim1.parquet` | Granted patent claim 1 (2001-2025) | 6.20M | per patent_id |
| `processed/pgpub_claims1.parquet` | All pre-grant pub claim 1 | 8.30M | per pgpub_id |
| `processed/claim1_lookup.parquet` | Legacy (older + design + plant + reissue) | 6.92M | per patent_id |
| `processed/google_patents_supplement.parquet` | Older US grants + design via scraping | ~1.4M (still growing) | per raw_id |
| `raw/oard/oard_rejections_by_app.csv` | Per-app rejection FLAGS | 2.19M apps | aggregated |
| `raw/oard/oard_citations.csv` | Per-OA-per-cite-per-rejection-row records | ~60M | **fine-grained** |

## Key insight: OARD citation rows ARE tied to specific rejection grounds

`oard_citations.csv` columns:
```
app_id, citation_pat_pgpub_id, parsed, ifw_number,
action_type, action_subtype, form892, form1449, citation_in_oa
```

The `action_type` field is per-row. Distribution (from a 5M-row sample):
- `''` (empty): 79% — IDS-form citations (Form 892/1449), not tied to a rejection
- `'103'`: 17%
- `'102'`: 4%

**Clean §102 supervision** = filter to `action_type == '102'`.
**Clean §103 supervision** = filter to `action_type == '103'`.

## What went wrong with v2 training pairs

`scripts/extract_anticipation_training_pairs_v2.py` did the loose join:
1. Filter to apps where `rejected_102=True` (anywhere in their OA history)
2. Take ALL examiner cites for those apps regardless of action_type
3. Pair each as positive

This conflates: §102-anticipating cites (real positives), §103-cited refs (related
but not anticipating), and IDS-disclosed prior art (cited by applicant, not used
in any rejection). Resulting pairs are ~80%+ noise from a §102 perspective.

## What went wrong with v2 FAISS index

`scripts/build_faiss_index_direct.py` only read from `patents_dataset.jsonl.gz`
(4.7M docs). But training pairs were built from 4 text sources (~20M docs).
So when evaluating recall, **79% of pairs had their `positive` not in the index**
— not because the doc didn't exist, but because the index missed sources we
already had. v3 builder (`build_faiss_index_v3.py`) reads all 5 sources.

## What went wrong with OA fetching

USPTO Open Data Portal has two formats per office action:
- **PDF**: large (1-3 MB), but for older OAs these are **scanned images** —
  pypdf returns ≤30 chars from a 29-page doc. Unusable without OCR.
- **MS_WORD** (.DOCM): real text source. BUT: download URL returns a plain-text
  message containing a 1-hour-valid redirect URL to `data-documents.uspto.gov`.
  Need to extract the redirect URL with regex and follow it.

Both prompted silent failures in early scripts because of swallowed exceptions.
Always surface errors verbatim during initial probing.

## What's actually in the OA text

Examiners cite patents in varied formats:
- `US 2006/0242362 A1` (pgpub with slash)
- `US 20060242362 A1` (pgpub no slash)
- `US 6,807,434 B2` (granted with commas + kind code)
- `Smith et al. (US 6,807,434)` (informal)

OARD stores the same patents as bare digits: `20060242362`, `6807434`. Substring
matching the OARD ID against OA text misses every formatted variant. Need to:
1. Generate format variants of the OARD ID
2. OR extract all patent-number-shaped tokens from OA text and normalize them

## Validation results on v2 (claim-1-only, noisy pairs, partial index)

- Recall@1: 2.40%
- Recall@10: 10.85%
- Recall@50: 18.64%
- MRR: 0.0523
- Same-CPC-section in top-K: 62.8%
- Same-section recall@10: 23.81%
- Cross-section recall@10: 4.17%
- Anchor↔positive similarity when found: mean 0.852, p10 0.782, p90 0.937

Interpretation: 79% of "positives" weren't even in the index. Of the 21% that
were, claim-1-vs-claim-1 recall@10 was only 23% same-section. **The claim-only
representation is fundamentally too coarse** because:
- Many "positives" are §103/IDS noise, not real §102 anticipations
- Real §102 anticipations often live in the cited ref's spec, not its claim 1

## Recommended v3 pipeline (clean version)

1. **Clean pair extraction**: filter `oard_citations.csv` to `action_type='102'`,
   join to OARD app text by app_id, get the SPECIFIC cite for each §102 rejection.
2. **Full-text corpus indexing**: index not just claim 1, but full spec passages
   for each prior art document. (Multi-day vLLM extraction)
3. **Limitation-level query**: decompose each examined claim into limitations,
   search per-limitation. §102 hit = all query limitations match in one prior
   art doc.
4. **OA-ground-truth validation**: download MS_WORD OAs via redirect, extract
   text, locate the §102 paragraph, verify our retriever recovers the actual
   cited ref. Use this as the honest eval metric.

## Pitfalls list (do not repeat)

- Don't swallow exceptions in fetchers — surface them
- Don't compare normalized vs raw patent numbers; choose one format
- Don't stream multi-GB JSONL once per query item; batch the lookups
- Don't trust PDF text without checking length
- Don't trust an index unless you've also verified the eval queries' positives
  are present in it
- Don't conflate "app had §102 rejection" with "this cite was a §102 anticipation"
- Always check the OARD `action_type` column at the row level, not the
  aggregated per-app flag

## Decision (2026-06-05): Drop unfindable §102/§103 training examples

We will **drop training pairs where we cannot find the cited prior art's text** (claims or spec) in our corpus, rather than including them as weak negatives or padding with random patents.

Rationale:
- Pairs where the positive isn't findable have an unrecoverable 0% recall ceiling on validation — they look like training noise.
- They also distort our recall metrics: the validation v2 reported 79% pairs unfindable, which inflated the "missing" rate and made the model look worse than it was.
- Once we ingest PVGPATTXT/PVPGPUBTXT (covers 1976-2025) coverage rises to ~95%+, so the residual dropped pairs are mostly design patents (~7%) and foreign refs (~0.2%).

Implementation:
- After building the new spec-augmented corpus, recompute `missing_after_local_sources.csv`.
- Filter `anticipation_training_pairs_v2.jsonl.gz` (and any clean §102 version) to drop pairs where the cited pgpub_id (or normalized variant) isn't present in the indexed corpus.
- Document the drop count and reason in the run notes.
