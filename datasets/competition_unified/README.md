# `datasets/competition_unified/`

Unified corpus of editorials + candidate submissions across competitive programming platforms (LeetCode, CodeChef, Codeforces, Luogu, AtCoder, USACO).

Files on **sk3** at `/lfs/skampere3/0/alexspan/norm-research/datasets/competition_unified/`:
- `editorials.parquet` — ~51K editorials
- `candidates.parquet` — ~808K candidate submissions
- `problems.parquet` — ~64K problems

## Schemas (key columns)

### `editorials.parquet`
```
platform, problem_id, problem_title, editorial_text, editorial_code,
editorial_lang, code_lang, source, difficulty, tags, editorial_id,
canonical_pid
```

### `candidates.parquet`
```
platform, problem_id, code, code_lang, language_norm, code_stripped,
source, verdict, submitter, runtime_ms, memory_kb, candidate_id,
canonical_pid
```

## Platform / source mapping — **READ THIS BEFORE FILTERING**

**Always key on the `platform` column, not on `source`.** `source` indicates where the row was scraped from; `platform` is what the row belongs to.

The same platform appears under many sources. The canonical mapping:

| `platform` | Editorial sources | Candidate sources |
|---|---|---|
| `lc` | `doocs/leetcode`, `leetcode-discuss` | `lc_discuss`, `taco`, `taco-verified` |
| `cc` | `codechef_discuss` | **`taco`, `taco-verified`** (the trap below) |
| `cf` | `cf_blog`, `organizer_blog`, `open-r1` | `matrixstudio`, `code_contests`, `taco`, `taco-verified` |
| `luogu` | `luogu_editorial` | `luogu_editorial_alt` |
| `ac` | (text-only editorials; `editorial_code` is NULL) | `atcoder_submission`, `taco`, `taco-verified` |
| `usaco` | `usaco_guide` | `usaco_guide` |

## ⚠️ Pitfall: filtering candidates by `source` will DROP whole platforms

2026-06-07 bug we don't want to repeat: a Phase 1 pool builder mapped candidates by `source` only:

```python
def map_cand_platform(src):              # BUG — drops CC + most AC entirely
    s = str(src).lower()
    if s == "matrixstudio" or s == "code_contests": return "cf"
    if s == "lc_discuss": return "lc"
    if "luogu" in s: return "luogu"
    if s == "atcoder_submission": return "ac"
    return None                          # taco / taco-verified → discarded
```

Because **45K of the CC candidates and ~17K of the AC candidates are tagged `source=taco-verified` or `source=taco`**, the function returned `None` for them and the build dropped them silently. Result: Phase 1 pool had 0 CC pairs (and most AC missing) even though the data was right there. The "CC = 0 candidates" claim was incorrect — CC has 45,521 candidates.

**The fix**: read `platform` directly.

```python
plat = cands["platform"]   # already 'lc' / 'cc' / 'cf' / 'luogu' / 'ac' / 'usaco' / etc.
```

If you really need to filter by source (e.g. exclude `taco` because of license), do `platform` first then exclude by source.

## Editorial-side gaps that ARE real (not just builder bugs)

| Gap | Cause | What to do |
|---|---|---|
| **CF editorials have no `editorial_code`** | `cf_blog` / `organizer_blog` editorials are blog **text** — the code is embedded in HTML/markdown but not pulled into a code column | Extract code blocks from `editorial_text` (Claude or HTML parser) |
| **AC editorials have no `editorial_code`** | Same — text editorials only | Extract code from `editorial_text` |
| **AC `submissions.parquet` has `code = NULL`** | The kenkoooo API scrape stored verdict/timing metadata but not the source code | Re-scrape via HF gated `Nan-Do/atcoder_abc_contests_small` (task #106) |

## Formable-pair counts (filtering by `platform`, requiring both ed-code and cand-code to be non-null and ≥30 chars)

As of 2026-06-07:

| Platform | Editorials w/ code | Candidates w/ code | Problems w/ both | Formable pairs |
|---|---:|---:|---:|---:|
| lc | 9,422 | 180,064 | 3,466 | 583,599 |
| cc | 3,876 | 45,469 | 3,784 | 21,758 |
| luogu | 9,434 | 52,268 | 9,429 | 52,207 |
| usaco | 900 | 900 | 900 | 900 |
| cf | 461 (extracted, see below) | 394,538 | 155 | 7,468 |
| ac | 3,651 (extracted, see below) | 39,623 | 1,122 | 12,444 |
| **TOTAL** | **27,744** | **712,862** | **18,856** | **678,376** |

## `editorials_code_extracted.parquet` — extracted CF + AC editorial code (2026-06-10)

CF + AC editorials had `editorial_code = ""` in `editorials.parquet` because the
upstream scrape stored only `editorial_text` and the code was inlined (often in
"one token per line" tokenized form, sometimes immediately duplicated after a
literal `Copy\n` AtCoder-button artifact). `scripts/competition_editorials/extract_editorial_code.py`
parses those code regions and writes a separate file (the original
`editorials.parquet` is left UNCHANGED):

```
/lfs/skampere3/0/alexspan/norm-research/datasets/competition_unified/editorials_code_extracted.parquet
```

Columns: `editorial_id, platform, problem_id, canonical_pid, source, extracted_code, code_lang, n_code_blocks, extraction_confidence`.

Confidence buckets and coverage (full run on 8,833 cf + 9,974 ac rows):

| Confidence | meaning | cf | ac |
|---|---|---:|---:|
| `confident` | one full program with `main()` / `if __name__` / `public static void main` | 250 (2.8%) | 2,910 (29.2%) |
| `confident_multi` | multiple full-program candidates (often C++ + Python for same problem) — best one is returned | 211 (2.4%) | 741 (7.4%) |
| `ambiguous` | code-ish block found, but missing a `main` — could still be useful | 7 | 75 |
| `ambiguous_short` | partial block, ≥30 chars, scanner stopped early on a prose-like line | 4 | 281 |
| `none` | no code markers (`#include`, `def`, `import sys`, etc.) found at all | 8,361 (94.7%) | 5,967 (59.8%) |

**The CF rate is at the data ceiling**: only ~460 of 8,833 (5.2%) CF editorials
contain any concrete code marker in their text. The rest are pure-prose
editorials (especially `open-r1`, 5,428 rows, virtually 0 code). LLM rescue
pass would gain ≤ ~5 rows for CF and ~40 for AC — not worth it.

Use only `extraction_confidence in {confident, confident_multi}` for downstream
pairing. The extracted code from heavily-tokenized blocks may have imperfect
whitespace/indentation (extra spaces around brackets) but is semantically
correct and is fine for embedding-based similarity / reward-model training.

## File versioning

`*.parquet` is the current version; `*_v1.parquet` and `_summary_v1.json` are the previous revision (kept for reproducibility). `_summary.json` documents the build inputs.
