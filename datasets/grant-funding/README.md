# Grant Funding

Predict whether a research grant proposal will be **funded** or **rejected** from the proposal text. This task is registered in the project pipeline (`grant-funding` in `scripts/sk3_train_ce.py`, `scripts/extract_rubric_features.py`) but is currently **not modeled end-to-end** because no source combines (a) full proposal text with (b) reliable accept/reject labels at meaningful scale. This README documents what we have, what we tried, and why it does not yet form a usable canonical dataset.

## Task

- **Label** — binary `funded` vs `unfunded` (or `rejected`).
- **Input** — proposal narrative / abstract / sections.
- **Open status** — there is currently **no canonical labeled dataset file** for this task analogous to those in `reference_v2_task_datasets.md`. The only labeled text we have is the ~236-row Open Grants set below; everything else lacks either text or rejection labels.

## Sources

### 1. NIH RePORTER — funded grants only
- Path: `datasets/grant-funding/nih_exporter/` (2.8 GB, FY1985–FY2024).
- Files: `RePORTER_PRJ_C_FY{YYYY}.zip` (project metadata) and `RePORTER_PRJABS_C_FY{YYYY}.zip` (abstracts), 80 zips total.
- **What it gives us**: PI, institution, FOA, award amount, *abstract* of the funded version.
- **What it does NOT give us**:
  - No full proposal narrative (only the abstract is public).
  - No rejected proposals at all — RePORTER is a database of *awarded* grants.
- **A0/A1 SUFFIX proxy** — see "Key decisions" below.

### 2. Open Grants — small, voluntarily shared, labeled
- Path: `datasets/grant-funding/open-source-grants/processed/`.
- Built from ogrants.org markdown frontmatter and RIO Journal full-text proposals.
- **Labeled rows**: 236 in `grants_labeled.csv` (200 funded / 36 unfunded; 145 from ogrants, 91 from RIO Journal).
- **Skew**: 85% funded; near-useless class balance, tiny absolute count.
- **Text coverage**: `grants_with_text.csv` (~131k lines after wrapping), `rio_full_text.jsonl` (135 lines = 135 RIO proposals), `grant_metadata.csv` (438 lines).
- This is the only source we have where *both* the proposal text and an accept/reject decision are present, but it is far too small to train a dense model and the label distribution is heavily biased toward funded submissions.

### 3. ERC (European Research Council) — funded only, used for clustering
- Path: `datasets/grant-funding/erc/`.
- Downloaded by `download_erc_projects.py` from CORDIS bulk open data + the ERC Datahub for panel codes (FP7 + Horizon 2020 schemes).
- **No rejection labels** — ERC bulk data is funded-only, same problem as NIH.
- Used so far only as a *clustering* corpus to study panel/topic structure: `erc_clusters__{random,semantic_tfidf,rhetorical,budget_duration}.csv` (produced by `cluster_erc_projects.py`).

### 4. Online rubrics — agency evaluation criteria
- Path: `datasets/grant-funding/online-rubrics/`.
- 3,040 raw HTML pages, 190 Claude-parsed markdown summaries of agency criteria (NIH, NSF, AHRQ, ANR, ARC Australia, ARPA-H, ERC panels, plus the Bornmann meta-analyses of peer-review reliability, etc.).
- Inputs to the proposer / rubric-discovery pipeline, **not** labeled training data.

### 5. Direct-from-NIH attempts (not yet successful)
- `nih_csr_collaboration_email.txt` — draft of an outreach email to the NIH Center for Scientific Review proposing a collaboration that would expose unfunded application text.
- `nih_foia_request.txt` — draft FOIA request. Expected to be denied or heavily redacted under Exemptions 4 (confidential commercial) and 5 (deliberative process).
- No data has come back from either channel as of this writing.

## Collection scripts

| Script | Purpose |
| --- | --- |
| `open-source-grants/build_grant_dataset.py` | Parse ogrants.org YAML frontmatter, merge with RIO Journal full-text, emit `processed/grants_labeled.csv` and `grants_with_text.csv`. |
| `open-source-grants/extract_text.py` | Extract text from downloaded PDFs/HTML proposals (pdfplumber + BeautifulSoup) and combine with RIO full-text. |
| `open-source-grants/fix_downloads.py` | Repair / retry download failures from the proposal scrape. |
| `download_erc_projects.py` | Download all funded ERC projects from CORDIS, enrich with ERC Datahub panel codes, emit `erc_projects.csv` + `erc_euroscivoc.csv`. |
| `cluster_erc_projects.py` | Cluster ERC projects four ways (random / semantic TF-IDF / rhetorical / budget+duration) for panel-stratified analysis. |

**Attempted but not committed as a per-grant labeled artifact**: the A0/A1 SUFFIX aggregation. A script `nih_a0_a1_aggregation.py` and a summary CSV `nih_a0_a1_summary.csv` were produced on sk3 during the 2026-04-10 investigation; the FY-level summary lives in `running-research-notes.md` but neither file is currently present at `datasets/grant-funding/` on this machine. Re-derive from `nih_exporter/RePORTER_PRJ_C_FY*.zip` if needed (see "Key decisions").

## File layout

```
datasets/grant-funding/
├── nih_exporter/                     # 2.8 GB, 80 NIH RePORTER zips FY1985–FY2024
│   ├── RePORTER_PRJ_C_FY{YYYY}.zip       # project metadata (incl. APPLICATION_TYPE, SUPPORT_YEAR, SUFFIX)
│   └── RePORTER_PRJABS_C_FY{YYYY}.zip    # abstracts
├── open-source-grants/
│   ├── ogrants-repo/                 # cloned ogrants.org markdown source
│   ├── proposals/{ogrants,rio}/      # downloaded proposal PDFs / HTML
│   ├── processed/
│   │   ├── grant_metadata.csv        # 438 lines, per-grant metadata
│   │   ├── grants_with_text.csv      # all grants for which we have any text
│   │   ├── grants_labeled.csv        # 236 rows; 200 funded / 36 unfunded
│   │   └── rio_full_text.jsonl       # 135 RIO Journal full-text proposals
│   ├── build_grant_dataset.py
│   ├── extract_text.py
│   └── fix_downloads.py
├── erc/                              # 4 clustering CSVs (no labels)
│   └── erc_clusters__{random,semantic_tfidf,rhetorical,budget_duration}.csv
├── online-rubrics/                   # 892 MB scrape of agency review criteria
│   ├── raw/                          # 3,040 raw HTML pages
│   ├── claude-parsed/                # 190 parsed markdown summaries
│   ├── gpt-parsed/                   # 1 file (parse run abandoned)
│   ├── urls-visited.csv
│   └── waveh{3,4,5,6}_{log.csv,seen.txt}  # scrape provenance per wave
├── download_erc_projects.py
├── cluster_erc_projects.py
├── nih_csr_collaboration_email.txt   # outreach draft to NIH CSR
├── nih_foia_request.txt              # FOIA draft
└── README.md (this file)
```

## Canonical dataset file

**There is no canonical labeled dataset for this task yet.** The closest thing is `open-source-grants/processed/grants_labeled.csv` (236 rows, 200/36 funded/unfunded), but it is too small and too skewed to be the canonical file. NIH RePORTER lacks rejection text entirely. ERC lacks rejection labels entirely.

When/if a usable dataset emerges, it should live at `open-source-grants/processed/` (or a new sibling) and be registered in `reference_v2_task_datasets.md`.

## Modeling state

- **No supervised model has been trained for `grant-funding`** in the v2 / cells / dense-sweep pipelines. The task is enumerated in `scripts/sk3_train_ce.py` and `scripts/extract_rubric_features.py`, but neither has produced cells / metrics for it because there is no labeled dataset to score.
- The `online-rubrics/` corpus has fed agency-criteria prompts to the proposer pipeline, but downstream metric extraction for grant-funding is not yet on disk.

## Key decisions

1. **NIH RePORTER alone cannot support a reject-vs-accept text classifier.** RePORTER contains funded grants only, so rejected A0 submissions never appear as rows. Empirically confirmed in FY2023: 0 `CORE_PROJECT_NUM` values had both an A0 and an A1 row.
2. **A0/A1 SUFFIX proxy** (investigated 2026-04-10, see `running-research-notes.md` → "Grant Funding → NIH A0/A1 resubmission signal"):
   - Filter to `APPLICATION_TYPE=1, SUPPORT_YEAR=1` (new-year-1 funded grants).
   - `SUFFIX = ''` → funded on original (A0).
   - `SUFFIX = 'A1'` → A0 was rejected, funded on 1st resubmission.
   - `SUFFIX = 'A2'` → A0 and A1 both rejected (rare post-FY2011 due to NIH's one-resubmission policy).
   - Aggregation across FY1985–FY2024 (506,587 new-year-1 funded grants): 340,096 funded on A0 (67.1%), 141,547 A1, 23,233 A2, 1,371 A3+ → **166,151 (~32.8%) had a rejected A0**.
   - **Limit**: only the *funded* version's abstract is available, never the rejected A0's text. So this gives a grant-level binary outcome but the input is the wrong text — you'd be predicting "did this funded abstract have a rejected predecessor" from the funded version, not predicting acceptance from the submitted version.
3. **Open Grants is the only text + label source** but is 236 rows total with 85% funded. Not enough for dense modeling; possibly usable as a held-out probe set if a richer training source is found.
4. **ERC is funded-only too**, so it serves clustering / topic analysis rather than classification.
5. **sk3 mirror**: the PRJ files (not abstracts) are mirrored at `/lfs/skampere3/0/alexspan/norm-research/datasets/grant-funding/nih_exporter/RePORTER_PRJ_C_FY*.zip` (~1.0 GB). Abstracts are not on sk3 — pull from the local copy if needed.

## Open questions / next steps

- **Get rejected proposal text.** Options, in rough order of tractability:
  - NIH CSR collaboration (draft: `nih_csr_collaboration_email.txt`).
  - FOIA (draft: `nih_foia_request.txt`) — expected to be denied/redacted under Exemptions 4 and 5.
  - PI partnerships: ask labs to share rejected A0 narratives.
  - Expand Open Grants scrape coverage (more agencies, more years).
- **Use A0/A1 SUFFIX as a *post-funding* signal**, not as the accept/reject label. E.g., predict from the funded abstract whether the project required resubmission — a weak proxy for "how borderline was this".
- **Replicate the 2026-04-10 aggregation locally**: re-run `nih_a0_a1_aggregation.py` (script lives on sk3) on the local zips and commit `nih_a0_a1_summary.csv` to this directory.
- **Decide whether to keep `grant-funding` as a registered task** in the pipeline if no labeled text source materializes within the paper window.
