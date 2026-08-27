# OARD Deployment Plan — adding examiner office-action text to the patents corpus

Investigation done: **2026-06-01**. Status: no data was downloaded or queried; this is a planning document only.

---

## TL;DR — the critical finding

**The OARD bulk dataset does NOT contain free-form examiner text.** Lu, Myers, Beliveau (2017) used NLP/regex on the underlying office action prose to extract *structured* fields (rejection grounds 101/102/103/112, claims affected, prior art IDs, examiner art unit, dates). The original text was discarded. The BigQuery copy `patents-public-data.uspto_oce_office_actions` is the **same three tables** as the bulk CSVs — no extra text columns.

So the existing scripts (`08_download_oa_citations.sh`, `09_query_oard_bigquery.py`) give us:
- accurate per-application examiner citation lists (for the existing `oard_examiner_cites` pipeline in `06_build_augmented_datasets.py`),
- per-application rejection-code histograms (101/102/103/112 counts),
- **but no examiner reasoning text**.

If we want the *articulated rationale* (the thing that makes this a "norm articulation" dataset), we have to go to a different source — most plausibly the **USPTO Office Action Weekly Zips / OACT bulk product**, which contains the JSON body of each office action issued from 2020-01-06 onward.

---

## 1. Existing OARD scripts — what they do, what they need

| Script | Purpose | Auth | Input | Output | TODO/issues |
|---|---|---|---|---|---|
| `08_download_oa_citations.sh` | Pull 3 OARD bulk zips from USPTO | none | none | `raw/oard/{office_actions,rejections,citations}.csv` (~250 MB after unzip) | **URL dead** — `bulkdata.uspto.gov` no longer resolves DNS. ODP fallback URL in script is also dead (returns SPA shell). |
| `09_query_oard_bigquery.py` | Pull `citations` table from BigQuery | GCP (ADC) | `--project usc-research` | `raw/oard/oard_citations.csv` | Hard-codes `usc-research` project. Only pulls citations, not the other tables. |
| `10_get_pgpub_app_mapping.py` | Build pgpub_id → application_number map | GCP (ADC) | none | `processed/pgpub_to_appnum.parquet` | Same project hard-code. |
| `06_build_augmented_datasets.py` | Use OARD citations to add examiner-cited prior art claim 1 text to the JSONL | none (reads local) | OARD csv + JSONL + claim-1 lookup + pg_published_application | augmented `.csv.gz` | Already expects `raw/oard/oard_citations.csv` with columns `app_id, citation_pat_pgpub_id, form892`. Crosswalk (app_id ↔ pgpub_id) is done via `pg_published_application.tsv.zip` from PatentsView, **not** a custom file. |
| `03_parse_labels.py` | Per-app event counts (CTNF/CTFR/etc) from PatEx transactions | none | local PatEx CSVs | labels parquet | Already counts `n_office_actions`. No text. |

### App-number ↔ ifw_number crosswalk
**Already built into OARD itself.** Per Lu et al. §IV, the `office_actions` table has both `app_id` (= application_number) AND `ifw_number` (the file-wrapper key). `rejections` and `citations` are keyed on `ifw_number` and join cleanly to `office_actions` to get `app_id`. No separate crosswalk needed.

For the existing JSONL (which is keyed on `pgpub_id`), the crosswalk to `application_number` is done in `06_build_augmented_datasets.py` via PatentsView's `pg_published_application.tsv.zip` (or BigQuery `patents-public-data.patents.publications`, see script 10). Both already exist.

---

## 2. Auth state (this machine)

| What | State |
|---|---|
| `gcloud` CLI installed | ✓ (`/Users/spangher/Downloads/google-cloud-sdk/bin/gcloud`, v503.0.0) |
| `bq` CLI installed | ✓ (v2.1.10) |
| `gcloud auth list` shows accounts | ✓ (`alexander.spangher@gmail.com` active; `spangher@usc.edu` also present) |
| Tokens valid | **✗** — `invalid_grant` on every API call. Needs `gcloud auth login` + `gcloud auth application-default login`. |
| `GOOGLE_APPLICATION_CREDENTIALS` env var | not set |
| Default project | `usc-research` |
| `~/.config/gcloud/application_default_credentials.json` | exists but stale |

**For BigQuery to work, user must run on this laptop:**
```bash
gcloud auth login                              # browser flow, picks active account
gcloud auth application-default login          # second browser flow, for ADC
gcloud config set project usc-research         # confirm project (or use a different one with BigQuery enabled + billing)
bq query --use_legacy_sql=false 'SELECT 1'     # smoke test
```

For sk3 the user already has a service-account / ADC path (per `reference_openai_key_sk3.md` analog there's an `OPENAI_API_KEY` setup but no record of a GCP key on sk3 — would need to copy the JSON or `gcloud auth application-default login` via SSH tunnel).

| URL liveness | State |
|---|---|
| `bulkdata.uspto.gov` (old OARD host) | **✗ dead** — no DNS A record |
| `developer.uspto.gov/ds-api/oa_actions/v1/records` (legacy DSAPI) | **✗ 503** — decommissioned 2026-05-29 (3 days ago) |
| `data.uspto.gov/` (new ODP) | ✓ 200 — but all data routes are a JS SPA + AWS WAF challenge |
| `api.uspto.gov/` | ✓ 403 (`MissingAuthenticationToken`) — requires `X-API-KEY` |
| `developer-hub.s3.amazonaws.com/bdr-oa-bulkdata/weekly/...` (old weekly S3) | **✗ 403** on direct file access; no bucket listing |
| `www.uspto.gov/sites/default/files/documents/dataset_schema_v20171120.pdf` | ✓ 200 (the 2017 Lu et al. schema PDF — confirmed structured-only) |
| `bigquery.googleapis.com` for `patents-public-data.uspto_oce_office_actions` | ✓ (still public, requires auth) |

---

## 3. What is and isn't in the OARD bulk

Confirmed from the Lu et al. (2017) paper, Tables 3-5 (read directly):

### `office_actions.csv` (~4.4 M rows, one per OA document, key = `ifw_number`)
`ifw_number, app_id, document_cd (CTNF|CTFR), mail_dt, art_unit, uspc_class, uspc_subclass, header_missing, fp_missing, closing_missing, rejection_fp_mismatch, rejection_101, rejection_102, rejection_103, rejection_112, rejection_dp, objection, allowed_claims, cite102_gt1, cite103_gt3, cite103_eq1, cite103_max, signature_type, alice_in, bilski_in, mayo_in, myriad_in`

### `rejections.csv` (~10.1 M rows, one per (document × rejection-grounds) pair, key = `ifw_number` + `action_type` + `action_subtype`)
`ifw_number, action_type (101|102|103|112|dp|obj|allow), action_subtype (e.g. 102(a), 103(a)), claim_numbers (list of rejected claims), alice_in, bilski_in, mayo_in, myriad_in`

### `citations.csv` (~58.9 M rows, one per (application × cited reference) pair, key = `app_id` + `citation_pat_pgpub_id`)
`app_id, citation_pat_pgpub_id (raw reference string), parsed (cleaned patent/pgpub number), form892 (1=PTO-892 examiner cite), form1449 (1=PTO-1449 applicant cite), citation_in_oa (1=cited in the OA text), ifw_number, action_type, action_subtype`

**There is no `text`/`remarks`/`rationale`/`body` column anywhere.** Lu et al. §III explicitly describes the dataset as the *output* of their NLP extraction — the raw OA prose is the input, not stored.

### What can we still do with the bulk OARD?
A lot, actually:
1. **Examiner-cited prior art lists** per application (already wired into `06_build_augmented_datasets.py` → `oard_examiner_cites`). This works for both granted AND abandoned apps, fixing the catastrophic leakage in `g_us_patent_citation.tsv.zip`.
2. **Per-application rejection profile**: how many 102 vs 103 vs 112 rejections, on which claims, in which actions. Useful as structured outcome features and as a coarse "what got the examiner upset" label.
3. **Art unit + examiner signature type** as confound controls.

But the *articulated reasoning* — the prose where the examiner explains *why* claim 7 fails the obviousness test in view of US 9,123,456 col 3 lines 10-25 — is **not** in OARD.

---

## 4. Where the actual examiner text lives

| Source | Coverage | Format | Access | Status |
|---|---|---|---|---|
| **OACT — Office Action Weekly Archives** (USPTO bulk product) | 2020-01-06 → present, all OAs nationwide | weekly ZIPs of JSON files; each JSON has the OA body text + metadata | Was at `developer-hub.s3.amazonaws.com/bdr-oa-bulkdata/weekly/bdr_oa_bulkdata_weekly_YYYY-MM-DD.zip` (now 403); migrated to ODP product `OACT` at `data.uspto.gov/bulkdata/datasets/oact` behind the JS SPA + API key | needs API key + ODP bulk-data API discovery |
| **ODP Office Action Text Retrieval API** (per-app query) | 2008 → present | JSON per OA with full text | Was `POST developer.uspto.gov/ds-api/oa_actions/v1/records` (now 503). Migration target is `api.uspto.gov` Office Actions DSAPI — endpoints not yet live per `patent-dev/uspto-odp` README. Lucene query body: `patentApplicationNumber:12190351` | **broken in transition**; should come back online on `api.uspto.gov/.../oa_actions/...` |
| **Google Patents BigQuery `patents-public-data.uspto_oce_office_actions`** | 2008 – mid-2017 only | structured only (same as bulk) | needs GCP auth | works, but **no text** |
| **USPTO Patent Center / Public PAIR / Global Dossier** (per-app web UI) | full history | PDFs (image-only for older OAs, text-extractable for newer) | scraping / image OCR, very slow | last resort |
| **Frakes & Wasserman / Kuhn OCR'd corpora** (NCSA hosted) | pre-2017 | OCR text | academic, contact authors | useful retroactively |

**Recommended path: OACT bulk via ODP API key.** This is the only legitimately-bulk text source, covers 2020-present (so it intersects most of our `n_office_actions > 0` granted+abandoned tail), and is the path the legacy weekly-zips users were migrated to.

---

## 5. Join strategy with our 500 K-application corpus

The local balanced corpus (`patents_final_outcome_balanced.csv.gz`) is just `(text, judgement)`, no IDs. The rich source on sk3 is `patents_dataset.jsonl.gz` (loaded in `06_build_augmented_datasets.py`) which has `pgpub_id`, `patent_id`, `application_number`, `first_draft_approved`, `final_outcome`, `n_office_actions`.

Coverage estimates:
- Our corpus is filed mostly 2008-2024 (per `03_parse_labels.py` defaults).
- OARD bulk: 2008 – mid-2017 → covers maybe **~35-45%** of our apps that had any OA (older cohorts). Lu et al. §V says OARD has 250-300 K apps per filing year (2010-2016) — comparable to our annual filing volume but limited to pre-2017.
- OACT bulk: 2020-01-06 → present → covers another **~40-50%** of our apps (the newer cohorts). Together with OARD, ≥80% time coverage; the gap is mid-2017 to end-2019.
- Of granted+abandoned apps with `n_office_actions ≥ 1`, expect realistic OARD-or-OACT text match for **~60-75%** after both pulls.

Join keys:
- OARD: `app_id` = `application_number` (no leading zeros, 8 digits) → join directly on `application_number` column in JSONL.
- OACT JSON: per the legacy weekly-zip schema, top-level has `patentApplicationNumber` → same join key.

---

## 6. Step-by-step deployment plan

### Phase A — Structured OARD (always-on, no text)

**Where:** sk3 (data lives at `/lfs/skampere3/0/alexspan/norm-research/datasets/patents/raw/oard`)
**Disk:** ~250 MB compressed + ~3 GB unzipped (citations.csv is large)
**Wall-time:** 15-30 min download + 5 min unzip

1. **Get the bulk CSVs.** Old URL is dead, so re-derive them from BigQuery (one-time):
   ```bash
   # on a machine with GCP auth
   gcloud auth application-default login
   gcloud config set project usc-research
   python scripts/09_query_oard_bigquery.py --output raw/oard/oard_citations.csv
   # also pull the other two tables (the script currently only does citations):
   bq query --use_legacy_sql=false --format=csv --max_rows=10000000 \
     'SELECT * FROM `patents-public-data.uspto_oce_office_actions.office_actions`' \
     > raw/oard/office_actions.csv
   bq query --use_legacy_sql=false --format=csv --max_rows=15000000 \
     'SELECT * FROM `patents-public-data.uspto_oce_office_actions.rejections`' \
     > raw/oard/rejections.csv
   ```
   *Or* find a mirror (some academic groups re-host the 2017 release). Try `archive.org` for `bulkdata.uspto.gov/data/patent/office/actions/bigdata/2017/` before pulling via BigQuery.

2. **Run `06_build_augmented_datasets.py` as-is.** It already consumes `oard_citations.csv` and produces `patents_final_outcome_with_examiner_cites.csv.gz` with the cited-prior-art claim 1 text appended. This already partially answers "what did the examiner think was relevant" without any text body.

### Phase B — Examiner text via OACT bulk (the actual goal)

**Where:** sk3 (large pull)
**Disk:** unknown — back-of-envelope, ~80 KB/OA × ~400 K OAs/year × 6 years × ~30% JSON overhead ≈ **~70-150 GB raw**, likely 15-30 GB zipped.
**Wall-time:** several hours for download + several hours to parse/index. Can be parallelized weekly.

1. **Get an ODP API key.** Free registration at https://data.uspto.gov/apis/getting-started. Pass via `X-API-KEY` header.

2. **Discover the OACT product file list.** Authenticated ODP bulk-data search endpoint (exact URL is behind the SPA; need to either inspect the network panel on `data.uspto.gov/bulkdata/datasets/oact` once or `gh api repos/patent-dev/uspto-odp/contents/...` to see how the Go client paginates). The URL pattern is `https://api.uspto.gov/api/v1/datasets/products/OACT/files` (approximate — confirm against the live API).
   - Expect ~330 weekly files (2020-01-06 to 2026-06-01 ≈ 6.4 years × 52 weeks).

3. **Bulk download.** Each file URL is shaped like `https://api.uspto.gov/api/v1/datasets/products/files/OACT/<fileName>` and likely 302s to a presigned S3 URL. Run `aria2c`/`curl --parallel-max 8 -L` against the file list.

4. **Parse JSON.** Each weekly zip contains many JSON files, one per OA. Schema (per Office Action Retrieval API docs): each record has `patentApplicationNumber`, `documentIdentifier`, `mailDate`, `documentCode` (CTNF/CTFR), and **the OA text body** (likely in a `bodyText` / `documentContent` field — confirm from a sample after pull).

5. **Build a per-application OA-text parquet.** Group by `patentApplicationNumber` → list of `{mail_dt, document_cd, body_text, ifw_number}`. Filter to apps in our 500K corpus first.

6. **Add to JSONL as a `oa_texts` field** and rebuild a third augmented variant: `patents_final_outcome_with_examiner_text.csv.gz` (e.g. first OA body, truncated to 8K chars).

### Phase C — Backfill 2017-2019 gap (deferrable)

The two OA Citations/Rejections beta APIs cover from 2018-06-01 onward (per the Go client source comment), partially closing the gap, but they only return *structured* fields, not text. For 2017-2019 examiner text, options are: (i) Frakes/Wasserman corpus (academic ask), (ii) Patent Center scraping (slow), (iii) accept the gap.

---

## 7. Concrete commands the user can run NOW (laptop)

```bash
# 1. Re-auth GCP (browser flow)
gcloud auth login
gcloud auth application-default login
gcloud config set project usc-research
bq query --use_legacy_sql=false --max_rows=1 'SELECT 1'    # smoke

# 2. Get an ODP API key (one-time, in a browser)
open "https://data.uspto.gov/apis/getting-started"
# save the key to ~/.config/uspto/odp_api_key and export:
export USPTO_API_KEY=$(cat ~/.config/uspto/odp_api_key)

# 3. Smoke-test the ODP API
curl -sIL -H "X-API-KEY: $USPTO_API_KEY" "https://api.uspto.gov/api/v1/datasets/products/OACT" | head
# (expect 200 + JSON; if 403, the product code or URL pattern needs adjustment)
```

---

## 8. Open questions / blockers for the user

1. **GCP project + billing.** `usc-research` is hard-coded everywhere. Confirm the project still exists and the user is still authorized. If not, swap to a project with BigQuery enabled. (BigQuery free tier covers 1 TB/month of queries — the three OARD tables together are ~10 GB scanned, well under the cap.)

2. **ODP API key.** No record of one in env vars or `~/.config/`. User needs to register (free, takes <5 minutes per https://data.uspto.gov/apis/getting-started) and decide whether to store it locally or on sk3.

3. **OACT URL pattern confirmation.** The legacy `developer-hub.s3.amazonaws.com/bdr-oa-bulkdata/weekly/...` returns 403; the new ODP equivalent endpoint isn't documented in any text-fetchable place (data.uspto.gov is a JS SPA). The `patent-dev/uspto-odp` Go client has the search/rejections endpoints wired but no file-download endpoint. **Action:** once the user has an API key, hit `GET https://api.uspto.gov/api/v1/datasets/products` with the key and look for the OACT product code, then follow its `files` link. (~30 min of API exploration.)

4. **Decide priority.** Do we want:
   - (a) just the *examiner-cited prior art* (Phase A only — already mostly done, just needs auth) — modest signal increment over current corpus,
   - (b) full *examiner reasoning text* (Phase A + B — the dataset's marketing claim) — requires API key + ~100 GB pull + parsing infrastructure.

   The original audit claim that the patents corpus needs office action text to be a "norm articulation" dataset implies (b). But (a) is a 2-hour change once auth works; (b) is a 1-2 day project.

5. **2017-2019 coverage gap.** Are we OK with no text coverage for apps with first OA mailed Jul 2017 – Dec 2019? That's ~3 filing-year cohorts.

6. **sk3 GCP auth.** No reference memory entry shows GCP creds on sk3. Easiest path: run Phase A on this laptop (small data), `scp` the CSVs to sk3, do Phase B on sk3 (where the big disk and the existing JSONL live).

---

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| BigQuery `patents-public-data.uspto_oce_office_actions` project sunset | low | high — would lose structured OARD too | Pull all 3 tables now; cache on sk3 |
| ODP `OACT` product not yet live | medium | high — Phase B blocked | Check after API key; if blocked, fall back to legacy S3 with API key, or wait |
| OACT JSON schema undocumented | medium | medium | Pull one week's zip, inspect, then commit parser |
| Text-extraction quality variable (templated form paragraphs vs free prose) | high (per Lu et al. §II) | medium | Same problem Lu et al. solved with NLP; we can either keep raw OR extract per-rejection paragraphs |
| Disk on sk3 for 100+ GB pull | low (sk3 has TB) | low | Confirm `/lfs` free space before pull |
| OA text leaks the label (mentions allowance / abandonment) | medium | high for some tasks | Filter text to *first* OA only; redact post-decision language; restrict to `n_office_actions ≥ 1` cohort (same trick already used in `06_build_augmented_datasets.py --require-oa`) |

---

## 10. Sources

- Lu, Q., Myers, A., Beliveau, S. (2017). *USPTO Patent Prosecution Research Data: Unlocking Office Action Traits*. USPTO Economic Working Paper 2017-10 — read pages 1-19; Tables 3-5 are the definitive schema. https://patentlyo.com/media/2017/11/USPTO-Patent-Prosecution-Research-Data_Unlocking-Office-Action-Traits-1.pdf
- USPTO ODP landing — https://data.uspto.gov/
- USPTO ODP Office Action APIs catalog (legacy, being deprecated) — https://developer.uspto.gov/api-catalog
- USPTO Office Action Weekly Zips (legacy product page) — https://developer.uspto.gov/api-catalog/uspto-office-action-weekly-zips-api
- BigQuery dataset (Google Cloud Marketplace) — https://console.cloud.google.com/marketplace/product/google_patents_public_datasets/uspto-oce-office-actions
- `patent-dev/uspto-odp` Go client (best living reference for ODP API URL patterns) — https://github.com/patent-dev/uspto-odp
- USPTO bulk data directory — https://data.uspto.gov/bulkdata
- BDSS-to-ODP migration mapping (PDF behind SPA; couldn't fetch automatically) — https://data.uspto.gov/documents/documents/BDSS-to-ODP-API-Mapping.pdf
