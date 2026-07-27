# Rubric variance analysis — research plan & experimental steps

*Started 2026-05-11. Living document — update as we lock in decisions.*

## Current state of the corpus

- **38,303 source pages** extracted (raw HTML/PDF + claude-curated `.md`), distributed across **11 tasks** (1,100–1,900 per task — well-balanced)
- **361,050 extracted rubrics** (avg ~9.4 per page) in GPT-5-mini structured output
- **Schema per page**: orientation, intended_audience, subtask_short, subtask_description, subtask_keywords, subtask_breadth, error, rubrics_metrics[]
- **Schema per rubric**: name, description, guidance
- **`rej_` filename prefix** flags ~7,000+ negative-example pages (retractions, sanctions, bombs, failures) collected explicitly to surface implicit norms
- **`manual_` prefix** for canonical books pulled by hand (Galtung-Ruge, McKee, Bernays, etc.) — full books needing chunked extraction
- **Wayback / edition history** for tracking norm drift over time on canonical URLs

## Goal hierarchy (from user discussion)

### 1. Task-level rubrics

**1a. Canonical-rubric count via cross-encoder linkage.**
Train a cross-encoder to decide "are these the same rubric?" then cluster.
- Flatten to atomic rubric tuples: `(task, source_file, source_orientation, rubric_name, rubric_description, rubric_guidance, subtask_short, subtask_breadth)` — ~300K tuples
- First-pass embed (text-embedding-3-large on `name + description`); near-duplicate dedup at cosine ≥ 0.9
- Fine-tune cross-encoder for "same rubric?": weak positives from same-page co-occurrence + name-overlap; negatives from rubric pairs across distant clusters
- Cluster connected components above cross-encoder threshold
- **Output**: per-task canonical rubric count; rubric-density (clusters per page)

**Open decision**: what counts as "same"?
- Same operationalization (same value assigned)?
- Same underlying construct (different operationalizations of one idea)?
- Same surface wording?
- → Reasonable to compute all three and report.

**1b. Variance metrics complementary to the cluster count.**
Report a battery:
- **Embedding dispersion**: mean pairwise distance of rubric embeddings; convex-hull volume; top-K covariance eigenvalues (direction of spread, not just magnitude)
- **Cluster-size entropy**: Shannon entropy of cluster-size distribution. Flat → many idiosyncratic rubrics; peaked → a few canonical with long tail
- **Zipf slope**: fit rank-frequency; slope tells you concentration of canonical rubrics
- **Inter-source-pair Jaccard**: random pairs of pages, compute rubric-cluster overlap. Average Jaccard = inter-source agreement
- **Edit-distance graph diameter**: longest shortest path in rubric graph (how far apart are the most-different rubrics)

Report all five alongside cluster count.

**1c. Distribution over conceptual scales.**
The originally-proposed scales (stylistic/methodological/articulable/vibes/verifiable/thin/thick) overlap heavily. Collapse to **3 orthogonal axes**:

- **Articulability** (Daston/Polanyi): bright-line rule → operational procedure → expert-judgment → ineffable "vibes"
- **Verifiability**: mechanically checkable → reasoned-from-evidence → defensible-judgment → unverifiable
- **Surface-vs-substance**: rule about form ("Oxford comma") → rule about content ("thesis must be falsifiable")

**Method**: second GPT pass over each extracted rubric, classifying on three 4-point scales. Cheap (~$10–20 across 300K rubrics on gpt-5-mini). Adds 3 numeric columns to each tuple.

**1d. Input variables + thin/thick of each.**
Per rubric, classifier also extracts:
- List of input feature(s) the rubric depends on (free-text)
- Per-input binary thin/thick (`word count` → thin; `narrative arc` → thick)
Gives a per-input articulability map — richer than per-rubric.

**1e. Tacit-knowledge required.**
Hardest dimension. Three proxies, combine into a score:
- **Guidance/description ratio**: tacit rubrics need more surrounding explanation. `len(guidance) / len(description)`; tail is the tacit zone
- **Reading-level / jargon density**: vocabulary outside top-10K English words per rubric
- **LLM probe**: "could a literate non-expert apply this reliably without training? rate 1–5"

Combine. Testable hypothesis: canonical rubrics (high cluster recurrence) should score MORE articulable / LESS tacit.

### 2. Task-level descriptions (control variables)

**2a. Variance in `subtask_description` / `subtask_keywords` per task.**
Same pipeline as 1a/1b but on subtask fields:
- Embed all `subtask_description` strings per parent-task
- Compute dispersion + cluster → **subtask-variance(task)**
- Control variable: when comparing rubric-variance across tasks, regress out subtask-variance, OR compute within-subtask-cluster rubric variance for apples-to-apples
- ANOVA-style: total rubric variance = within-subtask + between-subtask. Report F-ratio per task.

**Hunch (worth verifying)**: most tasks will show high subtask variance (creative-writing spans horror/romance/MFA/MG/picture-book; patents spans software/biotech/design/SEP; legal spans M&A/trial/appellate/admin). The `subtask_breadth` field gives a quick filter — drop `very_broad` pages when you want apples-to-apples comparison.

**2b. (User's forgotten point — possibilities to consider)**
Candidates worth checking with user:
- Control for **audience variance** too (`intended_audience` field). A "novice" rubric for the same subtask differs from an "expert" rubric on the tacit-knowledge dimension.
- **Source coverage / canonicity** — how many of the same rubrics appear under multiple `subtask_description`s? Cross-cutting rubrics = deeper norms.
- **Subtask-rubric ratio** — does a task with many subtasks inherit generic rubrics, or do subtasks have genuinely distinct rubrics?

### 3. Metric differences along dimensions

**3a. Across DOMAIN (source type)**: academic vs professional standard vs blog vs informal.
We have `orientation` field (research_article/blog_post/professional_standard/etc.) and source-type signals in filename prefix (`raw_*` vs `claude-parsed_*`) and source domain. For each rubric cluster, compute:
- Source-type distribution `Pr(rubric | source_type)`
- Authority-weighted incidence (canonical rubrics should appear more in formal_guideline > blog_post)
- For each source type, the **distinctive rubric set** (rubrics with highest TF-IDF for that type)

**3b. Across TIME (norm drift)**.
Rich temporal signal from Hist1/Hist2/Hist4/Hist5/Hist6 waves + Wayback snapshots. Pipeline:
1. Add a `year` column per page (via filename pattern, URL pattern, content date, or small LLM probe pass).
2. For each rubric cluster, track presence-by-decade.
3. Compute per-cluster: **emergence year** (first decade rubric appears), **dominance year** (peak), **stability score** (% of decades present after emergence).
4. New rubrics by decade: which clusters first appeared 2010s vs 1990s vs 1900s?
5. Disappearing rubrics: clusters with high pre-2000 incidence and low post-2010.

Cleanest signal for tasks with strong edition series: **news** (AP Stylebook annual), **legal** (Bluebook 1942–2025, Restatements), **patents** (MPEP editions), **peer-review** (ICLR 2018–2024). Weakest for humor and code-review (sparser historical sources).

## Authority weighting (separate from variance but should be applied first)

Compute a per-page authority score combining three orthogonal signals:

**(A) Source-tier weight (from URL + domain):**
| Tier | Weight | Examples |
|---|---:|---|
| Official body | 1.00 | uspto.gov MPEP, grants.nih.gov, regulations.gov, COPE, PRSA |
| Peer-reviewed academic | 0.95 | doi.org, plos.org, link.springer.com, arxiv.org, jstor.org |
| University course / handbook | 0.85 | *.edu syllabi, FJC Judicial Writing Manual |
| Established prof org / foundation | 0.80 | AMS, MAA, AAAS, CIPR, ABA |
| Recognized expert practitioner blog | 0.60 | Patently-O, Tao's blog, Lethain, Garner's LawProse |
| Generic how-to / wikihow / mass-market | 0.40 | wikihow.com, generic Medium/Quora |
| Marketing / landing / paywall stub | 0.10 | product pages, abstract-only, "just-a-moment" |

**(B) Content-quality weight (from GPT extractor output):**
- `orientation` mapping: `formal_guideline` 1.0, `professional_standard` 0.95, `research_article` 0.95, `course_syllabus` 0.85, `tutorial`/`how_to` 0.7, `blog_post` 0.5, `error` 0.0
- Bonus by `n_rubrics_extracted` — diminishing returns (e.g. `min(n/10, 1.0)`)
- Penalty if avg `guidance` length < 50 chars (shallow)

**(C) Independent-discovery weight:**
- Count of distinct WebSearch queries that returned this URL. URLs surfaced by 5+ different queries are canonical → boost.
- Cross-references in our other curated `.md` files (citation graph).

**Final score** = `tier × content × log(1 + independent_discoveries)`. Exposed as a column in `pages_df` and `rubrics_df` so downstream code can filter to e.g. top-quartile only.

## Suggested execution order

1. **Build the rubric tuples table** (~1 hour Python) → CSV with one row per rubric, ~300K rows. *Status: started in `notebooks/explore_rubrics.py` via `rubrics_df`.*
2. **Add `year` column** via small extraction pass (~$5–10 LLM cost).
3. **Add authority score** column via deterministic rules + cross-reference graph (~$0).
4. **Embed + cluster** (~30 min CPU; ~10 min GPU) → adds `cluster_id` column. Use cosine 0.9 dedup first, then HDBSCAN on residuals.
5. **Second LLM classification pass** for articulability / verifiability / surface-vs-substance / tacit-knowledge ratings (~$30–50).
6. **Compute variance metrics** (1b + 2a) — pure numpy, fast.
7. **Cross-encoder fine-tune** for canonical linking (optional, only if first-pass clustering is too noisy) — needs a few hundred hand-labeled pairs.

Whole pipeline: within a day's compute + $50–100 LLM cost.

## Pipeline status / dependencies

| Component | Status | Path |
|---|---|---|
| Source corpus (38K pages) | ✅ done | `datasets/<task>/online-rubrics/raw/` + `claude-parsed/` |
| Rejection sub-corpus (~7K pages, `rej_*` prefix) | ✅ done | same dirs, filename-tagged |
| GPT-5-mini extraction (361K rubrics) | ✅ done | `datasets/<task>/online-rubrics/gpt-parsed/gpt-5-mini/` |
| Chunked extractor (for full books) | ✅ done | `scripts/chunked_extract.py` |
| OCR fallback (image-only PDFs) | ✅ done | `pdf_to_text()` in `scripts/extract_rubric_features.py` |
| Truncation backfill (4,544 files w/ ≥14K tokens) | 🟡 running | `scripts/chunked_extract.py --skip-head-tail` |
| Manual canonical books (38 PDFs) chunked extraction | 🟡 running | `scripts/chunked_extract.py --all --pattern 'manual_*' --recurse` |
| Exploration notebook | ✅ done | `notebooks/2026-05-11__explore-online-rubrics.ipynb` |
| Rubric tuples table | 🟡 partial (in notebook) | `notebooks/_explore_cache/rubrics.parquet` |
| Year column | ⛔ todo | — |
| Authority score | ⛔ todo | — |
| Cluster column | ⛔ todo | — |
| Articulability/verifiability classification | ⛔ todo | — |
| Variance metrics | ⛔ todo | — |
| Cross-encoder linkage | ⛔ optional, todo | — |

## Outstanding canonical-source gaps (paywall-blocked)

Mostly closed via user's manual downloads on 2026-05-11. Still missing or only-partial:
- **Sigal *Reporters and Officials*** (1973) — only IA `.lcpl` (DRM)
- **Mancosu *Philosophy of Mathematical Practice*** (full version) — partial via Academia.edu chapters
- **Garner *Winning Brief*** — only "tips list" (99KB) from pdfcoffee.com; full book needs Stanford library
- **Faber *Mechanics of Patent Claim Drafting*** (Landis 1990 edition is the best available — newer editions paywalled)

For these: Stanford library proxy + Thorium Reader (for `.lcpl`) are the remaining paths.

## Key research questions the variance analysis should answer

(For the paper.)

1. **Within-task rubric concentration**: do tasks have a small set of canonical rubrics + long tail, or are rubrics largely idiosyncratic? (Zipf slope per task.)
2. **Cross-source agreement**: when two independent expert sources (one academic, one practitioner blog) describe the same evaluative task, how often do they articulate the same rubric? (Jaccard distribution.)
3. **Articulability spectrum**: what fraction of rubrics in each task are bright-line-rule vs operational vs judgment vs ineffable? Is this fraction stable across tasks or task-dependent?
4. **Subtask-induced variance**: when we control for subtask, does the residual rubric variance shrink dramatically (rubrics are largely subtask-specific) or only modestly (rubrics generalize across subtasks within a task)?
5. **Norm drift**: which rubrics have emerged in the last 20 years that didn't exist before? Which old ones have disappeared? Are there "stable cores" that persist across all decades?
6. **Authority correlation**: do high-authority sources articulate fewer, more concentrated rubrics, while low-authority sources produce more idiosyncratic ones? Or do all source types converge on the same canonical set?
7. **Negative-example signal**: do the `rej_*` rejection corpus rubrics (failures, sanctions, retractions) cluster around the same canonical rubrics as the positive corpus, but in negation? Or do they reveal additional norms not visible from positive-example sources?

## Tooling investments worth making

- A small **`build_rubric_table.py`** that flattens the GPT outputs once into a clean `rubrics_master.parquet` with all derived columns added — re-runnable as new chunks land.
- A **`classify_rubrics.py`** for the 3-axis LLM classification pass (articulability/verifiability/surface). Async with checkpointing.
- A **`cluster_rubrics.py`** that does the embed → dedup → cluster → fine-tune-link pipeline.
- A **`variance_report.py`** that computes the battery of metrics (1b) and outputs a per-task table + plots.

These four scripts, plus the existing extraction infrastructure, give us the full pipeline.


## References (auto-verified BibTeX, 2026-06-15)

> Extracted from this document and web-verified + independently audited by an automated fact-check pass (search → fetch → resolvable id; attributed claim checked against the located paper). 12 entries. Real located works; not hand-checked. See "needs manual review" for 0 contradicted-claim and 0 unlocatable/rejected items.

```bibtex
@misc{ali_restatements,
  author       = {{American Law Institute}},
  title        = {Restatement of the Law (series)},
  howpublished = {\url{https://www.ali.org/publications/}},
  year         = {1923},
  note         = {Series published 1923--present by the American Law Institute (ALI), founded 1923}
}

@book{apstylebook2024,
  author    = {{The Associated Press}},
  title     = {The Associated Press Stylebook 2024-2026},
  publisher = {Basic Books},
  year      = {2024},
  isbn      = {9781541605114}
}

@book{bernays1928propaganda,
  title={Propaganda},
  author={Bernays, Edward L.},
  year={1928},
  publisher={Horace Liveright},
  note={Reprint: Ig Publishing, 2004, isbn 9780970312594, intro by Mark Crispin Miller}
}

@book{bluebook2025,
  author    = {{Columbia Law Review} and {Harvard Law Review} and {University of Pennsylvania Law Review} and {Yale Law Journal}},
  title     = {The Bluebook: A Uniform System of Citation},
  edition   = {22nd},
  publisher = {Harvard Law Review Association},
  year      = {2025},
  url       = {https://www.legalbluebook.com/}
}

@book{faber1990landis,
  author    = {Faber, Robert C. and Landis, John L.},
  title     = {Landis on Mechanics of Patent Claim Drafting},
  publisher = {Practising Law Institute},
  year      = {1990},
  isbn      = {9780872240070}
}

@article{galtung1965structure,
  title={The Structure of Foreign News: The Presentation of the Congo, Cuba and Cyprus Crises in Four Norwegian Newspapers},
  author={Galtung, Johan and Ruge, Mari Holmboe},
  journal={Journal of Peace Research},
  volume={2},
  number={1},
  pages={64--91},
  year={1965},
  doi={10.1177/002234336500200104}
}

@book{garner1999winningbrief,
  author    = {Garner, Bryan A.},
  title     = {The Winning Brief: 100 Tips for Persuasive Briefing in Trial and Appellate Courts},
  publisher = {Oxford University Press},
  year      = {1999},
  isbn      = {9780195128086}
}

@book{mancosu2008philosophy,
  editor    = {Mancosu, Paolo},
  title     = {The Philosophy of Mathematical Practice},
  publisher = {Oxford University Press},
  year      = {2008},
  isbn      = {9780199296453}
}

@book{mckee1997story,
  title={Story: Substance, Structure, Style, and the Principles of Screenwriting},
  author={McKee, Robert},
  year={1997},
  publisher={ReganBooks},
  isbn={9780060391683}
}

@manual{mpep2024,
  author       = {{United States Patent and Trademark Office}},
  title        = {Manual of Patent Examining Procedure (MPEP)},
  edition      = {Ninth Edition, Revision 01.2024},
  organization = {U.S. Patent and Trademark Office},
  year         = {2024},
  url          = {https://www.uspto.gov/web/offices/pac/mpep/index.html}
}

@book{polanyi1966tacit,
  title={The Tacit Dimension},
  author={Polanyi, Michael},
  year={1966},
  publisher={Doubleday/Anchor}
}

@book{daston2007objectivity,
  title={Objectivity},
  author={Daston, Lorraine and Galison, Peter},
  year={2007},
  publisher={Zone Books},
  address={New York},
  isbn={9781890951788}
}

@book{sigal1973reporters,
  title={Reporters and Officials: The Organization and Politics of Newsmaking},
  author={Sigal, Leon V.},
  year={1973},
  publisher={D. C. Heath},
  isbn={9780669850352}
}

```

### Citations needing manual review

**Partial claim-match (3)** — spot-check exact numbers/wording:

- `apstylebook2024`; `bluebook2025`; `galtung1965structure`

