# Creative Writing

Story-quality prediction from r/WritingPrompts. Part of the norm-research project: a `judgement` label
on (prompt, story) pairs that the verifiability / articulability decomposition (dense reward model
ceiling vs. norm-library AUC vs. taste residual) is run against.

## 1. Task

Given a `(prompt, story)` pair from r/WritingPrompts, predict whether the story would receive a
non-trivial number of upvotes from the subreddit's readers. Operationalized as a binary label
`judgement ∈ {0, 1}` derived from the comment's Reddit `score`.

- **Positive (1)**: well-received reader story (upvoted past a threshold)
- **Negative (0)**: low-upvote reader story on the same prompt distribution

The label is a *crowd-vote* signal, not a curatorial/editorial one. This is its main known weakness
(see open questions below) — but the dataset is large, group-splittable by prompt, and labels are
cheap.

## 2. Sources

Two distinct upstream sources have been used in the project:

### 2a. LitBench (Stanford, 2025) — initial source, now superseded

Reddit-derived preference pairs (chosen/rejected stories) from r/WritingPrompts, released as
`LitBench-Train.csv` / `LitBench-Test.csv` / `LitBench-Rationales.csv`. Local copies live in this
directory (gzipped variants under `LitBench-*.csv.gz`).

Derived modeling file: `litbench-to-train.csv.gz` — 87,654 rows (27,393 positive, 60,261 negative),
binarized with `judgement = 1 if upvotes > 100 else 0` (see `running-research-notes.md` §"Creative
Writing (LitBench)"). Built by `notebooks/2026-02-17__data-processing.ipynb`.

### 2b. Direct r/WritingPrompts scrape (Arctic Shift API) — current canonical source

Built in-house against the Arctic Shift Reddit archive
(`https://arctic-shift.photon-reddit.com/api`) because LitBench inherited the same
upvote-laundering signal but exposed only ~87K pairs; a direct scrape gives ~1M comments + the full
prompt thread for group-aware splitting. Raw artifacts live alongside this README:

- `writingprompts_comments.jsonl.gz` — raw top-level story comments (filtered: bots removed,
  `parent_id` starts with `t3_`, `len(body) ≥ 200`)
- `writingprompts_submissions.jsonl.gz` (a.k.a. `writingprompts_posts.jsonl.gz`) — raw prompt posts

### 2c. Ruled out

LitBench itself is the in-family example. Other sources considered and ruled out for the
articulability-gap line of work are catalogued in
`memory/project_creative_writing_dataset_search.md`:

- LitBench (re-suggest blocker) — same crowd-vote source as our pipeline
- LitBank prizewinners — not fully released, would require email request
- WebNovelBench — Chinese, multidim — not yet tried (deferred)
- Bridport / Commonwealth shortlists — would need fresh scrape

## 3. Collection / preprocessing scripts

Order matters. Scripts marked **canonical** define the pipeline currently feeding modeling.

| Step | Script | Purpose |
|---|---|---|
| 1 | `download_writingprompts.py` | Full Arctic Shift download (comments + posts) with bot filter, retries, resume |
| 1 (alt) | `download_writingprompts_v2.py` | Slimmer backwards-paginating variant; same filter set, no posts |
| 2 | `build_writingprompts_dataset.py` | Joins posts↔comments, exact dedup, `score ≥ 10 → judgement=1`, naive 50K/50K class balance |
| 3 | `build_writingprompts_balanced.py` | **Length-balanced** rebuild: within 7 char-length buckets, balance pos/neg, then scale to 100K target. Avoids the trivial length confound |
| 4 | `../../scripts/dedup_creative_writing.py` | MinHash LSH near-dup pass (char 9-grams, 128 perms, threshold 0.5) over the LitBench-derived corpus (left over from the LitBench-era pipeline) |
| 5 | `../../scripts/analyze_creative_writing_leakage.py` | Audit: prompt-level label-consistency, near-dups, length residuals — feeds the rebuild step |
| 6 | `rebuild_writingprompts_clean.py` | **CANONICAL**. Drops mod-bot removal-template rows, adds `prompt_id` (md5 of prompt text), re-balances per length bucket |
| (aux) | `inspect_writingprompts.py` | Stats / sample dumper used to diagnose length and `[deleted]` skews |
| (aux) | `topic_model_writingprompts.py` | LDA(k=100) over the modeling file → `writingprompts_modeling_with_topics.csv.gz`, used for topic-balanced sampling experiments |
| (aux) | `../../scripts/cw_norm_categorization.py` | Manual RELAX/STRICT categorization of the 368 creative_writing rubric aspects (see §7 v2relax) |

All scripts use absolute sk3 paths (`/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing/`).

## 4. File layout

```
datasets/creative-writing/
├── README.md                                   # this file
├── LitBench-{Train,Test,Rationales}.csv[.gz]   # upstream LitBench drop (Jan 2026)
├── litbench-to-train.csv.gz                    # derived modeling file from LitBench (legacy)
├── build_writingprompts_dataset.py             # step 2
├── build_writingprompts_balanced.py            # step 3
├── download_writingprompts.py                  # step 1 (full)
├── download_writingprompts_v2.py               # step 1 (slim)
├── inspect_writingprompts.py                   # aux
├── rebuild_writingprompts_clean.py             # step 6 (canonical clean)
├── topic_model_writingprompts.py               # aux
└── online-rubrics/                             # scraped craft/judging rubrics for norm-library seeding
    ├── raw/         # ~3,217 raw HTML pages (Booker, AWP, Aristotle's Poetics, AP English, etc.)
    ├── claude-parsed/  # ~221 markdown-parsed rubric distillations
    ├── gpt-parsed/
    └── waveh{3,4,5,6}_{log,seen}.{csv,txt}     # crawler state files
```

The large data products (`writingprompts_*.jsonl.gz`, `writingprompts_modeling*.csv.gz`) live on sk3
under `/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing/` — not in the repo.

## 5. Canonical dataset file

Per `memory/reference_clean_datasets_per_task.md`:

- **Path**: `writingprompts_modeling_clean.csv.gz` (sk3-only; 135 MB, May 12)
- **Shape**: 96,080 rows, 70,453 unique prompts
- **Columns**: `text` (formatted `"PROMPT: ...\n\nSTORY: ..."`), `judgement ∈ {0,1}`, `prompt_id` (md5)
- **Group key**: `prompt_id` — required for any train/val/test split (avoids the prompt-level leakage
  surfaced by `analyze_creative_writing_leakage.py`)

This is the v2 task dataset per `memory/reference_v2_task_datasets.md` (`judgement` is the binary
column expected by the cells-DB pipeline).

The legacy LitBench-derived `litbench-to-train.csv.gz` is still on disk; sweeps under
`runs/litbench_8b/` use it. Newer sweeps use the clean v2 file.

## 6. Modeling state

### 6a. Dense reward model sweeps (Llama-3.1-8B)

From `memory/project_dense_model_sweeps.md` — data-scaling sweep on the LitBench-era file
(`runs/creative_writing_sweep_llama8b/subset_<frac>/trial_<N>/`, full train = 70,122 rows):

| Subset | Rows | #Trials | med AUC | max AUC |
|---|---:|---:|---:|---:|
| 0.1 | ~7K | 5 | 0.641 | 0.652 |
| 0.5 | ~35K | 5 | 0.627 | 0.762 |
| 0.8 | ~56K | 5 | 0.857 | 0.882 |
| 1.0 | ~70K | 5 | 0.868 | 0.904 |

**Still climbing at 1.0** — unlike press_release / peer_review which saturate, creative_writing has
not flattened. Variance is high (some trials fail). Cleaner reruns live under
`runs/creative_writing_clean_sweep_llama8b/` and `runs/creative_writing_groupsplit_sweep_llama8b/`.

### 6b. articulation_STaR — v1 / v2

End-to-end STaR loop training Llama-3.1-8B + LoRA to *articulate* the norms behind reader
preference, with a Qwen-3.5-122B-FP8 strong judge and a small weak judge for contrastive filtering.
Outputs under `outputs/articulation_star/creative_writing/`.

- **v1_overnight_logprob** (2026-05-30, complete): 3 STaR iters, logprob-scored contrastive filter,
  4-way DP gen, balanced 1500/label/iter. Train loss 0.76 → 0.64 → 0.50; format compliance ≈ 98%;
  test acc base 48% → iter02 56.2%. See `memory/project_articulation_star_overnight_run.md`.
- **v2_weak1b_logprob** (2026-05-31): switched weak judge to Llama-3.2-1B + baked-in leakage
  detection (auto + LLM-judged). Iter 0 died at combine because 1B-at-logprob hit 68% acc — weak
  exceeded strong, breaking the contrastive filter. Run was deliberately not retried; the failure
  *was* the signal. Leakage analysis on v1: specificity 2× up, sentiment vocab down, composite
  leakage +6% (template growth). See `memory/project_articulation_star_v2_run.md`.

### 6c. v2relax re-scoring & cw_pump (norm-library scoring)

Norm-library scoring of (artifact, aspect) cells with Qwen-3.5-122B-FP8 judge, feeding the
verifiability / RF analyses. Two June 2026 milestones:

- **cw_pump (2026-06-01)** grew the cells DB from 670 → 2,611 unique scored datapoints (Phase A
  retried existing prompts no-think; Phase B scored 5K new datapoints). See
  `memory/project_cw_pump_2026_06_01.md`.
- **v2relax (2026-06-01)** re-scored with fixed cross-task system prompt + relaxed-applicability
  interpretation + manual RELAX/STRICT/BORDERLINE categorization of all 368 aspects (252/108/8).
  First-pass RF AUC: legacy `qwen_thinking_fp8` 0.541 → `qwen_relaxed_v2_2026_06_01` 0.636 (Δ +0.095);
  union 0.643. See `memory/project_cw_relaxed_appl_v2_2026_06_01.md`.

## 7. Key decisions

- **Binarization**: two regimes coexist.
  - LitBench era: `judgement = 1 if upvotes > 100 else 0` (yields ~27K/60K).
  - Direct-scrape era: `judgement = 1 if score >= 10 else 0` (`build_writingprompts_dataset.py`,
    `build_writingprompts_balanced.py`). The lower threshold is appropriate because the direct
    scrape pulls a wider distribution of comments, not just curated pairs.
- **Length confound**: an early inspection (`inspect_writingprompts.py`) showed positive stories were
  systematically longer. Fixed by per-length-bucket balancing
  (`build_writingprompts_balanced.py`, buckets `[0, 500, 1K, 2K, 3K, 5K, 10K, ∞)` chars) and
  re-applied after the mod-template drop in `rebuild_writingprompts_clean.py`.
- **Mod-bot template leak**: many label-0 rows were the "this submission has been removed"
  moderator template, not real stories. `rebuild_writingprompts_clean.py` drops them via regex
  before re-balancing.
- **Group-splittable by prompt**: `prompt_id` is mandatory for splits. Multiple stories per prompt
  exist; mixing them across train/val/test leaks the prompt's typical-quality prior.
- **Dataset pivot LitBench → balanced WritingPrompts**: LitBench's preference-pair packaging was
  convenient but exposed the same Reddit-vote signal at a smaller scale and without group-split
  metadata; the in-house Arctic Shift scrape gives ~10× more raw comments and lets us reconstruct
  per-prompt grouping.
- **Online-rubrics corpus**: `online-rubrics/` carries 3K+ scraped craft / judging rubrics
  (Aristotle, Booker shortlisting criteria, AWP hallmarks, AP English rubric, MFA writers'
  manifestos, etc.) parsed into markdown, used as seed material for the norm library that scores
  this task. The rubric-to-task framing bug that plagued v2relax (legacy judge prompts all started
  with peer-review boilerplate) is documented in
  `memory/feedback_judge_prompt_cross_task_bug.md`.

## 8. Open questions / next steps

- **articulation_STaR v3**: design from `memory/project_articulation_star_v2_run.md` §"Proposed
  next experiments" — CoT-strong + logprob-weak contrastive (let Qwen-122B think before answering
  to recreate the strong-weak gap that v1's 3B and v2's 1B both failed at), plus a dedicated
  distilled-classifier leakage probe replacing the LLM-judged proxy.
- **The 0.5 problem under v2relax**: with relaxed applicability, ~68.7% of applicable cells get
  judge score 0.5 ("borderline"). Strategies to test (filter aspects by >K% 0.5 rate,
  confidence-weight features, treat 0.5 as missing, one-hot the three score levels, MI-based aspect
  pre-selection) are catalogued in `memory/project_judge_0p5_noise_filtering.md`.
- **Dense ceiling**: the 8B sweep still climbs at 70K rows; the next sweep should be on the clean
  96K-row file (`writingprompts_modeling_clean.csv.gz`) with `prompt_id` group splits to get a clean
  upper bound for the V/A/T decomposition.
- **Source diversity**: the project is bottlenecked on crowd-vote framing. Bridport / Commonwealth
  shortlists, WebNovelBench, or a small in-house expert-rated set would test whether the same
  norm library transfers to editor-curated quality.
- **Re-scoring other tasks** with the v2relax fix is still pending (peer_review, code_review,
  press_releases, etc.) — the per-task RELAX/STRICT categorization is the rate-limiting step.
