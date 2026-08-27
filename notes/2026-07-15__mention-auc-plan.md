# Mention-AUC: MI vs per-text silver-label discrimination — PLAN

**Date:** 2026-07-15
**Status:** design confirmed; NOT YET BUILT (this is the planned "Leg B mention-AUC" referenced in
`memory/project_mi_vs_silver_norms.md` and `project_silver_matching_audit_v2.md` — "validity case shifts
to mention-AUC"). It has never been implemented.

## 1. The experiment (user's specification, verbatim intent)

For a task with texts `T` (a creative-writing story, a joke, a PR, …) and bank metrics `M` (R2):

- A text gets human comments. Our extraction pass pulls a norm phrase from a comment
  ("I loved the plot structure here") and the **silver-matching** pipeline maps it to an R2 metric
  (`plot_structure`).
- **Silver label** `y[T, M] = f(T, M) ∈ {0,1}` = did the human comments on text `T` invoke metric `M`.
  (1 if any extracted-and-matched norm from `T`'s comments points to `M`.)
- **Certified prediction** `p[T, M]` = the likelihood the metric applies to `T`, produced by the
  **certified metric prompt** (the certified LLM judge for `M`) applied to text `T`.
- **Per-metric performance:** `AUC_M = AUC( p[:, M], y[:, M] )` over all texts `T`
  (does the certified judge fire on the texts where humans invoked `M`?).
- **The result we want:** `Spearman( AUC_M , MI(M) )` across metrics — i.e., do higher-MI metrics
  have judges that better predict where humans invoke them?

This is a per-text, per-metric **discrimination-of-the-silver-label** correlation. It is NOT a
frequency/salience correlation and NOT an outcome-label correlation.

## 2. Confirmation against the current pipeline

| Ingredient | Needed | Exists today? |
|---|---|---|
| Bank metrics `M` (R2) + names/rubrics | yes | ✅ `silver_match_v3` banks (285/371/221/133/141) |
| Label-free MI per metric `MI(M)` = OPT_Ω bits | yes | ✅ `analysis_inputs/mi_certificates/<task>.json` |
| Certified metric prompt for each `M` | yes | ✅ certified rubrics (GEPA/certified-unit); the m_bar prompt form is `alpha_probe._YESNO_TEXTFIRST` |
| Per-text silver labels `y[T,M]` | yes | ✅ **READY** — `data/silver_match_v3_20260712_faithful/norms/<corpus>.jsonl` carries `norm_uid` + `source_id` for EVERY task; join to the metric matches gives norm→metric→source. Humor: canonical release joins 100% by norm_uid (77,378). CW: v2 matches join 100% by norm text (2,877). |
| Certified score `p[T,M]` on the **silver-source texts** | yes | ❌ **must be generated** — a scoring run of the certified metric prompts over the source texts (recovered by `source_id`). Existing `mbar`/`mbar2` panels scored the WRONG corpus (the `*_modeling` outcome set), so they are not reusable. |

### CONFIRMED DATA (2026-07-15)
Per-task silver norms WITH source linkage exist for all tasks:
`data/silver_match_v3_20260712_faithful/norms/<corpus>.jsonl` — fields include `norm`, `norm_uid`,
`source_id`, `context`, `task`. Norms-per-source: humor med 3 / code-review med 11 / CW med 1 (max 19) /
PR med 1 (max 13) / math med 1 (max 5). Counts: humor 77,378 norms / 19,363 sources; code-review 572,639 /
50,151; CW 4,929 / 2,431; PR 26,996 / 15,067; math 8,238 / 6,492.
Metric matches join to these norms: humor by norm_uid (100%), CW by norm text (100%). So `y[text,metric]`
is buildable NOW for every task. **The user's data does exist; only `p` remains to be scored.**

### The remaining mismatch (why existing panels can't give `p`)
Every task's OSL/panel corpus is a separate `*_modeling` **outcome** dataset with a `judgement`
(upvote/accept) label, NOT the silver-norm source texts:
- humor → `datasets/humor/reddit_humor_modeling_dedup.csv.gz` (jokes + upvotes)
- creative-writing → `datasets/creative-writing/writingprompts_modeling_clean.csv.gz` (stories + upvotes)
- math → `math_se_modeling.csv.gz`; code-review → `code_review_dense_4096tok.csv.gz`;
  press-releases → `press_release_deconfounded.parquet`; etc.

The `m_bar`/`mbar2` panels compute `p` **on these modeling texts**, whose `judgement` label is the
**outcome** (upvote/merge), not "which metric a commenter invoked." So the existing panels cannot be
reused for mention-AUC — they score the wrong texts against the wrong label.

**Feasibility varies by task:** creative-writing is the best first target because the modeling corpus
(WritingPrompts stories) and the silver norms (WritingPrompts comments) plausibly share the SAME stories,
so `y` and `p` can live on the same text set. Humor is worst (modeling = r/Jokes upvotes; silver =
standup-critique comments = different texts).

## 3. Build steps

1. **Establish the norm→source-text join.** For a task, get, per extracted norm: (a) its matched metric
   (from `decisions_<task>.jsonl`), (b) its **source text id** (the story/joke/PR it comments on). Verify
   the source-text ids exist in the silver extraction provenance (humor has `source_id`; others need
   checking — matches_joined currently exposes only a per-norm `doc`, so the parent-text id must be pulled
   from the extraction/signals layer). **This is the first gate; if the link is missing, it must be
   recovered before anything else.**
2. **Build `y[T, M]`.** For each source text `T`, the set of metrics its comments invoked; expand to a
   binary text×metric matrix over the metrics that have MI certs.
3. **Choose the text set.** The texts `T` that have silver labels (comments). Confirm they are real
   evaluable texts (story/joke body), not just comment snippets.
4. **Score `p[T, M]`** = run each metric's certified prompt (YES/NO or soft prob) over the SAME texts `T`.
   This is a NEW LLM scoring run (offline batch vLLM), one pass per executor; start with one strong
   certified judge (frontier or the certified 8B), soft score via multi-form fraction for AUC.
5. **Per-metric mention-AUC** `AUC_M = AUC(p[:,M], y[:,M])`; require ≥ (say) 10 positive texts per metric.
6. **Correlate** `Spearman(AUC_M, MI(M))` + permutation p + partial|description-length; report per task.
   Guard: base rate of `y[:,M]` (invocation is sparse), and a random-metric-label null.
7. **Repeat** across tasks where the text join exists.

## 4. What we already have that is REUSABLE
- MI certs (all tasks). ✅
- The certified prompt form + executor infra (`glm_mbar_probe.py` / `alpha_probe`) — reusable to score
  `p` on the correct texts (step 4). ✅
- The silver matches `decisions_<task>.jsonl` (norm→metric). ✅ (need to add norm→source-text.)

## 5. Corrections to prior descriptions (things I ran that DIFFER from this view — struck through)

See the cross-outs applied in:
- `outputs/silver_match_v3/MI_ITEM_DISCRIMINATION_DOSE_RESPONSE_20260715.json`
- `outputs/silver_match_v3/FINAL_R2units_R3families_20260715.json`
- `memory/project_mi_vs_silver_norms.md`, `memory/project_silver_matching_audit_v2.md`

Summary of the two prior analyses and why they DIFFER from mention-AUC:

1. ~~**Salience correlation** (leaf/family, MI vs how many norms across the corpus matched the metric).~~
   DIFFERS: this is a corpus-level **frequency** of invocation, aggregated over all texts — there is no
   per-text `p`, no per-text `y`, and no AUC. It answers "are high-MI metrics invoked more *often*,"
   not "does the certified judge discriminate the texts where a metric is invoked." Keep as a separate,
   weaker channel; do NOT call it the silver-label validation.

2. ~~**Item/outcome dose-response** (MI vs |AUC−0.5| where AUC = metric judgments vs the item's
   `judgement`/upvote label).~~ DIFFERS on BOTH axes: (a) `y` is the **outcome** (upvote/merge), not the
   silver mention `f(T,M)`; (b) the texts are the `*_modeling` corpus, not the silver-norm source texts.
   The humor +0.47 is a real outcome-prediction correlation but it is NOT the mention-AUC and must not be
   reported as the silver-label result.

**The mention-AUC (this plan) is the correct silver-label validation and has not yet been computed.**

## 6. Data readiness inventory (2026-07-15)

24 norm corpora with `source_id` exist under `data/silver_match_v3_20260712_faithful/norms/`. Source-text
availability for the 7 tasks that have MI certs:
- **READY end-to-end:** creative-writing (story text via `wp_comments/input.jsonl`, `unit_id`→story),
  code-review (2636/2636 PRs join the modeling corpus by `paper_id`).
- **Text needs id-normalization:** peer-review (`iclr_<id>_r0` wrapper), press-releases (`pair_X`).
- **Text NOT directly joinable (needs raw source):** humor (standup), math-stackexchange, notice-and-comment.

`y` already built for creative-writing at story level: 1,040 labeled stories, 51 metrics ≥10 positives
(`/tmp/cw_story_y.json`, `/tmp/cw_story_texts.jsonl` on sk3).

## 7. MI source (v14 vs old cert)

- v13/v14 MI = R3 level (`<task>_R3_metric<N>`), `achieved_value` bits, channels {mcq, behavioral}. v14
  coverage: creative-writing 50, peer-review 8, math 4, humor 4, code-review 2. v14 behavioral lands overnight.
- **Old opt cert (R2, 285/task) vs v13 achieved_value, rolled R2→R3 (n=25): behavioral ρ=+0.65 (strong),
  MCQ ρ=+0.18 (weak).** → old cert is a usable large-n proxy for the behavioral channel only.
- **Decision:** wire mention-AUC for BOTH MI sources; report behavioral primarily.

## 8b. PROMPT-VARIANT EXTENSION (user 2026-07-15 — "the end-all-be-all"): validate MI as a METRIC

The strongest version of this experiment varies the PROMPT within a metric: prompts p, p′, p″ for the same
metric each induce a different measurement M′, M″, M‴, each with its OWN pipeline MI — and each gets its own
AUC(M, y). Correlating MI with AUC **within metric across prompt versions** (or pooled with metric fixed
effects) removes all metric-identity confounds and directly validates MI as a measurement-quality metric.
This is THE target readout.

Infrastructure check: the pipeline already supports this —
- the OSL panels carry `per_form` arrays (4 prompt forms per metric already scored on probes);
- the old cert's sigs carry per-form M_i and form-orbit variance (var_phi); OPT_Ω is the sup over forms,
  individual forms have their own achieved MI;
- v13/v14 report `achieved_value` vs `exact_structural_cap` (per arm/instrument = prompt-version-like variants).
Plan: phase 1 = per-metric mention-AUC with the certified (best) prompt (running now); phase 2 = score the
per-form prompt variants of each metric on the same texts → within-metric MI-vs-AUC dose-response.

## 9. Corpus recovery plan (recompute where needed) — LAUNCHED 2026-07-15

Universal recovery route: **the norm-extraction INPUT files necessarily contain the source texts** (the
extractor read them). Configs at `sk3:scripts/llama_norm_extraction/configs*/` point to each corpus's input.
Per task:
| task | route | status |
|---|---|---|
| creative-writing | done (`wp_comments/input.jsonl`) | ✅ p-scoring RUNNING (sk3 GPU1, 371 metrics × 1,040 stories) |
| press-releases | ✅ RECOVERED 15,067/15,067. ⚠️ LEAKAGE GUARD: norms were extracted from the NEWS ARTICLE (pair B) — the critique text. Scoring uses the PRESS RELEASE (pair A) only; 2,217/6,480 labeled pairs have PR text in the deconfounded parquet. **Queued for scoring (2,217 PRs, 73 metrics ≥10 pos pre-filter)** | queued |
| math-stackexchange | ✅ RECOVERED 6,492/6,492 (answer Body via Posts.xml; critique lives in separate comments — clean). **Queued for scoring (2,056 answers, 50 metrics ≥10 pos)** | queued |
| notice-and-comment | ✅ ids recovered BUT the recovered `rtc_text` IS the extraction input (the agency's response-to-comments = where the critique lives) → scoring on it would LEAK y. Needs the underlying public comments / proposed rule instead. **Blocked on leak-safe text.** | blocked |
| peer-review | ✅ texts recovered 23,398/23,398 (63% PDF full text, 37% title+abstract; clean — paper ≠ review). BUT pr_review_feedback norms were never metric-matched (no v2 decisions) → needs a matching run before y exists. | needs matching |
| humor | standup source text not recovered (skipped) | open |
| code-review | text ready; but matched norms ≠ source-carrying norms (1% join) → **re-match faithful norms** (Sonnet judge fan-out over a PR-stratified sample, e.g. 25-30K norms covering 2-3K PRs) | queued, needs launch decision |

## 8. Next action (the one run that is needed)
Score `p[T,M]` = certified metric prompt over the source texts for **creative-writing** (50 v14 metrics + full
R2 bank) and **code-review** (PR text, R2 bank), via local batch vLLM (NOT GLM — too many calls). Then compute
per-metric mention-AUC and correlate with (a) old opt cert, (b) v14 achieved_value. Do NOT reuse mbar panels.
