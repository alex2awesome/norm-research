# CW null-bank re-audit — RoyalRoad (VERDICT) + Wigleaf (CURATION) vs the pre-kill checklist

Charge: strict list `notes/2026-08-08__vat-3xN-decomposition-grid.md:17` marks both cells
"documented NULL BANK (cite, don't run)". That verdict predates the five-point pre-kill
checklist (`notes/2026-08-05__taste-decomposition-design.md:623-628`, dated "user 2026-08-08"
in the doc header despite the file's own 08-05 filename-date — the section itself is the
08-08 addendum). Neither cell has been run through it. This note does that, read-only, no
rebuild launched.

## 1. Where the null verdict actually lives

Both nulls trace to ONE campaign: `notes/2026-07-05__why-metric-discovery-plateaus.md`
§"Clean CW craft bank + overnight run matrix" (lines 166-356), completed 2026-07-06 06:43
("GPU3 freed 06:43", line 336). This is the **only** primary source; every later citation
(registry, strict list, taste-decomposition design doc) just re-quotes its numbers.

**Wigleaf (curation)** — clean-bank FINAL, line 308 and table lines 338-344:
> "wigleaf-clean FINAL (5 arms): 150 proposals, **0 kept**, floor ≈ 0, flux tail 0.255."
Full table (line 343): bank=CLEAN, base AUC **0.578**, proposals 150, kept 0, retest median 0.88.

**RoyalRoad (verdict)** — clean-bank COMPLETE, line 334:
> "royalroad-clean COMPLETE (5/5 arms): base AUC 0.505, 150 proposals, 0 kept, floor −0.022,
> retest median 0.90 (98% ≥ 0.5) → genuine null."

Re-quoted at `notes/2026-07-27__vat-run-registry.md:12-13` ("FICTION verdict = RoyalRoad
Amazon-KU stub cell EXISTS: n=1,274 canonical, floor .588, V .560-.570, clean craft-bank NULL
(.505, 150 proposals/0 kept, retest .90 = genuine)") and folded into the strict list at
`notes/2026-08-08__vat-3xN-decomposition-grid.md:17` and the full-grid drive plan at
`notes/2026-08-05__taste-decomposition-design.md:327-328` ("DOCUMENTED NULL BANK (craft-bank
0/150 kept, retest .90) — skip, cite the null" / "same null-bank status — skip, cite").

**Important mislabel carried forward:** the strict list and design-doc phrasing ("NULL BANK")
collapses two different findings into one word. Per the source (lines 285-329 of the 07-05
note), Wigleaf's clean bank AUC (0.578) is **not** at chance — it is the single highest
craft-rankability number in the whole CW leg, above even its own V floor (0.570, registry
line 49). "0 kept" there means no *new* proposed metric added bits beyond the existing
37-criterion bank (a saturation finding), not that craft is null for Wigleaf. RoyalRoad's
bank (0.505) genuinely is at chance. Both get the same "NULL BANK" tag; only RoyalRoad
earns it in the sense the tag implies.

Confound-clean status of the RoyalRoad text itself (per the task's pointer) is separately
verified at `notes/2026-06-12__taste-taxonomy.md:1299-1316` (§17m, 2026-06-16): stub-form
clean (both pools wayback-sourced, chapter_rank=1), time confound "REAL but WEAK + immaterial
to floor" (era→y corr −0.079), canonical rebuild `royalroad_v2_fiction_topicstrat.csv.gz`
n=**1,274**, pos=0.500, LEXICAL 0.588 / REGISTER 0.521, "<0.6 CLEAN, confound-clean by
construction." This is the dataset the 07-06 craft-bank run scored (registry line 12 quotes
the same n=1,274). The dataset is real and clean; the null is about the *craft-bank
instrument*, not the corpus.

Wigleaf's honest baseline is `notes/2026-06-12__taste-taxonomy.md:933-944` (2026-06-13,
"agent leak-free build"): **1,568 rows** (404 Top50 pos / 1,164 longlist neg), presentation
leak fixed (fetch_source AUC 0.90→0.500), TF-IDF 0.541 (near chance), V-feature 0.570.

No local data files exist under `datasets/creative-writing/{royalroad_stubs,wigleaf}/` beyond
scraper/parser scripts (`wigleaf/{parse,scrape}_wigleaf.py`) — the built csvs (`royalroad_v2_
fiction_topicstrat.csv.gz`, `wigleaf_labels_fixed.csv`, the medoid-bank-clean `bank.json`) live
on sk3 only, not synced to this checkout. All numbers below are read from notes, not
re-derived from raw files.

## 2. Pre-kill checklist applied

Checklist text, `notes/2026-08-05__taste-decomposition-design.md:623-628`:
> "(1) absolute minority-class count in train (not just the rate); (2) a simple baseline
> (TF-IDF/logistic) on the same split — baseline>chance while the big model is at chance =
> training-run failure, not cell failure; (3) registry search for historic working runs;
> (4) the verdict names WHICH DESIGN failed (grouping/transfer demand, k of groups);
> (5) seed spread vs the claimed effect."

| check | RoyalRoad (verdict) | Wigleaf (curation) |
|---|---|---|
| (a) minority-class count | n=1,274, pos=**0.500** → 637/637, balanced by topic×era stratified construction (taste-taxonomy.md:1314). Not a small-class risk. | n=1,568, **404 pos** / 1,164 neg (taste-taxonomy.md:933-934). 404 is absolute-count comparable to the mathlib retraction's ~360 train negatives that caused a false null (checklist's own origin case, design-doc:628-629) — same order of magnitude, worth flagging even though the rate (25.8%) is far less skewed than mathlib's 94%/6%. |
| (b) simple baseline on same split | LEXICAL/TF-IDF-style word floor **0.588** (taste-taxonomy.md:1315) vs craft-bank AUC 0.505 — baseline **beats** the bank. Passes: this is cell-specific (craft doesn't matter for market outcome), not a training failure, because *something* in the text predicts above chance. | TF-IDF **0.541**, V-feature **0.570** (taste-taxonomy.md:938-939) vs craft-bank AUC 0.578 — bank ≈ baseline, both mildly above chance. Passes on the same logic; also confirms the bank isn't a broken/degenerate predictor (it's not below the honest floor). |
| (c) registry search for historic runs | Two designs run per cell: dirty medoid bank (why-metric-discovery-plateaus.md:243-247, ~0.5, partially arm-disabled by a context-overflow bug) and clean k-medoid craft bank (lines 334, 5/5 arms, bug-fixed). Both converge on chance. No third design (e.g. GEPA-phrased bank, different judge) was ever tried. | Same two-design pattern (dirty 0.508 → clean 0.578, lines 226-227, 281). Clean-bank run is the only "decisive" one per the source's own framing (line 250: "the real experiment"). No GEPA-phrased or Gemma-4-31B-judged design has ever been tried on this cell. |
| (d) names which design failed | Explicit: 37/40-craft k-medoid bank (embedding-filtered from a 73.5k mined-rubric pool, NOT GEPA-phrased — `datasets/creative-writing/build_clean_craft_bank.py:1-20` docstring), 5-arm proposal system (unconditional/label_contrast/autometrics_iterative/metric_tree/residual), judge = "70B judge+proposer offline" (why-metric-discovery-plateaus.md:178) — model identity not stated in that note; contemporaneous repo convention for this exact pipeline shape (same week, same `medoid-bank-clean` pattern, math analog at `notes/2026-07-07__subcommunity-heterogeneity-screen.md:138-139`) is Llama-3.3-70B-FP8 (`scripts/tools/r3_cw_supervisor.sh:15`), **not** Gemma-4-31B — inferred, not directly confirmed in the CW note itself. Population = n=1,274 canonical topic×era-stratified fiction cell. | Same bank build, same 5-arm system, same unconfirmed-but-likely-Llama-70B judge. Population = the 1,568-row (or a subset — the exact n scored by the arms run is not logged in the 07-05/06 note; only proposal/kept/retest counts are given) leak-free Wigleaf build. |
| (e) seed spread vs claimed effect | **Not done.** The note reports per-metric judge *retest* reliability (same item judged twice, Spearman 0.90, 98%≥0.5) — a judge-consistency check, not a seed-spread check on the bank-AUC point estimate itself. Only one clean-bank run exists per cell; no repeat run at a different seed to bound 0.505's noise band. | Same gap: retest median 0.88, no independent seed rerun of the clean-bank AUC (0.578) or the "0 kept" proposal count. |

**3/5 checks pass cleanly (a-partial, b, c-partial, d); (e) is an outright gap for both
cells; (a) is a mild concern for Wigleaf specifically (404 abs. positives, same order as the
mathlib false-null case).**

## 3. Date check — instrument-version comparison

The "mature A-bank pipeline" (GEPA-iterated criteria, Gemma-4-31B judge, blinded anchor
battery K≥50, embedded in the Layer-1/2/3 VA_nl/T/Δ_beyond decomposition) is defined at
`notes/2026-08-05__taste-decomposition-design.md:74-77` (§3, "A-bank rule: GEPA-iterated +
Gemma-4-31b judge; blinded anchor battery EVERY batch"). The standing rule it codifies
(`feedback_a_bank_gepa_gemma4`) is already cited as binding in
`notes/2026-07-15__consul-collins-aside-status.md:37` ("GEPA-iterated + Gemma-4-31b judged
(feedback_a_bank_gepa_gemma4)") — **2026-07-15, nine days after** the RoyalRoad/Wigleaf null
runs completed (2026-07-06 06:43).

The mature bank was actually built and scored for `cw_community` (WritingPrompts upvotes)
under task `task-ms5c9kdd` "CW mature A bank" (`notes/2026-07-27__vat-run-registry.md:191`,
dispatched 2026-07-28 under the explicit "user directive: GEPA proposer AND executor",
line 185) → `datasets/creative-writing/va_bank_v2/rubrics_initial.jsonl`, GEPA-phrased,
scored by Gemma-4-31B offline-batch vLLM on **2026-07-29**
(`notes/2026-07-27__vat-run-registry.md:283-289`, "THREE NEW A CELLS MEASURED (Gemma-4-31B,
one engine, ~525K judge prompts)"). That task's own scope statement (line 191-192) says it
"closes the .520†-floor-harness hole AND is required to run on the SAME pool as the dense
model (writingprompts_modeling_clean grouped)" — i.e. it targeted the community/upvotes leg
only. Neither RoyalRoad nor Wigleaf appears anywhere in that dispatch or its harvest.

Confirming this from the registry's own bookkeeping: as of 2026-07-27 — **three weeks after**
the craft-bank null was recorded — the registry table still lists, for creative writing,
"✓V .570 (Wigleaf) · **◻ A bank (V5)** · ◻ T grouped" (curation column) and "✓ V .496/VA
.520†/T .820p (WritingPrompts) · **◻ mature A bank (V5)** · ◻ T grouped re-fit (V4)"
(community column) — `notes/2026-07-27__vat-run-registry.md:49`. The registry itself does
**not** treat the 07-06 craft-bank run as satisfying "A bank (V5)" for Wigleaf; it is carried
as still-outstanding, unchecked, right alongside the community leg's own not-yet-mature bank.
(RoyalRoad has no dedicated grid row at all in this registry table — it only surfaces via the
"lost-cells hunt" audit note at lines 12-14, never re-entered into the ✓/▶/◻ tracker.)

Instrument comparison, side by side:

| axis | RoyalRoad/Wigleaf null run (2026-07-05/06) | cw_community mature pipeline (2026-07-28/29 onward) |
|---|---|---|
| bank construction | embedding-cosine filter (MiniLM, ≥0.45 to craft anchors) + junk blocklist + **k-medoid selection** over a pre-existing 73.5k mined pool (`build_clean_craft_bank.py:1-20`) | **GEPA-iterated phrasing pass** over proposer-mined criteria, dispatched under explicit "GEPA proposer AND executor" directive (registry:185) |
| judge | "70B judge+proposer offline" (why-metric-discovery-plateaus.md:178), model not named in-note; likely Llama-3.3-70B-FP8 by contemporaneous convention, not confirmed | **Gemma-4-31B**, offline-batch vLLM, named explicitly (registry:283-289) |
| validity QC | per-metric retest-reliability only (same item scored twice) | retest reliability **+ blinded anchor battery K=50/class** (pos > neg > scrambled ordering) — e.g. cw_community's closure anchor check at `notes/2026-08-06__closure_cw_community.md:78`: "pos .7006 > neg .6598 > scrambled .0033, ordering holds" |
| decomposition framework | proposal-arm system (kept/floor bits); no V/A/VA_lin/VA_nl/T scaffolding | full Layer-1 nonlinear stack (`notes/2026-08-05__layer1_gemma_cells.md:1-36`) + Layer-3 GEPA closure mining (rounds 0-8, `notes/2026-08-06__closure_cw_community.md`) |
| dense ceiling (T) | **never computed** for either cell — `notes/ideas-backlog.md:185` still lists "`[ ]` dense ceiling on clean 96K... `[?]` RoyalRoad stubs... richer recovery; `[ ]` Wigleaf Top-50... full-text recovery" as open backlog items | T computed and reused across rounds (`Δ_beyond` is the program's central readout) |

The last row is the load-bearing gap: without T, **Δ_beyond (taste headroom beyond the bank)
is unknown for both cells** — the null-bank finding says the *existing* bank doesn't rank
(RoyalRoad) or can't be extended (Wigleaf), but says nothing about whether raw dense text
beats either bank, which is exactly the question every other terminal/closed cell on the
strict list answers before being called done.

## 4. Verdict per cell

**RoyalRoad (VERDICT) — NULL IS INSTRUMENT-LIMITED.**
The bank-AUC-at-chance (0.505) result itself is not obviously an artifact — checklist (a) and
(b) pass cleanly (balanced classes, and a same-split lexical baseline that beats the bank,
which is the checklist's own signature of a genuine cell-level null rather than a
training-run failure). What's unresolved is whether a *better* A-bank would do better: the
bank that produced the null is a k-medoid coverage-selected, non-GEPA, likely-Llama-70B-judged
instrument built three weeks before the GEPA+Gemma-4-31B standard existed, with no anchor
battery and no dense-model (T) comparison ever run. Rebuild that would test it: re-score the
existing n=1,274 canonical `royalroad_v2_fiction_topicstrat.csv.gz` population (already
confound-audited clean, no new data collection needed) with the `datasets/va_gemma_banks/
score_va_gemma_banks.py` machinery (GEPA-phrased criteria, Gemma-4-31B judge, K≥50 anchor
battery, same nuisance/reliability protocol used for cw_community) plus a dense LoRA T run
on the same split. Given n≈1,274 and the existing `score_va_gemma_banks.py`/dense-standard
recipes are reused wholesale (only a new bank builder + T split needed, per the
`score_scaleupC_banks.py` precedent at `notes/2026-08-08__scaleupC_builds.md:105-106`), this
is a small build — order a few GPU-hours for Gemma-4-31B scoring (n=1,274, single judge pass,
comparable in scale to HashtagWars' n=4,228 batch) plus a standard Llama-3.1-8B LoRA T run
(~similar cost to the existing dense-standard recipe, hours not days).

**Wigleaf (CURATION) — NULL IS INSTRUMENT-LIMITED, with an added mislabel flag.**
Same instrument gaps as RoyalRoad (pre-GEPA bank, unconfirmed non-Gemma judge, no anchor
battery, no T), but here the underlying finding is arguably not "null" at all: the clean
craft bank scores AUC 0.578 on Wigleaf, the single highest craft-rankability number measured
anywhere in the CW leg (above its own 0.570 V floor), and the "0 kept" result is a saturation
finding (no *new* metric adds bits beyond that bank), not a chance-level bank. Calling this
cell "documented NULL BANK" alongside RoyalRoad overstates the null; it should read
"documented SATURATED bank, un-audited against the mature pipeline." Checklist (a) also
flags a real, if modest, concern here that RoyalRoad doesn't have: 404 absolute train
positives is the same order of magnitude as the mathlib case that motivated this checklist
in the first place (design-doc:628-629), so a small-n artifact at the *margin* (e.g. exactly
which criteria get admitted) can't be ruled out from the record. Rebuild: identical
prescription to RoyalRoad — GEPA-phrased/Gemma-4-31B/K≥50-anchor rescoring of the existing
1,568-row leak-free build plus a same-split dense T run; similarly small (n=1,568, one Gemma
judge pass + one LoRA T run).

## 5. What's NOT resolvable from the record

Exact identity of the "70B judge+proposer" model used for the 2026-07-05/06 CW runs is
never named in the source note — I infer Llama-3.3-70B-FP8 from contemporaneous repo
convention (same week's math clean-bank run at `notes/2026-07-07__subcommunity-
heterogeneity-screen.md:138-139` explicitly names "Llama-3.3-70B-FP8"; `scripts/tools/
r3_cw_supervisor.sh:15` has that exact checkpoint path wired for CW-adjacent rescoring), but
this is circumstantial, not documented for this specific run. Also unresolvable: the exact
row count/class balance the 07-05/06 arms system actually scored for Wigleaf (only
proposal/kept/retest counts are logged for that run; the 1,568/404 figure is the nearest-in-
time honest build, not a number quoted inside the arms-run log itself) — a rebuild would
re-derive this trivially by re-running `build_clean_craft_bank.py`'s population loader, but
it cannot be pinned down from notes alone.

## Recommended next action

Do not touch RoyalRoad/Wigleaf's strict-list cells directly (out of scope for this pass).
Recommend the strict list's phrasing (`notes/2026-08-08__vat-3xN-decomposition-grid.md:17`)
be updated by its owner to distinguish "instrument-limited, rebuild-eligible" from a true
terminal null, and that a rebuild ticket be opened mirroring `task-ms5c9kdd` (CW mature A
bank) but scoped to RoyalRoad + Wigleaf specifically: GEPA-phrased bank + Gemma-4-31B judge +
K≥50 anchor battery + dense-T split, reusing `score_va_gemma_banks.py`/`score_scaleupC_
banks.py` machinery wholesale (per the `scaleupC` precedent), on the two already-clean,
already-audited populations (n=1,274 RoyalRoad / n=1,568 Wigleaf) — no new data collection,
CPU-cheap orchestration, one Gemma-4-31B GPU pass per cell plus one dense-standard LoRA T run
per cell.
