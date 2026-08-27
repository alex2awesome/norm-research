# mathlib style: where the deterministic signal lives, and the confounds it isn't

*Short note on the mathlib style-V analysis (2026-06-23). Companion to `mined_norms_taxonomy.md`.*

## ⛔⛔ HEADLINE (2026-06-25, LEAK-FREE REBUILD): the matched style-revision signal was 100% a label leak; the clean, leak-free signal is NULL (V≈C≈chance)

The entire matched-leg V/A/C ladder below (V 0.71, A 0.67, C 0.94, and the "size-control"
A>V correction) was a **label leak** (see RETRACTION section). The leak is now fixed by
restricting features to **line-anchored excerpts** (real code around a comment, not the
file-header fallback) and the signal has **disappeared entirely**, triangulated across two
independent labels:

| label (all leak-free; n_decl==0 share ≈ 0.08, was 0.37) | n | V (style) | V (size) | C (TF-IDF) |
|---|---|---|---|---|
| any-revision (PR has any outdated thread) | 6910 | 0.525 | 0.553 | 0.585 |
| style-specific revision (style-cat thread went outdated) | 5769 | **0.515** | 0.545 | **0.583** |
| *(old leaky, for reference)* | — | *(0.71)* | *(0.79)* | *(0.94)* |

**Interpretation:** first-push mathlib code does NOT predict whether it will be style-revised —
for deterministic style features OR full lexical content, on any-revision OR style-specific
labels. Everything ≈ chance. The old 0.71/0.94 was the leak, not signal. **A (the articulated
judge) was NOT re-run:** it reads the same first-push excerpt as V and C, so with both at chance
A will be ≈ 0.55 too — a GPU run would only confirm the null. The one caveat: the leak-free
excerpt is from a *different region* than the revised one (revised regions are the line=null
ones we can't use), so this tests PR-GLOBAL style → revision; A would judge that same global
excerpt and so cannot access local signal V/C miss.

**Why this reconciles the math tasks:** Math.SE/AoPS text carries quality signal (V 0.55–0.59,
C 0.73–0.79) → A articulates it → V<A<C. mathlib code carries NO revisability signal → nothing
for V, C, or A → **V≈A≈C≈chance.** This is the max-verifiability anchor in its purest form:
linters + compiler absorb correctness/style so thoroughly that residual revision is driven by
reviewer attention/topic/process, not the code. The leak had faked the opposite (0.94).
Scripts: `line_anchored_v2.py` (any-revision), `line_anchored_v3d.py` (style-specific,
`thread_norms.jsonl` `primary` ∈ {PROOF_STYLE,NAMING,DOCUMENTATION,FORMATTING,SIMP_ATTRS,
STATEMENT_FORM,ORGANIZATION,METAPROGRAMMING,API_COMPLETENESS} & isOutdated). The clean
mathlib result lives on the **accept/reject leg** (`accept_reject_clean.parquet`, see
`VAT_CLOSURE.md`): C 0.74 > V 0.65 > A 0.56 (V+A=V, judge redundant).

---

## The question
Can deterministic ("V") features capture mathlib's style norms? We mined ~3,500 review
comments (24 subagents) → ~50 distinct norms, built 23 checkable V metrics, and tested them
at three granularities.

## Headline: the signal is PR-level "global polish", not edit-level style

| task | V (all 23 rules) | what it tests |
|---|---|---|
| **strict pairwise** (before vs after of ONE edit, within-PR) | **~0.54 (chance)** | can a rule read THIS style change? → **no** |
| **matched** (revised vs not-revised PR, size+cat+year controlled) | ~0.74–0.80 | does a messier PR get revised more? → **yes** |
| lexical ceiling (TF-IDF, group-CV) | 0.94 | headroom only a reader reaches |

On the matched task, **two features carry essentially all the signal**: `import_ok` (import
sorting, LOO-drop +0.039) and `tactic_auto` (simp/grind-vs-manual ratio, +0.024). Every
fine-grained norm the subagents mined (naming, indentation, prime-suffix, non-terminal-simp,
docstring form, anti-patterns) contributes **~0** to the matched separation and is at chance
on the strict task. The fine-grained style signal is not capturable by any aggregate rule —
it needs a reader (A judge) or dense model (C).

## Confound check: is the matched signal really "style", or authorship/tenure?

Suspect: messy-import PRs are written by newcomers, who get revised more for many reasons.

- **Not author fingerprints.** ICC (between- ÷ total variance) for every feature < 0.22
  (line_len_ok 0.21, decl_spaced 0.18, import_ok 0.14, …). ~80–90% of variance is *within*
  an author, PR-to-PR — it's not "who wrote it."
- **Authors DO learn** (import_ok improves for 80% of authors across their PR sequence,
  line_len_ok 71%) — so the norms are learnable, not static habits.
- **But tenure is NOT the confound.** Adding career-stage matching leaves the signal
  unchanged (0.740 → 0.739). Newcomer-status explains nothing.
- **Author identity barely matters.** The strictest design — an author vs *themselves*
  (same author's revised vs non-revised PRs) — drops the signal only 0.740 → 0.714.

**Conclusion:** the matched signal is genuine **within-author, per-PR "global polish"**
(import organization + tactic-automation density). It is not an authorship or tenure
artifact. But it operates at the PR level — it cannot localize *which edit* reviewers will
change (that is at chance, and is the exclusive domain of an articulable judge / dense model).

## ⛔ RETRACTED (2026-06-25): the whole matched V/A/C ladder was a label-leak artifact

**Everything below this line that quotes a "SIZE-controlled" ladder is WITHDRAWN.** The
"SIZE" feature was never PR size — it was a mechanical proxy for the revision label, baked in
by the excerpt builder. Traced to `scripts/build_revision_pairs.py::excerpt()`:

```python
if line is None:  return "\n".join(lines[:50])    # ← file HEAD (copyright + imports)
else:             return lines[line-25 : line+25]  # ← ±25 lines around the comment
```

The label chain is deterministic:
`rev=1 ⟺ is_outdated=1 ⟺ line=null ⟺ excerpt = file-head fallback ⟺ n_decl≈0`.

| thread subset | share | **is_outdated (= rev label)** | n_decl=0 share |
|---|---|---|---|
| line=null (header fallback) | 82% | **1.000** | 22% |
| line-anchored | 18% | 0.049 | 10% |

So "SIZE alone AUC 0.79" was the model **reading the label off the header-fallback**, not
measuring complexity. The monotone *decreasing* rev-rate vs n_decl (0.76 → 0.13) is the same
artifact (fewer decls ⟺ header fallback ⟺ outdated). Residualizing V/A "against SIZE" was
residualizing against a **label proxy**, which is why V collapsed to 0.44 — that destroyed
information, it did not deconfound. **The "A (0.58) > V (0.44)" correction is itself retracted.**

**True whole-PR size is mild and honest** (joined from `friction_full_v2`: additions/deletions/
changed_files): `corr(rev, additions)=+0.24`, `corr(rev, size)=+0.16`, true-size AUC **0.668**.
And `corr(true_size, window-n_decl) = −0.09` — the leaked "SIZE" and real size are **unrelated**.
A genuine, weak opportunity effect exists; it is nothing like 0.79.

**Manual audit (50 excerpts, 2 Opus subagents) confirms the corpus is contaminated:**
- *small (n_decl=0)*: **84% are header/import fallbacks** whose reviewer comment is about code
  located elsewhere in the PR — useless for style judgment. 96% outdated, 88% line-null.
- *large (n_decl≥6)*: 84% real code, but only **36% style-judgeable** — reviewers use
  declaration-dense regions to debate *architecture/API* (lemma necessity, placement, instances),
  not *style* (naming, idiom, formatting). So even the clean rows are mostly off-target for style.

**What this invalidates:** the matched-leg AUCs (SIZE 0.77/0.79, V 0.71–0.74, A 0.67, C 0.91–0.94)
are all contaminated by the header-fallback/outdated leak to unknown degree, because the per-PR
excerpt is drawn from a thread that is line=null 82% of the time. **A valid V/A/C ladder requires
re-running on the line-anchored subset only** (18% of threads, the rows where the excerpt actually
contains the commented code) — not yet done.

**What still stands:**
- **Accept/reject leg is CLEAN** (separate corpus, real `diff` column, not header-windowed):
  true-size AUC 0.668 (mild, honest), A-judge's 10 metrics all |corr with size| ≤ 0.15,
  A-judge alone AUC 0.572. No leak. Those numbers are unaffected.
- **Edit-level pairwise V≈0.54 (chance)** is conservative under the bug (degenerate header-vs-header
  comparisons → chance), so the "no edit-level deterministic signal" finding is safe.

**Lesson:** before trusting *any* feature's AUC, inspect what the feature physically reads on real
rows. "SIZE" looked like a complexity confound, was defended as an "opportunity" signal by a 102-PR
audit, and was actually a builder artifact leaking the label. The 102-PR audit answered the wrong
question (causal scrutiny) on the comment text while the leak lived in the excerpt *geometry*.
See [[feedback_apples_to_apples_dense_vs_baseline]], [[feedback_validate_before_scaling]].

---

## ⚠️ ORIGINAL TEXT BELOW — SUPERSEDED BY THE RETRACTION ABOVE, kept for provenance

## Why this matters (V/A/C) — FINAL, with the A judge + SIZE control

**SIZE/COMPLEXITY DOMINATES.** The matched "revised" label is mostly a complexity label:
SIZE alone (n_decl, n_lines, n_auto, n_man — raw counts) predicts revision at **AUC 0.773**,
beating V (0.715) and A (0.666) standalone. The flen-octile matching was too coarse — these
counts vary within an octile and carry the real signal.

**"A < V" was an uncontrolled-SIZE artifact (RETRACTED).** V's style *ratios* (`tactic_auto`,
`import_ok`, …) are secretly SIZE-proxies (tactic_auto corr 0.63 with raw n_auto). Residualizing
both V and A against SIZE (per group-CV fold):
| model | raw AUC | residualized of SIZE |
|---|---|---|
| V (deterministic style ratios) | 0.715 | **0.440 (chance — V has NO size-independent style signal)** |
| A (articulated judge norms) | 0.666 | **0.578 (judge KEEPS real size-independent style signal)** |
| SIZE+V | 0.773 (V adds ~0 over SIZE) | |
| SIZE+A | 0.780 (A adds +0.007 over SIZE) | |

**CORRECTED conclusion:** once complexity is controlled, **A (0.58) > V (0.44)** — the judge
captures genuine size-independent style signal that deterministic ratios do not (their edge was
entirely a size confound). But both are modest because SIZE/complexity is the dominant driver of
"got revised." The lexical ceiling C=0.94 likewise partly reflects complexity (bigger PRs = more
tokens), so the V→C gap is NOT clean "style" either — much of it is volume.

**Is C just complexity? NO (c_vs_size.py).** TF-IDF (lexical C) alone AUC 0.914; SIZE alone 0.794;
SIZE+TF-IDF 0.913 (SIZE adds ~0 over TF-IDF — SIZE is SUBSUMED by token counts). But TF-IDF
**residualized of the size-predictive component still AUC 0.857** — the lexical ceiling is genuine
token-level signal, not volume. corr(pred_SIZE, pred_TFIDF)=0.72. So the size-controlled ladder is
**V 0.44 < A 0.58 ≪ C 0.86**: deterministic ratios have no size-independent signal; the judge
captures modest genuine style; the dominant predictor is lexical, which neither judge nor
deterministic features capture well. (Open confound: the 0.86 lexical-after-size could be
topic/area — modules that get revised more — rather than style quality; untested.)

**Area/topic is NOT a confound (area_confound.py).** AREA (71 modules) alone AUC 0.487 (chance);
corr(pred_TFIDF, pred_AREA)=0.032; TF-IDF residualized of SIZE+AREA still 0.858 (area adds nothing).
So the 0.86 lexical signal is genuine code CONTENT — not complexity, not topic — captured only by
bag-of-words. **Final: predicting mathlib style-revision = lexical code content (0.86) that is
neither mechanically checkable (V 0.44) nor well-articulable (A 0.58) — a "tacit-lexical" layer
captured only by C.**

**Revised mathlib-style story:** "predict whether a PR gets style-revised" is primarily a
*complexity* prediction (0.77); beneath that, articulated judge norms carry real style signal
(0.58) that deterministic features lack (0.44). Edit-level (pairwise): V≈0.54 and judge direct
discrimination BLIND 0.50 / GROUNDED 0.49 — all chance.

Caveat: A at 48% JSON parse (n=1935, label-balanced); spread 1.10, not collapsed. Lesson: the
apples-to-apples/control discipline applies to *ratios too* — a ratio of counts is still a size
proxy. See [[feedback_apples_to_apples_dense_vs_baseline]].

*Scripts: `lean_style_metrics.py`, `expand_v.py`, `v_strict_pairwise.py`, `matched_drivers.py`,
`authorship_check.py`, `tenure_matched.py`, `a_judge_style_final.py`, `size_control_check.py`.
Data: `revision_pairs.jsonl`, `first_push_diffs.jsonl`, `a_style_verdicts_final.jsonl`.
Retraction scripts: `size_truth.py`, `size_degen.py`, `ar_size_check.py`, `make_audit2.py`.*
