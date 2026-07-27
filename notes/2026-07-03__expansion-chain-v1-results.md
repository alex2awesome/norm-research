# Expansion-chain v1 results — the planted controls did their job

*Run completed 2026-07-03 01:24 (evening_gpu7_chain.sh: 70B grid reader → chains → 1B/3B/70B
scoring → reports; ~5h wall on GPU7). Artifacts: `outputs/r3_{cw,humor}/expansion_v1/`
{chains.json, chain_*.npz, expansion_report.json}; local snapshots + all analysis in notebook
`2026-07-02__two-faces-results-summary.ipynb` §4c. Design + registered predictions:
`2026-07-02__iso-performance-expansion-design.md`; formalism: two-faces-theory §2.4.*

## 1. What the planted controls caught (read this before the results)

The 4 mechanically-checkable control items (question-mark presence, quoted dialogue, >150 words,
second-person "you") were supposed to saturate ≥0.9 at level 0–1 for every competent reader. They
did NOT — and the failure pattern is diagnostic, not noise:

- **The 1B is instrument-INVALID.** Chance on `planted_question` in both domains (.48–.53), flat
  across all 8 expansion levels, while maximally self-consistent (self_agree ≈ .97). It cannot
  execute even a fully-stated mechanical rule: stable-but-flat verdicts = signal-extraction
  failure. ⇒ **1B-censoring cannot be read as tacitness** — it is a demonstrative floor only
  (prediction 3, "planted floors at L0–L1 for all readers", is REFUTED for the 1B). The graded
  iso-performance instrument is 3B→70B.
- **Gold-vs-view truncation artifact.** Planted gold was computed on FULL probe texts; readers see
  a `max_text_chars`=4000 view. Tail-sensitive rules are capped for long-text CW but not short-joke
  humor — exactly the observed asymmetry (70B question-detection: humor .81–.91, CW .66–.72;
  dialogue, which appears early in stories, is fine at .85+ in both). **v2 fix: compute planted
  gold on the truncated view** (`rule(t[:max_text_chars])`).
- **`planted_length150` was gold-imbalanced** (4% minority in both corpora — jokes are short,
  stories are long) — caught by the automatic `gold_flags`; its wild bal_acc swings are quantization
  on ~12 minority items. v2: median-split threshold per corpus.
- **A 3B compliance ceiling ≈ .70–.75 on mechanical rules** (question .55–.68, dialogue → .74). So
  metrics where the 70B bar exceeds ~.75 and the 3B plateaus near .73 may be COMPLIANCE-capped, not
  content-capped — applies to part of the humor censored group. v2: normalize matching by each
  reader's planted ceiling.

This is exactly the confound the user flagged ("model B just can't follow the instruction as
well") — measured, not assumed away. The controls earn their place in every future grid.

## 2. What survives (and it's the strongest validation of the program so far)

**Grid→chain replication 36/38.** The rescued/censored classification from the v1 GRID (different
messages, different rung construction) predicts the CHAIN outcomes: CW rescued 8/8 matched,
censored 0/10; humor rescued 8/10, censored 0/10. *Whether a concept's content verbally transmits
to a small reader is a stable, instrument-independent property of the concept.* This is the
program's central reliability claim, demonstrated across two independent operationalizations.

**Two regimes, sharply visible (3B reader, median curves over the nested chain):**
- *Rescued* concepts: start ~.57, RISE with expansion (CW .574→.648) — content is in-channel;
  words are a currency.
- *Censored* concepts: start HIGH from the bare name (.73–.77) and stay FLAT (humor mildly
  declines) — the name is a pointer into background the reader has (3B) or lacks irreparably (1B).
  Telling more adds nothing: the operational signature of prior-indexed / enculturated content.

**3B→70B expansion costs (the valid pair):** CW cheap — 8/18 need NO expansion, KM median L=1, 83%
matched at δ=0 (bootstrap P(match) median .95). Humor expensive — KM median L=4, 65% matched
(P .83), 7/20 censored (with the compliance-ceiling caveat above). Same two-regime geometry as the
vertical 70B ladder: CW saturates early; humor keeps demanding capacity.

**Type-tagged marginal gains (3B):** per-step medians are small (the aggregate rescued-group rise
is the robust signal), but the one consistent pattern across all 4 domain×group cells:
**contrastive increments are positive everywhere** (boundary +.016–.025, counterexample
+.014–.033) **while re-stating definitions is flat-to-negative** (3/4 cells negative; humor
rescued −.048). Near-miss/contrast content is the consistently useful thing to ADD to a name —
prediction 2 (mechanism/procedure > definition) half-right: definition is indeed the weakest, but
boundary/counterexample beat mechanism/procedure.

**Transitivity (prediction 1): UNDERPOWERED, additive-where-defined.** With the 1B invalidated,
most triangles are censored at the weak end (direct-censored 15–18 of 22–24). Among defined: CW
4/4 zero-slack at δ=0, 8/9 at δ=.05 (consistent with additivity/potential structure); humor 2–5
defined triangles — no call. Structural constraint: Llama gives 4 readers, 1 invalid (1B), 1
consumed as reference (8B) ⇒ only one valid pair per reference frame. **v2 design: rotate the
reference to a 70B-orbit target** (the MI-ONLY retarget artifacts supply genuine 70B M_i for CW;
humor needs a ~1h retarget-mi-only pass) → frees the 8B as a reader → valid 3B→8B pair, and
cross-frame chaining (8B-ref: 3B→70B; 70B-ref: 3B→8B) poses the potential-consistency test.

**Prediction scorecard (registered yesterday):** (1) transitivity — underpowered/no call;
(2) type ranking — half-right (contrast > mechanism > definition); (3) planted floors L0–L1 —
REFUTED for 1B, partially for 3B (the refutation IS the instrument finding); (4) gate-passers
cheaper to expand — not yet computed cleanly (needs compliance normalization first).

## 3. Diagram redo + knowledge-type histograms (user review, 2026-07-03)

**User caught a real flaw:** the first diagram's "matched" panel showed a FIRST-CROSSING (1B
overshot the 3B bar, .715 vs .598) — matching in the censoring sense, not iso-performance. Redo
process: searched for true ε-triples (70B@name ≈ 3B@ℓ₃ ≈ 1B@ℓ₁, ℓ₁>ℓ₃≥1). Three point-value
candidates existed at ε≤.02 and **all three failed vetting** — CW gi=24's 1B *starts* at .766 and
declines to its "match"; CW gi=25's 3B match-level is a one-level dip in an .82 plateau
(first-ε-touch artifact); humor gi=34 has H_M=.177 and a 70B curve that collapses with expansion.
Bootstrap says ε≤.03 triples aren't certifiable at 300 probes anyway (P(spread≤.03)=.07–.20).
**Conclusion: no honest 3-class ε-triple exists in v1 — because the 1B is instrument-invalid.**
The rebuilt diagram (notebook §4c-area cell + `figures/iso_performance_expansion.html`) shows the
two CERTIFIED climb-to-plateau iso PAIRS with live paired-bootstrap CIs:
- CW "Worldbuilding coherence": 3B .552@L0 → .688@L6 ≈ 70B@name .696 (Δ=−.005 [−.074,+.054],
  P(|Δ|≤.05)=.85; plateau ±.012), green = the literally-appended 6 typed increments.
- humor "Misdirection and reveal design": 3B .585@L0 → .691@L7 ≈ 70B@name .689 (Δ=+.002
  [−.059,+.068], P(|Δ|≤.03)=.65 — tightest cell; note 3B is at-or-above the bar from L2/~40 words).
- 1B column = instrument floor (planted-invalid), censored contrast strip retained.

**Knowledge-type histograms (§4d):** for 3B→70B (valid pair, δ=0, 38 metrics): **16 match at the
bare NAME** (no extra knowledge — the 3B decodes the pointer), 12 need expansion, 10 censored.
Closing increments among the 12: **boundary 4 + counterexample 2 (contrastive = half)**, checklist
3, definition 3. Consistent with the marginal-gain finding (contrast positive in 4/4 cells).
1B→3B (caveat): 8 at name, 8 expanded (closing: mechanism 3, checklist 2, others 3), 22 censored.
Position-type confound noted (fixed schedule; 4 reversed gis/domain break it partially).

**§4d/§4e ladder plots (REBUILT 2026-07-03 per user: 70B → 8B → 3B, 1B dropped):** grid data
(the chain never scored the 8B), matching in BITS (≥ bar − δ, δ = max(.02, 20% of bar), bars
< .03 bits excluded as uninformative). Matched-rung histograms: 8B matches at the NAME almost
always (CW 34/34; humor 28 name + 5 definition, 0 censored) — as expected from its D2
self-consistency advantage (its costs are optimistic lower bounds, caveat chip shown); 3B is
genuinely graded (CW: 15 name / 11 definition / 2 explanation / 2 rubric / 1 dossier / 3 censored;
humor: 13 / 14 / 3 / — / — / 3). Words boxplot: 70B bar ≈ 5w; 8B ≈ 5w (humor definition tail 24w);
3B median 15–18w, GIVEN expansion median 23–25w, ~10% censored. Chain typed-increment histograms
keep only the valid 3B→70B series. (Minor: the regenerated CW grid report moved one δ=0
boundary-tie metric — 1B-never-match 74%→72%; a single-metric tie wobble, direction-of-conclusion
unaffected.) Earlier 1B-based §4e numbers superseded by this block.

## 4. Bits realignment (user-flagged, 2026-07-03) + the 3-class prompt demonstration

User correctly objected that balanced accuracy appears nowhere in the theory doc and the 8B
reference deserved scrutiny. Resolution (notebook §3c + theory doc §2.2 pin): recovery re-measured
with the census's own `i_binary` (exact binary MI, bits; same conventions as `certificate()`).
Findings: (1) the ladder ordering survives in bits (1B≈0; 3B 0.03–0.10b; 70B 0.05–0.09b; 8B-ref
inflated 0.14–0.48b = self-consistency, visible directly); (2) the SCALE is humbling — cross-reader
single-message transmission is ≈5–15% of H_M vs census heads at ≈70% of H_M; (3) **ostension test
(now computable): ZERO exceedances of OPT_Ω+ε_adv** across all non-reference readers × rungs ×
99 kept metrics — every Face-2 channel stays inside the certified articulated census. Reference
choice (8B vs rotation vs consensus) flagged in §2.2 as a user decision. Demonstration rebuilt
(`figures/iso_staircase_3class.html`): TRUE bits-iso staircases with the ACTUAL prompts —
humor "Verbal economy and clarity": 70B@name(4w) ≈ 8B@definition(22w) ≈ 3B@explanation(34w),
spread .002 bits (bal_acc agrees: .789/.792/.779); CW "Exposition timing and integration":
70B@name(4w) ≈ 8B@definition(21w) ≈ 3B@full_rubric(24w); 8B shown with its D2 caveat; 1B row shows
its actual prompt at ~chance; the verbatim judge template included.

## 5. Anchor correction (user-flagged 2026-07-03 PM): scope and resolution

User asked whether the 8B-reference error requires re-running experiments or was notebook-only.
Traced through every layer: **Face 1 is clean** (anchor-free within-executor; `metric_verdict` is
the documented anchor-free reconstruction target; all certificates stand untouched). The
cross-model reference was a Face-2 drift confined to the ANALYSIS layer — the grid/chain drivers'
`report()` phases plus §2.2 of the two-faces doc (written the same day as the grid, so citing it
was circular). The collected score tensors are reference-free ⇒ **no GPU re-runs**; the
executor-consistent re-analysis was computed offline from existing artifacts (notebook §3d,
`grid_bits_self.json`). Code corrected: `run_decompression_grid.report()` now emits
`H_self`/`self_bits` (each reader vs its OWN full-rubric orbit judgment, census `i_binary`) as the
PRIMARY readout, both machines synced; §2.2 amended (anchored readout renamed "cross-executor
transmission", requires sign-off if ever promoted; D2 self-referentiality dissolves). Key §3d
results: H_self ladder CW 0.50→0.92 bits (1B→70B), humor 3B stable on only 34/60 — capacity =
judgment richness/stability; the self-consistency trap is measured (1B self-recovers 65% of its
own simple judgment from the name — self-recovery fraction is not a capacity claim without the
validity gates). Chain driver's report gets the same patch in v2 (task #9).

## 6. v2 fixes (small, ordered)

1. Planted gold on the truncated view + corpus-median length threshold + 2 extra balanced rules.
2. Compliance-normalized matching (match in units of each reader's planted ceiling).
3. Reference rotation to 70B-orbit M̄ (CW ready; humor needs 1h retarget pass) → 8B as reader,
   valid triangle work.
4. Per-group (not pooled) type-gain reporting as the default.

Related: [[project_isoperf_expansion_experiment]], notebook §4c,
`2026-07-02__iso-performance-expansion-design.md`, two-faces-theory §2.4.
