# FULL-GRID V+A+T battery — the master fused ledger

Date opened: 2026-08-08 (filed under the assigned 08-10 name).
**STATUS: GRID CLOSED (2026-08-08).** VAT 16/16 · Direction-1b 4/4 · V3 13/14 ·
matched-raw displacement control 4/4 · **zero §11 AUTO-FABLE-AUDIT flags** · T_decor
RESOLVED (not shipping) · mathlib RESOLVED (excluded, dense arm at chance) · 2 cells
structurally incapable of a V3 arm.
Owner: `claude-vat-fullgrid`. Descriptive only — no claim edits to other notes.

**One-line state:** every eligible cell has VA_lin / VA_nl / VA_nl@full / T / VAT_lin /
VAT_nl with group-bootstrap CIs; 13 V3 arms landed with re-gated intervals; the
matched-strength Direction-1b arm cleared the grid's only standing-rule flag; and the
matched-raw control settled the one open mechanistic question (displacement is real on
long documents, and is NOT the explanation for the negative N&C deltas). Remaining
exclusions are all measured, not pending.

**User directive (2026-08-10, via §12 of `notes/2026-08-05__taste-decomposition-design.md`):**
all 7 tasks × 3 preference variables, all free sk3 GPUs, one fused ledger row per
eligible cell. This file is the program's central results table.

## Terms, spelled out (every abbreviation, first use)

| symbol | meaning |
|---|---|
| **V** | verifiable / deterministic-surface features (length, digits, formatting, …) |
| **A** | articulated criterion scores — named rubric items, judged label-blind by Gemma-4-31B |
| **VA_lin** | logistic (linear) aggregation of the V+A score matrix, frozen Layer-1 protocol |
| **VA_nl** | HistGradientBoosting (nonlinear) aggregation of the SAME matrix, seed-mean over {0,1,2} |
| **T** | dense-standard clean-eval AUC — Llama-3.1-8B LoRA reward model reading raw text |
| **E** | evaluation-valid rows = rows OUTSIDE the dense model's own training split (its eval+test buckets); the only rows where the dense per-row score is out-of-sample |
| **VAT_lin / VAT_nl** | the V+A matrix with the dense per-row probability appended as ONE extra column, refit on E with the frozen Layer-1 linear / HistGB stacks (Direction 1 of `notes/2026-08-07__vat_fusion_directions.md`) |
| **D1b** | the Direction-1b two-stage fused arm at MATCHED training strength: stage 1 = the full-population grouped-OOF bank prediction, stage 2 on E = a 2-column logistic combiner [bank_oof, dense_prob] with GroupKFold(5) OOF, against a bank-alone combiner as calibration control. **This is the arm the §11 check should use where it exists** — VAT_nl is fitted on ~20% of the rows the fullfit bank sees, so "fused vs bank at full strength" is otherwise not like-for-like. Results in `results/direction1b.json`. |
| **V3** | "criteria-in-prompt" dense arm — a fresh Llama-3.1-8B LoRA trained on [raw text + top-k "criterion name: score" lines]; Direction 3 of the same note, production recipe from `notes/2026-08-09__v3_audit_fable.md` §5 |
| **T_decor** | dense model retrained with importance weights that decorrelate y from the mined nuisance set (design §12 step 4b) — **RESOLVED, not shipping; see the T_decor box** |
| **Δ_interact** | VA_nl − VA_lin (interactions of articulated criteria; articulable) |
| **Δ_beyond** | T − VA_nl (the only part eligible to be called taste) |

## Protocol (binding, inherited — not re-decided here)

- **Leakage rule:** every fused readout lives on E. Stacks refit on E with grouped
  OOF `GroupKFold(5)` on the cell's own canonical grouping unit. Bank families:
  `clean_once` (population-level unsupervised degeneracy screen once on the E
  submatrix, then StandardScaler+LR / raw HistGB) for the peer/N&C cells;
  `impute_perfold` (median impute + indicator fit per train fold) for CW/HashtagWars.
- **CIs:** group-level (cluster) paired bootstrap, 2,000 draws, on (VAT_nl − T),
  (VAT_nl − VA_nl), and where a V3 arm exists (V3 − T), (V3 − VA_nl).
- **Selection caveat (matched across arms, inherited from T itself):** on several
  cells the dense checkpoint was selected on a split inside E, so E is not
  selection-clean for ANY arm. Per-split readouts are kept wherever the two halves
  disagree.
- **V3 production recipe** (from the 08-09 audit §5): frozen 8B LoRA (r16/a32,
  lr 5e-5, bs16, max_len 1024, 2 epochs, seed 42); **k=20** criteria, ranked by
  TRAINING-FOLD-ONLY grouped permutation importance; **names only**; **PREPEND the block
  on long-text cells** (right-truncation otherwise deletes it on exactly the long
  documents). The k=20 choice is UNCONFIRMED — see the k-confirm box.

## Cell inventory (the strict list, `notes/2026-08-08__vat-3xN-decomposition-grid.md`)

16 eligible cells. EXCLUDED by directive: Style Invitational + homepage curation (both
TERMINAL) and patents (closed leak post-mortem). **mathlib was un-blocked during this
session and then excluded on measured grounds** — its class-weighted dense arm is at chance
on test across all three seeds; see the mathlib box.

## MASTER FUSED LEDGER

All numbers are AUC **on E** unless the column says otherwise. `—` = arm not landed.
Regenerated by `python3 methods/taste_decomposition/fusion/fullgrid_ledger.py` — never
hand-edited. `VA_nl@full` = the bank at full training strength (fullfit@E); the §11 check
uses the STRONGER of the two bank readings, and prefers D1b as the fused arm where it
exists. Terminal bank arm per cell (round4 peer_verdict, round5 nc_responded, round8
cw_community); dense-seed ensemble where the cell has several seeds.

<!-- TABLE-ANCHOR: regenerated on every landing -->

| field | cell | y-type | n_E | grp | T | VA_lin | VA_nl | VA_nl@full | VAT_lin | VAT_nl | V3 | D1b | VAT−VA_nl [CI] P | VAT−T [CI] P | §11 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| Peer review | `peer_verdict` | verdict | 1244 | 1239 | .7769 | .6682 | .6684 | — | .7364 | .7415 | .7544 | — | +.0693 [+.0433,+.0957] 1.00 | −.0394 [−.0555,−.0241] 0.00 | PASS |
| Peer review | `peer_curation` | curation | 1571 | 1571 | .5936 | .5533 | .5286 | .5891 | .5792 | .5542 | .6386 | .6054 | +.0340 [+.0030,+.0643] 0.98 | −.0341 [−.0661,−.0021] 0.02 | PASS |
| Peer review | `peer_revealed` | revealed | 478 | 478 | .8842 | .6901 | .6554 | .7240 | .8475 | .8478 | .8882 | .8805 | +.2051 [+.1558,+.2544] 1.00 | −.0374 [−.0562,−.0199] 0.00 | PASS |
| Regulatory (N&C) | `nc_responded` | verdict | 1904 | 1010 | .8167 | .6683 | .7912 | — | .7325 | .8319 | .7950 | — | +.0416 [+.0223,+.0598] 1.00 | +.0111 [−.0070,+.0293] 0.88 | PASS |
| Regulatory (N&C) | `nc_outcome` | curation | 1417 | 692 | .6238 | .5956 | .6121 | .6430 | .6072 | .6227 | .6196 | .6567 | +.0223 [+.0001,+.0435] 0.98 | +.0002 [−.0307,+.0332] 0.49 | PASS |
| Regulatory (N&C) | `nc_agree` | agree | 1009 | 498 | .6034 | .6122 | .5627 | .6134 | .6189 | .5713 | .5929 | .6256 | +.0330 [+.0076,+.0619] 1.00 | −.0227 [−.0907,+.0464] 0.30 | PASS |
| Creative writing | `cw_community` | community | 7008 | 5136 | .7921 | .6710 | .6652 | .6651 | .7787 | .7869 | — | — | +.1231 [+.1119,+.1353] 1.00 | −.0038 [−.0073,−.0003] 0.02 | PASS |
| Humor | `hashtagwars_verdict` | verdict | 924 | 8 | .7315 | .6192 | .5290 | — | .6651 | .6454 | .7063 | — | +.1114 [+.0463,+.1693] 1.00 | −.0953 [−.1687,−.0161] 0.01 | PASS |
| Humor | `cap_finalist` | curation | 1055 | 46 | .6124 | .5923 | .5806 | .6666 | .5935 | .6077 | .6707 | — | +.0349 [−.0024,+.0710] 0.97 | −.0165 [−.0711,+.0369] 0.27 | PASS |
| Humor | `cap_crowd` | community | 2190 | 64 | .5554 | .5863 | .5831 | .6217 | .5878 | .5920 | .6303 | — | +.0047 [−.0105,+.0191] 0.73 | +.0288 [+.0007,+.0565] 0.98 | PASS |
| Humor | `jokes_community` | community (replication) | 3163 | 10 | .7469 | .6643 | .6888 | .7242 | .7282 | .7375 | .7411 | — | +.0490 [+.0304,+.0636] 1.00 | −.0109 [−.0182,−.0022] 0.01 | PASS |
| Math | `mathse_accepted_verdict` | verdict | 2600 | 992 | .6439 | .5980 | .5737 | .6159 | .6316 | .6196 | .6303 | — | +.0419 [+.0226,+.0620] 1.00 | −.0218 [−.0375,−.0058] 0.00 | PASS |
| Math | `mathse_vote_score` | vote | 2326 | 1140 | .6538 | .6242 | .6107 | .6352 | .6555 | .6558 | .6460 | — | +.0438 [+.0262,+.0612] 1.00 | +.0007 [−.0155,+.0170] 0.53 | PASS |
| Math | `aops_curation` | curation | 5202 | 606 | .7806 | .7712 | .7705 | .7735 | .7799 | .7851 | — | — | +.0161 [+.0058,+.0269] 1.00 | +.0044 [−.0104,+.0199] 0.73 | PASS |
| Software code | `code_v3` ‖ | verdict | 11452 | 255 | .6933 | .6255 | .6932 | — | .6888 | .7449 | — | — | +.0441 [+.0202,+.0730] 1.00 | +.0476 [−.0480,+.1668] 0.68 | PASS |
| Journalism/press | `press_verdict` | verdict | 605 | 45 | .7744 | .5796 | .6810 | .7442 | .6226 | .7487 | .7714 | — | +.0786 [+.0513,+.1150] 1.00 | −.0189 [−.0449,+.0071] 0.06 | PASS |

AUTO-FABLE-AUDIT flags: **NONE.** Every eligible cell's best fused arm beats its bank at
full training strength. **‖ = the pooled row is NOT quotable for that cell** (see the
code_v3 box).

Caption rows carry ROW-level bootstrap CIs (the Direction-1 convention on those cells);
every other row carries GROUP-level CIs. They are not interchangeable — row-level
intervals understate width on coarsely-grouped cells, which the caption cells are (46 and
64 contests).

### ✅ The one §11 flag RESOLVED — and the reason it existed is a methodological result

`peer_curation` flagged earlier in this session (E-refit VAT_nl .5542 vs full-strength bank
.5891). I recorded it as CONDITIONAL rather than as a failure of fusion, on the argument
that the two arms were fitted on different numbers of rows, and queued Direction-1b to
adjudicate at matched footing. It came back **D1b .6054 — above the bank by +.0163
[−.0069,+.0404] P=.91, and above the bank-alone combiner by +.0267 [+.0043,+.0504]
P=.99** — and its V3 arm then landed at **.6386**, higher still. The flag clears twice over.

Worth stating as a finding rather than bookkeeping: **on all four cells tested, comparing an
E-refit fused arm against a full-strength bank gives a systematically pessimistic — and on
one cell, sign-flipping — read of whether fusion helps.** The §11 standing rule is only
meaningful when both sides are fitted on the same rows.

### Derived decomposition quantities (design §0)

| cell | Δ_interact = VA_nl−VA_lin | Δ_beyond = T−VA_nl | fusion gain = max(fused)−max(bank) |
|---|---:|---:|---:|
| `peer_verdict` | +.0002 | +.1085 | +.0860 |
| `peer_curation` | −.0247 | +.0651 | +.0495 |
| `peer_revealed` | −.0347 | +.2288 | +.1642 |
| `nc_responded` | +.1229 | +.0255 | +.0407 |
| `nc_outcome` | +.0165 | +.0117 | +.0137 |
| `nc_agree` | −.0496 | +.0408 | +.0122 |
| `cw_community` | −.0058 | +.1269 | +.1217 |
| `hashtagwars_verdict` | −.0902 | +.2025 | +.1773 |
| `cap_finalist` | −.0117 | +.0318 | +.0041 |
| `cap_crowd` | −.0032 | −.0277 | +.0086 |
| `jokes_community` | +.0246 | +.0580 | +.0169 |
| `mathse_accepted_verdict` | −.0243 | +.0702 | +.0144 |
| `mathse_vote_score` | −.0134 | +.0431 | +.0206 |
| `aops_curation` | −.0007 | +.0101 | +.0116 |
| `code_v3` | +.0677 | +.0001 | +.0517 |
| `press_verdict` | +.1014 | +.0934 | +.0272 |

Two things stand out and are worth stating plainly. **First, Δ_interact is NEGATIVE on 10
of 16 cells** — the HistGB bank aggregator is, on most cells, *worse* than the logistic one
at these sample sizes. On this evidence "nonlinear interactions among articulated criteria"
is not a general source of recoverable signal; it is positive only where n and group count
are large (nc_responded +.123, press +.101, code +.068). Δ_interact should not be reported
as a program-wide quantity without that split. **Second, Δ_beyond — the only quantity
eligible to be called taste — ranges from −.028 (cap_crowd, where the bank beats the dense
reader) to +.229 (peer_revealed), a spread far larger than any methodological delta in this
table.** Whatever taste is, its size is a property of the cell, not of the instrument.

### V3 results (13 of 14 possible cells) — all intervals re-gated

Every V3−VA_nl interval below comes from the harvester's **exact-identity gate**
(AUC(y, `<slug>_va_nl_oof_seed0*.npy`) must reproduce the cell's published
`nonlinear.VA.seed_aucs["0"]` to 1e-6) and, on the N&C cells, from the **`_v2`
regenerated arrays with ids** — which pass at full precision (observed == published
exactly). `null` means the script refused to difference against an ungated vector.

| cell | n_E | V3 on E | V3 − T [95% CI] P | V3 − VA_nl [95% CI] P |
|---|---:|---:|---|---|
| `peer_curation` | 1571 | **.6386** | **+.0450 [+.0223,+.0672] 1.00** | **+.0495 [+.0123,+.0866] 1.00** |
| `peer_revealed` | 478 | **.8882** | +.0040 [−.0072,+.0151] .76 | **+.1642 [+.1225,+.2079] 1.00** |
| `press_verdict` | 605 | .7714 | −.0030 [−.0369,+.0449] .46 | +.0272 [−.0114,+.0650] .93 |
| `jokes_community` | 3163 | .7411 | −.0057 [−.0177,+.0044] .16 | **+.0169 [+.0062,+.0263] 1.00** |
| `mathse_accepted_verdict` | 2600 | .6303 | −.0136 [−.0274,+.0008] .04 | +.0144 [−.0058,+.0349] .93 |
| `mathse_vote_score` | 2326 | .6460 | −.0077 [−.0226,+.0074] .15 | +.0108 [−.0108,+.0327] .84 |
| `nc_outcome` | 1417 | .6196 | −.0041 [−.0313,+.0248] .39 | **−.0234** [−.0645,+.0170] .13 |
| `nc_agree` | 1009 | .5929 | −.0105 [−.0494,+.0299] .31 | **−.0205** [−.0651,+.0256] .19 |
| `nc_responded` | 1904 | .7950 | −.0217 [−.0428,−.0000] .03 | +.0049 [−.0229,+.0337] .61 |
| `peer_verdict` | 1236 | .7544 | −.0223 [−.0400,−.0055] .01 (same-rows T .7761) | **+.0738 [+.0461,+.1032] 1.00** |
| `hashtagwars_verdict` | 924 | .7063 | −.0253 [−.0628,+.0163] .11 | null — cell has no mean3 bank vector |
| `cap_finalist` | 1055 | .6707 | +.0583 .99 | +.0041 vs fullfit .6666 (row-level) |
| `cap_crowd` | 2190 | .6303 | +.0749 | +.0086 vs fullfit .6217 (row-level) |
| `cw_community` | (mirror) | .6912 | −.1136 vs the 77K-row plateau | −.0106 vs terminal bank |

**One correction, and one correction-of-a-correction:**
1. **The N&C V3 − VA_nl intervals are NEGATIVE, not positive** (nc_outcome −.0234,
   nc_agree −.0205; both n.s.). They were `null` before the `_v2` arrays landed and I had
   generalised from the cells that did have intervals. This correction stands.
2. **I wrongly retracted `peer_verdict` V3 − VA_nl = +.0738 [+.0461,+.1032] P=1.00. It is
   valid and is REINSTATED.** My re-harvest returned `null` with "no bank index rule
   registered", and I took that at face value. It was a harvester defect, not a data fact:
   `peer_verdict` is a **single-fit cell** whose Layer-1 JSON stores `nonlinear.VA.auc`
   rather than `seed_aucs`, and it has only a seed0 array (no mean3). The version I ran
   (a) refused any cell lacking a mean3 array even though it gates on seed0, and (b) could
   not read the single-fit reference schema. Both are now fixed, and the gate passes at
   **exact identity**: observed .6876125399665395 vs published .6876125399665395 (tol 1e-6),
   which I verified independently against `peer_verdict_layer1.json`. Lesson: a `null` from
   a gate is only trustworthy if you have checked that the gate could have fired at all.

**Revised reading — "V3 sits below its dense parent and above its bank" holds on the first
half, and on the second everywhere except the truncation-hit cells.** V3 − T is negative on
8 of 10 cells with a gated interval (positive only on `peer_curation` +.045 and
`peer_revealed` +.004; only `peer_verdict` −.022 and `mathse_accepted` −.014 clear P<.05).
V3 − VA_nl is positive on **7 of 9** and **negative on exactly the two N&C cells**
(nc_responded, the third, ties at +.005). I originally attributed this to the k=20 block
displacing document text on those cells, and flagged it as a testable prediction. **The
matched-raw control has since REFUTED that explanation** — see the refutation box below.
The negative N&C deltas are a real property of those cells, not a text-budget artifact.

Two cells have no gateable bank vector at all and their `null` is irreducible without
recomputing that cell's bank OOF: `hashtagwars_verdict` (neither a mean3 nor a seed0 array
exists on disk) and the two caption cells, whose bank comparisons are row-level fullfit@E
figures inherited from the 08-07 note rather than group-level gated intervals.

**Two cells now have V3 at or above BOTH parents, and they are the two peer-review cells
at opposite ends of the signal range.** `peer_curation` (+.0450 over T, +.0495 over the
bank, both P=1.00) is the grid's weakest-signal cell — where the text is least learnable
and the articulated criteria are worth most. `peer_revealed` (**.8882**, +.0040 over T at
P=.76, +.1642 over the bank at P=1.00) is the grid's STRONGEST cell (T .884), and there V3
merely matches the dense reader rather than beating it — the +.004 is well inside noise.
So the honest generalisation is not "V3 wins on peer review" but: **V3 tracks whichever
parent is stronger, and only adds real value where the dense reader is weak.** Note also
that on peer_revealed V3 (.8882) edges the matched-strength D1b combiner (.8805) — putting
the criteria in the prompt beats appending the dense score to the bank, on the one cell
where both arms exist and the dense reader dominates.

### ⛔/✅ DISPLACEMENT: REFUTED where I predicted it, CONFIRMED where I did not

I predicted that V3's negative V3−VA_nl deltas on the N&C cells were an artifact of the
criterion block displacing document text, and that at matched budget V3 would beat a raw
arm there. The control (four raw-text-only twins, same rows/splits/recipe, `max_length =
1024 − block_tokens`, group bootstraps on docket/company) says **no on N&C and yes on
press** — the mechanism is real but I attached it to the wrong cells.

| cell | budget | T (1024) | raw_matched | V3 | **V3 − raw_matched** | **raw_matched − T** |
|---|---:|---:|---:|---:|---|---|
| `nc_agree` | 769 | .6034 | **.6071** | .5929 | −.0143 [−.0445,+.0163] .17 | +.0037 [−.0336,+.0437] .57 |
| `nc_outcome` | 775 | .6238 | **.6264** | .6196 | −.0068 [−.0316,+.0175] .28 | +.0026 [−.0235,+.0294] .60 |
| `nc_responded` | 764 | .8167 | **.8152** | .7950 | **−.0202 [−.0374,−.0030] .01** | −.0015 [−.0190,+.0161] .43 |
| `press_verdict` | 805 | .7744 | .7526 | **.7714** | +.0188 [−.0162,+.0531] .86 | **−.0218 [−.0509,+.0024] .03** |

**On N&C the displacement story fails at its first limb, which makes the second moot.** If
the block's deficit were displacement, cutting the document to the block's budget should
itself have cost signal. It cost nothing: `raw_matched − T` is +.0037 / +.0026 / −.0015 —
all null, two the wrong sign. About 30% of N&C rows lose their tail at the reduced budget
and the reader does not miss it, so **N&C signal is front-loaded and the ~250 displaced
tokens were carrying nothing.** With no displacement cost to refund, V3 does not close: it
stays below raw at matched budget on all three, and on `nc_responded` the matched control
makes V3 look **worse** — −.0202, P=.01, **the only interval in the table that excludes
zero** — against a V3−VA_nl of +.0049 that had looked mildly positive. On N&C the criterion
block is a mild net cost that survives full budget-matching.

**`press_verdict` is the genuine exception and it validates the mechanism where it actually
applies.** It is the one cell already truncating before the block (.312 → .451); there the
lost text really did cost signal (−.0218, P=.03 — the strongest displacement evidence in the
set), and once budget-matched the block more than pays for itself (+.0188, with V3 .7714
nearly recovering full-budget T .7744).

**Net:** displacement is a **long-document phenomenon**, not the explanation for the
negative N&C deltas. My error was generalising a real mechanism from a truncation *rate*
(~.30 with the block) without checking whether the displaced text carried anything — on
N&C it did not, on press it did. The right diagnostic is `raw_matched − T`, not the
truncation rate, and that is the number to demand of any future cell.

Consequences applied: the truncation caveat is **retired for the three N&C cells** (their
V3 ≤ T results are interpretable at face value) and **upheld for `press_verdict`**, whose
V3−T tie is genuinely pessimistic and whose true block contribution is the +.0188.

Provenance: no matching assertion failed on any cell — eval/test row sets byte-identical to
the V3 arms, raw text recovered two independent ways, budget equivalence verified by
reproducing each V3 arm's with-block truncation rate at the reduced budget (Δ .0001–.0088
vs a .01 tolerance), and T reproduced its published `T_same_rows_at_E` with delta **exactly
0.0** on all four. For nc_agree/nc_outcome the only non-path difference between the V3 and
raw-matched trainer configs is `max_length`.

### k-sensitivity probe — k=20 is not better than k=10 anywhere it has been tested

`results/v3_k_sensitivity_hashtagwars.json`. Identical rows, identical recipe, only the
block size differs:

| cell | k10 | k20 | k10 − k20 [95% CI] P(>0) |
|---|---:|---:|---|
| `hashtagwars_verdict` | **.7142** | .7063 | +.0079 [−.0152,+.0275] P=.68 (8 groups — wide by construction) |
| `jokes_community` | **.7435** | .7411 | +.0024 [−.0016,+.0068] P=.89 (10 groups) |
| `cap_finalist` (audit confirm cell) | **.6707** | .6431 | +.0276, P(k20>k10)=.043 |
| `cap_crowd` (audit sandbox cell) | .6190 | **.6303** | −.0113, P(k20>k10)=.98 |

**Four cells, three favour k=10, one favours k=20, and the only decisive pair disagree with
each other.** The audit's production recommendation rests entirely on its sandbox cell. The
defensible statement is that **k ∈ {10,20} is a wash** and the ±.03 uncertainty this grid
attaches to every V3 number is the right way to carry it. Both dedicated probes (hashtagwars
+.0079 P=.68, jokes +.0024 P=.89) lean k=10 but neither is decisive on its own; what IS
decisive is that the audit's only significant result points the opposite way from its own
confirm cell.

### D1b results (matched training strength) — run COMPLETE, 4/4

`results/direction1b.json`.

| cell | T on E | bank fullfit@E | combiner [bank] | **D1b [bank+T]** | D1b − bank-alone | D1b − bank fullfit | gate |
|---|---:|---:|---:|---:|---|---|---|
| `nc_outcome` | .6238 | .6430 | .6390 | **.6567** | +.0176 [−.0005,+.0347] P=.97 | +.0137 [−.0044,+.0306] P=.93 | PASS |
| `nc_agree` | .6034 | .6134 | .6098 | **.6256** | +.0158 [−.0112,+.0426] P=.87 | +.0122 [−.0149,+.0382] P=.81 | PLATFORM-CAVEATED |
| `peer_curation` | .5936 | .5891 | .5787 | **.6054** | +.0267 [+.0043,+.0504] P=.99 | +.0163 [−.0069,+.0404] P=.91 | PASS |
| `peer_revealed` | .8842 | .7240 | .7174 | **.8805** | +.1631 [+.1224,+.2072] P=1.00 | +.1565 [+.1157,+.1991] P=1.00 | PASS |

**All four agree in sign: at matched training strength the dense scalar adds to the
full-strength bank on every cell tested** — +.012 to +.018 where the two instruments are
close, and +.157 on `peer_revealed`, where the dense reader is far stronger (T .884 vs bank
.724) and the combiner correctly follows it (.8805 ≈ T, i.e. it recovers the dense model
without being dragged down by the weaker bank). Three of four clear P≥.90 against the bank;
all four clear P≥.87 against the bank-alone control.

The one asymmetry: on `peer_revealed` the combiner lands *at* T, not above it (.8805 vs
.8842) — fusion there is "no loss from carrying the bank", not a gain over the dense model.

### ‖ `code_v3`: the pooled row reverses within-repo — quote the within-repo readout

Flagged `POOLED_DO_NOT_QUOTE: true` because the cell's splits contradict each other (the
strict list already forbids pooled numbers here). Pooled says VAT_nl .7449 beats BOTH
parents (T .6933, VA_nl .6932) — the largest apparent complementarity in the grid.
**Within repo (the honest readout) it does not**, and both held-out splits agree:

| within-repo (n-weighted) | T | VA_nl | VAT_nl | Δ(VAT−VA_nl) | Δ(VAT−T) |
|---|---:|---:|---:|---|---|
| eval (71 repos ≥20 rows) | .7093 | .6466 | **.6955** | +.0489 jk-CI [+.020,+.078], Wilcoxon 3.6e-4 | −.0137 [−.042,+.015] p=.39 |
| test (73 repos) | .6540 | .5948 | **.6312** | +.0365 [+.007,+.066] p=.011 | −.0227 [−.050,+.005] p=.049 |

So within-repo the fusion gain over the bank is real, sizeable and replicated (+.049 /
+.037), but the fusion arm sits **at or below** the dense model. The pooled "beats both
parents" reading is a repo-composition artifact. Both readouts live in
`results/vat_stack_code_v3.json`; only the within-repo one may be quoted.

### ⚠ The V3 production recipe's k=20 choice FAILS its confirm cell

The 08-09 V3 audit recommended **k=20** over k=10 on the strength of the SANDBOX cell
cap_crowd (+.0113 on E, P(>0)=.98). Its designated CONFIRM cell, cap_finalist, had a
`CONFIRM_*` placeholder because the audit agent's session ended before harvest — but the
checkpoint had finished on sk3 at 2026-08-08T00:56 and was sitting unharvested. Harvested
with the audit's own script:

| cap_finalist arm | eval | test | E (n=1,055) | Δ vs k10 [95% CI], P(>0) | Δ vs bank fullfit@E (.6666) |
|---|---:|---:|---:|---|---|
| k10 (aug_a, the shipped baseline) | .6775 | .6658 | **.6707** | — | +.0041, P=.59 |
| **k20 (the "production" recipe)** | .6737 | .6121 | **.6431** | **−.0276 [−.058,+.003], P(>0)=.043** | −.0235, P=.14 |

**k=20 is NOT a confirmed improvement.** Sandbox +.011 (P=.98), confirm cell −.028 (P=.04)
— opposite signs at comparable magnitude. Consequences adopted: the grid still builds at
k=20 (it is the written recipe and the truncation budget is tuned to it), but **every V3
number here carries a ±.03 k-choice uncertainty**; k=10 probe twins were queued on the two
cheapest short-text cells so the grid contains its own internal check
(`hashtagwars_verdict_k10` has trained; `jokes_community_k10` never started). The audit
note's §5 should be amended by whoever next touches it — I did not edit another agent's
note, so the correction lives here and in `results/vat_fullgrid_cap_finalist.json`.

### ⚠ LANDMINE — `results/<cell>_va_nl_oof_*.npy` are unusable as per-row vectors

Tempting shortcut: read the saved full-population bank OOF array off disk and pair it with
the population. **It does not reproduce its own published VA_nl under any row order I could
construct**, including the cell's own adapter order (nc_outcome .5083 vs a published .6102;
nc_agree .5041 vs .5844 — i.e. essentially random permutations), and there is no index
stored beside the arrays to recover it. Two cells (peer_curation .5636 vs .5588,
peer_revealed .7712 vs .7667) *do* reproduce, so the corruption is cell-specific — which is
exactly why a per-cell **reproduction gate** is the right discipline rather than a blanket
rule. All fullfit@E values in the ledger come from a from-scratch recompute; the gate
outcome, including failures, is recorded beside every number in `results/fullfit_at_E.json`.
Both GPU workers were warned mid-run and instructed to emit a `null` CI with a stated reason
rather than an ungated one — which is why three N&C cells carry `boot_v3_minus_va_nl: null`.
**Fixed upstream 2026-08-09:** the N&C `_v2` regeneration now ships
`nc_*_va_nl_oof_{mean3,seed0..2}_v2.npy` **with `*_oof_ids_v2.npy`**; use those.

### ⚠ LANDMINE — GroupKFold fold assignment is ARCHITECTURE-dependent

Found by the VAT-stack agent, and the most portable finding of the session. `GroupKFold`
orders group sizes with `np.argsort` at default `kind='quicksort'` — an **unstable** sort —
so ties among equal-sized groups break by sort kernel, and numpy's vectorised sorts differ
between **linux/x86-64 and macOS/arm64**. The fold assignment itself differs by machine.

Verified head-on: byte-identical bank matrices (md5-equal npz shards), md5-equal Layer-1
scripts, identical sklearn 1.8.0 with md5-equal `GroupKFold` source, numpy pinned to 2.2.6
— **the fold-membership fingerprint still differed** (aops `89e4f78d` sk3 vs `267637f7`
mac) and aops VA_lin came out .77116 on sk3 (= the ledger exactly) vs .76922 on mac. On its
first local pass `code_v3` FAILED its gate by .012; on sk3 it reproduces at 0.0e+00.

This **generalises the sklearn-version landmine** in `press_verdict_layer1.py`: reproducing
a frozen grouped-OOF number requires matching the *platform*, not just the library version.
Rule: **re-run a cell's Layer-1 reproduction on the machine that produced the published
number.** Every vat_stack JSON now records versions plus `platform_this_run` /
`platform_required`.

It touches this session's own numbers: the D1b recompute ran on macOS/arm64 while the
published values came from sk3. `nc_outcome` still passed (.6134 vs .6102); `nc_agree` sits
at .5917 vs .5844 (Δ +.0073) and is marked **PLATFORM-CAVEATED, not failed** — inside the
predicted band. **The D1b contrast is unaffected**: both combiners are fitted on the same
folds in the same process, so the paired delta and its bootstrap are internally valid; only
the absolute level is machine-sensitive. Clean fix: re-run `recompute_fullfit.py` on sk3.

### STRUCTURAL FINDING — the two fusion arms have different data requirements

- **VAT (Direction 1)** needs bank and dense predictions to overlap **only on E**. Every
  cell satisfies that by construction, which is why the VAT column covers the whole grid.
- **V3 (Direction 3)** additionally needs criterion scores on the dense model's **TRAIN**
  split — you cannot train a reader on hints it will not have, and the importance ranking
  must be train-fold-only.

Several banks were scored economically, on held-out rows only, because that is all the
Layer-1 readouts needed. Those cells **cannot have a V3 arm without new judging**:

| cell | why V3 is impossible | what would unblock it |
|---|---|---|
| `aops_curation` | the 5,202-row bank population IS the dense arm's held-out set (eval 2,510 / test 2,692, **train 0**) | Gemma-score the ~22.7K dense-train rows, or define a NEW cell with a fresh grouped split + matched-n raw control |
| `code_v3` | the 83-criterion bank was scored only on `dense_standard_v3/split/{eval,test}.csv` (11,452 rows); the 47,659-row train split has **zero** scored rows, repo-disjoint so nothing can be borrowed | same two options; this cell's dense chain also deviates from the frozen recipe (`--max_length 2048`, `--class_weight_auto`) |

Both keep their VAT rows. **Landmine recorded while establishing this:**
`datasets/math/aops/va/dense_standard/split/` exists (10,457/1,307/1,307) and looks like the
AoPS dense split, but is a **stale orphan of a superseded 13,071-row draw** — no
`rm_out_seed*`, no `eval_pass_results.json`. It shares the `sha1(problem|body)[:20]` id
scheme with the live bank, so a naive merge on `row_id` silently matches 2,307 of 5,202 bank
rows and **mislabels 1,933 of them "train"**. It fails silently rather than erroring.

### V3 BUILD — three caveats that must travel with every V3 number

**(a) The block is NOT free on long-document cells — it displaces text.** At k=20 the block
costs 249-260 tokens, pushing tokenizer truncation from ~2% raw to **.314 (nc_agree) / .298
(nc_responded) / .288 (nc_outcome)**, and from .312 to **.451 (press_verdict)**. Prepending
guarantees the block survives, but on roughly a third of those rows the document tail is cut
that the raw arm kept. This looked like a serious confound and was flagged as one. **The matched-raw control has
now resolved it, and split it: RETIRED for all three N&C cells** (their displaced text
carried no measurable signal, `raw_matched − T` ≈ 0, so a V3 ≤ T result there is
interpretable at face value) **and UPHELD for `press_verdict`** (its displaced text cost
−.0218, P=.03, and budget-matching turns its V3−T tie into a +.0188 block contribution).
See the displacement box. **The diagnostic that matters is `raw_matched − T`, not the
truncation rate** — the rate was ~.30 on both groups of cells and predicted nothing. Short-text cells are unaffected (hashtagwars .000, jokes .0001, peer_* ≤.0005,
math.SE .050).

**(b) Two peer cells are SUBSETS of their dense population.** Built on the covered subset
rather than padded with all-NA blocks (padding would make "block present" a
coverage-correlated feature — the right call). Coverage: `peer_curation` 72%,
`peer_verdict` **21%** (4,743/601/635 of 22,773/2,808/2,821). Their V3 is differenced
against same-rows T. peer_verdict's V3 row is a 21%-coverage probe, not a cell-level result.

**(c) The ≥95%-modal screen is load-bearing** — 62 columns dropped on peer_curation, 51 on
peer_verdict, 44-46 on the N&C cells. Without it the k=20 block would be spent on
near-constant columns.

The builder confirms it never used `*_va_nl_oof_*.npy` anywhere — ranking comes from the
bank matrix via train-fold permutation importance — so the .npy landmine does not touch V3.

## PENDING ARMS AND THEIR TRIGGERS

| arm | cells | trigger / status | owner |
|---|---|---|---|
| **V3 ×3 remaining** | `peer_revealed` (training), `nc_responded` + `jokes_community_k10` (never started) | GPU2's worker was killed by a session limit; its detached trainings survived. Datasets are built and verified — a fresh worker can pick them straight up from `fusion/v3_grid_queue.txt`. | next session |
| **mathlib** | mathlib_verdict | **RESOLVED 2026-08-09: STAYS OUT, now for a measured reason.** See the box below. | `claude-scaleupA-dense` |
| **3-seed T means** | mathse ×2, AoPS, code_v3 | jokes is now complete and in the ledger (.7495 eval-mean, spread .0038). code seed 1 finished this morning. Cells still on a single seed (42) inherit a .02-.04 seed band, so single-seed T differences under ~.04 are not readable. | `claude-scaleupC` |
| **T_decor** | — | **RESOLVED — NOT SHIPPING.** See the box below. | `claude-decor-battery-fable` |
| **N&C vote column** | nc_cosigning | the V8 co-signing build completes AND gets a V+A bank; would make N&C the second field with all three preference types. Out of scope here, flagged so it is not lost. | `claude-scaleupC` |

### mathlib: RESOLVED — excluded, and the seed-42 preview was misleading

The class-weighted chain finished all three seeds and ran its scoring pass, so the trigger
fired. The full 3-seed picture kills the cell rather than admitting it:

| seed | eval AUC | test AUC |
|---|---:|---:|
| 42 | .6328 | .4599 |
| 1 | .5534 | .4346 |
| 2 | .5066 | .5065 |
| **mean** | **.5643** (spread **.1262**) | **.4670** (spread .0719) |

**The dense arm is not a usable instrument on this cell.** Test AUC is at or below chance on
all three seeds, eval and test disagree by ~.10, and the eval seed spread (.126) is larger
than most of the effects this grid measures. My earlier note that seed 42 looked
"non-degenerate (eval .6330)" was right about the *collapse* — class weighting did fix the
degenerate all-positive predictor on a 94.3%-positive population — but wrong as a signal
that the cell was ready. One seed was not enough to see it; the other two were still training.

Consequence: **mathlib gets no fused ledger row.** A VAT stack is mechanically buildable
(the bank is fine: VA_lin .6683, VA_nl seeds .687/.649/.680) but every fused quantity would
be differenced against a chance-level T, so Δ_beyond and the §11 verdict would both be
uninterpretable. The honest statement for this cell is "**the dense instrument fails here**",
not a number. Revival prerequisite: a dense arm whose test AUC clears chance on a majority
of seeds.

### T_decor: RESOLVED — the arm does not ship, and the reason is worth keeping

The decorrelated-training battery's **V2' REMOVAL-OF-RELIANCE gate FAILED** (rc=3; the
gated chain stopped per its own binding spec, so no downstream number from it is
gate-certified). The failure is specific and instructive:

| sub-gate | statistic | verdict |
|---|---|---|
| (b) causal reliance — PRIMARY | plant-ablation Δ +.0023 [−.0049,+.0096]; reliance removed vs planted vanilla +.0252 [+.0154,+.0354] = **92% of reliance gone** | **PASS** |
| (a) spec-literal | auc_eval(D02) − auc_eval(R00) = −.0276 [−.0480,−.0067] | FAIL (utility direction) |
| (c) seed band | D02 .7650 vs vanilla band [.7752,.7926] | FAIL, −.0101 below band min |

**The mechanism works and the utility gate is what fails**: reweighting collapses the
planted channel's causal contribution from +.0275 to +.0023 and never amplifies reliance
(unlike GRL), but costs ~.02–.03 task AUC — about 2 SD below the vanilla seed mean, and not
explained by the effective-sample-size arithmetic (n_eff .939).

Note this is **not** the Byrd & Lipton separability escape hatch the coordinator flagged.
That guidance said a plant-*reliance* failure might be the known "weights go inert once the
data are separable" pathology, in which case the V4' length arm would carry the verdict.
Here reliance removal **passed** — what failed is production utility, which is the axis that
actually decides whether T_decor belongs in a results table. So the pre-agreed fallback
applies as written: **this grid ships without T_decor**, and the honest one-line summary is
"decorrelated training removes 92% of measured shortcut reliance at a ~.02–.03 AUC cost,
which its own pre-registered utility gate judges too expensive."


## How to resume this grid (anything unfinished picks up from here)

1. **Regenerate the table**: `python3 methods/taste_decomposition/fusion/fullgrid_ledger.py`.
   It reads whatever has landed (`results/vat_stack_*.json`, `results/v3_grid_*.json`,
   `results/fullfit_at_E.json`, plus any `results/vat_fullgrid_<cell>.json` carrying
   `"MANUAL": true`) and rewrites both the master table above and every per-cell JSON.
   Paste its stdout over the TABLE-ANCHOR block. Nothing in the table is hand-maintained.
2. **V3 arms in flight**: the two GPU workers pull from
   `methods/taste_decomposition/fusion/v3_grid_queue.txt` (append-only CLAIM/DONE/FAIL).
   Datasets are `fusion/dense_data/v3grid_<slug>/`; the builder is
   `fusion/v3_grid/build_v3_cell.py --cell <slug>`; the trainer wrapper is
   `fusion/v3_grid/train_v3_cell.sh <slug> <cuda_device>`. A cell is runnable as soon as
   its `manifest.json` exists. GPU0 and GPU2 are ledger-claimed under
   `agent=claude-vat-fullgrid` — **release them in `gpu_ledger.txt` when the chains
   finish**.
3. **Before differencing anything against a per-row bank vector**, run the reproduction
   gate (AUC(y, vector) over the full population must equal the cell's published VA_nl to
   ~.003). See the landmine section — this is not a formality; three cells fail it.
4. **Cells with no V3 arm and why**: `aops_curation`, `code_v3` (bank never scored on the
   dense train split — structural, see above). Do not "fix" these by borrowing scores
   across splits; the splits are group-disjoint by construction.

## Session handoff state (as of the last checkpoint)

- **Complete:** the VAT column on all 16 eligible cells; per-cell fused JSONs
  (`results/vat_fullgrid_<cell>.json`, 16 of them); the regenerating assembler; the
  fullfit@E bank column wherever a per-row bank vector passes the reproduction gate.
- **Running, unattended, will keep going:** two GPU chain workers on sk3 GPU0/GPU2 (two
  8B LoRA trainings per card) working the V3 queue; the Direction-1b + bank recompute on
  local CPU.
- **GPU ledger: GPU0 and GPU2 remain CLAIMED** under `agent=claude-vat-fullgrid` because
  the chains are mid-flight. **Whoever sees the chains finish must post the RELEASE lines**
  — that is the one piece of hygiene this session cannot close itself.
- Nothing was killed that this session did not start; the only process terminated was my
  own recompute (PID 43110) to extend it, and it was relaunched.

## Landing log (append-only; checkpoint before every new launch)

- **2026-08-09 morning** — T_DECOR RESOLVED: gate table filled, **V2' FAILED (rc=3)**, chain
  stopped per spec. Reliance removal passed (92% of the planted channel's causal
  contribution gone) but the utility sub-gates failed (−.0276 eval, ~2 SD below the vanilla
  seed band). This is NOT the Byrd–Lipton separability escape hatch — that applies to a
  plant-*reliance* failure, and reliance passed. Pre-agreed fallback applies: **the grid
  ships without T_decor.** Full reasoning in the T_decor box.
- **2026-08-09 morning** — SELF-INFLICTED, RECOVERED: a scripted edit meant to update the
  T_decor row in the PENDING ARMS table matched the identically-named row in the TERMS table
  first and spliced out everything between them — Protocol, Cell inventory, the master
  ledger, and every boxed finding (~330 lines). The landing log survived. Rebuilt the whole
  middle from artifacts rather than memory: tables regenerated from
  `fullgrid_ledger.py`, every number re-read from `results/*.json`. Lesson worth keeping:
  **anchor scripted splices on unique text, and prefer regenerating a section to editing
  across it** — the reason this was recoverable at all is that the ledger is machine-
  generated and every finding is backed by a JSON on disk.

- **2026-08-08 09:02Z** — session opened. Read §12 + §0 of the design note, the strict
  list, the fusion-directions protocols, and the V3 audit (its production recipe HAD
  already landed: k20 / names-only / prepend-on-long-text — adopted for every V3 arm
  in this grid). Harvested the 8 delivered VAT stacks + the 2 caption cells. Claimed
  sk3 GPU0 and GPU2 (only two 0-MiB cards; GPU7 acquired a 35 GB co-tenant between
  the survey and the claim, GPU4 holds a 115 GB co-tenant — neither touched).
- **2026-08-08 09:1xZ** — LANDING: rescued the V3 audit's orphaned `cap_finalist_k20`
  checkpoint (finished 00:56Z, never harvested) and harvested it with the audit's own
  script. **k=20 fails its confirm cell** (see the boxed result above). Wrote
  `methods/taste_decomposition/results/vat_fullgrid_cap_finalist.json`. Sent a course
  correction to the V3-builder agent: keep k=20 primary, add k=10 sensitivity arms on the
  two cheapest short-text cells only.
- **2026-08-08 09:2xZ** — launched the two workstreams: (a) VAT stacks for the six
  cells the earlier stack agent did not cover (jokes, math.SE ×2, AoPS, press, code),
  CPU, reusing the frozen `direction1_mirror.py` engine by import; (b) the generic V3
  dataset builder + per-cell training wrapper for the 13 non-caption cells. Armed a
  tracked watcher for the four external triggers (T_decor battery note, mathlib
  class-weighted dense, jokes dense seed 2, code dense seed 1).
- In-flight on sk3 that this grid DEPENDS on but does NOT own (other agents' PIDs,
  never touched): mathlib class-weighted dense seed 42 (GPU3), reddit-jokes dense
  seed 2 (GPU1), code-review v3 dense seed 1 (GPU6, 6 h in), N&C **co-signing** dense
  seed 42 (GPU5 — this is the V8 build that would add N&C's missing vote column to the
  grid; not eligible this session, flagged for the next one).
- **2026-08-08 09:4xZ** — TRIGGER FIRED: **mathlib class-weighted dense seed 42 finished
  training.** Non-degenerate this time (epoch-2 eval precision .949 / recall .909 —
  contrast the collapsed unweighted run on the 94.3%-positive population), **eval AUC
  .6330** and still rising at epoch 2. This retires the never-quote .580/.473 pair as
  the mathlib dense number. It does NOT yet make mathlib a ledger row: the chain
  (owner `claude-scaleupA-dense`, GPU3) has moved on to seed 1 and no scoring pass has
  written `preds_{eval,test}.csv`, so there is no per-row dense column to append. mathlib
  stays PENDING with trigger = "class-weighted chain writes per-row preds"; I did not
  touch that chain.
- **2026-08-08 09:5xZ** — launched the two GPU chain workers (`v3chain-gpu0`,
  `v3chain-gpu2`) against a shared append-only work queue
  (`methods/taste_decomposition/fusion/v3_grid_queue.txt`) rather than a fixed
  cell→GPU assignment, so a cheap cell finishing early pulls the next one instead of
  idling behind an expensive neighbour. They poll for the builder's per-cell manifests.
  Also wrote `methods/taste_decomposition/fusion/fullgrid_ledger.py` — the ledger
  assembler that regenerates this note's master table and the per-cell
  `results/vat_fullgrid_<cell>.json` files from whatever arms have landed, so every
  checkpoint is a re-run rather than a hand-edit.
- **2026-08-08 10:2xZ** — LANDMINE (see the boxed section above): the saved
  `results/*_va_nl_oof_*.npy` per-row bank arrays do not reproduce their own published
  VA_nl under ANY row order I could construct, including the cell's own adapter order
  (nc_outcome .508 vs .610). Discarded them, warned both GPU workers mid-run with a
  mandatory reproduction gate, and started a from-scratch recompute of the
  full-population bank OOF under the frozen Layer-1 protocol — which both fixes the
  fullfit@E column and regenerates a per-row bank vector the V3 arms can safely
  difference against.
- **2026-08-08 10:2xZ** — first V3 datasets shipped by the builder:
  `v3grid_hashtagwars_verdict` (+ its k10 probe twin) and `v3grid_press_verdict`. Both
  manifests carry split-file sha256s, per-split join coverage, the ≥95%-modal column
  screen, tokenizer truncation rates with and without the block, and the k-caveat.
  hashtagwars: APPEND (0% of rows reach max_len; median 35 → 242 tokens with the
  207-token block), 10 of 50 bank columns dropped as ≥95%-modal, top criterion
  `v_char_count`. `v3chain-gpu2` claimed hashtagwars within two minutes of the manifest
  appearing — the shared-queue design worked as intended.
- **2026-08-08 10:4xZ** — first two V3 trainings VERIFIED RUNNING (not merely "launched"):
  `nvidia-smi` shows 28.6 GB on GPU0 and 28.6 GB on GPU2, and `ps` shows
  `train_reward_model.py` on `v3grid_press_verdict` (GPU0, seed 42, max_len 1024,
  selection_split eval) and on `v3grid_hashtagwars_verdict` (GPU2, same recipe). Both
  match the frozen 8B LoRA recipe flag-for-flag. Also replaced this note's hand-written
  ledger table with the generated one and wrote the two caption rows as `MANUAL: true`
  per-cell JSONs so the assembler carries them forward without re-deriving them.
- **2026-08-08 11:0xZ** — fullfit@E diagnostic finished. The saved-array gate SPLITS by
  cell: `peer_curation` (.5636 vs .5588), `peer_revealed` (.7712 vs .7667) and
  `cw_community` (.6651 vs .6652) reproduce their published VA_nl and are USABLE;
  `nc_outcome` (.5083 vs .6102) and `nc_agree` (.5041 vs .5844) do not and are
  DISCARDED. So the corruption is cell-specific, not universal — which is exactly why a
  per-cell gate, rather than a blanket rule, is the right discipline. Gated values written
  to `results/fullfit_at_E.json` with the gate outcome recorded next to every number,
  including the failures.
- **2026-08-08 11:0xZ** — **first §11 flag: `peer_curation`.** It appears only against the
  full-strength bank (see the boxed section under the ledger). Rather than let a
  training-strength mismatch decide a standing-rule verdict, I killed my own recompute
  job (PID 43110, my process, no one else's) and relaunched it extended with the
  **Direction-1b two-stage arm** — full-strength bank OOF + dense scalar in a combiner
  fitted on E, against a bank-alone combiner control — so the flag gets adjudicated at
  matched footing on the same single expensive cell load rather than a second one.
- **2026-08-08 11:0xZ** — the shared queue is behaving exactly as designed: GPU0 claimed
  `hashtagwars_verdict`, saw GPU2's earlier claim on re-read, yielded it and took
  `press_verdict`; same yield-and-advance on `peer_curation` → `jokes_community`. Ten V3
  dataset dirs built (eight cells + the two k10 probe twins).
- **2026-08-08 11:3xZ** — throughput confirmed: both claimed cards are running TWO 8B
  LoRA trainings each (62.6 GB / 100% util on GPU0 and GPU2), four V3 arms in flight
  (`press_verdict` + `jokes_community` on GPU0; `hashtagwars_verdict` + `peer_curation`
  on GPU2). Stacking doubles the grid's V3 throughput on the two cards we were able to
  claim without co-tenanting anyone.
- **2026-08-08 12:0xZ** — LANDING: `mathse_accepted_verdict` VAT stack (first of the six
  new-build cells). n_E 2,600 / 992 questions, `impute_perfold` family matching its own
  Layer-1 lineage, joined to the dense arm BY ID with assertions. It clears the VA
  reproduction gate on both aggregators (VA_lin .6353 vs published .6320; VA_nl .6331 vs
  .6320) and carries its OWN freshly-computed fullfit@E (.6179) — the right pattern, and
  I extended the ledger assembler to prefer a vat_stack's self-computed fullfit@E over
  the shared file. Row: T .6439, VA_nl .5736 (E-refit) / .6179 (full strength), VAT_nl
  **.6252**, so §11 PASSES but only by +.0073 against the full-strength bank — a thin
  pass, and thinner than the +.0616 the E-refit comparison advertises. Δ_beyond +.0703.
- **2026-08-08 1x:xxZ** — **FIRST V3 ARM OF THE GRID: `press_verdict` = .7714 on E**
  (eval .7409 / test .8087), against T .7744, VAT_nl .7487 and the full-strength bank
  .7442. §11 margin improves from +.0045 (VAT alone) to **+.0272** once V3 is in.
  The interesting part is the truncation accounting: this arm read a MORE truncated
  document than T did (44.3% of E rows over max_len with the block vs 31.2% raw — the
  219-token block displaced text) and still landed within .003 of T. So on this cell the
  criteria block roughly pays for the text it costs. That is a cleaner reading than
  "V3 ≈ T" alone, and it is only available because the builder measured truncation per
  split instead of assuming it. The worker also correctly declined to add
  `--class_weight_auto` (the cell's original chain had none), keeping V3-vs-T a
  recipe-parity comparison rather than confounding it with a loss change.
- **2026-08-09 morning** — OVERNIGHT HARVEST after the GPU2 worker was killed by a session
  limit (its detached trainings survived; nothing was lost). Re-inventoried sk3: 10 of 13
  V3 dataset dirs now have `RUN_DONE` + preds. Mirrored the five unharvested ones back and
  ran them through the worker's own `harvest_v3_grid_cell.py` (which enforces the
  seed0-exact reproduction gate on the bank vector before it will emit a V3−VA_nl CI).
  **V3 column now 11/14.** New rows: `jokes_community` .7411, `mathse_accepted_verdict`
  .6303, `mathse_vote_score` .6460, plus `peer_curation` .6386 / `peer_verdict` .7544 /
  `nc_agree` .5929 harvested by the worker before it died. Still training: `peer_revealed`.
  Never started: `jokes_community_k10`, `nc_responded`.
- **2026-08-09 morning** — the N&C **`_v2` OOF regeneration has landed** as
  `results/nc_{agree,outcome,responded}_va_nl_oof_{mean3,seed0,seed1,seed2}_v2.npy` **plus
  `*_oof_ids_v2.npy`**. The ids are the fix for this session's headline landmine: the
  original arrays had no index, which is exactly why they could not be re-joined. Any
  future N&C per-row bank work should use the `_v2` files and their id arrays, and can then
  produce the V3−VA_nl intervals that the three N&C cells currently carry as `null`.
- **2026-08-09 0x:xxZ** — **DIRECTION-1b RUN COMPLETE (4/4) AND THE GRID'S ONLY §11 FLAG IS
  CLEARED.** `peer_curation` D1b .6054 > bank .5891 (+.0163 P=.91; +.0267 P=.99 vs the
  bank-alone control) — the flag was an E-refit handicap, exactly as the conditional
  recording predicted, and no Fable audit needs to fire. `peer_revealed` D1b .8805 vs bank
  .7240 (+.1565, P=1.00). **The AUTO-FABLE-AUDIT list for the whole grid is now empty.**
  All four cells agree in sign, which upgrades the E-refit caveat from a caveat to a
  finding: comparing an E-refit fused arm against a full-strength bank reads systematically
  pessimistic and can flip the sign of the §11 verdict.
- **2026-08-09 0x:xxZ** — VAT-stack agent closed out: 6/6 cells delivered, 6/6 through the
  VA reproduction gate, none blocked, and it surfaced the **architecture-dependent
  GroupKFold landmine** boxed above — the most portable finding of the session. It also
  documented how it recovered ids: the dense-standard preds CSVs carry only
  `(judgement, prob, group)` with **no id**, so ids came from each dense build's own
  `split/{eval,test}.csv` by row order with judgement AND group asserted elementwise first;
  independent confirmation is that every recomputed per-split T matches
  `eval_pass_results.json` exactly on all six cells. Join tables in `fusion/dense_joins/`.
- **2026-08-09 0x:xxZ** — SECOND D1b CELL: `nc_agree` .6256 vs bank .6134 (+.0122), same
  sign and magnitude as nc_outcome. Its recompute lands .5917 vs published .5844 — recorded
  **PLATFORM-CAVEATED rather than failed**, because that Δ +.0073 is inside the band the
  architecture landmine predicts, and because the D1b contrast is fitted on identical folds
  within one process and so is unaffected by it.
- **2026-08-09 0x:xxZ** — **FIRST DIRECTION-1b CELL, and it vindicates the design of the
  adjudication.** `nc_outcome`: the from-scratch bank recompute reproduces its published
  VA_nl (.6134 fullpop vs .6102 published, **GATE PASS**) — proving the recompute route is
  sound and re-confirming that the discarded `.npy` for this same cell (.5083) was simply
  corrupt, not a hard measurement disagreement. fullfit@E is **.6430**, well above the
  E-refit .6121. At that honest bank level the E-refit VAT_nl (.6227) would have FAILED
  §11 — but the matched-strength D1b combiner reaches **.6567**, i.e. **+.0176
  [−.0005,+.0347] P=.97 over the bank-alone combiner** and **+.0137 [−.0044,+.0306] P=.93
  over the full-strength bank**. So on this cell fusion genuinely adds once both sides are
  fitted on the same rows, and the apparent failure was an artifact of the E-refit
  handicap. Added a **D1b column** to the ledger and made §11 use it where present.
  This is exactly the correction the `peer_curation` flag is waiting on.
- **2026-08-08 1x:xxZ** — **VAT COLUMN COMPLETE.** `jokes_community` was the last one
  (n_E 3,163 / 10 LDA topics, T .7469 — the 3-seed arm — VA_nl .6888 E-refit / .7242 full
  strength, VAT_nl **.7375**, §11 PASS +.0133). All 16 eligible cells now carry
  VA_lin / VA_nl / VA_nl@full / T / VAT_lin / VAT_nl with group-level bootstrap CIs on
  (VAT−VA_nl) and (VAT−T). Note jokes has only **10 grouping units** (LDA topics), so its
  CIs are narrow-looking for the wrong reason — the same coarse-group caveat as
  hashtagwars (8 contests) and press (45 companies).
- **2026-08-08 1x:xxZ** — **SECOND V3 ARM: `hashtagwars_verdict` = .7063 on E**
  (eval .6517 / test .7537), vs T .7315, VAT_nl .6454, bank .5290. V3 − T = −.0253
  [−.063,+.016], P(>0)=.11 — i.e. **V3 recovers most of the dense reader's signal from
  criteria alone** and beats the VAT stack by +.061 on the same rows, lifting this cell's
  fusion gain over the bank from +.116 to **+.177**. Two honest limits on that reading:
  the group bootstrap resamples only **8 hashtag contests**, so every CI on this cell is
  wide by construction; and the eval/test halves disagree by .10, which is the same
  8-group coarseness showing up as split instability. The worker correctly emitted
  `boot_v3_minus_va_nl: null` rather than an ungated bank difference — the reproduction
  gate I mandated after the `.npy` landmine did its job on its first real test.
- **2026-08-08 1x:xxZ** — the Direction-1b recompute crashed after ~21 min on a
  `group_paired_boot()` arity slip (my bug — the frozen helper takes `groups` as its 4th
  positional). The expensive per-cell artifacts survived because the `np.save` sits before
  the 1b block; fixed the call, wrapped 1b in its own try/except so a stats failure can
  never again discard a completed bank recompute, and relaunched.
- **2026-08-08 1x:xxZ** — LANDING: `press_verdict` VAT stack. T .7744, VAT_nl **.7487**,
  §11 PASS but by only **+.0045** over the full-strength bank (.7442) — the thinnest pass
  in the grid. This cell shows the E-refit distortion at its most extreme: E has just
  **45 companies**, so the E-refit bank collapses to .6810 while the same bank at full
  training strength reads .7442. Quoting the +.0786 "VAT beats bank" figure off the
  E-refit would be a ~17× overstatement of the real +.0045 margin. **Do not cross-quote
  this cell's Δ_beyond (+.0934 here, E-restricted) against the earlier honest-T note's
  +.0486** — that one is T .7497 vs the FULL-population VA_nl .7011; different
  populations, different numbers, same cell.
- **2026-08-08 1x:xxZ** — V3 BUILD STAGE COMPLETE (11 built / 2 structurally blocked), with
  the three caveats boxed above. Relayed the subset and truncation caveats to both GPU
  workers as harvest instructions, not footnotes: same-rows T for the two subset cells, and
  an explicit "not interpretable without a matched-n raw control" line in the four
  truncation-affected cells' result JSONs.
- **2026-08-08 1x:xxZ** — TRIGGER: reddit-jokes dense **seed 2** finished, so
  `jokes_community` now has all three seeds — eval .7470 / .7507 / .7508 (mean .7495,
  spread .0038, unusually tight) and test .7236 / .7254 / .7283 (mean .7258). Told the VAT
  agent to treat it as a 3-seed cell (per-seed + mean-probability ensemble, ensemble as the
  ledger row) rather than redoing seed 42.
- **2026-08-08 12:3xZ** — LANDING: `mathse_vote_score` VAT stack. n_E 2,326 / 1,140
  questions, VA reproduction gate PASS, own fullfit@E .6352. T .6538, VAT_nl **.6523**,
  §11 PASS by +.0171 against the full-strength bank. Note the pattern now visible across
  both math.SE cells: the E-refit comparison flatters fusion by ~4-5× (+.0511 vs +.0171
  here; +.0616 vs +.0073 on the accepted-verdict cell). **Reading the fusion gain off the
  E-refit bank alone would systematically overstate it, on every cell in this grid** —
  which is the practical reason the `VA_nl@full` column exists.
- **2026-08-08 12:1xZ** — coordinator guidance received on how to READ the T_decor gate
  (Byrd & Lipton separability pathology; V4' length arm carries verdict weight for
  realistic channels; ESS + clip rate mandatory; AFR as a stated alternative). I do not
  own the decorrelation battery — `claude-decor-battery-fable` does — so I applied it
  where it bears on this grid: the T_decor **acceptance criterion in the pending-arms
  table is amended** (see that row), and any T_decor row that lands under the
  "plant-fail + length-pass" branch must carry the stated-scope caveat in its per-cell
  JSON rather than being reported as an unqualified arm.
- **2026-08-08 11:2xZ** — the T_decor battery note landed but is not a verdict: the file
  exists and reads `Status: RUNNING`, with only the V1 EXPLOIT gate resolved (PASS) and
  the gate table still `[PENDING]`. My original watcher was armed on the file's
  EXISTENCE, which is the wrong condition — re-armed on the gate table actually filling
  in. Recording the miss because it is the kind of trigger that silently reads as "done".

- **2026-08-09 tail** — closing sequence per coordinator: (1) relaunched the two
  never-started V3 arms on my own already-claimed cards (`nc_responded` on GPU2,
  `jokes_community_k10` on GPU0 — stacking a 2nd/3rd job per card rather than claiming a
  new one, since every other GPU had a co-tenant; ledger line appended, both verified
  running with the trainer's own init line); (2) harvested `nc_outcome` V3 (.6196) and the
  `hashtagwars_verdict_k10` probe (.7142); (3) mathlib RESOLVED as an exclusion once its
  3-seed scoring pass exposed a chance-level test AUC; (4) **re-gated every V3−VA_nl
  interval** against the `_v2` regenerated arrays, which produced two corrections to the
  morning numbers — one of which I then had to un-correct: the two N&C deltas are indeed
  negative, but peer_verdict's +.0738 was NOT ungated, and my retraction of it was itself
  the error (see the V3 box); (5) computed the k10-vs-k20 paired probe on identical rows.
- **Reading discipline note for whoever picks this up:** the harvester writes
  `boot_v3_minus_VA_nl` (capital VA), not `boot_v3_minus_va_nl`. I read the lowercase key
  for several hours and concluded the intervals were missing when they were present. Any
  script consuming these JSONs should assert the key exists rather than defaulting to
  `None`.
- **2026-08-09 tail** — `peer_revealed` V3 landed: **.8882** (eval .8715 / test .9030),
  +.0040 over T (P=.76, inside noise) and +.1642 over the bank (P=1.00). It also edges its
  own D1b combiner (.8805). Twelve of fourteen possible V3 arms are now in; `nc_responded`
  and `jokes_community_k10` remain in training.
- **2026-08-08 (grid close)** — GPU0 worker closed out with all five of its cells harvested
  and, importantly, **four defects patched in the canonical harvester** that had been
  silently nulling valid results: (1) it demanded a `mean3` array even though it gates on
  seed0, which killed every seed0-only cell — this is what produced my bogus peer_verdict
  retraction; (2) no T source registered for peer_verdict / nc_responded; (3)
  `published_seed0` couldn't read the single-fit schema (`nonlinear.VA.auc`, no
  `seed_aucs`); (4) `.loc` on peer_verdict's duplicated `ntitle` index **silently expanded
  1,236 rows to 1,238**. Ambiguous keys are now dropped with counts recorded (31 keys / 2
  rows on peer_verdict, leaving n_groups 1,234) and a >2% drop refuses the CI outright.
  `press_verdict` is bit-identical before and after every patch, and the worker's
  independent second harvester agrees with the canonical one to every printed digit on
  press_verdict and peer_verdict — so the patches are corrections, not perturbations.
- **2026-08-08 — DO NOT ALIAS:** `results/code_competitions_va_nl_oof_mean3.npy` is **not**
  `code_v3`'s bank. It is the 999-row AtCoder *code-competitions curation* cell; `code_v3`
  is the 11,452-row GitHub PR-merge cell. Differencing one against the other would
  fabricate a number. Same family of error as the AoPS stale-split decoy: a plausible
  filename that silently half-matches.
- **2026-08-08 17:30Z** — **MATCHED-RAW DISPLACEMENT CONTROL GREEN-LIT AND DISPATCHED**
  (coordinator; GPUs 1 and 3 came free). Four arms on GPU1 (ledger-claimed, verified
  0 MiB / 0% util before the claim): `nc_agree`, `nc_outcome`, `nc_responded`,
  `press_verdict`, each a raw-text-only twin of its V3 arm at the SAME document budget
  (`max_length = 1024 − block_tokens`: 255 / 249 / 260 / 219 respectively), same rows, same
  splits, same frozen recipe, no `--class_weight_auto`. The only difference from V3 is the
  absence of the criterion block and the matching budget.
  **This is a genuine test of a pre-registered prediction, not a confirmation exercise.**
  The displacement story says V3's negative deltas on nc_agree/nc_outcome are caused by the
  block eating ~a third of the document; if so, V3 should BEAT raw at matched budget. If it
  does not, the displacement explanation is wrong and the negative V3−bank deltas on the
  N&C cells are a real property of those cells — which would need saying plainly and would
  weaken the "V3 imports the bank's signal" reading in the V3 box above. The agent has been
  told a refutation is as valuable as a confirmation.
  Headline readout per cell: **V3 − raw_matched** (group bootstrap, 2,000 draws), with
  `raw_matched − T_original` alongside to price the lost text on its own.
- **2026-08-08 (grid close)** — final two main-grid V3 arms landed. `nc_responded` **.7950**
  (V3−T −.0217 [−.0428,−.0000] P=.025 — a significant loss to its dense parent; V3−VA_nl
  +.0049 P=.61 — a tie with its bank). `jokes_community_k10` **.7435**, giving the second
  dedicated k probe: k10 − k20 = +.0024 [−.0016,+.0068] P=.89. **V3 column now 13/14**
  (only `hashtagwars_verdict_k10`'s companion T/bank vectors are missing, which is a
  readout gap not a training gap).
  Note what nc_responded does to the N&C picture: all three N&C cells are truncation-hit
  (~.29-.31 with the block) and their V3−VA_nl deltas are −.0205, −.0234, +.0049 — two
  clear negatives and a tie, against 7 positives everywhere else. That is now the sharpest
  version of the displacement prediction the matched-raw control is testing.
- **2026-08-08 — MY DISPLACEMENT PREDICTION IS REFUTED** (nc_agree, nc_outcome; see the box).
  At matched document budget V3 is still below raw (−.0143, −.0068, both n.s.), and the
  text the block displaced turns out to carry no signal at all (`raw_matched − T` = +.0037
  / +.0026, P≈.6). I had built the truncation confound into the V3 section as the leading
  explanation for the two negative N&C deltas; it is wrong. Recording it at the same
  prominence as the prediction, and retiring the truncation caveat for these two cells —
  their V3 ≤ T results are interpretable at face value. `nc_responded` and `press_verdict`
  controls still running; press is the interesting remaining case because its raw
  truncation was already .31 before the block.
- **2026-08-08 — MATCHED-RAW CONTROL COMPLETE (4/4), and it splits my prediction in two.**
  REFUTED on N&C (all three: displaced text carried nothing, `raw_matched − T` = +.0037 /
  +.0026 / −.0015, and V3 stays below raw at matched budget — on `nc_responded`
  significantly so, −.0202 P=.01, the only interval in the set excluding zero). CONFIRMED on
  `press_verdict`, the one cell already truncating before the block: its lost text cost
  −.0218 (P=.03) and budget-matching reveals a +.0188 block contribution. Net: displacement
  is a long-document phenomenon; I generalised it from a truncation RATE without checking
  whether the displaced text carried signal. `raw_matched − T` is the right diagnostic and
  is now written into the note as the test any future cell must pass.
- **Operational notes from that run, worth keeping:** (a) `nc_responded`'s raw-matched job
  hit a CUDA OOM *after* "Training complete", inside the optional post-hoc test evaluation,
  when two co-tenants (49 GB + 98.5 GB) filled GPU1 — both epochs, all 10 validation
  checkpoints and both model dirs were intact and the epoch-2 selection matched the V3 arm,
  so the scoring pass was re-run separately rather than retraining, and no co-tenant was
  touched; (b) `score_eval_dense_v4.py` gained a **default-preserving**
  `DENSE_SCORE_MAXLEN` env knob (needed because scoring at 1024 would not match training at
  the reduced budget) and the raw-matched trainer is a **new sibling script**, not an edit
  to the live `train_v3_cell.sh` — bash re-reads a script while executing it, and that file
  was in use by other jobs.
