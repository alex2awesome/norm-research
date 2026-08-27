# SO answer-votes (V6) — independent skeptical audit

Date: 2026-08-11. Agent: claude-so-votes-audit-fable (did NOT build the cell —
independence by assignment). Charge: user-ordered "is there something we might
be missing?" on the ARTICULABLE verdict (bank ≥ dense, same-rows Δ_beyond
−.0297 eval / −.0143 test; build note `notes/2026-08-08__v6_stackoverflow_build.md`).
Status: **COMPLETE (control arm landed; §7).**

FINAL VERDICT: **MIXED, mechanism named.** "ARTICULABLE" (Δ_beyond ≈ 0 at 8B
capacity, additive combination) STANDS on every reading. "Bank BEATS dense"
is an INSTRUMENT ARTIFACT with a two-part mechanism: (i) the headline dense
arm never saw the question body, worth +.014–.016 AUC where it fits; (ii) the
§6e control that appeared to rule this out was itself blinded by answer
displacement on 6.9% of rows (T→chance there). Under the fair
answer-preserving view Δ_beyond = −.0093 eval / +.0038 test (inside dense
seed spread). §11 fused check: fused beats bank +.018 both legs → no
escalation. The cell's grid row should quote the qtrunc-view T (.7277/.7231)
as the fair dense arm, with the title-view retained for math.SE comparability.

Terms unpacked: **Δ_beyond** = T − VA_nl on identical rows (negative = the
articulated bank matches/beats the dense model). **T** = dense-standard AUC
(Llama-3.1-8B LoRA reward model). **VA_nl** = nonlinear (GBM) aggregation of
the V (46 surface features) + A (39 Gemma-judged criteria) score matrix.
**title view** = dense text "QUESTION: title + ANSWER" (headline arm).
**abank view** = the A judge's context (title + tags + question body + answer),
run as the build's §6e sensitivity arm. **qtrunc view** (NEW, this audit) =
answer-preserving question-inclusive view: question body token-truncated so the
answer is never displaced from the 1024-token window.

## Verdict up front

**MIXED — the cell's scientific verdict STANDS, but its headline comparison is
partly an instrument artifact, now decomposed:**

1. **"ARTICULABLE" STANDS.** On every fair reading Δ_beyond ≈ 0 (bank ≈ dense
   at 8B capacity); nothing here rescues a tacit residual, and the Layer-3
   ineligibility conclusion is unchanged (if anything reinforced: fused adds
   only ~.018 over the bank, all of it attributable to the dense arm reading
   the same text with different errors).
2. **"Bank BEATS dense" does NOT stand as stated.** The margin is the sum of
   (a) a real context handicap on the title view — question context is worth
   **+.014–.016** to the dense arm when it fits the window — and (b) an
   answer-displacement artifact in the §6e control that was used to rule (a)
   out: on the 6.9% of rows where prepending the question pushed the ANSWER
   out of the 1024-token window, the context-dense arm collapses to chance
   (.48–.52) while the bank reads .75–.76. On the truncation-free 93% with
   full context, Δ_beyond = **−.0047 eval / +.0119 test** — a wash inside the
   dense seed spread (.0122/.0214).
3. **The §11 standing-rule trigger RESOLVES — no escalation.** Fused
   [bank + dense] beats bank on both legs (+.018), computed with
   alignment-gated joins (§5).

## 1. Item 1 — view asymmetry (the caption lesson): CONFIRMED IN REFINED FORM

Verified from the actual dense bundles (not the note):
- `dense_standard_so_votes/data.csv`: text = "QUESTION: {title}\n\nANSWER:\n
  {answer}" — the headline dense arm NEVER saw the question body. Verified
  ANSWER-marker on 100% of rows, TAGS on 0%.
- `dense_standard_so_votes_abankview/data.csv`: title + tags + question body +
  answer — matches the judge's context construction. Both views are Markdown-
  intact (code fences on 84%/97% of texts; `<p>` ≈ 0) — no strip_html damage.

The build DID run an item-view control (§6e, abank view, seed 42) and read
"+.0069 eval / −.0007 test ≈ noise". **That reading is an averaging artifact.**
Token-measured on the real tokenizer (Llama-3.1-8B, add_special_tokens):

| view | frac rows > 1024 tokens | note |
|---|---|---|
| title | 1.6% | truncation negligible on the headline arm |
| abank | **6.9%** | right-truncation cuts the ANSWER (it is last in the string); answer fully invisible on 0.5%, <50% visible on 2.6% |

Paired seed-42 readout, split by whether the abank text fits the window:

| leg | subset | n | T_title | T_abank | ctx gain | VA_nl | Δ_beyond (abank) |
|---|---|---|---|---|---|---|---|
| eval | fits (92.5%) | 1,128 | .7111 | **.7275** | **+.0163** | .7322 | −.0047 |
| eval | truncated | 92 | .6719 | **.5234** | −.1484 | .7500 | −.2266 |
| test | fits (94.5%) | 1,153 | .7141 | **.7278** | **+.0137** | .7159 | +.0119 |
| test | truncated | 67 | .7185 | **.4833** | −.2352 | .7620 | −.2787 |

Question context helps the dense model by +.014–.016 wherever it fits — a
consistent, both-legs effect — and the §6e whole-leg average hid it under the
7% of rows where the control arm was blinded to the answer it was scoring.
This is the press-verdict displacement confound (45% there, 7% here) landing
inside an item-view control. The eval-leg headline gap (−.0297) is roughly
half context handicap, half bank-hot-eval (VA_lin .7458 eval vs .7079 test);
the test-leg gap (−.0143) is fully covered by the context effect.

**Control arm launched per the charge** (§7): `qtrunc` view — abank text where
it fits; otherwise the QUESTION BODY is token-truncated so the full answer
always stays in window (the judge's own convention: it truncates the question,
never the answer). 11,363/12,202 rows byte-identical to abank; 632
question-truncated; 207 rows where title+answer alone exceed the window
(unavoidable, same for every view). Splits/labels byte-verified against the
existing bundles. Frozen recipe, seed 42, GPU 5 ledger-claimed, detached
PPID 1.

## 2. Item 2 — truncation + markup on the dense path: CLEAN (verified)

- Truncation is in TOKENS (max_length 1024) in both the trainer
  (`train_reward_model.py`, truncation=True right-side, padding_side left) and
  the scorer (`score_eval_dense_v4.py` MAXLEN env-default 1024; score.log shows
  no override). Trainer and scorer agree.
- No strip_html anywhere in the SO-votes path (`grep`: it appears only inside
  a warning comment in the population builder). Fences survive into the dense
  text (84%/97%).
- Title-view truncation touches only 1.6% of rows — the headline T is not
  materially truncation-limited; the problem was confined to the §6e control.

## 3. Item 3 — position proxying: RULED OUT as the bank's edge

Question-grouped OOF logistic stacks on all 12,202 rows (position = answer
order, the strongest trivial channel):

| model | pooled | within-question (pair-weighted) |
|---|---|---|
| position alone | .5932 | .6552 |
| bank (VA_nl) alone | .7012 | .7071 |
| bank + position | .7125 | **.7390** |

- **Bank increment over position, within-question: +.0838** — the bank is
  overwhelmingly NOT position-correlated text.
- Position increment over bank: +.0319 — position carries real signal the
  text-only bank cannot see (as expected; nothing text-side can read the
  clock). Anyone fielding this cell's stack should include position as a
  covariate — it is free and additive.

## 4. Item 4 — y-design selection: BENIGN, MECHANICAL

- Tie-droppage 23.7% (3,799/16,001), reproduced. It is a *mechanical* function
  of question size/parity, not a content filter: 2-answer questions tie 0%
  (sampled questions must carry a defined vote y), 3-answer questions tie
  53.5% (the middle answer IS the median), 5+ 27.3%. Year drift mild
  (.18–.25, no era cliff). The exclusion removes exactly the ambiguous middle
  — standard median-split behavior, sharpening not biasing.
- `y_accepted` never merged into `y_vote`: separate columns in
  `population.csv.gz`; φ = .552 reproduces exactly; the crosstab is the
  ledger's (4,959/852/1,951/4,440).

## 5. Item 5 — the pending §11 fused check: FUSED BEATS BANK, trigger resolves

Alignment discipline: the per-seed preds CSVs carry no ids → every positional
join was gated on row-by-row (judgement, group) equality against the split
manifests (8/8 gates pass; `results/so_votes_audit/alignment_gates.json`).
Published per-seed AUCs reproduce exactly (.7120/.7103/.6998 eval;
.7144/.7075/.6930 test).

Question-grouped OOF logistic stack on held-out rows (direction1-mirror
protocol), dense = seed-mean of the 3 title-view seeds:

| leg | bank | dense | FUSED | fused − bank |
|---|---|---|---|---|
| eval | .7357 | .7218 | **.7539** | **+.0182** |
| test | .7187 | .7175 | **.7369** | **+.0182** |
| both | .7275 | .7194 | **.7458** | +.0183 |

max(fused) > VA_nl on both legs → the §11 auto-audit condition is NOT met
after all; the build's trigger (logged, correctly not self-cleared) is hereby
resolved by the audit it asked for. Note the fusion gain (+.018) is the usual
two-readers-of-the-same-text effect, not evidence of a tacit residual: the
same-rows Δ_beyond stays ≈ 0.

## 6. Item 6 — surface-probe share: NEGLIGIBLE

The four declared Track-B probes are: `Contains a code block`, `Contains an
external hyperlink`, `Shows an output block`, `Uses a numbered or bulleted
list`. A_surface alone .5975 pooled / .6114 within-question; **dropping all
four from A costs .0015** (A_lin .6969 → A_real_lin .6954; within-q .7043 →
.7022). The bank's edge is carried by the real pole (diagnosis/correctness/
engagement families), not the surface probes.

## 7. Control arm (qtrunc) readout — COMPLETE, closes the item-1 loop

Seed-42, frozen recipe, same rows/splits (byte-verified), scored rc=0. All
conventions side by side (eval/test n=1,220 each):

| view | eval | test |
|---|---|---|
| T title (headline arm) | .7120 | .7144 |
| T abank (§6e control; answer-displaced on 7%) | .7189 | .7137 |
| **T qtrunc (fair: question in, answer never cut)** | **.7277** | **.7231** |
| VA_nl (same rows) | .7371 | .7192 |
| **Δ_beyond under the fair view** | **−.0093** | **+.0038** |

- The fair question-inclusive dense arm gains **+.0157 eval / +.0087 test**
  over the headline view (seed-matched, paired rows) — right in line with the
  fits-subset estimate (§1).
- **Δ_beyond collapses to −.0093 / +.0038** — sign-flips across legs, both
  inside the 3-seed dense spread (.0122/.0214). "Bank ≥ dense" does not
  survive the fair view; "bank ≈ dense at 8B capacity" does.
- On the 7% of rows the abank control had blinded, T recovers .52→.62 (eval)
  and .48→.65 (test) under qtrunc; the bank still leads there (.75–.76) —
  these are long-question items where any 1024-token view loses question
  content the judge's compressed context kept; a residual worth naming, not a
  verdict-changer at 7% mass.
- Caveat: qtrunc is one seed (42); its per-leg numbers carry the usual
  single-seed noise, but the +context direction is consistent across four
  independent readouts (fits-subset eval/test, qtrunc eval/test).

## 8. Artifacts

- Audit workspace: `methods/taste_decomposition/results/so_votes_audit/`
  (token_lengths_audit.csv, per-seed preds pulls, split manifests,
  alignment_gates.json).
- Control bundle: sk3
  `datasets/stackoverflow-votes/va/dense_standard_so_votes_qtrunc/`
  (data.csv, split/, manifest.json with build counts, chain.log,
  rm_out_seed42/).
- Reproductions: every ledger number touched (VA_nl same-rows eval/test,
  pooled instrument AUCs, per-seed dense AUCs, φ, tie rate, position
  baselines) reproduced exactly from
  `results/so_votes_oof_with_ids.npz` + stored preds before any new number
  was computed.
- GPU: GPU 5, ledger claim 2026-08-11 (claude-so-votes-audit-fable), release
  on completion inside the detached script. No other process touched.
