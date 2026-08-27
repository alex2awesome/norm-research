# math-stackexchange metric inventory (v1, 2026-06-10)

Distilled from the 122 parsed expert essays in `online-rubrics/claude-parsed/`
(Halmos, Gowers, Hardy, Knuth, Poonen, Lee, Lamport, Aigner–Ziegler, Avigad,
Hamkins, Andersen, Inglis–Aberdein, …). Format mirrors
`datasets/math/mathlib/METRICS.md`. Each metric: V (code-checkable on the
answer text — checker listed) or A (LLM-judge — judge input + source essay
listed). `gen` = generic tag per the bank-level scorecard in
`methods/metric_implementer/README.md`: G if the criterion applies verbatim to
≥3 of our 9 tasks, S if math-specific.

Label context (README §Dataset lineage): y = community judgment of answers
(accepted ∧ score≥3 vs score≤0) on **v3.3 propensity-balanced** (99,722 rows
50/50; question-only TF-IDF+LR floor 0.461). Working dimensions: elegance /
profundity / clarity / precision (Inglis–Aberdein four-factor result —
score each axis separately; beauty ≠ simplicity empirically).

## V metrics (mechanical, computed on answer text; sympy block on Q+A)

| id | metric | checker | gen |
|---|---|---|---|
| v01 | LaTeX well-formedness (balanced `$`/`$$`, parses in KaTeX/sympy-latex) | math-mode tokenizer + parse attempt | S |
| v02 | display vs inline discipline (unwieldy formula run inline; Knuth #5, Lee #17, Higham) | inline-math length scan (> ~1/3 line ⇒ should display) | S |
| v03 | sentence starts with a symbol (Halmos #6, Knuth #2, Hammack #4) | sentence-split + first-token math check | S |
| v04 | two formulas abutted with no words between (Halmos #6, Knuth #1/#4, Lee #22) | regex: `$…$` `[,;]?` `$…$` adjacency | S |
| v05 | symbol reused for two meanings / notation inconsistency (Halmos #9–10, Knuth #3, Higham cmd 6) — approximable | track binding sites ("let x…", "where x…"); flag re-binding to a different type | S |
| v06 | goal stated before argument ("we show / it suffices / claim:" early; Halmos #16, Knuth #25, Hammack #1, CS103 #1) | regex on first ~2 sentences | G |
| v07 | structured steps (numbered list / case markers / explicit QED; Lamport, Hammack #2) | list-marker + `\square`/QED detection | G |
| v08 | bare logical symbols in prose (∀ ∃ ⇒ ⇔ ∴ s.t. wlog iff; CS103 #7, Poonen #30–31, Lee #24) | grep lexicon | S |
| v09 | final-answer presence (`\boxed`, "therefore the answer is", concluding equation) | regex | G |
| v10 | citation presence (Wikipedia/OEIS/paper/textbook references; Higham cmd 8) | URL + reference-lexicon count | G |
| v11 | hedging density ("maybe", "I think", "not sure", "I guess") | lexicon count / length | G |
| v12 | direct-answer opening ("Yes"/"No"/"Note that"/"Hint:" as first token) | regex on opening | G |
| v13 | "clearly / obviously / trivially" count (Halmos #18, Poonen #26) — mechanical proxy of a04 | grep | G |
| v14 | connective scaffolding density ("hence", "thus", "therefore", "since", "it follows"; Hammack #6) | lexicon count / length | G |
| v15 | `n_claims`, `frac_checkable` — CAS-extractable claim coverage | `verification/` pipeline (extract→sympy) | S |
| v16 | `n_verified` (symbolic + numeric splits) | `verification/` pipeline | S |
| v17 | `n_refuted`, `has_refuted_load_bearing` — high-precision algebra-error flag | `verification/` pipeline | S |
| v18 | step-chain derivation check (`n_steps_verified/refuted`, `frac_steps_verified`) — sympy-checked `=`/`≤`/`≥` chains in display+inline math | `mathse_lint.py` step_chain (lark LaTeX backend, fork-isolated) | S |
| v19 | literal arithmetic check (`n_arith`, `n_arith_wrong`) — pure-numeric equalities verified exactly | `mathse_lint.py` literal_arith | S |
| v20 | LaTeX well-formedness (`n_latex_errors`, `frac_blocks_with_errors`) — implements v01 | `mathse_lint.py` latex_parse (brace/`\left\right`/`\begin\end` balance + strict pylatexenc + odd-`$`) | S |
| v21 | symbol hygiene (`n_undefined_symbols`, `n_unused_definitions`) — approximates v05 | `mathse_lint.py` symbol_hygiene (binder-site regexes vs usage counts) | S |
| v22 | dangling references (`n_refs`, `n_dangling`) — `\eqref`/"(N)"/"the lemma above" with no referent | `mathse_lint.py` dangling_refs | G |
| v23 | typo density (`typos_per_100_words`) | `mathse_lint.py` typo_density (pyspellchecker + 273-word jargon allowlist) | G |
| v24 | theorem-name mentions + misspellings (`n_theorem_mentions`, `n_misspelled_theorem_mentions`) | `mathse_lint.py` theorem_names (1,303 Wikipedia names; Levenshtein 1–2) | S |
| v25 | near-duplicate of sibling answer (`max_jaccard_to_sibling`) | `mathse_lint.py` near_dup (5-gram char-shingle jaccard within question) | G |
| v26 | question/answer form contract (`contract_met`, `has_boxed_or_final_numeric`, `has_proof_markers`, `has_example_marker`) — implements v09 | `mathse_lint.py` form_contract (speech-act + answer-form regexes) | G |

## A metrics (LLM-judge; judge input is Q+A unless noted)

| id | metric | judge input | source | gen |
|---|---|---|---|---|
| a01 | motivation before machinery: explains *why* before *what* (spiral plan; logical ≠ pedagogical order) | answer | halmos_how_to_write_mathematics #3–4/#15; lee #5; krantz #5 | G |
| a02 | audience calibration: detail level matches the asker's evident level; right things left implicit, hard moves explicit | Q + A | halmos #2; lee #1; avigad_understanding_proofs (formal/informal gap); andersen_acceptable_gaps | G |
| a03 | says something at the right generality: neither over-abstract machinery nor a one-off trick when the pattern is what helps | Q + A | gowers_two_cultures_of_mathematics; bourbaki_dieudonne (generalize) vs aigner_ziegler (no machinery overkill) | S |
| a04 | no unjustified gaps: every "clearly/obviously" step is actually routine for the asker; non-trivial gaps signposted | answer | halmos #18; poonen #26; andersen_acceptable_gaps; stanford_cs103 #8 | G |
| a05 | words–symbols balance: English glue carries the argument; symbols only where precision needs them | answer | halmos #7; higham cmd 4; stanford_cs103 #6 ("mugga mugga" test) | S |
| a06 | proof idea visible: reader learns *why* it's true; key idea extractable in one sentence | answer | hamkins (criterion 3–5); gowers_two_cultures_and_depth; avigad_understanding_proofs #2; rota (beauty = enlightenment) | G |
| a07 | elegance markers: one key idea, minimal machinery for the depth, surprise/aha, compact | answer | hardy (economy, unexpectedness); aigner_ziegler (Book criteria); rota | S |
| a08 | precision / rigor: each step justified, quantifiers explicit, variables introduced before use, no hand-waving | answer | poonen #1/#7; aops_proof_writing_wiki; stanford_cs103 #1/#3; inglis_aberdein (precision factor) | G |
| a09 | pedagogical scaffolding: concrete example or special case before (or alongside) the abstract argument | answer | halmos #25; krantz #7; polya (specialization); su (hospitality on-ramp) | G |
| a10 | honesty about epistemic status: assumptions, conditions, and unproven steps flagged as such (≠ hedging everything) | answer | atiyah_response_to_jaffe_quinn (shared rubric #2); andersen (signposted gaps) | G |
| a11 | directness: actually answers the question asked; conclusion matches the asker's want-to-show | Q + A | selden_selden (claim/proved match); stanford_cs103 #1; halmos #5 ("say something") | G |
| a12 | reusable technique: the method, not just the result, transfers to neighboring problems ("look back" carried out) | Q + A | polya step 4; avigad_mathematical_method #5/#8; mejia_ramos_weber (what readers seek) | G |
| a13 | notation choice quality: mnemonic, conventional, consistent with the question's own notation | Q + A | halmos #17; krantz §3; trzeciak | S |
| a14 | profundity: the answer required an idea, not mechanical definition-unwinding (depth ≠ length) | Q + A | gowers_two_cultures_and_depth (depth criterion); hardy (depth); inglis_aberdein (aesthetics vs intricacy) | S |

## Notes

- **Scorecard priority (expected discriminative power).** The v3 audit
  (`v3_leakage_audit/REPORT.md` §2) found the top legitimate LR features were
  *hedging* (maybe/think/not sure, coefs −1.9 to −2.8), *citations*
  (oeis/wikipedia/paper, +1.7 to +2.2), and *directness* (yes/no/note
  that/hence, +1.8 to +2.4). So prioritize v10–v12 and a10–a11 first, then
  a04/a08 (gap/rigor) — these are the judge-side versions of the same signals.
  v15–v17 are precision tools for *correctness*, not label proxies (pilot:
  9/10 checkable judgement=0 answers verified cleanly).
- **Floor rule.** Every metric/judge AUC is reported against the
  question-only floor of the dataset version used (v3.3: 0.461; v3: 0.6496).
  Only the margin above the floor is evidence of answer-quality measurement.
  Also report with/without pre-2014 rows (year drift, audit §4).
- **v13 vs a04, v11 vs a10**: deliberate V/A pairs — same construct,
  mechanical proxy vs judged version. They feed scorecard item 8
  (code↔judge convergence) directly.
- **Source tensions** (keep axes separate, don't average): Gowers vs Bourbaki
  on generality (sharp explicit constants vs maximal structural abstraction —
  a03 must judge *fit to the question*, not abstraction per se); Rota vs
  Hardy on surprise (Rota: unexpectedness neither necessary nor sufficient —
  so a07 weights "minimal machinery + extractable idea" over surprise);
  Lamport vs Halmos/Lee on form (numbered structure exposes errors vs
  paragraph prose communicates — v07 is a feature, not a virtue, on Math.SE);
  Polya vs polished style (showing how the solution was *found* conflicts
  with Halmos-style economy — covered by a09, not penalized by a07).
- **Generic fraction**: 17/31 G (55%; 8/17 V, 9/14 A) — feeds the bank-level
  generic-vs-task-specific scorecard.
- **v18–v26 results** (full run on v3.3, 2026-06-11; see
  `mathse_lint_REPORT.md` for AUC/coverage tables and the precision audit):
  combined LR over all lint features = 0.567 eval / 0.569 test AUC (floor
  0.461; claim-V v15–v17 = 0.541); lint+claim on the 9.8K joined subset =
  0.579 test. Highest-precision error flag: v24 misspellings (~55%, fixable
  to ~90% with an alias list); v21 is ~90% FP and effectively noise as a
  standalone signal.
