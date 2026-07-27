# Register instrument audit + comparison lenses beyond Latinate

*2026-07-21. Follows the Codex audit of the prereg tests (ledger 2026-07-21b/c). Scripts:
`methods/codability/lexicon/latinate_detector.py`, corrected `prereg_tests.py`.*

## 1. Codex fixes — status

All seven actions implemented: original JSON frozen + SHA-256 provenance sidecar
(`prereg_results_20260721_ORIGINAL_frozen.json`, `_PROVENANCE.json`, git 0c6d6e08); script
now versions output (never overwrites); valid-permutation denominators; same-document pairs
excluded; mirror-collapse after doc filtering; unique-doc inst_share; eligible-doc
permutation universe. **v2 deviation-corrected execution**: PREREG-1 primary Fisher p=.272
(was .253), PREREG-2 primary p=.574 (Codex predicted ≈.576). Sensitivity splits .082/.032 —
still non-primary, still sign-heterogeneous. Verdict unchanged: *neither preregistered
hypothesis supported by the executed primary census-grain analyses*. Registry wording,
grain clarification, and the ledger count correction are recorded in ledger 2026-07-21c.

## 2. Latinate detector iterations (deterministic cross-instrument)

| version | design | 4-way agreement w/ Sonnet (n=1,500) |
|---|---|---|
| v1 | morphology only (suffix/prefix) | .405–.433 |
| v2 | etymology-db lookup + suffix-strip | .669 |
| v2.1 | − Middle English votes, − cognate relations (fixes voice/evidence/imagery) | .655 4-way |
| v2.2 | + curated homonym overrides, majority term rule | **.681 3-way** (classical/germanic/mixed) |

Manual inspection drove each iteration; the residual disagreement concentrates on the
mixed-boundary where *neither* instrument is ground truth (arbiter rule). Instrument
statement for the paper: LLM-judged stratum and a Wiktionary-etymology deterministic
detector agree ≈.66–.68; conclusions should be robust to either.

## 3. Selection-bias audit (the user's "is the LLM rewriting?" question)

Rewriting is mechanically excluded (verbatim-in-source validation, `extract.py:77-109`).
The remaining channel was SELECTION among the author's own words — tested: latinate score
of `head_term` vs same-record `key_terms` (both verbatim). **Heads are +.047–.068 more
Latinate in all 4 fields (Wilcoxon p < 1e-6 each; pooled +.056).** Two live readings:
(a) extractor selection bias toward formal labels; (b) genuine register structure — *naming
is a nominal act*: labels are nominalizations, running evaluative text is verby/Germanic.
Disambiguation experiment (queued): re-extract a subsample with an "ALL in-source candidate
names" prompt; if the chosen head is systematically the most Latinate *of the candidate
name set*, it's (a); if candidate names as a class sit above key_terms, it's (b).
Until then, usage-weighted register statistics carry this as a caveat.

## 4. Manual concept audit (8 sampled multi-record concepts, humor + math)

Extraction faithfulness: clean — names verbatim, quotes genuine, exact-duplicate mirrors
present but handled by the mirror guard. Partition quality: 6/8 coherent (incl. a textbook
competing-codes exhibit: "step on laughs" / "talk over laughs" / "the post-punchline
interdiction" — two Germanic verb phrases vs one Latinate nominalization for the same
norm); 2/8 with a questionable member ("communications" [article genre] merged into
goal-stating; negation-template vs proof-by-contradiction boundary). Consistent with
Codex's note that the unsuffixed census partition is a post-L0v2 intermediate —
**future lexicon analyses should run on canonical L0v4 and R1 grains.**

## 5. How our register approach compares with the literature

- **Formality lexicons/classifiers** (Brooke & Hirst; Pavlick & Tetreault): sentence- or
  word-level formality scores learned from corpora. Ours differs: term-level scoring of a
  *naming inventory*, with an LLM judge + anchor gates instead of a trained lexicon; their
  approach is a candidate convergent-validity instrument, not a substitute.
- **Heylighen & Dewaele F-score**: POS-ratio formality (nouny = formal). Our nominalization
  flag is the term-level cousin; a full F-score needs POS tagging (queued).
- **Etymology as a formality FEATURE** (simplification/formality-transfer work uses
  Latinate ratio as one signal): we elevate it to a measured axis with a dedicated
  dual-instrument design; no prior work applies it to evaluation-criterion names.
- **Lexical simplification pairs** (SimplePPDB, complex→simple): supervised same-meaning
  register pairs — our anchor/validation resource, and the closest structural analogue to
  "competing codes for one concept," but their pairs are generic vocabulary, not
  community evaluative terms.
- **Psycholinguistic norms** (Kuperman AoA, SUBTLEX frequency, concreteness): standard
  height proxies; convergent-validity correlation queued (crr.ugent.be mirrors stale
  2026-07-21; fetch manually or via alternate mirror).
- Nothing in the community-norms literature (W7 recon) scores the register of rule/value
  vocabulary at all — that axis is ours.

## 6. Lens menu beyond Latinate (comparing codes for the same concept / metrics per field)

**Computable now from our own data:**
1. *Cross-field dispersion* — in how many of the 11 fields does a code appear (general
   evaluative vocabulary vs field jargon); joins richness data directly.
2. *Termhood/technicality* — frequency ratio of the code in-field vs across-field corpora
   (C-value-style terminology measures).
3. *Nominalization density & morphological depth* — have judged flags + suffix parser.
4. *Verb/gerund vs noun construal* — process ("pacing") vs property ("pace") coding.

**Needs norm datasets (cheap once mirrors found):** 5. Age-of-acquisition; 6. frequency/
rarity; 7. concreteness/abstractness of the code.

**Needs new judging batches (Sonnet lane):** 8. *Metaphoricity* ("landing", "punch",
"flow" vs literal codes); 9. *Semantic transparency* (compositional vs idiomatic:
"excessive setup" vs "kill your darlings"); 10. *Thick vs thin evaluative concepts*
(Williams: "clear/good" thin vs "hacky/derivative" thick) — arguably the most
theoretically apt axis for evaluation criteria; 11. artifact-referencing vs
audience-response-referencing codes (E4-adjacent: requires its own prereg + blind coding).

## 7. Proposed next confirmatory round ("try again, harder") — AWAITING SIGN-OFF TO FREEZE

- **PREREG-4**: within>cross naming coincidence at **R1 construct grain** on the **7
  widened fields** (data untouched: extraction still running; patents/legal/n&c have fat
  institutional cells unlike humor). Corrected v2 code; declared primary BEFORE any
  contact with the new data. Motivated-but-not-contaminated: census-grain exploratory R1
  trend (p≈.07, Codex read-only audit) is disclosed as motivation.
- **PREREG-5 (new lens)**: *adoption asymmetry* — for concepts used by both classes, do
  informal sources use the institutional dominant code more often than institutional
  sources use the informal dominant code (prestige borrowing directionality)? Sharper than
  symmetric coincidence; same permutation machinery.
- PREREG-3 (LLM mode collapse) still queued — needs the Sonnet lane back for a second
  model family.
