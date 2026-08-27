# #HashtagWars V/A readout

Run date: 2026-07-28

## Scope and outcome

The full SemEval-2017 Task 6 release has 12,734 labeled tweets in 112 hashtag
contests. To respect the scoring budget, this run used a precommitted,
label-blind grouped subset: the first 40 hashtag names under
`SHA-256("hashtagwars-va-v1:" + hashtag)`, retaining **every** tweet in each
selected contest. The exact analysis sample is **n = 4,228: 397 positives and
3,831 negatives across 40 hashtags**. Here, `y=1` means the staff verdict was
label 1 (top ten) or label 2 (winner); label 0 is negative.

All train/test folds are grouped by hashtag. No contest is divided between
folds. The hashtag prompt is supplied as shared context when scoring an entry;
because it is constant within a contest, it cannot separate entries inside the
grouping unit.

### Important execution limitation

The requested nested **gpt-5.6-sol** judge could not be started from the managed
local sandbox (`codex exec` failed while initializing its in-process app-server,
and no OpenAI API credential was exposed to the process). The A scores below
were therefore materialized by the deterministic `codebook` executor in
`score_va.py`, authored from the final articulated decision rules. They are
**not an LLM-judge run and must not be represented as gpt-5.6-sol judgments**.
The numbers are a completed local codebook readout, not a substitute estimate
of what gpt-5.6-sol would score. This limitation is also machine-recorded in
`results.json`.

## Grouped-CV AUC

Five-fold `GroupKFold` logistic readouts use fixed `C=1`, median imputation with
missingness indicators, standardization fitted inside each training fold, and
out-of-fold probabilities. Values are pooled out-of-fold ROC AUCs.

| Features | Full sample AUC | Balanced-subsample AUC |
|---|---:|---:|
| V | 0.5550 | 0.5494 |
| A | 0.5445 | 0.5511 |
| V+A | 0.5647 | 0.5701 |

The balanced readout retains all 397 positives and samples up to an equal
number of negatives **within each hashtag**, using seed 20260728. It has
**n = 788: 397 positives and 391 negatives across 40 hashtags**. The slight
six-row imbalance occurs because one or more very small contests contain fewer
available negatives than positives. It uses a fresh hashtag-grouped five-fold
readout on that subsample.

These are descriptive rank statistics only. No threshold, accuracy, or
classification claim is reported.

## Anchor check

Every one of the eight scoring shards included three blinded rows: a seeded
staff winner, a seeded random unselected entry, and scrambled nonsense. The
accepted-shard aggregate ordering was:

| Anchor | Mean A score |
|---|---:|
| Staff winner | 0.7813 |
| Random unselected | 0.5625 |
| Scrambled nonsense | 0.5278 |

The required ordering `winner > random > scrambled` passed in all eight
accepted shards. Invalid attempts were rejected rather than silently kept.
After a first-pass fidelity repair that stopped treating alphabetic word salad
as coherent, each shard required six deterministic anchor draws/rescore
attempts; the sixth passed. All attempt-level values are retained under
`anchors.batches[].attempt_history` in `results.json`. Because the same seeded
anchor draw is repeated across shards, the eight passing means are repeated
checks, not eight independent anchor samples.

## Articulated bank and label-blind GEPA pass

The final A bank contains 30 criteria inspired by the existing standup comedy
bank. The proposer did not inspect verdicts. Fidelity optimization operated on
criterion semantics and application behavior, never on AUC:

1. Broad or compound criteria were split into one-line, text-observable tests.
2. Each description states distinct `1.0`, `0.5`, `0.0`, and `NA` conditions.
3. Performance-, venue-, and audience-reaction claims not observable in a tweet
   were removed or rewritten as textual evidence.
4. Closely related devices were separated (for example, source
   recoverability, meaning-bearing transformation, sound play, and
   portmanteau formation).
5. The failed scrambled anchor exposed a fidelity error in coherence scoring;
   lexical plausibility was added without inspecting staff verdict outcomes.

Criteria were frozen before the verdict join. Verdicts were loaded only to
construct the mandatory role-defined anchors and to run the final readout.

## Top 10 univariate articulated criteria

These are raw-direction ROC AUCs of each criterion score over the full sample,
ranked descending. `NA` is replaced by that criterion's full-sample median for
this descriptive univariate calculation. AUC was never used to select or
rewrite the bank.

| Rank | Criterion | AUC | NA rate |
|---:|---|---:|---:|
| 1 | Economical wording | 0.5571 | 0.0000 |
| 2 | Self-contained comprehension | 0.5412 | 0.0000 |
| 3 | The joke is not explained after landing | 0.5348 | 0.0255 |
| 4 | Transformation changes the meaning | 0.5273 | 0.0648 |
| 5 | Analogy maps corresponding parts | 0.5244 | 0.2233 |
| 6 | Transformation is easy to recover | 0.5232 | 0.0648 |
| 7 | Parody preserves the source frame | 0.5220 | 0.0648 |
| 8 | Specific detail carries the joke | 0.5217 | 0.0702 |
| 9 | Recognizable source reference | 0.5197 | 0.0688 |
| 10 | Internal comic logic | 0.5165 | 0.0255 |

The complete 30-criterion ranking, fold membership, fold AUCs, selected hashtag
list, feature names, sample definition, and anchor histories are in
`results.json`.

## V bank

`v_features.py` computes 20 deterministic, label-blind features from the tweet
and contest prompt: content character/token length, average token length,
type-token ratio, uppercase ratio, all-caps count, question/exclamation/
ellipsis counts, hashtag/mention/URL/emoji/digit counts, repeated characters
and tokens, adjacent alliteration, nearby three-letter suffix rhyme, prompt
shared-substring ratio, and automated readability index. Submission plumbing
is removed only for content statistics; its counts remain separate features.

## Reproduction and caveats

Run:

```bash
python datasets/humor/hashtagwars/va/score_va.py all --force
```

Artifacts:

- `v_features.py`: 20 deterministic V checks.
- `rubrics.jsonl`: 30 final fidelity-rewritten articulated criteria.
- `score_va.py`: the runner actually used, including sampling, anchors,
  scoring, grouped CV, and balanced sampling.
- `scores_codebook.npz`: materialized token-valued A matrix and V matrix.
- `results.json`: full machine-readable output.

Additional caveats:

- This is a 4,228-row, 40-contest grouped subset, not all 12,734 tweets.
- The codebook backend is deterministic and reproducible but is not the
  requested gpt-5.6-sol semantic judge. Its AUCs should not be compared as if
  they were produced by that model.
- The 30 A dimensions are correlated and several are only conditionally
  applicable; median imputation plus indicators is a modeling choice.
- The balanced sample changes which negatives are compared and is not an
  uncertainty interval.
- No hyperparameter search, AUC-driven criterion selection, or threshold
  tuning was performed.
- Staff selection is an editorial verdict tied to each episode; the readout
  does not establish a general definition of humor.

> **SUPERSEDED (2026-07-29):** the A numbers below came from a deterministic codebook executor, not an LLM judge, and are not measurements under this project rules. The real LLM-judge run is in `RESULTS_gemma.md` / `results_gemma.json` in this directory (A .6350 vs .5445 here).
