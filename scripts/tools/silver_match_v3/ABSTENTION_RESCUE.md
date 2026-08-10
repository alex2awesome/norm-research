# Repeated full-bank abstention rescue

Run this only after a task's primary original+hashed adjudication and strict
two-order verifier have produced immutable provisional decisions.

1. Extract every non-`MATCH` UID with `export_rescue_uids.py`. For large tasks,
   use its immutable, hash-linked output with `retrieve.py --uid-file`; do not
   materialize full-bank rankings for already accepted rows unnecessarily.
2. Rank the complete frozen bank for those UIDs with at least the selected
   retriever, pretrained Nemotron, and BGE when available. Audit each output at
   `expected_k == bank_count`.
3. Build rescue with `--coverage-repeats 2 --reinclude-primary --block-size
   50`. Every metric, including the original K50, must appear exactly twice in
   shifted encoder/lane partitions. This controls both retrieval omission and
   a primary adjudicator that overlooked a visible metric.
4. Use the task's frozen K50 adjudicator as a high-recall proposal discoverer
   over each block. Aggregate only after exact trial UID/order/bank/exposure
   audits pass. Capture–recapture estimates describe proposal-discovery
   diversity only; they are not false-abstention probabilities.
5. Rejudge all discovered finalist IDs with original+hashed adjudication and
   the task's supported strict two-order contrastive verifier. A rescued
   `MATCH` requires both verifier orders to confirm the identical ID with high
   confidence.
6. For rows with no proposal, run the independent typed-abstention verifier in
   original+hashed trial-summary orders. Automatic abstention requires two
   identical high-confidence types. Any possible exact match, disagreement,
   low confidence, or parse failure remains unresolved for a blind independent
   label. `merge_rescue_decisions.py --unresolved-output` preserves every such
   row and reason before failing; `prepare_unresolved_decision_pack.py` then
   creates full-bank blind chunks with system outcomes hidden in a separate key.
7. Merge fail-closed, run exact task/corpus coverage and full decision-rate
   audits, then draw uniform blind samples of accepted MATCHes and abstentions.
   Report `<5%` false-abstention risk only if the exact one-sided 95% binomial
   upper bound is below `.05`.

Exact current-bank metric IDs remain primary. Family/equivalence relations are
reported only as a separately versioned sensitivity analysis and cannot turn a
failed leaf into a rescued exact match.
