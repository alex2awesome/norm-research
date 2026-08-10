# Silver matcher calibration labeling protocol

Label the criterion textually revealed by the human feedback, not a criterion
merely associated with its topic. A concrete directive, requested correction,
reported omission, or violation explicitly reveals its norm even when the
human does not say the abstract metric name. Map it when a bank definition
directly generalizes that directive or violation. Use the extracted `norm`
together with `context`; `aspect` is only an extractor hint and is not evidence.

Search the full frozen task bank before declaring a bank gap. Candidate ranks
are a convenience and a retrieval diagnostic, not a restriction on the human
label. Polarity never changes the metric identity.

Emit one JSON object per item with `norm_uid`, `decision`, `metric_id`,
`confidence`, and a brief contrastive `reason`.

Decisions:

- `MATCH`: exactly one frozen metric directly captures the explicit criterion.
- `MATCH_FAMILY_ONLY`: a construct family is explicit but sibling leaves cannot
  be distinguished from the human words.
- `NO_EXPLICIT_CRITERION`: genuinely topical, descriptive, or factual text
  without an evaluative attribute, prescription, omission, or violation.
- `CONTEXT_NEEDED`: potentially evaluative, but the supplied human text is too
  deictic or fragmentary to identify a criterion.
- `GENERIC_VERDICT`: only an undifferentiated verdict is explicit.
- `NO_CANDIDATE_FITS`: a clear, specific criterion is explicit but absent from
  the complete frozen bank. This is the bank-gap label.
- `NOISE`: garbled extraction debris or meaningless language.

Only `MATCH` carries a `metric_id`; every abstention uses JSON `null`.
Confidence is `high`, `medium`, or `low`. Do not force a match.
