# Independent full-bank norm-to-metric labeling

Label every supplied extracted human statement against the supplied frozen task bank.
The statement may praise a criterion, report its violation, request a correction,
identify an omission, compare items on it, or state a verdict. A concrete breach or
directive can reveal a norm even when the abstract metric name is absent. Polarity
does not change metric identity.

Use the `norm` as the human statement and `context` only to resolve its actual
referent and normative force. `aspect` is a weak extraction hint, never evidence.
Do not invent a criterion from subject matter alone.

Apply these decisions:

- `MATCH`: exactly one bank definition directly captures the explicit criterion and
  wins a contrast against every plausible sibling.
- `MATCH_FAMILY_ONLY`: the construct is clear, but two or more bank leaves remain
  genuinely indistinguishable from the supplied words.
- `NO_EXPLICIT_CRITERION`: factual or topical text without evaluative, prescriptive,
  omission, comparison, or violation force.
- `CONTEXT_NEEDED`: potentially evaluative, but even the supplied context cannot
  resolve the criterion.
- `GENERIC_VERDICT`: undifferentiated praise/blame without a discriminating quality
  dimension.
- `NO_CANDIDATE_FITS`: a clear specific criterion is expressed but no bank metric
  directly generalizes it.
- `NOISE`: garbled extraction debris or meaningless language.

For `MATCH`, return the exact `metric_id`. Every abstention must use `metric_id: null`.
Use `high` confidence only when the human words and bank definitions make the exact
leaf contrast decisive. Use `medium` when the decision is best supported but one
real boundary remains; use `low` for fragile interpretations. A typed abstention is
a correct positive outcome. Never force an exact ID to increase yield.

The reason must be brief but contrastive: state what human criterion is explicit and
why the selected leaf beats its nearest sibling, or why the precise abstention type
applies. Process every item exactly once and preserve its `norm_uid` verbatim.
