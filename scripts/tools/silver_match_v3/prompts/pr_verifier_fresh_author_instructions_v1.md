# Press-release verifier prompt-author contract

You are authoring exactly one precision-first verifier prompt for a frozen
press-release norm-to-metric pipeline.  You may learn only from the inline,
identity-stripped optimize examples supplied with this request.  Do not use
tools, inspect files, or infer any verifier-dev, select, test, blind-audit, MI,
or downstream-outcome information.

The verifier receives a human statement, its evidence passage, one proposed
metric card, and several strongest alternative metric cards.  Its purpose is
to prevent a plausible but wrong proposal from becoming a silver MATCH.
Related topic, shared words, and a fact that merely appears in communicative
text are not sufficient.

The authored prompt must enforce this order:

1. Identify the explicit human-evaluated or breach-revealed communication
   criterion and its evidence span.  Ordinary facts, predictions, outcomes,
   requests, quotations, and surrounding subject matter are not criteria by
   themselves.
2. State the proposal's unique operational nucleus.  Confirm only when that
   nucleus is explicitly entailed by the human evidence.
3. Contrast the proposal with the strongest supplied alternative.  A broader
   neighboring leaf loses to a more specific supplied leaf.  If a better leaf
   is not supplied, abstain rather than confirming a nearby proposal.
4. Prefer precision over yield.  Confidence and fluent rationales cannot
   compensate for a missing criterion or unresolved exact-leaf boundary.

Preserve the verifier's complete decision vocabulary and JSON contract:

- `CONFIRM_MATCH`: the proposal is the uniquely best exact leaf; `metric_id`
  must equal the proposal ID.
- `BETTER_CANDIDATE`: one supplied alternative is clearly better;
  `metric_id` must be that alternative ID.
- `AMBIGUOUS_MATCH`: multiple supplied leaves remain genuinely co-entailed or
  the exact boundary cannot be resolved; `metric_id` is null.
- `NO_EXPLICIT_CRITERION`: the text does not explicitly evaluate or reveal a
  communication norm; `metric_id` is null.
- `CONTEXT_NEEDED`: the phrase is evaluative but the supplied passage cannot
  resolve what is being evaluated; `metric_id` is null.
- `GENERIC_VERDICT`: only broad quality is expressed, without a bank-leaf
  property; `metric_id` is null.
- `NO_CANDIDATE_FITS`: a specific criterion exists but neither proposal nor
  supplied alternatives express it; `metric_id` is null and the row must be
  routed to full-bank rescue.
- `NOISE`: the extraction is garbled or is not a coherent human evaluative
  signal; `metric_id` is null.

The final prompt must request one JSON object with exactly `decision`,
`metric_id`, `confidence`, and `reason`.  `confidence` must be exactly one of
the production parser strings `high`, `medium`, or `low` (never a number).
Set and state a 24-word maximum for `reason`.  Reasons must quote or
pinpoint the criterion evidence, and contrast the winning leaf with the
strongest confounder.  Do not memorize example wording, identities, or UIDs;
derive reusable rules.
